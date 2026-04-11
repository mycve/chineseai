use crate::nnue::HistoryMove;
use crate::xiangqi::{Color, Move, Position, RuleHistoryEntry, RuleOutcome};

use super::{
    AzEvalScratch, AzNnue, COMPLETED_Q_MAXVISIT_INIT, COMPLETED_Q_RESCALE_EPSILON,
    COMPLETED_Q_VALUE_SCALE, SCORE_CONSIDERED_LOW_LOGIT, VALUE_SCALE_CP, deterministic_gumbel,
    softmax, softmax_into,
};

#[derive(Clone, Copy, Debug)]
pub struct AzSearchLimits {
    pub simulations: usize,
    pub top_k: usize,
    pub seed: u64,
    pub gumbel_scale: f32,
    pub workers: usize,
}

impl Default for AzSearchLimits {
    fn default() -> Self {
        Self {
            simulations: 10_000,
            top_k: 32,
            seed: 0,
            gumbel_scale: 1.0,
            workers: 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AzCandidate {
    pub mv: Move,
    pub visits: u32,
    pub q: f32,
    pub prior: f32,
    pub policy: f32,
}

#[derive(Clone, Debug)]
pub struct AzSearchResult {
    pub best_move: Option<Move>,
    pub value_cp: i32,
    pub simulations: usize,
    pub candidates: Vec<AzCandidate>,
}

/// Gumbel 搜索与训练目标对齐 [DeepMind mctx](https://github.com/google-deepmind/mctx)
/// 中 `gumbel_muzero_policy` / `qtransform_completed_by_mix_value` 的默认行为（根 Gumbel+顺序减半、
/// 非根 `softmax(logits+q)` 确定性选择、根打分 `score_considered`）。根上 Gumbel 噪声为可复现哈希 Gumbel，
/// 与 JAX 真随机采样不同。
pub fn gumbel_search_with_history_and_rules(
    position: &Position,
    history: &[HistoryMove],
    rule_history: Option<Vec<RuleHistoryEntry>>,
    root_moves: Option<Vec<Move>>,
    model: &AzNnue,
    limits: AzSearchLimits,
) -> AzSearchResult {
    let mut tree = AzTree::new(
        position.clone(),
        truncate_history(history),
        rule_history,
        root_moves,
        model,
    );
    let root = tree.root;
    tree.expand(root);
    if tree.nodes[root].children.is_empty() {
        return AzSearchResult {
            best_move: None,
            value_cp: (tree.nodes[root].value * VALUE_SCALE_CP) as i32,
            simulations: 0,
            candidates: Vec::new(),
        };
    }

    let num_considered = limits.top_k.max(1).min(tree.nodes[root].children.len());
    let considered_visits = considered_visit_sequence(num_considered, limits.simulations);
    let mut used = 0usize;
    for considered_visit in considered_visits {
        let child_index =
            tree.select_root_child(root, limits.seed, limits.gumbel_scale, considered_visit);
        tree.simulate_child(root, child_index);
        used += 1;
    }

    let searched_value = if tree.nodes[root].visits > 0 {
        tree.nodes[root].value_sum / tree.nodes[root].visits as f32
    } else {
        tree.nodes[root].value
    };

    let policy = tree.improved_policy(root);
    let mut candidates = tree.nodes[root]
        .children
        .iter()
        .enumerate()
        .map(|(child_index, child)| AzCandidate {
            mv: child.mv,
            visits: child.visits,
            q: child.q(),
            prior: child.prior,
            policy: policy[child_index],
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        right
            .policy
            .total_cmp(&left.policy)
            .then_with(|| right.visits.cmp(&left.visits))
            .then_with(|| right.q.total_cmp(&left.q))
    });
    let best_move = tree
        .select_root_action(root, limits.seed, limits.gumbel_scale)
        .map(|child_index| tree.nodes[root].children[child_index].mv)
        .or_else(|| candidates.first().map(|candidate| candidate.mv));
    AzSearchResult {
        best_move,
        value_cp: (searched_value * VALUE_SCALE_CP) as i32,
        simulations: used,
        candidates,
    }
}

pub fn gumbel_search(
    position: &Position,
    model: &AzNnue,
    limits: AzSearchLimits,
) -> AzSearchResult {
    gumbel_search_with_history(position, &[], model, limits)
}

pub fn gumbel_search_with_history(
    position: &Position,
    history: &[HistoryMove],
    model: &AzNnue,
    limits: AzSearchLimits,
) -> AzSearchResult {
    gumbel_search_with_history_and_rules(position, history, None, None, model, limits)
}

pub fn gumbel_search_with_history_and_root_moves(
    position: &Position,
    history: &[HistoryMove],
    root_moves: Option<Vec<Move>>,
    model: &AzNnue,
    limits: AzSearchLimits,
) -> AzSearchResult {
    gumbel_search_with_history_and_rules(position, history, None, root_moves, model, limits)
}

struct AzTree<'a> {
    nodes: Vec<AzNode>,
    model: &'a AzNnue,
    root_rule_history: Option<Vec<RuleHistoryEntry>>,
    root_moves: Option<Vec<Move>>,
    root: usize,
    eval_scratch: AzEvalScratch,
}

struct AzNode {
    position: Position,
    history: Vec<HistoryMove>,
    rule_history: Option<Vec<RuleHistoryEntry>>,
    children: Vec<AzChild>,
    visits: u32,
    value_sum: f32,
    value: f32,
    expanded: bool,
}

struct AzChild {
    mv: Move,
    prior: f32,
    prior_logit: f32,
    visits: u32,
    value_sum: f32,
    child: Option<usize>,
}

impl AzChild {
    fn q(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }
}

impl<'a> AzTree<'a> {
    fn new(
        position: Position,
        history: Vec<HistoryMove>,
        rule_history: Option<Vec<RuleHistoryEntry>>,
        root_moves: Option<Vec<Move>>,
        model: &'a AzNnue,
    ) -> Self {
        let eval_scratch = AzEvalScratch::new(model.hidden_size);
        Self {
            nodes: vec![AzNode {
                position,
                history,
                rule_history: None,
                children: Vec::new(),
                visits: 0,
                value_sum: 0.0,
                value: 0.0,
                expanded: false,
            }],
            model,
            root_rule_history: rule_history,
            root_moves,
            root: 0,
            eval_scratch,
        }
    }

    fn effective_rule_history(&self, node_index: usize) -> Option<&Vec<RuleHistoryEntry>> {
        if node_index == self.root {
            self.root_rule_history.as_ref()
        } else {
            self.nodes[node_index].rule_history.as_ref()
        }
    }

    fn expand(&mut self, node_index: usize) -> f32 {
        if self.nodes[node_index].expanded {
            return self.nodes[node_index].value;
        }
        if let Some(value) = terminal_value(&self.nodes[node_index].position) {
            self.nodes[node_index].children.clear();
            self.nodes[node_index].value = value;
            self.nodes[node_index].expanded = true;
            return value;
        }
        if let Some(rule_history) = self.effective_rule_history(node_index) {
            if let Some(outcome) = self.nodes[node_index]
                .position
                .rule_outcome_with_history(rule_history)
            {
                self.nodes[node_index].children.clear();
                self.nodes[node_index].value = match outcome {
                    RuleOutcome::Draw(_) => 0.0,
                    RuleOutcome::Win(color) => {
                        if color == self.nodes[node_index].position.side_to_move() {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                };
                self.nodes[node_index].expanded = true;
                return self.nodes[node_index].value;
            }
        }
        let moves = if node_index == 0 {
            self.root_moves.clone().unwrap_or_else(|| {
                self.root_rule_history.as_ref().map_or_else(
                    || self.nodes[node_index].position.legal_moves(),
                    |history| {
                        self.nodes[node_index]
                            .position
                            .legal_moves_with_rules(history)
                    },
                )
            })
        } else if let Some(history) = self.nodes[node_index].rule_history.as_ref() {
            self.nodes[node_index]
                .position
                .legal_moves_with_rules(history)
        } else {
            self.nodes[node_index].position.legal_moves()
        };
        if moves.is_empty() {
            self.nodes[node_index].children.clear();
            self.nodes[node_index].value = -1.0;
            self.nodes[node_index].expanded = true;
            return -1.0;
        }
        let n = moves.len();
        let value = self.model.evaluate_with_scratch(
            &self.nodes[node_index].position,
            &self.nodes[node_index].history,
            &moves,
            &mut self.eval_scratch,
        );
        let children: Vec<AzChild> = {
            let scratch = &mut self.eval_scratch;
            softmax_into(&scratch.logits[..n], &mut scratch.priors);
            let logits = &scratch.logits[..n];
            let priors = &scratch.priors[..n];
            moves
                .into_iter()
                .enumerate()
                .map(|(i, mv)| AzChild {
                    mv,
                    prior: priors[i],
                    prior_logit: logits[i],
                    visits: 0,
                    value_sum: 0.0,
                    child: None,
                })
                .collect()
        };
        self.nodes[node_index].children = children;
        self.nodes[node_index].value = value;
        self.nodes[node_index].expanded = true;
        value
    }

    fn simulate(&mut self, node_index: usize) -> f32 {
        if !self.nodes[node_index].expanded {
            let value = self.expand(node_index);
            self.nodes[node_index].visits += 1;
            self.nodes[node_index].value_sum += value;
            return value;
        }
        if self.nodes[node_index].children.is_empty() {
            self.nodes[node_index].visits += 1;
            self.nodes[node_index].value_sum += self.nodes[node_index].value;
            return self.nodes[node_index].value;
        }
        let child_index = self.select_child(node_index);
        self.simulate_child(node_index, child_index)
    }

    fn simulate_child(&mut self, node_index: usize, child_index: usize) -> f32 {
        let child_node =
            if let Some(child_node) = self.nodes[node_index].children[child_index].child {
                child_node
            } else {
                let mv = self.nodes[node_index].children[child_index].mv;
                let mut child_position = self.nodes[node_index].position.clone();
                let mut child_history = self.nodes[node_index].history.clone();
                let mut child_rule_history = if node_index == self.root {
                    self.root_rule_history.clone()
                } else {
                    self.nodes[node_index].rule_history.clone()
                };
                append_history(&mut child_history, &child_position, mv);
                if let Some(rule_history) = child_rule_history.as_mut() {
                    rule_history.push(child_position.rule_history_entry_after_move(mv));
                }
                child_position.make_move(mv);
                let child_node = self.nodes.len();
                self.nodes.push(AzNode {
                    position: child_position,
                    history: child_history,
                    rule_history: child_rule_history,
                    children: Vec::new(),
                    visits: 0,
                    value_sum: 0.0,
                    value: 0.0,
                    expanded: false,
                });
                self.nodes[node_index].children[child_index].child = Some(child_node);
                child_node
            };
        let child_value = self.simulate(child_node);
        let value = -child_value;
        let child = &mut self.nodes[node_index].children[child_index];
        child.visits += 1;
        child.value_sum += value;
        self.nodes[node_index].visits += 1;
        self.nodes[node_index].value_sum += value;
        value
    }

    fn select_child(&self, node_index: usize) -> usize {
        let improved_policy = self.improved_policy(node_index);
        let total_visits = self.nodes[node_index]
            .children
            .iter()
            .map(|child| child.visits)
            .sum::<u32>() as f32;
        self.nodes[node_index]
            .children
            .iter()
            .enumerate()
            .max_by(|(left_index, left_child), (right_index, right_child)| {
                let left_score = improved_policy[*left_index]
                    - left_child.visits as f32 / (1.0 + total_visits);
                let right_score = improved_policy[*right_index]
                    - right_child.visits as f32 / (1.0 + total_visits);
                left_score
                    .total_cmp(&right_score)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .unwrap_or(0)
    }

    fn select_root_child(
        &self,
        node_index: usize,
        seed: u64,
        gumbel_scale: f32,
        considered_visit: u32,
    ) -> usize {
        self.best_scored_child(node_index, seed, gumbel_scale, |child| {
            child.visits == considered_visit
        })
        .unwrap_or_else(|| self.select_child(node_index))
    }

    fn select_root_action(&self, node_index: usize, seed: u64, gumbel_scale: f32) -> Option<usize> {
        let considered_visit = self.nodes[node_index]
            .children
            .iter()
            .map(|child| child.visits)
            .max()?;
        self.best_scored_child(node_index, seed, gumbel_scale, |child| {
            child.visits == considered_visit
        })
    }

    fn best_scored_child(
        &self,
        node_index: usize,
        seed: u64,
        gumbel_scale: f32,
        mut is_considered: impl FnMut(&AzChild) -> bool,
    ) -> Option<usize> {
        let completed_q = self.completed_qvalues(node_index);
        let hash = self.nodes[node_index].position.hash() ^ seed;
        let max_prior_logit = self.nodes[node_index]
            .children
            .iter()
            .map(|child| child.prior_logit)
            .fold(f32::NEG_INFINITY, f32::max);
        self.nodes[node_index]
            .children
            .iter()
            .enumerate()
            .filter(|(_, child)| is_considered(child))
            .map(|(child_index, child)| {
                let logits = child.prior_logit - max_prior_logit;
                let g = gumbel_scale * deterministic_gumbel(hash, child.mv, child_index as u64);
                let score = (g + logits + completed_q[child_index]).max(SCORE_CONSIDERED_LOW_LOGIT);
                (score, child_index)
            })
            .max_by(|left, right| {
                left.0
                    .total_cmp(&right.0)
                    .then_with(|| right.1.cmp(&left.1))
            })
            .map(|(_, child_index)| child_index)
    }

    fn improved_policy(&self, node_index: usize) -> Vec<f32> {
        let completed_q = self.completed_qvalues(node_index);
        let logits = self.nodes[node_index]
            .children
            .iter()
            .enumerate()
            .map(|(index, child)| child.prior_logit + completed_q[index])
            .collect::<Vec<_>>();
        softmax(&logits)
    }

    fn completed_qvalues(&self, node_index: usize) -> Vec<f32> {
        let children = &self.nodes[node_index].children;
        if children.is_empty() {
            return Vec::new();
        }

        let sum_visit_counts: u32 = children.iter().map(|child| child.visits).sum();
        let max_visit_counts = children
            .iter()
            .map(|child| child.visits)
            .max()
            .unwrap_or_default();

        let raw_value = self.nodes[node_index].value;
        let sum_probs: f32 = children
            .iter()
            .filter(|child| child.visits > 0)
            .map(|child| child.prior.max(f32::MIN_POSITIVE))
            .sum();
        let weighted_q: f32 = if sum_probs > 0.0 {
            children
                .iter()
                .filter(|child| child.visits > 0)
                .map(|child| {
                    let p = child.prior.max(f32::MIN_POSITIVE);
                    p * child.q() / sum_probs
                })
                .sum()
        } else {
            0.0
        };
        let mixed_value = (raw_value + sum_visit_counts as f32 * weighted_q)
            / (sum_visit_counts as f32 + 1.0);

        let mut completed: Vec<f32> = children
            .iter()
            .map(|child| {
                if child.visits > 0 {
                    child.q()
                } else {
                    mixed_value
                }
            })
            .collect();
        rescale_completed_qvalues_mctx(&mut completed);
        let visit_scale = COMPLETED_Q_MAXVISIT_INIT + max_visit_counts as f32;
        for v in &mut completed {
            *v *= visit_scale * COMPLETED_Q_VALUE_SCALE;
        }
        completed
    }
}

fn rescale_completed_qvalues_mctx(qvalues: &mut [f32]) {
    if qvalues.is_empty() {
        return;
    }
    let min_value = qvalues.iter().copied().fold(f32::INFINITY, f32::min);
    let max_value = qvalues.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let denom = (max_value - min_value).max(COMPLETED_Q_RESCALE_EPSILON);
    for v in qvalues.iter_mut() {
        *v = (*v - min_value) / denom;
    }
}

fn considered_visit_sequence(
    max_num_considered_actions: usize,
    num_simulations: usize,
) -> Vec<u32> {
    if max_num_considered_actions == 0 || num_simulations == 0 {
        return Vec::new();
    }
    if max_num_considered_actions <= 1 {
        return (0..num_simulations as u32).collect();
    }

    let log2max = (max_num_considered_actions as f32).log2().ceil() as usize;
    let mut sequence = Vec::with_capacity(num_simulations);
    let mut visits = vec![0u32; max_num_considered_actions];
    let mut num_considered = max_num_considered_actions;
    while sequence.len() < num_simulations {
        let extra_visits = (num_simulations / (log2max * num_considered)).max(1);
        for _ in 0..extra_visits {
            sequence.extend_from_slice(&visits[..num_considered]);
            for visit in &mut visits[..num_considered] {
                *visit += 1;
            }
            if sequence.len() >= num_simulations {
                break;
            }
        }
        num_considered = (num_considered / 2).max(2);
    }
    sequence.truncate(num_simulations);
    sequence
}

fn terminal_value(position: &Position) -> Option<f32> {
    if !position.has_general(Color::Red) {
        return Some(if position.side_to_move() == Color::Red {
            -1.0
        } else {
            1.0
        });
    }
    if !position.has_general(Color::Black) {
        return Some(if position.side_to_move() == Color::Black {
            -1.0
        } else {
            1.0
        });
    }
    None
}

pub(super) fn append_history(history: &mut Vec<HistoryMove>, position: &Position, mv: Move) {
    if let Some(piece) = position.piece_at(mv.from as usize) {
        history.push(HistoryMove { piece, mv });
        let overflow = history.len().saturating_sub(crate::nnue::HISTORY_PLIES);
        if overflow > 0 {
            history.drain(0..overflow);
        }
    }
}

fn truncate_history(history: &[HistoryMove]) -> Vec<HistoryMove> {
    history
        .iter()
        .rev()
        .take(crate::nnue::HISTORY_PLIES)
        .copied()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn considered_visit_sequence_matches_sequential_halving_schedule() {
        assert_eq!(
            considered_visit_sequence(4, 16),
            vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        );
        assert_eq!(considered_visit_sequence(16, 512).len(), 512);
        assert_eq!(considered_visit_sequence(1, 5), vec![0, 1, 2, 3, 4]);
        assert_eq!(considered_visit_sequence(0, 5), Vec::<u32>::new());
    }

    #[test]
    fn completed_q_rescale_gives_zero_when_all_q_equal() {
        let mut qvalues = vec![0.5, 0.5, 0.5];
        rescale_completed_qvalues_mctx(&mut qvalues);
        assert_eq!(qvalues, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn non_root_selection_uses_mctx_visit_penalty() {
        let model = AzNnue::random_with_depth(4, 1, 19);
        let mut tree = AzTree::new(Position::startpos(), Vec::new(), None, None, &model);
        tree.nodes[0].expanded = true;
        tree.nodes[0].value = 0.5;
        tree.nodes[0].children = vec![
            AzChild {
                mv: Move::new(0, 9),
                prior: 0.5,
                prior_logit: 0.0,
                visits: 5,
                value_sum: 2.5,
                child: None,
            },
            AzChild {
                mv: Move::new(1, 10),
                prior: 0.5,
                prior_logit: 0.0,
                visits: 0,
                value_sum: 0.0,
                child: None,
            },
        ];

        assert_eq!(tree.completed_qvalues(0), vec![0.0, 0.0]);
        assert_eq!(tree.improved_policy(0), vec![0.5, 0.5]);
        assert_eq!(tree.select_child(0), 1);
    }

    #[test]
    fn gumbel_search_visits_initial_considered_actions() {
        let model = AzNnue::random_with_depth(4, 1, 7);
        let result = gumbel_search(
            &Position::startpos(),
            &model,
            AzSearchLimits {
                simulations: 512,
                top_k: 16,
                seed: 11,
                gumbel_scale: 1.0,
                workers: 1,
            },
        );
        let visited_actions = result
            .candidates
            .iter()
            .filter(|candidate| candidate.visits > 0)
            .count();

        assert_eq!(result.simulations, 512);
        assert!(visited_actions >= 16);
    }
}
