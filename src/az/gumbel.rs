use crate::nnue::HistoryMove;
use crate::xiangqi::{Color, Move, Position, RuleHistoryEntry, RuleOutcome};

use super::{AzEvalOutput, AzEvalScratch, AzNnue, SplitMix64};

#[derive(Clone, Copy, Debug)]
pub struct AzSearchLimits {
    pub simulations: usize,
    pub seed: u64,
    /// Maximum search depth in plies below root. 0 keeps the default:
    /// max_depth = num_simulations.
    pub max_depth: usize,
    pub value_scale: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct GumbelSearchConfig {
    pub max_num_considered_actions: usize,
    pub gumbel_scale: f32,
    pub value_scale: f32,
    pub maxvisit_init: f32,
}

impl Default for GumbelSearchConfig {
    fn default() -> Self {
        Self {
            max_num_considered_actions: 16,
            gumbel_scale: 1.0,
            value_scale: 0.02,
            maxvisit_init: 50.0,
        }
    }
}

impl Default for AzSearchLimits {
    fn default() -> Self {
        Self {
            simulations: 10_000,
            seed: 0,
            max_depth: 0,
            value_scale: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AzCandidate {
    pub mv: Move,
    pub visits: u32,
    pub q: f32,
    pub value_wdl: [f32; 3],
    pub moves_left: f32,
    pub raw_prior: f32,
    pub prior: f32,
    pub policy: f32,
}

#[derive(Clone, Debug)]
pub struct AzSearchResult {
    pub best_move: Option<Move>,
    pub value_q: f32,
    pub value_cp: i32,
    /// Root win/draw/loss probabilities from the side-to-move perspective.
    pub value_wdl: [f32; 3],
    pub simulations: usize,
    pub search_depth_avg: f32,
    pub search_depth_max: usize,
    pub search_depth_limit: usize,
    pub search_depth_cutoffs: usize,
    pub candidates: Vec<AzCandidate>,
}

pub fn gumbel_search_with_history_and_rules(
    position: &Position,
    history: &[HistoryMove],
    rule_history: Option<Vec<RuleHistoryEntry>>,
    root_moves: Option<Vec<Move>>,
    model: &AzNnue,
    limits: AzSearchLimits,
    config: GumbelSearchConfig,
) -> AzSearchResult {
    crate::scope_profile!("az.gumbel_search");
    let mut tree = AzTree::new(
        position.clone(),
        truncate_history(history),
        rule_history.unwrap_or_else(|| position.initial_rule_history()),
        root_moves,
        model,
        limits,
        config,
    );
    let root = tree.root;
    tree.expand(root);
    tree.prepare_gumbel_root(limits.simulations);
    if tree.nodes[root].children.is_empty() {
        return AzSearchResult {
            best_move: None,
            value_q: tree.nodes[root].value,
            value_cp: cp_from_q(tree.nodes[root].value),
            value_wdl: tree.nodes[root].value_wdl,
            simulations: 0,
            search_depth_avg: 0.0,
            search_depth_max: 0,
            search_depth_limit: tree.max_depth,
            search_depth_cutoffs: 0,
            candidates: Vec::new(),
        };
    }

    let mut used = 0usize;
    for _ in 0..limits.simulations {
        tree.simulate(root, 0);
        used += 1;
    }

    let root_node = &tree.nodes[root];
    let searched_value = if root_node.visits > 0 {
        root_node.value_sum / root_node.visits as f32
    } else {
        root_node.value
    };
    let searched_wdl = if root_node.visits > 0 {
        root_node
            .value_wdl_sum
            .map(|value| value / root_node.visits as f32)
    } else {
        root_node.value_wdl
    };
    let policy = tree.root_policy(root);
    let mut candidates = root_node
        .children
        .iter()
        .zip(policy)
        .map(|(child, policy)| AzCandidate {
            mv: child.mv,
            visits: child.visits,
            q: child.q(),
            value_wdl: child.mean_wdl(),
            moves_left: child.moves_left(),
            raw_prior: child.raw_prior,
            prior: child.prior,
            policy,
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
        .best_root_child(root)
        .map(|child_index| tree.nodes[root].children[child_index].mv)
        .or_else(|| candidates.first().map(|candidate| candidate.mv));
    AzSearchResult {
        best_move,
        value_q: searched_value,
        value_cp: cp_from_q(searched_value),
        value_wdl: searched_wdl,
        simulations: used,
        search_depth_avg: tree.search_depth_avg(),
        search_depth_max: tree.search_depth_max,
        search_depth_limit: tree.max_depth,
        search_depth_cutoffs: tree.search_depth_cutoffs,
        candidates,
    }
}

pub fn cp_from_q(q: f32) -> i32 {
    (q.clamp(-1.0, 1.0) * 1000.0).round() as i32
}

struct AzTree<'a> {
    nodes: Vec<AzNode>,
    model: &'a AzNnue,
    root_moves: Option<Vec<Move>>,
    root: usize,
    root_noise_seed: u64,
    value_scale: f32,
    max_depth: usize,
    search_depth_sum: usize,
    search_depth_count: usize,
    search_depth_max: usize,
    search_depth_cutoffs: usize,
    eval_scratch: AzEvalScratch,
    gumbel: GumbelSearchConfig,
    root_gumbels: Vec<f32>,
    root_considered_visits: Vec<u32>,
}

struct AzNode {
    position: Position,
    history: Vec<HistoryMove>,
    rule_history: Vec<RuleHistoryEntry>,
    children: Vec<AzChild>,
    visits: u32,
    value_sum: f32,
    value_wdl_sum: [f32; 3],
    value: f32,
    value_wdl: [f32; 3],
    moves_left: f32,
    expanded: bool,
}

#[derive(Clone)]
struct AzChild {
    mv: Move,
    raw_prior: f32,
    prior: f32,
    visits: u32,
    value_sum: f32,
    value_wdl_sum: [f32; 3],
    moves_left_sum: f32,
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

    fn moves_left(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.moves_left_sum / self.visits as f32
        }
    }

    fn mean_wdl(&self) -> [f32; 3] {
        if self.visits == 0 {
            [0.0, 1.0, 0.0]
        } else {
            self.value_wdl_sum.map(|value| value / self.visits as f32)
        }
    }
}

impl<'a> AzTree<'a> {
    fn new(
        position: Position,
        history: Vec<HistoryMove>,
        rule_history: Vec<RuleHistoryEntry>,
        root_moves: Option<Vec<Move>>,
        model: &'a AzNnue,
        limits: AzSearchLimits,
        gumbel: GumbelSearchConfig,
    ) -> Self {
        let mut nodes = Vec::with_capacity(limits.simulations.saturating_add(1));
        nodes.push(AzNode {
            position,
            history,
            rule_history,
            children: Vec::new(),
            visits: 0,
            value_sum: 0.0,
            value_wdl_sum: [0.0; 3],
            value: 0.0,
            value_wdl: [0.0, 1.0, 0.0],
            moves_left: 0.0,
            expanded: false,
        });
        Self {
            nodes,
            model,
            root_moves,
            root: 0,
            root_noise_seed: limits.seed,
            value_scale: limits.value_scale.clamp(0.0, 1.0),
            max_depth: if limits.max_depth == 0 {
                limits.simulations
            } else {
                limits.max_depth
            },
            search_depth_sum: 0,
            search_depth_count: 0,
            search_depth_max: 0,
            search_depth_cutoffs: 0,
            eval_scratch: AzEvalScratch::new(model.arch),
            gumbel,
            root_gumbels: Vec::new(),
            root_considered_visits: Vec::new(),
        }
    }

    fn expand(&mut self, node_index: usize) -> AzEvalOutput {
        if self.nodes[node_index].expanded {
            return self.node_eval(node_index);
        }

        if let Some(value) = terminal_value(
            &self.nodes[node_index].position,
            &self.nodes[node_index].rule_history,
        ) {
            let value_wdl = scalar_terminal_wdl(value);
            self.nodes[node_index].children.clear();
            self.nodes[node_index].value = value;
            self.nodes[node_index].value_wdl = value_wdl;
            self.nodes[node_index].moves_left = 0.0;
            self.nodes[node_index].expanded = true;
            return AzEvalOutput {
                value_wdl,
                value,
                moves_left: 0.0,
            };
        }

        let moves = if node_index == self.root {
            self.root_moves.clone().unwrap_or_else(|| {
                self.nodes[node_index]
                    .position
                    .legal_moves_with_rules(&self.nodes[node_index].rule_history)
            })
        } else {
            self.nodes[node_index]
                .position
                .legal_moves_with_rules(&self.nodes[node_index].rule_history)
        };
        if moves.is_empty() {
            self.nodes[node_index].children.clear();
            self.nodes[node_index].value = -1.0;
            self.nodes[node_index].value_wdl = [0.0, 0.0, 1.0];
            self.nodes[node_index].moves_left = 0.0;
            self.nodes[node_index].expanded = true;
            return AzEvalOutput {
                value_wdl: [0.0, 0.0, 1.0],
                value: -1.0,
                moves_left: 0.0,
            };
        }

        let mut eval = self.model.evaluate_with_scratch_output(
            &self.nodes[node_index].position,
            &self.nodes[node_index].history,
            &moves,
            &mut self.eval_scratch,
        );
        eval.value_wdl = scale_wdl_value(eval.value_wdl, self.value_scale);
        eval.value *= self.value_scale;
        let priors = softmax_into(
            &self.eval_scratch.logits[..moves.len()],
            &mut self.eval_scratch.priors,
        );
        let raw_priors = priors.clone();
        self.nodes[node_index].children = moves
            .into_iter()
            .zip(priors.drain(..))
            .zip(raw_priors)
            .map(|((mv, prior), raw_prior)| AzChild {
                mv,
                raw_prior,
                prior,
                visits: 0,
                value_sum: 0.0,
                value_wdl_sum: [0.0; 3],
                moves_left_sum: 0.0,
                child: None,
            })
            .collect();
        self.nodes[node_index].value = eval.value;
        self.nodes[node_index].value_wdl = eval.value_wdl;
        self.nodes[node_index].moves_left = eval.moves_left;
        self.nodes[node_index].expanded = true;
        eval
    }

    fn simulate(&mut self, node_index: usize, depth: usize) -> AzEvalOutput {
        if depth >= self.max_depth {
            let eval = self.cutoff_value(node_index);
            self.add_node_visit(node_index, eval);
            self.record_leaf_depth(depth, true);
            return eval;
        }
        if !self.nodes[node_index].expanded {
            let eval = self.expand(node_index);
            self.add_node_visit(node_index, eval);
            self.record_leaf_depth(depth, false);
            return eval;
        }
        if self.nodes[node_index].children.is_empty() {
            let eval = self.node_eval(node_index);
            self.add_node_visit(node_index, eval);
            self.record_leaf_depth(depth, false);
            return eval;
        }
        let child_index = self.select_child(node_index);
        self.simulate_child(node_index, child_index, depth + 1)
    }

    fn simulate_child(
        &mut self,
        node_index: usize,
        child_index: usize,
        child_depth: usize,
    ) -> AzEvalOutput {
        let child_node =
            if let Some(child_node) = self.nodes[node_index].children[child_index].child {
                child_node
            } else {
                let mv = self.nodes[node_index].children[child_index].mv;
                let mut child_position = self.nodes[node_index].position.clone();
                let child_history = clone_history_with_appended_move(
                    &self.nodes[node_index].history,
                    &child_position,
                    mv,
                );
                let child_rule_entry = child_position.rule_history_entry_after_move(mv);
                child_position.make_move(mv);
                let child_rule_history = clone_rule_history_with_appended_entry(
                    &self.nodes[node_index].rule_history,
                    child_rule_entry,
                );
                let child_node = self.nodes.len();
                self.nodes.push(AzNode {
                    position: child_position,
                    history: child_history,
                    rule_history: child_rule_history,
                    children: Vec::new(),
                    visits: 0,
                    value_sum: 0.0,
                    value_wdl_sum: [0.0; 3],
                    value: 0.0,
                    value_wdl: [0.0, 1.0, 0.0],
                    moves_left: 0.0,
                    expanded: false,
                });
                self.nodes[node_index].children[child_index].child = Some(child_node);
                child_node
            };
        let child_eval = self.simulate(child_node, child_depth);
        let eval = AzEvalOutput {
            value_wdl: flip_wdl(child_eval.value_wdl),
            value: -child_eval.value,
            moves_left: child_eval.moves_left,
        };
        let child = &mut self.nodes[node_index].children[child_index];
        child.visits += 1;
        child.value_sum += eval.value;
        add_wdl(&mut child.value_wdl_sum, eval.value_wdl);
        child.moves_left_sum += eval.moves_left;
        self.add_node_visit(node_index, eval);
        eval
    }

    fn cutoff_value(&mut self, node_index: usize) -> AzEvalOutput {
        if self.nodes[node_index].expanded {
            return self.node_eval(node_index);
        }
        if let Some(value) = terminal_value(
            &self.nodes[node_index].position,
            &self.nodes[node_index].rule_history,
        ) {
            let value_wdl = scalar_terminal_wdl(value);
            self.nodes[node_index].value = value;
            self.nodes[node_index].value_wdl = value_wdl;
            self.nodes[node_index].moves_left = 0.0;
            return AzEvalOutput {
                value_wdl,
                value,
                moves_left: 0.0,
            };
        }
        let moves = self.nodes[node_index]
            .position
            .legal_moves_with_rules(&self.nodes[node_index].rule_history);
        if moves.is_empty() {
            self.nodes[node_index].value = -1.0;
            self.nodes[node_index].value_wdl = [0.0, 0.0, 1.0];
            self.nodes[node_index].moves_left = 0.0;
            return AzEvalOutput {
                value_wdl: [0.0, 0.0, 1.0],
                value: -1.0,
                moves_left: 0.0,
            };
        }
        let mut eval = self.model.evaluate_with_scratch_output(
            &self.nodes[node_index].position,
            &self.nodes[node_index].history,
            &moves,
            &mut self.eval_scratch,
        );
        eval.value_wdl = scale_wdl_value(eval.value_wdl, self.value_scale);
        eval.value *= self.value_scale;
        self.nodes[node_index].value = eval.value;
        self.nodes[node_index].value_wdl = eval.value_wdl;
        self.nodes[node_index].moves_left = eval.moves_left;
        eval
    }

    fn node_eval(&self, node_index: usize) -> AzEvalOutput {
        AzEvalOutput {
            value_wdl: self.nodes[node_index].value_wdl,
            value: self.nodes[node_index].value,
            moves_left: self.nodes[node_index].moves_left,
        }
    }

    fn add_node_visit(&mut self, node_index: usize, eval: AzEvalOutput) {
        self.nodes[node_index].visits += 1;
        self.nodes[node_index].value_sum += eval.value;
        add_wdl(&mut self.nodes[node_index].value_wdl_sum, eval.value_wdl);
    }

    fn record_leaf_depth(&mut self, depth: usize, cutoff: bool) {
        self.search_depth_sum += depth;
        self.search_depth_count += 1;
        self.search_depth_max = self.search_depth_max.max(depth);
        if cutoff {
            self.search_depth_cutoffs += 1;
        }
    }

    fn search_depth_avg(&self) -> f32 {
        if self.search_depth_count == 0 {
            0.0
        } else {
            self.search_depth_sum as f32 / self.search_depth_count as f32
        }
    }

    fn prepare_gumbel_root(&mut self, simulations: usize) {
        let config = self.gumbel;
        let action_count = self.nodes[self.root].children.len();
        let considered = config
            .max_num_considered_actions
            .max(1)
            .min(action_count.max(1));
        let mut rng = SplitMix64::new(self.root_noise_seed ^ 0xD1B5_4A32_D192_ED03);
        self.root_gumbels = (0..action_count)
            .map(|_| {
                let uniform = rng.unit_f32().clamp(1e-7, 1.0 - 1e-7);
                config.gumbel_scale.max(0.0) * -(-uniform.ln()).ln()
            })
            .collect();
        self.root_considered_visits = sequential_halving_visits(considered, simulations);
    }

    fn gumbel_completed_q(&self, node_index: usize) -> Vec<f32> {
        let node = &self.nodes[node_index];
        if node.children.is_empty() {
            return Vec::new();
        }
        let visited_prior = node
            .children
            .iter()
            .filter(|child| child.visits > 0)
            .map(|child| child.prior.max(f32::MIN_POSITIVE))
            .sum::<f32>();
        let weighted_q = if visited_prior > 0.0 {
            node.children
                .iter()
                .filter(|child| child.visits > 0)
                .map(|child| child.prior.max(f32::MIN_POSITIVE) * child.q())
                .sum::<f32>()
                / visited_prior
        } else {
            node.value
        };
        let total_visits = node.children.iter().map(|child| child.visits).sum::<u32>();
        let mixed_value =
            (node.value + total_visits as f32 * weighted_q) / (total_visits as f32 + 1.0);
        let mut completed = node
            .children
            .iter()
            .map(|child| {
                if child.visits > 0 {
                    child.q()
                } else {
                    mixed_value
                }
            })
            .collect::<Vec<_>>();
        let min_q = completed.iter().copied().fold(f32::INFINITY, f32::min);
        let max_q = completed.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_q - min_q).max(1e-8);
        let config = self.gumbel;
        let max_visits = node
            .children
            .iter()
            .map(|child| child.visits)
            .max()
            .unwrap_or(0) as f32;
        let scale = (config.maxvisit_init.max(0.0) + max_visits) * config.value_scale.max(0.0);
        for q in &mut completed {
            *q = (*q - min_q) / range * scale;
        }
        completed
    }

    fn gumbel_score(&self, child_index: usize, completed_q: &[f32]) -> f32 {
        self.root_gumbels.get(child_index).copied().unwrap_or(0.0)
            + self.nodes[self.root].children[child_index]
                .prior
                .max(1e-12)
                .ln()
            + completed_q[child_index]
    }

    fn select_gumbel_root_child(&self, node_index: usize) -> usize {
        let node = &self.nodes[node_index];
        let simulation = node
            .children
            .iter()
            .map(|child| child.visits as usize)
            .sum::<usize>();
        let considered_visit = self
            .root_considered_visits
            .get(simulation)
            .copied()
            .unwrap_or_else(|| {
                node.children
                    .iter()
                    .map(|child| child.visits)
                    .max()
                    .unwrap_or(0)
            });
        let completed = self.gumbel_completed_q(node_index);
        node.children
            .iter()
            .enumerate()
            .filter(|(_, child)| child.visits == considered_visit)
            .max_by(|(left, _), (right, _)| {
                self.gumbel_score(*left, &completed)
                    .total_cmp(&self.gumbel_score(*right, &completed))
            })
            .map(|(index, _)| index)
            .unwrap_or_else(|| {
                node.children
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, child)| child.visits)
                    .map(|(index, _)| index)
                    .unwrap_or(0)
            })
    }

    fn select_gumbel_interior_child(&self, node_index: usize) -> usize {
        let node = &self.nodes[node_index];
        let completed = self.gumbel_completed_q(node_index);
        let logits = node
            .children
            .iter()
            .zip(completed)
            .map(|(child, q)| child.prior.max(1e-12).ln() + q)
            .collect::<Vec<_>>();
        let mut probs = Vec::new();
        softmax_into(&logits, &mut probs);
        let denominator = 1.0 + node.children.iter().map(|child| child.visits).sum::<u32>() as f32;
        node.children
            .iter()
            .enumerate()
            .max_by(|(left, left_child), (right, right_child)| {
                let left_score = probs[*left] - left_child.visits as f32 / denominator;
                let right_score = probs[*right] - right_child.visits as f32 / denominator;
                left_score.total_cmp(&right_score)
            })
            .map(|(index, _)| index)
            .unwrap_or(0)
    }

    fn select_child(&self, node_index: usize) -> usize {
        if node_index == self.root {
            self.select_gumbel_root_child(node_index)
        } else {
            self.select_gumbel_interior_child(node_index)
        }
    }

    fn best_root_child(&self, node_index: usize) -> Option<usize> {
        let completed = self.gumbel_completed_q(node_index);
        let max_visits = self.nodes[node_index]
            .children
            .iter()
            .map(|child| child.visits)
            .max()?;
        self.nodes[node_index]
            .children
            .iter()
            .enumerate()
            .filter(|(_, child)| child.visits == max_visits)
            .max_by(|(left, _), (right, _)| {
                self.gumbel_score(*left, &completed)
                    .total_cmp(&self.gumbel_score(*right, &completed))
            })
            .map(|(index, _)| index)
    }

    fn root_policy(&self, node_index: usize) -> Vec<f32> {
        let completed = self.gumbel_completed_q(node_index);
        let logits = self.nodes[node_index]
            .children
            .iter()
            .zip(completed)
            .map(|(child, q)| child.prior.max(1e-12).ln() + q)
            .collect::<Vec<_>>();
        let mut policy = Vec::new();
        softmax_into(&logits, &mut policy);
        policy
    }
}

fn add_wdl(sum: &mut [f32; 3], wdl: [f32; 3]) {
    sum[0] += wdl[0];
    sum[1] += wdl[1];
    sum[2] += wdl[2];
}

fn flip_wdl(wdl: [f32; 3]) -> [f32; 3] {
    [wdl[2], wdl[1], wdl[0]]
}

fn scalar_terminal_wdl(value: f32) -> [f32; 3] {
    if value > 0.0 {
        [1.0, 0.0, 0.0]
    } else if value < 0.0 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    }
}

fn scale_wdl_value(wdl: [f32; 3], scale: f32) -> [f32; 3] {
    let scale = scale.clamp(0.0, 1.0);
    [
        wdl[0] * scale,
        wdl[1] + (1.0 - scale) * (wdl[0] + wdl[2]),
        wdl[2] * scale,
    ]
}

fn terminal_value(position: &Position, rule_history: &[RuleHistoryEntry]) -> Option<f32> {
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
    if let Some(outcome) = position.rule_outcome_with_history(rule_history) {
        return Some(match outcome {
            RuleOutcome::Draw(_) => 0.0,
            RuleOutcome::Win(color) => {
                if color == position.side_to_move() {
                    1.0
                } else {
                    -1.0
                }
            }
        });
    }
    None
}

fn softmax_into<'a>(logits: &[f32], output: &'a mut Vec<f32>) -> &'a mut Vec<f32> {
    output.clear();
    if logits.is_empty() {
        return output;
    }
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    output.reserve(logits.len());
    for &logit in logits {
        let value = (logit - max_logit).exp();
        output.push(value);
        sum += value;
    }
    let inv_sum = sum.max(1e-12).recip();
    for value in output.iter_mut() {
        *value *= inv_sum;
    }
    output
}

fn sequential_halving_visits(max_actions: usize, simulations: usize) -> Vec<u32> {
    if max_actions <= 1 {
        return (0..simulations as u32).collect();
    }
    let rounds = (max_actions as f32).log2().ceil() as usize;
    let mut sequence = Vec::with_capacity(simulations);
    let mut visits = vec![0u32; max_actions];
    let mut considered = max_actions;
    while sequence.len() < simulations {
        let extra = (simulations / (rounds.max(1) * considered)).max(1);
        for _ in 0..extra {
            sequence.extend_from_slice(&visits[..considered]);
            for visit in &mut visits[..considered] {
                *visit += 1;
            }
        }
        considered = (considered / 2).max(2);
    }
    sequence.truncate(simulations);
    sequence
}

pub(super) fn append_history(history: &mut Vec<HistoryMove>, position: &Position, mv: Move) {
    if let Some(piece) = position.piece_at(mv.from as usize) {
        history.push(HistoryMove {
            piece,
            captured: position.piece_at(mv.to as usize),
            mv,
        });
        let overflow = history.len().saturating_sub(crate::nnue::HISTORY_PLIES);
        if overflow > 0 {
            history.drain(0..overflow);
        }
    }
}

fn clone_history_with_appended_move(
    history: &[HistoryMove],
    position: &Position,
    mv: Move,
) -> Vec<HistoryMove> {
    let mut out = Vec::with_capacity((history.len() + 1).min(crate::nnue::HISTORY_PLIES));
    out.extend_from_slice(history);
    append_history(&mut out, position, mv);
    out
}

fn clone_rule_history_with_appended_entry(
    rule_history: &[RuleHistoryEntry],
    entry: RuleHistoryEntry,
) -> Vec<RuleHistoryEntry> {
    let mut out = Vec::with_capacity(rule_history.len() + 1);
    out.extend_from_slice(rule_history);
    out.push(entry);
    out
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

    fn test_search(position: &Position, model: &AzNnue, limits: AzSearchLimits) -> AzSearchResult {
        gumbel_search_with_history_and_rules(
            position,
            &[],
            None,
            None,
            model,
            limits,
            GumbelSearchConfig {
                gumbel_scale: 0.0,
                ..GumbelSearchConfig::default()
            },
        )
    }
    use crate::xiangqi::{RuleDrawReason, RuleOutcome};

    #[test]
    fn gumbel_search_populates_policy_distribution() {
        let model = AzNnue::random(4, 7);
        let result = test_search(
            &Position::startpos(),
            &model,
            AzSearchLimits {
                simulations: 128,
                seed: 11,
                max_depth: 0,
                value_scale: 1.0,
                ..AzSearchLimits::default()
            },
        );

        let total_policy = result
            .candidates
            .iter()
            .map(|candidate| candidate.policy)
            .sum::<f32>();

        assert_eq!(result.simulations, 128);
        assert!(result.best_move.is_some());
        assert!(
            result
                .candidates
                .iter()
                .any(|candidate| candidate.visits > 0)
        );
        assert!((total_policy - 1.0).abs() < 1e-3);
    }

    #[test]
    fn gumbel_search_uses_sequential_halving_and_improved_policy() {
        let model = AzNnue::random(4, 7);
        let simulations = 128;
        let considered = 8;
        let result = gumbel_search_with_history_and_rules(
            &Position::startpos(),
            &[],
            None,
            None,
            &model,
            AzSearchLimits {
                simulations,
                seed: 17,
                ..AzSearchLimits::default()
            },
            GumbelSearchConfig {
                max_num_considered_actions: considered,
                gumbel_scale: 1.0,
                ..GumbelSearchConfig::default()
            },
        );
        assert_eq!(
            result
                .candidates
                .iter()
                .map(|candidate| candidate.visits)
                .sum::<u32>(),
            simulations as u32
        );
        assert!(
            result
                .candidates
                .iter()
                .filter(|candidate| candidate.visits > 0)
                .count()
                <= considered
        );
        assert!(
            (result
                .candidates
                .iter()
                .map(|candidate| candidate.policy)
                .sum::<f32>()
                - 1.0)
                .abs()
                < 1e-5
        );
        assert!(result.best_move.is_some());
    }

    #[test]
    fn sequential_halving_schedule_matches_budget() {
        let schedule = sequential_halving_visits(16, 1200);
        assert_eq!(schedule.len(), 1200);
        assert_eq!(&schedule[..16], &[0; 16]);
        assert!(schedule.iter().copied().max().unwrap() > 1);
    }

    #[test]
    fn search_reports_leaf_depth_and_depth_cutoffs() {
        let model = AzNnue::random(4, 7);
        let result = test_search(
            &Position::startpos(),
            &model,
            AzSearchLimits {
                simulations: 32,
                seed: 13,
                max_depth: 1,
                value_scale: 1.0,
                ..AzSearchLimits::default()
            },
        );

        assert_eq!(result.simulations, 32);
        assert_eq!(result.search_depth_max, 1);
        assert_eq!(result.search_depth_limit, 1);
        assert!((result.search_depth_avg - 1.0).abs() < 1e-6);
        assert_eq!(result.search_depth_cutoffs, 32);
    }

    #[test]
    fn select_child_breaks_equal_scores_by_higher_prior() {
        let model = AzNnue::random(4, 7);
        let position = Position::startpos();
        let legal = position.legal_moves();
        assert!(legal.len() >= 2);

        let mut tree = AzTree::new(
            position.clone(),
            Vec::new(),
            position.initial_rule_history(),
            None,
            &model,
            AzSearchLimits {
                simulations: 1,
                seed: 31,
                max_depth: 0,
                value_scale: 1.0,
                ..AzSearchLimits::default()
            },
            GumbelSearchConfig::default(),
        );
        tree.nodes[tree.root].children = vec![
            AzChild {
                mv: legal[0],
                raw_prior: 0.10,
                prior: 0.10,
                visits: 1,
                value_sum: 0.0,
                value_wdl_sum: [0.0, 1.0, 0.0],
                moves_left_sum: 0.0,
                child: None,
            },
            AzChild {
                mv: legal[1],
                raw_prior: 0.90,
                prior: 0.90,
                visits: 1,
                value_sum: 0.0,
                value_wdl_sum: [0.0, 1.0, 0.0],
                moves_left_sum: 0.0,
                child: None,
            },
        ];

        assert_eq!(tree.select_child(tree.root), 1);
    }

    #[test]
    fn search_value_scale_reduces_non_terminal_network_value() {
        let position = Position::startpos();
        let mut model = AzNnue::random(4, 7);
        model.value_head_bias[0] = 2.0;
        model.value_q_output[0] = 1.0;

        let full = test_search(
            &position,
            &model,
            AzSearchLimits {
                simulations: 0,
                seed: 29,
                value_scale: 1.0,
                ..AzSearchLimits::default()
            },
        );
        let scaled = test_search(
            &position,
            &model,
            AzSearchLimits {
                simulations: 0,
                seed: 29,
                value_scale: 0.25,
                ..AzSearchLimits::default()
            },
        );

        assert!(full.value_q > 0.0);
        assert!((scaled.value_q - full.value_q * 0.25).abs() <= 1e-5);
    }

    #[test]
    fn mcts_state_make_move_matches_manual_context_updates() {
        let position = Position::startpos();
        let mv = position.legal_moves()[0];
        let mut node_position = position.clone();
        let mut node_history = Vec::new();
        let mut node_rule_history = position.initial_rule_history();

        let mut manual_position = position;
        let mut manual_history = Vec::new();
        let mut manual_rule_history = manual_position.initial_rule_history();
        append_history(&mut manual_history, &manual_position, mv);
        manual_rule_history.push(manual_position.rule_history_entry_after_move(mv));
        manual_position.make_move(mv);

        append_history(&mut node_history, &node_position, mv);
        node_rule_history.push(node_position.rule_history_entry_after_move(mv));
        node_position.make_move(mv);

        assert_eq!(node_position, manual_position);
        assert_eq!(node_history, manual_history);
        assert_eq!(node_rule_history, manual_rule_history);
    }

    #[test]
    fn mcts_child_rule_history_uses_after_move_semantics() {
        let mut position = Position::from_fen(
            "r3kab1r/4a4/2n1bc2n/p1p1p1pc1/8p/5NP2/P1P1P3P/2N1C2C1/8R/1RBAKAB2 w",
        )
        .unwrap();
        let mut rule_history = position.initial_rule_history();
        let mut found = None;
        for text in [
            "f4d5", "c6c5", "d5c7", "f7c7", "i1d1", "a9d9", "d1d9", "e8d9", "b0b4", "i9i8", "c3c4",
            "i8d8", "c4c5", "e7c5", "b4f4", "i7h5", "f4f5", "h6h2", "f5h5", "c7c2", "h5c5", "d8d3",
            "e3e4", "d3e3", "a3a4", "c2c3", "c5i5", "e3e4", "i5c5", "c3b3", "c5c3", "b3b5", "c3c5",
            "b5b0", "c5h5", "h2f2", "h5b5", "b0a0", "b5b0", "a0a3", "b0b3", "a3a0", "b3a3", "a0b0",
            "a3b3", "b0a0", "b3a3", "a0b0", "a3b3", "b0a0", "b3a3", "a0b0", "a3b3", "b0a0",
        ] {
            let mv = Move::from_uci(text).unwrap();
            assert!(position.legal_moves_with_rules(&rule_history).contains(&mv));
            let mover = position.side_to_move();
            let expected = position.rule_history_entry_after_move(mv);
            let mut wrong_next = position.clone();
            wrong_next.make_move(mv);
            let wrong = wrong_next.rule_history_entry(Some(mover));
            if expected != wrong {
                found = Some((position.clone(), rule_history.clone(), mv, expected, wrong));
                break;
            }
            rule_history.push(expected);
            position.make_move(mv);
        }
        let Some((position, rule_history, mv, expected, wrong)) = found else {
            panic!("test line should contain a chased-piece escape");
        };
        assert_ne!(expected, wrong);

        let model = AzNnue::random(4, 11);
        let mut tree = AzTree::new(
            position.clone(),
            Vec::new(),
            rule_history,
            Some(vec![mv]),
            &model,
            AzSearchLimits {
                simulations: 1,
                seed: 3,
                ..AzSearchLimits::default()
            },
            GumbelSearchConfig::default(),
        );
        tree.expand(tree.root);
        tree.simulate_child(tree.root, 0, 1);
        let child_node = tree.nodes[tree.root].children[0].child.unwrap();
        assert_eq!(
            tree.nodes[child_node].rule_history.last().copied(),
            Some(expected)
        );
    }

    #[test]
    fn provided_root_moves_only_apply_at_root() {
        let position = Position::startpos();
        let legal = position.legal_moves();
        let root_moves = vec![legal[0]];
        let model = AzNnue::random(4, 7);
        let mut tree = AzTree::new(
            position,
            Vec::new(),
            Position::startpos().initial_rule_history(),
            Some(root_moves.clone()),
            &model,
            AzSearchLimits::default(),
            GumbelSearchConfig::default(),
        );

        tree.expand(tree.root);
        assert_eq!(tree.nodes[tree.root].children.len(), 1);
        let child_index = 0;
        tree.simulate_child(tree.root, child_index, 1);
        let child_node = tree.nodes[tree.root].children[child_index].child.unwrap();
        tree.expand(child_node);
        assert_ne!(tree.nodes[child_node].children.len(), root_moves.len());
    }

    #[test]
    fn terminal_value_uses_rule_history_not_just_board_hash() {
        let position = Position::startpos();
        let rule_history = vec![
            position.rule_history_entry(None),
            RuleHistoryEntry {
                hash: position.hash(),
                side_to_move: position.side_to_move(),
                mover: Some(Color::Black),
                gives_check: false,
                chased_mask: 0,
                chased_piece_mask: 0,
            },
            RuleHistoryEntry {
                hash: position.hash(),
                side_to_move: position.side_to_move(),
                mover: Some(Color::Black),
                gives_check: false,
                chased_mask: 0,
                chased_piece_mask: 0,
            },
            RuleHistoryEntry {
                hash: position.hash(),
                side_to_move: position.side_to_move(),
                mover: Some(Color::Black),
                gives_check: false,
                chased_mask: 0,
                chased_piece_mask: 0,
            },
            RuleHistoryEntry {
                hash: position.hash(),
                side_to_move: position.side_to_move(),
                mover: Some(Color::Black),
                gives_check: false,
                chased_mask: 0,
                chased_piece_mask: 0,
            },
            RuleHistoryEntry {
                hash: position.hash(),
                side_to_move: position.side_to_move(),
                mover: Some(Color::Black),
                gives_check: false,
                chased_mask: 0,
                chased_piece_mask: 0,
            },
        ];

        assert_eq!(
            terminal_value(&position, &rule_history),
            Some(0.0),
            "repetition outcome should come from rule history even when board is unchanged"
        );
        assert_eq!(
            position.rule_outcome_with_history(&rule_history),
            Some(RuleOutcome::Draw(RuleDrawReason::Repetition))
        );
    }
}
