use crate::nnue::HistoryMove;
use crate::xiangqi::{Color, Move, Position, RuleHistoryEntry, RuleOutcome};

use super::{AzEvalScratch, AzNnue, SplitMix64, VALUE_SCALE_CP};

const DEFAULT_CPUCT: f32 = 1.5;

#[derive(Clone, Copy, Debug)]
pub struct AzSearchLimits {
    pub simulations: usize,
    pub seed: u64,
    pub cpuct: f32,
    pub cpuct_at_root: f32,
    /// Maximum search depth in plies below root. 0 keeps the default:
    /// max_depth = num_simulations.
    pub max_depth: usize,
    pub root_dirichlet_alpha: f32,
    pub root_exploration_fraction: f32,
    pub fpu_value: f32,
    pub fpu_value_at_root: f32,
    pub value_scale: f32,
}

impl Default for AzSearchLimits {
    fn default() -> Self {
        Self {
            simulations: 10_000,
            seed: 0,
            cpuct: DEFAULT_CPUCT,
            cpuct_at_root: DEFAULT_CPUCT,
            max_depth: 0,
            root_dirichlet_alpha: 0.0,
            root_exploration_fraction: 0.0,
            fpu_value: 0.23,
            fpu_value_at_root: 1.0,
            value_scale: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AzCandidate {
    pub mv: Move,
    pub visits: u32,
    pub q: f32,
    pub raw_prior: f32,
    pub prior: f32,
    pub policy: f32,
}

#[derive(Clone, Debug)]
pub struct AzSearchResult {
    pub best_move: Option<Move>,
    pub value_cp: i32,
    pub simulations: usize,
    pub search_depth_avg: f32,
    pub search_depth_max: usize,
    pub search_depth_limit: usize,
    pub search_depth_cutoffs: usize,
    pub candidates: Vec<AzCandidate>,
}

pub fn alphazero_search_with_history_and_rules(
    position: &Position,
    history: &[HistoryMove],
    rule_history: Option<Vec<RuleHistoryEntry>>,
    root_moves: Option<Vec<Move>>,
    model: &AzNnue,
    limits: AzSearchLimits,
) -> AzSearchResult {
    crate::scope_profile!("az.alphazero_search");
    let mut tree = AzTree::new(
        position.clone(),
        truncate_history(history),
        rule_history.unwrap_or_else(|| position.initial_rule_history()),
        root_moves,
        model,
        limits,
    );
    let root = tree.root;
    tree.expand(root);
    if tree.nodes[root].children.is_empty() {
        return AzSearchResult {
            best_move: None,
            value_cp: (tree.nodes[root].value * VALUE_SCALE_CP) as i32,
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
    let policy = tree.root_policy(root);
    let mut candidates = root_node
        .children
        .iter()
        .zip(policy)
        .map(|(child, policy)| AzCandidate {
            mv: child.mv,
            visits: child.visits,
            q: child.q(),
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
        value_cp: (searched_value * VALUE_SCALE_CP) as i32,
        simulations: used,
        search_depth_avg: tree.search_depth_avg(),
        search_depth_max: tree.search_depth_max,
        search_depth_limit: tree.max_depth,
        search_depth_cutoffs: tree.search_depth_cutoffs,
        candidates,
    }
}

pub fn alphazero_search(
    position: &Position,
    model: &AzNnue,
    limits: AzSearchLimits,
) -> AzSearchResult {
    alphazero_search_with_history_and_rules(position, &[], None, None, model, limits)
}

struct AzTree<'a> {
    nodes: Vec<AzNode>,
    model: &'a AzNnue,
    root_moves: Option<Vec<Move>>,
    root: usize,
    cpuct: f32,
    cpuct_at_root: f32,
    root_dirichlet_alpha: f32,
    root_exploration_fraction: f32,
    root_noise_seed: u64,
    fpu_value: f32,
    fpu_value_at_root: f32,
    value_scale: f32,
    max_depth: usize,
    search_depth_sum: usize,
    search_depth_count: usize,
    search_depth_max: usize,
    search_depth_cutoffs: usize,
    eval_scratch: AzEvalScratch,
}

struct AzNode {
    position: Position,
    history: Vec<HistoryMove>,
    rule_history: Vec<RuleHistoryEntry>,
    children: Vec<AzChild>,
    visits: u32,
    value_sum: f32,
    value: f32,
    expanded: bool,
}

#[derive(Clone)]
struct AzChild {
    mv: Move,
    raw_prior: f32,
    prior: f32,
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
        rule_history: Vec<RuleHistoryEntry>,
        root_moves: Option<Vec<Move>>,
        model: &'a AzNnue,
        limits: AzSearchLimits,
    ) -> Self {
        let mut nodes = Vec::with_capacity(limits.simulations.saturating_add(1));
        nodes.push(AzNode {
            position,
            history,
            rule_history,
            children: Vec::new(),
            visits: 0,
            value_sum: 0.0,
            value: 0.0,
            expanded: false,
        });
        Self {
            nodes,
            model,
            root_moves,
            root: 0,
            cpuct: if limits.cpuct > 0.0 {
                limits.cpuct
            } else {
                DEFAULT_CPUCT
            },
            cpuct_at_root: if limits.cpuct_at_root > 0.0 {
                limits.cpuct_at_root
            } else if limits.cpuct > 0.0 {
                limits.cpuct
            } else {
                DEFAULT_CPUCT
            },
            root_dirichlet_alpha: limits.root_dirichlet_alpha.max(0.0),
            root_exploration_fraction: limits.root_exploration_fraction.clamp(0.0, 1.0),
            root_noise_seed: limits.seed,
            fpu_value: limits.fpu_value.max(0.0),
            fpu_value_at_root: limits.fpu_value_at_root.clamp(-1.0, 1.0),
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
        }
    }

    fn expand(&mut self, node_index: usize) -> f32 {
        if self.nodes[node_index].expanded {
            return self.nodes[node_index].value;
        }

        if let Some(value) = terminal_value(
            &self.nodes[node_index].position,
            &self.nodes[node_index].rule_history,
        ) {
            self.nodes[node_index].children.clear();
            self.nodes[node_index].value = value;
            self.nodes[node_index].expanded = true;
            return value;
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
            self.nodes[node_index].expanded = true;
            return -1.0;
        }

        let value = self.model.evaluate_with_scratch(
            &self.nodes[node_index].position,
            &self.nodes[node_index].history,
            &moves,
            &mut self.eval_scratch,
        ) * self.value_scale;
        let priors = softmax_into(
            &self.eval_scratch.logits[..moves.len()],
            &mut self.eval_scratch.priors,
        );
        let raw_priors = priors.clone();
        if node_index == self.root
            && self.root_dirichlet_alpha > 0.0
            && self.root_exploration_fraction > 0.0
        {
            apply_root_dirichlet_noise(
                priors,
                self.root_dirichlet_alpha,
                self.root_exploration_fraction,
                self.root_noise_seed,
            );
        }
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
                child: None,
            })
            .collect();
        self.nodes[node_index].value = value;
        self.nodes[node_index].expanded = true;
        value
    }

    fn simulate(&mut self, node_index: usize, depth: usize) -> f32 {
        if depth >= self.max_depth {
            let value = self.cutoff_value(node_index);
            self.nodes[node_index].visits += 1;
            self.nodes[node_index].value_sum += value;
            self.record_leaf_depth(depth, true);
            return value;
        }
        if !self.nodes[node_index].expanded {
            let value = self.expand(node_index);
            self.nodes[node_index].visits += 1;
            self.nodes[node_index].value_sum += value;
            self.record_leaf_depth(depth, false);
            return value;
        }
        if self.nodes[node_index].children.is_empty() {
            self.nodes[node_index].visits += 1;
            self.nodes[node_index].value_sum += self.nodes[node_index].value;
            self.record_leaf_depth(depth, false);
            return self.nodes[node_index].value;
        }
        let child_index = self.select_child(node_index);
        self.simulate_child(node_index, child_index, depth + 1)
    }

    fn simulate_child(&mut self, node_index: usize, child_index: usize, child_depth: usize) -> f32 {
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
                    value: 0.0,
                    expanded: false,
                });
                self.nodes[node_index].children[child_index].child = Some(child_node);
                child_node
            };
        let child_value = self.simulate(child_node, child_depth);
        let value = -child_value;
        let child = &mut self.nodes[node_index].children[child_index];
        child.visits += 1;
        child.value_sum += value;
        self.nodes[node_index].visits += 1;
        self.nodes[node_index].value_sum += value;
        value
    }

    fn cutoff_value(&mut self, node_index: usize) -> f32 {
        if self.nodes[node_index].expanded {
            return self.nodes[node_index].value;
        }
        if let Some(value) = terminal_value(
            &self.nodes[node_index].position,
            &self.nodes[node_index].rule_history,
        ) {
            self.nodes[node_index].value = value;
            return value;
        }
        let moves = self.nodes[node_index]
            .position
            .legal_moves_with_rules(&self.nodes[node_index].rule_history);
        if moves.is_empty() {
            self.nodes[node_index].value = -1.0;
            return -1.0;
        }
        let value = self.model.evaluate_with_scratch(
            &self.nodes[node_index].position,
            &self.nodes[node_index].history,
            &moves,
            &mut self.eval_scratch,
        ) * self.value_scale;
        self.nodes[node_index].value = value;
        value
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

    fn select_child(&self, node_index: usize) -> usize {
        let node = &self.nodes[node_index];
        let parent_visits_sqrt = (node.visits.max(1) as f32).sqrt();
        let is_root = node_index == self.root;
        let fpu_value = if is_root {
            self.fpu_value_at_root
        } else {
            alphazero_fpu_value_reduction(node, self.fpu_value)
        };
        let cpuct = if is_root {
            self.cpuct_at_root
        } else {
            self.cpuct
        };
        self.nodes[node_index]
            .children
            .iter()
            .enumerate()
            .max_by(|(left_index, left_child), (right_index, right_child)| {
                let left_score = puct_score(left_child, fpu_value, parent_visits_sqrt, cpuct);
                let right_score = puct_score(right_child, fpu_value, parent_visits_sqrt, cpuct);
                left_score
                    .total_cmp(&right_score)
                    .then_with(|| right_child.prior.total_cmp(&left_child.prior))
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .unwrap_or(0)
    }

    fn best_root_child(&self, node_index: usize) -> Option<usize> {
        self.nodes[node_index]
            .children
            .iter()
            .enumerate()
            .max_by(|(left_index, left_child), (right_index, right_child)| {
                left_child
                    .visits
                    .cmp(&right_child.visits)
                    .then_with(|| left_child.q().total_cmp(&right_child.q()))
                    .then_with(|| left_child.prior.total_cmp(&right_child.prior))
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
    }

    fn root_policy(&self, node_index: usize) -> Vec<f32> {
        let total_visits = self.nodes[node_index]
            .children
            .iter()
            .map(|child| child.visits as f32)
            .sum::<f32>()
            .max(1.0);
        if self.nodes[node_index]
            .children
            .iter()
            .any(|child| child.visits > 0)
        {
            return self.nodes[node_index]
                .children
                .iter()
                .map(|child| child.visits as f32 / total_visits)
                .collect();
        }

        let total_prior = self.nodes[node_index]
            .children
            .iter()
            .map(|child| child.prior)
            .sum::<f32>()
            .max(1e-12);
        self.nodes[node_index]
            .children
            .iter()
            .map(|child| child.prior / total_prior)
            .collect()
    }

}

fn alphazero_fpu_value_reduction(node: &AzNode, reduction: f32) -> f32 {
    let parent_q = if node.visits > 0 {
        node.value_sum / node.visits as f32
    } else {
        node.value
    };
    if reduction <= 0.0 {
        return parent_q;
    }

    let visited_prior = node
        .children
        .iter()
        .filter(|child| child.visits > 0)
        .map(|child| child.prior.max(0.0))
        .sum::<f32>()
        .clamp(0.0, 1.0);
    (parent_q - reduction * visited_prior.sqrt()).clamp(-1.0, 1.0)
}

fn puct_score(child: &AzChild, fpu_value: f32, parent_visits_sqrt: f32, cpuct: f32) -> f32 {
    let value_score = if child.visits > 0 {
        child.q()
    } else {
        fpu_value
    };
    value_score + cpuct * child.prior * parent_visits_sqrt / (1.0 + child.visits as f32)
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

fn apply_root_dirichlet_noise(
    priors: &mut [f32],
    alpha: f32,
    exploration_fraction: f32,
    seed: u64,
) {
    let noise = sample_dirichlet(priors.len(), alpha, seed);
    let keep = 1.0 - exploration_fraction;
    for (prior, noise_value) in priors.iter_mut().zip(noise) {
        *prior = keep * *prior + exploration_fraction * noise_value;
    }
}

fn sample_dirichlet(dim: usize, alpha: f32, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed ^ 0xD1A1_71C7_0000_0000u64 ^ dim as u64);
    let mut samples = Vec::with_capacity(dim);
    let mut sum = 0.0f32;
    for index in 0..dim {
        let value = sample_gamma(alpha.max(1e-3), &mut rng, seed ^ index as u64).max(1e-12);
        samples.push(value);
        sum += value;
    }
    let inv_sum = sum.max(1e-12).recip();
    for value in &mut samples {
        *value *= inv_sum;
    }
    samples
}

fn sample_gamma(alpha: f32, rng: &mut SplitMix64, salt: u64) -> f32 {
    if alpha < 1.0 {
        let u = rng.unit_f32().max(1e-12);
        return sample_gamma(alpha + 1.0, rng, salt) * u.powf(1.0 / alpha);
    }

    let d = alpha - 1.0 / 3.0;
    let c = (1.0 / (9.0 * d)).sqrt();
    loop {
        let x = sample_standard_normal(rng, salt);
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v3 = v * v * v;
        let u = rng.unit_f32().max(1e-12);
        if u < 1.0 - 0.0331 * x * x * x * x {
            return d * v3;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v3 + v3.ln()) {
            return d * v3;
        }
    }
}

fn sample_standard_normal(rng: &mut SplitMix64, salt: u64) -> f32 {
    let u1 = rng.unit_f32().max(1e-12);
    let mut aux = SplitMix64::new(rng.next_u64() ^ salt.rotate_left(17));
    let u2 = aux.unit_f32();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
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
    use crate::xiangqi::{RuleDrawReason, RuleOutcome};

    #[test]
    fn alphazero_search_populates_visit_distribution() {
        let model = AzNnue::random(4, 7);
        let result = alphazero_search(
            &Position::startpos(),
            &model,
            AzSearchLimits {
                simulations: 128,
                seed: 11,
                cpuct: 1.5,
                cpuct_at_root: 1.5,
                max_depth: 0,
                root_dirichlet_alpha: 0.0,
                root_exploration_fraction: 0.0,
                fpu_value: 0.23,
                fpu_value_at_root: 1.0,
                value_scale: 1.0,
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
    fn search_reports_leaf_depth_and_depth_cutoffs() {
        let model = AzNnue::random(4, 7);
        let result = alphazero_search(
            &Position::startpos(),
            &model,
            AzSearchLimits {
                simulations: 32,
                seed: 13,
                cpuct: 1.5,
                cpuct_at_root: 1.5,
                max_depth: 1,
                root_dirichlet_alpha: 0.0,
                root_exploration_fraction: 0.0,
                fpu_value: 0.23,
                fpu_value_at_root: 1.0,
                value_scale: 1.0,
            },
        );

        assert_eq!(result.simulations, 32);
        assert_eq!(result.search_depth_max, 1);
        assert_eq!(result.search_depth_limit, 1);
        assert!((result.search_depth_avg - 1.0).abs() < 1e-6);
        assert_eq!(result.search_depth_cutoffs, 32);
    }

    #[test]
    fn dirichlet_noise_changes_root_prior_distribution() {
        let position = Position::startpos();
        let model = AzNnue::random(4, 7);
        let plain = alphazero_search(
            &position,
            &model,
            AzSearchLimits {
                simulations: 1,
                seed: 19,
                cpuct: 1.5,
                cpuct_at_root: 1.5,
                max_depth: 0,
                root_dirichlet_alpha: 0.0,
                root_exploration_fraction: 0.0,
                fpu_value: 0.23,
                fpu_value_at_root: 1.0,
                value_scale: 1.0,
            },
        );
        let noisy = alphazero_search(
            &position,
            &model,
            AzSearchLimits {
                simulations: 1,
                seed: 19,
                cpuct: 1.5,
                cpuct_at_root: 1.5,
                max_depth: 0,
                root_dirichlet_alpha: 0.3,
                root_exploration_fraction: 0.25,
                fpu_value: 0.23,
                fpu_value_at_root: 1.0,
                value_scale: 1.0,
            },
        );

        assert_eq!(plain.candidates.len(), noisy.candidates.len());
        assert!(
            plain
                .candidates
                .iter()
                .zip(&noisy.candidates)
                .any(|(left, right)| (left.prior - right.prior).abs() > 1e-6)
        );
    }

    #[test]
    fn search_value_scale_reduces_non_terminal_network_value() {
        let position = Position::startpos();
        let mut model = AzNnue::random(4, 7);
        model.value_head_bias[0] = 2.0;
        model.value_head_output[0] = 1.0;

        let full = alphazero_search(
            &position,
            &model,
            AzSearchLimits {
                simulations: 0,
                seed: 29,
                value_scale: 1.0,
                ..AzSearchLimits::default()
            },
        );
        let scaled = alphazero_search(
            &position,
            &model,
            AzSearchLimits {
                simulations: 0,
                seed: 29,
                value_scale: 0.25,
                ..AzSearchLimits::default()
            },
        );

        assert!(full.value_cp > 0);
        assert!((scaled.value_cp as f32 - full.value_cp as f32 * 0.25).abs() <= 2.0);
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
