use crate::board_transform::HistoryMove;
use crate::xiangqi::{Color, Move, Position, RuleHistoryEntry, RuleOutcome};

use super::mctx::{self, ActionStats, AzGumbelConfig};
use super::{AzEvalScratch, AzModel, SplitMix64, VALUE_SCALE_CP};

const DEFAULT_CPUCT: f32 = 1.5;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AzSearchAlgorithm {
    #[default]
    AlphaZero,
    GumbelAlphaZero,
}

impl AzSearchAlgorithm {
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "alphazero" | "alpha_zero" | "puct" => Some(Self::AlphaZero),
            "gumbel" | "gumbel_alphazero" | "gumbel-alpha-zero" | "gumbel_alpha_zero" => {
                Some(Self::GumbelAlphaZero)
            }
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::AlphaZero => "alphazero",
            Self::GumbelAlphaZero => "gumbel_alphazero",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AzSearchLimits {
    pub simulations: usize,
    pub seed: u64,
    pub cpuct: f32,
    pub root_dirichlet_alpha: f32,
    pub root_exploration_fraction: f32,
    pub algorithm: AzSearchAlgorithm,
    pub gumbel: AzGumbelConfig,
}

impl Default for AzSearchLimits {
    fn default() -> Self {
        Self {
            simulations: 10_000,
            seed: 0,
            cpuct: DEFAULT_CPUCT,
            root_dirichlet_alpha: 0.0,
            root_exploration_fraction: 0.0,
            algorithm: AzSearchAlgorithm::AlphaZero,
            gumbel: AzGumbelConfig::default(),
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

pub fn alphazero_search_with_history_and_rules(
    position: &Position,
    history: &[HistoryMove],
    rule_history: Option<Vec<RuleHistoryEntry>>,
    root_moves: Option<Vec<Move>>,
    model: &AzModel,
    limits: AzSearchLimits,
) -> AzSearchResult {
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
            candidates: Vec::new(),
        };
    }

    let mut used = 0usize;
    for _ in 0..limits.simulations {
        tree.simulate(root);
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
        candidates,
    }
}

pub fn alphazero_search(
    position: &Position,
    model: &AzModel,
    limits: AzSearchLimits,
) -> AzSearchResult {
    alphazero_search_with_history_and_rules(position, &[], None, None, model, limits)
}

struct AzTree<'a> {
    nodes: Vec<AzNode>,
    model: &'a AzModel,
    root_moves: Option<Vec<Move>>,
    root: usize,
    cpuct: f32,
    root_dirichlet_alpha: f32,
    root_exploration_fraction: f32,
    root_noise_seed: u64,
    algorithm: AzSearchAlgorithm,
    gumbel: AzGumbelConfig,
    num_simulations: usize,
    root_gumbels: Vec<f32>,
    root_considered_visits: Vec<u32>,
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
    logit: f32,
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
        model: &'a AzModel,
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
            root_dirichlet_alpha: limits.root_dirichlet_alpha.max(0.0),
            root_exploration_fraction: limits.root_exploration_fraction.clamp(0.0, 1.0),
            root_noise_seed: limits.seed,
            algorithm: limits.algorithm,
            gumbel: AzGumbelConfig {
                max_num_considered_actions: limits.gumbel.max_num_considered_actions.max(1),
                gumbel_scale: limits.gumbel.gumbel_scale.max(0.0),
                value_scale: limits.gumbel.value_scale.max(0.0),
                maxvisit_init: limits.gumbel.maxvisit_init.max(0.0),
                rescale_values: limits.gumbel.rescale_values,
                use_mixed_value: limits.gumbel.use_mixed_value,
            },
            num_simulations: limits.simulations,
            root_gumbels: Vec::new(),
            root_considered_visits: Vec::new(),
            eval_scratch: AzEvalScratch::new(model.hidden_size),
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
        );
        let priors = softmax_into(
            &self.eval_scratch.logits[..moves.len()],
            &mut self.eval_scratch.priors,
        );
        if node_index == self.root
            && self.algorithm == AzSearchAlgorithm::AlphaZero
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
            .enumerate()
            .map(|(index, (mv, prior))| AzChild {
                mv,
                logit: self.eval_scratch.logits[index],
                prior,
                visits: 0,
                value_sum: 0.0,
                child: None,
            })
            .collect();
        if node_index == self.root && self.algorithm == AzSearchAlgorithm::GumbelAlphaZero {
            self.root_gumbels = mctx::sample_gumbels(
                self.nodes[node_index].children.len(),
                self.gumbel.gumbel_scale,
                self.root_noise_seed,
            );
            let considered = self
                .gumbel
                .max_num_considered_actions
                .min(self.nodes[node_index].children.len())
                .max(1);
            self.root_considered_visits =
                mctx::get_sequence_of_considered_visits(considered, self.num_simulations);
        }
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
                let child_history = clone_history_with_appended_move(
                    &self.nodes[node_index].history,
                    &child_position,
                    mv,
                );
                let child_rule_history = clone_rule_history_with_appended_move(
                    &self.nodes[node_index].rule_history,
                    &child_position,
                    mv,
                );
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
        if self.algorithm == AzSearchAlgorithm::GumbelAlphaZero {
            if node_index == self.root {
                return self.select_gumbel_root_child(node_index);
            }
            return self.select_gumbel_interior_child(node_index);
        }

        let parent_visits_sqrt = (self.nodes[node_index].visits.max(1) as f32).sqrt();
        self.nodes[node_index]
            .children
            .iter()
            .enumerate()
            .max_by(|(left_index, left_child), (right_index, right_child)| {
                let left_score = puct_score(left_child, parent_visits_sqrt, self.cpuct);
                let right_score = puct_score(right_child, parent_visits_sqrt, self.cpuct);
                left_score
                    .total_cmp(&right_score)
                    .then_with(|| right_child.prior.total_cmp(&left_child.prior))
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .unwrap_or(0)
    }

    fn best_root_child(&self, node_index: usize) -> Option<usize> {
        if self.algorithm == AzSearchAlgorithm::GumbelAlphaZero {
            return self.best_gumbel_root_child(node_index);
        }
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

    fn select_gumbel_root_child(&self, node_index: usize) -> usize {
        mctx::gumbel_muzero_root_action_selection(
            &self.action_stats(node_index),
            &self.root_gumbels,
            &self.root_considered_visits,
            self.gumbel,
            self.nodes[node_index].value,
        )
    }

    fn select_gumbel_interior_child(&self, node_index: usize) -> usize {
        mctx::gumbel_muzero_interior_action_selection(
            &self.action_stats(node_index),
            self.gumbel,
            self.nodes[node_index].value,
        )
    }

    fn best_gumbel_root_child(&self, node_index: usize) -> Option<usize> {
        mctx::gumbel_muzero_root_best_action(
            &self.action_stats(node_index),
            &self.root_gumbels,
            self.gumbel,
            self.nodes[node_index].value,
        )
    }

    fn root_policy(&self, node_index: usize) -> Vec<f32> {
        if self.algorithm == AzSearchAlgorithm::GumbelAlphaZero {
            return mctx::gumbel_muzero_root_policy(
                &self.action_stats(node_index),
                self.gumbel,
                self.nodes[node_index].value,
            );
        }

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

    fn action_stats(&self, node_index: usize) -> Vec<ActionStats> {
        self.nodes[node_index]
            .children
            .iter()
            .map(|child| ActionStats {
                logit: child.logit,
                visit_count: child.visits,
                qvalue: child.q(),
            })
            .collect()
    }
}

fn puct_score(child: &AzChild, parent_visits_sqrt: f32, cpuct: f32) -> f32 {
    child.q() + cpuct * child.prior * parent_visits_sqrt / (1.0 + child.visits as f32)
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
        let overflow = history
            .len()
            .saturating_sub(crate::board_transform::HISTORY_PLIES);
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
    let mut out =
        Vec::with_capacity((history.len() + 1).min(crate::board_transform::HISTORY_PLIES));
    out.extend_from_slice(history);
    append_history(&mut out, position, mv);
    out
}

fn clone_rule_history_with_appended_move(
    rule_history: &[RuleHistoryEntry],
    position: &Position,
    mv: Move,
) -> Vec<RuleHistoryEntry> {
    let mut out = Vec::with_capacity(rule_history.len() + 1);
    out.extend_from_slice(rule_history);
    out.push(position.rule_history_entry_after_move(mv));
    out
}

fn truncate_history(history: &[HistoryMove]) -> Vec<HistoryMove> {
    history
        .iter()
        .rev()
        .take(crate::board_transform::HISTORY_PLIES)
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
        let model = AzModel::random(4, 7);
        let result = alphazero_search(
            &Position::startpos(),
            &model,
            AzSearchLimits {
                simulations: 128,
                seed: 11,
                cpuct: 1.5,
                root_dirichlet_alpha: 0.0,
                root_exploration_fraction: 0.0,
                algorithm: AzSearchAlgorithm::AlphaZero,
                gumbel: AzGumbelConfig::default(),
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
    fn dirichlet_noise_changes_root_prior_distribution() {
        let position = Position::startpos();
        let model = AzModel::random(4, 7);
        let plain = alphazero_search(
            &position,
            &model,
            AzSearchLimits {
                simulations: 1,
                seed: 19,
                cpuct: 1.5,
                root_dirichlet_alpha: 0.0,
                root_exploration_fraction: 0.0,
                algorithm: AzSearchAlgorithm::AlphaZero,
                gumbel: AzGumbelConfig::default(),
            },
        );
        let noisy = alphazero_search(
            &position,
            &model,
            AzSearchLimits {
                simulations: 1,
                seed: 19,
                cpuct: 1.5,
                root_dirichlet_alpha: 0.3,
                root_exploration_fraction: 0.25,
                algorithm: AzSearchAlgorithm::AlphaZero,
                gumbel: AzGumbelConfig::default(),
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
    fn provided_root_moves_only_apply_at_root() {
        let position = Position::startpos();
        let legal = position.legal_moves();
        let root_moves = vec![legal[0]];
        let model = AzModel::random(4, 7);
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
        tree.simulate_child(tree.root, child_index);
        let child_node = tree.nodes[tree.root].children[child_index].child.unwrap();
        tree.expand(child_node);
        assert_ne!(tree.nodes[child_node].children.len(), root_moves.len());
    }

    #[test]
    fn gumbel_search_uses_improved_policy_targets() {
        let model = AzModel::random(4, 7);
        let result = alphazero_search(
            &Position::startpos(),
            &model,
            AzSearchLimits {
                simulations: 64,
                seed: 23,
                cpuct: 1.5,
                root_dirichlet_alpha: 0.3,
                root_exploration_fraction: 0.25,
                algorithm: AzSearchAlgorithm::GumbelAlphaZero,
                gumbel: AzGumbelConfig::default(),
            },
        );

        let total_policy = result
            .candidates
            .iter()
            .map(|candidate| candidate.policy)
            .sum::<f32>();

        assert_eq!(result.simulations, 64);
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
            },
            RuleHistoryEntry {
                hash: position.hash(),
                side_to_move: position.side_to_move(),
                mover: Some(Color::Black),
                gives_check: false,
                chased_mask: 0,
            },
            RuleHistoryEntry {
                hash: position.hash(),
                side_to_move: position.side_to_move(),
                mover: Some(Color::Black),
                gives_check: false,
                chased_mask: 0,
            },
            RuleHistoryEntry {
                hash: position.hash(),
                side_to_move: position.side_to_move(),
                mover: Some(Color::Black),
                gives_check: false,
                chased_mask: 0,
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
