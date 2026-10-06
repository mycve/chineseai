use crate::xiangqi::{Color, Move, Position, RuleHistoryEntry, RuleOutcome};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

#[path = "tactical.rs"]
mod tactical;
#[path = "uci_search.rs"]
mod uci_search;
pub(crate) use uci_search::{AzUciSearchResult, search_uci};

use super::{
    AzEvalAccumulator, AzEvalOutput, AzEvalScratch, AzNnue, POLICY_ACCUMULATOR_RANK, SplitMix64,
    check_context_features, color_index, mate, rule_context_features,
};

const DEFAULT_CPUCT: f32 = 1.0;
const DEFAULT_CPUCT_AT_ROOT: f32 = 1.9;
const DEFAULT_CPUCT_BASE: f32 = 38739.0;
const DEFAULT_CPUCT_FACTOR: f32 = 3.894;
const NO_CHILD: u32 = u32::MAX;
const SEARCH_PROGRESS_POLL_SIMULATIONS: usize = 64;
const SEARCH_PROGRESS_INTERVAL: Duration = Duration::from_millis(250);
const INITIAL_TREE_NODE_CAPACITY: usize = 4_096;
const INITIAL_CHILDREN_PER_NODE_ESTIMATE: usize = 8;
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AzSearchLimits {
    pub simulations: usize,
    pub seed: u64,
    pub cpuct: f32,
    pub cpuct_at_root: f32,
    pub cpuct_base: f32,
    pub cpuct_factor: f32,
    pub cpuct_base_at_root: f32,
    pub cpuct_factor_at_root: f32,
    /// Maximum search depth in plies below root. 0 keeps the default:
    /// max_depth = num_simulations.
    pub max_depth: usize,
    /// 每个根走法的 Dirichlet alpha，0 关闭噪声。
    pub root_dirichlet_alpha: f32,
    pub root_exploration_fraction: f32,
    pub fpu_value: f32,
    pub fpu_value_at_root: f32,
    pub fpu_absolute_at_root: bool,
    pub minimum_kldgain_per_node: f32,
    /// Divisor applied to policy logits before softmax. Values above 1 flatten priors.
    pub policy_softmax_temp: f32,
    pub draw_score: f32,
    pub value_scale: f32,
}

impl Default for AzSearchLimits {
    fn default() -> Self {
        Self {
            simulations: 10_000,
            seed: 0,
            cpuct: DEFAULT_CPUCT,
            cpuct_at_root: DEFAULT_CPUCT_AT_ROOT,
            cpuct_base: DEFAULT_CPUCT_BASE,
            cpuct_factor: DEFAULT_CPUCT_FACTOR,
            cpuct_base_at_root: DEFAULT_CPUCT_BASE,
            cpuct_factor_at_root: DEFAULT_CPUCT_FACTOR,
            max_depth: 0,
            root_dirichlet_alpha: 0.0,
            root_exploration_fraction: 0.0,
            fpu_value: 0.23,
            fpu_value_at_root: 1.0,
            fpu_absolute_at_root: true,
            minimum_kldgain_per_node: 0.0,
            policy_softmax_temp: 1.4,
            draw_score: 0.0,
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
    /// 已证明的结果，按根走棋方视角：胜 1、和 0、负 -1。
    pub solved: Option<i8>,
}

impl AzCandidate {
    pub(crate) fn proof_priority(&self) -> u8 {
        proof_priority(self.solved)
    }
}

fn proof_priority(solved: Option<i8>) -> u8 {
    match solved {
        Some(1) => 2,
        Some(-1) => 0,
        _ => 1,
    }
}

#[derive(Clone, Debug)]
pub struct AzSearchResult {
    pub best_move: Option<Move>,
    pub value_q: f32,
    pub value_cp: i32,
    /// Root win/draw/loss probabilities from the side-to-move perspective.
    pub value_wdl: [f32; 3],
    /// Raw network WDL at the root before search. Used only for TD bootstrapping.
    pub network_value_wdl: [f32; 3],
    pub best_value_wdl: [f32; 3],
    pub simulations: usize,
    pub search_depth_avg: f32,
    pub search_depth_max: usize,
    pub search_depth_limit: usize,
    pub search_depth_cutoffs: usize,
    pub tactical_nodes: usize,
    pub tactical_completed: usize,
    pub tactical_aborted: usize,
    /// 根节点连杀证明的诊断报告；`None` = 本次搜索没有跑证明（`mate_search_plies == 0`）。
    pub mate_search: Option<mate::MateSearchReport>,
    pub candidates: Vec<AzCandidate>,
}

#[derive(Clone, Debug)]
pub struct AzSearchTraceStep {
    pub ply: usize,
    pub mv: Move,
    pub visits: u32,
    pub q: f32,
    pub prior: f32,
    pub gives_check: bool,
    pub child_expanded: bool,
    pub child_value: f32,
    pub child_value_wdl: [f32; 3],
    pub child_fen: String,
}

#[derive(Clone, Debug)]
pub struct AzSearchControl {
    stop: Arc<AtomicBool>,
    deadline: Option<Instant>,
}

impl AzSearchControl {
    pub fn new(stop: Arc<AtomicBool>, deadline: Option<Instant>) -> Self {
        Self { stop, deadline }
    }

    fn should_stop(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
            || self
                .deadline
                .is_some_and(|deadline| Instant::now() >= deadline)
    }
}

pub fn alphazero_search_with_rules(
    position: &Position,
    rule_history: Option<Vec<RuleHistoryEntry>>,
    root_moves: Option<Vec<Move>>,
    model: &AzNnue,
    limits: AzSearchLimits,
) -> AzSearchResult {
    alphazero_search_with_rules_controlled(position, rule_history, root_moves, model, limits, None)
}

pub fn alphazero_search_trace_with_rules(
    position: &Position,
    rule_history: Option<Vec<RuleHistoryEntry>>,
    root_moves: Option<Vec<Move>>,
    model: &AzNnue,
    limits: AzSearchLimits,
    trace_move: Move,
) -> (AzSearchResult, Vec<AzSearchTraceStep>) {
    let mut tree = AzTree::new(
        position.clone(),
        rule_history.unwrap_or_else(|| position.initial_rule_history()),
        root_moves,
        model,
        limits,
    );
    let root = tree.root;
    tree.expand(root);
    tree.prove_root_mate(position, model);
    let mut stopper = KldGainStopper::default();
    let mut used = 0;
    for _ in 0..limits.simulations {
        tree.simulate(root, 0);
        used += 1;
        if tree.nodes[root].solved.is_some()
            || stopper.should_stop(&tree, limits.minimum_kldgain_per_node)
        {
            break;
        }
    }
    let result = tree.search_result(used);
    let trace = tree.trace_root_move(trace_move);
    (result, trace)
}

pub fn alphazero_search_with_rules_controlled(
    position: &Position,
    rule_history: Option<Vec<RuleHistoryEntry>>,
    root_moves: Option<Vec<Move>>,
    model: &AzNnue,
    limits: AzSearchLimits,
    control: Option<&AzSearchControl>,
) -> AzSearchResult {
    alphazero_search_with_rules_controlled_with_progress(
        position,
        rule_history,
        root_moves,
        model,
        limits,
        control,
        None,
    )
}

pub fn alphazero_search_with_rules_controlled_with_progress(
    position: &Position,
    rule_history: Option<Vec<RuleHistoryEntry>>,
    root_moves: Option<Vec<Move>>,
    model: &AzNnue,
    limits: AzSearchLimits,
    control: Option<&AzSearchControl>,
    progress: Option<&mut dyn FnMut(&AzSearchResult)>,
) -> AzSearchResult {
    alphazero_search_with_rules_controlled_with_progress_root_mode(
        position,
        rule_history,
        root_moves,
        model,
        limits,
        control,
        progress,
        true,
    )
}

/// Search a position supplied by an external game controller. The controller owns
/// root adjudication; rule outcomes are still applied below root and rule-forbidden
/// moves can still be excluded from `root_moves` by the caller.
pub fn alphazero_search_external_root_controlled_with_progress(
    position: &Position,
    rule_history: Option<Vec<RuleHistoryEntry>>,
    root_moves: Option<Vec<Move>>,
    model: &AzNnue,
    limits: AzSearchLimits,
    control: Option<&AzSearchControl>,
    progress: Option<&mut dyn FnMut(&AzSearchResult)>,
) -> AzSearchResult {
    alphazero_search_with_rules_controlled_with_progress_root_mode(
        position,
        rule_history,
        root_moves,
        model,
        limits,
        control,
        progress,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn alphazero_search_with_rules_controlled_with_progress_root_mode(
    position: &Position,
    rule_history: Option<Vec<RuleHistoryEntry>>,
    root_moves: Option<Vec<Move>>,
    model: &AzNnue,
    limits: AzSearchLimits,
    control: Option<&AzSearchControl>,
    mut progress: Option<&mut dyn FnMut(&AzSearchResult)>,
    adjudicate_root_rules: bool,
) -> AzSearchResult {
    crate::scope_profile!("az.alphazero_search");
    let mut tree = AzTree::new(
        position.clone(),
        rule_history.unwrap_or_else(|| position.initial_rule_history()),
        root_moves,
        model,
        limits,
    );
    tree.adjudicate_root_rules = adjudicate_root_rules;
    tree.search_control = control.cloned();
    let root = tree.root;
    {
        crate::scope_profile!("az.search.root_expand");
        tree.expand(root);
    }
    if tree.nodes[root].children_len == 0 {
        let value_q = wdl_utility(tree.nodes[root].value_wdl, tree.draw_score);
        return AzSearchResult {
            best_move: None,
            value_q,
            value_cp: cp_from_q(value_q),
            value_wdl: tree.nodes[root].value_wdl,
            network_value_wdl: tree.nodes[root].value_wdl,
            best_value_wdl: tree.nodes[root].value_wdl,
            simulations: 0,
            search_depth_avg: 0.0,
            search_depth_max: 0,
            search_depth_limit: tree.max_depth,
            search_depth_cutoffs: 0,
            tactical_nodes: 0,
            tactical_completed: 0,
            tactical_aborted: 0,
            mate_search: None,
            candidates: Vec::new(),
        };
    }

    let mut used = 0usize;
    tree.prove_root_mate(position, model);
    let mut kld_stopper = KldGainStopper::default();
    let mut last_progress = Instant::now();
    {
        crate::scope_profile!("az.search.simulations");
        for _ in 0..limits.simulations {
            if control.is_some_and(AzSearchControl::should_stop) {
                break;
            }
            tree.simulate(root, 0);
            used += 1;
            if tree.nodes[root].solved.is_some()
                || kld_stopper.should_stop(&tree, limits.minimum_kldgain_per_node)
            {
                break;
            }
            if used % SEARCH_PROGRESS_POLL_SIMULATIONS == 0
                && progress.is_some()
                && last_progress.elapsed() >= SEARCH_PROGRESS_INTERVAL
            {
                let snapshot = tree.search_result(used);
                if let Some(callback) = progress.as_deref_mut() {
                    callback(&snapshot);
                }
                last_progress = Instant::now();
            }
        }
    }
    tree.search_result(used)
}

pub fn alphazero_search(
    position: &Position,
    model: &AzNnue,
    limits: AzSearchLimits,
) -> AzSearchResult {
    alphazero_search_with_rules(position, None, None, model, limits)
}

pub(super) struct AzSearchWorkspace {
    nodes: Vec<AzNode>,
    children: Vec<AzChild>,
    accumulator_arena: Vec<f32>,
    root_raw_priors: Vec<f32>,
    eval_scratch: Option<AzEvalScratch>,
    rule_history_scratch: Vec<RuleHistoryEntry>,
}

impl AzSearchWorkspace {
    pub(super) fn new(model: &AzNnue) -> Self {
        Self {
            nodes: Vec::new(),
            children: Vec::new(),
            accumulator_arena: Vec::new(),
            root_raw_priors: Vec::new(),
            eval_scratch: Some(AzEvalScratch::new(model.arch)),
            rule_history_scratch: Vec::new(),
        }
    }
}

pub(super) fn alphazero_search_with_rules_reusing(
    position: &Position,
    rule_history: &[RuleHistoryEntry],
    root_moves: Vec<Move>,
    model: &AzNnue,
    limits: AzSearchLimits,
    workspace: &mut AzSearchWorkspace,
) -> AzSearchResult {
    crate::scope_profile!("az.alphazero_search");
    let mut tree = AzTree::new_reusing(
        position.clone(),
        rule_history,
        Some(root_moves),
        model,
        limits,
        workspace,
    );
    let root = tree.root;
    {
        crate::scope_profile!("az.search.root_expand");
        tree.expand(root);
    }
    tree.prove_root_mate(position, model);
    let used = if tree.nodes[root].children_len == 0 {
        0
    } else {
        crate::scope_profile!("az.search.simulations");
        let mut used = 0;
        let mut kld_stopper = KldGainStopper::default();
        for _ in 0..limits.simulations {
            tree.simulate(root, 0);
            used += 1;
            if tree.nodes[root].solved.is_some()
                || kld_stopper.should_stop(&tree, limits.minimum_kldgain_per_node)
            {
                break;
            }
        }
        used
    };
    let result = tree.search_result(used);
    tree.recycle_into(workspace);
    result
}

#[derive(Default)]
struct KldGainStopper {
    previous: Vec<u32>,
    total: u32,
}

impl KldGainStopper {
    fn should_stop(&mut self, tree: &AzTree<'_>, threshold: f32) -> bool {
        if threshold <= 0.0 {
            return false;
        }
        let total = tree.nodes[tree.root].visits;
        if total < self.total.saturating_add(200) {
            return false;
        }
        let children = tree.node_children(tree.root);
        let mut gain = 0.0f64;
        if self.total > 0 {
            for (previous, child) in self.previous.iter().zip(children) {
                if *previous > 0 {
                    let old_p = *previous as f64 / self.total as f64;
                    let new_p = child.visits as f64 / total as f64;
                    gain += old_p * (old_p / new_p).ln();
                }
            }
            if gain / ((total - self.total) as f64) < threshold as f64 {
                return true;
            }
        }
        self.previous.clear();
        self.previous
            .extend(children.iter().map(|child| child.visits));
        self.total = total;
        false
    }
}

pub fn cp_from_q(q: f32) -> i32 {
    (q.clamp(-1.0, 1.0) * 1000.0).round() as i32
}

struct AzTree<'a> {
    nodes: Vec<AzNode>,
    children: Vec<AzChild>,
    accumulator_arena: Vec<f32>,
    root_policy_accumulators: [[f32; POLICY_ACCUMULATOR_RANK]; 2],
    model: &'a AzNnue,
    root_moves: Option<Vec<Move>>,
    root_raw_priors: Vec<f32>,
    root: usize,
    adjudicate_root_rules: bool,
    cpuct: f32,
    cpuct_at_root: f32,
    cpuct_base: f32,
    cpuct_factor: f32,
    cpuct_base_at_root: f32,
    cpuct_factor_at_root: f32,
    root_dirichlet_alpha: f32,
    root_exploration_fraction: f32,
    root_noise_seed: u64,
    fpu_value: f32,
    fpu_value_at_root: f32,
    fpu_absolute_at_root: bool,
    policy_softmax_temp: f32,
    draw_score: f32,
    value_scale: f32,
    max_depth: usize,
    search_depth_sum: usize,
    search_depth_count: usize,
    search_depth_max: usize,
    search_depth_cutoffs: usize,
    eval_scratch: AzEvalScratch,
    rule_history_scratch: Vec<RuleHistoryEntry>,
    /// 最近一次根节点连杀证明的报告（只在 `prove_root_mate` 里写）。
    last_mate_search: Option<mate::MateSearchReport>,
    tactical_nodes: usize,
    tactical_completed: usize,
    tactical_aborted: usize,
    search_control: Option<AzSearchControl>,
}

#[derive(Clone)]
struct AzNode {
    position: Position,
    accumulator_offset: u32,
    policy_accumulator: [f32; POLICY_ACCUMULATOR_RANK],
    parent: u32,
    incoming_move: Option<Move>,
    tactical_reply: Option<Move>,
    tactical_pending: bool,
    rule_entry: Option<RuleHistoryEntry>,
    children_offset: u32,
    children_len: u16,
    visits: u32,
    value_wdl_sum: [f32; 3],
    value: f32,
    value_wdl: [f32; 3],
    expanded: bool,
    // 只由规则终局与完整子树证明产生，始终是当前走棋方视角。
    solved: Option<i8>,
    // Px0 sticky-endgames：未完全证明时也保留 can't-win/can't-lose 边界。
    bounds: (i8, i8),
}

#[derive(Clone)]
struct AzChild {
    mv: Move,
    prior: f32,
    visits: u32,
    value_wdl_sum: [f32; 3],
    child: u32,
}

impl AzChild {
    fn child_node(&self) -> Option<usize> {
        (self.child != NO_CHILD).then_some(self.child as usize)
    }

    fn set_child_node(&mut self, child: usize) {
        self.child = u32::try_from(child)
            .ok()
            .filter(|&child| child != NO_CHILD)
            .expect("MCTS node index exceeds compact child range");
    }

    fn q(&self, draw_score: f32) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            wdl_sum_utility(self.value_wdl_sum, self.visits, draw_score)
        }
    }
}

impl<'a> AzTree<'a> {
    fn search_result(&self, simulations: usize) -> AzSearchResult {
        let root_node = &self.nodes[self.root];
        let root_children = self.node_children(self.root);
        let searched_wdl = if let Some(value) = root_node.solved {
            scalar_terminal_wdl(value as f32)
        } else if root_node.visits > 0 {
            root_node
                .value_wdl_sum
                .map(|value| value / root_node.visits as f32)
        } else {
            root_node.value_wdl
        };
        let searched_value = wdl_utility(searched_wdl, self.draw_score);
        let policy = {
            crate::scope_profile!("az.search.root_policy");
            self.root_policy(self.root)
        };
        let mut candidates = root_children
            .iter()
            .zip(policy)
            .enumerate()
            .map(|(index, (child, policy))| AzCandidate {
                mv: child.mv,
                visits: child.visits,
                q: self.child_q(child, self.draw_score),
                raw_prior: self
                    .root_raw_priors
                    .get(index)
                    .copied()
                    .unwrap_or(child.prior),
                prior: child.prior,
                policy,
                solved: self.child_solved(child),
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            right
                .policy
                .total_cmp(&left.policy)
                .then_with(|| right.visits.cmp(&left.visits))
                .then_with(|| right.q.total_cmp(&left.q))
        });
        let best_move = self
            .best_root_child(self.root)
            .map(|child_index| root_children[child_index].mv)
            .or_else(|| candidates.first().map(|candidate| candidate.mv));
        AzSearchResult {
            best_move,
            value_q: searched_value,
            value_cp: cp_from_q(searched_value),
            value_wdl: searched_wdl,
            network_value_wdl: root_node.value_wdl,
            best_value_wdl: self
                .best_root_child(self.root)
                .map_or(searched_wdl, |index| {
                    let child = &root_children[index];
                    if let Some(value) = self.child_solved(child) {
                        return scalar_terminal_wdl(value as f32);
                    }
                    if child.visits == 0 {
                        searched_wdl
                    } else {
                        child.value_wdl_sum.map(|v| v / child.visits as f32)
                    }
                }),
            simulations,
            search_depth_avg: self.search_depth_avg(),
            search_depth_max: self.search_depth_max,
            search_depth_limit: self.max_depth,
            search_depth_cutoffs: self.search_depth_cutoffs,
            tactical_nodes: self.tactical_nodes,
            tactical_completed: self.tactical_completed,
            tactical_aborted: self.tactical_aborted,
            mate_search: self.last_mate_search,
            candidates,
        }
    }

    fn trace_root_move(&self, trace_move: Move) -> Vec<AzSearchTraceStep> {
        let mut node_index = self.root;
        let Some(mut child_index) = self
            .node_children(node_index)
            .iter()
            .position(|child| child.mv == trace_move)
        else {
            return Vec::new();
        };
        let mut trace = Vec::new();
        loop {
            let child = &self.node_children(node_index)[child_index];
            let Some(next_node_index) = child.child_node() else {
                break;
            };
            let next_node = &self.nodes[next_node_index];
            trace.push(AzSearchTraceStep {
                ply: trace.len() + 1,
                mv: child.mv,
                visits: child.visits,
                q: self.child_q(child, self.node_draw_score(node_index)),
                prior: child.prior,
                gives_check: self.nodes[node_index]
                    .position
                    .gives_check_after_move_fast(child.mv),
                child_expanded: next_node.expanded,
                child_value: next_node.value,
                child_value_wdl: next_node.value_wdl,
                child_fen: next_node.position.to_fen(),
            });
            if !next_node.expanded || next_node.children_len == 0 {
                break;
            }
            node_index = next_node_index;
            child_index = self.best_root_child(node_index).unwrap_or(0);
        }
        trace
    }

    fn new(
        position: Position,
        rule_history: Vec<RuleHistoryEntry>,
        root_moves: Option<Vec<Move>>,
        model: &'a AzNnue,
        limits: AzSearchLimits,
    ) -> Self {
        let initial_nodes = limits
            .simulations
            .saturating_add(1)
            .min(INITIAL_TREE_NODE_CAPACITY);
        Self::new_with_buffers(
            position,
            &rule_history,
            root_moves,
            model,
            limits,
            Vec::with_capacity(initial_nodes),
            Vec::with_capacity(initial_nodes.saturating_mul(INITIAL_CHILDREN_PER_NODE_ESTIMATE)),
            Vec::with_capacity(
                initial_nodes
                    .saturating_add(1)
                    .saturating_mul(model.hidden_size),
            ),
            Vec::new(),
            AzEvalScratch::new(model.arch),
            Vec::new(),
        )
    }

    fn new_reusing(
        position: Position,
        rule_history: &[RuleHistoryEntry],
        root_moves: Option<Vec<Move>>,
        model: &'a AzNnue,
        limits: AzSearchLimits,
        workspace: &mut AzSearchWorkspace,
    ) -> Self {
        Self::new_with_buffers(
            position,
            rule_history,
            root_moves,
            model,
            limits,
            std::mem::take(&mut workspace.nodes),
            std::mem::take(&mut workspace.children),
            std::mem::take(&mut workspace.accumulator_arena),
            std::mem::take(&mut workspace.root_raw_priors),
            workspace
                .eval_scratch
                .take()
                .unwrap_or_else(|| AzEvalScratch::new(model.arch)),
            std::mem::take(&mut workspace.rule_history_scratch),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_buffers(
        position: Position,
        rule_history: &[RuleHistoryEntry],
        root_moves: Option<Vec<Move>>,
        model: &'a AzNnue,
        limits: AzSearchLimits,
        mut nodes: Vec<AzNode>,
        mut children: Vec<AzChild>,
        mut accumulator_arena: Vec<f32>,
        mut root_raw_priors: Vec<f32>,
        eval_scratch: AzEvalScratch,
        mut rule_history_scratch: Vec<RuleHistoryEntry>,
    ) -> Self {
        nodes.clear();
        children.clear();
        accumulator_arena.clear();
        root_raw_priors.clear();
        rule_history_scratch.clear();
        rule_history_scratch.extend_from_slice(rule_history);
        let accumulator = AzEvalAccumulator::new(model, &position);
        accumulator_arena.extend_from_slice(&accumulator.into_hidden_sum());
        let root_policy_accumulators = [
            model.policy_accumulator(&position, Color::Red),
            model.policy_accumulator(&position, Color::Black),
        ];
        let root_accumulator_offset = match position.side_to_move() {
            Color::Red => 0,
            Color::Black => model.hidden_size,
        };
        let root_policy_accumulator =
            root_policy_accumulators[color_index(position.side_to_move())];
        nodes.push(AzNode {
            position,
            accumulator_offset: root_accumulator_offset as u32,
            policy_accumulator: root_policy_accumulator,
            parent: NO_CHILD,
            incoming_move: None,
            tactical_reply: None,
            tactical_pending: false,
            rule_entry: None,
            children_offset: 0,
            children_len: 0,
            visits: 0,
            value_wdl_sum: [0.0; 3],
            value: 0.0,
            value_wdl: [0.0, 1.0, 0.0],
            expanded: false,
            solved: None,
            bounds: (-1, 1),
        });
        Self {
            nodes,
            children,
            accumulator_arena,
            root_policy_accumulators,
            model,
            root_moves,
            root_raw_priors,
            root: 0,
            adjudicate_root_rules: true,
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
            cpuct_base: limits.cpuct_base.max(1.0),
            cpuct_factor: limits.cpuct_factor.max(0.0),
            cpuct_base_at_root: if limits.cpuct_base_at_root > 0.0 {
                limits.cpuct_base_at_root
            } else {
                limits.cpuct_base.max(1.0)
            },
            cpuct_factor_at_root: if limits.cpuct_factor_at_root >= 0.0 {
                limits.cpuct_factor_at_root
            } else {
                limits.cpuct_factor.max(0.0)
            },
            root_dirichlet_alpha: limits.root_dirichlet_alpha.max(0.0),
            root_exploration_fraction: limits.root_exploration_fraction.clamp(0.0, 1.0),
            root_noise_seed: limits.seed,
            fpu_value: limits.fpu_value.max(0.0),
            fpu_value_at_root: limits.fpu_value_at_root.max(0.0),
            fpu_absolute_at_root: limits.fpu_absolute_at_root,
            policy_softmax_temp: limits.policy_softmax_temp.max(1.0e-3),
            draw_score: limits.draw_score.clamp(-1.0, 1.0),
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
            eval_scratch,
            rule_history_scratch,
            last_mate_search: None,
            tactical_nodes: 0,
            tactical_completed: 0,
            tactical_aborted: 0,
            search_control: None,
        }
    }

    fn recycle_into(mut self, workspace: &mut AzSearchWorkspace) {
        workspace.nodes = std::mem::take(&mut self.nodes);
        workspace.children = std::mem::take(&mut self.children);
        workspace.accumulator_arena = std::mem::take(&mut self.accumulator_arena);
        workspace.root_raw_priors = std::mem::take(&mut self.root_raw_priors);
        workspace.eval_scratch = Some(std::mem::replace(
            &mut self.eval_scratch,
            AzEvalScratch::empty(),
        ));
        workspace.rule_history_scratch = std::mem::take(&mut self.rule_history_scratch);
    }

    fn node_children(&self, node_index: usize) -> &[AzChild] {
        let node = &self.nodes[node_index];
        let start = node.children_offset as usize;
        &self.children[start..start + node.children_len as usize]
    }

    fn node_children_mut(&mut self, node_index: usize) -> &mut [AzChild] {
        let node = &self.nodes[node_index];
        let start = node.children_offset as usize;
        let len = node.children_len as usize;
        &mut self.children[start..start + len]
    }

    #[cfg(test)]
    fn set_node_children(
        &mut self,
        node_index: usize,
        children: impl IntoIterator<Item = AzChild>,
    ) {
        debug_assert_eq!(self.nodes[node_index].children_len, 0);
        let offset = self.children.len();
        self.children.extend(children);
        let len = self.children.len() - offset;
        self.nodes[node_index].children_offset =
            u32::try_from(offset).expect("MCTS child arena exceeds compact offset range");
        self.nodes[node_index].children_len =
            u16::try_from(len).expect("MCTS node has too many legal moves");
    }

    fn expand(&mut self, node_index: usize) -> AzEvalOutput {
        crate::scope_profile!("az.search.expand");
        if self.nodes[node_index].expanded {
            return self.node_eval(node_index);
        }
        let terminal = (node_index != self.root || self.adjudicate_root_rules)
            .then(|| {
                crate::scope_profile!("az.search.terminal_value");
                terminal_value(&self.nodes[node_index].position, &self.rule_history_scratch)
            })
            .flatten();
        if let Some(value) = terminal {
            let value_wdl = scalar_terminal_wdl(value);
            self.nodes[node_index].value = value;
            self.nodes[node_index].value_wdl = value_wdl;
            self.nodes[node_index].expanded = true;
            self.nodes[node_index].solved = Some(value as i8);
            return AzEvalOutput { value_wdl, value };
        }

        let (moves, repetition_flags): (Vec<_>, Vec<_>) = {
            crate::scope_profile!("az.search.expand_legal_moves");
            if node_index == self.root {
                if let Some(moves) = self.root_moves.take() {
                    let flags = moves
                        .iter()
                        .map(|&mv| {
                            u8::from(
                                self.nodes[node_index]
                                    .position
                                    .move_repeats_history(&self.rule_history_scratch, mv),
                            )
                        })
                        .collect();
                    (moves, flags)
                } else {
                    self.nodes[node_index]
                        .position
                        .legal_moves_with_rules_and_repetition(&self.rule_history_scratch)
                        .into_iter()
                        .map(|(mv, repeats)| (mv, u8::from(repeats)))
                        .unzip()
                }
            } else {
                self.nodes[node_index]
                    .position
                    .legal_moves_with_rules_and_repetition(&self.rule_history_scratch)
                    .into_iter()
                    .map(|(mv, repeats)| (mv, u8::from(repeats)))
                    .unzip()
            }
        };
        if moves.is_empty() {
            self.nodes[node_index].value = -1.0;
            self.nodes[node_index].value_wdl = [0.0, 0.0, 1.0];
            self.nodes[node_index].expanded = true;
            self.nodes[node_index].solved = Some(-1);
            return AzEvalOutput {
                value_wdl: [0.0, 0.0, 1.0],
                value: -1.0,
            };
        }

        let mut eval = {
            crate::scope_profile!("az.search.nn_eval");
            let accumulator_start = self.nodes[node_index].accumulator_offset as usize;
            let accumulator_end = accumulator_start + self.model.hidden_size;
            self.model.evaluate_incremental_with_scratch_output(
                &self.nodes[node_index].position,
                &self.accumulator_arena[accumulator_start..accumulator_end],
                &self.nodes[node_index].policy_accumulator,
                &moves,
                &repetition_flags,
                &rule_context_features(
                    &self.nodes[node_index].position,
                    &self.rule_history_scratch,
                ),
                &mut self.eval_scratch,
            )
        };
        eval.value_wdl = scale_wdl_value(eval.value_wdl, self.value_scale);
        eval.value *= self.value_scale;
        if let Some(tactical_eval) = self.tactical_value(node_index, &moves) {
            eval = tactical_eval;
        }
        if node_index == self.root {
            softmax_into(
                &self.eval_scratch.logits[..moves.len()],
                1.0,
                &mut self.root_raw_priors,
            );
        }
        let priors = {
            crate::scope_profile!("az.search.softmax");
            softmax_into(
                &self.eval_scratch.logits[..moves.len()],
                self.policy_softmax_temp,
                &mut self.eval_scratch.priors,
            )
        };
        if node_index == self.root
            && self.root_dirichlet_alpha > 0.0
            && self.root_exploration_fraction > 0.0
        {
            let alpha = self.root_dirichlet_alpha;
            apply_root_dirichlet_noise(
                priors,
                alpha,
                self.root_exploration_fraction,
                self.root_noise_seed,
            );
        }
        {
            crate::scope_profile!("az.search.children_build");
            let priors = &mut self.eval_scratch.priors;
            let offset = self.children.len();
            self.children
                .extend(
                    moves
                        .into_iter()
                        .zip(priors.drain(..))
                        .map(|(mv, prior)| AzChild {
                            mv,
                            prior,
                            visits: 0,
                            value_wdl_sum: [0.0; 3],
                            child: NO_CHILD,
                        }),
                );
            let len = self.children.len() - offset;
            self.nodes[node_index].children_offset =
                u32::try_from(offset).expect("MCTS child arena exceeds compact offset range");
            self.nodes[node_index].children_len =
                u16::try_from(len).expect("MCTS node has too many legal moves");
        }
        self.nodes[node_index].value = eval.value;
        self.nodes[node_index].value_wdl = eval.value_wdl;
        self.nodes[node_index].expanded = true;
        if let Some(index) = self.immediate_mate_child(node_index) {
            let mut depth = 1;
            let mut parent = node_index;
            while parent != self.root {
                depth += 1;
                parent = self.nodes[parent].parent as usize;
            }
            // 建立终局子节点并传播证明；父节点的本次访问由 simulate 统一计数。
            let visits = self.nodes[node_index].visits;
            let sum = self.nodes[node_index].value_wdl_sum;
            let proven = self.simulate_child(node_index, index, depth);
            self.nodes[node_index].visits = visits;
            self.nodes[node_index].value_wdl_sum = sum;
            return proven;
        }
        eval
    }

    fn immediate_mate_child(&self, node_index: usize) -> Option<usize> {
        let position = &self.nodes[node_index].position;
        for (index, child) in self.node_children(node_index).iter().enumerate() {
            // expand 刚评估过同序走法，复用 policy 已计算的将军标记。
            if self.eval_scratch.policy_gives_check[index] == 0.0 {
                continue;
            }
            let mut reply = position.clone();
            reply.make_move(child.mv);
            if !reply.legal_moves().is_empty() {
                continue;
            }
            let mut history = self.rule_history_scratch.clone();
            history.push(position.rule_history_entry_after_move(child.mv));
            match reply.rule_outcome_with_history(&history) {
                None => return Some(index),
                Some(RuleOutcome::Win(side)) if side == position.side_to_move() => {
                    return Some(index);
                }
                _ => {}
            }
        }
        None
    }

    fn simulate(&mut self, node_index: usize, depth: usize) -> AzEvalOutput {
        crate::scope_profile!("az.search.simulate");
        if self.nodes[node_index].solved.is_some() {
            let eval = self.node_eval(node_index);
            self.add_node_visit(node_index, eval);
            self.record_leaf_depth(depth, false);
            return eval;
        }
        if depth >= self.max_depth {
            let eval = self.cutoff_value(node_index);
            self.add_node_visit(node_index, eval);
            self.record_leaf_depth(depth, true);
            return eval;
        }
        if !self.nodes[node_index].expanded {
            let tactical_completed_before = self.tactical_completed;
            let was_in_check = self.nodes[node_index]
                .position
                .in_check(self.nodes[node_index].position.side_to_move());
            let eval = self.expand(node_index);
            // 叶子正被将军时，网络在“尚未应将”的截断点上给值会把
            // 将军错当成终局收益。像 quiescence 搜索一样，至少完整搜索一手
            // 应将；若应将后仍被将军，会递归继续。唯一合法着同理不是
            // 需要策略分配预算的选择。
            if self.nodes[node_index].children_len > 0
                && self.nodes[node_index].solved.is_none()
                && ((was_in_check && self.tactical_completed == tactical_completed_before)
                    || self.nodes[node_index].children_len == 1)
            {
                let child_index = self.select_child(node_index);
                return self.simulate_child(node_index, child_index, depth + 1);
            }
            self.add_node_visit(node_index, eval);
            self.record_leaf_depth(depth, false);
            return eval;
        }
        if self.nodes[node_index].children_len == 0 {
            let eval = self.node_eval(node_index);
            self.add_node_visit(node_index, eval);
            self.record_leaf_depth(depth, false);
            return eval;
        }
        if self.nodes[node_index].tactical_pending && self.nodes[node_index].visits >= 8 {
            self.nodes[node_index].tactical_pending = false;
            let moves: Vec<_> = self
                .node_children(node_index)
                .iter()
                .map(|child| child.mv)
                .collect();
            if let Some(eval) = self.tactical_value(node_index, &moves) {
                self.nodes[node_index].value = eval.value;
                self.nodes[node_index].value_wdl = eval.value_wdl;
                self.add_node_visit(node_index, eval);
                self.record_leaf_depth(depth, false);
                return eval;
            }
        }
        let child_index = {
            crate::scope_profile!("az.search.select_child");
            self.select_child(node_index)
        };
        self.simulate_child(node_index, child_index, depth + 1)
    }

    fn simulate_child(
        &mut self,
        node_index: usize,
        child_index: usize,
        child_depth: usize,
    ) -> AzEvalOutput {
        crate::scope_profile!("az.search.simulate_child");
        let child_node =
            if let Some(child_node) = self.node_children(node_index)[child_index].child_node() {
                child_node
            } else {
                crate::scope_profile!("az.search.create_child");
                let mv = self.node_children(node_index)[child_index].mv;
                let mut child_position = self.nodes[node_index].position.clone();
                let moved = child_position.piece_at(mv.from as usize).unwrap();
                let captured = child_position.piece_at(mv.to as usize);
                let mover = child_position.side_to_move();
                {
                    crate::scope_profile!("az.search.child_make_move");
                    child_position.make_move(mv);
                }
                let perspective = child_position.side_to_move();
                let mut child_policy_accumulator = if node_index == self.root {
                    self.root_policy_accumulators[color_index(perspective)]
                } else {
                    let grandparent = self.nodes[node_index].parent as usize;
                    self.nodes[grandparent].policy_accumulator
                };
                let child_accumulator_offset = self.accumulator_arena.len();
                let base_offset = if node_index == self.root {
                    match perspective {
                        Color::Red => 0,
                        Color::Black => self.model.hidden_size,
                    }
                } else {
                    let grandparent = self.nodes[node_index].parent as usize;
                    self.nodes[grandparent].accumulator_offset as usize
                };
                self.accumulator_arena
                    .extend_from_within(base_offset..base_offset + self.model.hidden_size);
                let accumulator = &mut self.accumulator_arena
                    [child_accumulator_offset..child_accumulator_offset + self.model.hidden_size];
                if node_index != self.root {
                    let grandparent = self.nodes[node_index].parent as usize;
                    let parent_move = self.nodes[node_index]
                        .incoming_move
                        .expect("non-root node must have an incoming move");
                    let parent_moved = self.nodes[grandparent]
                        .position
                        .piece_at(parent_move.from as usize)
                        .expect("incoming move must start on an occupied square");
                    let parent_captured = self.nodes[grandparent]
                        .position
                        .piece_at(parent_move.to as usize);
                    AzEvalAccumulator::apply_transition_for_perspective(
                        self.model,
                        &self.nodes[grandparent].position,
                        &self.nodes[node_index].position,
                        parent_move,
                        parent_moved,
                        parent_captured,
                        perspective,
                        accumulator,
                    );
                    self.model.apply_policy_transition(
                        &self.nodes[grandparent].position,
                        &self.nodes[node_index].position,
                        parent_move,
                        parent_moved,
                        parent_captured,
                        perspective,
                        &mut child_policy_accumulator,
                    );
                }
                AzEvalAccumulator::apply_transition_for_perspective(
                    self.model,
                    &self.nodes[node_index].position,
                    &child_position,
                    mv,
                    moved,
                    captured,
                    perspective,
                    accumulator,
                );
                self.model.apply_policy_transition(
                    &self.nodes[node_index].position,
                    &child_position,
                    mv,
                    moved,
                    captured,
                    perspective,
                    &mut child_policy_accumulator,
                );
                let child_rule_entry =
                    child_position.rule_history_entry_after_moved(mover, mv, captured);
                let child_node = self.nodes.len();
                self.nodes.push(AzNode {
                    position: child_position,
                    accumulator_offset: u32::try_from(child_accumulator_offset)
                        .expect("MCTS accumulator arena exceeds compact offset range"),
                    policy_accumulator: child_policy_accumulator,
                    parent: u32::try_from(node_index)
                        .expect("MCTS node index exceeds compact parent range"),
                    incoming_move: Some(mv),
                    tactical_reply: None,
                    tactical_pending: false,
                    rule_entry: Some(child_rule_entry),
                    children_offset: 0,
                    children_len: 0,
                    visits: 0,
                    value_wdl_sum: [0.0; 3],
                    value: 0.0,
                    value_wdl: [0.0, 1.0, 0.0],
                    expanded: false,
                    solved: None,
                    bounds: (-1, 1),
                });
                self.node_children_mut(node_index)[child_index].set_child_node(child_node);
                child_node
            };
        let history_len = self.rule_history_scratch.len();
        if let Some(entry) = self.nodes[child_node].rule_entry {
            self.rule_history_scratch.push(entry);
        }
        let child_eval = self.simulate(child_node, child_depth);
        self.rule_history_scratch.truncate(history_len);
        let eval = AzEvalOutput {
            value_wdl: flip_wdl(child_eval.value_wdl),
            value: -child_eval.value,
        };
        let child = &mut self.node_children_mut(node_index)[child_index];
        child.visits += 1;
        add_wdl(&mut child.value_wdl_sum, eval.value_wdl);
        self.update_solved(node_index);
        let eval = if self.nodes[node_index].solved.is_some() {
            self.node_eval(node_index)
        } else {
            eval
        };
        self.add_node_visit(node_index, eval);
        eval
    }

    fn cutoff_value(&mut self, node_index: usize) -> AzEvalOutput {
        crate::scope_profile!("az.search.cutoff_value");
        if self.nodes[node_index].expanded {
            return self.node_eval(node_index);
        }
        let terminal = {
            crate::scope_profile!("az.search.terminal_value");
            terminal_value(&self.nodes[node_index].position, &self.rule_history_scratch)
        };
        if let Some(value) = terminal {
            let value_wdl = scalar_terminal_wdl(value);
            self.nodes[node_index].value = value;
            self.nodes[node_index].value_wdl = value_wdl;
            self.nodes[node_index].solved = Some(value as i8);
            return AzEvalOutput { value_wdl, value };
        }
        let (moves, repetition_flags): (Vec<_>, Vec<_>) = {
            crate::scope_profile!("az.search.expand_legal_moves");
            self.nodes[node_index]
                .position
                .legal_moves_with_rules_and_repetition(&self.rule_history_scratch)
                .into_iter()
                .map(|(mv, repeats)| (mv, u8::from(repeats)))
                .unzip()
        };
        if moves.is_empty() {
            self.nodes[node_index].value = -1.0;
            self.nodes[node_index].value_wdl = [0.0, 0.0, 1.0];
            self.nodes[node_index].solved = Some(-1);
            return AzEvalOutput {
                value_wdl: [0.0, 0.0, 1.0],
                value: -1.0,
            };
        }
        let mut eval = {
            crate::scope_profile!("az.search.nn_eval");
            let accumulator_start = self.nodes[node_index].accumulator_offset as usize;
            let accumulator_end = accumulator_start + self.model.hidden_size;
            self.model.evaluate_incremental_with_scratch_output(
                &self.nodes[node_index].position,
                &self.accumulator_arena[accumulator_start..accumulator_end],
                &self.nodes[node_index].policy_accumulator,
                &moves,
                &repetition_flags,
                &rule_context_features(
                    &self.nodes[node_index].position,
                    &self.rule_history_scratch,
                ),
                &mut self.eval_scratch,
            )
        };
        eval.value_wdl = scale_wdl_value(eval.value_wdl, self.value_scale);
        eval.value *= self.value_scale;
        if let Some(tactical_eval) = self.tactical_value(node_index, &moves) {
            eval = tactical_eval;
        }
        self.nodes[node_index].value = eval.value;
        self.nodes[node_index].value_wdl = eval.value_wdl;
        eval
    }

    fn tactical_value(&mut self, node_index: usize, moves: &[Move]) -> Option<AzEvalOutput> {
        if node_index == self.root
            || self.model.tactical_search_nodes == 0
            || self.model.tactical_search_plies == 0
        {
            return None;
        }
        let checked = self.nodes[node_index]
            .position
            .in_check(self.nodes[node_index].position.side_to_move());
        let root_threat = self.nodes[node_index].parent as usize == self.root
            && self.model.tactical_quiet_plies > 0
            && moves.iter().any(|&mv| {
                self.nodes[node_index]
                    .position
                    .gives_check_after_move_fast(mv)
            })
            && {
                let p = &self.nodes[node_index].position;
                let flags: Vec<_> = moves
                    .iter()
                    .map(|&mv| f32::from(p.gives_check_after_move_fast(mv)))
                    .collect();
                check_context_features(p, moves, &flags, p.attacked_squares_masks())[7] > 0.0
            };
        let mut quiet = if root_threat {
            self.model.tactical_quiet_plies.min(2)
        } else {
            0
        };
        if quiet > 0 && self.nodes[node_index].visits < 8 {
            let children = self.node_children(self.root);
            let index = children
                .iter()
                .position(|child| child.child_node() == Some(node_index))
                .unwrap();
            let prior = self.root_raw_priors[index];
            let rank = self
                .root_raw_priors
                .iter()
                .enumerate()
                .filter(|&(other, p)| *p > prior || (*p == prior && other < index))
                .count();
            if rank >= 4 {
                self.nodes[node_index].tactical_pending = true;
                quiet = 0;
            }
        }
        let p = &self.nodes[node_index].position;
        if quiet == 0
            && !checked
            && self.nodes[node_index]
                .rule_entry
                .is_none_or(|entry| entry.captured.is_none())
        {
            return None;
        }
        if quiet == 0
            && !p.in_check(p.side_to_move())
            && !moves
                .iter()
                .any(|&mv| p.is_capture(mv) || p.gives_check_after_move_fast(mv))
        {
            return None;
        }
        let mut probe = tactical::TacticalProbe::new(
            self.model,
            if quiet > 0 {
                self.model.tactical_search_nodes
            } else {
                self.model.tactical_search_nodes.min(128)
            },
            self.value_scale,
        );
        let draw_score = if p.side_to_move() == self.nodes[self.root].position.side_to_move() {
            self.draw_score
        } else {
            -self.draw_score
        };
        probe.control = self.search_control.as_ref();
        let plies = self.model.tactical_search_plies.min(16);
        let mut result = None;
        let mut reply = None;
        // 吃子/应将先完成，再补安静反击。后续撞预算只丢弃未完成的那次校验。
        for quiet_depth in 0..=quiet {
            match probe.search(
                p,
                &mut self.rule_history_scratch,
                quiet_depth,
                plies,
                0,
                -2.0,
                2.0,
                draw_score,
            ) {
                Ok(eval) => {
                    result = Some(eval);
                    reply = probe.best_reply;
                }
                Err(()) => {
                    self.tactical_aborted += 1;
                    break;
                }
            }
        }
        // 完整主结果已有后才尝试主动将军，避免将军链耗尽预算丢掉吃子搜索结果。
        if result.is_some() && probe.nodes < probe.max_nodes {
            match probe.search(
                p,
                &mut self.rule_history_scratch,
                quiet,
                plies,
                2,
                -2.0,
                2.0,
                draw_score,
            ) {
                Ok(eval) => {
                    result = Some(eval);
                    reply = probe.best_reply;
                }
                Err(()) => self.tactical_aborted += 1,
            }
        }
        self.tactical_nodes += probe.nodes;
        if result.is_some() {
            self.tactical_completed += 1;
            self.nodes[node_index].tactical_reply = reply;
        }
        result
    }

    fn node_eval(&self, node_index: usize) -> AzEvalOutput {
        if let Some(value) = self.nodes[node_index].solved {
            return AzEvalOutput {
                value_wdl: scalar_terminal_wdl(value as f32),
                value: value as f32,
            };
        }
        AzEvalOutput {
            value_wdl: self.nodes[node_index].value_wdl,
            value: self.nodes[node_index].value,
        }
    }

    fn add_node_visit(&mut self, node_index: usize, eval: AzEvalOutput) {
        self.nodes[node_index].visits += 1;
        add_wdl(&mut self.nodes[node_index].value_wdl_sum, eval.value_wdl);
    }

    fn node_draw_score(&self, node_index: usize) -> f32 {
        if self.nodes[node_index].position.side_to_move()
            == self.nodes[self.root].position.side_to_move()
        {
            self.draw_score
        } else {
            -self.draw_score
        }
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
        let children = self.node_children(node_index);
        let parent_visits_sqrt = (node.visits.max(1) as f32).sqrt();
        let is_root = node_index == self.root;
        let draw_score = self.node_draw_score(node_index);
        let fpu_reduction = if is_root {
            self.fpu_value_at_root
        } else {
            self.fpu_value
        };
        let fpu_value = if is_root && self.fpu_absolute_at_root {
            self.fpu_value_at_root
        } else {
            alphazero_fpu_value_reduction(node, children, fpu_reduction, draw_score)
        };
        let cpuct = self.compute_cpuct(node.visits, is_root);
        // 已完成战术搜索的反击至少实际访问两次，避免低策略先验把它埋掉。
        if let Some(reply) = node.tactical_reply {
            if let Some((index, _)) = children.iter().enumerate().find(|(_, child)| {
                child.mv == reply && child.visits < 2 && self.child_solved(child).is_none()
            }) {
                return index;
            }
        }
        // 一趟同时取"证明优先级 → PUCT 分数 → 先验"的字典序最大值。
        // 原实现先扫一遍找已证明胜的子节点、再扫一遍求最高优先级、最后在最高
        // 优先级里比分数，每个子节点要查三次 `child_solved`（每次都随机访问
        // `self.nodes`）。`proof_priority` 把 `Some(1)` 映到最高的 2，所以
        // "第一个已证明胜的子节点"就是"第一个 priority == 2"，顺序语义不变。
        let mut best: Option<(usize, u8, f32, f32)> = None;
        for (index, child) in children.iter().enumerate() {
            let solved = self.child_solved(child);
            let priority = proof_priority(solved);
            if priority == 2 {
                return index;
            }
            let score = self.child_score_with_solved(
                child,
                solved,
                draw_score,
                fpu_value,
                parent_visits_sqrt,
                cpuct,
            );
            let replace = best.is_none_or(|(_, best_priority, best_prior, best_score)| {
                priority > best_priority
                    || (priority == best_priority
                        && (score.total_cmp(&best_score).is_gt()
                            || (score.total_cmp(&best_score).is_eq()
                                && child.prior.total_cmp(&best_prior).is_gt())))
            });
            if replace {
                best = Some((index, priority, child.prior, score));
            }
        }
        best.map(|(index, _, _, _)| index).unwrap_or(0)
    }

    fn best_root_child(&self, node_index: usize) -> Option<usize> {
        let draw_score = self.node_draw_score(node_index);
        let children = self.node_children(node_index);
        children
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                self.child_priority(left)
                    .cmp(&self.child_priority(right))
                    .then_with(|| left.visits.cmp(&right.visits))
                    .then_with(|| {
                        self.child_q(left, draw_score)
                            .total_cmp(&self.child_q(right, draw_score))
                    })
                    .then_with(|| left.prior.total_cmp(&right.prior))
            })
            .map(|(index, _)| index)
    }

    fn compute_cpuct(&self, visits: u32, is_root: bool) -> f32 {
        let init = if is_root {
            self.cpuct_at_root
        } else {
            self.cpuct
        };
        let factor = if is_root {
            self.cpuct_factor_at_root
        } else {
            self.cpuct_factor
        };
        if factor <= 0.0 {
            return init;
        }
        let base = if is_root {
            self.cpuct_base_at_root
        } else {
            self.cpuct_base
        }
        .max(1.0);
        init + factor * ((visits as f32 + base) / base).ln()
    }

    /// 子节点的 PUCT 分数；`solved` 由调用方传入，避免重复随机访问 `self.nodes`。
    fn child_score_with_solved(
        &self,
        child: &AzChild,
        solved: Option<i8>,
        draw_score: f32,
        fpu_value: f32,
        parent_visits_sqrt: f32,
        cpuct: f32,
    ) -> f32 {
        let q = if child.visits > 0 {
            solved.map_or_else(
                || child.q(draw_score),
                |value| wdl_utility(scalar_terminal_wdl(value as f32), draw_score),
            )
        } else {
            fpu_value
        };
        let u = cpuct * child.prior * parent_visits_sqrt / (1.0 + child.visits as f32);
        q + u
    }

    fn root_policy(&self, node_index: usize) -> Vec<f32> {
        let children = self.node_children(node_index);
        let priority = children
            .iter()
            .map(|child| self.child_priority(child))
            .max()
            .unwrap_or(0);
        let eligible = |child: &&AzChild| self.child_priority(child) == priority;
        let total_visits = children
            .iter()
            .filter(eligible)
            .map(|child| child.visits as f32)
            .sum::<f32>()
            .max(1.0);
        if children
            .iter()
            .filter(eligible)
            .any(|child| child.visits > 0)
        {
            return children
                .iter()
                .map(|child| {
                    if self.child_priority(child) == priority {
                        child.visits as f32 / total_visits
                    } else {
                        0.0
                    }
                })
                .collect();
        }

        let total_prior = children
            .iter()
            .filter(eligible)
            .map(|child| child.prior)
            .sum::<f32>()
            .max(1e-12);
        children
            .iter()
            .map(|child| {
                if self.child_priority(child) == priority {
                    child.prior / total_prior
                } else {
                    0.0
                }
            })
            .collect()
    }

    fn child_solved(&self, child: &AzChild) -> Option<i8> {
        child
            .child_node()
            .and_then(|index| self.nodes[index].solved)
            .map(|value| -value)
    }

    fn child_priority(&self, child: &AzChild) -> u8 {
        proof_priority(self.child_solved(child))
    }

    fn child_q(&self, child: &AzChild, draw_score: f32) -> f32 {
        self.child_solved(child).map_or_else(
            || child.q(draw_score),
            |value| wdl_utility(scalar_terminal_wdl(value as f32), draw_score),
        )
    }

    /// 根节点 check-only 连杀证明：证明出来就把"杀着之后的那个子局面"标成对方必败。
    ///
    /// 复用搜索既有的 `solved` 传播，所以不需要任何新的判定路径：`child_solved` 会对子节点
    /// 的 solved 取反（这里写 -1 = 对方必败），于是 `root_policy` 把策略目标压到杀着上、
    /// `proven_root_value` 把价值目标设成必胜。
    ///
    /// 只在 `AzNnue::mate_search_plies > 0` 时运行。两个搜索入口（复用 / 不复用 workspace）
    /// 共用这一个实现，避免两处漂移。
    fn prove_root_mate(&mut self, position: &Position, model: &AzNnue) {
        if model.mate_search_plies == 0 || self.node_children(self.root).is_empty() {
            return;
        }
        crate::scope_profile!("az.search.root_mate");
        let outcome = mate::search_root_mate_profiled(
            position,
            &self.rule_history_scratch,
            mate::MateSearchLimits {
                max_plies: model.mate_search_plies,
                max_nodes: model.mate_search_nodes,
            },
        );
        // 报告写进搜索树，UCI 层再取出来打印：这样"证出来了 / 没杀 / 预算撞墙"
        // 三种情况在 GUI 里可区分，而不是都表现为"什么都没发生"。
        self.last_mate_search = Some(mate::MateSearchReport {
            max_plies: model.mate_search_plies,
            nodes: outcome.nodes,
            budget_exhausted: outcome.budget_exhausted,
            solution: outcome.solution,
        });
        let Some(solution) = outcome.solution else {
            return;
        };
        let Some(index) = self
            .node_children(self.root)
            .iter()
            .position(|child| child.mv == solution.mv)
        else {
            return;
        };
        // 建立杀着对应的子节点（含一次网络评估）。
        self.simulate_child(self.root, index, 1);
        let Some(child_node) = self.node_children(self.root)[index].child_node() else {
            return;
        };
        self.set_proven(child_node, -1);
        self.update_solved(self.root);
    }

    fn update_solved(&mut self, node_index: usize) {
        if self.nodes[node_index].solved.is_some() {
            return;
        }
        let children = self.node_children(node_index);
        if children.is_empty() {
            return;
        }
        let mut lower = -1;
        let mut upper = -1;
        for child in children {
            let (lo, hi) = child.child_node().map_or((-1, 1), |index| {
                let node = &self.nodes[index];
                let (lo, hi) = node.solved.map_or(node.bounds, |value| (value, value));
                (-hi, -lo)
            });
            lower = lower.max(lo);
            upper = upper.max(hi);
        }
        self.nodes[node_index].bounds = (lower, upper);
        if lower == upper {
            self.set_proven(node_index, lower);
        }
    }

    fn set_proven(&mut self, node_index: usize, value: i8) {
        if self.nodes[node_index].solved.is_some() {
            return;
        }
        let node = &mut self.nodes[node_index];
        node.solved = Some(value);
        node.bounds = (value, value);
        let exact_sum = scalar_terminal_wdl(value as f32).map(|p| p * node.visits as f32);
        let mut delta = std::array::from_fn(|i| exact_sum[i] - node.value_wdl_sum[i]);
        node.value_wdl_sum = exact_sum;
        // 与 Px0 AdjustForTerminal 相同：纠正此前的访问贡献，而非只替换下一次回传。
        // 本次访问尚未计入该祖先，仍由 simulate_child 正常计数。
        let mut current = node_index;
        while self.nodes[current].parent != NO_CHILD {
            let parent = self.nodes[current].parent as usize;
            delta = flip_wdl(delta);
            let edge = self
                .node_children_mut(parent)
                .iter_mut()
                .find(|edge| edge.child_node() == Some(current))
                .expect("missing incoming search edge");
            add_wdl(&mut edge.value_wdl_sum, delta);
            if self.nodes[parent].solved.is_some() {
                break;
            }
            add_wdl(&mut self.nodes[parent].value_wdl_sum, delta);
            current = parent;
        }
    }
}

fn wdl_utility(wdl: [f32; 3], draw_score: f32) -> f32 {
    (wdl[0] - wdl[2] + draw_score * wdl[1]).clamp(-1.0, 1.0)
}

fn wdl_sum_utility(wdl_sum: [f32; 3], visits: u32, draw_score: f32) -> f32 {
    if visits == 0 {
        return 0.0;
    }
    wdl_utility(wdl_sum.map(|part| part / visits as f32), draw_score)
}

fn alphazero_fpu_value_reduction(
    node: &AzNode,
    children: &[AzChild],
    reduction: f32,
    draw_score: f32,
) -> f32 {
    let parent_q = if node.visits > 0 {
        wdl_sum_utility(node.value_wdl_sum, node.visits, draw_score)
    } else {
        wdl_utility(node.value_wdl, draw_score)
    };
    if reduction <= 0.0 {
        return parent_q;
    }

    let visited_prior = children
        .iter()
        .filter(|child| child.visits > 0)
        .map(|child| child.prior.max(0.0))
        .sum::<f32>()
        .clamp(0.0, 1.0);
    (parent_q - reduction * visited_prior.sqrt()).clamp(-1.0, 1.0)
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

fn softmax_into<'a>(
    logits: &[f32],
    temperature: f32,
    output: &'a mut Vec<f32>,
) -> &'a mut Vec<f32> {
    output.clear();
    if logits.is_empty() {
        return output;
    }
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    output.reserve(logits.len());
    let inverse_temperature = temperature.max(1.0e-3).recip();
    for &logit in logits {
        let value = ((logit - max_logit) * inverse_temperature).exp();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xiangqi::{RuleDrawReason, RuleOutcome};

    const HIDDEN_MATE_FEN: &str =
        "2bak2r1/4a4/4b4/p2R4p/4C1n2/2P1c3P/P1r3P2/4B4/4A4/2BK1A2R w - - 1 1";

    #[test]
    fn expansion_proves_immediate_mate_without_prior_visits() {
        let position = Position::from_fen(HIDDEN_MATE_FEN).unwrap();
        let mut model = AzNnue::random(16, 7);
        let mate = position.parse_uci_move("d6d9").unwrap();
        model.policy_move_bias[super::super::dense_move_index(crate::az::nnue::canonical_move(
            position.side_to_move(),
            mate,
        ))] = -100.0;
        let mut tree = AzTree::new(
            position.clone(),
            position.initial_rule_history(),
            None,
            &model,
            AzSearchLimits {
                simulations: 1,
                ..Default::default()
            },
        );
        let eval = tree.simulate(tree.root, 0);
        assert_eq!(eval.value, 1.0);
        assert_eq!(tree.nodes[tree.root].solved, Some(1));
        assert_eq!(tree.nodes[tree.root].visits, 1);
        let winning: Vec<_> = tree
            .node_children(tree.root)
            .iter()
            .filter(|child| {
                child.child != NO_CHILD && tree.nodes[child.child as usize].solved == Some(-1)
            })
            .collect();
        assert_eq!(winning.len(), 1);
        assert_eq!(winning[0].mv.to_string(), "d6d9");
        assert_eq!(winning[0].visits, 1);
    }

    #[test]
    fn mate_probe_respects_root_move_restriction() {
        let position = Position::from_fen(HIDDEN_MATE_FEN).unwrap();
        let model = AzNnue::random(16, 7);
        let mv = position.parse_uci_move("g3g4").unwrap();
        let mut tree = AzTree::new(
            position.clone(),
            position.initial_rule_history(),
            Some(vec![mv]),
            &model,
            AzSearchLimits {
                max_depth: 1,
                ..Default::default()
            },
        );
        tree.expand(tree.root);
        assert_eq!(tree.nodes[tree.root].solved, None);
        assert_eq!(tree.node_children(tree.root).len(), 1);
    }

    #[test]
    fn checking_move_with_escape_is_not_mate() {
        let position = Position::from_fen("4k4/9/9/9/9/9/9/9/4P4/4K2R1 w - - 0 1").unwrap();
        let model = AzNnue::random(16, 7);
        let mv = position.parse_uci_move("h0h9").unwrap();
        assert!(position.gives_check_after_move_fast(mv));
        let mut reply = position.clone();
        reply.make_move(mv);
        assert!(!reply.legal_moves().is_empty());
        let mut tree = AzTree::new(
            position.clone(),
            position.initial_rule_history(),
            Some(vec![mv]),
            &model,
            AzSearchLimits::default(),
        );
        tree.expand(tree.root);
        assert_eq!(tree.nodes[tree.root].solved, None);
        assert_eq!(tree.nodes[tree.root].visits, 0);
    }

    #[test]
    fn absolute_root_fpu_explores_unvisited_move_despite_negative_parent_value() {
        let position = Position::startpos();
        let model = AzNnue::random(4, 7);
        let moves = position.legal_moves()[..2].to_vec();
        let mut tree = AzTree::new(
            position.clone(),
            position.initial_rule_history(),
            Some(moves),
            &model,
            AzSearchLimits {
                cpuct_at_root: 0.0,
                cpuct_factor_at_root: 0.0,
                ..AzSearchLimits::default()
            },
        );
        tree.expand(tree.root);
        tree.nodes[tree.root].value_wdl = [0.0, 0.0, 1.0];
        tree.nodes[tree.root].value = -1.0;
        let children = tree.node_children_mut(tree.root);
        children[0].visits = 10;
        children[0].value_wdl_sum = [0.0, 5.0, 5.0];
        assert_eq!(tree.select_child(tree.root), 1);
        tree.fpu_absolute_at_root = false;
        assert_eq!(tree.select_child(tree.root), 0);
    }

    #[test]
    fn kld_stopper_stops_stable_distribution_and_keeps_changing_distribution() {
        let position = Position::startpos();
        let model = AzNnue::random(4, 7);
        let moves = position.legal_moves()[..2].to_vec();
        let mut tree = AzTree::new(
            position.clone(),
            position.initial_rule_history(),
            Some(moves),
            &model,
            AzSearchLimits::default(),
        );
        tree.expand(tree.root);
        let mut stopper = KldGainStopper::default();
        tree.node_children_mut(tree.root)[0].visits = 100;
        tree.node_children_mut(tree.root)[1].visits = 100;
        tree.nodes[tree.root].visits = 200;
        assert!(!stopper.should_stop(&tree, 0.00005));
        tree.node_children_mut(tree.root)[0].visits = 300;
        tree.nodes[tree.root].visits = 400;
        assert!(!stopper.should_stop(&tree, 0.00005));
        tree.node_children_mut(tree.root)[0].visits = 450;
        tree.node_children_mut(tree.root)[1].visits = 150;
        tree.nodes[tree.root].visits = 600;
        assert!(stopper.should_stop(&tree, 0.00005));
        assert!(!stopper.should_stop(&tree, 0.0));
    }

    #[test]
    fn px0_kld_stopping_is_executed_by_all_search_paths() {
        let position = Position::startpos();
        let history = position.initial_rule_history();
        let mv = position.legal_moves()[0];
        let model = AzNnue::random(4, 7);
        let limits = AzSearchLimits {
            simulations: 10_000,
            minimum_kldgain_per_node: 0.00005,
            ..AzSearchLimits::default()
        };
        let ordinary = alphazero_search_with_rules(
            &position,
            Some(history.clone()),
            Some(vec![mv]),
            &model,
            limits,
        );
        let mut workspace = AzSearchWorkspace::new(&model);
        let reused = alphazero_search_with_rules_reusing(
            &position,
            &history,
            vec![mv],
            &model,
            limits,
            &mut workspace,
        );
        let (traced, _) = alphazero_search_trace_with_rules(
            &position,
            Some(history),
            Some(vec![mv]),
            &model,
            limits,
            mv,
        );
        for result in [ordinary, reused, traced] {
            assert_eq!(result.simulations, 400);
            assert_eq!(result.candidates[0].visits, 400);
        }
        let fixed_budget = alphazero_search_with_rules(
            &position,
            None,
            Some(vec![mv]),
            &model,
            AzSearchLimits {
                simulations: 512,
                minimum_kldgain_per_node: 0.0,
                ..limits
            },
        );
        assert_eq!(fixed_budget.simulations, 512);
    }

    #[test]
    fn fixed_alpha_noise_matches_px0_for_different_legal_move_counts() {
        let position = Position::startpos();
        let model = AzNnue::random(4, 31);
        for count in [1, 7, position.legal_moves().len()] {
            let moves = position.legal_moves()[..count].to_vec();
            let limits = AzSearchLimits {
                simulations: 32,
                root_dirichlet_alpha: 8.0,
                root_exploration_fraction: 0.15,
                ..AzSearchLimits::default()
            };
            let dynamic =
                alphazero_search_with_rules(&position, None, Some(moves.clone()), &model, limits);
            let plain = alphazero_search_with_rules(
                &position,
                None,
                Some(moves.clone()),
                &model,
                AzSearchLimits {
                    root_dirichlet_alpha: 0.0,
                    ..limits
                },
            );
            let mut expected: Vec<_> = moves
                .iter()
                .map(|mv| {
                    plain
                        .candidates
                        .iter()
                        .find(|candidate| candidate.mv == *mv)
                        .unwrap()
                        .prior
                })
                .collect();
            apply_root_dirichlet_noise(
                &mut expected,
                8.0,
                limits.root_exploration_fraction,
                limits.seed,
            );
            assert_eq!(dynamic.candidates.len(), count);
            assert!((dynamic.candidates.iter().map(|c| c.prior).sum::<f32>() - 1.0).abs() < 1e-6);
            for (mv, expected) in moves.iter().zip(expected) {
                let actual = dynamic
                    .candidates
                    .iter()
                    .find(|candidate| candidate.mv == *mv)
                    .unwrap();
                assert_eq!(actual.prior, expected);
            }
        }
    }

    #[test]
    fn policy_softmax_temperature_flattens_network_priors() {
        let logits = [2.0, 0.0];
        let mut normal = Vec::new();
        let mut softened = Vec::new();

        softmax_into(&logits, 1.0, &mut normal);
        softmax_into(&logits, 2.0, &mut softened);

        assert!(softened[0] < normal[0]);
        assert!(softened[1] > normal[1]);
        assert!((softened.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn search_extends_through_a_forced_reply_in_one_simulation() {
        let position =
            Position::from_fen("4k1b2/4a4/4ba3/p8/4cN3/3n2N1P/c8/4C4/4A4/2B1KAB2 b").unwrap();
        let checking_move = position.parse_uci_move("a3a0").unwrap();
        let mut checked = position.clone();
        checked.make_move(checking_move);
        assert_eq!(checked.legal_moves(), [Move::from_uci("c0a2").unwrap()]);

        let result = alphazero_search_with_rules(
            &position,
            None,
            Some(vec![checking_move]),
            &AzNnue::random(4, 23),
            AzSearchLimits {
                simulations: 1,
                max_depth: 8,
                ..AzSearchLimits::default()
            },
        );

        assert_eq!(result.search_depth_max, 2);
        assert_eq!(result.search_depth_cutoffs, 0);
    }

    #[test]
    fn search_extends_an_in_check_leaf_with_multiple_evasions() {
        let position = Position::from_fen(
            "2bakab2/9/5r1c1/p1PRC1p2/4P2nP/6P2/4N1r2/7c1/4A4/2BAK1B1R b - - 0 1",
        )
        .unwrap();
        let checking_move = position.parse_uci_move("h2h0").unwrap();
        let mut checked = position.clone();
        checked.make_move(checking_move);
        assert!(checked.in_check(checked.side_to_move()));
        assert!(checked.legal_moves().len() > 1);

        let result = alphazero_search_with_rules(
            &position,
            None,
            Some(vec![checking_move]),
            &AzNnue::random(4, 29),
            AzSearchLimits {
                simulations: 1,
                max_depth: 8,
                ..AzSearchLimits::default()
            },
        );

        assert_eq!(result.search_depth_max, 2);
        assert_eq!(result.search_depth_cutoffs, 0);
    }

    #[test]
    fn tactical_estimates_feed_mcts_without_claiming_a_proof() {
        let p = Position::from_fen(
            "Cn1akab2/5R3/2n1b4/p2Rp1P1p/2p3r2/5N3/P1c1P4/4B4/9/1r1AKAB2 b - - 0 1",
        )
        .unwrap();
        let mv = p.parse_uci_move("f9e8").unwrap();
        let mut model = AzNnue::random(4, 19);
        model.tactical_search_nodes = 128;
        model.tactical_search_plies = 2;
        model.tactical_quiet_plies = 1;
        let result = alphazero_search_with_rules(
            &p,
            None,
            Some(vec![mv]),
            &model,
            AzSearchLimits {
                simulations: 2,
                ..AzSearchLimits::default()
            },
        );
        assert!(result.tactical_nodes > 0);
        assert!(result.tactical_nodes <= 128 * 2);
        assert!(result.candidates.iter().all(|c| c.solved.is_none()));
        let wdl = model.evaluate_wdl_with_rules(&p, &p.initial_rule_history(), &[mv]);
        assert_eq!(result.network_value_wdl, wdl);
    }

    #[test]
    fn tactical_reply_receives_real_visits_despite_zero_prior() {
        let p = Position::startpos();
        let model = AzNnue::random(4, 19);
        let mut tree = AzTree::new(
            p.clone(),
            p.initial_rule_history(),
            None,
            &model,
            AzSearchLimits::default(),
        );
        tree.expand(0);
        let reply = tree.node_children(0)[0].mv;
        tree.nodes[0].tactical_reply = Some(reply);
        tree.node_children_mut(0)[0].prior = 0.0;
        assert_eq!(tree.select_child(0), 0);
        tree.simulate(0, 0);
        assert_eq!(tree.node_children(0)[0].visits, 1);
        assert_eq!(tree.select_child(0), 0);
        tree.simulate(0, 0);
        assert_eq!(tree.node_children(0)[0].visits, 2);
        assert_ne!(tree.select_child(0), 0);
    }

    #[test]
    fn quiet_opening_does_not_run_a_full_width_tactical_probe() {
        let p = Position::startpos();
        let mut model = AzNnue::random(4, 19);
        model.tactical_search_nodes = 4096;
        let mv = p.parse_uci_move("h2e2").unwrap();
        let result = alphazero_search_with_rules(
            &p,
            None,
            Some(vec![mv]),
            &model,
            AzSearchLimits {
                simulations: 1,
                ..AzSearchLimits::default()
            },
        );
        assert_eq!(result.tactical_nodes, 0);
    }

    #[test]
    fn low_prior_root_threat_is_deferred_then_audited() {
        let p = Position::from_fen(
            "Cn1akab2/5R3/2n1b4/p2Rp1P1p/2p3r2/5N3/P1c1P4/4B4/9/1r1AKAB2 b - - 0 1",
        )
        .unwrap();
        let mut model = AzNnue::random(4, 19);
        model.tactical_search_nodes = 64;
        let mut tree = AzTree::new(
            p.clone(),
            p.initial_rule_history(),
            None,
            &model,
            AzSearchLimits::default(),
        );
        tree.expand(0);
        let mv = p.parse_uci_move("b0b1").unwrap();
        let index = tree
            .node_children(0)
            .iter()
            .position(|child| child.mv == mv)
            .unwrap();
        tree.root_raw_priors.fill(1.0);
        tree.root_raw_priors[index] = 0.0;
        tree.simulate_child(0, index, 1);
        let child = tree.node_children(0)[index].child_node().unwrap();
        assert!(tree.nodes[child].tactical_pending);
        assert_eq!(tree.tactical_nodes, 0);
        tree.nodes[child].visits = 8;
        let entry = tree.nodes[child].rule_entry.unwrap();
        tree.rule_history_scratch.push(entry);
        tree.simulate(child, 1);
        tree.rule_history_scratch.pop();
        assert!(!tree.nodes[child].tactical_pending);
        assert!(tree.tactical_nodes > 0);
    }

    #[test]
    fn tactical_budget_fallback_matches_the_original_leaf() {
        let p = Position::from_fen(
            "Cn1akab2/5R3/2n1b4/p2Rp1P1p/2p3r2/5N3/P1c1P4/4B4/9/1r1AKAB2 b - - 0 1",
        )
        .unwrap();
        let mv = p.parse_uci_move("f9e8").unwrap();
        let mut model = AzNnue::random(4, 19);
        let limits = AzSearchLimits {
            simulations: 1,
            ..AzSearchLimits::default()
        };
        let baseline = alphazero_search_with_rules(&p, None, Some(vec![mv]), &model, limits);
        model.tactical_search_nodes = 1;
        let result = alphazero_search_with_rules(&p, None, Some(vec![mv]), &model, limits);
        assert!(result.tactical_aborted > 0);
        assert_eq!(result.tactical_nodes, 1);
        assert_eq!(result.value_wdl, baseline.value_wdl);
        assert_eq!(result.network_value_wdl, baseline.network_value_wdl);
        assert!(result.candidates[0].solved.is_none());
    }

    #[test]
    fn child_node_index_uses_compact_sentinel_representation() {
        assert!(std::mem::size_of::<AzChild>() <= 40);
        let mut child = AzChild {
            mv: Position::startpos().legal_moves()[0],
            prior: 1.0,
            visits: 0,
            value_wdl_sum: [0.0; 3],
            child: NO_CHILD,
        };
        assert_eq!(child.child_node(), None);
        child.set_child_node(17);
        assert_eq!(child.child_node(), Some(17));
    }

    #[test]
    fn stopped_search_returns_root_result_without_running_simulations() {
        let stop = Arc::new(AtomicBool::new(true));
        let control = AzSearchControl::new(stop, None);
        let result = alphazero_search_with_rules_controlled(
            &Position::startpos(),
            None,
            None,
            &AzNnue::random(4, 19),
            AzSearchLimits {
                simulations: 128,
                ..AzSearchLimits::default()
            },
            Some(&control),
        );

        assert_eq!(result.simulations, 0);
        assert!(result.best_move.is_some());
    }

    #[test]
    fn wdl_q_applies_draw_score_instead_of_discarding_draw_probability() {
        let child = AzChild {
            mv: Position::startpos().legal_moves()[0],
            prior: 1.0,
            visits: 4,
            value_wdl_sum: [1.0, 2.0, 1.0],
            child: NO_CHILD,
        };

        assert!((child.q(0.0) - 0.0).abs() < 1e-6);
        assert!((child.q(0.6) - 0.3).abs() < 1e-6);
        assert!((child.q(-0.6) + 0.3).abs() < 1e-6);
    }

    #[test]
    fn draw_preference_is_kept_in_the_root_players_perspective() {
        let position = Position::startpos();
        let legal = position.legal_moves();
        let model = AzNnue::random(4, 17);
        let mut tree = AzTree::new(
            position.clone(),
            position.initial_rule_history(),
            Some(vec![legal[0]]),
            &model,
            AzSearchLimits {
                draw_score: 0.4,
                ..AzSearchLimits::default()
            },
        );

        tree.expand(tree.root);
        tree.simulate_child(tree.root, 0, 1);
        let child_node = tree.node_children(tree.root)[0].child_node().unwrap();
        assert!((tree.node_draw_score(tree.root) - 0.4).abs() < 1e-6);
        assert!((tree.node_draw_score(child_node) + 0.4).abs() < 1e-6);
    }

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
                fpu_value: 0.33,
                fpu_value_at_root: 0.33,
                fpu_absolute_at_root: true,
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
    fn reusable_workspace_preserves_search_result() {
        let position = Position::startpos();
        let history = position.initial_rule_history();
        let legal = position.legal_moves_with_rules(&history);
        let model = AzNnue::random(32, 71);
        let limits = AzSearchLimits {
            simulations: 128,
            seed: 91,
            root_dirichlet_alpha: 8.0,
            root_exploration_fraction: 0.1,
            ..AzSearchLimits::default()
        };
        let expected = alphazero_search_with_rules(
            &position,
            Some(history.clone()),
            Some(legal.clone()),
            &model,
            limits,
        );
        let mut workspace = AzSearchWorkspace::new(&model);
        let actual = alphazero_search_with_rules_reusing(
            &position,
            &history,
            legal,
            &model,
            limits,
            &mut workspace,
        );

        assert_eq!(actual.best_move, expected.best_move);
        assert_eq!(actual.simulations, expected.simulations);
        assert_eq!(actual.value_q.to_bits(), expected.value_q.to_bits());
        assert_eq!(
            actual.value_wdl.map(f32::to_bits),
            expected.value_wdl.map(f32::to_bits)
        );
        assert_eq!(actual.candidates.len(), expected.candidates.len());
        for (actual, expected) in actual.candidates.iter().zip(&expected.candidates) {
            assert_eq!(actual.mv, expected.mv);
            assert_eq!(actual.visits, expected.visits);
            assert_eq!(actual.q.to_bits(), expected.q.to_bits());
            assert_eq!(actual.policy.to_bits(), expected.policy.to_bits());
            assert_eq!(actual.prior.to_bits(), expected.prior.to_bits());
        }
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
                fpu_value: 0.33,
                fpu_value_at_root: 0.33,
                fpu_absolute_at_root: true,
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
                fpu_value: 0.33,
                fpu_value_at_root: 0.33,
                fpu_absolute_at_root: true,
                value_scale: 1.0,
                ..AzSearchLimits::default()
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
                root_dirichlet_alpha: 8.0,
                root_exploration_fraction: 0.25,
                fpu_value: 0.33,
                fpu_value_at_root: 0.33,
                fpu_absolute_at_root: true,
                value_scale: 1.0,
                ..AzSearchLimits::default()
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
    fn select_child_breaks_equal_scores_by_higher_prior() {
        let model = AzNnue::random(4, 7);
        let position = Position::startpos();
        let legal = position.legal_moves();
        assert!(legal.len() >= 2);

        let mut tree = AzTree::new(
            position.clone(),
            position.initial_rule_history(),
            None,
            &model,
            AzSearchLimits {
                simulations: 1,
                seed: 31,
                cpuct: 1.5,
                cpuct_at_root: 1.5,
                max_depth: 0,
                root_dirichlet_alpha: 0.0,
                root_exploration_fraction: 0.0,
                fpu_value: 0.33,
                fpu_value_at_root: 0.33,
                fpu_absolute_at_root: true,
                value_scale: 1.0,
                ..AzSearchLimits::default()
            },
        );
        tree.cpuct_at_root = 0.0;
        tree.set_node_children(
            tree.root,
            vec![
                AzChild {
                    mv: legal[0],
                    prior: 0.10,
                    visits: 1,
                    value_wdl_sum: [0.0, 1.0, 0.0],
                    child: NO_CHILD,
                },
                AzChild {
                    mv: legal[1],
                    prior: 0.90,
                    visits: 1,
                    value_wdl_sum: [0.0, 1.0, 0.0],
                    child: NO_CHILD,
                },
            ],
        );

        assert_eq!(tree.select_child(tree.root), 1);
    }

    #[test]
    fn raw_prior_is_network_policy_before_temperature_and_noise() {
        let mut model = AzNnue::random(32, 83);
        for (index, bias) in model.policy_move_bias.iter_mut().enumerate() {
            *bias = (index % 17) as f32 * 0.1;
        }
        let position = Position::startpos();
        let normal = alphazero_search(
            &position,
            &model,
            AzSearchLimits {
                simulations: 1,
                policy_softmax_temp: 1.0,
                ..AzSearchLimits::default()
            },
        );
        let softened = alphazero_search(
            &position,
            &model,
            AzSearchLimits {
                simulations: 1,
                policy_softmax_temp: 3.0,
                ..AzSearchLimits::default()
            },
        );
        for candidate in &normal.candidates {
            let other = softened
                .candidates
                .iter()
                .find(|other| other.mv == candidate.mv)
                .unwrap();
            assert!((candidate.raw_prior - other.raw_prior).abs() < 1.0e-6);
        }
        assert!(
            softened
                .candidates
                .iter()
                .any(|candidate| (candidate.raw_prior - candidate.prior).abs() > 1.0e-5)
        );
    }

    #[test]
    fn root_fpu_is_a_parent_value_reduction_not_an_absolute_q() {
        let model = AzNnue::random(4, 41);
        let position = Position::startpos();
        let legal = position.legal_moves();
        let mut tree = AzTree::new(
            position.clone(),
            position.initial_rule_history(),
            None,
            &model,
            AzSearchLimits {
                cpuct_at_root: 0.0,
                cpuct_factor_at_root: 0.0,
                fpu_value_at_root: 0.33,
                fpu_absolute_at_root: false,
                ..AzSearchLimits::default()
            },
        );
        tree.cpuct_at_root = 0.0;
        tree.nodes[tree.root].visits = 1;
        tree.nodes[tree.root].value_wdl_sum = [0.0, 1.0, 0.0];
        tree.set_node_children(
            tree.root,
            [
                AzChild {
                    mv: legal[0],
                    prior: 0.25,
                    visits: 1,
                    value_wdl_sum: [0.0, 1.0, 0.0],
                    child: NO_CHILD,
                },
                AzChild {
                    mv: legal[1],
                    prior: 0.75,
                    visits: 0,
                    value_wdl_sum: [0.0; 3],
                    child: NO_CHILD,
                },
            ],
        );

        assert_eq!(tree.select_child(tree.root), 0);
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

        assert!(full.value_q > 0.0);
        assert!((scaled.value_q - full.value_q * 0.25).abs() <= 1e-5);
    }

    #[test]
    fn mcts_state_make_move_matches_manual_context_updates() {
        let position = Position::startpos();
        let mv = position.legal_moves()[0];
        let mut node_position = position.clone();
        let mut node_rule_history = position.initial_rule_history();

        let mut manual_position = position;
        let mut manual_rule_history = manual_position.initial_rule_history();
        manual_rule_history.push(manual_position.rule_history_entry_after_move(mv));
        manual_position.make_move(mv);

        node_rule_history.push(node_position.rule_history_entry_after_move(mv));
        node_position.make_move(mv);

        assert_eq!(node_position, manual_position);
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
        let child_node = tree.node_children(tree.root)[0].child_node().unwrap();
        assert_eq!(tree.nodes[child_node].rule_entry, Some(expected));
    }

    #[test]
    fn provided_root_moves_only_apply_at_root() {
        let position = Position::startpos();
        let legal = position.legal_moves();
        let root_moves = vec![legal[0]];
        let model = AzNnue::random(4, 7);
        let mut tree = AzTree::new(
            position,
            Position::startpos().initial_rule_history(),
            Some(root_moves.clone()),
            &model,
            AzSearchLimits::default(),
        );

        tree.expand(tree.root);
        assert_eq!(tree.node_children(tree.root).len(), 1);
        let child_index = 0;
        tree.simulate_child(tree.root, child_index, 1);
        let child_node = tree.node_children(tree.root)[child_index]
            .child_node()
            .unwrap();
        tree.expand(child_node);
        assert_ne!(tree.node_children(child_node).len(), root_moves.len());
    }

    #[test]
    fn terminal_value_uses_rule_history_not_just_board_hash() {
        let position = Position::startpos();
        let mut rule_history = vec![
            position.rule_history_entry(None),
            RuleHistoryEntry {
                hash: position.hash(),
                side_to_move: position.side_to_move(),
                mover: Some(Color::Black),
                gives_check: false,
                chased_mask: 0,
                mv: None,
                captured: None,
                rule60_clock: 0,
            },
            RuleHistoryEntry {
                hash: position.hash(),
                side_to_move: position.side_to_move(),
                mover: Some(Color::Black),
                gives_check: false,
                chased_mask: 0,
                mv: None,
                captured: None,
                rule60_clock: 0,
            },
            RuleHistoryEntry {
                hash: position.hash(),
                side_to_move: position.side_to_move(),
                mover: Some(Color::Black),
                gives_check: false,
                chased_mask: 0,
                mv: None,
                captured: None,
                rule60_clock: 0,
            },
            RuleHistoryEntry {
                hash: position.hash(),
                side_to_move: position.side_to_move(),
                mover: Some(Color::Black),
                gives_check: false,
                chased_mask: 0,
                mv: None,
                captured: None,
                rule60_clock: 0,
            },
            RuleHistoryEntry {
                hash: position.hash(),
                side_to_move: position.side_to_move(),
                mover: Some(Color::Black),
                gives_check: false,
                chased_mask: 0,
                mv: None,
                captured: None,
                rule60_clock: 0,
            },
        ];

        rule_history.extend_from_within(1..);
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

    #[test]
    fn external_controller_mode_does_not_adjudicate_the_root() {
        let position = Position::startpos();
        let mut rule_history = vec![
            position.rule_history_entry(None),
            RuleHistoryEntry {
                hash: position.hash(),
                side_to_move: position.side_to_move(),
                mover: Some(Color::Black),
                gives_check: false,
                chased_mask: 0,
                mv: None,
                captured: None,
                rule60_clock: 0,
            },
        ];
        rule_history.push(rule_history[1]);
        assert!(position.rule_outcome_with_history(&rule_history).is_some());

        let model = AzNnue::random(4, 72);
        let root_moves = position.legal_moves();
        let mut internal = AzTree::new(
            position.clone(),
            rule_history.clone(),
            Some(root_moves.clone()),
            &model,
            AzSearchLimits::default(),
        );
        internal.expand(internal.root);
        assert_eq!(internal.node_children(internal.root).len(), 0);

        let mut external = AzTree::new(
            position,
            rule_history,
            Some(root_moves),
            &model,
            AzSearchLimits::default(),
        );
        external.adjudicate_root_rules = false;
        external.expand(external.root);
        assert!(!external.node_children(external.root).is_empty());
    }

    #[test]
    #[ignore = "需要本地已训练检查点；不训练、不对弈"]
    fn trained_incremental_value_matches_full_recompute() {
        let path = std::env::var("CHINESEAI_AUDIT_MODEL")
            .unwrap_or_else(|_| "tmp/px0-reservoir-131072.epoch-3.safetensors".into());
        let model = AzNnue::load(path).unwrap();
        let mut rng = SplitMix64::new(20260928);
        let mut checked = 0;
        let mut captures = 0;
        let mut king_moves = 0;
        let mut max_hidden = 0f32;
        let mut max_value = 0f32;
        let mut max_policy = 0f32;
        for game in 0..8 {
            let mut position = Position::startpos();
            let mut history = position.initial_rule_history();
            let mut hidden = AzEvalAccumulator::new(&model, &position).into_hidden_sum();
            let mut policy = [
                model.policy_accumulator(&position, Color::Red),
                model.policy_accumulator(&position, Color::Black),
            ];
            for ply in 0..160 {
                let moves = position.legal_moves();
                if moves.is_empty() {
                    break;
                }
                let context = rule_context_features(&position, &history);
                let mut full = AzEvalScratch::new(model.arch);
                let mut incremental = AzEvalScratch::new(model.arch);
                let a = model.evaluate_with_scratch_output(&position, &moves, &context, &mut full);
                let b = model.evaluate_incremental_with_scratch_output(
                    &position,
                    &hidden,
                    &policy[color_index(position.side_to_move())],
                    &moves,
                    &[],
                    &context,
                    &mut incremental,
                );
                max_value = max_value.max((a.value - b.value).abs());
                for (a, b) in a.value_wdl.iter().zip(b.value_wdl) {
                    max_value = max_value.max((a - b).abs());
                }
                for (a, b) in full.logits.iter().zip(&incremental.logits) {
                    max_policy = max_policy.max((a - b).abs());
                }
                let refreshed = AzEvalAccumulator::new(&model, &position).into_hidden_sum();
                for (a, b) in hidden.iter().zip(refreshed) {
                    max_hidden = max_hidden.max((a - b).abs());
                }
                assert!(
                    (a.value - b.value).abs() < 1e-4,
                    "game={game} ply={ply} fen={}",
                    position.to_fen()
                );
                checked += 1;
                let preferred: Vec<_> = moves
                    .iter()
                    .copied()
                    .filter(|mv| {
                        if ply % 5 == 0 {
                            position.piece_at(mv.from as usize).unwrap().kind
                                == crate::xiangqi::PieceKind::General
                        } else {
                            position.piece_at(mv.to as usize).is_some()
                        }
                    })
                    .collect();
                let choices = if preferred.is_empty() {
                    &moves
                } else {
                    &preferred
                };
                let mv = choices[rng.next_u64() as usize % choices.len()];
                let before = position.clone();
                let moved = before.piece_at(mv.from as usize).unwrap();
                let captured = before.piece_at(mv.to as usize);
                captures += usize::from(captured.is_some());
                king_moves += usize::from(moved.kind == crate::xiangqi::PieceKind::General);
                position.make_move(mv);
                AzEvalAccumulator::apply_transition_to_hidden(
                    &model,
                    &before,
                    &position,
                    mv,
                    moved,
                    captured,
                    &mut hidden,
                );
                for side in [Color::Red, Color::Black] {
                    model.apply_policy_transition(
                        &before,
                        &position,
                        mv,
                        moved,
                        captured,
                        side,
                        &mut policy[color_index(side)],
                    );
                }
                history.push(position.rule_history_entry_after_moved(
                    before.side_to_move(),
                    mv,
                    captured,
                ));
            }
        }
        // 核对搜索实际存储的单视角、跨祖父节点更新的缓存。
        let position = Position::startpos();
        let root_history = position.initial_rule_history();
        let mut tree = AzTree::new(
            position,
            root_history.clone(),
            None,
            &model,
            AzSearchLimits {
                simulations: 800,
                ..AzSearchLimits::default()
            },
        );
        for _ in 0..800 {
            tree.simulate(tree.root, 0);
        }
        for node in &tree.nodes {
            let fresh = AzEvalAccumulator::new(&model, &node.position).into_hidden_sum();
            let expected = AzEvalAccumulator::hidden_for_slice(
                &fresh,
                model.hidden_size,
                node.position.side_to_move(),
            );
            let offset = node.accumulator_offset as usize;
            for (a, b) in tree.accumulator_arena[offset..offset + model.hidden_size]
                .iter()
                .zip(expected)
            {
                max_hidden = max_hidden.max((a - b).abs());
                assert!(
                    (a - b).abs() < 1e-4,
                    "tree cache fen={}",
                    node.position.to_fen()
                );
            }
            let expected_policy =
                model.policy_accumulator(&node.position, node.position.side_to_move());
            for (a, b) in node.policy_accumulator.iter().zip(expected_policy) {
                assert!((a - b).abs() < 1e-4, "tree policy cache");
            }
        }
        println!(
            "incremental audit: sequence_positions={checked} tree_nodes={} captures={captures} king_moves={king_moves} max_hidden={max_hidden:.8} max_value={max_value:.8} max_policy={max_policy:.8}",
            tree.nodes.len()
        );
        assert!(captures > 0 && king_moves > 0 && checked > 100);
        assert!(max_policy < 1e-3);
    }

    #[test]
    fn huge_timed_simulation_limit_uses_bounded_initial_capacity() {
        let position = Position::startpos();
        let model = AzNnue::random(4, 71);
        let tree = AzTree::new(
            position.clone(),
            position.initial_rule_history(),
            None,
            &model,
            AzSearchLimits {
                simulations: usize::MAX,
                ..AzSearchLimits::default()
            },
        );

        assert_eq!(tree.nodes.capacity(), INITIAL_TREE_NODE_CAPACITY);
        assert_eq!(
            tree.accumulator_arena.capacity(),
            (INITIAL_TREE_NODE_CAPACITY + 1) * model.hidden_size
        );
        assert_eq!(
            tree.children.capacity(),
            INITIAL_TREE_NODE_CAPACITY * INITIAL_CHILDREN_PER_NODE_ESTIMATE
        );
    }

    #[test]
    #[ignore = "需要本地 Px0 65536 蒸馏检查点"]
    fn solver_does_not_select_proven_mate_in_px0_trajectory() {
        let model = AzNnue::load("tmp/px0-reservoir-65536.best.safetensors").unwrap();
        let mut position = Position::from_fen(
            "rnbakab1r/9/1c4n2/2p1p1p1p/p8/1C7/P1P1P1PcP/3CB4/9/RNBAKA1NR w - - 0 1",
        )
        .unwrap();
        let mut history = position.initial_rule_history();
        for name in "h0g2 i9h9 c3c4 b9a7 b0c2 b7c7 a3a4 c6c5 c4c5 c7c2 a4a5 a9b9 a5b5 b9a9 i0i1 h9h5 b4g4 g7e8 i1f1 g9e7 a0a2 c2c3 a2a3 c3g3 a3a4 g6g5 a4f4".split_whitespace() {
            let mv = position.parse_uci_move(name).unwrap();
            history.push(position.rule_history_entry_after_move(mv));
            position.make_move(mv);
        }
        let mut tree = AzTree::new(position, history, None, &model, AzSearchLimits::default());
        let root = tree.root;
        tree.expand(root);
        for _ in 0..6400 {
            tree.simulate(root, 0);
        }
        let result = tree.search_result(6400);
        assert!(
            tree.node_children(root)
                .iter()
                .any(|c| tree.child_solved(c) != Some(-1))
        );
        let mut proven_losses = 0;
        for child in tree.node_children(root) {
            let candidate = result.candidates.iter().find(|c| c.mv == child.mv).unwrap();
            if tree.child_solved(child) == Some(-1) {
                proven_losses += 1;
                assert_ne!(result.best_move, Some(child.mv));
                assert_eq!(candidate.policy, 0.0);
                assert_eq!(candidate.q, -1.0);
            }
        }
        assert!(proven_losses > 0);
        assert!((result.candidates.iter().map(|c| c.policy).sum::<f32>() - 1.0).abs() < 1e-5);
        println!(
            "SOLVER selected={:?} excluded_proven_losses={proven_losses}",
            result.best_move
        );
        for name in ["a9b9", "g5g4", "h5h9"] {
            let child = tree
                .node_children(root)
                .iter()
                .find(|c| c.mv.to_string() == name)
                .unwrap();
            let Some(ni) = child.child_node() else {
                continue;
            };
            let node = &tree.nodes[ni];
            let children = tree.node_children(ni);
            let fpu = alphazero_fpu_value_reduction(node, children, 0.23, 0.0);
            println!(
                "AUDIT root_move={name} visits={} q={} node_nn_q={} internal_fpu={fpu}",
                child.visits,
                child.q(0.0),
                node.value
            );
            for reply in ["f4f9", "f4g4"] {
                if let Some(edge) = children.iter().find(|c| c.mv.to_string() == reply) {
                    println!(
                        "AUDIT reply={reply} visits={} prior={} q={} child_exists={}",
                        edge.visits,
                        edge.prior,
                        edge.q(0.0),
                        edge.child_node().is_some()
                    );
                }
            }
        }
    }

    fn solver_tree_with_two_unresolved_children(model: &AzNnue) -> AzTree<'_> {
        let position = Position::startpos();
        let history = position.initial_rule_history();
        let legal = position.legal_moves();
        let mut tree = AzTree::new(
            position,
            history,
            Some(legal[..2].to_vec()),
            model,
            AzSearchLimits::default(),
        );
        let root = tree.root;
        tree.expand(root);
        tree.simulate_child(root, 0, 1);
        tree.simulate_child(root, 1, 1);
        assert_eq!(tree.nodes[root].solved, None);
        tree
    }

    #[test]
    fn solver_draw_requires_all_replies_proven_and_ignores_network_draw() {
        let model = AzNnue::random(4, 7);
        let mut tree = solver_tree_with_two_unresolved_children(&model);
        let root = tree.root;
        let first = tree.node_children(root)[0].child_node().unwrap();
        let second = tree.node_children(root)[1].child_node().unwrap();
        tree.nodes[first].solved = Some(0);
        tree.nodes[second].value = 0.0;
        tree.nodes[second].value_wdl = [0.0, 1.0, 0.0];
        tree.update_solved(root);
        assert_eq!(tree.nodes[root].solved, None);
        assert_eq!(tree.nodes[root].bounds, (0, 1));
        assert_eq!(tree.nodes[second].solved, None);
        tree.nodes[second].solved = Some(1);
        tree.update_solved(root);
        assert_eq!(tree.nodes[root].solved, Some(0));
        assert_eq!(tree.node_eval(root).value_wdl, [0.0, 1.0, 0.0]);
        assert_eq!(tree.root_policy(root), vec![1.0, 0.0]);
    }

    #[test]
    fn sticky_proof_corrects_prior_visits_without_adding_visits_or_double_counting() {
        let model = AzNnue::random(4, 7);
        let mut tree = solver_tree_with_two_unresolved_children(&model);
        let root = tree.root;
        let first = tree.node_children(root)[0].child_node().unwrap();
        tree.nodes[root].visits = 10;
        tree.nodes[root].value_wdl_sum = [4.0, 2.0, 4.0];
        tree.nodes[first].visits = 6;
        tree.nodes[first].value_wdl_sum = [4.0, 1.0, 1.0];
        tree.node_children_mut(root)[0].visits = 6;
        tree.node_children_mut(root)[0].value_wdl_sum = [1.0, 1.0, 4.0];
        tree.set_proven(first, -1);
        assert_eq!(tree.nodes[first].value_wdl_sum, [0.0, 0.0, 6.0]);
        assert_eq!(tree.node_children(root)[0].value_wdl_sum, [6.0, 0.0, 0.0]);
        assert_eq!(tree.nodes[root].value_wdl_sum, [9.0, 1.0, 0.0]);
        assert_eq!(tree.nodes[root].visits, 10);
        assert_eq!(tree.node_children(root)[0].visits, 6);
        tree.update_solved(root);
        assert_eq!(tree.nodes[root].solved, Some(1));
        assert_eq!(tree.nodes[root].value_wdl_sum, [10.0, 0.0, 0.0]);
        tree.set_proven(first, -1);
        tree.update_solved(root);
        assert_eq!(tree.nodes[root].value_wdl_sum, [10.0, 0.0, 0.0]);
    }

    #[test]
    fn solver_win_and_loss_flip_child_perspective() {
        let model = AzNnue::random(4, 7);
        let mut winning = solver_tree_with_two_unresolved_children(&model);
        let root = winning.root;
        let first = winning.node_children(root)[0].child_node().unwrap();
        winning.nodes[first].solved = Some(-1);
        winning.update_solved(root);
        assert_eq!(winning.nodes[root].solved, Some(1));
        assert_eq!(winning.node_eval(root).value_wdl, [1.0, 0.0, 0.0]);
        assert_eq!(winning.root_policy(root), vec![1.0, 0.0]);
        let mut losing = solver_tree_with_two_unresolved_children(&model);
        let root = losing.root;
        for index in 0..2 {
            let child = losing.node_children(root)[index].child_node().unwrap();
            losing.nodes[child].solved = Some(1);
        }
        losing.update_solved(root);
        assert_eq!(losing.nodes[root].solved, Some(-1));
        assert_eq!(losing.node_eval(root).value_wdl, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn solver_marks_actual_rule_draw_without_network_prediction() {
        let model = AzNnue::random(4, 7);
        let mut position = Position::startpos();
        position.set_rule60_max_ply(Some(1));
        let mv = position.legal_moves()[0];
        let mut history = position.initial_rule_history();
        history.push(position.rule_history_entry_after_move(mv));
        position.make_move(mv);
        assert!(matches!(
            position.rule_outcome_with_history(&history),
            Some(RuleOutcome::Draw(_))
        ));
        let mut tree = AzTree::new(position, history, None, &model, AzSearchLimits::default());
        let root = tree.root;
        let output = tree.expand(root);
        assert_eq!(tree.nodes[root].solved, Some(0));
        assert_eq!(output.value_wdl, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn solver_proven_mate_stops_all_search_paths_before_kl_interval() {
        let model = AzNnue::random(4, 7);
        let mut position = Position::from_fen(
            "r1baka3/4n4/n3b4/4p3p/1PP3pr1/5RC2/4P1ccP/3CB1N2/5R3/2BAKA3 b - - 3 1",
        )
        .unwrap();
        let mut history = position.initial_rule_history();
        let mv = position.parse_uci_move("a9b9").unwrap();
        history.push(position.rule_history_entry_after_move(mv));
        position.make_move(mv);
        let mate = position.parse_uci_move("f4f9").unwrap();
        let limits = AzSearchLimits {
            simulations: 10000,
            ..AzSearchLimits::default()
        };
        let ordinary = alphazero_search_with_rules(
            &position,
            Some(history.clone()),
            Some(vec![mate]),
            &model,
            limits,
        );
        let mut workspace = AzSearchWorkspace::new(&model);
        let reused = alphazero_search_with_rules_reusing(
            &position,
            &history,
            vec![mate],
            &model,
            limits,
            &mut workspace,
        );
        let (traced, trace) = alphazero_search_trace_with_rules(
            &position,
            Some(history),
            Some(vec![mate]),
            &model,
            limits,
            mate,
        );
        for result in [ordinary, reused, traced] {
            assert_eq!(result.simulations, 1);
            assert_eq!(result.best_move, Some(mate));
            assert_eq!(result.value_wdl, [1.0, 0.0, 0.0]);
            assert_eq!(result.best_value_wdl, [1.0, 0.0, 0.0]);
            assert_ne!(result.network_value_wdl, result.value_wdl);
            assert_eq!(result.candidates[0].visits, 1);
            assert_eq!(result.candidates[0].policy, 1.0);
            assert_eq!(result.candidates[0].q, 1.0);
        }
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].q, 1.0);
    }
}
