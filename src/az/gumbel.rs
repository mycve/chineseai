use crate::xiangqi::{Color, Move, Position, RuleHistoryEntry, RuleOutcome};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use super::{AzEvalAccumulator, AzEvalOutput, AzEvalScratch, AzNnue, SplitMix64};

const DEFAULT_MAX_CONSIDERED_ACTIONS: usize = 16;
const DEFAULT_GUMBEL_SCALE: f32 = 0.0;
const DEFAULT_Q_VALUE_SCALE: f32 = 0.1;
const DEFAULT_Q_MAXVISIT_INIT: f32 = 50.0;
const NO_CHILD: u32 = u32::MAX;
const SEARCH_PROGRESS_POLL_SIMULATIONS: usize = 64;
const SEARCH_PROGRESS_INTERVAL: Duration = Duration::from_millis(250);

#[derive(Clone, Copy, Debug)]
pub struct GumbelSearchLimits {
    pub simulations: usize,
    pub seed: u64,
    pub max_num_considered_actions: usize,
    pub gumbel_scale: f32,
    pub q_value_scale: f32,
    pub q_maxvisit_init: f32,
    /// Maximum search depth in plies below root. 0 keeps the default:
    /// max_depth = num_simulations.
    pub max_depth: usize,
    pub draw_score: f32,
    pub value_scale: f32,
}

impl Default for GumbelSearchLimits {
    fn default() -> Self {
        Self {
            simulations: 10_000,
            seed: 0,
            max_num_considered_actions: DEFAULT_MAX_CONSIDERED_ACTIONS,
            gumbel_scale: DEFAULT_GUMBEL_SCALE,
            q_value_scale: DEFAULT_Q_VALUE_SCALE,
            q_maxvisit_init: DEFAULT_Q_MAXVISIT_INIT,
            max_depth: 0,
            draw_score: 0.0,
            value_scale: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GumbelCandidate {
    pub mv: Move,
    pub visits: u32,
    pub q: f32,
    pub moves_left: f32,
    pub raw_prior: f32,
    pub prior: f32,
    pub policy: f32,
}

#[derive(Clone, Debug)]
pub struct GumbelSearchResult {
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
    pub candidates: Vec<GumbelCandidate>,
}

#[derive(Clone, Debug)]
pub struct GumbelSearchControl {
    stop: Arc<AtomicBool>,
    deadline: Option<Instant>,
}

impl GumbelSearchControl {
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

pub fn gumbel_search_with_rules(
    position: &Position,
    rule_history: Option<Vec<RuleHistoryEntry>>,
    root_moves: Option<Vec<Move>>,
    model: &AzNnue,
    limits: GumbelSearchLimits,
) -> GumbelSearchResult {
    gumbel_search_with_rules_controlled(position, rule_history, root_moves, model, limits, None)
}

pub fn gumbel_search_with_rules_controlled(
    position: &Position,
    rule_history: Option<Vec<RuleHistoryEntry>>,
    root_moves: Option<Vec<Move>>,
    model: &AzNnue,
    limits: GumbelSearchLimits,
    control: Option<&GumbelSearchControl>,
) -> GumbelSearchResult {
    gumbel_search_with_rules_controlled_with_progress(
        position,
        rule_history,
        root_moves,
        model,
        limits,
        control,
        None,
    )
}

pub fn gumbel_search_with_rules_controlled_with_progress(
    position: &Position,
    rule_history: Option<Vec<RuleHistoryEntry>>,
    root_moves: Option<Vec<Move>>,
    model: &AzNnue,
    limits: GumbelSearchLimits,
    control: Option<&GumbelSearchControl>,
    mut progress: Option<&mut dyn FnMut(&GumbelSearchResult)>,
) -> GumbelSearchResult {
    crate::scope_profile!("az.gumbel_search");
    let mut tree = AzTree::new(
        position.clone(),
        rule_history.unwrap_or_else(|| position.initial_rule_history()),
        root_moves,
        model,
        limits,
    );
    let root = tree.root;
    {
        crate::scope_profile!("az.search.root_expand");
        tree.expand(root);
    }
    if tree.nodes[root].children_len == 0 {
        let value_q = wdl_utility(tree.nodes[root].value_wdl, tree.draw_score);
        return GumbelSearchResult {
            best_move: None,
            value_q,
            value_cp: cp_from_q(value_q),
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
    let schedule = sequential_halving_schedule(
        tree.nodes[root].children_len as usize,
        tree.max_num_considered_actions,
        limits.simulations,
    );
    let mut last_progress = Instant::now();
    {
        crate::scope_profile!("az.search.simulations");
        for &considered_visit in &schedule {
            if control.is_some_and(GumbelSearchControl::should_stop) {
                break;
            }
            let child_index = tree.select_root_child(considered_visit);
            tree.simulate_child(root, child_index, 1);
            used += 1;
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

pub fn gumbel_search(
    position: &Position,
    model: &AzNnue,
    limits: GumbelSearchLimits,
) -> GumbelSearchResult {
    gumbel_search_with_rules(position, None, None, model, limits)
}

pub fn cp_from_q(q: f32) -> i32 {
    (q.clamp(-1.0, 1.0) * 1000.0).round() as i32
}

struct AzTree<'a> {
    nodes: Vec<AzNode>,
    children: Vec<AzChild>,
    accumulator_arena: Vec<f32>,
    model: &'a AzNnue,
    root_moves: Option<Vec<Move>>,
    root_raw_priors: Vec<f32>,
    root_gumbels: Vec<f32>,
    root: usize,
    max_num_considered_actions: usize,
    gumbel_scale: f32,
    q_value_scale: f32,
    q_maxvisit_init: f32,
    seed: u64,
    draw_score: f32,
    value_scale: f32,
    max_depth: usize,
    search_depth_sum: usize,
    search_depth_count: usize,
    search_depth_max: usize,
    search_depth_cutoffs: usize,
    eval_scratch: AzEvalScratch,
    root_rule_history: Vec<RuleHistoryEntry>,
    rule_history_scratch: Vec<RuleHistoryEntry>,
}

struct AzNode {
    position: Position,
    accumulator_offset: u32,
    parent: u32,
    rule_entry: Option<RuleHistoryEntry>,
    children_offset: u32,
    children_len: u16,
    visits: u32,
    value_wdl_sum: [f32; 3],
    value: f32,
    value_wdl: [f32; 3],
    moves_left: f32,
    expanded: bool,
}

#[derive(Clone)]
struct AzChild {
    mv: Move,
    prior: f32,
    visits: u32,
    value_wdl_sum: [f32; 3],
    moves_left_sum: f32,
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

    fn moves_left(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.moves_left_sum / self.visits as f32
        }
    }
}

impl<'a> AzTree<'a> {
    fn search_result(&self, simulations: usize) -> GumbelSearchResult {
        let root_node = &self.nodes[self.root];
        let root_children = self.node_children(self.root);
        let searched_wdl = if root_node.visits > 0 {
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
            .map(|(index, (child, policy))| GumbelCandidate {
                mv: child.mv,
                visits: child.visits,
                q: child.q(self.draw_score),
                moves_left: child.moves_left(),
                raw_prior: self
                    .root_raw_priors
                    .get(index)
                    .copied()
                    .unwrap_or(child.prior),
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
        let best_move = self
            .final_root_child()
            .map(|child_index| root_children[child_index].mv)
            .or_else(|| candidates.first().map(|candidate| candidate.mv));
        GumbelSearchResult {
            best_move,
            value_q: searched_value,
            value_cp: cp_from_q(searched_value),
            value_wdl: searched_wdl,
            simulations,
            search_depth_avg: self.search_depth_avg(),
            search_depth_max: self.search_depth_max,
            search_depth_limit: self.max_depth,
            search_depth_cutoffs: self.search_depth_cutoffs,
            candidates,
        }
    }

    fn new(
        position: Position,
        rule_history: Vec<RuleHistoryEntry>,
        root_moves: Option<Vec<Move>>,
        model: &'a AzNnue,
        limits: GumbelSearchLimits,
    ) -> Self {
        let mut nodes = Vec::with_capacity(limits.simulations.saturating_add(1).min(16_384));
        let accumulator = AzEvalAccumulator::new(model, &position);
        let mut accumulator_arena = Vec::with_capacity(
            limits
                .simulations
                .saturating_add(1)
                .saturating_mul(model.hidden_size * 2),
        );
        accumulator_arena.extend_from_slice(&accumulator.into_hidden_sum());
        nodes.push(AzNode {
            position,
            accumulator_offset: 0,
            parent: NO_CHILD,
            rule_entry: None,
            children_offset: 0,
            children_len: 0,
            visits: 0,
            value_wdl_sum: [0.0; 3],
            value: 0.0,
            value_wdl: [0.0, 1.0, 0.0],
            moves_left: 0.0,
            expanded: false,
        });
        Self {
            nodes,
            children: Vec::with_capacity(limits.simulations.saturating_mul(8)),
            accumulator_arena,
            model,
            root_moves,
            root_raw_priors: Vec::new(),
            root_gumbels: Vec::new(),
            root: 0,
            max_num_considered_actions: limits.max_num_considered_actions.max(1),
            gumbel_scale: limits.gumbel_scale.max(0.0),
            q_value_scale: limits.q_value_scale.max(0.0),
            q_maxvisit_init: limits.q_maxvisit_init.max(0.0),
            seed: limits.seed,
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
            eval_scratch: AzEvalScratch::new(model.arch),
            rule_history_scratch: Vec::with_capacity(rule_history.len().saturating_add(64)),
            root_rule_history: rule_history,
        }
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

    fn prepare_rule_history(&mut self, node_index: usize) {
        self.rule_history_scratch.clear();
        self.rule_history_scratch
            .extend_from_slice(&self.root_rule_history);
        let root_len = self.rule_history_scratch.len();
        let mut current = node_index;
        while current != self.root {
            let node = &self.nodes[current];
            if let Some(entry) = node.rule_entry {
                self.rule_history_scratch.push(entry);
            }
            current = node.parent as usize;
        }
        self.rule_history_scratch[root_len..].reverse();
    }

    fn expand(&mut self, node_index: usize) -> AzEvalOutput {
        crate::scope_profile!("az.search.expand");
        if self.nodes[node_index].expanded {
            return self.node_eval(node_index);
        }
        self.prepare_rule_history(node_index);

        let terminal = {
            crate::scope_profile!("az.search.terminal_value");
            terminal_value(&self.nodes[node_index].position, &self.rule_history_scratch)
        };
        if let Some(value) = terminal {
            let value_wdl = scalar_terminal_wdl(value);
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

        let moves = {
            crate::scope_profile!("az.search.expand_legal_moves");
            if node_index == self.root {
                self.root_moves.clone().unwrap_or_else(|| {
                    self.nodes[node_index]
                        .position
                        .legal_moves_with_rules(&self.rule_history_scratch)
                })
            } else {
                self.nodes[node_index]
                    .position
                    .legal_moves_with_rules(&self.rule_history_scratch)
            }
        };
        if moves.is_empty() {
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

        let mut eval = {
            crate::scope_profile!("az.search.nn_eval");
            let accumulator_start = self.nodes[node_index].accumulator_offset as usize;
            let accumulator_end = accumulator_start + self.model.hidden_size * 2;
            self.model.evaluate_incremental_with_scratch_output(
                &self.nodes[node_index].position,
                &self.accumulator_arena[accumulator_start..accumulator_end],
                &moves,
                &mut self.eval_scratch,
            )
        };
        eval.value_wdl = scale_wdl_value(eval.value_wdl, self.value_scale);
        eval.value *= self.value_scale;
        let priors = {
            crate::scope_profile!("az.search.softmax");
            softmax_into(
                &self.eval_scratch.logits[..moves.len()],
                &mut self.eval_scratch.priors,
            )
        };
        if node_index == self.root {
            self.root_raw_priors.clone_from(priors);
            self.root_gumbels = sample_gumbels(moves.len(), self.gumbel_scale, self.seed);
        }
        {
            crate::scope_profile!("az.search.children_build");
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
                            moves_left_sum: 0.0,
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
        self.nodes[node_index].moves_left = eval.moves_left;
        self.nodes[node_index].expanded = true;
        eval
    }

    fn simulate(&mut self, node_index: usize, depth: usize) -> AzEvalOutput {
        crate::scope_profile!("az.search.simulate");
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
        if self.nodes[node_index].children_len == 0 {
            let eval = self.node_eval(node_index);
            self.add_node_visit(node_index, eval);
            self.record_leaf_depth(depth, false);
            return eval;
        }
        let child_index = {
            crate::scope_profile!("az.search.select_child");
            self.select_interior_child(node_index)
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
                let parent_accumulator_start = self.nodes[node_index].accumulator_offset as usize;
                let parent_accumulator_end = parent_accumulator_start + self.model.hidden_size * 2;
                let child_accumulator_offset = self.accumulator_arena.len();
                self.accumulator_arena
                    .extend_from_within(parent_accumulator_start..parent_accumulator_end);
                let mover = child_position.side_to_move();
                {
                    crate::scope_profile!("az.search.child_make_move");
                    child_position.make_move(mv);
                }
                AzEvalAccumulator::apply_transition_to_hidden(
                    self.model,
                    &self.nodes[node_index].position,
                    &child_position,
                    mv,
                    moved,
                    captured,
                    &mut self.accumulator_arena[child_accumulator_offset
                        ..child_accumulator_offset + self.model.hidden_size * 2],
                );
                let child_rule_entry =
                    child_position.rule_history_entry_after_moved(mover, mv.to as usize);
                let child_node = self.nodes.len();
                self.nodes.push(AzNode {
                    position: child_position,
                    accumulator_offset: u32::try_from(child_accumulator_offset)
                        .expect("MCTS accumulator arena exceeds compact offset range"),
                    parent: node_index as u32,
                    rule_entry: Some(child_rule_entry),
                    children_offset: 0,
                    children_len: 0,
                    visits: 0,
                    value_wdl_sum: [0.0; 3],
                    value: 0.0,
                    value_wdl: [0.0, 1.0, 0.0],
                    moves_left: 0.0,
                    expanded: false,
                });
                self.node_children_mut(node_index)[child_index].set_child_node(child_node);
                child_node
            };
        let child_eval = self.simulate(child_node, child_depth);
        let eval = AzEvalOutput {
            value_wdl: flip_wdl(child_eval.value_wdl),
            value: -child_eval.value,
            // moves-left由子节点视角预测“从子节点到终局”的剩余步数。
            // 回传到父节点边时必须计入刚走的这一着；否则不同搜索深度的
            // 叶子会被直接混合，utility会错误偏爱搜索得更深的分支。
            moves_left: child_eval.moves_left + 1.0,
        };
        let child = &mut self.node_children_mut(node_index)[child_index];
        child.visits += 1;
        add_wdl(&mut child.value_wdl_sum, eval.value_wdl);
        child.moves_left_sum += eval.moves_left;
        self.add_node_visit(node_index, eval);
        eval
    }

    fn cutoff_value(&mut self, node_index: usize) -> AzEvalOutput {
        crate::scope_profile!("az.search.cutoff_value");
        if self.nodes[node_index].expanded {
            return self.node_eval(node_index);
        }
        self.prepare_rule_history(node_index);
        let terminal = {
            crate::scope_profile!("az.search.terminal_value");
            terminal_value(&self.nodes[node_index].position, &self.rule_history_scratch)
        };
        if let Some(value) = terminal {
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
        let moves = {
            crate::scope_profile!("az.search.expand_legal_moves");
            self.nodes[node_index]
                .position
                .legal_moves_with_rules(&self.rule_history_scratch)
        };
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
        let mut eval = {
            crate::scope_profile!("az.search.nn_eval");
            let accumulator_start = self.nodes[node_index].accumulator_offset as usize;
            let accumulator_end = accumulator_start + self.model.hidden_size * 2;
            self.model.evaluate_incremental_with_scratch_output(
                &self.nodes[node_index].position,
                &self.accumulator_arena[accumulator_start..accumulator_end],
                &moves,
                &mut self.eval_scratch,
            )
        };
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

    fn completed_qvalues(&self, node_index: usize) -> Vec<f32> {
        let children = self.node_children(node_index);
        let draw_score = self.node_draw_score(node_index);
        let raw_value = wdl_utility(self.nodes[node_index].value_wdl, draw_score);
        let total_visits = children.iter().map(|child| child.visits).sum::<u32>();
        let visited_prior = children
            .iter()
            .filter(|child| child.visits > 0)
            .map(|child| child.prior.max(0.0))
            .sum::<f32>();
        let weighted_q = if visited_prior > 0.0 {
            children
                .iter()
                .filter(|child| child.visits > 0)
                .map(|child| child.prior.max(0.0) * child.q(draw_score))
                .sum::<f32>()
                / visited_prior
        } else {
            raw_value
        };
        // Gumbel 搜索的 mixed value：未访问动作不能使用固定首访估值，
        // 而是由节点原始价值和已访问动作的先验加权 Q 共同补全。
        let mixed_value =
            (raw_value + total_visits as f32 * weighted_q) / (total_visits as f32 + 1.0);
        children
            .iter()
            .map(|child| {
                if child.visits > 0 {
                    child.q(draw_score)
                } else {
                    mixed_value
                }
            })
            .collect()
    }

    fn transformed_completed_qvalues(&self, node_index: usize) -> Vec<f32> {
        let children = self.node_children(node_index);
        let mut qvalues = self.completed_qvalues(node_index);
        if qvalues.is_empty() {
            return qvalues;
        }
        let min_q = qvalues.iter().copied().fold(f32::INFINITY, f32::min);
        let max_q = qvalues.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max_q - min_q;
        let scale = (self.q_maxvisit_init
            + children.iter().map(|child| child.visits).max().unwrap_or(0) as f32)
            * self.q_value_scale;
        for q in &mut qvalues {
            *q = if range > 1.0e-8 {
                (*q - min_q) / range * scale
            } else {
                0.0
            };
        }
        qvalues
    }

    fn improved_policy(&self, node_index: usize) -> Vec<f32> {
        let children = self.node_children(node_index);
        let transformed_q = self.transformed_completed_qvalues(node_index);
        let logits = children
            .iter()
            .zip(transformed_q)
            .map(|(child, q)| child.prior.max(1.0e-12).ln() + q)
            .collect::<Vec<_>>();
        softmax(&logits)
    }

    fn select_interior_child(&self, node_index: usize) -> usize {
        let children = self.node_children(node_index);
        let policy = self.improved_policy(node_index);
        let total_visits = children.iter().map(|child| child.visits).sum::<u32>() as f32;
        children
            .iter()
            .zip(policy)
            .enumerate()
            .max_by(
                |(left_index, (left, left_policy)), (right_index, (right, right_policy))| {
                    // 确定性的改进策略访问匹配；内部节点不再使用 PUCT。
                    let left_score = left_policy - left.visits as f32 / (1.0 + total_visits);
                    let right_score = right_policy - right.visits as f32 / (1.0 + total_visits);
                    left_score
                        .total_cmp(&right_score)
                        .then_with(|| right_index.cmp(left_index))
                },
            )
            .map(|(index, _)| index)
            .unwrap_or(0)
    }

    fn select_root_child(&self, considered_visit: u32) -> usize {
        let transformed_q = self.transformed_completed_qvalues(self.root);
        self.node_children(self.root)
            .iter()
            .enumerate()
            .filter(|(_, child)| child.visits == considered_visit)
            .max_by(|(left_index, left), (right_index, right)| {
                let left_score = self.root_gumbels[*left_index]
                    + left.prior.max(1.0e-12).ln()
                    + transformed_q[*left_index];
                let right_score = self.root_gumbels[*right_index]
                    + right.prior.max(1.0e-12).ln()
                    + transformed_q[*right_index];
                left_score
                    .total_cmp(&right_score)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .unwrap_or_else(|| self.final_root_child().unwrap_or(0))
    }

    fn final_root_child(&self) -> Option<usize> {
        let children = self.node_children(self.root);
        let max_visits = children.iter().map(|child| child.visits).max()?;
        let transformed_q = self.transformed_completed_qvalues(self.root);
        children
            .iter()
            .enumerate()
            .filter(|(_, child)| child.visits == max_visits)
            .max_by(|(left_index, left), (right_index, right)| {
                let left_score = self.root_gumbels[*left_index]
                    + left.prior.max(1.0e-12).ln()
                    + transformed_q[*left_index];
                let right_score = self.root_gumbels[*right_index]
                    + right.prior.max(1.0e-12).ln()
                    + transformed_q[*right_index];
                left_score
                    .total_cmp(&right_score)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
    }

    fn root_policy(&self, node_index: usize) -> Vec<f32> {
        let children = self.node_children(node_index);
        if children.is_empty() {
            Vec::new()
        } else {
            self.improved_policy(node_index)
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

fn softmax(logits: &[f32]) -> Vec<f32> {
    let mut output = Vec::with_capacity(logits.len());
    softmax_into(logits, &mut output);
    output
}

fn sample_gumbels(dim: usize, scale: f32, seed: u64) -> Vec<f32> {
    if scale <= 0.0 {
        return vec![0.0; dim];
    }
    let mut rng = SplitMix64::new(seed ^ 0x4755_4d42_454c_0000u64 ^ dim as u64);
    (0..dim)
        .map(|_| {
            let uniform = rng.unit_f32().clamp(1.0e-7, 1.0 - 1.0e-7);
            -(-uniform.ln()).ln() * scale
        })
        .collect()
}

fn sequential_halving_schedule(
    num_actions: usize,
    max_num_considered_actions: usize,
    simulations: usize,
) -> Vec<u32> {
    // 与 DeepMind MCTX get_sequence_of_considered_visits 相同的预算调度。
    let considered = num_actions.min(max_num_considered_actions.max(1));
    if considered == 0 || simulations == 0 {
        return Vec::new();
    }
    if considered == 1 {
        return (0..simulations as u32).collect();
    }
    let num_halving_rounds = (considered as f32).log2().ceil().max(1.0) as usize;
    let mut visits = vec![0u32; considered];
    let mut num_considered = considered;
    let mut schedule = Vec::with_capacity(simulations);
    while schedule.len() < simulations {
        let visits_per_stage = (simulations / (num_halving_rounds * num_considered)).max(1);
        for _ in 0..visits_per_stage {
            for visit in visits.iter_mut().take(num_considered) {
                if schedule.len() == simulations {
                    return schedule;
                }
                schedule.push(*visit);
                *visit += 1;
            }
        }
        num_considered = (num_considered / 2).max(2);
    }
    schedule
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xiangqi::{RuleDrawReason, RuleOutcome};

    #[test]
    fn policy_softmax_is_normalized() {
        let logits = [2.0, 0.0];
        let normal = softmax(&logits);

        assert!(normal[0] > normal[1]);
        assert!((normal.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn child_node_index_uses_compact_sentinel_representation() {
        assert!(std::mem::size_of::<AzChild>() <= 40);
        let mut child = AzChild {
            mv: Position::startpos().legal_moves()[0],
            prior: 1.0,
            visits: 0,
            value_wdl_sum: [0.0; 3],
            moves_left_sum: 0.0,
            child: NO_CHILD,
        };
        assert_eq!(child.child_node(), None);
        child.set_child_node(17);
        assert_eq!(child.child_node(), Some(17));
    }

    #[test]
    fn stopped_search_returns_root_result_without_running_simulations() {
        let stop = Arc::new(AtomicBool::new(true));
        let control = GumbelSearchControl::new(stop, None);
        let result = gumbel_search_with_rules_controlled(
            &Position::startpos(),
            None,
            None,
            &AzNnue::random(4, 19),
            GumbelSearchLimits {
                simulations: 128,
                ..GumbelSearchLimits::default()
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
            moves_left_sum: 0.0,
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
            GumbelSearchLimits {
                draw_score: 0.4,
                ..GumbelSearchLimits::default()
            },
        );

        tree.expand(tree.root);
        tree.simulate_child(tree.root, 0, 1);
        let child_node = tree.node_children(tree.root)[0].child_node().unwrap();
        assert!((tree.node_draw_score(tree.root) - 0.4).abs() < 1e-6);
        assert!((tree.node_draw_score(child_node) + 0.4).abs() < 1e-6);
    }

    #[test]
    fn gumbel_search_populates_improved_policy() {
        let model = AzNnue::random(4, 7);
        let result = gumbel_search(
            &Position::startpos(),
            &model,
            GumbelSearchLimits {
                simulations: 128,
                seed: 11,
                max_depth: 0,
                value_scale: 1.0,
                ..GumbelSearchLimits::default()
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
        let result = gumbel_search(
            &Position::startpos(),
            &model,
            GumbelSearchLimits {
                simulations: 32,
                seed: 13,
                max_depth: 1,
                value_scale: 1.0,
                ..GumbelSearchLimits::default()
            },
        );

        assert_eq!(result.simulations, 32);
        assert_eq!(result.search_depth_max, 1);
        assert_eq!(result.search_depth_limit, 1);
        assert!((result.search_depth_avg - 1.0).abs() < 1e-6);
        assert_eq!(result.search_depth_cutoffs, 32);
    }

    #[test]
    fn sequential_halving_initially_visits_considered_actions_once() {
        let schedule = sequential_halving_schedule(40, 16, 64);
        assert_eq!(&schedule[..16], &[0; 16]);
        assert_eq!(schedule.len(), 64);
    }

    #[test]
    fn interior_selection_favors_higher_improved_policy() {
        let model = AzNnue::random(4, 7);
        let position = Position::startpos();
        let legal = position.legal_moves();
        assert!(legal.len() >= 2);

        let mut tree = AzTree::new(
            position.clone(),
            position.initial_rule_history(),
            None,
            &model,
            GumbelSearchLimits {
                simulations: 1,
                seed: 31,
                max_depth: 0,
                value_scale: 1.0,
                ..GumbelSearchLimits::default()
            },
        );
        tree.set_node_children(
            tree.root,
            vec![
                AzChild {
                    mv: legal[0],
                    prior: 0.10,
                    visits: 1,
                    value_wdl_sum: [0.0, 1.0, 0.0],
                    moves_left_sum: 0.0,
                    child: NO_CHILD,
                },
                AzChild {
                    mv: legal[1],
                    prior: 0.90,
                    visits: 1,
                    value_wdl_sum: [0.0, 1.0, 0.0],
                    moves_left_sum: 0.0,
                    child: NO_CHILD,
                },
            ],
        );

        assert_eq!(tree.select_interior_child(tree.root), 1);
    }

    #[test]
    fn completed_q_uses_mixed_value_for_unvisited_actions() {
        let model = AzNnue::random(4, 7);
        let position = Position::startpos();
        let legal = position.legal_moves();
        let mut tree = AzTree::new(
            position.clone(),
            position.initial_rule_history(),
            None,
            &model,
            GumbelSearchLimits::default(),
        );
        tree.nodes[tree.root].value_wdl = [0.6, 0.0, 0.4];
        tree.set_node_children(
            tree.root,
            vec![
                AzChild {
                    mv: legal[0],
                    prior: 0.25,
                    visits: 2,
                    value_wdl_sum: [1.6, 0.0, 0.4],
                    moves_left_sum: 0.0,
                    child: NO_CHILD,
                },
                AzChild {
                    mv: legal[1],
                    prior: 0.75,
                    visits: 0,
                    value_wdl_sum: [0.0; 3],
                    moves_left_sum: 0.0,
                    child: NO_CHILD,
                },
            ],
        );

        let completed = tree.completed_qvalues(tree.root);
        assert!((completed[0] - 0.6).abs() < 1e-6);
        assert!((completed[1] - (0.2 + 2.0 * 0.6) / 3.0).abs() < 1e-6);
        let transformed = tree.transformed_completed_qvalues(tree.root);
        assert!((transformed[0] - 5.2).abs() < 1e-5);
        assert!(transformed[1].abs() < 1e-6);
    }

    #[test]
    fn search_value_scale_reduces_non_terminal_network_value() {
        let position = Position::startpos();
        let mut model = AzNnue::random(4, 7);
        model.value_head_bias[0] = 2.0;
        model.value_head_output[0] = 1.0;

        let full = gumbel_search(
            &position,
            &model,
            GumbelSearchLimits {
                simulations: 0,
                seed: 29,
                value_scale: 1.0,
                ..GumbelSearchLimits::default()
            },
        );
        let scaled = gumbel_search(
            &position,
            &model,
            GumbelSearchLimits {
                simulations: 0,
                seed: 29,
                value_scale: 0.25,
                ..GumbelSearchLimits::default()
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
            GumbelSearchLimits {
                simulations: 1,
                seed: 3,
                ..GumbelSearchLimits::default()
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
            GumbelSearchLimits::default(),
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
