use crate::nnue::HistoryMove;
use crate::xiangqi::{Color, Move, Piece, PieceKind, Position, RuleHistoryEntry, RuleOutcome};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use super::{AzEvalAccumulator, AzEvalOutput, AzEvalScratch, AzNnue, SplitMix64};

const DEFAULT_CPUCT: f32 = 1.5;
const DEFAULT_CPUCT_BASE: f32 = 19652.0;
const DEFAULT_CPUCT_FACTOR: f32 = 2.0;
const NO_CHILD: u32 = u32::MAX;
const SEARCH_PROGRESS_POLL_SIMULATIONS: usize = 64;
const SEARCH_PROGRESS_INTERVAL: Duration = Duration::from_millis(250);

const EMPTY_HISTORY_MOVE: HistoryMove = HistoryMove {
    piece: Piece {
        color: Color::Red,
        kind: PieceKind::General,
    },
    captured: None,
    mv: Move { from: 0, to: 0 },
};

#[derive(Clone, Copy, Debug)]
struct SearchHistory {
    entries: [HistoryMove; crate::nnue::HISTORY_PLIES],
    len: u8,
}

impl SearchHistory {
    fn from_slice(history: &[HistoryMove]) -> Self {
        let tail = history.len().saturating_sub(crate::nnue::HISTORY_PLIES);
        let history = &history[tail..];
        let mut out = Self {
            entries: [EMPTY_HISTORY_MOVE; crate::nnue::HISTORY_PLIES],
            len: history.len() as u8,
        };
        out.entries[..history.len()].copy_from_slice(history);
        out
    }

    fn as_slice(&self) -> &[HistoryMove] {
        &self.entries[..self.len as usize]
    }

    fn with_appended_move(&self, position: &Position, mv: Move) -> Self {
        let Some(piece) = position.piece_at(mv.from as usize) else {
            return *self;
        };
        let mut out = *self;
        let entry = HistoryMove {
            piece,
            captured: position.piece_at(mv.to as usize),
            mv,
        };
        let len = out.len as usize;
        if len < crate::nnue::HISTORY_PLIES {
            out.entries[len] = entry;
            out.len += 1;
        } else {
            out.entries.copy_within(1..crate::nnue::HISTORY_PLIES, 0);
            out.entries[crate::nnue::HISTORY_PLIES - 1] = entry;
        }
        out
    }
}

#[derive(Clone, Copy, Debug)]
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
    pub root_dirichlet_alpha: f32,
    pub root_exploration_fraction: f32,
    pub fpu_value: f32,
    pub fpu_value_at_root: f32,
    pub draw_score: f32,
    pub moves_left_max_effect: f32,
    pub moves_left_slope: f32,
    pub moves_left_threshold: f32,
    pub moves_left_constant_factor: f32,
    pub moves_left_scaled_factor: f32,
    pub moves_left_quadratic_factor: f32,
    pub value_scale: f32,
}

impl Default for AzSearchLimits {
    fn default() -> Self {
        Self {
            simulations: 10_000,
            seed: 0,
            cpuct: DEFAULT_CPUCT,
            cpuct_at_root: DEFAULT_CPUCT,
            cpuct_base: DEFAULT_CPUCT_BASE,
            cpuct_factor: DEFAULT_CPUCT_FACTOR,
            cpuct_base_at_root: DEFAULT_CPUCT_BASE,
            cpuct_factor_at_root: DEFAULT_CPUCT_FACTOR,
            max_depth: 0,
            root_dirichlet_alpha: 0.0,
            root_exploration_fraction: 0.0,
            fpu_value: 0.23,
            fpu_value_at_root: 1.0,
            draw_score: 0.0,
            moves_left_max_effect: 0.25,
            moves_left_slope: 0.002,
            moves_left_threshold: 0.6,
            moves_left_constant_factor: 0.0,
            moves_left_scaled_factor: 0.15,
            moves_left_quadratic_factor: 0.85,
            value_scale: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AzCandidate {
    pub mv: Move,
    pub visits: u32,
    pub q: f32,
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

pub fn alphazero_search_with_history_and_rules(
    position: &Position,
    history: &[HistoryMove],
    rule_history: Option<Vec<RuleHistoryEntry>>,
    root_moves: Option<Vec<Move>>,
    model: &AzNnue,
    limits: AzSearchLimits,
) -> AzSearchResult {
    alphazero_search_with_history_and_rules_controlled(
        position,
        history,
        rule_history,
        root_moves,
        model,
        limits,
        None,
    )
}

pub fn alphazero_search_with_history_and_rules_controlled(
    position: &Position,
    history: &[HistoryMove],
    rule_history: Option<Vec<RuleHistoryEntry>>,
    root_moves: Option<Vec<Move>>,
    model: &AzNnue,
    limits: AzSearchLimits,
    control: Option<&AzSearchControl>,
) -> AzSearchResult {
    alphazero_search_with_history_and_rules_controlled_with_progress(
        position,
        history,
        rule_history,
        root_moves,
        model,
        limits,
        control,
        None,
    )
}

pub fn alphazero_search_with_history_and_rules_controlled_with_progress(
    position: &Position,
    history: &[HistoryMove],
    rule_history: Option<Vec<RuleHistoryEntry>>,
    root_moves: Option<Vec<Move>>,
    model: &AzNnue,
    limits: AzSearchLimits,
    control: Option<&AzSearchControl>,
    mut progress: Option<&mut dyn FnMut(&AzSearchResult)>,
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
    {
        crate::scope_profile!("az.search.root_expand");
        tree.expand(root);
    }
    if tree.nodes[root].children.is_empty() {
        let value_q = wdl_utility(tree.nodes[root].value_wdl, tree.draw_score);
        return AzSearchResult {
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
    let mut last_progress = Instant::now();
    {
        crate::scope_profile!("az.search.simulations");
        for _ in 0..limits.simulations {
            if control.is_some_and(AzSearchControl::should_stop) {
                break;
            }
            tree.simulate(root, 0);
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

pub fn alphazero_search(
    position: &Position,
    model: &AzNnue,
    limits: AzSearchLimits,
) -> AzSearchResult {
    alphazero_search_with_history_and_rules(position, &[], None, None, model, limits)
}

pub fn cp_from_q(q: f32) -> i32 {
    (q.clamp(-1.0, 1.0) * 1000.0).round() as i32
}

struct AzTree<'a> {
    nodes: Vec<AzNode>,
    model: &'a AzNnue,
    root_moves: Option<Vec<Move>>,
    root: usize,
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
    draw_score: f32,
    moves_left_max_effect: f32,
    moves_left_slope: f32,
    moves_left_threshold: f32,
    moves_left_constant_factor: f32,
    moves_left_scaled_factor: f32,
    moves_left_quadratic_factor: f32,
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
    accumulator: AzEvalAccumulator,
    history: SearchHistory,
    rule_history: Vec<RuleHistoryEntry>,
    children: Vec<AzChild>,
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
    raw_prior: f32,
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
    fn search_result(&self, simulations: usize) -> AzSearchResult {
        let root_node = &self.nodes[self.root];
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
        let mut candidates = root_node
            .children
            .iter()
            .zip(policy)
            .map(|(child, policy)| AzCandidate {
                mv: child.mv,
                visits: child.visits,
                q: child.q(self.draw_score),
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
        let best_move = self
            .best_root_child(self.root)
            .map(|child_index| root_node.children[child_index].mv)
            .or_else(|| candidates.first().map(|candidate| candidate.mv));
        AzSearchResult {
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
        history: SearchHistory,
        rule_history: Vec<RuleHistoryEntry>,
        root_moves: Option<Vec<Move>>,
        model: &'a AzNnue,
        limits: AzSearchLimits,
    ) -> Self {
        let mut nodes = Vec::with_capacity(limits.simulations.saturating_add(1).min(16_384));
        let accumulator = AzEvalAccumulator::new(model, &position);
        nodes.push(AzNode {
            position,
            accumulator,
            history,
            rule_history,
            children: Vec::new(),
            visits: 0,
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
            fpu_value_at_root: limits.fpu_value_at_root.clamp(-1.0, 1.0),
            draw_score: limits.draw_score.clamp(-1.0, 1.0),
            moves_left_max_effect: limits.moves_left_max_effect.max(0.0),
            moves_left_slope: limits.moves_left_slope.max(0.0),
            moves_left_threshold: limits.moves_left_threshold.clamp(0.0, 1.0),
            moves_left_constant_factor: limits.moves_left_constant_factor,
            moves_left_scaled_factor: limits.moves_left_scaled_factor,
            moves_left_quadratic_factor: limits.moves_left_quadratic_factor,
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

    fn expand(&mut self, node_index: usize) -> AzEvalOutput {
        crate::scope_profile!("az.search.expand");
        if self.nodes[node_index].expanded {
            return self.node_eval(node_index);
        }

        let terminal = {
            crate::scope_profile!("az.search.terminal_value");
            terminal_value(
                &self.nodes[node_index].position,
                &self.nodes[node_index].rule_history,
            )
        };
        if let Some(value) = terminal {
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

        let moves = {
            crate::scope_profile!("az.search.expand_legal_moves");
            if node_index == self.root {
                self.root_moves.clone().unwrap_or_else(|| {
                    self.nodes[node_index]
                        .position
                        .legal_moves_with_rules(&self.nodes[node_index].rule_history)
                })
            } else {
                self.nodes[node_index]
                    .position
                    .legal_moves_with_rules(&self.nodes[node_index].rule_history)
            }
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

        let mut eval = {
            crate::scope_profile!("az.search.nn_eval");
            self.model.evaluate_incremental_with_scratch_output(
                &self.nodes[node_index].position,
                &self.nodes[node_index].accumulator,
                self.nodes[node_index].history.as_slice(),
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
        {
            crate::scope_profile!("az.search.children_build");
            self.nodes[node_index].children = moves
                .into_iter()
                .zip(priors.drain(..))
                .zip(raw_priors)
                .map(|((mv, prior), raw_prior)| AzChild {
                    mv,
                    raw_prior,
                    prior,
                    visits: 0,
                    value_wdl_sum: [0.0; 3],
                    moves_left_sum: 0.0,
                    child: NO_CHILD,
                })
                .collect();
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
        if self.nodes[node_index].children.is_empty() {
            let eval = self.node_eval(node_index);
            self.add_node_visit(node_index, eval);
            self.record_leaf_depth(depth, false);
            return eval;
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
            if let Some(child_node) = self.nodes[node_index].children[child_index].child_node() {
                child_node
            } else {
                crate::scope_profile!("az.search.create_child");
                let mv = self.nodes[node_index].children[child_index].mv;
                let mut child_position = self.nodes[node_index].position.clone();
                let moved = child_position.piece_at(mv.from as usize).unwrap();
                let captured = child_position.piece_at(mv.to as usize);
                let mut child_accumulator = self.nodes[node_index].accumulator.clone();
                let child_history = {
                    crate::scope_profile!("az.search.clone_history");
                    self.nodes[node_index]
                        .history
                        .with_appended_move(&child_position, mv)
                };
                let mover = child_position.side_to_move();
                {
                    crate::scope_profile!("az.search.child_make_move");
                    child_position.make_move(mv);
                }
                child_accumulator.apply_transition(
                    self.model,
                    &self.nodes[node_index].position,
                    &child_position,
                    mv,
                    moved,
                    captured,
                );
                let child_rule_entry =
                    child_position.rule_history_entry_after_moved(mover, mv.to as usize);
                let child_rule_history = {
                    crate::scope_profile!("az.search.clone_rule_history");
                    clone_rule_history_with_appended_entry(
                        &self.nodes[node_index].rule_history,
                        child_rule_entry,
                    )
                };
                let child_node = self.nodes.len();
                self.nodes.push(AzNode {
                    position: child_position,
                    accumulator: child_accumulator,
                    history: child_history,
                    rule_history: child_rule_history,
                    children: Vec::new(),
                    visits: 0,
                    value_wdl_sum: [0.0; 3],
                    value: 0.0,
                    value_wdl: [0.0, 1.0, 0.0],
                    moves_left: 0.0,
                    expanded: false,
                });
                self.nodes[node_index].children[child_index].set_child_node(child_node);
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
        let child = &mut self.nodes[node_index].children[child_index];
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
        let terminal = {
            crate::scope_profile!("az.search.terminal_value");
            terminal_value(
                &self.nodes[node_index].position,
                &self.nodes[node_index].rule_history,
            )
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
                .legal_moves_with_rules(&self.nodes[node_index].rule_history)
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
            self.model.evaluate_incremental_with_scratch_output(
                &self.nodes[node_index].position,
                &self.nodes[node_index].accumulator,
                self.nodes[node_index].history.as_slice(),
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

    fn select_child(&self, node_index: usize) -> usize {
        let node = &self.nodes[node_index];
        let parent_visits_sqrt = (node.visits.max(1) as f32).sqrt();
        let is_root = node_index == self.root;
        let fpu_value = if is_root {
            self.fpu_value_at_root
        } else {
            alphazero_fpu_value_reduction(node, self.fpu_value, self.node_draw_score(node_index))
        };
        let cpuct = self.compute_cpuct(node.visits, is_root);
        self.nodes[node_index]
            .children
            .iter()
            .enumerate()
            .max_by(|(left_index, left_child), (right_index, right_child)| {
                let left_score = self.child_score(
                    node_index,
                    node,
                    left_child,
                    fpu_value,
                    parent_visits_sqrt,
                    cpuct,
                );
                let right_score = self.child_score(
                    node_index,
                    node,
                    right_child,
                    fpu_value,
                    parent_visits_sqrt,
                    cpuct,
                );
                left_score
                    .total_cmp(&right_score)
                    .then_with(|| left_child.prior.total_cmp(&right_child.prior))
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
                    .then_with(|| {
                        left_child
                            .q(self.node_draw_score(node_index))
                            .total_cmp(&right_child.q(self.node_draw_score(node_index)))
                    })
                    .then_with(|| left_child.prior.total_cmp(&right_child.prior))
                    .then_with(|| right_index.cmp(left_index))
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

    fn child_score(
        &self,
        node_index: usize,
        parent: &AzNode,
        child: &AzChild,
        fpu_value: f32,
        parent_visits_sqrt: f32,
        cpuct: f32,
    ) -> f32 {
        let q = if child.visits > 0 {
            child.q(self.node_draw_score(node_index))
        } else {
            fpu_value
        };
        let u = cpuct * child.prior * parent_visits_sqrt / (1.0 + child.visits as f32);
        q + u + self.moves_left_utility(parent, child, q)
    }

    fn moves_left_utility(&self, parent: &AzNode, child: &AzChild, q: f32) -> f32 {
        if self.moves_left_slope <= 0.0 || self.moves_left_max_effect <= 0.0 {
            return 0.0;
        }
        if q.abs() <= self.moves_left_threshold {
            return 0.0;
        }
        let child_m = if child.visits == 0 {
            parent.moves_left
        } else {
            child.moves_left()
        };
        let mut effect = self.moves_left_slope * (child_m - parent.moves_left);
        effect = effect.clamp(-self.moves_left_max_effect, self.moves_left_max_effect);
        effect *= -q.signum();

        let q_abs = if self.moves_left_threshold > 0.0 && self.moves_left_threshold < 1.0 {
            ((q.abs() - self.moves_left_threshold) / (1.0 - self.moves_left_threshold))
                .clamp(0.0, 1.0)
        } else {
            q.abs()
        };
        let weight = self.moves_left_constant_factor
            + self.moves_left_scaled_factor * q_abs
            + self.moves_left_quadratic_factor * q_abs * q_abs;
        effect * weight
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

fn wdl_utility(wdl: [f32; 3], draw_score: f32) -> f32 {
    (wdl[0] - wdl[2] + draw_score * wdl[1]).clamp(-1.0, 1.0)
}

fn wdl_sum_utility(wdl_sum: [f32; 3], visits: u32, draw_score: f32) -> f32 {
    if visits == 0 {
        return 0.0;
    }
    wdl_utility(wdl_sum.map(|part| part / visits as f32), draw_score)
}

fn alphazero_fpu_value_reduction(node: &AzNode, reduction: f32, draw_score: f32) -> f32 {
    let parent_q = if node.visits > 0 {
        wdl_sum_utility(node.value_wdl_sum, node.visits, draw_score)
    } else {
        wdl_utility(node.value_wdl, draw_score)
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

fn clone_rule_history_with_appended_entry(
    rule_history: &[RuleHistoryEntry],
    entry: RuleHistoryEntry,
) -> Vec<RuleHistoryEntry> {
    let mut out = Vec::with_capacity(rule_history.len() + 1);
    out.extend_from_slice(rule_history);
    out.push(entry);
    out
}

fn truncate_history(history: &[HistoryMove]) -> SearchHistory {
    SearchHistory::from_slice(history)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xiangqi::{RuleDrawReason, RuleOutcome};

    #[test]
    fn inline_search_history_matches_existing_history_updates() {
        let mut position = Position::startpos();
        let mut expected = Vec::new();
        let mut actual = SearchHistory::from_slice(&expected);

        for _ in 0..(crate::nnue::HISTORY_PLIES + 4) {
            let mv = position.legal_moves()[0];
            append_history(&mut expected, &position, mv);
            actual = actual.with_appended_move(&position, mv);
            assert_eq!(actual.as_slice(), expected.as_slice());
            position.make_move(mv);
        }
    }

    #[test]
    fn child_node_index_uses_compact_sentinel_representation() {
        assert!(std::mem::size_of::<AzChild>() <= 40);
        let mut child = AzChild {
            mv: Position::startpos().legal_moves()[0],
            raw_prior: 1.0,
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
        let control = AzSearchControl::new(stop, None);
        let result = alphazero_search_with_history_and_rules_controlled(
            &Position::startpos(),
            &[],
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
            raw_prior: 1.0,
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
            SearchHistory::from_slice(&[]),
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
        let child_node = tree.nodes[tree.root].children[0].child_node().unwrap();
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
                fpu_value: 0.23,
                fpu_value_at_root: 1.0,
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
                fpu_value: 0.23,
                fpu_value_at_root: 1.0,
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
                root_dirichlet_alpha: 0.3,
                root_exploration_fraction: 0.25,
                fpu_value: 0.23,
                fpu_value_at_root: 1.0,
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
            SearchHistory::from_slice(&[]),
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
                fpu_value: 0.23,
                fpu_value_at_root: 1.0,
                value_scale: 1.0,
                ..AzSearchLimits::default()
            },
        );
        tree.cpuct_at_root = 0.0;
        tree.nodes[tree.root].children = vec![
            AzChild {
                mv: legal[0],
                raw_prior: 0.10,
                prior: 0.10,
                visits: 1,
                value_wdl_sum: [0.0, 1.0, 0.0],
                moves_left_sum: 0.0,
                child: NO_CHILD,
            },
            AzChild {
                mv: legal[1],
                raw_prior: 0.90,
                prior: 0.90,
                visits: 1,
                value_wdl_sum: [0.0, 1.0, 0.0],
                moves_left_sum: 0.0,
                child: NO_CHILD,
            },
        ];

        assert_eq!(tree.select_child(tree.root), 1);
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
            SearchHistory::from_slice(&[]),
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
        let child_node = tree.nodes[tree.root].children[0].child_node().unwrap();
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
            SearchHistory::from_slice(&[]),
            Position::startpos().initial_rule_history(),
            Some(root_moves.clone()),
            &model,
            AzSearchLimits::default(),
        );

        tree.expand(tree.root);
        assert_eq!(tree.nodes[tree.root].children.len(), 1);
        let child_index = 0;
        tree.simulate_child(tree.root, child_index, 1);
        let child_node = tree.nodes[tree.root].children[child_index]
            .child_node()
            .unwrap();
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
