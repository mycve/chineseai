//! Bounded iterative deepening negamax for NNUE self-play and promotion matches.
use super::pikafish_candle::{PikafishCpuCache, PikafishCpuModel};
use super::{AbEvalAccumulator, AbEvalScratch, AbNnue};
use crate::nnue::pikafish_file::{PikafishEvalCache, PikafishNet};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub struct AbSearchLimits {
    pub nodes: usize,
    pub max_depth: usize,
}
impl Default for AbSearchLimits {
    fn default() -> Self {
        Self {
            nodes: 10_000,
            max_depth: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AbCandidate {
    pub mv: Move,
    pub q: f32,
    pub selection_weight: f32,
    pub solved: Option<i8>,
}
impl AbCandidate {
    pub(crate) fn proof_priority(&self) -> u8 {
        match self.solved {
            Some(1) => 2,
            Some(-1) => 0,
            _ => 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AbSearchResult {
    pub best_move: Option<Move>,
    pub value_q: f32,
    pub value_cp: i32,
    pub value_wdl: [f32; 3],
    pub network_value_wdl: [f32; 3],
    pub best_value_wdl: [f32; 3],
    pub nodes: usize,
    pub quiescence_nodes: usize,
    pub search_depth_avg: f32,
    pub search_depth_max: usize,
    /// Deepest visited ply, including quiescence; UCI `seldepth` is ply + 1.
    pub selective_depth: usize,
    pub search_depth_limit: usize,
    pub search_depth_cutoffs: usize,
    pub candidates: Vec<AbCandidate>,
}

#[derive(Clone, Debug)]
pub struct AbSearchControl {
    stop: Arc<AtomicBool>,
    deadline: Option<Instant>,
}
impl AbSearchControl {
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

pub fn cp_from_q(q: f32) -> i32 {
    (q.clamp(-1.0, 1.0) * 1000.0).round() as i32
}

pub(crate) struct AbUciPv {
    pub moves: Vec<Move>,
    pub wdl: [f32; 3],
    pub q: f32,
    pub proven: Option<i8>,
}

pub(crate) struct AbUciSearchResult {
    pub search: AbSearchResult,
    pub variations: Vec<AbUciPv>,
}
use crate::xiangqi::{Color, Move, PieceKind, Position, RuleHistoryEntry, RuleOutcome};

const MATE: f32 = 1.0;
const MAX_STATIC_SCORE: f32 = 0.95;
const MATE_PLY_PENALTY: f32 = 0.001;
const TT_SIZE: usize = 1 << 15;

#[derive(Clone, Copy)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

struct TtEntry {
    key: u64,
    history: Vec<RuleHistoryEntry>,
    depth: usize,
    value: f32,
    bound: Bound,
    best_move: Option<Move>,
}

// 固定哈希槽保持原有碰撞语义；仅给访问过的槽分配条目，并在线程内复用。
struct SparseTt {
    slots: Box<[u16]>,
    entries: Vec<Option<TtEntry>>,
    used: usize,
}
impl SparseTt {
    fn new() -> Self {
        Self {
            slots: vec![u16::MAX; TT_SIZE].into_boxed_slice(),
            entries: Vec::new(),
            used: 0,
        }
    }
    fn reset(&mut self) {
        for entry in self.entries[..self.used].iter().flatten() {
            self.slots[entry.key as usize & (TT_SIZE - 1)] = u16::MAX;
        }
        self.used = 0;
    }
}
thread_local! {static SEARCH_TT: std::cell::RefCell<Option<SparseTt>> = const {std::cell::RefCell::new(None)};}
struct TtLease(Option<SparseTt>);
impl TtLease {
    fn new() -> Self {
        let mut tt = SEARCH_TT
            .with(|cached| cached.borrow_mut().take())
            .unwrap_or_else(SparseTt::new);
        tt.reset();
        Self(Some(tt))
    }
}
impl std::ops::Index<usize> for TtLease {
    type Output = Option<TtEntry>;
    fn index(&self, slot: usize) -> &Self::Output {
        let tt = self.0.as_ref().unwrap();
        let index = tt.slots[slot];
        if index == u16::MAX {
            &None
        } else {
            &tt.entries[index as usize]
        }
    }
}
impl std::ops::IndexMut<usize> for TtLease {
    fn index_mut(&mut self, slot: usize) -> &mut Self::Output {
        let tt = self.0.as_mut().unwrap();
        let mut index = tt.slots[slot];
        if index == u16::MAX {
            index = tt.used as u16;
            tt.used += 1;
            tt.slots[slot] = index;
            if tt.entries.len() < tt.used {
                tt.entries.push(None);
            }
        }
        &mut tt.entries[index as usize]
    }
}
impl Drop for TtLease {
    fn drop(&mut self) {
        let tt = self.0.take();
        let _ = SEARCH_TT.try_with(|cached| *cached.borrow_mut() = tt);
    }
}

trait ValueModel {
    fn pikafish_cpu(&self) -> Option<&PikafishCpuModel> {
        None
    }
    fn pikafish_net(&self) -> Option<&PikafishNet> {
        None
    }
    fn root_hidden(&self, position: &Position) -> Vec<f32>;
    fn transition(
        &self,
        before_buckets: [(usize, usize); 2],
        after: &Position,
        mv: Move,
        moved: crate::xiangqi::Piece,
        captured: Option<crate::xiangqi::Piece>,
        hidden: &mut [f32],
    );
    fn evaluate(
        &self,
        position: &Position,
        history: &[RuleHistoryEntry],
        hidden: &[f32],
        scratch: &mut Option<AbEvalScratch>,
    ) -> Result<f32, String>;
    fn root_wdl(
        &self,
        position: &Position,
        history: &[RuleHistoryEntry],
    ) -> Result<[f32; 3], String>;
    fn scratch(&self) -> Option<AbEvalScratch>;
}

impl ValueModel for AbNnue {
    fn root_hidden(&self, position: &Position) -> Vec<f32> {
        AbEvalAccumulator::new(self, position).into_hidden_sum()
    }
    fn transition(
        &self,
        before_buckets: [(usize, usize); 2],
        after: &Position,
        mv: Move,
        moved: crate::xiangqi::Piece,
        captured: Option<crate::xiangqi::Piece>,
        hidden: &mut [f32],
    ) {
        AbEvalAccumulator::apply_transition_from_buckets(
            self,
            before_buckets,
            after,
            mv,
            moved,
            captured,
            hidden,
        );
    }
    fn evaluate(
        &self,
        position: &Position,
        history: &[RuleHistoryEntry],
        hidden: &[f32],
        scratch: &mut Option<AbEvalScratch>,
    ) -> Result<f32, String> {
        Ok(self.evaluate_incremental_value_with_rules(
            position,
            history,
            hidden,
            scratch.as_mut().unwrap(),
        ))
    }
    fn root_wdl(
        &self,
        position: &Position,
        history: &[RuleHistoryEntry],
    ) -> Result<[f32; 3], String> {
        Ok(self.evaluate_wdl_with_rules(position, history))
    }
    fn scratch(&self) -> Option<AbEvalScratch> {
        Some(AbEvalScratch::new(self.arch))
    }
}

impl ValueModel for PikafishCpuModel {
    fn pikafish_cpu(&self) -> Option<&PikafishCpuModel> {
        Some(self)
    }
    fn root_hidden(&self, _position: &Position) -> Vec<f32> {
        Vec::new()
    }
    fn transition(
        &self,
        _before_buckets: [(usize, usize); 2],
        _after: &Position,
        _mv: Move,
        _moved: crate::xiangqi::Piece,
        _captured: Option<crate::xiangqi::Piece>,
        _hidden: &mut [f32],
    ) {
    }
    fn evaluate(
        &self,
        position: &Position,
        _history: &[RuleHistoryEntry],
        _hidden: &[f32],
        _scratch: &mut Option<AbEvalScratch>,
    ) -> Result<f32, String> {
        self.evaluate(position, &mut PikafishCpuCache::default())
            .map_err(|error| error.to_string())
    }
    fn root_wdl(
        &self,
        position: &Position,
        history: &[RuleHistoryEntry],
    ) -> Result<[f32; 3], String> {
        let q = ValueModel::evaluate(self, position, history, &[], &mut None)?
            .clamp(-MAX_STATIC_SCORE, MAX_STATIC_SCORE);
        Ok([(1.0 + q) * 0.5, 0.0, (1.0 - q) * 0.5])
    }
    fn scratch(&self) -> Option<AbEvalScratch> {
        None
    }
}

impl ValueModel for PikafishNet {
    fn pikafish_net(&self) -> Option<&PikafishNet> {
        Some(self)
    }
    fn root_hidden(&self, _position: &Position) -> Vec<f32> {
        Vec::new()
    }
    fn transition(
        &self,
        _before_buckets: [(usize, usize); 2],
        _after: &Position,
        _mv: Move,
        _moved: crate::xiangqi::Piece,
        _captured: Option<crate::xiangqi::Piece>,
        _hidden: &mut [f32],
    ) {
    }
    fn evaluate(
        &self,
        position: &Position,
        history: &[RuleHistoryEntry],
        _hidden: &[f32],
        _scratch: &mut Option<AbEvalScratch>,
    ) -> Result<f32, String> {
        Ok(
            (self.evaluate_scaled(position, position.rule60_count_with_history(history))? as f32
                / 600.0)
                .tanh(),
        )
    }
    fn root_wdl(
        &self,
        position: &Position,
        history: &[RuleHistoryEntry],
    ) -> Result<[f32; 3], String> {
        let q = ValueModel::evaluate(self, position, history, &[], &mut None)?;
        Ok([(1.0 + q) * 0.5, 0.0, (1.0 - q) * 0.5])
    }
    fn scratch(&self) -> Option<AbEvalScratch> {
        None
    }
}

struct Search<'a, M: ValueModel + ?Sized> {
    model: &'a M,
    nodes: usize,
    limit: usize,
    exhausted: bool,
    quiescence_nodes: usize,
    selective_depth: usize,
    history_scores: [[i32; 90]; 90],
    killers: Vec<[Option<Move>; 2]>,
    tt: TtLease,
    hidden_pool: Vec<Vec<f32>>,
    scratch: Option<AbEvalScratch>,
    pikafish_cache: Option<PikafishEvalCache>,
    pikafish_cpu_cache: Option<PikafishCpuCache>,
    error: Option<String>,
    control: Option<&'a AbSearchControl>,
}

impl<M: ValueModel + ?Sized> Search<'_, M> {
    fn evaluate(
        &mut self,
        position: &Position,
        history: &[RuleHistoryEntry],
        hidden: &[f32],
        ply: usize,
    ) -> f32 {
        let result = if let (Some(model), Some(cache)) =
            (self.model.pikafish_cpu(), self.pikafish_cpu_cache.as_mut())
        {
            model
                .evaluate_search(
                    position,
                    cache,
                    ply,
                    history.iter().rev().nth(1).map(|entry| entry.hash),
                )
                .map_err(|e| e.to_string())
        } else if let (Some(net), Some(cache)) =
            (self.model.pikafish_net(), self.pikafish_cache.as_mut())
        {
            cache
                .evaluate_scaled(net, position, position.rule60_count_with_history(history))
                .map(|cp| (cp as f32 / 600.0).tanh())
        } else {
            self.model
                .evaluate(position, history, hidden, &mut self.scratch)
        };
        match result {
            Ok(value) => value.clamp(-MAX_STATIC_SCORE, MAX_STATIC_SCORE),
            Err(error) => {
                self.error = Some(error);
                self.exhausted = true;
                0.0
            }
        }
    }

    fn quiescence(
        &mut self,
        position: &mut Position,
        history: &mut Vec<RuleHistoryEntry>,
        hidden: &[f32],
        remaining: usize,
        ply: usize,
        mut alpha: f32,
        beta: f32,
    ) -> f32 {
        if self.nodes >= self.limit || self.control.is_some_and(AbSearchControl::should_stop) {
            self.exhausted = true;
            return 0.0;
        }
        self.nodes += 1;
        self.quiescence_nodes += 1;
        self.selective_depth = self.selective_depth.max(ply);
        if let Some(outcome) = position.rule_outcome_with_history(history) {
            return terminal_value(outcome, position.side_to_move(), ply);
        }
        let checked = position.in_check(position.side_to_move());
        let mut tactical = if checked {
            position.legal_moves_with_rules(history)
        } else {
            // A capture changes the board material, so it cannot repeat a
            // previous position. Avoid generating and rule-checking every
            // quiet move at each quiescence node.
            position.legal_capture_moves()
        };
        if tactical.is_empty() && (checked || position.legal_moves_with_rules(history).is_empty()) {
            return mated_at(ply);
        }
        let stand_pat = if checked {
            -2.0
        } else {
            self.evaluate(position, history, hidden, ply)
        };
        // A checked position has no legal stand-pat score. Keep searching evasions
        // even when the ordinary capture horizon has been reached.
        if remaining == 0 && !checked {
            return stand_pat;
        }
        if !checked {
            if stand_pat >= beta {
                return stand_pat;
            }
            alpha = alpha.max(stand_pat);
        }
        tactical.sort_by_key(|mv| {
            std::cmp::Reverse(match position.piece_at(mv.to as usize) {
                Some(victim) => {
                    let attacker = position.piece_at(mv.from as usize).unwrap();
                    100 * piece_value(victim.kind) - piece_value(attacker.kind)
                }
                None => i64::MIN,
            })
        });
        let mut best = if checked { -2.0 } else { stand_pat };
        let recapture_square = history.last().and_then(|entry| entry.mv.map(|mv| mv.to));
        for mv in tactical {
            if !checked {
                let attacker = position.piece_at(mv.from as usize).unwrap();
                let victim = position.piece_at(mv.to as usize).unwrap();
                let gives_check = position.gives_check_after_move_fast(mv);
                // A generous material upper bound. Recaptures and checks are
                // exempt because they can change the tactical sequence even
                // when their immediate material gain looks small.
                if q_capture_prunable(
                    stand_pat,
                    alpha,
                    piece_value(victim.kind),
                    gives_check,
                    recapture_square == Some(mv.to),
                ) {
                    continue;
                }
                // Pikafish qsearch discards losing captures below its SEE
                // threshold. Run the slower legal-exchange search only when
                // the captured piece cannot already pay for the attacker.
                if victim.kind != PieceKind::General
                    && piece_value(attacker.kind) > piece_value(victim.kind) + 106
                {
                    let (gain, _) = static_exchange_gain(position, mv);
                    if gain < -106 && !gives_check && recapture_square != Some(mv.to) {
                        continue;
                    }
                }
            }
            let before_buckets = self
                .model
                .pikafish_net()
                .is_none()
                .then(|| AbEvalAccumulator::buckets_for_position(position));
            let mover = position.side_to_move();
            let moved = position.piece_at(mv.from as usize).unwrap();
            let captured = position.piece_at(mv.to as usize);
            let undo = position.make_move(mv);
            let mut child_hidden = Vec::new();
            if let Some(before_buckets) = before_buckets {
                child_hidden = self.hidden_pool.pop().unwrap_or_default();
                child_hidden.resize(hidden.len(), 0.0);
                child_hidden.copy_from_slice(hidden);
                self.model.transition(
                    before_buckets,
                    position,
                    mv,
                    moved,
                    captured,
                    &mut child_hidden,
                );
            }
            history.push(position.rule_history_entry_after_moved(mover, mv, captured));
            let score = -self.quiescence(
                position,
                history,
                &child_hidden,
                remaining.saturating_sub(1),
                ply + 1,
                -beta,
                -alpha,
            );
            history.pop();
            position.unmake_move(mv, undo);
            if before_buckets.is_some() {
                self.hidden_pool.push(child_hidden);
            }
            if self.exhausted {
                return 0.0;
            }
            best = best.max(score);
            alpha = alpha.max(best);
            if alpha >= beta {
                break;
            }
        }
        best
    }

    fn negamax(
        &mut self,
        position: &mut Position,
        history: &mut Vec<RuleHistoryEntry>,
        hidden: &[f32],
        depth: usize,
        ply: usize,
        alpha: f32,
        beta: f32,
    ) -> f32 {
        self.negamax_inner(
            position, history, hidden, depth, ply, alpha, beta, false, true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn negamax_inner(
        &mut self,
        position: &mut Position,
        history: &mut Vec<RuleHistoryEntry>,
        hidden: &[f32],
        depth: usize,
        ply: usize,
        mut alpha: f32,
        beta: f32,
        previous_was_null: bool,
        null_enabled: bool,
    ) -> f32 {
        if depth == 0 {
            // The frontier is one quiescence node, not a negamax node plus a
            // second quiescence node at the same position.
            return self.quiescence(position, history, hidden, 6, ply, alpha, beta);
        }
        if self.nodes >= self.limit || self.control.is_some_and(AbSearchControl::should_stop) {
            self.exhausted = true;
            return 0.0;
        }
        self.nodes += 1;
        self.selective_depth = self.selective_depth.max(ply);
        if let Some(outcome) = position.rule_outcome_with_history(history) {
            return terminal_value(outcome, position.side_to_move(), ply);
        }
        // Chinese perpetual-check/chase adjudication depends on the whole path.
        // A board hash alone is never sufficient for a score cutoff.
        let slot = (position.hash() as usize) & (TT_SIZE - 1);
        let cached = self.tt[slot].as_ref().filter(|entry| {
            entry.key == position.hash() && entry.history.as_slice() == history.as_slice()
        });
        let tt_move = cached.and_then(|entry| entry.best_move);
        if let Some(entry) = cached.filter(|entry| null_enabled && entry.depth >= depth) {
            match entry.bound {
                Bound::Exact => return entry.value,
                Bound::Lower if entry.value >= beta => return entry.value,
                Bound::Upper if entry.value <= alpha => return entry.value,
                _ => {}
            }
        }
        let mut moves = position.legal_moves_with_rules(history);
        if moves.is_empty() {
            return mated_at(ply);
        }
        let original_alpha = alpha;
        let checked = position.in_check(position.side_to_move());
        // Search an isolated pass only at a non-PV cut node. The real position
        // has already passed terminal and legal-move checks above. A pass is
        // not a Xiangqi move, so its descendants need a fresh rule anchor.
        if null_move_eligible(
            position,
            history,
            depth,
            alpha,
            beta,
            checked,
            previous_was_null,
            null_enabled,
        ) {
            let static_eval = self.evaluate(position, history, hidden, ply);
            if self.exhausted {
                return 0.0;
            }
            if static_eval >= beta + 0.10 {
                let undo = position.make_null_move();
                let mut null_history = position.initial_rule_history();
                let reduced_depth = depth.saturating_sub(3);
                let null_score = -self.negamax_inner(
                    position,
                    &mut null_history,
                    hidden,
                    reduced_depth,
                    ply + 1,
                    -beta,
                    -beta + 0.0001,
                    true,
                    true,
                );
                position.unmake_null_move(undo);
                if self.exhausted {
                    return 0.0;
                }
                if null_score >= beta && null_score < 0.9 {
                    // A real-move verification excludes recursive null cutoffs,
                    // which protects zugzwang and forced quiet defenses.
                    let verified = self.negamax_inner(
                        position,
                        history,
                        hidden,
                        depth - 1,
                        ply,
                        beta - 0.0001,
                        beta,
                        false,
                        false,
                    );
                    if self.exhausted {
                        return 0.0;
                    }
                    if verified >= beta {
                        return verified;
                    }
                }
            }
        }
        let major_material = depth <= 2 && has_major_material(position, position.side_to_move());
        moves.sort_by_key(|&mv| {
            std::cmp::Reverse(self.move_order_score(position, mv, tt_move, ply))
        });
        let mut best = -2.0f32;
        let mut best_move = None;
        let mut first = true;
        let mut searched_quiets = [None; 32];
        let mut quiet_count = 0;
        for (index, mv) in moves.into_iter().enumerate() {
            let before_buckets = self
                .model
                .pikafish_net()
                .is_none()
                .then(|| AbEvalAccumulator::buckets_for_position(position));
            let mover = position.side_to_move();
            let moved = position.piece_at(mv.from as usize).unwrap();
            let captured = position.piece_at(mv.to as usize);
            let undo = position.make_move(mv);
            let mut child_hidden = Vec::new();
            if let Some(before_buckets) = before_buckets {
                child_hidden = self.hidden_pool.pop().unwrap_or_default();
                child_hidden.resize(hidden.len(), 0.0);
                child_hidden.copy_from_slice(hidden);
                self.model.transition(
                    before_buckets,
                    position,
                    mv,
                    moved,
                    captured,
                    &mut child_hidden,
                );
            }
            history.push(position.rule_history_entry_after_moved(mover, mv, captured));
            let gives_check = position.in_check(position.side_to_move());
            let quiet = captured.is_none();
            // Shallow late-move pruning applies only after at least one legal
            // move has been searched. Checks and captures must still be tried:
            // Xiangqi has many forcing cannon and rook continuations.
            if beta - alpha <= 0.001
                && major_material
                && !checked
                && !gives_check
                && quiet
                && best > -0.9
                && late_move_prunable(
                    depth,
                    index,
                    self.history_scores[mv.from as usize][mv.to as usize],
                    Some(mv) == tt_move
                        || self.killers[ply.min(self.killers.len() - 1)].contains(&Some(mv)),
                )
            {
                history.pop();
                position.unmake_move(mv, undo);
                if before_buckets.is_some() {
                    self.hidden_pool.push(child_hidden);
                }
                continue;
            }
            let reduction = quiet_reduction(
                depth,
                index,
                self.history_scores[mv.from as usize][mv.to as usize],
                checked || gives_check || !quiet,
            );
            let mut score = if first {
                -self.negamax_inner(
                    position,
                    history,
                    &child_hidden,
                    depth - 1,
                    ply + 1,
                    -beta,
                    -alpha,
                    false,
                    null_enabled,
                )
            } else {
                -self.negamax_inner(
                    position,
                    history,
                    &child_hidden,
                    depth - 1 - reduction,
                    ply + 1,
                    -alpha - 0.0001,
                    -alpha,
                    false,
                    null_enabled,
                )
            };
            if reduction != 0 && !self.exhausted && score > alpha {
                score = -self.negamax_inner(
                    position,
                    history,
                    &child_hidden,
                    depth - 1,
                    ply + 1,
                    -alpha - 0.0001,
                    -alpha,
                    false,
                    null_enabled,
                );
            }
            if !first && !self.exhausted && score > alpha && score < beta {
                score = -self.negamax_inner(
                    position,
                    history,
                    &child_hidden,
                    depth - 1,
                    ply + 1,
                    -beta,
                    -alpha,
                    false,
                    null_enabled,
                );
            }
            history.pop();
            position.unmake_move(mv, undo);
            if before_buckets.is_some() {
                self.hidden_pool.push(child_hidden);
            }
            if self.exhausted {
                return 0.0;
            }
            if score > best {
                best = score;
                best_move = Some(mv);
            }
            alpha = alpha.max(best);
            if alpha >= beta {
                if quiet {
                    let bonus = (depth * depth).min(1024) as i32;
                    for &failed in searched_quiets[..quiet_count].iter().flatten() {
                        update_history(&mut self.history_scores, failed, -bonus);
                    }
                    update_history(&mut self.history_scores, mv, bonus);
                    let killer_slot = ply.min(self.killers.len() - 1);
                    let killers = &mut self.killers[killer_slot];
                    if killers[0] != Some(mv) {
                        killers[1] = killers[0];
                        killers[0] = Some(mv);
                    }
                }
                break;
            }
            if quiet && quiet_count < searched_quiets.len() {
                searched_quiets[quiet_count] = Some(mv);
                quiet_count += 1;
            }
            first = false;
        }
        let bound = if best >= beta {
            Bound::Lower
        } else if best <= original_alpha {
            Bound::Upper
        } else {
            Bound::Exact
        };
        if self.tt[slot].as_ref().is_none_or(|entry| {
            entry.key != position.hash() || entry.history != *history || entry.depth <= depth
        }) {
            let entry = self.tt[slot].get_or_insert_with(|| TtEntry {
                key: 0,
                history: Vec::new(),
                depth: 0,
                value: 0.0,
                bound,
                best_move: None,
            });
            entry.key = position.hash();
            entry.history.clone_from(history);
            entry.depth = depth;
            entry.value = best;
            entry.bound = bound;
            entry.best_move = best_move;
        }
        best
    }

    fn move_order_score(
        &self,
        position: &Position,
        mv: Move,
        tt_move: Option<Move>,
        ply: usize,
    ) -> i64 {
        if Some(mv) == tt_move {
            return 1_000_000;
        }
        if let Some(victim) = position.piece_at(mv.to as usize) {
            let attacker = position.piece_at(mv.from as usize).unwrap();
            return 100_000 + 100 * piece_value(victim.kind) - piece_value(attacker.kind);
        }
        let killers = self.killers[ply.min(self.killers.len() - 1)];
        if killers[0] == Some(mv) {
            return 90_000;
        }
        if killers[1] == Some(mv) {
            return 80_000;
        }
        self.history_scores[mv.from as usize][mv.to as usize] as i64
    }
}

const HISTORY_LIMIT: i32 = 16_384;

fn update_history(history: &mut [[i32; 90]; 90], mv: Move, bonus: i32) {
    let entry = &mut history[mv.from as usize][mv.to as usize];
    *entry += bonus - *entry * bonus.abs() / HISTORY_LIMIT;
}

fn quiet_reduction(depth: usize, move_index: usize, history: i32, tactical: bool) -> usize {
    if tactical || depth < 3 || move_index < 4 {
        return 0;
    }
    let extra = usize::from(depth >= 7 && move_index >= 12 && history < 0);
    (1 + extra).min(depth - 2)
}

fn late_move_prunable(depth: usize, move_index: usize, history: i32, priority: bool) -> bool {
    !priority
        && history <= 0
        && match depth {
            1 => move_index >= 20,
            2 => move_index >= 32,
            _ => false,
        }
}

fn has_major_material(position: &Position, color: Color) -> bool {
    (0..90).any(|sq| {
        position.piece_at(sq).is_some_and(|piece| {
            piece.color == color
                && matches!(
                    piece.kind,
                    PieceKind::Rook | PieceKind::Cannon | PieceKind::Horse
                )
        })
    })
}

fn null_move_eligible(
    position: &Position,
    history: &[RuleHistoryEntry],
    depth: usize,
    alpha: f32,
    beta: f32,
    checked: bool,
    previous_was_null: bool,
    null_enabled: bool,
) -> bool {
    depth >= 3
        && null_enabled
        && !previous_was_null
        && !checked
        && beta - alpha <= 0.001
        && beta > -0.9
        && beta < 0.9
        && position.rule60_max_ply().is_none_or(|limit| {
            position
                .rule60_count_with_history(history)
                .saturating_add(20)
                < limit
        })
        && has_major_material(position, position.side_to_move())
}

fn q_delta_upper_bound(stand_pat: f32, victim_cp: i64) -> f32 {
    let base_cp = stand_pat.clamp(-0.95, 0.95).atanh() * 600.0;
    ((base_cp + victim_cp as f32 + 300.0) / 600.0).tanh()
}

fn q_capture_prunable(
    stand_pat: f32,
    alpha: f32,
    victim_cp: i64,
    gives_check: bool,
    recapture: bool,
) -> bool {
    !gives_check
        && !recapture
        && stand_pat > -0.9
        && alpha < 0.9
        && q_delta_upper_bound(stand_pat, victim_cp) <= alpha
}

fn piece_value(kind: PieceKind) -> i64 {
    match kind {
        PieceKind::General => 10_000,
        PieceKind::Rook => 900,
        PieceKind::Cannon => 450,
        PieceKind::Horse => 400,
        PieceKind::Advisor | PieceKind::Elephant => 200,
        PieceKind::Soldier => 100,
    }
}

/// Material result of an optional legal recapture sequence on one square.
/// This deliberately uses the move generator instead of duplicating Xiangqi
/// attack geometry; it is slower than Pikafish's bitboard SEE but respects
/// cannon screens, horse legs and king safety.
fn static_exchange_gain(position: &mut Position, mv: Move) -> (i64, bool) {
    crate::scope_profile!("search.see");
    let target = mv.to;
    let captured = position.piece_at(target as usize).unwrap();
    let undo = position.make_move(mv);
    let gives_check = position.in_check(position.side_to_move());
    let gain = piece_value(captured.kind) - best_exchange_reply(position, target, 16);
    position.unmake_move(mv, undo);
    (gain, gives_check)
}

fn best_exchange_reply(position: &mut Position, target: u8, remaining: usize) -> i64 {
    if remaining == 0 || !position.has_general(position.side_to_move()) {
        return 0;
    }
    let Some(victim) = position.piece_at(target as usize) else {
        return 0;
    };
    let captures = position.legal_capture_moves_to(target as usize);
    let mut best = 0;
    for mv in captures {
        let undo = position.make_move(mv);
        let reply = if victim.kind == PieceKind::General {
            0
        } else {
            best_exchange_reply(position, target, remaining - 1)
        };
        position.unmake_move(mv, undo);
        best = best.max(piece_value(victim.kind) - reply);
    }
    best
}

fn outcome_value(outcome: RuleOutcome, side: Color) -> f32 {
    match outcome {
        RuleOutcome::Draw(_) => 0.0,
        RuleOutcome::Win(winner) => {
            if winner == side {
                MATE
            } else {
                -MATE
            }
        }
    }
}

fn mated_at(ply: usize) -> f32 {
    -MATE + MATE_PLY_PENALTY * ply.min(40) as f32
}

fn terminal_value(outcome: RuleOutcome, side: Color, ply: usize) -> f32 {
    match outcome {
        RuleOutcome::Draw(_) => 0.0,
        RuleOutcome::Win(winner) if winner == side => -mated_at(ply),
        RuleOutcome::Win(_) => mated_at(ply),
    }
}

pub fn search(
    position: &Position,
    history: &[RuleHistoryEntry],
    moves: Vec<Move>,
    model: &AbNnue,
    node_limit: usize,
) -> AbSearchResult {
    search_with_control(
        position,
        history,
        moves,
        model,
        node_limit,
        16,
        None,
        |_| {},
    )
}

fn search_with_control(
    position: &Position,
    history: &[RuleHistoryEntry],
    moves: Vec<Move>,
    model: &AbNnue,
    node_limit: usize,
    max_depth: usize,
    control: Option<&AbSearchControl>,
    progress: impl FnMut(&AbSearchResult),
) -> AbSearchResult {
    search_with_model(
        position, history, moves, model, node_limit, max_depth, control, false, progress,
    )
    .expect("AB NNUE evaluation failed")
}

/// Search a trainable Pikafish-layout model with the same AB/PVS/TT path.
/// 搜索使用只读 CPU 权重和私有增量累加器；训练张量不进入叶评估。
pub fn search_pikafish_model(
    position: &Position,
    history: &[RuleHistoryEntry],
    model: &PikafishCpuModel,
    limits: AbSearchLimits,
) -> Result<AbSearchResult, String> {
    let moves = position.legal_moves_with_rules(history);
    search_with_model(
        position,
        history,
        moves,
        model,
        limits.nodes,
        if limits.max_depth == 0 {
            16
        } else {
            limits.max_depth
        },
        None,
        false,
        |_| {},
    )
}

fn search_with_model<M: ValueModel + ?Sized>(
    position: &Position,
    history: &[RuleHistoryEntry],
    moves: Vec<Move>,
    model: &M,
    node_limit: usize,
    max_depth: usize,
    control: Option<&AbSearchControl>,
    root_pvs: bool,
    mut progress: impl FnMut(&AbSearchResult),
) -> Result<AbSearchResult, String> {
    let root_wdl = model.root_wdl(position, history)?;
    let mut engine = Search {
        model,
        nodes: 0,
        limit: node_limit.max(1),
        exhausted: false,
        quiescence_nodes: 0,
        selective_depth: 0,
        history_scores: [[0; 90]; 90],
        killers: vec![[None; 2]; 128],
        tt: TtLease::new(),
        hidden_pool: Vec::new(),
        scratch: model.scratch(),
        pikafish_cache: model.pikafish_net().map(|_| PikafishEvalCache::new()),
        pikafish_cpu_cache: model.pikafish_cpu().map(|_| PikafishCpuCache::default()),
        error: None,
        control,
    };
    let root_hidden = model.root_hidden(position);
    let mut root_child_hidden = root_hidden.clone();
    let mut scores = moves
        .iter()
        .map(|&mv| {
            let mut next = position.clone();
            let mover = position.side_to_move();
            let captured = position.piece_at(mv.to as usize);
            next.make_move(mv);
            root_child_hidden.copy_from_slice(&root_hidden);
            if model.pikafish_net().is_none() {
                model.transition(
                    AbEvalAccumulator::buckets_for_position(position),
                    &next,
                    mv,
                    position.piece_at(mv.from as usize).unwrap(),
                    captured,
                    &mut root_child_hidden,
                );
            }
            let mut line = history.to_vec();
            line.push(next.rule_history_entry_after_moved(mover, mv, captured));
            if let Some(outcome) = next.rule_outcome_with_history(&line) {
                -outcome_value(outcome, next.side_to_move())
            } else {
                let replies = next.legal_moves_with_rules(&line);
                if replies.is_empty() {
                    MATE
                } else {
                    -engine.evaluate(&next, &line, &root_child_hidden, 1)
                }
            }
        })
        .collect::<Vec<_>>();
    if let Some(error) = engine.error.take() {
        return Err(error);
    }
    let solved = root_proofs(position, history, &moves);
    // The initial child evaluations form a complete one-ply fallback.
    let mut completed_depth = 1;
    // A complete iteration supplies comparable scores for every root move.
    for depth in 1..=max_depth.max(1) {
        if control.is_some_and(AbSearchControl::should_stop) {
            break;
        }
        let mut trial = vec![0.0; moves.len()];
        let mut order = (0..moves.len()).collect::<Vec<_>>();
        order.sort_by(|&left, &right| scores[right].total_cmp(&scores[left]));
        let mut root_alpha = -2.0f32;
        let mut first_root = true;
        for index in order {
            let mv = moves[index];
            let mut next = position.clone();
            let mover = position.side_to_move();
            let captured = position.piece_at(mv.to as usize);
            next.make_move(mv);
            root_child_hidden.copy_from_slice(&root_hidden);
            if model.pikafish_net().is_none() {
                model.transition(
                    AbEvalAccumulator::buckets_for_position(position),
                    &next,
                    mv,
                    position.piece_at(mv.from as usize).unwrap(),
                    captured,
                    &mut root_child_hidden,
                );
            }
            let mut line = history.to_vec();
            line.push(next.rule_history_entry_after_moved(mover, mv, captured));
            let (low, high) = if root_pvs {
                if first_root {
                    (-2.0, 2.0)
                } else {
                    (root_alpha, root_alpha + 0.0001)
                }
            } else if depth >= 3 {
                (
                    (scores[index] - 0.20).max(-2.0),
                    (scores[index] + 0.20).min(2.0),
                )
            } else {
                (-2.0, 2.0)
            };
            let mut value = -engine.negamax(
                &mut next,
                &mut line,
                &root_child_hidden,
                depth - 1,
                1,
                -high,
                -low,
            );
            if !engine.exhausted
                && if root_pvs {
                    !first_root && value > root_alpha
                } else {
                    value <= low || value >= high
                }
            {
                value = -engine.negamax(
                    &mut next,
                    &mut line,
                    &root_child_hidden,
                    depth - 1,
                    1,
                    -2.0,
                    2.0,
                );
            }
            trial[index] = value;
            if engine.exhausted {
                break;
            }
            root_alpha = root_alpha.max(value);
            first_root = false;
        }
        if engine.exhausted {
            break;
        }
        scores = trial;
        completed_depth = depth;
        progress(&build_result_with_proofs(
            &moves,
            &scores,
            &solved,
            root_wdl,
            engine.nodes,
            engine.quiescence_nodes,
            completed_depth,
            engine.selective_depth,
            false,
            max_depth,
        ));
        // A saturated network score is not a proof of mate.
        if solved.contains(&Some(1)) {
            break;
        }
    }
    if let Some(error) = engine.error {
        return Err(error);
    }
    Ok(build_result_with_proofs(
        &moves,
        &scores,
        &solved,
        root_wdl,
        engine.nodes,
        engine.quiescence_nodes,
        completed_depth,
        engine.selective_depth,
        engine.exhausted,
        max_depth,
    ))
}

fn root_proofs(
    position: &Position,
    history: &[RuleHistoryEntry],
    moves: &[Move],
) -> Vec<Option<i8>> {
    moves
        .iter()
        .map(|&mv| {
            let mut next = position.clone();
            let mover = position.side_to_move();
            let captured = position.piece_at(mv.to as usize);
            next.make_move(mv);
            let mut line = history.to_vec();
            line.push(next.rule_history_entry_after_moved(mover, mv, captured));
            if !next.has_general(next.side_to_move())
                || next.legal_moves_with_rules(&line).is_empty()
            {
                Some(1)
            } else {
                next.rule_outcome_with_history(&line)
                    .map(|outcome| -(outcome_value(outcome, next.side_to_move()) as i8))
            }
        })
        .collect()
}

fn proof_rank(proof: Option<i8>) -> u8 {
    match proof {
        Some(1) => 2,
        Some(-1) => 0,
        _ => 1,
    }
}

fn proof_score(proof: Option<i8>, score: f32) -> f32 {
    proof.map_or(score, f32::from)
}

fn build_result_with_proofs(
    moves: &[Move],
    scores: &[f32],
    solved: &[Option<i8>],
    root_wdl: [f32; 3],
    nodes: usize,
    quiescence_nodes: usize,
    completed_depth: usize,
    selective_depth: usize,
    exhausted: bool,
    max_depth: usize,
) -> AbSearchResult {
    let has_win = solved.contains(&Some(1));
    let has_non_loss = solved.iter().any(|proof| *proof != Some(-1));
    let best_index = scores
        .iter()
        .enumerate()
        .max_by(|a, b| {
            proof_rank(solved[a.0])
                .cmp(&proof_rank(solved[b.0]))
                .then_with(|| {
                    proof_score(solved[a.0], *a.1).total_cmp(&proof_score(solved[b.0], *b.1))
                })
        })
        .map(|(i, _)| i);
    let best_value = best_index.map_or(0.0, |i| proof_score(solved[i], scores[i]));
    let max_score = scores
        .iter()
        .enumerate()
        .filter(|(i, _)| !has_non_loss || solved[*i] != Some(-1))
        .map(|(i, &score)| proof_score(solved[i], score))
        .fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = scores
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            if has_win {
                f32::from(solved[i] == Some(1))
            } else if has_non_loss && solved[i] == Some(-1) {
                0.0
            } else {
                ((proof_score(solved[i], s) - max_score) * 8.0).exp()
            }
        })
        .collect();
    let total = weights.iter().sum::<f32>().max(1e-12);
    let candidates = moves
        .iter()
        .enumerate()
        .map(|(i, &mv)| AbCandidate {
            mv,
            q: proof_score(solved[i], scores[i]),
            selection_weight: weights[i] / total,
            solved: solved[i],
        })
        .collect();
    let wdl = if best_value >= MATE {
        [1.0, 0.0, 0.0]
    } else if best_value <= -MATE {
        [0.0, 0.0, 1.0]
    } else {
        [(best_value + 1.0) * 0.5, 0.0, (1.0 - best_value) * 0.5]
    };
    AbSearchResult {
        best_move: best_index.map(|i| moves[i]),
        value_q: best_value,
        value_cp: cp_from_q(best_value),
        value_wdl: wdl,
        network_value_wdl: root_wdl,
        best_value_wdl: wdl,
        nodes: nodes,
        quiescence_nodes,
        search_depth_avg: completed_depth as f32,
        search_depth_max: completed_depth,
        selective_depth,
        search_depth_limit: max_depth,
        search_depth_cutoffs: usize::from(exhausted),
        candidates,
    }
}

fn uci_report(search: AbSearchResult, multipv: usize) -> AbUciSearchResult {
    let mut ranked = search.candidates.iter().collect::<Vec<_>>();
    ranked.sort_by(|a, b| {
        b.proof_priority()
            .cmp(&a.proof_priority())
            .then_with(|| b.q.total_cmp(&a.q))
    });
    if let Some(best) = search.best_move
        && let Some(index) = ranked.iter().position(|candidate| candidate.mv == best)
    {
        let chosen = ranked.remove(index);
        ranked.insert(0, chosen);
    }
    let variations = ranked
        .into_iter()
        .take(multipv.max(1))
        .map(|candidate| {
            let q = candidate.q;
            let wdl = if let Some(proven) = candidate.solved {
                match proven {
                    1 => [1.0, 0.0, 0.0],
                    -1 => [0.0, 0.0, 1.0],
                    _ => [0.0, 1.0, 0.0],
                }
            } else {
                [(q + 1.0) * 0.5, 0.0, (1.0 - q) * 0.5]
            };
            AbUciPv {
                moves: vec![candidate.mv],
                wdl,
                q,
                proven: candidate.solved,
            }
        })
        .collect();
    AbUciSearchResult { search, variations }
}

pub(crate) fn search_uci(
    position: &Position,
    history: Vec<RuleHistoryEntry>,
    root_moves: Vec<Move>,
    model: &AbNnue,
    limits: AbSearchLimits,
    control: &AbSearchControl,
    multipv: usize,
    progress: impl FnMut(&AbUciSearchResult),
) -> AbUciSearchResult {
    search_uci_with_model(
        position, history, root_moves, model, limits, control, multipv, progress,
    )
}

pub(crate) fn search_uci_pikafish(
    position: &Position,
    history: Vec<RuleHistoryEntry>,
    root_moves: Vec<Move>,
    model: &PikafishNet,
    limits: AbSearchLimits,
    control: &AbSearchControl,
    multipv: usize,
    progress: impl FnMut(&AbUciSearchResult),
) -> AbUciSearchResult {
    search_uci_with_model(
        position, history, root_moves, model, limits, control, multipv, progress,
    )
}

pub(crate) fn search_uci_pikafish_float(
    position: &Position,
    history: Vec<RuleHistoryEntry>,
    root_moves: Vec<Move>,
    model: &PikafishCpuModel,
    limits: AbSearchLimits,
    control: &AbSearchControl,
    multipv: usize,
    progress: impl FnMut(&AbUciSearchResult),
) -> AbUciSearchResult {
    search_uci_with_model(
        position, history, root_moves, model, limits, control, multipv, progress,
    )
}

fn search_uci_with_model<M: ValueModel + ?Sized>(
    position: &Position,
    history: Vec<RuleHistoryEntry>,
    root_moves: Vec<Move>,
    model: &M,
    limits: AbSearchLimits,
    control: &AbSearchControl,
    multipv: usize,
    mut progress: impl FnMut(&AbUciSearchResult),
) -> AbUciSearchResult {
    let search = search_with_model(
        position,
        &history,
        root_moves,
        model,
        limits.nodes,
        if limits.max_depth == 0 {
            64
        } else {
            limits.max_depth.min(64)
        },
        Some(control),
        multipv == 1,
        |snapshot| progress(&uci_report(snapshot.clone(), multipv)),
    )
    .expect("NNUE evaluation failed");
    uci_report(search, multipv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, atomic::AtomicBool};

    #[test]
    fn transposition_slots_reset_reuse_and_collide() {
        let mut tt = TtLease::new();
        let entry = |key| {
            Some(TtEntry {
                key,
                history: Vec::new(),
                depth: 2,
                value: 0.5,
                bound: Bound::Exact,
                best_move: None,
            })
        };
        tt[17] = entry(17);
        tt[17] = entry(17 + TT_SIZE as u64);
        assert_eq!(tt.0.as_ref().unwrap().used, 1);
        assert_eq!(tt[17].as_ref().unwrap().key, 17 + TT_SIZE as u64);
        drop(tt);
        let tt = TtLease::new();
        assert!(tt[17].is_none());
        assert_eq!(tt.0.as_ref().unwrap().entries.len(), 1);
    }

    #[test]
    fn pikafish_value_model_uses_shared_search() {
        let trainer =
            crate::ab::pikafish_candle::PikafishModel::new(&candle_core::Device::Cpu).unwrap();
        let model = trainer.cpu_snapshot().unwrap();
        let position = Position::startpos();
        let example =
            crate::ab::pikafish_candle::PikafishExample::from_position(&position).unwrap();
        let raw = trainer
            .forward(&[example])
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()[0][0];
        let mut cache = PikafishCpuCache::default();
        let mut position = position.clone();
        for i in 0..80 {
            let example =
                crate::ab::pikafish_candle::PikafishExample::from_position(&position).unwrap();
            let expected = trainer
                .forward_inference(&[example])
                .unwrap()
                .to_vec2::<f32>()
                .unwrap()[0][0]
                .clamp(-1., 1.);
            let actual = model.evaluate(&position, &mut cache).unwrap();
            assert!(
                (actual - expected).abs() < 2e-5,
                "ply {i}: {actual} vs {expected}"
            );
            let moves = position.legal_moves();
            if moves.is_empty() {
                break;
            }
            position.make_move(moves[(i * 7 + 3) % moves.len()]);
            if !position.has_general(Color::Red) || !position.has_general(Color::Black) {
                break;
            }
        }
        let position = Position::startpos();
        let evaluated = ValueModel::evaluate(&model, &position, &[], &[], &mut None).unwrap();
        assert!((evaluated - raw.clamp(-1.0, 1.0)).abs() < 1e-6);
        let result = search_pikafish_model(
            &position,
            &[],
            &model,
            AbSearchLimits {
                nodes: 64,
                max_depth: 2,
            },
        )
        .unwrap();
        assert!(result.best_move.is_some());
        assert!(result.nodes <= 64);
        assert!(result.quiescence_nodes <= result.nodes);
        assert!(result.value_q.is_finite());
    }

    #[test]
    fn history_bonus_and_malus_stay_bounded() {
        let mv = Position::startpos().legal_moves()[0];
        let mut history = [[0; 90]; 90];
        for _ in 0..1_000 {
            update_history(&mut history, mv, 1024);
        }
        let positive = history[mv.from as usize][mv.to as usize];
        assert!(positive > 0 && positive <= HISTORY_LIMIT);
        for _ in 0..2_000 {
            update_history(&mut history, mv, -1024);
        }
        let negative = history[mv.from as usize][mv.to as usize];
        assert!(negative < 0 && negative >= -HISTORY_LIMIT);
    }

    #[test]
    fn late_quiet_reduction_respects_tactics_and_history() {
        assert_eq!(quiet_reduction(7, 20, -100, true), 0);
        assert_eq!(quiet_reduction(2, 20, -100, false), 0);
        assert_eq!(quiet_reduction(7, 20, 100, false), 1);
        assert_eq!(quiet_reduction(7, 20, -100, false), 2);
    }

    #[test]
    fn shallow_late_move_pruning_preserves_priority_and_deeper_nodes() {
        assert!(!late_move_prunable(1, 19, 0, false));
        assert!(late_move_prunable(1, 20, 0, false));
        assert!(!late_move_prunable(2, 32, 1, false));
        assert!(!late_move_prunable(2, 32, 0, true));
        assert!(!late_move_prunable(3, 100, 0, false));
    }

    #[test]
    fn qsearch_delta_preserves_checks_recaptures_and_mate_ranges() {
        assert!(q_capture_prunable(-0.5, 0.4, 100, false, false));
        assert!(!q_capture_prunable(-0.5, 0.4, 100, true, false));
        assert!(!q_capture_prunable(-0.5, 0.4, 100, false, true));
        assert!(!q_capture_prunable(-0.95, 0.4, 100, false, false));
        assert!(!q_capture_prunable(-0.5, 0.95, 100, false, false));
    }

    #[test]
    fn null_move_requires_safe_non_pv_dynamic_position() {
        let start = Position::startpos();
        let history = start.initial_rule_history();
        let allowed = |position: &Position,
                       history: &[RuleHistoryEntry],
                       depth,
                       alpha,
                       beta,
                       checked,
                       previous,
                       enabled| {
            null_move_eligible(
                position, history, depth, alpha, beta, checked, previous, enabled,
            )
        };
        assert!(allowed(
            &start, &history, 3, 0.1, 0.1001, false, false, true
        ));
        assert!(!allowed(
            &start, &history, 2, 0.1, 0.1001, false, false, true
        ));
        assert!(!allowed(&start, &history, 3, -2.0, 2.0, false, false, true));
        assert!(!allowed(
            &start, &history, 3, 0.1, 0.1001, true, false, true
        ));
        assert!(!allowed(
            &start, &history, 3, 0.1, 0.1001, false, true, true
        ));
        assert!(!allowed(
            &start, &history, 3, 0.1, 0.1001, false, false, false
        ));
        let bare = Position::from_fen("4k4/9/9/9/4p4/9/4P4/9/9/4K4 w - - 0 1").unwrap();
        assert!(!allowed(
            &bare,
            &bare.initial_rule_history(),
            3,
            0.1,
            0.1001,
            false,
            false,
            true
        ));
        let late = Position::from_fen(
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 110 1",
        )
        .unwrap();
        assert!(!allowed(
            &late,
            &late.initial_rule_history(),
            3,
            0.1,
            0.1001,
            false,
            false,
            true
        ));
    }

    #[test]
    fn bounded_search_returns_legal_move_and_selection_weights() {
        let position = Position::startpos();
        let history = position.initial_rule_history();
        let moves = position.legal_moves_with_rules(&history);
        let model = AbNnue::random(16, 7);
        let result = search(&position, &history, moves.clone(), &model, 64);
        assert!(moves.contains(&result.best_move.unwrap()));
        assert!(result.nodes <= 64);
        assert_eq!(result.candidates.len(), moves.len());
        assert!(
            (result
                .candidates
                .iter()
                .map(|c| c.selection_weight)
                .sum::<f32>()
                - 1.0)
                .abs()
                < 1e-5
        );
        assert_eq!(
            result.best_move,
            search(&position, &history, moves, &model, 64).best_move
        );
    }

    #[test]
    fn quiescence_restores_board_and_rule_history() {
        let mut position = Position::from_fen(
            "2bak2r1/4a4/4b4/p2R4p/4C1n2/2P1c3P/P1r3P2/4B4/4A4/2BK1A2R w - - 1 1",
        )
        .unwrap();
        let original = position.clone();
        let mut history = position.initial_rule_history();
        let original_history = history.clone();
        let model = AbNnue::random(16, 11);
        let mut engine = Search {
            model: &model,
            nodes: 0,
            limit: 256,
            exhausted: false,
            quiescence_nodes: 0,
            selective_depth: 0,
            history_scores: [[0; 90]; 90],
            killers: vec![[None; 2]; 128],
            tt: TtLease::new(),
            hidden_pool: Vec::new(),
            scratch: Some(AbEvalScratch::new(model.arch)),
            pikafish_cache: None,
            pikafish_cpu_cache: None,
            error: None,
            control: None,
        };
        let hidden = AbEvalAccumulator::new(&model, &position).into_hidden_sum();
        let _ = engine.quiescence(&mut position, &mut history, &hidden, 4, 1, -2.0, 2.0);
        assert_eq!(position, original);
        assert_eq!(history, original_history);
        assert!(engine.quiescence_nodes > 1);
        assert!(engine.selective_depth > 1);
    }

    #[test]
    fn depth_zero_counts_each_quiescence_node_once() {
        let mut position = Position::startpos();
        let mut history = position.initial_rule_history();
        let model = AbNnue::random(16, 31);
        let hidden = AbEvalAccumulator::new(&model, &position).into_hidden_sum();
        let mut engine = Search {
            model: &model,
            nodes: 0,
            limit: 256,
            exhausted: false,
            quiescence_nodes: 0,
            selective_depth: 0,
            history_scores: [[0; 90]; 90],
            killers: vec![[None; 2]; 128],
            tt: TtLease::new(),
            hidden_pool: Vec::new(),
            scratch: Some(AbEvalScratch::new(model.arch)),
            pikafish_cache: None,
            pikafish_cpu_cache: None,
            error: None,
            control: None,
        };
        let _ = engine.negamax(&mut position, &mut history, &hidden, 0, 0, -2.0, 2.0);
        assert_eq!(engine.nodes, engine.quiescence_nodes);
        assert!(engine.nodes > 0);
    }

    #[test]
    fn root_pvs_keeps_best_move_and_uses_fewer_nodes() {
        let position = Position::startpos();
        let history = position.initial_rule_history();
        let moves = position.legal_moves_with_rules(&history);
        let model = AbNnue::random(16, 37);
        let full = search_with_model(
            &position,
            &history,
            moves.clone(),
            &model,
            100_000,
            2,
            None,
            false,
            |_| {},
        )
        .unwrap();
        let pvs = search_with_model(
            &position,
            &history,
            moves,
            &model,
            100_000,
            2,
            None,
            true,
            |_| {},
        )
        .unwrap();
        assert_eq!(pvs.best_move, full.best_move);
        assert!((pvs.value_q - full.value_q).abs() < 1e-5);
        assert!(pvs.nodes < full.nodes, "{} >= {}", pvs.nodes, full.nodes);
    }

    #[test]
    fn see_respects_cannon_screen_and_restores_position() {
        let mut position = Position::from_fen("3k5/9/4r4/9/9/4p4/9/4P4/9/2K1C4 w - - 0 1").unwrap();
        let original = position.clone();
        let mv = position.parse_uci_move("e0e4").unwrap();
        assert!(position.legal_moves().contains(&mv));
        let (gain, check) = static_exchange_gain(&mut position, mv);
        assert_eq!(gain, 100 - 450);
        assert!(!check);
        assert_eq!(position, original);
        let without_screen = Position::from_fen("3k5/9/4r4/9/9/4p4/9/9/9/2K1C4 w - - 0 1").unwrap();
        assert!(!without_screen.legal_moves().contains(&mv));
    }

    #[test]
    fn see_respects_horse_leg_blocker() {
        let open = Position::from_fen("4k4/9/9/9/6n2/4p4/9/9/9/3KR4 w - - 0 1").unwrap();
        let blocked = Position::from_fen("4k4/9/9/9/5pn2/4p4/9/9/9/3KR4 w - - 0 1").unwrap();
        let mv = open.parse_uci_move("e0e4").unwrap();
        assert!(open.legal_moves().contains(&mv));
        let mut open = open;
        let mut blocked = blocked;
        assert_eq!(static_exchange_gain(&mut open, mv).0, 100 - 900);
        assert_eq!(static_exchange_gain(&mut blocked, mv).0, 100);
    }

    #[test]
    fn see_identifies_checking_material_sacrifice() {
        let mut position = Position::from_fen("4k4/9/4r4/9/9/4p4/9/4P4/9/3KC4 w - - 0 1").unwrap();
        let mv = position.parse_uci_move("e0e4").unwrap();
        let (gain, gives_check) = static_exchange_gain(&mut position, mv);
        assert!(gain < -200);
        assert!(gives_check);
    }

    #[test]
    fn uci_search_honors_stop_and_multipv() {
        let position = Position::startpos();
        let history = position.initial_rule_history();
        let moves = position.legal_moves_with_rules(&history);
        let model = AbNnue::random(16, 13);
        let stop = Arc::new(AtomicBool::new(false));
        let control = AbSearchControl::new(Arc::clone(&stop), None);
        let limits = AbSearchLimits {
            nodes: 64,
            max_depth: 2,
            ..Default::default()
        };
        let result = search_uci(
            &position,
            history.clone(),
            moves.clone(),
            &model,
            limits,
            &control,
            4,
            |_| {},
        );
        assert_eq!(result.variations.len(), 4);
        assert!(result.search.nodes <= 64);
        assert!(
            result
                .variations
                .iter()
                .all(|pv| moves.contains(&pv.moves[0]))
        );
        assert_eq!(
            result.variations[0].moves[0],
            result.search.best_move.unwrap()
        );
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        let stopped = search_uci(
            &position,
            history,
            moves,
            &model,
            limits,
            &control,
            1,
            |_| {},
        );
        assert_eq!(stopped.search.nodes, 0);
        assert!(stopped.search.best_move.is_some());
    }

    #[test]
    fn transposition_score_requires_identical_rule_history() {
        let mut position = Position::startpos();
        let history = position.initial_rule_history();
        let model = AbNnue::random(16, 19);
        let mut engine = Search {
            model: &model,
            nodes: 0,
            limit: 10_000,
            exhausted: false,
            quiescence_nodes: 0,
            selective_depth: 0,
            history_scores: [[0; 90]; 90],
            killers: vec![[None; 2]; 128],
            tt: TtLease::new(),
            hidden_pool: Vec::new(),
            scratch: Some(AbEvalScratch::new(model.arch)),
            pikafish_cache: None,
            pikafish_cpu_cache: None,
            error: None,
            control: None,
        };
        let slot = (position.hash() as usize) & (TT_SIZE - 1);
        engine.tt[slot] = Some(TtEntry {
            key: position.hash(),
            history: history.clone(),
            depth: 2,
            value: 0.777,
            bound: Bound::Exact,
            best_move: None,
        });
        let mut same_history = history.clone();
        let hidden = AbEvalAccumulator::new(&model, &position).into_hidden_sum();
        assert_eq!(
            engine.negamax(&mut position, &mut same_history, &hidden, 2, 0, -2.0, 2.0),
            0.777
        );
        let mut different_history = history;
        different_history[0].gives_check = !different_history[0].gives_check;
        let score = engine.negamax(
            &mut position,
            &mut different_history,
            &hidden,
            2,
            0,
            -2.0,
            2.0,
        );
        assert_ne!(score, 0.777);
        assert_eq!(position, Position::startpos());
    }

    #[test]
    fn deeper_search_respects_node_budget_after_aspiration_research() {
        let position = Position::startpos();
        let history = position.initial_rule_history();
        let moves = position.legal_moves_with_rules(&history);
        let model = AbNnue::random(16, 23);
        let result = search(&position, &history, moves.clone(), &model, 4_000);
        assert!(result.nodes <= 4_000);
        assert!(moves.contains(&result.best_move.unwrap()));
        assert!(
            result
                .candidates
                .iter()
                .all(|candidate| candidate.selection_weight.is_finite()
                    && candidate.selection_weight >= 0.0)
        );
    }

    #[test]
    fn proven_loss_cannot_outrank_unproven_move_or_receive_selection_weight() {
        let position = Position::startpos();
        let moves = position.legal_moves();
        let chosen = &moves[..2];
        let result = build_result_with_proofs(
            chosen,
            &[0.9, -0.4],
            &[Some(-1), None],
            [0.3, 0.4, 0.3],
            3,
            2,
            2,
            2,
            false,
            4,
        );
        assert_eq!(result.best_move, Some(chosen[1]));
        assert_eq!(result.candidates[0].selection_weight, 0.0);
        assert_eq!(result.candidates[1].selection_weight, 1.0);
        assert_eq!(result.candidates[0].q, -1.0);
    }

    #[test]
    fn checked_quiescence_searches_evasions_at_capture_horizon() {
        let mut position = Position::from_fen("4k4/9/9/9/9/9/9/9/4r4/4K4 w - - 0 1").unwrap();
        let mut history = position.initial_rule_history();
        assert!(position.in_check(position.side_to_move()));
        let model = AbNnue::random(16, 29);
        let mut engine = Search {
            model: &model,
            nodes: 0,
            limit: 128,
            exhausted: false,
            quiescence_nodes: 0,
            selective_depth: 0,
            history_scores: [[0; 90]; 90],
            killers: vec![[None; 2]; 128],
            tt: TtLease::new(),
            hidden_pool: Vec::new(),
            scratch: Some(AbEvalScratch::new(model.arch)),
            pikafish_cache: None,
            pikafish_cpu_cache: None,
            error: None,
            control: None,
        };
        let hidden = AbEvalAccumulator::new(&model, &position).into_hidden_sum();
        let _ = engine.quiescence(&mut position, &mut history, &hidden, 0, 1, -2.0, 2.0);
        assert!(engine.quiescence_nodes > 1);
    }

    #[test]
    fn capture_generator_matches_filtered_legal_moves_on_playout() {
        let mut position = Position::startpos();
        let mut seed = 0x5eed_u64;
        for _ in 0..160 {
            let legal = position.legal_moves();
            if legal.is_empty() {
                break;
            }
            let filtered = legal
                .iter()
                .copied()
                .filter(|mv| position.is_capture(*mv))
                .collect::<Vec<_>>();
            let captures = position.legal_capture_moves();
            assert_eq!(captures.len(), filtered.len(), "{}", position.to_fen());
            assert!(captures.iter().all(|mv| filtered.contains(mv)));
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let mv = legal[(seed as usize) % legal.len()];
            position.make_move(mv);
        }
    }

    #[test]
    fn mate_distance_prefers_faster_wins_and_slower_losses() {
        assert!(-mated_at(1) > -mated_at(3));
        assert!(mated_at(3) > mated_at(1));
        assert!(-mated_at(40) > MAX_STATIC_SCORE);
    }
}
