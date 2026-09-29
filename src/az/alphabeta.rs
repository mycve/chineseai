//! Bounded iterative deepening negamax for NNUE self-play and promotion matches.
use super::{AzEvalAccumulator, AzEvalScratch, AzNnue};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub struct AzSearchLimits {
    pub simulations: usize,
    pub max_depth: usize,
}
impl Default for AzSearchLimits {
    fn default() -> Self {
        Self {
            simulations: 10_000,
            max_depth: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AzCandidate {
    pub mv: Move,
    pub q: f32,
    pub raw_prior: f32,
    pub prior: f32,
    pub policy: f32,
    pub solved: Option<i8>,
}
impl AzCandidate {
    pub(crate) fn proof_priority(&self) -> u8 {
        match self.solved {
            Some(1) => 2,
            Some(-1) => 0,
            _ => 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AzSearchResult {
    pub best_move: Option<Move>,
    pub value_q: f32,
    pub value_cp: i32,
    pub value_wdl: [f32; 3],
    pub network_value_wdl: [f32; 3],
    pub best_value_wdl: [f32; 3],
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

pub fn cp_from_q(q: f32) -> i32 {
    (q.clamp(-1.0, 1.0) * 1000.0).round() as i32
}

pub(crate) struct AzUciPv {
    pub moves: Vec<Move>,
    pub wdl: [f32; 3],
    pub q: f32,
    pub proven: Option<i8>,
}

pub(crate) struct AzUciSearchResult {
    pub search: AzSearchResult,
    pub variations: Vec<AzUciPv>,
}
use crate::xiangqi::{Color, Move, PieceKind, Position, RuleHistoryEntry, RuleOutcome};

const MATE: f32 = 1.0;
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

struct Search<'a> {
    model: &'a AzNnue,
    nodes: usize,
    limit: usize,
    exhausted: bool,
    quiescence_nodes: usize,
    history_scores: [[u32; 90]; 90],
    killers: Vec<[Option<Move>; 2]>,
    tt: Vec<Option<TtEntry>>,
    hidden_pool: Vec<Vec<f32>>,
    scratch: AzEvalScratch,
    control: Option<&'a AzSearchControl>,
}

impl Search<'_> {
    fn evaluate(
        &mut self,
        position: &Position,
        history: &[RuleHistoryEntry],
        hidden: &[f32],
    ) -> f32 {
        self.model
            .evaluate_incremental_value_with_rules(position, history, hidden, &mut self.scratch)
            .clamp(-1.0, 1.0)
    }

    fn quiescence(
        &mut self,
        position: &mut Position,
        history: &mut Vec<RuleHistoryEntry>,
        hidden: &[f32],
        remaining: usize,
        mut alpha: f32,
        beta: f32,
    ) -> f32 {
        if self.nodes >= self.limit || self.control.is_some_and(AzSearchControl::should_stop) {
            self.exhausted = true;
            return 0.0;
        }
        self.nodes += 1;
        self.quiescence_nodes += 1;
        if let Some(outcome) = position.rule_outcome_with_history(history) {
            return outcome_value(outcome, position.side_to_move());
        }
        let moves = position.legal_moves_with_rules(history);
        if moves.is_empty() {
            return -MATE;
        }
        let checked = position.in_check(position.side_to_move());
        let stand_pat = self.evaluate(position, history, hidden);
        if remaining == 0 {
            return stand_pat;
        }
        if !checked {
            if stand_pat >= beta {
                return stand_pat;
            }
            alpha = alpha.max(stand_pat);
        }
        let mut tactical = moves
            .into_iter()
            .filter(|mv| checked || position.is_capture(*mv))
            .collect::<Vec<_>>();
        tactical.sort_by_key(|mv| position.piece_at(mv.to as usize).is_none());
        let mut best = if checked { -2.0 } else { stand_pat };
        for mv in tactical {
            let before_buckets = AzEvalAccumulator::buckets_for_position(position);
            let mover = position.side_to_move();
            let moved = position.piece_at(mv.from as usize).unwrap();
            let captured = position.piece_at(mv.to as usize);
            let undo = position.make_move(mv);
            let mut child_hidden = self.hidden_pool.pop().unwrap_or_default();
            child_hidden.resize(hidden.len(), 0.0);
            child_hidden.copy_from_slice(hidden);
            AzEvalAccumulator::apply_transition_from_buckets(
                self.model,
                before_buckets,
                position,
                mv,
                moved,
                captured,
                &mut child_hidden,
            );
            history.push(position.rule_history_entry_after_moved(mover, mv, captured));
            let score = -self.quiescence(
                position,
                history,
                &child_hidden,
                remaining - 1,
                -beta,
                -alpha,
            );
            history.pop();
            position.unmake_move(mv, undo);
            self.hidden_pool.push(child_hidden);
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
        mut alpha: f32,
        beta: f32,
    ) -> f32 {
        if self.nodes >= self.limit || self.control.is_some_and(AzSearchControl::should_stop) {
            self.exhausted = true;
            return 0.0;
        }
        self.nodes += 1;
        if let Some(outcome) = position.rule_outcome_with_history(history) {
            return outcome_value(outcome, position.side_to_move());
        }
        if depth == 0 {
            // Quiescence also checks legal-move exhaustion before static evaluation.
            return self.quiescence(position, history, hidden, 6, alpha, beta);
        }
        // Chinese perpetual-check/chase adjudication depends on the whole path.
        // A board hash alone is never sufficient for a score cutoff.
        let slot = (position.hash() as usize) & (TT_SIZE - 1);
        let cached = self.tt[slot].as_ref().filter(|entry| {
            entry.key == position.hash() && entry.history.as_slice() == history.as_slice()
        });
        let tt_move = cached.and_then(|entry| entry.best_move);
        if let Some(entry) = cached.filter(|entry| entry.depth >= depth) {
            match entry.bound {
                Bound::Exact => return entry.value,
                Bound::Lower if entry.value >= beta => return entry.value,
                Bound::Upper if entry.value <= alpha => return entry.value,
                _ => {}
            }
        }
        let mut moves = position.legal_moves_with_rules(history);
        if moves.is_empty() {
            return -MATE;
        }
        let original_alpha = alpha;
        let checked = position.in_check(position.side_to_move());
        moves.sort_by_key(|&mv| {
            std::cmp::Reverse(self.move_order_score(position, mv, tt_move, ply))
        });
        let mut best = -2.0f32;
        let mut best_move = None;
        let mut first = true;
        for (index, mv) in moves.into_iter().enumerate() {
            let before_buckets = AzEvalAccumulator::buckets_for_position(position);
            let mover = position.side_to_move();
            let moved = position.piece_at(mv.from as usize).unwrap();
            let captured = position.piece_at(mv.to as usize);
            let undo = position.make_move(mv);
            let mut child_hidden = self.hidden_pool.pop().unwrap_or_default();
            child_hidden.resize(hidden.len(), 0.0);
            child_hidden.copy_from_slice(hidden);
            AzEvalAccumulator::apply_transition_from_buckets(
                self.model,
                before_buckets,
                position,
                mv,
                moved,
                captured,
                &mut child_hidden,
            );
            history.push(position.rule_history_entry_after_moved(mover, mv, captured));
            let reduction = usize::from(
                depth >= 3
                    && index >= 4
                    && !checked
                    && captured.is_none()
                    && !position.in_check(position.side_to_move()),
            );
            let mut score = if first {
                -self.negamax(
                    position,
                    history,
                    &child_hidden,
                    depth - 1,
                    ply + 1,
                    -beta,
                    -alpha,
                )
            } else {
                -self.negamax(
                    position,
                    history,
                    &child_hidden,
                    depth - 1 - reduction,
                    ply + 1,
                    -alpha - 0.0001,
                    -alpha,
                )
            };
            if reduction != 0 && !self.exhausted && score > alpha {
                score = -self.negamax(
                    position,
                    history,
                    &child_hidden,
                    depth - 1,
                    ply + 1,
                    -alpha - 0.0001,
                    -alpha,
                );
            }
            if !first && !self.exhausted && score > alpha && score < beta {
                score = -self.negamax(
                    position,
                    history,
                    &child_hidden,
                    depth - 1,
                    ply + 1,
                    -beta,
                    -alpha,
                );
            }
            history.pop();
            position.unmake_move(mv, undo);
            self.hidden_pool.push(child_hidden);
            if self.exhausted {
                return 0.0;
            }
            if score > best {
                best = score;
                best_move = Some(mv);
            }
            alpha = alpha.max(best);
            if alpha >= beta {
                if captured.is_none() {
                    let history = &mut self.history_scores[mv.from as usize][mv.to as usize];
                    *history = history.saturating_add((depth * depth) as u32);
                    let killer_slot = ply.min(self.killers.len() - 1);
                    let killers = &mut self.killers[killer_slot];
                    if killers[0] != Some(mv) {
                        killers[1] = killers[0];
                        killers[0] = Some(mv);
                    }
                }
                break;
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
            self.tt[slot] = Some(TtEntry {
                key: position.hash(),
                history: history.clone(),
                depth,
                value: best,
                bound,
                best_move,
            });
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

pub fn search(
    position: &Position,
    history: &[RuleHistoryEntry],
    moves: Vec<Move>,
    model: &AzNnue,
    node_limit: usize,
) -> AzSearchResult {
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
    model: &AzNnue,
    node_limit: usize,
    max_depth: usize,
    control: Option<&AzSearchControl>,
    mut progress: impl FnMut(&AzSearchResult),
) -> AzSearchResult {
    let root_wdl = model.evaluate_wdl_with_rules(position, history, &moves);
    let mut engine = Search {
        model,
        nodes: 0,
        limit: node_limit.max(1),
        exhausted: false,
        quiescence_nodes: 0,
        history_scores: [[0; 90]; 90],
        killers: vec![[None; 2]; 128],
        tt: std::iter::repeat_with(|| None).take(TT_SIZE).collect(),
        hidden_pool: Vec::new(),
        scratch: AzEvalScratch::new(model.arch),
        control,
    };
    let root_hidden = AzEvalAccumulator::new(model, position).into_hidden_sum();
    let mut root_child_hidden = root_hidden.clone();
    let mut scores = moves
        .iter()
        .map(|&mv| {
            let mut next = position.clone();
            let mover = position.side_to_move();
            let captured = position.piece_at(mv.to as usize);
            next.make_move(mv);
            root_child_hidden.copy_from_slice(&root_hidden);
            AzEvalAccumulator::apply_transition_from_buckets(
                model,
                AzEvalAccumulator::buckets_for_position(position),
                &next,
                mv,
                position.piece_at(mv.from as usize).unwrap(),
                captured,
                &mut root_child_hidden,
            );
            let mut line = history.to_vec();
            line.push(next.rule_history_entry_after_moved(mover, mv, captured));
            if let Some(outcome) = next.rule_outcome_with_history(&line) {
                -outcome_value(outcome, next.side_to_move())
            } else {
                let replies = next.legal_moves_with_rules(&line);
                if replies.is_empty() {
                    MATE
                } else {
                    -engine.evaluate(&next, &line, &root_child_hidden)
                }
            }
        })
        .collect::<Vec<_>>();
    let solved = root_proofs(position, history, &moves);
    // The initial child evaluations form a complete one-ply fallback.
    let mut completed_depth = 1;
    // A complete iteration supplies comparable scores for every root move.
    for depth in 1..=max_depth.max(1) {
        if control.is_some_and(AzSearchControl::should_stop) {
            break;
        }
        let mut trial = vec![0.0; moves.len()];
        let mut order = (0..moves.len()).collect::<Vec<_>>();
        order.sort_by(|&left, &right| scores[right].total_cmp(&scores[left]));
        for index in order {
            let mv = moves[index];
            let mut next = position.clone();
            let mover = position.side_to_move();
            let captured = position.piece_at(mv.to as usize);
            next.make_move(mv);
            root_child_hidden.copy_from_slice(&root_hidden);
            AzEvalAccumulator::apply_transition_from_buckets(
                model,
                AzEvalAccumulator::buckets_for_position(position),
                &next,
                mv,
                position.piece_at(mv.from as usize).unwrap(),
                captured,
                &mut root_child_hidden,
            );
            let mut line = history.to_vec();
            line.push(next.rule_history_entry_after_moved(mover, mv, captured));
            let (low, high) = if depth >= 3 {
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
            if !engine.exhausted && (value <= low || value >= high) {
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
            completed_depth,
            false,
            max_depth,
        ));
        if scores.iter().any(|&score| score >= MATE) {
            break;
        }
    }
    build_result_with_proofs(
        &moves,
        &scores,
        &solved,
        root_wdl,
        engine.nodes,
        completed_depth,
        engine.exhausted,
        max_depth,
    )
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

fn build_result_with_proofs(
    moves: &[Move],
    scores: &[f32],
    solved: &[Option<i8>],
    root_wdl: [f32; 3],
    nodes: usize,
    completed_depth: usize,
    exhausted: bool,
    max_depth: usize,
) -> AzSearchResult {
    let has_win = solved.contains(&Some(1));
    let best_index = scores
        .iter()
        .enumerate()
        .max_by(|a, b| {
            (solved[a.0] == Some(1))
                .cmp(&(solved[b.0] == Some(1)))
                .then_with(|| a.1.total_cmp(b.1))
        })
        .map(|(i, _)| i);
    let best_value = best_index.map_or(0.0, |i| {
        if solved[i] == Some(1) {
            MATE
        } else {
            scores[i]
        }
    });
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = scores
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            if has_win {
                f32::from(solved[i] == Some(1))
            } else {
                ((s - max_score) * 8.0).exp()
            }
        })
        .collect();
    let total = weights.iter().sum::<f32>().max(1e-12);
    let candidates = moves
        .iter()
        .enumerate()
        .map(|(i, &mv)| AzCandidate {
            mv,
            q: scores[i],
            raw_prior: weights[i] / total,
            prior: weights[i] / total,
            policy: weights[i] / total,
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
    AzSearchResult {
        best_move: best_index.map(|i| moves[i]),
        value_q: best_value,
        value_cp: cp_from_q(best_value),
        value_wdl: wdl,
        network_value_wdl: root_wdl,
        best_value_wdl: wdl,
        simulations: nodes,
        search_depth_avg: completed_depth as f32,
        search_depth_max: completed_depth,
        search_depth_limit: max_depth,
        search_depth_cutoffs: usize::from(exhausted),
        candidates,
    }
}

fn uci_report(search: AzSearchResult, multipv: usize) -> AzUciSearchResult {
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
            AzUciPv {
                moves: vec![candidate.mv],
                wdl,
                q,
                proven: candidate.solved,
            }
        })
        .collect();
    AzUciSearchResult { search, variations }
}

pub(crate) fn search_uci(
    position: &Position,
    history: Vec<RuleHistoryEntry>,
    root_moves: Vec<Move>,
    model: &AzNnue,
    limits: AzSearchLimits,
    control: &AzSearchControl,
    multipv: usize,
    mut progress: impl FnMut(&AzUciSearchResult),
) -> AzUciSearchResult {
    let search = search_with_control(
        position,
        &history,
        root_moves,
        model,
        limits.simulations,
        if limits.max_depth == 0 {
            64
        } else {
            limits.max_depth.min(64)
        },
        Some(control),
        |snapshot| progress(&uci_report(snapshot.clone(), multipv)),
    );
    uci_report(search, multipv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, atomic::AtomicBool};

    #[test]
    fn bounded_search_returns_legal_move_and_policy() {
        let position = Position::startpos();
        let history = position.initial_rule_history();
        let moves = position.legal_moves_with_rules(&history);
        let model = AzNnue::random(16, 7);
        let result = search(&position, &history, moves.clone(), &model, 64);
        assert!(moves.contains(&result.best_move.unwrap()));
        assert!(result.simulations <= 64);
        assert_eq!(result.candidates.len(), moves.len());
        assert!((result.candidates.iter().map(|c| c.policy).sum::<f32>() - 1.0).abs() < 1e-5);
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
        let model = AzNnue::random(16, 11);
        let mut engine = Search {
            model: &model,
            nodes: 0,
            limit: 256,
            exhausted: false,
            quiescence_nodes: 0,
            history_scores: [[0; 90]; 90],
            killers: vec![[None; 2]; 128],
            tt: std::iter::repeat_with(|| None).take(TT_SIZE).collect(),
            hidden_pool: Vec::new(),
            scratch: AzEvalScratch::new(model.arch),
            control: None,
        };
        let hidden = AzEvalAccumulator::new(&model, &position).into_hidden_sum();
        let _ = engine.quiescence(&mut position, &mut history, &hidden, 4, -2.0, 2.0);
        assert_eq!(position, original);
        assert_eq!(history, original_history);
        assert!(engine.quiescence_nodes > 1);
    }

    #[test]
    fn uci_search_honors_stop_and_multipv() {
        let position = Position::startpos();
        let history = position.initial_rule_history();
        let moves = position.legal_moves_with_rules(&history);
        let model = AzNnue::random(16, 13);
        let stop = Arc::new(AtomicBool::new(false));
        let control = AzSearchControl::new(Arc::clone(&stop), None);
        let limits = AzSearchLimits {
            simulations: 64,
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
        assert!(result.search.simulations <= 64);
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
        assert_eq!(stopped.search.simulations, 0);
        assert!(stopped.search.best_move.is_some());
    }

    #[test]
    fn transposition_score_requires_identical_rule_history() {
        let mut position = Position::startpos();
        let history = position.initial_rule_history();
        let model = AzNnue::random(16, 19);
        let mut engine = Search {
            model: &model,
            nodes: 0,
            limit: 10_000,
            exhausted: false,
            quiescence_nodes: 0,
            history_scores: [[0; 90]; 90],
            killers: vec![[None; 2]; 128],
            tt: std::iter::repeat_with(|| None).take(TT_SIZE).collect(),
            hidden_pool: Vec::new(),
            scratch: AzEvalScratch::new(model.arch),
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
        let hidden = AzEvalAccumulator::new(&model, &position).into_hidden_sum();
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
        let model = AzNnue::random(16, 23);
        let result = search(&position, &history, moves.clone(), &model, 4_000);
        assert!(result.simulations <= 4_000);
        assert!(moves.contains(&result.best_move.unwrap()));
        assert!(
            result
                .candidates
                .iter()
                .all(|candidate| candidate.policy.is_finite() && candidate.policy >= 0.0)
        );
    }
}
