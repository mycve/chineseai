//! Bounded iterative deepening negamax for NNUE self-play and promotion matches.
use super::AzNnue;
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
    pub visits: u32,
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
use crate::xiangqi::{Color, Move, Position, RuleHistoryEntry, RuleOutcome};

const MATE: f32 = 1.0;

struct Search<'a> {
    model: &'a AzNnue,
    nodes: usize,
    limit: usize,
    exhausted: bool,
    quiescence_nodes: usize,
    history_scores: [[u32; 90]; 90],
    control: Option<&'a AzSearchControl>,
}

impl Search<'_> {
    fn evaluate(&self, position: &Position, history: &[RuleHistoryEntry], moves: &[Move]) -> f32 {
        self.model
            .evaluate_value_with_rules(position, history, moves)
            .clamp(-1.0, 1.0)
    }

    fn quiescence(
        &mut self,
        position: &mut Position,
        history: &mut Vec<RuleHistoryEntry>,
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
        let stand_pat = self.evaluate(position, history, &moves);
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
            let mover = position.side_to_move();
            let captured = position.piece_at(mv.to as usize);
            let undo = position.make_move(mv);
            history.push(position.rule_history_entry_after_moved(mover, mv, captured));
            let score = -self.quiescence(position, history, remaining - 1, -beta, -alpha);
            history.pop();
            position.unmake_move(mv, undo);
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
        depth: usize,
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
        let mut moves = position.legal_moves_with_rules(history);
        if moves.is_empty() {
            return -MATE;
        }
        if depth == 0 {
            // The legal-move check above keeps stalemate and checkmate exact.
            return self.quiescence(position, history, 6, alpha, beta);
        }
        moves.sort_by_key(|mv| {
            let capture = position.piece_at(mv.to as usize).is_some();
            let history = self.history_scores[mv.from as usize][mv.to as usize];
            (!capture, std::cmp::Reverse(history))
        });
        let mut best = -2.0f32;
        let mut first = true;
        for mv in moves {
            let mover = position.side_to_move();
            let captured = position.piece_at(mv.to as usize);
            let undo = position.make_move(mv);
            history.push(position.rule_history_entry_after_moved(mover, mv, captured));
            let mut score = if first {
                -self.negamax(position, history, depth - 1, -beta, -alpha)
            } else {
                -self.negamax(position, history, depth - 1, -alpha - 0.0001, -alpha)
            };
            if !first && !self.exhausted && score > alpha && score < beta {
                score = -self.negamax(position, history, depth - 1, -beta, -alpha);
            }
            history.pop();
            position.unmake_move(mv, undo);
            if self.exhausted {
                return 0.0;
            }
            best = best.max(score);
            alpha = alpha.max(best);
            if alpha >= beta {
                if captured.is_none() {
                    let history = &mut self.history_scores[mv.from as usize][mv.to as usize];
                    *history = history.saturating_add((depth * depth) as u32);
                }
                break;
            }
            first = false;
        }
        best
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
        control,
    };
    let mut scores = moves
        .iter()
        .map(|&mv| {
            let mut next = position.clone();
            let mover = position.side_to_move();
            let captured = position.piece_at(mv.to as usize);
            next.make_move(mv);
            let mut line = history.to_vec();
            line.push(next.rule_history_entry_after_moved(mover, mv, captured));
            if let Some(outcome) = next.rule_outcome_with_history(&line) {
                -outcome_value(outcome, next.side_to_move())
            } else {
                let replies = next.legal_moves_with_rules(&line);
                if replies.is_empty() {
                    MATE
                } else {
                    -model
                        .evaluate_value_with_rules(&next, &line, &replies)
                        .clamp(-1.0, 1.0)
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
            let mut line = history.to_vec();
            line.push(next.rule_history_entry_after_moved(mover, mv, captured));
            trial[index] = -engine.negamax(&mut next, &mut line, depth - 1, -2.0, 2.0);
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
            visits: (weights[i] / total * 1000.0).round().max(1.0) as u32,
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
            control: None,
        };
        let _ = engine.quiescence(&mut position, &mut history, 4, -2.0, 2.0);
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
}
