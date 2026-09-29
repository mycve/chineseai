//! Bounded iterative deepening negamax for NNUE self-play and promotion matches.
use super::{AzCandidate, AzNnue, AzSearchResult, cp_from_q};
use crate::xiangqi::{Color, Move, Position, RuleHistoryEntry, RuleOutcome};

const MATE: f32 = 1.0;

struct Search<'a> {
    model: &'a AzNnue,
    nodes: usize,
    limit: usize,
    exhausted: bool,
}

impl Search<'_> {
    fn evaluate(&self, position: &Position, history: &[RuleHistoryEntry], moves: &[Move]) -> f32 {
        self.model
            .evaluate_value_with_rules(position, history, moves)
            .clamp(-1.0, 1.0)
    }

    fn negamax(
        &mut self,
        position: &Position,
        history: &mut Vec<RuleHistoryEntry>,
        depth: usize,
        mut alpha: f32,
        beta: f32,
    ) -> f32 {
        if self.nodes >= self.limit {
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
            return self.evaluate(position, history, &moves);
        }
        moves.sort_by_key(|mv| position.piece_at(mv.to as usize).is_none());
        let mut best = -2.0f32;
        for mv in moves {
            let mut next = position.clone();
            let mover = position.side_to_move();
            let captured = position.piece_at(mv.to as usize);
            next.make_move(mv);
            history.push(next.rule_history_entry_after_moved(mover, mv, captured));
            let score = -self.negamax(&next, history, depth - 1, -beta, -alpha);
            history.pop();
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

pub(super) fn search(
    position: &Position,
    history: &[RuleHistoryEntry],
    moves: Vec<Move>,
    model: &AzNnue,
    node_limit: usize,
) -> AzSearchResult {
    let root_wdl = model.evaluate_wdl_with_rules(position, history, &moves);
    let mut engine = Search {
        model,
        nodes: 0,
        limit: node_limit.max(moves.len() + 1),
        exhausted: false,
    };
    let mut scores = vec![0.0; moves.len()];
    let mut completed_depth = 0;
    // A complete iteration supplies comparable scores for every root move.
    for depth in 1..=16 {
        let mut trial = vec![0.0; moves.len()];
        for (index, &mv) in moves.iter().enumerate() {
            let mut next = position.clone();
            let mover = position.side_to_move();
            let captured = position.piece_at(mv.to as usize);
            next.make_move(mv);
            let mut line = history.to_vec();
            line.push(next.rule_history_entry_after_moved(mover, mv, captured));
            trial[index] = -engine.negamax(&next, &mut line, depth - 1, -2.0, 2.0);
            if engine.exhausted {
                break;
            }
        }
        if engine.exhausted {
            break;
        }
        scores = trial;
        completed_depth = depth;
        if scores.iter().any(|&score| score >= MATE) {
            break;
        }
    }
    let solved: Vec<Option<i8>> = moves
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
        .collect();
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
        simulations: engine.nodes,
        search_depth_avg: completed_depth as f32,
        search_depth_max: completed_depth,
        search_depth_limit: 16,
        search_depth_cutoffs: usize::from(engine.exhausted),
        candidates,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
