//! 有预算的叶子战术搜索。返回估计 WDL，不向 MCTS 写入 solved。
use super::{AzEvalOutput, AzEvalScratch, AzNnue, rule_context_features};
use super::{flip_wdl, scalar_terminal_wdl, scale_wdl_value, terminal_value, wdl_utility};
use crate::xiangqi::{Move, Position, RuleHistoryEntry};

pub(super) struct TacticalProbe<'a> {
    pub model: &'a AzNnue,
    pub nodes: usize,
    pub max_nodes: usize,
    pub value_scale: f32,
    pub best_reply: Option<Move>,
    pub control: Option<&'a super::AzSearchControl>,
    root_history_len: usize,
    scratch: AzEvalScratch,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_width_compares_every_quiet_reply_and_preserves_wdl() {
        let model = AzNnue::random(8, 31);
        let p = Position::startpos();
        let mut history = p.initial_rule_history();
        let initial_len = history.len();
        let mut expected = None;
        let mut expected_score = -2.0;
        for mv in p.legal_moves_with_rules(&history) {
            let mut next = p.clone();
            let entry = p.rule_history_entry_after_move(mv);
            next.make_move(mv);
            let h = [history[0], entry];
            let wdl = model.evaluate_wdl_with_rules(&next, &h, &next.legal_moves_with_rules(&h));
            let flipped = flip_wdl(scale_wdl_value(wdl, 0.4));
            let score = wdl_utility(flipped, 0.2);
            if score > expected_score {
                expected_score = score;
                expected = Some(flipped);
            }
        }
        let mut probe = TacticalProbe::new(&model, 1000, 0.4);
        let actual = probe
            .search(&p, &mut history, 1, 0, 0, -2.0, 2.0, 0.2)
            .unwrap();
        assert_eq!(history.len(), initial_len);
        assert_eq!(actual.value_wdl, expected.unwrap());
        assert!((actual.value - (actual.value_wdl[0] - actual.value_wdl[2])).abs() < 1e-6);
        assert_eq!(probe.nodes, p.legal_moves().len() + 1);
    }

    #[test]
    fn node_budget_aborts_without_leaking_rule_history() {
        let model = AzNnue::random(4, 31);
        let p = Position::startpos();
        let mut h = p.initial_rule_history();
        let mut probe = TacticalProbe::new(&model, 1, 1.0);
        assert!(probe.search(&p, &mut h, 1, 8, 2, -2.0, 2.0, 0.0).is_err());
        assert_eq!(probe.nodes, 1);
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].hash, p.initial_rule_history()[0].hash);
    }

    #[test]
    fn checked_leaf_cannot_stand_pat_or_stop_before_evasion() {
        let model = AzNnue::random(4, 31);
        let p = Position::from_fen("4k4/9/4R4/9/9/9/9/9/9/4K4 b").unwrap();
        assert!(p.in_check(p.side_to_move()));
        assert_eq!(p.legal_moves().len(), 2);
        let mut h = p.initial_rule_history();
        let mut probe = TacticalProbe::new(&model, 100, 1.0);
        assert!(probe.search(&p, &mut h, 0, 0, 0, -2.0, 2.0, 0.0).is_err());
        let mut probe = TacticalProbe::new(&model, 100, 1.0);
        let result = probe.search(&p, &mut h, 0, 1, 0, -2.0, 2.0, 0.0).unwrap();
        assert_eq!(probe.nodes, 3);
        assert!(result.value.is_finite());
        assert_eq!(h.len(), 1);
    }

    #[test]
    fn value_only_matches_incremental_with_active_check_context() {
        use super::super::AzEvalAccumulator;
        let mut model = AzNnue::random(16, 71);
        for (i, weight) in model.value_head_output.iter_mut().enumerate() {
            *weight = ((i % 11) as f32 + 1.0) * 0.01;
        }
        for (i, weight) in model.check_context_hidden.iter_mut().enumerate() {
            *weight = ((i % 17) as f32 + 1.0) * 0.001;
        }
        model.rebuild_check_context();
        let positions = [
            Position::startpos(),
            Position::from_fen("3rk4/9/9/9/9/9/9/3K5/9/9 w - - 0 1").unwrap(),
            Position::from_fen("4k4/9/4R4/9/9/9/9/9/9/4K4 b").unwrap(),
        ];
        let mut scratch = AzEvalScratch::new(model.arch);
        for p in positions {
            let h = p.initial_rule_history();
            let moves = p.legal_moves_with_rules(&h);
            let context = rule_context_features(&p, &h);
            let value =
                model.evaluate_value_only_with_scratch_output(&p, &moves, &context, &mut scratch);
            assert!(scratch.logits.is_empty());
            let accumulator = AzEvalAccumulator::new(&model, &p);
            let incremental = model.evaluate_incremental_with_scratch_output(
                &p,
                &accumulator.hidden_sum,
                &model.policy_accumulator(&p, p.side_to_move()),
                &moves,
                &[],
                &context,
                &mut AzEvalScratch::new(model.arch),
            );
            for (actual, expected) in value.value_wdl.iter().zip(incremental.value_wdl) {
                assert!((actual - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn stopped_tactical_search_restores_history() {
        use std::sync::{Arc, atomic::AtomicBool};
        let model = AzNnue::random(4, 31);
        let p = Position::startpos();
        let mut h = p.initial_rule_history();
        let control = super::super::AzSearchControl::new(Arc::new(AtomicBool::new(true)), None);
        let mut probe = TacticalProbe::new(&model, 4096, 1.0);
        probe.control = Some(&control);
        assert!(probe.search(&p, &mut h, 2, 8, 2, -2.0, 2.0, 0.0).is_err());
        assert_eq!(probe.nodes, 0);
        assert_eq!(h.len(), 1);
    }
}

impl<'a> TacticalProbe<'a> {
    pub fn new(model: &'a AzNnue, max_nodes: usize, value_scale: f32) -> Self {
        Self {
            model,
            nodes: 0,
            max_nodes,
            value_scale,
            best_reply: None,
            control: None,
            root_history_len: 0,
            scratch: AzEvalScratch::new(model.arch),
        }
    }

    pub fn search(
        &mut self,
        p: &Position,
        h: &mut Vec<RuleHistoryEntry>,
        quiet: usize,
        plies: usize,
        checks: usize,
        alpha: f32,
        beta: f32,
        draw_score: f32,
    ) -> Result<AzEvalOutput, ()> {
        self.best_reply = None;
        self.root_history_len = h.len();
        self.search_inner(p, h, quiet, plies, checks, alpha, beta, draw_score)
    }

    fn search_inner(
        &mut self,
        p: &Position,
        h: &mut Vec<RuleHistoryEntry>,
        quiet: usize,
        plies: usize,
        checks: usize,
        mut alpha: f32,
        beta: f32,
        draw_score: f32,
    ) -> Result<AzEvalOutput, ()> {
        if self.nodes >= self.max_nodes {
            return Err(());
        }
        if self.nodes % 32 == 0 && self.control.is_some_and(|control| control.should_stop()) {
            return Err(());
        }
        self.nodes += 1;
        if let Some(value) = terminal_value(p, h) {
            return Ok(AzEvalOutput {
                moves_left: 0.0,
                value,
                value_wdl: scalar_terminal_wdl(value),
            });
        }
        let mut moves = p.legal_moves_with_rules(h);
        if moves.is_empty() {
            return Ok(AzEvalOutput {
                moves_left: 0.0,
                value: -1.0,
                value_wdl: [0.0, 0.0, 1.0],
            });
        }
        let checked = p.in_check(p.side_to_move());
        if quiet == 0 && plies == 0 && checked {
            // 未走完应将，不能用这个截断点覆盖原来的 MCTS 估计。
            return Err(());
        }
        let stand_pat = quiet == 0 && !checked;
        let mut best = if stand_pat {
            let mut value = self.model.evaluate_value_only_with_scratch_output(
                p,
                &moves,
                &rule_context_features(p, h),
                &mut self.scratch,
            );
            value.value_wdl = scale_wdl_value(value.value_wdl, self.value_scale);
            value.value *= self.value_scale;
            value
        } else {
            AzEvalOutput {
                moves_left: 0.0,
                value: 0.0,
                value_wdl: [0.0, 1.0, 0.0],
            }
        };
        if quiet == 0 && plies == 0 {
            return Ok(best);
        }
        let mut best_score = if stand_pat {
            wdl_utility(best.value_wdl, draw_score)
        } else {
            -2.0
        };
        if stand_pat {
            if best_score >= beta {
                return Ok(best);
            }
            alpha = alpha.max(best_score);
            moves.retain(|&mv| {
                p.is_capture(mv) || (checks > 0 && p.gives_check_after_move_fast(mv))
            });
        }
        moves.sort_by_cached_key(|&mv| {
            std::cmp::Reverse((p.gives_check_after_move_fast(mv), p.is_capture(mv)))
        });
        for mv in moves {
            let checking = p.gives_check_after_move_fast(mv);
            let captured = p.piece_at(mv.to as usize);
            let mover = p.side_to_move();
            let mut next = p.clone();
            next.make_move(mv);
            h.push(next.rule_history_entry_after_moved(mover, mv, captured));
            let child = self.search_inner(
                &next,
                h,
                quiet.saturating_sub(1),
                if quiet > 0 { plies } else { plies - 1 },
                if quiet == 0 && checking {
                    checks.saturating_sub(1)
                } else {
                    checks
                },
                -beta,
                -alpha,
                -draw_score,
            );
            h.pop();
            let child = child?;
            let eval = AzEvalOutput {
                moves_left: child.moves_left + 1.0,
                value: -child.value,
                value_wdl: flip_wdl(child.value_wdl),
            };
            let score = wdl_utility(eval.value_wdl, draw_score);
            if score > best_score {
                best = eval;
                best_score = score;
                if h.len() == self.root_history_len {
                    self.best_reply = Some(mv);
                }
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                break;
            }
        }
        Ok(best)
    }
}
