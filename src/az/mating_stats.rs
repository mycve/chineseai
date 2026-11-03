use crate::xiangqi::{MateClassification, MateKind, MatingPattern};

use super::{AzMatingGoal, MATING_HEAD_SIZE, MATING_TARGET_COUNT};

/// 按实际观察到的终局统计；各杀法可重叠，pattern_counts 不应相加作为总盘数。
#[derive(Clone, Copy, Debug, Default)]
pub struct AzMatingStats {
    pub games: usize,
    pub labeled_games: usize,
    pub checkmates: usize,
    pub stalemates: usize,
    pub unknown_games: usize,
    /// 顺序与 MatingPattern::ALL 相同。
    pub pattern_counts: [usize; MATING_HEAD_SIZE],
    /// 按 MatingTarget::ALL 排序；尝试分母保留失败和未知结局。
    pub goal_attempts: [usize; MATING_TARGET_COUNT],
    pub goal_successes: [usize; MATING_TARGET_COUNT],
    /// 成功对局从本盘指令起点到实际将死的半回合数总和。
    pub goal_success_plies: [usize; MATING_TARGET_COUNT],
    pub goal_unknown: [usize; MATING_TARGET_COUNT],
    /// [红方执行、黑方执行]，用于检查目标执行方的实际分布。
    pub goal_side_attempts: [usize; 2],
    /// 各目标执行方的 [胜、负、和、未知/截断]，以实际结果而非搜索证明计数。
    pub goal_results: [[usize; 4]; MATING_TARGET_COUNT],
    /// 普通对局的 [红胜、黑胜、和、未知/截断]。
    pub ordinary_results: [usize; 4],
}

impl AzMatingStats {
    pub fn add_assign(&mut self, other: &Self) {
        self.games += other.games;
        self.labeled_games += other.labeled_games;
        self.checkmates += other.checkmates;
        self.stalemates += other.stalemates;
        self.unknown_games += other.unknown_games;
        for (count, other_count) in self.pattern_counts.iter_mut().zip(other.pattern_counts) {
            *count += other_count;
        }
        for i in 0..MATING_TARGET_COUNT {
            self.goal_attempts[i] += other.goal_attempts[i];
            self.goal_successes[i] += other.goal_successes[i];
            self.goal_success_plies[i] += other.goal_success_plies[i];
            self.goal_unknown[i] += other.goal_unknown[i];
            for result in 0..4 {
                self.goal_results[i][result] += other.goal_results[i][result];
            }
        }
        for i in 0..4 {
            self.ordinary_results[i] += other.ordinary_results[i];
        }
        for i in 0..2 {
            self.goal_side_attempts[i] += other.goal_side_attempts[i];
        }
    }

    pub fn goal_result_totals(&self) -> [usize; 4] {
        std::array::from_fn(|result| self.goal_results.iter().map(|row| row[result]).sum())
    }

    pub(super) fn record_result(&mut self, goal: AzMatingGoal, result: Option<f32>) {
        let owner_sign = if goal.enabled() && goal.side == crate::xiangqi::Color::Black {
            -1.0
        } else {
            1.0
        };
        let outcome = match result.map(|result| result * owner_sign) {
            Some(result) if result > 0.0 => 0,
            Some(result) if result < 0.0 => 1,
            Some(_) => 2,
            None => 3,
        };
        if goal.enabled() {
            self.goal_results[goal.target.index().unwrap()][outcome] += 1;
        } else {
            self.ordinary_results[outcome] += 1;
        }
    }

    pub fn rate_all(&self, index: usize) -> f32 {
        self.pattern_counts[index] as f32 / self.games.max(1) as f32
    }

    pub fn rate_checkmates(&self, index: usize) -> f32 {
        self.pattern_counts[index] as f32 / self.checkmates.max(1) as f32
    }

    pub(super) fn record(&mut self, classification: Option<&MateClassification>, observed: bool) {
        self.games += 1;
        if !observed {
            self.unknown_games += 1;
            return;
        }
        self.labeled_games += 1;
        let Some(classification) = classification else {
            return;
        };
        match classification.kind {
            MateKind::Stalemate => self.stalemates += 1,
            MateKind::Checkmate => {
                self.checkmates += 1;
                for (index, pattern) in MatingPattern::ALL.into_iter().enumerate() {
                    self.pattern_counts[index] +=
                        usize::from(classification.patterns.contains(&pattern));
                }
            }
        }
    }

    pub(super) fn record_goal(
        &mut self,
        goal: AzMatingGoal,
        classification: Option<&MateClassification>,
        observed: bool,
        plies: usize,
    ) {
        if !goal.enabled() {
            return;
        }
        let Some(index) = goal.target.index() else {
            return;
        };
        self.goal_attempts[index] += 1;
        self.goal_side_attempts[usize::from(goal.side == crate::xiangqi::Color::Black)] += 1;
        if !observed {
            self.goal_unknown[index] += 1;
        } else if goal_succeeded(goal, classification) {
            self.goal_successes[index] += 1;
            self.goal_success_plies[index] += plies;
        }
    }
}

pub(super) fn goal_succeeded(
    goal: AzMatingGoal,
    classification: Option<&MateClassification>,
) -> bool {
    classification.is_some_and(|mate| goal.matches_mate(mate))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::az::MatingTarget;
    use crate::xiangqi::Position;

    #[test]
    fn goal_attempts_include_observed_failures_and_unknown_games() {
        let mate = Position::from_fen("2bakab2/9/9/4C4/9/4C4/9/9/9/4K4 b")
            .unwrap()
            .classify_mate()
            .unwrap();
        let goal = AzMatingGoal {
            target: MatingTarget::DoubleCannon,
            side: crate::xiangqi::Color::Red,
            strength: 1.0,
        };
        let mut stats = AzMatingStats::default();
        stats.record_goal(goal, Some(&mate), true, 30);
        stats.record_goal(goal, None, true, 100);
        stats.record_goal(goal, None, false, 200);
        let mut merged = AzMatingStats::default();
        merged.add_assign(&stats);
        assert_eq!(merged.goal_attempts[..2], [3, 0]);
        assert_eq!(merged.goal_successes[..2], [1, 0]);
        assert_eq!(merged.goal_success_plies[..2], [30, 0]);
        assert_eq!(merged.goal_unknown[..2], [1, 0]);
        stats.record_goal(
            AzMatingGoal {
                side: crate::xiangqi::Color::Black,
                ..goal
            },
            Some(&mate),
            true,
            40,
        );
        assert_eq!(stats.goal_successes[..2], [1, 0]);
    }

    #[test]
    fn merging_keeps_pattern_counts_and_both_rate_denominators() {
        let mate = Position::from_fen("2bakab2/9/9/4C4/9/4C4/9/9/9/4K4 b")
            .unwrap()
            .classify_mate()
            .unwrap();
        let mut first = AzMatingStats::default();
        first.record(Some(&mate), true);
        first.record(None, false);
        let mut second = AzMatingStats::default();
        second.record(None, true);
        second.add_assign(&first);
        assert_eq!(second.games, 3);
        assert_eq!(second.labeled_games, 2);
        assert_eq!(second.unknown_games, 1);
        assert_eq!(second.checkmates, 1);
        for pattern in [MatingPattern::DoubleCannon] {
            let index = MatingPattern::ALL
                .iter()
                .position(|&p| p == pattern)
                .unwrap();
            assert_eq!(second.pattern_counts[index], 1);
            assert_eq!(second.rate_all(index), 1.0 / 3.0);
            assert_eq!(second.rate_checkmates(index), 1.0);
        }
        assert_eq!(AzMatingStats::default().rate_all(0), 0.0);
        assert_eq!(AzMatingStats::default().rate_checkmates(0), 0.0);
    }
    #[test]
    fn actual_results_keep_goal_owner_and_ordinary_colors_separate() {
        use crate::xiangqi::Color;
        let goal = AzMatingGoal {
            target: MatingTarget::DoubleCannon,
            side: Color::Red,
            strength: 0.8,
        };
        let mut stats = AzMatingStats::default();
        for side in [Color::Red, Color::Black] {
            for result in [Some(1.0), Some(-1.0), Some(0.0), None] {
                stats.record_result(AzMatingGoal { side, ..goal }, result);
            }
        }
        assert_eq!(stats.goal_results[0], [2, 2, 2, 2]);
        assert_eq!(stats.ordinary_results, [0; 4]);
        stats.record_result(
            AzMatingGoal {
                side: Color::Black,
                ..goal
            },
            Some(-1.0),
        );
        assert_eq!(stats.goal_results[0][0], 3, "黑方执行时黑胜算执行方获胜");
        for result in [Some(1.0), Some(-1.0), Some(0.0), None] {
            stats.record_result(
                AzMatingGoal {
                    strength: 0.0,
                    ..goal
                },
                result,
            );
        }
        assert_eq!(stats.ordinary_results, [1; 4]);
        let mut merged = AzMatingStats::default();
        merged.add_assign(&stats);
        merged.add_assign(&stats);
        assert_eq!(merged.goal_result_totals(), [6, 4, 4, 4]);
        assert_eq!(merged.ordinary_results, [2; 4]);
    }
}
