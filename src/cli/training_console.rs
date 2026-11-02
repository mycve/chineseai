use chineseai::az::{AzHoldoutReport, AzLoopReport, PX0_CYCLE_STEPS};
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::io::IsTerminal;

pub struct TrainingConsole {
    interactive: bool,
    panel: MultiProgress,
    cycle: ProgressBar,
    production: ProgressBar,
    loss: ProgressBar,
    value: ProgressBar,
    mating: ProgressBar,
    test: ProgressBar,
    arena: ProgressBar,
}

impl TrainingConsole {
    pub fn new(steps: usize) -> Self {
        let interactive = std::io::stdout().is_terminal();
        let panel = MultiProgress::with_draw_target(if interactive {
            ProgressDrawTarget::stdout()
        } else {
            ProgressDrawTarget::hidden()
        });
        // 有些伪终端被std识别为TTY，但绘图库无法显示；此时必须保留文本输出。
        let interactive = interactive && !panel.is_hidden();
        let cycle = panel.add(ProgressBar::new(PX0_CYCLE_STEPS as u64));
        cycle.set_style(ProgressStyle::with_template("{prefix} [{wide_bar}] {pos}/{len}").unwrap());
        let row = || {
            let bar = panel.add(ProgressBar::new_spinner());
            bar.set_style(ProgressStyle::with_template("{msg}").unwrap());
            bar
        };
        let production = row();
        let loss = row();
        let value = row();
        let mating = row();
        let test = row();
        let arena = row();
        let cycle_number = steps / PX0_CYCLE_STEPS + 1;
        cycle.set_prefix(format!("cycle {cycle_number}"));
        cycle.set_position((steps % PX0_CYCLE_STEPS) as u64);
        production.set_message("等待自博弈样本");
        test.set_message("test: 等待首次留出测试");
        arena.set_message("arena: 尚无评测结果");
        Self {
            interactive,
            panel,
            cycle,
            production,
            loss,
            value,
            mating,
            test,
            arena,
        }
    }

    pub fn update(
        &self,
        update: usize,
        report: &AzLoopReport,
        games_total: u64,
        draws: usize,
        checkpoint: bool,
    ) {
        let production = format!(
            "update {update:04}: selfplay={:.1}s games={} total={} pool={}/{} chunks(train/test)={}/{} R/B/D={}/{}/{} cutoff={} failed={}{}",
            report.total_seconds,
            report.games,
            games_total,
            report.pool_samples,
            report.pool_capacity,
            report.training_chunks,
            report.test_chunks,
            report.red_wins,
            report.black_wins,
            draws,
            report.terminal_max_plies,
            report.terminal_search_no_move,
            if checkpoint { " checkpoint=saved" } else { "" }
        );
        let loss = format!(
            "train: samples={} loss={:.4} WDL={:.4} policy_KL={:.4} MLH={:.4}/{} MLH_pair={:.1}%/{}",
            report.train_samples,
            report.loss,
            report.value_loss,
            report.policy_kl,
            report.moves_left_loss,
            report.moves_left_samples,
            report.moves_left_pair_accuracy * 100.0,
            report.moves_left_pairs
        );
        let value = format!(
            "value: RMSE={:.4} corr={:.3} lr={:.6} sims={:.1} train={:.1}s",
            report.value_mse.max(0.0).sqrt(),
            report.value_corr,
            report.learning_rate,
            report.avg_search_simulations,
            report.train_seconds
        );
        let mating = mating_line(report);
        let cycle_number = report.training_steps.saturating_sub(1) / PX0_CYCLE_STEPS + 1;
        if self.interactive {
            self.cycle.set_prefix(format!("cycle {cycle_number}"));
            self.cycle.set_position(
                report
                    .training_steps
                    .saturating_sub((cycle_number - 1) * PX0_CYCLE_STEPS) as u64,
            );
            self.production.set_message(production);
            self.loss.set_message(loss);
            self.value.set_message(value);
            self.mating.set_message(mating);
        } else {
            println!(
                "cycle={} step={} {production} {loss} {value}",
                cycle_number, report.training_steps
            );
            println!("{mating}");
        }
    }

    pub fn test(&self, check: &AzHoldoutReport) {
        let value = if check.value_samples == 0 {
            "WDL=NA RMSE=NA".to_owned()
        } else {
            format!("WDL={:.4} RMSE={:.4}", check.value_loss, check.value_rmse)
        };
        let line = format!(
            "test: step={} samples={} value_samples={} loss={:.4} policy_KL={:.4} {value} MLH_RMSE={:.1}/{} MLH_pair={:.1}%/{} 杀法CE={}/{} 目标CE={}/{}",
            check.step,
            check.samples,
            check.value_samples,
            check.loss,
            check.policy_kl,
            check.moves_left_rmse,
            check.moves_left_samples,
            check.moves_left_pair_accuracy * 100.0,
            check.moves_left_pairs,
            if check.mating_samples > 0 {
                format!("{:.4}", check.mating_loss)
            } else {
                "NA".into()
            },
            check.mating_samples,
            if check.goal_samples > 0 {
                format!("{:.4}", check.goal_loss)
            } else {
                "NA".into()
            },
            check.goal_samples
        );
        if self.interactive {
            self.test.set_message(line);
        } else {
            println!("{line}");
        }
    }

    pub fn arena(&self, line: String) {
        if self.interactive {
            self.arena.set_message(line);
        } else {
            println!("{line}");
        }
    }

    pub fn event(&self, line: String) {
        if self.interactive {
            let _ = self.panel.println(line);
        } else {
            println!("{line}");
        }
    }

    pub fn finish(&mut self) {
        for row in [
            &self.cycle,
            &self.production,
            &self.loss,
            &self.value,
            &self.mating,
            &self.test,
            &self.arena,
        ] {
            row.abandon();
        }
    }
}

fn mating_line(report: &AzLoopReport) -> String {
    let stats = &report.mating;
    let counts = chineseai::xiangqi::MatingPattern::ALL
        .into_iter()
        .enumerate()
        .filter_map(|(i, pattern)| {
            let count = stats.pattern_counts[i];
            (count > 0).then(|| {
                format!(
                    "{}={}({:.1}%/{:.1}%)",
                    pattern.name(),
                    count,
                    count as f32 * 100.0 / stats.games.max(1) as f32,
                    count as f32 * 100.0 / stats.checkmates.max(1) as f32,
                )
            })
        })
        .collect::<Vec<_>>()
        .join(" ");
    let coverage = if stats.games > 0 {
        format!(
            "{:.1}%",
            stats.labeled_games as f32 * 100.0 / stats.games as f32
        )
    } else {
        "NA".into()
    };
    let loss = if report.mating_samples > 0 {
        format!("{:.4}", report.mating_loss)
    } else {
        "NA".into()
    };
    let goal_loss = if report.goal_samples > 0 {
        format!("{:.4}", report.goal_loss)
    } else {
        "NA".into()
    };
    let goal_counts = ["重炮", "铁门栓"]
        .into_iter()
        .enumerate()
        .filter_map(|(i, name)| {
            let attempts = stats.goal_attempts[i];
            (attempts > 0).then(|| {
                format!(
                    "{name}目标={}/{}({:.1}%)未知={}",
                    stats.goal_successes[i],
                    attempts,
                    stats.goal_successes[i] as f32 * 100.0 / attempts as f32,
                    stats.goal_unknown[i]
                )
            })
        })
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "本批杀法：实际将死={} 困毙={} 未知={} 覆盖={} CE={}/{} 目标CE={}/{} {} 计数(批内全部%/批内将死%)：{}",
        stats.checkmates,
        stats.stalemates,
        stats.unknown_games,
        coverage,
        loss,
        report.mating_samples,
        goal_loss,
        report.goal_samples,
        goal_counts,
        if counts.is_empty() {
            "无匹配标签"
        } else {
            &counts
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mating_console_distinguishes_unknown_and_overlapping_pattern_rates() {
        let mut report = AzLoopReport::default();
        report.mating.games = 4;
        report.mating.checkmates = 2;
        report.mating.unknown_games = 2;
        report.mating.labeled_games = 2;
        report.mating.pattern_counts[0] = 2;
        report.mating.pattern_counts[1] = 2;
        let line = mating_line(&report);
        assert!(line.contains("实际将死=2 困毙=0 未知=2 覆盖=50.0%"));
        assert!(line.contains("重炮=2(50.0%/100.0%)"));
        assert!(line.contains("马后炮=2(50.0%/100.0%)"));
        assert!(line.contains("CE=NA/0"));
    }
}
