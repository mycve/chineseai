use chineseai::ab::{AbEvolveReport, AbHoldoutReport};
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::io::IsTerminal;

pub struct TrainingConsole {
    interactive: bool,
    panel: MultiProgress,
    production: ProgressBar,
    loss: ProgressBar,
    value: ProgressBar,
    test: ProgressBar,
    arena: ProgressBar,
}

impl TrainingConsole {
    pub fn new(_steps: usize) -> Self {
        let interactive = std::io::stdout().is_terminal();
        let panel = MultiProgress::with_draw_target(if interactive {
            ProgressDrawTarget::stdout()
        } else {
            ProgressDrawTarget::hidden()
        });
        // 有些伪终端被std识别为TTY，但绘图库无法显示；此时必须保留文本输出。
        let interactive = interactive && !panel.is_hidden();
        let row = || {
            let bar = panel.add(ProgressBar::new_spinner());
            bar.set_style(ProgressStyle::with_template("{msg}").unwrap());
            bar
        };
        let production = row();
        let loss = row();
        let value = row();
        let test = row();
        let arena = row();
        production.set_message("等待自博弈样本");
        test.set_message("test: 等待首次留出测试");
        arena.set_message("arena: 尚无评测结果");
        Self {
            interactive,
            panel,
            production,
            loss,
            value,
            test,
            arena,
        }
    }

    pub fn update(
        &self,
        update: usize,
        report: &AbEvolveReport,
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
            "train: samples={} loss={:.4} WDL={:.4}",
            report.train_samples, report.loss, report.value_loss
        );
        let value = format!(
            "value: RMSE={:.4} corr={:.3} lr={:.6} nodes={:.1} train={:.1}s",
            report.value_mse.max(0.0).sqrt(),
            report.value_corr,
            report.learning_rate,
            report.avg_search_nodes,
            report.train_seconds
        );
        if self.interactive {
            self.production.set_message(production);
            self.loss.set_message(loss);
            self.value.set_message(value);
        } else {
            println!("step={} {production} {loss} {value}", report.training_steps);
        }
    }

    pub fn test(&self, check: &AbHoldoutReport) {
        let value = if check.value_samples == 0 {
            "WDL=NA RMSE=NA".to_owned()
        } else {
            format!("WDL={:.4} RMSE={:.4}", check.value_loss, check.value_rmse)
        };
        let line = format!(
            "test: step={} samples={} value_samples={} loss={:.4} {value}",
            check.step, check.samples, check.value_samples, check.loss
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
            &self.production,
            &self.loss,
            &self.value,
            &self.test,
            &self.arena,
        ] {
            row.abandon();
        }
    }
}
