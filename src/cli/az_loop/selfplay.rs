use chineseai::az::{AzLoopReport, AzNnue, AzSelfplayData};
use std::sync::{Arc, RwLock};

pub(crate) struct SelfplayBatch {
    pub(crate) data: AzSelfplayData,
}

pub(crate) struct TrainerEvent {
    pub(crate) report: AzLoopReport,
    pub(crate) candidate_model: AzNnue,
}

pub(crate) struct SharedSelfplayModel {
    pub(crate) version: u64,
    pub(crate) learner_update: u32,
    pub(crate) model: Arc<AzNnue>,
}

pub(crate) fn publish_selfplay_model(
    shared_model: &RwLock<SharedSelfplayModel>,
    model: Arc<AzNnue>,
    learner_update: usize,
) -> u64 {
    let mut shared = shared_model
        .write()
        .unwrap_or_else(|_| panic!("shared selfplay model poisoned"));
    shared.model = model;
    shared.version = shared.version.wrapping_add(1);
    shared.learner_update = learner_update.min(u32::MAX as usize) as u32;
    shared.version
}

#[derive(Default)]
pub(crate) struct PendingTrainingData {
    pub(crate) collection_seconds: f32,
    pub(crate) selfplay: AzSelfplayData,
}

impl PendingTrainingData {
    pub(crate) fn push(&mut self, batch: SelfplayBatch) {
        self.selfplay.add_assign(&batch.data);
    }
}

pub(crate) fn build_async_training_report(
    pending: PendingTrainingData,
    selfplay_games: usize,
    stats: chineseai::az::AzTrainStats,
    learning_rate: f32,
    train_data_len: usize,
    train_seconds: f32,
    pool_samples: usize,
    pool_capacity: usize,
    replay_window: chineseai::az::AzReplayWindowStats,
    target_entropy: f32,
) -> AzLoopReport {
    let selfplay_samples = pending.selfplay.samples.len();
    let total_seconds = pending.collection_seconds.max(1.0e-6);
    let train_stat_samples = stats
        .phase_value
        .iter()
        .map(|p| p.samples)
        .sum::<usize>()
        .max(1) as f32;
    let root_visit_entropy =
        pending.selfplay.entropy_all_sum / pending.selfplay.entropy_all_count.max(1) as f32;
    let shape_count = pending.selfplay.shape_count.max(1) as f32;
    let opening_shape_count = pending.selfplay.opening_shape_count.max(1) as f32;
    let sampled_moves = pending.selfplay.sampled_moves.max(1) as f32;
    let search_count = pending.selfplay.search_simulations.searches.max(1) as f32;
    let value_pred_mean = stats.value_pred_sum / train_stat_samples;
    let value_target_mean = stats.value_target_sum / train_stat_samples;
    let value_pred_var =
        (stats.value_pred_sq_sum / train_stat_samples - value_pred_mean * value_pred_mean).max(0.0);
    let value_target_var = (stats.value_target_sq_sum / train_stat_samples
        - value_target_mean * value_target_mean)
        .max(0.0);
    let value_cov =
        stats.value_pred_target_sum / train_stat_samples - value_pred_mean * value_target_mean;
    let value_corr =
        value_cov / (value_pred_var.max(1.0e-12).sqrt() * value_target_var.max(1.0e-12).sqrt());
    let value_calibration = value_cov / value_pred_var.max(1.0e-12);
    let value_report = |phase_stats: chineseai::az::AzValueMomentStats| {
        let count = phase_stats.samples.max(1) as f32;
        let pred_mean = phase_stats.pred_sum / count;
        let target_mean = phase_stats.target_sum / count;
        let pred_var = (phase_stats.pred_sq_sum / count - pred_mean * pred_mean).max(0.0);
        let target_var = (phase_stats.target_sq_sum / count - target_mean * target_mean).max(0.0);
        let covariance = phase_stats.pred_target_sum / count - pred_mean * target_mean;
        chineseai::az::AzPhaseValueReport {
            samples: phase_stats.samples,
            rmse: (phase_stats.error_sq_sum / count).max(0.0).sqrt(),
            corr: (covariance / (pred_var.max(1.0e-12).sqrt() * target_var.max(1.0e-12).sqrt()))
                .clamp(-1.0, 1.0),
            calibration: covariance / pred_var.max(1.0e-12),
        }
    };
    let phase_value = stats.phase_value.map(value_report);
    let source_phase_value = stats.source_phase_value.map(value_report);
    let start_source_rate = pending
        .selfplay
        .start_games
        .map(|count| count as f32 / selfplay_games.max(1) as f32);
    let start_phase_ply = std::array::from_fn(|source| {
        pending.selfplay.start_phase_ply_sum[source] as f32
            / pending.selfplay.start_games[source].max(1) as f32
    });
    let start_age = std::array::from_fn(|source| {
        pending.selfplay.start_age_sum[source] as f32
            / pending.selfplay.start_games[source].max(1) as f32
    });
    let start_temperature = std::array::from_fn(|source| {
        pending.selfplay.start_temperature_sum[source]
            / pending.selfplay.start_games[source].max(1) as f32
    });
    AzLoopReport {
        training_steps: 0,
        training_chunks: 0,
        test_chunks: 0,
        holdout_checks: Vec::new(),
        cycle_complete: false,
        games: selfplay_games,
        samples: selfplay_samples,
        avg_search_simulations: pending.selfplay.search_simulations.simulations_sum as f32
            / search_count,
        search_nn_cache_hit_rate: pending.selfplay.search_simulations.nn_cache_hits as f32
            / (pending.selfplay.search_simulations.nn_evaluations
                + pending.selfplay.search_simulations.nn_cache_hits)
                .max(1) as f32,
        red_wins: pending.selfplay.red_wins,
        black_wins: pending.selfplay.black_wins,
        draws: pending.selfplay.draws,
        avg_plies: if selfplay_games == 0 {
            0.0
        } else {
            pending.selfplay.plies_total as f32 / selfplay_games as f32
        },
        selfplay_start_source_rate: start_source_rate,
        selfplay_start_phase_ply: start_phase_ply,
        selfplay_start_age: start_age,
        selfplay_start_age_max: pending.selfplay.start_age_max,
        selfplay_start_temperature: start_temperature,
        loss: stats.loss,
        learning_rate,
        value_loss: stats.value_loss,
        mating: pending.selfplay.mating,
        mating_loss: stats.mating_loss,
        mating_samples: stats.mating_samples,
        mating_target_sum: stats.mating_target_sum,
        mating_pred_sum: stats.mating_pred_sum,
        mating_brier_sum: stats.mating_brier_sum,
        goal_loss: stats.goal_loss,
        goal_samples: stats.goal_samples,
        goal_target_sum: stats.goal_target_sum,
        goal_pred_sum: stats.goal_pred_sum,
        goal_brier_sum: stats.goal_brier_sum,
        goal_imitation_loss: stats.goal_imitation_loss / stats.goal_imitation_samples.max(1) as f32,
        goal_imitation_samples: stats.goal_imitation_samples,
        moves_left_loss: stats.moves_left_loss,
        moves_left_samples: stats.moves_left_samples,
        moves_left_pairs: stats.moves_left_pairs,
        moves_left_pair_accuracy: stats.moves_left_correct_pairs as f32
            / stats.moves_left_pairs.max(1) as f32,
        value_mse: stats.value_error_sq_sum / train_stat_samples,
        value_pred_mean,
        value_target_mean,
        value_pred_rms: (stats.value_pred_sq_sum / train_stat_samples)
            .max(0.0)
            .sqrt(),
        value_target_rms: (stats.value_target_sq_sum / train_stat_samples)
            .max(0.0)
            .sqrt(),
        value_corr: value_corr.clamp(-1.0, 1.0),
        value_calibration,
        phase_value,
        source_phase_value,
        policy_ce: stats.policy_ce,
        policy_target_entropy: target_entropy,
        policy_kl: stats.policy_ce - target_entropy,
        root_visit_entropy,
        entropy_opening: pending.selfplay.entropy_opening_sum
            / pending.selfplay.entropy_opening_count.max(1) as f32,
        entropy_mid: pending.selfplay.entropy_mid_sum
            / pending.selfplay.entropy_mid_count.max(1) as f32,
        raw_prior_top1: pending.selfplay.raw_prior_top1_sum / shape_count,
        raw_prior_top2: pending.selfplay.raw_prior_top2_sum / shape_count,
        policy_top1: pending.selfplay.policy_top1_sum / shape_count,
        policy_top2: pending.selfplay.policy_top2_sum / shape_count,
        root_q_gap: pending.selfplay.q_gap_sum / shape_count,
        root_q_top1_abs: pending.selfplay.q_top1_abs_sum / shape_count,
        visited_actions: pending.selfplay.visited_actions_sum as f32 / shape_count,
        opening_raw_prior_top1: pending.selfplay.opening_raw_prior_top1_sum / opening_shape_count,
        opening_raw_prior_top2: pending.selfplay.opening_raw_prior_top2_sum / opening_shape_count,
        opening_policy_top1: pending.selfplay.opening_policy_top1_sum / opening_shape_count,
        opening_policy_top2: pending.selfplay.opening_policy_top2_sum / opening_shape_count,
        opening_q_gap: pending.selfplay.opening_q_gap_sum / opening_shape_count,
        opening_q_top1_abs: pending.selfplay.opening_q_top1_abs_sum / opening_shape_count,
        opening_visited_actions: pending.selfplay.opening_visited_actions_sum as f32
            / opening_shape_count,
        sampled_best_rate: pending.selfplay.sampled_best_moves as f32 / sampled_moves,
        avg_best_played_q_gap: pending.selfplay.best_played_q_gap_sum / sampled_moves,
        avg_played_top_visit_ratio: pending.selfplay.played_top_visit_ratio_sum / sampled_moves,
        avg_best_q: pending.selfplay.best_q_sum / sampled_moves,
        avg_played_q: pending.selfplay.played_q_sum / sampled_moves,
        train_seconds,
        total_seconds,
        games_per_second: selfplay_games as f32 / total_seconds.max(1e-6),
        samples_per_second: selfplay_samples as f32 / total_seconds.max(1e-6),
        train_samples_per_second: train_data_len as f32 / train_seconds.max(1e-6),
        train_samples: train_data_len,
        pool_samples,
        pool_capacity,
        replay_chunks: replay_window.chunks,
        replay_oldest_update: replay_window.oldest_generation_update,
        replay_newest_update: replay_window.newest_generation_update,
        replay_avg_update: replay_window.avg_generation_update,
        replay_window_games: replay_window.window_games,
        replay_recent_window_fraction: replay_window.recent_window_sample_fraction,
        terminal_no_legal_moves: pending.selfplay.terminal.no_legal_moves,
        terminal_checkmate: pending.selfplay.terminal.checkmate,
        terminal_stalemate: pending.selfplay.terminal.stalemate,
        terminal_rule_blocked: pending.selfplay.terminal.rule_blocked,
        terminal_search_no_move: pending.selfplay.terminal.search_no_move,
        terminal_red_general_missing: pending.selfplay.terminal.red_general_missing,
        terminal_black_general_missing: pending.selfplay.terminal.black_general_missing,
        terminal_rule_draw: pending.selfplay.terminal.rule_draw,
        terminal_rule_draw_natural_limit: pending.selfplay.terminal.rule_draw_natural_limit,
        terminal_rule_draw_insufficient_material: pending
            .selfplay
            .terminal
            .rule_draw_insufficient_material,
        terminal_rule_draw_repetition: pending.selfplay.terminal.rule_draw_repetition,
        terminal_rule_draw_mutual_long_check: pending.selfplay.terminal.rule_draw_mutual_long_check,
        terminal_rule_draw_mutual_long_chase: pending.selfplay.terminal.rule_draw_mutual_long_chase,
        terminal_rule_win_red: pending.selfplay.terminal.rule_win_red,
        terminal_rule_win_black: pending.selfplay.terminal.rule_win_black,
        terminal_max_plies: pending.selfplay.terminal.max_plies,
        terminal_search_proven: pending.selfplay.terminal.search_proven,
    }
}

/// 标签允许重叠；分母分别为本批全部对局、实际将死与有效监督样本。
pub(crate) fn mating_metrics(report: &AzLoopReport) -> Vec<(String, f32)> {
    let mut metrics = mating_count_metrics(&report.mating, "mating");
    metrics.push(("train/mating_samples".into(), report.mating_samples as f32));
    metrics.push(("train/goal_samples".into(), report.goal_samples as f32));
    if report.goal_samples > 0 {
        let count = report.goal_samples as f32;
        metrics.extend([
            ("train/goal_bce".into(), report.goal_loss),
            (
                "train/goal_target_rate".into(),
                report.goal_target_sum / count,
            ),
            ("train/goal_pred_mean".into(), report.goal_pred_sum / count),
            ("train/goal_brier".into(), report.goal_brier_sum / count),
            ("train/goal_imitation_ce".into(), report.goal_imitation_loss),
            (
                "train/goal_imitation_samples".into(),
                report.goal_imitation_samples as f32,
            ),
        ]);
    }
    if report.mating_samples > 0 {
        metrics.push(("train/mating_bce".into(), report.mating_loss));
    }
    if report.train_samples > 0 {
        metrics.push((
            "train/mating_supervision_coverage".into(),
            report.mating_samples as f32 / report.train_samples as f32,
        ));
    }
    for (i, pattern) in chineseai::xiangqi::MatingPattern::ALL
        .into_iter()
        .enumerate()
    {
        let name = pattern.name();
        if report.mating_samples > 0 {
            let samples = report.mating_samples as f32;
            metrics.extend([
                (
                    format!("train/mating/{name}/target_rate"),
                    report.mating_target_sum[i] / samples,
                ),
                (
                    format!("train/mating/{name}/pred_mean"),
                    report.mating_pred_sum[i] / samples,
                ),
                (
                    format!("train/mating/{name}/brier"),
                    report.mating_brier_sum[i] / samples,
                ),
            ]);
        }
    }
    metrics
}

/// 同一口径用于本批 mating/* 和本进程累计 mating_total/*。
pub(crate) fn mating_count_metrics(
    stats: &chineseai::az::AzMatingStats,
    prefix: &str,
) -> Vec<(String, f32)> {
    let mut metrics = vec![
        (format!("{prefix}/games"), stats.games as f32),
        (
            format!("{prefix}/labeled_games"),
            stats.labeled_games as f32,
        ),
        (format!("{prefix}/checkmates"), stats.checkmates as f32),
        (format!("{prefix}/stalemates"), stats.stalemates as f32),
        (
            format!("{prefix}/unknown_games"),
            stats.unknown_games as f32,
        ),
    ];
    for (i, target) in chineseai::az::MatingTarget::ALL.into_iter().enumerate() {
        let name = target.name();
        let attempts = stats.goal_attempts[i] as f32;
        metrics.extend([
            (format!("{prefix}/goal/{name}/attempts"), attempts),
            (
                format!("{prefix}/goal/{name}/successes"),
                stats.goal_successes[i] as f32,
            ),
            (
                format!("{prefix}/goal/{name}/unknown"),
                stats.goal_unknown[i] as f32,
            ),
        ]);
        for (result, count) in ["wins", "losses", "draws", "unfinished"]
            .into_iter()
            .zip(stats.goal_results[i])
        {
            metrics.push((format!("{prefix}/goal/{name}/{result}"), count as f32));
        }
        if stats.goal_results[i][0] > 0 {
            metrics.push((
                format!("{prefix}/goal/{name}/success_given_win"),
                stats.goal_successes[i] as f32 / stats.goal_results[i][0] as f32,
            ));
        }
        if attempts > 0.0 {
            metrics.push((
                format!("{prefix}/goal/{name}/win_rate"),
                stats.goal_results[i][0] as f32 / attempts,
            ));
            metrics.push((
                format!("{prefix}/goal/{name}/success_rate"),
                stats.goal_successes[i] as f32 / attempts,
            ));
        }
    }
    let goal_results = stats.goal_result_totals();
    for (result, count) in ["wins", "losses", "draws", "unfinished"]
        .into_iter()
        .zip(goal_results)
    {
        metrics.push((format!("{prefix}/goal_owner/{result}"), count as f32));
    }
    let goal_games = goal_results.iter().sum::<usize>();
    if goal_games > 0 {
        metrics.push((
            format!("{prefix}/goal_owner/win_rate"),
            goal_results[0] as f32 / goal_games as f32,
        ));
    }
    if goal_results[0] > 0 {
        metrics.push((
            format!("{prefix}/goal_owner/success_given_win"),
            stats.goal_successes.iter().sum::<usize>() as f32 / goal_results[0] as f32,
        ));
    }
    for (result, count) in ["red_wins", "black_wins", "draws", "unfinished"]
        .into_iter()
        .zip(stats.ordinary_results)
    {
        metrics.push((format!("{prefix}/ordinary/{result}"), count as f32));
    }
    for (i, side) in ["red", "black"].into_iter().enumerate() {
        metrics.push((
            format!("{prefix}/goal_side/{side}/attempts"),
            stats.goal_side_attempts[i] as f32,
        ));
    }
    if stats.games > 0 {
        metrics.extend([
            (
                format!("{prefix}/supervision_coverage"),
                stats.labeled_games as f32 / stats.games as f32,
            ),
            (
                format!("{prefix}/unknown_rate"),
                stats.unknown_games as f32 / stats.games as f32,
            ),
        ]);
    }
    for (i, pattern) in chineseai::xiangqi::MatingPattern::ALL
        .into_iter()
        .enumerate()
    {
        let name = pattern.name();
        let count = stats.pattern_counts[i] as f32;
        metrics.push((format!("{prefix}/{name}/count"), count));
        if stats.games > 0 {
            metrics.push((
                format!("{prefix}/{name}/all_games_rate"),
                count / stats.games as f32,
            ));
        }
        if stats.checkmates > 0 {
            metrics.push((
                format!("{prefix}/{name}/checkmates_rate"),
                count / stats.checkmates as f32,
            ));
        }
    }
    metrics
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_totals_use_combined_counts_instead_of_averaging_batch_rates() {
        let mut total = chineseai::az::AzMatingStats::default();
        for (games, checkmates, unknown, count) in [(4, 1, 2, 1), (6, 2, 3, 2)] {
            let mut batch = chineseai::az::AzMatingStats {
                games,
                checkmates,
                unknown_games: unknown,
                labeled_games: games - unknown,
                ..Default::default()
            };
            batch.pattern_counts[0] = count;
            let horse = chineseai::xiangqi::MatingPattern::ALL
                .iter()
                .position(|p| *p == chineseai::xiangqi::MatingPattern::HorseCannon)
                .unwrap();
            batch.pattern_counts[horse] = count;
            total.add_assign(&batch);
        }
        let metrics = mating_count_metrics(&total, "mating_total")
            .into_iter()
            .collect::<std::collections::HashMap<_, _>>();
        assert_eq!(metrics["mating_total/games"], 10.0);
        assert_eq!(metrics["mating_total/labeled_games"], 5.0);
        assert_eq!(metrics["mating_total/unknown_games"], 5.0);
        assert_eq!(metrics["mating_total/重炮/count"], 3.0);
        assert_eq!(metrics["mating_total/重炮/all_games_rate"], 0.3);
        assert_eq!(metrics["mating_total/重炮/checkmates_rate"], 1.0);
        assert_eq!(metrics["mating_total/马后炮/checkmates_rate"], 1.0);
    }

    #[test]
    fn training_report_preserves_mating_counts_and_prediction_moments() {
        let mut pending = PendingTrainingData::default();
        pending.selfplay.mating.games = 3;
        pending.selfplay.mating.checkmates = 1;
        pending.selfplay.mating.unknown_games = 2;
        pending.selfplay.mating.pattern_counts[0] = 1;
        let mut stats = chineseai::az::AzTrainStats {
            mating_loss: 0.25,
            mating_samples: 2,
            ..Default::default()
        };
        stats.mating_target_sum[0] = 1.0;
        stats.mating_pred_sum[0] = 1.2;
        stats.mating_brier_sum[0] = 0.08;
        stats.goal_loss = 0.3;
        stats.goal_samples = 2;
        stats.goal_target_sum = 1.0;
        stats.goal_pred_sum = 1.2;
        stats.goal_brier_sum = 0.08;
        let report = build_async_training_report(
            pending,
            3,
            stats,
            0.001,
            4,
            0.1,
            4,
            100,
            Default::default(),
            0.0,
        );
        assert_eq!(report.mating.games, 3);
        assert_eq!(report.mating.checkmates, 1);
        assert_eq!(report.mating.unknown_games, 2);
        assert_eq!(report.mating.pattern_counts[0], 1);
        assert_eq!(report.mating_loss, 0.25);
        assert_eq!(report.mating_samples, 2);
        assert_eq!(report.mating_target_sum, stats.mating_target_sum);
        assert_eq!(report.mating_pred_sum, stats.mating_pred_sum);
        assert_eq!(report.mating_brier_sum, stats.mating_brier_sum);
        assert_eq!(report.goal_loss, stats.goal_loss);
        assert_eq!(report.goal_samples, stats.goal_samples);
        assert_eq!(report.goal_target_sum, stats.goal_target_sum);
        assert_eq!(report.goal_pred_sum, stats.goal_pred_sum);
        assert_eq!(report.goal_brier_sum, stats.goal_brier_sum);
    }

    #[test]
    fn mating_metrics_use_game_and_valid_sample_denominators() {
        let mut report = AzLoopReport::default();
        report.mating.games = 4;
        report.mating.checkmates = 2;
        report.mating.labeled_games = 3;
        report.mating.unknown_games = 1;
        report.mating.pattern_counts[0] = 2;
        let horse = chineseai::xiangqi::MatingPattern::ALL
            .iter()
            .position(|p| *p == chineseai::xiangqi::MatingPattern::HorseCannon)
            .unwrap();
        report.mating.pattern_counts[horse] = 2;
        report.mating_samples = 5;
        report.mating_target_sum[0] = 2.0;
        report.mating_pred_sum[0] = 3.0;
        report.mating_brier_sum[0] = 0.5;
        let metrics = mating_metrics(&report)
            .into_iter()
            .collect::<std::collections::HashMap<_, _>>();
        assert_eq!(metrics["mating/重炮/all_games_rate"], 0.5);
        assert_eq!(metrics["mating/重炮/checkmates_rate"], 1.0);
        assert_eq!(metrics["mating/马后炮/checkmates_rate"], 1.0);
        assert_eq!(metrics["mating/supervision_coverage"], 0.75);
        assert_eq!(metrics["train/mating/重炮/target_rate"], 0.4);
        assert_eq!(metrics["train/mating/重炮/pred_mean"], 0.6);
        assert_eq!(metrics["train/mating/重炮/brier"], 0.1);
    }

    #[test]
    fn unknown_batches_do_not_report_zero_loss_as_measured_supervision() {
        let metrics = mating_metrics(&AzLoopReport::default());
        assert!(metrics.iter().all(|(_, value)| value.is_finite()));
        assert!(
            !metrics
                .iter()
                .any(|(tag, _)| tag == "train/mating_bce" || tag.ends_with("/brier"))
        );
    }
    #[test]
    fn goal_result_metrics_separate_winning_from_forming_the_requested_mate() {
        let mut stats = chineseai::az::AzMatingStats::default();
        stats.goal_attempts[0] = 10;
        stats.goal_successes[0] = 1;
        stats.goal_results[0] = [2, 6, 1, 1];
        stats.ordinary_results = [1, 1, 0, 1];
        let metrics = mating_count_metrics(&stats, "mating")
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();
        assert_eq!(metrics["mating/goal/重炮/win_rate"], 0.2);
        assert_eq!(metrics["mating/goal/重炮/success_rate"], 0.1);
        assert_eq!(metrics["mating/goal/重炮/success_given_win"], 0.5);
        assert_eq!(metrics["mating/goal_owner/unfinished"], 1.0);
        assert_eq!(metrics["mating/ordinary/red_wins"], 1.0);
    }
}
