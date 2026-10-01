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
