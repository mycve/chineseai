use std::sync::Arc;

use crate::xiangqi::{BOARD_FILES, Color, Move, Position};

use super::*;

#[derive(Clone, Debug)]
pub struct AzLoopConfig {
    pub games: usize,
    pub max_plies: usize,
    pub rule60_max_ply: Option<u16>,
    pub simulations: usize,
    pub seed: u64,
    pub workers: usize,
    pub generation_update: u32,
    pub temperature_start: f32,
    pub temperature_cutoff_plies: usize,
    pub temperature_visit_offset: f32,
    pub temperature_endgame: f32,
    pub temperature_decay_delay_plies: usize,
    pub temperature_decay_plies: usize,
    pub cpuct: f32,
    pub cpuct_at_root: f32,
    pub cpuct_base: f32,
    pub cpuct_factor: f32,
    pub cpuct_base_at_root: f32,
    pub cpuct_factor_at_root: f32,
    /// 每个根走法的固定 Dirichlet alpha，0 关闭噪声。
    pub root_dirichlet_alpha: f32,
    pub root_exploration_fraction: f32,
    pub fpu_value: f32,
    pub fpu_value_at_root: f32,
    pub fpu_absolute_at_root: bool,
    pub minimum_kldgain_per_node: f32,
    pub draw_score: f32,
    pub policy_softmax_temp: f32,
    pub opening_positions: Arc<[AzStartSnapshot]>,
    pub mirror_probability: f32,
    pub record_fens: bool,
}

#[derive(Clone, Debug, Default)]
pub struct AzLoopReport {
    pub training_steps: usize,
    pub training_chunks: usize,
    pub test_chunks: usize,
    pub holdout_checks: Vec<AzHoldoutReport>,
    pub cycle_complete: bool,
    pub games: usize,
    pub samples: usize,
    pub avg_search_simulations: f32,
    pub red_wins: usize,
    pub black_wins: usize,
    pub draws: usize,
    pub avg_plies: f32,
    pub selfplay_start_source_rate: [f32; AzStartSource::COUNT],
    pub selfplay_start_phase_ply: [f32; AzStartSource::COUNT],
    pub selfplay_start_age: [f32; AzStartSource::COUNT],
    pub selfplay_start_age_max: [u32; AzStartSource::COUNT],
    pub selfplay_start_temperature: [f32; AzStartSource::COUNT],
    pub loss: f32,
    pub learning_rate: f32,
    pub value_loss: f32,
    pub value_mse: f32,
    pub value_pred_mean: f32,
    pub value_target_mean: f32,
    pub value_pred_rms: f32,
    pub value_target_rms: f32,
    pub value_corr: f32,
    pub value_calibration: f32,
    pub phase_value: [AzPhaseValueReport; 3],
    pub source_phase_value: [AzPhaseValueReport; 9],
    pub policy_ce: f32,
    pub policy_target_entropy: f32,
    pub policy_kl: f32,
    pub root_visit_entropy: f32,
    pub entropy_opening: f32,
    pub entropy_mid: f32,
    pub raw_prior_top1: f32,
    pub raw_prior_top2: f32,
    pub policy_top1: f32,
    pub policy_top2: f32,
    pub root_q_gap: f32,
    pub root_q_top1_abs: f32,
    pub visited_actions: f32,
    pub opening_raw_prior_top1: f32,
    pub opening_raw_prior_top2: f32,
    pub opening_policy_top1: f32,
    pub opening_policy_top2: f32,
    pub opening_q_gap: f32,
    pub opening_q_top1_abs: f32,
    pub opening_visited_actions: f32,
    pub sampled_best_rate: f32,
    pub avg_best_played_q_gap: f32,
    pub avg_played_top_visit_ratio: f32,
    pub avg_best_q: f32,
    pub avg_played_q: f32,
    pub train_seconds: f32,
    pub total_seconds: f32,
    pub games_per_second: f32,
    pub samples_per_second: f32,
    pub train_samples_per_second: f32,
    pub train_samples: usize,
    pub pool_samples: usize,
    pub pool_capacity: usize,
    pub replay_chunks: usize,
    pub replay_oldest_update: u32,
    pub replay_newest_update: u32,
    pub replay_avg_update: f32,
    pub replay_window_games: u32,
    pub replay_recent_window_fraction: f32,
    pub terminal_no_legal_moves: usize,
    pub terminal_checkmate: usize,
    pub terminal_stalemate: usize,
    pub terminal_rule_blocked: usize,
    pub terminal_search_no_move: usize,
    pub terminal_red_general_missing: usize,
    pub terminal_black_general_missing: usize,
    pub terminal_rule_draw: usize,
    pub terminal_rule_draw_natural_limit: usize,
    pub terminal_rule_draw_insufficient_material: usize,
    pub terminal_rule_draw_repetition: usize,
    pub terminal_rule_draw_mutual_long_check: usize,
    pub terminal_rule_draw_mutual_long_chase: usize,
    pub terminal_rule_win_red: usize,
    pub terminal_rule_win_black: usize,
    pub terminal_max_plies: usize,
    pub terminal_search_proven: [usize; 3],
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AzPhaseValueReport {
    pub samples: usize,
    pub rmse: f32,
    pub corr: f32,
    pub calibration: f32,
}

#[derive(Clone, Debug)]
pub struct AzTrainingSample {
    pub features: Vec<usize>,
    pub rule_context: [f32; RULE_CONTEXT_SIZE],
    pub move_indices: Vec<usize>,
    pub repetition_flags: Vec<u8>,
    pub policy: Vec<f32>,
    pub value_wdl: [f32; WDL_HEAD_SIZE],
    pub root_search_wdl: [f32; WDL_HEAD_SIZE],
    pub value: f32,
    pub side_sign: f32,
    pub policy_weight: f32,
    pub value_weight: f32,
    pub search_simulations: u32,
    pub meta: AzSampleMeta,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AzPolicyGroupStats {
    pub quiet_samples: usize,
    pub tactical_samples: usize,
    pub quiet_ce: f32,
    pub tactical_ce: f32,
    pub quiet_target_mass: f32,
    pub quiet_predicted_mass: f32,
    pub quiet_top1_rank: f32,
    pub repetition_samples: usize,
    pub no_repetition_samples: usize,
    pub repetition_kl: f32,
    pub no_repetition_kl: f32,
    pub repetition_target_mass: f32,
    pub repetition_predicted_mass: f32,
    pub repetition_target_top1_rate: f32,
    pub repetition_predicted_top1_rate: f32,
}

pub fn evaluate_policy_groups(model: &AzNnue, samples: &[AzTrainingSample]) -> AzPolicyGroupStats {
    let mut stats = AzPolicyGroupStats::default();
    let mut quiet_target_mass_sum = 0.0;
    let mut quiet_predicted_mass_sum = 0.0;
    let mut quiet_rank_sum = 0.0;
    let mut scratch = AzEvalScratch::new(model.arch);
    for sample in samples {
        if sample.move_indices.is_empty() || sample.policy.len() != sample.move_indices.len() {
            continue;
        }
        let pieces = sample
            .features
            .iter()
            .filter_map(|&feature| decode_current_piece_square_feature(feature))
            .map(|piece| (piece.piece_index, piece.rank * BOARD_FILES + piece.file))
            .collect::<Vec<_>>();
        let position = Position::from_canonical_piece_squares(&pieces);
        if !position.has_general(Color::Red) || !position.has_general(Color::Black) {
            continue;
        }
        let moves = sample
            .move_indices
            .iter()
            .filter_map(|&index| dense_move_squares(index))
            .map(|(from, to)| Move::new(from, to))
            .collect::<Vec<_>>();
        if moves.len() != sample.policy.len() {
            continue;
        }
        model.evaluate_with_scratch_output_with_repetition(
            &position,
            &moves,
            &sample.repetition_flags,
            &sample.rule_context,
            &mut scratch,
        );
        let max_logit = scratch
            .logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut predicted = scratch
            .logits
            .iter()
            .map(|&logit| (logit - max_logit).exp())
            .collect::<Vec<_>>();
        let total = predicted.iter().sum::<f32>().max(1.0e-12);
        for value in &mut predicted {
            *value /= total;
        }
        let quiet = moves
            .iter()
            .zip(&scratch.policy_gives_check)
            .map(|(&mv, &check)| position.piece_at(mv.to as usize).is_none() && check == 0.0)
            .collect::<Vec<_>>();
        let top1 = sample
            .policy
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index)
            .unwrap_or(0);
        let ce = sample
            .policy
            .iter()
            .zip(&predicted)
            .map(|(&target, &probability)| -target.max(0.0) * probability.max(1.0e-12).ln())
            .sum::<f32>();
        if sample.repetition_flags.len() == moves.len() {
            let entropy = sample
                .policy
                .iter()
                .filter(|&&target| target > 0.0)
                .map(|&target| -target * target.ln())
                .sum::<f32>();
            if sample.repetition_flags.iter().any(|&flag| flag != 0) {
                stats.repetition_samples += 1;
                stats.repetition_kl += ce - entropy;
                stats.repetition_target_mass += sample
                    .policy
                    .iter()
                    .zip(&sample.repetition_flags)
                    .filter_map(|(&target, &flag)| (flag != 0).then_some(target))
                    .sum::<f32>();
                stats.repetition_predicted_mass += predicted
                    .iter()
                    .zip(&sample.repetition_flags)
                    .filter_map(|(&probability, &flag)| (flag != 0).then_some(probability))
                    .sum::<f32>();
                stats.repetition_target_top1_rate += f32::from(sample.repetition_flags[top1] != 0);
                let predicted_top1 = predicted
                    .iter()
                    .enumerate()
                    .max_by(|(_, left), (_, right)| left.total_cmp(right))
                    .map(|(index, _)| index)
                    .unwrap_or(0);
                stats.repetition_predicted_top1_rate +=
                    f32::from(sample.repetition_flags[predicted_top1] != 0);
            } else {
                stats.no_repetition_samples += 1;
                stats.no_repetition_kl += ce - entropy;
            }
        }
        if quiet[top1] {
            stats.quiet_samples += 1;
            stats.quiet_ce += ce;
            quiet_rank_sum += 1.0
                + predicted
                    .iter()
                    .filter(|&&probability| probability > predicted[top1])
                    .count() as f32;
        } else {
            stats.tactical_samples += 1;
            stats.tactical_ce += ce;
        }
        quiet_target_mass_sum += sample
            .policy
            .iter()
            .zip(&quiet)
            .filter_map(|(&probability, &is_quiet)| is_quiet.then_some(probability))
            .sum::<f32>();
        quiet_predicted_mass_sum += predicted
            .iter()
            .zip(&quiet)
            .filter_map(|(&probability, &is_quiet)| is_quiet.then_some(probability))
            .sum::<f32>();
    }
    stats.quiet_ce /= stats.quiet_samples.max(1) as f32;
    stats.tactical_ce /= stats.tactical_samples.max(1) as f32;
    let total_samples = (stats.quiet_samples + stats.tactical_samples).max(1) as f32;
    stats.quiet_target_mass = quiet_target_mass_sum / total_samples;
    stats.quiet_predicted_mass = quiet_predicted_mass_sum / total_samples;
    stats.quiet_top1_rank = quiet_rank_sum / stats.quiet_samples.max(1) as f32;
    let repetition_count = stats.repetition_samples.max(1) as f32;
    stats.repetition_kl /= repetition_count;
    stats.no_repetition_kl /= stats.no_repetition_samples.max(1) as f32;
    stats.repetition_target_mass /= repetition_count;
    stats.repetition_predicted_mass /= repetition_count;
    stats.repetition_target_top1_rate /= repetition_count;
    stats.repetition_predicted_top1_rate /= repetition_count;
    stats
}

/// Compress exact rule history into bounded continuous inputs. Values are
/// perspective-relative to the side to move, so canonical board mirroring
/// remains valid.
pub fn rule_context_features(
    position: &Position,
    history: &[crate::xiangqi::RuleHistoryEntry],
) -> [f32; RULE_CONTEXT_SIZE] {
    let current = history.last();
    let (prior_matches, cycle_start) = current.map_or((0usize, history.len()), |entry| {
        let mut matches = 0usize;
        let mut last_match = None;
        for (index, old) in history[..history.len().saturating_sub(1)]
            .iter()
            .enumerate()
        {
            if old.hash == entry.hash && old.side_to_move == entry.side_to_move {
                matches += 1;
                last_match = Some(index);
            }
        }
        (matches, last_match.map_or(history.len(), |index| index + 1))
    });
    let cycle = &history[cycle_start.min(history.len())..];
    // 逐着记录不再保存捉子掩码（只扫描被移动棋子会漏掉被发现的攻击），
    // 这里按完整局面回滚重算，语义与旧版一致。
    let exact_cycle = position.recompute_cycle_chases(cycle);
    let cycle = exact_cycle.as_slice();
    let side = position.side_to_move();
    let cycle_count = |color: Color, predicate: fn(&crate::xiangqi::RuleHistoryEntry) -> bool| {
        cycle
            .iter()
            .filter(|entry| entry.mover == Some(color) && predicate(entry))
            .count()
    };
    let is_check = |entry: &crate::xiangqi::RuleHistoryEntry| entry.gives_check;
    let is_chase = |entry: &crate::xiangqi::RuleHistoryEntry| entry.chased_mask != 0;
    [
        position.rule60_max_ply().map_or(0.0, |max_ply| {
            position.rule60_count_with_history(history) as f32 / max_ply as f32
        }),
        (prior_matches as f32 / 3.0).min(1.0),
        (cycle.len() as f32 / 32.0).min(1.0),
        (cycle_count(side, is_check) as f32 / 4.0).min(1.0),
        (cycle_count(side.opposite(), is_check) as f32 / 4.0).min(1.0),
        (cycle_count(side, is_chase) as f32 / 4.0).min(1.0),
        (cycle_count(side.opposite(), is_chase) as f32 / 4.0).min(1.0),
    ]
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum AzStartSource {
    #[default]
    Startpos = 0,
    OpeningBook = 1,
    Midgame = 2,
}

impl AzStartSource {
    pub const COUNT: usize = 3;

    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Startpos),
            1 => Some(Self::OpeningBook),
            2 => Some(Self::Midgame),
            _ => None,
        }
    }

    pub const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AzSampleMeta {
    pub generation_update: u32,
    pub game_id: u64,
    pub ply: u16,
    pub root_q: f32,
    pub best_q: f32,
    pub played_q: f32,
    pub best_visits: u32,
    pub played_visits: u32,
    pub best_index: u16,
    pub played_index: u16,
    pub start_source: AzStartSource,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AzTrainStats {
    /// Mean optimized objective after a completed training call, including all weights.
    pub loss: f32,
    pub value_loss: f32,
    pub policy_ce: f32,
    pub value_pred_sum: f32,
    pub value_pred_sq_sum: f32,
    pub value_target_sum: f32,
    pub value_target_sq_sum: f32,
    pub value_pred_target_sum: f32,
    pub value_error_sq_sum: f32,
    pub samples: usize,
    pub phase_value: [AzValueMomentStats; 3],
    pub source_phase_value: [AzValueMomentStats; 9],
}

pub const PX0_CYCLE_STEPS: usize = 140_000;
pub const PX0_TEST_STEPS: usize = 2_000;

#[derive(Clone, Copy, Debug)]
pub struct AzHoldoutReport {
    pub step: usize,
    pub samples: usize,
    pub value_samples: usize,
    pub loss: f32,
    pub value_loss: f32,
    pub policy_kl: f32,
    pub value_rmse: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AzValueMomentStats {
    pub pred_sum: f32,
    pub pred_sq_sum: f32,
    pub target_sum: f32,
    pub target_sq_sum: f32,
    pub pred_target_sum: f32,
    pub error_sq_sum: f32,
    pub samples: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct AzTrainLossWeights {
    pub value: f32,
    pub policy: f32,
}

impl Default for AzTrainLossWeights {
    fn default() -> Self {
        Self {
            value: 1.0,
            policy: 1.0,
        }
    }
}

impl AzTrainStats {
    #[cfg_attr(not(feature = "gpu-train"), allow(dead_code))]
    pub(crate) fn add_assign(&mut self, other: &Self) {
        self.loss += other.loss;
        self.value_loss += other.value_loss;
        self.policy_ce += other.policy_ce;
        self.value_pred_sum += other.value_pred_sum;
        self.value_pred_sq_sum += other.value_pred_sq_sum;
        self.value_target_sum += other.value_target_sum;
        self.value_target_sq_sum += other.value_target_sq_sum;
        self.value_pred_target_sum += other.value_pred_target_sum;
        self.value_error_sq_sum += other.value_error_sq_sum;
        self.samples += other.samples;
        for (left, right) in self.phase_value.iter_mut().zip(other.phase_value) {
            left.pred_sum += right.pred_sum;
            left.pred_sq_sum += right.pred_sq_sum;
            left.target_sum += right.target_sum;
            left.target_sq_sum += right.target_sq_sum;
            left.pred_target_sum += right.pred_target_sum;
            left.error_sq_sum += right.error_sq_sum;
            left.samples += right.samples;
        }
        for (left, right) in self
            .source_phase_value
            .iter_mut()
            .zip(other.source_phase_value)
        {
            left.pred_sum += right.pred_sum;
            left.pred_sq_sum += right.pred_sq_sum;
            left.target_sum += right.target_sum;
            left.target_sq_sum += right.target_sq_sum;
            left.pred_target_sum += right.pred_target_sum;
            left.error_sq_sum += right.error_sq_sum;
            left.samples += right.samples;
        }
    }
}

pub struct SplitMix64 {
    pub(crate) state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = splitmix64(self.state);
        self.state
    }

    pub fn unit_f32(&mut self) -> f32 {
        let value = self.next_u64();
        (((value >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))) as f32
    }

    pub(crate) fn weight(&mut self, scale: f32) -> f32 {
        (self.unit_f32() * 2.0 - 1.0) * scale
    }
}

pub(crate) fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut mixed = value;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^ (mixed >> 31)
}

pub fn policy_target_entropy(samples: &[AzTrainingSample]) -> f32 {
    let total = samples
        .iter()
        .map(|sample| {
            let targets = || {
                sample
                    .move_indices
                    .iter()
                    .zip(&sample.policy)
                    .filter_map(|(&index, &p)| (index < DENSE_MOVE_SPACE).then_some(p.max(0.0)))
            };
            let sum = targets().sum::<f32>();
            if sum.is_finite() && sum > 1.0e-12 {
                targets()
                    .map(|p| p / sum)
                    .filter(|&p| p > 0.0)
                    .map(|p| -p * p.ln())
                    .sum::<f32>()
            } else {
                (targets().count().max(1) as f32).ln()
            }
        })
        .sum::<f32>();
    total / samples.len().max(1) as f32
}
