use std::io;
use std::path::Path;

use candle_core::{DType, Device, Shape, Var};
use candle_nn::VarMap;

mod alphazero;
#[cfg(any(
    test,
    all(feature = "gpu-train", not(target_os = "macos")),
    all(target_os = "linux", not(target_env = "musl"))
))]
mod candle_model;
mod dataloader;
mod play;
mod replay;
mod train;
mod train_gpu;
#[cfg(any(
    all(feature = "gpu-train", not(target_os = "macos")),
    all(target_os = "linux", not(target_env = "musl"))
))]
#[path = "az/train_gpu_candle.rs"]
mod train_gpu_candle;

use crate::nnue::{
    AZ_NNUE_INPUT_SIZE, AZ_SIDE_INPUT_OFFSET, V2_KING_BUCKETS, mirror_file_move, mirror_file_square,
};
use crate::xiangqi::{
    BOARD_FILES, BOARD_RANKS, BOARD_SIZE, Color, Move, Piece, PieceKind, Position,
};

pub use alphazero::{
    AzCandidate, AzSearchControl, AzSearchLimits, AzSearchResult, alphazero_search,
    alphazero_search_with_rules, alphazero_search_with_rules_controlled,
    alphazero_search_with_rules_controlled_with_progress, cp_from_q,
};
pub use play::{
    AzArenaConfig, AzArenaReport, AzSelfplayData, AzTerminalStats, generate_selfplay_data,
    play_arena_games_from_positions,
};
pub use replay::{AzExperiencePool, AzReplaySampleBatch, AzReplayWindowStats};
pub use train::{
    global_training_step_sample_count, train_samples, train_samples_weighted,
    train_samples_weighted_owned,
};

const SPARSE_MOVE_SPACE: usize = BOARD_SIZE * BOARD_SIZE;
pub const DENSE_MOVE_SPACE: usize = compute_dense_move_count();
pub(super) const POLICY_PAIR_CONTEXT_SIZE: usize = 32;
pub(super) const POLICY_MOVE_EMBED_SIZE: usize = 16;
pub(super) const VALUE_HEAD_SIZE: usize = 96;
pub(super) const WDL_HEAD_SIZE: usize = 3;
#[cfg_attr(not(feature = "gpu-train"), allow(dead_code))]
pub(super) const MOVES_LEFT_AUX_WEIGHT: f32 = 0.05;
#[cfg_attr(not(feature = "gpu-train"), allow(dead_code))]
const RMS_NORM_EPS: f32 = 1.0e-6;
pub(super) const PIECE_SQUARE_INPUT_SIZE: usize = BOARD_SIZE * 14;
pub(super) const STRUCTURAL_PIECE_SIZE: usize = 14;
pub(super) const STRUCTURAL_RANK_SIZE: usize = BOARD_RANKS;
pub(super) const STRUCTURAL_FILE_SIZE: usize = BOARD_FILES;
pub(super) const STRUCTURAL_KING_PIECE_SIZE: usize = 2 * V2_KING_BUCKETS * 14;

pub fn inference_simd_backend() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return "avx2+fma-4acc";
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return "avx2";
        }
    }
    #[cfg(target_arch = "x86")]
    if std::arch::is_x86_feature_detected!("avx2") {
        return "avx2";
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "neon";
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        "scalar"
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct StructuralPieceSquare {
    pub piece_index: usize,
    pub rank: usize,
    pub file: usize,
}

pub(super) fn decode_current_piece_square_feature(feature: usize) -> Option<StructuralPieceSquare> {
    if feature >= PIECE_SQUARE_INPUT_SIZE {
        return None;
    }
    let piece_index = feature / BOARD_SIZE;
    let sq = feature % BOARD_SIZE;
    Some(StructuralPieceSquare {
        piece_index,
        rank: sq / BOARD_FILES,
        file: sq % BOARD_FILES,
    })
}

pub(super) fn canonical_general_buckets_from_features(features: &[usize]) -> (usize, usize) {
    let mut us = 4;
    let mut them = 4;
    for &feature in features {
        if feature >= PIECE_SQUARE_INPUT_SIZE {
            continue;
        }
        let piece_index = feature / BOARD_SIZE;
        let sq = feature % BOARD_SIZE;
        match piece_index {
            0 => us = canonical_general_bucket(piece_index, sq),
            7 => them = canonical_general_bucket(piece_index, sq),
            _ => {}
        }
    }
    (us, them)
}

pub(super) fn structural_king_piece_index(
    perspective: usize,
    king_bucket: usize,
    piece_index: usize,
) -> usize {
    ((perspective * V2_KING_BUCKETS + king_bucket.min(V2_KING_BUCKETS - 1)) * 14) + piece_index
}

fn canonical_general_bucket(piece_index: usize, sq: usize) -> usize {
    let oriented_sq = if piece_index < 7 {
        sq
    } else {
        BOARD_SIZE - 1 - sq
    };
    let file = (oriented_sq % BOARD_FILES).clamp(3, 5) - 3;
    let rank = (oriented_sq / BOARD_FILES).clamp(7, 9) - 7;
    rank * 3 + file
}

fn candle_io_error(err: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err.to_string())
}

fn insert_candle_var(
    varmap: &VarMap,
    name: &str,
    data: &[f32],
    shape: impl Into<Shape>,
) -> io::Result<()> {
    let var = Var::from_slice(data, shape, &Device::Cpu).map_err(candle_io_error)?;
    varmap
        .data()
        .lock()
        .unwrap_or_else(|_| panic!("candle varmap poisoned"))
        .insert(name.to_string(), var);
    Ok(())
}

fn load_candle_f32_tensor(
    tensors: &candle_core::safetensors::MmapedSafetensors,
    name: &str,
) -> io::Result<Vec<f32>> {
    let tensor = tensors.load(name, &Device::Cpu).map_err(candle_io_error)?;
    if tensor.dtype() != DType::F32 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("tensor `{name}` is {:?}, expected F32", tensor.dtype()),
        ));
    }
    tensor
        .flatten_all()
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(candle_io_error)
}

macro_rules! az_weight_tensors {
    ($visit:ident, $h:expr) => {
        $visit!(input_hidden, [AZ_NNUE_INPUT_SIZE, $h / 2]);
        $visit!(input_piece_hidden, [STRUCTURAL_PIECE_SIZE, $h / 2]);
        $visit!(input_rank_hidden, [STRUCTURAL_RANK_SIZE, $h / 2]);
        $visit!(input_file_hidden, [STRUCTURAL_FILE_SIZE, $h / 2]);
        $visit!(
            input_king_piece_hidden,
            [STRUCTURAL_KING_PIECE_SIZE, $h / 2]
        );
        $visit!(hidden_bias, [$h / 2]);
        $visit!(value_head_hidden, [VALUE_HEAD_SIZE, $h / 2]);
        $visit!(value_head_bias, [VALUE_HEAD_SIZE]);
        $visit!(value_head_hidden2, [VALUE_HEAD_SIZE, VALUE_HEAD_SIZE]);
        $visit!(value_head_bias2, [VALUE_HEAD_SIZE]);
        $visit!(value_head_output, [WDL_HEAD_SIZE, VALUE_HEAD_SIZE]);
        $visit!(moves_left_hidden, [VALUE_HEAD_SIZE, $h / 2]);
        $visit!(moves_left_bias_hidden, [VALUE_HEAD_SIZE]);
        $visit!(moves_left_output, [VALUE_HEAD_SIZE]);
        $visit!(moves_left_bias, [1]);
        $visit!(policy_move_bias, [DENSE_MOVE_SPACE]);
        $visit!(policy_from_hidden, [BOARD_SIZE, $h / 2]);
        $visit!(policy_to_hidden, [BOARD_SIZE, $h / 2]);
        $visit!(
            policy_pair_context_hidden,
            [POLICY_PAIR_CONTEXT_SIZE, $h / 2]
        );
        $visit!(policy_pair_context_bias, [POLICY_PAIR_CONTEXT_SIZE]);
        $visit!(
            policy_pair_embedding,
            [DENSE_MOVE_SPACE, POLICY_PAIR_CONTEXT_SIZE]
        );
        $visit!(policy_move_context_hidden, [POLICY_MOVE_EMBED_SIZE, $h / 2]);
        $visit!(
            policy_move_embedding,
            [DENSE_MOVE_SPACE, POLICY_MOVE_EMBED_SIZE]
        );
    };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AzNnueArch {
    pub hidden_size: usize,
}

impl AzNnueArch {
    pub const fn default_const() -> Self {
        Self { hidden_size: 192 }
    }

    pub const fn with_hidden_size(hidden_size: usize) -> Self {
        let mut arch = Self::default_const();
        arch.hidden_size = hidden_size;
        arch
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_size == 0 || self.hidden_size % 2 != 0 {
            return Err(format!("invalid hidden_size {}", self.hidden_size));
        }
        Ok(())
    }
}

impl Default for AzNnueArch {
    fn default() -> Self {
        Self::default_const()
    }
}
pub(super) struct AzEvalScratch {
    // NNUE 热路径复用特征存储，避免每个 MCTS 叶节点分配并排序 Vec。
    features: Vec<usize>,
    hidden: Vec<f32>,
    value_head: Vec<f32>,
    value_head2: Vec<f32>,
    policy_pair_context: Vec<f32>,
    policy_move_context: Vec<f32>,
    policy_from_scores: Vec<f32>,
    policy_to_scores: Vec<f32>,
    logits: Vec<f32>,
    priors: Vec<f32>,
}

impl AzEvalScratch {
    pub(super) fn new(arch: AzNnueArch) -> Self {
        let hidden_size = arch.hidden_size;
        Self {
            features: Vec::with_capacity(48),
            hidden: vec![0.0; hidden_size],
            value_head: vec![0.0; VALUE_HEAD_SIZE],
            value_head2: vec![0.0; VALUE_HEAD_SIZE],
            policy_pair_context: vec![0.0; POLICY_PAIR_CONTEXT_SIZE],
            policy_move_context: vec![0.0; POLICY_MOVE_EMBED_SIZE],
            policy_from_scores: vec![0.0; BOARD_SIZE],
            policy_to_scores: vec![0.0; BOARD_SIZE],
            logits: Vec::with_capacity(192),
            priors: Vec::with_capacity(192),
        }
    }
}

/// 搜索节点使用的单份固定视角累加器。
/// 前半保存原盘分支，后半保存共享参数的左右镜像分支。
#[derive(Clone, Debug)]
pub(super) struct AzEvalAccumulator {
    hidden_sum: Vec<f32>,
}

impl AzEvalAccumulator {
    pub(super) fn new(model: &AzNnue, position: &Position) -> Self {
        let mut accumulator = Self {
            hidden_sum: vec![0.0; model.hidden_size],
        };
        accumulator.refresh(model, position);
        accumulator
    }

    fn refresh(&mut self, model: &AzNnue, position: &Position) {
        let width = model.branch_width();
        for branch in 0..2 {
            let hidden = &mut self.hidden_sum[branch * width..(branch + 1) * width];
            hidden.copy_from_slice(&model.hidden_bias);
            let mirrored = branch == 1;
            let buckets = fixed_buckets(position, mirrored);
            for sq in 0..BOARD_SIZE {
                if let Some(piece) = position.piece_at(sq) {
                    add_fixed_piece_contribution(model, hidden, mirrored, buckets, sq, piece, 1.0);
                }
            }
            add_scaled_feature_row(
                hidden,
                &model.input_hidden,
                width,
                AZ_SIDE_INPUT_OFFSET + usize::from(position.side_to_move() == Color::Black),
                1.0,
            );
        }
    }

    pub(super) fn apply_transition_to_hidden(
        model: &AzNnue,
        before: &Position,
        after: &Position,
        mv: Move,
        moved: Piece,
        captured: Option<Piece>,
        hidden_sum: &mut [f32],
    ) {
        debug_assert_eq!(hidden_sum.len(), model.hidden_size);
        let width = model.branch_width();
        for branch in 0..2 {
            let mirrored = branch == 1;
            let before_buckets = fixed_buckets(before, mirrored);
            let after_buckets = fixed_buckets(after, mirrored);
            if before_buckets != after_buckets {
                // 将帅移动会改变所有棋子的王桶结构项，少见且必须完整刷新。
                let refreshed = Self::new(model, after);
                hidden_sum.copy_from_slice(&refreshed.hidden_sum);
                return;
            }
            let hidden = &mut hidden_sum[branch * width..(branch + 1) * width];
            add_fixed_piece_contribution(
                model,
                hidden,
                mirrored,
                before_buckets,
                mv.from as usize,
                moved,
                -1.0,
            );
            if let Some(captured) = captured {
                add_fixed_piece_contribution(
                    model,
                    hidden,
                    mirrored,
                    before_buckets,
                    mv.to as usize,
                    captured,
                    -1.0,
                );
            }
            add_fixed_piece_contribution(
                model,
                hidden,
                mirrored,
                after_buckets,
                mv.to as usize,
                moved,
                1.0,
            );
            add_scaled_feature_row(
                hidden,
                &model.input_hidden,
                width,
                AZ_SIDE_INPUT_OFFSET + usize::from(before.side_to_move() == Color::Black),
                -1.0,
            );
            add_scaled_feature_row(
                hidden,
                &model.input_hidden,
                width,
                AZ_SIDE_INPUT_OFFSET + usize::from(after.side_to_move() == Color::Black),
                1.0,
            );
        }
    }

    pub(super) fn into_hidden_sum(self) -> Vec<f32> {
        self.hidden_sum
    }
}

fn fixed_buckets(position: &Position, mirrored: bool) -> (usize, usize) {
    let transform = |sq| if mirrored { mirror_file_square(sq) } else { sq };
    let red = position
        .general_square(Color::Red)
        .map(|sq| canonical_general_bucket(0, transform(sq)))
        .unwrap_or(4);
    let black = position
        .general_square(Color::Black)
        .map(|sq| canonical_general_bucket(7, transform(sq)))
        .unwrap_or(4);
    (red, black)
}

fn add_fixed_piece_contribution(
    model: &AzNnue,
    hidden: &mut [f32],
    mirrored: bool,
    buckets: (usize, usize),
    sq: usize,
    piece: Piece,
    scale: f32,
) {
    let piece_index = color_index(piece.color) * 7 + piece_kind_index(piece.kind);
    let square = if mirrored { mirror_file_square(sq) } else { sq };
    let feature = piece_index * BOARD_SIZE + square;
    let rank = square / BOARD_FILES;
    let file = square % BOARD_FILES;
    let width = model.branch_width();
    add_scaled_feature_row(hidden, &model.input_hidden, width, feature, scale);
    add_scaled_feature_row(hidden, &model.input_piece_hidden, width, piece_index, scale);
    add_scaled_feature_row(hidden, &model.input_rank_hidden, width, rank, scale);
    add_scaled_feature_row(hidden, &model.input_file_hidden, width, file, scale);
    add_scaled_feature_row(
        hidden,
        &model.input_king_piece_hidden,
        width,
        structural_king_piece_index(0, buckets.0, piece_index),
        scale,
    );
    add_scaled_feature_row(
        hidden,
        &model.input_king_piece_hidden,
        width,
        structural_king_piece_index(1, buckets.1, piece_index),
        scale,
    );
}

#[inline(always)]
const fn color_index(color: Color) -> usize {
    match color {
        Color::Red => 0,
        Color::Black => 1,
    }
}

#[inline(always)]
const fn piece_kind_index(kind: PieceKind) -> usize {
    match kind {
        PieceKind::General => 0,
        PieceKind::Advisor => 1,
        PieceKind::Elephant => 2,
        PieceKind::Horse => 3,
        PieceKind::Rook => 4,
        PieceKind::Cannon => 5,
        PieceKind::Soldier => 6,
    }
}

#[derive(Debug)]
pub struct AzNnue {
    pub hidden_size: usize,
    pub arch: AzNnueArch,
    pub input_hidden: Vec<f32>,
    pub input_piece_hidden: Vec<f32>,
    pub input_rank_hidden: Vec<f32>,
    pub input_file_hidden: Vec<f32>,
    pub input_king_piece_hidden: Vec<f32>,
    pub hidden_bias: Vec<f32>,
    pub value_head_hidden: Vec<f32>,
    pub value_head_bias: Vec<f32>,
    pub value_head_hidden2: Vec<f32>,
    pub value_head_bias2: Vec<f32>,
    pub value_head_output: Vec<f32>,
    pub moves_left_hidden: Vec<f32>,
    pub moves_left_bias_hidden: Vec<f32>,
    pub moves_left_output: Vec<f32>,
    pub moves_left_bias: Vec<f32>,
    pub policy_move_bias: Vec<f32>,
    pub policy_from_hidden: Vec<f32>,
    pub policy_to_hidden: Vec<f32>,
    pub policy_pair_context_hidden: Vec<f32>,
    pub policy_pair_context_bias: Vec<f32>,
    pub policy_pair_embedding: Vec<f32>,
    pub policy_move_context_hidden: Vec<f32>,
    pub policy_move_embedding: Vec<f32>,
    #[cfg_attr(not(feature = "gpu-train"), allow(dead_code))]
    gpu_trainer: Option<Box<train_gpu::GpuTrainer>>,
}

impl Clone for AzNnue {
    fn clone(&self) -> Self {
        Self {
            hidden_size: self.hidden_size,
            arch: self.arch,
            input_hidden: self.input_hidden.clone(),
            input_piece_hidden: self.input_piece_hidden.clone(),
            input_rank_hidden: self.input_rank_hidden.clone(),
            input_file_hidden: self.input_file_hidden.clone(),
            input_king_piece_hidden: self.input_king_piece_hidden.clone(),
            hidden_bias: self.hidden_bias.clone(),
            value_head_hidden: self.value_head_hidden.clone(),
            value_head_bias: self.value_head_bias.clone(),
            value_head_hidden2: self.value_head_hidden2.clone(),
            value_head_bias2: self.value_head_bias2.clone(),
            value_head_output: self.value_head_output.clone(),
            moves_left_hidden: self.moves_left_hidden.clone(),
            moves_left_bias_hidden: self.moves_left_bias_hidden.clone(),
            moves_left_output: self.moves_left_output.clone(),
            moves_left_bias: self.moves_left_bias.clone(),
            policy_move_bias: self.policy_move_bias.clone(),
            policy_from_hidden: self.policy_from_hidden.clone(),
            policy_to_hidden: self.policy_to_hidden.clone(),
            policy_pair_context_hidden: self.policy_pair_context_hidden.clone(),
            policy_pair_context_bias: self.policy_pair_context_bias.clone(),
            policy_pair_embedding: self.policy_pair_embedding.clone(),
            policy_move_context_hidden: self.policy_move_context_hidden.clone(),
            policy_move_embedding: self.policy_move_embedding.clone(),
            gpu_trainer: None,
        }
    }
}

impl AzNnue {
    #[inline]
    fn branch_width(&self) -> usize {
        self.hidden_size / 2
    }

    pub fn update_capped_swa(&mut self, current: &Self, count: usize, max_models: usize) -> usize {
        assert_eq!(self.arch, current.arch, "SWA model architecture mismatch");
        let max_models = max_models.max(1);
        let count = count.clamp(1, max_models);
        let old_weight = count as f32 / (count + 1) as f32;
        let new_weight = 1.0 / (count + 1) as f32;
        macro_rules! blend_weight {
            ($field:ident, $shape:expr) => {
                debug_assert_eq!(self.$field.len(), current.$field.len());
                for (average, &value) in self.$field.iter_mut().zip(&current.$field) {
                    *average = *average * old_weight + value * new_weight;
                }
            };
        }
        az_weight_tensors!(blend_weight, self.hidden_size);
        (count + 1).min(max_models)
    }
}

#[derive(Clone, Debug)]
pub struct AzLoopConfig {
    pub games: usize,
    pub max_plies: usize,
    pub simulations: usize,
    pub seed: u64,
    pub workers: usize,
    pub generation_update: u32,
    pub temperature_start: f32,
    pub temperature_endgame: f32,
    pub temperature_decay_delay_plies: usize,
    pub temperature_decay_plies: usize,
    pub temperature_value_cutoff: f32,
    pub temperature_visit_offset: f32,
    pub cpuct: f32,
    pub cpuct_at_root: f32,
    pub cpuct_base: f32,
    pub cpuct_factor: f32,
    pub cpuct_base_at_root: f32,
    pub cpuct_factor_at_root: f32,
    pub root_dirichlet_alpha: f32,
    pub root_exploration_fraction: f32,
    pub fpu_value: f32,
    pub fpu_value_at_root: f32,
    pub draw_score: f32,
    pub moves_left_max_effect: f32,
    pub moves_left_slope: f32,
    pub moves_left_threshold: f32,
    pub moves_left_constant_factor: f32,
    pub moves_left_scaled_factor: f32,
    pub moves_left_quadratic_factor: f32,
    pub policy_softmax_temp: f32,
    pub opening_positions: Vec<Position>,
    pub resign_percentage: f32,
    pub resign_playthrough: f32,
}

#[derive(Clone, Debug, Default)]
pub struct AzLoopReport {
    pub games: usize,
    pub samples: usize,
    pub total_games_generated: usize,
    pub total_samples_generated: usize,
    pub avg_search_simulations: f32,
    pub red_wins: usize,
    pub black_wins: usize,
    pub draws: usize,
    pub avg_plies: f32,
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
    pub policy_ce: f32,
    pub policy_target_entropy: f32,
    pub moves_left_loss: f32,
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
    pub selfplay_seconds: f32,
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
    pub train_fast_sample_rate: f32,
    pub train_policy_weight_mean: f32,
    pub train_value_weight_mean: f32,
    pub train_recent_quota_rate: f32,
    pub train_actual_recent_sample_rate: f32,
    pub train_policy_target_top1: f32,
    pub train_policy_target_top2: f32,
    pub terminal_no_legal_moves: usize,
    pub terminal_red_general_missing: usize,
    pub terminal_black_general_missing: usize,
    pub terminal_rule_draw: usize,
    pub terminal_rule_draw_halfmove120: usize,
    pub terminal_rule_draw_repetition: usize,
    pub terminal_rule_draw_mutual_long_check: usize,
    pub terminal_rule_draw_mutual_long_chase: usize,
    pub terminal_rule_win_red: usize,
    pub terminal_rule_win_black: usize,
    pub terminal_resign_red: usize,
    pub terminal_resign_black: usize,
    pub terminal_max_plies: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AzPhaseValueReport {
    pub samples: usize,
    pub rmse: f32,
    pub corr: f32,
    pub calibration: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AzTrainBenchmark {
    pub loss: f32,
    pub value_loss: f32,
    pub policy_ce: f32,
}

#[derive(Clone, Debug)]
pub struct AzTrainingSample {
    pub features: Vec<usize>,
    pub move_indices: Vec<usize>,
    pub policy: Vec<f32>,
    pub value_wdl: [f32; WDL_HEAD_SIZE],
    pub value: f32,
    pub side_sign: f32,
    pub moves_left: f32,
    pub policy_weight: f32,
    pub value_weight: f32,
    pub search_simulations: u32,
    pub meta: AzSampleMeta,
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
}

#[derive(Clone, Copy, Debug)]
pub(super) struct AzEvalOutput {
    pub value_wdl: [f32; WDL_HEAD_SIZE],
    pub value: f32,
    pub moves_left: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AzTrainStats {
    /// Mean optimized objective after a completed training call, including all weights.
    pub loss: f32,
    pub value_loss: f32,
    pub policy_ce: f32,
    pub moves_left_loss: f32,
    pub value_pred_sum: f32,
    pub value_pred_sq_sum: f32,
    pub value_target_sum: f32,
    pub value_target_sq_sum: f32,
    pub value_pred_target_sum: f32,
    pub value_error_sq_sum: f32,
    pub samples: usize,
    pub phase_value: [AzValueMomentStats; 3],
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
    fn add_assign(&mut self, other: &Self) {
        self.loss += other.loss;
        self.value_loss += other.value_loss;
        self.policy_ce += other.policy_ce;
        self.moves_left_loss += other.moves_left_loss;
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
    }
}

impl AzNnue {
    pub fn random_with_arch(arch: AzNnueArch, seed: u64) -> Self {
        if let Err(err) = arch.validate() {
            panic!("AzNnue::random_with_arch: invalid arch ({err})");
        }
        let hidden_size = arch.hidden_size;
        let branch_width = hidden_size / 2;
        let mut rng = SplitMix64::new(seed);
        let input_hidden: Vec<f32> = (0..AZ_NNUE_INPUT_SIZE * branch_width)
            .map(|_| rng.weight(0.015))
            .collect();
        // Learned structural factors recover row/file/material/king context from
        // piece-square facts without reintroducing those handcrafted feature ids.
        let input_piece_hidden = vec![0.0; STRUCTURAL_PIECE_SIZE * branch_width];
        let input_rank_hidden = vec![0.0; STRUCTURAL_RANK_SIZE * branch_width];
        let input_file_hidden = vec![0.0; STRUCTURAL_FILE_SIZE * branch_width];
        let input_king_piece_hidden = vec![0.0; STRUCTURAL_KING_PIECE_SIZE * branch_width];
        let hidden_bias = vec![0.0; branch_width];
        // 保持相同 seed 下其余权重与旧结构一致，便于公平比较删除注意力前后的速度和棋力。
        // 旧注意力初始化消耗 33 * hidden_size 个随机数，但这些权重不再存储。
        for _ in 0..33 * hidden_size {
            let _ = rng.next_u64();
        }
        // Start value-neutral. A random value head can evaluate startpos as a
        // large red/black advantage before any training, and MCTS amplifies
        // that noise into the first self-play dataset.
        let value_head_hidden = (0..VALUE_HEAD_SIZE * branch_width)
            .map(|_| rng.weight((2.0 / branch_width.max(1) as f32).sqrt() * 0.5))
            .collect();
        let value_head_bias = vec![0.0; VALUE_HEAD_SIZE];
        let value_head_hidden2 = (0..VALUE_HEAD_SIZE * VALUE_HEAD_SIZE)
            .map(|_| rng.weight((2.0 / VALUE_HEAD_SIZE as f32).sqrt() * 0.5))
            .collect();
        let value_head_bias2 = vec![0.0; VALUE_HEAD_SIZE];
        // Keep the value head output-neutral at initialization. This preserves
        // stable first self-play while giving value its own nonlinear capacity.
        let value_head_output = vec![0.0; WDL_HEAD_SIZE * VALUE_HEAD_SIZE];
        let moves_left_hidden = (0..VALUE_HEAD_SIZE * branch_width)
            .map(|_| rng.weight((2.0 / branch_width.max(1) as f32).sqrt() * 0.5))
            .collect();
        let moves_left_bias_hidden = vec![0.0; VALUE_HEAD_SIZE];
        let moves_left_output = vec![0.0; VALUE_HEAD_SIZE];
        let moves_left_bias = vec![0.0; 1];
        let policy_move_bias = vec![0.0; DENSE_MOVE_SPACE];
        let policy_from_hidden = (0..BOARD_SIZE * branch_width)
            .map(|_| rng.weight((2.0 / branch_width.max(1) as f32).sqrt() * 0.25))
            .collect();
        let policy_to_hidden = (0..BOARD_SIZE * branch_width)
            .map(|_| rng.weight((2.0 / branch_width.max(1) as f32).sqrt() * 0.25))
            .collect();
        let policy_pair_context_hidden = (0..POLICY_PAIR_CONTEXT_SIZE * branch_width)
            .map(|_| rng.weight((2.0 / branch_width.max(1) as f32).sqrt() * 0.25))
            .collect();
        let policy_pair_context_bias = vec![0.0; POLICY_PAIR_CONTEXT_SIZE];
        let policy_pair_embedding = (0..DENSE_MOVE_SPACE * POLICY_PAIR_CONTEXT_SIZE)
            .map(|_| rng.weight((2.0 / POLICY_PAIR_CONTEXT_SIZE as f32).sqrt() * 0.1))
            .collect();
        let policy_move_context_hidden = (0..POLICY_MOVE_EMBED_SIZE * branch_width)
            .map(|_| rng.weight((2.0 / branch_width.max(1) as f32).sqrt() * 0.25))
            .collect();
        let policy_move_embedding = vec![0.0; DENSE_MOVE_SPACE * POLICY_MOVE_EMBED_SIZE];
        Self {
            hidden_size,
            arch,
            input_hidden,
            input_piece_hidden,
            input_rank_hidden,
            input_file_hidden,
            input_king_piece_hidden,
            hidden_bias,
            value_head_hidden,
            value_head_bias,
            value_head_hidden2,
            value_head_bias2,
            value_head_output,
            moves_left_hidden,
            moves_left_bias_hidden,
            moves_left_output,
            moves_left_bias,
            policy_move_bias,
            policy_from_hidden,
            policy_to_hidden,
            policy_pair_context_hidden,
            policy_pair_context_bias,
            policy_pair_embedding,
            policy_move_context_hidden,
            policy_move_embedding,
            gpu_trainer: None,
        }
    }

    pub fn random(hidden_size: usize, seed: u64) -> Self {
        Self::random_with_arch(AzNnueArch::with_hidden_size(hidden_size), seed)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let h = self.hidden_size;
        let varmap = VarMap::new();
        macro_rules! save_tensor {
            ($field:ident, [$($dim:expr),+]) => {
                insert_candle_var(&varmap, stringify!($field), &self.$field, ($($dim),+))?;
            };
        }
        az_weight_tensors!(save_tensor, h);
        varmap.save(path).map_err(candle_io_error)
    }

    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let tensors = unsafe {
            candle_core::safetensors::MmapedSafetensors::new(path.as_ref())
                .map_err(candle_io_error)?
        };
        let hidden_bias = load_candle_f32_tensor(&tensors, "hidden_bias")?;
        let hidden_size = hidden_bias.len() * 2;
        let arch = AzNnueArch { hidden_size };
        let model = Self {
            hidden_size,
            arch,
            input_hidden: load_candle_f32_tensor(&tensors, "input_hidden")?,
            input_piece_hidden: load_candle_f32_tensor(&tensors, "input_piece_hidden")?,
            input_rank_hidden: load_candle_f32_tensor(&tensors, "input_rank_hidden")?,
            input_file_hidden: load_candle_f32_tensor(&tensors, "input_file_hidden")?,
            input_king_piece_hidden: load_candle_f32_tensor(&tensors, "input_king_piece_hidden")?,
            hidden_bias,
            value_head_hidden: load_candle_f32_tensor(&tensors, "value_head_hidden")?,
            value_head_bias: load_candle_f32_tensor(&tensors, "value_head_bias")?,
            value_head_hidden2: load_candle_f32_tensor(&tensors, "value_head_hidden2")?,
            value_head_bias2: load_candle_f32_tensor(&tensors, "value_head_bias2")?,
            value_head_output: load_candle_f32_tensor(&tensors, "value_head_output")?,
            moves_left_hidden: load_candle_f32_tensor(&tensors, "moves_left_hidden")?,
            moves_left_bias_hidden: load_candle_f32_tensor(&tensors, "moves_left_bias_hidden")?,
            moves_left_output: load_candle_f32_tensor(&tensors, "moves_left_output")?,
            moves_left_bias: load_candle_f32_tensor(&tensors, "moves_left_bias")?,
            policy_move_bias: load_candle_f32_tensor(&tensors, "policy_move_bias")?,
            policy_from_hidden: load_candle_f32_tensor(&tensors, "policy_from_hidden")?,
            policy_to_hidden: load_candle_f32_tensor(&tensors, "policy_to_hidden")?,
            policy_pair_context_hidden: load_candle_f32_tensor(
                &tensors,
                "policy_pair_context_hidden",
            )?,
            policy_pair_context_bias: load_candle_f32_tensor(&tensors, "policy_pair_context_bias")?,
            policy_pair_embedding: load_candle_f32_tensor(&tensors, "policy_pair_embedding")?,
            policy_move_context_hidden: load_candle_f32_tensor(
                &tensors,
                "policy_move_context_hidden",
            )?,
            policy_move_embedding: load_candle_f32_tensor(&tensors, "policy_move_embedding")?,
            gpu_trainer: None,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn evaluate_value(&self, position: &Position, moves: &[Move]) -> f32 {
        let mut scratch = AzEvalScratch::new(self.arch);
        self.evaluate_with_scratch(position, moves, &mut scratch)
    }

    pub(super) fn evaluate_with_scratch(
        &self,
        position: &Position,
        moves: &[Move],
        scratch: &mut AzEvalScratch,
    ) -> f32 {
        self.evaluate_with_scratch_output(position, moves, scratch)
            .value
    }

    pub(super) fn evaluate_with_scratch_output(
        &self,
        position: &Position,
        moves: &[Move],
        scratch: &mut AzEvalScratch,
    ) -> AzEvalOutput {
        crate::scope_profile!("az.evaluate_with_scratch");
        let features = std::mem::take(&mut scratch.features);
        let accumulator = AzEvalAccumulator::new(self, position);
        scratch.hidden = accumulator.into_hidden_sum();
        {
            crate::scope_profile!("az.eval.activation_norm");
            relu_in_place(&mut scratch.hidden);
            rms_norm_in_place(&mut scratch.hidden);
        }
        let (value_wdl, value) = {
            crate::scope_profile!("az.eval.value_head");
            self.value_wdl_from_hidden_into(
                &scratch.hidden,
                &features,
                &mut scratch.value_head,
                &mut scratch.value_head2,
            )
        };
        let moves_left = {
            crate::scope_profile!("az.eval.moves_left_head");
            self.moves_left_from_hidden_into(&scratch.hidden, &mut scratch.value_head)
        };
        self.evaluate_prepared_hidden_with_scratch(position, &features, value, moves, scratch);
        scratch.features = features;
        AzEvalOutput {
            value_wdl,
            value,
            moves_left,
        }
    }

    pub(super) fn evaluate_incremental_with_scratch_output(
        &self,
        position: &Position,
        accumulator_hidden: &[f32],
        moves: &[Move],
        scratch: &mut AzEvalScratch,
    ) -> AzEvalOutput {
        crate::scope_profile!("az.evaluate_incremental_with_scratch");
        scratch.hidden.resize(self.hidden_size, 0.0);
        scratch.hidden.copy_from_slice(accumulator_hidden);
        {
            crate::scope_profile!("az.eval.activation_norm");
            relu_in_place(&mut scratch.hidden);
            rms_norm_in_place(&mut scratch.hidden);
        }
        let (value_wdl, value) = {
            crate::scope_profile!("az.eval.value_head");
            self.value_wdl_from_hidden_into(
                &scratch.hidden,
                &[],
                &mut scratch.value_head,
                &mut scratch.value_head2,
            )
        };
        let moves_left = {
            crate::scope_profile!("az.eval.moves_left_head");
            self.moves_left_from_hidden_into(&scratch.hidden, &mut scratch.value_head)
        };
        self.evaluate_prepared_hidden_with_scratch(position, &[], value, moves, scratch);
        AzEvalOutput {
            value_wdl,
            value,
            moves_left,
        }
    }

    fn evaluate_prepared_hidden_with_scratch(
        &self,
        _position: &Position,
        features: &[usize],
        value: f32,
        moves: &[Move],
        scratch: &mut AzEvalScratch,
    ) -> f32 {
        scratch.logits.resize(moves.len(), 0.0);
        scratch.logits.fill(0.0);
        let move_map = move_map();
        let width = self.branch_width();
        for branch in 0..2 {
            let hidden = &scratch.hidden[branch * width..(branch + 1) * width];
            {
                crate::scope_profile!("az.eval.policy_embeddings");
                self.policy_pair_context_into(hidden, &mut scratch.policy_pair_context);
                self.policy_move_context_into(hidden, &mut scratch.policy_move_context);
            }
            let mut from_used = [false; BOARD_SIZE];
            let mut to_used = [false; BOARD_SIZE];
            let mut from_squares = [0usize; BOARD_SIZE];
            let mut to_squares = [0usize; BOARD_SIZE];
            let mut from_count = 0usize;
            let mut to_count = 0usize;
            for &mv in moves {
                let transformed = if branch == 0 {
                    mv
                } else {
                    mirror_file_move(mv)
                };
                let from = transformed.from as usize;
                let to = transformed.to as usize;
                if !from_used[from] {
                    from_used[from] = true;
                    from_squares[from_count] = from;
                    from_count += 1;
                }
                if !to_used[to] {
                    to_used[to] = true;
                    to_squares[to_count] = to;
                    to_count += 1;
                }
            }
            self.policy_square_scores_for_squares_into(
                hidden,
                &from_squares[..from_count],
                &to_squares[..to_count],
                &mut scratch.policy_from_scores,
                &mut scratch.policy_to_scores,
            );
            for (index, &mv) in moves.iter().enumerate() {
                let transformed = if branch == 0 {
                    mv
                } else {
                    mirror_file_move(mv)
                };
                let sparse = transformed.from as usize * BOARD_SIZE + transformed.to as usize;
                let dense = move_map.sparse_to_dense[sparse];
                debug_assert!(
                    dense != u16::MAX,
                    "invalid policy move {}->{}",
                    mv.from,
                    mv.to
                );
                let move_index = dense as usize;
                scratch.logits[index] += 0.5
                    * self.policy_logit_from_hidden_index(
                        &scratch.policy_pair_context,
                        &scratch.policy_move_context,
                        &scratch.policy_from_scores,
                        &scratch.policy_to_scores,
                        move_index,
                        transformed.from as usize,
                        transformed.to as usize,
                    );
            }
        }
        let _ = features;
        value
    }

    #[allow(dead_code)]
    fn value_from_hidden_into(
        &self,
        hidden: &[f32],
        features: &[usize],
        value_head: &mut Vec<f32>,
    ) -> f32 {
        let mut value_head2 = Vec::new();
        let probs = self.value_wdl_from_hidden_into(hidden, features, value_head, &mut value_head2);
        probs.1
    }

    fn value_wdl_from_hidden_into(
        &self,
        hidden: &[f32],
        features: &[usize],
        value_head: &mut Vec<f32>,
        value_head2: &mut Vec<f32>,
    ) -> ([f32; WDL_HEAD_SIZE], f32) {
        let _ = features;
        let mut logits = [0.0f32; WDL_HEAD_SIZE];
        let width = self.branch_width();
        for branch in 0..2 {
            self.value_logits_from_branch(
                &hidden[branch * width..(branch + 1) * width],
                value_head,
                value_head2,
                &mut logits,
            );
        }
        for logit in &mut logits {
            *logit *= 0.5;
        }
        let wdl = softmax_fixed3(logits);
        let q = wdl[0] - wdl[2];
        (wdl, q)
    }

    fn value_logits_from_branch(
        &self,
        hidden: &[f32],
        value_head: &mut Vec<f32>,
        value_head2: &mut Vec<f32>,
        logits: &mut [f32; WDL_HEAD_SIZE],
    ) {
        let width = self.branch_width();
        value_head.resize(VALUE_HEAD_SIZE, 0.0);
        value_head.copy_from_slice(&self.value_head_bias);
        for (feature, value) in value_head.iter_mut().enumerate().take(VALUE_HEAD_SIZE) {
            let row = &self.value_head_hidden[feature * width..(feature + 1) * width];
            *value = (*value + dot_product(hidden, row)).max(0.0);
        }
        value_head2.resize(VALUE_HEAD_SIZE, 0.0);
        value_head2.copy_from_slice(&self.value_head_bias2);
        for (feature, value) in value_head2.iter_mut().enumerate().take(VALUE_HEAD_SIZE) {
            let row = &self.value_head_hidden2
                [feature * VALUE_HEAD_SIZE..(feature + 1) * VALUE_HEAD_SIZE];
            *value = (*value + dot_product(value_head, row)).max(0.0);
        }
        for (out, logit) in logits.iter_mut().enumerate() {
            let row = &self.value_head_output[out * VALUE_HEAD_SIZE..(out + 1) * VALUE_HEAD_SIZE];
            *logit += dot_product(value_head2, row);
        }
    }

    fn moves_left_from_hidden_into(&self, hidden: &[f32], moves_left_head: &mut Vec<f32>) -> f32 {
        let width = self.branch_width();
        let mut logit = 0.0;
        for branch in 0..2 {
            logit += self.moves_left_logit_from_branch(
                &hidden[branch * width..(branch + 1) * width],
                moves_left_head,
            );
        }
        softplus(logit * 0.5)
    }

    fn moves_left_logit_from_branch(&self, hidden: &[f32], moves_left_head: &mut Vec<f32>) -> f32 {
        moves_left_head.resize(VALUE_HEAD_SIZE, 0.0);
        moves_left_head.copy_from_slice(&self.moves_left_bias_hidden);
        for (feature, value) in moves_left_head.iter_mut().enumerate().take(VALUE_HEAD_SIZE) {
            let width = self.branch_width();
            let hidden_row = &self.moves_left_hidden[feature * width..(feature + 1) * width];
            *value += dot_product(hidden, hidden_row);
            *value = (*value).max(0.0);
        }
        dot_product(moves_left_head, &self.moves_left_output) + self.moves_left_bias[0]
    }

    fn policy_square_scores_for_squares_into(
        &self,
        hidden: &[f32],
        from_squares: &[usize],
        to_squares: &[usize],
        from_scores: &mut Vec<f32>,
        to_scores: &mut Vec<f32>,
    ) {
        from_scores.resize(BOARD_SIZE, 0.0);
        to_scores.resize(BOARD_SIZE, 0.0);
        for &square in from_squares {
            let start = square * self.branch_width();
            let end = start + self.branch_width();
            from_scores[square] = dot_product(hidden, &self.policy_from_hidden[start..end]);
        }
        for &square in to_squares {
            let start = square * self.branch_width();
            let end = start + self.branch_width();
            to_scores[square] = dot_product(hidden, &self.policy_to_hidden[start..end]);
        }
    }

    fn policy_pair_context_into(&self, hidden: &[f32], out: &mut Vec<f32>) {
        out.resize(POLICY_PAIR_CONTEXT_SIZE, 0.0);
        out.copy_from_slice(&self.policy_pair_context_bias);
        for (feature, value) in out.iter_mut().enumerate().take(POLICY_PAIR_CONTEXT_SIZE) {
            let width = self.branch_width();
            let hidden_row =
                &self.policy_pair_context_hidden[feature * width..(feature + 1) * width];
            *value += dot_product(hidden, hidden_row);
            *value = (*value).max(0.0);
        }
    }

    fn policy_move_context_into(&self, hidden: &[f32], out: &mut Vec<f32>) {
        out.resize(POLICY_MOVE_EMBED_SIZE, 0.0);
        for (feature, value) in out.iter_mut().enumerate().take(POLICY_MOVE_EMBED_SIZE) {
            let width = self.branch_width();
            let hidden_row =
                &self.policy_move_context_hidden[feature * width..(feature + 1) * width];
            *value = dot_product(hidden, hidden_row);
        }
    }

    fn validate(&self) -> io::Result<()> {
        let arch = &self.arch;
        if arch.hidden_size != self.hidden_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "aznnue arch.hidden_size does not match the cached hidden_size field",
            ));
        }
        if let Err(err) = arch.validate() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("aznnue arch invalid: {err}"),
            ));
        }
        let hidden = arch.hidden_size;
        macro_rules! validate_tensor {
            ($field:ident, [$($dim:expr),+]) => {
                let expected = [$($dim),+].into_iter().product::<usize>();
                if self.$field.len() != expected {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "az model tensor `{}` length mismatch: got {}, expected {}",
                            stringify!($field),
                            self.$field.len(),
                            expected
                        ),
                    ));
                }
            };
        }
        az_weight_tensors!(validate_tensor, hidden);
        Ok(())
    }

    fn policy_logit_from_hidden_index(
        &self,
        policy_pair_context: &[f32],
        policy_move_context: &[f32],
        from_scores: &[f32],
        to_scores: &[f32],
        move_index: usize,
        from: usize,
        to: usize,
    ) -> f32 {
        self.policy_move_bias[move_index]
            + from_scores[from]
            + to_scores[to]
            + dot_product(
                policy_pair_context,
                &self.policy_pair_embedding[move_index * POLICY_PAIR_CONTEXT_SIZE
                    ..(move_index + 1) * POLICY_PAIR_CONTEXT_SIZE],
            )
            + dot_product(
                policy_move_context,
                &self.policy_move_embedding[move_index * POLICY_MOVE_EMBED_SIZE
                    ..(move_index + 1) * POLICY_MOVE_EMBED_SIZE],
            )
    }
}

pub fn benchmark_training(
    model: &mut AzNnue,
    sample_count: usize,
    epochs: usize,
    batch_size: usize,
    lr: f32,
    seed: u64,
) -> AzTrainBenchmark {
    let mut rng = SplitMix64::new(seed);
    let mut samples = Vec::with_capacity(sample_count);
    for index in 0..sample_count {
        let feature_count = 24 + (rng.next_u64() as usize % 16);
        let mut features = Vec::with_capacity(feature_count);
        for _ in 0..feature_count {
            features.push((rng.next_u64() as usize) % AZ_NNUE_INPUT_SIZE);
        }
        features.sort_unstable();
        features.dedup();

        let value = rng.unit_f32() * 2.0 - 1.0;
        let move_count = 12 + (rng.next_u64() as usize % 24);
        let mut move_indices = Vec::with_capacity(move_count);
        while move_indices.len() < move_count {
            let candidate = (rng.next_u64() as usize) % DENSE_MOVE_SPACE;
            if !move_indices.contains(&candidate) {
                move_indices.push(candidate);
            }
        }
        let mut policy = (0..move_count)
            .map(|_| rng.unit_f32().max(1e-6))
            .collect::<Vec<_>>();
        let policy_sum = policy.iter().sum::<f32>().max(1e-6);
        for value in &mut policy {
            *value /= policy_sum;
        }
        samples.push(AzTrainingSample {
            features,
            move_indices,
            policy,
            value_wdl: scalar_value_to_wdl_target(value),
            value,
            side_sign: 1.0,
            moves_left: 0.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            search_simulations: 0,
            meta: AzSampleMeta::default(),
        });
        if index + 1 == sample_count {
            break;
        }
    }
    let stats = train_samples(model, &samples, epochs, lr, batch_size, &mut rng);
    AzTrainBenchmark {
        loss: stats.loss,
        value_loss: stats.value_loss,
        policy_ce: stats.policy_ce,
    }
}

fn softmax_fixed3(logits: [f32; 3]) -> [f32; 3] {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out = [
        (logits[0] - max_logit).exp(),
        (logits[1] - max_logit).exp(),
        (logits[2] - max_logit).exp(),
    ];
    let sum = (out[0] + out[1] + out[2]).max(f32::MIN_POSITIVE);
    out[0] /= sum;
    out[1] /= sum;
    out[2] /= sum;
    out
}

pub(super) fn scalar_value_to_wdl_target(value: f32) -> [f32; 3] {
    let value = value.clamp(-1.0, 1.0);
    if value >= 0.0 {
        [value, 1.0 - value, 0.0]
    } else {
        [0.0, 1.0 + value, -value]
    }
}

pub(super) fn normalize_wdl_target(mut wdl: [f32; WDL_HEAD_SIZE]) -> [f32; WDL_HEAD_SIZE] {
    for value in &mut wdl {
        *value = value.max(0.0);
    }
    let sum = wdl.iter().sum::<f32>();
    if sum.is_finite() && sum > 1.0e-6 {
        for value in &mut wdl {
            *value /= sum;
        }
        wdl
    } else {
        [0.0, 1.0, 0.0]
    }
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else if value < -20.0 {
        value.exp()
    } else {
        value.exp().ln_1p()
    }
}

fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    #[cfg(target_arch = "aarch64")]
    if left.len() >= 16 {
        // AArch64 guarantees NEON; avoid scalar floating-point dependency chains.
        return unsafe { dot_product_neon(left, right) };
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(target_arch = "x86_64")]
        if left.len() >= 64
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: runtime detection above guarantees AVX2 and FMA support.
            return unsafe { dot_product_avx2_fma(left, right) };
        }
        if left.len() >= 64 && std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime detection above guarantees AVX2 support.
            return unsafe { dot_product_avx2(left, right) };
        }
    }
    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;
    let chunks = left.len() / 4;
    for chunk in 0..chunks {
        let index = chunk * 4;
        sum0 += left[index] * right[index];
        sum1 += left[index + 1] * right[index + 1];
        sum2 += left[index + 2] * right[index + 2];
        sum3 += left[index + 3] * right[index + 3];
    }
    let mut sum = (sum0 + sum1) + (sum2 + sum3);
    for index in (chunks * 4)..left.len() {
        sum += left[index] * right[index];
    }
    sum
}

fn add_scaled_feature_row(
    hidden: &mut [f32],
    input_hidden: &[f32],
    hidden_size: usize,
    feature: usize,
    scale: f32,
) {
    let row = &input_hidden[feature * hidden_size..(feature + 1) * hidden_size];
    debug_assert_eq!(hidden.len(), row.len());
    #[cfg(target_arch = "aarch64")]
    if hidden_size >= 32 {
        // AArch64 guarantees NEON.
        unsafe { add_scaled_feature_row_neon(hidden, row, scale) };
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(target_arch = "x86_64")]
        if hidden_size >= 64
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: runtime detection above guarantees AVX2 and FMA support.
            unsafe { add_scaled_feature_row_avx2_fma(hidden, row, scale) };
            return;
        }
        if hidden_size >= 64 && std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime detection above guarantees AVX2 support.
            unsafe {
                add_scaled_feature_row_avx2(hidden, row, scale);
            }
            return;
        }
    }
    for (left, &right) in hidden.iter_mut().zip(row) {
        *left += scale * right;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_product_neon(left: &[f32], right: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let chunks = left.len() / 16;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    for chunk in 0..chunks {
        let index = chunk * 16;
        unsafe {
            acc0 = vfmaq_f32(
                acc0,
                vld1q_f32(left.as_ptr().add(index)),
                vld1q_f32(right.as_ptr().add(index)),
            );
            acc1 = vfmaq_f32(
                acc1,
                vld1q_f32(left.as_ptr().add(index + 4)),
                vld1q_f32(right.as_ptr().add(index + 4)),
            );
            acc2 = vfmaq_f32(
                acc2,
                vld1q_f32(left.as_ptr().add(index + 8)),
                vld1q_f32(right.as_ptr().add(index + 8)),
            );
            acc3 = vfmaq_f32(
                acc3,
                vld1q_f32(left.as_ptr().add(index + 12)),
                vld1q_f32(right.as_ptr().add(index + 12)),
            );
        }
    }
    let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    for index in (chunks * 16)..left.len() {
        sum += left[index] * right[index];
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn add_scaled_feature_row_neon(hidden: &mut [f32], row: &[f32], scale: f32) {
    use std::arch::aarch64::*;
    let scale_vector = vdupq_n_f32(scale);
    let chunks = hidden.len() / 4;
    for chunk in 0..chunks {
        let index = chunk * 4;
        unsafe {
            let left = vld1q_f32(hidden.as_ptr().add(index));
            let right = vld1q_f32(row.as_ptr().add(index));
            vst1q_f32(
                hidden.as_mut_ptr().add(index),
                vfmaq_f32(left, right, scale_vector),
            );
        }
    }
    for index in (chunks * 4)..hidden.len() {
        hidden[index] += row[index] * scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_avx2_fma(left: &[f32], right: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let chunks = left.len() / 32;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    for chunk in 0..chunks {
        let index = chunk * 32;
        unsafe {
            acc0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(left.as_ptr().add(index)),
                _mm256_loadu_ps(right.as_ptr().add(index)),
                acc0,
            );
            acc1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(left.as_ptr().add(index + 8)),
                _mm256_loadu_ps(right.as_ptr().add(index + 8)),
                acc1,
            );
            acc2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(left.as_ptr().add(index + 16)),
                _mm256_loadu_ps(right.as_ptr().add(index + 16)),
                acc2,
            );
            acc3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(left.as_ptr().add(index + 24)),
                _mm256_loadu_ps(right.as_ptr().add(index + 24)),
                acc3,
            );
        }
    }
    let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let mut lanes = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
    let mut sum = lanes.iter().sum::<f32>();
    for index in (chunks * 32)..left.len() {
        sum += left[index] * right[index];
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2(left: &[f32], right: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let chunks = left.len() / 8;
    let mut acc = _mm256_setzero_ps();
    for chunk in 0..chunks {
        let index = chunk * 8;
        unsafe {
            let l = _mm256_loadu_ps(left.as_ptr().add(index));
            let r = _mm256_loadu_ps(right.as_ptr().add(index));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(l, r));
        }
    }
    let mut lanes = [0.0f32; 8];
    unsafe {
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    }
    let mut sum = lanes.iter().sum::<f32>();
    for index in (chunks * 8)..left.len() {
        sum += left[index] * right[index];
    }
    sum
}

#[cfg(target_arch = "x86")]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2(left: &[f32], right: &[f32]) -> f32 {
    use std::arch::x86::*;
    let chunks = left.len() / 8;
    let mut acc = _mm256_setzero_ps();
    for chunk in 0..chunks {
        let index = chunk * 8;
        unsafe {
            let l = _mm256_loadu_ps(left.as_ptr().add(index));
            let r = _mm256_loadu_ps(right.as_ptr().add(index));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(l, r));
        }
    }
    let mut lanes = [0.0f32; 8];
    unsafe {
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    }
    let mut sum = lanes.iter().sum::<f32>();
    for index in (chunks * 8)..left.len() {
        sum += left[index] * right[index];
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn add_scaled_feature_row_avx2_fma(hidden: &mut [f32], row: &[f32], scale: f32) {
    use std::arch::x86_64::*;
    let scale_scalar = scale;
    let scale = _mm256_set1_ps(scale_scalar);
    let chunks = hidden.len() / 8;
    for chunk in 0..chunks {
        let index = chunk * 8;
        unsafe {
            let left = _mm256_loadu_ps(hidden.as_ptr().add(index));
            let right = _mm256_loadu_ps(row.as_ptr().add(index));
            _mm256_storeu_ps(
                hidden.as_mut_ptr().add(index),
                _mm256_fmadd_ps(right, scale, left),
            );
        }
    }
    for index in (chunks * 8)..hidden.len() {
        hidden[index] += row[index] * scale_scalar;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_scaled_feature_row_avx2(hidden: &mut [f32], row: &[f32], scale: f32) {
    use std::arch::x86_64::*;
    let scale_scalar = scale;
    let scale = _mm256_set1_ps(scale_scalar);
    let chunks = hidden.len() / 8;
    for chunk in 0..chunks {
        let index = chunk * 8;
        unsafe {
            let left = _mm256_loadu_ps(hidden.as_ptr().add(index));
            let right = _mm256_loadu_ps(row.as_ptr().add(index));
            _mm256_storeu_ps(
                hidden.as_mut_ptr().add(index),
                _mm256_add_ps(left, _mm256_mul_ps(scale, right)),
            );
        }
    }
    for index in (chunks * 8)..hidden.len() {
        hidden[index] += row[index] * scale_scalar;
    }
}

#[cfg(target_arch = "x86")]
#[target_feature(enable = "avx2")]
unsafe fn add_scaled_feature_row_avx2(hidden: &mut [f32], row: &[f32], scale: f32) {
    use std::arch::x86::*;
    let scale_scalar = scale;
    let scale = _mm256_set1_ps(scale_scalar);
    let chunks = hidden.len() / 8;
    for chunk in 0..chunks {
        let index = chunk * 8;
        unsafe {
            let left = _mm256_loadu_ps(hidden.as_ptr().add(index));
            let right = _mm256_loadu_ps(row.as_ptr().add(index));
            _mm256_storeu_ps(
                hidden.as_mut_ptr().add(index),
                _mm256_add_ps(left, _mm256_mul_ps(scale, right)),
            );
        }
    }
    for index in (chunks * 8)..hidden.len() {
        hidden[index] += row[index] * scale_scalar;
    }
}

fn relu_in_place(values: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    if values.len() >= 32 {
        // AArch64 guarantees NEON.
        unsafe { relu_in_place_neon(values) };
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if values.len() >= 64 && std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime detection above guarantees AVX2 support.
            unsafe {
                relu_in_place_avx2(values);
            }
            return;
        }
    }
    for value in values {
        *value = value.max(0.0);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn relu_in_place_neon(values: &mut [f32]) {
    use std::arch::aarch64::*;
    let zero = vdupq_n_f32(0.0);
    let chunks = values.len() / 4;
    for chunk in 0..chunks {
        let index = chunk * 4;
        unsafe {
            let value = vld1q_f32(values.as_ptr().add(index));
            vst1q_f32(values.as_mut_ptr().add(index), vmaxq_f32(value, zero));
        }
    }
    for value in &mut values[(chunks * 4)..] {
        *value = value.max(0.0);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn relu_in_place_avx2(values: &mut [f32]) {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_ps();
    let chunks = values.len() / 8;
    for chunk in 0..chunks {
        let index = chunk * 8;
        unsafe {
            let value = _mm256_loadu_ps(values.as_ptr().add(index));
            _mm256_storeu_ps(values.as_mut_ptr().add(index), _mm256_max_ps(value, zero));
        }
    }
    for value in &mut values[(chunks * 8)..] {
        *value = value.max(0.0);
    }
}

#[cfg(target_arch = "x86")]
#[target_feature(enable = "avx2")]
unsafe fn relu_in_place_avx2(values: &mut [f32]) {
    use std::arch::x86::*;
    let zero = _mm256_setzero_ps();
    let chunks = values.len() / 8;
    for chunk in 0..chunks {
        let index = chunk * 8;
        unsafe {
            let value = _mm256_loadu_ps(values.as_ptr().add(index));
            _mm256_storeu_ps(values.as_mut_ptr().add(index), _mm256_max_ps(value, zero));
        }
    }
    for value in &mut values[(chunks * 8)..] {
        *value = value.max(0.0);
    }
}

fn rms_norm_in_place(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }
    let sum_squares = dot_product(values, values);
    let inv_rms = (sum_squares / values.len() as f32 + RMS_NORM_EPS)
        .sqrt()
        .recip();
    #[cfg(target_arch = "aarch64")]
    if values.len() >= 32 {
        unsafe { scale_in_place_neon(values, inv_rms) };
        return;
    }
    for value in values {
        *value *= inv_rms;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn scale_in_place_neon(values: &mut [f32], scale: f32) {
    use std::arch::aarch64::*;
    let scale = vdupq_n_f32(scale);
    let chunks = values.len() / 4;
    for chunk in 0..chunks {
        let index = chunk * 4;
        unsafe {
            let value = vld1q_f32(values.as_ptr().add(index));
            vst1q_f32(values.as_mut_ptr().add(index), vmulq_f32(value, scale));
        }
    }
    for value in &mut values[(chunks * 4)..] {
        *value *= vgetq_lane_f32::<0>(scale);
    }
}

pub struct SplitMix64 {
    state: u64,
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

    fn weight(&mut self, scale: f32) -> f32 {
        (self.unit_f32() * 2.0 - 1.0) * scale
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut mixed = value;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^ (mixed >> 31)
}

const fn is_advisor_pos(rank: usize, file: usize) -> bool {
    matches!(
        (rank, file),
        (0, 3) | (0, 5) | (1, 4) | (2, 3) | (2, 5) | (7, 3) | (7, 5) | (8, 4) | (9, 3) | (9, 5)
    )
}

const fn is_elephant_pos(rank: usize, file: usize) -> bool {
    matches!(
        (rank, file),
        (0, 2)
            | (0, 6)
            | (2, 0)
            | (2, 4)
            | (2, 8)
            | (4, 2)
            | (4, 6)
            | (5, 2)
            | (5, 6)
            | (7, 0)
            | (7, 4)
            | (7, 8)
            | (9, 2)
            | (9, 6)
    )
}

const fn is_valid_policy_move(from: usize, to: usize) -> bool {
    let from_file = from % BOARD_FILES;
    let from_rank = from / BOARD_FILES;
    let to_file = to % BOARD_FILES;
    let to_rank = to / BOARD_FILES;

    let df_signed = to_file as i32 - from_file as i32;
    let dr_signed = to_rank as i32 - from_rank as i32;
    let df = if df_signed < 0 { -df_signed } else { df_signed };
    let dr = if dr_signed < 0 { -dr_signed } else { dr_signed };

    if df == 0 || dr == 0 {
        return true;
    }
    if (df == 1 && dr == 2) || (df == 2 && dr == 1) {
        return true;
    }
    if df == 1
        && dr == 1
        && is_advisor_pos(from_rank, from_file)
        && is_advisor_pos(to_rank, to_file)
    {
        return true;
    }
    if df == 2
        && dr == 2
        && is_elephant_pos(from_rank, from_file)
        && is_elephant_pos(to_rank, to_file)
    {
        return true;
    }
    false
}

const fn compute_dense_move_count() -> usize {
    let mut count = 0;
    let mut from = 0;
    while from < BOARD_SIZE {
        let mut to = 0;
        while to < BOARD_SIZE {
            if from != to && is_valid_policy_move(from, to) {
                count += 1;
            }
            to += 1;
        }
        from += 1;
    }
    count
}

struct MoveMap {
    sparse_to_dense: [u16; SPARSE_MOVE_SPACE],
    #[allow(dead_code)]
    dense_to_sparse: [u16; DENSE_MOVE_SPACE],
}

fn move_map() -> &'static MoveMap {
    use std::sync::OnceLock;
    static MAP: OnceLock<MoveMap> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut sparse_to_dense = [u16::MAX; SPARSE_MOVE_SPACE];
        let mut dense_to_sparse = [0u16; DENSE_MOVE_SPACE];
        let mut idx = 0usize;
        for from in 0..BOARD_SIZE {
            for to in 0..BOARD_SIZE {
                if from != to && is_valid_policy_move(from, to) {
                    let sparse = from * BOARD_SIZE + to;
                    sparse_to_dense[sparse] = idx as u16;
                    dense_to_sparse[idx] = sparse as u16;
                    idx += 1;
                }
            }
        }
        assert_eq!(idx, DENSE_MOVE_SPACE);
        MoveMap {
            sparse_to_dense,
            dense_to_sparse,
        }
    })
}

#[cfg_attr(not(feature = "gpu-train"), allow(dead_code))]
pub(super) fn policy_move_from_features() -> &'static [f32] {
    use std::sync::OnceLock;
    static FEATURES: OnceLock<Vec<f32>> = OnceLock::new();
    FEATURES.get_or_init(|| {
        let mut features = vec![0.0; DENSE_MOVE_SPACE * BOARD_SIZE];
        for (move_index, &sparse) in move_map().dense_to_sparse.iter().enumerate() {
            let from = sparse as usize / BOARD_SIZE;
            features[move_index * BOARD_SIZE + from] = 1.0;
        }
        features
    })
}

#[cfg_attr(not(feature = "gpu-train"), allow(dead_code))]
pub(super) fn policy_move_to_features() -> &'static [f32] {
    use std::sync::OnceLock;
    static FEATURES: OnceLock<Vec<f32>> = OnceLock::new();
    FEATURES.get_or_init(|| {
        let mut features = vec![0.0; DENSE_MOVE_SPACE * BOARD_SIZE];
        for (move_index, &sparse) in move_map().dense_to_sparse.iter().enumerate() {
            let to = sparse as usize % BOARD_SIZE;
            features[move_index * BOARD_SIZE + to] = 1.0;
        }
        features
    })
}

#[cfg_attr(not(feature = "gpu-train"), allow(dead_code))]
pub(super) fn policy_mirror_indices() -> &'static [u32] {
    use std::sync::OnceLock;
    static INDICES: OnceLock<Vec<u32>> = OnceLock::new();
    INDICES.get_or_init(|| {
        move_map()
            .dense_to_sparse
            .iter()
            .map(|&sparse| {
                let from = sparse as usize / BOARD_SIZE;
                let to = sparse as usize % BOARD_SIZE;
                dense_move_index(mirror_file_move(Move::new(from, to))) as u32
            })
            .collect()
    })
}

fn dense_move_index(mv: Move) -> usize {
    let sparse = mv.from as usize * BOARD_SIZE + mv.to as usize;
    let dense = move_map().sparse_to_dense[sparse];
    debug_assert!(
        dense != u16::MAX,
        "invalid policy move {}->{}",
        mv.from,
        mv.to
    );
    dense as usize
}

#[cfg(test)]
fn replay_pool_test_fixture() -> AzExperiencePool {
    fn sample(update: u32, game_id: u64, ply: u16) -> AzTrainingSample {
        AzTrainingSample {
            features: vec![1, 2, 3],
            move_indices: vec![0, 1],
            policy: vec![0.6, 0.4],
            value_wdl: scalar_value_to_wdl_target(0.1),
            value: 0.1,
            side_sign: 1.0,
            moves_left: 0.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            search_simulations: 0,
            meta: AzSampleMeta {
                generation_update: update,
                game_id,
                ply,
                root_q: 0.11,
                best_q: 0.33,
                played_q: 0.02,
                best_visits: 88,
                played_visits: 13,
                best_index: 1,
                played_index: 0,
            },
        }
    }
    let mut pool = AzExperiencePool::new(100);
    pool.add_games(vec![
        vec![sample(7, 42, 9)],
        vec![sample(7, 43, 1), sample(7, 43, 2)],
    ]);
    pool
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn dense_move_space_matches_enumeration() {
        let map = move_map();
        assert_eq!(DENSE_MOVE_SPACE, 2086);
        for i in 0..DENSE_MOVE_SPACE {
            let sparse = map.dense_to_sparse[i] as usize;
            assert_eq!(map.sparse_to_dense[sparse], i as u16);
        }
    }

    #[test]
    fn random_initial_value_head_is_neutral() {
        let model = AzNnue::random_with_arch(AzNnueArch::with_hidden_size(512), 20260409);
        assert!(model.value_head_output.iter().all(|&weight| weight == 0.0));

        let position = Position::startpos();
        let moves = position.legal_moves();
        let value = model.evaluate_value(&position, &moves);
        assert!(value.abs() < 1e-6, "initial startpos value={value}");
    }

    #[test]
    fn network_is_exactly_file_reflection_equivariant() {
        let mut model = AzNnue::random(192, 20260723);
        for (index, weight) in model.value_head_output.iter_mut().enumerate() {
            *weight = ((index * 17 + 3) as f32).sin() * 0.05;
        }
        for (index, weight) in model.moves_left_output.iter_mut().enumerate() {
            *weight = ((index * 11 + 5) as f32).cos() * 0.03;
        }
        let position =
            Position::from_fen("4k1b2/4a4/4ba3/p8/4cN3/3n2N1P/c8/4C4/4A4/2B1KAB2 b - - 0 1")
                .unwrap();
        let mirrored = position.mirror_files();
        let moves = position.legal_moves();
        let mirrored_moves = mirrored.legal_moves();
        let mut scratch = AzEvalScratch::new(model.arch);
        let output = model.evaluate_with_scratch_output(&position, &moves, &mut scratch);
        let logits = scratch.logits.clone();
        let mut mirrored_scratch = AzEvalScratch::new(model.arch);
        let mirrored_output =
            model.evaluate_with_scratch_output(&mirrored, &mirrored_moves, &mut mirrored_scratch);

        assert!((output.value - mirrored_output.value).abs() < 1e-6);
        assert!((output.moves_left - mirrored_output.moves_left).abs() < 1e-5);
        for (index, &mv) in moves.iter().enumerate() {
            let mirrored_mv = mirror_file_move(mv);
            let mirrored_index = mirrored_moves
                .iter()
                .position(|&candidate| candidate == mirrored_mv)
                .unwrap();
            assert!(
                (logits[index] - mirrored_scratch.logits[mirrored_index]).abs() < 1e-5,
                "policy mismatch for {mv}"
            );
        }
    }

    #[test]
    fn single_accumulator_increment_matches_full_refresh() {
        let model = AzNnue::random(192, 41);
        let before = Position::startpos();
        let mv = before.legal_moves()[7];
        let moved = before.piece_at(mv.from as usize).unwrap();
        let captured = before.piece_at(mv.to as usize);
        let mut after = before.clone();
        after.make_move(mv);
        let mut incremental = AzEvalAccumulator::new(&model, &before).into_hidden_sum();
        AzEvalAccumulator::apply_transition_to_hidden(
            &model,
            &before,
            &after,
            mv,
            moved,
            captured,
            &mut incremental,
        );
        let refreshed = AzEvalAccumulator::new(&model, &after).into_hidden_sum();
        assert_eq!(incremental.len(), model.hidden_size);
        for (left, right) in incremental.iter().zip(refreshed) {
            assert!((left - right).abs() < 1e-5);
        }
    }

    #[test]
    fn scalar_value_head_starts_neutral() {
        let model = AzNnue::random(16, 7);
        let mut scratch = AzEvalScratch::new(model.arch);
        let value = model.value_from_hidden_into(&scratch.hidden, &[], &mut scratch.value_head);

        assert!(value.abs() < 1e-6);
    }

    #[test]
    fn arena_report_anchored_elo_tracks_score_vs_reference() {
        let reference = 1500.0f32;
        let stronger = AzArenaReport {
            wins: 6,
            losses: 3,
            draws: 1,
            ..AzArenaReport::default()
        };
        let weaker = AzArenaReport {
            wins: 3,
            losses: 6,
            draws: 1,
            ..AzArenaReport::default()
        };

        assert!(stronger.score_rate() > 0.5);
        assert!(stronger.anchored_elo(reference) > reference);
        assert!(weaker.score_rate() < 0.5);
        assert!(weaker.anchored_elo(reference) < reference);
    }

    #[cfg(feature = "gpu-train")]
    #[test]
    fn value_head_can_overfit_tiny_fixed_dataset() {
        let mut model = AzNnue::random(16, 7);
        model.hidden_bias.fill(0.1);
        model.hidden_bias.fill(0.1);

        let samples = vec![
            AzTrainingSample {
                features: vec![0],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value_wdl: scalar_value_to_wdl_target(1.0),
                value: 1.0,
                side_sign: 1.0,
                moves_left: 0.0,
                policy_weight: 1.0,
                value_weight: 1.0,
                search_simulations: 0,
                meta: AzSampleMeta::default(),
            },
            AzTrainingSample {
                features: vec![1],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value_wdl: scalar_value_to_wdl_target(-1.0),
                value: -1.0,
                side_sign: 1.0,
                moves_left: 0.0,
                policy_weight: 1.0,
                value_weight: 1.0,
                search_simulations: 0,
                meta: AzSampleMeta::default(),
            },
            AzTrainingSample {
                features: vec![2],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value_wdl: scalar_value_to_wdl_target(0.75),
                value: 0.75,
                side_sign: 1.0,
                moves_left: 0.0,
                policy_weight: 1.0,
                value_weight: 1.0,
                search_simulations: 0,
                meta: AzSampleMeta::default(),
            },
            AzTrainingSample {
                features: vec![3],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value_wdl: scalar_value_to_wdl_target(-0.75),
                value: -0.75,
                side_sign: 1.0,
                moves_left: 0.0,
                policy_weight: 1.0,
                value_weight: 1.0,
                search_simulations: 0,
                meta: AzSampleMeta::default(),
            },
        ];

        let mut rng = SplitMix64::new(17);
        let before = train_samples(&mut model, &samples, 1, 0.003, 4, &mut rng).value_loss;
        let after = train_samples(&mut model, &samples, 300, 0.003, 4, &mut rng).value_loss;

        assert!(after < before * 0.5, "before={before} after={after}");
        assert!(after < 0.35, "after={after}");
    }

    #[cfg(feature = "gpu-train")]
    #[test]
    fn batched_training_is_deterministic() {
        let samples = vec![
            AzTrainingSample {
                features: vec![0, 4, 8],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value_wdl: scalar_value_to_wdl_target(1.0),
                value: 1.0,
                side_sign: 1.0,
                moves_left: 0.0,
                policy_weight: 1.0,
                value_weight: 1.0,
                search_simulations: 0,
                meta: AzSampleMeta::default(),
            },
            AzTrainingSample {
                features: vec![1, 5, 9],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value_wdl: scalar_value_to_wdl_target(-1.0),
                value: -1.0,
                side_sign: 1.0,
                moves_left: 0.0,
                policy_weight: 1.0,
                value_weight: 1.0,
                search_simulations: 0,
                meta: AzSampleMeta::default(),
            },
            AzTrainingSample {
                features: vec![2, 6, 10],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value_wdl: scalar_value_to_wdl_target(0.5),
                value: 0.5,
                side_sign: 1.0,
                moves_left: 0.0,
                policy_weight: 1.0,
                value_weight: 1.0,
                search_simulations: 0,
                meta: AzSampleMeta::default(),
            },
            AzTrainingSample {
                features: vec![3, 7, 11],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value_wdl: scalar_value_to_wdl_target(-0.5),
                value: -0.5,
                side_sign: 1.0,
                moves_left: 0.0,
                policy_weight: 1.0,
                value_weight: 1.0,
                search_simulations: 0,
                meta: AzSampleMeta::default(),
            },
        ];
        let mut single = AzNnue::random(16, 23);
        single.hidden_bias.fill(0.1);
        single.hidden_bias.fill(0.1);
        let mut repeated = single.clone();

        let mut rng_single = SplitMix64::new(99);
        let mut rng_repeated = SplitMix64::new(99);
        let single_stats = train_samples(&mut single, &samples, 5, 0.003, 4, &mut rng_single);
        let repeated_stats = train_samples(&mut repeated, &samples, 5, 0.003, 4, &mut rng_repeated);

        assert!((single_stats.loss - repeated_stats.loss).abs() < 1e-5);
        assert!((single_stats.value_loss - repeated_stats.value_loss).abs() < 1e-5);
        assert!((single_stats.value_pred_sum - repeated_stats.value_pred_sum).abs() < 1e-4);
        assert!((single_stats.value_target_sum - repeated_stats.value_target_sum).abs() < 1e-6);
        assert!(
            single
                .value_head_output
                .iter()
                .zip(&repeated.value_head_output)
                .all(|(left, right)| (*left - *right).abs() < 1e-5)
        );
    }

    #[cfg(feature = "gpu-train")]
    #[test]
    fn value_only_training_updates_trunk_when_trunk_training_enabled() {
        let samples = vec![
            AzTrainingSample {
                features: vec![0, 4, 8],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value_wdl: scalar_value_to_wdl_target(1.0),
                value: 1.0,
                side_sign: 1.0,
                moves_left: 0.0,
                policy_weight: 1.0,
                value_weight: 1.0,
                search_simulations: 0,
                meta: AzSampleMeta::default(),
            },
            AzTrainingSample {
                features: vec![1, 5, 9],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value_wdl: scalar_value_to_wdl_target(-1.0),
                value: -1.0,
                side_sign: 1.0,
                moves_left: 0.0,
                policy_weight: 1.0,
                value_weight: 1.0,
                search_simulations: 0,
                meta: AzSampleMeta::default(),
            },
            AzTrainingSample {
                features: vec![2, 6, 10],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value_wdl: scalar_value_to_wdl_target(0.75),
                value: 0.75,
                side_sign: 1.0,
                moves_left: 0.0,
                policy_weight: 1.0,
                value_weight: 1.0,
                search_simulations: 0,
                meta: AzSampleMeta::default(),
            },
            AzTrainingSample {
                features: vec![3, 7, 11],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value_wdl: scalar_value_to_wdl_target(-0.75),
                value: -0.75,
                side_sign: 1.0,
                moves_left: 0.0,
                policy_weight: 1.0,
                value_weight: 1.0,
                search_simulations: 0,
                meta: AzSampleMeta::default(),
            },
        ];
        let mut model = AzNnue::random(8, 31);
        model.hidden_bias.fill(0.1);
        let before_input = model.input_hidden.clone();
        let before_bias = model.hidden_bias.clone();

        let mut rng = SplitMix64::new(32);
        let weights = AzTrainLossWeights {
            value: 1.0,
            policy: 0.0,
        };
        train_samples_weighted(&mut model, &samples, 20, 0.01, 4, &mut rng, weights);

        let input_changed = before_input
            .iter()
            .zip(&model.input_hidden)
            .any(|(left, right)| (*left - *right).abs() > 1e-7);
        let bias_changed = before_bias
            .iter()
            .zip(&model.hidden_bias)
            .any(|(left, right)| (*left - *right).abs() > 1e-7);
        assert!(
            input_changed || bias_changed,
            "value-only training should update trunk"
        );
    }

    #[test]
    fn aznnue_safetensors_roundtrip_matches_weights() {
        let model = AzNnue::random(16, 42);
        let path = std::env::temp_dir().join("chineseai_test_aznnue_roundtrip.safetensors");
        let _ = fs::remove_file(&path);
        model.save(&path).unwrap();
        let loaded = AzNnue::load(&path).unwrap();
        let _ = fs::remove_file(&path);
        assert_eq!(model.hidden_size, loaded.hidden_size);
        assert_eq!(model.input_hidden, loaded.input_hidden);
        assert_eq!(model.input_piece_hidden, loaded.input_piece_hidden);
        assert_eq!(model.input_rank_hidden, loaded.input_rank_hidden);
        assert_eq!(model.input_file_hidden, loaded.input_file_hidden);
        assert_eq!(
            model.input_king_piece_hidden,
            loaded.input_king_piece_hidden
        );
        assert_eq!(model.hidden_bias, loaded.hidden_bias);
        assert_eq!(model.value_head_hidden, loaded.value_head_hidden);
        assert_eq!(model.value_head_bias, loaded.value_head_bias);
        assert_eq!(model.value_head_hidden2, loaded.value_head_hidden2);
        assert_eq!(model.value_head_bias2, loaded.value_head_bias2);
        assert_eq!(model.value_head_output, loaded.value_head_output);
        assert_eq!(model.moves_left_hidden, loaded.moves_left_hidden);
        assert_eq!(model.moves_left_bias_hidden, loaded.moves_left_bias_hidden);
        assert_eq!(model.moves_left_output, loaded.moves_left_output);
        assert_eq!(model.moves_left_bias, loaded.moves_left_bias);
        assert_eq!(model.policy_move_bias, loaded.policy_move_bias);
        assert_eq!(model.policy_from_hidden, loaded.policy_from_hidden);
        assert_eq!(model.policy_to_hidden, loaded.policy_to_hidden);
        assert_eq!(
            model.policy_pair_context_hidden,
            loaded.policy_pair_context_hidden
        );
        assert_eq!(
            model.policy_pair_context_bias,
            loaded.policy_pair_context_bias
        );
        assert_eq!(model.policy_pair_embedding, loaded.policy_pair_embedding);
        assert_eq!(
            model.policy_move_context_hidden,
            loaded.policy_move_context_hidden
        );
        assert_eq!(model.policy_move_embedding, loaded.policy_move_embedding);
    }

    #[test]
    fn capped_swa_averages_all_model_weights_and_caps_count() {
        let mut average = AzNnue::random(4, 101);
        let current = AzNnue::random(4, 202);
        let first_before = average.input_hidden[17];
        let last_before = average.policy_move_embedding[31];
        let first_current = current.input_hidden[17];
        let last_current = current.policy_move_embedding[31];

        let mut count = average.update_capped_swa(&current, 1, 10);
        assert_eq!(count, 2);
        assert!((average.input_hidden[17] - (first_before + first_current) * 0.5).abs() < 1e-7);
        assert!(
            (average.policy_move_embedding[31] - (last_before + last_current) * 0.5).abs() < 1e-7
        );

        for _ in 0..20 {
            count = average.update_capped_swa(&current, count, 10);
        }
        assert_eq!(count, 10);
    }

    #[test]
    fn replay_pool_lz4_snapshot_roundtrip() {
        let path = std::env::temp_dir().join("chineseai_test_replay_roundtrip.replay.lz4");
        let _ = fs::remove_file(&path);
        let pool = super::replay_pool_test_fixture();
        pool.save_snapshot_lz4(&path).unwrap();
        let file_blob = fs::read(&path).unwrap();
        assert_eq!(&file_blob[0..4], b"AZRP");
        assert_eq!(&file_blob[8..12], b"CHNK");
        let loaded = AzExperiencePool::load_snapshot_lz4(&path, 100).unwrap();
        let _ = fs::remove_file(&path);
        assert_eq!(loaded.sample_count(), pool.sample_count());
        assert_eq!(loaded.capacity(), pool.capacity());
        let loaded_samples = loaded.all_samples();
        assert_eq!(loaded_samples[0].meta.generation_update, 7);
        assert_eq!(loaded_samples[0].meta.game_id, 42);
        assert_eq!(loaded_samples[0].meta.ply, 9);
        assert!((loaded_samples[0].meta.best_q - 0.33).abs() < 1e-6);
        assert_eq!(loaded_samples[0].meta.played_visits, 13);
    }

    #[test]
    fn replay_pool_prunes_whole_game_chunks() {
        fn sample(update: u32, game_id: u64, ply: u16) -> AzTrainingSample {
            AzTrainingSample {
                features: vec![1],
                move_indices: vec![0],
                policy: vec![1.0],
                value_wdl: scalar_value_to_wdl_target(0.0),
                value: 0.0,
                side_sign: 1.0,
                moves_left: 0.0,
                policy_weight: 1.0,
                value_weight: 1.0,
                search_simulations: 0,
                meta: AzSampleMeta {
                    generation_update: update,
                    game_id,
                    ply,
                    ..AzSampleMeta::default()
                },
            }
        }

        let mut pool = AzExperiencePool::new(4);
        pool.add_games(vec![
            vec![sample(1, 1, 0), sample(1, 1, 1)],
            vec![sample(2, 2, 0), sample(2, 2, 1)],
            vec![sample(3, 3, 0), sample(3, 3, 1)],
        ]);

        let stats = pool.window_stats(1);
        assert_eq!(pool.sample_count(), 4);
        assert_eq!(stats.chunks, 2);
        assert_eq!(stats.oldest_generation_update, 2);
        assert_eq!(stats.newest_generation_update, 3);
        assert_eq!(stats.window_games, 2);
        assert!((stats.recent_window_sample_fraction - 0.5).abs() < 1e-6);
        assert_eq!(pool.all_sample_groups().len(), 2);
    }

    #[test]
    fn replay_pool_mixed_recent_sampling_uses_requested_recent_fraction() {
        fn sample(update: u32, game_id: u64, ply: u16) -> AzTrainingSample {
            AzTrainingSample {
                features: vec![1],
                move_indices: vec![0],
                policy: vec![1.0],
                value_wdl: scalar_value_to_wdl_target(0.0),
                value: 0.0,
                side_sign: 1.0,
                moves_left: 0.0,
                policy_weight: 1.0,
                value_weight: 1.0,
                search_simulations: 0,
                meta: AzSampleMeta {
                    generation_update: update,
                    game_id,
                    ply,
                    ..AzSampleMeta::default()
                },
            }
        }

        let mut pool = AzExperiencePool::new(12);
        for update in 1..=4 {
            pool.add_games(vec![vec![
                sample(update, update as u64, 0),
                sample(update, update as u64, 1),
                sample(update, update as u64, 2),
            ]]);
        }
        let mut rng = SplitMix64::new(123);
        let batch = pool.sample_mixed_recent(10, 0.4, 2, &mut rng);

        assert_eq!(batch.samples.len(), 10);
        assert_eq!(batch.recent_samples, 4);
        assert_eq!(batch.full_window_samples, 6);
    }
}
