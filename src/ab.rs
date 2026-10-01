use std::io;
use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device, Shape, Var};
use candle_nn::VarMap;

mod alphabeta;
pub(crate) use alphabeta::{
    AbUciSearchResult, search_uci, search_uci_pikafish, search_uci_pikafish_float,
};
#[cfg(any(
    all(feature = "gpu-train", not(target_os = "macos")),
    all(target_os = "linux", not(target_env = "musl")),
    all(test, target_os = "macos"),
    target_os = "windows",
))]
#[cfg_attr(all(test, target_os = "macos"), allow(dead_code))]
mod candle_model;
mod dataloader;
mod fused_feature_pool;
mod optimizer;
pub mod pikafish_candle;
mod pikafish_sparse;
mod play;
mod replay;
mod start;
mod train;
mod train_gpu;
#[cfg(any(
    all(feature = "gpu-train", not(target_os = "macos")),
    all(target_os = "linux", not(target_env = "musl")),
    target_os = "windows",
))]
#[path = "ab/train_gpu_candle.rs"]
mod train_gpu_candle;

use crate::nnue::{
    AB_NNUE_INPUT_SIZE, V2_KING_BUCKETS, canonical_square, fill_sparse_features_ab,
    piece_absolute_feature_index,
};
use crate::version::MODEL_FORMAT_VERSION;
use crate::xiangqi::{
    BOARD_FILES, BOARD_RANKS, BOARD_SIZE, Color, Move, Piece, Position, color_index,
    piece_kind_index,
};

pub use alphabeta::search as alphabeta_search;
pub use alphabeta::search_pikafish_model;
pub use alphabeta::{AbCandidate, AbSearchControl, AbSearchLimits, AbSearchResult, cp_from_q};
pub use play::{
    AbArenaConfig, AbArenaReport, AbSelfplayData, AbTerminalStats, generate_selfplay_data,
    play_arena_games_from_positions, play_arena_games_from_snapshots,
};
pub use replay::{AbExperiencePool, AbReplaySampleBatch, AbReplayWindowStats, ReplaySampler};
pub use start::AbStartSnapshot;
pub use train::{
    train_samples, train_samples_weighted, train_samples_weighted_owned,
    train_samples_weighted_shared,
};

pub(super) const VALUE_HEAD_SIZE: usize = 48;
pub(super) const WDL_HEAD_SIZE: usize = 3;
/// Small, exact-history-derived signals.  These deliberately replace the old
/// high-dimensional history planes: rules stay in the environment, while the
/// network only gets enough context to recognize an approaching repetition.
pub const RULE_CONTEXT_SIZE: usize = 7;
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

macro_rules! ab_weight_tensors {
    ($visit:ident, $h:expr) => {
        $visit!(input_hidden, [AB_NNUE_INPUT_SIZE, $h]);
        $visit!(input_piece_hidden, [STRUCTURAL_PIECE_SIZE, $h]);
        $visit!(input_rank_hidden, [STRUCTURAL_RANK_SIZE, $h]);
        $visit!(input_file_hidden, [STRUCTURAL_FILE_SIZE, $h]);
        $visit!(input_king_piece_hidden, [STRUCTURAL_KING_PIECE_SIZE, $h]);
        $visit!(rule_context_hidden, [RULE_CONTEXT_SIZE, $h]);
        $visit!(hidden_bias, [$h]);
        $visit!(value_head_hidden, [VALUE_HEAD_SIZE, $h]);
        $visit!(value_head_bias, [VALUE_HEAD_SIZE]);
        $visit!(value_head_output, [WDL_HEAD_SIZE, VALUE_HEAD_SIZE]);
    };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AbNnueArch {
    pub hidden_size: usize,
}

impl AbNnueArch {
    pub const fn default_const() -> Self {
        Self { hidden_size: 256 }
    }

    pub const fn with_hidden_size(hidden_size: usize) -> Self {
        let mut arch = Self::default_const();
        arch.hidden_size = hidden_size;
        arch
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_size == 0 {
            return Err(format!("invalid hidden_size {}", self.hidden_size));
        }
        Ok(())
    }
}

impl Default for AbNnueArch {
    fn default() -> Self {
        Self::default_const()
    }
}
pub(super) struct AbEvalScratch {
    // NNUE 热路径复用特征存储，避免每个搜索节点分配并排序 Vec。
    features: Vec<usize>,
    hidden: Vec<f32>,
    value_head: Vec<f32>,
}

impl AbEvalScratch {
    pub(super) fn new(arch: AbNnueArch) -> Self {
        let hidden_size = arch.hidden_size;
        Self {
            features: Vec::with_capacity(48),
            hidden: vec![0.0; hidden_size],
            value_head: vec![0.0; VALUE_HEAD_SIZE],
        }
    }
}

/// 搜索节点使用的双视角 NNUE 累加器，不包含随每步老化的历史特征。
#[derive(Clone, Debug)]
pub(super) struct AbEvalAccumulator {
    hidden_sum: Vec<f32>,
}

impl AbEvalAccumulator {
    /// 在 make_move 前捕获两个视角的将帅桶；可避免为增量更新克隆整个棋盘。
    pub(super) fn buckets_for_position(position: &Position) -> [(usize, usize); 2] {
        [
            canonical_buckets_for_perspective(position, Color::Red),
            canonical_buckets_for_perspective(position, Color::Black),
        ]
    }

    pub(super) fn new(model: &AbNnue, position: &Position) -> Self {
        let mut accumulator = Self {
            hidden_sum: vec![0.0; model.hidden_size * 2],
        };
        accumulator.refresh(model, position);
        accumulator
    }

    fn refresh(&mut self, model: &AbNnue, position: &Position) {
        for perspective in [Color::Red, Color::Black] {
            let index = color_index(perspective);
            let start = index * model.hidden_size;
            Self::refresh_perspective(
                model,
                position,
                perspective,
                &mut self.hidden_sum[start..start + model.hidden_size],
            );
        }
    }

    fn refresh_perspective(
        model: &AbNnue,
        position: &Position,
        perspective: Color,
        hidden: &mut [f32],
    ) {
        let mut features = Vec::with_capacity(32);
        for sq in 0..BOARD_SIZE {
            if let Some(piece) = position.piece_at(sq) {
                let piece_index = piece_absolute_feature_index(perspective, piece);
                features.push(piece_index * BOARD_SIZE + canonical_square(perspective, sq));
            }
        }
        model.input_embedding_linear_into_slice(&features, hidden);
    }

    pub(super) fn apply_transition_from_buckets(
        model: &AbNnue,
        before_buckets: [(usize, usize); 2],
        after: &Position,
        mv: Move,
        moved: Piece,
        captured: Option<Piece>,
        hidden_sum: &mut [f32],
    ) {
        debug_assert_eq!(hidden_sum.len(), model.hidden_size * 2);
        for perspective in [Color::Red, Color::Black] {
            let start = color_index(perspective) * model.hidden_size;
            Self::apply_transition_for_perspective_from_buckets(
                model,
                before_buckets[color_index(perspective)],
                after,
                mv,
                moved,
                captured,
                perspective,
                &mut hidden_sum[start..start + model.hidden_size],
            );
        }
    }

    fn apply_transition_for_perspective_from_buckets(
        model: &AbNnue,
        before_buckets: (usize, usize),
        after: &Position,
        mv: Move,
        moved: Piece,
        captured: Option<Piece>,
        perspective: Color,
        hidden: &mut [f32],
    ) {
        let after_buckets = canonical_buckets_for_perspective(after, perspective);
        if before_buckets != after_buckets {
            // 将帅移动会改变所有棋子的王桶结构项，少见且必须完整刷新。
            Self::refresh_perspective(model, after, perspective, hidden);
            return;
        }
        add_canonical_piece_contribution(
            model,
            hidden,
            perspective,
            before_buckets,
            mv.from as usize,
            moved,
            -1.0,
        );
        if let Some(captured) = captured {
            add_canonical_piece_contribution(
                model,
                hidden,
                perspective,
                before_buckets,
                mv.to as usize,
                captured,
                -1.0,
            );
        }
        add_canonical_piece_contribution(
            model,
            hidden,
            perspective,
            after_buckets,
            mv.to as usize,
            moved,
            1.0,
        );
    }

    pub(super) fn hidden_for_slice(hidden_sum: &[f32], hidden_size: usize, side: Color) -> &[f32] {
        let start = color_index(side) * hidden_size;
        &hidden_sum[start..start + hidden_size]
    }

    pub(super) fn into_hidden_sum(self) -> Vec<f32> {
        self.hidden_sum
    }
}

fn canonical_buckets_for_perspective(position: &Position, perspective: Color) -> (usize, usize) {
    let us = position
        .general_square(perspective)
        .map(|sq| canonical_general_bucket(0, canonical_square_for(perspective, sq)))
        .unwrap_or(4);
    let them = position
        .general_square(perspective.opposite())
        .map(|sq| canonical_general_bucket(7, canonical_square_for(perspective, sq)))
        .unwrap_or(4);
    (us, them)
}

fn add_canonical_piece_contribution(
    model: &AbNnue,
    hidden: &mut [f32],
    perspective: Color,
    buckets: (usize, usize),
    sq: usize,
    piece: Piece,
    scale: f32,
) {
    let relative_color = if piece.color == perspective { 0 } else { 7 };
    let piece_index = relative_color + piece_kind_index(piece.kind);
    let relative_square = canonical_square_for(perspective, sq);
    let feature = piece_index * BOARD_SIZE + relative_square;
    let rank = relative_square / BOARD_FILES;
    let file = relative_square % BOARD_FILES;
    add_scaled_feature_row(
        hidden,
        &model.input_hidden,
        model.hidden_size,
        feature,
        scale,
    );
    add_scaled_feature_row(
        hidden,
        &model.input_piece_hidden,
        model.hidden_size,
        piece_index,
        scale,
    );
    add_scaled_feature_row(
        hidden,
        &model.input_rank_hidden,
        model.hidden_size,
        rank,
        scale,
    );
    add_scaled_feature_row(
        hidden,
        &model.input_file_hidden,
        model.hidden_size,
        file,
        scale,
    );
    add_scaled_feature_row(
        hidden,
        &model.input_king_piece_hidden,
        model.hidden_size,
        structural_king_piece_index(0, buckets.0, piece_index),
        scale,
    );
    add_scaled_feature_row(
        hidden,
        &model.input_king_piece_hidden,
        model.hidden_size,
        structural_king_piece_index(1, buckets.1, piece_index),
        scale,
    );
}

#[inline(always)]
fn canonical_square_for(perspective: Color, sq: usize) -> usize {
    if perspective == Color::Red {
        sq
    } else {
        BOARD_SIZE - 1 - sq
    }
}

#[derive(Debug)]
pub struct AbNnue {
    pub hidden_size: usize,
    pub arch: AbNnueArch,
    pub input_hidden: Vec<f32>,
    pub input_piece_hidden: Vec<f32>,
    pub input_rank_hidden: Vec<f32>,
    pub input_file_hidden: Vec<f32>,
    pub input_king_piece_hidden: Vec<f32>,
    pub rule_context_hidden: Vec<f32>,
    pub hidden_bias: Vec<f32>,
    pub value_head_hidden: Vec<f32>,
    pub value_head_bias: Vec<f32>,
    pub value_head_output: Vec<f32>,
    #[cfg_attr(not(feature = "gpu-train"), allow(dead_code))]
    gpu_trainer: Option<Box<train_gpu::GpuTrainer>>,
}

impl Clone for AbNnue {
    fn clone(&self) -> Self {
        Self {
            hidden_size: self.hidden_size,
            arch: self.arch,
            input_hidden: self.input_hidden.clone(),
            input_piece_hidden: self.input_piece_hidden.clone(),
            input_rank_hidden: self.input_rank_hidden.clone(),
            input_file_hidden: self.input_file_hidden.clone(),
            input_king_piece_hidden: self.input_king_piece_hidden.clone(),
            rule_context_hidden: self.rule_context_hidden.clone(),
            hidden_bias: self.hidden_bias.clone(),
            value_head_hidden: self.value_head_hidden.clone(),
            value_head_bias: self.value_head_bias.clone(),
            value_head_output: self.value_head_output.clone(),
            gpu_trainer: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AbEvolveConfig {
    pub games: usize,
    pub max_plies: usize,
    pub rule60_max_ply: Option<u16>,
    pub nodes: usize,
    pub seed: u64,
    pub workers: usize,
    pub generation_update: u32,
    pub temperature_start: f32,
    pub temperature_cutoff_plies: usize,
    pub temperature_endgame: f32,
    pub temperature_decay_delay_plies: usize,
    pub temperature_decay_plies: usize,
    pub opening_positions: Arc<[AbStartSnapshot]>,
    pub mirror_probability: f32,
    pub record_fens: bool,
}

#[derive(Clone, Debug, Default)]
pub struct AbEvolveReport {
    pub training_steps: usize,
    pub training_chunks: usize,
    pub test_chunks: usize,
    pub holdout_checks: Vec<AbHoldoutReport>,
    pub games: usize,
    pub samples: usize,
    pub avg_search_nodes: f32,
    pub red_wins: usize,
    pub black_wins: usize,
    pub draws: usize,
    pub avg_plies: f32,
    pub selfplay_start_source_rate: [f32; AbStartSource::COUNT],
    pub selfplay_start_phase_ply: [f32; AbStartSource::COUNT],
    pub selfplay_start_age: [f32; AbStartSource::COUNT],
    pub selfplay_start_age_max: [u32; AbStartSource::COUNT],
    pub selfplay_start_temperature: [f32; AbStartSource::COUNT],
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
    pub phase_value: [AbPhaseValueReport; 3],
    pub source_phase_value: [AbPhaseValueReport; 9],
    pub entropy_opening: f32,
    pub entropy_mid: f32,
    pub root_q_gap: f32,
    pub root_q_top1_abs: f32,
    pub root_actions: f32,
    pub opening_q_gap: f32,
    pub opening_q_top1_abs: f32,
    pub opening_root_actions: f32,
    pub sampled_best_rate: f32,
    pub avg_best_played_q_gap: f32,
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
pub struct AbPhaseValueReport {
    pub samples: usize,
    pub rmse: f32,
    pub corr: f32,
    pub calibration: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AbTrainBenchmark {
    pub loss: f32,
    pub value_loss: f32,
}

#[derive(Clone, Debug)]
pub struct AbTrainingSample {
    pub features: Vec<usize>,
    pub rule_context: [f32; RULE_CONTEXT_SIZE],
    pub value_wdl: [f32; WDL_HEAD_SIZE],
    pub root_search_wdl: [f32; WDL_HEAD_SIZE],
    pub value: f32,
    pub side_sign: f32,
    pub value_weight: f32,
    pub search_nodes: u32,
    pub meta: AbSampleMeta,
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
pub enum AbStartSource {
    #[default]
    Startpos = 0,
    OpeningBook = 1,
    Midgame = 2,
}

impl AbStartSource {
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
pub struct AbSampleMeta {
    pub generation_update: u32,
    pub game_id: u64,
    pub ply: u16,
    pub root_q: f32,
    pub best_q: f32,
    pub played_q: f32,
    pub best_index: u16,
    pub played_index: u16,
    pub start_source: AbStartSource,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AbTrainStats {
    /// Mean optimized objective after a completed training call, including all weights.
    pub loss: f32,
    pub value_loss: f32,
    pub value_pred_sum: f32,
    pub value_pred_sq_sum: f32,
    pub value_target_sum: f32,
    pub value_target_sq_sum: f32,
    pub value_pred_target_sum: f32,
    pub value_error_sq_sum: f32,
    pub samples: usize,
    pub phase_value: [AbValueMomentStats; 3],
    pub source_phase_value: [AbValueMomentStats; 9],
}

pub const HOLDOUT_INTERVAL_STEPS: usize = 2_000;

#[derive(Clone, Copy, Debug)]
pub struct AbHoldoutReport {
    pub step: usize,
    pub samples: usize,
    pub value_samples: usize,
    pub loss: f32,
    pub value_loss: f32,
    pub value_rmse: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AbValueMomentStats {
    pub pred_sum: f32,
    pub pred_sq_sum: f32,
    pub target_sum: f32,
    pub target_sq_sum: f32,
    pub pred_target_sum: f32,
    pub error_sq_sum: f32,
    pub samples: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct AbTrainLossWeights {
    pub value: f32,
}

impl Default for AbTrainLossWeights {
    fn default() -> Self {
        Self { value: 1.0 }
    }
}

impl AbTrainStats {
    #[cfg_attr(not(feature = "gpu-train"), allow(dead_code))]
    fn add_assign(&mut self, other: &Self) {
        self.loss += other.loss;
        self.value_loss += other.value_loss;
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

impl AbNnue {
    pub fn random_with_arch(arch: AbNnueArch, seed: u64) -> Self {
        if let Err(err) = arch.validate() {
            panic!("AbNnue::random_with_arch: invalid arch ({err})");
        }
        let hidden_size = arch.hidden_size;
        let mut rng = SplitMix64::new(seed);
        let input_hidden: Vec<f32> = (0..AB_NNUE_INPUT_SIZE * hidden_size)
            .map(|_| rng.weight(0.015))
            .collect();
        // Learned structural factors recover row/file/material/king context from
        // piece-square facts without reintroducing those handcrafted feature ids.
        let input_piece_hidden = vec![0.0; STRUCTURAL_PIECE_SIZE * hidden_size];
        let input_rank_hidden = vec![0.0; STRUCTURAL_RANK_SIZE * hidden_size];
        let input_file_hidden = vec![0.0; STRUCTURAL_FILE_SIZE * hidden_size];
        let input_king_piece_hidden = vec![0.0; STRUCTURAL_KING_PIECE_SIZE * hidden_size];
        // Start history-neutral; rule context is learned from self-play.
        let rule_context_hidden = vec![0.0; RULE_CONTEXT_SIZE * hidden_size];
        let hidden_bias = vec![0.0; hidden_size];
        // Start value-neutral. A random value head can evaluate startpos as a
        // large red/black advantage before any training, and search amplifies
        // that noise into the first self-play dataset.
        let value_head_hidden = (0..VALUE_HEAD_SIZE * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt() * 0.5))
            .collect();
        let value_head_bias = vec![0.0; VALUE_HEAD_SIZE];
        // Keep the value head output-neutral at initialization. This preserves
        // stable first self-play while giving value its own nonlinear capacity.
        let value_head_output = vec![0.0; WDL_HEAD_SIZE * VALUE_HEAD_SIZE];
        Self {
            hidden_size,
            arch,
            input_hidden,
            input_piece_hidden,
            input_rank_hidden,
            input_file_hidden,
            input_king_piece_hidden,
            rule_context_hidden,
            hidden_bias,
            value_head_hidden,
            value_head_bias,
            value_head_output,
            gpu_trainer: None,
        }
    }

    pub fn random(hidden_size: usize, seed: u64) -> Self {
        Self::random_with_arch(AbNnueArch::with_hidden_size(hidden_size), seed)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let h = self.hidden_size;
        let varmap = VarMap::new();
        insert_candle_var(
            &varmap,
            "ab_model_format_version",
            &[MODEL_FORMAT_VERSION],
            (1,),
        )?;
        macro_rules! save_tensor {
            ($field:ident, [$($dim:expr),+]) => {
                insert_candle_var(&varmap, stringify!($field), &self.$field, ($($dim),+))?;
            };
        }
        ab_weight_tensors!(save_tensor, h);
        varmap.save(path).map_err(candle_io_error)
    }

    pub fn save_training_state(
        &self,
        path: impl AsRef<Path>,
        next_update: usize,
    ) -> io::Result<bool> {
        let Some(trainer) = self.gpu_trainer.as_ref() else {
            return Ok(false);
        };
        trainer
            .save_state(path.as_ref(), next_update)
            .map_err(candle_io_error)?;
        Ok(true)
    }

    pub fn restore_training_state(
        &mut self,
        path: impl AsRef<Path>,
        next_update: usize,
        lr: f32,
    ) -> io::Result<()> {
        let mut trainer = train_gpu::GpuTrainer::new(self, lr).map_err(candle_io_error)?;
        trainer
            .restore_state(path.as_ref(), next_update)
            .map_err(candle_io_error)?;
        self.gpu_trainer = Some(Box::new(trainer));
        Ok(())
    }

    pub fn training_steps(&self) -> usize {
        self.gpu_trainer
            .as_ref()
            .map_or(0, |trainer| trainer.steps())
    }

    pub fn set_training_holdout(
        &mut self,
        samples: Vec<AbTrainingSample>,
        lr: f32,
    ) -> io::Result<()> {
        if self.gpu_trainer.is_none() {
            self.gpu_trainer = Some(Box::new(
                train_gpu::GpuTrainer::new(self, lr).map_err(candle_io_error)?,
            ));
        }
        self.gpu_trainer.as_mut().unwrap().set_holdout(samples);
        Ok(())
    }

    pub fn take_training_checks(&mut self) -> Vec<AbHoldoutReport> {
        self.gpu_trainer
            .as_mut()
            .map_or_else(Vec::new, |trainer| trainer.take_checks())
    }

    pub fn last_training_learning_rate(&self) -> Option<f32> {
        self.gpu_trainer
            .as_ref()
            .map(|trainer| trainer.last_learning_rate())
    }

    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let tensors = unsafe {
            candle_core::safetensors::MmapedSafetensors::new(path.as_ref())
                .map_err(candle_io_error)?
        };
        let mut expected_tensors = vec!["ab_model_format_version"];
        macro_rules! expect_tensor {
            ($field:ident, [$($dim:expr),+]) => {
                expected_tensors.push(stringify!($field));
            };
        }
        ab_weight_tensors!(expect_tensor, 0);
        if let Some((name, _)) = tensors
            .tensors()
            .iter()
            .find(|(name, _)| !expected_tensors.contains(&name.as_str()))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported AB model tensor `{name}`"),
            ));
        }
        let format_version = load_candle_f32_tensor(&tensors, "ab_model_format_version")?;
        let Some(&format_version) = format_version.first() else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "missing AB model format",
            ));
        };
        if format_version != MODEL_FORMAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported AB model format {:?}; expected v{}",
                    format_version, MODEL_FORMAT_VERSION
                ),
            ));
        }
        let hidden_bias = load_candle_f32_tensor(&tensors, "hidden_bias")?;
        let hidden_size = hidden_bias.len();
        let arch = AbNnueArch { hidden_size };
        let model = Self {
            hidden_size,
            arch,
            input_hidden: load_candle_f32_tensor(&tensors, "input_hidden")?,
            input_piece_hidden: load_candle_f32_tensor(&tensors, "input_piece_hidden")?,
            input_rank_hidden: load_candle_f32_tensor(&tensors, "input_rank_hidden")?,
            input_file_hidden: load_candle_f32_tensor(&tensors, "input_file_hidden")?,
            input_king_piece_hidden: load_candle_f32_tensor(&tensors, "input_king_piece_hidden")?,
            rule_context_hidden: load_candle_f32_tensor(&tensors, "rule_context_hidden")?,
            hidden_bias,
            value_head_hidden: load_candle_f32_tensor(&tensors, "value_head_hidden")?,
            value_head_bias: load_candle_f32_tensor(&tensors, "value_head_bias")?,
            value_head_output: load_candle_f32_tensor(&tensors, "value_head_output")?,
            gpu_trainer: None,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn evaluate_value(&self, position: &Position) -> f32 {
        let mut scratch = AbEvalScratch::new(self.arch);
        self.evaluate_value_only_with_scratch(position, &[0.0; RULE_CONTEXT_SIZE], &mut scratch)
    }

    pub fn evaluate_value_with_rules(
        &self,
        position: &Position,
        history: &[crate::xiangqi::RuleHistoryEntry],
    ) -> f32 {
        let mut scratch = AbEvalScratch::new(self.arch);
        self.evaluate_value_only_with_scratch(
            position,
            &rule_context_features(position, history),
            &mut scratch,
        )
    }

    /// 搜索叶节点的增量价值评估。
    pub(super) fn evaluate_incremental_value_with_rules(
        &self,
        position: &Position,
        history: &[crate::xiangqi::RuleHistoryEntry],
        accumulator_hidden: &[f32],
        scratch: &mut AbEvalScratch,
    ) -> f32 {
        let hidden = if accumulator_hidden.len() == self.hidden_size {
            accumulator_hidden
        } else {
            AbEvalAccumulator::hidden_for_slice(
                accumulator_hidden,
                self.hidden_size,
                position.side_to_move(),
            )
        };
        scratch.hidden.resize(self.hidden_size, 0.0);
        scratch.hidden.copy_from_slice(hidden);
        self.add_rule_context_to_hidden(
            &rule_context_features(position, history),
            &mut scratch.hidden,
        );
        relu_in_place(&mut scratch.hidden);
        rms_norm_in_place(&mut scratch.hidden);
        self.value_wdl_from_hidden_into(&scratch.hidden, &mut scratch.value_head)
            .1
    }

    fn evaluate_value_only_with_scratch(
        &self,
        position: &Position,
        rule_context: &[f32; RULE_CONTEXT_SIZE],
        scratch: &mut AbEvalScratch,
    ) -> f32 {
        fill_sparse_features_ab(position, &mut scratch.features);
        self.input_embedding_linear_into(&scratch.features, &mut scratch.hidden);
        self.add_rule_context_to_hidden(rule_context, &mut scratch.hidden);
        relu_in_place(&mut scratch.hidden);
        rms_norm_in_place(&mut scratch.hidden);
        self.value_wdl_from_hidden_into(&scratch.hidden, &mut scratch.value_head)
            .1
    }

    pub fn evaluate_wdl_with_rules(
        &self,
        position: &Position,
        history: &[crate::xiangqi::RuleHistoryEntry],
    ) -> [f32; WDL_HEAD_SIZE] {
        let mut scratch = AbEvalScratch::new(self.arch);
        self.evaluate_with_scratch_output(
            position,
            &rule_context_features(position, history),
            &mut scratch,
        )
    }

    pub(super) fn evaluate_with_scratch_output(
        &self,
        position: &Position,
        rule_context: &[f32; RULE_CONTEXT_SIZE],
        scratch: &mut AbEvalScratch,
    ) -> [f32; WDL_HEAD_SIZE] {
        crate::scope_profile!("ab.evaluate_with_scratch");
        let mut features = std::mem::take(&mut scratch.features);
        {
            crate::scope_profile!("ab.eval.extract_features");
            fill_sparse_features_ab(position, &mut features);
        }
        {
            crate::scope_profile!("ab.eval.input_embedding");
            self.input_embedding_linear_into(&features, &mut scratch.hidden);
            self.add_rule_context_to_hidden(rule_context, &mut scratch.hidden);
        }
        {
            crate::scope_profile!("ab.eval.activation_norm");
            relu_in_place(&mut scratch.hidden);
            rms_norm_in_place(&mut scratch.hidden);
        }
        let (value_wdl, _) = {
            crate::scope_profile!("ab.eval.value_head");
            self.value_wdl_from_hidden_into(&scratch.hidden, &mut scratch.value_head)
        };
        scratch.features = features;
        value_wdl
    }

    #[inline]
    fn add_rule_context_to_hidden(
        &self,
        rule_context: &[f32; RULE_CONTEXT_SIZE],
        hidden: &mut [f32],
    ) {
        for (feature, &value) in rule_context.iter().enumerate() {
            if value == 0.0 {
                continue;
            }
            let row = &self.rule_context_hidden
                [feature * self.hidden_size..(feature + 1) * self.hidden_size];
            for (target, &weight) in hidden.iter_mut().zip(row) {
                *target += value * weight;
            }
        }
    }

    fn add_factorized_structure_into(&self, features: &[usize], hidden: &mut [f32]) {
        let mut us_king_bucket = 4;
        let mut them_king_bucket = 4;
        let mut structural_features = [StructuralPieceSquare {
            piece_index: 0,
            rank: 0,
            file: 0,
        }; BOARD_SIZE];
        let mut structural_count = 0usize;
        for &feature in features {
            let Some(structural) = decode_current_piece_square_feature(feature) else {
                continue;
            };
            let sq = feature % BOARD_SIZE;
            match structural.piece_index {
                0 => us_king_bucket = canonical_general_bucket(structural.piece_index, sq),
                7 => them_king_bucket = canonical_general_bucket(structural.piece_index, sq),
                _ => {}
            }
            structural_features[structural_count] = structural;
            structural_count += 1;
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if self.hidden_size >= 64 && std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: runtime detection above guarantees AVX2 support.
                unsafe {
                    self.add_factorized_structure_avx2(
                        &structural_features[..structural_count],
                        us_king_bucket,
                        them_king_bucket,
                        hidden,
                    );
                }
                return;
            }
        }

        for &structural in &structural_features[..structural_count] {
            add_scaled_feature_row(
                hidden,
                &self.input_piece_hidden,
                self.hidden_size,
                structural.piece_index,
                1.0,
            );
            add_scaled_feature_row(
                hidden,
                &self.input_rank_hidden,
                self.hidden_size,
                structural.rank,
                1.0,
            );
            add_scaled_feature_row(
                hidden,
                &self.input_file_hidden,
                self.hidden_size,
                structural.file,
                1.0,
            );
            add_scaled_feature_row(
                hidden,
                &self.input_king_piece_hidden,
                self.hidden_size,
                structural_king_piece_index(0, us_king_bucket, structural.piece_index),
                1.0,
            );
            add_scaled_feature_row(
                hidden,
                &self.input_king_piece_hidden,
                self.hidden_size,
                structural_king_piece_index(1, them_king_bucket, structural.piece_index),
                1.0,
            );
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe fn add_factorized_structure_avx2(
        &self,
        structural_features: &[StructuralPieceSquare],
        us_king_bucket: usize,
        them_king_bucket: usize,
        hidden: &mut [f32],
    ) {
        for &structural in structural_features {
            unsafe {
                add_feature_row_avx2(
                    hidden,
                    feature_row(
                        &self.input_piece_hidden,
                        self.hidden_size,
                        structural.piece_index,
                    ),
                );
                add_feature_row_avx2(
                    hidden,
                    feature_row(&self.input_rank_hidden, self.hidden_size, structural.rank),
                );
                add_feature_row_avx2(
                    hidden,
                    feature_row(&self.input_file_hidden, self.hidden_size, structural.file),
                );
                add_feature_row_avx2(
                    hidden,
                    feature_row(
                        &self.input_king_piece_hidden,
                        self.hidden_size,
                        structural_king_piece_index(0, us_king_bucket, structural.piece_index),
                    ),
                );
                add_feature_row_avx2(
                    hidden,
                    feature_row(
                        &self.input_king_piece_hidden,
                        self.hidden_size,
                        structural_king_piece_index(1, them_king_bucket, structural.piece_index),
                    ),
                );
            }
        }
    }

    fn input_embedding_linear_into(&self, features: &[usize], hidden: &mut Vec<f32>) {
        hidden.resize(self.hidden_size, 0.0);
        self.input_embedding_linear_into_slice(features, hidden);
    }

    fn input_embedding_linear_into_slice(&self, features: &[usize], hidden: &mut [f32]) {
        debug_assert_eq!(hidden.len(), self.hidden_size);
        hidden.copy_from_slice(&self.hidden_bias);
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if self.hidden_size >= 64 && std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: runtime detection above guarantees AVX2 support.
                unsafe {
                    input_embedding_add_features_avx2(
                        &self.input_hidden,
                        self.hidden_size,
                        features,
                        hidden,
                    );
                }
                self.add_factorized_structure_into(features, hidden);
                return;
            }
        }
        for &feature in features {
            let row =
                &self.input_hidden[feature * self.hidden_size..(feature + 1) * self.hidden_size];
            for (left, &right) in hidden.iter_mut().zip(row) {
                *left += right;
            }
        }
        self.add_factorized_structure_into(features, hidden);
    }

    fn value_wdl_from_hidden_into(
        &self,
        hidden: &[f32],
        value_head: &mut Vec<f32>,
    ) -> ([f32; WDL_HEAD_SIZE], f32) {
        value_head.resize(VALUE_HEAD_SIZE, 0.0);
        value_head.copy_from_slice(&self.value_head_bias);
        for (feature, value) in value_head.iter_mut().enumerate().take(VALUE_HEAD_SIZE) {
            let hidden_row = &self.value_head_hidden
                [feature * self.hidden_size..(feature + 1) * self.hidden_size];
            *value += dot_product(hidden, hidden_row);
            *value = (*value).max(0.0);
        }
        let mut logits = [0.0f32; WDL_HEAD_SIZE];
        for (out, logit) in logits.iter_mut().enumerate() {
            let row = &self.value_head_output[out * VALUE_HEAD_SIZE..(out + 1) * VALUE_HEAD_SIZE];
            *logit = dot_product(value_head, row);
        }
        let wdl = softmax_fixed3(logits);
        let q = wdl[0] - wdl[2];
        (wdl, q)
    }

    fn validate(&self) -> io::Result<()> {
        let arch = &self.arch;
        if arch.hidden_size != self.hidden_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "AB NNUE arch.hidden_size does not match the cached hidden_size field",
            ));
        }
        if let Err(err) = arch.validate() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("AB NNUE arch invalid: {err}"),
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
                            "AB model tensor `{}` length mismatch: got {}, expected {}",
                            stringify!($field),
                            self.$field.len(),
                            expected
                        ),
                    ));
                }
            };
        }
        ab_weight_tensors!(validate_tensor, hidden);
        Ok(())
    }
}

pub fn benchmark_training(
    model: &mut AbNnue,
    sample_count: usize,
    epochs: usize,
    batch_size: usize,
    lr: f32,
    seed: u64,
) -> AbTrainBenchmark {
    let mut rng = SplitMix64::new(seed);
    let mut samples = Vec::with_capacity(sample_count);
    for index in 0..sample_count {
        let feature_count = 24 + (rng.next_u64() as usize % 16);
        let mut features = Vec::with_capacity(feature_count);
        for _ in 0..feature_count {
            features.push((rng.next_u64() as usize) % AB_NNUE_INPUT_SIZE);
        }
        features.sort_unstable();
        features.dedup();

        let value = rng.unit_f32() * 2.0 - 1.0;
        samples.push(AbTrainingSample {
            features,
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            value_wdl: scalar_value_to_wdl_target(value),
            root_search_wdl: scalar_value_to_wdl_target(value),
            value,
            side_sign: 1.0,
            value_weight: 1.0,
            search_nodes: 0,
            meta: AbSampleMeta::default(),
        });
        if index + 1 == sample_count {
            break;
        }
    }
    let stats = train_samples(model, &samples, epochs, lr, batch_size, &mut rng)
        .unwrap_or_else(|err| panic!("training failed: {err}"));
    AbTrainBenchmark {
        loss: stats.loss,
        value_loss: stats.value_loss,
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn feature_row(input_hidden: &[f32], hidden_size: usize, feature: usize) -> &[f32] {
    &input_hidden[feature * hidden_size..(feature + 1) * hidden_size]
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2(left: &[f32], right: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn add_feature_row_avx2(hidden: &mut [f32], row: &[f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let chunks = hidden.len() / 8;
    for chunk in 0..chunks {
        let index = chunk * 8;
        unsafe {
            let left = _mm256_loadu_ps(hidden.as_ptr().add(index));
            let right = _mm256_loadu_ps(row.as_ptr().add(index));
            _mm256_storeu_ps(hidden.as_mut_ptr().add(index), _mm256_add_ps(left, right));
        }
    }
    for index in (chunks * 8)..hidden.len() {
        hidden[index] += row[index];
    }
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn add_scaled_feature_row_avx2(hidden: &mut [f32], row: &[f32], scale: f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn input_embedding_add_features_avx2(
    input_hidden: &[f32],
    hidden_size: usize,
    features: &[usize],
    hidden: &mut [f32],
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let chunks = hidden_size / 8;
    for &feature in features {
        let row = &input_hidden[feature * hidden_size..(feature + 1) * hidden_size];
        for chunk in 0..chunks {
            let index = chunk * 8;
            unsafe {
                let left = _mm256_loadu_ps(hidden.as_ptr().add(index));
                let right = _mm256_loadu_ps(row.as_ptr().add(index));
                _mm256_storeu_ps(hidden.as_mut_ptr().add(index), _mm256_add_ps(left, right));
            }
        }
        for index in (chunks * 8)..hidden_size {
            hidden[index] += row[index];
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn relu_in_place_avx2(values: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_model_roundtrip_contains_only_value_tensors() {
        let model = AbNnue::random(32, 42);
        let path = std::env::current_dir()
            .unwrap()
            .join("target")
            .join("fast")
            .join(format!("ab-value-model-{}.safetensors", std::process::id()));
        model.save(&path).unwrap();
        let loaded = AbNnue::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(model.input_hidden, loaded.input_hidden);
        assert_eq!(model.value_head_output, loaded.value_head_output);
        let position = Position::startpos();
        assert_eq!(
            model.evaluate_value(&position),
            loaded.evaluate_value(&position)
        );
    }

    #[test]
    fn incremental_value_matches_full_value() {
        let model = AbNnue::random(64, 7);
        let position = Position::startpos();
        let history = Vec::new();
        let hidden = AbEvalAccumulator::new(&model, &position).into_hidden_sum();
        let mut scratch = AbEvalScratch::new(model.arch);
        let incremental =
            model.evaluate_incremental_value_with_rules(&position, &history, &hidden, &mut scratch);
        let full = model.evaluate_value_with_rules(&position, &history);
        assert!((incremental - full).abs() < 1.0e-6);
    }

    #[test]
    fn incremental_accumulator_tracks_legal_game() {
        let model = AbNnue::random(64, 17);
        let mut position = Position::startpos();
        let mut hidden = AbEvalAccumulator::new(&model, &position).into_hidden_sum();
        let mut scratch = AbEvalScratch::new(model.arch);
        let mut seed = 0x9e3779b97f4a7c15u64;
        for _ in 0..80 {
            let moves = position.legal_moves();
            if moves.is_empty() {
                break;
            }
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let mv = moves[seed as usize % moves.len()];
            let moved = position.piece_at(mv.from as usize).unwrap();
            let captured = position.piece_at(mv.to as usize);
            let buckets = AbEvalAccumulator::buckets_for_position(&position);
            position.make_move(mv);
            AbEvalAccumulator::apply_transition_from_buckets(
                &model,
                buckets,
                &position,
                mv,
                moved,
                captured,
                &mut hidden,
            );
            let refreshed = AbEvalAccumulator::new(&model, &position).into_hidden_sum();
            for (&incremental, &full) in hidden.iter().zip(&refreshed) {
                assert!((incremental - full).abs() < 1.0e-4);
            }
            let incremental =
                model.evaluate_incremental_value_with_rules(&position, &[], &hidden, &mut scratch);
            let full = model.evaluate_value_with_rules(&position, &[]);
            assert!((incremental - full).abs() < 1.0e-5);
        }
    }
}
