use std::io;
use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device, Shape, Var};
use candle_nn::VarMap;

mod alphazero;
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
mod fused_policy;
mod fused_sparse_policy;
mod midgame;
mod play;
mod replay;
mod train;
mod train_gpu;
#[cfg(any(
    all(feature = "gpu-train", not(target_os = "macos")),
    all(target_os = "linux", not(target_env = "musl")),
    target_os = "windows",
))]
#[path = "az/train_gpu_candle.rs"]
mod train_gpu_candle;

use crate::nnue::{
    AZ_NNUE_INPUT_SIZE, V2_KING_BUCKETS, canonical_move, canonical_square, fill_sparse_features_az,
    piece_absolute_feature_index,
};
use crate::version::MODEL_FORMAT_VERSION;
use crate::xiangqi::{
    BOARD_FILES, BOARD_RANKS, BOARD_SIZE, Color, Move, Piece, Position, color_index,
    piece_kind_index,
};

pub use alphazero::{
    AzBatchSearchInput, AzCandidate, AzSearchControl, AzSearchLimits, AzSearchResult,
    AzSearchTraceStep, alphazero_search, alphazero_search_batch4,
    alphazero_search_external_root_controlled_with_progress, alphazero_search_trace_with_rules,
    alphazero_search_with_rules, alphazero_search_with_rules_controlled,
    alphazero_search_with_rules_controlled_with_progress, cp_from_q,
};
pub use midgame::{AzMidgamePool, AzStartSnapshot};
pub use play::{
    AzArenaConfig, AzArenaReport, AzSelfplayData, AzTerminalStats, generate_selfplay_data,
    play_arena_games_from_positions,
};
pub use replay::{AzExperiencePool, AzReplaySampleBatch, AzReplayWindowStats};
pub use train::{train_samples, train_samples_weighted, train_samples_weighted_owned};

const SPARSE_MOVE_SPACE: usize = BOARD_SIZE * BOARD_SIZE;
pub const DENSE_MOVE_SPACE: usize = compute_dense_move_count();
pub(super) const POLICY_CONSEQUENCE_SIZE: usize = 32;
pub(super) const POLICY_MOVE_CONTEXT_SIZE: usize = 16;
pub(super) const POLICY_THREAT_CONTEXT_SIZE: usize = 16;
pub(super) const POLICY_ACCUMULATOR_RANK: usize = 32;
pub(super) const POLICY_TACTICAL_SIGNATURE_BUCKETS: usize = 64;
pub(super) const POLICY_TACTICAL_EXACT_SIZE: usize =
    DENSE_MOVE_SPACE * (STRUCTURAL_PIECE_SIZE / 2) * POLICY_TACTICAL_SIGNATURE_BUCKETS;
pub(super) const POLICY_TACTICAL_FACTOR_SIZE: usize =
    (STRUCTURAL_PIECE_SIZE / 2) * POLICY_TACTICAL_SIGNATURE_BUCKETS;
pub(super) const POLICY_TACTICAL_SIZE: usize =
    POLICY_TACTICAL_EXACT_SIZE + POLICY_TACTICAL_FACTOR_SIZE;
pub(super) const POLICY_SPARSE_CAPTURE_CLASSES: usize = STRUCTURAL_PIECE_SIZE + 1;
pub(super) const POLICY_SPARSE_MAIN_SIZE: usize =
    DENSE_MOVE_SPACE * STRUCTURAL_PIECE_SIZE * V2_KING_BUCKETS * V2_KING_BUCKETS;
pub(super) const POLICY_SPARSE_CAPTURE_SIZE: usize =
    DENSE_MOVE_SPACE * POLICY_SPARSE_CAPTURE_CLASSES;
pub(super) const POLICY_SPARSE_TABLE_SIZE: usize =
    POLICY_SPARSE_MAIN_SIZE + POLICY_SPARSE_CAPTURE_SIZE + 1;
pub(super) const POLICY_SPARSE_MOVE_PIECE_SIZE: usize = DENSE_MOVE_SPACE * STRUCTURAL_PIECE_SIZE;
pub(super) const POLICY_SPARSE_MOVE_KING_SIZE: usize =
    DENSE_MOVE_SPACE * V2_KING_BUCKETS * V2_KING_BUCKETS;
pub(super) const POLICY_SPARSE_PIECE_KING_SIZE: usize =
    STRUCTURAL_PIECE_SIZE * V2_KING_BUCKETS * V2_KING_BUCKETS;
pub(super) const POLICY_KING_DISTANCE_BUCKETS: usize = 6;
pub(super) const POLICY_KING_APPROACH_BUCKETS: usize = 5;
pub(super) const POLICY_SPARSE_DISTANCE_SIZE: usize =
    STRUCTURAL_PIECE_SIZE * POLICY_KING_DISTANCE_BUCKETS;
pub(super) const POLICY_SPARSE_APPROACH_SIZE: usize =
    STRUCTURAL_PIECE_SIZE * POLICY_KING_APPROACH_BUCKETS;
pub(super) const POLICY_SPARSE_FACTOR_SIZE: usize = POLICY_SPARSE_MOVE_PIECE_SIZE
    + POLICY_SPARSE_MOVE_KING_SIZE
    + POLICY_SPARSE_PIECE_KING_SIZE
    + POLICY_SPARSE_DISTANCE_SIZE
    + POLICY_SPARSE_APPROACH_SIZE;
const POLICY_ACCUMULATOR_PIECE_OFFSET: usize = AZ_NNUE_INPUT_SIZE;
const POLICY_ACCUMULATOR_RANK_OFFSET: usize =
    POLICY_ACCUMULATOR_PIECE_OFFSET + STRUCTURAL_PIECE_SIZE;
const POLICY_ACCUMULATOR_FILE_OFFSET: usize = POLICY_ACCUMULATOR_RANK_OFFSET + STRUCTURAL_RANK_SIZE;
const POLICY_ACCUMULATOR_KING_PIECE_OFFSET: usize =
    POLICY_ACCUMULATOR_FILE_OFFSET + STRUCTURAL_FILE_SIZE;
const POLICY_ACCUMULATOR_BIAS_ROW: usize =
    POLICY_ACCUMULATOR_KING_PIECE_OFFSET + STRUCTURAL_KING_PIECE_SIZE;
const POLICY_ACCUMULATOR_QUANT_ROWS: usize = POLICY_ACCUMULATOR_BIAS_ROW + 1;
pub(super) const VALUE_HEAD_SIZE: usize = 96;
pub(super) const VALUE_THREAT_RANK: usize = 64;
pub(super) const VALUE_THREAT_VOCAB: usize = 57_702;
pub(super) const VALUE_THREAT_MAX_ACTIVE: usize = 96;
/// 自对弈 WDL TD(λ) 的默认迹衰减系数。
pub const DEFAULT_VALUE_TD_LAMBDA: f32 = 1.0;
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

macro_rules! az_weight_tensors {
    ($visit:ident, $h:expr) => {
        $visit!(input_hidden, [AZ_NNUE_INPUT_SIZE, $h]);
        $visit!(input_piece_hidden, [STRUCTURAL_PIECE_SIZE, $h]);
        $visit!(input_rank_hidden, [STRUCTURAL_RANK_SIZE, $h]);
        $visit!(input_file_hidden, [STRUCTURAL_FILE_SIZE, $h]);
        $visit!(input_king_piece_hidden, [STRUCTURAL_KING_PIECE_SIZE, $h]);
        $visit!(rule_context_hidden, [RULE_CONTEXT_SIZE, $h]);
        $visit!(hidden_bias, [$h]);
        $visit!(value_head_hidden, [VALUE_HEAD_SIZE, $h]);
        $visit!(value_head_bias, [VALUE_HEAD_SIZE]);
        $visit!(value_head_output, [WDL_HEAD_SIZE, VALUE_HEAD_SIZE]);
        $visit!(
            value_threat_embedding,
            [VALUE_THREAT_VOCAB, VALUE_THREAT_RANK]
        );
        $visit!(value_threat_output, [WDL_HEAD_SIZE, VALUE_THREAT_RANK * 2]);
        $visit!(
            policy_threat_context,
            [POLICY_THREAT_CONTEXT_SIZE, VALUE_THREAT_RANK * 2]
        );
        $visit!(policy_move_bias, [DENSE_MOVE_SPACE]);
        $visit!(policy_consequence_output, [POLICY_CONSEQUENCE_SIZE]);
        $visit!(policy_context_hidden, [POLICY_MOVE_CONTEXT_SIZE, $h]);
        $visit!(
            policy_move_context,
            [DENSE_MOVE_SPACE, POLICY_MOVE_CONTEXT_SIZE]
        );
        $visit!(policy_accumulator_hidden, [POLICY_ACCUMULATOR_RANK, $h]);
        $visit!(
            policy_accumulator_move,
            [DENSE_MOVE_SPACE, POLICY_ACCUMULATOR_RANK]
        );
        $visit!(policy_sparse_table, [POLICY_SPARSE_TABLE_SIZE]);
        $visit!(policy_sparse_factor, [POLICY_SPARSE_FACTOR_SIZE]);
        $visit!(policy_tactical, [POLICY_TACTICAL_SIZE]);
    };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AzNnueArch {
    pub hidden_size: usize,
}

impl AzNnueArch {
    pub const fn default_const() -> Self {
        Self { hidden_size: 128 }
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

impl Default for AzNnueArch {
    fn default() -> Self {
        Self::default_const()
    }
}
pub(super) struct AzEvalScratch {
    // NNUE 热路径复用特征存储，避免每个 MCTS 叶节点分配并排序 Vec。
    features: Vec<usize>,
    hidden: Vec<f32>,
    policy_context: Vec<f32>,
    policy_accumulator_context: [i16; POLICY_ACCUMULATOR_RANK],
    policy_piece_square_scores: Vec<f32>,
    value_head: Vec<f32>,
    value_threat_accumulator: Vec<i16>,
    value_threat_activation: Vec<f32>,
    policy_gives_check: Vec<f32>,
    logits: Vec<f32>,
    priors: Vec<f32>,
}

#[allow(dead_code)]
pub(super) struct AzIncrementalEvalRequest<'a> {
    pub position: &'a Position,
    pub accumulator_hidden: &'a [f32],
    pub policy_accumulator: &'a [i16; POLICY_ACCUMULATOR_RANK],
    pub moves: &'a [Move],
    pub rule_context: &'a [f32; RULE_CONTEXT_SIZE],
    pub scratch: &'a mut AzEvalScratch,
}

impl AzEvalScratch {
    pub(super) fn new(arch: AzNnueArch) -> Self {
        let hidden_size = arch.hidden_size;
        Self {
            features: Vec::with_capacity(48),
            hidden: vec![0.0; hidden_size],
            policy_context: vec![0.0; POLICY_MOVE_CONTEXT_SIZE],
            policy_accumulator_context: [0; POLICY_ACCUMULATOR_RANK],
            policy_piece_square_scores: Vec::new(),
            value_head: vec![0.0; VALUE_HEAD_SIZE],
            value_threat_accumulator: vec![0; VALUE_THREAT_RANK],
            value_threat_activation: vec![0.0; VALUE_THREAT_RANK * 2],
            policy_gives_check: Vec::with_capacity(192),
            logits: Vec::with_capacity(192),
            priors: Vec::with_capacity(192),
        }
    }

    pub(super) fn empty() -> Self {
        Self {
            features: Vec::new(),
            hidden: Vec::new(),
            policy_context: Vec::new(),
            policy_accumulator_context: [0; POLICY_ACCUMULATOR_RANK],
            policy_piece_square_scores: Vec::new(),
            value_head: Vec::new(),
            value_threat_accumulator: Vec::new(),
            value_threat_activation: Vec::new(),
            policy_gives_check: Vec::new(),
            logits: Vec::new(),
            priors: Vec::new(),
        }
    }
}

/// 搜索节点使用的双视角 NNUE 累加器，不包含随每步老化的历史特征。
#[derive(Clone, Debug)]
pub(super) struct AzEvalAccumulator {
    hidden_sum: Vec<f32>,
}

impl AzEvalAccumulator {
    pub(super) fn new(model: &AzNnue, position: &Position) -> Self {
        let mut accumulator = Self {
            hidden_sum: vec![0.0; model.hidden_size * 2],
        };
        accumulator.refresh(model, position);
        accumulator
    }

    fn refresh(&mut self, model: &AzNnue, position: &Position) {
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
        model: &AzNnue,
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

    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn apply_transition_to_hidden(
        model: &AzNnue,
        before: &Position,
        after: &Position,
        mv: Move,
        moved: Piece,
        captured: Option<Piece>,
        hidden_sum: &mut [f32],
    ) {
        debug_assert_eq!(hidden_sum.len(), model.hidden_size * 2);
        for perspective in [Color::Red, Color::Black] {
            let start = color_index(perspective) * model.hidden_size;
            Self::apply_transition_for_perspective(
                model,
                before,
                after,
                mv,
                moved,
                captured,
                perspective,
                &mut hidden_sum[start..start + model.hidden_size],
            );
        }
    }

    pub(super) fn apply_transition_for_perspective(
        model: &AzNnue,
        before: &Position,
        after: &Position,
        mv: Move,
        moved: Piece,
        captured: Option<Piece>,
        perspective: Color,
        hidden: &mut [f32],
    ) {
        let before_buckets = canonical_buckets_for_perspective(before, perspective);
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
    model: &AzNnue,
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

fn threat_relation_map() -> &'static [u32] {
    use std::sync::OnceLock;
    static MAP: OnceLock<Vec<u32>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut map = vec![u32::MAX; STRUCTURAL_PIECE_SIZE * BOARD_SIZE * BOARD_SIZE];
        let mut next = 0u32;
        for attacker in 0..STRUCTURAL_PIECE_SIZE {
            let ours = attacker < 7;
            let kind = attacker % 7;
            for source in 0..BOARD_SIZE {
                if !threat_reachable_square(attacker, source) {
                    continue;
                }
                let rank = source / BOARD_FILES;
                let file = source % BOARD_FILES;
                let mut targets = Vec::with_capacity(18);
                if kind == 4 || kind == 5 {
                    targets.extend((0..BOARD_SIZE).filter(|&target| {
                        target != source
                            && (target / BOARD_FILES == rank || target % BOARD_FILES == file)
                    }));
                } else {
                    let steps: &[(isize, isize)] = match kind {
                        0 => &[(0, -1), (0, 1), (-1, 0), (1, 0)],
                        1 => &[(-1, -1), (1, -1), (-1, 1), (1, 1)],
                        2 => &[(-2, -2), (2, -2), (-2, 2), (2, 2)],
                        3 => &[
                            (-1, -2),
                            (1, -2),
                            (-1, 2),
                            (1, 2),
                            (-2, -1),
                            (-2, 1),
                            (2, -1),
                            (2, 1),
                        ],
                        6 if ours && rank <= 4 => &[(0, -1), (-1, 0), (1, 0)],
                        6 if !ours && rank >= 5 => &[(0, 1), (-1, 0), (1, 0)],
                        6 if ours => &[(0, -1)],
                        6 => &[(0, 1)],
                        _ => unreachable!(),
                    };
                    for &(df, dr) in steps {
                        let target_file = file as isize + df;
                        let target_rank = rank as isize + dr;
                        if !(0..BOARD_FILES as isize).contains(&target_file)
                            || !(0..BOARD_RANKS as isize).contains(&target_rank)
                        {
                            continue;
                        }
                        if (kind == 0 || kind == 1)
                            && (!(3..=5).contains(&(target_file as usize))
                                || if ours {
                                    target_rank < 7
                                } else {
                                    target_rank > 2
                                })
                        {
                            continue;
                        }
                        if kind == 2
                            && if ours {
                                target_rank < 5
                            } else {
                                target_rank > 4
                            }
                        {
                            continue;
                        }
                        targets.push(target_rank as usize * BOARD_FILES + target_file as usize);
                    }
                }
                for target in targets {
                    let relation = (attacker * BOARD_SIZE + source) * BOARD_SIZE + target;
                    map[relation] = next;
                    next += (0..STRUCTURAL_PIECE_SIZE)
                        .filter(|&piece| threat_reachable_square(piece, target))
                        .count() as u32;
                }
            }
        }
        assert_eq!(next as usize, VALUE_THREAT_VOCAB);
        map
    })
}

fn threat_reachable_square(piece: usize, square: usize) -> bool {
    let rank = square / BOARD_FILES;
    let file = square % BOARD_FILES;
    let ours = piece < 7;
    match piece % 7 {
        0 => (3..=5).contains(&file) && if ours { rank >= 7 } else { rank <= 2 },
        1 => {
            if ours {
                matches!((rank, file), (7, 3) | (7, 5) | (8, 4) | (9, 3) | (9, 5))
            } else {
                matches!((rank, file), (0, 3) | (0, 5) | (1, 4) | (2, 3) | (2, 5))
            }
        }
        2 => {
            if ours {
                matches!(
                    (rank, file),
                    (5, 2) | (5, 6) | (7, 0) | (7, 4) | (7, 8) | (9, 2) | (9, 6)
                )
            } else {
                matches!(
                    (rank, file),
                    (0, 2) | (0, 6) | (2, 0) | (2, 4) | (2, 8) | (4, 2) | (4, 6)
                )
            }
        }
        6 => {
            if ours {
                rank <= 4 || (rank == 5 || rank == 6) && file.is_multiple_of(2)
            } else {
                rank >= 5 || (rank == 3 || rank == 4) && file.is_multiple_of(2)
            }
        }
        _ => true,
    }
}

fn threat_attacked_offsets() -> &'static [u8] {
    use std::sync::OnceLock;
    static OFFSETS: OnceLock<Vec<u8>> = OnceLock::new();
    OFFSETS.get_or_init(|| {
        let mut offsets = vec![u8::MAX; BOARD_SIZE * STRUCTURAL_PIECE_SIZE];
        for square in 0..BOARD_SIZE {
            let mut next = 0u8;
            for piece in 0..STRUCTURAL_PIECE_SIZE {
                if threat_reachable_square(piece, square) {
                    offsets[square * STRUCTURAL_PIECE_SIZE + piece] = next;
                    next += 1;
                }
            }
        }
        offsets
    })
}

#[inline]
fn value_threat_index(
    perspective: Color,
    source: usize,
    attacker: Piece,
    target: usize,
    attacked: Piece,
) -> usize {
    let attacker =
        (if attacker.color == perspective { 0 } else { 7 }) + piece_kind_index(attacker.kind);
    let attacked =
        (if attacked.color == perspective { 0 } else { 7 }) + piece_kind_index(attacked.kind);
    let source = canonical_square_for(perspective, source);
    let target = canonical_square_for(perspective, target);
    let relation = (attacker * BOARD_SIZE + source) * BOARD_SIZE + target;
    let base = threat_relation_map()[relation];
    if base == u32::MAX {
        return VALUE_THREAT_VOCAB;
    }
    let offset = threat_attacked_offsets()[target * STRUCTURAL_PIECE_SIZE + attacked];
    if offset == u8::MAX {
        return VALUE_THREAT_VOCAB;
    }
    base as usize + usize::from(offset)
}

#[inline]
fn policy_consequence_features(
    position: &Position,
    side: Color,
    mv: Move,
) -> Option<(usize, usize, Option<usize>)> {
    let moved = position.piece_at(mv.from as usize)?;
    let canonical = canonical_move(side, mv);
    let moved_piece_index =
        (if moved.color == side { 0 } else { 7 }) + piece_kind_index(moved.kind);
    let from = moved_piece_index * BOARD_SIZE + canonical.from as usize;
    let to = moved_piece_index * BOARD_SIZE + canonical.to as usize;
    let captured = position.piece_at(mv.to as usize).map(|piece| {
        let piece_index = (if piece.color == side { 0 } else { 7 }) + piece_kind_index(piece.kind);
        piece_index * BOARD_SIZE + canonical.to as usize
    });
    Some((from, to, captured))
}

#[inline]
pub(super) const fn policy_sparse_main_index(
    move_index: usize,
    moved_piece: usize,
    us_king_bucket: usize,
    them_king_bucket: usize,
) -> usize {
    (((move_index * STRUCTURAL_PIECE_SIZE + moved_piece) * V2_KING_BUCKETS + us_king_bucket)
        * V2_KING_BUCKETS)
        + them_king_bucket
}

#[inline]
pub(super) const fn policy_sparse_capture_index(
    move_index: usize,
    captured_piece: Option<usize>,
) -> usize {
    POLICY_SPARSE_MAIN_SIZE
        + move_index * POLICY_SPARSE_CAPTURE_CLASSES
        + match captured_piece {
            Some(piece) => piece,
            None => STRUCTURAL_PIECE_SIZE,
        }
}

#[inline]
pub(super) fn policy_sparse_factor_indices(
    move_index: usize,
    moved_piece: usize,
    us_king_bucket: usize,
    them_king_bucket: usize,
) -> [usize; 5] {
    let king_pair = us_king_bucket * V2_KING_BUCKETS + them_king_bucket;
    let (distance, approach) = policy_king_distance_buckets(move_index, them_king_bucket);
    let distance_offset = POLICY_SPARSE_MOVE_PIECE_SIZE
        + POLICY_SPARSE_MOVE_KING_SIZE
        + POLICY_SPARSE_PIECE_KING_SIZE;
    [
        move_index * STRUCTURAL_PIECE_SIZE + moved_piece,
        POLICY_SPARSE_MOVE_PIECE_SIZE + move_index * V2_KING_BUCKETS * V2_KING_BUCKETS + king_pair,
        POLICY_SPARSE_MOVE_PIECE_SIZE
            + POLICY_SPARSE_MOVE_KING_SIZE
            + moved_piece * V2_KING_BUCKETS * V2_KING_BUCKETS
            + king_pair,
        distance_offset + moved_piece * POLICY_KING_DISTANCE_BUCKETS + distance,
        distance_offset
            + POLICY_SPARSE_DISTANCE_SIZE
            + moved_piece * POLICY_KING_APPROACH_BUCKETS
            + approach,
    ]
}

#[inline]
pub(super) fn policy_tactical_indices(
    move_index: usize,
    moved_piece: usize,
    source_attacked: bool,
    destination_attacked: bool,
    source_defended: bool,
    destination_defended: bool,
    capture: bool,
    check: bool,
) -> [usize; 2] {
    debug_assert!(moved_piece < STRUCTURAL_PIECE_SIZE / 2);
    let signature = usize::from(source_attacked)
        | usize::from(destination_attacked) << 1
        | usize::from(source_defended) << 2
        | usize::from(destination_defended) << 3
        | usize::from(capture) << 4
        | usize::from(check) << 5;
    let exact = (move_index * (STRUCTURAL_PIECE_SIZE / 2) + moved_piece)
        * POLICY_TACTICAL_SIGNATURE_BUCKETS
        + signature;
    let factor =
        POLICY_TACTICAL_EXACT_SIZE + moved_piece * POLICY_TACTICAL_SIGNATURE_BUCKETS + signature;
    [exact, factor]
}

fn policy_king_distance_buckets(move_index: usize, them_king_bucket: usize) -> (usize, usize) {
    let sparse = move_map().dense_to_sparse[move_index] as usize;
    let from = sparse / BOARD_SIZE;
    let to = sparse % BOARD_SIZE;
    let king_own_rank = 7 + them_king_bucket / 3;
    let king_own_file = 3 + them_king_bucket % 3;
    let king = BOARD_SIZE - 1 - (king_own_rank * BOARD_FILES + king_own_file);
    let distance = |square: usize| {
        (square / BOARD_FILES).abs_diff(king / BOARD_FILES)
            + (square % BOARD_FILES).abs_diff(king % BOARD_FILES)
    };
    let before = distance(from);
    let after = distance(to);
    let approach = (before as isize - after as isize).clamp(-2, 2) + 2;
    (
        after.min(POLICY_KING_DISTANCE_BUCKETS - 1),
        approach as usize,
    )
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
    pub rule_context_hidden: Vec<f32>,
    pub hidden_bias: Vec<f32>,
    pub value_head_hidden: Vec<f32>,
    pub value_head_bias: Vec<f32>,
    pub value_head_output: Vec<f32>,
    pub value_threat_embedding: Vec<f32>,
    pub value_threat_output: Vec<f32>,
    pub policy_threat_context: Vec<f32>,
    pub policy_move_bias: Vec<f32>,
    pub policy_consequence_output: Vec<f32>,
    pub policy_context_hidden: Vec<f32>,
    pub policy_move_context: Vec<f32>,
    pub policy_accumulator_hidden: Vec<f32>,
    pub policy_accumulator_move: Vec<f32>,
    pub policy_sparse_table: Vec<f32>,
    pub policy_sparse_factor: Vec<f32>,
    pub policy_tactical: Vec<f32>,
    policy_accumulator_feature_q: Vec<i8>,
    policy_accumulator_move_q: Vec<i8>,
    policy_accumulator_moved_delta_q: Vec<i32>,
    policy_accumulator_capture_q: Vec<i32>,
    policy_accumulator_feature_scale: f32,
    policy_accumulator_move_scale: f32,
    policy_sparse_table_q: Vec<i8>,
    policy_sparse_table_scale: f32,
    value_threat_embedding_q: Vec<i8>,
    value_threat_embedding_scale: f32,
    value_threat_active: bool,
    policy_tactical_active: bool,
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
            rule_context_hidden: self.rule_context_hidden.clone(),
            hidden_bias: self.hidden_bias.clone(),
            value_head_hidden: self.value_head_hidden.clone(),
            value_head_bias: self.value_head_bias.clone(),
            value_head_output: self.value_head_output.clone(),
            value_threat_embedding: self.value_threat_embedding.clone(),
            value_threat_output: self.value_threat_output.clone(),
            policy_threat_context: self.policy_threat_context.clone(),
            policy_move_bias: self.policy_move_bias.clone(),
            policy_consequence_output: self.policy_consequence_output.clone(),
            policy_context_hidden: self.policy_context_hidden.clone(),
            policy_move_context: self.policy_move_context.clone(),
            policy_accumulator_hidden: self.policy_accumulator_hidden.clone(),
            policy_accumulator_move: self.policy_accumulator_move.clone(),
            policy_sparse_table: self.policy_sparse_table.clone(),
            policy_sparse_factor: self.policy_sparse_factor.clone(),
            policy_tactical: self.policy_tactical.clone(),
            policy_accumulator_feature_q: self.policy_accumulator_feature_q.clone(),
            policy_accumulator_move_q: self.policy_accumulator_move_q.clone(),
            policy_accumulator_moved_delta_q: self.policy_accumulator_moved_delta_q.clone(),
            policy_accumulator_capture_q: self.policy_accumulator_capture_q.clone(),
            policy_accumulator_feature_scale: self.policy_accumulator_feature_scale,
            policy_accumulator_move_scale: self.policy_accumulator_move_scale,
            policy_sparse_table_q: self.policy_sparse_table_q.clone(),
            policy_sparse_table_scale: self.policy_sparse_table_scale,
            value_threat_embedding_q: self.value_threat_embedding_q.clone(),
            value_threat_embedding_scale: self.value_threat_embedding_scale,
            value_threat_active: self.value_threat_active,
            policy_tactical_active: self.policy_tactical_active,
            gpu_trainer: None,
        }
    }
}

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
    pub temperature_endgame: f32,
    pub temperature_decay_delay_plies: usize,
    pub temperature_decay_plies: usize,
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
    pub policy_softmax_temp: f32,
    pub value_td_lambda: f32,
    pub opening_positions: Arc<[AzStartSnapshot]>,
    pub opening_start_fraction: f32,
    pub midgame_positions: Arc<[AzStartSnapshot]>,
    pub midgame_start_fraction: f32,
    pub mirror_probability: f32,
    pub record_fens: bool,
}

#[derive(Clone, Debug, Default)]
pub struct AzLoopReport {
    pub games: usize,
    pub samples: usize,
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
    pub train_fast_sample_rate: f32,
    pub train_policy_weight_mean: f32,
    pub train_value_weight_mean: f32,
    pub train_recent_quota_rate: f32,
    pub train_actual_recent_sample_rate: f32,
    pub train_start_source_rate: [f32; 3],
    pub train_policy_target_top1: f32,
    pub train_policy_target_top2: f32,
    pub terminal_no_legal_moves: usize,
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
    pub rule_context: [f32; RULE_CONTEXT_SIZE],
    pub move_indices: Vec<usize>,
    pub policy: Vec<f32>,
    pub value_wdl: [f32; WDL_HEAD_SIZE],
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
        model.evaluate_with_scratch_output(&position, &moves, &sample.rule_context, &mut scratch);
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
    OpeningPool = 1,
    Midgame = 2,
}

impl AzStartSource {
    pub const COUNT: usize = 3;

    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Startpos),
            1 => Some(Self::OpeningPool),
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

#[derive(Clone, Copy, Debug)]
pub(super) struct AzEvalOutput {
    pub value_wdl: [f32; WDL_HEAD_SIZE],
    pub value: f32,
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

impl AzNnue {
    pub fn random_with_arch(arch: AzNnueArch, seed: u64) -> Self {
        if let Err(err) = arch.validate() {
            panic!("AzNnue::random_with_arch: invalid arch ({err})");
        }
        let hidden_size = arch.hidden_size;
        let mut rng = SplitMix64::new(seed);
        let input_hidden: Vec<f32> = (0..AZ_NNUE_INPUT_SIZE * hidden_size)
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
        // large red/black advantage before any training, and MCTS amplifies
        // that noise into the first self-play dataset.
        let value_head_hidden = (0..VALUE_HEAD_SIZE * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt() * 0.5))
            .collect();
        let value_head_bias = vec![0.0; VALUE_HEAD_SIZE];
        // Keep the value head output-neutral at initialization. This preserves
        // stable first self-play while giving value its own nonlinear capacity.
        let value_head_output = vec![0.0; WDL_HEAD_SIZE * VALUE_HEAD_SIZE];
        let value_threat_embedding = (0..VALUE_THREAT_VOCAB * VALUE_THREAT_RANK)
            .map(|_| rng.weight(0.02))
            .collect();
        let value_threat_output = vec![0.0; WDL_HEAD_SIZE * VALUE_THREAT_RANK * 2];
        let policy_threat_context = vec![0.0; POLICY_THREAT_CONTEXT_SIZE * VALUE_THREAT_RANK * 2];
        let policy_move_bias = vec![0.0; DENSE_MOVE_SPACE];
        // Zero output preserves the exact policy distribution until this branch is trained.
        let policy_consequence_output = vec![0.0; POLICY_CONSEQUENCE_SIZE];
        // One factor starts random and the other at zero: the new branch is
        // exactly policy-neutral at initialization, while gradients can update
        // move embeddings on the first optimization step.
        let policy_context_hidden = (0..POLICY_MOVE_CONTEXT_SIZE * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt() * 0.5))
            .collect();
        let policy_move_context = vec![0.0; DENSE_MOVE_SPACE * POLICY_MOVE_CONTEXT_SIZE];
        let policy_accumulator_hidden = (0..POLICY_ACCUMULATOR_RANK * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt() * 0.5))
            .collect();
        let policy_accumulator_move = vec![0.0; DENSE_MOVE_SPACE * POLICY_ACCUMULATOR_RANK];
        let policy_sparse_table = vec![0.0; POLICY_SPARSE_TABLE_SIZE];
        let policy_sparse_factor = vec![0.0; POLICY_SPARSE_FACTOR_SIZE];
        let policy_tactical = vec![0.0; POLICY_TACTICAL_SIZE];
        let mut model = Self {
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
            value_threat_embedding,
            value_threat_output,
            policy_threat_context,
            policy_move_bias,
            policy_consequence_output,
            policy_context_hidden,
            policy_move_context,
            policy_accumulator_hidden,
            policy_accumulator_move,
            policy_sparse_table,
            policy_sparse_factor,
            policy_tactical,
            policy_accumulator_feature_q: Vec::new(),
            policy_accumulator_move_q: Vec::new(),
            policy_accumulator_moved_delta_q: Vec::new(),
            policy_accumulator_capture_q: Vec::new(),
            policy_accumulator_feature_scale: 1.0,
            policy_accumulator_move_scale: 1.0,
            policy_sparse_table_q: Vec::new(),
            policy_sparse_table_scale: 1.0,
            value_threat_embedding_q: Vec::new(),
            value_threat_embedding_scale: 1.0,
            value_threat_active: false,
            policy_tactical_active: false,
            gpu_trainer: None,
        };
        model.rebuild_policy_accumulator_quantization();
        model.rebuild_value_threat_quantization();
        model
    }

    pub fn random(hidden_size: usize, seed: u64) -> Self {
        Self::random_with_arch(AzNnueArch::with_hidden_size(hidden_size), seed)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let h = self.hidden_size;
        let varmap = VarMap::new();
        insert_candle_var(
            &varmap,
            "az_model_format_version",
            &[MODEL_FORMAT_VERSION],
            (1,),
        )?;
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
        let format_version = load_candle_f32_tensor(&tensors, "az_model_format_version")?;
        let Some(&format_version) = format_version.first() else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "missing AZ model format",
            ));
        };
        if format_version != MODEL_FORMAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported AZ model format {:?}; expected v{}",
                    format_version, MODEL_FORMAT_VERSION
                ),
            ));
        }
        let hidden_bias = load_candle_f32_tensor(&tensors, "hidden_bias")?;
        let hidden_size = hidden_bias.len();
        let arch = AzNnueArch { hidden_size };
        let mut model = Self {
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
            value_threat_embedding: load_candle_f32_tensor(&tensors, "value_threat_embedding")?,
            value_threat_output: load_candle_f32_tensor(&tensors, "value_threat_output")?,
            policy_threat_context: load_candle_f32_tensor(&tensors, "policy_threat_context")?,
            policy_move_bias: load_candle_f32_tensor(&tensors, "policy_move_bias")?,
            policy_consequence_output: load_candle_f32_tensor(
                &tensors,
                "policy_consequence_output",
            )?,
            policy_context_hidden: load_candle_f32_tensor(&tensors, "policy_context_hidden")?,
            policy_move_context: load_candle_f32_tensor(&tensors, "policy_move_context")?,
            policy_accumulator_hidden: load_candle_f32_tensor(
                &tensors,
                "policy_accumulator_hidden",
            )?,
            policy_accumulator_move: load_candle_f32_tensor(&tensors, "policy_accumulator_move")?,
            policy_sparse_table: load_candle_f32_tensor(&tensors, "policy_sparse_table")?,
            policy_sparse_factor: load_candle_f32_tensor(&tensors, "policy_sparse_factor")?,
            policy_tactical: load_candle_f32_tensor(&tensors, "policy_tactical")?,
            policy_accumulator_feature_q: Vec::new(),
            policy_accumulator_move_q: Vec::new(),
            policy_accumulator_moved_delta_q: Vec::new(),
            policy_accumulator_capture_q: Vec::new(),
            policy_accumulator_feature_scale: 1.0,
            policy_accumulator_move_scale: 1.0,
            policy_sparse_table_q: Vec::new(),
            policy_sparse_table_scale: 1.0,
            value_threat_embedding_q: Vec::new(),
            value_threat_embedding_scale: 1.0,
            value_threat_active: false,
            policy_tactical_active: false,
            gpu_trainer: None,
        };
        model.rebuild_policy_accumulator_quantization();
        model.rebuild_value_threat_quantization();
        model.rebuild_policy_tactical();
        model.validate()?;
        Ok(model)
    }

    pub fn evaluate_value(&self, position: &Position, moves: &[Move]) -> f32 {
        let mut scratch = AzEvalScratch::new(self.arch);
        self.evaluate_with_scratch(position, moves, &mut scratch)
    }

    pub fn evaluate_value_with_rules(
        &self,
        position: &Position,
        history: &[crate::xiangqi::RuleHistoryEntry],
        moves: &[Move],
    ) -> f32 {
        let mut scratch = AzEvalScratch::new(self.arch);
        self.evaluate_with_scratch_output(
            position,
            moves,
            &rule_context_features(position, history),
            &mut scratch,
        )
        .value
    }

    pub(super) fn evaluate_with_scratch(
        &self,
        position: &Position,
        moves: &[Move],
        scratch: &mut AzEvalScratch,
    ) -> f32 {
        self.evaluate_with_scratch_output(position, moves, &[0.0; RULE_CONTEXT_SIZE], scratch)
            .value
    }

    pub(super) fn evaluate_with_scratch_output(
        &self,
        position: &Position,
        moves: &[Move],
        rule_context: &[f32; RULE_CONTEXT_SIZE],
        scratch: &mut AzEvalScratch,
    ) -> AzEvalOutput {
        crate::scope_profile!("az.evaluate_with_scratch");
        let mut features = std::mem::take(&mut scratch.features);
        {
            crate::scope_profile!("az.eval.extract_features");
            fill_sparse_features_az(position, &mut features);
        }
        {
            crate::scope_profile!("az.eval.input_embedding");
            self.input_embedding_linear_into(&features, &mut scratch.hidden);
            self.add_rule_context_to_hidden(rule_context, &mut scratch.hidden);
        }
        scratch.policy_accumulator_context =
            self.quantized_policy_accumulator(position, position.side_to_move());
        {
            crate::scope_profile!("az.eval.activation_norm");
            relu_in_place(&mut scratch.hidden);
            rms_norm_in_place(&mut scratch.hidden);
        }
        let (value_wdl, value) = {
            crate::scope_profile!("az.eval.value_head");
            let threat_logits = self.value_threat_logits(
                position,
                &mut scratch.value_threat_accumulator,
                &mut scratch.value_threat_activation,
            );
            self.value_wdl_from_hidden_into(
                &scratch.hidden,
                &features,
                &mut scratch.value_head,
                threat_logits,
            )
        };
        self.evaluate_prepared_hidden_with_scratch(position, &features, value, moves, scratch);
        scratch.features = features;
        AzEvalOutput { value_wdl, value }
    }

    pub(super) fn evaluate_incremental_with_scratch_output(
        &self,
        position: &Position,
        accumulator_hidden: &[f32],
        policy_accumulator: &[i16; POLICY_ACCUMULATOR_RANK],
        moves: &[Move],
        rule_context: &[f32; RULE_CONTEXT_SIZE],
        scratch: &mut AzEvalScratch,
    ) -> AzEvalOutput {
        crate::scope_profile!("az.evaluate_incremental_with_scratch");
        scratch.hidden.resize(self.hidden_size, 0.0);
        let hidden = if accumulator_hidden.len() == self.hidden_size {
            accumulator_hidden
        } else {
            AzEvalAccumulator::hidden_for_slice(
                accumulator_hidden,
                self.hidden_size,
                position.side_to_move(),
            )
        };
        scratch.hidden.copy_from_slice(hidden);
        self.add_rule_context_to_hidden(rule_context, &mut scratch.hidden);
        scratch
            .policy_accumulator_context
            .copy_from_slice(policy_accumulator);
        {
            crate::scope_profile!("az.eval.activation_norm");
            relu_in_place(&mut scratch.hidden);
            rms_norm_in_place(&mut scratch.hidden);
        }
        let (value_wdl, value) = {
            crate::scope_profile!("az.eval.value_head");
            let threat_logits = self.value_threat_logits(
                position,
                &mut scratch.value_threat_accumulator,
                &mut scratch.value_threat_activation,
            );
            self.value_wdl_from_hidden_into(
                &scratch.hidden,
                &[],
                &mut scratch.value_head,
                threat_logits,
            )
        };
        self.evaluate_prepared_hidden_with_scratch(position, &[], value, moves, scratch);
        AzEvalOutput { value_wdl, value }
    }

    #[allow(dead_code)]
    pub(super) fn evaluate_incremental_batch4(
        &self,
        requests: &mut [AzIncrementalEvalRequest<'_>; 4],
    ) -> [AzEvalOutput; 4] {
        for request in requests.iter_mut() {
            request.scratch.hidden.resize(self.hidden_size, 0.0);
            let hidden = if request.accumulator_hidden.len() == self.hidden_size {
                request.accumulator_hidden
            } else {
                AzEvalAccumulator::hidden_for_slice(
                    request.accumulator_hidden,
                    self.hidden_size,
                    request.position.side_to_move(),
                )
            };
            request.scratch.hidden.copy_from_slice(hidden);
            self.add_rule_context_to_hidden(request.rule_context, &mut request.scratch.hidden);
            request
                .scratch
                .policy_accumulator_context
                .copy_from_slice(request.policy_accumulator);
            relu_in_place(&mut request.scratch.hidden);
            rms_norm_in_place(&mut request.scratch.hidden);
        }

        let threat_logits = std::array::from_fn(|index| {
            self.value_threat_logits(
                requests[index].position,
                &mut requests[index].scratch.value_threat_accumulator,
                &mut requests[index].scratch.value_threat_activation,
            )
        });
        let hiddens = [
            requests[0].scratch.hidden.as_slice(),
            requests[1].scratch.hidden.as_slice(),
            requests[2].scratch.hidden.as_slice(),
            requests[3].scratch.hidden.as_slice(),
        ];
        let (value_wdls, values) = self.value_wdl_batch4(hiddens, threat_logits);
        let mut contexts = [[0.0f32; POLICY_MOVE_CONTEXT_SIZE]; 4];
        for context_index in 0..POLICY_MOVE_CONTEXT_SIZE {
            let start = context_index * self.hidden_size;
            let dots = dot_product_f32_batch4(
                hiddens,
                &self.policy_context_hidden[start..start + self.hidden_size],
            );
            for batch in 0..4 {
                let threat = &requests[batch].scratch.value_threat_activation;
                let threat_logit = if context_index < POLICY_THREAT_CONTEXT_SIZE
                    && threat.len() == VALUE_THREAT_RANK * 2
                {
                    let start = context_index * VALUE_THREAT_RANK * 2;
                    dot_product(
                        threat,
                        &self.policy_threat_context[start..start + VALUE_THREAT_RANK * 2],
                    )
                } else {
                    0.0
                };
                contexts[batch][context_index] = dots[batch] + threat_logit;
            }
        }
        for index in 0..4 {
            let request = &mut requests[index];
            request.scratch.policy_context.clear();
            request
                .scratch
                .policy_context
                .extend_from_slice(&contexts[index]);
            self.evaluate_prepared_hidden_with_context(
                request.position,
                &[],
                values[index],
                request.moves,
                request.scratch,
            );
        }
        std::array::from_fn(|index| AzEvalOutput {
            value_wdl: value_wdls[index],
            value: values[index],
        })
    }

    fn evaluate_prepared_hidden_with_scratch(
        &self,
        position: &Position,
        features: &[usize],
        value: f32,
        moves: &[Move],
        scratch: &mut AzEvalScratch,
    ) -> f32 {
        scratch.policy_context.resize(POLICY_MOVE_CONTEXT_SIZE, 0.0);
        for (context_index, context) in scratch.policy_context.iter_mut().enumerate() {
            let start = context_index * self.hidden_size;
            *context = dot_product(
                &scratch.hidden,
                &self.policy_context_hidden[start..start + self.hidden_size],
            );
            if context_index < POLICY_THREAT_CONTEXT_SIZE
                && scratch.value_threat_activation.len() == VALUE_THREAT_RANK * 2
            {
                let threat_start = context_index * VALUE_THREAT_RANK * 2;
                *context += dot_product(
                    &scratch.value_threat_activation,
                    &self.policy_threat_context[threat_start..threat_start + VALUE_THREAT_RANK * 2],
                );
            }
        }
        self.evaluate_prepared_hidden_with_context(position, features, value, moves, scratch)
    }

    fn evaluate_prepared_hidden_with_context(
        &self,
        position: &Position,
        features: &[usize],
        value: f32,
        moves: &[Move],
        scratch: &mut AzEvalScratch,
    ) -> f32 {
        scratch.logits.resize(moves.len(), 0.0);
        if scratch.policy_piece_square_scores.is_empty() {
            self.fill_policy_piece_square_scores(&mut scratch.policy_piece_square_scores);
        }
        let move_map = move_map();
        let side = position.side_to_move();
        let king_buckets = canonical_buckets_for_perspective(position, side);
        self.fill_policy_gives_checks(position, moves, &mut scratch.policy_gives_check);
        let attack_masks = self
            .policy_tactical_active
            .then(|| position.attacked_squares_masks())
            .unwrap_or_default();
        let opponent_attacks = attack_masks[color_index(side.opposite())];
        let own_attacks = attack_masks[color_index(side)];
        {
            crate::scope_profile!("az.eval.policy_logits");
            {
                crate::scope_profile!("az.eval.policy.logit_arith");
                for (index, mv) in moves.iter().enumerate() {
                    let canonical = canonical_move(side, *mv);
                    let sparse = canonical.from as usize * BOARD_SIZE + canonical.to as usize;
                    let dense = move_map.sparse_to_dense[sparse];
                    debug_assert!(
                        dense != u16::MAX,
                        "invalid policy move {}->{}",
                        mv.from,
                        mv.to
                    );
                    let move_index = dense as usize;
                    let context_start = move_index * POLICY_MOVE_CONTEXT_SIZE;
                    let accumulator_start = move_index * POLICY_ACCUMULATOR_RANK;
                    let accumulator_move = &self.policy_accumulator_move_q
                        [accumulator_start..accumulator_start + POLICY_ACCUMULATOR_RANK];
                    let consequence = policy_consequence_features(position, side, *mv);
                    let piece_square_logit = consequence.map_or(0.0, |(from, to, captured)| {
                        scratch.policy_piece_square_scores[to]
                            - scratch.policy_piece_square_scores[from]
                            - captured
                                .map_or(0.0, |feature| scratch.policy_piece_square_scores[feature])
                    });
                    let accumulator_logit = if let Some((from, to, captured)) = consequence {
                        debug_assert_eq!(from / BOARD_SIZE, to / BOARD_SIZE);
                        let cache_start = move_index * STRUCTURAL_PIECE_SIZE;
                        let mut value = dot_product_i16_i8_32(
                            &scratch.policy_accumulator_context,
                            accumulator_move,
                        ) + self.policy_accumulator_moved_delta_q
                            [cache_start + from / BOARD_SIZE];
                        if let Some(captured) = captured {
                            value -= self.policy_accumulator_capture_q
                                [cache_start + captured / BOARD_SIZE];
                        }
                        value as f32
                            * self.policy_accumulator_feature_scale
                            * self.policy_accumulator_move_scale
                    } else {
                        0.0
                    };
                    let sparse_logit = consequence.map_or(0.0, |(from, _, captured)| {
                        let moved_piece = from / BOARD_SIZE;
                        let captured_piece = captured.map(|feature| feature / BOARD_SIZE);
                        let main = policy_sparse_main_index(
                            move_index,
                            moved_piece,
                            king_buckets.0,
                            king_buckets.1,
                        );
                        let capture = policy_sparse_capture_index(move_index, captured_piece);
                        (i32::from(self.policy_sparse_table_q[main])
                            + i32::from(self.policy_sparse_table_q[capture]))
                            as f32
                            * self.policy_sparse_table_scale
                    });
                    let tactical_logit = if self.policy_tactical_active {
                        crate::scope_profile!("az.eval.policy.tactical");
                        consequence.map_or(0.0, |(from, _, captured)| {
                            let moved_piece = from / BOARD_SIZE;
                            let check = scratch.policy_gives_check[index];
                            let source_attacked =
                                opponent_attacks & (1u128 << mv.from as usize) != 0;
                            let destination_attacked =
                                opponent_attacks & (1u128 << mv.to as usize) != 0;
                            let source_defended = own_attacks & (1u128 << mv.from as usize) != 0;
                            let destination_defended = own_attacks & (1u128 << mv.to as usize) != 0;
                            policy_tactical_indices(
                                move_index,
                                moved_piece,
                                source_attacked,
                                destination_attacked,
                                source_defended,
                                destination_defended,
                                captured.is_some(),
                                check != 0.0,
                            )
                            .into_iter()
                            .map(|tactical| self.policy_tactical[tactical])
                            .sum()
                        })
                    } else {
                        0.0
                    };
                    scratch.logits[index] = self.policy_move_bias[move_index]
                        + piece_square_logit
                        + dot_product(
                            &scratch.policy_context,
                            &self.policy_move_context
                                [context_start..context_start + POLICY_MOVE_CONTEXT_SIZE],
                        )
                        + accumulator_logit
                        + sparse_logit
                        + tactical_logit;
                }
            }
        }
        let _ = features;
        value
    }

    fn fill_policy_gives_checks(&self, position: &Position, moves: &[Move], output: &mut Vec<f32>) {
        crate::scope_profile!("az.eval.policy.gives_check");
        output.resize(moves.len(), 0.0);
        for (flag, &mv) in output.iter_mut().zip(moves) {
            *flag = f32::from(position.gives_check_after_move_fast(mv));
        }
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

    fn value_threat_logits(
        &self,
        position: &Position,
        accumulator: &mut Vec<i16>,
        activation: &mut Vec<f32>,
    ) -> [f32; WDL_HEAD_SIZE] {
        if !self.value_threat_active {
            return [0.0; WDL_HEAD_SIZE];
        }
        crate::scope_profile!("az.eval.value_threat");
        accumulator.resize(VALUE_THREAT_RANK, 0);
        accumulator.fill(0);
        let perspective = position.side_to_move();
        {
            crate::scope_profile!("az.eval.value_threat.accumulate");
            position.visit_occupied_relations(|source, attacker, target, attacked| {
                let feature = value_threat_index(perspective, source, attacker, target, attacked);
                if feature == VALUE_THREAT_VOCAB {
                    return;
                }
                let row = &self.value_threat_embedding_q
                    [feature * VALUE_THREAT_RANK..(feature + 1) * VALUE_THREAT_RANK];
                add_i8_row_to_i16(accumulator, row);
            });
        }
        let mut logits = [0.0; WDL_HEAD_SIZE];
        {
            crate::scope_profile!("az.eval.value_threat.output");
            activation.resize(VALUE_THREAT_RANK * 2, 0.0);
            for rank in 0..VALUE_THREAT_RANK {
                let value = (f32::from(accumulator[rank]) * self.value_threat_embedding_scale)
                    .clamp(0.0, 1.0);
                activation[rank] = value;
                activation[VALUE_THREAT_RANK + rank] = value * value;
            }
            for (output, logit) in logits.iter_mut().enumerate() {
                let row = &self.value_threat_output
                    [output * VALUE_THREAT_RANK * 2..(output + 1) * VALUE_THREAT_RANK * 2];
                *logit = dot_product(activation, row);
            }
        }
        logits
    }

    #[allow(dead_code)]
    fn value_from_hidden_into(
        &self,
        hidden: &[f32],
        features: &[usize],
        value_head: &mut Vec<f32>,
    ) -> f32 {
        let probs = self.value_wdl_from_hidden_into(hidden, features, value_head, [0.0; 3]);
        probs.1
    }

    fn value_wdl_from_hidden_into(
        &self,
        hidden: &[f32],
        features: &[usize],
        value_head: &mut Vec<f32>,
        threat_logits: [f32; WDL_HEAD_SIZE],
    ) -> ([f32; WDL_HEAD_SIZE], f32) {
        value_head.resize(VALUE_HEAD_SIZE, 0.0);
        value_head.copy_from_slice(&self.value_head_bias);
        for (feature, value) in value_head.iter_mut().enumerate().take(VALUE_HEAD_SIZE) {
            let hidden_row = &self.value_head_hidden
                [feature * self.hidden_size..(feature + 1) * self.hidden_size];
            *value += dot_product(hidden, hidden_row);
            *value = (*value).max(0.0);
        }
        let _ = features;
        let mut logits = [0.0f32; WDL_HEAD_SIZE];
        for (out, logit) in logits.iter_mut().enumerate() {
            let row = &self.value_head_output[out * VALUE_HEAD_SIZE..(out + 1) * VALUE_HEAD_SIZE];
            *logit = dot_product(value_head, row) + threat_logits[out];
        }
        let wdl = softmax_fixed3(logits);
        let q = wdl[0] - wdl[2];
        (wdl, q)
    }

    #[allow(dead_code)]
    fn value_wdl_batch4(
        &self,
        hiddens: [&[f32]; 4],
        threat_logits: [[f32; WDL_HEAD_SIZE]; 4],
    ) -> ([[f32; WDL_HEAD_SIZE]; 4], [f32; 4]) {
        let mut heads = [[0.0f32; VALUE_HEAD_SIZE]; 4];
        for feature in 0..VALUE_HEAD_SIZE {
            let row = &self.value_head_hidden
                [feature * self.hidden_size..(feature + 1) * self.hidden_size];
            let dots = dot_product_f32_batch4(hiddens, row);
            for batch in 0..4 {
                heads[batch][feature] = (self.value_head_bias[feature] + dots[batch]).max(0.0);
            }
        }
        let mut wdls = [[0.0; WDL_HEAD_SIZE]; 4];
        let mut values = [0.0; 4];
        for batch in 0..4 {
            let mut logits = [0.0; WDL_HEAD_SIZE];
            for (out, logit) in logits.iter_mut().enumerate() {
                let row =
                    &self.value_head_output[out * VALUE_HEAD_SIZE..(out + 1) * VALUE_HEAD_SIZE];
                *logit = dot_product(&heads[batch], row) + threat_logits[batch][out];
            }
            wdls[batch] = softmax_fixed3(logits);
            values[batch] = wdls[batch][0] - wdls[batch][2];
        }
        (wdls, values)
    }

    fn fill_policy_piece_square_scores(&self, scores: &mut Vec<f32>) {
        let consequence_size = POLICY_CONSEQUENCE_SIZE.min(self.hidden_size);
        scores.resize(PIECE_SQUARE_INPUT_SIZE, 0.0);
        for (feature, score) in scores.iter_mut().enumerate() {
            let start = feature * self.hidden_size;
            *score = dot_product(
                &self.input_hidden[start..start + consequence_size],
                &self.policy_consequence_output[..consequence_size],
            );
        }
    }

    fn rebuild_value_threat_quantization(&mut self) {
        self.value_threat_active = self.value_threat_output.iter().any(|&weight| weight != 0.0)
            || self
                .policy_threat_context
                .iter()
                .any(|&weight| weight != 0.0);
        let maximum = self
            .value_threat_embedding
            .iter()
            .fold(0.0f32, |current, value| current.max(value.abs()));
        self.value_threat_embedding_scale = (maximum / 127.0).max(1.0e-12);
        self.value_threat_embedding_q = self
            .value_threat_embedding
            .iter()
            .map(|value| {
                (*value / self.value_threat_embedding_scale)
                    .round()
                    .clamp(-127.0, 127.0) as i8
            })
            .collect();
    }

    fn rebuild_policy_tactical(&mut self) {
        self.policy_tactical_active = self.policy_tactical.iter().any(|&weight| weight != 0.0);
    }

    fn rebuild_policy_accumulator_quantization(&mut self) {
        let mut projected =
            Vec::with_capacity(POLICY_ACCUMULATOR_QUANT_ROWS * POLICY_ACCUMULATOR_RANK);
        let projection = &self.policy_accumulator_hidden;
        let hidden = self.hidden_size;
        let mut append = |table: &[f32]| {
            debug_assert_eq!(table.len() % hidden, 0);
            for row in table.chunks_exact(hidden) {
                for rank in 0..POLICY_ACCUMULATOR_RANK {
                    let weights = &projection[rank * hidden..(rank + 1) * hidden];
                    projected.push(dot_product(row, weights));
                }
            }
        };
        append(&self.input_hidden);
        append(&self.input_piece_hidden);
        append(&self.input_rank_hidden);
        append(&self.input_file_hidden);
        append(&self.input_king_piece_hidden);
        append(&self.hidden_bias);
        debug_assert_eq!(
            projected.len(),
            POLICY_ACCUMULATOR_QUANT_ROWS * POLICY_ACCUMULATOR_RANK
        );
        let feature_max = projected
            .iter()
            .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
        self.policy_accumulator_feature_scale = (feature_max / 127.0).max(1.0e-12);
        self.policy_accumulator_feature_q = projected
            .into_iter()
            .map(|value| {
                (value / self.policy_accumulator_feature_scale)
                    .round()
                    .clamp(-127.0, 127.0) as i8
            })
            .collect();

        let move_max = self
            .policy_accumulator_move
            .iter()
            .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
        self.policy_accumulator_move_scale = (move_max / 127.0).max(1.0e-12);
        self.policy_accumulator_move_q = self
            .policy_accumulator_move
            .iter()
            .map(|value| {
                (*value / self.policy_accumulator_move_scale)
                    .round()
                    .clamp(-127.0, 127.0) as i8
            })
            .collect();

        let mut folded_sparse = self.policy_sparse_table.clone();
        for move_index in 0..DENSE_MOVE_SPACE {
            for moved_piece in 0..STRUCTURAL_PIECE_SIZE {
                for us_bucket in 0..V2_KING_BUCKETS {
                    for them_bucket in 0..V2_KING_BUCKETS {
                        let main = policy_sparse_main_index(
                            move_index,
                            moved_piece,
                            us_bucket,
                            them_bucket,
                        );
                        for factor in policy_sparse_factor_indices(
                            move_index,
                            moved_piece,
                            us_bucket,
                            them_bucket,
                        ) {
                            folded_sparse[main] += self.policy_sparse_factor[factor];
                        }
                    }
                }
            }
        }
        let sparse_max = folded_sparse
            .iter()
            .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
        self.policy_sparse_table_scale = (sparse_max / 127.0).max(1.0e-12);
        self.policy_sparse_table_q = folded_sparse
            .iter()
            .map(|value| {
                (*value / self.policy_sparse_table_scale)
                    .round()
                    .clamp(-127.0, 127.0) as i8
            })
            .collect();

        let cache_size = DENSE_MOVE_SPACE * STRUCTURAL_PIECE_SIZE;
        self.policy_accumulator_moved_delta_q = vec![0; cache_size];
        self.policy_accumulator_capture_q = vec![0; cache_size];
        for (move_index, &sparse) in move_map().dense_to_sparse.iter().enumerate() {
            let sparse = sparse as usize;
            let from_square = sparse / BOARD_SIZE;
            let to_square = sparse % BOARD_SIZE;
            let move_start = move_index * POLICY_ACCUMULATOR_RANK;
            let move_weights =
                &self.policy_accumulator_move_q[move_start..move_start + POLICY_ACCUMULATOR_RANK];
            for piece_index in 0..STRUCTURAL_PIECE_SIZE {
                let from_start = (piece_index * BOARD_SIZE + from_square) * POLICY_ACCUMULATOR_RANK;
                let to_start = (piece_index * BOARD_SIZE + to_square) * POLICY_ACCUMULATOR_RANK;
                let mut moved_delta = 0i32;
                let mut capture = 0i32;
                for rank in 0..POLICY_ACCUMULATOR_RANK {
                    let weight = i32::from(move_weights[rank]);
                    let to = i32::from(self.policy_accumulator_feature_q[to_start + rank]);
                    let from = i32::from(self.policy_accumulator_feature_q[from_start + rank]);
                    moved_delta += (to - from) * weight;
                    capture += to * weight;
                }
                let cache_index = move_index * STRUCTURAL_PIECE_SIZE + piece_index;
                self.policy_accumulator_moved_delta_q[cache_index] = moved_delta;
                self.policy_accumulator_capture_q[cache_index] = capture;
            }
        }
    }

    pub(super) fn quantized_policy_accumulator(
        &self,
        position: &Position,
        perspective: Color,
    ) -> [i16; POLICY_ACCUMULATOR_RANK] {
        let mut accumulator = [0i16; POLICY_ACCUMULATOR_RANK];
        self.add_quantized_policy_row(&mut accumulator, POLICY_ACCUMULATOR_BIAS_ROW, 1);
        let buckets = canonical_buckets_for_perspective(position, perspective);
        for square in 0..BOARD_SIZE {
            if let Some(piece) = position.piece_at(square) {
                self.add_quantized_policy_piece(
                    &mut accumulator,
                    perspective,
                    buckets,
                    square,
                    piece,
                    1,
                );
            }
        }
        accumulator
    }

    pub(super) fn apply_quantized_policy_transition(
        &self,
        before: &Position,
        after: &Position,
        mv: Move,
        moved: Piece,
        captured: Option<Piece>,
        perspective: Color,
        accumulator: &mut [i16; POLICY_ACCUMULATOR_RANK],
    ) {
        let before_buckets = canonical_buckets_for_perspective(before, perspective);
        let after_buckets = canonical_buckets_for_perspective(after, perspective);
        if before_buckets != after_buckets {
            *accumulator = self.quantized_policy_accumulator(after, perspective);
            return;
        }
        self.add_quantized_policy_piece(
            accumulator,
            perspective,
            before_buckets,
            mv.from as usize,
            moved,
            -1,
        );
        if let Some(captured) = captured {
            self.add_quantized_policy_piece(
                accumulator,
                perspective,
                before_buckets,
                mv.to as usize,
                captured,
                -1,
            );
        }
        self.add_quantized_policy_piece(
            accumulator,
            perspective,
            after_buckets,
            mv.to as usize,
            moved,
            1,
        );
    }

    fn add_quantized_policy_piece(
        &self,
        accumulator: &mut [i16; POLICY_ACCUMULATOR_RANK],
        perspective: Color,
        buckets: (usize, usize),
        square: usize,
        piece: Piece,
        sign: i32,
    ) {
        let relative_color = if piece.color == perspective { 0 } else { 7 };
        let piece_index = relative_color + piece_kind_index(piece.kind);
        let relative_square = canonical_square_for(perspective, square);
        let rank = relative_square / BOARD_FILES;
        let file = relative_square % BOARD_FILES;
        for row in [
            piece_index * BOARD_SIZE + relative_square,
            POLICY_ACCUMULATOR_PIECE_OFFSET + piece_index,
            POLICY_ACCUMULATOR_RANK_OFFSET + rank,
            POLICY_ACCUMULATOR_FILE_OFFSET + file,
            POLICY_ACCUMULATOR_KING_PIECE_OFFSET
                + structural_king_piece_index(0, buckets.0, piece_index),
            POLICY_ACCUMULATOR_KING_PIECE_OFFSET
                + structural_king_piece_index(1, buckets.1, piece_index),
        ] {
            self.add_quantized_policy_row(accumulator, row, sign);
        }
    }

    fn add_quantized_policy_row(
        &self,
        accumulator: &mut [i16; POLICY_ACCUMULATOR_RANK],
        row: usize,
        sign: i32,
    ) {
        let start = row * POLICY_ACCUMULATOR_RANK;
        let values = &self.policy_accumulator_feature_q[start..start + POLICY_ACCUMULATOR_RANK];
        for (target, &value) in accumulator.iter_mut().zip(values) {
            *target = (i32::from(*target) + sign * i32::from(value))
                .clamp(i16::MIN as i32, i16::MAX as i32) as i16;
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
        if self.policy_accumulator_feature_q.len()
            != POLICY_ACCUMULATOR_QUANT_ROWS * POLICY_ACCUMULATOR_RANK
            || self.policy_accumulator_move_q.len() != DENSE_MOVE_SPACE * POLICY_ACCUMULATOR_RANK
            || self.policy_accumulator_moved_delta_q.len()
                != DENSE_MOVE_SPACE * STRUCTURAL_PIECE_SIZE
            || self.policy_accumulator_capture_q.len() != DENSE_MOVE_SPACE * STRUCTURAL_PIECE_SIZE
            || self.policy_sparse_table_q.len() != POLICY_SPARSE_TABLE_SIZE
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "az model derived quantized policy accumulator cache length mismatch",
            ));
        }
        Ok(())
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
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            move_indices,
            policy,
            value_wdl: scalar_value_to_wdl_target(value),
            value,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            search_simulations: 0,
            meta: AzSampleMeta::default(),
        });
        if index + 1 == sample_count {
            break;
        }
    }
    let stats = train_samples(model, &samples, epochs, lr, batch_size, &mut rng)
        .unwrap_or_else(|err| panic!("training failed: {err}"));
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

#[inline]
fn dot_product_i16_i8_32(left: &[i16; POLICY_ACCUMULATOR_RANK], right: &[i8]) -> i32 {
    debug_assert_eq!(right.len(), POLICY_ACCUMULATOR_RANK);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was checked at runtime and both inputs contain exactly 32 elements.
        return unsafe { dot_product_i16_i8_32_avx2(left, right) };
    }
    left.iter()
        .zip(right)
        .map(|(&a, &b)| i32::from(a) * i32::from(b))
        .sum()
}

#[inline]
fn add_i8_row_to_i16(accumulator: &mut [i16], row: &[i8]) {
    debug_assert_eq!(accumulator.len(), row.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if accumulator.len().is_multiple_of(32) && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was checked at runtime and both slices have equal 32-byte chunks.
        unsafe { add_i8_row_to_i16_avx2(accumulator, row) };
        return;
    }
    for (sum, &weight) in accumulator.iter_mut().zip(row) {
        *sum += i16::from(weight);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn add_i8_row_to_i16_avx2(accumulator: &mut [i16], row: &[i8]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    for offset in (0..row.len()).step_by(32) {
        let packed = unsafe { _mm256_loadu_si256(row.as_ptr().add(offset).cast()) };
        let low = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(packed));
        let high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(packed));
        unsafe {
            let low_sum = _mm256_loadu_si256(accumulator.as_ptr().add(offset).cast());
            let high_sum = _mm256_loadu_si256(accumulator.as_ptr().add(offset + 16).cast());
            _mm256_storeu_si256(
                accumulator.as_mut_ptr().add(offset).cast(),
                _mm256_add_epi16(low_sum, low),
            );
            _mm256_storeu_si256(
                accumulator.as_mut_ptr().add(offset + 16).cast(),
                _mm256_add_epi16(high_sum, high),
            );
        }
    }
}

#[allow(dead_code)]
#[inline]
fn dot_product_f32_batch4(inputs: [&[f32]; 4], weights: &[f32]) -> [f32; 4] {
    debug_assert!(inputs.iter().all(|input| input.len() == weights.len()));
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if weights.len() >= 32
        && std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
    {
        // SAFETY: AVX2/FMA were checked at runtime and all slices have equal lengths.
        return unsafe { dot_product_f32_batch4_avx2_fma(inputs, weights) };
    }
    let mut sums = [0.0f32; 4];
    for index in 0..weights.len() {
        for batch in 0..4 {
            sums[batch] += inputs[batch][index] * weights[index];
        }
    }
    sums
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(dead_code)]
unsafe fn dot_product_f32_batch4_avx2_fma(inputs: [&[f32]; 4], weights: &[f32]) -> [f32; 4] {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let chunks = weights.len() / 8;
    let mut accumulators = [_mm256_setzero_ps(); 4];
    for chunk in 0..chunks {
        let offset = chunk * 8;
        let weight = unsafe { _mm256_loadu_ps(weights.as_ptr().add(offset)) };
        for batch in 0..4 {
            let input = unsafe { _mm256_loadu_ps(inputs[batch].as_ptr().add(offset)) };
            accumulators[batch] = _mm256_fmadd_ps(input, weight, accumulators[batch]);
        }
    }
    let mut sums = [0.0f32; 4];
    for batch in 0..4 {
        let mut lanes = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), accumulators[batch]) };
        sums[batch] = lanes.into_iter().sum();
    }
    for index in (chunks * 8)..weights.len() {
        for batch in 0..4 {
            sums[batch] += inputs[batch][index] * weights[index];
        }
    }
    sums
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_i16_i8_32_avx2(left: &[i16; POLICY_ACCUMULATOR_RANK], right: &[i8]) -> i32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut sums = _mm256_setzero_si256();
    for offset in [0, 16] {
        let a = unsafe { _mm256_loadu_si256(left.as_ptr().add(offset).cast()) };
        let b8 = unsafe { _mm_loadu_si128(right.as_ptr().add(offset).cast()) };
        let b = _mm256_cvtepi8_epi16(b8);
        sums = _mm256_add_epi32(sums, _mm256_madd_epi16(a, b));
    }
    let mut lanes = [0i32; 8];
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), sums) };
    lanes.into_iter().sum()
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

pub(super) fn dense_move_squares(move_index: usize) -> Option<(usize, usize)> {
    let sparse = *move_map().dense_to_sparse.get(move_index)? as usize;
    Some((sparse / BOARD_SIZE, sparse % BOARD_SIZE))
}

pub fn dense_move_index(mv: Move) -> usize {
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
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            move_indices: vec![0, 1],
            policy: vec![0.6, 0.4],
            value_wdl: scalar_value_to_wdl_target(0.1),
            value: 0.1,
            side_sign: 1.0,
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
                start_source: AzStartSource::Startpos,
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

    #[cfg(feature = "profile")]
    #[test]
    #[ignore = "manual profile harness"]
    fn manual_profile_policy_head() {
        let position = Position::startpos();
        let moves = position.legal_moves();
        let model = AzNnue::random(128, 999);
        let mut scratch = AzEvalScratch::new(model.arch);
        for _ in 0..50_000 {
            model.evaluate_with_scratch_output(
                &position,
                &moves,
                &[0.0; RULE_CONTEXT_SIZE],
                &mut scratch,
            );
        }
        crate::profile::print_report();
    }

    #[test]
    fn zero_initialized_consequence_branch_preserves_policy_logits() {
        let position = Position::startpos();
        let moves = position.legal_moves();
        let model = AzNnue::random(32, 20260729);
        assert!(
            model
                .policy_consequence_output
                .iter()
                .all(|&weight| weight == 0.0)
        );

        let mut baseline = AzEvalScratch::new(model.arch);
        model.evaluate_with_scratch_output(
            &position,
            &moves,
            &[0.0; RULE_CONTEXT_SIZE],
            &mut baseline,
        );

        let mut active = model.clone();
        active.policy_consequence_output.fill(0.1);
        let mut changed = AzEvalScratch::new(model.arch);
        active.evaluate_with_scratch_output(
            &position,
            &moves,
            &[0.0; RULE_CONTEXT_SIZE],
            &mut changed,
        );
        assert!(
            baseline
                .logits
                .iter()
                .zip(&changed.logits)
                .any(|(left, right)| left != right)
        );
    }

    #[test]
    fn zero_initialized_policy_accumulator_is_neutral_and_trainable() {
        let position = Position::startpos();
        let moves = position.legal_moves();
        let model = AzNnue::random(32, 20260816);
        assert!(
            model
                .policy_accumulator_move
                .iter()
                .all(|&weight| weight == 0.0)
        );

        let mut baseline = AzEvalScratch::new(model.arch);
        model.evaluate_with_scratch_output(
            &position,
            &moves,
            &[0.0; RULE_CONTEXT_SIZE],
            &mut baseline,
        );

        let mut active = model.clone();
        active.policy_accumulator_move.fill(0.01);
        active.rebuild_policy_accumulator_quantization();
        let mut changed = AzEvalScratch::new(model.arch);
        active.evaluate_with_scratch_output(
            &position,
            &moves,
            &[0.0; RULE_CONTEXT_SIZE],
            &mut changed,
        );
        assert!(
            baseline
                .logits
                .iter()
                .zip(&changed.logits)
                .any(|(left, right)| left != right)
        );
    }

    #[test]
    fn scalar_value_head_starts_neutral() {
        let model = AzNnue::random(16, 7);
        let mut scratch = AzEvalScratch::new(model.arch);
        let value = model.value_from_hidden_into(&scratch.hidden, &[], &mut scratch.value_head);

        assert!(value.abs() < 1e-6);
    }

    #[test]
    fn incremental_accumulator_matches_full_refresh() {
        let model = AzNnue::random(128, 20260807);
        let mut position = Position::startpos();
        let mut hidden = AzEvalAccumulator::new(&model, &position).into_hidden_sum();
        for _ in 0..32 {
            let mv = position.legal_moves()[0];
            let moved = position.piece_at(mv.from as usize).unwrap();
            let captured = position.piece_at(mv.to as usize);
            let before = position.clone();
            position.make_move(mv);
            AzEvalAccumulator::apply_transition_to_hidden(
                &model,
                &before,
                &position,
                mv,
                moved,
                captured,
                &mut hidden,
            );
            let refreshed = AzEvalAccumulator::new(&model, &position).into_hidden_sum();
            for (&incremental, &full) in hidden.iter().zip(&refreshed) {
                assert!((incremental - full).abs() < 2.0e-5);
            }
        }
    }

    #[test]
    fn quantized_policy_accumulator_matches_full_refresh() {
        let model = AzNnue::random(128, 20260817);
        let mut position = Position::startpos();
        let mut accumulators = [
            model.quantized_policy_accumulator(&position, Color::Red),
            model.quantized_policy_accumulator(&position, Color::Black),
        ];
        for _ in 0..32 {
            let mv = position.legal_moves()[0];
            let moved = position.piece_at(mv.from as usize).unwrap();
            let captured = position.piece_at(mv.to as usize);
            let before = position.clone();
            position.make_move(mv);
            for perspective in [Color::Red, Color::Black] {
                model.apply_quantized_policy_transition(
                    &before,
                    &position,
                    mv,
                    moved,
                    captured,
                    perspective,
                    &mut accumulators[color_index(perspective)],
                );
                assert_eq!(
                    accumulators[color_index(perspective)],
                    model.quantized_policy_accumulator(&position, perspective)
                );
            }
        }
    }

    #[test]
    fn incremental_batch4_matches_scalar_evaluation() {
        let model = AzNnue::random(128, 20260819);
        let position = Position::startpos();
        let moves = position.legal_moves();
        let hidden = AzEvalAccumulator::new(&model, &position).into_hidden_sum();
        let policy = model.quantized_policy_accumulator(&position, position.side_to_move());
        let rule_context = [0.0; RULE_CONTEXT_SIZE];

        let mut scalar_scratch: [AzEvalScratch; 4] =
            std::array::from_fn(|_| AzEvalScratch::new(model.arch));
        let scalar: [AzEvalOutput; 4] = std::array::from_fn(|index| {
            model.evaluate_incremental_with_scratch_output(
                &position,
                &hidden,
                &policy,
                &moves,
                &rule_context,
                &mut scalar_scratch[index],
            )
        });

        let mut batch_scratch = std::array::from_fn(|_| AzEvalScratch::new(model.arch));
        let [scratch0, scratch1, scratch2, scratch3] = &mut batch_scratch;
        let mut requests = [
            AzIncrementalEvalRequest {
                position: &position,
                accumulator_hidden: &hidden,
                policy_accumulator: &policy,
                moves: &moves,
                rule_context: &rule_context,
                scratch: scratch0,
            },
            AzIncrementalEvalRequest {
                position: &position,
                accumulator_hidden: &hidden,
                policy_accumulator: &policy,
                moves: &moves,
                rule_context: &rule_context,
                scratch: scratch1,
            },
            AzIncrementalEvalRequest {
                position: &position,
                accumulator_hidden: &hidden,
                policy_accumulator: &policy,
                moves: &moves,
                rule_context: &rule_context,
                scratch: scratch2,
            },
            AzIncrementalEvalRequest {
                position: &position,
                accumulator_hidden: &hidden,
                policy_accumulator: &policy,
                moves: &moves,
                rule_context: &rule_context,
                scratch: scratch3,
            },
        ];
        let batch = model.evaluate_incremental_batch4(&mut requests);
        for index in 0..4 {
            assert!((scalar[index].value - batch[index].value).abs() < 1.0e-5);
            for (left, right) in scalar_scratch[index]
                .logits
                .iter()
                .zip(&batch_scratch[index].logits)
            {
                assert!((left - right).abs() < 1.0e-5);
            }
        }
    }

    #[test]
    #[ignore = "manual fast-profile batch evaluator benchmark"]
    fn benchmark_incremental_batch4() {
        use std::hint::black_box;
        use std::time::Instant;

        let model = AzNnue::random(128, 20260820);
        let position = Position::startpos();
        let moves = position.legal_moves();
        let hidden = AzEvalAccumulator::new(&model, &position).into_hidden_sum();
        let policy = model.quantized_policy_accumulator(&position, position.side_to_move());
        let rule_context = [0.0; RULE_CONTEXT_SIZE];
        let repeats = 5_000;

        let mut scalar_scratch: [AzEvalScratch; 4] =
            std::array::from_fn(|_| AzEvalScratch::new(model.arch));
        let scalar_started = Instant::now();
        for _ in 0..repeats {
            for scratch in &mut scalar_scratch {
                black_box(model.evaluate_incremental_with_scratch_output(
                    &position,
                    &hidden,
                    &policy,
                    &moves,
                    &rule_context,
                    scratch,
                ));
            }
        }
        let scalar = scalar_started.elapsed();

        let mut batch_scratch = std::array::from_fn(|_| AzEvalScratch::new(model.arch));
        let [scratch0, scratch1, scratch2, scratch3] = &mut batch_scratch;
        let mut requests = [
            AzIncrementalEvalRequest {
                position: &position,
                accumulator_hidden: &hidden,
                policy_accumulator: &policy,
                moves: &moves,
                rule_context: &rule_context,
                scratch: scratch0,
            },
            AzIncrementalEvalRequest {
                position: &position,
                accumulator_hidden: &hidden,
                policy_accumulator: &policy,
                moves: &moves,
                rule_context: &rule_context,
                scratch: scratch1,
            },
            AzIncrementalEvalRequest {
                position: &position,
                accumulator_hidden: &hidden,
                policy_accumulator: &policy,
                moves: &moves,
                rule_context: &rule_context,
                scratch: scratch2,
            },
            AzIncrementalEvalRequest {
                position: &position,
                accumulator_hidden: &hidden,
                policy_accumulator: &policy,
                moves: &moves,
                rule_context: &rule_context,
                scratch: scratch3,
            },
        ];
        let batch_started = Instant::now();
        for _ in 0..repeats {
            black_box(model.evaluate_incremental_batch4(&mut requests));
        }
        let batch = batch_started.elapsed();
        eprintln!(
            "scalar={:.3}ms batch4={:.3}ms speedup={:.3}x",
            scalar.as_secs_f64() * 1e3,
            batch.as_secs_f64() * 1e3,
            scalar.as_secs_f64() / batch.as_secs_f64()
        );
    }

    #[test]
    fn arena_report_relative_elo_tracks_score_and_bounds() {
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
        assert!(stronger.elo_diff_vs_even() > 0.0);
        assert!(weaker.score_rate() < 0.5);
        assert!(weaker.elo_diff_vs_even() < 0.0);
        let (lower, upper) = stronger.elo_diff_bounds(1.96);
        assert!(lower <= stronger.elo_diff_vs_even());
        assert!(upper >= stronger.elo_diff_vs_even());
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
                rule_context: [0.0; RULE_CONTEXT_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value_wdl: scalar_value_to_wdl_target(1.0),
                value: 1.0,
                side_sign: 1.0,
                policy_weight: 1.0,
                value_weight: 1.0,
                search_simulations: 0,
                meta: AzSampleMeta::default(),
            },
            AzTrainingSample {
                features: vec![1],
                rule_context: [0.0; RULE_CONTEXT_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value_wdl: scalar_value_to_wdl_target(-1.0),
                value: -1.0,
                side_sign: 1.0,
                policy_weight: 1.0,
                value_weight: 1.0,
                search_simulations: 0,
                meta: AzSampleMeta::default(),
            },
            AzTrainingSample {
                features: vec![2],
                rule_context: [0.0; RULE_CONTEXT_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value_wdl: scalar_value_to_wdl_target(0.75),
                value: 0.75,
                side_sign: 1.0,
                policy_weight: 1.0,
                value_weight: 1.0,
                search_simulations: 0,
                meta: AzSampleMeta::default(),
            },
            AzTrainingSample {
                features: vec![3],
                rule_context: [0.0; RULE_CONTEXT_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value_wdl: scalar_value_to_wdl_target(-0.75),
                value: -0.75,
                side_sign: 1.0,
                policy_weight: 1.0,
                value_weight: 1.0,
                search_simulations: 0,
                meta: AzSampleMeta::default(),
            },
        ];

        let mut rng = SplitMix64::new(17);
        let before = train_samples(&mut model, &samples, 1, 0.003, 4, &mut rng)
            .unwrap()
            .value_loss;
        let after = train_samples(&mut model, &samples, 300, 0.003, 4, &mut rng)
            .unwrap()
            .value_loss;

        assert!(after < before * 0.5, "before={before} after={after}");
        assert!(after < 0.35, "after={after}");
    }

    #[cfg(feature = "gpu-train")]
    #[test]
    fn batched_training_is_deterministic() {
        let samples = vec![
            AzTrainingSample {
                features: vec![0, 4, 8],
                rule_context: [0.0; RULE_CONTEXT_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value_wdl: scalar_value_to_wdl_target(1.0),
                value: 1.0,
                side_sign: 1.0,
                policy_weight: 1.0,
                value_weight: 1.0,
                search_simulations: 0,
                meta: AzSampleMeta::default(),
            },
            AzTrainingSample {
                features: vec![1, 5, 9],
                rule_context: [0.0; RULE_CONTEXT_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value_wdl: scalar_value_to_wdl_target(-1.0),
                value: -1.0,
                side_sign: 1.0,
                policy_weight: 1.0,
                value_weight: 1.0,
                search_simulations: 0,
                meta: AzSampleMeta::default(),
            },
            AzTrainingSample {
                features: vec![2, 6, 10],
                rule_context: [0.0; RULE_CONTEXT_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value_wdl: scalar_value_to_wdl_target(0.5),
                value: 0.5,
                side_sign: 1.0,
                policy_weight: 1.0,
                value_weight: 1.0,
                search_simulations: 0,
                meta: AzSampleMeta::default(),
            },
            AzTrainingSample {
                features: vec![3, 7, 11],
                rule_context: [0.0; RULE_CONTEXT_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value_wdl: scalar_value_to_wdl_target(-0.5),
                value: -0.5,
                side_sign: 1.0,
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
        let single_stats =
            train_samples(&mut single, &samples, 5, 0.003, 4, &mut rng_single).unwrap();
        let repeated_stats =
            train_samples(&mut repeated, &samples, 5, 0.003, 4, &mut rng_repeated).unwrap();

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
                rule_context: [0.0; RULE_CONTEXT_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value_wdl: scalar_value_to_wdl_target(1.0),
                value: 1.0,
                side_sign: 1.0,
                policy_weight: 1.0,
                value_weight: 1.0,
                search_simulations: 0,
                meta: AzSampleMeta::default(),
            },
            AzTrainingSample {
                features: vec![1, 5, 9],
                rule_context: [0.0; RULE_CONTEXT_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value_wdl: scalar_value_to_wdl_target(-1.0),
                value: -1.0,
                side_sign: 1.0,
                policy_weight: 1.0,
                value_weight: 1.0,
                search_simulations: 0,
                meta: AzSampleMeta::default(),
            },
            AzTrainingSample {
                features: vec![2, 6, 10],
                rule_context: [0.0; RULE_CONTEXT_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value_wdl: scalar_value_to_wdl_target(0.75),
                value: 0.75,
                side_sign: 1.0,
                policy_weight: 1.0,
                value_weight: 1.0,
                search_simulations: 0,
                meta: AzSampleMeta::default(),
            },
            AzTrainingSample {
                features: vec![3, 7, 11],
                rule_context: [0.0; RULE_CONTEXT_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value_wdl: scalar_value_to_wdl_target(-0.75),
                value: -0.75,
                side_sign: 1.0,
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
        train_samples_weighted(&mut model, &samples, 20, 0.01, 4, &mut rng, weights).unwrap();

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
        assert_eq!(model.value_head_output, loaded.value_head_output);
        assert_eq!(model.policy_move_bias, loaded.policy_move_bias);
        assert_eq!(
            model.policy_consequence_output,
            loaded.policy_consequence_output
        );
        assert_eq!(model.policy_context_hidden, loaded.policy_context_hidden);
        assert_eq!(model.policy_move_context, loaded.policy_move_context);
        assert_eq!(
            model.policy_accumulator_hidden,
            loaded.policy_accumulator_hidden
        );
        assert_eq!(
            model.policy_accumulator_move,
            loaded.policy_accumulator_move
        );
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
                rule_context: [0.0; RULE_CONTEXT_SIZE],
                move_indices: vec![0],
                policy: vec![1.0],
                value_wdl: scalar_value_to_wdl_target(0.0),
                value: 0.0,
                side_sign: 1.0,
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
                rule_context: [0.0; RULE_CONTEXT_SIZE],
                move_indices: vec![0],
                policy: vec![1.0],
                value_wdl: scalar_value_to_wdl_target(0.0),
                value: 0.0,
                side_sign: 1.0,
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
        assert!((4..=10).contains(&batch.actual_recent_samples));
    }

    #[test]
    fn replay_recent_games_counts_complete_games_not_generation_batches() {
        fn sample(game_id: u64) -> AzTrainingSample {
            AzTrainingSample {
                features: vec![1],
                rule_context: [0.0; RULE_CONTEXT_SIZE],
                move_indices: vec![0],
                policy: vec![1.0],
                value_wdl: scalar_value_to_wdl_target(0.0),
                value: 0.0,
                side_sign: 1.0,
                policy_weight: 1.0,
                value_weight: 1.0,
                search_simulations: 0,
                meta: AzSampleMeta {
                    generation_update: 7,
                    game_id,
                    ..AzSampleMeta::default()
                },
            }
        }

        let mut pool = AzExperiencePool::new(8);
        for game_id in 1..=4 {
            pool.add_games(vec![vec![sample(game_id)]]);
        }
        let stats = pool.window_stats(2);
        assert_eq!(stats.window_games, 4);
        assert!((stats.recent_window_sample_fraction - 0.5).abs() < 1e-6);

        let mut rng = SplitMix64::new(9);
        let batch = pool.sample_mixed_recent(100, 1.0, 2, &mut rng);
        assert!(batch.samples.iter().all(|sample| sample.meta.game_id >= 3));
        assert_eq!(batch.actual_recent_samples, 100);
    }

    #[test]
    fn rule_context_exposes_repetition_without_history_planes() {
        let position = Position::startpos();
        let entry = position.rule_history_entry(None);
        let context = rule_context_features(&position, &[entry, entry, entry]);

        assert!((context[1] - 2.0 / 3.0).abs() < 1e-6);
        assert!(context[2] > 0.0);
        assert_eq!(context[3..], [0.0; 4]);
    }
}
