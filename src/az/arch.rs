//! 双视角NNUE：红黑各维护一份棋子位置累加器，按行棋方、对方拼接后进入共享主干。
//! 两份视角共用特征表；规则与将军标量作用于拼接表示。
//! 价值头负责子力分桶、将位空间关系与历史；威胁嵌入同时服务价值和策略。
//! 策略由非线性局面上下文、线性走后累加器、将位走法表、战术表和重复项组成。
//! 精确表已覆盖的因子不再单独参数化，训练与推理共用表布局。

use crate::az::nnue::{AZ_NNUE_INPUT_SIZE, V2_KING_BUCKETS};
use crate::xiangqi::{BOARD_FILES, BOARD_SIZE, Color, Move};

use super::px0_policy_map;

pub(crate) const SPARSE_MOVE_SPACE: usize = BOARD_SIZE * BOARD_SIZE;
pub const DENSE_MOVE_SPACE: usize = 2062;
pub const HISTORY_CONTEXT_SIZE: usize = 96;
pub(crate) const POLICY_MOVE_CONTEXT_SIZE: usize = 16;
pub(crate) const POLICY_THREAT_CONTEXT_SIZE: usize = 16;
pub(crate) const POLICY_ACCUMULATOR_RANK: usize = 64;
pub(crate) const POLICY_TACTICAL_SIGNATURE_BUCKETS: usize = 64;
pub(crate) const POLICY_TACTICAL_TERMS: usize = 2;
pub const POLICY_TACTICAL_EXACT_SIZE: usize =
    DENSE_MOVE_SPACE * POLICY_PIECE_SIZE * POLICY_TACTICAL_SIGNATURE_BUCKETS;
pub(crate) const POLICY_CAPTURE_RELATION_OFFSET: usize = POLICY_TACTICAL_EXACT_SIZE;
pub(crate) const POLICY_CAPTURE_RELATION_BUCKETS: usize = 32;
pub(crate) const POLICY_CAPTURE_RELATION_SIZE: usize = 7 * 7 * POLICY_CAPTURE_RELATION_BUCKETS;
pub(crate) const POLICY_TACTICAL_SIZE: usize =
    POLICY_CAPTURE_RELATION_OFFSET + POLICY_CAPTURE_RELATION_SIZE;
// 策略只编码己方走子和敌方被吃子，训练与推理使用同一布局。
pub(crate) const POLICY_PIECE_SIZE: usize = 7;
pub(crate) const POLICY_CAPTURE_CLASSES: usize = POLICY_PIECE_SIZE + 1;
pub const POLICY_SPARSE_MAIN_SIZE: usize =
    DENSE_MOVE_SPACE * POLICY_PIECE_SIZE * V2_KING_BUCKETS * V2_KING_BUCKETS;
pub(crate) const POLICY_SPARSE_TABLE_SIZE: usize =
    POLICY_SPARSE_MAIN_SIZE + DENSE_MOVE_SPACE * POLICY_CAPTURE_CLASSES + 1;
pub(crate) const POLICY_SPARSE_TERMS: usize = 2;
pub(crate) const POLICY_ACCUMULATOR_BIAS_ROW: usize = AZ_NNUE_INPUT_SIZE;
pub(crate) const POLICY_ACCUMULATOR_ROWS: usize = POLICY_ACCUMULATOR_BIAS_ROW + 1;
pub(crate) const VALUE_HEAD_SIZE: usize = 96;
pub(crate) const VALUE_BUCKETS: usize = 32;
/// 王棋子价值表先在16维累加，再投影到价值隐藏层。
pub(crate) const VALUE_KING_PIECE_RANK: usize = 16;
pub(crate) const VALUE_KING_PIECE_VOCAB: usize = 2 * V2_KING_BUCKETS * 14 * BOARD_SIZE;
pub(crate) const VALUE_KING_PIECE_MAX_ACTIVE: usize = 64;
pub(crate) const VALUE_THREAT_RANK: usize = 64;
pub(crate) const VALUE_THREAT_PAIR_VOCAB: usize = 10_646;
pub(crate) const VALUE_RAY_VOCAB: usize = 4 * 2 * 4 * 9 * 15 * 4;
pub(crate) const VALUE_CANNON_TRIPLE_VOCAB: usize = 32_768;
pub(crate) const VALUE_THREAT_VOCAB: usize =
    VALUE_THREAT_PAIR_VOCAB + VALUE_RAY_VOCAB + VALUE_CANNON_TRIPLE_VOCAB;
pub(crate) const VALUE_THREAT_MAX_ACTIVE: usize = 192;
pub(crate) const WDL_HEAD_SIZE: usize = 3;
/// Small, exact-history-derived signals.  These deliberately replace the old
/// high-dimensional history planes: rules stay in the environment, while the
/// network only gets enough context to recognize an approaching repetition.
pub const RULE_CONTEXT_SIZE: usize = 7;
/// 引擎**已经算过、却一直没喂给模型**的那几个信号的维度。
///
/// 都是每节点一次就能拿到的量：`is_in_check`（搜索本来就在算，算完就扔）、将军着法数
/// （`policy_gives_check` 本来就无条件算）、双方将的安全逃格数 / 九宫被攻击格数
/// （都只是对已经算好的 `attacked_squares_masks` 做几次位测 + popcount）、机动性。
/// 归一化后作为标量块加到主干上，用来给"杀势"这类整体态势一个直接入口。
pub const CHECK_CONTEXT_SIZE: usize = 8;
#[cfg_attr(not(feature = "gpu-train"), allow(dead_code))]
pub(crate) const RMS_NORM_EPS: f32 = 1.0e-6;
pub(crate) const PIECE_SQUARE_INPUT_SIZE: usize = BOARD_SIZE * 14;
pub(crate) const STRUCTURAL_PIECE_SIZE: usize = 14;

#[derive(Clone, Copy, Debug)]
pub(crate) struct StructuralPieceSquare {
    pub piece_index: usize,
    pub rank: usize,
    pub file: usize,
}

pub(crate) fn decode_current_piece_square_feature(feature: usize) -> Option<StructuralPieceSquare> {
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

pub(crate) fn canonical_general_buckets_from_features(features: &[usize]) -> (usize, usize) {
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

pub(crate) fn canonical_general_bucket(piece_index: usize, sq: usize) -> usize {
    let oriented_sq = if piece_index < 7 {
        sq
    } else {
        BOARD_SIZE - 1 - sq
    };
    let file = (oriented_sq % BOARD_FILES).clamp(3, 5) - 3;
    let rank = (oriented_sq / BOARD_FILES).clamp(7, 9) - 7;
    rank * 3 + file
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AzNnueArch {
    pub hidden_size: usize,
}

impl AzNnueArch {
    pub const fn default_const() -> Self {
        Self { hidden_size: 96 }
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

#[inline(always)]
pub(crate) fn canonical_square_for(perspective: Color, sq: usize) -> usize {
    if perspective == Color::Red {
        sq
    } else {
        BOARD_SIZE - 1 - sq
    }
}

pub(crate) struct MoveMap {
    pub(crate) sparse_to_dense: [u16; SPARSE_MOVE_SPACE],
    #[allow(dead_code)]
    pub(crate) dense_to_sparse: [u16; DENSE_MOVE_SPACE],
}

pub(crate) fn move_map() -> &'static MoveMap {
    use std::sync::OnceLock;
    static MAP: OnceLock<MoveMap> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut sparse_to_dense = [u16::MAX; SPARSE_MOVE_SPACE];
        let dense_to_sparse = px0_policy_map::PX0_MOVES;
        for (index, &sparse) in dense_to_sparse.iter().enumerate() {
            assert_eq!(sparse_to_dense[sparse as usize], u16::MAX);
            sparse_to_dense[sparse as usize] = index as u16;
        }
        MoveMap {
            sparse_to_dense,
            dense_to_sparse,
        }
    })
}

pub(crate) fn dense_move_squares(move_index: usize) -> Option<(usize, usize)> {
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

/// MLH 使用半回合；归一化用于稳定回归梯度。
pub(crate) const MOVES_LEFT_SCALE: f32 = 100.0;
pub(crate) const MOVES_LEFT_LOSS_WEIGHT: f32 = 0.05;
/// 剩余步数对搜索的影响；默认参数来自 PX0/LC0 classic 搜索。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AzMovesLeftParams {
    pub enabled: bool,
    pub threshold: f32,
    pub max_effect: f32,
    pub slope: f32,
    pub constant_factor: f32,
    pub scaled_factor: f32,
    pub quadratic_factor: f32,
}

impl Default for AzMovesLeftParams {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: 0.8,
            max_effect: 0.0345,
            slope: 0.0027,
            constant_factor: 0.0,
            scaled_factor: 1.6521,
            quadratic_factor: -0.6521,
        }
    }
}

impl AzMovesLeftParams {
    pub fn normalize(mut self) -> Self {
        fn bounded(value: f32, min: f32, max: f32, fallback: f32) -> f32 {
            if value.is_finite() {
                value.clamp(min, max)
            } else {
                fallback
            }
        }
        let defaults = Self::default();
        self.threshold = bounded(self.threshold, 0.0, 1.0, defaults.threshold);
        self.max_effect = bounded(self.max_effect, 0.0, 1.0, defaults.max_effect);
        self.slope = bounded(self.slope, 0.0, 1.0, defaults.slope);
        self.constant_factor = bounded(self.constant_factor, -1.0, 1.0, defaults.constant_factor);
        self.scaled_factor = bounded(self.scaled_factor, -2.0, 2.0, defaults.scaled_factor);
        self.quadratic_factor =
            bounded(self.quadratic_factor, -1.0, 1.0, defaults.quadratic_factor);
        self
    }
}

// 参数形状及是否施加稠密权重衰减的唯一清单。
macro_rules! az_weight_tensors {
    ($visit:ident, $h:expr) => {
        $visit!(input_hidden, [AZ_NNUE_INPUT_SIZE, $h], false);
        $visit!(rule_context_hidden, [RULE_CONTEXT_SIZE, 2 * $h], true);
        $visit!(check_context_hidden, [CHECK_CONTEXT_SIZE, 2 * $h], true);
        $visit!(hidden_bias, [$h], false);
        $visit!(shared_hidden, [$h, 2 * $h], true);
        $visit!(shared_bias, [$h], false);
        $visit!(value_head_hidden, [VALUE_HEAD_SIZE, $h], true);
        $visit!(value_head_bias, [VALUE_HEAD_SIZE], false);
        $visit!(
            value_king_piece_hidden,
            [VALUE_KING_PIECE_VOCAB, VALUE_KING_PIECE_RANK],
            false
        );
        $visit!(
            value_king_piece_projection,
            [VALUE_KING_PIECE_RANK, VALUE_HEAD_SIZE],
            true
        );
        $visit!(
            value_head_output,
            [VALUE_BUCKETS, WDL_HEAD_SIZE, VALUE_HEAD_SIZE],
            true
        );
        $visit!(
            value_history_output,
            [WDL_HEAD_SIZE, HISTORY_CONTEXT_SIZE],
            true
        );
        $visit!(moves_left_output, [1, $h], true);
        $visit!(moves_left_bias, [1], false);
        $visit!(
            value_threat_embedding,
            [VALUE_THREAT_VOCAB, VALUE_THREAT_RANK],
            false
        );
        $visit!(
            value_threat_output,
            [WDL_HEAD_SIZE, VALUE_THREAT_RANK * 2],
            true
        );
        $visit!(
            policy_threat_context,
            [POLICY_THREAT_CONTEXT_SIZE, VALUE_THREAT_RANK * 2],
            true
        );
        $visit!(policy_move_bias, [DENSE_MOVE_SPACE], false);
        $visit!(policy_context_hidden, [POLICY_MOVE_CONTEXT_SIZE, $h], true);
        $visit!(
            policy_move_context,
            [DENSE_MOVE_SPACE, POLICY_MOVE_CONTEXT_SIZE],
            false
        );
        $visit!(
            policy_accumulator_hidden,
            [POLICY_ACCUMULATOR_RANK, $h],
            true
        );
        $visit!(
            policy_accumulator_move,
            [DENSE_MOVE_SPACE, POLICY_ACCUMULATOR_RANK],
            false
        );
        $visit!(policy_sparse_table, [POLICY_SPARSE_TABLE_SIZE], false);
        $visit!(policy_tactical, [POLICY_TACTICAL_SIZE], false);
        $visit!(policy_repetition_hidden, [$h], true);
        $visit!(policy_repetition_bias, [1], false);
    };
}

pub(crate) use az_weight_tensors;
