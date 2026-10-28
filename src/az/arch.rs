//! 双视角NNUE：红黑各维护一份棋子位置累加器，按行棋方、对方拼接后进入共享主干。
//! 两份视角共用特征表；规则、将军态势与历史通过统一上下文进入共享主干。
//! 共享编码提供主干与威胁关系；价值头负责子力分桶与将位空间关系。
//! 策略由非线性局面上下文、线性局面累加器、将位走法表、战术表和重复项组成。
//! 精确表已覆盖的因子不再单独参数化，训练与推理共用表布局。

use crate::az::nnue::{AZ_NNUE_INPUT_SIZE, V2_KING_BUCKETS, canonical_piece_square_reachable};
use crate::xiangqi::{BOARD_FILES, BOARD_SIZE, Color, Move};

use super::px0_policy_map;

pub(crate) const SPARSE_MOVE_SPACE: usize = BOARD_SIZE * BOARD_SIZE;
pub const DENSE_MOVE_SPACE: usize = 2062;
pub const HISTORY_CONTEXT_SIZE: usize = 96;
pub(crate) const POLICY_MOVE_CONTEXT_SIZE: usize = 32;
pub(crate) const POLICY_THREAT_CONTEXT_SIZE: usize = POLICY_MOVE_CONTEXT_SIZE;
pub(crate) const POLICY_ACCUMULATOR_RANK: usize = 64;
/// 原始六位战术签名的取值数，不是紧凑表的行跨度。
pub(crate) const POLICY_TACTICAL_SIGNATURES: usize = 64;
pub(crate) const POLICY_TACTICAL_TERMS: usize = 2;
/// 几何可行的走法×棋种组合：将24、士8、象16、马508、车1530、炮1530、兵126。
pub(crate) const POLICY_MOVE_PIECE_COUNT: usize = 3_742;
/// 去除恒定控制位、相邻炮吃子和不可能将军的签名。
pub const POLICY_TACTICAL_EXACT_SIZE: usize = 120_128;
pub(crate) const POLICY_CAPTURE_RELATION_OFFSET: usize = POLICY_TACTICAL_EXACT_SIZE;
pub(crate) const POLICY_CAPTURE_RELATION_BUCKETS: usize = 16;
pub(crate) const POLICY_CAPTURE_RELATION_SIZE: usize = 36 * POLICY_CAPTURE_RELATION_BUCKETS;
pub(crate) const POLICY_TACTICAL_SIZE: usize =
    POLICY_CAPTURE_RELATION_OFFSET + POLICY_CAPTURE_RELATION_SIZE;
// 策略只编码己方走子和敌方被吃子，训练与推理使用同一布局。
pub(crate) const POLICY_PIECE_SIZE: usize = 7;
const POLICY_CAPTURE_CLASSES: usize = POLICY_PIECE_SIZE + 1;
pub const POLICY_SPARSE_MAIN_SIZE: usize = 279_603;
pub(crate) const POLICY_SPARSE_CAPTURE_SIZE: usize = 9_799;
pub(crate) const POLICY_SPARSE_TABLE_SIZE: usize =
    POLICY_SPARSE_MAIN_SIZE + POLICY_SPARSE_CAPTURE_SIZE + 1;
pub(crate) const POLICY_SPARSE_TERMS: usize = 2;
pub(crate) const POLICY_ACCUMULATOR_BIAS_ROW: usize = AZ_NNUE_INPUT_SIZE;
pub(crate) const POLICY_ACCUMULATOR_ROWS: usize = POLICY_ACCUMULATOR_BIAS_ROW + 1;
pub(crate) const VALUE_HEAD_SIZE: usize = 96;
pub(crate) const VALUE_BUCKETS: usize = 32;
/// 王棋子价值表先在16维累加，再投影到价值隐藏层。
pub(crate) const VALUE_KING_PIECE_RANK: usize = 16;
/// 仅保留棋子可达且与对应王位不冲突的组合。
pub(crate) const VALUE_KING_PIECE_VOCAB: usize = 12_174;
pub(crate) const VALUE_KING_PIECE_MAX_ACTIVE: usize = 64;
pub(crate) const VALUE_THREAT_RANK: usize = 64;
/// 去除同色双将死槽，并共享155组严格互反关系；访问次数仍各自保留。
pub(crate) const VALUE_THREAT_PAIR_VOCAB: usize = 10_443;
/// 仅保留方向、距离、阻挡次序及数量可同时满足的射线组合。
pub(crate) const VALUE_RAY_VOCAB: usize = 7_040;
pub(crate) const VALUE_CANNON_TRIPLE_VOCAB: usize = 32_768;
pub(crate) const HORSE_TWO_HOP_OFFSET: usize =
    VALUE_THREAT_PAIR_VOCAB + VALUE_RAY_VOCAB + VALUE_CANNON_TRIPLE_VOCAB;
/// 双方马 × 1708个两跳起终点组合 × 空格/敌子。
pub(crate) const HORSE_TWO_HOP_VOCAB: usize = 6_832;
pub(crate) const HORSE_LEG_OFFSET: usize = HORSE_TWO_HOP_OFFSET + HORSE_TWO_HOP_VOCAB;
/// 4368种堵腿关系：548行复用旧配对，492对互反共享，其余分配新行。
pub(crate) const HORSE_LEG_VOCAB: usize = 3_328;
pub(crate) const VALUE_THREAT_VOCAB: usize = HORSE_LEG_OFFSET + HORSE_LEG_VOCAB;
pub(crate) const VALUE_THREAT_MAX_ACTIVE: usize = 336;
pub(crate) const THREAT_SOURCE_MAX_ACTIVE: usize = 44;
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
pub(crate) const CHECK_CONTEXT_OFFSET: usize = RULE_CONTEXT_SIZE;
pub(crate) const HISTORY_CONTEXT_OFFSET: usize = RULE_CONTEXT_SIZE + CHECK_CONTEXT_SIZE;
pub(crate) const POSITION_CONTEXT_SIZE: usize = HISTORY_CONTEXT_OFFSET + HISTORY_CONTEXT_SIZE;
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
    pub(crate) move_piece_to_compact: [[u16; POLICY_PIECE_SIZE]; DENSE_MOVE_SPACE],
    tactical_offsets: Vec<u32>,
    tactical_masks: Vec<u64>,
    king_offsets: Vec<u32>,
    king_masks: Vec<u128>,
    capture_offsets: [u16; DENSE_MOVE_SPACE],
    capture_masks: [u8; DENSE_MOVE_SPACE],
}

pub(crate) fn move_map() -> &'static MoveMap {
    use std::sync::OnceLock;
    static MAP: OnceLock<MoveMap> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut sparse_to_dense = [u16::MAX; SPARSE_MOVE_SPACE];
        let dense_to_sparse = px0_policy_map::PX0_MOVES;
        let mut move_piece_to_compact = [[u16::MAX; POLICY_PIECE_SIZE]; DENSE_MOVE_SPACE];
        let mut tactical_offsets = Vec::with_capacity(POLICY_MOVE_PIECE_COUNT);
        let mut tactical_masks = Vec::with_capacity(POLICY_MOVE_PIECE_COUNT);
        let mut king_offsets = Vec::with_capacity(POLICY_MOVE_PIECE_COUNT);
        let mut king_masks = Vec::with_capacity(POLICY_MOVE_PIECE_COUNT);
        let mut capture_offsets = [0; DENSE_MOVE_SPACE];
        let mut capture_masks = [0; DENSE_MOVE_SPACE];
        let mut pair_count = 0;
        let mut tactical_count = 0;
        let mut king_count = 0;
        let mut capture_count = 0;
        for (index, &sparse) in dense_to_sparse.iter().enumerate() {
            assert_eq!(sparse_to_dense[sparse as usize], u16::MAX);
            sparse_to_dense[sparse as usize] = index as u16;
            let from = sparse as usize / BOARD_SIZE;
            let to = sparse as usize % BOARD_SIZE;
            capture_offsets[index] = capture_count;
            for class in 0..POLICY_CAPTURE_CLASSES {
                if class == 7 || (class != 0 && canonical_piece_square_reachable(class + 7, to)) {
                    capture_masks[index] |= 1 << class;
                    capture_count += 1;
                }
            }
            for piece in 0..POLICY_PIECE_SIZE {
                if policy_move_piece_is_geometric(from, to, piece) {
                    move_piece_to_compact[index][piece] = pair_count as u16;
                    pair_count += 1;
                    tactical_offsets.push(tactical_count);
                    let can_check = policy_check_dependency_possible(from, to, piece);
                    let adjacent = (from / BOARD_FILES).abs_diff(to / BOARD_FILES)
                        + (from % BOARD_FILES).abs_diff(to % BOARD_FILES)
                        == 1;
                    let mut mask = 0u64;
                    for signature in 0..POLICY_TACTICAL_SIGNATURES {
                        let capture = signature & 16 != 0;
                        if ((piece != 5 || capture) && signature & 8 == 0)
                            || (piece == 5 && adjacent && capture)
                            || (signature & 32 != 0 && !can_check)
                        {
                            continue;
                        }
                        mask |= 1 << signature;
                    }
                    tactical_count += mask.count_ones();
                    tactical_masks.push(mask);
                    let mut mask = 0u128;
                    for us in 0..V2_KING_BUCKETS {
                        for them in 0..V2_KING_BUCKETS {
                            if policy_kings_possible(from, to, piece, us, them) {
                                mask |= 1 << (us * V2_KING_BUCKETS + them);
                            }
                        }
                    }
                    king_offsets.push(king_count);
                    king_count += mask.count_ones();
                    king_masks.push(mask);
                }
            }
        }
        assert_eq!(pair_count, POLICY_MOVE_PIECE_COUNT);
        assert_eq!(tactical_count as usize, POLICY_TACTICAL_EXACT_SIZE);
        assert_eq!(king_count as usize, POLICY_SPARSE_MAIN_SIZE);
        assert_eq!(capture_count as usize, POLICY_SPARSE_CAPTURE_SIZE);
        MoveMap {
            sparse_to_dense,
            dense_to_sparse,
            move_piece_to_compact,
            tactical_offsets,
            tactical_masks,
            king_offsets,
            king_masks,
            capture_offsets,
            capture_masks,
        }
    })
}

/// 仅编码走前攻击位板下可能出现的签名：非炮及炮吃子必然控制目标格。
pub(crate) fn policy_tactical_exact_index(
    move_index: usize,
    piece: usize,
    signature: usize,
) -> Option<usize> {
    let map = move_map();
    let pair = policy_move_piece_index(move_index, piece)?;
    if signature >= POLICY_TACTICAL_SIGNATURES {
        return None;
    }
    let mask = map.tactical_masks[pair];
    let bit = 1u64 << signature;
    (mask & bit != 0)
        .then(|| map.tactical_offsets[pair] as usize + (mask & (bit - 1)).count_ones() as usize)
}

/// 位掩码与前缀计数避免维护完整的走法×棋种×双将位索引大表。
pub(crate) fn policy_main_compact_index(
    mv: usize,
    piece: usize,
    us: usize,
    them: usize,
) -> Option<usize> {
    let pair = policy_move_piece_index(mv, piece)?;
    let map = move_map();
    let bit = 1u128 << (us * V2_KING_BUCKETS + them);
    let mask = map.king_masks[pair];
    (mask & bit != 0)
        .then(|| map.king_offsets[pair] as usize + (mask & (bit - 1)).count_ones() as usize)
}

pub(crate) fn policy_capture_compact_index(mv: usize, victim: Option<usize>) -> Option<usize> {
    debug_assert!(victim.is_none_or(|piece| (7..14).contains(&piece)));
    let class = victim.map_or(7, |p| p - 7);
    let map = move_map();
    let bit = 1u8 << class;
    let mask = map.capture_masks[mv];
    (mask & bit != 0).then(|| {
        POLICY_SPARSE_MAIN_SIZE
            + map.capture_offsets[mv] as usize
            + (mask & (bit - 1)).count_ones() as usize
    })
}

pub(crate) fn policy_capture_relation_index(
    piece: usize,
    victim: usize,
    signature: usize,
) -> Option<usize> {
    debug_assert!(piece < POLICY_PIECE_SIZE && signature < POLICY_TACTICAL_SIGNATURES);
    if !(8..14).contains(&victim) || (piece < 3 && victim < 10) {
        return None;
    }
    let pair = if piece < 3 {
        piece * 4 + victim - 10
    } else {
        12 + (piece - 3) * 6 + victim - 8
    };
    Some(
        POLICY_CAPTURE_RELATION_OFFSET
            + pair * POLICY_CAPTURE_RELATION_BUCKETS
            + (signature & 7)
            + ((signature >> 5) << 3),
    )
}

fn king_square(bucket: usize) -> usize {
    (7 + bucket / 3) * BOARD_FILES + 3 + bucket % 3
}

fn policy_kings_possible(from: usize, to: usize, piece: usize, us: usize, them: usize) -> bool {
    let us = king_square(us);
    let them = BOARD_SIZE - 1 - king_square(them);
    if (piece == 0 && from != us)
        || (piece != 0 && (us == from || us == to))
        || them == from
        || them == to
    {
        return false;
    }
    let (sr, sf) = (from / BOARD_FILES, from % BOARD_FILES);
    let (tr, tf) = (to / BOARD_FILES, to % BOARD_FILES);
    if piece == 4 || piece == 5 {
        let blockers = [us, them]
            .into_iter()
            .filter(|&k| {
                let (kr, kf) = (k / BOARD_FILES, k % BOARD_FILES);
                (sr == tr && tr == kr && sf.min(tf) < kf && kf < sf.max(tf))
                    || (sf == tf && tf == kf && sr.min(tr) < kr && kr < sr.max(tr))
            })
            .count();
        return blockers <= usize::from(piece == 5);
    }
    let block = match piece {
        2 => Some((sr + tr) / 2 * BOARD_FILES + (sf + tf) / 2),
        3 if sr.abs_diff(tr) == 2 => Some((sr + tr) / 2 * BOARD_FILES + sf),
        3 => Some(sr * BOARD_FILES + (sf + tf) / 2),
        _ => None,
    };
    !block.is_some_and(|sq| sq == us || sq == them)
}

/// 必要条件：将位附近的射线/马腿占用变化，或移动马兵直接将军。
fn policy_check_dependency_possible(from: usize, to: usize, piece: usize) -> bool {
    let (sr, sf) = (from / BOARD_FILES, from % BOARD_FILES);
    let (tr, tf) = (to / BOARD_FILES, to % BOARD_FILES);
    (0..V2_KING_BUCKETS).any(|bucket| {
        let king = BOARD_SIZE - 1 - king_square(bucket);
        if king == from || king == to {
            return false;
        }
        let (kr, kf) = (king / BOARD_FILES, king % BOARD_FILES);
        sr == kr
            || sf == kf
            || tr == kr
            || tf == kf
            || (sr.abs_diff(kr) == 1 && sf.abs_diff(kf) == 1)
            || (piece == 3 && matches!((tr.abs_diff(kr), tf.abs_diff(kf)), (1, 2) | (2, 1)))
            || (piece == 6
                && ((tr == kr + 1 && tf == kf) || (tr <= 4 && tr == kr && tf.abs_diff(kf) == 1)))
    })
}

/// 不可能的走法×棋种没有参数行；训练与推理共用同一紧凑编号。
#[inline]
pub(crate) fn policy_move_piece_index(move_index: usize, piece: usize) -> Option<usize> {
    let compact = move_map().move_piece_to_compact[move_index][piece];
    (compact != u16::MAX).then_some(compact as usize)
}

fn policy_move_piece_is_geometric(from: usize, to: usize, piece: usize) -> bool {
    if from == to
        || !canonical_piece_square_reachable(piece, from)
        || !canonical_piece_square_reachable(piece, to)
    {
        return false;
    }
    let rank = from / BOARD_FILES;
    let target_rank = to / BOARD_FILES;
    let dr = rank.abs_diff(target_rank);
    let df = (from % BOARD_FILES).abs_diff(to % BOARD_FILES);
    match piece {
        0 => dr + df == 1,
        1 => dr == 1 && df == 1,
        2 => dr == 2 && df == 2,
        3 => matches!((dr, df), (1, 2) | (2, 1)),
        4 | 5 => dr == 0 || df == 0,
        6 => (rank == target_rank + 1 && df == 0) || (rank <= 4 && dr == 0 && df == 1),
        _ => unreachable!(),
    }
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
        $visit!(context_hidden, [POSITION_CONTEXT_SIZE, 2 * $h], true);
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
        $visit!(moves_left_output, [1, $h], true);
        $visit!(moves_left_bias, [1], false);
        $visit!(
            threat_embedding,
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
