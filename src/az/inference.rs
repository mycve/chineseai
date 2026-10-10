use std::io;
use std::path::Path;

use candle_core::{DType, Device, Shape, Var};
use candle_nn::VarMap;

use crate::az::nnue::{
    AZ_NNUE_INPUT_SIZE, V2_KING_BUCKETS, canonical_move, canonical_square, fill_sparse_features_az,
    piece_absolute_feature_index,
};
use crate::infra::version::MODEL_FORMAT_VERSION;
use crate::xiangqi::{
    BOARD_FILES, BOARD_RANKS, BOARD_SIZE, Color, Move, Piece, PieceKind, Position, color_index,
    piece_kind_index,
};

use super::*;

pub(crate) fn candle_io_error(err: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err.to_string())
}

pub(crate) fn insert_candle_var(
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

pub(crate) fn load_candle_f32_tensor(
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
        $visit!(check_context_hidden, [CHECK_CONTEXT_SIZE, $h]);
        $visit!(hidden_bias, [$h]);
        $visit!(shared_hidden, [$h, $h]);
        $visit!(shared_bias, [$h]);
        $visit!(value_head_hidden, [VALUE_HEAD_SIZE, $h]);
        $visit!(value_head_bias, [VALUE_HEAD_SIZE]);
        $visit!(
            value_king_piece_hidden,
            [VALUE_KING_PIECE_VOCAB, VALUE_KING_PIECE_RANK]
        );
        $visit!(
            value_king_piece_projection,
            [VALUE_KING_PIECE_RANK, VALUE_HEAD_SIZE]
        );
        $visit!(value_head_output, [WDL_HEAD_SIZE, VALUE_HEAD_SIZE]);
        $visit!(value_history_output, [WDL_HEAD_SIZE, HISTORY_CONTEXT_SIZE]);
        $visit!(moves_left_output, [1, $h]);
        $visit!(moves_left_bias, [1]);
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
        $visit!(policy_repetition_hidden, [$h]);
        $visit!(policy_repetition_bias, [1]);
    };
}

pub(crate) fn visit_value_king_piece_features(
    position: &Position,
    perspective: Color,
    mut visitor: impl FnMut(usize),
) {
    let buckets = canonical_buckets_for_perspective(position, perspective);
    for square in 0..BOARD_SIZE {
        let Some(piece) = position.piece_at(square) else {
            continue;
        };
        let piece_index = piece_absolute_feature_index(perspective, piece);
        let canonical = canonical_square_for(perspective, square);
        for (king_side, bucket) in [(0, buckets.0), (1, buckets.1)] {
            visitor(
                ((king_side * V2_KING_BUCKETS + bucket) * 14 + piece_index) * BOARD_SIZE
                    + canonical,
            );
        }
    }
}

pub(crate) fn threat_relation_map() -> &'static [u32] {
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
        assert_eq!(next as usize, VALUE_THREAT_PAIR_VOCAB);
        map
    })
}

pub(crate) fn threat_reachable_square(piece: usize, square: usize) -> bool {
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

pub(crate) fn threat_attacked_offsets() -> &'static [u8] {
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
pub(crate) fn value_threat_index(
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
        return VALUE_THREAT_PAIR_VOCAB;
    }
    let offset = threat_attacked_offsets()[target * STRUCTURAL_PIECE_SIZE + attacked];
    if offset == u8::MAX {
        return VALUE_THREAT_PAIR_VOCAB;
    }
    base as usize + usize::from(offset)
}

pub(crate) fn visit_value_threat_features(
    position: &Position,
    perspective: Color,
    mut visitor: impl FnMut(usize),
) {
    position.visit_occupied_relations(|source, attacker, target, attacked| {
        if matches!(attacker.kind, PieceKind::Rook | PieceKind::Cannon) {
            return;
        }
        let feature = value_threat_index(perspective, source, attacker, target, attacked);
        if feature != VALUE_THREAT_PAIR_VOCAB {
            visitor(feature);
        }
    });

    for source in 0..BOARD_SIZE {
        let Some(attacker) = position.piece_at(source) else {
            continue;
        };
        if !matches!(attacker.kind, PieceKind::Rook | PieceKind::Cannon) {
            continue;
        }
        let source_file = (source % BOARD_FILES) as i32;
        let source_rank = (source / BOARD_FILES) as i32;
        for (df, dr) in [(0, -1), (0, 1), (-1, 0), (1, 0)] {
            let mut first = None;
            let mut second = None;
            let mut blocker_count = 0usize;
            let (mut file, mut rank) = (source_file + df, source_rank + dr);
            while (0..BOARD_FILES as i32).contains(&file) && (0..BOARD_RANKS as i32).contains(&rank)
            {
                let square = rank as usize * BOARD_FILES + file as usize;
                if let Some(piece) = position.piece_at(square) {
                    blocker_count += 1;
                    if first.is_none() {
                        first = Some((square, piece));
                    } else if second.is_none() {
                        second = Some((square, piece));
                    }
                }
                file += df;
                rank += dr;
            }
            let owner = usize::from(attacker.color != perspective);
            let slider = usize::from(attacker.kind == PieceKind::Cannon);
            let attacker_class = owner * 2 + slider;
            if blocker_count == 0 {
                let (canonical_df, canonical_dr) = if perspective == Color::Red {
                    (df, dr)
                } else {
                    (-df, -dr)
                };
                let direction = if canonical_df == 0 {
                    usize::from(canonical_dr > 0)
                } else {
                    2 + usize::from(canonical_df > 0)
                };
                let ray = ((((attacker_class * 2) * 4 + direction) * 9) * 15 + 14) * 4;
                visitor(VALUE_THREAT_PAIR_VOCAB + ray);
                continue;
            }
            let ray_state = blocker_count.min(3);
            let canonical_source = canonical_square(perspective, source);
            for (ordinal, (square, blocked)) in [first, second].into_iter().flatten().enumerate() {
                let canonical_target = canonical_square(perspective, square);
                let sf = canonical_source % BOARD_FILES;
                let sr = canonical_source / BOARD_FILES;
                let tf = canonical_target % BOARD_FILES;
                let tr = canonical_target / BOARD_FILES;
                let direction = if tf == sf {
                    usize::from(tr > sr)
                } else {
                    2 + usize::from(tf > sf)
                };
                let distance = sf.abs_diff(tf).max(sr.abs_diff(tr)) - 1;
                let blocked_class = (if blocked.color == perspective { 0 } else { 7 })
                    + piece_kind_index(blocked.kind);
                let ray = (((((attacker_class * 2 + ordinal) * 4 + direction) * 9 + distance)
                    * 15
                    + blocked_class)
                    * 4)
                    + ray_state;
                visitor(VALUE_THREAT_PAIR_VOCAB + ray);

                if attacker.kind == PieceKind::Cannon && ordinal == 1 {
                    let (screen_source, screen) = first.expect("second blocker requires first");
                    let screen_square = canonical_square(perspective, screen_source);
                    let screen_distance = (canonical_source % BOARD_FILES)
                        .abs_diff(screen_square % BOARD_FILES)
                        .max(
                            (canonical_source / BOARD_FILES).abs_diff(screen_square / BOARD_FILES),
                        )
                        - 1;
                    let screen_class = (if screen.color == perspective { 0 } else { 7 })
                        + piece_kind_index(screen.kind);
                    let target_class = blocked_class;
                    let mut triple = canonical_source as u64;
                    for field in [
                        owner,
                        direction,
                        screen_class,
                        screen_distance,
                        target_class,
                        distance,
                    ] {
                        triple = triple
                            .wrapping_mul(0x9E37_79B1_85EB_CA87)
                            .wrapping_add(field as u64 + 0xC2B2_AE3D_27D4_EB4F);
                        triple ^= triple >> 29;
                    }
                    let triple = triple as usize & (VALUE_CANNON_TRIPLE_VOCAB - 1);
                    visitor(VALUE_THREAT_PAIR_VOCAB + VALUE_RAY_VOCAB + triple);
                }
            }
        }
    }
}

#[inline]
pub(crate) fn policy_consequence_features(
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
pub(crate) fn policy_cache_main_index(mv: usize, piece: usize, us: usize, them: usize) -> usize {
    debug_assert!(piece < POLICY_CACHE_PIECE_SIZE);
    ((mv * POLICY_CACHE_PIECE_SIZE + piece) * V2_KING_BUCKETS + us) * V2_KING_BUCKETS + them
}

#[inline]
pub(crate) fn policy_cache_capture_index(mv: usize, captured: Option<usize>) -> usize {
    let class = captured.map_or(POLICY_CACHE_PIECE_SIZE, |piece| {
        debug_assert!((POLICY_CACHE_PIECE_SIZE..STRUCTURAL_PIECE_SIZE).contains(&piece));
        piece - POLICY_CACHE_PIECE_SIZE
    });
    POLICY_CACHE_MAIN_SIZE + mv * POLICY_CACHE_CAPTURE_CLASSES + class
}

#[inline]
pub(crate) const fn policy_sparse_main_index(
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
pub(crate) const fn policy_sparse_capture_index(
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
pub(crate) fn policy_sparse_factor_indices(
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
pub(crate) fn policy_tactical_indices(
    move_index: usize,
    moved_piece: usize,
    source_attacked: bool,
    destination_attacked: bool,
    source_defended: bool,
    destination_defended: bool,
    captured_piece: Option<usize>,
    check: bool,
) -> [usize; POLICY_TACTICAL_TERMS] {
    debug_assert!(moved_piece < STRUCTURAL_PIECE_SIZE / 2);
    let signature = usize::from(source_attacked)
        | usize::from(destination_attacked) << 1
        | usize::from(source_defended) << 2
        | usize::from(destination_defended) << 3
        | usize::from(captured_piece.is_some()) << 4
        | usize::from(check) << 5;
    let exact = (move_index * (STRUCTURAL_PIECE_SIZE / 2) + moved_piece)
        * POLICY_TACTICAL_SIGNATURE_BUCKETS
        + signature;
    let piece_factor =
        POLICY_TACTICAL_EXACT_SIZE + moved_piece * POLICY_TACTICAL_SIGNATURE_BUCKETS + signature;
    let relation = captured_piece.map_or(POLICY_TACTICAL_SIZE, |victim| {
        debug_assert!((7..14).contains(&victim));
        // 吃子位恒为 1，去掉它；其余五位沿用已有战术状态。
        let state = (signature & 15) | ((signature >> 5) << 4);
        POLICY_CAPTURE_RELATION_OFFSET
            + (moved_piece * 7 + victim - 7) * POLICY_CAPTURE_RELATION_BUCKETS
            + state
    });
    [exact, piece_factor, relation]
}

/// 策略头战术签名用到的 4 个布尔量，返回顺序与 `policy_tactical_indices` 的入参一致：
/// `(source_attacked, destination_attacked, source_defended, destination_defended)`。
///
/// 四个位都用**走前**攻击位板，语义要按"现在"读，不要按"走后"读：
/// - `source_attacked/defended` = 我此刻站着的这格是否被攻击/被保护；
/// - `destination_attacked/defended` = **此刻**落点这一格是否被攻击/被保护。
///
/// 注意第二组**不是**"落子之后那枚子是否被攻击/被保护"：两者实测约 **21%** 不一致，
/// 而且差异是系统性的、不是罕见边角——沿直线走子时走子方自己就攻击着落点（车/炮/兵
/// 沿线移动），于是走前 `defended` 必然为真，走后却只取决于有没有**别的**子保护它；
/// 此外 `from` 腾空还会让炮失去炮架、让车线打开、松开马腿/象眼。
///
/// 明知有约 21% 的语义差仍用近似，是因为精确值要按每个候选走法各查两次，实测
/// **−25%~−39% NPS**（startpos 201k → 123k、中局 111k → 83k sims/s）。训练侧与推理侧
/// 共用这一个实现，所以模型学到的就是"我们控制这一格"这个自洽信号。精确查询留在
/// `Position::is_square_attacked_after_move`，只作审计/测试口径
/// （见 `tactical_flags_are_pre_move_and_the_gap_is_audited`），生产路径不调用它。
#[inline]
pub(crate) fn policy_move_tactical_flags(
    mv: Move,
    opponent_attacks: u128,
    own_attacks: u128,
) -> (bool, bool, bool, bool) {
    let from = mv.from as usize;
    let to = mv.to as usize;
    let source_attacked = opponent_attacks & (1u128 << from) != 0;
    let source_defended = own_attacks & (1u128 << from) != 0;
    let destination_attacked = opponent_attacks & (1u128 << to) != 0;
    let destination_defended = own_attacks & (1u128 << to) != 0;
    (
        source_attacked,
        destination_attacked,
        source_defended,
        destination_defended,
    )
}

pub(crate) fn policy_king_distance_buckets(
    move_index: usize,
    them_king_bucket: usize,
) -> (usize, usize) {
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
    /// "引擎已经算过、却没喂给模型"的标量块（见 `CHECK_CONTEXT_SIZE`）。
    ///
    /// 全零初始化；训练后非零时启用。
    pub check_context_hidden: Vec<f32>,
    pub hidden_bias: Vec<f32>,
    pub shared_hidden: Vec<f32>,
    pub shared_bias: Vec<f32>,
    pub value_head_hidden: Vec<f32>,
    pub value_head_bias: Vec<f32>,
    pub value_king_piece_hidden: Vec<f32>,
    pub value_king_piece_projection: Vec<f32>,
    pub value_head_output: Vec<f32>,
    /// 当前48维空间摘要与最近两步48维变化的 WDL 旁路，布局为 [3, 96]。
    pub value_history_output: Vec<f32>,
    pub(crate) value_history_cache: super::history::ValueHistoryCache,
    pub moves_left_output: Vec<f32>,
    pub moves_left_bias: Vec<f32>,
    pub(crate) moves_left_active: bool,
    pub moves_left_params: AzMovesLeftParams,
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
    pub policy_repetition_hidden: Vec<f32>,
    pub policy_repetition_bias: Vec<f32>,
    pub(crate) policy_accumulator_features: Vec<f32>,
    pub(crate) policy_accumulator_moved_delta: Vec<f32>,
    pub(crate) policy_accumulator_capture: Vec<f32>,
    pub(crate) policy_sparse_table_folded: Vec<f32>,
    pub(crate) policy_tactical_folded: Vec<f32>,
    pub(crate) value_threat_active: bool,
    pub(crate) policy_tactical_active: bool,
    /// `check_context_hidden` 是否非零（全零时跳过）。
    pub(crate) check_context_active: bool,
    /// 根节点 check-only 连杀证明搜索的最大半回合数（0 = 关闭）。
    ///
    /// 挂在模型上而不是 `AzSearchLimits` 上，是因为后者的结构体字面量全仓有 41 处，
    /// 而它只在根节点用一次、与网络质量无关，属于"这一次搜索肯花多少额外预算"。
    /// 打开后：证明出连杀就直接把该子局面标成对方必败，`root_policy` 会把策略目标
    /// 压到杀着上、`proven_root_value` 会把价值目标设为必胜。
    ///
    /// 代价：每次根搜索多一次受限搜索。**实测**（`az-bench best.safetensors 800 5 1.4`，
    /// ms/search 取 5 次均值）：
    /// - 根局面**没有**将军着法（startpos）：4.08 → 3.99ms，无可测代价（只多一遍 check 过滤）；
    /// - 有将军但**没杀**（中局）：7.31 → 7.59ms（plies=9，**+4%**）、12.52ms（plies=15，**+71%**）；
    /// - **真出杀**（mate-in-8 的局面）：5.4 → 36.5ms，一次证明约 **31ms / 58,178 节点**；
    ///   证完根节点已 solved，`simulate` 立即返回，所以每次搜索只付一次。
    ///
    /// 建议：**自博弈/数据生成用 9**（把可证射程从 mate-in-4 推到 mate-in-5，成本被深度上限兜住）；
    /// **分析、UCI `go mate N`、高预算评测用 15~17**（那个 mate-in-8 需要 15）。
    ///
    /// 想证更长的杀只能加预算（[`Self::mate_search_nodes`]）：置换表和着法排序都实测过，
    /// 在 check-only 树上没有收益，原因记在 `mate.rs` 的模块注释里。
    pub mate_search_plies: usize,
    /// 连杀证明搜索的全局节点预算上限。默认 20 万覆盖实测的 mate-in-8（58,178 节点），
    /// 同时是"最坏情况花多少时间"的硬兜底。
    ///
    /// UCI 侧对应 `MateSearchNodes`。实测抽帧库最坏的中局局面在 depth=31 下要 **656,431**
    /// 节点才能判"无杀"（约 380ms）；把预算开到 2,000,000 就能让它走到结论，代价是这一手
    /// 慢 10 倍。预算撞墙时会打印 `info string mate ... source=node-budget`。
    pub mate_search_nodes: usize,
    #[cfg_attr(not(feature = "gpu-train"), allow(dead_code))]
    pub(super) gpu_trainer: Option<Box<train_gpu::GpuTrainer>>,
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
            check_context_hidden: self.check_context_hidden.clone(),
            hidden_bias: self.hidden_bias.clone(),
            shared_hidden: self.shared_hidden.clone(),
            shared_bias: self.shared_bias.clone(),
            value_head_hidden: self.value_head_hidden.clone(),
            value_head_bias: self.value_head_bias.clone(),
            value_king_piece_hidden: self.value_king_piece_hidden.clone(),
            value_king_piece_projection: self.value_king_piece_projection.clone(),
            value_head_output: self.value_head_output.clone(),
            value_history_output: self.value_history_output.clone(),
            value_history_cache: self.value_history_cache.clone(),
            moves_left_output: self.moves_left_output.clone(),
            moves_left_bias: self.moves_left_bias.clone(),
            moves_left_active: self.moves_left_active,
            moves_left_params: self.moves_left_params,
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
            policy_repetition_hidden: self.policy_repetition_hidden.clone(),
            policy_repetition_bias: self.policy_repetition_bias.clone(),
            policy_accumulator_features: self.policy_accumulator_features.clone(),
            policy_accumulator_moved_delta: self.policy_accumulator_moved_delta.clone(),
            policy_accumulator_capture: self.policy_accumulator_capture.clone(),
            policy_sparse_table_folded: self.policy_sparse_table_folded.clone(),
            policy_tactical_folded: self.policy_tactical_folded.clone(),
            value_threat_active: self.value_threat_active,
            policy_tactical_active: self.policy_tactical_active,
            check_context_active: self.check_context_active,
            mate_search_plies: self.mate_search_plies,
            mate_search_nodes: self.mate_search_nodes,
            gpu_trainer: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct AzEvalOutput {
    pub value_wdl: [f32; WDL_HEAD_SIZE],
    pub value: f32,
    pub moves_left: f32,
}

/// 九宫的 9 个格子（按颜色）。
fn palace_mask(color: Color) -> u128 {
    let ranks: [usize; 3] = match color {
        Color::Red => [7, 8, 9],
        Color::Black => [0, 1, 2],
    };
    let mut mask = 0u128;
    for rank in ranks {
        for file in 3..=5 {
            mask |= 1u128 << (rank * BOARD_FILES + file);
        }
    }
    mask
}

/// 同线且中间全空（用于飞将判断）。
fn file_clear_between(position: &Position, a: usize, b: usize) -> bool {
    let file = a % BOARD_FILES;
    if b % BOARD_FILES != file {
        return false;
    }
    let (ra, rb) = (a / BOARD_FILES, b / BOARD_FILES);
    let (start, end) = if ra < rb { (ra + 1, rb) } else { (rb + 1, ra) };
    (start..end).all(|rank| position.piece_at(rank * BOARD_FILES + file).is_none())
}

/// `color` 的将还有几个**安全逃格**：宫内相邻、不被己方子挡住、不被 `attacker_mask` 覆盖、
/// 且走进去不会与对方将照面。
///
/// 0 就是"杀网已经成形"。这里用对方攻击位板做判定，正好复用策略头已经算好的那张位板。
fn king_safe_escapes(position: &Position, color: Color, attacker_mask: u128) -> usize {
    let Some(king) = position.general_square(color) else {
        return 0;
    };
    let enemy_king = position.general_square(color.opposite());
    let file = king % BOARD_FILES;
    let rank = king / BOARD_FILES;
    let mut safe = 0usize;
    for (df, dr) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
        let nf = file as i32 + df;
        let nr = rank as i32 + dr;
        if !(3..=5).contains(&nf) {
            continue;
        }
        let rank_ok = match color {
            Color::Red => (7..=9).contains(&nr),
            Color::Black => (0..=2).contains(&nr),
        };
        if !rank_ok {
            continue;
        }
        let square = nr as usize * BOARD_FILES + nf as usize;
        if position
            .piece_at(square)
            .is_some_and(|piece| piece.color == color)
        {
            continue;
        }
        if attacker_mask & (1u128 << square) != 0 {
            continue;
        }
        if enemy_king.is_some_and(|enemy| {
            enemy % BOARD_FILES == square % BOARD_FILES
                && file_clear_between(position, enemy, square)
        }) {
            continue;
        }
        safe += 1;
    }
    safe
}

/// 引擎**已经算过、却一直没喂给模型**的 8 个标量。全部来自调用方已经算好的量：
/// `gives_check`（`fill_policy_gives_checks` 本来就无条件算）、`masks`（策略头本来就要的
/// 全盘攻击位板）、`moves.len()`（调用方手里就有）。

pub(crate) fn check_context_features(
    position: &Position,
    moves: &[Move],
    gives_check: &[f32],
    masks: [u128; 2],
) -> [f32; CHECK_CONTEXT_SIZE] {
    let side = position.side_to_move();
    let enemy = side.opposite();
    let own_attacks = masks[color_index(side)];
    let enemy_attacks = masks[color_index(enemy)];
    let checks = gives_check.iter().filter(|&&flag| flag != 0.0).count();
    // 缺少将位（理论上不该出现）时不问 in_check，避免它的 expect 崩掉。
    let in_check = position
        .general_square(side)
        .is_some_and(|_| position.in_check(side));
    let enemy_escapes = king_safe_escapes(position, enemy, own_attacks);
    let our_escapes = king_safe_escapes(position, side, enemy_attacks);
    [
        f32::from(in_check),
        (checks as f32 / 4.0).min(1.0),
        (enemy_escapes as f32 / 4.0).min(1.0),
        (our_escapes as f32 / 4.0).min(1.0),
        ((own_attacks & palace_mask(enemy)).count_ones() as f32 / 9.0).min(1.0),
        ((enemy_attacks & palace_mask(side)).count_ones() as f32 / 9.0).min(1.0),
        (moves.len() as f32 / 64.0).min(1.0),
        // 线性层拼不出这个交互："我有将军着法、而对方的将几乎没有安全逃格"。
        f32::from(checks > 0 && enemy_escapes <= 1),
    ]
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
        // 同样从零开始：新模型一开始不依赖它，梯度再把它学出来。
        let check_context_hidden = vec![0.0; CHECK_CONTEXT_SIZE * hidden_size];
        let hidden_bias = vec![0.0; hidden_size];
        // Start value-neutral. A random value head can evaluate startpos as a
        // large red/black advantage before any training, and MCTS amplifies
        // that noise into the first self-play dataset.
        let value_head_hidden = (0..VALUE_HEAD_SIZE * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt() * 0.5))
            .collect();
        let value_head_bias = vec![0.0; VALUE_HEAD_SIZE];
        let value_king_piece_hidden = vec![0.0; VALUE_KING_PIECE_VOCAB * VALUE_KING_PIECE_RANK];
        // 零表保持初始旁路输出为零；非零投影使价值输出头开始学习后，表即可收到梯度。
        let mut value_king_piece_projection = vec![0.0; VALUE_KING_PIECE_RANK * VALUE_HEAD_SIZE];
        for channel in 0..VALUE_KING_PIECE_RANK {
            value_king_piece_projection[channel * VALUE_HEAD_SIZE + channel] = 1.0;
        }
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
        let policy_repetition_hidden = vec![0.0; hidden_size];
        let policy_repetition_bias = vec![0.0; 1];
        let shared_hidden = (0..hidden_size * hidden_size)
            .map(|_| rng.weight((6.0 / hidden_size as f32).sqrt()))
            .collect();
        let mut model = Self {
            hidden_size,
            arch,
            input_hidden,
            input_piece_hidden,
            input_rank_hidden,
            input_file_hidden,
            input_king_piece_hidden,
            rule_context_hidden,
            check_context_hidden,
            hidden_bias,
            shared_hidden,
            shared_bias: vec![0.0; hidden_size],
            value_head_hidden,
            value_head_bias,
            value_king_piece_hidden,
            value_king_piece_projection,
            value_head_output,
            value_history_output: vec![0.0; WDL_HEAD_SIZE * HISTORY_CONTEXT_SIZE],
            value_history_cache: super::history::ValueHistoryCache::default(),
            moves_left_output: vec![0.0; hidden_size],
            moves_left_bias: vec![0.0],
            moves_left_active: false,
            moves_left_params: AzMovesLeftParams::default(),
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
            policy_repetition_hidden,
            policy_repetition_bias,
            policy_accumulator_features: Vec::new(),
            policy_accumulator_moved_delta: Vec::new(),
            policy_accumulator_capture: Vec::new(),
            policy_sparse_table_folded: Vec::new(),
            policy_tactical_folded: Vec::new(),
            value_threat_active: false,
            policy_tactical_active: false,
            check_context_active: false,
            mate_search_plies: 0,
            mate_search_nodes: 200_000,
            gpu_trainer: None,
        };
        model.rebuild_policy_cache();
        model.rebuild_value_threat();
        model.rebuild_check_context();
        model.rebuild_moves_left();
        model.rebuild_value_history();
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

    /// 优化器检查点同时记录更新序号；缺失独立进度文件时用它恢复。
    pub fn training_state_next_update(path: impl AsRef<Path>) -> io::Result<usize> {
        let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }
            .map_err(candle_io_error)?;
        let state = tensors
            .load("state", &Device::Cpu)
            .and_then(|state| state.to_vec1::<i64>())
            .map_err(candle_io_error)?;
        if state.len() != 4
            || !matches!(state[0], 1 | 2)
            || state[1] != MODEL_FORMAT_VERSION as i64
            || state[2] < 0
            || state[3] < 1
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid optimizer resume metadata",
            ));
        }
        usize::try_from(state[3]).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
    }

    pub fn restore_training_state(
        &mut self,
        path: impl AsRef<Path>,
        next_update: usize,
        lr: f32,
        optimizer: AzTrainOptimizer,
    ) -> io::Result<()> {
        let mut trainer =
            train_gpu::GpuTrainer::new(self, lr, optimizer).map_err(candle_io_error)?;
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
        samples: Vec<AzTrainingSample>,
        lr: f32,
        optimizer: AzTrainOptimizer,
    ) -> io::Result<()> {
        if self.gpu_trainer.is_none() {
            self.gpu_trainer = Some(Box::new(
                train_gpu::GpuTrainer::new(self, lr, optimizer).map_err(candle_io_error)?,
            ));
        }
        self.gpu_trainer.as_mut().unwrap().set_holdout(samples);
        Ok(())
    }

    pub fn take_training_checks(&mut self) -> Vec<AzHoldoutReport> {
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
        let mut expected_tensors = vec!["az_model_format_version"];
        macro_rules! expect_tensor {
            ($field:ident, [$($dim:expr),+]) => {
                expected_tensors.push(stringify!($field));
            };
        }
        az_weight_tensors!(expect_tensor, 0);
        if let Some((name, _)) = tensors
            .tensors()
            .iter()
            .find(|(name, _)| !expected_tensors.contains(&name.as_str()))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported AZ model tensor `{name}`"),
            ));
        }
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
        let policy_accumulator_hidden =
            load_candle_f32_tensor(&tensors, "policy_accumulator_hidden")?;
        let policy_accumulator_move = load_candle_f32_tensor(&tensors, "policy_accumulator_move")?;
        let mut model = Self {
            hidden_size,
            arch,
            input_hidden: load_candle_f32_tensor(&tensors, "input_hidden")?,
            input_piece_hidden: load_candle_f32_tensor(&tensors, "input_piece_hidden")?,
            input_rank_hidden: load_candle_f32_tensor(&tensors, "input_rank_hidden")?,
            input_file_hidden: load_candle_f32_tensor(&tensors, "input_file_hidden")?,
            input_king_piece_hidden: load_candle_f32_tensor(&tensors, "input_king_piece_hidden")?,
            rule_context_hidden: load_candle_f32_tensor(&tensors, "rule_context_hidden")?,
            check_context_hidden: load_candle_f32_tensor(&tensors, "check_context_hidden")?,
            hidden_bias,
            shared_hidden: load_candle_f32_tensor(&tensors, "shared_hidden")?,
            shared_bias: load_candle_f32_tensor(&tensors, "shared_bias")?,
            value_head_hidden: load_candle_f32_tensor(&tensors, "value_head_hidden")?,
            value_head_bias: load_candle_f32_tensor(&tensors, "value_head_bias")?,
            value_king_piece_hidden: load_candle_f32_tensor(&tensors, "value_king_piece_hidden")?,
            value_king_piece_projection: load_candle_f32_tensor(
                &tensors,
                "value_king_piece_projection",
            )?,
            value_head_output: load_candle_f32_tensor(&tensors, "value_head_output")?,
            value_history_output: load_candle_f32_tensor(&tensors, "value_history_output")?,
            value_history_cache: super::history::ValueHistoryCache::default(),
            moves_left_output: load_candle_f32_tensor(&tensors, "moves_left_output")?,
            moves_left_bias: load_candle_f32_tensor(&tensors, "moves_left_bias")?,
            moves_left_active: false,
            moves_left_params: AzMovesLeftParams::default(),
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
            policy_accumulator_hidden,
            policy_accumulator_move,
            policy_sparse_table: load_candle_f32_tensor(&tensors, "policy_sparse_table")?,
            policy_sparse_factor: load_candle_f32_tensor(&tensors, "policy_sparse_factor")?,
            policy_repetition_hidden: load_candle_f32_tensor(&tensors, "policy_repetition_hidden")?,
            policy_repetition_bias: load_candle_f32_tensor(&tensors, "policy_repetition_bias")?,
            policy_tactical: load_candle_f32_tensor(&tensors, "policy_tactical")?,
            policy_accumulator_features: Vec::new(),
            policy_accumulator_moved_delta: Vec::new(),
            policy_accumulator_capture: Vec::new(),
            policy_sparse_table_folded: Vec::new(),
            policy_tactical_folded: Vec::new(),
            value_threat_active: false,
            policy_tactical_active: false,
            check_context_active: false,
            mate_search_plies: 0,
            mate_search_nodes: 200_000,
            gpu_trainer: None,
        };
        model.rebuild_policy_cache();
        model.rebuild_value_threat();
        model.rebuild_check_context();
        model.rebuild_moves_left();
        model.rebuild_policy_tactical();
        model.rebuild_value_history();
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
        self.evaluate_with_scratch_output_with_repetition_and_history(
            position,
            moves,
            &[],
            &rule_context_features(position, history),
            history,
            &mut scratch,
        )
        .value
    }

    pub fn evaluate_wdl_with_rules(
        &self,
        position: &Position,
        history: &[crate::xiangqi::RuleHistoryEntry],
        moves: &[Move],
    ) -> [f32; WDL_HEAD_SIZE] {
        let mut scratch = AzEvalScratch::new(self.arch);
        self.evaluate_with_scratch_output_with_repetition_and_history(
            position,
            moves,
            &[],
            &rule_context_features(position, history),
            history,
            &mut scratch,
        )
        .value_wdl
    }

    pub(crate) fn evaluate_with_scratch(
        &self,
        position: &Position,
        moves: &[Move],
        scratch: &mut AzEvalScratch,
    ) -> f32 {
        self.evaluate_with_scratch_output(position, moves, &[0.0; RULE_CONTEXT_SIZE], scratch)
            .value
    }

    pub(crate) fn evaluate_with_scratch_output(
        &self,
        position: &Position,
        moves: &[Move],
        rule_context: &[f32; RULE_CONTEXT_SIZE],
        scratch: &mut AzEvalScratch,
    ) -> AzEvalOutput {
        self.evaluate_with_scratch_output_with_repetition(
            position,
            moves,
            &[],
            rule_context,
            scratch,
        )
    }

    pub(crate) fn evaluate_with_scratch_output_with_repetition(
        &self,
        position: &Position,
        moves: &[Move],
        repetition_flags: &[u8],
        rule_context: &[f32; RULE_CONTEXT_SIZE],
        scratch: &mut AzEvalScratch,
    ) -> AzEvalOutput {
        self.evaluate_with_scratch_output_with_repetition_and_history(
            position,
            moves,
            repetition_flags,
            rule_context,
            &[],
            scratch,
        )
    }

    pub(crate) fn evaluate_with_scratch_output_with_repetition_and_history(
        &self,
        position: &Position,
        moves: &[Move],
        repetition_flags: &[u8],
        rule_context: &[f32; RULE_CONTEXT_SIZE],
        history: &[crate::xiangqi::RuleHistoryEntry],
        scratch: &mut AzEvalScratch,
    ) -> AzEvalOutput {
        let orientation = self.history_orientation(position, history, moves, repetition_flags);
        self.evaluate_oriented(
            position,
            moves,
            repetition_flags,
            rule_context,
            history,
            None,
            orientation,
            scratch,
        )
    }

    pub(crate) fn evaluate_with_scratch_output_with_repetition_and_history_features(
        &self,
        position: &Position,
        moves: &[Move],
        repetition_flags: &[u8],
        rule_context: &[f32; RULE_CONTEXT_SIZE],
        history_features: &[f32; HISTORY_CONTEXT_SIZE],
        scratch: &mut AzEvalScratch,
    ) -> AzEvalOutput {
        let orientation = super::reflection::input_orientation(
            position,
            history_features,
            moves,
            repetition_flags,
        );
        self.evaluate_oriented(
            position,
            moves,
            repetition_flags,
            rule_context,
            &[],
            Some(history_features),
            orientation,
            scratch,
        )
    }

    fn history_orientation(
        &self,
        position: &Position,
        history: &[crate::xiangqi::RuleHistoryEntry],
        moves: &[Move],
        flags: &[u8],
    ) -> std::cmp::Ordering {
        let board = super::reflection::board_orientation(position);
        if board != std::cmp::Ordering::Equal {
            return board;
        }
        super::reflection::input_orientation(
            position,
            &super::history::history_features(position, history),
            moves,
            flags,
        )
    }

    fn oriented_history_logits(
        &self,
        position: &Position,
        history: &[crate::xiangqi::RuleHistoryEntry],
        explicit: Option<&[f32; HISTORY_CONTEXT_SIZE]>,
        reflected: bool,
    ) -> [f32; WDL_HEAD_SIZE] {
        if let Some(features) = explicit {
            return self.value_history_logits_from_features(&if reflected {
                super::reflection::mirror_history_features(features)
            } else {
                *features
            });
        }
        if !reflected {
            return self.value_history_cache.logits(position, history);
        }
        // 历史头只读取最近两着；规则历史与真实棋盘始终由调用者保留。
        let Some(last) = history.last() else {
            return self.value_history_cache.logits(position, history);
        };
        let mut recent = [*last; 2];
        let count = history.len().min(2);
        for (dst, src) in recent.iter_mut().zip(&history[history.len() - count..]) {
            *dst = *src;
            dst.mv = dst.mv.map(crate::az::nnue::mirror_file_move);
        }
        self.value_history_cache.logits(position, &recent[..count])
    }

    fn evaluate_oriented(
        &self,
        position: &Position,
        moves: &[Move],
        flags: &[u8],
        rule_context: &[f32; RULE_CONTEXT_SIZE],
        history: &[crate::xiangqi::RuleHistoryEntry],
        explicit: Option<&[f32; HISTORY_CONTEXT_SIZE]>,
        orientation: std::cmp::Ordering,
        scratch: &mut AzEvalScratch,
    ) -> AzEvalOutput {
        let reflected = orientation == std::cmp::Ordering::Greater;
        let mirrored;
        let mut mapped = std::mem::take(&mut scratch.reflected_moves);
        let (position, evaluated_moves) = if reflected {
            mirrored = position.mirror_files();
            mapped.clear();
            mapped.extend(moves.iter().copied().map(crate::az::nnue::mirror_file_move));
            (&mirrored, mapped.as_slice())
        } else {
            (position, moves)
        };
        let history_logits = self.oriented_history_logits(position, history, explicit, reflected);
        let output = self.evaluate_with_scratch_output_with_history_logits(
            position,
            evaluated_moves,
            flags,
            rule_context,
            history_logits,
            scratch,
        );
        if orientation == std::cmp::Ordering::Equal {
            super::reflection::symmetrize_policy_logits(moves, &mut scratch.logits);
        }
        scratch.reflected_moves = mapped;
        output
    }

    fn evaluate_with_scratch_output_with_history_logits(
        &self,
        position: &Position,
        moves: &[Move],
        repetition_flags: &[u8],
        rule_context: &[f32; RULE_CONTEXT_SIZE],
        history_logits: [f32; WDL_HEAD_SIZE],
        scratch: &mut AzEvalScratch,
    ) -> AzEvalOutput {
        let value = self.evaluate_value_only_with_history_logits(
            position,
            moves,
            rule_context,
            history_logits,
            scratch,
        );
        scratch.policy_accumulator_context =
            self.policy_accumulator(position, position.side_to_move());
        self.evaluate_policy_with_scratch(position, moves, repetition_flags, scratch);
        value
    }

    fn evaluate_value_only_with_history_logits(
        &self,
        position: &Position,
        moves: &[Move],
        rule_context: &[f32; RULE_CONTEXT_SIZE],
        history_logits: [f32; WDL_HEAD_SIZE],
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
        // 全零的将军上下文权重不影响隐藏层，跳过其特征计算。
        scratch.policy_inputs_ready = false;
        if self.check_context_active {
            crate::scope_profile!("az.eval.check_context");
            self.fill_policy_inputs(position, moves, scratch, true);
            let context = check_context_features(
                position,
                moves,
                &scratch.policy_gives_check,
                scratch.attack_masks,
            );
            self.add_check_context_to_hidden(&context, &mut scratch.hidden);
        }
        {
            crate::scope_profile!("az.eval.activation_norm");
            relu_in_place(&mut scratch.hidden);
            rms_norm_in_place(&mut scratch.hidden);
            self.shared_hidden_into(&scratch.hidden, &mut scratch.shared_hidden);
            std::mem::swap(&mut scratch.hidden, &mut scratch.shared_hidden);
        }
        let (value_wdl, value) = {
            crate::scope_profile!("az.eval.value_head");
            self.value_king_piece_accumulate(position, &mut scratch.value_king_piece_accumulator);
            let mut threat_logits = self.value_threat_logits(
                position,
                &mut scratch.value_threat_accumulator,
                &mut scratch.value_threat_activation,
            );
            for j in 0..WDL_HEAD_SIZE {
                threat_logits[j] += history_logits[j];
            }
            self.value_wdl_from_hidden_into(
                &scratch.hidden,
                &scratch.value_king_piece_accumulator,
                &mut scratch.value_head,
                threat_logits,
            )
        };
        scratch.features = features;
        AzEvalOutput {
            value_wdl,
            value,
            moves_left: self.moves_left_from_hidden(&scratch.hidden),
        }
    }

    #[cfg(test)]
    pub(crate) fn evaluate_incremental_with_scratch_output(
        &self,
        position: &Position,
        accumulator_hidden: &[f32],
        policy_accumulator: &[f32; POLICY_ACCUMULATOR_RANK],
        moves: &[Move],
        repetition_flags: &[u8],
        rule_context: &[f32; RULE_CONTEXT_SIZE],
        scratch: &mut AzEvalScratch,
    ) -> AzEvalOutput {
        self.evaluate_incremental_with_scratch_output_with_history(
            position,
            accumulator_hidden,
            policy_accumulator,
            moves,
            repetition_flags,
            rule_context,
            &[],
            scratch,
        )
    }

    pub(crate) fn evaluate_incremental_with_scratch_output_with_history(
        &self,
        position: &Position,
        accumulator_hidden: &[f32],
        policy_accumulator: &[f32; POLICY_ACCUMULATOR_RANK],
        moves: &[Move],
        repetition_flags: &[u8],
        rule_context: &[f32; RULE_CONTEXT_SIZE],
        history: &[crate::xiangqi::RuleHistoryEntry],
        scratch: &mut AzEvalScratch,
    ) -> AzEvalOutput {
        let orientation = self.history_orientation(position, history, moves, repetition_flags);
        let reflected = orientation == std::cmp::Ordering::Greater;
        let mirrored;
        let mut mapped = std::mem::take(&mut scratch.reflected_moves);
        let (canonical_position, evaluated_moves) = if reflected {
            mirrored = position.mirror_files();
            mapped.clear();
            mapped.extend(moves.iter().copied().map(crate::az::nnue::mirror_file_move));
            (&mirrored, mapped.as_slice())
        } else {
            (position, moves)
        };
        let history_logits =
            self.oriented_history_logits(canonical_position, history, None, reflected);
        let output = self.evaluate_incremental_oriented(
            canonical_position,
            accumulator_hidden,
            policy_accumulator,
            evaluated_moves,
            repetition_flags,
            rule_context,
            history_logits,
            scratch,
        );
        if orientation == std::cmp::Ordering::Equal {
            super::reflection::symmetrize_policy_logits(moves, &mut scratch.logits);
        }
        scratch.reflected_moves = mapped;
        output
    }

    fn evaluate_incremental_oriented(
        &self,
        position: &Position,
        accumulator_hidden: &[f32],
        policy_accumulator: &[f32; POLICY_ACCUMULATOR_RANK],
        moves: &[Move],
        repetition_flags: &[u8],
        rule_context: &[f32; RULE_CONTEXT_SIZE],
        history_logits: [f32; WDL_HEAD_SIZE],
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
        // 与全量评估同一条标量块路径（见上面 `evaluate_with_scratch_output_with_repetition`）。
        scratch.policy_inputs_ready = false;
        if self.check_context_active {
            crate::scope_profile!("az.eval.check_context");
            self.fill_policy_inputs(position, moves, scratch, true);
            let context = check_context_features(
                position,
                moves,
                &scratch.policy_gives_check,
                scratch.attack_masks,
            );
            self.add_check_context_to_hidden(&context, &mut scratch.hidden);
        }
        scratch
            .policy_accumulator_context
            .copy_from_slice(policy_accumulator);
        {
            crate::scope_profile!("az.eval.activation_norm");
            relu_in_place(&mut scratch.hidden);
            rms_norm_in_place(&mut scratch.hidden);
            self.shared_hidden_into(&scratch.hidden, &mut scratch.shared_hidden);
            std::mem::swap(&mut scratch.hidden, &mut scratch.shared_hidden);
        }
        let (value_wdl, value) = {
            crate::scope_profile!("az.eval.value_head");
            self.value_king_piece_accumulate(position, &mut scratch.value_king_piece_accumulator);
            let mut threat_logits = self.value_threat_logits(
                position,
                &mut scratch.value_threat_accumulator,
                &mut scratch.value_threat_activation,
            );
            for j in 0..WDL_HEAD_SIZE {
                threat_logits[j] += history_logits[j];
            }
            self.value_wdl_from_hidden_into(
                &scratch.hidden,
                &scratch.value_king_piece_accumulator,
                &mut scratch.value_head,
                threat_logits,
            )
        };
        self.evaluate_policy_with_scratch(position, moves, repetition_flags, scratch);
        AzEvalOutput {
            value_wdl,
            value,
            moves_left: self.moves_left_from_hidden(&scratch.hidden),
        }
    }

    pub(crate) fn evaluate_policy_with_scratch(
        &self,
        position: &Position,
        moves: &[Move],
        repetition_flags: &[u8],
        scratch: &mut AzEvalScratch,
    ) {
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
        scratch.logits.resize(moves.len(), 0.0);
        if scratch.policy_piece_square_scores.is_empty() {
            self.fill_policy_piece_square_scores(&mut scratch.policy_piece_square_scores);
        }
        let move_map = move_map();
        let side = position.side_to_move();
        let king_buckets = canonical_buckets_for_perspective(position, side);
        // 主干之前如果已经算过（标量块需要），这里直接复用，避免同一个局面算两遍：
        // 提前算的总工作量不变，只是把它挪到主干之前。
        if !scratch.policy_inputs_ready {
            self.fill_policy_inputs(position, moves, scratch, self.policy_tactical_active);
        }
        let attack_masks = if self.policy_tactical_active {
            scratch.attack_masks
        } else {
            [0u128; 2]
        };
        let opponent_attacks = attack_masks[color_index(side.opposite())];
        let own_attacks = attack_masks[color_index(side)];
        let repetition_logit = dot_product(&scratch.hidden, &self.policy_repetition_hidden)
            + self.policy_repetition_bias[0];
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
                    let accumulator_move = &self.policy_accumulator_move
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
                        let cache_start = move_index * POLICY_CACHE_PIECE_SIZE;
                        let mut value =
                            dot_product(&scratch.policy_accumulator_context, accumulator_move)
                                + self.policy_accumulator_moved_delta
                                    [cache_start + from / BOARD_SIZE];
                        if let Some(captured) = captured {
                            value -= self.policy_accumulator_capture
                                [cache_start + captured / BOARD_SIZE - POLICY_CACHE_PIECE_SIZE];
                        }
                        value
                    } else {
                        0.0
                    };
                    let sparse_logit = consequence.map_or(0.0, |(from, _, captured)| {
                        let moved_piece = from / BOARD_SIZE;
                        let captured_piece = captured.map(|feature| feature / BOARD_SIZE);
                        let main = policy_cache_main_index(
                            move_index,
                            moved_piece,
                            king_buckets.0,
                            king_buckets.1,
                        );
                        let capture = policy_cache_capture_index(move_index, captured_piece);
                        self.policy_sparse_table_folded[main]
                            + self.policy_sparse_table_folded[capture]
                    });
                    let tactical_logit = if self.policy_tactical_active {
                        crate::scope_profile!("az.eval.policy.tactical");
                        consequence.map_or(0.0, |(from, _, captured)| {
                            let moved_piece = from / BOARD_SIZE;
                            let check = scratch.policy_gives_check[index];
                            let (
                                source_attacked,
                                destination_attacked,
                                source_defended,
                                destination_defended,
                            ) = policy_move_tactical_flags(*mv, opponent_attacks, own_attacks);
                            let tactical = policy_tactical_indices(
                                move_index,
                                moved_piece,
                                source_attacked,
                                destination_attacked,
                                source_defended,
                                destination_defended,
                                captured.map(|feature| feature / BOARD_SIZE),
                                check != 0.0,
                            );
                            let base = self.policy_tactical_folded[tactical[0]];
                            captured.map_or(base, |_| base + self.policy_tactical[tactical[2]])
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
                        + tactical_logit
                        + f32::from(repetition_flags.get(index).copied().unwrap_or(0))
                            * repetition_logit;
                }
            }
        }
    }

    pub(crate) fn fill_policy_gives_checks(
        &self,
        position: &Position,
        moves: &[Move],
        output: &mut Vec<f32>,
    ) {
        crate::scope_profile!("az.eval.policy.gives_check");
        output.resize(moves.len(), 0.0);
        for (flag, &mv) in output.iter_mut().zip(moves) {
            *flag = f32::from(position.gives_check_after_move_fast(mv));
        }
    }

    #[inline]

    pub(crate) fn add_rule_context_to_hidden(
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

    /// 把"引擎已经算过、却没喂给模型"的标量块加到主干上。
    pub(crate) fn add_check_context_to_hidden(
        &self,
        check_context: &[f32; CHECK_CONTEXT_SIZE],
        hidden: &mut [f32],
    ) {
        for (feature, &value) in check_context.iter().enumerate() {
            if value == 0.0 {
                continue;
            }
            let row = &self.check_context_hidden
                [feature * self.hidden_size..(feature + 1) * self.hidden_size];
            for (target, &weight) in hidden.iter_mut().zip(row) {
                *target += value * weight;
            }
        }
    }

    /// 在主干之前把"每走法将军 flag + 双方攻击位板"算好一次，供标量块与策略头共用。
    ///
    /// 这两样东西策略头本来就要算（位板受 `policy_tactical_active` 门控、flag 无条件），
    /// 而标量块需要在主干之前用它们，所以只能提前算：总工作量不变，只是把顺序挪前。
    pub(crate) fn fill_policy_inputs(
        &self,
        position: &Position,
        moves: &[Move],
        scratch: &mut AzEvalScratch,
        with_masks: bool,
    ) {
        self.fill_policy_gives_checks(position, moves, &mut scratch.policy_gives_check);
        if with_masks {
            scratch.attack_masks = position.attacked_squares_masks();
        }
        scratch.policy_inputs_ready = true;
    }

    pub(crate) fn add_factorized_structure_into(&self, features: &[usize], hidden: &mut [f32]) {
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
    pub(crate) unsafe fn add_factorized_structure_avx2(
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

    pub(crate) fn input_embedding_linear_into(&self, features: &[usize], hidden: &mut Vec<f32>) {
        hidden.resize(self.hidden_size, 0.0);
        self.input_embedding_linear_into_slice(features, hidden);
    }

    pub(crate) fn input_embedding_linear_into_slice(&self, features: &[usize], hidden: &mut [f32]) {
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

    pub(crate) fn value_threat_logits(
        &self,
        position: &Position,
        accumulator: &mut Vec<f32>,
        activation: &mut Vec<f32>,
    ) -> [f32; WDL_HEAD_SIZE] {
        if !self.value_threat_active {
            return [0.0; WDL_HEAD_SIZE];
        }
        crate::scope_profile!("az.eval.value_threat");
        accumulator.resize(VALUE_THREAT_RANK, 0.0);
        accumulator.fill(0.0);
        let perspective = position.side_to_move();
        {
            crate::scope_profile!("az.eval.value_threat.accumulate");
            let mut active = 0usize;
            visit_value_threat_features(position, perspective, |feature| {
                active += 1;
                let row = &self.value_threat_embedding
                    [feature * VALUE_THREAT_RANK..(feature + 1) * VALUE_THREAT_RANK];
                for (sum, weight) in accumulator.iter_mut().zip(row) {
                    *sum += weight;
                }
            });
            let scale = 1.0 / (active.max(1) as f32).sqrt();
            for value in accumulator.iter_mut() {
                *value *= scale;
            }
        }
        let mut logits = [0.0; WDL_HEAD_SIZE];
        {
            crate::scope_profile!("az.eval.value_threat.output");
            activation.resize(VALUE_THREAT_RANK * 2, 0.0);
            for rank in 0..VALUE_THREAT_RANK {
                let value = accumulator[rank];
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

    pub(crate) fn value_king_piece_accumulate(
        &self,
        position: &Position,
        accumulator: &mut Vec<f32>,
    ) {
        crate::scope_profile!("az.eval.value_king_piece");
        let mut pooled = [0.0; VALUE_KING_PIECE_RANK];
        let mut active = 0;
        visit_value_king_piece_features(position, position.side_to_move(), |feature| {
            active += 1;
            let row = &self.value_king_piece_hidden
                [feature * VALUE_KING_PIECE_RANK..(feature + 1) * VALUE_KING_PIECE_RANK];
            for (sum, &weight) in pooled.iter_mut().zip(row) {
                *sum += weight;
            }
        });
        let scale = 1.0 / (active.max(1) as f32).sqrt();
        accumulator.resize(VALUE_HEAD_SIZE, 0.0);
        accumulator.fill(0.0);
        for (channel, value) in pooled.into_iter().enumerate() {
            let value = value * scale;
            let row = &self.value_king_piece_projection
                [channel * VALUE_HEAD_SIZE..(channel + 1) * VALUE_HEAD_SIZE];
            for (sum, &weight) in accumulator.iter_mut().zip(row) {
                *sum += value * weight;
            }
        }
    }

    pub(crate) fn value_wdl_from_hidden_into(
        &self,
        hidden: &[f32],
        king_piece: &[f32],
        value_head: &mut Vec<f32>,
        threat_logits: [f32; WDL_HEAD_SIZE],
    ) -> ([f32; WDL_HEAD_SIZE], f32) {
        value_head.resize(VALUE_HEAD_SIZE, 0.0);
        value_head.copy_from_slice(&self.value_head_bias);
        for (feature, value) in value_head.iter_mut().enumerate().take(VALUE_HEAD_SIZE) {
            let hidden_row = &self.value_head_hidden
                [feature * self.hidden_size..(feature + 1) * self.hidden_size];
            *value += king_piece[feature] + dot_product(hidden, hidden_row);
            *value = (*value).max(0.0);
        }
        let mut logits = [0.0f32; WDL_HEAD_SIZE];
        for (out, logit) in logits.iter_mut().enumerate() {
            let row = &self.value_head_output[out * VALUE_HEAD_SIZE..(out + 1) * VALUE_HEAD_SIZE];
            *logit = dot_product(value_head, row) + threat_logits[out];
        }
        let wdl = softmax_fixed3(logits);
        let q = wdl[0] - wdl[2];
        (wdl, q)
    }

    pub(crate) fn fill_policy_piece_square_scores(&self, scores: &mut Vec<f32>) {
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

    pub(crate) fn rebuild_value_threat(&mut self) {
        self.value_threat_active = self.value_threat_output.iter().any(|&weight| weight != 0.0)
            || self
                .policy_threat_context
                .iter()
                .any(|&weight| weight != 0.0);
    }

    pub(crate) fn shared_hidden_into(&self, hidden: &[f32], output: &mut Vec<f32>) {
        output.resize(self.hidden_size, 0.0);
        for (row, value) in output.iter_mut().enumerate() {
            let start = row * self.hidden_size;
            *value = (dot_product(hidden, &self.shared_hidden[start..start + self.hidden_size])
                + self.shared_bias[row])
                .max(0.0);
        }
    }

    /// 从共享隐藏层计算剩余步数预测。
    pub(crate) fn moves_left_from_hidden(&self, hidden: &[f32]) -> f32 {
        ((dot_product(hidden, &self.moves_left_output) + self.moves_left_bias[0] + 1.0).max(0.0)
            * MOVES_LEFT_SCALE)
            .min(4096.0)
    }

    pub(crate) fn rebuild_moves_left(&mut self) {
        self.moves_left_active =
            self.moves_left_output.iter().any(|&v| v != 0.0) || self.moves_left_bias[0] != 0.0;
    }

    pub(crate) fn rebuild_check_context(&mut self) {
        self.check_context_active = self
            .check_context_hidden
            .iter()
            .any(|&weight| weight != 0.0);
    }

    pub(crate) fn rebuild_policy_tactical(&mut self) {
        self.policy_tactical_active = self.policy_tactical.iter().any(|&weight| weight != 0.0);
        self.policy_tactical_folded = self.policy_tactical[..POLICY_TACTICAL_EXACT_SIZE].to_vec();
        for move_index in 0..DENSE_MOVE_SPACE {
            for moved_piece in 0..STRUCTURAL_PIECE_SIZE / 2 {
                for signature in 0..POLICY_TACTICAL_SIGNATURE_BUCKETS {
                    let exact = (move_index * (STRUCTURAL_PIECE_SIZE / 2) + moved_piece)
                        * POLICY_TACTICAL_SIGNATURE_BUCKETS
                        + signature;
                    let piece_factor = POLICY_TACTICAL_EXACT_SIZE
                        + moved_piece * POLICY_TACTICAL_SIGNATURE_BUCKETS
                        + signature;
                    self.policy_tactical_folded[exact] += self.policy_tactical[piece_factor];
                }
            }
        }
    }

    pub(crate) fn rebuild_value_history(&mut self) {
        self.value_history_cache =
            super::history::ValueHistoryCache::new(&self.value_history_output);
    }

    pub fn value_history_logits_from_features(
        &self,
        features: &[f32; HISTORY_CONTEXT_SIZE],
    ) -> [f32; WDL_HEAD_SIZE] {
        std::array::from_fn(|output| {
            dot_product(
                features,
                &self.value_history_output
                    [output * HISTORY_CONTEXT_SIZE..(output + 1) * HISTORY_CONTEXT_SIZE],
            )
        })
    }

    pub fn value_history_logits(
        &self,
        position: &Position,
        history: &[crate::xiangqi::RuleHistoryEntry],
    ) -> [f32; WDL_HEAD_SIZE] {
        self.value_history_cache.logits(position, history)
    }

    pub(crate) fn rebuild_policy_cache(&mut self) {
        let mut projected = Vec::with_capacity(POLICY_ACCUMULATOR_ROWS * POLICY_ACCUMULATOR_RANK);
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
            POLICY_ACCUMULATOR_ROWS * POLICY_ACCUMULATOR_RANK
        );
        self.policy_accumulator_features = projected;

        let mut folded_sparse = vec![0.0; POLICY_CACHE_TABLE_SIZE];
        for move_index in 0..DENSE_MOVE_SPACE {
            for moved_piece in 0..POLICY_CACHE_PIECE_SIZE {
                for us_bucket in 0..V2_KING_BUCKETS {
                    for them_bucket in 0..V2_KING_BUCKETS {
                        let raw = policy_sparse_main_index(
                            move_index,
                            moved_piece,
                            us_bucket,
                            them_bucket,
                        );
                        let main = policy_cache_main_index(
                            move_index,
                            moved_piece,
                            us_bucket,
                            them_bucket,
                        );
                        folded_sparse[main] = self.policy_sparse_table[raw];
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
            for captured in (POLICY_CACHE_PIECE_SIZE..STRUCTURAL_PIECE_SIZE)
                .map(Some)
                .chain(std::iter::once(None))
            {
                folded_sparse[policy_cache_capture_index(move_index, captured)] =
                    self.policy_sparse_table[policy_sparse_capture_index(move_index, captured)];
            }
        }
        self.policy_sparse_table_folded = folded_sparse;

        let cache_size = DENSE_MOVE_SPACE * POLICY_CACHE_PIECE_SIZE;
        self.policy_accumulator_moved_delta = vec![0.0; cache_size];
        self.policy_accumulator_capture = vec![0.0; cache_size];
        for (move_index, &sparse) in move_map().dense_to_sparse.iter().enumerate() {
            let sparse = sparse as usize;
            let from_square = sparse / BOARD_SIZE;
            let to_square = sparse % BOARD_SIZE;
            let move_start = move_index * POLICY_ACCUMULATOR_RANK;
            let move_weights =
                &self.policy_accumulator_move[move_start..move_start + POLICY_ACCUMULATOR_RANK];
            for piece_index in 0..POLICY_CACHE_PIECE_SIZE {
                let from_start = (piece_index * BOARD_SIZE + from_square) * POLICY_ACCUMULATOR_RANK;
                let to_start = (piece_index * BOARD_SIZE + to_square) * POLICY_ACCUMULATOR_RANK;
                let capture_start = ((piece_index + POLICY_CACHE_PIECE_SIZE) * BOARD_SIZE
                    + to_square)
                    * POLICY_ACCUMULATOR_RANK;
                let mut moved_delta = 0.0;
                let mut capture = 0.0;
                for rank in 0..POLICY_ACCUMULATOR_RANK {
                    let weight = move_weights[rank];
                    let to = self.policy_accumulator_features[to_start + rank];
                    let from = self.policy_accumulator_features[from_start + rank];
                    moved_delta += (to - from) * weight;
                    capture += self.policy_accumulator_features[capture_start + rank] * weight;
                }
                let cache_index = move_index * POLICY_CACHE_PIECE_SIZE + piece_index;
                self.policy_accumulator_moved_delta[cache_index] = moved_delta;
                self.policy_accumulator_capture[cache_index] = capture;
            }
        }
    }

    pub(crate) fn policy_accumulator(
        &self,
        position: &Position,
        perspective: Color,
    ) -> [f32; POLICY_ACCUMULATOR_RANK] {
        let reflected = super::reflection::board_orientation_for(position, perspective)
            == std::cmp::Ordering::Greater;
        let buckets =
            super::accumulator::canonical_buckets_for_reflection(position, perspective, reflected);
        self.policy_accumulator_oriented(position, perspective, reflected, buckets)
    }

    fn policy_accumulator_oriented(
        &self,
        position: &Position,
        perspective: Color,
        reflected: bool,
        buckets: (usize, usize),
    ) -> [f32; POLICY_ACCUMULATOR_RANK] {
        let mut accumulator = [0.0; POLICY_ACCUMULATOR_RANK];
        self.add_policy_row(&mut accumulator, POLICY_ACCUMULATOR_BIAS_ROW, 1);
        for square in 0..BOARD_SIZE {
            let source = if reflected {
                crate::az::nnue::mirror_file_square(square)
            } else {
                square
            };
            if let Some(piece) = position.piece_at(source) {
                self.add_policy_piece(&mut accumulator, perspective, buckets, square, piece, 1);
            }
        }
        accumulator
    }

    #[cfg(test)]
    pub(crate) fn apply_policy_transition(
        &self,
        before: &Position,
        after: &Position,
        mv: Move,
        moved: Piece,
        captured: Option<Piece>,
        perspective: Color,
        accumulator: &mut [f32; POLICY_ACCUMULATOR_RANK],
    ) {
        let context = super::accumulator::CanonicalTransition::new(before, after, perspective);
        self.apply_policy_canonical_transition(after, mv, moved, captured, &context, accumulator);
    }

    pub(crate) fn apply_policy_canonical_transition(
        &self,
        after: &Position,
        mv: Move,
        moved: Piece,
        captured: Option<Piece>,
        context: &super::accumulator::CanonicalTransition,
        accumulator: &mut [f32; POLICY_ACCUMULATOR_RANK],
    ) {
        let perspective = context.perspective;
        let before_buckets = context.before_buckets;
        let after_buckets = context.after_buckets;
        if context.needs_refresh() {
            *accumulator = self.policy_accumulator_oriented(
                after,
                perspective,
                context.after_reflected,
                after_buckets,
            );
            return;
        }
        let mv = if context.reflected {
            crate::az::nnue::mirror_file_move(mv)
        } else {
            mv
        };
        self.add_policy_piece(
            accumulator,
            perspective,
            before_buckets,
            mv.from as usize,
            moved,
            -1,
        );
        if let Some(captured) = captured {
            self.add_policy_piece(
                accumulator,
                perspective,
                before_buckets,
                mv.to as usize,
                captured,
                -1,
            );
        }
        self.add_policy_piece(
            accumulator,
            perspective,
            after_buckets,
            mv.to as usize,
            moved,
            1,
        );
    }

    pub(crate) fn add_policy_piece(
        &self,
        accumulator: &mut [f32; POLICY_ACCUMULATOR_RANK],
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
            self.add_policy_row(accumulator, row, sign);
        }
    }

    pub(crate) fn add_policy_row(
        &self,
        accumulator: &mut [f32; POLICY_ACCUMULATOR_RANK],
        row: usize,
        sign: i32,
    ) {
        let start = row * POLICY_ACCUMULATOR_RANK;
        let values = &self.policy_accumulator_features[start..start + POLICY_ACCUMULATOR_RANK];
        for (target, &value) in accumulator.iter_mut().zip(values) {
            *target += sign as f32 * value;
        }
    }

    pub(crate) fn validate(&self) -> io::Result<()> {
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
        if self
            .value_king_piece_hidden
            .iter()
            .chain(&self.value_king_piece_projection)
            .any(|weight| !weight.is_finite())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "az king-piece factors have invalid parameters",
            ));
        }
        if self.value_history_output.iter().any(|x| !x.is_finite())
            || !self.value_history_cache.valid()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "az history head has invalid parameters or derived cache",
            ));
        }
        if self.policy_accumulator_features.len()
            != POLICY_ACCUMULATOR_ROWS * POLICY_ACCUMULATOR_RANK
            || self.policy_accumulator_move.len() != DENSE_MOVE_SPACE * POLICY_ACCUMULATOR_RANK
            || self.policy_accumulator_moved_delta.len()
                != DENSE_MOVE_SPACE * POLICY_CACHE_PIECE_SIZE
            || self.policy_accumulator_capture.len() != DENSE_MOVE_SPACE * POLICY_CACHE_PIECE_SIZE
            || self.policy_sparse_table_folded.len() != POLICY_CACHE_TABLE_SIZE
            || self.policy_tactical_folded.len() != POLICY_TACTICAL_EXACT_SIZE
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "az model derived policy accumulator cache length mismatch",
            ));
        }
        Ok(())
    }
}

pub(crate) fn softmax_fixed3(logits: [f32; 3]) -> [f32; 3] {
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

pub fn outputs_for_training_sample(
    model: &AzNnue,
    sample: &AzTrainingSample,
) -> Option<([f32; WDL_HEAD_SIZE], Vec<f32>)> {
    let position = position_for_training_sample(sample)?;
    let moves = sample
        .move_indices
        .iter()
        .filter_map(|&index| dense_move_squares(index))
        .map(|(from, to)| Move::new(from, to))
        .collect::<Vec<_>>();
    if moves.len() != sample.move_indices.len() {
        return None;
    }
    let mut scratch = AzEvalScratch::new(model.arch);
    let evaluated = model.evaluate_with_scratch_output_with_repetition_and_history_features(
        &position,
        &moves,
        &sample.repetition_flags,
        &sample.rule_context,
        &sample.history_features,
        &mut scratch,
    );
    Some((evaluated.value_wdl, scratch.logits))
}

pub fn position_for_training_sample(sample: &AzTrainingSample) -> Option<Position> {
    let pieces = sample
        .features
        .iter()
        .filter_map(|&feature| decode_current_piece_square_feature(feature))
        .map(|piece| (piece.piece_index, piece.rank * BOARD_FILES + piece.file))
        .collect::<Vec<_>>();
    let position = Position::from_canonical_piece_squares(&pieces);
    (position.has_general(Color::Red) && position.has_general(Color::Black)).then_some(position)
}

pub(crate) fn scalar_value_to_wdl_target(value: f32) -> [f32; 3] {
    let value = value.clamp(-1.0, 1.0);
    if value >= 0.0 {
        [value, 1.0 - value, 0.0]
    } else {
        [0.0, 1.0 + value, -value]
    }
}

pub(crate) fn normalize_wdl_target(mut wdl: [f32; WDL_HEAD_SIZE]) -> [f32; WDL_HEAD_SIZE] {
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
