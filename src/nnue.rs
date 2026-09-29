#[cfg(test)]
use crate::xiangqi::PieceKind;
use crate::xiangqi::{BOARD_FILES, BOARD_SIZE, Color, Move, Piece, Position, piece_kind_index};

pub const CANONICAL_PIECE_INPUT_SIZE: usize = BOARD_SIZE * 14;
pub const V2_KING_BUCKETS: usize = 9;
/// 当前网络仅使用面向行棋方的 canonical 棋子位置，不输入历史步。
/// 重复、长将、长捉等依赖历史的规则由环境精确处理。
pub const AB_NNUE_INPUT_SIZE: usize = CANONICAL_PIECE_INPUT_SIZE;

#[path = "nnue/full_threats.rs"]
mod full_threats;

/// Pikafish 当前 HalfKAv2_hm + FullThreats 网络的权重形状。
/// 这些常量是训练格式约束；现有 AB 模型仍使用上面的独立特征编码。
pub mod pikafish {
    use crate::xiangqi::{BOARD_SIZE, Color, Piece, PieceKind, Position};

    pub const KING_BUCKETS: usize = 6;
    pub const ATTACK_BUCKETS: usize = 4;
    pub const PSQ_FEATURES_PER_BUCKET: usize = 689;
    pub const PSQ_INPUTS: usize = KING_BUCKETS * ATTACK_BUCKETS * PSQ_FEATURES_PER_BUCKET;
    pub const THREAT_INPUTS: usize = 45_547;
    pub use super::full_threats::{fill_threat_features, threat_index};
    pub const TRANSFORMER_WIDTH: usize = 1_024;
    pub const TRANSFORMED_PER_PERSPECTIVE: usize = TRANSFORMER_WIDTH / 2;
    pub const NETWORK_INPUTS: usize = TRANSFORMER_WIDTH;
    pub const HIDDEN_WIDTH: usize = 32;
    pub const PSQT_BUCKETS: usize = 16;
    pub const LAYER_STACKS: usize = 16;
    pub const WEIGHT_SCALE_BITS: usize = 6;
    pub const FT_MAX: usize = 255;
    pub const HIDDEN_ONE: usize = 128;
    pub const OUTPUT_SCALE: usize = 16;

    /// 每一张权重表的逻辑形状；量化导出时变为 i16/i8/i32。
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct TensorSpec {
        pub name: &'static str,
        pub shape: &'static [usize],
        pub export_dtype: &'static str,
    }

    pub const WEIGHT_LAYOUT: [TensorSpec; 11] = [
        TensorSpec {
            name: "ft_bias",
            shape: &[TRANSFORMER_WIDTH],
            export_dtype: "i16",
        },
        TensorSpec {
            name: "ft_psq",
            shape: &[PSQ_INPUTS, TRANSFORMER_WIDTH],
            export_dtype: "i8",
        },
        TensorSpec {
            name: "ft_threat",
            shape: &[THREAT_INPUTS, TRANSFORMER_WIDTH],
            export_dtype: "i8",
        },
        TensorSpec {
            name: "psqt",
            shape: &[PSQ_INPUTS, PSQT_BUCKETS],
            export_dtype: "i32",
        },
        TensorSpec {
            name: "threat_psqt",
            shape: &[THREAT_INPUTS, PSQT_BUCKETS],
            export_dtype: "i32",
        },
        TensorSpec {
            name: "fc0_weight",
            shape: &[LAYER_STACKS, HIDDEN_WIDTH, NETWORK_INPUTS],
            export_dtype: "i8",
        },
        TensorSpec {
            name: "fc0_bias",
            shape: &[LAYER_STACKS, HIDDEN_WIDTH],
            export_dtype: "i32",
        },
        TensorSpec {
            name: "fc1_weight",
            shape: &[LAYER_STACKS, HIDDEN_WIDTH, HIDDEN_WIDTH * 2],
            export_dtype: "i8",
        },
        TensorSpec {
            name: "fc1_bias",
            shape: &[LAYER_STACKS, HIDDEN_WIDTH],
            export_dtype: "i32",
        },
        TensorSpec {
            name: "fc2_weight",
            shape: &[LAYER_STACKS, 1, HIDDEN_WIDTH * 4],
            export_dtype: "i8",
        },
        TensorSpec {
            name: "fc2_bias",
            shape: &[LAYER_STACKS, 1],
            export_dtype: "i32",
        },
    ];

    const BALANCE_ENCODING: u64 = 0xa4a9_2a74_e989_d3a7;
    const INVALID_INDEX: usize = usize::MAX;
    const KINDS: [PieceKind; 7] = [
        PieceKind::Rook,
        PieceKind::Advisor,
        PieceKind::Cannon,
        PieceKind::Soldier,
        PieceKind::Horse,
        PieceKind::Elephant,
        PieceKind::General,
    ];

    // 官方 Square A0 为红方底线；本项目棋盘数组从黑方底线开始。
    fn native_square(square: usize) -> usize {
        (9 - square / 9) * 9 + square % 9
    }

    fn piece_plane(piece: Piece) -> usize {
        let color = usize::from(piece.color == Color::Black) * 7;
        color + KINDS.iter().position(|kind| *kind == piece.kind).unwrap()
    }

    pub(super) fn valid_square(plane: usize, square: usize) -> bool {
        let rank = square / 9;
        let file = square % 9;
        let black = plane >= 7;
        match plane % 7 {
            0 | 2 | 4 => true,
            1 => {
                let (back, middle, front) = if black { (9, 8, 7) } else { (0, 1, 2) };
                ((rank == back || rank == front) && (file == 3 || file == 5))
                    || (rank == middle && file == 4)
            }
            3 => {
                if black {
                    rank <= 4 || ((5..=6).contains(&rank) && file % 2 == 0)
                } else {
                    rank >= 5 || ((3..=4).contains(&rank) && file % 2 == 0)
                }
            }
            5 => {
                let (back, middle, front) = if black { (9, 7, 5) } else { (0, 2, 4) };
                ((rank == back || rank == front) && (file == 2 || file == 6))
                    || (rank == middle && (file == 0 || file == 4 || file == 8))
            }
            6 => {
                let in_palace = if black { rank >= 7 } else { rank <= 2 };
                in_palace && (3..=5).contains(&file) && (black || file != 5)
            }
            _ => unreachable!(),
        }
    }

    fn psq_offsets() -> &'static [[usize; BOARD_SIZE]; 14] {
        static OFFSETS: std::sync::OnceLock<[[usize; BOARD_SIZE]; 14]> = std::sync::OnceLock::new();
        OFFSETS.get_or_init(|| {
            let mut offsets = [[INVALID_INDEX; BOARD_SIZE]; 14];
            let mut next = 0;
            for (plane, row) in offsets.iter_mut().enumerate() {
                for (square, index) in row.iter_mut().enumerate() {
                    if valid_square(plane, square) {
                        *index = next;
                        next += 1;
                    }
                }
            }
            assert_eq!(next, PSQ_FEATURES_PER_BUCKET);
            offsets
        })
    }

    fn king_bucket(square: usize) -> (usize, bool) {
        let rank = square / 9;
        let file = square % 9;
        let bucket = match rank {
            0 | 9 => match file {
                4 => 1,
                _ => 0,
            },
            1 | 8 => match file {
                3 | 5 => 2,
                4 => 3,
                _ => 0,
            },
            2 | 7 => match file {
                3 | 5 => 4,
                4 => 5,
                _ => 0,
            },
            _ => 0,
        };
        (bucket, file == 5 && matches!(rank, 0..=2 | 7..=9))
    }

    fn mid_encoding(position: &Position, color: Color) -> u64 {
        let mut encoding = BALANCE_ENCODING;
        for square in 0..BOARD_SIZE {
            let Some(piece) = position.piece_at(square) else {
                continue;
            };
            if piece.color != color {
                continue;
            }
            let native = native_square(square);
            let file = native % 9;
            let rank = native / 9;
            let contribution = if piece.kind == PieceKind::General {
                if file != 4 { 1_u64 << 63 } else { 0 }
            } else if file == 4 {
                0
            } else {
                let (count_shift, square_shift) = match piece.kind {
                    PieceKind::Rook => (44, 0),
                    PieceKind::Advisor => (60, 36),
                    PieceKind::Cannon => (47, 7),
                    PieceKind::Soldier => (53, 21),
                    PieceKind::Horse => (50, 14),
                    PieceKind::Elephant => (57, 29),
                    PieceKind::General => unreachable!(),
                };
                let oriented_rank = if color == Color::Red { rank } else { 9 - rank };
                let left_file = if file < 4 { file } else { 8 - file };
                let value = (1_u64 << count_shift)
                    | (((3 - left_file) * 10 + oriented_rank) as u64) << square_shift;
                if file < 4 {
                    value
                } else {
                    value.wrapping_neg()
                }
            };
            encoding = encoding.wrapping_add(contribution);
        }
        encoding
    }

    /// 官方 KingBuckets 与中线对称决策；返回 0..23 的 HalfKAv2_hm 桶。
    pub fn feature_bucket(position: &Position, perspective: Color) -> Option<(usize, bool)> {
        let mut king = None;
        let mut opponent_king = None;
        for square in 0..BOARD_SIZE {
            let Some(piece) = position.piece_at(square) else {
                continue;
            };
            if piece.kind == PieceKind::General {
                if piece.color == perspective {
                    king = Some(native_square(square));
                } else {
                    opponent_king = Some(native_square(square));
                }
            }
        }
        let (king, opponent_king) = (king?, opponent_king?);
        let (bucket, mirrored_king) = king_bucket(king);
        let (opponent_bucket, mirrored_opponent) = king_bucket(opponent_king);
        let own_encoding = mid_encoding(position, perspective);
        let opponent_encoding = mid_encoding(position, perspective.opposite());
        let mid_mirror = (own_encoding & opponent_encoding & (1_u64 << 63)) != 0
            && (own_encoding < BALANCE_ENCODING
                || (own_encoding == BALANCE_ENCODING && opponent_encoding < BALANCE_ENCODING));
        let mirror = mirrored_king
            || (bucket & 1 != 0 && (mirrored_opponent || (opponent_bucket & 1 != 0 && mid_mirror)));
        Some((
            bucket * ATTACK_BUCKETS + attack_bucket(position, perspective),
            mirror,
        ))
    }

    /// 精确的 PSQ 稀疏编号。FullThreats 输入须另外追加到独立权重表。
    pub fn fill_psq_features(
        position: &Position,
        perspective: Color,
        output: &mut Vec<usize>,
    ) -> Option<()> {
        let (bucket, mirror) = feature_bucket(position, perspective)?;
        output.clear();
        output.reserve(32);
        for square in 0..BOARD_SIZE {
            let Some(mut piece) = position.piece_at(square) else {
                continue;
            };
            let mut native = native_square(square);
            if mirror {
                native = (native / 9) * 9 + 8 - native % 9;
            }
            if perspective == Color::Black {
                native = (9 - native / 9) * 9 + native % 9;
                piece.color = piece.color.opposite();
            }
            let offset = psq_offsets()[piece_plane(piece)][native];
            if offset == INVALID_INDEX {
                return None;
            }
            output.push(bucket * PSQ_FEATURES_PER_BUCKET + offset);
        }
        Some(())
    }

    /// 对应 HalfKAv2_hm::make_attack_bucket，仅依赖己方车、马、炮数量。
    pub fn attack_bucket(position: &Position, perspective: Color) -> usize {
        let mut rooks = 0;
        let mut horse_or_cannon = 0;
        for square in 0..BOARD_SIZE {
            let Some(piece) = position.piece_at(square) else {
                continue;
            };
            if piece.color != perspective {
                continue;
            }
            match piece.kind {
                PieceKind::Rook => rooks += 1,
                PieceKind::Horse | PieceKind::Cannon => horse_or_cannon += 1,
                _ => {}
            }
        }
        usize::from(rooks > 0) * 2 + usize::from(horse_or_cannon > 0)
    }

    /// 对应 HalfKAv2_hm::make_layer_stack_bucket，按行棋方的主力子力分 16 桶。
    pub fn layer_stack_bucket(position: &Position) -> usize {
        let side = position.side_to_move();
        let mut rooks = [0_usize; 2];
        let mut horse_cannons = [0_usize; 2];
        for square in 0..BOARD_SIZE {
            let Some(piece) = position.piece_at(square) else {
                continue;
            };
            let index = usize::from(piece.color != side);
            match piece.kind {
                PieceKind::Rook => rooks[index] += 1,
                PieceKind::Horse | PieceKind::Cannon => horse_cannons[index] += 1,
                _ => {}
            }
        }
        match (rooks[0], rooks[1]) {
            (us, them) if us == them => {
                us * 4
                    + usize::from(horse_cannons[0] + horse_cannons[1] >= 4) * 2
                    + usize::from(horse_cannons[0] == horse_cannons[1])
            }
            (2, 1) => 12,
            (1, 2) => 13,
            (us, 0) if us > 0 => 14,
            _ => 15,
        }
    }
}

pub fn extract_sparse_features_ab(position: &Position) -> Vec<usize> {
    let mut features = Vec::with_capacity(96);
    fill_sparse_features_ab(position, &mut features);
    features.sort_unstable();
    features
}

/// 填充面向走子方的 NNUE 稀疏特征，复用调用方缓冲区。
///
/// 推理只对特征行求和，不依赖特征顺序，因此热路径不做排序，也不产生堆分配。
/// 需要稳定顺序（例如序列化或测试）时使用 `extract_sparse_features_ab`。
/// 将棋子映射到当前视角的 14 个通道：[0, 6] 是己方，[7, 13] 是对方。
/// 这里只转换颜色和棋种，坐标由 `canonical_square` 单独转换。
#[inline]
pub fn piece_absolute_feature_index(perspective: Color, piece: Piece) -> usize {
    let base = if piece.color == perspective { 0 } else { 7 };
    base + piece_kind_index(piece.kind)
}

#[inline]
pub fn fill_sparse_features_ab(position: &Position, features: &mut Vec<usize>) {
    features.clear();
    features.reserve(32);
    let side = position.side_to_move();
    for sq in 0..BOARD_SIZE {
        let Some(piece) = position.piece_at(sq) else {
            continue;
        };
        features.push(
            piece_absolute_feature_index(side, piece) * BOARD_SIZE + canonical_square(side, sq),
        );
    }
}

pub fn mirror_file_square(sq: usize) -> usize {
    let rank = sq / BOARD_FILES;
    let file = sq % BOARD_FILES;
    rank * BOARD_FILES + (BOARD_FILES - 1 - file)
}

pub fn mirror_file_move(mv: Move) -> Move {
    Move::new(
        mirror_file_square(mv.from as usize),
        mirror_file_square(mv.to as usize),
    )
}

pub fn canonical_square(side: Color, sq: usize) -> usize {
    orient_square(side, sq)
}

pub fn canonical_move(side: Color, mv: Move) -> Move {
    Move::new(
        canonical_square(side, mv.from as usize),
        canonical_square(side, mv.to as usize),
    )
}

pub fn mirror_sparse_features_ab_canonical_file(features: &mut [usize]) {
    for feature in features.iter_mut() {
        if *feature < CANONICAL_PIECE_INPUT_SIZE {
            let piece_index = *feature / BOARD_SIZE;
            let sq = *feature % BOARD_SIZE;
            *feature = piece_index * BOARD_SIZE + mirror_file_square(sq);
        }
    }
    features.sort_unstable();
}

fn orient_square(side: Color, sq: usize) -> usize {
    match side {
        Color::Red => sq,
        Color::Black => BOARD_SIZE - 1 - sq,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirror_file_move_flips_left_and_right() {
        assert_eq!(mirror_file_move(Move::new(47, 38)), Move::new(51, 42));
    }

    #[test]
    fn ab_features_use_side_to_move_canonical_coordinates() {
        let position = Position::from_fen("4k4/9/9/9/4p4/9/9/9/9/4K4 b - - 0 1").unwrap();
        let features = extract_sparse_features_ab(&position);
        let side = position.side_to_move();
        let us_general = piece_absolute_feature_index(
            side,
            Piece {
                color: side,
                kind: PieceKind::General,
            },
        ) * BOARD_SIZE
            + canonical_square(side, 4);
        let them_general = piece_absolute_feature_index(
            side,
            Piece {
                color: side.opposite(),
                kind: PieceKind::General,
            },
        ) * BOARD_SIZE
            + canonical_square(side, 85);

        assert!(features.contains(&us_general));
        assert!(features.contains(&them_general));
    }

    #[test]
    fn ab_features_use_only_current_board() {
        let position = Position::startpos();
        let features = extract_sparse_features_ab(&position);
        assert_eq!(AB_NNUE_INPUT_SIZE, 1_260);
        assert!(features.iter().all(|&feature| feature < AB_NNUE_INPUT_SIZE));
        assert_eq!(features.len(), 32);
    }

    #[test]
    fn pikafish_weight_shape_and_start_buckets() {
        use pikafish::*;
        assert_eq!(PSQ_INPUTS, 16_536);
        assert_eq!(WEIGHT_LAYOUT.len(), 11);
        assert_eq!(WEIGHT_LAYOUT[1].shape, &[16_536, 1_024]);
        assert_eq!(WEIGHT_LAYOUT[2].shape, &[45_547, 1_024]);
        assert_eq!(WEIGHT_LAYOUT[5].shape, &[16, 32, 1_024]);
        let position = Position::startpos();
        assert_eq!(attack_bucket(&position, Color::Red), 3);
        assert_eq!(attack_bucket(&position, Color::Black), 3);
        assert_eq!(layer_stack_bucket(&position), 11);
        for perspective in [Color::Red, Color::Black] {
            let mut features = Vec::new();
            fill_psq_features(&position, perspective, &mut features).unwrap();
            assert_eq!(features.len(), 32);
            assert!(features.iter().all(|&index| index < PSQ_INPUTS));
        }
    }

    #[test]
    fn pikafish_psq_mirroring_is_canonical() {
        use pikafish::*;
        let position = Position::from_fen("4k4/9/9/9/9/9/9/9/9/3K1R3 w - - 0 1").unwrap();
        let mirrored = position.mirror_files();
        for perspective in [Color::Red, Color::Black] {
            let mut first = Vec::new();
            let mut second = Vec::new();
            fill_psq_features(&position, perspective, &mut first).unwrap();
            fill_psq_features(&mirrored, perspective, &mut second).unwrap();
            first.sort_unstable();
            second.sort_unstable();
            assert_eq!(first, second);
        }
    }
}
