#[cfg(test)]
use crate::xiangqi::PieceKind;
use crate::xiangqi::{BOARD_FILES, BOARD_SIZE, Color, Move, Piece, Position, piece_kind_index};

pub const CANONICAL_PIECE_INPUT_SIZE: usize = BOARD_SIZE * 14;
pub const V2_KING_BUCKETS: usize = 9;
/// 每份视角使用1260维棋子位置；网络同时读取行棋方与对方两份累加器。
/// 重复、长将、长捉等规则由环境精确处理。
pub const AZ_NNUE_INPUT_SIZE: usize = CANONICAL_PIECE_INPUT_SIZE;

/// 同一棋子位置在另一份视角中的索引；变换两次回到原特征。
#[inline]
pub(crate) const fn opposite_feature(feature: usize) -> usize {
    let piece = feature / BOARD_SIZE;
    ((piece + 7) % 14) * BOARD_SIZE + (BOARD_SIZE - 1 - feature % BOARD_SIZE)
}

/// 面向当前视角的棋子位置几何约束，不考虑占用、蹩马腿或塞象眼。
pub(crate) fn canonical_piece_square_reachable(piece: usize, square: usize) -> bool {
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

pub fn extract_sparse_features_az(position: &Position) -> Vec<usize> {
    let mut features = Vec::with_capacity(96);
    fill_sparse_features_az(position, &mut features);
    features.sort_unstable();
    features
}

/// 填充面向走子方的 NNUE 稀疏特征，复用调用方缓冲区。
///
/// 推理只对特征行求和，不依赖特征顺序，因此热路径不做排序，也不产生堆分配。
/// 需要稳定顺序（例如序列化或测试）时使用 `extract_sparse_features_az`。
#[inline]
pub fn piece_absolute_feature_index(perspective: Color, piece: Piece) -> usize {
    let base = if piece.color == perspective { 0 } else { 7 };
    base + piece_kind_index(piece.kind)
}

#[inline]
pub fn fill_sparse_features_az(position: &Position, features: &mut Vec<usize>) {
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

pub fn mirror_sparse_features_az_canonical_file(features: &mut [usize]) {
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
    fn opposite_view_matches_board_and_is_reversible() {
        for feature in 0..AZ_NNUE_INPUT_SIZE {
            assert_eq!(opposite_feature(opposite_feature(feature)), feature);
        }
        let position = Position::from_fen("4k4/9/9/9/4p4/9/9/9/9/4K4 b - - 0 1").unwrap();
        let mut derived = extract_sparse_features_az(&position)
            .into_iter()
            .map(opposite_feature)
            .collect::<Vec<_>>();
        let other = position.side_to_move().opposite();
        let mut expected = position
            .occupied_squares()
            .map(|square| {
                piece_absolute_feature_index(other, position.piece_at(square).unwrap()) * BOARD_SIZE
                    + canonical_square(other, square)
            })
            .collect::<Vec<_>>();
        derived.sort_unstable();
        expected.sort_unstable();
        assert_eq!(derived, expected);
    }

    #[test]
    fn mirror_file_move_flips_left_and_right() {
        assert_eq!(mirror_file_move(Move::new(47, 38)), Move::new(51, 42));
    }

    #[test]
    fn az_features_use_side_to_move_canonical_coordinates() {
        let position = Position::from_fen("4k4/9/9/9/4p4/9/9/9/9/4K4 b - - 0 1").unwrap();
        let features = extract_sparse_features_az(&position);
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
    fn az_features_use_only_current_board() {
        let position = Position::startpos();
        let features = extract_sparse_features_az(&position);
        assert_eq!(AZ_NNUE_INPUT_SIZE, 1_260);
        assert!(features.iter().all(|&feature| feature < AZ_NNUE_INPUT_SIZE));
        assert_eq!(features.len(), 32);
    }
}
