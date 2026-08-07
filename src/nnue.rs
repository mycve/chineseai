use crate::xiangqi::{
    BOARD_FILES, BOARD_SIZE, Color, Move, PieceKind, Position,
};

pub const CANONICAL_PIECE_INPUT_SIZE: usize = BOARD_SIZE * 14;
pub const V2_KING_BUCKETS: usize = 9;
/// 当前网络仅使用面向行棋方的 canonical 棋子位置，不输入历史步。
/// 重复、长将、长捉等依赖历史的规则由环境精确处理。
pub const AZ_NNUE_INPUT_SIZE: usize = CANONICAL_PIECE_INPUT_SIZE;

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
pub fn fill_sparse_features_az(position: &Position, features: &mut Vec<usize>) {
    features.clear();
    features.reserve(32);
    let side = position.side_to_move();
    for sq in 0..BOARD_SIZE {
        let Some(piece) = position.piece_at(sq) else {
            continue;
        };
        let relative_color = if piece.color == side {
            Color::Red
        } else {
            Color::Black
        };
        let relative_square = orient_square(side, sq);
        let piece_index = absolute_piece_index(relative_color, piece.kind);
        features.push(piece_index * BOARD_SIZE + relative_square);
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

fn absolute_piece_index(color: Color, kind: PieceKind) -> usize {
    let base = if color == Color::Red { 0 } else { 7 };
    base + match kind {
        PieceKind::General => 0,
        PieceKind::Advisor => 1,
        PieceKind::Elephant => 2,
        PieceKind::Horse => 3,
        PieceKind::Rook => 4,
        PieceKind::Cannon => 5,
        PieceKind::Soldier => 6,
    }
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
    fn az_features_use_side_to_move_canonical_coordinates() {
        let position = Position::from_fen("4k4/9/9/9/4p4/9/9/9/9/4K4 b - - 0 1").unwrap();
        let features = extract_sparse_features_az(&position);
        let us_general = absolute_piece_index(Color::Red, PieceKind::General) * BOARD_SIZE + 85;
        let them_general = absolute_piece_index(Color::Black, PieceKind::General) * BOARD_SIZE + 4;

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
