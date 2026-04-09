use crate::xiangqi::{BOARD_SIZE, Color, PieceKind, Position};

pub const INPUT_SIZE: usize = BOARD_SIZE * 14 + 1;
pub const V2_KING_BUCKETS: usize = 9;
pub const V2_INPUT_SIZE: usize = INPUT_SIZE + 2 * V2_KING_BUCKETS * 14 * BOARD_SIZE;

pub fn extract_sparse_features_v2(position: &Position) -> Vec<usize> {
    let red_king_bucket = position
        .general_square(Color::Red)
        .map(|sq| palace_bucket(Color::Red, sq))
        .unwrap_or(4);
    let black_king_bucket = position
        .general_square(Color::Black)
        .map(|sq| palace_bucket(Color::Black, sq))
        .unwrap_or(4);
    let mut features = Vec::with_capacity(97);

    for sq in 0..BOARD_SIZE {
        let Some(piece) = position.piece_at(sq) else {
            continue;
        };
        let piece_index = color_piece_index(piece.color, piece.kind);
        features.push(piece_index * BOARD_SIZE + sq);
        features.push(king_aware_feature_index(
            0,
            red_king_bucket,
            piece_index,
            sq,
        ));
        features.push(king_aware_feature_index(
            1,
            black_king_bucket,
            piece_index,
            sq,
        ));
    }

    if position.side_to_move() == Color::Red {
        features.push(INPUT_SIZE - 1);
    }

    features
}

fn king_aware_feature_index(
    perspective: usize,
    king_bucket: usize,
    piece_index: usize,
    sq: usize,
) -> usize {
    INPUT_SIZE
        + (((perspective * V2_KING_BUCKETS + king_bucket) * 14 + piece_index) * BOARD_SIZE + sq)
}

fn palace_bucket(color: Color, sq: usize) -> usize {
    let file = (sq % 9).clamp(3, 5) - 3;
    let rank = sq / 9;
    let rank = match color {
        Color::Red => rank.clamp(7, 9) - 7,
        Color::Black => rank.clamp(0, 2),
    };
    rank * 3 + file
}

fn color_piece_index(color: Color, kind: PieceKind) -> usize {
    let base = match color {
        Color::Red => 0,
        Color::Black => 7,
    };
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v2_features_include_base_and_king_aware_inputs() {
        let position = Position::startpos();
        let features = extract_sparse_features_v2(&position);

        assert_eq!(features.len(), 32 * 3 + 1);
        assert!(features.iter().all(|feature| *feature < V2_INPUT_SIZE));
        assert!(features.contains(&(INPUT_SIZE - 1)));
        assert!(features.iter().any(|feature| *feature >= INPUT_SIZE));
    }
}
