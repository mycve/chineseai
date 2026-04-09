use crate::xiangqi::{BOARD_FILES, BOARD_SIZE, Color, Move, Piece, PieceKind, Position};

pub const INPUT_SIZE: usize = BOARD_SIZE * 14 + 1;
pub const V2_KING_BUCKETS: usize = 9;
pub const V2_INPUT_SIZE: usize = INPUT_SIZE + 2 * V2_KING_BUCKETS * 14 * BOARD_SIZE;
pub const HISTORY_PLIES: usize = 8;
const HISTORY_EVENT_TYPES: usize = 2;
const HISTORY_INPUT_SIZE: usize = HISTORY_PLIES * HISTORY_EVENT_TYPES * 14 * BOARD_SIZE;
pub const V3_INPUT_SIZE: usize = V2_INPUT_SIZE + HISTORY_INPUT_SIZE;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HistoryMove {
    pub piece: Piece,
    pub mv: Move,
}

pub fn extract_sparse_features_v2(position: &Position) -> Vec<usize> {
    let side = position.side_to_move();
    let us_king_bucket = position
        .general_square(side)
        .map(|sq| palace_bucket(orient_square(side, sq)))
        .unwrap_or(4);
    let them_king_bucket = position
        .general_square(side.opposite())
        .map(|sq| palace_bucket(orient_square(side, sq)))
        .unwrap_or(4);
    let mut features = Vec::with_capacity(97);

    for sq in 0..BOARD_SIZE {
        let Some(piece) = position.piece_at(sq) else {
            continue;
        };
        let oriented_sq = orient_square(side, sq);
        let piece_index = relative_piece_index(side, piece.color, piece.kind);
        features.push(piece_index * BOARD_SIZE + oriented_sq);
        features.push(king_aware_feature_index(
            0,
            us_king_bucket,
            piece_index,
            oriented_sq,
        ));
        features.push(king_aware_feature_index(
            1,
            them_king_bucket,
            piece_index,
            oriented_sq,
        ));
    }

    features.push(INPUT_SIZE - 1);
    features.sort_unstable();
    features
}

pub fn extract_sparse_features_v3(position: &Position, history: &[HistoryMove]) -> Vec<usize> {
    let side = position.side_to_move();
    let mut features = extract_sparse_features_v2(position);
    features.reserve(history.len().min(HISTORY_PLIES) * HISTORY_EVENT_TYPES);
    for (age, entry) in history.iter().rev().take(HISTORY_PLIES).enumerate() {
        let piece_index = relative_piece_index(side, entry.piece.color, entry.piece.kind);
        features.push(history_feature_index(
            age,
            0,
            piece_index,
            orient_square(side, entry.mv.from as usize),
        ));
        features.push(history_feature_index(
            age,
            1,
            piece_index,
            orient_square(side, entry.mv.to as usize),
        ));
    }
    features.sort_unstable();
    features
}

pub fn orient_move(side: Color, mv: Move) -> Move {
    Move::new(
        orient_square(side, mv.from as usize),
        orient_square(side, mv.to as usize),
    )
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

pub fn mirror_sparse_features_file(features: &mut [usize]) {
    for feature in features.iter_mut() {
        *feature = mirror_sparse_feature_file(*feature);
    }
    features.sort_unstable();
}

fn mirror_sparse_feature_file(feature: usize) -> usize {
    if feature == INPUT_SIZE - 1 {
        return feature;
    }
    if feature < INPUT_SIZE - 1 {
        let piece_index = feature / BOARD_SIZE;
        let sq = feature % BOARD_SIZE;
        return piece_index * BOARD_SIZE + mirror_file_square(sq);
    }
    if feature < V2_INPUT_SIZE {
        let offset = feature - INPUT_SIZE;
        let sq = offset % BOARD_SIZE;
        let partial = offset / BOARD_SIZE;
        let piece_index = partial % 14;
        let partial = partial / 14;
        let king_bucket = partial % V2_KING_BUCKETS;
        let perspective = partial / V2_KING_BUCKETS;
        return king_aware_feature_index(
            perspective,
            mirror_palace_bucket_file(king_bucket),
            piece_index,
            mirror_file_square(sq),
        );
    }
    if feature < V3_INPUT_SIZE {
        let offset = feature - V2_INPUT_SIZE;
        let sq = offset % BOARD_SIZE;
        let partial = offset / BOARD_SIZE;
        let piece_index = partial % 14;
        let partial = partial / 14;
        let event = partial % HISTORY_EVENT_TYPES;
        let age = partial / HISTORY_EVENT_TYPES;
        return history_feature_index(age, event, piece_index, mirror_file_square(sq));
    }
    feature
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

fn history_feature_index(age: usize, event: usize, piece_index: usize, sq: usize) -> usize {
    V2_INPUT_SIZE + (((age * HISTORY_EVENT_TYPES + event) * 14 + piece_index) * BOARD_SIZE + sq)
}

fn palace_bucket(sq: usize) -> usize {
    let file = (sq % BOARD_FILES).clamp(3, 5) - 3;
    let rank = (sq / BOARD_FILES).clamp(7, 9) - 7;
    rank * 3 + file
}

fn mirror_palace_bucket_file(bucket: usize) -> usize {
    let rank = bucket / 3;
    let file = bucket % 3;
    rank * 3 + (2 - file)
}

fn relative_piece_index(side: Color, color: Color, kind: PieceKind) -> usize {
    let base = if color == side { 0 } else { 7 };
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
    fn v3_features_include_base_king_aware_and_history_inputs() {
        let position = Position::startpos();
        let history = [HistoryMove {
            piece: position.piece_at(64).unwrap(),
            mv: Move::new(64, 55),
        }];
        let features = extract_sparse_features_v3(&position, &history);

        assert_eq!(features.len(), 32 * 3 + 1 + 2);
        assert!(features.iter().all(|feature| *feature < V3_INPUT_SIZE));
        assert!(features.contains(&(INPUT_SIZE - 1)));
        assert!(features.iter().any(|feature| *feature >= INPUT_SIZE));
        assert!(features.iter().any(|feature| *feature >= V2_INPUT_SIZE));
    }

    #[test]
    fn features_are_relative_to_side_to_move() {
        let red_to_move = Position::from_fen("4k4/9/9/9/4p4/4P4/9/9/9/4K4 w").unwrap();
        let black_to_move = Position::from_fen("4k4/9/9/9/4p4/4P4/9/9/9/4K4 b").unwrap();

        assert_eq!(
            extract_sparse_features_v3(&red_to_move, &[]),
            extract_sparse_features_v3(&black_to_move, &[])
        );
    }

    #[test]
    fn history_features_are_relative_to_side_to_move() {
        let red_to_move = Position::from_fen("4k4/9/9/9/4p4/4P4/9/9/9/4K4 w").unwrap();
        let black_to_move = Position::from_fen("4k4/9/9/9/4p4/4P4/9/9/9/4K4 b").unwrap();
        let red_history = [HistoryMove {
            piece: red_to_move.piece_at(49).unwrap(),
            mv: Move::new(49, 40),
        }];
        let black_history = [HistoryMove {
            piece: black_to_move.piece_at(40).unwrap(),
            mv: Move::new(40, 49),
        }];

        assert_eq!(
            extract_sparse_features_v3(&red_to_move, &red_history),
            extract_sparse_features_v3(&black_to_move, &black_history)
        );
    }

    #[test]
    fn mirror_file_move_flips_left_and_right() {
        assert_eq!(mirror_file_move(Move::new(47, 38)), Move::new(51, 42));
    }

    #[test]
    fn mirrored_features_match_mirrored_position_and_history() {
        let left = Position::from_fen("4k4/9/9/9/3pp4/5P3/9/9/9/4K4 w").unwrap();
        let right = Position::from_fen("4k4/9/9/9/4pp3/3P5/9/9/9/4K4 w").unwrap();
        let left_history = [HistoryMove {
            piece: left.piece_at(50).unwrap(),
            mv: Move::new(50, 41),
        }];
        let right_history = [HistoryMove {
            piece: right.piece_at(48).unwrap(),
            mv: Move::new(48, 39),
        }];
        let mut mirrored_left = extract_sparse_features_v3(&left, &left_history);
        mirror_sparse_features_file(&mut mirrored_left);

        assert_eq!(
            mirrored_left,
            extract_sparse_features_v3(&right, &right_history)
        );
    }
}
