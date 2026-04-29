use crate::xiangqi::{
    BOARD_FILES, BOARD_RANKS, BOARD_SIZE, Color, Move, Piece, PieceKind, Position,
};

pub const INPUT_SIZE: usize = BOARD_SIZE * 14 + 1;
pub const V2_KING_BUCKETS: usize = 9;
pub const V2_INPUT_SIZE: usize = INPUT_SIZE + 2 * V2_KING_BUCKETS * 14 * BOARD_SIZE;
pub const HISTORY_PLIES: usize = 8;
const HISTORY_EVENT_TYPES: usize = 2;
const HISTORY_INPUT_SIZE: usize = HISTORY_PLIES * HISTORY_EVENT_TYPES * 14 * BOARD_SIZE;
pub const V3_INPUT_SIZE: usize = V2_INPUT_SIZE + HISTORY_INPUT_SIZE;
const ROW_INPUT_SIZE: usize = 14 * BOARD_RANKS;
const COL_INPUT_SIZE: usize = 14 * BOARD_FILES;
const NEIGHBOR_OFFSETS: [(i32, i32); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];
const NEIGHBOR_STATE_COUNT: usize = 14;
const NEIGHBOR_INPUT_SIZE: usize = 14 * NEIGHBOR_OFFSETS.len() * NEIGHBOR_STATE_COUNT;
pub const V4_INPUT_SIZE: usize =
    V3_INPUT_SIZE + ROW_INPUT_SIZE + COL_INPUT_SIZE + NEIGHBOR_INPUT_SIZE;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HistoryMove {
    pub piece: Piece,
    pub captured: Option<Piece>,
    pub mv: Move,
}

pub fn extract_sparse_features_v2(position: &Position) -> Vec<usize> {
    let red_king_bucket = position
        .general_square(Color::Red)
        .map(|sq| palace_bucket_for_color(Color::Red, sq))
        .unwrap_or(4);
    let black_king_bucket = position
        .general_square(Color::Black)
        .map(|sq| palace_bucket_for_color(Color::Black, sq))
        .unwrap_or(4);
    let mut features = Vec::with_capacity(97);

    for sq in 0..BOARD_SIZE {
        let Some(piece) = position.piece_at(sq) else {
            continue;
        };
        let piece_index = absolute_piece_index(piece.color, piece.kind);
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

    if position.side_to_move() == Color::Black {
        features.push(INPUT_SIZE - 1);
    }
    features.sort_unstable();
    features
}

pub fn extract_sparse_features_v3(position: &Position, history: &[HistoryMove]) -> Vec<usize> {
    let mut features = extract_sparse_features_v2(position);
    features.reserve(history.len().min(HISTORY_PLIES) * HISTORY_EVENT_TYPES);
    for (age, entry) in history.iter().rev().take(HISTORY_PLIES).enumerate() {
        let piece_index = absolute_piece_index(entry.piece.color, entry.piece.kind);
        features.push(history_feature_index(
            age,
            0,
            piece_index,
            entry.mv.from as usize,
        ));
        features.push(history_feature_index(
            age,
            1,
            piece_index,
            entry.mv.to as usize,
        ));
    }
    features.sort_unstable();
    features
}

pub fn extract_sparse_features_v4(position: &Position, history: &[HistoryMove]) -> Vec<usize> {
    let mut features = extract_sparse_features_v3(position, history);
    features.reserve(32 * (2 + NEIGHBOR_OFFSETS.len()));
    for sq in 0..BOARD_SIZE {
        let Some(piece) = position.piece_at(sq) else {
            continue;
        };
        let piece_index = absolute_piece_index(piece.color, piece.kind);
        let center_file = file_of(sq) as i32;
        let center_rank = rank_of(sq) as i32;
        features.push(row_feature_index(piece_index, center_rank as usize));
        features.push(col_feature_index(piece_index, center_file as usize));

        for (offset_index, (df, dr)) in NEIGHBOR_OFFSETS.iter().enumerate() {
            let target_file = center_file + df;
            let target_rank = center_rank + dr;
            if !inside_board(target_file, target_rank) {
                continue;
            }
            let target = index(target_file as usize, target_rank as usize);
            let Some(target_piece) = position.piece_at(target) else {
                continue;
            };
            features.push(neighbor_feature_index(
                piece_index,
                offset_index,
                absolute_piece_index(target_piece.color, target_piece.kind),
            ));
        }
    }
    features.sort_unstable();
    features
}

pub fn extract_sparse_features_v4_canonical(
    position: &Position,
    history: &[HistoryMove],
) -> Vec<usize> {
    let side = position.side_to_move();
    let own_king_bucket = position
        .general_square(side)
        .map(|sq| palace_bucket(orient_square(side, sq)))
        .unwrap_or(4);
    let enemy_king_bucket = position
        .general_square(side.opposite())
        .map(|sq| palace_bucket(orient_square(side, sq)))
        .unwrap_or(4);
    let mut features = Vec::with_capacity(160);

    for sq in 0..BOARD_SIZE {
        let Some(piece) = position.piece_at(sq) else {
            continue;
        };
        let piece_index = canonical_piece_index(side, piece);
        let canonical_sq = orient_square(side, sq);
        features.push(piece_index * BOARD_SIZE + canonical_sq);
        features.push(king_aware_feature_index(
            0,
            own_king_bucket,
            piece_index,
            canonical_sq,
        ));
        features.push(king_aware_feature_index(
            1,
            enemy_king_bucket,
            piece_index,
            canonical_sq,
        ));
    }

    for (age, entry) in history.iter().rev().take(HISTORY_PLIES).enumerate() {
        let piece_index = canonical_piece_index(side, entry.piece);
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

    for sq in 0..BOARD_SIZE {
        let Some(piece) = position.piece_at(sq) else {
            continue;
        };
        let piece_index = canonical_piece_index(side, piece);
        let canonical_sq = orient_square(side, sq);
        let center_file = file_of(canonical_sq) as i32;
        let center_rank = rank_of(canonical_sq) as i32;
        features.push(row_feature_index(piece_index, center_rank as usize));
        features.push(col_feature_index(piece_index, center_file as usize));

        for (offset_index, (df, dr)) in NEIGHBOR_OFFSETS.iter().enumerate() {
            let target_file = center_file + df;
            let target_rank = center_rank + dr;
            if !inside_board(target_file, target_rank) {
                continue;
            }
            let canonical_target = index(target_file as usize, target_rank as usize);
            let target = orient_square(side, canonical_target);
            let Some(target_piece) = position.piece_at(target) else {
                continue;
            };
            features.push(neighbor_feature_index(
                piece_index,
                offset_index,
                canonical_piece_index(side, target_piece),
            ));
        }
    }
    features.sort_unstable();
    features
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
        orient_square(side, mv.from as usize),
        orient_square(side, mv.to as usize),
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
    if feature < V4_INPUT_SIZE {
        let offset = feature - V3_INPUT_SIZE;
        if offset < ROW_INPUT_SIZE {
            let piece_index = offset / BOARD_RANKS;
            let rank = offset % BOARD_RANKS;
            return row_feature_index(piece_index, rank);
        }
        let offset = offset - ROW_INPUT_SIZE;
        if offset < COL_INPUT_SIZE {
            let piece_index = offset / BOARD_FILES;
            let file = offset % BOARD_FILES;
            return col_feature_index(piece_index, BOARD_FILES - 1 - file);
        }
        let offset = offset - COL_INPUT_SIZE;
        let neighbor_state = offset % NEIGHBOR_STATE_COUNT;
        let partial = offset / NEIGHBOR_STATE_COUNT;
        let offset_index = partial % NEIGHBOR_OFFSETS.len();
        let piece_index = partial / NEIGHBOR_OFFSETS.len();
        return neighbor_feature_index(
            piece_index,
            mirror_neighbor_offset_index(offset_index),
            neighbor_state,
        );
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

fn row_feature_index(piece_index: usize, rank: usize) -> usize {
    V3_INPUT_SIZE + piece_index * BOARD_RANKS + rank
}

fn col_feature_index(piece_index: usize, file: usize) -> usize {
    V3_INPUT_SIZE + ROW_INPUT_SIZE + piece_index * BOARD_FILES + file
}

fn neighbor_feature_index(piece_index: usize, offset_index: usize, neighbor_state: usize) -> usize {
    V3_INPUT_SIZE
        + ROW_INPUT_SIZE
        + COL_INPUT_SIZE
        + ((piece_index * NEIGHBOR_OFFSETS.len() + offset_index) * NEIGHBOR_STATE_COUNT
            + neighbor_state)
}

fn mirror_neighbor_offset_index(offset_index: usize) -> usize {
    let (df, dr) = NEIGHBOR_OFFSETS[offset_index];
    neighbor_offset_index(-df, dr).expect("mirrored offset must exist")
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

fn canonical_piece_index(side: Color, piece: Piece) -> usize {
    let canonical_color = if piece.color == side {
        Color::Red
    } else {
        Color::Black
    };
    absolute_piece_index(canonical_color, piece.kind)
}

fn palace_bucket_for_color(color: Color, sq: usize) -> usize {
    palace_bucket(orient_square(color, sq))
}

fn orient_square(side: Color, sq: usize) -> usize {
    match side {
        Color::Red => sq,
        Color::Black => BOARD_SIZE - 1 - sq,
    }
}

#[inline(always)]
fn index(file: usize, rank: usize) -> usize {
    rank * BOARD_FILES + file
}

#[inline(always)]
fn file_of(sq: usize) -> usize {
    sq % BOARD_FILES
}

#[inline(always)]
fn rank_of(sq: usize) -> usize {
    sq / BOARD_FILES
}

#[inline(always)]
fn inside_board(file: i32, rank: i32) -> bool {
    (0..BOARD_FILES as i32).contains(&file) && (0..BOARD_RANKS as i32).contains(&rank)
}

fn neighbor_offset_index(df: i32, dr: i32) -> Option<usize> {
    NEIGHBOR_OFFSETS
        .iter()
        .position(|&(offset_df, offset_dr)| offset_df == df && offset_dr == dr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v3_features_include_base_king_aware_and_history_inputs() {
        let position = Position::startpos();
        let history = [HistoryMove {
            piece: position.piece_at(64).unwrap(),
            captured: None,
            mv: Move::new(64, 55),
        }];
        let features = extract_sparse_features_v3(&position, &history);

        assert_eq!(features.len(), 32 * 3 + 2);
        assert!(features.iter().all(|feature| *feature < V3_INPUT_SIZE));
        assert!(!features.contains(&(INPUT_SIZE - 1)));
        assert!(features.iter().any(|feature| *feature >= INPUT_SIZE));
        assert!(features.iter().any(|feature| *feature >= V2_INPUT_SIZE));

        let black_to_move =
            Position::from_fen("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b")
                .unwrap();
        let black_features = extract_sparse_features_v3(&black_to_move, &history);
        assert!(black_features.contains(&(INPUT_SIZE - 1)));
    }

    #[test]
    fn v4_features_include_row_file_and_neighbor_context() {
        let position = Position::from_fen("4k4/9/9/9/4p4/4R4/9/9/9/4K4 w").unwrap();
        let features = extract_sparse_features_v4(&position, &[]);

        assert!(features.iter().all(|feature| *feature < V4_INPUT_SIZE));
        assert!(features.iter().any(|feature| *feature >= V3_INPUT_SIZE));
        assert!(
            features
                .iter()
                .any(|feature| *feature >= V3_INPUT_SIZE + ROW_INPUT_SIZE + COL_INPUT_SIZE)
        );
    }

    #[test]
    fn features_are_absolute_with_side_to_move_marker() {
        let red_to_move = Position::from_fen("4k4/9/9/9/4p4/4P4/9/9/9/4K4 w").unwrap();
        let black_to_move = Position::from_fen("4k4/9/9/9/4p4/4P4/9/9/9/4K4 b").unwrap();

        let red_v3 = extract_sparse_features_v3(&red_to_move, &[]);
        let black_v3 = extract_sparse_features_v3(&black_to_move, &[]);
        assert_ne!(red_v3, black_v3);
        assert!(!red_v3.contains(&(INPUT_SIZE - 1)));
        assert!(black_v3.contains(&(INPUT_SIZE - 1)));

        let red_v4 = extract_sparse_features_v4(&red_to_move, &[]);
        let black_v4 = extract_sparse_features_v4(&black_to_move, &[]);
        assert_ne!(red_v4, black_v4);
        assert!(!red_v4.contains(&(INPUT_SIZE - 1)));
        assert!(black_v4.contains(&(INPUT_SIZE - 1)));
    }

    #[test]
    fn history_features_are_absolute_with_side_to_move_marker() {
        let red_to_move = Position::from_fen("4k4/9/9/9/4p4/4P4/9/9/9/4K4 w").unwrap();
        let black_to_move = Position::from_fen("4k4/9/9/9/4p4/4P4/9/9/9/4K4 b").unwrap();
        let history = [HistoryMove {
            piece: red_to_move.piece_at(49).unwrap(),
            captured: None,
            mv: Move::new(49, 40),
        }];

        let red_features = extract_sparse_features_v3(&red_to_move, &history);
        let mut black_features = extract_sparse_features_v3(&black_to_move, &history);
        assert_ne!(red_features, black_features);
        black_features.retain(|feature| *feature != INPUT_SIZE - 1);
        assert_eq!(red_features, black_features);
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
            captured: None,
            mv: Move::new(50, 41),
        }];
        let right_history = [HistoryMove {
            piece: right.piece_at(48).unwrap(),
            captured: None,
            mv: Move::new(48, 39),
        }];
        let mut mirrored_left = extract_sparse_features_v3(&left, &left_history);
        mirror_sparse_features_file(&mut mirrored_left);

        assert_eq!(
            mirrored_left,
            extract_sparse_features_v3(&right, &right_history)
        );

        let mut mirrored_left_v4 = extract_sparse_features_v4(&left, &left_history);
        mirror_sparse_features_file(&mut mirrored_left_v4);
        assert_eq!(
            mirrored_left_v4,
            extract_sparse_features_v4(&right, &right_history)
        );
    }
}
