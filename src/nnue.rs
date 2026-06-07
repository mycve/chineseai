use crate::xiangqi::{
    BOARD_FILES, BOARD_RANKS, BOARD_SIZE, Color, Move, Piece, PieceKind, Position, piece_base_value,
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
const STRATEGIC_STATES: usize = 3;
const STRATEGIC_BUCKETS: usize = 16;
const STRATEGIC_INPUT_SIZE: usize = STRATEGIC_STATES * STRATEGIC_BUCKETS;
const AZ_ROW_INPUT_OFFSET: usize = V3_INPUT_SIZE;
const AZ_COL_INPUT_OFFSET: usize = AZ_ROW_INPUT_OFFSET + ROW_INPUT_SIZE;
const AZ_STRATEGIC_INPUT_OFFSET: usize = AZ_COL_INPUT_OFFSET + COL_INPUT_SIZE;
const AZ_RELATION_INPUT_OFFSET: usize = AZ_STRATEGIC_INPUT_OFFSET + STRATEGIC_INPUT_SIZE;
const RELATION_STATES: usize = 4;
const RELATION_BUCKETS: usize = 16;
const RELATION_INPUT_SIZE: usize = RELATION_STATES * 14 * RELATION_BUCKETS;
pub const AZ_NNUE_INPUT_SIZE: usize = AZ_RELATION_INPUT_OFFSET + RELATION_INPUT_SIZE;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HistoryMove {
    pub piece: Piece,
    pub captured: Option<Piece>,
    pub mv: Move,
}

pub fn extract_sparse_features_az_absolute(
    position: &Position,
    history: &[HistoryMove],
) -> Vec<usize> {
    let mut features = extract_sparse_features_az_absolute_current(position);
    add_az_absolute_history_features(history, &mut features);
    features.sort_unstable();
    features
}

pub fn extract_sparse_features_az_canonical(
    position: &Position,
    history: &[HistoryMove],
) -> Vec<usize> {
    let side = position.side_to_move();
    let mut features = extract_sparse_features_az_canonical_current(position);
    add_az_canonical_history_features(side, history, &mut features);
    features.sort_unstable();
    features
}

pub fn extract_sparse_features_az_absolute_current(position: &Position) -> Vec<usize> {
    let red_king_bucket = position
        .general_square(Color::Red)
        .map(|sq| palace_bucket_for_color(Color::Red, sq))
        .unwrap_or(4);
    let black_king_bucket = position
        .general_square(Color::Black)
        .map(|sq| palace_bucket_for_color(Color::Black, sq))
        .unwrap_or(4);
    let mut features = Vec::with_capacity(220);
    let mut red_material = 0;
    let mut black_material = 0;

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
        features.push(az_row_feature_index(piece_index, rank_of(sq)));
        features.push(az_col_feature_index(piece_index, file_of(sq)));

        let value = piece_base_value(piece.kind);
        if piece.color == Color::Red {
            red_material += value;
        } else {
            black_material += value;
        }
    }

    if position.side_to_move() == Color::Black {
        features.push(INPUT_SIZE - 1);
    }
    add_az_absolute_strategic_features(position, red_material, black_material, &mut features);
    features.sort_unstable();
    features
}

pub fn extract_sparse_features_az_canonical_current(position: &Position) -> Vec<usize> {
    let side = position.side_to_move();
    let mut features = Vec::with_capacity(96);

    for sq in 0..BOARD_SIZE {
        let Some(piece) = position.piece_at(sq) else {
            continue;
        };
        let rel_color = if piece.color == side {
            Color::Red
        } else {
            Color::Black
        };
        let rel_sq = orient_square(side, sq);
        let piece_index = absolute_piece_index(rel_color, piece.kind);
        features.push(piece_index * BOARD_SIZE + rel_sq);
    }

    add_az_canonical_relation_features(position, side, &mut features);
    features.sort_unstable();
    features
}

pub fn add_az_absolute_history_features(history: &[HistoryMove], out: &mut Vec<usize>) {
    out.reserve(history.len().min(HISTORY_PLIES) * HISTORY_EVENT_TYPES);
    for (age, entry) in history.iter().rev().take(HISTORY_PLIES).enumerate() {
        let piece_index = absolute_piece_index(entry.piece.color, entry.piece.kind);
        out.push(history_feature_index(
            age,
            0,
            piece_index,
            entry.mv.from as usize,
        ));
        out.push(history_feature_index(
            age,
            1,
            piece_index,
            entry.mv.to as usize,
        ));
    }
}

pub fn add_az_canonical_history_features(
    side: Color,
    history: &[HistoryMove],
    out: &mut Vec<usize>,
) {
    out.reserve(history.len().min(HISTORY_PLIES) * HISTORY_EVENT_TYPES);
    for (age, entry) in history.iter().rev().take(HISTORY_PLIES).enumerate() {
        let rel_color = if entry.piece.color == side {
            Color::Red
        } else {
            Color::Black
        };
        let piece_index = absolute_piece_index(rel_color, entry.piece.kind);
        out.push(history_feature_index(
            age,
            0,
            piece_index,
            orient_square(side, entry.mv.from as usize),
        ));
        out.push(history_feature_index(
            age,
            1,
            piece_index,
            orient_square(side, entry.mv.to as usize),
        ));
    }
}

pub fn az_absolute_general_bucket(position: &Position, color: Color) -> usize {
    position
        .general_square(color)
        .map(|sq| palace_bucket_for_color(color, sq))
        .unwrap_or(4)
}

pub fn az_absolute_side_to_move_feature(position: &Position) -> Option<usize> {
    (position.side_to_move() == Color::Black).then_some(INPUT_SIZE - 1)
}

pub fn az_absolute_piece_non_king_features(sq: usize, piece: Piece, out: &mut Vec<usize>) {
    let piece_index = absolute_piece_index(piece.color, piece.kind);
    out.push(piece_index * BOARD_SIZE + sq);
    out.push(az_row_feature_index(piece_index, rank_of(sq)));
    out.push(az_col_feature_index(piece_index, file_of(sq)));
}

pub fn az_absolute_piece_king_feature(
    perspective: Color,
    king_bucket: usize,
    sq: usize,
    piece: Piece,
) -> usize {
    let perspective_index = if perspective == Color::Red { 0 } else { 1 };
    king_aware_feature_index(
        perspective_index,
        king_bucket,
        absolute_piece_index(piece.color, piece.kind),
        sq,
    )
}

pub fn az_absolute_strategic_features(position: &Position, out: &mut Vec<usize>) {
    let mut red_material = 0;
    let mut black_material = 0;
    for sq in 0..BOARD_SIZE {
        let Some(piece) = position.piece_at(sq) else {
            continue;
        };
        let value = piece_base_value(piece.kind);
        if piece.color == Color::Red {
            red_material += value;
        } else {
            black_material += value;
        }
    }
    add_az_absolute_strategic_features(position, red_material, black_material, out);
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

pub fn mirror_sparse_features_az_absolute_file(features: &mut [usize]) {
    for feature in features.iter_mut() {
        *feature = mirror_sparse_feature_az_absolute_file(*feature);
    }
    features.sort_unstable();
}

fn mirror_sparse_feature_az_absolute_file(feature: usize) -> usize {
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
    if feature < AZ_NNUE_INPUT_SIZE {
        let offset = feature - AZ_ROW_INPUT_OFFSET;
        if offset < ROW_INPUT_SIZE {
            return feature;
        }
        let offset = offset - ROW_INPUT_SIZE;
        if offset < COL_INPUT_SIZE {
            let piece_index = offset / BOARD_FILES;
            let file = offset % BOARD_FILES;
            return az_col_feature_index(piece_index, BOARD_FILES - 1 - file);
        }
        return feature;
    }
    if feature < AZ_NNUE_INPUT_SIZE {
        let offset = feature - AZ_RELATION_INPUT_OFFSET;
        let bucket = offset % RELATION_BUCKETS;
        let partial = offset / RELATION_BUCKETS;
        let piece_index = partial % 14;
        let state = partial / 14;
        return az_relation_feature_index(state, piece_index, mirror_relation_bucket_file(bucket));
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

fn az_row_feature_index(piece_index: usize, rank: usize) -> usize {
    AZ_ROW_INPUT_OFFSET + piece_index * BOARD_RANKS + rank
}

fn az_col_feature_index(piece_index: usize, file: usize) -> usize {
    AZ_COL_INPUT_OFFSET + piece_index * BOARD_FILES + file
}

fn az_strategic_feature_index(state: usize, bucket: usize) -> usize {
    AZ_STRATEGIC_INPUT_OFFSET + state * STRATEGIC_BUCKETS + bucket.min(STRATEGIC_BUCKETS - 1)
}

fn az_relation_feature_index(state: usize, piece_index: usize, bucket: usize) -> usize {
    AZ_RELATION_INPUT_OFFSET
        + ((state * 14 + piece_index) * RELATION_BUCKETS + bucket.min(RELATION_BUCKETS - 1))
}

fn relation_square_bucket(sq: usize) -> usize {
    let file_bucket = (file_of(sq) * 4) / BOARD_FILES;
    let rank_bucket = (rank_of(sq) * 4) / BOARD_RANKS;
    rank_bucket * 4 + file_bucket
}

fn mirror_relation_bucket_file(bucket: usize) -> usize {
    let rank_bucket = bucket / 4;
    let file_bucket = bucket % 4;
    rank_bucket * 4 + (3 - file_bucket)
}

#[allow(dead_code)]
fn add_az_absolute_relation_features(position: &Position, out: &mut Vec<usize>) {
    add_az_relation_features(
        position,
        |piece, sq| {
            (
                absolute_piece_index(piece.color, piece.kind),
                relation_square_bucket(sq),
            )
        },
        out,
    );
}

fn add_az_canonical_relation_features(position: &Position, side: Color, out: &mut Vec<usize>) {
    add_az_relation_features(
        position,
        |piece, sq| {
            let rel_color = if piece.color == side {
                Color::Red
            } else {
                Color::Black
            };
            (
                absolute_piece_index(rel_color, piece.kind),
                relation_square_bucket(orient_square(side, sq)),
            )
        },
        out,
    );
}

fn add_az_relation_features<F>(position: &Position, mut encode: F, out: &mut Vec<usize>)
where
    F: FnMut(Piece, usize) -> (usize, usize),
{
    for sq in 0..BOARD_SIZE {
        let Some(piece) = position.piece_at(sq) else {
            continue;
        };
        let (piece_index, bucket) = encode(piece, sq);
        if position.is_square_attacked(sq, piece.color.opposite()) {
            out.push(az_relation_feature_index(0, piece_index, bucket));
        }
        if position.is_piece_protected(sq, piece.color) {
            out.push(az_relation_feature_index(1, piece_index, bucket));
        }
    }
    add_king_line_screen_features(position, &mut encode, out);
    add_cannon_screen_features(position, &mut encode, out);
}

fn add_king_line_screen_features<F>(position: &Position, encode: &mut F, out: &mut Vec<usize>)
where
    F: FnMut(Piece, usize) -> (usize, usize),
{
    for color in [Color::Red, Color::Black] {
        let Some(king_sq) = position.general_square(color) else {
            continue;
        };
        let king_file = file_of(king_sq) as i32;
        let king_rank = rank_of(king_sq) as i32;
        for (df, dr) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let mut screen: Option<(usize, Piece)> = None;
            let mut file = king_file + df;
            let mut rank = king_rank + dr;
            while inside_local(file, rank) {
                let sq = rank as usize * BOARD_FILES + file as usize;
                if let Some(piece) = position.piece_at(sq) {
                    if let Some((screen_sq, screen_piece)) = screen {
                        if screen_piece.color == color
                            && piece.color == color.opposite()
                            && (piece.kind == PieceKind::Rook
                                || piece.kind == PieceKind::Cannon
                                || (piece.kind == PieceKind::General && df == 0))
                        {
                            let (piece_index, bucket) = encode(screen_piece, screen_sq);
                            out.push(az_relation_feature_index(2, piece_index, bucket));
                        }
                        break;
                    }
                    screen = Some((sq, piece));
                }
                file += df;
                rank += dr;
            }
        }
    }
}

fn add_cannon_screen_features<F>(position: &Position, encode: &mut F, out: &mut Vec<usize>)
where
    F: FnMut(Piece, usize) -> (usize, usize),
{
    for sq in 0..BOARD_SIZE {
        let Some(cannon) = position.piece_at(sq) else {
            continue;
        };
        if cannon.kind != PieceKind::Cannon {
            continue;
        }
        let cannon_file = file_of(sq) as i32;
        let cannon_rank = rank_of(sq) as i32;
        for (df, dr) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let mut screen: Option<(usize, Piece)> = None;
            let mut file = cannon_file + df;
            let mut rank = cannon_rank + dr;
            while inside_local(file, rank) {
                let target_sq = rank as usize * BOARD_FILES + file as usize;
                if let Some(piece) = position.piece_at(target_sq) {
                    if let Some((screen_sq, screen_piece)) = screen {
                        if piece.color != cannon.color {
                            let (piece_index, bucket) = encode(screen_piece, screen_sq);
                            out.push(az_relation_feature_index(3, piece_index, bucket));
                        }
                        break;
                    }
                    screen = Some((target_sq, piece));
                }
                file += df;
                rank += dr;
            }
        }
    }
}

fn inside_local(file: i32, rank: i32) -> bool {
    (0..BOARD_FILES as i32).contains(&file) && (0..BOARD_RANKS as i32).contains(&rank)
}

fn add_az_absolute_strategic_features(
    position: &Position,
    red_material: i32,
    black_material: i32,
    features: &mut Vec<usize>,
) {
    let total_material = red_material + black_material;
    let material_balance = red_material - black_material;
    features.push(az_strategic_feature_index(
        0,
        bucket_div(position.halfmove_clock() as usize, 8),
    ));
    features.push(az_strategic_feature_index(
        1,
        bucket_scaled(total_material.max(0) as usize, 6220),
    ));
    features.push(az_strategic_feature_index(
        2,
        bucket_signed(material_balance, 1200),
    ));
}

fn bucket_div(value: usize, div: usize) -> usize {
    (value / div.max(1)).min(STRATEGIC_BUCKETS - 1)
}

fn bucket_scaled(value: usize, max_value: usize) -> usize {
    ((value.min(max_value) * (STRATEGIC_BUCKETS - 1)) / max_value.max(1)).min(STRATEGIC_BUCKETS - 1)
}

fn bucket_signed(value: i32, max_abs: i32) -> usize {
    let clamped = value.clamp(-max_abs, max_abs);
    let shifted = (clamped + max_abs) as usize;
    let scale = (max_abs as usize).saturating_mul(2).max(1);
    ((shifted * (STRATEGIC_BUCKETS - 1)) / scale).min(STRATEGIC_BUCKETS - 1)
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
fn file_of(sq: usize) -> usize {
    sq % BOARD_FILES
}

#[inline(always)]
fn rank_of(sq: usize) -> usize {
    sq / BOARD_FILES
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn az_features_include_history_inputs() {
        let position = Position::startpos();
        let history = [HistoryMove {
            piece: position.piece_at(64).unwrap(),
            captured: None,
            mv: Move::new(64, 55),
        }];
        let features = extract_sparse_features_az_absolute(&position, &history);
        let piece_index = absolute_piece_index(history[0].piece.color, history[0].piece.kind);
        let history_from = history_feature_index(0, 0, piece_index, history[0].mv.from as usize);
        let history_to = history_feature_index(0, 1, piece_index, history[0].mv.to as usize);

        assert!(features.iter().all(|feature| *feature < AZ_NNUE_INPUT_SIZE));
        assert!(features.contains(&history_from));
        assert!(features.contains(&history_to));
        assert!(features.iter().any(|feature| *feature < INPUT_SIZE - 1));
    }

    #[test]
    fn mirror_file_move_flips_left_and_right() {
        assert_eq!(mirror_file_move(Move::new(47, 38)), Move::new(51, 42));
    }

    #[test]
    fn canonical_features_make_side_to_move_red_relative() {
        let position = Position::from_fen("4k4/9/9/9/4p4/9/9/9/9/4K4 b - - 0 1").unwrap();
        let features = extract_sparse_features_az_canonical(&position, &[]);
        let black_general_as_us = absolute_piece_index(Color::Red, PieceKind::General) * BOARD_SIZE
            + canonical_square(Color::Black, 4);
        let red_general_as_them = absolute_piece_index(Color::Black, PieceKind::General)
            * BOARD_SIZE
            + canonical_square(Color::Black, 85);

        assert!(features.contains(&black_general_as_us));
        assert!(features.contains(&red_general_as_them));
        assert!(!features.contains(&(INPUT_SIZE - 1)));
    }

    #[test]
    fn canonical_move_flips_black_to_move_board() {
        assert_eq!(
            canonical_move(Color::Red, Move::new(76, 67)),
            Move::new(76, 67)
        );
        assert_eq!(
            canonical_move(Color::Black, Move::new(4, 13)),
            Move::new(85, 76)
        );
    }

    #[test]
    fn mirrored_az_features_match_mirrored_position_and_history() {
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
        let mut mirrored_left = extract_sparse_features_az_absolute(&left, &left_history);
        mirror_sparse_features_az_absolute_file(&mut mirrored_left);

        assert_eq!(
            mirrored_left,
            extract_sparse_features_az_absolute(&right, &right_history)
        );
    }
}
