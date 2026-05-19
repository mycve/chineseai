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
const KING_ATTACK_BUCKETS: usize = 16;
const KING_ATTACK_INPUT_SIZE: usize = 2 * KING_ATTACK_BUCKETS;
const TACTICAL_STATES: usize = 6;
const TACTICAL_INPUT_SIZE: usize = TACTICAL_STATES * 14 * BOARD_SIZE;
const STRATEGIC_STATES: usize = 9;
const STRATEGIC_BUCKETS: usize = 16;
const STRATEGIC_INPUT_SIZE: usize = STRATEGIC_STATES * STRATEGIC_BUCKETS;
const THREAT_TARGET_PIECES: usize = 14;
const THREAT_KIND_SLOTS: [usize; 7] = [4, 4, 4, 8, 17, 17, 3];
const THREAT_PIECE_SLOTS: usize = threat_piece_slots();
pub const THREAT_INPUT_SIZE: usize = THREAT_PIECE_SLOTS * THREAT_TARGET_PIECES;
pub const POSITIONAL_NNUE_INPUT_SIZE: usize = V3_INPUT_SIZE
    + ROW_INPUT_SIZE
    + COL_INPUT_SIZE
    + KING_ATTACK_INPUT_SIZE
    + TACTICAL_INPUT_SIZE
    + STRATEGIC_INPUT_SIZE;
pub const THREAT_FEATURE_START: usize = POSITIONAL_NNUE_INPUT_SIZE;
pub const PURE_NNUE_INPUT_SIZE: usize = POSITIONAL_NNUE_INPUT_SIZE + THREAT_INPUT_SIZE;

pub fn layer_stack_bucket(position: &Position, side: Color) -> usize {
    let us_rook = piece_count(position, side, PieceKind::Rook).min(2);
    let opp_rook = piece_count(position, side.opposite(), PieceKind::Rook).min(2);
    let us_knight_cannon = (piece_count(position, side, PieceKind::Horse)
        + piece_count(position, side, PieceKind::Cannon))
    .min(4);
    let opp_knight_cannon = (piece_count(position, side.opposite(), PieceKind::Horse)
        + piece_count(position, side.opposite(), PieceKind::Cannon))
    .min(4);

    if us_rook == opp_rook {
        us_rook * 4
            + usize::from(us_knight_cannon + opp_knight_cannon >= 4) * 2
            + usize::from(us_knight_cannon == opp_knight_cannon)
    } else if us_rook == 2 && opp_rook == 1 {
        12
    } else if us_rook == 1 && opp_rook == 2 {
        13
    } else if us_rook > 0 && opp_rook == 0 {
        14
    } else {
        15
    }
}
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

pub fn extract_sparse_features_pure_canonical(
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
    let mut features = Vec::with_capacity(192);

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
        features.push(row_feature_index(piece_index, rank_of(canonical_sq)));
        features.push(col_feature_index(piece_index, file_of(canonical_sq)));
    }

    let tactical_counts = add_tactical_features(position, side, &mut features);
    add_king_attack_features(position, side, tactical_counts, &mut features);
    add_strategic_features(position, side, tactical_counts, &mut features);
    add_full_threat_features(position, side, &mut features);

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
    if feature < PURE_NNUE_INPUT_SIZE {
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
        if offset < KING_ATTACK_INPUT_SIZE {
            return feature;
        }
        let offset = offset - KING_ATTACK_INPUT_SIZE;
        if offset < TACTICAL_INPUT_SIZE {
            let sq = offset % BOARD_SIZE;
            let partial = offset / BOARD_SIZE;
            let piece_index = partial % 14;
            let state = partial / 14;
            return tactical_feature_index(state, piece_index, mirror_file_square(sq));
        }
        let offset = offset - TACTICAL_INPUT_SIZE;
        if offset < STRATEGIC_INPUT_SIZE {
            return feature;
        }
        let offset = offset - STRATEGIC_INPUT_SIZE;
        if offset < THREAT_INPUT_SIZE {
            return mirror_threat_feature(offset);
        }
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

fn king_attack_feature_index(perspective: usize, bucket: usize) -> usize {
    V3_INPUT_SIZE
        + ROW_INPUT_SIZE
        + COL_INPUT_SIZE
        + perspective * KING_ATTACK_BUCKETS
        + bucket.min(KING_ATTACK_BUCKETS - 1)
}

fn tactical_feature_index(state: usize, piece_index: usize, sq: usize) -> usize {
    V3_INPUT_SIZE
        + ROW_INPUT_SIZE
        + COL_INPUT_SIZE
        + KING_ATTACK_INPUT_SIZE
        + (state * 14 + piece_index) * BOARD_SIZE
        + sq
}

fn strategic_feature_index(state: usize, bucket: usize) -> usize {
    V3_INPUT_SIZE
        + ROW_INPUT_SIZE
        + COL_INPUT_SIZE
        + KING_ATTACK_INPUT_SIZE
        + TACTICAL_INPUT_SIZE
        + state * STRATEGIC_BUCKETS
        + bucket.min(STRATEGIC_BUCKETS - 1)
}

fn threat_feature_index(
    attacker_piece: usize,
    from: usize,
    slot: usize,
    target_piece: usize,
) -> usize {
    V3_INPUT_SIZE
        + ROW_INPUT_SIZE
        + COL_INPUT_SIZE
        + KING_ATTACK_INPUT_SIZE
        + TACTICAL_INPUT_SIZE
        + STRATEGIC_INPUT_SIZE
        + (((threat_piece_offset(attacker_piece)
            + from * threat_slots_for_piece(attacker_piece)
            + slot)
            * THREAT_TARGET_PIECES)
            + target_piece)
}

const fn threat_piece_slots() -> usize {
    let mut total = 0;
    let mut piece = 0;
    while piece < 14 {
        total += BOARD_SIZE * THREAT_KIND_SLOTS[piece % 7];
        piece += 1;
    }
    total
}

fn threat_piece_offset(piece_index: usize) -> usize {
    let mut offset = 0;
    for piece in 0..piece_index {
        offset += BOARD_SIZE * threat_slots_for_piece(piece);
    }
    offset
}

fn threat_slots_for_piece(piece_index: usize) -> usize {
    THREAT_KIND_SLOTS[piece_index % 7]
}

fn mirror_threat_feature(offset: usize) -> usize {
    let target_piece = offset % THREAT_TARGET_PIECES;
    let mut geometry = offset / THREAT_TARGET_PIECES;
    for attacker_piece in 0..14 {
        let slots = threat_slots_for_piece(attacker_piece);
        let piece_size = BOARD_SIZE * slots;
        if geometry < piece_size {
            let from = geometry / slots;
            let slot = geometry % slots;
            if let Some(to) = threat_slot_to_square(attacker_piece, from, slot) {
                return threat_feature_index(
                    attacker_piece,
                    mirror_file_square(from),
                    threat_slot_for(
                        attacker_piece,
                        mirror_file_square(from),
                        mirror_file_square(to),
                    )
                    .unwrap_or(slot),
                    target_piece,
                );
            }
            return threat_feature_index(
                attacker_piece,
                mirror_file_square(from),
                slot,
                target_piece,
            );
        }
        geometry -= piece_size;
    }
    THREAT_FEATURE_START + offset
}

#[derive(Clone, Copy, Debug, Default)]
struct TacticalCounts {
    own_attacked: usize,
    enemy_attacked: usize,
    own_protected: usize,
    enemy_protected: usize,
    own_hanging: usize,
    enemy_hanging: usize,
}

fn add_tactical_features(
    position: &Position,
    side: Color,
    features: &mut Vec<usize>,
) -> TacticalCounts {
    let mut counts = TacticalCounts::default();
    for sq in 0..BOARD_SIZE {
        let Some(piece) = position.piece_at(sq) else {
            continue;
        };
        let own_piece = piece.color == side;
        let canonical_sq = orient_square(side, sq);
        let piece_index = canonical_piece_index(side, piece);
        let attacked_by_own = position.is_piece_protected(sq, side);
        let attacked_by_enemy = position.is_piece_protected(sq, side.opposite());
        let (attacked, protected, hanging, base_state) = if own_piece {
            (
                attacked_by_enemy,
                attacked_by_own,
                attacked_by_enemy && !attacked_by_own,
                0,
            )
        } else {
            (
                attacked_by_own,
                attacked_by_enemy,
                attacked_by_own && !attacked_by_enemy,
                3,
            )
        };
        if attacked {
            features.push(tactical_feature_index(
                base_state,
                piece_index,
                canonical_sq,
            ));
        }
        if protected {
            features.push(tactical_feature_index(
                base_state + 1,
                piece_index,
                canonical_sq,
            ));
        }
        if hanging {
            features.push(tactical_feature_index(
                base_state + 2,
                piece_index,
                canonical_sq,
            ));
        }
        if own_piece {
            counts.own_attacked += attacked as usize;
            counts.own_protected += protected as usize;
            counts.own_hanging += hanging as usize;
        } else {
            counts.enemy_attacked += attacked as usize;
            counts.enemy_protected += protected as usize;
            counts.enemy_hanging += hanging as usize;
        }
    }
    counts
}

fn add_strategic_features(
    position: &Position,
    side: Color,
    tactical: TacticalCounts,
    features: &mut Vec<usize>,
) {
    let mut own_material = 0;
    let mut enemy_material = 0;
    for sq in 0..BOARD_SIZE {
        let Some(piece) = position.piece_at(sq) else {
            continue;
        };
        let value = piece_base_value(piece.kind);
        if piece.color == side {
            own_material += value;
        } else {
            enemy_material += value;
        }
    }
    let total_material = own_material + enemy_material;
    let material_balance = own_material - enemy_material;

    features.push(strategic_feature_index(
        0,
        bucket_div(position.halfmove_clock() as usize, 8),
    ));
    features.push(strategic_feature_index(
        1,
        bucket_scaled(total_material.max(0) as usize, 6220),
    ));
    features.push(strategic_feature_index(
        2,
        bucket_signed(material_balance, 1200),
    ));
    features.push(strategic_feature_index(3, tactical.own_attacked));
    features.push(strategic_feature_index(4, tactical.enemy_attacked));
    features.push(strategic_feature_index(5, tactical.own_hanging));
    features.push(strategic_feature_index(6, tactical.enemy_hanging));
    features.push(strategic_feature_index(7, tactical.own_protected));
    features.push(strategic_feature_index(8, tactical.enemy_protected));
}

fn add_king_attack_features(
    position: &Position,
    side: Color,
    tactical: TacticalCounts,
    features: &mut Vec<usize>,
) {
    features.push(king_attack_feature_index(
        0,
        king_attack_bucket(position, side, side, tactical.own_attacked),
    ));
    features.push(king_attack_feature_index(
        1,
        king_attack_bucket(position, side, side.opposite(), tactical.enemy_attacked),
    ));
}

fn king_attack_bucket(
    position: &Position,
    perspective: Color,
    king_color: Color,
    attacked_count: usize,
) -> usize {
    let Some(general) = position.general_square(king_color) else {
        return 0;
    };
    let mut attacked_palace = 0usize;
    let enemy = king_color.opposite();
    for rank in 0..BOARD_RANKS {
        for file in 3..=5 {
            if !inside_palace(king_color, file, rank) {
                continue;
            }
            let sq = square(file, rank);
            if square_attacked_by(position, sq, enemy) {
                attacked_palace += 1;
            }
        }
    }
    let canonical_general = orient_square(perspective, general);
    let center_distance =
        file_of(canonical_general).abs_diff(4) + rank_of(canonical_general).abs_diff(8);
    (attacked_palace * 3 + attacked_count.min(3) + center_distance.min(3))
        .min(KING_ATTACK_BUCKETS - 1)
}

fn piece_count(position: &Position, color: Color, kind: PieceKind) -> usize {
    let mut count = 0;
    for sq in 0..BOARD_SIZE {
        let Some(piece) = position.piece_at(sq) else {
            continue;
        };
        if piece.color == color && piece.kind == kind {
            count += 1;
        }
    }
    count
}

fn add_full_threat_features(position: &Position, side: Color, features: &mut Vec<usize>) {
    for from in 0..BOARD_SIZE {
        let Some(attacker) = position.piece_at(from) else {
            continue;
        };
        let attacker_piece = canonical_piece_index(side, attacker);
        let canonical_from = orient_square(side, from);
        visit_piece_attacks(position, from, attacker, |to| {
            let Some(target) = position.piece_at(to) else {
                return;
            };
            if target.color == attacker.color {
                return;
            }
            let canonical_to = orient_square(side, to);
            let Some(slot) = threat_slot_for(attacker_piece, canonical_from, canonical_to) else {
                return;
            };
            features.push(threat_feature_index(
                attacker_piece,
                canonical_from,
                slot,
                canonical_piece_index(side, target),
            ));
        });
    }
}

fn square_attacked_by(position: &Position, target: usize, by: Color) -> bool {
    for from in 0..BOARD_SIZE {
        let Some(piece) = position.piece_at(from) else {
            continue;
        };
        if piece.color != by {
            continue;
        }
        let mut attacked = false;
        visit_piece_attacks(position, from, piece, |to| {
            if to == target {
                attacked = true;
            }
        });
        if attacked {
            return true;
        }
    }
    false
}

fn visit_piece_attacks(
    position: &Position,
    from: usize,
    piece: Piece,
    mut visit: impl FnMut(usize),
) {
    match piece.kind {
        PieceKind::General => visit_general_attacks(position, from, piece.color, visit),
        PieceKind::Advisor => {
            for (df, dr) in [(-1, -1), (1, -1), (-1, 1), (1, 1)] {
                let nf = file_of(from) as i32 + df;
                let nr = rank_of(from) as i32 + dr;
                if inside_board(nf, nr) && inside_palace(piece.color, nf as usize, nr as usize) {
                    visit(square(nf as usize, nr as usize));
                }
            }
        }
        PieceKind::Elephant => {
            for (eye_df, eye_dr, df, dr) in [
                (-1, -1, -2, -2),
                (1, -1, 2, -2),
                (-1, 1, -2, 2),
                (1, 1, 2, 2),
            ] {
                let file = file_of(from) as i32;
                let rank = rank_of(from) as i32;
                let eye_f = file + eye_df;
                let eye_r = rank + eye_dr;
                let nf = file + df;
                let nr = rank + dr;
                if inside_board(eye_f, eye_r)
                    && inside_board(nf, nr)
                    && elephant_stays_home(piece.color, nr as usize)
                    && position
                        .piece_at(square(eye_f as usize, eye_r as usize))
                        .is_none()
                {
                    visit(square(nf as usize, nr as usize));
                }
            }
        }
        PieceKind::Horse => {
            for (leg_df, leg_dr, df, dr) in [
                (0, -1, -1, -2),
                (0, -1, 1, -2),
                (0, 1, -1, 2),
                (0, 1, 1, 2),
                (-1, 0, -2, -1),
                (-1, 0, -2, 1),
                (1, 0, 2, -1),
                (1, 0, 2, 1),
            ] {
                let file = file_of(from) as i32;
                let rank = rank_of(from) as i32;
                let leg_f = file + leg_df;
                let leg_r = rank + leg_dr;
                let nf = file + df;
                let nr = rank + dr;
                if inside_board(leg_f, leg_r)
                    && inside_board(nf, nr)
                    && position
                        .piece_at(square(leg_f as usize, leg_r as usize))
                        .is_none()
                {
                    visit(square(nf as usize, nr as usize));
                }
            }
        }
        PieceKind::Rook => visit_slider_attacks(position, from, false, visit),
        PieceKind::Cannon => visit_slider_attacks(position, from, true, visit),
        PieceKind::Soldier => {
            let file = file_of(from) as i32;
            let rank = rank_of(from) as i32;
            let forward = rank + piece.color.forward_step();
            if inside_board(file, forward) {
                visit(square(file as usize, forward as usize));
            }
            if soldier_crossed_river(piece.color, rank as usize) {
                for df in [-1, 1] {
                    let nf = file + df;
                    if inside_board(nf, rank) {
                        visit(square(nf as usize, rank as usize));
                    }
                }
            }
        }
    }
}

fn visit_general_attacks(
    position: &Position,
    from: usize,
    color: Color,
    mut visit: impl FnMut(usize),
) {
    for (df, dr) in [(0, -1), (0, 1), (-1, 0), (1, 0)] {
        let nf = file_of(from) as i32 + df;
        let nr = rank_of(from) as i32 + dr;
        if inside_board(nf, nr) && inside_palace(color, nf as usize, nr as usize) {
            visit(square(nf as usize, nr as usize));
        }
    }
    if let Some(enemy_general) = position.general_square(color.opposite())
        && file_of(enemy_general) == file_of(from)
    {
        let file = file_of(from);
        let start = rank_of(from).min(rank_of(enemy_general)) + 1;
        let end = rank_of(from).max(rank_of(enemy_general));
        if (start..end).all(|rank| position.piece_at(square(file, rank)).is_none()) {
            visit(enemy_general);
        }
    }
}

fn visit_slider_attacks(
    position: &Position,
    from: usize,
    is_cannon: bool,
    mut visit: impl FnMut(usize),
) {
    for (df, dr) in [(0, -1), (0, 1), (-1, 0), (1, 0)] {
        let mut nf = file_of(from) as i32 + df;
        let mut nr = rank_of(from) as i32 + dr;
        let mut seen_screen = false;
        while inside_board(nf, nr) {
            let to = square(nf as usize, nr as usize);
            if position.piece_at(to).is_some() {
                if !is_cannon || seen_screen {
                    visit(to);
                    break;
                }
                seen_screen = true;
            }
            nf += df;
            nr += dr;
        }
    }
}

fn threat_slot_for(piece_index: usize, from: usize, to: usize) -> Option<usize> {
    let kind = piece_index % 7;
    let from_file = file_of(from);
    let from_rank = rank_of(from);
    let to_file = file_of(to);
    let to_rank = rank_of(to);
    match kind {
        0 => match (
            to_file as i32 - from_file as i32,
            to_rank as i32 - from_rank as i32,
        ) {
            (0, -1) => Some(0),
            (0, 1) => Some(1),
            (-1, 0) => Some(2),
            (1, 0) => Some(3),
            _ => None,
        },
        1 => match (
            to_file as i32 - from_file as i32,
            to_rank as i32 - from_rank as i32,
        ) {
            (-1, -1) => Some(0),
            (1, -1) => Some(1),
            (-1, 1) => Some(2),
            (1, 1) => Some(3),
            _ => None,
        },
        2 => match (
            to_file as i32 - from_file as i32,
            to_rank as i32 - from_rank as i32,
        ) {
            (-2, -2) => Some(0),
            (2, -2) => Some(1),
            (-2, 2) => Some(2),
            (2, 2) => Some(3),
            _ => None,
        },
        3 => match (
            to_file as i32 - from_file as i32,
            to_rank as i32 - from_rank as i32,
        ) {
            (-1, -2) => Some(0),
            (1, -2) => Some(1),
            (-1, 2) => Some(2),
            (1, 2) => Some(3),
            (-2, -1) => Some(4),
            (-2, 1) => Some(5),
            (2, -1) => Some(6),
            (2, 1) => Some(7),
            _ => None,
        },
        4 | 5 => {
            if from_file == to_file && from_rank != to_rank {
                Some(if to_rank < from_rank {
                    to_rank
                } else {
                    to_rank - 1
                })
            } else if from_rank == to_rank && from_file != to_file {
                Some(
                    9 + if to_file < from_file {
                        to_file
                    } else {
                        to_file - 1
                    },
                )
            } else {
                None
            }
        }
        6 => {
            let forward = if piece_index < 7 { -1 } else { 1 };
            match (
                to_file as i32 - from_file as i32,
                to_rank as i32 - from_rank as i32,
            ) {
                (0, dr) if dr == forward => Some(0),
                (-1, 0) => Some(1),
                (1, 0) => Some(2),
                _ => None,
            }
        }
        _ => None,
    }
}

fn threat_slot_to_square(piece_index: usize, from: usize, slot: usize) -> Option<usize> {
    let file = file_of(from);
    let rank = rank_of(from);
    match piece_index % 7 {
        0 => match slot {
            0 if rank > 0 => Some(square(file, rank - 1)),
            1 if rank + 1 < BOARD_RANKS => Some(square(file, rank + 1)),
            2 if file > 0 => Some(square(file - 1, rank)),
            3 if file + 1 < BOARD_FILES => Some(square(file + 1, rank)),
            _ => None,
        },
        1 => match slot {
            0 if file > 0 && rank > 0 => Some(square(file - 1, rank - 1)),
            1 if file + 1 < BOARD_FILES && rank > 0 => Some(square(file + 1, rank - 1)),
            2 if file > 0 && rank + 1 < BOARD_RANKS => Some(square(file - 1, rank + 1)),
            3 if file + 1 < BOARD_FILES && rank + 1 < BOARD_RANKS => {
                Some(square(file + 1, rank + 1))
            }
            _ => None,
        },
        2 => match slot {
            0 if file >= 2 && rank >= 2 => Some(square(file - 2, rank - 2)),
            1 if file + 2 < BOARD_FILES && rank >= 2 => Some(square(file + 2, rank - 2)),
            2 if file >= 2 && rank + 2 < BOARD_RANKS => Some(square(file - 2, rank + 2)),
            3 if file + 2 < BOARD_FILES && rank + 2 < BOARD_RANKS => {
                Some(square(file + 2, rank + 2))
            }
            _ => None,
        },
        3 => match slot {
            0 if file > 0 && rank >= 2 => Some(square(file - 1, rank - 2)),
            1 if file + 1 < BOARD_FILES && rank >= 2 => Some(square(file + 1, rank - 2)),
            2 if file > 0 && rank + 2 < BOARD_RANKS => Some(square(file - 1, rank + 2)),
            3 if file + 1 < BOARD_FILES && rank + 2 < BOARD_RANKS => {
                Some(square(file + 1, rank + 2))
            }
            4 if file >= 2 && rank > 0 => Some(square(file - 2, rank - 1)),
            5 if file >= 2 && rank + 1 < BOARD_RANKS => Some(square(file - 2, rank + 1)),
            6 if file + 2 < BOARD_FILES && rank > 0 => Some(square(file + 2, rank - 1)),
            7 if file + 2 < BOARD_FILES && rank + 1 < BOARD_RANKS => {
                Some(square(file + 2, rank + 1))
            }
            _ => None,
        },
        4 | 5 => {
            if slot < 9 {
                let to_rank = if slot < rank { slot } else { slot + 1 };
                (to_rank < BOARD_RANKS).then_some(square(file, to_rank))
            } else {
                let file_slot = slot - 9;
                let to_file = if file_slot < file {
                    file_slot
                } else {
                    file_slot + 1
                };
                (to_file < BOARD_FILES).then_some(square(to_file, rank))
            }
        }
        6 => {
            let forward = if piece_index < 7 { -1 } else { 1 };
            match slot {
                0 => {
                    let to_rank = rank as i32 + forward;
                    inside_board(file as i32, to_rank).then_some(square(file, to_rank as usize))
                }
                1 if file > 0 => Some(square(file - 1, rank)),
                2 if file + 1 < BOARD_FILES => Some(square(file + 1, rank)),
                _ => None,
            }
        }
        _ => None,
    }
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

fn square(file: usize, rank: usize) -> usize {
    rank * BOARD_FILES + file
}

fn inside_board(file: i32, rank: i32) -> bool {
    (0..BOARD_FILES as i32).contains(&file) && (0..BOARD_RANKS as i32).contains(&rank)
}

fn inside_palace(color: Color, file: usize, rank: usize) -> bool {
    (3..=5).contains(&file)
        && match color {
            Color::Black => rank <= 2,
            Color::Red => rank >= 7,
        }
}

fn elephant_stays_home(color: Color, rank: usize) -> bool {
    match color {
        Color::Black => rank <= 4,
        Color::Red => rank >= 5,
    }
}

fn soldier_crossed_river(color: Color, rank: usize) -> bool {
    match color {
        Color::Black => rank >= 5,
        Color::Red => rank <= 4,
    }
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
    fn pure_features_include_row_file_context() {
        let position = Position::from_fen("4k4/9/9/9/4p4/4R4/9/9/9/4K4 w").unwrap();
        let features = extract_sparse_features_pure_canonical(&position, &[]);

        assert!(
            features
                .iter()
                .all(|feature| *feature < PURE_NNUE_INPUT_SIZE)
        );
        assert!(features.iter().any(|feature| *feature >= V3_INPUT_SIZE));
        assert!(
            features
                .iter()
                .any(|feature| *feature >= V3_INPUT_SIZE + ROW_INPUT_SIZE)
        );
        assert!(features.iter().any(|feature| *feature
            >= V3_INPUT_SIZE + ROW_INPUT_SIZE + COL_INPUT_SIZE + KING_ATTACK_INPUT_SIZE));
        assert!(features.iter().any(|feature| *feature
            >= V3_INPUT_SIZE + ROW_INPUT_SIZE + COL_INPUT_SIZE
            && *feature
                < V3_INPUT_SIZE + ROW_INPUT_SIZE + COL_INPUT_SIZE + KING_ATTACK_INPUT_SIZE));
        assert!(features.iter().any(|feature| *feature
            >= V3_INPUT_SIZE
                + ROW_INPUT_SIZE
                + COL_INPUT_SIZE
                + KING_ATTACK_INPUT_SIZE
                + TACTICAL_INPUT_SIZE
                + STRATEGIC_INPUT_SIZE));
        assert!(features.iter().any(|feature| *feature
            >= V3_INPUT_SIZE
                + ROW_INPUT_SIZE
                + COL_INPUT_SIZE
                + KING_ATTACK_INPUT_SIZE
                + TACTICAL_INPUT_SIZE
                + STRATEGIC_INPUT_SIZE));
    }

    #[test]
    fn full_threat_features_encode_attacker_from_target_to() {
        let position = Position::from_fen("4k4/9/9/9/4p4/4R4/9/9/9/4K4 w").unwrap();
        let features = extract_sparse_features_pure_canonical(&position, &[]);
        let attacker = canonical_piece_index(
            Color::Red,
            Piece {
                color: Color::Red,
                kind: PieceKind::Rook,
            },
        );
        let target = canonical_piece_index(
            Color::Red,
            Piece {
                color: Color::Black,
                kind: PieceKind::Soldier,
            },
        );
        let from = square(4, 5);
        let to = square(4, 4);
        let threat = threat_feature_index(
            attacker,
            from,
            threat_slot_for(attacker, from, to).unwrap(),
            target,
        );
        assert!(features.contains(&threat));
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

        let red_pure = extract_sparse_features_pure_canonical(&red_to_move, &[]);
        let black_pure = extract_sparse_features_pure_canonical(&black_to_move, &[]);
        assert!(
            red_pure
                .iter()
                .all(|feature| *feature < PURE_NNUE_INPUT_SIZE)
        );
        assert!(
            black_pure
                .iter()
                .all(|feature| *feature < PURE_NNUE_INPUT_SIZE)
        );
        assert!(!red_pure.contains(&(INPUT_SIZE - 1)));
        assert!(!black_pure.contains(&(INPUT_SIZE - 1)));
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

        let mut mirrored_left_pure = extract_sparse_features_pure_canonical(&left, &left_history);
        mirror_sparse_features_file(&mut mirrored_left_pure);
        assert_eq!(
            mirrored_left_pure,
            extract_sparse_features_pure_canonical(&right, &right_history)
        );
    }
}
