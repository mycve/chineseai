// Exact feature-index port from official Pikafish commit
// de9348bc6ea83113ab5901ee6ee182d17d2f2efb, chiefly:
// `half_ka_v2_hm.h`, `full_threats.h`, and `evaluate.cpp`.
// The golden test vectors below were generated once with a temporary MSVC C++ exporter linked
// against that commit; the exporter and build products are intentionally not kept in this tree.

#[cfg(test)]
use std::collections::BTreeSet;
use std::sync::OnceLock;

use smallvec::SmallVec;

use crate::xiangqi::{BOARD_FILES, BOARD_RANKS, BOARD_SIZE, Color, Piece, PieceKind, Position};

use super::{STRUCTURAL_PIECE_SIZE, piece_kind_index};

pub(super) const PIKAFISH_TRANSFORMER_DIMENSIONS: usize = 1024;
pub(super) const PIKAFISH_TRANSFORMER_HALF: usize = PIKAFISH_TRANSFORMER_DIMENSIONS / 2;
pub(super) const PIKAFISH_TRANSFORMED_DIMENSIONS: usize = PIKAFISH_TRANSFORMER_DIMENSIONS;
pub(super) const PIKAFISH_PSQT_BUCKETS: usize = 16;
pub(super) const PIKAFISH_LAYER_STACKS: usize = 16;
pub(super) const PIKAFISH_PS_NB: usize = 689;
pub(super) const PIKAFISH_ATTACK_BUCKETS: usize = 4;
pub(super) const PIKAFISH_KING_BUCKETS: usize = 6;
pub(super) const PIKAFISH_PSQ_DIMENSIONS: usize =
    PIKAFISH_KING_BUCKETS * PIKAFISH_ATTACK_BUCKETS * PIKAFISH_PS_NB;
pub(super) const PIKAFISH_VALUE_THREAT_DIMENSIONS: usize = 45_547;

const BALANCE_ENCODING: u64 = 0xa4a9_2a74_e989_d3a7;
const OFFICIAL_PIECE_ORDER: [usize; STRUCTURAL_PIECE_SIZE] =
    [4, 1, 5, 6, 3, 2, 0, 11, 8, 12, 13, 10, 9, 7];
// Pikafish Piece enum values for our perspective-relative structural piece ids
// [K,A,B,N,R,C,P,k,a,b,n,r,c,p].
const CPP_PIECE: [usize; STRUCTURAL_PIECE_SIZE] = [7, 2, 6, 5, 1, 3, 4, 15, 10, 14, 13, 9, 11, 12];

const VALID_PAIRS: [[bool; 16]; 16] = [
    [false; 16],
    [
        false, true, true, true, true, true, true, true, false, true, true, true, true, true, true,
        false,
    ],
    [
        false, true, true, true, false, true, false, false, false, true, false, true, true, true,
        false, false,
    ],
    [
        false, true, true, true, true, true, true, true, false, true, true, true, true, true, true,
        false,
    ],
    [
        false, false, false, true, true, true, true, false, false, false, true, true, true, true,
        true, false,
    ],
    [
        false, true, true, true, true, true, true, true, false, true, true, true, true, true, true,
        false,
    ],
    [
        false, true, false, true, true, true, true, true, false, true, false, true, true, true,
        false, false,
    ],
    [false; 16],
    [false; 16],
    [
        false, true, true, true, true, true, true, false, false, true, true, true, true, true,
        true, true,
    ],
    [
        false, true, false, true, true, true, false, false, false, true, true, true, false, true,
        false, true,
    ],
    [
        false, true, true, true, true, true, true, false, false, true, true, true, true, true,
        true, true,
    ],
    [
        false, false, true, true, true, true, true, false, false, false, false, true, true, true,
        true, false,
    ],
    [
        false, true, true, true, true, true, true, false, false, true, true, true, true, true,
        true, true,
    ],
    [
        false, true, false, true, true, true, false, false, false, true, false, true, true, true,
        true, true,
    ],
    [
        false, false, false, true, true, true, false, false, false, false, true, true, false, true,
        true, false,
    ],
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ActiveValueFeatures {
    pub psq: Vec<usize>,
    pub threats: Vec<usize>,
    pub feature_bucket: usize,
    pub mirror: bool,
    pub mid_encoding: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ValueFeatureState {
    pub feature_bucket: usize,
    pub mirror: bool,
    pub mid_encoding: u64,
}

#[cfg(test)]
pub(super) fn active_value_features(
    position: &Position,
    perspective: Color,
) -> ActiveValueFeatures {
    active_value_features_pair(position)[match perspective {
        Color::Red => 0,
        Color::Black => 1,
    }]
    .clone()
}

pub(super) fn active_value_features_pair(position: &Position) -> [ActiveValueFeatures; 2] {
    let perspectives = [Color::Red, Color::Black];
    let mids = perspectives.map(|perspective| mid_encoding(position, perspective));
    let buckets: [(usize, bool); 2] = std::array::from_fn(|view| {
        feature_bucket_from_mid(position, perspectives[view], mids[view], mids[1 - view])
    });
    let mut active = std::array::from_fn(|view| ActiveValueFeatures {
        psq: Vec::with_capacity(32),
        threats: Vec::with_capacity(64),
        feature_bucket: buckets[view].0,
        mirror: buckets[view].1,
        mid_encoding: mids[view],
    });
    for square in 0..BOARD_SIZE {
        if let Some(piece) = position.piece_at(square) {
            for view in 0..2 {
                let index = psq_index(
                    perspectives[view],
                    square,
                    piece,
                    active[view].feature_bucket,
                    active[view].mirror,
                );
                debug_assert!(index < PIKAFISH_PSQ_DIMENSIONS);
                active[view].psq.push(index);
            }
        }
    }
    visit_official_occupied_relations(position, |source, attacker, target, attacked| {
        for view in 0..2 {
            let index = threat_index(
                perspectives[view],
                source,
                attacker,
                target,
                attacked,
                active[view].mirror,
            );
            if index < PIKAFISH_VALUE_THREAT_DIMENSIONS {
                active[view].threats.push(index);
            }
        }
    });
    for features in &mut active {
        features.psq.sort_unstable();
        features.threats.sort_unstable();
        features.threats.dedup();
    }
    active
}

pub(super) fn value_feature_states_pair(position: &Position) -> [ValueFeatureState; 2] {
    let perspectives = [Color::Red, Color::Black];
    let mids = perspectives.map(|perspective| mid_encoding(position, perspective));
    std::array::from_fn(|view| {
        let perspective = perspectives[view];
        let (feature_bucket, mirror) =
            feature_bucket_from_mid(position, perspective, mids[view], mids[1 - view]);
        ValueFeatureState {
            feature_bucket,
            mirror,
            mid_encoding: mids[view],
        }
    })
}

pub(super) fn transitioned_psq_features_pair(
    after: &Position,
    before: &[ValueFeatureState; 2],
    from: usize,
    to: usize,
    moved: Piece,
    captured: Option<Piece>,
) -> [ValueFeatureState; 2] {
    let perspectives = [Color::Red, Color::Black];
    let mut mids = [before[0].mid_encoding, before[1].mid_encoding];
    let moved_view = match moved.color {
        Color::Red => 0,
        Color::Black => 1,
    };
    mids[moved_view] = mids[moved_view]
        .wrapping_sub(piece_mid_encoding(moved, from))
        .wrapping_add(piece_mid_encoding(moved, to));
    if let Some(captured) = captured {
        let captured_view = match captured.color {
            Color::Red => 0,
            Color::Black => 1,
        };
        mids[captured_view] = mids[captured_view].wrapping_sub(piece_mid_encoding(captured, to));
    }

    std::array::from_fn(|view| {
        let perspective = perspectives[view];
        let mut attack_bucket = before[view].feature_bucket % PIKAFISH_ATTACK_BUCKETS;
        if let Some(piece) = captured.filter(|piece| piece.color == perspective) {
            match piece.kind {
                PieceKind::Rook => {
                    attack_bucket = (attack_bucket & 1)
                        | usize::from(piece_count(after, perspective, PieceKind::Rook) > 0) * 2;
                }
                PieceKind::Horse | PieceKind::Cannon => {
                    attack_bucket = (attack_bucket & 2)
                        | usize::from(
                            piece_count(after, perspective, PieceKind::Horse)
                                + piece_count(after, perspective, PieceKind::Cannon)
                                > 0,
                        );
                }
                _ => {}
            }
        }
        let (feature_bucket, mirror) = feature_bucket_with_attack_from_mid(
            after,
            perspective,
            mids[view],
            mids[1 - view],
            attack_bucket,
        );
        ValueFeatureState {
            feature_bucket,
            mirror,
            mid_encoding: mids[view],
        }
    })
}

pub(super) type ThreatIndexList = SmallVec<[usize; 16]>;

/// Exact changed FullThreats indices for a normal move. This mirrors the official
/// `DirtyThreats`/`append_changed_indices_both` semantics. Only attackers whose relation can be
/// affected by occupancy at a changed square are visited; the randomized oracle test below proves
/// equality to full refresh.
#[cfg(test)]
pub(super) fn changed_threat_indices(
    before: &Position,
    after: &Position,
    mirrors: [bool; 2],
) -> [(ThreatIndexList, ThreatIndexList); 2] {
    let mut changed = [usize::MAX; 2];
    let mut changed_len = 0;
    for square in 0..BOARD_SIZE {
        if before.piece_at(square) != after.piece_at(square) {
            changed[changed_len] = square;
            changed_len += 1;
        }
    }
    debug_assert!((1..=2).contains(&changed_len));

    changed_threat_indices_for_squares(before, after, mirrors, &changed[..changed_len])
}

pub(super) fn changed_threat_indices_for_move(
    before: &Position,
    after: &Position,
    mirrors: [bool; 2],
    from: usize,
    to: usize,
) -> [(ThreatIndexList, ThreatIndexList); 2] {
    changed_threat_indices_for_squares(before, after, mirrors, &[from, to])
}

fn changed_threat_indices_for_squares(
    before: &Position,
    after: &Position,
    mirrors: [bool; 2],
    changed: &[usize],
) -> [(ThreatIndexList, ThreatIndexList); 2] {
    let before_indices = affected_threat_indices(before, changed, mirrors);
    let after_indices = affected_threat_indices(after, changed, mirrors);
    std::array::from_fn(|view| sorted_index_diff(&before_indices[view], &after_indices[view]))
}

fn sorted_index_diff(before: &[usize], after: &[usize]) -> (ThreatIndexList, ThreatIndexList) {
    let mut removed = SmallVec::new();
    let mut added = SmallVec::new();
    let (mut left, mut right) = (0, 0);
    while left < before.len() || right < after.len() {
        if right == after.len() || (left < before.len() && before[left] < after[right]) {
            removed.push(before[left]);
            left += 1;
        } else if left == before.len() || after[right] < before[left] {
            added.push(after[right]);
            right += 1;
        } else {
            left += 1;
            right += 1;
        }
    }
    (removed, added)
}

fn affected_threat_indices(
    position: &Position,
    changed: &[usize],
    mirrors: [bool; 2],
) -> [SmallVec<[usize; 64]>; 2] {
    let mut indices: [SmallVec<[usize; 64]>; 2] = std::array::from_fn(|_| SmallVec::new());
    let mut occupied = position.occupied_squares();
    while occupied != 0 {
        let source = occupied.trailing_zeros() as usize;
        occupied &= occupied - 1;
        let piece = position.piece_at(source).unwrap();
        if !changed
            .iter()
            .any(|&square| relation_source_affected(source, piece, square))
        {
            continue;
        }
        visit_official_occupied_relations_from(
            position,
            source,
            |source, attacker, target, attacked| {
                let attacker = structural_piece(attacker, Color::Red);
                let attacked = structural_piece(attacked, Color::Red);
                for (view, perspective) in [Color::Red, Color::Black].into_iter().enumerate() {
                    let index = threat_index_structural(
                        perspective,
                        source,
                        attacker,
                        target,
                        attacked,
                        mirrors[view],
                    );
                    if index < PIKAFISH_VALUE_THREAT_DIMENSIONS {
                        indices[view].push(index);
                    }
                }
            },
        );
    }
    for view in &mut indices {
        view.sort_unstable();
        view.dedup();
    }
    indices
}

fn relation_source_affected(source: usize, piece: Piece, changed: usize) -> bool {
    if source == changed {
        return true;
    }
    let sf = source % BOARD_FILES;
    let sr = source / BOARD_FILES;
    let cf = changed % BOARD_FILES;
    let cr = changed / BOARD_FILES;
    let df = sf.abs_diff(cf);
    let dr = sr.abs_diff(cr);
    match piece.kind {
        PieceKind::Rook | PieceKind::Cannon => sf == cf || sr == cr,
        PieceKind::Horse => {
            (df == 1 && dr == 0)
                || (df == 0 && dr == 1)
                || (df == 1 && dr == 2)
                || (df == 2 && dr == 1)
        }
        PieceKind::Elephant => (df == 1 && dr == 1) || (df == 2 && dr == 2),
        PieceKind::Advisor => df == 1 && dr == 1,
        PieceKind::General => df + dr == 1,
        PieceKind::Soldier => {
            (sf == cf && (sr as isize - cr as isize).unsigned_abs() == 1) || (sr == cr && df == 1)
        }
    }
}

fn feature_bucket_from_mid(
    position: &Position,
    perspective: Color,
    ours_mid: u64,
    theirs_mid: u64,
) -> (usize, bool) {
    let attack_bucket = usize::from(piece_count(position, perspective, PieceKind::Rook) > 0) * 2
        + usize::from(
            piece_count(position, perspective, PieceKind::Horse)
                + piece_count(position, perspective, PieceKind::Cannon)
                > 0,
        );
    feature_bucket_with_attack_from_mid(position, perspective, ours_mid, theirs_mid, attack_bucket)
}

fn feature_bucket_with_attack_from_mid(
    position: &Position,
    perspective: Color,
    ours_mid: u64,
    theirs_mid: u64,
    attack_bucket: usize,
) -> (usize, bool) {
    let king = position
        .general_square(perspective)
        .unwrap_or(match perspective {
            Color::Red => 85,
            Color::Black => 4,
        });
    let opponent_king =
        position
            .general_square(perspective.opposite())
            .unwrap_or(match perspective {
                Color::Red => 4,
                Color::Black => 85,
            });
    let king_code = king_bucket_code(king);
    let opponent_code = king_bucket_code(opponent_king);
    let king_bucket = usize::from(king_code & 7);
    let opponent_bucket = opponent_code & 7;
    let mid_mirror = requires_mid_mirror_from_encodings(ours_mid, theirs_mid);
    let mirror = king_code >> 3 != 0
        || (king_bucket & 1 != 0
            && (opponent_code >> 3 != 0 || (opponent_bucket & 1 != 0 && mid_mirror)));
    (
        king_bucket * PIKAFISH_ATTACK_BUCKETS + attack_bucket,
        mirror,
    )
}

pub(super) fn layer_stack_bucket(position: &Position) -> usize {
    let us = position.side_to_move();
    let them = us.opposite();
    let us_rook = piece_count(position, us, PieceKind::Rook).min(2);
    let them_rook = piece_count(position, them, PieceKind::Rook).min(2);
    let us_minor = (piece_count(position, us, PieceKind::Horse)
        + piece_count(position, us, PieceKind::Cannon))
    .min(4);
    let them_minor = (piece_count(position, them, PieceKind::Horse)
        + piece_count(position, them, PieceKind::Cannon))
    .min(4);
    if us_rook == them_rook {
        us_rook * 4
            + usize::from(us_minor + them_minor >= 4) * 2
            + usize::from(us_minor == them_minor)
    } else if us_rook == 2 && them_rook == 1 {
        12
    } else if us_rook == 1 && them_rook == 2 {
        13
    } else if us_rook > 0 && them_rook == 0 {
        14
    } else {
        15
    }
}

pub(super) fn psq_index(
    perspective: Color,
    square: usize,
    piece: Piece,
    bucket: usize,
    mirror: bool,
) -> usize {
    let square = orient_square(perspective, square, mirror);
    let piece = if piece.color == perspective {
        piece_kind_index(piece.kind)
    } else {
        7 + piece_kind_index(piece.kind)
    };
    let offset = psq_offsets()[piece * BOARD_SIZE + square];
    assert_ne!(offset, u16::MAX, "invalid HalfKAv2_hm piece-square");
    usize::from(offset) + PIKAFISH_PS_NB * bucket
}

fn psq_offsets() -> &'static [u16] {
    static OFFSETS: OnceLock<Vec<u16>> = OnceLock::new();
    OFFSETS.get_or_init(|| {
        let mut offsets = vec![u16::MAX; STRUCTURAL_PIECE_SIZE * BOARD_SIZE];
        let mut next = 0usize;
        for piece in OFFICIAL_PIECE_ORDER {
            for square in 0..BOARD_SIZE {
                if valid_piece_square(piece, square) {
                    offsets[piece * BOARD_SIZE + square] = next as u16;
                    next += 1;
                }
            }
        }
        assert_eq!(next, PIKAFISH_PS_NB);
        offsets
    })
}

fn valid_piece_square(piece: usize, square: usize) -> bool {
    let rank = square / BOARD_FILES;
    let file = square % BOARD_FILES;
    match piece % 7 {
        4 | 5 | 3 => true, // rook, cannon, horse
        1 => match piece / 7 {
            0 => matches!((file, rank), (3, 0) | (5, 0) | (4, 1) | (3, 2) | (5, 2)),
            _ => matches!((file, rank), (3, 7) | (5, 7) | (4, 8) | (3, 9) | (5, 9)),
        },
        6 => match piece / 7 {
            0 => rank >= 5 || ((rank == 3 || rank == 4) && file % 2 == 0),
            _ => rank <= 4 || ((rank == 5 || rank == 6) && file % 2 == 0),
        },
        2 => match piece / 7 {
            0 => matches!(
                (file, rank),
                (2, 0) | (6, 0) | (0, 2) | (4, 2) | (8, 2) | (2, 4) | (6, 4)
            ),
            _ => matches!(
                (file, rank),
                (2, 5) | (6, 5) | (0, 7) | (4, 7) | (8, 7) | (2, 9) | (6, 9)
            ),
        },
        0 => match piece / 7 {
            0 => rank <= 2 && (file == 3 || file == 4),
            _ => rank >= 7 && (3..=5).contains(&file),
        },
        _ => unreachable!(),
    }
}

fn threat_key(attacker: usize, from: usize, to: usize, attacked: usize) -> usize {
    (((attacker * BOARD_SIZE + from) * BOARD_SIZE + to) * STRUCTURAL_PIECE_SIZE) + attacked
}

fn threat_offsets() -> &'static [u16] {
    static OFFSETS: OnceLock<Vec<u16>> = OnceLock::new();
    OFFSETS.get_or_init(|| {
        let mut offsets =
            vec![
                PIKAFISH_VALUE_THREAT_DIMENSIONS as u16;
                STRUCTURAL_PIECE_SIZE * BOARD_SIZE * BOARD_SIZE * STRUCTURAL_PIECE_SIZE
            ];
        let mut next = 0usize;
        for attacker in OFFICIAL_PIECE_ORDER {
            let cpp_attacker = CPP_PIECE[attacker];
            let attacker_type = cpp_attacker & 7;
            for from in 0..BOARD_SIZE {
                if !valid_piece_square(attacker, from) {
                    continue;
                }
                let attacks = pseudo_targets(attacker, from);
                for attacked in OFFICIAL_PIECE_ORDER {
                    let cpp_attacked = CPP_PIECE[attacked];
                    if !VALID_PAIRS[cpp_attacker][cpp_attacked] {
                        continue;
                    }
                    for &to in &attacks {
                        if !valid_piece_square(attacked, to) {
                            continue;
                        }
                        let enemy = attacker / 7 != attacked / 7;
                        let same_file = from % BOARD_FILES == to % BOARD_FILES;
                        let same_rank = from / BOARD_FILES == to / BOARD_FILES;
                        let attacked_type = cpp_attacked & 7;
                        let semi_excluded = attacker_type == attacked_type
                            && (attacker_type != 4
                                || (enemy && same_file)
                                || (!enemy && same_rank))
                            && attacker_type != 5;
                        if !semi_excluded || from > to {
                            offsets[threat_key(attacker, from, to, attacked)] = next as u16;
                            next += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(next, PIKAFISH_VALUE_THREAT_DIMENSIONS);
        offsets
    })
}

fn pseudo_targets(piece: usize, from: usize) -> Vec<usize> {
    let rank = (from / BOARD_FILES) as i32;
    let file = (from % BOARD_FILES) as i32;
    let mut targets = Vec::with_capacity(32);
    let mut add = |f: i32, r: i32| {
        if (0..BOARD_FILES as i32).contains(&f) && (0..BOARD_RANKS as i32).contains(&r) {
            targets.push(r as usize * BOARD_FILES + f as usize);
        }
    };
    match piece % 7 {
        4 => {
            for f in 0..BOARD_FILES as i32 {
                if f != file {
                    add(f, rank);
                }
            }
            for r in 0..BOARD_RANKS as i32 {
                if r != rank {
                    add(file, r);
                }
            }
        }
        5 => {
            for f in 0..BOARD_FILES as i32 {
                if (f - file).abs() >= 2 {
                    add(f, rank);
                }
            }
            for r in 0..BOARD_RANKS as i32 {
                if (r - rank).abs() >= 2 {
                    add(file, r);
                }
            }
        }
        3 => {
            for (df, dr) in [
                (-1, -2),
                (1, -2),
                (-2, -1),
                (2, -1),
                (-2, 1),
                (2, 1),
                (-1, 2),
                (1, 2),
            ] {
                add(file + df, rank + dr);
            }
        }
        2 => {
            for (df, dr) in [(-2, -2), (2, -2), (-2, 2), (2, 2)] {
                let r = rank + dr;
                if (piece / 7 == 0 && r <= 4) || (piece / 7 == 1 && r >= 5) {
                    add(file + df, r);
                }
            }
        }
        1 => {
            for (df, dr) in [(-1, -1), (1, -1), (-1, 1), (1, 1)] {
                let f = file + df;
                let r = rank + dr;
                if (3..=5).contains(&f) && ((0..=2).contains(&r) || (7..=9).contains(&r)) {
                    add(f, r);
                }
            }
        }
        0 => {
            for (df, dr) in [(0, -1), (0, 1), (-1, 0), (1, 0)] {
                let f = file + df;
                let r = rank + dr;
                if (3..=5).contains(&f) && ((0..=2).contains(&r) || (7..=9).contains(&r)) {
                    add(f, r);
                }
            }
        }
        6 => {
            let white = piece / 7 == 0;
            add(file, rank + if white { 1 } else { -1 });
            if (white && rank > 4) || (!white && rank < 5) {
                add(file - 1, rank);
                add(file + 1, rank);
            }
        }
        _ => unreachable!(),
    }
    targets.sort_unstable();
    targets.dedup();
    targets
}

fn visit_official_occupied_relations(
    position: &Position,
    mut visitor: impl FnMut(usize, Piece, usize, Piece),
) {
    position.visit_occupied_relations(|source, attacker, target, attacked| {
        visit_official_relation(position, source, attacker, target, attacked, &mut visitor);
    });
}

fn visit_official_occupied_relations_from(
    position: &Position,
    source: usize,
    mut visitor: impl FnMut(usize, Piece, usize, Piece),
) {
    position.visit_occupied_relations_from(source, |source, attacker, target, attacked| {
        visit_official_relation(position, source, attacker, target, attacked, &mut visitor);
    });
}

fn visit_official_relation(
    position: &Position,
    source: usize,
    attacker: Piece,
    target: usize,
    attacked: Piece,
    visitor: &mut impl FnMut(usize, Piece, usize, Piece),
) {
    if attacker.kind == PieceKind::Cannon {
        let sf = source % BOARD_FILES;
        let sr = source / BOARD_FILES;
        let tf = target % BOARD_FILES;
        let tr = target / BOARD_FILES;
        let (df, dr) = (
            (tf as i32 - sf as i32).signum(),
            (tr as i32 - sr as i32).signum(),
        );
        let mut f = sf as i32 + df;
        let mut r = sr as i32 + dr;
        let mut occupied_between = 0;
        while f != tf as i32 || r != tr as i32 {
            if position
                .piece_at(r as usize * BOARD_FILES + f as usize)
                .is_some()
            {
                occupied_between += 1;
            }
            f += df;
            r += dr;
        }
        if occupied_between != 1 {
            return;
        }
    }
    visitor(source, attacker, target, attacked);
}

fn threat_index(
    perspective: Color,
    source: usize,
    attacker: Piece,
    target: usize,
    attacked: Piece,
    mirror: bool,
) -> usize {
    let attacker = structural_piece(attacker, Color::Red);
    let attacked = structural_piece(attacked, Color::Red);
    threat_index_structural(perspective, source, attacker, target, attacked, mirror)
}

fn structural_piece(piece: Piece, perspective: Color) -> usize {
    (if piece.color == perspective { 0 } else { 7 }) + piece_kind_index(piece.kind)
}

fn threat_index_structural(
    perspective: Color,
    source: usize,
    attacker: usize,
    target: usize,
    attacked: usize,
    mirror: bool,
) -> usize {
    let (attacker, attacked) = if perspective == Color::Red {
        (attacker, attacked)
    } else {
        (
            if attacker < 7 {
                attacker + 7
            } else {
                attacker - 7
            },
            if attacked < 7 {
                attacked + 7
            } else {
                attacked - 7
            },
        )
    };
    let source = orient_square(perspective, source, mirror);
    let target = orient_square(perspective, target, mirror);
    let key = threat_key(attacker, source, target, attacked);
    usize::from(threat_offsets()[key])
}

fn orient_square(perspective: Color, square: usize, mirror: bool) -> usize {
    // Our board indexes FEN top-to-bottom, while Pikafish Square indexes A0 at
    // Red's home rank. Convert first, then apply the official IndexMap.
    let rank = square / BOARD_FILES;
    let file = square % BOARD_FILES;
    let square = (BOARD_RANKS - 1 - rank) * BOARD_FILES + file;
    let rank = square / BOARD_FILES;
    let file = square % BOARD_FILES;
    let file = if mirror { BOARD_FILES - 1 - file } else { file };
    let rank = if perspective == Color::Black {
        BOARD_RANKS - 1 - rank
    } else {
        rank
    };
    rank * BOARD_FILES + file
}

fn king_bucket_code(square: usize) -> u8 {
    let rank = square / BOARD_FILES;
    let file = square % BOARD_FILES;
    if !(3..=5).contains(&file) {
        return 0;
    }
    let relative_file = file - 3;
    let bucket_rank = match rank {
        0 | 9 => 0,
        1 | 8 => 2,
        2 | 7 => 4,
        _ => return 0,
    };
    match relative_file {
        0 => bucket_rank,
        1 => bucket_rank + 1,
        _ => 8 | bucket_rank,
    }
}

fn requires_mid_mirror_from_encodings(ours: u64, theirs: u64) -> bool {
    (ours & (1u64 << 63) != 0)
        && (theirs & (1u64 << 63) != 0)
        && (ours < BALANCE_ENCODING || (ours == BALANCE_ENCODING && theirs < BALANCE_ENCODING))
}

fn mid_encoding(position: &Position, color: Color) -> u64 {
    let mut encoding = BALANCE_ENCODING;
    for square in 0..BOARD_SIZE {
        let Some(piece) = position.piece_at(square) else {
            continue;
        };
        if piece.color == color {
            encoding = encoding.wrapping_add(piece_mid_encoding(piece, square));
        }
    }
    encoding
}

fn piece_mid_encoding(piece: Piece, square: usize) -> u64 {
    let rank = BOARD_RANKS - 1 - square / BOARD_FILES;
    let file = square % BOARD_FILES;
    if file == 4 {
        return 0;
    }
    if piece.kind == PieceKind::General {
        return 1u64 << 63;
    }
    let rank = if piece.color == Color::Red {
        rank
    } else {
        BOARD_RANKS - 1 - rank
    };
    let folded_file = if file < 4 {
        file
    } else {
        BOARD_FILES - 1 - file
    };
    let (count_shift, square_shift) = match piece.kind {
        PieceKind::Rook => (44, 0),
        PieceKind::Advisor => (60, 36),
        PieceKind::Cannon => (47, 7),
        PieceKind::Soldier => (53, 21),
        PieceKind::Horse => (50, 14),
        PieceKind::Elephant => (57, 29),
        PieceKind::General => unreachable!(),
    };
    let value =
        (1u64 << count_shift) | ((((3 - folded_file) * BOARD_RANKS + rank) as u64) << square_shift);
    if file < 4 {
        value
    } else {
        value.wrapping_neg()
    }
}

fn piece_count(position: &Position, color: Color, kind: PieceKind) -> usize {
    (0..BOARD_SIZE)
        .filter(|&square| {
            position
                .piece_at(square)
                .is_some_and(|piece| piece.color == color && piece.kind == kind)
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn indices(text: &str) -> Vec<usize> {
        if text.is_empty() {
            return Vec::new();
        }
        text.split(',')
            .map(|value| value.parse().unwrap())
            .collect()
    }

    fn assert_golden(
        fen: &str,
        stack: usize,
        perspective: Color,
        bucket: usize,
        mirror: bool,
        psq: &str,
        threats: &str,
    ) {
        let position = Position::from_fen(fen).unwrap();
        assert_eq!(layer_stack_bucket(&position), stack);
        let active = active_value_features(&position, perspective);
        assert_eq!((active.feature_bucket, active.mirror), (bucket, mirror));
        assert_eq!(active.psq, indices(psq));
        assert_eq!(active.threats, indices(threats));
    }

    #[test]
    fn official_compact_piece_square_map_has_689_entries() {
        assert_eq!(
            psq_offsets()
                .iter()
                .filter(|&&offset| offset != u16::MAX)
                .count(),
            PIKAFISH_PS_NB
        );
    }

    #[test]
    fn halfka_features_are_file_mirror_invariant() {
        let position =
            Position::from_fen("3ak4/4a4/2n1b4/p3p3p/4R4/2P6/P3P3P/2N1C4/4A4/2BAK4 b").unwrap();
        let mirrored = position.mirror_files();
        for perspective in [Color::Red, Color::Black] {
            let left = active_value_features(&position, perspective);
            let right = active_value_features(&mirrored, perspective);
            assert_eq!(left.psq, right.psq);
            assert_eq!(left.threats, right.threats);
            assert_eq!(left.feature_bucket, right.feature_bucket);
            assert_ne!(left.mirror, right.mirror);
        }
    }

    #[test]
    fn start_position_uses_official_material_stack() {
        assert_eq!(layer_stack_bucket(&Position::startpos()), 11);
    }

    #[test]
    fn dirty_threat_indices_match_full_refresh_on_random_legal_moves() {
        let mut state = 0x4554_7de9_348b_c6eau64;
        for game in 0..20 {
            let mut position = Position::startpos();
            for ply in 0..100 {
                let legal = position.legal_moves();
                if legal.is_empty() {
                    break;
                }
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                let mv = legal[(state.wrapping_mul(0x2545_f491_4f6c_dd1d) as usize) % legal.len()];
                let before = position.clone();
                let before_active = active_value_features_pair(&before);
                position.make_move(mv);
                let after_active = active_value_features_pair(&position);
                if (0..2).any(|view| {
                    before_active[view].feature_bucket != after_active[view].feature_bucket
                        || before_active[view].mirror != after_active[view].mirror
                }) {
                    continue;
                }
                let dirty = changed_threat_indices(
                    &before,
                    &position,
                    [before_active[0].mirror, before_active[1].mirror],
                );
                for view in 0..2 {
                    let before_set = before_active[view]
                        .threats
                        .iter()
                        .copied()
                        .collect::<BTreeSet<_>>();
                    let after_set = after_active[view]
                        .threats
                        .iter()
                        .copied()
                        .collect::<BTreeSet<_>>();
                    let removed = before_set
                        .difference(&after_set)
                        .copied()
                        .collect::<Vec<_>>();
                    let added = after_set
                        .difference(&before_set)
                        .copied()
                        .collect::<Vec<_>>();
                    assert_eq!(
                        dirty[view].0.as_slice(),
                        removed,
                        "removed game={game} ply={ply}"
                    );
                    assert_eq!(
                        dirty[view].1.as_slice(),
                        added,
                        "added game={game} ply={ply}"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "manual fast-profile FullThreats delta benchmark"]
    fn benchmark_dirty_threat_indices() {
        use std::hint::black_box;
        use std::time::Instant;

        let mut positions = Vec::new();
        let mut position = Position::startpos();
        for ply in 0..100 {
            let legal = position.legal_moves();
            if legal.is_empty() {
                break;
            }
            let before = position.clone();
            position.make_move(legal[(ply * 17 + 3) % legal.len()]);
            let active = value_feature_states_pair(&before);
            positions.push((
                before,
                position.clone(),
                [active[0].mirror, active[1].mirror],
            ));
        }
        let started = Instant::now();
        for _ in 0..100 {
            for (before, after, mirrors) in &positions {
                black_box(changed_threat_indices(before, after, *mirrors));
            }
        }
        let dirty = started.elapsed();
        let started = Instant::now();
        for _ in 0..100 {
            for (_, after, _) in &positions {
                black_box(active_value_features_pair(after));
            }
        }
        let refresh = started.elapsed();
        eprintln!(
            "FullThreats per transition: dirty={:.3}us refresh={:.3}us",
            dirty.as_secs_f64() * 1.0e6 / (positions.len() * 100) as f64,
            refresh.as_secs_f64() * 1.0e6 / (positions.len() * 100) as f64,
        );
    }

    #[test]
    fn incremental_feature_state_matches_full_refresh_on_random_moves() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..20 {
            let mut position = Position::startpos();
            let mut state = value_feature_states_pair(&position);
            for _ in 0..100 {
                let legal = position.legal_moves();
                if legal.is_empty() {
                    break;
                }
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let mv = legal[seed as usize % legal.len()];
                let moved = position.piece_at(mv.from as usize).unwrap();
                let captured = position.piece_at(mv.to as usize);
                position.make_move(mv);
                state = transitioned_psq_features_pair(
                    &position,
                    &state,
                    mv.from as usize,
                    mv.to as usize,
                    moved,
                    captured,
                );
                assert_eq!(state, value_feature_states_pair(&position));
            }
        }
    }

    #[test]
    fn features_match_official_cpp_golden() {
        let start = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w";
        let start_psq = "4823,4831,4913,4914,4937,4943,5008,5009,5010,5011,5012,5064,5070,5153,5154,5161,5247,5255,5259,5260,5325,5331,5401,5402,5403,5404,5405,5488,5494,5501,5502,5510";
        let start_threats =
            "19,26,843,857,11720,12217,31712,31722,32697,32714,38265,38820,45529,45530";
        assert_golden(start, 11, Color::Red, 7, false, start_psq, start_threats);
        assert_golden(start, 11, Color::Black, 7, false, start_psq, start_threats);

        let asymmetric = "3ak4/4a4/2n1b4/p3p3p/4R4/2P6/P3P3P/2N1C4/4A4/2BAK4 b";
        assert_golden(
            asymmetric,
            15,
            Color::Red,
            7,
            false,
            "4872,4913,4915,4940,5008,5010,5012,5014,5083,5153,5161,5258,5259,5401,5403,5405,5471,5499,5510",
            "5273,5344,10048,11888,11918,19304,19305,19314,19315,22515,32776,44362,44363,44372,44373,45528,45529",
        );
        assert_golden(
            asymmetric,
            15,
            Color::Black,
            5,
            false,
            "3535,3537,3630,3632,3634,3705,3778,3783,3828,3880,3881,3950,4019,4023,4025,4027,4093,4123,4132",
            "10048,19304,19305,19314,19315,26866,26935,32776,38554,38593,44362,44363,44372,44373,45389,45528,45529",
        );

        let mirrored = "4ka3/4a4/4b1n2/p3p3p/4R4/6P2/P3P3P/4C1N2/4A4/4KAB2 b";
        assert_golden(
            mirrored,
            15,
            Color::Red,
            7,
            true,
            "4872,4913,4915,4940,5008,5010,5012,5014,5083,5153,5161,5258,5259,5401,5403,5405,5471,5499,5510",
            "5273,5344,10048,11888,11918,19304,19305,19314,19315,22515,32776,44362,44363,44372,44373,45528,45529",
        );
        assert_golden(
            mirrored,
            15,
            Color::Black,
            5,
            true,
            "3535,3537,3630,3632,3634,3705,3778,3783,3828,3880,3881,3950,4019,4023,4025,4027,4093,4123,4132",
            "10048,19304,19305,19314,19315,26866,26935,32776,38554,38593,44362,44363,44372,44373,45389,45528,45529",
        );
    }
}
