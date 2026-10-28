//! 马的两跳占用可达与蹩马腿关系，均作为现有共享威胁表的普通特征行。
use std::sync::OnceLock;

use super::arch::{
    HORSE_LEG_OFFSET, HORSE_LEG_VOCAB, HORSE_TWO_HOP_OFFSET, HORSE_TWO_HOP_VOCAB,
    canonical_square_for,
};
use super::nnue::{canonical_piece_square_reachable, piece_absolute_feature_index};
use crate::xiangqi::{
    BOARD_FILES, BOARD_RANKS, BOARD_SIZE, Color, HORSE_STEPS, PieceKind, Position,
};

const TWO_HOP_PAIRS: usize = 1_708;
const HORSE_ENTRY_TAG: u32 = 1 << 24;

struct Jump {
    to: usize,
    leg: usize,
}

struct HorseGeometry {
    jumps: [Vec<Jump>; BOARD_SIZE],
    legs: [u128; BOARD_SIZE],
    pairs: Vec<u16>,
    blockers: Vec<u16>,
    dependencies: [(u128, u128); BOARD_SIZE],
}

fn leg_direction(source: usize, leg: usize) -> usize {
    if source / BOARD_FILES != leg / BOARD_FILES {
        usize::from(leg > source)
    } else {
        2 + usize::from(leg > source)
    }
}

fn blocker_key(owner: usize, source: usize, direction: usize, blocker: usize) -> usize {
    ((owner * BOARD_SIZE + source) * 4 + direction) * 14 + blocker
}

fn geometry() -> &'static HorseGeometry {
    static MAP: OnceLock<HorseGeometry> = OnceLock::new();
    MAP.get_or_init(|| {
        let jumps: [Vec<Jump>; BOARD_SIZE] = std::array::from_fn(|source| {
            let (file, rank) = ((source % BOARD_FILES) as i32, (source / BOARD_FILES) as i32);
            HORSE_STEPS
                .into_iter()
                .filter_map(|((lf, lr), (df, dr))| {
                    let (f, r) = (file + df, rank + dr);
                    ((0..BOARD_FILES as i32).contains(&f) && (0..BOARD_RANKS as i32).contains(&r))
                        .then(|| Jump {
                            to: r as usize * BOARD_FILES + f as usize,
                            leg: (rank + lr) as usize * BOARD_FILES + (file + lf) as usize,
                        })
                })
                .collect()
        });
        let legs = std::array::from_fn(|source| {
            jumps[source]
                .iter()
                .fold(0, |mask, j| mask | 1u128 << j.leg)
        });
        let mut pairs = vec![u16::MAX; BOARD_SIZE * BOARD_SIZE];
        let mut dependencies = [(0, 0); BOARD_SIZE];
        let mut count = 0;
        for source in 0..BOARD_SIZE {
            let mut targets = legs[source];
            let mut blockers = legs[source];
            let mut endpoints = 0u128;
            for first in &jumps[source] {
                targets |= 1u128 << first.to;
                for second in &jumps[first.to] {
                    blockers |= 1u128 << second.leg;
                    if second.to != source {
                        endpoints |= 1u128 << second.to;
                    }
                }
            }
            targets |= endpoints;
            dependencies[source] = (targets, blockers);
            while endpoints != 0 {
                let to = endpoints.trailing_zeros() as usize;
                endpoints &= endpoints - 1;
                pairs[source * BOARD_SIZE + to] = count;
                count += 1;
            }
        }
        assert_eq!(usize::from(count), TWO_HOP_PAIRS);
        let mut blockers = vec![u16::MAX; 2 * BOARD_SIZE * 4 * 14];
        let mut count = 0;
        for owner in 0..2 {
            for (source, &mask) in legs.iter().enumerate() {
                let mut mask = mask;
                while mask != 0 {
                    let leg = mask.trailing_zeros() as usize;
                    mask &= mask - 1;
                    for blocker in 0..14 {
                        if canonical_piece_square_reachable(blocker, leg) {
                            blockers
                                [blocker_key(owner, source, leg_direction(source, leg), blocker)] =
                                count;
                            count += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(usize::from(count), HORSE_LEG_VOCAB);
        assert_eq!(TWO_HOP_PAIRS * 2 * 2, HORSE_TWO_HOP_VOCAB);
        HorseGeometry {
            jumps,
            legs,
            pairs,
            blockers,
            dependencies,
        }
    })
}

fn two_hop_index(owner: usize, source: usize, target: usize, capture: bool) -> Option<usize> {
    let pair = geometry().pairs[source * BOARD_SIZE + target];
    (pair != u16::MAX).then(|| {
        HORSE_TWO_HOP_OFFSET
            + (owner * TWO_HOP_PAIRS + usize::from(pair)) * 2
            + usize::from(capture)
    })
}

fn leg_index(owner: usize, source: usize, direction: usize, blocker: usize) -> Option<usize> {
    let row = geometry().blockers[blocker_key(owner, source, direction, blocker)];
    (row != u16::MAX).then(|| HORSE_LEG_OFFSET + usize::from(row))
}

/// 其他棋子静止时的两次马跳；允许第一跳吃敌子，排除取将、回原位和己方占用落点。
/// 终点敌将作为潜在将军目标保留；不判己将安全，也不模拟对方应手。
fn two_hop_targets(position: &Position, source: usize, color: Color) -> u128 {
    let mut targets = 0;
    for first in &geometry().jumps[source] {
        if position.piece_at(first.leg).is_some()
            || position
                .piece_at(first.to)
                .is_some_and(|p| p.color == color || p.kind == PieceKind::General)
        {
            continue;
        }
        for second in &geometry().jumps[first.to] {
            if second.to == source
                || (second.leg != source && position.piece_at(second.leg).is_some())
                || position
                    .piece_at(second.to)
                    .is_some_and(|p| p.color == color)
            {
                continue;
            }
            targets |= 1u128 << second.to;
        }
    }
    targets
}

pub(crate) fn horse_dependency_masks(source: usize) -> (u128, u128) {
    geometry().dependencies[source]
}

pub(crate) fn visit_horse_threat_entries_from(
    position: &Position,
    perspective: Color,
    sources: u128,
    mut visitor: impl FnMut(u32),
) {
    for source in position.occupied_squares_from(sources) {
        let horse = position.piece_at(source).unwrap();
        if horse.kind != PieceKind::Horse {
            continue;
        }
        let owner = usize::from(horse.color != perspective);
        let canonical = canonical_square_for(perspective, source);
        let tag = HORSE_ENTRY_TAG | (source as u32) << 16;
        let mut targets = two_hop_targets(position, source, horse.color);
        while targets != 0 {
            let target = targets.trailing_zeros() as usize;
            targets &= targets - 1;
            let row = two_hop_index(
                owner,
                canonical,
                canonical_square_for(perspective, target),
                position.piece_at(target).is_some(),
            )
            .expect("两跳终点必须有几何特征行");
            visitor(tag | row as u32);
        }
        let mut legs = geometry().legs[source];
        while legs != 0 {
            let leg = legs.trailing_zeros() as usize;
            legs &= legs - 1;
            if let Some(blocker) = position.piece_at(leg) {
                let direction = leg_direction(canonical, canonical_square_for(perspective, leg));
                let blocker = piece_absolute_feature_index(perspective, blocker);
                let row = leg_index(owner, canonical, direction, blocker)
                    .expect("堵腿棋子必须有可达特征行");
                visitor(tag | row as u32);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::az::{
        SplitMix64, THREAT_SOURCE_MAX_ACTIVE, VALUE_THREAT_MAX_ACTIVE,
        visit_value_threat_entries_from,
    };
    use crate::xiangqi::Move;

    fn slow_leg(from: usize, to: usize) -> Option<usize> {
        let (r, f) = (from / 9, from % 9);
        let (tr, tf) = (to / 9, to % 9);
        match (r.abs_diff(tr), f.abs_diff(tf)) {
            (2, 1) => Some((r + tr) / 2 * 9 + f),
            (1, 2) => Some(r * 9 + (f + tf) / 2),
            _ => None,
        }
    }

    // 逐一枚举中间格，并实际走第一步；独立核对虚拟占用与吃子语义。
    fn slow_targets(position: &Position, source: usize) -> u128 {
        let color = position.piece_at(source).unwrap().color;
        let mut result = 0;
        for middle in 0..BOARD_SIZE {
            let Some(leg) = slow_leg(source, middle) else {
                continue;
            };
            if position.piece_at(leg).is_some()
                || position
                    .piece_at(middle)
                    .is_some_and(|p| p.color == color || p.kind == PieceKind::General)
            {
                continue;
            }
            let mut moved = position.clone();
            moved.make_move(Move::new(source, middle));
            for target in 0..BOARD_SIZE {
                let Some(leg) = slow_leg(middle, target) else {
                    continue;
                };
                if target != source
                    && moved.piece_at(leg).is_none()
                    && moved.piece_at(target).is_none_or(|p| p.color != color)
                {
                    result |= 1u128 << target;
                }
            }
        }
        result
    }

    #[test]
    fn horse_tables_have_only_geometric_rows_and_no_aliases() {
        let mut hops = vec![false; HORSE_TWO_HOP_VOCAB];
        let mut legs = vec![false; HORSE_LEG_VOCAB];
        for source in 0..BOARD_SIZE {
            for target in 0..BOARD_SIZE {
                let possible = target != source
                    && (0..BOARD_SIZE)
                        .any(|m| slow_leg(source, m).is_some() && slow_leg(m, target).is_some());
                for owner in 0..2 {
                    for capture in [false, true] {
                        let row = two_hop_index(owner, source, target, capture);
                        assert_eq!(row.is_some(), possible);
                        if let Some(row) = row {
                            let row = row - HORSE_TWO_HOP_OFFSET;
                            assert!(!hops[row]);
                            hops[row] = true;
                        }
                    }
                }
            }
            let valid_legs = (0..BOARD_SIZE)
                .filter_map(|t| slow_leg(source, t))
                .fold(0u128, |m, l| m | 1 << l);
            for (direction, (dr, df)) in [(-1, 0), (1, 0), (0, -1), (0, 1)].into_iter().enumerate()
            {
                let (r, f) = (source as i32 / 9 + dr, source as i32 % 9 + df);
                for owner in 0..2 {
                    for blocker in 0..14 {
                        let possible = (0..10).contains(&r)
                            && (0..9).contains(&f)
                            && valid_legs & (1u128 << (r * 9 + f)) != 0
                            && canonical_piece_square_reachable(blocker, (r * 9 + f) as usize);
                        let row = leg_index(owner, source, direction, blocker);
                        assert_eq!(row.is_some(), possible);
                        if let Some(row) = row {
                            let row = row - HORSE_LEG_OFFSET;
                            assert!(!legs[row]);
                            legs[row] = true;
                        }
                    }
                }
            }
        }
        assert!(hops.into_iter().all(|x| x));
        assert!(legs.into_iter().all(|x| x));
    }

    #[test]
    fn horse_features_match_virtual_moves_in_both_views_and_keep_source_tags() {
        let mut rng = SplitMix64::new(20261028);
        let mut seen = [0; 3];
        for _ in 0..3 {
            let mut position = Position::startpos();
            for _ in 0..100 {
                for perspective in [Color::Red, Color::Black] {
                    let mut expected = Vec::new();
                    for source in position.occupied_squares() {
                        let horse = position.piece_at(source).unwrap();
                        if horse.kind != PieceKind::Horse {
                            continue;
                        }
                        let mut targets = slow_targets(&position, source);
                        assert_eq!(two_hop_targets(&position, source, horse.color), targets);
                        let owner = usize::from(horse.color != perspective);
                        let canonical = canonical_square_for(perspective, source);
                        let tag = HORSE_ENTRY_TAG | (source as u32) << 16;
                        while targets != 0 {
                            let target = targets.trailing_zeros() as usize;
                            targets &= targets - 1;
                            let capture = position.piece_at(target).is_some();
                            seen[usize::from(capture)] += 1;
                            expected.push(
                                tag | two_hop_index(
                                    owner,
                                    canonical,
                                    canonical_square_for(perspective, target),
                                    capture,
                                )
                                .unwrap() as u32,
                            );
                        }
                        let mut legs = (0..BOARD_SIZE)
                            .filter_map(|t| slow_leg(source, t))
                            .fold(0u128, |m, l| m | 1 << l);
                        while legs != 0 {
                            let leg = legs.trailing_zeros() as usize;
                            legs &= legs - 1;
                            if let Some(p) = position.piece_at(leg) {
                                seen[2] += 1;
                                let direction = leg_direction(
                                    canonical,
                                    canonical_square_for(perspective, leg),
                                );
                                expected.push(
                                    tag | leg_index(
                                        owner,
                                        canonical,
                                        direction,
                                        piece_absolute_feature_index(perspective, p),
                                    )
                                    .unwrap() as u32,
                                );
                            }
                        }
                    }
                    let mut actual = Vec::new();
                    visit_horse_threat_entries_from(&position, perspective, u128::MAX, |e| {
                        actual.push(e)
                    });
                    assert_eq!(actual, expected, "{} {perspective:?}", position.to_fen());
                    let mut all = Vec::new();
                    visit_value_threat_entries_from(&position, perspective, u128::MAX, |e| {
                        all.push(e)
                    });
                    assert!(all.len() <= VALUE_THREAT_MAX_ACTIVE);
                    assert!(all.windows(2).all(|w| w[0] >> 16 <= w[1] >> 16));
                    let mut counts = std::collections::HashMap::new();
                    for e in all {
                        *counts.entry(e >> 16).or_insert(0) += 1;
                    }
                    assert!(counts.values().all(|&n| n <= THREAT_SOURCE_MAX_ACTIVE));
                }
                let moves = position.legal_moves();
                if moves.is_empty() {
                    break;
                }
                position.make_move(moves[rng.next_u64() as usize % moves.len()]);
            }
        }
        assert!(seen.into_iter().all(|n| n > 100));
    }

    #[test]
    fn occupied_middle_captures_leg_blocks_and_king_targets_are_distinct() {
        let p = |extra: &[(usize, usize)]| {
            let mut pieces = vec![(0, 85), (7, 4), (3, 40)];
            pieces.extend_from_slice(extra);
            Position::from_canonical_piece_squares(&pieces)
        };
        // e5-d7-c9 唯一路径，中间格d7可吃敌子，己子禁止通过，第二马腿为d8。
        let empty = p(&[]);
        let target = 2;
        assert_ne!(two_hop_targets(&empty, 40, Color::Red) & (1 << target), 0);
        assert_eq!(
            two_hop_targets(&p(&[(4, 21)]), 40, Color::Red) & (1 << target),
            0
        );
        assert_ne!(
            two_hop_targets(&p(&[(11, 21)]), 40, Color::Red) & (1 << target),
            0
        );
        assert_eq!(
            two_hop_targets(&p(&[(11, 12)]), 40, Color::Red) & (1 << target),
            0
        );
        assert_eq!(
            two_hop_targets(&p(&[(4, target)]), 40, Color::Red) & (1 << target),
            0
        );
        assert_ne!(two_hop_targets(&empty, 40, Color::Red) & (1 << 4), 0);
        let king_middle = Position::from_canonical_piece_squares(&[(0, 85), (7, 21), (3, 40)]);
        assert_eq!(
            two_hop_targets(&king_middle, 40, Color::Red) & (1 << target),
            0
        );
        assert_eq!(two_hop_targets(&empty, 40, Color::Red) & (1 << 40), 0);
    }
}
