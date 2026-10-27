//! 稀疏特征的几何紧凑编号；训练、完整推理与增量推理共用。

use std::sync::OnceLock;

use crate::az::nnue::{V2_KING_BUCKETS, canonical_piece_square_reachable};
use crate::xiangqi::{BOARD_FILES, BOARD_RANKS, BOARD_SIZE};

use super::arch::{VALUE_KING_PIECE_VOCAB, VALUE_RAY_VOCAB};

fn king_piece_map() -> &'static [u16] {
    static MAP: OnceLock<Vec<u16>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut map = vec![u16::MAX; 2 * V2_KING_BUCKETS * 14 * BOARD_SIZE];
        let mut next = 0;
        for side in 0..2 {
            for bucket in 0..V2_KING_BUCKETS {
                let local = (7 + bucket / 3) * BOARD_FILES + 3 + bucket % 3;
                let king = if side == 0 {
                    local
                } else {
                    BOARD_SIZE - 1 - local
                };
                for piece in 0..14 {
                    for square in 0..BOARD_SIZE {
                        if !canonical_piece_square_reachable(piece, square)
                            || (piece == side * 7) != (square == king)
                        {
                            continue;
                        }
                        let key =
                            ((side * V2_KING_BUCKETS + bucket) * 14 + piece) * BOARD_SIZE + square;
                        map[key] = next;
                        next += 1;
                    }
                }
            }
        }
        assert_eq!(usize::from(next), VALUE_KING_PIECE_VOCAB);
        map
    })
}

#[inline]
pub(crate) fn value_king_piece_index(
    side: usize,
    bucket: usize,
    piece: usize,
    square: usize,
) -> Option<usize> {
    let key = ((side * V2_KING_BUCKETS + bucket) * 14 + piece) * BOARD_SIZE + square;
    let index = king_piece_map()[key];
    (index != u16::MAX).then_some(usize::from(index))
}

fn ray_key(
    attacker: usize,
    ordinal: usize,
    direction: usize,
    distance: usize,
    blocked: usize,
    state: usize,
) -> usize {
    (((((attacker * 2 + ordinal) * 4 + direction) * 9 + distance) * 15 + blocked) * 4) + state
}

fn ray_map() -> &'static [u16] {
    static MAP: OnceLock<Vec<u16>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut map = vec![u16::MAX; 4 * 2 * 4 * 9 * 15 * 4];
        for attacker in 0..4 {
            for (direction, (dr, df)) in [(-1, 0), (1, 0), (0, -1), (0, 1)].into_iter().enumerate()
            {
                map[ray_key(attacker, 0, direction, 0, 14, 0)] = 0;
                for source in 0..BOARD_SIZE {
                    let rank = source / BOARD_FILES;
                    let file = source % BOARD_FILES;
                    let length = match direction {
                        0 => rank,
                        1 => BOARD_RANKS - 1 - rank,
                        2 => file,
                        3 => BOARD_FILES - 1 - file,
                        _ => unreachable!(),
                    };
                    for distance in 0..length {
                        let target_rank = rank as isize + dr * (distance + 1) as isize;
                        let target_file = file as isize + df * (distance + 1) as isize;
                        let target = target_rank as usize * BOARD_FILES + target_file as usize;
                        for blocked in 0..14 {
                            if !canonical_piece_square_reachable(blocked, target) {
                                continue;
                            }
                            for ordinal in 0..2 {
                                if distance < ordinal {
                                    continue;
                                }
                                for state in ordinal + 1..=3 {
                                    // 第二阻挡子前至少留一格；state=3 表示至少三个阻挡子。
                                    if length - distance - 1 < state - ordinal - 1 {
                                        continue;
                                    }
                                    map[ray_key(
                                        attacker, ordinal, direction, distance, blocked, state,
                                    )] = 0;
                                }
                            }
                        }
                    }
                }
            }
        }
        let mut next = 0;
        for index in &mut map {
            if *index != u16::MAX {
                *index = next;
                next += 1;
            }
        }
        assert_eq!(usize::from(next), VALUE_RAY_VOCAB);
        map
    })
}

#[inline]
pub(crate) fn value_ray_index(
    attacker: usize,
    ordinal: usize,
    direction: usize,
    distance: usize,
    blocked: usize,
    state: usize,
) -> Option<usize> {
    let index = ray_map()[ray_key(attacker, ordinal, direction, distance, blocked, state)];
    (index != u16::MAX).then_some(usize::from(index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::az::nnue::{canonical_square, piece_absolute_feature_index};
    use crate::az::{
        SplitMix64, VALUE_THREAT_PAIR_VOCAB, canonical_buckets_for_perspective,
        visit_value_king_piece_features, visit_value_threat_entries_from,
    };
    use crate::xiangqi::{Color, PieceKind, Position};

    fn ray_squares(source: usize, direction: usize) -> Vec<usize> {
        let (dr, df) = [(-1, 0), (1, 0), (0, -1), (0, 1)][direction];
        let (mut rank, mut file) = (source as isize / 9, source as isize % 9);
        let mut squares = Vec::new();
        loop {
            rank += dr;
            file += df;
            if !(0..10).contains(&rank) || !(0..9).contains(&file) {
                return squares;
            }
            squares.push((rank * 9 + file) as usize);
        }
    }

    fn inverse(map: &[u16], size: usize) -> Vec<usize> {
        let mut keys = vec![usize::MAX; size];
        for (key, &index) in map.iter().enumerate() {
            if index != u16::MAX {
                assert_eq!(keys[index as usize], usize::MAX, "feature collision");
                keys[index as usize] = key;
            }
        }
        assert!(keys.iter().all(|&key| key != usize::MAX));
        keys
    }

    #[test]
    fn king_piece_compaction_has_no_aliases_or_impossible_rows() {
        inverse(king_piece_map(), VALUE_KING_PIECE_VOCAB);
        let mut count = 0;
        for side in 0..2 {
            for king in 0..BOARD_SIZE {
                if !canonical_piece_square_reachable(side * 7, king) {
                    continue;
                }
                let bucket = crate::az::arch::canonical_general_bucket(side * 7, king);
                for piece in 0..14 {
                    for square in 0..BOARD_SIZE {
                        let possible = canonical_piece_square_reachable(piece, square)
                            && if piece == side * 7 {
                                square == king
                            } else {
                                square != king
                            };
                        let index = value_king_piece_index(side, bucket, piece, square);
                        assert_eq!(index.is_some(), possible);
                        count += usize::from(index.is_some());
                    }
                }
            }
        }
        assert_eq!(count, 12_174);
    }

    #[test]
    fn ray_compaction_matches_exhaustive_blocker_patterns() {
        let mut possible = vec![false; ray_map().len()];
        for source in 0..BOARD_SIZE {
            for direction in 0..4 {
                let squares = ray_squares(source, direction);
                // 独立枚举整条射线的所有占用位板，覆盖首/次阻挡子和至少三个阻挡子。
                for occupancy in 0usize..1 << squares.len() {
                    let blockers: Vec<_> = squares
                        .iter()
                        .enumerate()
                        .filter(|(distance, _)| occupancy & (1 << distance) != 0)
                        .collect();
                    for attacker in 0..4 {
                        if blockers.is_empty() {
                            possible[ray_key(attacker, 0, direction, 0, 14, 0)] = true;
                        }
                        for (ordinal, &(distance, &target)) in blockers.iter().take(2).enumerate() {
                            for blocked in 0..14 {
                                if canonical_piece_square_reachable(blocked, target) {
                                    possible[ray_key(
                                        attacker,
                                        ordinal,
                                        direction,
                                        distance,
                                        blocked,
                                        blockers.len().min(3),
                                    )] = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        for (&possible, &index) in possible.iter().zip(ray_map()) {
            assert_eq!(possible, index != u16::MAX);
        }
        assert_eq!(possible.iter().filter(|&&x| x).count(), 7_040);
        inverse(ray_map(), VALUE_RAY_VOCAB);
    }

    #[test]
    fn compact_features_preserve_all_relations_in_both_perspectives() {
        let king_keys = inverse(king_piece_map(), VALUE_KING_PIECE_VOCAB);
        let ray_keys = inverse(ray_map(), VALUE_RAY_VOCAB);
        let mut rng = SplitMix64::new(20261027);
        for _ in 0..4 {
            let mut position = Position::startpos();
            for _ in 0..160 {
                if [Color::Red, Color::Black]
                    .iter()
                    .any(|&side| position.general_square(side).is_none())
                {
                    break;
                }
                for side in [Color::Red, Color::Black] {
                    let buckets = canonical_buckets_for_perspective(&position, side);
                    let mut expected_kings = Vec::new();
                    let mut expected_rays = Vec::new();
                    for source in position.occupied_squares() {
                        let piece = position.piece_at(source).unwrap();
                        let class = piece_absolute_feature_index(side, piece);
                        let canonical = canonical_square(side, source);
                        for (king, bucket) in [(0, buckets.0), (1, buckets.1)] {
                            expected_kings
                                .push(((king * 9 + bucket) * 14 + class) * 90 + canonical);
                        }
                        if !matches!(piece.kind, PieceKind::Rook | PieceKind::Cannon) {
                            continue;
                        }
                        let attacker = usize::from(piece.color != side) * 2
                            + usize::from(piece.kind == PieceKind::Cannon);
                        for direction in 0..4 {
                            let squares = ray_squares(canonical, direction);
                            let blockers: Vec<_> = squares
                                .iter()
                                .enumerate()
                                .filter_map(|(distance, &sq)| {
                                    position
                                        .piece_at(canonical_square(side, sq))
                                        .map(|p| (distance, piece_absolute_feature_index(side, p)))
                                })
                                .collect();
                            if blockers.is_empty() {
                                expected_rays
                                    .push((source, ray_key(attacker, 0, direction, 0, 14, 0)));
                            }
                            for (ordinal, &(distance, blocked)) in
                                blockers.iter().take(2).enumerate()
                            {
                                expected_rays.push((
                                    source,
                                    ray_key(
                                        attacker,
                                        ordinal,
                                        direction,
                                        distance,
                                        blocked,
                                        blockers.len().min(3),
                                    ),
                                ));
                            }
                        }
                    }
                    let mut actual_kings = Vec::new();
                    visit_value_king_piece_features(&position, side, |index| {
                        actual_kings.push(king_keys[index])
                    });
                    assert_eq!(actual_kings, expected_kings);
                    let mut actual_rays = Vec::new();
                    visit_value_threat_entries_from(&position, side, u128::MAX, |entry| {
                        let feature = (entry & 0xffff) as usize;
                        if (VALUE_THREAT_PAIR_VOCAB..VALUE_THREAT_PAIR_VOCAB + VALUE_RAY_VOCAB)
                            .contains(&feature)
                        {
                            actual_rays.push((
                                ((entry >> 16) & 127) as usize,
                                ray_keys[feature - VALUE_THREAT_PAIR_VOCAB],
                            ));
                        }
                    });
                    actual_rays.sort_unstable();
                    expected_rays.sort_unstable();
                    assert_eq!(actual_rays, expected_rays, "{} {side:?}", position.to_fen());
                }
                let moves = position.legal_moves();
                if moves.is_empty() {
                    break;
                }
                position.make_move(moves[rng.next_u64() as usize % moves.len()]);
            }
        }
    }
}
