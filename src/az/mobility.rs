//! 车马炮的占用可走落点与当前敌控覆盖，作为现有共享威胁表的一组联合特征。
use super::arch::MOBILITY_OFFSET;
use super::horse::horse_move_targets;
use crate::xiangqi::{
    Color, Piece, PieceKind, Position, color_index, nearest_on_ray, orthogonal_ray_masks,
    ray_through,
};

pub(crate) const MOBILITY_ENTRY_TAG: u32 = 1 << 25;

fn bucket(count: u32) -> usize {
    match count {
        0 => 0,
        1 => 1,
        2..=3 => 2,
        _ => 3,
    }
}

/// 按占用与走子规则计算，不检验自将；受敌控落点仅作为评估信息。
pub(crate) fn move_targets(position: &Position, source: usize, piece: Piece) -> u128 {
    if piece.kind == PieceKind::Horse {
        return horse_move_targets(position, source, piece.color);
    }
    debug_assert!(matches!(piece.kind, PieceKind::Rook | PieceKind::Cannon));
    let mut targets = 0;
    for direction in 0..4 {
        let ray = orthogonal_ray_masks()[source][direction];
        let blockers = position.orthogonal_blockers(source, direction);
        if blockers == 0 {
            targets |= ray;
            continue;
        }
        let increasing = direction == 0 || direction == 2;
        let first = nearest_on_ray(blockers, increasing);
        targets |= ray_through(ray, first, increasing) & !(1u128 << first);
        let capture = if piece.kind == PieceKind::Rook {
            Some(first)
        } else {
            let rest = blockers & !(1u128 << first);
            (rest != 0).then(|| nearest_on_ray(rest, increasing))
        };
        if let Some(capture) = capture {
            let victim = position.piece_at(capture).unwrap();
            if victim.color != piece.color && victim.kind != PieceKind::General {
                targets |= 1u128 << capture;
            }
        }
    }
    targets
}

pub(crate) fn visit_mobility_entries_from(
    position: &Position,
    perspective: Color,
    sources: u128,
    attacks: [u128; 2],
    mut visitor: impl FnMut(u32),
) {
    for source in position.occupied_squares_from(sources) {
        let piece = position.piece_at(source).unwrap();
        let kind = match piece.kind {
            PieceKind::Rook => 0,
            PieceKind::Horse => 1,
            PieceKind::Cannon => 2,
            _ => continue,
        };
        let targets = move_targets(position, source, piece);
        let controlled = targets & attacks[color_index(piece.color.opposite())];
        let free = targets & !controlled;
        let owner = usize::from(piece.color != perspective);
        let row = MOBILITY_OFFSET
            + ((owner * 3 + kind) * 4 + bucket(free.count_ones())) * 4
            + bucket(controlled.count_ones());
        visitor(MOBILITY_ENTRY_TAG | (source as u32) << 16 | row as u32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::az::{AzEvalScratch, AzNnue, MOBILITY_VOCAB, RULE_CONTEXT_SIZE, SplitMix64};
    use crate::xiangqi::BOARD_SIZE;

    fn slow_targets(position: &Position, source: usize, piece: Piece) -> u128 {
        let (r, f) = (source / 9, source % 9);
        let mut targets = 0;
        for target in 0..BOARD_SIZE {
            if target == source
                || position
                    .piece_at(target)
                    .is_some_and(|p| p.color == piece.color || p.kind == PieceKind::General)
            {
                continue;
            }
            let (tr, tf) = (target / 9, target % 9);
            let possible = if piece.kind == PieceKind::Horse {
                let leg = match (r.abs_diff(tr), f.abs_diff(tf)) {
                    (2, 1) => Some((r + tr) / 2 * 9 + f),
                    (1, 2) => Some(r * 9 + (f + tf) / 2),
                    _ => None,
                };
                leg.is_some_and(|l| position.piece_at(l).is_none())
            } else if r == tr || f == tf {
                let step = if r == tr {
                    if tf > f { 1 } else { -1 }
                } else if tr > r {
                    9
                } else {
                    -9
                };
                let mut square = source as i32 + step;
                let mut blockers = 0;
                while square != target as i32 {
                    blockers += usize::from(position.piece_at(square as usize).is_some());
                    square += step;
                }
                if piece.kind == PieceKind::Cannon && position.piece_at(target).is_some() {
                    blockers == 1
                } else {
                    blockers == 0
                }
            } else {
                false
            };
            if possible {
                targets |= 1u128 << target;
            }
        }
        targets
    }

    #[test]
    fn mobility_matches_independent_move_and_attack_queries_in_both_views() {
        let mut rng = SplitMix64::new(50);
        let mut seen = 0;
        for _ in 0..4 {
            let mut position = Position::startpos();
            for _ in 0..120 {
                let attacks = position.attacked_squares_masks();
                for perspective in [Color::Red, Color::Black] {
                    let mut expected = Vec::new();
                    for source in position.occupied_squares() {
                        let piece = position.piece_at(source).unwrap();
                        let kind = match piece.kind {
                            PieceKind::Rook => 0,
                            PieceKind::Horse => 1,
                            PieceKind::Cannon => 2,
                            _ => continue,
                        };
                        let targets = slow_targets(&position, source, piece);
                        assert_eq!(move_targets(&position, source, piece), targets);
                        let controlled = (0..BOARD_SIZE)
                            .filter(|&t| {
                                targets & (1 << t) != 0
                                    && position.is_square_attacked(t, piece.color.opposite())
                            })
                            .count() as u32;
                        let free = targets.count_ones() - controlled;
                        let owner = usize::from(piece.color != perspective);
                        let row = MOBILITY_OFFSET
                            + ((owner * 3 + kind) * 4 + bucket(free)) * 4
                            + bucket(controlled);
                        expected.push(MOBILITY_ENTRY_TAG | (source as u32) << 16 | row as u32);
                        seen += 1;
                    }
                    let mut actual = Vec::new();
                    visit_mobility_entries_from(&position, perspective, u128::MAX, attacks, |e| {
                        actual.push(e)
                    });
                    assert_eq!(actual, expected, "{} {perspective:?}", position.to_fen());
                    assert!(actual.iter().all(|e| {
                        (MOBILITY_OFFSET..MOBILITY_OFFSET + MOBILITY_VOCAB)
                            .contains(&(*e as u16 as usize))
                    }));
                    if let Some(&entry) = actual.first() {
                        let source = (entry >> 16) & 127;
                        let mut partial = Vec::new();
                        visit_mobility_entries_from(
                            &position,
                            perspective,
                            1u128 << source,
                            attacks,
                            |e| partial.push(e),
                        );
                        assert_eq!(partial, [entry]);
                    }
                }
                let moves = position.legal_moves();
                if moves.is_empty() {
                    break;
                }
                position.make_move(moves[rng.next_u64() as usize % moves.len()]);
            }
        }
        assert!(seen > 1000);
    }

    #[test]
    fn cannon_quiet_capture_and_blocked_horse_zero_mobility_are_distinct() {
        // 炮在e5，e6是炮架，e7为空，e8有敌车：可走e4，可吃e8，不能走e6/e7。
        let position =
            Position::from_canonical_piece_squares(&[(0, 85), (7, 4), (5, 40), (6, 31), (11, 13)]);
        let targets = move_targets(&position, 40, position.piece_at(40).unwrap());
        assert_ne!(targets & (1 << 49), 0);
        assert_ne!(targets & (1 << 13), 0);
        assert_eq!(targets & ((1 << 31) | (1 << 22) | (1 << 4)), 0);
        let blocked = Position::from_canonical_piece_squares(&[
            (0, 85),
            (7, 4),
            (3, 40),
            (6, 31),
            (6, 49),
            (6, 39),
            (6, 41),
        ]);
        assert_eq!(move_targets(&blocked, 40, blocked.piece_at(40).unwrap()), 0);
        let mut entries = Vec::new();
        visit_mobility_entries_from(
            &blocked,
            Color::Red,
            u128::MAX,
            blocked.attacked_squares_masks(),
            |e| entries.push(e),
        );
        assert_eq!(
            entries,
            [MOBILITY_ENTRY_TAG | (40 << 16) | (MOBILITY_OFFSET + 16) as u32]
        );
    }

    #[test]
    fn threat_only_runtime_uses_current_attacks_without_context_weights() {
        let position = Position::startpos();
        let mut model = AzNnue::random(32, 50);
        model.value_threat_output.fill(0.01);
        model.rebuild_threat();
        assert!(!model.context_active);
        let mut scratch = AzEvalScratch::new(model.arch);
        model.evaluate_with_scratch_output(
            &position,
            &position.legal_moves(),
            &[0.0; RULE_CONTEXT_SIZE],
            &mut scratch,
        );
        assert_eq!(scratch.attack_masks, position.attacked_squares_masks());
        let mut count = 0;
        let mut sum = [0.0; crate::az::VALUE_THREAT_RANK];
        crate::az::visit_value_threat_features(&position, Color::Red, |row| {
            count += 1;
            let start = row * crate::az::VALUE_THREAT_RANK;
            for (x, w) in sum
                .iter_mut()
                .zip(&model.threat_embedding[start..start + crate::az::VALUE_THREAT_RANK])
            {
                *x += w;
            }
        });
        for (x, y) in scratch.threat_accumulator.iter().zip(sum) {
            assert!((x - y / (count as f32).sqrt()).abs() < 1e-6);
        }
    }
}
