use crate::az::nnue::canonical_move;
use crate::xiangqi::{BOARD_SIZE, Color, Move, Position, color_index};

use super::{
    AzArenaReport, AzEvalAccumulator, AzEvalScratch, AzExperiencePool, AzNnue, AzNnueArch,
    AzSampleMeta, AzSearchLimits, AzStartSource, AzTrainingSample, DENSE_MOVE_SPACE,
    MateSearchLimits, POLICY_ACCUMULATOR_RANK, POLICY_CAPTURE_RELATION_OFFSET,
    POLICY_CAPTURE_RELATION_SIZE, POLICY_TACTICAL_SIZE, RULE_CONTEXT_SIZE, SplitMix64,
    VALUE_HEAD_SIZE, VALUE_KING_PIECE_VOCAB, VALUE_RAY_VOCAB, VALUE_THREAT_PAIR_VOCAB,
    VALUE_THREAT_VOCAB, WDL_HEAD_SIZE, alphazero_search_with_rules, check_context_features,
    dense_move_index, dense_move_squares, evaluate_policy_groups, move_map, policy_move_pieces,
    policy_move_tactical_flags, policy_tactical_indices, rule_context_features,
    scalar_value_to_wdl_target, search_root_mate, visit_value_king_piece_features,
    visit_value_threat_features,
};

fn geometric_move_index(piece: usize, ordinal: usize) -> usize {
    (0..DENSE_MOVE_SPACE)
        .filter(|&mv| super::policy_move_piece_index(mv, piece).is_some())
        .nth(ordinal)
        .unwrap()
}

// 下面这三个测试只在 slow-tests 下编译，它们独占的导入也一并门控，避免默认构建出现 unused import。
#[cfg(feature = "slow-tests")]
use super::{AzTrainLossWeights, train_samples, train_samples_weighted};

#[cfg(test)]
fn replay_pool_test_fixture() -> AzExperiencePool {
    fn sample(update: u32, game_id: u64, ply: u16) -> AzTrainingSample {
        AzTrainingSample {
            repetition_flags: Vec::new(),
            features: vec![1, 2, 3],
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            history_features: [0.0; super::HISTORY_CONTEXT_SIZE],
            move_indices: vec![0, 1],
            policy: vec![0.6, 0.4],
            value_wdl: scalar_value_to_wdl_target(0.1),
            root_search_wdl: scalar_value_to_wdl_target(0.1),
            value: 0.1,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            moves_left: 0.0,
            moves_left_weight: 0.0,
            search_simulations: 0,
            meta: AzSampleMeta {
                generation_update: update,
                game_id,
                ply,
                root_q: 0.11,
                best_q: 0.33,
                played_q: 0.02,
                best_visits: 88,
                played_visits: 13,
                best_index: 1,
                played_index: 0,
                start_source: AzStartSource::Startpos,
            },
        }
    }
    let mut pool = AzExperiencePool::new(100);
    pool.add_games(vec![
        vec![sample(7, 42, 9)],
        vec![sample(7, 43, 1), sample(7, 43, 2)],
    ]);
    pool
}

#[test]
fn policy_accumulator_cache_matches_direct_projection() {
    let model = AzNnue::random(32, 20260928);
    for feature in 0..super::POLICY_ACCUMULATOR_ROWS {
        let row = if feature == super::POLICY_ACCUMULATOR_BIAS_ROW {
            &model.hidden_bias[..]
        } else {
            &model.input_hidden[feature * model.hidden_size..(feature + 1) * model.hidden_size]
        };
        for channel in 0..POLICY_ACCUMULATOR_RANK {
            let projection = &model.policy_accumulator_hidden
                [channel * model.hidden_size..(channel + 1) * model.hidden_size];
            assert_eq!(
                model.policy_accumulator_features[feature * POLICY_ACCUMULATOR_RANK + channel]
                    .to_bits(),
                super::dot_product(row, projection).to_bits()
            );
        }
    }
}

#[test]
fn policy_accumulator_scores_current_position_without_move_deltas() {
    let position =
        Position::from_fen("1rbakab1r/9/4c3n/p3p3P/2p6/1C2c1pN1/P1P6/4B2C1/4A4/1RBAK3R w").unwrap();
    let moves = position.legal_moves();
    assert!(
        moves
            .iter()
            .any(|mv| position.piece_at(mv.to as usize).is_some())
    );
    assert!(
        moves
            .iter()
            .any(|mv| position.piece_at(mv.to as usize).is_none())
    );
    let mut model = AzNnue::random(32, 20261027);
    model.policy_accumulator_move.fill(0.03);
    let mut scratch = AzEvalScratch::new(model.arch);
    model.evaluate_with_scratch_output(&position, &moves, &[0.0; RULE_CONTEXT_SIZE], &mut scratch);
    assert!(scratch.logits[0].abs() > 1e-8);
    // 走法向量相同时，线性局面分支对吃子、非吃子和不同棋种必须给出同一分数。
    assert!(
        scratch
            .logits
            .iter()
            .all(|value| value.to_bits() == scratch.logits[0].to_bits())
    );
}

#[test]
fn capture_table_supplies_move_baseline_in_both_perspectives() {
    let position =
        Position::from_fen("1rbakab1r/9/4c3n/p3p3P/2p6/1C2c1pN1/P1P6/4B2C1/4A4/1RBAK3R w").unwrap();
    let mut black = position.clone();
    black.make_move(black.legal_moves()[0]);
    let mut model = AzNnue::random(96, 20261027);
    let mut scratch = AzEvalScratch::new(model.arch);
    let baseline = model.evaluate_with_scratch_output(
        &position,
        &position.legal_moves(),
        &[0.0; RULE_CONTEXT_SIZE],
        &mut scratch,
    );
    let score = |mv: usize| ((mv % 31) as f32 - 15.0) / 32.0;
    for mv in 0..DENSE_MOVE_SPACE {
        for victim in (7..14).map(Some).chain([None]) {
            if let Some(row) = super::policy_capture_compact_index(mv, victim) {
                model.policy_sparse_table[row] = score(mv);
            }
        }
    }
    let mut kinds = [false; 2];
    for position in [position, black] {
        let side = position.side_to_move();
        let moves = position.legal_moves();
        let output = model.evaluate_with_scratch_output(
            &position,
            &moves,
            &[0.0; RULE_CONTEXT_SIZE],
            &mut scratch,
        );
        assert_eq!(output.value_wdl, baseline.value_wdl);
        for (&mv, &logit) in moves.iter().zip(&scratch.logits) {
            kinds[usize::from(position.piece_at(mv.to as usize).is_some())] = true;
            assert_eq!(logit, score(dense_move_index(canonical_move(side, mv))));
        }
    }
    assert!(kinds.into_iter().all(|x| x));
}

#[test]
fn repetition_policy_metrics_separate_opportunities_and_model_mass() {
    let position = Position::startpos();
    let moves = position.legal_moves();
    let sample = AzTrainingSample {
        features: crate::az::nnue::extract_sparse_features_az(&position),
        rule_context: [0.0; RULE_CONTEXT_SIZE],
        history_features: [0.0; super::HISTORY_CONTEXT_SIZE],
        move_indices: moves
            .iter()
            .take(2)
            .map(|&mv| dense_move_index(mv))
            .collect(),
        repetition_flags: vec![1, 0],
        policy: vec![1.0, 0.0],
        value_wdl: [0.0, 1.0, 0.0],
        root_search_wdl: [0.0, 1.0, 0.0],
        value: 0.0,
        side_sign: 1.0,
        policy_weight: 1.0,
        value_weight: 1.0,
        moves_left: 0.0,
        moves_left_weight: 0.0,
        search_simulations: 1,
        meta: AzSampleMeta::default(),
    };
    let model = AzNnue::random(8, 19);
    let baseline = evaluate_policy_groups(&model, std::slice::from_ref(&sample));
    assert_eq!(baseline.repetition_samples, 1);
    assert_eq!(baseline.no_repetition_samples, 0);
    assert_eq!(baseline.repetition_target_mass, 1.0);
    let mut favored = model.clone();
    favored.policy_repetition_bias[0] = 5.0;
    let changed = evaluate_policy_groups(&favored, &[sample]);
    assert!(changed.repetition_predicted_mass > baseline.repetition_predicted_mass);
    assert!(changed.repetition_kl < baseline.repetition_kl);
}

#[test]
fn cannon_features_bind_screen_target_and_ray_state() {
    let position = Position::from_fen("4k4/9/9/9/9/9/9/4r4/4P4/3KC4 w - - 0 1").unwrap();
    let mut features = Vec::new();
    visit_value_threat_features(&position, Color::Red, |feature| features.push(feature));
    assert!(features.iter().any(|&feature| {
        (VALUE_THREAT_PAIR_VOCAB..VALUE_THREAT_PAIR_VOCAB + VALUE_RAY_VOCAB).contains(&feature)
    }));
    assert!(
        features
            .iter()
            .any(|&feature| feature >= VALUE_THREAT_PAIR_VOCAB + VALUE_RAY_VOCAB)
    );
    assert!(features.iter().all(|&feature| feature < VALUE_THREAT_VOCAB));
}

#[test]
fn value_king_piece_features_change_with_king_bucket() {
    let first = Position::from_canonical_piece_squares(&[(0, 85), (7, 4), (4, 54)]);
    let second = Position::from_canonical_piece_squares(&[(0, 86), (7, 4), (4, 54)]);
    let mut first_features = Vec::new();
    let mut second_features = Vec::new();
    visit_value_king_piece_features(&first, Color::Red, |feature| first_features.push(feature));
    visit_value_king_piece_features(&second, Color::Red, |feature| second_features.push(feature));
    assert_eq!(first_features.len(), 6);
    assert_eq!(second_features.len(), 6);
    let rook_first = first_features
        .iter()
        .copied()
        .filter(|&feature| {
            [(0, 7), (1, 7)].into_iter().any(|(side, bucket)| {
                super::feature_map::value_king_piece_index(side, bucket, 4, 54) == Some(feature)
            })
        })
        .collect::<Vec<_>>();
    let rook_second = second_features
        .iter()
        .copied()
        .filter(|&feature| {
            [(0, 8), (1, 7)].into_iter().any(|(side, bucket)| {
                super::feature_map::value_king_piece_index(side, bucket, 4, 54) == Some(feature)
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(rook_first.len(), 2);
    assert_ne!(rook_first, rook_second);
    assert!(
        first_features
            .iter()
            .all(|&feature| feature < VALUE_KING_PIECE_VOCAB)
    );
    assert!(
        second_features
            .iter()
            .all(|&feature| feature < VALUE_KING_PIECE_VOCAB)
    );
}

#[test]
fn capture_relation_changes_only_capture_policy_not_value() {
    let position =
        Position::from_fen("1rbakab1r/9/4c3n/p3p3P/2p6/1C2c1pN1/P1P6/4B2C1/4A4/1RBAK3R w").unwrap();
    let moves = position.legal_moves();
    let mut model = AzNnue::random(32, 20260928);
    model.policy_tactical[..POLICY_CAPTURE_RELATION_OFFSET].fill(0.125);
    model.rebuild_policy_tactical();
    let mut before = AzEvalScratch::new(model.arch);
    let old = model.evaluate_with_scratch_output(
        &position,
        &moves,
        &[0.0; RULE_CONTEXT_SIZE],
        &mut before,
    );
    model.policy_tactical[POLICY_CAPTURE_RELATION_OFFSET..].fill(0.25);
    model.rebuild_policy_tactical();
    let mut after = AzEvalScratch::new(model.arch);
    let new = model.evaluate_with_scratch_output(
        &position,
        &moves,
        &[0.0; RULE_CONTEXT_SIZE],
        &mut after,
    );
    assert_eq!(old.value_wdl, new.value_wdl);
    let mut captures = 0;
    let mut quiet = 0;
    for (index, mv) in moves.iter().enumerate() {
        let (_, victim) = policy_move_pieces(&position, position.side_to_move(), *mv).unwrap();
        if victim.is_some() {
            assert_eq!(
                after.logits[index].to_bits(),
                (before.logits[index] + 0.25).to_bits()
            );
            captures += 1;
        } else {
            assert_eq!(
                after.logits[index].to_bits(),
                before.logits[index].to_bits()
            );
            quiet += 1;
        }
    }
    assert!(captures > 0 && quiet > 0);
}

#[test]
fn capture_relation_is_shared_and_excludes_quiet_moves() {
    let quiet = policy_tactical_indices(0, 4, true, false, true, true, None, false);
    assert_eq!(quiet[1], POLICY_TACTICAL_SIZE);
    let pawn = policy_tactical_indices(0, 4, true, false, true, true, Some(13), false);
    let rook = policy_tactical_indices(0, 4, true, false, true, true, Some(11), false);
    let elsewhere = policy_tactical_indices(
        geometric_move_index(4, 1),
        4,
        true,
        false,
        true,
        true,
        Some(11),
        false,
    );
    assert_ne!(pawn[1], rook[1]);
    assert_eq!(rook[1], elsewhere[1]);
    let mut seen = std::collections::HashSet::new();
    for mover in 0..7 {
        for victim in 8..14 {
            for state in 0..16 {
                let signature = (state & 7) | 24 | ((state >> 3) << 5);
                let Some(row) = super::policy_capture_relation_index(mover, victim, signature)
                else {
                    assert!(mover < 3 && victim < 10);
                    continue;
                };
                assert!((POLICY_CAPTURE_RELATION_OFFSET..POLICY_TACTICAL_SIZE).contains(&row));
                assert!(seen.insert(row));
            }
        }
    }
    assert_eq!(seen.len(), POLICY_CAPTURE_RELATION_SIZE);
}

#[test]
fn tactical_policy_terms_distinguish_move_state() {
    let base = policy_tactical_indices(0, 4, true, true, false, true, Some(13), false);
    let changed = policy_tactical_indices(0, 4, false, true, false, true, Some(13), false);
    assert_ne!(base, changed);
    assert!(base.into_iter().all(|index| index < POLICY_TACTICAL_SIZE));
    assert!(
        changed
            .into_iter()
            .all(|index| index < POLICY_TACTICAL_SIZE)
    );
}
use std::fs;

#[test]
fn compact_policy_rows_are_unique_and_cover_both_tables() {
    let mut pairs = vec![false; super::POLICY_MOVE_PIECE_COUNT];
    let mut sparse = vec![false; super::POLICY_SPARSE_MAIN_SIZE];
    let mut tactical = vec![false; super::POLICY_TACTICAL_EXACT_SIZE];
    let mut captures = vec![false; super::POLICY_SPARSE_CAPTURE_SIZE];
    let mut counts = [0usize; 7];
    for mv in 0..DENSE_MOVE_SPACE {
        let (_, to) = dense_move_squares(mv).unwrap();
        for victim in (7..14).map(Some).chain([None]) {
            let row = super::policy_capture_compact_index(mv, victim);
            assert_eq!(
                row.is_some(),
                victim
                    .is_none_or(|p| p != 7 && super::nnue::canonical_piece_square_reachable(p, to))
            );
            if let Some(row) = row {
                let row = row - super::POLICY_SPARSE_MAIN_SIZE;
                assert!(!captures[row]);
                captures[row] = true;
            }
        }
        for piece in 0..7 {
            let Some(pair) = super::policy_move_piece_index(mv, piece) else {
                continue;
            };
            assert!(!pairs[pair], "重复的紧凑编号 {pair}");
            pairs[pair] = true;
            counts[piece] += 1;
            for us in 0..9 {
                for them in 0..9 {
                    let Some(row) = super::policy_main_compact_index(mv, piece, us, them) else {
                        continue;
                    };
                    assert_eq!(super::policy_sparse_main_index(mv, piece, us, them), row);
                    assert!(!sparse[row]);
                    sparse[row] = true;
                }
            }
            for signature in 0..64 {
                let Some(expected) = super::policy_tactical_exact_index(mv, piece, signature)
                else {
                    continue;
                };
                let row = policy_tactical_indices(
                    mv,
                    piece,
                    signature & 1 != 0,
                    signature & 2 != 0,
                    signature & 4 != 0,
                    signature & 8 != 0,
                    (signature & 16 != 0).then_some(13),
                    signature & 32 != 0,
                )[0];
                assert_eq!(row, expected);
                assert!(!tactical[row]);
                tactical[row] = true;
            }
        }
    }
    assert_eq!(counts, [24, 8, 16, 508, 1530, 1530, 126]);
    assert!(pairs.into_iter().all(|used| used));
    assert!(sparse.into_iter().all(|used| used));
    assert!(tactical.into_iter().all(|used| used));
    assert!(captures.into_iter().all(|used| used));
}

#[test]
fn compact_policy_preserves_legal_candidates_and_check_signatures() {
    let mut rng = SplitMix64::new(20261028);
    let mut checked = 0;
    for _ in 0..8 {
        let mut position = Position::startpos();
        for _ in 0..200 {
            let moves = position.legal_moves();
            if moves.is_empty() {
                break;
            }
            let side = position.side_to_move();
            let kings = super::canonical_buckets_for_perspective(&position, side);
            let masks = position.attacked_squares_masks();
            for &mv in &moves {
                let dense = dense_move_index(canonical_move(side, mv));
                let (piece, victim) = policy_move_pieces(&position, side, mv).unwrap();
                assert!(
                    super::policy_main_compact_index(dense, piece, kings.0, kings.1).is_some(),
                    "{} {mv:?}",
                    position.to_fen()
                );
                assert!(super::policy_capture_compact_index(dense, victim).is_some());
                let (sa, da, sd, dd) = policy_move_tactical_flags(
                    mv,
                    masks[color_index(side.opposite())],
                    masks[color_index(side)],
                );
                let indices = policy_tactical_indices(
                    dense,
                    piece,
                    sa,
                    da,
                    sd,
                    dd,
                    victim,
                    position.gives_check_after_move_fast(mv),
                );
                assert!(indices[0] < super::POLICY_TACTICAL_EXACT_SIZE);
                assert!(indices[1] <= POLICY_TACTICAL_SIZE);
                checked += 1;
            }
            position.make_move(moves[rng.next_u64() as usize % moves.len()]);
        }
    }
    assert!(checked > 20_000);
}

#[test]
fn compact_policy_rejects_impossible_piece_geometry() {
    let mapped = |uci: &str, piece| {
        let mv = Move::from_uci(uci).unwrap();
        let dense = move_map().sparse_to_dense[mv.from as usize * BOARD_SIZE + mv.to as usize];
        dense != u16::MAX && super::policy_move_piece_index(dense as usize, piece).is_some()
    };
    for (uci, piece) in [
        ("e0e1", 0),
        ("d0e1", 1),
        ("c0e2", 2),
        ("b0c2", 3),
        ("a0a9", 4),
        ("a0a9", 5),
        ("a3a4", 6),
        ("a5b5", 6),
    ] {
        assert!(mapped(uci, piece), "合法几何组合被丢弃: {uci}, {piece}");
    }
    for (uci, piece) in [
        ("a0a1", 0),
        ("e0e1", 1),
        ("g4e6", 2),
        ("a0a1", 3),
        ("b0c2", 4),
        ("b0c2", 5),
        ("a3b3", 6),
        ("b3b4", 6),
        ("a4a3", 6),
    ] {
        assert!(
            !mapped(uci, piece),
            "不可能的几何组合仍有参数行: {uci}, {piece}"
        );
    }
}

#[test]
fn px0_policy_indices_cover_both_sides_of_legal_selfplay() {
    assert_eq!(dense_move_index(Move::new(81, 72)), 0); // Px0 a0a1
    assert_eq!(dense_move_index(Move::new(81, 63)), 1); // Px0 a0a2
    assert_eq!(dense_move_index(Move::new(81, 82)), 9); // Px0 a0b0
    let mut position = Position::startpos();
    let mut rng = SplitMix64::new(17);
    for _ in 0..500 {
        let moves = position.legal_moves();
        if moves.is_empty() {
            position = Position::startpos();
            continue;
        }
        for &mv in &moves {
            let normalized = canonical_move(position.side_to_move(), mv);
            let index = dense_move_index(normalized);
            assert!(index < 2062);
            let piece = crate::az::nnue::piece_absolute_feature_index(
                position.side_to_move(),
                position.piece_at(mv.from as usize).unwrap(),
            );
            assert!(
                super::policy_move_piece_index(index, piece).is_some(),
                "合法走法没有紧凑策略行: {} {}",
                position.to_fen(),
                mv.to_uci()
            );
            assert_eq!(
                dense_move_squares(index),
                Some((normalized.from as usize, normalized.to as usize))
            );
        }
        position.make_move(moves[rng.next_u64() as usize % moves.len()]);
    }
}

#[test]
fn dense_move_space_matches_enumeration() {
    let map = move_map();
    assert_eq!(DENSE_MOVE_SPACE, 2062);
    for i in 0..DENSE_MOVE_SPACE {
        let sparse = map.dense_to_sparse[i] as usize;
        assert_eq!(map.sparse_to_dense[sparse], i as u16);
    }
}

#[test]
fn random_initial_value_head_is_neutral() {
    let model = AzNnue::random_with_arch(AzNnueArch::with_hidden_size(512), 20260409);
    assert!(model.value_head_output.iter().all(|&weight| weight == 0.0));

    let position = Position::startpos();
    let moves = position.legal_moves();
    let value = model.evaluate_value(&position, &moves);
    assert!(value.abs() < 1e-6, "initial startpos value={value}");
}

#[test]
fn material_bucket_selects_only_its_own_value_output() {
    let position = Position::startpos();
    let bucket = super::value_material_bucket(&position, Color::Red);
    assert_eq!(bucket, 23);
    let bare = Position::from_fen("4k4/9/9/9/9/9/9/9/9/3K5 w").unwrap();
    assert_eq!(super::value_material_bucket(&bare, Color::Red), 4);
    let rook = Position::from_fen("4k4/9/9/9/9/9/9/9/9/R2K5 w").unwrap();
    assert_eq!(super::value_material_bucket(&rook, Color::Red), 28);
    assert_eq!(super::value_material_bucket(&rook, Color::Black), 30);
    let mut model = AzNnue::random(16, 5);
    model.value_head_bias.fill(1.0);
    model.value_head_output[bucket * WDL_HEAD_SIZE * VALUE_HEAD_SIZE] = 1.0;
    let hidden = [0.0; 16];
    let king = [0.0; VALUE_HEAD_SIZE];
    let mut scratch = Vec::new();
    assert!(
        model
            .value_wdl_from_hidden_into(bucket, &hidden, &king, &mut scratch, [0.0; 3])
            .1
            > 0.0
    );
    assert_eq!(
        model
            .value_wdl_from_hidden_into(4, &hidden, &king, &mut scratch, [0.0; 3])
            .1,
        0.0
    );
}

#[cfg(feature = "profile")]
#[test]
#[ignore = "manual profile harness"]
fn manual_profile_policy_head() {
    let position = Position::startpos();
    let moves = position.legal_moves();
    let mut model = AzNnue::random(128, 999);
    model.policy_tactical[0] = 1.0e-6;
    model.value_threat_output[0] = 1.0e-6;
    model.rebuild_policy_tactical();
    model.rebuild_threat();
    let mut scratch = AzEvalScratch::new(model.arch);
    for _ in 0..2_000 {
        model.evaluate_with_scratch_output(
            &position,
            &moves,
            &[0.0; RULE_CONTEXT_SIZE],
            &mut scratch,
        );
    }
    crate::infra::profile::print_report();
}

#[test]
fn zero_initialized_policy_accumulator_is_neutral_and_trainable() {
    let position = Position::startpos();
    let moves = position.legal_moves();
    let model = AzNnue::random(32, 20260816);
    assert!(
        model
            .policy_accumulator_move
            .iter()
            .all(|&weight| weight == 0.0)
    );

    let mut baseline = AzEvalScratch::new(model.arch);
    model.evaluate_with_scratch_output(&position, &moves, &[0.0; RULE_CONTEXT_SIZE], &mut baseline);

    let mut active = model.clone();
    active.policy_accumulator_move.fill(0.01);
    active.rebuild_policy_cache();
    let mut changed = AzEvalScratch::new(model.arch);
    active.evaluate_with_scratch_output(&position, &moves, &[0.0; RULE_CONTEXT_SIZE], &mut changed);
    assert!(
        baseline
            .logits
            .iter()
            .zip(&changed.logits)
            .any(|(left, right)| left != right)
    );
}

#[test]
fn scalar_value_head_starts_neutral() {
    let model = AzNnue::random(16, 7);
    let mut scratch = AzEvalScratch::new(model.arch);
    let (_, value) = model.value_wdl_from_hidden_into(
        0,
        &scratch.hidden,
        &scratch.value_king_piece_accumulator,
        &mut scratch.value_head,
        [0.0; WDL_HEAD_SIZE],
    );

    assert!(value.abs() < 1e-6);
}

#[test]
fn incremental_accumulator_matches_full_refresh() {
    let model = AzNnue::random(128, 20260807);
    let mut position = Position::startpos();
    let mut hidden = AzEvalAccumulator::new(&model, &position).into_hidden_sum();
    for _ in 0..32 {
        let mv = position.legal_moves()[0];
        let moved = position.piece_at(mv.from as usize).unwrap();
        let captured = position.piece_at(mv.to as usize);
        position.make_move(mv);
        AzEvalAccumulator::apply_transition_to_hidden(&model, mv, moved, captured, &mut hidden);
        let refreshed = AzEvalAccumulator::new(&model, &position).into_hidden_sum();
        for (&incremental, &full) in hidden.iter().zip(&refreshed) {
            assert!((incremental - full).abs() < 2.0e-5);
        }
    }
}

#[test]
fn policy_accumulator_matches_full_refresh() {
    let model = AzNnue::random(128, 20260817);
    let mut position = Position::startpos();
    let mut accumulators = [
        model.policy_accumulator(&position, Color::Red),
        model.policy_accumulator(&position, Color::Black),
    ];
    for _ in 0..32 {
        let mv = position.legal_moves()[0];
        let moved = position.piece_at(mv.from as usize).unwrap();
        let captured = position.piece_at(mv.to as usize);
        position.make_move(mv);
        for perspective in [Color::Red, Color::Black] {
            model.apply_policy_transition(
                mv,
                moved,
                captured,
                perspective,
                &mut accumulators[color_index(perspective)],
            );
            let refreshed = model.policy_accumulator(&position, perspective);
            for (incremental, full) in accumulators[color_index(perspective)].iter().zip(refreshed)
            {
                assert!((incremental - full).abs() < 2.0e-5);
            }
        }
    }
}

#[test]
fn arena_report_relative_elo_tracks_score_and_bounds() {
    let stronger = AzArenaReport {
        wins: 6,
        losses: 3,
        draws: 1,
        ..AzArenaReport::default()
    };
    let weaker = AzArenaReport {
        wins: 3,
        losses: 6,
        draws: 1,
        ..AzArenaReport::default()
    };

    assert!(stronger.score_rate() > 0.5);
    assert!(stronger.elo_diff_vs_even() > 0.0);
    assert!(weaker.score_rate() < 0.5);
    assert!(weaker.elo_diff_vs_even() < 0.0);
    let (lower, upper) = stronger.elo_diff_bounds(1.96);
    assert!(lower <= stronger.elo_diff_vs_even());
    assert!(upper >= stronger.elo_diff_vs_even());
}

#[cfg(feature = "gpu-train")]
#[cfg(feature = "slow-tests")]
#[test]
fn value_head_can_overfit_tiny_fixed_dataset() {
    let mut model = AzNnue::random(16, 7);
    model.hidden_bias.fill(0.1);
    model.hidden_bias.fill(0.1);

    let samples = vec![
        AzTrainingSample {
            repetition_flags: Vec::new(),
            features: vec![0],
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            history_features: [0.0; super::HISTORY_CONTEXT_SIZE],
            move_indices: Vec::new(),
            policy: Vec::new(),
            value_wdl: scalar_value_to_wdl_target(1.0),
            root_search_wdl: scalar_value_to_wdl_target(1.0),
            value: 1.0,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            moves_left: 0.0,
            moves_left_weight: 0.0,
            search_simulations: 0,
            meta: AzSampleMeta::default(),
        },
        AzTrainingSample {
            repetition_flags: Vec::new(),
            features: vec![1],
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            history_features: [0.0; super::HISTORY_CONTEXT_SIZE],
            move_indices: Vec::new(),
            policy: Vec::new(),
            value_wdl: scalar_value_to_wdl_target(-1.0),
            root_search_wdl: scalar_value_to_wdl_target(-1.0),
            value: -1.0,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            moves_left: 0.0,
            moves_left_weight: 0.0,
            search_simulations: 0,
            meta: AzSampleMeta::default(),
        },
        AzTrainingSample {
            repetition_flags: Vec::new(),
            features: vec![2],
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            history_features: [0.0; super::HISTORY_CONTEXT_SIZE],
            move_indices: Vec::new(),
            policy: Vec::new(),
            value_wdl: scalar_value_to_wdl_target(0.75),
            root_search_wdl: scalar_value_to_wdl_target(0.75),
            value: 0.75,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            moves_left: 0.0,
            moves_left_weight: 0.0,
            search_simulations: 0,
            meta: AzSampleMeta::default(),
        },
        AzTrainingSample {
            repetition_flags: Vec::new(),
            features: vec![3],
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            history_features: [0.0; super::HISTORY_CONTEXT_SIZE],
            move_indices: Vec::new(),
            policy: Vec::new(),
            value_wdl: scalar_value_to_wdl_target(-0.75),
            root_search_wdl: scalar_value_to_wdl_target(-0.75),
            value: -0.75,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            moves_left: 0.0,
            moves_left_weight: 0.0,
            search_simulations: 0,
            meta: AzSampleMeta::default(),
        },
    ];

    let mut rng = SplitMix64::new(17);
    let before = train_samples(&mut model, &samples, 1, 0.003, 4, &mut rng)
        .unwrap()
        .value_loss;
    let after = train_samples(&mut model, &samples, 300, 0.003, 4, &mut rng)
        .unwrap()
        .value_loss;

    assert!(after < before * 0.5, "before={before} after={after}");
    assert!(after < 0.35, "after={after}");
}

#[cfg(feature = "gpu-train")]
#[cfg(feature = "slow-tests")]
#[test]
fn batched_training_is_deterministic() {
    let samples = vec![
        AzTrainingSample {
            repetition_flags: Vec::new(),
            features: vec![0, 4, 8],
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            history_features: [0.0; super::HISTORY_CONTEXT_SIZE],
            move_indices: Vec::new(),
            policy: Vec::new(),
            value_wdl: scalar_value_to_wdl_target(1.0),
            root_search_wdl: scalar_value_to_wdl_target(1.0),
            value: 1.0,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            moves_left: 0.0,
            moves_left_weight: 0.0,
            search_simulations: 0,
            meta: AzSampleMeta::default(),
        },
        AzTrainingSample {
            repetition_flags: Vec::new(),
            features: vec![1, 5, 9],
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            history_features: [0.0; super::HISTORY_CONTEXT_SIZE],
            move_indices: Vec::new(),
            policy: Vec::new(),
            value_wdl: scalar_value_to_wdl_target(-1.0),
            root_search_wdl: scalar_value_to_wdl_target(-1.0),
            value: -1.0,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            moves_left: 0.0,
            moves_left_weight: 0.0,
            search_simulations: 0,
            meta: AzSampleMeta::default(),
        },
        AzTrainingSample {
            repetition_flags: Vec::new(),
            features: vec![2, 6, 10],
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            history_features: [0.0; super::HISTORY_CONTEXT_SIZE],
            move_indices: Vec::new(),
            policy: Vec::new(),
            value_wdl: scalar_value_to_wdl_target(0.5),
            root_search_wdl: scalar_value_to_wdl_target(0.5),
            value: 0.5,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            moves_left: 0.0,
            moves_left_weight: 0.0,
            search_simulations: 0,
            meta: AzSampleMeta::default(),
        },
        AzTrainingSample {
            repetition_flags: Vec::new(),
            features: vec![3, 7, 11],
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            history_features: [0.0; super::HISTORY_CONTEXT_SIZE],
            move_indices: Vec::new(),
            policy: Vec::new(),
            value_wdl: scalar_value_to_wdl_target(-0.5),
            root_search_wdl: scalar_value_to_wdl_target(-0.5),
            value: -0.5,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            moves_left: 0.0,
            moves_left_weight: 0.0,
            search_simulations: 0,
            meta: AzSampleMeta::default(),
        },
    ];
    let mut single = AzNnue::random(16, 23);
    single.hidden_bias.fill(0.1);
    single.hidden_bias.fill(0.1);
    let mut repeated = single.clone();

    let mut rng_single = SplitMix64::new(99);
    let mut rng_repeated = SplitMix64::new(99);
    let single_stats = train_samples(&mut single, &samples, 5, 0.003, 4, &mut rng_single).unwrap();
    let repeated_stats =
        train_samples(&mut repeated, &samples, 5, 0.003, 4, &mut rng_repeated).unwrap();

    assert!((single_stats.loss - repeated_stats.loss).abs() < 1e-5);
    assert!((single_stats.value_loss - repeated_stats.value_loss).abs() < 1e-5);
    assert!((single_stats.value_pred_sum - repeated_stats.value_pred_sum).abs() < 1e-4);
    assert!((single_stats.value_target_sum - repeated_stats.value_target_sum).abs() < 1e-6);
    assert!(
        single
            .value_head_output
            .iter()
            .zip(&repeated.value_head_output)
            .all(|(left, right)| (*left - *right).abs() < 1e-5)
    );
}

#[cfg(feature = "gpu-train")]
#[cfg(feature = "slow-tests")]
#[test]
fn value_only_training_updates_trunk_when_trunk_training_enabled() {
    let samples = vec![
        AzTrainingSample {
            repetition_flags: Vec::new(),
            features: vec![0, 4, 8],
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            history_features: [0.0; super::HISTORY_CONTEXT_SIZE],
            move_indices: Vec::new(),
            policy: Vec::new(),
            value_wdl: scalar_value_to_wdl_target(1.0),
            root_search_wdl: scalar_value_to_wdl_target(1.0),
            value: 1.0,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            moves_left: 0.0,
            moves_left_weight: 0.0,
            search_simulations: 0,
            meta: AzSampleMeta::default(),
        },
        AzTrainingSample {
            repetition_flags: Vec::new(),
            features: vec![1, 5, 9],
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            history_features: [0.0; super::HISTORY_CONTEXT_SIZE],
            move_indices: Vec::new(),
            policy: Vec::new(),
            value_wdl: scalar_value_to_wdl_target(-1.0),
            root_search_wdl: scalar_value_to_wdl_target(-1.0),
            value: -1.0,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            moves_left: 0.0,
            moves_left_weight: 0.0,
            search_simulations: 0,
            meta: AzSampleMeta::default(),
        },
        AzTrainingSample {
            repetition_flags: Vec::new(),
            features: vec![2, 6, 10],
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            history_features: [0.0; super::HISTORY_CONTEXT_SIZE],
            move_indices: Vec::new(),
            policy: Vec::new(),
            value_wdl: scalar_value_to_wdl_target(0.75),
            root_search_wdl: scalar_value_to_wdl_target(0.75),
            value: 0.75,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            moves_left: 0.0,
            moves_left_weight: 0.0,
            search_simulations: 0,
            meta: AzSampleMeta::default(),
        },
        AzTrainingSample {
            repetition_flags: Vec::new(),
            features: vec![3, 7, 11],
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            history_features: [0.0; super::HISTORY_CONTEXT_SIZE],
            move_indices: Vec::new(),
            policy: Vec::new(),
            value_wdl: scalar_value_to_wdl_target(-0.75),
            root_search_wdl: scalar_value_to_wdl_target(-0.75),
            value: -0.75,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            moves_left: 0.0,
            moves_left_weight: 0.0,
            search_simulations: 0,
            meta: AzSampleMeta::default(),
        },
    ];
    let mut model = AzNnue::random(8, 31);
    model.hidden_bias.fill(0.1);
    let before_input = model.input_hidden.clone();
    let before_bias = model.hidden_bias.clone();

    let mut rng = SplitMix64::new(32);
    let weights = AzTrainLossWeights {
        value: 1.0,
        policy: 0.0,
    };
    train_samples_weighted(&mut model, &samples, 20, 0.01, 4, &mut rng, weights).unwrap();

    let input_changed = before_input
        .iter()
        .zip(&model.input_hidden)
        .any(|(left, right)| (*left - *right).abs() > 1e-7);
    let bias_changed = before_bias
        .iter()
        .zip(&model.hidden_bias)
        .any(|(left, right)| (*left - *right).abs() > 1e-7);
    assert!(
        input_changed || bias_changed,
        "value-only training should update trunk"
    );
}

#[test]
fn aznnue_safetensors_roundtrip_matches_weights() {
    let mut model = AzNnue::random(16, 42);
    model.moves_left_output.fill(0.03);
    model.moves_left_bias[0] = 0.2;
    model.shared_bias.fill(0.1);
    model.rebuild_moves_left();
    let path = std::env::temp_dir().join("chineseai_test_aznnue_roundtrip.safetensors");
    let _ = fs::remove_file(&path);
    model.save(&path).unwrap();
    let loaded = AzNnue::load(&path).unwrap();
    let _ = fs::remove_file(&path);
    assert_eq!(model.hidden_size, loaded.hidden_size);
    assert_eq!(model.input_hidden, loaded.input_hidden);
    assert_eq!(model.hidden_bias, loaded.hidden_bias);
    assert_eq!(model.shared_hidden, loaded.shared_hidden);
    assert_eq!(model.shared_bias, loaded.shared_bias);
    assert_eq!(model.value_head_hidden, loaded.value_head_hidden);
    assert_eq!(model.value_head_bias, loaded.value_head_bias);
    assert_eq!(model.value_head_output, loaded.value_head_output);
    assert_eq!(
        model.value_king_piece_hidden,
        loaded.value_king_piece_hidden
    );
    assert_eq!(
        model.value_king_piece_projection,
        loaded.value_king_piece_projection
    );
    assert_eq!(model.moves_left_output, loaded.moves_left_output);
    assert_eq!(model.moves_left_bias, loaded.moves_left_bias);
    assert!(loaded.moves_left_active);
    assert_eq!(model.policy_context_hidden, loaded.policy_context_hidden);
    assert_eq!(model.policy_move_context, loaded.policy_move_context);
    assert_eq!(
        model.policy_accumulator_hidden,
        loaded.policy_accumulator_hidden
    );
    assert_eq!(
        model.policy_accumulator_move,
        loaded.policy_accumulator_move
    );
}

#[test]
fn replay_pool_lz4_snapshot_roundtrip() {
    let path = std::env::temp_dir().join("chineseai_test_replay_roundtrip.replay.lz4");
    let _ = fs::remove_file(&path);
    let pool = replay_pool_test_fixture();
    pool.save_snapshot_lz4(&path).unwrap();
    let file_blob = fs::read(&path).unwrap();
    assert_eq!(&file_blob[0..4], b"AZRP");
    assert_eq!(&file_blob[8..12], b"CHNK");
    let loaded = AzExperiencePool::load_snapshot_lz4(&path, 100).unwrap();
    let _ = fs::remove_file(&path);
    assert_eq!(loaded.sample_count(), pool.sample_count());
    assert_eq!(loaded.capacity(), pool.capacity());
    let loaded_samples = loaded.all_samples();
    assert_eq!(loaded_samples[0].meta.generation_update, 7);
    assert_eq!(loaded_samples[0].meta.game_id, 42);
    assert_eq!(loaded_samples[0].meta.ply, 9);
    assert!((loaded_samples[0].meta.best_q - 0.33).abs() < 1e-6);
    assert_eq!(loaded_samples[0].meta.played_visits, 13);
    assert_eq!(
        loaded_samples[0].root_search_wdl,
        pool.all_samples()[0].root_search_wdl
    );
}

#[test]
fn replay_pool_prunes_whole_game_chunks() {
    fn sample(update: u32, game_id: u64, ply: u16) -> AzTrainingSample {
        AzTrainingSample {
            repetition_flags: Vec::new(),
            features: vec![1],
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            history_features: [0.0; super::HISTORY_CONTEXT_SIZE],
            move_indices: vec![0],
            policy: vec![1.0],
            value_wdl: scalar_value_to_wdl_target(0.0),
            root_search_wdl: scalar_value_to_wdl_target(0.0),
            value: 0.0,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            moves_left: 0.0,
            moves_left_weight: 0.0,
            search_simulations: 0,
            meta: AzSampleMeta {
                generation_update: update,
                game_id,
                ply,
                ..AzSampleMeta::default()
            },
        }
    }

    let mut pool = AzExperiencePool::new(4);
    pool.add_games(vec![
        vec![sample(1, 1, 0), sample(1, 1, 1)],
        vec![sample(2, 2, 0), sample(2, 2, 1)],
        vec![sample(3, 3, 0), sample(3, 3, 1)],
    ]);

    let stats = pool.window_stats(1);
    assert_eq!(pool.sample_count(), 4);
    assert_eq!(stats.chunks, 2);
    assert_eq!(stats.oldest_generation_update, 2);
    assert_eq!(stats.newest_generation_update, 3);
    assert_eq!(stats.window_games, 2);
    assert!((stats.recent_window_sample_fraction - 0.5).abs() < 1e-6);
    assert_eq!(pool.all_sample_groups().len(), 2);
}

#[test]
fn replay_pool_mixed_recent_sampling_uses_requested_recent_fraction() {
    fn sample(update: u32, game_id: u64, ply: u16) -> AzTrainingSample {
        AzTrainingSample {
            repetition_flags: Vec::new(),
            features: vec![1],
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            history_features: [0.0; super::HISTORY_CONTEXT_SIZE],
            move_indices: vec![0],
            policy: vec![1.0],
            value_wdl: scalar_value_to_wdl_target(0.0),
            root_search_wdl: scalar_value_to_wdl_target(0.0),
            value: 0.0,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            moves_left: 0.0,
            moves_left_weight: 0.0,
            search_simulations: 0,
            meta: AzSampleMeta {
                generation_update: update,
                game_id,
                ply,
                ..AzSampleMeta::default()
            },
        }
    }

    let mut pool = AzExperiencePool::new(12);
    for update in 1..=4 {
        pool.add_games(vec![vec![
            sample(update, update as u64, 0),
            sample(update, update as u64, 1),
            sample(update, update as u64, 2),
        ]]);
    }
    let mut rng = SplitMix64::new(123);
    let batch = pool.sample_mixed_recent(10, 0.4, 2, &mut rng);

    assert_eq!(batch.samples.len(), 10);
    assert_eq!(batch.recent_samples, 4);
    assert_eq!(batch.full_window_samples, 6);
    assert!((4..=10).contains(&batch.actual_recent_samples));
}

#[test]
fn replay_recent_games_counts_complete_games_not_generation_batches() {
    fn sample(game_id: u64) -> AzTrainingSample {
        AzTrainingSample {
            repetition_flags: Vec::new(),
            features: vec![1],
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            history_features: [0.0; super::HISTORY_CONTEXT_SIZE],
            move_indices: vec![0],
            policy: vec![1.0],
            value_wdl: scalar_value_to_wdl_target(0.0),
            root_search_wdl: scalar_value_to_wdl_target(0.0),
            value: 0.0,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            moves_left: 0.0,
            moves_left_weight: 0.0,
            search_simulations: 0,
            meta: AzSampleMeta {
                generation_update: 7,
                game_id,
                ..AzSampleMeta::default()
            },
        }
    }

    let mut pool = AzExperiencePool::new(8);
    for game_id in 1..=4 {
        pool.add_games(vec![vec![sample(game_id)]]);
    }
    let stats = pool.window_stats(2);
    assert_eq!(stats.window_games, 4);
    assert!((stats.recent_window_sample_fraction - 0.5).abs() < 1e-6);

    let mut rng = SplitMix64::new(9);
    let batch = pool.sample_mixed_recent(100, 1.0, 2, &mut rng);
    assert!(batch.samples.iter().all(|sample| sample.meta.game_id >= 3));
    assert_eq!(batch.actual_recent_samples, 100);
}

#[test]
fn rule_context_exposes_repetition_without_history_planes() {
    let position = Position::startpos();
    let entry = position.rule_history_entry(None);
    let context = rule_context_features(&position, &[entry, entry, entry]);

    assert!((context[1] - 2.0 / 3.0).abs() < 1e-6);
    // cycle 从"最后一次命中之后"开始，因此这里只有 1 个 entry；这份 cycle 里没有任何
    // 真实着法（全是锚点），但它仍必须计入 [2] —— 空 cycle 优化不能把它一起跳过。
    assert_eq!(context[2], 1.0 / 32.0);
    assert_eq!(context[3..], [0.0; 4]);
}

/// 没有重复局面时 `rule_context_features` 的 cycle 分支必须完全不贡献任何分量。
///
/// 这条正对着 `cycle.is_empty()` 的短路：跳过的只是一次白克隆，7 个分量一个都不许变。
#[test]
fn rule_context_is_cycle_free_without_repetition() {
    let mut position = Position::startpos();
    let mut history = position.initial_rule_history();
    for notation in ["b0c2", "b9c7", "a0b0", "a9b9"] {
        let mv = position.parse_uci_move(notation).unwrap();
        let mover = position.side_to_move();
        let captured = position.piece_at(mv.to as usize);
        position.make_move(mv);
        history.push(position.rule_history_entry_after_moved(mover, mv, captured));
    }
    // 前提：历史里确实没有任何重复局面，否则这条测试测不到空 cycle 分支。
    for (index, entry) in history.iter().enumerate() {
        for old in &history[..index] {
            assert!(
                entry.hash != old.hash || entry.side_to_move != old.side_to_move,
                "unexpected repetition at history index {index}"
            );
        }
    }

    let context = rule_context_features(&position, &history);
    assert_eq!(context[1], 0.0, "no prior match => [1] must be zero");
    assert_eq!(context[2], 0.0, "empty cycle => [2] must be zero");
    assert_eq!(
        context[3..],
        [0.0; 4],
        "empty cycle => no check/chase counts"
    );
    assert_eq!(
        context[0],
        position.rule60_count_with_history(&history) as f32 / 120.0,
        "[0] must come from the whole history, not from the cycle"
    );
}

/// `evaluate_wdl_with_rules` 是公开 API，但此前全仓没有任何调用者、也没有测试。
/// 它必须（1）返回归一化的 WDL 分布，（2）真的把 rule_context 喂进主干。
#[test]
fn evaluate_wdl_with_rules_returns_normalized_distribution() {
    let model = AzNnue::random(16, 97);
    // 必须用 60 回合计数非零的局面：startpos 的 7 个分量全是 0，而
    // `add_rule_context_to_hidden` 会跳过零，那样根本测不出 context 有没有进主干。
    let position = Position::from_fen(
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 30 1",
    )
    .unwrap();
    let history = position.initial_rule_history();
    let moves = position.legal_moves();
    assert!(!moves.is_empty());
    assert!(
        rule_context_features(&position, &history)
            .iter()
            .any(|value| *value != 0.0)
    );

    let wdl = model.evaluate_wdl_with_rules(&position, &history, &moves);
    assert!(wdl.iter().all(|value| value.is_finite() && *value >= 0.0));
    assert!((wdl.iter().sum::<f32>() - 1.0).abs() < 1e-4, "{wdl:?}");

    // 随机初始化的 `value_head_output` 恒为 0（刻意保持价值中性），此时 softmax 恒为均匀
    // 分布、任何输入变化都观察不到。先把它打开，再比较"只差 context_hidden"的两个
    // 模型：唯一变量就是 context 有没有进主干。
    let mut base = model.clone();
    for (index, weight) in base.value_head_output.iter_mut().enumerate() {
        *weight = ((index % 11) as f32 + 1.0) * 1.0e-2;
    }
    let without_context = base.evaluate_wdl_with_rules(&position, &history, &moves);

    let mut with_context = base.clone();
    let rule_end = RULE_CONTEXT_SIZE * 2 * with_context.hidden_size;
    for (index, weight) in with_context.context_hidden[..rule_end]
        .iter_mut()
        .enumerate()
    {
        *weight = ((index % 17) as f32 + 1.0) * 1.0e-3;
    }
    with_context.rebuild_context();
    let shifted = with_context.evaluate_wdl_with_rules(&position, &history, &moves);
    assert!(
        shifted
            .iter()
            .zip(without_context.iter())
            .any(|(left, right)| (left - right).abs() > 1e-6),
        "rule_context did not reach the value head: {shifted:?} vs {without_context:?}"
    );
}

/// 连杀证明搜索：mate-in-1 必须能证，且必须**不能**在 7 手内证明一个 mate-in-8 的局面
/// （与 Pikafish `go mate 7` → 仍报 mate 8 的 oracle 一致）。
#[test]
fn mate_search_proves_mate_in_one_and_respects_ply_limit() {
    // 黑车 a9→d9 之后：红将 d2 的 d3 在九宫外、c2 在九宫外，只剩 d1（仍在 d 线上）与 e2
    // （被 e9 黑将的飞将封住），且红方没有别的子可以垫将或吃车 ⇒ 将死。
    let position = Position::from_fen("r3k4/9/9/9/9/9/9/3K5/9/9 b - - 0 1").unwrap();
    let history = position.initial_rule_history();
    let solution = search_root_mate(
        &position,
        &history,
        MateSearchLimits {
            max_plies: 1,
            max_nodes: 10_000,
        },
    )
    .expect("mate in 1 must be proven");
    assert_eq!(solution.mv.to_uci(), "a9d9");
    assert_eq!(solution.plies, 1);
    // 0 半回合（关闭）必须什么都不做。
    assert!(
        search_root_mate(&position, &history, MateSearchLimits::OFF).is_none(),
        "MateSearchLimits::OFF must disable the search"
    );
}

/// oracle 局面：Pikafish `go mate 8` → 黑方强制连杀、首着 h2h0、共 15 半回合，
/// 且 `go mate 7` 21 秒仍只报 mate 8（7 手内无杀）。证明树只有 24 条终端线。
///
/// MCTS 靠访问分配撞不出这条杀（实测 64–4096 sims 全走 h5f4，h2h0 从未超过 4 次访问；
/// 把全部 8192 次压到 h2h0 上才浮现 +0.97），而这套受限证明搜索应该直接证出来。
#[test]
fn mate_search_proves_mate_in_eight() {
    let position =
        Position::from_fen("2bakab2/9/5r1c1/p1PRC1p2/4P2nP/6P2/4N1r2/7c1/4A4/2BAK1B1R b - - 0 1")
            .unwrap();
    let history = position.initial_rule_history();
    // 实测：这一条杀要 58,178 个节点才证出来（无置换表、无着法排序），所以预算给 20 万。
    let limits = MateSearchLimits {
        max_plies: 15,
        max_nodes: 200_000,
    };
    let solution = search_root_mate(&position, &history, limits).expect("mate in 8 must be proven");
    assert_eq!(solution.plies, 15, "oracle 说最快 8 手（15 半回合）");
    assert_eq!(solution.mv.to_uci(), "h2h0", "oracle 首着");
    assert!(solution.nodes <= limits.max_nodes);
    // 节点预算必须是真的兜底，而不是摆设：有限预算下宁可证不出来。
    assert!(
        search_root_mate(
            &position,
            &history,
            MateSearchLimits {
                max_plies: 15,
                max_nodes: 1,
            }
        )
        .is_none(),
        "tiny node budget must not prove anything"
    );
    // 7 手（13 半回合）内无杀。
    assert!(
        search_root_mate(
            &position,
            &history,
            MateSearchLimits {
                max_plies: 13,
                max_nodes: 50_000,
            }
        )
        .is_none(),
        "oracle: 7 手内无杀"
    );
}

/// 证明结果必须经由**既有**机制喂进搜索结果：杀着拿到 `solved = +1`、策略目标变成它上面
/// 的一热分布、价值目标变成必胜。全程不需要 checkpoint（连杀搜索不走网络）。
#[test]
fn root_mate_proof_drives_search_result() {
    let position =
        Position::from_fen("2bakab2/9/5r1c1/p1PRC1p2/4P2nP/6P2/4N1r2/7c1/4A4/2BAK1B1R b - - 0 1")
            .unwrap();
    let mut model = AzNnue::random(16, 3);
    model.mate_search_plies = 15;

    let provable = alphazero_search_with_rules(
        &position,
        None,
        None,
        &model,
        AzSearchLimits {
            simulations: 64,
            ..Default::default()
        },
    );
    assert_eq!(
        provable.best_move.map(|mv| mv.to_uci()),
        Some("h2h0".to_owned())
    );
    assert_eq!(provable.value_wdl, [1.0, 0.0, 0.0]);
    let mate = provable
        .candidates
        .iter()
        .find(|candidate| candidate.mv.to_uci() == "h2h0")
        .expect("mate move must be among the candidates");
    assert_eq!(mate.solved, Some(1));
    assert!((mate.policy - 1.0).abs() < 1e-6, "policy={}", mate.policy);

    // 关掉之后同一局面（同一随机权重）必须回到普通 MCTS：证明不出来，就没有任何子节点
    // 带 solved，价值也不再是必胜。
    model.mate_search_plies = 0;
    let plain = alphazero_search_with_rules(
        &position,
        None,
        None,
        &model,
        AzSearchLimits {
            simulations: 64,
            ..Default::default()
        },
    );
    assert!(plain.candidates.iter().all(|c| c.solved.is_none()));
    assert_ne!(plain.value_wdl, [1.0, 0.0, 0.0]);
}

/// 标量块的内容：startpos 上必须全是"中性"值，mate-in-1 局面必须直接暴露杀势。
#[test]
fn check_context_features_describe_check_and_mate_net() {
    let flags = |position: &Position| {
        position
            .legal_moves()
            .iter()
            .map(|&mv| f32::from(position.gives_check_after_move_fast(mv)))
            .collect::<Vec<_>>()
    };

    // startpos：不被将军、没有将军着法、双方将都还有 4 个安全逃格、九宫没被攻击。
    let startpos = Position::startpos();
    let moves = startpos.legal_moves();
    let context = check_context_features(
        &startpos,
        &moves,
        &flags(&startpos),
        startpos.attacked_squares_masks(),
    );
    assert_eq!(context[0], 0.0, "startpos 不应被将军");
    assert_eq!(context[1], 0.0, "startpos 没有将军着法");
    // 双方将的 d/f 逃格都被自己的士占着，只剩 e 线那一格；而 e3 有红兵挡线，
    // 所以两边都**不**构成飞将 ⇒ 各恰好 1 个安全逃格。
    assert_eq!(context[2], 0.25, "黑将只剩 1 个安全逃格");
    assert_eq!(context[3], 0.25, "红将只剩 1 个安全逃格");
    assert_eq!(context[4], 0.0, "没打到对方九宫");
    assert_eq!(context[5], 0.0, "自己九宫也没被打");
    assert_eq!(context[7], 0.0, "没有将军着法就没有杀势");
    assert!(
        (context[6] - 44.0 / 64.0).abs() < 1e-6,
        "startpos 44 个合法着法"
    );

    // mate-in-1（黑车 a9d9 杀）：黑方有 1 个将军着法，红将只剩 1 个安全逃格（d1；
    // e2 被飞将封住、c2 出九宫、d3 出九宫），于是"杀势"标志直接立起来。
    let mate = Position::from_fen("r3k4/9/9/9/9/9/9/3K5/9/9 b - - 0 1").unwrap();
    let mate_moves = mate.legal_moves();
    let mate_context = check_context_features(
        &mate,
        &mate_moves,
        &flags(&mate),
        mate.attacked_squares_masks(),
    );
    assert_eq!(mate_context[0], 0.0, "走子方自己没被将军");
    // 黑方有 2 个将军着法：车 a9d9 沿 d 线，以及将 e9d9 走成飞将。
    assert_eq!(mate_context[1], 0.5, "2 个将军着法 / 4");
    // 红将 d2 只有 d1 一个安全逃格：c2/d3 出九宫、e2 会被黑将飞将封住。
    assert_eq!(mate_context[2], 0.25, "红将只剩 1 个安全逃格 / 4");
    assert_eq!(
        mate_context[7], 1.0,
        "有将军着法 + 对方将几乎无处可逃 = 杀势"
    );

    // 被将军的一侧：把黑车摆到 d9（d 线全空），红将 d2 就被将军。
    let in_check = Position::from_fen("3rk4/9/9/9/9/9/9/3K5/9/9 w - - 0 1").unwrap();
    let checked_moves = in_check.legal_moves();
    let checked_context = check_context_features(
        &in_check,
        &checked_moves,
        &flags(&in_check),
        in_check.attacked_squares_masks(),
    );
    assert_eq!(checked_context[0], 1.0, "红方被黑车 d9 沿 d 线将军");
}

/// 标量块必须真的进到主干：只差 `context_hidden` 的两个模型，价值头的输出必须不同。
#[test]
fn check_context_block_reaches_the_trunk_when_active() {
    let position = Position::startpos();
    let history = position.initial_rule_history();
    let moves = position.legal_moves();

    let mut base = AzNnue::random(16, 71);
    // 随机初始化时 `value_head_output` 恒为 0（价值刻意中性），softmax 恒为均匀分布，
    // 任何主干变化都观察不到；先把它打开。
    for (index, weight) in base.value_head_output.iter_mut().enumerate() {
        *weight = ((index % 11) as f32 + 1.0) * 1.0e-2;
    }
    // 全零 ⇒ 未激活 ⇒ 评估路径整段跳过它（新模型初始也应保持中性）。
    assert!(!base.context_active);
    let without_block = base.evaluate_wdl_with_rules(&position, &history, &moves);

    let mut active = base.clone();
    let check_start = super::CHECK_CONTEXT_OFFSET * 2 * active.hidden_size;
    let check_end = super::HISTORY_CONTEXT_OFFSET * 2 * active.hidden_size;
    for (index, weight) in active.context_hidden[check_start..check_end]
        .iter_mut()
        .enumerate()
    {
        *weight = ((index % 17) as f32 + 1.0) * 1.0e-3;
    }
    active.rebuild_context();
    assert!(active.context_active);
    let with_block = active.evaluate_wdl_with_rules(&position, &history, &moves);

    assert!(
        with_block
            .iter()
            .zip(without_block.iter())
            .any(|(left, right)| (left - right).abs() > 1e-6),
        "check context 没有进到主干：{with_block:?} vs {without_block:?}"
    );
}

/// 模型仅接受当前格式及完整的张量集合。
#[test]
fn model_rejects_old_format_and_missing_tensors() {
    let model = AzNnue::random(16, 73);
    let path = std::env::temp_dir().join("chineseai_test_strict_model.safetensors");
    model.save(&path).unwrap();
    let tensors = candle_core::safetensors::load(&path, &candle_core::Device::Cpu).unwrap();
    assert_eq!(tensors["input_hidden"].dims(), &[692, 16]);
    assert_eq!(tensors["context_hidden"].dims(), &[111, 32]);
    assert!(!tensors.contains_key("rule_context_hidden"));
    assert!(!tensors.contains_key("check_context_hidden"));
    assert!(!tensors.contains_key("value_history_output"));
    assert_eq!(tensors["policy_tactical"].elem_count(), 120_704);
    assert_eq!(tensors["policy_sparse_table"].elem_count(), 289_403);
    assert_eq!(tensors["threat_embedding"].dims(), &[61_451, 64]);
    assert!(!tensors.contains_key("policy_move_bias"));
    for name in [
        "context_hidden",
        "moves_left_output",
        "moves_left_bias",
        "shared_hidden",
        "shared_bias",
    ] {
        let mut missing = tensors.clone();
        missing.remove(name);
        candle_core::safetensors::save(&missing, &path).unwrap();
        assert!(AzNnue::load(&path).is_err(), "missing {name}");
    }
    let mut old = tensors;
    old.insert(
        "az_model_format_version".into(),
        candle_core::Tensor::new(&[47.0f32], &candle_core::Device::Cpu).unwrap(),
    );
    candle_core::safetensors::save(&old, &path).unwrap();
    assert!(AzNnue::load(&path).is_err());
    fs::remove_file(&path).unwrap();
}

/// 策略头的 `destination_attacked/defended` 用的是**走前**攻击位板，而不是走完之后的
/// 真值——这是刻意保留的近似：精确值要按每个候选走法各查两次，实测 −25%~−39% NPS。
///
/// 这条测试做两件事：(1) 用一个炮架局面钉住近似到底差在哪、差多少；
/// (2) 在随机对局里统计"近似与真值不一致"的比例，给这个取舍一个可复查的数字。
/// `Position::is_square_attacked_after_move` 就是这里的对照口径——它是环境层唯一
/// 需要保留精确查询的地方（审计/测试/将来的"落子悬不悬"检测），生产路径不调用它。
#[test]
fn tactical_flags_are_pre_move_and_the_gap_is_audited() {
    const RED_GENERAL: usize = 0;
    const RED_ROOK: usize = 4;
    const BLACK_GENERAL: usize = 7;
    const BLACK_CANNON: usize = 12;
    let square = |name: &str| crate::xiangqi::parse_square(name).unwrap();
    let position = Position::from_canonical_piece_squares(&[
        (BLACK_CANNON, square("a9")),
        (RED_ROOK, square("a7")),
        (RED_GENERAL, square("d0")),
        (BLACK_GENERAL, square("e9")),
    ]);
    let mv = Move::new(square("a7"), square("a5"));
    let side = position.side_to_move();
    let masks = position.attacked_squares_masks();
    // (source_attacked, destination_attacked, source_defended, destination_defended)
    let approximate = policy_move_tactical_flags(
        mv,
        masks[color_index(side.opposite())],
        masks[color_index(side)],
    );
    // 走前：黑炮隔着红车打 a5（1 个炮架）⇒ attacked=true；红车自己盯着 a5 ⇒ defended=true。
    assert_eq!(approximate, (false, true, false, true));
    // 走后真值：炮架没了、红车自己站在 a5（不攻击自己）⇒ 两位都变 false。
    assert!(!position.is_square_attacked_after_move(square("a5"), side.opposite(), mv));
    assert!(!position.is_square_attacked_after_move(square("a5"), side, mv));

    // 随机对局里统计落点这一位的近似误差率。
    let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let (mut differing, mut total) = (0usize, 0usize);
    for _ in 0..6 {
        let mut position = Position::startpos();
        for _ in 0..120 {
            let moves = position.legal_moves();
            if moves.is_empty() {
                break;
            }
            let side = position.side_to_move();
            let masks = position.attacked_squares_masks();
            for _ in 0..moves.len().min(4) {
                let mv = moves[(next() as usize) % moves.len()];
                let to = mv.to as usize;
                let (pre_attacked, pre_defended) = {
                    let (sa, attacked, sd, defended) = policy_move_tactical_flags(
                        mv,
                        masks[color_index(side.opposite())],
                        masks[color_index(side)],
                    );
                    let (moved, captured) = policy_move_pieces(&position, side, mv).unwrap();
                    let dense = dense_move_index(canonical_move(side, mv));
                    let indices = policy_tactical_indices(
                        dense,
                        moved,
                        sa,
                        attacked,
                        sd,
                        defended,
                        captured,
                        position.gives_check_after_move_fast(mv),
                    );
                    assert!(indices[0] < super::POLICY_TACTICAL_EXACT_SIZE);
                    assert!(indices[1] <= POLICY_TACTICAL_SIZE);
                    (attacked, defended)
                };
                let post_attacked = position.is_square_attacked_after_move(to, side.opposite(), mv);
                let post_defended = position.is_square_attacked_after_move(to, side, mv);
                differing += usize::from(pre_attacked != post_attacked)
                    + usize::from(pre_defended != post_defended);
                total += 2;
            }
            let mv = moves[(next() as usize) % moves.len()];
            position.make_move(mv);
        }
    }
    // 存在的意义：这个比例必须**非零**（否则近似就是精确的，精确查询可以整个删掉），
    // 也不能大到失真。实测约 **21%**——远高于"罕见炮架边角"，因为沿直线走子时走子方
    // 自己就攻击着落点，走前 `defended` 必然为真、走后未必。
    assert!(differing > 0, "近似与真值完全一致？那精确查询就该删掉");
    let rate = differing as f64 / total as f64;
    assert!(
        (0.10..0.35).contains(&rate),
        "近似误差率与预期不符：{rate:.4}（{differing}/{total}）"
    );
}
