use crate::az::nnue::{V2_KING_BUCKETS, canonical_move};
use crate::xiangqi::{BOARD_SIZE, Color, Move, Position, color_index};

use super::{
    AzArenaReport, AzEvalAccumulator, AzEvalScratch, AzExperiencePool, AzNnue, AzNnueArch,
    AzSampleMeta, AzStartSource, AzTrainingSample, DENSE_MOVE_SPACE,
    POLICY_ACCUMULATOR_RANK, POLICY_CACHE_PIECE_SIZE, POLICY_CAPTURE_RELATION_OFFSET,
    POLICY_CAPTURE_RELATION_SIZE, POLICY_TACTICAL_EXACT_SIZE, POLICY_TACTICAL_SIZE,
    RULE_CONTEXT_SIZE, STRUCTURAL_PIECE_SIZE, SplitMix64, VALUE_KING_PIECE_VOCAB, VALUE_RAY_VOCAB,
    VALUE_THREAT_PAIR_VOCAB, VALUE_THREAT_VOCAB, WDL_HEAD_SIZE, dense_move_index,
    dense_move_squares, evaluate_policy_groups, move_map, policy_cache_capture_index,
    policy_cache_main_index, policy_consequence_features, policy_sparse_capture_index,
    policy_sparse_factor_indices, policy_sparse_main_index, policy_tactical_indices,
    rule_context_features, scalar_value_to_wdl_target,
    visit_value_king_piece_features, visit_value_threat_features,
};

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
            move_indices: vec![0, 1],
            policy: vec![0.6, 0.4],
            value_wdl: scalar_value_to_wdl_target(0.1),
            root_search_wdl: scalar_value_to_wdl_target(0.1),
            value: 0.1,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
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
fn packed_policy_cache_matches_original_formulas() {
    let mut model = AzNnue::random(32, 20260928);
    for (i, w) in model.policy_sparse_table.iter_mut().enumerate() {
        *w = (i % 101) as f32 * 0.003;
    }
    for (i, w) in model.policy_sparse_factor.iter_mut().enumerate() {
        *w = (i % 71) as f32 * -0.007;
    }
    for (i, w) in model.policy_accumulator_hidden.iter_mut().enumerate() {
        *w = (i % 13) as f32 * 0.001;
    }
    for (i, w) in model.policy_accumulator_move.iter_mut().enumerate() {
        *w = (i % 17) as f32 * -0.002;
    }
    model.rebuild_policy_cache();
    for mv in 0..DENSE_MOVE_SPACE {
        for piece in 0..POLICY_CACHE_PIECE_SIZE {
            for us in 0..V2_KING_BUCKETS {
                for them in 0..V2_KING_BUCKETS {
                    let mut expected = model.policy_sparse_table
                        [policy_sparse_main_index(mv, piece, us, them)];
                    for factor in policy_sparse_factor_indices(mv, piece, us, them) {
                        expected += model.policy_sparse_factor[factor];
                    }
                    assert_eq!(
                        model.policy_sparse_table_folded
                            [policy_cache_main_index(mv, piece, us, them)]
                        .to_bits(),
                        expected.to_bits()
                    );
                }
            }
            let sparse = move_map().dense_to_sparse[mv] as usize;
            let from = (piece * BOARD_SIZE + sparse / BOARD_SIZE) * POLICY_ACCUMULATOR_RANK;
            let to = (piece * BOARD_SIZE + sparse % BOARD_SIZE) * POLICY_ACCUMULATOR_RANK;
            let victim = ((piece + POLICY_CACHE_PIECE_SIZE) * BOARD_SIZE + sparse % BOARD_SIZE)
                * POLICY_ACCUMULATOR_RANK;
            let mut delta = 0.0f32;
            let mut capture = 0.0f32;
            for rank in 0..POLICY_ACCUMULATOR_RANK {
                let w = model.policy_accumulator_move[mv * POLICY_ACCUMULATOR_RANK + rank];
                delta += (model.policy_accumulator_features[to + rank]
                    - model.policy_accumulator_features[from + rank])
                    * w;
                capture += model.policy_accumulator_features[victim + rank] * w;
            }
            assert_eq!(
                model.policy_accumulator_moved_delta[mv * POLICY_CACHE_PIECE_SIZE + piece]
                    .to_bits(),
                delta.to_bits()
            );
            assert_eq!(
                model.policy_accumulator_capture[mv * POLICY_CACHE_PIECE_SIZE + piece]
                    .to_bits(),
                capture.to_bits()
            );
        }
        for captured in (POLICY_CACHE_PIECE_SIZE..STRUCTURAL_PIECE_SIZE)
            .map(Some)
            .chain(std::iter::once(None))
        {
            assert_eq!(
                model.policy_sparse_table_folded[policy_cache_capture_index(mv, captured)]
                    .to_bits(),
                model.policy_sparse_table[policy_sparse_capture_index(mv, captured)].to_bits()
            );
        }
    }
}

#[test]
fn repetition_policy_metrics_separate_opportunities_and_model_mass() {
    let position = Position::startpos();
    let moves = position.legal_moves();
    let sample = AzTrainingSample {
        features: crate::az::nnue::extract_sparse_features_az(&position),
        rule_context: [0.0; RULE_CONTEXT_SIZE],
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
fn tactical_piece_factor_is_folded_into_exact_cpu_table() {
    let mut model = AzNnue::random(16, 20260922);
    model.policy_tactical.fill(0.0);
    model.policy_tactical[POLICY_TACTICAL_EXACT_SIZE] = 0.75;
    model.rebuild_policy_tactical();
    let tactical = policy_tactical_indices(0, 0, false, false, false, false, None, false);
    assert_eq!(model.policy_tactical_folded[tactical[0]], 0.75);
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
    visit_value_king_piece_features(&second, Color::Red, |feature| {
        second_features.push(feature)
    });
    assert_eq!(first_features.len(), 6);
    assert_eq!(second_features.len(), 6);
    let rook_first = first_features
        .iter()
        .copied()
        .filter(|&feature| feature % BOARD_SIZE == 54)
        .collect::<Vec<_>>();
    let rook_second = second_features
        .iter()
        .copied()
        .filter(|&feature| feature % BOARD_SIZE == 54)
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
        Position::from_fen("1rbakab1r/9/4c3n/p3p3P/2p6/1C2c1pN1/P1P6/4B2C1/4A4/1RBAK3R w")
            .unwrap();
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
        let (_, _, victim) =
            policy_consequence_features(&position, position.side_to_move(), *mv).unwrap();
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
    let quiet = policy_tactical_indices(0, 4, true, false, true, false, None, false);
    assert_eq!(quiet[2], POLICY_TACTICAL_SIZE);
    let pawn = policy_tactical_indices(0, 4, true, false, true, false, Some(7), false);
    let rook = policy_tactical_indices(0, 4, true, false, true, false, Some(11), false);
    let elsewhere = policy_tactical_indices(100, 4, true, false, true, false, Some(11), false);
    assert_ne!(pawn[2], rook[2]);
    assert_eq!(rook[2], elsewhere[2]);
    let mut seen = std::collections::HashSet::new();
    for mover in 0..7 {
        for victim in 7..14 {
            for state in 0..32 {
                let indices = policy_tactical_indices(
                    0,
                    mover,
                    state & 1 != 0,
                    state & 2 != 0,
                    state & 4 != 0,
                    state & 8 != 0,
                    Some(victim),
                    state & 16 != 0,
                );
                assert!(
                    (POLICY_CAPTURE_RELATION_OFFSET..POLICY_TACTICAL_SIZE)
                        .contains(&indices[2])
                );
                assert!(seen.insert(indices[2]));
            }
        }
    }
    assert_eq!(seen.len(), POLICY_CAPTURE_RELATION_SIZE);
}

#[test]
fn tactical_policy_terms_distinguish_move_state() {
    let base = policy_tactical_indices(0, 4, true, true, false, true, Some(7), false);
    let changed = policy_tactical_indices(0, 4, false, true, false, true, Some(7), false);
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
    model.rebuild_value_threat();
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
fn zero_initialized_consequence_branch_preserves_policy_logits() {
    let position = Position::startpos();
    let moves = position.legal_moves();
    let model = AzNnue::random(32, 20260729);
    assert!(
        model
            .policy_consequence_output
            .iter()
            .all(|&weight| weight == 0.0)
    );

    let mut baseline = AzEvalScratch::new(model.arch);
    model.evaluate_with_scratch_output(
        &position,
        &moves,
        &[0.0; RULE_CONTEXT_SIZE],
        &mut baseline,
    );

    let mut active = model.clone();
    active.policy_consequence_output.fill(0.1);
    let mut changed = AzEvalScratch::new(model.arch);
    active.evaluate_with_scratch_output(
        &position,
        &moves,
        &[0.0; RULE_CONTEXT_SIZE],
        &mut changed,
    );
    assert!(
        baseline
            .logits
            .iter()
            .zip(&changed.logits)
            .any(|(left, right)| left != right)
    );
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
    model.evaluate_with_scratch_output(
        &position,
        &moves,
        &[0.0; RULE_CONTEXT_SIZE],
        &mut baseline,
    );

    let mut active = model.clone();
    active.policy_accumulator_move.fill(0.01);
    active.rebuild_policy_cache();
    let mut changed = AzEvalScratch::new(model.arch);
    active.evaluate_with_scratch_output(
        &position,
        &moves,
        &[0.0; RULE_CONTEXT_SIZE],
        &mut changed,
    );
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
        let before = position.clone();
        position.make_move(mv);
        AzEvalAccumulator::apply_transition_to_hidden(
            &model,
            &before,
            &position,
            mv,
            moved,
            captured,
            &mut hidden,
        );
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
        let before = position.clone();
        position.make_move(mv);
        for perspective in [Color::Red, Color::Black] {
            model.apply_policy_transition(
                &before,
                &position,
                mv,
                moved,
                captured,
                perspective,
                &mut accumulators[color_index(perspective)],
            );
            let refreshed = model.policy_accumulator(&position, perspective);
            for (incremental, full) in
                accumulators[color_index(perspective)].iter().zip(refreshed)
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
            move_indices: Vec::new(),
            policy: Vec::new(),
            value_wdl: scalar_value_to_wdl_target(1.0),
            root_search_wdl: scalar_value_to_wdl_target(1.0),
            value: 1.0,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            search_simulations: 0,
            meta: AzSampleMeta::default(),
        },
        AzTrainingSample {
            repetition_flags: Vec::new(),
            features: vec![1],
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            move_indices: Vec::new(),
            policy: Vec::new(),
            value_wdl: scalar_value_to_wdl_target(-1.0),
            root_search_wdl: scalar_value_to_wdl_target(-1.0),
            value: -1.0,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            search_simulations: 0,
            meta: AzSampleMeta::default(),
        },
        AzTrainingSample {
            repetition_flags: Vec::new(),
            features: vec![2],
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            move_indices: Vec::new(),
            policy: Vec::new(),
            value_wdl: scalar_value_to_wdl_target(0.75),
            root_search_wdl: scalar_value_to_wdl_target(0.75),
            value: 0.75,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            search_simulations: 0,
            meta: AzSampleMeta::default(),
        },
        AzTrainingSample {
            repetition_flags: Vec::new(),
            features: vec![3],
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            move_indices: Vec::new(),
            policy: Vec::new(),
            value_wdl: scalar_value_to_wdl_target(-0.75),
            root_search_wdl: scalar_value_to_wdl_target(-0.75),
            value: -0.75,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
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
            move_indices: Vec::new(),
            policy: Vec::new(),
            value_wdl: scalar_value_to_wdl_target(1.0),
            root_search_wdl: scalar_value_to_wdl_target(1.0),
            value: 1.0,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            search_simulations: 0,
            meta: AzSampleMeta::default(),
        },
        AzTrainingSample {
            repetition_flags: Vec::new(),
            features: vec![1, 5, 9],
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            move_indices: Vec::new(),
            policy: Vec::new(),
            value_wdl: scalar_value_to_wdl_target(-1.0),
            root_search_wdl: scalar_value_to_wdl_target(-1.0),
            value: -1.0,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            search_simulations: 0,
            meta: AzSampleMeta::default(),
        },
        AzTrainingSample {
            repetition_flags: Vec::new(),
            features: vec![2, 6, 10],
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            move_indices: Vec::new(),
            policy: Vec::new(),
            value_wdl: scalar_value_to_wdl_target(0.5),
            root_search_wdl: scalar_value_to_wdl_target(0.5),
            value: 0.5,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            search_simulations: 0,
            meta: AzSampleMeta::default(),
        },
        AzTrainingSample {
            repetition_flags: Vec::new(),
            features: vec![3, 7, 11],
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            move_indices: Vec::new(),
            policy: Vec::new(),
            value_wdl: scalar_value_to_wdl_target(-0.5),
            root_search_wdl: scalar_value_to_wdl_target(-0.5),
            value: -0.5,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
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
    let single_stats =
        train_samples(&mut single, &samples, 5, 0.003, 4, &mut rng_single).unwrap();
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
            move_indices: Vec::new(),
            policy: Vec::new(),
            value_wdl: scalar_value_to_wdl_target(1.0),
            root_search_wdl: scalar_value_to_wdl_target(1.0),
            value: 1.0,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            search_simulations: 0,
            meta: AzSampleMeta::default(),
        },
        AzTrainingSample {
            repetition_flags: Vec::new(),
            features: vec![1, 5, 9],
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            move_indices: Vec::new(),
            policy: Vec::new(),
            value_wdl: scalar_value_to_wdl_target(-1.0),
            root_search_wdl: scalar_value_to_wdl_target(-1.0),
            value: -1.0,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            search_simulations: 0,
            meta: AzSampleMeta::default(),
        },
        AzTrainingSample {
            repetition_flags: Vec::new(),
            features: vec![2, 6, 10],
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            move_indices: Vec::new(),
            policy: Vec::new(),
            value_wdl: scalar_value_to_wdl_target(0.75),
            root_search_wdl: scalar_value_to_wdl_target(0.75),
            value: 0.75,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            search_simulations: 0,
            meta: AzSampleMeta::default(),
        },
        AzTrainingSample {
            repetition_flags: Vec::new(),
            features: vec![3, 7, 11],
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            move_indices: Vec::new(),
            policy: Vec::new(),
            value_wdl: scalar_value_to_wdl_target(-0.75),
            root_search_wdl: scalar_value_to_wdl_target(-0.75),
            value: -0.75,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
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
    let model = AzNnue::random(16, 42);
    let path = std::env::temp_dir().join("chineseai_test_aznnue_roundtrip.safetensors");
    let _ = fs::remove_file(&path);
    model.save(&path).unwrap();
    let loaded = AzNnue::load(&path).unwrap();
    let _ = fs::remove_file(&path);
    assert_eq!(model.hidden_size, loaded.hidden_size);
    assert_eq!(model.input_hidden, loaded.input_hidden);
    assert_eq!(model.input_piece_hidden, loaded.input_piece_hidden);
    assert_eq!(model.input_rank_hidden, loaded.input_rank_hidden);
    assert_eq!(model.input_file_hidden, loaded.input_file_hidden);
    assert_eq!(
        model.input_king_piece_hidden,
        loaded.input_king_piece_hidden
    );
    assert_eq!(model.hidden_bias, loaded.hidden_bias);
    assert_eq!(model.value_head_hidden, loaded.value_head_hidden);
    assert_eq!(model.value_head_bias, loaded.value_head_bias);
    assert_eq!(model.value_head_output, loaded.value_head_output);
    assert_eq!(model.policy_move_bias, loaded.policy_move_bias);
    assert_eq!(
        model.policy_consequence_output,
        loaded.policy_consequence_output
    );
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
            move_indices: vec![0],
            policy: vec![1.0],
            value_wdl: scalar_value_to_wdl_target(0.0),
            root_search_wdl: scalar_value_to_wdl_target(0.0),
            value: 0.0,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
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
            move_indices: vec![0],
            policy: vec![1.0],
            value_wdl: scalar_value_to_wdl_target(0.0),
            root_search_wdl: scalar_value_to_wdl_target(0.0),
            value: 0.0,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
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
            move_indices: vec![0],
            policy: vec![1.0],
            value_wdl: scalar_value_to_wdl_target(0.0),
            root_search_wdl: scalar_value_to_wdl_target(0.0),
            value: 0.0,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
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
    assert!(context[2] > 0.0);
    assert_eq!(context[3..], [0.0; 4]);
}
