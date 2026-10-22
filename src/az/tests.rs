use super::features::*;
use super::*;
use crate::xiangqi::{Position, color_index};
use std::collections::BTreeMap;

fn sample(position: &Position) -> AzTrainingSample {
    let moves = position.legal_moves();
    let side = position.side_to_move();
    AzTrainingSample {
        features: nnue::extract_sparse_features_az(position),
        rule_context: rule_context_features(position, &position.initial_rule_history()),
        move_indices: moves
            .iter()
            .map(|&m| dense_move_index(nnue::canonical_move(side, m)))
            .collect(),
        repetition_flags: vec![0; moves.len()],
        policy: vec![1. / moves.len() as f32; moves.len()],
        value_wdl: [0.2, 0.5, 0.3],
        root_search_wdl: [0.2, 0.5, 0.3],
        value: -0.1,
        side_sign: 1.,
        policy_weight: 1.,
        value_weight: 1.,
        moves_left: 40.,
        moves_left_weight: 1.,
        search_simulations: 32,
        meta: AzSampleMeta::default(),
    }
}
fn next_position(p: &mut Position, rng: &mut SplitMix64) -> Option<Move> {
    let moves = p.legal_moves();
    if moves.is_empty() {
        return None;
    }
    let m = moves[rng.next_u64() as usize % moves.len()];
    p.make_move(m);
    Some(m)
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= 1e-4 + 1e-5 * y.abs(),
            "index {i}: {x} vs {y}"
        );
    }
}

#[test]
fn geometric_graph_matches_rules_for_every_square() {
    let mut rng = SplitMix64::new(3901);
    let mut p = Position::startpos();
    for _ in 0..240 {
        let g = Graph::new(Board::new(&p));
        let masks = p.attacked_squares_masks();
        for side in [Color::Red, Color::Black] {
            let mut actual = 0;
            for target in 0..90 {
                let attack = g.origins(target, side) != 0;
                assert_eq!(
                    attack,
                    p.is_square_attacked(target, side),
                    "{} target={target} side={side:?}",
                    p.to_fen()
                );
                if attack {
                    actual |= 1u128 << target;
                }
            }
            assert_eq!(actual, masks[color_index(side)]);
        }
        if next_position(&mut p, &mut rng).is_none() {
            p = Position::startpos();
        }
    }
}

#[test]
fn virtual_move_origins_equal_make_move_including_captures_and_kings() {
    let mut rng = SplitMix64::new(229);
    let mut p = Position::startpos();
    for ply in 0..100 {
        let graph = Graph::new(Board::new(&p));
        let moves = p.legal_moves();
        if moves.is_empty() {
            p = Position::startpos();
            continue;
        }
        for &mv in moves.iter().step_by((moves.len() / 5).max(1)) {
            let patch = graph.patch(mv);
            let mut after = p.clone();
            after.make_move(mv);
            let full = Graph::new(Board::new(&after));
            for target in 0..90 {
                for side in [Color::Red, Color::Black] {
                    assert_eq!(
                        patch.origins(&graph, target, side),
                        full.origins(target, side),
                        "ply={ply} move={mv} target={target}"
                    );
                    assert_eq!(
                        patch.origins(&graph, target, side) != 0,
                        p.is_square_attacked_after_move(target, side, mv)
                    );
                }
            }
        }
        p.make_move(moves[rng.next_u64() as usize % moves.len()]);
    }
}

#[test]
fn destination_support_excludes_moving_rook_itself() {
    let p = Position::from_fen("4k4/9/9/9/4P4/9/9/9/9/R3K4 w").unwrap();
    let g = Graph::new(Board::new(&p));
    let mv = Move::new(81, 72);
    assert!(g.origins(72, Color::Red) & (1u128 << 81) != 0);
    let f = move_features(&g, mv, Color::Red, None);
    assert_eq!(f.local[4], 0.);
    assert_eq!(f.local[11], 0.);
}

#[test]
fn cannon_screen_removal_can_destroy_or_create_attack() {
    for (fen, before, after) in [
        ("c3k4/9/9/9/R3P4/9/9/9/9/R3K4 w", true, false),
        ("c3k4/9/9/9/R3P4/9/P8/9/9/R3K4 w", false, true),
    ] {
        let p = Position::from_fen(fen).unwrap();
        let g = Graph::new(Board::new(&p));
        let patch = g.patch(Move::new(36, 37));
        assert_eq!(g.origins(81, Color::Black) & 1 != 0, before);
        assert_eq!(patch.origins(&g, 81, Color::Black) & 1 != 0, after);
    }
}

#[test]
fn horse_leg_unlock_changes_unmoved_attacker() {
    let p = Position::from_fen("3k5/9/9/5r3/4P4/4N4/9/9/9/4K4 w").unwrap();
    let g = Graph::new(Board::new(&p));
    assert_eq!(g.controls[49] & (1u128 << 32), 0);
    let patch = g.patch(Move::new(40, 31));
    assert_ne!(patch.controls(&g, 49) & (1u128 << 32), 0);
    assert!(
        motifs(&g, u128::MAX, Color::Red)
            .iter()
            .any(|m| m.family == 2 && m.root == 49 && m.b == 40 && m.c == 32)
    );
}

#[test]
fn motifs_bind_common_attacker_and_defender_identity() {
    let p = Position::from_fen("3k5/9/9/3r1c3/9/4N4/9/9/9/4K4 w").unwrap();
    let g = Graph::new(Board::new(&p));
    assert!(
        motifs(&g, u128::MAX, Color::Red)
            .iter()
            .any(|m| m.family == 3
                && m.root == 49
                && [m.b, m.c].contains(&30)
                && [m.b, m.c].contains(&32))
    );
    let p = Position::from_fen("3k5/9/9/3P5/9/4Nr3/9/9/9/4K4 w").unwrap();
    let g = Graph::new(Board::new(&p));
    assert!(
        motifs(&g, u128::MAX, Color::Red)
            .iter()
            .any(|m| m.family == 5 && m.root == 49 && m.b == 50 && m.c == 30)
    );
}

#[test]
fn every_static_motif_family_has_an_explicit_positive_witness() {
    let mut seen = [false; 12];
    for fen in [
        "c3k4/9/9/9/R3P4/9/9/9/9/R3K4 w",
        "3k5/9/9/5r3/4P4/4N4/9/9/9/4K4 w",
        "3k5/9/9/3r1c3/9/4N4/9/9/9/4K4 w",
        "r3k4/9/1N7/9/4P4/9/9/9/9/R3K4 w",
        "3k5/9/9/3P5/9/4Nr3/9/9/9/4K4 w",
        "3k5/9/9/3P1P3/9/4N4/9/9/9/4K4 w",
        "r3k4/r8/1N7/9/4P4/9/9/9/9/R3K4 w",
        "r3k4/9/9/9/C3P4/9/9/9/9/R3K4 w",
        "3kr4/9/9/9/4P4/9/9/9/9/4K4 w",
        "3kr4/9/9/9/9/9/9/9/3r5/4K4 w",
        "c2k5/9/1r7/9/N8/9/9/9/9/R3K4 w",
    ] {
        let p = Position::from_fen(fen).unwrap();
        let g = Graph::new(Board::new(&p));
        for event in motifs(&g, u128::MAX, Color::Red) {
            seen[event.family as usize] = true;
        }
        if !king_context(&g, Color::Red).1.is_empty() {
            seen[10] = true;
        }
    }
    assert_eq!(seen, [true; 12]);
    let p = Position::from_fen("c2k5/9/9/9/N8/9/9/9/9/R3K4 w").unwrap();
    assert!(
        !motifs(&Graph::new(Board::new(&p)), u128::MAX, Color::Red)
            .iter()
            .any(|m| m.family == 11 && m.root == 36)
    );
}

#[test]
fn king_line_gate_does_not_erase_legal_along_line_capture() {
    let p = Position::from_fen("3kr4/9/9/9/4R4/9/9/9/9/4K4 w").unwrap();
    let graph = Graph::new(Board::new(&p));
    assert!(graph.constrained(40));
    assert_ne!(graph.controls[40] & (1u128 << 4), 0);
    assert!(p.is_legal_move(Move::new(40, 4)));
}

#[test]
fn move_vacancy_cache_is_identical_and_guard_events_have_real_dependencies() {
    let mut p = Position::startpos();
    let mut rng = SplitMix64::new(9931);
    for _ in 0..120 {
        let graph = Graph::new(Board::new(&p));
        let moves = p.legal_moves();
        let mut cache = MoveFeatureCache::new(&graph, moves.iter().copied());
        for &mv in &moves {
            let a = move_features(&graph, mv, p.side_to_move(), Some(false));
            let b = cache.features(mv, p.side_to_move(), Some(false));
            assert_eq!(a.local, b.local);
            assert_eq!(a.joint, b.joint);
            assert_eq!(a.events, b.events);
            let mut after = p.clone();
            after.make_move(mv);
            let after = Graph::new(Board::new(&after));
            // 独立从两张完整图求带颜色/棋种的边差，不能仅比较控制位。
            let mut edges = BTreeMap::new();
            for (g, sign) in [(&graph, -1), (&after, 1)] {
                for source in bits(g.board.colors[color_index(p.side_to_move())]) {
                    let piece = g.board.cells[source].unwrap();
                    for target in bits(g.controls[source] & g.board.occupied) {
                        let victim = g.board.cells[target].unwrap();
                        *edges
                            .entry((
                                source,
                                target,
                                crate::xiangqi::piece_kind_index(piece.kind),
                                nnue::piece_absolute_feature_index(p.side_to_move(), victim),
                            ))
                            .or_insert(0i32) += sign;
                    }
                }
            }
            let mut expected = BTreeMap::new();
            for ((_, _, source_kind, victim_class), count) in edges {
                let family = match source_kind {
                    4 => 0,
                    5 => 1,
                    2 | 3 => 2,
                    _ => continue,
                };
                if count != 0 {
                    let row = (((family * 7 + a.moved) * 8 + victim_class % 7) * 2
                        + usize::from(count > 0)) as u32;
                    *expected.entry(row).or_insert(0usize) += count.unsigned_abs() as usize;
                }
            }
            let mut actual = BTreeMap::new();
            for &row in &a.events {
                if row < (3 * 7 * 8 * 2) as u32 {
                    *actual.entry(row).or_insert(0usize) += 1;
                }
            }
            assert_eq!(actual, expected, "{} move={mv}", p.to_fen());
        }
        if next_position(&mut p, &mut rng).is_none() {
            p = Position::startpos();
        }
    }
    let p = Position::from_fen("4k4/9/9/9/4P4/9/9/9/9/R3K4 w").unwrap();
    let effect = move_features(
        &Graph::new(Board::new(&p)),
        Move::new(81, 72),
        Color::Red,
        None,
    );
    assert!(!effect.events.iter().any(|&e| e as usize / (7 * 8 * 2) == 5));
    for pawn in ["p8", "9"] {
        let p = Position::from_fen(&format!("3k5/9/9/9/9/{pawn}/9/1N7/9/R3K4 w")).unwrap();
        let effect = move_features(
            &Graph::new(Board::new(&p)),
            Move::new(64, 45),
            Color::Red,
            None,
        );
        // 占用不变的吃子也把车对敌兵的攻击变成对己马的支援。
        assert!(effect.events.contains(&55));
        assert_eq!(effect.events.contains(&60), pawn == "p8");
    }
}

#[test]
fn king_escape_context_uses_post_move_attack_query() {
    let mut rng = SplitMix64::new(81);
    let mut p = Position::startpos();
    for _ in 0..160 {
        let g = Graph::new(Board::new(&p));
        let (context, seals) = king_context(&g, p.side_to_move());
        assert!(seals.iter().all(|&i| i < MOTIF_ROWS as u32));
        let side = p.side_to_move();
        let king = p.general_square(side).unwrap();
        let legal = p
            .legal_moves()
            .iter()
            .filter(|m| m.from as usize == king)
            .count();
        assert_eq!(context[6] * 4., legal as f32, "{}", p.to_fen());
        assert_eq!(context[0] != 0., p.in_check(side));
        if next_position(&mut p, &mut rng).is_none() {
            p = Position::startpos();
        }
    }
}

fn row_difference(old: Vec<u32>, new: Vec<u32>) -> BTreeMap<u32, i32> {
    let mut result = BTreeMap::new();
    for row in old {
        *result.entry(row).or_default() -= 1;
    }
    for row in new {
        *result.entry(row).or_default() += 1;
    }
    result.retain(|_, n| *n != 0);
    result
}

#[test]
fn dirty_root_integer_deltas_equal_full_extraction_on_both_views() {
    let mut rng = SplitMix64::new(449);
    let mut p = Position::startpos();
    for _ in 0..180 {
        let before = p.clone();
        let Some(mv) = next_position(&mut p, &mut rng) else {
            p = Position::startpos();
            continue;
        };
        let old = Graph::new(Board::new(&before));
        let new = Graph::new(Board::new(&p));
        let (sources, roots) = dirty_roots(&old, &new, (1u128 << mv.from) | (1u128 << mv.to));
        for side in [Color::Red, Color::Black] {
            let a = extract(&old, side, u128::MAX);
            let b = extract(&new, side, u128::MAX);
            let x = extract(&old, side, roots);
            let y = extract(&new, side, roots);
            assert!(
                a.relations
                    .iter()
                    .chain(&b.relations)
                    .all(|&i| i < RELATION_ROWS as u32)
            );
            assert!(
                a.motifs
                    .iter()
                    .chain(&b.motifs)
                    .all(|&i| i < MOTIF_ROWS as u32)
            );
            assert_eq!(
                row_difference(a.relations, b.relations),
                row_difference(
                    relation_rows(&old, side, sources),
                    relation_rows(&new, side, sources)
                )
            );
            assert_eq!(
                row_difference(a.motifs, b.motifs),
                row_difference(x.motifs, y.motifs),
                "{} => {}",
                before.to_fen(),
                p.to_fen()
            );
        }
    }
}

#[test]
fn nnue_single_and_two_ply_updates_match_full_network() {
    let model = AzNnue::random(16, 19);
    let mut rng = SplitMix64::new(991);
    let mut p = Position::startpos();
    let mut arena = node_features::NodeFeatureArena::default();
    let mut ids = [
        arena.root(&model, &p, Color::Red),
        arena.root(&model, &p, Color::Black),
    ];
    let mut accumulator = AzEvalAccumulator::new(&model, &p);
    let mut history = p.initial_rule_history();
    for ply in 0..160 {
        let before = p.clone();
        let moves = p.legal_moves();
        if moves.is_empty() {
            break;
        }
        let mv = moves[rng.next_u64() as usize % moves.len()];
        let moved = p.piece_at(mv.from as usize).unwrap();
        let captured = p.piece_at(mv.to as usize);
        history.push(p.rule_history_entry_after_move(mv));
        p.make_move(mv);
        AzEvalAccumulator::apply_transition_to_hidden(
            &model,
            &before,
            &p,
            mv,
            moved,
            captured,
            &mut accumulator.hidden_sum,
        );
        for side in [Color::Red, Color::Black] {
            let c = color_index(side);
            ids[c] = arena.update(
                &model,
                ids[c],
                &before,
                &p,
                side,
                (1u128 << mv.from) | (1u128 << mv.to),
            );
            let full = model.feature_sums(&Graph::new(Board::new(&p)), side);
            let delta = arena.value(ids[c]);
            close(&full.king, &delta.king);
            close(&full.relations, &delta.relations);
            close(&full.motifs, &delta.motifs);
        }
        let full_acc = AzEvalAccumulator::new(&model, &p);
        close(&accumulator.hidden_sum, &full_acc.hidden_sum);
        let moves = p.legal_moves();
        let rules = rule_context_features(&p, &history);
        let mut full = AzEvalScratch::new(model.arch);
        let mut inc = AzEvalScratch::new(model.arch);
        inc.feature_precomputed = Some(arena.value(ids[color_index(p.side_to_move())]));
        let a = model.evaluate_with_scratch_output(&p, &moves, &rules, &mut full);
        let b = model.evaluate_incremental_with_scratch_output(
            &p,
            &accumulator.hidden_sum,
            &moves,
            &[],
            &rules,
            &mut inc,
        );
        close(&a.value_wdl, &b.value_wdl);
        close(&full.logits, &inc.logits);
        assert!((a.moves_left - b.moves_left).abs() < 1e-3, "ply {ply}");
    }
    let p = Position::startpos();
    let mut work = alphazero::AzSearchWorkspace::new(&model);
    let limits = AzSearchLimits {
        simulations: 80,
        seed: 77,
        ..Default::default()
    };
    let a = alphazero_search_with_rules(&p, Some(p.initial_rule_history()), None, &model, limits);
    let b = alphazero::alphazero_search_with_rules_reusing(
        &p,
        &p.initial_rule_history(),
        p.legal_moves(),
        limits,
        &mut work,
    );
    assert_eq!(a.best_move, b.best_move);
}

#[test]
fn unknown_rules_are_distinct_from_known_absence() {
    let p = Position::startpos();
    let unknown = rule_context_features(&p, &[]);
    let known = rule_context_features(&p, &p.initial_rule_history());
    assert_eq!(&unknown[12..], &[0.; 6]);
    assert_eq!(&known[12..], &[1.; 6]);
    assert_eq!(known[0], 1.);
    assert_eq!(known[2], 1.);
    let p = Position::from_fen("4k4/9/9/9/4P4/9/9/9/9/4K4 w - - 50 1").unwrap();
    let missing = rule_context_features(&p, &p.initial_rule_history());
    assert_eq!(&missing[10..], &[0.; 8]);
    assert_eq!(&missing[1..9], &[0.; 8]);
    assert_eq!(repetition_flag(false, false), 2);
    assert_eq!(repetition_flag(true, false), 1);
    assert_eq!(repetition_flag(false, true), 0);
}

#[test]
fn two_ply_feature_deltas_and_branch_restores_are_exact() {
    let model = AzNnue::random(12, 171);
    let mut rng = SplitMix64::new(921);
    let mut p = Position::startpos();
    for _ in 0..80 {
        let before = p.clone();
        let mut changed = 0;
        for _ in 0..2 {
            let Some(m) = next_position(&mut p, &mut rng) else {
                return;
            };
            changed |= (1u128 << m.from) | (1u128 << m.to);
        }
        let old = Graph::new(Board::new(&before));
        let new = Graph::new(Board::new(&p));
        let (sources, roots) = dirty_roots(&old, &new, changed);
        for side in [Color::Red, Color::Black] {
            let a = extract(&old, side, u128::MAX);
            let b = extract(&new, side, u128::MAX);
            assert_eq!(
                row_difference(a.relations, b.relations),
                row_difference(
                    relation_rows(&old, side, sources),
                    relation_rows(&new, side, sources)
                )
            );
            assert_eq!(
                row_difference(a.motifs, b.motifs),
                row_difference(
                    extract(&old, side, roots).motifs,
                    extract(&new, side, roots).motifs
                )
            );
            let mut arena = node_features::NodeFeatureArena::default();
            let base = arena.root(&model, &before, side);
            for _ in 0..3 {
                let updated = arena.update(&model, base, &before, &p, side, changed);
                let delta = arena.value(updated);
                let full = model.feature_sums(&new, side);
                close(&delta.king, &full.king);
                close(&delta.relations, &full.relations);
                close(&delta.motifs, &full.motifs);
            }
        }
    }
}

#[test]
fn canonical_training_board_matches_real_black_and_red_perspectives() {
    let model = AzNnue::random(12, 91);
    let mut p = Position::startpos();
    let mut rng = SplitMix64::new(881);
    for _ in 0..100 {
        let s = sample(&p);
        let canonical = position_for_training_sample(&s).unwrap();
        let g = Graph::new(Board::new(&p));
        let c = Graph::new(Board::new(&canonical));
        let mut a = extract(&g, p.side_to_move(), u128::MAX);
        let mut b = extract(&c, Color::Red, u128::MAX);
        a.relations.sort_unstable();
        b.relations.sort_unstable();
        a.motifs.sort_unstable();
        b.motifs.sort_unstable();
        assert_eq!(a.relations, b.relations);
        assert_eq!(a.motifs, b.motifs, "{}", p.to_fen());
        assert_eq!(
            king_context(&g, p.side_to_move()),
            king_context(&c, Color::Red)
        );
        let mut scratch = AzEvalScratch::new(model.arch);
        let real = model.evaluate_with_scratch_output_with_repetition(
            &p,
            &p.legal_moves(),
            &s.repetition_flags,
            &s.rule_context,
            &mut scratch,
        );
        let (wdl, policy) = outputs_for_training_sample(&model, &s).unwrap();
        close(&real.value_wdl, &wdl);
        close(&scratch.logits, &policy);
        if next_position(&mut p, &mut rng).is_none() {
            p = Position::startpos();
        }
    }
}

fn training_parity(device: &candle_core::Device) {
    use candle_nn::ops::log_softmax;
    let mut p = Position::startpos();
    let mut rng = SplitMix64::new(7);
    let mut samples = vec![sample(&p)];
    for _ in 0..7 {
        next_position(&mut p, &mut rng);
    }
    samples.push(sample(&p));
    samples.push(sample(
        &Position::from_fen("3k5/9/9/3r1c3/9/4N4/9/9/9/4K4 w").unwrap(),
    ));
    let mut model = AzNnue::random(96, 17);
    let candle = candle_model::AzCandleModel::from_model(&model, device).unwrap();
    let packed = dataloader::PackedBatch::from_indices(&samples, &[0, 1, 2]);
    let batch = candle_model::BatchTensors::from_packed(packed, device).unwrap();
    for iteration in 0..2 {
        let output = candle.forward(&batch).unwrap();
        let wdl = candle_nn::ops::softmax(&output.value_logits, 1)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let logits = output.policy_logits.to_vec2::<f32>().unwrap();
        for (i, sample) in samples.iter().enumerate() {
            let (expected, scores) = outputs_for_training_sample(&model, sample).unwrap();
            close(&wdl[i], &expected);
            close(&logits[i][..scores.len()], &scores);
        }
        if iteration == 0 {
            let loss = ((log_softmax(&output.policy_logits, 1).unwrap() * &batch.policy_targets)
                .unwrap()
                .sum_all()
                .unwrap()
                .neg()
                .unwrap()
                + output.value_logits.sqr().unwrap().sum_all().unwrap())
            .unwrap();
            let grads = loss.backward().unwrap();
            let (vars, decay) = candle.all_vars_with_decay();
            let mut opt = adamw::AzAdamW::new(vars, decay, 0.01).unwrap();
            opt.step(&grads).unwrap();
            let before = model.clone();
            candle.copy_to_model(&mut model).unwrap();
            assert_ne!(before.relation_embedding, model.relation_embedding);
            assert_ne!(before.motif_embedding, model.motif_embedding);
            assert_ne!(before.king_piece, model.king_piece);
            assert_ne!(before.policy_local, model.policy_local);
        }
    }
}

#[test]
fn training_forward_backward_matches_cpu_inference() {
    training_parity(&candle_core::Device::Cpu);
}

#[cfg(feature = "slow-tests")]
#[test]
fn training_forward_backward_matches_cuda_inference() {
    training_parity(&candle_core::Device::new_cuda(0).expect("CUDA requested"));
}

#[test]
fn model_schema_roundtrip_and_old_version_rejection() {
    let model = AzNnue::random(8, 57);
    let path =
        std::env::temp_dir().join(format!("tactical-model-{}.safetensors", std::process::id()));
    model.save(&path).unwrap();
    let restored = AzNnue::load(&path).unwrap();
    let sample = sample(&Position::startpos());
    assert_eq!(
        outputs_for_training_sample(&model, &sample),
        outputs_for_training_sample(&restored, &sample)
    );
    let mut tensors = candle_core::safetensors::load(&path, &candle_core::Device::Cpu).unwrap();
    tensors.insert(
        "az_model_format_version".to_owned(),
        candle_core::Tensor::new(&[40f32], &candle_core::Device::Cpu).unwrap(),
    );
    candle_core::safetensors::save(&tensors, &path).unwrap();
    assert!(AzNnue::load(&path).is_err());
    std::fs::remove_file(path).unwrap();
}
