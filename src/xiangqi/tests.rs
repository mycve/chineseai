use super::*;

#[test]
fn startpos_roundtrip_fen() {
    let position = Position::startpos();
    assert_eq!(position.to_fen(), format!("{STARTPOS_FEN} - - 0 1"));
}

#[test]
fn full_fen_preserves_rule60_clock() {
    let fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b - - 57 80";
    let position = Position::from_fen(fen).unwrap();
    assert_eq!(position.halfmove_clock(), 57);
    assert_eq!(
        position.to_fen(),
        format!("{} b - - 57 1", fen.split_once(' ').unwrap().0)
    );
}

#[test]
fn to_fen_with_history_counts_check_exemptions() {
    let position = Position::from_fen(
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 30 1",
    )
    .unwrap();
    assert_eq!(position.halfmove_clock(), 30);
    let mut history = position.initial_rule_history();
    // 红方连续将军 11 次：前 10 次照常计数，第 11 次起进入豁免、不再计入 60 回合。
    for _ in 0..11 {
        history.push(RuleHistoryEntry {
            hash: 0,
            side_to_move: Color::Black,
            mover: Some(Color::Red),
            gives_check: true,
            chased_mask: 0,
            mv: None,
            captured: None,
            rule60_clock: 0,
        });
    }
    assert_eq!(position.rule60_count_with_history(&history), 40);
    // 两个导出函数写的是不同的数：`to_fen` 是原始 halfmove clock，`to_fen_with_history`
    // 才是计入将军豁免之后的有效计数。谁把后者"简化"成前者，这条就会红。
    assert_eq!(position.to_fen().split(' ').nth(4), Some("30"));
    assert_eq!(
        position.to_fen_with_history(&history).split(' ').nth(4),
        Some("40")
    );
    // 注意：40 回读进 halfmove_clock 只是近似值，豁免状态本身无法用 FEN 表达。
    assert_eq!(
        position.to_fen_with_history(&history).split(' ').next(),
        position.to_fen().split(' ').next()
    );
}

#[test]
fn soldier_move_does_not_reset_rule60_clock() {
    let mut position = Position::from_fen(
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 57 1",
    )
    .unwrap();
    let mv = position.parse_uci_move("a3a4").unwrap();
    position.make_move(mv);
    assert_eq!(position.halfmove_clock(), 58);
}

#[test]
fn capture_resets_rule60_clock() {
    let mut position = Position::from_fen("4k4/9/9/9/9/9/4p4/4R4/9/3K5 w - - 57 1").unwrap();
    let mv = position.parse_uci_move("e2e3").unwrap();
    position.make_move(mv);
    assert_eq!(position.halfmove_clock(), 0);
}

#[test]
fn rule60_can_be_disabled_or_use_a_custom_limit() {
    let mut position = Position::from_fen("4k4/9/9/9/9/9/9/4R4/9/3K5 w - - 20 1").unwrap();
    let history = position.initial_rule_history();
    position.set_rule60_max_ply(Some(20));
    assert_eq!(
        position.rule_outcome_with_history(&history),
        Some(RuleOutcome::Draw(RuleDrawReason::NaturalMoveLimit))
    );
    position.set_rule60_max_ply(None);
    assert_eq!(position.rule_outcome_with_history(&history), None);
}

#[test]
fn insufficient_material_draws_bare_defenders_and_lone_cannon() {
    for fen in ["3k5/9/9/9/9/9/9/9/9/4K4 w", "3k5/9/9/9/9/9/9/4C4/9/4K4 w"] {
        let position = Position::from_fen(fen).unwrap();
        assert_eq!(
            position.rule_outcome_with_history(&position.initial_rule_history()),
            Some(RuleOutcome::Draw(RuleDrawReason::InsufficientMaterial)),
            "{fen}"
        );
    }
}

#[test]
fn horse_material_is_not_an_automatic_draw() {
    let position = Position::from_fen("3k5/9/9/9/9/9/9/4N4/9/4K4 w").unwrap();
    assert_eq!(
        position.rule_outcome_with_history(&position.initial_rule_history()),
        None
    );
}

#[test]
fn repetition_violation_precedes_natural_move_limit() {
    let position = Position::from_fen("3k5/9/9/9/9/9/9/4R4/9/4K4 w - - 120 1").unwrap();
    let mut history = vec![
        test_rule_entry(1, Color::Red, None, false, 0),
        test_rule_entry(2, Color::Black, Some(Color::Red), true, 0),
        test_rule_entry(3, Color::Red, Some(Color::Black), false, 0),
        test_rule_entry(4, Color::Black, Some(Color::Red), true, 0),
        test_rule_entry(1, Color::Red, Some(Color::Black), false, 0),
    ];
    history.extend_from_within(1..);
    assert_eq!(
        position.rule_outcome_with_history(&history),
        Some(RuleOutcome::Win(Color::Black))
    );
}

#[test]
fn checks_after_ten_and_their_replies_do_not_advance_rule60() {
    let mut position = Position::from_fen("3k5/9/9/9/9/9/9/4R4/9/4K4 w - - 22 1").unwrap();
    let mut history = vec![test_rule_entry(1, Color::Red, None, false, 0)];
    for index in 0..11u64 {
        history.push(test_rule_entry(
            10 + index * 2,
            Color::Black,
            Some(Color::Red),
            true,
            0,
        ));
        history.push(test_rule_entry(
            11 + index * 2,
            Color::Red,
            Some(Color::Black),
            false,
            0,
        ));
    }
    assert_eq!(position.rule60_count_with_history(&history), 20);
    assert!(
        position
            .to_fen_with_history(&history)
            .ends_with(" - - 20 1")
    );
    position.set_rule60_max_ply(Some(21));
    assert_eq!(position.rule_outcome_with_history(&history), None);
    position.set_rule60_max_ply(Some(20));
    assert_eq!(
        position.rule_outcome_with_history(&history),
        Some(RuleOutcome::Draw(RuleDrawReason::NaturalMoveLimit))
    );
}

#[test]
fn square_names_follow_pikafish_uci_coordinates() {
    assert_eq!(square_name(index(0, 9)), "a0");
    assert_eq!(square_name(index(8, 0)), "i9");
    assert_eq!(parse_square("a0"), Some(index(0, 9)));
    assert_eq!(parse_square("i9"), Some(index(8, 0)));
}

#[test]
fn mirror_files_preserves_side_and_roundtrips() {
    let position =
        Position::from_fen("3ak4/9/2n1b4/p3p3p/4R4/2P6/P3P3P/2N1C4/4A4/2BAK3c b").unwrap();
    let mirrored = position.mirror_files();
    assert_eq!(mirrored.side_to_move(), position.side_to_move());
    assert_eq!(mirrored.mirror_files(), position);
    assert_ne!(mirrored.to_fen(), position.to_fen());
}

#[test]
fn horse_leg_block_prevents_move() {
    let position = Position::from_fen("4k4/9/9/9/4P4/9/3P5/3H5/9/4K4 w").unwrap();
    let moves = position.legal_moves();
    assert!(!moves.iter().any(|mv| mv.to == index(2, 5) as u8));
    assert!(!moves.iter().any(|mv| mv.to == index(4, 5) as u8));
}

#[test]
fn elephant_cannot_cross_river() {
    let position = Position::from_fen("4k4/9/9/9/4P4/9/9/2E6/9/4K4 w").unwrap();
    let moves = position.legal_moves();
    let elephant_moves: Vec<_> = moves
        .iter()
        .filter(|mv| mv.from == index(2, 7) as u8)
        .copied()
        .collect();
    assert!(!elephant_moves.is_empty());
    assert!(elephant_moves.iter().all(|mv| rank_of(mv.to as usize) >= 5));
}

#[test]
fn cannon_requires_exactly_one_screen_to_capture() {
    let position = Position::from_fen("4k4/9/9/9/4C4/4P4/4r4/9/9/3K5 w").unwrap();
    let moves = position.legal_moves();
    assert!(moves.iter().any(|mv| mv.to == index(4, 6) as u8));

    let position_without_screen = Position::from_fen("4k4/9/9/9/4C4/9/4r4/9/9/3K5 w").unwrap();
    let moves_without_screen = position_without_screen.legal_moves();
    assert!(
        !moves_without_screen
            .iter()
            .any(|mv| mv.to == index(4, 6) as u8)
    );
}

#[test]
fn network_relations_include_cannon_screen_and_target() {
    let position = Position::from_canonical_piece_squares(&[
        (5, 40),  // red cannon
        (6, 41),  // red soldier used as screen
        (11, 43), // black rook behind the screen
    ]);
    let mut targets = Vec::new();
    position.visit_occupied_relations(|source, attacker, target, _| {
        if source == 40 && attacker.kind == PieceKind::Cannon {
            targets.push(target);
        }
    });
    targets.sort_unstable();
    assert_eq!(targets, vec![41, 43]);
}

#[test]
fn facing_generals_exposure_is_illegal() {
    let position = Position::from_fen("4k4/9/9/9/9/9/4R4/9/9/4K4 w").unwrap();
    let moves = position.legal_moves();
    assert!(
        !moves
            .iter()
            .any(|mv| mv.from == index(4, 6) as u8 && mv.to == index(3, 6) as u8)
    );
}

#[test]
fn legal_moves_do_not_capture_general() {
    let position = Position::from_fen("4k4/9/9/9/9/9/4R4/9/9/4K4 w").unwrap();
    assert!(position.in_check(Color::Black));
    assert!(
        !position
            .legal_moves()
            .iter()
            .any(|mv| { mv.from == index(4, 6) as u8 && mv.to == index(4, 0) as u8 })
    );
}

#[test]
fn facing_generals_position_is_rejected() {
    let position = Position::from_fen("4k4/9/9/9/9/9/9/9/9/4K4 w").unwrap_err();
    assert_eq!(position, "illegal position: generals are facing");
}

#[test]
fn make_and_unmake_restores_position() {
    let mut position = Position::startpos();
    let original = position.clone();
    let mv = position.legal_moves()[0];
    let undo = position.make_move(mv);
    position.unmake_move(mv, undo);
    assert_eq!(position, original);
}

#[test]
fn hash_is_stable_across_make_and_unmake() {
    let mut position = Position::startpos();
    let original_hash = position.hash();
    let mv = position.legal_moves()[0];
    let undo = position.make_move(mv);
    position.unmake_move(mv, undo);
    assert_eq!(position.hash(), original_hash);
}

#[test]
fn incremental_hash_matches_full_recomputation() {
    let mut position = Position::startpos();
    for mv in position.legal_moves().into_iter().take(8) {
        let undo = position.make_move(mv);
        assert_eq!(position.hash(), position.compute_hash());
        position.unmake_move(mv, undo);
        assert_eq!(position.hash(), position.compute_hash());
    }
}

#[test]
fn parses_official_uci_move_notation() {
    let position = Position::startpos();
    let mv = position.parse_uci_move("h2e2").unwrap();
    assert_eq!(mv, Move::new(index(7, 7), index(4, 7)));
    assert_eq!(mv.to_string(), "h2e2");
}

#[test]
fn legal_capture_moves_match_filtered_legal_moves() {
    let position = Position::from_fen("4k4/9/9/9/4C4/4P4/4r4/9/9/3K5 w").unwrap();
    let captures = position.legal_capture_moves();
    let filtered: Vec<_> = position
        .legal_moves()
        .into_iter()
        .filter(|mv| position.is_capture(*mv))
        .collect();

    assert_eq!(captures, filtered);
}

#[test]
fn legal_capture_moves_to_matches_filtered_capture_moves() {
    let position = Position::from_fen("4k4/9/9/9/4C4/4P4/4r4/9/9/3K5 w").unwrap();
    let target = index(4, 6);
    let targeted = position.legal_capture_moves_to(target);
    let filtered: Vec<_> = position
        .legal_capture_moves()
        .into_iter()
        .filter(|mv| mv.to as usize == target)
        .collect();

    assert_eq!(targeted, filtered);
}

#[test]
fn fast_attack_detection_matches_slow_scan() {
    let samples = [
        Position::startpos(),
        Position::from_fen("4k4/9/9/9/4C4/4P4/4r4/9/9/3K5 w").unwrap(),
        Position::from_fen("4k4/9/9/9/9/9/4R4/9/9/4K4 w").unwrap(),
    ];

    for (sample_index, position) in samples.into_iter().enumerate() {
        for sq in 0..BOARD_SIZE {
            if !position
                .piece_at(sq)
                .is_some_and(|piece| piece.color == Color::Red)
            {
                assert_eq!(
                    position.is_square_attacked(sq, Color::Red),
                    position.is_square_attacked_slow(sq, Color::Red),
                    "sample={sample_index} color=Red sq={sq}"
                );
            }
            if !position
                .piece_at(sq)
                .is_some_and(|piece| piece.color == Color::Black)
            {
                assert_eq!(
                    position.is_square_attacked(sq, Color::Black),
                    position.is_square_attacked_slow(sq, Color::Black),
                    "sample={sample_index} color=Black sq={sq}"
                );
            }
        }
    }
}

#[test]
fn batched_attack_mask_matches_individual_queries() {
    let mut rng = 0xD1B54A32D192ED03u64;
    let mut position = Position::startpos();
    for ply in 0..1_000 {
        let masks = position.attacked_squares_masks();
        for color in [Color::Red, Color::Black] {
            let mask = masks[color_index(color)];
            for square in 0..BOARD_SIZE {
                assert_eq!(
                    mask & (1u128 << square) != 0,
                    position.is_square_attacked(square, color),
                    "ply={ply} color={color:?} square={square} fen={}",
                    position.to_fen()
                );
            }
        }
        let moves = position.legal_moves();
        if moves.is_empty() {
            position = Position::startpos();
            continue;
        }
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        position.make_move(moves[rng as usize % moves.len()]);
    }
}

#[test]
fn dynamic_material_cache_updates_across_make_and_unmake() {
    let mut position = Position::from_fen("4k4/9/9/9/9/9/4R4/9/9/4K4 w").unwrap();
    assert!(position.has_dynamic_material(Color::Red));
    assert!(!position.has_dynamic_material(Color::Black));

    let mv = Move::from_uci("e3e9").unwrap();
    let undo = position.make_move(mv);
    assert!(position.has_dynamic_material(Color::Red));
    assert!(!position.has_dynamic_material(Color::Black));

    position.unmake_move(mv, undo);
    assert!(position.has_dynamic_material(Color::Red));
    assert!(!position.has_dynamic_material(Color::Black));
}

#[test]
fn cannon_check_allows_moving_screen_piece_away() {
    let position = Position::from_fen("4k4/9/9/9/4c4/9/9/4R4/9/4K4 w").unwrap();
    assert!(position.in_check(Color::Red));
    let moves = position.legal_moves();
    assert!(moves.contains(&Move::from_uci("e2a2").unwrap()));
}

#[test]
fn rule_entry_marks_direct_legal_chase() {
    let position = Position::from_fen("4k4/9/9/4n4/9/9/4R4/9/9/4K4 b").unwrap();
    let chased = position.rule_history_entry(Some(Color::Red)).chased_mask;
    assert_ne!(chased & (1u128 << index(4, 3)), 0);
}

#[test]
fn rule_entry_ignores_uncrossed_soldier_chase() {
    let position = Position::from_fen("4k4/9/9/9/4p4/9/4R4/9/9/4K4 b").unwrap();
    let chased = position.rule_history_entry(Some(Color::Red)).chased_mask;
    assert_eq!(chased & (1u128 << index(4, 4)), 0);
}

#[test]
fn rule_entry_ignores_advisor_and_elephant_chase() {
    let advisor = Position::from_fen("4k4/9/9/9/9/4c4/9/4B4/4A4/4K4 b").unwrap();
    let advisor_chased = advisor.rule_history_entry(Some(Color::Black)).chased_mask;
    assert_eq!(advisor_chased & (1u128 << index(4, 8)), 0);

    let elephant = Position::from_fen("4k4/9/9/9/9/4c4/9/4B4/9/4K4 b").unwrap();
    let elephant_chased = elephant.rule_history_entry(Some(Color::Black)).chased_mask;
    assert_eq!(elephant_chased & (1u128 << index(4, 7)), 0);
}

#[test]
fn unprotected_advisor_can_be_a_chase_target() {
    let position = Position::from_fen("4k4/9/3a5/9/9/9/3R5/9/9/K8 b").unwrap();
    let advisor = index(3, 2);
    let chased = position.rule_history_entry(Some(Color::Red)).chased_mask;
    assert_ne!(chased & (1u128 << advisor), 0);
}

#[test]
fn pinned_recapture_does_not_protect_chase_target() {
    let position = Position::from_fen("4k4/9/3nr4/9/9/9/3RR4/9/9/K8 b").unwrap();
    let horse = index(3, 2);
    assert!(position.is_piece_protected(horse, Color::Black));
    let chased = position.rule_history_entry(Some(Color::Red)).chased_mask;
    assert_ne!(chased & (1u128 << horse), 0);
}

#[test]
fn rule_entry_ignores_protected_chase_target() {
    let protected = Position::from_fen("4k4/9/9/4r4/4P4/4R4/9/9/9/4K4 b").unwrap();
    let protected_chased = protected.rule_history_entry(Some(Color::Black)).chased_mask;
    assert_eq!(protected_chased & (1u128 << index(4, 4)), 0);

    let unprotected = Position::from_fen("4k4/9/9/4r4/4P4/9/9/9/9/4K4 b").unwrap();
    let unprotected_chased = unprotected
        .rule_history_entry(Some(Color::Black))
        .chased_mask;
    assert_ne!(unprotected_chased & (1u128 << index(4, 4)), 0);
}

#[test]
fn rule_entry_ignores_cannon_chase_of_protected_crossed_soldier() {
    // 实战残局：红炮d1隔着红仕e1可以打黑卒f1，但黑炮f8隔着
    // 黑仕f7保护该卒。有根的过河卒不能被记为红方长捉目标。
    let position = Position::from_fen("4k4/4ac3/5a3/9/9/9/9/4B4/3CAp3/4K4 w").unwrap();
    let soldier = index(5, 8);
    assert!(position.is_piece_protected(soldier, Color::Black));

    let chased = position.rule_history_entry(Some(Color::Red)).chased_mask;
    assert_eq!(chased & (1u128 << soldier), 0);
}

fn test_rule_entry(
    hash: u64,
    side_to_move: Color,
    mover: Option<Color>,
    gives_check: bool,
    chased_mask: u128,
) -> RuleHistoryEntry {
    RuleHistoryEntry {
        hash,
        side_to_move,
        mover,
        gives_check,
        chased_mask,
        mv: None,
        captured: None,
        rule60_clock: 0,
    }
}

fn test_rule_move_entry(
    hash: u64,
    side_to_move: Color,
    mover: Color,
    mv: Move,
    chased_mask: u128,
) -> RuleHistoryEntry {
    RuleHistoryEntry {
        mv: Some(mv),
        ..test_rule_entry(hash, side_to_move, Some(mover), false, chased_mask)
    }
}

#[test]
fn long_chase_tracks_one_piece_across_squares() {
    let mut history = vec![
        test_rule_entry(1, Color::Red, None, false, 0),
        test_rule_move_entry(2, Color::Black, Color::Red, Move::new(0, 1), 1 << 10),
        test_rule_move_entry(3, Color::Red, Color::Black, Move::new(10, 11), 0),
        test_rule_move_entry(4, Color::Black, Color::Red, Move::new(1, 0), 1 << 11),
        test_rule_move_entry(1, Color::Red, Color::Black, Move::new(11, 10), 0),
    ];
    history.extend_from_within(1..);
    assert_eq!(
        Position::rule_outcome(&history),
        Some(RuleOutcome::Win(Color::Black))
    );
}

#[test]
fn alternating_same_kind_targets_are_not_one_long_chase() {
    let mut history = vec![
        test_rule_entry(1, Color::Red, None, false, 0),
        test_rule_move_entry(2, Color::Black, Color::Red, Move::new(0, 1), 1 << 10),
        test_rule_move_entry(3, Color::Red, Color::Black, Move::new(20, 21), 0),
        test_rule_move_entry(4, Color::Black, Color::Red, Move::new(1, 0), 1 << 11),
        test_rule_move_entry(1, Color::Red, Color::Black, Move::new(21, 20), 0),
    ];
    history.extend_from_within(1..);
    assert_eq!(
        Position::rule_outcome(&history),
        Some(RuleOutcome::Draw(RuleDrawReason::Repetition))
    );
}

#[test]
fn five_long_check_cycles_lose() {
    let mut history = vec![test_rule_entry(1, Color::Red, None, false, 0)];
    for _ in 0..5 {
        history.push(test_rule_entry(2, Color::Black, Some(Color::Red), true, 0));
        history.push(test_rule_entry(3, Color::Red, Some(Color::Black), false, 0));
        history.push(test_rule_entry(4, Color::Black, Some(Color::Red), true, 0));
        history.push(test_rule_entry(1, Color::Red, Some(Color::Black), false, 0));
    }
    assert_eq!(
        Position::rule_outcome(&history),
        Some(RuleOutcome::Win(Color::Black))
    );
}

#[test]
fn one_long_check_cycle_is_not_terminal() {
    let mut history = vec![
        test_rule_entry(1, Color::Red, None, false, 0),
        test_rule_entry(2, Color::Black, Some(Color::Red), true, 0),
        test_rule_entry(3, Color::Red, Some(Color::Black), false, 0),
        test_rule_entry(4, Color::Black, Some(Color::Red), true, 0),
        test_rule_entry(1, Color::Red, Some(Color::Black), false, 0),
    ];
    assert_eq!(Position::rule_outcome(&history), None);
    history.extend_from_within(1..);
    assert_eq!(
        Position::rule_outcome(&history),
        Some(RuleOutcome::Win(Color::Black))
    );
}

#[test]
fn three_long_check_cycles_lose() {
    let mut history = vec![test_rule_entry(1, Color::Red, None, false, 0)];
    for _ in 0..3 {
        history.push(test_rule_entry(2, Color::Black, Some(Color::Red), true, 0));
        history.push(test_rule_entry(3, Color::Red, Some(Color::Black), false, 0));
        history.push(test_rule_entry(4, Color::Black, Some(Color::Red), true, 0));
        history.push(test_rule_entry(1, Color::Red, Some(Color::Black), false, 0));
    }
    assert_eq!(
        Position::rule_outcome(&history),
        Some(RuleOutcome::Win(Color::Black))
    );
}

#[test]
fn mutual_long_check_cycles_draw() {
    let mut history = vec![test_rule_entry(1, Color::Red, None, false, 0)];
    for _ in 0..5 {
        history.push(test_rule_entry(2, Color::Black, Some(Color::Red), true, 0));
        history.push(test_rule_entry(3, Color::Red, Some(Color::Black), true, 0));
        history.push(test_rule_entry(4, Color::Black, Some(Color::Red), true, 0));
        history.push(test_rule_entry(1, Color::Red, Some(Color::Black), true, 0));
    }
    assert_eq!(
        Position::rule_outcome(&history),
        Some(RuleOutcome::Draw(RuleDrawReason::MutualLongCheck))
    );
}

#[test]
fn five_long_chase_cycles_lose() {
    let mut history = vec![test_rule_entry(10, Color::Red, None, false, 0)];
    for _ in 0..5 {
        history.push(test_rule_entry(
            11,
            Color::Black,
            Some(Color::Red),
            false,
            1 << 20,
        ));
        history.push(test_rule_entry(
            12,
            Color::Red,
            Some(Color::Black),
            false,
            0,
        ));
        history.push(test_rule_entry(
            13,
            Color::Black,
            Some(Color::Red),
            false,
            1 << 20,
        ));
        history.push(test_rule_entry(
            10,
            Color::Red,
            Some(Color::Black),
            false,
            0,
        ));
    }
    assert_eq!(
        Position::rule_outcome(&history),
        Some(RuleOutcome::Win(Color::Black))
    );
}

#[test]
fn mixed_check_and_chase_cycle_requires_the_forcing_side_to_change() {
    let mut history = vec![
        test_rule_entry(1, Color::Red, None, false, 0),
        test_rule_entry(2, Color::Black, Some(Color::Red), true, 0),
        test_rule_entry(3, Color::Red, Some(Color::Black), false, 0),
        test_rule_entry(4, Color::Black, Some(Color::Red), false, 1 << 20),
        test_rule_entry(1, Color::Red, Some(Color::Black), false, 0),
    ];
    let cycle = history[1..].to_vec();
    assert_eq!(Position::rule_outcome(&history), None);
    for _ in 0..2 {
        history.extend_from_slice(&cycle);
        assert_eq!(Position::rule_outcome(&history), None);
    }
    history.extend_from_slice(&cycle);
    assert_eq!(
        Position::rule_outcome(&history),
        Some(RuleOutcome::Win(Color::Black))
    );
}

#[test]
fn mixed_check_and_idle_cycle_remains_a_draw() {
    let mut history = vec![
        test_rule_entry(1, Color::Red, None, false, 0),
        test_rule_entry(2, Color::Black, Some(Color::Red), true, 0),
        test_rule_entry(3, Color::Red, Some(Color::Black), false, 0),
        test_rule_entry(4, Color::Black, Some(Color::Red), false, 0),
        test_rule_entry(1, Color::Red, Some(Color::Black), false, 0),
    ];
    history.extend_from_within(1..);
    assert_eq!(
        Position::rule_outcome(&history),
        Some(RuleOutcome::Draw(RuleDrawReason::Repetition))
    );
}

#[test]
fn tiantian_check_and_chase_loop_filters_the_black_rook_repeat() {
    let mut position =
        Position::from_fen("3k2b2/9/3a5/p7p/9/2P6/P2n4P/2C1R4/1r1KN4/3A1A3 w - - 3 36").unwrap();
    let mut history = position.initial_rule_history();
    let cycle = "d1d2 b1b0 d2d1 b0b1";
    for (index, text) in cycle.split_whitespace().cycle().take(21).enumerate() {
        let mv = position.parse_uci_move(text).unwrap();
        let allowed = position.legal_moves_with_rules(&history).contains(&mv);
        // 允许三个完整循环，第四个循环（原棋谱第 86 步）黑方必须变招。
        assert_eq!(allowed, index < 15 || position.side_to_move() == Color::Red);
        history.push(position.rule_history_entry_after_move(mv));
        position.make_move(mv);
        assert_eq!(
            position.rule_outcome_with_history(&history),
            (index >= 15).then_some(RuleOutcome::Win(Color::Red))
        );
    }
    let repeat = position.parse_uci_move("b1b0").unwrap();
    let moves = position.legal_moves_with_rules(&history);
    assert!(!moves.contains(&repeat));
    assert!(!moves.is_empty(), "黑方应有可用的变招");
}

#[test]
fn one_long_chase_cycle_is_not_terminal() {
    let mut history = vec![
        test_rule_entry(10, Color::Red, None, false, 0),
        test_rule_entry(11, Color::Black, Some(Color::Red), false, 1 << 20),
        test_rule_entry(12, Color::Red, Some(Color::Black), false, 0),
        test_rule_entry(13, Color::Black, Some(Color::Red), false, 1 << 20),
        test_rule_entry(10, Color::Red, Some(Color::Black), false, 0),
    ];
    assert_eq!(Position::rule_outcome(&history), None);
    history.extend_from_within(1..);
    assert_eq!(
        Position::rule_outcome(&history),
        Some(RuleOutcome::Win(Color::Black))
    );
}

#[test]
fn three_long_chase_cycles_lose() {
    let mut history = vec![test_rule_entry(10, Color::Red, None, false, 0)];
    for _ in 0..3 {
        history.push(test_rule_entry(
            11,
            Color::Black,
            Some(Color::Red),
            false,
            1 << 20,
        ));
        history.push(test_rule_entry(
            12,
            Color::Red,
            Some(Color::Black),
            false,
            0,
        ));
        history.push(test_rule_entry(
            13,
            Color::Black,
            Some(Color::Red),
            false,
            1 << 20,
        ));
        history.push(test_rule_entry(
            10,
            Color::Red,
            Some(Color::Black),
            false,
            0,
        ));
    }
    assert_eq!(
        Position::rule_outcome(&history),
        Some(RuleOutcome::Win(Color::Black))
    );
}

#[test]
fn chased_piece_escape_does_not_make_mutual_long_chase() {
    let mut position =
        Position::from_fen("r3kab1r/4a4/2n1bc2n/p1p1p1pc1/8p/5NP2/P1P1P3P/2N1C2C1/8R/1RBAKAB2 w")
            .unwrap();
    let mut history = position.initial_rule_history();
    let mut outcome = None;
    for text in [
        "f4d5", "c6c5", "d5c7", "f7c7", "i1d1", "a9d9", "d1d9", "e8d9", "b0b4", "i9i8", "c3c4",
        "i8d8", "c4c5", "e7c5", "b4f4", "i7h5", "f4f5", "h6h2", "f5h5", "c7c2", "h5c5", "d8d3",
        "e3e4", "d3e3", "a3a4", "c2c3", "c5i5", "e3e4", "i5c5", "c3b3", "c5c3", "b3b5", "c3c5",
        "b5b0", "c5h5", "h2f2", "h5b5", "b0a0", "b5b0", "a0a3", "b0b3", "a3a0", "b3a3", "a0b0",
        "a3b3", "b0a0", "b3a3", "a0b0", "a3b3", "b0a0", "b3a3", "a0b0", "a3b3", "b0a0",
    ] {
        if let Some(o) = position.rule_outcome_with_history(&history) {
            outcome = Some(o);
            break;
        }
        let mv = position.parse_uci_move(text).unwrap();
        assert!(position.legal_moves_with_rules(&history).contains(&mv));
        history.push(position.rule_history_entry_after_move(mv));
        position.make_move(mv);
    }

    // 黑方被捉的车在逃，因此不是"互相长捉"和棋；红方是唯一长捉方所以判红负（黑胜）。
    // 同一局面第三次出现后判定长捉。
    assert_eq!(outcome, Some(RuleOutcome::Win(Color::Black)));
}

#[test]
fn mutual_long_chase_cycles_draw() {
    let mut history = vec![test_rule_entry(10, Color::Red, None, false, 0)];
    for _ in 0..5 {
        history.push(test_rule_entry(
            11,
            Color::Black,
            Some(Color::Red),
            false,
            1 << 20,
        ));
        history.push(test_rule_entry(
            12,
            Color::Red,
            Some(Color::Black),
            false,
            1 << 21,
        ));
        history.push(test_rule_entry(
            13,
            Color::Black,
            Some(Color::Red),
            false,
            1 << 20,
        ));
        history.push(test_rule_entry(
            10,
            Color::Red,
            Some(Color::Black),
            false,
            1 << 21,
        ));
    }
    assert_eq!(
        Position::rule_outcome(&history),
        Some(RuleOutcome::Draw(RuleDrawReason::MutualLongChase))
    );
}

#[test]
fn one_long_check_cycle_allows_the_next_repeated_check() {
    let mut position =
        Position::from_fen("2Rakab2/8r/4c1n2/p3p1p1p/2p6/9/P3P3P/1CN1NC3/9/1RBAKArc1 b - - 0 1")
            .unwrap();
    let mut history = position.initial_rule_history();
    for text in ["g0g1", "f0e1", "g1g0", "e1f0"] {
        let mv = position.parse_uci_move(text).unwrap();
        history.push(position.rule_history_entry_after_move(mv));
        position.make_move(mv);
    }
    assert_eq!(
        position.to_fen(),
        "2Rakab2/8r/4c1n2/p3p1p1p/2p6/9/P3P3P/1CN1NC3/9/1RBAKArc1 b - - 4 1"
    );
    assert_eq!(position.rule_outcome_with_history(&history), None);
    assert_eq!(position.legal_moves().len(), 44);
    let repeated_check = position.parse_uci_move("g0g1").unwrap();
    assert!(position.legal_moves().contains(&repeated_check));
    assert!(
        position
            .legal_moves_with_rules(&history)
            .contains(&repeated_check)
    );
}

#[test]
fn horse_repeatedly_chasing_rook_is_forbidden() {
    let mut position =
        Position::from_fen("2bak4/4a4/2ncb2c1/p3p2CP/9/1N1RP4/P5r2/4C4/9/2BAKA3 b - - 0 1")
            .unwrap();
    let mut history = position.initial_rule_history();
    let moves = ["c7b5", "d4d5", "b5c7", "d5d4"];
    for _ in 0..3 {
        for text in moves {
            let mv = position.parse_uci_move(text).unwrap();
            history.push(position.rule_history_entry_after_move(mv));
            position.make_move(mv);
        }
    }

    let mv = position.parse_uci_move("c7b5").unwrap();
    assert!(position.legal_moves().contains(&mv));
    let mut next = position.clone();
    let mut next_history = history.clone();
    next_history.push(position.rule_history_entry_after_move(mv));
    next.make_move(mv);
    assert_eq!(
        next.rule_outcome_with_history(&next_history),
        Some(RuleOutcome::Win(Color::Red))
    );
    assert!(!position.legal_moves_with_rules(&history).contains(&mv));
}

#[test]
fn rook_repeated_long_check_is_forbidden() {
    let mut position =
        Position::from_fen("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1")
            .unwrap();
    let mut history = position.initial_rule_history();
    let moves = "g3g4 b7e7 b0c2 b9c7 a0b0 a9b9 h0g2 c6c5 i0i1 h9i7 i1f1 h7g7 g2h4 g6g5 f1f4 g5g4 f4g4 g7g8 b2b6 i6i5 g4d4 f9e8 b6c6 e7g7 d4b4 g8g0 f0e1 g0i0 e0f0 g7f7 b4b9 c7b9 b0b9 g9e7 b9b4 i9g9 h2g2 g9g3 c0e2 i7h5 f0e0 g3h3 e1f2 h3h0 e0e1 h0h1 e1e0 h1h0 e0e1 h0h1 e1e0 h1h0 e0e1 h0h1";
    let mut first_forbidden = None;
    for text in moves.split_whitespace() {
        let mv = position.parse_uci_move(text).unwrap();
        if !position.legal_moves_with_rules(&history).contains(&mv) {
            first_forbidden = Some(text);
            break;
        }
        history.push(position.rule_history_entry_after_move(mv));
        position.make_move(mv);
    }
    assert_eq!(first_forbidden, Some("h1h0"));
}

#[test]
fn cannon_repetition_chasing_advisor_is_not_long_chase_loss() {
    let mut position =
        Position::from_fen("r2akab1r/9/1cn1b1nc1/p1p1p3p/6p2/2P3P2/P3P3P/C1N3C2/9/R1BAKABNR b")
            .unwrap();
    let mut history = position.initial_rule_history();
    for text in [
        "g7f5", "g4g5", "e7g5", "a0b0", "a9b9", "b0b6", "b7a7", "b6c6", "g5e7", "h0i2", "a7a8",
        "i0h0", "a8c8", "c4c5", "c8c6", "c5c6", "h7f7", "h0h5", "b9b5", "i2g3", "b5c5", "g3f5",
        "c5c2", "c6c7", "c2g2", "g0e2", "g2g5", "h5g5", "e7g5", "c7d7", "d9e8", "d7d8", "e6e5",
        "a2a6", "f7f8", "a6e6", "e8f7", "f5e7", "f9e8", "d8e8", "f7e8", "e7g8", "e8f7", "g8i9",
        "f8i8", "i9g8", "e9d9", "a3a4", "d9d8", "a4a5", "g9e7", "a5b5", "d8d7", "b5b6", "d7d8",
        "b6c6", "f7e8", "c6c7", "i8i7", "c7c8", "d8d7", "g8f6", "e7c5", "f6e8", "g5e7", "e8g7",
        "i6i5", "g7e8", "i7i6", "e8c7", "i6i7", "e3e4", "e7c9", "e4e5", "i7c7", "e5d5", "d7e7",
        "d5c5", "c9a7", "c5c6", "c7d7", "c6c7", "d7d3", "c8d8", "d3h3", "c7c8", "h3h8", "f0e1",
        "h8i8", "e0f0", "i8i3", "f0f1", "i3e3", "f1f0", "i5i4", "f0f1", "i4h4", "f1f0", "h4h3",
        "f0f1", "h3g3", "f1f0", "a7c5", "f0f1", "e3e4", "f1f0", "e4e3", "f0f1", "e3e4", "f1f0",
        "e4e3", "f0f1", "e3e4", "f1f0",
    ] {
        let mv = position.parse_uci_move(text).unwrap();
        assert!(
            position.legal_moves_with_rules(&history).contains(&mv),
            "{text} was filtered at {}",
            position.to_fen()
        );
        history.push(position.rule_history_entry_after_move(mv));
        position.make_move(mv);
    }

    let mv = position.parse_uci_move("e4e3").unwrap();
    assert!(position.legal_moves().contains(&mv));
    assert!(position.legal_moves_with_rules(&history).contains(&mv));
}

#[test]
fn single_side_long_chase_survives_opponents_interleaved_check() {
    let mut position =
        Position::from_fen("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1")
            .unwrap();
    let mut history = position.initial_rule_history();
    let moves = "b2e2 b9c7 b0c2 a9b9 a0b0 c6c5 b0b4 h9g7 h0i2 h7h5 h2f2 i9h9 i0h0 c7b5 b4d4 b5c3 f0e1 c9e7 g3g4 f9e8 h0h4 b7c7 d4d6 c3b5 c2a1 c7c9 d6c6 c5c4 g4g5 g6g5 c6c4 b5a3 c4c7 a3b5 c7c4 g7f5 c4b4 f5e3 f2h2 e3d5 b4e4 b5c3 e4c4 h9h6 h2h5 c3e2 c0e2 d5f6 a1c0 b9b6 h4f4 f6h5 f4h4 h6h9 c0d2 h5g7 h4h9 g7h9 d2e4 c9b9 e4f6 h9i7 c4f4 e6e5 i2g3 b9b8 g3h5 b6d6 h5i7 g9i7 f6h7 g5g4 f4g4 d6h6 g4b4 b8c8 b4c4 c8b8 c4c8 b8b0 c8c0 b0b3 c0c3 b3b0 c3c0 b0b3 c0c3 b3b0 c3c0 b0b3 c0c3 b3b0 c3c0 b0b3";
    let mut first_forbidden = None;
    for (index, text) in moves.split_whitespace().enumerate() {
        let mv = position.parse_uci_move(text).unwrap();
        let allowed = position.legal_moves_with_rules(&history).contains(&mv);
        if !allowed && first_forbidden.is_none() {
            first_forbidden = Some(index + 1);
        }
        let entry = position.rule_history_entry_after_move(mv);
        history.push(entry);
        position.make_move(mv);
    }
    assert_eq!(first_forbidden, Some(89));
    assert_eq!(position.side_to_move(), Color::Red);
    assert_eq!(
        position.rule_outcome_with_history(&history),
        Some(RuleOutcome::Win(Color::Black))
    );
    let repeat = position.parse_uci_move("c0c3").unwrap();
    assert!(!position.legal_moves_with_rules(&history).contains(&repeat));
}

#[test]
fn three_repetition_cycles_without_forcing_draw() {
    let history = vec![
        test_rule_entry(21, Color::Red, None, false, 0),
        test_rule_entry(22, Color::Black, Some(Color::Red), false, 0),
        test_rule_entry(23, Color::Red, Some(Color::Black), false, 0),
        test_rule_entry(21, Color::Red, Some(Color::Black), false, 0),
        test_rule_entry(22, Color::Black, Some(Color::Red), false, 0),
        test_rule_entry(23, Color::Red, Some(Color::Black), false, 0),
        test_rule_entry(21, Color::Red, Some(Color::Black), false, 0),
        test_rule_entry(22, Color::Black, Some(Color::Red), false, 0),
        test_rule_entry(23, Color::Red, Some(Color::Black), false, 0),
        test_rule_entry(21, Color::Red, Some(Color::Black), false, 0),
        test_rule_entry(22, Color::Black, Some(Color::Red), false, 0),
        test_rule_entry(23, Color::Red, Some(Color::Black), false, 0),
        test_rule_entry(21, Color::Red, Some(Color::Black), false, 0),
    ];
    assert_eq!(
        Position::rule_outcome(&history),
        Some(RuleOutcome::Draw(RuleDrawReason::Repetition))
    );
}

#[test]
fn five_repetition_cycles_without_forcing_draw() {
    let mut history = vec![test_rule_entry(21, Color::Red, None, false, 0)];
    for _ in 0..5 {
        history.push(test_rule_entry(
            22,
            Color::Black,
            Some(Color::Red),
            false,
            0,
        ));
        history.push(test_rule_entry(
            23,
            Color::Red,
            Some(Color::Black),
            false,
            0,
        ));
        history.push(test_rule_entry(
            21,
            Color::Red,
            Some(Color::Black),
            false,
            0,
        ));
    }
    assert_eq!(
        Position::rule_outcome(&history),
        Some(RuleOutcome::Draw(RuleDrawReason::Repetition))
    );
}

#[test]
fn double_check_can_be_evaded_by_moving_cannon_screen_to_capture_checker() {
    let position =
        Position::from_fen("4k4/4a1c2/b1nN1a3/2C5p/7r1/2P1C4/P3P1n1P/4B1N2/4A4/2BAK4 b").unwrap();
    assert!(position.in_check(Color::Black));

    let mv = Move::from_uci("e8d7").unwrap();
    assert!(position.is_legal_move(mv));
    assert!(position.legal_moves().contains(&mv));
}

fn slow_legal_moves(position: &Position) -> Vec<Move> {
    let pseudo = position.pseudo_legal_moves();
    let mut work = position.clone();
    let mut legal = Vec::new();
    for mv in pseudo {
        let undo = work.make_move(mv);
        if !work.in_check(position.side_to_move()) {
            legal.push(mv);
        }
        work.unmake_move(mv, undo);
    }
    legal
}

#[test]
fn fast_legal_moves_match_slow_on_vs_pikafish_repetition_game() {
    let mut position =
        Position::from_fen("r2akab1r/c8/2n1b2c1/p3p3p/7n1/2R6/P3P3P/C1N6/6C2/2BAKABNR w").unwrap();
    let mut history = position.initial_rule_history();
    let moves = "c4c7 h7c7 i0i2 c7c0 d0e1 c0a0 i2d2 a9b9 d2d8 a8a7 g1g7 a7g7 e0d0 g7g1 a2b2 a0a2 c2b4 b9b4 d8d6 f9e8 d6d8 b4b9 d8d6 g1f1 d6d8 b9c9 d0e0 c9c0 e1d0 c0c9 b2e2 a2a0 d0e1 c9c0 e1d0 c0c1 d0e1 c1c0 e1d0 c0c1 d0e1 i9i7 e2e6 c1c0 e1d0 c0c1 d0e1 c1c0 e1d0 c0c9 d0e1 e9f9 d8e8 c9c0 e1d0 c0c1 d0e1 d9e8 h0g2 c1c0 e1d0 c0c1 d0e1 c1c0 e1d0 i7g7 e6e5 c0c1 d0e1 c1c0 e1d0 c0c1 d0e1 f1f6 e0d0 c1c0 d0d1 c0c1 d1d0 c1c0 d0d1 f6f1 e1f2 c0c1 d1d0 c1c0 d0d1 c0c1 d1d0 h5i7 g2h4 c1c0 d0d1 c0c1 d1d0 c1c0 d0d1 c0c9 h4f5 c9c1 d1d0 c1c0 d0d1 c0c1 d1d0 c1c0 d0d1 g7g0 f5g7 i7g8 g7e6 c0c1 d1d2 c1c2 d2d1 c2c1 d1d2 c1c2 d2d1 c2c9 e6c7 g0g3 f0e1";

    for (ply, text) in moves.split_whitespace().enumerate() {
        let fast = position.legal_moves();
        let slow = slow_legal_moves(&position);
        let mut fast_sorted = fast.clone();
        let mut slow_sorted = slow.clone();
        fast_sorted.sort_by_key(|mv| (mv.from, mv.to));
        slow_sorted.sort_by_key(|mv| (mv.from, mv.to));
        assert_eq!(
            fast_sorted,
            slow_sorted,
            "fast legal mismatch before ply {} move {} at {}",
            ply + 1,
            text,
            position.to_fen()
        );
        let mv = Move::from_uci(text).unwrap();
        if !fast.contains(&mv) {
            assert_eq!(
                text,
                "f0e1",
                "unexpected illegal move {text} before ply {} at {}",
                ply + 1,
                position.to_fen()
            );
            assert!(
                !slow.contains(&mv),
                "slow legality still accepts final illegal move {text}"
            );
            return;
        }
        assert!(
            fast.contains(&mv),
            "move {text} is illegal before ply {} at {}",
            ply + 1,
            position.to_fen()
        );
        history.push(position.rule_history_entry_after_move(mv));
        position.make_move(mv);
    }
}

#[test]
fn fast_legal_moves_match_slow_on_random_games() {
    let mut rng: u64 = 0xD1B54A32D192ED03;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    for game in 0..30u64 {
        let mut position = Position::startpos();
        for ply in 0..200usize {
            let mut fast = position.legal_moves();
            let mut slow = slow_legal_moves(&position);
            fast.sort_by_key(|mv| (mv.from, mv.to));
            slow.sort_by_key(|mv| (mv.from, mv.to));
            assert_eq!(
                fast,
                slow,
                "fast legal mismatch game {game} ply {ply} at {}",
                position.to_fen()
            );
            if fast.is_empty() {
                break;
            }
            let mv = fast[(next() as usize) % fast.len()];
            position.make_move(mv);
        }
    }
}

#[test]
fn gives_check_fast_matches_bruteforce() {
    let mut rng: u64 = 0x9E3779B97F4A7C15;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    for game in 0..30u64 {
        let mut position = Position::startpos();
        for ply in 0..200usize {
            let moves = position.legal_moves();
            if moves.is_empty() {
                break;
            }
            let mv = moves[(next() as usize) % moves.len()];
            let mut clone = position.clone();
            let old = clone.gives_check_after_move(mv);
            let fast = position.gives_check_after_move_fast(mv);
            assert_eq!(
                old,
                fast,
                "fast mismatch game {game} ply {ply} mv {:?}\n{}",
                mv,
                position.to_fen()
            );
            let mut c2 = position.clone();
            let captured = c2.make_move_board_only(mv);
            let real = c2.in_check(position.side_to_move().opposite());
            c2.unmake_move_board_only(mv, captured);
            assert_eq!(
                fast,
                real,
                "real mismatch game {game} ply {ply} mv {:?}\n{}",
                mv,
                position.to_fen()
            );
            position.rule_history_entry_after_move(mv);
            position.make_move(mv);
        }
    }
}

/// `is_square_attacked_after_move` 必须和"真的走一步再看 `is_square_attacked`"一致。
///
/// 每个抽样走法都**扫全部 90 格 × 两种颜色**（而不是只问落点），这样才会真的走到那些
/// 只在特定目标几何下才成立的分支：飞将、象眼、过河兵侧移、九宫内的士/将。
#[test]
fn attacked_after_move_matches_make_move_on_random_games() {
    let mut rng: u64 = 0x243F_6A88_85A3_08D3;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut compared = 0usize;
    for game in 0..8u64 {
        let mut position = Position::startpos();
        for ply in 0..160usize {
            let moves = position.legal_moves();
            if moves.is_empty() {
                break;
            }
            for sample in 0..moves.len().min(3) {
                let mv = moves[(next() as usize).wrapping_add(sample) % moves.len()];
                let mut after = position.clone();
                after.make_move(mv);
                for color in [Color::Red, Color::Black] {
                    for target in 0..BOARD_SIZE {
                        assert_eq!(
                            position.is_square_attacked_after_move(target, color, mv),
                            after.is_square_attacked(target, color),
                            "target={target} by={color:?} mv={mv} game={game} ply={ply}\n{}",
                            position.to_fen()
                        );
                        compared += 1;
                    }
                }
            }
            let mv = moves[(next() as usize) % moves.len()];
            position.make_move(mv);
        }
    }
    assert!(compared > 500_000, "coverage too small: {compared}");
}

/// 逐条覆盖"只有 `from` 腾空与 `to` 被占两处占位变化"会改变攻击关系的情形。
///
/// 每条都同时断言(1)走前值、(2)真 make_move 之后的真值、(3)虚拟占位查询的值，
/// 并要求走前走后必须翻转——否则这条用例没有真的测到东西。
/// `from_canonical_piece_squares` 不做合法性校验，因此能覆盖飞将这类合法走法
/// 到不了的边界。
#[test]
fn attacked_after_move_covers_from_and_to_occupancy_cases() {
    // 棋子类别索引：0..7 为红方（将/士/象/马/车/炮/兵），7..14 为黑方。
    const RED_GENERAL: usize = 0;
    const RED_ROOK: usize = 4;
    const RED_CANNON: usize = 5;
    const RED_SOLDIER: usize = 6;
    const BLACK_GENERAL: usize = 7;
    const BLACK_ELEPHANT: usize = 9;
    const BLACK_HORSE: usize = 10;
    const BLACK_ROOK: usize = 11;
    const BLACK_CANNON: usize = 12;

    let check = |pieces: &[(usize, usize)],
                 target: usize,
                 by: Color,
                 mv: Move,
                 before: bool,
                 after_expected: bool| {
        assert_ne!(before, after_expected, "case must flip: mv={mv}");
        let position = Position::from_canonical_piece_squares(pieces);
        assert_eq!(
            position.is_square_attacked(target, by),
            before,
            "pre-move mismatch: target={target} by={by:?} mv={mv}"
        );
        let mut after = position.clone();
        after.make_move(mv);
        assert_eq!(
            after.is_square_attacked(target, by),
            after_expected,
            "oracle mismatch: target={target} by={by:?} mv={mv}"
        );
        assert_eq!(
            position.is_square_attacked_after_move(target, by, mv),
            after_expected,
            "virtual mismatch: target={target} by={by:?} mv={mv}"
        );
    };

    // 1. `from` 原本是炮架：走后炮失去对落点的攻击。
    check(
        &[
            (BLACK_CANNON, index(0, 0)),
            (RED_ROOK, index(0, 2)),
            (RED_GENERAL, index(3, 9)),
            (BLACK_GENERAL, index(4, 0)),
        ],
        index(0, 4),
        Color::Black,
        Move::new(index(0, 2), index(0, 4)),
        true,
        false,
    );

    // 2. `from` 原本挡住车线：走后车获得对落点的攻击。
    check(
        &[
            (BLACK_ROOK, index(0, 0)),
            (RED_ROOK, index(0, 2)),
            (RED_GENERAL, index(3, 9)),
            (BLACK_GENERAL, index(4, 0)),
        ],
        index(0, 4),
        Color::Black,
        Move::new(index(0, 2), index(0, 4)),
        false,
        true,
    );

    // 3. `to` 被占之后给敌方炮造出炮架（目标格不是落点）。
    check(
        &[
            (BLACK_CANNON, index(0, 0)),
            (RED_SOLDIER, index(0, 6)),
            (RED_GENERAL, index(3, 9)),
            (BLACK_GENERAL, index(4, 0)),
        ],
        index(0, 4),
        Color::Black,
        Move::new(index(0, 6), index(0, 2)),
        false,
        true,
    );

    // 4. 马腿恰好是 `from`：走后马松开（目标格不是落点，且与 from 不共线）。
    check(
        &[
            (BLACK_HORSE, index(1, 1)),
            (RED_SOLDIER, index(1, 2)),
            (RED_GENERAL, index(3, 9)),
            (BLACK_GENERAL, index(4, 0)),
        ],
        index(0, 3),
        Color::Black,
        Move::new(index(1, 2), index(1, 3)),
        false,
        true,
    );

    // 5. 象眼恰好是 `from`：走后象松开。
    check(
        &[
            (BLACK_ELEPHANT, index(0, 0)),
            (RED_SOLDIER, index(1, 1)),
            (RED_GENERAL, index(3, 9)),
            (BLACK_GENERAL, index(4, 0)),
        ],
        index(2, 2),
        Color::Black,
        Move::new(index(1, 1), index(1, 0)),
        false,
        true,
    );

    // 6. 飞将：`from` 腾空后同线打通（目标格不是落点）。
    check(
        &[
            (BLACK_GENERAL, index(4, 0)),
            (RED_GENERAL, index(4, 7)),
            (RED_SOLDIER, index(4, 3)),
        ],
        index(4, 7),
        Color::Black,
        Move::new(index(4, 3), index(3, 3)),
        false,
        true,
    );

    // 7. 飞将落在己方将的新格子上（目标格就是落点）。
    check(
        &[(BLACK_GENERAL, index(4, 0)), (RED_GENERAL, index(4, 8))],
        index(4, 7),
        Color::Black,
        Move::new(index(4, 8), index(4, 7)),
        false,
        true,
    );

    // 8. 己方炮失去炮架：己方对目标格的"保护"消失（`by` 是走子方）。
    check(
        &[
            (RED_CANNON, index(0, 9)),
            (RED_SOLDIER, index(0, 7)),
            (RED_GENERAL, index(3, 9)),
            (BLACK_GENERAL, index(4, 0)),
        ],
        index(0, 5),
        Color::Red,
        Move::new(index(0, 7), index(1, 7)),
        true,
        false,
    );

    // 9. 炮失去炮架、但**另一条射线上还有攻击者**：真值保持不变。
    //
    // 这条与第 1 条在"走前 mask 位 / to→from 射线状态 / 跃子查表"三样输入上完全相同，
    // 真值却相反（第 1 条由真变假，这里仍为真）。因此"只看一条 to→from 射线加跃子"
    // 的修正函数不可能精确，必须有完整的走后攻击查询。它是防止将来退回近似实现的关键
    // 回归用例。
    {
        let pieces = [
            (BLACK_CANNON, index(0, 0)),
            (BLACK_GENERAL, index(3, 0)),
            (RED_ROOK, index(0, 2)),
            (BLACK_ROOK, index(4, 4)),
            (RED_GENERAL, index(3, 9)),
        ];
        let position = Position::from_canonical_piece_squares(&pieces);
        let mv = Move::new(index(0, 2), index(0, 4));
        let target = index(0, 4);
        assert!(position.is_square_attacked(target, Color::Black));
        let mut after = position.clone();
        after.make_move(mv);
        assert!(after.is_square_attacked(target, Color::Black));
        assert!(position.is_square_attacked_after_move(target, Color::Black, mv));
    }
}

#[test]
fn stalemate_precedes_natural_move_limit() {
    let position = Position::from_fen("4k4/3R1R3/9/9/9/9/9/9/9/3K5 b - - 120 1").unwrap();
    assert!(!position.in_check(Color::Black));
    assert!(position.legal_moves().is_empty());
    assert_eq!(
        position.rule_outcome_with_history(&position.initial_rule_history()),
        Some(RuleOutcome::Win(Color::Red)),
    );
}

#[test]
fn checkmate_precedes_natural_move_limit() {
    let position = Position::from_fen("3RkR3/9/2N6/9/9/9/9/9/9/3K5 b - - 120 1").unwrap();
    assert!(position.in_check(Color::Black));
    assert!(position.legal_moves().is_empty());
    assert_eq!(
        position.rule_outcome_with_history(&position.initial_rule_history()),
        Some(RuleOutcome::Win(Color::Red)),
    );
}

/// 合法性过滤的快路径必须和 `make_move_board_only` + `in_check` 逐着一致。
///
/// 快路径的前提是"当前不在被将军状态"，此时只有 `from` 空出来可能让国王挨打，
/// 而且新增的占用只可能挡线、不可能造出攻击。这里对随机对局里每一步非国王着法
/// 逐着对拍，覆盖面大于实际走快路径的场合（实际只对落在 safety mask 里的着法走）。
#[test]
fn vacate_attack_check_matches_make_unmake() {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut compared = 0usize;
    for _game in 0..12 {
        let mut position = Position::startpos();
        for ply in 0..240 {
            let side = position.side_to_move();
            if let Some(king_sq) = position.find_general(side) {
                let enemy = side.opposite();
                let unblock = position.leaper_unblock_mask(king_sq, enemy);
                let checked = position.in_check(side);
                for &mv in &position.legal_moves() {
                    let from = mv.from as usize;
                    if checked || from == king_sq {
                        continue;
                    }
                    let mut work = position.clone();
                    let fast = !position.king_attacked_after_vacating(
                        king_sq,
                        side,
                        enemy,
                        from,
                        mv.to as usize,
                        unblock,
                        &mut work,
                    );
                    let mut reference_work = position.clone();
                    let captured = reference_work.make_move_board_only(mv);
                    let reference = !reference_work.in_check(side);
                    reference_work.unmake_move_board_only(mv, captured);
                    assert_eq!(
                        fast,
                        reference,
                        "game {_game} ply {ply} mv {mv} fen {}",
                        position.to_fen()
                    );
                    compared += 1;
                }
            }
            let legal = position.legal_moves();
            if legal.is_empty() {
                break;
            }
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            position.make_move(legal[(state as usize) % legal.len()]);
        }
    }
    assert!(compared > 40000, "compare count too small: {compared}");
    println!("vacate-vs-make_unmake comparisons: {compared}");
}
