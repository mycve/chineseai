use super::*;

#[test]
fn startpos_roundtrip_fen() {
    let position = Position::startpos();
    assert_eq!(position.to_fen(), STARTPOS_FEN);
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
        chased_piece_mask: 0,
    }
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
    for text in [
        "f4d5", "c6c5", "d5c7", "f7c7", "i1d1", "a9d9", "d1d9", "e8d9", "b0b4", "i9i8", "c3c4",
        "i8d8", "c4c5", "e7c5", "b4f4", "i7h5", "f4f5", "h6h2", "f5h5", "c7c2", "h5c5", "d8d3",
        "e3e4", "d3e3", "a3a4", "c2c3", "c5i5", "e3e4", "i5c5", "c3b3", "c5c3", "b3b5", "c3c5",
        "b5b0", "c5h5", "h2f2", "h5b5", "b0a0", "b5b0", "a0a3", "b0b3", "a3a0", "b3a3", "a0b0",
        "a3b3", "b0a0", "b3a3", "a0b0", "a3b3", "b0a0", "b3a3", "a0b0", "a3b3", "b0a0",
    ] {
        assert_eq!(position.rule_outcome_with_history(&history), None);
        let mv = position.parse_uci_move(text).unwrap();
        assert!(position.legal_moves_with_rules(&history).contains(&mv));
        history.push(position.rule_history_entry_after_move(mv));
        position.make_move(mv);
    }

    assert_eq!(
        position.rule_outcome_with_history(&history),
        Some(RuleOutcome::Win(Color::Black))
    );
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
fn one_cycle_repetition_does_not_end_by_force_rule() {
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
        "2Rakab2/8r/4c1n2/p3p1p1p/2p6/9/P3P3P/1CN1NC3/9/1RBAKArc1 b"
    );
    assert_eq!(position.rule_outcome_with_history(&history), None);
    assert_eq!(position.legal_moves().len(), 44);
    assert_eq!(position.legal_moves_with_rules(&history).len(), 44);
}

#[test]
fn horse_repeatedly_chasing_rook_is_forbidden() {
    let mut position = Position::from_fen(
        "2bak4/4a4/2ncb2c1/p3p2CP/9/1N1RP4/P5r2/4C4/9/2BAKA3 b - - 0 1",
    )
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
    assert!(!position.legal_moves_with_rules(&history).contains(&mv));
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
    assert!(!position.legal_moves_with_rules(&history).contains(&mv));
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
