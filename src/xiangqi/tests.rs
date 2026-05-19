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
fn four_repetition_cycles_without_forcing_do_not_draw() {
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
    assert_eq!(Position::rule_outcome(&history), None);
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
