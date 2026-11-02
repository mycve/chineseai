//! 公开教学棋例与独立构造的边界用例；坐标采用 a0 红方底线。
//! 教学棋例原始来源：https://www.xqarena.com/zh-hans/guides/checkmate-patterns

use chineseai::xiangqi::{Color, MateClassification, MateKind, MatingPattern, Move, Position};

fn play(fen: &str, moves: &[&str]) -> Position {
    let mut position = Position::from_fen(fen).expect("valid fixture FEN");
    for text in moves {
        assert!(position.classify_mate().is_none(), "premature mate: {fen}");
        let mv = position
            .parse_uci_move(text)
            .unwrap_or_else(|| panic!("illegal fixture move {text}: {}", position.to_fen()));
        position.make_move(mv);
    }
    position
}

fn rotate_and_swap(position: &Position) -> Position {
    let fen = position.to_fen();
    let board = fen.split_whitespace().next().unwrap();
    let board = board
        .split('/')
        .rev()
        .map(|rank| {
            rank.chars()
                .rev()
                .map(|ch| {
                    if ch.is_ascii_lowercase() {
                        ch.to_ascii_uppercase()
                    } else {
                        ch.to_ascii_lowercase()
                    }
                })
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("/");
    let side = if position.side_to_move() == Color::Red {
        "b"
    } else {
        "w"
    };
    Position::from_fen(&format!("{board} {side}")).unwrap()
}

fn assert_same_patterns(original: &MateClassification, transformed: &MateClassification) {
    assert_eq!(original.kind, transformed.kind);
    assert_eq!(original.patterns.len(), transformed.patterns.len());
    for pattern in &original.patterns {
        assert!(
            transformed.patterns.contains(pattern),
            "missing {pattern:?}"
        );
    }
}

fn assert_scene(fen: &str, moves: &[&str], expected: &[MatingPattern]) {
    let position = play(fen, moves);
    let report = position
        .classify_mate()
        .unwrap_or_else(|| panic!("expected terminal position: {}", position.to_fen()));
    assert_eq!(report.kind, MateKind::Checkmate);
    assert_eq!(report.winner, Color::Red);
    assert!(position.in_check(Color::Black));
    assert!(position.legal_moves().is_empty());
    assert!(!report.checkers.is_empty());
    for pattern in expected {
        assert!(
            report.patterns.contains(pattern),
            "missing {pattern:?}: {report:?}"
        );
    }

    let mirrored = position.mirror_files().classify_mate().unwrap();
    assert_same_patterns(&report, &mirrored);
    assert_eq!(mirrored.winner, report.winner);
    let mut expected_checkers = report
        .checkers
        .iter()
        .map(|&sq| sq / 9 * 9 + 8 - sq % 9)
        .collect::<Vec<_>>();
    let mut actual_checkers = mirrored.checkers.clone();
    expected_checkers.sort_unstable();
    actual_checkers.sort_unstable();
    assert_eq!(expected_checkers, actual_checkers);

    let rotated = rotate_and_swap(&position).classify_mate().unwrap();
    assert_same_patterns(&report, &rotated);
    assert_eq!(rotated.winner, Color::Black);
    let mut expected_checkers = report
        .checkers
        .iter()
        .map(|&sq| 89 - sq)
        .collect::<Vec<_>>();
    let mut actual_checkers = rotated.checkers.clone();
    expected_checkers.sort_unstable();
    actual_checkers.sort_unstable();
    assert_eq!(expected_checkers, actual_checkers);
}

macro_rules! scene {
    ($name:ident, $fen:literal, [$($mv:literal),+], [$($pattern:ident),+]) => {
        #[test]
        fn $name() {
            assert_scene($fen, &[$($mv),+], &[$(MatingPattern::$pattern),+]);
        }
    };
}

scene!(
    double_cannon,
    "2bakab2/9/9/4C4/9/7C1/9/9/9/4K4 w",
    ["h4e4"],
    [DoubleCannon, CannonCheck]
);
scene!(
    iron_bolt,
    "2bak4/4a4/4b4/4C4/9/9/5R3/9/9/5K3 w",
    ["f3f9"],
    [IronBolt, RookCheck]
);
scene!(
    horse_cannon,
    "4k4/9/4N4/9/9/9/7C1/9/9/4K4 w",
    ["h3e3"],
    [HorseCannon, CannonCheck]
);

#[test]
fn horizontal_horse_cannon_uses_the_same_screen_and_escape_constraints() {
    // 自构：f8 马为 h8 炮作架，并封住 d8 黑将的 d9/d7 两个纵向逃点。
    assert_scene(
        "9/3k1N1C1/9/9/9/9/9/9/9/4K4 b",
        &[],
        &[MatingPattern::HorseCannon, MatingPattern::CannonCheck],
    );
}
scene!(
    elbow_horse,
    "3ak1b2/4a4/9/7N1/9/5R3/9/9/9/4K4 w",
    ["h6g8"],
    [ElbowHorse, HorseCheck]
);
scene!(
    palace_corner_horse,
    "3aka3/3P5/9/9/6N2/9/9/9/9/3K5 w",
    ["g5f7"],
    [PalaceCornerHorse, HorseCheck]
);
scene!(
    double_rook,
    "9/4k4/9/4P4/9/9/9/7R1/R8/4K4 w",
    ["h2h8", "e8e9", "a1a9"],
    [DoubleRook, RookCheck]
);
scene!(
    smothered_cannon,
    "4ka3/4a4/9/9/9/6C2/9/9/9/5K3 w",
    ["g4g9"],
    [SmotheredCannon, CannonCheck]
);
// 此例来自“海底捞月”教学，但单个终局不能证明完整残局过程，只要求实际的车杀标签。
scene!(
    cannon_protects_checking_rook,
    "C1b4R1/3k5/9/9/9/9/9/9/9/4K4 w",
    ["h9d9"],
    [RookCheck]
);
scene!(
    two_soldiers,
    "2b1kab2/3P1P3/9/9/9/9/9/9/9/5K3 w",
    ["f8f9"],
    [TwoSoldiers, SoldierCheck]
);
scene!(
    heaven_earth_cannon,
    "2b1k4/1P2a4/4b4/9/4C4/9/C8/9/9/4K4 w",
    ["a3a9"],
    [HeavenEarthCannon, CannonCheck]
);
scene!(
    white_faced_general,
    "3k2b2/9/1N1P5/9/9/9/9/9/9/4K4 w",
    ["d7d8"],
    [WhiteFacedGeneral, SoldierCheck]
);
scene!(
    double_check,
    "3aka3/9/9/9/4N4/9/9/4R4/9/3K5 w",
    ["e5d7"],
    [DoubleCheck, HorseCheck, RookCheck]
);

#[test]
fn normal_and_checked_positions_are_not_mates() {
    assert!(Position::startpos().classify_mate().is_none());
    // 自构：炮借黑卒将军，但将可以离开中路。
    let checked = Position::from_fen("4k4/9/4p4/9/4C4/9/9/9/9/3K5 b").unwrap();
    assert!(checked.in_check(Color::Black));
    assert!(!checked.legal_moves().is_empty());
    assert!(checked.classify_mate().is_none());
}

#[test]
fn two_cannons_with_an_extra_blocker_do_not_make_double_cannon_mate() {
    // 自构：在重炮例的将与前炮之间加入黑卒，黑方可以移士挡住前炮。
    let position = Position::from_fen("2bakab2/9/4p4/4C4/9/4C4/9/9/9/4K4 b").unwrap();
    assert!(!position.legal_moves().is_empty());
    assert!(position.classify_mate().is_none());
}

#[test]
fn blocked_horse_leg_does_not_count_as_a_horse_check() {
    // 自构：卧槽马例在 f8 加入黑卒，阻塞 g8 马攻击 e9 将的马腿。
    let position = Position::from_fen("3ak1b2/4apN2/9/9/9/5R3/9/9/9/4K4 b").unwrap();
    assert!(!position.in_check(Color::Black));
    assert!(position.classify_mate().is_none());
}

#[test]
fn cannon_mates_do_not_automatically_count_as_double_cannon_or_horse_cannon() {
    let position = play("4ka3/4a4/9/9/9/6C2/9/9/9/5K3 w", &["g4g9"]);
    let report = position.classify_mate().unwrap();
    assert!(!report.patterns.contains(&MatingPattern::DoubleCannon));
    assert!(!report.patterns.contains(&MatingPattern::HorseCannon));
}

#[test]
fn stalemate_has_no_checkmate_labels() {
    // 自构：红车封第八横线，红将封 e 路；d9 黑将未被将军却无着。
    let position = Position::from_fen("3k5/8R/9/9/9/9/9/9/9/4K4 b").unwrap();
    assert!(!position.in_check(Color::Black));
    assert!(position.legal_moves().is_empty());
    let report = position.classify_mate().unwrap();
    assert_eq!(report.kind, MateKind::Stalemate);
    assert_eq!(report.winner, Color::Red);
    assert!(report.checkers.is_empty());
    assert!(report.patterns.is_empty());
}

#[test]
fn captured_general_is_not_a_pattern_label() {
    // 原始 make_move 不检查合法性；用它模拟外部调用者传入缺将局面。
    let mut position = Position::from_fen("3k5/9/9/9/9/9/9/9/3R5/4K4 w").unwrap();
    position.make_move(Move::from_uci("d1d9").unwrap());
    assert!(!position.has_general(Color::Black));
    assert!(position.classify_mate().is_none());
}

#[test]
fn several_labels_can_describe_the_same_mate() {
    let position = play("4k4/9/4N4/9/9/9/7C1/9/9/4K4 w", &["h3e3"]);
    let report = position.classify_mate().unwrap();
    assert!(report.patterns.contains(&MatingPattern::HorseCannon));
    assert!(report.patterns.contains(&MatingPattern::CannonCheck));
    assert_eq!(report.checkers.len(), 1);
}

#[test]
fn smothered_mate_requires_own_pieces_on_every_palace_exit() {
    // 自构：两士和未过河的卒占住全部宫内相邻点，无法挡住 f7 马的将军。
    let position = Position::from_fen("3aka3/4p4/5N3/9/9/9/9/9/9/3K5 b").unwrap();
    let report = position.classify_mate().unwrap();
    assert_eq!(report.kind, MateKind::Checkmate);
    assert!(report.patterns.contains(&MatingPattern::SmotheredMate));
    assert!(report.patterns.contains(&MatingPattern::PalaceCornerHorse));

    let open_exit = Position::from_fen("3ak4/4p4/5N3/9/9/9/9/9/9/3K5 b").unwrap();
    assert!(!open_exit.legal_moves().is_empty());
    assert!(open_exit.classify_mate().is_none());
}

#[test]
fn angler_horse_must_really_control_an_escape_square() {
    // 自构：g7 马不直接将军，但独自控制 e8 逃点；d9 车受 d0 帅保护。
    let position = Position::from_fen("3Rk4/9/6N2/9/9/9/9/9/9/3K5 b").unwrap();
    let report = position.classify_mate().unwrap();
    assert!(report.patterns.contains(&MatingPattern::AnglerHorse));
    assert!(!report.patterns.contains(&MatingPattern::HorseCheck));
    assert_same_patterns(&report, &position.mirror_files().classify_mate().unwrap());
    assert_same_patterns(
        &report,
        &rotate_and_swap(&position).classify_mate().unwrap(),
    );

    // 同在 g7 的马只是重复控制 e8；h8 车已经封住该逃点，不应按站位贴标签。
    let redundant = Position::from_fen("R3k4/7R1/6N2/4P4/9/9/9/9/9/4K4 b").unwrap();
    let report = redundant.classify_mate().unwrap();
    assert!(report.patterns.contains(&MatingPattern::RookCheck));
    assert!(!report.patterns.contains(&MatingPattern::AnglerHorse));
}

#[test]
fn octagonal_horse_support_is_distinct_from_a_corner_horse_check() {
    // 自构：f7 马独自控制 e9，d1 车将军并控制 d8。
    let position = Position::from_fen("3k5/9/5N3/9/9/9/9/9/3R5/5K3 b").unwrap();
    let report = position.classify_mate().unwrap();
    assert!(report.patterns.contains(&MatingPattern::OctagonalHorse));
    assert!(!report.patterns.contains(&MatingPattern::PalaceCornerHorse));
    assert!(!report.patterns.contains(&MatingPattern::HorseCheck));
    assert_same_patterns(&report, &position.mirror_files().classify_mate().unwrap());
    assert_same_patterns(
        &report,
        &rotate_and_swap(&position).classify_mate().unwrap(),
    );

    let without_horse = Position::from_fen("3k5/9/9/9/9/9/9/9/3R5/5K3 b").unwrap();
    assert!(!without_horse.legal_moves().is_empty());
    assert!(without_horse.classify_mate().is_none());
}

#[test]
fn central_cannon_without_a_real_pin_is_not_iron_bolt() {
    // 自构：双车已封死黑将，中炮仅挡住将帅照面，没有钉住任何防守子。
    let position = Position::from_fen("R3k4/7R1/9/9/4C4/9/9/9/9/4K4 b").unwrap();
    let report = position.classify_mate().unwrap();
    assert!(report.patterns.contains(&MatingPattern::DoubleRook));
    assert!(!report.patterns.contains(&MatingPattern::IronBolt));
    assert!(!report.patterns.contains(&MatingPattern::HeavenEarthCannon));
}

#[test]
fn two_soldiers_near_the_palace_are_not_enough_without_real_support() {
    // 自构：f9 兵参与将军，d8 兵的控格全部被车重复覆盖，并非二鬼配合。
    let position = Position::from_fen("R3kP3/3P3R1/9/9/4C4/9/9/9/9/4K4 b").unwrap();
    let report = position.classify_mate().unwrap();
    assert!(report.patterns.contains(&MatingPattern::SoldierCheck));
    assert!(!report.patterns.contains(&MatingPattern::TwoSoldiers));
}

#[test]
fn horizontal_double_cannon_has_only_the_expected_labels() {
    // 自构：两炮横列，黑方上下两士不能垫住炮架链。
    let position = Position::from_fen("3a5/3k1C1C1/3a5/9/9/9/9/9/9/4K4 b").unwrap();
    let before = position.clone();
    let report = position.classify_mate().unwrap();
    assert_eq!(report.kind, MateKind::Checkmate);
    assert_eq!(
        report.patterns,
        [MatingPattern::DoubleCannon, MatingPattern::CannonCheck]
    );
    assert_eq!(position, before, "识别不得改变棋盘状态");
    assert_same_patterns(&report, &position.mirror_files().classify_mate().unwrap());
    assert_same_patterns(
        &report,
        &rotate_and_swap(&position).classify_mate().unwrap(),
    );
}

#[test]
fn attacker_in_check_is_not_a_valid_mating_position() {
    // 自构的非法轮走关系：黑将被重炮将死，但黑车也正在攻击红帅。
    let position = Position::from_fen("3aka3/9/9/4C4/9/4C4/9/9/9/3rK4 b").unwrap();
    assert!(position.in_check(Color::Red));
    assert!(position.in_check(Color::Black));
    assert!(position.classify_mate().is_none());
}
