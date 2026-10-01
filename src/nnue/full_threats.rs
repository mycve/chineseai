//! Pikafish b562d6ae FullThreats indexing and occupied-board attacks.
use std::sync::OnceLock;

use crate::xiangqi::{BOARD_SIZE, Color, Piece, PieceKind, Position};

use super::pikafish::THREAT_INPUTS;

const INVALID: u16 = THREAT_INPUTS as u16;
const PLANES: usize = 14;
const TABLE_LEN: usize = PLANES * BOARD_SIZE * BOARD_SIZE * PLANES;
const KINDS: [PieceKind; 7] = [
    PieceKind::Rook,
    PieceKind::Advisor,
    PieceKind::Cannon,
    PieceKind::Soldier,
    PieceKind::Horse,
    PieceKind::Elephant,
    PieceKind::General,
];

// 原表含 Piece 编号 0、8 两个空位；此处删去空位，顺序保持相同。
const VALID_PAIRS: [&[u8; PLANES]; PLANES] = [
    b"11111111111110", // R
    b"11101001011100", // A
    b"11111111111110", // C
    b"00111100111110", // P
    b"11111111111110", // N
    b"10111111011100", // B
    b"00000000000000", // K
    b"11111101111111", // r
    b"10111001110101", // a
    b"11111101111111", // c
    b"01111100011110", // p
    b"11111101111111", // n
    b"10111001011111", // b
    b"00111000110110", // k
];

fn plane(piece: Piece) -> usize {
    usize::from(piece.color == Color::Black) * 7
        + KINDS.iter().position(|&kind| kind == piece.kind).unwrap()
}

fn native(square: usize) -> usize {
    (9 - square / 9) * 9 + square % 9
}

fn valid_square(plane: usize, square: usize) -> bool {
    super::pikafish::valid_square(plane, square)
}

fn slot(attacker: usize, from: usize, to: usize, attacked: usize) -> usize {
    (((attacker * BOARD_SIZE + from) * BOARD_SIZE + to) * PLANES) + attacked
}

fn orthogonal(from: usize, to: usize) -> bool {
    from != to && (from / 9 == to / 9 || from % 9 == to % 9)
}

fn in_palace(square: usize) -> bool {
    (3..=5).contains(&(square % 9)) && matches!(square / 9, 0..=2 | 7..=9)
}

fn pseudo_attack(kind: PieceKind, color: Color, from: usize, to: usize) -> bool {
    let (fr, ff, tr, tf) = (
        (from / 9) as i32,
        (from % 9) as i32,
        (to / 9) as i32,
        (to % 9) as i32,
    );
    let dr = tr - fr;
    let df = tf - ff;
    match kind {
        PieceKind::Rook => orthogonal(from, to),
        // 静态炮架为每个正交相邻格；炮的最短可攻击距离为 2。
        PieceKind::Cannon => orthogonal(from, to) && dr.abs() + df.abs() >= 2,
        PieceKind::Soldier => {
            dr == if color == Color::Red { 1 } else { -1 } && df == 0
                || ((color == Color::Red && fr > 4 || color == Color::Black && fr < 5)
                    && dr == 0
                    && df.abs() == 1)
        }
        PieceKind::Horse => (dr.abs() == 2 && df.abs() == 1) || (dr.abs() == 1 && df.abs() == 2),
        PieceKind::Elephant => {
            dr.abs() == 2 && df.abs() == 2 && (if fr > 4 { tr > 4 } else { tr < 5 })
        }
        PieceKind::Advisor => in_palace(from) && in_palace(to) && dr.abs() == 1 && df.abs() == 1,
        PieceKind::General => in_palace(from) && in_palace(to) && dr.abs() + df.abs() == 1,
    }
}

fn offsets() -> &'static [u16] {
    static OFFSETS: OnceLock<Vec<u16>> = OnceLock::new();
    OFFSETS.get_or_init(|| {
        let mut offsets = vec![INVALID; TABLE_LEN];
        let mut next = 0_usize;
        for attacker in 0..PLANES {
            let kind = KINDS[attacker % 7];
            let color = if attacker < 7 {
                Color::Red
            } else {
                Color::Black
            };
            for from in 0..BOARD_SIZE {
                if !valid_square(attacker, from) {
                    continue;
                }
                for attacked in 0..PLANES {
                    if VALID_PAIRS[attacker][attacked] != b'1' {
                        continue;
                    }
                    for to in 0..BOARD_SIZE {
                        if !valid_square(attacked, to) || !pseudo_attack(kind, color, from, to) {
                            continue;
                        }
                        let enemy = attacker / 7 != attacked / 7;
                        let same_kind = attacker % 7 == attacked % 7;
                        let same_file = from % 9 == to % 9;
                        let same_rank = from / 9 == to / 9;
                        let semi_excluded = same_kind
                            && (kind != PieceKind::Soldier
                                || (enemy && same_file)
                                || (!enemy && same_rank))
                            && kind != PieceKind::Horse;
                        if !semi_excluded || from > to {
                            offsets[slot(attacker, from, to, attacked)] = next as u16;
                            next += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(next, THREAT_INPUTS, "FullThreats 静态编号与 Pikafish 不符");
        offsets
    })
}

/// 返回 FullThreats 索引；官方无效特征返回 None。
pub fn threat_index(
    perspective: Color,
    attacker: Piece,
    from: usize,
    to: usize,
    attacked: Piece,
    mirror: bool,
) -> Option<usize> {
    if from >= BOARD_SIZE || to >= BOARD_SIZE {
        return None;
    }
    let map = |square: usize| {
        let mut square = native(square);
        if mirror {
            square = square / 9 * 9 + 8 - square % 9;
        }
        if perspective == Color::Black {
            square = (9 - square / 9) * 9 + square % 9;
        }
        square
    };
    let mut attacker = attacker;
    let mut attacked = attacked;
    if perspective == Color::Black {
        attacker.color = attacker.color.opposite();
        attacked.color = attacked.color.opposite();
    }
    let index = offsets()[slot(plane(attacker), map(from), map(to), plane(attacked))];
    (index != INVALID).then_some(index as usize)
}

fn attacks(position: &Position, kind: PieceKind, color: Color, from: usize, to: usize) -> bool {
    if !pseudo_attack(kind, color, from, to) {
        return false;
    }
    let (fr, ff, tr, tf) = (
        (from / 9) as i32,
        (from % 9) as i32,
        (to / 9) as i32,
        (to % 9) as i32,
    );
    match kind {
        PieceKind::Rook | PieceKind::Cannon => {
            let dr = (tr - fr).signum();
            let df = (tf - ff).signum();
            let mut r = fr + dr;
            let mut f = ff + df;
            let mut occupied = 0;
            while r != tr || f != tf {
                occupied += usize::from(position.piece_at(native((r * 9 + f) as usize)).is_some());
                r += dr;
                f += df;
            }
            if kind == PieceKind::Rook {
                occupied == 0
            } else {
                occupied == 1
            }
        }
        PieceKind::Horse => {
            let leg = if (tr - fr).abs() == 2 {
                ((fr + (tr - fr).signum()) * 9 + ff) as usize
            } else {
                (fr * 9 + ff + (tf - ff).signum()) as usize
            };
            position.piece_at(native(leg)).is_none()
        }
        PieceKind::Elephant => {
            let eye = ((fr + tr) / 2 * 9 + (ff + tf) / 2) as usize;
            position.piece_at(native(eye)).is_none()
        }
        _ => true,
    }
}

/// 追加双方攻击占据格的索引，包括己方子力之间的保护关系。
/// 非法 FEN 中的无效子力位置会被静态索引表过滤。
pub fn fill_threat_features(
    position: &Position,
    perspective: Color,
    output: &mut Vec<usize>,
) -> Option<()> {
    let (_, mirror) = super::pikafish::feature_bucket(position, perspective)?;
    output.clear();
    output.reserve(64);
    let occupied = (0..BOARD_SIZE)
        .filter_map(|square| position.piece_at(square).map(|piece| (square, piece)))
        .collect::<Vec<_>>();
    for &(from, attacker) in &occupied {
        for &(to, attacked) in &occupied {
            if attacks(
                position,
                attacker.kind,
                attacker.color,
                native(from),
                native(to),
            ) {
                if let Some(index) = threat_index(perspective, attacker, from, to, attacked, mirror)
                {
                    output.push(index);
                }
            }
        }
    }
    Some(())
}

/// 两个视角共享一次攻击关系遍历；输出顺序与分别调用上面的函数完全相同。
pub(crate) fn fill_threat_features_both(
    position: &Position,
    red: &mut Vec<usize>,
    black: &mut Vec<usize>,
) -> Option<()> {
    let (_, red_mirror) = super::pikafish::feature_bucket(position, Color::Red)?;
    let (_, black_mirror) = super::pikafish::feature_bucket(position, Color::Black)?;
    red.clear();
    black.clear();
    red.reserve(64);
    black.reserve(64);
    let mut occupied = [(
        0,
        Piece {
            kind: PieceKind::General,
            color: Color::Red,
        },
    ); BOARD_SIZE];
    let mut count = 0;
    for square in 0..BOARD_SIZE {
        if let Some(piece) = position.piece_at(square) {
            occupied[count] = (square, piece);
            count += 1;
        }
    }
    let occupied = &occupied[..count];
    for &(from, attacker) in occupied {
        for &(to, attacked) in occupied {
            if attacks(
                position,
                attacker.kind,
                attacker.color,
                native(from),
                native(to),
            ) {
                if let Some(index) =
                    threat_index(Color::Red, attacker, from, to, attacked, red_mirror)
                {
                    red.push(index);
                }
                if let Some(index) =
                    threat_index(Color::Black, attacker, from, to, attacked, black_mirror)
                {
                    black.push(index);
                }
            }
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_features(position: &Position, perspective: Color, output: &mut Vec<usize>) {
        let (_, mirror) = super::super::pikafish::feature_bucket(position, perspective).unwrap();
        output.clear();
        for from in 0..BOARD_SIZE {
            let Some(attacker) = position.piece_at(from) else {
                continue;
            };
            for to in 0..BOARD_SIZE {
                let Some(attacked) = position.piece_at(to) else {
                    continue;
                };
                if attacks(
                    position,
                    attacker.kind,
                    attacker.color,
                    native(from),
                    native(to),
                ) {
                    if let Some(index) =
                        threat_index(perspective, attacker, from, to, attacked, mirror)
                    {
                        output.push(index);
                    }
                }
            }
        }
    }

    #[test]
    fn occupied_pairs_match_original_random_legal_positions() {
        let mut position = Position::startpos();
        let mut random = 0x1234_5678_9abc_def0_u64;
        let mut positions = Vec::new();
        for _ in 0..100 {
            positions.push(position.clone());
            let legal = position.legal_moves();
            if legal.is_empty() {
                break;
            }
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            position.make_move(legal[(random as usize) % legal.len()]);
        }
        let mut reference = Vec::new();
        let mut optimized = Vec::new();
        let mut red = Vec::new();
        let mut black = Vec::new();
        for position in &positions {
            fill_threat_features_both(position, &mut red, &mut black).unwrap();
            for perspective in [Color::Red, Color::Black] {
                reference_features(position, perspective, &mut reference);
                fill_threat_features(position, perspective, &mut optimized).unwrap();
                assert_eq!(
                    optimized,
                    reference,
                    "{} {perspective:?}",
                    position.to_fen()
                );
                assert_eq!(
                    if perspective == Color::Red {
                        &red
                    } else {
                        &black
                    },
                    &reference,
                    "shared attack pass: {} {perspective:?}",
                    position.to_fen()
                );
            }
        }
        let start = std::time::Instant::now();
        for _ in 0..10 {
            for position in &positions {
                for perspective in [Color::Red, Color::Black] {
                    reference_features(position, perspective, &mut reference);
                    std::hint::black_box(&reference);
                }
            }
        }
        let original = start.elapsed();
        let start = std::time::Instant::now();
        for _ in 0..10 {
            for position in &positions {
                for perspective in [Color::Red, Color::Black] {
                    fill_threat_features(position, perspective, &mut optimized).unwrap();
                    std::hint::black_box(&optimized);
                }
            }
        }
        let separate = start.elapsed();
        let start = std::time::Instant::now();
        for _ in 0..10 {
            for position in &positions {
                fill_threat_features_both(position, &mut red, &mut black).unwrap();
                std::hint::black_box((&red, &black));
            }
        }
        eprintln!(
            "FullThreats both perspectives: old={original:?}, occupied={separate:?}, shared={:?}",
            start.elapsed()
        );
    }

    #[test]
    fn static_dimension_and_startpos() {
        assert_eq!(
            offsets().iter().filter(|&&index| index != INVALID).count(),
            THREAT_INPUTS
        );
        let position = Position::startpos();
        for perspective in [Color::Red, Color::Black] {
            let mut indices = Vec::new();
            fill_threat_features(&position, perspective, &mut indices).unwrap();
            assert!(!indices.is_empty());
            assert!(indices.iter().all(|&index| index < THREAT_INPUTS));
            assert!(indices.len() <= 64);
        }
    }

    #[test]
    fn file_mirror_preserves_both_perspectives() {
        let position = Position::from_fen("4k4/9/1r7/9/4p4/9/9/7R1/9/3K5 w").unwrap();
        let mirrored = position.mirror_files();
        for perspective in [Color::Red, Color::Black] {
            let mut first = Vec::new();
            let mut second = Vec::new();
            fill_threat_features(&position, perspective, &mut first).unwrap();
            fill_threat_features(&mirrored, perspective, &mut second).unwrap();
            first.sort_unstable();
            second.sort_unstable();
            assert_eq!(first, second);
        }
    }

    #[test]
    fn legal_play_keeps_threat_indices_in_range() {
        let mut position = Position::startpos();
        for ply in 0..80 {
            for perspective in [Color::Red, Color::Black] {
                let mut indices = Vec::new();
                fill_threat_features(&position, perspective, &mut indices).unwrap();
                assert!(indices.len() <= 64);
                assert!(indices.iter().all(|&index| index < THREAT_INPUTS));
            }
            let legal = position.legal_moves();
            if legal.is_empty() {
                break;
            }
            position.make_move(legal[(ply * 37 + 11) % legal.len()]);
        }
    }

    #[test]
    fn cannon_threat_needs_exactly_one_screen() {
        let with_screen = Position::from_fen("5k3/9/9/9/9/9/9/9/3RPC3/4K4 w").unwrap();
        let without_screen = Position::from_fen("5k3/9/9/9/9/9/9/9/3R1C3/4K4 w").unwrap();
        let attacker = with_screen.piece_at(77).unwrap(); // f1 炮
        let attacked = with_screen.piece_at(75).unwrap(); // d1 车
        let (_, mirror) = super::super::pikafish::feature_bucket(&with_screen, Color::Red).unwrap();
        let index = threat_index(Color::Red, attacker, 77, 75, attacked, mirror).unwrap();
        let mut present = Vec::new();
        let mut absent = Vec::new();
        fill_threat_features(&with_screen, Color::Red, &mut present).unwrap();
        fill_threat_features(&without_screen, Color::Red, &mut absent).unwrap();
        assert!(present.contains(&index));
        assert!(!absent.contains(&index));
    }
}
