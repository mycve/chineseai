//! 当前空间摘要与最近两步的二阶空间变化，统一使用当前行棋方视角。
use super::arch::HISTORY_CONTEXT_SIZE;
use crate::xiangqi::{Color, Piece, PieceKind, Position, RuleHistoryEntry};

const BASIS: usize = HISTORY_CONTEXT_SIZE / 2;

// Px0 input1：racpnbk，当前方在前；只翻rank，不翻file。
fn orient_square(square: usize, side: Color) -> usize {
    let rank = if side == Color::Red {
        9 - square / 9
    } else {
        square / 9
    };
    rank * 9 + square % 9
}

fn piece_square(piece: Piece, square: usize, side: Color) -> usize {
    let kind = match piece.kind {
        PieceKind::Rook => 0,
        PieceKind::Advisor => 1,
        PieceKind::Cannon => 2,
        PieceKind::Soldier => 3,
        PieceKind::Horse => 4,
        PieceKind::Elephant => 5,
        PieceKind::General => 6,
    };
    let plane = kind + if piece.color == side { 0 } else { 7 };
    plane * 90 + orient_square(square, side)
}

fn contribution(index: usize) -> ([(usize, f32); 4], [(usize, f32); 4]) {
    let plane = index / 90;
    let square = index % 90;
    let x = (square % 9) as f32 / 8.0;
    let y = (square / 9) as f32 / 9.0;
    let region = usize::from(square / 9 >= 5) * 3 + square % 9 / 3;
    let sign = if plane < 7 { 1.0 } else { -1.0 };
    (
        [
            (plane * 3, 0.2),
            (plane * 3 + 1, x * 0.2),
            (plane * 3 + 2, y * 0.2),
            (42 + region, sign / 16.0),
        ],
        [
            (plane * 3, x * x * 0.2),
            (plane * 3 + 1, y * y * 0.2),
            (plane * 3 + 2, x * y * 0.2),
            (42 + region, sign * (x + y) / 32.0),
        ],
    )
}

fn descriptor(planes: &[u128; 14]) -> ([f32; BASIS], [f32; BASIS]) {
    let mut first = [0.0; BASIS];
    let mut second = [0.0; BASIS];
    for (plane, &mask) in planes.iter().enumerate() {
        for square in 0..90 {
            if mask & (1u128 << square) == 0 {
                continue;
            }
            let (a, b) = contribution(plane * 90 + square);
            for (i, value) in a {
                first[i] += value;
            }
            for (i, value) in b {
                second[i] += value;
            }
        }
    }
    (first, second)
}

/// `planes` 使用 Px0 input1 的当前方坐标及 racpnbk 棋种顺序；
/// `available` 为真实可用的过去步数（0、1或2），缺失项不参与差分。
pub fn history_features_from_planes(
    planes: &[[u128; 14]; 3],
    available: usize,
) -> [f32; HISTORY_CONTEXT_SIZE] {
    let (first, current) = descriptor(&planes[0]);
    let mut features = [0.0; HISTORY_CONTEXT_SIZE];
    features[..BASIS].copy_from_slice(&first);
    for h in 1..=available.min(2) {
        let (_, past) = descriptor(&planes[h]);
        for i in 0..BASIS {
            features[BASIS + i] += (current[i] - past[i]) / h as f32;
        }
    }
    features
}

/// 不复制棋盘，用实际走子与被吃子逆还原两个历史时刻的棋子位板。
pub fn history_features(
    position: &Position,
    history: &[RuleHistoryEntry],
) -> [f32; HISTORY_CONTEXT_SIZE] {
    let side = position.side_to_move();
    let mut planes = [[0u128; 14]; 3];
    for square in 0..90 {
        if let Some(piece) = position.piece_at(square) {
            let index = piece_square(piece, square, side);
            planes[0][index / 90] |= 1u128 << (index % 90);
        }
    }
    let mut available = 0;
    for (h, entry) in history.iter().rev().take(2).enumerate() {
        let (Some(mv), Some(_)) = (entry.mv, entry.mover) else {
            break;
        };
        planes[h + 1] = planes[h];
        let to = orient_square(mv.to as usize, side);
        let from = orient_square(mv.from as usize, side);
        let Some(plane) = (0..14).find(|&p| planes[h][p] & (1u128 << to) != 0) else {
            break;
        };
        planes[h + 1][plane] &= !(1u128 << to);
        planes[h + 1][plane] |= 1u128 << from;
        if let Some(captured) = entry.captured {
            let index = piece_square(captured, mv.to as usize, side);
            planes[h + 1][index / 90] |= 1u128 << (index % 90);
        }
        available += 1;
    }
    history_features_from_planes(&planes, available)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xiangqi::Move;

    fn planes(position: &Position, side: Color) -> [u128; 14] {
        let mut masks = [0u128; 14];
        for square in 0..90 {
            if let Some(piece) = position.piece_at(square) {
                let i = piece_square(piece, square, side);
                masks[i / 90] |= 1u128 << (i % 90);
            }
        }
        masks
    }

    #[test]
    fn projected_history_matches_real_boards_through_captures_and_both_sides() {
        let mut position = Position::from_fen(
            "1nbak2nr/4a4/1c2b1c2/p1p1p1p1p/9/6P2/P1P1P1r1P/1R2C1N1C/9/1NBAKABR1 w - - 0 1",
        )
        .unwrap();
        let mut history = position.initial_rule_history();
        let mut positions = vec![position.clone()];
        // 最后一步吃掉上一步的移动子，验证虚拟逆走时恢复 captured。
        for mv in [
            Move::new(67, 31),
            Move::new(7, 14),
            Move::new(31, 35),
            Move::new(8, 35),
        ] {
            assert!(position.legal_moves().contains(&mv));
            history.push(position.rule_history_entry_after_move(mv));
            position.make_move(mv);
            positions.push(position.clone());
            let side = position.side_to_move();
            let mut raw = [[0u128; 14]; 3];
            let available = (positions.len() - 1).min(2);
            for h in 0..=available {
                raw[h] = planes(&positions[positions.len() - 1 - h], side);
            }
            let expected = history_features_from_planes(&raw, available);
            let actual = history_features(&position, &history);
            assert_eq!(actual, expected);
            let missing = history_features(&position, &position.initial_rule_history());
            assert_eq!(missing[..48], actual[..48]);
            assert!(missing[48..].iter().all(|&v| v == 0.0));
        }
        assert_eq!(orient_square(8, Color::Red), 89);
        assert_eq!(orient_square(8, Color::Black), 8);
    }
}
