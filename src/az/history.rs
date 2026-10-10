//! 当前空间摘要与最近两步的二阶空间变化，统一使用当前行棋方视角。
//! 横坐标以棋盘中心为零；镜像只改变符号与左右区域顺序。
use super::arch::HISTORY_CONTEXT_SIZE;
use crate::xiangqi::{Color, Piece, PieceKind, Position, RuleHistoryEntry};

const BASIS: usize = HISTORY_CONTEXT_SIZE / 2;
const PIECE_SQUARES: usize = 14 * 90;

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

fn integer_contribution(index: usize) -> ([(usize, i32); 4], [(usize, i32); 4]) {
    let plane = index / 90;
    let square = index % 90;
    let x = (square % 9) as i32 - 4;
    let y = (square / 9) as i32;
    let region = usize::from(square / 9 >= 5) * 3 + square % 9 / 3;
    let sign = if plane < 7 { 1 } else { -1 };
    (
        [
            (plane * 3, 1),
            (plane * 3 + 1, x),
            (plane * 3 + 2, y),
            (42 + region, sign),
        ],
        [
            (plane * 3, x * x),
            (plane * 3 + 1, y * y),
            (plane * 3 + 2, x * y),
            (42 + region, sign * y),
        ],
    )
}

fn feature_scale(index: usize, second: bool) -> f32 {
    if index >= 42 {
        return if second { 1.0 / 288.0 } else { 1.0 / 16.0 };
    }
    match (second, index % 3) {
        (false, 0) => 0.2,
        (false, 1) => 0.2 / 8.0,
        (false, _) => 0.2 / 9.0,
        (true, 0) => 0.2 / 64.0,
        (true, 1) => 0.2 / 81.0,
        (true, _) => 0.2 / 72.0,
    }
}

fn contribution(index: usize) -> ([(usize, f32); 4], [(usize, f32); 4]) {
    let (first, second) = integer_contribution(index);
    (
        first.map(|(i, v)| (i, v as f32 * feature_scale(i, false))),
        second.map(|(i, v)| (i, v as f32 * feature_scale(i, true))),
    )
}

fn descriptor(planes: &[u128; 14]) -> ([i32; BASIS], [i32; BASIS]) {
    let mut first = [0; BASIS];
    let mut second = [0; BASIS];
    for (plane, &mask) in planes.iter().enumerate() {
        for square in 0..90 {
            if mask & (1u128 << square) == 0 {
                continue;
            }
            let (a, b) = integer_contribution(plane * 90 + square);
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
    for i in 0..BASIS {
        features[i] = first[i] as f32 * feature_scale(i, false);
    }
    // 先以二倍整数累加两步差分，再缩放。左右变换只有符号与置换，
    // 不受棋子枚举顺序或浮点累加舍入影响。
    let mut delta_twice = [0; BASIS];
    for h in 1..=available.min(2) {
        let (_, past) = descriptor(&planes[h]);
        for i in 0..BASIS {
            delta_twice[i] += (current[i] - past[i]) * (2 / h as i32);
        }
    }
    for i in 0..BASIS {
        features[BASIS + i] = delta_twice[i] as f32 * (feature_scale(i, true) * 0.5);
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

#[derive(Clone, Debug, Default)]
pub(crate) struct ValueHistoryCache {
    current: Vec<[f32; 3]>,
    second: Vec<[f32; 3]>,
    active: bool,
}

impl ValueHistoryCache {
    pub(crate) fn new(weights: &[f32]) -> Self {
        assert_eq!(weights.len(), 3 * HISTORY_CONTEXT_SIZE);
        let mut current = vec![[0.0; 3]; PIECE_SQUARES];
        let mut second = vec![[0.0; 3]; PIECE_SQUARES];
        for index in 0..PIECE_SQUARES {
            let (a, b) = contribution(index);
            for output in 0..3 {
                for (i, value) in a {
                    current[index][output] += value * weights[output * HISTORY_CONTEXT_SIZE + i];
                }
                for (i, value) in b {
                    second[index][output] +=
                        value * weights[output * HISTORY_CONTEXT_SIZE + BASIS + i];
                }
            }
        }
        Self {
            current,
            second,
            active: weights.iter().any(|&w| w != 0.0),
        }
    }

    pub(crate) fn valid(&self) -> bool {
        self.current.len() == PIECE_SQUARES && self.second.len() == PIECE_SQUARES
    }

    fn delta(&self, moved: Piece, entry: &RuleHistoryEntry, side: Color) -> [f32; 3] {
        let mv = entry.mv.unwrap();
        let from = self.second[piece_square(moved, mv.from as usize, side)];
        let to = self.second[piece_square(moved, mv.to as usize, side)];
        let captured = entry
            .captured
            .map(|p| self.second[piece_square(p, mv.to as usize, side)])
            .unwrap_or([0.0; 3]);
        std::array::from_fn(|j| to[j] - from[j] - captured[j])
    }

    pub(crate) fn logits(&self, position: &Position, history: &[RuleHistoryEntry]) -> [f32; 3] {
        if !self.active {
            return [0.0; 3];
        }
        let side = position.side_to_move();
        let mut result = [0.0; 3];
        for square in 0..90 {
            if let Some(piece) = position.piece_at(square) {
                let value = self.current[piece_square(piece, square, side)];
                for j in 0..3 {
                    result[j] += value[j];
                }
            }
        }
        let Some(last) = history
            .last()
            .filter(|e| e.mv.is_some() && e.mover.is_some())
        else {
            return result;
        };
        let mv = last.mv.unwrap();
        let Some(moved) = position.piece_at(mv.to as usize) else {
            return result;
        };
        let d1 = self.delta(moved, last, side);
        for j in 0..3 {
            result[j] += d1[j];
        }
        let Some(previous) = history
            .len()
            .checked_sub(2)
            .and_then(|i| history.get(i))
            .filter(|e| e.mv.is_some() && e.mover.is_some())
        else {
            return result;
        };
        let previous_to = previous.mv.unwrap().to as usize;
        let previous_moved = if previous_to == mv.from as usize {
            Some(moved)
        } else if previous_to == mv.to as usize {
            last.captured
        } else {
            position.piece_at(previous_to)
        };
        if let Some(moved) = previous_moved {
            let d2 = self.delta(moved, previous, side);
            for j in 0..3 {
                result[j] += (d1[j] + d2[j]) * 0.5;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::az::{AzEvalAccumulator, AzEvalScratch, AzNnue, RULE_CONTEXT_SIZE};
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
    fn reflection_history_planes_match_signed_permutation_bit_for_bit() {
        let mut original = [[0u128; 14]; 3];
        for (h, boards) in original.iter_mut().enumerate() {
            for (p, mask) in boards.iter_mut().enumerate() {
                for square in 0..90 {
                    if (square * 17 + p * 13 + h * 7) % 23 < 3 {
                        *mask |= 1u128 << square;
                    }
                }
            }
        }
        let mut mirrored = [[0u128; 14]; 3];
        for h in 0..3 {
            for p in 0..14 {
                for square in 0..90 {
                    if original[h][p] & (1u128 << square) != 0 {
                        mirrored[h][p] |= 1u128 << super::super::nnue::mirror_file_square(square);
                    }
                }
            }
        }
        for available in 0..=2 {
            let expected = super::super::reflection::mirror_history_features(
                &history_features_from_planes(&original, available),
            );
            let actual = history_features_from_planes(&mirrored, available);
            for i in 0..HISTORY_CONTEXT_SIZE {
                assert_eq!(actual[i].to_bits(), expected[i].to_bits(), "feature {i}");
            }
        }
    }

    #[test]
    fn projected_history_matches_real_boards_through_captures_and_both_sides() {
        let mut position = Position::from_fen(
            "1nbak2nr/4a4/1c2b1c2/p1p1p1p1p/9/6P2/P1P1P1r1P/1R2C1N1C/9/1NBAKABR1 w - - 0 1",
        )
        .unwrap();
        let mut history = position.initial_rule_history();
        let mut positions = vec![position.clone()];
        let weights: Vec<f32> = (0..3 * HISTORY_CONTEXT_SIZE)
            .map(|i| (i % 37) as f32 * 0.003 - 0.05)
            .collect();
        let cache = ValueHistoryCache::new(&weights);
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
            let dense: [f32; 3] = std::array::from_fn(|j| {
                actual
                    .iter()
                    .zip(&weights[j * HISTORY_CONTEXT_SIZE..(j + 1) * HISTORY_CONTEXT_SIZE])
                    .map(|(f, w)| f * w)
                    .sum()
            });
            let projected = cache.logits(&position, &history);
            for j in 0..3 {
                assert!((projected[j] - dense[j]).abs() < 1e-6);
            }
            let missing = history_features(&position, &position.initial_rule_history());
            assert_eq!(missing[..48], actual[..48]);
            assert!(missing[48..].iter().all(|&v| v == 0.0));
        }
        assert_eq!(orient_square(8, Color::Red), 89);
        assert_eq!(orient_square(8, Color::Black), 8);
    }

    #[test]
    fn history_raw_logits_match_full_incremental_and_reload_without_changing_policy() {
        let position = Position::startpos();
        let moves = position.legal_moves();
        let mut model = AzNnue::random(8, 17);
        model.rebuild_policy_tactical();
        let mut before = AzEvalScratch::new(model.arch);
        let original = model.evaluate_with_scratch_output_with_repetition(
            &position,
            &moves,
            &[],
            &[0.0; RULE_CONTEXT_SIZE],
            &mut before,
        );
        for (i, weight) in model.value_history_output.iter_mut().enumerate() {
            *weight = (i % 13) as f32 * 0.01 - 0.04;
        }
        model.rebuild_value_history();
        let history = position.initial_rule_history();
        let features = history_features(&position, &history);
        let mut full_scratch = AzEvalScratch::new(model.arch);
        let full = model.evaluate_with_scratch_output_with_repetition_and_history_features(
            &position,
            &moves,
            &[],
            &[0.0; RULE_CONTEXT_SIZE],
            &features,
            &mut full_scratch,
        );
        assert_eq!(before.logits, full_scratch.logits);
        let residual = model.value_history_logits_from_features(&features);
        let expected = crate::az::softmax_fixed3(std::array::from_fn(|j| {
            original.value_wdl[j].ln() + residual[j]
        }));
        for j in 0..3 {
            assert!((expected[j] - full.value_wdl[j]).abs() < 1e-6);
        }
        assert_eq!(full.value, full.value_wdl[0] - full.value_wdl[2]);
        let accumulator = AzEvalAccumulator::new(&model, &position);
        let policy = model.policy_accumulator(&position, position.side_to_move());
        let mut incremental_scratch = AzEvalScratch::new(model.arch);
        let incremental = model.evaluate_incremental_with_scratch_output_with_history(
            &position,
            &accumulator.into_hidden_sum(),
            &policy,
            &moves,
            &[],
            &[0.0; RULE_CONTEXT_SIZE],
            &history,
            &mut incremental_scratch,
        );
        for j in 0..3 {
            assert!((incremental.value_wdl[j] - full.value_wdl[j]).abs() < 1e-6);
        }
        assert_eq!(incremental_scratch.logits, full_scratch.logits);
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tmp")
            .join(format!("history-test-{}.safetensors", std::process::id()));
        model.save(&path).unwrap();
        let loaded = AzNnue::load(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(loaded.value_history_output, model.value_history_output);
        assert_eq!(
            loaded.value_history_logits(&position, &history),
            model.value_history_logits(&position, &history)
        );
        assert_eq!(
            model.clone().value_history_logits(&position, &history),
            model.value_history_logits(&position, &history)
        );
    }
}
