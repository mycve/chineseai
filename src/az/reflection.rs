//! 网络输入左右规范化；规则引擎始终保留原始坐标与完整历史。
use std::cmp::Ordering;

use crate::xiangqi::{Color, Move, Position};

use super::arch::HISTORY_CONTEXT_SIZE;
use super::nnue::{
    canonical_square, mirror_file_move, mirror_file_square, piece_absolute_feature_index,
};

/// 比较当前行棋方坐标下的棋盘及其左右镜像，不使用可能碰撞的哈希。
/// Less 保留原输入，Greater 镜像，Equal 继续比较历史与逐走法规则标志。
pub(crate) fn board_orientation(position: &Position) -> Ordering {
    board_orientation_for(position, position.side_to_move())
}

/// 增量缓存使用其固定视角，不能被中间节点的实际行棋方改变。
pub(crate) fn board_orientation_for(position: &Position, side: Color) -> Ordering {
    let code = |square| {
        position
            .piece_at(canonical_square(side, square))
            .map_or(0, |piece| piece_absolute_feature_index(side, piece) + 1)
    };
    for rank in 0..10 {
        // 左半行若全部相等，中心与右半行必然也相等。
        for file in 0..4 {
            let square = rank * 9 + file;
            let ordering = code(square).cmp(&code(mirror_file_square(square)));
            if ordering != Ordering::Equal {
                return ordering;
            }
        }
    }
    Ordering::Equal
}

fn clean_zero(value: f32) -> f32 {
    if value == 0.0 { 0.0 } else { value }
}

/// 新历史基只需符号与区域置换；零统一为 +0，保证镜像往返逐位稳定。
pub(crate) fn mirror_history_features(
    features: &[f32; HISTORY_CONTEXT_SIZE],
) -> [f32; HISTORY_CONTEXT_SIZE] {
    let mut result = *features;
    for plane in 0..14 {
        result[plane * 3 + 1] = -features[plane * 3 + 1];
        result[48 + plane * 3 + 2] = -features[48 + plane * 3 + 2];
    }
    for base in [42, 90] {
        for half in 0..2 {
            for file in 0..3 {
                result[base + half * 3 + file] = features[base + half * 3 + 2 - file];
            }
        }
    }
    result.map(clean_zero)
}

/// 当前棋盘对称时，继续比较历史和按 canonical 走法排序的重复标志。
/// flags 缺失项视为0，与现有CPU/GPU输入语义一致。
pub(crate) fn input_orientation(
    position: &Position,
    history: &[f32; HISTORY_CONTEXT_SIZE],
    moves: &[Move],
    flags: &[u8],
) -> Ordering {
    let board = board_orientation(position);
    if board != Ordering::Equal {
        return board;
    }
    let mirrored = mirror_history_features(history);
    for (&original, &reflected) in history.iter().zip(&mirrored) {
        let ordering = clean_zero(original).total_cmp(&reflected);
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    let side = position.side_to_move();
    let key = |mv: Move| {
        canonical_square(side, mv.from as usize) * 90 + canonical_square(side, mv.to as usize)
    };
    let mut original = Vec::with_capacity(moves.len());
    let mut reflected = Vec::with_capacity(moves.len());
    for (i, &mv) in moves.iter().enumerate() {
        let flag = flags.get(i).copied().unwrap_or(0);
        original.push((key(mv), flag));
        reflected.push((key(mirror_file_move(mv)), flag));
    }
    original.sort_unstable();
    reflected.sort_unstable();
    original.cmp(&reflected)
}

/// 仅对完整网络输入的左右固定点调用。在最终logit、softmax之前约束走法对。
/// 保持候选槽位；中心线自镜像走法不变，缺失的镜像候选不参与。
pub(crate) fn symmetrize_policy_logits(moves: &[Move], logits: &mut [f32]) {
    assert_eq!(moves.len(), logits.len());
    for (i, &mv) in moves.iter().enumerate() {
        let mirror = mirror_file_move(mv);
        if let Some(j) = moves.iter().position(|&other| other == mirror) {
            if j > i {
                let mean = clean_zero(logits[i] * 0.5 + logits[j] * 0.5);
                logits[i] = mean;
                logits[j] = mean;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflection_board_orientation_preserves_order_for_both_perspectives() {
        let mut position = Position::startpos();
        position.make_move(Move::new(54, 45));
        let mirrored = position.mirror_files();
        for side in [Color::Red, Color::Black] {
            let orientation = board_orientation_for(&position, side);
            assert_ne!(orientation, Ordering::Equal);
            assert_eq!(
                board_orientation_for(&mirrored, side),
                orientation.reverse()
            );
            let code = |square| {
                position
                    .piece_at(canonical_square(side, square))
                    .map_or(0, |piece| piece_absolute_feature_index(side, piece) + 1)
            };
            let full_scan = (0..90)
                .map(|square| code(square).cmp(&code(mirror_file_square(square))))
                .find(|&ordering| ordering != Ordering::Equal)
                .unwrap_or(Ordering::Equal);
            assert_eq!(orientation, full_scan);
        }
    }

    #[test]
    fn reflection_history_is_an_involution_including_zero_bits() {
        let features = std::array::from_fn(|i| {
            if i % 7 == 0 {
                -0.0
            } else {
                (i as f32 - 50.0) / 64.0
            }
        });
        let twice = mirror_history_features(&mirror_history_features(&features));
        for (a, b) in features.map(clean_zero).iter().zip(twice) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn reflection_orientation_uses_history_and_candidate_flags() {
        let position = Position::startpos();
        assert_eq!(board_orientation(&position), Ordering::Equal);
        let moves = position.legal_moves();
        let mut history = [0.0; HISTORY_CONTEXT_SIZE];
        history[1] = 0.25;
        let orientation = input_orientation(&position, &history, &moves, &[]);
        assert_ne!(orientation, Ordering::Equal);
        assert_eq!(
            input_orientation(&position, &mirror_history_features(&history), &moves, &[]),
            orientation.reverse()
        );
        history[1] = 0.0;
        let pair = moves
            .iter()
            .position(|&mv| mirror_file_move(mv) != mv)
            .unwrap();
        let mut flags = vec![0; moves.len()];
        flags[pair] = 1;
        let mirrored_moves = moves
            .iter()
            .copied()
            .map(mirror_file_move)
            .collect::<Vec<_>>();
        let orientation = input_orientation(&position, &history, &moves, &flags);
        assert_ne!(orientation, Ordering::Equal);
        assert_eq!(
            input_orientation(&position, &history, &mirrored_moves, &flags),
            orientation.reverse()
        );
        assert_eq!(
            input_orientation(&position, &history, &moves, &[]),
            Ordering::Equal
        );
    }

    #[test]
    fn reflection_fixedpoint_logits_are_equal_for_each_move_pair() {
        let position = Position::startpos();
        let moves = position.legal_moves();
        let mut logits = (0..moves.len()).map(|i| i as f32).collect::<Vec<_>>();
        symmetrize_policy_logits(&moves, &mut logits);
        for (i, &mv) in moves.iter().enumerate() {
            let j = moves
                .iter()
                .position(|&other| other == mirror_file_move(mv))
                .unwrap();
            assert_eq!(logits[i].to_bits(), logits[j].to_bits());
        }
    }
}
