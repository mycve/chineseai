use super::{
    BOARD_SIZE, Color, OnceLock, file_of, index, inside_board, inside_palace, rank_of,
    soldier_crossed_river,
};

pub(super) fn orthogonal_ray_masks() -> &'static [[u128; 4]; BOARD_SIZE] {
    static RAYS: OnceLock<[[u128; 4]; BOARD_SIZE]> = OnceLock::new();
    RAYS.get_or_init(|| {
        let mut rays = [[0u128; 4]; BOARD_SIZE];
        for source in 0..BOARD_SIZE {
            let file = file_of(source) as i32;
            let rank = rank_of(source) as i32;
            for (direction, (df, dr)) in ORTHOGONAL_STEPS.into_iter().enumerate() {
                let mut target_file = file + df;
                let mut target_rank = rank + dr;
                while inside_board(target_file, target_rank) {
                    rays[source][direction] |=
                        1u128 << index(target_file as usize, target_rank as usize);
                    target_file += df;
                    target_rank += dr;
                }
            }
        }
        rays
    })
}

pub(super) fn fixed_attack_masks() -> &'static [[[u128; BOARD_SIZE]; 3]; 2] {
    static MASKS: OnceLock<[[[u128; BOARD_SIZE]; 3]; 2]> = OnceLock::new();
    MASKS.get_or_init(|| {
        let mut masks = [[[0u128; BOARD_SIZE]; 3]; 2];
        for (color_index, color) in [Color::Red, Color::Black].into_iter().enumerate() {
            for source in 0..BOARD_SIZE {
                let file = file_of(source) as i32;
                let rank = rank_of(source) as i32;
                let mut add = |kind: usize, df: i32, dr: i32, palace: bool| {
                    let target_file = file + df;
                    let target_rank = rank + dr;
                    if inside_board(target_file, target_rank)
                        && (!palace
                            || inside_palace(color, target_file as usize, target_rank as usize))
                    {
                        masks[color_index][kind][source] |=
                            1u128 << index(target_file as usize, target_rank as usize);
                    }
                };
                for (df, dr) in ORTHOGONAL_STEPS {
                    add(0, df, dr, true);
                }
                for (df, dr) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
                    add(1, df, dr, true);
                }
                add(2, 0, color.forward_step(), false);
                if soldier_crossed_river(color, rank as usize) {
                    add(2, -1, 0, false);
                    add(2, 1, 0, false);
                }
            }
        }
        masks
    })
}

#[inline(always)]
pub(crate) fn nearest_on_ray(blockers: u128, increasing: bool) -> usize {
    if increasing {
        blockers.trailing_zeros() as usize
    } else {
        127 - blockers.leading_zeros() as usize
    }
}

#[inline(always)]
pub(super) fn ray_through(ray: u128, square: usize, increasing: bool) -> u128 {
    if increasing {
        ray & ((1u128 << (square + 1)) - 1)
    } else {
        ray & (!0u128 << square)
    }
}

/// 从棋盘坐标加上偏移得到格子；越界返回 `None`。
#[inline]
pub(super) fn offset_square(file: i32, rank: i32, df: i32, dr: i32) -> Option<usize> {
    let file = file + df;
    let rank = rank + dr;
    inside_board(file, rank).then(|| index(file as usize, rank as usize))
}

pub(super) const ORTHOGONAL_STEPS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
pub(super) const DIAGONAL_STEPS: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
pub(super) const ELEPHANT_STEPS: [((i32, i32), (i32, i32)); 4] = [
    ((1, 1), (2, 2)),
    ((1, -1), (2, -2)),
    ((-1, 1), (-2, 2)),
    ((-1, -1), (-2, -2)),
];
pub(super) const HORSE_STEPS: [((i32, i32), (i32, i32)); 8] = [
    ((0, -1), (-1, -2)),
    ((0, -1), (1, -2)),
    ((0, 1), (-1, 2)),
    ((0, 1), (1, 2)),
    ((-1, 0), (-2, -1)),
    ((-1, 0), (-2, 1)),
    ((1, 0), (2, -1)),
    ((1, 0), (2, 1)),
];
