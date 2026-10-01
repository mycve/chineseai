use super::{
    BOARD_FILES, Color, DIAGONAL_STEPS, ELEPHANT_STEPS, HORSE_STEPS, ORTHOGONAL_STEPS, Piece,
    PieceKind, Position, color_hash_index, elephant_stays_home, file_of, fixed_attack_masks, index,
    inside_board, inside_palace, nearest_on_ray, orthogonal_ray_masks, rank_of, ray_through,
    soldier_crossed_river,
};

impl Position {
    pub(crate) fn attacked_squares_mask(&self, by: Color) -> u128 {
        self.attacked_squares_masks()[color_hash_index(by)]
    }

    pub(crate) fn attacked_squares_masks(&self) -> [u128; 2] {
        let mut masks = [0u128; 2];
        let occupancy = self.occupied;
        let mut pieces = occupancy;
        while pieces != 0 {
            let source = pieces.trailing_zeros() as usize;
            pieces &= pieces - 1;
            let Some(piece) = self.board[source] else {
                continue;
            };
            let by = piece.color;
            let mask = &mut masks[color_hash_index(by)];
            let file = file_of(source) as i32;
            let rank = rank_of(source) as i32;
            let mut add_step = |df: i32, dr: i32| {
                let target_file = file + df;
                let target_rank = rank + dr;
                if inside_board(target_file, target_rank) {
                    *mask |= 1u128 << index(target_file as usize, target_rank as usize);
                }
            };
            match piece.kind {
                PieceKind::General => {
                    *mask |= fixed_attack_masks()[color_hash_index(by)][0][source];
                    if let Some(enemy_general) = self.find_general(by.opposite())
                        && file_of(enemy_general) == file as usize
                        && self.clear_file_between(source, enemy_general)
                    {
                        *mask |= 1u128 << enemy_general;
                    }
                }
                PieceKind::Advisor => {
                    *mask |= fixed_attack_masks()[color_hash_index(by)][1][source];
                }
                PieceKind::Elephant => {
                    for (df, dr) in [(-2, -2), (-2, 2), (2, -2), (2, 2)] {
                        let target_file = file + df;
                        let target_rank = rank + dr;
                        if inside_board(target_file, target_rank)
                            && elephant_stays_home(by, target_rank as usize)
                            && self.board[index((file + df / 2) as usize, (rank + dr / 2) as usize)]
                                .is_none()
                        {
                            add_step(df, dr);
                        }
                    }
                }
                PieceKind::Horse => {
                    for ((leg_df, leg_dr), (df, dr)) in HORSE_STEPS {
                        if inside_board(file + df, rank + dr)
                            && self.board[index((file + leg_df) as usize, (rank + leg_dr) as usize)]
                                .is_none()
                        {
                            add_step(df, dr);
                        }
                    }
                }
                PieceKind::Rook | PieceKind::Cannon => {
                    for (direction, (df, dr)) in ORTHOGONAL_STEPS.into_iter().enumerate() {
                        let ray = orthogonal_ray_masks()[source][direction];
                        let increasing = dr * BOARD_FILES as i32 + df > 0;
                        let blockers = ray & occupancy;
                        if piece.kind == PieceKind::Rook {
                            *mask |= if blockers == 0 {
                                ray
                            } else {
                                ray_through(ray, nearest_on_ray(blockers, increasing), increasing)
                            };
                            continue;
                        }
                        if blockers == 0 {
                            continue;
                        }
                        let screen = nearest_on_ray(blockers, increasing);
                        let beyond_screen = if increasing {
                            ray & (!0u128 << (screen + 1))
                        } else {
                            ray & ((1u128 << screen) - 1)
                        };
                        let second_blockers = beyond_screen & occupancy;
                        *mask |= if second_blockers == 0 {
                            beyond_screen
                        } else {
                            ray_through(
                                beyond_screen,
                                nearest_on_ray(second_blockers, increasing),
                                increasing,
                            )
                        };
                    }
                }
                PieceKind::Soldier => {
                    *mask |= fixed_attack_masks()[color_hash_index(by)][2][source];
                }
            }
        }
        masks
    }

    pub(super) fn add_horse_leg_attack_mask(&self, target: usize, by: Color, mask: &mut u128) {
        let file = file_of(target) as i32;
        let rank = rank_of(target) as i32;
        for ((leg_df, leg_dr), (move_df, move_dr)) in HORSE_STEPS {
            let from_file = file - move_df;
            let from_rank = rank - move_dr;
            if !inside_board(from_file, from_rank) {
                continue;
            }
            let leg_file = from_file + leg_df;
            let leg_rank = from_rank + leg_dr;
            if !inside_board(leg_file, leg_rank) {
                continue;
            }
            let from = index(from_file as usize, from_rank as usize);
            if matches!(
                self.board[from],
                Some(Piece {
                    color,
                    kind: PieceKind::Horse
                }) if color == by
            ) {
                *mask |= 1u128 << index(leg_file as usize, leg_rank as usize);
            }
        }
    }

    pub(crate) fn is_square_attacked(&self, target: usize, by: Color) -> bool {
        self.is_square_attacked_by_leapers(target, by)
            || self.is_square_attacked_by_sliders(target, by)
    }

    pub(super) fn visit_attacker_origins_to<F>(
        &self,
        target: usize,
        by: Color,
        mut visitor: F,
    ) -> bool
    where
        F: FnMut(usize) -> bool,
    {
        self.visit_leaper_attackers(target, by, &mut visitor)
            || self.visit_slider_attackers(target, by, &mut visitor)
    }

    fn visit_leaper_attackers<F>(&self, target: usize, by: Color, visitor: &mut F) -> bool
    where
        F: FnMut(usize) -> bool,
    {
        let file = file_of(target) as i32;
        let rank = rank_of(target) as i32;

        for (df, dr) in ORTHOGONAL_STEPS {
            let from_file = file - df;
            let from_rank = rank - dr;
            if !inside_board(from_file, from_rank) {
                continue;
            }
            let from = index(from_file as usize, from_rank as usize);
            if matches!(
                self.board[from],
                Some(Piece {
                    color,
                    kind: PieceKind::General
                }) if color == by
                    && inside_palace(by, file as usize, rank as usize)
                    && inside_palace(by, from_file as usize, from_rank as usize)
            ) && visitor(from)
            {
                return true;
            }
        }

        for (df, dr) in DIAGONAL_STEPS {
            let from_file = file - df;
            let from_rank = rank - dr;
            if !inside_board(from_file, from_rank) {
                continue;
            }
            let from = index(from_file as usize, from_rank as usize);
            if matches!(
                self.board[from],
                Some(Piece {
                    color,
                    kind: PieceKind::Advisor
                }) if color == by
                    && inside_palace(by, file as usize, rank as usize)
                    && inside_palace(by, from_file as usize, from_rank as usize)
            ) && visitor(from)
            {
                return true;
            }
        }

        for ((leg_df, leg_dr), (move_df, move_dr)) in HORSE_STEPS {
            let from_file = file - move_df;
            let from_rank = rank - move_dr;
            if !inside_board(from_file, from_rank) {
                continue;
            }
            let leg_file = from_file + leg_df;
            let leg_rank = from_rank + leg_dr;
            if !inside_board(leg_file, leg_rank) {
                continue;
            }
            let from = index(from_file as usize, from_rank as usize);
            let leg = index(leg_file as usize, leg_rank as usize);
            if self.board[leg].is_none()
                && matches!(
                    self.board[from],
                    Some(Piece {
                        color,
                        kind: PieceKind::Horse
                    }) if color == by
                )
                && visitor(from)
            {
                return true;
            }
        }

        for ((eye_df, eye_dr), (move_df, move_dr)) in ELEPHANT_STEPS {
            let from_file = file - move_df;
            let from_rank = rank - move_dr;
            if !inside_board(from_file, from_rank) {
                continue;
            }
            let eye_file = from_file + eye_df;
            let eye_rank = from_rank + eye_dr;
            if !inside_board(eye_file, eye_rank) {
                continue;
            }
            let from = index(from_file as usize, from_rank as usize);
            let eye = index(eye_file as usize, eye_rank as usize);
            if self.board[eye].is_none()
                && matches!(
                    self.board[from],
                    Some(Piece {
                        color,
                        kind: PieceKind::Elephant
                    }) if color == by && elephant_stays_home(by, rank as usize)
                )
                && visitor(from)
            {
                return true;
            }
        }

        let soldier_forward_from_rank = rank - by.forward_step();
        if inside_board(file, soldier_forward_from_rank) {
            let from = index(file as usize, soldier_forward_from_rank as usize);
            if matches!(
                self.board[from],
                Some(Piece {
                    color,
                    kind: PieceKind::Soldier
                }) if color == by
            ) && visitor(from)
            {
                return true;
            }
        }

        for side_df in [-1, 1] {
            let from_file = file - side_df;
            if !inside_board(from_file, rank) {
                continue;
            }
            let from = index(from_file as usize, rank as usize);
            if matches!(
                self.board[from],
                Some(Piece {
                    color,
                    kind: PieceKind::Soldier
                }) if color == by && soldier_crossed_river(by, rank as usize)
            ) && visitor(from)
            {
                return true;
            }
        }

        false
    }

    fn visit_slider_attackers<F>(&self, target: usize, by: Color, visitor: &mut F) -> bool
    where
        F: FnMut(usize) -> bool,
    {
        let file = file_of(target) as i32;
        let rank = rank_of(target) as i32;

        for (df, dr) in ORTHOGONAL_STEPS {
            let mut seen_screen = false;
            let mut nf = file + df;
            let mut nr = rank + dr;

            while inside_board(nf, nr) {
                let sq = index(nf as usize, nr as usize);
                if let Some(piece) = self.board[sq] {
                    if !seen_screen {
                        if piece.color == by {
                            if piece.kind == PieceKind::Rook && visitor(sq) {
                                return true;
                            }
                            if piece.kind == PieceKind::General
                                && df == 0
                                && matches!(
                                    self.board[target],
                                    Some(Piece {
                                        color,
                                        kind: PieceKind::General
                                    }) if color == by.opposite()
                                )
                                && visitor(sq)
                            {
                                return true;
                            }
                        }
                        seen_screen = true;
                    } else {
                        if piece.color == by && piece.kind == PieceKind::Cannon && visitor(sq) {
                            return true;
                        }
                        break;
                    }
                }

                nf += df;
                nr += dr;
            }
        }

        false
    }

    pub(super) fn is_square_attacked_by_leapers(&self, target: usize, by: Color) -> bool {
        let file = file_of(target) as i32;
        let rank = rank_of(target) as i32;

        for (df, dr) in ORTHOGONAL_STEPS {
            let from_file = file - df;
            let from_rank = rank - dr;
            if !inside_board(from_file, from_rank) {
                continue;
            }
            let from = index(from_file as usize, from_rank as usize);
            if matches!(
                self.board[from],
                Some(Piece {
                    color,
                    kind: PieceKind::General
                }) if color == by
                    && inside_palace(by, file as usize, rank as usize)
                    && inside_palace(by, from_file as usize, from_rank as usize)
            ) {
                return true;
            }
        }

        for (df, dr) in DIAGONAL_STEPS {
            let from_file = file - df;
            let from_rank = rank - dr;
            if !inside_board(from_file, from_rank) {
                continue;
            }
            let from = index(from_file as usize, from_rank as usize);
            if matches!(
                self.board[from],
                Some(Piece {
                    color,
                    kind: PieceKind::Advisor
                }) if color == by
                    && inside_palace(by, file as usize, rank as usize)
                    && inside_palace(by, from_file as usize, from_rank as usize)
            ) {
                return true;
            }
        }

        for ((leg_df, leg_dr), (move_df, move_dr)) in HORSE_STEPS {
            let from_file = file - move_df;
            let from_rank = rank - move_dr;
            if !inside_board(from_file, from_rank) {
                continue;
            }
            let leg_file = from_file + leg_df;
            let leg_rank = from_rank + leg_dr;
            if !inside_board(leg_file, leg_rank) {
                continue;
            }
            let from = index(from_file as usize, from_rank as usize);
            let leg = index(leg_file as usize, leg_rank as usize);
            if self.board[leg].is_none()
                && matches!(
                    self.board[from],
                    Some(Piece {
                        color,
                        kind: PieceKind::Horse
                    }) if color == by
                )
            {
                return true;
            }
        }

        for ((eye_df, eye_dr), (move_df, move_dr)) in ELEPHANT_STEPS {
            let from_file = file - move_df;
            let from_rank = rank - move_dr;
            if !inside_board(from_file, from_rank) {
                continue;
            }
            let eye_file = from_file + eye_df;
            let eye_rank = from_rank + eye_dr;
            if !inside_board(eye_file, eye_rank) {
                continue;
            }
            let from = index(from_file as usize, from_rank as usize);
            let eye = index(eye_file as usize, eye_rank as usize);
            if self.board[eye].is_none()
                && matches!(
                    self.board[from],
                    Some(Piece {
                        color,
                        kind: PieceKind::Elephant
                    }) if color == by && elephant_stays_home(by, rank as usize)
                )
            {
                return true;
            }
        }

        let soldier_forward_from_rank = rank - by.forward_step();
        if inside_board(file, soldier_forward_from_rank) {
            let from = index(file as usize, soldier_forward_from_rank as usize);
            if matches!(
                self.board[from],
                Some(Piece {
                    color,
                    kind: PieceKind::Soldier
                }) if color == by
            ) {
                return true;
            }
        }

        for side_df in [-1, 1] {
            let from_file = file - side_df;
            if !inside_board(from_file, rank) {
                continue;
            }
            let from = index(from_file as usize, rank as usize);
            if matches!(
                self.board[from],
                Some(Piece {
                    color,
                    kind: PieceKind::Soldier
                }) if color == by && soldier_crossed_river(by, rank as usize)
            ) {
                return true;
            }
        }

        false
    }

    fn is_square_attacked_by_sliders(&self, target: usize, by: Color) -> bool {
        let file = file_of(target) as i32;
        let rank = rank_of(target) as i32;

        for (df, dr) in ORTHOGONAL_STEPS {
            let mut seen_screen = false;
            let mut nf = file + df;
            let mut nr = rank + dr;

            while inside_board(nf, nr) {
                let sq = index(nf as usize, nr as usize);
                if let Some(piece) = self.board[sq] {
                    if !seen_screen {
                        if piece.color == by {
                            if piece.kind == PieceKind::Rook {
                                return true;
                            }
                            if piece.kind == PieceKind::General
                                && df == 0
                                && matches!(
                                    self.board[target],
                                    Some(Piece {
                                        color,
                                        kind: PieceKind::General
                                    }) if color == by.opposite()
                                )
                            {
                                return true;
                            }
                        }
                        seen_screen = true;
                    } else if piece.color == by && piece.kind == PieceKind::Cannon {
                        return true;
                    } else {
                        break;
                    }
                }

                nf += df;
                nr += dr;
            }
        }

        false
    }
}
