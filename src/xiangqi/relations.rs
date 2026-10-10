use super::{
    HORSE_STEPS, ORTHOGONAL_STEPS, Piece, PieceKind, Position, elephant_stays_home, file_of, index,
    inside_board, inside_palace, nearest_on_ray, orthogonal_ray_masks, rank_of,
    soldier_crossed_river,
};

impl Position {
    /// Visit occupied piece relations used by the network's relation head.
    ///
    /// For a cannon the first occupied square is emitted as its screen and the
    /// second as its capture target.  This gives the network the missing
    /// cannon-screen relation without another board scan.  Slider rays use the
    /// position occupancy bitboard, so they jump directly between blockers.
    pub(crate) fn visit_occupied_relations(&self, visitor: impl FnMut(usize, Piece, usize, Piece)) {
        self.visit_piece_relations::<true>(visitor);
    }

    pub(crate) fn visit_occupied_leaper_relations(
        &self,
        visitor: impl FnMut(usize, Piece, usize, Piece),
    ) {
        self.visit_piece_relations::<false>(visitor);
    }

    fn visit_piece_relations<const SLIDERS: bool>(
        &self,
        mut visitor: impl FnMut(usize, Piece, usize, Piece),
    ) {
        let mut pieces = self.occupied;
        while pieces != 0 {
            let source = pieces.trailing_zeros() as usize;
            pieces &= pieces - 1;
            let piece = self.board[source].expect("occupancy and board must agree");
            if !SLIDERS && matches!(piece.kind, PieceKind::Rook | PieceKind::Cannon) {
                continue;
            }
            let file = file_of(source) as i32;
            let rank = rank_of(source) as i32;
            let mut visit_step = |df: i32, dr: i32| {
                let target_file = file + df;
                let target_rank = rank + dr;
                if inside_board(target_file, target_rank) {
                    let target = index(target_file as usize, target_rank as usize);
                    if let Some(attacked) = self.board[target] {
                        visitor(source, piece, target, attacked);
                    }
                }
            };
            match piece.kind {
                PieceKind::General => {
                    for (df, dr) in ORTHOGONAL_STEPS {
                        let target_file = file + df;
                        let target_rank = rank + dr;
                        if inside_board(target_file, target_rank)
                            && inside_palace(
                                piece.color,
                                target_file as usize,
                                target_rank as usize,
                            )
                        {
                            visit_step(df, dr);
                        }
                    }
                }
                PieceKind::Advisor => {
                    for (df, dr) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
                        let target_file = file + df;
                        let target_rank = rank + dr;
                        if inside_board(target_file, target_rank)
                            && inside_palace(
                                piece.color,
                                target_file as usize,
                                target_rank as usize,
                            )
                        {
                            visit_step(df, dr);
                        }
                    }
                }
                PieceKind::Elephant => {
                    for (df, dr) in [(-2, -2), (-2, 2), (2, -2), (2, 2)] {
                        let target_file = file + df;
                        let target_rank = rank + dr;
                        if inside_board(target_file, target_rank)
                            && elephant_stays_home(piece.color, target_rank as usize)
                            && self.board[index((file + df / 2) as usize, (rank + dr / 2) as usize)]
                                .is_none()
                        {
                            visit_step(df, dr);
                        }
                    }
                }
                PieceKind::Horse => {
                    for ((leg_df, leg_dr), (df, dr)) in HORSE_STEPS {
                        if inside_board(file + df, rank + dr)
                            && inside_board(file + leg_df, rank + leg_dr)
                            && self.board[index((file + leg_df) as usize, (rank + leg_dr) as usize)]
                                .is_none()
                        {
                            visit_step(df, dr);
                        }
                    }
                }
                PieceKind::Rook | PieceKind::Cannon => {
                    for direction in 0..4 {
                        let increasing = direction == 0 || direction == 2;
                        let blockers = orthogonal_ray_masks()[source][direction] & self.occupied;
                        if blockers == 0 {
                            continue;
                        }
                        let first = nearest_on_ray(blockers, increasing);
                        visitor(
                            source,
                            piece,
                            first,
                            self.board[first].expect("blocker must be occupied"),
                        );
                        if piece.kind == PieceKind::Cannon {
                            let remaining = blockers & !(1u128 << first);
                            if remaining != 0 {
                                let second = nearest_on_ray(remaining, increasing);
                                visitor(
                                    source,
                                    piece,
                                    second,
                                    self.board[second].expect("blocker must be occupied"),
                                );
                            }
                        }
                    }
                }
                PieceKind::Soldier => {
                    visit_step(0, piece.color.forward_step());
                    if soldier_crossed_river(piece.color, rank as usize) {
                        visit_step(-1, 0);
                        visit_step(1, 0);
                    }
                }
            }
        }
    }
}
