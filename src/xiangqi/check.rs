use super::{
    Color, ELEPHANT_STEPS, HORSE_STEPS, Move, ORTHOGONAL_STEPS, Piece, PieceKind, Position,
    elephant_stays_home, file_of, index, inside_board, inside_palace, rank_of,
    soldier_crossed_river,
};

impl Position {
    #[cfg(test)]
    pub(crate) fn gives_check_after_move(&mut self, mv: Move) -> bool {
        let captured = self.make_move_board_only(mv);
        let gives_check = self.in_check(self.side_to_move.opposite());
        self.unmake_move_board_only(mv, captured);
        gives_check
    }

    /// Non-mutating equivalent of `gives_check_after_move`.
    ///
    /// It is only valid to call this while it is `side_to_move`'s turn and the
    /// opponent's king is not already in check by `side_to_move` (true after any
    /// legal opponent move). Under that precondition, after `mv` the enemy king
    /// can only newly be in check via the changes at `mv.from`/`mv.to`, so no
    /// full-board make/unmake or scan is needed.
    pub fn gives_check_after_move_fast(&self, mv: Move) -> bool {
        let s = self.side_to_move;
        let from = mv.from as usize;
        let to = mv.to as usize;
        let Some(g) = self.find_general(s.opposite()) else {
            return false;
        };
        let Some(moving) = self.board[from] else {
            return false;
        };
        // Virtual occupancy after mv: `from` vacated, `to` holds `moving`.
        let occ = |sq: usize| -> Option<Piece> {
            if sq == from {
                None
            } else if sq == to {
                Some(moving)
            } else {
                self.board[sq]
            }
        };

        // Direct check by the moved piece from its destination.
        if self.piece_attacks_virtual(moving, to, g, &occ) {
            return true;
        }

        // Discovered checks on the orthogonal sliders / facing-generals rays
        // that pass through `from` (vacated) or `to` (newly occupied as a screen).
        for (df, dr) in ORTHOGONAL_STEPS {
            if !self.ray_contains(g, from, df, dr) && !self.ray_contains(g, to, df, dr) {
                continue;
            }
            let gf = file_of(g) as i32;
            let gr = rank_of(g) as i32;
            let mut nf = gf + df;
            let mut nr = gr + dr;
            let mut seen_screen = false;
            while inside_board(nf, nr) {
                let sq = index(nf as usize, nr as usize);
                if let Some(piece) = occ(sq) {
                    if !seen_screen {
                        if piece.color == s
                            && (piece.kind == PieceKind::Rook
                                || (piece.kind == PieceKind::General && df == 0))
                        {
                            return true;
                        }
                        seen_screen = true;
                    } else if piece.color == s && piece.kind == PieceKind::Cannon {
                        return true;
                    } else {
                        break;
                    }
                }
                nf += df;
                nr += dr;
            }
        }

        // Vacating `from` may uncover a friendly horse/elephant whose leg/eye
        // square was `from`.
        let from_file_delta = file_of(g).abs_diff(file_of(from));
        let from_rank_delta = rank_of(g).abs_diff(rank_of(from));
        if from_file_delta == 1 && from_rank_delta == 1 {
            if self.leaper_revealed_horse(g, from, s, &occ)
                || self.leaper_revealed_elephant(g, from, s, &occ)
            {
                return true;
            }
        }

        false
    }

    fn ray_contains(&self, g: usize, q: usize, df: i32, dr: i32) -> bool {
        let gf = file_of(g) as i32;
        let gr = rank_of(g) as i32;
        let qf = file_of(q) as i32;
        let qr = rank_of(q) as i32;
        if dr == 0 {
            qr == gr && (qf - gf) * df > 0
        } else {
            qf == gf && (qr - gr) * dr > 0
        }
    }

    fn leaper_revealed_horse(
        &self,
        g: usize,
        from: usize,
        s: Color,
        occ: &dyn Fn(usize) -> Option<Piece>,
    ) -> bool {
        let gf = file_of(g) as i32;
        let gr = rank_of(g) as i32;
        for ((leg_df, leg_dr), (move_df, move_dr)) in HORSE_STEPS {
            let from_f = gf - move_df;
            let from_r = gr - move_dr;
            if !inside_board(from_f, from_r) {
                continue;
            }
            let leg_f = from_f + leg_df;
            let leg_r = from_r + leg_dr;
            if !inside_board(leg_f, leg_r) {
                continue;
            }
            let origin = index(from_f as usize, from_r as usize);
            let leg = index(leg_f as usize, leg_r as usize);
            if leg == from
                && matches!(
                    occ(origin),
                    Some(Piece {
                        color: c,
                        kind: PieceKind::Horse
                    }) if c == s
                )
            {
                return true;
            }
        }
        false
    }

    fn leaper_revealed_elephant(
        &self,
        g: usize,
        from: usize,
        s: Color,
        occ: &dyn Fn(usize) -> Option<Piece>,
    ) -> bool {
        let gf = file_of(g) as i32;
        let gr = rank_of(g) as i32;
        for ((eye_df, eye_dr), (move_df, move_dr)) in ELEPHANT_STEPS {
            let from_f = gf - move_df;
            let from_r = gr - move_dr;
            if !inside_board(from_f, from_r) {
                continue;
            }
            let eye_f = from_f + eye_df;
            let eye_r = from_r + eye_dr;
            if !inside_board(eye_f, eye_r) {
                continue;
            }
            let origin = index(from_f as usize, from_r as usize);
            let eye = index(eye_f as usize, eye_r as usize);
            if eye == from
                && matches!(
                    occ(origin),
                    Some(Piece {
                        color: c,
                        kind: PieceKind::Elephant
                    }) if c == s
                )
                && elephant_stays_home(s, rank_of(g))
            {
                return true;
            }
        }
        false
    }

    fn piece_attacks_virtual(
        &self,
        piece: Piece,
        sq: usize,
        target: usize,
        occ: &dyn Fn(usize) -> Option<Piece>,
    ) -> bool {
        let pf = file_of(sq) as i32;
        let pr = rank_of(sq) as i32;
        let tf = file_of(target) as i32;
        let tr = rank_of(target) as i32;
        let df = tf - pf;
        let dr = tr - pr;
        match piece.kind {
            PieceKind::General => {
                if (df.abs() + dr.abs()) == 1
                    && inside_palace(piece.color, tf as usize, tr as usize)
                {
                    return true;
                }
                pf == tf
                    && occ(target)
                        .is_some_and(|p| p.kind == PieceKind::General && p.color != piece.color)
                    && self.clear_line_between_virtual(sq, target, occ)
            }
            PieceKind::Advisor => {
                df.abs() == 1
                    && dr.abs() == 1
                    && inside_palace(piece.color, tf as usize, tr as usize)
            }
            PieceKind::Elephant => {
                df.abs() == 2
                    && dr.abs() == 2
                    && elephant_stays_home(piece.color, tr as usize)
                    && occ(index(((pf + tf) / 2) as usize, ((pr + tr) / 2) as usize)).is_none()
            }
            PieceKind::Horse => {
                for ((leg_df, leg_dr), (move_df, move_dr)) in HORSE_STEPS {
                    if df == move_df && dr == move_dr {
                        let leg_f = pf + leg_df;
                        let leg_r = pr + leg_dr;
                        return occ(index(leg_f as usize, leg_r as usize)).is_none();
                    }
                }
                false
            }
            PieceKind::Rook => {
                (pf == tf || pr == tr) && self.clear_line_between_virtual(sq, target, occ)
            }
            PieceKind::Cannon => {
                (pf == tf || pr == tr) && self.count_between_virtual(sq, target, occ) == 1
            }
            PieceKind::Soldier => {
                if tf == pf && tr == pr + piece.color.forward_step() {
                    return true;
                }
                soldier_crossed_river(piece.color, tr as usize) && tr == pr && df.abs() == 1
            }
        }
    }

    fn clear_line_between_virtual(
        &self,
        a: usize,
        b: usize,
        occ: &dyn Fn(usize) -> Option<Piece>,
    ) -> bool {
        self.count_between_virtual(a, b, occ) == 0
    }

    fn count_between_virtual(
        &self,
        a: usize,
        b: usize,
        occ: &dyn Fn(usize) -> Option<Piece>,
    ) -> usize {
        if file_of(a) == file_of(b) {
            let file = file_of(a);
            let start = rank_of(a).min(rank_of(b)) + 1;
            let end = rank_of(a).max(rank_of(b));
            (start..end)
                .filter(|rank| occ(index(file, *rank)).is_some())
                .count()
        } else if rank_of(a) == rank_of(b) {
            let rank = rank_of(a);
            let start = file_of(a).min(file_of(b)) + 1;
            let end = file_of(a).max(file_of(b));
            (start..end)
                .filter(|file| occ(index(*file, rank)).is_some())
                .count()
        } else {
            usize::MAX
        }
    }
}
