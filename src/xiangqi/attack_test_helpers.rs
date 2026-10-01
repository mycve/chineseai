#[cfg(test)]
use super::geom::same_rank_or_file;
use super::{
    BOARD_SIZE, Color, HORSE_STEPS, Piece, PieceKind, Position, elephant_stays_home, file_of,
    index, inside_palace, rank_of, soldier_crossed_river,
};

impl Position {
    #[cfg(test)]
    pub(super) fn piece_attacks_square(&self, sq: usize, piece: Piece, target: usize) -> bool {
        match piece.kind {
            PieceKind::General => self.general_attacks_square(sq, piece, target),
            PieceKind::Advisor => self.advisor_attacks_square(sq, piece, target),
            PieceKind::Elephant => self.elephant_attacks_square(sq, piece, target),
            PieceKind::Horse => self.horse_attacks_square(sq, piece, target),
            PieceKind::Rook => self.rook_attacks_square(sq, target),
            PieceKind::Cannon => self.cannon_attacks_square(sq, target),
            PieceKind::Soldier => self.soldier_attacks_square(sq, piece, target),
        }
    }

    #[cfg(test)]
    pub(super) fn general_attacks_square(&self, sq: usize, piece: Piece, target: usize) -> bool {
        let file = file_of(sq) as i32;
        let rank = rank_of(sq) as i32;
        let tf = file_of(target) as i32;
        let tr = rank_of(target) as i32;

        if (file - tf).abs() + (rank - tr).abs() == 1
            && inside_palace(piece.color, tf as usize, tr as usize)
        {
            return true;
        }

        matches!(
            self.board[target],
            Some(Piece {
                color,
                kind: PieceKind::General
            }) if color != piece.color
        ) && file_of(sq) == file_of(target)
            && self.clear_file_between(sq, target)
    }

    #[cfg(test)]
    pub(super) fn advisor_attacks_square(&self, sq: usize, piece: Piece, target: usize) -> bool {
        let file = file_of(sq) as i32;
        let rank = rank_of(sq) as i32;
        let tf = file_of(target) as i32;
        let tr = rank_of(target) as i32;
        (file - tf).abs() == 1
            && (rank - tr).abs() == 1
            && inside_palace(piece.color, tf as usize, tr as usize)
    }

    #[cfg(test)]
    pub(super) fn elephant_attacks_square(&self, sq: usize, piece: Piece, target: usize) -> bool {
        let file = file_of(sq) as i32;
        let rank = rank_of(sq) as i32;
        let tf = file_of(target) as i32;
        let tr = rank_of(target) as i32;
        if (file - tf).abs() != 2 || (rank - tr).abs() != 2 {
            return false;
        }
        if !elephant_stays_home(piece.color, tr as usize) {
            return false;
        }
        let eye_f = (file + tf) / 2;
        let eye_r = (rank + tr) / 2;
        self.board[index(eye_f as usize, eye_r as usize)].is_none()
    }

    #[cfg(test)]
    pub(super) fn horse_attacks_square(&self, sq: usize, _piece: Piece, target: usize) -> bool {
        let file = file_of(sq) as i32;
        let rank = rank_of(sq) as i32;
        let tf = file_of(target) as i32;
        let tr = rank_of(target) as i32;
        let df = tf - file;
        let dr = tr - rank;

        for ((leg_df, leg_dr), (move_df, move_dr)) in HORSE_STEPS {
            if df == move_df && dr == move_dr {
                let leg_f = file + leg_df;
                let leg_r = rank + leg_dr;
                return self.board[index(leg_f as usize, leg_r as usize)].is_none();
            }
        }
        false
    }

    #[cfg(test)]
    pub(super) fn rook_attacks_square(&self, sq: usize, target: usize) -> bool {
        if !same_rank_or_file(sq, target) {
            return false;
        }
        self.clear_line_between(sq, target)
    }

    #[cfg(test)]
    pub(super) fn cannon_attacks_square(&self, sq: usize, target: usize) -> bool {
        if !same_rank_or_file(sq, target) {
            return false;
        }
        self.count_between(sq, target) == 1
    }

    #[cfg(test)]
    pub(super) fn soldier_attacks_square(&self, sq: usize, piece: Piece, target: usize) -> bool {
        let file = file_of(sq) as i32;
        let rank = rank_of(sq) as i32;
        let tf = file_of(target) as i32;
        let tr = rank_of(target) as i32;

        if tf == file && tr == rank + piece.color.forward_step() {
            return true;
        }

        soldier_crossed_river(piece.color, rank as usize) && tr == rank && (tf - file).abs() == 1
    }

    #[cfg(test)]
    pub(super) fn clear_line_between(&self, a: usize, b: usize) -> bool {
        self.count_between(a, b) == 0
    }

    #[cfg(test)]
    pub(super) fn count_between(&self, a: usize, b: usize) -> usize {
        if file_of(a) == file_of(b) {
            let file = file_of(a);
            let start = rank_of(a).min(rank_of(b)) + 1;
            let end = rank_of(a).max(rank_of(b));
            (start..end)
                .filter(|rank| self.board[index(file, *rank)].is_some())
                .count()
        } else if rank_of(a) == rank_of(b) {
            let rank = rank_of(a);
            let start = file_of(a).min(file_of(b)) + 1;
            let end = file_of(a).max(file_of(b));
            (start..end)
                .filter(|file| self.board[index(*file, rank)].is_some())
                .count()
        } else {
            usize::MAX
        }
    }

    #[cfg(test)]
    #[cfg(test)]
    pub(super) fn is_square_attacked_slow(&self, target: usize, by: Color) -> bool {
        for sq in 0..BOARD_SIZE {
            let Some(piece) = self.board[sq] else {
                continue;
            };
            if piece.color != by {
                continue;
            }
            if self.piece_attacks_square(sq, piece, target) {
                return true;
            }
        }
        false
    }
}
