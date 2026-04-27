use crate::xiangqi::{BOARD_FILES, BOARD_SIZE, Color, Move, Piece};

pub const HISTORY_PLIES: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HistoryMove {
    pub piece: Piece,
    pub captured: Option<Piece>,
    pub mv: Move,
}

pub fn mirror_file_square(sq: usize) -> usize {
    let rank = sq / BOARD_FILES;
    let file = sq % BOARD_FILES;
    rank * BOARD_FILES + (BOARD_FILES - 1 - file)
}

pub fn mirror_file_move(mv: Move) -> Move {
    Move::new(
        mirror_file_square(mv.from as usize),
        mirror_file_square(mv.to as usize),
    )
}

pub fn canonical_square(side: Color, sq: usize) -> usize {
    orient_square(side, sq)
}

pub fn canonical_move(side: Color, mv: Move) -> Move {
    Move::new(
        orient_square(side, mv.from as usize),
        orient_square(side, mv.to as usize),
    )
}

fn orient_square(side: Color, sq: usize) -> usize {
    match side {
        Color::Red => sq,
        Color::Black => BOARD_SIZE - 1 - sq,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirror_file_move_flips_left_and_right() {
        assert_eq!(mirror_file_move(Move::new(47, 38)), Move::new(51, 42));
    }

    #[test]
    fn canonical_black_rotates_board() {
        assert_eq!(canonical_square(Color::Black, 0), BOARD_SIZE - 1);
        assert_eq!(
            canonical_move(Color::Black, Move::new(0, 9)),
            Move::new(89, 80)
        );
    }
}
