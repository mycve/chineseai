pub const BOARD_FILES: usize = 9;
pub const BOARD_RANKS: usize = 10;
pub const BOARD_SIZE: usize = BOARD_FILES * BOARD_RANKS;
pub const STARTPOS_FEN: &str = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w";

mod access;
#[cfg(test)]
mod attack_test_helpers;
mod attacks;
mod check;
mod env;
mod r#gen;
mod geom;
mod hash;
mod legality;
mod make_move;
mod masks;
#[cfg(test)]
mod relations;
mod rules;
mod setup;
mod types;

pub use env::{AppliedMove, IllegalMove, StepOutcome, XiangqiEnv};
pub use geom::{parse_square, square_name};
pub use types::{
    Color, Move, Piece, PieceKind, Position, RuleDrawReason, RuleHistoryEntry, RuleOutcome, Undo,
};
pub(crate) use types::{color_index, piece_kind_index};

use geom::{
    elephant_stays_home, file_of, horse_leg_square, index, inside_board, inside_palace,
    line_between_squares, rank_of, soldier_crossed_river,
};
use hash::{SIDE_TO_MOVE_KEY, color_hash_index, zobrist_piece_key};
#[cfg(test)]
pub(crate) use masks::nearest_on_ray;
use masks::{DIAGONAL_STEPS, ELEPHANT_STEPS, HORSE_STEPS, ORTHOGONAL_STEPS, offset_square};
#[cfg(test)]
use masks::{fixed_attack_masks, orthogonal_ray_masks, ray_through};
#[cfg(test)]
use std::sync::OnceLock;
use types::{CheckerInfo, MoveGenMode, PositionState};

#[cfg(test)]
mod tests;
