use std::fmt;

use super::{BOARD_SIZE, parse_square, square_name};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Color {
    Red,
    Black,
}

impl Color {
    #[inline(always)]
    pub fn opposite(self) -> Self {
        match self {
            Self::Red => Self::Black,
            Self::Black => Self::Red,
        }
    }

    #[inline(always)]
    pub fn forward_step(self) -> i32 {
        match self {
            Self::Red => -1,
            Self::Black => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PieceKind {
    General,
    Advisor,
    Elephant,
    Horse,
    Rook,
    Cannon,
    Soldier,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MoveGenMode {
    All,
    Captures,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CheckerInfo {
    pub(super) from: usize,
    pub(super) kind: PieceKind,
    pub(super) screen_square: Option<usize>,
    pub(super) block_square: Option<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Piece {
    pub color: Color,
    pub kind: PieceKind,
}

impl Piece {
    pub(super) fn from_fen(ch: char) -> Option<Self> {
        let color = if ch.is_ascii_uppercase() {
            Color::Red
        } else {
            Color::Black
        };

        let kind = match ch.to_ascii_lowercase() {
            'k' => PieceKind::General,
            'a' => PieceKind::Advisor,
            'b' | 'e' => PieceKind::Elephant,
            'n' | 'h' => PieceKind::Horse,
            'r' => PieceKind::Rook,
            'c' => PieceKind::Cannon,
            'p' => PieceKind::Soldier,
            _ => return None,
        };

        Some(Self { color, kind })
    }

    pub(super) fn to_fen(self) -> char {
        let ch = match self.kind {
            PieceKind::General => 'k',
            PieceKind::Advisor => 'a',
            PieceKind::Elephant => 'b',
            PieceKind::Horse => 'n',
            PieceKind::Rook => 'r',
            PieceKind::Cannon => 'c',
            PieceKind::Soldier => 'p',
        };

        match self.color {
            Color::Red => ch.to_ascii_uppercase(),
            Color::Black => ch,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Move {
    pub from: u8,
    pub to: u8,
}

impl Move {
    #[inline(always)]
    pub fn new(from: usize, to: usize) -> Self {
        Self {
            from: from as u8,
            to: to as u8,
        }
    }

    pub fn from_uci(uci: &str) -> Option<Self> {
        if uci.len() != 4 {
            return None;
        }
        let from = parse_square(&uci[0..2])?;
        let to = parse_square(&uci[2..4])?;
        Some(Self::new(from, to))
    }

    pub fn to_uci(self) -> String {
        format!(
            "{}{}",
            square_name(self.from as usize),
            square_name(self.to as usize)
        )
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_uci())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Undo {
    pub(super) captured: Option<Piece>,
    pub(super) side_to_move: Color,
    pub(super) halfmove_clock: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuleHistoryEntry {
    pub hash: u64,
    pub side_to_move: Color,
    pub mover: Option<Color>,
    pub gives_check: bool,
    pub chased_mask: u128,
    pub chased_piece_mask: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuleDrawReason {
    Halfmove120,
    Repetition,
    MutualLongCheck,
    MutualLongChase,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuleOutcome {
    Draw(RuleDrawReason),
    Win(Color),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Position {
    pub(super) board: [Option<Piece>; BOARD_SIZE],
    pub(super) side_to_move: Color,
    pub(super) hash: u64,
    pub(super) base_eval: i32,
    pub(super) advisor_counts: [u8; 2],
    pub(super) elephant_counts: [u8; 2],
    pub(super) dynamic_material_counts: [u8; 2],
    pub(super) general_squares: [Option<usize>; 2],
    pub(super) halfmove_clock: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PositionState {
    pub(super) hash: u64,
    pub(super) base_eval: i32,
    pub(super) advisor_counts: [u8; 2],
    pub(super) elephant_counts: [u8; 2],
    pub(super) dynamic_material_counts: [u8; 2],
    pub(super) general_squares: [Option<usize>; 2],
}
