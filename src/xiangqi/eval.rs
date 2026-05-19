use super::{BOARD_FILES, BOARD_RANKS, BOARD_SIZE, Color, Piece, PieceKind};

#[inline(always)]
pub(super) fn signed_piece_contrib(piece: Piece, sq: usize) -> i32 {
    SIGNED_PIECE_CONTRIB_TABLE[color_index(piece.color)][piece_kind_index(piece.kind)][sq]
}

#[inline(always)]
pub(crate) fn piece_base_value(kind: PieceKind) -> i32 {
    PIECE_BASE_VALUES[piece_kind_index(kind)]
}

const PIECE_BASE_VALUES: [i32; 7] = [0, 110, 110, 420, 900, 460, 110];
const PIECE_SQUARE_TABLE: [[[i32; BOARD_SIZE]; 7]; 2] = build_piece_square_table();
const SIGNED_PIECE_CONTRIB_TABLE: [[[i32; BOARD_SIZE]; 7]; 2] = build_signed_piece_contrib_table();

const fn build_piece_square_table() -> [[[i32; BOARD_SIZE]; 7]; 2] {
    let mut table = [[[0; BOARD_SIZE]; 7]; 2];
    let mut color = 0usize;
    while color < 2 {
        let mut kind = 0usize;
        while kind < 7 {
            let mut sq = 0usize;
            while sq < BOARD_SIZE {
                table[color][kind][sq] = piece_square_formula(color, kind, sq);
                sq += 1;
            }
            kind += 1;
        }
        color += 1;
    }
    table
}

const fn build_signed_piece_contrib_table() -> [[[i32; BOARD_SIZE]; 7]; 2] {
    let mut table = [[[0; BOARD_SIZE]; 7]; 2];
    let mut color = 0usize;
    while color < 2 {
        let mut kind = 0usize;
        while kind < 7 {
            let mut sq = 0usize;
            while sq < BOARD_SIZE {
                let value = PIECE_BASE_VALUES[kind] + PIECE_SQUARE_TABLE[color][kind][sq];
                table[color][kind][sq] = if color == 0 { value } else { -value };
                sq += 1;
            }
            kind += 1;
        }
        color += 1;
    }
    table
}

const fn piece_square_formula(color: usize, kind: usize, sq: usize) -> i32 {
    let file = (sq % BOARD_FILES) as i32;
    let rank = (sq / BOARD_FILES) as i32;
    let relative_rank = if color == 0 {
        (BOARD_RANKS as i32 - 1) - rank
    } else {
        rank
    };
    let center_distance = abs_i32(file - 4);

    match kind {
        0 => -(center_distance * 3),
        1 => 8 - center_distance * 4,
        2 => 10 - center_distance * 2 - abs_i32(relative_rank - 1) * 2,
        3 => 40 - center_distance * 8 - abs_i32(relative_rank - 4) * 4,
        4 => 18 - center_distance * 3 + relative_rank * 2,
        5 => 26 - center_distance * 5 + relative_rank * 3,
        6 => {
            let crossed = if relative_rank >= 5 { 1 } else { 0 };
            6 + relative_rank * 16 + crossed * 40 - center_distance * 4
        }
        _ => 0,
    }
}

const fn abs_i32(value: i32) -> i32 {
    if value < 0 { -value } else { value }
}

#[inline(always)]
const fn piece_kind_index(kind: PieceKind) -> usize {
    match kind {
        PieceKind::General => 0,
        PieceKind::Advisor => 1,
        PieceKind::Elephant => 2,
        PieceKind::Horse => 3,
        PieceKind::Rook => 4,
        PieceKind::Cannon => 5,
        PieceKind::Soldier => 6,
    }
}

#[inline(always)]
const fn color_index(color: Color) -> usize {
    match color {
        Color::Red => 0,
        Color::Black => 1,
    }
}
