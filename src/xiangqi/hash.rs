use super::{Color, Piece, PieceKind};

pub(super) const SIDE_TO_MOVE_KEY: u64 = 0x9e37_79b9_7f4a_7c15;

#[inline(always)]
pub(super) fn zobrist_piece_key(sq: usize, piece: Piece) -> u64 {
    splitmix64(((sq as u64) << 5) ^ piece_hash_index(piece) as u64)
}

#[inline(always)]
fn piece_hash_index(piece: Piece) -> usize {
    color_hash_index(piece.color) * 7
        + match piece.kind {
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
pub(super) fn color_hash_index(color: Color) -> usize {
    match color {
        Color::Red => 0,
        Color::Black => 1,
    }
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}
