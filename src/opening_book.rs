use std::{collections::HashMap, path::Path};

use rusqlite::{Connection, types::ValueRef};

use crate::{
    az::SplitMix64,
    obk_zobrist::{OBK_ZOBRIST_PLAYER, OBK_ZOBRIST_TABLE},
    xiangqi::{BOARD_FILES, BOARD_RANKS, Color, Move, Piece, PieceKind, Position},
};

#[derive(Clone, Copy, Debug)]
struct ObkMove {
    vmove: u16,
    score: i32,
}

#[derive(Clone, Debug, Default)]
pub struct ObkBook {
    moves_by_key: HashMap<u64, Vec<ObkMove>>,
    move_count: usize,
}

impl ObkBook {
    pub fn load(path: impl AsRef<Path>) -> rusqlite::Result<Self> {
        let connection = Connection::open(path)?;
        let mut statement =
            connection.prepare("select vkey, vmove, vscore from bhobk where vvalid = 1")?;
        let mut rows = statement.query([])?;
        let mut moves_by_key: HashMap<u64, Vec<ObkMove>> = HashMap::new();
        let mut move_count = 0usize;
        while let Some(row) = rows.next()? {
            let key = match row.get_ref(0)? {
                ValueRef::Integer(value) => value as u64,
                ValueRef::Real(value) => value.to_bits(),
                _ => continue,
            };
            let vmove = row.get::<_, i64>(1)? as u16;
            let score = row.get::<_, i64>(2).unwrap_or(1) as i32;
            moves_by_key
                .entry(key)
                .or_default()
                .push(ObkMove { vmove, score });
            move_count += 1;
        }

        Ok(Self {
            moves_by_key,
            move_count,
        })
    }

    pub fn key_count(&self) -> usize {
        self.moves_by_key.len()
    }

    pub fn move_count(&self) -> usize {
        self.move_count
    }

    pub fn random_prefix_position(
        &self,
        min_plies: usize,
        max_plies: usize,
        rng: &mut SplitMix64,
    ) -> Position {
        let mut position = Position::startpos();
        let span = max_plies.saturating_sub(min_plies);
        let target_plies = min_plies + (rng.next_u64() as usize % (span + 1));
        for _ in 0..target_plies {
            let legal = position.legal_moves();
            if legal.is_empty() {
                break;
            }
            let book_moves = self.legal_book_moves(&position, &legal);
            if book_moves.is_empty() {
                break;
            }
            let mv = choose_book_move(&book_moves, rng);
            position.make_move(mv);
        }
        position
    }

    fn legal_book_moves(&self, position: &Position, legal: &[Move]) -> Vec<(Move, i32)> {
        let mut out = Vec::new();
        self.push_legal_book_moves(position, legal, false, &mut out);
        self.push_legal_book_moves(position, legal, true, &mut out);
        out.sort_by_key(|(mv, score)| (mv.from, mv.to, *score));
        out.dedup_by_key(|(mv, _)| (mv.from, mv.to));
        out
    }

    fn push_legal_book_moves(
        &self,
        position: &Position,
        legal: &[Move],
        left_right_swap: bool,
        out: &mut Vec<(Move, i32)>,
    ) {
        let key = obk_zobrist(position, left_right_swap);
        let Some(book_moves) = self.moves_by_key.get(&key) else {
            return;
        };
        for book_move in book_moves {
            if let Some(mv) = decode_obk_move(book_move.vmove, left_right_swap) {
                if legal.contains(&mv) {
                    out.push((mv, book_move.score));
                }
            }
        }
    }
}

fn choose_book_move(moves: &[(Move, i32)], rng: &mut SplitMix64) -> Move {
    let min_score = moves.iter().map(|(_, score)| *score).min().unwrap_or(0);
    let total = moves
        .iter()
        .map(|(_, score)| (*score - min_score + 1).max(1) as u64)
        .sum::<u64>()
        .max(1);
    let mut pick = rng.next_u64() % total;
    for (mv, score) in moves {
        let weight = (*score - min_score + 1).max(1) as u64;
        if pick < weight {
            return *mv;
        }
        pick -= weight;
    }
    moves[0].0
}

fn decode_obk_move(vmove: u16, left_right_swap: bool) -> Option<Move> {
    let from = decode_obk_square((vmove >> 8) as u8, left_right_swap)?;
    let to = decode_obk_square((vmove & 0xff) as u8, left_right_swap)?;
    Some(Move::new(from, to))
}

fn decode_obk_square(value: u8, left_right_swap: bool) -> Option<usize> {
    let rank = ((value >> 4) as i32) - 3;
    let mut file = ((value & 0x0f) as i32) - 3;
    if left_right_swap {
        file = 8 - file;
    }
    if !(0..BOARD_FILES as i32).contains(&file) || !(0..BOARD_RANKS as i32).contains(&rank) {
        return None;
    }
    Some(rank as usize * BOARD_FILES + file as usize)
}

fn obk_zobrist(position: &Position, left_right_swap: bool) -> u64 {
    let mut key = 0u64;
    for rank in 0..BOARD_RANKS {
        for file in 0..BOARD_FILES {
            let sq = rank * BOARD_FILES + file;
            let Some(piece) = position.piece_at(sq) else {
                continue;
            };
            let obk_file = if left_right_swap {
                BOARD_FILES - 1 - file
            } else {
                file
            };
            let obk_square = ((rank + 3) << 4) + obk_file + 3;
            key ^= OBK_ZOBRIST_TABLE[obk_piece_index(piece) * 256 + obk_square];
        }
    }
    if position.side_to_move() == Color::Red {
        key ^= OBK_ZOBRIST_PLAYER;
    }
    key
}

fn obk_piece_index(piece: Piece) -> usize {
    let base = match piece.color {
        Color::Red => 0,
        Color::Black => 7,
    };
    base + match piece.kind {
        PieceKind::General => 0,
        PieceKind::Advisor => 1,
        PieceKind::Elephant => 2,
        PieceKind::Horse => 3,
        PieceKind::Rook => 4,
        PieceKind::Cannon => 5,
        PieceKind::Soldier => 6,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_local_obk_when_present() {
        let path = Path::new("opening.obk");
        if !path.exists() {
            return;
        }
        let book = ObkBook::load(path).unwrap();
        assert!(book.key_count() > 0);
        assert!(book.move_count() > 0);
        let mut rng = SplitMix64::new(1);
        let position = book.random_prefix_position(4, 8, &mut rng);
        assert_ne!(position.to_fen(), Position::startpos().to_fen());
    }
}
