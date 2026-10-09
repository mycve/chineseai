use super::{
    BOARD_FILES, BOARD_RANKS, BOARD_SIZE, Color, Piece, PieceKind, Position, PositionState,
    RuleHistoryEntry, SIDE_TO_MOVE_KEY, STARTPOS_FEN, color_hash_index, file_of, index, rank_of,
    zobrist_piece_key,
};

impl Default for Position {
    fn default() -> Self {
        Self::startpos()
    }
}

impl Position {
    pub(crate) fn from_canonical_piece_squares(pieces: &[(usize, usize)]) -> Self {
        let kinds = [
            PieceKind::General,
            PieceKind::Advisor,
            PieceKind::Elephant,
            PieceKind::Horse,
            PieceKind::Rook,
            PieceKind::Cannon,
            PieceKind::Soldier,
        ];
        let mut board = [None; BOARD_SIZE];
        for &(piece_index, square) in pieces {
            if piece_index < 14 && square < BOARD_SIZE {
                board[square] = Some(Piece {
                    color: if piece_index < 7 {
                        Color::Red
                    } else {
                        Color::Black
                    },
                    kind: kinds[piece_index % 7],
                });
            }
        }
        let position = Self {
            board,
            occupied: 0,
            side_to_move: Color::Red,
            hash: 0,
            advisor_counts: [0; 2],
            elephant_counts: [0; 2],
            dynamic_material_counts: [0; 2],
            general_squares: [None; 2],
            halfmove_clock: 0,
            rule60_max_ply: Some(120),
            repetition_draw_enabled: true,
        };
        let state = position.compute_state();
        Self {
            hash: state.hash,
            occupied: state.occupied,
            advisor_counts: state.advisor_counts,
            elephant_counts: state.elephant_counts,
            dynamic_material_counts: state.dynamic_material_counts,
            general_squares: state.general_squares,
            ..position
        }
    }

    pub fn startpos() -> Self {
        Self::from_fen(STARTPOS_FEN).expect("valid start position")
    }

    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let mut parts = fen.split_whitespace();
        let board_part = parts.next().ok_or("missing board description")?;
        let side_part = parts.next().unwrap_or("w");
        let remaining = parts.collect::<Vec<_>>();
        let halfmove_clock = match remaining.as_slice() {
            ["-", "-", value, ..] => value
                .parse::<u16>()
                .map_err(|_| format!("invalid halfmove clock: {value}"))?,
            [value, ..] => value.parse::<u16>().unwrap_or(0),
            [] => 0,
        };

        let mut board = [None; BOARD_SIZE];
        let ranks: Vec<&str> = board_part.split('/').collect();
        if ranks.len() != BOARD_RANKS {
            return Err(format!("expected {BOARD_RANKS} ranks in FEN"));
        }

        for (rank, rank_data) in ranks.iter().enumerate() {
            let mut file = 0usize;
            for ch in rank_data.chars() {
                if let Some(empty) = ch.to_digit(10) {
                    file += empty as usize;
                    continue;
                }

                let piece = Piece::from_fen(ch).ok_or_else(|| format!("invalid piece: {ch}"))?;
                if file >= BOARD_FILES {
                    return Err("file overflow in FEN".into());
                }
                board[index(file, rank)] = Some(piece);
                file += 1;
            }

            if file != BOARD_FILES {
                return Err(format!("rank {rank} does not contain {BOARD_FILES} files"));
            }
        }

        let side_to_move = match side_part {
            "w" | "r" => Color::Red,
            "b" => Color::Black,
            other => return Err(format!("invalid side to move: {other}")),
        };

        let position = Self {
            board,
            occupied: 0,
            side_to_move,
            hash: 0,
            advisor_counts: [0; 2],
            elephant_counts: [0; 2],
            dynamic_material_counts: [0; 2],
            general_squares: [None; 2],
            halfmove_clock,
            rule60_max_ply: Some(120),
            repetition_draw_enabled: true,
        };
        let state = position.compute_state();
        let position = Self {
            hash: state.hash,
            occupied: state.occupied,
            advisor_counts: state.advisor_counts,
            elephant_counts: state.elephant_counts,
            dynamic_material_counts: state.dynamic_material_counts,
            general_squares: state.general_squares,
            ..position
        };
        position.validate()?;
        Ok(position)
    }

    pub fn to_fen(&self) -> String {
        self.to_fen_with_rule60_clock(self.halfmove_clock)
    }

    pub fn to_fen_with_history(&self, history: &[RuleHistoryEntry]) -> String {
        self.to_fen_with_rule60_clock(self.rule60_count_with_history(history))
    }

    fn to_fen_with_rule60_clock(&self, rule60_clock: u16) -> String {
        let mut board_part = String::new();
        for rank in 0..BOARD_RANKS {
            if rank > 0 {
                board_part.push('/');
            }

            let mut empty = 0usize;
            for file in 0..BOARD_FILES {
                match self.board[index(file, rank)] {
                    Some(piece) => {
                        if empty > 0 {
                            board_part.push(char::from_digit(empty as u32, 10).unwrap());
                            empty = 0;
                        }
                        board_part.push(piece.to_fen());
                    }
                    None => empty += 1,
                }
            }
            if empty > 0 {
                board_part.push(char::from_digit(empty as u32, 10).unwrap());
            }
        }

        let side = match self.side_to_move {
            Color::Red => "w",
            Color::Black => "b",
        };

        format!("{board_part} {side} - - {rule60_clock} 1")
    }

    pub fn set_rule60_max_ply(&mut self, max_ply: Option<u16>) {
        self.rule60_max_ply = max_ply.map(|value| value.max(1));
    }

    pub fn mirror_files(&self) -> Self {
        let mut board = [None; BOARD_SIZE];
        for sq in 0..BOARD_SIZE {
            let rank = rank_of(sq);
            let file = file_of(sq);
            let mirrored = index(BOARD_FILES - 1 - file, rank);
            board[mirrored] = self.board[sq];
        }

        let position = Self {
            board,
            occupied: 0,
            side_to_move: self.side_to_move,
            hash: 0,
            advisor_counts: [0; 2],
            elephant_counts: [0; 2],
            dynamic_material_counts: [0; 2],
            general_squares: [None; 2],
            halfmove_clock: self.halfmove_clock,
            rule60_max_ply: self.rule60_max_ply,
            repetition_draw_enabled: self.repetition_draw_enabled,
        };
        let state = position.compute_state();
        Self {
            hash: state.hash,
            occupied: state.occupied,
            advisor_counts: state.advisor_counts,
            elephant_counts: state.elephant_counts,
            dynamic_material_counts: state.dynamic_material_counts,
            general_squares: state.general_squares,
            ..position
        }
    }

    fn validate(&self) -> Result<(), String> {
        let red_general = self.find_general(Color::Red);
        let black_general = self.find_general(Color::Black);
        if red_general.is_none() || black_general.is_none() {
            return Err("both generals must be present".into());
        }

        if self.generals_face() {
            return Err("illegal position: generals are facing".into());
        }

        Ok(())
    }

    fn compute_state(&self) -> PositionState {
        let mut hash = 0u64;
        let mut occupied = 0u128;
        let mut advisor_counts = [0u8; 2];
        let mut elephant_counts = [0u8; 2];
        let mut dynamic_material_counts = [0u8; 2];
        let mut general_squares = [None; 2];
        for sq in 0..BOARD_SIZE {
            let Some(piece) = self.board[sq] else {
                continue;
            };
            hash ^= zobrist_piece_key(sq, piece);
            occupied |= 1u128 << sq;
            match piece.kind {
                PieceKind::Advisor => advisor_counts[color_hash_index(piece.color)] += 1,
                PieceKind::Elephant => elephant_counts[color_hash_index(piece.color)] += 1,
                PieceKind::Rook | PieceKind::Cannon | PieceKind::Horse | PieceKind::Soldier => {
                    dynamic_material_counts[color_hash_index(piece.color)] += 1
                }
                PieceKind::General => general_squares[color_hash_index(piece.color)] = Some(sq),
            }
        }

        if self.side_to_move == Color::Red {
            hash ^= SIDE_TO_MOVE_KEY;
        }

        PositionState {
            hash,
            occupied,
            advisor_counts,
            elephant_counts,
            dynamic_material_counts,
            general_squares,
        }
    }

    #[cfg(test)]
    pub(super) fn compute_hash(&self) -> u64 {
        self.compute_state().hash
    }

    pub(super) fn adjust_minor_counts(&mut self, piece: Piece, delta: i8) {
        let index = color_hash_index(piece.color);
        match piece.kind {
            PieceKind::Advisor => {
                self.advisor_counts[index] =
                    (self.advisor_counts[index] as i16 + delta as i16).max(0) as u8;
            }
            PieceKind::Elephant => {
                self.elephant_counts[index] =
                    (self.elephant_counts[index] as i16 + delta as i16).max(0) as u8;
            }
            _ => {}
        }
    }

    pub(super) fn adjust_dynamic_material_counts(&mut self, piece: Piece, delta: i8) {
        let index = color_hash_index(piece.color);
        match piece.kind {
            PieceKind::Rook | PieceKind::Cannon | PieceKind::Horse | PieceKind::Soldier => {
                self.dynamic_material_counts[index] =
                    (self.dynamic_material_counts[index] as i16 + delta as i16).max(0) as u8;
            }
            _ => {}
        }
    }
}
