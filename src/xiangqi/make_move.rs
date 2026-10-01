use super::{
    Move, Piece, PieceKind, Position, SIDE_TO_MOVE_KEY, Undo, color_hash_index, zobrist_piece_key,
};

impl Position {
    pub fn make_move(&mut self, mv: Move) -> Undo {
        crate::scope_profile!("xiangqi.make_move");
        let from = mv.from as usize;
        let to = mv.to as usize;
        let moving = self.board[from].expect("move from occupied square");
        let undo = Undo {
            captured: self.board[to],
            side_to_move: self.side_to_move,
            halfmove_clock: self.halfmove_clock,
        };

        self.hash ^= zobrist_piece_key(from, moving);
        if let Some(captured) = undo.captured {
            self.hash ^= zobrist_piece_key(to, captured);
            self.adjust_minor_counts(captured, -1);
            self.adjust_dynamic_material_counts(captured, -1);
            if captured.kind == PieceKind::General {
                self.general_squares[color_hash_index(captured.color)] = None;
            }
        }

        self.board[to] = Some(moving);
        self.board[from] = None;
        self.occupied = (self.occupied & !(1u128 << from)) | (1u128 << to);
        if moving.kind == PieceKind::General {
            self.general_squares[color_hash_index(moving.color)] = Some(to);
        }
        self.hash ^= zobrist_piece_key(to, moving);
        self.side_to_move = self.side_to_move.opposite();
        self.hash ^= SIDE_TO_MOVE_KEY;
        self.halfmove_clock = if undo.captured.is_some() {
            0
        } else {
            self.halfmove_clock.saturating_add(1)
        };
        undo
    }

    pub fn unmake_move(&mut self, mv: Move, undo: Undo) {
        let from = mv.from as usize;
        let to = mv.to as usize;
        let moving = self.board[to].expect("move to occupied square");
        self.hash ^= SIDE_TO_MOVE_KEY;
        self.hash ^= zobrist_piece_key(to, moving);
        self.board[from] = Some(moving);
        self.board[to] = undo.captured;
        self.occupied |= 1u128 << from;
        if undo.captured.is_none() {
            self.occupied &= !(1u128 << to);
        }
        if moving.kind == PieceKind::General {
            self.general_squares[color_hash_index(moving.color)] = Some(from);
        }
        self.hash ^= zobrist_piece_key(from, moving);
        if let Some(captured) = undo.captured {
            self.hash ^= zobrist_piece_key(to, captured);
            self.adjust_minor_counts(captured, 1);
            self.adjust_dynamic_material_counts(captured, 1);
            if captured.kind == PieceKind::General {
                self.general_squares[color_hash_index(captured.color)] = Some(to);
            }
        }
        self.side_to_move = undo.side_to_move;
        self.halfmove_clock = undo.halfmove_clock;
    }

    pub(super) fn make_move_board_only(&mut self, mv: Move) -> Option<Piece> {
        let from = mv.from as usize;
        let to = mv.to as usize;
        let moving = self.board[from].expect("move from occupied square");
        let captured = self.board[to];
        self.board[to] = Some(moving);
        self.board[from] = None;
        self.occupied = (self.occupied & !(1u128 << from)) | (1u128 << to);
        if moving.kind == PieceKind::General {
            self.general_squares[color_hash_index(moving.color)] = Some(to);
        }
        captured
    }

    pub(super) fn unmake_move_board_only(&mut self, mv: Move, captured: Option<Piece>) {
        let from = mv.from as usize;
        let to = mv.to as usize;
        let moving = self.board[to].expect("move to occupied square");
        self.board[from] = Some(moving);
        self.board[to] = captured;
        self.occupied |= 1u128 << from;
        if captured.is_none() {
            self.occupied &= !(1u128 << to);
        }
        if moving.kind == PieceKind::General {
            self.general_squares[color_hash_index(moving.color)] = Some(from);
        }
        if let Some(captured) = captured
            && captured.kind == PieceKind::General
        {
            self.general_squares[color_hash_index(captured.color)] = Some(to);
        }
    }
}
