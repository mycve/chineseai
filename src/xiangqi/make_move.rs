use super::{
    Move, Piece, PieceKind, Position, RuleHistoryEntry, SIDE_TO_MOVE_KEY, Undo, color_hash_index,
    zobrist_piece_key,
};

impl Position {
    /// 从完整规则历史恢复初始局面，并重新验证每一步；用于向外部教师传递历史。
    /// 历史时钟是原始无吃子计数，不能替换为带将军豁免的有效自然限着计数。
    pub fn initial_position_from_history(
        &self,
        history: &[RuleHistoryEntry],
    ) -> Result<Self, String> {
        let first = history.first().ok_or("empty position history")?;
        let mut initial = self.clone();
        for index in (1..history.len()).rev() {
            let entry = history[index];
            let previous = history[index - 1];
            if entry.hash != initial.hash
                || entry.side_to_move != initial.side_to_move
                || entry.rule60_clock != initial.halfmove_clock
            {
                return Err(format!("history does not match position at ply {index}"));
            }
            let mv = entry.mv.ok_or("history move missing")?;
            let from = mv.from as usize;
            let to = mv.to as usize;
            if from >= initial.board.len()
                || to >= initial.board.len()
                || from == to
                || initial.board[from].is_some()
                || initial.board[to].is_none_or(|piece| piece.color != previous.side_to_move)
                || entry.mover != Some(previous.side_to_move)
                || entry.side_to_move == previous.side_to_move
                || entry
                    .captured
                    .is_some_and(|piece| piece.color == previous.side_to_move)
            {
                return Err(format!("invalid historical move at ply {index}"));
            }
            initial.unmake_move(
                mv,
                Undo {
                    captured: entry.captured,
                    side_to_move: previous.side_to_move,
                    halfmove_clock: previous.rule60_clock,
                },
            );
        }
        if initial.rule_history_entry(None) != *first {
            return Err("history initial position mismatch".into());
        }
        let mut replay = initial.clone();
        for index in 1..history.len() {
            let entry = history[index];
            let mv = entry.mv.ok_or("history move missing")?;
            if replay
                .rule_outcome_with_history(&history[..index])
                .is_some()
                || !replay
                    .legal_moves_with_rules(&history[..index])
                    .contains(&mv)
                || replay.rule_history_entry_after_move(mv) != entry
            {
                return Err(format!("illegal or inconsistent history at ply {index}"));
            }
            replay.make_move(mv);
        }
        if replay != *self {
            return Err("history replay does not reproduce position".into());
        }
        Ok(initial)
    }

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
