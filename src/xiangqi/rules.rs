use super::{
    Color, Move, PieceKind, Position, RuleDrawReason, RuleHistoryEntry, RuleOutcome,
    geom::soldier_crossed_river,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RuleViolation {
    LongCheck,
    LongChase,
}

impl Position {
    pub fn initial_rule_history(&self) -> Vec<RuleHistoryEntry> {
        vec![self.rule_history_entry(None)]
    }

    pub fn rule_history_entry(&self, mover: Option<Color>) -> RuleHistoryEntry {
        let (chased_mask, chased_piece_mask) =
            mover.map_or((0, 0), |color| self.chased_masks_by(color));
        RuleHistoryEntry {
            hash: self.hash,
            side_to_move: self.side_to_move,
            mover,
            gives_check: self.in_check(self.side_to_move),
            chased_mask,
            chased_piece_mask,
        }
    }

    pub fn rule_history_entry_after_move(&self, mv: Move) -> RuleHistoryEntry {
        let mut next = self.clone();
        let mover = self.side_to_move;
        let moved_piece_was_chased = self.square_is_chased_by(mv.from as usize, mover.opposite());
        next.make_move(mv);
        let (chased_mask, chased_piece_mask) = if moved_piece_was_chased {
            (0, 0)
        } else {
            next.chased_masks_by_origin(mover, mv.to as usize)
        };
        RuleHistoryEntry {
            hash: next.hash,
            side_to_move: next.side_to_move,
            mover: Some(mover),
            gives_check: next.in_check(next.side_to_move),
            chased_mask,
            chased_piece_mask,
        }
    }

    pub fn rule_outcome_with_history(&self, history: &[RuleHistoryEntry]) -> Option<RuleOutcome> {
        if self.halfmove_clock >= 120 {
            return Some(RuleOutcome::Draw(RuleDrawReason::Halfmove120));
        }
        Self::rule_outcome(history)
    }

    pub fn rule_outcome(history: &[RuleHistoryEntry]) -> Option<RuleOutcome> {
        let current_index = history.len().checked_sub(1)?;
        let current = history[current_index];
        let repeated_indices = history[..current_index]
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| {
                (entry.hash == current.hash && entry.side_to_move == current.side_to_move)
                    .then_some(index)
            })
            .collect::<Vec<_>>();
        if repeated_indices.is_empty() {
            return None;
        }

        if repeated_indices.len() < 3 {
            return None;
        }

        let cycle_start = repeated_indices[0] + 1;
        let red_violation =
            repeated_rule_violation(&history[cycle_start..=current_index], Color::Red);
        let black_violation =
            repeated_rule_violation(&history[cycle_start..=current_index], Color::Black);

        match (red_violation, black_violation) {
            (Some(RuleViolation::LongCheck), Some(RuleViolation::LongCheck)) => {
                return Some(RuleOutcome::Draw(RuleDrawReason::MutualLongCheck));
            }
            (Some(RuleViolation::LongChase), Some(RuleViolation::LongChase)) => {
                return Some(RuleOutcome::Draw(RuleDrawReason::MutualLongChase));
            }
            (Some(RuleViolation::LongCheck), _) => return Some(RuleOutcome::Win(Color::Black)),
            (_, Some(RuleViolation::LongCheck)) => return Some(RuleOutcome::Win(Color::Red)),
            (Some(RuleViolation::LongChase), _) => return Some(RuleOutcome::Win(Color::Black)),
            (_, Some(RuleViolation::LongChase)) => return Some(RuleOutcome::Win(Color::Red)),
            (None, None) => {}
        }

        (repeated_indices.len() >= 5).then_some(RuleOutcome::Draw(RuleDrawReason::Repetition))
    }

    pub fn legal_moves_with_rules(&self, history: &[RuleHistoryEntry]) -> Vec<Move> {
        crate::scope_profile!("xiangqi.legal_moves_with_rules");
        let legal = self.legal_moves();
        if legal.is_empty() {
            return legal;
        }

        let base_history = if history
            .last()
            .is_some_and(|entry| entry.hash == self.hash && entry.side_to_move == self.side_to_move)
        {
            history.to_vec()
        } else {
            let mut normalized = history.to_vec();
            normalized.push(self.rule_history_entry(None));
            normalized
        };

        let mover = self.side_to_move;
        legal
            .into_iter()
            .filter(|&mv| {
                let mut next = self.clone();
                next.make_move(mv);
                if !base_history
                    .iter()
                    .any(|entry| entry.hash == next.hash && entry.side_to_move == next.side_to_move)
                {
                    return true;
                }
                let mut next_history = base_history.clone();
                next_history.push(self.rule_history_entry_after_move(mv));
                !matches!(
                    next.rule_outcome_with_history(&next_history),
                    Some(RuleOutcome::Win(winner)) if winner == mover.opposite()
                )
            })
            .collect()
    }

    fn chased_masks_by(&self, color: Color) -> (u128, u16) {
        crate::scope_profile!("xiangqi.chased_mask_by");
        let mut work = self.clone();
        work.side_to_move = color;

        let mut square_mask = 0u128;
        let mut piece_mask = 0u16;
        for target in 0..super::BOARD_SIZE {
            let Some(target_piece) = self.board[target] else {
                continue;
            };
            if target_piece.color == color || target_piece.kind == PieceKind::General {
                continue;
            }
            if target_piece.kind == PieceKind::Soldier
                && !soldier_crossed_river(target_piece.color, super::geom::rank_of(target))
            {
                continue;
            }

            self.visit_attacker_origins_to(target, color, |from| {
                let mv = Move::new(from, target);
                let undo = work.make_move(mv);
                let legal = !work.in_check(color);
                work.unmake_move(mv, undo);
                if legal {
                    square_mask |= 1u128 << target;
                    piece_mask |= 1u16 << chased_piece_index(target_piece);
                    return true;
                }
                false
            });
        }
        (square_mask, piece_mask)
    }

    fn square_is_chased_by(&self, target: usize, color: Color) -> bool {
        let Some(target_piece) = self.board[target] else {
            return false;
        };
        if target_piece.color == color || target_piece.kind == PieceKind::General {
            return false;
        }
        if target_piece.kind == PieceKind::Soldier
            && !soldier_crossed_river(target_piece.color, super::geom::rank_of(target))
        {
            return false;
        }

        let mut work = self.clone();
        work.side_to_move = color;
        let mut chased = false;
        self.visit_attacker_origins_to(target, color, |from| {
            let mv = Move::new(from, target);
            let undo = work.make_move(mv);
            chased = !work.in_check(color);
            work.unmake_move(mv, undo);
            chased
        });
        chased
    }

    fn chased_masks_by_origin(&self, color: Color, origin: usize) -> (u128, u16) {
        crate::scope_profile!("xiangqi.chased_mask_by");
        let mut work = self.clone();
        work.side_to_move = color;

        let mut square_mask = 0u128;
        let mut piece_mask = 0u16;
        for target in 0..super::BOARD_SIZE {
            let Some(target_piece) = self.board[target] else {
                continue;
            };
            if target_piece.color == color || target_piece.kind == PieceKind::General {
                continue;
            }
            if target_piece.kind == PieceKind::Soldier
                && !soldier_crossed_river(target_piece.color, super::geom::rank_of(target))
            {
                continue;
            }

            self.visit_attacker_origins_to(target, color, |from| {
                if from != origin {
                    return false;
                }
                let mv = Move::new(from, target);
                let undo = work.make_move(mv);
                let legal = !work.in_check(color);
                work.unmake_move(mv, undo);
                if legal {
                    square_mask |= 1u128 << target;
                    piece_mask |= 1u16 << chased_piece_index(target_piece);
                }
                true
            });
        }
        (square_mask, piece_mask)
    }
}

fn repeated_rule_violation(entries: &[RuleHistoryEntry], color: Color) -> Option<RuleViolation> {
    let mover_entries = entries
        .iter()
        .filter(|entry| entry.mover == Some(color))
        .collect::<Vec<_>>();
    if mover_entries.is_empty() {
        return None;
    }

    if mover_entries.iter().all(|entry| entry.gives_check) {
        return Some(RuleViolation::LongCheck);
    }

    let chased_intersection = mover_entries
        .iter()
        .map(|entry| entry.chased_mask)
        .reduce(|a, b| a & b)
        .unwrap_or(0);
    let chased_piece_intersection = mover_entries
        .iter()
        .map(|entry| entry.chased_piece_mask)
        .reduce(|a, b| a & b)
        .unwrap_or(0);
    (chased_intersection != 0 || chased_piece_intersection != 0).then_some(RuleViolation::LongChase)
}

fn chased_piece_index(piece: super::Piece) -> usize {
    let color_offset = match piece.color {
        Color::Red => 0,
        Color::Black => 7,
    };
    let kind_offset = match piece.kind {
        PieceKind::General => 0,
        PieceKind::Advisor => 1,
        PieceKind::Elephant => 2,
        PieceKind::Horse => 3,
        PieceKind::Rook => 4,
        PieceKind::Cannon => 5,
        PieceKind::Soldier => 6,
    };
    color_offset + kind_offset
}
