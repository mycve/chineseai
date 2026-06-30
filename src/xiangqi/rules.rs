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
        crate::scope_profile!("xiangqi.rule_history_entry");
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
        crate::scope_profile!("xiangqi.rule_history_after_move");
        let mut next = {
            crate::scope_profile!("xiangqi.rule_history.clone_position");
            self.clone()
        };
        let mover = self.side_to_move;
        {
            crate::scope_profile!("xiangqi.rule_history.make_move");
            next.make_move(mv);
        }
        next.rule_history_entry_after_moved(mover, mv.to as usize)
    }

    pub fn rule_history_entry_after_moved(&self, mover: Color, origin: usize) -> RuleHistoryEntry {
        crate::scope_profile!("xiangqi.rule_history_after_moved");
        let (chased_mask, chased_piece_mask) = {
            crate::scope_profile!("xiangqi.rule_history.chased_origin");
            self.chased_masks_by_origin(mover, origin)
        };
        let gives_check = {
            crate::scope_profile!("xiangqi.rule_history.gives_check");
            self.in_check(self.side_to_move)
        };
        RuleHistoryEntry {
            hash: self.hash,
            side_to_move: self.side_to_move,
            mover: Some(mover),
            gives_check,
            chased_mask,
            chased_piece_mask,
        }
    }

    pub fn rule_outcome_with_history(&self, history: &[RuleHistoryEntry]) -> Option<RuleOutcome> {
        crate::scope_profile!("xiangqi.rule_outcome_with_history");
        if self.halfmove_clock >= 120 {
            return Some(RuleOutcome::Draw(RuleDrawReason::Halfmove120));
        }
        Self::rule_outcome(history)
    }

    pub fn rule_outcome(history: &[RuleHistoryEntry]) -> Option<RuleOutcome> {
        crate::scope_profile!("xiangqi.rule_outcome");
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

        let cycle_start = repeated_indices[0] + 1;
        let red_violation =
            repeated_rule_violation(&history[cycle_start..=current_index], Color::Red);
        let black_violation =
            repeated_rule_violation(&history[cycle_start..=current_index], Color::Black);

        // One matching position only establishes the first complete cycle.
        // Stop long check as soon as the next cycle closes; ordinary
        // repetition and long chase keep their existing limit.
        if repeated_indices.len() < 2 {
            return None;
        }
        if repeated_indices.len() == 2 {
            return match (red_violation, black_violation) {
                (Some(RuleViolation::LongCheck), Some(RuleViolation::LongCheck)) => {
                    Some(RuleOutcome::Draw(RuleDrawReason::MutualLongCheck))
                }
                (Some(RuleViolation::LongCheck), _) => Some(RuleOutcome::Win(Color::Black)),
                (_, Some(RuleViolation::LongCheck)) => Some(RuleOutcome::Win(Color::Red)),
                _ => None,
            };
        }

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

        Some(RuleOutcome::Draw(RuleDrawReason::Repetition))
    }

    pub fn legal_moves_with_rules(&self, history: &[RuleHistoryEntry]) -> Vec<Move> {
        crate::scope_profile!("xiangqi.legal_moves_with_rules");
        let legal = self.legal_moves();
        if legal.is_empty() {
            return legal;
        }

        let current_entry = (!history.last().is_some_and(|entry| {
            entry.hash == self.hash && entry.side_to_move == self.side_to_move
        }))
        .then(|| self.rule_history_entry(None));

        let mover = self.side_to_move;
        legal
            .into_iter()
            .filter(|&mv| {
                let next_hash = self.hash_after_move(mv);
                let next_side_to_move = mover.opposite();
                if !history
                    .iter()
                    .any(|entry| entry.hash == next_hash && entry.side_to_move == next_side_to_move)
                {
                    return true;
                }
                let mut next = self.clone();
                next.make_move(mv);
                let mut next_history =
                    Vec::with_capacity(history.len() + usize::from(current_entry.is_some()) + 1);
                next_history.extend_from_slice(history);
                if let Some(entry) = current_entry {
                    next_history.push(entry);
                }
                next_history.push(self.rule_history_entry_after_move(mv));
                !rule_outcome_forbidden_for_mover(
                    next.rule_outcome_with_history(&next_history),
                    mover,
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
            if !self.is_chase_target_piece(target_piece, color, target) {
                continue;
            }

            self.visit_attacker_origins_to(target, color, |from| {
                if !self.is_effective_chase(target_piece, target, from) {
                    return false;
                }
                let mv = Move::new(from, target);
                let captured = work.make_move_board_only(mv);
                let legal = !work.in_check(color);
                work.unmake_move_board_only(mv, captured);
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
            if !self.is_chase_target_piece(target_piece, color, target) {
                continue;
            }

            self.visit_attacker_origins_to(target, color, |from| {
                if from != origin {
                    return false;
                }
                if !self.is_effective_chase(target_piece, target, from) {
                    return true;
                }
                let mv = Move::new(from, target);
                let captured = work.make_move_board_only(mv);
                let legal = !work.in_check(color);
                work.unmake_move_board_only(mv, captured);
                if legal {
                    square_mask |= 1u128 << target;
                    piece_mask |= 1u16 << chased_piece_index(target_piece);
                }
                true
            });
        }
        (square_mask, piece_mask)
    }

    fn is_chase_target_piece(&self, piece: super::Piece, attacker: Color, sq: usize) -> bool {
        if piece.color == attacker {
            return false;
        }
        match piece.kind {
            PieceKind::General | PieceKind::Advisor | PieceKind::Elephant => false,
            PieceKind::Soldier => soldier_crossed_river(piece.color, super::geom::rank_of(sq)),
            PieceKind::Horse | PieceKind::Rook | PieceKind::Cannon => true,
        }
    }

    fn is_effective_chase(&self, target: super::Piece, target_sq: usize, from: usize) -> bool {
        if !self.is_piece_protected(target_sq, target.color) {
            return true;
        }
        matches!(
            (self.board[from].map(|piece| piece.kind), target.kind),
            (Some(PieceKind::Horse), PieceKind::Rook)
        )
    }
}

fn rule_outcome_forbidden_for_mover(outcome: Option<RuleOutcome>, mover: Color) -> bool {
    matches!(
        outcome,
        Some(RuleOutcome::Win(winner)) if winner == mover.opposite()
    ) || matches!(
        outcome,
        Some(RuleOutcome::Draw(
            RuleDrawReason::MutualLongCheck | RuleDrawReason::MutualLongChase
        ))
    )
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
