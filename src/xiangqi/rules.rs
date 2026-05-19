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
        RuleHistoryEntry {
            hash: self.hash,
            side_to_move: self.side_to_move,
            mover,
            gives_check: self.in_check(self.side_to_move),
            chased_mask: mover.map_or(0, |color| self.chased_mask_by(color)),
        }
    }

    pub fn rule_history_entry_after_move(&self, mv: Move) -> RuleHistoryEntry {
        let mut next = self.clone();
        let mover = self.side_to_move;
        next.make_move(mv);
        next.rule_history_entry(Some(mover))
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

        if repeated_indices.len() < 5 {
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
                let mut next_history = base_history.clone();
                next_history.push(next.rule_history_entry(Some(mover)));
                !matches!(
                    next.rule_outcome_with_history(&next_history),
                    Some(RuleOutcome::Win(winner)) if winner == mover.opposite()
                )
            })
            .collect()
    }

    fn chased_mask_by(&self, color: Color) -> u128 {
        let mut work = self.clone();
        work.side_to_move = color;
        work.legal_moves()
            .into_iter()
            .filter_map(|mv| {
                let target = mv.to as usize;
                let target_piece = self.board[target]?;
                if target_piece.color == color || target_piece.kind == PieceKind::General {
                    return None;
                }
                if target_piece.kind == PieceKind::Soldier
                    && !soldier_crossed_river(target_piece.color, super::geom::rank_of(target))
                {
                    return None;
                }
                Some(1u128 << target)
            })
            .fold(0u128, |mask, bit| mask | bit)
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
    (chased_intersection != 0).then_some(RuleViolation::LongChase)
}
