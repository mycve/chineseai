use super::{
    Color, Move, MoveGenMode, PieceKind, Position, RuleDrawReason, RuleHistoryEntry, RuleOutcome,
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
        let chased_mask = mover.map_or(0, |color| self.chased_masks_by(color));
        RuleHistoryEntry {
            hash: self.hash,
            side_to_move: self.side_to_move,
            mover,
            gives_check: self.in_check(self.side_to_move),
            chased_mask,
            mv: None,
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
        next.rule_history_entry_after_moved(mover, mv)
    }

    pub fn rule_history_entry_after_moved(&self, mover: Color, mv: Move) -> RuleHistoryEntry {
        crate::scope_profile!("xiangqi.rule_history_after_moved");
        let chased_mask = {
            crate::scope_profile!("xiangqi.rule_history.chased_origin");
            self.chased_masks_by_origin(mover, mv.to as usize)
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
            mv: Some(mv),
        }
    }

    pub fn rule_outcome_with_history(&self, history: &[RuleHistoryEntry]) -> Option<RuleOutcome> {
        crate::scope_profile!("xiangqi.rule_outcome_with_history");
        if let Some(outcome) = Self::rule_outcome(history) {
            return Some(outcome);
        }
        if self
            .rule60_max_ply
            .is_some_and(|max_ply| self.halfmove_clock >= max_ply)
        {
            return Some(RuleOutcome::Draw(RuleDrawReason::NaturalMoveLimit));
        }
        self.insufficient_material_outcome()
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

        // 长将和长捉都只允许一轮：同局面第 2 次出现时立即裁决，
        // 并据此在走法生成时过滤第二轮的重复将军或捉子。优先级：将 > 捉。
        match (red_violation, black_violation) {
            (Some(RuleViolation::LongCheck), Some(RuleViolation::LongCheck)) => {
                return Some(RuleOutcome::Draw(RuleDrawReason::MutualLongCheck));
            }
            (Some(RuleViolation::LongCheck), _) => return Some(RuleOutcome::Win(Color::Black)),
            (_, Some(RuleViolation::LongCheck)) => return Some(RuleOutcome::Win(Color::Red)),
            _ => {}
        }

        match (red_violation, black_violation) {
            (Some(RuleViolation::LongChase), Some(RuleViolation::LongChase)) => {
                return Some(RuleOutcome::Draw(RuleDrawReason::MutualLongChase));
            }
            (Some(RuleViolation::LongChase), _) => return Some(RuleOutcome::Win(Color::Black)),
            (_, Some(RuleViolation::LongChase)) => return Some(RuleOutcome::Win(Color::Red)),
            _ => {}
        }

        Some(RuleOutcome::Draw(RuleDrawReason::Repetition))
    }

    pub fn legal_moves_with_rules(&self, history: &[RuleHistoryEntry]) -> Vec<Move> {
        crate::scope_profile!("xiangqi.legal_moves_with_rules");
        self.legal_moves_with_rules_and_repetition(history)
            .into_iter()
            .map(|(mv, _)| mv)
            .collect()
    }

    pub fn legal_moves_with_rules_and_repetition(
        &self,
        history: &[RuleHistoryEntry],
    ) -> Vec<(Move, bool)> {
        crate::scope_profile!("xiangqi.legal_moves_with_rules_and_repetition");
        let legal = self.legal_moves();
        if legal.is_empty() {
            return Vec::new();
        }

        let current_entry = (!history.last().is_some_and(|entry| {
            entry.hash == self.hash && entry.side_to_move == self.side_to_move
        }))
        .then(|| self.rule_history_entry(None));

        let mover = self.side_to_move;
        legal
            .into_iter()
            .filter_map(|mv| {
                let next_hash = self.hash_after_move(mv);
                let next_side_to_move = mover.opposite();
                let repeats_history = history.iter().any(|entry| {
                    entry.hash == next_hash && entry.side_to_move == next_side_to_move
                });
                if !repeats_history {
                    return Some((mv, false));
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
                (!rule_outcome_forbidden_for_mover(
                    next.rule_outcome_with_history(&next_history),
                    mover,
                ))
                .then_some((mv, true))
            })
            .collect()
    }

    fn chased_masks_by(&self, color: Color) -> u128 {
        crate::scope_profile!("xiangqi.chased_mask_by");
        let mut work = self.clone();
        work.side_to_move = color;

        let mut square_mask = 0u128;
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
                    return true;
                }
                false
            });
        }
        square_mask
    }

    fn chased_masks_by_origin(&self, color: Color, origin: usize) -> u128 {
        crate::scope_profile!("xiangqi.chased_mask_by");
        let Some(piece) = self.board[origin].filter(|piece| piece.color == color) else {
            return 0;
        };
        let mut captures = Vec::with_capacity(8);
        self.gen_piece_moves(origin, piece, MoveGenMode::Captures, &mut captures);
        let mut work = self.clone();
        work.side_to_move = color;

        let mut square_mask = 0u128;
        for mv in captures {
            let target = mv.to as usize;
            let Some(target_piece) = self.board[target] else {
                continue;
            };
            if !self.is_chase_target_piece(target_piece, color, target) {
                continue;
            }
            if !self.is_effective_chase(target_piece, target, origin) {
                continue;
            }
            let captured = work.make_move_board_only(mv);
            let legal = !work.in_check(color);
            work.unmake_move_board_only(mv, captured);
            if legal {
                square_mask |= 1u128 << target;
            }
        }
        square_mask
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

    fn insufficient_material_outcome(&self) -> Option<RuleOutcome> {
        if self.dynamic_material_counts[0] + self.dynamic_material_counts[1] > 2 {
            return None;
        }
        let mut counts = [[0u8; 7]; 2];
        for piece in self.board.iter().flatten() {
            let color = super::color_index(piece.color);
            let kind = super::piece_kind_index(piece.kind);
            counts[color][kind] += 1;
        }
        let count = |color: usize, kind: PieceKind| counts[color][super::piece_kind_index(kind)];
        let total = |kind: PieceKind| count(0, kind) + count(1, kind);
        if total(PieceKind::Soldier) != 0 {
            return None;
        }

        let attacking = |color: usize| {
            count(color, PieceKind::Rook)
                + count(color, PieceKind::Cannon)
                + count(color, PieceKind::Horse)
        };
        let total_attacking = attacking(0) + attacking(1);
        let direct_draw = if total_attacking == 0 {
            true
        } else if total_attacking == 1 && total(PieceKind::Cannon) == 1 {
            let cannon_side = usize::from(count(0, PieceKind::Cannon) == 0);
            let other = 1 - cannon_side;
            count(cannon_side, PieceKind::Advisor) == 0
                && match count(other, PieceKind::Advisor) {
                    0 => true,
                    1 => count(cannon_side, PieceKind::Elephant) == 0,
                    _ => false,
                }
        } else {
            total_attacking == 2
                && count(0, PieceKind::Cannon) == 1
                && count(1, PieceKind::Cannon) == 1
                && total(PieceKind::Advisor) == 0
                && total(PieceKind::Elephant) == 0
        };
        if direct_draw {
            return Some(RuleOutcome::Draw(RuleDrawReason::InsufficientMaterial));
        }

        let mate_sensitive_draw = if total_attacking == 1 && total(PieceKind::Cannon) == 1 {
            let cannon_side = usize::from(count(0, PieceKind::Cannon) == 0);
            let other = 1 - cannon_side;
            count(cannon_side, PieceKind::Advisor) == 0
                && ((count(other, PieceKind::Advisor) == 1
                    && count(cannon_side, PieceKind::Elephant) != 0)
                    || (count(other, PieceKind::Advisor) >= 2
                        && count(cannon_side, PieceKind::Elephant) == 0))
        } else {
            total_attacking == 2
                && count(0, PieceKind::Cannon) == 1
                && count(1, PieceKind::Cannon) == 1
                && total(PieceKind::Advisor) == 0
        };
        if !mate_sensitive_draw {
            return None;
        }

        let legal = self.legal_moves();
        if legal.is_empty() {
            return Some(RuleOutcome::Win(self.side_to_move.opposite()));
        }
        for mv in legal {
            let mut next = self.clone();
            next.make_move(mv);
            if next.legal_moves().is_empty() {
                return None;
            }
        }
        Some(RuleOutcome::Draw(RuleDrawReason::InsufficientMaterial))
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

    let mut identity_at_square = std::array::from_fn::<_, { super::BOARD_SIZE }, _>(|sq| sq as u8);
    let mut chased_identity_intersection = None::<u128>;
    for entry in entries {
        if let Some(mv) = entry.mv {
            let identity = identity_at_square[mv.from as usize];
            identity_at_square[mv.from as usize] = u8::MAX;
            identity_at_square[mv.to as usize] = identity;
        }
        if entry.mover != Some(color) {
            continue;
        }
        let mut squares = entry.chased_mask;
        let mut identities = 0u128;
        while squares != 0 {
            let square = squares.trailing_zeros() as usize;
            squares &= squares - 1;
            let identity = identity_at_square[square];
            if identity != u8::MAX {
                identities |= 1u128 << identity;
            }
        }
        chased_identity_intersection =
            Some(chased_identity_intersection.map_or(identities, |current| current & identities));
    }
    chased_identity_intersection
        .is_some_and(|identities| identities != 0)
        .then_some(RuleViolation::LongChase)
}
