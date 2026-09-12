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
            captured: None,
            rule60_clock: self.halfmove_clock,
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
        next.rule_history_entry_after_moved(mover, mv, self.piece_at(mv.to as usize))
    }

    pub fn rule_history_entry_after_moved(
        &self,
        mover: Color,
        mv: Move,
        captured: Option<super::Piece>,
    ) -> RuleHistoryEntry {
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
            captured,
            rule60_clock: self.halfmove_clock,
        }
    }

    pub fn rule_outcome_with_history(&self, history: &[RuleHistoryEntry]) -> Option<RuleOutcome> {
        crate::scope_profile!("xiangqi.rule_outcome_with_history");
        if let Some(entries) = repetition_cycle(history) {
            let exact_entries = self.recompute_cycle_chases(entries);
            return Some(adjudicate_repetition(
                exact_entries.as_deref().unwrap_or(entries),
            ));
        }
        if self
            .rule60_max_ply
            .is_some_and(|max_ply| self.rule60_count_with_history(history) >= max_ply)
        {
            return Some(RuleOutcome::Draw(RuleDrawReason::NaturalMoveLimit));
        }
        self.insufficient_material_outcome()
    }

    pub fn rule60_count_with_history(&self, history: &[RuleHistoryEntry]) -> u16 {
        // Exemptions cannot begin before one side has made its 11th check,
        // which takes at least 21 capture-free plies. Keep the overwhelmingly
        // common search path O(1).
        if self.halfmove_clock <= 20 {
            return self.halfmove_clock;
        }
        let Some(initial) = history.first() else {
            return self.halfmove_clock;
        };
        let mut clock = initial.rule60_clock;
        let mut checks = [0u16; 2];
        let mut previous_excess_check = false;
        for entry in &history[1..] {
            if entry.captured.is_some() {
                clock = 0;
                checks = [0; 2];
                previous_excess_check = false;
                continue;
            }
            if entry.gives_check {
                let Some(mover) = entry.mover else {
                    continue;
                };
                let index = super::color_index(mover);
                checks[index] = checks[index].saturating_add(1);
                previous_excess_check = checks[index] > 10;
                if !previous_excess_check {
                    clock = clock.saturating_add(1);
                }
            } else if previous_excess_check {
                previous_excess_check = false;
            } else {
                clock = clock.saturating_add(1);
            }
        }
        clock
    }

    pub fn rule_outcome(history: &[RuleHistoryEntry]) -> Option<RuleOutcome> {
        crate::scope_profile!("xiangqi.rule_outcome");
        repetition_cycle(history).map(adjudicate_repetition)
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
        let mut captures = Vec::with_capacity(16);
        let mut square_mask = 0u128;
        for origin in 0..super::BOARD_SIZE {
            let Some(piece) = self.board[origin].filter(|piece| piece.color == color) else {
                continue;
            };
            if matches!(piece.kind, PieceKind::General | PieceKind::Soldier) {
                continue;
            }
            captures.clear();
            self.gen_piece_moves(origin, piece, MoveGenMode::Captures, &mut captures);
            for &mv in &captures {
                let target = mv.to as usize;
                let Some(target_piece) = self.board[target] else {
                    continue;
                };
                if !self.is_chase_target_piece(target_piece, color, target)
                    || !self.is_effective_chase(target_piece, target, origin)
                {
                    continue;
                }
                let captured = work.make_move_board_only(mv);
                let legal = !work.in_check(color);
                work.unmake_move_board_only(mv, captured);
                if legal {
                    square_mask |= 1u128 << target;
                }
            }
        }
        square_mask
    }

    #[cfg(test)]
    fn chased_masks_by_target_scan(&self, color: Color) -> u128 {
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
                if self.board[from].is_some_and(|piece| {
                    matches!(piece.kind, PieceKind::General | PieceKind::Soldier)
                }) || !self.is_effective_chase(target_piece, target, from)
                {
                    return false;
                }
                let mv = Move::new(from, target);
                let captured = work.make_move_board_only(mv);
                let legal = !work.in_check(color);
                work.unmake_move_board_only(mv, captured);
                if legal {
                    square_mask |= 1u128 << target;
                }
                legal
            });
        }
        square_mask
    }

    fn chased_masks_by_origin(&self, color: Color, origin: usize) -> u128 {
        crate::scope_profile!("xiangqi.chased_mask_by");
        let Some(piece) = self.board[origin].filter(|piece| piece.color == color) else {
            return 0;
        };
        if matches!(piece.kind, PieceKind::General | PieceKind::Soldier) {
            return 0;
        }
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
            PieceKind::General => false,
            PieceKind::Soldier => soldier_crossed_river(piece.color, super::geom::rank_of(sq)),
            PieceKind::Advisor
            | PieceKind::Elephant
            | PieceKind::Horse
            | PieceKind::Rook
            | PieceKind::Cannon => true,
        }
    }

    fn is_effective_chase(&self, target: super::Piece, target_sq: usize, from: usize) -> bool {
        let Some(attacker) = self.board[from] else {
            return false;
        };
        if matches!(
            (attacker.kind, target.kind),
            (PieceKind::Horse | PieceKind::Cannon, PieceKind::Rook)
                | (
                    PieceKind::Advisor | PieceKind::Elephant,
                    PieceKind::Rook | PieceKind::Cannon | PieceKind::Horse
                )
        ) {
            return true;
        }
        if attacker.kind == target.kind {
            let mut reverse = self.clone();
            reverse.side_to_move = target.color;
            if reverse.is_legal_move(Move::new(target_sq, from)) {
                return false;
            }
        }

        let mut after = self.clone();
        after.side_to_move = attacker.color;
        after.make_move_board_only(Move::new(from, target_sq));
        after.side_to_move = target.color;
        let recapture_position = after.clone();
        let mut legal_recapture = false;
        recapture_position.visit_attacker_origins_to(target_sq, target.color, |recapture_from| {
            let mv = Move::new(recapture_from, target_sq);
            let captured = after.make_move_board_only(mv);
            let legal = !after.in_check(target.color);
            after.unmake_move_board_only(mv, captured);
            if legal {
                legal_recapture = true;
                return true;
            }
            false
        });
        !legal_recapture
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

    fn recompute_cycle_chases(
        &self,
        entries: &[RuleHistoryEntry],
    ) -> Option<Vec<RuleHistoryEntry>> {
        let mut position = self.clone();
        for entry in entries.iter().rev() {
            let mv = entry.mv?;
            position.unmake_move_board_only(mv, entry.captured);
            position.side_to_move = entry.mover?;
        }

        let mut exact = Vec::with_capacity(entries.len());
        for &entry in entries {
            let mv = entry.mv?;
            let mover = entry.mover?;
            let before = position.chased_masks_by(mover);
            let captured = position.make_move_board_only(mv);
            debug_assert_eq!(captured, entry.captured);
            position.side_to_move = mover.opposite();
            let after = position.chased_masks_by(mover);
            exact.push(RuleHistoryEntry {
                chased_mask: after & !before,
                ..entry
            });
        }
        Some(exact)
    }
}

fn repetition_cycle(history: &[RuleHistoryEntry]) -> Option<&[RuleHistoryEntry]> {
    let current_index = history.len().checked_sub(1)?;
    let current = history[current_index];
    let cycle_start = history[..current_index].iter().position(|entry| {
        entry.hash == current.hash && entry.side_to_move == current.side_to_move
    })? + 1;
    Some(&history[cycle_start..=current_index])
}

fn adjudicate_repetition(entries: &[RuleHistoryEntry]) -> RuleOutcome {
    let red_violation = repeated_rule_violation(entries, Color::Red);
    let black_violation = repeated_rule_violation(entries, Color::Black);

    // Asian 2fold: 长将 > 长捉同一子 > 其他循环。
    match (red_violation, black_violation) {
        (Some(RuleViolation::LongCheck), Some(RuleViolation::LongCheck)) => {
            return RuleOutcome::Draw(RuleDrawReason::MutualLongCheck);
        }
        (Some(RuleViolation::LongCheck), _) => return RuleOutcome::Win(Color::Black),
        (_, Some(RuleViolation::LongCheck)) => return RuleOutcome::Win(Color::Red),
        _ => {}
    }
    match (red_violation, black_violation) {
        (Some(RuleViolation::LongChase), Some(RuleViolation::LongChase)) => {
            RuleOutcome::Draw(RuleDrawReason::MutualLongChase)
        }
        (Some(RuleViolation::LongChase), _) => RuleOutcome::Win(Color::Black),
        (_, Some(RuleViolation::LongChase)) => RuleOutcome::Win(Color::Red),
        _ => RuleOutcome::Draw(RuleDrawReason::Repetition),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_cycle_chase_diff_detects_discovered_attack() {
        let mut position = Position::from_fen("3k5/9/9/9/n8/9/9/C8/9/R3K4 w").unwrap();
        let horse = super::super::geom::index(0, 4);
        let before = position.chased_masks_by(Color::Red);
        let mv = position.parse_uci_move("a2b2").unwrap();
        let recorded = position.rule_history_entry_after_move(mv);
        assert_eq!(recorded.chased_mask & (1u128 << horse), 0);
        position.make_move(mv);
        let after = position.chased_masks_by(Color::Red);
        assert_ne!((after & !before) & (1u128 << horse), 0);
        let exact = position.recompute_cycle_chases(&[recorded]).unwrap();
        assert_ne!(exact[0].chased_mask & (1u128 << horse), 0);
    }

    #[test]
    fn origin_scan_chase_masks_match_target_scan() {
        let mut position = Position::startpos();
        for ply in 0..160usize {
            for color in [Color::Red, Color::Black] {
                assert_eq!(
                    position.chased_masks_by(color),
                    position.chased_masks_by_target_scan(color),
                    "mismatch at ply {ply}, color {color:?}, fen {}",
                    position.to_fen()
                );
            }
            let legal = position.legal_moves();
            if legal.is_empty() {
                break;
            }
            let index = (ply.wrapping_mul(37).wrapping_add(11)) % legal.len();
            position.make_move(legal[index]);
        }
    }
}
