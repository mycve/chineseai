use super::{
    BOARD_FILES, BOARD_RANKS, Color, ELEPHANT_STEPS, HORSE_STEPS, Move, MoveGenMode,
    ORTHOGONAL_STEPS, Piece, PieceKind, Position, file_of, index, inside_board, offset_square,
    rank_of,
};

impl Position {
    pub fn legal_moves(&self) -> Vec<Move> {
        crate::scope_profile!("xiangqi.legal_moves");
        self.collect_legal_moves(false, self.in_check(self.side_to_move))
    }

    #[cfg(test)]
    pub(super) fn legal_capture_moves(&self) -> Vec<Move> {
        self.collect_legal_moves(true, self.in_check(self.side_to_move))
    }

    #[cfg(test)]
    pub(super) fn legal_capture_moves_to(&self, target: usize) -> Vec<Move> {
        let Some(target_piece) = self.board.get(target).and_then(|piece| *piece) else {
            return Vec::new();
        };
        if target_piece.color == self.side_to_move || target_piece.kind == PieceKind::General {
            return Vec::new();
        }

        let mut legal = Vec::new();
        let mut work = self.clone();
        self.visit_attacker_origins_to(target, self.side_to_move, |from| {
            let mv = Move::new(from, target);
            let captured = work.make_move_board_only(mv);
            if !work.in_check(self.side_to_move) {
                legal.push(mv);
            }
            work.unmake_move_board_only(mv, captured);
            false
        });

        legal
    }

    pub fn parse_uci_move(&self, uci: &str) -> Option<Move> {
        let candidate = Move::from_uci(uci)?;
        self.is_legal_move(candidate).then_some(candidate)
    }

    pub fn is_legal_move(&self, mv: Move) -> bool {
        let from = mv.from as usize;
        let to = mv.to as usize;
        let Some(piece) = self.board.get(from).and_then(|piece| *piece) else {
            return false;
        };
        if piece.color != self.side_to_move {
            return false;
        }
        if self
            .board
            .get(to)
            .and_then(|target| *target)
            .is_some_and(|target| {
                target.color == self.side_to_move || target.kind == PieceKind::General
            })
        {
            return false;
        }

        let mut pseudo = Vec::with_capacity(16);
        self.gen_piece_moves(from, piece, MoveGenMode::All, &mut pseudo);
        if !pseudo.contains(&mv) {
            return false;
        }

        let mut work = self.clone();
        let captured = work.make_move_board_only(mv);
        let legal = !work.in_check(self.side_to_move);
        work.unmake_move_board_only(mv, captured);
        legal
    }

    pub fn in_check(&self, color: Color) -> bool {
        crate::scope_profile!("xiangqi.in_check");
        let king_sq = self
            .find_general(color)
            .expect("every valid position must contain both generals");
        self.is_square_attacked(king_sq, color.opposite())
    }

    fn collect_legal_moves(&self, captures_only: bool, in_check: bool) -> Vec<Move> {
        crate::scope_profile!("xiangqi.collect_legal_moves");
        let mut moves = if in_check {
            crate::scope_profile!("xiangqi.pseudo_legal_moves");
            self.pseudo_legal_evasions()
        } else if captures_only {
            crate::scope_profile!("xiangqi.pseudo_legal_moves");
            self.pseudo_legal_capture_moves()
        } else {
            crate::scope_profile!("xiangqi.pseudo_legal_moves");
            self.pseudo_legal_moves()
        };
        let mut work = self.clone();
        let needs_capture_filter = captures_only && in_check;
        let mut legal_len = 0usize;
        let safety_check_from_mask = (!in_check)
            .then(|| self.safety_check_from_mask_when_not_in_check(self.side_to_move))
            .unwrap_or(u128::MAX);
        // 不在被将军状态时，落子只会让 `from` 变空；`to` 的占用状态不变
        // （己方子落到 `to`，或被吃的敌子被同格的己方子替换），而增加占用只能挡线、
        // 不可能造出攻击。因此"这步是否会暴露国王"只取决于 `from` 空出来之后
        // 国王是否挨打——不需要 make/unmake，也不需要重扫全部攻击者。
        let side = self.side_to_move();
        let enemy = side.opposite();
        let fast = (!in_check)
            .then(|| self.find_general(side))
            .flatten()
            .map(|king_sq| (king_sq, self.leaper_unblock_mask(king_sq, enemy)));

        {
            crate::scope_profile!("xiangqi.legal_filter");
            for read_index in 0..moves.len() {
                let mv = moves[read_index];
                let from = mv.from as usize;
                let to = mv.to as usize;
                let requires_safety_check = in_check
                    || matches!(
                        self.board[from],
                        Some(Piece {
                            kind: PieceKind::General,
                            ..
                        })
                    )
                    || ((safety_check_from_mask >> from) & 1) != 0
                    || ((safety_check_from_mask >> to) & 1) != 0;
                if !requires_safety_check {
                    moves[legal_len] = mv;
                    legal_len += 1;
                    continue;
                }

                let safe_after_move = match fast {
                    // 国王自己走子时不能用这条捷径（走完国王换了格子）。
                    Some((king_sq, unblock)) if from != king_sq => !self
                        .king_attacked_after_vacating(king_sq, side, enemy, from, to, unblock, &mut work),
                    _ => {
                        let captured = work.make_move_board_only(mv);
                        let ok = !work.in_check(side);
                        work.unmake_move_board_only(mv, captured);
                        ok
                    }
                };
                if safe_after_move && (!needs_capture_filter || self.is_capture(mv)) {
                    moves[legal_len] = mv;
                    legal_len += 1;
                }
            }
        }

        moves.truncate(legal_len);
        moves
    }

    /// `from` 空出来之后，敌方的马腿/象眼是否会被松开；只有这些格子需要重扫 leaper。
    pub(super) fn leaper_unblock_mask(&self, king_sq: usize, enemy: Color) -> u128 {
        let file = file_of(king_sq) as i32;
        let rank = rank_of(king_sq) as i32;
        let mut mask = 0u128;
        for ((leg_df, leg_dr), (move_df, move_dr)) in HORSE_STEPS {
            let Some(square) = offset_square(file, rank, -move_df, -move_dr) else {
                continue;
            };
            if matches!(
                self.board[square],
                Some(Piece {
                    color,
                    kind: PieceKind::Horse
                }) if color == enemy
            ) {
                if let Some(leg) = offset_square(
                    file_of(square) as i32,
                    rank_of(square) as i32,
                    leg_df,
                    leg_dr,
                ) {
                    mask |= 1u128 << leg;
                }
            }
        }
        for ((eye_df, eye_dr), (move_df, move_dr)) in ELEPHANT_STEPS {
            let Some(square) = offset_square(file, rank, -move_df, -move_dr) else {
                continue;
            };
            if matches!(
                self.board[square],
                Some(Piece {
                    color,
                    kind: PieceKind::Elephant
                }) if color == enemy
            ) {
                if let Some(eye) = offset_square(
                    file_of(square) as i32,
                    rank_of(square) as i32,
                    eye_df,
                    eye_dr,
                ) {
                    mask |= 1u128 << eye;
                }
            }
        }
        mask
    }

    /// `from` 空出来、`to` 落下己方子之后，国王是否被攻击。
    /// 前提：当前不在被将军状态，且 `from` 不是国王的格子。
    ///
    /// 两处占用变化都要考虑：`from` 变空可能松开直线或马腿/象眼；
    /// `to` 由空变满可能**给敌方炮造出一个炮架**（0 个挡子 → 1 个挡子），
    /// 所以不能只算 `from`。
    #[allow(clippy::too_many_arguments)]
    pub(super) fn king_attacked_after_vacating(
        &self,
        king_sq: usize,
        side: Color,
        enemy: Color,
        from: usize,
        to: usize,
        unblock: u128,
        work: &mut Position,
    ) -> bool {
        // 直线子与飞将：沿四个方向从国王向外走，把 `from` 当作空格、把 `to` 当作己方子。
        let king_file = file_of(king_sq) as i32;
        let king_rank = rank_of(king_sq) as i32;
        for (df, dr) in ORTHOGONAL_STEPS {
            let mut seen_screen = false;
            let mut nf = king_file + df;
            let mut nr = king_rank + dr;
            while inside_board(nf, nr) {
                let sq = index(nf as usize, nr as usize);
                if sq != from {
                    // `to` 一格走完一定由己方子占据：原本是空格则新落下，
                    // 原本是敌子则被吃掉替换。两种情况都必须按己方子读，
                    // 否则会把被吃掉的敌车当成还在攻击。
                    let occupant = if sq == to {
                        Some(Piece {
                            color: side,
                            kind: PieceKind::Soldier,
                        })
                    } else {
                        self.board[sq]
                    };
                    if let Some(piece) = occupant {
                        if !seen_screen {
                            if piece.color == enemy {
                                if piece.kind == PieceKind::Rook {
                                    return true;
                                }
                                if piece.kind == PieceKind::General && df == 0 {
                                    return true;
                                }
                            }
                            seen_screen = true;
                        } else if piece.color == enemy && piece.kind == PieceKind::Cannon {
                            return true;
                        } else {
                            break;
                        }
                    }
                }
                nf += df;
                nr += dr;
            }
        }
        // 马腿/象眼：只有 `from` 恰好是那条腿/眼时才可能从"挡住"变成"松开"。
        if unblock & (1u128 << from) != 0 {
            let saved_from = work.board[from];
            let saved_to = work.board[to];
            work.board[from] = None;
            // 走完之后 `to` 一定由己方子占据（含吃掉敌子的情况）。
            work.board[to] = Some(Piece {
                color: side,
                kind: PieceKind::Soldier,
            });
            let attacked = work.is_square_attacked_by_leapers(king_sq, enemy);
            work.board[from] = saved_from;
            work.board[to] = saved_to;
            return attacked;
        }
        false
    }

    fn safety_check_from_mask_when_not_in_check(&self, color: Color) -> u128 {
        let Some(king_sq) = self.find_general(color) else {
            return u128::MAX;
        };
        let mut mask = 0u128;
        let king_file = file_of(king_sq);
        let king_rank = rank_of(king_sq);
        let enemy = color.opposite();
        let mut needs_file = false;
        let mut needs_rank = false;
        let mut pieces = self.occupied;
        while pieces != 0 && !(needs_file && needs_rank) {
            let square = pieces.trailing_zeros() as usize;
            pieces &= pieces - 1;
            let Some(piece) = self.board[square] else {
                continue;
            };
            if piece.color != enemy
                || !matches!(
                    piece.kind,
                    PieceKind::Rook | PieceKind::Cannon | PieceKind::General
                )
            {
                continue;
            }
            needs_file |= file_of(square) == king_file;
            needs_rank |= rank_of(square) == king_rank;
        }
        if needs_file {
            for rank in 0..BOARD_RANKS {
                mask |= 1u128 << index(king_file, rank);
            }
        }
        if needs_rank {
            for file in 0..BOARD_FILES {
                mask |= 1u128 << index(file, king_rank);
            }
        }

        self.add_horse_leg_attack_mask(king_sq, color.opposite(), &mut mask);
        mask
    }
}
