pub const BOARD_FILES: usize = 9;
pub const BOARD_RANKS: usize = 10;
pub const BOARD_SIZE: usize = BOARD_FILES * BOARD_RANKS;
pub const STARTPOS_FEN: &str = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w";

mod env;
mod eval;
mod geom;
mod hash;
mod rules;
mod types;

pub use env::{AppliedMove, IllegalMove, StepOutcome, XiangqiEnv};
pub(crate) use eval::piece_base_value;
pub use geom::{parse_square, square_name};
pub use types::{
    Color, Move, Piece, PieceKind, Position, RuleDrawReason, RuleHistoryEntry, RuleOutcome, Undo,
};

use eval::signed_piece_contrib;
#[cfg(test)]
use geom::same_rank_or_file;
use geom::{
    elephant_stays_home, file_of, horse_leg_square, index, inside_board, inside_palace,
    line_between_squares, rank_of, soldier_crossed_river,
};
use hash::{SIDE_TO_MOVE_KEY, color_hash_index, zobrist_piece_key};
use types::{CheckerInfo, MoveGenMode, PositionState};

impl Default for Position {
    fn default() -> Self {
        Self::startpos()
    }
}

impl Position {
    pub fn startpos() -> Self {
        Self::from_fen(STARTPOS_FEN).expect("valid start position")
    }

    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let mut parts = fen.split_whitespace();
        let board_part = parts.next().ok_or("missing board description")?;
        let side_part = parts.next().unwrap_or("w");
        let halfmove_clock = parts
            .next()
            .and_then(|value| value.parse::<u16>().ok())
            .unwrap_or(0);

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
            side_to_move,
            hash: 0,
            base_eval: 0,
            advisor_counts: [0; 2],
            elephant_counts: [0; 2],
            dynamic_material_counts: [0; 2],
            general_squares: [None; 2],
            halfmove_clock,
        };
        let state = position.compute_state();
        let position = Self {
            hash: state.hash,
            base_eval: state.base_eval,
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

        format!("{board_part} {side}")
    }

    #[inline(always)]
    pub fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    #[inline(always)]
    pub fn piece_at(&self, sq: usize) -> Option<Piece> {
        self.board[sq]
    }

    #[inline(always)]
    pub fn has_general(&self, color: Color) -> bool {
        self.general_squares[color_hash_index(color)].is_some()
    }

    #[inline(always)]
    pub fn general_square(&self, color: Color) -> Option<usize> {
        self.general_squares[color_hash_index(color)]
    }

    #[cfg(test)]
    #[inline(always)]
    fn has_dynamic_material(&self, color: Color) -> bool {
        self.dynamic_material_counts[color_hash_index(color)] > 0
    }

    #[inline(always)]
    pub fn is_capture(&self, mv: Move) -> bool {
        self.board[mv.to as usize].is_some()
    }

    #[inline(always)]
    pub fn hash(&self) -> u64 {
        self.hash
    }

    #[inline(always)]
    pub fn halfmove_clock(&self) -> u16 {
        self.halfmove_clock
    }

    pub fn is_piece_protected(&self, sq: usize, color: Color) -> bool {
        self.visit_attacker_origins_to(sq, color, |from| from != sq)
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        crate::scope_profile!("xiangqi.legal_moves");
        self.collect_legal_moves(false, self.in_check(self.side_to_move))
    }

    #[cfg(test)]
    pub(super) fn legal_capture_moves(&self) -> Vec<Move> {
        self.collect_legal_moves(true, self.in_check(self.side_to_move))
    }

    #[cfg(test)]
    fn legal_capture_moves_to(&self, target: usize) -> Vec<Move> {
        let Some(target_piece) = self.board.get(target).and_then(|piece| *piece) else {
            return Vec::new();
        };
        if target_piece.color == self.side_to_move {
            return Vec::new();
        }

        let mut legal = Vec::new();
        let mut work = self.clone();
        self.visit_attacker_origins_to(target, self.side_to_move, |from| {
            let mv = Move::new(from, target);
            let undo = work.make_move(mv);
            if !work.in_check(self.side_to_move) {
                legal.push(mv);
            }
            work.unmake_move(mv, undo);
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
            .is_some_and(|target| target.color == self.side_to_move)
        {
            return false;
        }

        let mut pseudo = Vec::with_capacity(16);
        self.gen_piece_moves(from, piece, MoveGenMode::All, &mut pseudo);
        if !pseudo.contains(&mv) {
            return false;
        }

        let mut work = self.clone();
        let undo = work.make_move(mv);
        let legal = !work.in_check(self.side_to_move);
        work.unmake_move(mv, undo);
        legal
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
            self.base_eval -= signed_piece_contrib(captured, to);
            self.adjust_minor_counts(captured, -1);
            self.adjust_dynamic_material_counts(captured, -1);
            if captured.kind == PieceKind::General {
                self.general_squares[color_hash_index(captured.color)] = None;
            }
        }

        self.base_eval -= signed_piece_contrib(moving, from);
        self.board[to] = Some(moving);
        self.board[from] = None;
        if moving.kind == PieceKind::General {
            self.general_squares[color_hash_index(moving.color)] = Some(to);
        }
        self.hash ^= zobrist_piece_key(to, moving);
        self.base_eval += signed_piece_contrib(moving, to);
        self.side_to_move = self.side_to_move.opposite();
        self.hash ^= SIDE_TO_MOVE_KEY;
        self.halfmove_clock = if undo.captured.is_some() || moving.kind == PieceKind::Soldier {
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
        self.base_eval -= signed_piece_contrib(moving, to);
        self.board[from] = Some(moving);
        self.board[to] = undo.captured;
        if moving.kind == PieceKind::General {
            self.general_squares[color_hash_index(moving.color)] = Some(from);
        }
        self.hash ^= zobrist_piece_key(from, moving);
        self.base_eval += signed_piece_contrib(moving, from);
        if let Some(captured) = undo.captured {
            self.hash ^= zobrist_piece_key(to, captured);
            self.base_eval += signed_piece_contrib(captured, to);
            self.adjust_minor_counts(captured, 1);
            self.adjust_dynamic_material_counts(captured, 1);
            if captured.kind == PieceKind::General {
                self.general_squares[color_hash_index(captured.color)] = Some(to);
            }
        }
        self.side_to_move = undo.side_to_move;
        self.halfmove_clock = undo.halfmove_clock;
    }

    pub fn in_check(&self, color: Color) -> bool {
        crate::scope_profile!("xiangqi.in_check");
        let king_sq = self
            .find_general(color)
            .expect("every valid position must contain both generals");
        self.is_square_attacked(king_sq, color.opposite())
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
        let mut base_eval = 0;
        let mut advisor_counts = [0u8; 2];
        let mut elephant_counts = [0u8; 2];
        let mut dynamic_material_counts = [0u8; 2];
        let mut general_squares = [None; 2];
        for sq in 0..BOARD_SIZE {
            let Some(piece) = self.board[sq] else {
                continue;
            };
            hash ^= zobrist_piece_key(sq, piece);
            base_eval += signed_piece_contrib(piece, sq);
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
            base_eval,
            advisor_counts,
            elephant_counts,
            dynamic_material_counts,
            general_squares,
        }
    }

    #[cfg(test)]
    fn compute_hash(&self) -> u64 {
        self.compute_state().hash
    }

    fn adjust_minor_counts(&mut self, piece: Piece, delta: i8) {
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

    fn adjust_dynamic_material_counts(&mut self, piece: Piece, delta: i8) {
        let index = color_hash_index(piece.color);
        match piece.kind {
            PieceKind::Rook | PieceKind::Cannon | PieceKind::Horse | PieceKind::Soldier => {
                self.dynamic_material_counts[index] =
                    (self.dynamic_material_counts[index] as i16 + delta as i16).max(0) as u8;
            }
            _ => {}
        }
    }

    fn pseudo_legal_moves(&self) -> Vec<Move> {
        self.pseudo_legal_moves_with_mode(MoveGenMode::All)
    }

    fn pseudo_legal_capture_moves(&self) -> Vec<Move> {
        self.pseudo_legal_moves_with_mode(MoveGenMode::Captures)
    }

    fn pseudo_legal_moves_with_mode(&self, mode: MoveGenMode) -> Vec<Move> {
        let mut moves = Vec::with_capacity(64);

        for sq in 0..BOARD_SIZE {
            let Some(piece) = self.board[sq] else {
                continue;
            };
            if piece.color != self.side_to_move {
                continue;
            }

            self.gen_piece_moves(sq, piece, mode, &mut moves);
        }

        moves
    }

    fn collect_legal_moves(&self, captures_only: bool, in_check: bool) -> Vec<Move> {
        crate::scope_profile!("xiangqi.collect_legal_moves");
        let mut moves = if in_check {
            self.pseudo_legal_evasions()
        } else if captures_only {
            self.pseudo_legal_capture_moves()
        } else {
            self.pseudo_legal_moves()
        };
        let mut work = self.clone();
        let needs_capture_filter = captures_only && in_check;
        let mut legal_len = 0usize;

        for read_index in 0..moves.len() {
            let mv = moves[read_index];
            let undo = work.make_move(mv);
            if !work.in_check(self.side_to_move) && (!needs_capture_filter || self.is_capture(mv)) {
                moves[legal_len] = mv;
                legal_len += 1;
            }
            work.unmake_move(mv, undo);
        }

        moves.truncate(legal_len);
        moves
    }

    fn pseudo_legal_evasions(&self) -> Vec<Move> {
        let king_sq = self
            .find_general(self.side_to_move)
            .expect("side to move must have a general");
        let Some(king_piece) = self.board[king_sq] else {
            return Vec::new();
        };
        let checkers = self.checkers_to(king_sq, self.side_to_move.opposite());

        let mut evasions = Vec::with_capacity(24);
        self.gen_general_moves(king_sq, king_piece, MoveGenMode::All, &mut evasions);
        if checkers.len() != 1 {
            return evasions;
        }

        let checker = checkers[0];
        let mut allow_to = [false; BOARD_SIZE];
        allow_to[checker.from] = true;
        for sq in self.interposition_squares(king_sq, &checker) {
            allow_to[sq] = true;
        }

        let mut piece_moves = Vec::with_capacity(16);
        for sq in 0..BOARD_SIZE {
            let Some(piece) = self.board[sq] else {
                continue;
            };
            if piece.color != self.side_to_move || piece.kind == PieceKind::General {
                continue;
            }

            piece_moves.clear();
            self.gen_piece_moves(sq, piece, MoveGenMode::All, &mut piece_moves);
            for &mv in &piece_moves {
                let from = mv.from as usize;
                let to = mv.to as usize;
                if allow_to[to] || checker.screen_square == Some(from) {
                    evasions.push(mv);
                }
            }
        }

        evasions
    }

    fn gen_piece_moves(&self, sq: usize, piece: Piece, mode: MoveGenMode, moves: &mut Vec<Move>) {
        match piece.kind {
            PieceKind::General => self.gen_general_moves(sq, piece, mode, moves),
            PieceKind::Advisor => self.gen_advisor_moves(sq, piece, mode, moves),
            PieceKind::Elephant => self.gen_elephant_moves(sq, piece, mode, moves),
            PieceKind::Horse => self.gen_horse_moves(sq, piece, mode, moves),
            PieceKind::Rook => self.gen_rook_moves(sq, piece, mode, moves),
            PieceKind::Cannon => self.gen_cannon_moves(sq, piece, mode, moves),
            PieceKind::Soldier => self.gen_soldier_moves(sq, piece, mode, moves),
        }
    }

    fn gen_general_moves(&self, sq: usize, piece: Piece, mode: MoveGenMode, moves: &mut Vec<Move>) {
        let file = file_of(sq) as i32;
        let rank = rank_of(sq) as i32;
        for (df, dr) in ORTHOGONAL_STEPS {
            let nf = file + df;
            let nr = rank + dr;
            if !inside_board(nf, nr) || !inside_palace(piece.color, nf as usize, nr as usize) {
                continue;
            }
            self.push_if_valid_target(sq, nf as usize, nr as usize, piece.color, mode, moves);
        }

        let enemy_king = self.find_general(piece.color.opposite()).unwrap();
        if file_of(enemy_king) == file_of(sq) && self.clear_file_between(sq, enemy_king) {
            moves.push(Move::new(sq, enemy_king));
        }
    }

    fn gen_advisor_moves(&self, sq: usize, piece: Piece, mode: MoveGenMode, moves: &mut Vec<Move>) {
        let file = file_of(sq) as i32;
        let rank = rank_of(sq) as i32;
        for (df, dr) in DIAGONAL_STEPS {
            let nf = file + df;
            let nr = rank + dr;
            if !inside_board(nf, nr) || !inside_palace(piece.color, nf as usize, nr as usize) {
                continue;
            }
            self.push_if_valid_target(sq, nf as usize, nr as usize, piece.color, mode, moves);
        }
    }

    fn gen_elephant_moves(
        &self,
        sq: usize,
        piece: Piece,
        mode: MoveGenMode,
        moves: &mut Vec<Move>,
    ) {
        let file = file_of(sq) as i32;
        let rank = rank_of(sq) as i32;
        for ((eye_df, eye_dr), (df, dr)) in ELEPHANT_STEPS {
            let eye_f = file + eye_df;
            let eye_r = rank + eye_dr;
            let nf = file + df;
            let nr = rank + dr;
            if !inside_board(nf, nr) || !inside_board(eye_f, eye_r) {
                continue;
            }
            if !elephant_stays_home(piece.color, nr as usize) {
                continue;
            }
            if self.board[index(eye_f as usize, eye_r as usize)].is_some() {
                continue;
            }
            self.push_if_valid_target(sq, nf as usize, nr as usize, piece.color, mode, moves);
        }
    }

    fn gen_horse_moves(&self, sq: usize, piece: Piece, mode: MoveGenMode, moves: &mut Vec<Move>) {
        let file = file_of(sq) as i32;
        let rank = rank_of(sq) as i32;
        for ((leg_df, leg_dr), (df, dr)) in HORSE_STEPS {
            let leg_f = file + leg_df;
            let leg_r = rank + leg_dr;
            let nf = file + df;
            let nr = rank + dr;
            if !inside_board(leg_f, leg_r) || !inside_board(nf, nr) {
                continue;
            }
            if self.board[index(leg_f as usize, leg_r as usize)].is_some() {
                continue;
            }
            self.push_if_valid_target(sq, nf as usize, nr as usize, piece.color, mode, moves);
        }
    }

    fn gen_rook_moves(&self, sq: usize, piece: Piece, mode: MoveGenMode, moves: &mut Vec<Move>) {
        self.gen_slider_moves(sq, piece.color, false, mode, moves);
    }

    fn gen_cannon_moves(&self, sq: usize, piece: Piece, mode: MoveGenMode, moves: &mut Vec<Move>) {
        self.gen_slider_moves(sq, piece.color, true, mode, moves);
    }

    fn gen_slider_moves(
        &self,
        sq: usize,
        color: Color,
        is_cannon: bool,
        mode: MoveGenMode,
        moves: &mut Vec<Move>,
    ) {
        let file = file_of(sq) as i32;
        let rank = rank_of(sq) as i32;
        for (df, dr) in ORTHOGONAL_STEPS {
            let mut nf = file + df;
            let mut nr = rank + dr;
            let mut seen_screen = false;

            while inside_board(nf, nr) {
                let target = index(nf as usize, nr as usize);
                match self.board[target] {
                    None => {
                        if mode == MoveGenMode::All && (!is_cannon || !seen_screen) {
                            moves.push(Move::new(sq, target));
                        }
                    }
                    Some(target_piece) => {
                        if !is_cannon {
                            if target_piece.color != color {
                                moves.push(Move::new(sq, target));
                            }
                            break;
                        }

                        if !seen_screen {
                            seen_screen = true;
                        } else {
                            if target_piece.color != color {
                                moves.push(Move::new(sq, target));
                            }
                            break;
                        }
                    }
                }

                nf += df;
                nr += dr;
            }
        }
    }

    fn gen_soldier_moves(&self, sq: usize, piece: Piece, mode: MoveGenMode, moves: &mut Vec<Move>) {
        let file = file_of(sq) as i32;
        let rank = rank_of(sq) as i32;
        let forward_rank = rank + piece.color.forward_step();
        if inside_board(file, forward_rank) {
            self.push_if_valid_target(
                sq,
                file as usize,
                forward_rank as usize,
                piece.color,
                mode,
                moves,
            );
        }

        if soldier_crossed_river(piece.color, rank as usize) {
            for df in [-1, 1] {
                let nf = file + df;
                if inside_board(nf, rank) {
                    self.push_if_valid_target(
                        sq,
                        nf as usize,
                        rank as usize,
                        piece.color,
                        mode,
                        moves,
                    );
                }
            }
        }
    }

    fn push_if_valid_target(
        &self,
        from: usize,
        to_file: usize,
        to_rank: usize,
        color: Color,
        mode: MoveGenMode,
        moves: &mut Vec<Move>,
    ) {
        let to = index(to_file, to_rank);
        match self.board[to] {
            Some(piece) if piece.color == color => {}
            Some(_) => moves.push(Move::new(from, to)),
            None if mode == MoveGenMode::All => moves.push(Move::new(from, to)),
            None => {}
        }
    }

    fn find_general(&self, color: Color) -> Option<usize> {
        self.general_squares[color_hash_index(color)]
    }

    fn generals_face(&self) -> bool {
        let red = self.find_general(Color::Red).unwrap();
        let black = self.find_general(Color::Black).unwrap();
        file_of(red) == file_of(black) && self.clear_file_between(red, black)
    }

    fn clear_file_between(&self, a: usize, b: usize) -> bool {
        if file_of(a) != file_of(b) {
            return false;
        }

        let file = file_of(a);
        let start = rank_of(a).min(rank_of(b)) + 1;
        let end = rank_of(a).max(rank_of(b));

        for rank in start..end {
            if self.board[index(file, rank)].is_some() {
                return false;
            }
        }
        true
    }

    fn is_square_attacked(&self, target: usize, by: Color) -> bool {
        self.is_square_attacked_by_leapers(target, by)
            || self.is_square_attacked_by_sliders(target, by)
    }

    #[cfg(test)]
    fn piece_attacks_square(&self, sq: usize, piece: Piece, target: usize) -> bool {
        match piece.kind {
            PieceKind::General => self.general_attacks_square(sq, piece, target),
            PieceKind::Advisor => self.advisor_attacks_square(sq, piece, target),
            PieceKind::Elephant => self.elephant_attacks_square(sq, piece, target),
            PieceKind::Horse => self.horse_attacks_square(sq, piece, target),
            PieceKind::Rook => self.rook_attacks_square(sq, target),
            PieceKind::Cannon => self.cannon_attacks_square(sq, target),
            PieceKind::Soldier => self.soldier_attacks_square(sq, piece, target),
        }
    }

    #[cfg(test)]
    fn general_attacks_square(&self, sq: usize, piece: Piece, target: usize) -> bool {
        let file = file_of(sq) as i32;
        let rank = rank_of(sq) as i32;
        let tf = file_of(target) as i32;
        let tr = rank_of(target) as i32;

        if (file - tf).abs() + (rank - tr).abs() == 1
            && inside_palace(piece.color, tf as usize, tr as usize)
        {
            return true;
        }

        matches!(
            self.board[target],
            Some(Piece {
                color,
                kind: PieceKind::General
            }) if color != piece.color
        ) && file_of(sq) == file_of(target)
            && self.clear_file_between(sq, target)
    }

    #[cfg(test)]
    fn advisor_attacks_square(&self, sq: usize, piece: Piece, target: usize) -> bool {
        let file = file_of(sq) as i32;
        let rank = rank_of(sq) as i32;
        let tf = file_of(target) as i32;
        let tr = rank_of(target) as i32;
        (file - tf).abs() == 1
            && (rank - tr).abs() == 1
            && inside_palace(piece.color, tf as usize, tr as usize)
    }

    #[cfg(test)]
    fn elephant_attacks_square(&self, sq: usize, piece: Piece, target: usize) -> bool {
        let file = file_of(sq) as i32;
        let rank = rank_of(sq) as i32;
        let tf = file_of(target) as i32;
        let tr = rank_of(target) as i32;
        if (file - tf).abs() != 2 || (rank - tr).abs() != 2 {
            return false;
        }
        if !elephant_stays_home(piece.color, tr as usize) {
            return false;
        }
        let eye_f = (file + tf) / 2;
        let eye_r = (rank + tr) / 2;
        self.board[index(eye_f as usize, eye_r as usize)].is_none()
    }

    #[cfg(test)]
    fn horse_attacks_square(&self, sq: usize, _piece: Piece, target: usize) -> bool {
        let file = file_of(sq) as i32;
        let rank = rank_of(sq) as i32;
        let tf = file_of(target) as i32;
        let tr = rank_of(target) as i32;
        let df = tf - file;
        let dr = tr - rank;

        for ((leg_df, leg_dr), (move_df, move_dr)) in HORSE_STEPS {
            if df == move_df && dr == move_dr {
                let leg_f = file + leg_df;
                let leg_r = rank + leg_dr;
                return self.board[index(leg_f as usize, leg_r as usize)].is_none();
            }
        }
        false
    }

    #[cfg(test)]
    fn rook_attacks_square(&self, sq: usize, target: usize) -> bool {
        if !same_rank_or_file(sq, target) {
            return false;
        }
        self.clear_line_between(sq, target)
    }

    #[cfg(test)]
    fn cannon_attacks_square(&self, sq: usize, target: usize) -> bool {
        if !same_rank_or_file(sq, target) {
            return false;
        }
        self.count_between(sq, target) == 1
    }

    #[cfg(test)]
    fn soldier_attacks_square(&self, sq: usize, piece: Piece, target: usize) -> bool {
        let file = file_of(sq) as i32;
        let rank = rank_of(sq) as i32;
        let tf = file_of(target) as i32;
        let tr = rank_of(target) as i32;

        if tf == file && tr == rank + piece.color.forward_step() {
            return true;
        }

        soldier_crossed_river(piece.color, rank as usize) && tr == rank && (tf - file).abs() == 1
    }

    #[cfg(test)]
    fn clear_line_between(&self, a: usize, b: usize) -> bool {
        self.count_between(a, b) == 0
    }

    #[cfg(test)]
    fn count_between(&self, a: usize, b: usize) -> usize {
        if file_of(a) == file_of(b) {
            let file = file_of(a);
            let start = rank_of(a).min(rank_of(b)) + 1;
            let end = rank_of(a).max(rank_of(b));
            (start..end)
                .filter(|rank| self.board[index(file, *rank)].is_some())
                .count()
        } else if rank_of(a) == rank_of(b) {
            let rank = rank_of(a);
            let start = file_of(a).min(file_of(b)) + 1;
            let end = file_of(a).max(file_of(b));
            (start..end)
                .filter(|file| self.board[index(*file, rank)].is_some())
                .count()
        } else {
            usize::MAX
        }
    }

    fn checkers_to(&self, target: usize, by: Color) -> Vec<CheckerInfo> {
        let mut checkers = Vec::with_capacity(2);
        self.visit_attacker_origins_to(target, by, |sq| {
            let Some(piece) = self.board[sq] else {
                return false;
            };
            checkers.push(CheckerInfo {
                from: sq,
                kind: piece.kind,
                screen_square: (piece.kind == PieceKind::Cannon)
                    .then(|| self.single_screen_square_between(sq, target))
                    .flatten(),
                block_square: (piece.kind == PieceKind::Horse)
                    .then(|| horse_leg_square(sq, target))
                    .flatten(),
            });
            false
        });
        checkers
    }

    fn interposition_squares(&self, king_sq: usize, checker: &CheckerInfo) -> Vec<usize> {
        match checker.kind {
            PieceKind::Rook | PieceKind::General | PieceKind::Cannon => {
                line_between_squares(checker.from, king_sq)
            }
            PieceKind::Horse => checker.block_square.into_iter().collect(),
            _ => Vec::new(),
        }
    }

    fn single_screen_square_between(&self, a: usize, b: usize) -> Option<usize> {
        let mut screen = None;
        if file_of(a) == file_of(b) {
            let file = file_of(a);
            let start = rank_of(a).min(rank_of(b)) + 1;
            let end = rank_of(a).max(rank_of(b));
            for rank in start..end {
                let sq = index(file, rank);
                if self.board[sq].is_some() {
                    if screen.is_some() {
                        return None;
                    }
                    screen = Some(sq);
                }
            }
        } else if rank_of(a) == rank_of(b) {
            let rank = rank_of(a);
            let start = file_of(a).min(file_of(b)) + 1;
            let end = file_of(a).max(file_of(b));
            for file in start..end {
                let sq = index(file, rank);
                if self.board[sq].is_some() {
                    if screen.is_some() {
                        return None;
                    }
                    screen = Some(sq);
                }
            }
        }
        screen
    }

    pub(super) fn visit_attacker_origins_to<F>(
        &self,
        target: usize,
        by: Color,
        mut visitor: F,
    ) -> bool
    where
        F: FnMut(usize) -> bool,
    {
        self.visit_leaper_attackers(target, by, &mut visitor)
            || self.visit_slider_attackers(target, by, &mut visitor)
    }

    fn visit_leaper_attackers<F>(&self, target: usize, by: Color, visitor: &mut F) -> bool
    where
        F: FnMut(usize) -> bool,
    {
        let file = file_of(target) as i32;
        let rank = rank_of(target) as i32;

        for (df, dr) in ORTHOGONAL_STEPS {
            let from_file = file - df;
            let from_rank = rank - dr;
            if !inside_board(from_file, from_rank) {
                continue;
            }
            let from = index(from_file as usize, from_rank as usize);
            if matches!(
                self.board[from],
                Some(Piece {
                    color,
                    kind: PieceKind::General
                }) if color == by
                    && inside_palace(by, file as usize, rank as usize)
                    && inside_palace(by, from_file as usize, from_rank as usize)
            ) && visitor(from)
            {
                return true;
            }
        }

        for (df, dr) in DIAGONAL_STEPS {
            let from_file = file - df;
            let from_rank = rank - dr;
            if !inside_board(from_file, from_rank) {
                continue;
            }
            let from = index(from_file as usize, from_rank as usize);
            if matches!(
                self.board[from],
                Some(Piece {
                    color,
                    kind: PieceKind::Advisor
                }) if color == by
                    && inside_palace(by, file as usize, rank as usize)
                    && inside_palace(by, from_file as usize, from_rank as usize)
            ) && visitor(from)
            {
                return true;
            }
        }

        for ((leg_df, leg_dr), (move_df, move_dr)) in HORSE_STEPS {
            let from_file = file - move_df;
            let from_rank = rank - move_dr;
            if !inside_board(from_file, from_rank) {
                continue;
            }
            let leg_file = from_file + leg_df;
            let leg_rank = from_rank + leg_dr;
            if !inside_board(leg_file, leg_rank) {
                continue;
            }
            let from = index(from_file as usize, from_rank as usize);
            let leg = index(leg_file as usize, leg_rank as usize);
            if self.board[leg].is_none()
                && matches!(
                    self.board[from],
                    Some(Piece {
                        color,
                        kind: PieceKind::Horse
                    }) if color == by
                )
                && visitor(from)
            {
                return true;
            }
        }

        for ((eye_df, eye_dr), (move_df, move_dr)) in ELEPHANT_STEPS {
            let from_file = file - move_df;
            let from_rank = rank - move_dr;
            if !inside_board(from_file, from_rank) {
                continue;
            }
            let eye_file = from_file + eye_df;
            let eye_rank = from_rank + eye_dr;
            if !inside_board(eye_file, eye_rank) {
                continue;
            }
            let from = index(from_file as usize, from_rank as usize);
            let eye = index(eye_file as usize, eye_rank as usize);
            if self.board[eye].is_none()
                && matches!(
                    self.board[from],
                    Some(Piece {
                        color,
                        kind: PieceKind::Elephant
                    }) if color == by && elephant_stays_home(by, rank as usize)
                )
                && visitor(from)
            {
                return true;
            }
        }

        let soldier_forward_from_rank = rank - by.forward_step();
        if inside_board(file, soldier_forward_from_rank) {
            let from = index(file as usize, soldier_forward_from_rank as usize);
            if matches!(
                self.board[from],
                Some(Piece {
                    color,
                    kind: PieceKind::Soldier
                }) if color == by
            ) && visitor(from)
            {
                return true;
            }
        }

        for side_df in [-1, 1] {
            let from_file = file - side_df;
            if !inside_board(from_file, rank) {
                continue;
            }
            let from = index(from_file as usize, rank as usize);
            if matches!(
                self.board[from],
                Some(Piece {
                    color,
                    kind: PieceKind::Soldier
                }) if color == by && soldier_crossed_river(by, rank as usize)
            ) && visitor(from)
            {
                return true;
            }
        }

        false
    }

    fn visit_slider_attackers<F>(&self, target: usize, by: Color, visitor: &mut F) -> bool
    where
        F: FnMut(usize) -> bool,
    {
        let file = file_of(target) as i32;
        let rank = rank_of(target) as i32;

        for (df, dr) in ORTHOGONAL_STEPS {
            let mut seen_screen = false;
            let mut nf = file + df;
            let mut nr = rank + dr;

            while inside_board(nf, nr) {
                let sq = index(nf as usize, nr as usize);
                if let Some(piece) = self.board[sq] {
                    if !seen_screen {
                        if piece.color == by {
                            if piece.kind == PieceKind::Rook && visitor(sq) {
                                return true;
                            }
                            if piece.kind == PieceKind::General
                                && df == 0
                                && matches!(
                                    self.board[target],
                                    Some(Piece {
                                        color,
                                        kind: PieceKind::General
                                    }) if color == by.opposite()
                                )
                                && visitor(sq)
                            {
                                return true;
                            }
                        }
                        seen_screen = true;
                    } else {
                        if piece.color == by && piece.kind == PieceKind::Cannon && visitor(sq) {
                            return true;
                        }
                        break;
                    }
                }

                nf += df;
                nr += dr;
            }
        }

        false
    }

    fn is_square_attacked_by_leapers(&self, target: usize, by: Color) -> bool {
        let file = file_of(target) as i32;
        let rank = rank_of(target) as i32;

        for (df, dr) in ORTHOGONAL_STEPS {
            let from_file = file - df;
            let from_rank = rank - dr;
            if !inside_board(from_file, from_rank) {
                continue;
            }
            let from = index(from_file as usize, from_rank as usize);
            if matches!(
                self.board[from],
                Some(Piece {
                    color,
                    kind: PieceKind::General
                }) if color == by
                    && inside_palace(by, file as usize, rank as usize)
                    && inside_palace(by, from_file as usize, from_rank as usize)
            ) {
                return true;
            }
        }

        for (df, dr) in DIAGONAL_STEPS {
            let from_file = file - df;
            let from_rank = rank - dr;
            if !inside_board(from_file, from_rank) {
                continue;
            }
            let from = index(from_file as usize, from_rank as usize);
            if matches!(
                self.board[from],
                Some(Piece {
                    color,
                    kind: PieceKind::Advisor
                }) if color == by
                    && inside_palace(by, file as usize, rank as usize)
                    && inside_palace(by, from_file as usize, from_rank as usize)
            ) {
                return true;
            }
        }

        for ((leg_df, leg_dr), (move_df, move_dr)) in HORSE_STEPS {
            let from_file = file - move_df;
            let from_rank = rank - move_dr;
            if !inside_board(from_file, from_rank) {
                continue;
            }
            let leg_file = from_file + leg_df;
            let leg_rank = from_rank + leg_dr;
            if !inside_board(leg_file, leg_rank) {
                continue;
            }
            let from = index(from_file as usize, from_rank as usize);
            let leg = index(leg_file as usize, leg_rank as usize);
            if self.board[leg].is_none()
                && matches!(
                    self.board[from],
                    Some(Piece {
                        color,
                        kind: PieceKind::Horse
                    }) if color == by
                )
            {
                return true;
            }
        }

        for ((eye_df, eye_dr), (move_df, move_dr)) in ELEPHANT_STEPS {
            let from_file = file - move_df;
            let from_rank = rank - move_dr;
            if !inside_board(from_file, from_rank) {
                continue;
            }
            let eye_file = from_file + eye_df;
            let eye_rank = from_rank + eye_dr;
            if !inside_board(eye_file, eye_rank) {
                continue;
            }
            let from = index(from_file as usize, from_rank as usize);
            let eye = index(eye_file as usize, eye_rank as usize);
            if self.board[eye].is_none()
                && matches!(
                    self.board[from],
                    Some(Piece {
                        color,
                        kind: PieceKind::Elephant
                    }) if color == by && elephant_stays_home(by, rank as usize)
                )
            {
                return true;
            }
        }

        let soldier_forward_from_rank = rank - by.forward_step();
        if inside_board(file, soldier_forward_from_rank) {
            let from = index(file as usize, soldier_forward_from_rank as usize);
            if matches!(
                self.board[from],
                Some(Piece {
                    color,
                    kind: PieceKind::Soldier
                }) if color == by
            ) {
                return true;
            }
        }

        for side_df in [-1, 1] {
            let from_file = file - side_df;
            if !inside_board(from_file, rank) {
                continue;
            }
            let from = index(from_file as usize, rank as usize);
            if matches!(
                self.board[from],
                Some(Piece {
                    color,
                    kind: PieceKind::Soldier
                }) if color == by && soldier_crossed_river(by, rank as usize)
            ) {
                return true;
            }
        }

        false
    }

    fn is_square_attacked_by_sliders(&self, target: usize, by: Color) -> bool {
        let file = file_of(target) as i32;
        let rank = rank_of(target) as i32;

        for (df, dr) in ORTHOGONAL_STEPS {
            let mut seen_screen = false;
            let mut nf = file + df;
            let mut nr = rank + dr;

            while inside_board(nf, nr) {
                let sq = index(nf as usize, nr as usize);
                if let Some(piece) = self.board[sq] {
                    if !seen_screen {
                        if piece.color == by {
                            if piece.kind == PieceKind::Rook {
                                return true;
                            }
                            if piece.kind == PieceKind::General
                                && df == 0
                                && matches!(
                                    self.board[target],
                                    Some(Piece {
                                        color,
                                        kind: PieceKind::General
                                    }) if color == by.opposite()
                                )
                            {
                                return true;
                            }
                        }
                        seen_screen = true;
                    } else if piece.color == by && piece.kind == PieceKind::Cannon {
                        return true;
                    } else {
                        break;
                    }
                }

                nf += df;
                nr += dr;
            }
        }

        false
    }

    #[cfg(test)]
    #[cfg(test)]
    fn is_square_attacked_slow(&self, target: usize, by: Color) -> bool {
        for sq in 0..BOARD_SIZE {
            let Some(piece) = self.board[sq] else {
                continue;
            };
            if piece.color != by {
                continue;
            }
            if self.piece_attacks_square(sq, piece, target) {
                return true;
            }
        }
        false
    }
}

const ORTHOGONAL_STEPS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const DIAGONAL_STEPS: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
const ELEPHANT_STEPS: [((i32, i32), (i32, i32)); 4] = [
    ((1, 1), (2, 2)),
    ((1, -1), (2, -2)),
    ((-1, 1), (-2, 2)),
    ((-1, -1), (-2, -2)),
];
const HORSE_STEPS: [((i32, i32), (i32, i32)); 8] = [
    ((0, -1), (-1, -2)),
    ((0, -1), (1, -2)),
    ((0, 1), (-1, 2)),
    ((0, 1), (1, 2)),
    ((-1, 0), (-2, -1)),
    ((-1, 0), (-2, 1)),
    ((1, 0), (2, -1)),
    ((1, 0), (2, 1)),
];

#[cfg(test)]
mod tests;
