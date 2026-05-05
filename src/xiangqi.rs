use std::fmt;

pub const BOARD_FILES: usize = 9;
pub const BOARD_RANKS: usize = 10;
pub const BOARD_SIZE: usize = BOARD_FILES * BOARD_RANKS;
pub const STARTPOS_FEN: &str = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Color {
    Red,
    Black,
}

impl Color {
    #[inline(always)]
    pub fn opposite(self) -> Self {
        match self {
            Self::Red => Self::Black,
            Self::Black => Self::Red,
        }
    }

    #[inline(always)]
    pub fn forward_step(self) -> i32 {
        match self {
            Self::Red => -1,
            Self::Black => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PieceKind {
    General,
    Advisor,
    Elephant,
    Horse,
    Rook,
    Cannon,
    Soldier,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MoveGenMode {
    All,
    Captures,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CheckerInfo {
    from: usize,
    kind: PieceKind,
    screen_square: Option<usize>,
    block_square: Option<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Piece {
    pub color: Color,
    pub kind: PieceKind,
}

impl Piece {
    fn from_fen(ch: char) -> Option<Self> {
        let color = if ch.is_ascii_uppercase() {
            Color::Red
        } else {
            Color::Black
        };

        let kind = match ch.to_ascii_lowercase() {
            'k' => PieceKind::General,
            'a' => PieceKind::Advisor,
            'b' | 'e' => PieceKind::Elephant,
            'n' | 'h' => PieceKind::Horse,
            'r' => PieceKind::Rook,
            'c' => PieceKind::Cannon,
            'p' => PieceKind::Soldier,
            _ => return None,
        };

        Some(Self { color, kind })
    }

    fn to_fen(self) -> char {
        let ch = match self.kind {
            PieceKind::General => 'k',
            PieceKind::Advisor => 'a',
            PieceKind::Elephant => 'b',
            PieceKind::Horse => 'n',
            PieceKind::Rook => 'r',
            PieceKind::Cannon => 'c',
            PieceKind::Soldier => 'p',
        };

        match self.color {
            Color::Red => ch.to_ascii_uppercase(),
            Color::Black => ch,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Move {
    pub from: u8,
    pub to: u8,
}

impl Move {
    #[inline(always)]
    pub fn new(from: usize, to: usize) -> Self {
        Self {
            from: from as u8,
            to: to as u8,
        }
    }

    pub fn from_uci(uci: &str) -> Option<Self> {
        if uci.len() != 4 {
            return None;
        }
        let from = parse_square(&uci[0..2])?;
        let to = parse_square(&uci[2..4])?;
        Some(Self::new(from, to))
    }

    pub fn to_uci(self) -> String {
        format!(
            "{}{}",
            square_name(self.from as usize),
            square_name(self.to as usize)
        )
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_uci())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Undo {
    captured: Option<Piece>,
    side_to_move: Color,
    halfmove_clock: u16,
    check_in_no_capture: u16,
}

#[derive(Clone, Copy, Debug)]
pub struct NullUndo {
    side_to_move: Color,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuleHistoryEntry {
    pub hash: u64,
    pub side_to_move: Color,
    pub mover: Option<Color>,
    pub is_capture: bool,
    pub gives_check: bool,
    pub checking_pieces: u8,
    pub chased_mask: u128,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuleDrawReason {
    Halfmove120,
    Repetition,
    MutualLongCheck,
    MutualLongChase,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuleOutcome {
    Draw(RuleDrawReason),
    Win(Color),
}

const PERPETUAL_CHECK_LIMIT_1: u16 = 6;
const PERPETUAL_CHECK_LIMIT_2: u16 = 12;
const PERPETUAL_CHECK_LIMIT_3: u16 = 18;
const PERPETUAL_CHASE_LIMIT: u16 = 6;
const CHECK_CHASE_ALT_LIMIT_1: u16 = 12;
const CHECK_CHASE_ALT_LIMIT_N: u16 = 18;

#[derive(Clone, Copy, Debug, Default)]
struct RuleCounters {
    check: [u16; 2],
    chase: [u16; 2],
    alt: [u16; 2],
    max_check_pieces: [u8; 2],
}

impl RuleCounters {
    fn apply(&mut self, entry: RuleHistoryEntry, mover: Color) {
        if entry.is_capture {
            *self = Self::default();
            return;
        }

        let index = color_hash_index(mover);
        if entry.gives_check {
            self.check[index] = self.check[index].saturating_add(1);
            self.max_check_pieces[index] = self.max_check_pieces[index].max(entry.checking_pieces);
        } else {
            self.check[index] = 0;
            self.max_check_pieces[index] = 0;
        }

        let gives_chase = entry.chased_mask != 0 && !entry.gives_check;
        if gives_chase {
            self.chase[index] = self.chase[index].saturating_add(1);
        } else {
            self.chase[index] = 0;
        }

        if entry.gives_check || gives_chase {
            self.alt[index] = self.alt[index].saturating_add(1);
        } else {
            self.alt[index] = 0;
        }
    }

    fn violation_winner(&self, mover: Color) -> Option<Color> {
        let index = color_hash_index(mover);
        let check_limit = perpetual_check_limit(self.max_check_pieces[index]);
        let alt_limit = check_chase_alt_limit(self.max_check_pieces[index]);
        let violates = self.check[index] >= check_limit
            || self.chase[index] >= PERPETUAL_CHASE_LIMIT
            || (self.alt[index] >= alt_limit && (self.check[index] > 0 || self.chase[index] > 0));
        violates.then_some(mover.opposite())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Position {
    board: [Option<Piece>; BOARD_SIZE],
    side_to_move: Color,
    hash: u64,
    base_eval: i32,
    advisor_counts: [u8; 2],
    elephant_counts: [u8; 2],
    dynamic_material_counts: [u8; 2],
    general_squares: [Option<usize>; 2],
    halfmove_clock: u16,
    check_in_no_capture: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PositionState {
    hash: u64,
    base_eval: i32,
    advisor_counts: [u8; 2],
    elephant_counts: [u8; 2],
    dynamic_material_counts: [u8; 2],
    general_squares: [Option<usize>; 2],
}

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
            check_in_no_capture: 0,
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

    #[inline(always)]
    pub fn has_dynamic_material(&self, color: Color) -> bool {
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

    pub fn check_in_no_capture(&self) -> u16 {
        self.check_in_no_capture
    }

    fn effective_no_capture_clock(&self) -> u16 {
        self.halfmove_clock
            .saturating_sub(self.check_in_no_capture.saturating_sub(40))
    }

    pub fn is_piece_protected(&self, sq: usize, color: Color) -> bool {
        self.visit_attacker_origins_to(sq, color, |from| from != sq)
    }

    pub fn piece_attacks_square_from(&self, from: usize, target: usize) -> bool {
        let Some(piece) = self.board[from] else {
            return false;
        };
        self.piece_attacks_square(from, piece, target)
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        self.collect_legal_moves(false, self.in_check(self.side_to_move))
    }

    pub fn legal_capture_moves(&self) -> Vec<Move> {
        self.collect_legal_moves(true, self.in_check(self.side_to_move))
    }

    pub fn legal_moves_with_check_hint(&self, in_check: bool) -> Vec<Move> {
        self.collect_legal_moves(false, in_check)
    }

    pub fn legal_capture_moves_with_check_hint(&self, in_check: bool) -> Vec<Move> {
        self.collect_legal_moves(true, in_check)
    }

    pub fn legal_capture_moves_to(&self, target: usize) -> Vec<Move> {
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

    pub fn initial_rule_history(&self) -> Vec<RuleHistoryEntry> {
        vec![self.rule_history_entry(None)]
    }

    pub fn rule_history_entry(&self, mover: Option<Color>) -> RuleHistoryEntry {
        let has_both_generals = self.has_general(Color::Red) && self.has_general(Color::Black);
        RuleHistoryEntry {
            hash: self.hash,
            side_to_move: self.side_to_move,
            mover,
            is_capture: false,
            gives_check: has_both_generals && self.in_check(self.side_to_move),
            checking_pieces: 0,
            chased_mask: 0,
        }
    }

    pub fn rule_history_entry_after_move(&self, mv: Move) -> RuleHistoryEntry {
        let mut next = self.clone();
        let mover = self.side_to_move;
        let is_capture = self.board[mv.to as usize].is_some();
        next.make_move(mv);
        let has_both_generals = next.has_general(Color::Red) && next.has_general(Color::Black);
        let gives_check = has_both_generals && next.in_check(next.side_to_move);
        let checking_pieces = if gives_check {
            next.checkers_to(
                next.find_general(next.side_to_move)
                    .expect("checked side must have a general"),
                mover,
            )
            .len()
            .min(u8::MAX as usize) as u8
        } else {
            0
        };
        RuleHistoryEntry {
            hash: next.hash,
            side_to_move: next.side_to_move,
            mover: Some(mover),
            is_capture,
            gives_check,
            checking_pieces,
            chased_mask: if has_both_generals && !gives_check {
                self.chased_mask_after_move(mv, &next)
            } else {
                0
            },
        }
    }

    pub fn rule_outcome_with_history(&self, history: &[RuleHistoryEntry]) -> Option<RuleOutcome> {
        if self.effective_no_capture_clock() >= 120 {
            return Some(RuleOutcome::Draw(RuleDrawReason::Halfmove120));
        }
        Self::rule_outcome(history)
    }

    pub fn rule_outcome(history: &[RuleHistoryEntry]) -> Option<RuleOutcome> {
        let current_index = history.len().checked_sub(1)?;
        let current = history[current_index];
        if let Some(outcome) = Self::rule_violation_outcome(history) {
            return Some(outcome);
        }
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

        for &start_index in repeated_indices.iter().rev() {
            if let Some(outcome) = Self::rule_outcome_for_cycle(history, start_index, current_index)
            {
                return Some(outcome);
            }
        }

        (repeated_indices.len() >= 4).then_some(RuleOutcome::Draw(RuleDrawReason::Repetition))
    }

    fn rule_violation_outcome(history: &[RuleHistoryEntry]) -> Option<RuleOutcome> {
        let mut counters = RuleCounters::default();
        for entry in history.iter().copied() {
            let Some(mover) = entry.mover else {
                continue;
            };
            counters.apply(entry, mover);
            if let Some(winner) = counters.violation_winner(mover) {
                return Some(RuleOutcome::Win(winner));
            }
        }
        None
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

    pub fn perft(&self, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }

        let moves = self.legal_moves();
        if depth == 1 {
            return moves.len() as u64;
        }

        let mut nodes = 0u64;
        for mv in moves {
            let mut next = self.clone();
            next.make_move(mv);
            nodes += next.perft(depth - 1);
        }
        nodes
    }

    pub fn make_move(&mut self, mv: Move) -> Undo {
        let from = mv.from as usize;
        let to = mv.to as usize;
        let moving = self.board[from].expect("move from occupied square");
        let undo = Undo {
            captured: self.board[to],
            side_to_move: self.side_to_move,
            halfmove_clock: self.halfmove_clock,
            check_in_no_capture: self.check_in_no_capture,
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
        let gives_check = self.has_general(Color::Red)
            && self.has_general(Color::Black)
            && self.in_check(self.side_to_move);
        self.halfmove_clock = if undo.captured.is_some() {
            0
        } else {
            self.halfmove_clock.saturating_add(1)
        };
        self.check_in_no_capture = if undo.captured.is_some() {
            0
        } else if gives_check {
            self.check_in_no_capture.saturating_add(1)
        } else {
            self.check_in_no_capture
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
        self.check_in_no_capture = undo.check_in_no_capture;
    }

    pub fn make_null_move(&mut self) -> NullUndo {
        let undo = NullUndo {
            side_to_move: self.side_to_move,
        };
        self.side_to_move = self.side_to_move.opposite();
        self.hash ^= SIDE_TO_MOVE_KEY;
        undo
    }

    pub fn unmake_null_move(&mut self, undo: NullUndo) {
        self.side_to_move = undo.side_to_move;
        self.hash ^= SIDE_TO_MOVE_KEY;
    }

    pub fn in_check(&self, color: Color) -> bool {
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

    fn rule_outcome_for_cycle(
        history: &[RuleHistoryEntry],
        start_index: usize,
        end_index: usize,
    ) -> Option<RuleOutcome> {
        if end_index <= start_index + 1 {
            return None;
        }

        let mut has_turn = [false; 2];
        let mut all_checks = [true; 2];
        let mut all_chases = [true; 2];
        let mut chase_masks = [u128::MAX; 2];

        for entry in &history[start_index + 1..=end_index] {
            let Some(mover) = entry.mover else {
                continue;
            };
            let mover_index = color_hash_index(mover);
            has_turn[mover_index] = true;
            all_checks[mover_index] &= entry.gives_check;
            if entry.chased_mask == 0 {
                all_chases[mover_index] = false;
                chase_masks[mover_index] = 0;
            } else {
                chase_masks[mover_index] &= entry.chased_mask;
            }
        }

        let long_check = [has_turn[0] && all_checks[0], has_turn[1] && all_checks[1]];
        match (long_check[0], long_check[1]) {
            (true, false) => return Some(RuleOutcome::Win(Color::Black)),
            (false, true) => return Some(RuleOutcome::Win(Color::Red)),
            (true, true) => {
                let violator = history[end_index].mover?;
                return Some(RuleOutcome::Win(violator.opposite()));
            }
            (false, false) => {}
        }

        let long_chase = [
            has_turn[0] && all_chases[0] && chase_masks[0] != 0,
            has_turn[1] && all_chases[1] && chase_masks[1] != 0,
        ];
        match (long_chase[0], long_chase[1]) {
            (true, false) => Some(RuleOutcome::Win(Color::Black)),
            (false, true) => Some(RuleOutcome::Win(Color::Red)),
            (true, true) => {
                let violator = history[end_index].mover?;
                Some(RuleOutcome::Win(violator.opposite()))
            }
            (false, false) => None,
        }
    }

    fn chased_mask_after_move(&self, mv: Move, after: &Position) -> u128 {
        let from = mv.from as usize;
        let to = mv.to as usize;
        let Some(attacker) = after.board[to] else {
            return 0;
        };
        if !is_long_chase_attacker(attacker.kind) || after.is_piece_pinned(to, attacker.color) {
            return 0;
        }

        let defender = attacker.color.opposite();
        let mut mask = 0u128;
        let mut threat_count = 0usize;

        for target in 0..BOARD_SIZE {
            let Some(target_piece) = after.board[target] else {
                continue;
            };
            if target_piece.color != defender || !is_long_chase_target(target_piece.kind) {
                continue;
            }
            if !after.piece_attacks_square(to, attacker, target) {
                continue;
            }
            if self
                .board
                .get(target)
                .and_then(|piece| *piece)
                .is_some_and(|before_target| before_target.color == defender)
                && self
                    .board
                    .get(from)
                    .and_then(|piece| *piece)
                    .is_some_and(|before_attacker| {
                        self.piece_attacks_square(from, before_attacker, target)
                    })
            {
                continue;
            }
            if target_piece.kind == PieceKind::Soldier
                && !soldier_crossed_river(target_piece.color, rank_of(target))
            {
                continue;
            }
            if attacker.kind == target_piece.kind
                && after.piece_attacks_square(target, target_piece, to)
                && !after.is_piece_pinned(target, target_piece.color)
            {
                continue;
            }
            let special_rook_chase = target_piece.kind == PieceKind::Rook
                && !matches!(attacker.kind, PieceKind::Rook | PieceKind::Soldier);
            if special_rook_chase || !after.has_real_protector(target, to, defender) {
                threat_count += 1;
                mask |= 1u128 << target;
            }
        }

        if threat_count == 1 { mask } else { 0 }
    }

    fn is_piece_pinned(&self, sq: usize, color: Color) -> bool {
        if !self.board[sq].is_some_and(|piece| piece.color == color) {
            return false;
        }
        let mut work = self.clone();
        work.board[sq] = None;
        work.in_check(color)
    }

    fn has_real_protector(&self, target: usize, attacker_sq: usize, defender: Color) -> bool {
        for protector_sq in 0..BOARD_SIZE {
            if protector_sq == target {
                continue;
            }
            let Some(protector) = self.board[protector_sq] else {
                continue;
            };
            if protector.color != defender
                || !self.piece_attacks_square(protector_sq, protector, target)
            {
                continue;
            }
            if self.is_piece_pinned(protector_sq, defender) {
                continue;
            }
            let Some(attacker) = self.board[attacker_sq] else {
                continue;
            };
            let mut after_capture = self.clone();
            after_capture.board[target] = Some(attacker);
            after_capture.board[attacker_sq] = None;
            if !after_capture.piece_attacks_square(protector_sq, protector, target) {
                continue;
            }
            let mut after_recapture = after_capture;
            after_recapture.board[target] = Some(protector);
            after_recapture.board[protector_sq] = None;
            if !after_recapture.in_check(defender) {
                return true;
            }
        }
        false
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

        // Facing generals is an attack/check relation, not a capturable king move.
        // Game end is decided by the opponent having no legal reply after the move.
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
                            if target_piece.color != color
                                && target_piece.kind != PieceKind::General
                            {
                                moves.push(Move::new(sq, target));
                            }
                            break;
                        }

                        if !seen_screen {
                            seen_screen = true;
                        } else {
                            if target_piece.color != color
                                && target_piece.kind != PieceKind::General
                            {
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
            Some(piece) if piece.kind == PieceKind::General => {}
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

    fn advisor_attacks_square(&self, sq: usize, piece: Piece, target: usize) -> bool {
        let file = file_of(sq) as i32;
        let rank = rank_of(sq) as i32;
        let tf = file_of(target) as i32;
        let tr = rank_of(target) as i32;
        (file - tf).abs() == 1
            && (rank - tr).abs() == 1
            && inside_palace(piece.color, tf as usize, tr as usize)
    }

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

    fn rook_attacks_square(&self, sq: usize, target: usize) -> bool {
        if !same_rank_or_file(sq, target) {
            return false;
        }
        self.clear_line_between(sq, target)
    }

    fn cannon_attacks_square(&self, sq: usize, target: usize) -> bool {
        if !same_rank_or_file(sq, target) {
            return false;
        }
        self.count_between(sq, target) == 1
    }

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

    fn clear_line_between(&self, a: usize, b: usize) -> bool {
        self.count_between(a, b) == 0
    }

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

    fn visit_attacker_origins_to<F>(&self, target: usize, by: Color, mut visitor: F) -> bool
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

pub fn square_name(sq: usize) -> String {
    let file = (b'a' + file_of(sq) as u8) as char;
    let rank = (BOARD_RANKS - 1 - rank_of(sq)).to_string();
    format!("{file}{rank}")
}

pub fn parse_square(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.len() != 2 {
        return None;
    }

    let file = bytes[0].to_ascii_lowercase();
    let rank = bytes[1];
    if !(b'a'..=b'i').contains(&file) || !rank.is_ascii_digit() {
        return None;
    }

    let file_idx = (file - b'a') as usize;
    let rank_from_bottom = (rank - b'0') as usize;
    let rank_idx = BOARD_RANKS - 1 - rank_from_bottom;
    Some(index(file_idx, rank_idx))
}

#[inline(always)]
fn index(file: usize, rank: usize) -> usize {
    rank * BOARD_FILES + file
}

#[inline(always)]
fn file_of(sq: usize) -> usize {
    sq % BOARD_FILES
}

#[inline(always)]
fn rank_of(sq: usize) -> usize {
    sq / BOARD_FILES
}

#[inline(always)]
fn inside_board(file: i32, rank: i32) -> bool {
    (0..BOARD_FILES as i32).contains(&file) && (0..BOARD_RANKS as i32).contains(&rank)
}

fn inside_palace(color: Color, file: usize, rank: usize) -> bool {
    let file_ok = (3..=5).contains(&file);
    let rank_ok = match color {
        Color::Black => (0..=2).contains(&rank),
        Color::Red => (7..=9).contains(&rank),
    };
    file_ok && rank_ok
}

fn elephant_stays_home(color: Color, rank: usize) -> bool {
    match color {
        Color::Black => rank <= 4,
        Color::Red => rank >= 5,
    }
}

fn soldier_crossed_river(color: Color, rank: usize) -> bool {
    match color {
        Color::Black => rank >= 5,
        Color::Red => rank <= 4,
    }
}

#[inline(always)]
fn same_rank_or_file(a: usize, b: usize) -> bool {
    file_of(a) == file_of(b) || rank_of(a) == rank_of(b)
}

fn line_between_squares(a: usize, b: usize) -> Vec<usize> {
    let mut squares = Vec::new();
    if file_of(a) == file_of(b) {
        let file = file_of(a);
        let start = rank_of(a).min(rank_of(b)) + 1;
        let end = rank_of(a).max(rank_of(b));
        for rank in start..end {
            squares.push(index(file, rank));
        }
    } else if rank_of(a) == rank_of(b) {
        let rank = rank_of(a);
        let start = file_of(a).min(file_of(b)) + 1;
        let end = file_of(a).max(file_of(b));
        for file in start..end {
            squares.push(index(file, rank));
        }
    }
    squares
}

fn horse_leg_square(from: usize, target: usize) -> Option<usize> {
    let file = file_of(from) as i32;
    let rank = rank_of(from) as i32;
    let tf = file_of(target) as i32;
    let tr = rank_of(target) as i32;
    let df = tf - file;
    let dr = tr - rank;

    for ((leg_df, leg_dr), (move_df, move_dr)) in HORSE_STEPS {
        if df == move_df && dr == move_dr {
            let leg_file = file + leg_df;
            let leg_rank = rank + leg_dr;
            if inside_board(leg_file, leg_rank) {
                return Some(index(leg_file as usize, leg_rank as usize));
            }
        }
    }
    None
}

#[inline(always)]
fn is_long_chase_attacker(kind: PieceKind) -> bool {
    !matches!(kind, PieceKind::General | PieceKind::Soldier)
}

#[inline(always)]
fn is_long_chase_target(kind: PieceKind) -> bool {
    !matches!(kind, PieceKind::General | PieceKind::Soldier)
}

#[inline(always)]
fn perpetual_check_limit(num_pieces: u8) -> u16 {
    match num_pieces {
        0 | 1 => PERPETUAL_CHECK_LIMIT_1,
        2 => PERPETUAL_CHECK_LIMIT_2,
        _ => PERPETUAL_CHECK_LIMIT_3,
    }
}

#[inline(always)]
fn check_chase_alt_limit(num_pieces: u8) -> u16 {
    if num_pieces <= 1 {
        CHECK_CHASE_ALT_LIMIT_1
    } else {
        CHECK_CHASE_ALT_LIMIT_N
    }
}

#[inline(always)]
fn signed_piece_contrib(piece: Piece, sq: usize) -> i32 {
    SIGNED_PIECE_CONTRIB_TABLE[color_index_const(piece.color)][piece_kind_index(piece.kind)][sq]
}

const PIECE_BASE_VALUES: [i32; 7] = [0, 110, 110, 420, 900, 460, 110];
const PIECE_SQUARE_TABLE: [[[i32; BOARD_SIZE]; 7]; 2] = build_piece_square_table();
const SIGNED_PIECE_CONTRIB_TABLE: [[[i32; BOARD_SIZE]; 7]; 2] = build_signed_piece_contrib_table();

const fn build_piece_square_table() -> [[[i32; BOARD_SIZE]; 7]; 2] {
    let mut table = [[[0; BOARD_SIZE]; 7]; 2];
    let mut color = 0usize;
    while color < 2 {
        let mut kind = 0usize;
        while kind < 7 {
            let mut sq = 0usize;
            while sq < BOARD_SIZE {
                table[color][kind][sq] = piece_square_formula(color, kind, sq);
                sq += 1;
            }
            kind += 1;
        }
        color += 1;
    }
    table
}

const fn build_signed_piece_contrib_table() -> [[[i32; BOARD_SIZE]; 7]; 2] {
    let mut table = [[[0; BOARD_SIZE]; 7]; 2];
    let mut color = 0usize;
    while color < 2 {
        let mut kind = 0usize;
        while kind < 7 {
            let mut sq = 0usize;
            while sq < BOARD_SIZE {
                let value = PIECE_BASE_VALUES[kind] + PIECE_SQUARE_TABLE[color][kind][sq];
                table[color][kind][sq] = if color == 0 { value } else { -value };
                sq += 1;
            }
            kind += 1;
        }
        color += 1;
    }
    table
}

const fn piece_square_formula(color: usize, kind: usize, sq: usize) -> i32 {
    let file = (sq % BOARD_FILES) as i32;
    let rank = (sq / BOARD_FILES) as i32;
    let relative_rank = if color == 0 {
        (BOARD_RANKS as i32 - 1) - rank
    } else {
        rank
    };
    let center_distance = abs_i32(file - 4);

    match kind {
        0 => -(center_distance * 3),
        1 => 8 - center_distance * 4,
        2 => 10 - center_distance * 2 - abs_i32(relative_rank - 1) * 2,
        3 => 40 - center_distance * 8 - abs_i32(relative_rank - 4) * 4,
        4 => 18 - center_distance * 3 + relative_rank * 2,
        5 => 26 - center_distance * 5 + relative_rank * 3,
        6 => {
            let crossed = if relative_rank >= 5 { 1 } else { 0 };
            6 + relative_rank * 16 + crossed * 40 - center_distance * 4
        }
        _ => 0,
    }
}

const fn abs_i32(value: i32) -> i32 {
    if value < 0 { -value } else { value }
}

#[inline(always)]
const fn piece_kind_index(kind: PieceKind) -> usize {
    match kind {
        PieceKind::General => 0,
        PieceKind::Advisor => 1,
        PieceKind::Elephant => 2,
        PieceKind::Horse => 3,
        PieceKind::Rook => 4,
        PieceKind::Cannon => 5,
        PieceKind::Soldier => 6,
    }
}

#[inline(always)]
const fn color_index_const(color: Color) -> usize {
    match color {
        Color::Red => 0,
        Color::Black => 1,
    }
}

const SIDE_TO_MOVE_KEY: u64 = 0x9e37_79b9_7f4a_7c15;

#[inline(always)]
fn zobrist_piece_key(sq: usize, piece: Piece) -> u64 {
    splitmix64(((sq as u64) << 5) ^ piece_hash_index(piece) as u64)
}

#[inline(always)]
fn piece_hash_index(piece: Piece) -> usize {
    color_hash_index(piece.color) * 7
        + match piece.kind {
            PieceKind::General => 0,
            PieceKind::Advisor => 1,
            PieceKind::Elephant => 2,
            PieceKind::Horse => 3,
            PieceKind::Rook => 4,
            PieceKind::Cannon => 5,
            PieceKind::Soldier => 6,
        }
}

#[inline(always)]
fn color_hash_index(color: Color) -> usize {
    match color {
        Color::Red => 0,
        Color::Black => 1,
    }
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startpos_roundtrip_fen() {
        let position = Position::startpos();
        assert_eq!(position.to_fen(), STARTPOS_FEN);
    }

    #[test]
    fn square_names_follow_pikafish_uci_coordinates() {
        assert_eq!(square_name(index(0, 9)), "a0");
        assert_eq!(square_name(index(8, 0)), "i9");
        assert_eq!(parse_square("a0"), Some(index(0, 9)));
        assert_eq!(parse_square("i9"), Some(index(8, 0)));
    }

    #[test]
    fn startpos_perft_matches_reference_depths_1_to_3() {
        let position = Position::startpos();
        assert_eq!(position.perft(1), 44);
        assert_eq!(position.perft(2), 1_920);
        assert_eq!(position.perft(3), 79_666);
    }

    #[test]
    fn horse_leg_block_prevents_move() {
        let position = Position::from_fen("4k4/9/9/9/4P4/9/3P5/3H5/9/4K4 w").unwrap();
        let moves = position.legal_moves();
        assert!(!moves.iter().any(|mv| mv.to == index(2, 5) as u8));
        assert!(!moves.iter().any(|mv| mv.to == index(4, 5) as u8));
    }

    #[test]
    fn elephant_cannot_cross_river() {
        let position = Position::from_fen("4k4/9/9/9/4P4/9/9/2E6/9/4K4 w").unwrap();
        let moves = position.legal_moves();
        let elephant_moves: Vec<_> = moves
            .iter()
            .filter(|mv| mv.from == index(2, 7) as u8)
            .copied()
            .collect();
        assert!(!elephant_moves.is_empty());
        assert!(elephant_moves.iter().all(|mv| rank_of(mv.to as usize) >= 5));
    }

    #[test]
    fn cannon_requires_exactly_one_screen_to_capture() {
        let position = Position::from_fen("4k4/9/9/9/4C4/4P4/4r4/9/9/3K5 w").unwrap();
        let moves = position.legal_moves();
        assert!(moves.iter().any(|mv| mv.to == index(4, 6) as u8));

        let position_without_screen = Position::from_fen("4k4/9/9/9/4C4/9/4r4/9/9/3K5 w").unwrap();
        let moves_without_screen = position_without_screen.legal_moves();
        assert!(
            !moves_without_screen
                .iter()
                .any(|mv| mv.to == index(4, 6) as u8)
        );
    }

    #[test]
    fn facing_generals_exposure_is_illegal() {
        let position = Position::from_fen("4k4/9/9/9/9/9/4R4/9/9/4K4 w").unwrap();
        let moves = position.legal_moves();
        assert!(
            !moves
                .iter()
                .any(|mv| mv.from == index(4, 6) as u8 && mv.to == index(3, 6) as u8)
        );
    }

    #[test]
    fn legal_moves_do_not_capture_general() {
        let position = Position::from_fen("4k4/9/9/9/9/9/4R4/9/9/4K4 w").unwrap();
        let black_general = index(4, 0) as u8;
        let capture_general = Move::from_uci("e3e9").unwrap();
        let moves = position.legal_moves();

        assert!(!moves.contains(&capture_general));
        assert!(!moves.iter().any(|mv| mv.to == black_general));
    }

    #[test]
    fn checkmate_is_no_legal_reply_with_both_generals_present() {
        let position = Position::from_fen("3rkr3/4R4/9/9/9/9/9/9/9/4K4 b").unwrap();

        assert!(position.find_general(Color::Red).is_some());
        assert!(position.find_general(Color::Black).is_some());
        assert!(position.in_check(Color::Black));
        assert!(position.legal_moves().is_empty());
    }

    #[test]
    fn facing_generals_position_is_rejected() {
        let position = Position::from_fen("4k4/9/9/9/9/9/9/9/9/4K4 w").unwrap_err();
        assert_eq!(position, "illegal position: generals are facing");
    }

    #[test]
    fn make_and_unmake_restores_position() {
        let mut position = Position::startpos();
        let original = position.clone();
        let mv = position.legal_moves()[0];
        let undo = position.make_move(mv);
        position.unmake_move(mv, undo);
        assert_eq!(position, original);
    }

    #[test]
    fn hash_is_stable_across_make_and_unmake() {
        let mut position = Position::startpos();
        let original_hash = position.hash();
        let mv = position.legal_moves()[0];
        let undo = position.make_move(mv);
        position.unmake_move(mv, undo);
        assert_eq!(position.hash(), original_hash);
    }

    #[test]
    fn no_capture_clock_resets_only_on_capture() {
        let mut position = Position::from_fen("4k4/9/9/9/9/9/4P4/9/9/4K4 w").unwrap();
        let mv = Move::from_uci("e3e4").unwrap();

        position.make_move(mv);

        assert_eq!(position.halfmove_clock(), 1);
    }

    #[test]
    fn incremental_hash_matches_full_recomputation() {
        let mut position = Position::startpos();
        for mv in position.legal_moves().into_iter().take(8) {
            let undo = position.make_move(mv);
            assert_eq!(position.hash(), position.compute_hash());
            position.unmake_move(mv, undo);
            assert_eq!(position.hash(), position.compute_hash());
        }
    }

    #[test]
    fn parses_official_uci_move_notation() {
        let position = Position::startpos();
        let mv = position.parse_uci_move("h2e2").unwrap();
        assert_eq!(mv, Move::new(index(7, 7), index(4, 7)));
        assert_eq!(mv.to_string(), "h2e2");
    }

    #[test]
    fn legal_capture_moves_match_filtered_legal_moves() {
        let position = Position::from_fen("4k4/9/9/9/4C4/4P4/4r4/9/9/3K5 w").unwrap();
        let captures = position.legal_capture_moves();
        let filtered: Vec<_> = position
            .legal_moves()
            .into_iter()
            .filter(|mv| position.is_capture(*mv))
            .collect();

        assert_eq!(captures, filtered);
    }

    #[test]
    fn legal_capture_moves_to_matches_filtered_capture_moves() {
        let position = Position::from_fen("4k4/9/9/9/4C4/4P4/4r4/9/9/3K5 w").unwrap();
        let target = index(4, 6);
        let targeted = position.legal_capture_moves_to(target);
        let filtered: Vec<_> = position
            .legal_capture_moves()
            .into_iter()
            .filter(|mv| mv.to as usize == target)
            .collect();

        assert_eq!(targeted, filtered);
    }

    #[test]
    fn fast_attack_detection_matches_slow_scan() {
        let samples = [
            Position::startpos(),
            Position::from_fen("4k4/9/9/9/4C4/4P4/4r4/9/9/3K5 w").unwrap(),
            Position::from_fen("4k4/9/9/9/9/9/4R4/9/9/4K4 w").unwrap(),
        ];

        for (sample_index, position) in samples.into_iter().enumerate() {
            for sq in 0..BOARD_SIZE {
                if !position
                    .piece_at(sq)
                    .is_some_and(|piece| piece.color == Color::Red)
                {
                    assert_eq!(
                        position.is_square_attacked(sq, Color::Red),
                        position.is_square_attacked_slow(sq, Color::Red),
                        "sample={sample_index} color=Red sq={sq}"
                    );
                }
                if !position
                    .piece_at(sq)
                    .is_some_and(|piece| piece.color == Color::Black)
                {
                    assert_eq!(
                        position.is_square_attacked(sq, Color::Black),
                        position.is_square_attacked_slow(sq, Color::Black),
                        "sample={sample_index} color=Black sq={sq}"
                    );
                }
            }
        }
    }

    #[test]
    fn dynamic_material_cache_updates_across_make_and_unmake() {
        let mut position = Position::from_fen("4k4/9/9/9/9/9/4R4/9/9/4K4 w").unwrap();
        assert!(position.has_dynamic_material(Color::Red));
        assert!(!position.has_dynamic_material(Color::Black));

        let mv = Move::from_uci("e3e9").unwrap();
        let undo = position.make_move(mv);
        assert!(position.has_dynamic_material(Color::Red));
        assert!(!position.has_dynamic_material(Color::Black));

        position.unmake_move(mv, undo);
        assert!(position.has_dynamic_material(Color::Red));
        assert!(!position.has_dynamic_material(Color::Black));
    }

    #[test]
    fn cannon_check_allows_moving_screen_piece_away() {
        let position = Position::from_fen("4k4/9/9/9/4c4/9/9/4R4/9/4K4 w").unwrap();
        assert!(position.in_check(Color::Red));
        let moves = position.legal_moves();
        assert!(moves.contains(&Move::from_uci("e2a2").unwrap()));
    }

    #[test]
    fn excess_checks_do_not_advance_effective_no_capture_draw() {
        let mut position = Position::from_fen("4k4/4R4/9/9/9/9/9/9/9/4K4 b 120").unwrap();
        position.check_in_no_capture = 41;

        assert_eq!(
            position.rule_outcome_with_history(&position.initial_rule_history()),
            None
        );

        position.check_in_no_capture = 40;
        assert_eq!(
            position.rule_outcome_with_history(&position.initial_rule_history()),
            Some(RuleOutcome::Draw(RuleDrawReason::Halfmove120))
        );
    }

    fn test_rule_entry(
        hash: u64,
        side_to_move: Color,
        mover: Option<Color>,
        gives_check: bool,
        chased_mask: u128,
    ) -> RuleHistoryEntry {
        RuleHistoryEntry {
            hash,
            side_to_move,
            mover,
            is_capture: false,
            gives_check,
            checking_pieces: u8::from(gives_check),
            chased_mask,
        }
    }

    #[test]
    fn repetition_with_long_check_loses_for_checker() {
        let history = vec![
            test_rule_entry(1, Color::Red, None, false, 0),
            test_rule_entry(2, Color::Black, Some(Color::Red), true, 0),
            test_rule_entry(3, Color::Red, Some(Color::Black), false, 0),
            test_rule_entry(4, Color::Black, Some(Color::Red), true, 0),
            test_rule_entry(1, Color::Red, Some(Color::Black), false, 0),
        ];
        assert_eq!(
            Position::rule_outcome(&history),
            Some(RuleOutcome::Win(Color::Black))
        );
    }

    #[test]
    fn continuous_long_check_loses_without_waiting_for_repetition() {
        let history = vec![
            test_rule_entry(100, Color::Red, None, false, 0),
            test_rule_entry(101, Color::Black, Some(Color::Red), true, 0),
            test_rule_entry(102, Color::Red, Some(Color::Black), false, 0),
            test_rule_entry(103, Color::Black, Some(Color::Red), true, 0),
            test_rule_entry(104, Color::Red, Some(Color::Black), false, 0),
            test_rule_entry(105, Color::Black, Some(Color::Red), true, 0),
            test_rule_entry(106, Color::Red, Some(Color::Black), false, 0),
            test_rule_entry(107, Color::Black, Some(Color::Red), true, 0),
            test_rule_entry(108, Color::Red, Some(Color::Black), false, 0),
            test_rule_entry(109, Color::Black, Some(Color::Red), true, 0),
            test_rule_entry(110, Color::Red, Some(Color::Black), false, 0),
            test_rule_entry(111, Color::Black, Some(Color::Red), true, 0),
        ];
        assert_eq!(
            Position::rule_outcome(&history),
            Some(RuleOutcome::Win(Color::Black))
        );
    }

    #[test]
    fn continuous_long_chase_loses_without_waiting_for_repetition() {
        let history = vec![
            test_rule_entry(200, Color::Red, None, false, 0),
            test_rule_entry(201, Color::Black, Some(Color::Red), false, 1 << 20),
            test_rule_entry(202, Color::Red, Some(Color::Black), false, 0),
            test_rule_entry(203, Color::Black, Some(Color::Red), false, 1 << 21),
            test_rule_entry(204, Color::Red, Some(Color::Black), false, 0),
            test_rule_entry(205, Color::Black, Some(Color::Red), false, 1 << 22),
            test_rule_entry(206, Color::Red, Some(Color::Black), false, 0),
            test_rule_entry(207, Color::Black, Some(Color::Red), false, 1 << 23),
            test_rule_entry(208, Color::Red, Some(Color::Black), false, 0),
            test_rule_entry(209, Color::Black, Some(Color::Red), false, 1 << 24),
            test_rule_entry(210, Color::Red, Some(Color::Black), false, 0),
            test_rule_entry(211, Color::Black, Some(Color::Red), false, 1 << 25),
        ];
        assert_eq!(
            Position::rule_outcome(&history),
            Some(RuleOutcome::Win(Color::Black))
        );
    }

    #[test]
    fn repetition_with_long_chase_loses_for_chaser() {
        let history = vec![
            test_rule_entry(10, Color::Red, None, false, 0),
            test_rule_entry(11, Color::Black, Some(Color::Red), false, 1 << 20),
            test_rule_entry(12, Color::Red, Some(Color::Black), false, 0),
            test_rule_entry(13, Color::Black, Some(Color::Red), false, 1 << 20),
            test_rule_entry(10, Color::Red, Some(Color::Black), false, 0),
        ];
        assert_eq!(
            Position::rule_outcome(&history),
            Some(RuleOutcome::Win(Color::Black))
        );
    }

    #[test]
    fn mutual_long_chase_loses_for_last_mover() {
        let history = vec![
            test_rule_entry(30, Color::Red, None, false, 0),
            test_rule_entry(31, Color::Black, Some(Color::Red), false, 1 << 20),
            test_rule_entry(32, Color::Red, Some(Color::Black), false, 1 << 40),
            test_rule_entry(30, Color::Red, Some(Color::Black), false, 1 << 40),
        ];
        assert_eq!(
            Position::rule_outcome(&history),
            Some(RuleOutcome::Win(Color::Red))
        );
    }

    #[test]
    fn mutual_long_check_loses_for_last_mover() {
        let history = vec![
            test_rule_entry(40, Color::Red, None, false, 0),
            test_rule_entry(41, Color::Black, Some(Color::Red), true, 0),
            test_rule_entry(42, Color::Red, Some(Color::Black), true, 0),
            test_rule_entry(40, Color::Red, Some(Color::Black), true, 0),
        ];
        assert_eq!(
            Position::rule_outcome(&history),
            Some(RuleOutcome::Win(Color::Red))
        );
    }

    #[test]
    fn fifth_repetition_without_forcing_is_draw() {
        let history = vec![
            test_rule_entry(21, Color::Red, None, false, 0),
            test_rule_entry(22, Color::Black, Some(Color::Red), false, 0),
            test_rule_entry(23, Color::Red, Some(Color::Black), false, 0),
            test_rule_entry(21, Color::Red, Some(Color::Black), false, 0),
            test_rule_entry(22, Color::Black, Some(Color::Red), false, 0),
            test_rule_entry(23, Color::Red, Some(Color::Black), false, 0),
            test_rule_entry(21, Color::Red, Some(Color::Black), false, 0),
            test_rule_entry(22, Color::Black, Some(Color::Red), false, 0),
            test_rule_entry(23, Color::Red, Some(Color::Black), false, 0),
            test_rule_entry(21, Color::Red, Some(Color::Black), false, 0),
            test_rule_entry(22, Color::Black, Some(Color::Red), false, 0),
            test_rule_entry(23, Color::Red, Some(Color::Black), false, 0),
            test_rule_entry(21, Color::Red, Some(Color::Black), false, 0),
        ];
        assert_eq!(
            Position::rule_outcome(&history),
            Some(RuleOutcome::Draw(RuleDrawReason::Repetition))
        );
    }
}
