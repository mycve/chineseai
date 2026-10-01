use super::{
    BOARD_SIZE, CheckerInfo, Color, DIAGONAL_STEPS, ELEPHANT_STEPS, HORSE_STEPS, Move, MoveGenMode,
    ORTHOGONAL_STEPS, Piece, PieceKind, Position, elephant_stays_home, file_of, horse_leg_square,
    index, inside_board, inside_palace, line_between_squares, rank_of, soldier_crossed_river,
};

impl Position {
    pub(super) fn pseudo_legal_moves(&self) -> Vec<Move> {
        self.pseudo_legal_moves_with_mode(MoveGenMode::All)
    }

    pub(super) fn pseudo_legal_capture_moves(&self) -> Vec<Move> {
        self.pseudo_legal_moves_with_mode(MoveGenMode::Captures)
    }

    fn pseudo_legal_moves_with_mode(&self, mode: MoveGenMode) -> Vec<Move> {
        let mut moves = Vec::with_capacity(64);

        let mut pieces = self.occupied;
        while pieces != 0 {
            let sq = pieces.trailing_zeros() as usize;
            pieces &= pieces - 1;
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

    pub(super) fn pseudo_legal_evasions(&self) -> Vec<Move> {
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
            return self.pseudo_legal_moves();
        }

        let checker = checkers[0];
        let mut allow_to = [false; BOARD_SIZE];
        allow_to[checker.from] = true;
        for sq in self.interposition_squares(king_sq, &checker) {
            allow_to[sq] = true;
        }

        let mut piece_moves = Vec::with_capacity(16);
        let mut pieces = self.occupied;
        while pieces != 0 {
            let sq = pieces.trailing_zeros() as usize;
            pieces &= pieces - 1;
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

    pub(super) fn gen_piece_moves(
        &self,
        sq: usize,
        piece: Piece,
        mode: MoveGenMode,
        moves: &mut Vec<Move>,
    ) {
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
}
