use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use crate::nnue::NnueModel;
use crate::rules::{
    MoveRecord, RootMovePolicy, derive_policy_from_histories, make_record_after_move_mut,
};
use crate::xiangqi::{
    BOARD_FILES, BOARD_RANKS, BOARD_SIZE, Color, Move, Piece, PieceKind, Position,
    piece_base_value, piece_square_bonus,
};

const INF: i32 = 32_000;
pub const MATE_SCORE: i32 = 30_000;
const QUIESCENCE_MAX_PLY: usize = 32;
const MAX_PLY: usize = 128;
const DEFAULT_TT_ENTRIES: usize = 1 << 18;
const CAPTURE_SCORE_TABLE: [[i32; 7]; 7] = build_capture_score_table();

#[derive(Clone, Copy, Debug)]
pub struct SearchLimits {
    pub depth: u8,
    pub movetime: Option<Duration>,
    pub nodes: Option<u64>,
}

impl Default for SearchLimits {
    fn default() -> Self {
        Self {
            depth: 5,
            movetime: None,
            nodes: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub best_move: Option<Move>,
    pub score: i32,
    pub depth: u8,
    pub seldepth: u8,
    pub nodes: u64,
    pub pv: Vec<Move>,
}

#[derive(Clone, Debug)]
pub struct SearchProgress {
    pub depth: u8,
    pub seldepth: u8,
    pub score: i32,
    pub nodes: u64,
    pub elapsed: Duration,
    pub pv: Vec<Move>,
}

type SearchReporter = Arc<dyn Fn(SearchProgress) + Send + Sync>;

impl SearchResult {
    pub fn score_string(&self) -> String {
        if self.score.abs() >= MATE_SCORE - 256 {
            let plies = (MATE_SCORE - self.score.abs()).max(0);
            let mate = (plies + 1) / 2;
            if self.score > 0 {
                format!("mate {mate}")
            } else {
                format!("mate -{mate}")
            }
        } else {
            format!("cp {}", self.score)
        }
    }

    pub fn pv_string(&self) -> String {
        self.pv
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" ")
    }
}

pub struct Engine {
    table: TranspositionTable,
    killers: [[Option<Move>; 2]; MAX_PLY],
    history: [[[i32; BOARD_SIZE]; BOARD_SIZE]; 2],
    root_history: Vec<u64>,
    root_records: Vec<MoveRecord>,
    root_policy: RootMovePolicy,
    nnue_model: Option<Arc<NnueModel>>,
    reporter: Option<SearchReporter>,
    search_started_at: Option<Instant>,
    deadline: Option<Instant>,
    node_budget: Option<u64>,
    stop_flag: Option<Arc<AtomicBool>>,
    aborted: bool,
    seldepth: usize,
    nodes: u64,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new(DEFAULT_TT_ENTRIES)
    }
}

impl Engine {
    pub fn new(table_entries: usize) -> Self {
        Self {
            table: TranspositionTable::new(table_entries.max(1)),
            killers: [[None; 2]; MAX_PLY],
            history: [[[0; BOARD_SIZE]; BOARD_SIZE]; 2],
            root_history: Vec::new(),
            root_records: Vec::new(),
            root_policy: RootMovePolicy::default(),
            nnue_model: None,
            reporter: None,
            search_started_at: None,
            deadline: None,
            node_budget: None,
            stop_flag: None,
            aborted: false,
            seldepth: 0,
            nodes: 0,
        }
    }

    pub fn search(&mut self, position: &Position, limits: SearchLimits) -> SearchResult {
        self.search_with_history_and_records(position, &[], &[], limits)
    }

    pub fn search_with_history(
        &mut self,
        position: &Position,
        history: &[u64],
        limits: SearchLimits,
    ) -> SearchResult {
        self.search_with_history_and_records(position, history, &[], limits)
    }

    pub fn search_with_history_and_records(
        &mut self,
        position: &Position,
        history: &[u64],
        records: &[MoveRecord],
        limits: SearchLimits,
    ) -> SearchResult {
        self.nodes = 0;
        self.root_history.clear();
        self.root_history.extend_from_slice(history);
        self.root_records.clear();
        self.root_records.extend_from_slice(records);
        self.search_started_at = Some(Instant::now());
        self.deadline = limits.movetime.map(|limit| Instant::now() + limit);
        self.node_budget = limits.nodes;
        self.aborted = false;
        self.seldepth = 0;

        let mut root = position.clone();
        let mut previous_score = 0;
        let mut best_move = None;
        let mut best_pv = Vec::new();
        let mut stack = Vec::with_capacity(MAX_PLY);
        let mut record_stack = Vec::with_capacity(MAX_PLY);
        let mut completed_depth = 0;

        for current_depth in 1..=limits.depth {
            let mut aspiration = 48;
            let mut alpha = if current_depth == 1 {
                -INF
            } else {
                previous_score - aspiration
            };
            let mut beta = if current_depth == 1 {
                INF
            } else {
                previous_score + aspiration
            };

            loop {
                let score = self.alpha_beta(
                    &mut root,
                    current_depth as i32,
                    alpha,
                    beta,
                    0,
                    &mut stack,
                    &mut record_stack,
                );
                if self.aborted {
                    break;
                }
                if score <= alpha {
                    alpha = (alpha - aspiration).max(-INF);
                    aspiration *= 2;
                    continue;
                }
                if score >= beta {
                    beta = (beta + aspiration).min(INF);
                    aspiration *= 2;
                    continue;
                }

                previous_score = score;
                best_move = self.table.best_move(root.hash());
                best_pv = self.extract_pv(position, current_depth);
                completed_depth = current_depth;
                if let Some(reporter) = &self.reporter {
                    reporter(SearchProgress {
                        depth: current_depth,
                        seldepth: self.seldepth.min(u8::MAX as usize) as u8,
                        score,
                        nodes: self.nodes,
                        elapsed: self
                            .search_started_at
                            .unwrap_or_else(Instant::now)
                            .elapsed(),
                        pv: best_pv.clone(),
                    });
                }
                break;
            }

            if self.aborted {
                break;
            }
        }

        self.deadline = None;
        self.node_budget = None;
        self.search_started_at = None;
        SearchResult {
            best_move,
            score: previous_score,
            depth: completed_depth,
            seldepth: self.seldepth.min(u8::MAX as usize) as u8,
            nodes: self.nodes,
            pv: best_pv,
        }
    }

    pub fn clear_hash(&mut self) {
        self.table.clear();
    }

    pub fn resize_hash_mb(&mut self, hash_mb: usize) {
        let entries = hash_mb.max(1) * 8192;
        self.table.resize(entries);
    }

    pub fn set_root_policy(&mut self, policy: RootMovePolicy) {
        self.root_policy = policy;
    }

    pub fn set_nnue_model(&mut self, model: Option<Arc<NnueModel>>) {
        self.nnue_model = model;
    }

    pub fn set_reporter(&mut self, reporter: Option<SearchReporter>) {
        self.reporter = reporter;
    }

    pub fn set_stop_flag(&mut self, stop_flag: Option<Arc<AtomicBool>>) {
        self.stop_flag = stop_flag;
    }

    fn alpha_beta(
        &mut self,
        position: &mut Position,
        mut depth: i32,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        stack: &mut Vec<u64>,
        record_stack: &mut Vec<MoveRecord>,
    ) -> i32 {
        self.nodes += 1;
        self.seldepth = self.seldepth.max(ply);
        if self.should_stop() {
            self.aborted = true;
            return self.evaluate_position(position);
        }
        let alpha_orig = alpha;
        let key = position.hash();

        if self.repetition_count_at_least(key, stack, 2) {
            return 0;
        }
        stack.push(key);

        if !position.has_general(position.side_to_move()) {
            stack.pop();
            return -mate_score(ply);
        }
        if !position.has_general(position.side_to_move().opposite()) {
            stack.pop();
            return mate_score(ply);
        }
        if position.halfmove_clock() >= 120 {
            stack.pop();
            return 0;
        }

        if ply >= MAX_PLY - 1 {
            let eval = self.evaluate_position(position);
            stack.pop();
            return eval;
        }

        let in_check = position.in_check(position.side_to_move());
        if in_check {
            depth += 1;
        }
        if depth <= 0 {
            let score = self.quiescence(position, alpha, beta, ply, stack, record_stack);
            stack.pop();
            return score;
        }

        let tt_entry = self.table.probe(key);
        if let Some(entry) = tt_entry {
            if entry.depth as i32 >= depth {
                match entry.bound {
                    Bound::Exact => {
                        stack.pop();
                        return entry.score;
                    }
                    Bound::Lower if entry.score >= beta => {
                        stack.pop();
                        return entry.score;
                    }
                    Bound::Upper if entry.score <= alpha => {
                        stack.pop();
                        return entry.score;
                    }
                    _ => {}
                }
            }
        }
        let tt_move = tt_entry.and_then(|entry| entry.best_move);

        if !in_check && depth >= 3 && position.has_dynamic_material(position.side_to_move()) {
            let undo = position.make_null_move();
            let reduction = if depth >= 6 { 3 } else { 2 };
            let score = -self.alpha_beta(
                position,
                depth - 1 - reduction,
                -beta,
                -beta + 1,
                ply + 1,
                stack,
                record_stack,
            );
            position.unmake_null_move(undo);
            if score >= beta {
                stack.pop();
                return beta;
            }
        }

        let node_policy = self.node_policy(position, stack, record_stack, ply);
        let policy_active = !node_policy.is_empty();
        let mut moves = position.legal_moves_with_check_hint(in_check);
        if policy_active {
            moves.retain(|mv| !node_policy.forbidden.contains(mv));
        }
        if moves.is_empty() {
            stack.pop();
            return -mate_score(ply);
        }
        let mut move_scores = self.score_moves(position, &moves, tt_move, ply);
        let static_eval = if !in_check && depth <= 3 {
            Some(self.evaluate_position(position))
        } else {
            None
        };

        let mut best_score = -INF;
        let mut best_move = None;

        for index in 0..moves.len() {
            let mv = pick_next_move(&mut moves, &mut move_scores, index);
            let was_capture = position.is_capture(mv);
            let is_quiet = !was_capture;
            if is_quiet
                && index > 0
                && !in_check
                && beta < MATE_SCORE - 512
                && alpha > -MATE_SCORE + 512
                && static_eval.is_some()
                && should_prune_quiet_move(depth, index, static_eval.unwrap(), alpha)
            {
                continue;
            }
            let score = if policy_active && node_policy.forced_draw.contains(&mv) {
                0
            } else {
                let mover = position.side_to_move();
                let moving_kind = position
                    .piece_at(mv.from as usize)
                    .expect("legal move must have mover")
                    .kind;
                let undo = position.make_move(mv);
                let record =
                    make_record_after_move_mut(position, mover, moving_kind, mv.to as usize);
                record_stack.push(record);
                let score = if index == 0 {
                    -self.alpha_beta(
                        position,
                        depth - 1,
                        -beta,
                        -alpha,
                        ply + 1,
                        stack,
                        record_stack,
                    )
                } else {
                    let reduction = quiet_reduction(depth, index, in_check, is_quiet);
                    let mut probe = -self.alpha_beta(
                        position,
                        (depth - 1 - reduction).max(0),
                        -alpha - 1,
                        -alpha,
                        ply + 1,
                        stack,
                        record_stack,
                    );
                    if probe > alpha && probe < beta {
                        probe = -self.alpha_beta(
                            position,
                            depth - 1,
                            -beta,
                            -alpha,
                            ply + 1,
                            stack,
                            record_stack,
                        );
                    }
                    probe
                };
                record_stack.pop();
                position.unmake_move(mv, undo);
                score
            };

            if score > best_score {
                best_score = score;
                best_move = Some(mv);
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                if !was_capture {
                    self.remember_killer(mv, ply);
                    self.bump_history(position.side_to_move(), mv, depth);
                }
                break;
            }
        }

        let bound = if best_score <= alpha_orig {
            Bound::Upper
        } else if best_score >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.table.store(TTEntry {
            key,
            depth: depth as u8,
            score: best_score,
            bound,
            best_move,
        });

        stack.pop();
        best_score
    }

    fn quiescence(
        &mut self,
        position: &mut Position,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        stack: &mut Vec<u64>,
        record_stack: &mut Vec<MoveRecord>,
    ) -> i32 {
        self.nodes += 1;
        self.seldepth = self.seldepth.max(ply);
        if self.should_stop() {
            self.aborted = true;
            return self.evaluate_position(position);
        }
        let alpha_orig = alpha;
        let key = position.hash();
        let tt_entry = self.table.probe(key);
        if let Some(entry) = tt_entry {
            match entry.bound {
                Bound::Exact => return entry.score,
                Bound::Lower if entry.score >= beta => return entry.score,
                Bound::Upper if entry.score <= alpha => return entry.score,
                _ => {}
            }
        }
        if self.repetition_count_at_least(key, stack, 2) {
            return 0;
        }
        stack.push(key);
        if !position.has_general(position.side_to_move()) {
            stack.pop();
            return -mate_score(ply);
        }
        if !position.has_general(position.side_to_move().opposite()) {
            stack.pop();
            return mate_score(ply);
        }
        if position.halfmove_clock() >= 120 {
            stack.pop();
            return 0;
        }

        let side_to_move = position.side_to_move();
        let in_check = position.in_check(side_to_move);
        let stand_pat = if in_check {
            -INF
        } else {
            let eval = self.evaluate_position(position);
            if eval >= beta {
                stack.pop();
                return beta;
            }
            if eval > alpha {
                alpha = eval;
            }
            eval
        };
        if in_check {
            alpha = alpha.max(-mate_score(ply));
        }
        if ply >= QUIESCENCE_MAX_PLY {
            stack.pop();
            return alpha;
        }

        let node_policy = self.node_policy(position, stack, record_stack, ply);
        let policy_active = !node_policy.is_empty();
        let mut captures = if in_check {
            position.legal_moves_with_check_hint(true)
        } else {
            position.legal_capture_moves_with_check_hint(false)
        };
        if policy_active {
            captures.retain(|mv| !node_policy.forbidden.contains(mv));
        }
        if captures.is_empty() {
            stack.pop();
            return alpha;
        }
        let mut best_move = None;
        let mut capture_scores = self.score_moves(
            position,
            &captures,
            tt_entry.and_then(|entry| entry.best_move),
            ply,
        );

        for index in 0..captures.len() {
            let mv = pick_next_move(&mut captures, &mut capture_scores, index);
            if policy_active && node_policy.forced_draw.contains(&mv) {
                alpha = alpha.max(0);
                if alpha >= beta {
                    stack.pop();
                    return alpha;
                }
                continue;
            }

            if !in_check {
                let victim = position
                    .piece_at(mv.to as usize)
                    .expect("capture move must have victim");
                let victim_value = piece_base_value(victim.kind);
                let optimistic = stand_pat + victim_value + 64;
                if optimistic <= alpha {
                    continue;
                }

                let attacker = position
                    .piece_at(mv.from as usize)
                    .expect("capture move must have attacker");
                let attacker_value = piece_base_value(attacker.kind);
                if attacker_value > victim_value {
                    let protected =
                        position.is_piece_protected(mv.to as usize, side_to_move.opposite());
                    if protected && attacker_value >= victim_value + 300 {
                        continue;
                    }
                    if protected && static_exchange_eval(position, mv) < 0 {
                        continue;
                    }
                }
            }

            let mover = position.side_to_move();
            let moving_kind = position
                .piece_at(mv.from as usize)
                .expect("legal move must have mover")
                .kind;
            let undo = position.make_move(mv);
            let record = make_record_after_move_mut(position, mover, moving_kind, mv.to as usize);
            record_stack.push(record);
            let score = -self.quiescence(position, -beta, -alpha, ply + 1, stack, record_stack);
            record_stack.pop();
            position.unmake_move(mv, undo);

            if score >= beta {
                self.table.store(TTEntry {
                    key,
                    depth: 0,
                    score: beta,
                    bound: Bound::Lower,
                    best_move: Some(mv),
                });
                stack.pop();
                return beta;
            }
            if score > alpha {
                alpha = score;
                best_move = Some(mv);
            }
        }

        let bound = if alpha <= alpha_orig {
            Bound::Upper
        } else {
            Bound::Exact
        };
        self.table.store(TTEntry {
            key,
            depth: 0,
            score: alpha,
            bound,
            best_move,
        });
        stack.pop();
        alpha
    }

    fn score_moves(
        &self,
        position: &Position,
        moves: &[Move],
        tt_move: Option<Move>,
        ply: usize,
    ) -> Vec<i32> {
        let side_index = color_index(position.side_to_move());
        moves
            .iter()
            .map(|mv| {
                if Some(*mv) == tt_move {
                    2_000_000
                } else if self.killers[ply][0] == Some(*mv) {
                    1_900_000
                } else if self.killers[ply][1] == Some(*mv) {
                    1_800_000
                } else if position.is_capture(*mv) {
                    1_000_000 + capture_score(position, *mv)
                } else {
                    self.history[side_index][mv.from as usize][mv.to as usize]
                        + quiet_positional_score(position, *mv)
                }
            })
            .collect()
    }

    fn remember_killer(&mut self, mv: Move, ply: usize) {
        if self.killers[ply][0] == Some(mv) {
            return;
        }
        self.killers[ply][1] = self.killers[ply][0];
        self.killers[ply][0] = Some(mv);
    }

    fn bump_history(&mut self, color: Color, mv: Move, depth: i32) {
        let side = color_index(color);
        let bonus = depth * depth;
        self.history[side][mv.from as usize][mv.to as usize] += bonus;
    }

    fn extract_pv(&self, position: &Position, depth: u8) -> Vec<Move> {
        let mut pv = Vec::new();
        let mut cursor = position.clone();
        for _ in 0..depth {
            let Some(mv) = self.table.best_move(cursor.hash()) else {
                break;
            };
            if !cursor.legal_moves().contains(&mv) {
                break;
            }
            pv.push(mv);
            cursor.make_move(mv);
        }
        pv
    }

    #[inline(always)]
    fn repetition_count_at_least(&self, key: u64, stack: &[u64], target: usize) -> bool {
        let mut count = 0usize;
        for &hash in self.root_history.iter().rev().skip(1).step_by(2) {
            if hash == key {
                count += 1;
                if count >= target {
                    return true;
                }
            }
        }
        for &hash in stack.iter().rev().skip(1).step_by(2) {
            if hash == key {
                count += 1;
                if count >= target {
                    return true;
                }
            }
        }
        false
    }

    fn node_policy(
        &self,
        position: &Position,
        stack: &[u64],
        record_stack: &[MoveRecord],
        ply: usize,
    ) -> RootMovePolicy {
        if !self.should_apply_node_policy(position, stack, record_stack, ply) {
            return RootMovePolicy::default();
        }
        let mut policy = derive_policy_from_histories(position, &self.root_records, record_stack);
        if ply == 0 {
            for mv in &self.root_policy.forbidden {
                if !policy.forbidden.contains(mv) {
                    policy.forbidden.push(*mv);
                }
            }
            for mv in &self.root_policy.forced_draw {
                if !policy.forced_draw.contains(mv) {
                    policy.forced_draw.push(*mv);
                }
            }
        }
        policy
    }

    fn should_apply_node_policy(
        &self,
        position: &Position,
        stack: &[u64],
        record_stack: &[MoveRecord],
        ply: usize,
    ) -> bool {
        if ply == 0
            && (!self.root_policy.forbidden.is_empty() || !self.root_policy.forced_draw.is_empty())
        {
            return true;
        }

        let record_len = self.root_records.len() + record_stack.len();
        if record_len < 4 {
            return false;
        }

        let current_hash = position.hash();
        if self.repetition_count_at_least(current_hash, stack, 1) {
            return true;
        }

        self.root_records
            .iter()
            .chain(record_stack.iter())
            .rev()
            .take(12)
            .any(|record| record.resulting_hash == current_hash)
    }

    fn evaluate_position(&self, position: &Position) -> i32 {
        let handcrafted = evaluate(position);
        let nnue = self
            .nnue_model
            .as_ref()
            .map(|model| model.evaluate(position))
            .unwrap_or(0);
        handcrafted + nnue / 2
    }

    fn should_stop(&self) -> bool {
        self.aborted
            || self
                .stop_flag
                .as_ref()
                .map(|flag| flag.load(Ordering::Relaxed))
                .unwrap_or(false)
            || self
                .deadline
                .map(|deadline| Instant::now() >= deadline)
                .unwrap_or(false)
            || self
                .node_budget
                .map(|limit| self.nodes >= limit)
                .unwrap_or(false)
    }
}

fn evaluate(position: &Position) -> i32 {
    let mut score = position.cached_base_eval();

    for sq in 0..BOARD_SIZE {
        let Some(piece) = position.piece_at(sq) else {
            continue;
        };

        match piece.color {
            Color::Red => score += line_activity_bonus(position, sq, piece),
            Color::Black => score -= line_activity_bonus(position, sq, piece),
        }
    }

    score += (position.advisor_count(Color::Red) as i32
        + position.elephant_count(Color::Red) as i32)
        * 18;
    score -= (position.advisor_count(Color::Black) as i32
        + position.elephant_count(Color::Black) as i32)
        * 18;

    match position.side_to_move() {
        Color::Red => score,
        Color::Black => -score,
    }
}

fn line_activity_bonus(position: &Position, sq: usize, piece: Piece) -> i32 {
    match piece.kind {
        PieceKind::Rook | PieceKind::Cannon => {
            let file = sq % BOARD_FILES;
            let rank = sq / BOARD_FILES;
            let mut mobility = 0;
            for (df, dr) in [(1_i32, 0_i32), (-1, 0), (0, 1), (0, -1)] {
                let mut nf = file as i32 + df;
                let mut nr = rank as i32 + dr;
                while (0..BOARD_FILES as i32).contains(&nf) && (0..BOARD_RANKS as i32).contains(&nr)
                {
                    let target = nr as usize * BOARD_FILES + nf as usize;
                    mobility += 1;
                    if position.piece_at(target).is_some() {
                        break;
                    }
                    nf += df;
                    nr += dr;
                }
            }
            mobility * 2
        }
        _ => 0,
    }
}

fn capture_score(position: &Position, mv: Move) -> i32 {
    let attacker = position
        .piece_at(mv.from as usize)
        .expect("capture move must have attacker");
    let victim = position
        .piece_at(mv.to as usize)
        .expect("capture move must have victim");
    CAPTURE_SCORE_TABLE[piece_kind_order_index(victim.kind)][piece_kind_order_index(attacker.kind)]
}

fn quiet_positional_score(position: &Position, mv: Move) -> i32 {
    let moving = position
        .piece_at(mv.from as usize)
        .expect("quiet move must have attacker");
    piece_square_bonus(moving, mv.to as usize) - piece_square_bonus(moving, mv.from as usize)
}

#[inline(always)]
fn mate_score(ply: usize) -> i32 {
    MATE_SCORE - ply as i32
}

fn static_exchange_eval(position: &Position, mv: Move) -> i32 {
    let mut sim = position.clone();
    let mut current = mv;
    let mut gains = [0i32; 32];
    let mut depth = 0usize;
    let target = mv.to as usize;

    let Some(captured) = sim.piece_at(target) else {
        return 0;
    };
    gains[0] = piece_base_value(captured.kind);

    loop {
        sim.make_move(current);
        let Some(reply) = least_valuable_capture_to(&sim, target) else {
            break;
        };
        depth += 1;
        if depth >= gains.len() {
            break;
        }
        let attacker = sim
            .piece_at(reply.from as usize)
            .expect("reply capture must have attacker");
        gains[depth] = piece_base_value(attacker.kind) - gains[depth - 1];
        current = reply;
    }

    while depth > 0 {
        gains[depth - 1] = -(-gains[depth - 1]).max(gains[depth]);
        depth -= 1;
    }

    gains[0]
}

fn least_valuable_capture_to(position: &Position, target: usize) -> Option<Move> {
    position.least_valuable_legal_capture_to(target)
}

#[inline(always)]
fn pick_next_move(moves: &mut [Move], scores: &mut [i32], start: usize) -> Move {
    let mut best = start;
    for idx in (start + 1)..moves.len() {
        if scores[idx] > scores[best] {
            best = idx;
        }
    }
    if best != start {
        moves.swap(start, best);
        scores.swap(start, best);
    }
    moves[start]
}

#[inline(always)]
fn should_prune_quiet_move(depth: i32, move_index: usize, static_eval: i32, alpha: i32) -> bool {
    match depth {
        1 => move_index >= 3 && static_eval + 90 <= alpha,
        2 => move_index >= 5 && static_eval + 160 <= alpha,
        3 => move_index >= 7 && static_eval + 240 <= alpha,
        _ => false,
    }
}

#[inline(always)]
fn quiet_reduction(depth: i32, move_index: usize, in_check: bool, is_quiet: bool) -> i32 {
    if in_check || !is_quiet || depth < 3 || move_index < 3 {
        return 0;
    }

    let mut reduction = 1;
    if depth >= 5 && move_index >= 5 {
        reduction += 1;
    }
    if depth >= 7 && move_index >= 8 {
        reduction += 1;
    }
    reduction
}

#[inline(always)]
fn color_index(color: Color) -> usize {
    match color {
        Color::Red => 0,
        Color::Black => 1,
    }
}

const fn build_capture_score_table() -> [[i32; 7]; 7] {
    let values = [0, 110, 110, 420, 900, 460, 110];
    let mut table = [[0; 7]; 7];
    let mut victim = 0usize;
    while victim < 7 {
        let mut attacker = 0usize;
        while attacker < 7 {
            table[victim][attacker] = values[victim] * 32 - values[attacker];
            attacker += 1;
        }
        victim += 1;
    }
    table
}

#[inline(always)]
const fn piece_kind_order_index(kind: PieceKind) -> usize {
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

#[derive(Clone, Copy)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy)]
struct TTEntry {
    key: u64,
    depth: u8,
    score: i32,
    bound: Bound,
    best_move: Option<Move>,
}

struct TranspositionTable {
    entries: Vec<Option<TTEntry>>,
}

impl TranspositionTable {
    fn new(size: usize) -> Self {
        let size = normalized_tt_size(size);
        Self {
            entries: vec![None; size],
        }
    }

    fn probe(&self, key: u64) -> Option<TTEntry> {
        let entry = self.entries[self.index(key)]?;
        (entry.key == key).then_some(entry)
    }

    fn best_move(&self, key: u64) -> Option<Move> {
        self.probe(key).and_then(|entry| entry.best_move)
    }

    fn store(&mut self, entry: TTEntry) {
        let index = self.index(entry.key);
        let should_replace = match self.entries[index] {
            Some(existing) => entry.depth >= existing.depth || matches!(entry.bound, Bound::Exact),
            None => true,
        };
        if should_replace {
            self.entries[index] = Some(entry);
        }
    }

    fn index(&self, key: u64) -> usize {
        key as usize % self.entries.len()
    }

    fn clear(&mut self) {
        self.entries.fill(None);
    }

    fn resize(&mut self, size: usize) {
        let size = normalized_tt_size(size);
        self.entries = vec![None; size];
    }
}

#[inline(always)]
fn normalized_tt_size(size: usize) -> usize {
    size.max(1)
}

pub fn render_principal_variation(pv: &[Move]) -> String {
    if pv.is_empty() {
        return "(none)".into();
    }
    pv.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xiangqi::STARTPOS_FEN;
    use std::sync::Mutex;

    #[test]
    fn search_finds_general_capture() {
        let position = Position::from_fen("4k4/9/9/9/9/9/4R4/9/9/4K4 w").unwrap();
        let mut engine = Engine::default();
        let result = engine.search(
            &position,
            SearchLimits {
                depth: 3,
                movetime: None,
                nodes: None,
            },
        );

        assert_eq!(result.best_move, Some(Move::new(4 + 6 * BOARD_FILES, 4)));
        assert!(result.score > MATE_SCORE - 10);
    }

    #[test]
    fn search_prefers_winning_capture_with_cannon() {
        let position = Position::from_fen("4k4/9/9/9/4C4/4P4/4r4/9/9/3K5 w").unwrap();
        let mut engine = Engine::default();
        let result = engine.search(
            &position,
            SearchLimits {
                depth: 3,
                movetime: None,
                nodes: None,
            },
        );

        assert_eq!(
            result.best_move,
            Some(Move::new(4 + 4 * BOARD_FILES, 4 + 6 * BOARD_FILES))
        );
    }

    #[test]
    fn search_scores_threefold_repetition_as_draw_baseline() {
        let mut position = Position::startpos();
        let mut history = Vec::new();

        for mv_text in [
            "h2e2", "h7e7", "e2h2", "e7h7", "h2e2", "h7e7", "e2h2", "e7h7",
        ] {
            history.push(position.hash());
            let mv = position.parse_uci_move(mv_text).unwrap();
            position.make_move(mv);
        }

        let mut engine = Engine::default();
        let result = engine.search_with_history(
            &position,
            &history,
            SearchLimits {
                depth: 3,
                movetime: None,
                nodes: None,
            },
        );

        assert_eq!(position.to_fen(), STARTPOS_FEN);
        assert_eq!(result.score, 0);
        assert_eq!(result.best_move, None);
    }

    #[test]
    fn search_respects_node_budget() {
        let position = Position::startpos();
        let mut engine = Engine::default();
        let result = engine.search(
            &position,
            SearchLimits {
                depth: 64,
                movetime: None,
                nodes: Some(200),
            },
        );

        assert!(result.nodes >= 200);
        assert!(result.depth <= 2);
        assert!(result.seldepth >= result.depth);
    }

    #[test]
    fn search_reports_iterative_progress() {
        let position = Position::startpos();
        let depths = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&depths);
        let mut engine = Engine::default();
        engine.set_reporter(Some(Arc::new(move |progress| {
            sink.lock().unwrap().push(progress.depth);
        })));

        let _ = engine.search(
            &position,
            SearchLimits {
                depth: 3,
                movetime: None,
                nodes: None,
            },
        );

        let seen = depths.lock().unwrap().clone();
        assert!(seen.contains(&1));
        assert!(seen.contains(&2) || seen.contains(&3));
    }

    #[test]
    fn search_tracks_selective_depth() {
        let position = Position::startpos();
        let mut engine = Engine::default();
        let result = engine.search(
            &position,
            SearchLimits {
                depth: 3,
                movetime: None,
                nodes: None,
            },
        );

        assert!(result.seldepth >= result.depth);
    }
}
