use std::io::{self, BufRead, Write};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::thread;
use std::time::{Duration, Instant};

use crate::nnue::NnueModel;
use crate::rules::{MoveRecord, make_record};
use crate::search::{
    Engine, SearchLimits, SearchProgress, SearchResult, render_principal_variation,
};
use crate::xiangqi::{Color, Position};

const ENGINE_NAME: &str = "ChineseAI";
const ENGINE_AUTHOR: &str = "OpenAI Codex";
const DEFAULT_HASH_MB: usize = 16;

pub fn run_loop() -> io::Result<()> {
    let (command_tx, command_rx) = mpsc::channel::<String>();
    let (search_tx, search_rx) = mpsc::channel::<SearchEvent>();

    thread::spawn(move || {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            match line {
                Ok(line) => {
                    if command_tx.send(line).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    let mut session = UciSession::new(search_tx);

    loop {
        session.drain_search_events(&search_rx)?;

        match command_rx.recv_timeout(Duration::from_millis(20)) {
            Ok(line) => {
                let trimmed = line.trim().to_string();
                if !session.handle_command(&line)? {
                    session.stop_search();
                    session.wait_for_search_completion(&search_rx, Duration::from_millis(200))?;
                    break;
                }
                if trimmed == "stop" {
                    session.wait_for_search_completion(&search_rx, Duration::from_millis(250))?;
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                session.stop_search();
                session.wait_for_search_completion(&search_rx, Duration::from_millis(200))?;
                break;
            }
        }
    }

    Ok(())
}

struct UciSession {
    engine: Arc<Mutex<Engine>>,
    search_tx: mpsc::Sender<SearchEvent>,
    active_search: Option<ActiveSearch>,
    next_search_id: u64,
    position: Position,
    history_hashes: Vec<u64>,
    move_records: Vec<MoveRecord>,
    hash_mb: usize,
}

struct ActiveSearch {
    id: u64,
    stop: Arc<AtomicBool>,
    last_snapshot: Option<SearchSnapshot>,
}

struct SearchEvent {
    id: u64,
    kind: SearchEventKind,
}

enum SearchEventKind {
    Progress(SearchProgress),
    Finished {
        result: SearchResult,
        elapsed: Duration,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SearchSnapshot {
    depth: u8,
    seldepth: u8,
    score: i32,
    nodes: u64,
    pv: Vec<crate::xiangqi::Move>,
}

impl UciSession {
    fn new(search_tx: mpsc::Sender<SearchEvent>) -> Self {
        let mut engine = Engine::default();
        engine.resize_hash_mb(DEFAULT_HASH_MB);
        Self {
            engine: Arc::new(Mutex::new(engine)),
            search_tx,
            active_search: None,
            next_search_id: 1,
            position: Position::startpos(),
            history_hashes: Vec::new(),
            move_records: Vec::new(),
            hash_mb: DEFAULT_HASH_MB,
        }
    }

    fn handle_command(&mut self, command: &str) -> io::Result<bool> {
        let trimmed = command.trim();
        if trimmed.is_empty() {
            return Ok(true);
        }

        let mut parts = trimmed.split_whitespace();
        let keyword = parts.next().unwrap();

        match keyword {
            "uci" => {
                self.print_id()?;
                self.print_options()?;
                self.out("uciok")?;
            }
            "isready" => {
                self.out("readyok")?;
            }
            "ucinewgame" => {
                self.stop_search();
                self.position = Position::startpos();
                self.history_hashes.clear();
                self.move_records.clear();
                self.with_engine(|engine| engine.clear_hash())?;
            }
            "setoption" => {
                self.handle_setoption(trimmed)?;
            }
            "position" => {
                self.stop_search();
                self.handle_position(trimmed)?;
            }
            "go" => {
                self.handle_go(trimmed)?;
            }
            "stop" => {
                self.stop_search();
            }
            "quit" => return Ok(false),
            "d" => {
                self.out(&format!("info string fen {}", self.position.to_fen()))?;
            }
            _ => {
                self.out(&format!("info string unsupported command: {trimmed}"))?;
            }
        }

        Ok(true)
    }

    fn handle_position(&mut self, command: &str) -> io::Result<()> {
        let tokens: Vec<&str> = command.split_whitespace().collect();
        let Some(rest) = tokens.get(1..) else {
            return Ok(());
        };

        let moves_index = rest.iter().position(|token| *token == "moves");
        let setup_tokens = moves_index.map(|idx| &rest[..idx]).unwrap_or(rest);
        let move_tokens = moves_index.map(|idx| &rest[idx + 1..]).unwrap_or(&[]);

        if setup_tokens.is_empty() || setup_tokens[0] == "startpos" {
            self.position = Position::startpos();
        } else if setup_tokens[0] == "fen" {
            let fen = setup_tokens[1..].join(" ");
            self.position = Position::from_fen(&fen)
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?;
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid position command: {command}"),
            ));
        }

        self.history_hashes.clear();
        self.move_records.clear();
        for mv_text in move_tokens {
            self.history_hashes.push(self.position.hash());
            let Some(mv) = self.position.parse_uci_move(mv_text) else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("illegal move in position command: {mv_text}"),
                ));
            };
            self.move_records.push(make_record(&self.position, mv));
            self.position.make_move(mv);
        }

        Ok(())
    }

    fn handle_go(&mut self, command: &str) -> io::Result<()> {
        if let Some(active) = &self.active_search {
            active.stop.store(true, Ordering::Relaxed);
        }

        let tokens: Vec<&str> = command.split_whitespace().collect();
        let mut depth = None;
        let mut perft = None;
        let mut nodes = None;
        let mut movetime_ms = None;
        let mut wtime_ms = None;
        let mut btime_ms = None;
        let mut winc_ms = 0u64;
        let mut binc_ms = 0u64;
        let mut moves_to_go = None;
        let mut infinite = false;

        let mut index = 1usize;
        while index < tokens.len() {
            match tokens[index] {
                "depth" => {
                    if let Some(value) = tokens.get(index + 1).and_then(|v| v.parse::<u8>().ok()) {
                        depth = Some(value);
                    }
                    index += 2;
                }
                "perft" => {
                    if let Some(value) = tokens.get(index + 1).and_then(|v| v.parse::<u32>().ok()) {
                        perft = Some(value);
                    }
                    index += 2;
                }
                "nodes" => {
                    if let Some(value) = tokens.get(index + 1).and_then(|v| v.parse::<u64>().ok()) {
                        nodes = Some(value);
                    }
                    index += 2;
                }
                "movetime" => {
                    if let Some(value) = tokens.get(index + 1).and_then(|v| v.parse::<u64>().ok()) {
                        movetime_ms = Some(value);
                    }
                    index += 2;
                }
                "wtime" => {
                    if let Some(value) = tokens.get(index + 1).and_then(|v| v.parse::<u64>().ok()) {
                        wtime_ms = Some(value);
                    }
                    index += 2;
                }
                "btime" => {
                    if let Some(value) = tokens.get(index + 1).and_then(|v| v.parse::<u64>().ok()) {
                        btime_ms = Some(value);
                    }
                    index += 2;
                }
                "winc" => {
                    if let Some(value) = tokens.get(index + 1).and_then(|v| v.parse::<u64>().ok()) {
                        winc_ms = value;
                    }
                    index += 2;
                }
                "binc" => {
                    if let Some(value) = tokens.get(index + 1).and_then(|v| v.parse::<u64>().ok()) {
                        binc_ms = value;
                    }
                    index += 2;
                }
                "movestogo" => {
                    if let Some(value) = tokens.get(index + 1).and_then(|v| v.parse::<u64>().ok()) {
                        moves_to_go = Some(value.max(1));
                    }
                    index += 2;
                }
                "infinite" => {
                    infinite = true;
                    index += 1;
                }
                _ => index += 1,
            }
        }

        if let Some(perft_depth) = perft {
            let started = Instant::now();
            let nodes = self.position.perft(perft_depth);
            let time_ms = started.elapsed().as_millis();
            self.out(&format!(
                "info depth {perft_depth} nodes {nodes} time {time_ms} string perft"
            ))?;
            self.out("bestmove 0000")?;
            return Ok(());
        }

        let time_budget = if infinite {
            None
        } else if let Some(movetime) = movetime_ms {
            Some(Duration::from_millis(movetime.max(1)))
        } else {
            self.time_budget_from_clock(wtime_ms, btime_ms, winc_ms, binc_ms, moves_to_go)
        };
        let depth = depth.unwrap_or(if time_budget.is_some() || nodes.is_some() {
            64
        } else {
            5
        });

        let position = self.position.clone();
        let history_hashes = self.history_hashes.clone();
        let move_records = self.move_records.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let search_id = self.next_search_id;
        self.next_search_id += 1;
        self.active_search = Some(ActiveSearch {
            id: search_id,
            stop: Arc::clone(&stop),
            last_snapshot: None,
        });

        let engine = Arc::clone(&self.engine);
        let tx = self.search_tx.clone();
        thread::spawn(move || {
            let started = Instant::now();
            let result = {
                let mut engine = match engine.lock() {
                    Ok(guard) => guard,
                    Err(_) => return,
                };
                let progress_tx = tx.clone();
                let reporter = Arc::new(move |progress: SearchProgress| {
                    let _ = progress_tx.send(SearchEvent {
                        id: search_id,
                        kind: SearchEventKind::Progress(progress),
                    });
                });
                engine.set_stop_flag(Some(Arc::clone(&stop)));
                engine.set_reporter(Some(reporter));
                let result = engine.search_with_history_and_records(
                    &position,
                    &history_hashes,
                    &move_records,
                    SearchLimits {
                        depth,
                        movetime: time_budget,
                        nodes,
                    },
                );
                engine.set_reporter(None);
                engine.set_stop_flag(None);
                result
            };

            let _ = tx.send(SearchEvent {
                id: search_id,
                kind: SearchEventKind::Finished {
                    result,
                    elapsed: started.elapsed(),
                },
            });
        });

        Ok(())
    }

    fn time_budget_from_clock(
        &self,
        wtime_ms: Option<u64>,
        btime_ms: Option<u64>,
        winc_ms: u64,
        binc_ms: u64,
        moves_to_go: Option<u64>,
    ) -> Option<Duration> {
        let (remaining, increment) = match self.position.side_to_move() {
            Color::Red => (wtime_ms?, winc_ms),
            Color::Black => (btime_ms?, binc_ms),
        };

        let divisor = moves_to_go.unwrap_or(24).saturating_add(4).max(6);
        let base = remaining / divisor;
        let reserve = remaining / 8;
        let bonus = increment / 2;
        let spend = base.saturating_add(bonus).max(20);
        Some(Duration::from_millis(
            spend.min(remaining.saturating_sub(reserve)).max(1),
        ))
    }

    fn handle_setoption(&mut self, command: &str) -> io::Result<()> {
        let lower = command.to_ascii_lowercase();
        if lower.contains("name clear hash") {
            self.with_engine(|engine| engine.clear_hash())?;
            return Ok(());
        }

        if let Some(hash_index) = lower.find("name hash") {
            let value_slice = &command[hash_index..];
            if let Some(value_pos) = value_slice.to_ascii_lowercase().find("value") {
                let value_text = value_slice[value_pos + 5..].trim();
                if let Ok(hash_mb) = value_text.parse::<usize>() {
                    self.hash_mb = hash_mb.max(1);
                    let size = self.hash_mb;
                    self.with_engine(|engine| engine.resize_hash_mb(size))?;
                }
            }
        }

        if let Some(eval_index) = lower.find("name evalfile") {
            let value_slice = &command[eval_index..];
            if let Some(value_pos) = value_slice.to_ascii_lowercase().find("value") {
                let value_text = value_slice[value_pos + 5..].trim();
                if value_text.is_empty() || value_text.eq_ignore_ascii_case("<empty>") {
                    self.with_engine(|engine| engine.set_nnue_model(None))?;
                } else {
                    let model = Arc::new(NnueModel::load_text(value_text)?);
                    self.with_engine(|engine| engine.set_nnue_model(Some(model)))?;
                }
            }
        }

        Ok(())
    }

    fn drain_search_events(&mut self, rx: &mpsc::Receiver<SearchEvent>) -> io::Result<()> {
        while let Ok(event) = rx.try_recv() {
            self.handle_search_event(event)?;
        }
        Ok(())
    }

    fn wait_for_search_completion(
        &mut self,
        rx: &mpsc::Receiver<SearchEvent>,
        timeout: Duration,
    ) -> io::Result<()> {
        let Some(active) = &self.active_search else {
            return Ok(());
        };
        let target_id = active.id;
        let deadline = Instant::now() + timeout;

        while Instant::now() < deadline {
            match rx.recv_timeout(Duration::from_millis(10)) {
                Ok(event) => {
                    let is_target = event.id == target_id;
                    self.handle_search_event(event)?;
                    if is_target {
                        break;
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }

        Ok(())
    }

    fn handle_search_event(&mut self, event: SearchEvent) -> io::Result<()> {
        let Some(active) = &self.active_search else {
            return Ok(());
        };
        if active.id != event.id {
            return Ok(());
        }

        match event.kind {
            SearchEventKind::Progress(progress) => {
                let snapshot = SearchSnapshot {
                    depth: progress.depth,
                    seldepth: progress.seldepth,
                    score: progress.score,
                    nodes: progress.nodes,
                    pv: progress.pv.clone(),
                };
                if let Some(active) = &mut self.active_search {
                    active.last_snapshot = Some(snapshot);
                }
                self.emit_search_progress(&progress)
            }
            SearchEventKind::Finished { result, elapsed } => {
                let suppress_info = self
                    .active_search
                    .as_ref()
                    .and_then(|active| active.last_snapshot.as_ref())
                    .is_some_and(|snapshot| {
                        snapshot.depth == result.depth
                            && snapshot.seldepth == result.seldepth
                            && snapshot.score == result.score
                            && snapshot.nodes == result.nodes
                            && snapshot.pv == result.pv
                    });
                self.active_search = None;
                self.emit_search_result(&result, elapsed, suppress_info)
            }
        }
    }

    fn emit_search_progress(&self, progress: &SearchProgress) -> io::Result<()> {
        let score = if progress.score.abs() >= crate::search::MATE_SCORE - 256 {
            let plies = (crate::search::MATE_SCORE - progress.score.abs()).max(0);
            let mate = (plies + 1) / 2;
            format!(
                "score mate {}",
                if progress.score >= 0 { mate } else { -mate }
            )
        } else {
            format!("score cp {}", progress.score)
        };
        let time_ms = progress.elapsed.as_millis();
        let nps = (progress.nodes as f64 / progress.elapsed.as_secs_f64().max(0.000_001)) as u64;
        let pv = render_principal_variation(&progress.pv);
        if progress.pv.is_empty() {
            self.out(&format!(
                "info depth {} seldepth {} {score} nodes {} time {time_ms} nps {nps}",
                progress.depth, progress.seldepth, progress.nodes
            ))
        } else {
            self.out(&format!(
                "info depth {} seldepth {} {score} nodes {} time {time_ms} nps {nps} pv {pv}",
                progress.depth, progress.seldepth, progress.nodes
            ))
        }
    }

    fn emit_search_result(
        &self,
        result: &SearchResult,
        elapsed: Duration,
        suppress_info: bool,
    ) -> io::Result<()> {
        let time_ms = elapsed.as_millis();
        let nps = (result.nodes as f64 / elapsed.as_secs_f64().max(0.000_001)) as u64;
        let score = if result.score.abs() >= crate::search::MATE_SCORE - 256 {
            let plies = (crate::search::MATE_SCORE - result.score.abs()).max(0);
            let mate = (plies + 1) / 2;
            format!(
                "score mate {}",
                if result.score >= 0 { mate } else { -mate }
            )
        } else {
            format!("score cp {}", result.score)
        };
        let pv = render_principal_variation(&result.pv);
        if !suppress_info {
            if result.pv.is_empty() {
                self.out(&format!(
                    "info depth {} seldepth {} {score} nodes {} time {time_ms} nps {nps}",
                    result.depth, result.seldepth, result.nodes
                ))?;
            } else {
                self.out(&format!(
                    "info depth {} seldepth {} {score} nodes {} time {time_ms} nps {nps} pv {pv}",
                    result.depth, result.seldepth, result.nodes
                ))?;
            }
        }
        let bestmove = result
            .best_move
            .or_else(|| self.position.legal_moves().into_iter().next())
            .map(|mv| mv.to_string())
            .unwrap_or_else(|| "0000".into());
        self.out(&format!("bestmove {bestmove}"))
    }

    fn stop_search(&mut self) {
        if let Some(active) = &self.active_search {
            active.stop.store(true, Ordering::Relaxed);
        }
    }

    fn with_engine<F>(&self, f: F) -> io::Result<()>
    where
        F: FnOnce(&mut Engine),
    {
        let mut engine = self
            .engine
            .lock()
            .map_err(|_| io::Error::other("engine lock poisoned"))?;
        f(&mut engine);
        Ok(())
    }

    fn print_id(&self) -> io::Result<()> {
        self.out(&format!("id name {ENGINE_NAME}"))?;
        self.out(&format!("id author {ENGINE_AUTHOR}"))
    }

    fn print_options(&self) -> io::Result<()> {
        self.out(&format!(
            "option name Hash type spin default {DEFAULT_HASH_MB} min 1 max 1024"
        ))?;
        self.out("option name Clear Hash type button")?;
        self.out("option name EvalFile type string default <empty>")
    }

    fn out(&self, text: &str) -> io::Result<()> {
        let mut stdout = io::stdout().lock();
        writeln!(stdout, "{text}")?;
        stdout.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_command_tracks_history_and_official_uci_moves() {
        let (tx, _rx) = mpsc::channel();
        let mut session = UciSession::new(tx);
        session
            .handle_position("position startpos moves h2e2 h7e7")
            .unwrap();

        assert_eq!(
            session.position.to_fen(),
            "rnbakabnr/9/1c2c4/p1p1p1p1p/9/9/P1P1P1P1P/1C2C4/9/RNBAKABNR w"
        );
        assert_eq!(session.history_hashes.len(), 2);
        assert_eq!(session.move_records.len(), 2);
    }

    #[test]
    fn stop_search_sets_async_flag() {
        let (tx, _rx) = mpsc::channel();
        let mut session = UciSession::new(tx);
        let stop = Arc::new(AtomicBool::new(false));
        session.active_search = Some(ActiveSearch {
            id: 7,
            stop: Arc::clone(&stop),
            last_snapshot: None,
        });

        session.stop_search();
        assert!(stop.load(Ordering::Relaxed));
    }
}
