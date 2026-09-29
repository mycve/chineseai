//! Paired, timed UCI match from shuffled book FENs.
use crate::{
    opening_book::OpeningBook,
    xiangqi::{Color, Position, RuleOutcome},
};
use std::{
    collections::HashSet,
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::mpsc::{self, Receiver},
    thread,
    time::{Duration, Instant},
};

#[derive(Clone, Debug)]
pub struct TournamentConfig {
    pub chinese_exe: PathBuf,
    pub pikafish_exe: PathBuf,
    pub nnue: PathBuf,
    pub opening_book: PathBuf,
    pub opening_positions: usize,
    pub parallel_games: usize,
    pub movetime_ms: u64,
    pub max_plies: usize,
    pub seed: u64,
    pub output: PathBuf,
}

#[derive(Default, Debug)]
pub struct TournamentReport {
    pub games: usize,
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
    pub abnormal: usize,
}

#[derive(Debug)]
struct GameRecord {
    index: usize,
    opening: usize,
    chinese_red: bool,
    result: &'static str,
    reason: String,
    plies: usize,
    start_fen: String,
    final_fen: String,
    moves: String,
    chinese_ms: u128,
    pikafish_ms: u128,
}

struct Engine {
    name: &'static str,
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<io::Result<String>>,
}

impl Engine {
    fn start(name: &'static str, exe: &Path, nnue: &Path) -> io::Result<Self> {
        let mut child = Command::new(exe)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| io::Error::other("missing engine stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("missing engine stdout"))?;
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let mut engine = Self {
            name,
            child,
            stdin,
            lines: rx,
        };
        engine.send("uci")?;
        let mut has_evalfile = false;
        engine.until(Duration::from_secs(30), |line| {
            has_evalfile |= line.starts_with("option name EvalFile ");
            line == "uciok"
        })?;
        if !has_evalfile {
            return Err(io::Error::other(format!(
                "{name}: missing UCI EvalFile option"
            )));
        }
        engine.send(&format!("setoption name EvalFile value {}", nnue.display()))?;
        engine.send("isready")?;
        let mut loaded = false;
        engine.until(Duration::from_secs(60), |line| {
            loaded |= line.contains("loaded ") || line.contains("NNUE evaluation using ");
            line == "readyok"
        })?;
        // Pikafish reports the active network only after its first search.
        if name == "Pikafish" {
            engine.send("position startpos")?;
            engine.send("go depth 1")?;
            engine.until(Duration::from_secs(60), |line| {
                loaded |= line.contains("NNUE evaluation using ");
                line.starts_with("bestmove ")
            })?;
        }
        if !loaded {
            return Err(io::Error::other(format!(
                "{name}: did not confirm NNUE load"
            )));
        }
        Ok(engine)
    }

    fn send(&mut self, command: &str) -> io::Result<()> {
        writeln!(self.stdin, "{command}")?;
        self.stdin.flush()
    }
    fn until(
        &mut self,
        timeout: Duration,
        mut done: impl FnMut(&str) -> bool,
    ) -> io::Result<String> {
        let deadline = Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!("{}: UCI deadline exceeded", self.name),
                ));
            }
            let line = self.lines.recv_timeout(remaining).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!("{}: UCI response timeout/disconnect: {e}", self.name),
                )
            })??;
            if line.starts_with("info string failed to load") {
                return Err(io::Error::other(format!("{}: {line}", self.name)));
            }
            if done(line.trim()) {
                return Ok(line);
            }
        }
    }
    fn bestmove(&mut self, fen: &str, moves: &[String], movetime_ms: u64) -> io::Result<String> {
        let mut command = format!("position fen {fen}");
        if !moves.is_empty() {
            command.push_str(" moves ");
            command.push_str(&moves.join(" "));
        }
        self.send(&command)?;
        self.send(&format!("go movetime {movetime_ms}"))?;
        let line = self.until(
            Duration::from_millis(movetime_ms.saturating_mul(10).max(15_000)),
            |line| line.starts_with("bestmove "),
        )?;
        Ok(line.split_whitespace().nth(1).unwrap_or("0000").to_owned())
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn play(
    index: usize,
    position: &Position,
    chinese_red: bool,
    chinese: &mut Engine,
    pika: &mut Engine,
    config: &TournamentConfig,
) -> io::Result<GameRecord> {
    chinese.send("ucinewgame")?;
    pika.send("ucinewgame")?;
    let start_fen = position.to_fen();
    let mut pos = position.clone();
    let mut history = pos.initial_rule_history();
    let mut moves = Vec::<String>::new();
    let mut chinese_ms = 0_u128;
    let mut pikafish_ms = 0_u128;
    let mut result = "draw";
    let reason;
    loop {
        let side = pos.side_to_move();
        let winner = if !pos.has_general(Color::Red) {
            Some(Color::Black)
        } else if !pos.has_general(Color::Black) {
            Some(Color::Red)
        } else {
            None
        };
        if let Some(winner) = winner {
            result = if (winner == Color::Red) == chinese_red {
                "win"
            } else {
                "loss"
            };
            reason = "general_capture".to_owned();
            break;
        }
        if let Some(outcome) = pos.rule_outcome_with_history(&history) {
            if let RuleOutcome::Win(winner) = outcome {
                result = if (winner == Color::Red) == chinese_red {
                    "win"
                } else {
                    "loss"
                };
            }
            reason = "rule".to_owned();
            break;
        }
        let legal = pos.legal_moves_with_rules(&history);
        if legal.is_empty() {
            result = if (side == Color::Red) == chinese_red {
                "loss"
            } else {
                "win"
            };
            reason = "no_legal_moves".to_owned();
            break;
        }
        if moves.len() >= config.max_plies {
            reason = "max_plies".to_owned();
            break;
        }
        let chinese_turn = (side == Color::Red) == chinese_red;
        let engine = if chinese_turn {
            &mut *chinese
        } else {
            &mut *pika
        };
        let started = Instant::now();
        let response = engine.bestmove(&start_fen, &moves, config.movetime_ms);
        if chinese_turn {
            chinese_ms += started.elapsed().as_millis();
        } else {
            pikafish_ms += started.elapsed().as_millis();
        }
        let token = match response {
            Ok(token) => token,
            Err(error) => {
                result = "abnormal";
                reason = format!("engine_error: {}: {error}", engine.name);
                break;
            }
        };
        let Some(mv) = pos.parse_uci_move(&token) else {
            result = "abnormal";
            reason = format!("invalid_move: {}: {token}", engine.name);
            break;
        };
        if !legal.contains(&mv) {
            result = "abnormal";
            reason = format!("illegal_move: {}: {token}", engine.name);
            break;
        }
        history.push(pos.rule_history_entry_after_move(mv));
        pos.make_move(mv);
        moves.push(token);
    }
    Ok(GameRecord {
        index,
        opening: index / 2,
        chinese_red,
        result,
        reason,
        plies: moves.len(),
        start_fen,
        final_fen: pos.to_fen(),
        moves: moves.join(" "),
        chinese_ms,
        pikafish_ms,
    })
}

pub fn run(config: TournamentConfig) -> io::Result<TournamentReport> {
    if config.opening_positions == 0 || config.parallel_games == 0 || config.movetime_ms == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "opening_positions, parallel_games, movetime_ms must be positive",
        ));
    }
    let nnue = config.nnue.canonicalize()?;
    let chinese_exe = config.chinese_exe.canonicalize()?;
    let pika_exe = config.pikafish_exe.canonicalize()?;
    let mut book = OpeningBook::load(&config.opening_book, config.seed)?;
    if book.len() < config.opening_positions {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "book has {} FENs; requested {} unique openings",
                book.len(),
                config.opening_positions
            ),
        ));
    }
    let mut starts = Vec::with_capacity(config.opening_positions);
    let mut seen = HashSet::new();
    for _ in 0..book.len() {
        let position = book.next_batch(1, 0)?.pop().unwrap().position;
        if seen.insert(position.to_fen()) {
            starts.push(position);
        }
        if starts.len() == config.opening_positions {
            break;
        }
    }
    if starts.len() < config.opening_positions {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("book has only {} distinct FENs", starts.len()),
        ));
    }
    if let Some(parent) = config
        .output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut output = BufWriter::new(File::create(&config.output)?);
    writeln!(
        output,
        "game\topening\tchinese_color\tresult\treason\tplies\tchinese_ms\tpikafish_ms\tstart_fen\tfinal_fen\tmoves"
    )?;
    output.flush()?;
    let (tx, rx) = mpsc::channel::<io::Result<GameRecord>>();
    let workers = config.parallel_games.min(config.opening_positions * 2);
    let config = std::sync::Arc::new(config);
    let starts = std::sync::Arc::new(starts);
    let mut handles = Vec::new();
    for worker in 0..workers {
        let tx = tx.clone();
        let config = config.clone();
        let starts = starts.clone();
        let nnue = nnue.clone();
        let chinese_exe = chinese_exe.clone();
        let pika_exe = pika_exe.clone();
        handles.push(thread::spawn(move || {
            let mut chinese = match Engine::start("ChineseAI", &chinese_exe, &nnue) {
                Ok(engine) => engine,
                Err(error) => {
                    let _ = tx.send(Err(error));
                    return;
                }
            };
            let mut pika = match Engine::start("Pikafish", &pika_exe, &nnue) {
                Ok(engine) => engine,
                Err(error) => {
                    let _ = tx.send(Err(error));
                    return;
                }
            };
            for index in (worker..starts.len() * 2).step_by(workers) {
                let result = play(
                    index,
                    &starts[index / 2],
                    index % 2 == 0,
                    &mut chinese,
                    &mut pika,
                    &config,
                );
                let restart =
                    matches!(&result, Ok(game) if game.reason.starts_with("engine_error:"));
                if tx.send(result).is_err() {
                    break;
                }
                if restart {
                    chinese = match Engine::start("ChineseAI", &chinese_exe, &nnue) {
                        Ok(engine) => engine,
                        Err(error) => {
                            let _ = tx.send(Err(error));
                            break;
                        }
                    };
                    pika = match Engine::start("Pikafish", &pika_exe, &nnue) {
                        Ok(engine) => engine,
                        Err(error) => {
                            let _ = tx.send(Err(error));
                            break;
                        }
                    };
                }
            }
        }));
    }
    drop(tx);
    let mut report = TournamentReport::default();
    for received in rx {
        let game = received?;
        writeln!(
            output,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            game.index,
            game.opening,
            if game.chinese_red { "red" } else { "black" },
            game.result,
            game.reason,
            game.plies,
            game.chinese_ms,
            game.pikafish_ms,
            game.start_fen,
            game.final_fen,
            game.moves
        )?;
        output.flush()?;
        report.games += 1;
        if report.games % 10 == 0 {
            eprintln!(
                "uci-tournament progress: {}/{} games",
                report.games,
                starts.len() * 2
            );
        }
        match game.result {
            "win" => report.wins += 1,
            "loss" => report.losses += 1,
            "draw" => report.draws += 1,
            _ => {}
        }
        if game.reason.starts_with("engine_error:")
            || game.reason.starts_with("invalid_move:")
            || game.reason.starts_with("illegal_move:")
        {
            report.abnormal += 1;
        }
    }
    for handle in handles {
        handle
            .join()
            .map_err(|_| io::Error::other("tournament worker panicked"))?;
    }
    if report.games != starts.len() * 2 {
        return Err(io::Error::other(format!(
            "tournament incomplete: {}/{} games",
            report.games,
            starts.len() * 2
        )));
    }
    Ok(report)
}
