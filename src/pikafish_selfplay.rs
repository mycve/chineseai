//! Temporary Pikafish bootstrap: search both colors with the same external NNUE.

use crate::xiangqi::{Color, Position, RuleOutcome};
use std::{
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Write},
    path::Path,
    process::{Child, Command, Stdio},
};

pub struct SelfplayConfig {
    pub games: usize,
    pub depth: u32,
    pub max_plies: usize,
    pub opening_plies: usize,
    pub seed: u64,
}

#[derive(Default)]
pub struct SelfplaySummary {
    pub games: usize,
    pub decisive: usize,
    pub draws: usize,
    pub truncated: usize,
    pub positions: usize,
}

struct Engine {
    child: Child,
    stdin: BufWriter<std::process::ChildStdin>,
    stdout: BufReader<std::process::ChildStdout>,
}

impl Engine {
    fn spawn(exe: &Path, nnue: &Path) -> io::Result<Self> {
        let mut child = Command::new(exe)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let stdin = BufWriter::new(
            child
                .stdin
                .take()
                .ok_or_else(|| io::Error::other("missing Pikafish stdin"))?,
        );
        let stdout = BufReader::new(
            child
                .stdout
                .take()
                .ok_or_else(|| io::Error::other("missing Pikafish stdout"))?,
        );
        let mut engine = Self {
            child,
            stdin,
            stdout,
        };
        engine.send("uci")?;
        let mut has_eval_file = false;
        loop {
            let line = engine.read()?;
            has_eval_file |= line.starts_with("option name EvalFile type string");
            if line == "uciok" {
                break;
            }
        }
        if !has_eval_file {
            return Err(io::Error::other("Pikafish does not expose EvalFile"));
        }
        engine.send(&format!("setoption name EvalFile value {}", nnue.display()))?;
        engine.send("isready")?;
        loop {
            if engine.read()? == "readyok" {
                break;
            }
        }
        // Pikafish confirms loading on first search, after isready.
        engine.send("position startpos")?;
        engine.send("go depth 1")?;
        let mut confirmed = false;
        loop {
            let line = engine.read()?;
            if let Some(loaded) = line.strip_prefix("info string NNUE evaluation using ") {
                confirmed = loaded
                    .strip_prefix(nnue.to_string_lossy().as_ref())
                    .is_some_and(|suffix| suffix.starts_with(" ("));
            }
            if line.starts_with("bestmove ") {
                break;
            }
        }
        if !confirmed {
            return Err(io::Error::other(format!(
                "Pikafish did not confirm loading {}",
                nnue.display()
            )));
        }
        Ok(engine)
    }

    fn send(&mut self, line: &str) -> io::Result<()> {
        writeln!(self.stdin, "{line}")?;
        self.stdin.flush()
    }

    fn read(&mut self) -> io::Result<String> {
        let mut line = String::new();
        if self.stdout.read_line(&mut line)? == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Pikafish exited",
            ));
        }
        Ok(line.trim().to_owned())
    }

    fn bestmove(&mut self, moves: &[String], depth: u32) -> io::Result<String> {
        let mut command = "position startpos".to_owned();
        if !moves.is_empty() {
            command.push_str(" moves ");
            command.push_str(&moves.join(" "));
        }
        self.send(&command)?;
        self.send(&format!("go depth {depth}"))?;
        loop {
            if let Some(token) = self.read()?.strip_prefix("bestmove ") {
                return Ok(token.split_whitespace().next().unwrap_or("").to_owned());
            }
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.send("quit");
        let _ = self.child.wait();
    }
}

fn next_random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Writes one searched position per TSV row. `red_result` is `?` on truncated games.
pub fn generate(
    exe: &Path,
    nnue: &Path,
    output: &Path,
    config: SelfplayConfig,
) -> io::Result<SelfplaySummary> {
    if config.games == 0 || config.depth == 0 || config.max_plies == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "games, depth and max_plies must be positive",
        ));
    }
    let mut engine = Engine::spawn(exe, nnue)?;
    let mut out = BufWriter::new(File::create(output)?);
    writeln!(out, "game\tply\tfen\tbestmove\tred_result\ttermination")?;
    let mut summary = SelfplaySummary::default();
    let mut random = config.seed;
    for game in 0..config.games {
        engine.send("ucinewgame")?;
        let mut position = Position::startpos();
        let mut history = position.initial_rule_history();
        let mut moves = Vec::new();
        let mut samples = Vec::new();
        let (result, termination) = loop {
            if !position.has_general(Color::Red) {
                break ("0", "general");
            }
            if !position.has_general(Color::Black) {
                break ("1", "general");
            }
            if let Some(outcome) = position.rule_outcome_with_history(&history) {
                break match outcome {
                    RuleOutcome::Draw(_) => ("1/2", "rule"),
                    RuleOutcome::Win(Color::Red) => ("1", "rule"),
                    RuleOutcome::Win(Color::Black) => ("0", "rule"),
                };
            }
            let legal = position.legal_moves_with_rules(&history);
            if legal.is_empty() {
                break (
                    if position.side_to_move() == Color::Red {
                        "0"
                    } else {
                        "1"
                    },
                    "no_legal_moves",
                );
            }
            if moves.len() >= config.max_plies {
                break ("?", "max_plies");
            }
            let is_opening = moves.len() < config.opening_plies;
            let mv = if is_opening {
                legal[(next_random(&mut random) as usize) % legal.len()]
            } else {
                let token = engine.bestmove(&moves, config.depth)?;
                let parsed = position.parse_uci_move(&token).ok_or_else(|| {
                    io::Error::other(format!("Pikafish invalid bestmove {token}"))
                })?;
                if !legal.contains(&parsed) {
                    return Err(io::Error::other(format!(
                        "Pikafish illegal bestmove {token}"
                    )));
                }
                samples.push((moves.len(), position.to_fen(), token));
                parsed
            };
            history.push(position.rule_history_entry_after_move(mv));
            position.make_move(mv);
            moves.push(mv.to_uci());
        };
        for (ply, fen, bestmove) in &samples {
            writeln!(
                out,
                "{}\t{}\t{}\t{}\t{}\t{}",
                game + 1,
                ply,
                fen,
                bestmove,
                result,
                termination
            )?;
        }
        summary.games += 1;
        summary.positions += samples.len();
        match result {
            "?" => summary.truncated += 1,
            "1/2" => summary.draws += 1,
            _ => summary.decisive += 1,
        }
    }
    out.flush()?;
    Ok(summary)
}
