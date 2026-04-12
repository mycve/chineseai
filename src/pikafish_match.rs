//! UCI 联机：ChineseAI（AZ-NNUE + AlphaZero MCTS）对 Pikafish；按局交替红黑并汇总胜负。

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::thread;

use crate::az::{AzNnue, AzSearchLimits, alphazero_search_with_history_and_rules};
use crate::nnue::{HistoryMove, HISTORY_PLIES};
use crate::xiangqi::{Color, Move, Position, RuleHistoryEntry, RuleOutcome};

/// 对 Pikafish 使用固定的 PUCT 常数，与 UCI 默认保持一致，便于对比。
const VS_PIKAFISH_CPUCT: f32 = 1.5;

#[derive(Clone, Debug, Default)]
pub struct VsPikafishResult {
    pub total_games: usize,
    pub chinese_wins: usize,
    pub chinese_losses: usize,
    pub draws: usize,
    pub chinese_wins_as_red: usize,
    pub chinese_wins_as_black: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GameEnd {
    RedWin,
    BlackWin,
    Draw,
}

struct ExternalUci {
    child: Child,
    stdin: BufWriter<std::process::ChildStdin>,
    stdout: BufReader<std::process::ChildStdout>,
}

impl ExternalUci {
    fn spawn(exe: &Path) -> std::io::Result<Self> {
        let mut child = Command::new(exe)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let stdin = BufWriter::new(child.stdin.take().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::Other, "pikafish: missing stdin")
        })?);
        let stdout = BufReader::new(child.stdout.take().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::Other, "pikafish: missing stdout")
        })?);
        Ok(Self {
            child,
            stdin,
            stdout,
        })
    }

    fn write_line(&mut self, line: &str) -> std::io::Result<()> {
        writeln!(self.stdin, "{line}")?;
        self.stdin.flush()
    }

    fn read_line_into(&mut self, buf: &mut String) -> std::io::Result<usize> {
        buf.clear();
        self.stdout.read_line(buf)
    }

    fn handshake(&mut self) -> std::io::Result<()> {
        self.write_line("uci")?;
        let mut buf = String::new();
        loop {
            if self.read_line_into(&mut buf)? == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "pikafish: EOF before uciok",
                ));
            }
            if buf.trim() == "uciok" {
                break;
            }
        }
        self.write_line("isready")?;
        loop {
            if self.read_line_into(&mut buf)? == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "pikafish: EOF before readyok",
                ));
            }
            if buf.trim() == "readyok" {
                break;
            }
        }
        Ok(())
    }

    fn read_bestmove_token(&mut self) -> std::io::Result<String> {
        let mut buf = String::new();
        loop {
            if self.read_line_into(&mut buf)? == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "pikafish: EOF before bestmove",
                ));
            }
            let line = buf.trim();
            if let Some(rest) = line.strip_prefix("bestmove ") {
                let token = rest.split_whitespace().next().unwrap_or("").to_string();
                return Ok(token);
            }
        }
    }

    /// 仅发送 `position startpos moves ...` 与 `go movetime`（不 `ucinewgame`，保留哈希）。
    fn query_move(&mut self, moves_uci: &[String], movetime_ms: u32) -> std::io::Result<String> {
        let pos_cmd = if moves_uci.is_empty() {
            "position startpos".to_string()
        } else {
            format!("position startpos moves {}", moves_uci.join(" "))
        };
        self.write_line(&pos_cmd)?;
        self.write_line(&format!("go movetime {movetime_ms}"))?;
        self.read_bestmove_token()
    }

    fn quit(&mut self) {
        let _ = self.write_line("quit");
        let _ = self.child.wait();
    }
}

impl Drop for ExternalUci {
    fn drop(&mut self) {
        self.quit();
    }
}

fn apply_move_recorded(
    position: &mut Position,
    history: &mut Vec<HistoryMove>,
    rule_history: &mut Vec<RuleHistoryEntry>,
    mv: Move,
) {
    if let Some(piece) = position.piece_at(mv.from as usize) {
        history.push(HistoryMove { piece, mv });
        let overflow = history.len().saturating_sub(HISTORY_PLIES);
        if overflow > 0 {
            history.drain(0..overflow);
        }
    }
    rule_history.push(position.rule_history_entry_after_move(mv));
    position.make_move(mv);
}

fn terminal_before_side_selects(
    position: &Position,
    rule_history: &[RuleHistoryEntry],
    ply_count: usize,
    max_plies: usize,
) -> Option<GameEnd> {
    if ply_count >= max_plies {
        return Some(GameEnd::Draw);
    }
    if !position.has_general(Color::Red) {
        return Some(GameEnd::BlackWin);
    }
    if !position.has_general(Color::Black) {
        return Some(GameEnd::RedWin);
    }
    if let Some(outcome) = position.rule_outcome_with_history(rule_history) {
        return Some(match outcome {
            RuleOutcome::Draw(_) => GameEnd::Draw,
            RuleOutcome::Win(c) => {
                if c == Color::Red {
                    GameEnd::RedWin
                } else {
                    GameEnd::BlackWin
                }
            }
        });
    }
    let legal = position.legal_moves_with_rules(rule_history);
    if legal.is_empty() {
        return Some(match position.side_to_move() {
            Color::Red => GameEnd::BlackWin,
            Color::Black => GameEnd::RedWin,
        });
    }
    None
}

fn play_one_game(
    model: &AzNnue,
    external: &mut ExternalUci,
    chinese_plays_red: bool,
    movetime_ms: u32,
    max_plies: usize,
    simulations: usize,
    mut seed: u64,
) -> std::io::Result<GameEnd> {
    let _ = external.write_line("ucinewgame");
    let mut position = Position::startpos();
    let mut history: Vec<HistoryMove> = Vec::new();
    let mut rule_history = position.initial_rule_history();
    let mut moves_uci: Vec<String> = Vec::new();
    let mut ply_count = 0usize;

    loop {
        if let Some(end) = terminal_before_side_selects(&position, &rule_history, ply_count, max_plies)
        {
            return Ok(end);
        }

        let side = position.side_to_move();
        let legal = position.legal_moves_with_rules(&rule_history);
        let chinese_to_move = (chinese_plays_red && side == Color::Red)
            || (!chinese_plays_red && side == Color::Black);

        if chinese_to_move {
            let search = alphazero_search_with_history_and_rules(
                &position,
                &history,
                Some(rule_history.clone()),
                Some(legal.clone()),
                model,
                AzSearchLimits {
                    simulations,
                    seed,
                    cpuct: VS_PIKAFISH_CPUCT,
                    workers: 1,
                    root_dirichlet_alpha: 0.0,
                    root_exploration_fraction: 0.0,
                },
            );
            seed = seed.wrapping_add(1);
            let Some(mv) = search.best_move else {
                return Ok(match side {
                    Color::Red => GameEnd::BlackWin,
                    Color::Black => GameEnd::RedWin,
                });
            };
            let uci = mv.to_string();
            apply_move_recorded(&mut position, &mut history, &mut rule_history, mv);
            moves_uci.push(uci);
        } else {
            let token = external.query_move(&moves_uci, movetime_ms)?;
            if token.is_empty() || token == "(none)" || token == "0000" {
                return Ok(match side {
                    Color::Red => GameEnd::BlackWin,
                    Color::Black => GameEnd::RedWin,
                });
            }
            let Some(mv) = position.parse_uci_move(&token) else {
                return Ok(match side {
                    Color::Red => GameEnd::BlackWin,
                    Color::Black => GameEnd::RedWin,
                });
            };
            if !legal.iter().any(|m| *m == mv) {
                return Ok(match side {
                    Color::Red => GameEnd::BlackWin,
                    Color::Black => GameEnd::RedWin,
                });
            }
            apply_move_recorded(&mut position, &mut history, &mut rule_history, mv);
            moves_uci.push(token);
        }
        ply_count += 1;
    }
}

/// `total_games` 局中，第 `i` 局 Chinese 执红当且仅当 `i % 2 == 0`。
///
/// `parallel_games`：同时进行的对局数（每局独立 Pikafish 子进程），用于加快总耗时。
pub fn run_vs_pikafish(
    pikafish_exe: &Path,
    chinese_model_path: &Path,
    movetime_ms: u32,
    total_games: usize,
    max_plies: usize,
    simulations: usize,
    seed: u64,
    parallel_games: usize,
) -> std::io::Result<VsPikafishResult> {
    let model = Arc::new(AzNnue::load(chinese_model_path).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("load chinese model `{}`: {e}", chinese_model_path.display()),
        )
    })?);

    let pikafish_path = pikafish_exe.to_path_buf();
    let parallel = parallel_games.max(1).min(total_games);

    let mut out = VsPikafishResult {
        total_games: total_games,
        ..Default::default()
    };

    for batch_start in (0..total_games).step_by(parallel) {
        let batch_end = (batch_start + parallel).min(total_games);
        let mut handles = Vec::with_capacity(batch_end - batch_start);
        for game_index in batch_start..batch_end {
            let exe = pikafish_path.clone();
            let m = Arc::clone(&model);
            handles.push(thread::spawn(move || -> std::io::Result<(bool, GameEnd)> {
                let mut ext = ExternalUci::spawn(&exe)?;
                ext.handshake()?;
                let chinese_red = game_index % 2 == 0;
                let end = play_one_game(
                    m.as_ref(),
                    &mut ext,
                    chinese_red,
                    movetime_ms,
                    max_plies,
                    simulations,
                    seed ^ (game_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                )?;
                Ok((chinese_red, end))
            }));
        }
        for handle in handles {
            let join = handle.join().map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "vs-pikafish: worker thread panicked",
                )
            })?;
            let (chinese_red, end) = join?;
            match (end, chinese_red) {
                (GameEnd::Draw, _) => out.draws += 1,
                (GameEnd::RedWin, true) | (GameEnd::BlackWin, false) => {
                    out.chinese_wins += 1;
                    if chinese_red {
                        out.chinese_wins_as_red += 1;
                    } else {
                        out.chinese_wins_as_black += 1;
                    }
                }
                (GameEnd::RedWin, false) | (GameEnd::BlackWin, true) => {
                    out.chinese_losses += 1;
                }
            }
        }
    }

    Ok(out)
}
