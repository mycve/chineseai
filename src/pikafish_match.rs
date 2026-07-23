//! UCI match runner: ChineseAI (AZ-NNUE search) vs Pikafish.

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::thread;

use crate::az::{AzNnue, AzSearchLimits, alphazero_search_with_rules};
use crate::xiangqi::{Color, Move, Position, RuleHistoryEntry, RuleOutcome};

#[derive(Clone, Debug, Default)]
pub struct VsPikafishResult {
    pub total_games: usize,
    pub chinese_wins: usize,
    pub chinese_losses: usize,
    pub draws: usize,
    pub chinese_wins_as_red: usize,
    pub chinese_wins_as_black: usize,
    pub chinese_win_by_general_capture: usize,
    pub chinese_win_by_no_legal_moves: usize,
    pub chinese_win_by_rule: usize,
    pub chinese_win_by_pikafish_no_bestmove: usize,
    pub chinese_win_by_pikafish_invalid_move: usize,
    pub chinese_win_by_pikafish_illegal_move: usize,
    pub abnormal_ends: Vec<VsPikafishAbnormalEnd>,
}

#[derive(Clone, Debug)]
pub struct VsPikafishAbnormalEnd {
    pub game_index: usize,
    pub chinese_plays_red: bool,
    pub end: String,
    pub final_fen: String,
    pub position_command: String,
}

#[derive(Clone, Copy, Debug)]
pub struct VsPikafishConfig {
    pub pikafish_depth: u32,
    pub total_games: usize,
    pub max_plies: usize,
    pub simulations: usize,
    pub seed: u64,
    pub parallel_games: usize,
    pub cpuct: f32,
    pub cpuct_at_root: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GameEnd {
    RedWin(GameEndReason),
    BlackWin(GameEndReason),
    Draw(GameEndReason),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GameEndReason {
    GeneralCaptured,
    NoLegalMoves,
    Rule,
    MaxPlies,
    SearchNoMove,
    PikafishNoBestMove,
    PikafishInvalidMove,
    PikafishIllegalMove,
}

#[derive(Clone, Copy, Debug)]
struct GameConfig {
    chinese_plays_red: bool,
    pikafish_depth: u32,
    max_plies: usize,
    simulations: usize,
    seed: u64,
    cpuct: f32,
    cpuct_at_root: f32,
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
        let stdin = BufWriter::new(
            child
                .stdin
                .take()
                .ok_or_else(|| std::io::Error::other("pikafish: missing stdin"))?,
        );
        let stdout = BufReader::new(
            child
                .stdout
                .take()
                .ok_or_else(|| std::io::Error::other("pikafish: missing stdout"))?,
        );
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

    fn query_move(
        &mut self,
        initial_fen: Option<&str>,
        moves_uci: &[String],
        depth: u32,
    ) -> std::io::Result<String> {
        let mut pos_cmd = if let Some(fen) = initial_fen {
            format!("position fen {fen}")
        } else {
            "position startpos".to_string()
        };
        if !moves_uci.is_empty() {
            pos_cmd.push_str(" moves ");
            pos_cmd.push_str(&moves_uci.join(" "));
        }
        self.write_line(&pos_cmd)?;
        self.write_line(&format!("go depth {depth}"))?;
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
    rule_history: &mut Vec<RuleHistoryEntry>,
    mv: Move,
) {
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
        return Some(GameEnd::Draw(GameEndReason::MaxPlies));
    }
    if !position.has_general(Color::Red) {
        return Some(GameEnd::BlackWin(GameEndReason::GeneralCaptured));
    }
    if !position.has_general(Color::Black) {
        return Some(GameEnd::RedWin(GameEndReason::GeneralCaptured));
    }
    if let Some(outcome) = position.rule_outcome_with_history(rule_history) {
        return Some(match outcome {
            RuleOutcome::Draw(_) => GameEnd::Draw(GameEndReason::Rule),
            RuleOutcome::Win(c) => {
                if c == Color::Red {
                    GameEnd::RedWin(GameEndReason::Rule)
                } else {
                    GameEnd::BlackWin(GameEndReason::Rule)
                }
            }
        });
    }
    let legal = position.legal_moves_with_rules(rule_history);
    if legal.is_empty() {
        return Some(match position.side_to_move() {
            Color::Red => GameEnd::BlackWin(GameEndReason::NoLegalMoves),
            Color::Black => GameEnd::RedWin(GameEndReason::NoLegalMoves),
        });
    }
    None
}

fn play_one_game(
    model: &AzNnue,
    external: &mut ExternalUci,
    initial_position: &Position,
    config: GameConfig,
) -> std::io::Result<(GameEnd, String, String)> {
    let _ = external.write_line("ucinewgame");
    let mut position = initial_position.clone();
    let initial_fen =
        (position.to_fen() != crate::xiangqi::STARTPOS_FEN).then(|| position.to_fen());
    let mut rule_history = position.initial_rule_history();
    let mut moves_uci: Vec<String> = Vec::new();
    let mut ply_count = 0usize;
    let mut seed = config.seed;

    loop {
        if let Some(end) =
            terminal_before_side_selects(&position, &rule_history, ply_count, config.max_plies)
        {
            return Ok((
                end,
                position.to_fen(),
                position_command(initial_fen.as_deref(), &moves_uci),
            ));
        }

        let side = position.side_to_move();
        let legal = position.legal_moves_with_rules(&rule_history);
        let chinese_to_move = (config.chinese_plays_red && side == Color::Red)
            || (!config.chinese_plays_red && side == Color::Black);

        if chinese_to_move {
            let search = alphazero_search_with_rules(
                &position,
                Some(rule_history.clone()),
                Some(legal.clone()),
                model,
                AzSearchLimits {
                    simulations: config.simulations,
                    seed,
                    cpuct: config.cpuct,
                    cpuct_at_root: config.cpuct_at_root,
                    cpuct_base: 19652.0,
                    cpuct_factor: 2.0,
                    cpuct_base_at_root: 19652.0,
                    cpuct_factor_at_root: 2.0,
                    max_depth: 0,
                    root_dirichlet_alpha: 0.0,
                    root_exploration_fraction: 0.0,
                    fpu_value: 0.23,
                    fpu_value_at_root: 1.0,
                    draw_score: 0.0,
                    moves_left_max_effect: 0.0,
                    moves_left_slope: 0.0,
                    moves_left_threshold: 0.6,
                    moves_left_constant_factor: 0.0,
                    moves_left_scaled_factor: 0.0,
                    moves_left_quadratic_factor: 0.0,
                    value_scale: 1.0,
                },
            );
            seed = seed.wrapping_add(1);
            let Some(mv) = search.best_move else {
                return Ok((
                    match side {
                        Color::Red => GameEnd::BlackWin(GameEndReason::SearchNoMove),
                        Color::Black => GameEnd::RedWin(GameEndReason::SearchNoMove),
                    },
                    position.to_fen(),
                    position_command(initial_fen.as_deref(), &moves_uci),
                ));
            };
            let uci = mv.to_string();
            apply_move_recorded(&mut position, &mut rule_history, mv);
            moves_uci.push(uci);
        } else {
            let token =
                external.query_move(initial_fen.as_deref(), &moves_uci, config.pikafish_depth)?;
            if token.is_empty() || token == "(none)" || token == "0000" {
                return Ok((
                    match side {
                        Color::Red => GameEnd::BlackWin(GameEndReason::PikafishNoBestMove),
                        Color::Black => GameEnd::RedWin(GameEndReason::PikafishNoBestMove),
                    },
                    position.to_fen(),
                    position_command(initial_fen.as_deref(), &moves_uci),
                ));
            }
            let Some(mv) = position.parse_uci_move(&token) else {
                return Ok((
                    match side {
                        Color::Red => GameEnd::BlackWin(GameEndReason::PikafishInvalidMove),
                        Color::Black => GameEnd::RedWin(GameEndReason::PikafishInvalidMove),
                    },
                    position.to_fen(),
                    position_command(initial_fen.as_deref(), &moves_uci),
                ));
            };
            if !legal.contains(&mv) {
                return Ok((
                    match side {
                        Color::Red => GameEnd::BlackWin(GameEndReason::PikafishIllegalMove),
                        Color::Black => GameEnd::RedWin(GameEndReason::PikafishIllegalMove),
                    },
                    position.to_fen(),
                    position_command(initial_fen.as_deref(), &moves_uci),
                ));
            }
            apply_move_recorded(&mut position, &mut rule_history, mv);
            moves_uci.push(token);
        }
        ply_count += 1;
    }
}

/// ChineseAI plays Red in even-indexed games and Black in odd-indexed games.
///
/// `parallel_games` is the number of long-lived Pikafish worker processes.
/// Each worker keeps one UCI child alive and plays multiple assigned games.
pub fn run_vs_pikafish(
    pikafish_exe: &Path,
    chinese_model_path: &Path,
    start_positions: &[Position],
    config: VsPikafishConfig,
) -> std::io::Result<VsPikafishResult> {
    let model = Arc::new(AzNnue::load(chinese_model_path).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("load chinese model `{}`: {e}", chinese_model_path.display()),
        )
    })?);

    let pikafish_path = pikafish_exe.to_path_buf();
    let parallel = config.parallel_games.max(1).min(config.total_games);
    let start_positions = Arc::new(start_positions.to_vec());

    let mut out = VsPikafishResult {
        total_games: config.total_games,
        ..Default::default()
    };

    let mut handles = Vec::with_capacity(parallel);
    for worker_id in 0..parallel {
        let exe = pikafish_path.clone();
        let m = Arc::clone(&model);
        let positions = Arc::clone(&start_positions);
        handles.push(thread::spawn(
            move || -> std::io::Result<Vec<(usize, bool, GameEnd, String, String)>> {
                let mut ext = ExternalUci::spawn(&exe)?;
                ext.handshake()?;
                let mut games = Vec::new();
                for game_index in (worker_id..config.total_games).step_by(parallel) {
                    let chinese_red = game_index % 2 == 0;
                    let start_position = positions
                        .get(game_index % positions.len().max(1))
                        .cloned()
                        .unwrap_or_else(Position::startpos);
                    let (end, final_fen, position_command) = play_one_game(
                        m.as_ref(),
                        &mut ext,
                        &start_position,
                        GameConfig {
                            chinese_plays_red: chinese_red,
                            pikafish_depth: config.pikafish_depth,
                            max_plies: config.max_plies,
                            simulations: config.simulations,
                            seed: config.seed
                                ^ (game_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                            cpuct: config.cpuct,
                            cpuct_at_root: config.cpuct_at_root,
                        },
                    )?;
                    games.push((game_index, chinese_red, end, final_fen, position_command));
                }
                Ok(games)
            },
        ));
    }
    for handle in handles {
        let worker_games = handle
            .join()
            .map_err(|_| std::io::Error::other("vs-pikafish: worker thread panicked"))??;
        for (game_index, chinese_red, end, final_fen, position_command) in worker_games {
            if should_report_final_position(end.reason()) {
                out.abnormal_ends.push(VsPikafishAbnormalEnd {
                    game_index,
                    chinese_plays_red: chinese_red,
                    end: format!("{end:?}"),
                    final_fen,
                    position_command,
                });
            }
            match (end, chinese_red) {
                (GameEnd::Draw(_), _) => out.draws += 1,
                (GameEnd::RedWin(reason), true) | (GameEnd::BlackWin(reason), false) => {
                    out.chinese_wins += 1;
                    out.record_chinese_win_reason(reason);
                    if chinese_red {
                        out.chinese_wins_as_red += 1;
                    } else {
                        out.chinese_wins_as_black += 1;
                    }
                }
                (GameEnd::RedWin(_), false) | (GameEnd::BlackWin(_), true) => {
                    out.chinese_losses += 1;
                }
            }
        }
    }
    out.abnormal_ends.sort_by_key(|item| item.game_index);

    Ok(out)
}

impl GameEnd {
    fn reason(self) -> GameEndReason {
        match self {
            Self::RedWin(reason) | Self::BlackWin(reason) | Self::Draw(reason) => reason,
        }
    }
}

fn should_report_final_position(reason: GameEndReason) -> bool {
    !matches!(reason, GameEndReason::NoLegalMoves)
}

impl VsPikafishResult {
    fn record_chinese_win_reason(&mut self, reason: GameEndReason) {
        match reason {
            GameEndReason::GeneralCaptured => self.chinese_win_by_general_capture += 1,
            GameEndReason::NoLegalMoves => self.chinese_win_by_no_legal_moves += 1,
            GameEndReason::Rule => self.chinese_win_by_rule += 1,
            GameEndReason::PikafishNoBestMove => self.chinese_win_by_pikafish_no_bestmove += 1,
            GameEndReason::PikafishInvalidMove => self.chinese_win_by_pikafish_invalid_move += 1,
            GameEndReason::PikafishIllegalMove => self.chinese_win_by_pikafish_illegal_move += 1,
            GameEndReason::MaxPlies | GameEndReason::SearchNoMove => {}
        }
    }
}

fn position_command(initial_fen: Option<&str>, moves_uci: &[String]) -> String {
    let mut command = if let Some(fen) = initial_fen {
        format!("position fen {fen}")
    } else {
        "position startpos".to_string()
    };
    if !moves_uci.is_empty() {
        command.push_str(" moves ");
        command.push_str(&moves_uci.join(" "));
    }
    command
}
