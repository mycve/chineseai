//! UCI match runner: ChineseAI (AZ-NNUE search) vs Pikafish.

pub mod dive_store;
pub mod dive;
pub mod opening_book;

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use crate::az::{AzNnue, AzSearchLimits, alphazero_search_with_rules};
use crate::xiangqi::{Color, Move, Position, RuleHistoryEntry, RuleOutcome};

use dive::{DiveCandidate, DiveCollector, DiveConfig, DiveSink, PlyEval, SharedDiveSink, UciEval};

#[derive(Clone, Debug, Default)]
pub struct VsPikafishResult {
    pub total_games: usize,
    /// 实际跑完的对局数（被 Ctrl+C 打断或 `--stop-after` 截断时会小于 `total_games`）。
    pub played_games: usize,
    /// 续跑时跳过的对局数。
    pub skipped_games: usize,
    /// 是否是被停止信号中断的。
    pub interrupted: bool,
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
    /// 因抽满帧数上限而提前结束的对局数。
    pub frames_collected_games: usize,
    pub abnormal_ends: Vec<VsPikafishAbnormalEnd>,
    /// 抽到的跳水局面（本次运行内去重后）。
    pub dives: Vec<DiveCandidate>,
}

#[derive(Clone, Debug)]
pub struct VsPikafishAbnormalEnd {
    pub game_index: usize,
    pub chinese_plays_red: bool,
    pub end: String,
    pub final_fen: String,
    pub position_command: String,
}

#[derive(Clone, Debug)]
pub struct VsPikafishConfig {
    pub pikafish_depth: u32,
    pub total_games: usize,
    pub max_plies: usize,
    pub simulations: usize,
    pub seed: u64,
    pub parallel_games: usize,
    pub cpuct: f32,
    pub cpuct_at_root: f32,
    pub cpuct_base: f32,
    pub cpuct_factor: f32,
    pub cpuct_base_at_root: f32,
    pub cpuct_factor_at_root: f32,
    pub fpu_value: f32,
    pub fpu_value_at_root: f32,
    pub policy_softmax_temp: f32,
    pub report_games: bool,
    /// Pikafish 引擎设置。
    pub engine: PikafishEngineConfig,
    /// 指定后启用跳水抽帧。
    pub dive: Option<PikafishDiveConfig>,
    /// 跳过前 N 局（续跑：与上次相同的 seed 会把相同局面分给相同 game_index）。
    pub skip_games: usize,
    /// 只跑 N 局后收工；0 表示跑满 `total_games`。
    pub stop_after: usize,
}

/// Pikafish 进程的启动参数。
#[derive(Clone, Debug, Default)]
pub struct PikafishEngineConfig {
    /// NNUE 权重路径；None 表示让引擎自己在工作目录里找 `pikafish.nnue`。
    pub nnue: Option<PathBuf>,
    pub threads: u32,
    pub hash_mb: u32,
    /// 关掉引擎自带开局库，避免开局阶段的 book 着法污染评价。None 表示不改这个选项。
    pub use_book: Option<bool>,
}

impl PikafishEngineConfig {
    fn apply(&self, uci: &mut ExternalUci) -> std::io::Result<()> {
        if self.threads > 0 {
            uci.set_option("Threads", &self.threads.to_string())?;
        }
        if self.hash_mb > 0 {
            uci.set_option("Hash", &self.hash_mb.to_string())?;
        }
        if let Some(nnue) = self.nnue.as_ref() {
            uci.set_option("EvalFile", &nnue.display().to_string())?;
        }
        if let Some(use_book) = self.use_book {
            uci.set_option("UseBook", if use_book { "true" } else { "false" })?;
        }
        Ok(())
    }
}

/// 跳水抽帧配置：判定阈值 + 复算深度。
#[derive(Clone, Debug, Default)]
pub struct PikafishDiveConfig {
    pub dive: DiveConfig,
    /// 复算深度；0 表示复用对局中的 `pikafish_depth` 评分，不额外搜索。
    pub analyze_depth: u32,
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
    /// 本局已经抽满帧数上限，主动收工换下一局（只有 `dive-games` 会触发）。
    FramesCollected,
}

#[derive(Clone, Debug)]
struct GameConfig {
    chinese_plays_red: bool,
    pikafish_depth: u32,
    max_plies: usize,
    simulations: usize,
    seed: u64,
    cpuct: f32,
    cpuct_at_root: f32,
    cpuct_base: f32,
    cpuct_factor: f32,
    cpuct_base_at_root: f32,
    cpuct_factor_at_root: f32,
    fpu_value: f32,
    fpu_value_at_root: f32,
    policy_softmax_temp: f32,
    dive: Option<PikafishDiveConfig>,
}

struct ExternalUci {
    child: Child,
    stdin: BufWriter<std::process::ChildStdin>,
    stdout: BufReader<std::process::ChildStdout>,
    scratch: String,
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
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| std::io::Error::other("pikafish: missing stdout"))?;
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            scratch: String::new(),
        })
    }

    fn write_line(&mut self, line: &str) -> std::io::Result<()> {
        writeln!(self.stdin, "{line}")?;
        self.stdin.flush()
    }

    fn read_line(&mut self) -> std::io::Result<String> {
        self.scratch.clear();
        if self.stdout.read_line(&mut self.scratch)? == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "pikafish: unexpected EOF",
            ));
        }
        Ok(self.scratch.trim().to_string())
    }

    fn handshake(&mut self) -> std::io::Result<()> {
        self.write_line("uci")?;
        loop {
            if self.read_line()? == "uciok" {
                break;
            }
        }
        self.wait_ready()
    }

    fn wait_ready(&mut self) -> std::io::Result<()> {
        self.write_line("isready")?;
        loop {
            if self.read_line()? == "readyok" {
                break;
            }
        }
        Ok(())
    }

    /// 在 `uci` 之后、`isready` 之前应用 UCI 选项；引擎不认的选项会被忽略。
    fn set_option(&mut self, name: &str, value: &str) -> std::io::Result<()> {
        self.write_line(&format!("setoption name {name} value {value}"))
    }

    /// 只分析、不走子：`go depth D`，读到 bestmove 就返回当前根评价。
    fn evaluate_fen(&mut self, fen: &str, depth: u32) -> std::io::Result<UciEval> {
        self.write_line(&format!("position fen {fen}"))?;
        self.write_line(&format!("go depth {depth}"))?;
        let mut eval = UciEval::default();
        loop {
            let line = self.read_line()?;
            if let Some(rest) = line.strip_prefix("bestmove ") {
                eval.bestmove = rest.split_whitespace().next().unwrap_or("").to_string();
                break;
            }
            dive::parse_info_line(&line, &mut eval);
        }
        Ok(eval)
    }

    /// 对局中的一手：既拿 Pikafish 的走法，也顺手拿到它对这个局面的评分。
    fn query_move(
        &mut self,
        initial_fen: Option<&str>,
        moves_uci: &[String],
        depth: u32,
    ) -> std::io::Result<(String, UciEval)> {
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
        let mut eval = UciEval::default();
        loop {
            let line = self.read_line()?;
            if let Some(rest) = line.strip_prefix("bestmove ") {
                let token = rest.split_whitespace().next().unwrap_or("").to_string();
                eval.bestmove = token.clone();
                return Ok((token, eval));
            }
            dive::parse_info_line(&line, &mut eval);
        }
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

/// 单局结果：终局、终局 FEN、可复现的 position 命令、本局抽到的跳水局面。
type GameOutcome = (GameEnd, String, String, Vec<DiveCandidate>);

/// 抽帧记录用的对局 id（与 `game_index` 一一对应）。
fn game_id_of(game_index: usize) -> u64 {
    (game_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xC0FF_EE00_1234_5678
}

fn play_one_game(
    model: &AzNnue,
    external: &mut ExternalUci,
    initial_position: &Position,
    config: GameConfig,
    sink: Option<Arc<Mutex<Box<dyn DiveSink>>>>,
    game_index: usize,
) -> std::io::Result<GameOutcome> {
    let _ = external.write_line("ucinewgame");
    let mut position = initial_position.clone();
    let initial_fen = (position != Position::startpos()).then(|| position.to_fen());
    let mut rule_history = position.initial_rule_history();
    let mut moves_uci: Vec<String> = Vec::new();
    let mut ply_count = 0usize;
    let mut seed = config.seed;
    let our_color = if config.chinese_plays_red {
        Color::Red
    } else {
        Color::Black
    };
    let mut collector = config.dive.clone().map(|dive| {
        let sink: Option<Box<dyn DiveSink>> = sink.map(|shared| {
            Box::new(SharedDiveSink(shared)) as Box<dyn DiveSink>
        });
        let mut collector = DiveCollector::new(dive.dive, sink);
        // 对局 id 只用于回溯这批抽帧来自哪一局。
        collector.begin_game(our_color, game_id_of(game_index));
        collector
    });
    // 我方行棋那个局面的 Pikafish 评分（"不认同"判定的另一半），由 make_eval
    // 换算到我方视角；我方搜索的 Q 在搜索完成后合并进 `our_eval`。
    let mut our_eval;
    let mut pika_eval = PlyEval::default();

    loop {
        if let Some(end) =
            terminal_before_side_selects(&position, &rule_history, ply_count, config.max_plies)
        {
            if let Some(collector) = collector.as_mut() {
                collector.flush()?;
            }
            return Ok((
                end,
                position.to_fen(),
                position_command(initial_fen.as_deref(), &moves_uci),
                collector.as_mut().map(DiveCollector::take_candidates).unwrap_or_default(),
            ));
        }

        let side = position.side_to_move();
        let legal = position.legal_moves_with_rules(&rule_history);
        let chinese_to_move = (config.chinese_plays_red && side == Color::Red)
            || (!config.chinese_plays_red && side == Color::Black);

        if chinese_to_move {
            // 先让 Pikafish 对同一个局面表态（只分析不走子），再让我们搜索。
            if let Some(collector) = collector.as_ref() {
                let eval = quiet_pikafish_eval(external, &position, &rule_history, &config);
                pika_eval = collector.make_eval(&position, None, eval);
            }
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
                    cpuct_base: config.cpuct_base,
                    cpuct_factor: config.cpuct_factor,
                    cpuct_base_at_root: config.cpuct_base_at_root,
                    cpuct_factor_at_root: config.cpuct_factor_at_root,
                    max_depth: 0,
                    root_dirichlet_alpha: 0.0,
                    root_exploration_fraction: 0.0,
                    fpu_value: config.fpu_value,
                    fpu_value_at_root: config.fpu_value_at_root,
                    fpu_absolute_at_root: true,
                    minimum_kldgain_per_node: 0.0,
                    policy_softmax_temp: config.policy_softmax_temp,
                    draw_score: 0.0,
                    value_scale: 1.0,
                },
            );
            seed = seed.wrapping_add(1);
            our_eval = pika_eval
                .clone()
                .with_ours(search.value_q, search.value_wdl, search.simulations);

            // 这里**故意不认输**：Pikafish 判定"我们已输"的那一刻正是我们要的局面。
            // 抽帧的目的就是把这些我们从未探索过的劣势局面喂回强化学习，让它自己
            // 探索到那个劣势；提前收工反而丢掉了样本。只有抽满帧数上限才收工。

            let Some(mv) = search.best_move else {
                if let Some(collector) = collector.as_mut() {
                    collector.flush()?;
                }
                return Ok((
                    match side {
                        Color::Red => GameEnd::BlackWin(GameEndReason::SearchNoMove),
                        Color::Black => GameEnd::RedWin(GameEndReason::SearchNoMove),
                    },
                    position.to_fen(),
                    position_command(initial_fen.as_deref(), &moves_uci),
                    collector.as_mut().map(DiveCollector::take_candidates).unwrap_or_default(),
                ));
            };
            let uci = mv.to_string();
            apply_move_recorded(&mut position, &mut rule_history, mv);
            moves_uci.push(uci);
        } else {
            let (token, _eval) =
                external.query_move(initial_fen.as_deref(), &moves_uci, config.pikafish_depth)?;
            let outcome = reject_pikafish_move(&position, &legal, &token, side);
            if let Some(end) = outcome {
                if let Some(collector) = collector.as_mut() {
                    collector.flush()?;
                }
                return Ok((
                    end,
                    position.to_fen(),
                    position_command(initial_fen.as_deref(), &moves_uci),
                    collector.as_mut().map(DiveCollector::take_candidates).unwrap_or_default(),
                ));
            }
            // 对手出手前那一手的"跳水"判定不需要这里的评分（我方出手时已经评过
            // 同一个局面），所以直接丢掉。
            let mv = position.parse_uci_move(&token).expect("checked above");
            apply_move_recorded(&mut position, &mut rule_history, mv);
            moves_uci.push(token);
            ply_count += 1;
            continue;
        }
        ply_count += 1;

        // 我方刚走完：判定这一手是否跳水（"不认同"取落子前局面，落差取落子前后）。
        if let Some(collector) = collector.as_mut() {
            let eval = quiet_pikafish_eval(external, &position, &rule_history, &config);
            let pika_after = collector.make_eval(&position, None, eval);
            let _ = collector.judge_ply(&position, ply_count, &our_eval, &pika_after, None)?;
            // 抽满帧数上限就收工：后面的局面是"已经崩盘之后"，不是我们要的跳水瞬间。
            if collector.frames_exhausted() {
                collector.flush()?;
                return Ok((
                    match side {
                        Color::Red => GameEnd::BlackWin(GameEndReason::FramesCollected),
                        Color::Black => GameEnd::RedWin(GameEndReason::FramesCollected),
                    },
                    position.to_fen(),
                    position_command(initial_fen.as_deref(), &moves_uci),
                    collector.take_candidates(),
                ));
            }
        }
    }
}

/// 拒绝对手给出的非法/缺失着法，返回终局原因。
fn reject_pikafish_move(
    position: &Position,
    legal: &[Move],
    token: &str,
    side: Color,
) -> Option<GameEnd> {
    let outcome = |reason: GameEndReason| {
        Some(match side {
            Color::Red => GameEnd::BlackWin(reason),
            Color::Black => GameEnd::RedWin(reason),
        })
    };
    if token.is_empty() || token == "(none)" || token == "0000" {
        return outcome(GameEndReason::PikafishNoBestMove);
    }
    let Some(mv) = position.parse_uci_move(token) else {
        return outcome(GameEndReason::PikafishInvalidMove);
    };
    if !legal.contains(&mv) {
        return outcome(GameEndReason::PikafishIllegalMove);
    }
    None
}

/// 让 Pikafish 只分析不走子，返回该局面的根评价。
///
/// `analyze_depth == 0` 表示复用对局深度，不额外加深。
fn quiet_pikafish_eval(
    external: &mut ExternalUci,
    position: &Position,
    rule_history: &[RuleHistoryEntry],
    config: &GameConfig,
) -> Option<UciEval> {
    let depth = match config.dive.as_ref() {
        Some(dive) if dive.analyze_depth > 0 => dive.analyze_depth,
        _ => config.pikafish_depth,
    };
    if position.rule_outcome_with_history(rule_history).is_some()
        || position.legal_moves_with_rules(rule_history).is_empty()
    {
        return None;
    }
    external.evaluate_fen(&position.to_fen(), depth).ok()
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
    sink: Option<Arc<Mutex<Box<dyn DiveSink>>>>,
) -> std::io::Result<VsPikafishResult> {
    let model = Arc::new(AzNnue::load(chinese_model_path).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("load chinese model `{}`: {e}", chinese_model_path.display()),
        )
    })?);

    let pikafish_path = pikafish_exe.to_path_buf();
    let engine = config.engine.clone();
    let parallel = config.parallel_games.max(1).min(config.total_games);
    let start_positions = Arc::new(start_positions.to_vec());
    let dive_count = Arc::new(AtomicUsize::new(0));

    let mut out = VsPikafishResult {
        total_games: config.total_games,
        ..Default::default()
    };

    let stop = Arc::new(AtomicBool::new(false));
    // Ctrl+C 只置位，由各 worker 在每局结束时检查：当前这局照常下完并把抽帧落库，
    // 不会留下半局数据，也不会丢已经跑完的结果。
    {
        let stop = Arc::clone(&stop);
        if let Err(err) = ctrlc::set_handler(move || {
            if stop.swap(true, Ordering::SeqCst) {
                eprintln!("vs-pikafish: 再次收到中断信号，强制退出");
                std::process::exit(130);
            }
            eprintln!("vs-pikafish: 收到中断信号，跑完当前对局后收工（再按一次强制退出）");
        }) {
            // 同一个进程里可能已经注册过（az-loop 路径），此处不致命。
            eprintln!("vs-pikafish: 跳过 Ctrl+C 注册: {err}");
        }
    }
    let played = Arc::new(AtomicUsize::new(0));
    let progress = Arc::new(ProgressCounters::new(
        config.total_games,
        config.skip_games.min(config.total_games),
    ));
    let parallel_games = parallel;
    let skip_games = config.skip_games.min(config.total_games);
    let stop_after = config.stop_after;
    // 汇报线程：每 REPORT_INTERVAL 打一次。用一个独立的 finished 标志退出——
    // 不能用 `stop`，因为正常跑完最后一批时 `stop` 不会被置位。
    let finished = Arc::new(AtomicBool::new(false));
    let reporter = {
        let progress = Arc::clone(&progress);
        let finished = Arc::clone(&finished);
        std::thread::spawn(move || {
            while !finished.load(Ordering::Relaxed) {
                std::thread::sleep(REPORT_INTERVAL);
                if finished.load(Ordering::Relaxed) {
                    break;
                }
                progress.report();
            }
        })
    };

    let mut handles = Vec::with_capacity(parallel);
    for worker_id in 0..parallel {
        let exe = pikafish_path.clone();
        let engine = engine.clone();
        let m = Arc::clone(&model);
        let positions = Arc::clone(&start_positions);
        let sink = sink.clone();
        let dive_count = Arc::clone(&dive_count);
        let config = config.clone();
        let stop = Arc::clone(&stop);
        let played = Arc::clone(&played);
        let progress = Arc::clone(&progress);
        handles.push(thread::spawn(
            move || -> std::io::Result<Vec<(usize, bool, GameOutcome)>> {
                let mut ext = ExternalUci::spawn(&exe)?;
                ext.handshake()?;
                engine.apply(&mut ext)?;
                ext.wait_ready()?;
                let mut games = Vec::new();
                for game_index in (worker_id..config.total_games).step_by(parallel_games) {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    // 续跑：同一 seed 下 game_index 与开局局面的对应关系是确定的，
                    // 所以跳过前 N 局等价于"接着上次继续"。
                    if game_index < skip_games {
                        continue;
                    }
                    let chinese_red = game_index % 2 == 0;
                    let start_position = positions
                        .get((game_index / 2) % positions.len().max(1))
                        .cloned()
                        .unwrap_or_else(Position::startpos);
                    let outcome = play_one_game(
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
                            cpuct_base: config.cpuct_base,
                            cpuct_factor: config.cpuct_factor,
                            cpuct_base_at_root: config.cpuct_base_at_root,
                            cpuct_factor_at_root: config.cpuct_factor_at_root,
                            fpu_value: config.fpu_value,
                            fpu_value_at_root: config.fpu_value_at_root,
                            policy_softmax_temp: config.policy_softmax_temp,
                            dive: config.dive.clone(),
                        },
                        sink.clone(),
                        game_index,
                    )?;
                    let stored = outcome.3.len();
                    dive_count.fetch_add(stored, Ordering::Relaxed);
                    progress.dives.fetch_add(stored, Ordering::Relaxed);
                    progress.written.fetch_add(stored, Ordering::Relaxed);
                    let done = played.fetch_add(1, Ordering::Relaxed) + 1;
                    progress.done.store(done, Ordering::Relaxed);
                    games.push((game_index, chinese_red, outcome));
                    // 全局计数判断收工。多线程下这里天然会超出至多 parallel-1 局：
                    // 判定瞬间其他 worker 手里可能刚好各有一局在跑，那是收尾，不是 bug。
                    if stop_after > 0 && done >= stop_after {
                        stop.store(true, Ordering::Relaxed);
                        break;
                    }
                }
                Ok(games)
            },
        ));
    }
    let mut worker_results = Vec::with_capacity(handles.len());
    for handle in handles {
        worker_results.push(
            handle
                .join()
                .map_err(|_| std::io::Error::other("vs-pikafish: worker thread panicked"))??,
        );
    }
    // worker 全部结束（它们已经各自 flush 过），现在收掉汇报线程并做最后一次落盘。
    finished.store(true, Ordering::Relaxed);
    let _ = reporter.join();
    progress.report();
    if let Some(sink) = sink.as_ref()
        && let Ok(mut sink) = sink.lock()
    {
        let _ = sink.flush();
    }
    for worker_games in worker_results {
        for (game_index, chinese_red, (end, final_fen, position_command, dives)) in worker_games {
            if config.report_games || should_report_final_position(end.reason()) {
                out.abnormal_ends.push(VsPikafishAbnormalEnd {
                    game_index,
                    chinese_plays_red: chinese_red,
                    end: format!("{end:?}"),
                    final_fen,
                    position_command,
                });
            }
            out.dives.extend(dives);
            if end.reason() == GameEndReason::FramesCollected {
                out.frames_collected_games += 1;
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
    // 按盲点明显程度排序，报告里最值得看的排在前面。
    out.dives.sort_by(|left, right| {
        right
            .delta_q
            .total_cmp(&left.delta_q)
            .then_with(|| left.fen.cmp(&right.fen))
    });
    out.dives.dedup_by(|left, right| left.fen == right.fen);
    out.played_games = played.load(Ordering::Relaxed);
    out.skipped_games = skip_games;
    out.interrupted = stop.load(Ordering::Relaxed);

    Ok(out)
}

/// 长跑时的进度上报间隔。worker 只做原子自增，由独立的汇报线程按时间打印，
/// 所以 16 路并发不会刷屏，也不会在热路径上加锁。
const REPORT_INTERVAL: Duration = Duration::from_secs(30);

/// 进度计数器。worker 每完成一局自增一次。
struct ProgressCounters {
    total: usize,
    skipped: usize,
    started: Instant,
    done: AtomicUsize,
    dives: AtomicUsize,
    written: AtomicUsize,
}

impl ProgressCounters {
    fn new(total: usize, skipped: usize) -> Self {
        Self {
            total,
            skipped,
            started: Instant::now(),
            done: AtomicUsize::new(0),
            dives: AtomicUsize::new(0),
            written: AtomicUsize::new(0),
        }
    }

    fn report(&self) {
        let done = self.done.load(Ordering::Relaxed);
        let absolute = self.skipped + done;
        let elapsed = self.started.elapsed().as_secs_f64().max(1.0e-6);
        let rate = absolute as f64 / elapsed;
        let eta = if rate > 0.0 {
            (self.total.saturating_sub(absolute)) as f64 / rate
        } else {
            0.0
        };
        println!(
            "dive-games: progress games={}/{} dives={} stored={} rate={:.2}/s elapsed={:.0}s eta={:.0}s",
            absolute,
            self.total,
            self.dives.load(Ordering::Relaxed),
            self.written.load(Ordering::Relaxed),
            rate,
            elapsed,
            eta
        );
    }
}

impl GameEnd {
    fn reason(self) -> GameEndReason {
        match self {
            Self::RedWin(reason) | Self::BlackWin(reason) | Self::Draw(reason) => reason,
        }
    }
}

fn should_report_final_position(reason: GameEndReason) -> bool {
    !matches!(
        reason,
        GameEndReason::NoLegalMoves | GameEndReason::FramesCollected
    )
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
            GameEndReason::MaxPlies
            | GameEndReason::SearchNoMove
            | GameEndReason::FramesCollected => {}
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
