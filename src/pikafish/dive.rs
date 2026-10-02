//! 抽帧热身库：找"跳水局面"。
//!
//! 用开局库开局，让权重网络与 Pikafish 交换对弈（红黑各半），对局中逐 ply 记录
//! 双方对同一局面的评价：
//!
//! * `our_q`：我们用权重网络做 MCTS 搜索得到的根 Q（胜率视角，`W - L`）；
//! * `pika_q`：Pikafish 在同一局面 `go depth D` 的 `score cp` / `wdl`。
//!
//! 视角统一：UCI 与搜索的 Q 都是"该局面行棋方"视角。判定时把两方评价都换算到
//! **我们**的视角（对手行棋的局面取相反数），所以比较的两列符号一致。
//!
//! 一个局面被判为"跳水"要同时满足：
//!
//! 1. **不认同** `delta_cp >= --delta-cp`（默认 150）：我们和 Pikafish 对同一局面的
//!    胜率差超过阈值；
//! 2. **决定性** `max(|our_q|, |pika_q|) >= --q-floor`（默认 0.35）：至少一方认为这
//!    不是均势。这条正是"双方共同认为好或者坏的局面不放"——都不确定或都同意，就丢；
//! 3. **落差** `drop_cp >= --delta-cp`：相邻 ply 的评价落差超过阈值（真正的"掉水"）。
//!
//! 残局不抽帧：`--min-pieces` 以下子力、或 `--max-ply` 之后一律不记录。

use std::io;
use std::path::Path;
use std::sync::{Arc, Mutex};

use rusqlite::{Connection, params};

use crate::az::{AzNnue, AzSearchLimits, alphazero_search_with_rules};
use crate::xiangqi::{Color, Move, Position, RuleHistoryEntry};

use super::dive_store::{io_error, normalize_fen};

/// 一次搜索后拿到的 Pikafish 评估。
#[derive(Clone, Debug, Default)]
pub struct UciEval {
    pub bestmove: String,
    pub score_cp: Option<i32>,
    pub mate: Option<i32>,
    pub wdl: Option<[u16; 3]>,
    pub depth: u32,
    pub pv: String,
    pub nodes: u64,
}

impl UciEval {
    /// 归一化胜率视角（`W - L`）；没有 WDL 时用 `tanh(cp / 400)` 近似。
    pub fn q(&self) -> f32 {
        if let Some(wdl) = self.wdl {
            let total = f64::from(wdl[0]) + f64::from(wdl[1]) + f64::from(wdl[2]);
            if total > 0.0 {
                return ((f64::from(wdl[0]) - f64::from(wdl[2])) / total) as f32;
            }
        }
        match (self.score_cp, self.mate) {
            (_, Some(mate)) => {
                if mate > 0 {
                    1.0
                } else {
                    -1.0
                }
            }
            (Some(cp), _) => (cp as f32 / 400.0).tanh(),
            _ => 0.0,
        }
    }
}

/// 解析一行 UCI `info`，只保留我们关心的字段。
///
/// 形如：
/// `info depth 12 seldepth 20 multipv 1 score cp 6 wdl 19 980 1 nodes 33048 ... pv c3c4 h9g7`
pub fn parse_info_line(line: &str, eval: &mut UciEval) -> bool {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.len() < 3 || tokens[0] != "info" {
        return false;
    }
    // multipv > 1 的行只描述次优变化，不能当根评价。
    if let Some(index) = tokens.iter().position(|&token| token == "multipv")
        && let Some(value) = tokens.get(index + 1).and_then(|v| v.parse::<u32>().ok())
        && value > 1
    {
        return false;
    }
    if let Some(index) = tokens.iter().position(|&token| token == "depth")
        && let Some(value) = tokens.get(index + 1).and_then(|v| v.parse::<u32>().ok())
    {
        eval.depth = eval.depth.max(value);
    }
    if let Some(index) = tokens.iter().position(|&token| token == "nodes")
        && let Some(value) = tokens.get(index + 1).and_then(|v| v.parse::<u64>().ok())
    {
        eval.nodes = value;
    }
    if let Some(index) = tokens.iter().position(|&token| token == "score") {
        match (tokens.get(index + 1), tokens.get(index + 2)) {
            (Some(&"cp"), Some(value)) => {
                if let Ok(parsed) = value.parse::<i32>() {
                    eval.score_cp = Some(parsed);
                    eval.mate = None;
                }
            }
            (Some(&"mate"), Some(value)) => {
                if let Ok(parsed) = value.parse::<i32>() {
                    eval.mate = Some(parsed);
                    eval.score_cp = None;
                }
            }
            _ => {}
        }
    }
    if let Some(index) = tokens.iter().position(|&token| token == "wdl")
        && let (Some(w), Some(d), Some(l)) = (
            tokens.get(index + 1).and_then(|v| v.parse::<u16>().ok()),
            tokens.get(index + 2).and_then(|v| v.parse::<u16>().ok()),
            tokens.get(index + 3).and_then(|v| v.parse::<u16>().ok()),
        )
    {
        eval.wdl = Some([w, d, l]);
    }
    if let Some(index) = tokens.iter().position(|&token| token == "pv") {
        eval.pv = tokens[index + 1..].join(" ");
    }
    true
}

/// 一个 ply 的评价，**统一换算到我方视角**（对手行棋的局面取相反数）。
///
/// 这样跨 ply 可以直接相加："上一 ply 与这一 ply 的 `our_persp` 之和"就是
/// 我们在这两个 ply 之间净赚/净亏了多少（见 `judge_ply` 的落差定义）。
#[derive(Clone, Debug, Default)]
pub struct PlyEval {
    /// 我方搜索根评价 `(Q, WDL, 模拟数)`；只有我方行棋的局面才有。
    pub our: Option<(f32, [f32; 3], usize)>,
    /// Pikafish 对该局面的胜率，已换算到我方视角。
    pub pika_q: Option<f32>,
    /// Pikafish 原始 `score cp`，已换算到我方视角。
    pub pika_cp: Option<i32>,
    pub pika_wdl: Option<[u16; 3]>,
    pub pika_depth: u32,
    pub pika_bestmove: String,
}

impl PlyEval {
    pub fn our_q(&self) -> Option<f32> {
        self.our.map(|(q, _, _)| q)
    }

    pub fn our_wdl(&self) -> [f32; 3] {
        self.our.map(|(_, wdl, _)| wdl).unwrap_or_default()
    }

    pub fn our_sims(&self) -> usize {
        self.our.map(|(_, _, sims)| sims).unwrap_or(0)
    }

    pub fn with_ours(mut self, q: f32, wdl: [f32; 3], simulations: usize) -> Self {
        self.our = Some((q, wdl, simulations));
        self
    }
}

/// 一个抽到的局面。
///
/// **只存局面本身 + 抽帧当时的一点判定依据**。这里刻意不产出 policy / value 目标：
/// 这些 FEN 是喂给强化学习去探索的，程序要自己走一遍才能认识到这个局面已经劣势；
/// 直接塞标签那是监督学习，不是我们要的。留下 `our_q` / `pika_q` / `delta_q` 只是
/// 为了抽查抽帧质量、以及事后按阈值再筛一遍。
#[derive(Clone, Debug)]
pub struct DiveCandidate {
    /// 规范化 FEN（棋盘 + 行棋方），自博弈开局库直接用这一列。
    pub fen: String,
    /// 抽到时所处的 ply。
    pub ply: usize,
    /// 来源对局 id（对局内唯一），用于回溯是哪一局。
    pub game_id: u64,
    /// 我方视角胜率（`W - L`）。
    pub our_q: f32,
    /// Pikafish 视角胜率，已换算到我们视角。
    pub pika_q: f32,
    /// 两者的差距（胜率），即"盲点有多明显"。
    pub delta_q: f32,
}

/// 抽帧配置。
///
/// 抽的就是"**我们没意识到自己已经输了**"的局面：Pikafish（真值）判定这个局面我们
/// 已经输定，而我们的网络仍然乐观。这类局面在自博弈里从未被探索过，所以它的价值
/// 一直是错的；把 FEN 喂回去让程序自己探索到那个劣势，是修正而不是打标签。
///
/// 因此这里**不产出任何 value/policy 目标**：需要的只是"局面本身 + 判定依据"。
#[derive(Clone, Copy, Debug)]
pub struct DiveConfig {
    /// **我们没看出来**：`our_q >= lost_below`。我们的网络几乎不输出极端值
    /// （实测必败局面里也只有 -0.03 ~ -0.36），所以这条通常自动成立。
    pub lost_below: f32,
    /// **不认同**：`|our_q - pika_q| >= delta_q`。决定这个盲点有多明显。
    pub delta_q: f32,
    /// 可选附加条件：不认同同时达到这个 cp 量级（0 表示不要求）。
    pub delta_cp: i32,
    /// 可选附加条件：相邻 ply 落差达到这个胜率量级（0 表示不要求）。
    pub drop_q: f32,
    /// 可选附加条件：相邻 ply 落差达到这个 cp 量级（0 表示不要求）。
    pub drop_cp: i32,
    /// 残局排除：双方棋子总数下限（含将帅）。
    pub min_pieces: usize,
    /// 残局排除：超过这个 ply 不再抽帧。
    pub max_ply: usize,
    /// **每局最多抽多少帧**。抽够就收工换下一局。0 表示不限制。
    pub max_frames_per_game: usize,
}

impl Default for DiveConfig {
    fn default() -> Self {
        Self {
            lost_below: -0.80,
            delta_q: 0.35,
            delta_cp: 0,
            drop_q: 0.0,
            drop_cp: 0,
            min_pieces: 20,
            max_ply: 80,
            max_frames_per_game: 4,
        }
    }
}

/// 这个局面是不是"我们没意识到自己已经输了"。
///
/// `our_q` 与 `pika_q` 都在我方视角。真值说我们输（`pika_q <= lost_below`），
/// 而我们没看出来（`our_q >= lost_below`），并且两边差距够大。
pub fn is_blind_spot(our_q: f32, pika_q: f32, lost_below: f32, delta_q: f32) -> bool {
    pika_q <= lost_below && our_q >= lost_below && (our_q - pika_q).abs() >= delta_q
}

/// 落库接口：测试用内存实现，正式路径写 SQLite。
pub trait DiveSink: Send {
    fn store(&mut self, candidate: &DiveCandidate) -> io::Result<()>;
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// 多个 worker 线程共用一个 sink。
pub struct SharedDiveSink(pub Arc<Mutex<Box<dyn DiveSink>>>);

impl DiveSink for SharedDiveSink {
    fn store(&mut self, candidate: &DiveCandidate) -> io::Result<()> {
        self.0
            .lock()
            .unwrap_or_else(|_| panic!("dive sink poisoned"))
            .store(candidate)
    }


    fn flush(&mut self) -> io::Result<()> {
        self.0
            .lock()
            .unwrap_or_else(|_| panic!("dive sink poisoned"))
            .flush()
    }
}

/// 写 `dives` 表。
pub struct SqliteDiveSink {
    conn: Connection,
    written: usize,
}

impl SqliteDiveSink {
    /// 打开（必要时新建）抽帧库并建表。
    pub fn open(path: &Path) -> io::Result<Self> {
        let conn = super::dive_store::open_dive_db(path)?;
        Ok(Self { conn, written: 0 })
    }

    /// 清空历史抽帧结果，返回删除的行数。
    pub fn clear(path: &Path) -> io::Result<usize> {
        let conn = super::dive_store::open_dive_db(path)?;
        let removed = conn
            .execute("DELETE FROM dives", [])
            .map_err(io_error)?;
        conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")
            .map_err(io_error)?;
        Ok(removed)
    }

    pub fn count(path: &Path) -> io::Result<usize> {
        let conn = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .map_err(io_error)?;
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM dives", [], |row| row.get(0))
            .map_err(io_error)?;
        Ok(count as usize)
    }

    pub fn written(&self) -> usize {
        self.written
    }
}

impl DiveSink for SqliteDiveSink {
    fn store(&mut self, candidate: &DiveCandidate) -> io::Result<()> {
        self.conn
            .execute(
                "INSERT INTO dives (fen, plies, our_q, pika_q, delta_q)
                 VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(fen) DO UPDATE SET
                     delta_q = MAX(dives.delta_q, excluded.delta_q),
                     plies = MIN(dives.plies, excluded.plies)",
                params![
                    candidate.fen,
                    candidate.ply as i64,
                    candidate.our_q as f64,
                    candidate.pika_q as f64,
                    candidate.delta_q as f64,
                ],
            )
            .map_err(io_error)?;
        self.written += 1;
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        // 每局结束时调用，也保证 Ctrl+C 收尾时已写入的行真的落到主库文件。
        self.conn
            .execute_batch("PRAGMA wal_checkpoint(PASSIVE);")
            .map_err(io_error)
    }

}

/// 单局对局的抽帧器。每个 worker 线程/游戏独占一个实例。
///
/// `judge_ply` 只依赖调用方传进来的两个 ply 评价，内部不保留 ply 表，所以"落差"
/// 的 ply 编号语义可以在单元测试里逐条钉住。
pub struct DiveCollector {
    config: DiveConfig,
    sink: Option<Mutex<Box<dyn DiveSink>>>,
    our_color: Color,
    game_id: u64,
    /// 本局已抽到的帧数（egin_game 清零）。
    frames_this_game: usize,
    total_frames: usize,
    candidates: Vec<DiveCandidate>,
}

impl DiveCollector {
    pub fn new(config: DiveConfig, sink: Option<Box<dyn DiveSink>>) -> Self {
        Self {
            config,
            sink: sink.map(Mutex::new),
            our_color: Color::Red,
            game_id: 0,
            frames_this_game: 0,
            total_frames: 0,
            candidates: Vec::new(),
        }
    }

    pub fn config(&self) -> &DiveConfig {
        &self.config
    }

    /// 本局已抽到的帧数。
    pub fn frames_this_game(&self) -> usize {
        self.frames_this_game
    }

    /// 本局帧数是否已达上限（max_frames_per_game == 0 表示不限制）。
    ///
    /// 达到之后调用方应当直接结束本局：掉水之后的局面已经不是我们要的样本。
    pub fn frames_exhausted(&self) -> bool {
        self.config.max_frames_per_game > 0
            && self.frames_this_game >= self.config.max_frames_per_game
    }

    pub fn our_color(&self) -> Color {
        self.our_color
    }

    /// 拿走本局抽到的全部候选（用于汇总报告）。
    pub fn take_candidates(&mut self) -> Vec<DiveCandidate> {
        std::mem::take(&mut self.candidates)
    }

    /// 该局面是否轮到我方出手。
    pub fn is_our_turn(&self, position: &Position) -> bool {
        position.side_to_move() == self.our_color
    }

    /// 对局开始：记录我方执色与对局 id。
    pub fn begin_game(&mut self, our_color: Color, game_id: u64) {
        self.our_color = our_color;
        self.game_id = game_id;
        self.frames_this_game = 0;
        self.candidates.clear();
    }

    pub fn flush(&mut self) -> io::Result<()> {
        if let Some(sink) = self.sink.as_ref() {
            sink.lock()
                .unwrap_or_else(|_| panic!("dive sink poisoned"))
                .flush()?;
        }
        Ok(())
    }

    /// 组装一个 ply 的评价，并把我方/对手的分数统一换算到**我方**视角。
    ///
    /// * `ours`：我方搜索根评价。它总是站在**该局面行棋方**的立场（搜索的输出就是这样），
    ///   所以轮到我方时直接可用；轮到对手时（`dive-games` 里我们会替对手也搜一次）
    ///   需要取反，才能和我们自己视角的 `pika_q` 直接相加。
    /// * `pika`：Pikafish 对该局面的评估（它给的是该局面行棋方视角）。
    pub fn make_eval(
        &self,
        position: &Position,
        ours: Option<(f32, [f32; 3], usize)>,
        pika: Option<UciEval>,
    ) -> PlyEval {
        let ours_is_to_move = position.side_to_move() == self.our_color;
        let mut eval = PlyEval::default();
        if let Some((q, wdl, sims)) = ours {
            eval.our = Some(if ours_is_to_move {
                (q, wdl, sims)
            } else {
                (
                    -q,
                    [wdl[2], wdl[1], wdl[0]],
                    sims,
                )
            });
        }
        if let Some(pika) = pika {
            let sign = if ours_is_to_move { 1.0 } else { -1.0 };
            let q = pika.q() * sign;
            eval.pika_q = Some(q);
            eval.pika_cp = pika
                .score_cp
                .map(|cp| if ours_is_to_move { cp } else { -cp })
                .or(Some((q * 1000.0).round() as i32));
            eval.pika_wdl = pika.wdl;
            eval.pika_depth = pika.depth;
            eval.pika_bestmove = pika.bestmove;
        }
        eval
    }

    /// 落子后判定这一手是否构成"我们没意识到自己已经输了"的盲点。
    ///
    /// * `position`：落子**后**的局面（用于 FEN / 子力 / 行棋方）；
    /// * `ply`：落子后的 ply 计数；
    /// * `previous`：落子**前**局面（我方行棋）的评价，`our_q` 与 `pika_q` 都在我方视角；
    /// * `current`：落子**后**局面（对手行棋）的评价；
    /// * `our_on_current`：落子后局面由我们搜出的根评价（用于可选的落差条件）。
    ///
    /// 主判据都在落子**前**那个局面：真值说我们输、而我们没看出来、差距够大。
    /// 这里不产出 value/policy 目标——只有局面本身会被存下来喂给强化学习探索。
    #[allow(clippy::too_many_arguments)]
    pub fn judge_ply(
        &mut self,
        position: &Position,
        ply: usize,
        previous: &PlyEval,
        current: &PlyEval,
        our_on_current: Option<(f32, [f32; 3], usize)>,
    ) -> io::Result<Option<DiveCandidate>> {
        if ply > self.config.max_ply || piece_count(position) < self.config.min_pieces {
            return Ok(None);
        }
        let (Some(our_q), Some(pika_q)) = (previous.our_q(), previous.pika_q) else {
            return Ok(None);
        };
        // 真值说我们输、而我们没看出来、差距够大。
        if !is_blind_spot(our_q, pika_q, self.config.lost_below, self.config.delta_q) {
            return Ok(None);
        }
        let _ = (current, our_on_current);
        let delta_q = (our_q - pika_q).abs();
        let display_fen = position.to_fen();
        let candidate = DiveCandidate {
            fen: normalize_fen(&display_fen).map_err(|err| io::Error::other(err.to_string()))?,
            ply,
            game_id: self.game_id,
            our_q,
            pika_q,
            delta_q,
        };
        if let Some(sink) = self.sink.as_ref() {
            sink.lock()
                .unwrap_or_else(|_| panic!("dive sink poisoned"))
                .store(&candidate)?;
        }
        self.frames_this_game += 1;
        self.total_frames += 1;
        self.candidates.push(candidate.clone());
        Ok(Some(candidate))
    }
}

/// 残局排除：双方子力总数（含将帅）。
pub fn piece_count(position: &Position) -> usize {
    (0..90)
        .filter(|&square| position.piece_at(square).is_some())
        .count()
}

/// 以固定模拟数做一次我方搜索，返回根评价。
pub fn search_root(
    model: &AzNnue,
    position: &Position,
    rule_history: &[RuleHistoryEntry],
    legal: &[Move],
    limits: AzSearchLimits,
) -> (f32, [f32; 3], usize) {
    let result = alphazero_search_with_rules(
        position,
        Some(rule_history.to_vec()),
        Some(legal.to_vec()),
        model,
        limits,
    );
    (result.value_q, result.value_wdl, result.simulations)
}

/// 抽帧判定的单元测试：这里固定我方执红，所以偶数 ply（含 24）轮到我方。
#[cfg(test)]
mod tests {
    use super::*;

    struct MemorySink {
        stored: Vec<DiveCandidate>,
    }

    impl DiveSink for MemorySink {
        fn store(&mut self, candidate: &DiveCandidate) -> io::Result<()> {
            self.stored.push(candidate.clone());
            Ok(())
        }
    }

    /// 用目标胜率视角构造一个 Pikafish 评估（和棋 0，两端必胜/必败）。
    /// Pikafish 的 `wdl` 是千分比，和我们 WDL 头的胜率同量纲。
    fn eval_q(q: f32) -> UciEval {
        let win = ((1.0 + q) / 2.0 * 1000.0).round() as u16;
        UciEval {
            bestmove: "a0a1".into(),
            score_cp: Some((q * 1000.0).round() as i32),
            wdl: Some([win, 0, 1000 - win]),
            depth: 20,
            ..Default::default()
        }
    }

    fn position_after_plies(plies: usize) -> Position {
        let mut position = Position::startpos();
        for index in 0..plies {
            let legal = position.legal_moves();
            let mv = legal[(index * 7 + 3) % legal.len()];
            position.make_move(mv);
        }
        position
    }

    fn collector_with(sink: Option<Box<dyn DiveSink>>) -> DiveCollector {
        let mut collector = DiveCollector::new(DiveConfig::default(), sink);
        collector.begin_game(Color::Red, 7);
        collector
    }

    /// 构造"我方出手前"那个局面的评价（双方都在我方视角）。
    fn our_turn(our_q: f32, pika_q: f32) -> PlyEval {
        let position = position_after_plies(24);
        let collector = DiveCollector::new(DiveConfig::default(), None);
        collector.make_eval(
            &position,
            Some((our_q, [0.5, 0.0, 0.5], 10_000)),
            Some(eval_q(pika_q)),
        )
    }

    /// 判定一个 (our_q, pika_q) 是否会被抽中。
    fn judges(our_q: f32, pika_q: f32) -> bool {
        let mut collector = collector_with(None);
        let after = position_after_plies(25);
        let previous = our_turn(our_q, pika_q);
        let current = collector.make_eval(&after, None, Some(eval_q(0.0)));
        collector
            .judge_ply(&after, 25, &previous, &current, None)
            .unwrap()
            .is_some()
    }

    #[test]
    fn parses_depth_score_wdl_and_pv_from_info_lines() {
        let line = "info depth 12 seldepth 20 multipv 1 score cp 6 wdl 19 980 1 nodes 33048 nps 1502181 hashfull 9 tbhits 0 time 22 pv c3c4 h9g7 g3g4";
        let mut eval = UciEval::default();
        assert!(parse_info_line(line, &mut eval));
        assert_eq!(eval.depth, 12);
        assert_eq!(eval.score_cp, Some(6));
        assert_eq!(eval.wdl, Some([19, 980, 1]));
        assert_eq!(eval.nodes, 33048);
        assert_eq!(eval.pv, "c3c4 h9g7 g3g4");
        assert!((eval.q() - (19.0 - 1.0) / 1000.0).abs() < 1.0e-6);
    }

    #[test]
    fn ignores_multipv_secondary_lines_and_reads_mate() {
        let mut eval = UciEval::default();
        parse_info_line(
            "info depth 8 multipv 2 score cp 300 wdl 700 300 0 pv a0a1",
            &mut eval,
        );
        assert_eq!(eval.score_cp, None);
        assert!(parse_info_line("info depth 9 score mate -3 pv a0a1", &mut eval));
        assert_eq!(eval.mate, Some(-3));
        assert_eq!(eval.score_cp, None);
        assert!((eval.q() + 1.0).abs() < 1.0e-6);
    }

    /// 判据以 `wdl` 为准：cp 与 wdl 冲突时用 wdl。
    #[test]
    fn q_comes_from_wdl_not_cp() {
        let contradictory = UciEval {
            score_cp: Some(900),
            wdl: Some([0, 0, 1000]),
            ..Default::default()
        };
        assert!((contradictory.q() + 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn mate_score_saturates_q_without_wdl() {
        assert!(UciEval::default().q().abs() < 1.0e-6);
        let mate = UciEval {
            mate: Some(4),
            score_cp: None,
            ..Default::default()
        };
        assert!((mate.q() - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn resolves_whose_turn_it_is_from_the_colour() {
        let red = collector_with(None);
        assert!(red.is_our_turn(&position_after_plies(24)));
        assert!(!red.is_our_turn(&position_after_plies(25)));
        let mut black = DiveCollector::new(DiveConfig::default(), None);
        black.begin_game(Color::Black, 7);
        assert!(black.is_our_turn(&position_after_plies(25)));
        assert!(!black.is_our_turn(&position_after_plies(24)));
    }

    /// `make_eval` 把对手行棋局面的双方评分都翻到我方视角。
    #[test]
    fn make_eval_normalizes_both_sides_to_our_perspective() {
        let collector = collector_with(None);
        let ours = collector.make_eval(&position_after_plies(24), None, Some(eval_q(0.6)));
        assert!((ours.pika_q.unwrap() - 0.6).abs() < 1.0e-3);
        let theirs = collector.make_eval(&position_after_plies(25), None, Some(eval_q(0.7)));
        assert!((theirs.pika_q.unwrap() + 0.7).abs() < 1.0e-3);
        let flipped = collector.make_eval(
            &position_after_plies(25),
            Some((0.5, [0.8, 0.1, 0.1], 10)),
            None,
        );
        let (q, wdl, sims) = flipped.our.unwrap();
        assert!((q + 0.5).abs() < 1.0e-6);
        assert_eq!(wdl, [0.1, 0.1, 0.8]);
        assert_eq!(sims, 10);
    }

    /// 盲点定义：真值说我们输、我们却乐观。
    #[test]
    fn blind_spot_definition() {
        let lost = -0.80;
        let delta = 0.35;
        // 实测形态：Pikafish -0.98，我们 -0.03 —— 典型盲点。
        assert!(is_blind_spot(-0.03, -0.98, lost, delta));
        // 我们也看出来输了：不是盲点（那是"承认"之后的局面）。
        assert!(!is_blind_spot(-0.95, -0.98, lost, delta));
        // 真值认为还没输定。
        assert!(!is_blind_spot(0.20, -0.60, lost, delta));
        // 差距不够大。
        assert!(!is_blind_spot(-0.55, -0.85, lost, delta));
    }

    /// 抽帧：Pikafish 说我们输定、我们没看出来、差距够大。
    #[test]
    fn flags_the_blind_spot() {
        let mut collector = collector_with(Some(Box::new(MemorySink { stored: Vec::new() })));
        let after = position_after_plies(25);
        let previous = our_turn(-0.05, -0.98);
        let current = collector.make_eval(&after, None, Some(eval_q(0.0)));
        let candidate = collector
            .judge_ply(&after, 25, &previous, &current, None)
            .unwrap()
            .expect("盲点应当抽帧");
        assert!((candidate.our_q + 0.05).abs() < 1.0e-3);
        assert!((candidate.pika_q + 0.98).abs() < 1.0e-3);
        assert!((candidate.delta_q - 0.93).abs() < 1.0e-3);
        assert_eq!(candidate.ply, 25);
        assert_eq!(candidate.game_id, 7);
        assert!(candidate.fen.ends_with(" b"), "落子后轮到对手: {}", candidate.fen);
        assert_eq!(collector.frames_this_game(), 1);
        assert_eq!(collector.take_candidates().len(), 1);
        assert_eq!(collector.take_candidates().len(), 0);
    }

    /// 我们已经承认劣势（our_q 也低于阈值）：不抽。
    #[test]
    fn skips_positions_we_already_recognize_as_lost() {
        assert!(!judges(-0.95, -0.99));
    }

    /// 真值认为还没输定：不抽。
    #[test]
    fn skips_positions_the_truth_still_calls_playable() {
        assert!(!judges(-0.02, -0.60));
        assert!(!judges(0.30, -0.10));
    }

    /// 双方都认为我们大优：不抽。
    #[test]
    fn skips_positions_where_both_sides_agree_we_are_winning() {
        assert!(!judges(0.90, 0.88));
    }

    /// 没有我方搜索结果时不判定（无从比较盲点）。
    #[test]
    fn skips_positions_without_our_search() {
        let mut collector = collector_with(None);
        let after = position_after_plies(25);
        let previous = collector.make_eval(&position_after_plies(24), None, Some(eval_q(-0.98)));
        let current = collector.make_eval(&after, None, Some(eval_q(0.0)));
        assert!(
            collector
                .judge_ply(&after, 25, &previous, &current, None)
                .unwrap()
                .is_none()
        );
    }

    /// 子力太少（残局）不抽帧。
    #[test]
    fn skips_endgame_material() {
        let mut collector = collector_with(None);
        let position = Position::from_fen("4k4/9/9/9/9/9/9/9/4P4/4K4 w").unwrap();
        assert!(piece_count(&position) < collector.config().min_pieces);
        let previous = our_turn(-0.05, -0.98);
        let current = collector.make_eval(&position, None, Some(eval_q(0.0)));
        assert!(
            collector
                .judge_ply(&position, 25, &previous, &current, None)
                .unwrap()
                .is_none()
        );
    }

    /// `max_ply` 是候选局面的 ply 上限：80 仍然抽，81 不抽。
    #[test]
    fn skips_plies_beyond_the_limit() {
        let mut collector = collector_with(None);
        let after = position_after_plies(25);
        let previous = our_turn(-0.05, -0.98);
        let current = collector.make_eval(&after, None, Some(eval_q(0.0)));
        assert!(
            collector
                .judge_ply(&after, 80, &previous, &current, None)
                .unwrap()
                .is_some()
        );
        assert!(
            collector
                .judge_ply(&after, 81, &previous, &current, None)
                .unwrap()
                .is_none()
        );
    }

    /// 帧上限：抽够 max_frames_per_game 之后 frames_exhausted() 为真，
    /// 调用方据此收工；0 表示不限制。
    #[test]
    fn frame_budget_stops_the_game() {
        let mut collector = DiveCollector::new(
            DiveConfig {
                max_frames_per_game: 3,
                ..DiveConfig::default()
            },
            None,
        );
        collector.begin_game(Color::Red, 7);
        let after = position_after_plies(25);
        assert!(!collector.frames_exhausted());
        for expected in 1..=3usize {
            let previous = our_turn(-0.05, -0.98);
            let current = collector.make_eval(&after, None, Some(eval_q(0.0)));
            assert!(
                collector
                    .judge_ply(&after, 25, &previous, &current, None)
                    .unwrap()
                    .is_some()
            );
            assert_eq!(collector.frames_this_game(), expected);
            assert_eq!(collector.frames_exhausted(), expected >= 3);
        }
        collector.begin_game(Color::Red, 8);
        assert_eq!(collector.frames_this_game(), 0);
        assert!(!collector.frames_exhausted());

        let mut unlimited = DiveCollector::new(
            DiveConfig {
                max_frames_per_game: 0,
                ..DiveConfig::default()
            },
            None,
        );
        unlimited.begin_game(Color::Red, 7);
        for _ in 0..10 {
            let previous = our_turn(-0.05, -0.98);
            let current = unlimited.make_eval(&after, None, Some(eval_q(0.0)));
            let _ = unlimited
                .judge_ply(&after, 25, &previous, &current, None)
                .unwrap();
        }
        assert_eq!(unlimited.frames_this_game(), 10);
        assert!(!unlimited.frames_exhausted());
    }

    #[test]
    fn piece_count_tracks_the_start_position() {
        assert_eq!(piece_count(&Position::startpos()), 32);
        assert_eq!(piece_count(&position_after_plies(1)), 32);
    }
}
