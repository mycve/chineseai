//! Pikafish 浮点网络的持久化自博弈、回放训练与冠军晋级循环。
use crate::{
    ab::{
        SplitMix64,
        pikafish_candle::{PikafishCpuModel, PikafishExample, PikafishModel},
    },
    opening_book::OpeningBook,
    pikafish_candidate_arena::{CandidateArenaDecision, CandidateArenaResult},
    pikafish_pretrain::{cp_to_value, train_batch},
    xiangqi::Position,
};
use candle_core::Device;
use candle_nn::{Optimizer, SGD};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashSet, VecDeque},
    fs::{self, File},
    io::{self, BufRead, BufReader, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
mod pipeline;
use pipeline::{ArenaOutcome, ArenaService, ArenaTask, Champion, PipelineStats, SelfplayService};

pub const DEFAULT_CONFIG: &str = "pikafish-evolve.toml";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct EvolveConfig {
    pub output_dir: PathBuf,
    pub seed_model: Option<PathBuf>,
    #[serde(default)]
    pub bootstrap_sqlite: Option<PathBuf>,
    pub bootstrap_samples: usize,
    pub bootstrap_epochs: usize,
    pub opening_book: PathBuf,
    pub games_per_update: usize,
    pub selfplay_workers: usize,
    pub arena_workers: usize,
    pub queue_games: usize,
    pub max_champion_lag: usize,
    pub selfplay_nodes: usize,
    pub arena_nodes: usize,
    pub max_depth: usize,
    pub max_plies: usize,
    pub temperature: crate::pikafish_candidate_selfplay::SelfplayTemperature,
    pub replay_capacity: usize,
    pub train_samples_per_update: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub arena_interval: usize,
    pub arena_pairs: usize,
    pub promotion_rate: f32,
    pub confidence_z: f32,
    pub seed: u64,
    pub cuda: bool,
}

impl Default for EvolveConfig {
    fn default() -> Self {
        Self {
            output_dir: "runs/pikafish-evolve-temperature".into(),
            seed_model: None,
            bootstrap_sqlite: Some("eval/pikafish-selfplay-5000-d20.sqlite".into()),
            bootstrap_samples: 2048,
            bootstrap_epochs: 4,
            opening_book: "book.pgn.gz".into(),
            games_per_update: 8,
            selfplay_workers: num_cpus::get_physical().saturating_sub(4).clamp(1, 16),
            arena_workers: num_cpus::get_physical().clamp(1, 4),
            queue_games: 24,
            max_champion_lag: 1,
            selfplay_nodes: 256,
            arena_nodes: 256,
            max_depth: 8,
            max_plies: 600,
            temperature: Default::default(),
            replay_capacity: 50_000,
            train_samples_per_update: 2048,
            batch_size: 16,
            learning_rate: 0.001,
            arena_interval: 5,
            arena_pairs: 32,
            promotion_rate: 0.55,
            confidence_z: 1.96,
            seed: 20261001,
            cuda: true,
        }
    }
}

fn io_error(error: impl std::fmt::Display) -> io::Error {
    io::Error::other(error.to_string())
}

fn path_error(label: &str, path: &Path, error: impl std::fmt::Display) -> io::Error {
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir().unwrap_or_default().join(path)
    };
    io_error(format!("{label} {}：{error}", absolute.display()))
}

impl EvolveConfig {
    pub fn load_or_create(path: &Path) -> io::Result<Self> {
        let config: Self = if path.exists() {
            toml::from_str(&fs::read_to_string(path)?).map_err(io_error)?
        } else {
            let config = Self::default();
            fs::write(path, toml::to_string_pretty(&config).map_err(io_error)?)?;
            config
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> io::Result<()> {
        self.temperature.validate()?;
        if [
            self.games_per_update,
            self.selfplay_workers,
            self.arena_workers,
            self.queue_games,
            self.selfplay_nodes,
            self.arena_nodes,
            self.max_depth,
            self.max_plies,
            self.replay_capacity,
            self.train_samples_per_update,
            self.batch_size,
            self.arena_interval,
            self.bootstrap_samples,
            self.bootstrap_epochs,
        ]
        .contains(&0)
            || self.arena_pairs < 8
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.0
            || !self.promotion_rate.is_finite()
            || !(0.5..1.0).contains(&self.promotion_rate)
            || !self.confidence_z.is_finite()
            || self.confidence_z < 1.96
        {
            return Err(io_error(
                "训练参数必须为正；晋级至少 8 对开局、得分门槛 >=0.5、置信 z>=1.96",
            ));
        }
        Ok(())
    }

    fn check_inputs(&self) -> io::Result<()> {
        let check = |label, path: &Path| {
            File::open(path)
                .map(drop)
                .map_err(|error| path_error(label, path, error))
        };
        check("无法读取 opening_book", &self.opening_book)?;
        if !self.output_dir.join("progress.toml").exists() {
            if let Some(path) = &self.seed_model {
                check("无法读取 seed_model", path)?;
            }
            if let Some(path) = &self.bootstrap_sqlite {
                check("无法读取 bootstrap_sqlite", path)?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct ReplayRow {
    fen: String,
    target: f32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Metric {
    pub kind: String,
    pub cycle: usize,
    pub update: usize,
    pub generation: usize,
    pub games: usize,
    pub completed: usize,
    pub truncated: usize,
    pub new_samples: usize,
    pub replay_samples: usize,
    pub trained_samples: usize,
    pub bootstrap_samples: usize,
    pub loss: Option<f64>,
    pub score: Option<f32>,
    pub lower: Option<f32>,
    pub upper: Option<f32>,
    pub champion_wins: usize,
    pub champion_losses: usize,
    pub champion_draws: usize,
    #[serde(default)]
    pub champion_truncated: usize,
    pub reference_score: Option<f32>,
    pub reference_lower: Option<f32>,
    pub reference_upper: Option<f32>,
    pub reference_wins: usize,
    pub reference_losses: usize,
    pub reference_draws: usize,
    #[serde(default)]
    pub reference_truncated: usize,
    pub decision: String,
    pub seconds: f64,
    pub data_wait_seconds: f64,
    pub training_seconds: f64,
    pub search_nodes_during_training: u64,
    pub trained_during_arena: bool,
    pub peak_selfplay_workers: usize,
    pub queue_depth: usize,
    pub discarded_stale_games: usize,
    pub source_generation_min: usize,
    pub source_generation_max: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PendingArena {
    pub cycle: usize,
    pub update: usize,
    pub champion_cycle: usize,
    pub champion_generation: usize,
    pub openings: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Progress {
    version: u32,
    config: EvolveConfig,
    cycle: usize,
    update: usize,
    generation: usize,
    champion_cycle: usize,
    bootstrap_done: bool,
    reference_openings: Vec<String>,
    metrics: Vec<Metric>,
    next_game_id: u64,
    opening_cursor: usize,
    last_arena_update: usize,
    pending_arena: Option<PendingArena>,
}

fn checkpoint(dir: &Path, cycle: usize) -> PathBuf {
    dir.join(format!("update-{cycle:06}.safetensors"))
}

fn atomic_write(path: &Path, contents: impl AsRef<[u8]>) -> io::Result<()> {
    let temporary = path.with_extension("tmp");
    let mut file = File::create(&temporary)?;
    file.write_all(contents.as_ref())?;
    file.sync_all()?;
    drop(file);
    fs::rename(temporary, path)
}

fn publish(dir: &Path, champion_cycle: usize) -> io::Result<()> {
    let target = dir.join("best.safetensors");
    let temporary = dir.join("best.tmp");
    fs::copy(checkpoint(dir, champion_cycle), &temporary)?;
    fs::OpenOptions::new()
        .write(true)
        .open(&temporary)?
        .sync_all()?;
    fs::rename(temporary, target)
}

fn save_replay(path: &Path, replay: &VecDeque<ReplayRow>) -> io::Result<()> {
    let mut text = String::from("fen\ttarget\n");
    for row in replay {
        text.push_str(&format!("{}\t{}\n", row.fen, row.target));
    }
    atomic_write(path, text)
}

fn load_replay(path: &Path) -> io::Result<VecDeque<ReplayRow>> {
    let mut result = VecDeque::new();
    for line in BufReader::new(File::open(path)?).lines().skip(1) {
        let line = line?;
        let (fen, value) = line
            .split_once('\t')
            .ok_or_else(|| io_error("损坏的回放行"))?;
        let target: f32 = value.parse().map_err(io_error)?;
        if !target.is_finite() || !(-1.0..=1.0).contains(&target) {
            return Err(io_error("损坏的回放标签"));
        }
        Position::from_fen(fen).map_err(io_error)?;
        result.push_back(ReplayRow {
            fen: fen.into(),
            target,
        });
    }
    Ok(result)
}

#[cfg(test)]
fn append_results(
    path: &Path,
    replay: &mut VecDeque<ReplayRow>,
    capacity: usize,
) -> io::Result<usize> {
    let mut lines = BufReader::new(File::open(path)?).lines();
    let header = lines
        .next()
        .transpose()?
        .ok_or_else(|| io_error("空自博弈文件"))?;
    let columns: Vec<_> = header.split('\t').collect();
    let column = |name| {
        columns
            .iter()
            .position(|&x| x == name)
            .ok_or_else(|| io_error(format!("缺少 {name}")))
    };
    let fen = column("fen")?;
    let result = column("red_result")?;
    let source = column("source")?;
    let mut added = 0;
    for line in lines {
        let line = line?;
        if line.split('\t').nth(source) != Some("candidate") {
            return Err(io_error("回放必须来自候选真实终局"));
        }
        if let Some((position, target)) =
            crate::pikafish_pretrain::parse_result_row(&line, fen, result)?
        {
            if PikafishExample::from_position(&position).is_none() {
                continue;
            }
            replay.push_back(ReplayRow {
                fen: position.to_fen(),
                target,
            });
            while replay.len() > capacity {
                replay.pop_front();
            }
            added += 1;
        }
    }
    Ok(added)
}

fn train_rows(
    model: &PikafishModel,
    rows: &[&ReplayRow],
    config: &EvolveConfig,
    stop: &AtomicBool,
) -> io::Result<(usize, f64)> {
    // SGD 没有动量或额外状态；保存权重和固定学习率即可精确恢复优化器。
    let device = model.vars()[0].device().clone();
    let mut optimizer = SGD::new(model.vars(), config.learning_rate).map_err(io_error)?;
    let mut used = 0;
    let mut total_loss = 0.0;
    for batch in rows.chunks(config.batch_size) {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let mut examples = Vec::with_capacity(batch.len());
        let mut targets = Vec::with_capacity(batch.len());
        for row in batch {
            let position = Position::from_fen(&row.fen).map_err(io_error)?;
            let example = PikafishExample::from_position(&position)
                .ok_or_else(|| io_error("无效训练特征"))?;
            examples.push(example);
            targets.push(row.target);
        }
        let loss =
            train_batch(model, &mut optimizer, &examples, &targets, &device).map_err(io_error)?;
        total_loss += loss * batch.len() as f64;
        used += batch.len();
    }
    Ok((used, total_loss / used.max(1) as f64))
}

fn bootstrap(
    model: &PikafishModel,
    config: &EvolveConfig,
    stop: &AtomicBool,
) -> io::Result<(usize, f64)> {
    let Some(path) = &config.bootstrap_sqlite else {
        return Ok((0, 0.0));
    };
    if !path.exists() {
        return Err(io_error(format!(
            "找不到预训练数据库 {}；可删除配置中的 bootstrap_sqlite",
            path.display()
        )));
    }
    let connection =
        rusqlite::Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .map_err(io_error)?;
    let mut statement = connection.prepare(
        "SELECT fen, best_score_cp FROM pikafish_labels WHERE best_score_cp IS NOT NULL ORDER BY ((id * 1103515245 + ?1) % 2147483647) LIMIT ?2"
    ).map_err(io_error)?;
    let rows = statement
        .query_map(
            rusqlite::params![config.seed as i64, config.bootstrap_samples as i64],
            |row| {
                Ok(ReplayRow {
                    fen: row.get(0)?,
                    target: cp_to_value(row.get(1)?),
                })
            },
        )
        .map_err(io_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(io_error)?;
    if rows.is_empty() {
        return Err(io_error("教师数据库无可用标签"));
    }
    let mut samples = 0;
    let mut weighted_loss = 0.0;
    for epoch in 0..config.bootstrap_epochs {
        let mut indices: Vec<_> = (0..rows.len()).collect();
        shuffle(&mut indices, config.seed ^ epoch as u64);
        let batch: Vec<_> = indices.iter().map(|&i| &rows[i]).collect();
        let (used, loss) = train_rows(model, &batch, config, stop)?;
        samples += used;
        weighted_loss += used as f64 * loss;
        println!(
            "预训练 {}/{}：样本={} loss={loss:.6}",
            epoch + 1,
            config.bootstrap_epochs,
            used
        );
        if stop.load(Ordering::Relaxed) {
            break;
        }
    }
    Ok((samples, weighted_loss / samples.max(1) as f64))
}

fn shuffle<T>(items: &mut [T], seed: u64) {
    let mut random = SplitMix64::new(seed);
    for i in (1..items.len()).rev() {
        items.swap(i, random.next_u64() as usize % (i + 1));
    }
}

fn distinct_openings(
    book: &mut OpeningBook,
    count: usize,
    excluded: &HashSet<String>,
) -> io::Result<Vec<Position>> {
    let mut positions = Vec::with_capacity(count);
    let mut seen = excluded.clone();
    for _ in 0..book.len() {
        if positions.len() == count {
            return Ok(positions);
        }
        let position = book.next_batch(1, 0)?.remove(0).position;
        if position.has_general(crate::xiangqi::Color::Red)
            && position.has_general(crate::xiangqi::Color::Black)
            && position
                .rule_outcome_with_history(&position.initial_rule_history())
                .is_none()
            && !position.legal_moves().is_empty()
            && PikafishExample::from_position(&position).is_some()
            && seen.insert(position.to_fen())
        {
            positions.push(position);
        }
    }
    if positions.len() == count {
        Ok(positions)
    } else {
        Err(io_error("开局库没有足够的不同非终局局面"))
    }
}

fn promotion(current: &CandidateArenaResult, reference: &CandidateArenaResult) -> bool {
    current.decision == CandidateArenaDecision::Promote && reference.upper_bound >= 0.5
}

struct StopMonitor {
    finished: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl StopMonitor {
    fn start(path: PathBuf, stop: Arc<AtomicBool>) -> Self {
        let finished = Arc::new(AtomicBool::new(false));
        let done = Arc::clone(&finished);
        let thread = std::thread::spawn(move || {
            while !done.load(Ordering::Relaxed) {
                if path.exists() {
                    stop.store(true, Ordering::Relaxed);
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
        });
        Self {
            finished,
            thread: Some(thread),
        }
    }
}

impl Drop for StopMonitor {
    fn drop(&mut self) {
        self.finished.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

pub fn request_stop(config: &EvolveConfig) -> io::Result<()> {
    let path = config.output_dir.join("stop.request");
    if !config.output_dir.join("progress.toml").exists() {
        return Err(io_error("该配置尚无训练进度"));
    }
    let lock = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(config.output_dir.join("training.lock"))?;
    if lock.try_lock().is_ok() {
        return Err(io_error("该目录当前没有运行中的训练进程"));
    }
    fs::write(&path, "保存后停止\n")?;
    println!("已请求保存后停止：{}", config.output_dir.display());
    Ok(())
}

fn commit_progress(dir: &Path, state: &Progress) -> io::Result<()> {
    atomic_write(
        &dir.join("progress.toml"),
        toml::to_string_pretty(state).map_err(io_error)?,
    )
}

fn runtime_status(
    stats: &PipelineStats,
    data: &SelfplayService,
    config: &EvolveConfig,
    update: usize,
    pending: Option<&PendingArena>,
) -> String {
    let arena = match pending {
        Some(candidate) => format!(
            "候选 update={} · {} · 新候选等待本轮结束",
            candidate.update,
            if stats.arena_active.load(Ordering::Relaxed) {
                "进行中"
            } else {
                "等待提交结果"
            }
        ),
        None => "空闲".into(),
    };
    format!(
        "更新 {update} · 自博弈 {}/{} 个 worker 在搜索，结果队列 {}/{} · 已生成 {} 局、{} 搜索节点 · 后台测评 {}，已完成 {}/{} 对",
        stats.active_selfplay.load(Ordering::Relaxed),
        config.selfplay_workers,
        data.results.len(),
        config.queue_games,
        stats.generated_games.load(Ordering::Relaxed),
        stats.search_nodes.load(Ordering::Relaxed),
        arena,
        stats.arena_pairs.load(Ordering::Relaxed),
        config.arena_pairs * 2
    )
}

fn collection_status(metric: &Metric, config: &EvolveConfig, elapsed: Duration) -> String {
    let seconds = elapsed.as_secs_f64();
    let rate = if seconds > 0.0 {
        metric.games as f64 * 60.0 / seconds
    } else {
        0.0
    };
    format!(
        "采集 {}：{}/{} ({:.0}%) · 终局 {} / 截断 {} · 新样本 {} · 等待 {:.0}s · 平均 {:.1} 局/分 · 过期丢弃 {}",
        metric.cycle,
        metric.games,
        config.games_per_update,
        metric.games as f64 * 100.0 / config.games_per_update as f64,
        metric.completed,
        metric.truncated,
        metric.new_samples,
        seconds,
        rate,
        metric.discarded_stale_games
    )
}

fn receive_arena(
    service: &ArenaService,
    shared: &RwLock<Champion>,
    state: &mut Progress,
    dir: &Path,
) -> io::Result<()> {
    while let Ok(outcome) = service.results.try_recv() {
        let ArenaOutcome {
            pending,
            result,
            seconds,
            candidate,
        } = outcome;
        let (current, anchor) = result.map_err(io_error)?;
        let mut metric = Metric {
            kind: "arena".into(),
            cycle: pending.cycle,
            update: pending.update,
            generation: state.generation,
            seconds,
            ..Metric::default()
        };
        metric.completed = current.report.total_games() + anchor.report.total_games();
        metric.truncated = current.truncated_games + anchor.truncated_games;
        metric.games = metric.completed + metric.truncated;
        metric.score = (current.report.total_games() > 0).then(|| current.report.score_rate());
        metric.lower = Some(current.lower_bound);
        metric.upper = Some(current.upper_bound);
        metric.champion_wins = current.report.wins;
        metric.champion_losses = current.report.losses;
        metric.champion_draws = current.report.draws;
        metric.champion_truncated = current.truncated_games;
        metric.reference_score =
            (anchor.report.total_games() > 0).then(|| anchor.report.score_rate());
        metric.reference_lower = Some(anchor.lower_bound);
        metric.reference_upper = Some(anchor.upper_bound);
        metric.reference_wins = anchor.report.wins;
        metric.reference_losses = anchor.report.losses;
        metric.reference_draws = anchor.report.draws;
        metric.reference_truncated = anchor.truncated_games;
        metric.decision = format!("{:?}", current.decision);
        if pending.champion_generation != state.generation {
            metric.decision = "旧冠军测评结果，不晋级".into();
        } else if promotion(&current, &anchor) {
            state.generation += 1;
            state.champion_cycle = pending.cycle;
            metric.generation = state.generation;
            metric.decision = "晋级".into();
            let mut champion = shared.write().map_err(|_| io_error("冠军锁损坏"))?;
            *champion = Champion {
                generation: state.generation,
                model: candidate,
            };
        } else if anchor.upper_bound < 0.5 {
            metric.decision = "固定基准退步，拒绝晋级".into();
        }
        println!(
            "后台晋级 update={}：得分={} CI={}..{} 基准={} 决定={}；冠军第 {} 代",
            metric.update,
            optional(metric.score),
            optional(metric.lower),
            optional(metric.upper),
            optional(metric.reference_score),
            metric.decision,
            state.generation
        );
        state.pending_arena = None;
        state.metrics.push(metric);
        commit_progress(dir, state)?;
        if state.champion_cycle == pending.cycle {
            publish(dir, state.champion_cycle)?;
        }
        render_metrics(dir, &state.metrics)?;
    }
    Ok(())
}

fn submit_arena(
    service: &ArenaService,
    shared: &RwLock<Champion>,
    learner: &PikafishModel,
    state: &mut Progress,
    book: &Mutex<OpeningBook>,
    excluded: &HashSet<String>,
    dir: &Path,
) -> io::Result<()> {
    let pending = if let Some(pending) = &state.pending_arena {
        pending.clone()
    } else {
        let mut book = book.lock().map_err(|_| io_error("开局池锁损坏"))?;
        let openings = distinct_openings(&mut book, state.config.arena_pairs, excluded)?;
        let pending = PendingArena {
            cycle: state.cycle,
            update: state.update,
            champion_cycle: state.champion_cycle,
            champion_generation: state.generation,
            openings: openings.iter().map(Position::to_fen).collect(),
        };
        state.last_arena_update = state.update;
        state.pending_arena = Some(pending.clone());
        commit_progress(dir, state)?;
        pending
    };
    let candidate = if pending.cycle == state.cycle {
        learner.cpu_snapshot().map_err(io_error)?
    } else {
        PikafishCpuModel::load(&checkpoint(dir, pending.cycle)).map_err(io_error)?
    };
    let champion = {
        let current = shared.read().map_err(|_| io_error("冠军锁损坏"))?;
        if current.generation == pending.champion_generation {
            Arc::clone(&current.model)
        } else {
            let model = PikafishCpuModel::load(&checkpoint(dir, pending.champion_cycle))
                .map_err(io_error)?;
            Arc::new(model)
        }
    };
    let openings = pending
        .openings
        .iter()
        .map(|fen| Position::from_fen(fen).map_err(io_error))
        .collect::<io::Result<_>>()?;
    println!(
        "提交后台晋级：候选 update={}，{} 个测评 worker；GPU 训练继续",
        pending.update, state.config.arena_workers
    );
    service
        .tasks
        .send(ArenaTask {
            pending,
            candidate: Arc::new(candidate),
            champion,
            openings,
        })
        .map_err(io_error)
}

/// CPU 自博弈持续供数，GPU 更新与后台晋级独立推进；只发布通过门槛的固定快照。
pub fn run(config: EvolveConfig, target_update: Option<usize>) -> io::Result<()> {
    config.validate()?;
    config.check_inputs()?;
    fs::create_dir_all(&config.output_dir)
        .map_err(|error| path_error("无法创建 output_dir", &config.output_dir, error))?;
    let dir = &config.output_dir;
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(dir.join("training.lock"))?;
    lock.try_lock()
        .map_err(|error| io_error(format!("训练目录已被占用：{error}")))?;
    let stop = Arc::new(AtomicBool::new(false));
    let signal = Arc::clone(&stop);
    ctrlc::set_handler(move || signal.store(true, Ordering::Relaxed)).map_err(io_error)?;
    let stop_path = dir.join("stop.request");
    let _monitor = StopMonitor::start(stop_path.clone(), Arc::clone(&stop));
    let device = if config.cuda {
        Device::new_cuda(0).map_err(io_error)?
    } else {
        Device::Cpu
    };
    println!(
        "Pikafish 流水线：训练={device:?} 自博弈 workers={} 测评 workers={} 队列上限={}；共享只读 CPU 权重",
        config.selfplay_workers, config.arena_workers, config.queue_games
    );
    let learner = PikafishModel::new(&device).map_err(io_error)?;
    let mut opening_book = OpeningBook::load(&config.opening_book, config.seed)
        .map_err(|error| path_error("加载 opening_book 失败", &config.opening_book, error))?;
    let (mut state, mut replay, reference, champion) = if dir.join("progress.toml").exists() {
        let mut state: Progress =
            toml::from_str(&fs::read_to_string(dir.join("progress.toml"))?).map_err(io_error)?;
        let mut saved = state.config.clone();
        saved.selfplay_workers = config.selfplay_workers;
        saved.arena_workers = config.arena_workers;
        saved.queue_games = config.queue_games;
        if state.version != 3 || saved != config {
            return Err(io_error(
                "进度格式或训练参数不一致；只允许续训时调整 worker 数和队列容量",
            ));
        }
        state.config = config.clone();
        learner
            .load(&checkpoint(dir, state.cycle))
            .map_err(|error| {
                path_error("加载训练权重失败", &checkpoint(dir, state.cycle), error)
            })?;
        let reference = PikafishCpuModel::load(&checkpoint(dir, 0))
            .map_err(|error| path_error("加载基准权重失败", &checkpoint(dir, 0), error))?;
        let reference = Arc::new(reference);
        let champion = if state.champion_cycle == 0 {
            Arc::clone(&reference)
        } else {
            let model = PikafishCpuModel::load(&checkpoint(dir, state.champion_cycle)).map_err(
                |error| {
                    path_error(
                        "加载冠军权重失败",
                        &checkpoint(dir, state.champion_cycle),
                        error,
                    )
                },
            )?;
            Arc::new(model)
        };
        let replay = load_replay(&dir.join(format!("replay-{:06}.tsv", state.cycle)))?;
        opening_book.seek(state.opening_cursor);
        println!(
            "恢复：update={} generation={} replay={} pending_arena={}",
            state.update,
            state.generation,
            replay.len(),
            state.pending_arena.is_some()
        );
        (state, replay, reference, champion)
    } else {
        if let Some(seed) = &config.seed_model {
            learner
                .load(seed)
                .map_err(|error| path_error("加载 seed_model 失败", seed, error))?;
        }
        learner.save(&checkpoint(dir, 0)).map_err(io_error)?;
        let reference = Arc::new(learner.cpu_snapshot().map_err(io_error)?);
        let reference_openings =
            distinct_openings(&mut opening_book, config.arena_pairs, &HashSet::new())?
                .iter()
                .map(Position::to_fen)
                .collect();
        let state = Progress {
            version: 3,
            config: config.clone(),
            cycle: 0,
            update: 0,
            generation: 0,
            champion_cycle: 0,
            bootstrap_done: false,
            reference_openings,
            metrics: Vec::new(),
            next_game_id: 0,
            opening_cursor: opening_book.cursor(),
            last_arena_update: 0,
            pending_arena: None,
        };
        let replay = VecDeque::new();
        save_replay(&dir.join("replay-000000.tsv"), &replay)?;
        commit_progress(dir, &state)?;
        (state, replay, Arc::clone(&reference), reference)
    };
    publish(dir, state.champion_cycle)?;
    let reference_openings = state
        .reference_openings
        .iter()
        .map(|fen| Position::from_fen(fen).map_err(io_error))
        .collect::<io::Result<Vec<_>>>()?;
    let excluded: HashSet<_> = state.reference_openings.iter().cloned().collect();
    let book = Arc::new(Mutex::new(opening_book));
    let shared = Arc::new(RwLock::new(Champion {
        generation: state.generation,
        model: champion,
    }));
    let stats = Arc::new(PipelineStats::default());
    let selfplay = SelfplayService::start(
        &config,
        Arc::clone(&book),
        excluded.clone(),
        Arc::clone(&shared),
        state.next_game_id,
        Arc::clone(&stop),
        Arc::clone(&stats),
    )?;
    let arena = ArenaService::start(
        &config,
        reference,
        reference_openings,
        Arc::clone(&stop),
        Arc::clone(&stats),
    )?;
    if state.pending_arena.is_some() {
        submit_arena(&arena, &shared, &learner, &mut state, &book, &excluded, dir)?;
    }
    while !stop.load(Ordering::Relaxed)
        && !target_update.is_some_and(|target| state.update >= target)
    {
        receive_arena(&arena, &shared, &mut state, dir)?;
        if state.pending_arena.is_none()
            && state.update > 0
            && (state.last_arena_update == 0
                || state.update.saturating_sub(state.last_arena_update) >= config.arena_interval)
        {
            submit_arena(&arena, &shared, &learner, &mut state, &book, &excluded, dir)?;
        }
        let started = Instant::now();
        let cycle = state.cycle + 1;
        let mut metric = Metric {
            kind: "train".into(),
            cycle,
            generation: state.generation,
            update: state.update,
            decision: "训练".into(),
            source_generation_min: state.generation,
            ..Metric::default()
        };
        if !state.bootstrap_done {
            set_status(dir, &state.metrics, "GPU 教师预训练与 CPU 自博弈并行进行")?;
            let before = stats.search_nodes.load(Ordering::Relaxed);
            let training = Instant::now();
            let (used, loss) = bootstrap(&learner, &config, &stop)?;
            metric.training_seconds += training.elapsed().as_secs_f64();
            metric.search_nodes_during_training +=
                stats.search_nodes.load(Ordering::Relaxed) - before;
            metric.trained_samples = used;
            metric.bootstrap_samples = used;
            metric.loss = (used > 0).then_some(loss);
            state.bootstrap_done = !stop.load(Ordering::Relaxed);
        }
        let waiting = Instant::now();
        let mut last_status = Instant::now() - Duration::from_secs(2);
        let mut last_log = Instant::now();
        let mut last_nodes = stats.search_nodes.load(Ordering::Relaxed);
        println!(
            "开始采集 {cycle}：目标 {} 局，自博弈 {} workers；每 10 秒汇总",
            config.games_per_update, config.selfplay_workers
        );
        while metric.games < config.games_per_update && !stop.load(Ordering::Relaxed) {
            receive_arena(&arena, &shared, &mut state, dir)?;
            if last_status.elapsed() >= Duration::from_secs(1) {
                set_status(
                    dir,
                    &state.metrics,
                    &format!(
                        "{} · {}",
                        collection_status(&metric, &config, waiting.elapsed()),
                        runtime_status(
                            &stats,
                            &selfplay,
                            &config,
                            state.update,
                            state.pending_arena.as_ref()
                        )
                    ),
                )?;
                last_status = Instant::now();
            }
            if last_log.elapsed() >= Duration::from_secs(10) {
                let nodes = stats.search_nodes.load(Ordering::Relaxed);
                println!(
                    "{} · 搜索 {}/{} workers ({:.0} 节点/s) · 队列 {}/{}",
                    collection_status(&metric, &config, waiting.elapsed()),
                    stats.active_selfplay.load(Ordering::Relaxed),
                    config.selfplay_workers,
                    nodes.saturating_sub(last_nodes) as f64 / last_log.elapsed().as_secs_f64(),
                    selfplay.results.len(),
                    config.queue_games
                );
                last_nodes = nodes;
                last_log = Instant::now();
            }
            match selfplay.results.recv_timeout(Duration::from_millis(100)) {
                Ok(result) => {
                    let generated = result?;
                    if state.generation.saturating_sub(generated.generation)
                        > config.max_champion_lag
                    {
                        metric.discarded_stale_games += 1;
                        continue;
                    }
                    metric.source_generation_min = if metric.games == 0 {
                        generated.generation
                    } else {
                        metric.source_generation_min.min(generated.generation)
                    };
                    metric.source_generation_max =
                        metric.source_generation_max.max(generated.generation);
                    metric.games += 1;
                    if generated.game.red_result.is_some() {
                        metric.completed += 1;
                    } else {
                        metric.truncated += 1;
                    }
                    for (fen, target) in generated.game.labels() {
                        replay.push_back(ReplayRow {
                            fen: fen.into(),
                            target,
                        });
                        metric.new_samples += 1;
                        while replay.len() > config.replay_capacity {
                            replay.pop_front();
                        }
                    }
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    return Err(io_error("自博弈 worker 全部退出"));
                }
            }
        }
        println!("{}", collection_status(&metric, &config, waiting.elapsed()));
        metric.data_wait_seconds = waiting.elapsed().as_secs_f64();
        metric.replay_samples = replay.len();
        if metric.new_samples > 0 && !stop.load(Ordering::Relaxed) {
            set_status(
                dir,
                &state.metrics,
                &format!(
                    "GPU 更新候选；{} · 回放 {}",
                    runtime_status(
                        &stats,
                        &selfplay,
                        &config,
                        state.update,
                        state.pending_arena.as_ref()
                    ),
                    replay.len()
                ),
            )?;
            let mut indices: Vec<_> = (0..replay.len()).collect();
            shuffle(
                &mut indices,
                config.seed ^ (cycle as u64).wrapping_mul(0x9E3779B97F4A7C15),
            );
            indices.truncate(config.train_samples_per_update);
            let rows: Vec<_> = indices.iter().map(|&index| &replay[index]).collect();
            let before = stats.search_nodes.load(Ordering::Relaxed);
            let training = Instant::now();
            metric.trained_during_arena = stats.arena_active.load(Ordering::Relaxed);
            let (used, loss) = train_rows(&learner, &rows, &config, &stop)?;
            metric.training_seconds += training.elapsed().as_secs_f64();
            metric.search_nodes_during_training +=
                stats.search_nodes.load(Ordering::Relaxed) - before;
            if used > 0 {
                let combined =
                    metric.loss.unwrap_or(0.0) * metric.trained_samples as f64 + loss * used as f64;
                metric.trained_samples += used;
                metric.loss = Some(combined / metric.trained_samples as f64);
            }
        }
        if metric.trained_samples > 0 {
            state.update += 1;
        } else {
            metric.decision = if stop.load(Ordering::Relaxed) {
                "停止请求，跳过训练"
            } else {
                "无终局标签，跳过训练"
            }
            .into();
        }
        metric.update = state.update;
        metric.generation = state.generation;
        metric.seconds = started.elapsed().as_secs_f64();
        metric.peak_selfplay_workers = stats.peak_selfplay.load(Ordering::Relaxed);
        metric.queue_depth = selfplay.results.len();
        println!(
            "更新 {}：loss={} 训练样本={} 回放={} GPU训练={:.2}s 数据等待={:.2}s 训练期间CPU搜索={}节点 并行峰值={} 测评重叠={}",
            state.update,
            metric
                .loss
                .map(|loss| format!("{loss:.6}"))
                .unwrap_or_else(|| "—".into()),
            metric.trained_samples,
            metric.replay_samples,
            metric.training_seconds,
            metric.data_wait_seconds,
            metric.search_nodes_during_training,
            metric.peak_selfplay_workers,
            metric.trained_during_arena
        );
        learner.save(&checkpoint(dir, cycle)).map_err(io_error)?;
        save_replay(&dir.join(format!("replay-{cycle:06}.tsv")), &replay)?;
        state.cycle = cycle;
        state.next_game_id = selfplay.next_game.load(Ordering::Relaxed);
        state.opening_cursor = book.lock().map_err(|_| io_error("开局池锁损坏"))?.cursor();
        state.metrics.push(metric);
        commit_progress(dir, &state)?;
        render_metrics(dir, &state.metrics)?;
        prune(dir, &state)?;
    }
    receive_arena(&arena, &shared, &mut state, dir)?;
    stop.store(true, Ordering::Relaxed);
    let next_game = Arc::clone(&selfplay.next_game);
    drop(selfplay);
    drop(arena);
    state.next_game_id = next_game.load(Ordering::Relaxed);
    state.opening_cursor = book.lock().map_err(|_| io_error("开局池锁损坏"))?.cursor();
    commit_progress(dir, &state)?;
    if stop_path.exists() {
        fs::remove_file(stop_path)?;
    }
    set_status(
        dir,
        &state.metrics,
        &format!(
            "已保存并停止：更新 {}，冠军第 {} 代；未完成的后台测评下次恢复",
            state.update, state.generation
        ),
    )?;
    println!(
        "已保存并停止：update={} generation={} pending_arena={}",
        state.update,
        state.generation,
        state.pending_arena.is_some()
    );
    Ok(())
}

fn prune(dir: &Path, state: &Progress) -> io::Result<()> {
    if state.cycle < 2 {
        return Ok(());
    }
    let previous = state.cycle - 1;
    let retained: HashSet<_> = [0, state.champion_cycle, previous, state.cycle]
        .into_iter()
        .chain(
            state
                .pending_arena
                .iter()
                .flat_map(|pending| [pending.cycle, pending.champion_cycle]),
        )
        .collect();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let cycle = |prefix, suffix| {
            name.strip_prefix(prefix)?
                .strip_suffix(suffix)?
                .parse::<usize>()
                .ok()
        };
        if cycle("update-", ".safetensors").is_some_and(|cycle| !retained.contains(&cycle))
            || cycle("replay-", ".tsv").is_some_and(|cycle| cycle < previous)
        {
            fs::remove_file(entry.path())?;
        }
    }
    Ok(())
}

fn optional(value: Option<f32>) -> String {
    value.map(|x| format!("{x:.4}")).unwrap_or_default()
}

fn arena_score_cell(
    score: Option<f32>,
    lower: Option<f32>,
    upper: Option<f32>,
    wins: usize,
    losses: usize,
    draws: usize,
    truncated: usize,
    arena: bool,
) -> String {
    if !arena {
        return String::new();
    }
    format!(
        "{}<br><small>胜/负/和/未决={wins}/{losses}/{draws}/{truncated}<br>区间 {}–{}</small>",
        score
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "无终局".into()),
        optional(lower),
        optional(upper)
    )
}

fn set_status(dir: &Path, metrics: &[Metric], status: &str) -> io::Result<()> {
    atomic_write(&dir.join("status.txt"), status)?;
    render_metrics(dir, metrics)
}

fn render_metrics(dir: &Path, metrics: &[Metric]) -> io::Result<()> {
    let mut csv = csv::Writer::from_writer(Vec::new());
    let mut rows = String::new();
    let mut points = Vec::new();
    for item in metrics {
        csv.serialize(item).map_err(io_error)?;
        rows.push_str(&format!("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}/{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            if item.kind == "arena" { "测评" } else { "训练" }, item.update, item.generation, item.peak_selfplay_workers,
            item.completed, item.truncated, item.replay_samples,
            item.loss.map(|x| format!("{x:.5}")).unwrap_or_else(|| "—".into()), item.search_nodes_during_training,
            if item.trained_during_arena { "是" } else { "—" }, arena_score_cell(item.score, item.lower, item.upper, item.champion_wins, item.champion_losses, item.champion_draws, item.champion_truncated, item.kind == "arena"), arena_score_cell(item.reference_score, item.reference_lower, item.reference_upper, item.reference_wins, item.reference_losses, item.reference_draws, item.reference_truncated, item.kind == "arena"), item.decision));
        if item.decision == "晋级" {
            if let Some(score) = item.reference_score {
                points.push((item.cycle, score));
            }
        }
    }
    let width = metrics.iter().map(|x| x.cycle).max().unwrap_or(1).max(1) as f32;
    let polyline = std::iter::once("40,150".into())
        .chain(points.iter().map(|&(cycle, score)| {
            format!(
                "{:.1},{:.1}",
                40.0 + cycle as f32 / width * 720.0,
                270.0 - score * 240.0
            )
        }))
        .collect::<Vec<String>>()
        .join(" ");
    let latest = metrics.last();
    let generation = latest.map(|x| x.generation).unwrap_or(0);
    let status = fs::read_to_string(dir.join("status.txt"))
        .unwrap_or_else(|_| "准备训练".into())
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");
    let html = format!(
        r##"<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>象棋自动进化</title>
<style>body{{font:16px system-ui;background:#101926;color:#e4edf7;margin:40px auto;max-width:1100px;padding:0 20px}}h1{{color:#70dbc1}}small,p{{color:#aabbd0}}svg{{width:100%;background:#172437;border-radius:12px}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #2b3c51;text-align:left}}.scroll{{overflow:auto}}a{{color:#70dbc1}}</style>
<h1>象棋自动进化 · 冠军第 {generation} 代</h1><p>{status}</p><p>曲线仅显示通过晋级的冠军，对固定初始模型的实际比赛得分。起点 0.5 是自身对比基线，尚未晋级时只有起点。每 30 秒刷新。</p>
<svg viewBox="0 0 800 300"><text x="5" y="35" fill="#aabbd0">1.0</text><text x="5" y="155" fill="#aabbd0">0.5</text><text x="5" y="275" fill="#aabbd0">0.0</text><path d="M40 150H760" stroke="#51647e" stroke-dasharray="6 6"/><circle cx="40" cy="150" r="4" fill="#70dbc1"/><polyline points="{polyline}" fill="none" stroke="#70dbc1" stroke-width="3"/></svg>
<p>晋级要求：冠军赛得分置信下界超过门槛，且固定基准无显著退步；截断记为未决，整批继续；置信下界将未决视为负、上界视为胜。表格得分仅统计终局，候选训练损失下降不代表棋力上涨。重复测评的置信区间是逐次区间，不是全程错误率保证。</p>
<div class="scroll"><table><thead><tr><th>事件</th><th>更新</th><th>冠军代数</th><th>并行峰值</th><th>终局/截断</th><th>回放</th><th>Loss</th><th>训练时搜索节点</th><th>训练/测评重叠</th><th>对冠军得分</th><th>对基准得分</th><th>决定</th></tr></thead><tbody>{rows}</tbody></table></div><p><a href="metrics.csv">下载完整比赛记录与置信区间</a></p></html>"##
    );
    atomic_write(
        &dir.join("metrics.csv"),
        csv.into_inner().map_err(io_error)?,
    )?;
    atomic_write(&dir.join("dashboard.html"), html)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn config_rejects_weak_gates_and_invalid_training() {
        let mut config = EvolveConfig::default();
        assert!(config.validate().is_ok());
        config.confidence_z = 0.0;
        assert!(config.validate().is_err());
        let from_empty: EvolveConfig = toml::from_str("").unwrap();
        assert!(from_empty.bootstrap_sqlite.is_none());
        config = EvolveConfig::default();
        config.learning_rate = f64::NAN;
        assert!(config.validate().is_err());
        config = EvolveConfig::default();
        config.arena_pairs = 2;
        assert!(config.validate().is_err());
    }
    #[test]
    fn startup_checks_inputs_before_initialization() -> io::Result<()> {
        let dir = std::env::temp_dir().join(format!("evolve-inputs-{}", std::process::id()));
        fs::create_dir_all(&dir)?;
        let book = dir.join("book.pgn.gz");
        let seed = dir.join("missing-seed.safetensors");
        let database = dir.join("missing-teacher.sqlite");
        let mut config = EvolveConfig {
            output_dir: dir.join("run"),
            opening_book: book.clone(),
            seed_model: Some(seed.clone()),
            bootstrap_sqlite: Some(database.clone()),
            ..Default::default()
        };
        let error = config.check_inputs().unwrap_err().to_string();
        assert!(error.contains("opening_book") && error.contains(&book.display().to_string()));
        fs::write(&book, [])?;
        let error = config.check_inputs().unwrap_err().to_string();
        assert!(error.contains("seed_model") && error.contains(&seed.display().to_string()));
        config.seed_model = None;
        let error = config.check_inputs().unwrap_err().to_string();
        assert!(
            error.contains("bootstrap_sqlite") && error.contains(&database.display().to_string())
        );
        config.bootstrap_sqlite = None;
        config.check_inputs()?;
        fs::create_dir_all(&config.output_dir)?;
        fs::write(config.output_dir.join("progress.toml"), [])?;
        config.seed_model = Some(seed);
        config.bootstrap_sqlite = Some(database);
        config.check_inputs()?;
        fs::remove_dir_all(dir)
    }

    #[test]
    fn arena_dashboard_preserves_counts_and_uncertainty() -> io::Result<()> {
        let dir = std::env::temp_dir().join(format!("arena-dashboard-{}", std::process::id()));
        fs::create_dir_all(&dir)?;
        let metric = Metric {
            kind: "arena".into(),
            completed: 15,
            truncated: 1,
            champion_wins: 15,
            champion_truncated: 1,
            score: Some(1.),
            lower: Some(0.6),
            upper: Some(1.),
            decision: "Inconclusive".into(),
            ..Default::default()
        };
        render_metrics(&dir, &[metric.clone()])?;
        let html = fs::read_to_string(dir.join("dashboard.html"))?;
        assert!(html.contains("<td>15/1</td>"));
        assert!(html.contains("胜/负/和/未决=15/0/0/1"));
        assert!(html.contains("区间 0.6000–1.0000"));
        let mut old = toml::to_string(&metric).map_err(io_error)?;
        old = old
            .lines()
            .filter(|line| {
                !line.starts_with("champion_truncated") && !line.starts_with("reference_truncated")
            })
            .collect::<Vec<_>>()
            .join("\n");
        let restored: Metric = toml::from_str(&old).map_err(io_error)?;
        assert_eq!(restored.champion_wins, 15);
        assert_eq!(restored.champion_truncated, 0);
        fs::remove_dir_all(dir)
    }

    #[test]
    fn only_evidence_of_improvement_can_promote() {
        use crate::ab::AbArenaReport;
        let result = |decision, lower, upper| CandidateArenaResult {
            report: AbArenaReport::default(),
            truncated_games: 0,
            lower_bound: lower,
            upper_bound: upper,
            decision,
        };
        let current = result(CandidateArenaDecision::Promote, 0.6, 0.8);
        let regressed = result(CandidateArenaDecision::Reject, 0.1, 0.4);
        assert!(!promotion(&current, &regressed));
        let even = result(CandidateArenaDecision::Inconclusive, 0.4, 0.6);
        assert!(promotion(&current, &even));
        assert!(!promotion(&even, &even));
    }
    #[test]
    fn replay_uses_terminal_results_and_enforces_capacity() -> io::Result<()> {
        let dir = std::env::temp_dir().join(format!("evolve-replay-{}", std::process::id()));
        fs::create_dir_all(&dir)?;
        let path = dir.join("input.tsv");
        let red = Position::startpos().to_fen();
        let black = red.replacen(" w ", " b ", 1);
        fs::write(
            &path,
            format!(
                "fen\tred_result\tsource\n{red}\t?\tcandidate\n{red}\t1\tcandidate\n{black}\t1\tcandidate\n{red}\t0\tcandidate\n"
            ),
        )?;
        let mut replay = VecDeque::new();
        assert_eq!(append_results(&path, &mut replay, 2)?, 3);
        assert_eq!(replay.len(), 2);
        assert_eq!(replay[0].target, -1.0);
        assert_eq!(replay[1].target, -1.0);
        let saved = dir.join("replay.tsv");
        save_replay(&saved, &replay)?;
        save_replay(&saved, &replay)?;
        assert_eq!(load_replay(&saved)?.len(), 2);
        fs::write(
            &path,
            format!("fen\tred_result\tsource\n{red}\t1\tpikafish\n"),
        )?;
        assert!(append_results(&path, &mut replay, 2).is_err());
        fs::remove_file(path)?;
        fs::remove_file(saved)?;
        fs::remove_dir(dir)?;
        Ok(())
    }

    #[test]
    fn committed_metrics_survive_resume_and_hide_unpromoted_candidates() -> io::Result<()> {
        let dir = std::env::temp_dir().join(format!("evolve-metrics-{}", std::process::id()));
        fs::create_dir_all(&dir)?;
        let metrics = vec![
            Metric {
                cycle: 1,
                update: 1,
                reference_score: Some(0.9),
                decision: "Inconclusive".into(),
                ..Metric::default()
            },
            Metric {
                cycle: 2,
                update: 2,
                generation: 1,
                reference_score: Some(0.8),
                decision: "晋级".into(),
                ..Metric::default()
            },
        ];
        let mut state = Progress {
            version: 3,
            config: EvolveConfig::default(),
            cycle: 2,
            update: 2,
            generation: 1,
            champion_cycle: 2,
            bootstrap_done: true,
            reference_openings: vec![Position::startpos().to_fen()],
            metrics,
            next_game_id: 20,
            opening_cursor: 20,
            last_arena_update: 2,
            pending_arena: None,
        };
        let path = dir.join("progress.toml");
        atomic_write(&path, toml::to_string_pretty(&state).map_err(io_error)?)?;
        let restored: Progress = toml::from_str(&fs::read_to_string(&path)?).map_err(io_error)?;
        assert_eq!((restored.update, restored.champion_cycle), (2, 2));
        render_metrics(&dir, &restored.metrics)?;
        let html = fs::read_to_string(dir.join("dashboard.html"))?;
        assert!(html.contains("points=\"40,150 760.0,78.0\""));
        assert_eq!(
            fs::read_to_string(dir.join("metrics.csv"))?.lines().count(),
            3
        );
        // 后台仍在测评旧候选时，清理必须保留它及其对手；历史晋级不无限保留权重。
        state.cycle = 8;
        state.champion_cycle = 4;
        state.pending_arena = Some(PendingArena {
            cycle: 2,
            update: 2,
            champion_cycle: 0,
            champion_generation: 0,
            openings: vec![Position::startpos().to_fen()],
        });
        for cycle in 0..=8 {
            fs::write(checkpoint(&dir, cycle), b"snapshot")?;
            fs::write(dir.join(format!("replay-{cycle:06}.tsv")), b"replay")?;
        }
        prune(&dir, &state)?;
        for cycle in 0..=8 {
            assert_eq!(
                checkpoint(&dir, cycle).exists(),
                [0, 2, 4, 7, 8].contains(&cycle)
            );
            let path = dir.join(format!("replay-{cycle:06}.tsv"));
            assert_eq!(path.exists(), cycle >= 7);
            if checkpoint(&dir, cycle).exists() {
                fs::remove_file(checkpoint(&dir, cycle))?;
            }
            if path.exists() {
                fs::remove_file(path)?;
            }
        }
        for file in ["progress.toml", "dashboard.html", "metrics.csv"] {
            fs::remove_file(dir.join(file))?;
        }
        fs::remove_dir(dir)?;
        Ok(())
    }
}
