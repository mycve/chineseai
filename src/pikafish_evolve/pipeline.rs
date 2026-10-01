//! 有界自博弈流水线和独立晋级服务；所有搜索共享不可变模型快照。
use super::{EvolveConfig, PendingArena, distinct_openings, io_error};
use crate::{
    ab::pikafish_candle::PikafishModel,
    opening_book::OpeningBook,
    pikafish_candidate_arena::{
        CandidateArenaConfig, CandidateArenaResult, play_paired_parallel_with_stop,
    },
    pikafish_candidate_selfplay::{CandidateSelfplayConfig, SelfplayGame, play_game},
    xiangqi::Position,
};
use crossbeam_channel::{Receiver, SendTimeoutError, Sender, bounded};
use std::{
    collections::HashSet,
    io,
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

#[derive(Default)]
pub struct PipelineStats {
    pub active_selfplay: AtomicUsize,
    pub peak_selfplay: AtomicUsize,
    pub generated_games: AtomicU64,
    pub search_nodes: AtomicU64,
    pub arena_active: AtomicBool,
    pub arena_pairs: AtomicUsize,
}

pub struct Champion {
    pub generation: usize,
    pub model: Arc<PikafishModel>,
}

pub struct GeneratedGame {
    pub id: u64,
    pub generation: usize,
    pub game: SelfplayGame,
}

fn send<T>(sender: &Sender<T>, stop: &AtomicBool, mut value: T) -> bool {
    while !stop.load(Ordering::Relaxed) {
        match sender.send_timeout(value, Duration::from_millis(100)) {
            Ok(()) => return true,
            Err(SendTimeoutError::Timeout(returned)) => value = returned,
            Err(SendTimeoutError::Disconnected(_)) => return false,
        }
    }
    false
}

/// 任务队列和结果队列都受限；结果积压时 worker 停止产生更多数据。
pub struct SelfplayService {
    pub results: Receiver<io::Result<GeneratedGame>>,
    pub next_game: Arc<AtomicU64>,
    _workers: WorkerGroup,
}

struct WorkerGroup {
    stop: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
}

impl Drop for WorkerGroup {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

impl SelfplayService {
    pub fn start(
        config: &EvolveConfig,
        book: Arc<Mutex<OpeningBook>>,
        excluded: HashSet<String>,
        champion: Arc<RwLock<Champion>>,
        next_id: u64,
        stop: Arc<AtomicBool>,
        stats: Arc<PipelineStats>,
    ) -> io::Result<Self> {
        let (jobs_tx, jobs_rx) = bounded::<(u64, Position)>(config.selfplay_workers);
        let (results_tx, results) = bounded(config.queue_games);
        let next_game = Arc::new(AtomicU64::new(next_id));
        let mut workers = WorkerGroup {
            stop: Arc::clone(&stop),
            threads: Vec::with_capacity(config.selfplay_workers + 1),
        };
        let dispatcher_stop = Arc::clone(&stop);
        let dispatcher_id = Arc::clone(&next_game);
        let dispatcher_error = results_tx.clone();
        workers.threads.push(
            std::thread::Builder::new()
                .name("pikafish-openings".into())
                .spawn(move || {
                    while !dispatcher_stop.load(Ordering::Relaxed) {
                        let position = (|| {
                            let mut book = book.lock().map_err(|_| io_error("开局池锁损坏"))?;
                            Ok::<_, io::Error>(
                                distinct_openings(&mut book, 1, &excluded)?.remove(0),
                            )
                        })();
                        match position {
                            Ok(position) => {
                                let id = dispatcher_id.fetch_add(1, Ordering::Relaxed);
                                if !send(&jobs_tx, &dispatcher_stop, (id, position)) {
                                    break;
                                }
                            }
                            Err(error) => {
                                send(&dispatcher_error, &dispatcher_stop, Err(error));
                                break;
                            }
                        }
                    }
                })?,
        );
        for worker in 0..config.selfplay_workers {
            let jobs = jobs_rx.clone();
            let results = results_tx.clone();
            let stop = Arc::clone(&stop);
            let stats = Arc::clone(&stats);
            let champion = Arc::clone(&champion);
            let config = config.clone();
            workers.threads.push(
                std::thread::Builder::new()
                    .name(format!("pikafish-selfplay-{worker}"))
                    .spawn(move || {
                        while !stop.load(Ordering::Relaxed) {
                            let (id, position) = match jobs.recv_timeout(Duration::from_millis(100))
                            {
                                Ok(job) => job,
                                Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
                                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                            };
                            let snapshot = champion
                                .read()
                                .map(|shared| (shared.generation, Arc::clone(&shared.model)));
                            let (generation, model) = match snapshot {
                                Ok(snapshot) => snapshot,
                                Err(_) => {
                                    send(&results, &stop, Err(io_error("冠军锁损坏")));
                                    break;
                                }
                            };
                            let active = stats.active_selfplay.fetch_add(1, Ordering::Relaxed) + 1;
                            stats.peak_selfplay.fetch_max(active, Ordering::Relaxed);
                            let game = play_game(
                                &model,
                                position,
                                CandidateSelfplayConfig {
                                    games: 1,
                                    nodes: config.selfplay_nodes,
                                    max_depth: config.max_depth,
                                    max_plies: config.max_plies,
                                    opening_plies: config.opening_plies,
                                    seed: config.seed ^ id.wrapping_mul(0x9E3779B97F4A7C15),
                                },
                                &stop,
                                |nodes| {
                                    stats
                                        .search_nodes
                                        .fetch_add(nodes as u64, Ordering::Relaxed);
                                },
                            );
                            stats.active_selfplay.fetch_sub(1, Ordering::Relaxed);
                            let result = game.map(|game| GeneratedGame {
                                id,
                                generation,
                                game,
                            });
                            stats.generated_games.fetch_add(1, Ordering::Relaxed);
                            let failed = result.is_err();
                            if !send(&results, &stop, result) || failed {
                                break;
                            }
                        }
                    })?,
            );
        }
        drop(results_tx);
        Ok(Self {
            results,
            next_game,
            _workers: workers,
        })
    }
}

pub struct ArenaTask {
    pub pending: PendingArena,
    pub candidate: Arc<PikafishModel>,
    pub champion: Arc<PikafishModel>,
    pub openings: Vec<Position>,
}

pub struct ArenaOutcome {
    pub pending: PendingArena,
    pub result: Result<(Option<CandidateArenaResult>, Option<CandidateArenaResult>), String>,
    pub seconds: f64,
    pub candidate: Arc<PikafishModel>,
}

pub struct ArenaService {
    pub tasks: Sender<ArenaTask>,
    pub results: Receiver<ArenaOutcome>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl ArenaService {
    pub fn start(
        config: &EvolveConfig,
        reference: Arc<PikafishModel>,
        reference_openings: Vec<Position>,
        stop: Arc<AtomicBool>,
        stats: Arc<PipelineStats>,
    ) -> io::Result<Self> {
        let (tasks, jobs) = bounded::<ArenaTask>(1);
        let (outputs, results) = bounded(1);
        let config = config.clone();
        let cancel = Arc::clone(&stop);
        let thread = std::thread::Builder::new()
            .name("pikafish-arena-service".into())
            .spawn(move || {
                while !cancel.load(Ordering::Relaxed) {
                    let task = match jobs.recv_timeout(Duration::from_millis(100)) {
                        Ok(job) => job,
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
                        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                    };
                    stats.arena_active.store(true, Ordering::Relaxed);
                    stats.arena_pairs.store(0, Ordering::Relaxed);
                    let started = Instant::now();
                    let arena_config = CandidateArenaConfig {
                        pairs: config.arena_pairs,
                        nodes: config.arena_nodes,
                        max_depth: config.max_depth,
                        max_plies: config.max_plies,
                        promotion_rate: config.promotion_rate,
                        confidence_z: config.confidence_z,
                    };
                    let run = |opponent: &PikafishModel, openings: &[Position], offset: usize| {
                        match play_paired_parallel_with_stop(
                            &task.candidate,
                            opponent,
                            openings,
                            arena_config,
                            config.arena_workers,
                            &cancel,
                            |pair, _| {
                                stats.arena_pairs.store(pair + offset, Ordering::Relaxed);
                            },
                        ) {
                            Ok(result) => Ok(Some(result)),
                            Err(message)
                                if message == "arena game reached max_plies without result" =>
                            {
                                Ok(None)
                            }
                            Err(message) => Err(message),
                        }
                    };
                    let result = (|| {
                        let current = run(&task.champion, &task.openings, 0)?;
                        if current.is_none() {
                            return Ok((None, None));
                        }
                        let anchor = run(&reference, &reference_openings, config.arena_pairs)?;
                        Ok((current, anchor))
                    })();
                    stats.arena_active.store(false, Ordering::Relaxed);
                    if !send(
                        &outputs,
                        &cancel,
                        ArenaOutcome {
                            pending: task.pending,
                            result,
                            seconds: started.elapsed().as_secs_f64(),
                            candidate: task.candidate,
                        },
                    ) {
                        break;
                    }
                }
            })?;
        Ok(Self {
            tasks,
            results,
            stop,
            thread: Some(thread),
        })
    }
}

impl Drop for ArenaService {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::{Compression, write::GzEncoder};
    use std::{fs, io::Write};

    #[test]
    fn full_result_queue_applies_backpressure_and_shutdown_joins_workers() -> io::Result<()> {
        let dir = std::env::temp_dir().join(format!("selfplay-queue-{}", std::process::id()));
        fs::create_dir_all(&dir)?;
        let path = dir.join("book.gz");
        let mut encoder = GzEncoder::new(fs::File::create(&path)?, Compression::fast());
        writeln!(
            encoder,
            "[FEN \"3k5/9/9/9/9/9/9/4R4/9/4K4 w - - 119 1\"]\n{{}}"
        )?;
        encoder.finish()?;
        let model = Arc::new(PikafishModel::new(&candle_core::Device::Cpu).map_err(io_error)?);
        let mut config = EvolveConfig::default();
        config.selfplay_workers = 3;
        config.queue_games = 1;
        config.selfplay_nodes = 8;
        config.max_depth = 1;
        config.max_plies = 8;
        config.opening_plies = 0;
        let stop = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(PipelineStats::default());
        let book = Arc::new(Mutex::new(OpeningBook::load(&path, 1)?));
        let champion = Arc::new(RwLock::new(Champion {
            generation: 2,
            model,
        }));
        let service = SelfplayService::start(
            &config,
            book,
            HashSet::new(),
            champion,
            10,
            Arc::clone(&stop),
            Arc::clone(&stats),
        )?;
        let deadline = Instant::now() + Duration::from_secs(5);
        while stats.generated_games.load(Ordering::Relaxed) < 4 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(service.results.len(), 1);
        let generated = stats.generated_games.load(Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(150));
        assert_eq!(stats.generated_games.load(Ordering::Relaxed), generated);
        assert_eq!(generated, 4); // 一局在队列中，三个 worker 最多各持有一局。
        let result = service.results.recv().unwrap()?;
        assert_eq!(result.generation, 2);
        assert!(result.game.red_result.is_some());
        let shutdown = Instant::now();
        drop(service);
        assert!(stop.load(Ordering::Relaxed));
        assert!(shutdown.elapsed() < Duration::from_secs(2));
        fs::remove_file(path)?;
        fs::remove_dir(dir)?;
        Ok(())
    }
}
