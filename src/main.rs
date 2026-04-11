#[cfg(all(target_os = "linux", not(target_env = "musl")))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use chineseai::{
    az::{
        AzArenaReport, AzExperiencePool, AzLoopConfig, AzNnue, AzSearchLimits,
        benchmark_training, gumbel_search, gumbel_search_with_history_and_rules, play_arena_games,
        selfplay_train_iteration_with_pool,
    },
    nnue::{HISTORY_PLIES, HistoryMove},
    pikafish_match::run_vs_pikafish,
    xiangqi::{Position, RuleHistoryEntry, STARTPOS_FEN},
};
use std::{
    fs,
    io::{self, BufRead, BufWriter, Write},
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use tensorboard_rs::summary_writer::SummaryWriter;

// ---------------------------------------------------------------------------
// UCI 磁盘日志（写入程序所在目录的 chineseai-uci.log）
// ---------------------------------------------------------------------------

struct UciLogger {
    file: Option<BufWriter<fs::File>>,
    elapsed: std::time::Instant,
}

impl UciLogger {
    fn new() -> Self {
        let path = std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(|dir| dir.join("chineseai-uci.log")));

        let file = path.and_then(|p| {
            fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
                .ok()
                .map(BufWriter::new)
        });

        let logger = Self {
            file,
            elapsed: std::time::Instant::now(),
        };
        logger
    }

    fn log(&mut self, msg: &str) {
        let Some(f) = self.file.as_mut() else { return };
        let ms = self.elapsed.elapsed().as_millis();
        let _ = writeln!(f, "[{ms:>8}ms] {msg}");
        let _ = f.flush();
    }
}

macro_rules! ulog {
    ($logger:expr, $($arg:tt)*) => {
        $logger.log(&format!($($arg)*))
    };
}

const DEFAULT_AZ_LOOP_CONFIG: &str = "chineseai.azloop.conf";

fn default_parallel_workers() -> usize {
    std::thread::available_parallelism()
        .map(|count| count.get().saturating_sub(1).max(1))
        .unwrap_or(8)
}

fn best_model_path(model_path: &str) -> PathBuf {
    PathBuf::from(format!("{model_path}.best"))
}

/// `{config_path}.progress` — 记录下一轮应使用的全局 `iteration`（TensorBoard / checkpoint / 日志连续）。
fn az_loop_progress_path(config_path: &str) -> PathBuf {
    PathBuf::from(format!("{config_path}.progress"))
}

fn az_loop_replay_snapshot_path(config_path: &str) -> PathBuf {
    PathBuf::from(format!("{config_path}.replay.lz4"))
}

fn read_az_loop_next_iteration(config_path: &str) -> Option<usize> {
    let path = az_loop_progress_path(config_path);
    let text = fs::read_to_string(&path).ok()?;
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if key.trim() == "next_iteration" {
            return value.trim().parse().ok();
        }
    }
    None
}

fn write_az_loop_next_iteration(config_path: &str, next: usize) {
    let path = az_loop_progress_path(config_path);
    fs::write(&path, format!("next_iteration={next}\n")).unwrap_or_else(|err| {
        panic!(
            "failed to write resume cursor `{}`: {err}",
            path.display()
        );
    });
}

/// 将 `tensorboard_logdir` 作为**根目录**，其下追加由训练超参编码的子目录名（仅 `a-z0-9_`，`p`/`m` 代替 `.`/`-`），便于 Web 对照实验。
fn tensorboard_encoded_subdir(config: &AzLoopFileConfig) -> String {
    fn f32_slug(x: f32) -> String {
        if x == 0.0 {
            return "0".to_string();
        }
        let s = format!("{:.8}", x)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
        if s.is_empty() || s == "-" {
            return "0".to_string();
        }
        s.replace('.', "p").replace('-', "m")
    }

    format!(
        concat!(
            "it{}_g{}_sim{}_tk{}_bs{}_lr{}_ep{}_h{}_d{}_mxp{}_wk{}_",
            "gsm{}_tb{}_te{}_tde{}_rg{}_rs{}_mp{}_cpi{}_",
            "ai{}_ags{}_agp{}_sd{}"
        ),
        config.iterations,
        config.games,
        config.simulations,
        config.top_k,
        config.batch_size,
        f32_slug(config.lr),
        config.epochs,
        config.hidden_size,
        config.trunk_depth,
        config.max_plies,
        config.workers,
        f32_slug(config.gumbel_scale),
        f32_slug(config.temperature_start),
        f32_slug(config.temperature_end),
        config.temperature_decay_plies,
        config.replay_games,
        config.replay_samples,
        f32_slug(config.mirror_probability),
        config.checkpoint_interval,
        config.arena_interval,
        f32_slug(config.arena_gumbel_scale),
        config.arena_gumbel_plies,
        config.seed,
    )
}

fn tensorboard_effective_logdir(config: &AzLoopFileConfig) -> PathBuf {
    Path::new(&config.tensorboard_logdir).join(tensorboard_encoded_subdir(config))
}

fn checkpoint_path(model_path: &str, checkpoint_dir: &str, iteration: usize) -> PathBuf {
    let base = Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.nnue");
    Path::new(checkpoint_dir).join(format!("iter-{iteration:04}-{base}"))
}

fn save_checkpoint_copy(model_path: &str, checkpoint_dir: &str, iteration: usize) -> PathBuf {
    fs::create_dir_all(checkpoint_dir).unwrap_or_else(|err| {
        panic!("failed to create checkpoint dir `{checkpoint_dir}`: {err}");
    });
    let path = checkpoint_path(model_path, checkpoint_dir, iteration);
    fs::copy(model_path, &path).unwrap_or_else(|err| {
        panic!("failed to copy `{model_path}` to `{}`: {err}", path.display());
    });
    path
}

fn prune_old_checkpoints(
    model_path: &str,
    checkpoint_dir: &str,
    max_checkpoints: usize,
) -> io::Result<()> {
    if max_checkpoints == 0 {
        return Ok(());
    }
    let base = Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.nnue")
        .to_string();
    let prefix = "iter-";
    let suffix = format!("-{base}");
    let mut entries = fs::read_dir(checkpoint_dir)?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let path = entry.path();
            let name = path.file_name()?.to_str()?;
            (name.starts_with(prefix) && name.ends_with(&suffix)).then_some((name.to_string(), path))
        })
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    let to_remove = entries.len().saturating_sub(max_checkpoints);
    for (_, path) in entries.into_iter().take(to_remove) {
        fs::remove_file(path)?;
    }
    Ok(())
}

fn run_arena_processes(
    candidate_path: &str,
    baseline_path: &str,
    games_per_side: usize,
    simulations: usize,
    top_k: usize,
    max_plies: usize,
    arena_gumbel_scale: f32,
    arena_gumbel_plies: usize,
    process_count: usize,
    seed: u64,
) -> AzArenaReport {
    let process_count = process_count.max(1).min(games_per_side.max(1));
    let exe = std::env::current_exe().unwrap_or_else(|err| {
        panic!("failed to locate current executable: {err}");
    });
    let mut merged = AzArenaReport::default();
    let mut children = Vec::new();
    for index in 0..process_count {
        let red_games = games_per_side / process_count + usize::from(index < games_per_side % process_count);
        let black_games =
            games_per_side / process_count + usize::from(index < games_per_side % process_count);
        if red_games == 0 && black_games == 0 {
            continue;
        }
        let output = Command::new(&exe)
            .arg("az-arena-worker")
            .arg(candidate_path)
            .arg(baseline_path)
            .arg(red_games.to_string())
            .arg(black_games.to_string())
            .arg(simulations.to_string())
            .arg(top_k.to_string())
            .arg(max_plies.to_string())
            .arg(arena_gumbel_scale.to_string())
            .arg(arena_gumbel_plies.to_string())
            .arg((seed ^ index as u64).to_string())
            .output()
            .unwrap_or_else(|err| {
                panic!("failed to spawn arena worker process: {err}");
            });
        if !output.status.success() {
            panic!(
                "arena worker failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        children.push(output);
    }

    for output in children {
        let text = String::from_utf8_lossy(&output.stdout);
        merged.add_assign(&parse_arena_report(&text));
    }
    merged
}

fn parse_arena_report(text: &str) -> AzArenaReport {
    let mut report = AzArenaReport::default();
    for token in text.split_whitespace() {
        let Some((key, value)) = token.split_once('=') else {
            continue;
        };
        let parsed = value.parse::<usize>().unwrap_or(0);
        match key {
            "wins" => report.wins = parsed,
            "losses" => report.losses = parsed,
            "draws" => report.draws = parsed,
            "wins_as_red" => report.wins_as_red = parsed,
            "losses_as_red" => report.losses_as_red = parsed,
            "wins_as_black" => report.wins_as_black = parsed,
            "losses_as_black" => report.losses_as_black = parsed,
            _ => {}
        }
    }
    report
}

fn log_scalar(writer: &mut SummaryWriter, tag: &str, step: usize, value: f32) {
    writer.add_scalar(tag, value, step);
}

fn az_arena_worker_next_usize(
    args: &mut impl Iterator<Item = String>,
    field: &'static str,
) -> usize {
    let text = args.next().unwrap_or_else(|| {
        panic!("az-arena-worker: missing `{field}` (need 10 args after subcommand; see help)")
    });
    text.parse().unwrap_or_else(|_| {
        panic!("az-arena-worker: `{field}` must be usize, got {text:?}")
    })
}

fn az_arena_worker_next_f32(args: &mut impl Iterator<Item = String>, field: &'static str) -> f32 {
    let text = args.next().unwrap_or_else(|| {
        panic!("az-arena-worker: missing `{field}` (need 10 args after subcommand; see help)")
    });
    text.parse().unwrap_or_else(|_| {
        panic!("az-arena-worker: `{field}` must be f32, got {text:?}")
    })
}

fn az_arena_worker_next_u64(args: &mut impl Iterator<Item = String>, field: &'static str) -> u64 {
    let text = args.next().unwrap_or_else(|| {
        panic!("az-arena-worker: missing `{field}` (need 10 args after subcommand; see help)")
    });
    text.parse().unwrap_or_else(|_| {
        panic!("az-arena-worker: `{field}` must be u64, got {text:?}")
    })
}

fn main() {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        None => run_uci(),
        Some("uci") => run_uci(),
        Some("az-init") => {
            let hidden = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(128);
            let output = args.next().unwrap_or_else(|| "chineseai.nnue".into());
            let seed = args
                .next()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(20260409);
            let trunk_depth = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(2);
            let model = AzNnue::random_with_depth(hidden, trunk_depth, seed);
            model.save(&output).unwrap_or_else(|err| {
                panic!("failed to write `{output}`: {err}");
            });
            println!("aznnue   : initialized (nnue binary, magic AZB1)");
            println!("hidden   : {hidden}");
            println!("depth    : {trunk_depth}");
            println!("seed     : {seed}");
            println!("output   : {output}");
        }
        Some("az-gumbel") => {
            let model_path = args.next().unwrap_or_else(|| {
                panic!("usage: az-gumbel <model> [simulations] [top_k] [gumbel_scale] [fen]")
            });
            let simulations = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(10_000);
            let top_k = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(32);
            let gumbel_scale = args
                .next()
                .and_then(|value| value.parse::<f32>().ok())
                .unwrap_or(0.0)
                .max(0.0);
            let fen = args.collect::<Vec<_>>().join(" ");
            let position = parse_position(&fen);
            let model = AzNnue::load(&model_path).unwrap_or_else(|err| {
                panic!("failed to load `{model_path}`: {err}");
            });
            let result = gumbel_search(
                &position,
                &model,
                AzSearchLimits {
                    simulations,
                    top_k,
                    seed: 0,
                    gumbel_scale,
                    workers: 1,
                },
            );
            println!("fen      : {}", position.to_fen());
            println!("model    : {model_path}");
            println!("sims     : {}", result.simulations);
            println!("top_k    : {top_k}");
            println!("gumbel   : {gumbel_scale}");
            println!("value_cp : {}", result.value_cp);
            println!(
                "bestmove : {}",
                result
                    .best_move
                    .map(|mv| mv.to_string())
                    .unwrap_or_else(|| "(none)".into())
            );
            println!(
                "visited_actions: {}",
                result
                    .candidates
                    .iter()
                    .filter(|candidate| candidate.visits > 0)
                    .count()
            );
            println!("by_policy:");
            for candidate in result.candidates.iter().take(top_k) {
                println!(
                    "candidate: {} visits={} q={:.3} prior={:.5} policy={:.5}",
                    candidate.mv, candidate.visits, candidate.q, candidate.prior, candidate.policy
                );
            }
            println!("by_visits:");
            let mut by_visits = result.candidates.clone();
            by_visits.sort_by(|left, right| {
                right
                    .visits
                    .cmp(&left.visits)
                    .then_with(|| right.policy.total_cmp(&left.policy))
                    .then_with(|| right.q.total_cmp(&left.q))
            });
            for candidate in by_visits.iter().take(top_k) {
                println!(
                    "visited: {} visits={} q={:.3} prior={:.5} policy={:.5}",
                    candidate.mv, candidate.visits, candidate.q, candidate.prior, candidate.policy
                );
            }
        }
        Some("az-bench") => {
            let model_path = args.next().unwrap_or_else(|| {
                panic!(
                    "usage: az-bench <model> [simulations] [top_k] [repeat] [gumbel_scale] [fen]"
                )
            });
            let simulations = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(512);
            let top_k = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(32);
            let repeat = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(100)
                .max(1);
            let gumbel_scale = args
                .next()
                .and_then(|value| value.parse::<f32>().ok())
                .unwrap_or(0.0)
                .max(0.0);
            let fen = args.collect::<Vec<_>>().join(" ");
            let position = parse_position(&fen);
            let model = AzNnue::load(&model_path).unwrap_or_else(|err| {
                panic!("failed to load `{model_path}`: {err}");
            });

            let _ = gumbel_search(
                &position,
                &model,
                AzSearchLimits {
                    simulations,
                    top_k,
                    seed: 0,
                    gumbel_scale,
                    workers: 1,
                },
            );

            let started = std::time::Instant::now();
            let mut total_sims = 0usize;
            let mut best_move = None;
            for iteration in 0..repeat {
                let result = gumbel_search(
                    &position,
                    &model,
                    AzSearchLimits {
                        simulations,
                        top_k,
                        seed: iteration as u64,
                        gumbel_scale,
                        workers: 1,
                    },
                );
                total_sims += result.simulations;
                best_move = result.best_move;
            }
            let elapsed = started.elapsed();
            let elapsed_secs = elapsed.as_secs_f64().max(f64::EPSILON);
            println!("bench        : fixed-search");
            println!("model        : {model_path}");
            println!("fen          : {}", position.to_fen());
            println!("sims/search  : {simulations}");
            println!("top_k        : {top_k}");
            println!("repeat       : {repeat}");
            println!("gumbel       : {gumbel_scale}");
            println!("total_sims   : {total_sims}");
            println!("elapsed_ms   : {:.3}", elapsed.as_secs_f64() * 1000.0);
            println!(
                "ms/search    : {:.3}",
                elapsed.as_secs_f64() * 1000.0 / repeat as f64
            );
            println!("sims/sec     : {:.0}", total_sims as f64 / elapsed_secs);
            println!(
                "last_bestmove: {}",
                best_move
                    .map(|mv| mv.to_string())
                    .unwrap_or_else(|| "(none)".into())
            );
        }
        Some("az-train-bench") => {
            let model_path = args.next().unwrap_or_else(|| {
                panic!(
                    "usage: az-train-bench <model> [samples] [epochs] [batch_size] [lr] [seed]"
                )
            });
            let sample_count = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(8192)
                .max(1);
            let epochs = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(2)
                .max(1);
            let batch_size = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(1024)
                .max(1);
            let lr = args
                .next()
                .and_then(|value| value.parse::<f32>().ok())
                .unwrap_or(0.0003)
                .max(0.0);
            let seed = args
                .next()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(20260411);
            let mut model = AzNnue::load(&model_path).unwrap_or_else(|err| {
                panic!("failed to load `{model_path}`: {err}");
            });
            let started = std::time::Instant::now();
            let stats = benchmark_training(&mut model, sample_count, epochs, batch_size, lr, seed);
            let elapsed = started.elapsed().as_secs_f64().max(f64::EPSILON);
            let processed = (sample_count * epochs) as f64;
            println!("bench        : training");
            println!("model        : {model_path}");
            println!("samples      : {sample_count}");
            println!("epochs       : {epochs}");
            println!("batch_size   : {batch_size}");
            println!("lr           : {lr}");
            println!("elapsed_ms   : {:.3}", elapsed * 1000.0);
            println!("processed    : {}", sample_count * epochs);
            println!("samples/sec  : {:.0}", processed / elapsed);
            println!("loss         : {:.4}", stats.loss);
            println!("value_mse    : {:.4}", stats.value_loss);
            println!("policy_ce    : {:.4}", stats.policy_ce);
        }
        Some("az-loop") => {
            let config_path = args.next().unwrap_or_else(|| DEFAULT_AZ_LOOP_CONFIG.into());
            let Some(config) = load_or_create_az_loop_config(&config_path) else {
                return;
            };
            let start_iteration = read_az_loop_next_iteration(&config_path)
                .unwrap_or(1)
                .max(1);
            if start_iteration > 1 {
                println!(
                    "resume   : global_iteration starts at {} (from `{}`)",
                    start_iteration,
                    az_loop_progress_path(&config_path).display()
                );
            }
            let best_path = best_model_path(&config.model_path);

            let mut model = if Path::new(&config.model_path).exists() {
                println!("model    : load {}", config.model_path);
                match AzNnue::load(&config.model_path) {
                    Ok(model) => model,
                    Err(err) => {
                        println!(
                            "model    : reinit {} as random nnue ({err})",
                            config.model_path
                        );
                        AzNnue::random_with_depth(
                            config.hidden_size,
                            config.trunk_depth,
                            config.seed,
                        )
                    }
                }
            } else {
                println!("model    : init {}", config.model_path);
                AzNnue::random_with_depth(config.hidden_size, config.trunk_depth, config.seed)
            };
            let replay_snapshot_path = az_loop_replay_snapshot_path(&config_path);
            let mut replay_pool =
                (config.replay_games > 0).then(|| AzExperiencePool::new(config.replay_games));
            if config.replay_games > 0 && replay_snapshot_path.exists() {
                match AzExperiencePool::load_snapshot_lz4(&replay_snapshot_path, config.replay_games)
                {
                    Ok(pool) => {
                        fs::remove_file(&replay_snapshot_path).unwrap_or_else(|err| {
                            panic!(
                                "failed to remove replay snapshot `{}`: {err}",
                                replay_snapshot_path.display()
                            );
                        });
                        println!(
                            "replay   : restored {} games / {} samples from `{}` (file removed)",
                            pool.game_count(),
                            pool.sample_count(),
                            replay_snapshot_path.display()
                        );
                        replay_pool = Some(pool);
                    }
                    Err(err) => {
                        eprintln!(
                            "replay   : corrupt snapshot `{}`: {err}; removing",
                            replay_snapshot_path.display()
                        );
                        let _ = fs::remove_file(&replay_snapshot_path);
                    }
                }
            }
            let interrupted = Arc::new(AtomicBool::new(false));
            let interrupted_flag = interrupted.clone();
            ctrlc::set_handler(move || {
                interrupted_flag.store(true, Ordering::SeqCst);
            })
            .unwrap_or_else(|err| panic!("failed to register Ctrl+C handler: {err}"));
            let tb_dir = tensorboard_effective_logdir(&config);
            fs::create_dir_all(&tb_dir).unwrap_or_else(|err| {
                panic!(
                    "failed to create tensorboard log dir `{}`: {err}",
                    tb_dir.display()
                );
            });
            let mut tb = SummaryWriter::new(&tb_dir);

            println!(
                "loop     : config={} iterations={} games={} sims={} top_k={} epochs={} lr={} batch_size={} max_plies={} workers={} temp={}->{}/{}ply gumbel_scale={} depth={} replay_games={} replay_samples={} mirror_probability={} checkpoint_interval={} max_checkpoints={} arena_interval={} arena_games_per_side={} arena_gumbel_scale={} arena_gumbel_plies={} arena_processes={} tb_base={} tb_run={}",
                config_path,
                config.iterations,
                config.games,
                config.simulations,
                config.top_k,
                config.epochs,
                config.lr,
                config.batch_size,
                config.max_plies,
                config.workers,
                config.temperature_start,
                config.temperature_end,
                config.temperature_decay_plies,
                config.gumbel_scale,
                config.trunk_depth,
                config.replay_games,
                config.replay_samples,
                config.mirror_probability,
                config.checkpoint_interval,
                config.max_checkpoints,
                config.arena_interval,
                config.arena_games_per_side,
                config.arena_gumbel_scale,
                config.arena_gumbel_plies,
                config.arena_processes,
                config.tensorboard_logdir,
                tensorboard_encoded_subdir(&config)
            );
            let mut exited_after_ctrl_c = false;
            for offset in 0..config.iterations {
                if interrupted.load(Ordering::SeqCst) {
                    if let Some(ref pool) = replay_pool {
                        match pool.save_snapshot_lz4(&replay_snapshot_path) {
                            Ok(()) => {
                                if pool.game_count() > 0 {
                                    println!(
                                        "replay   : interrupt snapshot `{}` ({} games)",
                                        replay_snapshot_path.display(),
                                        pool.game_count()
                                    );
                                }
                            }
                            Err(err) => eprintln!("replay   : failed to write snapshot: {err}"),
                        }
                    }
                    exited_after_ctrl_c = true;
                    break;
                }
                let iteration = start_iteration + offset;
                let started = std::time::Instant::now();
                let report = selfplay_train_iteration_with_pool(
                    &mut model,
                    &AzLoopConfig {
                        games: config.games,
                        max_plies: config.max_plies,
                        simulations: config.simulations,
                        top_k: config.top_k,
                        epochs: config.epochs,
                        lr: config.lr,
                        batch_size: config.batch_size,
                        seed: config.seed ^ iteration as u64,
                        workers: config.workers,
                        temperature_start: config.temperature_start,
                        temperature_end: config.temperature_end,
                        temperature_decay_plies: config.temperature_decay_plies,
                        gumbel_scale: config.gumbel_scale,
                        replay_games: config.replay_games,
                        replay_samples: config.replay_samples,
                        mirror_probability: config.mirror_probability,
                    },
                    replay_pool.as_mut(),
                );
                model.save(&config.model_path).unwrap_or_else(|err| {
                    panic!("failed to write `{}`: {err}", config.model_path);
                });
                write_az_loop_next_iteration(&config_path, iteration.saturating_add(1));
                if !best_path.exists() {
                    fs::copy(&config.model_path, &best_path).unwrap_or_else(|err| {
                        panic!(
                            "failed to initialize best model `{}` from `{}`: {err}",
                            best_path.display(),
                            config.model_path
                        );
                    });
                }
                let checkpoint_saved = if config.checkpoint_interval > 0
                    && iteration % config.checkpoint_interval == 0
                {
                    let path = save_checkpoint_copy(
                        &config.model_path,
                        &config.checkpoint_dir,
                        iteration,
                    );
                    prune_old_checkpoints(
                        &config.model_path,
                        &config.checkpoint_dir,
                        config.max_checkpoints,
                    )
                    .unwrap_or_else(|err| {
                        panic!(
                            "failed to prune checkpoints in `{}`: {err}",
                            config.checkpoint_dir
                        );
                    });
                    Some(path)
                } else {
                    None
                };
                println!(
                    "iter {iteration:04}: games={} samples={} train_samples={} pool={}/{} W/B/D={}/{}/{} avg_plies={:.1} loss={:.4} value_mse={:.4} policy_ce={:.4} lr={:.6} tempH={:.3}/{:.3} drawR[120/rep/lchk/lchs]={}/{}/{}/{} selfplay={:.1}s train={:.1}s gps={:.2} sps={:.1} train_sps={:.1} elapsed={:.1}s saved={}{}",
                    report.games,
                    report.samples,
                    report.train_samples,
                    report.pool_games,
                    report.pool_samples,
                    report.red_wins,
                    report.black_wins,
                    report.draws,
                    report.avg_plies,
                    report.loss,
                    report.value_loss,
                    report.policy_ce,
                    config.lr,
                    report.temperature_early_entropy,
                    report.temperature_mid_entropy,
                    report.terminal_rule_draw_halfmove120,
                    report.terminal_rule_draw_repetition,
                    report.terminal_rule_draw_mutual_long_check,
                    report.terminal_rule_draw_mutual_long_chase,
                    report.selfplay_seconds,
                    report.train_seconds,
                    report.games_per_second,
                    report.samples_per_second,
                    report.train_samples_per_second,
                    started.elapsed().as_secs_f32(),
                    config.model_path,
                    checkpoint_saved.as_ref().map_or_else(
                        String::new,
                        |path| format!(" checkpoint={}", path.display())
                    )
                );
                log_scalar(&mut tb, "train/loss", iteration, report.loss);
                log_scalar(&mut tb, "train/value_mse", iteration, report.value_loss);
                log_scalar(&mut tb, "train/policy_ce", iteration, report.policy_ce);
                log_scalar(&mut tb, "train/lr", iteration, config.lr);
                log_scalar(&mut tb, "selfplay/games", iteration, report.games as f32);
                log_scalar(&mut tb, "selfplay/samples", iteration, report.samples as f32);
                log_scalar(&mut tb, "selfplay/avg_plies", iteration, report.avg_plies);
                log_scalar(&mut tb, "selfplay/temp_entropy_early", iteration, report.temperature_early_entropy);
                log_scalar(&mut tb, "selfplay/temp_entropy_mid", iteration, report.temperature_mid_entropy);
                log_scalar(&mut tb, "selfplay/games_per_second", iteration, report.games_per_second);
                log_scalar(&mut tb, "selfplay/samples_per_second", iteration, report.samples_per_second);
                log_scalar(&mut tb, "train/samples_per_second", iteration, report.train_samples_per_second);
                log_scalar(&mut tb, "timing/selfplay_seconds", iteration, report.selfplay_seconds);
                log_scalar(&mut tb, "timing/train_seconds", iteration, report.train_seconds);
                log_scalar(&mut tb, "timing/iteration_seconds", iteration, report.total_seconds);
                log_scalar(&mut tb, "outcome/red_wins", iteration, report.red_wins as f32);
                log_scalar(&mut tb, "outcome/black_wins", iteration, report.black_wins as f32);
                log_scalar(&mut tb, "outcome/draws", iteration, report.draws as f32);
                log_scalar(&mut tb, "terminal/no_legal_moves", iteration, report.terminal_no_legal_moves as f32);
                log_scalar(&mut tb, "terminal/red_general_missing", iteration, report.terminal_red_general_missing as f32);
                log_scalar(&mut tb, "terminal/black_general_missing", iteration, report.terminal_black_general_missing as f32);
                log_scalar(&mut tb, "terminal/rule_draw", iteration, report.terminal_rule_draw as f32);
                log_scalar(&mut tb, "terminal/rule_draw_halfmove120", iteration, report.terminal_rule_draw_halfmove120 as f32);
                log_scalar(&mut tb, "terminal/rule_draw_repetition", iteration, report.terminal_rule_draw_repetition as f32);
                log_scalar(&mut tb, "terminal/rule_draw_mutual_long_check", iteration, report.terminal_rule_draw_mutual_long_check as f32);
                log_scalar(&mut tb, "terminal/rule_draw_mutual_long_chase", iteration, report.terminal_rule_draw_mutual_long_chase as f32);
                log_scalar(&mut tb, "terminal/rule_win_red", iteration, report.terminal_rule_win_red as f32);
                log_scalar(&mut tb, "terminal/rule_win_black", iteration, report.terminal_rule_win_black as f32);
                log_scalar(&mut tb, "terminal/max_plies", iteration, report.terminal_max_plies as f32);
                if config.arena_interval > 0 && iteration % config.arena_interval == 0 {
                    let arena = run_arena_processes(
                        &config.model_path,
                        best_path.to_str().unwrap_or_else(|| {
                            panic!("best model path is not valid UTF-8")
                        }),
                        config.arena_games_per_side,
                        config.simulations,
                        config.top_k,
                        config.max_plies,
                        config.arena_gumbel_scale,
                        config.arena_gumbel_plies,
                        config.arena_processes,
                        config.seed ^ (iteration as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                    );
                    let promoted = arena.score()
                        > config.arena_games_per_side as f32;
                    if promoted {
                        fs::copy(&config.model_path, &best_path).unwrap_or_else(|err| {
                            panic!(
                                "failed to promote `{}` to `{}`: {err}",
                                config.model_path,
                                best_path.display()
                            );
                        });
                    }
                    println!(
                        "arena {iteration:04}: total={} W/L/D={}/{}/{} red={}/{} black={}/{} score={:.1} best={}{}",
                        arena.total_games(),
                        arena.wins,
                        arena.losses,
                        arena.draws,
                        arena.wins_as_red,
                        arena.losses_as_red,
                        arena.wins_as_black,
                        arena.losses_as_black,
                        arena.score(),
                        best_path.display(),
                        if promoted { " promoted=current" } else { "" }
                    );
                    log_scalar(&mut tb, "arena/wins", iteration, arena.wins as f32);
                    log_scalar(&mut tb, "arena/losses", iteration, arena.losses as f32);
                    log_scalar(&mut tb, "arena/draws", iteration, arena.draws as f32);
                    log_scalar(&mut tb, "arena/score", iteration, arena.score());
                    log_scalar(&mut tb, "arena/win_rate", iteration, arena.wins as f32 / arena.total_games().max(1) as f32);
                    log_scalar(&mut tb, "arena/wins_as_red", iteration, arena.wins_as_red as f32);
                    log_scalar(&mut tb, "arena/losses_as_red", iteration, arena.losses_as_red as f32);
                    log_scalar(&mut tb, "arena/wins_as_black", iteration, arena.wins_as_black as f32);
                    log_scalar(&mut tb, "arena/losses_as_black", iteration, arena.losses_as_black as f32);
                    log_scalar(&mut tb, "arena/promoted", iteration, if promoted { 1.0 } else { 0.0 });
                }
                if interrupted.load(Ordering::SeqCst) {
                    if let Some(ref pool) = replay_pool {
                        match pool.save_snapshot_lz4(&replay_snapshot_path) {
                            Ok(()) => {
                                if pool.game_count() > 0 {
                                    println!(
                                        "replay   : interrupt snapshot `{}` ({} games)",
                                        replay_snapshot_path.display(),
                                        pool.game_count()
                                    );
                                }
                            }
                            Err(err) => eprintln!("replay   : failed to write snapshot: {err}"),
                        }
                    }
                    exited_after_ctrl_c = true;
                    break;
                }
            }
            if !exited_after_ctrl_c {
                let _ = fs::remove_file(&replay_snapshot_path);
            }
        }
        Some("az-arena-worker") => {
            let candidate_path = args.next().unwrap_or_else(|| {
                panic!("az-arena-worker: missing <candidate> (see help)")
            });
            let baseline_path = args.next().unwrap_or_else(|| {
                panic!("az-arena-worker: missing <baseline>")
            });
            let red_games = az_arena_worker_next_usize(&mut args, "red_games");
            let black_games = az_arena_worker_next_usize(&mut args, "black_games");
            let simulations = az_arena_worker_next_usize(&mut args, "simulations").max(1);
            let top_k = az_arena_worker_next_usize(&mut args, "top_k").max(1);
            let max_plies = az_arena_worker_next_usize(&mut args, "max_plies").max(1);
            let arena_gumbel_scale = az_arena_worker_next_f32(&mut args, "arena_gumbel_scale").max(0.0);
            let arena_gumbel_plies = az_arena_worker_next_usize(&mut args, "arena_gumbel_plies");
            let seed = az_arena_worker_next_u64(&mut args, "seed");
            if args.next().is_some() {
                panic!("az-arena-worker: trailing arguments (expected exactly 10 after candidate and baseline)");
            }
            let candidate = AzNnue::load(&candidate_path).unwrap_or_else(|err| {
                panic!("failed to load `{candidate_path}`: {err}");
            });
            let baseline = AzNnue::load(&baseline_path).unwrap_or_else(|err| {
                panic!("failed to load `{baseline_path}`: {err}");
            });
            let report = play_arena_games(
                &candidate,
                &baseline,
                simulations,
                top_k,
                max_plies,
                red_games,
                black_games,
                seed,
                arena_gumbel_scale,
                arena_gumbel_plies,
            );
            println!(
                "wins={} losses={} draws={} wins_as_red={} losses_as_red={} wins_as_black={} losses_as_black={}",
                report.wins,
                report.losses,
                report.draws,
                report.wins_as_red,
                report.losses_as_red,
                report.wins_as_black,
                report.losses_as_black
            );
        }
        Some("perft") => {
            let depth = args
                .next()
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or(1);
            let fen = args.collect::<Vec<_>>().join(" ");
            let position = parse_position(&fen);
            println!("fen   : {}", position.to_fen());
            println!("depth : {depth}");
            println!("nodes : {}", position.perft(depth));
        }
        Some("vs-pikafish") => {
            let pikafish_exe = args.next().unwrap_or_else(|| {
                panic!(
                    "usage: vs-pikafish <pikafish_exe> [model.nnue] [movetime_ms] [games] [max_plies] [simulations] [top_k] [parallel_games]"
                )
            });
            let model_path = args.next().unwrap_or_else(|| "chineseai.nnue".into());
            let movetime_ms = args
                .next()
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or(10)
                .max(1);
            let games = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(20)
                .max(1);
            let max_plies = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(300)
                .max(1);
            let simulations = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(10_000)
                .max(1);
            let top_k = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(32)
                .max(1);
            let parallel_games = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(5)
                .max(1);
            let seed = 20260411_u64;
            let summary = run_vs_pikafish(
                Path::new(&pikafish_exe),
                Path::new(&model_path),
                movetime_ms,
                games,
                max_plies,
                simulations,
                top_k,
                seed,
                parallel_games,
            )
            .unwrap_or_else(|err| panic!("vs-pikafish failed: {err}"));
            println!(
                "vs-pikafish: games={} parallel={} chinese W/L/D={}/{}/{} (as_red={} as_black={}) | pikafish movetime={}ms max_plies={} sims={} top_k={}",
                summary.total_games,
                parallel_games.min(games),
                summary.chinese_wins,
                summary.chinese_losses,
                summary.draws,
                summary.chinese_wins_as_red,
                summary.chinese_wins_as_black,
                movetime_ms,
                max_plies,
                simulations,
                top_k
            );
        }
        Some("help") | _ => print_help(),
    }
}

fn print_help() {
    let position = Position::startpos();
    println!("ChineseAI AZ-NNUE Gumbel core");
    println!("start : {STARTPOS_FEN}");
    println!("moves : {}", position.legal_moves().len());
    println!("hint  : cargo run --release -- az-init 128 chineseai.nnue 20260409 2");
    println!("hint  : cargo run --release -- uci");
    println!("hint  : cargo run --release -- az-gumbel chineseai.nnue 10000 32 0.0 startpos");
    println!("hint  : cargo run --release -- az-bench chineseai.nnue 512 32 100 0.0 startpos");
    println!("hint  : cargo run --release -- az-train-bench chineseai.nnue 8192 2 1024 0.0003");
    println!("hint  : cargo run --release -- az-loop {DEFAULT_AZ_LOOP_CONFIG}");
    println!("hint  : az-arena-worker <cand> <base> <red_n> <black_n> <sims> <top_k> <max_plies> <arena_gumbel_scale> <arena_gumbel_plies> <seed>");
    println!("hint  : cargo run --release -- vs-pikafish ./pikafish chineseai.nnue 10 40 300 10000 32 5");
}

#[derive(Clone, Debug)]
struct AzLoopFileConfig {
    model_path: String,
    iterations: usize,
    games: usize,
    simulations: usize,
    top_k: usize,
    epochs: usize,
    lr: f32,
    batch_size: usize,
    max_plies: usize,
    hidden_size: usize,
    trunk_depth: usize,
    seed: u64,
    workers: usize,
    temperature_start: f32,
    temperature_end: f32,
    temperature_decay_plies: usize,
    gumbel_scale: f32,
    replay_games: usize,
    replay_samples: usize,
    mirror_probability: f32,
    checkpoint_interval: usize,
    checkpoint_dir: String,
    max_checkpoints: usize,
    arena_interval: usize,
    arena_games_per_side: usize,
    arena_gumbel_scale: f32,
    arena_gumbel_plies: usize,
    arena_processes: usize,
    tensorboard_logdir: String,
}

impl Default for AzLoopFileConfig {
    fn default() -> Self {
        let default_workers = default_parallel_workers();
        Self {
            model_path: "chineseai.nnue".into(),
            iterations: 100,
            games: 400,
            simulations: 512,
            top_k: 16,
            epochs: 2,
            lr: 0.0003,
            batch_size: 1024,
            max_plies: 300,
            hidden_size: 128,
            trunk_depth: 2,
            seed: 20260409,
            workers: default_workers,
            temperature_start: 1.0,
            temperature_end: 0.1,
            temperature_decay_plies: 30,
            gumbel_scale: 1.0,
            replay_games: 5000,
            replay_samples: 0,
            mirror_probability: 0.3,
            checkpoint_interval: 20,
            checkpoint_dir: "checkpoints".into(),
            max_checkpoints: 50,
            arena_interval: 20,
            arena_games_per_side: 50,
            arena_gumbel_scale: 1.0,
            arena_gumbel_plies: 8,
            arena_processes: default_workers,
            tensorboard_logdir: "runs/chineseai".into(),
        }
    }
}

impl AzLoopFileConfig {
    fn to_file_text(&self) -> String {
        format!(
            r#"# ChineseAI AZ self-play training config.
# Run: ./target/release/chineseai az-loop {DEFAULT_AZ_LOOP_CONFIG}
#
# model_path: AzNnue binary (magic AZB1, little-endian f32), e.g. chineseai.nnue
#
# Value targets (AlphaZero-style): each training position uses only the final game outcome,
# encoded from the side-to-move at that position (via side_sign × game_result). MCTS root
# is not mixed into the value label.
#
# Self-play policy temperature (linear in ply index, 0-based before each search):
#   temperature_start -> temperature_end over plies [0, temperature_decay_plies), then constant.
#   Defaults: 1.0 -> 0.1 by ply 30 (sharper sampling than 0.2@40); raise decay_plies if openings collapse.
#
# Replay:
#   replay_games keeps the most recent N complete games in memory.
#   replay_samples=0 trains on about the same number of positions as newly generated this iteration.
#   set replay_samples larger, e.g. 200000, to train more from the pool per iteration.
#   Ctrl+C writes "<this_conf_filename>.replay.lz4" (LZ4); next az-loop loads it into the pool then
#   deletes the file. A full run without interrupt removes any leftover snapshot at exit.
#   Replay snapshot format is versioned; older .replay.lz4 files from previous formats are rejected.
#
# Optimizer:
#   AdamW is used with mini-batch gradient accumulation.
#   batch_size=1024 is the default; lower it if memory is tight, raise it if loss is noisy.
#   workers defaults to about all logical CPUs minus one.
#   lr=0.0003 is a safer default than the old SGD-style 0.001 for self-play targets.
#
# Augmentation:
#   mirror_probability mirrors board files a<->i for this fraction of training samples.
#   Xiangqi rules are left/right symmetric, so value stays unchanged and policy moves are mirrored.
#
# Checkpoints & Arena:
#   Training appends "<this_conf_filename>.progress" with next_iteration=... after each global iter.
#   Delete that file to reset the global iteration counter to 1 (TensorBoard/checkpoint numbering).
#   checkpoint_interval saves a timestamp-free numbered copy every N iterations.
#   max_checkpoints keeps only the newest N checkpoint files in checkpoint_dir.
#   arena_interval runs current-vs-best evaluation every N iterations.
#   arena_games_per_side=50 means 50 games as Red and 50 as Black.
#   arena_gumbel_scale: root Gumbel noise for the first arena_gumbel_plies half-moves only (then 0).
#   arena_gumbel_plies=0 means no opening noise (fully deterministic).
#   tensorboard_logdir is the ROOT; each run writes under a subdir whose name encodes it_*, g_*,
#   sim_*, bs_*, lr_*, … so TensorBoard Web can compare experiments side by side.

model_path = {model_path}
iterations = {iterations}
games = {games}
simulations = {simulations}
top_k = {top_k}
epochs = {epochs}
lr = {lr}
batch_size = {batch_size}
max_plies = {max_plies}
hidden_size = {hidden_size}
trunk_depth = {trunk_depth}
seed = {seed}
workers = {workers}
temperature_start = {temperature_start}
temperature_end = {temperature_end}
temperature_decay_plies = {temperature_decay_plies}
gumbel_scale = {gumbel_scale}
replay_games = {replay_games}
replay_samples = {replay_samples}
mirror_probability = {mirror_probability}
checkpoint_interval = {checkpoint_interval}
checkpoint_dir = {checkpoint_dir}
max_checkpoints = {max_checkpoints}
arena_interval = {arena_interval}
arena_games_per_side = {arena_games_per_side}
arena_gumbel_scale = {arena_gumbel_scale}
arena_gumbel_plies = {arena_gumbel_plies}
arena_processes = {arena_processes}
tensorboard_logdir = {tensorboard_logdir}
"#,
            model_path = self.model_path,
            iterations = self.iterations,
            games = self.games,
            simulations = self.simulations,
            top_k = self.top_k,
            epochs = self.epochs,
            lr = self.lr,
            batch_size = self.batch_size,
            max_plies = self.max_plies,
            hidden_size = self.hidden_size,
            trunk_depth = self.trunk_depth,
            seed = self.seed,
            workers = self.workers,
            temperature_start = self.temperature_start,
            temperature_end = self.temperature_end,
            temperature_decay_plies = self.temperature_decay_plies,
            gumbel_scale = self.gumbel_scale,
            replay_games = self.replay_games,
            replay_samples = self.replay_samples,
            mirror_probability = self.mirror_probability,
            checkpoint_interval = self.checkpoint_interval,
            checkpoint_dir = self.checkpoint_dir,
            max_checkpoints = self.max_checkpoints,
            arena_interval = self.arena_interval,
            arena_games_per_side = self.arena_games_per_side,
            arena_gumbel_scale = self.arena_gumbel_scale,
            arena_gumbel_plies = self.arena_gumbel_plies,
            arena_processes = self.arena_processes,
            tensorboard_logdir = self.tensorboard_logdir,
        )
    }

    fn parse(text: &str) -> Self {
        let mut config = Self::default();
        for line in text.lines() {
            let line = line.split('#').next().unwrap_or_default().trim();
            if line.is_empty() {
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            config.set(key.trim(), value.trim());
        }
        config.normalize()
    }

    fn set(&mut self, key: &str, value: &str) {
        match key {
            "model_path" => self.model_path = value.to_string(),
            "iterations" => self.iterations = parse_config_value(value, self.iterations),
            "games" => self.games = parse_config_value(value, self.games),
            "simulations" => self.simulations = parse_config_value(value, self.simulations),
            "top_k" => self.top_k = parse_config_value(value, self.top_k),
            "epochs" => self.epochs = parse_config_value(value, self.epochs),
            "lr" => self.lr = parse_config_value(value, self.lr),
            "batch_size" => self.batch_size = parse_config_value(value, self.batch_size),
            "max_plies" => self.max_plies = parse_config_value(value, self.max_plies),
            "hidden_size" => self.hidden_size = parse_config_value(value, self.hidden_size),
            "trunk_depth" => self.trunk_depth = parse_config_value(value, self.trunk_depth),
            "seed" => self.seed = parse_config_value(value, self.seed),
            "workers" => self.workers = parse_config_value(value, self.workers),
            "temperature_start" => {
                self.temperature_start = parse_config_value(value, self.temperature_start)
            }
            "temperature_end" => {
                self.temperature_end = parse_config_value(value, self.temperature_end)
            }
            "temperature_decay_plies" => {
                self.temperature_decay_plies =
                    parse_config_value(value, self.temperature_decay_plies)
            }
            "gumbel_scale" => self.gumbel_scale = parse_config_value(value, self.gumbel_scale),
            "replay_games" => self.replay_games = parse_config_value(value, self.replay_games),
            "replay_samples" => {
                self.replay_samples = parse_config_value(value, self.replay_samples)
            }
            "mirror_probability" => {
                self.mirror_probability = parse_config_value(value, self.mirror_probability)
            }
            "checkpoint_interval" => {
                self.checkpoint_interval = parse_config_value(value, self.checkpoint_interval)
            }
            "checkpoint_dir" => self.checkpoint_dir = value.to_string(),
            "max_checkpoints" => {
                self.max_checkpoints = parse_config_value(value, self.max_checkpoints)
            }
            "arena_interval" => self.arena_interval = parse_config_value(value, self.arena_interval),
            "arena_games_per_side" => {
                self.arena_games_per_side = parse_config_value(value, self.arena_games_per_side)
            }
            "arena_gumbel_scale" => {
                self.arena_gumbel_scale = parse_config_value(value, self.arena_gumbel_scale)
            }
            "arena_gumbel_plies" => {
                self.arena_gumbel_plies = parse_config_value(value, self.arena_gumbel_plies)
            }
            "arena_processes" => {
                self.arena_processes = parse_config_value(value, self.arena_processes)
            }
            "tensorboard_logdir" => self.tensorboard_logdir = value.to_string(),
            _ => {}
        }
    }

    fn normalize(mut self) -> Self {
        self.iterations = self.iterations.max(1);
        self.games = self.games.max(1);
        self.simulations = self.simulations.max(1);
        self.top_k = self.top_k.max(1);
        self.epochs = self.epochs.max(1);
        self.batch_size = self.batch_size.max(1);
        self.max_plies = self.max_plies.max(1);
        self.hidden_size = self.hidden_size.max(1);
        self.workers = self.workers.max(1);
        self.temperature_start = self.temperature_start.max(0.0);
        self.temperature_end = self.temperature_end.max(0.0);
        self.gumbel_scale = self.gumbel_scale.max(0.0);
        self.arena_gumbel_scale = self.arena_gumbel_scale.max(0.0);
        self.arena_gumbel_plies = self.arena_gumbel_plies.max(0);
        self.mirror_probability = self.mirror_probability.clamp(0.0, 1.0);
        self.max_checkpoints = self.max_checkpoints.max(1);
        self.arena_games_per_side = self.arena_games_per_side.max(1);
        self.arena_processes = self.arena_processes.max(1);
        self
    }
}

fn load_or_create_az_loop_config(path: &str) -> Option<AzLoopFileConfig> {
    if !Path::new(path).exists() {
        let config = AzLoopFileConfig::default();
        fs::write(path, config.to_file_text()).unwrap_or_else(|err| {
            panic!("failed to create `{path}`: {err}");
        });
        println!("created config: {path}");
        println!("edit it, then run: ./target/release/chineseai az-loop {path}");
        return None;
    }
    let text = fs::read_to_string(path).unwrap_or_else(|err| {
        panic!("failed to read `{path}`: {err}");
    });
    Some(AzLoopFileConfig::parse(&text))
}

fn parse_config_value<T: std::str::FromStr>(text: &str, default: T) -> T {
    text.parse::<T>().unwrap_or(default)
}

#[derive(Clone, Debug)]
struct UciState {
    position: Position,
    history: Vec<HistoryMove>,
    rule_history: Vec<RuleHistoryEntry>,
    eval_file: String,
    model: Option<AzNnue>,
    simulations: usize,
    top_k: usize,
    threads: usize,
    gumbel_scale: f32,
    policy_debug: bool,
    policy_debug_limit: usize,
    seed: u64,
}

impl Default for UciState {
    fn default() -> Self {
        Self {
            position: Position::startpos(),
            history: Vec::new(),
            rule_history: Position::startpos().initial_rule_history(),
            eval_file: "chineseai.nnue".into(),
            model: None,
            simulations: 10_000,
            top_k: 32,
            threads: 1,
            gumbel_scale: 0.0,
            policy_debug: false,
            policy_debug_limit: 16,
            seed: 20260409,
        }
    }
}

fn run_uci() {
    let stdin = io::stdin();
    let mut state = UciState::default();
    let mut logger = UciLogger::new();
    ulog!(logger, "=== UCI session started ===");
    for line in stdin.lock().lines() {
        let Ok(line) = line else {
            break;
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        ulog!(logger, ">> {line}");
        match line.split_whitespace().next() {
            Some("uci") => print_uci_id(),
            Some("isready") => {
                ensure_uci_model(&mut state);
                println!("readyok");
                uci_flush();
            }
            Some("ucinewgame") => {
                state.position = Position::startpos();
                state.history.clear();
                state.rule_history = state.position.initial_rule_history();
                state.seed = 20260409;
                ulog!(logger, "[ucinewgame] reset to startpos");
            }
            Some("setoption") => handle_setoption(line, &mut state),
            Some("position") => handle_position(line, &mut state, &mut logger),
            Some("go") => handle_go(line, &mut state, &mut logger),
            Some("stop") => {}
            Some("quit") => {
                ulog!(logger, "=== UCI session ended ===");
                break;
            }
            _ => {}
        }
    }
}

fn print_uci_id() {
    println!("id name ChineseAI AZ-NNUE");
    println!("id author ChineseAI");
    println!("option name EvalFile type string default chineseai.nnue");
    println!("option name Simulations type spin default 10000 min 1 max 100000000");
    println!("option name TopK type spin default 32 min 1 max 256");
    println!("option name Threads type spin default 1 min 1 max 1");
    println!("option name GumbelScale type string default 0.0");
    println!("option name PolicyDebug type check default false");
    println!("option name PolicyDebugLimit type spin default 16 min 1 max 256");
    println!("uciok");
    uci_flush();
}

fn ensure_uci_model(state: &mut UciState) {
    if state.model.is_some() {
        return;
    }
    state.model = Some(AzNnue::load(&state.eval_file).unwrap_or_else(|err| {
        println!(
            "info string failed to load {}, using random model: {}",
            state.eval_file, err
        );
        uci_flush();
        AzNnue::random(128, state.seed)
    }));
}

fn handle_setoption(line: &str, state: &mut UciState) {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    let Some(name_index) = tokens.iter().position(|token| *token == "name") else {
        return;
    };
    let value_index = tokens.iter().position(|token| *token == "value");
    let name_end = value_index.unwrap_or(tokens.len());
    let name = tokens[name_index + 1..name_end]
        .join(" ")
        .to_ascii_lowercase();
    let value = value_index
        .map(|index| tokens[index + 1..].join(" "))
        .unwrap_or_default();

    match name.as_str() {
        "evalfile" => {
            state.eval_file = value;
            state.model = None;
        }
        "simulations" => {
            state.simulations = value.parse::<usize>().unwrap_or(state.simulations).max(1);
        }
        "topk" => {
            state.top_k = value.parse::<usize>().unwrap_or(state.top_k).max(1);
        }
        "threads" => {
            let _ = value;
            state.threads = 1;
        }
        "gumbelscale" => {
            state.gumbel_scale = value.parse::<f32>().unwrap_or(state.gumbel_scale).max(0.0);
        }
        "policydebug" => {
            state.policy_debug = matches!(value.to_ascii_lowercase().as_str(), "true" | "1" | "on");
        }
        "policydebuglimit" => {
            state.policy_debug_limit = value
                .parse::<usize>()
                .unwrap_or(state.policy_debug_limit)
                .clamp(1, 256);
        }
        _ => {}
    }
}

fn handle_position(line: &str, state: &mut UciState, logger: &mut UciLogger) {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    if tokens.get(1) == Some(&"startpos") {
        state.position = Position::startpos();
        state.history.clear();
        state.rule_history = state.position.initial_rule_history();
        if let Some(moves_index) = tokens.iter().position(|token| *token == "moves") {
            let move_list = &tokens[moves_index + 1..];
            ulog!(logger, "[position] startpos moves={}", move_list.join(" "));
            apply_uci_moves(
                &mut state.position,
                &mut state.history,
                &mut state.rule_history,
                move_list,
                logger,
            );
        } else {
            ulog!(logger, "[position] startpos (no moves)");
        }
        ulog!(
            logger,
            "[position] result fen={} halfmove_clock={}",
            state.position.to_fen(),
            state.position.halfmove_clock()
        );
        return;
    }

    if tokens.get(1) == Some(&"fen") {
        let moves_index = tokens.iter().position(|token| *token == "moves");
        let fen_end = moves_index.unwrap_or(tokens.len());
        let fen = tokens[2..fen_end].join(" ");
        if let Ok(position) = Position::from_fen(&fen) {
            state.position = position;
            state.history.clear();
            state.rule_history = state.position.initial_rule_history();
            if let Some(moves_index) = moves_index {
                let move_list = &tokens[moves_index + 1..];
                ulog!(logger, "[position] fen={fen} moves={}", move_list.join(" "));
                apply_uci_moves(
                    &mut state.position,
                    &mut state.history,
                    &mut state.rule_history,
                    move_list,
                    logger,
                );
            } else {
                ulog!(logger, "[position] fen={fen} (no moves)");
            }
            ulog!(
                logger,
                "[position] result fen={} halfmove_clock={}",
                state.position.to_fen(),
                state.position.halfmove_clock()
            );
        } else {
            eprintln!("info string invalid fen: {fen}");
            ulog!(logger, "[position] ERROR invalid fen={fen}");
        }
    }
}

fn apply_uci_moves(
    position: &mut Position,
    history: &mut Vec<HistoryMove>,
    rule_history: &mut Vec<RuleHistoryEntry>,
    moves: &[&str],
    logger: &mut UciLogger,
) {
    for (i, text) in moves.iter().enumerate() {
        let Some(mv) = position.parse_uci_move(text) else {
            eprintln!("info string illegal move ignored: {text}");
            ulog!(
                logger,
                "[apply_move] #{i} {text} ILLEGAL — fen={} halfmove_clock={}",
                position.to_fen(),
                position.halfmove_clock()
            );
            break;
        };
        if let Some(piece) = position.piece_at(mv.from as usize) {
            history.push(HistoryMove { piece, mv });
            let overflow = history.len().saturating_sub(HISTORY_PLIES);
            if overflow > 0 {
                history.drain(0..overflow);
            }
        }
        rule_history.push(position.rule_history_entry_after_move(mv));
        position.make_move(mv);
        ulog!(logger, "[apply_move] #{i} {text} ok → fen={}", position.to_fen());
    }
}

fn handle_go(_line: &str, state: &mut UciState, logger: &mut UciLogger) {
    ensure_uci_model(state);
    let simulations = state.simulations.max(1);
    let model = state.model.as_ref().expect("model was loaded");

    let fen = state.position.to_fen();
    let raw_legal = state.position.legal_moves();
    let legal = state.position.legal_moves_with_rules(&state.rule_history);

    ulog!(
        logger,
        "[go] fen={fen} halfmove_clock={} rule_history_len={} raw_legal={} filtered_legal={}",
        state.position.halfmove_clock(),
        state.rule_history.len(),
        raw_legal.len(),
        legal.len()
    );

    if legal.is_empty() {
        // 写入完整诊断以便排查"有走法却判无合法走法"的问题
        ulog!(logger, "[no_legal_moves] fen={fen}");
        ulog!(
            logger,
            "[no_legal_moves] raw_legal_moves({})={}",
            raw_legal.len(),
            raw_legal
                .iter()
                .map(|mv| mv.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        );
        ulog!(
            logger,
            "[no_legal_moves] rule_history ({} entries):",
            state.rule_history.len()
        );
        for (i, entry) in state.rule_history.iter().enumerate() {
            ulog!(
                logger,
                "[no_legal_moves]   [{i:>3}] hash={:#018x} side={:?} mover={:?} check={} chase_mask={:#034x}",
                entry.hash,
                entry.side_to_move,
                entry.mover,
                entry.gives_check,
                entry.chased_mask
            );
        }
        println!("info depth 1 nodes 0 time 0 score cp -32000 pv 0000");
        println!("bestmove 0000");
        uci_flush();
        return;
    }
    let started = std::time::Instant::now();
    let result = gumbel_search_with_history_and_rules(
        &state.position,
        &state.history,
        Some(state.rule_history.clone()),
        Some(legal),
        model,
        AzSearchLimits {
            simulations,
            top_k: state.top_k,
            seed: state.seed,
            gumbel_scale: state.gumbel_scale,
            workers: 1,
        },
    );
    state.seed = state.seed.wrapping_add(1);
    if state.policy_debug {
        print_policy_debug(&result, state.policy_debug_limit);
    }
    println!(
        "info depth 1 nodes {} time {} score cp {} pv {}",
        result.simulations,
        started.elapsed().as_millis(),
        result.value_cp,
        result
            .best_move
            .map(|mv| mv.to_string())
            .unwrap_or_else(|| "0000".into())
    );
    println!(
        "bestmove {}",
        result
            .best_move
            .map(|mv| mv.to_string())
            .unwrap_or_else(|| "0000".into())
    );
    uci_flush();
}

fn print_policy_debug(result: &chineseai::az::AzSearchResult, limit: usize) {
    let visited_actions = result
        .candidates
        .iter()
        .filter(|candidate| candidate.visits > 0)
        .count();
    println!("info string policy_debug visited_actions {visited_actions}");
    println!("info string policy_debug columns move visits q raw_policy searched_policy");
    for candidate in result.candidates.iter().take(limit) {
        println!(
            "info string policy_debug {} {} {:.4} {:.6} {:.6}",
            candidate.mv, candidate.visits, candidate.q, candidate.prior, candidate.policy
        );
    }
    println!("info string policy_debug_by_visits columns move visits q raw_policy searched_policy");
    let mut by_visits = result.candidates.clone();
    by_visits.sort_by(|left, right| {
        right
            .visits
            .cmp(&left.visits)
            .then_with(|| right.policy.total_cmp(&left.policy))
            .then_with(|| right.q.total_cmp(&left.q))
    });
    for candidate in by_visits.iter().take(limit) {
        println!(
            "info string policy_debug_by_visits {} {} {:.4} {:.6} {:.6}",
            candidate.mv, candidate.visits, candidate.q, candidate.prior, candidate.policy
        );
    }
}

fn uci_flush() {
    let _ = io::stdout().flush();
}

fn parse_position(text: &str) -> Position {
    if text.trim().is_empty() || text == "startpos" {
        Position::startpos()
    } else {
        Position::from_fen(text).unwrap_or_else(|err| {
            panic!("invalid FEN `{text}`: {err}");
        })
    }
}
