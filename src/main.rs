#[cfg(all(target_os = "linux", not(target_env = "musl")))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

mod az_loop_config;

use az_loop_config::{AzLoopFileConfig, DEFAULT_AZ_LOOP_CONFIG, load_or_create_az_loop_config};

use chineseai::{
    az::{
        AzArenaConfig, AzArenaReport, AzExperiencePool, AzLoopConfig, AzLoopReport, AzNnue,
        AzSearchLimits, AzSelfplayData, AzTrainLossWeights, AzTrainingSample, DENSE_MOVE_SPACE,
        SplitMix64, alphazero_search, alphazero_search_with_rules, benchmark_training,
        generate_selfplay_data, global_training_step_sample_count, play_arena_games_from_positions,
        train_samples_weighted_owned,
    },
    opening_book::ObkBook,
    pikafish_match::{VsPikafishConfig, run_vs_pikafish},
    uci::run_uci,
    xiangqi::{Move, Position, RuleOutcome},
};
use clap::{Args, Parser, Subcommand};
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    fs, io,
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        Arc, Condvar, Mutex, RwLock,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};
use tensorboard_rs::summary_writer::SummaryWriter;

const DEFAULT_VS_PIKAFISH_DEPTH: u32 = 10;
const DEFAULT_VS_PIKAFISH_GAMES: usize = 20;
const DEFAULT_VS_PIKAFISH_PARALLEL_GAMES: usize = 5;

#[derive(Parser, Debug)]
#[command(
    name = "chineseai",
    version,
    about = "ChineseAI AZ-NNUE search and training tools",
    long_about = "ChineseAI AZ-NNUE search and training tools.\n\nIf no command is given, the program starts in UCI mode."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<CliCommand>,
}

#[derive(Subcommand, Debug)]
enum CliCommand {
    /// Run the UCI engine loop.
    Uci,
    /// Create a random AZ-NNUE model.
    AzInit(AzInitArgs),
    /// Search one position and print policy/debug details.
    AzSearch(AzSearchArgs),
    /// Benchmark fixed-position search speed.
    AzBench(AzBenchArgs),
    /// Benchmark a synthetic training workload.
    AzTrainBench(AzTrainBenchArgs),
    /// Run self-play training from a TOML config.
    AzLoop(AzLoopArgs),
    /// Run ChineseAI against a Pikafish UCI engine.
    VsPikafish(VsPikafishArgs),
    /// Generate random positions and label them with Pikafish best moves.
    PikafishLabelRandom(PikafishLabelRandomArgs),
    /// Evaluate a model against Pikafish labels stored in SQLite.
    PikafishLabelEval(PikafishLabelEvalArgs),
}

#[derive(Args, Debug, Clone)]
struct AzInitArgs {
    /// Hidden size of the model.
    #[arg(default_value_t = 128)]
    hidden: usize,
    /// Output model path.
    #[arg(default_value = "model.safetensors")]
    output: String,
    /// Random seed.
    #[arg(default_value_t = 20260409)]
    seed: u64,
}

impl AzInitArgs {
    fn arch(&self) -> chineseai::az::AzNnueArch {
        chineseai::az::AzNnueArch::with_hidden_size(self.hidden.max(1))
    }
}

#[derive(Args, Debug, Clone)]
#[command(after_long_help = "\
Examples:
  chineseai az-search model.safetensors
  chineseai az-search model.safetensors 50000 1.5 --cpuct-at-root 3.0 startpos
  chineseai az-search model.safetensors 10000 1.5 --cpuct-at-root 3.0 startpos")]
struct AzSearchArgs {
    /// AZ-NNUE model path.
    model: String,
    /// Number of MCTS simulations.
    #[arg(default_value_t = 10_000)]
    simulations: usize,
    /// Non-root PUCT init.
    #[arg(default_value_t = 1.5)]
    cpuct: f32,
    /// Root PUCT init.
    #[arg(long, default_value_t = 3.0)]
    cpuct_at_root: f32,
    /// Non-root first-play urgency reduction.
    #[arg(long, default_value_t = 0.23)]
    fpu_value: f32,
    /// Root first-play urgency value.
    #[arg(long, default_value_t = 1.0)]
    fpu_value_at_root: f32,
    /// Dynamic PUCT base.
    #[arg(long, default_value_t = 19652.0)]
    cpuct_base: f32,
    /// Dynamic PUCT growth factor.
    #[arg(long, default_value_t = 2.0)]
    cpuct_factor: f32,
    /// Root dynamic PUCT base.
    #[arg(long, default_value_t = 19652.0)]
    cpuct_base_at_root: f32,
    /// Root dynamic PUCT growth factor.
    #[arg(long, default_value_t = 2.0)]
    cpuct_factor_at_root: f32,
    /// Maximum search depth in plies below root; 0 keeps the MCTX default (simulations).
    #[arg(long, default_value_t = 0)]
    max_depth: usize,
    /// Draw value in Q = W - L + draw_score * D.
    #[arg(long, default_value_t = 0.0)]
    draw_score: f32,
    /// Enable moves-left utility.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    moves_left_utility: bool,
    /// Independently re-search this many top-visited root moves after making each move.
    #[arg(long, default_value_t = 0)]
    verify_top: usize,
    /// Independently re-search specific root moves (repeat the option for multiple moves).
    #[arg(long = "verify-move")]
    verify_moves: Vec<String>,
    /// Simulations for every independent child verification; 0 uses the root simulation count.
    #[arg(long, default_value_t = 0)]
    verify_sims: usize,
    /// Apply legal UCI moves before searching; repeat for a move sequence.
    #[arg(long = "move")]
    moves: Vec<String>,
    /// FEN string, or startpos if omitted.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    fen: Vec<String>,
}

#[derive(Args, Debug)]
#[command(after_long_help = "\
Examples:
  chineseai az-bench model.safetensors 512 100 1.5 startpos
  chineseai az-bench model.safetensors 512 100 1.5 startpos")]
struct AzBenchArgs {
    /// AZ-NNUE model path.
    model: String,
    /// Simulations per search.
    #[arg(default_value_t = 512)]
    simulations: usize,
    /// Number of repeated searches.
    #[arg(default_value_t = 100)]
    repeat: usize,
    /// PUCT constant for AlphaZero search.
    #[arg(default_value_t = 1.5)]
    cpuct: f32,
    /// FEN string, or startpos if omitted.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    fen: Vec<String>,
}

#[derive(Args, Debug)]
struct AzTrainBenchArgs {
    /// AZ-NNUE model path.
    model: String,
    /// Generated sample count.
    #[arg(default_value_t = 8192)]
    samples: usize,
    /// Passes over generated samples.
    #[arg(default_value_t = 2)]
    epochs: usize,
    /// Micro-batch size per visible GPU.
    #[arg(default_value_t = 1024)]
    batch_size_per_gpu: usize,
    /// Learning rate.
    #[arg(default_value_t = 0.0003)]
    lr: f32,
    /// Random seed.
    #[arg(default_value_t = 20260411)]
    seed: u64,
}

#[derive(Args, Debug)]
struct AzLoopArgs {
    /// Training config path.
    #[arg(default_value = DEFAULT_AZ_LOOP_CONFIG)]
    config: String,
    /// Stop after completing this absolute update number and save the model/progress.
    #[arg(long)]
    target_update: Option<usize>,
}

#[derive(Args, Debug)]
#[command(after_long_help = "\
Examples:
  chineseai vs-pikafish ./tools/pikafish model.safetensors
  chineseai vs-pikafish ./tools/pikafish checkpoints/update-0620-model.safetensors --simulations 192
  chineseai vs-pikafish ./tools/pikafish model.safetensors --pikafish-depth 10 --games 40 --parallel-games 5
  chineseai vs-pikafish ./tools/pikafish model.safetensors --opening-book opening.obk --opening-plies-min 6 --opening-plies-max 10")]
struct VsPikafishArgs {
    /// Pikafish UCI executable path.
    pikafish_exe: String,
    /// ChineseAI AZ-NNUE model path.
    model: String,
    /// ChineseAI MCTS simulations per move.
    #[arg(short = 's', long)]
    simulations: Option<usize>,
    /// ChineseAI PUCT constant.
    #[arg(long, default_value_t = 1.5)]
    cpuct: f32,
    /// ChineseAI root PUCT constant.
    #[arg(long, default_value_t = 3.0)]
    cpuct_at_root: f32,
    /// Draw after this many plies.
    #[arg(long, default_value_t = 300)]
    max_plies: usize,
    /// Random seed.
    #[arg(long, default_value_t = 20260411)]
    seed: u64,
    /// Pikafish search depth.
    #[arg(long, default_value_t = DEFAULT_VS_PIKAFISH_DEPTH)]
    pikafish_depth: u32,
    /// Total games.
    #[arg(long, default_value_t = DEFAULT_VS_PIKAFISH_GAMES)]
    games: usize,
    /// Simultaneous games/processes.
    #[arg(long, default_value_t = DEFAULT_VS_PIKAFISH_PARALLEL_GAMES)]
    parallel_games: usize,
    /// OBK opening book used to generate random start positions. Empty uses startpos.
    #[arg(long, default_value = "opening.obk")]
    opening_book: String,
    /// Number of random opening positions to generate from the OBK book.
    #[arg(long, default_value_t = 300)]
    opening_positions: usize,
    /// Minimum book plies before handing the position to both engines.
    #[arg(long, default_value_t = 6)]
    opening_plies_min: usize,
    /// Maximum book plies before handing the position to both engines.
    #[arg(long, default_value_t = 10)]
    opening_plies_max: usize,
}

#[derive(Args, Debug)]
#[command(after_long_help = "\
Examples:
  chineseai pikafish-label-random ./tools/pikafish-avx2.exe --count 5000 --depth 8
  chineseai pikafish-label-random ./tools/pikafish-avx2.exe --fens eval/random.fens --output eval/pikafish-labels.csv --sqlite eval/pikafish-labels.sqlite")]
struct PikafishLabelRandomArgs {
    /// Pikafish UCI executable path.
    pikafish_exe: String,
    /// Output FEN list. Existing file is reused unless --regenerate is set.
    #[arg(long, default_value = "eval/random.fens")]
    fens: String,
    /// Output CSV labels.
    #[arg(long, default_value = "eval/pikafish-labels.csv")]
    output: String,
    /// Output SQLite labels.
    #[arg(long, default_value = "eval/pikafish-labels.sqlite")]
    sqlite: String,
    /// Number of unique random positions.
    #[arg(long, default_value_t = 5000)]
    count: usize,
    /// Random seed for FEN generation.
    #[arg(long, default_value_t = 20260628)]
    seed: u64,
    /// Minimum random plies from startpos.
    #[arg(long, default_value_t = 12)]
    min_plies: usize,
    /// Maximum random plies from startpos.
    #[arg(long, default_value_t = 80)]
    max_plies: usize,
    /// Pikafish search depth.
    #[arg(long, default_value_t = 8)]
    depth: u32,
    /// Regenerate the FEN file even when it already exists.
    #[arg(long)]
    regenerate: bool,
}

#[derive(Args, Debug)]
#[command(after_long_help = "\
Examples:
  chineseai pikafish-label-eval model.safetensors eval/pikafish-labels.sqlite --simulations 64")]
struct PikafishLabelEvalArgs {
    /// ChineseAI AZ-NNUE model path.
    model: String,
    /// SQLite labels produced by pikafish-label-random.
    sqlite: String,
    /// ChineseAI MCTS simulations per position.
    #[arg(short = 's', long, default_value_t = 64)]
    simulations: usize,
    /// ChineseAI PUCT constant.
    #[arg(long, default_value_t = 1.5)]
    cpuct: f32,
    /// Maximum search depth in plies below root; 0 keeps the MCTS default.
    #[arg(long, default_value_t = 0)]
    max_depth: usize,
    /// Random seed.
    #[arg(long, default_value_t = 20260628)]
    seed: u64,
    /// Limit number of positions; 0 means all.
    #[arg(long, default_value_t = 0)]
    limit: usize,
}

fn best_model_path(model_path: &str) -> PathBuf {
    Path::new(model_path)
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .join("best.safetensors")
}

const SWA_MAX_MODELS: usize = 10;

fn swa_model_path(model_path: &str) -> PathBuf {
    Path::new(model_path)
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .join("swa.safetensors")
}

fn update_swa_model(swa: &mut Option<AzNnue>, count: &mut usize, current: &AzNnue) {
    if let Some(average) = swa {
        *count = average.update_capped_swa(current, *count, SWA_MAX_MODELS);
    } else {
        *swa = Some(current.clone());
        *count = 1;
    }
}

fn az_loop_progress_path(config_path: &str) -> PathBuf {
    PathBuf::from(format!("{config_path}.progress"))
}

fn az_loop_replay_snapshot_path(config_path: &str) -> PathBuf {
    PathBuf::from(format!("{config_path}.replay.lz4"))
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
struct AzLoopProgressState {
    next_update: usize,
    best_elo: f32,
    swa_count: usize,
    #[serde(skip_serializing)]
    pikafish_depth: u32,
    #[serde(skip_serializing)]
    pikafish_best_wins: usize,
}

impl Default for AzLoopProgressState {
    fn default() -> Self {
        Self {
            next_update: 1,
            best_elo: 1500.0,
            swa_count: 0,
            pikafish_depth: 1,
            pikafish_best_wins: 0,
        }
    }
}

impl AzLoopProgressState {
    fn normalize(mut self) -> Self {
        self.next_update = self.next_update.max(1);
        if !self.best_elo.is_finite() {
            self.best_elo = 1500.0;
        }
        self.swa_count = self.swa_count.min(SWA_MAX_MODELS);
        self
    }
}

fn load_az_loop_progress(config_path: &str) -> AzLoopProgressState {
    let path = az_loop_progress_path(config_path);
    let Ok(text) = fs::read_to_string(&path) else {
        return AzLoopProgressState::default();
    };
    let state = toml::from_str::<AzLoopProgressState>(&text)
        .unwrap_or_else(|err| panic!("failed to parse `{}`: {err}", path.display()))
        .normalize();
    fs::remove_file(&path).unwrap_or_else(|err| {
        panic!(
            "loaded progress but failed to remove consumed `{}`: {err}",
            path.display()
        )
    });
    state
}

fn save_az_loop_progress(config_path: &str, state: &AzLoopProgressState) {
    let path = az_loop_progress_path(config_path);
    fs::write(
        &path,
        toml::to_string_pretty(&state.clone().normalize()).unwrap(),
    )
    .unwrap_or_else(|err| panic!("failed to write `{}`: {err}", path.display()));
}

fn save_az_loop_progress_pair(
    config_path: &str,
    next_update: usize,
    best_elo: f32,
    swa_count: usize,
) {
    save_az_loop_progress(
        config_path,
        &AzLoopProgressState {
            next_update,
            best_elo,
            swa_count,
            ..Default::default()
        },
    );
}

fn save_model(model: &AzNnue, path: &Path) {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).unwrap_or_else(|err| {
            panic!(
                "failed to create model directory `{}`: {err}",
                parent.display()
            );
        });
    }
    model
        .save(path)
        .unwrap_or_else(|err| panic!("failed to save model `{}`: {err}", path.display()));
}

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

    let encoded = format!(
        concat!(
            "sim{}_sspu{}_bs{}_lr{}_h{}_mxp{}_wk{}_",
            "rrf{}_rrw{}_lrm{}_lds{}_ldi{}_ldf{}_cp{}_cpr{}_fv{}_fvr{}_pst{}_tb{}_teg{}_tdd{}_tde{}_tvc{}_tvo{}_op{}_rs{}_rp{}_rc{}_",
            "tspu{}_tepu{}_mp{}_cpi{}_ai{}_as{}_acp{}_rda{}_ref{}_sd{}"
        ),
        config.simulations,
        config.selfplay_samples_per_update,
        config.batch_size,
        f32_slug(config.lr),
        config.hidden_size,
        config.max_plies,
        config.workers,
        f32_slug(config.replay_recent_sample_fraction),
        config.replay_recent_games,
        f32_slug(config.lr_min),
        config.lr_decay_start_update,
        config.lr_decay_interval,
        f32_slug(config.lr_decay_factor),
        f32_slug(config.cpuct),
        f32_slug(config.cpuct_at_root),
        f32_slug(config.fpu_value),
        f32_slug(config.fpu_value_at_root),
        f32_slug(config.policy_softmax_temp),
        f32_slug(config.temperature_start),
        f32_slug(config.temperature_endgame),
        config.temperature_decay_delay_plies,
        config.temperature_decay_plies,
        f32_slug(config.temperature_value_cutoff),
        f32_slug(config.temperature_visit_offset),
        if config.opening_fens_path.trim().is_empty() {
            "none".to_string()
        } else {
            format!("{:016x}", fnv1a64(config.opening_fens_path.as_bytes()))
        },
        f32_slug(config.resign_percentage),
        f32_slug(config.resign_playthrough),
        config.replay_capacity,
        config.train_samples_per_update,
        config.train_epochs_per_update,
        f32_slug(config.mirror_probability),
        config.checkpoint_interval,
        config.arena_interval,
        config.arena_simulations,
        f32_slug(config.arena_cpuct),
        f32_slug(config.root_dirichlet_alpha),
        f32_slug(config.root_exploration_fraction),
        config.seed,
    );
    if encoded.len() <= 180 {
        return encoded;
    }

    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in encoded.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    format!(
        "sim{}_bs{}_lr{}_h{}_sd{}_cfg{:016x}",
        config.simulations,
        config.batch_size,
        f32_slug(config.lr),
        config.hidden_size,
        config.seed,
        hash
    )
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

fn learning_rate_for_update(config: &AzLoopFileConfig, update: usize) -> f32 {
    if config.lr <= 0.0 {
        return 0.0;
    }
    let factor = config.lr_decay_factor.clamp(0.0, 1.0);
    if factor <= 0.0 {
        return config.lr_min.min(config.lr).max(0.0);
    }
    if factor >= 1.0 || update < config.lr_decay_start_update {
        return config.lr;
    }

    let interval = config.lr_decay_interval.max(1);
    let steps = 1 + (update - config.lr_decay_start_update) / interval;
    let decayed = config.lr * factor.powi(steps as i32);
    decayed.max(config.lr_min.min(config.lr).max(0.0))
}

fn tensorboard_effective_logdir(config: &AzLoopFileConfig) -> PathBuf {
    Path::new(&config.tensorboard_logdir).join(tensorboard_encoded_subdir(config))
}

fn checkpoint_path(model_path: &str, checkpoint_dir: &str, update: usize) -> PathBuf {
    let base = Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.safetensors");
    Path::new(checkpoint_dir).join(format!("update-{update:06}-{base}"))
}

fn best_checkpoint_path(model_path: &str, checkpoint_dir: &str, update: usize) -> PathBuf {
    let base = Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.safetensors");
    Path::new(checkpoint_dir).join(format!("best-update-{update:06}-{base}"))
}

fn save_checkpoint_model(
    model: &AzNnue,
    model_path: &str,
    checkpoint_dir: &str,
    update: usize,
) -> PathBuf {
    fs::create_dir_all(checkpoint_dir).unwrap_or_else(|err| {
        panic!("failed to create checkpoint dir `{checkpoint_dir}`: {err}");
    });
    let path = checkpoint_path(model_path, checkpoint_dir, update);
    save_model(model, &path);
    path
}

fn save_best_checkpoint_model(
    model: &AzNnue,
    model_path: &str,
    checkpoint_dir: &str,
    update: usize,
) -> PathBuf {
    fs::create_dir_all(checkpoint_dir).unwrap_or_else(|err| {
        panic!("failed to create checkpoint dir `{checkpoint_dir}`: {err}");
    });
    let path = best_checkpoint_path(model_path, checkpoint_dir, update);
    save_model(model, &path);
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
        .unwrap_or("model.safetensors")
        .to_string();
    let prefix = "update-";
    let suffix = format!("-{base}");
    let mut entries = fs::read_dir(checkpoint_dir)?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let path = entry.path();
            let name = path.file_name()?.to_str()?;
            if !name.starts_with(prefix) || !name.ends_with(&suffix) {
                return None;
            }
            let update_text = name
                .strip_prefix(prefix)?
                .strip_suffix(&suffix)?
                .split('-')
                .next()?;
            let update = update_text.parse::<usize>().ok()?;
            Some((update, name.to_string(), path))
        })
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    let to_remove = entries.len().saturating_sub(max_checkpoints);
    for (_, _, path) in entries.into_iter().take(to_remove) {
        fs::remove_file(path)?;
    }
    Ok(())
}

struct SelfplayBatch {
    data: AzSelfplayData,
    selfplay_seconds: f32,
}

struct TrainerEvent {
    report: AzLoopReport,
    candidate_model: AzNnue,
}

struct SharedSelfplayModel {
    version: u64,
    models_by_numa_node: Vec<Arc<AzNnue>>,
}

fn build_numa_model_replicas(model: &AzNnue, numa_nodes: &[(usize, usize)]) -> Vec<Arc<AzNnue>> {
    thread::scope(|scope| {
        numa_nodes
            .iter()
            .map(|&(_node, cpu)| {
                scope.spawn(move || {
                    let _ = chineseai::cpu_topology::pin_current_thread(cpu);
                    Arc::new(model.clone())
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().expect("NUMA model replica thread panicked"))
            .collect()
    })
}

#[derive(Default)]
struct SelfplayPauseState {
    arena_paused: bool,
    backlog_paused: bool,
}

impl SelfplayPauseState {
    fn is_paused(&self) -> bool {
        self.arena_paused || self.backlog_paused
    }
}

#[derive(Default)]
struct PendingTrainingData {
    selfplay_seconds: f32,
    started: Option<Instant>,
    selfplay: AzSelfplayData,
}

#[derive(Clone, Copy, Debug, Default)]
struct TrainBatchSourceStats {
    fast_sample_rate: f32,
    policy_weight_mean: f32,
    value_weight_mean: f32,
    recent_quota_rate: f32,
    actual_recent_sample_rate: f32,
    policy_target_entropy: f32,
    policy_target_top1: f32,
    policy_target_top2: f32,
}

impl PendingTrainingData {
    fn push(&mut self, batch: SelfplayBatch) {
        if self.started.is_none() {
            self.started = Some(Instant::now());
        }
        self.selfplay_seconds += batch.selfplay_seconds;
        self.selfplay.add_assign(&batch.data);
    }
}

fn build_az_loop_config(
    config: &AzLoopFileConfig,
    seed: u64,
    workers: usize,
    generation_update: u32,
    opening_positions: &[Position],
) -> AzLoopConfig {
    AzLoopConfig {
        games: 1,
        max_plies: config.max_plies,
        simulations: config.simulations,
        seed,
        workers,
        generation_update,
        temperature_start: config.temperature_start,
        temperature_endgame: config.temperature_endgame,
        temperature_decay_delay_plies: config.temperature_decay_delay_plies,
        temperature_decay_plies: config.temperature_decay_plies,
        temperature_value_cutoff: config.temperature_value_cutoff,
        temperature_visit_offset: config.temperature_visit_offset,
        cpuct: config.cpuct,
        cpuct_at_root: config.cpuct_at_root,
        cpuct_base: config.cpuct_base,
        cpuct_factor: config.cpuct_factor,
        cpuct_base_at_root: config.cpuct_base_at_root,
        cpuct_factor_at_root: config.cpuct_factor_at_root,
        root_dirichlet_alpha: config.root_dirichlet_alpha,
        root_exploration_fraction: config.root_exploration_fraction,
        fpu_value: config.fpu_value,
        fpu_value_at_root: config.fpu_value_at_root,
        draw_score: config.draw_score,
        moves_left_max_effect: config.moves_left_max_effect,
        moves_left_slope: config.moves_left_slope,
        moves_left_threshold: config.moves_left_threshold,
        moves_left_constant_factor: config.moves_left_constant_factor,
        moves_left_scaled_factor: config.moves_left_scaled_factor,
        moves_left_quadratic_factor: config.moves_left_quadratic_factor,
        policy_softmax_temp: config.policy_softmax_temp,
        opening_positions: opening_positions.to_vec(),
        resign_percentage: config.resign_percentage,
        resign_playthrough: config.resign_playthrough,
        mirror_probability: config.mirror_probability,
    }
}

fn load_opening_positions(path: &str) -> Vec<Position> {
    let path = path.trim();
    if path.is_empty() {
        return Vec::new();
    }
    let text = fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("failed to read opening_fens_path `{path}`: {err}"));
    text.lines()
        .enumerate()
        .filter_map(|(index, line)| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                return None;
            }
            Some(Position::from_fen(line).unwrap_or_else(|err| {
                panic!("invalid opening FEN at `{path}` line {}: {err}", index + 1)
            }))
        })
        .collect()
}

fn build_async_training_report(
    pending: PendingTrainingData,
    selfplay_games: usize,
    total_games_generated: usize,
    total_samples_generated: usize,
    stats: chineseai::az::AzTrainStats,
    learning_rate: f32,
    train_data_len: usize,
    train_seconds: f32,
    pool_samples: usize,
    pool_capacity: usize,
    replay_window: chineseai::az::AzReplayWindowStats,
    train_source: TrainBatchSourceStats,
) -> AzLoopReport {
    let selfplay_samples = pending.selfplay.samples.len();
    let total_seconds = pending
        .started
        .map(|started| started.elapsed().as_secs_f32())
        .unwrap_or(train_seconds);
    let train_stat_samples = stats.samples.max(1) as f32;
    let root_visit_entropy =
        pending.selfplay.entropy_all_sum / pending.selfplay.entropy_all_count.max(1) as f32;
    let shape_count = pending.selfplay.shape_count.max(1) as f32;
    let opening_shape_count = pending.selfplay.opening_shape_count.max(1) as f32;
    let sampled_moves = pending.selfplay.sampled_moves.max(1) as f32;
    let search_count = pending.selfplay.search_simulations.searches.max(1) as f32;
    let value_pred_mean = stats.value_pred_sum / train_stat_samples;
    let value_target_mean = stats.value_target_sum / train_stat_samples;
    let value_pred_var =
        (stats.value_pred_sq_sum / train_stat_samples - value_pred_mean * value_pred_mean).max(0.0);
    let value_target_var = (stats.value_target_sq_sum / train_stat_samples
        - value_target_mean * value_target_mean)
        .max(0.0);
    let value_cov =
        stats.value_pred_target_sum / train_stat_samples - value_pred_mean * value_target_mean;
    let value_corr =
        value_cov / (value_pred_var.max(1.0e-12).sqrt() * value_target_var.max(1.0e-12).sqrt());
    let value_calibration = value_cov / value_pred_var.max(1.0e-12);
    let phase_value = std::array::from_fn(|phase| {
        let phase_stats = stats.phase_value[phase];
        let count = phase_stats.samples.max(1) as f32;
        let pred_mean = phase_stats.pred_sum / count;
        let target_mean = phase_stats.target_sum / count;
        let pred_var = (phase_stats.pred_sq_sum / count - pred_mean * pred_mean).max(0.0);
        let target_var = (phase_stats.target_sq_sum / count - target_mean * target_mean).max(0.0);
        let covariance = phase_stats.pred_target_sum / count - pred_mean * target_mean;
        chineseai::az::AzPhaseValueReport {
            samples: phase_stats.samples,
            rmse: (phase_stats.error_sq_sum / count).max(0.0).sqrt(),
            corr: (covariance / (pred_var.max(1.0e-12).sqrt() * target_var.max(1.0e-12).sqrt()))
                .clamp(-1.0, 1.0),
            calibration: covariance / pred_var.max(1.0e-12),
        }
    });
    AzLoopReport {
        games: selfplay_games,
        samples: selfplay_samples,
        total_games_generated,
        total_samples_generated,
        avg_search_simulations: pending.selfplay.search_simulations.simulations_sum as f32
            / search_count,
        red_wins: pending.selfplay.red_wins,
        black_wins: pending.selfplay.black_wins,
        draws: pending.selfplay.draws,
        avg_plies: if selfplay_games == 0 {
            0.0
        } else {
            pending.selfplay.plies_total as f32 / selfplay_games as f32
        },
        loss: stats.loss,
        learning_rate,
        value_loss: stats.value_loss,
        value_mse: stats.value_error_sq_sum / train_stat_samples,
        value_pred_mean,
        value_target_mean,
        value_pred_rms: (stats.value_pred_sq_sum / train_stat_samples)
            .max(0.0)
            .sqrt(),
        value_target_rms: (stats.value_target_sq_sum / train_stat_samples)
            .max(0.0)
            .sqrt(),
        value_corr: value_corr.clamp(-1.0, 1.0),
        value_calibration,
        phase_value,
        policy_ce: stats.policy_ce,
        policy_target_entropy: train_source.policy_target_entropy,
        moves_left_loss: stats.moves_left_loss,
        policy_kl: stats.policy_ce - train_source.policy_target_entropy,
        root_visit_entropy,
        entropy_opening: pending.selfplay.entropy_opening_sum
            / pending.selfplay.entropy_opening_count.max(1) as f32,
        entropy_mid: pending.selfplay.entropy_mid_sum
            / pending.selfplay.entropy_mid_count.max(1) as f32,
        raw_prior_top1: pending.selfplay.raw_prior_top1_sum / shape_count,
        raw_prior_top2: pending.selfplay.raw_prior_top2_sum / shape_count,
        policy_top1: pending.selfplay.policy_top1_sum / shape_count,
        policy_top2: pending.selfplay.policy_top2_sum / shape_count,
        root_q_gap: pending.selfplay.q_gap_sum / shape_count,
        root_q_top1_abs: pending.selfplay.q_top1_abs_sum / shape_count,
        visited_actions: pending.selfplay.visited_actions_sum as f32 / shape_count,
        opening_raw_prior_top1: pending.selfplay.opening_raw_prior_top1_sum / opening_shape_count,
        opening_raw_prior_top2: pending.selfplay.opening_raw_prior_top2_sum / opening_shape_count,
        opening_policy_top1: pending.selfplay.opening_policy_top1_sum / opening_shape_count,
        opening_policy_top2: pending.selfplay.opening_policy_top2_sum / opening_shape_count,
        opening_q_gap: pending.selfplay.opening_q_gap_sum / opening_shape_count,
        opening_q_top1_abs: pending.selfplay.opening_q_top1_abs_sum / opening_shape_count,
        opening_visited_actions: pending.selfplay.opening_visited_actions_sum as f32
            / opening_shape_count,
        sampled_best_rate: pending.selfplay.sampled_best_moves as f32 / sampled_moves,
        avg_best_played_q_gap: pending.selfplay.best_played_q_gap_sum / sampled_moves,
        avg_played_top_visit_ratio: pending.selfplay.played_top_visit_ratio_sum / sampled_moves,
        avg_best_q: pending.selfplay.best_q_sum / sampled_moves,
        avg_played_q: pending.selfplay.played_q_sum / sampled_moves,
        selfplay_seconds: pending.selfplay_seconds,
        train_seconds,
        total_seconds,
        games_per_second: selfplay_games as f32 / total_seconds.max(1e-6),
        samples_per_second: selfplay_samples as f32 / total_seconds.max(1e-6),
        train_samples_per_second: train_data_len as f32 / train_seconds.max(1e-6),
        train_samples: train_data_len,
        pool_samples,
        pool_capacity,
        replay_chunks: replay_window.chunks,
        replay_oldest_update: replay_window.oldest_generation_update,
        replay_newest_update: replay_window.newest_generation_update,
        replay_avg_update: replay_window.avg_generation_update,
        replay_window_games: replay_window.window_games,
        replay_recent_window_fraction: replay_window.recent_window_sample_fraction,
        train_fast_sample_rate: train_source.fast_sample_rate,
        train_policy_weight_mean: train_source.policy_weight_mean,
        train_value_weight_mean: train_source.value_weight_mean,
        train_recent_quota_rate: train_source.recent_quota_rate,
        train_actual_recent_sample_rate: train_source.actual_recent_sample_rate,
        train_policy_target_top1: train_source.policy_target_top1,
        train_policy_target_top2: train_source.policy_target_top2,
        terminal_no_legal_moves: pending.selfplay.terminal.no_legal_moves,
        terminal_red_general_missing: pending.selfplay.terminal.red_general_missing,
        terminal_black_general_missing: pending.selfplay.terminal.black_general_missing,
        terminal_rule_draw: pending.selfplay.terminal.rule_draw,
        terminal_rule_draw_halfmove120: pending.selfplay.terminal.rule_draw_halfmove120,
        terminal_rule_draw_repetition: pending.selfplay.terminal.rule_draw_repetition,
        terminal_rule_draw_mutual_long_check: pending.selfplay.terminal.rule_draw_mutual_long_check,
        terminal_rule_draw_mutual_long_chase: pending.selfplay.terminal.rule_draw_mutual_long_chase,
        terminal_rule_win_red: pending.selfplay.terminal.rule_win_red,
        terminal_rule_win_black: pending.selfplay.terminal.rule_win_black,
        terminal_resign_red: pending.selfplay.terminal.resign_red,
        terminal_resign_black: pending.selfplay.terminal.resign_black,
        terminal_max_plies: pending.selfplay.terminal.max_plies,
    }
}

fn train_batch_source_stats(
    samples: &[AzTrainingSample],
    full_simulations: usize,
    recent_quota_samples: usize,
    newest_generation: u32,
    recent_window_games: u32,
) -> TrainBatchSourceStats {
    if samples.is_empty() {
        return TrainBatchSourceStats::default();
    }
    let full_simulations = full_simulations.max(1) as u32;
    let mut fast = 0usize;
    let mut policy_weight_sum = 0.0f32;
    let mut value_weight_sum = 0.0f32;
    let mut actual_recent = 0usize;
    let mut target_entropy_sum = 0.0f32;
    let mut target_top1_sum = 0.0f32;
    let mut target_top2_sum = 0.0f32;
    let recent_oldest =
        newest_generation.saturating_sub(recent_window_games.max(1).saturating_sub(1));
    for sample in samples {
        fast += usize::from(
            sample.search_simulations > 0 && sample.search_simulations < full_simulations,
        );
        policy_weight_sum += sample.policy_weight.max(0.0);
        value_weight_sum += sample.value_weight.max(0.0);
        actual_recent += usize::from(sample.meta.generation_update >= recent_oldest);
        let active_targets = sample
            .move_indices
            .iter()
            .zip(&sample.policy)
            .filter_map(|(&move_index, &target)| {
                (move_index < DENSE_MOVE_SPACE).then_some(target.max(0.0))
            })
            .collect::<Vec<_>>();
        let target_sum = active_targets.iter().copied().sum::<f32>();
        let uniform_target = if active_targets.is_empty() {
            0.0
        } else {
            1.0 / active_targets.len() as f32
        };
        let normalize_target = |target: f32| {
            if target_sum.is_finite() && target_sum > 1.0e-12 {
                target / target_sum
            } else {
                uniform_target
            }
        };
        let mut top = [0.0f32; 2];
        for &target in &active_targets {
            let p = normalize_target(target);
            if p > 0.0 {
                target_entropy_sum -= p * p.ln();
            }
            if p > top[0] {
                top[1] = top[0];
                top[0] = p;
            } else if p > top[1] {
                top[1] = p;
            }
        }
        target_top1_sum += top[0];
        target_top2_sum += top[0] + top[1];
    }
    let denom = samples.len() as f32;
    TrainBatchSourceStats {
        fast_sample_rate: fast as f32 / denom,
        policy_weight_mean: policy_weight_sum / denom,
        value_weight_mean: value_weight_sum / denom,
        recent_quota_rate: recent_quota_samples.min(samples.len()) as f32 / denom,
        actual_recent_sample_rate: actual_recent as f32 / denom,
        policy_target_entropy: target_entropy_sum / denom,
        policy_target_top1: target_top1_sum / denom,
        policy_target_top2: target_top2_sum / denom,
    }
}

struct ArenaThreadConfig {
    candidate: Arc<AzNnue>,
    baseline: Arc<AzNnue>,
    eval_positions: Arc<Vec<Position>>,
    simulations: usize,
    max_plies: usize,
    cpuct: f32,
    thread_count: usize,
    seed: u64,
}

fn run_arena_threads(config: ArenaThreadConfig) -> AzArenaReport {
    let games_per_side = if config.eval_positions.is_empty() {
        1
    } else {
        config.eval_positions.len().max(1)
    };
    let thread_count = config.thread_count.max(1).min(games_per_side);
    let mut handles = Vec::with_capacity(thread_count);
    let mut start_index = 0usize;
    for index in 0..thread_count {
        let red_games =
            games_per_side / thread_count + usize::from(index < games_per_side % thread_count);
        let black_games = red_games;
        if red_games == 0 && black_games == 0 {
            continue;
        }
        let candidate = Arc::clone(&config.candidate);
        let baseline = Arc::clone(&config.baseline);
        let eval_positions = Arc::clone(&config.eval_positions);
        let simulations = config.simulations;
        let max_plies = config.max_plies;
        let cpuct = config.cpuct;
        let seed = config.seed ^ index as u64;
        let thread_start_index = start_index;
        start_index += red_games;
        handles.push(thread::spawn(move || {
            play_arena_games_from_positions(
                candidate.as_ref(),
                baseline.as_ref(),
                eval_positions.as_slice(),
                AzArenaConfig {
                    simulations,
                    max_plies,
                    games_as_red: red_games,
                    games_as_black: black_games,
                    start_index: thread_start_index,
                    seed,
                    cpuct,
                },
            )
        }));
    }

    let mut merged = AzArenaReport::default();
    for handle in handles {
        merged.add_assign(
            &handle
                .join()
                .unwrap_or_else(|_| panic!("arena thread panicked")),
        );
    }
    merged
}

fn build_arena_start_positions(
    config: &AzLoopFileConfig,
    update: usize,
) -> (Vec<Position>, String) {
    if !config.arena_opening_book.trim().is_empty() {
        let book = ObkBook::load(&config.arena_opening_book).unwrap_or_else(|err| {
            panic!(
                "failed to load arena opening book `{}`: {err}",
                config.arena_opening_book
            )
        });
        let mut rng =
            SplitMix64::new(config.seed ^ (update as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
        let mut positions = Vec::with_capacity(config.arena_opening_positions);
        for _ in 0..config.arena_opening_positions {
            positions.push(book.random_prefix_position(
                config.arena_opening_plies_min,
                config.arena_opening_plies_max,
                &mut rng,
            ));
        }
        let mode = format!(
            "obk_openings(keys={},moves={},plies={}-{})",
            book.key_count(),
            book.move_count(),
            config.arena_opening_plies_min,
            config.arena_opening_plies_max
        );
        return (positions, mode);
    }

    (Vec::new(), "startpos_fallback".to_string())
}

fn fixed_az_search_limits(
    simulations: usize,
    seed: u64,
    cpuct: f32,
    max_depth: usize,
) -> AzSearchLimits {
    AzSearchLimits {
        simulations,
        seed,
        cpuct,
        cpuct_at_root: cpuct,
        cpuct_base: 19652.0,
        cpuct_factor: 2.0,
        cpuct_base_at_root: 19652.0,
        cpuct_factor_at_root: 2.0,
        max_depth,
        root_dirichlet_alpha: 0.0,
        root_exploration_fraction: 0.0,
        fpu_value: 0.23,
        fpu_value_at_root: 1.0,
        policy_softmax_temp: 1.0,
        draw_score: 0.0,
        moves_left_max_effect: 0.0,
        moves_left_slope: 0.0,
        moves_left_threshold: 0.6,
        moves_left_constant_factor: 0.0,
        moves_left_scaled_factor: 0.0,
        moves_left_quadratic_factor: 0.0,
        value_scale: 1.0,
    }
}

fn log_scalar(writer: &mut SummaryWriter, tag: &str, step: usize, value: f32) {
    writer.add_scalar(tag, value, step);
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        None => run_uci(),
        Some(CliCommand::Uci) => run_uci(),
        Some(CliCommand::AzInit(cmd)) => {
            let arch = cmd.arch();
            let output = cmd.output;
            let seed = cmd.seed;
            let model = AzNnue::random_with_arch(arch, seed);
            model.save(&output).unwrap_or_else(|err| {
                panic!("failed to write `{output}`: {err}");
            });
            println!("aznnue   : initialized (nnue binary, magic AZB1)");
            println!("arch     : hidden={}", arch.hidden_size,);
            println!("seed     : {seed}");
            println!("output   : {output}");
        }
        Some(CliCommand::AzSearch(cmd)) => {
            let model_path = cmd.model;
            let simulations = cmd.simulations.max(1);
            let cpuct = cmd.cpuct.max(0.0);
            let cpuct_at_root = cmd.cpuct_at_root.max(0.0);
            let fen = cmd.fen.join(" ");
            let mut position = parse_position(&fen);
            let mut rule_history = position.initial_rule_history();
            for text in &cmd.moves {
                let mv = position.parse_uci_move(text).unwrap_or_else(|| {
                    panic!("invalid or illegal --move `{text}` for this position")
                });
                rule_history.push(position.rule_history_entry_after_move(mv));
                position.make_move(mv);
            }
            let model = AzNnue::load(&model_path).unwrap_or_else(|err| {
                panic!("failed to load `{model_path}`: {err}");
            });
            let search_limits = AzSearchLimits {
                simulations,
                seed: 0,
                cpuct,
                cpuct_at_root,
                cpuct_base: cmd.cpuct_base.max(1.0),
                cpuct_factor: cmd.cpuct_factor.max(0.0),
                cpuct_base_at_root: cmd.cpuct_base_at_root.max(1.0),
                cpuct_factor_at_root: cmd.cpuct_factor_at_root.max(0.0),
                max_depth: cmd.max_depth,
                root_dirichlet_alpha: 0.0,
                root_exploration_fraction: 0.0,
                fpu_value: cmd.fpu_value.max(0.0),
                fpu_value_at_root: cmd.fpu_value_at_root.clamp(-1.0, 1.0),
                policy_softmax_temp: 1.0,
                draw_score: cmd.draw_score.clamp(-1.0, 1.0),
                moves_left_max_effect: if cmd.moves_left_utility { 0.25 } else { 0.0 },
                moves_left_slope: if cmd.moves_left_utility { 0.004 } else { 0.0 },
                moves_left_threshold: 0.7,
                moves_left_constant_factor: if cmd.moves_left_utility { 0.05 } else { 0.0 },
                moves_left_scaled_factor: if cmd.moves_left_utility { 0.20 } else { 0.0 },
                moves_left_quadratic_factor: if cmd.moves_left_utility { 0.75 } else { 0.0 },
                value_scale: 1.0,
            };
            let result = alphazero_search_with_rules(
                &position,
                Some(rule_history.clone()),
                None,
                &model,
                search_limits,
            );
            println!("fen      : {}", position.to_fen());
            println!("model    : {model_path}");
            println!("sims     : {}", result.simulations);
            println!("search   : alphazero");
            println!("cpuct    : {cpuct}");
            println!("cpuct_at_root: {cpuct_at_root}");
            println!("fpu_value: {}", cmd.fpu_value);
            println!("fpu_value_at_root: {}", cmd.fpu_value_at_root);
            println!("draw_score: {}", cmd.draw_score);
            println!("moves_left_utility: {}", cmd.moves_left_utility);
            println!(
                "depth    : avg={:.2} max={} limit={} cutoffs={}",
                result.search_depth_avg,
                result.search_depth_max,
                result.search_depth_limit,
                result.search_depth_cutoffs
            );
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
            for candidate in &result.candidates {
                println!(
                    "candidate: {} visits={} q={:.3} ml={:.1} prior={:.5} policy={:.5}",
                    candidate.mv,
                    candidate.visits,
                    candidate.q,
                    candidate.moves_left,
                    candidate.prior,
                    candidate.policy
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
            for candidate in &by_visits {
                println!(
                    "visited: {} visits={} q={:.3} ml={:.1} prior={:.5} policy={:.5}",
                    candidate.mv,
                    candidate.visits,
                    candidate.q,
                    candidate.moves_left,
                    candidate.prior,
                    candidate.policy
                );
            }
            let verify_sims = if cmd.verify_sims == 0 {
                simulations
            } else {
                cmd.verify_sims
            };
            let mut verify_moves = by_visits
                .iter()
                .take(cmd.verify_top)
                .map(|candidate| candidate.mv)
                .collect::<Vec<_>>();
            for text in &cmd.verify_moves {
                let mv = position.parse_uci_move(text).unwrap_or_else(|| {
                    panic!("invalid or illegal --verify-move `{text}` for this position")
                });
                if !verify_moves.contains(&mv) {
                    verify_moves.push(mv);
                }
            }
            if !verify_moves.is_empty() {
                println!("verify_children: sims={verify_sims}");
            }
            for mv in verify_moves {
                let Some(root_candidate) = result.candidates.iter().find(|item| item.mv == mv)
                else {
                    println!("verify: {mv} unavailable_at_root");
                    continue;
                };
                let mut child_rule_history = rule_history.clone();
                child_rule_history.push(position.rule_history_entry_after_move(mv));
                let mut child = position.clone();
                child.make_move(mv);
                let child_legal = child.legal_moves_with_rules(&child_rule_history);
                let child_nn_q = model.evaluate_value(&child, &child_legal);
                let mut verify_limits = search_limits;
                verify_limits.simulations = verify_sims.max(1);
                verify_limits.seed = 0;
                let verified = alphazero_search_with_rules(
                    &child,
                    Some(child_rule_history),
                    Some(child_legal),
                    &model,
                    verify_limits,
                );
                let verified_root_q = -verified.value_q;
                let verified_root_cp = -verified.value_cp;
                println!(
                    "verify: {} root_q={:.3} root_visits={} prior={:.5} child_nn_q={:.3} verified_q={:.3} verified_cp={} delta={:+.3} child_best={}",
                    mv,
                    root_candidate.q,
                    root_candidate.visits,
                    root_candidate.prior,
                    -child_nn_q,
                    verified_root_q,
                    verified_root_cp,
                    verified_root_q - root_candidate.q,
                    verified
                        .best_move
                        .map(|best| best.to_string())
                        .unwrap_or_else(|| "(none)".into())
                );
            }
        }
        Some(CliCommand::AzBench(cmd)) => {
            let model_path = cmd.model;
            let simulations = cmd.simulations.max(1);
            let repeat = cmd.repeat.max(1);
            let cpuct = cmd.cpuct.max(0.0);
            let fen = cmd.fen.join(" ");
            let position = parse_position(&fen);
            let model = AzNnue::load(&model_path).unwrap_or_else(|err| {
                panic!("failed to load `{model_path}`: {err}");
            });

            let _ = alphazero_search(
                &position,
                &model,
                fixed_az_search_limits(simulations, 0, cpuct, 0),
            );

            let started = std::time::Instant::now();
            let mut total_sims = 0usize;
            let mut best_move = None;
            for iteration in 0..repeat {
                let result = alphazero_search(
                    &position,
                    &model,
                    fixed_az_search_limits(simulations, iteration as u64, cpuct, 0),
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
            println!("repeat       : {repeat}");
            println!("search       : alphazero");
            println!("simd         : {}", chineseai::az::inference_simd_backend());
            println!("cpuct        : {cpuct}");
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
        Some(CliCommand::AzTrainBench(cmd)) => {
            let model_path = cmd.model;
            let sample_count = cmd.samples.max(1);
            let epochs = cmd.epochs.max(1);
            let batch_size = cmd.batch_size_per_gpu.max(1);
            let lr = cmd.lr.max(0.0);
            let seed = cmd.seed;
            let mut model = AzNnue::load(&model_path).unwrap_or_else(|err| {
                panic!("failed to load `{model_path}`: {err}");
            });
            let started = std::time::Instant::now();
            let stats = benchmark_training(&mut model, sample_count, epochs, batch_size, lr, seed);
            let elapsed = started.elapsed().as_secs_f64().max(f64::EPSILON);
            let processed = (sample_count * epochs) as f64;
            let g_step = global_training_step_sample_count(batch_size);
            let n_gpu = global_training_step_sample_count(1);
            println!("bench        : training");
            println!("model        : {model_path}");
            println!("samples      : {sample_count}");
            println!("epochs       : {epochs}");
            println!("batch(per_gpu) : {batch_size}");
            println!("cuda_devices  : {n_gpu}  (global batch {g_step})");
            println!("lr             : {lr}");
            println!("elapsed_ms   : {:.3}", elapsed * 1000.0);
            println!("processed    : {}", sample_count * epochs);
            println!("samples/sec  : {:.0}", processed / elapsed);
            println!("loss         : {:.4}", stats.loss);
            println!("value_ce     : {:.4}", stats.value_loss);
            println!("policy_ce    : {:.4}", stats.policy_ce);
        }
        Some(CliCommand::AzLoop(cmd)) => {
            let config_path = cmd.config;
            let Some(config) = load_or_create_az_loop_config(&config_path) else {
                return;
            };
            let target_update = cmd.target_update.map(|update| update.max(1));
            let progress_boot = load_az_loop_progress(&config_path);
            let start_update = progress_boot.next_update.max(1);
            let mut arena_best_elo = progress_boot.best_elo;
            if let Some(target_update) = target_update
                && start_update > target_update
            {
                println!(
                    "target   : already complete, start_update={} target_update={}",
                    start_update, target_update
                );
                return;
            }
            if start_update > 1 {
                println!(
                    "resume   : update starts at {} (from `{}`) arena_ref_elo={:.1}",
                    start_update,
                    az_loop_progress_path(&config_path).display(),
                    arena_best_elo
                );
            }
            let best_path = best_model_path(&config.model_path);

            let config_arch = config.arch();
            let model_path = Path::new(&config.model_path);
            let (model, resumed_model) = if model_path.exists() {
                println!("model    : load {}", config.model_path);
                match AzNnue::load(model_path) {
                    Ok(model) => {
                        if model.arch != config_arch {
                            println!(
                                "model    : loaded arch={:?} differs from config arch={:?}; keep loaded arch",
                                model.arch, config_arch
                            );
                        }
                        fs::remove_file(model_path).unwrap_or_else(|err| {
                            panic!(
                                "loaded model but failed to remove consumed `{}`: {err}",
                                model_path.display()
                            )
                        });
                        println!("resume   : consumed `{}` into memory", model_path.display());
                        (model, true)
                    }
                    Err(err) => {
                        println!(
                            "model    : reinit {} as random nnue ({err})",
                            config.model_path
                        );
                        (AzNnue::random_with_arch(config_arch, config.seed), false)
                    }
                }
            } else if config.arena_interval > 0 && best_path.exists() {
                println!("model    : load best `{}` as current", best_path.display());
                (
                    AzNnue::load(&best_path).unwrap_or_else(|err| {
                        panic!("failed to load best model `{}`: {err}", best_path.display());
                    }),
                    true,
                )
            } else {
                println!("model    : init {}", config.model_path);
                (AzNnue::random_with_arch(config_arch, config.seed), false)
            };
            let swa_path = swa_model_path(&config.model_path);
            let (mut swa_model, mut swa_count) = if resumed_model && swa_path.exists() {
                let loaded = AzNnue::load(&swa_path).unwrap_or_else(|err| {
                    panic!("failed to load SWA model `{}`: {err}", swa_path.display())
                });
                if loaded.arch != model.arch {
                    panic!(
                        "SWA model `{}` architecture {:?} differs from raw {:?}",
                        swa_path.display(),
                        loaded.arch,
                        model.arch
                    );
                }
                fs::remove_file(&swa_path).unwrap_or_else(|err| {
                    panic!(
                        "loaded SWA model but failed to remove consumed `{}`: {err}",
                        swa_path.display()
                    )
                });
                let count = progress_boot.swa_count.clamp(1, SWA_MAX_MODELS);
                println!(
                    "swa      : load `{}` count={}/{}",
                    swa_path.display(),
                    count,
                    SWA_MAX_MODELS
                );
                (Some(loaded), count)
            } else if resumed_model {
                println!("swa      : initialize from resumed raw model count=1");
                (Some(model.clone()), 1)
            } else {
                println!("swa      : initialize after first trained update");
                (None, 0)
            };
            let selfplay_model = {
                if let Some(average) = swa_model.as_ref() {
                    println!("selfplay : start from SWA count={swa_count}");
                    average.clone()
                } else {
                    println!("selfplay : start from initial raw model");
                    model.clone()
                }
            };
            let initial_arena_reference_model = if config.arena_interval == 0 {
                selfplay_model.clone()
            } else {
                if !best_path.exists() {
                    save_model(&selfplay_model, &best_path);
                } else if let Err(err) = AzNnue::load(&best_path) {
                    println!(
                        "best     : reset incompatible `{}` from deployed model ({err})",
                        best_path.display()
                    );
                    save_model(&selfplay_model, &best_path);
                }
                AzNnue::load(&best_path).unwrap_or_else(|err| {
                    panic!("failed to load best model `{}`: {err}", best_path.display());
                })
            };
            let replay_snapshot_path = az_loop_replay_snapshot_path(&config_path);
            let mut replay_pool =
                (config.replay_capacity > 0).then(|| AzExperiencePool::new(config.replay_capacity));
            if config.replay_capacity > 0 && replay_snapshot_path.exists() {
                match AzExperiencePool::load_snapshot_lz4(
                    &replay_snapshot_path,
                    config.replay_capacity,
                ) {
                    Ok(pool) => {
                        fs::remove_file(&replay_snapshot_path).unwrap_or_else(|err| {
                            panic!(
                                "failed to remove replay snapshot `{}`: {err}",
                                replay_snapshot_path.display()
                            );
                        });
                        println!(
                            "replay   : restored {}/{} samples from `{}` (file removed)",
                            pool.sample_count(),
                            pool.capacity(),
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
            let stop_requested = Arc::new(AtomicBool::new(false));
            let interrupted_flag = interrupted.clone();
            let stop_flag = stop_requested.clone();
            ctrlc::set_handler(move || {
                interrupted_flag.store(true, Ordering::SeqCst);
                stop_flag.store(true, Ordering::SeqCst);
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
            let opening_positions = load_opening_positions(&config.opening_fens_path);
            let next_selfplay_generation = replay_pool
                .as_ref()
                .map(AzExperiencePool::max_generation_update)
                .unwrap_or(0)
                .saturating_add(1)
                .max(1);
            let effective_train_to_selfplay_ratio = (config.train_samples_per_update as f32
                * config.train_epochs_per_update as f32)
                / config.selfplay_samples_per_update.max(1) as f32;

            println!(
                "loop     : config={} mode=batch search=alphazero sims={} swa_max_models={} replay_recent(fraction={},games={}) selfplay_samples_per_update={} train_to_selfplay_ratio={:.2} lr={} lr_decay(min={},start={},interval={},factor={}) batch_size(per_gpu)={} global_step_samples={} train_warmup_samples={} train_samples_per_update={} train_epochs_per_update={} max_plies={} selfplay_workers={} temp(start={},endgame={},delay={}ply,decay={}ply,value_cutoff={},visit_offset={}) cpuct={} cpuct_at_root={} fpu(value={},root={}) policy_softmax_temp={} root_noise(alpha={},fraction={}) opening_fens={} opening_count={} resign(percentage={},playthrough={}) replay_capacity={} mirror_probability={} train(value={},policy={}) checkpoint_interval={} max_checkpoints={} arena_interval={} arena_sims={} arena_cpuct={} arena_promotion_rate={} arena_promotion_z={} arena_processes={} arena_opening_book={} arena_opening_positions={} arena_opening_plies={}-{} pikafish_label_eval(sqlite={},interval={},limit={},sims={},cpuct={}) tb_base={} tb_run={}",
                config_path,
                config.simulations,
                SWA_MAX_MODELS,
                config.replay_recent_sample_fraction,
                config.replay_recent_games,
                config.selfplay_samples_per_update,
                effective_train_to_selfplay_ratio,
                config.lr,
                config.lr_min,
                config.lr_decay_start_update,
                config.lr_decay_interval,
                config.lr_decay_factor,
                config.batch_size,
                global_training_step_sample_count(config.batch_size),
                config.train_warmup_samples,
                config.train_samples_per_update,
                config.train_epochs_per_update,
                config.max_plies,
                config.workers,
                config.temperature_start,
                config.temperature_endgame,
                config.temperature_decay_delay_plies,
                config.temperature_decay_plies,
                config.temperature_value_cutoff,
                config.temperature_visit_offset,
                config.cpuct,
                config.cpuct_at_root,
                config.fpu_value,
                config.fpu_value_at_root,
                config.policy_softmax_temp,
                config.root_dirichlet_alpha,
                config.root_exploration_fraction,
                if config.opening_fens_path.trim().is_empty() {
                    "(none)"
                } else {
                    config.opening_fens_path.as_str()
                },
                opening_positions.len(),
                config.resign_percentage,
                config.resign_playthrough,
                config.replay_capacity,
                config.mirror_probability,
                config.train_value_weight,
                config.train_policy_weight,
                config.checkpoint_interval,
                config.max_checkpoints,
                config.arena_interval,
                config.arena_simulations,
                config.arena_cpuct,
                config.arena_promotion_rate,
                config.arena_promotion_confidence_z,
                config.arena_processes,
                if config.arena_opening_book.trim().is_empty() {
                    "(none)"
                } else {
                    config.arena_opening_book.as_str()
                },
                config.arena_opening_positions,
                config.arena_opening_plies_min,
                config.arena_opening_plies_max,
                if config.pikafish_label_eval_sqlite.trim().is_empty() {
                    "(none)"
                } else {
                    config.pikafish_label_eval_sqlite.as_str()
                },
                config.pikafish_label_eval_interval,
                config.pikafish_label_eval_limit,
                config.pikafish_label_eval_simulations,
                config.pikafish_label_eval_cpuct,
                config.tensorboard_logdir,
                tensorboard_encoded_subdir(&config)
            );
            let cpu_placements = chineseai::cpu_topology::cpu_placements();
            let numa_nodes = chineseai::cpu_topology::numa_nodes(&cpu_placements);
            let selfplay_worker_count = config.workers.max(1).min(cpu_placements.len().max(1));
            // Trainer在一次GPU反向传播期间不会消费自博弈结果。双worker容量在高吞吐
            // 服务器上会很快填满，令全部worker周期性阻塞在send。保留有界背压，
            // 但缓冲足够覆盖一个训练窗口。
            let selfplay_queue_capacity = selfplay_worker_count.saturating_mul(8).max(32);
            let (selfplay_tx, selfplay_rx) =
                mpsc::sync_channel::<SelfplayBatch>(selfplay_queue_capacity);
            let (trainer_tx, trainer_rx) = mpsc::sync_channel::<TrainerEvent>(2);
            let physical_cpus = cpu_placements
                .iter()
                .filter(|placement| placement.smt_level == 0)
                .count();
            let llc_domains = cpu_placements
                .iter()
                .map(|placement| (placement.node, placement.package, placement.llc))
                .collect::<HashSet<_>>()
                .len();
            let smt_workers = selfplay_worker_count.saturating_sub(physical_cpus);
            println!(
                "cpu      : allowed={} physical={} numa_nodes={} llc_domains={} selfplay_workers={} physical_workers={} smt_workers={} selfplay_queue={} affinity={} model_replicas={}",
                cpu_placements.len(),
                physical_cpus,
                numa_nodes.len(),
                llc_domains,
                selfplay_worker_count,
                selfplay_worker_count.min(physical_cpus),
                smt_workers,
                selfplay_queue_capacity,
                if cfg!(target_os = "linux") {
                    "on"
                } else {
                    "unsupported"
                },
                numa_nodes.len(),
            );
            let initial_numa_models = build_numa_model_replicas(&selfplay_model, &numa_nodes);
            let shared_model = Arc::new(RwLock::new(SharedSelfplayModel {
                version: start_update.saturating_sub(1) as u64,
                models_by_numa_node: initial_numa_models,
            }));
            let mut arena_reference_model = initial_arena_reference_model;
            let selfplay_pause =
                Arc::new((Mutex::new(SelfplayPauseState::default()), Condvar::new()));
            println!("generation: next_selfplay={next_selfplay_generation}");
            let selfplay_generation = Arc::new(std::sync::atomic::AtomicU64::new(
                next_selfplay_generation as u64,
            ));
            let mut selfplay_handles = Vec::with_capacity(selfplay_worker_count);
            for worker_id in 0..selfplay_worker_count {
                let placement = cpu_placements[worker_id % cpu_placements.len()];
                let model_slot = numa_nodes
                    .iter()
                    .position(|&(node, _)| node == placement.node)
                    .unwrap_or(0);
                let selfplay_stop = stop_requested.clone();
                let selfplay_config = config.clone();
                let selfplay_opening_positions = opening_positions.clone();
                let selfplay_tx = selfplay_tx.clone();
                let shared_model = Arc::clone(&shared_model);
                let selfplay_pause = Arc::clone(&selfplay_pause);
                let selfplay_generation = Arc::clone(&selfplay_generation);
                selfplay_handles.push(thread::spawn(move || {
                    if let Err(err) = chineseai::cpu_topology::pin_current_thread(placement.cpu) {
                        eprintln!(
                            "warning: failed to pin selfplay worker {worker_id} to cpu {}: {err}",
                            placement.cpu
                        );
                    }
                    let mut batch_index = 0usize;
                    let mut local_version = u64::MAX;
                    let mut local_model: Option<Arc<AzNnue>> = None;
                    while !selfplay_stop.load(Ordering::SeqCst) {
                        {
                            let (pause_lock, pause_cvar) = &*selfplay_pause;
                            let mut pause_state = pause_lock
                                .lock()
                                .unwrap_or_else(|_| panic!("selfplay pause state poisoned"));
                            while pause_state.is_paused() && !selfplay_stop.load(Ordering::SeqCst) {
                                pause_state = pause_cvar
                                    .wait(pause_state)
                                    .unwrap_or_else(|_| panic!("selfplay pause state poisoned"));
                            }
                        }
                        if selfplay_stop.load(Ordering::SeqCst) {
                            break;
                        }
                        {
                            let shared = shared_model
                                .read()
                                .unwrap_or_else(|_| panic!("shared selfplay model poisoned"));
                            if shared.version != local_version {
                                local_model =
                                    Some(Arc::clone(&shared.models_by_numa_node[model_slot]));
                                local_version = shared.version;
                            }
                        }
                        let batch_seed = selfplay_config.seed
                            ^ ((worker_id as u64).wrapping_add(1) << 32)
                            ^ (batch_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                        let generation_update = selfplay_generation
                            .fetch_add(1, Ordering::SeqCst)
                            .min(u32::MAX as u64)
                            as u32;
                        let loop_config = build_az_loop_config(
                            &selfplay_config,
                            batch_seed,
                            1,
                            generation_update,
                            &selfplay_opening_positions,
                        );
                        let started = Instant::now();
                        let data = generate_selfplay_data(
                            local_model
                                .as_deref()
                                .expect("selfplay model not initialized"),
                            &loop_config,
                        );
                        let batch = SelfplayBatch {
                            data,
                            selfplay_seconds: started.elapsed().as_secs_f32(),
                        };
                        if selfplay_tx.send(batch).is_err() {
                            break;
                        }
                        batch_index += 1;
                    }
                }));
            }
            drop(selfplay_tx);
            // 独立收集线程持续排空worker结果，并在CPU侧组装完整更新批次。
            // GPU训练期间下一批仍可并行生成；只缓存一个完整更新，限制模型滞后。
            let (ready_tx, ready_rx) = mpsc::sync_channel::<PendingTrainingData>(1);
            let collector_config = config.clone();
            let collector_needs_warmup = !resumed_model;
            println!(
                "warmup   : {} (model={})",
                if collector_needs_warmup {
                    format!("{} samples", config.train_warmup_samples)
                } else {
                    "skipped".to_string()
                },
                if resumed_model { "resumed" } else { "random" }
            );
            let collector_handle = thread::spawn(move || {
                let mut pending = PendingTrainingData::default();
                let mut batch_index = 0usize;
                while let Ok(batch) = selfplay_rx.recv() {
                    pending.push(batch);
                    let required_samples = if collector_needs_warmup && batch_index == 0 {
                        collector_config
                            .train_warmup_samples
                            .max(collector_config.selfplay_samples_per_update)
                    } else {
                        collector_config.selfplay_samples_per_update
                    };
                    if pending.selfplay.samples.len() < required_samples {
                        continue;
                    }
                    if ready_tx.send(std::mem::take(&mut pending)).is_err() {
                        break;
                    }
                    batch_index += 1;
                }
            });
            let trainer_stop = stop_requested.clone();
            let trainer_interrupted = interrupted.clone();
            let trainer_config = config.clone();
            let trainer_start_update = start_update;
            let trainer_snapshot_path = replay_snapshot_path.clone();
            let trainer_handle = thread::spawn(move || {
                let mut trainer_model = model;
                let mut trainer_pool = replay_pool;
                let mut total_games_generated = 0usize;
                let mut total_samples_generated = 0usize;
                let mut train_index = 0usize;
                let min_train_samples =
                    global_training_step_sample_count(trainer_config.batch_size);
                'training: while let Ok(mut pending) = ready_rx.recv() {
                    let pending_games = pending.selfplay.games.len();
                    let pending_samples = pending.selfplay.samples.len();
                    if let Some(pool) = trainer_pool.as_mut() {
                        pool.add_games(std::mem::take(&mut pending.selfplay.games));
                    }
                    total_games_generated += pending_games;
                    total_samples_generated += pending_samples;
                    if trainer_stop.load(Ordering::SeqCst) {
                        continue;
                    }
                    if trainer_stop.load(Ordering::SeqCst) {
                        continue;
                    }
                    let Some(pool) = trainer_pool.as_mut() else {
                        continue;
                    };
                    if pool.sample_count() < min_train_samples {
                        continue;
                    }
                    let mut rng = chineseai::az::SplitMix64::new(
                        trainer_config.seed
                            ^ (train_index as u64).wrapping_mul(0xD1B5_4A32_D192_ED03),
                    );
                    let sampled_batch = pool.sample_mixed_recent(
                        trainer_config.train_samples_per_update,
                        trainer_config.replay_recent_sample_fraction,
                        trainer_config.replay_recent_games,
                        &mut rng,
                    );
                    let train_data = sampled_batch.samples;
                    if train_data.is_empty() {
                        continue;
                    }
                    let train_data_len = train_data.len();
                    let train_source_stats = train_batch_source_stats(
                        &train_data,
                        trainer_config.simulations,
                        sampled_batch.recent_samples,
                        pool.max_generation_update(),
                        trainer_config.replay_recent_games,
                    );
                    let train_update = trainer_start_update.saturating_add(train_index);
                    let current_lr = learning_rate_for_update(&trainer_config, train_update);
                    let train_started = Instant::now();
                    let stats = train_samples_weighted_owned(
                        &mut trainer_model,
                        train_data,
                        trainer_config.train_epochs_per_update,
                        current_lr,
                        trainer_config.batch_size,
                        &mut rng,
                        AzTrainLossWeights {
                            value: trainer_config.train_value_weight,
                            policy: trainer_config.train_policy_weight,
                        },
                    );
                    let train_seconds = train_started.elapsed().as_secs_f32();
                    let report = build_async_training_report(
                        pending,
                        pending_games,
                        total_games_generated,
                        total_samples_generated,
                        stats,
                        current_lr,
                        train_data_len,
                        train_seconds,
                        pool.sample_count(),
                        pool.capacity(),
                        pool.window_stats(trainer_config.replay_recent_games),
                        train_source_stats,
                    );
                    if trainer_tx
                        .send(TrainerEvent {
                            report,
                            candidate_model: trainer_model.clone(),
                        })
                        .is_err()
                    {
                        break 'training;
                    }
                    train_index += 1;
                }
                if let Some(pool) = trainer_pool.as_mut()
                    && trainer_interrupted.load(Ordering::SeqCst)
                {
                    match pool.save_snapshot_lz4(&trainer_snapshot_path) {
                        Ok(()) => {
                            if pool.sample_count() > 0 {
                                println!(
                                    "replay   : interrupt snapshot `{}` ({}/{} samples)",
                                    trainer_snapshot_path.display(),
                                    pool.sample_count(),
                                    pool.capacity()
                                );
                            }
                        }
                        Err(err) => eprintln!("replay   : failed to write snapshot: {err}"),
                    }
                }
            });
            let mut exited_after_ctrl_c = false;
            let mut exited_after_target_update = false;
            let mut update = start_update;
            let mut interrupt_save_model: Option<AzNnue> = None;
            let mut interrupt_save_next_update = start_update;
            loop {
                if interrupted.load(Ordering::SeqCst) {
                    exited_after_ctrl_c = true;
                    break;
                }
                let started = Instant::now();
                let (report, candidate_model) = loop {
                    match trainer_rx.recv_timeout(Duration::from_millis(100)) {
                        Ok(TrainerEvent {
                            report,
                            candidate_model,
                        }) => break (report, candidate_model),
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            if interrupted.load(Ordering::SeqCst) {
                                exited_after_ctrl_c = true;
                                break (
                                    AzLoopReport {
                                        games: 0,
                                        samples: 0,
                                        red_wins: 0,
                                        black_wins: 0,
                                        draws: 0,
                                        avg_plies: 0.0,
                                        loss: 0.0,
                                        learning_rate: 0.0,
                                        value_loss: 0.0,
                                        value_mse: 0.0,
                                        value_pred_mean: 0.0,
                                        value_target_mean: 0.0,
                                        value_pred_rms: 0.0,
                                        value_target_rms: 0.0,
                                        value_corr: 0.0,
                                        value_calibration: 0.0,
                                        policy_ce: 0.0,
                                        policy_kl: 0.0,
                                        root_visit_entropy: 0.0,
                                        entropy_opening: 0.0,
                                        entropy_mid: 0.0,
                                        raw_prior_top1: 0.0,
                                        raw_prior_top2: 0.0,
                                        policy_top1: 0.0,
                                        policy_top2: 0.0,
                                        root_q_gap: 0.0,
                                        root_q_top1_abs: 0.0,
                                        visited_actions: 0.0,
                                        opening_raw_prior_top1: 0.0,
                                        opening_raw_prior_top2: 0.0,
                                        opening_policy_top1: 0.0,
                                        opening_policy_top2: 0.0,
                                        opening_q_gap: 0.0,
                                        opening_q_top1_abs: 0.0,
                                        opening_visited_actions: 0.0,
                                        sampled_best_rate: 0.0,
                                        avg_best_played_q_gap: 0.0,
                                        avg_played_top_visit_ratio: 0.0,
                                        avg_best_q: 0.0,
                                        avg_played_q: 0.0,
                                        selfplay_seconds: 0.0,
                                        train_seconds: 0.0,
                                        total_seconds: 0.0,
                                        games_per_second: 0.0,
                                        samples_per_second: 0.0,
                                        train_samples_per_second: 0.0,
                                        train_samples: 0,
                                        pool_samples: 0,
                                        pool_capacity: config.replay_capacity,
                                        terminal_no_legal_moves: 0,
                                        terminal_red_general_missing: 0,
                                        terminal_black_general_missing: 0,
                                        terminal_rule_draw: 0,
                                        terminal_rule_draw_halfmove120: 0,
                                        terminal_rule_draw_repetition: 0,
                                        terminal_rule_draw_mutual_long_check: 0,
                                        terminal_rule_draw_mutual_long_chase: 0,
                                        terminal_rule_win_red: 0,
                                        terminal_rule_win_black: 0,
                                        terminal_resign_red: 0,
                                        terminal_resign_black: 0,
                                        terminal_max_plies: 0,
                                        ..AzLoopReport::default()
                                    },
                                    AzNnue::random_with_arch(config.arch(), config.seed),
                                );
                            }
                        }
                        Err(mpsc::RecvTimeoutError::Disconnected) => {
                            if interrupted.load(Ordering::SeqCst) {
                                exited_after_ctrl_c = true;
                                break (
                                    AzLoopReport {
                                        games: 0,
                                        samples: 0,
                                        red_wins: 0,
                                        black_wins: 0,
                                        draws: 0,
                                        avg_plies: 0.0,
                                        loss: 0.0,
                                        learning_rate: 0.0,
                                        value_loss: 0.0,
                                        value_mse: 0.0,
                                        value_pred_mean: 0.0,
                                        value_target_mean: 0.0,
                                        value_pred_rms: 0.0,
                                        value_target_rms: 0.0,
                                        value_corr: 0.0,
                                        value_calibration: 0.0,
                                        policy_ce: 0.0,
                                        policy_kl: 0.0,
                                        root_visit_entropy: 0.0,
                                        entropy_opening: 0.0,
                                        entropy_mid: 0.0,
                                        raw_prior_top1: 0.0,
                                        raw_prior_top2: 0.0,
                                        policy_top1: 0.0,
                                        policy_top2: 0.0,
                                        root_q_gap: 0.0,
                                        root_q_top1_abs: 0.0,
                                        visited_actions: 0.0,
                                        opening_raw_prior_top1: 0.0,
                                        opening_raw_prior_top2: 0.0,
                                        opening_policy_top1: 0.0,
                                        opening_policy_top2: 0.0,
                                        opening_q_gap: 0.0,
                                        opening_q_top1_abs: 0.0,
                                        opening_visited_actions: 0.0,
                                        sampled_best_rate: 0.0,
                                        avg_best_played_q_gap: 0.0,
                                        avg_played_top_visit_ratio: 0.0,
                                        avg_best_q: 0.0,
                                        avg_played_q: 0.0,
                                        selfplay_seconds: 0.0,
                                        train_seconds: 0.0,
                                        total_seconds: 0.0,
                                        games_per_second: 0.0,
                                        samples_per_second: 0.0,
                                        train_samples_per_second: 0.0,
                                        train_samples: 0,
                                        pool_samples: 0,
                                        pool_capacity: config.replay_capacity,
                                        terminal_no_legal_moves: 0,
                                        terminal_red_general_missing: 0,
                                        terminal_black_general_missing: 0,
                                        terminal_rule_draw: 0,
                                        terminal_rule_draw_halfmove120: 0,
                                        terminal_rule_draw_repetition: 0,
                                        terminal_rule_draw_mutual_long_check: 0,
                                        terminal_rule_draw_mutual_long_chase: 0,
                                        terminal_rule_win_red: 0,
                                        terminal_rule_win_black: 0,
                                        terminal_resign_red: 0,
                                        terminal_resign_black: 0,
                                        terminal_max_plies: 0,
                                        ..AzLoopReport::default()
                                    },
                                    AzNnue::random_with_arch(config.arch(), config.seed),
                                );
                            }
                            panic!("training thread exited before update {update}");
                        }
                    }
                };
                if exited_after_ctrl_c {
                    break;
                }
                update_swa_model(&mut swa_model, &mut swa_count, &candidate_model);
                let deployed_model = swa_model
                    .as_ref()
                    .expect("SWA model must exist after a trained update")
                    .clone();
                interrupt_save_model = Some(candidate_model.clone());
                interrupt_save_next_update = update.saturating_add(1);
                let checkpoint_saved = if config.checkpoint_interval > 0
                    && update.is_multiple_of(config.checkpoint_interval)
                {
                    let path = save_checkpoint_model(
                        &candidate_model,
                        &config.model_path,
                        &config.checkpoint_dir,
                        update,
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
                let value_rmse = report.value_mse.max(0.0).sqrt();
                println!(
                    "update {update:04}: games={} samples={} total_samples={} train_samples={} pool={}/{} fill={:.0}% replay(chunks={} games={}-{} span_games={} recent_pool={:.3}) train_src(recent_quota={:.3} actual_recent={:.3} fast={:.3} pw={:.3} vw={:.3}) R/B/D={}/{}/{} red_win_all={:.3} avg_plies={:.1} avg_sims={:.1} swa_n={} opt_loss={:.4} wdl_ce={:.4} ml_log_mse={:.4} trainQ_rmse={:.4} trainQ_mu={:.3}/{:.3} trainQ_rms={:.3}/{:.3} trainQ_corr={:.3} trainQ_cal={:.3} trainPhaseQ(p0_39={}/{:.3}/{:.3}/{:.3} p40_119={}/{:.3}/{:.3}/{:.3} p120plus={}/{:.3}/{:.3}/{:.3}) policy_kl={:.4} trainTargetH={:.4} lr={:.6} visitH={:.3} visitH_p0_89={:.3} visitH_p90plus={:.3} rawP={:.3}/{:.3} visitP={:.3}/{:.3} trainTargetP={:.3}/{:.3} topQgap={:.3} topQabs={:.3} visitA={:.1} sampTopQ={:.3} playQGap={:.3} visitRatio={:.3} maxQ={:.3} playedQ={:.3} train={:.1}s gps={:.2} sps={:.1} train_sps={:.1} elapsed={:.1}s{}",
                    report.games,
                    report.samples,
                    report.total_samples_generated,
                    report.train_samples,
                    report.pool_samples,
                    report.pool_capacity,
                    if report.pool_capacity == 0 {
                        0.0
                    } else {
                        100.0 * report.pool_samples as f32 / report.pool_capacity as f32
                    },
                    report.replay_chunks,
                    report.replay_oldest_update,
                    report.replay_newest_update,
                    report.replay_window_games,
                    report.replay_recent_window_fraction,
                    report.train_recent_quota_rate,
                    report.train_actual_recent_sample_rate,
                    report.train_fast_sample_rate,
                    report.train_policy_weight_mean,
                    report.train_value_weight_mean,
                    report.red_wins,
                    report.black_wins,
                    report.draws,
                    report.red_wins as f32 / report.games.max(1) as f32,
                    report.avg_plies,
                    report.avg_search_simulations,
                    swa_count,
                    report.loss,
                    report.value_loss,
                    report.moves_left_loss,
                    value_rmse,
                    report.value_pred_mean,
                    report.value_target_mean,
                    report.value_pred_rms,
                    report.value_target_rms,
                    report.value_corr,
                    report.value_calibration,
                    report.phase_value[0].samples,
                    report.phase_value[0].rmse,
                    report.phase_value[0].corr,
                    report.phase_value[0].calibration,
                    report.phase_value[1].samples,
                    report.phase_value[1].rmse,
                    report.phase_value[1].corr,
                    report.phase_value[1].calibration,
                    report.phase_value[2].samples,
                    report.phase_value[2].rmse,
                    report.phase_value[2].corr,
                    report.phase_value[2].calibration,
                    report.policy_kl,
                    report.policy_target_entropy,
                    report.learning_rate,
                    report.root_visit_entropy,
                    report.entropy_opening,
                    report.entropy_mid,
                    report.raw_prior_top1,
                    report.raw_prior_top2,
                    report.policy_top1,
                    report.policy_top2,
                    report.train_policy_target_top1,
                    report.train_policy_target_top2,
                    report.root_q_gap,
                    report.root_q_top1_abs,
                    report.visited_actions,
                    report.sampled_best_rate,
                    report.avg_best_played_q_gap,
                    report.avg_played_top_visit_ratio,
                    report.avg_best_q,
                    report.avg_played_q,
                    report.train_seconds,
                    report.games_per_second,
                    report.samples_per_second,
                    report.train_samples_per_second,
                    started.elapsed().as_secs_f32(),
                    checkpoint_saved
                        .as_ref()
                        .map_or_else(String::new, |path| format!(
                            " checkpoint={}",
                            path.display()
                        ))
                );
                log_scalar(&mut tb, "train/optimized_loss", update, report.loss);
                log_scalar(&mut tb, "train/wdl_ce", update, report.value_loss);
                log_scalar(
                    &mut tb,
                    "train/moves_left_log_mse",
                    update,
                    report.moves_left_loss,
                );
                log_scalar(&mut tb, "train/value_rmse", update, value_rmse);
                log_scalar(
                    &mut tb,
                    "train/value_pred_mean",
                    update,
                    report.value_pred_mean,
                );
                log_scalar(
                    &mut tb,
                    "train/value_target_mean",
                    update,
                    report.value_target_mean,
                );
                log_scalar(&mut tb, "train/value_corr", update, report.value_corr);
                log_scalar(
                    &mut tb,
                    "train/value_calibration",
                    update,
                    report.value_calibration,
                );
                for (phase, name) in ["ply_0_39", "ply_40_119", "ply_120_plus"]
                    .into_iter()
                    .enumerate()
                {
                    let phase_value = report.phase_value[phase];
                    log_scalar(
                        &mut tb,
                        &format!("train/value_{name}_samples"),
                        update,
                        phase_value.samples as f32,
                    );
                    log_scalar(
                        &mut tb,
                        &format!("train/value_{name}_rmse"),
                        update,
                        phase_value.rmse,
                    );
                    log_scalar(
                        &mut tb,
                        &format!("train/value_{name}_corr"),
                        update,
                        phase_value.corr,
                    );
                    log_scalar(
                        &mut tb,
                        &format!("train/value_{name}_calibration"),
                        update,
                        phase_value.calibration,
                    );
                }
                log_scalar(&mut tb, "train/policy_ce", update, report.policy_ce);
                log_scalar(&mut tb, "train/policy_kl", update, report.policy_kl);
                log_scalar(
                    &mut tb,
                    "train/policy_target_entropy",
                    update,
                    report.policy_target_entropy,
                );
                log_scalar(
                    &mut tb,
                    "train/policy_target_top1",
                    update,
                    report.train_policy_target_top1,
                );
                log_scalar(
                    &mut tb,
                    "train/policy_target_top2",
                    update,
                    report.train_policy_target_top2,
                );
                log_scalar(&mut tb, "train/lr", update, report.learning_rate);
                log_scalar(
                    &mut tb,
                    "pool/fill_ratio",
                    update,
                    if report.pool_capacity == 0 {
                        0.0
                    } else {
                        report.pool_samples as f32 / report.pool_capacity as f32
                    },
                );
                log_scalar(&mut tb, "selfplay/games", update, report.games as f32);
                log_scalar(&mut tb, "selfplay/samples", update, report.samples as f32);
                log_scalar(&mut tb, "train/swa_count", update, swa_count as f32);
                log_scalar(
                    &mut tb,
                    "selfplay/avg_search_simulations",
                    update,
                    report.avg_search_simulations,
                );
                log_scalar(
                    &mut tb,
                    "train/train_to_selfplay_ratio",
                    update,
                    (config.train_samples_per_update as f32
                        * config.train_epochs_per_update as f32)
                        / config.selfplay_samples_per_update.max(1) as f32,
                );
                log_scalar(
                    &mut tb,
                    "train/fast_sample_rate",
                    update,
                    report.train_fast_sample_rate,
                );
                log_scalar(
                    &mut tb,
                    "train/recent_quota_rate",
                    update,
                    report.train_recent_quota_rate,
                );
                log_scalar(
                    &mut tb,
                    "train/actual_recent_sample_rate",
                    update,
                    report.train_actual_recent_sample_rate,
                );
                log_scalar(
                    &mut tb,
                    "train/policy_weight_mean",
                    update,
                    report.train_policy_weight_mean,
                );
                log_scalar(
                    &mut tb,
                    "train/value_weight_mean",
                    update,
                    report.train_value_weight_mean,
                );
                log_scalar(
                    &mut tb,
                    "selfplay/total_games_generated",
                    update,
                    report.total_games_generated as f32,
                );
                log_scalar(
                    &mut tb,
                    "selfplay/total_samples_generated",
                    update,
                    report.total_samples_generated as f32,
                );
                log_scalar(
                    &mut tb,
                    "replay/chunks",
                    update,
                    report.replay_chunks as f32,
                );
                log_scalar(
                    &mut tb,
                    "replay/oldest_generation_game",
                    update,
                    report.replay_oldest_update as f32,
                );
                log_scalar(
                    &mut tb,
                    "replay/newest_generation_game",
                    update,
                    report.replay_newest_update as f32,
                );
                log_scalar(
                    &mut tb,
                    "replay/avg_generation_game",
                    update,
                    report.replay_avg_update,
                );
                log_scalar(
                    &mut tb,
                    "replay/window_games",
                    update,
                    report.replay_window_games as f32,
                );
                log_scalar(
                    &mut tb,
                    "replay/recent_window_fraction",
                    update,
                    report.replay_recent_window_fraction,
                );
                log_scalar(&mut tb, "selfplay/avg_plies", update, report.avg_plies);
                log_scalar(
                    &mut tb,
                    "selfplay/visit_policy_entropy",
                    update,
                    report.root_visit_entropy,
                );
                log_scalar(
                    &mut tb,
                    "selfplay/visit_policy_entropy_ply_0_89",
                    update,
                    report.entropy_opening,
                );
                log_scalar(
                    &mut tb,
                    "selfplay/visit_policy_entropy_ply_90_plus",
                    update,
                    report.entropy_mid,
                );
                log_scalar(
                    &mut tb,
                    "stats/raw_prior_top1",
                    update,
                    report.raw_prior_top1,
                );
                log_scalar(
                    &mut tb,
                    "stats/raw_prior_top2",
                    update,
                    report.raw_prior_top2,
                );
                log_scalar(
                    &mut tb,
                    "selfplay/visit_policy_top1",
                    update,
                    report.policy_top1,
                );
                log_scalar(
                    &mut tb,
                    "selfplay/visit_policy_top2",
                    update,
                    report.policy_top2,
                );
                log_scalar(&mut tb, "stats/top_q_gap", update, report.root_q_gap);
                log_scalar(
                    &mut tb,
                    "stats/max_child_q_abs",
                    update,
                    report.root_q_top1_abs,
                );
                log_scalar(
                    &mut tb,
                    "stats/visited_actions",
                    update,
                    report.visited_actions,
                );
                log_scalar(
                    &mut tb,
                    "selfplay/raw_prior_top1_ply_0_89",
                    update,
                    report.opening_raw_prior_top1,
                );
                log_scalar(
                    &mut tb,
                    "selfplay/raw_prior_top2_ply_0_89",
                    update,
                    report.opening_raw_prior_top2,
                );
                log_scalar(
                    &mut tb,
                    "selfplay/visit_policy_top1_ply_0_89",
                    update,
                    report.opening_policy_top1,
                );
                log_scalar(
                    &mut tb,
                    "selfplay/visit_policy_top2_ply_0_89",
                    update,
                    report.opening_policy_top2,
                );
                log_scalar(
                    &mut tb,
                    "stats/top_q_gap_ply_0_89",
                    update,
                    report.opening_q_gap,
                );
                log_scalar(
                    &mut tb,
                    "stats/max_child_q_abs_ply_0_89",
                    update,
                    report.opening_q_top1_abs,
                );
                log_scalar(
                    &mut tb,
                    "stats/visited_actions_ply_0_89",
                    update,
                    report.opening_visited_actions,
                );
                log_scalar(
                    &mut tb,
                    "stats/sampled_top_q_rate",
                    update,
                    report.sampled_best_rate,
                );
                log_scalar(
                    &mut tb,
                    "stats/avg_best_played_q_gap",
                    update,
                    report.avg_best_played_q_gap,
                );
                log_scalar(
                    &mut tb,
                    "stats/avg_played_top_visit_ratio",
                    update,
                    report.avg_played_top_visit_ratio,
                );
                log_scalar(&mut tb, "stats/avg_max_child_q", update, report.avg_best_q);
                log_scalar(&mut tb, "stats/avg_played_q", update, report.avg_played_q);
                log_scalar(
                    &mut tb,
                    "terminal/checkmate_no_legal_moves",
                    update,
                    report.terminal_no_legal_moves as f32,
                );
                log_scalar(
                    &mut tb,
                    "terminal/red_general_missing",
                    update,
                    report.terminal_red_general_missing as f32,
                );
                log_scalar(
                    &mut tb,
                    "terminal/black_general_missing",
                    update,
                    report.terminal_black_general_missing as f32,
                );
                log_scalar(
                    &mut tb,
                    "terminal/rule_draw",
                    update,
                    report.terminal_rule_draw as f32,
                );
                log_scalar(
                    &mut tb,
                    "terminal/rule_draw_halfmove120",
                    update,
                    report.terminal_rule_draw_halfmove120 as f32,
                );
                log_scalar(
                    &mut tb,
                    "terminal/rule_draw_repetition",
                    update,
                    report.terminal_rule_draw_repetition as f32,
                );
                log_scalar(
                    &mut tb,
                    "terminal/rule_draw_mutual_long_check",
                    update,
                    report.terminal_rule_draw_mutual_long_check as f32,
                );
                log_scalar(
                    &mut tb,
                    "terminal/rule_draw_mutual_long_chase",
                    update,
                    report.terminal_rule_draw_mutual_long_chase as f32,
                );
                log_scalar(
                    &mut tb,
                    "terminal/rule_win_red",
                    update,
                    report.terminal_rule_win_red as f32,
                );
                log_scalar(
                    &mut tb,
                    "terminal/rule_win_black",
                    update,
                    report.terminal_rule_win_black as f32,
                );
                log_scalar(
                    &mut tb,
                    "terminal/resign_red",
                    update,
                    report.terminal_resign_red as f32,
                );
                log_scalar(
                    &mut tb,
                    "terminal/resign_black",
                    update,
                    report.terminal_resign_black as f32,
                );
                log_scalar(
                    &mut tb,
                    "terminal/max_plies",
                    update,
                    report.terminal_max_plies as f32,
                );
                let updated_numa_models = build_numa_model_replicas(&deployed_model, &numa_nodes);
                {
                    let mut shared = shared_model
                        .write()
                        .unwrap_or_else(|_| panic!("shared selfplay model poisoned"));
                    shared.models_by_numa_node = updated_numa_models;
                    shared.version = shared.version.wrapping_add(1);
                }
                if config.arena_interval > 0 && update.is_multiple_of(config.arena_interval) {
                    {
                        let (pause_lock, _) = &*selfplay_pause;
                        let mut pause_state = pause_lock
                            .lock()
                            .unwrap_or_else(|_| panic!("selfplay pause state poisoned"));
                        pause_state.arena_paused = true;
                    }
                    println!("pause    : selfplay paused for arena");
                    {
                        let (arena_start_positions, arena_mode) =
                            build_arena_start_positions(&config, update);
                        let arena_position_count = arena_start_positions.len();
                        let candidate = Arc::new(deployed_model.clone());
                        let baseline = Arc::new(arena_reference_model.clone());
                        let arena = run_arena_threads(ArenaThreadConfig {
                            candidate,
                            baseline,
                            eval_positions: Arc::new(arena_start_positions),
                            simulations: config.arena_simulations,
                            max_plies: config.max_plies,
                            cpuct: config.arena_cpuct,
                            thread_count: config.arena_processes,
                            seed: config.seed ^ (update as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                        });
                        let ref_elo = arena_best_elo;
                        let candidate_elo = arena.anchored_elo(ref_elo);
                        let elo_diff = arena.elo_diff_vs_even();
                        let arena_se = arena.score_rate_standard_error();
                        let arena_lcb =
                            arena.score_rate_lower_bound(config.arena_promotion_confidence_z);
                        let promoted = arena.promotes_with_lower_bound(
                            config.arena_promotion_rate,
                            config.arena_promotion_confidence_z,
                        );
                        if promoted {
                            arena_reference_model = deployed_model.clone();
                            let best_checkpoint = save_best_checkpoint_model(
                                &deployed_model,
                                &config.model_path,
                                &config.checkpoint_dir,
                                update,
                            );
                            save_model(&deployed_model, &best_path);
                            arena_best_elo = candidate_elo;
                            println!("best     : saved {}", best_checkpoint.display());
                        }
                        println!(
                            "arena {update:04}: mode={} total={} positions={} W/L/D={}/{}/{} red={}/{} black={}/{} score={:.1} rate={:.3} se={:.3} lcb={:.3} promote_at={:.3} z={:.2} ref_elo={:.1} elo={:.1} elo_diff={:+.1} best_ref=memory{}",
                            arena_mode,
                            arena.total_games(),
                            arena_position_count,
                            arena.wins,
                            arena.losses,
                            arena.draws,
                            arena.wins_as_red,
                            arena.losses_as_red,
                            arena.wins_as_black,
                            arena.losses_as_black,
                            arena.score(),
                            arena.score_rate(),
                            arena_se,
                            arena_lcb,
                            config.arena_promotion_rate,
                            config.arena_promotion_confidence_z,
                            ref_elo,
                            candidate_elo,
                            elo_diff,
                            if promoted {
                                " promoted=current saved_best"
                            } else {
                                ""
                            }
                        );
                        log_scalar(&mut tb, "arena/score_rate", update, arena.score_rate());
                        log_scalar(&mut tb, "arena/score_rate_se", update, arena_se);
                        log_scalar(&mut tb, "arena/score_rate_lcb", update, arena_lcb);
                        log_scalar(&mut tb, "arena/ref_elo", update, ref_elo);
                        log_scalar(&mut tb, "arena/elo", update, candidate_elo);
                        log_scalar(&mut tb, "arena/elo_diff", update, elo_diff);
                        log_scalar(
                            &mut tb,
                            "arena/wins_as_red",
                            update,
                            arena.wins_as_red as f32,
                        );
                        log_scalar(
                            &mut tb,
                            "arena/losses_as_red",
                            update,
                            arena.losses_as_red as f32,
                        );
                        log_scalar(
                            &mut tb,
                            "arena/wins_as_black",
                            update,
                            arena.wins_as_black as f32,
                        );
                        log_scalar(
                            &mut tb,
                            "arena/losses_as_black",
                            update,
                            arena.losses_as_black as f32,
                        );
                        log_scalar(
                            &mut tb,
                            "arena/promoted",
                            update,
                            if promoted { 1.0 } else { 0.0 },
                        );
                    }
                    {
                        let (pause_lock, pause_cvar) = &*selfplay_pause;
                        let mut pause_state = pause_lock
                            .lock()
                            .unwrap_or_else(|_| panic!("selfplay pause state poisoned"));
                        pause_state.arena_paused = false;
                        pause_cvar.notify_all();
                    }
                    println!("resume   : selfplay resumed after arena");
                }
                if config.pikafish_label_eval_interval > 0
                    && update.is_multiple_of(config.pikafish_label_eval_interval)
                    && !config.pikafish_label_eval_sqlite.trim().is_empty()
                {
                    let sqlite_path = Path::new(&config.pikafish_label_eval_sqlite);
                    if sqlite_path.exists() {
                        {
                            let (pause_lock, _) = &*selfplay_pause;
                            let mut pause_state = pause_lock
                                .lock()
                                .unwrap_or_else(|_| panic!("selfplay pause state poisoned"));
                            pause_state.arena_paused = true;
                        }
                        println!("pause    : selfplay paused for pikafish label eval");
                        let started = Instant::now();
                        let eval_result = (|| -> io::Result<LabelEvalStats> {
                            let conn = Connection::open(sqlite_path).map_err(sqlite_io_error)?;
                            let rows =
                                load_pikafish_label_rows(&conn, config.pikafish_label_eval_limit)
                                    .map_err(sqlite_io_error)?;
                            evaluate_pikafish_labels_parallel(
                                Arc::new(deployed_model.clone()),
                                rows,
                                config.pikafish_label_eval_simulations,
                                config.seed ^ (update as u64).wrapping_mul(0xD6E8_FD50_19B7_8421),
                                config.pikafish_label_eval_cpuct,
                                config.max_plies,
                                config.arena_processes,
                            )
                        })();
                        {
                            let (pause_lock, pause_cvar) = &*selfplay_pause;
                            let mut pause_state = pause_lock
                                .lock()
                                .unwrap_or_else(|_| panic!("selfplay pause state poisoned"));
                            pause_state.arena_paused = false;
                            pause_cvar.notify_all();
                        }
                        match eval_result {
                            Ok(stats) => {
                                println!(
                                    "pikafish-label {update:04}: sqlite={} evaluated={} legal={} value_labels={} sims={} threads={} search_top1={:.3}% search_top2={:.3}% search_top4={:.3}% search_top8={:.3}% raw_prior_top1={:.3}% value_corr={:.4} value_mae={:.4} elapsed={:.1}s",
                                    config.pikafish_label_eval_sqlite,
                                    stats.count,
                                    stats.legal_bestmove,
                                    stats.value_count(),
                                    config.pikafish_label_eval_simulations,
                                    config.arena_processes,
                                    100.0 * stats.top1_rate(),
                                    100.0 * stats.top2_rate(),
                                    100.0 * stats.top4_rate(),
                                    100.0 * stats.top8_rate(),
                                    100.0 * stats.prior_top1_rate(),
                                    stats.value_corr(),
                                    stats.value_mae_tanh_cp(),
                                    started.elapsed().as_secs_f32()
                                );
                                log_scalar(
                                    &mut tb,
                                    "pikafish_label/evaluated_positions",
                                    update,
                                    stats.count as f32,
                                );
                                log_scalar(
                                    &mut tb,
                                    "pikafish_label/search_top1",
                                    update,
                                    stats.top1_rate(),
                                );
                                log_scalar(
                                    &mut tb,
                                    "pikafish_label/search_top2",
                                    update,
                                    stats.top2_rate(),
                                );
                                log_scalar(
                                    &mut tb,
                                    "pikafish_label/search_top4",
                                    update,
                                    stats.top4_rate(),
                                );
                                log_scalar(
                                    &mut tb,
                                    "pikafish_label/search_top8",
                                    update,
                                    stats.top8_rate(),
                                );
                                log_scalar(
                                    &mut tb,
                                    "pikafish_label/raw_prior_top1",
                                    update,
                                    stats.prior_top1_rate(),
                                );
                                log_scalar(
                                    &mut tb,
                                    "pikafish_label/value_labels",
                                    update,
                                    stats.value_count() as f32,
                                );
                                log_scalar(
                                    &mut tb,
                                    "pikafish_label/value_corr",
                                    update,
                                    stats.value_corr() as f32,
                                );
                                log_scalar(
                                    &mut tb,
                                    "pikafish_label/value_mae_tanh_cp",
                                    update,
                                    stats.value_mae_tanh_cp(),
                                );
                            }
                            Err(err) => {
                                eprintln!(
                                    "pikafish-label {update:04}: failed sqlite={}: {err}",
                                    config.pikafish_label_eval_sqlite
                                );
                            }
                        }
                        println!("resume   : selfplay resumed after pikafish label eval");
                    } else {
                        println!(
                            "pikafish-label {update:04}: skipped missing sqlite={}",
                            config.pikafish_label_eval_sqlite
                        );
                    }
                }
                update = update.saturating_add(1);
                if let Some(target_update) = target_update
                    && update > target_update
                {
                    exited_after_target_update = true;
                    break;
                }
            }
            stop_requested.store(true, Ordering::SeqCst);
            {
                let (pause_lock, pause_cvar) = &*selfplay_pause;
                let mut pause_state = pause_lock
                    .lock()
                    .unwrap_or_else(|_| panic!("selfplay pause state poisoned"));
                pause_state.arena_paused = false;
                pause_state.backlog_paused = false;
                pause_cvar.notify_all();
            }
            for handle in selfplay_handles {
                handle
                    .join()
                    .unwrap_or_else(|_| panic!("selfplay thread panicked"));
            }
            collector_handle
                .join()
                .unwrap_or_else(|_| panic!("selfplay collector thread panicked"));
            trainer_handle
                .join()
                .unwrap_or_else(|_| panic!("training thread panicked"));
            if exited_after_ctrl_c {
                while let Ok(event) = trainer_rx.try_recv() {
                    update_swa_model(&mut swa_model, &mut swa_count, &event.candidate_model);
                    interrupt_save_model = Some(event.candidate_model);
                    interrupt_save_next_update = update.saturating_add(1);
                    update = update.saturating_add(1);
                }
            }
            if exited_after_ctrl_c || exited_after_target_update {
                if let Some(model) = interrupt_save_model.as_ref() {
                    save_model(model, Path::new(&config.model_path));
                    if let Some(average) = swa_model.as_ref() {
                        save_model(average, &swa_path);
                    }
                    save_az_loop_progress_pair(
                        &config_path,
                        interrupt_save_next_update,
                        arena_best_elo,
                        swa_count,
                    );
                    println!(
                        "model    : {} save raw=`{}` swa=`{}` swa_count={} next_update={}",
                        if exited_after_target_update {
                            "target"
                        } else {
                            "interrupt"
                        },
                        config.model_path,
                        swa_path.display(),
                        swa_count,
                        interrupt_save_next_update
                    );
                } else {
                    println!(
                        "model    : no completed update to save on {}",
                        if exited_after_target_update {
                            "target stop"
                        } else {
                            "interrupt"
                        }
                    );
                }
            }
            if !exited_after_ctrl_c {
                let _ = fs::remove_file(&replay_snapshot_path);
            }
        }
        Some(CliCommand::VsPikafish(cmd)) => {
            let pikafish_exe = cmd.pikafish_exe;
            let model_path = cmd.model;
            let simulations = cmd.simulations.unwrap_or(192).max(1);
            let cpuct = cmd.cpuct.max(0.0);
            let cpuct_at_root = cmd.cpuct_at_root.max(0.0);
            let max_plies = cmd.max_plies.max(1);
            let pikafish_depth = cmd.pikafish_depth.max(1);
            let games = cmd.games.max(1);
            let parallel_games = cmd.parallel_games.max(1);
            let (opening_plies_min, opening_plies_max) =
                if cmd.opening_plies_min <= cmd.opening_plies_max {
                    (cmd.opening_plies_min, cmd.opening_plies_max)
                } else {
                    (cmd.opening_plies_max, cmd.opening_plies_min)
                };
            let (start_positions, opening_mode) = if cmd.opening_book.trim().is_empty() {
                (Vec::new(), "startpos_fallback".to_string())
            } else {
                let book = ObkBook::load(&cmd.opening_book).unwrap_or_else(|err| {
                    panic!(
                        "failed to load vs-pikafish opening book `{}`: {err}",
                        cmd.opening_book
                    )
                });
                let mut rng = SplitMix64::new(cmd.seed ^ 0xA24B_AED4_963E_E407);
                let count = cmd.opening_positions.max(1);
                let mut positions = Vec::with_capacity(count);
                for _ in 0..count {
                    positions.push(book.random_prefix_position(
                        opening_plies_min,
                        opening_plies_max,
                        &mut rng,
                    ));
                }
                (
                    positions,
                    format!(
                        "obk_openings(book={},keys={},moves={},plies={}-{})",
                        cmd.opening_book,
                        book.key_count(),
                        book.move_count(),
                        opening_plies_min,
                        opening_plies_max
                    ),
                )
            };
            let summary = run_vs_pikafish(
                Path::new(&pikafish_exe),
                Path::new(&model_path),
                &start_positions,
                VsPikafishConfig {
                    pikafish_depth,
                    total_games: games,
                    max_plies,
                    simulations,
                    seed: cmd.seed,
                    parallel_games,
                    cpuct,
                    cpuct_at_root,
                },
            )
            .unwrap_or_else(|err| panic!("vs-pikafish failed: {err}"));
            for item in &summary.abnormal_ends {
                println!(
                    "vs-pikafish-final: game={} chinese={} end={} final_fen=\"{}\" {}",
                    item.game_index,
                    if item.chinese_plays_red {
                        "red"
                    } else {
                        "black"
                    },
                    item.end,
                    item.final_fen,
                    item.position_command
                );
            }
            println!(
                "vs-pikafish: model={} search=alphazero games={} fens={} opening={} parallel={} chinese W/L/D={}/{}/{} (as_red={} as_black={}) win_reasons(general_capture={} checkmate_no_legal_moves={} rule={} pikafish_no_bestmove={} pikafish_invalid_move={} pikafish_illegal_move={}) | pikafish_depth={} max_plies={} sims={} cpuct={} cpuct_at_root={}",
                model_path,
                summary.total_games,
                start_positions.len(),
                opening_mode,
                parallel_games.min(games),
                summary.chinese_wins,
                summary.chinese_losses,
                summary.draws,
                summary.chinese_wins_as_red,
                summary.chinese_wins_as_black,
                summary.chinese_win_by_general_capture,
                summary.chinese_win_by_no_legal_moves,
                summary.chinese_win_by_rule,
                summary.chinese_win_by_pikafish_no_bestmove,
                summary.chinese_win_by_pikafish_invalid_move,
                summary.chinese_win_by_pikafish_illegal_move,
                pikafish_depth,
                max_plies,
                simulations,
                cpuct,
                cpuct_at_root
            );
        }
        Some(CliCommand::PikafishLabelRandom(cmd)) => {
            run_pikafish_label_random(cmd)
                .unwrap_or_else(|err| panic!("pikafish-label-random failed: {err}"));
        }
        Some(CliCommand::PikafishLabelEval(cmd)) => {
            run_pikafish_label_eval(cmd)
                .unwrap_or_else(|err| panic!("pikafish-label-eval failed: {err}"));
        }
    };
    chineseai::profile::print_report();
}

#[derive(Clone, Debug)]
struct PikafishLabelRow {
    id: i64,
    fen: String,
    bestmove: String,
    best_score_cp: Option<i32>,
}

#[derive(Default)]
struct LabelEvalStats {
    count: usize,
    legal_bestmove: usize,
    top1_hits: usize,
    top2_hits: usize,
    top4_hits: usize,
    top8_hits: usize,
    prior_top1_hits: usize,
    value_pairs: usize,
    value_q_sum: f64,
    cp_tanh_sum: f64,
    value_q_sq_sum: f64,
    cp_tanh_sq_sum: f64,
    value_cp_cross_sum: f64,
    abs_value_error_sum: f64,
}

impl LabelEvalStats {
    fn merge(&mut self, other: LabelEvalStats) {
        self.count += other.count;
        self.legal_bestmove += other.legal_bestmove;
        self.top1_hits += other.top1_hits;
        self.top2_hits += other.top2_hits;
        self.top4_hits += other.top4_hits;
        self.top8_hits += other.top8_hits;
        self.prior_top1_hits += other.prior_top1_hits;
        self.value_pairs += other.value_pairs;
        self.value_q_sum += other.value_q_sum;
        self.cp_tanh_sum += other.cp_tanh_sum;
        self.value_q_sq_sum += other.value_q_sq_sum;
        self.cp_tanh_sq_sum += other.cp_tanh_sq_sum;
        self.value_cp_cross_sum += other.value_cp_cross_sum;
        self.abs_value_error_sum += other.abs_value_error_sum;
    }

    fn denom(&self) -> f32 {
        self.count.max(1) as f32
    }

    fn top1_rate(&self) -> f32 {
        self.top1_hits as f32 / self.denom()
    }

    fn top2_rate(&self) -> f32 {
        self.top2_hits as f32 / self.denom()
    }

    fn top4_rate(&self) -> f32 {
        self.top4_hits as f32 / self.denom()
    }

    fn top8_rate(&self) -> f32 {
        self.top8_hits as f32 / self.denom()
    }

    fn prior_top1_rate(&self) -> f32 {
        self.prior_top1_hits as f32 / self.denom()
    }

    fn value_mae_tanh_cp(&self) -> f32 {
        (self.abs_value_error_sum / self.value_count().max(1) as f64) as f32
    }

    fn push_value_pair(&mut self, value_q: f32, score_cp: Option<i32>) {
        let Some(score_cp) = score_cp else {
            return;
        };
        let target = ((score_cp as f64) / 600.0).tanh();
        let value = value_q as f64;
        self.value_pairs += 1;
        self.value_q_sum += value;
        self.cp_tanh_sum += target;
        self.value_q_sq_sum += value * value;
        self.cp_tanh_sq_sum += target * target;
        self.value_cp_cross_sum += value * target;
        self.abs_value_error_sum += (value - target).abs();
    }

    fn value_count(&self) -> usize {
        self.value_pairs
    }

    fn value_corr(&self) -> f64 {
        let n = self.value_count() as f64;
        if n <= 1.0 {
            return 0.0;
        }
        let cov = self.value_cp_cross_sum - self.value_q_sum * self.cp_tanh_sum / n;
        let left = self.value_q_sq_sum - self.value_q_sum * self.value_q_sum / n;
        let right = self.cp_tanh_sq_sum - self.cp_tanh_sum * self.cp_tanh_sum / n;
        if left <= 0.0 || right <= 0.0 {
            0.0
        } else {
            cov / (left * right).sqrt()
        }
    }
}

fn run_pikafish_label_eval(cmd: PikafishLabelEvalArgs) -> io::Result<()> {
    let model = AzNnue::load(&cmd.model).map_err(|err| {
        io::Error::new(
            err.kind(),
            format!("failed to load model `{}`: {err}", cmd.model),
        )
    })?;
    let conn = Connection::open(&cmd.sqlite).map_err(sqlite_io_error)?;
    let rows = load_pikafish_label_rows(&conn, cmd.limit).map_err(sqlite_io_error)?;
    if rows.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("no labels in {}", cmd.sqlite),
        ));
    }

    let started = Instant::now();
    let stats = evaluate_pikafish_labels(
        &model,
        &rows,
        cmd.simulations.max(1),
        cmd.seed,
        cmd.cpuct.max(0.0),
        cmd.max_depth,
        |done, total| {
            if done % 100 == 0 || done == total {
                println!("pikafish-label-eval: searched {done}/{total}");
            }
        },
    )?;

    println!(
        "pikafish-label-eval: model={} sqlite={} evaluated={} legal_labels={} value_labels={} sims={} search_top1={:.3}% search_top2={:.3}% search_top4={:.3}% search_top8={:.3}% raw_prior_top1={:.3}% value_corr={:.4} value_mae_tanh_cp={:.4} elapsed={:.1}s",
        cmd.model,
        cmd.sqlite,
        stats.count,
        stats.legal_bestmove,
        stats.value_count(),
        cmd.simulations.max(1),
        100.0 * stats.top1_rate(),
        100.0 * stats.top2_rate(),
        100.0 * stats.top4_rate(),
        100.0 * stats.top8_rate(),
        100.0 * stats.prior_top1_rate(),
        stats.value_corr(),
        stats.value_mae_tanh_cp(),
        started.elapsed().as_secs_f32()
    );
    Ok(())
}

fn evaluate_pikafish_labels(
    model: &AzNnue,
    rows: &[PikafishLabelRow],
    simulations: usize,
    seed: u64,
    cpuct: f32,
    max_depth: usize,
    mut progress: impl FnMut(usize, usize),
) -> io::Result<LabelEvalStats> {
    let mut stats = LabelEvalStats::default();
    for (offset, row) in rows.iter().enumerate() {
        let position = Position::from_fen(&row.fen).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid FEN id={}: {err}", row.id),
            )
        })?;
        let Some(label_move) = position.parse_uci_move(&row.bestmove) else {
            continue;
        };
        stats.legal_bestmove += 1;
        let result = alphazero_search(
            &position,
            model,
            fixed_az_search_limits(simulations.max(1), seed ^ row.id as u64, cpuct, max_depth),
        );
        stats.count += 1;
        if result.best_move == Some(label_move) {
            stats.top1_hits += 1;
        }
        let mut by_visits = result.candidates.clone();
        by_visits.sort_by(|left, right| {
            right
                .visits
                .cmp(&left.visits)
                .then_with(|| right.policy.total_cmp(&left.policy))
        });
        if by_visits
            .iter()
            .take(2)
            .any(|candidate| candidate.mv == label_move)
        {
            stats.top2_hits += 1;
        }
        if by_visits
            .iter()
            .take(4)
            .any(|candidate| candidate.mv == label_move)
        {
            stats.top4_hits += 1;
        }
        if by_visits
            .iter()
            .take(8)
            .any(|candidate| candidate.mv == label_move)
        {
            stats.top8_hits += 1;
        }
        if result
            .candidates
            .iter()
            .max_by(|left, right| left.raw_prior.total_cmp(&right.raw_prior))
            .is_some_and(|candidate| candidate.mv == label_move)
        {
            stats.prior_top1_hits += 1;
        }
        stats.push_value_pair(result.value_q, row.best_score_cp);
        progress(offset + 1, rows.len());
    }
    Ok(stats)
}

fn evaluate_pikafish_labels_parallel(
    model: Arc<AzNnue>,
    rows: Vec<PikafishLabelRow>,
    simulations: usize,
    seed: u64,
    cpuct: f32,
    max_depth: usize,
    thread_count: usize,
) -> io::Result<LabelEvalStats> {
    if rows.is_empty() {
        return Ok(LabelEvalStats::default());
    }
    let thread_count = thread_count.max(1).min(rows.len());
    let rows = Arc::new(rows);
    let mut handles = Vec::with_capacity(thread_count);
    for thread_id in 0..thread_count {
        let model = Arc::clone(&model);
        let rows = Arc::clone(&rows);
        handles.push(thread::spawn(move || {
            let shard: Vec<_> = rows
                .iter()
                .enumerate()
                .filter(|(index, _)| index % thread_count == thread_id)
                .map(|(_, row)| row.clone())
                .collect();
            evaluate_pikafish_labels(
                &model,
                &shard,
                simulations,
                seed ^ (thread_id as u64).wrapping_mul(0x517C_C1B7_2722_0A95),
                cpuct,
                max_depth,
                |_, _| {},
            )
        }));
    }

    let mut merged = LabelEvalStats::default();
    for handle in handles {
        let stats = handle
            .join()
            .map_err(|_| io::Error::other("pikafish label eval thread panicked"))??;
        merged.merge(stats);
    }
    Ok(merged)
}

fn load_pikafish_label_rows(
    conn: &Connection,
    limit: usize,
) -> rusqlite::Result<Vec<PikafishLabelRow>> {
    let mut query =
        "SELECT id, fen, bestmove, best_score_cp FROM pikafish_labels ORDER BY id".to_string();
    if limit > 0 {
        query.push_str(" LIMIT ?1");
        let mut stmt = conn.prepare(&query)?;
        stmt.query_map(params![limit as i64], |row| {
            Ok(PikafishLabelRow {
                id: row.get(0)?,
                fen: row.get(1)?,
                bestmove: row.get(2)?,
                best_score_cp: row.get(3)?,
            })
        })?
        .collect()
    } else {
        let mut stmt = conn.prepare(&query)?;
        stmt.query_map([], |row| {
            Ok(PikafishLabelRow {
                id: row.get(0)?,
                fen: row.get(1)?,
                bestmove: row.get(2)?,
                best_score_cp: row.get(3)?,
            })
        })?
        .collect()
    }
}

#[derive(Clone, Debug, Default)]
struct PikafishPv {
    multipv: usize,
    score_cp: Option<i32>,
    mate: Option<i32>,
    moves: Vec<String>,
}

struct PikafishLabelUci {
    child: Child,
    stdin: BufWriter<std::process::ChildStdin>,
    stdout: BufReader<std::process::ChildStdout>,
}

impl PikafishLabelUci {
    fn spawn(exe: &Path) -> io::Result<Self> {
        let mut child = Command::new(exe)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let stdin = BufWriter::new(
            child
                .stdin
                .take()
                .ok_or_else(|| io::Error::other("pikafish: missing stdin"))?,
        );
        let stdout = BufReader::new(
            child
                .stdout
                .take()
                .ok_or_else(|| io::Error::other("pikafish: missing stdout"))?,
        );
        let mut out = Self {
            child,
            stdin,
            stdout,
        };
        out.handshake()?;
        Ok(out)
    }

    fn write_line(&mut self, line: &str) -> io::Result<()> {
        writeln!(self.stdin, "{line}")?;
        self.stdin.flush()
    }

    fn read_line_into(&mut self, buf: &mut String) -> io::Result<usize> {
        buf.clear();
        self.stdout.read_line(buf)
    }

    fn handshake(&mut self) -> io::Result<()> {
        self.write_line("uci")?;
        self.wait_for("uciok")?;
        self.write_line("isready")?;
        self.wait_for("readyok")
    }

    fn wait_for(&mut self, token: &str) -> io::Result<()> {
        let mut buf = String::new();
        loop {
            if self.read_line_into(&mut buf)? == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("pikafish: EOF before {token}"),
                ));
            }
            if buf.trim() == token {
                return Ok(());
            }
        }
    }

    fn query(&mut self, fen: &str, depth: u32) -> io::Result<(String, Vec<PikafishPv>)> {
        self.write_line(&format!("position fen {fen}"))?;
        self.write_line(&format!("go depth {}", depth.max(1)))?;
        let mut buf = String::new();
        let mut pvs = Vec::<PikafishPv>::new();
        loop {
            if self.read_line_into(&mut buf)? == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "pikafish: EOF before bestmove",
                ));
            }
            let line = buf.trim();
            if let Some(rest) = line.strip_prefix("bestmove ") {
                let bestmove = rest.split_whitespace().next().unwrap_or("").to_string();
                pvs.sort_by_key(|pv| pv.multipv);
                return Ok((bestmove, pvs));
            }
            if let Some(pv) = parse_pikafish_info_pv(line) {
                if let Some(slot) = pvs.iter_mut().find(|old| old.multipv == pv.multipv) {
                    *slot = pv;
                } else {
                    pvs.push(pv);
                }
            }
        }
    }

    fn quit(&mut self) {
        let _ = self.write_line("quit");
        let _ = self.child.wait();
    }
}

impl Drop for PikafishLabelUci {
    fn drop(&mut self) {
        self.quit();
    }
}

fn parse_pikafish_info_pv(line: &str) -> Option<PikafishPv> {
    if !line.starts_with("info ") || !line.contains(" pv ") {
        return None;
    }
    let parts: Vec<&str> = line.split_whitespace().collect();
    let mut multipv = 1usize;
    let mut score_cp = None;
    let mut mate = None;
    let mut moves = Vec::new();
    let mut i = 0usize;
    while i < parts.len() {
        match parts[i] {
            "multipv" if i + 1 < parts.len() => {
                multipv = parts[i + 1].parse().ok()?;
                i += 2;
            }
            "score" if i + 2 < parts.len() => {
                match parts[i + 1] {
                    "cp" => score_cp = parts[i + 2].parse().ok(),
                    "mate" => mate = parts[i + 2].parse().ok(),
                    _ => {}
                }
                i += 3;
            }
            "pv" => {
                moves.extend(parts[i + 1..].iter().map(|item| (*item).to_string()));
                break;
            }
            _ => i += 1,
        }
    }
    (!moves.is_empty()).then_some(PikafishPv {
        multipv,
        score_cp,
        mate,
        moves,
    })
}

fn run_pikafish_label_random(cmd: PikafishLabelRandomArgs) -> io::Result<()> {
    let fens_path = Path::new(&cmd.fens);
    let output_path = Path::new(&cmd.output);
    let sqlite_path = Path::new(&cmd.sqlite);
    if cmd.regenerate || !fens_path.exists() {
        let fens = generate_random_eval_fens(
            cmd.count.max(1),
            cmd.min_plies.min(cmd.max_plies),
            cmd.min_plies.max(cmd.max_plies),
            cmd.seed,
        );
        if let Some(parent) = fens_path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        fs::write(fens_path, format!("{}\n", fens.join("\n")))?;
        println!(
            "pikafish-label-random: generated {} fens -> {}",
            fens.len(),
            fens_path.display()
        );
    }

    let fens_text = fs::read_to_string(fens_path)?;
    let fens: Vec<String> = fens_text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_string)
        .collect();
    if fens.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("no FENs in {}", fens_path.display()),
        ));
    }

    if let Some(parent) = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = sqlite_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let mut conn = Connection::open(sqlite_path).map_err(sqlite_io_error)?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS pikafish_labels (
            id INTEGER PRIMARY KEY,
            fen TEXT NOT NULL UNIQUE,
            side_to_move TEXT NOT NULL,
            depth INTEGER NOT NULL,
            bestmove TEXT NOT NULL,
            best_score_cp INTEGER,
            best_mate INTEGER,
            best_pv TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_pikafish_labels_bestmove ON pikafish_labels(bestmove);",
    )
    .map_err(sqlite_io_error)?;
    migrate_pikafish_label_schema(&conn).map_err(sqlite_io_error)?;
    let tx = conn.transaction().map_err(sqlite_io_error)?;
    let mut writer = BufWriter::new(fs::File::create(output_path)?);
    writeln!(
        writer,
        "index,fen,side_to_move,depth,bestmove,best_score_cp,best_mate,best_pv"
    )?;

    let mut engine = PikafishLabelUci::spawn(Path::new(&cmd.pikafish_exe))?;
    for (index, fen) in fens.iter().enumerate() {
        let position = Position::from_fen(fen).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid FEN at {}: {err}", index + 1),
            )
        })?;
        let (bestmove, pvs) = engine.query(fen, cmd.depth.max(1))?;
        let best = pvs.iter().find(|pv| pv.multipv == 1);
        let best_pv = best.map(|pv| pv.moves.join(" ")).unwrap_or_default();
        let side_to_move = match position.side_to_move() {
            chineseai::xiangqi::Color::Red => "w",
            chineseai::xiangqi::Color::Black => "b",
        };
        write!(
            writer,
            "{},{},{},{},{},{},{},{}",
            index,
            csv_escape(fen),
            side_to_move,
            cmd.depth.max(1),
            csv_escape(&bestmove),
            best.and_then(|pv| pv.score_cp)
                .map(|v| v.to_string())
                .unwrap_or_default(),
            best.and_then(|pv| pv.mate)
                .map(|v| v.to_string())
                .unwrap_or_default(),
            csv_escape(&best_pv)
        )?;
        writeln!(writer)?;
        tx.execute(
            "INSERT INTO pikafish_labels (
                id, fen, side_to_move, depth, bestmove, best_score_cp, best_mate, best_pv, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, CURRENT_TIMESTAMP)
            ON CONFLICT(fen) DO UPDATE SET
                side_to_move=excluded.side_to_move,
                depth=excluded.depth,
                bestmove=excluded.bestmove,
                best_score_cp=excluded.best_score_cp,
                best_mate=excluded.best_mate,
                best_pv=excluded.best_pv,
                updated_at=CURRENT_TIMESTAMP",
            params![
                index as i64,
                fen,
                side_to_move,
                cmd.depth.max(1) as i64,
                bestmove,
                best.and_then(|pv| pv.score_cp).map(i64::from),
                best.and_then(|pv| pv.mate).map(i64::from),
                best_pv,
            ],
        )
        .map_err(sqlite_io_error)?;
        if (index + 1) % 100 == 0 || index + 1 == fens.len() {
            println!(
                "pikafish-label-random: labeled {}/{} -> {} {}",
                index + 1,
                fens.len(),
                output_path.display(),
                sqlite_path.display()
            );
        }
    }
    writer.flush()?;
    tx.commit().map_err(sqlite_io_error)?;
    Ok(())
}

fn sqlite_io_error(err: rusqlite::Error) -> io::Error {
    io::Error::other(err.to_string())
}

#[cfg(test)]
mod reporting_tests {
    use super::*;
    use chineseai::az::AzSampleMeta;

    fn reporting_sample(generation: u32, policy: Vec<f32>) -> AzTrainingSample {
        AzTrainingSample {
            features: vec![0],
            move_indices: (0..policy.len()).collect(),
            policy,
            value_wdl: [0.0, 1.0, 0.0],
            value: 0.0,
            side_sign: 1.0,
            moves_left: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            search_simulations: 2_000,
            meta: AzSampleMeta {
                generation_update: generation,
                ..AzSampleMeta::default()
            },
        }
    }

    #[test]
    fn train_source_reports_actual_recent_and_real_target_shape() {
        let samples = vec![
            reporting_sample(10, vec![3.0, 1.0]),
            reporting_sample(5, vec![2.0, 2.0]),
        ];
        let stats = train_batch_source_stats(&samples, 4_000, 1, 10, 3);

        assert!((stats.recent_quota_rate - 0.5).abs() < 1e-6);
        assert!((stats.actual_recent_sample_rate - 0.5).abs() < 1e-6);
        let expected_entropy =
            (-(0.75f32 * 0.75f32.ln() + 0.25f32 * 0.25f32.ln()) - 0.5f32.ln()) / 2.0;
        assert!((stats.policy_target_entropy - expected_entropy).abs() < 1e-6);
        assert!((stats.policy_target_top1 - 0.625).abs() < 1e-6);
        assert!((stats.policy_target_top2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn pikafish_value_metrics_count_only_rows_with_scores() {
        let mut stats = LabelEvalStats {
            count: 2,
            ..LabelEvalStats::default()
        };
        stats.push_value_pair(0.25, None);
        stats.push_value_pair(0.25, Some(0));

        assert_eq!(stats.value_count(), 1);
        assert!((stats.value_mae_tanh_cp() - 0.25).abs() < 1e-6);
    }
}

fn migrate_pikafish_label_schema(conn: &Connection) -> rusqlite::Result<()> {
    let mut stmt = conn.prepare("PRAGMA table_info(pikafish_labels)")?;
    let columns = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<Result<Vec<_>, _>>()?;
    if !columns.iter().any(|column| column == "multipv") {
        return Ok(());
    }
    conn.execute_batch(
        "ALTER TABLE pikafish_labels RENAME TO pikafish_labels_old;
        CREATE TABLE pikafish_labels (
            id INTEGER PRIMARY KEY,
            fen TEXT NOT NULL UNIQUE,
            side_to_move TEXT NOT NULL,
            depth INTEGER NOT NULL,
            bestmove TEXT NOT NULL,
            best_score_cp INTEGER,
            best_mate INTEGER,
            best_pv TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        INSERT OR REPLACE INTO pikafish_labels (
            id, fen, side_to_move, depth, bestmove, best_score_cp, best_mate, best_pv, updated_at
        )
        SELECT id, fen, side_to_move, depth, bestmove, best_score_cp, best_mate, best_pv, updated_at
        FROM pikafish_labels_old;
        DROP TABLE pikafish_labels_old;
        CREATE INDEX IF NOT EXISTS idx_pikafish_labels_bestmove ON pikafish_labels(bestmove);",
    )
}

fn generate_random_eval_fens(
    count: usize,
    min_plies: usize,
    max_plies: usize,
    seed: u64,
) -> Vec<String> {
    let mut rng = SplitMix64::new(seed);
    let mut seen = HashSet::with_capacity(count * 2);
    let mut out = Vec::with_capacity(count);
    let mut attempts = 0usize;
    let max_attempts = count.saturating_mul(200).max(10_000);
    while out.len() < count && attempts < max_attempts {
        attempts += 1;
        let span = max_plies.saturating_sub(min_plies);
        let target_plies = min_plies + (rng.next_u64() as usize % (span + 1));
        if let Some(fen) = random_position_fen(target_plies, &mut rng)
            && seen.insert(fen.clone())
        {
            out.push(fen);
        }
    }
    if out.len() < count {
        panic!(
            "only generated {} unique random FENs after {} attempts",
            out.len(),
            attempts
        );
    }
    out
}

fn random_position_fen(target_plies: usize, rng: &mut SplitMix64) -> Option<String> {
    let mut position = Position::startpos();
    let mut rule_history = position.initial_rule_history();
    for _ in 0..target_plies {
        if position.rule_outcome_with_history(&rule_history).is_some() {
            return None;
        }
        let legal = position.legal_moves_with_rules(&rule_history);
        if legal.is_empty() {
            return None;
        }
        let mv: Move = legal[(rng.next_u64() as usize) % legal.len()];
        rule_history.push(position.rule_history_entry_after_move(mv));
        position.make_move(mv);
    }
    match position.rule_outcome_with_history(&rule_history) {
        Some(RuleOutcome::Draw(_) | RuleOutcome::Win(_)) => None,
        None if position.legal_moves_with_rules(&rule_history).is_empty() => None,
        None => Some(position.to_fen()),
    }
}

fn csv_escape(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
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
