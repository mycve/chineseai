#[cfg(all(target_os = "linux", not(target_env = "musl")))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

mod az_loop_config;

use az_loop_config::{AzLoopFileConfig, DEFAULT_AZ_LOOP_CONFIG, load_or_create_az_loop_config};

use chineseai::{
    az::{
        AzArenaConfig, AzArenaReport, AzExperiencePool, AzLoopConfig, AzLoopReport, AzNnue,
        AzSearchLimits, AzSelfplayData, AzTrainLossWeights,
        AzTrainingSample, SplitMix64, alphazero_search, alphazero_search_with_history_and_rules,
        benchmark_fixed_policy_fit, benchmark_fixed_policy_fit_with_trace, benchmark_policy_fit,
        benchmark_training, generate_selfplay_data, global_training_step_sample_count, lc0_cp_from_q,
        play_arena_games_from_positions, train_samples_weighted,
    },
    pikafish_match::{VsPikafishConfig, run_vs_pikafish},
    uci::run_uci,
    xiangqi::{Move, Position},
};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs, io,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::{
        Arc, Condvar, Mutex, RwLock,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};
use tensorboard_rs::summary_writer::SummaryWriter;

const DEFAULT_ARENA_EVAL_FENS: &str = "eval_fens.txt";
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
    /// Diagnose policy/search/value on one position.
    AzDiagnose(AzDiagnoseArgs),
    /// Benchmark fixed-position search speed.
    AzBench(AzBenchArgs),
    /// Benchmark a synthetic training workload.
    AzTrainBench(AzTrainBenchArgs),
    /// Benchmark fixed teacher-policy fitting on generated positions.
    AzPolicyFitBench(AzPolicyFitBenchArgs),
    /// Generate fixed self-play data and benchmark policy fitting on it.
    AzSelfplayFitBench(AzSelfplayFitBenchArgs),
    /// Generate a fixed self-play replay dataset once.
    AzReplayGenerateFixed(AzReplayGenerateFixedArgs),
    /// Train and evaluate against a fixed replay dataset without self-play.
    AzTrainFixedReplay(AzTrainFixedReplayArgs),
    /// Run self-play training from a TOML config.
    AzLoop(AzLoopArgs),
    /// Create and run the local 100-update baseline recipe.
    #[command(name = "az-baseline-100")]
    AzBaseline100(AzBaseline100Args),
    /// Label FENs with Pikafish bestmove/value for teacher probes.
    #[command(name = "pikafish-label")]
    PikafishLabel(PikafishLabelArgs),
    /// Compare an AZ-NNUE model against a Pikafish teacher-label TSV.
    #[command(name = "az-teacher-probe")]
    AzTeacherProbe(AzTeacherProbeArgs),
    /// Collect self-play/search FENs for tactical teacher labeling.
    #[command(name = "az-collect-fens")]
    AzCollectFens(AzCollectFensArgs),
    /// Run ChineseAI against a Pikafish UCI engine.
    VsPikafish(VsPikafishArgs),
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
  chineseai az-search model.safetensors 50000 0.65 --cpuct-at-root 2.53 startpos
  chineseai az-search model.safetensors 10000 0.65 --cpuct-at-root 2.53 startpos")]
struct AzSearchArgs {
    /// AZ-NNUE model path.
    model: String,
    /// Number of MCTS simulations.
    #[arg(default_value_t = 10_000)]
    simulations: usize,
    /// Non-root PUCT init.
    #[arg(default_value_t = 0.65)]
    cpuct: f32,
    /// Root PUCT init.
    #[arg(long, default_value_t = 2.53)]
    cpuct_at_root: f32,
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
    #[arg(long, default_value_t = true)]
    moves_left_utility: bool,
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
struct AzBaseline100Args {
    /// Baseline config path to create/use.
    #[arg(long, default_value = "baseline-h128-u100.azloop.toml")]
    config: String,
    /// Output model path in the generated config.
    #[arg(long, default_value = "baseline-h128-u100.safetensors")]
    model: String,
    /// Absolute update number to stop at.
    #[arg(long, default_value_t = 100)]
    target_update: usize,
    /// Self-play worker threads.
    #[arg(long, default_value_t = 30)]
    workers: usize,
    /// Hidden size for the baseline model.
    #[arg(long, default_value_t = 128)]
    hidden: usize,
    /// Visible CUDA device index for the child az-loop process.
    #[arg(long, default_value = "0")]
    gpu: String,
    /// Only create the config; do not start training.
    #[arg(long)]
    no_run: bool,
}

#[derive(Args, Debug)]
#[command(after_long_help = "\
Examples:
  chineseai pikafish-label ./tools/pikafish.exe hard_fens.txt --output hard_teacher.tsv --movetime 5000 --threads 4 --multipv 4")]
struct PikafishLabelArgs {
    /// Pikafish UCI executable path.
    pikafish_exe: String,
    /// Input FEN list. Blank lines and lines starting with # are ignored.
    input: String,
    /// Output TSV path. Defaults to stdout.
    #[arg(long)]
    output: Option<String>,
    /// Time per position in milliseconds.
    #[arg(long, default_value_t = 5000)]
    movetime: u64,
    /// Pikafish threads.
    #[arg(long, default_value_t = 4)]
    threads: usize,
    /// Number of MultiPV lines to request.
    #[arg(long, default_value_t = 4)]
    multipv: usize,
}

#[derive(Args, Debug)]
#[command(after_long_help = "\
Examples:
  chineseai az-teacher-probe pure-canonical-h128-u100.safetensors hard_teacher.tsv 1200 1.5")]
struct AzTeacherProbeArgs {
    /// AZ-NNUE model path.
    model: String,
    /// Teacher TSV generated by pikafish-label.
    teacher: String,
    /// Number of MCTS simulations.
    #[arg(default_value_t = 1200)]
    simulations: usize,
    /// PUCT constant for AlphaZero search.
    #[arg(default_value_t = 1.5)]
    cpuct: f32,
}

#[derive(Args, Debug, Clone)]
#[command(after_long_help = "\
Examples:
  chineseai az-collect-fens ./tools/pikafish tactical_teacher.tsv --positions 2000 --random-prefix-min 20 --random-prefix-max 80 --takeover-plies 2 --movetime 1000 --threads 4 --multipv 4")]
struct AzCollectFensArgs {
    /// Pikafish UCI executable path.
    pikafish_exe: String,
    /// Output teacher TSV.
    #[arg(default_value = "tactical_teacher.tsv")]
    output: String,
    /// Target unique FEN count.
    #[arg(long, default_value_t = 1000)]
    positions: usize,
    /// Random legal-move prefix lower bound.
    #[arg(long, default_value_t = 20)]
    random_prefix_min: usize,
    /// Random legal-move prefix upper bound.
    #[arg(long, default_value_t = 80)]
    random_prefix_max: usize,
    /// Pikafish plies after the random prefix before labeling the final FEN.
    #[arg(long, default_value_t = 2)]
    takeover_plies: usize,
    /// Time per Pikafish query in milliseconds.
    #[arg(long, default_value_t = 1000)]
    movetime: u64,
    /// Pikafish threads.
    #[arg(long, default_value_t = 4)]
    threads: usize,
    /// Parallel Pikafish worker processes.
    #[arg(long, default_value_t = 4)]
    workers: usize,
    /// Number of MultiPV lines to request.
    #[arg(long, default_value_t = 4)]
    multipv: usize,
    /// Maximum attempts before stopping. Defaults to positions * 50.
    #[arg(long, default_value_t = 0)]
    max_attempts: usize,
    /// Random seed.
    #[arg(long, default_value_t = 20260527)]
    seed: u64,
}

#[derive(Args, Debug)]
#[command(after_long_help = "\
Examples:
  chineseai vs-pikafish ./tools/pikafish model.safetensors
  chineseai vs-pikafish ./tools/pikafish checkpoints/update-0620-model.safetensors --simulations 192
  chineseai vs-pikafish ./tools/pikafish model.safetensors --pikafish-depth 10 --games 40 --parallel-games 5 --eval-fens eval_fens.txt")]
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
    /// Start-position file.
    #[arg(long = "eval-fens", default_value = DEFAULT_ARENA_EVAL_FENS)]
    eval_fens_path: String,
}

#[derive(Args, Debug)]
#[command(after_long_help = "\
Examples:
  chineseai az-diagnose model.safetensors 512 \"5k3/9/9/9/9/9/4r4/4R4/4R1r2/4K4 w - - 0 1\"")]
struct AzDiagnoseArgs {
    /// AZ-NNUE model path.
    model: String,
    /// Number of MCTS simulations.
    #[arg(default_value_t = 512)]
    simulations: usize,
    /// PUCT constant for AlphaZero search.
    #[arg(default_value_t = 0.0)]
    cpuct: f32,
    /// Print child reply diagnostics for this many top static moves.
    #[arg(long, default_value_t = 3)]
    replies: usize,
    /// FEN string, or startpos if omitted.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    fen: Vec<String>,
}

fn best_model_path(model_path: &str) -> PathBuf {
    Path::new(model_path)
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .join("best.safetensors")
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
    pikafish_depth: u32,
    pikafish_best_wins: usize,
}

impl Default for AzLoopProgressState {
    fn default() -> Self {
        Self {
            next_update: 1,
            best_elo: 1500.0,
            pikafish_depth: 1,
            pikafish_best_wins: 0,
        }
    }
}

impl AzLoopProgressState {
    fn normalize(mut self) -> Self {
        self.next_update = self.next_update.max(1);
        self.pikafish_depth = self.pikafish_depth.max(1);
        if !self.best_elo.is_finite() {
            self.best_elo = 1500.0;
        }
        self
    }
}

fn load_az_loop_progress(config_path: &str) -> AzLoopProgressState {
    let path = az_loop_progress_path(config_path);
    let Ok(text) = fs::read_to_string(&path) else {
        return AzLoopProgressState::default();
    };
    toml::from_str::<AzLoopProgressState>(&text)
        .unwrap_or_else(|err| panic!("failed to parse `{}`: {err}", path.display()))
        .normalize()
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
    pikafish_depth: u32,
    pikafish_best_wins: usize,
) {
    save_az_loop_progress(
        config_path,
        &AzLoopProgressState {
            next_update,
            best_elo,
            pikafish_depth,
            pikafish_best_wins,
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
            "lrm{}_lds{}_ldi{}_ldf{}_cp{}_cpr{}_fv{}_fvr{}_pst{}_tb{}_teg{}_tdd{}_tde{}_tco{}_tvc{}_tvo{}_op{}_rs{}_rp{}_rc{}_tspu{}_mp{}_cpi{}_",
            "tepu{}_mstc{}_vtd{}_ai{}_acp{}_rda{}_ref{}_sd{}"
        ),
        config.simulations,
        config.selfplay_samples_per_update,
        config.batch_size,
        f32_slug(config.lr),
        config.hidden_size,
        config.max_plies,
        config.workers,
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
        config.temperature_cutoff_plies,
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
        config.max_sample_train_count,
        f32_slug(config.mirror_probability),
        f32_slug(config.value_td_lambda),
        config.checkpoint_interval,
        config.arena_interval,
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

fn baseline_100_config(cmd: &AzBaseline100Args) -> AzLoopFileConfig {
    let mut config = AzLoopFileConfig::default();
    config.model_path = cmd.model.clone();
    config.simulations = 800;
    config.selfplay_samples_per_update = 12000;
    config.train_samples_per_update = 24000;
    config.train_warmup_samples = config.train_samples_per_update;
    config.replay_capacity = 80000;
    config.train_epochs_per_update = 2;
    config.max_sample_train_count = 3;
    config.hidden_size = cmd.hidden.max(1);
    config.workers = cmd.workers.max(1);
    config.checkpoint_dir = format!(
        "baseline-checkpoints-h{}-u{}",
        cmd.hidden.max(1),
        cmd.target_update.max(1)
    );
    config.tensorboard_logdir = format!(
        "runs/baseline-h{}-u{}",
        cmd.hidden.max(1),
        cmd.target_update.max(1)
    );
    config.arena_interval = 0;
    config
}

fn run_baseline_100(cmd: AzBaseline100Args) {
    let config_path = Path::new(&cmd.config);
    if !config_path.exists() {
        let config = baseline_100_config(&cmd);
        fs::write(config_path, config.to_file_text()).unwrap_or_else(|err| {
            panic!(
                "failed to create baseline config `{}`: {err}",
                config_path.display()
            );
        });
        println!("created baseline config: {}", config_path.display());
    } else {
        println!("use existing baseline config: {}", config_path.display());
    }
    println!(
        "baseline : target_update={} workers={} hidden={} cuda_visible_devices={}",
        cmd.target_update.max(1),
        cmd.workers.max(1),
        cmd.hidden.max(1),
        cmd.gpu
    );
    println!(
        "baseline : run `az-loop {} --target-update {}`",
        cmd.config,
        cmd.target_update.max(1)
    );
    if cmd.no_run {
        return;
    }

    let exe = std::env::current_exe()
        .unwrap_or_else(|err| panic!("failed to resolve current executable: {err}"));
    let status = Command::new(exe)
        .env("CUDA_VISIBLE_DEVICES", &cmd.gpu)
        .arg("az-loop")
        .arg(&cmd.config)
        .arg("--target-update")
        .arg(cmd.target_update.max(1).to_string())
        .status()
        .unwrap_or_else(|err| panic!("failed to launch az-loop baseline child: {err}"));
    if !status.success() {
        panic!("az-loop baseline child failed with status {status}");
    }
}

fn checkpoint_path(model_path: &str, checkpoint_dir: &str, update: usize) -> PathBuf {
    let base = Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.safetensors");
    Path::new(checkpoint_dir).join(format!("update-{update:06}-{base}"))
}

fn arena_pikafish_candidate_path(model_path: &str, checkpoint_dir: &str, update: usize) -> PathBuf {
    let base = Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.safetensors");
    Path::new(checkpoint_dir).join(format!("arena-pikafish-candidate-{update:04}-{base}"))
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

#[derive(Args, Debug)]
struct AzPolicyFitBenchArgs {
    /// Student AZ-NNUE model path.
    model: String,
    /// Generated teacher-labeled sample count.
    #[arg(default_value_t = 8192)]
    samples: usize,
    /// Passes over generated samples.
    #[arg(default_value_t = 20)]
    epochs: usize,
    /// Micro-batch size per visible GPU.
    #[arg(default_value_t = 1024)]
    batch_size_per_gpu: usize,
    /// Learning rate.
    #[arg(default_value_t = 0.0003)]
    lr: f32,
    /// Random seed for data order and positions.
    #[arg(default_value_t = 20260522)]
    seed: u64,
    /// Random seed for the frozen teacher.
    #[arg(long, default_value_t = 20260523)]
    teacher_seed: u64,
    /// Maximum random plies used to generate positions from startpos.
    #[arg(long, default_value_t = 80)]
    max_random_plies: usize,
    /// Temperature applied to teacher policy logits before creating targets.
    #[arg(long, default_value_t = 1.0)]
    target_temperature: f32,
}

#[derive(Args, Debug)]
struct AzSelfplayFitBenchArgs {
    /// AZ-NNUE model path.
    model: String,
    /// Self-play games to generate before freezing the dataset.
    #[arg(default_value_t = 2000)]
    games: usize,
    /// MCTS simulations per self-play move.
    #[arg(default_value_t = 800)]
    simulations: usize,
    /// Self-play worker threads.
    #[arg(default_value_t = 44)]
    workers: usize,
    /// Training epochs over the frozen self-play dataset.
    #[arg(default_value_t = 20)]
    epochs: usize,
    /// Micro-batch size per visible GPU.
    #[arg(default_value_t = 1024)]
    batch_size_per_gpu: usize,
    /// Learning rate.
    #[arg(default_value_t = 0.0003)]
    lr: f32,
    /// Random seed.
    #[arg(default_value_t = 20260522)]
    seed: u64,
    /// Maximum plies per self-play game.
    #[arg(long, default_value_t = 300)]
    max_plies: usize,
    /// PUCT constant for AlphaZero search.
    #[arg(long, default_value_t = 1.5)]
    cpuct: f32,
    /// Root Dirichlet alpha.
    #[arg(long, default_value_t = 0.0)]
    root_dirichlet_alpha: f32,
    /// Root Dirichlet exploration fraction.
    #[arg(long, default_value_t = 0.0)]
    root_exploration_fraction: f32,
    /// Opening temperature.
    #[arg(long, default_value_t = 0.9)]
    temperature_start: f32,
    /// Endgame temperature after cutoff.
    #[arg(long, default_value_t = 0.6)]
    temperature_endgame: f32,
    /// Plies before temperature starts decaying.
    #[arg(long, default_value_t = 20)]
    temperature_decay_delay_plies: usize,
    /// Plies over which temperature decays.
    #[arg(long, default_value_t = 60)]
    temperature_decay_plies: usize,
    /// Ply from which endgame temperature is used.
    #[arg(long, default_value_t = 40)]
    temperature_cutoff_plies: usize,
    /// Exclude temperature-sampled moves worse than best Q by this much.
    #[arg(long, default_value_t = 0.15)]
    temperature_value_cutoff: f32,
    /// Visit offset applied before temperature sampling.
    #[arg(long, default_value_t = -0.8)]
    temperature_visit_offset: f32,
    /// File-mirror augmentation probability.
    #[arg(long, default_value_t = 0.5)]
    mirror_probability: f32,
    /// TD lambda used to mix search value with terminal game return.
    #[arg(long, default_value_t = 0.85)]
    value_td_lambda: f32,
    /// Save generated fixed self-play data as replay lz4.
    #[arg(long)]
    replay_out: Option<String>,
    /// Load fixed self-play data from replay lz4 instead of regenerating.
    #[arg(long)]
    replay_in: Option<String>,
    /// Stop after this many epochs without policy CE improvement; 0 disables.
    #[arg(long, default_value_t = 5)]
    early_stop_patience: usize,
    /// Minimum policy CE improvement counted by early stopping.
    #[arg(long, default_value_t = 0.0005)]
    min_delta: f32,
    /// Print fixed-fit train/holdout metrics every N epochs; 0 disables trace.
    #[arg(long, default_value_t = 1)]
    trace_interval: usize,
    /// Fraction of fixed samples reserved for holdout evaluation only.
    #[arg(long, default_value_t = 0.0)]
    holdout_fraction: f32,
    /// Holdout split granularity for fixed-fit diagnostics.
    #[arg(long, value_enum, default_value_t = HoldoutSplit::Sample)]
    holdout_split: HoldoutSplit,
    /// Value loss weight used by fixed-fit diagnostics.
    #[arg(long, default_value_t = 1.0)]
    fit_value_weight: f32,
    /// Policy loss weight used by fixed-fit diagnostics.
    #[arg(long, default_value_t = 1.0)]
    fit_policy_weight: f32,
}

#[derive(Args, Debug)]
struct AzReplayGenerateFixedArgs {
    /// Teacher AZ-NNUE model path used to generate fixed self-play.
    model: String,
    /// Output replay lz4 path.
    output: String,
    /// Target number of samples to keep in the replay.
    #[arg(default_value_t = 1_000_000)]
    samples: usize,
    /// Self-play games per generation batch.
    #[arg(long, default_value_t = 300)]
    batch_games: usize,
    /// MCTS simulations per self-play move.
    #[arg(long, default_value_t = 800)]
    simulations: usize,
    /// Self-play worker threads.
    #[arg(long, default_value_t = 30)]
    workers: usize,
    /// Random seed.
    #[arg(long, default_value_t = 20260411)]
    seed: u64,
    /// Maximum plies per self-play game.
    #[arg(long, default_value_t = 300)]
    max_plies: usize,
    /// PUCT constant for AlphaZero search.
    #[arg(long, default_value_t = 1.5)]
    cpuct: f32,
    /// Root Dirichlet alpha.
    #[arg(long, default_value_t = 0.0)]
    root_dirichlet_alpha: f32,
    /// Root Dirichlet exploration fraction.
    #[arg(long, default_value_t = 0.0)]
    root_exploration_fraction: f32,
    /// Opening temperature.
    #[arg(long, default_value_t = 0.9)]
    temperature_start: f32,
    /// Endgame temperature after cutoff.
    #[arg(long, default_value_t = 0.6)]
    temperature_endgame: f32,
    /// Plies before temperature starts decaying.
    #[arg(long, default_value_t = 20)]
    temperature_decay_delay_plies: usize,
    /// Plies over which temperature decays.
    #[arg(long, default_value_t = 60)]
    temperature_decay_plies: usize,
    /// Ply from which endgame temperature is used.
    #[arg(long, default_value_t = 40)]
    temperature_cutoff_plies: usize,
    /// Exclude temperature-sampled moves worse than best Q by this much.
    #[arg(long, default_value_t = 0.15)]
    temperature_value_cutoff: f32,
    /// Visit offset applied before temperature sampling.
    #[arg(long, default_value_t = -0.8)]
    temperature_visit_offset: f32,
    /// File-mirror augmentation probability.
    #[arg(long, default_value_t = 0.3)]
    mirror_probability: f32,
    /// TD lambda used to mix search value with terminal game return.
    #[arg(long, default_value_t = 0.85)]
    value_td_lambda: f32,
}

#[derive(Args, Debug)]
struct AzTrainFixedReplayArgs {
    /// Student model path to load before fixed replay training.
    model: String,
    /// Fixed replay lz4 path.
    replay: String,
    /// Output model path. Defaults to overwriting model.
    #[arg(long)]
    output: Option<String>,
    /// Number of fixed-data update steps.
    #[arg(long, default_value_t = 100)]
    target_update: usize,
    /// Samples drawn from the fixed train split per update.
    #[arg(long, default_value_t = 24_000)]
    train_samples_per_update: usize,
    /// Epochs over each update draw.
    #[arg(long, default_value_t = 2)]
    train_epochs_per_update: usize,
    /// Micro-batch size per visible GPU.
    #[arg(long, default_value_t = 4096)]
    batch_size_per_gpu: usize,
    /// Learning rate.
    #[arg(long, default_value_t = 0.001)]
    lr: f32,
    /// Random seed for holdout split and training draws.
    #[arg(long, default_value_t = 20260411)]
    seed: u64,
    /// Fraction of replay samples reserved for holdout evaluation only.
    #[arg(long, default_value_t = 0.1)]
    holdout_fraction: f32,
    /// Print full train/holdout evaluation every N updates.
    #[arg(long, default_value_t = 10)]
    eval_interval: usize,
    /// Max train-split samples used for fixed evaluation. 0 means all samples.
    #[arg(long, default_value_t = 50_000)]
    eval_train_samples: usize,
    /// Max holdout-split samples used for fixed evaluation. 0 means all samples.
    #[arg(long, default_value_t = 50_000)]
    eval_holdout_samples: usize,
    /// Value loss weight.
    #[arg(long, default_value_t = 1.0)]
    value_weight: f32,
    /// Policy loss weight.
    #[arg(long, default_value_t = 1.0)]
    policy_weight: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum HoldoutSplit {
    Sample,
    Game,
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
    model: AzNnue,
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
    opening_positions: &[Position],
) -> AzLoopConfig {
    AzLoopConfig {
        games: 1,
        max_plies: config.max_plies,
        simulations: config.simulations,
        seed,
        workers,
        temperature_start: config.temperature_start,
        temperature_endgame: config.temperature_endgame,
        temperature_decay_delay_plies: config.temperature_decay_delay_plies,
        temperature_decay_plies: config.temperature_decay_plies,
        temperature_cutoff_plies: config.temperature_cutoff_plies,
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
        root_exploration_plies: config.root_exploration_plies,
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
        value_td_lambda: config.value_td_lambda,
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
                panic!(
                    "invalid opening FEN at `{path}` line {}: {err}",
                    index + 1
                )
            }))
        })
        .collect()
}

fn build_async_training_report(
    pending: PendingTrainingData,
    stats: chineseai::az::AzTrainStats,
    learning_rate: f32,
    train_data_len: usize,
    train_seconds: f32,
    pool_samples: usize,
    pool_capacity: usize,
) -> AzLoopReport {
    let selfplay_games = pending.selfplay.games.len();
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
    AzLoopReport {
        games: selfplay_games,
        samples: selfplay_samples,
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
        policy_ce: stats.policy_ce,
        policy_kl: stats.policy_ce - root_visit_entropy,
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
        selfplay_seconds: pending.selfplay_seconds,
        train_seconds,
        total_seconds,
        games_per_second: selfplay_games as f32 / total_seconds.max(1e-6),
        samples_per_second: selfplay_samples as f32 / total_seconds.max(1e-6),
        train_samples_per_second: train_data_len as f32 / train_seconds.max(1e-6),
        train_samples: train_data_len,
        pool_samples,
        pool_capacity,
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

fn split_fixed_fit_samples(
    mut samples: Vec<AzTrainingSample>,
    holdout_fraction: f32,
    seed: u64,
) -> (Vec<AzTrainingSample>, Vec<AzTrainingSample>) {
    let holdout_fraction = holdout_fraction.clamp(0.0, 0.9);
    if samples.len() < 2 || holdout_fraction <= 0.0 {
        return (samples, Vec::new());
    }
    let mut rng = SplitMix64::new(seed ^ 0xA5A5_5A5A_D3C1_B2E7);
    for i in (1..samples.len()).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        samples.swap(i, j);
    }
    let holdout_len = ((samples.len() as f32) * holdout_fraction).round() as usize;
    let holdout_len = holdout_len.clamp(1, samples.len() - 1);
    let holdout = samples.split_off(samples.len() - holdout_len);
    (samples, holdout)
}

fn split_fixed_fit_games(
    mut games: Vec<Vec<AzTrainingSample>>,
    holdout_fraction: f32,
    seed: u64,
) -> (Vec<AzTrainingSample>, Vec<AzTrainingSample>) {
    let holdout_fraction = holdout_fraction.clamp(0.0, 0.9);
    if games.len() < 2 || holdout_fraction <= 0.0 {
        return (games.into_iter().flatten().collect(), Vec::new());
    }
    let mut rng = SplitMix64::new(seed ^ 0xB47C_9D31_A5A5_5A5A);
    for i in (1..games.len()).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        games.swap(i, j);
    }
    let holdout_len = ((games.len() as f32) * holdout_fraction).round() as usize;
    let holdout_len = holdout_len.clamp(1, games.len() - 1);
    let holdout_games = games.split_off(games.len() - holdout_len);
    (
        games.into_iter().flatten().collect(),
        holdout_games.into_iter().flatten().collect(),
    )
}

fn sample_fixed_training_batch(
    samples: &[AzTrainingSample],
    count: usize,
    rng: &mut SplitMix64,
) -> Vec<AzTrainingSample> {
    if samples.is_empty() || count == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let index = (rng.next_u64() as usize) % samples.len();
        out.push(samples[index].clone());
    }
    out
}

fn fixed_eval_subset(
    samples: &[AzTrainingSample],
    max_samples: usize,
    seed: u64,
) -> Vec<AzTrainingSample> {
    if max_samples == 0 || samples.len() <= max_samples {
        return samples.to_vec();
    }
    let mut indices: Vec<usize> = (0..samples.len()).collect();
    let mut rng = SplitMix64::new(seed);
    for i in (1..indices.len()).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        indices.swap(i, j);
    }
    indices.truncate(max_samples);
    indices
        .into_iter()
        .map(|index| samples[index].clone())
        .collect()
}

fn print_fixed_replay_fit_stats(prefix: &str, stats: &chineseai::az::AzPolicyFitBenchmark) {
    println!("{prefix}_samples: {}", stats.train_samples);
    println!("{prefix}_holdout_samples: {}", stats.holdout_samples);
    println!("{prefix}_targetH: {:.4}", stats.target_entropy);
    println!("{prefix}_value_ce: {:.4}", stats.final_value_ce);
    println!("{prefix}_value_mse: {:.4}", stats.final_value_mse);
    println!("{prefix}_policy_ce: {:.4}", stats.final_policy_ce);
    println!("{prefix}_policy_kl: {:.4}", stats.final_policy_kl);
    if stats.holdout_samples > 0 {
        println!(
            "{prefix}_holdout_targetH: {:.4}",
            stats.holdout_target_entropy
        );
        println!(
            "{prefix}_holdout_value_ce: {:.4}",
            stats.holdout_final_value_ce
        );
        println!(
            "{prefix}_holdout_value_mse: {:.4}",
            stats.holdout_final_value_mse
        );
        println!(
            "{prefix}_holdout_policy_ce: {:.4}",
            stats.holdout_final_policy_ce
        );
        println!(
            "{prefix}_holdout_policy_kl: {:.4}",
            stats.holdout_final_policy_kl
        );
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

fn load_arena_eval_positions(path: &str) -> Vec<Position> {
    let path = Path::new(path);
    if !path.exists() {
        return Vec::new();
    }
    let text = fs::read_to_string(path).unwrap_or_else(|err| {
        panic!("failed to read arena eval fens `{}`: {err}", path.display());
    });
    let mut positions = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        let fen = line.split('#').next().unwrap_or("").trim();
        if fen.is_empty() {
            continue;
        }
        let position = Position::from_fen(fen).unwrap_or_else(|err| {
            panic!(
                "invalid arena eval fen at `{}:{}`: {err}",
                path.display(),
                line_index + 1
            );
        });
        positions.push(position);
    }
    positions
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
        draw_score: 0.0,
        moves_left_max_effect: 0.0,
        moves_left_slope: 0.0,
        moves_left_threshold: 0.8,
        moves_left_constant_factor: 0.0,
        moves_left_scaled_factor: 0.0,
        moves_left_quadratic_factor: 0.0,
        value_scale: 1.0,
    }
}

fn log_scalar(writer: &mut SummaryWriter, tag: &str, step: usize, value: f32) {
    writer.add_scalar(tag, value, step);
}

fn piece_label(piece: Option<chineseai::xiangqi::Piece>) -> String {
    piece
        .map(|piece| format!("{:?}{:?}", piece.color, piece.kind))
        .unwrap_or_else(|| "-".to_string())
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
            let position = parse_position(&fen);
            let model = AzNnue::load(&model_path).unwrap_or_else(|err| {
                panic!("failed to load `{model_path}`: {err}");
            });
            let result = alphazero_search(
                &position,
                &model,
                AzSearchLimits {
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
                    fpu_value: 0.23,
                    fpu_value_at_root: 1.0,
                    draw_score: cmd.draw_score.clamp(-1.0, 1.0),
                    moves_left_max_effect: if cmd.moves_left_utility { 0.3 } else { 0.0 },
                    moves_left_slope: if cmd.moves_left_utility { 0.007 } else { 0.0 },
                    moves_left_threshold: 0.8,
                    moves_left_constant_factor: 0.0,
                    moves_left_scaled_factor: if cmd.moves_left_utility { 0.15 } else { 0.0 },
                    moves_left_quadratic_factor: if cmd.moves_left_utility { 0.85 } else { 0.0 },
                    value_scale: 1.0,
                },
            );
            println!("fen      : {}", position.to_fen());
            println!("model    : {model_path}");
            println!("sims     : {}", result.simulations);
            println!("search   : alphazero");
            println!("cpuct    : {cpuct}");
            println!("cpuct_at_root: {cpuct_at_root}");
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
            for candidate in &by_visits {
                println!(
                    "visited: {} visits={} q={:.3} prior={:.5} policy={:.5}",
                    candidate.mv, candidate.visits, candidate.q, candidate.prior, candidate.policy
                );
            }
        }
        Some(CliCommand::AzDiagnose(cmd)) => {
            let model_path = cmd.model;
            let simulations = cmd.simulations.max(1);
            let cpuct = cmd.cpuct.max(0.0);
            let fen = cmd.fen.join(" ");
            let position = parse_position(&fen);
            let model = AzNnue::load(&model_path).unwrap_or_else(|err| {
                panic!("failed to load `{model_path}`: {err}");
            });
            let rule_history = position.initial_rule_history();
            let legal = position.legal_moves_with_rules(&rule_history);
            let root_static = model.evaluate_value(&position, &[], &legal);
            let search = alphazero_search_with_history_and_rules(
                &position,
                &[],
                Some(rule_history.clone()),
                Some(legal.clone()),
                &model,
                fixed_az_search_limits(simulations, 0, cpuct, 0),
            );
            println!("fen          : {}", position.to_fen());
            println!("model        : {model_path}");
            println!("side         : {:?}", position.side_to_move());
            println!("legal_moves  : {}", legal.len());
            println!(
                "root_static  : {:.4} cp={}",
                root_static,
                lc0_cp_from_q(root_static)
            );
            println!(
                "search       : alphazero sims={} cpuct={}",
                search.simulations, cpuct
            );
            println!("search_value : cp={}", search.value_cp);
            println!(
                "bestmove     : {}",
                search
                    .best_move
                    .map(|mv| mv.to_string())
                    .unwrap_or_else(|| "(none)".into())
            );

            let mut rows = Vec::new();
            for mv in legal {
                let captured = position.piece_at(mv.to as usize);
                let moved = position.piece_at(mv.from as usize);
                let mut child = position.clone();
                let mut child_history = Vec::new();
                if let Some(piece) = moved {
                    child_history.push(chineseai::nnue::HistoryMove {
                        piece,
                        captured,
                        mv,
                    });
                }
                let mut child_rule_history = rule_history.clone();
                child_rule_history.push(position.rule_history_entry_after_move(mv));
                child.make_move(mv);
                let child_legal = child.legal_moves_with_rules(&child_rule_history);
                let child_value_for_child =
                    model.evaluate_value(&child, &child_history, &child_legal);
                let child_value_for_root = -child_value_for_child;
                let candidate = search
                    .candidates
                    .iter()
                    .find(|candidate| candidate.mv == mv);
                rows.push((
                    mv,
                    captured,
                    child_value_for_root,
                    child_legal.len(),
                    candidate.map(|candidate| candidate.visits).unwrap_or(0),
                    candidate.map(|candidate| candidate.q).unwrap_or(0.0),
                    candidate.map(|candidate| candidate.prior).unwrap_or(0.0),
                    candidate.map(|candidate| candidate.policy).unwrap_or(0.0),
                ));
            }
            rows.sort_by(|left, right| {
                right
                    .2
                    .total_cmp(&left.2)
                    .then_with(|| right.4.cmp(&left.4))
                    .then_with(|| right.7.total_cmp(&left.7))
            });
            println!("children_by_static_value:");
            println!("move capture child_static_cp child_legal visits q prior policy");
            for (mv, captured, child_value, child_legal, visits, q, prior, policy) in &rows {
                println!(
                    "{} {} {} {} {} {:.4} {:.6} {:.6}",
                    mv,
                    piece_label(*captured),
                    lc0_cp_from_q(*child_value),
                    child_legal,
                    visits,
                    q,
                    prior,
                    policy
                );
            }
            for (mv, _, _, _, _, _, _, _) in rows.iter().take(cmd.replies) {
                let moved = position.piece_at(mv.from as usize);
                let captured = position.piece_at(mv.to as usize);
                let mut child = position.clone();
                let mut child_history = Vec::new();
                if let Some(piece) = moved {
                    child_history.push(chineseai::nnue::HistoryMove {
                        piece,
                        captured,
                        mv: *mv,
                    });
                }
                let mut child_rule_history = rule_history.clone();
                child_rule_history.push(position.rule_history_entry_after_move(*mv));
                child.make_move(*mv);
                let child_legal = child.legal_moves_with_rules(&child_rule_history);
                println!("replies_after {} fen={}", mv, child.to_fen());
                let mut reply_rows = Vec::new();
                for reply in child_legal {
                    let reply_captured = child.piece_at(reply.to as usize);
                    let mut grandchild = child.clone();
                    let mut reply_history = child_history.clone();
                    if let Some(piece) = child.piece_at(reply.from as usize) {
                        reply_history.push(chineseai::nnue::HistoryMove {
                            piece,
                            captured: reply_captured,
                            mv: reply,
                        });
                    }
                    let mut grandchild_rule_history = child_rule_history.clone();
                    grandchild_rule_history.push(child.rule_history_entry_after_move(reply));
                    grandchild.make_move(reply);
                    let grandchild_legal =
                        grandchild.legal_moves_with_rules(&grandchild_rule_history);
                    let grandchild_value_for_grandchild =
                        model.evaluate_value(&grandchild, &reply_history, &grandchild_legal);
                    let reply_value_for_child = -grandchild_value_for_grandchild;
                    reply_rows.push((
                        reply,
                        reply_captured,
                        reply_value_for_child,
                        grandchild.to_fen(),
                    ));
                }
                reply_rows.sort_by(|left, right| right.2.total_cmp(&left.2));
                for (reply, reply_captured, reply_value, reply_fen) in reply_rows {
                    println!(
                        "  reply {} capture={} child_static_cp={} fen={}",
                        reply,
                        piece_label(reply_captured),
                        lc0_cp_from_q(reply_value),
                        reply_fen
                    );
                }
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
        Some(CliCommand::AzPolicyFitBench(cmd)) => {
            let model_path = cmd.model;
            let sample_count = cmd.samples.max(1);
            let epochs = cmd.epochs.max(1);
            let batch_size = cmd.batch_size_per_gpu.max(1);
            let lr = cmd.lr.max(0.0);
            let seed = cmd.seed;
            let teacher_seed = cmd.teacher_seed;
            let max_random_plies = cmd.max_random_plies;
            let target_temperature = cmd.target_temperature.max(1e-3);
            let mut model = AzNnue::load(&model_path).unwrap_or_else(|err| {
                panic!("failed to load `{model_path}`: {err}");
            });
            let started = std::time::Instant::now();
            let stats = benchmark_policy_fit(
                &mut model,
                sample_count,
                epochs,
                batch_size,
                lr,
                seed,
                teacher_seed,
                max_random_plies,
                target_temperature,
            );
            let elapsed = started.elapsed().as_secs_f64().max(f64::EPSILON);
            let processed = (sample_count * epochs) as f64;
            let g_step = global_training_step_sample_count(batch_size);
            let n_gpu = global_training_step_sample_count(1);
            println!("bench           : policy-fit");
            println!("model           : {model_path}");
            println!("samples         : {}", stats.samples);
            println!("epochs          : {epochs}");
            println!("batch(per_gpu)  : {batch_size}");
            println!("cuda_devices    : {n_gpu}  (global batch {g_step})");
            println!("lr              : {lr}");
            println!("teacher_seed    : {teacher_seed}");
            println!("max_random_plies: {max_random_plies}");
            println!("target_temp     : {target_temperature}");
            println!("elapsed_ms      : {:.3}", elapsed * 1000.0);
            println!("processed       : {}", sample_count * epochs);
            println!("samples/sec     : {:.0}", processed / elapsed);
            println!("targetH         : {:.4}", stats.target_entropy);
            println!("initial_value_ce: {:.4}", stats.initial_value_ce);
            println!("initial_value_mse: {:.4}", stats.initial_value_mse);
            println!("initial_policy_ce: {:.4}", stats.initial_policy_ce);
            println!("initial_policy_kl: {:.4}", stats.initial_policy_kl);
            println!("final_value_ce  : {:.4}", stats.final_value_ce);
            println!("final_value_mse : {:.4}", stats.final_value_mse);
            println!("final_policy_ce : {:.4}", stats.final_policy_ce);
            println!("final_policy_kl : {:.4}", stats.final_policy_kl);
            println!("train_loss      : {:.4}", stats.train_loss);
            println!("train_value_ce  : {:.4}", stats.train_value_loss);
        }
        Some(CliCommand::AzSelfplayFitBench(cmd)) => {
            let model_path = cmd.model;
            let games = cmd.games.max(1);
            let simulations = cmd.simulations.max(1);
            let workers = cmd.workers.max(1);
            let epochs = cmd.epochs.max(1);
            let batch_size = cmd.batch_size_per_gpu.max(1);
            let lr = cmd.lr.max(0.0);
            let seed = cmd.seed;
            let mut model = AzNnue::load(&model_path).unwrap_or_else(|err| {
                panic!("failed to load `{model_path}`: {err}");
            });
            let started = std::time::Instant::now();
            let (
                samples,
                game_samples,
                source_groups,
                red_wins,
                black_wins,
                draws,
                avg_plies,
                selfplay_seconds,
            ) = if let Some(replay_in) = cmd.replay_in.as_ref() {
                let pool = AzExperiencePool::load_snapshot_lz4(Path::new(replay_in), games.max(1))
                    .unwrap_or_else(|err| panic!("failed to load replay `{replay_in}`: {err}"));
                (
                    pool.all_samples(),
                    pool.all_sample_groups(),
                    pool.sample_count(),
                    0,
                    0,
                    0,
                    0.0,
                    0.0,
                )
            } else {
                let config = AzLoopConfig {
                    games,
                    max_plies: cmd.max_plies.max(1),
                    simulations,
                    seed,
                    workers,
                    temperature_start: cmd.temperature_start,
                    temperature_endgame: cmd.temperature_endgame,
                    temperature_decay_delay_plies: cmd.temperature_decay_delay_plies,
                    temperature_decay_plies: cmd.temperature_decay_plies,
                    temperature_cutoff_plies: cmd.temperature_cutoff_plies,
                    temperature_value_cutoff: cmd.temperature_value_cutoff,
                    temperature_visit_offset: cmd.temperature_visit_offset,
                    cpuct: cmd.cpuct,
                    cpuct_at_root: 2.53,
                    cpuct_base: 19652.0,
                    cpuct_factor: 2.0,
                    cpuct_base_at_root: 19652.0,
                    cpuct_factor_at_root: 2.0,
                    root_dirichlet_alpha: cmd.root_dirichlet_alpha,
                    root_exploration_fraction: cmd.root_exploration_fraction,
                    root_exploration_plies: cmd.temperature_decay_plies,
                    fpu_value: 0.23,
                    fpu_value_at_root: 1.0,
                    draw_score: 0.0,
                    moves_left_max_effect: 0.3,
                    moves_left_slope: 0.007,
                    moves_left_threshold: 0.8,
                    moves_left_constant_factor: 0.0,
                    moves_left_scaled_factor: 0.15,
                    moves_left_quadratic_factor: 0.85,
                    policy_softmax_temp: 1.45,
                    opening_positions: Vec::new(),
                    resign_percentage: 0.0,
                    resign_playthrough: 100.0,
                    mirror_probability: cmd.mirror_probability,
                    value_td_lambda: cmd.value_td_lambda,
                };
                let selfplay_started = Instant::now();
                let data = generate_selfplay_data(&model, &config);
                let selfplay_seconds = selfplay_started.elapsed().as_secs_f32();
                if let Some(replay_out) = cmd.replay_out.as_ref() {
                    let mut pool = AzExperiencePool::new(data.samples.len().max(1));
                    pool.add_samples(data.samples.clone());
                    pool.save_snapshot_lz4(Path::new(replay_out))
                        .unwrap_or_else(|err| {
                            panic!("failed to save replay `{replay_out}`: {err}");
                        });
                }
                let games_len = data.games.len();
                let avg_plies = if games_len == 0 {
                    0.0
                } else {
                    data.plies_total as f32 / games_len as f32
                };
                (
                    data.samples,
                    data.games,
                    games_len,
                    data.red_wins,
                    data.black_wins,
                    data.draws,
                    avg_plies,
                    selfplay_seconds,
                )
            };
            let (train_samples, holdout_samples) = match cmd.holdout_split {
                HoldoutSplit::Sample => {
                    split_fixed_fit_samples(samples, cmd.holdout_fraction, seed)
                }
                HoldoutSplit::Game => {
                    split_fixed_fit_games(game_samples, cmd.holdout_fraction, seed)
                }
            };
            let fit_loss_weights = AzTrainLossWeights {
                value: cmd.fit_value_weight,
                policy: cmd.fit_policy_weight,
            };
            let (stats, trace) = benchmark_fixed_policy_fit_with_trace(
                &mut model,
                &train_samples,
                &holdout_samples,
                epochs,
                batch_size,
                lr,
                seed,
                cmd.early_stop_patience,
                cmd.min_delta,
                fit_loss_weights,
                cmd.trace_interval,
            );
            let elapsed = started.elapsed().as_secs_f64().max(f64::EPSILON);
            let processed = (stats.train_samples * stats.epochs_completed) as f64;
            let g_step = global_training_step_sample_count(batch_size);
            let n_gpu = global_training_step_sample_count(1);
            println!("bench           : selfplay-policy-fit");
            println!("model           : {model_path}");
            println!("source_groups   : {source_groups}");
            println!("samples         : {}", stats.samples);
            println!("train_samples   : {}", stats.train_samples);
            println!("holdout_samples : {}", stats.holdout_samples);
            println!("holdout_split   : {:?}", cmd.holdout_split);
            println!("R/B/D           : {red_wins}/{black_wins}/{draws}");
            println!("avg_plies       : {:.1}", avg_plies);
            println!("selfplay_sec    : {:.3}", selfplay_seconds);
            println!("simulations     : {simulations}");
            println!("workers         : {workers}");
            println!("search          : alphazero");
            println!("epochs          : {epochs}");
            println!("epochs_done     : {}", stats.epochs_completed);
            println!("batch(per_gpu)  : {batch_size}");
            println!("cuda_devices    : {n_gpu}  (global batch {g_step})");
            println!("lr              : {lr}");
            if let Some(replay_in) = cmd.replay_in.as_ref() {
                println!("replay_in       : {replay_in}");
            }
            if let Some(replay_out) = cmd.replay_out.as_ref() {
                println!("replay_out      : {replay_out}");
            }
            println!("early_stop_patience: {}", cmd.early_stop_patience);
            println!("min_delta       : {}", cmd.min_delta);
            println!("trace_interval  : {}", cmd.trace_interval);
            println!(
                "fit_loss        : value={} policy={}",
                fit_loss_weights.value, fit_loss_weights.policy
            );
            println!("elapsed_ms      : {:.3}", elapsed * 1000.0);
            println!(
                "processed       : {}",
                stats.train_samples * stats.epochs_completed
            );
            println!("train_samples/sec: {:.0}", processed / elapsed);
            println!("targetH         : {:.4}", stats.target_entropy);
            println!("initial_value_ce: {:.4}", stats.initial_value_ce);
            println!("initial_value_mse: {:.4}", stats.initial_value_mse);
            println!("initial_policy_ce: {:.4}", stats.initial_policy_ce);
            println!("initial_policy_kl: {:.4}", stats.initial_policy_kl);
            for entry in &trace {
                if stats.holdout_samples > 0 {
                    println!(
                        "fit_epoch {epoch:04}: train_policy_ce={train_policy_ce:.4} train_policy_kl={train_policy_kl:.4} holdout_policy_ce={holdout_policy_ce:.4} holdout_policy_kl={holdout_policy_kl:.4} policy_gap={policy_gap:+.4} train_value_ce={train_value_ce:.4} holdout_value_ce={holdout_value_ce:.4} value_gap={value_gap:+.4}",
                        epoch = entry.epoch,
                        train_policy_ce = entry.train_policy_ce,
                        train_policy_kl = entry.train_policy_kl,
                        holdout_policy_ce = entry.holdout_policy_ce,
                        holdout_policy_kl = entry.holdout_policy_kl,
                        policy_gap = entry.holdout_policy_ce - entry.train_policy_ce,
                        train_value_ce = entry.train_value_ce,
                        holdout_value_ce = entry.holdout_value_ce,
                        value_gap = entry.holdout_value_ce - entry.train_value_ce,
                    );
                } else {
                    println!(
                        "fit_epoch {epoch:04}: train_policy_ce={train_policy_ce:.4} train_policy_kl={train_policy_kl:.4} train_value_ce={train_value_ce:.4} train_value_mse={train_value_mse:.4}",
                        epoch = entry.epoch,
                        train_policy_ce = entry.train_policy_ce,
                        train_policy_kl = entry.train_policy_kl,
                        train_value_ce = entry.train_value_ce,
                        train_value_mse = entry.train_value_mse,
                    );
                }
            }
            println!("final_value_ce  : {:.4}", stats.final_value_ce);
            println!("final_value_mse : {:.4}", stats.final_value_mse);
            println!("final_policy_ce : {:.4}", stats.final_policy_ce);
            println!("final_policy_kl : {:.4}", stats.final_policy_kl);
            if stats.holdout_samples > 0 {
                println!("holdout_targetH : {:.4}", stats.holdout_target_entropy);
                println!(
                    "holdout_initial_value_ce: {:.4}",
                    stats.holdout_initial_value_ce
                );
                println!(
                    "holdout_initial_value_mse: {:.4}",
                    stats.holdout_initial_value_mse
                );
                println!(
                    "holdout_initial_policy_ce: {:.4}",
                    stats.holdout_initial_policy_ce
                );
                println!(
                    "holdout_initial_policy_kl: {:.4}",
                    stats.holdout_initial_policy_kl
                );
                println!(
                    "holdout_final_value_ce: {:.4}",
                    stats.holdout_final_value_ce
                );
                println!(
                    "holdout_final_value_mse: {:.4}",
                    stats.holdout_final_value_mse
                );
                println!(
                    "holdout_final_policy_ce: {:.4}",
                    stats.holdout_final_policy_ce
                );
                println!(
                    "holdout_final_policy_kl: {:.4}",
                    stats.holdout_final_policy_kl
                );
            }
            println!("train_loss      : {:.4}", stats.train_loss);
            println!("train_value_ce  : {:.4}", stats.train_value_loss);
        }
        Some(CliCommand::AzReplayGenerateFixed(cmd)) => {
            let model_path = cmd.model;
            let output = cmd.output;
            let target_samples = cmd.samples.max(1);
            let batch_games = cmd.batch_games.max(1);
            let simulations = cmd.simulations.max(1);
            let workers = cmd.workers.max(1);
            let seed = cmd.seed;
            let model = AzNnue::load(&model_path).unwrap_or_else(|err| {
                panic!("failed to load `{model_path}`: {err}");
            });
            let mut pool = AzExperiencePool::new(target_samples);
            let started = Instant::now();
            let mut batch_index = 0usize;
            let mut total_games = 0usize;
            let mut total_generated_samples = 0usize;
            let mut red_wins = 0usize;
            let mut black_wins = 0usize;
            let mut draws = 0usize;
            let mut plies_total = 0usize;
            while pool.sample_count() < target_samples {
                let config = AzLoopConfig {
                    games: batch_games,
                    max_plies: cmd.max_plies.max(1),
                    simulations,
                    seed: seed.wrapping_add(batch_index as u64 * 0x9E37_79B9_7F4A_7C15),
                    workers,
                    temperature_start: cmd.temperature_start,
                    temperature_endgame: cmd.temperature_endgame,
                    temperature_decay_delay_plies: cmd.temperature_decay_delay_plies,
                    temperature_decay_plies: cmd.temperature_decay_plies,
                    temperature_cutoff_plies: cmd.temperature_cutoff_plies,
                    temperature_value_cutoff: cmd.temperature_value_cutoff,
                    temperature_visit_offset: cmd.temperature_visit_offset,
                    cpuct: cmd.cpuct,
                    cpuct_at_root: 2.53,
                    cpuct_base: 19652.0,
                    cpuct_factor: 2.0,
                    cpuct_base_at_root: 19652.0,
                    cpuct_factor_at_root: 2.0,
                    root_dirichlet_alpha: cmd.root_dirichlet_alpha,
                    root_exploration_fraction: cmd.root_exploration_fraction,
                    root_exploration_plies: cmd.temperature_decay_plies,
                    fpu_value: 0.23,
                    fpu_value_at_root: 1.0,
                    draw_score: 0.0,
                    moves_left_max_effect: 0.3,
                    moves_left_slope: 0.007,
                    moves_left_threshold: 0.8,
                    moves_left_constant_factor: 0.0,
                    moves_left_scaled_factor: 0.15,
                    moves_left_quadratic_factor: 0.85,
                    policy_softmax_temp: 1.45,
                    opening_positions: Vec::new(),
                    resign_percentage: 0.0,
                    resign_playthrough: 100.0,
                    mirror_probability: cmd.mirror_probability,
                    value_td_lambda: cmd.value_td_lambda,
                };
                let data = generate_selfplay_data(&model, &config);
                total_games += data.games.len();
                total_generated_samples += data.samples.len();
                red_wins += data.red_wins;
                black_wins += data.black_wins;
                draws += data.draws;
                plies_total += data.plies_total;
                pool.add_samples(data.samples);
                batch_index += 1;
                println!(
                    "batch {:04}: games={} generated_samples={} kept={}/{} R/B/D={}/{}/{} avg_plies={:.1} elapsed={:.1}s",
                    batch_index,
                    total_games,
                    total_generated_samples,
                    pool.sample_count(),
                    target_samples,
                    red_wins,
                    black_wins,
                    draws,
                    if total_games == 0 {
                        0.0
                    } else {
                        plies_total as f32 / total_games as f32
                    },
                    started.elapsed().as_secs_f32()
                );
            }
            pool.save_snapshot_lz4(Path::new(&output))
                .unwrap_or_else(|err| panic!("failed to save replay `{output}`: {err}"));
            println!("fixed_replay    : generated");
            println!("model           : {model_path}");
            println!("output          : {output}");
            println!("samples         : {}", pool.sample_count());
            println!("generated_samples: {total_generated_samples}");
            println!("games           : {total_games}");
            println!("R/B/D           : {red_wins}/{black_wins}/{draws}");
            println!(
                "avg_plies       : {:.1}",
                if total_games == 0 {
                    0.0
                } else {
                    plies_total as f32 / total_games as f32
                }
            );
            println!("simulations     : {simulations}");
            println!("workers         : {workers}");
            println!("search          : alphazero");
            println!("elapsed_sec     : {:.3}", started.elapsed().as_secs_f32());
        }
        Some(CliCommand::AzTrainFixedReplay(cmd)) => {
            let model_path = cmd.model;
            let replay_path = cmd.replay;
            let output_path = cmd.output.unwrap_or_else(|| model_path.clone());
            let mut model = AzNnue::load(&model_path).unwrap_or_else(|err| {
                panic!("failed to load `{model_path}`: {err}");
            });
            let pool = AzExperiencePool::load_snapshot_lz4(Path::new(&replay_path), 100_000_000)
                .unwrap_or_else(|err| panic!("failed to load replay `{replay_path}`: {err}"));
            let samples = pool.all_samples();
            let (train_samples, holdout_samples) =
                split_fixed_fit_samples(samples, cmd.holdout_fraction, cmd.seed);
            if train_samples.is_empty() {
                panic!("fixed replay `{replay_path}` has no train samples");
            }
            let eval_train_samples = fixed_eval_subset(
                &train_samples,
                cmd.eval_train_samples,
                cmd.seed ^ 0xE7A1_100D,
            );
            let eval_holdout_samples = fixed_eval_subset(
                &holdout_samples,
                cmd.eval_holdout_samples,
                cmd.seed ^ 0xE7A1_110D,
            );
            let updates = cmd.target_update.max(1);
            let draw_count = cmd.train_samples_per_update.max(1);
            let epochs = cmd.train_epochs_per_update.max(1);
            let batch_size = cmd.batch_size_per_gpu.max(1);
            let lr = cmd.lr.max(0.0);
            let eval_interval = cmd.eval_interval.max(1);
            let loss_weights = AzTrainLossWeights {
                value: cmd.value_weight,
                policy: cmd.policy_weight,
            };
            let mut rng = SplitMix64::new(cmd.seed ^ 0x7E57_F1ED_DA7A_5EED);
            let started = Instant::now();
            let g_step = global_training_step_sample_count(batch_size);
            let n_gpu = global_training_step_sample_count(1);
            println!("fixed_train     : start");
            println!("model           : {model_path}");
            println!("replay          : {replay_path}");
            println!("output          : {output_path}");
            println!("train_samples   : {}", train_samples.len());
            println!("holdout_samples : {}", holdout_samples.len());
            println!("eval_train      : {}", eval_train_samples.len());
            println!("eval_holdout    : {}", eval_holdout_samples.len());
            println!("target_update   : {updates}");
            println!("draw_per_update : {draw_count}");
            println!("epochs_per_update: {epochs}");
            println!("batch(per_gpu)  : {batch_size}");
            println!("cuda_devices    : {n_gpu}  (global batch {g_step})");
            println!("lr              : {lr}");
            println!(
                "fit_loss        : value={} policy={}",
                loss_weights.value, loss_weights.policy
            );
            let initial_stats = benchmark_fixed_policy_fit(
                &mut model,
                &eval_train_samples,
                &eval_holdout_samples,
                0,
                batch_size,
                lr,
                cmd.seed,
                0,
                0.0,
                loss_weights,
            );
            print_fixed_replay_fit_stats("initial", &initial_stats);
            for update in 1..=updates {
                let draw = sample_fixed_training_batch(&train_samples, draw_count, &mut rng);
                let train_started = Instant::now();
                let stats = train_samples_weighted(
                    &mut model,
                    &draw,
                    epochs,
                    lr,
                    batch_size,
                    &mut rng,
                    loss_weights,
                );
                let train_seconds = train_started.elapsed().as_secs_f32();
                if update == 1 || update == updates || update % eval_interval == 0 {
                    let eval_stats = benchmark_fixed_policy_fit(
                        &mut model,
                        &eval_train_samples,
                        &eval_holdout_samples,
                        0,
                        batch_size,
                        lr,
                        cmd.seed,
                        0,
                        0.0,
                        loss_weights,
                    );
                    println!(
                        "update {:04}: loss={:.4} value_ce={:.4} policy_ce={:.4} train_batch_samples={} train={:.3}s elapsed={:.1}s",
                        update,
                        stats.loss,
                        stats.value_loss,
                        stats.policy_ce,
                        draw.len() * epochs,
                        train_seconds,
                        started.elapsed().as_secs_f32()
                    );
                    print_fixed_replay_fit_stats("eval", &eval_stats);
                } else {
                    println!(
                        "update {:04}: loss={:.4} value_ce={:.4} policy_ce={:.4} train_batch_samples={} train={:.3}s elapsed={:.1}s",
                        update,
                        stats.loss,
                        stats.value_loss,
                        stats.policy_ce,
                        draw.len() * epochs,
                        train_seconds,
                        started.elapsed().as_secs_f32()
                    );
                }
            }
            save_model(&model, Path::new(&output_path));
            println!("fixed_train     : complete");
            println!("output          : {output_path}");
            println!("elapsed_sec     : {:.3}", started.elapsed().as_secs_f32());
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
            let mut arena_pikafish_depth = progress_boot
                .pikafish_depth
                .max(config.arena_pikafish_depth.max(1));
            let mut arena_pikafish_best_wins = progress_boot.pikafish_best_wins;
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
                    "resume   : update starts at {} (from `{}`) arena_ref_elo={:.1} pikafish_depth={} pikafish_best_wins={}",
                    start_update,
                    az_loop_progress_path(&config_path).display(),
                    arena_best_elo,
                    arena_pikafish_depth,
                    arena_pikafish_best_wins
                );
            }
            let best_path = best_model_path(&config.model_path);

            let config_arch = config.arch();
            let model_path = Path::new(&config.model_path);
            let model = if model_path.exists() {
                println!("model    : load {}", config.model_path);
                match AzNnue::load(model_path) {
                    Ok(model) => {
                        if model.arch != config_arch {
                            println!(
                                "model    : loaded arch={:?} differs from config arch={:?}; keep loaded arch",
                                model.arch, config_arch
                            );
                        }
                        model
                    }
                    Err(err) => {
                        println!(
                            "model    : reinit {} as random nnue ({err})",
                            config.model_path
                        );
                        AzNnue::random_with_arch(config_arch, config.seed)
                    }
                }
            } else if config.arena_interval > 0 && best_path.exists() {
                println!("model    : load best `{}` as current", best_path.display());
                AzNnue::load(&best_path).unwrap_or_else(|err| {
                    panic!("failed to load best model `{}`: {err}", best_path.display());
                })
            } else {
                println!("model    : init {}", config.model_path);
                AzNnue::random_with_arch(config_arch, config.seed)
            };
            let selfplay_model = if config.arena_interval == 0 {
                println!("selfplay : start from current `{}`", config.model_path);
                model.clone()
            } else {
                if !best_path.exists() {
                    save_model(&model, &best_path);
                } else if let Err(err) = AzNnue::load(&best_path) {
                    println!(
                        "best     : reset incompatible `{}` from current model ({err})",
                        best_path.display()
                    );
                    save_model(&model, &best_path);
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

            println!(
                "loop     : config={} mode=batch search=alphazero sims={} selfplay_samples_per_update={} lr={} lr_decay(min={},start={},interval={},factor={}) batch_size(per_gpu)={} global_step_samples={} train_warmup_samples={} train_samples_per_update={} train_epochs_per_update={} max_sample_train_count={} max_plies={} selfplay_workers={} temp(start={},endgame={},delay={}ply,decay={}ply,cutoff={}ply,value_cutoff={},visit_offset={}) cpuct={} cpuct_at_root={} fpu(value={},root={}) policy_softmax_temp={} root_noise(alpha={},fraction={},plies={}) opening_fens={} opening_count={} resign(percentage={},playthrough={}) replay_capacity={} mirror_probability={} value_td_lambda={} train(value={},policy={}) checkpoint_interval={} max_checkpoints={} arena_interval={} arena_cpuct={} arena_promotion_rate={} arena_promotion_z={} arena_processes={} arena_eval_fens={} arena_pikafish(exe={},start_update={},depth={},games={},parallel={},promotion_rate={},eval_fens={}) tb_base={} tb_run={}",
                config_path,
                config.simulations,
                config.selfplay_samples_per_update,
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
                config.max_sample_train_count,
                config.max_plies,
                config.workers,
                config.temperature_start,
                config.temperature_endgame,
                config.temperature_decay_delay_plies,
                config.temperature_decay_plies,
                config.temperature_cutoff_plies,
                config.temperature_value_cutoff,
                config.temperature_visit_offset,
                config.cpuct,
                config.cpuct_at_root,
                config.fpu_value,
                config.fpu_value_at_root,
                config.policy_softmax_temp,
                config.root_dirichlet_alpha,
                config.root_exploration_fraction,
                config.root_exploration_plies,
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
                config.value_td_lambda,
                config.train_value_weight,
                config.train_policy_weight,
                config.checkpoint_interval,
                config.max_checkpoints,
                config.arena_interval,
                config.arena_cpuct,
                config.arena_promotion_rate,
                config.arena_promotion_confidence_z,
                config.arena_processes,
                config.arena_pikafish_eval_fens,
                config.arena_pikafish_exe,
                config.arena_pikafish_start_update,
                config.arena_pikafish_depth,
                config.arena_pikafish_games,
                config.arena_pikafish_parallel_games,
                config.arena_pikafish_promotion_rate,
                config.arena_pikafish_eval_fens,
                config.tensorboard_logdir,
                tensorboard_encoded_subdir(&config)
            );
            let (selfplay_tx, selfplay_rx) =
                mpsc::sync_channel::<SelfplayBatch>(config.workers.max(1) * 2);
            let (trainer_tx, trainer_rx) = mpsc::sync_channel::<TrainerEvent>(2);
            let shared_model = Arc::new(RwLock::new(SharedSelfplayModel {
                version: 0,
                model: selfplay_model.clone(),
            }));
            let mut arena_reference_model = selfplay_model.clone();
            let selfplay_pause =
                Arc::new((Mutex::new(SelfplayPauseState::default()), Condvar::new()));
            let mut selfplay_handles = Vec::with_capacity(config.workers.max(1));
            for worker_id in 0..config.workers.max(1) {
                let selfplay_stop = stop_requested.clone();
                let selfplay_config = config.clone();
                let selfplay_opening_positions = opening_positions.clone();
                let selfplay_tx = selfplay_tx.clone();
                let shared_model = Arc::clone(&shared_model);
                let selfplay_pause = Arc::clone(&selfplay_pause);
                selfplay_handles.push(thread::spawn(move || {
                    let mut batch_index = 0usize;
                    let mut local_version = u64::MAX;
                    let mut local_model = AzNnue::random_with_arch(
                        selfplay_config.arch(),
                        selfplay_config.seed ^ worker_id as u64,
                    );
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
                                local_model = shared.model.clone();
                                local_version = shared.version;
                            }
                        }
                        let batch_seed = selfplay_config.seed
                            ^ ((worker_id as u64).wrapping_add(1) << 32)
                            ^ (batch_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                        let loop_config = build_az_loop_config(
                            &selfplay_config,
                            batch_seed,
                            1,
                            &selfplay_opening_positions,
                        );
                        let started = Instant::now();
                        let data = generate_selfplay_data(&local_model, &loop_config);
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
            let trainer_stop = stop_requested.clone();
            let trainer_interrupted = interrupted.clone();
            let trainer_config = config.clone();
            let trainer_start_update = start_update;
            let trainer_snapshot_path = replay_snapshot_path.clone();
            let trainer_handle = thread::spawn(move || {
                let mut trainer_model = model;
                let mut trainer_pool = replay_pool;
                let mut pending = PendingTrainingData::default();
                let mut train_index = 0usize;
                let min_train_samples =
                    global_training_step_sample_count(trainer_config.batch_size);
                'training: while let Ok(batch) = selfplay_rx.recv() {
                    if let Some(pool) = trainer_pool.as_mut() {
                        pool.add_samples(batch.data.samples.clone());
                    }
                    pending.push(batch);
                    while let Ok(batch) = selfplay_rx.try_recv() {
                        if let Some(pool) = trainer_pool.as_mut() {
                            pool.add_samples(batch.data.samples.clone());
                        }
                        pending.push(batch);
                    }
                    if trainer_stop.load(Ordering::SeqCst) {
                        continue;
                    }
                    let required_selfplay_samples = if trainer_start_update == 1 && train_index == 0
                    {
                        trainer_config
                            .train_warmup_samples
                            .max(trainer_config.selfplay_samples_per_update)
                    } else {
                        trainer_config.selfplay_samples_per_update
                    };
                    if pending.selfplay.samples.len() < required_selfplay_samples {
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
                    let train_data = pool.sample_uniform_marked_capped_by(
                        trainer_config.train_samples_per_update,
                        trainer_config.train_epochs_per_update as u32,
                        trainer_config.max_sample_train_count,
                        &mut rng,
                    );
                    if train_data.is_empty() {
                        continue;
                    }
                    let train_update = trainer_start_update.saturating_add(train_index);
                    let current_lr = learning_rate_for_update(&trainer_config, train_update);
                    let train_started = Instant::now();
                    let stats = train_samples_weighted(
                        &mut trainer_model,
                        &train_data,
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
                        std::mem::take(&mut pending),
                        stats,
                        current_lr,
                        train_data.len(),
                        train_seconds,
                        pool.sample_count(),
                        pool.capacity(),
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
                    while let Ok(batch) = selfplay_rx.try_recv() {
                        if let Some(pool) = trainer_pool.as_mut() {
                            pool.add_samples(batch.data.samples.clone());
                        }
                        pending.push(batch);
                    }
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
                let policy_target_entropy = report.policy_ce - report.policy_kl;
                println!(
                    "update {update:04}: games={} samples={} train_samples={} pool={}/{} fill={:.0}% R/B/D={}/{}/{} red_rate={:.3} avg_plies={:.1} loss={:.4} value_mse={:.4} value_rmse={:.4} v_mu={:.3}/{:.3} v_rms={:.3}/{:.3} v_corr={:.3} v_cal={:.3} policy_ce={:.4} policy_kl={:.4} targetH={:.4} lr={:.6} rootH={:.3} openH={:.3} midH={:.3} rawP={:.3}/{:.3} tgtP={:.3}/{:.3} qgap={:.3} qabs={:.3} visitA={:.1} openRawP={:.3}/{:.3} openTgtP={:.3}/{:.3} openQgap={:.3} openQabs={:.3} openVisitA={:.1} train={:.1}s gps={:.2} sps={:.1} train_sps={:.1} elapsed={:.1}s{}",
                    report.games,
                    report.samples,
                    report.train_samples,
                    report.pool_samples,
                    report.pool_capacity,
                    if report.pool_capacity == 0 {
                        0.0
                    } else {
                        100.0 * report.pool_samples as f32 / report.pool_capacity as f32
                    },
                    report.red_wins,
                    report.black_wins,
                    report.draws,
                    report.red_wins as f32 / report.games.max(1) as f32,
                    report.avg_plies,
                    report.loss,
                    report.value_mse,
                    value_rmse,
                    report.value_pred_mean,
                    report.value_target_mean,
                    report.value_pred_rms,
                    report.value_target_rms,
                    report.value_corr,
                    report.value_calibration,
                    report.policy_ce,
                    report.policy_kl,
                    policy_target_entropy,
                    report.learning_rate,
                    report.root_visit_entropy,
                    report.entropy_opening,
                    report.entropy_mid,
                    report.raw_prior_top1,
                    report.raw_prior_top2,
                    report.policy_top1,
                    report.policy_top2,
                    report.root_q_gap,
                    report.root_q_top1_abs,
                    report.visited_actions,
                    report.opening_raw_prior_top1,
                    report.opening_raw_prior_top2,
                    report.opening_policy_top1,
                    report.opening_policy_top2,
                    report.opening_q_gap,
                    report.opening_q_top1_abs,
                    report.opening_visited_actions,
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
                log_scalar(&mut tb, "train/loss", update, report.loss);
                log_scalar(&mut tb, "train/value_loss", update, report.value_loss);
                log_scalar(&mut tb, "train/value_mse", update, report.value_mse);
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
                log_scalar(
                    &mut tb,
                    "train/value_pred_rms",
                    update,
                    report.value_pred_rms,
                );
                log_scalar(
                    &mut tb,
                    "train/value_target_rms",
                    update,
                    report.value_target_rms,
                );
                log_scalar(&mut tb, "train/value_corr", update, report.value_corr);
                log_scalar(
                    &mut tb,
                    "train/value_calibration",
                    update,
                    report.value_calibration,
                );
                log_scalar(&mut tb, "train/policy_ce", update, report.policy_ce);
                log_scalar(&mut tb, "train/policy_kl", update, report.policy_kl);
                log_scalar(
                    &mut tb,
                    "train/policy_target_entropy",
                    update,
                    policy_target_entropy,
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
                log_scalar(&mut tb, "selfplay/avg_plies", update, report.avg_plies);
                log_scalar(
                    &mut tb,
                    "stats/root_visit_entropy",
                    update,
                    report.root_visit_entropy,
                );
                log_scalar(
                    &mut tb,
                    "stats/entropy_opening",
                    update,
                    report.entropy_opening,
                );
                log_scalar(&mut tb, "stats/entropy_mid", update, report.entropy_mid);
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
                log_scalar(&mut tb, "stats/policy_top1", update, report.policy_top1);
                log_scalar(&mut tb, "stats/policy_top2", update, report.policy_top2);
                log_scalar(&mut tb, "stats/root_q_gap", update, report.root_q_gap);
                log_scalar(
                    &mut tb,
                    "stats/root_q_top1_abs",
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
                    "stats/opening_raw_prior_top1",
                    update,
                    report.opening_raw_prior_top1,
                );
                log_scalar(
                    &mut tb,
                    "stats/opening_raw_prior_top2",
                    update,
                    report.opening_raw_prior_top2,
                );
                log_scalar(
                    &mut tb,
                    "stats/opening_policy_top1",
                    update,
                    report.opening_policy_top1,
                );
                log_scalar(
                    &mut tb,
                    "stats/opening_policy_top2",
                    update,
                    report.opening_policy_top2,
                );
                log_scalar(&mut tb, "stats/opening_q_gap", update, report.opening_q_gap);
                log_scalar(
                    &mut tb,
                    "stats/opening_q_top1_abs",
                    update,
                    report.opening_q_top1_abs,
                );
                log_scalar(
                    &mut tb,
                    "stats/opening_visited_actions",
                    update,
                    report.opening_visited_actions,
                );
                log_scalar(
                    &mut tb,
                    "selfplay/games_per_second",
                    update,
                    report.games_per_second,
                );
                log_scalar(
                    &mut tb,
                    "selfplay/samples_per_second",
                    update,
                    report.samples_per_second,
                );
                log_scalar(
                    &mut tb,
                    "train/samples_per_second",
                    update,
                    report.train_samples_per_second,
                );
                log_scalar(
                    &mut tb,
                    "timing/train_seconds",
                    update,
                    report.train_seconds,
                );
                log_scalar(
                    &mut tb,
                    "timing/update_seconds",
                    update,
                    report.total_seconds,
                );
                log_scalar(&mut tb, "outcome/red_wins", update, report.red_wins as f32);
                log_scalar(
                    &mut tb,
                    "outcome/red_win_rate",
                    update,
                    report.red_wins as f32 / report.games.max(1) as f32,
                );
                log_scalar(
                    &mut tb,
                    "outcome/black_wins",
                    update,
                    report.black_wins as f32,
                );
                log_scalar(
                    &mut tb,
                    "outcome/black_win_rate",
                    update,
                    report.black_wins as f32 / report.games.max(1) as f32,
                );
                log_scalar(&mut tb, "outcome/draws", update, report.draws as f32);
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
                if config.arena_interval == 0 {
                    let mut shared = shared_model
                        .write()
                        .unwrap_or_else(|_| panic!("shared selfplay model poisoned"));
                    shared.model = candidate_model.clone();
                    shared.version = shared.version.wrapping_add(1);
                } else if update.is_multiple_of(config.arena_interval) {
                    {
                        let (pause_lock, _) = &*selfplay_pause;
                        let mut pause_state = pause_lock
                            .lock()
                            .unwrap_or_else(|_| panic!("selfplay pause state poisoned"));
                        pause_state.arena_paused = true;
                    }
                    println!("pause    : selfplay paused for arena");
                    let pikafish_arena_enabled = !config.arena_pikafish_exe.trim().is_empty();
                    let use_pikafish_arena =
                        pikafish_arena_enabled && update >= config.arena_pikafish_start_update;
                    if pikafish_arena_enabled && !use_pikafish_arena {
                        arena_reference_model = candidate_model.clone();
                        {
                            let mut shared = shared_model
                                .write()
                                .unwrap_or_else(|_| panic!("shared selfplay model poisoned"));
                            shared.model = candidate_model.clone();
                            shared.version = shared.version.wrapping_add(1);
                        }
                        save_model(&candidate_model, &best_path);
                        println!(
                            "arena-warmup {update:04}: before_pikafish_start={} adopt=current saved_best",
                            config.arena_pikafish_start_update
                        );
                        log_scalar(&mut tb, "arena_warmup/adopted_latest", update, 1.0);
                    } else if use_pikafish_arena {
                        let candidate_path = arena_pikafish_candidate_path(
                            &config.model_path,
                            &config.checkpoint_dir,
                            update,
                        );
                        save_model(&candidate_model, &candidate_path);
                        let arena_eval_positions =
                            load_arena_eval_positions(&config.arena_pikafish_eval_fens);
                        let arena_eval_fens = arena_eval_positions.len();
                        let eval_pikafish_depth = arena_pikafish_depth.max(1);
                        let summary = run_vs_pikafish(
                            Path::new(&config.arena_pikafish_exe),
                            &candidate_path,
                            &arena_eval_positions,
                            VsPikafishConfig {
                                pikafish_depth: eval_pikafish_depth,
                                total_games: config.arena_pikafish_games,
                                max_plies: config.max_plies,
                                simulations: config.simulations,
                                seed: config.seed
                                    ^ (update as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                                parallel_games: config.arena_pikafish_parallel_games,
                                cpuct: config.arena_cpuct,
                            },
                        )
                        .unwrap_or_else(|err| panic!("arena pikafish failed: {err}"));
                        let wins = summary.chinese_wins;
                        let win_rate = wins as f32 / summary.total_games.max(1) as f32;
                        let promoted = wins >= arena_pikafish_best_wins;
                        let old_best_wins = arena_pikafish_best_wins;
                        if promoted {
                            arena_reference_model = candidate_model.clone();
                            {
                                let mut shared = shared_model
                                    .write()
                                    .unwrap_or_else(|_| panic!("shared selfplay model poisoned"));
                                shared.model = candidate_model.clone();
                                shared.version = shared.version.wrapping_add(1);
                            }
                            save_model(&candidate_model, &best_path);
                            arena_pikafish_best_wins = wins;
                            if win_rate > config.arena_pikafish_promotion_rate {
                                arena_pikafish_depth = arena_pikafish_depth.saturating_add(1);
                                arena_pikafish_best_wins = 0;
                            }
                        }
                        println!(
                            "arena-pikafish {update:04}: model={} total={} fens={} W/L/D={}/{}/{} as_red={} as_black={} win_rate={:.3} min_wins={} depth={} promote_at_wins>=previous promote_depth_at>{:.3}{} next_depth={} next_min_wins={}",
                            candidate_path.display(),
                            summary.total_games,
                            arena_eval_fens,
                            summary.chinese_wins,
                            summary.chinese_losses,
                            summary.draws,
                            summary.chinese_wins_as_red,
                            summary.chinese_wins_as_black,
                            win_rate,
                            old_best_wins,
                            eval_pikafish_depth,
                            config.arena_pikafish_promotion_rate,
                            if promoted {
                                " promoted=current saved_best"
                            } else {
                                ""
                            },
                            arena_pikafish_depth,
                            arena_pikafish_best_wins
                        );
                        log_scalar(
                            &mut tb,
                            "arena_pikafish/wins",
                            update,
                            summary.chinese_wins as f32,
                        );
                        log_scalar(
                            &mut tb,
                            "arena_pikafish/losses",
                            update,
                            summary.chinese_losses as f32,
                        );
                        log_scalar(
                            &mut tb,
                            "arena_pikafish/draws",
                            update,
                            summary.draws as f32,
                        );
                        log_scalar(&mut tb, "arena_pikafish/win_rate", update, win_rate);
                        log_scalar(
                            &mut tb,
                            "arena_pikafish/depth",
                            update,
                            arena_pikafish_depth as f32,
                        );
                        log_scalar(
                            &mut tb,
                            "arena_pikafish/best_wins",
                            update,
                            arena_pikafish_best_wins as f32,
                        );
                        log_scalar(
                            &mut tb,
                            "arena_pikafish/promoted",
                            update,
                            if promoted { 1.0 } else { 0.0 },
                        );
                        let _ = fs::remove_file(&candidate_path);
                    } else {
                        let arena_eval_positions =
                            load_arena_eval_positions(&config.arena_pikafish_eval_fens);
                        let arena_eval_fens = arena_eval_positions.len();
                        let arena_mode = if arena_eval_fens == 0 {
                            "startpos_fallback"
                        } else {
                            "all_fens"
                        };
                        let candidate = Arc::new(candidate_model.clone());
                        let baseline = Arc::new(arena_reference_model.clone());
                        let arena = run_arena_threads(ArenaThreadConfig {
                            candidate,
                            baseline,
                            eval_positions: Arc::new(arena_eval_positions),
                            simulations: config.simulations,
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
                            arena_reference_model = candidate_model.clone();
                            {
                                let mut shared = shared_model
                                    .write()
                                    .unwrap_or_else(|_| panic!("shared selfplay model poisoned"));
                                shared.model = candidate_model.clone();
                                shared.version = shared.version.wrapping_add(1);
                            }
                            save_model(&candidate_model, &best_path);
                            arena_best_elo = candidate_elo;
                        }
                        println!(
                            "arena {update:04}: mode={} total={} fens={} W/L/D={}/{}/{} red={}/{} black={}/{} score={:.1} rate={:.3} se={:.3} lcb={:.3} promote_at={:.3} z={:.2} ref_elo={:.1} elo={:.1} elo_diff={:+.1} best_ref=memory{}",
                            arena_mode,
                            arena.total_games(),
                            arena_eval_fens,
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
                        log_scalar(&mut tb, "arena/wins", update, arena.wins as f32);
                        log_scalar(&mut tb, "arena/losses", update, arena.losses as f32);
                        log_scalar(&mut tb, "arena/draws", update, arena.draws as f32);
                        log_scalar(&mut tb, "arena/score", update, arena.score());
                        log_scalar(&mut tb, "arena/score_rate", update, arena.score_rate());
                        log_scalar(&mut tb, "arena/score_rate_se", update, arena_se);
                        log_scalar(&mut tb, "arena/score_rate_lcb", update, arena_lcb);
                        log_scalar(&mut tb, "arena/ref_elo", update, ref_elo);
                        log_scalar(&mut tb, "arena/elo", update, candidate_elo);
                        log_scalar(&mut tb, "arena/elo_diff", update, elo_diff);
                        log_scalar(
                            &mut tb,
                            "arena/win_rate",
                            update,
                            arena.wins as f32 / arena.total_games().max(1) as f32,
                        );
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
            trainer_handle
                .join()
                .unwrap_or_else(|_| panic!("training thread panicked"));
            if exited_after_ctrl_c {
                while let Ok(event) = trainer_rx.try_recv() {
                    interrupt_save_model = Some(event.candidate_model);
                    interrupt_save_next_update = update.saturating_add(1);
                    update = update.saturating_add(1);
                }
            }
            if exited_after_ctrl_c || exited_after_target_update {
                if let Some(model) = interrupt_save_model.as_ref() {
                    save_model(model, Path::new(&config.model_path));
                    save_az_loop_progress_pair(
                        &config_path,
                        interrupt_save_next_update,
                        arena_best_elo,
                        arena_pikafish_depth,
                        arena_pikafish_best_wins,
                    );
                    println!(
                        "model    : {} save `{}` next_update={}",
                        if exited_after_target_update {
                            "target"
                        } else {
                            "interrupt"
                        },
                        config.model_path,
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
        Some(CliCommand::AzBaseline100(cmd)) => {
            run_baseline_100(cmd);
        }
        Some(CliCommand::PikafishLabel(cmd)) => {
            run_pikafish_label(cmd);
        }
        Some(CliCommand::AzTeacherProbe(cmd)) => {
            run_az_teacher_probe(cmd);
        }
        Some(CliCommand::AzCollectFens(cmd)) => {
            run_az_collect_fens(cmd);
        }
        Some(CliCommand::VsPikafish(cmd)) => {
            let pikafish_exe = cmd.pikafish_exe;
            let model_path = cmd.model;
            let simulations = cmd.simulations.unwrap_or(192).max(1);
            let cpuct = cmd.cpuct.max(0.0);
            let max_plies = cmd.max_plies.max(1);
            let pikafish_depth = cmd.pikafish_depth.max(1);
            let games = cmd.games.max(1);
            let parallel_games = cmd.parallel_games.max(1);
            let eval_fens_path = cmd.eval_fens_path;
            let start_positions = load_arena_eval_positions(&eval_fens_path);
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
                "vs-pikafish: model={} search=alphazero games={} fens={} parallel={} chinese W/L/D={}/{}/{} (as_red={} as_black={}) win_reasons(general_capture={} checkmate_no_legal_moves={} rule={} pikafish_no_bestmove={} pikafish_invalid_move={} pikafish_illegal_move={}) | pikafish_depth={} max_plies={} sims={} cpuct={}",
                model_path,
                summary.total_games,
                start_positions.len(),
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
                cpuct
            );
        }
    };
    chineseai::profile::print_report();
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

#[derive(Clone, Debug)]
struct PikafishPvLine {
    multipv: usize,
    score_cp: i32,
    bestmove: String,
}

#[derive(Clone, Debug)]
struct PikafishLabel {
    fen: String,
    bestmove: String,
    score_cp: i32,
    multipv: Vec<PikafishPvLine>,
}

struct PikafishLabeler {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl PikafishLabeler {
    fn start(path: &Path, threads: usize, multipv: usize) -> io::Result<Self> {
        let mut child = Command::new(path)
            .current_dir(path.parent().unwrap_or_else(|| Path::new(".")))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| io::Error::other("pikafish: missing stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("pikafish: missing stdout"))?;
        let mut engine = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        };
        engine.write_line("uci")?;
        engine.read_until("uciok")?;
        engine.write_line(&format!("setoption name Threads value {}", threads.max(1)))?;
        engine.write_line(&format!("setoption name MultiPV value {}", multipv.max(1)))?;
        engine.write_line("isready")?;
        engine.read_until("readyok")?;
        Ok(engine)
    }

    fn write_line(&mut self, line: &str) -> io::Result<()> {
        writeln!(self.stdin, "{line}")?;
        self.stdin.flush()
    }

    fn read_until(&mut self, token: &str) -> io::Result<()> {
        let mut line = String::new();
        loop {
            line.clear();
            if self.stdout.read_line(&mut line)? == 0 {
                return Err(io::Error::other(format!("pikafish: EOF before `{token}`")));
            }
            if line.trim() == token {
                return Ok(());
            }
        }
    }

    fn query(&mut self, fen: &str, movetime_ms: u64) -> io::Result<PikafishLabel> {
        self.write_line(&format!("position fen {fen}"))?;
        self.write_line(&format!("go movetime {}", movetime_ms.max(1)))?;
        let mut by_multipv = std::collections::BTreeMap::<usize, PikafishPvLine>::new();
        let mut line = String::new();
        let bestmove = loop {
            line.clear();
            if self.stdout.read_line(&mut line)? == 0 {
                return Err(io::Error::other("pikafish: EOF before bestmove"));
            }
            let trimmed = line.trim();
            if let Some(pv) = parse_pikafish_info(trimmed) {
                by_multipv.insert(pv.multipv, pv);
            }
            if let Some(rest) = trimmed.strip_prefix("bestmove ") {
                break rest
                    .split_whitespace()
                    .next()
                    .unwrap_or("(none)")
                    .to_string();
            }
        };
        let multipv = by_multipv.into_values().collect::<Vec<_>>();
        let score_cp = multipv
            .iter()
            .find(|pv| pv.multipv == 1)
            .map(|pv| pv.score_cp)
            .unwrap_or(0);
        Ok(PikafishLabel {
            fen: fen.to_string(),
            bestmove,
            score_cp,
            multipv,
        })
    }
}

impl Drop for PikafishLabeler {
    fn drop(&mut self) {
        let _ = self.write_line("quit");
        let _ = self.child.wait();
    }
}

fn parse_pikafish_info(line: &str) -> Option<PikafishPvLine> {
    if !line.starts_with("info ") || !line.contains(" pv ") || !line.contains(" score ") {
        return None;
    }
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    let mut multipv = 1usize;
    let mut score_cp = None;
    let mut bestmove = None;
    let mut index = 0usize;
    while index < tokens.len() {
        match tokens[index] {
            "multipv" if index + 1 < tokens.len() => {
                multipv = tokens[index + 1].parse().ok()?;
                index += 2;
            }
            "score" if index + 2 < tokens.len() => {
                let raw = tokens[index + 2].parse::<i32>().ok()?;
                score_cp = Some(match tokens[index + 1] {
                    "cp" => raw,
                    "mate" => {
                        if raw >= 0 {
                            32_000 - raw.min(1000)
                        } else {
                            -32_000 - raw.max(-1000)
                        }
                    }
                    _ => return None,
                });
                index += 3;
            }
            "pv" if index + 1 < tokens.len() => {
                bestmove = Some(tokens[index + 1].to_string());
                break;
            }
            _ => index += 1,
        }
    }
    Some(PikafishPvLine {
        multipv,
        score_cp: score_cp?,
        bestmove: bestmove?,
    })
}

fn read_fen_lines(path: &str) -> Vec<String> {
    let text = fs::read_to_string(path).unwrap_or_else(|err| {
        panic!("failed to read `{path}`: {err}");
    });
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_string)
        .collect()
}

fn tsv_clean(text: &str) -> String {
    text.replace(['\t', '\r', '\n'], " ")
}

fn format_multipv_tsv(items: &[PikafishPvLine]) -> String {
    items
        .iter()
        .map(|pv| format!("{}:{}:{}", pv.multipv, pv.bestmove, pv.score_cp))
        .collect::<Vec<_>>()
        .join(";")
}

fn run_pikafish_label(cmd: PikafishLabelArgs) {
    let fens = read_fen_lines(&cmd.input);
    let mut engine = PikafishLabeler::start(Path::new(&cmd.pikafish_exe), cmd.threads, cmd.multipv)
        .unwrap_or_else(|err| panic!("failed to start Pikafish `{}`: {err}", cmd.pikafish_exe));
    let mut out = String::from("fen\tbestmove\tscore_cp\tmultipv\n");
    for (index, fen) in fens.iter().enumerate() {
        let label = engine.query(fen, cmd.movetime).unwrap_or_else(|err| {
            panic!("pikafish label failed at line {} `{fen}`: {err}", index + 1);
        });
        eprintln!(
            "label {}/{}: bestmove={} score_cp={}",
            index + 1,
            fens.len(),
            label.bestmove,
            label.score_cp
        );
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\n",
            tsv_clean(&label.fen),
            label.bestmove,
            label.score_cp,
            format_multipv_tsv(&label.multipv)
        ));
    }
    if let Some(output) = cmd.output {
        fs::write(&output, out).unwrap_or_else(|err| {
            panic!("failed to write `{output}`: {err}");
        });
        println!("pikafish_label : wrote {output}");
        println!("positions      : {}", fens.len());
    } else {
        print!("{out}");
    }
}

fn run_az_collect_fens(cmd: AzCollectFensArgs) {
    let target_positions = cmd.positions.max(1);
    let mut seen = BTreeSet::new();
    let mut rows = Vec::new();
    let max_attempts = if cmd.max_attempts == 0 {
        target_positions.saturating_mul(50).max(1)
    } else {
        cmd.max_attempts
    };
    let workers = cmd.workers.max(1);
    let attempts_per_worker = max_attempts.div_ceil(workers);
    let stop = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel::<CollectedFenCandidate>();
    let mut handles = Vec::with_capacity(workers);

    for worker_id in 0..workers {
        let worker_cmd = cmd.clone();
        let worker_tx = tx.clone();
        let worker_stop = Arc::clone(&stop);
        handles.push(thread::spawn(move || {
            collect_fens_worker(
                worker_id,
                worker_cmd,
                attempts_per_worker,
                worker_stop,
                worker_tx,
            );
        }));
    }
    drop(tx);

    for candidate in rx {
        if rows.len() >= target_positions {
            stop.store(true, Ordering::Relaxed);
            break;
        }
        if !seen.insert(candidate.label.fen.clone()) {
            continue;
        }
        eprintln!(
            "collect {}/{} worker={} attempt={} prefix={} takeover={} bestmove={} score_cp={}",
            rows.len() + 1,
            target_positions,
            candidate.worker_id,
            candidate.attempt,
            candidate.random_prefix,
            candidate.takeover_plies,
            candidate.label.bestmove,
            candidate.label.score_cp
        );
        rows.push(CollectedFenRow {
            label: candidate.label,
            worker_id: candidate.worker_id,
            attempt: candidate.attempt,
            random_prefix: candidate.random_prefix,
            takeover_plies: candidate.takeover_plies,
            final_ply: candidate.final_ply,
        });
    }
    stop.store(true, Ordering::Relaxed);
    for handle in handles {
        handle
            .join()
            .unwrap_or_else(|_| panic!("az-collect-fens worker panicked"));
    }

    write_collected_teacher_tsv(&cmd.output, &rows);
    let prefix_min = cmd.random_prefix_min.min(cmd.random_prefix_max);
    let prefix_max = cmd.random_prefix_max.max(cmd.random_prefix_min);
    eprintln!("collect_fens_summary:");
    eprintln!("  pikafish={}", cmd.pikafish_exe);
    eprintln!("  output={}", cmd.output);
    eprintln!("  positions={}", rows.len());
    eprintln!("  attempts={max_attempts}");
    eprintln!("  workers={workers}");
    eprintln!("  random_prefix={prefix_min}-{prefix_max}");
    eprintln!("  takeover_plies={}", cmd.takeover_plies);
    eprintln!("  movetime={}", cmd.movetime.max(1));
}

#[derive(Clone, Debug)]
struct CollectedFenRow {
    label: PikafishLabel,
    worker_id: usize,
    attempt: usize,
    random_prefix: usize,
    takeover_plies: usize,
    final_ply: usize,
}

#[derive(Clone, Debug)]
struct CollectedFenCandidate {
    label: PikafishLabel,
    worker_id: usize,
    attempt: usize,
    random_prefix: usize,
    takeover_plies: usize,
    final_ply: usize,
}

fn write_collected_teacher_tsv(output: &str, rows: &[CollectedFenRow]) {
    let mut out = String::from(
        "fen\tbestmove\tscore_cp\tmultipv\tsuite\tsplit\tcase_id\trandom_prefix\ttakeover_plies\tfinal_ply\tworker\tattempt\n",
    );
    for (index, row) in rows.iter().enumerate() {
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\trandom_pikafish_takeover\tseed\tcase-{:06}\t{}\t{}\t{}\t{}\t{}\n",
            tsv_clean(&row.label.fen),
            row.label.bestmove,
            row.label.score_cp,
            format_multipv_tsv(&row.label.multipv),
            index + 1,
            row.random_prefix,
            row.takeover_plies,
            row.final_ply,
            row.worker_id,
            row.attempt
        ));
    }
    fs::write(output, out).unwrap_or_else(|err| {
        panic!("failed to write `{output}`: {err}");
    });
}

fn collect_fens_worker(
    worker_id: usize,
    cmd: AzCollectFensArgs,
    attempts: usize,
    stop: Arc<AtomicBool>,
    tx: mpsc::Sender<CollectedFenCandidate>,
) {
    let mut engine =
        match PikafishLabeler::start(Path::new(&cmd.pikafish_exe), cmd.threads, cmd.multipv) {
            Ok(engine) => engine,
            Err(err) => {
                eprintln!(
                    "az-collect-fens worker {worker_id}: failed to start Pikafish `{}`: {err}",
                    cmd.pikafish_exe
                );
                return;
            }
        };
    let prefix_min = cmd.random_prefix_min.min(cmd.random_prefix_max);
    let prefix_max = cmd.random_prefix_max.max(cmd.random_prefix_min);
    let mut rng =
        SplitMix64::new(cmd.seed ^ ((worker_id as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)));

    for attempt in 1..=attempts {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let mut position = Position::startpos();
        let mut rule_history = position.initial_rule_history();
        let prefix_len = random_usize_inclusive(&mut rng, prefix_min, prefix_max);
        let mut ok = play_random_prefix(&mut position, &mut rule_history, prefix_len, &mut rng);
        if ok {
            ok = play_pikafish_takeover(
                &mut engine,
                &mut position,
                &mut rule_history,
                cmd.takeover_plies,
                cmd.movetime,
            );
        }
        if !ok {
            continue;
        }
        let fen = position.to_fen();
        let label = match engine.query(&fen, cmd.movetime) {
            Ok(label) => label,
            Err(err) => {
                eprintln!(
                    "az-collect-fens worker {worker_id}: final label failed at attempt {attempt}: {err}"
                );
                continue;
            }
        };
        if label.bestmove == "(none)" {
            continue;
        }
        if tx
            .send(CollectedFenCandidate {
                label,
                worker_id,
                attempt,
                random_prefix: prefix_len,
                takeover_plies: cmd.takeover_plies,
                final_ply: prefix_len + cmd.takeover_plies,
            })
            .is_err()
        {
            break;
        }
    }
}

fn play_random_prefix(
    position: &mut Position,
    rule_history: &mut Vec<chineseai::xiangqi::RuleHistoryEntry>,
    plies: usize,
    rng: &mut SplitMix64,
) -> bool {
    for _ in 0..plies {
        let legal = position.legal_moves_with_rules(rule_history);
        let Some(mv) = random_legal_move(&legal, rng) else {
            return false;
        };
        rule_history.push(position.rule_history_entry_after_move(mv));
        position.make_move(mv);
        if position.rule_outcome_with_history(rule_history).is_some() {
            return false;
        }
    }
    true
}

fn play_pikafish_takeover(
    engine: &mut PikafishLabeler,
    position: &mut Position,
    rule_history: &mut Vec<chineseai::xiangqi::RuleHistoryEntry>,
    plies: usize,
    movetime: u64,
) -> bool {
    for _ in 0..plies {
        let legal = position.legal_moves_with_rules(rule_history);
        if legal.is_empty() {
            return false;
        }
        let fen = position.to_fen();
        let label = match engine.query(&fen, movetime) {
            Ok(label) => label,
            Err(_) => return false,
        };
        let Some(mv) = position.parse_uci_move(&label.bestmove) else {
            return false;
        };
        if !legal.contains(&mv) {
            return false;
        }
        rule_history.push(position.rule_history_entry_after_move(mv));
        position.make_move(mv);
        if position.rule_outcome_with_history(rule_history).is_some() {
            return false;
        }
    }
    true
}

fn random_legal_move(legal: &[Move], rng: &mut SplitMix64) -> Option<Move> {
    if legal.is_empty() {
        None
    } else {
        Some(legal[(rng.next_u64() as usize) % legal.len()])
    }
}

fn random_usize_inclusive(rng: &mut SplitMix64, min: usize, max: usize) -> usize {
    if max <= min {
        min
    } else {
        min + (rng.next_u64() as usize % (max - min + 1))
    }
}

#[derive(Clone, Debug)]
struct TeacherLabel {
    fen: String,
    bestmove: String,
    score_cp: Option<i32>,
    multipv: String,
    suite: String,
    split: String,
    case_id: String,
}

fn read_teacher_labels(path: &str) -> Vec<TeacherLabel> {
    let text = fs::read_to_string(path).unwrap_or_else(|err| {
        panic!("failed to read `{path}`: {err}");
    });
    let mut labels = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts = trimmed.split('\t').collect::<Vec<_>>();
        if parts.first().is_some_and(|first| *first == "fen") {
            continue;
        }
        if parts.len() < 2 {
            panic!(
                "teacher TSV line {} needs at least fen and bestmove",
                line_index + 1
            );
        }
        labels.push(TeacherLabel {
            fen: parts[0].to_string(),
            bestmove: parts[1].to_string(),
            score_cp: parts.get(2).and_then(|value| value.parse().ok()),
            multipv: parts.get(3).copied().unwrap_or("").to_string(),
            suite: parts.get(4).copied().unwrap_or("default").to_string(),
            split: parts.get(5).copied().unwrap_or("all").to_string(),
            case_id: parts
                .get(6)
                .copied()
                .map(str::to_string)
                .unwrap_or_else(|| format!("case-{}", labels.len() + 1)),
        });
    }
    labels
}

fn rank_move_by<F>(candidates: &[chineseai::az::AzCandidate], mv: Move, mut less: F) -> usize
where
    F: FnMut(&chineseai::az::AzCandidate, &chineseai::az::AzCandidate) -> std::cmp::Ordering,
{
    let mut sorted = candidates.to_vec();
    sorted.sort_by(|left, right| less(left, right));
    sorted
        .iter()
        .position(|candidate| candidate.mv == mv)
        .map(|index| index + 1)
        .unwrap_or(0)
}

fn run_az_teacher_probe(cmd: AzTeacherProbeArgs) {
    let labels = read_teacher_labels(&cmd.teacher);
    let model = AzNnue::load(&cmd.model).unwrap_or_else(|err| {
        panic!("failed to load `{}`: {err}", cmd.model);
    });
    let simulations = cmd.simulations.max(1);
    let cpuct = cmd.cpuct.max(0.0);

    println!(
        "fen\tteacher_best\tteacher_cp\tmodel_best\tmodel_value_cp\tvalue_error_cp\tteacher_prior\tteacher_policy\tteacher_visits\tteacher_q\trank_prior\trank_policy\trank_visits\trank_q\tfound\tsuite\tsplit\tcase_id\tmultipv"
    );
    let mut found = 0usize;
    let mut best_matches = 0usize;
    let mut value_abs_error_sum = 0f64;
    let mut value_error_count = 0usize;
    let mut rank_visits_sum = 0usize;
    let mut rank_visits_count = 0usize;
    let mut group_stats = BTreeMap::<String, TeacherProbeGroupStats>::new();
    for label in &labels {
        let position = parse_position(&label.fen);
        let Some(teacher_move) = position.parse_uci_move(&label.bestmove) else {
            eprintln!(
                "skip invalid teacher move `{}` for fen `{}`",
                label.bestmove, label.fen
            );
            continue;
        };
        let result = alphazero_search(
            &position,
            &model,
            fixed_az_search_limits(simulations, 0, cpuct, 0),
        );
        let model_best = result
            .best_move
            .map(|mv| mv.to_string())
            .unwrap_or_else(|| "(none)".into());
        if model_best == label.bestmove {
            best_matches += 1;
        }
        let candidate = result
            .candidates
            .iter()
            .find(|candidate| candidate.mv == teacher_move);
        let target_found = candidate.is_some();
        if target_found {
            found += 1;
        }
        let rank_prior = rank_move_by(&result.candidates, teacher_move, |left, right| {
            right.prior.total_cmp(&left.prior)
        });
        let rank_policy = rank_move_by(&result.candidates, teacher_move, |left, right| {
            right.policy.total_cmp(&left.policy)
        });
        let rank_visits = rank_move_by(&result.candidates, teacher_move, |left, right| {
            right.visits.cmp(&left.visits)
        });
        let rank_q = rank_move_by(&result.candidates, teacher_move, |left, right| {
            right.q.total_cmp(&left.q)
        });
        if rank_visits > 0 {
            rank_visits_sum += rank_visits;
            rank_visits_count += 1;
        }
        let value_error = label
            .score_cp
            .map(|teacher_cp| result.value_cp - teacher_cp);
        if let Some(error) = value_error {
            value_abs_error_sum += (error as f64).abs();
            value_error_count += 1;
        }
        let group_key = format!("{}/{}", label.suite, label.split);
        group_stats.entry(group_key).or_default().add(
            target_found,
            model_best == label.bestmove,
            rank_prior,
            rank_policy,
            rank_visits,
            value_error,
        );
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{}\t{:.4}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            tsv_clean(&label.fen),
            label.bestmove,
            label
                .score_cp
                .map(|value| value.to_string())
                .unwrap_or_else(|| "".into()),
            model_best,
            result.value_cp,
            value_error
                .map(|value| value.to_string())
                .unwrap_or_else(|| "".into()),
            candidate.map(|candidate| candidate.prior).unwrap_or(0.0),
            candidate.map(|candidate| candidate.policy).unwrap_or(0.0),
            candidate.map(|candidate| candidate.visits).unwrap_or(0),
            candidate.map(|candidate| candidate.q).unwrap_or(0.0),
            rank_prior,
            rank_policy,
            rank_visits,
            rank_q,
            target_found,
            tsv_clean(&label.suite),
            tsv_clean(&label.split),
            tsv_clean(&label.case_id),
            tsv_clean(&label.multipv)
        );
    }
    let count = labels.len().max(1);
    eprintln!("teacher_probe_summary:");
    eprintln!("  model={}", cmd.model);
    eprintln!("  teacher={}", cmd.teacher);
    eprintln!("  positions={}", labels.len());
    eprintln!("  teacher_move_found={found}");
    eprintln!(
        "  bestmove_match_rate={:.4}",
        best_matches as f64 / count as f64
    );
    if value_error_count > 0 {
        eprintln!(
            "  value_mae_cp={:.1}",
            value_abs_error_sum / value_error_count as f64
        );
    }
    if rank_visits_count > 0 {
        eprintln!(
            "  avg_teacher_rank_by_visits={:.2}",
            rank_visits_sum as f64 / rank_visits_count as f64
        );
    }
    if !group_stats.is_empty() {
        eprintln!("teacher_probe_groups:");
        for (group, stats) in group_stats {
            eprintln!(
                "  {group}: positions={} found={} bestmatch={:.4} avg_rank_prior={:.2} avg_rank_policy={:.2} avg_rank_visits={:.2} value_mae_cp={}",
                stats.positions,
                stats.found,
                stats.best_matches as f64 / stats.positions.max(1) as f64,
                stats.avg_rank_prior(),
                stats.avg_rank_policy(),
                stats.avg_rank_visits(),
                stats
                    .value_mae_cp()
                    .map(|value| format!("{value:.1}"))
                    .unwrap_or_else(|| "n/a".into())
            );
        }
    }
}

#[derive(Default)]
struct TeacherProbeGroupStats {
    positions: usize,
    found: usize,
    best_matches: usize,
    rank_prior_sum: usize,
    rank_prior_count: usize,
    rank_policy_sum: usize,
    rank_policy_count: usize,
    rank_visits_sum: usize,
    rank_visits_count: usize,
    value_abs_error_sum: f64,
    value_error_count: usize,
}

impl TeacherProbeGroupStats {
    fn add(
        &mut self,
        found: bool,
        best_match: bool,
        rank_prior: usize,
        rank_policy: usize,
        rank_visits: usize,
        value_error: Option<i32>,
    ) {
        self.positions += 1;
        self.found += usize::from(found);
        self.best_matches += usize::from(best_match);
        if rank_prior > 0 {
            self.rank_prior_sum += rank_prior;
            self.rank_prior_count += 1;
        }
        if rank_policy > 0 {
            self.rank_policy_sum += rank_policy;
            self.rank_policy_count += 1;
        }
        if rank_visits > 0 {
            self.rank_visits_sum += rank_visits;
            self.rank_visits_count += 1;
        }
        if let Some(error) = value_error {
            self.value_abs_error_sum += (error as f64).abs();
            self.value_error_count += 1;
        }
    }

    fn avg_rank_prior(&self) -> f64 {
        self.rank_prior_sum as f64 / self.rank_prior_count.max(1) as f64
    }

    fn avg_rank_policy(&self) -> f64 {
        self.rank_policy_sum as f64 / self.rank_policy_count.max(1) as f64
    }

    fn avg_rank_visits(&self) -> f64 {
        self.rank_visits_sum as f64 / self.rank_visits_count.max(1) as f64
    }

    fn value_mae_cp(&self) -> Option<f64> {
        (self.value_error_count > 0)
            .then(|| self.value_abs_error_sum / self.value_error_count as f64)
    }
}
