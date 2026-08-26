#[cfg(all(target_os = "linux", not(target_env = "musl")))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

mod az_loop_config;

use az_loop_config::{AzLoopFileConfig, DEFAULT_AZ_LOOP_CONFIG, load_or_create_az_loop_config};

use chineseai::version::AZ_LOOP_PROGRESS_VERSION;

use byteorder::{LittleEndian, WriteBytesExt};
use chineseai::{
    az::{
        AzArenaConfig, AzArenaReport, AzExperiencePool, AzLoopConfig, AzLoopReport, AzMidgamePool,
        AzNnue, AzSampleMeta, AzSearchLimits, AzSelfplayData, AzTrainLossWeights, AzTrainingSample,
        DENSE_MOVE_SPACE, SplitMix64, alphazero_search, alphazero_search_trace_with_rules,
        alphazero_search_with_rules, benchmark_training, dense_move_index, evaluate_policy_groups,
        generate_selfplay_data, play_arena_games_from_positions, train_samples_weighted,
        train_samples_weighted_owned,
    },
    nnue::{canonical_move, extract_sparse_features_az},
    opening_book::ObkBook,
    pikafish_match::{VsPikafishConfig, run_vs_pikafish},
    xiangqi::{Move, Position, RuleOutcome},
};
use clap::{Args, CommandFactory, Parser, Subcommand};
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fs, io,
    io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write},
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
    long_about = "ChineseAI AZ-NNUE search and training tools."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<CliCommand>,
}

#[derive(Subcommand, Debug)]
enum CliCommand {
    /// Create a random AZ-NNUE model.
    AzInit(AzInitArgs),
    /// Search one position and print policy/debug details.
    AzSearch(AzSearchArgs),
    /// Benchmark fixed-position search speed.
    AzBench(AzBenchArgs),
    /// Benchmark a synthetic training workload.
    AzTrainBench(AzTrainBenchArgs),
    /// Fit a model on a fixed replay snapshot and report future-game validation loss.
    AzReplayFit(AzReplayFitArgs),
    /// Report start-position policy targets, played moves, and outcomes from a replay snapshot.
    AzReplayOpeningStats(AzReplayOpeningStatsArgs),
    /// Run self-play training from a TOML config.
    AzLoop(AzLoopArgs),
    /// Evaluate checkpoint non-transitivity and historical regressions.
    CheckpointCycles(CheckpointCyclesArgs),
    /// Run ChineseAI against a Pikafish UCI engine.
    VsPikafish(VsPikafishArgs),
    /// Generate random positions and label them with Pikafish best moves.
    PikafishLabelRandom(PikafishLabelRandomArgs),
    /// Generate positions with a trained network, then label them with Pikafish.
    PikafishLabelSelfplay(PikafishLabelSelfplayArgs),
    /// Fit several current-network widths on Pikafish labels with equal wall time.
    PikafishPolicyFit(PikafishPolicyFitArgs),
    /// Export Pikafish labels as canonical features for PyTorch experiments.
    PikafishExportTorch(PikafishExportTorchArgs),
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
  chineseai az-search model.safetensors 50000 1.5 --top 12 startpos
  chineseai az-search model.safetensors 10000 --trace-move b0c2 --verify-top 3 startpos")]
struct AzSearchArgs {
    /// AZ-NNUE model path.
    model: String,
    /// Number of MCTS simulations.
    #[arg(default_value_t = 10_000)]
    simulations: usize,
    /// Non-root PUCT init.
    #[arg(default_value_t = 0.9)]
    cpuct: f32,
    /// Root PUCT init.
    #[arg(long, default_value_t = 2.0)]
    cpuct_at_root: f32,
    /// Non-root first-play urgency reduction.
    #[arg(long, default_value_t = 0.20)]
    fpu_value: f32,
    /// Root first-play urgency reduction.
    #[arg(long, default_value_t = 0.10)]
    fpu_value_at_root: f32,
    /// Divisor applied to policy logits before root search; above 1 flattens priors.
    #[arg(long, default_value_t = 1.2)]
    policy_softmax_temp: f32,
    /// Dynamic PUCT base.
    #[arg(long, default_value_t = 19652.0)]
    cpuct_base: f32,
    /// Dynamic PUCT growth factor.
    #[arg(long, default_value_t = 1.5)]
    cpuct_factor: f32,
    /// Root dynamic PUCT base.
    #[arg(long, default_value_t = 19652.0)]
    cpuct_base_at_root: f32,
    /// Root dynamic PUCT growth factor.
    #[arg(long, default_value_t = 1.5)]
    cpuct_factor_at_root: f32,
    /// Maximum search depth in plies below root; 0 keeps the MCTX default (simulations).
    #[arg(long, default_value_t = 0)]
    max_depth: usize,
    /// Draw value in Q = W - L + draw_score * D.
    #[arg(long, default_value_t = 0.0)]
    draw_score: f32,
    /// Scale non-terminal network values during search; 0 isolates policy priors.
    #[arg(long, default_value_t = 1.0)]
    value_scale: f32,
    /// Independently re-search this many top-visited root moves after making each move.
    #[arg(long, default_value_t = 0)]
    verify_top: usize,
    /// Independently re-search specific root moves (repeat the option for multiple moves).
    #[arg(long = "verify-move")]
    verify_moves: Vec<String>,
    /// Restrict the root search to these legal moves (repeat for multiple moves).
    #[arg(long = "root-move")]
    root_moves: Vec<String>,
    /// Print the most-visited continuation below this root move with network leaf values.
    #[arg(long = "trace-move")]
    trace_move: Option<String>,
    /// Simulations for every independent child verification; 0 uses the root simulation count.
    #[arg(long, default_value_t = 0)]
    verify_sims: usize,
    /// Candidate rows to display, sorted by visits; 0 displays every legal root move.
    #[arg(long, default_value_t = 20)]
    top: usize,
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
    /// Batch size per optimizer step.
    #[arg(default_value_t = 1024)]
    batch_size: usize,
    /// Learning rate.
    #[arg(default_value_t = 0.0003)]
    lr: f32,
    /// Random seed.
    #[arg(default_value_t = 20260411)]
    seed: u64,
}

#[derive(Args, Debug)]
struct AzReplayFitArgs {
    /// Replay snapshot produced by az-loop.
    replay: String,
    /// Output model path.
    #[arg(long, default_value = "replay-fit.safetensors")]
    output: String,
    /// Hidden width of the model under test.
    #[arg(long, default_value_t = 192)]
    hidden: usize,
    /// Latest replay samples retained for the experiment.
    #[arg(long, default_value_t = 300_000)]
    samples: usize,
    /// Fraction of complete games reserved from the newest end of the snapshot.
    #[arg(long, default_value_t = 0.10)]
    validation_fraction: f32,
    /// Training passes over the fixed training split.
    #[arg(long, default_value_t = 2)]
    epochs: usize,
    /// Batch size per optimizer step.
    #[arg(long, default_value_t = 1024)]
    batch_size: usize,
    /// Learning rate.
    #[arg(long, default_value_t = 0.0007)]
    lr: f32,
    /// Initialization and shuffle seed.
    #[arg(long, default_value_t = 20260802)]
    seed: u64,
}

#[derive(Args, Debug)]
struct AzReplayOpeningStatsArgs {
    /// Replay snapshot produced by az-loop.
    replay: String,
    /// Rows to display.
    #[arg(long, default_value_t = 20)]
    top: usize,
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
  chineseai checkpoint-cycles checkpoints
  chineseai checkpoint-cycles checkpoints --contains best --max-models 12 --opening-positions 100
  chineseai checkpoint-cycles checkpoints --adjacent-only --simulations 400")]
struct CheckpointCyclesArgs {
    /// Directory containing checkpoint .safetensors files.
    directory: String,
    /// Keep only filenames containing this text; empty keeps every .safetensors file.
    #[arg(long, default_value = "")]
    contains: String,
    /// Evaluate only the latest N checkpoints ordered by the filename's last number; 0 keeps all.
    #[arg(long, default_value_t = 8)]
    max_models: usize,
    /// Minimum numeric update gap between selected checkpoints; 0 disables spacing.
    #[arg(long, default_value_t = 100)]
    min_update_gap: u64,
    /// Test only consecutive checkpoints. This cannot detect three-model cycles.
    #[arg(long)]
    adjacent_only: bool,
    /// MCTS simulations per move.
    #[arg(short = 's', long, default_value_t = 400)]
    simulations: usize,
    /// Random OBK positions; every pair uses the same positions with colors swapped.
    #[arg(long, default_value_t = 50)]
    opening_positions: usize,
    /// OBK opening book. Empty uses startpos.
    #[arg(long, default_value = "opening.obk")]
    opening_book: String,
    #[arg(long, default_value_t = 6)]
    opening_plies_min: usize,
    #[arg(long, default_value_t = 10)]
    opening_plies_max: usize,
    /// Parallel arena workers for each checkpoint pair.
    #[arg(long, default_value_t = 8)]
    threads: usize,
    #[arg(long, default_value_t = 300)]
    max_plies: usize,
    /// Minimum score-rate excess over 50% used to report a directed edge or cycle.
    #[arg(long, default_value_t = 0.02)]
    cycle_margin: f32,
    /// One-sided confidence multiplier used by cycle and regression detection.
    #[arg(long, default_value_t = 1.28)]
    confidence_z: f32,
    #[arg(long, default_value_t = 20260823)]
    seed: u64,
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
    #[arg(long, default_value_t = 0.9)]
    cpuct: f32,
    /// ChineseAI root PUCT constant.
    #[arg(long, default_value_t = 2.0)]
    cpuct_at_root: f32,
    /// ChineseAI dynamic PUCT base.
    #[arg(long, default_value_t = 19652.0)]
    cpuct_base: f32,
    /// ChineseAI dynamic PUCT growth factor.
    #[arg(long, default_value_t = 1.5)]
    cpuct_factor: f32,
    /// ChineseAI root dynamic PUCT base.
    #[arg(long, default_value_t = 19652.0)]
    cpuct_base_at_root: f32,
    /// ChineseAI root dynamic PUCT growth factor.
    #[arg(long, default_value_t = 1.5)]
    cpuct_factor_at_root: f32,
    /// ChineseAI non-root first-play urgency reduction.
    #[arg(long, default_value_t = 0.2)]
    fpu_value: f32,
    /// ChineseAI root first-play urgency reduction.
    #[arg(long, default_value_t = 0.1)]
    fpu_value_at_root: f32,
    /// Divisor applied to ChineseAI policy logits before search.
    #[arg(long, default_value_t = 1.2)]
    policy_softmax_temp: f32,
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
    /// Print the final FEN and complete move list for every game.
    #[arg(long)]
    report_games: bool,
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
  chineseai pikafish-label-random ./tools/pikafish-avx2.exe --count 5000 --depth 20 --threads 16
  chineseai pikafish-label-random ./tools/pikafish-avx2.exe --fens eval/random.fens --sqlite eval/pikafish-selfplay-5000-d20.sqlite")]
struct PikafishLabelRandomArgs {
    /// Pikafish UCI executable path.
    pikafish_exe: String,
    /// Output FEN list. Existing file is reused unless --regenerate is set.
    #[arg(long, default_value = "eval/random.fens")]
    fens: String,
    /// Output SQLite labels.
    #[arg(long, default_value = "eval/pikafish-selfplay-5000-d20.sqlite")]
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
    #[arg(long, default_value_t = 20)]
    depth: u32,
    /// Independent single-threaded Pikafish workers.
    #[arg(long, default_value_t = 16)]
    threads: usize,
    /// Regenerate the FEN file even when it already exists.
    #[arg(long)]
    regenerate: bool,
}

#[derive(Args, Debug)]
struct PikafishLabelSelfplayArgs {
    pikafish_exe: String,
    model: String,
    #[arg(long, default_value = "eval/selfplay-100000.fens")]
    fens: String,
    #[arg(long, default_value = "eval/pikafish-selfplay-100000-d12.sqlite")]
    sqlite: String,
    #[arg(long, default_value_t = 100_000)]
    count: usize,
    #[arg(long, default_value_t = 64)]
    simulations: usize,
    #[arg(long, default_value_t = 128)]
    max_plies: usize,
    #[arg(long, default_value_t = 16)]
    workers: usize,
    #[arg(long, default_value_t = 12)]
    depth: u32,
    #[arg(long, default_value_t = 16)]
    pikafish_threads: usize,
    #[arg(long, default_value_t = 20260816)]
    seed: u64,
}

#[derive(Args, Debug)]
struct PikafishPolicyFitArgs {
    sqlite: String,
    #[arg(long, value_delimiter = ',', default_value = "96,128,160,192")]
    hidden: Vec<usize>,
    #[arg(long, default_value_t = 300)]
    wall_seconds: u64,
    #[arg(long, default_value_t = 1024)]
    batch_size: usize,
    #[arg(long, default_value_t = 0.0007)]
    lr: f32,
    #[arg(long, default_value_t = 0.1)]
    validation_fraction: f32,
    #[arg(long, default_value = "eval/policy-fit")]
    output_dir: String,
    #[arg(long, default_value_t = 20260816)]
    seed: u64,
}

#[derive(Args, Debug)]
struct PikafishExportTorchArgs {
    sqlite: String,
    #[arg(long, default_value = "eval/pikafish-policy.bin")]
    output: String,
    /// Include up to this many positions from each stored principal variation.
    #[arg(long, default_value_t = 0)]
    pv_plies: usize,
}

#[derive(Args, Debug)]
#[command(after_long_help = "\
Examples:
  chineseai pikafish-label-eval model.safetensors eval/pikafish-selfplay-5000-d20.sqlite --simulations 64")]
struct PikafishLabelEvalArgs {
    /// ChineseAI AZ-NNUE model path.
    model: String,
    /// SQLite labels produced by pikafish-label-random.
    sqlite: String,
    /// ChineseAI MCTS simulations per position.
    #[arg(short = 's', long, default_value_t = 64)]
    simulations: usize,
    /// ChineseAI PUCT constant.
    #[arg(long, default_value_t = 0.65)]
    cpuct: f32,
    /// ChineseAI root PUCT constant.
    #[arg(long, default_value_t = 1.5)]
    cpuct_at_root: f32,
    /// Divisor applied to policy logits before search; above 1 flattens priors.
    #[arg(long, default_value_t = 1.5)]
    policy_softmax_temp: f32,
    /// Maximum search depth in plies below root; 0 keeps the MCTS default.
    #[arg(long, default_value_t = 0)]
    max_depth: usize,
    /// Random seed.
    #[arg(long, default_value_t = 20260628)]
    seed: u64,
    /// Limit number of positions; 0 means all.
    #[arg(long, default_value_t = 0)]
    limit: usize,
    /// Parallel evaluator threads.
    #[arg(long, default_value_t = 1)]
    threads: usize,
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
#[serde(deny_unknown_fields)]
struct AzLoopProgressState {
    format_version: u32,
    next_update: usize,
    nemesis_update: Option<u64>,
}

impl Default for AzLoopProgressState {
    fn default() -> Self {
        Self {
            format_version: AZ_LOOP_PROGRESS_VERSION,
            next_update: 1,
            nemesis_update: None,
        }
    }
}

impl AzLoopProgressState {
    fn normalize(mut self) -> Self {
        if self.format_version != AZ_LOOP_PROGRESS_VERSION {
            panic!(
                "unsupported AZ loop progress version {}; expected {}",
                self.format_version, AZ_LOOP_PROGRESS_VERSION
            );
        }
        self.next_update = self.next_update.max(1);
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

fn save_az_loop_progress_pair(config_path: &str, next_update: usize, nemesis_update: Option<u64>) {
    save_az_loop_progress(
        config_path,
        &AzLoopProgressState {
            next_update,
            nemesis_update,
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
            "sim{}_sspu{}_bs{}_lr{}_h{}_mxp{}_sr{}_r60{}_wk{}_",
            "rrf{}_rrw{}_lrm{}_lds{}_ldi{}_ldf{}_cp{}_cpr{}_fv{}_fvr{}_pst{}_tb{}_teg{}_tdd{}_tde{}_tvc{}_tvo{}_tdl{}_op{}_rs{}_rp{}_rc{}_",
            "tspu{}_tepu{}_mp{}_cpi{}_ai{}_as{}_acp{}_acpr{}_apst{}_rda{}_ref{}_pef{}_pet{}_pera{}_peref{}_sd{}"
        ),
        config.simulations,
        config.selfplay_samples_per_update,
        config.batch_size,
        f32_slug(config.lr),
        config.hidden_size,
        config.max_plies,
        u8::from(config.sixty_move_rule),
        config.rule60_max_ply,
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
        f32_slug(config.value_td_lambda),
        format!(
            "{}x{}",
            f32_slug(config.opening_start_fraction),
            config.opening_reservoir_capacity
        ),
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
        f32_slug(config.arena_cpuct_at_root),
        f32_slug(config.arena_policy_softmax_temp),
        f32_slug(config.root_dirichlet_alpha),
        f32_slug(config.root_exploration_fraction),
        f32_slug(config.persistent_exploration_fraction),
        f32_slug(config.persistent_exploration_temperature),
        f32_slug(config.persistent_exploration_root_dirichlet_alpha),
        f32_slug(config.persistent_exploration_root_exploration_fraction),
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

fn champion_checkpoint_paths(model_path: &str, checkpoint_dir: &str) -> io::Result<Vec<PathBuf>> {
    let directory = Path::new(checkpoint_dir);
    if !directory.exists() {
        return Ok(Vec::new());
    }
    let base = Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.safetensors");
    let prefix = "best-update-";
    let suffix = format!("-{base}");
    let mut paths = fs::read_dir(directory)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(prefix) && name.ends_with(&suffix))
        })
        .collect::<Vec<_>>();
    paths.sort_by_key(|path| checkpoint_number(path).unwrap_or(0));
    Ok(paths)
}

fn historical_anchor_index(champion_count: usize, gate_index: usize) -> Option<usize> {
    let current = champion_count.checked_sub(1)?;
    let offsets = [2usize, 4, 8, 16, 32]
        .into_iter()
        .filter(|&offset| offset <= current)
        .collect::<Vec<_>>();
    let offset = offsets.get(gate_index % offsets.len().max(1))?;
    Some(current - offset)
}

fn arena_gate_position_counts(
    total: usize,
    has_previous: bool,
    has_anchor: bool,
) -> (usize, usize, usize) {
    let previous = has_previous.then_some(total / 5).unwrap_or(0);
    let anchor = has_anchor.then_some(total / 5).unwrap_or(0);
    (total - previous - anchor, previous, anchor)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArenaGateDecision {
    Promote,
    Continue,
    Reject,
}

fn arena_gate_decision(
    current: &AzArenaReport,
    previous: Option<&AzArenaReport>,
    anchor: Option<&AzArenaReport>,
    current_threshold: f32,
    confidence_z: f32,
) -> ArenaGateDecision {
    let z = confidence_z.max(0.0);
    let proven_current_regression = current.score_rate_upper_bound(z) < 0.50;
    let proven_history_regression = previous
        .into_iter()
        .chain(anchor)
        .any(|report| report.score_rate_upper_bound(z) < 0.50);
    let mut combined_history = AzArenaReport::default();
    for report in previous.into_iter().chain(anchor) {
        combined_history.add_assign(report);
    }
    let proven_combined_history_regression =
        combined_history.total_games() > 0 && combined_history.score_rate_upper_bound(z) < 0.50;
    if proven_current_regression || proven_history_regression || proven_combined_history_regression
    {
        return ArenaGateDecision::Reject;
    }

    if current.score_rate_lower_bound(z) > current_threshold {
        ArenaGateDecision::Promote
    } else {
        ArenaGateDecision::Continue
    }
}

fn shuffle_positions(positions: &mut [Position], rng: &mut SplitMix64) {
    for index in (1..positions.len()).rev() {
        positions.swap(index, rng.next_u64() as usize % (index + 1));
    }
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
}

struct TrainerEvent {
    report: AzLoopReport,
    candidate_model: AzNnue,
}

struct SharedSelfplayModel {
    version: u64,
    learner_update: u32,
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
}

impl SelfplayPauseState {
    fn is_paused(&self) -> bool {
        self.arena_paused
    }
}

#[derive(Default)]
struct PendingTrainingData {
    collection_seconds: f32,
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
    start_source_rate: [f32; 3],
}

impl PendingTrainingData {
    fn push(&mut self, batch: SelfplayBatch) {
        self.selfplay.add_assign(&batch.data);
    }
}

fn build_az_loop_config(
    config: &AzLoopFileConfig,
    seed: u64,
    workers: usize,
    generation_update: u32,
    opening_positions: &Arc<[chineseai::az::AzStartSnapshot]>,
) -> AzLoopConfig {
    AzLoopConfig {
        games: 1,
        max_plies: config.max_plies,
        rule60_max_ply: config.sixty_move_rule.then_some(config.rule60_max_ply),
        simulations: config.simulations,
        seed,
        workers,
        generation_update,
        temperature_start: config.temperature_start,
        temperature_endgame: config.temperature_endgame,
        persistent_exploration_fraction: config.persistent_exploration_fraction,
        persistent_exploration_temperature: config.persistent_exploration_temperature,
        persistent_exploration_root_dirichlet_alpha: config
            .persistent_exploration_root_dirichlet_alpha,
        persistent_exploration_root_exploration_fraction: config
            .persistent_exploration_root_exploration_fraction,
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
        policy_softmax_temp: config.policy_softmax_temp,
        value_td_lambda: config.value_td_lambda,
        opening_positions: Arc::clone(opening_positions),
        opening_start_fraction: config.opening_start_fraction,
        midgame_positions: Arc::default(),
        midgame_start_fraction: config.midgame_start_fraction,
        resign_percentage: config.resign_percentage,
        resign_playthrough: config.resign_playthrough,
        mirror_probability: config.mirror_probability,
        record_fens: false,
    }
}

fn build_async_training_report(
    pending: PendingTrainingData,
    selfplay_games: usize,
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
    let total_seconds = pending.collection_seconds.max(1.0e-6);
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
    let value_report = |phase_stats: chineseai::az::AzValueMomentStats| {
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
    };
    let phase_value = stats.phase_value.map(value_report);
    let source_phase_value = stats.source_phase_value.map(value_report);
    AzLoopReport {
        games: selfplay_games,
        samples: selfplay_samples,
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
        source_phase_value,
        policy_ce: stats.policy_ce,
        policy_target_entropy: train_source.policy_target_entropy,
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
        train_start_source_rate: train_source.start_source_rate,
        train_policy_target_top1: train_source.policy_target_top1,
        train_policy_target_top2: train_source.policy_target_top2,
        terminal_no_legal_moves: pending.selfplay.terminal.no_legal_moves,
        terminal_red_general_missing: pending.selfplay.terminal.red_general_missing,
        terminal_black_general_missing: pending.selfplay.terminal.black_general_missing,
        terminal_rule_draw: pending.selfplay.terminal.rule_draw,
        terminal_rule_draw_natural_limit: pending.selfplay.terminal.rule_draw_natural_limit,
        terminal_rule_draw_insufficient_material: pending
            .selfplay
            .terminal
            .rule_draw_insufficient_material,
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
    actual_recent_samples: usize,
) -> TrainBatchSourceStats {
    if samples.is_empty() {
        return TrainBatchSourceStats::default();
    }
    let full_simulations = full_simulations.max(1) as u32;
    let mut fast = 0usize;
    let mut policy_weight_sum = 0.0f32;
    let mut value_weight_sum = 0.0f32;
    let mut target_entropy_sum = 0.0f32;
    let mut target_top1_sum = 0.0f32;
    let mut target_top2_sum = 0.0f32;
    let mut start_source_count = [0usize; 3];
    for sample in samples {
        start_source_count[sample.meta.start_source.index()] += 1;
        fast += usize::from(
            sample.search_simulations > 0 && sample.search_simulations < full_simulations,
        );
        policy_weight_sum += sample.policy_weight.max(0.0);
        value_weight_sum += sample.value_weight.max(0.0);
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
        actual_recent_sample_rate: actual_recent_samples.min(samples.len()) as f32 / denom,
        policy_target_entropy: target_entropy_sum / denom,
        policy_target_top1: target_top1_sum / denom,
        policy_target_top2: target_top2_sum / denom,
        start_source_rate: start_source_count.map(|count| count as f32 / denom),
    }
}

struct ArenaThreadConfig {
    candidate: Arc<AzNnue>,
    baseline: Arc<AzNnue>,
    eval_positions: Arc<Vec<Position>>,
    simulations: usize,
    max_plies: usize,
    rule60_max_ply: Option<u16>,
    cpuct: f32,
    cpuct_at_root: f32,
    cpuct_base: f32,
    cpuct_factor: f32,
    cpuct_base_at_root: f32,
    cpuct_factor_at_root: f32,
    fpu_value: f32,
    fpu_value_at_root: f32,
    draw_score: f32,
    policy_softmax_temp: f32,
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
        let rule60_max_ply = config.rule60_max_ply;
        let cpuct = config.cpuct;
        let cpuct_at_root = config.cpuct_at_root;
        let cpuct_base = config.cpuct_base;
        let cpuct_factor = config.cpuct_factor;
        let cpuct_base_at_root = config.cpuct_base_at_root;
        let cpuct_factor_at_root = config.cpuct_factor_at_root;
        let fpu_value = config.fpu_value;
        let fpu_value_at_root = config.fpu_value_at_root;
        let draw_score = config.draw_score;
        let policy_softmax_temp = config.policy_softmax_temp;
        // 由全局开局索引派生每对随机流；结果不应随线程切分变化。
        let seed = config.seed;
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
                    rule60_max_ply,
                    games_as_red: red_games,
                    games_as_black: black_games,
                    start_index: thread_start_index,
                    seed,
                    cpuct,
                    cpuct_at_root,
                    cpuct_base,
                    cpuct_factor,
                    cpuct_base_at_root,
                    cpuct_factor_at_root,
                    fpu_value,
                    fpu_value_at_root,
                    draw_score,
                    policy_softmax_temp,
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
    // 每次门控使用新的确定性留出折，避免反复在同一小批局面上选择导致评测过拟合。
    // 折内每个局面仍严格交换红黑配对。
    let gate_index = update / config.arena_interval.max(1);
    let seed = config.seed
        ^ 0xD1B5_4A32_D192_ED03
        ^ (gate_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut positions = Vec::with_capacity(
        config
            .arena_opening_positions
            .saturating_add(config.arena_random_positions),
    );
    let mut modes = Vec::with_capacity(2);
    if !config.arena_opening_book.trim().is_empty() {
        let book = ObkBook::load(&config.arena_opening_book).unwrap_or_else(|err| {
            panic!(
                "failed to load arena opening book `{}`: {err}",
                config.arena_opening_book
            )
        });
        let mut rng = SplitMix64::new(seed);
        for _ in 0..config.arena_opening_positions {
            positions.push(book.random_prefix_position(
                config.arena_opening_plies_min,
                config.arena_opening_plies_max,
                &mut rng,
            ));
        }
        modes.push(format!(
            "obk(count={},keys={},moves={},plies={}-{})",
            config.arena_opening_positions,
            book.key_count(),
            book.move_count(),
            config.arena_opening_plies_min,
            config.arena_opening_plies_max
        ));
    }

    if config.arena_random_positions > 0 {
        let random_fens = generate_random_eval_fens(
            config.arena_random_positions,
            config.arena_random_plies_min,
            config.arena_random_plies_max,
            seed ^ 0xA076_1D64_78BD_642F,
        );
        positions.extend(random_fens.iter().map(|fen| {
            Position::from_fen(fen)
                .unwrap_or_else(|err| panic!("generated invalid arena FEN `{fen}`: {err}"))
        }));
        modes.push(format!(
            "random(count={},plies={}-{})",
            config.arena_random_positions,
            config.arena_random_plies_min,
            config.arena_random_plies_max
        ));
    }

    if positions.is_empty() {
        (Vec::new(), "startpos_fallback".to_string())
    } else {
        (positions, modes.join("+"))
    }
}

fn fixed_az_search_limits(
    simulations: usize,
    seed: u64,
    cpuct: f32,
    cpuct_at_root: f32,
    max_depth: usize,
    policy_softmax_temp: f32,
) -> AzSearchLimits {
    AzSearchLimits {
        simulations,
        seed,
        cpuct,
        cpuct_at_root,
        cpuct_base: 19652.0,
        cpuct_factor: 1.5,
        cpuct_base_at_root: 19652.0,
        cpuct_factor_at_root: 1.5,
        max_depth,
        root_dirichlet_alpha: 0.0,
        root_exploration_fraction: 0.0,
        fpu_value: 0.30,
        fpu_value_at_root: 0.20,
        policy_softmax_temp: policy_softmax_temp.max(1.0e-3),
        draw_score: 0.0,
        value_scale: 1.0,
    }
}

fn log_scalar(writer: &mut SummaryWriter, tag: &str, step: usize, value: f32) {
    writer.add_scalar(tag, value, step);
}

fn print_az_search_candidates(result: &chineseai::az::AzSearchResult, top: usize) {
    let mut candidates = result.candidates.iter().collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        right
            .visits
            .cmp(&left.visits)
            .then_with(|| right.policy.total_cmp(&left.policy))
            .then_with(|| right.q.total_cmp(&left.q))
    });
    let shown = if top == 0 {
        candidates.len()
    } else {
        top.min(candidates.len())
    };
    println!(
        "\nCANDIDATES — visits descending ({shown}/{})",
        candidates.len()
    );
    println!("    #  B  MOVE      VISITS  VISIT P       Q      CP       NET P      TREE P");
    println!("  ---- --  -------  --------  -------  ------  ------  ----------  ----------");
    for (rank, candidate) in candidates.into_iter().take(shown).enumerate() {
        let best = if Some(candidate.mv) == result.best_move {
            "*"
        } else {
            " "
        };
        println!(
            "  {:>4}  {}  {:<7}  {:>8}  {:>6.2}%  {:>+6.3}  {:>+6}  {:>9.5}  {:>9.5}",
            rank + 1,
            best,
            candidate.mv,
            candidate.visits,
            candidate.policy * 100.0,
            candidate.q,
            chineseai::az::cp_from_q(candidate.q),
            candidate.raw_prior,
            candidate.prior,
        );
    }
}

fn print_az_search_trace(trace_move: Move, trace: &[chineseai::az::AzSearchTraceStep]) {
    println!(
        "\nPRINCIPAL TRACE — root {trace_move}, {} plies",
        trace.len()
    );
    if trace.is_empty() {
        println!("  Root move was not expanded.");
        return;
    }
    println!(
        "  PLY  MOVE      VISITS       Q     PRIOR  CHECK  EXPANDED    CHILD Q       CHILD W/D/L"
    );
    println!(
        "  ---  -------  --------  ------  --------  -----  --------  --------  ------------------"
    );
    for step in trace {
        println!(
            "  {:>3}  {:<7}  {:>8}  {:>+6.3}  {:>7.5}  {:>5}  {:>8}  {:>+8.3}  {:>5.1}%/{:>5.1}%/{:>5.1}%",
            step.ply,
            step.mv,
            step.visits,
            step.q,
            step.prior,
            if step.gives_check { "yes" } else { "no" },
            if step.child_expanded { "yes" } else { "no" },
            step.child_value,
            step.child_value_wdl[0] * 100.0,
            step.child_value_wdl[1] * 100.0,
            step.child_value_wdl[2] * 100.0,
        );
        println!("       fen: {}", step.child_fen);
    }
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        None => {
            let _ = Cli::command().print_help();
            std::process::exit(0);
        }
        Some(CliCommand::AzInit(cmd)) => {
            let arch = cmd.arch();
            let output = cmd.output;
            let seed = cmd.seed;
            let model = AzNnue::random_with_arch(arch, seed);
            model.save(&output).unwrap_or_else(|err| {
                panic!("failed to write `{output}`: {err}");
            });
            println!(
                "aznnue   : initialized (safetensors, format v{})",
                chineseai::version::MODEL_FORMAT_VERSION
            );
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
                fpu_value_at_root: cmd.fpu_value_at_root.max(0.0),
                policy_softmax_temp: cmd.policy_softmax_temp.max(1.0e-3),
                draw_score: cmd.draw_score.clamp(-1.0, 1.0),
                value_scale: cmd.value_scale.clamp(0.0, 1.0),
            };
            let root_moves = if cmd.root_moves.is_empty() {
                None
            } else {
                Some(
                    cmd.root_moves
                        .iter()
                        .map(|text| {
                            position.parse_uci_move(text).unwrap_or_else(|| {
                                panic!("invalid or illegal --root-move `{text}` for this position")
                            })
                        })
                        .collect::<Vec<_>>(),
                )
            };
            let trace_move = cmd.trace_move.as_deref().map(|text| {
                position
                    .parse_uci_move(text)
                    .unwrap_or_else(|| panic!("invalid or illegal --trace-move `{text}`"))
            });
            let search_started = Instant::now();
            let (result, trace) = if let Some(trace_move) = trace_move {
                alphazero_search_trace_with_rules(
                    &position,
                    Some(rule_history.clone()),
                    root_moves,
                    &model,
                    search_limits,
                    trace_move,
                )
            } else {
                (
                    alphazero_search_with_rules(
                        &position,
                        Some(rule_history.clone()),
                        root_moves,
                        &model,
                        search_limits,
                    ),
                    Vec::new(),
                )
            };
            let search_elapsed = search_started.elapsed();
            let mut by_visits = result.candidates.clone();
            by_visits.sort_by(|left, right| {
                right
                    .visits
                    .cmp(&left.visits)
                    .then_with(|| right.policy.total_cmp(&left.policy))
                    .then_with(|| right.q.total_cmp(&left.q))
            });
            let visited_actions = by_visits
                .iter()
                .filter(|candidate| candidate.visits > 0)
                .count();
            let elapsed_seconds = search_elapsed.as_secs_f64().max(f64::EPSILON);
            let best_move = result
                .best_move
                .map(|mv| mv.to_string())
                .unwrap_or_else(|| "(none)".into());
            println!("AZ SEARCH");
            println!("=========");
            println!("\nPOSITION");
            println!("  FEN          {}", position.to_fen());
            println!("  Side         {:?}", position.side_to_move());
            println!(
                "  Applied      {}",
                if cmd.moves.is_empty() {
                    "(none)".to_string()
                } else {
                    cmd.moves.join(" ")
                }
            );
            println!(
                "  Root moves   {}",
                if cmd.root_moves.is_empty() {
                    "all legal".to_string()
                } else {
                    cmd.root_moves.join(" ")
                }
            );
            println!("\nCONFIGURATION");
            println!("  Model        {model_path}");
            println!("  Simulations  {simulations}");
            println!(
                "  PUCT         non-root={cpuct:.3} root={cpuct_at_root:.3} base={:.1}/{:.1} factor={:.3}/{:.3}",
                search_limits.cpuct_base,
                search_limits.cpuct_base_at_root,
                search_limits.cpuct_factor,
                search_limits.cpuct_factor_at_root
            );
            println!(
                "  FPU reduce   non-root={:.3} root={:.3}",
                search_limits.fpu_value, search_limits.fpu_value_at_root
            );
            println!("  Policy temp  {:.3}", search_limits.policy_softmax_temp);
            println!("  Draw score   {:.3}", search_limits.draw_score);
            println!("\nRESULT");
            println!("  Best move    {best_move}");
            println!(
                "  Search value Q={:+.4}  CP={:+}  W/D/L={:.2}%/{:.2}%/{:.2}%",
                result.value_q,
                result.value_cp,
                result.value_wdl[0] * 100.0,
                result.value_wdl[1] * 100.0,
                result.value_wdl[2] * 100.0
            );
            println!(
                "  Network WDL  {:.2}%/{:.2}%/{:.2}%",
                result.network_value_wdl[0] * 100.0,
                result.network_value_wdl[1] * 100.0,
                result.network_value_wdl[2] * 100.0
            );
            println!(
                "  Root actions {} legal, {} visited",
                result.candidates.len(),
                visited_actions
            );
            println!(
                "  Depth        avg={:.2} max={} limit={} cutoffs={}",
                result.search_depth_avg,
                result.search_depth_max,
                result.search_depth_limit,
                result.search_depth_cutoffs
            );
            println!(
                "  Performance  {:.3} ms, {:.0} simulations/s",
                search_elapsed.as_secs_f64() * 1000.0,
                result.simulations as f64 / elapsed_seconds
            );
            print_az_search_candidates(&result, cmd.top);
            if let Some(trace_move) = trace_move {
                print_az_search_trace(trace_move, &trace);
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
                println!("\nCHILD VERIFICATION — {verify_sims} simulations each");
                println!(
                    "  MOVE     VISITS   ROOT Q     NN Q   DEEP Q       ΔQ      CP  OPPONENT REPLY"
                );
                println!(
                    "  -------  -------  -------  -------  -------  -------  ------  --------------"
                );
            }
            for mv in verify_moves {
                let Some(root_candidate) = result.candidates.iter().find(|item| item.mv == mv)
                else {
                    println!("  {mv:<7}  unavailable at root");
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
                    "  {:<7}  {:>7}  {:>+7.3}  {:>+7.3}  {:>+7.3}  {:>+7.3}  {:>+6}  {}",
                    mv,
                    root_candidate.visits,
                    root_candidate.q,
                    -child_nn_q,
                    verified_root_q,
                    verified_root_q - root_candidate.q,
                    verified_root_cp,
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
                fixed_az_search_limits(simulations, 0, cpuct, cpuct, 0, 1.0),
            );

            let started = std::time::Instant::now();
            let mut total_sims = 0usize;
            let mut best_move = None;
            for iteration in 0..repeat {
                let result = alphazero_search(
                    &position,
                    &model,
                    fixed_az_search_limits(simulations, iteration as u64, cpuct, cpuct, 0, 1.0),
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
            let batch_size = cmd.batch_size.max(1);
            let lr = cmd.lr.max(0.0);
            let seed = cmd.seed;
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
            println!("lr             : {lr}");
            println!("elapsed_ms   : {:.3}", elapsed * 1000.0);
            println!("processed    : {}", sample_count * epochs);
            println!("samples/sec  : {:.0}", processed / elapsed);
            println!("loss         : {:.4}", stats.loss);
            println!("value_ce     : {:.4}", stats.value_loss);
            println!("policy_ce    : {:.4}", stats.policy_ce);
        }
        Some(CliCommand::AzReplayOpeningStats(cmd)) => {
            #[derive(Default)]
            struct MoveStats {
                target_mass: f64,
                played: usize,
                wins: usize,
                draws: usize,
                losses: usize,
                target_q_sum: f64,
            }

            let pool = AzExperiencePool::load_snapshot_lz4(Path::new(&cmd.replay), usize::MAX)
                .unwrap_or_else(|err| panic!("failed to load replay `{}`: {err}", cmd.replay));
            let groups = pool.all_sample_groups();
            let start = Position::startpos();
            let names = start
                .legal_moves()
                .into_iter()
                .map(|mv| {
                    (
                        dense_move_index(canonical_move(start.side_to_move(), mv)),
                        mv.to_uci(),
                    )
                })
                .collect::<HashMap<_, _>>();
            let mut stats = HashMap::<usize, MoveStats>::new();
            let mut start_games = 0usize;
            for group in &groups {
                let Some(sample) = group.iter().find(|sample| sample.meta.ply == 0) else {
                    continue;
                };
                start_games += 1;
                let target_q = sample.value_wdl[0] - sample.value_wdl[2];
                for (&move_index, &mass) in sample.move_indices.iter().zip(&sample.policy) {
                    let row = stats.entry(move_index).or_default();
                    row.target_mass += mass.max(0.0) as f64;
                    row.target_q_sum += target_q as f64 * mass.max(0.0) as f64;
                }
                let played = sample.meta.played_index as usize;
                let Some(&move_index) = sample.move_indices.get(played) else {
                    continue;
                };
                let row = stats.entry(move_index).or_default();
                row.played += 1;
                let result_red = group.last().map_or(0.0, |last| last.value * last.side_sign);
                if result_red > 0.5 {
                    row.wins += 1;
                } else if result_red < -0.5 {
                    row.losses += 1;
                } else {
                    row.draws += 1;
                }
            }
            let mut rows = stats.into_iter().collect::<Vec<_>>();
            rows.sort_by(|left, right| right.1.target_mass.total_cmp(&left.1.target_mass));
            println!(
                "replay-opening: snapshot={} groups={} start_games={}",
                cmd.replay,
                groups.len(),
                start_games
            );
            println!("MOVE target% played% games W/D/L score targetQ");
            for (move_index, row) in rows.into_iter().take(cmd.top.max(1)) {
                let games = row.wins + row.draws + row.losses;
                let score = if games == 0 {
                    0.0
                } else {
                    (row.wins as f64 + 0.5 * row.draws as f64) / games as f64
                };
                println!(
                    "{:<5} {:>7.3} {:>7.3} {:>5} {}/{}/{} {:>5.3} {:+.3}",
                    names.get(&move_index).map_or("?", String::as_str),
                    100.0 * row.target_mass / start_games.max(1) as f64,
                    100.0 * row.played as f64 / start_games.max(1) as f64,
                    games,
                    row.wins,
                    row.draws,
                    row.losses,
                    score,
                    row.target_q_sum / row.target_mass.max(f64::EPSILON),
                );
            }
        }
        Some(CliCommand::AzReplayFit(cmd)) => {
            let capacity = cmd.samples.max(2);
            let pool = AzExperiencePool::load_snapshot_lz4(Path::new(&cmd.replay), capacity)
                .unwrap_or_else(|err| panic!("failed to load replay `{}`: {err}", cmd.replay));
            let window = pool.window_stats(5000);
            let groups = pool.all_sample_groups();
            drop(pool);
            let validation_groups = ((groups.len() as f32)
                * cmd.validation_fraction.clamp(0.01, 0.5))
            .round()
            .max(1.0) as usize;
            let split = groups.len().saturating_sub(validation_groups).max(1);
            let mut train = Vec::new();
            let mut validation = Vec::new();
            for (index, group) in groups.into_iter().enumerate() {
                if index < split {
                    train.extend(group);
                } else {
                    validation.extend(group);
                }
            }
            let arch = chineseai::az::AzNnueArch::with_hidden_size(cmd.hidden.max(1));
            let mut model = AzNnue::random_with_arch(arch, cmd.seed);
            let weights = AzTrainLossWeights::default();
            let mut eval_rng = SplitMix64::new(cmd.seed ^ 0xD1B5_4A32_D192_ED03);
            let baseline = train_samples_weighted(
                &mut model.clone(),
                &validation,
                1,
                1.0e-12,
                cmd.batch_size.max(1),
                &mut eval_rng,
                weights,
            )
            .expect("replay-fit baseline training failed");
            let mut train_rng = SplitMix64::new(cmd.seed);
            let started = Instant::now();
            let trained = train_samples_weighted(
                &mut model,
                &train,
                cmd.epochs.max(1),
                cmd.lr.max(0.0),
                cmd.batch_size.max(1),
                &mut train_rng,
                weights,
            )
            .expect("replay-fit training failed");
            let train_seconds = started.elapsed().as_secs_f32();
            let mut final_eval_model = model.clone();
            let mut final_eval_rng = SplitMix64::new(cmd.seed ^ 0x94D0_49BB_1331_11EB);
            let validation_stats = train_samples_weighted(
                &mut final_eval_model,
                &validation,
                1,
                1.0e-12,
                cmd.batch_size.max(1),
                &mut final_eval_rng,
                weights,
            )
            .expect("replay-fit validation training failed");
            let policy_groups = evaluate_policy_groups(&model, &validation);
            let mut ablated_model = model.clone();
            ablated_model.policy_consequence_output.fill(0.0);
            let ablated_groups = evaluate_policy_groups(&ablated_model, &validation);
            model.save(&cmd.output).unwrap_or_else(|err| {
                panic!("failed to save replay-fit model `{}`: {err}", cmd.output)
            });
            println!("replay-fit : {}", cmd.replay);
            println!("window     : {:?}", window);
            println!("arch       : hidden={}", arch.hidden_size);
            println!(
                "split      : train={} validation={} games={}",
                train.len(),
                validation.len(),
                split + validation_groups
            );
            println!(
                "baseline   : loss={:.5} value_ce={:.5} policy_ce={:.5}",
                baseline.loss, baseline.value_loss, baseline.policy_ce
            );
            println!(
                "train      : loss={:.5} value_ce={:.5} policy_ce={:.5} seconds={:.2}",
                trained.loss, trained.value_loss, trained.policy_ce, train_seconds
            );
            println!(
                "validation : loss={:.5} value_ce={:.5} policy_ce={:.5}",
                validation_stats.loss, validation_stats.value_loss, validation_stats.policy_ce
            );
            println!("groups     : {policy_groups:?}");
            println!("no-delta   : {ablated_groups:?}");
            println!("output     : {}", cmd.output);
        }
        Some(CliCommand::AzLoop(cmd)) => {
            let config_path = cmd.config;
            let Some(config) = load_or_create_az_loop_config(&config_path) else {
                return;
            };
            let target_update = cmd.target_update.map(|update| update.max(1));
            let progress_boot = load_az_loop_progress(&config_path);
            let start_update = progress_boot.next_update.max(1);
            let mut arena_nemesis_update = progress_boot.nemesis_update;
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
                    "resume   : update starts at {} (from `{}`)",
                    start_update,
                    az_loop_progress_path(&config_path).display()
                );
            }
            let best_path = best_model_path(&config.model_path);

            let config_arch = config.arch();
            let model_path = Path::new(&config.model_path);
            let (model, resumed_model) = if model_path.exists() {
                println!("model    : load {}", config.model_path);
                let model = AzNnue::load(model_path).unwrap_or_else(|err| {
                    panic!(
                        "refusing to resume incompatible model `{}`: {err}",
                        model_path.display()
                    )
                });
                if model.arch != config_arch {
                    panic!(
                        "model `{}` architecture {:?} differs from config {:?}",
                        model_path.display(),
                        model.arch,
                        config_arch
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
            } else if config.arena_interval > 0 && best_path.exists() {
                println!("model    : load best `{}` as current", best_path.display());
                let best = AzNnue::load(&best_path).unwrap_or_else(|err| {
                    panic!("failed to load best model `{}`: {err}", best_path.display());
                });
                if best.arch != config_arch {
                    panic!(
                        "best model `{}` architecture {:?} differs from config {:?}",
                        best_path.display(),
                        best.arch,
                        config_arch
                    );
                }
                (best, true)
            } else {
                println!("model    : init {}", config.model_path);
                (AzNnue::random_with_arch(config_arch, config.seed), false)
            };
            let selfplay_model = model.clone();
            println!("selfplay : start from raw model");
            let initial_arena_reference_model = if config.arena_interval == 0 {
                selfplay_model.clone()
            } else {
                if !best_path.exists() {
                    save_model(&selfplay_model, &best_path);
                }
                let reference = AzNnue::load(&best_path).unwrap_or_else(|err| {
                    panic!("failed to load best model `{}`: {err}", best_path.display());
                });
                if reference.arch != selfplay_model.arch {
                    panic!(
                        "best model `{}` architecture {:?} differs from self-play {:?}",
                        best_path.display(),
                        reference.arch,
                        selfplay_model.arch
                    );
                }
                reference
            };
            let initial_selfplay_model = if config.arena_interval > 0 {
                println!("selfplay : actor starts from champion; arena controls publication");
                initial_arena_reference_model.clone()
            } else {
                println!(
                    "selfplay : ungated actor starts from learner; publish every {} updates",
                    config.actor_publish_interval_updates
                );
                selfplay_model.clone()
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
                        println!(
                            "replay   : restored {}/{} samples from `{}`",
                            pool.sample_count(),
                            pool.capacity(),
                            replay_snapshot_path.display()
                        );
                        replay_pool = Some(pool);
                    }
                    Err(err) => {
                        panic!(
                            "refusing incompatible replay snapshot `{}`: {err}",
                            replay_snapshot_path.display()
                        );
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
            let opening_snapshot_path = PathBuf::from(&config.opening_snapshot_path);
            let opening_pool = if config.opening_reservoir_capacity == 0 {
                AzMidgamePool::new(0)
            } else if opening_snapshot_path.exists() {
                AzMidgamePool::load_lz4(&opening_snapshot_path, config.opening_reservoir_capacity)
                    .unwrap_or_else(|err| {
                        panic!(
                            "failed to restore opening pool `{}`: {err}",
                            opening_snapshot_path.display()
                        )
                    })
            } else {
                AzMidgamePool::new(config.opening_reservoir_capacity)
            };
            println!(
                "opening  : restored {}/{} snapshots from `{}`",
                opening_pool.len(),
                opening_pool.capacity(),
                opening_snapshot_path.display()
            );
            let shared_opening_pool = Arc::new(RwLock::new(opening_pool));
            let midgame_snapshot_path = PathBuf::from(&config.midgame_snapshot_path);
            let midgame_pool = if config.midgame_reservoir_capacity == 0 {
                AzMidgamePool::new(0)
            } else if midgame_snapshot_path.exists() {
                AzMidgamePool::load_lz4(&midgame_snapshot_path, config.midgame_reservoir_capacity)
                    .unwrap_or_else(|err| {
                        panic!(
                            "failed to restore midgame pool `{}`: {err}",
                            midgame_snapshot_path.display()
                        )
                    })
            } else {
                AzMidgamePool::new(config.midgame_reservoir_capacity)
            };
            println!(
                "midgame  : restored {}/{} snapshots from `{}`",
                midgame_pool.len(),
                midgame_pool.capacity(),
                midgame_snapshot_path.display()
            );
            let shared_midgame_pool = Arc::new(RwLock::new(midgame_pool));
            let effective_train_to_selfplay_ratio = (config.train_samples_per_update as f32
                * config.train_epochs_per_update as f32)
                / config.selfplay_samples_per_update.max(1) as f32;
            let replay_update_span =
                config.replay_capacity as f32 / config.selfplay_samples_per_update.max(1) as f32;
            let warmup_update_span = config.train_warmup_samples as f32
                / config.selfplay_samples_per_update.max(1) as f32;
            let replay_actor_span = if config.arena_interval > 0 {
                "promotion-dependent".to_string()
            } else {
                format!(
                    "{:.1}",
                    replay_update_span / config.actor_publish_interval_updates.max(1) as f32
                )
            };

            println!(
                "design   : replay={:.1}updates actor_generations={} warmup={:.1}updates expected_sample_exposures={:.2} optimizer_steps_per_update={}",
                replay_update_span,
                replay_actor_span,
                warmup_update_span,
                effective_train_to_selfplay_ratio,
                config
                    .train_samples_per_update
                    .div_ceil(config.batch_size.max(1))
                    .saturating_mul(config.train_epochs_per_update),
            );

            println!(
                "loop     : config={} mode=batch search=alphazero sims={} value_td_lambda={} replay_recent(fraction={},games={}) selfplay_samples_per_update={} train_to_selfplay_ratio={:.2} lr={} lr_decay(min={},start={},interval={},factor={}) batch_size={} train_warmup_samples={} train_samples_per_update={} train_epochs_per_update={} max_plies={} rules(repetition=asian2fold,sixty={},max_ply={}) selfplay_workers={} temp(start={},endgame={},delay={}ply,decay={}ply,value_cutoff={},visit_offset={}) cpuct={} cpuct_at_root={} fpu(value={},root={}) policy_softmax_temp={} root_noise(alpha={},fraction={}) opening_pool={}/{} resign(percentage={},playthrough={}) replay_capacity={} mirror_probability={} train(value={},policy={}) checkpoint_interval={} max_checkpoints={} arena_interval={} arena_sims={} arena(cpuct={}/{},policy_temp={}) arena_promotion(rate={},z={}) arena_processes={} arena_opening_book={} arena_opening_positions={} arena_opening_plies={}-{} arena_random_positions={} arena_random_plies={}-{} pikafish_label_eval(sqlite={},interval={},limit={},sims={},cpuct={}/{},policy_temp={}) tb_base={} tb_run={}",
                config_path,
                config.simulations,
                config.value_td_lambda,
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
                config.train_warmup_samples,
                config.train_samples_per_update,
                config.train_epochs_per_update,
                config.max_plies,
                config.sixty_move_rule,
                config.rule60_max_ply,
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
                config.opening_snapshot_path,
                config.opening_reservoir_capacity,
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
                config.arena_cpuct_at_root,
                config.arena_policy_softmax_temp,
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
                config.arena_random_positions,
                config.arena_random_plies_min,
                config.arena_random_plies_max,
                if config.pikafish_label_eval_sqlite.trim().is_empty() {
                    "(none)"
                } else {
                    config.pikafish_label_eval_sqlite.as_str()
                },
                config.pikafish_label_eval_interval,
                config.pikafish_label_eval_limit,
                config.pikafish_label_eval_simulations,
                config.pikafish_label_eval_cpuct,
                config.pikafish_label_eval_cpuct_at_root,
                config.pikafish_label_eval_policy_softmax_temp,
                config.tensorboard_logdir,
                tensorboard_encoded_subdir(&config)
            );
            println!(
                "explore  : root_noise(alpha={},fraction={}) persistent_actor={:.1}% actor_root_noise(alpha={},fraction={}) actor_move_temp={} move_temp={}..{} policy_temp={} starts(start/opening/midgame)={:.1}%/{:.1}%/{:.1}% pools(opening/midgame)={}/{} actor_publish={}",
                config.root_dirichlet_alpha,
                config.root_exploration_fraction,
                config.persistent_exploration_fraction * 100.0,
                config.persistent_exploration_root_dirichlet_alpha,
                config.persistent_exploration_root_exploration_fraction,
                config.persistent_exploration_temperature,
                config.temperature_start,
                config.temperature_endgame,
                config.policy_softmax_temp,
                (1.0 - config.opening_start_fraction - config.midgame_start_fraction) * 100.0,
                config.opening_start_fraction * 100.0,
                config.midgame_start_fraction * 100.0,
                config.opening_reservoir_capacity,
                config.midgame_reservoir_capacity,
                if config.arena_interval > 0 {
                    "arena-promote".to_string()
                } else {
                    format!("{}updates", config.actor_publish_interval_updates)
                }
            );
            let cpu_placements = chineseai::cpu_topology::cpu_placements();
            let numa_nodes = chineseai::cpu_topology::numa_nodes(&cpu_placements);
            let selfplay_worker_count = config.workers.max(1);
            // 覆盖一次GPU更新期间完成的批次，同时限制旧模型样本和内存积压。
            let selfplay_queue_capacity = selfplay_worker_count.saturating_mul(2).max(32);
            let (selfplay_tx, selfplay_rx) =
                mpsc::sync_channel::<SelfplayBatch>(selfplay_queue_capacity);
            let (trainer_tx, trainer_rx) = mpsc::sync_channel::<TrainerEvent>(2);
            let llc_domains = cpu_placements
                .iter()
                .map(|placement| (placement.node, placement.package, placement.llc))
                .collect::<HashSet<_>>()
                .len();
            println!(
                "cpu      : physical={} numa_nodes={} llc_domains={} selfplay_workers={} selfplay_queue={} affinity={} model_replicas={}",
                cpu_placements.len(),
                numa_nodes.len(),
                llc_domains,
                selfplay_worker_count,
                selfplay_queue_capacity,
                if cfg!(target_os = "linux") {
                    "on"
                } else {
                    "unsupported"
                },
                numa_nodes.len(),
            );
            let initial_numa_models =
                build_numa_model_replicas(&initial_selfplay_model, &numa_nodes);
            let shared_model = Arc::new(RwLock::new(SharedSelfplayModel {
                version: start_update.saturating_sub(1) as u64,
                learner_update: start_update.saturating_sub(1).min(u32::MAX as usize) as u32,
                models_by_numa_node: initial_numa_models,
            }));
            let mut arena_reference_model = initial_arena_reference_model;
            let mut champion_paths =
                champion_checkpoint_paths(&config.model_path, &config.checkpoint_dir)
                    .unwrap_or_else(|err| panic!("failed to load champion history: {err}"));
            if champion_paths.is_empty() {
                let initial_champion = save_best_checkpoint_model(
                    &arena_reference_model,
                    &config.model_path,
                    &config.checkpoint_dir,
                    start_update.saturating_sub(1),
                );
                println!("champion : initialized {}", initial_champion.display());
                champion_paths.push(initial_champion);
            } else {
                println!("champion : loaded history={}", champion_paths.len());
            }
            let selfplay_pause =
                Arc::new((Mutex::new(SelfplayPauseState::default()), Condvar::new()));
            let mut selfplay_handles = Vec::with_capacity(selfplay_worker_count);
            for worker_id in 0..selfplay_worker_count {
                let placement = cpu_placements[worker_id % cpu_placements.len()];
                let model_slot = numa_nodes
                    .iter()
                    .position(|&(node, _)| node == placement.node)
                    .unwrap_or(0);
                let selfplay_stop = stop_requested.clone();
                let selfplay_config = config.clone();
                let selfplay_tx = selfplay_tx.clone();
                let shared_model = Arc::clone(&shared_model);
                let selfplay_pause = Arc::clone(&selfplay_pause);
                let selfplay_opening_pool = Arc::clone(&shared_opening_pool);
                let selfplay_midgame_pool = Arc::clone(&shared_midgame_pool);
                selfplay_handles.push(thread::spawn(move || {
                    if let Err(err) = chineseai::cpu_topology::pin_current_thread(placement.cpu) {
                        eprintln!(
                            "warning: failed to pin selfplay worker {worker_id} to cpu {}: {err}",
                            placement.cpu
                        );
                    }
                    let mut batch_index = 0usize;
                    let mut local_version = u64::MAX;
                    let mut local_learner_update = 0u32;
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
                                local_learner_update = shared.learner_update;
                            }
                        }
                        let batch_seed = selfplay_config.seed
                            ^ ((worker_id as u64).wrapping_add(1) << 32)
                            ^ (batch_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                        let mut loop_config = build_az_loop_config(
                            &selfplay_config,
                            batch_seed,
                            1,
                            local_learner_update,
                            &Arc::default(),
                        );
                        loop_config.games = 4;
                        let mut pool_rng = SplitMix64::new(batch_seed ^ 0xA076_1D64_78BD_642F);
                        loop_config.opening_positions = selfplay_opening_pool
                            .read()
                            .unwrap_or_else(|_| panic!("opening pool poisoned"))
                            .sample(loop_config.games, &mut pool_rng)
                            .into();
                        loop_config.midgame_positions = selfplay_midgame_pool
                            .read()
                            .unwrap_or_else(|_| panic!("midgame pool poisoned"))
                            .sample(loop_config.games, &mut pool_rng)
                            .into();
                        let data = generate_selfplay_data(
                            local_model
                                .as_deref()
                                .expect("selfplay model not initialized"),
                            &loop_config,
                        );
                        let batch = SelfplayBatch { data };
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
            let replay_samples_at_start = replay_pool
                .as_ref()
                .map(AzExperiencePool::sample_count)
                .unwrap_or(0);
            let collector_warmup_missing = config
                .train_warmup_samples
                .saturating_sub(replay_samples_at_start);
            let collector_midgame_pool = Arc::clone(&shared_midgame_pool);
            let collector_opening_pool = Arc::clone(&shared_opening_pool);
            println!(
                "warmup   : {} (model={} replay_start={})",
                if collector_warmup_missing > 0 {
                    format!(
                        "collect {} missing samples to reach {}",
                        collector_warmup_missing, config.train_warmup_samples
                    )
                } else {
                    "skipped".to_string()
                },
                if resumed_model { "resumed" } else { "random" },
                replay_samples_at_start,
            );
            let collector_handle = thread::spawn(move || {
                let mut pending = PendingTrainingData::default();
                let mut batch_index = 0usize;
                let mut window_started = Instant::now();
                while let Ok(mut batch) = selfplay_rx.recv() {
                    let opening_snapshots = std::mem::take(&mut batch.data.opening_snapshots);
                    if !opening_snapshots.is_empty() {
                        collector_opening_pool
                            .write()
                            .unwrap_or_else(|_| panic!("opening pool poisoned"))
                            .add_snapshots(
                                opening_snapshots,
                                collector_config.seed ^ !(batch_index as u64),
                            );
                    }
                    let snapshots = std::mem::take(&mut batch.data.midgame_snapshots);
                    if !snapshots.is_empty() {
                        collector_midgame_pool
                            .write()
                            .unwrap_or_else(|_| panic!("midgame pool poisoned"))
                            .add_snapshots(snapshots, collector_config.seed ^ batch_index as u64);
                    }
                    pending.push(batch);
                    let required_samples = if batch_index == 0 {
                        collector_warmup_missing.max(collector_config.selfplay_samples_per_update)
                    } else {
                        collector_config.selfplay_samples_per_update
                    };
                    if pending.selfplay.samples.len() < required_samples {
                        continue;
                    }
                    pending.collection_seconds = window_started.elapsed().as_secs_f32();
                    window_started = Instant::now();
                    if ready_tx.send(std::mem::take(&mut pending)).is_err() {
                        break;
                    }
                    batch_index += 1;
                }
            });
            let trainer_stop = stop_requested.clone();
            let trainer_config = config.clone();
            let trainer_start_update = start_update;
            let trainer_snapshot_path = replay_snapshot_path.clone();
            let trainer_handle = thread::spawn(move || {
                let mut trainer_model = model;
                let mut trainer_pool = replay_pool;
                let mut train_index = 0usize;
                let min_train_samples = trainer_config.batch_size.max(1);
                'training: while let Ok(mut pending) = ready_rx.recv() {
                    let pending_games = pending.selfplay.games.len();
                    if let Some(pool) = trainer_pool.as_mut() {
                        pool.add_games(std::mem::take(&mut pending.selfplay.games));
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
                    let sampled_batch = pool.sample_phase_stratified_recent(
                        trainer_config.train_samples_per_update,
                        [
                            trainer_config.replay_phase_0_29_fraction,
                            trainer_config.replay_phase_30_59_fraction,
                            trainer_config.replay_phase_60_99_fraction,
                            trainer_config.replay_phase_100_139_fraction,
                            trainer_config.replay_phase_140_plus_fraction,
                        ],
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
                        sampled_batch.actual_recent_samples,
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
                            ..AzTrainLossWeights::default()
                        },
                    )
                    .unwrap_or_else(|err| panic!("training update {} failed: {err}", train_update));
                    let train_seconds = train_started.elapsed().as_secs_f32();
                    let report = build_async_training_report(
                        pending,
                        pending_games,
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
                    && trainer_stop.load(Ordering::SeqCst)
                {
                    match pool.save_snapshot_lz4(&trainer_snapshot_path) {
                        Ok(()) => {
                            if pool.sample_count() > 0 {
                                println!(
                                    "replay   : shutdown snapshot `{}` ({}/{} samples)",
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
                                        terminal_rule_draw_natural_limit: 0,
                                        terminal_rule_draw_insufficient_material: 0,
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
                                        terminal_rule_draw_natural_limit: 0,
                                        terminal_rule_draw_insufficient_material: 0,
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
                let deployed_model = candidate_model.clone();
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
                let midgame_pool_len = shared_midgame_pool
                    .read()
                    .unwrap_or_else(|_| panic!("midgame pool poisoned"))
                    .len();
                println!(
                    "update {update:04}: games={} samples={} train_samples={} pool={}/{} fill={:.0}% midpool={}/{} replay(chunks={} actor_updates={}-{} actor_update_span={} span_games={} recent_pool={:.3}) train_src(recent={:.3} start/opening/mid={:.3}/{:.3}/{:.3} fast={:.3} pw={:.3} vw={:.3}) R/B/D={}/{}/{} red_win_all={:.3} avg_plies={:.1} avg_sims={:.1} opt_loss={:.4} wdl_ce={:.4} trainQ_rmse={:.4} trainQ_mu={:.3}/{:.3} trainQ_rms={:.3}/{:.3} trainQ_corr={:.3} trainQ_cal={:.3} trainPhaseQ(p0_39={}/{:.3}/{:.3}/{:.3} p40_119={}/{:.3}/{:.3}/{:.3} p120plus={}/{:.3}/{:.3}/{:.3}) policy_kl={:.4} trainTargetH={:.4} lr={:.6} visitH={:.3} visitH_p0_89={:.3} visitH_p90plus={:.3} rawP={:.3}/{:.3} visitP={:.3}/{:.3} trainTargetP={:.3}/{:.3} topQgap={:.3} topQabs={:.3} visitA={:.1} sampTopQ={:.3} playQGap={:.3} visitRatio={:.3} maxQ={:.3} playedQ={:.3} train={:.1}s gps={:.2} sps={:.1} train_sps={:.1} elapsed={:.1}s{}",
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
                    midgame_pool_len,
                    config.midgame_reservoir_capacity,
                    report.replay_chunks,
                    report.replay_oldest_update,
                    report.replay_newest_update,
                    report
                        .replay_newest_update
                        .saturating_sub(report.replay_oldest_update),
                    report.replay_window_games,
                    report.replay_recent_window_fraction,
                    report.train_actual_recent_sample_rate,
                    report.train_start_source_rate[0],
                    report.train_start_source_rate[1],
                    report.train_start_source_rate[2],
                    report.train_fast_sample_rate,
                    report.train_policy_weight_mean,
                    report.train_value_weight_mean,
                    report.red_wins,
                    report.black_wins,
                    report.draws,
                    report.red_wins as f32 / report.games.max(1) as f32,
                    report.avg_plies,
                    report.avg_search_simulations,
                    report.loss,
                    report.value_loss,
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
                let source_phase =
                    |source: usize, phase: usize| report.source_phase_value[source * 3 + phase];
                println!(
                    "valueSrc {update:04}: startpos(p0={}/{:.3}/{:.3}/{:.3} p40={}/{:.3}/{:.3}/{:.3}) opening_pool(p0={}/{:.3}/{:.3}/{:.3}) midgame(p0={}/{:.3}/{:.3}/{:.3} p40={}/{:.3}/{:.3}/{:.3})",
                    source_phase(0, 0).samples,
                    source_phase(0, 0).rmse,
                    source_phase(0, 0).corr,
                    source_phase(0, 0).calibration,
                    source_phase(0, 1).samples,
                    source_phase(0, 1).rmse,
                    source_phase(0, 1).corr,
                    source_phase(0, 1).calibration,
                    source_phase(1, 0).samples,
                    source_phase(1, 0).rmse,
                    source_phase(1, 0).corr,
                    source_phase(1, 0).calibration,
                    source_phase(2, 0).samples,
                    source_phase(2, 0).rmse,
                    source_phase(2, 0).corr,
                    source_phase(2, 0).calibration,
                    source_phase(2, 1).samples,
                    source_phase(2, 1).rmse,
                    source_phase(2, 1).corr,
                    source_phase(2, 1).calibration,
                );
                log_scalar(&mut tb, "train/optimized_loss", update, report.loss);
                log_scalar(&mut tb, "train/wdl_ce", update, report.value_loss);
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
                for (source, source_name) in ["startpos", "opening_pool", "midgame"]
                    .into_iter()
                    .enumerate()
                {
                    for (phase, phase_name) in ["ply_0_39", "ply_40_119", "ply_120_plus"]
                        .into_iter()
                        .enumerate()
                    {
                        let value = report.source_phase_value[source * 3 + phase];
                        log_scalar(
                            &mut tb,
                            &format!("value_source/{source_name}_{phase_name}_rmse"),
                            update,
                            value.rmse,
                        );
                        log_scalar(
                            &mut tb,
                            &format!("value_source/{source_name}_{phase_name}_corr"),
                            update,
                            value.corr,
                        );
                    }
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
                    "terminal/rule_draw_natural_limit",
                    update,
                    report.terminal_rule_draw_natural_limit as f32,
                );
                log_scalar(
                    &mut tb,
                    "terminal/rule_draw_insufficient_material",
                    update,
                    report.terminal_rule_draw_insufficient_material as f32,
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
                if config.arena_interval == 0
                    && update.is_multiple_of(config.actor_publish_interval_updates)
                {
                    let updated_numa_models =
                        build_numa_model_replicas(&deployed_model, &numa_nodes);
                    let actor_version = {
                        let mut shared = shared_model
                            .write()
                            .unwrap_or_else(|_| panic!("shared selfplay model poisoned"));
                        shared.models_by_numa_node = updated_numa_models;
                        shared.version = shared.version.wrapping_add(1);
                        shared.learner_update = update.min(u32::MAX as usize) as u32;
                        shared.version
                    };
                    println!(
                        "actor    : published learner update {update} as generation {actor_version}"
                    );
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
                        let (mut arena_start_positions, arena_mode) =
                            build_arena_start_positions(&config, update);
                        shuffle_positions(
                            &mut arena_start_positions,
                            &mut SplitMix64::new(
                                config.seed ^ (update as u64).wrapping_mul(0xE703_7ED1_A0B4_28DB),
                            ),
                        );
                        let arena_position_count = arena_start_positions.len();
                        let previous_index = champion_paths.len().checked_sub(2);
                        let gate_index = update / config.arena_interval.max(1);
                        let nemesis_index = arena_nemesis_update.and_then(|nemesis_update| {
                            champion_paths
                                .iter()
                                .position(|path| checkpoint_number(path) == Some(nemesis_update))
                        });
                        let anchor_index = nemesis_index
                            .or_else(|| historical_anchor_index(champion_paths.len(), gate_index));
                        let (current_count, previous_count, _) = arena_gate_position_counts(
                            arena_position_count,
                            previous_index.is_some(),
                            anchor_index.is_some(),
                        );
                        let anchor_positions = arena_start_positions
                            .split_off(current_count.saturating_add(previous_count));
                        let previous_positions = arena_start_positions.split_off(current_count);
                        let current_positions = arena_start_positions;
                        let candidate = Arc::new(deployed_model.clone());
                        let run_gate_match =
                            |baseline: Arc<AzNnue>, positions: Vec<Position>, seed_salt: u64| {
                                run_arena_threads(ArenaThreadConfig {
                                    candidate: Arc::clone(&candidate),
                                    baseline,
                                    eval_positions: Arc::new(positions),
                                    simulations: config.arena_simulations,
                                    max_plies: config.max_plies,
                                    rule60_max_ply: config
                                        .sixty_move_rule
                                        .then_some(config.rule60_max_ply),
                                    cpuct: config.arena_cpuct,
                                    cpuct_at_root: config.arena_cpuct_at_root,
                                    cpuct_base: config.cpuct_base,
                                    cpuct_factor: config.cpuct_factor,
                                    cpuct_base_at_root: config.cpuct_base_at_root,
                                    cpuct_factor_at_root: config.cpuct_factor_at_root,
                                    fpu_value: config.fpu_value,
                                    fpu_value_at_root: config.fpu_value_at_root,
                                    draw_score: config.draw_score,
                                    policy_softmax_temp: config.arena_policy_softmax_temp,
                                    thread_count: config.arena_processes,
                                    seed: config.seed
                                        ^ (update as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                                        ^ seed_salt,
                                })
                            };
                        let current_arena = run_gate_match(
                            Arc::new(arena_reference_model.clone()),
                            current_positions,
                            0,
                        );
                        let load_champion = |index: usize| {
                            let path = &champion_paths[index];
                            let model = AzNnue::load(path).unwrap_or_else(|err| {
                                panic!("failed to load champion `{}`: {err}", path.display())
                            });
                            assert_eq!(
                                model.arch,
                                deployed_model.arch,
                                "champion `{}` architecture mismatch",
                                path.display()
                            );
                            Arc::new(model)
                        };
                        let previous_arena = previous_index.map(|index| {
                            run_gate_match(
                                load_champion(index),
                                previous_positions,
                                0xA076_1D64_78BD_642F,
                            )
                        });
                        let anchor_arena = anchor_index.map(|index| {
                            run_gate_match(
                                load_champion(index),
                                anchor_positions,
                                0xE703_7ED1_A0B4_28DB,
                            )
                        });
                        let elo_diff = current_arena.elo_diff_vs_even();
                        let (elo_lower, elo_upper) =
                            current_arena.elo_diff_bounds(config.arena_promotion_confidence_z);
                        let gate_decision = arena_gate_decision(
                            &current_arena,
                            previous_arena.as_ref(),
                            anchor_arena.as_ref(),
                            config.arena_promotion_rate,
                            config.arena_promotion_confidence_z,
                        );
                        let promoted = gate_decision == ArenaGateDecision::Promote;
                        if let (Some(index), Some(report)) = (anchor_index, anchor_arena.as_ref()) {
                            if report.score_rate_upper_bound(config.arena_promotion_confidence_z)
                                < 0.50
                            {
                                arena_nemesis_update = checkpoint_number(&champion_paths[index]);
                                println!(
                                    "nemesis  : set {} rate={:.3}",
                                    checkpoint_label(&champion_paths[index]),
                                    report.score_rate()
                                );
                            } else if promoted && nemesis_index == Some(index) {
                                println!(
                                    "nemesis  : cleared {} rate={:.3}",
                                    checkpoint_label(&champion_paths[index]),
                                    report.score_rate()
                                );
                                arena_nemesis_update = None;
                            }
                        }
                        if promoted {
                            arena_reference_model = deployed_model.clone();
                            let updated_numa_models =
                                build_numa_model_replicas(&deployed_model, &numa_nodes);
                            let actor_version = {
                                let mut shared = shared_model
                                    .write()
                                    .unwrap_or_else(|_| panic!("shared selfplay model poisoned"));
                                shared.models_by_numa_node = updated_numa_models;
                                shared.version = shared.version.wrapping_add(1);
                                shared.learner_update = update.min(u32::MAX as usize) as u32;
                                shared.version
                            };
                            println!(
                                "actor    : published promoted update {update} as generation {actor_version}"
                            );
                            let best_checkpoint = save_best_checkpoint_model(
                                &deployed_model,
                                &config.model_path,
                                &config.checkpoint_dir,
                                update,
                            );
                            save_model(&deployed_model, &best_path);
                            champion_paths.push(best_checkpoint.clone());
                            println!("best     : saved {}", best_checkpoint.display());
                        }
                        println!(
                            "arena {update:04}: mode={} positions={} current_games={} pairs={} current_W/L/D={}/{}/{} current_rate={:.3} paired_se={:.4} ci={:.3}..{:.3} promote_at={:.3} z={:.2} decision={:?} elo_diff={:+.1} elo_ci={:+.1}..{:+.1} best_ref=memory{}",
                            arena_mode,
                            arena_position_count,
                            current_arena.total_games(),
                            current_arena.paired_openings,
                            current_arena.wins,
                            current_arena.losses,
                            current_arena.draws,
                            current_arena.score_rate(),
                            current_arena.score_rate_standard_error(),
                            current_arena
                                .score_rate_lower_bound(config.arena_promotion_confidence_z),
                            current_arena
                                .score_rate_upper_bound(config.arena_promotion_confidence_z),
                            config.arena_promotion_rate,
                            config.arena_promotion_confidence_z,
                            gate_decision,
                            elo_diff,
                            elo_lower,
                            elo_upper,
                            if promoted {
                                " promoted=current saved_best"
                            } else {
                                ""
                            }
                        );
                        if let (Some(index), Some(report)) =
                            (previous_index, previous_arena.as_ref())
                        {
                            println!(
                                "arena {update:04}: previous={} games={} W/L/D={}/{}/{} rate={:.3} ucb={:.3}",
                                checkpoint_label(&champion_paths[index]),
                                report.total_games(),
                                report.wins,
                                report.losses,
                                report.draws,
                                report.score_rate(),
                                report.score_rate_upper_bound(config.arena_promotion_confidence_z)
                            );
                        }
                        if let (Some(index), Some(report)) = (anchor_index, anchor_arena.as_ref()) {
                            println!(
                                "arena {update:04}: anchor={} games={} W/L/D={}/{}/{} rate={:.3} ucb={:.3}",
                                checkpoint_label(&champion_paths[index]),
                                report.total_games(),
                                report.wins,
                                report.losses,
                                report.draws,
                                report.score_rate(),
                                report.score_rate_upper_bound(config.arena_promotion_confidence_z)
                            );
                        }
                        let mut historical_arena = AzArenaReport::default();
                        if let Some(report) = previous_arena.as_ref() {
                            historical_arena.add_assign(report);
                        }
                        if let Some(report) = anchor_arena.as_ref() {
                            historical_arena.add_assign(report);
                        }
                        if historical_arena.total_games() > 0 {
                            println!(
                                "arena {update:04}: history_games={} history_rate={:.3} history_ucb={:.3}",
                                historical_arena.total_games(),
                                historical_arena.score_rate(),
                                historical_arena
                                    .score_rate_upper_bound(config.arena_promotion_confidence_z)
                            );
                            log_scalar(
                                &mut tb,
                                "arena/history_score_rate",
                                update,
                                historical_arena.score_rate(),
                            );
                        }
                        log_scalar(
                            &mut tb,
                            "arena/score_rate",
                            update,
                            current_arena.score_rate(),
                        );
                        if let Some(report) = previous_arena.as_ref() {
                            log_scalar(
                                &mut tb,
                                "arena/previous_score_rate",
                                update,
                                report.score_rate(),
                            );
                        }
                        if let Some(report) = anchor_arena.as_ref() {
                            log_scalar(
                                &mut tb,
                                "arena/anchor_score_rate",
                                update,
                                report.score_rate(),
                            );
                        }
                        log_scalar(&mut tb, "arena/elo_diff", update, elo_diff);
                        log_scalar(&mut tb, "arena/elo_diff_lower", update, elo_lower);
                        log_scalar(&mut tb, "arena/elo_diff_upper", update, elo_upper);
                        log_scalar(
                            &mut tb,
                            "arena/wins_as_red",
                            update,
                            current_arena.wins_as_red as f32,
                        );
                        log_scalar(
                            &mut tb,
                            "arena/losses_as_red",
                            update,
                            current_arena.losses_as_red as f32,
                        );
                        log_scalar(
                            &mut tb,
                            "arena/wins_as_black",
                            update,
                            current_arena.wins_as_black as f32,
                        );
                        log_scalar(
                            &mut tb,
                            "arena/losses_as_black",
                            update,
                            current_arena.losses_as_black as f32,
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
                                AzSearchLimits {
                                    simulations: config.pikafish_label_eval_simulations,
                                    seed: config.seed
                                        ^ (update as u64).wrapping_mul(0xD6E8_FD50_19B7_8421),
                                    cpuct: config.pikafish_label_eval_cpuct,
                                    cpuct_at_root: config.pikafish_label_eval_cpuct_at_root,
                                    cpuct_base: config.cpuct_base,
                                    cpuct_factor: config.cpuct_factor,
                                    cpuct_base_at_root: config.cpuct_base_at_root,
                                    cpuct_factor_at_root: config.cpuct_factor_at_root,
                                    max_depth: config.max_plies,
                                    root_dirichlet_alpha: 0.0,
                                    root_exploration_fraction: 0.0,
                                    fpu_value: config.fpu_value,
                                    fpu_value_at_root: config.fpu_value_at_root,
                                    policy_softmax_temp: config
                                        .pikafish_label_eval_policy_softmax_temp,
                                    draw_score: config.draw_score,
                                    value_scale: 1.0,
                                },
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
                                    "pikafish-label {update:04}: sqlite={} evaluated={} legal={} value_labels={} sims={} threads={} search_top1={:.3}% search_top2={:.3}% search_top4={:.3}% search_top8={:.3}% raw_prior_top1={:.3}% raw_value_corr={:.4} raw_value_mae={:.4} search_value_corr={:.4} search_value_mae={:.4} elapsed={:.1}s",
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
                                    stats.raw_value_corr(),
                                    stats.raw_value_mae_wdl_q(),
                                    stats.value_corr(),
                                    stats.value_mae_wdl_q(),
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
                                    "pikafish_label/search_value_corr",
                                    update,
                                    stats.value_corr() as f32,
                                );
                                log_scalar(
                                    &mut tb,
                                    "pikafish_label/search_value_mae_wdl_q",
                                    update,
                                    stats.value_mae_wdl_q(),
                                );
                                log_scalar(
                                    &mut tb,
                                    "pikafish_label/raw_value_corr",
                                    update,
                                    stats.raw_value_corr() as f32,
                                );
                                log_scalar(
                                    &mut tb,
                                    "pikafish_label/raw_value_mae_wdl_q",
                                    update,
                                    stats.raw_value_mae_wdl_q(),
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
                        let resolved = if sqlite_path.is_absolute() {
                            sqlite_path.to_path_buf()
                        } else {
                            std::env::current_dir()
                                .unwrap_or_else(|_| PathBuf::from("."))
                                .join(sqlite_path)
                        };
                        println!(
                            "pikafish-label {update:04}: skipped missing sqlite={} resolved={} (copy the label DB or update pikafish_label_eval_sqlite)",
                            config.pikafish_label_eval_sqlite,
                            resolved.display()
                        );
                    }
                }
                tb.flush();
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
            if config.opening_reservoir_capacity > 0 {
                let pool = shared_opening_pool
                    .read()
                    .unwrap_or_else(|_| panic!("opening pool poisoned"));
                pool.save_lz4(&opening_snapshot_path).unwrap_or_else(|err| {
                    panic!(
                        "failed to save opening pool `{}`: {err}",
                        opening_snapshot_path.display()
                    )
                });
                println!(
                    "opening  : saved {}/{} snapshots to `{}`",
                    pool.len(),
                    pool.capacity(),
                    opening_snapshot_path.display()
                );
            }
            if config.midgame_reservoir_capacity > 0 {
                let pool = shared_midgame_pool
                    .read()
                    .unwrap_or_else(|_| panic!("midgame pool poisoned"));
                pool.save_lz4(&midgame_snapshot_path).unwrap_or_else(|err| {
                    panic!(
                        "failed to save midgame pool `{}`: {err}",
                        midgame_snapshot_path.display()
                    )
                });
                println!(
                    "midgame  : saved {}/{} snapshots to `{}`",
                    pool.len(),
                    pool.capacity(),
                    midgame_snapshot_path.display()
                );
            }
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
                        arena_nemesis_update,
                    );
                    println!(
                        "model    : {} save raw=`{}` next_update={}",
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
        }
        Some(CliCommand::VsPikafish(cmd)) => {
            let pikafish_exe = cmd.pikafish_exe;
            let model_path = cmd.model;
            let simulations = cmd.simulations.unwrap_or(192).max(1);
            let cpuct = cmd.cpuct.max(0.0);
            let cpuct_at_root = cmd.cpuct_at_root.max(0.0);
            let cpuct_base = cmd.cpuct_base.max(1.0);
            let cpuct_factor = cmd.cpuct_factor.max(0.0);
            let cpuct_base_at_root = cmd.cpuct_base_at_root.max(1.0);
            let cpuct_factor_at_root = cmd.cpuct_factor_at_root.max(0.0);
            let fpu_value = cmd.fpu_value.max(0.0);
            let fpu_value_at_root = cmd.fpu_value_at_root.max(0.0);
            let policy_softmax_temp = cmd.policy_softmax_temp.max(1.0e-3);
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
                    cpuct_base,
                    cpuct_factor,
                    cpuct_base_at_root,
                    cpuct_factor_at_root,
                    fpu_value,
                    fpu_value_at_root,
                    policy_softmax_temp,
                    report_games: cmd.report_games,
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
                "vs-pikafish: model={} search=alphazero games={} fens={} opening={} parallel={} chinese W/L/D={}/{}/{} (as_red={} as_black={}) win_reasons(general_capture={} checkmate_no_legal_moves={} rule={} pikafish_no_bestmove={} pikafish_invalid_move={} pikafish_illegal_move={}) | pikafish_depth={} max_plies={} sims={} cpuct={}/{} base={}/{} factor={}/{} fpu={}/{} policy_temp={}",
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
                cpuct_at_root,
                cpuct_base,
                cpuct_base_at_root,
                cpuct_factor,
                cpuct_factor_at_root,
                fpu_value,
                fpu_value_at_root,
                policy_softmax_temp
            );
        }
        Some(CliCommand::PikafishLabelRandom(cmd)) => {
            run_pikafish_label_random(cmd)
                .unwrap_or_else(|err| panic!("pikafish-label-random failed: {err}"));
        }
        Some(CliCommand::PikafishLabelSelfplay(cmd)) => {
            run_pikafish_label_selfplay(cmd)
                .unwrap_or_else(|err| panic!("pikafish-label-selfplay failed: {err}"));
        }
        Some(CliCommand::PikafishPolicyFit(cmd)) => {
            run_pikafish_policy_fit(cmd)
                .unwrap_or_else(|err| panic!("pikafish-policy-fit failed: {err}"));
        }
        Some(CliCommand::PikafishExportTorch(cmd)) => {
            run_pikafish_export_torch(cmd)
                .unwrap_or_else(|err| panic!("pikafish-export-torch failed: {err}"));
        }
        Some(CliCommand::PikafishLabelEval(cmd)) => {
            run_pikafish_label_eval(cmd)
                .unwrap_or_else(|err| panic!("pikafish-label-eval failed: {err}"));
        }
        Some(CliCommand::CheckpointCycles(cmd)) => {
            run_checkpoint_cycles(cmd)
                .unwrap_or_else(|err| panic!("checkpoint-cycles failed: {err}"));
        }
    };
    chineseai::profile::print_report();
}

fn checkpoint_number(path: &Path) -> Option<u64> {
    let name = path.file_name()?.to_str()?;
    let mut last = None;
    let mut value = 0u64;
    let mut active = false;
    for byte in name.bytes() {
        if byte.is_ascii_digit() {
            value = value
                .saturating_mul(10)
                .saturating_add(u64::from(byte - b'0'));
            active = true;
        } else if active {
            last = Some(value);
            value = 0;
            active = false;
        }
    }
    if active { Some(value) } else { last }
}

fn checkpoint_label(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("<invalid-name>")
        .to_string()
}

fn cycle_edge(lower_bounds: &[Vec<Option<f32>>], from: usize, to: usize, margin: f32) -> bool {
    lower_bounds[from][to].is_some_and(|score| score > 0.5 + margin)
}

fn run_checkpoint_cycles(cmd: CheckpointCyclesArgs) -> io::Result<()> {
    let directory = Path::new(&cmd.directory);
    let mut paths = fs::read_dir(directory)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .is_some_and(|extension| extension == "safetensors")
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.contains(&cmd.contains))
        })
        .collect::<Vec<_>>();
    paths.sort_by(|left, right| {
        checkpoint_number(left)
            .cmp(&checkpoint_number(right))
            .then_with(|| checkpoint_label(left).cmp(&checkpoint_label(right)))
    });
    if cmd.min_update_gap > 0 {
        let mut spaced = Vec::new();
        let mut newest_selected: Option<u64> = None;
        for path in paths.into_iter().rev() {
            let number = checkpoint_number(&path);
            let keep = match (newest_selected, number) {
                (Some(newer), Some(current)) => newer.saturating_sub(current) >= cmd.min_update_gap,
                _ => true,
            };
            if keep {
                if number.is_some() {
                    newest_selected = number;
                }
                spaced.push(path);
            }
        }
        spaced.reverse();
        paths = spaced;
    }
    if cmd.max_models > 0 && paths.len() > cmd.max_models {
        paths.drain(..paths.len() - cmd.max_models);
    }
    if paths.len() < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "need at least two matching checkpoints in {}",
                directory.display()
            ),
        ));
    }

    let mut rng = SplitMix64::new(cmd.seed);
    let (positions, opening_mode) = if cmd.opening_book.trim().is_empty() {
        (vec![Position::startpos()], "startpos".to_string())
    } else {
        let book = ObkBook::load(&cmd.opening_book).map_err(sqlite_io_error)?;
        let count = cmd.opening_positions.max(1);
        let positions = (0..count)
            .map(|_| {
                book.random_prefix_position(cmd.opening_plies_min, cmd.opening_plies_max, &mut rng)
            })
            .collect::<Vec<_>>();
        (
            positions,
            format!(
                "obk:{} positions={} plies={}-{}",
                cmd.opening_book, count, cmd.opening_plies_min, cmd.opening_plies_max
            ),
        )
    };
    let positions = Arc::new(positions);
    println!(
        "checkpoint-cycles: models={} pairs={} min_update_gap={} sims={} games_per_pair={} threads={} opening={} margin={:.3} z={:.2}",
        paths.len(),
        if cmd.adjacent_only {
            paths.len() - 1
        } else {
            paths.len() * (paths.len() - 1) / 2
        },
        cmd.min_update_gap,
        cmd.simulations.max(1),
        positions.len() * 2,
        cmd.threads.max(1),
        opening_mode,
        cmd.cycle_margin.max(0.0),
        cmd.confidence_z.max(0.0)
    );
    for (index, path) in paths.iter().enumerate() {
        println!("  model[{index}] {}", path.display());
    }
    let models = paths
        .iter()
        .map(|path| AzNnue::load(path).map(Arc::new))
        .collect::<io::Result<Vec<_>>>()?;
    let mut scores = vec![vec![None; models.len()]; models.len()];
    let mut lower_bounds = vec![vec![None; models.len()]; models.len()];
    for index in 0..models.len() {
        scores[index][index] = Some(0.5);
        lower_bounds[index][index] = Some(0.5);
    }
    let started = Instant::now();
    for newer in 1..models.len() {
        let older_start = if cmd.adjacent_only { newer - 1 } else { 0 };
        for older in older_start..newer {
            let report = run_arena_threads(ArenaThreadConfig {
                candidate: Arc::clone(&models[newer]),
                baseline: Arc::clone(&models[older]),
                eval_positions: Arc::clone(&positions),
                simulations: cmd.simulations.max(1),
                max_plies: cmd.max_plies.max(1),
                rule60_max_ply: Some(120),
                cpuct: 0.9,
                cpuct_at_root: 2.0,
                cpuct_base: 19652.0,
                cpuct_factor: 1.5,
                cpuct_base_at_root: 19652.0,
                cpuct_factor_at_root: 1.5,
                fpu_value: 0.2,
                fpu_value_at_root: 0.1,
                draw_score: 0.0,
                policy_softmax_temp: 1.2,
                thread_count: cmd.threads.max(1),
                seed: cmd.seed ^ ((newer as u64) << 32) ^ older as u64,
            });
            let rate = report.score_rate();
            let se = report.score_rate_standard_error();
            let z = cmd.confidence_z.max(0.0);
            scores[newer][older] = Some(rate);
            scores[older][newer] = Some(1.0 - rate);
            lower_bounds[newer][older] = Some(rate - z * se);
            lower_bounds[older][newer] = Some(1.0 - rate - z * se);
            println!(
                "pair {} > {}: W/L/D={}/{}/{} rate={:.4} se={:.4} lcb(z={:.2})={:.4} elo={:+.1}",
                checkpoint_label(&paths[newer]),
                checkpoint_label(&paths[older]),
                report.wins,
                report.losses,
                report.draws,
                rate,
                se,
                z,
                report.score_rate_lower_bound(z),
                report.elo_diff_vs_even()
            );
        }
    }

    println!("\nSCORE MATRIX — row score against column");
    print!("{:>4}", "row");
    for column in 0..models.len() {
        print!(" {:>7}", column);
    }
    println!();
    for row in 0..models.len() {
        print!("{:>4}", row);
        for column in 0..models.len() {
            match scores[row][column] {
                Some(score) => print!(" {:>7.3}", score),
                None => print!(" {:>7}", "-"),
            }
        }
        println!("  {}", checkpoint_label(&paths[row]));
    }

    let margin = cmd.cycle_margin.max(0.0);
    let mut cycles = 0usize;
    if !cmd.adjacent_only {
        println!("\nNON-TRANSITIVE CYCLES — every edge > {:.3}", 0.5 + margin);
        for a in 0..models.len() {
            for b in a + 1..models.len() {
                for c in b + 1..models.len() {
                    for &(x, y, z) in &[(a, b, c), (a, c, b)] {
                        if cycle_edge(&lower_bounds, x, y, margin)
                            && cycle_edge(&lower_bounds, y, z, margin)
                            && cycle_edge(&lower_bounds, z, x, margin)
                        {
                            cycles += 1;
                            println!(
                                "  {} > {} > {} > {}",
                                checkpoint_label(&paths[x]),
                                checkpoint_label(&paths[y]),
                                checkpoint_label(&paths[z]),
                                checkpoint_label(&paths[x])
                            );
                        }
                    }
                }
            }
        }
        if cycles == 0 {
            println!("  none");
        }
    }

    println!("\nHISTORICAL REGRESSIONS — beats predecessor but loses an older model");
    let mut regressions = 0usize;
    for current in 2..models.len() {
        if !cycle_edge(&lower_bounds, current, current - 1, margin) {
            continue;
        }
        for older in 0..current - 1 {
            if cycle_edge(&lower_bounds, older, current, margin) {
                regressions += 1;
                println!(
                    "  {} beats {}, but loses to {}",
                    checkpoint_label(&paths[current]),
                    checkpoint_label(&paths[current - 1]),
                    checkpoint_label(&paths[older])
                );
            }
        }
    }
    if regressions == 0 {
        println!("  none");
    }
    println!(
        "\nSUMMARY models={} cycles={} regressions={} elapsed={:.1}s",
        models.len(),
        cycles,
        regressions,
        started.elapsed().as_secs_f32()
    );
    Ok(())
}

#[derive(Clone, Debug)]
struct PikafishLabelRow {
    id: i64,
    fen: String,
    bestmove: String,
    best_wdl: [u16; 3],
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
    target_q_sum: f64,
    value_q_sq_sum: f64,
    target_q_sq_sum: f64,
    value_target_cross_sum: f64,
    abs_value_error_sum: f64,
    raw_value_pairs: usize,
    raw_value_q_sum: f64,
    raw_target_q_sum: f64,
    raw_value_q_sq_sum: f64,
    raw_target_q_sq_sum: f64,
    raw_value_target_cross_sum: f64,
    raw_abs_value_error_sum: f64,
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
        self.target_q_sum += other.target_q_sum;
        self.value_q_sq_sum += other.value_q_sq_sum;
        self.target_q_sq_sum += other.target_q_sq_sum;
        self.value_target_cross_sum += other.value_target_cross_sum;
        self.abs_value_error_sum += other.abs_value_error_sum;
        self.raw_value_pairs += other.raw_value_pairs;
        self.raw_value_q_sum += other.raw_value_q_sum;
        self.raw_target_q_sum += other.raw_target_q_sum;
        self.raw_value_q_sq_sum += other.raw_value_q_sq_sum;
        self.raw_target_q_sq_sum += other.raw_target_q_sq_sum;
        self.raw_value_target_cross_sum += other.raw_value_target_cross_sum;
        self.raw_abs_value_error_sum += other.raw_abs_value_error_sum;
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

    fn value_mae_wdl_q(&self) -> f32 {
        (self.abs_value_error_sum / self.value_count().max(1) as f64) as f32
    }

    fn target_q(wdl: [u16; 3]) -> f64 {
        (f64::from(wdl[0]) - f64::from(wdl[2])) / 1000.0
    }

    fn push_value_pair(&mut self, value_q: f32, wdl: [u16; 3]) {
        let target = Self::target_q(wdl);
        let value = value_q as f64;
        self.value_pairs += 1;
        self.value_q_sum += value;
        self.target_q_sum += target;
        self.value_q_sq_sum += value * value;
        self.target_q_sq_sum += target * target;
        self.value_target_cross_sum += value * target;
        self.abs_value_error_sum += (value - target).abs();
    }

    fn value_count(&self) -> usize {
        self.value_pairs
    }

    fn push_raw_value_pair(&mut self, value_q: f32, wdl: [u16; 3]) {
        let target = Self::target_q(wdl);
        let value = value_q as f64;
        self.raw_value_pairs += 1;
        self.raw_value_q_sum += value;
        self.raw_target_q_sum += target;
        self.raw_value_q_sq_sum += value * value;
        self.raw_target_q_sq_sum += target * target;
        self.raw_value_target_cross_sum += value * target;
        self.raw_abs_value_error_sum += (value - target).abs();
    }

    fn raw_value_mae_wdl_q(&self) -> f32 {
        (self.raw_abs_value_error_sum / self.raw_value_pairs.max(1) as f64) as f32
    }

    fn raw_value_corr(&self) -> f64 {
        let n = self.raw_value_pairs as f64;
        if n <= 1.0 {
            return 0.0;
        }
        let cov =
            self.raw_value_target_cross_sum - self.raw_value_q_sum * self.raw_target_q_sum / n;
        let left = self.raw_value_q_sq_sum - self.raw_value_q_sum * self.raw_value_q_sum / n;
        let right = self.raw_target_q_sq_sum - self.raw_target_q_sum * self.raw_target_q_sum / n;
        if left <= 0.0 || right <= 0.0 {
            0.0
        } else {
            cov / (left * right).sqrt()
        }
    }

    fn value_corr(&self) -> f64 {
        let n = self.value_count() as f64;
        if n <= 1.0 {
            return 0.0;
        }
        let cov = self.value_target_cross_sum - self.value_q_sum * self.target_q_sum / n;
        let left = self.value_q_sq_sum - self.value_q_sum * self.value_q_sum / n;
        let right = self.target_q_sq_sum - self.target_q_sum * self.target_q_sum / n;
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
    let stats = evaluate_pikafish_labels_parallel(
        Arc::new(model),
        rows,
        fixed_az_search_limits(
            cmd.simulations.max(1),
            cmd.seed,
            cmd.cpuct.max(0.0),
            cmd.cpuct_at_root.max(0.0),
            cmd.max_depth,
            cmd.policy_softmax_temp,
        ),
        cmd.threads,
    )?;

    println!(
        "pikafish-label-eval: model={} sqlite={} evaluated={} legal_labels={} value_labels={} sims={} threads={} cpuct={}/{} policy_temp={} search_top1={:.3}% search_top2={:.3}% search_top4={:.3}% search_top8={:.3}% raw_prior_top1={:.3}% raw_value_corr={:.4} raw_value_mae_wdl_q={:.4} search_value_corr={:.4} search_value_mae_wdl_q={:.4} elapsed={:.1}s",
        cmd.model,
        cmd.sqlite,
        stats.count,
        stats.legal_bestmove,
        stats.value_count(),
        cmd.simulations.max(1),
        cmd.threads.max(1),
        cmd.cpuct,
        cmd.cpuct_at_root,
        cmd.policy_softmax_temp,
        100.0 * stats.top1_rate(),
        100.0 * stats.top2_rate(),
        100.0 * stats.top4_rate(),
        100.0 * stats.top8_rate(),
        100.0 * stats.prior_top1_rate(),
        stats.raw_value_corr(),
        stats.raw_value_mae_wdl_q(),
        stats.value_corr(),
        stats.value_mae_wdl_q(),
        started.elapsed().as_secs_f32()
    );
    Ok(())
}

fn evaluate_pikafish_labels(
    model: &AzNnue,
    rows: &[PikafishLabelRow],
    search_limits: AzSearchLimits,
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
        let rule_history = position.initial_rule_history();
        let legal_moves = position.legal_moves_with_rules(&rule_history);
        let raw_value = model.evaluate_value_with_rules(&position, &rule_history, &legal_moves);
        stats.push_raw_value_pair(raw_value, row.best_wdl);
        let result = alphazero_search(
            &position,
            model,
            AzSearchLimits {
                seed: search_limits.seed ^ row.id as u64,
                ..search_limits
            },
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
        stats.push_value_pair(result.value_q, row.best_wdl);
        progress(offset + 1, rows.len());
    }
    Ok(stats)
}

fn evaluate_pikafish_labels_parallel(
    model: Arc<AzNnue>,
    rows: Vec<PikafishLabelRow>,
    search_limits: AzSearchLimits,
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
            evaluate_pikafish_labels(&model, &shard, search_limits, |_, _| {})
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
        "SELECT id, fen, bestmove, wdl_win, wdl_draw, wdl_loss FROM pikafish_labels ORDER BY id"
            .to_string();
    if limit > 0 {
        query.push_str(" LIMIT ?1");
        let mut stmt = conn.prepare(&query)?;
        stmt.query_map(params![limit as i64], |row| {
            Ok(PikafishLabelRow {
                id: row.get(0)?,
                fen: row.get(1)?,
                bestmove: row.get(2)?,
                best_wdl: [row.get(3)?, row.get(4)?, row.get(5)?],
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
                best_wdl: [row.get(3)?, row.get(4)?, row.get(5)?],
            })
        })?
        .collect()
    }
}

#[derive(Clone, Debug, Default)]
struct PikafishPv {
    multipv: usize,
    depth: u32,
    nodes: u64,
    score_cp: Option<i32>,
    mate: Option<i32>,
    wdl: Option<[u16; 3]>,
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
        self.write_line("setoption name Threads value 1")?;
        self.write_line("setoption name Repetition Rule value ChineseRule")?;
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
    let mut depth = 0u32;
    let mut nodes = 0u64;
    let mut score_cp = None;
    let mut mate = None;
    let mut wdl = None;
    let mut moves = Vec::new();
    let mut i = 0usize;
    while i < parts.len() {
        match parts[i] {
            "depth" if i + 1 < parts.len() => {
                depth = parts[i + 1].parse().ok()?;
                i += 2;
            }
            "nodes" if i + 1 < parts.len() => {
                nodes = parts[i + 1].parse().unwrap_or(0);
                i += 2;
            }
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
            "wdl" if i + 3 < parts.len() => {
                wdl = Some([
                    parts[i + 1].parse().ok()?,
                    parts[i + 2].parse().ok()?,
                    parts[i + 3].parse().ok()?,
                ]);
                i += 4;
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
        depth,
        nodes,
        score_cp,
        mate,
        wdl,
        moves,
    })
}

fn label_fens_parallel(
    exe: &Path,
    fens: &[String],
    depth: u32,
    threads: usize,
) -> io::Result<Vec<(usize, String, Vec<PikafishPv>)>> {
    let worker_count = threads.max(1).min(fens.len());
    let fens = Arc::new(fens.to_vec());
    let exe = Arc::new(exe.to_path_buf());
    let mut handles = Vec::with_capacity(worker_count);
    for worker in 0..worker_count {
        let fens = Arc::clone(&fens);
        let exe = Arc::clone(&exe);
        handles.push(thread::spawn(move || -> io::Result<Vec<_>> {
            let mut engine = PikafishLabelUci::spawn(&exe)?;
            let mut out = Vec::new();
            for index in (worker..fens.len()).step_by(worker_count) {
                let (bestmove, pvs) = engine.query(&fens[index], depth)?;
                out.push((index, bestmove, pvs));
            }
            Ok(out)
        }));
    }
    let mut out = Vec::with_capacity(fens.len());
    for handle in handles {
        out.extend(
            handle
                .join()
                .map_err(|_| io::Error::other("Pikafish label worker panicked"))??,
        );
    }
    out.sort_by_key(|row| row.0);
    Ok(out)
}

fn run_pikafish_label_random(cmd: PikafishLabelRandomArgs) -> io::Result<()> {
    let fens_path = Path::new(&cmd.fens);
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
            wdl_win INTEGER NOT NULL,
            wdl_draw INTEGER NOT NULL,
            wdl_loss INTEGER NOT NULL,
            nodes INTEGER NOT NULL DEFAULT 0,
            best_pv TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_pikafish_labels_bestmove ON pikafish_labels(bestmove);",
    )
    .map_err(sqlite_io_error)?;
    let completed = {
        let mut stmt = conn
            .prepare("SELECT fen FROM pikafish_labels WHERE depth >= ?1")
            .map_err(sqlite_io_error)?;
        let rows = stmt
            .query_map([cmd.depth.max(1)], |row| row.get::<_, String>(0))
            .map_err(sqlite_io_error)?;
        rows.collect::<Result<HashSet<_>, _>>()
            .map_err(sqlite_io_error)?
    };
    let pending = fens
        .iter()
        .enumerate()
        .filter(|(_, fen)| !completed.contains(*fen))
        .map(|(index, fen)| (index, fen.clone()))
        .collect::<Vec<_>>();

    println!(
        "pikafish-label-random: labeling {} pending of {} positions depth={} workers={}",
        pending.len(),
        fens.len(),
        cmd.depth.max(1),
        cmd.threads.max(1).min(pending.len().max(1))
    );
    let mut done = fens.len() - pending.len();
    for chunk in pending.chunks(256) {
        let chunk_fens = chunk.iter().map(|(_, fen)| fen.clone()).collect::<Vec<_>>();
        let labeled = label_fens_parallel(
            Path::new(&cmd.pikafish_exe),
            &chunk_fens,
            cmd.depth.max(1),
            cmd.threads,
        )?;
        let tx = conn.transaction().map_err(sqlite_io_error)?;
        for (local_index, bestmove, pvs) in labeled {
            let (index, fen) = &chunk[local_index];
            let position = Position::from_fen(fen).map_err(|err| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid FEN at {}: {err}", index + 1),
                )
            })?;
            let best = pvs.iter().find(|pv| pv.multipv == 1);
            let best_pv = best.map(|pv| pv.moves.join(" ")).unwrap_or_default();
            let side_to_move = match position.side_to_move() {
                chineseai::xiangqi::Color::Red => "w",
                chineseai::xiangqi::Color::Black => "b",
            };
            let wdl = best.and_then(|pv| pv.wdl).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Pikafish returned no WDL at position {}", index + 1),
                )
            })?;
            tx.execute(
                "INSERT INTO pikafish_labels (
                id, fen, side_to_move, depth, bestmove, best_score_cp, best_mate,
                wdl_win, wdl_draw, wdl_loss, nodes, best_pv, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, CURRENT_TIMESTAMP)
            ON CONFLICT(fen) DO UPDATE SET
                side_to_move=excluded.side_to_move,
                depth=excluded.depth,
                bestmove=excluded.bestmove,
                best_score_cp=excluded.best_score_cp,
                best_mate=excluded.best_mate,
                wdl_win=excluded.wdl_win,
                wdl_draw=excluded.wdl_draw,
                wdl_loss=excluded.wdl_loss,
                nodes=excluded.nodes,
                best_pv=excluded.best_pv,
                updated_at=CURRENT_TIMESTAMP",
                params![
                    *index as i64,
                    fen,
                    side_to_move,
                    best.map(|pv| pv.depth).unwrap_or(cmd.depth.max(1)) as i64,
                    &bestmove,
                    best.and_then(|pv| pv.score_cp).map(i64::from),
                    best.and_then(|pv| pv.mate).map(i64::from),
                    i64::from(wdl[0]),
                    i64::from(wdl[1]),
                    i64::from(wdl[2]),
                    best.map(|pv| pv.nodes as i64).unwrap_or(0),
                    &best_pv,
                ],
            )
            .map_err(sqlite_io_error)?;
            done += 1;
        }
        tx.commit().map_err(sqlite_io_error)?;
        println!(
            "pikafish-label-random: labeled {}/{} -> {}",
            done,
            fens.len(),
            sqlite_path.display()
        );
    }
    Ok(())
}

fn run_pikafish_label_selfplay(cmd: PikafishLabelSelfplayArgs) -> io::Result<()> {
    let model = AzNnue::load(&cmd.model)?;
    let fens_path = Path::new(&cmd.fens);
    let mut fens = if fens_path.exists() {
        fs::read_to_string(fens_path)?
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(str::to_owned)
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let mut unique = fens.iter().cloned().collect::<HashSet<_>>();
    let base = AzLoopFileConfig::default();
    let started = Instant::now();
    let mut batch = 0u64;
    while fens.len() < cmd.count.max(1) {
        let remaining = cmd.count - fens.len();
        let games = (remaining.div_ceil(cmd.max_plies.max(1))).clamp(1, cmd.workers.max(1) * 2);
        let mut config = build_az_loop_config(
            &base,
            cmd.seed ^ batch.wrapping_mul(0x9E37_79B9_7F4A_7C15),
            cmd.workers.max(1),
            0,
            &Arc::default(),
        );
        config.games = games;
        config.simulations = cmd.simulations.max(1);
        config.max_plies = cmd.max_plies.max(1);
        config.opening_start_fraction = 0.0;
        config.record_fens = true;
        config.mirror_probability = 0.0;
        let data = generate_selfplay_data(&model, &config);
        for fen in data.position_fens {
            if unique.insert(fen.clone()) {
                fens.push(fen);
                if fens.len() == cmd.count {
                    break;
                }
            }
        }
        batch += 1;
        if let Some(parent) = fens_path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        fs::write(fens_path, format!("{}\n", fens.join("\n")))?;
        println!(
            "pikafish-label-selfplay: sampled {}/{} unique positions, elapsed={:.1}s",
            fens.len(),
            cmd.count,
            started.elapsed().as_secs_f32()
        );
    }
    run_pikafish_label_random(PikafishLabelRandomArgs {
        pikafish_exe: cmd.pikafish_exe,
        fens: cmd.fens,
        sqlite: cmd.sqlite,
        count: cmd.count,
        seed: cmd.seed,
        min_plies: 0,
        max_plies: cmd.max_plies,
        depth: cmd.depth,
        threads: cmd.pikafish_threads,
        regenerate: false,
    })
}

fn load_pikafish_training_samples(path: &str) -> io::Result<Vec<AzTrainingSample>> {
    let conn = Connection::open(path).map_err(sqlite_io_error)?;
    let mut stmt = conn
        .prepare(
            "SELECT fen, bestmove, wdl_win, wdl_draw, wdl_loss FROM pikafish_labels ORDER BY id",
        )
        .map_err(sqlite_io_error)?;
    let rows = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                [
                    row.get::<_, f32>(2)?,
                    row.get::<_, f32>(3)?,
                    row.get::<_, f32>(4)?,
                ],
            ))
        })
        .map_err(sqlite_io_error)?;
    let mut samples = Vec::new();
    for row in rows {
        let (fen, bestmove, mut wdl) = row.map_err(sqlite_io_error)?;
        let position = Position::from_fen(&fen)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        let Some(best) = position.parse_uci_move(&bestmove) else {
            continue;
        };
        let legal = position.legal_moves();
        let side = position.side_to_move();
        let move_indices = legal
            .iter()
            .map(|&mv| chineseai::az::dense_move_index(canonical_move(side, mv)))
            .collect::<Vec<_>>();
        let Some(best_index) = legal.iter().position(|&mv| mv == best) else {
            continue;
        };
        let mut policy = vec![0.0; legal.len()];
        policy[best_index] = 1.0;
        let sum = wdl.iter().sum::<f32>().max(1.0);
        wdl.iter_mut().for_each(|x| *x /= sum);
        samples.push(AzTrainingSample {
            features: extract_sparse_features_az(&position),
            rule_context: chineseai::az::rule_context_features(
                &position,
                &position.initial_rule_history(),
            ),
            move_indices,
            policy,
            value_wdl: wdl,
            value: wdl[0] - wdl[2],
            side_sign: if side == chineseai::xiangqi::Color::Red {
                1.0
            } else {
                -1.0
            },
            policy_weight: 1.0,
            value_weight: 1.0,
            search_simulations: 0,
            meta: AzSampleMeta::default(),
        });
    }
    Ok(samples)
}

fn run_pikafish_policy_fit(cmd: PikafishPolicyFitArgs) -> io::Result<()> {
    let mut samples = load_pikafish_training_samples(&cmd.sqlite)?;
    if samples.len() < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "need at least two labels",
        ));
    }
    let mut rng = SplitMix64::new(cmd.seed);
    for index in (1..samples.len()).rev() {
        let other = (rng.next_u64() as usize) % (index + 1);
        samples.swap(index, other);
    }
    let validation_len =
        ((samples.len() as f32 * cmd.validation_fraction.clamp(0.01, 0.5)) as usize).max(1);
    let validation = samples.split_off(samples.len() - validation_len);
    let train = samples;
    fs::create_dir_all(&cmd.output_dir)?;
    println!(
        "policy-fit: train={} validation={} budget={}s/arch",
        train.len(),
        validation.len(),
        cmd.wall_seconds
    );
    for &hidden in &cmd.hidden {
        let mut model = AzNnue::random_with_arch(
            chineseai::az::AzNnueArch::with_hidden_size(hidden.max(1)),
            cmd.seed,
        );
        let started = Instant::now();
        let mut processed = 0usize;
        let mut offset = 0usize;
        while processed == 0 || started.elapsed() < Duration::from_secs(cmd.wall_seconds.max(1)) {
            let take = 8192.min(train.len());
            let batch = (0..take)
                .map(|i| train[(offset + i) % train.len()].clone())
                .collect::<Vec<_>>();
            let mut train_rng = SplitMix64::new(cmd.seed ^ processed as u64 ^ hidden as u64);
            train_samples_weighted(
                &mut model,
                &batch,
                1,
                cmd.lr,
                cmd.batch_size.max(1),
                &mut train_rng,
                AzTrainLossWeights::default(),
            )
            .map_err(io::Error::other)?;
            processed += take;
            offset = (offset + take) % train.len();
        }
        let elapsed = started.elapsed().as_secs_f32();
        let mut eval_model = model.clone();
        let mut eval_rng = SplitMix64::new(cmd.seed ^ 0xD1B5_4A32_D192_ED03);
        let stats = train_samples_weighted(
            &mut eval_model,
            &validation,
            1,
            1e-12,
            cmd.batch_size.max(1),
            &mut eval_rng,
            AzTrainLossWeights::default(),
        )
        .map_err(io::Error::other)?;
        let output = Path::new(&cmd.output_dir).join(format!("h{hidden}.safetensors"));
        model.save(&output)?;
        println!(
            "policy-fit-result: hidden={} seconds={:.2} processed={} samples_per_sec={:.0} validation_policy_ce={:.6} validation_value_ce={:.6} output={}",
            hidden,
            elapsed,
            processed,
            processed as f32 / elapsed.max(1e-6),
            stats.policy_ce,
            stats.value_loss,
            output.display()
        );
    }
    Ok(())
}

fn run_pikafish_export_torch(cmd: PikafishExportTorchArgs) -> io::Result<()> {
    let conn = Connection::open(&cmd.sqlite).map_err(sqlite_io_error)?;
    let mut stmt = conn
        .prepare(
            "SELECT fen, best_pv, wdl_win, wdl_draw, wdl_loss FROM pikafish_labels ORDER BY id",
        )
        .map_err(sqlite_io_error)?;
    let rows = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                [
                    row.get::<_, f32>(2)?,
                    row.get::<_, f32>(3)?,
                    row.get::<_, f32>(4)?,
                ],
            ))
        })
        .map_err(sqlite_io_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(sqlite_io_error)?;
    let output = Path::new(&cmd.output);
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(fs::File::create(output)?);
    writer.write_all(b"XQPF")?;
    writer.write_u32::<LittleEndian>(3)?;
    writer.write_u32::<LittleEndian>(0)?;
    let mut exported = 0u32;
    for (group, (fen, pv, raw_wdl)) in rows.iter().enumerate() {
        let mut position = Position::from_fen(fen)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        let pv = pv
            .split_whitespace()
            .take(cmd.pv_plies.max(1))
            .collect::<Vec<_>>();
        for (ply, bestmove) in pv.into_iter().enumerate() {
            if cmd.pv_plies == 0 && ply > 0 {
                break;
            }
            let Some(best_move) = position.parse_uci_move(bestmove) else {
                break;
            };
            let legal = position.legal_moves();
            let Some(best) = legal.iter().position(|&mv| mv == best_move) else {
                break;
            };
            let side = position.side_to_move();
            let features = extract_sparse_features_az(&position);
            let move_indices = legal
                .iter()
                .map(|&mv| chineseai::az::dense_move_index(canonical_move(side, mv)))
                .collect::<Vec<_>>();
            let sum = raw_wdl.iter().sum::<f32>().max(1.0);
            let mut wdl = raw_wdl.map(|value| value / sum);
            if ply % 2 == 1 {
                wdl.swap(0, 2);
            }
            writer.write_u32::<LittleEndian>(group as u32)?;
            writer.write_u8(ply as u8)?;
            writer.write_u16::<LittleEndian>(features.len() as u16)?;
            writer.write_u16::<LittleEndian>(move_indices.len() as u16)?;
            writer.write_u16::<LittleEndian>(best as u16)?;
            for value in wdl {
                writer.write_f32::<LittleEndian>(value)?;
            }
            for feature in features {
                writer.write_u16::<LittleEndian>(feature as u16)?;
            }
            for mv in move_indices {
                writer.write_u16::<LittleEndian>(mv as u16)?;
            }
            for &mv in &legal {
                writer.write_u8(u8::from(position.gives_check_after_move_fast(mv)))?;
            }
            exported += 1;
            position.make_move(best_move);
        }
    }
    writer.flush()?;
    writer.seek(SeekFrom::Start(8))?;
    writer.write_u32::<LittleEndian>(exported)?;
    println!(
        "pikafish-export-torch: samples={} output={}",
        exported,
        output.display()
    );
    Ok(())
}

fn sqlite_io_error(err: rusqlite::Error) -> io::Error {
    io::Error::other(err.to_string())
}

#[cfg(test)]
mod reporting_tests {
    use super::*;
    use chineseai::az::AzSampleMeta;

    #[test]
    fn az_search_defaults_match_selfplay_and_uci_search() {
        let cli =
            Cli::try_parse_from(["chineseai", "az-search", "model.safetensors", "3200"]).unwrap();
        let Some(CliCommand::AzSearch(args)) = cli.command else {
            panic!("expected az-search command");
        };
        assert_eq!(args.cpuct, 0.9);
        assert_eq!(args.cpuct_at_root, 2.0);
        assert_eq!(args.cpuct_factor, 1.5);
        assert_eq!(args.cpuct_factor_at_root, 1.5);
        assert_eq!(args.fpu_value, 0.20);
        assert_eq!(args.fpu_value_at_root, 0.10);
        assert_eq!(args.policy_softmax_temp, 1.2);
    }

    fn reporting_sample(generation: u32, policy: Vec<f32>) -> AzTrainingSample {
        AzTrainingSample {
            features: vec![0],
            rule_context: [0.0; chineseai::az::RULE_CONTEXT_SIZE],
            move_indices: (0..policy.len()).collect(),
            policy,
            value_wdl: [0.0, 1.0, 0.0],
            value: 0.0,
            side_sign: 1.0,
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
        let stats = train_batch_source_stats(&samples, 4_000, 1, 1);

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
        stats.push_value_pair(0.25, [500, 500, 0]);

        assert_eq!(stats.value_count(), 1);
        assert!((stats.value_mae_wdl_q() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn arena_adds_random_takeover_positions() {
        let mut config = AzLoopFileConfig::default();
        config.arena_opening_book.clear();
        config.arena_random_positions = 8;
        config.arena_random_plies_min = 4;
        config.arena_random_plies_max = 12;

        let (positions, mode) = build_arena_start_positions(&config, 7);

        assert_eq!(positions.len(), 8);
        assert_eq!(mode, "random(count=8,plies=4-12)");
        assert!(positions.iter().all(|position| {
            position.has_general(chineseai::xiangqi::Color::Red)
                && position.has_general(chineseai::xiangqi::Color::Black)
                && !position.legal_moves().is_empty()
        }));

        let (next_fold, _) = build_arena_start_positions(&config, 17);
        assert_ne!(
            positions.iter().map(Position::hash).collect::<Vec<_>>(),
            next_fold.iter().map(Position::hash).collect::<Vec<_>>()
        );
    }

    #[test]
    fn arena_history_uses_logarithmic_champion_offsets() {
        assert_eq!(historical_anchor_index(1, 0), None);
        assert_eq!(historical_anchor_index(3, 0), Some(0));
        assert_eq!(historical_anchor_index(10, 0), Some(7));
        assert_eq!(historical_anchor_index(10, 1), Some(5));
        assert_eq!(historical_anchor_index(10, 2), Some(1));
        assert_eq!(historical_anchor_index(10, 3), Some(7));
    }

    #[test]
    fn arena_gate_is_three_state_and_uses_confidence_bounds() {
        assert_eq!(
            arena_gate_position_counts(1_000, true, true),
            (600, 200, 200)
        );
        assert_eq!(
            arena_gate_position_counts(1_000, true, false),
            (800, 200, 0)
        );
        assert_eq!(
            arena_gate_position_counts(1_000, false, false),
            (1_000, 0, 0)
        );

        let report = |wins, losses| AzArenaReport {
            wins,
            losses,
            ..AzArenaReport::default()
        };
        let current = report(120, 80);
        let previous = report(100, 100);
        let anchor = report(110, 90);
        assert_eq!(
            arena_gate_decision(&current, Some(&previous), Some(&anchor), 0.50, 1.28),
            ArenaGateDecision::Promote
        );

        let uncertain = report(102, 98);
        assert_eq!(
            arena_gate_decision(&uncertain, None, None, 0.50, 1.28),
            ArenaGateDecision::Continue
        );

        let all_draws = AzArenaReport {
            draws: 200,
            ..AzArenaReport::default()
        };
        assert_eq!(
            arena_gate_decision(&all_draws, None, None, 0.50, 1.28),
            ArenaGateDecision::Continue
        );

        let regressed_anchor = report(70, 130);
        assert_eq!(
            arena_gate_decision(
                &current,
                Some(&previous),
                Some(&regressed_anchor),
                0.50,
                1.28,
            ),
            ArenaGateDecision::Reject
        );

        let regressed_current = report(70, 130);
        assert_eq!(
            arena_gate_decision(
                &regressed_current,
                Some(&previous),
                Some(&anchor),
                0.50,
                1.28,
            ),
            ArenaGateDecision::Reject
        );

        // Each historical opponent is individually inconclusive, but their
        // combined 800 games prove the same regression seen at update 3760.
        let previous_split = AzArenaReport {
            wins: 141,
            losses: 163,
            draws: 96,
            ..AzArenaReport::default()
        };
        let anchor_split = AzArenaReport {
            wins: 147,
            losses: 160,
            draws: 93,
            ..AzArenaReport::default()
        };
        assert!(previous_split.score_rate_upper_bound(1.28) >= 0.50);
        assert!(anchor_split.score_rate_upper_bound(1.28) >= 0.50);
        assert_eq!(
            arena_gate_decision(
                &current,
                Some(&previous_split),
                Some(&anchor_split),
                0.50,
                1.28,
            ),
            ArenaGateDecision::Reject
        );
    }
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

fn parse_position(text: &str) -> Position {
    if text.trim().is_empty() || text == "startpos" {
        Position::startpos()
    } else {
        Position::from_fen(text).unwrap_or_else(|err| {
            panic!("invalid FEN `{text}`: {err}");
        })
    }
}
