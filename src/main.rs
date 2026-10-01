#[cfg(all(target_os = "linux", not(target_env = "musl")))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

mod az_loop_config;
mod training_console;

use az_loop_config::{AzLoopFileConfig, DEFAULT_AZ_LOOP_CONFIG, load_or_create_az_loop_config};

use chineseai::version::AZ_LOOP_PROGRESS_VERSION;

use chineseai::{
    az::{
        AzArenaConfig, AzArenaReport, AzExperiencePool, AzLoopConfig, AzLoopReport, AzNnue,
        AzSearchLimits, AzSelfplayData, AzTrainLossWeights, POLICY_SPARSE_MAIN_SIZE,
        POLICY_TACTICAL_EXACT_SIZE, Px0ReplaySampler, SplitMix64, alphazero_search,
        alphazero_search_trace_with_rules, alphazero_search_with_rules, generate_selfplay_data,
        play_arena_games_from_positions, policy_target_entropy, train_samples_weighted_owned,
    },
    pikafish_match::{VsPikafishConfig, run_vs_pikafish},
    px0_opening_book::Px0OpeningBook,
    xiangqi::{Move, Position},
};
use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::{
    fs, io,
    path::{Path, PathBuf},
    sync::{
        Arc, RwLock,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};
use tensorboard_rs::summary_writer::SummaryWriter;

const DEFAULT_VS_PIKAFISH_DEPTH: u32 = 10;
const DEFAULT_VS_PIKAFISH_GAMES: usize = 20;
const DEFAULT_VS_PIKAFISH_PARALLEL_GAMES: usize = 128;

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
    /// Scale one policy component for structural ablation.
    AzPolicyScale(AzPolicyScaleArgs),
    /// Search one position and print policy/debug details.
    AzSearch(AzSearchArgs),
    /// Benchmark fixed-position search speed.
    AzBench(AzBenchArgs),
    /// Run self-play training from a TOML config.
    AzLoop(AzLoopArgs),
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
struct AzPolicyScaleArgs {
    /// Existing model path.
    input: String,
    /// Modified model path.
    output: String,
    /// Policy component to scale.
    #[arg(long, value_enum)]
    component: PolicyComponent,
    /// Multiplier applied to the selected component.
    #[arg(long)]
    scale: f32,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PolicyComponent {
    Exact,
    Capture,
    Factor,
    Tactical,
    TacticalExact,
    TacticalFactor,
    Accumulator,
    Context,
    Consequence,
    MoveBias,
    ThreatContext,
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
    #[arg(default_value_t = 800)]
    simulations: usize,
    /// Non-root PUCT init.
    #[arg(default_value_t = 1.0)]
    cpuct: f32,
    /// Root PUCT init.
    #[arg(long, default_value_t = 1.9)]
    cpuct_at_root: f32,
    /// Non-root first-play urgency reduction.
    #[arg(long, default_value_t = 0.23)]
    fpu_value: f32,
    /// Root absolute first-play urgency.
    #[arg(long, default_value_t = 1.0)]
    fpu_value_at_root: f32,
    /// Divisor applied to policy logits before root search; above 1 flattens priors.
    #[arg(long, default_value_t = 1.4)]
    policy_softmax_temp: f32,
    /// Dynamic PUCT base.
    #[arg(long, default_value_t = 38739.0)]
    cpuct_base: f32,
    /// Dynamic PUCT growth factor.
    #[arg(long, default_value_t = 3.894)]
    cpuct_factor: f32,
    /// Root dynamic PUCT base.
    #[arg(long, default_value_t = 38739.0)]
    cpuct_base_at_root: f32,
    /// Root dynamic PUCT growth factor.
    #[arg(long, default_value_t = 3.894)]
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
    #[arg(default_value_t = 800)]
    simulations: usize,
    /// Number of repeated searches.
    #[arg(default_value_t = 100)]
    repeat: usize,
    /// PUCT constant for AlphaZero search.
    #[arg(default_value_t = 1.0)]
    cpuct: f32,
    /// FEN string, or startpos if omitted.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    fen: Vec<String>,
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
  chineseai vs-pikafish ./tools/pikafish model.safetensors --opening-book book.pgn.gz")]
struct VsPikafishArgs {
    /// Pikafish UCI executable path.
    pikafish_exe: String,
    /// ChineseAI AZ-NNUE model path.
    model: String,
    /// ChineseAI MCTS simulations per move.
    #[arg(short = 's', long, default_value = "800")]
    simulations: Option<usize>,
    /// ChineseAI PUCT constant.
    #[arg(long, default_value_t = 1.0)]
    cpuct: f32,
    /// ChineseAI root PUCT constant.
    #[arg(long, default_value_t = 1.9)]
    cpuct_at_root: f32,
    /// ChineseAI dynamic PUCT base.
    #[arg(long, default_value_t = 38739.0)]
    cpuct_base: f32,
    /// ChineseAI dynamic PUCT growth factor.
    #[arg(long, default_value_t = 3.894)]
    cpuct_factor: f32,
    /// ChineseAI root dynamic PUCT base.
    #[arg(long, default_value_t = 38739.0)]
    cpuct_base_at_root: f32,
    /// ChineseAI root dynamic PUCT growth factor.
    #[arg(long, default_value_t = 3.894)]
    cpuct_factor_at_root: f32,
    /// ChineseAI non-root first-play urgency reduction.
    #[arg(long, default_value_t = 0.23)]
    fpu_value: f32,
    /// ChineseAI root first-play urgency reduction.
    #[arg(long, default_value_t = 1.0)]
    fpu_value_at_root: f32,
    /// Divisor applied to ChineseAI policy logits before search.
    #[arg(long, default_value_t = 1.4)]
    policy_softmax_temp: f32,
    /// Draw after this many plies.
    #[arg(long, default_value_t = 200)]
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
    /// Px0 book.pgn.gz used to generate random start positions. Empty uses startpos.
    #[arg(long, default_value = "book.pgn.gz")]
    opening_book: String,
    /// Number of shuffled FEN positions to take from the Px0 book.
    #[arg(long, default_value_t = 1000)]
    opening_positions: usize,
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
    generated_games: u64,
    generated_samples: u64,
}

impl Default for AzLoopProgressState {
    fn default() -> Self {
        Self {
            format_version: AZ_LOOP_PROGRESS_VERSION,
            next_update: 1,
            nemesis_update: None,
            generated_games: 0,
            generated_samples: 0,
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

fn save_az_loop_progress_pair(
    config_path: &str,
    next_update: usize,
    nemesis_update: Option<u64>,
    generated_games: u64,
    generated_samples: u64,
) {
    save_az_loop_progress(
        config_path,
        &AzLoopProgressState {
            next_update,
            nemesis_update,
            generated_games,
            generated_samples,
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
            "shuf{}_rrw{}_sgd09nesterov_cp{}_cpr{}_fv{}_fvr{}_pst{}_tb{}_teg{}_tdd{}_tde{}_rc{}_",
            "tspu{}_mp{}_cpi{}_ai{}_as{}_acp{}_acpr{}_apst{}_rda{}_ref{}_sd{}"
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
        config.shuffle_size,
        config.replay_recent_games,
        f32_slug(config.cpuct),
        f32_slug(config.cpuct_at_root),
        f32_slug(config.fpu_value),
        f32_slug(config.fpu_value_at_root),
        f32_slug(config.policy_softmax_temp),
        f32_slug(config.temperature_start),
        f32_slug(config.temperature_endgame),
        config.temperature_decay_delay_plies,
        config.temperature_decay_plies,
        config.replay_capacity,
        config.train_samples_per_update,
        f32_slug(config.mirror_probability),
        config.checkpoint_interval,
        config.arena_interval,
        config.arena_simulations,
        f32_slug(config.arena_cpuct),
        f32_slug(config.arena_cpuct_at_root),
        f32_slug(config.arena_policy_softmax_temp),
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

fn optimizer_checkpoint_path(model_path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.sgd.safetensors", model_path.display()))
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
        let optimizer_path = optimizer_checkpoint_path(&path);
        fs::remove_file(path)?;
        if optimizer_path.exists() {
            fs::remove_file(optimizer_path)?;
        }
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
    model: Arc<AzNnue>,
}

fn publish_selfplay_model(
    shared_model: &RwLock<SharedSelfplayModel>,
    model: Arc<AzNnue>,
    learner_update: usize,
) -> u64 {
    let mut shared = shared_model
        .write()
        .unwrap_or_else(|_| panic!("shared selfplay model poisoned"));
    shared.model = model;
    shared.version = shared.version.wrapping_add(1);
    shared.learner_update = learner_update.min(u32::MAX as usize) as u32;
    shared.version
}

#[derive(Default)]
struct PendingTrainingData {
    collection_seconds: f32,
    selfplay: AzSelfplayData,
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
        temperature_cutoff_plies: config.temperature_cutoff_plies,
        temperature_visit_offset: config.temperature_visit_offset,
        temperature_endgame: config.temperature_endgame,
        temperature_decay_delay_plies: config.temperature_decay_delay_plies,
        temperature_decay_plies: config.temperature_decay_plies,
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
        fpu_absolute_at_root: config.fpu_absolute_at_root,
        minimum_kldgain_per_node: config.minimum_kldgain_per_node,
        draw_score: config.draw_score,
        policy_softmax_temp: config.policy_softmax_temp,
        opening_positions: Arc::clone(opening_positions),
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
    target_entropy: f32,
) -> AzLoopReport {
    let selfplay_samples = pending.selfplay.samples.len();
    let total_seconds = pending.collection_seconds.max(1.0e-6);
    let train_stat_samples = stats
        .phase_value
        .iter()
        .map(|p| p.samples)
        .sum::<usize>()
        .max(1) as f32;
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
    let start_source_rate = pending
        .selfplay
        .start_games
        .map(|count| count as f32 / selfplay_games.max(1) as f32);
    let start_phase_ply = std::array::from_fn(|source| {
        pending.selfplay.start_phase_ply_sum[source] as f32
            / pending.selfplay.start_games[source].max(1) as f32
    });
    let start_age = std::array::from_fn(|source| {
        pending.selfplay.start_age_sum[source] as f32
            / pending.selfplay.start_games[source].max(1) as f32
    });
    let start_temperature = std::array::from_fn(|source| {
        pending.selfplay.start_temperature_sum[source]
            / pending.selfplay.start_games[source].max(1) as f32
    });
    AzLoopReport {
        training_steps: 0,
        training_chunks: 0,
        test_chunks: 0,
        holdout_checks: Vec::new(),
        cycle_complete: false,
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
        selfplay_start_source_rate: start_source_rate,
        selfplay_start_phase_ply: start_phase_ply,
        selfplay_start_age: start_age,
        selfplay_start_age_max: pending.selfplay.start_age_max,
        selfplay_start_temperature: start_temperature,
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
        policy_target_entropy: target_entropy,
        policy_kl: stats.policy_ce - target_entropy,
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
        terminal_no_legal_moves: pending.selfplay.terminal.no_legal_moves,
        terminal_checkmate: pending.selfplay.terminal.checkmate,
        terminal_stalemate: pending.selfplay.terminal.stalemate,
        terminal_rule_blocked: pending.selfplay.terminal.rule_blocked,
        terminal_search_no_move: pending.selfplay.terminal.search_no_move,
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
        terminal_max_plies: pending.selfplay.terminal.max_plies,
        terminal_search_proven: pending.selfplay.terminal.search_proven,
    }
}

struct ArenaThreadConfig {
    candidate: Arc<AzNnue>,
    baseline: Arc<AzNnue>,
    eval_starts: ArenaStarts,
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

#[derive(Clone)]
enum ArenaStarts {
    Positions(Arc<Vec<Position>>),
}

impl ArenaStarts {
    fn len(&self) -> usize {
        match self {
            Self::Positions(positions) => positions.len(),
        }
    }
}

fn run_arena_threads(config: ArenaThreadConfig) -> AzArenaReport {
    let games_per_side = if config.eval_starts.len() == 0 {
        1
    } else {
        config.eval_starts.len().max(1)
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
        let eval_starts = config.eval_starts.clone();
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
            let arena_config = AzArenaConfig {
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
                fpu_absolute_at_root: true,
                minimum_kldgain_per_node: 0.0,
                draw_score,
                policy_softmax_temp,
            };
            match eval_starts {
                ArenaStarts::Positions(positions) => play_arena_games_from_positions(
                    candidate.as_ref(),
                    baseline.as_ref(),
                    positions.as_slice(),
                    arena_config,
                ),
            }
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
    let mut book = Px0OpeningBook::load(&config.arena_opening_book, seed)
        .unwrap_or_else(|err| panic!("failed to load Px0 arena book: {err}"));
    let count = book.len();
    let positions = book
        .next_batch(1000, 0)
        .unwrap_or_else(|err| panic!("invalid Px0 arena FEN: {err}"))
        .into_iter()
        .map(|snapshot| snapshot.position)
        .collect();
    (
        positions,
        format!("px0(shuffled,count=1000,book_positions={count})"),
    )
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
        cpuct_base: 38739.0,
        cpuct_factor: 3.894,
        cpuct_base_at_root: 38739.0,
        cpuct_factor_at_root: 3.894,
        max_depth,
        root_dirichlet_alpha: 0.0,
        root_exploration_fraction: 0.0,
        fpu_value: 0.23,
        fpu_value_at_root: 1.0,
        fpu_absolute_at_root: true,
        minimum_kldgain_per_node: 0.0,
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
            println!("arch     : hidden={}", arch.hidden_size);
            println!("seed     : {seed}");
            println!("output   : {output}");
        }
        Some(CliCommand::AzPolicyScale(cmd)) => {
            assert!(
                cmd.scale.is_finite() && cmd.scale >= 0.0,
                "policy exact scale must be finite and non-negative"
            );
            let mut model = AzNnue::load(&cmd.input)
                .unwrap_or_else(|err| panic!("failed to load `{}`: {err}", cmd.input));
            let sparse_len = model.policy_sparse_table.len();
            let weights: &mut [f32] = match cmd.component {
                PolicyComponent::Exact => &mut model.policy_sparse_table[..POLICY_SPARSE_MAIN_SIZE],
                PolicyComponent::Capture => {
                    &mut model.policy_sparse_table[POLICY_SPARSE_MAIN_SIZE..sparse_len - 1]
                }
                PolicyComponent::Factor => &mut model.policy_sparse_factor,
                PolicyComponent::Tactical => &mut model.policy_tactical,
                PolicyComponent::TacticalExact => {
                    &mut model.policy_tactical[..POLICY_TACTICAL_EXACT_SIZE]
                }
                PolicyComponent::TacticalFactor => {
                    &mut model.policy_tactical[POLICY_TACTICAL_EXACT_SIZE..]
                }
                PolicyComponent::Accumulator => &mut model.policy_accumulator_move,
                PolicyComponent::Context => &mut model.policy_move_context,
                PolicyComponent::Consequence => &mut model.policy_consequence_output,
                PolicyComponent::MoveBias => &mut model.policy_move_bias,
                PolicyComponent::ThreatContext => &mut model.policy_threat_context,
            };
            let (rows, nonzero, before_l2) = {
                let nonzero = weights.iter().filter(|&&weight| weight != 0.0).count();
                let before_l2 = weights
                    .iter()
                    .map(|&weight| f64::from(weight) * f64::from(weight))
                    .sum::<f64>()
                    .sqrt();
                for weight in weights.iter_mut() {
                    *weight *= cmd.scale;
                }
                (weights.len(), nonzero, before_l2)
            };
            let after_l2 = before_l2 * f64::from(cmd.scale);
            model
                .save(&cmd.output)
                .unwrap_or_else(|err| panic!("failed to write `{}`: {err}", cmd.output));
            println!("input    : {}", cmd.input);
            println!("output   : {}", cmd.output);
            println!("component: {:?}", cmd.component);
            println!("scale    : {}", cmd.scale);
            println!("weights  : rows={} nonzero={}", rows, nonzero);
            println!("l2       : {:.6} -> {:.6}", before_l2, after_l2);
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
                fpu_absolute_at_root: true,
                minimum_kldgain_per_node: 0.0,
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
                let child_nn_q =
                    model.evaluate_value_with_rules(&child, &child_rule_history, &child_legal);
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
                fixed_az_search_limits(simulations, 0, cpuct, cpuct, 0, 1.4),
            );

            let started = std::time::Instant::now();
            let mut total_sims = 0usize;
            let mut best_move = None;
            for iteration in 0..repeat {
                let result = alphazero_search(
                    &position,
                    &model,
                    fixed_az_search_limits(simulations, iteration as u64, cpuct, cpuct, 0, 1.4),
                );
                total_sims += result.simulations;
                best_move = result.best_move;
            }
            let elapsed = started.elapsed();
            let elapsed_secs = elapsed.as_secs_f64().max(f64::EPSILON);
            println!("bench        : fixed-search");
            println!("model        : {model_path}");
            println!("arch         : hidden={}", model.arch.hidden_size);
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
        Some(CliCommand::AzLoop(cmd)) => {
            let config_path = cmd.config;
            let Some(config) = load_or_create_az_loop_config(&config_path) else {
                return;
            };
            let target_update = cmd.target_update.map(|update| update.max(1));
            let progress_boot = load_az_loop_progress(&config_path);
            let start_update = progress_boot.next_update.max(1);
            let mut arena_nemesis_update = progress_boot.nemesis_update;
            let mut generated_games_total = progress_boot.generated_games;
            let mut generated_samples_total = progress_boot.generated_samples;
            if let Some(target_update) = target_update
                && start_update > target_update
            {
                println!(
                    "target   : already complete, start_update={} target_update={}",
                    start_update, target_update
                );
                return;
            }
            let best_path = best_model_path(&config.model_path);

            let config_arch = config.arch();
            let model_path = Path::new(&config.model_path);
            let (mut model, resumed_model) = if model_path.exists() {
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
            let optimizer_state_path = PathBuf::from(format!("{config_path}.sgd.safetensors"));
            let model_optimizer_path = optimizer_checkpoint_path(model_path);
            let restore_path = if model_optimizer_path.exists() {
                &model_optimizer_path
            } else {
                &optimizer_state_path
            };
            let optimizer_state_path = restore_path.clone();
            if restore_path.exists() {
                model
                    .restore_training_state(restore_path, start_update, config.lr)
                    .unwrap_or_else(|err| panic!("refusing mismatched SGD resume state: {err}"));
                println!(
                    "optimizer: restored SGD momentum and global step from `{}`",
                    restore_path.display()
                );
            } else {
                println!("optimizer: fresh SGD momentum; global step=0, warmup=250");
            }
            let selfplay_model = model.clone();
            let initial_arena_reference_model = {
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
            let initial_selfplay_model = selfplay_model;
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
            println!(
                "train: config={} update={} sims={} batch={} optimizer=SGD+Nesterov lr={} max_plies={} book={} tensorboard={}",
                config_path,
                start_update,
                config.simulations,
                config.batch_size,
                config.lr,
                config.max_plies,
                config.selfplay_opening_book,
                tb_dir.display()
            );
            let selfplay_worker_count = config.workers.max(1);
            // 覆盖一次GPU更新期间完成的批次，同时限制旧模型样本和内存积压。
            let selfplay_queue_capacity = selfplay_worker_count.saturating_mul(2).max(32);
            let (selfplay_tx, selfplay_rx) =
                mpsc::sync_channel::<SelfplayBatch>(selfplay_queue_capacity);
            // 评估在主线程同步汇总时，训练结果仍可排队，避免反压训练和自对弈流水线。
            let (trainer_tx, trainer_rx) = mpsc::channel::<TrainerEvent>();
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
                champion_paths.push(initial_champion);
            }
            let shared_model = Arc::new(RwLock::new(SharedSelfplayModel {
                version: start_update.saturating_sub(1) as u64,
                learner_update: start_update.saturating_sub(1).min(u32::MAX as usize) as u32,
                model: Arc::new(initial_selfplay_model),
            }));
            let book_openings = Arc::new(std::sync::Mutex::new(
                chineseai::px0_opening_book::Px0OpeningBook::load(
                    &config.selfplay_opening_book,
                    config.seed,
                )
                .unwrap_or_else(|err| {
                    panic!(
                        "failed to load Px0 opening book `{}`: {err}",
                        config.selfplay_opening_book
                    )
                }),
            ));
            let mut selfplay_handles = Vec::with_capacity(selfplay_worker_count);
            for worker_id in 0..selfplay_worker_count {
                let selfplay_stop = stop_requested.clone();
                let selfplay_config = config.clone();
                let selfplay_tx = selfplay_tx.clone();
                let shared_model = Arc::clone(&shared_model);
                let book_openings = Arc::clone(&book_openings);
                selfplay_handles.push(thread::spawn(move || {
                    let mut batch_index = 0usize;
                    let mut local_version = u64::MAX;
                    let mut local_learner_update = 0u32;
                    let mut local_model: Option<Arc<AzNnue>> = None;
                    while !selfplay_stop.load(Ordering::SeqCst) {
                        if selfplay_stop.load(Ordering::SeqCst) {
                            break;
                        }
                        {
                            let shared = shared_model
                                .read()
                                .unwrap_or_else(|_| panic!("shared selfplay model poisoned"));
                            if shared.version != local_version {
                                local_model = Some(Arc::clone(&shared.model));
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
                        loop_config.opening_positions = book_openings
                            .lock()
                            .unwrap_or_else(|_| panic!("Px0 opening book poisoned"))
                            .next_batch(loop_config.games, local_learner_update)
                            .unwrap_or_else(|err| panic!("invalid Px0 opening: {err}"))
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
            let mut console = training_console::TrainingConsole::new(model.training_steps());
            let collector_handle = thread::spawn(move || {
                let mut pending = PendingTrainingData::default();
                let mut batch_index = 0usize;
                let mut window_started = Instant::now();
                while let Ok(batch) = selfplay_rx.recv() {
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
                    if ready_tx.send(std::mem::take(&mut pending)).is_err() {
                        break;
                    }
                    window_started = Instant::now();
                    batch_index += 1;
                }
            });
            let trainer_stop = stop_requested.clone();
            let trainer_config = config.clone();
            let trainer_start_update = start_update;
            let trainer_snapshot_path = replay_snapshot_path.clone();
            let trainer_shared_model = Arc::clone(&shared_model);
            let trainer_handle = thread::spawn(move || -> io::Result<()> {
                let mut trainer_model = model;
                let mut trainer_pool = replay_pool;
                let mut train_index = 0usize;
                let mut replay_sampler = Px0ReplaySampler::partitioned(
                    trainer_config.shuffle_size,
                    trainer_config.seed,
                    false,
                );
                let mut test_sampler = Px0ReplaySampler::partitioned(
                    (trainer_config.shuffle_size / 10).max(1),
                    trainer_config.seed,
                    true,
                );
                let mut cycle_end =
                    (trainer_model.training_steps() / chineseai::az::PX0_CYCLE_STEPS + 1)
                        * chineseai::az::PX0_CYCLE_STEPS;
                let min_train_samples = trainer_config.batch_size.max(1);
                'training: while let Ok(mut pending) = ready_rx.recv() {
                    let pending_games = pending.selfplay.games.len();
                    if let Some(pool) = trainer_pool.as_mut() {
                        pool.add_games(std::mem::take(&mut pending.selfplay.games));
                    }
                    if trainer_stop.load(Ordering::SeqCst)
                        || target_update.is_some_and(|target| {
                            trainer_start_update.saturating_add(train_index) > target
                        })
                    {
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
                    let steps_before = trainer_model.training_steps();
                    let train_steps = trainer_config
                        .train_samples_per_update
                        .div_ceil(trainer_config.batch_size)
                        .min(cycle_end - steps_before);
                    let (training_chunks, test_chunks) = pool.partition_chunks(trainer_config.seed);
                    if training_chunks == 0 || test_chunks == 0 {
                        continue;
                    }
                    let need_test = steps_before.is_multiple_of(chineseai::az::PX0_CYCLE_STEPS)
                        || steps_before / chineseai::az::PX0_TEST_STEPS
                            != (steps_before + train_steps) / chineseai::az::PX0_TEST_STEPS;
                    if need_test {
                        // 与公开入口相同：估计每个测试chunk约10个SKIP=32后的局面。
                        let count = (test_chunks * 10 / trainer_config.batch_size).max(1)
                            * trainer_config.batch_size;
                        let mut test_rng = SplitMix64::new(
                            trainer_config.seed ^ steps_before as u64 ^ 0xE703_7ED1_A0B4_28DB,
                        );
                        let test_data = test_sampler.sample(pool, count, 0, &mut test_rng).samples;
                        trainer_model.set_training_holdout(test_data, trainer_config.lr)?;
                    }
                    let sampled_batch = replay_sampler.sample(
                        pool,
                        train_steps * trainer_config.batch_size,
                        trainer_config.replay_recent_games,
                        &mut rng,
                    );
                    let train_data = sampled_batch.samples;
                    if train_data.is_empty() {
                        continue;
                    }
                    let train_data_len = train_data.len();
                    let target_entropy = policy_target_entropy(&train_data);
                    let train_update = trainer_start_update.saturating_add(train_index);
                    let current_lr = trainer_config.lr;
                    let train_started = Instant::now();
                    let stats = train_samples_weighted_owned(
                        &mut trainer_model,
                        train_data,
                        1,
                        current_lr,
                        trainer_config.batch_size,
                        &mut rng,
                        AzTrainLossWeights {
                            value: trainer_config.train_value_weight,
                            policy: trainer_config.train_policy_weight,
                        },
                    )
                    .unwrap_or_else(|err| panic!("training update {} failed: {err}", train_update));
                    let train_seconds = train_started.elapsed().as_secs_f32();
                    let current_lr = trainer_model
                        .last_training_learning_rate()
                        .unwrap_or(current_lr);
                    if trainer_config.checkpoint_interval > 0
                        && train_update.is_multiple_of(trainer_config.checkpoint_interval)
                    {
                        let path = save_checkpoint_model(
                            &trainer_model,
                            &trainer_config.model_path,
                            &trainer_config.checkpoint_dir,
                            train_update,
                        );
                        trainer_model
                            .save_training_state(
                                optimizer_checkpoint_path(&path),
                                train_update.saturating_add(1),
                            )
                            .unwrap_or_else(|err| {
                                panic!("failed to save checkpoint SGD state: {err}")
                            });
                    }
                    let mut report = build_async_training_report(
                        pending,
                        pending_games,
                        stats,
                        current_lr,
                        train_data_len,
                        train_seconds,
                        pool.sample_count(),
                        pool.capacity(),
                        pool.window_stats(trainer_config.replay_recent_games),
                        target_entropy,
                    );
                    report.training_steps = trainer_model.training_steps();
                    report.training_chunks = training_chunks;
                    report.test_chunks = test_chunks;
                    report.holdout_checks = trainer_model.take_training_checks();
                    report.cycle_complete = report.training_steps == cycle_end;
                    if report.cycle_complete {
                        save_model(&trainer_model, Path::new(&trainer_config.model_path));
                        trainer_model.save_training_state(
                            &optimizer_state_path,
                            train_update.saturating_add(1),
                        )?;
                        pool.save_snapshot_lz4(&trainer_snapshot_path)?;
                        cycle_end += chineseai::az::PX0_CYCLE_STEPS;
                    }
                    let candidate_model = trainer_model.clone();
                    publish_selfplay_model(
                        &trainer_shared_model,
                        Arc::new(candidate_model.clone()),
                        train_update,
                    );
                    if trainer_tx
                        .send(TrainerEvent {
                            report,
                            candidate_model,
                        })
                        .is_err()
                    {
                        break 'training;
                    }
                    train_index += 1;
                }
                if train_index > 0 {
                    trainer_model.save_training_state(
                        &optimizer_state_path,
                        trainer_start_update.saturating_add(train_index),
                    )?;
                }
                if let Some(pool) = trainer_pool.as_mut()
                    && trainer_stop.load(Ordering::SeqCst)
                {
                    pool.save_snapshot_lz4(&trainer_snapshot_path)?;
                }
                Ok(())
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
                generated_games_total = generated_games_total.saturating_add(report.games as u64);
                generated_samples_total =
                    generated_samples_total.saturating_add(report.samples as u64);
                let deployed_model = candidate_model.clone();
                interrupt_save_model = Some(candidate_model.clone());
                interrupt_save_next_update = update.saturating_add(1);
                let checkpoint_saved = if config.checkpoint_interval > 0
                    && update.is_multiple_of(config.checkpoint_interval)
                {
                    let path = checkpoint_path(&config.model_path, &config.checkpoint_dir, update);
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
                let truncated = report.terminal_max_plies + report.terminal_search_no_move;
                let completed = report.games.saturating_sub(truncated);
                let true_draws = report.draws.saturating_sub(truncated);
                console.update(
                    update,
                    &report,
                    generated_games_total,
                    true_draws,
                    checkpoint_saved.is_some(),
                );
                for check in &report.holdout_checks {
                    console.test(check);
                    for (tag, value) in [
                        ("test/loss", check.loss),
                        ("test/policy_kl", check.policy_kl),
                        ("test/samples", check.samples as f32),
                        ("test/value_samples", check.value_samples as f32),
                    ] {
                        log_scalar(&mut tb, tag, check.step, value);
                    }
                    if check.value_samples > 0 {
                        log_scalar(&mut tb, "test/wdl_ce", check.step, check.value_loss);
                        log_scalar(&mut tb, "test/value_rmse", check.step, check.value_rmse);
                    }
                }
                for (tag, value) in [
                    ("train/optimized_loss", report.loss),
                    ("train/wdl_ce", report.value_loss),
                    ("train/policy_kl", report.policy_kl),
                    ("train/value_rmse", value_rmse),
                    ("train/value_corr", report.value_corr),
                    ("train/value_calibration", report.value_calibration),
                    ("train/learning_rate", report.learning_rate),
                    ("train/samples", report.train_samples as f32),
                    (
                        "train/value_samples",
                        report
                            .phase_value
                            .iter()
                            .map(|phase| phase.samples)
                            .sum::<usize>() as f32,
                    ),
                    ("train/seconds", report.train_seconds),
                    ("replay/samples", report.pool_samples as f32),
                    ("selfplay/games_total", generated_games_total as f32),
                    ("selfplay/samples_total", generated_samples_total as f32),
                    (
                        "selfplay/avg_search_simulations",
                        report.avg_search_simulations,
                    ),
                    ("selfplay/avg_plies", report.avg_plies),
                    ("selfplay/completed_games", completed as f32),
                    ("selfplay/visit_policy_entropy", report.root_visit_entropy),
                    (
                        "truncation/rate",
                        truncated as f32 / report.games.max(1) as f32,
                    ),
                    ("truncation/max_plies", report.terminal_max_plies as f32),
                    (
                        "truncation/search_no_move",
                        report.terminal_search_no_move as f32,
                    ),
                    ("terminal/checkmate", report.terminal_checkmate as f32),
                    ("terminal/stalemate", report.terminal_stalemate as f32),
                    ("terminal/rule_blocked", report.terminal_rule_blocked as f32),
                    (
                        "terminal/red_general_missing",
                        report.terminal_red_general_missing as f32,
                    ),
                    (
                        "terminal/black_general_missing",
                        report.terminal_black_general_missing as f32,
                    ),
                    ("terminal/rule_draw", report.terminal_rule_draw as f32),
                    (
                        "terminal/draw_natural_limit",
                        report.terminal_rule_draw_natural_limit as f32,
                    ),
                    (
                        "terminal/draw_insufficient_material",
                        report.terminal_rule_draw_insufficient_material as f32,
                    ),
                    (
                        "terminal/draw_repetition",
                        report.terminal_rule_draw_repetition as f32,
                    ),
                    (
                        "terminal/draw_mutual_long_check",
                        report.terminal_rule_draw_mutual_long_check as f32,
                    ),
                    (
                        "terminal/draw_mutual_long_chase",
                        report.terminal_rule_draw_mutual_long_chase as f32,
                    ),
                    ("terminal/rule_win_red", report.terminal_rule_win_red as f32),
                    (
                        "terminal/rule_win_black",
                        report.terminal_rule_win_black as f32,
                    ),
                    (
                        "terminal/search_proven",
                        report.terminal_search_proven.iter().sum::<usize>() as f32,
                    ),
                ] {
                    log_scalar(&mut tb, tag, report.training_steps, value);
                }
                if completed > 0 {
                    log_scalar(
                        &mut tb,
                        "selfplay/draw_rate_completed",
                        update,
                        true_draws as f32 / completed as f32,
                    );
                }
                if config.arena_interval > 0 && update.is_multiple_of(config.arena_interval) {
                    {
                        let (mut arena_start_positions, _arena_mode) =
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
                                    eval_starts: ArenaStarts::Positions(Arc::new(positions)),
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
                                    fpu_value: 0.23,
                                    fpu_value_at_root: 1.0,
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
                            } else if promoted && nemesis_index == Some(index) {
                                arena_nemesis_update = None;
                            }
                        }
                        if promoted {
                            arena_reference_model = deployed_model.clone();
                            let best_checkpoint = save_best_checkpoint_model(
                                &deployed_model,
                                &config.model_path,
                                &config.checkpoint_dir,
                                update,
                            );
                            save_model(&deployed_model, &best_path);
                            champion_paths.push(best_checkpoint.clone());
                        }

                        console.arena(format!(
                            "arena {update:04}: games={} W/L/D={}/{}/{} score={:.3} ci={:.3}..{:.3} previous={} anchor={} decision={:?}",
                            current_arena.total_games(),
                            current_arena.wins,
                            current_arena.losses,
                            current_arena.draws,
                            current_arena.score_rate(),
                            current_arena
                                .score_rate_lower_bound(config.arena_promotion_confidence_z),
                            current_arena
                                .score_rate_upper_bound(config.arena_promotion_confidence_z),
                            previous_arena
                                .as_ref()
                                .map_or_else(|| "-".into(), |r| format!("{:.3}", r.score_rate())),
                            anchor_arena
                                .as_ref()
                                .map_or_else(|| "-".into(), |r| format!("{:.3}", r.score_rate())),
                            gate_decision
                        ));
                        let mut historical_arena = AzArenaReport::default();
                        if let Some(report) = previous_arena.as_ref() {
                            historical_arena.add_assign(report);
                        }
                        if let Some(report) = anchor_arena.as_ref() {
                            historical_arena.add_assign(report);
                        }
                        if historical_arena.total_games() > 0 {
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
                }
                if config.pikafish_label_eval_interval > 0
                    && update.is_multiple_of(config.pikafish_label_eval_interval)
                    && !config.pikafish_label_eval_sqlite.trim().is_empty()
                {
                    let sqlite_path = Path::new(&config.pikafish_label_eval_sqlite);
                    if sqlite_path.exists() {
                        let started = Instant::now();
                        let eval_result = (|| -> io::Result<LabelEvalStats> {
                            let conn = Connection::open(sqlite_path).map_err(sqlite_io_error)?;
                            let rows = load_pikafish_label_rows(
                                &conn,
                                config.pikafish_label_eval_limit,
                                config.seed,
                            )
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
                                    fpu_value: 0.23,
                                    fpu_value_at_root: 1.0,
                                    fpu_absolute_at_root: true,
                                    minimum_kldgain_per_node: 0.0,
                                    policy_softmax_temp: config
                                        .pikafish_label_eval_policy_softmax_temp,
                                    draw_score: config.draw_score,
                                    value_scale: 1.0,
                                },
                                config.arena_processes,
                            )
                        })();
                        match eval_result {
                            Ok(stats) => {
                                console.event(format!(
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
                                ));
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
                                console.event(format!(
                                    "pikafish-label {update:04}: failed sqlite={}: {err}",
                                    config.pikafish_label_eval_sqlite
                                ));
                            }
                        }
                    } else {
                        let resolved = if sqlite_path.is_absolute() {
                            sqlite_path.to_path_buf()
                        } else {
                            std::env::current_dir()
                                .unwrap_or_else(|_| PathBuf::from("."))
                                .join(sqlite_path)
                        };
                        console.event(format!(
                            "pikafish-label {update:04}: skipped missing sqlite={} resolved={} (copy the label DB or update pikafish_label_eval_sqlite)",
                            config.pikafish_label_eval_sqlite,
                            resolved.display()
                        ));
                    }
                }
                tb.flush();
                update = update.saturating_add(1);
                if report.cycle_complete {
                    save_az_loop_progress_pair(
                        &config_path,
                        interrupt_save_next_update,
                        arena_nemesis_update,
                        generated_games_total,
                        generated_samples_total,
                    );
                    console.event(format!(
                        "saved: cycle {} complete; optimizer+replay saved; continuing cycle {} next_update={}",
                        report.training_steps / chineseai::az::PX0_CYCLE_STEPS,
                        report.training_steps / chineseai::az::PX0_CYCLE_STEPS + 1,
                        interrupt_save_next_update,
                    ));
                }
                if let Some(target_update) = target_update
                    && update > target_update
                {
                    exited_after_target_update = true;
                    break;
                }
            }
            console.finish();
            stop_requested.store(true, Ordering::SeqCst);
            // 等待线程前持续排空结果队列，避免满队列让训练及产数线程相互等待。
            for event in trainer_rx {
                if exited_after_ctrl_c {
                    generated_games_total =
                        generated_games_total.saturating_add(event.report.games as u64);
                    generated_samples_total =
                        generated_samples_total.saturating_add(event.report.samples as u64);
                    interrupt_save_model = Some(event.candidate_model);
                    interrupt_save_next_update = update.saturating_add(1);
                    update = update.saturating_add(1);
                }
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
                .unwrap_or_else(|_| panic!("training thread panicked"))
                .unwrap_or_else(|err| panic!("failed to save training state: {err}"));
            if exited_after_ctrl_c || exited_after_target_update {
                if let Some(model) = interrupt_save_model.as_ref() {
                    save_model(model, Path::new(&config.model_path));
                    save_az_loop_progress_pair(
                        &config_path,
                        interrupt_save_next_update,
                        arena_nemesis_update,
                        generated_games_total,
                        generated_samples_total,
                    );
                    println!(
                        "saved: {} model=`{}` optimizer+replay saved next_update={}",
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
            let simulations = cmd.simulations.unwrap_or(800).max(1);
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
            let (start_positions, opening_mode) = if cmd.opening_book.trim().is_empty() {
                (Vec::new(), "startpos_fallback".to_string())
            } else {
                let mut book = Px0OpeningBook::load(&cmd.opening_book, cmd.seed)
                    .unwrap_or_else(|err| panic!("failed to load Px0 opening book: {err}"));
                let positions = book
                    .next_batch(cmd.opening_positions.max(1), 0)
                    .unwrap_or_else(|err| panic!("invalid Px0 FEN: {err}"))
                    .into_iter()
                    .map(|s| s.position)
                    .collect();
                (
                    positions,
                    format!("px0(shuffled,book={})", cmd.opening_book),
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
        let rule_history = position.initial_rule_history();
        if position.rule_outcome_with_history(&rule_history).is_some() {
            continue;
        }
        let Some(label_move) = position.parse_uci_move(&row.bestmove) else {
            continue;
        };
        let legal_moves = position.legal_moves_with_rules(&rule_history);
        if !legal_moves.contains(&label_move) {
            continue;
        }
        stats.legal_bestmove += 1;
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
    seed: u64,
) -> rusqlite::Result<Vec<PikafishLabelRow>> {
    let mut stmt = conn.prepare(
        "SELECT id, fen, bestmove, wdl_win, wdl_draw, wdl_loss FROM pikafish_labels ORDER BY id",
    )?;
    let mut rows: Vec<_> = stmt
        .query_map([], |row| {
            Ok(PikafishLabelRow {
                id: row.get(0)?,
                fen: row.get(1)?,
                bestmove: row.get(2)?,
                best_wdl: [row.get(3)?, row.get(4)?, row.get(5)?],
            })
        })?
        .collect::<rusqlite::Result<_>>()?;
    if limit > 0 && limit < rows.len() {
        let mut rng = SplitMix64::new(seed ^ 0xA076_1D64_78BD_642F);
        for index in (1..rows.len()).rev() {
            rows.swap(index, rng.next_u64() as usize % (index + 1));
        }
        rows.truncate(limit);
    }
    Ok(rows)
}

fn sqlite_io_error(err: rusqlite::Error) -> io::Error {
    io::Error::other(err.to_string())
}

#[cfg(test)]
mod reporting_tests {
    use super::*;
    use chineseai::az::{AzSampleMeta, AzTrainingSample};
    use rusqlite::params;

    #[test]
    fn learner_publish_advances_actor_without_arena_decision() {
        let initial = Arc::new(AzNnue::random(8, 1));
        let latest = Arc::new(AzNnue::random(8, 2));
        let shared = RwLock::new(SharedSelfplayModel {
            version: 7,
            learner_update: 6,
            model: initial,
        });

        let version = publish_selfplay_model(&shared, Arc::clone(&latest), 8);
        let published = shared.read().unwrap();

        assert_eq!(version, 8);
        assert_eq!(published.version, 8);
        assert_eq!(published.learner_update, 8);
        assert!(Arc::ptr_eq(&published.model, &latest));
    }

    #[test]
    fn progress_roundtrip_preserves_generated_totals() {
        let state = AzLoopProgressState {
            next_update: 17,
            generated_games: 12_345,
            generated_samples: 678_901,
            ..Default::default()
        };

        let text = toml::to_string(&state).unwrap();
        let loaded = toml::from_str::<AzLoopProgressState>(&text)
            .unwrap()
            .normalize();

        assert_eq!(loaded.next_update, 17);
        assert_eq!(loaded.generated_games, 12_345);
        assert_eq!(loaded.generated_samples, 678_901);
    }

    #[test]
    fn az_search_defaults_match_px0_match_settings() {
        let cli =
            Cli::try_parse_from(["chineseai", "az-search", "model.safetensors", "3200"]).unwrap();
        let Some(CliCommand::AzSearch(args)) = cli.command else {
            panic!("expected az-search command");
        };
        assert_eq!(args.cpuct, 1.0);
        assert_eq!(args.cpuct_at_root, 1.9);
        assert_eq!(args.cpuct_factor, 3.894);
        assert_eq!(args.cpuct_factor_at_root, 3.894);
        assert_eq!(args.fpu_value, 0.23);
        assert_eq!(args.fpu_value_at_root, 1.0);
        assert_eq!(args.policy_softmax_temp, 1.4);
    }

    fn reporting_sample(generation: u32, policy: Vec<f32>) -> AzTrainingSample {
        AzTrainingSample {
            repetition_flags: Vec::new(),
            features: vec![0],
            rule_context: [0.0; chineseai::az::RULE_CONTEXT_SIZE],
            move_indices: (0..policy.len()).collect(),
            policy,
            value_wdl: [0.0, 1.0, 0.0],
            root_search_wdl: [0.0, 1.0, 0.0],
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
    fn policy_entropy_normalizes_targets() {
        let samples = vec![
            reporting_sample(10, vec![3.0, 1.0]),
            reporting_sample(5, vec![2.0, 2.0]),
        ];
        let expected = (-(0.75f32 * 0.75f32.ln() + 0.25f32 * 0.25f32.ln()) - 0.5f32.ln()) / 2.0;
        assert!((policy_target_entropy(&samples) - expected).abs() < 1e-6);
    }

    #[test]
    fn pikafish_label_eval_excludes_rule_terminal_positions() {
        let model = AzNnue::random(8, 7);
        let terminal = Position::from_fen("9/4a4/3k5/9/9/9/9/4B4/9/2B1KA3 b").unwrap();
        let rows = vec![
            PikafishLabelRow {
                id: 1,
                fen: terminal.to_fen(),
                bestmove: terminal.legal_moves()[0].to_uci(),
                best_wdl: [0, 1000, 0],
            },
            PikafishLabelRow {
                id: 2,
                fen: Position::startpos().to_fen(),
                bestmove: "b0c2".into(),
                best_wdl: [500, 0, 500],
            },
        ];
        let stats = evaluate_pikafish_labels(
            &model,
            &rows,
            AzSearchLimits {
                simulations: 4,
                ..AzSearchLimits::default()
            },
            |_, _| {},
        )
        .unwrap();
        assert_eq!(stats.count, 1);
        assert_eq!(stats.legal_bestmove, 1);
        assert_eq!(stats.value_count(), 1);
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
    fn pikafish_label_limit_is_seeded_uniform_sample() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE pikafish_labels (
                id INTEGER PRIMARY KEY,
                fen TEXT NOT NULL,
                bestmove TEXT NOT NULL,
                wdl_win INTEGER NOT NULL,
                wdl_draw INTEGER NOT NULL,
                wdl_loss INTEGER NOT NULL
            );",
        )
        .unwrap();
        for id in 1..=20 {
            conn.execute(
                "INSERT INTO pikafish_labels VALUES (?1, '', '', 0, 1000, 0)",
                params![id],
            )
            .unwrap();
        }

        let first = load_pikafish_label_rows(&conn, 5, 42).unwrap();
        let repeated = load_pikafish_label_rows(&conn, 5, 42).unwrap();
        let different = load_pikafish_label_rows(&conn, 5, 43).unwrap();
        let ids = |rows: &[PikafishLabelRow]| rows.iter().map(|row| row.id).collect::<Vec<_>>();

        assert_eq!(ids(&first), ids(&repeated));
        assert_ne!(ids(&first), vec![1, 2, 3, 4, 5]);
        assert_ne!(ids(&first), ids(&different));
        assert_eq!(
            ids(&load_pikafish_label_rows(&conn, 0, 42).unwrap()),
            (1..=20).collect::<Vec<_>>()
        );
    }

    #[test]
    fn arena_uses_only_book_positions() {
        use std::io::Write;
        let dir = std::env::current_dir().unwrap().join("tmp");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!(
            "chineseai-arena-book-{}.pgn.gz",
            std::process::id()
        ));
        let start = Position::startpos();
        let mut other = start.clone();
        other.make_move(start.parse_uci_move("a0a1").unwrap());
        let mut writer = flate2::write::GzEncoder::new(
            std::fs::File::create(&path).unwrap(),
            flate2::Compression::default(),
        );
        for position in [&start, &other] {
            writeln!(writer, "[FEN \"{}\"]\n{{}}", position.to_fen()).unwrap();
        }
        writer.finish().unwrap();
        let mut config = AzLoopFileConfig::default();
        config.arena_opening_book = path.to_string_lossy().into_owned();
        let (positions, mode) = build_arena_start_positions(&config, 20);
        assert_eq!(positions.len() * 2, 2000);
        assert_eq!(mode, "px0(shuffled,count=1000,book_positions=2)");
        assert!(
            positions
                .iter()
                .all(|p| [start.hash(), other.hash()].contains(&p.hash()))
        );
        std::fs::remove_file(path).unwrap();
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

fn parse_position(text: &str) -> Position {
    if text.trim().is_empty() || text == "startpos" {
        Position::startpos()
    } else {
        Position::from_fen(text).unwrap_or_else(|err| {
            panic!("invalid FEN `{text}`: {err}");
        })
    }
}
