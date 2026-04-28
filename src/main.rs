#[cfg(all(target_os = "linux", not(target_env = "musl")))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

mod az_loop_config;

use az_loop_config::{AzLoopFileConfig, DEFAULT_AZ_LOOP_CONFIG, load_or_create_az_loop_config};

use chineseai::{
    az::{
        AzArenaConfig, AzArenaReport, AzExperiencePool, AzLoopConfig, AzLoopReport, AzModel,
        AzModelConfig, AzSearchLimits, AzSelfplayData, alphazero_search, generate_selfplay_data,
        global_training_step_sample_count, play_arena_games_from_positions, train_samples,
    },
    board_transform::HISTORY_PLIES,
    pikafish_match::{VsPikafishConfig, run_vs_pikafish},
    uci::run_uci,
    xiangqi::{BOARD_FILES, BOARD_RANKS, BOARD_SIZE, Position},
};
use clap::{Args, Parser, Subcommand};
use std::{
    fs, io,
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

const DEFAULT_ARENA_EVAL_FENS: &str = "eval_fens.txt";
const DEFAULT_VS_PIKAFISH_DEPTH: u32 = 10;
const DEFAULT_VS_PIKAFISH_GAMES: usize = 20;
const DEFAULT_VS_PIKAFISH_PARALLEL_GAMES: usize = 5;

#[derive(Parser, Debug)]
#[command(
    name = "chineseai",
    version,
    about = "ChineseAI AZ search and training tools",
    long_about = "ChineseAI AZ search and training tools.\n\nIf no command is given, the program starts in UCI mode."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<CliCommand>,
}

#[derive(Subcommand, Debug)]
enum CliCommand {
    /// Run the UCI engine loop.
    Uci,
    /// Create a random AZ model.
    AzInit(AzInitArgs),
    /// Probe AZ model structures with cheap CPU root-eval tests.
    AzArchProbe(AzArchProbeArgs),
    /// Run self-play training from a TOML config.
    AzLoop(AzLoopArgs),
    /// Internal arena worker process.
    AzArenaWorker(AzArenaWorkerArgs),
    /// Run ChineseAI against a Pikafish UCI engine.
    VsPikafish(VsPikafishArgs),
}

#[derive(Args, Debug)]
struct AzInitArgs {
    /// Hidden size of the model.
    #[arg(default_value_t = 128)]
    hidden: usize,
    /// Output model path.
    #[arg(default_value = "chineseai.azm")]
    output: String,
    /// Random seed.
    #[arg(default_value_t = 20260409)]
    seed: u64,
    /// Model square embedding channels.
    #[arg(long)]
    model_channels: Option<usize>,
    /// Residual mobile board blocks.
    #[arg(long)]
    model_blocks: Option<usize>,
    /// Value head channels.
    #[arg(long)]
    value_head_channels: Option<usize>,
    /// Value hidden width. This build currently supports 256.
    #[arg(long)]
    value_hidden_size: Option<usize>,
}

#[derive(Args, Debug)]
struct AzLoopArgs {
    /// Training config path.
    #[arg(default_value = DEFAULT_AZ_LOOP_CONFIG)]
    config: String,
    /// Stop after this many training updates. Omit for continuous training.
    #[arg(long)]
    updates: Option<usize>,
}

#[derive(Args, Debug)]
struct AzArchProbeArgs {
    /// Root evaluations per candidate and position.
    #[arg(long, default_value_t = 128)]
    reps: usize,
    /// Random initializations tested for each candidate.
    #[arg(long, default_value_t = 3)]
    seed_trials: usize,
    /// Random seed used to initialize candidate models.
    #[arg(long, default_value_t = 20260428)]
    seed: u64,
    /// FEN file used as the probe position set.
    #[arg(long = "eval-fens", default_value = DEFAULT_ARENA_EVAL_FENS)]
    eval_fens_path: String,
    /// Maximum positions read from the FEN file.
    #[arg(long, default_value_t = 16)]
    max_positions: usize,
}

#[derive(Args, Debug)]
struct AzArenaWorkerArgs {
    /// Candidate model path.
    candidate: String,
    /// Baseline model path.
    baseline: String,
    /// Games with candidate as Red.
    red_games: usize,
    /// Games with candidate as Black.
    black_games: usize,
    /// Simulations per move.
    simulations: usize,
    /// Draw after this many plies.
    max_plies: usize,
    /// PUCT constant for arena search.
    arena_cpuct: f32,
    /// Start-position file.
    eval_fens_path: String,
    /// Random seed.
    seed: u64,
}

#[derive(Args, Debug)]
#[command(after_long_help = "\
Examples:
  chineseai vs-pikafish ./tools/pikafish 10000
  chineseai vs-pikafish ./tools/pikafish chineseai.azloop.toml -s 10000
  chineseai vs-pikafish ./tools/pikafish chineseai.azloop.toml -s 10000 --pikafish-depth 10 --games 40 --parallel-games 5 --eval-fens eval_fens.txt")]
struct VsPikafishArgs {
    /// Pikafish UCI executable path.
    pikafish_exe: String,
    /// Config path or simulations override. Numeric values use the default config.
    config_or_simulations: Option<String>,
    /// Override config.simulations.
    #[arg(short = 's', long)]
    simulations: Option<usize>,
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

fn best_model_path(model_path: &str) -> PathBuf {
    PathBuf::from(format!("{model_path}.best"))
}

fn az_loop_progress_path(config_path: &str) -> PathBuf {
    PathBuf::from(format!("{config_path}.progress"))
}

fn az_loop_replay_snapshot_path(config_path: &str) -> PathBuf {
    PathBuf::from(format!("{config_path}.replay.lz4"))
}

fn read_az_loop_next_update(config_path: &str) -> Option<usize> {
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
        if key.trim() == "next_update" {
            return value.trim().parse().ok();
        }
    }
    None
}

fn write_az_loop_next_update(config_path: &str, next: usize) {
    let path = az_loop_progress_path(config_path);
    fs::write(&path, format!("next_update={next}\n")).unwrap_or_else(|err| {
        panic!("failed to write resume cursor `{}`: {err}", path.display());
    });
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

    format!(
        concat!(
            "sim{}_bs{}_lr{}_ep{}_mx{}_h{}_c{}_rb{}_vhc{}_vh{}_mxp{}_wk{}_",
            "sa{}_gsm{}_tb{}_te{}_tde{}_rg{}_rs{}_mp{}_tl{}_cpi{}_",
            "ai{}_acp{}_rda{}_ref{}_gma{}_gs{}_gvs{}_gmv{}_sd{}"
        ),
        config.simulations,
        config.batch_size,
        f32_slug(config.lr),
        config.epochs,
        config.max_sample_train_count,
        config.hidden_size,
        config.model_channels,
        config.model_blocks,
        config.value_head_channels,
        config.value_hidden_size,
        config.max_plies,
        config.workers,
        config.search_algorithm.as_str(),
        f32_slug(config.cpuct),
        f32_slug(config.temperature_start),
        f32_slug(config.temperature_end),
        config.temperature_decay_plies,
        config.replay_games,
        config.replay_samples,
        f32_slug(config.mirror_probability),
        f32_slug(config.td_lambda),
        config.checkpoint_interval,
        config.arena_interval,
        f32_slug(config.arena_cpuct),
        f32_slug(config.root_dirichlet_alpha),
        f32_slug(config.root_exploration_fraction),
        config.gumbel.max_num_considered_actions,
        f32_slug(config.gumbel.gumbel_scale),
        f32_slug(config.gumbel.value_scale),
        f32_slug(config.gumbel.maxvisit_init),
        config.seed,
    )
}

fn tensorboard_effective_logdir(config: &AzLoopFileConfig) -> PathBuf {
    Path::new(&config.tensorboard_logdir).join(tensorboard_encoded_subdir(config))
}

fn checkpoint_path(model_path: &str, checkpoint_dir: &str, update: usize) -> PathBuf {
    let base = Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.azm");
    Path::new(checkpoint_dir).join(format!("update-{update:04}-{base}"))
}

fn save_checkpoint_copy(model_path: &str, checkpoint_dir: &str, update: usize) -> PathBuf {
    fs::create_dir_all(checkpoint_dir).unwrap_or_else(|err| {
        panic!("failed to create checkpoint dir `{checkpoint_dir}`: {err}");
    });
    let path = checkpoint_path(model_path, checkpoint_dir, update);
    fs::copy(model_path, &path).unwrap_or_else(|err| {
        panic!(
            "failed to copy `{model_path}` to `{}`: {err}",
            path.display()
        );
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
        .unwrap_or("model.azm")
        .to_string();
    let prefix = "update-";
    let suffix = format!("-{base}");
    let mut entries = fs::read_dir(checkpoint_dir)?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let path = entry.path();
            let name = path.file_name()?.to_str()?;
            (name.starts_with(prefix) && name.ends_with(&suffix))
                .then_some((name.to_string(), path))
        })
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    let to_remove = entries.len().saturating_sub(max_checkpoints);
    for (_, path) in entries.into_iter().take(to_remove) {
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
}

struct SharedSelfplayModel {
    version: u64,
    model: AzModel,
}

#[derive(Default)]
struct SelfplayPauseState {
    paused: bool,
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

fn build_az_loop_config(config: &AzLoopFileConfig, seed: u64, workers: usize) -> AzLoopConfig {
    AzLoopConfig {
        games: 1,
        max_plies: config.max_plies,
        simulations: config.simulations,
        seed,
        workers,
        temperature_start: config.temperature_start,
        temperature_end: config.temperature_end,
        temperature_decay_plies: config.temperature_decay_plies,
        search_algorithm: config.search_algorithm,
        cpuct: config.cpuct,
        root_dirichlet_alpha: config.root_dirichlet_alpha,
        root_exploration_fraction: config.root_exploration_fraction,
        gumbel: config.gumbel,
        td_lambda: config.td_lambda,
        mirror_probability: config.mirror_probability,
    }
}

fn build_async_training_report(
    pending: PendingTrainingData,
    stats: chineseai::az::AzTrainStats,
    train_data_len: usize,
    epochs: usize,
    train_seconds: f32,
    pool_games: usize,
    pool_samples: usize,
) -> AzLoopReport {
    let selfplay_games = pending.selfplay.games.len();
    let selfplay_samples = pending.selfplay.samples.len();
    let total_seconds = pending
        .started
        .map(|started| started.elapsed().as_secs_f32())
        .unwrap_or(train_seconds);
    let train_stat_samples = stats.samples.max(1) as f32;
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
        value_loss: stats.value_loss,
        value_mse: stats.value_error_sq_sum / train_stat_samples,
        value_pred_mean: stats.value_pred_sum / train_stat_samples,
        value_target_mean: stats.value_target_sum / train_stat_samples,
        policy_ce: stats.policy_ce,
        temperature_early_entropy: pending.selfplay.temperature_early_entropy_sum
            / pending.selfplay.temperature_early_entropy_count.max(1) as f32,
        temperature_mid_entropy: pending.selfplay.temperature_mid_entropy_sum
            / pending.selfplay.temperature_mid_entropy_count.max(1) as f32,
        selfplay_seconds: pending.selfplay_seconds,
        train_seconds,
        total_seconds,
        games_per_second: selfplay_games as f32 / total_seconds.max(1e-6),
        samples_per_second: selfplay_samples as f32 / total_seconds.max(1e-6),
        train_samples_per_second: (train_data_len * epochs.max(1)) as f32 / train_seconds.max(1e-6),
        train_samples: train_data_len,
        pool_games,
        pool_samples,
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
        terminal_max_plies: pending.selfplay.terminal.max_plies,
    }
}

struct ArenaProcessConfig<'a> {
    candidate_path: &'a str,
    baseline_path: &'a str,
    games_per_side: usize,
    simulations: usize,
    max_plies: usize,
    cpuct: f32,
    eval_fens_path: &'a str,
    process_count: usize,
    seed: u64,
}

fn run_arena_processes(config: ArenaProcessConfig<'_>) -> AzArenaReport {
    let process_count = config
        .process_count
        .max(1)
        .min(config.games_per_side.max(1));
    let exe = std::env::current_exe().unwrap_or_else(|err| {
        panic!("failed to locate current executable: {err}");
    });
    let mut merged = AzArenaReport::default();
    let mut children: Vec<(usize, Child)> = Vec::new();
    for index in 0..process_count {
        let red_games = config.games_per_side / process_count
            + usize::from(index < config.games_per_side % process_count);
        let black_games = config.games_per_side / process_count
            + usize::from(index < config.games_per_side % process_count);
        if red_games == 0 && black_games == 0 {
            continue;
        }
        let child = Command::new(&exe)
            .arg("az-arena-worker")
            .arg(config.candidate_path)
            .arg(config.baseline_path)
            .arg(red_games.to_string())
            .arg(black_games.to_string())
            .arg(config.simulations.to_string())
            .arg(config.max_plies.to_string())
            .arg(config.cpuct.to_string())
            .arg(config.eval_fens_path)
            .arg((config.seed ^ index as u64).to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap_or_else(|err| {
                panic!("failed to spawn arena worker process: {err}");
            });
        children.push((index, child));
    }

    for (index, child) in children {
        let output = child.wait_with_output().unwrap_or_else(|err| {
            panic!("failed to wait for arena worker {index}: {err}");
        });
        if !output.status.success() {
            panic!(
                "arena worker {index} failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
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

fn log_scalar(writer: &mut SummaryWriter, tag: &str, step: usize, value: f32) {
    writer.add_scalar(tag, value, step);
}

fn az_model_parameter_count(model: &AzModel) -> usize {
    model.board_conv1_weights.len()
        + model.board_conv1_bias.len()
        + model.board_conv2_weights.len()
        + model.board_conv2_bias.len()
        + model.position_embed.len()
        + model.board_hidden.len()
        + model.board_hidden_bias.len()
        + model.value_relation_weights.len()
        + model.value_relation_bias.len()
        + model.value_tail_conv_weights.len()
        + model.value_tail_conv_bias.len()
        + model.value_intermediate_hidden.len()
        + model.value_intermediate_bias.len()
        + model.value_logits_weights.len()
        + model.value_direct_logits_weights.len()
        + model.value_logits_bias.len()
        + model.policy_from_weights.len()
        + model.policy_from_bias.len()
        + model.policy_to_weights.len()
        + model.policy_to_bias.len()
        + model.policy_pair_weights.len()
        + model.policy_move_bias.len()
        + model.policy_feature_hidden.len()
        + model.policy_feature_cnn.len()
        + model.policy_feature_bias.len()
}

fn az_arch_probe_candidates() -> Vec<AzModelConfig> {
    [
        (128, 16, 2, 4, 128),
        (160, 20, 3, 5, 160),
        (192, 24, 4, 6, 192),
        (192, 24, 6, 6, 192),
        (192, 24, 2, 6, 192),
        (192, 24, 3, 6, 192),
        (256, 32, 2, 8, 224),
        (256, 32, 3, 8, 256),
        (256, 32, 6, 8, 256),
        (256, 32, 12, 8, 256),
        (288, 40, 2, 8, 256),
    ]
    .into_iter()
    .map(
        |(hidden_size, model_channels, model_blocks, value_head_channels, value_hidden_size)| {
            AzModelConfig {
                hidden_size,
                model_channels,
                model_blocks,
                value_head_channels,
                value_hidden_size,
                policy_condition_size: 32,
            }
            .normalized()
        },
    )
    .collect()
}

fn az_arch_local_window(config: AzModelConfig) -> usize {
    let radius = config.model_blocks + 1;
    let edge = radius * 2 + 1;
    edge.min(BOARD_FILES.max(BOARD_RANKS))
}

fn az_arch_value_feature_count(config: AzModelConfig) -> usize {
    let channels = config.model_channels;
    let value_channels = config.value_head_channels;
    value_channels * BOARD_SIZE + value_channels * 4 + channels * 8
}

fn az_arch_dense64x6_mac_ratio(config: AzModelConfig) -> f64 {
    let board_channels = 14usize * (HISTORY_PLIES + 1);
    let board = BOARD_SIZE as f64;
    let channels = config.model_channels as f64;
    let dense_channels = 64.0f64;
    let stem = board * board_channels as f64 * 9.0 * channels;
    let mobile_blocks = config.model_blocks as f64
        * board
        * (channels * (9.0 + BOARD_FILES as f64 + BOARD_RANKS as f64) + channels * channels);
    let value_relation_layers = 4.0;
    let value_relation_ffn_mult = 2.0;
    let value_relation = value_relation_layers
        * board
        * (channels * (BOARD_FILES as f64 + BOARD_RANKS as f64)
            + 2.0 * value_relation_ffn_mult * channels * channels);
    let head = value_relation
        + (channels * 4.0) * config.hidden_size as f64
        + az_arch_value_feature_count(config) as f64 * config.value_hidden_size as f64
        + channels * 90.0 * 3.0;
    let mobile_total = stem + mobile_blocks + head;

    let dense_stem = board * board_channels as f64 * 9.0 * dense_channels;
    let dense_blocks = 6.0 * board * dense_channels * dense_channels * 9.0;
    mobile_total / (dense_stem + dense_blocks)
}

#[derive(Clone, Copy, Debug, Default)]
struct AzArchProbeStats {
    us_per_root: f64,
    value_abs: f32,
    entropy_ratio: f32,
    avg_moves: f32,
}

fn mean_and_std(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;
    (mean, variance.sqrt())
}

fn run_single_az_arch_probe(
    model: &AzModel,
    positions: &[Position],
    reps: usize,
    seed: u64,
) -> AzArchProbeStats {
    let limits = AzSearchLimits {
        simulations: 0,
        seed,
        ..AzSearchLimits::default()
    };

    for position in positions {
        let _ = alphazero_search(position, model, limits);
    }

    let started = Instant::now();
    let mut value_abs_sum = 0.0f32;
    let mut entropy_ratio_sum = 0.0f32;
    let mut move_count_sum = 0usize;
    let mut root_count = 0usize;
    for _ in 0..reps {
        for position in positions {
            let result = alphazero_search(position, model, limits);
            value_abs_sum += (result.value_cp as f32 / 1000.0).abs();
            let moves = result.candidates.len();
            move_count_sum += moves;
            if moves > 1 {
                let entropy = result
                    .candidates
                    .iter()
                    .filter(|candidate| candidate.policy > 0.0)
                    .map(|candidate| -candidate.policy * candidate.policy.ln())
                    .sum::<f32>();
                entropy_ratio_sum += entropy / (moves as f32).ln();
            }
            root_count += 1;
        }
    }
    let elapsed = started.elapsed();
    let root_count_f = root_count.max(1) as f64;
    AzArchProbeStats {
        us_per_root: elapsed.as_secs_f64() * 1_000_000.0 / root_count_f,
        value_abs: value_abs_sum / root_count.max(1) as f32,
        entropy_ratio: entropy_ratio_sum / root_count.max(1) as f32,
        avg_moves: move_count_sum as f32 / root_count.max(1) as f32,
    }
}

fn run_az_arch_probe(cmd: AzArchProbeArgs) {
    let mut positions = load_arena_eval_positions(&cmd.eval_fens_path);
    if positions.is_empty() {
        positions.push(Position::startpos());
    }
    positions.truncate(cmd.max_positions.max(1));
    let reps = cmd.reps.max(1);
    let seed_trials = cmd.seed_trials.max(1);

    println!(
        "arch-probe: positions={} reps={} seed_trials={} seed={} eval_fens={}",
        positions.len(),
        reps,
        seed_trials,
        cmd.seed,
        cmd.eval_fens_path
    );
    println!(
        "{:>3} {:>4} {:>2} {:>4} {:>3} {:>4} {:>9} {:>7} {:>5} {:>6} {:>10} {:>7} {:>8} {:>8} {:>8}",
        "#",
        "chan",
        "bk",
        "hid",
        "vhc",
        "vhh",
        "params",
        "win",
        "line",
        "mac%",
        "us/root",
        "us_sd",
        "v_abs",
        "ent",
        "moves"
    );

    for (index, config) in az_arch_probe_candidates().into_iter().enumerate() {
        config
            .validate_supported()
            .unwrap_or_else(|err| panic!("{err}"));
        let mut stats = Vec::with_capacity(seed_trials);
        let mut params = 0usize;
        for trial in 0..seed_trials {
            let seed = cmd.seed ^ ((index as u64) << 32) ^ trial as u64;
            let model = AzModel::random_with_config(config, seed);
            params = az_model_parameter_count(&model);
            stats.push(run_single_az_arch_probe(&model, &positions, reps, seed));
        }
        let us_values = stats
            .iter()
            .map(|stats| stats.us_per_root)
            .collect::<Vec<_>>();
        let (us_mean, us_std) = mean_and_std(&us_values);
        let value_abs_max = stats
            .iter()
            .map(|stats| stats.value_abs)
            .fold(0.0f32, f32::max);
        let entropy_min = stats
            .iter()
            .map(|stats| stats.entropy_ratio)
            .fold(f32::INFINITY, f32::min);
        let avg_moves = stats.iter().map(|stats| stats.avg_moves).sum::<f32>() / stats.len() as f32;

        println!(
            "{:>3} {:>4} {:>2} {:>4} {:>3} {:>4} {:>9} {:>7} {:>5} {:>6.1} {:>10.2} {:>7.2} {:>8.3} {:>8.3} {:>8.1}",
            index + 1,
            config.model_channels,
            config.model_blocks,
            config.hidden_size,
            config.value_head_channels,
            config.value_hidden_size,
            params,
            az_arch_local_window(config),
            config.model_channels * 2,
            az_arch_dense64x6_mac_ratio(config) * 100.0,
            us_mean,
            us_std,
            value_abs_max,
            entropy_min,
            avg_moves
        );
    }
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        None => run_uci(),
        Some(CliCommand::Uci) => run_uci(),
        Some(CliCommand::AzInit(cmd)) => {
            let model_config = AzModelConfig {
                hidden_size: cmd.hidden,
                model_channels: cmd.model_channels.unwrap_or(32),
                model_blocks: cmd.model_blocks.unwrap_or(3),
                value_head_channels: cmd.value_head_channels.unwrap_or(8),
                value_hidden_size: cmd.value_hidden_size.unwrap_or(256),
                policy_condition_size: 32,
            }
            .normalized();
            model_config
                .validate_supported()
                .unwrap_or_else(|err| panic!("{err}"));
            let output = cmd.output;
            let seed = cmd.seed;
            let model = AzModel::random_with_config(model_config, seed);
            model.save(&output).unwrap_or_else(|err| {
                panic!("failed to write `{output}`: {err}");
            });
            println!("azmodel  : initialized (binary magic AZM1)");
            println!("arch     : {:?}", model.model_config);
            println!("seed     : {seed}");
            println!("output   : {output}");
        }
        Some(CliCommand::AzArchProbe(cmd)) => {
            run_az_arch_probe(cmd);
        }
        Some(CliCommand::AzLoop(cmd)) => {
            let config_path = cmd.config;
            let max_updates = cmd.updates;
            let Some(config) = load_or_create_az_loop_config(&config_path) else {
                return;
            };
            let start_update = read_az_loop_next_update(&config_path).unwrap_or(1).max(1);
            if start_update > 1 {
                println!(
                    "resume   : update starts at {} (from `{}`)",
                    start_update,
                    az_loop_progress_path(&config_path).display()
                );
            }
            let best_path = best_model_path(&config.model_path);
            let model_config = config.model_config();
            model_config
                .validate_supported()
                .unwrap_or_else(|err| panic!("{err}"));

            let model = if Path::new(&config.model_path).exists() {
                println!("model    : load {}", config.model_path);
                match AzModel::load(&config.model_path) {
                    Ok(model) => {
                        if model.model_config != model_config {
                            panic!(
                                "model config mismatch: file has {:?}, config has {:?}; initialize a new model or edit config",
                                model.model_config, model_config
                            );
                        }
                        model
                    }
                    Err(err) => {
                        println!(
                            "model    : reinit {} as random model ({err})",
                            config.model_path
                        );
                        AzModel::random_with_config(model_config, config.seed)
                    }
                }
            } else {
                println!("model    : init {}", config.model_path);
                AzModel::random_with_config(model_config, config.seed)
            };
            let replay_snapshot_path = az_loop_replay_snapshot_path(&config_path);
            let mut replay_pool =
                (config.replay_games > 0).then(|| AzExperiencePool::new(config.replay_games));
            if config.replay_games > 0 && replay_snapshot_path.exists() {
                match AzExperiencePool::load_snapshot_lz4(
                    &replay_snapshot_path,
                    config.replay_games,
                ) {
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
                "loop     : config={} mode=batch search={} sims={} model={:?} selfplay_batch_games={} epochs/update={} lr={} batch_size(per_gpu)={} global_step_samples={} max_sample_train_count={} max_plies={} selfplay_workers={} temp={}->{}/{}ply cpuct={} gumbel(max_actions={},scale={},value_scale={},maxvisit_init={},rescale={},mixed={}) td_lambda={} replay_games={} replay_samples={} mirror_probability={} checkpoint_interval={} max_checkpoints={} arena_interval={} arena_games_per_side={} arena_cpuct={} arena_processes={} tb_base={} tb_run={}",
                config_path,
                config.search_algorithm.as_str(),
                config.simulations,
                model_config,
                config.selfplay_batch_games,
                config.epochs,
                config.lr,
                config.batch_size,
                global_training_step_sample_count(config.batch_size),
                config.max_sample_train_count,
                config.max_plies,
                config.workers,
                config.temperature_start,
                config.temperature_end,
                config.temperature_decay_plies,
                config.cpuct,
                config.gumbel.max_num_considered_actions,
                config.gumbel.gumbel_scale,
                config.gumbel.value_scale,
                config.gumbel.maxvisit_init,
                config.gumbel.rescale_values,
                config.gumbel.use_mixed_value,
                config.td_lambda,
                config.replay_games,
                config.replay_samples,
                config.mirror_probability,
                config.checkpoint_interval,
                config.max_checkpoints,
                config.arena_interval,
                config.arena_games_per_side,
                config.arena_cpuct,
                config.arena_processes,
                config.tensorboard_logdir,
                tensorboard_encoded_subdir(&config)
            );
            if let Some(max_updates) = max_updates {
                println!("loop     : bounded run updates={}", max_updates.max(1));
            }
            let (selfplay_tx, selfplay_rx) = mpsc::channel::<SelfplayBatch>();
            let (trainer_tx, trainer_rx) = mpsc::channel::<TrainerEvent>();
            let shared_model = Arc::new(RwLock::new(SharedSelfplayModel {
                version: 0,
                model: model.clone(),
            }));
            let selfplay_pause =
                Arc::new((Mutex::new(SelfplayPauseState::default()), Condvar::new()));
            let mut selfplay_handles = Vec::with_capacity(config.workers.max(1));
            for worker_id in 0..config.workers.max(1) {
                let selfplay_stop = stop_requested.clone();
                let selfplay_config = config.clone();
                let selfplay_tx = selfplay_tx.clone();
                let shared_model = Arc::clone(&shared_model);
                let selfplay_pause = Arc::clone(&selfplay_pause);
                selfplay_handles.push(thread::spawn(move || {
                    let mut batch_index = 0usize;
                    let mut local_version = u64::MAX;
                    let mut local_model = AzModel::random_with_config(
                        selfplay_config.model_config(),
                        selfplay_config.seed ^ worker_id as u64,
                    );
                    while !selfplay_stop.load(Ordering::SeqCst) {
                        {
                            let (pause_lock, pause_cvar) = &*selfplay_pause;
                            let mut pause_state = pause_lock
                                .lock()
                                .unwrap_or_else(|_| panic!("selfplay pause state poisoned"));
                            while pause_state.paused && !selfplay_stop.load(Ordering::SeqCst) {
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
                        let loop_config = build_az_loop_config(&selfplay_config, batch_seed, 1);
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
            let trainer_model_path = config.model_path.clone();
            let trainer_snapshot_path = replay_snapshot_path.clone();
            let trainer_shared_model = Arc::clone(&shared_model);
            let trainer_handle = thread::spawn(move || {
                let mut trainer_model = model;
                let mut trainer_pool = replay_pool;
                let mut pending = PendingTrainingData::default();
                let mut train_index = 0usize;
                while let Ok(batch) = selfplay_rx.recv() {
                    if let Some(pool) = trainer_pool.as_mut() {
                        pool.add_games(batch.data.games.clone());
                    }
                    pending.push(batch);
                    while let Ok(batch) = selfplay_rx.try_recv() {
                        if let Some(pool) = trainer_pool.as_mut() {
                            pool.add_games(batch.data.games.clone());
                        }
                        pending.push(batch);
                    }
                    if trainer_stop.load(Ordering::SeqCst) {
                        continue;
                    }
                    if pending.selfplay.games.len() < trainer_config.selfplay_batch_games {
                        continue;
                    }
                    let Some(pool) = trainer_pool.as_mut() else {
                        continue;
                    };
                    if pool.sample_count()
                        < global_training_step_sample_count(trainer_config.batch_size)
                    {
                        continue;
                    }
                    let mut rng = chineseai::az::SplitMix64::new(
                        trainer_config.seed
                            ^ (train_index as u64).wrapping_mul(0xD1B5_4A32_D192_ED03),
                    );
                    let fresh_games = pending.selfplay.games.len();
                    let mut train_data = pending.selfplay.samples.clone();
                    let replay_data = pool.sample_uniform_games_marked_excluding_newest(
                        trainer_config.replay_samples,
                        trainer_config.max_sample_train_count as u32,
                        fresh_games,
                        &mut rng,
                    );
                    train_data.extend(replay_data);
                    if train_data.is_empty() {
                        continue;
                    }
                    let train_started = Instant::now();
                    let stats = train_samples(
                        &mut trainer_model,
                        &train_data,
                        trainer_config.epochs.max(1),
                        trainer_config.lr,
                        trainer_config.batch_size,
                        &mut rng,
                    );
                    let train_seconds = train_started.elapsed().as_secs_f32();
                    trainer_model
                        .save(&trainer_model_path)
                        .unwrap_or_else(|err| {
                            panic!("failed to write `{trainer_model_path}`: {err}")
                        });
                    {
                        let mut shared = trainer_shared_model
                            .write()
                            .unwrap_or_else(|_| panic!("shared selfplay model poisoned"));
                        shared.model = trainer_model.clone();
                        shared.version = shared.version.wrapping_add(1);
                    }
                    let report = build_async_training_report(
                        std::mem::take(&mut pending),
                        stats,
                        train_data.len(),
                        trainer_config.epochs,
                        train_seconds,
                        pool.game_count(),
                        pool.sample_count(),
                    );
                    if trainer_tx.send(TrainerEvent { report }).is_err() {
                        break;
                    }
                    train_index += 1;
                }
                if let Some(pool) = trainer_pool.as_mut()
                    && trainer_interrupted.load(Ordering::SeqCst)
                {
                    match pool.save_snapshot_lz4(&trainer_snapshot_path) {
                        Ok(()) => {
                            if pool.game_count() > 0 {
                                println!(
                                    "replay   : interrupt snapshot `{}` ({} games)",
                                    trainer_snapshot_path.display(),
                                    pool.game_count()
                                );
                            }
                        }
                        Err(err) => eprintln!("replay   : failed to write snapshot: {err}"),
                    }
                }
            });
            let mut exited_after_ctrl_c = false;
            let mut update = start_update;
            loop {
                if interrupted.load(Ordering::SeqCst) {
                    exited_after_ctrl_c = true;
                    break;
                }
                let started = Instant::now();
                let report = loop {
                    match trainer_rx.recv_timeout(Duration::from_millis(100)) {
                        Ok(TrainerEvent { report }) => break report,
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            if interrupted.load(Ordering::SeqCst) {
                                exited_after_ctrl_c = true;
                                break AzLoopReport {
                                    games: 0,
                                    samples: 0,
                                    red_wins: 0,
                                    black_wins: 0,
                                    draws: 0,
                                    avg_plies: 0.0,
                                    loss: 0.0,
                                    value_loss: 0.0,
                                    value_mse: 0.0,
                                    value_pred_mean: 0.0,
                                    value_target_mean: 0.0,
                                    policy_ce: 0.0,
                                    temperature_early_entropy: 0.0,
                                    temperature_mid_entropy: 0.0,
                                    selfplay_seconds: 0.0,
                                    train_seconds: 0.0,
                                    total_seconds: 0.0,
                                    games_per_second: 0.0,
                                    samples_per_second: 0.0,
                                    train_samples_per_second: 0.0,
                                    train_samples: 0,
                                    pool_games: 0,
                                    pool_samples: 0,
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
                                    terminal_max_plies: 0,
                                };
                            }
                        }
                        Err(mpsc::RecvTimeoutError::Disconnected) => {
                            if interrupted.load(Ordering::SeqCst) {
                                exited_after_ctrl_c = true;
                                break AzLoopReport {
                                    games: 0,
                                    samples: 0,
                                    red_wins: 0,
                                    black_wins: 0,
                                    draws: 0,
                                    avg_plies: 0.0,
                                    loss: 0.0,
                                    value_loss: 0.0,
                                    value_mse: 0.0,
                                    value_pred_mean: 0.0,
                                    value_target_mean: 0.0,
                                    policy_ce: 0.0,
                                    temperature_early_entropy: 0.0,
                                    temperature_mid_entropy: 0.0,
                                    selfplay_seconds: 0.0,
                                    train_seconds: 0.0,
                                    total_seconds: 0.0,
                                    games_per_second: 0.0,
                                    samples_per_second: 0.0,
                                    train_samples_per_second: 0.0,
                                    train_samples: 0,
                                    pool_games: 0,
                                    pool_samples: 0,
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
                                    terminal_max_plies: 0,
                                };
                            }
                            panic!("training thread exited before update {update}");
                        }
                    }
                };
                if exited_after_ctrl_c {
                    break;
                }
                write_az_loop_next_update(&config_path, update.saturating_add(1));
                if !best_path.exists() {
                    fs::copy(&config.model_path, &best_path).unwrap_or_else(|err| {
                        panic!(
                            "failed to initialize best model `{}` from `{}`: {err}",
                            best_path.display(),
                            config.model_path
                        );
                    });
                } else if let Err(err) = AzModel::load(&best_path) {
                    println!(
                        "best     : reset incompatible `{}` from current ({err})",
                        best_path.display()
                    );
                    fs::copy(&config.model_path, &best_path).unwrap_or_else(|copy_err| {
                        panic!(
                            "failed to reset best model `{}` from `{}`: {copy_err}",
                            best_path.display(),
                            config.model_path
                        );
                    });
                }
                let checkpoint_saved = if config.checkpoint_interval > 0
                    && update.is_multiple_of(config.checkpoint_interval)
                {
                    let path =
                        save_checkpoint_copy(&config.model_path, &config.checkpoint_dir, update);
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
                    "update {update:04}: games={} samples={} train_samples={} pool={}/{} fill={:.0}% R/B/D={}/{}/{} red_rate={:.3} avg_plies={:.1} loss={:.4} value_loss={:.4} value_mse={:.4} v_mu={:.3}/{:.3} policy_ce={:.4} lr={:.6} tempH={:.3}/{:.3} selfplay={:.1}s train={:.1}s gps={:.2} sps={:.1} train_sps={:.1} elapsed={:.1}s{}",
                    report.games,
                    report.samples,
                    report.train_samples,
                    report.pool_games,
                    report.pool_samples,
                    if config.replay_games == 0 {
                        0.0
                    } else {
                        100.0 * report.pool_games as f32 / config.replay_games as f32
                    },
                    report.red_wins,
                    report.black_wins,
                    report.draws,
                    report.red_wins as f32 / report.games.max(1) as f32,
                    report.avg_plies,
                    report.loss,
                    report.value_loss,
                    report.value_mse,
                    report.value_pred_mean,
                    report.value_target_mean,
                    report.policy_ce,
                    config.lr,
                    report.temperature_early_entropy,
                    report.temperature_mid_entropy,
                    report.selfplay_seconds,
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
                log_scalar(&mut tb, "train/policy_ce", update, report.policy_ce);
                log_scalar(&mut tb, "train/lr", update, config.lr);
                log_scalar(
                    &mut tb,
                    "pool/fill_ratio",
                    update,
                    if config.replay_games == 0 {
                        0.0
                    } else {
                        report.pool_games as f32 / config.replay_games as f32
                    },
                );
                log_scalar(&mut tb, "selfplay/games", update, report.games as f32);
                log_scalar(&mut tb, "selfplay/samples", update, report.samples as f32);
                log_scalar(&mut tb, "selfplay/avg_plies", update, report.avg_plies);
                log_scalar(
                    &mut tb,
                    "selfplay/temp_entropy_early",
                    update,
                    report.temperature_early_entropy,
                );
                log_scalar(
                    &mut tb,
                    "selfplay/temp_entropy_mid",
                    update,
                    report.temperature_mid_entropy,
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
                    "timing/selfplay_seconds",
                    update,
                    report.selfplay_seconds,
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
                    "terminal/no_legal_moves",
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
                    "terminal/max_plies",
                    update,
                    report.terminal_max_plies as f32,
                );
                if config.arena_interval > 0 && update.is_multiple_of(config.arena_interval) {
                    {
                        let (pause_lock, _) = &*selfplay_pause;
                        let mut pause_state = pause_lock
                            .lock()
                            .unwrap_or_else(|_| panic!("selfplay pause state poisoned"));
                        pause_state.paused = true;
                    }
                    println!("pause    : selfplay paused for arena");
                    let arena_eval_positions = load_arena_eval_positions(DEFAULT_ARENA_EVAL_FENS);
                    let arena_eval_fens = arena_eval_positions.len();
                    drop(arena_eval_positions);
                    let arena = run_arena_processes(ArenaProcessConfig {
                        candidate_path: &config.model_path,
                        baseline_path: best_path
                            .to_str()
                            .unwrap_or_else(|| panic!("best model path is not valid UTF-8")),
                        games_per_side: config.arena_games_per_side,
                        simulations: config.simulations,
                        max_plies: config.max_plies,
                        cpuct: config.arena_cpuct,
                        eval_fens_path: DEFAULT_ARENA_EVAL_FENS,
                        process_count: config.arena_processes,
                        seed: config.seed ^ (update as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                    });
                    let promoted = arena.score_rate() > 0.5;
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
                        "arena {update:04}: total={} fens={} W/L/D={}/{}/{} red={}/{} black={}/{} score={:.1} rate={:.3} elo={:.1} best={}{}",
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
                        arena.elo(),
                        best_path.display(),
                        if promoted { " promoted=current" } else { "" }
                    );
                    log_scalar(&mut tb, "arena/wins", update, arena.wins as f32);
                    log_scalar(&mut tb, "arena/losses", update, arena.losses as f32);
                    log_scalar(&mut tb, "arena/draws", update, arena.draws as f32);
                    log_scalar(&mut tb, "arena/score", update, arena.score());
                    log_scalar(&mut tb, "arena/score_rate", update, arena.score_rate());
                    log_scalar(&mut tb, "arena/elo", update, arena.elo());
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
                    {
                        let (pause_lock, pause_cvar) = &*selfplay_pause;
                        let mut pause_state = pause_lock
                            .lock()
                            .unwrap_or_else(|_| panic!("selfplay pause state poisoned"));
                        pause_state.paused = false;
                        pause_cvar.notify_all();
                    }
                    println!("resume   : selfplay resumed after arena");
                }
                update = update.saturating_add(1);
                if let Some(max_updates) = max_updates
                    && update.saturating_sub(start_update) >= max_updates.max(1)
                {
                    println!("loop     : reached bounded update limit ({max_updates})");
                    break;
                }
            }
            stop_requested.store(true, Ordering::SeqCst);
            {
                let (pause_lock, pause_cvar) = &*selfplay_pause;
                let mut pause_state = pause_lock
                    .lock()
                    .unwrap_or_else(|_| panic!("selfplay pause state poisoned"));
                pause_state.paused = false;
                pause_cvar.notify_all();
            }
            drop(trainer_rx);
            for handle in selfplay_handles {
                handle
                    .join()
                    .unwrap_or_else(|_| panic!("selfplay thread panicked"));
            }
            trainer_handle
                .join()
                .unwrap_or_else(|_| panic!("training thread panicked"));
            if !exited_after_ctrl_c {
                let _ = fs::remove_file(&replay_snapshot_path);
            }
        }
        Some(CliCommand::AzArenaWorker(cmd)) => {
            let candidate_path = cmd.candidate;
            let baseline_path = cmd.baseline;
            let red_games = cmd.red_games;
            let black_games = cmd.black_games;
            let simulations = cmd.simulations.max(1);
            let max_plies = cmd.max_plies.max(1);
            let arena_cpuct = cmd.arena_cpuct.max(0.0);
            let eval_fens_path = cmd.eval_fens_path;
            let seed = cmd.seed;
            let candidate = AzModel::load(&candidate_path).unwrap_or_else(|err| {
                panic!("failed to load `{candidate_path}`: {err}");
            });
            let baseline = AzModel::load(&baseline_path).unwrap_or_else(|err| {
                panic!("failed to load `{baseline_path}`: {err}");
            });
            let eval_positions = load_arena_eval_positions(&eval_fens_path);
            let report = play_arena_games_from_positions(
                &candidate,
                &baseline,
                &eval_positions,
                AzArenaConfig {
                    simulations,
                    max_plies,
                    games_as_red: red_games,
                    games_as_black: black_games,
                    seed,
                    cpuct: arena_cpuct,
                },
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
        Some(CliCommand::VsPikafish(cmd)) => {
            let pikafish_exe = cmd.pikafish_exe;
            let (config_path, simulations_override) = if let Some(value) = cmd.config_or_simulations
            {
                if let Ok(simulations) = value.parse::<usize>() {
                    (
                        DEFAULT_AZ_LOOP_CONFIG.to_string(),
                        Some(cmd.simulations.unwrap_or(simulations).max(1)),
                    )
                } else {
                    (value, cmd.simulations.map(|value| value.max(1)))
                }
            } else {
                (
                    DEFAULT_AZ_LOOP_CONFIG.to_string(),
                    cmd.simulations.map(|value| value.max(1)),
                )
            };
            let Some(config) = load_or_create_az_loop_config(&config_path) else {
                return;
            };
            let simulations = simulations_override.unwrap_or(config.simulations).max(1);
            let pikafish_depth = cmd.pikafish_depth.max(1);
            let games = cmd.games.max(1);
            let parallel_games = cmd.parallel_games.max(1);
            let eval_fens_path = cmd.eval_fens_path;
            let start_positions = load_arena_eval_positions(&eval_fens_path);
            let seed = 20260411_u64;
            let summary = run_vs_pikafish(
                Path::new(&pikafish_exe),
                Path::new(&config.model_path),
                &start_positions,
                VsPikafishConfig {
                    pikafish_depth,
                    total_games: games,
                    max_plies: config.max_plies,
                    simulations,
                    seed: seed ^ config.seed,
                    parallel_games,
                    search_algorithm: config.search_algorithm,
                    cpuct: config.cpuct,
                    gumbel: config.gumbel,
                },
            )
            .unwrap_or_else(|err| panic!("vs-pikafish failed: {err}"));
            println!(
                "vs-pikafish: config={} model={} search={} games={} fens={} parallel={} chinese W/L/D={}/{}/{} (as_red={} as_black={}) | pikafish_depth={} max_plies={} sims={}",
                config_path,
                config.model_path,
                config.search_algorithm.as_str(),
                summary.total_games,
                start_positions.len(),
                parallel_games.min(games),
                summary.chinese_wins,
                summary.chinese_losses,
                summary.draws,
                summary.chinese_wins_as_red,
                summary.chinese_wins_as_black,
                pikafish_depth,
                config.max_plies,
                simulations
            );
        }
    }
}
