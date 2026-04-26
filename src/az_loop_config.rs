use chineseai::az::{AzGumbelConfig, AzSearchAlgorithm};
use serde::Deserialize;
use std::{fs, path::Path};

pub const DEFAULT_AZ_LOOP_CONFIG: &str = "chineseai.azloop.toml";
const DEFAULT_WORKER_CAP: usize = 32;

fn default_parallel_workers() -> usize {
    std::thread::available_parallelism()
        .map(|count| count.get().saturating_sub(1).max(1))
        .map(|workers| workers.min(DEFAULT_WORKER_CAP))
        .unwrap_or(8)
}
#[derive(Clone, Debug)]
pub struct AzLoopFileConfig {
    pub model_path: String,
    pub simulations: usize,
    pub selfplay_batch_games: usize,
    pub epochs: usize,
    pub lr: f32,
    pub batch_size: usize,
    pub max_sample_train_count: usize,
    pub max_plies: usize,
    pub hidden_size: usize,
    pub trunk_depth: usize,
    pub seed: u64,
    pub workers: usize,
    pub temperature_start: f32,
    pub temperature_end: f32,
    pub temperature_decay_plies: usize,
    pub search_algorithm: AzSearchAlgorithm,
    pub cpuct: f32,
    pub root_dirichlet_alpha: f32,
    pub root_exploration_fraction: f32,
    pub gumbel: AzGumbelConfig,
    pub td_lambda: f32,
    pub replay_games: usize,
    pub replay_samples: usize,
    pub mirror_probability: f32,
    pub checkpoint_interval: usize,
    pub checkpoint_dir: String,
    pub max_checkpoints: usize,
    pub arena_interval: usize,
    pub arena_games_per_side: usize,
    pub arena_cpuct: f32,
    pub arena_processes: usize,
    pub tensorboard_logdir: String,
}

impl Default for AzLoopFileConfig {
    fn default() -> Self {
        let default_workers = default_parallel_workers();
        Self {
            model_path: "chineseai.nnue".into(),
            simulations: 1200,
            selfplay_batch_games: default_workers.max(1),
            epochs: 2,
            lr: 0.0003,
            batch_size: 1024,
            max_sample_train_count: 3,
            max_plies: 300,
            hidden_size: 128,
            trunk_depth: 2,
            seed: 20260409,
            workers: default_workers,
            temperature_start: 1.0,
            temperature_end: 0.1,
            temperature_decay_plies: 40,
            search_algorithm: AzSearchAlgorithm::AlphaZero,
            cpuct: 1.5,
            root_dirichlet_alpha: 0.3,
            root_exploration_fraction: 0.25,
            gumbel: AzGumbelConfig::default(),
            td_lambda: 0.75,
            replay_games: 5000,
            replay_samples: 0,
            mirror_probability: 0.3,
            checkpoint_interval: 20,
            checkpoint_dir: "checkpoints".into(),
            max_checkpoints: 50,
            arena_interval: 20,
            arena_games_per_side: 50,
            arena_cpuct: 1.5,
            arena_processes: default_workers,
            tensorboard_logdir: "runs/chineseai".into(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AzLoopTomlConfig {
    pub model_path: Option<String>,
    pub simulations: Option<usize>,
    pub selfplay_batch_games: Option<usize>,
    pub epochs: Option<usize>,
    pub lr: Option<f32>,
    pub batch_size: Option<usize>,
    pub max_sample_train_count: Option<usize>,
    pub max_plies: Option<usize>,
    pub hidden_size: Option<usize>,
    pub trunk_depth: Option<usize>,
    pub seed: Option<u64>,
    pub workers: Option<usize>,
    pub temperature_start: Option<f32>,
    pub temperature_end: Option<f32>,
    pub temperature_decay_plies: Option<usize>,
    pub search_algorithm: Option<String>,
    pub cpuct: Option<f32>,
    pub root_dirichlet_alpha: Option<f32>,
    pub root_exploration_fraction: Option<f32>,
    gumbel_max_num_considered_actions: Option<usize>,
    gumbel_scale: Option<f32>,
    gumbel_value_scale: Option<f32>,
    gumbel_maxvisit_init: Option<f32>,
    gumbel_rescale_values: Option<bool>,
    gumbel_use_mixed_value: Option<bool>,
    pub td_lambda: Option<f32>,
    pub replay_games: Option<usize>,
    pub replay_samples: Option<usize>,
    pub mirror_probability: Option<f32>,
    pub checkpoint_interval: Option<usize>,
    pub checkpoint_dir: Option<String>,
    pub max_checkpoints: Option<usize>,
    pub arena_interval: Option<usize>,
    pub arena_games_per_side: Option<usize>,
    pub arena_cpuct: Option<f32>,
    pub arena_processes: Option<usize>,
    pub tensorboard_logdir: Option<String>,
}

impl AzLoopTomlConfig {
    fn apply_to(self, config: &mut AzLoopFileConfig) {
        if let Some(value) = self.model_path {
            config.model_path = value;
        }
        if let Some(value) = self.simulations {
            config.simulations = value;
        }
        if let Some(value) = self.selfplay_batch_games {
            config.selfplay_batch_games = value;
        }
        if let Some(value) = self.epochs {
            config.epochs = value;
        }
        if let Some(value) = self.lr {
            config.lr = value;
        }
        if let Some(value) = self.batch_size {
            config.batch_size = value;
        }
        if let Some(value) = self.max_sample_train_count {
            config.max_sample_train_count = value;
        }
        if let Some(value) = self.max_plies {
            config.max_plies = value;
        }
        if let Some(value) = self.hidden_size {
            config.hidden_size = value;
        }
        if let Some(value) = self.trunk_depth {
            config.trunk_depth = value;
        }
        if let Some(value) = self.seed {
            config.seed = value;
        }
        if let Some(value) = self.workers {
            config.workers = value;
        }
        if let Some(value) = self.temperature_start {
            config.temperature_start = value;
        }
        if let Some(value) = self.temperature_end {
            config.temperature_end = value;
        }
        if let Some(value) = self.temperature_decay_plies {
            config.temperature_decay_plies = value;
        }
        if let Some(value) = self.search_algorithm {
            config.search_algorithm = AzSearchAlgorithm::parse(&value)
                .unwrap_or_else(|| panic!("invalid search_algorithm `{value}`"));
        }
        if let Some(value) = self.cpuct {
            config.cpuct = value;
        }
        if let Some(value) = self.root_dirichlet_alpha {
            config.root_dirichlet_alpha = value;
        }
        if let Some(value) = self.root_exploration_fraction {
            config.root_exploration_fraction = value;
        }
        if let Some(value) = self.gumbel_max_num_considered_actions {
            config.gumbel.max_num_considered_actions = value;
        }
        if let Some(value) = self.gumbel_scale {
            config.gumbel.gumbel_scale = value;
        }
        if let Some(value) = self.gumbel_value_scale {
            config.gumbel.value_scale = value;
        }
        if let Some(value) = self.gumbel_maxvisit_init {
            config.gumbel.maxvisit_init = value;
        }
        if let Some(value) = self.gumbel_rescale_values {
            config.gumbel.rescale_values = value;
        }
        if let Some(value) = self.gumbel_use_mixed_value {
            config.gumbel.use_mixed_value = value;
        }
        if let Some(value) = self.td_lambda {
            config.td_lambda = value;
        }
        if let Some(value) = self.replay_games {
            config.replay_games = value;
        }
        if let Some(value) = self.replay_samples {
            config.replay_samples = value;
        }
        if let Some(value) = self.mirror_probability {
            config.mirror_probability = value;
        }
        if let Some(value) = self.checkpoint_interval {
            config.checkpoint_interval = value;
        }
        if let Some(value) = self.checkpoint_dir {
            config.checkpoint_dir = value;
        }
        if let Some(value) = self.max_checkpoints {
            config.max_checkpoints = value;
        }
        if let Some(value) = self.arena_interval {
            config.arena_interval = value;
        }
        if let Some(value) = self.arena_games_per_side {
            config.arena_games_per_side = value;
        }
        if let Some(value) = self.arena_cpuct {
            config.arena_cpuct = value;
        }
        if let Some(value) = self.arena_processes {
            config.arena_processes = value;
        }
        if let Some(value) = self.tensorboard_logdir {
            config.tensorboard_logdir = value;
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
# selfplay_batch_games:
#   Generate this many self-play games, then run one training update.
#   With multiple workers, batches are accumulated across all workers.
#
# Value targets:
#   td_lambda mixes each position's future MCTS root values with the final game outcome.
#   1.0 keeps pure AlphaZero-style terminal labels; 0.75 is the default.
#
# Self-play policy temperature (linear in ply index, 0-based before each search):
#   temperature_start -> temperature_end over plies [0, temperature_decay_plies), then constant.
#   Defaults: 1.0 -> 0.1 by ply 40; raise decay_plies or temperature_end if openings collapse too early.
#
# Pipeline:
#   Self-play and training run in separate long-lived threads.
#   workers controls how many independent self-play threads run in parallel.
#   The generated default is capped at 32 because this scalar MCTS/CNN code often slows down
#   when hundreds of tiny self-play threads fight over cache and memory bandwidth.
#   Self-play batches accumulate globally across workers.
#   Every time selfplay_batch_games complete, one training update runs and then
#   the fresh weights are published back to all self-play threads.
#
# Replay:
#   replay_games keeps the most recent N complete games in memory.
#   Each training update always includes all fresh samples collected since the previous update.
#   replay_samples adds this many extra old samples from replay; replay_samples=0 means fresh-only.
#   The current fresh games are excluded from replay sampling for that same update.
#   Ctrl+C writes "<this_conf_filename>.replay.lz4" (LZ4); next az-loop loads it into the pool then
#   deletes the file. A full run without interrupt removes any leftover snapshot at exit.
#   Replay snapshot format is versioned; older .replay.lz4 files from previous formats are rejected.
#
# Optimizer:
#   AdamW is used with mini-batch gradient accumulation.
#   epochs means how many passes to make over each fresh training window; 2-3 is a good range.
#   batch_size is per GPU per training step. Effective global batch size is
#   batch_size multiplied by the number of visible CUDA devices.
#   Example: with 4 GPUs, batch_size=256 processes 1024 samples per step.
#   max_sample_train_count removes samples after they have been used this many times.
#   workers is the number of independent self-play threads.
#   lr=0.0003 is a safer default than the old SGD-style 0.001 for self-play targets.
#
# Augmentation:
#   mirror_probability mirrors board files a<->i for this fraction of training samples.
#   Xiangqi rules are left/right symmetric, so value stays unchanged and policy moves are mirrored.
#
# Checkpoints & Arena:
#   Training appends "<this_conf_filename>.progress" with next_update=... after each weight update.
#   Delete that file to reset the update counter to 1 (TensorBoard/checkpoint numbering).
#   checkpoint_interval saves a timestamp-free numbered copy every N updates.
#   max_checkpoints keeps only the newest N checkpoint files in checkpoint_dir.
#   arena_interval runs current-vs-best evaluation every N updates.
#   arena_games_per_side=50 means 50 games as Red and 50 as Black.
#   search_algorithm="alphazero" keeps the original PUCT search; "gumbel_alphazero" uses
#   mctx-style root Gumbel top-k, Sequential Halving, deterministic interior selection, and
#   softmax(policy_logits + completed_qvalues) policy targets.
#   cpuct/root_dirichlet_* are used only by AlphaZero self-play search.
#   gumbel_* follows mctx defaults: max_num_considered_actions=16, gumbel_scale=1.0,
#   value_scale=0.1, maxvisit_init=50.0, rescale_values=true, use_mixed_value=true.
#   For perfect-information eval you can set gumbel_scale=0.0.
#   arena_cpuct applies during arena search.
#   tensorboard_logdir is the ROOT; each run writes under a subdir whose name encodes it_*,
#   sim_*, bs_*, lr_*, so TensorBoard Web can compare experiments side by side.

model_path = "{model_path}"
simulations = {simulations}
selfplay_batch_games = {selfplay_batch_games}
epochs = {epochs}
lr = {lr}
batch_size = {batch_size}
max_sample_train_count = {max_sample_train_count}
max_plies = {max_plies}
hidden_size = {hidden_size}
trunk_depth = {trunk_depth}
seed = {seed}
workers = {workers}
temperature_start = {temperature_start}
temperature_end = {temperature_end}
temperature_decay_plies = {temperature_decay_plies}
search_algorithm = "{search_algorithm}"
cpuct = {cpuct}
root_dirichlet_alpha = {root_dirichlet_alpha}
root_exploration_fraction = {root_exploration_fraction}
gumbel_max_num_considered_actions = {gumbel_max_num_considered_actions}
gumbel_scale = {gumbel_scale}
gumbel_value_scale = {gumbel_value_scale}
gumbel_maxvisit_init = {gumbel_maxvisit_init}
gumbel_rescale_values = {gumbel_rescale_values}
gumbel_use_mixed_value = {gumbel_use_mixed_value}
td_lambda = {td_lambda}
replay_games = {replay_games}
replay_samples = {replay_samples}
mirror_probability = {mirror_probability}
checkpoint_interval = {checkpoint_interval}
checkpoint_dir = "{checkpoint_dir}"
max_checkpoints = {max_checkpoints}
arena_interval = {arena_interval}
arena_games_per_side = {arena_games_per_side}
arena_cpuct = {arena_cpuct}
arena_processes = {arena_processes}
tensorboard_logdir = "{tensorboard_logdir}"
"#,
            model_path = self.model_path,
            simulations = self.simulations,
            selfplay_batch_games = self.selfplay_batch_games,
            epochs = self.epochs,
            lr = self.lr,
            batch_size = self.batch_size,
            max_sample_train_count = self.max_sample_train_count,
            max_plies = self.max_plies,
            hidden_size = self.hidden_size,
            trunk_depth = self.trunk_depth,
            seed = self.seed,
            workers = self.workers,
            temperature_start = self.temperature_start,
            temperature_end = self.temperature_end,
            temperature_decay_plies = self.temperature_decay_plies,
            search_algorithm = self.search_algorithm.as_str(),
            cpuct = self.cpuct,
            root_dirichlet_alpha = self.root_dirichlet_alpha,
            root_exploration_fraction = self.root_exploration_fraction,
            gumbel_max_num_considered_actions = self.gumbel.max_num_considered_actions,
            gumbel_scale = self.gumbel.gumbel_scale,
            gumbel_value_scale = self.gumbel.value_scale,
            gumbel_maxvisit_init = self.gumbel.maxvisit_init,
            gumbel_rescale_values = self.gumbel.rescale_values,
            gumbel_use_mixed_value = self.gumbel.use_mixed_value,
            td_lambda = self.td_lambda,
            replay_games = self.replay_games,
            replay_samples = self.replay_samples,
            mirror_probability = self.mirror_probability,
            checkpoint_interval = self.checkpoint_interval,
            checkpoint_dir = self.checkpoint_dir,
            max_checkpoints = self.max_checkpoints,
            arena_interval = self.arena_interval,
            arena_games_per_side = self.arena_games_per_side,
            arena_cpuct = self.arena_cpuct,
            arena_processes = self.arena_processes,
            tensorboard_logdir = self.tensorboard_logdir,
        )
    }

    fn parse(text: &str) -> Self {
        let mut config = Self::default();
        toml::from_str::<AzLoopTomlConfig>(text)
            .unwrap_or_else(|err| panic!("invalid az-loop TOML config: {err}"))
            .apply_to(&mut config);
        config.normalize()
    }

    fn normalize(mut self) -> Self {
        self.simulations = self.simulations.max(1);
        self.selfplay_batch_games = self.selfplay_batch_games.max(1);
        self.epochs = self.epochs.max(1);
        self.batch_size = self.batch_size.max(1);
        self.max_sample_train_count = self.max_sample_train_count.max(1);
        self.max_plies = self.max_plies.max(1);
        self.hidden_size = self.hidden_size.max(1);
        self.workers = self.workers.max(1);
        self.temperature_start = self.temperature_start.max(0.0);
        self.temperature_end = self.temperature_end.max(0.0);
        self.cpuct = self.cpuct.max(0.0);
        self.root_dirichlet_alpha = self.root_dirichlet_alpha.max(0.0);
        self.root_exploration_fraction = self.root_exploration_fraction.clamp(0.0, 1.0);
        self.gumbel.max_num_considered_actions = self.gumbel.max_num_considered_actions.max(1);
        self.gumbel.gumbel_scale = self.gumbel.gumbel_scale.max(0.0);
        self.gumbel.value_scale = self.gumbel.value_scale.max(0.0);
        self.gumbel.maxvisit_init = self.gumbel.maxvisit_init.max(0.0);
        self.td_lambda = self.td_lambda.clamp(0.0, 1.0);
        self.arena_cpuct = self.arena_cpuct.max(0.0);
        self.mirror_probability = self.mirror_probability.clamp(0.0, 1.0);
        self.max_checkpoints = self.max_checkpoints.max(1);
        self.arena_games_per_side = self.arena_games_per_side.max(1);
        self.arena_processes = self.arena_processes.max(1);
        self
    }
}

pub fn load_or_create_az_loop_config(path: &str) -> Option<AzLoopFileConfig> {
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
