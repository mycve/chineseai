use chineseai::az::AzNnueArch;
use serde::{Deserialize, Serialize};
use std::{fmt::Write, fs, path::Path};

pub const DEFAULT_AZ_LOOP_CONFIG: &str = "chineseai.azloop.toml";
#[derive(Clone, Debug)]
pub struct AzLoopFileConfig {
    pub model_path: String,
    pub simulations: usize,
    pub low_simulations: usize,
    pub low_simulation_probability: f32,
    pub low_simulation_policy_weight: f32,
    pub opening_policy_zero_plies: usize,
    pub selfplay_samples_per_update: usize,
    pub lr: f32,
    pub lr_min: f32,
    pub lr_decay_start_update: usize,
    pub lr_decay_interval: usize,
    pub lr_decay_factor: f32,
    pub batch_size: usize,
    pub max_plies: usize,
    pub hidden_size: usize,
    pub seed: u64,
    pub workers: usize,
    pub temperature_start: f32,
    pub temperature_endgame: f32,
    pub temperature_decay_delay_plies: usize,
    pub temperature_decay_plies: usize,
    pub temperature_value_cutoff: f32,
    pub temperature_visit_offset: f32,
    pub cpuct: f32,
    pub cpuct_at_root: f32,
    pub cpuct_base: f32,
    pub cpuct_factor: f32,
    pub cpuct_base_at_root: f32,
    pub cpuct_factor_at_root: f32,
    pub root_dirichlet_alpha: f32,
    pub root_exploration_fraction: f32,
    pub fpu_value: f32,
    pub fpu_value_at_root: f32,
    pub draw_score: f32,
    pub moves_left_max_effect: f32,
    pub moves_left_slope: f32,
    pub moves_left_threshold: f32,
    pub moves_left_constant_factor: f32,
    pub moves_left_scaled_factor: f32,
    pub moves_left_quadratic_factor: f32,
    pub policy_softmax_temp: f32,
    pub opening_fens_path: String,
    pub resign_percentage: f32,
    pub resign_playthrough: f32,
    pub replay_capacity: usize,
    pub replay_recent_sample_fraction: f32,
    pub replay_recent_window_updates: u32,
    pub train_warmup_samples: usize,
    pub train_samples_per_update: usize,
    pub train_epochs_per_update: usize,
    pub mirror_probability: f32,
    pub deblunder_q_gap: f32,
    pub train_value_weight: f32,
    pub train_policy_weight: f32,
    pub checkpoint_interval: usize,
    pub checkpoint_dir: String,
    pub max_checkpoints: usize,
    pub arena_interval: usize,
    pub arena_cpuct: f32,
    pub arena_promotion_rate: f32,
    pub arena_promotion_confidence_z: f32,
    pub arena_processes: usize,
    pub arena_opening_book: String,
    pub arena_opening_positions: usize,
    pub arena_opening_plies_min: usize,
    pub arena_opening_plies_max: usize,
    pub pikafish_label_eval_sqlite: String,
    pub pikafish_label_eval_interval: usize,
    pub pikafish_label_eval_limit: usize,
    pub pikafish_label_eval_simulations: usize,
    pub pikafish_label_eval_cpuct: f32,
    pub tensorboard_logdir: String,
}

impl Default for AzLoopFileConfig {
    fn default() -> Self {
        Self {
            model_path: "model.safetensors".into(),
            simulations: 512,
            low_simulations: 256,
            low_simulation_probability: 0.35,
            low_simulation_policy_weight: 0.35,
            opening_policy_zero_plies: 4,
            selfplay_samples_per_update: 240000,
            lr: 0.001,
            lr_min: 0.0001,
            lr_decay_start_update: 800,
            lr_decay_interval: 1000,
            lr_decay_factor: 0.33333334,
            batch_size: 1024,
            max_plies: 300,
            hidden_size: 192,
            seed: 20260412,
            workers: 250,
            temperature_start: 0.9,
            temperature_endgame: 0.5,
            temperature_decay_delay_plies: 20,
            temperature_decay_plies: 60,
            temperature_value_cutoff: 0.12,
            temperature_visit_offset: -0.8,
            cpuct: 0.65,
            cpuct_at_root: 2.53,
            cpuct_base: 19652.0,
            cpuct_factor: 2.0,
            cpuct_base_at_root: 19652.0,
            cpuct_factor_at_root: 2.0,
            root_dirichlet_alpha: 0.12,
            root_exploration_fraction: 0.1,
            fpu_value: 0.23,
            fpu_value_at_root: 1.0,
            draw_score: 0.0,
            moves_left_max_effect: 0.25,
            moves_left_slope: 0.002,
            moves_left_threshold: 0.6,
            moves_left_constant_factor: 0.0,
            moves_left_scaled_factor: 0.15,
            moves_left_quadratic_factor: 0.85,
            policy_softmax_temp: 1.45,
            opening_fens_path: String::new(),
            resign_percentage: 1.0,
            resign_playthrough: 20.0,
            replay_capacity: 5000000,
            replay_recent_sample_fraction: 0.4,
            replay_recent_window_updates: 5000,
            train_warmup_samples: 240000,
            train_samples_per_update: 120000,
            train_epochs_per_update: 1,
            mirror_probability: 0.3,
            deblunder_q_gap: 0.15,
            train_value_weight: 1.0,
            train_policy_weight: 1.0,
            checkpoint_interval: 20,
            checkpoint_dir: "checkpoints".into(),
            max_checkpoints: 50,
            arena_interval: 20,
            arena_cpuct: 1.5,
            arena_promotion_rate: 0.50,
            arena_promotion_confidence_z: 1.28,
            arena_processes: 250,
            arena_opening_book: "opening.obk".into(),
            arena_opening_positions: 300,
            arena_opening_plies_min: 6,
            arena_opening_plies_max: 10,
            pikafish_label_eval_sqlite: "eval/pikafish-random-5000-d8.sqlite".into(),
            pikafish_label_eval_interval: 20,
            pikafish_label_eval_limit: 1000,
            pikafish_label_eval_simulations: 256,
            pikafish_label_eval_cpuct: 1.5,
            tensorboard_logdir: "runs/chineseai".into(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct AzLoopTomlConfig {
    pub model_path: String,
    pub simulations: usize,
    pub low_simulations: usize,
    pub low_simulation_probability: f32,
    pub low_simulation_policy_weight: f32,
    pub opening_policy_zero_plies: usize,
    pub selfplay_samples_per_update: usize,
    pub lr: f32,
    pub lr_min: f32,
    pub lr_decay_start_update: usize,
    pub lr_decay_interval: usize,
    pub lr_decay_factor: f32,
    pub batch_size: usize,
    pub max_plies: usize,
    pub hidden_size: usize,
    pub seed: u64,
    pub workers: usize,
    pub temperature_start: f32,
    pub temperature_endgame: f32,
    pub temperature_decay_delay_plies: usize,
    pub temperature_decay_plies: usize,
    pub temperature_value_cutoff: f32,
    pub temperature_visit_offset: f32,
    pub cpuct: f32,
    pub cpuct_at_root: f32,
    pub cpuct_base: f32,
    pub cpuct_factor: f32,
    pub cpuct_base_at_root: f32,
    pub cpuct_factor_at_root: f32,
    pub root_dirichlet_alpha: f32,
    pub root_exploration_fraction: f32,
    pub fpu_value: f32,
    pub fpu_value_at_root: f32,
    pub draw_score: f32,
    pub moves_left_max_effect: f32,
    pub moves_left_slope: f32,
    pub moves_left_threshold: f32,
    pub moves_left_constant_factor: f32,
    pub moves_left_scaled_factor: f32,
    pub moves_left_quadratic_factor: f32,
    pub policy_softmax_temp: f32,
    pub opening_fens_path: String,
    pub resign_percentage: f32,
    pub resign_playthrough: f32,
    pub replay_capacity: usize,
    pub replay_recent_sample_fraction: f32,
    pub replay_recent_window_updates: u32,
    pub train_warmup_samples: usize,
    pub train_samples_per_update: usize,
    pub train_epochs_per_update: usize,
    pub mirror_probability: f32,
    pub deblunder_q_gap: f32,
    pub train_value_weight: f32,
    pub train_policy_weight: f32,
    pub checkpoint_interval: usize,
    pub checkpoint_dir: String,
    pub max_checkpoints: usize,
    pub arena_interval: usize,
    pub arena_cpuct: f32,
    pub arena_promotion_rate: f32,
    pub arena_promotion_confidence_z: f32,
    pub arena_processes: usize,
    pub arena_opening_book: String,
    pub arena_opening_positions: usize,
    pub arena_opening_plies_min: usize,
    pub arena_opening_plies_max: usize,
    pub pikafish_label_eval_sqlite: String,
    pub pikafish_label_eval_interval: usize,
    pub pikafish_label_eval_limit: usize,
    pub pikafish_label_eval_simulations: usize,
    pub pikafish_label_eval_cpuct: f32,
    #[serde(skip_serializing)]
    pub arena_pikafish_exe: String,
    #[serde(skip_serializing)]
    pub arena_pikafish_start_update: usize,
    #[serde(skip_serializing)]
    pub arena_pikafish_depth: u32,
    #[serde(skip_serializing)]
    pub arena_pikafish_games: usize,
    #[serde(skip_serializing)]
    pub arena_pikafish_parallel_games: usize,
    #[serde(skip_serializing)]
    pub arena_pikafish_promotion_rate: f32,
    pub tensorboard_logdir: String,
}

impl Default for AzLoopTomlConfig {
    fn default() -> Self {
        Self::from(&AzLoopFileConfig::default())
    }
}

impl From<&AzLoopFileConfig> for AzLoopTomlConfig {
    fn from(config: &AzLoopFileConfig) -> Self {
        Self {
            model_path: config.model_path.clone(),
            simulations: config.simulations,
            low_simulations: config.low_simulations,
            low_simulation_probability: config.low_simulation_probability,
            low_simulation_policy_weight: config.low_simulation_policy_weight,
            opening_policy_zero_plies: config.opening_policy_zero_plies,
            selfplay_samples_per_update: config.selfplay_samples_per_update,
            lr: config.lr,
            lr_min: config.lr_min,
            lr_decay_start_update: config.lr_decay_start_update,
            lr_decay_interval: config.lr_decay_interval,
            lr_decay_factor: config.lr_decay_factor,
            batch_size: config.batch_size,
            max_plies: config.max_plies,
            hidden_size: config.hidden_size,
            seed: config.seed,
            workers: config.workers,
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
            opening_fens_path: config.opening_fens_path.clone(),
            resign_percentage: config.resign_percentage,
            resign_playthrough: config.resign_playthrough,
            replay_capacity: config.replay_capacity,
            replay_recent_sample_fraction: config.replay_recent_sample_fraction,
            replay_recent_window_updates: config.replay_recent_window_updates,
            train_warmup_samples: config.train_warmup_samples,
            train_samples_per_update: config.train_samples_per_update,
            train_epochs_per_update: config.train_epochs_per_update,
            mirror_probability: config.mirror_probability,
            deblunder_q_gap: config.deblunder_q_gap,
            train_value_weight: config.train_value_weight,
            train_policy_weight: config.train_policy_weight,
            checkpoint_interval: config.checkpoint_interval,
            checkpoint_dir: config.checkpoint_dir.clone(),
            max_checkpoints: config.max_checkpoints,
            arena_interval: config.arena_interval,
            arena_cpuct: config.arena_cpuct,
            arena_promotion_rate: config.arena_promotion_rate,
            arena_promotion_confidence_z: config.arena_promotion_confidence_z,
            arena_processes: config.arena_processes,
            arena_opening_book: config.arena_opening_book.clone(),
            arena_opening_positions: config.arena_opening_positions,
            arena_opening_plies_min: config.arena_opening_plies_min,
            arena_opening_plies_max: config.arena_opening_plies_max,
            pikafish_label_eval_sqlite: config.pikafish_label_eval_sqlite.clone(),
            pikafish_label_eval_interval: config.pikafish_label_eval_interval,
            pikafish_label_eval_limit: config.pikafish_label_eval_limit,
            pikafish_label_eval_simulations: config.pikafish_label_eval_simulations,
            pikafish_label_eval_cpuct: config.pikafish_label_eval_cpuct,
            arena_pikafish_exe: String::new(),
            arena_pikafish_start_update: 1,
            arena_pikafish_depth: 1,
            arena_pikafish_games: 1,
            arena_pikafish_parallel_games: 1,
            arena_pikafish_promotion_rate: 0.0,
            tensorboard_logdir: config.tensorboard_logdir.clone(),
        }
    }
}

impl From<AzLoopTomlConfig> for AzLoopFileConfig {
    fn from(config: AzLoopTomlConfig) -> Self {
        Self {
            model_path: config.model_path,
            simulations: config.simulations,
            low_simulations: config.low_simulations,
            low_simulation_probability: config.low_simulation_probability,
            low_simulation_policy_weight: config.low_simulation_policy_weight,
            opening_policy_zero_plies: config.opening_policy_zero_plies,
            selfplay_samples_per_update: config.selfplay_samples_per_update,
            lr: config.lr,
            lr_min: config.lr_min,
            lr_decay_start_update: config.lr_decay_start_update,
            lr_decay_interval: config.lr_decay_interval,
            lr_decay_factor: config.lr_decay_factor,
            batch_size: config.batch_size,
            max_plies: config.max_plies,
            hidden_size: config.hidden_size,
            seed: config.seed,
            workers: config.workers,
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
            opening_fens_path: config.opening_fens_path,
            resign_percentage: config.resign_percentage,
            resign_playthrough: config.resign_playthrough,
            replay_capacity: config.replay_capacity,
            replay_recent_sample_fraction: config.replay_recent_sample_fraction,
            replay_recent_window_updates: config.replay_recent_window_updates,
            train_warmup_samples: config.train_warmup_samples,
            train_samples_per_update: config.train_samples_per_update,
            train_epochs_per_update: config.train_epochs_per_update,
            mirror_probability: config.mirror_probability,
            deblunder_q_gap: config.deblunder_q_gap,
            train_value_weight: config.train_value_weight,
            train_policy_weight: config.train_policy_weight,
            checkpoint_interval: config.checkpoint_interval,
            checkpoint_dir: config.checkpoint_dir,
            max_checkpoints: config.max_checkpoints,
            arena_interval: config.arena_interval,
            arena_cpuct: config.arena_cpuct,
            arena_promotion_rate: config.arena_promotion_rate,
            arena_promotion_confidence_z: config.arena_promotion_confidence_z,
            arena_processes: config.arena_processes,
            arena_opening_book: config.arena_opening_book,
            arena_opening_positions: config.arena_opening_positions,
            arena_opening_plies_min: config.arena_opening_plies_min,
            arena_opening_plies_max: config.arena_opening_plies_max,
            pikafish_label_eval_sqlite: config.pikafish_label_eval_sqlite,
            pikafish_label_eval_interval: config.pikafish_label_eval_interval,
            pikafish_label_eval_limit: config.pikafish_label_eval_limit,
            pikafish_label_eval_simulations: config.pikafish_label_eval_simulations,
            pikafish_label_eval_cpuct: config.pikafish_label_eval_cpuct,
            tensorboard_logdir: config.tensorboard_logdir,
        }
    }
}

impl AzLoopFileConfig {
    pub fn to_file_text(&self) -> String {
        fn q(value: &str) -> String {
            format!("{value:?}")
        }
        fn f(value: f32) -> String {
            if value == 0.0 {
                return "0.0".into();
            }
            let out = value.to_string();
            if out == "-0" {
                return "0.0".into();
            }
            if out.contains('.') {
                out
            } else {
                format!("{out}.0")
            }
        }
        let mut out = String::new();
        macro_rules! line {
            ($name:literal, $value:expr) => {
                writeln!(out, "{} = {}", $name, $value).unwrap();
            };
        }
        line!("model_path", q(&self.model_path));
        line!("simulations", self.simulations);
        line!("low_simulations", self.low_simulations);
        line!(
            "low_simulation_probability",
            f(self.low_simulation_probability)
        );
        line!(
            "low_simulation_policy_weight",
            f(self.low_simulation_policy_weight)
        );
        line!("opening_policy_zero_plies", self.opening_policy_zero_plies);
        line!(
            "selfplay_samples_per_update",
            self.selfplay_samples_per_update
        );
        line!("lr", f(self.lr));
        line!("lr_min", f(self.lr_min));
        line!("lr_decay_start_update", self.lr_decay_start_update);
        line!("lr_decay_interval", self.lr_decay_interval);
        line!("lr_decay_factor", f(self.lr_decay_factor));
        line!("batch_size", self.batch_size);
        line!("max_plies", self.max_plies);
        line!("hidden_size", self.hidden_size);
        line!("seed", self.seed);
        line!("workers", self.workers);
        line!("temperature_start", f(self.temperature_start));
        line!("temperature_endgame", f(self.temperature_endgame));
        line!(
            "temperature_decay_delay_plies",
            self.temperature_decay_delay_plies
        );
        line!("temperature_decay_plies", self.temperature_decay_plies);
        line!("temperature_value_cutoff", f(self.temperature_value_cutoff));
        line!("temperature_visit_offset", f(self.temperature_visit_offset));
        line!("cpuct", f(self.cpuct));
        line!("cpuct_at_root", f(self.cpuct_at_root));
        line!("cpuct_base", f(self.cpuct_base));
        line!("cpuct_factor", f(self.cpuct_factor));
        line!("cpuct_base_at_root", f(self.cpuct_base_at_root));
        line!("cpuct_factor_at_root", f(self.cpuct_factor_at_root));
        line!("root_dirichlet_alpha", f(self.root_dirichlet_alpha));
        line!(
            "root_exploration_fraction",
            f(self.root_exploration_fraction)
        );
        line!("fpu_value", f(self.fpu_value));
        line!("fpu_value_at_root", f(self.fpu_value_at_root));
        line!("draw_score", f(self.draw_score));
        line!("moves_left_max_effect", f(self.moves_left_max_effect));
        line!("moves_left_slope", f(self.moves_left_slope));
        line!("moves_left_threshold", f(self.moves_left_threshold));
        line!(
            "moves_left_constant_factor",
            f(self.moves_left_constant_factor)
        );
        line!("moves_left_scaled_factor", f(self.moves_left_scaled_factor));
        line!(
            "moves_left_quadratic_factor",
            f(self.moves_left_quadratic_factor)
        );
        line!("policy_softmax_temp", f(self.policy_softmax_temp));
        line!("opening_fens_path", q(&self.opening_fens_path));
        line!("resign_percentage", f(self.resign_percentage));
        line!("resign_playthrough", f(self.resign_playthrough));
        line!("replay_capacity", self.replay_capacity);
        line!(
            "replay_recent_sample_fraction",
            f(self.replay_recent_sample_fraction)
        );
        line!(
            "replay_recent_window_updates",
            self.replay_recent_window_updates
        );
        line!("train_warmup_samples", self.train_warmup_samples);
        line!("train_samples_per_update", self.train_samples_per_update);
        line!("train_epochs_per_update", self.train_epochs_per_update);
        line!("mirror_probability", f(self.mirror_probability));
        line!("deblunder_q_gap", f(self.deblunder_q_gap));
        line!("train_value_weight", f(self.train_value_weight));
        line!("train_policy_weight", f(self.train_policy_weight));
        line!("checkpoint_interval", self.checkpoint_interval);
        line!("checkpoint_dir", q(&self.checkpoint_dir));
        line!("max_checkpoints", self.max_checkpoints);
        line!("arena_interval", self.arena_interval);
        line!("arena_cpuct", f(self.arena_cpuct));
        line!("arena_promotion_rate", f(self.arena_promotion_rate));
        line!(
            "arena_promotion_confidence_z",
            f(self.arena_promotion_confidence_z)
        );
        line!("arena_processes", self.arena_processes);
        line!("arena_opening_book", q(&self.arena_opening_book));
        line!("arena_opening_positions", self.arena_opening_positions);
        line!("arena_opening_plies_min", self.arena_opening_plies_min);
        line!("arena_opening_plies_max", self.arena_opening_plies_max);
        line!(
            "pikafish_label_eval_sqlite",
            q(&self.pikafish_label_eval_sqlite)
        );
        line!(
            "pikafish_label_eval_interval",
            self.pikafish_label_eval_interval
        );
        line!("pikafish_label_eval_limit", self.pikafish_label_eval_limit);
        line!(
            "pikafish_label_eval_simulations",
            self.pikafish_label_eval_simulations
        );
        line!(
            "pikafish_label_eval_cpuct",
            f(self.pikafish_label_eval_cpuct)
        );
        line!("tensorboard_logdir", q(&self.tensorboard_logdir));
        out
    }

    fn parse(text: &str) -> Self {
        let config = toml::from_str::<AzLoopTomlConfig>(text)
            .unwrap_or_else(|err| panic!("invalid az-loop TOML config: {err}"));
        AzLoopFileConfig::from(config).normalize()
    }

    pub fn arch(&self) -> AzNnueArch {
        AzNnueArch {
            hidden_size: self.hidden_size,
        }
    }

    fn normalize(mut self) -> Self {
        self.simulations = self.simulations.max(1);
        self.low_simulations = self.low_simulations.max(1).min(self.simulations);
        self.low_simulation_probability = self.low_simulation_probability.clamp(0.0, 1.0);
        self.low_simulation_policy_weight = self.low_simulation_policy_weight.max(0.0);
        self.selfplay_samples_per_update = self.selfplay_samples_per_update.max(1);
        self.lr = self.lr.max(0.0);
        self.lr_min = self.lr_min.max(0.0).min(self.lr);
        self.lr_decay_interval = self.lr_decay_interval.max(1);
        self.lr_decay_factor = self.lr_decay_factor.clamp(0.0, 1.0);
        self.batch_size = self.batch_size.max(1);
        self.max_plies = self.max_plies.max(1);
        self.opening_policy_zero_plies = self.opening_policy_zero_plies.min(self.max_plies);
        self.hidden_size = self.hidden_size.max(1);
        self.workers = self.workers.max(1);
        self.temperature_start = self.temperature_start.max(0.0);
        self.temperature_endgame = self.temperature_endgame.max(0.0);
        self.temperature_decay_delay_plies = self.temperature_decay_delay_plies.min(self.max_plies);
        self.temperature_decay_plies = self.temperature_decay_plies.min(self.max_plies);
        self.temperature_value_cutoff = self.temperature_value_cutoff.max(0.0);
        self.cpuct = self.cpuct.max(0.0);
        self.cpuct_at_root = self.cpuct_at_root.max(0.0);
        self.cpuct_base = self.cpuct_base.max(1.0);
        self.cpuct_factor = self.cpuct_factor.max(0.0);
        self.cpuct_base_at_root = self.cpuct_base_at_root.max(1.0);
        self.cpuct_factor_at_root = self.cpuct_factor_at_root.max(0.0);
        self.root_dirichlet_alpha = self.root_dirichlet_alpha.max(0.0);
        self.root_exploration_fraction = self.root_exploration_fraction.clamp(0.0, 1.0);
        self.fpu_value = self.fpu_value.max(0.0);
        self.fpu_value_at_root = self.fpu_value_at_root.clamp(-1.0, 1.0);
        self.draw_score = self.draw_score.clamp(-1.0, 1.0);
        self.moves_left_max_effect = self.moves_left_max_effect.max(0.0);
        self.moves_left_slope = self.moves_left_slope.max(0.0);
        self.moves_left_threshold = self.moves_left_threshold.clamp(0.0, 1.0);
        self.policy_softmax_temp = self.policy_softmax_temp.max(1e-3);
        self.resign_percentage = self.resign_percentage.clamp(0.0, 100.0);
        self.resign_playthrough = self.resign_playthrough.clamp(0.0, 100.0);
        self.replay_recent_sample_fraction = self.replay_recent_sample_fraction.clamp(0.0, 1.0);
        self.replay_recent_window_updates = self.replay_recent_window_updates.max(1);
        self.train_warmup_samples = self.train_warmup_samples.max(1);
        self.train_samples_per_update = self.train_samples_per_update.max(1);
        self.train_epochs_per_update = self.train_epochs_per_update.max(1);
        self.arena_cpuct = self.arena_cpuct.max(0.0);
        self.mirror_probability = self.mirror_probability.clamp(0.0, 1.0);
        self.deblunder_q_gap = self.deblunder_q_gap.max(0.0);
        self.train_value_weight = self.train_value_weight.max(0.0);
        self.train_policy_weight = self.train_policy_weight.max(0.0);
        self.max_checkpoints = self.max_checkpoints.max(1);
        self.arena_processes = self.arena_processes.max(1);
        self.arena_promotion_rate = self.arena_promotion_rate.clamp(0.0, 1.0);
        self.arena_promotion_confidence_z = self.arena_promotion_confidence_z.max(0.0);
        self.arena_opening_positions = self.arena_opening_positions.max(1);
        self.pikafish_label_eval_simulations = self.pikafish_label_eval_simulations.max(1);
        self.pikafish_label_eval_cpuct = self.pikafish_label_eval_cpuct.max(0.0);
        if self.arena_opening_plies_min > self.arena_opening_plies_max {
            std::mem::swap(
                &mut self.arena_opening_plies_min,
                &mut self.arena_opening_plies_max,
            );
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_writer_uses_short_float_literals() {
        let text = AzLoopFileConfig::default().to_file_text();

        assert!(text.contains("lr = 0.001\n"));
        assert!(text.contains("lr_min = 0.0001\n"));
        assert!(text.contains("opening_policy_zero_plies = 4\n"));
        assert!(text.contains("temperature_start = 0.9\n"));
        assert!(text.contains("temperature_endgame = 0.5\n"));
        assert!(text.contains("temperature_decay_delay_plies = 20\n"));
        assert!(text.contains("temperature_decay_plies = 60\n"));
        assert!(!text.contains("temperature_cutoff_plies"));
        assert!(text.contains("temperature_value_cutoff = 0.12\n"));
        assert!(text.contains("temperature_visit_offset = -0.8\n"));
        assert!(text.contains("cpuct = 0.65\n"));
        assert!(text.contains("cpuct_at_root = 2.53\n"));
        assert!(text.contains("cpuct_base = 19652.0\n"));
        assert!(text.contains("cpuct_factor = 2.0\n"));
        assert!(text.contains("cpuct_base_at_root = 19652.0\n"));
        assert!(text.contains("cpuct_factor_at_root = 2.0\n"));
        assert!(text.contains("fpu_value = 0.23\n"));
        assert!(text.contains("fpu_value_at_root = 1.0\n"));
        assert!(text.contains("draw_score = 0.0\n"));
        assert!(text.contains("moves_left_max_effect = 0.25\n"));
        assert!(text.contains("moves_left_slope = 0.002\n"));
        assert!(text.contains("moves_left_threshold = 0.6\n"));
        assert!(text.contains("moves_left_constant_factor = 0.0\n"));
        assert!(text.contains("moves_left_scaled_factor = 0.15\n"));
        assert!(text.contains("moves_left_quadratic_factor = 0.85\n"));
        assert!(text.contains("policy_softmax_temp = 1.45\n"));
        assert!(text.contains("opening_fens_path = \"\"\n"));
        assert!(text.contains("resign_percentage = 1.0\n"));
        assert!(text.contains("resign_playthrough = 20.0\n"));
        assert!(text.contains("selfplay_samples_per_update = 240000\n"));
        assert!(text.contains("replay_recent_window_updates = 5000\n"));
        assert!(text.contains("deblunder_q_gap = 0.15\n"));
        assert!(text.contains("arena_opening_book = \"opening.obk\"\n"));
        assert!(text.contains("arena_opening_positions = 300\n"));
        assert!(text.contains("arena_opening_plies_min = 6\n"));
        assert!(text.contains("arena_opening_plies_max = 10\n"));
        assert!(text.contains("arena_interval = 20\n"));
        assert!(
            text.contains("pikafish_label_eval_sqlite = \"eval/pikafish-random-5000-d8.sqlite\"\n")
        );
        assert!(text.contains("pikafish_label_eval_interval = 20\n"));
        assert!(text.contains("pikafish_label_eval_limit = 1000\n"));
        assert!(text.contains("pikafish_label_eval_simulations = 256\n"));
        assert!(text.contains("pikafish_label_eval_cpuct = 1.5\n"));
        assert!(!text.contains("root_exploration_plies"));
        assert!(!text.contains("gumbel"));
        assert!(!text.contains("search_algorithm"));
        assert!(!text.contains("arena_pikafish"));
        assert!(!text.contains("arena_eval_fens"));
        assert!(!text.contains("000000047"));
        assert!(!text.contains("000000023"));

        let parsed = AzLoopFileConfig::parse(&text);
        assert_eq!(parsed.model_path, "model.safetensors");
        assert!((parsed.lr - 0.001).abs() < 1e-9);
        assert!((parsed.deblunder_q_gap - 0.15).abs() < 1e-6);
        assert_eq!(parsed.arena_interval, 20);
        assert_eq!(parsed.pikafish_label_eval_interval, 20);
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
