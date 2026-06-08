use chineseai::az::{AzGumbelConfig, AzNnueArch, AzSearchAlgorithm};
use serde::{Deserialize, Serialize};
use std::{fmt::Write, fs, path::Path};

pub const DEFAULT_AZ_LOOP_CONFIG: &str = "chineseai.azloop.toml";
#[derive(Clone, Debug)]
pub struct AzLoopFileConfig {
    pub model_path: String,
    pub simulations: usize,
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
    pub temperature_end: f32,
    pub temperature_decay_plies: usize,
    pub search_algorithm: AzSearchAlgorithm,
    pub cpuct: f32,
    pub root_dirichlet_alpha: f32,
    pub root_exploration_fraction: f32,
    pub gumbel: AzGumbelConfig,
    pub replay_capacity: usize,
    pub train_warmup_samples: usize,
    pub train_samples_per_update: usize,
    pub train_epochs_per_update: usize,
    pub max_sample_train_count: u32,
    pub mirror_probability: f32,
    pub value_td_lambda: f32,
    pub train_value_weight: f32,
    pub train_policy_weight: f32,
    pub checkpoint_interval: usize,
    pub checkpoint_dir: String,
    pub max_checkpoints: usize,
    pub arena_interval: usize,
    pub arena_games_per_side: usize,
    pub arena_cpuct: f32,
    pub arena_promotion_rate: f32,
    pub arena_promotion_confidence_z: f32,
    pub arena_processes: usize,
    pub arena_pikafish_exe: String,
    pub arena_pikafish_start_update: usize,
    pub arena_pikafish_depth: u32,
    pub arena_pikafish_games: usize,
    pub arena_pikafish_parallel_games: usize,
    pub arena_pikafish_promotion_rate: f32,
    pub arena_pikafish_eval_fens: String,
    pub tensorboard_logdir: String,
}

impl Default for AzLoopFileConfig {
    fn default() -> Self {
        let gumbel = AzGumbelConfig {
            max_num_considered_actions: 32,
            ..AzGumbelConfig::default()
        };
        Self {
            model_path: "model.safetensors".into(),
            simulations: 256,
            selfplay_samples_per_update: 60000,
            lr: 0.0005,
            lr_min: 0.00003,
            lr_decay_start_update: 800,
            lr_decay_interval: 1000,
            lr_decay_factor: 0.33333334,
            batch_size: 1024,
            max_plies: 300,
            hidden_size: 192,
            seed: 20260412,
            workers: 250,
            temperature_start: 1.0,
            temperature_end: 0.1,
            temperature_decay_plies: 80,
            search_algorithm: AzSearchAlgorithm::AlphaZero,
            cpuct: 1.5,
            root_dirichlet_alpha: 0.3,
            root_exploration_fraction: 0.25,
            gumbel: AzGumbelConfig {
                gumbel_scale: 1.0,
                value_scale: 0.1,
                maxvisit_init: 50.0,
                ..gumbel
            },
            replay_capacity: 500000,
            train_warmup_samples: 150000,
            train_samples_per_update: 120000,
            train_epochs_per_update: 2,
            max_sample_train_count: 3,
            mirror_probability: 0.3,
            value_td_lambda: 0.95,
            train_value_weight: 1.0,
            train_policy_weight: 1.0,
            checkpoint_interval: 20,
            checkpoint_dir: "checkpoints".into(),
            max_checkpoints: 50,
            arena_interval: 10,
            arena_games_per_side: 100,
            arena_cpuct: 1.5,
            arena_promotion_rate: 0.50,
            arena_promotion_confidence_z: 1.28,
            arena_processes: 100,
            arena_pikafish_exe: String::new(),
            arena_pikafish_start_update: 1,
            arena_pikafish_depth: 1,
            arena_pikafish_games: 200,
            arena_pikafish_parallel_games: 100,
            arena_pikafish_promotion_rate: 0.60,
            arena_pikafish_eval_fens: "eval_fens.txt".into(),
            tensorboard_logdir: "runs/chineseai".into(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct AzLoopTomlConfig {
    pub model_path: String,
    pub simulations: usize,
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
    pub temperature_end: f32,
    pub temperature_decay_plies: usize,
    pub search_algorithm: String,
    pub cpuct: f32,
    pub root_dirichlet_alpha: f32,
    pub root_exploration_fraction: f32,
    gumbel_max_num_considered_actions: usize,
    gumbel_scale: f32,
    gumbel_value_scale: f32,
    gumbel_maxvisit_init: f32,
    gumbel_rescale_values: bool,
    gumbel_use_mixed_value: bool,
    pub replay_capacity: usize,
    pub train_warmup_samples: usize,
    pub train_samples_per_update: usize,
    pub train_epochs_per_update: usize,
    pub max_sample_train_count: u32,
    pub mirror_probability: f32,
    pub value_td_lambda: f32,
    pub train_value_weight: f32,
    pub train_policy_weight: f32,
    pub checkpoint_interval: usize,
    pub checkpoint_dir: String,
    pub max_checkpoints: usize,
    pub arena_interval: usize,
    pub arena_games_per_side: usize,
    pub arena_cpuct: f32,
    pub arena_promotion_rate: f32,
    pub arena_promotion_confidence_z: f32,
    pub arena_processes: usize,
    pub arena_pikafish_exe: String,
    pub arena_pikafish_start_update: usize,
    pub arena_pikafish_depth: u32,
    pub arena_pikafish_games: usize,
    pub arena_pikafish_parallel_games: usize,
    pub arena_pikafish_promotion_rate: f32,
    pub arena_pikafish_eval_fens: String,
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
            temperature_end: config.temperature_end,
            temperature_decay_plies: config.temperature_decay_plies,
            search_algorithm: config.search_algorithm.as_str().into(),
            cpuct: config.cpuct,
            root_dirichlet_alpha: config.root_dirichlet_alpha,
            root_exploration_fraction: config.root_exploration_fraction,
            gumbel_max_num_considered_actions: config.gumbel.max_num_considered_actions,
            gumbel_scale: config.gumbel.gumbel_scale,
            gumbel_value_scale: config.gumbel.value_scale,
            gumbel_maxvisit_init: config.gumbel.maxvisit_init,
            gumbel_rescale_values: config.gumbel.rescale_values,
            gumbel_use_mixed_value: config.gumbel.use_mixed_value,
            replay_capacity: config.replay_capacity,
            train_warmup_samples: config.train_warmup_samples,
            train_samples_per_update: config.train_samples_per_update,
            train_epochs_per_update: config.train_epochs_per_update,
            max_sample_train_count: config.max_sample_train_count,
            mirror_probability: config.mirror_probability,
            value_td_lambda: config.value_td_lambda,
            train_value_weight: config.train_value_weight,
            train_policy_weight: config.train_policy_weight,
            checkpoint_interval: config.checkpoint_interval,
            checkpoint_dir: config.checkpoint_dir.clone(),
            max_checkpoints: config.max_checkpoints,
            arena_interval: config.arena_interval,
            arena_games_per_side: config.arena_games_per_side,
            arena_cpuct: config.arena_cpuct,
            arena_promotion_rate: config.arena_promotion_rate,
            arena_promotion_confidence_z: config.arena_promotion_confidence_z,
            arena_processes: config.arena_processes,
            arena_pikafish_exe: config.arena_pikafish_exe.clone(),
            arena_pikafish_start_update: config.arena_pikafish_start_update,
            arena_pikafish_depth: config.arena_pikafish_depth,
            arena_pikafish_games: config.arena_pikafish_games,
            arena_pikafish_parallel_games: config.arena_pikafish_parallel_games,
            arena_pikafish_promotion_rate: config.arena_pikafish_promotion_rate,
            arena_pikafish_eval_fens: config.arena_pikafish_eval_fens.clone(),
            tensorboard_logdir: config.tensorboard_logdir.clone(),
        }
    }
}

impl From<AzLoopTomlConfig> for AzLoopFileConfig {
    fn from(config: AzLoopTomlConfig) -> Self {
        Self {
            model_path: config.model_path,
            simulations: config.simulations,
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
            temperature_end: config.temperature_end,
            temperature_decay_plies: config.temperature_decay_plies,
            search_algorithm: AzSearchAlgorithm::parse(&config.search_algorithm).unwrap_or_else(
                || panic!("invalid search_algorithm `{}`", config.search_algorithm),
            ),
            cpuct: config.cpuct,
            root_dirichlet_alpha: config.root_dirichlet_alpha,
            root_exploration_fraction: config.root_exploration_fraction,
            gumbel: AzGumbelConfig {
                max_num_considered_actions: config.gumbel_max_num_considered_actions,
                gumbel_scale: config.gumbel_scale,
                value_scale: config.gumbel_value_scale,
                maxvisit_init: config.gumbel_maxvisit_init,
                rescale_values: config.gumbel_rescale_values,
                use_mixed_value: config.gumbel_use_mixed_value,
            },
            replay_capacity: config.replay_capacity,
            train_warmup_samples: config.train_warmup_samples,
            train_samples_per_update: config.train_samples_per_update,
            train_epochs_per_update: config.train_epochs_per_update,
            max_sample_train_count: config.max_sample_train_count,
            mirror_probability: config.mirror_probability,
            value_td_lambda: config.value_td_lambda,
            train_value_weight: config.train_value_weight,
            train_policy_weight: config.train_policy_weight,
            checkpoint_interval: config.checkpoint_interval,
            checkpoint_dir: config.checkpoint_dir,
            max_checkpoints: config.max_checkpoints,
            arena_interval: config.arena_interval,
            arena_games_per_side: config.arena_games_per_side,
            arena_cpuct: config.arena_cpuct,
            arena_promotion_rate: config.arena_promotion_rate,
            arena_promotion_confidence_z: config.arena_promotion_confidence_z,
            arena_processes: config.arena_processes,
            arena_pikafish_exe: config.arena_pikafish_exe,
            arena_pikafish_start_update: config.arena_pikafish_start_update,
            arena_pikafish_depth: config.arena_pikafish_depth,
            arena_pikafish_games: config.arena_pikafish_games,
            arena_pikafish_parallel_games: config.arena_pikafish_parallel_games,
            arena_pikafish_promotion_rate: config.arena_pikafish_promotion_rate,
            arena_pikafish_eval_fens: config.arena_pikafish_eval_fens,
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
        line!("temperature_end", f(self.temperature_end));
        line!("temperature_decay_plies", self.temperature_decay_plies);
        line!("search_algorithm", q(self.search_algorithm.as_str()));
        line!("cpuct", f(self.cpuct));
        line!("root_dirichlet_alpha", f(self.root_dirichlet_alpha));
        line!(
            "root_exploration_fraction",
            f(self.root_exploration_fraction)
        );
        line!(
            "gumbel_max_num_considered_actions",
            self.gumbel.max_num_considered_actions
        );
        line!("gumbel_scale", f(self.gumbel.gumbel_scale));
        line!("gumbel_value_scale", f(self.gumbel.value_scale));
        line!("gumbel_maxvisit_init", f(self.gumbel.maxvisit_init));
        line!("gumbel_rescale_values", self.gumbel.rescale_values);
        line!("gumbel_use_mixed_value", self.gumbel.use_mixed_value);
        line!("replay_capacity", self.replay_capacity);
        line!("train_warmup_samples", self.train_warmup_samples);
        line!("train_samples_per_update", self.train_samples_per_update);
        line!("train_epochs_per_update", self.train_epochs_per_update);
        line!("max_sample_train_count", self.max_sample_train_count);
        line!("mirror_probability", f(self.mirror_probability));
        line!("value_td_lambda", f(self.value_td_lambda));
        line!("train_value_weight", f(self.train_value_weight));
        line!("train_policy_weight", f(self.train_policy_weight));
        line!("checkpoint_interval", self.checkpoint_interval);
        line!("checkpoint_dir", q(&self.checkpoint_dir));
        line!("max_checkpoints", self.max_checkpoints);
        line!("arena_interval", self.arena_interval);
        line!("arena_games_per_side", self.arena_games_per_side);
        line!("arena_cpuct", f(self.arena_cpuct));
        line!("arena_promotion_rate", f(self.arena_promotion_rate));
        line!(
            "arena_promotion_confidence_z",
            f(self.arena_promotion_confidence_z)
        );
        line!("arena_processes", self.arena_processes);
        line!("arena_pikafish_exe", q(&self.arena_pikafish_exe));
        line!(
            "arena_pikafish_start_update",
            self.arena_pikafish_start_update
        );
        line!("arena_pikafish_depth", self.arena_pikafish_depth);
        line!("arena_pikafish_games", self.arena_pikafish_games);
        line!(
            "arena_pikafish_parallel_games",
            self.arena_pikafish_parallel_games
        );
        line!(
            "arena_pikafish_promotion_rate",
            f(self.arena_pikafish_promotion_rate)
        );
        line!(
            "arena_pikafish_eval_fens",
            q(&self.arena_pikafish_eval_fens)
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
        self.selfplay_samples_per_update = self.selfplay_samples_per_update.max(1);
        self.lr = self.lr.max(0.0);
        self.lr_min = self.lr_min.max(0.0).min(self.lr);
        self.lr_decay_interval = self.lr_decay_interval.max(1);
        self.lr_decay_factor = self.lr_decay_factor.clamp(0.0, 1.0);
        self.batch_size = self.batch_size.max(1);
        self.max_plies = self.max_plies.max(1);
        self.hidden_size = self.hidden_size.max(1);
        self.workers = self.workers.max(1);
        self.temperature_start = self.temperature_start.max(0.0);
        self.temperature_end = self.temperature_end.max(0.0);
        self.cpuct = self.cpuct.max(0.0);
        self.root_dirichlet_alpha = self.root_dirichlet_alpha.max(0.0);
        self.root_exploration_fraction = self.root_exploration_fraction.clamp(0.0, 1.0);
        self.train_warmup_samples = self.train_warmup_samples.max(1);
        self.train_samples_per_update = self.train_samples_per_update.max(1);
        self.train_epochs_per_update = self.train_epochs_per_update.max(1);
        self.gumbel.max_num_considered_actions = self.gumbel.max_num_considered_actions.max(1);
        self.gumbel.gumbel_scale = self.gumbel.gumbel_scale.max(0.0);
        self.gumbel.value_scale = self.gumbel.value_scale.max(0.0);
        self.gumbel.maxvisit_init = self.gumbel.maxvisit_init.max(0.0);
        self.arena_cpuct = self.arena_cpuct.max(0.0);
        self.mirror_probability = self.mirror_probability.clamp(0.0, 1.0);
        self.value_td_lambda = self.value_td_lambda.clamp(0.0, 1.0);
        self.train_value_weight = self.train_value_weight.max(0.0);
        self.train_policy_weight = self.train_policy_weight.max(0.0);
        self.max_checkpoints = self.max_checkpoints.max(1);
        self.arena_games_per_side = self.arena_games_per_side.max(1);
        self.arena_processes = self.arena_processes.max(1);
        self.arena_promotion_rate = self.arena_promotion_rate.clamp(0.0, 1.0);
        self.arena_promotion_confidence_z = self.arena_promotion_confidence_z.max(0.0);
        self.arena_pikafish_start_update = self.arena_pikafish_start_update.max(1);
        self.arena_pikafish_depth = self.arena_pikafish_depth.max(1);
        self.arena_pikafish_games = self.arena_pikafish_games.max(1);
        self.arena_pikafish_parallel_games = self.arena_pikafish_parallel_games.max(1);
        self.arena_pikafish_promotion_rate = self.arena_pikafish_promotion_rate.clamp(0.0, 1.0);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_writer_uses_short_float_literals() {
        let text = AzLoopFileConfig::default().to_file_text();

        assert!(text.contains("lr = 0.0005\n"));
        assert!(text.contains("lr_min = 0.00003\n"));
        assert!(text.contains("temperature_start = 1.0\n"));
        assert!(text.contains("temperature_end = 0.1\n"));
        assert!(text.contains("value_td_lambda = 0.95\n"));
        assert!(!text.contains("train_value_head"));
        assert!(!text.contains("train_policy_head"));
        assert!(!text.contains("000000047"));
        assert!(!text.contains("000000023"));

        let parsed = AzLoopFileConfig::parse(&text);
        assert_eq!(parsed.model_path, "model.safetensors");
        assert!((parsed.lr - 0.0005).abs() < 1e-9);
        assert!((parsed.value_td_lambda - 0.95).abs() < 1e-6);
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
