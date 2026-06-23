use chineseai::az::AzNnueArch;
use serde::{Deserialize, Serialize};
use std::{fmt::Write, fs, path::Path};

pub const DEFAULT_AZ_LOOP_CONFIG: &str = "chineseai.azloop.toml";
#[derive(Clone, Debug)]
pub struct AzLoopFileConfig {
    pub model_path: String,
    pub gumbel_actions: usize,
    pub gumbel_scale: f32,
    pub gumbel_value_scale: f32,
    pub gumbel_maxvisit_init: f32,
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
    pub opening_fens_path: String,
    pub resign_percentage: f32,
    pub resign_playthrough: f32,
    pub replay_capacity: usize,
    pub train_warmup_samples: usize,
    pub train_samples_per_update: usize,
    pub train_epochs_per_update: usize,
    pub max_sample_train_count: u32,
    pub mirror_probability: f32,
    pub td_lambda: f32,
    pub train_value_weight: f32,
    pub train_policy_weight: f32,
    pub checkpoint_interval: usize,
    pub checkpoint_dir: String,
    pub max_checkpoints: usize,
    pub arena_interval: usize,
    pub arena_promotion_rate: f32,
    pub arena_promotion_confidence_z: f32,
    pub arena_processes: usize,
    pub arena_opening_book: String,
    pub arena_opening_positions: usize,
    pub arena_opening_plies_min: usize,
    pub arena_opening_plies_max: usize,
    pub tensorboard_logdir: String,
}

impl Default for AzLoopFileConfig {
    fn default() -> Self {
        Self {
            model_path: "model.safetensors".into(),
            gumbel_actions: 24,
            gumbel_scale: 1.0,
            gumbel_value_scale: 0.1,
            gumbel_maxvisit_init: 50.0,
            simulations: 800,
            selfplay_samples_per_update: 60000,
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
            opening_fens_path: String::new(),
            resign_percentage: 1.0,
            resign_playthrough: 20.0,
            replay_capacity: 500000,
            train_warmup_samples: 300000,
            train_samples_per_update: 120000,
            train_epochs_per_update: 2,
            max_sample_train_count: 3,
            mirror_probability: 0.3,
            td_lambda: 0.95,
            train_value_weight: 1.0,
            train_policy_weight: 1.0,
            checkpoint_interval: 20,
            checkpoint_dir: "checkpoints".into(),
            max_checkpoints: 50,
            arena_interval: 10,
            arena_promotion_rate: 0.50,
            arena_promotion_confidence_z: 1.28,
            arena_processes: 250,
            arena_opening_book: "opening.obk".into(),
            arena_opening_positions: 300,
            arena_opening_plies_min: 6,
            arena_opening_plies_max: 10,
            tensorboard_logdir: "runs/chineseai".into(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct AzLoopTomlConfig {
    pub model_path: String,
    pub gumbel_actions: usize,
    pub gumbel_scale: f32,
    pub gumbel_value_scale: f32,
    pub gumbel_maxvisit_init: f32,
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
    pub opening_fens_path: String,
    pub resign_percentage: f32,
    pub resign_playthrough: f32,
    pub replay_capacity: usize,
    pub train_warmup_samples: usize,
    pub train_samples_per_update: usize,
    pub train_epochs_per_update: usize,
    pub max_sample_train_count: u32,
    pub mirror_probability: f32,
    pub td_lambda: f32,
    pub train_value_weight: f32,
    pub train_policy_weight: f32,
    pub checkpoint_interval: usize,
    pub checkpoint_dir: String,
    pub max_checkpoints: usize,
    pub arena_interval: usize,
    pub arena_promotion_rate: f32,
    pub arena_promotion_confidence_z: f32,
    pub arena_processes: usize,
    pub arena_opening_book: String,
    pub arena_opening_positions: usize,
    pub arena_opening_plies_min: usize,
    pub arena_opening_plies_max: usize,
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
            gumbel_actions: config.gumbel_actions,
            gumbel_scale: config.gumbel_scale,
            gumbel_value_scale: config.gumbel_value_scale,
            gumbel_maxvisit_init: config.gumbel_maxvisit_init,
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
            opening_fens_path: config.opening_fens_path.clone(),
            resign_percentage: config.resign_percentage,
            resign_playthrough: config.resign_playthrough,
            replay_capacity: config.replay_capacity,
            train_warmup_samples: config.train_warmup_samples,
            train_samples_per_update: config.train_samples_per_update,
            train_epochs_per_update: config.train_epochs_per_update,
            max_sample_train_count: config.max_sample_train_count,
            mirror_probability: config.mirror_probability,
            td_lambda: config.td_lambda,
            train_value_weight: config.train_value_weight,
            train_policy_weight: config.train_policy_weight,
            checkpoint_interval: config.checkpoint_interval,
            checkpoint_dir: config.checkpoint_dir.clone(),
            max_checkpoints: config.max_checkpoints,
            arena_interval: config.arena_interval,
            arena_promotion_rate: config.arena_promotion_rate,
            arena_promotion_confidence_z: config.arena_promotion_confidence_z,
            arena_processes: config.arena_processes,
            arena_opening_book: config.arena_opening_book.clone(),
            arena_opening_positions: config.arena_opening_positions,
            arena_opening_plies_min: config.arena_opening_plies_min,
            arena_opening_plies_max: config.arena_opening_plies_max,
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
            gumbel_actions: config.gumbel_actions,
            gumbel_scale: config.gumbel_scale,
            gumbel_value_scale: config.gumbel_value_scale,
            gumbel_maxvisit_init: config.gumbel_maxvisit_init,
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
            opening_fens_path: config.opening_fens_path,
            resign_percentage: config.resign_percentage,
            resign_playthrough: config.resign_playthrough,
            replay_capacity: config.replay_capacity,
            train_warmup_samples: config.train_warmup_samples,
            train_samples_per_update: config.train_samples_per_update,
            train_epochs_per_update: config.train_epochs_per_update,
            max_sample_train_count: config.max_sample_train_count,
            mirror_probability: config.mirror_probability,
            td_lambda: config.td_lambda,
            train_value_weight: config.train_value_weight,
            train_policy_weight: config.train_policy_weight,
            checkpoint_interval: config.checkpoint_interval,
            checkpoint_dir: config.checkpoint_dir,
            max_checkpoints: config.max_checkpoints,
            arena_interval: config.arena_interval,
            arena_promotion_rate: config.arena_promotion_rate,
            arena_promotion_confidence_z: config.arena_promotion_confidence_z,
            arena_processes: config.arena_processes,
            arena_opening_book: config.arena_opening_book,
            arena_opening_positions: config.arena_opening_positions,
            arena_opening_plies_min: config.arena_opening_plies_min,
            arena_opening_plies_max: config.arena_opening_plies_max,
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
        line!("gumbel_actions", self.gumbel_actions);
        line!("gumbel_scale", f(self.gumbel_scale));
        line!("gumbel_value_scale", f(self.gumbel_value_scale));
        line!("gumbel_maxvisit_init", f(self.gumbel_maxvisit_init));
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
        line!("opening_fens_path", q(&self.opening_fens_path));
        line!("resign_percentage", f(self.resign_percentage));
        line!("resign_playthrough", f(self.resign_playthrough));
        line!("replay_capacity", self.replay_capacity);
        line!("train_warmup_samples", self.train_warmup_samples);
        line!("train_samples_per_update", self.train_samples_per_update);
        line!("train_epochs_per_update", self.train_epochs_per_update);
        line!("max_sample_train_count", self.max_sample_train_count);
        line!("mirror_probability", f(self.mirror_probability));
        line!("td_lambda", f(self.td_lambda));
        line!("train_value_weight", f(self.train_value_weight));
        line!("train_policy_weight", f(self.train_policy_weight));
        line!("checkpoint_interval", self.checkpoint_interval);
        line!("checkpoint_dir", q(&self.checkpoint_dir));
        line!("max_checkpoints", self.max_checkpoints);
        line!("arena_interval", self.arena_interval);
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
        self.gumbel_actions = self.gumbel_actions.max(1);
        self.gumbel_scale = self.gumbel_scale.max(0.0);
        self.gumbel_value_scale = self.gumbel_value_scale.max(0.0);
        self.gumbel_maxvisit_init = self.gumbel_maxvisit_init.max(0.0);
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
        self.resign_percentage = self.resign_percentage.clamp(0.0, 100.0);
        self.resign_playthrough = self.resign_playthrough.clamp(0.0, 100.0);
        self.train_warmup_samples = self.train_warmup_samples.max(1);
        self.train_samples_per_update = self.train_samples_per_update.max(1);
        self.train_epochs_per_update = self.train_epochs_per_update.max(1);
        self.mirror_probability = self.mirror_probability.clamp(0.0, 1.0);
        self.td_lambda = self.td_lambda.clamp(0.0, 1.0);
        self.train_value_weight = self.train_value_weight.max(0.0);
        self.train_policy_weight = self.train_policy_weight.max(0.0);
        self.max_checkpoints = self.max_checkpoints.max(1);
        self.arena_processes = self.arena_processes.max(1);
        self.arena_promotion_rate = self.arena_promotion_rate.clamp(0.0, 1.0);
        self.arena_promotion_confidence_z = self.arena_promotion_confidence_z.max(0.0);
        self.arena_opening_positions = self.arena_opening_positions.max(1);
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
        assert!(text.contains("opening_fens_path = \"\"\n"));
        assert!(text.contains("resign_percentage = 1.0\n"));
        assert!(text.contains("resign_playthrough = 20.0\n"));
        assert!(text.contains("td_lambda = 0.95\n"));
        assert!(text.contains("arena_opening_book = \"opening.obk\"\n"));
        assert!(text.contains("arena_opening_positions = 300\n"));
        assert!(text.contains("arena_opening_plies_min = 6\n"));
        assert!(text.contains("arena_opening_plies_max = 10\n"));
        assert!(text.contains("gumbel_actions = 24\n"));
        assert!(text.contains("gumbel_scale = 1.0\n"));
        assert!(text.contains("gumbel_value_scale = 0.1\n"));
        assert!(text.contains("gumbel_maxvisit_init = 50.0\n"));
        assert!(!text.contains("search"));
        assert!(!text.contains("temperature"));
        assert!(!text.contains("cpuct"));
        assert!(!text.contains("fpu"));
        assert!(!text.contains("dirichlet"));
        assert!(!text.contains("deblunder"));
        assert!(!text.contains("search_algorithm"));
        assert!(!text.contains("arena_pikafish"));
        assert!(!text.contains("arena_eval_fens"));
        assert!(!text.contains("000000047"));
        assert!(!text.contains("000000023"));

        let parsed = AzLoopFileConfig::parse(&text);
        assert_eq!(parsed.model_path, "model.safetensors");
        assert!((parsed.lr - 0.001).abs() < 1e-9);
        assert!((parsed.td_lambda - 0.95).abs() < 1e-6);
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
