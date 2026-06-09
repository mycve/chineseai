use chineseai::az::AzNnueArch;
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
    pub temperature_endgame: f32,
    pub temperature_decay_delay_plies: usize,
    pub temperature_decay_plies: usize,
    pub temperature_cutoff_plies: usize,
    pub temperature_value_cutoff: f32,
    pub temperature_visit_offset: f32,
    pub cpuct: f32,
    pub cpuct_at_root: f32,
    pub root_dirichlet_alpha: f32,
    pub root_exploration_fraction: f32,
    pub root_exploration_plies: usize,
    pub fpu_value: f32,
    pub fpu_value_at_root: f32,
    pub policy_softmax_temp: f32,
    pub opening_fens_path: String,
    pub resign_percentage: f32,
    pub resign_playthrough: f32,
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
        Self {
            model_path: "model.safetensors".into(),
            simulations: 800,
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
            temperature_start: 0.9,
            temperature_endgame: 0.6,
            temperature_decay_delay_plies: 20,
            temperature_decay_plies: 60,
            temperature_cutoff_plies: 40,
            temperature_value_cutoff: 0.15,
            temperature_visit_offset: -0.8,
            cpuct: 0.65,
            cpuct_at_root: 2.53,
            root_dirichlet_alpha: 0.12,
            root_exploration_fraction: 0.1,
            root_exploration_plies: 60,
            fpu_value: 0.23,
            fpu_value_at_root: 1.0,
            policy_softmax_temp: 1.45,
            opening_fens_path: String::new(),
            resign_percentage: 1.0,
            resign_playthrough: 20.0,
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
    pub temperature_endgame: f32,
    pub temperature_decay_delay_plies: usize,
    pub temperature_decay_plies: usize,
    pub temperature_cutoff_plies: usize,
    pub temperature_value_cutoff: f32,
    pub temperature_visit_offset: f32,
    pub cpuct: f32,
    pub cpuct_at_root: f32,
    pub root_dirichlet_alpha: f32,
    pub root_exploration_fraction: f32,
    pub root_exploration_plies: usize,
    pub fpu_value: f32,
    pub fpu_value_at_root: f32,
    pub policy_softmax_temp: f32,
    pub opening_fens_path: String,
    pub resign_percentage: f32,
    pub resign_playthrough: f32,
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
            temperature_endgame: config.temperature_endgame,
            temperature_decay_delay_plies: config.temperature_decay_delay_plies,
            temperature_decay_plies: config.temperature_decay_plies,
            temperature_cutoff_plies: config.temperature_cutoff_plies,
            temperature_value_cutoff: config.temperature_value_cutoff,
            temperature_visit_offset: config.temperature_visit_offset,
            cpuct: config.cpuct,
            cpuct_at_root: config.cpuct_at_root,
            root_dirichlet_alpha: config.root_dirichlet_alpha,
            root_exploration_fraction: config.root_exploration_fraction,
            root_exploration_plies: config.root_exploration_plies,
            fpu_value: config.fpu_value,
            fpu_value_at_root: config.fpu_value_at_root,
            policy_softmax_temp: config.policy_softmax_temp,
            opening_fens_path: config.opening_fens_path.clone(),
            resign_percentage: config.resign_percentage,
            resign_playthrough: config.resign_playthrough,
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
            temperature_endgame: config.temperature_endgame,
            temperature_decay_delay_plies: config.temperature_decay_delay_plies,
            temperature_decay_plies: config.temperature_decay_plies,
            temperature_cutoff_plies: config.temperature_cutoff_plies,
            temperature_value_cutoff: config.temperature_value_cutoff,
            temperature_visit_offset: config.temperature_visit_offset,
            cpuct: config.cpuct,
            cpuct_at_root: config.cpuct_at_root,
            root_dirichlet_alpha: config.root_dirichlet_alpha,
            root_exploration_fraction: config.root_exploration_fraction,
            root_exploration_plies: config.root_exploration_plies,
            fpu_value: config.fpu_value,
            fpu_value_at_root: config.fpu_value_at_root,
            policy_softmax_temp: config.policy_softmax_temp,
            opening_fens_path: config.opening_fens_path,
            resign_percentage: config.resign_percentage,
            resign_playthrough: config.resign_playthrough,
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
        line!("temperature_endgame", f(self.temperature_endgame));
        line!(
            "temperature_decay_delay_plies",
            self.temperature_decay_delay_plies
        );
        line!("temperature_decay_plies", self.temperature_decay_plies);
        line!("temperature_cutoff_plies", self.temperature_cutoff_plies);
        line!("temperature_value_cutoff", f(self.temperature_value_cutoff));
        line!("temperature_visit_offset", f(self.temperature_visit_offset));
        line!("cpuct", f(self.cpuct));
        line!("cpuct_at_root", f(self.cpuct_at_root));
        line!("root_dirichlet_alpha", f(self.root_dirichlet_alpha));
        line!(
            "root_exploration_fraction",
            f(self.root_exploration_fraction)
        );
        line!("root_exploration_plies", self.root_exploration_plies);
        line!("fpu_value", f(self.fpu_value));
        line!("fpu_value_at_root", f(self.fpu_value_at_root));
        line!("policy_softmax_temp", f(self.policy_softmax_temp));
        line!("opening_fens_path", q(&self.opening_fens_path));
        line!("resign_percentage", f(self.resign_percentage));
        line!("resign_playthrough", f(self.resign_playthrough));
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
        self.temperature_endgame = self.temperature_endgame.max(0.0);
        self.temperature_decay_delay_plies =
            self.temperature_decay_delay_plies.min(self.max_plies);
        self.temperature_decay_plies = self.temperature_decay_plies.min(self.max_plies);
        self.temperature_cutoff_plies = self.temperature_cutoff_plies.min(self.max_plies);
        self.temperature_value_cutoff = self.temperature_value_cutoff.max(0.0);
        self.cpuct = self.cpuct.max(0.0);
        self.cpuct_at_root = self.cpuct_at_root.max(0.0);
        self.root_dirichlet_alpha = self.root_dirichlet_alpha.max(0.0);
        self.root_exploration_fraction = self.root_exploration_fraction.clamp(0.0, 1.0);
        self.root_exploration_plies = self.root_exploration_plies.min(self.max_plies);
        self.fpu_value = self.fpu_value.max(0.0);
        self.fpu_value_at_root = self.fpu_value_at_root.clamp(-1.0, 1.0);
        self.policy_softmax_temp = self.policy_softmax_temp.max(1e-3);
        self.resign_percentage = self.resign_percentage.clamp(0.0, 100.0);
        self.resign_playthrough = self.resign_playthrough.clamp(0.0, 100.0);
        self.train_warmup_samples = self.train_warmup_samples.max(1);
        self.train_samples_per_update = self.train_samples_per_update.max(1);
        self.train_epochs_per_update = self.train_epochs_per_update.max(1);
        self.arena_cpuct = self.arena_cpuct.max(0.0);
        self.mirror_probability = self.mirror_probability.clamp(0.0, 1.0);
        self.value_td_lambda = self.value_td_lambda.clamp(0.0, 1.0);
        self.train_value_weight = self.train_value_weight.max(0.0);
        self.train_policy_weight = self.train_policy_weight.max(0.0);
        self.max_checkpoints = self.max_checkpoints.max(1);
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
        assert!(text.contains("temperature_start = 0.9\n"));
        assert!(text.contains("temperature_endgame = 0.6\n"));
        assert!(text.contains("temperature_decay_delay_plies = 20\n"));
        assert!(text.contains("temperature_cutoff_plies = 40\n"));
        assert!(text.contains("temperature_value_cutoff = 0.15\n"));
        assert!(text.contains("temperature_visit_offset = -0.8\n"));
        assert!(text.contains("cpuct = 0.65\n"));
        assert!(text.contains("cpuct_at_root = 2.53\n"));
        assert!(text.contains("fpu_value = 0.23\n"));
        assert!(text.contains("fpu_value_at_root = 1.0\n"));
        assert!(text.contains("policy_softmax_temp = 1.45\n"));
        assert!(text.contains("opening_fens_path = \"\"\n"));
        assert!(text.contains("resign_percentage = 1.0\n"));
        assert!(text.contains("resign_playthrough = 20.0\n"));
        assert!(text.contains("value_td_lambda = 0.95\n"));
        assert!(!text.contains("gumbel"));
        assert!(!text.contains("search_algorithm"));
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
