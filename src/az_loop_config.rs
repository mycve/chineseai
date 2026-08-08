use chineseai::az::AzNnueArch;
use chineseai::version::AZ_LOOP_CONFIG_FORMAT_VERSION;
use serde::{Deserialize, Serialize};
use std::{fmt::Write, fs, path::Path};

pub const DEFAULT_AZ_LOOP_CONFIG: &str = "chineseai.azloop.toml";

fn system_physical_cores() -> usize {
    let physical = num_cpus::get_physical();
    if physical > 0 {
        physical
    } else {
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AzLoopFileConfig {
    pub format_version: u32,
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
    pub value_target_search_q_mix: f32,
    pub opening_fens_path: String,
    pub resign_percentage: f32,
    pub resign_playthrough: f32,
    pub replay_capacity: usize,
    pub replay_recent_sample_fraction: f32,
    pub replay_recent_games: u32,
    pub train_warmup_samples: usize,
    pub train_samples_per_update: usize,
    pub train_epochs_per_update: usize,
    pub mirror_probability: f32,
    pub train_value_weight: f32,
    pub train_policy_weight: f32,
    pub checkpoint_interval: usize,
    pub checkpoint_dir: String,
    pub max_checkpoints: usize,
    pub arena_interval: usize,
    pub arena_simulations: usize,
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
            format_version: AZ_LOOP_CONFIG_FORMAT_VERSION,
            model_path: "model.safetensors".into(),
            simulations: 2400,
            selfplay_samples_per_update: 120000,
            lr: 0.001,
            lr_min: 0.0003,
            lr_decay_start_update: 100,
            lr_decay_interval: 200,
            lr_decay_factor: 0.97,
            batch_size: 1024,
            max_plies: 200,
            hidden_size: 128,
            seed: 20260420,
            workers: 0,
            temperature_start: 0.9,
            temperature_endgame: 0.30,
            temperature_decay_delay_plies: 30,
            temperature_decay_plies: 60,
            temperature_value_cutoff: 0.12,
            temperature_visit_offset: -0.8,
            cpuct: 0.65,
            cpuct_at_root: 1.5,
            cpuct_base: 19652.0,
            cpuct_factor: 1.5,
            cpuct_base_at_root: 19652.0,
            cpuct_factor_at_root: 1.5,
            root_dirichlet_alpha: 0.12,
            root_exploration_fraction: 0.10,
            fpu_value: 0.0,
            fpu_value_at_root: 1.0,
            draw_score: 0.0,
            moves_left_max_effect: 0.25,
            moves_left_slope: 0.004,
            moves_left_threshold: 0.7,
            moves_left_constant_factor: 0.05,
            moves_left_scaled_factor: 0.20,
            moves_left_quadratic_factor: 0.75,
            policy_softmax_temp: 1.45,
            value_target_search_q_mix: chineseai::az::VALUE_TARGET_SEARCH_Q_MIX,
            opening_fens_path: String::new(),
            resign_percentage: 1.0,
            resign_playthrough: 20.0,
            replay_capacity: 1000000,
            replay_recent_sample_fraction: 0.35,
            replay_recent_games: 5000,
            train_warmup_samples: 240000,
            train_samples_per_update: 240000,
            train_epochs_per_update: 1,
            mirror_probability: 0.3,
            train_value_weight: 1.0,
            train_policy_weight: 1.0,
            checkpoint_interval: 20,
            checkpoint_dir: "checkpoints".into(),
            max_checkpoints: 50,
            arena_interval: 20,
            arena_simulations: 4000,
            arena_cpuct: 1.5,
            arena_promotion_rate: 0.50,
            arena_promotion_confidence_z: 1.28,
            arena_processes: 128,
            arena_opening_book: "opening.obk".into(),
            arena_opening_positions: 300,
            arena_opening_plies_min: 6,
            arena_opening_plies_max: 10,
            pikafish_label_eval_sqlite: "eval/pikafish-selfplay-5000-d20.sqlite".into(),
            pikafish_label_eval_interval: 20,
            pikafish_label_eval_limit: 1000,
            pikafish_label_eval_simulations: 3000,
            pikafish_label_eval_cpuct: 1.5,
            tensorboard_logdir: "runs/chineseai".into(),
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
        line!("format_version", AZ_LOOP_CONFIG_FORMAT_VERSION);
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
        line!(
            "value_target_search_q_mix",
            f(self.value_target_search_q_mix)
        );
        line!("opening_fens_path", q(&self.opening_fens_path));
        line!("resign_percentage", f(self.resign_percentage));
        line!("resign_playthrough", f(self.resign_playthrough));
        line!("replay_capacity", self.replay_capacity);
        line!(
            "replay_recent_sample_fraction",
            f(self.replay_recent_sample_fraction)
        );
        line!("replay_recent_games", self.replay_recent_games);
        line!("train_warmup_samples", self.train_warmup_samples);
        line!("train_samples_per_update", self.train_samples_per_update);
        line!("train_epochs_per_update", self.train_epochs_per_update);
        line!("mirror_probability", f(self.mirror_probability));
        line!("train_value_weight", f(self.train_value_weight));
        line!("train_policy_weight", f(self.train_policy_weight));
        line!("checkpoint_interval", self.checkpoint_interval);
        line!("checkpoint_dir", q(&self.checkpoint_dir));
        line!("max_checkpoints", self.max_checkpoints);
        line!("arena_interval", self.arena_interval);
        line!("arena_simulations", self.arena_simulations);
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
        let config = toml::from_str::<AzLoopFileConfig>(text)
            .unwrap_or_else(|err| panic!("invalid az-loop TOML config: {err}"));
        if config.format_version != AZ_LOOP_CONFIG_FORMAT_VERSION {
            panic!(
                "unsupported az-loop config format {}; expected {}",
                config.format_version, AZ_LOOP_CONFIG_FORMAT_VERSION
            );
        }
        config.normalize()
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
        if self.workers == 0 {
            self.workers = system_physical_cores();
        }
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
        self.value_target_search_q_mix = self.value_target_search_q_mix.clamp(0.0, 1.0);
        self.resign_percentage = self.resign_percentage.clamp(0.0, 100.0);
        self.resign_playthrough = self.resign_playthrough.clamp(0.0, 100.0);
        self.replay_recent_sample_fraction = self.replay_recent_sample_fraction.clamp(0.0, 1.0);
        self.replay_recent_games = self.replay_recent_games.max(1);
        self.train_warmup_samples = self.train_warmup_samples.max(1);
        self.train_samples_per_update = self.train_samples_per_update.max(1);
        self.train_epochs_per_update = self.train_epochs_per_update.max(1);
        self.arena_cpuct = self.arena_cpuct.max(0.0);
        self.mirror_probability = self.mirror_probability.clamp(0.0, 1.0);
        self.train_value_weight = self.train_value_weight.max(0.0);
        self.train_policy_weight = self.train_policy_weight.max(0.0);
        self.max_checkpoints = self.max_checkpoints.max(1);
        self.arena_processes = self.arena_processes.max(1);
        self.arena_promotion_rate = self.arena_promotion_rate.clamp(0.0, 1.0);
        self.arena_promotion_confidence_z = self.arena_promotion_confidence_z.max(0.0);
        self.arena_simulations = self.arena_simulations.max(1);
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

        assert!(text.starts_with("format_version = 4\n"));
        assert!(text.contains("lr = 0.001\n"));
        assert!(text.contains("lr_min = 0.0003\n"));
        assert!(text.contains("temperature_start = 0.9\n"));
        assert!(text.contains("temperature_endgame = 0.3\n"));
        assert!(text.contains("temperature_decay_delay_plies = 30\n"));
        assert!(text.contains("temperature_decay_plies = 60\n"));
        assert!(!text.contains("temperature_cutoff_plies"));
        assert!(text.contains("temperature_value_cutoff = 0.12\n"));
        assert!(text.contains("temperature_visit_offset = -0.8\n"));
        assert!(text.contains("cpuct = 0.65\n"));
        assert!(text.contains("cpuct_at_root = 1.5\n"));
        assert!(text.contains("cpuct_base = 19652.0\n"));
        assert!(text.contains("cpuct_factor = 1.5\n"));
        assert!(text.contains("cpuct_base_at_root = 19652.0\n"));
        assert!(text.contains("cpuct_factor_at_root = 1.5\n"));
        assert!(text.contains("root_dirichlet_alpha = 0.12\n"));
        assert!(text.contains("root_exploration_fraction = 0.1\n"));
        assert!(text.contains("fpu_value = 0.0\n"));
        assert!(text.contains("fpu_value_at_root = 1.0\n"));
        assert!(text.contains("draw_score = 0.0\n"));
        assert!(text.contains("moves_left_max_effect = 0.25\n"));
        assert!(text.contains("moves_left_slope = 0.004\n"));
        assert!(text.contains("moves_left_threshold = 0.7\n"));
        assert!(text.contains("moves_left_constant_factor = 0.05\n"));
        assert!(text.contains("moves_left_scaled_factor = 0.2\n"));
        assert!(text.contains("moves_left_quadratic_factor = 0.75\n"));
        assert!(text.contains("policy_softmax_temp = 1.45\n"));
        assert!(text.contains("value_target_search_q_mix = 0.4\n"));
        assert!(text.contains("opening_fens_path = \"\"\n"));
        assert!(text.contains("resign_percentage = 1.0\n"));
        assert!(text.contains("resign_playthrough = 20.0\n"));
        assert!(text.contains("simulations = 2400\n"));
        assert!(!text.contains("low_simulations"));
        assert!(!text.contains("low_simulation_probability"));
        assert!(!text.contains("low_simulation_policy_weight"));
        assert!(!text.contains("high_simulations"));
        assert!(!text.contains("high_simulation_probability"));
        assert!(!text.contains("high_simulation_start_plies"));
        assert!(text.contains("selfplay_samples_per_update = 120000\n"));
        assert!(text.contains("workers = 0\n"));
        assert!(text.contains("batch_size = 1024\n"));
        assert!(text.contains("max_plies = 200\n"));
        assert!(text.contains("hidden_size = 128\n"));
        assert!(text.contains("replay_capacity = 1000000\n"));
        assert!(text.contains("train_samples_per_update = 240000\n"));
        assert!(text.contains("train_epochs_per_update = 1\n"));
        assert!(text.contains("replay_recent_games = 5000\n"));
        assert!(text.contains("arena_processes = 128\n"));
        assert!(text.contains("arena_opening_book = \"opening.obk\"\n"));
        assert!(text.contains("arena_opening_positions = 300\n"));
        assert!(text.contains("arena_opening_plies_min = 6\n"));
        assert!(text.contains("arena_opening_plies_max = 10\n"));
        assert!(text.contains("arena_interval = 20\n"));
        assert!(text.contains("arena_simulations = 4000\n"));
        assert!(
            text.contains(
                "pikafish_label_eval_sqlite = \"eval/pikafish-selfplay-5000-d20.sqlite\"\n"
            )
        );
        assert!(text.contains("pikafish_label_eval_interval = 20\n"));
        assert!(text.contains("pikafish_label_eval_limit = 1000\n"));
        assert!(text.contains("pikafish_label_eval_simulations = 3000\n"));
        assert!(text.contains("pikafish_label_eval_cpuct = 1.5\n"));
        assert!(!text.contains("root_exploration_plies"));
        assert!(!text.contains("search_algorithm"));
        assert!(!text.contains("arena_pikafish"));
        assert!(!text.contains("arena_eval_fens"));
        assert!(!text.contains("000000047"));
        assert!(!text.contains("000000023"));

        let parsed = AzLoopFileConfig::parse(&text);
        assert_eq!(parsed.model_path, "model.safetensors");
        assert!((parsed.lr - 0.001).abs() < 1e-9);
        assert_eq!(parsed.arena_interval, 20);
        assert_eq!(parsed.pikafish_label_eval_interval, 20);
    }

    #[test]
    fn removed_config_names_are_rejected() {
        for removed in [
            "replay_recent_window_updates = 5000\n",
            "deblunder_q_gap = 0.05\n",
            "low_simulations = 2000\n",
            "low_simulation_probability = 0.2\n",
            "low_simulation_policy_weight = 0.5\n",
            "high_simulations = 20000\n",
            "high_simulation_probability = 0.1\n",
            "high_simulation_start_plies = 40\n",
            "arena_pikafish_exe = \"./pikafish\"\n",
            "arena_pikafish_depth = 10\n",
            "arena_pikafish_games = 20\n",
        ] {
            let error = toml::from_str::<AzLoopFileConfig>(removed)
                .expect_err("removed config keys must not be accepted");
            let key = removed.split_once(' ').unwrap().0;
            assert!(error.to_string().contains(key));
        }
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
