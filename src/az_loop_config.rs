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
    pub policy_reanalysis_simulations: usize,
    pub selfplay_samples_per_update: usize,
    pub lr: f32,
    pub lr_min: f32,
    pub lr_decay_start_update: usize,
    pub lr_decay_interval: usize,
    pub lr_decay_factor: f32,
    pub batch_size: usize,
    pub max_plies: usize,
    pub sixty_move_rule: bool,
    pub rule60_max_ply: u16,
    pub hidden_size: usize,
    pub seed: u64,
    pub workers: usize,
    pub temperature_start: f32,
    pub temperature_endgame: f32,
    pub temperature_decay_delay_plies: usize,
    pub temperature_decay_plies: usize,
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
    pub policy_softmax_temp: f32,
    pub value_td_lambda: f32,
    pub opening_start_fraction: f32,
    pub opening_reservoir_capacity: usize,
    pub opening_snapshot_path: String,
    pub midgame_start_fraction: f32,
    pub midgame_reservoir_capacity: usize,
    pub midgame_snapshot_path: String,
    pub actor_publish_interval_updates: usize,
    pub replay_capacity: usize,
    pub replay_recent_sample_fraction: f32,
    pub replay_recent_games: u32,
    pub replay_phase_0_29_fraction: f32,
    pub replay_phase_30_59_fraction: f32,
    pub replay_phase_60_99_fraction: f32,
    pub replay_phase_100_139_fraction: f32,
    pub replay_phase_140_plus_fraction: f32,
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
    pub arena_cpuct_at_root: f32,
    pub arena_policy_softmax_temp: f32,
    pub arena_promotion_rate: f32,
    pub arena_promotion_confidence_z: f32,
    pub arena_processes: usize,
    pub arena_opening_book: String,
    pub arena_opening_positions: usize,
    pub arena_opening_plies_min: usize,
    pub arena_opening_plies_max: usize,
    pub arena_random_positions: usize,
    pub arena_random_plies_min: usize,
    pub arena_random_plies_max: usize,
    pub pikafish_label_eval_sqlite: String,
    pub pikafish_label_eval_interval: usize,
    pub pikafish_label_eval_limit: usize,
    pub pikafish_label_eval_simulations: usize,
    pub pikafish_label_eval_cpuct: f32,
    pub pikafish_label_eval_cpuct_at_root: f32,
    pub pikafish_label_eval_policy_softmax_temp: f32,
    pub tensorboard_logdir: String,
}

impl Default for AzLoopFileConfig {
    fn default() -> Self {
        Self {
            format_version: AZ_LOOP_CONFIG_FORMAT_VERSION,
            model_path: "model.safetensors".into(),
            simulations: 400,
            policy_reanalysis_simulations: 0,
            selfplay_samples_per_update: 120000,
            lr: 0.0004,
            lr_min: 0.00001,
            lr_decay_start_update: 100,
            lr_decay_interval: 200,
            lr_decay_factor: 0.97,
            batch_size: 1024,
            max_plies: 200,
            sixty_move_rule: true,
            rule60_max_ply: 120,
            hidden_size: 128,
            seed: 20260420,
            workers: 0,
            temperature_start: 0.9,
            temperature_endgame: 0.10,
            temperature_decay_delay_plies: 8,
            temperature_decay_plies: 14,
            cpuct: 0.9,
            cpuct_at_root: 2.0,
            cpuct_base: 19652.0,
            cpuct_factor: 1.5,
            cpuct_base_at_root: 19652.0,
            cpuct_factor_at_root: 1.5,
            root_dirichlet_alpha: 0.12,
            root_exploration_fraction: 0.08,
            fpu_value: 0.20,
            fpu_value_at_root: 0.10,
            draw_score: 0.0,
            policy_softmax_temp: 1.15,
            value_td_lambda: chineseai::az::DEFAULT_VALUE_TD_LAMBDA,
            opening_start_fraction: 0.30,
            opening_reservoir_capacity: 50_000,
            opening_snapshot_path: "opening-pool.lz4".into(),
            midgame_start_fraction: 0.40,
            midgame_reservoir_capacity: 50_000,
            midgame_snapshot_path: "midgame-pool.lz4".into(),
            actor_publish_interval_updates: 5,
            replay_capacity: 2400000,
            replay_recent_sample_fraction: 0.35,
            replay_recent_games: 7500,
            replay_phase_0_29_fraction: 0.30,
            replay_phase_30_59_fraction: 0.25,
            replay_phase_60_99_fraction: 0.25,
            replay_phase_100_139_fraction: 0.15,
            replay_phase_140_plus_fraction: 0.05,
            train_warmup_samples: 600000,
            train_samples_per_update: 120000,
            train_epochs_per_update: 1,
            mirror_probability: 0.5,
            train_value_weight: 1.0,
            train_policy_weight: 1.0,
            checkpoint_interval: 20,
            checkpoint_dir: "checkpoints".into(),
            max_checkpoints: 50,
            arena_interval: 10,
            arena_simulations: 400,
            arena_cpuct: 0.9,
            arena_cpuct_at_root: 2.0,
            arena_policy_softmax_temp: 1.15,
            arena_promotion_rate: 0.50,
            arena_promotion_confidence_z: 1.96,
            arena_processes: 128,
            arena_opening_book: "opening.obk".into(),
            arena_opening_positions: 1000,
            arena_opening_plies_min: 8,
            arena_opening_plies_max: 10,
            arena_random_positions: 1000,
            arena_random_plies_min: 6,
            arena_random_plies_max: 12,
            pikafish_label_eval_sqlite: "eval/pikafish-selfplay-5000-d20.sqlite".into(),
            pikafish_label_eval_interval: 10,
            pikafish_label_eval_limit: 1000,
            pikafish_label_eval_simulations: 6000,
            pikafish_label_eval_cpuct: 0.9,
            pikafish_label_eval_cpuct_at_root: 1.5,
            pikafish_label_eval_policy_softmax_temp: 1.15,
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
            "policy_reanalysis_simulations",
            self.policy_reanalysis_simulations
        );
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
        line!("sixty_move_rule", self.sixty_move_rule);
        line!("rule60_max_ply", self.rule60_max_ply);
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
        line!("policy_softmax_temp", f(self.policy_softmax_temp));
        line!("value_td_lambda", f(self.value_td_lambda));
        line!("opening_start_fraction", f(self.opening_start_fraction));
        line!(
            "opening_reservoir_capacity",
            self.opening_reservoir_capacity
        );
        line!("opening_snapshot_path", q(&self.opening_snapshot_path));
        line!("midgame_start_fraction", f(self.midgame_start_fraction));
        line!(
            "midgame_reservoir_capacity",
            self.midgame_reservoir_capacity
        );
        line!("midgame_snapshot_path", q(&self.midgame_snapshot_path));
        line!(
            "actor_publish_interval_updates",
            self.actor_publish_interval_updates
        );
        line!("replay_capacity", self.replay_capacity);
        line!(
            "replay_recent_sample_fraction",
            f(self.replay_recent_sample_fraction)
        );
        line!("replay_recent_games", self.replay_recent_games);
        line!(
            "replay_phase_0_29_fraction",
            f(self.replay_phase_0_29_fraction)
        );
        line!(
            "replay_phase_30_59_fraction",
            f(self.replay_phase_30_59_fraction)
        );
        line!(
            "replay_phase_60_99_fraction",
            f(self.replay_phase_60_99_fraction)
        );
        line!(
            "replay_phase_100_139_fraction",
            f(self.replay_phase_100_139_fraction)
        );
        line!(
            "replay_phase_140_plus_fraction",
            f(self.replay_phase_140_plus_fraction)
        );
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
        line!("arena_cpuct_at_root", f(self.arena_cpuct_at_root));
        line!(
            "arena_policy_softmax_temp",
            f(self.arena_policy_softmax_temp)
        );
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
        line!("arena_random_positions", self.arena_random_positions);
        line!("arena_random_plies_min", self.arena_random_plies_min);
        line!("arena_random_plies_max", self.arena_random_plies_max);
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
        line!(
            "pikafish_label_eval_cpuct_at_root",
            f(self.pikafish_label_eval_cpuct_at_root)
        );
        line!(
            "pikafish_label_eval_policy_softmax_temp",
            f(self.pikafish_label_eval_policy_softmax_temp)
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
        if self.policy_reanalysis_simulations <= self.simulations {
            self.policy_reanalysis_simulations = 0;
        }
        self.selfplay_samples_per_update = self.selfplay_samples_per_update.max(1);
        self.lr = self.lr.max(0.0);
        self.lr_min = self.lr_min.max(0.0).min(self.lr);
        self.lr_decay_interval = self.lr_decay_interval.max(1);
        self.lr_decay_factor = self.lr_decay_factor.clamp(0.0, 1.0);
        self.batch_size = self.batch_size.max(1);
        self.max_plies = self.max_plies.max(1);
        self.rule60_max_ply = self.rule60_max_ply.clamp(1, 150);
        self.hidden_size = self.hidden_size.max(1);
        if self.workers == 0 {
            self.workers = system_physical_cores();
        }
        self.temperature_start = self.temperature_start.max(0.0);
        self.temperature_endgame = self.temperature_endgame.max(0.0);
        self.temperature_decay_delay_plies = self.temperature_decay_delay_plies.min(self.max_plies);
        self.temperature_decay_plies = self.temperature_decay_plies.min(self.max_plies);
        self.cpuct = self.cpuct.max(0.0);
        self.cpuct_at_root = self.cpuct_at_root.max(0.0);
        self.cpuct_base = self.cpuct_base.max(1.0);
        self.cpuct_factor = self.cpuct_factor.max(0.0);
        self.cpuct_base_at_root = self.cpuct_base_at_root.max(1.0);
        self.cpuct_factor_at_root = self.cpuct_factor_at_root.max(0.0);
        self.root_dirichlet_alpha = self.root_dirichlet_alpha.max(0.0);
        self.root_exploration_fraction = self.root_exploration_fraction.clamp(0.0, 1.0);
        self.fpu_value = self.fpu_value.max(0.0);
        self.fpu_value_at_root = self.fpu_value_at_root.max(0.0);
        self.draw_score = self.draw_score.clamp(-1.0, 1.0);
        self.policy_softmax_temp = self.policy_softmax_temp.max(1e-3);
        self.midgame_start_fraction = self.midgame_start_fraction.clamp(0.0, 1.0);
        self.opening_start_fraction = self.opening_start_fraction.clamp(0.0, 1.0);
        let start_fraction = 1.0 - self.opening_start_fraction - self.midgame_start_fraction;
        assert!(
            start_fraction >= -1.0e-6,
            "opening_start_fraction ({}) + midgame_start_fraction ({}) must not exceed 1; the remainder is the start-position share",
            self.opening_start_fraction,
            self.midgame_start_fraction
        );
        self.value_td_lambda = self.value_td_lambda.clamp(0.0, 1.0);
        self.replay_recent_sample_fraction = self.replay_recent_sample_fraction.clamp(0.0, 1.0);
        self.replay_recent_games = self.replay_recent_games.max(1);
        self.actor_publish_interval_updates = self.actor_publish_interval_updates.max(1);
        let mut replay_phase_fractions = [
            self.replay_phase_0_29_fraction.max(0.0),
            self.replay_phase_30_59_fraction.max(0.0),
            self.replay_phase_60_99_fraction.max(0.0),
            self.replay_phase_100_139_fraction.max(0.0),
            self.replay_phase_140_plus_fraction.max(0.0),
        ];
        let replay_source_total = replay_phase_fractions.iter().sum::<f32>();
        assert!(
            replay_source_total > 0.0,
            "replay phase fractions must have positive total"
        );
        for fraction in &mut replay_phase_fractions {
            *fraction /= replay_source_total;
        }
        self.replay_phase_0_29_fraction = replay_phase_fractions[0];
        self.replay_phase_30_59_fraction = replay_phase_fractions[1];
        self.replay_phase_60_99_fraction = replay_phase_fractions[2];
        self.replay_phase_100_139_fraction = replay_phase_fractions[3];
        self.replay_phase_140_plus_fraction = replay_phase_fractions[4];
        self.train_warmup_samples = self.train_warmup_samples.max(1);
        self.train_samples_per_update = self.train_samples_per_update.max(1);
        self.train_epochs_per_update = self.train_epochs_per_update.max(1);
        self.arena_cpuct = self.arena_cpuct.max(0.0);
        self.arena_cpuct_at_root = self.arena_cpuct_at_root.max(0.0);
        self.arena_policy_softmax_temp = self.arena_policy_softmax_temp.max(1e-3);
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
        self.pikafish_label_eval_cpuct_at_root = self.pikafish_label_eval_cpuct_at_root.max(0.0);
        self.pikafish_label_eval_policy_softmax_temp =
            self.pikafish_label_eval_policy_softmax_temp.max(1e-3);
        if self.arena_opening_plies_min > self.arena_opening_plies_max {
            std::mem::swap(
                &mut self.arena_opening_plies_min,
                &mut self.arena_opening_plies_max,
            );
        }
        if self.arena_random_plies_min > self.arena_random_plies_max {
            std::mem::swap(
                &mut self.arena_random_plies_min,
                &mut self.arena_random_plies_max,
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
        let config = AzLoopFileConfig::default();
        let text = config.to_file_text();

        assert!(text.starts_with("format_version = 23\n"));
        assert!(text.contains("lr = 0.0004\n"));
        assert!(text.contains("lr_min = 0.00001\n"));
        assert!(text.contains("temperature_start = 0.9\n"));
        assert!(text.contains("sixty_move_rule = true\n"));
        assert!(text.contains("rule60_max_ply = 120\n"));
        assert!(text.contains("temperature_endgame = 0.1\n"));
        assert!(text.contains("temperature_decay_delay_plies = 8\n"));
        assert!(text.contains("temperature_decay_plies = 14\n"));
        assert!(!text.contains("temperature_cutoff_plies"));
        assert!(text.contains("cpuct = 0.9\n"));
        assert!(text.contains("cpuct_at_root = 2.0\n"));
        assert!(text.contains("cpuct_base = 19652.0\n"));
        assert!(text.contains("cpuct_factor = 1.5\n"));
        assert!(text.contains("cpuct_base_at_root = 19652.0\n"));
        assert!(text.contains("cpuct_factor_at_root = 1.5\n"));
        assert!(text.contains("root_dirichlet_alpha = 0.12\n"));
        assert!(text.contains("root_exploration_fraction = 0.08\n"));
        assert!(text.contains("fpu_value = 0.2\n"));
        assert!(text.contains("fpu_value_at_root = 0.1\n"));
        assert!(text.contains("draw_score = 0.0\n"));
        assert!(text.contains("policy_softmax_temp = 1.15\n"));
        assert!(text.contains("value_td_lambda = 1.0\n"));
        assert!(!text.contains("value_target_search_q_mix"));
        assert!(text.contains("opening_start_fraction = 0.3\n"));
        assert!(text.contains("opening_reservoir_capacity = 50000\n"));
        assert!(text.contains("opening_snapshot_path = \"opening-pool.lz4\"\n"));
        assert!(text.contains("midgame_start_fraction = 0.4\n"));
        assert!(text.contains("midgame_reservoir_capacity = 50000\n"));
        assert!(text.contains("midgame_snapshot_path = \"midgame-pool.lz4\"\n"));
        assert!(text.contains("actor_publish_interval_updates = 5\n"));
        assert!(text.contains("simulations = 400\n"));
        assert!(text.contains("policy_reanalysis_simulations = 0\n"));
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
        assert!(text.contains("replay_capacity = 2400000\n"));
        assert!(text.contains("train_samples_per_update = 120000\n"));
        assert!(text.contains("train_warmup_samples = 600000\n"));
        assert!(text.contains("train_epochs_per_update = 1\n"));
        assert!(text.contains("replay_recent_games = 7500\n"));
        assert_eq!(
            config.replay_capacity / config.selfplay_samples_per_update,
            20
        );
        assert_eq!(
            config.train_warmup_samples / config.selfplay_samples_per_update,
            5
        );
        assert_eq!(
            config.train_samples_per_update / config.selfplay_samples_per_update,
            1
        );
        assert!(text.contains("replay_phase_0_29_fraction = 0.3\n"));
        assert!(text.contains("replay_phase_30_59_fraction = 0.25\n"));
        assert!(text.contains("replay_phase_60_99_fraction = 0.25\n"));
        assert!(text.contains("replay_phase_100_139_fraction = 0.15\n"));
        assert!(text.contains("replay_phase_140_plus_fraction = 0.05\n"));
        assert!(text.contains("mirror_probability = 0.5\n"));
        assert!(text.contains("arena_processes = 128\n"));
        assert!(text.contains("arena_opening_book = \"opening.obk\"\n"));
        assert!(text.contains("arena_opening_positions = 1000\n"));
        assert!(text.contains("arena_opening_plies_min = 8\n"));
        assert!(text.contains("arena_opening_plies_max = 10\n"));
        assert!(text.contains("arena_random_positions = 1000\n"));
        assert!(text.contains("arena_random_plies_min = 6\n"));
        assert!(text.contains("arena_random_plies_max = 12\n"));
        assert!(text.contains("arena_interval = 10\n"));
        assert!(text.contains("arena_simulations = 400\n"));
        assert!(text.contains("arena_promotion_rate = 0.5\n"));
        assert!(text.contains("arena_promotion_confidence_z = 1.96\n"));
        assert!(text.contains("arena_cpuct = 0.9\n"));
        assert!(text.contains("arena_cpuct_at_root = 2.0\n"));
        assert!(text.contains("arena_policy_softmax_temp = 1.15\n"));
        assert!(
            text.contains(
                "pikafish_label_eval_sqlite = \"eval/pikafish-selfplay-5000-d20.sqlite\"\n"
            )
        );
        assert!(text.contains("pikafish_label_eval_interval = 10\n"));
        assert!(text.contains("pikafish_label_eval_limit = 1000\n"));
        assert!(text.contains("pikafish_label_eval_simulations = 6000\n"));
        assert!(text.contains("pikafish_label_eval_cpuct = 0.9\n"));
        assert!(text.contains("pikafish_label_eval_cpuct_at_root = 1.5\n"));
        assert!(text.contains("pikafish_label_eval_policy_softmax_temp = 1.15\n"));
        assert!(!text.contains("root_exploration_plies"));
        assert!(!text.contains("search_algorithm"));
        assert!(!text.contains("arena_pikafish"));
        assert!(!text.contains("arena_eval_fens"));
        assert!(!text.contains("000000047"));
        assert!(!text.contains("000000023"));

        let parsed = AzLoopFileConfig::parse(&text);
        assert_eq!(parsed.model_path, "model.safetensors");
        assert!((parsed.lr - 0.0004).abs() < 1e-9);
        assert_eq!(parsed.arena_interval, 10);
        assert_eq!(parsed.pikafish_label_eval_interval, 10);
    }

    #[test]
    fn removed_config_names_are_rejected() {
        for removed in [
            "selfplay_update_warmup_updates = 5\n",
            "opening_temperature = 1.25\n",
            "replay_recent_window_updates = 5000\n",
            "deblunder_q_gap = 0.05\n",
            "low_simulations = 2000\n",
            "low_simulation_probability = 0.2\n",
            "low_simulation_policy_weight = 0.5\n",
            "high_simulations = 20000\n",
            "high_simulation_probability = 0.1\n",
            "high_simulation_start_plies = 40\n",
            "value_target_search_q_mix = 0.4\n",
            "arena_pikafish_exe = \"./pikafish\"\n",
            "arena_pikafish_depth = 10\n",
            "arena_pikafish_games = 20\n",
            "persistent_exploration_fraction = 0.1\n",
            "persistent_exploration_temperature = 0.8\n",
            "persistent_exploration_root_dirichlet_alpha = 0.15\n",
            "persistent_exploration_root_exploration_fraction = 0.35\n",
            "resign_percentage = 1.0\n",
            "resign_playthrough = 20.0\n",
            "temperature_value_cutoff = 0.07\n",
            "temperature_visit_offset = -0.8\n",
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
