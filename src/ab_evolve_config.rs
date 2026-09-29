use chineseai::ab::AbNnueArch;
use chineseai::version::AB_EVOLVE_CONFIG_FORMAT_VERSION;
use serde::{Deserialize, Serialize};
use std::{fmt::Write, fs, path::Path};

pub const DEFAULT_AB_EVOLVE_CONFIG: &str = "chineseai.ab-evolve.toml";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AbEvolveFileConfig {
    pub format_version: u32,
    pub model_path: String,
    pub selfplay_nodes: usize,
    pub selfplay_samples_per_update: usize,
    pub lr: f32,
    pub batch_size: usize,
    pub max_plies: usize,
    pub sixty_move_rule: bool,
    pub rule60_max_ply: u16,
    pub hidden_size: usize,
    pub seed: u64,
    pub workers: usize,
    pub temperature_start: f32,
    pub temperature_cutoff_plies: usize,
    pub temperature_endgame: f32,
    pub temperature_decay_delay_plies: usize,
    pub temperature_decay_plies: usize,
    pub selfplay_opening_book: String,
    pub replay_capacity: usize,
    pub shuffle_size: usize,
    pub replay_recent_games: u32,
    pub train_warmup_samples: usize,
    pub train_samples_per_update: usize,
    pub mirror_probability: f32,
    pub train_value_weight: f32,
    pub checkpoint_interval: usize,
    pub checkpoint_dir: String,
    pub max_checkpoints: usize,
    pub arena_interval: usize,
    pub arena_nodes: usize,
    pub arena_openings: usize,
    pub arena_promotion_rate: f32,
    pub arena_promotion_confidence_z: f32,
    pub arena_processes: usize,
    pub arena_opening_book: String,
    pub pikafish_label_eval_sqlite: String,
    pub pikafish_label_eval_interval: usize,
    pub pikafish_label_eval_limit: usize,
    pub pikafish_label_eval_nodes: usize,
    pub tensorboard_logdir: String,
}

impl Default for AbEvolveFileConfig {
    fn default() -> Self {
        Self {
            format_version: AB_EVOLVE_CONFIG_FORMAT_VERSION,
            model_path: "model.safetensors".into(),
            selfplay_nodes: 10_000,
            selfplay_samples_per_update: 120_000,
            lr: 0.02,
            batch_size: 2048,
            max_plies: 450,
            sixty_move_rule: true,
            rule60_max_ply: 120,
            hidden_size: 256,
            seed: 20260420,
            workers: 0,
            temperature_start: 0.9,
            temperature_cutoff_plies: 78,
            temperature_endgame: 0.6,
            temperature_decay_delay_plies: 40,
            temperature_decay_plies: 120,
            selfplay_opening_book: "book.pgn.gz".into(),
            replay_capacity: 2_400_000,
            shuffle_size: 524_288,
            replay_recent_games: 7500,
            train_warmup_samples: 600_000,
            train_samples_per_update: 120_000,
            mirror_probability: 0.5,
            train_value_weight: 1.0,
            checkpoint_interval: 20,
            checkpoint_dir: "checkpoints".into(),
            max_checkpoints: 50,
            arena_interval: 20,
            arena_nodes: 10_000,
            arena_openings: 1000,
            arena_promotion_rate: 0.5,
            arena_promotion_confidence_z: 1.96,
            arena_processes: 128,
            arena_opening_book: "book.pgn.gz".into(),
            pikafish_label_eval_sqlite: "eval/pikafish-selfplay-5000-d20.sqlite".into(),
            pikafish_label_eval_interval: 20,
            pikafish_label_eval_limit: 1000,
            pikafish_label_eval_nodes: 6000,
            tensorboard_logdir: "runs/chineseai".into(),
        }
    }
}

impl AbEvolveFileConfig {
    pub fn to_file_text(&self) -> String {
        let mut out = String::new();
        macro_rules! line {
            ($key:ident) => {
                writeln!(out, "{} = {}", stringify!($key), self.$key).unwrap()
            };
            ($key:ident, string) => {
                writeln!(out, "{} = {:?}", stringify!($key), self.$key).unwrap()
            };
        }
        line!(format_version);
        line!(model_path, string);
        line!(selfplay_nodes);
        line!(selfplay_samples_per_update);
        line!(lr);
        line!(batch_size);
        line!(max_plies);
        line!(sixty_move_rule);
        line!(rule60_max_ply);
        line!(hidden_size);
        line!(seed);
        line!(workers);
        line!(temperature_start);
        line!(temperature_cutoff_plies);
        line!(temperature_endgame);
        line!(temperature_decay_delay_plies);
        line!(temperature_decay_plies);
        line!(selfplay_opening_book, string);
        line!(replay_capacity);
        line!(shuffle_size);
        line!(replay_recent_games);
        line!(train_warmup_samples);
        line!(train_samples_per_update);
        line!(mirror_probability);
        line!(train_value_weight);
        line!(checkpoint_interval);
        line!(checkpoint_dir, string);
        line!(max_checkpoints);
        line!(arena_interval);
        line!(arena_nodes);
        line!(arena_openings);
        line!(arena_promotion_rate);
        line!(arena_promotion_confidence_z);
        line!(arena_processes);
        line!(arena_opening_book, string);
        line!(pikafish_label_eval_sqlite, string);
        line!(pikafish_label_eval_interval);
        line!(pikafish_label_eval_limit);
        line!(pikafish_label_eval_nodes);
        line!(tensorboard_logdir, string);
        out
    }

    pub(crate) fn parse(text: &str) -> Self {
        let config = toml::from_str::<Self>(text)
            .unwrap_or_else(|err| panic!("invalid ab-evolve TOML config: {err}"));
        assert_eq!(
            config.format_version, AB_EVOLVE_CONFIG_FORMAT_VERSION,
            "unsupported ab-evolve config format"
        );
        config.normalize()
    }

    pub fn arch(&self) -> AbNnueArch {
        AbNnueArch {
            hidden_size: self.hidden_size,
        }
    }

    fn normalize(mut self) -> Self {
        self.selfplay_nodes = self.selfplay_nodes.max(1);
        self.selfplay_samples_per_update = self.selfplay_samples_per_update.max(1);
        self.lr = self.lr.max(0.0);
        self.batch_size = self.batch_size.max(1);
        self.max_plies = self.max_plies.max(1);
        self.rule60_max_ply = self.rule60_max_ply.clamp(1, 150);
        self.hidden_size = self.hidden_size.max(1);
        if self.workers == 0 {
            self.workers = num_cpus::get_physical().max(1);
        }
        self.temperature_start = self.temperature_start.max(0.0);
        self.temperature_endgame = self.temperature_endgame.max(0.0);
        self.temperature_decay_delay_plies = self.temperature_decay_delay_plies.min(self.max_plies);
        self.temperature_decay_plies = self.temperature_decay_plies.min(self.max_plies);
        self.replay_recent_games = self.replay_recent_games.max(1);
        self.replay_capacity = self.replay_capacity.max(self.batch_size);
        self.shuffle_size = self.shuffle_size.max(1);
        self.train_warmup_samples = self.train_warmup_samples.max(1);
        self.train_samples_per_update = self.train_samples_per_update.max(1);
        self.mirror_probability = self.mirror_probability.clamp(0.0, 1.0);
        self.train_value_weight = self.train_value_weight.max(0.0);
        self.max_checkpoints = self.max_checkpoints.max(1);
        self.arena_processes = self.arena_processes.max(1);
        self.arena_promotion_rate = self.arena_promotion_rate.clamp(0.0, 1.0);
        self.arena_promotion_confidence_z = self.arena_promotion_confidence_z.max(0.0);
        self.arena_nodes = self.arena_nodes.max(1);
        self.arena_openings = self.arena_openings.max(1);
        self.arena_interval = self.arena_interval.max(1);
        self.pikafish_label_eval_nodes = self.pikafish_label_eval_nodes.max(1);
        self
    }
}

pub fn load_or_create_ab_evolve_config(path: &str) -> Option<AbEvolveFileConfig> {
    if !Path::new(path).exists() {
        fs::write(path, AbEvolveFileConfig::default().to_file_text())
            .unwrap_or_else(|err| panic!("failed to create `{path}`: {err}"));
        println!("created config: {path}");
        println!("edit it, then run: chineseai ab-evolve {path}");
        return None;
    }
    let text =
        fs::read_to_string(path).unwrap_or_else(|err| panic!("failed to read `{path}`: {err}"));
    Some(AbEvolveFileConfig::parse(&text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_config_roundtrips_with_node_budgets() {
        let text = AbEvolveFileConfig::default().to_file_text();
        let config = AbEvolveFileConfig::parse(&text);
        assert_eq!(config.selfplay_nodes, 10_000);
        assert_eq!(config.arena_nodes, 10_000);
        assert_eq!(config.arena_openings, 1000);
        assert_eq!(config.hidden_size, 256);
        assert!(text.contains("selfplay_nodes = 10000"));
        assert!(text.contains("arena_nodes = 10000"));
        assert!(text.contains("arena_openings = 1000"));
    }

    #[test]
    fn normalize_keeps_training_and_arena_live() {
        let mut config = AbEvolveFileConfig::default();
        config.replay_capacity = 0;
        config.train_samples_per_update = 0;
        config.arena_openings = 0;
        config.arena_interval = 0;
        let config = config.normalize();
        assert!(config.replay_capacity >= config.batch_size);
        assert_eq!(config.train_samples_per_update, 1);
        assert_eq!(config.arena_openings, 1);
        assert_eq!(config.arena_interval, 1);
    }
}
