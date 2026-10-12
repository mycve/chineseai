use chineseai::az::{AzMovesLeftParams, AzNnueArch, AzTrainOptimizer};
use chineseai::infra::version::AZ_LOOP_CONFIG_FORMAT_VERSION;
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
    pub batch_size: usize,
    pub max_plies: usize,
    pub sixty_move_rule: bool,
    pub rule60_max_ply: u16,
    pub hidden_size: usize,
    pub seed: u64,
    pub workers: usize,
    pub temperature_start: f32,
    pub temperature_cutoff_plies: usize,
    pub temperature_visit_offset: f32,
    pub temperature_endgame: f32,
    pub temperature_decay_delay_plies: usize,
    pub temperature_decay_plies: usize,
    pub cpuct: f32,
    pub cpuct_at_root: f32,
    pub cpuct_base: f32,
    pub cpuct_factor: f32,
    pub cpuct_base_at_root: f32,
    pub cpuct_factor_at_root: f32,
    /// 每个根走法的 Dirichlet alpha，0 关闭噪声。
    pub root_dirichlet_alpha: f32,
    pub root_exploration_fraction: f32,
    pub fpu_value: f32,
    pub fpu_value_at_root: f32,
    pub fpu_absolute_at_root: bool,
    pub minimum_kldgain_per_node: f32,
    pub draw_score: f32,
    pub policy_softmax_temp: f32,
    pub selfplay_opening_book: String,
    /// 根节点连杀证明搜索的最大半回合数（0 = 关闭，9 = 最多 mate in 5，15 = mate in 8）。
    ///
    /// 带 `#[serde(default)]`：老配置文件里没有这一项时按 0（关闭）解析，不必改格式版本。
    #[serde(default)]
    pub mate_search_plies: usize,
    pub moves_left_enabled: bool,
    pub moves_left_threshold: f32,
    pub moves_left_max_effect: f32,
    pub moves_left_slope: f32,
    pub moves_left_constant_factor: f32,
    pub moves_left_scaled_factor: f32,
    pub moves_left_quadratic_factor: f32,

    pub replay_capacity: usize,
    pub shuffle_size: usize,
    pub replay_recent_games: u32,
    pub train_warmup_samples: usize,
    pub train_samples_per_update: usize,
    /// 训练优化器内核：`"adamw"`（默认）或 `"px0-sgd"`。
    ///
    /// 带 `#[serde(default)]`：老配置文件里没有这一项时按默认值解析，不必改格式版本。
    /// 注意 `lr` 的语义随内核变化 —— AdamW 用常数 lr（逐坐标自适应，步长 ≈ lr），
    /// Px0Sgd 用 `lr` 作 base 并乘 warmup 与按累计步数分段的阶梯，两者尺度不可直接比较。
    #[serde(default)]
    pub train_optimizer: AzTrainOptimizer,
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
        let moves_left = AzMovesLeftParams::default();
        Self {
            format_version: AZ_LOOP_CONFIG_FORMAT_VERSION,
            model_path: "model.safetensors".into(),
            simulations: 10_000,
            selfplay_samples_per_update: 120000,
            // 默认内核是 AdamW（逐坐标自适应，步长 ≈ lr），所以 `lr` 用旧配方的 4e-4。
            // 改用 `train_optimizer = "px0-sgd"` 时必须把 `lr` 提到 0.02 量级：带动量 SGD 的
            // 步长 ∝ lr·g，两个尺度不可混用。
            lr: 0.0004,
            batch_size: 2048,
            // Px0 SelfPlayGame采用450步上限，200步会过早丢失终局价值标签。
            max_plies: 450,
            sixty_move_rule: true,
            rule60_max_ply: 120,
            hidden_size: AzNnueArch::default().hidden_size,
            seed: 20260420,
            workers: 0,
            temperature_start: 0.9,
            temperature_cutoff_plies: 78,
            temperature_visit_offset: -0.8,
            temperature_endgame: 0.6,
            temperature_decay_delay_plies: 40,
            temperature_decay_plies: 120,
            cpuct: 1.2,
            cpuct_at_root: 2.0,
            cpuct_base: 38739.0,
            cpuct_factor: 3.894,
            cpuct_base_at_root: 38739.0,
            cpuct_factor_at_root: 3.894,
            root_dirichlet_alpha: 0.12,
            root_exploration_fraction: 0.1,
            fpu_value: 0.49,
            fpu_value_at_root: 1.0,
            fpu_absolute_at_root: true,
            minimum_kldgain_per_node: 0.00005,
            draw_score: 0.0,
            policy_softmax_temp: 1.45,
            selfplay_opening_book: "book.pgn.gz".into(),
            // 9 半回合 = mate in 5：实测把可证射程从 mate-in-4 推到 mate-in-5，
            // 而"有将军但无杀"的局面只 +4%，根局面没有将军着法时为 0。
            mate_search_plies: 9,
            moves_left_enabled: moves_left.enabled,
            moves_left_threshold: moves_left.threshold,
            moves_left_max_effect: moves_left.max_effect,
            moves_left_slope: moves_left.slope,
            moves_left_constant_factor: moves_left.constant_factor,
            moves_left_scaled_factor: moves_left.scaled_factor,
            moves_left_quadratic_factor: moves_left.quadratic_factor,

            replay_capacity: 2400000,
            shuffle_size: 524_288,
            replay_recent_games: 7500,
            train_warmup_samples: 600000,
            train_samples_per_update: 120000,
            train_optimizer: AzTrainOptimizer::default(),
            mirror_probability: 0.5,
            train_value_weight: 1.0,
            train_policy_weight: 1.0,
            checkpoint_interval: 20,
            checkpoint_dir: "checkpoints".into(),
            max_checkpoints: 50,
            arena_interval: 20,
            arena_simulations: 800,
            arena_cpuct: 1.0,
            arena_cpuct_at_root: 1.9,
            arena_policy_softmax_temp: 1.4,
            arena_promotion_rate: 0.50,
            arena_promotion_confidence_z: 1.96,
            arena_processes: 128,
            arena_opening_book: "book.pgn.gz".into(),
            pikafish_label_eval_sqlite: "eval/pikafish-selfplay-5000-d20.sqlite".into(),
            pikafish_label_eval_interval: 20,
            pikafish_label_eval_limit: 1000,
            pikafish_label_eval_simulations: 6000,
            pikafish_label_eval_cpuct: 1.0,
            pikafish_label_eval_cpuct_at_root: 1.9,
            pikafish_label_eval_policy_softmax_temp: 1.4,
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
        line!("selfplay_opening_book", q(&self.selfplay_opening_book));
        line!("mate_search_plies", self.mate_search_plies);
        line!("moves_left_enabled", self.moves_left_enabled);
        line!("moves_left_threshold", f(self.moves_left_threshold));
        line!("moves_left_max_effect", f(self.moves_left_max_effect));
        line!("moves_left_slope", f(self.moves_left_slope));
        line!(
            "moves_left_constant_factor",
            f(self.moves_left_constant_factor)
        );
        line!("moves_left_scaled_factor", f(self.moves_left_scaled_factor));
        line!(
            "moves_left_quadratic_factor",
            f(self.moves_left_quadratic_factor)
        );

        line!("minimum_kldgain_per_node", f(self.minimum_kldgain_per_node));
        line!("fpu_absolute_at_root", self.fpu_absolute_at_root);
        line!("temperature_visit_offset", f(self.temperature_visit_offset));
        line!("temperature_cutoff_plies", self.temperature_cutoff_plies);
        line!("simulations", self.simulations);
        line!(
            "selfplay_samples_per_update",
            self.selfplay_samples_per_update
        );
        line!("lr", f(self.lr));
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
        line!("replay_capacity", self.replay_capacity);
        line!("shuffle_size", self.shuffle_size);
        line!("replay_recent_games", self.replay_recent_games);
        line!("train_warmup_samples", self.train_warmup_samples);
        line!("train_samples_per_update", self.train_samples_per_update);
        line!("train_optimizer", q(self.train_optimizer.as_str()));
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

    pub(crate) fn parse(text: &str) -> Self {
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

    pub fn moves_left_params(&self) -> AzMovesLeftParams {
        AzMovesLeftParams {
            enabled: self.moves_left_enabled,
            threshold: self.moves_left_threshold,
            max_effect: self.moves_left_max_effect,
            slope: self.moves_left_slope,
            constant_factor: self.moves_left_constant_factor,
            scaled_factor: self.moves_left_scaled_factor,
            quadratic_factor: self.moves_left_quadratic_factor,
        }
        .normalize()
    }

    pub fn arch(&self) -> AzNnueArch {
        AzNnueArch {
            hidden_size: self.hidden_size,
        }
    }

    fn normalize(mut self) -> Self {
        let moves_left = self.moves_left_params();
        self.moves_left_enabled = moves_left.enabled;
        self.moves_left_threshold = moves_left.threshold;
        self.moves_left_max_effect = moves_left.max_effect;
        self.moves_left_slope = moves_left.slope;
        self.moves_left_constant_factor = moves_left.constant_factor;
        self.moves_left_scaled_factor = moves_left.scaled_factor;
        self.moves_left_quadratic_factor = moves_left.quadratic_factor;

        self.simulations = self.simulations.max(1);
        self.selfplay_samples_per_update = self.selfplay_samples_per_update.max(1);
        self.lr = self.lr.max(0.0);
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
        self.minimum_kldgain_per_node = self.minimum_kldgain_per_node.max(0.0);
        self.policy_softmax_temp = self.policy_softmax_temp.max(1e-3);
        // 上限 31 半回合（mate in 16）与 `go mate 0` 的取值一致：再深只会烧时间。
        self.mate_search_plies = self.mate_search_plies.min(31);

        self.replay_recent_games = self.replay_recent_games.max(1);
        self.shuffle_size = self.shuffle_size.max(1);
        self.train_warmup_samples = self.train_warmup_samples.max(1);
        self.train_samples_per_update = self.train_samples_per_update.max(1);
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
        self.pikafish_label_eval_simulations = self.pikafish_label_eval_simulations.max(1);
        self.pikafish_label_eval_cpuct = self.pikafish_label_eval_cpuct.max(0.0);
        self.pikafish_label_eval_cpuct_at_root = self.pikafish_label_eval_cpuct_at_root.max(0.0);
        self.pikafish_label_eval_policy_softmax_temp =
            self.pikafish_label_eval_policy_softmax_temp.max(1e-3);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_defaults_match_public_px0_selfplay_settings() {
        let config = AzLoopFileConfig::parse(&AzLoopFileConfig::default().to_file_text());
        let expected = AzLoopFileConfig::default();
        assert_eq!(config.selfplay_opening_book, "book.pgn.gz");
        assert_eq!(config.arena_opening_book, "book.pgn.gz");
        assert_eq!(config.simulations, 10_000);
        assert_eq!(config.cpuct, expected.cpuct);
        assert_eq!(config.root_dirichlet_alpha, 0.12);
        assert_eq!(config.temperature_cutoff_plies, 78);
        assert!(config.fpu_absolute_at_root);
    }

    /// 连杀预算要能往返、能 clamp；默认值就是推荐的 9；**老配置文件缺这一项时必须按 0
    /// 解析而不是报错**（`#[serde(default)]`）。
    #[test]
    fn moves_left_config_roundtrips_defaults_and_clamps() {
        let config = AzLoopFileConfig {
            moves_left_enabled: false,
            moves_left_threshold: 0.9,
            moves_left_max_effect: 0.02,
            moves_left_slope: 0.001,
            moves_left_constant_factor: 0.1,
            moves_left_scaled_factor: 1.2,
            moves_left_quadratic_factor: -0.4,
            ..AzLoopFileConfig::default()
        };
        let text = config.to_file_text();
        assert_eq!(
            AzLoopFileConfig::parse(&text).moves_left_params(),
            config.moves_left_params()
        );
        let legacy = text
            .lines()
            .filter(|line| !line.starts_with("moves_left_"))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            AzLoopFileConfig::parse(&legacy).moves_left_params(),
            AzMovesLeftParams::default()
        );
        let clamped = AzLoopFileConfig {
            moves_left_threshold: 5.0,
            moves_left_max_effect: -1.0,
            moves_left_slope: f32::NAN,
            ..AzLoopFileConfig::default()
        }
        .normalize();
        assert_eq!(clamped.moves_left_threshold, 1.0);
        assert_eq!(clamped.moves_left_max_effect, 0.0);
        assert_eq!(clamped.moves_left_slope, AzMovesLeftParams::default().slope);
    }

    #[test]
    fn mate_search_config_roundtrips_and_clamps() {
        assert_eq!(AzLoopFileConfig::default().mate_search_plies, 9);

        let config = AzLoopFileConfig {
            mate_search_plies: 15,
            ..AzLoopFileConfig::default()
        };
        let text = config.to_file_text();
        assert!(text.contains("mate_search_plies = 15\n"));
        assert_eq!(AzLoopFileConfig::parse(&text).mate_search_plies, 15);

        let legacy = text
            .lines()
            .filter(|line| !line.starts_with("mate_search_plies"))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            AzLoopFileConfig::parse(&legacy).mate_search_plies,
            0,
            "老配置缺这一项时应按关闭解析"
        );

        let clamped = AzLoopFileConfig::parse(
            &AzLoopFileConfig {
                mate_search_plies: 99,
                ..AzLoopFileConfig::default()
            }
            .to_file_text(),
        );
        assert_eq!(clamped.mate_search_plies, 31);
    }

    /// 优化器选择要能原样往返；老配置缺这一项时按默认（AdamW）解析。
    #[test]
    fn train_optimizer_config_roundtrips_and_defaults() {
        assert_eq!(
            AzLoopFileConfig::default().train_optimizer,
            AzTrainOptimizer::AdamW
        );

        let config = AzLoopFileConfig {
            train_optimizer: AzTrainOptimizer::Px0Sgd,
            ..AzLoopFileConfig::default()
        };
        let text = config.to_file_text();
        assert!(text.contains("train_optimizer = \"px0-sgd\"\n"));
        assert_eq!(
            AzLoopFileConfig::parse(&text).train_optimizer,
            AzTrainOptimizer::Px0Sgd
        );

        let legacy = text
            .lines()
            .filter(|line| !line.starts_with("train_optimizer"))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            AzLoopFileConfig::parse(&legacy).train_optimizer,
            AzTrainOptimizer::AdamW,
            "老配置缺这一项时应按默认的 adamw 解析"
        );
    }

    #[test]
    fn fixed_dirichlet_config_roundtrips() {
        let config = AzLoopFileConfig {
            root_dirichlet_alpha: 0.12,
            ..AzLoopFileConfig::default()
        };
        let restored: AzLoopFileConfig = toml::from_str(&config.to_file_text()).unwrap();
        assert_eq!(restored.root_dirichlet_alpha, 0.12);
    }

    #[test]
    fn config_writer_uses_short_float_literals() {
        let config = AzLoopFileConfig::default();
        let text = config.to_file_text();

        assert!(text.starts_with("format_version = 32\n"));
        assert!(text.contains("lr = 0.0004\n"));
        assert!(text.contains("temperature_start = 0.9\n"));
        assert!(text.contains("sixty_move_rule = true\n"));
        assert!(text.contains("rule60_max_ply = 120\n"));
        assert!(text.contains("temperature_endgame = 0.6\n"));
        assert!(text.contains("temperature_decay_delay_plies = 40\n"));
        assert!(text.contains("temperature_decay_plies = 120\n"));
        assert!(text.contains("temperature_cutoff_plies = 78"));
        assert!(text.contains("cpuct = 1.2\n"));
        assert!(text.contains("cpuct_at_root = 2.0\n"));
        assert!(text.contains("cpuct_base = 38739.0\n"));
        assert!(text.contains("cpuct_factor = 3.894\n"));
        assert!(text.contains("cpuct_base_at_root = 38739.0\n"));
        assert!(text.contains("cpuct_factor_at_root = 3.894\n"));
        assert!(text.contains("root_dirichlet_alpha = 0.12\n"));
        assert!(text.contains("root_exploration_fraction = 0.1\n"));
        assert!(text.contains("fpu_value = 0.49\n"));
        assert!(text.contains("fpu_value_at_root = 1.0\n"));
        assert!(text.contains("draw_score = 0.0\n"));
        assert!(text.contains("policy_softmax_temp = 1.45\n"));
        assert!(!text.contains("value_target_search_q_mix"));
        assert!(text.contains("simulations = 10000\n"));
        assert!(!text.contains("low_simulations"));
        assert!(!text.contains("low_simulation_probability"));
        assert!(!text.contains("low_simulation_policy_weight"));
        assert!(!text.contains("high_simulations"));
        assert!(!text.contains("high_simulation_probability"));
        assert!(!text.contains("high_simulation_start_plies"));
        assert!(text.contains("selfplay_samples_per_update = 120000\n"));
        assert!(text.contains("workers = 0\n"));
        assert!(text.contains("batch_size = 2048\n"));
        assert!(text.contains("max_plies = 450\n"));
        assert!(text.contains("hidden_size = 96\n"));
        assert!(text.contains("replay_capacity = 2400000\n"));
        assert!(text.contains("train_samples_per_update = 120000\n"));
        assert!(text.contains("train_warmup_samples = 600000\n"));
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
        assert!(text.contains("mirror_probability = 0.5\n"));
        assert!(text.contains("arena_processes = 128\n"));
        assert!(text.contains("arena_opening_book = \"book.pgn.gz\"\n"));
        assert!(text.contains("arena_interval = 20\n"));
        assert!(text.contains("arena_simulations = 800\n"));
        assert!(text.contains("arena_promotion_rate = 0.5\n"));
        assert!(text.contains("arena_promotion_confidence_z = 1.96\n"));
        assert!(text.contains("arena_cpuct = 1.0\n"));
        assert!(text.contains("arena_cpuct_at_root = 1.9\n"));
        assert!(text.contains("arena_policy_softmax_temp = 1.4\n"));
        assert!(
            text.contains(
                "pikafish_label_eval_sqlite = \"eval/pikafish-selfplay-5000-d20.sqlite\"\n"
            )
        );
        assert!(text.contains("pikafish_label_eval_interval = 20\n"));
        assert!(text.contains("pikafish_label_eval_limit = 1000\n"));
        assert!(text.contains("pikafish_label_eval_simulations = 6000\n"));
        assert!(text.contains("pikafish_label_eval_cpuct = 1.0\n"));
        assert!(text.contains("pikafish_label_eval_cpuct_at_root = 1.9\n"));
        assert!(text.contains("pikafish_label_eval_policy_softmax_temp = 1.4\n"));
        assert!(!text.contains("root_exploration_plies"));
        assert!(!text.contains("search_algorithm"));
        assert!(!text.contains("arena_pikafish"));
        assert!(!text.contains("arena_eval_fens"));
        assert!(!text.contains("000000047"));
        assert!(!text.contains("000000023"));

        let parsed = AzLoopFileConfig::parse(&text);
        assert_eq!(parsed.model_path, "model.safetensors");
        assert!((parsed.lr - 0.0004).abs() < 1e-9);
        assert_eq!(parsed.arena_interval, 20);
        assert_eq!(parsed.pikafish_label_eval_interval, 20);
    }

    #[test]
    fn old_config_versions_are_rejected() {
        let text = AzLoopFileConfig::default()
            .to_file_text()
            .replace("format_version = 32", "format_version = 27");
        let error = std::panic::catch_unwind(|| AzLoopFileConfig::parse(&text));
        assert!(error.is_err());
    }

    #[test]
    fn removed_config_names_are_rejected() {
        for removed in [
            "opening_start_fraction = 1.0\n",
            "midgame_start_fraction = 0.0\n",
            "opening_reservoir_capacity = 0\n",
            "midgame_reservoir_capacity = 0\n",
            "root_dirichlet_total_concentration = 8.0\n",
            "actor_publish_interval_updates = 5\n",
            "actor_noninferiority_margin = 0.02\n",
            "actor_gate_min_games = 400\n",
            "persistent_exploration_root_dirichlet_alpha = 0.15\n",
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
            "train_epochs_per_update = 1\n",
            "resign_percentage = 2.0\n",
            "resign_playthrough = 0.2\n",
            "arena_pikafish_exe = \"./pikafish\"\n",
            "arena_pikafish_depth = 10\n",
            "arena_pikafish_games = 20\n",
            "persistent_exploration_fraction = 0.1\n",
            "persistent_exploration_temperature = 0.8\n",
            "persistent_exploration_root_exploration_fraction = 0.35\n",
            "temperature_value_cutoff = 0.07\n",
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
