use std::collections::VecDeque;
use std::fs;
use std::io;
use matrixmultiply::sgemm;
use std::path::Path;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use crate::nnue::{
    HistoryMove, V4_INPUT_SIZE, extract_sparse_features_v4, mirror_file_move,
    mirror_sparse_features_file, orient_move,
};
use crate::xiangqi::{
    BOARD_FILES, BOARD_SIZE, Color, Move, Position, RuleDrawReason, RuleHistoryEntry,
    RuleOutcome,
};

pub const AZNNUE_FORMAT: &str = "aznnue-v14";
const SPARSE_MOVE_SPACE: usize = BOARD_SIZE * BOARD_SIZE;
const DENSE_MOVE_SPACE: usize = compute_dense_move_count();
const GLOBAL_CONTEXT_SIZE: usize = 32;
const VALUE_HIDDEN_SIZE: usize = 64;
const VALUE_LOGITS: usize = 3;
const VALUE_SCALE_CP: f32 = 800.0;
const COMPLETED_Q_VALUE_SCALE: f32 = 0.1;
const COMPLETED_Q_MAXVISIT_INIT: f32 = 50.0;
const RESIDUAL_TRUNK_SCALE: f32 = 0.5;
const ADAMW_BETA1: f32 = 0.9;
const ADAMW_BETA2: f32 = 0.999;
const ADAMW_EPSILON: f32 = 1e-8;
const ADAMW_WEIGHT_DECAY: f32 = 1e-4;

/// 单次 NN 前向复用的临时张量，由 [`AzTree`] 持有，避免每步模拟反复 `Vec` 分配。
struct AzEvalScratch {
    hidden: Vec<f32>,
    next: Vec<f32>,
    global: Vec<f32>,
    value_intermediate: Vec<f32>,
    value_logits: Vec<f32>,
    logits: Vec<f32>,
    priors: Vec<f32>,
}

impl AzEvalScratch {
    fn new(hidden_size: usize) -> Self {
        Self {
            hidden: vec![0.0; hidden_size],
            next: vec![0.0; hidden_size],
            global: vec![0.0; GLOBAL_CONTEXT_SIZE],
            value_intermediate: vec![0.0; VALUE_HIDDEN_SIZE],
            value_logits: vec![0.0; VALUE_LOGITS],
            logits: Vec::with_capacity(192),
            priors: Vec::with_capacity(192),
        }
    }
}

#[derive(Debug)]
pub struct AzNnue {
    pub hidden_size: usize,
    pub trunk_depth: usize,
    pub input_hidden: Vec<f32>,
    pub hidden_bias: Vec<f32>,
    pub trunk_weights: Vec<f32>,
    pub trunk_biases: Vec<f32>,
    pub trunk_global_weights: Vec<f32>,
    pub global_hidden: Vec<f32>,
    pub global_bias: Vec<f32>,
    pub value_intermediate_hidden: Vec<f32>,
    pub value_intermediate_global: Vec<f32>,
    pub value_intermediate_bias: Vec<f32>,
    pub value_logits_weights: Vec<f32>,
    pub value_logits_bias: Vec<f32>,
    pub policy_move_hidden: Vec<f32>,
    pub policy_move_global: Vec<f32>,
    pub policy_move_bias: Vec<f32>,
    optimizer: Option<Box<AdamWState>>,
}

impl Clone for AzNnue {
    fn clone(&self) -> Self {
        Self {
            hidden_size: self.hidden_size,
            trunk_depth: self.trunk_depth,
            input_hidden: self.input_hidden.clone(),
            hidden_bias: self.hidden_bias.clone(),
            trunk_weights: self.trunk_weights.clone(),
            trunk_biases: self.trunk_biases.clone(),
            trunk_global_weights: self.trunk_global_weights.clone(),
            global_hidden: self.global_hidden.clone(),
            global_bias: self.global_bias.clone(),
            value_intermediate_hidden: self.value_intermediate_hidden.clone(),
            value_intermediate_global: self.value_intermediate_global.clone(),
            value_intermediate_bias: self.value_intermediate_bias.clone(),
            value_logits_weights: self.value_logits_weights.clone(),
            value_logits_bias: self.value_logits_bias.clone(),
            policy_move_hidden: self.policy_move_hidden.clone(),
            policy_move_global: self.policy_move_global.clone(),
            policy_move_bias: self.policy_move_bias.clone(),
            optimizer: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AzSearchLimits {
    pub simulations: usize,
    pub top_k: usize,
    pub seed: u64,
    pub gumbel_scale: f32,
    pub workers: usize,
}

impl Default for AzSearchLimits {
    fn default() -> Self {
        Self {
            simulations: 10_000,
            top_k: 32,
            seed: 0,
            gumbel_scale: 1.0,
            workers: 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AzCandidate {
    pub mv: Move,
    pub visits: u32,
    pub q: f32,
    pub prior: f32,
    pub policy: f32,
}

#[derive(Clone, Debug)]
pub struct AzSearchResult {
    pub best_move: Option<Move>,
    pub value_cp: i32,
    pub simulations: usize,
    pub candidates: Vec<AzCandidate>,
}

#[derive(Clone, Debug)]
pub struct AzLoopConfig {
    pub games: usize,
    pub max_plies: usize,
    pub simulations: usize,
    pub top_k: usize,
    pub epochs: usize,
    pub lr: f32,
    pub batch_size: usize,
    pub seed: u64,
    pub workers: usize,
    pub temperature_start: f32,
    pub temperature_end: f32,
    pub temperature_decay_plies: usize,
    pub gumbel_scale: f32,
    pub td_lambda: f32,
    pub replay_games: usize,
    pub replay_samples: usize,
    pub mirror_probability: f32,
}

#[derive(Clone, Debug)]
pub struct AzLoopReport {
    pub games: usize,
    pub samples: usize,
    pub red_wins: usize,
    pub black_wins: usize,
    pub draws: usize,
    pub avg_plies: f32,
    pub loss: f32,
    pub value_loss: f32,
    pub policy_ce: f32,
    pub temperature_early_entropy: f32,
    pub temperature_mid_entropy: f32,
    pub selfplay_seconds: f32,
    pub train_seconds: f32,
    pub total_seconds: f32,
    pub games_per_second: f32,
    pub samples_per_second: f32,
    pub train_samples_per_second: f32,
    pub train_samples: usize,
    pub pool_games: usize,
    pub pool_samples: usize,
    pub terminal_no_legal_moves: usize,
    pub terminal_red_general_missing: usize,
    pub terminal_black_general_missing: usize,
    pub terminal_rule_draw: usize,
    pub terminal_rule_draw_halfmove120: usize,
    pub terminal_rule_draw_repetition: usize,
    pub terminal_rule_draw_mutual_long_check: usize,
    pub terminal_rule_draw_mutual_long_chase: usize,
    pub terminal_rule_win_red: usize,
    pub terminal_rule_win_black: usize,
    pub terminal_max_plies: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct AzTerminalStats {
    no_legal_moves: usize,
    red_general_missing: usize,
    black_general_missing: usize,
    rule_draw: usize,
    rule_draw_halfmove120: usize,
    rule_draw_repetition: usize,
    rule_draw_mutual_long_check: usize,
    rule_draw_mutual_long_chase: usize,
    rule_win_red: usize,
    rule_win_black: usize,
    max_plies: usize,
}

impl AzTerminalStats {
    fn add_assign(&mut self, other: &Self) {
        self.no_legal_moves += other.no_legal_moves;
        self.red_general_missing += other.red_general_missing;
        self.black_general_missing += other.black_general_missing;
        self.rule_draw += other.rule_draw;
        self.rule_draw_halfmove120 += other.rule_draw_halfmove120;
        self.rule_draw_repetition += other.rule_draw_repetition;
        self.rule_draw_mutual_long_check += other.rule_draw_mutual_long_check;
        self.rule_draw_mutual_long_chase += other.rule_draw_mutual_long_chase;
        self.rule_win_red += other.rule_win_red;
        self.rule_win_black += other.rule_win_black;
        self.max_plies += other.max_plies;
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AzArenaReport {
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
    pub wins_as_red: usize,
    pub losses_as_red: usize,
    pub wins_as_black: usize,
    pub losses_as_black: usize,
}

impl AzArenaReport {
    pub fn add_assign(&mut self, other: &Self) {
        self.wins += other.wins;
        self.losses += other.losses;
        self.draws += other.draws;
        self.wins_as_red += other.wins_as_red;
        self.losses_as_red += other.losses_as_red;
        self.wins_as_black += other.wins_as_black;
        self.losses_as_black += other.losses_as_black;
    }

    pub fn total_games(&self) -> usize {
        self.wins + self.losses + self.draws
    }

    pub fn score(&self) -> f32 {
        self.wins as f32 + 0.5 * self.draws as f32
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AzTrainBenchmark {
    pub loss: f32,
    pub value_loss: f32,
    pub policy_ce: f32,
}

#[derive(Clone, Debug)]
struct AzTrainingSample {
    features: Vec<usize>,
    move_indices: Vec<usize>,
    policy: Vec<f32>,
    value: f32,
    side_sign: f32,
    reward: f32,
    discount: f32,
    bootstrap_value: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct AzTrainStats {
    loss: f32,
    value_loss: f32,
    policy_ce: f32,
    value_pred_sum: f32,
    value_pred_sq_sum: f32,
    value_target_sum: f32,
    value_target_sq_sum: f32,
    samples: usize,
}

#[derive(Debug)]
struct AzGrad {
    input_hidden: Vec<f32>,
    hidden_bias: Vec<f32>,
    trunk_weights: Vec<f32>,
    trunk_biases: Vec<f32>,
    trunk_global_weights: Vec<f32>,
    global_hidden: Vec<f32>,
    global_bias: Vec<f32>,
    value_intermediate_hidden: Vec<f32>,
    value_intermediate_global: Vec<f32>,
    value_intermediate_bias: Vec<f32>,
    value_logits_weights: Vec<f32>,
    value_logits_bias: Vec<f32>,
    policy_move_hidden: Vec<f32>,
    policy_move_global: Vec<f32>,
    policy_move_bias: Vec<f32>,
}

impl AzGrad {
    fn new(model: &AzNnue) -> Self {
        Self {
            input_hidden: vec![0.0; model.input_hidden.len()],
            hidden_bias: vec![0.0; model.hidden_bias.len()],
            trunk_weights: vec![0.0; model.trunk_weights.len()],
            trunk_biases: vec![0.0; model.trunk_biases.len()],
            trunk_global_weights: vec![0.0; model.trunk_global_weights.len()],
            global_hidden: vec![0.0; model.global_hidden.len()],
            global_bias: vec![0.0; model.global_bias.len()],
            value_intermediate_hidden: vec![0.0; model.value_intermediate_hidden.len()],
            value_intermediate_global: vec![0.0; model.value_intermediate_global.len()],
            value_intermediate_bias: vec![0.0; model.value_intermediate_bias.len()],
            value_logits_weights: vec![0.0; model.value_logits_weights.len()],
            value_logits_bias: vec![0.0; model.value_logits_bias.len()],
            policy_move_hidden: vec![0.0; model.policy_move_hidden.len()],
            policy_move_global: vec![0.0; model.policy_move_global.len()],
            policy_move_bias: vec![0.0; model.policy_move_bias.len()],
        }
    }

    fn clear(&mut self) {
        self.input_hidden.fill(0.0);
        self.hidden_bias.fill(0.0);
        self.trunk_weights.fill(0.0);
        self.trunk_biases.fill(0.0);
        self.global_hidden.fill(0.0);
        self.global_bias.fill(0.0);
        self.value_intermediate_hidden.fill(0.0);
        self.value_intermediate_global.fill(0.0);
        self.value_intermediate_bias.fill(0.0);
        self.value_logits_weights.fill(0.0);
        self.value_logits_bias.fill(0.0);
        self.policy_move_hidden.fill(0.0);
        self.policy_move_global.fill(0.0);
        self.policy_move_bias.fill(0.0);
    }

}

impl AzTrainStats {
    fn add_assign(&mut self, other: &Self) {
        self.loss += other.loss;
        self.value_loss += other.value_loss;
        self.policy_ce += other.policy_ce;
        self.value_pred_sum += other.value_pred_sum;
        self.value_pred_sq_sum += other.value_pred_sq_sum;
        self.value_target_sum += other.value_target_sum;
        self.value_target_sq_sum += other.value_target_sq_sum;
        self.samples += other.samples;
    }
}

#[derive(Debug)]
struct AdamWState {
    step: u64,
    beta1_power: f32,
    beta2_power: f32,
    input_hidden_m: Vec<f32>,
    input_hidden_v: Vec<f32>,
    hidden_bias_m: Vec<f32>,
    hidden_bias_v: Vec<f32>,
    trunk_weights_m: Vec<f32>,
    trunk_weights_v: Vec<f32>,
    trunk_biases_m: Vec<f32>,
    trunk_biases_v: Vec<f32>,
    trunk_global_weights_m: Vec<f32>,
    trunk_global_weights_v: Vec<f32>,
    global_hidden_m: Vec<f32>,
    global_hidden_v: Vec<f32>,
    global_bias_m: Vec<f32>,
    global_bias_v: Vec<f32>,
    value_intermediate_hidden_m: Vec<f32>,
    value_intermediate_hidden_v: Vec<f32>,
    value_intermediate_global_m: Vec<f32>,
    value_intermediate_global_v: Vec<f32>,
    value_intermediate_bias_m: Vec<f32>,
    value_intermediate_bias_v: Vec<f32>,
    value_logits_weights_m: Vec<f32>,
    value_logits_weights_v: Vec<f32>,
    value_logits_bias_m: Vec<f32>,
    value_logits_bias_v: Vec<f32>,
    policy_move_hidden_m: Vec<f32>,
    policy_move_hidden_v: Vec<f32>,
    policy_move_global_m: Vec<f32>,
    policy_move_global_v: Vec<f32>,
    policy_move_bias_m: Vec<f32>,
    policy_move_bias_v: Vec<f32>,
}

impl AdamWState {
    fn new(model: &AzNnue) -> Self {
        Self {
            step: 0,
            beta1_power: 1.0,
            beta2_power: 1.0,
            input_hidden_m: vec![0.0; model.input_hidden.len()],
            input_hidden_v: vec![0.0; model.input_hidden.len()],
            hidden_bias_m: vec![0.0; model.hidden_bias.len()],
            hidden_bias_v: vec![0.0; model.hidden_bias.len()],
            trunk_weights_m: vec![0.0; model.trunk_weights.len()],
            trunk_weights_v: vec![0.0; model.trunk_weights.len()],
            trunk_biases_m: vec![0.0; model.trunk_biases.len()],
            trunk_biases_v: vec![0.0; model.trunk_biases.len()],
            trunk_global_weights_m: vec![0.0; model.trunk_global_weights.len()],
            trunk_global_weights_v: vec![0.0; model.trunk_global_weights.len()],
            global_hidden_m: vec![0.0; model.global_hidden.len()],
            global_hidden_v: vec![0.0; model.global_hidden.len()],
            global_bias_m: vec![0.0; model.global_bias.len()],
            global_bias_v: vec![0.0; model.global_bias.len()],
            value_intermediate_hidden_m: vec![0.0; model.value_intermediate_hidden.len()],
            value_intermediate_hidden_v: vec![0.0; model.value_intermediate_hidden.len()],
            value_intermediate_global_m: vec![0.0; model.value_intermediate_global.len()],
            value_intermediate_global_v: vec![0.0; model.value_intermediate_global.len()],
            value_intermediate_bias_m: vec![0.0; model.value_intermediate_bias.len()],
            value_intermediate_bias_v: vec![0.0; model.value_intermediate_bias.len()],
            value_logits_weights_m: vec![0.0; model.value_logits_weights.len()],
            value_logits_weights_v: vec![0.0; model.value_logits_weights.len()],
            value_logits_bias_m: vec![0.0; model.value_logits_bias.len()],
            value_logits_bias_v: vec![0.0; model.value_logits_bias.len()],
            policy_move_hidden_m: vec![0.0; model.policy_move_hidden.len()],
            policy_move_hidden_v: vec![0.0; model.policy_move_hidden.len()],
            policy_move_global_m: vec![0.0; model.policy_move_global.len()],
            policy_move_global_v: vec![0.0; model.policy_move_global.len()],
            policy_move_bias_m: vec![0.0; model.policy_move_bias.len()],
            policy_move_bias_v: vec![0.0; model.policy_move_bias.len()],
        }
    }

    fn matches(&self, model: &AzNnue) -> bool {
        self.input_hidden_m.len() == model.input_hidden.len()
            && self.input_hidden_v.len() == model.input_hidden.len()
            && self.hidden_bias_m.len() == model.hidden_bias.len()
            && self.hidden_bias_v.len() == model.hidden_bias.len()
            && self.trunk_weights_m.len() == model.trunk_weights.len()
            && self.trunk_weights_v.len() == model.trunk_weights.len()
            && self.trunk_biases_m.len() == model.trunk_biases.len()
            && self.trunk_biases_v.len() == model.trunk_biases.len()
            && self.trunk_global_weights_m.len() == model.trunk_global_weights.len()
            && self.trunk_global_weights_v.len() == model.trunk_global_weights.len()
            && self.global_hidden_m.len() == model.global_hidden.len()
            && self.global_hidden_v.len() == model.global_hidden.len()
            && self.global_bias_m.len() == model.global_bias.len()
            && self.global_bias_v.len() == model.global_bias.len()
            && self.value_intermediate_hidden_m.len() == model.value_intermediate_hidden.len()
            && self.value_intermediate_hidden_v.len() == model.value_intermediate_hidden.len()
            && self.value_intermediate_global_m.len() == model.value_intermediate_global.len()
            && self.value_intermediate_global_v.len() == model.value_intermediate_global.len()
            && self.value_intermediate_bias_m.len() == model.value_intermediate_bias.len()
            && self.value_intermediate_bias_v.len() == model.value_intermediate_bias.len()
            && self.value_logits_weights_m.len() == model.value_logits_weights.len()
            && self.value_logits_weights_v.len() == model.value_logits_weights.len()
            && self.value_logits_bias_m.len() == model.value_logits_bias.len()
            && self.value_logits_bias_v.len() == model.value_logits_bias.len()
            && self.policy_move_hidden_m.len() == model.policy_move_hidden.len()
            && self.policy_move_hidden_v.len() == model.policy_move_hidden.len()
            && self.policy_move_global_m.len() == model.policy_move_global.len()
            && self.policy_move_global_v.len() == model.policy_move_global.len()
            && self.policy_move_bias_m.len() == model.policy_move_bias.len()
            && self.policy_move_bias_v.len() == model.policy_move_bias.len()
    }

    fn advance(&mut self) -> (f32, f32) {
        self.step += 1;
        self.beta1_power *= ADAMW_BETA1;
        self.beta2_power *= ADAMW_BETA2;
        (1.0 - self.beta1_power, 1.0 - self.beta2_power)
    }
}

#[derive(Clone, Debug, Default)]
struct AzSelfplayData {
    samples: Vec<AzTrainingSample>,
    games: Vec<Vec<AzTrainingSample>>,
    red_wins: usize,
    black_wins: usize,
    draws: usize,
    plies_total: usize,
    temperature_early_entropy_sum: f32,
    temperature_early_entropy_count: usize,
    temperature_mid_entropy_sum: f32,
    temperature_mid_entropy_count: usize,
    terminal: AzTerminalStats,
}

#[derive(Clone, Debug)]
pub struct AzExperiencePool {
    game_capacity: usize,
    games: VecDeque<Vec<AzTrainingSample>>,
    samples: usize,
}

impl AzExperiencePool {
    pub fn new(game_capacity: usize) -> Self {
        Self {
            game_capacity,
            games: VecDeque::new(),
            samples: 0,
        }
    }

    pub fn game_count(&self) -> usize {
        self.games.len()
    }

    pub fn sample_count(&self) -> usize {
        self.samples
    }

    fn add_games(&mut self, games: Vec<Vec<AzTrainingSample>>) {
        if self.game_capacity == 0 {
            return;
        }
        for game in games.into_iter().filter(|game| !game.is_empty()) {
            self.samples += game.len();
            self.games.push_back(game);
            while self.games.len() > self.game_capacity {
                if let Some(removed) = self.games.pop_front() {
                    self.samples = self.samples.saturating_sub(removed.len());
                }
            }
        }
    }

    fn sample_uniform_games(&self, count: usize, rng: &mut SplitMix64) -> Vec<AzTrainingSample> {
        if self.games.is_empty() || count == 0 {
            return Vec::new();
        }
        let mut samples = Vec::with_capacity(count);
        for _ in 0..count {
            let game_index = (rng.next() as usize) % self.games.len();
            let game = &self.games[game_index];
            if game.is_empty() {
                continue;
            }
            let sample_index = (rng.next() as usize) % game.len();
            samples.push(game[sample_index].clone());
        }
        samples
    }
}

impl AzNnue {
    pub fn random(hidden_size: usize, seed: u64) -> Self {
        Self::random_with_depth(hidden_size, 2, seed)
    }

    pub fn random_with_depth(hidden_size: usize, trunk_depth: usize, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let input_hidden = (0..V4_INPUT_SIZE * hidden_size)
            .map(|_| rng.weight(0.015))
            .collect();
        let hidden_bias = vec![0.0; hidden_size];
        let trunk_weights = (0..trunk_depth * hidden_size * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt()))
            .collect();
        let trunk_biases = vec![0.0; trunk_depth * hidden_size];
        let trunk_global_weights = (0..trunk_depth * hidden_size * GLOBAL_CONTEXT_SIZE)
            .map(|_| rng.weight((2.0 / GLOBAL_CONTEXT_SIZE as f32).sqrt()))
            .collect();
        let global_hidden = (0..GLOBAL_CONTEXT_SIZE * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt()))
            .collect();
        let global_bias = vec![0.0; GLOBAL_CONTEXT_SIZE];
        let value_intermediate_hidden = (0..VALUE_HIDDEN_SIZE * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt()))
            .collect();
        let value_intermediate_global = (0..VALUE_HIDDEN_SIZE * GLOBAL_CONTEXT_SIZE)
            .map(|_| rng.weight((2.0 / GLOBAL_CONTEXT_SIZE as f32).sqrt()))
            .collect();
        let value_intermediate_bias = vec![0.0; VALUE_HIDDEN_SIZE];
        let value_logits_weights = (0..VALUE_LOGITS * VALUE_HIDDEN_SIZE)
            .map(|_| rng.weight((2.0 / VALUE_HIDDEN_SIZE as f32).sqrt()))
            .collect();
        let value_logits_bias = vec![0.0; VALUE_LOGITS];
        let policy_move_hidden = (0..DENSE_MOVE_SPACE * hidden_size)
            .map(|_| rng.weight(0.01))
            .collect();
        let policy_move_global = (0..DENSE_MOVE_SPACE * GLOBAL_CONTEXT_SIZE)
            .map(|_| rng.weight(0.01))
            .collect();
        let policy_move_bias = vec![0.0; DENSE_MOVE_SPACE];
        Self {
            hidden_size,
            trunk_depth,
            input_hidden,
            hidden_bias,
            trunk_weights,
            trunk_biases,
            trunk_global_weights,
            global_hidden,
            global_bias,
            value_intermediate_hidden,
            value_intermediate_global,
            value_intermediate_bias,
            value_logits_weights,
            value_logits_bias,
            policy_move_hidden,
            policy_move_global,
            policy_move_bias,
            optimizer: None,
        }
    }

    pub fn save_text(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let text = format!(
            "format: {AZNNUE_FORMAT}\ninput_size: {V4_INPUT_SIZE}\nhidden_size: {}\ntrunk_depth: {}\ninput_hidden: {}\nhidden_bias: {}\ntrunk_weights: {}\ntrunk_biases: {}\ntrunk_global_weights: {}\nglobal_hidden: {}\nglobal_bias: {}\nvalue_intermediate_hidden: {}\nvalue_intermediate_global: {}\nvalue_intermediate_bias: {}\nvalue_logits_weights: {}\nvalue_logits_bias: {}\npolicy_move_hidden: {}\npolicy_move_global: {}\npolicy_move_bias: {}\n",
            self.hidden_size,
            self.trunk_depth,
            format_floats(&self.input_hidden),
            format_floats(&self.hidden_bias),
            format_floats(&self.trunk_weights),
            format_floats(&self.trunk_biases),
            format_floats(&self.trunk_global_weights),
            format_floats(&self.global_hidden),
            format_floats(&self.global_bias),
            format_floats(&self.value_intermediate_hidden),
            format_floats(&self.value_intermediate_global),
            format_floats(&self.value_intermediate_bias),
            format_floats(&self.value_logits_weights),
            format_floats(&self.value_logits_bias),
            format_floats(&self.policy_move_hidden),
            format_floats(&self.policy_move_global),
            format_floats(&self.policy_move_bias),
        );
        fs::write(path, text)
    }

    pub fn load_text(path: impl AsRef<Path>) -> io::Result<Self> {
        let text = fs::read_to_string(path)?;
        let mut model_format = None;
        let mut input_size = None;
        let mut hidden_size = None;
        let mut trunk_depth = None;
        let mut input_hidden = None;
        let mut hidden_bias = None;
        let mut trunk_weights = None;
        let mut trunk_biases = None;
        let mut trunk_global_weights = None;
        let mut global_hidden = None;
        let mut global_bias = None;
        let mut value_intermediate_hidden = None;
        let mut value_intermediate_global = None;
        let mut value_intermediate_bias = None;
        let mut value_logits_weights = None;
        let mut value_logits_bias = None;
        let mut policy_move_hidden = None;
        let mut policy_move_global = None;
        let mut policy_move_bias = None;

        for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            let value = value.trim();
            match key.trim() {
                "format" => model_format = Some(value.to_string()),
                "input_size" => input_size = value.parse::<usize>().ok(),
                "hidden_size" => hidden_size = value.parse::<usize>().ok(),
                "trunk_depth" => trunk_depth = value.parse::<usize>().ok(),
                "input_hidden" => input_hidden = Some(parse_floats(value)?),
                "hidden_bias" => hidden_bias = Some(parse_floats(value)?),
                "trunk_weights" => trunk_weights = Some(parse_floats(value)?),
                "trunk_biases" => trunk_biases = Some(parse_floats(value)?),
                "trunk_global_weights" => trunk_global_weights = Some(parse_floats(value)?),
                "global_hidden" => global_hidden = Some(parse_floats(value)?),
                "global_bias" => global_bias = Some(parse_floats(value)?),
                "value_intermediate_hidden" => value_intermediate_hidden = Some(parse_floats(value)?),
                "value_intermediate_global" => value_intermediate_global = Some(parse_floats(value)?),
                "value_intermediate_bias" => value_intermediate_bias = Some(parse_floats(value)?),
                "value_logits_weights" => value_logits_weights = Some(parse_floats(value)?),
                "value_logits_bias" => value_logits_bias = Some(parse_floats(value)?),
                "policy_move_hidden" => policy_move_hidden = Some(parse_floats(value)?),
                "policy_move_global" => policy_move_global = Some(parse_floats(value)?),
                "policy_move_bias" => policy_move_bias = Some(parse_floats(value)?),
                _ => {}
            }
        }

        if model_format.as_deref() != Some(AZNNUE_FORMAT) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{AZNNUE_FORMAT} model format required"),
            ));
        }
        if input_size != Some(V4_INPUT_SIZE) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{AZNNUE_FORMAT} requires fixed relative-view v4 input_size"),
            ));
        }
        let hidden_size = hidden_size
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing hidden_size"))?;
        let trunk_depth = trunk_depth
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing trunk_depth"))?;
        let model = Self {
            hidden_size,
            trunk_depth,
            input_hidden: input_hidden.ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing input_hidden")
            })?,
            hidden_bias: hidden_bias
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing hidden_bias"))?,
            trunk_weights: trunk_weights.ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing trunk_weights")
            })?,
            trunk_biases: trunk_biases.ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing trunk_biases")
            })?,
            trunk_global_weights: trunk_global_weights.ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing trunk_global_weights")
            })?,
            global_hidden: global_hidden.ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing global_hidden")
            })?,
            global_bias: global_bias
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing global_bias"))?,
            value_intermediate_hidden: value_intermediate_hidden.ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing value_intermediate_hidden")
            })?,
            value_intermediate_global: value_intermediate_global.ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing value_intermediate_global")
            })?,
            value_intermediate_bias: value_intermediate_bias.ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing value_intermediate_bias")
            })?,
            value_logits_weights: value_logits_weights.ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing value_logits_weights")
            })?,
            value_logits_bias: value_logits_bias.ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing value_logits_bias")
            })?,
            policy_move_hidden: policy_move_hidden.ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing policy_move_hidden")
            })?,
            policy_move_global: policy_move_global.ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing policy_move_global")
            })?,
            policy_move_bias: policy_move_bias.ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing policy_move_bias")
            })?,
            optimizer: None,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn evaluate(
        &self,
        position: &Position,
        history: &[HistoryMove],
        moves: &[Move],
    ) -> (f32, Vec<f32>) {
        let mut scratch = AzEvalScratch::new(self.hidden_size);
        let value = self.evaluate_with_scratch(position, history, moves, &mut scratch);
        let logits = std::mem::take(&mut scratch.logits);
        (value, logits)
    }

    fn evaluate_with_scratch(
        &self,
        position: &Position,
        history: &[HistoryMove],
        moves: &[Move],
        scratch: &mut AzEvalScratch,
    ) -> f32 {
        let features = extract_sparse_features_v4(position, history);
        self.input_embedding_into(&features, &mut scratch.hidden);
        self.global_from_hidden_into(&scratch.hidden, &mut scratch.global);
        self.forward_trunk_into(&mut scratch.hidden, &mut scratch.next, &scratch.global);
        let value = self.value_from_hidden_scratch(scratch);
        let side = position.side_to_move();
        scratch.logits.resize(moves.len(), 0.0);
        for (index, mv) in moves.iter().enumerate() {
            scratch.logits[index] = self.policy_logit_from_hidden_index(
                &scratch.hidden,
                &scratch.global,
                dense_move_index(orient_move(side, *mv)),
            );
        }
        value
    }

    fn input_embedding_into(&self, features: &[usize], hidden: &mut Vec<f32>) {
        hidden.resize(self.hidden_size, 0.0);
        hidden.copy_from_slice(&self.hidden_bias);
        for &feature in features {
            let row =
                &self.input_hidden[feature * self.hidden_size..(feature + 1) * self.hidden_size];
            for idx in 0..self.hidden_size {
                hidden[idx] += row[idx];
            }
        }
        for value in hidden.iter_mut() {
            *value = value.max(0.0);
        }
    }

    fn global_from_hidden_into(&self, hidden: &[f32], global: &mut Vec<f32>) {
        global.resize(GLOBAL_CONTEXT_SIZE, 0.0);
        global.copy_from_slice(&self.global_bias);
        for out in 0..GLOBAL_CONTEXT_SIZE {
            let row = &self.global_hidden[out * self.hidden_size..(out + 1) * self.hidden_size];
            for idx in 0..self.hidden_size {
                global[out] += hidden[idx] * row[idx];
            }
            global[out] = global[out].max(0.0);
        }
    }

    fn forward_trunk_into(
        &self,
        hidden: &mut Vec<f32>,
        next: &mut Vec<f32>,
        global: &[f32],
    ) {
        next.resize(self.hidden_size, 0.0);
        for layer in 0..self.trunk_depth {
            let weight_offset = layer * self.hidden_size * self.hidden_size;
            let global_weight_offset = layer * self.hidden_size * GLOBAL_CONTEXT_SIZE;
            let bias_offset = layer * self.hidden_size;
            for out in 0..self.hidden_size {
                let mut value = self.trunk_biases[bias_offset + out];
                let row = &self.trunk_weights[weight_offset + out * self.hidden_size
                    ..weight_offset + (out + 1) * self.hidden_size];
                for idx in 0..self.hidden_size {
                    value += row[idx] * hidden[idx];
                }
                let grow = &self.trunk_global_weights
                    [global_weight_offset + out * GLOBAL_CONTEXT_SIZE
                        ..global_weight_offset + (out + 1) * GLOBAL_CONTEXT_SIZE];
                for k in 0..GLOBAL_CONTEXT_SIZE {
                    value += grow[k] * global[k];
                }
                next[out] = hidden[out] + RESIDUAL_TRUNK_SCALE * value.max(0.0);
            }
            std::mem::swap(hidden, next);
        }
    }

    fn value_from_hidden_scratch(&self, scratch: &mut AzEvalScratch) -> f32 {
        scratch
            .value_intermediate
            .copy_from_slice(&self.value_intermediate_bias);
        for j in 0..VALUE_HIDDEN_SIZE {
            let h_row = &self.value_intermediate_hidden
                [j * self.hidden_size..(j + 1) * self.hidden_size];
            for i in 0..self.hidden_size {
                scratch.value_intermediate[j] += scratch.hidden[i] * h_row[i];
            }
            let g_row = &self.value_intermediate_global
                [j * GLOBAL_CONTEXT_SIZE..(j + 1) * GLOBAL_CONTEXT_SIZE];
            for k in 0..GLOBAL_CONTEXT_SIZE {
                scratch.value_intermediate[j] += scratch.global[k] * g_row[k];
            }
            scratch.value_intermediate[j] = scratch.value_intermediate[j].max(0.0);
        }
        scratch.value_logits.copy_from_slice(&self.value_logits_bias);
        for out in 0..VALUE_LOGITS {
            let row = &self.value_logits_weights
                [out * VALUE_HIDDEN_SIZE..(out + 1) * VALUE_HIDDEN_SIZE];
            for j in 0..VALUE_HIDDEN_SIZE {
                scratch.value_logits[out] += scratch.value_intermediate[j] * row[j];
            }
        }
        scalar_value_from_logits(&scratch.value_logits).0
    }

    fn validate(&self) -> io::Result<()> {
        if self.input_hidden.len() != V4_INPUT_SIZE * self.hidden_size
            || self.hidden_bias.len() != self.hidden_size
            || self.trunk_weights.len() != self.trunk_depth * self.hidden_size * self.hidden_size
            || self.trunk_biases.len() != self.trunk_depth * self.hidden_size
            || self.trunk_global_weights.len()
                != self.trunk_depth * self.hidden_size * GLOBAL_CONTEXT_SIZE
            || self.global_hidden.len() != GLOBAL_CONTEXT_SIZE * self.hidden_size
            || self.global_bias.len() != GLOBAL_CONTEXT_SIZE
            || self.value_intermediate_hidden.len() != VALUE_HIDDEN_SIZE * self.hidden_size
            || self.value_intermediate_global.len() != VALUE_HIDDEN_SIZE * GLOBAL_CONTEXT_SIZE
            || self.value_intermediate_bias.len() != VALUE_HIDDEN_SIZE
            || self.value_logits_weights.len() != VALUE_LOGITS * VALUE_HIDDEN_SIZE
            || self.value_logits_bias.len() != VALUE_LOGITS
            || self.policy_move_hidden.len() != DENSE_MOVE_SPACE * self.hidden_size
            || self.policy_move_global.len() != DENSE_MOVE_SPACE * GLOBAL_CONTEXT_SIZE
            || self.policy_move_bias.len() != DENSE_MOVE_SPACE
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "aznnue vector length mismatch",
            ));
        }
        Ok(())
    }

    fn policy_logit_from_hidden_index(
        &self,
        hidden: &[f32],
        global: &[f32],
        move_index: usize,
    ) -> f32 {
        let hidden_offset = move_index * self.hidden_size;
        let hidden_row = &self.policy_move_hidden[hidden_offset..hidden_offset + self.hidden_size];
        let global_offset = move_index * GLOBAL_CONTEXT_SIZE;
        let global_row =
            &self.policy_move_global[global_offset..global_offset + GLOBAL_CONTEXT_SIZE];
        self.policy_move_bias[move_index]
            + dot_product(hidden, hidden_row)
            + dot_product(global, global_row)
    }
}

pub fn selfplay_train_iteration(model: &mut AzNnue, config: &AzLoopConfig) -> AzLoopReport {
    selfplay_train_iteration_with_pool(model, config, None)
}

pub fn benchmark_training(
    model: &mut AzNnue,
    sample_count: usize,
    epochs: usize,
    batch_size: usize,
    lr: f32,
    seed: u64,
) -> AzTrainBenchmark {
    let mut rng = SplitMix64::new(seed);
    let mut samples = Vec::with_capacity(sample_count);
    for index in 0..sample_count {
        let feature_count = 24 + (rng.next() as usize % 16);
        let mut features = Vec::with_capacity(feature_count);
        for _ in 0..feature_count {
            features.push((rng.next() as usize) % V4_INPUT_SIZE);
        }
        features.sort_unstable();
        features.dedup();

        let value = rng.unit_f32() * 2.0 - 1.0;
        let move_count = 12 + (rng.next() as usize % 24);
        let mut move_indices = Vec::with_capacity(move_count);
        while move_indices.len() < move_count {
            let candidate = (rng.next() as usize) % DENSE_MOVE_SPACE;
            if !move_indices.contains(&candidate) {
                move_indices.push(candidate);
            }
        }
        let mut policy = (0..move_count)
            .map(|_| rng.unit_f32().max(1e-6))
            .collect::<Vec<_>>();
        let policy_sum = policy.iter().sum::<f32>().max(1e-6);
        for value in &mut policy {
            *value /= policy_sum;
        }
        samples.push(AzTrainingSample {
            features,
            move_indices,
            policy,
            value,
            side_sign: 1.0,
            reward: 0.0,
            discount: 0.0,
            bootstrap_value: 0.0,
        });
        if index + 1 == sample_count {
            break;
        }
    }
    let stats = train_samples(model, &samples, epochs, lr, batch_size, &mut rng);
    AzTrainBenchmark {
        loss: stats.loss,
        value_loss: stats.value_loss,
        policy_ce: stats.policy_ce,
    }
}

pub fn selfplay_train_iteration_with_pool(
    model: &mut AzNnue,
    config: &AzLoopConfig,
    pool: Option<&mut AzExperiencePool>,
) -> AzLoopReport {
    let started = Instant::now();
    let selfplay_started = Instant::now();
    let data = generate_selfplay_data(model, config);
    let selfplay_seconds = selfplay_started.elapsed().as_secs_f32();
    let mut rng = SplitMix64::new(config.seed ^ 0xA5A5_5A5A_D3C3_B4B4);
    let generated_samples = data.samples.len();
    let (train_data, pool_games, pool_samples) = if let Some(pool) = pool {
        pool.add_games(data.games.clone());
        let train_count = if config.replay_samples == 0 {
            generated_samples
        } else {
            config.replay_samples
        };
        (
            pool.sample_uniform_games(train_count, &mut rng),
            pool.game_count(),
            pool.sample_count(),
        )
    } else {
        (data.samples.clone(), 0, 0)
    };
    let train_started = Instant::now();
    let stats = train_samples(
        model,
        &train_data,
        config.epochs,
        config.lr,
        config.batch_size,
        &mut rng,
    );
    let train_seconds = train_started.elapsed().as_secs_f32();
    let total_seconds = started.elapsed().as_secs_f32();
    AzLoopReport {
        games: config.games,
        samples: generated_samples,
        red_wins: data.red_wins,
        black_wins: data.black_wins,
        draws: data.draws,
        avg_plies: if config.games == 0 {
            0.0
        } else {
            data.plies_total as f32 / config.games as f32
        },
        loss: stats.loss,
        value_loss: stats.value_loss,
        policy_ce: stats.policy_ce,
        temperature_early_entropy: data.temperature_early_entropy_sum
            / data.temperature_early_entropy_count.max(1) as f32,
        temperature_mid_entropy: data.temperature_mid_entropy_sum
            / data.temperature_mid_entropy_count.max(1) as f32,
        selfplay_seconds,
        train_seconds,
        total_seconds,
        games_per_second: config.games as f32 / selfplay_seconds.max(1e-6),
        samples_per_second: generated_samples as f32 / selfplay_seconds.max(1e-6),
        train_samples_per_second: (train_data.len() * config.epochs) as f32 / train_seconds.max(1e-6),
        train_samples: train_data.len(),
        pool_games,
        pool_samples,
        terminal_no_legal_moves: data.terminal.no_legal_moves,
        terminal_red_general_missing: data.terminal.red_general_missing,
        terminal_black_general_missing: data.terminal.black_general_missing,
        terminal_rule_draw: data.terminal.rule_draw,
        terminal_rule_draw_halfmove120: data.terminal.rule_draw_halfmove120,
        terminal_rule_draw_repetition: data.terminal.rule_draw_repetition,
        terminal_rule_draw_mutual_long_check: data.terminal.rule_draw_mutual_long_check,
        terminal_rule_draw_mutual_long_chase: data.terminal.rule_draw_mutual_long_chase,
        terminal_rule_win_red: data.terminal.rule_win_red,
        terminal_rule_win_black: data.terminal.rule_win_black,
        terminal_max_plies: data.terminal.max_plies,
    }
}

fn generate_selfplay_data(model: &AzNnue, config: &AzLoopConfig) -> AzSelfplayData {
    let workers = config.workers.max(1).min(config.games.max(1));
    if workers == 1 || config.games <= 1 {
        return generate_selfplay_chunk(model, config);
    }

    let shared_model = Arc::new(model.clone());
    let mut handles = Vec::with_capacity(workers);
    for worker in 0..workers {
        let games = config.games / workers + usize::from(worker < config.games % workers);
        if games == 0 {
            continue;
        }

        let worker_model = Arc::clone(&shared_model);
        let mut worker_config = config.clone();
        worker_config.games = games;
        worker_config.workers = 1;
        worker_config.seed ^= (worker as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        handles.push(thread::spawn(move || {
            generate_selfplay_chunk(&worker_model, &worker_config)
        }));
    }

    let mut merged = AzSelfplayData::default();
    for handle in handles {
        let chunk = handle.join().expect("selfplay worker panicked");
        merged.samples.extend(chunk.samples);
        merged.games.extend(chunk.games);
        merged.red_wins += chunk.red_wins;
        merged.black_wins += chunk.black_wins;
        merged.draws += chunk.draws;
        merged.plies_total += chunk.plies_total;
        merged.temperature_early_entropy_sum += chunk.temperature_early_entropy_sum;
        merged.temperature_early_entropy_count += chunk.temperature_early_entropy_count;
        merged.temperature_mid_entropy_sum += chunk.temperature_mid_entropy_sum;
        merged.temperature_mid_entropy_count += chunk.temperature_mid_entropy_count;
        merged.terminal.add_assign(&chunk.terminal);
    }
    merged
}

fn generate_selfplay_chunk(model: &AzNnue, config: &AzLoopConfig) -> AzSelfplayData {
    let mut rng = SplitMix64::new(config.seed);
    let mut samples = Vec::new();
    let mut red_wins = 0usize;
    let mut black_wins = 0usize;
    let mut draws = 0usize;
    let mut plies_total = 0usize;
    let mut games = Vec::with_capacity(config.games);
    let mut temperature_early_entropy_sum = 0.0f32;
    let mut temperature_early_entropy_count = 0usize;
    let mut temperature_mid_entropy_sum = 0.0f32;
    let mut temperature_mid_entropy_count = 0usize;
    let mut terminal = AzTerminalStats::default();

    for game_index in 0..config.games {
        let mut position = Position::startpos();
        let mut history = Vec::new();
        let mut rule_history = position.initial_rule_history();
        let mut game_samples = Vec::new();
        let mut result = None;
        let mut plies = 0usize;

        for ply in 0..config.max_plies {
            plies = ply + 1;
            let legal = position.legal_moves_with_rules(&rule_history);
            if legal.is_empty() {
                result = Some(if position.side_to_move() == Color::Red {
                    -1.0
                } else {
                    1.0
                });
                terminal.no_legal_moves += 1;
                finalize_last_transition(&mut game_samples, result.unwrap_or(0.0));
                break;
            }

            let search = gumbel_search_with_history_and_rules(
                &position,
                &history,
                Some(rule_history.clone()),
                Some(legal),
                model,
                AzSearchLimits {
                    simulations: config.simulations,
                    top_k: config.top_k,
                    seed: rng.next() ^ ((game_index as u64) << 32) ^ ply as u64,
                    gumbel_scale: config.gumbel_scale,
                    workers: 1,
                },
            );
            let entropy = policy_entropy(&search.candidates);
            let split = config.temperature_decay_plies.max(2).div_ceil(2);
            if ply < split {
                temperature_early_entropy_sum += entropy;
                temperature_early_entropy_count += 1;
            } else if ply < config.temperature_decay_plies.max(split + 1) {
                temperature_mid_entropy_sum += entropy;
                temperature_mid_entropy_count += 1;
            }
            let temperature = temperature_for_ply(config, ply);
            let Some(mv) = choose_selfplay_move(&search.candidates, temperature, &mut rng) else {
                result = Some(0.0);
                break;
            };
            game_samples.push(make_training_sample(
                &position,
                &history,
                &search.candidates,
                search.value_cp,
                rng.unit_f32() < config.mirror_probability.clamp(0.0, 1.0),
            ));
            append_history(&mut history, &position, mv);
            rule_history.push(position.rule_history_entry_after_move(mv));
            position.make_move(mv);

            if !position.has_general(Color::Red) {
                result = Some(-1.0);
                terminal.red_general_missing += 1;
                finalize_last_transition(&mut game_samples, -1.0);
                break;
            }
            if !position.has_general(Color::Black) {
                result = Some(1.0);
                terminal.black_general_missing += 1;
                finalize_last_transition(&mut game_samples, 1.0);
                break;
            }
            if let Some(rule_outcome) = position.rule_outcome_with_history(&rule_history) {
                result = Some(match rule_outcome {
                    RuleOutcome::Draw(_) => 0.0,
                    RuleOutcome::Win(Color::Red) => 1.0,
                    RuleOutcome::Win(Color::Black) => -1.0,
                });
                match rule_outcome {
                    RuleOutcome::Draw(reason) => {
                        terminal.rule_draw += 1;
                        match reason {
                            RuleDrawReason::Halfmove120 => terminal.rule_draw_halfmove120 += 1,
                            RuleDrawReason::Repetition => terminal.rule_draw_repetition += 1,
                            RuleDrawReason::MutualLongCheck => {
                                terminal.rule_draw_mutual_long_check += 1
                            }
                            RuleDrawReason::MutualLongChase => {
                                terminal.rule_draw_mutual_long_chase += 1
                            }
                        }
                    }
                    RuleOutcome::Win(Color::Red) => terminal.rule_win_red += 1,
                    RuleOutcome::Win(Color::Black) => terminal.rule_win_black += 1,
                }
                finalize_last_transition(&mut game_samples, result.unwrap_or(0.0));
                break;
            }
        }
        if result.is_none() {
            terminal.max_plies += 1;
        }

        let result: f32 = result.unwrap_or(0.0);
        match result.total_cmp(&0.0) {
            std::cmp::Ordering::Greater => red_wins += 1,
            std::cmp::Ordering::Less => black_wins += 1,
            std::cmp::Ordering::Equal => draws += 1,
        }
        plies_total += plies;

        assign_lambda_targets(&mut game_samples, result, config.td_lambda);
        samples.extend(game_samples.clone());
        games.push(game_samples);
    }

    AzSelfplayData {
        samples,
        games,
        red_wins,
        black_wins,
        draws,
        plies_total,
        temperature_early_entropy_sum,
        temperature_early_entropy_count,
        temperature_mid_entropy_sum,
        temperature_mid_entropy_count,
        terminal,
    }
}

fn finalize_last_transition(game_samples: &mut [AzTrainingSample], game_result: f32) {
    let Some(last) = game_samples.last_mut() else {
        return;
    };
    last.reward = game_result * last.side_sign;
    last.discount = 0.0;
}

pub fn gumbel_search(
    position: &Position,
    model: &AzNnue,
    limits: AzSearchLimits,
) -> AzSearchResult {
    gumbel_search_with_history(position, &[], model, limits)
}

pub fn gumbel_search_with_history(
    position: &Position,
    history: &[HistoryMove],
    model: &AzNnue,
    limits: AzSearchLimits,
) -> AzSearchResult {
    gumbel_search_with_history_and_rules(position, history, None, None, model, limits)
}

pub fn gumbel_search_with_history_and_root_moves(
    position: &Position,
    history: &[HistoryMove],
    root_moves: Option<Vec<Move>>,
    model: &AzNnue,
    limits: AzSearchLimits,
) -> AzSearchResult {
    gumbel_search_with_history_and_rules(position, history, None, root_moves, model, limits)
}

pub fn gumbel_search_with_history_and_rules(
    position: &Position,
    history: &[HistoryMove],
    rule_history: Option<Vec<RuleHistoryEntry>>,
    root_moves: Option<Vec<Move>>,
    model: &AzNnue,
    limits: AzSearchLimits,
) -> AzSearchResult {
    let mut tree = AzTree::new(
        position.clone(),
        truncate_history(history),
        rule_history,
        root_moves,
        model,
    );
    let root = tree.root;
    tree.expand(root);
    if tree.nodes[root].children.is_empty() {
        return AzSearchResult {
            best_move: None,
            value_cp: (tree.nodes[root].value * VALUE_SCALE_CP) as i32,
            simulations: 0,
            candidates: Vec::new(),
        };
    }

    let num_considered = limits.top_k.max(1).min(tree.nodes[root].children.len());
    let considered_visits = considered_visit_sequence(num_considered, limits.simulations);
    let mut used = 0usize;
    for considered_visit in considered_visits {
        let child_index =
            tree.select_root_child(root, limits.seed, limits.gumbel_scale, considered_visit);
        tree.simulate_child(root, child_index);
        used += 1;
    }

    let searched_value = if tree.nodes[root].visits > 0 {
        tree.nodes[root].value_sum / tree.nodes[root].visits as f32
    } else {
        tree.nodes[root].value
    };

    let policy = tree.improved_policy(root);
    let mut candidates = tree.nodes[root]
        .children
        .iter()
        .enumerate()
        .map(|(child_index, child)| AzCandidate {
            mv: child.mv,
            visits: child.visits,
            q: child.q(),
            prior: child.prior,
            policy: policy[child_index],
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        right
            .policy
            .total_cmp(&left.policy)
            .then_with(|| right.visits.cmp(&left.visits))
            .then_with(|| right.q.total_cmp(&left.q))
    });
    let best_move = tree
        .select_root_action(root, limits.seed, limits.gumbel_scale)
        .map(|child_index| tree.nodes[root].children[child_index].mv)
        .or_else(|| candidates.first().map(|candidate| candidate.mv));
    AzSearchResult {
        best_move,
        value_cp: (searched_value * VALUE_SCALE_CP) as i32,
        simulations: used,
        candidates,
    }
}

struct AzTree<'a> {
    nodes: Vec<AzNode>,
    model: &'a AzNnue,
    root_rule_history: Option<Vec<RuleHistoryEntry>>,
    root_moves: Option<Vec<Move>>,
    root: usize,
    eval_scratch: AzEvalScratch,
}

struct AzNode {
    position: Position,
    history: Vec<HistoryMove>,
    rule_history: Option<Vec<RuleHistoryEntry>>,
    children: Vec<AzChild>,
    visits: u32,
    value_sum: f32,
    value: f32,
    expanded: bool,
}

struct AzChild {
    mv: Move,
    prior: f32,
    prior_logit: f32,
    visits: u32,
    value_sum: f32,
    child: Option<usize>,
}

impl AzChild {
    fn q(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }
}

impl<'a> AzTree<'a> {
    fn new(
        position: Position,
        history: Vec<HistoryMove>,
        rule_history: Option<Vec<RuleHistoryEntry>>,
        root_moves: Option<Vec<Move>>,
        model: &'a AzNnue,
    ) -> Self {
        let eval_scratch = AzEvalScratch::new(model.hidden_size);
        Self {
            nodes: vec![AzNode {
                position,
                history,
                rule_history: None,
                children: Vec::new(),
                visits: 0,
                value_sum: 0.0,
                value: 0.0,
                expanded: false,
            }],
            model,
            root_rule_history: rule_history,
            root_moves,
            root: 0,
            eval_scratch,
        }
    }

    fn effective_rule_history(&self, node_index: usize) -> Option<&Vec<RuleHistoryEntry>> {
        if node_index == self.root {
            self.root_rule_history.as_ref()
        } else {
            self.nodes[node_index].rule_history.as_ref()
        }
    }

    fn expand(&mut self, node_index: usize) -> f32 {
        if self.nodes[node_index].expanded {
            return self.nodes[node_index].value;
        }
        if let Some(value) = terminal_value(&self.nodes[node_index].position) {
            self.nodes[node_index].children.clear();
            self.nodes[node_index].value = value;
            self.nodes[node_index].expanded = true;
            return value;
        }
        if let Some(rule_history) = self.effective_rule_history(node_index) {
            if let Some(outcome) = self.nodes[node_index]
                .position
                .rule_outcome_with_history(rule_history)
            {
                self.nodes[node_index].children.clear();
                self.nodes[node_index].value = match outcome {
                    RuleOutcome::Draw(_) => 0.0,
                    RuleOutcome::Win(color) => {
                        if color == self.nodes[node_index].position.side_to_move() {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                };
                self.nodes[node_index].expanded = true;
                return self.nodes[node_index].value;
            }
        }
        let moves = if node_index == self.root {
            self.root_moves.clone().unwrap_or_else(|| {
                self.root_rule_history.as_ref().map_or_else(
                    || self.nodes[node_index].position.legal_moves(),
                    |history| {
                        self.nodes[node_index]
                            .position
                            .legal_moves_with_rules(history)
                    },
                )
            })
        } else if let Some(history) = self.nodes[node_index].rule_history.as_ref() {
            self.nodes[node_index]
                .position
                .legal_moves_with_rules(history)
        } else {
            self.nodes[node_index].position.legal_moves()
        };
        if moves.is_empty() {
            self.nodes[node_index].children.clear();
            self.nodes[node_index].value = -1.0;
            self.nodes[node_index].expanded = true;
            return -1.0;
        }
        let n = moves.len();
        let value = self.model.evaluate_with_scratch(
            &self.nodes[node_index].position,
            &self.nodes[node_index].history,
            &moves,
            &mut self.eval_scratch,
        );
        let children: Vec<AzChild> = {
            let scratch = &mut self.eval_scratch;
            softmax_into(&scratch.logits[..n], &mut scratch.priors);
            let logits = &scratch.logits[..n];
            let priors = &scratch.priors[..n];
            moves
                .into_iter()
                .enumerate()
                .map(|(i, mv)| AzChild {
                    mv,
                    prior: priors[i],
                    prior_logit: logits[i],
                    visits: 0,
                    value_sum: 0.0,
                    child: None,
                })
                .collect()
        };
        self.nodes[node_index].children = children;
        self.nodes[node_index].value = value;
        self.nodes[node_index].expanded = true;
        value
    }

    fn simulate(&mut self, node_index: usize) -> f32 {
        if !self.nodes[node_index].expanded {
            let value = self.expand(node_index);
            self.nodes[node_index].visits += 1;
            self.nodes[node_index].value_sum += value;
            return value;
        }
        if self.nodes[node_index].children.is_empty() {
            self.nodes[node_index].visits += 1;
            self.nodes[node_index].value_sum += self.nodes[node_index].value;
            return self.nodes[node_index].value;
        }
        let child_index = self.select_child(node_index);
        self.simulate_child(node_index, child_index)
    }

    fn simulate_child(&mut self, node_index: usize, child_index: usize) -> f32 {
        let child_node =
            if let Some(child_node) = self.nodes[node_index].children[child_index].child {
                child_node
            } else {
                let mv = self.nodes[node_index].children[child_index].mv;
                let mut child_position = self.nodes[node_index].position.clone();
                let mut child_history = self.nodes[node_index].history.clone();
                let mut child_rule_history = if node_index == self.root {
                    self.root_rule_history.clone()
                } else {
                    self.nodes[node_index].rule_history.clone()
                };
                append_history(&mut child_history, &child_position, mv);
                if let Some(rule_history) = child_rule_history.as_mut() {
                    rule_history.push(child_position.rule_history_entry_after_move(mv));
                }
                child_position.make_move(mv);
                let child_node = self.nodes.len();
                self.nodes.push(AzNode {
                    position: child_position,
                    history: child_history,
                    rule_history: child_rule_history,
                    children: Vec::new(),
                    visits: 0,
                    value_sum: 0.0,
                    value: 0.0,
                    expanded: false,
                });
                self.nodes[node_index].children[child_index].child = Some(child_node);
                child_node
            };
        let child_value = self.simulate(child_node);
        let value = -child_value;
        let child = &mut self.nodes[node_index].children[child_index];
        child.visits += 1;
        child.value_sum += value;
        self.nodes[node_index].visits += 1;
        self.nodes[node_index].value_sum += value;
        value
    }

    fn select_child(&self, node_index: usize) -> usize {
        let policy = self.improved_policy(node_index);
        let total_visits = self.nodes[node_index]
            .children
            .iter()
            .map(|child| child.visits)
            .sum::<u32>() as f32;
        let mut best = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (index, child) in self.nodes[node_index].children.iter().enumerate() {
            let visited_fraction = child.visits as f32 / (1.0 + total_visits);
            let score = policy[index] - visited_fraction;
            if score > best_score {
                best_score = score;
                best = index;
            }
        }
        best
    }

    fn select_root_child(
        &self,
        node_index: usize,
        seed: u64,
        gumbel_scale: f32,
        considered_visit: u32,
    ) -> usize {
        self.best_scored_child(node_index, seed, gumbel_scale, |child| {
            child.visits == considered_visit
        })
        .unwrap_or_else(|| self.select_child(node_index))
    }

    fn select_root_action(&self, node_index: usize, seed: u64, gumbel_scale: f32) -> Option<usize> {
        let considered_visit = self.nodes[node_index]
            .children
            .iter()
            .map(|child| child.visits)
            .max()?;
        self.best_scored_child(node_index, seed, gumbel_scale, |child| {
            child.visits == considered_visit
        })
    }

    fn best_scored_child(
        &self,
        node_index: usize,
        seed: u64,
        gumbel_scale: f32,
        mut is_considered: impl FnMut(&AzChild) -> bool,
    ) -> Option<usize> {
        let completed_q = self.completed_qvalues(node_index);
        let hash = self.nodes[node_index].position.hash() ^ seed;
        let max_prior_logit = self.nodes[node_index]
            .children
            .iter()
            .map(|child| child.prior_logit)
            .fold(f32::NEG_INFINITY, f32::max);
        self.nodes[node_index]
            .children
            .iter()
            .enumerate()
            .filter(|(_, child)| is_considered(child))
            .map(|(child_index, child)| {
                let score = (child.prior_logit - max_prior_logit).max(-1e9)
                    + completed_q[child_index]
                    + gumbel_scale * deterministic_gumbel(hash, child.mv, child_index as u64);
                (score, child_index)
            })
            .max_by(|left, right| {
                left.0
                    .total_cmp(&right.0)
                    .then_with(|| right.1.cmp(&left.1))
            })
            .map(|(_, child_index)| child_index)
    }

    fn improved_policy(&self, node_index: usize) -> Vec<f32> {
        let completed_q = self.completed_qvalues(node_index);
        let logits = self.nodes[node_index]
            .children
            .iter()
            .enumerate()
            .map(|(index, child)| child.prior_logit + completed_q[index])
            .collect::<Vec<_>>();
        softmax(&logits)
    }

    fn completed_qvalues(&self, node_index: usize) -> Vec<f32> {
        let children = &self.nodes[node_index].children;
        if children.is_empty() {
            return Vec::new();
        }

        let total_visits = children.iter().map(|child| child.visits).sum::<u32>();
        let max_visits = children
            .iter()
            .map(|child| child.visits)
            .max()
            .unwrap_or_default();
        let visited_policy_sum = children
            .iter()
            .filter(|child| child.visits > 0)
            .map(|child| child.prior)
            .sum::<f32>();
        let weighted_q = if visited_policy_sum > 0.0 {
            children
                .iter()
                .filter(|child| child.visits > 0)
                .map(|child| child.prior * child.q())
                .sum::<f32>()
                / visited_policy_sum
        } else {
            self.nodes[node_index].value
        };
        let mixed_value = (self.nodes[node_index].value + total_visits as f32 * weighted_q)
            / (total_visits as f32 + 1.0);

        let mut qvalues = children
            .iter()
            .map(|child| {
                if child.visits > 0 {
                    child.q()
                } else {
                    mixed_value
                }
            })
            .collect::<Vec<_>>();
        normalize_completed_q(&mut qvalues, max_visits);
        qvalues
    }
}

fn normalize_completed_q(qvalues: &mut [f32], total_visits: u32) {
    let min_value = qvalues.iter().copied().fold(f32::INFINITY, f32::min);
    let max_value = qvalues.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let denominator = (max_value - min_value).max(1e-8);
    for value in qvalues.iter_mut() {
        *value = (*value - min_value) / denominator;
    }

    let maxvisit_scale =
        (COMPLETED_Q_MAXVISIT_INIT + total_visits as f32) * COMPLETED_Q_VALUE_SCALE;
    for value in qvalues.iter_mut() {
        *value *= maxvisit_scale;
    }
}

fn considered_visit_sequence(
    max_num_considered_actions: usize,
    num_simulations: usize,
) -> Vec<u32> {
    if max_num_considered_actions == 0 || num_simulations == 0 {
        return Vec::new();
    }
    if max_num_considered_actions <= 1 {
        return (0..num_simulations as u32).collect();
    }

    let log2max = (max_num_considered_actions as f32).log2().ceil() as usize;
    let mut sequence = Vec::with_capacity(num_simulations);
    let mut visits = vec![0u32; max_num_considered_actions];
    let mut num_considered = max_num_considered_actions;
    while sequence.len() < num_simulations {
        let extra_visits = (num_simulations / (log2max * num_considered)).max(1);
        for _ in 0..extra_visits {
            sequence.extend_from_slice(&visits[..num_considered]);
            for visit in &mut visits[..num_considered] {
                *visit += 1;
            }
            if sequence.len() >= num_simulations {
                break;
            }
        }
        num_considered = (num_considered / 2).max(2);
    }
    sequence.truncate(num_simulations);
    sequence
}

fn terminal_value(position: &Position) -> Option<f32> {
    if !position.has_general(Color::Red) {
        return Some(if position.side_to_move() == Color::Red {
            -1.0
        } else {
            1.0
        });
    }
    if !position.has_general(Color::Black) {
        return Some(if position.side_to_move() == Color::Black {
            -1.0
        } else {
            1.0
        });
    }
    None
}

fn make_training_sample(
    position: &Position,
    history: &[HistoryMove],
    candidates: &[AzCandidate],
    value_cp: i32,
    mirror_file: bool,
) -> AzTrainingSample {
    let total_policy = candidates
        .iter()
        .map(|candidate| candidate.policy.max(0.0))
        .sum::<f32>()
        .max(1.0);
    let side = position.side_to_move();
    let mut features = extract_sparse_features_v4(position, history);
    let mut moves = candidates
        .iter()
        .map(|candidate| orient_move(side, candidate.mv))
        .collect::<Vec<_>>();
    if mirror_file {
        mirror_sparse_features_file(&mut features);
        for mv in &mut moves {
            *mv = mirror_file_move(*mv);
        }
    }
    let move_indices = moves.iter().copied().map(dense_move_index).collect();
    AzTrainingSample {
        features,
        move_indices,
        policy: candidates
            .iter()
            .map(|candidate| candidate.policy.max(0.0) / total_policy)
            .collect(),
        value: 0.0,
        side_sign: if position.side_to_move() == Color::Red {
            1.0
        } else {
            -1.0
        },
        reward: 0.0,
        discount: -1.0,
        bootstrap_value: (value_cp as f32 / VALUE_SCALE_CP).clamp(-1.0, 1.0),
    }
}

fn assign_lambda_targets(samples: &mut [AzTrainingSample], game_result: f32, td_lambda: f32) {
    let lambda = td_lambda.clamp(0.0, 1.0);
    if let Some(last) = samples.last_mut() {
        if last.discount < 0.0 && game_result == 0.0 {
            last.discount = 0.0;
            last.reward = 0.0;
        }
    }

    let mut carry = 0.0f32;
    for index in (0..samples.len()).rev() {
        let next_bootstrap = samples
            .get(index + 1)
            .map(|sample| sample.bootstrap_value)
            .unwrap_or(0.0);
        let blended = (1.0 - lambda) * next_bootstrap + lambda * carry;
        let target = samples[index].reward + samples[index].discount * blended;
        samples[index].value = target.clamp(-1.0, 1.0);
        carry = samples[index].value;
    }
}

fn append_history(history: &mut Vec<HistoryMove>, position: &Position, mv: Move) {
    if let Some(piece) = position.piece_at(mv.from as usize) {
        history.push(HistoryMove { piece, mv });
        let overflow = history.len().saturating_sub(crate::nnue::HISTORY_PLIES);
        if overflow > 0 {
            history.drain(0..overflow);
        }
    }
}

fn truncate_history(history: &[HistoryMove]) -> Vec<HistoryMove> {
    history
        .iter()
        .rev()
        .take(crate::nnue::HISTORY_PLIES)
        .copied()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

fn temperature_for_ply(config: &AzLoopConfig, ply: usize) -> f32 {
    if config.temperature_decay_plies == 0 || ply >= config.temperature_decay_plies {
        return config.temperature_end;
    }
    let progress = ply as f32 / config.temperature_decay_plies as f32;
    config.temperature_start + (config.temperature_end - config.temperature_start) * progress
}

fn choose_selfplay_move(
    candidates: &[AzCandidate],
    temperature: f32,
    rng: &mut SplitMix64,
) -> Option<Move> {
    if temperature <= 1e-6 {
        return candidates
            .iter()
            .max_by(|left, right| {
                left.policy
                    .total_cmp(&right.policy)
                    .then_with(|| left.visits.cmp(&right.visits))
            })
            .map(|candidate| candidate.mv);
    }

    let inv_temperature = 1.0 / temperature.max(1e-3);
    let weights = candidates
        .iter()
        .map(|candidate| candidate.policy.max(1e-9).powf(inv_temperature))
        .collect::<Vec<_>>();
    let total = candidates
        .iter()
        .zip(&weights)
        .map(|(_, weight)| *weight)
        .sum::<f32>();
    if total <= 0.0 {
        return candidates.first().map(|candidate| candidate.mv);
    }

    let mut ticket = rng.unit_f32() * total;
    for (candidate, weight) in candidates.iter().zip(weights) {
        if ticket < weight {
            return Some(candidate.mv);
        }
        ticket -= weight;
    }
    candidates.first().map(|candidate| candidate.mv)
}

fn policy_entropy(candidates: &[AzCandidate]) -> f32 {
    let total = candidates
        .iter()
        .map(|candidate| candidate.policy.max(0.0))
        .sum::<f32>();
    if total <= 0.0 {
        return 0.0;
    }
    candidates
        .iter()
        .map(|candidate| candidate.policy.max(0.0) / total)
        .filter(|probability| *probability > 1e-9)
        .map(|probability| -probability * probability.ln())
        .sum()
}

pub fn play_arena_games(
    candidate: &AzNnue,
    baseline: &AzNnue,
    simulations: usize,
    top_k: usize,
    max_plies: usize,
    games_as_red: usize,
    games_as_black: usize,
    seed: u64,
) -> AzArenaReport {
    let mut report = AzArenaReport::default();
    let mut game_seed = seed;
    for _ in 0..games_as_red {
        let outcome = play_arena_game(candidate, baseline, simulations, top_k, max_plies, game_seed);
        match outcome.total_cmp(&0.0) {
            std::cmp::Ordering::Greater => {
                report.wins += 1;
                report.wins_as_red += 1;
            }
            std::cmp::Ordering::Less => {
                report.losses += 1;
                report.losses_as_red += 1;
            }
            std::cmp::Ordering::Equal => report.draws += 1,
        }
        game_seed = game_seed.wrapping_add(1);
    }
    for _ in 0..games_as_black {
        let outcome = play_arena_game(baseline, candidate, simulations, top_k, max_plies, game_seed);
        match outcome.total_cmp(&0.0) {
            std::cmp::Ordering::Greater => {
                report.losses += 1;
                report.losses_as_black += 1;
            }
            std::cmp::Ordering::Less => {
                report.wins += 1;
                report.wins_as_black += 1;
            }
            std::cmp::Ordering::Equal => report.draws += 1,
        }
        game_seed = game_seed.wrapping_add(1);
    }
    report
}

fn play_arena_game(
    red_model: &AzNnue,
    black_model: &AzNnue,
    simulations: usize,
    top_k: usize,
    max_plies: usize,
    seed: u64,
) -> f32 {
    let mut position = Position::startpos();
    let mut history = Vec::new();
    let mut rule_history = position.initial_rule_history();
    for ply in 0..max_plies {
        let legal = position.legal_moves_with_rules(&rule_history);
        if legal.is_empty() {
            return if position.side_to_move() == Color::Red {
                -1.0
            } else {
                1.0
            };
        }
        let model = if position.side_to_move() == Color::Red {
            red_model
        } else {
            black_model
        };
        let result = gumbel_search_with_history_and_rules(
            &position,
            &history,
            Some(rule_history.clone()),
            Some(legal),
            model,
            AzSearchLimits {
                simulations,
                top_k,
                seed: seed ^ ((ply as u64) << 32),
                gumbel_scale: 0.0,
                workers: 1,
            },
        );
        let Some(mv) = result.best_move else {
            return 0.0;
        };
        append_history(&mut history, &position, mv);
        rule_history.push(position.rule_history_entry_after_move(mv));
        position.make_move(mv);

        if !position.has_general(Color::Red) {
            return -1.0;
        }
        if !position.has_general(Color::Black) {
            return 1.0;
        }
        if let Some(rule_outcome) = position.rule_outcome_with_history(&rule_history) {
            return match rule_outcome {
                RuleOutcome::Draw(_) => 0.0,
                RuleOutcome::Win(Color::Red) => 1.0,
                RuleOutcome::Win(Color::Black) => -1.0,
            };
        }
    }
    0.0
}

fn train_samples(
    model: &mut AzNnue,
    samples: &[AzTrainingSample],
    epochs: usize,
    lr: f32,
    batch_size: usize,
    rng: &mut SplitMix64,
) -> AzTrainStats {
    if samples.is_empty() || epochs == 0 || lr <= 0.0 {
        return AzTrainStats::default();
    }

    let mut optimizer = match model.optimizer.take() {
        Some(optimizer) if optimizer.matches(model) => optimizer,
        _ => Box::new(AdamWState::new(model)),
    };
    let mut gradient = AzGrad::new(model);
    let mut order = (0..samples.len()).collect::<Vec<_>>();
    let mut stats = AzTrainStats::default();
    let batch_size = batch_size.max(1);
    for _ in 0..epochs {
        shuffle(&mut order, rng);
        stats = AzTrainStats::default();
        for batch in order.chunks(batch_size) {
            let batch_stats = train_batch(
                model,
                &mut optimizer,
                &mut gradient,
                samples,
                batch,
                lr,
            );
            stats.add_assign(&batch_stats);
        }
    }
    model.optimizer = Some(optimizer);
    if stats.samples > 0 {
        let denom = stats.samples as f32;
        stats.loss /= denom;
        stats.value_loss /= denom;
        stats.policy_ce /= denom;
    }
    stats
}

fn train_batch(
    model: &mut AzNnue,
    optimizer: &mut AdamWState,
    gradient: &mut AzGrad,
    samples: &[AzTrainingSample],
    batch: &[usize],
    lr: f32,
) -> AzTrainStats {
    gradient.clear();
    let cache = train_batch_forward_cache(model, samples, batch);
    let stats = accumulate_batch_cached(model, gradient, samples, batch, &cache);
    if stats.samples > 0 {
        apply_adamw_gradient(model, optimizer, gradient, lr, stats.samples as f32);
    }
    stats
}

fn accumulate_batch_cached(
    model: &AzNnue,
    gradient: &mut AzGrad,
    samples: &[AzTrainingSample],
    batch: &[usize],
    cache: &TrainBatchCache,
) -> AzTrainStats {
    let batch_size = batch.len();
    let mut stats = AzTrainStats::default();
    let hidden_all = cache
        .activations
        .last()
        .expect("at least one activation exists");
    let global_all = &cache.global;
    let mut activation_grads = vec![0.0; batch_size * model.hidden_size];
    let mut global_grads = vec![0.0; batch_size * GLOBAL_CONTEXT_SIZE];
    let mut value_logit_grads = vec![0.0; batch_size * VALUE_LOGITS];
    let policy_layout = build_policy_batch_layout(samples, batch);
    let mut policy_logits = vec![0.0; policy_layout.move_indices.len()];
    let mut policy_probs = vec![0.0; policy_layout.move_indices.len()];

    for (row, &sample_index) in batch.iter().enumerate() {
        let sample = &samples[sample_index];
        let value_probs: [f32; VALUE_LOGITS] = softmax_fixed(cache.row_value_logits(row));
        let value = if value_probs.len() >= VALUE_LOGITS {
            value_probs[0] - value_probs[2]
        } else {
            0.0
        };
        let value_target = scalar_to_wdl_target(sample.value);
        let value_error = value - sample.value;
        let value_loss = value_error * value_error;
        let value_train_loss = value_probs
            .iter()
            .zip(value_target.iter())
            .map(|(predicted, target)| -target * predicted.max(1e-9).ln())
            .sum::<f32>();

        stats.loss += value_train_loss;
        stats.value_loss += value_loss;
        stats.value_pred_sum += value;
        stats.value_pred_sq_sum += value * value;
        stats.value_target_sum += sample.value;
        stats.value_target_sq_sum += sample.value * sample.value;
        stats.samples += 1;

        for out in 0..VALUE_LOGITS {
            value_logit_grads[row * VALUE_LOGITS + out] = value_probs[out] - value_target[out];
        }
    }

    grad_weights_batch(
        &mut gradient.value_logits_weights,
        &value_logit_grads,
        VALUE_LOGITS,
        &cache.value_intermediate,
        VALUE_HIDDEN_SIZE,
        batch_size,
    );
    add_bias_grad(
        &mut gradient.value_logits_bias,
        &value_logit_grads,
        batch_size,
        VALUE_LOGITS,
    );

    let mut intermediate_grads = batch_times_weights(
        &value_logit_grads,
        batch_size,
        VALUE_LOGITS,
        &model.value_logits_weights,
        VALUE_HIDDEN_SIZE,
    );
    apply_relu_mask_and_clamp(
        &mut intermediate_grads,
        &cache.value_intermediate_pre,
        -4.0,
        4.0,
    );

    grad_weights_batch(
        &mut gradient.value_intermediate_hidden,
        &intermediate_grads,
        VALUE_HIDDEN_SIZE,
        hidden_all,
        model.hidden_size,
        batch_size,
    );
    grad_weights_batch(
        &mut gradient.value_intermediate_global,
        &intermediate_grads,
        VALUE_HIDDEN_SIZE,
        global_all,
        GLOBAL_CONTEXT_SIZE,
        batch_size,
    );
    add_bias_grad(
        &mut gradient.value_intermediate_bias,
        &intermediate_grads,
        batch_size,
        VALUE_HIDDEN_SIZE,
    );
    add_batch_times_weights(
        &mut activation_grads,
        &intermediate_grads,
        batch_size,
        VALUE_HIDDEN_SIZE,
        &model.value_intermediate_hidden,
        model.hidden_size,
    );
    add_batch_times_weights(
        &mut global_grads,
        &intermediate_grads,
        batch_size,
        VALUE_HIDDEN_SIZE,
        &model.value_intermediate_global,
        GLOBAL_CONTEXT_SIZE,
    );

    compute_policy_batch_logits(
        model,
        hidden_all,
        global_all,
        &policy_layout,
        &mut policy_logits,
    );
    compute_policy_batch_probs(&policy_layout, &policy_logits, &mut policy_probs);

    for row in 0..batch_size {
        let policy_range = policy_layout.sample_range(row);
        let policy_ce = policy_probs[policy_range.clone()]
            .iter()
            .zip(policy_layout.targets[policy_range.clone()].iter())
            .map(|(predicted, target)| -target * predicted.max(1e-9).ln())
            .sum::<f32>();
        stats.loss += policy_ce;
        stats.policy_ce += policy_ce;

        let activation_grad = row_slice_mut(&mut activation_grads, row, model.hidden_size);
        let global_grad = row_slice_mut(&mut global_grads, row, GLOBAL_CONTEXT_SIZE);
        let hidden = row_slice(hidden_all, row, model.hidden_size);
        let global = row_slice(global_all, row, GLOBAL_CONTEXT_SIZE);
        for flat_index in policy_range {
            let move_index = policy_layout.move_indices[flat_index];
            let policy_grad = (policy_probs[flat_index] - policy_layout.targets[flat_index])
                .clamp(-4.0, 4.0);
            let hidden_offset = move_index * model.hidden_size;
            let hidden_row =
                &model.policy_move_hidden[hidden_offset..hidden_offset + model.hidden_size];
            let hidden_grad_row = &mut gradient.policy_move_hidden
                [hidden_offset..hidden_offset + model.hidden_size];
            add_scaled(activation_grad, hidden_row, policy_grad);
            add_scaled(hidden_grad_row, hidden, policy_grad);
            let global_offset = move_index * GLOBAL_CONTEXT_SIZE;
            let global_row =
                &model.policy_move_global[global_offset..global_offset + GLOBAL_CONTEXT_SIZE];
            let global_grad_row = &mut gradient.policy_move_global
                [global_offset..global_offset + GLOBAL_CONTEXT_SIZE];
            add_scaled(global_grad, global_row, policy_grad);
            add_scaled(global_grad_row, global, policy_grad);
            gradient.policy_move_bias[move_index] += policy_grad;
        }
    }

    let mut input_grads = activation_grads;
    for layer in (0..model.trunk_depth).rev() {
        let input = &cache.activations[layer];
        let output = &cache.activations[layer + 1];
        let weight_offset = layer * model.hidden_size * model.hidden_size;
        let global_weight_offset = layer * model.hidden_size * GLOBAL_CONTEXT_SIZE;
        let bias_offset = layer * model.hidden_size;

        clamp_inplace(&mut input_grads, -4.0, 4.0);
        let mut residual_grads = vec![0.0; batch_size * model.hidden_size];
        for idx in 0..residual_grads.len() {
            if output[idx] > input[idx] {
                residual_grads[idx] = input_grads[idx] * RESIDUAL_TRUNK_SCALE;
            }
        }
        let mut previous_grads = input_grads.clone();
        grad_weights_batch(
            &mut gradient.trunk_weights
                [weight_offset..weight_offset + model.hidden_size * model.hidden_size],
            &residual_grads,
            model.hidden_size,
            input,
            model.hidden_size,
            batch_size,
        );
        grad_weights_batch(
            &mut gradient.trunk_global_weights[global_weight_offset
                ..global_weight_offset + model.hidden_size * GLOBAL_CONTEXT_SIZE],
            &residual_grads,
            model.hidden_size,
            global_all,
            GLOBAL_CONTEXT_SIZE,
            batch_size,
        );
        add_bias_grad(
            &mut gradient.trunk_biases[bias_offset..bias_offset + model.hidden_size],
            &residual_grads,
            batch_size,
            model.hidden_size,
        );
        add_batch_times_weights(
            &mut previous_grads,
            &residual_grads,
            batch_size,
            model.hidden_size,
            &model.trunk_weights[weight_offset..weight_offset + model.hidden_size * model.hidden_size],
            model.hidden_size,
        );
        add_batch_times_weights(
            &mut global_grads,
            &residual_grads,
            batch_size,
            model.hidden_size,
            &model.trunk_global_weights[global_weight_offset
                ..global_weight_offset + model.hidden_size * GLOBAL_CONTEXT_SIZE],
            GLOBAL_CONTEXT_SIZE,
        );
        input_grads = previous_grads;
    }

    let initial_hidden = &cache.activations[0];
    apply_relu_mask_and_clamp(&mut global_grads, global_all, -4.0, 4.0);
    grad_weights_batch(
        &mut gradient.global_hidden,
        &global_grads,
        GLOBAL_CONTEXT_SIZE,
        initial_hidden,
        model.hidden_size,
        batch_size,
    );
    add_bias_grad(
        &mut gradient.global_bias,
        &global_grads,
        batch_size,
        GLOBAL_CONTEXT_SIZE,
    );
    add_batch_times_weights(
        &mut input_grads,
        &global_grads,
        batch_size,
        GLOBAL_CONTEXT_SIZE,
        &model.global_hidden,
        model.hidden_size,
    );

    for (row, &sample_index) in batch.iter().enumerate() {
        let sample = &samples[sample_index];
        let initial_hidden_row = row_slice(initial_hidden, row, model.hidden_size);
        let input_grad_row = row_slice(&input_grads, row, model.hidden_size);
        let input_scale = 1.0 / (sample.features.len() as f32).sqrt().max(1.0);
        for idx in 0..model.hidden_size {
            if initial_hidden_row[idx] <= 0.0 {
                continue;
            }
            let grad = input_grad_row[idx].clamp(-4.0, 4.0);
            gradient.hidden_bias[idx] += grad;
            for &feature in &sample.features {
                gradient.input_hidden[feature * model.hidden_size + idx] += grad * input_scale;
            }
        }
    }

    stats
}

struct TrainBatchCache {
    activations: Vec<Vec<f32>>,
    global: Vec<f32>,
    value_intermediate_pre: Vec<f32>,
    value_intermediate: Vec<f32>,
    value_logits: Vec<f32>,
}

struct PolicyBatchLayout {
    sample_offsets: Vec<usize>,
    move_indices: Vec<usize>,
    targets: Vec<f32>,
}

impl TrainBatchCache {
    fn row_value_logits(&self, row: usize) -> &[f32] {
        row_slice(&self.value_logits, row, VALUE_LOGITS)
    }
}

impl PolicyBatchLayout {
    fn sample_range(&self, row: usize) -> std::ops::Range<usize> {
        self.sample_offsets[row]..self.sample_offsets[row + 1]
    }
}

fn row_slice(values: &[f32], row: usize, width: usize) -> &[f32] {
    let start = row * width;
    &values[start..start + width]
}

fn row_slice_mut(values: &mut [f32], row: usize, width: usize) -> &mut [f32] {
    let start = row * width;
    &mut values[start..start + width]
}

fn build_policy_batch_layout(samples: &[AzTrainingSample], batch: &[usize]) -> PolicyBatchLayout {
    let total_moves = batch
        .iter()
        .map(|&sample_index| samples[sample_index].move_indices.len())
        .sum();
    let mut sample_offsets = Vec::with_capacity(batch.len() + 1);
    let mut move_indices = Vec::with_capacity(total_moves);
    let mut targets = Vec::with_capacity(total_moves);
    sample_offsets.push(0);
    for &sample_index in batch {
        let sample = &samples[sample_index];
        move_indices.extend(sample.move_indices.iter().copied());
        targets.extend(sample.policy.iter().copied());
        sample_offsets.push(move_indices.len());
    }
    PolicyBatchLayout {
        sample_offsets,
        move_indices,
        targets,
    }
}

fn compute_policy_batch_logits(
    model: &AzNnue,
    hidden_all: &[f32],
    global_all: &[f32],
    layout: &PolicyBatchLayout,
    logits: &mut [f32],
) {
    for row in 0..(layout.sample_offsets.len() - 1) {
        let range = layout.sample_range(row);
        let hidden = row_slice(hidden_all, row, model.hidden_size);
        let global = row_slice(global_all, row, GLOBAL_CONTEXT_SIZE);
        for (slot, &move_index) in logits[range.clone()]
            .iter_mut()
            .zip(layout.move_indices[range].iter())
        {
            *slot = model.policy_logit_from_hidden_index(hidden, global, move_index);
        }
    }
}

fn compute_policy_batch_probs(layout: &PolicyBatchLayout, logits: &[f32], probs: &mut [f32]) {
    for row in 0..(layout.sample_offsets.len() - 1) {
        let range = layout.sample_range(row);
        softmax_slice(logits, probs, range);
    }
}


fn clamp_inplace(values: &mut [f32], min: f32, max: f32) {
    for value in values {
        *value = value.clamp(min, max);
    }
}

fn apply_relu_mask_and_clamp(grads: &mut [f32], pre_activation: &[f32], min: f32, max: f32) {
    for idx in 0..grads.len() {
        if pre_activation[idx] <= 0.0 {
            grads[idx] = 0.0;
        } else {
            grads[idx] = grads[idx].clamp(min, max);
        }
    }
}

fn add_bias_grad(dst: &mut [f32], grads: &[f32], batch_size: usize, width: usize) {
    for row in 0..batch_size {
        for col in 0..width {
            dst[col] += grads[row * width + col];
        }
    }
}

fn grad_weights_batch(
    dst_out_in: &mut [f32],
    grads_b_out: &[f32],
    out_dim: usize,
    input_b_in: &[f32],
    in_dim: usize,
    batch_size: usize,
) {
    unsafe {
        sgemm(
            out_dim,
            batch_size,
            in_dim,
            1.0,
            grads_b_out.as_ptr(),
            1,
            out_dim as isize,
            input_b_in.as_ptr(),
            in_dim as isize,
            1,
            1.0,
            dst_out_in.as_mut_ptr(),
            in_dim as isize,
            1,
        );
    }
}

fn batch_times_weights(
    input_b_in: &[f32],
    batch_size: usize,
    in_dim: usize,
    weights_out_in: &[f32],
    out_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; batch_size * out_dim];
    add_batch_times_weights(
        &mut output,
        input_b_in,
        batch_size,
        in_dim,
        weights_out_in,
        out_dim,
    );
    output
}

fn add_batch_times_weights(
    output_b_out: &mut [f32],
    input_b_in: &[f32],
    batch_size: usize,
    in_dim: usize,
    weights_out_in: &[f32],
    out_dim: usize,
) {
    unsafe {
        sgemm(
            batch_size,
            in_dim,
            out_dim,
            1.0,
            input_b_in.as_ptr(),
            in_dim as isize,
            1,
            weights_out_in.as_ptr(),
            1,
            in_dim as isize,
            1.0,
            output_b_out.as_mut_ptr(),
            out_dim as isize,
            1,
        );
    }
}

fn train_batch_forward_cache(
    model: &AzNnue,
    samples: &[AzTrainingSample],
    batch: &[usize],
) -> TrainBatchCache {
    let batch_size = batch.len();
    let mut hidden = vec![0.0; batch_size * model.hidden_size];
    for row in 0..batch_size {
        let start = row * model.hidden_size;
        hidden[start..start + model.hidden_size].copy_from_slice(&model.hidden_bias);
    }
    for (row, &index) in batch.iter().enumerate() {
        let sample = &samples[index];
        let row_offset = row * model.hidden_size;
        for &feature in &sample.features {
            let weight_row = &model.input_hidden
                [feature * model.hidden_size..(feature + 1) * model.hidden_size];
            for idx in 0..model.hidden_size {
                hidden[row_offset + idx] += weight_row[idx];
            }
        }
    }
    relu_inplace(&mut hidden);

    let global = affine_relu_batch(
        &hidden,
        batch_size,
        model.hidden_size,
        &model.global_hidden,
        GLOBAL_CONTEXT_SIZE,
        &model.global_bias,
    );

    let mut activations = Vec::with_capacity(model.trunk_depth + 1);
    activations.push(hidden);
    for layer in 0..model.trunk_depth {
        let previous = activations.last().expect("previous activation exists");
        let weight_offset = layer * model.hidden_size * model.hidden_size;
        let global_weight_offset = layer * model.hidden_size * GLOBAL_CONTEXT_SIZE;
        let bias_offset = layer * model.hidden_size;
        let mut next = affine_batch(
            previous,
            batch_size,
            model.hidden_size,
            &model.trunk_weights[weight_offset..weight_offset + model.hidden_size * model.hidden_size],
            model.hidden_size,
            &model.trunk_biases[bias_offset..bias_offset + model.hidden_size],
        );
        add_affine_batch(
            &mut next,
            &global,
            batch_size,
            GLOBAL_CONTEXT_SIZE,
            &model.trunk_global_weights
                [global_weight_offset..global_weight_offset + model.hidden_size * GLOBAL_CONTEXT_SIZE],
            model.hidden_size,
        );
        relu_inplace(&mut next);
        for idx in 0..next.len() {
            next[idx] = previous[idx] + RESIDUAL_TRUNK_SCALE * next[idx];
        }
        activations.push(next);
    }

    let hidden = activations
        .last()
        .expect("at least one activation exists");
    let mut value_intermediate_pre = affine_batch(
        hidden,
        batch_size,
        model.hidden_size,
        &model.value_intermediate_hidden,
        VALUE_HIDDEN_SIZE,
        &model.value_intermediate_bias,
    );
    add_affine_batch(
        &mut value_intermediate_pre,
        &global,
        batch_size,
        GLOBAL_CONTEXT_SIZE,
        &model.value_intermediate_global,
        VALUE_HIDDEN_SIZE,
    );
    let mut value_intermediate = value_intermediate_pre.clone();
    relu_inplace(&mut value_intermediate);
    let value_logits = affine_batch(
        &value_intermediate,
        batch_size,
        VALUE_HIDDEN_SIZE,
        &model.value_logits_weights,
        VALUE_LOGITS,
        &model.value_logits_bias,
    );

    TrainBatchCache {
        activations,
        global,
        value_intermediate_pre,
        value_intermediate,
        value_logits,
    }
}

fn relu_inplace(values: &mut [f32]) {
    for value in values {
        *value = value.max(0.0);
    }
}

fn affine_relu_batch(
    input: &[f32],
    batch_size: usize,
    in_dim: usize,
    weights_out_in: &[f32],
    out_dim: usize,
    bias: &[f32],
) -> Vec<f32> {
    let mut output = affine_batch(input, batch_size, in_dim, weights_out_in, out_dim, bias);
    relu_inplace(&mut output);
    output
}

fn affine_batch(
    input: &[f32],
    batch_size: usize,
    in_dim: usize,
    weights_out_in: &[f32],
    out_dim: usize,
    bias: &[f32],
) -> Vec<f32> {
    let mut output = vec![0.0; batch_size * out_dim];
    for row in 0..batch_size {
        output[row * out_dim..(row + 1) * out_dim].copy_from_slice(bias);
    }
    unsafe {
        sgemm(
            batch_size,
            in_dim,
            out_dim,
            1.0,
            input.as_ptr(),
            in_dim as isize,
            1,
            weights_out_in.as_ptr(),
            1,
            in_dim as isize,
            1.0,
            output.as_mut_ptr(),
            out_dim as isize,
            1,
        );
    }
    output
}

fn add_affine_batch(
    output: &mut [f32],
    input: &[f32],
    batch_size: usize,
    in_dim: usize,
    weights_out_in: &[f32],
    out_dim: usize,
) {
    unsafe {
        sgemm(
            batch_size,
            in_dim,
            out_dim,
            1.0,
            input.as_ptr(),
            in_dim as isize,
            1,
            weights_out_in.as_ptr(),
            1,
            in_dim as isize,
            1.0,
            output.as_mut_ptr(),
            out_dim as isize,
            1,
        );
    }
}

fn apply_adamw_gradient(
    model: &mut AzNnue,
    optimizer: &mut AdamWState,
    gradient: &AzGrad,
    lr: f32,
    batch_len: f32,
) {
    let inv_batch = 1.0 / batch_len.max(1.0);
    let (bias_correction1, bias_correction2) = optimizer.advance();

    for idx in 0..model.global_hidden.len() {
        adamw_update(
            &mut model.global_hidden[idx],
            &mut optimizer.global_hidden_m[idx],
            &mut optimizer.global_hidden_v[idx],
            gradient.global_hidden[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            ADAMW_WEIGHT_DECAY,
        );
    }
    for idx in 0..model.global_bias.len() {
        adamw_update(
            &mut model.global_bias[idx],
            &mut optimizer.global_bias_m[idx],
            &mut optimizer.global_bias_v[idx],
            gradient.global_bias[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            0.0,
        );
    }
    for idx in 0..model.value_intermediate_hidden.len() {
        adamw_update(
            &mut model.value_intermediate_hidden[idx],
            &mut optimizer.value_intermediate_hidden_m[idx],
            &mut optimizer.value_intermediate_hidden_v[idx],
            gradient.value_intermediate_hidden[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            ADAMW_WEIGHT_DECAY,
        );
    }
    for idx in 0..model.value_intermediate_global.len() {
        adamw_update(
            &mut model.value_intermediate_global[idx],
            &mut optimizer.value_intermediate_global_m[idx],
            &mut optimizer.value_intermediate_global_v[idx],
            gradient.value_intermediate_global[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            ADAMW_WEIGHT_DECAY,
        );
    }
    for idx in 0..model.value_intermediate_bias.len() {
        adamw_update(
            &mut model.value_intermediate_bias[idx],
            &mut optimizer.value_intermediate_bias_m[idx],
            &mut optimizer.value_intermediate_bias_v[idx],
            gradient.value_intermediate_bias[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            0.0,
        );
    }
    for idx in 0..model.value_logits_weights.len() {
        adamw_update(
            &mut model.value_logits_weights[idx],
            &mut optimizer.value_logits_weights_m[idx],
            &mut optimizer.value_logits_weights_v[idx],
            gradient.value_logits_weights[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            ADAMW_WEIGHT_DECAY,
        );
    }
    for idx in 0..model.value_logits_bias.len() {
        adamw_update(
            &mut model.value_logits_bias[idx],
            &mut optimizer.value_logits_bias_m[idx],
            &mut optimizer.value_logits_bias_v[idx],
            gradient.value_logits_bias[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            0.0,
        );
    }

    for idx in 0..model.policy_move_hidden.len() {
        adamw_update(
            &mut model.policy_move_hidden[idx],
            &mut optimizer.policy_move_hidden_m[idx],
            &mut optimizer.policy_move_hidden_v[idx],
            gradient.policy_move_hidden[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            ADAMW_WEIGHT_DECAY,
        );
    }
    for idx in 0..model.policy_move_global.len() {
        adamw_update(
            &mut model.policy_move_global[idx],
            &mut optimizer.policy_move_global_m[idx],
            &mut optimizer.policy_move_global_v[idx],
            gradient.policy_move_global[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            ADAMW_WEIGHT_DECAY,
        );
    }
    for idx in 0..model.policy_move_bias.len() {
        adamw_update(
            &mut model.policy_move_bias[idx],
            &mut optimizer.policy_move_bias_m[idx],
            &mut optimizer.policy_move_bias_v[idx],
            gradient.policy_move_bias[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            0.0,
        );
    }

    for idx in 0..model.trunk_weights.len() {
        adamw_update(
            &mut model.trunk_weights[idx],
            &mut optimizer.trunk_weights_m[idx],
            &mut optimizer.trunk_weights_v[idx],
            gradient.trunk_weights[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            ADAMW_WEIGHT_DECAY,
        );
    }
    for idx in 0..model.trunk_biases.len() {
        adamw_update(
            &mut model.trunk_biases[idx],
            &mut optimizer.trunk_biases_m[idx],
            &mut optimizer.trunk_biases_v[idx],
            gradient.trunk_biases[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            0.0,
        );
    }
    for idx in 0..model.trunk_global_weights.len() {
        adamw_update(
            &mut model.trunk_global_weights[idx],
            &mut optimizer.trunk_global_weights_m[idx],
            &mut optimizer.trunk_global_weights_v[idx],
            gradient.trunk_global_weights[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            ADAMW_WEIGHT_DECAY,
        );
    }

    for idx in 0..model.hidden_bias.len() {
        adamw_update(
            &mut model.hidden_bias[idx],
            &mut optimizer.hidden_bias_m[idx],
            &mut optimizer.hidden_bias_v[idx],
            gradient.hidden_bias[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            0.0,
        );
    }
    for idx in 0..model.input_hidden.len() {
        adamw_update(
            &mut model.input_hidden[idx],
            &mut optimizer.input_hidden_m[idx],
            &mut optimizer.input_hidden_v[idx],
            gradient.input_hidden[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            ADAMW_WEIGHT_DECAY,
        );
    }
}

fn adamw_update(
    parameter: &mut f32,
    first_moment: &mut f32,
    second_moment: &mut f32,
    gradient: f32,
    lr: f32,
    bias_correction1: f32,
    bias_correction2: f32,
    weight_decay: f32,
) {
    let gradient = gradient.clamp(-4.0, 4.0);
    *first_moment = ADAMW_BETA1 * *first_moment + (1.0 - ADAMW_BETA1) * gradient;
    *second_moment = ADAMW_BETA2 * *second_moment + (1.0 - ADAMW_BETA2) * gradient * gradient;
    let first_unbiased = *first_moment / bias_correction1.max(1e-12);
    let second_unbiased = *second_moment / bias_correction2.max(1e-12);
    if weight_decay > 0.0 {
        *parameter -= lr * weight_decay * *parameter;
    }
    *parameter -= lr * first_unbiased / (second_unbiased.sqrt() + ADAMW_EPSILON);
}

fn shuffle(values: &mut [usize], rng: &mut SplitMix64) {
    for index in (1..values.len()).rev() {
        let swap_with = (rng.next() as usize) % (index + 1);
        values.swap(index, swap_with);
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let mut values = Vec::with_capacity(logits.len());
    softmax_into(logits, &mut values);
    values
}

fn softmax_into(logits: &[f32], output: &mut Vec<f32>) {
    output.clear();
    if logits.is_empty() {
        return;
    }
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    output.reserve(logits.len());
    for &logit in logits {
        let value = (logit - max_logit).exp();
        sum += value;
        output.push(value);
    }
    let inv_sum = sum.max(1e-12).recip();
    for value in output {
        *value *= inv_sum;
    }
}

fn softmax_slice(input: &[f32], output: &mut [f32], range: std::ops::Range<usize>) {
    if range.is_empty() {
        return;
    }
    let max_logit = input[range.clone()]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for index in range.clone() {
        let value = (input[index] - max_logit).exp();
        output[index] = value;
        sum += value;
    }
    let inv_sum = sum.max(1e-12).recip();
    for index in range {
        output[index] *= inv_sum;
    }
}

fn softmax_fixed<const N: usize>(logits: &[f32]) -> [f32; N] {
    let mut output = [0.0; N];
    if logits.len() < N {
        return output;
    }
    let max_logit = logits[..N]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for index in 0..N {
        let value = (logits[index] - max_logit).exp();
        output[index] = value;
        sum += value;
    }
    let inv_sum = sum.max(1e-12).recip();
    for value in &mut output {
        *value *= inv_sum;
    }
    output
}

fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;
    let chunks = left.len() / 4;
    for chunk in 0..chunks {
        let index = chunk * 4;
        sum0 += left[index] * right[index];
        sum1 += left[index + 1] * right[index + 1];
        sum2 += left[index + 2] * right[index + 2];
        sum3 += left[index + 3] * right[index + 3];
    }
    let mut sum = (sum0 + sum1) + (sum2 + sum3);
    for index in (chunks * 4)..left.len() {
        sum += left[index] * right[index];
    }
    sum
}

fn add_scaled(dst: &mut [f32], src: &[f32], scale: f32) {
    debug_assert_eq!(dst.len(), src.len());
    for (dst_value, src_value) in dst.iter_mut().zip(src.iter()) {
        *dst_value += scale * *src_value;
    }
}

fn scalar_value_from_logits(logits: &[f32]) -> (f32, Vec<f32>) {
    let probs = softmax(logits);
    if probs.len() < VALUE_LOGITS {
        return (0.0, probs);
    }
    (probs[0] - probs[2], probs)
}

#[allow(dead_code)]
fn scalar_to_wdl_target(value: f32) -> [f32; VALUE_LOGITS] {
    let v = value.clamp(-1.0, 1.0);
    if v >= 0.0 {
        [v, 1.0 - v, 0.0]
    } else {
        [0.0, 1.0 + v, -v]
    }
}

fn parse_floats(text: &str) -> io::Result<Vec<f32>> {
    text.split_whitespace()
        .map(|value| {
            value.parse::<f32>().map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid float: {value}"),
                )
            })
        })
        .collect()
}

fn format_floats(values: &[f32]) -> String {
    values
        .iter()
        .map(f32::to_string)
        .collect::<Vec<_>>()
        .join(" ")
}

fn deterministic_gumbel(hash: u64, mv: Move, salt: u64) -> f32 {
    let seed = hash
        ^ ((mv.from as u64) << 48)
        ^ ((mv.to as u64) << 32)
        ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let value = splitmix64(seed);
    let unit = (((value >> 11) as f64) + 0.5) * (1.0 / ((1u64 << 53) as f64));
    (-(-unit.ln()).ln()) as f32
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = splitmix64(self.state);
        self.state
    }

    fn unit_f32(&mut self) -> f32 {
        let value = self.next();
        (((value >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))) as f32
    }

    fn weight(&mut self, scale: f32) -> f32 {
        (self.unit_f32() * 2.0 - 1.0) * scale
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut mixed = value;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^ (mixed >> 31)
}

const fn is_advisor_pos(rank: usize, file: usize) -> bool {
    (rank == 7 && file == 3)
        || (rank == 7 && file == 5)
        || (rank == 8 && file == 4)
        || (rank == 9 && file == 3)
        || (rank == 9 && file == 5)
}

const fn is_elephant_pos(rank: usize, file: usize) -> bool {
    (rank == 5 && file == 2)
        || (rank == 5 && file == 6)
        || (rank == 7 && file == 0)
        || (rank == 7 && file == 4)
        || (rank == 7 && file == 8)
        || (rank == 9 && file == 2)
        || (rank == 9 && file == 6)
}

const fn is_valid_policy_move(from: usize, to: usize) -> bool {
    let from_file = from % BOARD_FILES;
    let from_rank = from / BOARD_FILES;
    let to_file = to % BOARD_FILES;
    let to_rank = to / BOARD_FILES;

    let df_signed = to_file as i32 - from_file as i32;
    let dr_signed = to_rank as i32 - from_rank as i32;
    let df = if df_signed < 0 { -df_signed } else { df_signed };
    let dr = if dr_signed < 0 { -dr_signed } else { dr_signed };

    if df == 0 || dr == 0 {
        return true;
    }
    if (df == 1 && dr == 2) || (df == 2 && dr == 1) {
        return true;
    }
    if df == 1 && dr == 1 && is_advisor_pos(from_rank, from_file) && is_advisor_pos(to_rank, to_file)
    {
        return true;
    }
    if df == 2
        && dr == 2
        && is_elephant_pos(from_rank, from_file)
        && is_elephant_pos(to_rank, to_file)
    {
        return true;
    }
    false
}

const fn compute_dense_move_count() -> usize {
    let mut count = 0;
    let mut from = 0;
    while from < BOARD_SIZE {
        let mut to = 0;
        while to < BOARD_SIZE {
            if from != to && is_valid_policy_move(from, to) {
                count += 1;
            }
            to += 1;
        }
        from += 1;
    }
    count
}

struct MoveMap {
    sparse_to_dense: [u16; SPARSE_MOVE_SPACE],
    #[allow(dead_code)]
    dense_to_sparse: [u16; DENSE_MOVE_SPACE],
}

fn move_map() -> &'static MoveMap {
    use std::sync::OnceLock;
    static MAP: OnceLock<MoveMap> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut sparse_to_dense = [u16::MAX; SPARSE_MOVE_SPACE];
        let mut dense_to_sparse = [0u16; DENSE_MOVE_SPACE];
        let mut idx = 0usize;
        for from in 0..BOARD_SIZE {
            for to in 0..BOARD_SIZE {
                if from != to && is_valid_policy_move(from, to) {
                    let sparse = from * BOARD_SIZE + to;
                    sparse_to_dense[sparse] = idx as u16;
                    dense_to_sparse[idx] = sparse as u16;
                    idx += 1;
                }
            }
        }
        assert_eq!(idx, DENSE_MOVE_SPACE);
        MoveMap {
            sparse_to_dense,
            dense_to_sparse,
        }
    })
}

fn dense_move_index(mv: Move) -> usize {
    let sparse = mv.from as usize * BOARD_SIZE + mv.to as usize;
    let dense = move_map().sparse_to_dense[sparse];
    debug_assert!(
        dense != u16::MAX,
        "invalid policy move {}->{}",
        mv.from,
        mv.to
    );
    dense as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn considered_visit_sequence_matches_sequential_halving_schedule() {
        assert_eq!(
            considered_visit_sequence(4, 16),
            vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        );
        assert_eq!(considered_visit_sequence(1, 5), vec![0, 1, 2, 3, 4]);
        assert_eq!(considered_visit_sequence(0, 5), Vec::<u32>::new());
    }

    #[test]
    fn dense_move_space_matches_enumeration() {
        let map = move_map();
        assert_eq!(DENSE_MOVE_SPACE, 2062);
        for i in 0..DENSE_MOVE_SPACE {
            let sparse = map.dense_to_sparse[i] as usize;
            assert_eq!(map.sparse_to_dense[sparse], i as u16);
        }
    }

    #[test]
    fn completed_q_normalization_turns_equal_values_into_zero_bonus() {
        let mut qvalues = vec![0.5, 0.5, 0.5];
        normalize_completed_q(&mut qvalues, 0);
        assert_eq!(qvalues, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn gumbel_search_visits_initial_considered_actions() {
        let model = AzNnue::random_with_depth(4, 1, 7);
        let result = gumbel_search(
            &Position::startpos(),
            &model,
            AzSearchLimits {
                simulations: 512,
                top_k: 16,
                seed: 11,
                gumbel_scale: 1.0,
                workers: 1,
            },
        );
        let visited_actions = result
            .candidates
            .iter()
            .filter(|candidate| candidate.visits > 0)
            .count();

        assert_eq!(result.simulations, 512);
        assert!(visited_actions >= 16);
    }

    #[test]
    fn lambda_targets_use_next_root_bootstrap() {
        let mut samples = vec![
            AzTrainingSample {
                features: Vec::new(),
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 0.0,
                side_sign: 1.0,
                reward: 0.0,
                discount: -1.0,
                bootstrap_value: 0.2,
            },
            AzTrainingSample {
                features: Vec::new(),
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 0.0,
                side_sign: -1.0,
                reward: 1.0,
                discount: 0.0,
                bootstrap_value: -0.4,
            },
        ];

        assign_lambda_targets(&mut samples, 1.0, 0.5);

        assert!((samples[1].value - 1.0).abs() < 1e-6);
        assert!((samples[0].value + 0.3).abs() < 1e-6);
    }

    #[test]
    fn value_head_can_overfit_tiny_fixed_dataset() {
        let mut model = AzNnue::random_with_depth(16, 1, 7);
        model.hidden_bias.fill(0.1);
        model.trunk_biases.fill(0.1);
        model.global_bias.fill(0.1);

        let samples = vec![
            AzTrainingSample {
                features: vec![0],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 1.0,
                side_sign: 1.0,
                reward: 0.0,
                discount: 0.0,
                bootstrap_value: 0.0,
            },
            AzTrainingSample {
                features: vec![1],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: -1.0,
                side_sign: 1.0,
                reward: 0.0,
                discount: 0.0,
                bootstrap_value: 0.0,
            },
            AzTrainingSample {
                features: vec![2],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 0.75,
                side_sign: 1.0,
                reward: 0.0,
                discount: 0.0,
                bootstrap_value: 0.0,
            },
            AzTrainingSample {
                features: vec![3],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: -0.75,
                side_sign: 1.0,
                reward: 0.0,
                discount: 0.0,
                bootstrap_value: 0.0,
            },
        ];

        let mut rng = SplitMix64::new(17);
        let before = train_samples(&mut model, &samples, 1, 0.003, 4, &mut rng).value_loss;
        let after = train_samples(&mut model, &samples, 300, 0.003, 4, &mut rng).value_loss;

        assert!(after < before * 0.5, "before={before} after={after}");
        assert!(after < 0.35, "after={after}");
    }

    #[test]
    fn batched_training_is_deterministic() {
        let samples = vec![
            AzTrainingSample {
                features: vec![0, 4, 8],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 1.0,
                side_sign: 1.0,
                reward: 0.0,
                discount: 0.0,
                bootstrap_value: 0.0,
            },
            AzTrainingSample {
                features: vec![1, 5, 9],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: -1.0,
                side_sign: 1.0,
                reward: 0.0,
                discount: 0.0,
                bootstrap_value: 0.0,
            },
            AzTrainingSample {
                features: vec![2, 6, 10],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 0.5,
                side_sign: 1.0,
                reward: 0.0,
                discount: 0.0,
                bootstrap_value: 0.0,
            },
            AzTrainingSample {
                features: vec![3, 7, 11],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: -0.5,
                side_sign: 1.0,
                reward: 0.0,
                discount: 0.0,
                bootstrap_value: 0.0,
            },
        ];
        let mut single = AzNnue::random_with_depth(16, 1, 23);
        single.hidden_bias.fill(0.1);
        single.trunk_biases.fill(0.1);
        single.global_bias.fill(0.1);
        let mut repeated = single.clone();

        let mut rng_single = SplitMix64::new(99);
        let mut rng_repeated = SplitMix64::new(99);
        let single_stats = train_samples(&mut single, &samples, 5, 0.003, 4, &mut rng_single);
        let repeated_stats =
            train_samples(&mut repeated, &samples, 5, 0.003, 4, &mut rng_repeated);

        assert!((single_stats.loss - repeated_stats.loss).abs() < 1e-5);
        assert!((single_stats.value_loss - repeated_stats.value_loss).abs() < 1e-5);
        assert!((single_stats.value_pred_sum - repeated_stats.value_pred_sum).abs() < 1e-4);
        assert!((single_stats.value_target_sum - repeated_stats.value_target_sum).abs() < 1e-6);
        assert!(single
            .value_logits_bias
            .iter()
            .zip(&repeated.value_logits_bias)
            .all(|(left, right)| (*left - *right).abs() < 1e-5));
    }
}
