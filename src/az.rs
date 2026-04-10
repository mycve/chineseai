use std::collections::VecDeque;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::Arc;
use std::thread;

use crate::nnue::{
    HistoryMove, V4_INPUT_SIZE, extract_sparse_features_v4, mirror_file_move,
    mirror_sparse_features_file, orient_move,
};
use crate::xiangqi::{BOARD_SIZE, Color, Move, Position, RuleHistoryEntry, RuleOutcome};

pub const AZNNUE_FORMAT: &str = "aznnue-v11";
const MOVE_SPACE: usize = BOARD_SIZE * BOARD_SIZE;
const GLOBAL_CONTEXT_SIZE: usize = 8;
const VALUE_LOGITS: usize = 3;
const VALUE_SCALE_CP: f32 = 800.0;
const COMPLETED_Q_VALUE_SCALE: f32 = 0.1;
const COMPLETED_Q_MAXVISIT_INIT: f32 = 50.0;
const RESIDUAL_TRUNK_SCALE: f32 = 0.5;
const ADAMW_BETA1: f32 = 0.9;
const ADAMW_BETA2: f32 = 0.999;
const ADAMW_EPSILON: f32 = 1e-8;
const ADAMW_WEIGHT_DECAY: f32 = 1e-4;

#[derive(Debug)]
pub struct AzNnue {
    pub hidden_size: usize,
    pub trunk_depth: usize,
    pub input_hidden: Vec<f32>,
    pub hidden_bias: Vec<f32>,
    pub trunk_weights: Vec<f32>,
    pub trunk_biases: Vec<f32>,
    pub global_hidden: Vec<f32>,
    pub global_bias: Vec<f32>,
    pub value_logits_hidden: Vec<f32>,
    pub value_logits_global: Vec<f32>,
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
            global_hidden: self.global_hidden.clone(),
            global_bias: self.global_bias.clone(),
            value_logits_hidden: self.value_logits_hidden.clone(),
            value_logits_global: self.value_logits_global.clone(),
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
    pub train_workers: usize,
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
    pub value_mse: f32,
    pub policy_ce: f32,
    pub value_pred_mean: f32,
    pub value_pred_std: f32,
    pub value_target_mean: f32,
    pub value_target_std: f32,
    pub train_samples: usize,
    pub pool_games: usize,
    pub pool_samples: usize,
}

#[derive(Clone, Debug)]
struct AzTrainingSample {
    features: Vec<usize>,
    moves: Vec<Move>,
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
    value_mse: f32,
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
    global_hidden: Vec<f32>,
    global_bias: Vec<f32>,
    value_logits_hidden: Vec<f32>,
    value_logits_global: Vec<f32>,
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
            global_hidden: vec![0.0; model.global_hidden.len()],
            global_bias: vec![0.0; model.global_bias.len()],
            value_logits_hidden: vec![0.0; model.value_logits_hidden.len()],
            value_logits_global: vec![0.0; model.value_logits_global.len()],
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
        self.value_logits_hidden.fill(0.0);
        self.value_logits_global.fill(0.0);
        self.value_logits_bias.fill(0.0);
        self.policy_move_hidden.fill(0.0);
        self.policy_move_global.fill(0.0);
        self.policy_move_bias.fill(0.0);
    }

    fn add_assign(&mut self, other: &Self) {
        add_assign_slice(&mut self.input_hidden, &other.input_hidden);
        add_assign_slice(&mut self.hidden_bias, &other.hidden_bias);
        add_assign_slice(&mut self.trunk_weights, &other.trunk_weights);
        add_assign_slice(&mut self.trunk_biases, &other.trunk_biases);
        add_assign_slice(&mut self.global_hidden, &other.global_hidden);
        add_assign_slice(&mut self.global_bias, &other.global_bias);
        add_assign_slice(&mut self.value_logits_hidden, &other.value_logits_hidden);
        add_assign_slice(&mut self.value_logits_global, &other.value_logits_global);
        add_assign_slice(&mut self.value_logits_bias, &other.value_logits_bias);
        add_assign_slice(&mut self.policy_move_hidden, &other.policy_move_hidden);
        add_assign_slice(&mut self.policy_move_global, &other.policy_move_global);
        add_assign_slice(&mut self.policy_move_bias, &other.policy_move_bias);
    }
}

impl AzTrainStats {
    fn add_assign(&mut self, other: &Self) {
        self.loss += other.loss;
        self.value_mse += other.value_mse;
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
    global_hidden_m: Vec<f32>,
    global_hidden_v: Vec<f32>,
    global_bias_m: Vec<f32>,
    global_bias_v: Vec<f32>,
    value_logits_hidden_m: Vec<f32>,
    value_logits_hidden_v: Vec<f32>,
    value_logits_global_m: Vec<f32>,
    value_logits_global_v: Vec<f32>,
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
            global_hidden_m: vec![0.0; model.global_hidden.len()],
            global_hidden_v: vec![0.0; model.global_hidden.len()],
            global_bias_m: vec![0.0; model.global_bias.len()],
            global_bias_v: vec![0.0; model.global_bias.len()],
            value_logits_hidden_m: vec![0.0; model.value_logits_hidden.len()],
            value_logits_hidden_v: vec![0.0; model.value_logits_hidden.len()],
            value_logits_global_m: vec![0.0; model.value_logits_global.len()],
            value_logits_global_v: vec![0.0; model.value_logits_global.len()],
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
            && self.global_hidden_m.len() == model.global_hidden.len()
            && self.global_hidden_v.len() == model.global_hidden.len()
            && self.global_bias_m.len() == model.global_bias.len()
            && self.global_bias_v.len() == model.global_bias.len()
            && self.value_logits_hidden_m.len() == model.value_logits_hidden.len()
            && self.value_logits_hidden_v.len() == model.value_logits_hidden.len()
            && self.value_logits_global_m.len() == model.value_logits_global.len()
            && self.value_logits_global_v.len() == model.value_logits_global.len()
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
        let global_hidden = (0..GLOBAL_CONTEXT_SIZE * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt()))
            .collect();
        let global_bias = vec![0.0; GLOBAL_CONTEXT_SIZE];
        let value_logits_hidden = (0..VALUE_LOGITS * hidden_size)
            .map(|_| rng.weight(0.05))
            .collect();
        let value_logits_global = (0..VALUE_LOGITS * GLOBAL_CONTEXT_SIZE)
            .map(|_| rng.weight(0.05))
            .collect();
        let value_logits_bias = vec![0.0; VALUE_LOGITS];
        let policy_move_hidden = (0..MOVE_SPACE * hidden_size)
            .map(|_| rng.weight(0.01))
            .collect();
        let policy_move_global = (0..MOVE_SPACE * GLOBAL_CONTEXT_SIZE)
            .map(|_| rng.weight(0.01))
            .collect();
        let policy_move_bias = vec![0.0; MOVE_SPACE];
        Self {
            hidden_size,
            trunk_depth,
            input_hidden,
            hidden_bias,
            trunk_weights,
            trunk_biases,
            global_hidden,
            global_bias,
            value_logits_hidden,
            value_logits_global,
            value_logits_bias,
            policy_move_hidden,
            policy_move_global,
            policy_move_bias,
            optimizer: None,
        }
    }

    pub fn save_text(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let text = format!(
            "format: {AZNNUE_FORMAT}\ninput_size: {V4_INPUT_SIZE}\nhidden_size: {}\ntrunk_depth: {}\ninput_hidden: {}\nhidden_bias: {}\ntrunk_weights: {}\ntrunk_biases: {}\nglobal_hidden: {}\nglobal_bias: {}\nvalue_logits_hidden: {}\nvalue_logits_global: {}\nvalue_logits_bias: {}\npolicy_move_hidden: {}\npolicy_move_global: {}\npolicy_move_bias: {}\n",
            self.hidden_size,
            self.trunk_depth,
            format_floats(&self.input_hidden),
            format_floats(&self.hidden_bias),
            format_floats(&self.trunk_weights),
            format_floats(&self.trunk_biases),
            format_floats(&self.global_hidden),
            format_floats(&self.global_bias),
            format_floats(&self.value_logits_hidden),
            format_floats(&self.value_logits_global),
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
        let mut global_hidden = None;
        let mut global_bias = None;
        let mut value_logits_hidden = None;
        let mut value_logits_global = None;
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
                "global_hidden" => global_hidden = Some(parse_floats(value)?),
                "global_bias" => global_bias = Some(parse_floats(value)?),
                "value_logits_hidden" => value_logits_hidden = Some(parse_floats(value)?),
                "value_logits_global" => value_logits_global = Some(parse_floats(value)?),
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
            global_hidden: global_hidden.ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing global_hidden")
            })?,
            global_bias: global_bias
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing global_bias"))?,
            value_logits_hidden: value_logits_hidden.ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing value_logits_hidden")
            })?,
            value_logits_global: value_logits_global.ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing value_logits_global")
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
        let hidden = self.embedding(position, history);
        let global = self.global_from_hidden(&hidden);
        let value = self.value_from_hidden(&hidden, &global);
        let side = position.side_to_move();
        let logits = moves
            .iter()
            .map(|mv| self.policy_logit_from_hidden(&hidden, &global, orient_move(side, *mv)))
            .collect();
        (value, logits)
    }

    fn validate(&self) -> io::Result<()> {
        if self.input_hidden.len() != V4_INPUT_SIZE * self.hidden_size
            || self.hidden_bias.len() != self.hidden_size
            || self.trunk_weights.len() != self.trunk_depth * self.hidden_size * self.hidden_size
            || self.trunk_biases.len() != self.trunk_depth * self.hidden_size
            || self.global_hidden.len() != GLOBAL_CONTEXT_SIZE * self.hidden_size
            || self.global_bias.len() != GLOBAL_CONTEXT_SIZE
            || self.value_logits_hidden.len() != VALUE_LOGITS * self.hidden_size
            || self.value_logits_global.len() != VALUE_LOGITS * GLOBAL_CONTEXT_SIZE
            || self.value_logits_bias.len() != VALUE_LOGITS
            || self.policy_move_hidden.len() != MOVE_SPACE * self.hidden_size
            || self.policy_move_global.len() != MOVE_SPACE * GLOBAL_CONTEXT_SIZE
            || self.policy_move_bias.len() != MOVE_SPACE
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "aznnue vector length mismatch",
            ));
        }
        Ok(())
    }

    fn embedding(&self, position: &Position, history: &[HistoryMove]) -> Vec<f32> {
        self.embedding_from_features(&extract_sparse_features_v4(position, history))
    }

    fn embedding_from_features(&self, features: &[usize]) -> Vec<f32> {
        let mut hidden = self.hidden_bias.clone();
        for &feature in features {
            let row =
                &self.input_hidden[feature * self.hidden_size..(feature + 1) * self.hidden_size];
            for idx in 0..self.hidden_size {
                hidden[idx] += row[idx];
            }
        }
        for value in &mut hidden {
            *value = value.max(0.0);
        }
        self.forward_trunk(hidden)
    }

    fn forward_trunk(&self, mut hidden: Vec<f32>) -> Vec<f32> {
        let mut next = vec![0.0; self.hidden_size];
        for layer in 0..self.trunk_depth {
            let weight_offset = layer * self.hidden_size * self.hidden_size;
            let bias_offset = layer * self.hidden_size;
            for out in 0..self.hidden_size {
                let mut value = self.trunk_biases[bias_offset + out];
                let row = &self.trunk_weights[weight_offset + out * self.hidden_size
                    ..weight_offset + (out + 1) * self.hidden_size];
                for idx in 0..self.hidden_size {
                    value += row[idx] * hidden[idx];
                }
                next[out] = hidden[out] + RESIDUAL_TRUNK_SCALE * value.max(0.0);
            }
            std::mem::swap(&mut hidden, &mut next);
        }
        hidden
    }

    fn global_from_hidden(&self, hidden: &[f32]) -> Vec<f32> {
        let mut global = self.global_bias.clone();
        for out in 0..GLOBAL_CONTEXT_SIZE {
            let row = &self.global_hidden[out * self.hidden_size..(out + 1) * self.hidden_size];
            for idx in 0..self.hidden_size {
                global[out] += hidden[idx] * row[idx];
            }
            global[out] = global[out].max(0.0);
        }
        global
    }

    fn value_logits_from_hidden(&self, hidden: &[f32], global: &[f32]) -> Vec<f32> {
        let mut logits = self.value_logits_bias.clone();
        for out in 0..VALUE_LOGITS {
            let hidden_row =
                &self.value_logits_hidden[out * self.hidden_size..(out + 1) * self.hidden_size];
            for idx in 0..self.hidden_size {
                logits[out] += hidden[idx] * hidden_row[idx];
            }
            let global_row = &self.value_logits_global
                [out * GLOBAL_CONTEXT_SIZE..(out + 1) * GLOBAL_CONTEXT_SIZE];
            for idx in 0..GLOBAL_CONTEXT_SIZE {
                logits[out] += global[idx] * global_row[idx];
            }
        }
        logits
    }

    fn value_from_hidden(&self, hidden: &[f32], global: &[f32]) -> f32 {
        let logits = self.value_logits_from_hidden(hidden, global);
        scalar_value_from_logits(&logits).0
    }

    fn policy_logit_from_hidden(&self, hidden: &[f32], global: &[f32], mv: Move) -> f32 {
        let move_index = mv.from as usize * BOARD_SIZE + mv.to as usize;
        let hidden_offset = move_index * self.hidden_size;
        let hidden_row = &self.policy_move_hidden[hidden_offset..hidden_offset + self.hidden_size];
        let global_offset = move_index * GLOBAL_CONTEXT_SIZE;
        let global_row =
            &self.policy_move_global[global_offset..global_offset + GLOBAL_CONTEXT_SIZE];
        let mut logit = self.policy_move_bias[move_index];
        for (idx, hidden_value) in hidden.iter().enumerate() {
            logit += hidden_value * hidden_row[idx];
        }
        for (idx, global_value) in global.iter().enumerate() {
            logit += global_value * global_row[idx];
        }
        logit
    }
}

pub fn selfplay_train_iteration(model: &mut AzNnue, config: &AzLoopConfig) -> AzLoopReport {
    selfplay_train_iteration_with_pool(model, config, None)
}

pub fn selfplay_train_iteration_with_pool(
    model: &mut AzNnue,
    config: &AzLoopConfig,
    pool: Option<&mut AzExperiencePool>,
) -> AzLoopReport {
    let data = generate_selfplay_data(model, config);
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
    let stats = train_samples(
        model,
        &train_data,
        config.epochs,
        config.lr,
        config.batch_size,
        config.train_workers,
        &mut rng,
    );
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
        value_mse: stats.value_mse,
        policy_ce: stats.policy_ce,
        value_pred_mean: stats.value_pred_sum / stats.samples.max(1) as f32,
        value_pred_std: variance_to_std(
            stats.value_pred_sum,
            stats.value_pred_sq_sum,
            stats.samples,
        ),
        value_target_mean: stats.value_target_sum / stats.samples.max(1) as f32,
        value_target_std: variance_to_std(
            stats.value_target_sum,
            stats.value_target_sq_sum,
            stats.samples,
        ),
        train_samples: train_data.len(),
        pool_games,
        pool_samples,
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
                finalize_last_transition(&mut game_samples, -1.0);
                break;
            }
            if !position.has_general(Color::Black) {
                result = Some(1.0);
                finalize_last_transition(&mut game_samples, 1.0);
                break;
            }
            if let Some(rule_outcome) = position.rule_outcome_with_history(&rule_history) {
                result = Some(match rule_outcome {
                    RuleOutcome::Draw => 0.0,
                    RuleOutcome::Win(Color::Red) => 1.0,
                    RuleOutcome::Win(Color::Black) => -1.0,
                });
                finalize_last_transition(&mut game_samples, result.unwrap_or(0.0));
                break;
            }
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
        Self {
            nodes: vec![AzNode {
                position,
                history,
                rule_history: rule_history.clone(),
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
        if let Some(rule_history) = self.nodes[node_index].rule_history.as_ref() {
            if let Some(outcome) = self.nodes[node_index]
                .position
                .rule_outcome_with_history(rule_history)
            {
                self.nodes[node_index].children.clear();
                self.nodes[node_index].value = match outcome {
                    RuleOutcome::Draw => 0.0,
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
        let (value, logits) = self.model.evaluate(
            &self.nodes[node_index].position,
            &self.nodes[node_index].history,
            &moves,
        );
        let priors = softmax(&logits);
        self.nodes[node_index].children = moves
            .into_iter()
            .zip(logits)
            .zip(priors)
            .map(|((mv, prior_logit), prior)| AzChild {
                mv,
                prior,
                prior_logit,
                visits: 0,
                value_sum: 0.0,
                child: None,
            })
            .collect();
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
                let mut child_rule_history = self.nodes[node_index].rule_history.clone();
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
    AzTrainingSample {
        features,
        moves,
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

fn train_samples(
    model: &mut AzNnue,
    samples: &[AzTrainingSample],
    epochs: usize,
    lr: f32,
    batch_size: usize,
    train_workers: usize,
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
                train_workers,
            );
            stats.add_assign(&batch_stats);
        }
    }
    model.optimizer = Some(optimizer);
    if stats.samples > 0 {
        let denom = stats.samples as f32;
        stats.loss /= denom;
        stats.value_mse /= denom;
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
    train_workers: usize,
) -> AzTrainStats {
    let worker_count = train_workers.max(1).min(batch.len().max(1));
    let stats = if worker_count == 1 || batch.len() <= 1 {
        gradient.clear();
        let mut stats = AzTrainStats::default();
        for &index in batch {
            let sample_stats = accumulate_one(model, gradient, &samples[index]);
            stats.add_assign(&sample_stats);
        }
        stats
    } else {
        gradient.clear();
        let chunk_size = (batch.len() + worker_count - 1) / worker_count;
        let model_ref: &AzNnue = &*model;
        let samples_ref = samples;
        let partials = thread::scope(|scope| {
            let mut handles = Vec::new();
            for chunk in batch.chunks(chunk_size) {
                handles.push(scope.spawn(move || {
                    let mut local_gradient = AzGrad::new(model_ref);
                    let mut local_stats = AzTrainStats::default();
                    for &index in chunk {
                        let sample_stats =
                            accumulate_one(model_ref, &mut local_gradient, &samples_ref[index]);
                        local_stats.add_assign(&sample_stats);
                    }
                    (local_gradient, local_stats)
                }));
            }

            let mut partials = Vec::with_capacity(handles.len());
            for handle in handles {
                partials.push(handle.join().expect("training worker panicked"));
            }
            partials
        });

        let mut stats = AzTrainStats::default();
        for (local_gradient, local_stats) in partials {
            gradient.add_assign(&local_gradient);
            stats.add_assign(&local_stats);
        }
        stats
    };
    if stats.samples > 0 {
        apply_adamw_gradient(model, optimizer, gradient, lr, stats.samples as f32);
    }
    stats
}

fn accumulate_one(
    model: &AzNnue,
    gradient: &mut AzGrad,
    sample: &AzTrainingSample,
) -> AzTrainStats {
    let activations = train_forward_activations(model, &sample.features);
    let hidden = activations
        .last()
        .expect("at least input activation exists");
    let global = model.global_from_hidden(hidden);

    let value_logits = model.value_logits_from_hidden(hidden, &global);
    let (value, value_probs) = scalar_value_from_logits(&value_logits);
    let value_error = value - sample.value;
    let value_mse = value_error * value_error;
    let value_grad_scale = 2.0 * value_error;

    let logits = sample
        .moves
        .iter()
        .map(|mv| model.policy_logit_from_hidden(hidden, &global, *mv))
        .collect::<Vec<_>>();
    let prediction = softmax(&logits);
    let policy_ce = prediction
        .iter()
        .zip(&sample.policy)
        .map(|(predicted, target)| -target * predicted.max(1e-9).ln())
        .sum::<f32>();

    let mut activation_grad = vec![0.0; model.hidden_size];
    let mut global_grad = vec![0.0; GLOBAL_CONTEXT_SIZE];
    for out in 0..VALUE_LOGITS {
        let target_sign = match out {
            0 => 1.0,
            1 => 0.0,
            _ => -1.0,
        };
        let logit_grad =
            (value_grad_scale * value_probs[out] * (target_sign - value)).clamp(-4.0, 4.0);
        let hidden_offset = out * model.hidden_size;
        for idx in 0..model.hidden_size {
            activation_grad[idx] += logit_grad * model.value_logits_hidden[hidden_offset + idx];
            gradient.value_logits_hidden[hidden_offset + idx] += logit_grad * hidden[idx];
        }
        let global_offset = out * GLOBAL_CONTEXT_SIZE;
        for idx in 0..GLOBAL_CONTEXT_SIZE {
            global_grad[idx] += logit_grad * model.value_logits_global[global_offset + idx];
            gradient.value_logits_global[global_offset + idx] += logit_grad * global[idx];
        }
        gradient.value_logits_bias[out] += logit_grad;
    }
    for ((mv, predicted), target) in sample.moves.iter().zip(&prediction).zip(&sample.policy) {
        let move_index = mv.from as usize * BOARD_SIZE + mv.to as usize;
        let policy_grad = (predicted - target).clamp(-4.0, 4.0);
        let hidden_offset = move_index * model.hidden_size;
        for idx in 0..model.hidden_size {
            activation_grad[idx] += policy_grad * model.policy_move_hidden[hidden_offset + idx];
        }
        let global_offset = move_index * GLOBAL_CONTEXT_SIZE;
        for idx in 0..GLOBAL_CONTEXT_SIZE {
            global_grad[idx] += policy_grad * model.policy_move_global[global_offset + idx];
        }
    }

    for ((mv, predicted), target) in sample.moves.iter().zip(&prediction).zip(&sample.policy) {
        let move_index = mv.from as usize * BOARD_SIZE + mv.to as usize;
        let policy_grad = (predicted - target).clamp(-4.0, 4.0);
        let hidden_offset = move_index * model.hidden_size;
        for idx in 0..model.hidden_size {
            let weight_index = hidden_offset + idx;
            gradient.policy_move_hidden[weight_index] += policy_grad * hidden[idx];
        }
        let global_offset = move_index * GLOBAL_CONTEXT_SIZE;
        for idx in 0..GLOBAL_CONTEXT_SIZE {
            let weight_index = global_offset + idx;
            gradient.policy_move_global[weight_index] += policy_grad * global[idx];
        }
        gradient.policy_move_bias[move_index] += policy_grad;
    }

    for out in 0..GLOBAL_CONTEXT_SIZE {
        if global[out] <= 0.0 {
            continue;
        }
        let grad = global_grad[out].clamp(-4.0, 4.0);
        let weight_offset = out * model.hidden_size;
        for idx in 0..model.hidden_size {
            let weight_index = weight_offset + idx;
            activation_grad[idx] += grad * model.global_hidden[weight_index];
            gradient.global_hidden[weight_index] += grad * hidden[idx];
        }
        gradient.global_bias[out] += grad;
    }

    let mut input_grad = activation_grad;
    for layer in (0..model.trunk_depth).rev() {
        let output = &activations[layer + 1];
        let input = &activations[layer];
        let mut previous_grad = vec![0.0; model.hidden_size];
        let weight_offset = layer * model.hidden_size * model.hidden_size;
        let bias_offset = layer * model.hidden_size;
        for out in 0..model.hidden_size {
            let grad = input_grad[out].clamp(-4.0, 4.0);
            previous_grad[out] += grad;
            if output[out] <= input[out] {
                continue;
            }
            let residual_grad = grad * RESIDUAL_TRUNK_SCALE;
            for idx in 0..model.hidden_size {
                let weight_index = weight_offset + out * model.hidden_size + idx;
                let old_weight = model.trunk_weights[weight_index];
                previous_grad[idx] += residual_grad * old_weight;
                gradient.trunk_weights[weight_index] += residual_grad * input[idx];
            }
            let bias_index = bias_offset + out;
            gradient.trunk_biases[bias_index] += residual_grad;
        }
        input_grad = previous_grad;
    }

    let input_scale = 1.0 / (sample.features.len() as f32).sqrt().max(1.0);
    for idx in 0..model.hidden_size {
        if activations[0][idx] <= 0.0 {
            continue;
        }
        let grad = input_grad[idx].clamp(-4.0, 4.0);
        gradient.hidden_bias[idx] += grad;
        for &feature in &sample.features {
            let weight_index = feature * model.hidden_size + idx;
            gradient.input_hidden[weight_index] += grad * input_scale;
        }
    }

    AzTrainStats {
        loss: value_mse + policy_ce,
        value_mse,
        policy_ce,
        value_pred_sum: value,
        value_pred_sq_sum: value * value,
        value_target_sum: sample.value,
        value_target_sq_sum: sample.value * sample.value,
        samples: 1,
    }
}

fn train_forward_activations(model: &AzNnue, features: &[usize]) -> Vec<Vec<f32>> {
    let mut hidden = model.hidden_bias.clone();
    for &feature in features {
        let row =
            &model.input_hidden[feature * model.hidden_size..(feature + 1) * model.hidden_size];
        for idx in 0..model.hidden_size {
            hidden[idx] += row[idx];
        }
    }
    for value in &mut hidden {
        *value = value.max(0.0);
    }

    let mut activations = Vec::with_capacity(model.trunk_depth + 1);
    activations.push(hidden);
    for layer in 0..model.trunk_depth {
        let previous = activations.last().expect("previous activation exists");
        let mut next = vec![0.0; model.hidden_size];
        let weight_offset = layer * model.hidden_size * model.hidden_size;
        let bias_offset = layer * model.hidden_size;
        for out in 0..model.hidden_size {
            let mut value = model.trunk_biases[bias_offset + out];
            let row = &model.trunk_weights[weight_offset + out * model.hidden_size
                ..weight_offset + (out + 1) * model.hidden_size];
            for idx in 0..model.hidden_size {
                value += row[idx] * previous[idx];
            }
            next[out] = previous[out] + RESIDUAL_TRUNK_SCALE * value.max(0.0);
        }
        activations.push(next);
    }
    activations
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
    for idx in 0..model.value_logits_hidden.len() {
        adamw_update(
            &mut model.value_logits_hidden[idx],
            &mut optimizer.value_logits_hidden_m[idx],
            &mut optimizer.value_logits_hidden_v[idx],
            gradient.value_logits_hidden[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            ADAMW_WEIGHT_DECAY,
        );
    }
    for idx in 0..model.value_logits_global.len() {
        adamw_update(
            &mut model.value_logits_global[idx],
            &mut optimizer.value_logits_global_m[idx],
            &mut optimizer.value_logits_global_v[idx],
            gradient.value_logits_global[idx] * inv_batch,
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
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    let mut values = Vec::with_capacity(logits.len());
    for logit in logits {
        let value = (*logit - max_logit).exp();
        sum += value;
        values.push(value);
    }
    for value in &mut values {
        *value /= sum.max(1e-12);
    }
    values
}

fn scalar_value_from_logits(logits: &[f32]) -> (f32, Vec<f32>) {
    let probs = softmax(logits);
    if probs.len() < VALUE_LOGITS {
        return (0.0, probs);
    }
    (probs[0] - probs[2], probs)
}

fn variance_to_std(sum: f32, sq_sum: f32, count: usize) -> f32 {
    if count == 0 {
        return 0.0;
    }
    let mean = sum / count as f32;
    let variance = (sq_sum / count as f32) - mean * mean;
    variance.max(0.0).sqrt()
}

fn add_assign_slice(dst: &mut [f32], src: &[f32]) {
    for (left, right) in dst.iter_mut().zip(src) {
        *left += *right;
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
                moves: Vec::new(),
                policy: Vec::new(),
                value: 0.0,
                side_sign: 1.0,
                reward: 0.0,
                discount: -1.0,
                bootstrap_value: 0.2,
            },
            AzTrainingSample {
                features: Vec::new(),
                moves: Vec::new(),
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
                moves: Vec::new(),
                policy: Vec::new(),
                value: 1.0,
                side_sign: 1.0,
                reward: 0.0,
                discount: 0.0,
                bootstrap_value: 0.0,
            },
            AzTrainingSample {
                features: vec![1],
                moves: Vec::new(),
                policy: Vec::new(),
                value: -1.0,
                side_sign: 1.0,
                reward: 0.0,
                discount: 0.0,
                bootstrap_value: 0.0,
            },
            AzTrainingSample {
                features: vec![2],
                moves: Vec::new(),
                policy: Vec::new(),
                value: 0.75,
                side_sign: 1.0,
                reward: 0.0,
                discount: 0.0,
                bootstrap_value: 0.0,
            },
            AzTrainingSample {
                features: vec![3],
                moves: Vec::new(),
                policy: Vec::new(),
                value: -0.75,
                side_sign: 1.0,
                reward: 0.0,
                discount: 0.0,
                bootstrap_value: 0.0,
            },
        ];

        let mut rng = SplitMix64::new(17);
        let before = train_samples(&mut model, &samples, 1, 0.003, 4, 1, &mut rng).value_mse;
        let after = train_samples(&mut model, &samples, 300, 0.003, 4, 1, &mut rng).value_mse;

        assert!(after < before * 0.2, "before={before} after={after}");
        assert!(after < 0.05, "after={after}");
    }

    #[test]
    fn synchronized_parallel_training_matches_single_thread() {
        let samples = vec![
            AzTrainingSample {
                features: vec![0, 4, 8],
                moves: Vec::new(),
                policy: Vec::new(),
                value: 1.0,
                side_sign: 1.0,
                reward: 0.0,
                discount: 0.0,
                bootstrap_value: 0.0,
            },
            AzTrainingSample {
                features: vec![1, 5, 9],
                moves: Vec::new(),
                policy: Vec::new(),
                value: -1.0,
                side_sign: 1.0,
                reward: 0.0,
                discount: 0.0,
                bootstrap_value: 0.0,
            },
            AzTrainingSample {
                features: vec![2, 6, 10],
                moves: Vec::new(),
                policy: Vec::new(),
                value: 0.5,
                side_sign: 1.0,
                reward: 0.0,
                discount: 0.0,
                bootstrap_value: 0.0,
            },
            AzTrainingSample {
                features: vec![3, 7, 11],
                moves: Vec::new(),
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
        let mut parallel = single.clone();

        let mut rng_single = SplitMix64::new(99);
        let mut rng_parallel = SplitMix64::new(99);
        let single_stats = train_samples(&mut single, &samples, 5, 0.003, 4, 1, &mut rng_single);
        let parallel_stats =
            train_samples(&mut parallel, &samples, 5, 0.003, 4, 4, &mut rng_parallel);

        assert!((single_stats.loss - parallel_stats.loss).abs() < 1e-5);
        assert!((single_stats.value_mse - parallel_stats.value_mse).abs() < 1e-5);
        assert!((single_stats.value_pred_sum - parallel_stats.value_pred_sum).abs() < 1e-4);
        assert!((single_stats.value_target_sum - parallel_stats.value_target_sum).abs() < 1e-6);
        assert!(single
            .value_logits_bias
            .iter()
            .zip(&parallel.value_logits_bias)
            .all(|(left, right)| (*left - *right).abs() < 1e-5));
    }
}
