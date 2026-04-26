use std::fs;
use std::io::{self, BufWriter, Cursor, Read, Write};
use std::path::Path;

mod alphazero;
mod mctx;
mod play;
mod replay;
mod train;
mod train_gpu;

use crate::nnue::{HISTORY_PLIES, HistoryMove, V4_INPUT_SIZE, extract_sparse_features_v4};
use crate::xiangqi::{BOARD_FILES, BOARD_SIZE, Color, Move, PieceKind, Position};

pub use alphazero::{
    AzCandidate, AzSearchAlgorithm, AzSearchLimits, AzSearchResult, alphazero_search,
    alphazero_search_with_history_and_rules,
};
pub use mctx::AzGumbelConfig;
pub use play::{
    AzArenaReport, AzSelfplayData, AzTerminalStats, generate_selfplay_data,
    play_arena_games_from_positions,
};
pub use replay::AzExperiencePool;
pub use train::{global_training_step_sample_count, train_samples};

pub const AZNNUE_BINARY_MAGIC: &[u8] = b"AZB1";
const AZNNUE_BINARY_VERSION: u32 = 8;
const AZNNUE_BINARY_HEADER_LEN: usize = 24;

fn write_f32_slice_le<W: Write>(writer: &mut W, slice: &[f32]) -> io::Result<()> {
    for &value in slice {
        writer.write_all(&value.to_bits().to_le_bytes())?;
    }
    Ok(())
}

fn read_u32_le(reader: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_f32_vec_le(reader: &mut impl Read, len: usize) -> io::Result<Vec<f32>> {
    let mut out = vec![0.0f32; len];
    let mut buf = [0u8; 4];
    for slot in &mut out {
        reader.read_exact(&mut buf)?;
        *slot = f32::from_bits(u32::from_le_bytes(buf));
    }
    Ok(out)
}

const SPARSE_MOVE_SPACE: usize = BOARD_SIZE * BOARD_SIZE;
const DENSE_MOVE_SPACE: usize = compute_dense_move_count();
const GLOBAL_CONTEXT_SIZE: usize = 32;
const VALUE_HIDDEN_SIZE: usize = 64;
const VALUE_LOGITS: usize = 3;
pub(super) const BOARD_PLANES_SIZE: usize = BOARD_SIZE;
pub(super) const BOARD_HISTORY_FRAMES: usize = HISTORY_PLIES + 1;
pub(super) const BOARD_HISTORY_SIZE: usize = BOARD_HISTORY_FRAMES * BOARD_PLANES_SIZE;
pub(super) const PIECE_BOARD_CHANNELS: usize = 14;
pub(super) const SIDE_TO_MOVE_BOARD_CHANNEL: usize = PIECE_BOARD_CHANNELS * BOARD_HISTORY_FRAMES;
pub(super) const BOARD_CHANNELS: usize = SIDE_TO_MOVE_BOARD_CHANNEL + 1;
const CNN_CHANNELS: usize = 16;
const CNN_KERNEL_AREA: usize = 9;
pub(super) const CNN_POOLED_SIZE: usize = CNN_CHANNELS * 3;
const VALUE_SCALE_CP: f32 = 1000.0;
const RESIDUAL_TRUNK_SCALE: f32 = 0.5;
pub(super) struct AzEvalScratch {
    hidden: Vec<f32>,
    next: Vec<f32>,
    board: Vec<u8>,
    conv1: Vec<f32>,
    conv2: Vec<f32>,
    cnn_global: Vec<f32>,
    global: Vec<f32>,
    value_intermediate: Vec<f32>,
    value_logits: Vec<f32>,
    logits: Vec<f32>,
    priors: Vec<f32>,
}

impl AzEvalScratch {
    pub(super) fn new(hidden_size: usize) -> Self {
        Self {
            hidden: vec![0.0; hidden_size],
            next: vec![0.0; hidden_size],
            board: vec![0; BOARD_HISTORY_SIZE],
            conv1: vec![0.0; CNN_CHANNELS * BOARD_PLANES_SIZE],
            conv2: vec![0.0; CNN_CHANNELS * BOARD_PLANES_SIZE],
            cnn_global: vec![0.0; CNN_POOLED_SIZE],
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
    pub board_conv1_weights: Vec<f32>,
    pub board_conv1_bias: Vec<f32>,
    pub board_conv2_weights: Vec<f32>,
    pub board_conv2_bias: Vec<f32>,
    pub board_attention_query: Vec<f32>,
    pub board_hidden: Vec<f32>,
    pub board_hidden_bias: Vec<f32>,
    pub board_global: Vec<f32>,
    pub global_hidden: Vec<f32>,
    pub global_bias: Vec<f32>,
    pub value_intermediate_hidden: Vec<f32>,
    pub value_intermediate_bias: Vec<f32>,
    pub value_logits_weights: Vec<f32>,
    pub value_logits_bias: Vec<f32>,
    pub policy_move_hidden: Vec<f32>,
    pub policy_move_cnn: Vec<f32>,
    pub policy_move_bias: Vec<f32>,
    gpu_trainer: Option<Box<train_gpu::GpuTrainer>>,
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
            board_conv1_weights: self.board_conv1_weights.clone(),
            board_conv1_bias: self.board_conv1_bias.clone(),
            board_conv2_weights: self.board_conv2_weights.clone(),
            board_conv2_bias: self.board_conv2_bias.clone(),
            board_attention_query: self.board_attention_query.clone(),
            board_hidden: self.board_hidden.clone(),
            board_hidden_bias: self.board_hidden_bias.clone(),
            board_global: self.board_global.clone(),
            global_hidden: self.global_hidden.clone(),
            global_bias: self.global_bias.clone(),
            value_intermediate_hidden: self.value_intermediate_hidden.clone(),
            value_intermediate_bias: self.value_intermediate_bias.clone(),
            value_logits_weights: self.value_logits_weights.clone(),
            value_logits_bias: self.value_logits_bias.clone(),
            policy_move_hidden: self.policy_move_hidden.clone(),
            policy_move_cnn: self.policy_move_cnn.clone(),
            policy_move_bias: self.policy_move_bias.clone(),
            gpu_trainer: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AzLoopConfig {
    pub games: usize,
    pub max_plies: usize,
    pub simulations: usize,
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
    pub td_lambda: f32,
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
    pub value_mse: f32,
    pub value_pred_mean: f32,
    pub value_target_mean: f32,
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
pub struct AzTrainBenchmark {
    pub loss: f32,
    pub value_loss: f32,
    pub policy_ce: f32,
}

#[derive(Clone, Debug)]
pub struct AzTrainingSample {
    pub features: Vec<usize>,
    pub board: Vec<u8>,
    pub move_indices: Vec<usize>,
    pub policy: Vec<f32>,
    pub value: f32,
    pub side_sign: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AzTrainStats {
    pub loss: f32,
    pub value_loss: f32,
    pub policy_ce: f32,
    pub value_pred_sum: f32,
    pub value_pred_sq_sum: f32,
    pub value_target_sum: f32,
    pub value_target_sq_sum: f32,
    pub value_error_sq_sum: f32,
    pub samples: usize,
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
        self.value_error_sq_sum += other.value_error_sq_sum;
        self.samples += other.samples;
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
        let board_conv1_weights = (0..CNN_CHANNELS * BOARD_CHANNELS * CNN_KERNEL_AREA)
            .map(|_| rng.weight(0.08))
            .collect();
        let board_conv1_bias = vec![0.0; CNN_CHANNELS];
        let board_conv2_weights = (0..CNN_CHANNELS * CNN_CHANNELS * CNN_KERNEL_AREA)
            .map(|_| rng.weight(0.08))
            .collect();
        let board_conv2_bias = vec![0.0; CNN_CHANNELS];
        let board_attention_query = (0..CNN_CHANNELS).map(|_| rng.weight(0.08)).collect();
        let board_hidden = (0..hidden_size * CNN_POOLED_SIZE)
            .map(|_| rng.weight((2.0 / CNN_POOLED_SIZE as f32).sqrt()))
            .collect();
        let board_hidden_bias = vec![0.0; hidden_size];
        let board_global = (0..GLOBAL_CONTEXT_SIZE * CNN_POOLED_SIZE)
            .map(|_| rng.weight((2.0 / CNN_POOLED_SIZE as f32).sqrt()))
            .collect();
        let global_hidden = (0..GLOBAL_CONTEXT_SIZE * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt()))
            .collect();
        let global_bias = vec![0.0; GLOBAL_CONTEXT_SIZE];
        let value_intermediate_hidden = (0..VALUE_HIDDEN_SIZE * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt()))
            .collect();
        let value_intermediate_bias = vec![0.0; VALUE_HIDDEN_SIZE];
        let value_logits_weights = (0..VALUE_LOGITS * VALUE_HIDDEN_SIZE)
            .map(|_| rng.weight((2.0 / VALUE_HIDDEN_SIZE as f32).sqrt()))
            .collect();
        let value_logits_bias = vec![0.0; VALUE_LOGITS];
        let policy_move_hidden = (0..DENSE_MOVE_SPACE * hidden_size)
            .map(|_| rng.weight(0.01))
            .collect();
        let policy_move_cnn = (0..DENSE_MOVE_SPACE * CNN_POOLED_SIZE)
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
            board_conv1_weights,
            board_conv1_bias,
            board_conv2_weights,
            board_conv2_bias,
            board_attention_query,
            board_hidden,
            board_hidden_bias,
            board_global,
            global_hidden,
            global_bias,
            value_intermediate_hidden,
            value_intermediate_bias,
            value_logits_weights,
            value_logits_bias,
            policy_move_hidden,
            policy_move_cnn,
            policy_move_bias,
            gpu_trainer: None,
        }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let file = fs::File::create(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(AZNNUE_BINARY_MAGIC)?;
        writer.write_all(&AZNNUE_BINARY_VERSION.to_le_bytes())?;
        writer.write_all(&(V4_INPUT_SIZE as u32).to_le_bytes())?;
        writer.write_all(&(self.hidden_size as u32).to_le_bytes())?;
        writer.write_all(&(self.trunk_depth as u32).to_le_bytes())?;
        writer.write_all(&0u32.to_le_bytes())?;
        write_f32_slice_le(&mut writer, &self.input_hidden)?;
        write_f32_slice_le(&mut writer, &self.hidden_bias)?;
        write_f32_slice_le(&mut writer, &self.trunk_weights)?;
        write_f32_slice_le(&mut writer, &self.trunk_biases)?;
        write_f32_slice_le(&mut writer, &self.trunk_global_weights)?;
        write_f32_slice_le(&mut writer, &self.board_conv1_weights)?;
        write_f32_slice_le(&mut writer, &self.board_conv1_bias)?;
        write_f32_slice_le(&mut writer, &self.board_conv2_weights)?;
        write_f32_slice_le(&mut writer, &self.board_conv2_bias)?;
        write_f32_slice_le(&mut writer, &self.board_attention_query)?;
        write_f32_slice_le(&mut writer, &self.board_hidden)?;
        write_f32_slice_le(&mut writer, &self.board_hidden_bias)?;
        write_f32_slice_le(&mut writer, &self.board_global)?;
        write_f32_slice_le(&mut writer, &self.global_hidden)?;
        write_f32_slice_le(&mut writer, &self.global_bias)?;
        write_f32_slice_le(&mut writer, &self.value_intermediate_hidden)?;
        write_f32_slice_le(&mut writer, &self.value_intermediate_bias)?;
        write_f32_slice_le(&mut writer, &self.value_logits_weights)?;
        write_f32_slice_le(&mut writer, &self.value_logits_bias)?;
        write_f32_slice_le(&mut writer, &self.policy_move_hidden)?;
        write_f32_slice_le(&mut writer, &self.policy_move_cnn)?;
        write_f32_slice_le(&mut writer, &self.policy_move_bias)?;
        writer.flush()?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let bytes = fs::read(path.as_ref())?;
        Self::decode_binary(&bytes)
    }

    fn decode_binary(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < AZNNUE_BINARY_HEADER_LEN || !bytes.starts_with(AZNNUE_BINARY_MAGIC) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "truncated or invalid AZNNUE binary",
            ));
        }
        let mut reader = Cursor::new(&bytes[AZNNUE_BINARY_MAGIC.len()..]);
        let version = read_u32_le(&mut reader)?;
        if version != AZNNUE_BINARY_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported AZNNUE binary version {version} (expected {AZNNUE_BINARY_VERSION})"
                ),
            ));
        }
        let input_size = read_u32_le(&mut reader)? as usize;
        let hidden_size = read_u32_le(&mut reader)? as usize;
        let trunk_depth = read_u32_le(&mut reader)? as usize;
        let _reserved = read_u32_le(&mut reader)?;
        if input_size != V4_INPUT_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "binary input_size does not match this build (V4_INPUT_SIZE)",
            ));
        }
        let input_hidden_len = V4_INPUT_SIZE * hidden_size;
        let hidden_bias_len = hidden_size;
        let trunk_weights_len = trunk_depth * hidden_size * hidden_size;
        let trunk_biases_len = trunk_depth * hidden_size;
        let trunk_global_len = trunk_depth * hidden_size * GLOBAL_CONTEXT_SIZE;
        let board_conv1_weights_len = CNN_CHANNELS * BOARD_CHANNELS * CNN_KERNEL_AREA;
        let board_conv1_bias_len = CNN_CHANNELS;
        let board_conv2_weights_len = CNN_CHANNELS * CNN_CHANNELS * CNN_KERNEL_AREA;
        let board_conv2_bias_len = CNN_CHANNELS;
        let board_attention_query_len = CNN_CHANNELS;
        let board_hidden_len = hidden_size * CNN_POOLED_SIZE;
        let board_hidden_bias_len = hidden_size;
        let board_global_len = GLOBAL_CONTEXT_SIZE * CNN_POOLED_SIZE;
        let global_hidden_len = GLOBAL_CONTEXT_SIZE * hidden_size;
        let global_bias_len = GLOBAL_CONTEXT_SIZE;
        let vih_len = VALUE_HIDDEN_SIZE * hidden_size;
        let vib_len = VALUE_HIDDEN_SIZE;
        let vlw_len = VALUE_LOGITS * VALUE_HIDDEN_SIZE;
        let vlb_len = VALUE_LOGITS;
        let pmh_len = DENSE_MOVE_SPACE * hidden_size;
        let pmc_len = DENSE_MOVE_SPACE * CNN_POOLED_SIZE;
        let pmb_len = DENSE_MOVE_SPACE;
        let float_count = input_hidden_len
            + hidden_bias_len
            + trunk_weights_len
            + trunk_biases_len
            + trunk_global_len
            + board_conv1_weights_len
            + board_conv1_bias_len
            + board_conv2_weights_len
            + board_conv2_bias_len
            + board_attention_query_len
            + board_hidden_len
            + board_hidden_bias_len
            + board_global_len
            + global_hidden_len
            + global_bias_len
            + vih_len
            + vib_len
            + vlw_len
            + vlb_len
            + pmh_len
            + pmc_len
            + pmb_len;
        let expected_len = AZNNUE_BINARY_HEADER_LEN + float_count * 4;
        if bytes.len() != expected_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "AZNNUE binary size mismatch: got {} bytes, expected {} (floats {})",
                    bytes.len(),
                    expected_len,
                    float_count
                ),
            ));
        }
        let input_hidden = read_f32_vec_le(&mut reader, input_hidden_len)?;
        let hidden_bias = read_f32_vec_le(&mut reader, hidden_bias_len)?;
        let trunk_weights = read_f32_vec_le(&mut reader, trunk_weights_len)?;
        let trunk_biases = read_f32_vec_le(&mut reader, trunk_biases_len)?;
        let trunk_global_weights = read_f32_vec_le(&mut reader, trunk_global_len)?;
        let board_conv1_weights = read_f32_vec_le(&mut reader, board_conv1_weights_len)?;
        let board_conv1_bias = read_f32_vec_le(&mut reader, board_conv1_bias_len)?;
        let board_conv2_weights = read_f32_vec_le(&mut reader, board_conv2_weights_len)?;
        let board_conv2_bias = read_f32_vec_le(&mut reader, board_conv2_bias_len)?;
        let board_attention_query = read_f32_vec_le(&mut reader, board_attention_query_len)?;
        let board_hidden = read_f32_vec_le(&mut reader, board_hidden_len)?;
        let board_hidden_bias = read_f32_vec_le(&mut reader, board_hidden_bias_len)?;
        let board_global = read_f32_vec_le(&mut reader, board_global_len)?;
        let global_hidden = read_f32_vec_le(&mut reader, global_hidden_len)?;
        let global_bias = read_f32_vec_le(&mut reader, global_bias_len)?;
        let value_intermediate_hidden = read_f32_vec_le(&mut reader, vih_len)?;
        let value_intermediate_bias = read_f32_vec_le(&mut reader, vib_len)?;
        let value_logits_weights = read_f32_vec_le(&mut reader, vlw_len)?;
        let value_logits_bias = read_f32_vec_le(&mut reader, vlb_len)?;
        let policy_move_hidden = read_f32_vec_le(&mut reader, pmh_len)?;
        let policy_move_cnn = read_f32_vec_le(&mut reader, pmc_len)?;
        let policy_move_bias = read_f32_vec_le(&mut reader, pmb_len)?;
        let model = Self {
            hidden_size,
            trunk_depth,
            input_hidden,
            hidden_bias,
            trunk_weights,
            trunk_biases,
            trunk_global_weights,
            board_conv1_weights,
            board_conv1_bias,
            board_conv2_weights,
            board_conv2_bias,
            board_attention_query,
            board_hidden,
            board_hidden_bias,
            board_global,
            global_hidden,
            global_bias,
            value_intermediate_hidden,
            value_intermediate_bias,
            value_logits_weights,
            value_logits_bias,
            policy_move_hidden,
            policy_move_cnn,
            policy_move_bias,
            gpu_trainer: None,
        };
        model.validate()?;
        Ok(model)
    }

    pub(super) fn evaluate_with_scratch(
        &self,
        position: &Position,
        history: &[HistoryMove],
        moves: &[Move],
        scratch: &mut AzEvalScratch,
    ) -> f32 {
        let features = extract_sparse_features_v4(position, history);
        extract_board_planes(position, history, &mut scratch.board);
        self.input_embedding_into(&features, &mut scratch.hidden);
        self.board_forward_into(
            &scratch.board,
            position.side_to_move() == Color::Black,
            &mut scratch.conv1,
            &mut scratch.conv2,
            &mut scratch.cnn_global,
        );
        self.board_hidden_into(&scratch.cnn_global, &mut scratch.next);
        for idx in 0..self.hidden_size {
            scratch.hidden[idx] += scratch.next[idx];
        }
        self.global_from_hidden_into(&scratch.hidden, &scratch.cnn_global, &mut scratch.global);
        self.forward_trunk_into(&mut scratch.hidden, &mut scratch.next, &scratch.global);
        let value = self.value_from_hidden_scratch(scratch);
        scratch.logits.resize(moves.len(), 0.0);
        for (index, mv) in moves.iter().enumerate() {
            scratch.logits[index] = self.policy_logit_from_hidden_index(
                &scratch.hidden,
                &scratch.cnn_global,
                dense_move_index(*mv),
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

    fn global_from_hidden_into(&self, hidden: &[f32], cnn_global: &[f32], global: &mut Vec<f32>) {
        global.resize(GLOBAL_CONTEXT_SIZE, 0.0);
        global.copy_from_slice(&self.global_bias);
        for out in 0..GLOBAL_CONTEXT_SIZE {
            let row = &self.global_hidden[out * self.hidden_size..(out + 1) * self.hidden_size];
            for idx in 0..self.hidden_size {
                global[out] += hidden[idx] * row[idx];
            }
            let cnn_row = &self.board_global[out * CNN_POOLED_SIZE..(out + 1) * CNN_POOLED_SIZE];
            for idx in 0..CNN_POOLED_SIZE {
                global[out] += cnn_global[idx] * cnn_row[idx];
            }
            global[out] = global[out].max(0.0);
        }
    }

    fn board_hidden_into(&self, cnn_global: &[f32], hidden: &mut Vec<f32>) {
        hidden.resize(self.hidden_size, 0.0);
        hidden.copy_from_slice(&self.board_hidden_bias);
        for out in 0..self.hidden_size {
            let row = &self.board_hidden[out * CNN_POOLED_SIZE..(out + 1) * CNN_POOLED_SIZE];
            for idx in 0..CNN_POOLED_SIZE {
                hidden[out] += cnn_global[idx] * row[idx];
            }
            hidden[out] = hidden[out].max(0.0);
        }
    }

    fn board_forward_into(
        &self,
        board: &[u8],
        black_to_move: bool,
        conv1: &mut Vec<f32>,
        conv2: &mut Vec<f32>,
        cnn_global: &mut Vec<f32>,
    ) {
        conv1.resize(CNN_CHANNELS * BOARD_PLANES_SIZE, 0.0);
        conv2.resize(CNN_CHANNELS * BOARD_PLANES_SIZE, 0.0);
        cnn_global.resize(CNN_POOLED_SIZE, 0.0);
        conv_relu_layer(
            board,
            black_to_move,
            &self.board_conv1_weights,
            &self.board_conv1_bias,
            conv1,
        );
        conv_relu_layer_dense(
            conv1,
            CNN_CHANNELS,
            &self.board_conv2_weights,
            &self.board_conv2_bias,
            conv2,
        );
        let mut attention_logits = [0.0f32; BOARD_PLANES_SIZE];
        let mut max_logit = f32::NEG_INFINITY;
        for sq in 0..BOARD_PLANES_SIZE {
            let mut logit = 0.0;
            for channel in 0..CNN_CHANNELS {
                logit +=
                    conv2[channel * BOARD_PLANES_SIZE + sq] * self.board_attention_query[channel];
            }
            attention_logits[sq] = logit;
            max_logit = max_logit.max(logit);
        }
        let mut denom = 0.0;
        for logit in &mut attention_logits {
            *logit = (*logit - max_logit).exp();
            denom += *logit;
        }
        let scale = 1.0 / BOARD_PLANES_SIZE as f32;
        for channel in 0..CNN_CHANNELS {
            let start = channel * BOARD_PLANES_SIZE;
            let row = &conv2[start..start + BOARD_PLANES_SIZE];
            let mut sum = 0.0;
            let mut max_value = 0.0;
            let mut attn_sum = 0.0;
            for (idx, value) in row.iter().enumerate() {
                sum += *value;
                if idx == 0 || *value > max_value {
                    max_value = *value;
                }
                attn_sum += (*value) * attention_logits[idx] / denom.max(1e-12);
            }
            cnn_global[channel] = sum * scale;
            cnn_global[CNN_CHANNELS + channel] = max_value;
            cnn_global[CNN_CHANNELS * 2 + channel] = attn_sum;
        }
    }

    fn forward_trunk_into(&self, hidden: &mut Vec<f32>, next: &mut Vec<f32>, global: &[f32]) {
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
                let grow = &self.trunk_global_weights[global_weight_offset
                    + out * GLOBAL_CONTEXT_SIZE
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
            let h_row =
                &self.value_intermediate_hidden[j * self.hidden_size..(j + 1) * self.hidden_size];
            for i in 0..self.hidden_size {
                scratch.value_intermediate[j] += scratch.hidden[i] * h_row[i];
            }
            scratch.value_intermediate[j] = scratch.value_intermediate[j].max(0.0);
        }
        scratch
            .value_logits
            .copy_from_slice(&self.value_logits_bias);
        for out in 0..VALUE_LOGITS {
            let row =
                &self.value_logits_weights[out * VALUE_HIDDEN_SIZE..(out + 1) * VALUE_HIDDEN_SIZE];
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
            || self.board_conv1_weights.len() != CNN_CHANNELS * BOARD_CHANNELS * CNN_KERNEL_AREA
            || self.board_conv1_bias.len() != CNN_CHANNELS
            || self.board_conv2_weights.len() != CNN_CHANNELS * CNN_CHANNELS * CNN_KERNEL_AREA
            || self.board_conv2_bias.len() != CNN_CHANNELS
            || self.board_attention_query.len() != CNN_CHANNELS
            || self.board_hidden.len() != self.hidden_size * CNN_POOLED_SIZE
            || self.board_hidden_bias.len() != self.hidden_size
            || self.board_global.len() != GLOBAL_CONTEXT_SIZE * CNN_POOLED_SIZE
            || self.global_hidden.len() != GLOBAL_CONTEXT_SIZE * self.hidden_size
            || self.global_bias.len() != GLOBAL_CONTEXT_SIZE
            || self.value_intermediate_hidden.len() != VALUE_HIDDEN_SIZE * self.hidden_size
            || self.value_intermediate_bias.len() != VALUE_HIDDEN_SIZE
            || self.value_logits_weights.len() != VALUE_LOGITS * VALUE_HIDDEN_SIZE
            || self.value_logits_bias.len() != VALUE_LOGITS
            || self.policy_move_hidden.len() != DENSE_MOVE_SPACE * self.hidden_size
            || self.policy_move_cnn.len() != DENSE_MOVE_SPACE * CNN_POOLED_SIZE
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
        cnn_global: &[f32],
        move_index: usize,
    ) -> f32 {
        let hidden_offset = move_index * self.hidden_size;
        let hidden_row = &self.policy_move_hidden[hidden_offset..hidden_offset + self.hidden_size];
        let cnn_offset = move_index * CNN_POOLED_SIZE;
        let cnn_row = &self.policy_move_cnn[cnn_offset..cnn_offset + CNN_POOLED_SIZE];
        self.policy_move_bias[move_index]
            + dot_product(hidden, hidden_row)
            + dot_product(cnn_global, cnn_row)
    }
}

pub(super) fn extract_board_planes(
    position: &Position,
    history: &[HistoryMove],
    board: &mut Vec<u8>,
) {
    board.resize(BOARD_HISTORY_SIZE, 0);
    board.fill(0);
    let mut frame = [0u8; BOARD_PLANES_SIZE];
    extract_position_piece_planes(position, &mut frame);
    board[..BOARD_PLANES_SIZE].copy_from_slice(&frame);

    let mut rewound = frame;
    for (history_index, entry) in history.iter().rev().take(HISTORY_PLIES).enumerate() {
        let piece_plane = (absolute_piece_plane(entry.piece.color, entry.piece.kind) + 1) as u8;
        rewound[entry.mv.to as usize] = 0;
        rewound[entry.mv.from as usize] = piece_plane;
        let start = (history_index + 1) * BOARD_PLANES_SIZE;
        board[start..start + BOARD_PLANES_SIZE].copy_from_slice(&rewound);
    }
}

fn extract_position_piece_planes(position: &Position, board: &mut [u8; BOARD_PLANES_SIZE]) {
    board.fill(0);
    for sq in 0..BOARD_SIZE {
        let Some(piece) = position.piece_at(sq) else {
            continue;
        };
        let plane = absolute_piece_plane(piece.color, piece.kind);
        board[sq] = (plane + 1) as u8;
    }
}

fn absolute_piece_plane(piece_color: Color, kind: PieceKind) -> usize {
    let own_offset = if piece_color == Color::Red { 0 } else { 7 };
    own_offset
        + match kind {
            PieceKind::General => 0,
            PieceKind::Advisor => 1,
            PieceKind::Elephant => 2,
            PieceKind::Horse => 3,
            PieceKind::Rook => 4,
            PieceKind::Cannon => 5,
            PieceKind::Soldier => 6,
        }
}

fn conv_relu_layer(
    board: &[u8],
    black_to_move: bool,
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    for out_channel in 0..CNN_CHANNELS {
        let out_start = out_channel * BOARD_PLANES_SIZE;
        for sq in 0..BOARD_PLANES_SIZE {
            let mut value = bias[out_channel];
            let file = sq % BOARD_FILES;
            let rank = sq / BOARD_FILES;
            for kr in 0..3 {
                let nr = rank as isize + kr as isize - 1;
                if !(0..10).contains(&nr) {
                    continue;
                }
                for kf in 0..3 {
                    let nf = file as isize + kf as isize - 1;
                    if !(0..BOARD_FILES as isize).contains(&nf) {
                        continue;
                    }
                    let board_index = nr as usize * BOARD_FILES + nf as usize;
                    for frame in 0..BOARD_HISTORY_FRAMES {
                        let plane = board[frame * BOARD_PLANES_SIZE + board_index];
                        if plane == 0 {
                            continue;
                        }
                        let in_channel = frame * PIECE_BOARD_CHANNELS + plane as usize - 1;
                        debug_assert!(in_channel < SIDE_TO_MOVE_BOARD_CHANNEL);
                        let weight_index = ((out_channel * BOARD_CHANNELS + in_channel)
                            * CNN_KERNEL_AREA)
                            + kr * 3
                            + kf;
                        value += weights[weight_index];
                    }
                    if black_to_move {
                        let weight_index = ((out_channel * BOARD_CHANNELS
                            + SIDE_TO_MOVE_BOARD_CHANNEL)
                            * CNN_KERNEL_AREA)
                            + kr * 3
                            + kf;
                        value += weights[weight_index];
                    }
                }
            }
            output[out_start + sq] = value.max(0.0);
        }
    }
}

fn conv_relu_layer_dense(
    input: &[f32],
    input_channels: usize,
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    for out_channel in 0..CNN_CHANNELS {
        let out_start = out_channel * BOARD_PLANES_SIZE;
        for sq in 0..BOARD_PLANES_SIZE {
            let mut value = bias[out_channel];
            let file = sq % BOARD_FILES;
            let rank = sq / BOARD_FILES;
            for kr in 0..3 {
                let nr = rank as isize + kr as isize - 1;
                if !(0..10).contains(&nr) {
                    continue;
                }
                for kf in 0..3 {
                    let nf = file as isize + kf as isize - 1;
                    if !(0..BOARD_FILES as isize).contains(&nf) {
                        continue;
                    }
                    let board_index = nr as usize * BOARD_FILES + nf as usize;
                    for in_channel in 0..input_channels {
                        let weight_index = ((out_channel * input_channels + in_channel)
                            * CNN_KERNEL_AREA)
                            + kr * 3
                            + kf;
                        value += input[in_channel * BOARD_PLANES_SIZE + board_index]
                            * weights[weight_index];
                    }
                }
            }
            output[out_start + sq] = value.max(0.0);
        }
    }
}

/// `batch_size` 为**每块 GPU 每步**的样本数；与配置文件中 `batch_size` 含义一致。
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
            board: vec![0; BOARD_HISTORY_SIZE],
            move_indices,
            policy,
            value,
            side_sign: 1.0,
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

fn scalar_value_from_logits(logits: &[f32]) -> (f32, Vec<f32>) {
    let probs = softmax(logits);
    if probs.len() < VALUE_LOGITS {
        return (0.0, probs);
    }
    (probs[0] - probs[2], probs)
}

pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next(&mut self) -> u64 {
        self.state = splitmix64(self.state);
        self.state
    }

    pub fn unit_f32(&mut self) -> f32 {
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
    (rank == 0 && file == 3)
        || (rank == 0 && file == 5)
        || (rank == 1 && file == 4)
        || (rank == 2 && file == 3)
        || (rank == 2 && file == 5)
        || (rank == 7 && file == 3)
        || (rank == 7 && file == 5)
        || (rank == 8 && file == 4)
        || (rank == 9 && file == 3)
        || (rank == 9 && file == 5)
}

const fn is_elephant_pos(rank: usize, file: usize) -> bool {
    (rank == 0 && file == 2)
        || (rank == 0 && file == 6)
        || (rank == 2 && file == 0)
        || (rank == 2 && file == 4)
        || (rank == 2 && file == 8)
        || (rank == 4 && file == 2)
        || (rank == 4 && file == 6)
        || (rank == 5 && file == 2)
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
    if df == 1
        && dr == 1
        && is_advisor_pos(from_rank, from_file)
        && is_advisor_pos(to_rank, to_file)
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
fn replay_pool_test_fixture() -> AzExperiencePool {
    let sample = AzTrainingSample {
        features: vec![1, 2, 3],
        board: vec![0; BOARD_HISTORY_SIZE],
        move_indices: vec![0, 1],
        policy: vec![0.6, 0.4],
        value: 0.1,
        side_sign: 1.0,
    };
    let mut pool = AzExperiencePool::new(100);
    pool.add_games(vec![vec![sample.clone()], vec![sample.clone(), sample]]);
    pool
}

#[cfg(test)]
mod tests {
    use super::play::{assign_td_lambda_value_targets, assign_terminal_value_targets};
    use super::*;

    #[test]
    fn dense_move_space_matches_enumeration() {
        let map = move_map();
        assert_eq!(DENSE_MOVE_SPACE, 2086);
        for i in 0..DENSE_MOVE_SPACE {
            let sparse = map.dense_to_sparse[i] as usize;
            assert_eq!(map.sparse_to_dense[sparse], i as u16);
        }
    }

    #[test]
    fn board_history_planes_include_rewound_previous_position() {
        let mut position = Position::startpos();
        let mv = position.legal_moves()[0];
        let moved_piece = position.piece_at(mv.from as usize).unwrap();
        let history = vec![HistoryMove {
            piece: moved_piece,
            mv,
        }];
        position.make_move(mv);

        let mut board = Vec::new();
        extract_board_planes(&position, &history, &mut board);
        let piece_plane = (absolute_piece_plane(moved_piece.color, moved_piece.kind) + 1) as u8;

        assert_eq!(board.len(), BOARD_HISTORY_SIZE);
        assert_eq!(board[mv.to as usize], piece_plane);
        assert_eq!(board[BOARD_PLANES_SIZE + mv.from as usize], piece_plane);
        assert_eq!(board[BOARD_PLANES_SIZE + mv.to as usize], 0);
    }

    #[test]
    fn terminal_value_targets_match_outcome_for_side_to_move() {
        let mut samples = vec![
            AzTrainingSample {
                features: Vec::new(),
                board: vec![0; BOARD_HISTORY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 0.0,
                side_sign: 1.0,
            },
            AzTrainingSample {
                features: Vec::new(),
                board: vec![0; BOARD_HISTORY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 0.0,
                side_sign: -1.0,
            },
        ];

        assign_terminal_value_targets(&mut samples, 1.0);

        assert!((samples[0].value - 1.0).abs() < 1e-6);
        assert!((samples[1].value + 1.0).abs() < 1e-6);
    }

    #[test]
    fn td_lambda_value_targets_mix_future_bootstrap_values() {
        let mut samples = vec![
            AzTrainingSample {
                features: Vec::new(),
                board: vec![0; BOARD_HISTORY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 0.2,
                side_sign: 1.0,
            },
            AzTrainingSample {
                features: Vec::new(),
                board: vec![0; BOARD_HISTORY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: -0.4,
                side_sign: -1.0,
            },
            AzTrainingSample {
                features: Vec::new(),
                board: vec![0; BOARD_HISTORY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 0.6,
                side_sign: 1.0,
            },
        ];

        assign_td_lambda_value_targets(&mut samples, 1.0, 0.5);

        assert!((samples[2].value - 1.0).abs() < 1e-6);
        assert!((samples[1].value + 0.8).abs() < 1e-6);
        assert!((samples[0].value - 0.6).abs() < 1e-6);
    }

    #[test]
    fn arena_report_elo_tracks_score_rate_direction() {
        let stronger = AzArenaReport {
            wins: 6,
            losses: 3,
            draws: 1,
            ..AzArenaReport::default()
        };
        let weaker = AzArenaReport {
            wins: 3,
            losses: 6,
            draws: 1,
            ..AzArenaReport::default()
        };

        assert!(stronger.score_rate() > 0.5);
        assert!(stronger.elo() > 0.0);
        assert!(weaker.score_rate() < 0.5);
        assert!(weaker.elo() < 0.0);
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
                board: vec![0; BOARD_HISTORY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 1.0,
                side_sign: 1.0,
            },
            AzTrainingSample {
                features: vec![1],
                board: vec![0; BOARD_HISTORY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: -1.0,
                side_sign: 1.0,
            },
            AzTrainingSample {
                features: vec![2],
                board: vec![0; BOARD_HISTORY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 0.75,
                side_sign: 1.0,
            },
            AzTrainingSample {
                features: vec![3],
                board: vec![0; BOARD_HISTORY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: -0.75,
                side_sign: 1.0,
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
                board: vec![0; BOARD_HISTORY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 1.0,
                side_sign: 1.0,
            },
            AzTrainingSample {
                features: vec![1, 5, 9],
                board: vec![0; BOARD_HISTORY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: -1.0,
                side_sign: 1.0,
            },
            AzTrainingSample {
                features: vec![2, 6, 10],
                board: vec![0; BOARD_HISTORY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 0.5,
                side_sign: 1.0,
            },
            AzTrainingSample {
                features: vec![3, 7, 11],
                board: vec![0; BOARD_HISTORY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: -0.5,
                side_sign: 1.0,
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
        let repeated_stats = train_samples(&mut repeated, &samples, 5, 0.003, 4, &mut rng_repeated);

        assert!((single_stats.loss - repeated_stats.loss).abs() < 1e-5);
        assert!((single_stats.value_loss - repeated_stats.value_loss).abs() < 1e-5);
        assert!((single_stats.value_pred_sum - repeated_stats.value_pred_sum).abs() < 1e-4);
        assert!((single_stats.value_target_sum - repeated_stats.value_target_sum).abs() < 1e-6);
        assert!(
            single
                .value_logits_bias
                .iter()
                .zip(&repeated.value_logits_bias)
                .all(|(left, right)| (*left - *right).abs() < 1e-5)
        );
    }

    #[test]
    fn aznnue_binary_roundtrip_matches_weights() {
        let model = AzNnue::random_with_depth(16, 2, 42);
        let path = std::env::temp_dir().join("chineseai_test_aznnue_roundtrip.nnue");
        let _ = fs::remove_file(&path);
        model.save(&path).unwrap();
        let loaded = AzNnue::load(&path).unwrap();
        let _ = fs::remove_file(&path);
        assert_eq!(model.hidden_size, loaded.hidden_size);
        assert_eq!(model.trunk_depth, loaded.trunk_depth);
        assert_eq!(model.input_hidden, loaded.input_hidden);
        assert_eq!(model.policy_move_bias, loaded.policy_move_bias);
    }

    #[test]
    fn replay_pool_lz4_snapshot_roundtrip() {
        let path = std::env::temp_dir().join("chineseai_test_replay_roundtrip.replay.lz4");
        let _ = fs::remove_file(&path);
        let pool = super::replay_pool_test_fixture();
        pool.save_snapshot_lz4(&path).unwrap();
        let loaded = AzExperiencePool::load_snapshot_lz4(&path, 100).unwrap();
        let _ = fs::remove_file(&path);
        assert_eq!(loaded.game_count(), pool.game_count());
        assert_eq!(loaded.sample_count(), pool.sample_count());
    }
}
