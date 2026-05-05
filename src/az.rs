use std::fs;
use std::io::{self, BufWriter, Cursor, Read, Write};
use std::path::Path;

mod alphazero;
mod env;
mod mctx;
mod play;
mod replay;
mod train;
mod train_gpu;

#[cfg(test)]
use crate::nnue::canonical_square;
#[cfg(test)]
use crate::nnue::extract_sparse_features_v4_canonical;
use crate::nnue::{
    HistoryMove, V4_INPUT_SIZE, canonical_move, extract_sparse_features_v4_canonical_with_rules,
};
use crate::xiangqi::{BOARD_FILES, BOARD_SIZE, Move, Position, RuleHistoryEntry};
#[cfg(test)]
use crate::xiangqi::{Color, PieceKind};

pub use alphazero::{
    AzCandidate, AzSearchAlgorithm, AzSearchLimits, AzSearchResult, alphazero_search,
    alphazero_search_env, alphazero_search_with_history_and_rules,
};
pub use env::{AzEnv, AzGameEndReason, AzRuleSet};
pub use mctx::AzGumbelConfig;
pub use play::{
    AzArenaConfig, AzArenaReport, AzSelfplayData, AzTerminalStats, generate_selfplay_data,
    play_arena_games_from_positions,
};
pub use replay::AzExperiencePool;
pub use train::{global_training_step_sample_count, train_samples, train_samples_weighted};

pub const AZNNUE_BINARY_MAGIC: &[u8] = b"AZB1";
const AZNNUE_BINARY_VERSION: u32 = 26;
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
pub(super) const VALUE_BRANCH_SIZE: usize = 128;
pub(super) const VALUE_BRANCH_DEPTH: usize = 2;
const VALUE_HIDDEN_SIZE: usize = 256;
const VALUE_LOGITS: usize = 3;
pub(super) const AUX_MATERIAL_SIZE: usize = 14;
pub(super) const AUX_OCCUPANCY_SIZE: usize = BOARD_SIZE;
pub(super) const AUX_MATERIAL_WEIGHT: f32 = 0.02;
pub(super) const AUX_OCCUPANCY_WEIGHT: f32 = 0.01;
#[cfg(test)]
pub(super) const BOARD_PLANES_SIZE: usize = BOARD_SIZE;
#[cfg(test)]
pub(super) const BOARD_HISTORY_FRAMES: usize = crate::nnue::HISTORY_PLIES + 1;
#[cfg(test)]
pub(super) const BOARD_HISTORY_SIZE: usize = BOARD_HISTORY_FRAMES * BOARD_PLANES_SIZE;
pub(super) const POLICY_CONDITION_SIZE: usize = 32;
const VALUE_SCALE_CP: f32 = 1000.0;
const RESIDUAL_TRUNK_SCALE: f32 = 0.5;
pub(super) struct AzEvalScratch {
    hidden: Vec<f32>,
    value_hidden: Vec<f32>,
    value_next: Vec<f32>,
    value_intermediate: Vec<f32>,
    value_logits: Vec<f32>,
    policy_condition: Vec<f32>,
    logits: Vec<f32>,
    priors: Vec<f32>,
}

impl AzEvalScratch {
    pub(super) fn new(hidden_size: usize) -> Self {
        Self {
            hidden: vec![0.0; hidden_size],
            value_hidden: vec![0.0; VALUE_BRANCH_SIZE],
            value_next: vec![0.0; VALUE_BRANCH_SIZE],
            value_intermediate: vec![0.0; VALUE_HIDDEN_SIZE],
            value_logits: vec![0.0; VALUE_LOGITS],
            policy_condition: vec![0.0; POLICY_CONDITION_SIZE],
            logits: Vec::with_capacity(192),
            priors: Vec::with_capacity(192),
        }
    }
}

#[derive(Debug)]
pub struct AzNnue {
    pub hidden_size: usize,
    pub input_hidden: Vec<f32>,
    pub hidden_bias: Vec<f32>,
    pub value_trunk_weights: Vec<f32>,
    pub value_trunk_biases: Vec<f32>,
    pub value_shared_hidden: Vec<f32>,
    pub value_shared_hidden_bias: Vec<f32>,
    pub value_intermediate_hidden: Vec<f32>,
    pub value_intermediate_bias: Vec<f32>,
    pub value_logits_weights: Vec<f32>,
    pub value_logits_bias: Vec<f32>,
    pub policy_move_hidden: Vec<f32>,
    pub policy_move_bias: Vec<f32>,
    pub policy_feature_hidden: Vec<f32>,
    pub policy_feature_bias: Vec<f32>,
    pub aux_material_weights: Vec<f32>,
    pub aux_material_bias: Vec<f32>,
    pub aux_occupancy_weights: Vec<f32>,
    pub aux_occupancy_bias: Vec<f32>,
    gpu_trainer: Option<Box<train_gpu::GpuTrainer>>,
}

// Architecture notes:
// - v26 is intentionally incompatible with older AZB1 files. It keeps the
//   shared sparse NNUE trunk, adds ZeroForge-style rule/material/region/frame
//   sparse features, and trains lightweight material/occupancy auxiliary heads.
// - Do not add a from/to-square factorized policy head on top of this absolute
//   board representation. A previous v14/v15 experiment mixed absolute board
//   features with partially relative action-square sharing and immediately
//   produced a severe red/black bias in self-play. Factorized policy should
//   wait until the whole model has a consistent per-square or canonical view.
// - V4 sparse features now include row/column buckets and nearest-piece line
//   relations. This is the cheap part borrowed from row/column attention: it
//   helps long rook/cannon/general files without paying CNN cost.
// - v18 adds a static geometry-conditioned policy residual. It shares policy
//   knowledge across similar move shapes without resurrecting the bad v14/v15
//   from/to factorization or changing the absolute board view.
// - v20 removes the old policy residual trunk completely. It was not a useful
//   runtime knob: width, cheap board summaries, and move-conditioned policy
//   sharing were more valuable than hidden->hidden depth for this CPU MCTS net.
// - v21 tried a value-only relation/move encoder. It helped loss in places but
//   made the value experiment muddy: hand summaries can become shortcuts and
//   hide whether the board model itself understands positions.
// - v22 switches the net to side-to-move canonical inputs and canonical policy
//   actions: the mover is always represented as Red, and Black-to-move boards
//   are rotated 180 degrees with colors swapped. The old side-to-move board
//   channel is removed; do not feed canonical boards with absolute move labels,
//   because that recreates the red/black leakage bug.

impl Clone for AzNnue {
    fn clone(&self) -> Self {
        Self {
            hidden_size: self.hidden_size,
            input_hidden: self.input_hidden.clone(),
            hidden_bias: self.hidden_bias.clone(),
            value_trunk_weights: self.value_trunk_weights.clone(),
            value_trunk_biases: self.value_trunk_biases.clone(),
            value_shared_hidden: self.value_shared_hidden.clone(),
            value_shared_hidden_bias: self.value_shared_hidden_bias.clone(),
            value_intermediate_hidden: self.value_intermediate_hidden.clone(),
            value_intermediate_bias: self.value_intermediate_bias.clone(),
            value_logits_weights: self.value_logits_weights.clone(),
            value_logits_bias: self.value_logits_bias.clone(),
            policy_move_hidden: self.policy_move_hidden.clone(),
            policy_move_bias: self.policy_move_bias.clone(),
            policy_feature_hidden: self.policy_feature_hidden.clone(),
            policy_feature_bias: self.policy_feature_bias.clone(),
            aux_material_weights: self.aux_material_weights.clone(),
            aux_material_bias: self.aux_material_bias.clone(),
            aux_occupancy_weights: self.aux_occupancy_weights.clone(),
            aux_occupancy_bias: self.aux_occupancy_bias.clone(),
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
    pub mirror_probability: f32,
    pub selfplay_repetition_as_loss: bool,
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
    pub aux_material_loss: f32,
    pub aux_occupancy_loss: f32,
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
    pub terminal_no_attacking_material: usize,
    pub terminal_halfmove120: usize,
    pub terminal_repetition: usize,
    pub terminal_repetition_quiet: usize,
    pub terminal_repetition_current_check: usize,
    pub terminal_repetition_current_chase: usize,
    pub terminal_repetition_rule_pressure: usize,
    pub terminal_mutual_long_check: usize,
    pub terminal_mutual_long_chase: usize,
    pub terminal_rule_win_red: usize,
    pub terminal_rule_win_black: usize,
    pub terminal_max_plies: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AzTrainBenchmark {
    pub loss: f32,
    pub value_loss: f32,
    pub policy_ce: f32,
    pub aux_material_loss: f32,
    pub aux_occupancy_loss: f32,
}

#[derive(Clone, Debug)]
pub struct AzTrainingSample {
    pub features: Vec<usize>,
    pub board: Vec<u8>,
    pub aux_material: Vec<f32>,
    pub aux_occupancy: Vec<f32>,
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
    pub aux_material_loss: f32,
    pub aux_occupancy_loss: f32,
    pub value_pred_sum: f32,
    pub value_pred_sq_sum: f32,
    pub value_target_sum: f32,
    pub value_target_sq_sum: f32,
    pub value_error_sq_sum: f32,
    pub samples: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct AzTrainLossWeights {
    pub value: f32,
    pub policy: f32,
    pub train_shared: bool,
    pub train_value_head: bool,
    pub train_policy_head: bool,
    pub train_aux_heads: bool,
}

impl Default for AzTrainLossWeights {
    fn default() -> Self {
        Self {
            value: 0.25,
            policy: 1.0,
            train_shared: true,
            train_value_head: true,
            train_policy_head: true,
            train_aux_heads: true,
        }
    }
}

impl AzTrainStats {
    fn add_assign(&mut self, other: &Self) {
        self.loss += other.loss;
        self.value_loss += other.value_loss;
        self.policy_ce += other.policy_ce;
        self.aux_material_loss += other.aux_material_loss;
        self.aux_occupancy_loss += other.aux_occupancy_loss;
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
        let mut rng = SplitMix64::new(seed);
        let input_hidden: Vec<f32> = (0..V4_INPUT_SIZE * hidden_size)
            .map(|_| rng.weight(0.015))
            .collect();
        let hidden_bias = vec![0.0; hidden_size];
        let value_trunk_weights: Vec<f32> =
            (0..VALUE_BRANCH_DEPTH * VALUE_BRANCH_SIZE * VALUE_BRANCH_SIZE)
                .map(|_| rng.weight((2.0 / VALUE_BRANCH_SIZE as f32).sqrt()))
                .collect();
        let value_trunk_biases = vec![0.0; VALUE_BRANCH_DEPTH * VALUE_BRANCH_SIZE];
        let value_shared_hidden: Vec<f32> = (0..hidden_size * VALUE_BRANCH_SIZE)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt()))
            .collect();
        let value_shared_hidden_bias = vec![0.0; VALUE_BRANCH_SIZE];
        let value_intermediate_hidden = (0..VALUE_HIDDEN_SIZE * VALUE_BRANCH_SIZE)
            .map(|_| rng.weight((2.0 / VALUE_BRANCH_SIZE as f32).sqrt()))
            .collect();
        let value_intermediate_bias = vec![0.0; VALUE_HIDDEN_SIZE];
        let value_logits_weights = (0..VALUE_LOGITS * VALUE_HIDDEN_SIZE)
            .map(|_| rng.weight((2.0 / VALUE_HIDDEN_SIZE as f32).sqrt()))
            .collect();
        let value_logits_bias = vec![-0.25, 0.5, -0.25];
        let policy_move_hidden = (0..DENSE_MOVE_SPACE * hidden_size)
            .map(|_| rng.weight(0.01))
            .collect();
        let policy_move_bias = vec![0.0; DENSE_MOVE_SPACE];
        let policy_feature_hidden = (0..POLICY_CONDITION_SIZE * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt() * 0.5))
            .collect();
        let policy_feature_bias = vec![0.0; POLICY_CONDITION_SIZE];
        let aux_material_weights = (0..AUX_MATERIAL_SIZE * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt() * 0.25))
            .collect();
        let aux_material_bias = vec![0.0; AUX_MATERIAL_SIZE];
        let aux_occupancy_weights = (0..AUX_OCCUPANCY_SIZE * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt() * 0.25))
            .collect();
        let aux_occupancy_bias = vec![0.0; AUX_OCCUPANCY_SIZE];
        Self {
            hidden_size,
            input_hidden,
            hidden_bias,
            value_trunk_weights,
            value_trunk_biases,
            value_shared_hidden,
            value_shared_hidden_bias,
            value_intermediate_hidden,
            value_intermediate_bias,
            value_logits_weights,
            value_logits_bias,
            policy_move_hidden,
            policy_move_bias,
            policy_feature_hidden,
            policy_feature_bias,
            aux_material_weights,
            aux_material_bias,
            aux_occupancy_weights,
            aux_occupancy_bias,
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
        writer.write_all(&0u32.to_le_bytes())?;
        writer.write_all(&0u32.to_le_bytes())?;
        write_f32_slice_le(&mut writer, &self.input_hidden)?;
        write_f32_slice_le(&mut writer, &self.hidden_bias)?;
        write_f32_slice_le(&mut writer, &self.value_trunk_weights)?;
        write_f32_slice_le(&mut writer, &self.value_trunk_biases)?;
        write_f32_slice_le(&mut writer, &self.value_shared_hidden)?;
        write_f32_slice_le(&mut writer, &self.value_shared_hidden_bias)?;
        write_f32_slice_le(&mut writer, &self.value_intermediate_hidden)?;
        write_f32_slice_le(&mut writer, &self.value_intermediate_bias)?;
        write_f32_slice_le(&mut writer, &self.value_logits_weights)?;
        write_f32_slice_le(&mut writer, &self.value_logits_bias)?;
        write_f32_slice_le(&mut writer, &self.policy_move_hidden)?;
        write_f32_slice_le(&mut writer, &self.policy_move_bias)?;
        write_f32_slice_le(&mut writer, &self.policy_feature_hidden)?;
        write_f32_slice_le(&mut writer, &self.policy_feature_bias)?;
        write_f32_slice_le(&mut writer, &self.aux_material_weights)?;
        write_f32_slice_le(&mut writer, &self.aux_material_bias)?;
        write_f32_slice_le(&mut writer, &self.aux_occupancy_weights)?;
        write_f32_slice_le(&mut writer, &self.aux_occupancy_bias)?;
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
        let _arch_reserved = read_u32_le(&mut reader)?;
        let _reserved = read_u32_le(&mut reader)?;
        if input_size != V4_INPUT_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "binary input_size does not match this build (V4_INPUT_SIZE)",
            ));
        }
        let input_hidden_len = V4_INPUT_SIZE * hidden_size;
        let hidden_bias_len = hidden_size;
        let value_trunk_weights_len = VALUE_BRANCH_DEPTH * VALUE_BRANCH_SIZE * VALUE_BRANCH_SIZE;
        let value_trunk_biases_len = VALUE_BRANCH_DEPTH * VALUE_BRANCH_SIZE;
        let value_shared_hidden_len = hidden_size * VALUE_BRANCH_SIZE;
        let value_shared_hidden_bias_len = VALUE_BRANCH_SIZE;
        let vih_len = VALUE_HIDDEN_SIZE * VALUE_BRANCH_SIZE;
        let vib_len = VALUE_HIDDEN_SIZE;
        let vlw_len = VALUE_LOGITS * VALUE_HIDDEN_SIZE;
        let vlb_len = VALUE_LOGITS;
        let pmh_len = DENSE_MOVE_SPACE * hidden_size;
        let pmb_len = DENSE_MOVE_SPACE;
        let pfh_len = POLICY_CONDITION_SIZE * hidden_size;
        let pfb_len = POLICY_CONDITION_SIZE;
        let aux_material_weights_len = AUX_MATERIAL_SIZE * hidden_size;
        let aux_material_bias_len = AUX_MATERIAL_SIZE;
        let aux_occupancy_weights_len = AUX_OCCUPANCY_SIZE * hidden_size;
        let aux_occupancy_bias_len = AUX_OCCUPANCY_SIZE;
        let float_count = input_hidden_len
            + hidden_bias_len
            + value_trunk_weights_len
            + value_trunk_biases_len
            + value_shared_hidden_len
            + value_shared_hidden_bias_len
            + vih_len
            + vib_len
            + vlw_len
            + vlb_len
            + pmh_len
            + pmb_len
            + pfh_len
            + pfb_len
            + aux_material_weights_len
            + aux_material_bias_len
            + aux_occupancy_weights_len
            + aux_occupancy_bias_len;
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
        let value_trunk_weights = read_f32_vec_le(&mut reader, value_trunk_weights_len)?;
        let value_trunk_biases = read_f32_vec_le(&mut reader, value_trunk_biases_len)?;
        let value_shared_hidden = read_f32_vec_le(&mut reader, value_shared_hidden_len)?;
        let value_shared_hidden_bias = read_f32_vec_le(&mut reader, value_shared_hidden_bias_len)?;
        let value_intermediate_hidden = read_f32_vec_le(&mut reader, vih_len)?;
        let value_intermediate_bias = read_f32_vec_le(&mut reader, vib_len)?;
        let value_logits_weights = read_f32_vec_le(&mut reader, vlw_len)?;
        let value_logits_bias = read_f32_vec_le(&mut reader, vlb_len)?;
        let policy_move_hidden = read_f32_vec_le(&mut reader, pmh_len)?;
        let policy_move_bias = read_f32_vec_le(&mut reader, pmb_len)?;
        let policy_feature_hidden = read_f32_vec_le(&mut reader, pfh_len)?;
        let policy_feature_bias = read_f32_vec_le(&mut reader, pfb_len)?;
        let aux_material_weights = read_f32_vec_le(&mut reader, aux_material_weights_len)?;
        let aux_material_bias = read_f32_vec_le(&mut reader, aux_material_bias_len)?;
        let aux_occupancy_weights = read_f32_vec_le(&mut reader, aux_occupancy_weights_len)?;
        let aux_occupancy_bias = read_f32_vec_le(&mut reader, aux_occupancy_bias_len)?;
        let model = Self {
            hidden_size,
            input_hidden,
            hidden_bias,
            value_trunk_weights,
            value_trunk_biases,
            value_shared_hidden,
            value_shared_hidden_bias,
            value_intermediate_hidden,
            value_intermediate_bias,
            value_logits_weights,
            value_logits_bias,
            policy_move_hidden,
            policy_move_bias,
            policy_feature_hidden,
            policy_feature_bias,
            aux_material_weights,
            aux_material_bias,
            aux_occupancy_weights,
            aux_occupancy_bias,
            gpu_trainer: None,
        };
        model.validate()?;
        Ok(model)
    }

    pub(super) fn evaluate_with_scratch(
        &self,
        position: &Position,
        history: &[HistoryMove],
        rule_history: &[RuleHistoryEntry],
        moves: &[Move],
        scratch: &mut AzEvalScratch,
    ) -> f32 {
        let side = position.side_to_move();
        let features =
            extract_sparse_features_v4_canonical_with_rules(position, history, Some(rule_history));
        self.input_embedding_into(&features, &mut scratch.hidden);
        self.value_shared_hidden_into(&scratch.hidden, &mut scratch.value_hidden);
        self.forward_value_trunk_into(&mut scratch.value_hidden, &mut scratch.value_next);
        let value = self.value_from_hidden_scratch(scratch);
        self.policy_condition_into(&scratch.hidden, &mut scratch.policy_condition);
        scratch.logits.resize(moves.len(), 0.0);
        for (index, mv) in moves.iter().enumerate() {
            scratch.logits[index] = self.policy_logit_from_hidden_index(
                &scratch.hidden,
                &scratch.policy_condition,
                dense_move_index(canonical_move(side, *mv)),
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

    fn value_shared_hidden_into(&self, shared_hidden: &[f32], hidden: &mut Vec<f32>) {
        hidden.resize(VALUE_BRANCH_SIZE, 0.0);
        hidden.copy_from_slice(&self.value_shared_hidden_bias);
        for (out, hidden_value) in hidden.iter_mut().enumerate().take(VALUE_BRANCH_SIZE) {
            let row =
                &self.value_shared_hidden[out * self.hidden_size..(out + 1) * self.hidden_size];
            for (shared_value, weight) in shared_hidden.iter().zip(row) {
                *hidden_value += shared_value * weight;
            }
            *hidden_value = (*hidden_value).max(0.0);
        }
    }

    fn forward_value_trunk_into(&self, hidden: &mut Vec<f32>, next: &mut Vec<f32>) {
        next.resize(VALUE_BRANCH_SIZE, 0.0);
        for layer in 0..VALUE_BRANCH_DEPTH {
            let weight_offset = layer * VALUE_BRANCH_SIZE * VALUE_BRANCH_SIZE;
            let bias_offset = layer * VALUE_BRANCH_SIZE;
            for out in 0..VALUE_BRANCH_SIZE {
                let mut value = self.value_trunk_biases[bias_offset + out];
                let row = &self.value_trunk_weights[weight_offset + out * VALUE_BRANCH_SIZE
                    ..weight_offset + (out + 1) * VALUE_BRANCH_SIZE];
                for idx in 0..VALUE_BRANCH_SIZE {
                    value += row[idx] * hidden[idx];
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
        for (j, value) in scratch
            .value_intermediate
            .iter_mut()
            .enumerate()
            .take(VALUE_HIDDEN_SIZE)
        {
            let h_row =
                &self.value_intermediate_hidden[j * VALUE_BRANCH_SIZE..(j + 1) * VALUE_BRANCH_SIZE];
            for (hidden_value, weight) in scratch.value_hidden.iter().zip(h_row) {
                *value += hidden_value * weight;
            }
            *value = (*value).max(0.0);
        }
        scratch
            .value_logits
            .copy_from_slice(&self.value_logits_bias);
        for out in 0..VALUE_LOGITS {
            let row =
                &self.value_logits_weights[out * VALUE_HIDDEN_SIZE..(out + 1) * VALUE_HIDDEN_SIZE];
            for (intermediate, weight) in scratch.value_intermediate.iter().zip(row) {
                scratch.value_logits[out] += intermediate * weight;
            }
        }
        scalar_value_from_logits(&scratch.value_logits).0
    }

    fn policy_condition_into(&self, hidden: &[f32], out: &mut Vec<f32>) {
        out.resize(POLICY_CONDITION_SIZE, 0.0);
        out.copy_from_slice(&self.policy_feature_bias);
        for (feature, value) in out.iter_mut().enumerate().take(POLICY_CONDITION_SIZE) {
            let hidden_row = &self.policy_feature_hidden
                [feature * self.hidden_size..(feature + 1) * self.hidden_size];
            *value += dot_product(hidden, hidden_row);
        }
    }

    fn validate(&self) -> io::Result<()> {
        if self.input_hidden.len() != V4_INPUT_SIZE * self.hidden_size
            || self.hidden_bias.len() != self.hidden_size
            || self.value_trunk_weights.len()
                != VALUE_BRANCH_DEPTH * VALUE_BRANCH_SIZE * VALUE_BRANCH_SIZE
            || self.value_trunk_biases.len() != VALUE_BRANCH_DEPTH * VALUE_BRANCH_SIZE
            || self.value_shared_hidden.len() != self.hidden_size * VALUE_BRANCH_SIZE
            || self.value_shared_hidden_bias.len() != VALUE_BRANCH_SIZE
            || self.value_intermediate_hidden.len() != VALUE_HIDDEN_SIZE * VALUE_BRANCH_SIZE
            || self.value_intermediate_bias.len() != VALUE_HIDDEN_SIZE
            || self.value_logits_weights.len() != VALUE_LOGITS * VALUE_HIDDEN_SIZE
            || self.value_logits_bias.len() != VALUE_LOGITS
            || self.policy_move_hidden.len() != DENSE_MOVE_SPACE * self.hidden_size
            || self.policy_move_bias.len() != DENSE_MOVE_SPACE
            || self.policy_feature_hidden.len() != POLICY_CONDITION_SIZE * self.hidden_size
            || self.policy_feature_bias.len() != POLICY_CONDITION_SIZE
            || self.aux_material_weights.len() != AUX_MATERIAL_SIZE * self.hidden_size
            || self.aux_material_bias.len() != AUX_MATERIAL_SIZE
            || self.aux_occupancy_weights.len() != AUX_OCCUPANCY_SIZE * self.hidden_size
            || self.aux_occupancy_bias.len() != AUX_OCCUPANCY_SIZE
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
        policy_condition: &[f32],
        move_index: usize,
    ) -> f32 {
        let hidden_offset = move_index * self.hidden_size;
        let hidden_row = &self.policy_move_hidden[hidden_offset..hidden_offset + self.hidden_size];
        let feature_offset = move_index * POLICY_CONDITION_SIZE;
        let move_features =
            &policy_move_features()[feature_offset..feature_offset + POLICY_CONDITION_SIZE];
        self.policy_move_bias[move_index]
            + dot_product(hidden, hidden_row)
            + dot_product(policy_condition, move_features)
    }
}

#[cfg(test)]
pub(super) fn extract_board_planes(
    position: &Position,
    history: &[HistoryMove],
    board: &mut Vec<u8>,
) {
    let side = position.side_to_move();
    board.resize(BOARD_HISTORY_SIZE, 0);
    board.fill(0);
    let mut frame = [0u8; BOARD_PLANES_SIZE];
    extract_position_piece_planes(position, &mut frame);
    board[..BOARD_PLANES_SIZE].copy_from_slice(&frame);

    let mut rewound = frame;
    for (history_index, entry) in history
        .iter()
        .rev()
        .take(crate::nnue::HISTORY_PLIES)
        .enumerate()
    {
        let piece_plane =
            (canonical_piece_plane(side, entry.piece.color, entry.piece.kind) + 1) as u8;
        rewound[canonical_square(side, entry.mv.to as usize)] = entry.captured.map_or(0, |piece| {
            (canonical_piece_plane(side, piece.color, piece.kind) + 1) as u8
        });
        rewound[canonical_square(side, entry.mv.from as usize)] = piece_plane;
        let start = (history_index + 1) * BOARD_PLANES_SIZE;
        board[start..start + BOARD_PLANES_SIZE].copy_from_slice(&rewound);
    }
}

#[cfg(test)]
fn extract_position_piece_planes(position: &Position, board: &mut [u8; BOARD_PLANES_SIZE]) {
    board.fill(0);
    let side = position.side_to_move();
    for sq in 0..BOARD_SIZE {
        let Some(piece) = position.piece_at(sq) else {
            continue;
        };
        let plane = canonical_piece_plane(side, piece.color, piece.kind);
        board[canonical_square(side, sq)] = (plane + 1) as u8;
    }
}

#[cfg(test)]
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

#[cfg(test)]
fn canonical_piece_plane(side: Color, piece_color: Color, kind: PieceKind) -> usize {
    let canonical_color = if piece_color == side {
        Color::Red
    } else {
        Color::Black
    };
    absolute_piece_plane(canonical_color, kind)
}

/// `batch_size` is the per-device training batch size.
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
        let feature_count = 24 + (rng.next_u64() as usize % 16);
        let mut features = Vec::with_capacity(feature_count);
        for _ in 0..feature_count {
            features.push((rng.next_u64() as usize) % V4_INPUT_SIZE);
        }
        features.sort_unstable();
        features.dedup();

        let value = rng.unit_f32() * 2.0 - 1.0;
        let move_count = 12 + (rng.next_u64() as usize % 24);
        let mut move_indices = Vec::with_capacity(move_count);
        while move_indices.len() < move_count {
            let candidate = (rng.next_u64() as usize) % DENSE_MOVE_SPACE;
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
            board: Vec::new(),
            aux_material: vec![0.0; AUX_MATERIAL_SIZE],
            aux_occupancy: vec![0.0; AUX_OCCUPANCY_SIZE],
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
        aux_material_loss: stats.aux_material_loss,
        aux_occupancy_loss: stats.aux_occupancy_loss,
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
    let mut sum4 = 0.0;
    let mut sum5 = 0.0;
    let mut sum6 = 0.0;
    let mut sum7 = 0.0;
    let chunks = left.len() / 8;
    for chunk in 0..chunks {
        let index = chunk * 8;
        sum0 += left[index] * right[index];
        sum1 += left[index + 1] * right[index + 1];
        sum2 += left[index + 2] * right[index + 2];
        sum3 += left[index + 3] * right[index + 3];
        sum4 += left[index + 4] * right[index + 4];
        sum5 += left[index + 5] * right[index + 5];
        sum6 += left[index + 6] * right[index + 6];
        sum7 += left[index + 7] * right[index + 7];
    }
    let mut sum = ((sum0 + sum1) + (sum2 + sum3)) + ((sum4 + sum5) + (sum6 + sum7));
    for index in (chunks * 8)..left.len() {
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

    pub fn next_u64(&mut self) -> u64 {
        self.state = splitmix64(self.state);
        self.state
    }

    pub fn unit_f32(&mut self) -> f32 {
        let value = self.next_u64();
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
    matches!(
        (rank, file),
        (0, 3) | (0, 5) | (1, 4) | (2, 3) | (2, 5) | (7, 3) | (7, 5) | (8, 4) | (9, 3) | (9, 5)
    )
}

const fn is_palace_pos(rank: usize, file: usize) -> bool {
    file >= 3 && file <= 5 && (rank <= 2 || rank >= 7)
}

const fn is_elephant_pos(rank: usize, file: usize) -> bool {
    matches!(
        (rank, file),
        (0, 2)
            | (0, 6)
            | (2, 0)
            | (2, 4)
            | (2, 8)
            | (4, 2)
            | (4, 6)
            | (5, 2)
            | (5, 6)
            | (7, 0)
            | (7, 4)
            | (7, 8)
            | (9, 2)
            | (9, 6)
    )
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

pub(super) fn policy_move_features() -> &'static [f32] {
    use std::sync::OnceLock;
    static FEATURES: OnceLock<Vec<f32>> = OnceLock::new();
    FEATURES.get_or_init(|| {
        let mut features = Vec::with_capacity(DENSE_MOVE_SPACE * POLICY_CONDITION_SIZE);
        for &sparse in &move_map().dense_to_sparse {
            let from = sparse as usize / BOARD_SIZE;
            let to = sparse as usize % BOARD_SIZE;
            let from_file = from % BOARD_FILES;
            let from_rank = from / BOARD_FILES;
            let to_file = to % BOARD_FILES;
            let to_rank = to / BOARD_FILES;
            let df_signed = to_file as i32 - from_file as i32;
            let dr_signed = to_rank as i32 - from_rank as i32;
            let df = df_signed.unsigned_abs() as usize;
            let dr = dr_signed.unsigned_abs() as usize;
            let line = (df == 0 || dr == 0) as u8 as f32;
            let horse = ((df == 1 && dr == 2) || (df == 2 && dr == 1)) as u8 as f32;
            let advisor = (df == 1 && dr == 1) as u8 as f32;
            let elephant = (df == 2 && dr == 2) as u8 as f32;
            let distance = (df + dr) as f32;
            let from_palace = is_palace_pos(from_rank, from_file) as u8 as f32;
            let to_palace = is_palace_pos(to_rank, to_file) as u8 as f32;
            let crosses_river = ((from_rank < 5) != (to_rank < 5)) as u8 as f32;
            let forward_red = (to_rank as i32 - from_rank as i32).signum() as f32;
            let center_from = 1.0 - ((from_file as f32 - 4.0).abs() / 4.0);
            let center_to = 1.0 - ((to_file as f32 - 4.0).abs() / 4.0);
            let raw = [
                1.0,
                from_file as f32 / 8.0,
                from_rank as f32 / 9.0,
                to_file as f32 / 8.0,
                to_rank as f32 / 9.0,
                df as f32 / 8.0,
                dr as f32 / 9.0,
                df_signed as f32 / 8.0,
                dr_signed as f32 / 9.0,
                line,
                (df == 0) as u8 as f32,
                (dr == 0) as u8 as f32,
                horse,
                advisor,
                elephant,
                distance / 9.0,
                (distance <= 1.0) as u8 as f32,
                (distance <= 2.0) as u8 as f32,
                (distance >= 5.0) as u8 as f32,
                from_palace,
                to_palace,
                crosses_river,
                forward_red,
                center_from,
                center_to,
                ((from_rank <= 2) || (from_rank >= 7)) as u8 as f32,
                ((to_rank <= 2) || (to_rank >= 7)) as u8 as f32,
                ((from_file == 0) || (from_file == 8)) as u8 as f32,
                ((to_file == 0) || (to_file == 8)) as u8 as f32,
                (df == dr) as u8 as f32,
                ((df + dr) % 2 == 0) as u8 as f32,
                ((from + to) % 2 == 0) as u8 as f32,
            ];
            debug_assert_eq!(raw.len(), POLICY_CONDITION_SIZE);
            features.extend_from_slice(&raw);
        }
        features
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
        board: Vec::new(),
        aux_material: vec![0.0; AUX_MATERIAL_SIZE],
        aux_occupancy: vec![0.0; AUX_OCCUPANCY_SIZE],
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
    use super::play::assign_terminal_value_targets;
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
            captured: None,
            mv,
        }];
        position.make_move(mv);

        let mut board = Vec::new();
        extract_board_planes(&position, &history, &mut board);
        let side = position.side_to_move();
        let piece_plane =
            (canonical_piece_plane(side, moved_piece.color, moved_piece.kind) + 1) as u8;
        let from = canonical_square(side, mv.from as usize);
        let to = canonical_square(side, mv.to as usize);

        assert_eq!(board.len(), BOARD_HISTORY_SIZE);
        assert_eq!(board[to], piece_plane);
        assert_eq!(board[BOARD_PLANES_SIZE + from], piece_plane);
        assert_eq!(board[BOARD_PLANES_SIZE + to], 0);
    }

    #[test]
    fn board_history_planes_restore_captured_piece_when_rewound() {
        let mut position = Position::from_fen("4k4/9/9/9/r3c4/9/9/9/R8/4K4 w").unwrap();
        let mv = Move::new(72, 36);
        assert!(position.is_legal_move(mv));
        let moved_piece = position.piece_at(mv.from as usize).unwrap();
        let captured_piece = position.piece_at(mv.to as usize).unwrap();
        let history = vec![HistoryMove {
            piece: moved_piece,
            captured: Some(captured_piece),
            mv,
        }];
        position.make_move(mv);

        let mut board = Vec::new();
        extract_board_planes(&position, &history, &mut board);
        let side = position.side_to_move();
        let moved_plane =
            (canonical_piece_plane(side, moved_piece.color, moved_piece.kind) + 1) as u8;
        let captured_plane =
            (canonical_piece_plane(side, captured_piece.color, captured_piece.kind) + 1) as u8;
        let from = canonical_square(side, mv.from as usize);
        let to = canonical_square(side, mv.to as usize);

        assert_eq!(board[to], moved_plane);
        assert_eq!(board[BOARD_PLANES_SIZE + from], moved_plane);
        assert_eq!(board[BOARD_PLANES_SIZE + to], captured_plane);
    }

    #[test]
    fn canonical_inputs_match_for_startpos_from_either_side() {
        let red_to_move =
            Position::from_fen("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w")
                .unwrap();
        let black_to_move =
            Position::from_fen("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b")
                .unwrap();

        assert_eq!(
            extract_sparse_features_v4_canonical(&red_to_move, &[]),
            extract_sparse_features_v4_canonical(&black_to_move, &[])
        );

        let mut red_board = Vec::new();
        let mut black_board = Vec::new();
        extract_board_planes(&red_to_move, &[], &mut red_board);
        extract_board_planes(&black_to_move, &[], &mut black_board);

        assert_eq!(red_board, black_board);
    }

    #[test]
    fn terminal_value_targets_match_outcome_for_side_to_move() {
        let mut samples = vec![
            AzTrainingSample {
                features: Vec::new(),
                board: Vec::new(),
                aux_material: vec![0.0; AUX_MATERIAL_SIZE],
                aux_occupancy: vec![0.0; AUX_OCCUPANCY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 0.0,
                side_sign: 1.0,
            },
            AzTrainingSample {
                features: Vec::new(),
                board: Vec::new(),
                aux_material: vec![0.0; AUX_MATERIAL_SIZE],
                aux_occupancy: vec![0.0; AUX_OCCUPANCY_SIZE],
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
        let mut model = AzNnue::random(16, 7);
        model.hidden_bias.fill(0.1);
        model.hidden_bias.fill(0.1);
        let board_with = |sq: usize, plane: u8| {
            let mut board = vec![0; BOARD_HISTORY_SIZE];
            board[sq] = plane;
            board
        };

        let samples = vec![
            AzTrainingSample {
                features: vec![0],
                board: board_with(0, 1),
                aux_material: vec![0.0; AUX_MATERIAL_SIZE],
                aux_occupancy: vec![0.0; AUX_OCCUPANCY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 1.0,
                side_sign: 1.0,
            },
            AzTrainingSample {
                features: vec![1],
                board: board_with(10, 2),
                aux_material: vec![0.0; AUX_MATERIAL_SIZE],
                aux_occupancy: vec![0.0; AUX_OCCUPANCY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: -1.0,
                side_sign: 1.0,
            },
            AzTrainingSample {
                features: vec![2],
                board: board_with(40, 3),
                aux_material: vec![0.0; AUX_MATERIAL_SIZE],
                aux_occupancy: vec![0.0; AUX_OCCUPANCY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 0.75,
                side_sign: 1.0,
            },
            AzTrainingSample {
                features: vec![3],
                board: board_with(80, 4),
                aux_material: vec![0.0; AUX_MATERIAL_SIZE],
                aux_occupancy: vec![0.0; AUX_OCCUPANCY_SIZE],
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
                board: Vec::new(),
                aux_material: vec![0.0; AUX_MATERIAL_SIZE],
                aux_occupancy: vec![0.0; AUX_OCCUPANCY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 1.0,
                side_sign: 1.0,
            },
            AzTrainingSample {
                features: vec![1, 5, 9],
                board: Vec::new(),
                aux_material: vec![0.0; AUX_MATERIAL_SIZE],
                aux_occupancy: vec![0.0; AUX_OCCUPANCY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: -1.0,
                side_sign: 1.0,
            },
            AzTrainingSample {
                features: vec![2, 6, 10],
                board: Vec::new(),
                aux_material: vec![0.0; AUX_MATERIAL_SIZE],
                aux_occupancy: vec![0.0; AUX_OCCUPANCY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 0.5,
                side_sign: 1.0,
            },
            AzTrainingSample {
                features: vec![3, 7, 11],
                board: Vec::new(),
                aux_material: vec![0.0; AUX_MATERIAL_SIZE],
                aux_occupancy: vec![0.0; AUX_OCCUPANCY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: -0.5,
                side_sign: 1.0,
            },
        ];
        let mut single = AzNnue::random(16, 23);
        single.hidden_bias.fill(0.1);
        single.hidden_bias.fill(0.1);
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
        let model = AzNnue::random(16, 42);
        let path = std::env::temp_dir().join("chineseai_test_aznnue_roundtrip.nnue");
        let _ = fs::remove_file(&path);
        model.save(&path).unwrap();
        let loaded = AzNnue::load(&path).unwrap();
        let _ = fs::remove_file(&path);
        assert_eq!(model.hidden_size, loaded.hidden_size);
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
