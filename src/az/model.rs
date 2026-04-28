use std::io;

use crate::board_transform::{HISTORY_PLIES, HistoryMove, canonical_move, canonical_square};
use crate::xiangqi::{BOARD_FILES, BOARD_RANKS, BOARD_SIZE, Color, Move, PieceKind, Position};

use super::model_config::AzModelConfig;
use super::model_ops::{
    conv_relu_board_layer_sparse_3x3, conv1x1_linear_layer_dense_generic, dot_product,
    pool_cnn_features, residual_mobile_block_generic, residual_value_relation_block_generic,
};
use super::train_gpu;

pub const AZ_MODEL_BINARY_MAGIC: &[u8] = b"AZM1";
pub(super) const AZ_MODEL_BINARY_VERSION: u32 = 12;
pub(super) const AZ_MODEL_BINARY_HEADER_LEN: usize = 40;

const SPARSE_MOVE_SPACE: usize = BOARD_SIZE * BOARD_SIZE;
pub(super) const DENSE_MOVE_SPACE: usize = compute_dense_move_count();
pub(super) const VALUE_HIDDEN_SIZE: usize = 256;
pub(super) const VALUE_LOGITS: usize = 3;
pub(super) const BOARD_PLANES_SIZE: usize = BOARD_SIZE;
pub(super) const BOARD_HISTORY_FRAMES: usize = HISTORY_PLIES + 1;
pub(super) const BOARD_HISTORY_SIZE: usize = BOARD_HISTORY_FRAMES * BOARD_PLANES_SIZE;
pub(super) const PIECE_BOARD_CHANNELS: usize = 14;
pub(super) const BOARD_CHANNELS: usize = PIECE_BOARD_CHANNELS * BOARD_HISTORY_FRAMES;
pub(super) const CNN_CHANNELS: usize = 32;
pub(super) const RESIDUAL_BLOCKS: usize = 3;
pub(super) const VALUE_HEAD_CHANNELS: usize = 8;
pub(super) const VALUE_RELATION_LAYERS: usize = 4;
pub(super) const VALUE_RELATION_FFN_MULT: usize = 2;
pub(super) const BOARD_INPUT_KERNEL_AREA: usize = 9;
pub(super) const CNN_KERNEL_AREA: usize = 9;
pub(super) const CNN_POOL_BLOCKS: usize = 4;
pub(super) const VALUE_HEAD_LEAK: f32 = 0.05;
pub(super) const POLICY_CONDITION_SIZE: usize = 32;
pub(super) const VALUE_SCALE_CP: f32 = 1000.0;

pub(super) fn cnn_pooled_size(channels: usize) -> usize {
    channels * CNN_POOL_BLOCKS
}

pub(super) fn mobile_block_weight_size(channels: usize) -> usize {
    channels * (CNN_KERNEL_AREA + BOARD_FILES * BOARD_FILES + BOARD_RANKS * BOARD_RANKS + channels)
}

pub(super) fn mobile_block_bias_size(channels: usize) -> usize {
    channels * 4
}

pub(super) fn value_head_map_size(value_channels: usize) -> usize {
    value_channels * BOARD_PLANES_SIZE
}

pub(super) fn value_head_features(channels: usize, value_channels: usize) -> usize {
    value_head_map_size(value_channels) + value_channels * 4 + cnn_pooled_size(channels) * 2
}

pub(super) fn value_relation_weight_size(channels: usize) -> usize {
    let ffn_channels = channels * VALUE_RELATION_FFN_MULT;
    channels * (BOARD_FILES * BOARD_FILES + BOARD_RANKS * BOARD_RANKS)
        + ffn_channels * channels
        + channels * ffn_channels
}

pub(super) fn value_relation_bias_size(channels: usize) -> usize {
    channels * 3 + channels * VALUE_RELATION_FFN_MULT + channels
}

pub(super) struct AzEvalScratch {
    pub(super) hidden: Vec<f32>,
    pub(super) board: Vec<u8>,
    pub(super) conv1: Vec<f32>,
    pub(super) conv2: Vec<f32>,
    pub(super) conv_residual: Vec<f32>,
    pub(super) cnn_global: Vec<f32>,
    pub(super) value_features: Vec<f32>,
    pub(super) value_relation_tmp: Vec<f32>,
    pub(super) value_relation_hidden: Vec<f32>,
    pub(super) value_relation_delta: Vec<f32>,
    pub(super) value_relation_global: Vec<f32>,
    pub(super) value_tail: Vec<f32>,
    pub(super) value_pool: Vec<f32>,
    pub(super) value_hidden: Vec<f32>,
    pub(super) policy_condition: Vec<f32>,
    pub(super) logits: Vec<f32>,
    pub(super) priors: Vec<f32>,
}

impl AzEvalScratch {
    pub(super) fn new(config: AzModelConfig) -> Self {
        let config = config.normalized();
        let channels = config.model_channels;
        let value_channels = config.value_head_channels;
        let pooled_size = cnn_pooled_size(channels);
        let value_map_size = value_head_map_size(value_channels);
        let value_features = value_head_features(channels, value_channels);
        Self {
            hidden: vec![0.0; config.hidden_size],
            board: vec![0; BOARD_HISTORY_SIZE],
            conv1: vec![0.0; channels * BOARD_PLANES_SIZE],
            conv2: vec![0.0; channels * BOARD_PLANES_SIZE],
            conv_residual: vec![0.0; channels * BOARD_PLANES_SIZE],
            cnn_global: vec![0.0; pooled_size],
            value_features: vec![0.0; channels * BOARD_PLANES_SIZE],
            value_relation_tmp: vec![0.0; channels * BOARD_PLANES_SIZE],
            value_relation_hidden: vec![
                0.0;
                channels * VALUE_RELATION_FFN_MULT * BOARD_PLANES_SIZE
            ],
            value_relation_delta: vec![0.0; channels * BOARD_PLANES_SIZE],
            value_relation_global: vec![0.0; pooled_size],
            value_tail: vec![0.0; value_map_size],
            value_pool: vec![0.0; value_features],
            value_hidden: vec![0.0; config.value_hidden_size],
            policy_condition: vec![0.0; POLICY_CONDITION_SIZE],
            logits: Vec::with_capacity(192),
            priors: Vec::with_capacity(192),
        }
    }
}

#[derive(Debug)]
pub struct AzModel {
    pub model_config: AzModelConfig,
    pub hidden_size: usize,
    pub board_conv1_weights: Vec<f32>,
    pub board_conv1_bias: Vec<f32>,
    pub board_conv2_weights: Vec<f32>,
    pub board_conv2_bias: Vec<f32>,
    pub position_embed: Vec<f32>,
    pub board_hidden: Vec<f32>,
    pub board_hidden_bias: Vec<f32>,
    pub value_relation_weights: Vec<f32>,
    pub value_relation_bias: Vec<f32>,
    pub value_tail_conv_weights: Vec<f32>,
    pub value_tail_conv_bias: Vec<f32>,
    pub value_intermediate_hidden: Vec<f32>,
    pub value_intermediate_bias: Vec<f32>,
    pub value_logits_weights: Vec<f32>,
    pub value_direct_logits_weights: Vec<f32>,
    pub value_logits_bias: Vec<f32>,
    pub value_scalar_hidden_weights: Vec<f32>,
    pub value_scalar_direct_weights: Vec<f32>,
    pub value_scalar_bias: Vec<f32>,
    pub policy_from_weights: Vec<f32>,
    pub policy_from_bias: Vec<f32>,
    pub policy_to_weights: Vec<f32>,
    pub policy_to_bias: Vec<f32>,
    pub policy_pair_weights: Vec<f32>,
    pub policy_move_bias: Vec<f32>,
    pub policy_feature_hidden: Vec<f32>,
    pub policy_feature_cnn: Vec<f32>,
    pub policy_feature_bias: Vec<f32>,
    pub(super) gpu_trainer: Option<Box<train_gpu::GpuTrainer>>,
}

// Architecture notes:
// - v8 is a Gated Multi-Scale Ray Mobile-CNN. The board is canonical
//   side-to-move input, then a sparse 3x3 stem and residual blocks with local
//   depthwise-3x3, learnable row mixing, learnable column mixing, and
//   pointwise-1x1 channel mixing. Row/column gates start small, so early
//   training remains close to the old local mobile block while value/policy can
//   learn to amplify long-line context for rooks, cannons, pins, and king lines.
// - Value is intentionally closer to the ZeroForge GNN readout than to the old
//   NNUE head: 1x1 conv -> attention/mean/max/std line-aware pooling plus a
//   small spatial tail -> leaky MLP -> three outcome logits. The leaky gate is
//   important; a hard ReLU can kill value gradients on tiny or early self-play
//   batches and masquerade as a search/TD problem.
// - Policy is a lightweight legal-move scorer: from-square score + to-square
//   score + channelwise from/to interaction + move-geometry condition + move
//   bias. This is much cheaper than a dense move tower while still giving the
//   action scorer a direct pair feature.
// - Do not resurrect the old sparse V4/NNUE-style board identities here. They
//   made offline fitting easier but muddied whether self-play value was learning
//   from board structure.

impl Clone for AzModel {
    fn clone(&self) -> Self {
        Self {
            model_config: self.model_config,
            hidden_size: self.hidden_size,
            board_conv1_weights: self.board_conv1_weights.clone(),
            board_conv1_bias: self.board_conv1_bias.clone(),
            board_conv2_weights: self.board_conv2_weights.clone(),
            board_conv2_bias: self.board_conv2_bias.clone(),
            position_embed: self.position_embed.clone(),
            board_hidden: self.board_hidden.clone(),
            board_hidden_bias: self.board_hidden_bias.clone(),
            value_relation_weights: self.value_relation_weights.clone(),
            value_relation_bias: self.value_relation_bias.clone(),
            value_tail_conv_weights: self.value_tail_conv_weights.clone(),
            value_tail_conv_bias: self.value_tail_conv_bias.clone(),
            value_intermediate_hidden: self.value_intermediate_hidden.clone(),
            value_intermediate_bias: self.value_intermediate_bias.clone(),
            value_logits_weights: self.value_logits_weights.clone(),
            value_direct_logits_weights: self.value_direct_logits_weights.clone(),
            value_logits_bias: self.value_logits_bias.clone(),
            value_scalar_hidden_weights: self.value_scalar_hidden_weights.clone(),
            value_scalar_direct_weights: self.value_scalar_direct_weights.clone(),
            value_scalar_bias: self.value_scalar_bias.clone(),
            policy_from_weights: self.policy_from_weights.clone(),
            policy_from_bias: self.policy_from_bias.clone(),
            policy_to_weights: self.policy_to_weights.clone(),
            policy_to_bias: self.policy_to_bias.clone(),
            policy_pair_weights: self.policy_pair_weights.clone(),
            policy_move_bias: self.policy_move_bias.clone(),
            policy_feature_hidden: self.policy_feature_hidden.clone(),
            policy_feature_cnn: self.policy_feature_cnn.clone(),
            policy_feature_bias: self.policy_feature_bias.clone(),
            gpu_trainer: None,
        }
    }
}

impl AzModel {
    pub fn random(hidden_size: usize, seed: u64) -> Self {
        Self::random_with_config(AzModelConfig::with_hidden_size(hidden_size), seed)
    }

    pub fn random_with_config(config: AzModelConfig, seed: u64) -> Self {
        let config = config.normalized();
        config
            .validate_supported()
            .expect("unsupported model config");
        let hidden_size = config.hidden_size;
        let channels = config.model_channels;
        let blocks = config.model_blocks;
        let value_channels = config.value_head_channels;
        let value_hidden_size = config.value_hidden_size;
        let pooled_size = cnn_pooled_size(channels);
        let mobile_weight_size = mobile_block_weight_size(channels);
        let mobile_bias_size = mobile_block_bias_size(channels);
        let value_relation_weight_size = value_relation_weight_size(channels);
        let value_relation_bias_size = value_relation_bias_size(channels);
        let value_features = value_head_features(channels, value_channels);
        let mut rng = SplitMix64::new(seed);
        let board_conv1_weights: Vec<f32> = (0..channels
            * BOARD_CHANNELS
            * BOARD_INPUT_KERNEL_AREA)
            .map(|_| rng.weight((2.0 / (BOARD_CHANNELS * BOARD_INPUT_KERNEL_AREA) as f32).sqrt()))
            .collect();
        let board_conv1_bias = vec![0.0; channels];
        let dw_len = channels * CNN_KERNEL_AREA;
        let row_len = channels * BOARD_FILES * BOARD_FILES;
        let col_len = channels * BOARD_RANKS * BOARD_RANKS;
        let board_conv2_weights: Vec<f32> = (0..blocks * mobile_weight_size)
            .map(|index| {
                let block_offset = index % mobile_weight_size;
                let scale = if block_offset < dw_len {
                    (2.0 / CNN_KERNEL_AREA as f32).sqrt()
                } else if block_offset < dw_len + row_len {
                    (2.0 / BOARD_FILES as f32).sqrt() * 0.25
                } else if block_offset < dw_len + row_len + col_len {
                    (2.0 / BOARD_RANKS as f32).sqrt() * 0.25
                } else {
                    (2.0 / channels as f32).sqrt()
                };
                rng.weight(scale)
            })
            .collect();
        let mut board_conv2_bias = vec![0.0; blocks * mobile_bias_size];
        for block in 0..blocks {
            let bias_start = block * mobile_bias_size;
            for channel in 0..channels {
                board_conv2_bias[bias_start + channels + channel] = 0.1;
                board_conv2_bias[bias_start + channels * 2 + channel] = 0.1;
            }
        }
        let position_embed = (0..channels * BOARD_PLANES_SIZE)
            .map(|_| rng.weight(0.02))
            .collect();
        let board_hidden: Vec<f32> = (0..hidden_size * pooled_size)
            .map(|_| rng.weight((2.0 / pooled_size as f32).sqrt()))
            .collect();
        let board_hidden_bias = vec![0.0; hidden_size];
        let value_relation_weights = (0..VALUE_RELATION_LAYERS * value_relation_weight_size)
            .map(|index| {
                let block_offset = index % value_relation_weight_size;
                let row_len = channels * BOARD_FILES * BOARD_FILES;
                let col_len = channels * BOARD_RANKS * BOARD_RANKS;
                let up_end = row_len + col_len + channels * channels * VALUE_RELATION_FFN_MULT;
                let scale = if block_offset < row_len {
                    (2.0 / BOARD_FILES as f32).sqrt() * 0.20
                } else if block_offset < row_len + col_len {
                    (2.0 / BOARD_RANKS as f32).sqrt() * 0.20
                } else if block_offset < up_end {
                    (2.0 / channels as f32).sqrt()
                } else {
                    (2.0 / (channels * VALUE_RELATION_FFN_MULT) as f32).sqrt()
                };
                rng.weight(scale)
            })
            .collect();
        let mut value_relation_bias = vec![0.0; VALUE_RELATION_LAYERS * value_relation_bias_size];
        for layer in 0..VALUE_RELATION_LAYERS {
            let bias_start = layer * value_relation_bias_size;
            for channel in 0..channels {
                value_relation_bias[bias_start + channel] = 0.1;
                value_relation_bias[bias_start + channels + channel] = 0.1;
                value_relation_bias[bias_start + channels * 2 + channel] = 0.1;
            }
        }
        let value_tail_conv_weights: Vec<f32> = (0..value_channels * channels)
            .map(|_| rng.weight((2.0 / channels as f32).sqrt()))
            .collect();
        let value_tail_conv_bias = vec![0.0; value_channels];
        let value_intermediate_hidden = (0..value_hidden_size * value_features)
            .map(|_| rng.weight((2.0 / value_features as f32).sqrt()))
            .collect();
        let value_intermediate_bias = vec![0.0; value_hidden_size];
        let value_logits_weights = (0..VALUE_LOGITS * value_hidden_size)
            .map(|_| rng.weight((2.0 / value_hidden_size as f32).sqrt()))
            .collect();
        let value_direct_logits_weights = vec![0.0; VALUE_LOGITS * value_features];
        let value_logits_bias = vec![0.0; VALUE_LOGITS];
        let value_scalar_hidden_weights = (0..value_hidden_size)
            .map(|_| rng.weight((2.0 / value_hidden_size as f32).sqrt() * 0.5))
            .collect();
        let value_scalar_direct_weights = (0..value_features)
            .map(|_| rng.weight((2.0 / value_features as f32).sqrt() * 0.1))
            .collect();
        let value_scalar_bias = vec![0.0; 1];
        let policy_from_weights = (0..channels)
            .map(|_| rng.weight((2.0 / channels as f32).sqrt() * 0.25))
            .collect();
        let policy_from_bias = vec![0.0; 1];
        let policy_to_weights = (0..channels)
            .map(|_| rng.weight((2.0 / channels as f32).sqrt() * 0.25))
            .collect();
        let policy_to_bias = vec![0.0; 1];
        let policy_pair_weights = vec![0.0; channels];
        let policy_move_bias = vec![0.0; DENSE_MOVE_SPACE];
        let policy_feature_hidden = (0..POLICY_CONDITION_SIZE * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt() * 0.5))
            .collect();
        let policy_feature_cnn = (0..POLICY_CONDITION_SIZE * pooled_size)
            .map(|_| rng.weight((2.0 / pooled_size as f32).sqrt() * 0.5))
            .collect();
        let policy_feature_bias = vec![0.0; POLICY_CONDITION_SIZE];
        Self {
            model_config: config,
            hidden_size,
            board_conv1_weights,
            board_conv1_bias,
            board_conv2_weights,
            board_conv2_bias,
            position_embed,
            board_hidden,
            board_hidden_bias,
            value_relation_weights,
            value_relation_bias,
            value_tail_conv_weights,
            value_tail_conv_bias,
            value_intermediate_hidden,
            value_intermediate_bias,
            value_logits_weights,
            value_direct_logits_weights,
            value_logits_bias,
            value_scalar_hidden_weights,
            value_scalar_direct_weights,
            value_scalar_bias,
            policy_from_weights,
            policy_from_bias,
            policy_to_weights,
            policy_to_bias,
            policy_pair_weights,
            policy_move_bias,
            policy_feature_hidden,
            policy_feature_cnn,
            policy_feature_bias,
            gpu_trainer: None,
        }
    }

    pub(super) fn evaluate_with_scratch(
        &self,
        position: &Position,
        history: &[HistoryMove],
        moves: &[Move],
        scratch: &mut AzEvalScratch,
    ) -> f32 {
        let side = position.side_to_move();
        extract_board_planes(position, history, &mut scratch.board);
        self.board_forward_into(
            &scratch.board,
            &mut scratch.conv1,
            &mut scratch.conv2,
            &mut scratch.conv_residual,
            &mut scratch.cnn_global,
        );
        self.board_hidden_into(&scratch.cnn_global, &mut scratch.hidden);
        self.value_relation_into(
            &scratch.conv1,
            &mut scratch.value_features,
            &mut scratch.value_relation_tmp,
            &mut scratch.value_relation_hidden,
            &mut scratch.value_relation_delta,
        );
        self.value_head_into(
            &scratch.value_features,
            &scratch.cnn_global,
            &mut scratch.value_relation_global,
            &mut scratch.value_tail,
            &mut scratch.value_pool,
            &mut scratch.value_hidden,
        );
        let value = self.value_from_hidden_scratch(scratch);
        self.policy_condition_into(
            &scratch.hidden,
            &scratch.cnn_global,
            &mut scratch.policy_condition,
        );
        scratch.logits.resize(moves.len(), 0.0);
        for (index, mv) in moves.iter().enumerate() {
            scratch.logits[index] = self.policy_logit_from_hidden_index(
                &scratch.conv1,
                &scratch.cnn_global,
                &scratch.policy_condition,
                side,
                *mv,
                dense_move_index(canonical_move(side, *mv)),
            );
        }
        value
    }

    fn board_forward_into(
        &self,
        board: &[u8],
        features: &mut Vec<f32>,
        tmp: &mut Vec<f32>,
        delta: &mut Vec<f32>,
        cnn_global: &mut Vec<f32>,
    ) {
        let channels = self.model_config.model_channels;
        let blocks = self.model_config.model_blocks;
        let pooled_size = cnn_pooled_size(channels);
        let mobile_weight_size = mobile_block_weight_size(channels);
        let mobile_bias_size = mobile_block_bias_size(channels);
        features.resize(channels * BOARD_PLANES_SIZE, 0.0);
        tmp.resize(channels * BOARD_PLANES_SIZE, 0.0);
        delta.resize(channels * BOARD_PLANES_SIZE, 0.0);
        cnn_global.resize(pooled_size, 0.0);
        conv_relu_board_layer_sparse_3x3(
            board,
            channels,
            &self.board_conv1_weights,
            &self.board_conv1_bias,
            features,
        );
        for (feature, pos) in features.iter_mut().zip(&self.position_embed) {
            *feature = (*feature + pos).max(0.0);
        }
        for block in 0..blocks {
            let weight_start = block * mobile_weight_size;
            let bias_start = block * mobile_bias_size;
            residual_mobile_block_generic(
                features,
                tmp,
                delta,
                channels,
                &self.board_conv2_weights[weight_start..weight_start + mobile_weight_size],
                &self.board_conv2_bias[bias_start..bias_start + mobile_bias_size],
            );
        }
        pool_cnn_features(features, channels, cnn_global);
    }

    fn board_hidden_into(&self, cnn_global: &[f32], hidden: &mut Vec<f32>) {
        let pooled_size = cnn_pooled_size(self.model_config.model_channels);
        hidden.resize(self.hidden_size, 0.0);
        hidden.copy_from_slice(&self.board_hidden_bias);
        for (out, hidden_value) in hidden.iter_mut().enumerate().take(self.hidden_size) {
            let row = &self.board_hidden[out * pooled_size..(out + 1) * pooled_size];
            for (cnn_value, weight) in cnn_global.iter().zip(row) {
                *hidden_value += cnn_value * weight;
            }
            *hidden_value = (*hidden_value).max(0.0);
        }
    }

    fn value_relation_into(
        &self,
        features: &[f32],
        value_features: &mut Vec<f32>,
        tmp: &mut Vec<f32>,
        hidden: &mut Vec<f32>,
        delta: &mut Vec<f32>,
    ) {
        let channels = self.model_config.model_channels;
        let ffn_channels = channels * VALUE_RELATION_FFN_MULT;
        let weight_size = value_relation_weight_size(channels);
        let bias_size = value_relation_bias_size(channels);
        value_features.resize(channels * BOARD_PLANES_SIZE, 0.0);
        value_features.copy_from_slice(features);
        tmp.resize(channels * BOARD_PLANES_SIZE, 0.0);
        hidden.resize(ffn_channels * BOARD_PLANES_SIZE, 0.0);
        delta.resize(channels * BOARD_PLANES_SIZE, 0.0);
        for layer in 0..VALUE_RELATION_LAYERS {
            let weight_start = layer * weight_size;
            let bias_start = layer * bias_size;
            residual_value_relation_block_generic(
                value_features,
                tmp,
                hidden,
                delta,
                channels,
                &self.value_relation_weights[weight_start..weight_start + weight_size],
                &self.value_relation_bias[bias_start..bias_start + bias_size],
            );
        }
    }

    fn value_head_into(
        &self,
        features: &[f32],
        cnn_global: &[f32],
        relation_global: &mut Vec<f32>,
        value_tail: &mut Vec<f32>,
        pooled: &mut Vec<f32>,
        hidden: &mut Vec<f32>,
    ) {
        let channels = self.model_config.model_channels;
        let value_channels = self.model_config.value_head_channels;
        let value_hidden_size = self.model_config.value_hidden_size;
        let value_map_size = value_head_map_size(value_channels);
        let value_features = value_head_features(channels, value_channels);
        value_tail.resize(value_map_size, 0.0);
        conv1x1_linear_layer_dense_generic(
            features,
            channels,
            value_channels,
            &self.value_tail_conv_weights,
            &self.value_tail_conv_bias,
            value_tail,
        );
        relation_global.resize(cnn_pooled_size(channels), 0.0);
        pool_cnn_features(features, channels, relation_global);
        value_pool_features(
            value_tail,
            relation_global,
            cnn_global,
            value_channels,
            channels,
            pooled,
        );
        hidden.resize(value_hidden_size, 0.0);
        hidden.copy_from_slice(&self.value_intermediate_bias);
        for (j, value) in hidden.iter_mut().enumerate().take(value_hidden_size) {
            let row = &self.value_intermediate_hidden[j * value_features..(j + 1) * value_features];
            for (feature, weight) in pooled.iter().zip(row) {
                *value += feature * weight;
            }
            if *value < 0.0 {
                *value *= VALUE_HEAD_LEAK;
            }
        }
    }

    fn value_from_hidden_scratch(&self, scratch: &mut AzEvalScratch) -> f32 {
        let mut scalar_logit = self.value_scalar_bias[0];
        for (hidden, weight) in scratch
            .value_hidden
            .iter()
            .zip(self.value_scalar_hidden_weights.iter())
        {
            scalar_logit += hidden * weight;
        }
        for (feature, weight) in scratch
            .value_pool
            .iter()
            .zip(self.value_scalar_direct_weights.iter())
        {
            scalar_logit += feature * weight;
        }
        scalar_logit.tanh()
    }

    fn policy_condition_into(&self, hidden: &[f32], cnn_global: &[f32], out: &mut Vec<f32>) {
        let pooled_size = cnn_pooled_size(self.model_config.model_channels);
        out.resize(POLICY_CONDITION_SIZE, 0.0);
        out.copy_from_slice(&self.policy_feature_bias);
        for (feature, value) in out.iter_mut().enumerate().take(POLICY_CONDITION_SIZE) {
            let hidden_row = &self.policy_feature_hidden
                [feature * self.hidden_size..(feature + 1) * self.hidden_size];
            *value += dot_product(hidden, hidden_row);
            let cnn_row =
                &self.policy_feature_cnn[feature * pooled_size..(feature + 1) * pooled_size];
            *value += dot_product(cnn_global, cnn_row);
        }
    }

    pub(super) fn validate(&self) -> io::Result<()> {
        let config = self.model_config.normalized();
        let channels = config.model_channels;
        let blocks = config.model_blocks;
        let value_channels = config.value_head_channels;
        let value_hidden_size = config.value_hidden_size;
        let pooled_size = cnn_pooled_size(channels);
        let value_features = value_head_features(channels, value_channels);
        if self.hidden_size != config.hidden_size
            || self.board_conv1_weights.len() != channels * BOARD_CHANNELS * BOARD_INPUT_KERNEL_AREA
            || self.board_conv1_bias.len() != channels
            || self.board_conv2_weights.len() != blocks * mobile_block_weight_size(channels)
            || self.board_conv2_bias.len() != blocks * mobile_block_bias_size(channels)
            || self.position_embed.len() != channels * BOARD_PLANES_SIZE
            || self.board_hidden.len() != self.hidden_size * pooled_size
            || self.board_hidden_bias.len() != self.hidden_size
            || self.value_relation_weights.len()
                != VALUE_RELATION_LAYERS * value_relation_weight_size(channels)
            || self.value_relation_bias.len()
                != VALUE_RELATION_LAYERS * value_relation_bias_size(channels)
            || self.value_tail_conv_weights.len() != value_channels * channels
            || self.value_tail_conv_bias.len() != value_channels
            || self.value_intermediate_hidden.len() != value_hidden_size * value_features
            || self.value_intermediate_bias.len() != value_hidden_size
            || self.value_logits_weights.len() != VALUE_LOGITS * value_hidden_size
            || self.value_direct_logits_weights.len() != VALUE_LOGITS * value_features
            || self.value_logits_bias.len() != VALUE_LOGITS
            || self.value_scalar_hidden_weights.len() != value_hidden_size
            || self.value_scalar_direct_weights.len() != value_features
            || self.value_scalar_bias.len() != 1
            || self.policy_from_weights.len() != channels
            || self.policy_from_bias.len() != 1
            || self.policy_to_weights.len() != channels
            || self.policy_to_bias.len() != 1
            || self.policy_pair_weights.len() != channels
            || self.policy_move_bias.len() != DENSE_MOVE_SPACE
            || self.policy_feature_hidden.len() != POLICY_CONDITION_SIZE * self.hidden_size
            || self.policy_feature_cnn.len() != POLICY_CONDITION_SIZE * pooled_size
            || self.policy_feature_bias.len() != POLICY_CONDITION_SIZE
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "AzModel vector length mismatch",
            ));
        }
        Ok(())
    }

    fn policy_logit_from_hidden_index(
        &self,
        features: &[f32],
        _cnn_global: &[f32],
        policy_condition: &[f32],
        side: Color,
        mv: Move,
        move_index: usize,
    ) -> f32 {
        let canonical = canonical_move(side, mv);
        let from = canonical.from as usize;
        let to = canonical.to as usize;
        let feature_offset = move_index * POLICY_CONDITION_SIZE;
        let move_features =
            &policy_move_features()[feature_offset..feature_offset + POLICY_CONDITION_SIZE];
        self.policy_move_bias[move_index]
            + self.policy_square_score(
                features,
                from,
                &self.policy_from_weights,
                self.policy_from_bias[0],
            )
            + self.policy_square_score(
                features,
                to,
                &self.policy_to_weights,
                self.policy_to_bias[0],
            )
            + self.policy_pair_score(features, from, to)
            + dot_product(policy_condition, move_features)
    }

    fn policy_square_score(&self, features: &[f32], sq: usize, weights: &[f32], bias: f32) -> f32 {
        let mut value = bias;
        for channel in 0..self.model_config.model_channels {
            value += features[channel * BOARD_PLANES_SIZE + sq] * weights[channel];
        }
        value
    }

    fn policy_pair_score(&self, features: &[f32], from: usize, to: usize) -> f32 {
        let mut value = 0.0;
        for channel in 0..self.model_config.model_channels {
            let base = channel * BOARD_PLANES_SIZE;
            value +=
                features[base + from] * features[base + to] * self.policy_pair_weights[channel];
        }
        value
    }
}
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
    for (history_index, entry) in history.iter().rev().take(HISTORY_PLIES).enumerate() {
        let piece_plane =
            (canonical_piece_plane(side, entry.piece.color, entry.piece.kind) + 1) as u8;
        let captured_plane = entry
            .captured
            .map(|piece| (canonical_piece_plane(side, piece.color, piece.kind) + 1) as u8)
            .unwrap_or(0);
        rewound[canonical_square(side, entry.mv.to as usize)] = captured_plane;
        rewound[canonical_square(side, entry.mv.from as usize)] = piece_plane;
        let start = (history_index + 1) * BOARD_PLANES_SIZE;
        board[start..start + BOARD_PLANES_SIZE].copy_from_slice(&rewound);
    }
}

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

pub(super) fn canonical_piece_plane(side: Color, piece_color: Color, kind: PieceKind) -> usize {
    let canonical_color = if piece_color == side {
        Color::Red
    } else {
        Color::Black
    };
    absolute_piece_plane(canonical_color, kind)
}

fn value_pool_features(
    value_tail: &[f32],
    relation_global: &[f32],
    cnn_global: &[f32],
    value_channels: usize,
    channels: usize,
    pooled: &mut Vec<f32>,
) {
    let value_map_size = value_head_map_size(value_channels);
    let value_features = value_head_features(channels, value_channels);
    debug_assert_eq!(value_tail.len(), value_map_size);
    debug_assert_eq!(relation_global.len(), cnn_pooled_size(channels));
    debug_assert_eq!(cnn_global.len(), cnn_pooled_size(channels));
    pooled.resize(value_features, 0.0);
    let logits = &value_tail[..BOARD_PLANES_SIZE];
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut weights = [0.0f32; BOARD_PLANES_SIZE];
    let mut denom = 0.0f32;
    for (weight, &logit) in weights.iter_mut().zip(logits) {
        let value = (logit - max_logit).exp();
        *weight = value;
        denom += value;
    }
    let inv_denom = denom.max(1e-12).recip();
    for weight in &mut weights {
        *weight *= inv_denom;
    }

    for channel in 0..value_channels {
        let start = channel * BOARD_PLANES_SIZE;
        let row = &value_tail[start..start + BOARD_PLANES_SIZE];
        let mut attn = 0.0f32;
        let mut sum = 0.0f32;
        let mut sum_sq = 0.0f32;
        let mut max_value = f32::NEG_INFINITY;
        for (&feature, &weight) in row.iter().zip(weights.iter()) {
            attn += feature * weight;
            sum += feature;
            sum_sq += feature * feature;
            max_value = max_value.max(feature);
        }
        let mean = sum / BOARD_PLANES_SIZE as f32;
        let variance = (sum_sq / BOARD_PLANES_SIZE as f32 - mean * mean).max(0.0);
        pooled[channel] = attn;
        pooled[value_channels + channel] = mean;
        pooled[value_channels * 2 + channel] = max_value;
        pooled[value_channels * 3 + channel] = variance.sqrt();
    }
    let flat_start = value_channels * 4;
    pooled[flat_start..flat_start + value_map_size].copy_from_slice(value_tail);
    let relation_start = flat_start + value_map_size;
    let pooled_size = cnn_pooled_size(channels);
    pooled[relation_start..relation_start + pooled_size].copy_from_slice(relation_global);
    pooled[relation_start + pooled_size..].copy_from_slice(cnn_global);
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

pub(super) struct MoveMap {
    pub(super) sparse_to_dense: [u16; SPARSE_MOVE_SPACE],
    #[allow(dead_code)]
    pub(super) dense_to_sparse: [u16; DENSE_MOVE_SPACE],
}

pub(super) fn move_map() -> &'static MoveMap {
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

pub(super) fn policy_move_from_select() -> &'static [f32] {
    use std::sync::OnceLock;
    static SELECT: OnceLock<Vec<f32>> = OnceLock::new();
    SELECT.get_or_init(|| {
        let mut select = vec![0.0f32; BOARD_SIZE * DENSE_MOVE_SPACE];
        for (move_index, &sparse) in move_map().dense_to_sparse.iter().enumerate() {
            let from = sparse as usize / BOARD_SIZE;
            select[from * DENSE_MOVE_SPACE + move_index] = 1.0;
        }
        select
    })
}

pub(super) fn policy_move_to_select() -> &'static [f32] {
    use std::sync::OnceLock;
    static SELECT: OnceLock<Vec<f32>> = OnceLock::new();
    SELECT.get_or_init(|| {
        let mut select = vec![0.0f32; BOARD_SIZE * DENSE_MOVE_SPACE];
        for (move_index, &sparse) in move_map().dense_to_sparse.iter().enumerate() {
            let to = sparse as usize % BOARD_SIZE;
            select[to * DENSE_MOVE_SPACE + move_index] = 1.0;
        }
        select
    })
}

pub(super) fn dense_move_index(mv: Move) -> usize {
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
