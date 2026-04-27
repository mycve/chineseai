use std::io;

use crate::board_transform::{HISTORY_PLIES, HistoryMove, canonical_move, canonical_square};
use crate::xiangqi::{BOARD_FILES, BOARD_SIZE, Color, Move, PieceKind, Position};

use super::model_config::AzModelConfig;
use super::model_ops::{
    conv_relu_board_layer_sparse_3x3, conv1x1_relu_layer_dense_generic, dot_product,
    pool_cnn_features, residual_cnn_block_generic, scalar_value_from_logits,
};
use super::train_gpu;

pub const AZ_MODEL_BINARY_MAGIC: &[u8] = b"AZM1";
pub(super) const AZ_MODEL_BINARY_VERSION: u32 = 2;
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
pub(super) const BOARD_INPUT_KERNEL_AREA: usize = 9;
pub(super) const CNN_KERNEL_AREA: usize = 9;
pub(super) const CNN_POOL_BLOCKS: usize = 4;
pub(super) const CNN_POOLED_SIZE: usize = CNN_CHANNELS * CNN_POOL_BLOCKS;
pub(super) const VALUE_HEAD_FEATURES: usize = VALUE_HEAD_CHANNELS * BOARD_PLANES_SIZE;
pub(super) const POLICY_CONDITION_SIZE: usize = 32;
pub(super) const VALUE_SCALE_CP: f32 = 1000.0;

pub(super) struct AzEvalScratch {
    pub(super) hidden: Vec<f32>,
    pub(super) board: Vec<u8>,
    pub(super) conv1: Vec<f32>,
    pub(super) conv2: Vec<f32>,
    pub(super) conv_residual: Vec<f32>,
    pub(super) cnn_global: Vec<f32>,
    pub(super) value_tail: Vec<f32>,
    pub(super) value_hidden: Vec<f32>,
    pub(super) value_logits: Vec<f32>,
    pub(super) policy_condition: Vec<f32>,
    pub(super) logits: Vec<f32>,
    pub(super) priors: Vec<f32>,
}

impl AzEvalScratch {
    pub(super) fn new(hidden_size: usize) -> Self {
        Self {
            hidden: vec![0.0; hidden_size],
            board: vec![0; BOARD_HISTORY_SIZE],
            conv1: vec![0.0; CNN_CHANNELS * BOARD_PLANES_SIZE],
            conv2: vec![0.0; CNN_CHANNELS * BOARD_PLANES_SIZE],
            conv_residual: vec![0.0; CNN_CHANNELS * BOARD_PLANES_SIZE],
            cnn_global: vec![0.0; CNN_POOLED_SIZE],
            value_tail: vec![0.0; VALUE_HEAD_FEATURES],
            value_hidden: vec![0.0; VALUE_HIDDEN_SIZE],
            value_logits: vec![0.0; VALUE_LOGITS],
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
    pub board_hidden: Vec<f32>,
    pub board_hidden_bias: Vec<f32>,
    pub value_tail_conv_weights: Vec<f32>,
    pub value_tail_conv_bias: Vec<f32>,
    pub value_intermediate_hidden: Vec<f32>,
    pub value_intermediate_bias: Vec<f32>,
    pub value_logits_weights: Vec<f32>,
    pub value_logits_bias: Vec<f32>,
    pub policy_move_hidden: Vec<f32>,
    pub policy_move_cnn: Vec<f32>,
    pub policy_move_bias: Vec<f32>,
    pub policy_feature_hidden: Vec<f32>,
    pub policy_feature_cnn: Vec<f32>,
    pub policy_feature_bias: Vec<f32>,
    pub(super) gpu_trainer: Option<Box<train_gpu::GpuTrainer>>,
}

// Architecture notes:
// - v29 is a Tiny ResCNN. The board is canonical side-to-move input, then a
//   3x3 sparse stem and three 3x3 residual blocks keep spatial features alive
//   until the heads. This replaces the old early-pool/attention/value-embedding
//   hybrid that was too easy for policy and value to pull in different ways.
// - Value is now a standard CNN value head: 1x1 conv -> flatten -> MLP ->
//   three outcome logits. There is no piece-square value shortcut, no relation
//   shortcut, and no value-only attention path; if value improves, it is because
//   the spatial tower learned useful board features.
// - Policy still uses a CPU-friendly legal-move scorer. It reads the same CNN
//   pooled features plus a small move-geometry condition, so legal move masking
//   stays cheap while the board representation is now much closer to a normal
//   AlphaZero-style CNN tower.
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
            board_hidden: self.board_hidden.clone(),
            board_hidden_bias: self.board_hidden_bias.clone(),
            value_tail_conv_weights: self.value_tail_conv_weights.clone(),
            value_tail_conv_bias: self.value_tail_conv_bias.clone(),
            value_intermediate_hidden: self.value_intermediate_hidden.clone(),
            value_intermediate_bias: self.value_intermediate_bias.clone(),
            value_logits_weights: self.value_logits_weights.clone(),
            value_logits_bias: self.value_logits_bias.clone(),
            policy_move_hidden: self.policy_move_hidden.clone(),
            policy_move_cnn: self.policy_move_cnn.clone(),
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
        let mut rng = SplitMix64::new(seed);
        let board_conv1_weights: Vec<f32> = (0..CNN_CHANNELS
            * BOARD_CHANNELS
            * BOARD_INPUT_KERNEL_AREA)
            .map(|_| rng.weight((2.0 / (BOARD_CHANNELS * BOARD_INPUT_KERNEL_AREA) as f32).sqrt()))
            .collect();
        let board_conv1_bias = vec![0.0; CNN_CHANNELS];
        let residual_layer_count = RESIDUAL_BLOCKS * 2;
        let board_conv2_weights: Vec<f32> =
            (0..residual_layer_count * CNN_CHANNELS * CNN_CHANNELS * CNN_KERNEL_AREA)
                .map(|_| rng.weight((2.0 / (CNN_CHANNELS * CNN_KERNEL_AREA) as f32).sqrt()))
                .collect();
        let board_conv2_bias = vec![0.0; residual_layer_count * CNN_CHANNELS];
        let board_hidden: Vec<f32> = (0..hidden_size * CNN_POOLED_SIZE)
            .map(|_| rng.weight((2.0 / CNN_POOLED_SIZE as f32).sqrt()))
            .collect();
        let board_hidden_bias = vec![0.0; hidden_size];
        let value_tail_conv_weights: Vec<f32> = (0..VALUE_HEAD_CHANNELS * CNN_CHANNELS)
            .map(|_| rng.weight((2.0 / CNN_CHANNELS as f32).sqrt()))
            .collect();
        let value_tail_conv_bias = vec![0.0; VALUE_HEAD_CHANNELS];
        let value_intermediate_hidden = (0..VALUE_HIDDEN_SIZE * VALUE_HEAD_FEATURES)
            .map(|_| rng.weight((2.0 / VALUE_HEAD_FEATURES as f32).sqrt()))
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
        let policy_feature_hidden = (0..POLICY_CONDITION_SIZE * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt() * 0.5))
            .collect();
        let policy_feature_cnn = (0..POLICY_CONDITION_SIZE * CNN_POOLED_SIZE)
            .map(|_| rng.weight((2.0 / CNN_POOLED_SIZE as f32).sqrt() * 0.5))
            .collect();
        let policy_feature_bias = vec![0.0; POLICY_CONDITION_SIZE];
        Self {
            model_config: config,
            hidden_size,
            board_conv1_weights,
            board_conv1_bias,
            board_conv2_weights,
            board_conv2_bias,
            board_hidden,
            board_hidden_bias,
            value_tail_conv_weights,
            value_tail_conv_bias,
            value_intermediate_hidden,
            value_intermediate_bias,
            value_logits_weights,
            value_logits_bias,
            policy_move_hidden,
            policy_move_cnn,
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
        self.value_head_into(
            &scratch.conv1,
            &mut scratch.value_tail,
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
                &scratch.hidden,
                &scratch.cnn_global,
                &scratch.policy_condition,
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
        features.resize(CNN_CHANNELS * BOARD_PLANES_SIZE, 0.0);
        tmp.resize(CNN_CHANNELS * BOARD_PLANES_SIZE, 0.0);
        delta.resize(CNN_CHANNELS * BOARD_PLANES_SIZE, 0.0);
        cnn_global.resize(CNN_POOLED_SIZE, 0.0);
        conv_relu_board_layer_sparse_3x3(
            board,
            CNN_CHANNELS,
            &self.board_conv1_weights,
            &self.board_conv1_bias,
            features,
        );
        let block_weight_len = 2 * CNN_CHANNELS * CNN_CHANNELS * CNN_KERNEL_AREA;
        let block_bias_len = 2 * CNN_CHANNELS;
        for block in 0..RESIDUAL_BLOCKS {
            let weight_start = block * block_weight_len;
            let bias_start = block * block_bias_len;
            residual_cnn_block_generic(
                features,
                tmp,
                delta,
                CNN_CHANNELS,
                &self.board_conv2_weights[weight_start..weight_start + block_weight_len],
                &self.board_conv2_bias[bias_start..bias_start + block_bias_len],
            );
        }
        pool_cnn_features(features, CNN_CHANNELS, cnn_global);
    }

    fn board_hidden_into(&self, cnn_global: &[f32], hidden: &mut Vec<f32>) {
        hidden.resize(self.hidden_size, 0.0);
        hidden.copy_from_slice(&self.board_hidden_bias);
        for (out, hidden_value) in hidden.iter_mut().enumerate().take(self.hidden_size) {
            let row = &self.board_hidden[out * CNN_POOLED_SIZE..(out + 1) * CNN_POOLED_SIZE];
            for (cnn_value, weight) in cnn_global.iter().zip(row) {
                *hidden_value += cnn_value * weight;
            }
            *hidden_value = (*hidden_value).max(0.0);
        }
    }

    fn value_head_into(&self, features: &[f32], value_tail: &mut Vec<f32>, hidden: &mut Vec<f32>) {
        value_tail.resize(VALUE_HEAD_FEATURES, 0.0);
        conv1x1_relu_layer_dense_generic(
            features,
            CNN_CHANNELS,
            VALUE_HEAD_CHANNELS,
            &self.value_tail_conv_weights,
            &self.value_tail_conv_bias,
            value_tail,
        );
        hidden.resize(VALUE_HIDDEN_SIZE, 0.0);
        hidden.copy_from_slice(&self.value_intermediate_bias);
        for (j, value) in hidden.iter_mut().enumerate().take(VALUE_HIDDEN_SIZE) {
            let row = &self.value_intermediate_hidden
                [j * VALUE_HEAD_FEATURES..(j + 1) * VALUE_HEAD_FEATURES];
            for (feature, weight) in value_tail.iter().zip(row) {
                *value += feature * weight;
            }
            *value = (*value).max(0.0);
        }
    }

    fn value_from_hidden_scratch(&self, scratch: &mut AzEvalScratch) -> f32 {
        scratch
            .value_logits
            .copy_from_slice(&self.value_logits_bias);
        for out in 0..VALUE_LOGITS {
            let row =
                &self.value_logits_weights[out * VALUE_HIDDEN_SIZE..(out + 1) * VALUE_HIDDEN_SIZE];
            for (hidden, weight) in scratch.value_hidden.iter().zip(row) {
                scratch.value_logits[out] += hidden * weight;
            }
        }
        scalar_value_from_logits(&scratch.value_logits).0
    }

    fn policy_condition_into(&self, hidden: &[f32], cnn_global: &[f32], out: &mut Vec<f32>) {
        out.resize(POLICY_CONDITION_SIZE, 0.0);
        out.copy_from_slice(&self.policy_feature_bias);
        for (feature, value) in out.iter_mut().enumerate().take(POLICY_CONDITION_SIZE) {
            let hidden_row = &self.policy_feature_hidden
                [feature * self.hidden_size..(feature + 1) * self.hidden_size];
            *value += dot_product(hidden, hidden_row);
            let cnn_row = &self.policy_feature_cnn
                [feature * CNN_POOLED_SIZE..(feature + 1) * CNN_POOLED_SIZE];
            *value += dot_product(cnn_global, cnn_row);
        }
    }

    pub(super) fn validate(&self) -> io::Result<()> {
        if self.board_conv1_weights.len() != CNN_CHANNELS * BOARD_CHANNELS * BOARD_INPUT_KERNEL_AREA
            || self.board_conv1_bias.len() != CNN_CHANNELS
            || self.board_conv2_weights.len()
                != RESIDUAL_BLOCKS * 2 * CNN_CHANNELS * CNN_CHANNELS * CNN_KERNEL_AREA
            || self.board_conv2_bias.len() != RESIDUAL_BLOCKS * 2 * CNN_CHANNELS
            || self.board_hidden.len() != self.hidden_size * CNN_POOLED_SIZE
            || self.board_hidden_bias.len() != self.hidden_size
            || self.value_tail_conv_weights.len() != VALUE_HEAD_CHANNELS * CNN_CHANNELS
            || self.value_tail_conv_bias.len() != VALUE_HEAD_CHANNELS
            || self.value_intermediate_hidden.len() != VALUE_HIDDEN_SIZE * VALUE_HEAD_FEATURES
            || self.value_intermediate_bias.len() != VALUE_HIDDEN_SIZE
            || self.value_logits_weights.len() != VALUE_LOGITS * VALUE_HIDDEN_SIZE
            || self.value_logits_bias.len() != VALUE_LOGITS
            || self.policy_move_hidden.len() != DENSE_MOVE_SPACE * self.hidden_size
            || self.policy_move_cnn.len() != DENSE_MOVE_SPACE * CNN_POOLED_SIZE
            || self.policy_move_bias.len() != DENSE_MOVE_SPACE
            || self.policy_feature_hidden.len() != POLICY_CONDITION_SIZE * self.hidden_size
            || self.policy_feature_cnn.len() != POLICY_CONDITION_SIZE * CNN_POOLED_SIZE
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
        hidden: &[f32],
        cnn_global: &[f32],
        policy_condition: &[f32],
        move_index: usize,
    ) -> f32 {
        let hidden_offset = move_index * self.hidden_size;
        let hidden_row = &self.policy_move_hidden[hidden_offset..hidden_offset + self.hidden_size];
        let cnn_offset = move_index * CNN_POOLED_SIZE;
        let cnn_row = &self.policy_move_cnn[cnn_offset..cnn_offset + CNN_POOLED_SIZE];
        let feature_offset = move_index * POLICY_CONDITION_SIZE;
        let move_features =
            &policy_move_features()[feature_offset..feature_offset + POLICY_CONDITION_SIZE];
        self.policy_move_bias[move_index]
            + dot_product(hidden, hidden_row)
            + dot_product(cnn_global, cnn_row)
            + dot_product(policy_condition, move_features)
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
