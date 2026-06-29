use candle_core::{Device, Result as CandleResult, Tensor, Var, backprop::GradStore};
use candle_nn::ops::log_softmax;

use super::{
    AzNnue, AzNnueArch, DENSE_MOVE_SPACE, PIECE_ATTENTION_HEADS, PIECE_ATTENTION_SIZE,
    PIECE_ATTENTION_TOTAL_SIZE, POLICY_MOVE_EMBED_SIZE, POLICY_PAIR_CONTEXT_SIZE,
    POLICY_TOWER_BLOCKS, STRUCTURAL_FILE_SIZE, STRUCTURAL_KING_PIECE_SIZE, STRUCTURAL_PIECE_SIZE,
    STRUCTURAL_RANK_SIZE, TRUNK_INNER_MULT, VALUE_HEAD_SIZE, VALUE_TOWER_BLOCKS, WDL_HEAD_SIZE,
    dataloader::PackedBatch, policy_move_from_features, policy_move_sparse_indices,
    policy_move_to_features, policy_move_transposed_sparse_indices,
};
use crate::nnue::AZ_NNUE_INPUT_SIZE;
use crate::xiangqi::BOARD_SIZE;

const RMS_NORM_EPS: f64 = 1.0e-6;

#[derive(Debug)]
pub(super) struct AzCandleModel {
    arch: AzNnueArch,
    input_hidden: Var,
    input_piece_hidden: Var,
    input_rank_hidden: Var,
    input_file_hidden: Var,
    input_king_piece_hidden: Var,
    hidden_bias: Var,
    piece_attention_query: Var,
    piece_attention_value: Var,
    piece_attention_output: Var,
    policy_tower_hidden: Var,
    policy_tower_bias: Var,
    policy_tower_output: Var,
    value_tower_hidden: Var,
    value_tower_bias: Var,
    value_tower_output: Var,
    value_head_hidden: Var,
    value_head_bias: Var,
    value_head_hidden2: Var,
    value_head_bias2: Var,
    value_head_output: Var,
    moves_left_hidden: Var,
    moves_left_bias_hidden: Var,
    moves_left_output: Var,
    moves_left_bias: Var,
    policy_move_bias: Var,
    policy_from_hidden: Var,
    policy_to_hidden: Var,
    policy_pair_context_hidden: Var,
    policy_pair_context_bias: Var,
    policy_pair_embedding: Var,
    policy_move_context_hidden: Var,
    policy_move_embedding: Var,
    policy_from_token_to_hidden: Var,
    policy_to_token_from_hidden: Var,
    policy_move_from_features: Tensor,
    policy_move_to_features: Tensor,
    policy_move_sparse_indices: Tensor,
    policy_move_transposed_sparse_indices: Tensor,
}

impl AzCandleModel {
    pub(super) fn forward(&self, batch: &BatchTensors) -> CandleResult<ForwardOutput> {
        let bsz = batch.batch_size;
        let hidden_size = self.arch.hidden_size;
        let feature_embeddings = self
            .input_hidden
            .index_select(&batch.feature_indices.flatten_all()?, 0)?
            .reshape((bsz, batch.max_features, hidden_size))?;
        let structural_piece = self
            .input_piece_hidden
            .index_select(&batch.structural_piece_indices.flatten_all()?, 0)?
            .reshape((bsz, batch.max_features, hidden_size))?;
        let structural_rank = self
            .input_rank_hidden
            .index_select(&batch.structural_rank_indices.flatten_all()?, 0)?
            .reshape((bsz, batch.max_features, hidden_size))?;
        let structural_file = self
            .input_file_hidden
            .index_select(&batch.structural_file_indices.flatten_all()?, 0)?
            .reshape((bsz, batch.max_features, hidden_size))?;
        let structural_us_king_piece = self
            .input_king_piece_hidden
            .index_select(&batch.structural_us_king_piece_indices.flatten_all()?, 0)?
            .reshape((bsz, batch.max_features, hidden_size))?;
        let structural_them_king_piece = self
            .input_king_piece_hidden
            .index_select(&batch.structural_them_king_piece_indices.flatten_all()?, 0)?
            .reshape((bsz, batch.max_features, hidden_size))?;
        let structural_embeddings = (structural_piece + structural_rank)?;
        let structural_embeddings = (structural_embeddings + structural_file)?;
        let structural_embeddings = (structural_embeddings + structural_us_king_piece)?;
        let structural_embeddings = (structural_embeddings + structural_them_king_piece)?;
        let structural_embeddings = structural_embeddings.broadcast_mul(&batch.structural_mask)?;
        let feature_embeddings = (feature_embeddings + structural_embeddings)?;
        let square_tokens = feature_embeddings
            .flatten_to(1)?
            .index_select(&batch.square_token_feature_indices.flatten_all()?, 0)?
            .reshape((bsz, BOARD_SIZE, hidden_size))?
            .broadcast_mul(&batch.square_token_mask)?;
        let sparse_pre = feature_embeddings
            .broadcast_mul(&batch.feature_mask)?
            .sum(1)?
            .broadcast_add(&self.hidden_bias)?;
        let attention_scores = feature_embeddings
            .unsqueeze(2)?
            .broadcast_mul(&self.piece_attention_query.reshape((
                1,
                1,
                PIECE_ATTENTION_HEADS,
                hidden_size,
            ))?)?
            .sum(3)?;
        let attention_mask = batch.structural_mask.squeeze(2)?.affine(1.0e9, -1.0e9)?;
        let attention_scores = attention_scores.broadcast_add(&attention_mask.unsqueeze(2)?)?;
        let attention_weights = log_softmax(&attention_scores, 1)?.exp()?.unsqueeze(3)?;
        let attention_values = feature_embeddings
            .flatten_to(1)?
            .matmul(&self.piece_attention_value.t()?)?
            .reshape((
                bsz,
                batch.max_features,
                PIECE_ATTENTION_HEADS,
                PIECE_ATTENTION_SIZE,
            ))?;
        let attention_context = attention_values
            .broadcast_mul(&attention_weights)?
            .sum(1)?
            .reshape((bsz, PIECE_ATTENTION_TOTAL_SIZE))?;
        let attention_residual = attention_context.matmul(&self.piece_attention_output.t()?)?;
        let sparse_pre = (sparse_pre + attention_residual)?;
        let sparse_hidden = sparse_pre.relu()?;
        let rms = sparse_hidden
            .sqr()?
            .mean_keepdim(1)?
            .affine(1.0, RMS_NORM_EPS)?
            .sqrt()?;
        let hidden = sparse_hidden.broadcast_div(&rms)?;
        let trunk_inner = hidden_size * TRUNK_INNER_MULT;
        let mut policy_hidden = hidden.clone();
        for block in 0..POLICY_TOWER_BLOCKS {
            let tower_hidden =
                self.policy_tower_hidden
                    .narrow(0, block * trunk_inner, trunk_inner)?;
            let tower_bias = self
                .policy_tower_bias
                .narrow(0, block * trunk_inner, trunk_inner)?;
            let tower_output =
                self.policy_tower_output
                    .narrow(0, block * hidden_size, hidden_size)?;
            let tower = policy_hidden
                .matmul(&tower_hidden.t()?)?
                .broadcast_add(&tower_bias)?
                .silu()?;
            let residual = tower.matmul(&tower_output.t()?)?;
            policy_hidden = (policy_hidden + residual)?;
            let tower_rms = policy_hidden
                .sqr()?
                .mean_keepdim(1)?
                .affine(1.0, RMS_NORM_EPS)?
                .sqrt()?;
            policy_hidden = policy_hidden.broadcast_div(&tower_rms)?;
        }
        let mut value_hidden = hidden.clone();
        for block in 0..VALUE_TOWER_BLOCKS {
            let tower_hidden =
                self.value_tower_hidden
                    .narrow(0, block * trunk_inner, trunk_inner)?;
            let tower_bias = self
                .value_tower_bias
                .narrow(0, block * trunk_inner, trunk_inner)?;
            let tower_output =
                self.value_tower_output
                    .narrow(0, block * hidden_size, hidden_size)?;
            let tower = value_hidden
                .matmul(&tower_hidden.t()?)?
                .broadcast_add(&tower_bias)?
                .silu()?;
            let residual = tower.matmul(&tower_output.t()?)?;
            value_hidden = (value_hidden + residual)?;
            let tower_rms = value_hidden
                .sqr()?
                .mean_keepdim(1)?
                .affine(1.0, RMS_NORM_EPS)?
                .sqrt()?;
            value_hidden = value_hidden.broadcast_div(&tower_rms)?;
        }

        let value_head = value_hidden
            .matmul(&self.value_head_hidden.t()?)?
            .broadcast_add(&self.value_head_bias)?
            .relu()?;
        let value_head = value_head
            .matmul(&self.value_head_hidden2.t()?)?
            .broadcast_add(&self.value_head_bias2)?
            .relu()?;
        let value_logits = value_head.matmul(&self.value_head_output.t()?)?;
        let moves_left_head = value_hidden
            .matmul(&self.moves_left_hidden.t()?)?
            .broadcast_add(&self.moves_left_bias_hidden)?
            .relu()?;
        let moves_left_logits = moves_left_head
            .matmul(&self.moves_left_output.reshape((VALUE_HEAD_SIZE, 1))?)?
            .broadcast_add(&self.moves_left_bias)?;
        let policy_bias = self.policy_move_bias.reshape((1, DENSE_MOVE_SPACE))?;
        let policy_from_scores = policy_hidden.matmul(&self.policy_from_hidden.t()?)?;
        let policy_to_scores = policy_hidden.matmul(&self.policy_to_hidden.t()?)?;
        let policy_from_logits = policy_from_scores.matmul(&self.policy_move_from_features.t()?)?;
        let policy_to_logits = policy_to_scores.matmul(&self.policy_move_to_features.t()?)?;
        let policy_pair_context = policy_hidden
            .matmul(&self.policy_pair_context_hidden.t()?)?
            .broadcast_add(&self.policy_pair_context_bias)?
            .relu()?;
        let policy_pair_logits = policy_pair_context.matmul(&self.policy_pair_embedding.t()?)?;
        let policy_move_context = policy_hidden.matmul(&self.policy_move_context_hidden.t()?)?;
        let policy_move_logits = policy_move_context.matmul(&self.policy_move_embedding.t()?)?;
        let token_from_to_logits = square_tokens
            .flatten_to(1)?
            .matmul(&self.policy_from_token_to_hidden.t()?)?
            .reshape((bsz, BOARD_SIZE * BOARD_SIZE))?
            .index_select(&self.policy_move_sparse_indices, 1)?;
        let token_to_from_logits = square_tokens
            .flatten_to(1)?
            .matmul(&self.policy_to_token_from_hidden.t()?)?
            .reshape((bsz, BOARD_SIZE * BOARD_SIZE))?
            .index_select(&self.policy_move_transposed_sparse_indices, 1)?;
        let policy_logits = ((((((policy_from_logits + policy_to_logits)?
            + policy_pair_logits)?
            + policy_move_logits)?
            + token_from_to_logits)?
            + token_to_from_logits)?
            .broadcast_add(&policy_bias))?;

        Ok(ForwardOutput {
            value_logits,
            policy_logits,
            moves_left_logits,
        })
    }
}

pub(super) struct ForwardOutput {
    pub(super) value_logits: Tensor,
    pub(super) policy_logits: Tensor,
    pub(super) moves_left_logits: Tensor,
}

pub(super) struct BatchTensors {
    pub(super) batch_size: usize,
    pub(super) max_features: usize,
    pub(super) feature_indices: Tensor,
    pub(super) feature_mask: Tensor,
    pub(super) structural_piece_indices: Tensor,
    pub(super) structural_rank_indices: Tensor,
    pub(super) structural_file_indices: Tensor,
    pub(super) structural_us_king_piece_indices: Tensor,
    pub(super) structural_them_king_piece_indices: Tensor,
    pub(super) structural_mask: Tensor,
    pub(super) square_token_feature_indices: Tensor,
    pub(super) square_token_mask: Tensor,
    pub(super) policy_indices: Tensor,
    pub(super) policy_targets: Tensor,
    pub(super) policy_mask: Tensor,
    pub(super) value_wdl: Tensor,
    pub(super) values: Tensor,
    pub(super) moves_left: Tensor,
    pub(super) policy_weights: Tensor,
    pub(super) value_weights: Tensor,
}

impl BatchTensors {
    pub(super) fn from_packed(packed: PackedBatch, device: &Device) -> CandleResult<Self> {
        let batch_size = packed.batch_size;
        let max_features = packed.max_features;
        let max_policy_moves = packed.max_policy_moves;
        Ok(Self {
            batch_size,
            max_features,
            feature_indices: Tensor::from_vec(
                packed.feature_indices,
                (batch_size, max_features),
                device,
            )?,
            feature_mask: Tensor::from_vec(
                packed.feature_mask,
                (batch_size, max_features, 1),
                device,
            )?,
            structural_piece_indices: Tensor::from_vec(
                packed.structural_piece_indices,
                (batch_size, max_features),
                device,
            )?,
            structural_rank_indices: Tensor::from_vec(
                packed.structural_rank_indices,
                (batch_size, max_features),
                device,
            )?,
            structural_file_indices: Tensor::from_vec(
                packed.structural_file_indices,
                (batch_size, max_features),
                device,
            )?,
            structural_us_king_piece_indices: Tensor::from_vec(
                packed.structural_us_king_piece_indices,
                (batch_size, max_features),
                device,
            )?,
            structural_them_king_piece_indices: Tensor::from_vec(
                packed.structural_them_king_piece_indices,
                (batch_size, max_features),
                device,
            )?,
            structural_mask: Tensor::from_vec(
                packed.structural_mask,
                (batch_size, max_features, 1),
                device,
            )?,
            square_token_feature_indices: Tensor::from_vec(
                packed.square_token_feature_indices,
                (batch_size, BOARD_SIZE),
                device,
            )?,
            square_token_mask: Tensor::from_vec(
                packed.square_token_mask,
                (batch_size, BOARD_SIZE, 1),
                device,
            )?,
            policy_indices: Tensor::from_vec(
                packed.policy_indices,
                (batch_size, max_policy_moves),
                device,
            )?,
            policy_targets: Tensor::from_vec(
                packed.policy_targets,
                (batch_size, max_policy_moves),
                device,
            )?,
            policy_mask: Tensor::from_vec(
                packed.policy_mask,
                (batch_size, max_policy_moves),
                device,
            )?,
            value_wdl: Tensor::from_vec(packed.value_wdl, (batch_size, WDL_HEAD_SIZE), device)?,
            values: Tensor::from_vec(packed.values, batch_size, device)?,
            moves_left: Tensor::from_vec(packed.moves_left, batch_size, device)?,
            policy_weights: Tensor::from_vec(packed.policy_weights, batch_size, device)?,
            value_weights: Tensor::from_vec(packed.value_weights, batch_size, device)?,
        })
    }
}

impl AzCandleModel {
    pub(super) fn from_model(model: &AzNnue, device: &Device) -> CandleResult<Self> {
        let arch = model.arch;
        let hidden = arch.hidden_size;
        Ok(Self {
            arch,
            input_hidden: var_from_slice(
                &model.input_hidden,
                (AZ_NNUE_INPUT_SIZE, hidden),
                device,
            )?,
            input_piece_hidden: var_from_slice(
                &model.input_piece_hidden,
                (STRUCTURAL_PIECE_SIZE, hidden),
                device,
            )?,
            input_rank_hidden: var_from_slice(
                &model.input_rank_hidden,
                (STRUCTURAL_RANK_SIZE, hidden),
                device,
            )?,
            input_file_hidden: var_from_slice(
                &model.input_file_hidden,
                (STRUCTURAL_FILE_SIZE, hidden),
                device,
            )?,
            input_king_piece_hidden: var_from_slice(
                &model.input_king_piece_hidden,
                (STRUCTURAL_KING_PIECE_SIZE, hidden),
                device,
            )?,
            hidden_bias: var_from_slice(&model.hidden_bias, hidden, device)?,
            piece_attention_query: var_from_slice(
                &model.piece_attention_query,
                (PIECE_ATTENTION_HEADS, hidden),
                device,
            )?,
            piece_attention_value: var_from_slice(
                &model.piece_attention_value,
                (PIECE_ATTENTION_TOTAL_SIZE, hidden),
                device,
            )?,
            piece_attention_output: var_from_slice(
                &model.piece_attention_output,
                (hidden, PIECE_ATTENTION_TOTAL_SIZE),
                device,
            )?,
            policy_tower_hidden: var_from_slice(
                &model.policy_tower_hidden,
                (POLICY_TOWER_BLOCKS * TRUNK_INNER_MULT * hidden, hidden),
                device,
            )?,
            policy_tower_bias: var_from_slice(
                &model.policy_tower_bias,
                POLICY_TOWER_BLOCKS * TRUNK_INNER_MULT * hidden,
                device,
            )?,
            policy_tower_output: var_from_slice(
                &model.policy_tower_output,
                (POLICY_TOWER_BLOCKS * hidden, TRUNK_INNER_MULT * hidden),
                device,
            )?,
            value_tower_hidden: var_from_slice(
                &model.value_tower_hidden,
                (VALUE_TOWER_BLOCKS * TRUNK_INNER_MULT * hidden, hidden),
                device,
            )?,
            value_tower_bias: var_from_slice(
                &model.value_tower_bias,
                VALUE_TOWER_BLOCKS * TRUNK_INNER_MULT * hidden,
                device,
            )?,
            value_tower_output: var_from_slice(
                &model.value_tower_output,
                (VALUE_TOWER_BLOCKS * hidden, TRUNK_INNER_MULT * hidden),
                device,
            )?,
            value_head_hidden: var_from_slice(
                &model.value_head_hidden,
                (VALUE_HEAD_SIZE, hidden),
                device,
            )?,
            value_head_bias: var_from_slice(&model.value_head_bias, VALUE_HEAD_SIZE, device)?,
            value_head_hidden2: var_from_slice(
                &model.value_head_hidden2,
                (VALUE_HEAD_SIZE, VALUE_HEAD_SIZE),
                device,
            )?,
            value_head_bias2: var_from_slice(&model.value_head_bias2, VALUE_HEAD_SIZE, device)?,
            value_head_output: var_from_slice(
                &model.value_head_output,
                (WDL_HEAD_SIZE, VALUE_HEAD_SIZE),
                device,
            )?,
            moves_left_hidden: var_from_slice(
                &model.moves_left_hidden,
                (VALUE_HEAD_SIZE, hidden),
                device,
            )?,
            moves_left_bias_hidden: var_from_slice(
                &model.moves_left_bias_hidden,
                VALUE_HEAD_SIZE,
                device,
            )?,
            moves_left_output: var_from_slice(&model.moves_left_output, VALUE_HEAD_SIZE, device)?,
            moves_left_bias: var_from_slice(&model.moves_left_bias, 1, device)?,
            policy_move_bias: var_from_slice(&model.policy_move_bias, DENSE_MOVE_SPACE, device)?,
            policy_from_hidden: var_from_slice(
                &model.policy_from_hidden,
                (BOARD_SIZE, hidden),
                device,
            )?,
            policy_to_hidden: var_from_slice(
                &model.policy_to_hidden,
                (BOARD_SIZE, hidden),
                device,
            )?,
            policy_pair_context_hidden: var_from_slice(
                &model.policy_pair_context_hidden,
                (POLICY_PAIR_CONTEXT_SIZE, hidden),
                device,
            )?,
            policy_pair_context_bias: var_from_slice(
                &model.policy_pair_context_bias,
                POLICY_PAIR_CONTEXT_SIZE,
                device,
            )?,
            policy_pair_embedding: var_from_slice(
                &model.policy_pair_embedding,
                (DENSE_MOVE_SPACE, POLICY_PAIR_CONTEXT_SIZE),
                device,
            )?,
            policy_move_context_hidden: var_from_slice(
                &model.policy_move_context_hidden,
                (POLICY_MOVE_EMBED_SIZE, hidden),
                device,
            )?,
            policy_move_embedding: var_from_slice(
                &model.policy_move_embedding,
                (DENSE_MOVE_SPACE, POLICY_MOVE_EMBED_SIZE),
                device,
            )?,
            policy_from_token_to_hidden: var_from_slice(
                &model.policy_from_token_to_hidden,
                (BOARD_SIZE, hidden),
                device,
            )?,
            policy_to_token_from_hidden: var_from_slice(
                &model.policy_to_token_from_hidden,
                (BOARD_SIZE, hidden),
                device,
            )?,
            policy_move_from_features: Tensor::from_vec(
                policy_move_from_features().to_vec(),
                (DENSE_MOVE_SPACE, BOARD_SIZE),
                device,
            )?,
            policy_move_to_features: Tensor::from_vec(
                policy_move_to_features().to_vec(),
                (DENSE_MOVE_SPACE, BOARD_SIZE),
                device,
            )?,
            policy_move_sparse_indices: Tensor::from_vec(
                policy_move_sparse_indices().to_vec(),
                DENSE_MOVE_SPACE,
                device,
            )?,
            policy_move_transposed_sparse_indices: Tensor::from_vec(
                policy_move_transposed_sparse_indices().to_vec(),
                DENSE_MOVE_SPACE,
                device,
            )?,
        })
    }

    pub(super) fn all_vars(&self) -> Vec<Var> {
        let mut vars = Vec::new();
        vars.push(self.input_hidden.clone());
        vars.push(self.input_piece_hidden.clone());
        vars.push(self.input_rank_hidden.clone());
        vars.push(self.input_file_hidden.clone());
        vars.push(self.input_king_piece_hidden.clone());
        vars.push(self.hidden_bias.clone());
        vars.push(self.piece_attention_query.clone());
        vars.push(self.piece_attention_value.clone());
        vars.push(self.piece_attention_output.clone());
        vars.push(self.policy_tower_hidden.clone());
        vars.push(self.policy_tower_bias.clone());
        vars.push(self.policy_tower_output.clone());
        vars.push(self.value_tower_hidden.clone());
        vars.push(self.value_tower_bias.clone());
        vars.push(self.value_tower_output.clone());
        vars.push(self.value_head_hidden.clone());
        vars.push(self.value_head_bias.clone());
        vars.push(self.value_head_hidden2.clone());
        vars.push(self.value_head_bias2.clone());
        vars.push(self.value_head_output.clone());
        vars.push(self.moves_left_hidden.clone());
        vars.push(self.moves_left_bias_hidden.clone());
        vars.push(self.moves_left_output.clone());
        vars.push(self.moves_left_bias.clone());
        vars.push(self.policy_move_bias.clone());
        vars.push(self.policy_from_hidden.clone());
        vars.push(self.policy_to_hidden.clone());
        vars.push(self.policy_pair_context_hidden.clone());
        vars.push(self.policy_pair_context_bias.clone());
        vars.push(self.policy_pair_embedding.clone());
        vars.push(self.policy_move_context_hidden.clone());
        vars.push(self.policy_move_embedding.clone());
        vars.push(self.policy_from_token_to_hidden.clone());
        vars.push(self.policy_to_token_from_hidden.clone());
        vars
    }

    pub(super) fn copy_to_model(&self, model: &mut AzNnue) -> CandleResult<()> {
        copy_var(&self.input_hidden, &mut model.input_hidden)?;
        copy_var(&self.input_piece_hidden, &mut model.input_piece_hidden)?;
        copy_var(&self.input_rank_hidden, &mut model.input_rank_hidden)?;
        copy_var(&self.input_file_hidden, &mut model.input_file_hidden)?;
        copy_var(
            &self.input_king_piece_hidden,
            &mut model.input_king_piece_hidden,
        )?;
        copy_var(&self.hidden_bias, &mut model.hidden_bias)?;
        copy_var(
            &self.piece_attention_query,
            &mut model.piece_attention_query,
        )?;
        copy_var(
            &self.piece_attention_value,
            &mut model.piece_attention_value,
        )?;
        copy_var(
            &self.piece_attention_output,
            &mut model.piece_attention_output,
        )?;
        copy_var(&self.policy_tower_hidden, &mut model.policy_tower_hidden)?;
        copy_var(&self.policy_tower_bias, &mut model.policy_tower_bias)?;
        copy_var(&self.policy_tower_output, &mut model.policy_tower_output)?;
        copy_var(&self.value_tower_hidden, &mut model.value_tower_hidden)?;
        copy_var(&self.value_tower_bias, &mut model.value_tower_bias)?;
        copy_var(&self.value_tower_output, &mut model.value_tower_output)?;
        copy_var(&self.value_head_hidden, &mut model.value_head_hidden)?;
        copy_var(&self.value_head_bias, &mut model.value_head_bias)?;
        copy_var(&self.value_head_hidden2, &mut model.value_head_hidden2)?;
        copy_var(&self.value_head_bias2, &mut model.value_head_bias2)?;
        copy_var(&self.value_head_output, &mut model.value_head_output)?;
        copy_var(&self.moves_left_hidden, &mut model.moves_left_hidden)?;
        copy_var(
            &self.moves_left_bias_hidden,
            &mut model.moves_left_bias_hidden,
        )?;
        copy_var(&self.moves_left_output, &mut model.moves_left_output)?;
        copy_var(&self.moves_left_bias, &mut model.moves_left_bias)?;
        copy_var(&self.policy_move_bias, &mut model.policy_move_bias)?;
        copy_var(&self.policy_from_hidden, &mut model.policy_from_hidden)?;
        copy_var(&self.policy_to_hidden, &mut model.policy_to_hidden)?;
        copy_var(
            &self.policy_pair_context_hidden,
            &mut model.policy_pair_context_hidden,
        )?;
        copy_var(
            &self.policy_pair_context_bias,
            &mut model.policy_pair_context_bias,
        )?;
        copy_var(
            &self.policy_pair_embedding,
            &mut model.policy_pair_embedding,
        )?;
        copy_var(
            &self.policy_move_context_hidden,
            &mut model.policy_move_context_hidden,
        )?;
        copy_var(
            &self.policy_move_embedding,
            &mut model.policy_move_embedding,
        )?;
        copy_var(
            &self.policy_from_token_to_hidden,
            &mut model.policy_from_token_to_hidden,
        )?;
        copy_var(
            &self.policy_to_token_from_hidden,
            &mut model.policy_to_token_from_hidden,
        )?;
        model.refresh_policy_derived_caches();
        Ok(())
    }

    pub(super) fn cpu_grads(&self, grads: &GradStore) -> CandleResult<Vec<Option<Vec<f32>>>> {
        let mut out = Vec::new();
        for var in self.all_vars() {
            let grad = grads
                .get(&var)
                .map(|grad| grad.flatten_all()?.to_vec1::<f32>())
                .transpose()?;
            out.push(grad);
        }
        Ok(out)
    }

    pub(super) fn to_cpu_values(&self) -> CandleResult<Vec<Vec<f32>>> {
        let mut values = Vec::new();
        for var in self.all_vars() {
            values.push(var.as_detached_tensor().flatten_all()?.to_vec1::<f32>()?);
        }
        Ok(values)
    }

    pub(super) fn set_from_cpu_values(&self, values: &[Vec<f32>]) -> CandleResult<()> {
        for (var, values) in self.all_vars().iter().zip(values.iter()) {
            let tensor = Tensor::from_vec(values.clone(), var.shape().clone(), var.device())?;
            var.set(&tensor)?;
        }
        Ok(())
    }
}

fn var_from_slice<S: Into<candle_core::Shape>>(
    values: &[f32],
    shape: S,
    device: &Device,
) -> CandleResult<Var> {
    Var::from_slice(values, shape, device)
}

fn copy_var(var: &Var, dst: &mut [f32]) -> CandleResult<()> {
    let values = var.as_detached_tensor().flatten_all()?.to_vec1::<f32>()?;
    dst.copy_from_slice(&values);
    Ok(())
}
