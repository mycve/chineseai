use candle_core::{Device, Result as CandleResult, Tensor, Var, backprop::GradStore};
use candle_nn::ops::log_softmax;

use super::{
    AUTO_FEATURE_SIZE, AzNnue, AzNnueArch, AzTrainLossWeights, AzTrainingSample, DENSE_MOVE_SPACE,
    PIECE_ATTENTION_SIZE, POLICY_MOVE_EMBED_SIZE, POLICY_PAIR_CONTEXT_SIZE, STRUCTURAL_FILE_SIZE,
    STRUCTURAL_KING_PIECE_SIZE, STRUCTURAL_PIECE_SIZE, STRUCTURAL_RANK_SIZE, TRUNK_LAYERS,
    VALUE_HEAD_SIZE, dataloader::PackedBatch, policy_move_from_features, policy_move_to_features,
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
    input_quadratic_scale: Var,
    piece_attention_query: Var,
    piece_attention_value: Var,
    piece_attention_output: Var,
    trunk_residual_hidden: Var,
    trunk_residual_bias: Var,
    auto_feature_hidden: Var,
    auto_feature_bias: Var,
    auto_feature_output: Var,
    value_head_hidden: Var,
    value_head_bias: Var,
    value_head_output: Var,
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
    policy_move_from_features: Tensor,
    policy_move_to_features: Tensor,
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
        let sparse_pre = feature_embeddings
            .broadcast_mul(&batch.feature_mask)?
            .sum(1)?
            .broadcast_add(&self.hidden_bias)?;
        let attention_scores = feature_embeddings
            .broadcast_mul(&self.piece_attention_query.reshape((1, 1, hidden_size))?)?
            .sum(2)?;
        let attention_mask = batch.feature_mask.squeeze(2)?.affine(1.0e9, -1.0e9)?;
        let attention_weights = log_softmax(&(attention_scores + attention_mask)?, 1)?
            .exp()?
            .unsqueeze(2)?;
        let attention_values = feature_embeddings
            .flatten_to(1)?
            .matmul(&self.piece_attention_value.t()?)?
            .reshape((bsz, batch.max_features, PIECE_ATTENTION_SIZE))?;
        let attention_context = attention_values.broadcast_mul(&attention_weights)?.sum(1)?;
        let attention_residual = attention_context.matmul(&self.piece_attention_output.t()?)?;
        let sparse_pre = (sparse_pre + attention_residual)?;
        let quadratic = sparse_pre
            .sqr()?
            .broadcast_mul(&self.input_quadratic_scale.reshape((1, hidden_size))?)?;
        let mut sparse_hidden = (sparse_pre + quadratic)?.relu()?;
        for layer in 0..TRUNK_LAYERS {
            let weights = self.trunk_residual_hidden.narrow(0, layer, 1)?.squeeze(0)?;
            let bias = self.trunk_residual_bias.narrow(0, layer, 1)?.squeeze(0)?;
            let residual = sparse_hidden.matmul(&weights.t()?)?.broadcast_add(&bias)?;
            sparse_hidden = (sparse_hidden + residual)?.relu()?;
        }
        let auto_features = sparse_hidden
            .matmul(&self.auto_feature_hidden.t()?)?
            .broadcast_add(&self.auto_feature_bias)?
            .relu()?;
        let auto_residual = auto_features.matmul(&self.auto_feature_output.t()?)?;
        let sparse_hidden = (sparse_hidden + auto_residual)?.relu()?;
        let rms = sparse_hidden
            .sqr()?
            .mean_keepdim(1)?
            .affine(1.0, RMS_NORM_EPS)?
            .sqrt()?;
        let hidden = sparse_hidden.broadcast_div(&rms)?;

        let value_head = hidden
            .matmul(&self.value_head_hidden.t()?)?
            .broadcast_add(&self.value_head_bias)?
            .relu()?;
        let value_head_output =
            value_head.matmul(&self.value_head_output.reshape((VALUE_HEAD_SIZE, 1))?)?;
        let moves_left_logits = value_head
            .matmul(&self.moves_left_output.reshape((VALUE_HEAD_SIZE, 1))?)?
            .broadcast_add(&self.moves_left_bias)?;
        let values = value_head_output.tanh()?;
        let policy_bias = self.policy_move_bias.reshape((1, DENSE_MOVE_SPACE))?;
        let policy_from_scores = hidden.matmul(&self.policy_from_hidden.t()?)?;
        let policy_to_scores = hidden.matmul(&self.policy_to_hidden.t()?)?;
        let policy_from_logits = policy_from_scores.matmul(&self.policy_move_from_features.t()?)?;
        let policy_to_logits = policy_to_scores.matmul(&self.policy_move_to_features.t()?)?;
        let policy_pair_context = hidden
            .matmul(&self.policy_pair_context_hidden.t()?)?
            .broadcast_add(&self.policy_pair_context_bias)?
            .relu()?;
        let policy_pair_logits = policy_pair_context.matmul(&self.policy_pair_embedding.t()?)?;
        let policy_move_context = hidden.matmul(&self.policy_move_context_hidden.t()?)?;
        let policy_move_logits = policy_move_context.matmul(&self.policy_move_embedding.t()?)?;
        let policy_logits = ((((policy_from_logits + policy_to_logits)? + policy_pair_logits)?
            + policy_move_logits)?
            .broadcast_add(&policy_bias))?;

        Ok(ForwardOutput {
            values,
            policy_logits,
            moves_left_logits,
        })
    }
}

pub(super) struct ForwardOutput {
    pub(super) values: Tensor,
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
    pub(super) policy_indices: Tensor,
    pub(super) policy_targets: Tensor,
    pub(super) policy_mask: Tensor,
    pub(super) values: Tensor,
    pub(super) moves_left: Tensor,
}

impl BatchTensors {
    pub(super) fn new(
        samples: &[AzTrainingSample],
        batch: &[usize],
        device: &Device,
    ) -> CandleResult<Self> {
        Self::from_packed(PackedBatch::from_indices(samples, batch), device)
    }

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
            values: Tensor::from_vec(packed.values, batch_size, device)?,
            moves_left: Tensor::from_vec(packed.moves_left, batch_size, device)?,
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
            input_quadratic_scale: var_from_slice(&model.input_quadratic_scale, hidden, device)?,
            piece_attention_query: var_from_slice(&model.piece_attention_query, hidden, device)?,
            piece_attention_value: var_from_slice(
                &model.piece_attention_value,
                (PIECE_ATTENTION_SIZE, hidden),
                device,
            )?,
            piece_attention_output: var_from_slice(
                &model.piece_attention_output,
                (hidden, PIECE_ATTENTION_SIZE),
                device,
            )?,
            trunk_residual_hidden: var_from_slice(
                &model.trunk_residual_hidden,
                (TRUNK_LAYERS, hidden, hidden),
                device,
            )?,
            trunk_residual_bias: var_from_slice(
                &model.trunk_residual_bias,
                (TRUNK_LAYERS, hidden),
                device,
            )?,
            auto_feature_hidden: var_from_slice(
                &model.auto_feature_hidden,
                (AUTO_FEATURE_SIZE, hidden),
                device,
            )?,
            auto_feature_bias: var_from_slice(&model.auto_feature_bias, AUTO_FEATURE_SIZE, device)?,
            auto_feature_output: var_from_slice(
                &model.auto_feature_output,
                (hidden, AUTO_FEATURE_SIZE),
                device,
            )?,
            value_head_hidden: var_from_slice(
                &model.value_head_hidden,
                (VALUE_HEAD_SIZE, hidden),
                device,
            )?,
            value_head_bias: var_from_slice(&model.value_head_bias, VALUE_HEAD_SIZE, device)?,
            value_head_output: var_from_slice(&model.value_head_output, VALUE_HEAD_SIZE, device)?,
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
        vars.push(self.input_quadratic_scale.clone());
        vars.push(self.piece_attention_query.clone());
        vars.push(self.piece_attention_value.clone());
        vars.push(self.piece_attention_output.clone());
        vars.push(self.trunk_residual_hidden.clone());
        vars.push(self.trunk_residual_bias.clone());
        vars.push(self.auto_feature_hidden.clone());
        vars.push(self.auto_feature_bias.clone());
        vars.push(self.auto_feature_output.clone());
        vars.push(self.value_head_hidden.clone());
        vars.push(self.value_head_bias.clone());
        vars.push(self.value_head_output.clone());
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
        vars
    }

    fn value_head_vars(&self) -> Vec<Var> {
        vec![
            self.value_head_hidden.clone(),
            self.value_head_bias.clone(),
            self.value_head_output.clone(),
            self.moves_left_output.clone(),
            self.moves_left_bias.clone(),
        ]
    }

    fn policy_head_vars(&self) -> Vec<Var> {
        vec![
            self.policy_move_bias.clone(),
            self.policy_from_hidden.clone(),
            self.policy_to_hidden.clone(),
            self.policy_pair_context_hidden.clone(),
            self.policy_pair_context_bias.clone(),
            self.policy_pair_embedding.clone(),
            self.policy_move_context_hidden.clone(),
            self.policy_move_embedding.clone(),
        ]
    }

    pub(super) fn remove_frozen_grads(
        &self,
        grads: &mut GradStore,
        loss_weights: AzTrainLossWeights,
    ) {
        self.mask_grads(
            grads,
            loss_weights.train_trunk,
            loss_weights.train_value_head,
            loss_weights.train_policy_head,
        );
    }

    fn mask_grads(
        &self,
        grads: &mut GradStore,
        train_trunk: bool,
        train_value_head: bool,
        train_policy_head: bool,
    ) {
        if !train_trunk {
            grads.remove(&self.input_hidden);
            grads.remove(&self.input_piece_hidden);
            grads.remove(&self.input_rank_hidden);
            grads.remove(&self.input_file_hidden);
            grads.remove(&self.input_king_piece_hidden);
            grads.remove(&self.hidden_bias);
            grads.remove(&self.input_quadratic_scale);
            grads.remove(&self.piece_attention_query);
            grads.remove(&self.piece_attention_value);
            grads.remove(&self.piece_attention_output);
            grads.remove(&self.trunk_residual_hidden);
            grads.remove(&self.trunk_residual_bias);
            grads.remove(&self.auto_feature_hidden);
            grads.remove(&self.auto_feature_bias);
            grads.remove(&self.auto_feature_output);
        }
        if !train_value_head {
            for var in self.value_head_vars() {
                grads.remove(&var);
            }
        }
        if !train_policy_head {
            for var in self.policy_head_vars() {
                grads.remove(&var);
            }
        }
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
            &self.input_quadratic_scale,
            &mut model.input_quadratic_scale,
        )?;
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
        copy_var(
            &self.trunk_residual_hidden,
            &mut model.trunk_residual_hidden,
        )?;
        copy_var(&self.trunk_residual_bias, &mut model.trunk_residual_bias)?;
        copy_var(&self.auto_feature_hidden, &mut model.auto_feature_hidden)?;
        copy_var(&self.auto_feature_bias, &mut model.auto_feature_bias)?;
        copy_var(&self.auto_feature_output, &mut model.auto_feature_output)?;
        copy_var(&self.value_head_hidden, &mut model.value_head_hidden)?;
        copy_var(&self.value_head_bias, &mut model.value_head_bias)?;
        copy_var(&self.value_head_output, &mut model.value_head_output)?;
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
