use candle_core::{Device, Result as CandleResult, Tensor, Var, backprop::GradStore};
use candle_nn::ops::log_softmax;
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use std::{process::Command, thread};

use super::{
    AUTO_FEATURE_SIZE, AzNnue, AzNnueArch, AzTrainLossWeights, AzTrainStats, AzTrainingSample,
    DENSE_MOVE_SPACE, LINE_COUNT, LINE_MIXER_RANK, MOVES_LEFT_AUX_WEIGHT, PIECE_ATTENTION_SIZE,
    POLICY_MOVE_EMBED_SIZE, POLICY_PAIR_CONTEXT_SIZE, SQUARE_TOKEN_PIECES, STRUCTURAL_FILE_SIZE,
    STRUCTURAL_KING_PIECE_SIZE, STRUCTURAL_PIECE_SIZE, STRUCTURAL_RANK_SIZE, TRUNK_LAYERS,
    VALUE_HEAD_SIZE, canonical_general_buckets_from_features, decode_current_piece_square_feature,
    policy_move_from_features, policy_move_to_features, structural_king_piece_index,
};
use crate::nnue::AZ_NNUE_INPUT_SIZE;
use crate::xiangqi::{BOARD_FILES, BOARD_RANKS, BOARD_SIZE};

const POLICY_MASK_VALUE: f32 = -1.0e9;
const ADAMW_WEIGHT_DECAY: f64 = 1e-4;
const RMS_NORM_EPS: f64 = 1.0e-6;

#[derive(Debug)]
pub(super) struct GpuTrainer {
    arch: AzNnueArch,
    device_indices: Vec<usize>,
    replicas: Vec<GpuReplica>,
    optimizers: Vec<AdamW>,
}

#[derive(Debug)]
struct GpuReplica {
    device: Device,
    arch: AzNnueArch,
    vars: AzCandleModel,
}

#[derive(Debug)]
struct AzCandleModel {
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
    square_piece_token: Var,
    square_position_token: Var,
    line_mixer_down: Var,
    line_mixer_bias: Var,
    line_mixer_up: Var,
    line_context_bias: Var,
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
    line_square_mask: Tensor,
    policy_move_from_features: Tensor,
    policy_move_to_features: Tensor,
}

pub(crate) fn training_cuda_device_count() -> usize {
    cuda_device_indices().len().max(1)
}

pub(super) fn train_samples_gpu(
    model: &mut AzNnue,
    samples: &[AzTrainingSample],
    epochs: usize,
    lr: f32,
    per_gpu_batch_size: usize,
    rng: &mut super::SplitMix64,
    loss_weights: AzTrainLossWeights,
) -> CandleResult<AzTrainStats> {
    if samples.is_empty() || epochs == 0 || lr <= 0.0 {
        return Ok(AzTrainStats::default());
    }

    if model
        .gpu_trainer
        .as_ref()
        .is_none_or(|trainer| !trainer.matches(model))
    {
        model.gpu_trainer = Some(Box::new(GpuTrainer::new(model, lr)?));
    }
    let mut order = (0..samples.len()).collect::<Vec<_>>();
    let mut stats = AzTrainStats::default();
    {
        let trainer = model
            .gpu_trainer
            .as_mut()
            .expect("gpu trainer was initialized");
        let per_gpu = per_gpu_batch_size.max(1);
        let step_chunk = (per_gpu * trainer.replicas.len().max(1)).max(1);
        trainer.set_learning_rate(lr);
        for _ in 0..epochs {
            shuffle(&mut order, rng);
            stats = AzTrainStats::default();
            for batch in order.chunks(step_chunk) {
                let batch_stats = trainer.train_batch(samples, batch, loss_weights)?;
                stats.add_assign(&batch_stats);
            }
        }
    }
    if stats.samples > 0 {
        let denom = stats.samples as f32;
        stats.loss /= denom;
        stats.value_loss /= denom;
        stats.policy_ce /= denom;
    }
    let trainer = model
        .gpu_trainer
        .take()
        .expect("gpu trainer was initialized");
    trainer.copy_to_model(model)?;
    model.gpu_trainer = Some(trainer);
    Ok(stats)
}

impl GpuTrainer {
    fn new(model: &AzNnue, lr: f32) -> CandleResult<Self> {
        let device_indices = cuda_device_indices();
        let mut replicas = Vec::with_capacity(device_indices.len());
        for &device_index in &device_indices {
            replicas.push(GpuReplica::new(model, device_index)?);
        }
        let mut optimizers = Vec::with_capacity(replicas.len());
        for replica in &replicas {
            optimizers.push(AdamW::new(
                replica.vars.all_vars(),
                ParamsAdamW {
                    lr: lr as f64,
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: 1e-8,
                    weight_decay: ADAMW_WEIGHT_DECAY,
                },
            )?);
        }
        if device_indices.len() > 1 {
            eprintln!(
                "[chineseai] 多卡 CUDA 训练: 设备下标={:?}；同步数据并行，每个 step 聚合所有 shard 梯度。",
                device_indices
            );
        }
        Ok(Self {
            arch: model.arch,
            device_indices,
            replicas,
            optimizers,
        })
    }

    fn matches(&self, model: &AzNnue) -> bool {
        self.arch == model.arch && self.device_indices == cuda_device_indices()
    }

    fn set_learning_rate(&mut self, lr: f32) {
        for optimizer in &mut self.optimizers {
            optimizer.set_learning_rate(lr as f64);
        }
    }

    fn train_batch(
        &mut self,
        samples: &[AzTrainingSample],
        batch: &[usize],
        loss_weights: AzTrainLossWeights,
    ) -> CandleResult<AzTrainStats> {
        let active_replicas = self.active_replica_count(batch.len());
        if active_replicas <= 1 {
            return self.train_batch_single(samples, batch, loss_weights);
        }

        let shard_size = batch.len().div_ceil(active_replicas);
        let mut shard_ranges = Vec::with_capacity(active_replicas);
        for shard_index in 0..active_replicas {
            let start = shard_index * shard_size;
            let end = ((shard_index + 1) * shard_size).min(batch.len());
            if start < end {
                shard_ranges.push(start..end);
            }
        }
        let shard_outputs = thread::scope(|scope| {
            let mut handles = Vec::new();
            for (shard_index, range) in shard_ranges.iter().cloned().enumerate() {
                let replica = &self.replicas[shard_index];
                handles.push(scope.spawn(move || {
                    replica.compute_batch_grads(
                        samples,
                        &batch[range],
                        batch.len(),
                        shard_index == 0,
                        loss_weights,
                    )
                }));
            }
            let mut outputs = Vec::with_capacity(handles.len());
            for handle in handles {
                let output = handle
                    .join()
                    .map_err(|_| candle_core::Error::Msg("multi-gpu worker panicked".into()))??;
                outputs.push(output);
            }
            CandleResult::Ok(outputs)
        })?;

        let mut outputs = shard_outputs.into_iter();
        let primary = outputs
            .next()
            .ok_or_else(|| candle_core::Error::Msg("empty multi-gpu shard output".into()))?;
        let mut stats = primary.stats;
        let mut grads = primary
            .grads
            .expect("primary shard gradients should be kept on device");
        for output in outputs {
            stats.add_assign(&output.stats);
            self.add_cpu_grads_to_primary(&mut grads, &output.cpu_grads)?;
        }
        self.optimizers[0].step(&grads)?;
        self.broadcast_primary_to_workers()?;
        Ok(stats)
    }

    fn active_replica_count(&self, batch_len: usize) -> usize {
        let replicas = self.replicas.len();
        if replicas <= 1 {
            return 1;
        }
        replicas.min(batch_len)
    }

    fn train_batch_single(
        &mut self,
        samples: &[AzTrainingSample],
        batch: &[usize],
        loss_weights: AzTrainLossWeights,
    ) -> CandleResult<AzTrainStats> {
        let output = self.replicas[0].compute_batch_grads(
            samples,
            batch,
            batch.len(),
            true,
            loss_weights,
        )?;
        self.optimizers[0].step(
            output
                .grads
                .as_ref()
                .expect("single-gpu gradients should be kept on device"),
        )?;
        Ok(output.stats)
    }

    fn add_cpu_grads_to_primary(
        &self,
        grads: &mut GradStore,
        cpu_grads: &[Option<Vec<f32>>],
    ) -> CandleResult<()> {
        let primary_vars = self.replicas[0].vars.all_vars();
        for (var, cpu_grad) in primary_vars.iter().zip(cpu_grads.iter()) {
            let Some(cpu_grad) = cpu_grad else {
                continue;
            };
            let worker_grad =
                Tensor::from_vec(cpu_grad.clone(), var.shape().clone(), var.device())?;
            let next_grad = if let Some(current) = grads.get(var) {
                (current + worker_grad)?
            } else {
                worker_grad
            };
            grads.insert(var, next_grad);
        }
        Ok(())
    }

    fn broadcast_primary_to_workers(&self) -> CandleResult<()> {
        if self.replicas.len() <= 1 {
            return Ok(());
        }
        let primary_values = self.replicas[0].vars.to_cpu_values()?;
        for replica in self.replicas.iter().skip(1) {
            replica.vars.set_from_cpu_values(&primary_values)?;
        }
        Ok(())
    }

    fn copy_to_model(&self, model: &mut AzNnue) -> CandleResult<()> {
        self.replicas[0].vars.copy_to_model(model)
    }
}

impl GpuReplica {
    fn new(model: &AzNnue, device_index: usize) -> CandleResult<Self> {
        let device = Device::new_cuda(device_index)?;
        let vars = AzCandleModel::from_model(model, &device)?;
        Ok(Self {
            device,
            arch: model.arch,
            vars,
        })
    }

    fn compute_batch_grads(
        &self,
        samples: &[AzTrainingSample],
        batch: &[usize],
        global_batch_len: usize,
        keep_grads: bool,
        loss_weights: AzTrainLossWeights,
    ) -> CandleResult<ShardOutput> {
        let batch_tensors = BatchTensors::new(samples, batch, &self.device)?;
        let output = self.compute_batch_loss(
            &batch_tensors,
            global_batch_len,
            loss_weights.value,
            loss_weights.policy,
        )?;
        let mut grads = output.loss_tensor.backward()?;
        self.vars.remove_frozen_grads(&mut grads, loss_weights);
        let stats = output.stats;
        let cpu_grads = if keep_grads {
            Vec::new()
        } else {
            self.vars.cpu_grads(&grads)?
        };
        Ok(ShardOutput {
            stats,
            grads: if keep_grads { Some(grads) } else { None },
            cpu_grads,
        })
    }

    fn compute_batch_loss(
        &self,
        batch_tensors: &BatchTensors,
        global_batch_len: usize,
        value_weight: f32,
        policy_weight: f32,
    ) -> CandleResult<BatchLossOutput> {
        let forward = self.forward(&batch_tensors)?;
        let value = forward.values.squeeze(1)?;
        let value_error = (&value - &batch_tensors.values)?;
        let value_sse = value_error.sqr()?.sum_all()?;
        let moves_left_pred = forward
            .moves_left_logits
            .tanh()?
            .affine(0.5, 0.5)?
            .squeeze(1)?;
        let moves_left_error = (&moves_left_pred - &batch_tensors.moves_left)?;
        let moves_left_sse = moves_left_error.sqr()?.sum_all()?;

        let legal_policy_logits = forward
            .policy_logits
            .gather(&batch_tensors.policy_indices, 1)?;
        let masked_policy_logits = (&legal_policy_logits + &batch_tensors.policy_mask)?;
        let log_policy = log_softmax(&masked_policy_logits, 1)?;
        let policy_ce_per_sample = ((&batch_tensors.policy_targets * &log_policy)? * -1.0)?;
        let policy_ce_per_sample = policy_ce_per_sample.sum(1)?;
        let policy_ce = policy_ce_per_sample.sum_all()?;
        let weighted_value_loss = (&value_sse * value_weight.max(0.0) as f64)?;
        let weighted_policy_ce = (&policy_ce * policy_weight.max(0.0) as f64)?;
        let weighted_moves_left_loss = (&moves_left_sse * MOVES_LEFT_AUX_WEIGHT as f64)?;
        let loss_tensor = (((weighted_value_loss + weighted_policy_ce)?
            + weighted_moves_left_loss)?
            / global_batch_len as f64)?;

        let value_sse = value_sse.to_scalar::<f32>()?;
        let policy_ce = policy_ce.to_scalar::<f32>()?;
        let stats = AzTrainStats {
            loss: value_sse
                + policy_ce
                + moves_left_sse.to_scalar::<f32>()? * MOVES_LEFT_AUX_WEIGHT,
            value_loss: value_sse,
            policy_ce,
            value_pred_sum: value.sum_all()?.to_scalar::<f32>()?,
            value_pred_sq_sum: value.sqr()?.sum_all()?.to_scalar::<f32>()?,
            value_target_sum: batch_tensors.values.sum_all()?.to_scalar::<f32>()?,
            value_target_sq_sum: batch_tensors.values.sqr()?.sum_all()?.to_scalar::<f32>()?,
            value_pred_target_sum: value
                .broadcast_mul(&batch_tensors.values)?
                .sum_all()?
                .to_scalar::<f32>()?,
            value_error_sq_sum: value_sse,
            samples: batch_tensors.batch_size,
        };
        Ok(BatchLossOutput { loss_tensor, stats })
    }

    fn forward(&self, batch: &BatchTensors) -> CandleResult<ForwardOutput> {
        let bsz = batch.batch_size;
        let hidden_size = self.arch.hidden_size;
        let feature_embeddings = self
            .vars
            .input_hidden
            .index_select(&batch.feature_indices.flatten_all()?, 0)?
            .reshape((bsz, batch.max_features, hidden_size))?;
        let structural_piece = self
            .vars
            .input_piece_hidden
            .index_select(&batch.structural_piece_indices.flatten_all()?, 0)?
            .reshape((bsz, batch.max_features, hidden_size))?;
        let structural_rank = self
            .vars
            .input_rank_hidden
            .index_select(&batch.structural_rank_indices.flatten_all()?, 0)?
            .reshape((bsz, batch.max_features, hidden_size))?;
        let structural_file = self
            .vars
            .input_file_hidden
            .index_select(&batch.structural_file_indices.flatten_all()?, 0)?
            .reshape((bsz, batch.max_features, hidden_size))?;
        let structural_us_king_piece = self
            .vars
            .input_king_piece_hidden
            .index_select(&batch.structural_us_king_piece_indices.flatten_all()?, 0)?
            .reshape((bsz, batch.max_features, hidden_size))?;
        let structural_them_king_piece = self
            .vars
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
            .broadcast_add(&self.vars.hidden_bias)?;
        let attention_scores = feature_embeddings
            .broadcast_mul(
                &self
                    .vars
                    .piece_attention_query
                    .reshape((1, 1, hidden_size))?,
            )?
            .sum(2)?;
        let attention_mask = batch.feature_mask.squeeze(2)?.affine(1.0e9, -1.0e9)?;
        let attention_weights = log_softmax(&(attention_scores + attention_mask)?, 1)?
            .exp()?
            .unsqueeze(2)?;
        let attention_values = feature_embeddings
            .flatten_to(1)?
            .matmul(&self.vars.piece_attention_value.t()?)?
            .reshape((bsz, batch.max_features, PIECE_ATTENTION_SIZE))?;
        let attention_context = attention_values.broadcast_mul(&attention_weights)?.sum(1)?;
        let attention_residual =
            attention_context.matmul(&self.vars.piece_attention_output.t()?)?;
        let sparse_pre = (sparse_pre + attention_residual)?;
        let quadratic = sparse_pre
            .sqr()?
            .broadcast_mul(&self.vars.input_quadratic_scale.reshape((1, hidden_size))?)?;
        let sparse_hidden = (sparse_pre + quadratic)?.relu()?;
        let line_piece_repr = batch
            .line_piece_counts
            .flatten_to(1)?
            .matmul(&self.vars.square_piece_token)?
            .reshape((bsz, LINE_COUNT, hidden_size))?;
        let line_position_repr = self
            .vars
            .line_square_mask
            .matmul(&self.vars.square_position_token)?
            .reshape((1, LINE_COUNT, hidden_size))?
            .broadcast_as((bsz, LINE_COUNT, hidden_size))?;
        let line_repr = (line_piece_repr + line_position_repr)?;
        let line_hidden = line_repr
            .flatten_to(1)?
            .matmul(&self.vars.line_mixer_down.t()?)?
            .broadcast_add(&self.vars.line_mixer_bias)?
            .relu()?
            .reshape((bsz, LINE_COUNT, LINE_MIXER_RANK))?
            .sum(1)?;
        let line_context = line_hidden
            .matmul(&self.vars.line_mixer_up.t()?)?
            .broadcast_add(&self.vars.line_context_bias)?;
        let mut sparse_hidden = (sparse_hidden + line_context)?.relu()?;
        for layer in 0..TRUNK_LAYERS {
            let weights = self
                .vars
                .trunk_residual_hidden
                .narrow(0, layer, 1)?
                .squeeze(0)?;
            let bias = self
                .vars
                .trunk_residual_bias
                .narrow(0, layer, 1)?
                .squeeze(0)?;
            let residual = sparse_hidden.matmul(&weights.t()?)?.broadcast_add(&bias)?;
            sparse_hidden = (sparse_hidden + residual)?.relu()?;
        }
        let auto_features = sparse_hidden
            .matmul(&self.vars.auto_feature_hidden.t()?)?
            .broadcast_add(&self.vars.auto_feature_bias)?
            .relu()?;
        let auto_residual = auto_features.matmul(&self.vars.auto_feature_output.t()?)?;
        let sparse_hidden = (sparse_hidden + auto_residual)?.relu()?;
        let rms = sparse_hidden
            .sqr()?
            .mean_keepdim(1)?
            .affine(1.0, RMS_NORM_EPS)?
            .sqrt()?;
        let hidden = sparse_hidden.broadcast_div(&rms)?;

        let value_head = hidden
            .matmul(&self.vars.value_head_hidden.t()?)?
            .broadcast_add(&self.vars.value_head_bias)?
            .relu()?;
        let value_head_output =
            value_head.matmul(&self.vars.value_head_output.reshape((VALUE_HEAD_SIZE, 1))?)?;
        let moves_left_logits = value_head
            .matmul(&self.vars.moves_left_output.reshape((VALUE_HEAD_SIZE, 1))?)?
            .broadcast_add(&self.vars.moves_left_bias)?;
        let values = value_head_output.tanh()?;
        let policy_bias = self.vars.policy_move_bias.reshape((1, DENSE_MOVE_SPACE))?;
        let policy_from_scores = hidden.matmul(&self.vars.policy_from_hidden.t()?)?;
        let policy_to_scores = hidden.matmul(&self.vars.policy_to_hidden.t()?)?;
        let policy_from_logits =
            policy_from_scores.matmul(&self.vars.policy_move_from_features.t()?)?;
        let policy_to_logits = policy_to_scores.matmul(&self.vars.policy_move_to_features.t()?)?;
        let policy_pair_context = hidden
            .matmul(&self.vars.policy_pair_context_hidden.t()?)?
            .broadcast_add(&self.vars.policy_pair_context_bias)?
            .relu()?;
        let policy_pair_logits =
            policy_pair_context.matmul(&self.vars.policy_pair_embedding.t()?)?;
        let policy_move_context = hidden.matmul(&self.vars.policy_move_context_hidden.t()?)?;
        let policy_move_logits =
            policy_move_context.matmul(&self.vars.policy_move_embedding.t()?)?;
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

struct ShardOutput {
    stats: AzTrainStats,
    grads: Option<GradStore>,
    cpu_grads: Vec<Option<Vec<f32>>>,
}

struct ForwardOutput {
    values: Tensor,
    policy_logits: Tensor,
    moves_left_logits: Tensor,
}

struct BatchLossOutput {
    loss_tensor: Tensor,
    stats: AzTrainStats,
}

struct BatchTensors {
    batch_size: usize,
    max_features: usize,
    feature_indices: Tensor,
    feature_mask: Tensor,
    structural_piece_indices: Tensor,
    structural_rank_indices: Tensor,
    structural_file_indices: Tensor,
    structural_us_king_piece_indices: Tensor,
    structural_them_king_piece_indices: Tensor,
    structural_mask: Tensor,
    line_piece_counts: Tensor,
    policy_indices: Tensor,
    policy_targets: Tensor,
    policy_mask: Tensor,
    values: Tensor,
    moves_left: Tensor,
}

impl BatchTensors {
    fn new(samples: &[AzTrainingSample], batch: &[usize], device: &Device) -> CandleResult<Self> {
        let batch_size = batch.len();
        let max_features = batch
            .iter()
            .map(|&sample_index| samples[sample_index].features.len())
            .max()
            .unwrap_or(0)
            .max(1);
        let max_policy_moves = batch
            .iter()
            .map(|&sample_index| {
                samples[sample_index]
                    .move_indices
                    .iter()
                    .filter(|&&move_index| move_index < DENSE_MOVE_SPACE)
                    .count()
            })
            .max()
            .unwrap_or(0)
            .max(1);
        let mut feature_indices = vec![0u32; batch_size * max_features];
        let mut feature_mask = vec![0.0f32; batch_size * max_features];
        let mut structural_piece_indices = vec![0u32; batch_size * max_features];
        let mut structural_rank_indices = vec![0u32; batch_size * max_features];
        let mut structural_file_indices = vec![0u32; batch_size * max_features];
        let mut structural_us_king_piece_indices = vec![0u32; batch_size * max_features];
        let mut structural_them_king_piece_indices = vec![0u32; batch_size * max_features];
        let mut structural_mask = vec![0.0f32; batch_size * max_features];
        let mut line_piece_counts = vec![0.0f32; batch_size * LINE_COUNT * SQUARE_TOKEN_PIECES];
        let mut policy_indices = vec![0u32; batch_size * max_policy_moves];
        let mut policy_targets = vec![0.0f32; batch_size * max_policy_moves];
        let mut policy_mask = vec![POLICY_MASK_VALUE; batch_size * max_policy_moves];
        let mut values = vec![0.0f32; batch_size];
        let mut moves_left = vec![0.0f32; batch_size];

        for (row, &sample_index) in batch.iter().enumerate() {
            let sample = &samples[sample_index];
            let (us_king_bucket, them_king_bucket) =
                canonical_general_buckets_from_features(&sample.features);
            let feature_base = row * max_features;
            for (feature_offset, &feature) in sample.features.iter().enumerate() {
                if feature < AZ_NNUE_INPUT_SIZE {
                    let batch_feature_index = feature_base + feature_offset;
                    feature_indices[batch_feature_index] = feature as u32;
                    feature_mask[batch_feature_index] = 1.0;
                    if let Some(structural) = decode_current_piece_square_feature(feature) {
                        structural_piece_indices[batch_feature_index] =
                            structural.piece_index as u32;
                        structural_rank_indices[batch_feature_index] = structural.rank as u32;
                        structural_file_indices[batch_feature_index] = structural.file as u32;
                        structural_us_king_piece_indices[batch_feature_index] =
                            structural_king_piece_index(0, us_king_bucket, structural.piece_index)
                                as u32;
                        structural_them_king_piece_indices[batch_feature_index] =
                            structural_king_piece_index(1, them_king_bucket, structural.piece_index)
                                as u32;
                        structural_mask[batch_feature_index] = 1.0;
                        let line_base = row * LINE_COUNT * SQUARE_TOKEN_PIECES;
                        line_piece_counts[line_base
                            + structural.rank * SQUARE_TOKEN_PIECES
                            + structural.piece_index] += 1.0;
                        line_piece_counts[line_base
                            + (BOARD_RANKS + structural.file) * SQUARE_TOKEN_PIECES
                            + structural.piece_index] += 1.0;
                    }
                }
            }

            let policy_base = row * max_policy_moves;
            let mut policy_offset = 0;
            for (&move_index, &target) in sample.move_indices.iter().zip(sample.policy.iter()) {
                if move_index < DENSE_MOVE_SPACE {
                    policy_indices[policy_base + policy_offset] = move_index as u32;
                    policy_targets[policy_base + policy_offset] = target.max(0.0);
                    policy_mask[policy_base + policy_offset] = 0.0;
                    policy_offset += 1;
                }
            }
            normalize_policy_targets(
                &mut policy_targets[policy_base..policy_base + max_policy_moves],
                policy_offset,
            );
            let value = sample.value.clamp(-1.0, 1.0);
            values[row] = value;
            moves_left[row] = sample.moves_left.clamp(0.0, 1.0);
        }
        Ok(Self {
            batch_size,
            max_features,
            feature_indices: Tensor::from_vec(feature_indices, (batch_size, max_features), device)?,
            feature_mask: Tensor::from_vec(feature_mask, (batch_size, max_features, 1), device)?,
            structural_piece_indices: Tensor::from_vec(
                structural_piece_indices,
                (batch_size, max_features),
                device,
            )?,
            structural_rank_indices: Tensor::from_vec(
                structural_rank_indices,
                (batch_size, max_features),
                device,
            )?,
            structural_file_indices: Tensor::from_vec(
                structural_file_indices,
                (batch_size, max_features),
                device,
            )?,
            structural_us_king_piece_indices: Tensor::from_vec(
                structural_us_king_piece_indices,
                (batch_size, max_features),
                device,
            )?,
            structural_them_king_piece_indices: Tensor::from_vec(
                structural_them_king_piece_indices,
                (batch_size, max_features),
                device,
            )?,
            structural_mask: Tensor::from_vec(
                structural_mask,
                (batch_size, max_features, 1),
                device,
            )?,
            line_piece_counts: Tensor::from_vec(
                line_piece_counts,
                (batch_size, LINE_COUNT, SQUARE_TOKEN_PIECES),
                device,
            )?,
            policy_indices: Tensor::from_vec(
                policy_indices,
                (batch_size, max_policy_moves),
                device,
            )?,
            policy_targets: Tensor::from_vec(
                policy_targets,
                (batch_size, max_policy_moves),
                device,
            )?,
            policy_mask: Tensor::from_vec(policy_mask, (batch_size, max_policy_moves), device)?,
            values: Tensor::from_vec(values, batch_size, device)?,
            moves_left: Tensor::from_vec(moves_left, batch_size, device)?,
        })
    }
}

fn normalize_policy_targets(targets: &mut [f32], active: usize) {
    if active == 0 {
        return;
    }
    let active_targets = &mut targets[..active];
    let sum = active_targets.iter().copied().sum::<f32>();
    if sum.is_finite() && sum > 1.0e-12 {
        for target in active_targets.iter_mut() {
            *target = (*target / sum).max(0.0);
        }
    } else {
        let uniform = 1.0 / active as f32;
        active_targets.fill(uniform);
    }
}

fn line_square_mask() -> &'static [f32] {
    use std::sync::OnceLock;
    static MASK: OnceLock<Vec<f32>> = OnceLock::new();
    MASK.get_or_init(|| {
        let mut mask = vec![0.0f32; LINE_COUNT * BOARD_SIZE];
        for rank in 0..BOARD_RANKS {
            for file in 0..BOARD_FILES {
                mask[rank * BOARD_SIZE + rank * BOARD_FILES + file] = 1.0;
            }
        }
        for file in 0..BOARD_FILES {
            for rank in 0..BOARD_RANKS {
                let line = BOARD_RANKS + file;
                mask[line * BOARD_SIZE + rank * BOARD_FILES + file] = 1.0;
            }
        }
        mask
    })
}

impl AzCandleModel {
    fn from_model(model: &AzNnue, device: &Device) -> CandleResult<Self> {
        let arch = model.arch;
        let hidden = arch.hidden_size;
        Ok(Self {
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
            square_piece_token: var_from_slice(
                &model.square_piece_token,
                (SQUARE_TOKEN_PIECES, hidden),
                device,
            )?,
            square_position_token: var_from_slice(
                &model.square_position_token,
                (BOARD_SIZE, hidden),
                device,
            )?,
            line_mixer_down: var_from_slice(
                &model.line_mixer_down,
                (LINE_MIXER_RANK, hidden),
                device,
            )?,
            line_mixer_bias: var_from_slice(&model.line_mixer_bias, LINE_MIXER_RANK, device)?,
            line_mixer_up: var_from_slice(&model.line_mixer_up, (hidden, LINE_MIXER_RANK), device)?,
            line_context_bias: var_from_slice(&model.line_context_bias, hidden, device)?,
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
            line_square_mask: Tensor::from_vec(
                line_square_mask().to_vec(),
                (LINE_COUNT, BOARD_SIZE),
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

    fn all_vars(&self) -> Vec<Var> {
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
        vars.push(self.square_piece_token.clone());
        vars.push(self.square_position_token.clone());
        vars.push(self.line_mixer_down.clone());
        vars.push(self.line_mixer_bias.clone());
        vars.push(self.line_mixer_up.clone());
        vars.push(self.line_context_bias.clone());
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

    fn remove_frozen_grads(&self, grads: &mut GradStore, loss_weights: AzTrainLossWeights) {
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
            grads.remove(&self.square_piece_token);
            grads.remove(&self.square_position_token);
            grads.remove(&self.line_mixer_down);
            grads.remove(&self.line_mixer_bias);
            grads.remove(&self.line_mixer_up);
            grads.remove(&self.line_context_bias);
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

    fn copy_to_model(&self, model: &mut AzNnue) -> CandleResult<()> {
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
        copy_var(&self.square_piece_token, &mut model.square_piece_token)?;
        copy_var(
            &self.square_position_token,
            &mut model.square_position_token,
        )?;
        copy_var(&self.line_mixer_down, &mut model.line_mixer_down)?;
        copy_var(&self.line_mixer_bias, &mut model.line_mixer_bias)?;
        copy_var(&self.line_mixer_up, &mut model.line_mixer_up)?;
        copy_var(&self.line_context_bias, &mut model.line_context_bias)?;
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

    fn cpu_grads(&self, grads: &GradStore) -> CandleResult<Vec<Option<Vec<f32>>>> {
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

    fn to_cpu_values(&self) -> CandleResult<Vec<Vec<f32>>> {
        let mut values = Vec::new();
        for var in self.all_vars() {
            values.push(var.as_detached_tensor().flatten_all()?.to_vec1::<f32>()?);
        }
        Ok(values)
    }

    fn set_from_cpu_values(&self, values: &[Vec<f32>]) -> CandleResult<()> {
        for (var, values) in self.all_vars().iter().zip(values.iter()) {
            let tensor = Tensor::from_vec(values.clone(), var.shape().clone(), var.device())?;
            var.set(&tensor)?;
        }
        Ok(())
    }
}

fn cuda_device_indices() -> Vec<usize> {
    let candidates = if let Some(count) = cuda_visible_device_count() {
        (0..count).collect()
    } else if let Some(count) = nvidia_smi_device_count() {
        (0..count).collect()
    } else {
        probe_cuda_device_indices(64)
    };
    let probed = candidates
        .into_iter()
        .filter(|&index| cuda_device_supports_training(index))
        .collect::<Vec<_>>();
    if !probed.is_empty() {
        return probed;
    }
    vec![0]
}

fn cuda_visible_device_count() -> Option<usize> {
    let value = std::env::var("CUDA_VISIBLE_DEVICES").ok()?;
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed == "-1" || trimmed.eq_ignore_ascii_case("NoDevFiles") {
        return None;
    }
    let count = trimmed
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .count();
    (count > 0).then_some(count)
}

fn nvidia_smi_device_count() -> Option<usize> {
    let output = Command::new("nvidia-smi").arg("-L").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let count = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|line| line.trim_start().starts_with("GPU "))
        .count();
    (count > 0).then_some(count)
}

fn probe_cuda_device_indices(max_devices: usize) -> Vec<usize> {
    let mut devices = Vec::new();
    for index in 0..max_devices {
        if Device::new_cuda(index).is_ok() {
            devices.push(index);
        } else if !devices.is_empty() {
            break;
        }
    }
    devices
}

fn cuda_device_supports_training(index: usize) -> bool {
    let Ok(device) = Device::new_cuda(index) else {
        return false;
    };
    let values = match Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device) {
        Ok(tensor) => tensor,
        Err(_) => return false,
    };
    let indices = match Tensor::from_vec(vec![0u32, 1], 2, &device) {
        Ok(tensor) => tensor,
        Err(_) => return false,
    };
    values
        .index_select(&indices, 0)
        .and_then(|tensor| tensor.sum_all())
        .and_then(|tensor| tensor.to_scalar::<f32>())
        .is_ok()
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

fn shuffle(values: &mut [usize], rng: &mut super::SplitMix64) {
    for index in (1..values.len()).rev() {
        let swap_with = (rng.next_u64() as usize) % (index + 1);
        values.swap(index, swap_with);
    }
}
