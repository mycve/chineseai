use candle_core::{Device, Result as CandleResult, Tensor, Var, backprop::GradStore};
use candle_nn::ops::{log_softmax, softmax};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use std::{process::Command, thread};

use super::{
    AzNnue, AzNnueArch, AzTrainLossWeights, AzTrainStats, AzTrainingSample, DENSE_MOVE_SPACE,
    LAYER_STACKS, POLICY_CONDITION_SIZE, POLICY_CONTEXT_SIZE, VALUE_HIDDEN_SIZE, VALUE_LOGITS,
    policy_move_features, policy_move_from_features, policy_move_to_features,
};
use crate::nnue::{
    POSITIONAL_NNUE_INPUT_SIZE, PURE_NNUE_INPUT_SIZE, THREAT_FEATURE_START, THREAT_INPUT_SIZE,
    layer_stack_bucket,
};
use crate::xiangqi::BOARD_SIZE;

const POLICY_MASK_VALUE: f32 = -1.0e9;
const ADAMW_WEIGHT_DECAY: f64 = 1e-4;
const DEFAULT_MULTI_GPU_SYNC_EVERY: usize = 8;
const RMS_NORM_EPS: f64 = 1.0e-6;
const DEFAULT_MIN_SAMPLES_PER_GPU: usize = 1;

#[derive(Debug)]
pub(super) struct GpuTrainer {
    arch: AzNnueArch,
    device_indices: Vec<usize>,
    replicas: Vec<GpuReplica>,
    optimizers: Vec<AdamW>,
    sync_every: usize,
    local_steps_since_sync: usize,
}

#[derive(Debug)]
struct GpuReplica {
    device: Device,
    arch: AzNnueArch,
    vars: GpuVars,
}

#[derive(Debug)]
struct GpuVars {
    input_hidden: Var,
    threat_hidden: Var,
    hidden_bias: Var,
    psqt_weights: Var,
    threat_psqt_weights: Var,
    value_hidden_weights: Var,
    value_hidden_bias: Var,
    value_logits_weights: Var,
    value_logits_bias: Var,
    policy_move_bias: Var,
    policy_context_hidden: Var,
    policy_context_bias: Var,
    policy_feature_hidden: Var,
    policy_feature_bias: Var,
    policy_from_hidden: Var,
    policy_to_hidden: Var,
    policy_move_features: Tensor,
    policy_move_from_features: Tensor,
    policy_move_to_features: Tensor,
    value_psqt_direction: Tensor,
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
        trainer.finish_local_sync()?;
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
        let sync_every = multi_gpu_sync_every();
        if device_indices.len() > 1 {
            eprintln!(
                "[chineseai] 多卡 CUDA 训练: 设备下标={:?}；各 micro-batch 在样本间均分。\
                 可设 CHINESEAI_CUDA_DEVICES=0,1 固定顺序；设 CHINESEAI_MIN_SAMPLES_PER_GPU 限制每卡最少样本（默认 1）。",
                device_indices
            );
        }
        if device_indices.len() > 1 {
            eprintln!(
                "[chineseai] multi-gpu optimizer: local AdamW, sync_every={}",
                sync_every
            );
        }
        Ok(Self {
            arch: model.arch,
            device_indices,
            replicas,
            optimizers,
            sync_every,
            local_steps_since_sync: 0,
        })
    }

    fn matches(&self, model: &AzNnue) -> bool {
        self.arch == model.arch
            && self.device_indices == cuda_device_indices()
            && self.sync_every == multi_gpu_sync_every()
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
        if self.sync_every > 1 && active_replicas == self.replicas.len() {
            return self.train_batch_local_sgd(samples, batch, &shard_ranges, loss_weights);
        }
        self.finish_local_sync()?;

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

    fn train_batch_local_sgd(
        &mut self,
        samples: &[AzTrainingSample],
        batch: &[usize],
        shard_ranges: &[std::ops::Range<usize>],
        loss_weights: AzTrainLossWeights,
    ) -> CandleResult<AzTrainStats> {
        let shard_outputs = {
            let replicas = &self.replicas;
            let optimizers = &mut self.optimizers;
            thread::scope(|scope| {
                let mut handles = Vec::new();
                for ((replica, optimizer), range) in replicas
                    .iter()
                    .zip(optimizers.iter_mut())
                    .zip(shard_ranges.iter().cloned())
                {
                    handles.push(scope.spawn(move || {
                        replica.train_local_batch(optimizer, samples, &batch[range], loss_weights)
                    }));
                }
                let mut outputs = Vec::with_capacity(handles.len());
                for handle in handles {
                    let output = handle.join().map_err(|_| {
                        candle_core::Error::Msg("local-sgd worker panicked".into())
                    })??;
                    outputs.push(output);
                }
                CandleResult::Ok(outputs)
            })?
        };

        let mut stats = AzTrainStats::default();
        for output in shard_outputs {
            stats.add_assign(&output);
        }
        self.local_steps_since_sync += 1;
        if self.local_steps_since_sync >= self.sync_every {
            self.average_replicas_to_all()?;
            self.local_steps_since_sync = 0;
        }
        Ok(stats)
    }

    fn active_replica_count(&self, batch_len: usize) -> usize {
        let replicas = self.replicas.len();
        if replicas <= 1 {
            return 1;
        }
        let per_gpu_min = min_samples_per_gpu_limit();
        let max_usable = (batch_len / per_gpu_min).max(1);
        replicas.min(batch_len).min(max_usable)
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

    fn finish_local_sync(&mut self) -> CandleResult<()> {
        if self.sync_every <= 1 || self.replicas.len() <= 1 || self.local_steps_since_sync == 0 {
            return Ok(());
        }
        self.average_replicas_to_all()?;
        self.local_steps_since_sync = 0;
        Ok(())
    }

    fn average_replicas_to_all(&self) -> CandleResult<()> {
        if self.replicas.len() <= 1 {
            return Ok(());
        }
        let replica_values = self
            .replicas
            .iter()
            .map(|replica| replica.vars.to_cpu_values())
            .collect::<CandleResult<Vec<_>>>()?;
        let mut average_values = replica_values[0].clone();
        for values in replica_values.iter().skip(1) {
            for (avg, value) in average_values.iter_mut().zip(values.iter()) {
                for (slot, &x) in avg.iter_mut().zip(value.iter()) {
                    *slot += x;
                }
            }
        }
        let scale = 1.0 / self.replicas.len() as f32;
        for avg in &mut average_values {
            for slot in avg {
                *slot *= scale;
            }
        }
        for replica in &self.replicas {
            replica.vars.set_from_cpu_values(&average_values)?;
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
        let vars = GpuVars::from_model(model, &device)?;
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
        let forward = self.forward(&batch_tensors)?;
        let value_probs = softmax(&forward.value_logits, 1)?;
        let win = value_probs.narrow(1, 0, 1)?;
        let loss_prob = value_probs.narrow(1, 2, 1)?;
        let value = (win - loss_prob)?.squeeze(1)?;
        let value_error = (&value - &batch_tensors.values)?;
        let log_value = log_softmax(&forward.value_logits, 1)?;
        let value_ce_per_sample = ((&batch_tensors.value_targets * &log_value)? * -1.0)?;
        let value_loss = value_ce_per_sample.sum_all()?;

        let masked_policy_logits = (&forward.policy_logits + &batch_tensors.policy_mask)?;
        let log_policy = log_softmax(&masked_policy_logits, 1)?;
        let policy_ce_per_sample = ((&batch_tensors.policy_targets * &log_policy)? * -1.0)?;
        let policy_ce = policy_ce_per_sample.sum_all()?;
        let weighted_value_loss = (&value_loss * loss_weights.value.max(0.0) as f64)?;
        let weighted_policy_ce = (&policy_ce * loss_weights.policy.max(0.0) as f64)?;
        let loss_tensor = ((weighted_value_loss + weighted_policy_ce)? / global_batch_len as f64)?;
        let mut grads = loss_tensor.backward()?;
        self.vars.remove_frozen_grads(&mut grads, loss_weights);

        let value_loss = value_loss.to_scalar::<f32>()?;
        let policy_ce = policy_ce.to_scalar::<f32>()?;
        let stats = AzTrainStats {
            loss: value_loss + policy_ce,
            value_loss,
            policy_ce,
            value_pred_sum: value.sum_all()?.to_scalar::<f32>()?,
            value_pred_sq_sum: value.sqr()?.sum_all()?.to_scalar::<f32>()?,
            value_target_sum: batch_tensors.values.sum_all()?.to_scalar::<f32>()?,
            value_target_sq_sum: batch_tensors.values.sqr()?.sum_all()?.to_scalar::<f32>()?,
            value_error_sq_sum: value_error.sqr()?.sum_all()?.to_scalar::<f32>()?,
            samples: batch.len(),
        };
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

    fn train_local_batch(
        &self,
        optimizer: &mut AdamW,
        samples: &[AzTrainingSample],
        batch: &[usize],
        loss_weights: AzTrainLossWeights,
    ) -> CandleResult<AzTrainStats> {
        let output = self.compute_batch_grads(samples, batch, batch.len(), true, loss_weights)?;
        optimizer.step(
            output
                .grads
                .as_ref()
                .expect("local gradients should be kept on device"),
        )?;
        Ok(output.stats)
    }

    fn forward(&self, batch: &BatchTensors) -> CandleResult<ForwardOutput> {
        let bsz = batch.batch_size;
        let hidden_size = self.arch.hidden_size;
        let positional_embeddings = self
            .vars
            .input_hidden
            .index_select(&batch.positional_feature_indices.flatten_all()?, 0)?
            .reshape((bsz, batch.max_positional_features, hidden_size))?;
        let threat_embeddings = self
            .vars
            .threat_hidden
            .index_select(&batch.threat_feature_indices.flatten_all()?, 0)?
            .reshape((bsz, batch.max_threat_features, hidden_size))?;
        let positional_hidden = positional_embeddings
            .broadcast_mul(&batch.positional_feature_mask)?
            .sum(1)?;
        let threat_hidden = threat_embeddings
            .broadcast_mul(&batch.threat_feature_mask)?
            .sum(1)?;
        let sparse_hidden = (positional_hidden + threat_hidden)?
            .broadcast_add(&self.vars.hidden_bias)?
            .relu()?;
        let rms = sparse_hidden
            .sqr()?
            .mean_keepdim(1)?
            .affine(1.0, RMS_NORM_EPS)?
            .sqrt()?;
        let hidden = sparse_hidden.broadcast_div(&rms)?;
        let psqt = self.psqt_output(batch)?;

        let value_hidden_weights = self
            .vars
            .value_hidden_weights
            .index_select(&batch.stack_indices, 0)?;
        let value_hidden_bias = self
            .vars
            .value_hidden_bias
            .index_select(&batch.stack_indices, 0)?;
        let value_hidden = stacked_linear(&hidden, &value_hidden_weights)?
            .broadcast_add(&value_hidden_bias)?
            .relu()?;
        let value_logits_weights = self
            .vars
            .value_logits_weights
            .index_select(&batch.stack_indices, 0)?;
        let value_logits_bias = self
            .vars
            .value_logits_bias
            .index_select(&batch.stack_indices, 0)?;
        let value_logits = stacked_linear(&value_hidden, &value_logits_weights)?
            .broadcast_add(&value_logits_bias)?;
        let value_logits = (value_logits
            + psqt
                .unsqueeze(1)?
                .broadcast_mul(&self.vars.value_psqt_direction)?)?;
        let policy_bias = self.vars.policy_move_bias.reshape((1, DENSE_MOVE_SPACE))?;
        let policy_context_hidden = self
            .vars
            .policy_context_hidden
            .index_select(&batch.stack_indices, 0)?;
        let policy_context_bias = self
            .vars
            .policy_context_bias
            .index_select(&batch.stack_indices, 0)?;
        let policy_context = stacked_linear(&hidden, &policy_context_hidden)?
            .broadcast_add(&policy_context_bias)?
            .relu()?;
        let policy_feature_hidden = self
            .vars
            .policy_feature_hidden
            .index_select(&batch.stack_indices, 0)?;
        let policy_feature_bias = self
            .vars
            .policy_feature_bias
            .index_select(&batch.stack_indices, 0)?;
        let policy_condition = stacked_linear(&policy_context, &policy_feature_hidden)?
            .broadcast_add(&policy_feature_bias)?;
        let policy_feature_logits =
            policy_condition.matmul(&self.vars.policy_move_features.t()?)?;
        let policy_from_hidden = self
            .vars
            .policy_from_hidden
            .index_select(&batch.stack_indices, 0)?;
        let policy_to_hidden = self
            .vars
            .policy_to_hidden
            .index_select(&batch.stack_indices, 0)?;
        let policy_from_scores = stacked_linear(&hidden, &policy_from_hidden)?;
        let policy_to_scores = stacked_linear(&hidden, &policy_to_hidden)?;
        let policy_from_logits =
            policy_from_scores.matmul(&self.vars.policy_move_from_features.t()?)?;
        let policy_to_logits = policy_to_scores.matmul(&self.vars.policy_move_to_features.t()?)?;
        let policy_logits = (((policy_feature_logits + policy_from_logits)? + policy_to_logits)?
            .broadcast_add(&policy_bias))?;

        Ok(ForwardOutput {
            value_logits,
            policy_logits,
        })
    }

    fn psqt_output(&self, batch: &BatchTensors) -> CandleResult<Tensor> {
        let bsz = batch.batch_size;
        let positional = self
            .vars
            .psqt_weights
            .index_select(&batch.positional_psqt_indices.flatten_all()?, 0)?
            .reshape((bsz, batch.max_positional_features))?;
        let threat = self
            .vars
            .threat_psqt_weights
            .index_select(&batch.threat_psqt_indices.flatten_all()?, 0)?
            .reshape((bsz, batch.max_threat_features))?;
        let positional = positional
            .broadcast_mul(&batch.positional_feature_mask.squeeze(2)?)?
            .sum(1)?;
        let threat = threat
            .broadcast_mul(&batch.threat_feature_mask.squeeze(2)?)?
            .sum(1)?;
        positional + threat
    }
}

fn stacked_linear(input: &Tensor, weights: &Tensor) -> CandleResult<Tensor> {
    input.unsqueeze(1)?.broadcast_mul(weights)?.sum(2)
}

struct ShardOutput {
    stats: AzTrainStats,
    grads: Option<GradStore>,
    cpu_grads: Vec<Option<Vec<f32>>>,
}

struct ForwardOutput {
    value_logits: Tensor,
    policy_logits: Tensor,
}

struct BatchTensors {
    batch_size: usize,
    max_positional_features: usize,
    positional_feature_indices: Tensor,
    positional_psqt_indices: Tensor,
    positional_feature_mask: Tensor,
    max_threat_features: usize,
    threat_feature_indices: Tensor,
    threat_psqt_indices: Tensor,
    threat_feature_mask: Tensor,
    stack_indices: Tensor,
    policy_targets: Tensor,
    policy_mask: Tensor,
    value_targets: Tensor,
    values: Tensor,
}

impl BatchTensors {
    fn new(samples: &[AzTrainingSample], batch: &[usize], device: &Device) -> CandleResult<Self> {
        let batch_size = batch.len();
        let max_positional_features = batch
            .iter()
            .map(|&sample_index| {
                samples[sample_index]
                    .features
                    .iter()
                    .filter(|&&feature| feature < THREAT_FEATURE_START)
                    .count()
            })
            .max()
            .unwrap_or(0)
            .max(1);
        let max_threat_features = batch
            .iter()
            .map(|&sample_index| {
                samples[sample_index]
                    .features
                    .iter()
                    .filter(|&&feature| {
                        (THREAT_FEATURE_START..PURE_NNUE_INPUT_SIZE).contains(&feature)
                    })
                    .count()
            })
            .max()
            .unwrap_or(0)
            .max(1);
        let mut positional_feature_indices = vec![0u32; batch_size * max_positional_features];
        let mut positional_psqt_indices = vec![0u32; batch_size * max_positional_features];
        let mut positional_feature_mask = vec![0.0f32; batch_size * max_positional_features];
        let mut threat_feature_indices = vec![0u32; batch_size * max_threat_features];
        let mut threat_psqt_indices = vec![0u32; batch_size * max_threat_features];
        let mut threat_feature_mask = vec![0.0f32; batch_size * max_threat_features];
        let mut stack_indices = vec![0u32; batch_size];
        let mut policy_targets = vec![0.0f32; batch_size * DENSE_MOVE_SPACE];
        let mut policy_mask = vec![POLICY_MASK_VALUE; batch_size * DENSE_MOVE_SPACE];
        let mut value_targets = vec![0.0f32; batch_size * VALUE_LOGITS];
        let mut values = vec![0.0f32; batch_size];

        for (row, &sample_index) in batch.iter().enumerate() {
            let sample = &samples[sample_index];
            let stack = board_stack_bucket(sample);
            stack_indices[row] = stack as u32;
            let positional_feature_base = row * max_positional_features;
            let threat_feature_base = row * max_threat_features;
            let mut positional_offset = 0usize;
            let mut threat_offset = 0usize;
            for &feature in sample.features.iter() {
                if feature < THREAT_FEATURE_START {
                    positional_feature_indices[positional_feature_base + positional_offset] =
                        feature as u32;
                    positional_psqt_indices[positional_feature_base + positional_offset] =
                        (stack * POSITIONAL_NNUE_INPUT_SIZE + feature) as u32;
                    positional_feature_mask[positional_feature_base + positional_offset] = 1.0;
                    positional_offset += 1;
                } else if feature < PURE_NNUE_INPUT_SIZE {
                    let threat_feature = feature - THREAT_FEATURE_START;
                    threat_feature_indices[threat_feature_base + threat_offset] =
                        threat_feature as u32;
                    threat_psqt_indices[threat_feature_base + threat_offset] =
                        (stack * THREAT_INPUT_SIZE + threat_feature) as u32;
                    threat_feature_mask[threat_feature_base + threat_offset] = 1.0;
                    threat_offset += 1;
                }
            }

            let policy_base = row * DENSE_MOVE_SPACE;
            for (&move_index, &target) in sample.move_indices.iter().zip(sample.policy.iter()) {
                if move_index < DENSE_MOVE_SPACE {
                    policy_targets[policy_base + move_index] = target.max(0.0);
                    policy_mask[policy_base + move_index] = 0.0;
                }
            }
            let value = sample.value.clamp(-1.0, 1.0);
            values[row] = value;
            let value_target_base = row * VALUE_LOGITS;
            if value >= 0.0 {
                value_targets[value_target_base] = value;
                value_targets[value_target_base + 1] = 1.0 - value;
            } else {
                value_targets[value_target_base + 1] = 1.0 + value;
                value_targets[value_target_base + 2] = -value;
            }
        }

        Ok(Self {
            batch_size,
            max_positional_features,
            positional_feature_indices: Tensor::from_vec(
                positional_feature_indices,
                (batch_size, max_positional_features),
                device,
            )?,
            positional_psqt_indices: Tensor::from_vec(
                positional_psqt_indices,
                (batch_size, max_positional_features),
                device,
            )?,
            positional_feature_mask: Tensor::from_vec(
                positional_feature_mask,
                (batch_size, max_positional_features, 1),
                device,
            )?,
            max_threat_features,
            threat_feature_indices: Tensor::from_vec(
                threat_feature_indices,
                (batch_size, max_threat_features),
                device,
            )?,
            threat_psqt_indices: Tensor::from_vec(
                threat_psqt_indices,
                (batch_size, max_threat_features),
                device,
            )?,
            threat_feature_mask: Tensor::from_vec(
                threat_feature_mask,
                (batch_size, max_threat_features, 1),
                device,
            )?,
            stack_indices: Tensor::from_vec(stack_indices, batch_size, device)?,
            policy_targets: Tensor::from_vec(
                policy_targets,
                (batch_size, DENSE_MOVE_SPACE),
                device,
            )?,
            policy_mask: Tensor::from_vec(policy_mask, (batch_size, DENSE_MOVE_SPACE), device)?,
            value_targets: Tensor::from_vec(value_targets, (batch_size, VALUE_LOGITS), device)?,
            values: Tensor::from_vec(values, batch_size, device)?,
        })
    }
}

fn board_stack_bucket(sample: &AzTrainingSample) -> usize {
    use crate::xiangqi::{Color, Position};

    let mut position = Position::empty_for_features(Color::Red);
    for (sq, &plane) in sample.board.iter().take(BOARD_SIZE).enumerate() {
        let Some(piece) = piece_from_plane(plane) else {
            continue;
        };
        position.set_piece_for_features(sq, piece);
    }
    layer_stack_bucket(&position, Color::Red).min(LAYER_STACKS - 1)
}

fn piece_from_plane(plane: u8) -> Option<crate::xiangqi::Piece> {
    use crate::xiangqi::{Color, Piece, PieceKind};

    let index = plane.checked_sub(1)? as usize;
    let color = if index < 7 { Color::Red } else { Color::Black };
    let kind = match index % 7 {
        0 => PieceKind::General,
        1 => PieceKind::Advisor,
        2 => PieceKind::Elephant,
        3 => PieceKind::Horse,
        4 => PieceKind::Rook,
        5 => PieceKind::Cannon,
        6 => PieceKind::Soldier,
        _ => return None,
    };
    Some(Piece { color, kind })
}

impl GpuVars {
    fn from_model(model: &AzNnue, device: &Device) -> CandleResult<Self> {
        let arch = model.arch;
        let hidden = arch.hidden_size;
        Ok(Self {
            input_hidden: var_from_slice(
                &model.input_hidden,
                (POSITIONAL_NNUE_INPUT_SIZE, hidden),
                device,
            )?,
            threat_hidden: var_from_slice(
                &model.threat_hidden,
                (THREAT_INPUT_SIZE, hidden),
                device,
            )?,
            hidden_bias: var_from_slice(&model.hidden_bias, hidden, device)?,
            psqt_weights: var_from_slice(
                &model.psqt_weights,
                LAYER_STACKS * POSITIONAL_NNUE_INPUT_SIZE,
                device,
            )?,
            threat_psqt_weights: var_from_slice(
                &model.threat_psqt_weights,
                LAYER_STACKS * THREAT_INPUT_SIZE,
                device,
            )?,
            value_hidden_weights: var_from_slice(
                &model.value_hidden_weights,
                (LAYER_STACKS, VALUE_HIDDEN_SIZE, hidden),
                device,
            )?,
            value_hidden_bias: var_from_slice(
                &model.value_hidden_bias,
                (LAYER_STACKS, VALUE_HIDDEN_SIZE),
                device,
            )?,
            value_logits_weights: var_from_slice(
                &model.value_logits_weights,
                (LAYER_STACKS, VALUE_LOGITS, VALUE_HIDDEN_SIZE),
                device,
            )?,
            value_logits_bias: var_from_slice(
                &model.value_logits_bias,
                (LAYER_STACKS, VALUE_LOGITS),
                device,
            )?,
            policy_move_bias: var_from_slice(&model.policy_move_bias, DENSE_MOVE_SPACE, device)?,
            policy_context_hidden: var_from_slice(
                &model.policy_context_hidden,
                (LAYER_STACKS, POLICY_CONTEXT_SIZE, hidden),
                device,
            )?,
            policy_context_bias: var_from_slice(
                &model.policy_context_bias,
                (LAYER_STACKS, POLICY_CONTEXT_SIZE),
                device,
            )?,
            policy_feature_hidden: var_from_slice(
                &model.policy_feature_hidden,
                (LAYER_STACKS, POLICY_CONDITION_SIZE, POLICY_CONTEXT_SIZE),
                device,
            )?,
            policy_feature_bias: var_from_slice(
                &model.policy_feature_bias,
                (LAYER_STACKS, POLICY_CONDITION_SIZE),
                device,
            )?,
            policy_from_hidden: var_from_slice(
                &model.policy_from_hidden,
                (LAYER_STACKS, BOARD_SIZE, hidden),
                device,
            )?,
            policy_to_hidden: var_from_slice(
                &model.policy_to_hidden,
                (LAYER_STACKS, BOARD_SIZE, hidden),
                device,
            )?,
            policy_move_features: Tensor::from_vec(
                policy_move_features().to_vec(),
                (DENSE_MOVE_SPACE, POLICY_CONDITION_SIZE),
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
            value_psqt_direction: Tensor::from_vec(
                vec![1.0f32, 0.0, -1.0],
                (1, VALUE_LOGITS),
                device,
            )?,
        })
    }

    fn all_vars(&self) -> Vec<Var> {
        let mut vars = Vec::new();
        vars.push(self.input_hidden.clone());
        vars.push(self.threat_hidden.clone());
        vars.push(self.hidden_bias.clone());
        vars.push(self.psqt_weights.clone());
        vars.push(self.threat_psqt_weights.clone());
        vars.push(self.value_hidden_weights.clone());
        vars.push(self.value_hidden_bias.clone());
        vars.push(self.value_logits_weights.clone());
        vars.push(self.value_logits_bias.clone());
        vars.push(self.policy_move_bias.clone());
        vars.push(self.policy_context_hidden.clone());
        vars.push(self.policy_context_bias.clone());
        vars.push(self.policy_feature_hidden.clone());
        vars.push(self.policy_feature_bias.clone());
        vars.push(self.policy_from_hidden.clone());
        vars.push(self.policy_to_hidden.clone());
        vars
    }

    fn shared_vars(&self) -> Vec<Var> {
        let mut vars = Vec::new();
        vars.push(self.input_hidden.clone());
        vars.push(self.threat_hidden.clone());
        vars.push(self.hidden_bias.clone());
        vars.push(self.psqt_weights.clone());
        vars.push(self.threat_psqt_weights.clone());
        vars
    }

    fn value_head_vars(&self) -> Vec<Var> {
        vec![
            self.value_hidden_weights.clone(),
            self.value_hidden_bias.clone(),
            self.value_logits_weights.clone(),
            self.value_logits_bias.clone(),
        ]
    }

    fn policy_head_vars(&self) -> Vec<Var> {
        vec![
            self.policy_move_bias.clone(),
            self.policy_context_hidden.clone(),
            self.policy_context_bias.clone(),
            self.policy_feature_hidden.clone(),
            self.policy_feature_bias.clone(),
            self.policy_from_hidden.clone(),
            self.policy_to_hidden.clone(),
        ]
    }

    fn remove_frozen_grads(&self, grads: &mut GradStore, loss_weights: AzTrainLossWeights) {
        if !loss_weights.train_shared {
            for var in self.shared_vars() {
                grads.remove(&var);
            }
        }
        if !loss_weights.train_value_head {
            for var in self.value_head_vars() {
                grads.remove(&var);
            }
        }
        if !loss_weights.train_policy_head {
            for var in self.policy_head_vars() {
                grads.remove(&var);
            }
        }
    }

    fn copy_to_model(&self, model: &mut AzNnue) -> CandleResult<()> {
        copy_var(&self.input_hidden, &mut model.input_hidden)?;
        copy_var(&self.threat_hidden, &mut model.threat_hidden)?;
        copy_var(&self.hidden_bias, &mut model.hidden_bias)?;
        copy_var(&self.psqt_weights, &mut model.psqt_weights)?;
        copy_var(&self.threat_psqt_weights, &mut model.threat_psqt_weights)?;
        copy_var(&self.value_hidden_weights, &mut model.value_hidden_weights)?;
        copy_var(&self.value_hidden_bias, &mut model.value_hidden_bias)?;
        copy_var(&self.value_logits_weights, &mut model.value_logits_weights)?;
        copy_var(&self.value_logits_bias, &mut model.value_logits_bias)?;
        copy_var(&self.policy_move_bias, &mut model.policy_move_bias)?;
        copy_var(
            &self.policy_context_hidden,
            &mut model.policy_context_hidden,
        )?;
        copy_var(&self.policy_context_bias, &mut model.policy_context_bias)?;
        copy_var(
            &self.policy_feature_hidden,
            &mut model.policy_feature_hidden,
        )?;
        copy_var(&self.policy_feature_bias, &mut model.policy_feature_bias)?;
        copy_var(&self.policy_from_hidden, &mut model.policy_from_hidden)?;
        copy_var(&self.policy_to_hidden, &mut model.policy_to_hidden)?;
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
    if let Some(devices) = parse_cuda_device_list_env("CHINESEAI_CUDA_DEVICES") {
        return devices;
    }
    if let Some(device) = parse_cuda_device_env("CHINESEAI_CUDA_DEVICE") {
        return vec![device];
    }
    if let Some(count) = cuda_visible_device_count() {
        return (0..count).collect();
    }
    if let Some(count) = nvidia_smi_device_count() {
        return (0..count).collect();
    }
    let probed = probe_cuda_device_indices(64);
    if !probed.is_empty() {
        return probed;
    }
    vec![0]
}

fn min_samples_per_gpu_limit() -> usize {
    std::env::var("CHINESEAI_MIN_SAMPLES_PER_GPU")
        .or_else(|_| std::env::var("CHINESEAI_MIN_GPU_SHARD_SIZE"))
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_MIN_SAMPLES_PER_GPU)
}

fn multi_gpu_sync_every() -> usize {
    std::env::var("CHINESEAI_SYNC_EVERY")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(DEFAULT_MULTI_GPU_SYNC_EVERY)
}

fn parse_cuda_device_list_env(name: &str) -> Option<Vec<usize>> {
    let devices = std::env::var(name)
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|devices| !devices.is_empty())?;
    Some(devices)
}

fn parse_cuda_device_env(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
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
