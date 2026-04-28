use candle_core::{Device, Result as CandleResult, Tensor, Var, backprop::GradStore};
use candle_nn::ops::{log_softmax, softmax};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use std::{process::Command, thread};

use super::{
    AzModel, AzTrainStats, AzTrainingSample, BOARD_CHANNELS, BOARD_HISTORY_SIZE, BOARD_PLANES_SIZE,
    CNN_KERNEL_AREA, DENSE_MOVE_SPACE, PIECE_BOARD_CHANNELS, POLICY_CONDITION_SIZE,
    VALUE_HEAD_LEAK, VALUE_LOGIT_SCALE, VALUE_LOGITS, cnn_pooled_size, mobile_block_bias_size,
    mobile_block_weight_size, policy_move_features, policy_move_from_select, policy_move_to_select,
    value_head_features, value_head_map_size,
};
use crate::xiangqi::{BOARD_FILES, BOARD_RANKS};

const POLICY_MASK_VALUE: f32 = -1.0e9;
const ADAMW_WEIGHT_DECAY: f64 = 1e-4;
const DEFAULT_MULTI_GPU_SYNC_EVERY: usize = 8;
/// Minimum samples per GPU before that GPU participates in a split batch.
const DEFAULT_MIN_SAMPLES_PER_GPU: usize = 1;

#[derive(Debug)]
pub(super) struct GpuTrainer {
    model_config: super::AzModelConfig,
    hidden_size: usize,
    device_indices: Vec<usize>,
    replicas: Vec<GpuReplica>,
    optimizers: Vec<AdamW>,
    sync_every: usize,
    local_steps_since_sync: usize,
}

#[derive(Debug)]
struct GpuReplica {
    device: Device,
    model_config: super::AzModelConfig,
    vars: GpuVars,
}

#[derive(Debug)]
struct GpuVars {
    board_conv1_weights: Var,
    board_conv1_bias: Var,
    residual_dw_weights: Vec<Var>,
    residual_dw_biases: Vec<Var>,
    residual_pw_weights: Vec<Var>,
    residual_pw_biases: Vec<Var>,
    position_embed: Var,
    board_hidden: Var,
    board_hidden_bias: Var,
    value_tail_conv_weights: Var,
    value_tail_conv_bias: Var,
    value_intermediate_hidden: Var,
    value_intermediate_bias: Var,
    value_logits_weights: Var,
    value_direct_logits_weights: Var,
    value_logits_bias: Var,
    policy_from_weights: Var,
    policy_from_bias: Var,
    policy_to_weights: Var,
    policy_to_bias: Var,
    policy_pair_weights: Var,
    policy_move_bias: Var,
    policy_feature_hidden: Var,
    policy_feature_cnn: Var,
    policy_feature_bias: Var,
    policy_move_features: Tensor,
    policy_move_from_select: Tensor,
    policy_move_to_select: Tensor,
}

pub(crate) fn training_cuda_device_count() -> usize {
    cuda_device_indices().len().max(1)
}

pub(super) fn train_samples_gpu(
    model: &mut AzModel,
    samples: &[AzTrainingSample],
    epochs: usize,
    lr: f32,
    // Per-GPU micro-batch size for one training step.
    per_gpu_batch_size: usize,
    rng: &mut super::SplitMix64,
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
                let batch_stats = trainer.train_batch(samples, batch)?;
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
    fn new(model: &AzModel, lr: f32) -> CandleResult<Self> {
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
                "[chineseai] multi-gpu CUDA training: devices={:?}; set CHINESEAI_CUDA_DEVICES=0,1 to pin order; set CHINESEAI_MIN_SAMPLES_PER_GPU to avoid tiny shards",
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
            hidden_size: model.hidden_size,
            model_config: model.model_config,
            device_indices,
            replicas,
            optimizers,
            sync_every,
            local_steps_since_sync: 0,
        })
    }

    fn matches(&self, model: &AzModel) -> bool {
        self.hidden_size == model.hidden_size
            && self.model_config == model.model_config
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
    ) -> CandleResult<AzTrainStats> {
        let active_replicas = self.active_replica_count(batch.len());
        if active_replicas <= 1 {
            return self.train_batch_single(samples, batch);
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
            return self.train_batch_local_sgd(samples, batch, &shard_ranges);
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
                        replica.train_local_batch(optimizer, samples, &batch[range])
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
    ) -> CandleResult<AzTrainStats> {
        let output = self.replicas[0].compute_batch_grads(samples, batch, batch.len(), true)?;
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

    fn copy_to_model(&self, model: &mut AzModel) -> CandleResult<()> {
        self.replicas[0].vars.copy_to_model(model)
    }
}

impl GpuReplica {
    fn new(model: &AzModel, device_index: usize) -> CandleResult<Self> {
        let device = Device::new_cuda(device_index)?;
        let vars = GpuVars::from_model(model, &device)?;
        Ok(Self {
            device,
            model_config: model.model_config,
            vars,
        })
    }

    fn compute_batch_grads(
        &self,
        samples: &[AzTrainingSample],
        batch: &[usize],
        global_batch_len: usize,
        keep_grads: bool,
    ) -> CandleResult<ShardOutput> {
        let batch_tensors = BatchTensors::new(samples, batch, &self.device)?;
        let forward = self.forward(&batch_tensors)?;
        let win_logit = forward.value_logits.narrow(1, 0, 1)?;
        let loss_logit = forward.value_logits.narrow(1, 2, 1)?;
        let value = ((win_logit - loss_logit)? * VALUE_LOGIT_SCALE as f64)?
            .squeeze(1)?
            .tanh()?;
        let value_error = (&value - &batch_tensors.values)?;
        let value_loss_tensor = value_error.sqr()?.sum_all()?;

        let policy_ce = if batch_tensors.has_policy_targets {
            let masked_policy_logits = (&forward.policy_logits + &batch_tensors.policy_mask)?;
            let log_policy = log_softmax(&masked_policy_logits, 1)?;
            let policy_ce_per_sample = ((&batch_tensors.policy_targets * &log_policy)? * -1.0)?;
            policy_ce_per_sample.sum_all()?
        } else {
            (&value_loss_tensor * 0.0)?
        };
        let loss_tensor = ((&value_loss_tensor + &policy_ce)? / global_batch_len as f64)?;
        let grads = loss_tensor.backward()?;
        let value_loss = value_loss_tensor.to_scalar::<f32>()?;
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
    ) -> CandleResult<AzTrainStats> {
        let output = self.compute_batch_grads(samples, batch, batch.len(), true)?;
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
        let channels = self.model_config.model_channels;
        let blocks = self.model_config.model_blocks;
        let value_channels = self.model_config.value_head_channels;
        let value_map_size = value_head_map_size(value_channels);
        let mut features = batch
            .board_onehot
            .conv2d(&self.vars.board_conv1_weights, 1, 1, 1, 1)?
            .broadcast_add(&self.vars.board_conv1_bias.reshape((1, channels, 1, 1))?)?
            .relu()?;
        features = features
            .broadcast_add(&self.vars.position_embed.reshape((
                1,
                channels,
                BOARD_RANKS,
                BOARD_FILES,
            ))?)?
            .relu()?;
        for block in 0..blocks {
            let hidden = features
                .conv2d(&self.vars.residual_dw_weights[block], 1, 1, 1, channels)?
                .broadcast_add(&self.vars.residual_dw_biases[block].reshape((1, channels, 1, 1))?)?
                .relu()?;
            let delta = hidden
                .conv2d(&self.vars.residual_pw_weights[block], 0, 1, 1, 1)?
                .broadcast_add(
                    &self.vars.residual_pw_biases[block].reshape((1, channels, 1, 1))?,
                )?;
            features = (features + delta)?.relu()?;
        }

        let feature_flat = features.reshape((bsz, channels, BOARD_PLANES_SIZE))?;
        let avg_pool = feature_flat.mean(2)?;
        let max_pool = feature_flat.max(2)?;
        let feature_grid = features.reshape((bsz, channels, BOARD_RANKS, BOARD_FILES))?;
        let row_line_pool = (feature_grid.sum(3)?.max(2)? * (1.0 / BOARD_FILES as f64))?;
        let col_line_pool = (feature_grid.sum(2)?.max(2)? * (1.0 / BOARD_RANKS as f64))?;
        let cnn_global = Tensor::cat(&[avg_pool, max_pool, row_line_pool, col_line_pool], 1)?;

        let hidden = cnn_global
            .matmul(&self.vars.board_hidden.t()?)?
            .broadcast_add(&self.vars.board_hidden_bias)?
            .relu()?;

        let value_tail = features
            .conv2d(&self.vars.value_tail_conv_weights, 0, 1, 1, 1)?
            .broadcast_add(
                &self
                    .vars
                    .value_tail_conv_bias
                    .reshape((1, value_channels, 1, 1))?,
            )?;
        let value_flat = value_tail.reshape((bsz, value_channels, BOARD_PLANES_SIZE))?;
        let attn_logits = value_flat.narrow(1, 0, 1)?.squeeze(1)?;
        let attn_weights = softmax(&attn_logits, 1)?;
        let attn_pooled = value_flat
            .broadcast_mul(&attn_weights.unsqueeze(1)?)?
            .sum(2)?;
        let mean_pooled = value_flat.mean(2)?;
        let max_pooled = value_flat.max(2)?;
        let sq_mean = value_flat.sqr()?.mean(2)?;
        let variance = (sq_mean - mean_pooled.sqr()?)?.relu()?;
        let std_pooled = variance.sqrt()?;
        let value_features = Tensor::cat(
            &[
                attn_pooled,
                mean_pooled,
                max_pooled,
                std_pooled,
                value_flat.reshape((bsz, value_map_size))?,
                cnn_global.clone(),
            ],
            1,
        )?;
        let value_preact = value_features
            .matmul(&self.vars.value_intermediate_hidden.t()?)?
            .broadcast_add(&self.vars.value_intermediate_bias)?;
        let value_relu = value_preact.relu()?;
        let value_negative_path = (&value_preact * VALUE_HEAD_LEAK as f64)?;
        let value_positive_path = (value_relu * (1.0 - VALUE_HEAD_LEAK as f64))?;
        let value_hidden = (value_negative_path + value_positive_path)?;
        let value_logits = value_hidden
            .matmul(&self.vars.value_logits_weights.t()?)?
            .broadcast_add(&value_features.matmul(&self.vars.value_direct_logits_weights.t()?)?)?
            .broadcast_add(&self.vars.value_logits_bias)?;

        let from_scores = features
            .conv2d(&self.vars.policy_from_weights, 0, 1, 1, 1)?
            .broadcast_add(&self.vars.policy_from_bias.reshape((1, 1, 1, 1))?)?
            .reshape((bsz, BOARD_PLANES_SIZE))?;
        let to_scores = features
            .conv2d(&self.vars.policy_to_weights, 0, 1, 1, 1)?
            .broadcast_add(&self.vars.policy_to_bias.reshape((1, 1, 1, 1))?)?
            .reshape((bsz, BOARD_PLANES_SIZE))?;
        let policy_logits = from_scores
            .matmul(&self.vars.policy_move_from_select)?
            .broadcast_add(&to_scores.matmul(&self.vars.policy_move_to_select)?)?
            .broadcast_add(&self.vars.policy_move_bias)?;
        let from_features = feature_flat
            .reshape((bsz * channels, BOARD_PLANES_SIZE))?
            .matmul(&self.vars.policy_move_from_select)?
            .reshape((bsz, channels, DENSE_MOVE_SPACE))?;
        let to_features = feature_flat
            .reshape((bsz * channels, BOARD_PLANES_SIZE))?
            .matmul(&self.vars.policy_move_to_select)?
            .reshape((bsz, channels, DENSE_MOVE_SPACE))?;
        let policy_pair_logits = (&from_features * &to_features)?
            .broadcast_mul(&self.vars.policy_pair_weights.reshape((1, channels, 1))?)?
            .sum(1)?;
        let policy_logits = (policy_logits + policy_pair_logits)?;
        let policy_condition = hidden
            .matmul(&self.vars.policy_feature_hidden.t()?)?
            .broadcast_add(
                &cnn_global
                    .matmul(&self.vars.policy_feature_cnn.t()?)?
                    .broadcast_add(&self.vars.policy_feature_bias)?,
            )?;
        let policy_feature_logits =
            policy_condition.matmul(&self.vars.policy_move_features.t()?)?;
        let policy_logits = (policy_logits + policy_feature_logits)?;

        Ok(ForwardOutput {
            value_logits,
            policy_logits,
        })
    }
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
    board_onehot: Tensor,
    policy_targets: Tensor,
    policy_mask: Tensor,
    has_policy_targets: bool,
    values: Tensor,
}

impl BatchTensors {
    fn new(samples: &[AzTrainingSample], batch: &[usize], device: &Device) -> CandleResult<Self> {
        let batch_size = batch.len();
        let mut board_onehot = vec![0.0f32; batch_size * BOARD_CHANNELS * BOARD_PLANES_SIZE];
        let mut policy_targets = vec![0.0f32; batch_size * DENSE_MOVE_SPACE];
        let mut policy_mask = vec![POLICY_MASK_VALUE; batch_size * DENSE_MOVE_SPACE];
        let mut values = vec![0.0f32; batch_size];
        let mut has_policy_targets = false;

        for (row, &sample_index) in batch.iter().enumerate() {
            let sample = &samples[sample_index];
            let board_base = row * BOARD_CHANNELS * BOARD_PLANES_SIZE;
            for (idx, &plane) in sample.board.iter().take(BOARD_HISTORY_SIZE).enumerate() {
                if plane > 0 {
                    let frame = idx / BOARD_PLANES_SIZE;
                    let sq = idx % BOARD_PLANES_SIZE;
                    let channel = frame * PIECE_BOARD_CHANNELS + plane as usize - 1;
                    if channel < BOARD_CHANNELS {
                        board_onehot[board_base + channel * BOARD_PLANES_SIZE + sq] = 1.0;
                    }
                }
            }

            let policy_base = row * DENSE_MOVE_SPACE;
            for (&move_index, &target) in sample.move_indices.iter().zip(sample.policy.iter()) {
                if move_index < DENSE_MOVE_SPACE {
                    policy_targets[policy_base + move_index] = target.max(0.0);
                    policy_mask[policy_base + move_index] = 0.0;
                    if target > 0.0 {
                        has_policy_targets = true;
                    }
                }
            }
            let value = sample.value.clamp(-1.0, 1.0);
            values[row] = value;
        }

        Ok(Self {
            batch_size,
            board_onehot: Tensor::from_vec(
                board_onehot,
                (batch_size, BOARD_CHANNELS, BOARD_RANKS, BOARD_FILES),
                device,
            )?,
            policy_targets: Tensor::from_vec(
                policy_targets,
                (batch_size, DENSE_MOVE_SPACE),
                device,
            )?,
            policy_mask: Tensor::from_vec(policy_mask, (batch_size, DENSE_MOVE_SPACE), device)?,
            has_policy_targets,
            values: Tensor::from_vec(values, batch_size, device)?,
        })
    }
}

impl GpuVars {
    fn from_model(model: &AzModel, device: &Device) -> CandleResult<Self> {
        let hidden = model.hidden_size;
        let config = model.model_config.normalized();
        let channels = config.model_channels;
        let blocks = config.model_blocks;
        let value_channels = config.value_head_channels;
        let value_hidden_size = config.value_hidden_size;
        let pooled_size = cnn_pooled_size(channels);
        let value_features = value_head_features(channels, value_channels);
        let mobile_weight_size = mobile_block_weight_size(channels);
        let mobile_bias_size = mobile_block_bias_size(channels);
        let mut residual_dw_weights = Vec::with_capacity(blocks);
        let mut residual_dw_biases = Vec::with_capacity(blocks);
        let mut residual_pw_weights = Vec::with_capacity(blocks);
        let mut residual_pw_biases = Vec::with_capacity(blocks);
        let dw_len = channels * CNN_KERNEL_AREA;
        let pw_len = channels * channels;
        for block in 0..blocks {
            let weight_offset = block * mobile_weight_size;
            let bias_offset = block * mobile_bias_size;
            residual_dw_weights.push(var_from_slice(
                &model.board_conv2_weights[weight_offset..weight_offset + dw_len],
                (channels, 1, 3, 3),
                device,
            )?);
            residual_dw_biases.push(var_from_slice(
                &model.board_conv2_bias[bias_offset..bias_offset + channels],
                channels,
                device,
            )?);
            residual_pw_weights.push(var_from_slice(
                &model.board_conv2_weights[weight_offset + dw_len..weight_offset + dw_len + pw_len],
                (channels, channels, 1, 1),
                device,
            )?);
            residual_pw_biases.push(var_from_slice(
                &model.board_conv2_bias[bias_offset + channels..bias_offset + mobile_bias_size],
                channels,
                device,
            )?);
        }
        Ok(Self {
            board_conv1_weights: var_from_slice(
                &model.board_conv1_weights,
                (channels, BOARD_CHANNELS, 3, 3),
                device,
            )?,
            board_conv1_bias: var_from_slice(&model.board_conv1_bias, channels, device)?,
            residual_dw_weights,
            residual_dw_biases,
            residual_pw_weights,
            residual_pw_biases,
            position_embed: var_from_slice(
                &model.position_embed,
                (channels, BOARD_RANKS, BOARD_FILES),
                device,
            )?,
            board_hidden: var_from_slice(&model.board_hidden, (hidden, pooled_size), device)?,
            board_hidden_bias: var_from_slice(&model.board_hidden_bias, hidden, device)?,
            value_tail_conv_weights: var_from_slice(
                &model.value_tail_conv_weights,
                (value_channels, channels, 1, 1),
                device,
            )?,
            value_tail_conv_bias: var_from_slice(
                &model.value_tail_conv_bias,
                value_channels,
                device,
            )?,
            value_intermediate_hidden: var_from_slice(
                &model.value_intermediate_hidden,
                (value_hidden_size, value_features),
                device,
            )?,
            value_intermediate_bias: var_from_slice(
                &model.value_intermediate_bias,
                value_hidden_size,
                device,
            )?,
            value_logits_weights: var_from_slice(
                &model.value_logits_weights,
                (VALUE_LOGITS, value_hidden_size),
                device,
            )?,
            value_direct_logits_weights: var_from_slice(
                &model.value_direct_logits_weights,
                (VALUE_LOGITS, value_features),
                device,
            )?,
            value_logits_bias: var_from_slice(&model.value_logits_bias, VALUE_LOGITS, device)?,
            policy_from_weights: var_from_slice(
                &model.policy_from_weights,
                (1, channels, 1, 1),
                device,
            )?,
            policy_from_bias: var_from_slice(&model.policy_from_bias, 1, device)?,
            policy_to_weights: var_from_slice(
                &model.policy_to_weights,
                (1, channels, 1, 1),
                device,
            )?,
            policy_to_bias: var_from_slice(&model.policy_to_bias, 1, device)?,
            policy_pair_weights: var_from_slice(&model.policy_pair_weights, channels, device)?,
            policy_move_bias: var_from_slice(&model.policy_move_bias, DENSE_MOVE_SPACE, device)?,
            policy_feature_hidden: var_from_slice(
                &model.policy_feature_hidden,
                (POLICY_CONDITION_SIZE, hidden),
                device,
            )?,
            policy_feature_cnn: var_from_slice(
                &model.policy_feature_cnn,
                (POLICY_CONDITION_SIZE, pooled_size),
                device,
            )?,
            policy_feature_bias: var_from_slice(
                &model.policy_feature_bias,
                POLICY_CONDITION_SIZE,
                device,
            )?,
            policy_move_features: Tensor::from_vec(
                policy_move_features().to_vec(),
                (DENSE_MOVE_SPACE, POLICY_CONDITION_SIZE),
                device,
            )?,
            policy_move_from_select: Tensor::from_vec(
                policy_move_from_select().to_vec(),
                (BOARD_PLANES_SIZE, DENSE_MOVE_SPACE),
                device,
            )?,
            policy_move_to_select: Tensor::from_vec(
                policy_move_to_select().to_vec(),
                (BOARD_PLANES_SIZE, DENSE_MOVE_SPACE),
                device,
            )?,
        })
    }

    fn all_vars(&self) -> Vec<Var> {
        let mut vars = Vec::new();
        vars.push(self.board_conv1_weights.clone());
        vars.push(self.board_conv1_bias.clone());
        vars.extend(self.residual_dw_weights.iter().cloned());
        vars.extend(self.residual_dw_biases.iter().cloned());
        vars.extend(self.residual_pw_weights.iter().cloned());
        vars.extend(self.residual_pw_biases.iter().cloned());
        vars.push(self.position_embed.clone());
        vars.push(self.board_hidden.clone());
        vars.push(self.board_hidden_bias.clone());
        vars.push(self.value_tail_conv_weights.clone());
        vars.push(self.value_tail_conv_bias.clone());
        vars.push(self.value_intermediate_hidden.clone());
        vars.push(self.value_intermediate_bias.clone());
        vars.push(self.value_logits_weights.clone());
        vars.push(self.value_direct_logits_weights.clone());
        vars.push(self.value_logits_bias.clone());
        vars.push(self.policy_from_weights.clone());
        vars.push(self.policy_from_bias.clone());
        vars.push(self.policy_to_weights.clone());
        vars.push(self.policy_to_bias.clone());
        vars.push(self.policy_pair_weights.clone());
        vars.push(self.policy_move_bias.clone());
        vars.push(self.policy_feature_hidden.clone());
        vars.push(self.policy_feature_cnn.clone());
        vars.push(self.policy_feature_bias.clone());
        vars
    }

    fn copy_to_model(&self, model: &mut AzModel) -> CandleResult<()> {
        let config = model.model_config.normalized();
        let channels = config.model_channels;
        let blocks = config.model_blocks;
        let mobile_weight_size = mobile_block_weight_size(channels);
        let mobile_bias_size = mobile_block_bias_size(channels);
        copy_var(&self.board_conv1_weights, &mut model.board_conv1_weights)?;
        copy_var(&self.board_conv1_bias, &mut model.board_conv1_bias)?;
        let dw_len = channels * CNN_KERNEL_AREA;
        let pw_len = channels * channels;
        for block in 0..blocks {
            let weight_offset = block * mobile_weight_size;
            let bias_offset = block * mobile_bias_size;
            copy_var(
                &self.residual_dw_weights[block],
                &mut model.board_conv2_weights[weight_offset..weight_offset + dw_len],
            )?;
            copy_var(
                &self.residual_dw_biases[block],
                &mut model.board_conv2_bias[bias_offset..bias_offset + channels],
            )?;
            copy_var(
                &self.residual_pw_weights[block],
                &mut model.board_conv2_weights
                    [weight_offset + dw_len..weight_offset + dw_len + pw_len],
            )?;
            copy_var(
                &self.residual_pw_biases[block],
                &mut model.board_conv2_bias[bias_offset + channels..bias_offset + mobile_bias_size],
            )?;
        }
        copy_var(&self.position_embed, &mut model.position_embed)?;
        copy_var(&self.board_hidden, &mut model.board_hidden)?;
        copy_var(&self.board_hidden_bias, &mut model.board_hidden_bias)?;
        copy_var(
            &self.value_tail_conv_weights,
            &mut model.value_tail_conv_weights,
        )?;
        copy_var(&self.value_tail_conv_bias, &mut model.value_tail_conv_bias)?;
        copy_var(
            &self.value_intermediate_hidden,
            &mut model.value_intermediate_hidden,
        )?;
        copy_var(
            &self.value_intermediate_bias,
            &mut model.value_intermediate_bias,
        )?;
        copy_var(&self.value_logits_weights, &mut model.value_logits_weights)?;
        copy_var(
            &self.value_direct_logits_weights,
            &mut model.value_direct_logits_weights,
        )?;
        copy_var(&self.value_logits_bias, &mut model.value_logits_bias)?;
        copy_var(&self.policy_from_weights, &mut model.policy_from_weights)?;
        copy_var(&self.policy_from_bias, &mut model.policy_from_bias)?;
        copy_var(&self.policy_to_weights, &mut model.policy_to_weights)?;
        copy_var(&self.policy_to_bias, &mut model.policy_to_bias)?;
        copy_var(&self.policy_pair_weights, &mut model.policy_pair_weights)?;
        copy_var(&self.policy_move_bias, &mut model.policy_move_bias)?;
        copy_var(
            &self.policy_feature_hidden,
            &mut model.policy_feature_hidden,
        )?;
        copy_var(&self.policy_feature_cnn, &mut model.policy_feature_cnn)?;
        copy_var(&self.policy_feature_bias, &mut model.policy_feature_bias)?;
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
