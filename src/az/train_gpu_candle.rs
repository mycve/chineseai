use candle_core::{Device, Result as CandleResult, Tensor, backprop::GradStore};
use candle_nn::ops::log_softmax;
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use std::{process::Command, thread};

use super::{
    AzNnue, AzNnueArch, AzTrainLossWeights, AzTrainStats, AzTrainingSample, MOVES_LEFT_AUX_WEIGHT,
    candle_model::{AzCandleModel, BatchTensors},
};

const ADAMW_WEIGHT_DECAY: f64 = 1e-4;

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
    model: AzCandleModel,
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
                replica.model.all_vars(),
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
        let primary_vars = self.replicas[0].model.all_vars();
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
        let primary_values = self.replicas[0].model.to_cpu_values()?;
        for replica in self.replicas.iter().skip(1) {
            replica.model.set_from_cpu_values(&primary_values)?;
        }
        Ok(())
    }

    fn copy_to_model(&self, model: &mut AzNnue) -> CandleResult<()> {
        self.replicas[0].model.copy_to_model(model)
    }
}

impl GpuReplica {
    fn new(model: &AzNnue, device_index: usize) -> CandleResult<Self> {
        let device = Device::new_cuda(device_index)?;
        let model = AzCandleModel::from_model(model, &device)?;
        Ok(Self { device, model })
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
        self.model.remove_frozen_grads(&mut grads, loss_weights);
        let stats = output.stats;
        let cpu_grads = if keep_grads {
            Vec::new()
        } else {
            self.model.cpu_grads(&grads)?
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
        let forward = self.model.forward(batch_tensors)?;
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
}

struct ShardOutput {
    stats: AzTrainStats,
    grads: Option<GradStore>,
    cpu_grads: Vec<Option<Vec<f32>>>,
}

struct BatchLossOutput {
    loss_tensor: Tensor,
    stats: AzTrainStats,
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

fn shuffle(values: &mut [usize], rng: &mut super::SplitMix64) {
    for index in (1..values.len()).rev() {
        let swap_with = (rng.next_u64() as usize) % (index + 1);
        values.swap(index, swap_with);
    }
}
