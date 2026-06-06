use candle_core::{Device, Result as CandleResult, Tensor, backprop::GradStore};
use candle_nn::ops::log_softmax;
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use std::{process::Command, sync::Arc, thread, time::Instant};

use super::{
    AzNnue, AzNnueArch, AzTrainLossWeights, AzTrainStats, AzTrainingSample, MOVES_LEFT_AUX_WEIGHT,
    candle_model::{AzCandleModel, BatchTensors},
    dataloader::{BatchPlan, DataLoaderConfig, PackedBatch, PackedStepBatch, PrefetchDataLoader},
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
    let mut stats = AzTrainStats::default();
    let samples = Arc::new(samples.to_vec());
    let profile_enabled = train_profile_enabled();
    let mut profile = TrainProfile::default();
    {
        let trainer = model
            .gpu_trainer
            .as_mut()
            .expect("gpu trainer was initialized");
        let per_gpu = per_gpu_batch_size.max(1);
        let step_chunk = (per_gpu * trainer.replicas.len().max(1)).max(1);
        trainer.set_learning_rate(lr);
        for _ in 0..epochs {
            let config = DataLoaderConfig {
                batch_size: step_chunk,
                seed: rng.next_u64(),
                num_workers: dataloader_worker_count(),
                prefetch_batches: 2,
                shard_count: trainer.replicas.len().max(1),
                ..DataLoaderConfig::default()
            };
            let plan = BatchPlan::epoch(samples.len(), &config);
            let mut loader = PrefetchDataLoader::new(Arc::clone(&samples), plan, &config);
            stats = AzTrainStats::default();
            loop {
                let wait_started = Instant::now();
                let Some(batch) = loader.next_packed().map_err(dataloader_error)? else {
                    break;
                };
                profile.loader_wait_seconds += wait_started.elapsed().as_secs_f64();
                profile.loader_pack_seconds += batch.pack_seconds;
                let step_started = Instant::now();
                let (batch_stats, step_profile) = trainer.train_batch(batch, loss_weights)?;
                profile.train_step_seconds += step_started.elapsed().as_secs_f64();
                profile.add_step(step_profile);
                stats.add_assign(&batch_stats);
                profile.steps += 1;
            }
            loader.join().map_err(dataloader_error)?;
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
    if profile_enabled {
        profile.print(stats.samples);
    }
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
        batch: PackedStepBatch,
        loss_weights: AzTrainLossWeights,
    ) -> CandleResult<(AzTrainStats, StepProfile)> {
        let batch_len = batch.batch_size;
        let active_replicas = self.active_replica_count(batch.shards.len());
        if active_replicas <= 1 {
            let shard = batch
                .shards
                .into_iter()
                .next()
                .ok_or_else(|| candle_core::Error::Msg("empty dataloader batch".into()))?;
            return self.train_batch_single(shard, loss_weights);
        }

        let shards = batch.shards;
        let shard_outputs = thread::scope(|scope| {
            let mut handles = Vec::new();
            for (shard_index, shard) in shards.into_iter().enumerate() {
                let replica = &self.replicas[shard_index];
                handles.push(scope.spawn(move || {
                    replica.compute_batch_grads(shard, batch_len, shard_index == 0, loss_weights)
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
        let mut profile = primary.profile;
        let mut grads = primary
            .grads
            .expect("primary shard gradients should be kept on device");
        for output in outputs {
            stats.add_assign(&output.stats);
            profile.add_shard(output.profile);
            self.add_cpu_grads_to_primary(&mut grads, &output.cpu_grads)?;
        }
        profile_sync(&self.replicas[0].device)?;
        let optimizer_started = Instant::now();
        self.optimizers[0].step(&grads)?;
        profile_sync(&self.replicas[0].device)?;
        profile.optimizer_seconds += optimizer_started.elapsed().as_secs_f64();
        self.broadcast_primary_to_workers()?;
        Ok((stats, profile))
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
        batch: PackedBatch,
        loss_weights: AzTrainLossWeights,
    ) -> CandleResult<(AzTrainStats, StepProfile)> {
        let batch_len = batch.batch_size;
        let output = self.replicas[0].compute_batch_grads(batch, batch_len, true, loss_weights)?;
        let mut profile = output.profile;
        profile_sync(&self.replicas[0].device)?;
        let optimizer_started = Instant::now();
        self.optimizers[0].step(
            output
                .grads
                .as_ref()
                .expect("single-gpu gradients should be kept on device"),
        )?;
        profile_sync(&self.replicas[0].device)?;
        profile.optimizer_seconds += optimizer_started.elapsed().as_secs_f64();
        Ok((output.stats, profile))
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
        batch: PackedBatch,
        global_batch_len: usize,
        keep_grads: bool,
        loss_weights: AzTrainLossWeights,
    ) -> CandleResult<ShardOutput> {
        profile_sync(&self.device)?;
        let tensor_started = Instant::now();
        let batch_tensors = BatchTensors::from_packed(batch, &self.device)?;
        profile_sync(&self.device)?;
        let tensor_seconds = tensor_started.elapsed().as_secs_f64();
        let loss_started = Instant::now();
        let output = self.compute_batch_loss(
            &batch_tensors,
            global_batch_len,
            loss_weights.value,
            loss_weights.policy,
        )?;
        profile_sync(&self.device)?;
        let loss_seconds = loss_started.elapsed().as_secs_f64();
        let backward_started = Instant::now();
        let mut grads = output.loss_tensor.backward()?;
        profile_sync(&self.device)?;
        let backward_seconds = backward_started.elapsed().as_secs_f64();
        self.model.remove_frozen_grads(&mut grads, loss_weights);
        let stats = output.stats;
        let cpu_grad_started = Instant::now();
        let cpu_grads = if keep_grads {
            Vec::new()
        } else {
            self.model.cpu_grads(&grads)?
        };
        let cpu_grad_seconds = cpu_grad_started.elapsed().as_secs_f64();
        Ok(ShardOutput {
            stats,
            profile: StepProfile {
                tensor_seconds,
                loss_seconds,
                backward_seconds,
                cpu_grad_seconds,
                optimizer_seconds: 0.0,
            },
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
    profile: StepProfile,
    grads: Option<GradStore>,
    cpu_grads: Vec<Option<Vec<f32>>>,
}

#[derive(Clone, Copy, Debug, Default)]
struct StepProfile {
    tensor_seconds: f64,
    loss_seconds: f64,
    backward_seconds: f64,
    cpu_grad_seconds: f64,
    optimizer_seconds: f64,
}

impl StepProfile {
    fn add_shard(&mut self, other: Self) {
        self.tensor_seconds = self.tensor_seconds.max(other.tensor_seconds);
        self.loss_seconds = self.loss_seconds.max(other.loss_seconds);
        self.backward_seconds = self.backward_seconds.max(other.backward_seconds);
        self.cpu_grad_seconds = self.cpu_grad_seconds.max(other.cpu_grad_seconds);
        self.optimizer_seconds += other.optimizer_seconds;
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct TrainProfile {
    steps: usize,
    loader_wait_seconds: f64,
    loader_pack_seconds: f64,
    train_step_seconds: f64,
    tensor_seconds: f64,
    loss_seconds: f64,
    backward_seconds: f64,
    cpu_grad_seconds: f64,
    optimizer_seconds: f64,
}

impl TrainProfile {
    fn add_step(&mut self, step: StepProfile) {
        self.tensor_seconds += step.tensor_seconds;
        self.loss_seconds += step.loss_seconds;
        self.backward_seconds += step.backward_seconds;
        self.cpu_grad_seconds += step.cpu_grad_seconds;
        self.optimizer_seconds += step.optimizer_seconds;
    }

    fn print(&self, samples: usize) {
        let total = self.train_step_seconds.max(f64::EPSILON);
        eprintln!(
            "[chineseai] train-profile: steps={} samples={} train={:.3}s loader_wait={:.3}s loader_pack(worker_sum)={:.3}s tensor_h2d={:.3}s loss_fwd={:.3}s backward={:.3}s optimizer={:.3}s cpu_grads={:.3}s tensor%={:.1} loss%={:.1} backward%={:.1}",
            self.steps,
            samples,
            self.train_step_seconds,
            self.loader_wait_seconds,
            self.loader_pack_seconds,
            self.tensor_seconds,
            self.loss_seconds,
            self.backward_seconds,
            self.optimizer_seconds,
            self.cpu_grad_seconds,
            self.tensor_seconds * 100.0 / total,
            self.loss_seconds * 100.0 / total,
            self.backward_seconds * 100.0 / total,
        );
    }
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

fn dataloader_worker_count() -> usize {
    let available = thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    available.clamp(1, 4)
}

fn train_profile_enabled() -> bool {
    std::env::var("CHINESEAI_TRAIN_PROFILE")
        .is_ok_and(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
}

fn profile_sync(device: &Device) -> CandleResult<()> {
    if train_profile_enabled() {
        device.synchronize()?;
    }
    Ok(())
}

fn dataloader_error(error: super::dataloader::DataLoaderError) -> candle_core::Error {
    candle_core::Error::Msg(format!("dataloader failed: {error:?}"))
}
