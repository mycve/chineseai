use candle_core::{Device, Result as CandleResult, Tensor, backprop::GradStore};
use candle_nn::ops::log_softmax;
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use std::{sync::Arc, thread, time::Instant};

use super::{
    AzNnue, AzNnueArch, AzTrainLossWeights, AzTrainStats, AzTrainingSample, AzValueMomentStats,
    WDL_HEAD_SIZE,
    candle_model::{AzCandleModel, BatchTensors},
    dataloader::{BatchPlan, DataLoaderConfig, PackedBatch, PackedStepBatch, PrefetchDataLoader},
};

const ADAMW_WEIGHT_DECAY: f64 = 1e-4;

#[derive(Debug)]
pub(super) struct GpuTrainer {
    arch: AzNnueArch,
    replica: GpuReplica,
    optimizer: AdamW,
}

#[derive(Debug)]
struct GpuReplica {
    device: Device,
    model: AzCandleModel,
}

pub(super) fn train_samples_gpu(
    model: &mut AzNnue,
    samples: Arc<Vec<AzTrainingSample>>,
    epochs: usize,
    lr: f32,
    batch_size: usize,
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
    let profile_enabled = train_profile_enabled();
    let mut profile = TrainProfile::default();
    {
        let trainer = model
            .gpu_trainer
            .as_mut()
            .expect("gpu trainer was initialized");
        let step_chunk = batch_size.max(1);
        trainer.set_learning_rate(lr);
        for _ in 0..epochs {
            let config = DataLoaderConfig {
                batch_size: step_chunk,
                seed: rng.next_u64(),
                num_workers: dataloader_worker_count(),
                prefetch_batches: 2,
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
        stats.value_loss /= stats.value_samples.max(1) as f32;
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
        let replica = match GpuReplica::new(model, 0) {
            Ok(replica) => replica,
            Err(_) => {
                eprintln!("[chineseai] no usable CUDA device; falling back to CPU training");
                GpuReplica::new_cpu(model)?
            }
        };
        let optimizer = AdamW::new(
            replica.model.all_vars(),
            ParamsAdamW {
                lr: lr as f64,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: ADAMW_WEIGHT_DECAY,
            },
        )?;

        Ok(Self {
            arch: model.arch,
            replica,
            optimizer,
        })
    }

    fn matches(&self, model: &AzNnue) -> bool {
        self.arch == model.arch
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.optimizer.set_learning_rate(lr as f64);
    }

    fn train_batch(
        &mut self,
        batch: PackedStepBatch,
        loss_weights: AzTrainLossWeights,
    ) -> CandleResult<(AzTrainStats, StepProfile)> {
        self.train_batch_single(batch.batch, loss_weights)
    }

    fn train_batch_single(
        &mut self,
        batch: PackedBatch,
        loss_weights: AzTrainLossWeights,
    ) -> CandleResult<(AzTrainStats, StepProfile)> {
        let batch_len = batch.batch_size;
        let output = self
            .replica
            .compute_batch_grads(batch, batch_len, loss_weights)?;
        let mut profile = output.profile;
        profile_sync(&self.replica.device)?;
        let optimizer_started = Instant::now();
        self.optimizer.step(&output.grads)?;
        profile_sync(&self.replica.device)?;
        profile.optimizer_seconds += optimizer_started.elapsed().as_secs_f64();
        Ok((output.stats, profile))
    }

    fn copy_to_model(&self, model: &mut AzNnue) -> CandleResult<()> {
        self.replica.model.copy_to_model(model)
    }
}

impl GpuReplica {
    fn new(model: &AzNnue, device_index: usize) -> CandleResult<Self> {
        let device = Device::new_cuda(device_index)?;
        let model = AzCandleModel::from_model(model, &device)?;
        Ok(Self { device, model })
    }

    /// 无可用 CUDA 设备时的 CPU 训练副本。
    fn new_cpu(model: &AzNnue) -> CandleResult<Self> {
        let device = Device::Cpu;
        let model = AzCandleModel::from_model(model, &device)?;
        Ok(Self { device, model })
    }

    fn compute_batch_grads(
        &self,
        batch: PackedBatch,
        batch_len: usize,
        loss_weights: AzTrainLossWeights,
    ) -> CandleResult<BatchOutput> {
        profile_sync(&self.device)?;
        let tensor_started = Instant::now();
        let batch_tensors = BatchTensors::from_packed(batch, &self.device)?;
        profile_sync(&self.device)?;
        let tensor_seconds = tensor_started.elapsed().as_secs_f64();
        let loss_started = Instant::now();
        let output = self.compute_batch_loss(
            &batch_tensors,
            batch_len,
            loss_weights.value,
            loss_weights.policy,
        )?;
        profile_sync(&self.device)?;
        let loss_seconds = loss_started.elapsed().as_secs_f64();
        let backward_started = Instant::now();
        let grads = output.loss_tensor.backward()?;
        profile_sync(&self.device)?;
        let backward_seconds = backward_started.elapsed().as_secs_f64();
        let stats = output.stats;
        Ok(BatchOutput {
            stats,
            profile: StepProfile {
                tensor_seconds,
                loss_seconds,
                backward_seconds,
                optimizer_seconds: 0.0,
            },
            grads,
        })
    }

    fn compute_batch_loss(
        &self,
        batch_tensors: &BatchTensors,
        batch_len: usize,
        value_weight: f32,
        policy_weight: f32,
    ) -> CandleResult<BatchLossOutput> {
        let forward = self.model.forward(batch_tensors)?;
        let value_log_probs = log_softmax(&forward.value_logits, 1)?;
        let value_probs = value_log_probs.exp()?;
        let value = wdl_probs_to_q(&value_probs)?.squeeze(1)?;
        let value_error = (&value - &batch_tensors.values)?;
        let value_ce_per_sample = ((&batch_tensors.value_wdl * &value_log_probs)? * -1.0)?;
        let value_ce_per_sample = value_ce_per_sample.sum(1)?;
        let valid_value = batch_tensors
            .value_weights
            .gt(0.0)?
            .to_dtype(candle_core::DType::F32)?;
        let value_ce = (&value_ce_per_sample * &valid_value)?.sum_all()?;
        let masked_policy_logits = (&forward.policy_logits + &batch_tensors.policy_mask)?;
        let log_policy = log_softmax(&masked_policy_logits, 1)?;
        let policy_ce_per_sample = ((&batch_tensors.policy_targets * &log_policy)? * -1.0)?;
        let policy_ce_per_sample = policy_ce_per_sample.sum(1)?;
        let policy_ce = policy_ce_per_sample.sum_all()?;
        let weighted_value_loss = value_ce_per_sample
            .broadcast_mul(&batch_tensors.value_weights)?
            .sum_all()?
            .affine(value_weight.max(0.0) as f64, 0.0)?;
        let weighted_policy_ce = policy_ce_per_sample
            .broadcast_mul(&batch_tensors.policy_weights)?
            .sum_all()?
            .affine(policy_weight.max(0.0) as f64, 0.0)?;
        let value_denom = batch_tensors
            .value_weights
            .sum_all()?
            .clamp(1.0, f32::MAX)?;
        let policy_denom = batch_tensors
            .policy_weights
            .sum_all()?
            .clamp(1.0, f32::MAX)?;
        let loss_tensor =
            ((weighted_value_loss / value_denom)? + (weighted_policy_ce / policy_denom)?)?;
        let loss_sum = (&loss_tensor * batch_len as f64)?;

        let value_sq = value.sqr()?;
        let target_sq = batch_tensors.values.sqr()?;
        let pred_target = value.broadcast_mul(&batch_tensors.values)?;
        let error_sq = value_error.sqr()?;
        let mut metrics = Vec::with_capacity(9 + (3 + 9) * 7);
        metrics.push(loss_sum);
        metrics.push(value_ce);
        metrics.push(policy_ce);
        metrics.push((&value * &valid_value)?.sum_all()?);
        metrics.push((&value_sq * &valid_value)?.sum_all()?);
        metrics.push((&batch_tensors.values * &valid_value)?.sum_all()?);
        metrics.push((&target_sq * &valid_value)?.sum_all()?);
        metrics.push((&pred_target * &valid_value)?.sum_all()?);
        metrics.push((&error_sq * &valid_value)?.sum_all()?);
        for phase in 0..3 {
            let mask = batch_tensors
                .value_phase_masks
                .narrow(1, phase, 1)?
                .squeeze(1)?;
            let mask = (&mask * &valid_value)?;
            metrics.push(mask.sum_all()?);
            metrics.push((&value * &mask)?.sum_all()?);
            metrics.push((&value_sq * &mask)?.sum_all()?);
            metrics.push((&batch_tensors.values * &mask)?.sum_all()?);
            metrics.push((&target_sq * &mask)?.sum_all()?);
            metrics.push((&pred_target * &mask)?.sum_all()?);
            metrics.push((&error_sq * &mask)?.sum_all()?);
        }
        for source_phase in 0..9 {
            let mask = batch_tensors
                .value_source_phase_masks
                .narrow(1, source_phase, 1)?
                .squeeze(1)?;
            let mask = (&mask * &valid_value)?;
            metrics.push(mask.sum_all()?);
            metrics.push((&value * &mask)?.sum_all()?);
            metrics.push((&value_sq * &mask)?.sum_all()?);
            metrics.push((&batch_tensors.values * &mask)?.sum_all()?);
            metrics.push((&target_sq * &mask)?.sum_all()?);
            metrics.push((&pred_target * &mask)?.sum_all()?);
            metrics.push((&error_sq * &mask)?.sum_all()?);
        }
        let metrics = Tensor::stack(&metrics, 0)?.to_vec1::<f32>()?;
        let mut phase_value = [AzValueMomentStats::default(); 3];
        for (phase_stats, values) in phase_value.iter_mut().zip(metrics[9..30].chunks_exact(7)) {
            phase_stats.samples = values[0].round().max(0.0) as usize;
            phase_stats.pred_sum = values[1];
            phase_stats.pred_sq_sum = values[2];
            phase_stats.target_sum = values[3];
            phase_stats.target_sq_sum = values[4];
            phase_stats.pred_target_sum = values[5];
            phase_stats.error_sq_sum = values[6];
        }
        let mut source_phase_value = [AzValueMomentStats::default(); 9];
        for (phase_stats, values) in source_phase_value
            .iter_mut()
            .zip(metrics[30..].chunks_exact(7))
        {
            phase_stats.samples = values[0].round().max(0.0) as usize;
            phase_stats.pred_sum = values[1];
            phase_stats.pred_sq_sum = values[2];
            phase_stats.target_sum = values[3];
            phase_stats.target_sq_sum = values[4];
            phase_stats.pred_target_sum = values[5];
            phase_stats.error_sq_sum = values[6];
        }
        let stats = AzTrainStats {
            loss: metrics[0],
            value_loss: metrics[1],
            policy_ce: metrics[2],
            value_pred_sum: metrics[3],
            value_pred_sq_sum: metrics[4],
            value_target_sum: metrics[5],
            value_target_sq_sum: metrics[6],
            value_pred_target_sum: metrics[7],
            value_error_sq_sum: metrics[8],
            value_samples: phase_value.iter().map(|phase| phase.samples).sum(),
            samples: batch_tensors.batch_size,
            phase_value,
            source_phase_value,
        };
        Ok(BatchLossOutput { loss_tensor, stats })
    }
}

fn wdl_probs_to_q(probs: &Tensor) -> CandleResult<Tensor> {
    let weights = Tensor::from_vec(vec![1.0f32, 0.0, -1.0], (WDL_HEAD_SIZE, 1), probs.device())?;
    probs.matmul(&weights)
}

struct BatchOutput {
    stats: AzTrainStats,
    profile: StepProfile,
    grads: GradStore,
}

#[derive(Clone, Copy, Debug, Default)]
struct StepProfile {
    tensor_seconds: f64,
    loss_seconds: f64,
    backward_seconds: f64,
    optimizer_seconds: f64,
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
    optimizer_seconds: f64,
}

impl TrainProfile {
    fn add_step(&mut self, step: StepProfile) {
        self.tensor_seconds += step.tensor_seconds;
        self.loss_seconds += step.loss_seconds;
        self.backward_seconds += step.backward_seconds;
        self.optimizer_seconds += step.optimizer_seconds;
    }

    fn print(&self, samples: usize) {
        let total = self.train_step_seconds.max(f64::EPSILON);
        eprintln!(
            "[chineseai] train-profile: steps={} samples={} train={:.3}s loader_wait={:.3}s loader_pack(worker_sum)={:.3}s tensor_h2d={:.3}s loss_fwd={:.3}s backward={:.3}s optimizer={:.3}s tensor%={:.1} loss%={:.1} backward%={:.1}",
            self.steps,
            samples,
            self.train_step_seconds,
            self.loader_wait_seconds,
            self.loader_pack_seconds,
            self.tensor_seconds,
            self.loss_seconds,
            self.backward_seconds,
            self.optimizer_seconds,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masked_value_targets_do_not_change_loss_or_metrics() {
        let position = crate::xiangqi::Position::startpos();
        let moves = position.legal_moves();
        let sample = AzTrainingSample {
            features: crate::nnue::extract_sparse_features_az(&position),
            rule_context: [0.0; crate::az::RULE_CONTEXT_SIZE],
            move_indices: moves
                .iter()
                .map(|&mv| crate::az::dense_move_index(mv))
                .collect(),
            policy: vec![1.0; moves.len()],
            value_wdl: [1.0, 0.0, 0.0],
            value: 1.0,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            search_simulations: 1,
            meta: crate::az::AzSampleMeta::default(),
        };
        let model = AzNnue::random(32, 41);
        let replica = GpuReplica::new_cpu(&model).unwrap();
        let evaluate = |samples: &[AzTrainingSample]| {
            let indices: Vec<_> = (0..samples.len()).collect();
            let batch = BatchTensors::from_packed(
                PackedBatch::from_indices(samples, &indices),
                &Device::Cpu,
            )
            .unwrap();
            replica
                .compute_batch_loss(&batch, samples.len(), 1.0, 1.0)
                .unwrap()
        };
        let baseline = evaluate(std::slice::from_ref(&sample));
        let mut masked = sample.clone();
        masked.value_weight = 0.0;
        masked.policy_weight = 0.0;
        masked.value = -1.0;
        masked.value_wdl = [0.0, 0.0, 1.0];
        let mixed = evaluate(&[sample, masked.clone()]);
        assert!(
            (baseline.loss_tensor.to_scalar::<f32>().unwrap()
                - mixed.loss_tensor.to_scalar::<f32>().unwrap())
            .abs()
                < 1e-5
        );
        assert_eq!(mixed.stats.value_samples, 1);
        assert!((baseline.stats.value_error_sq_sum - mixed.stats.value_error_sq_sum).abs() < 1e-5);
        let unknown = evaluate(&[masked]);
        assert_eq!(unknown.stats.value_samples, 0);
        assert_eq!(unknown.stats.value_loss, 0.0);
        assert_eq!(unknown.stats.value_error_sq_sum, 0.0);
        assert_eq!(unknown.loss_tensor.to_scalar::<f32>().unwrap(), 0.0);
        unknown.loss_tensor.backward().unwrap();
    }
}
