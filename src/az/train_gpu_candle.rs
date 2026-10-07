use super::adamw::AzAdamW;
use super::px0_sgd::Px0Sgd;
use candle_core::{Device, Result as CandleResult, Tensor, Var, backprop::GradStore};
use candle_nn::ops::log_softmax;
use std::{sync::Arc, thread, time::Instant};

use super::{
    AzNnue, AzNnueArch, AzTrainLossWeights, AzTrainOptimizer, AzTrainStats, AzTrainingSample,
    AzValueMomentStats, WDL_HEAD_SIZE,
    candle_model::{AzCandleModel, BatchTensors},
    dataloader::{BatchPlan, DataLoaderConfig, PackedBatch, PackedStepBatch, PrefetchDataLoader},
};

/// 优化器内核的两种实现，对外暴露同一组接口（`set_base_lr`/`step`/`steps`/`last_lr`/
/// `save`/`restore`），这样自博弈循环与检查点逻辑都不必关心选的是哪一个。
#[derive(Debug)]
enum TrainOptimizer {
    Px0(Box<Px0Sgd>),
    AdamW(Box<AzAdamW>),
}

impl TrainOptimizer {
    fn new(
        kind: AzTrainOptimizer,
        vars: Vec<Var>,
        decay: Vec<bool>,
        lr: f64,
    ) -> CandleResult<Self> {
        Ok(match kind {
            AzTrainOptimizer::Px0Sgd => Self::Px0(Box::new(Px0Sgd::new(vars, lr)?)),
            AzTrainOptimizer::AdamW => Self::AdamW(Box::new(AzAdamW::new(vars, decay, lr)?)),
        })
    }

    fn set_base_lr(&mut self, lr: f64) {
        match self {
            Self::Px0(opt) => opt.base_lr = lr,
            Self::AdamW(opt) => opt.base_lr = lr,
        }
    }

    fn step(&mut self, grads: &GradStore) -> CandleResult<()> {
        match self {
            Self::Px0(opt) => opt.step(grads),
            Self::AdamW(opt) => opt.step(grads),
        }
    }

    fn steps(&self) -> usize {
        match self {
            Self::Px0(opt) => opt.steps,
            Self::AdamW(opt) => opt.steps,
        }
    }

    fn last_lr(&self) -> f64 {
        match self {
            Self::Px0(opt) => opt.last_lr,
            Self::AdamW(opt) => opt.last_lr,
        }
    }

    /// 测试用：把累计步数拨到指定值，触发检查点/留出集相关的分支。
    #[cfg(test)]
    fn set_steps(&mut self, steps: usize) {
        match self {
            Self::Px0(opt) => opt.steps = steps,
            Self::AdamW(opt) => opt.steps = steps,
        }
    }

    fn save(&self, path: &std::path::Path, next_update: usize) -> CandleResult<()> {
        match self {
            Self::Px0(opt) => opt.save(path, next_update),
            Self::AdamW(opt) => opt.save(path, next_update),
        }
    }

    fn restore(&mut self, path: &std::path::Path, next_update: usize) -> CandleResult<()> {
        match self {
            Self::Px0(opt) => opt.restore(path, next_update),
            Self::AdamW(opt) => opt.restore(path, next_update),
        }
    }
}

#[derive(Debug)]
pub(super) struct GpuTrainer {
    arch: AzNnueArch,
    optimizer_kind: AzTrainOptimizer,
    replica: GpuReplica,
    optimizer: TrainOptimizer,
    holdout: Option<Arc<Vec<AzTrainingSample>>>,
    checks: Vec<super::AzHoldoutReport>,
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
    optimizer_kind: AzTrainOptimizer,
) -> CandleResult<AzTrainStats> {
    if samples.is_empty() || epochs == 0 || lr <= 0.0 {
        return Ok(AzTrainStats::default());
    }

    if model
        .gpu_trainer
        .as_ref()
        .is_none_or(|trainer| !trainer.matches(model, optimizer_kind))
    {
        model.gpu_trainer = Some(Box::new(GpuTrainer::new(model, lr, optimizer_kind)?));
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
        if trainer.steps().is_multiple_of(super::PX0_CYCLE_STEPS) {
            trainer.check_holdout(batch_size, loss_weights)?;
        }
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
                if trainer.steps().is_multiple_of(super::PX0_TEST_STEPS) {
                    trainer.check_holdout(batch_size, loss_weights)?;
                }
                profile.steps += 1;
            }
            loader.join().map_err(dataloader_error)?;
        }
    }
    if stats.samples > 0 {
        let denom = stats.samples as f32;
        stats.loss /= denom;
        let valid = stats
            .phase_value
            .iter()
            .map(|p| p.samples)
            .sum::<usize>()
            .max(1) as f32;
        stats.value_loss /= valid;
        stats.policy_ce /= denom;
        stats.moves_left_loss /= stats.moves_left_samples.max(1) as f32;
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
    pub(super) fn new(
        model: &AzNnue,
        lr: f32,
        optimizer_kind: AzTrainOptimizer,
    ) -> CandleResult<Self> {
        let replica = match GpuReplica::new(model, 0) {
            Ok(replica) => replica,
            Err(_) => {
                eprintln!("[chineseai] no usable CUDA device; falling back to CPU training");
                GpuReplica::new_cpu(model)?
            }
        };
        let (vars, decay) = replica.model.all_vars_with_decay();
        let optimizer = TrainOptimizer::new(optimizer_kind, vars, decay, lr as f64)?;

        Ok(Self {
            arch: model.arch,
            optimizer_kind,
            replica,
            optimizer,
            holdout: None,
            checks: Vec::new(),
        })
    }

    fn matches(&self, model: &AzNnue, optimizer_kind: AzTrainOptimizer) -> bool {
        self.arch == model.arch && self.optimizer_kind == optimizer_kind
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.optimizer.set_base_lr(lr as f64);
    }

    pub(super) fn save_state(
        &self,
        path: &std::path::Path,
        next_update: usize,
    ) -> CandleResult<()> {
        self.optimizer.save(path, next_update)
    }

    pub(super) fn restore_state(
        &mut self,
        path: &std::path::Path,
        next_update: usize,
    ) -> CandleResult<()> {
        self.optimizer.restore(path, next_update)
    }

    pub(super) fn steps(&self) -> usize {
        self.optimizer.steps()
    }

    pub(super) fn set_holdout(&mut self, samples: Vec<AzTrainingSample>) {
        self.holdout = (!samples.is_empty()).then(|| Arc::new(samples));
    }

    pub(super) fn take_checks(&mut self) -> Vec<super::AzHoldoutReport> {
        std::mem::take(&mut self.checks)
    }

    fn check_holdout(
        &mut self,
        batch_size: usize,
        weights: AzTrainLossWeights,
    ) -> CandleResult<()> {
        let Some(samples) = self.holdout.as_ref() else {
            return Ok(());
        };
        let mut stats = AzTrainStats::default();
        for start in (0..samples.len()).step_by(batch_size.max(1)) {
            let ids: Vec<_> = (start..(start + batch_size.max(1)).min(samples.len())).collect();
            let tensors = BatchTensors::from_packed(
                PackedBatch::from_indices(samples, &ids),
                &self.replica.device,
            )?;
            // 只做forward与统计，不backward、不调用optimizer。
            let output = self.replica.compute_batch_loss(
                &tensors,
                ids.len(),
                weights.value,
                weights.policy,
            )?;
            stats.add_assign(&output.stats);
        }
        let valid = stats.phase_value.iter().map(|p| p.samples).sum::<usize>();
        self.checks.push(super::AzHoldoutReport {
            step: self.steps(),
            samples: stats.samples,
            value_samples: valid,
            loss: stats.loss / stats.samples.max(1) as f32,
            value_loss: stats.value_loss / valid.max(1) as f32,
            moves_left_rmse: (stats.moves_left_loss / stats.moves_left_samples.max(1) as f32)
                .sqrt()
                * super::MOVES_LEFT_SCALE,
            moves_left_samples: stats.moves_left_samples,
            policy_kl: stats.policy_ce / stats.samples.max(1) as f32
                - super::policy_target_entropy(samples),
            value_rmse: (stats.value_error_sq_sum / valid.max(1) as f32)
                .max(0.0)
                .sqrt(),
        });
        Ok(())
    }

    pub(super) fn last_learning_rate(&self) -> f32 {
        self.optimizer.last_lr() as f32
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
        // slow-tests 下复用进程内共享的 CUDA 设备：每个测试各建一次 context 会让 candle
        // 重复 JIT 编译 kernel（约 10s/测试）。该分支只在 test + slow-tests 下编译，
        // 生产构建与默认测试构建仍走下面的 `Device::new_cuda(device_index)`。
        #[cfg(all(test, feature = "slow-tests"))]
        if device_index == 0 {
            let Some(device) = crate::az::cuda_test_device::shared_cuda_device() else {
                candle_core::bail!("no usable CUDA device")
            };
            let device = device.clone();
            let model = AzCandleModel::from_model(model, &device)?;
            return Ok(Self { device, model });
        }
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
        let moves_left_error =
            (forward.moves_left.squeeze(1)? - &batch_tensors.moves_left_targets)?.sqr()?;
        let moves_left_loss = (&moves_left_error * &batch_tensors.moves_left_weights)?.sum_all()?;
        let valid_moves_left = batch_tensors
            .moves_left_weights
            .gt(0.0)?
            .to_dtype(candle_core::DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()? as usize;
        let loss_sum = ((weighted_value_loss + weighted_policy_ce)?
            + moves_left_loss.affine(super::MOVES_LEFT_LOSS_WEIGHT as f64, 0.0)?)?;
        let loss_tensor = (&loss_sum / batch_len as f64)?;
        let moments = masked_value_moments(
            &value,
            &batch_tensors.values,
            &valid_value,
            &batch_tensors.value_phase_masks,
            &batch_tensors.value_source_phase_masks,
        )?
        .flatten_all()?
        .to_vec1::<f32>()?;
        let global = moment_stats(&moments[..7]);
        let phase_value = std::array::from_fn(|i| moment_stats(&moments[(i + 1) * 7..(i + 2) * 7]));
        let source_phase_value =
            std::array::from_fn(|i| moment_stats(&moments[(i + 4) * 7..(i + 5) * 7]));
        let metrics = Tensor::stack(
            &[
                loss_sum.detach(),
                value_ce.detach(),
                policy_ce.detach(),
                moves_left_loss.detach(),
            ],
            0,
        )?
        .to_vec1::<f32>()?;
        let stats = AzTrainStats {
            loss: metrics[0],
            value_loss: metrics[1],
            policy_ce: metrics[2],
            moves_left_loss: metrics[3],
            moves_left_samples: valid_moves_left,
            value_pred_sum: global.pred_sum,
            value_pred_sq_sum: global.pred_sq_sum,
            value_target_sum: global.target_sum,
            value_target_sq_sum: global.target_sq_sum,
            value_pred_target_sum: global.pred_target_sum,
            value_error_sq_sum: global.error_sq_sum,
            samples: batch_tensors.batch_size,
            phase_value,
            source_phase_value,
        };
        Ok(BatchLossOutput { loss_tensor, stats })
    }
}

fn masked_value_moments(
    value: &Tensor,
    target: &Tensor,
    valid: &Tensor,
    phases: &Tensor,
    source_phases: &Tensor,
) -> CandleResult<Tensor> {
    let value = value.detach();
    let target = target.detach();
    let moments = Tensor::stack(
        &[
            value.ones_like()?,
            value.clone(),
            value.sqr()?,
            target.clone(),
            target.sqr()?,
            (&value * &target)?,
            (&value - &target)?.sqr()?,
        ],
        1,
    )?;
    let valid = valid.unsqueeze(1)?;
    let masks = Tensor::cat(&[&valid, phases, source_phases], 1)?.broadcast_mul(&valid)?;
    masks
        .unsqueeze(2)?
        .broadcast_mul(&moments.unsqueeze(1)?)?
        .sum(0)
}

fn moment_stats(values: &[f32]) -> AzValueMomentStats {
    AzValueMomentStats {
        samples: values[0].round().max(0.0) as usize,
        pred_sum: values[1],
        pred_sq_sum: values[2],
        target_sum: values[3],
        target_sq_sum: values[4],
        pred_target_sum: values[5],
        error_sq_sum: values[6],
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
mod monitoring_tests {
    use super::*;

    #[cfg(feature = "slow-tests")]
    #[test]
    fn complete_sgd_batch_resume_keeps_all_model_tensors_and_momentum() {
        let position = crate::xiangqi::Position::startpos();
        let moves = position.legal_moves();
        let sample = AzTrainingSample {
            repetition_flags: Vec::new(),
            features: crate::az::nnue::extract_sparse_features_az(&position),
            rule_context: Default::default(),
            move_indices: moves
                .iter()
                .map(|&mv| crate::az::dense_move_index(mv))
                .collect(),
            policy: vec![1.0 / moves.len() as f32; moves.len()],
            value_wdl: [1.0, 0.0, 0.0],
            root_search_wdl: [1.0, 0.0, 0.0],
            value: 1.0,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            moves_left: 0.0,
            moves_left_weight: 0.0,
            search_simulations: 0,
            meta: Default::default(),
        };
        let batch = || PackedBatch::from_indices(std::slice::from_ref(&sample), &[0]);
        let weights = AzTrainLossWeights {
            value: 1.0,
            policy: 1.0,
        };
        // 两个优化器内核都要在真实（含融合 CUDA 算子）路径上验证"恢复 == 不中断"。
        // 学习率按各自尺度给：Px0Sgd 用 0.02，AdamW 用 4e-4。
        for (kind, lr) in [
            (AzTrainOptimizer::Px0Sgd, 0.02f32),
            (AzTrainOptimizer::AdamW, 4e-4f32),
        ] {
            let mut model = AzNnue::random(16, 20260927);
            let mut uninterrupted = GpuTrainer::new(&model, lr, kind).unwrap();
            eprintln!(
                "batch resume device for {kind:?}: {:?}",
                uninterrupted.replica.device
            );
            let (stats, _) = uninterrupted.train_batch_single(batch(), weights).unwrap();
            assert!(stats.loss.is_finite());
            uninterrupted.copy_to_model(&mut model).unwrap();
            let dir = std::env::current_dir().unwrap().join("tmp");
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join(format!(
                "batch-resume-test-{}-{}.safetensors",
                kind.as_str(),
                std::process::id()
            ));
            uninterrupted.save_state(&path, 42).unwrap();
            let mut restored = GpuTrainer::new(&model, lr, kind).unwrap();
            restored.restore_state(&path, 42).unwrap();
            std::fs::remove_file(path).unwrap();
            uninterrupted.train_batch_single(batch(), weights).unwrap();
            restored.train_batch_single(batch(), weights).unwrap();
            assert_eq!(restored.optimizer.steps(), uninterrupted.optimizer.steps());
            assert_eq!(
                restored.last_learning_rate(),
                uninterrupted.last_learning_rate()
            );
            for (a, b) in uninterrupted
                .replica
                .model
                .all_vars()
                .iter()
                .zip(restored.replica.model.all_vars())
            {
                let delta = (a.as_tensor() - b.as_tensor())
                    .unwrap()
                    .abs()
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .max(0)
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap();
                assert!(delta < 1e-6, "{kind:?} resumed model tensor delta={delta}");
            }
        }
    }

    #[test]
    fn moves_left_upgrade_preserves_both_optimizer_states_and_steps() {
        for (kind, lr) in [
            (AzTrainOptimizer::AdamW, 4e-4),
            (AzTrainOptimizer::Px0Sgd, 0.02),
        ] {
            let mut model = AzNnue::random(8, 87);
            let legacy_candle = AzCandleModel::from_model(&model, &Device::Cpu).unwrap();
            let (mut vars, mut decay) = legacy_candle.all_vars_with_decay();
            vars.truncate(26);
            decay.truncate(26);
            let mut legacy = TrainOptimizer::new(kind, vars.clone(), decay, lr).unwrap();
            legacy.set_steps(70000);
            legacy
                .step(&vars[0].sum_all().unwrap().backward().unwrap())
                .unwrap();
            legacy_candle.copy_to_model(&mut model).unwrap();
            let migrated = AzCandleModel::from_model(&model, &Device::Cpu).unwrap();
            let (new_vars, new_decay) = migrated.all_vars_with_decay();
            let mut resumed =
                TrainOptimizer::new(kind, new_vars.clone(), new_decay.clone(), lr).unwrap();
            let path = std::env::temp_dir().join(format!(
                "chineseai-mlh-upgrade-{}-{}.safetensors",
                kind.as_str(),
                std::process::id()
            ));
            let migrated_path = path.with_extension("migrated.safetensors");
            legacy.save(&path, 44109).unwrap();
            assert_eq!(AzNnue::training_state_next_update(&path).unwrap(), 44109);
            assert!(resumed.restore(&path, 44110).is_err());
            let mut wrong_lr =
                TrainOptimizer::new(kind, new_vars.clone(), new_decay.clone(), lr * 2.0).unwrap();
            assert!(wrong_lr.restore(&path, 44109).is_err());
            new_vars[27]
                .set(&Tensor::new(&[0.5f32], &Device::Cpu).unwrap())
                .unwrap();
            assert!(resumed.restore(&path, 44109).is_err());
            new_vars[27]
                .set(&new_vars[27].zeros_like().unwrap())
                .unwrap();
            resumed.restore(&path, 44109).unwrap();
            assert_eq!(resumed.steps(), legacy.steps());
            resumed.save(&migrated_path, 44109).unwrap();
            let old_state = candle_core::safetensors::load(&path, &Device::Cpu).unwrap();
            let new_state = candle_core::safetensors::load(&migrated_path, &Device::Cpu).unwrap();
            for (name, old) in old_state {
                let new = &new_state[&name];
                if name == "decay_mask" {
                    assert_eq!(
                        old.to_vec1::<i64>().unwrap(),
                        new.to_vec1::<i64>().unwrap()[..26]
                    );
                } else {
                    assert_eq!(
                        old.flatten_all()
                            .unwrap()
                            .to_dtype(candle_core::DType::F64)
                            .unwrap()
                            .to_vec1::<f64>()
                            .unwrap(),
                        new.flatten_all()
                            .unwrap()
                            .to_dtype(candle_core::DType::F64)
                            .unwrap()
                            .to_vec1::<f64>()
                            .unwrap(),
                        "{kind:?}: {name}"
                    );
                }
            }
            for index in 26..28 {
                for prefix in if kind == AzTrainOptimizer::AdamW {
                    vec!["first_moment", "second_moment"]
                } else {
                    vec!["velocity"]
                } {
                    assert!(
                        new_state[&format!("{prefix}_{index}")]
                            .flatten_all()
                            .unwrap()
                            .to_vec1::<f32>()
                            .unwrap()
                            .iter()
                            .all(|&v| v == 0.0)
                    );
                }
            }
            // 继续训练原参数，恢复后与未中断优化器完全一致。
            legacy
                .step(&vars[0].sum_all().unwrap().backward().unwrap())
                .unwrap();
            resumed
                .step(&new_vars[0].sum_all().unwrap().backward().unwrap())
                .unwrap();
            assert_eq!(
                vars[0].flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                new_vars[0].flatten_all().unwrap().to_vec1::<f32>().unwrap()
            );
            // 新头在相同全局步数下也能接收梯度。
            resumed
                .step(&new_vars[27].sum_all().unwrap().backward().unwrap())
                .unwrap();
            assert_ne!(new_vars[27].to_vec1::<f32>().unwrap()[0], 0.0);
            std::fs::remove_file(path).unwrap();
            std::fs::remove_file(migrated_path).unwrap();
        }
    }

    #[test]
    fn moves_left_loss_trains_head_and_masks_unknown_distance() {
        let position = crate::xiangqi::Position::startpos();
        let moves = position.legal_moves();
        let sample = AzTrainingSample {
            features: crate::az::nnue::extract_sparse_features_az(&position),
            rule_context: Default::default(),
            move_indices: moves
                .iter()
                .map(|&mv| crate::az::dense_move_index(mv))
                .collect(),
            repetition_flags: Vec::new(),
            policy: vec![1.0 / moves.len() as f32; moves.len()],
            value_wdl: [0.0, 1.0, 0.0],
            root_search_wdl: [0.0, 1.0, 0.0],
            value: 0.0,
            side_sign: 1.0,
            policy_weight: 0.0,
            value_weight: 0.0,
            moves_left: 40.0,
            moves_left_weight: 1.0,
            search_simulations: 0,
            meta: Default::default(),
        };
        let mut model = AzNnue::random(8, 81);
        let replica = GpuReplica::new_cpu(&model).unwrap();
        let batch = BatchTensors::from_packed(
            PackedBatch::from_indices(&[sample.clone()], &[0]),
            &Device::Cpu,
        )
        .unwrap();
        let initial = replica.compute_batch_loss(&batch, 1, 0.0, 0.0).unwrap();
        assert_eq!(initial.stats.moves_left_samples, 1);
        let initial_loss = initial.stats.moves_left_loss;
        let (vars, decay) = replica.model.all_vars_with_decay();
        let mut optimizer =
            TrainOptimizer::new(AzTrainOptimizer::Px0Sgd, vars, decay, 0.02).unwrap();
        for _ in 0..5 {
            let loss = replica.compute_batch_loss(&batch, 1, 0.0, 0.0).unwrap();
            optimizer
                .step(&loss.loss_tensor.backward().unwrap())
                .unwrap();
        }
        let final_loss = replica
            .compute_batch_loss(&batch, 1, 0.0, 0.0)
            .unwrap()
            .stats
            .moves_left_loss;
        assert!(final_loss < initial_loss, "{final_loss} >= {initial_loss}");
        replica.model.copy_to_model(&mut model).unwrap();
        assert!(model.moves_left_active);
        let mut masked = sample;
        masked.moves_left_weight = 0.0;
        let batch =
            BatchTensors::from_packed(PackedBatch::from_indices(&[masked], &[0]), &Device::Cpu)
                .unwrap();
        let loss = replica.compute_batch_loss(&batch, 1, 0.0, 0.0).unwrap();
        assert_eq!(loss.stats.moves_left_samples, 0);
        assert_eq!(loss.loss_tensor.to_scalar::<f32>().unwrap(), 0.0);
    }

    #[test]
    fn masked_samples_do_not_pollute_value_monitoring() {
        let position = crate::xiangqi::Position::startpos();
        let moves = position.legal_moves();
        let sample = AzTrainingSample {
            repetition_flags: Vec::new(),
            features: crate::az::nnue::extract_sparse_features_az(&position),
            rule_context: Default::default(),
            move_indices: moves
                .iter()
                .map(|&mv| crate::az::dense_move_index(mv))
                .collect(),
            policy: vec![1.0; moves.len()],
            value_wdl: [0.0, 1.0, 0.0],
            root_search_wdl: [0.0, 1.0, 0.0],
            value: 0.0,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            moves_left: 0.0,
            moves_left_weight: 0.0,
            search_simulations: 0,
            meta: Default::default(),
        };
        let mut masked = sample.clone();
        masked.value = 1.0;
        masked.value_wdl = [1.0, 0.0, 0.0];
        masked.value_weight = 0.0;
        masked.policy_weight = 0.0;
        let replica = GpuReplica::new_cpu(&AzNnue::random(16, 20260907)).unwrap();
        let evaluate = |samples: &[AzTrainingSample]| {
            let ids: Vec<_> = (0..samples.len()).collect();
            let batch =
                BatchTensors::from_packed(PackedBatch::from_indices(samples, &ids), &Device::Cpu)
                    .unwrap();
            replica
                .compute_batch_loss(&batch, samples.len(), 1.0, 1.0)
                .unwrap()
        };
        let baseline = evaluate(&[sample.clone()]);
        let holdout_samples = vec![sample.clone(), masked.clone()];
        let mixed = evaluate(&[sample, masked.clone()]);
        assert_eq!(
            mixed
                .stats
                .phase_value
                .iter()
                .map(|p| p.samples)
                .sum::<usize>(),
            1
        );
        assert_eq!(
            mixed.stats.value_target_sum,
            baseline.stats.value_target_sum
        );
        assert_eq!(
            mixed.stats.value_error_sq_sum,
            baseline.stats.value_error_sq_sum
        );
        assert!((mixed.stats.value_loss - baseline.stats.value_loss).abs() < 1e-5);
        assert!(
            (mixed.loss_tensor.to_scalar::<f32>().unwrap() * 2.0
                - baseline.loss_tensor.to_scalar::<f32>().unwrap())
            .abs()
                < 1e-5
        );
        let empty = evaluate(&[masked]);
        assert_eq!(
            empty
                .stats
                .phase_value
                .iter()
                .map(|p| p.samples)
                .sum::<usize>(),
            0
        );
        assert_eq!(empty.stats.value_loss, 0.0);
        let snapshot = replica
            .model
            .all_vars()
            .iter()
            .map(|v| v.as_tensor().clone())
            .collect::<Vec<_>>();
        let optimizer = TrainOptimizer::new(
            AzTrainOptimizer::Px0Sgd,
            replica.model.all_vars(),
            vec![true; replica.model.all_vars().len()],
            0.02,
        )
        .unwrap();
        let mut trainer = GpuTrainer {
            arch: AzNnueArch { hidden_size: 16 },
            optimizer_kind: AzTrainOptimizer::Px0Sgd,
            replica,
            optimizer,
            holdout: None,
            checks: Vec::new(),
        };
        trainer.set_holdout(holdout_samples.clone());
        trainer
            .check_holdout(1, AzTrainLossWeights::default())
            .unwrap();
        assert_eq!(trainer.steps(), 0);
        let checks = trainer.take_checks();
        assert_eq!(checks[0].samples, 2);
        assert_eq!(checks[0].value_samples, 1);
        assert!(checks[0].policy_kl.is_finite());
        for (before, after) in snapshot.iter().zip(trainer.replica.model.all_vars()) {
            assert_eq!(
                (before - after.as_tensor())
                    .unwrap()
                    .abs()
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .max(0)
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap(),
                0.0
            );
        }
        trainer
            .optimizer
            .set_steps(super::super::PX0_TEST_STEPS - 1);
        let mut model = AzNnue::random(16, 20260907);
        model.gpu_trainer = Some(Box::new(trainer));
        train_samples_gpu(
            &mut model,
            Arc::new(holdout_samples),
            1,
            0.02,
            1,
            &mut super::super::SplitMix64::new(8),
            AzTrainLossWeights::default(),
            AzTrainOptimizer::Px0Sgd,
        )
        .unwrap();
        let checks = model.take_training_checks();
        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].step, super::super::PX0_TEST_STEPS);
        assert_eq!(model.training_steps(), super::super::PX0_TEST_STEPS + 1);
    }
    #[cfg(feature = "slow-tests")]
    #[test]
    fn fused_moments_match_scalar_reference_on_cpu_and_cuda() {
        let mut devices = vec![Device::Cpu];
        if let Some(cuda) = crate::az::cuda_test_device::shared_cuda_device() {
            devices.push(cuda.clone());
        }
        for device in devices {
            for n in [1, 7, 257] {
                let values = (0..n)
                    .map(|i| (i % 13) as f32 / 6.0 - 1.0)
                    .collect::<Vec<_>>();
                let targets = (0..n)
                    .map(|i| (i % 17) as f32 / 8.0 - 1.0)
                    .collect::<Vec<_>>();
                let valid = (0..n).map(|i| f32::from(i % 4 != 0)).collect::<Vec<_>>();
                let mut phases = vec![0.0f32; n * 3];
                let mut sources = vec![0.0f32; n * 9];
                let mut expected = vec![0.0f64; 13 * 7];
                for i in 0..n {
                    let phase = (i / 3) % 3;
                    let source_phase = (i % 3) * 3 + phase;
                    phases[i * 3 + phase] = 1.0;
                    sources[i * 9 + source_phase] = 1.0;
                    if valid[i] == 0.0 {
                        continue;
                    }
                    let v = values[i];
                    let t = targets[i];
                    let moments = [1.0, v, v * v, t, t * t, v * t, (v - t) * (v - t)];
                    for group in [0, 1 + phase, 4 + source_phase] {
                        for (column, value) in moments.into_iter().enumerate() {
                            expected[group * 7 + column] += value as f64;
                        }
                    }
                }
                let actual = masked_value_moments(
                    &Tensor::from_vec(values, n, &device).unwrap(),
                    &Tensor::from_vec(targets, n, &device).unwrap(),
                    &Tensor::from_vec(valid, n, &device).unwrap(),
                    &Tensor::from_vec(phases, (n, 3), &device).unwrap(),
                    &Tensor::from_vec(sources, (n, 9), &device).unwrap(),
                )
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
                for (i, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                    if i % 7 == 0 {
                        assert_eq!(*actual as f64, expected);
                    }
                    assert!(
                        (*actual as f64 - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                        "moment {i}: {actual} vs {expected}"
                    );
                }
            }
        }
    }
}
