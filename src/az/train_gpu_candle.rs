use candle_core::{Device, Result as CandleResult, Tensor, backprop::GradStore};
use candle_nn::ops::log_softmax;
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use std::{process::Command, sync::Arc, thread, time::Instant};

#[cfg(feature = "nccl-train")]
use candle_core::{
    CudaStorage, Storage,
    cuda_backend::cudarc::nccl::{result as nccl_result, safe as nccl},
    op::BackpropOp,
};

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
    #[cfg(feature = "nccl-train")]
    nccl: Option<NcclAllReduce>,
}

#[derive(Debug)]
struct GpuReplica {
    device: Device,
    model: AzCandleModel,
}

#[cfg(feature = "nccl-train")]
struct NcclAllReduce {
    comms: Vec<nccl::Comm>,
}

#[cfg(feature = "nccl-train")]
impl std::fmt::Debug for NcclAllReduce {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NcclAllReduce")
            .field("world_size", &self.comms.len())
            .finish()
    }
}

#[cfg(feature = "nccl-train")]
unsafe impl Send for NcclAllReduce {}

#[cfg(feature = "nccl-train")]
unsafe impl Sync for NcclAllReduce {}

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
                "[chineseai] multi-gpu CUDA training: devices={:?}; synchronize shards each step.",
                device_indices
            );
        }
        #[cfg(feature = "nccl-train")]
        let nccl = init_nccl_all_reduce(&replicas)?;

        Ok(Self {
            arch: model.arch,
            device_indices,
            replicas,
            optimizers,
            #[cfg(feature = "nccl-train")]
            nccl,
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
        let uses_nccl_all_reduce = self.uses_nccl_all_reduce();
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
                    replica.compute_batch_grads(
                        shard,
                        batch_len,
                        uses_nccl_all_reduce,
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

        #[cfg(feature = "nccl-train")]
        if self.nccl.is_some() {
            return self.train_batch_nccl(shard_outputs, loss_weights);
        }

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
        let output =
            self.replicas[0].compute_batch_grads(batch, batch_len, false, true, loss_weights)?;
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

    #[cfg(feature = "nccl-train")]
    fn train_batch_nccl(
        &mut self,
        mut outputs: Vec<ShardOutput>,
        _loss_weights: AzTrainLossWeights,
    ) -> CandleResult<(AzTrainStats, StepProfile)> {
        let mut stats = AzTrainStats::default();
        let mut profile = StepProfile::default();
        for output in &outputs {
            stats.add_assign(&output.stats);
            profile.add_shard(output.profile);
        }
        let all_reduce_started = Instant::now();
        self.nccl_all_reduce_grads(&mut outputs)?;
        profile.grad_sync_seconds += all_reduce_started.elapsed().as_secs_f64();

        let optimizer_started = Instant::now();
        let adam_started = Instant::now();
        self.optimizers[0].step(
            outputs[0]
                .grads
                .as_ref()
                .expect("NCCL all-reduce keeps primary gradients on GPU"),
        )?;
        profile.optimizer_step_seconds += adam_started.elapsed().as_secs_f64();
        let param_sync_started = Instant::now();
        self.nccl_broadcast_vars()?;
        profile.param_sync_seconds += param_sync_started.elapsed().as_secs_f64();
        for replica in &self.replicas {
            profile_sync(&replica.device)?;
        }
        profile.optimizer_seconds += optimizer_started.elapsed().as_secs_f64();
        Ok((stats, profile))
    }

    #[cfg(feature = "nccl-train")]
    fn nccl_all_reduce_grads(&self, outputs: &mut [ShardOutput]) -> CandleResult<()> {
        let vars_by_rank = self
            .replicas
            .iter()
            .map(|replica| replica.model.all_vars())
            .collect::<Vec<_>>();
        for var_index in 0..vars_by_rank[0].len() {
            self.nccl_all_reduce_var(outputs, &vars_by_rank, var_index)?;
        }
        Ok(())
    }

    #[cfg(feature = "nccl-train")]
    fn nccl_broadcast_vars(&self) -> CandleResult<()> {
        let vars_by_rank = self
            .replicas
            .iter()
            .map(|replica| replica.model.all_vars())
            .collect::<Vec<_>>();
        for var_index in 0..vars_by_rank[0].len() {
            self.nccl_broadcast_var(&vars_by_rank, var_index)?;
        }
        Ok(())
    }

    #[cfg(feature = "nccl-train")]
    fn nccl_broadcast_var(
        &self,
        vars_by_rank: &[Vec<candle_core::Var>],
        var_index: usize,
    ) -> CandleResult<()> {
        let nccl = self
            .nccl
            .as_ref()
            .expect("NCCL reducer should be initialized");
        let root_var = vars_by_rank[0][var_index]
            .as_detached_tensor()
            .contiguous()?;
        let mut recv_slices = Vec::with_capacity(vars_by_rank.len());
        for vars in vars_by_rank {
            let device = vars[var_index].device().as_cuda_device()?.clone();
            let recv = device
                .cuda_stream()
                .alloc_zeros::<f32>(root_var.elem_count())
                .map_err(nccl_cuda_error)?;
            recv_slices.push(recv);
        }
        let (root_storage, root_layout) = root_var.storage_and_layout();
        if !root_layout.is_contiguous() || root_layout.start_offset() != 0 {
            return Err(candle_core::Error::Msg(
                "NCCL broadcast source must be contiguous with zero offset".into(),
            ));
        }

        nccl::group_start().map_err(nccl_error)?;
        let broadcast = (|| -> CandleResult<()> {
            let root_send = match &*root_storage {
                Storage::Cuda(storage) => storage.as_cuda_slice::<f32>()?,
                _ => {
                    return Err(candle_core::Error::Msg(
                        "NCCL broadcast source must live on CUDA".into(),
                    ));
                }
            };
            for rank in 0..vars_by_rank.len() {
                let send = (rank == 0).then_some(root_send);
                nccl.comms[rank]
                    .broadcast(send, &mut recv_slices[rank], 0)
                    .map_err(nccl_error)?;
            }
            Ok(())
        })();
        let group_end = nccl::group_end().map_err(nccl_error);
        broadcast?;
        group_end?;

        for (rank, recv) in recv_slices.into_iter().enumerate().skip(1) {
            let var = &vars_by_rank[rank][var_index];
            let device = self.replicas[rank].device.as_cuda_device()?.clone();
            let storage = Storage::Cuda(CudaStorage::wrap_cuda_slice(recv, device));
            let tensor =
                Tensor::from_storage(storage, var.shape().clone(), BackpropOp::none(), false);
            var.set(&tensor)?;
        }
        Ok(())
    }

    #[cfg(feature = "nccl-train")]
    fn nccl_all_reduce_var(
        &self,
        outputs: &mut [ShardOutput],
        vars_by_rank: &[Vec<candle_core::Var>],
        var_index: usize,
    ) -> CandleResult<()> {
        let nccl = self
            .nccl
            .as_ref()
            .expect("NCCL reducer should be initialized");
        let has_grad = outputs
            .iter()
            .zip(vars_by_rank.iter())
            .map(|(output, vars)| {
                output
                    .grads
                    .as_ref()
                    .is_some_and(|grads| grads.get(&vars[var_index]).is_some())
            })
            .collect::<Vec<_>>();
        if has_grad.iter().all(|has_grad| !has_grad) {
            return Ok(());
        }
        if has_grad.iter().any(|has_grad| !has_grad) {
            return Err(candle_core::Error::Msg(format!(
                "partial gradient set for NCCL var index {var_index}: {has_grad:?}"
            )));
        }

        let mut send_tensors = Vec::with_capacity(outputs.len());
        let mut recv_slices = Vec::with_capacity(outputs.len());
        for (rank, output) in outputs.iter_mut().enumerate() {
            let var = &vars_by_rank[rank][var_index];
            let grad = output
                .grads
                .as_ref()
                .and_then(|grads| grads.get(var))
                .ok_or_else(|| {
                    candle_core::Error::Msg(format!("missing gradient for NCCL rank {rank}"))
                })?
                .contiguous()?;
            let device = grad.device().as_cuda_device()?.clone();
            let recv = device
                .cuda_stream()
                .alloc_zeros::<f32>(grad.elem_count())
                .map_err(nccl_cuda_error)?;
            send_tensors.push(grad);
            recv_slices.push(recv);
        }

        let send_storages = send_tensors
            .iter()
            .map(|tensor| {
                let (storage, layout) = tensor.storage_and_layout();
                if !layout.is_contiguous() || layout.start_offset() != 0 {
                    return Err(candle_core::Error::Msg(
                        "NCCL gradient tensor must be contiguous with zero offset".into(),
                    ));
                }
                Ok(storage)
            })
            .collect::<CandleResult<Vec<_>>>()?;

        nccl::group_start().map_err(nccl_error)?;
        let all_reduce = (|| -> CandleResult<()> {
            for rank in 0..outputs.len() {
                let send = match &*send_storages[rank] {
                    Storage::Cuda(storage) => storage.as_cuda_slice::<f32>()?,
                    _ => {
                        return Err(candle_core::Error::Msg(
                            "NCCL gradient tensor must live on CUDA".into(),
                        ));
                    }
                };
                nccl.comms[rank]
                    .all_reduce(send, &mut recv_slices[rank], &nccl::ReduceOp::Sum)
                    .map_err(nccl_error)?;
            }
            Ok(())
        })();
        let group_end = nccl::group_end().map_err(nccl_error);
        all_reduce?;
        group_end?;

        for (rank, recv) in recv_slices.into_iter().enumerate() {
            let var = &vars_by_rank[rank][var_index];
            let device = self.replicas[rank].device.as_cuda_device()?.clone();
            let storage = Storage::Cuda(CudaStorage::wrap_cuda_slice(recv, device));
            let reduced =
                Tensor::from_storage(storage, var.shape().clone(), BackpropOp::none(), false);
            outputs[rank]
                .grads
                .as_mut()
                .expect("NCCL all-reduce keeps gradients on every GPU")
                .insert(var, reduced);
        }
        Ok(())
    }

    fn uses_nccl_all_reduce(&self) -> bool {
        #[cfg(feature = "nccl-train")]
        {
            self.nccl.is_some()
        }
        #[cfg(not(feature = "nccl-train"))]
        {
            false
        }
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
        keep_worker_grads: bool,
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
        let grads = output.loss_tensor.backward()?;
        profile_sync(&self.device)?;
        let backward_seconds = backward_started.elapsed().as_secs_f64();
        let stats = output.stats;
        let cpu_grad_started = Instant::now();
        let cpu_grads = if keep_grads || keep_worker_grads {
            Vec::new()
        } else {
            self.model.cpu_grads(&grads)?
        };
        let grad_sync_seconds = cpu_grad_started.elapsed().as_secs_f64();
        Ok(ShardOutput {
            stats,
            profile: StepProfile {
                tensor_seconds,
                loss_seconds,
                backward_seconds,
                grad_sync_seconds,
                optimizer_seconds: 0.0,
                optimizer_step_seconds: 0.0,
                param_sync_seconds: 0.0,
            },
            grads: if keep_grads || keep_worker_grads {
                Some(grads)
            } else {
                None
            },
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
        let value_log_probs = log_softmax(&forward.value_logits, 1)?;
        let value_ce_per_sample = ((&batch_tensors.value_wdl * &value_log_probs)? * -1.0)?;
        let value_ce = value_ce_per_sample.sum(1)?;
        let value_ce = value_ce.sum_all()?;
        let moves_left_pred = tensor_softplus(&forward.moves_left_logits)?.squeeze(1)?;
        let moves_left_error =
            (&tensor_log1p(&moves_left_pred)? - &tensor_log1p(&batch_tensors.moves_left)?)?;
        let moves_left_sse = moves_left_error.sqr()?.sum_all()?;

        let legal_policy_logits = forward
            .policy_logits
            .gather(&batch_tensors.policy_indices, 1)?;
        let masked_policy_logits = (&legal_policy_logits + &batch_tensors.policy_mask)?;
        let log_policy = log_softmax(&masked_policy_logits, 1)?;
        let policy_ce_per_sample = ((&batch_tensors.policy_targets * &log_policy)? * -1.0)?;
        let policy_ce_per_sample = policy_ce_per_sample.sum(1)?;
        let policy_ce = policy_ce_per_sample.sum_all()?;
        let weighted_value_loss = (&value_ce * value_weight.max(0.0) as f64)?;
        let weighted_policy_ce = (&policy_ce * policy_weight.max(0.0) as f64)?;
        let weighted_moves_left_loss = (&moves_left_sse * MOVES_LEFT_AUX_WEIGHT as f64)?;
        let loss_sum = (weighted_value_loss + weighted_policy_ce + weighted_moves_left_loss)?;
        let loss_tensor = (loss_sum / global_batch_len as f64)?;

        let value_sse = value_sse.to_scalar::<f32>()?;
        let value_ce = value_ce.to_scalar::<f32>()?;
        let policy_ce = policy_ce.to_scalar::<f32>()?;
        let stats = AzTrainStats {
            loss: value_ce + policy_ce + moves_left_sse.to_scalar::<f32>()? * MOVES_LEFT_AUX_WEIGHT,
            value_loss: value_ce,
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

fn tensor_softplus(input: &Tensor) -> CandleResult<Tensor> {
    let relu = input.relu()?;
    let exp_neg_abs = input.abs()?.neg()?.exp()?;
    Ok((relu + tensor_log1p(&exp_neg_abs)?)?)
}

fn tensor_log1p(input: &Tensor) -> CandleResult<Tensor> {
    input.affine(1.0, 1.0)?.log()
}

#[cfg(feature = "nccl-train")]
fn init_nccl_all_reduce(replicas: &[GpuReplica]) -> CandleResult<Option<NcclAllReduce>> {
    if replicas.len() <= 1 {
        return Ok(None);
    }
    let streams = replicas
        .iter()
        .map(|replica| {
            replica
                .device
                .as_cuda_device()
                .map(|device| device.cuda_stream())
        })
        .collect::<CandleResult<Vec<_>>>()?;
    let comms = nccl::Comm::from_devices(streams).map_err(nccl_error)?;
    Ok(Some(NcclAllReduce { comms }))
}

#[cfg(feature = "nccl-train")]
fn nccl_error(error: nccl_result::NcclError) -> candle_core::Error {
    candle_core::Error::Msg(format!("NCCL failed: {error:?}"))
}

#[cfg(feature = "nccl-train")]
fn nccl_cuda_error(
    error: candle_core::cuda_backend::cudarc::driver::DriverError,
) -> candle_core::Error {
    candle_core::Error::Msg(format!("CUDA allocation for NCCL failed: {error:?}"))
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
    grad_sync_seconds: f64,
    optimizer_seconds: f64,
    optimizer_step_seconds: f64,
    param_sync_seconds: f64,
}

impl StepProfile {
    fn add_shard(&mut self, other: Self) {
        self.tensor_seconds = self.tensor_seconds.max(other.tensor_seconds);
        self.loss_seconds = self.loss_seconds.max(other.loss_seconds);
        self.backward_seconds = self.backward_seconds.max(other.backward_seconds);
        self.grad_sync_seconds = self.grad_sync_seconds.max(other.grad_sync_seconds);
        self.optimizer_seconds += other.optimizer_seconds;
        self.optimizer_step_seconds += other.optimizer_step_seconds;
        self.param_sync_seconds += other.param_sync_seconds;
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
    grad_sync_seconds: f64,
    optimizer_seconds: f64,
    optimizer_step_seconds: f64,
    param_sync_seconds: f64,
}

impl TrainProfile {
    fn add_step(&mut self, step: StepProfile) {
        self.tensor_seconds += step.tensor_seconds;
        self.loss_seconds += step.loss_seconds;
        self.backward_seconds += step.backward_seconds;
        self.grad_sync_seconds += step.grad_sync_seconds;
        self.optimizer_seconds += step.optimizer_seconds;
        self.optimizer_step_seconds += step.optimizer_step_seconds;
        self.param_sync_seconds += step.param_sync_seconds;
    }

    fn print(&self, samples: usize) {
        let total = self.train_step_seconds.max(f64::EPSILON);
        eprintln!(
            "[chineseai] train-profile: steps={} samples={} train={:.3}s loader_wait={:.3}s loader_pack(worker_sum)={:.3}s tensor_h2d={:.3}s loss_fwd={:.3}s backward={:.3}s optimizer={:.3}s adam_step={:.3}s param_sync={:.3}s grad_sync={:.3}s tensor%={:.1} loss%={:.1} backward%={:.1}",
            self.steps,
            samples,
            self.train_step_seconds,
            self.loader_wait_seconds,
            self.loader_pack_seconds,
            self.tensor_seconds,
            self.loss_seconds,
            self.backward_seconds,
            self.optimizer_seconds,
            self.optimizer_step_seconds,
            self.param_sync_seconds,
            self.grad_sync_seconds,
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
