use candle_core::{Device, Result as CandleResult, Tensor, Var, backprop::GradStore};
use candle_nn::ops::{log_softmax, softmax};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use std::{process::Command, thread};

use super::{
    AzNnue, AzNnueArch, AzTrainLossWeights, AzTrainStats, AzTrainingSample, BOARD_CHANNELS,
    BOARD_HISTORY_SIZE, BOARD_PLANES_SIZE, DENSE_MOVE_SPACE, PIECE_BOARD_CHANNELS,
    POLICY_CONDITION_SIZE, POLICY_CONTEXT_SIZE, VALUE_LOGITS, policy_move_features,
    policy_move_from_indices, policy_move_to_indices,
};
use crate::nnue::V4_INPUT_SIZE;
use crate::xiangqi::{BOARD_FILES, BOARD_RANKS};

const POLICY_MASK_VALUE: f32 = -1.0e9;
const ADAMW_WEIGHT_DECAY: f64 = 1e-4;
const DEFAULT_MULTI_GPU_SYNC_EVERY: usize = 8;
const RMS_NORM_EPS: f64 = 1.0e-6;
/// 多卡时每张卡上最少样本数，低于此则少用一些卡、避免过小的子 batch（默认 1，即不额外限制）
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
    hidden_bias: Var,
    node_input_weights: Var,
    node_input_bias: Var,
    graph_self_gate: Var,
    graph_local_gate: Var,
    graph_row_gate: Var,
    graph_col_gate: Var,
    graph_bias: Var,
    node_attention_query: Var,
    node_hidden: Var,
    node_hidden_bias: Var,
    value_intermediate_hidden: Var,
    value_intermediate_nodes: Var,
    value_intermediate_bias: Var,
    value_logits_weights: Var,
    value_logits_bias: Var,
    policy_node_query: Var,
    policy_node_key: Var,
    policy_from_bias: Var,
    policy_to_bias: Var,
    policy_move_bias: Var,
    policy_context_hidden: Var,
    policy_context_nodes: Var,
    policy_context_bias: Var,
    policy_feature_hidden: Var,
    policy_feature_nodes: Var,
    policy_feature_bias: Var,
    policy_move_features: Tensor,
    policy_from_map: Tensor,
    policy_to_map: Tensor,
    graph_local_kernel: Tensor,
    graph_local_count: Tensor,
}

/// 与 `GpuTrainer` 使用的 `cuda_device_indices()` 一致。配置里「每卡 batch」× 本值 = 单步训练总 micro-batch 样本数。
pub(crate) fn training_cuda_device_count() -> usize {
    cuda_device_indices().len().max(1)
}

pub(super) fn train_samples_gpu(
    model: &mut AzNnue,
    samples: &[AzTrainingSample],
    epochs: usize,
    lr: f32,
    // 每张 GPU 上每个 training step 的样本数；单步总样本 = 该值 × 参与训练的 GPU 数
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
        // 与 active_replica_count 一致：整批按卡均分，每子批约 per_gpu 条/卡
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
        // 任一 trunk 旋钮变化都强制重建 GpuTrainer，避免 Tensor 形状错配。
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

    /// 实际参与本步前向/反传的 GPU 数量：数据并行下不会超过 batch 内样本数；
    /// 若设定了每卡最小样本数，会少用几张卡，避免子 batch 过小（可选，默认不限制）。
    fn active_replica_count(&self, batch_len: usize) -> usize {
        let replicas = self.replicas.len();
        if replicas <= 1 {
            return 1;
        }
        let per_gpu_min = min_samples_per_gpu_limit();
        // 在「每卡至少 per_gpu_min 个样本」约束下，最多能同时用多少张卡
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
        let channels = self.arch.gnn_node_channels;
        let proj = self.arch.policy_node_proj_size;
        let layers = self.arch.gnn_node_layers;
        let feature_embeddings = self
            .vars
            .input_hidden
            .index_select(&batch.feature_indices.flatten_all()?, 0)?
            .reshape((bsz, batch.max_features, hidden_size))?;
        let sparse_hidden = feature_embeddings
            .broadcast_mul(&batch.feature_mask)?
            .sum(1)?
            .broadcast_add(&self.vars.hidden_bias)?
            .relu()?;

        let node_input = batch
            .board_onehot
            .reshape((bsz, BOARD_CHANNELS, BOARD_PLANES_SIZE))?
            .transpose(1, 2)?
            .reshape((bsz * BOARD_PLANES_SIZE, BOARD_CHANNELS))?
            .matmul(&self.vars.node_input_weights.t()?)?
            .reshape((bsz, BOARD_PLANES_SIZE, channels))?
            .broadcast_add(&self.vars.node_input_bias.reshape((1, 1, channels))?)?
            .transpose(1, 2)?
            .reshape((bsz, channels, BOARD_RANKS, BOARD_FILES))?
            .relu()?;
        // 至少跑一层（已在 arch.validate 处保证 layers >= 1）。
        let mut graph = self.graph_node_layer(&node_input, 0)?;
        for layer in 1..layers {
            graph = self.graph_node_layer(&graph, layer)?;
        }
        let policy_nodes = graph
            .reshape((bsz, channels, BOARD_PLANES_SIZE))?
            .transpose(1, 2)?
            .contiguous()?;
        let policy_nodes_by_channel = policy_nodes.transpose(1, 2)?.contiguous()?;
        let avg_pool = policy_nodes_by_channel.mean(2)?;
        let max_pool = policy_nodes_by_channel.max(2)?;
        let centered_nodes = policy_nodes_by_channel.broadcast_sub(&avg_pool.unsqueeze(2)?)?;
        let std_pool = centered_nodes
            .sqr()?
            .mean(2)?
            .affine(1.0, RMS_NORM_EPS)?
            .sqrt()?;
        let attn_query = self.vars.node_attention_query.reshape((1, channels, 1))?;
        let attn_logits = policy_nodes_by_channel.broadcast_mul(&attn_query)?.sum(1)?;
        let attn_weights = softmax(&attn_logits, 1)?;
        let attn_pool = policy_nodes_by_channel
            .broadcast_mul(&attn_weights.unsqueeze(1)?)?
            .sum(2)?;
        let policy_node_grid =
            policy_nodes_by_channel.reshape((bsz, channels, BOARD_RANKS, BOARD_FILES))?;
        let row_line_pool = (policy_node_grid.sum(3)?.max(2)? * (1.0 / BOARD_FILES as f64))?;
        let col_line_pool = (policy_node_grid.sum(2)?.max(2)? * (1.0 / BOARD_RANKS as f64))?;
        let node_global = Tensor::cat(
            &[
                avg_pool,
                max_pool,
                attn_pool,
                row_line_pool,
                col_line_pool,
                std_pool,
            ],
            1,
        )?;

        let pooled_node_hidden = node_global
            .matmul(&self.vars.node_hidden.t()?)?
            .broadcast_add(&self.vars.node_hidden_bias)?
            .relu()?;
        let hidden = {
            let hidden = (sparse_hidden + pooled_node_hidden)?;
            let rms = hidden
                .sqr()?
                .mean_keepdim(1)?
                .affine(1.0, RMS_NORM_EPS)?
                .sqrt()?;
            hidden.broadcast_div(&rms)?
        };

        let value_intermediate = hidden
            .matmul(&self.vars.value_intermediate_hidden.t()?)?
            .broadcast_add(&node_global.matmul(&self.vars.value_intermediate_nodes.t()?)?)?
            .broadcast_add(&self.vars.value_intermediate_bias)?
            .relu()?;
        let value_logits = value_intermediate
            .matmul(&self.vars.value_logits_weights.t()?)?
            .broadcast_add(&self.vars.value_logits_bias)?;
        let policy_nodes_2d = policy_nodes.reshape((bsz * BOARD_PLANES_SIZE, channels))?;
        let policy_q_nodes = policy_nodes_2d
            .matmul(&self.vars.policy_node_query.t()?)?
            .reshape((bsz, BOARD_PLANES_SIZE, proj))?;
        let policy_k_nodes = policy_nodes_2d
            .matmul(&self.vars.policy_node_key.t()?)?
            .reshape((bsz, BOARD_PLANES_SIZE, proj))?;
        let policy_bias = self.vars.policy_move_bias.reshape((1, DENSE_MOVE_SPACE))?;
        let policy_context = hidden
            .matmul(&self.vars.policy_context_hidden.t()?)?
            .broadcast_add(&node_global.matmul(&self.vars.policy_context_nodes.t()?)?)?
            .broadcast_add(&self.vars.policy_context_bias)?
            .relu()?;
        let policy_condition = policy_context
            .matmul(&self.vars.policy_feature_hidden.t()?)?
            .broadcast_add(
                &node_global
                    .matmul(&self.vars.policy_feature_nodes.t()?)?
                    .broadcast_add(&self.vars.policy_feature_bias)?,
            )?;
        let policy_feature_logits =
            policy_condition.matmul(&self.vars.policy_move_features.t()?)?;
        let from_nodes = policy_nodes
            .index_select(&self.vars.policy_from_map, 1)?
            .contiguous()?;
        let to_nodes = policy_nodes
            .index_select(&self.vars.policy_to_map, 1)?
            .contiguous()?;
        let from_q = policy_q_nodes
            .index_select(&self.vars.policy_from_map, 1)?
            .contiguous()?;
        let to_k = policy_k_nodes
            .index_select(&self.vars.policy_to_map, 1)?
            .contiguous()?;
        let from_logits = from_nodes
            .broadcast_mul(&self.vars.policy_from_bias.reshape((1, 1, channels))?)?
            .sum(2)?;
        let to_logits = to_nodes
            .broadcast_mul(&self.vars.policy_to_bias.reshape((1, 1, channels))?)?
            .sum(2)?;
        let pair_logits = (from_q.broadcast_mul(&to_k)?.sum(2)? * (1.0 / (proj as f64).sqrt()))?;
        let policy_logits =
            (((policy_feature_logits.broadcast_add(&policy_bias)? + from_logits)? + to_logits)?
                + pair_logits)?;

        Ok(ForwardOutput {
            value_logits,
            policy_logits,
        })
    }

    fn graph_node_layer(&self, node_grid: &Tensor, layer: usize) -> CandleResult<Tensor> {
        let channels = self.arch.gnn_node_channels;
        let gate_range = layer * channels..(layer + 1) * channels;
        let local_mean = node_grid
            .conv2d(&self.vars.graph_local_kernel, 1, 1, 1, 1)?
            .broadcast_div(&self.vars.graph_local_count)?;
        let row_mean = node_grid.mean(3)?.unsqueeze(3)?;
        let col_mean = node_grid.mean(2)?.unsqueeze(2)?;
        let gate_shape = (1, channels, 1, 1);
        let self_gate = self
            .vars
            .graph_self_gate
            .narrow(0, gate_range.start, channels)?
            .reshape(gate_shape)?;
        let local_gate = self
            .vars
            .graph_local_gate
            .narrow(0, gate_range.start, channels)?
            .reshape(gate_shape)?;
        let row_gate = self
            .vars
            .graph_row_gate
            .narrow(0, gate_range.start, channels)?
            .reshape(gate_shape)?;
        let col_gate = self
            .vars
            .graph_col_gate
            .narrow(0, gate_range.start, channels)?
            .reshape(gate_shape)?;
        let bias = self
            .vars
            .graph_bias
            .narrow(0, gate_range.start, channels)?
            .reshape(gate_shape)?;
        let local_term = local_mean.broadcast_mul(&local_gate)?;
        let row_term = row_mean.broadcast_mul(&row_gate)?;
        let col_term = col_mean.broadcast_mul(&col_gate)?;
        node_grid
            .broadcast_mul(&self_gate)?
            .broadcast_add(&local_term)?
            .broadcast_add(&row_term)?
            .broadcast_add(&col_term)?
            .broadcast_add(&bias)?
            .relu()
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
    max_features: usize,
    feature_indices: Tensor,
    feature_mask: Tensor,
    board_onehot: Tensor,
    policy_targets: Tensor,
    policy_mask: Tensor,
    value_targets: Tensor,
    values: Tensor,
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
        let mut feature_indices = vec![0u32; batch_size * max_features];
        let mut feature_mask = vec![0.0f32; batch_size * max_features];
        let mut board_onehot = vec![0.0f32; batch_size * BOARD_CHANNELS * BOARD_PLANES_SIZE];
        let mut policy_targets = vec![0.0f32; batch_size * DENSE_MOVE_SPACE];
        let mut policy_mask = vec![POLICY_MASK_VALUE; batch_size * DENSE_MOVE_SPACE];
        let mut value_targets = vec![0.0f32; batch_size * VALUE_LOGITS];
        let mut values = vec![0.0f32; batch_size];

        for (row, &sample_index) in batch.iter().enumerate() {
            let sample = &samples[sample_index];
            let feature_base = row * max_features;
            for (feature_offset, &feature) in sample.features.iter().enumerate() {
                if feature < V4_INPUT_SIZE {
                    feature_indices[feature_base + feature_offset] = feature as u32;
                    feature_mask[feature_base + feature_offset] = 1.0;
                }
            }

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
            max_features,
            feature_indices: Tensor::from_vec(feature_indices, (batch_size, max_features), device)?,
            feature_mask: Tensor::from_vec(feature_mask, (batch_size, max_features, 1), device)?,
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
            value_targets: Tensor::from_vec(value_targets, (batch_size, VALUE_LOGITS), device)?,
            values: Tensor::from_vec(values, batch_size, device)?,
        })
    }
}

impl GpuVars {
    fn from_model(model: &AzNnue, device: &Device) -> CandleResult<Self> {
        let arch = model.arch;
        let hidden = arch.hidden_size;
        let channels = arch.gnn_node_channels;
        let pool = arch.node_pool_size();
        let value_hidden = arch.value_hidden_size;
        let proj = arch.policy_node_proj_size;
        Ok(Self {
            input_hidden: var_from_slice(&model.input_hidden, (V4_INPUT_SIZE, hidden), device)?,
            hidden_bias: var_from_slice(&model.hidden_bias, hidden, device)?,
            node_input_weights: var_from_slice(
                &model.node_input_weights,
                (channels, BOARD_CHANNELS),
                device,
            )?,
            node_input_bias: var_from_slice(&model.node_input_bias, channels, device)?,
            graph_self_gate: var_from_slice(
                &model.graph_self_gate,
                arch.gnn_node_layers * channels,
                device,
            )?,
            graph_local_gate: var_from_slice(
                &model.graph_local_gate,
                arch.gnn_node_layers * channels,
                device,
            )?,
            graph_row_gate: var_from_slice(
                &model.graph_row_gate,
                arch.gnn_node_layers * channels,
                device,
            )?,
            graph_col_gate: var_from_slice(
                &model.graph_col_gate,
                arch.gnn_node_layers * channels,
                device,
            )?,
            graph_bias: var_from_slice(&model.graph_bias, arch.gnn_node_layers * channels, device)?,
            node_attention_query: var_from_slice(&model.node_attention_query, channels, device)?,
            node_hidden: var_from_slice(&model.node_hidden, (hidden, pool), device)?,
            node_hidden_bias: var_from_slice(&model.node_hidden_bias, hidden, device)?,
            value_intermediate_hidden: var_from_slice(
                &model.value_intermediate_hidden,
                (value_hidden, hidden),
                device,
            )?,
            value_intermediate_nodes: var_from_slice(
                &model.value_intermediate_nodes,
                (value_hidden, pool),
                device,
            )?,
            value_intermediate_bias: var_from_slice(
                &model.value_intermediate_bias,
                value_hidden,
                device,
            )?,
            value_logits_weights: var_from_slice(
                &model.value_logits_weights,
                (VALUE_LOGITS, value_hidden),
                device,
            )?,
            value_logits_bias: var_from_slice(&model.value_logits_bias, VALUE_LOGITS, device)?,
            policy_node_query: var_from_slice(&model.policy_node_query, (proj, channels), device)?,
            policy_node_key: var_from_slice(&model.policy_node_key, (proj, channels), device)?,
            policy_from_bias: var_from_slice(&model.policy_from_bias, channels, device)?,
            policy_to_bias: var_from_slice(&model.policy_to_bias, channels, device)?,
            policy_move_bias: var_from_slice(&model.policy_move_bias, DENSE_MOVE_SPACE, device)?,
            policy_context_hidden: var_from_slice(
                &model.policy_context_hidden,
                (POLICY_CONTEXT_SIZE, hidden),
                device,
            )?,
            policy_context_nodes: var_from_slice(
                &model.policy_context_nodes,
                (POLICY_CONTEXT_SIZE, pool),
                device,
            )?,
            policy_context_bias: var_from_slice(
                &model.policy_context_bias,
                POLICY_CONTEXT_SIZE,
                device,
            )?,
            policy_feature_hidden: var_from_slice(
                &model.policy_feature_hidden,
                (POLICY_CONDITION_SIZE, POLICY_CONTEXT_SIZE),
                device,
            )?,
            policy_feature_nodes: var_from_slice(
                &model.policy_feature_nodes,
                (POLICY_CONDITION_SIZE, pool),
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
            policy_from_map: Tensor::from_vec(
                policy_move_from_indices(),
                DENSE_MOVE_SPACE,
                device,
            )?,
            policy_to_map: Tensor::from_vec(policy_move_to_indices(), DENSE_MOVE_SPACE, device)?,
            graph_local_kernel: Tensor::from_vec(
                graph_local_kernel_values(channels),
                (channels, channels, 3, 3),
                device,
            )?,
            graph_local_count: Tensor::from_vec(
                graph_local_count_values(),
                (1, 1, BOARD_RANKS, BOARD_FILES),
                device,
            )?,
        })
    }

    fn all_vars(&self) -> Vec<Var> {
        let mut vars = Vec::new();
        vars.push(self.input_hidden.clone());
        vars.push(self.hidden_bias.clone());
        vars.push(self.node_input_weights.clone());
        vars.push(self.node_input_bias.clone());
        vars.push(self.graph_self_gate.clone());
        vars.push(self.graph_local_gate.clone());
        vars.push(self.graph_row_gate.clone());
        vars.push(self.graph_col_gate.clone());
        vars.push(self.graph_bias.clone());
        vars.push(self.node_attention_query.clone());
        vars.push(self.node_hidden.clone());
        vars.push(self.node_hidden_bias.clone());
        vars.push(self.value_intermediate_hidden.clone());
        vars.push(self.value_intermediate_nodes.clone());
        vars.push(self.value_intermediate_bias.clone());
        vars.push(self.value_logits_weights.clone());
        vars.push(self.value_logits_bias.clone());
        vars.push(self.policy_node_query.clone());
        vars.push(self.policy_node_key.clone());
        vars.push(self.policy_from_bias.clone());
        vars.push(self.policy_to_bias.clone());
        vars.push(self.policy_move_bias.clone());
        vars.push(self.policy_context_hidden.clone());
        vars.push(self.policy_context_nodes.clone());
        vars.push(self.policy_context_bias.clone());
        vars.push(self.policy_feature_hidden.clone());
        vars.push(self.policy_feature_nodes.clone());
        vars.push(self.policy_feature_bias.clone());
        vars
    }

    fn shared_vars(&self) -> Vec<Var> {
        let mut vars = Vec::new();
        vars.push(self.input_hidden.clone());
        vars.push(self.hidden_bias.clone());
        vars.push(self.node_input_weights.clone());
        vars.push(self.node_input_bias.clone());
        vars.push(self.graph_self_gate.clone());
        vars.push(self.graph_local_gate.clone());
        vars.push(self.graph_row_gate.clone());
        vars.push(self.graph_col_gate.clone());
        vars.push(self.graph_bias.clone());
        vars.push(self.node_attention_query.clone());
        vars.push(self.node_hidden.clone());
        vars.push(self.node_hidden_bias.clone());
        vars
    }

    fn value_head_vars(&self) -> Vec<Var> {
        vec![
            self.value_intermediate_hidden.clone(),
            self.value_intermediate_nodes.clone(),
            self.value_intermediate_bias.clone(),
            self.value_logits_weights.clone(),
            self.value_logits_bias.clone(),
        ]
    }

    fn policy_head_vars(&self) -> Vec<Var> {
        vec![
            self.policy_node_query.clone(),
            self.policy_node_key.clone(),
            self.policy_from_bias.clone(),
            self.policy_to_bias.clone(),
            self.policy_move_bias.clone(),
            self.policy_context_hidden.clone(),
            self.policy_context_nodes.clone(),
            self.policy_context_bias.clone(),
            self.policy_feature_hidden.clone(),
            self.policy_feature_nodes.clone(),
            self.policy_feature_bias.clone(),
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
        copy_var(&self.hidden_bias, &mut model.hidden_bias)?;
        copy_var(&self.node_input_weights, &mut model.node_input_weights)?;
        copy_var(&self.node_input_bias, &mut model.node_input_bias)?;
        copy_var(&self.graph_self_gate, &mut model.graph_self_gate)?;
        copy_var(&self.graph_local_gate, &mut model.graph_local_gate)?;
        copy_var(&self.graph_row_gate, &mut model.graph_row_gate)?;
        copy_var(&self.graph_col_gate, &mut model.graph_col_gate)?;
        copy_var(&self.graph_bias, &mut model.graph_bias)?;
        copy_var(&self.node_attention_query, &mut model.node_attention_query)?;
        copy_var(&self.node_hidden, &mut model.node_hidden)?;
        copy_var(&self.node_hidden_bias, &mut model.node_hidden_bias)?;
        copy_var(
            &self.value_intermediate_hidden,
            &mut model.value_intermediate_hidden,
        )?;
        copy_var(
            &self.value_intermediate_nodes,
            &mut model.value_intermediate_nodes,
        )?;
        copy_var(
            &self.value_intermediate_bias,
            &mut model.value_intermediate_bias,
        )?;
        copy_var(&self.value_logits_weights, &mut model.value_logits_weights)?;
        copy_var(&self.value_logits_bias, &mut model.value_logits_bias)?;
        copy_var(&self.policy_node_query, &mut model.policy_node_query)?;
        copy_var(&self.policy_node_key, &mut model.policy_node_key)?;
        copy_var(&self.policy_from_bias, &mut model.policy_from_bias)?;
        copy_var(&self.policy_to_bias, &mut model.policy_to_bias)?;
        copy_var(&self.policy_move_bias, &mut model.policy_move_bias)?;
        copy_var(
            &self.policy_context_hidden,
            &mut model.policy_context_hidden,
        )?;
        copy_var(&self.policy_context_nodes, &mut model.policy_context_nodes)?;
        copy_var(&self.policy_context_bias, &mut model.policy_context_bias)?;
        copy_var(
            &self.policy_feature_hidden,
            &mut model.policy_feature_hidden,
        )?;
        copy_var(&self.policy_feature_nodes, &mut model.policy_feature_nodes)?;
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

/// 多卡时「每张卡至少几个样本才参与切分」；为 1 时等价于不额外限制（仅受 batch 样本数与卡数约束）。
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

/// 构造 `[channels, channels, 3, 3]` 的固定卷积核：仅在 (c, c, 3, 3) 对角通道位置上、
/// 除中心 (1,1) 外的 8 个邻域位置写 1.0，相当于按通道分别求 8 邻域之和。
fn graph_local_kernel_values(channels: usize) -> Vec<f32> {
    let mut values = vec![0.0f32; channels * channels * 3 * 3];
    for channel in 0..channels {
        for kr in 0..3 {
            for kf in 0..3 {
                if kr == 1 && kf == 1 {
                    continue;
                }
                let idx = ((channel * channels + channel) * 3 + kr) * 3 + kf;
                values[idx] = 1.0;
            }
        }
    }
    values
}

fn graph_local_count_values() -> Vec<f32> {
    let mut values = vec![0.0f32; BOARD_RANKS * BOARD_FILES];
    for rank in 0..BOARD_RANKS {
        for file in 0..BOARD_FILES {
            let mut count = 0.0f32;
            for dr in -1i32..=1 {
                for df in -1i32..=1 {
                    if dr == 0 && df == 0 {
                        continue;
                    }
                    let nr = rank as i32 + dr;
                    let nf = file as i32 + df;
                    if (0..BOARD_RANKS as i32).contains(&nr)
                        && (0..BOARD_FILES as i32).contains(&nf)
                    {
                        count += 1.0;
                    }
                }
            }
            values[rank * BOARD_FILES + file] = count.max(1.0);
        }
    }
    values
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
