use candle_core::{Device, Result as CandleResult, Tensor, Var};
use candle_nn::ops::{log_softmax, softmax};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};

use super::{
    AzNnue, AzTrainStats, AzTrainingSample, BOARD_CHANNELS, BOARD_PLANES_SIZE, CNN_CHANNELS,
    CNN_POOLED_SIZE, DENSE_MOVE_SPACE, GLOBAL_CONTEXT_SIZE, RESIDUAL_TRUNK_SCALE,
    VALUE_HIDDEN_SIZE, VALUE_LOGITS,
};
use crate::nnue::V4_INPUT_SIZE;
use crate::xiangqi::{BOARD_FILES, BOARD_RANKS};

const POLICY_MASK_VALUE: f32 = -1.0e9;
const ADAMW_WEIGHT_DECAY: f64 = 1e-4;

#[derive(Debug)]
pub(super) struct GpuTrainer {
    device: Device,
    hidden_size: usize,
    trunk_depth: usize,
    vars: GpuVars,
    optimizer: AdamW,
}

#[derive(Debug)]
struct GpuVars {
    input_hidden: Var,
    hidden_bias: Var,
    trunk_weights: Vec<Var>,
    trunk_biases: Vec<Var>,
    trunk_global_weights: Vec<Var>,
    board_conv1_weights: Var,
    board_conv1_bias: Var,
    board_conv2_weights: Var,
    board_conv2_bias: Var,
    board_attention_query: Var,
    board_hidden: Var,
    board_hidden_bias: Var,
    board_global: Var,
    global_hidden: Var,
    global_bias: Var,
    value_intermediate_hidden: Var,
    value_intermediate_bias: Var,
    value_logits_weights: Var,
    value_logits_bias: Var,
    policy_move_hidden: Var,
    policy_move_cnn: Var,
    policy_move_bias: Var,
}

pub(super) fn train_samples_gpu(
    model: &mut AzNnue,
    samples: &[AzTrainingSample],
    epochs: usize,
    lr: f32,
    batch_size: usize,
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
    let batch_size = batch_size.max(1);
    {
        let trainer = model
            .gpu_trainer
            .as_mut()
            .expect("gpu trainer was initialized");
        trainer.set_learning_rate(lr);
        for _ in 0..epochs {
            shuffle(&mut order, rng);
            stats = AzTrainStats::default();
            for batch in order.chunks(batch_size) {
                let batch_stats = trainer.train_batch(samples, batch)?;
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
        let device = Device::new_cuda(cuda_device_index())?;
        let vars = GpuVars::from_model(model, &device)?;
        let optimizer = AdamW::new(
            vars.all_vars(),
            ParamsAdamW {
                lr: lr as f64,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: ADAMW_WEIGHT_DECAY,
            },
        )?;
        Ok(Self {
            device,
            hidden_size: model.hidden_size,
            trunk_depth: model.trunk_depth,
            vars,
            optimizer,
        })
    }

    fn matches(&self, model: &AzNnue) -> bool {
        self.hidden_size == model.hidden_size && self.trunk_depth == model.trunk_depth
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.optimizer.set_learning_rate(lr as f64);
    }

    fn train_batch(
        &mut self,
        samples: &[AzTrainingSample],
        batch: &[usize],
    ) -> CandleResult<AzTrainStats> {
        let batch_tensors = BatchTensors::new(samples, batch, &self.device)?;
        let forward = self.forward(&batch_tensors)?;
        let value_probs = softmax(&forward.value_logits, 1)?;
        let win = value_probs.narrow(1, 0, 1)?;
        let loss_prob = value_probs.narrow(1, 2, 1)?;
        let value = (win - loss_prob)?.squeeze(1)?;
        let value_error = (&value - &batch_tensors.values)?;
        let value_loss_per_sample = value_error.sqr()?;
        let value_loss = value_loss_per_sample.sum_all()?;

        let masked_policy_logits = (&forward.policy_logits + &batch_tensors.policy_mask)?;
        let log_policy = log_softmax(&masked_policy_logits, 1)?;
        let policy_ce_per_sample = ((&batch_tensors.policy_targets * &log_policy)? * -1.0)?;
        let policy_ce = policy_ce_per_sample.sum_all()?;
        let loss_tensor = ((&value_loss + &policy_ce)? / batch.len() as f64)?;

        self.optimizer.backward_step(&loss_tensor)?;

        let mut stats = AzTrainStats::default();
        stats.loss = loss_tensor.to_scalar::<f32>()? * batch.len() as f32;
        stats.value_loss = value_loss.to_scalar::<f32>()?;
        stats.policy_ce = policy_ce.to_scalar::<f32>()?;
        stats.value_pred_sum = value.sum_all()?.to_scalar::<f32>()?;
        stats.value_pred_sq_sum = value.sqr()?.sum_all()?.to_scalar::<f32>()?;
        stats.value_target_sum = batch_tensors.values.sum_all()?.to_scalar::<f32>()?;
        stats.value_target_sq_sum = batch_tensors.values.sqr()?.sum_all()?.to_scalar::<f32>()?;
        stats.samples = batch.len();
        Ok(stats)
    }

    fn forward(&self, batch: &BatchTensors) -> CandleResult<ForwardOutput> {
        let bsz = batch.batch_size;
        let feature_embeddings = self
            .vars
            .input_hidden
            .index_select(&batch.feature_indices.flatten_all()?, 0)?
            .reshape((bsz, batch.max_features, self.hidden_size))?;
        let sparse_hidden = feature_embeddings
            .broadcast_mul(&batch.feature_mask)?
            .sum(1)?
            .broadcast_add(&self.vars.hidden_bias)?
            .relu()?;

        let conv1 = batch
            .board_onehot
            .conv2d(&self.vars.board_conv1_weights, 1, 1, 1, 1)?
            .broadcast_add(
                &self
                    .vars
                    .board_conv1_bias
                    .reshape((1, CNN_CHANNELS, 1, 1))?,
            )?
            .relu()?;
        let conv2 = conv1
            .conv2d(&self.vars.board_conv2_weights, 1, 1, 1, 1)?
            .broadcast_add(
                &self
                    .vars
                    .board_conv2_bias
                    .reshape((1, CNN_CHANNELS, 1, 1))?,
            )?
            .relu()?;

        let conv2_flat = conv2.reshape((bsz, CNN_CHANNELS, BOARD_PLANES_SIZE))?;
        let avg_pool = conv2_flat.mean(2)?;
        let max_pool = conv2_flat.max(2)?;
        let attn_query = self
            .vars
            .board_attention_query
            .reshape((1, CNN_CHANNELS, 1))?;
        let attn_logits = conv2_flat.broadcast_mul(&attn_query)?.sum(1)?;
        let attn_weights = softmax(&attn_logits, 1)?;
        let attn_pool = conv2_flat
            .broadcast_mul(&attn_weights.unsqueeze(1)?)?
            .sum(2)?;
        let cnn_global = Tensor::cat(&[avg_pool, max_pool, attn_pool], 1)?;

        let cnn_hidden = cnn_global
            .matmul(&self.vars.board_hidden.t()?)?
            .broadcast_add(&self.vars.board_hidden_bias)?
            .relu()?;
        let mut hidden = (sparse_hidden + cnn_hidden)?;

        let global = hidden
            .matmul(&self.vars.global_hidden.t()?)?
            .broadcast_add(
                &cnn_global
                    .matmul(&self.vars.board_global.t()?)?
                    .broadcast_add(&self.vars.global_bias)?,
            )?
            .relu()?;

        for layer in 0..self.trunk_depth {
            let dense = hidden
                .matmul(&self.vars.trunk_weights[layer].t()?)?
                .broadcast_add(
                    &global
                        .matmul(&self.vars.trunk_global_weights[layer].t()?)?
                        .broadcast_add(&self.vars.trunk_biases[layer])?,
                )?
                .relu()?;
            hidden = (hidden + (dense * RESIDUAL_TRUNK_SCALE as f64)?)?;
        }

        let value_intermediate = hidden
            .matmul(&self.vars.value_intermediate_hidden.t()?)?
            .broadcast_add(&self.vars.value_intermediate_bias)?
            .relu()?;
        let value_logits = value_intermediate
            .matmul(&self.vars.value_logits_weights.t()?)?
            .broadcast_add(&self.vars.value_logits_bias)?;
        let policy_logits = hidden
            .matmul(&self.vars.policy_move_hidden.t()?)?
            .broadcast_add(
                &cnn_global
                    .matmul(&self.vars.policy_move_cnn.t()?)?
                    .broadcast_add(&self.vars.policy_move_bias)?,
            )?;

        Ok(ForwardOutput {
            value_logits,
            policy_logits,
        })
    }

    fn copy_to_model(&self, model: &mut AzNnue) -> CandleResult<()> {
        self.vars.copy_to_model(model)
    }
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
            for sq in 0..BOARD_PLANES_SIZE {
                let plane = sample.board[sq];
                if plane > 0 {
                    let channel = plane as usize - 1;
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
            values[row] = sample.value.clamp(-1.0, 1.0);
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
            values: Tensor::from_vec(values, batch_size, device)?,
        })
    }
}

impl GpuVars {
    fn from_model(model: &AzNnue, device: &Device) -> CandleResult<Self> {
        let hidden = model.hidden_size;
        let mut trunk_weights = Vec::with_capacity(model.trunk_depth);
        let mut trunk_biases = Vec::with_capacity(model.trunk_depth);
        let mut trunk_global_weights = Vec::with_capacity(model.trunk_depth);
        for layer in 0..model.trunk_depth {
            let weight_offset = layer * hidden * hidden;
            let bias_offset = layer * hidden;
            let global_offset = layer * hidden * GLOBAL_CONTEXT_SIZE;
            trunk_weights.push(var_from_slice(
                &model.trunk_weights[weight_offset..weight_offset + hidden * hidden],
                (hidden, hidden),
                device,
            )?);
            trunk_biases.push(var_from_slice(
                &model.trunk_biases[bias_offset..bias_offset + hidden],
                hidden,
                device,
            )?);
            trunk_global_weights.push(var_from_slice(
                &model.trunk_global_weights
                    [global_offset..global_offset + hidden * GLOBAL_CONTEXT_SIZE],
                (hidden, GLOBAL_CONTEXT_SIZE),
                device,
            )?);
        }

        Ok(Self {
            input_hidden: var_from_slice(&model.input_hidden, (V4_INPUT_SIZE, hidden), device)?,
            hidden_bias: var_from_slice(&model.hidden_bias, hidden, device)?,
            trunk_weights,
            trunk_biases,
            trunk_global_weights,
            board_conv1_weights: var_from_slice(
                &model.board_conv1_weights,
                (CNN_CHANNELS, BOARD_CHANNELS, 3, 3),
                device,
            )?,
            board_conv1_bias: var_from_slice(&model.board_conv1_bias, CNN_CHANNELS, device)?,
            board_conv2_weights: var_from_slice(
                &model.board_conv2_weights,
                (CNN_CHANNELS, CNN_CHANNELS, 3, 3),
                device,
            )?,
            board_conv2_bias: var_from_slice(&model.board_conv2_bias, CNN_CHANNELS, device)?,
            board_attention_query: var_from_slice(
                &model.board_attention_query,
                CNN_CHANNELS,
                device,
            )?,
            board_hidden: var_from_slice(&model.board_hidden, (hidden, CNN_POOLED_SIZE), device)?,
            board_hidden_bias: var_from_slice(&model.board_hidden_bias, hidden, device)?,
            board_global: var_from_slice(
                &model.board_global,
                (GLOBAL_CONTEXT_SIZE, CNN_POOLED_SIZE),
                device,
            )?,
            global_hidden: var_from_slice(
                &model.global_hidden,
                (GLOBAL_CONTEXT_SIZE, hidden),
                device,
            )?,
            global_bias: var_from_slice(&model.global_bias, GLOBAL_CONTEXT_SIZE, device)?,
            value_intermediate_hidden: var_from_slice(
                &model.value_intermediate_hidden,
                (VALUE_HIDDEN_SIZE, hidden),
                device,
            )?,
            value_intermediate_bias: var_from_slice(
                &model.value_intermediate_bias,
                VALUE_HIDDEN_SIZE,
                device,
            )?,
            value_logits_weights: var_from_slice(
                &model.value_logits_weights,
                (VALUE_LOGITS, VALUE_HIDDEN_SIZE),
                device,
            )?,
            value_logits_bias: var_from_slice(&model.value_logits_bias, VALUE_LOGITS, device)?,
            policy_move_hidden: var_from_slice(
                &model.policy_move_hidden,
                (DENSE_MOVE_SPACE, hidden),
                device,
            )?,
            policy_move_cnn: var_from_slice(
                &model.policy_move_cnn,
                (DENSE_MOVE_SPACE, CNN_POOLED_SIZE),
                device,
            )?,
            policy_move_bias: var_from_slice(&model.policy_move_bias, DENSE_MOVE_SPACE, device)?,
        })
    }

    fn all_vars(&self) -> Vec<Var> {
        let mut vars = Vec::new();
        vars.push(self.input_hidden.clone());
        vars.push(self.hidden_bias.clone());
        vars.extend(self.trunk_weights.iter().cloned());
        vars.extend(self.trunk_biases.iter().cloned());
        vars.extend(self.trunk_global_weights.iter().cloned());
        vars.push(self.board_conv1_weights.clone());
        vars.push(self.board_conv1_bias.clone());
        vars.push(self.board_conv2_weights.clone());
        vars.push(self.board_conv2_bias.clone());
        vars.push(self.board_attention_query.clone());
        vars.push(self.board_hidden.clone());
        vars.push(self.board_hidden_bias.clone());
        vars.push(self.board_global.clone());
        vars.push(self.global_hidden.clone());
        vars.push(self.global_bias.clone());
        vars.push(self.value_intermediate_hidden.clone());
        vars.push(self.value_intermediate_bias.clone());
        vars.push(self.value_logits_weights.clone());
        vars.push(self.value_logits_bias.clone());
        vars.push(self.policy_move_hidden.clone());
        vars.push(self.policy_move_cnn.clone());
        vars.push(self.policy_move_bias.clone());
        vars
    }

    fn copy_to_model(&self, model: &mut AzNnue) -> CandleResult<()> {
        copy_var(&self.input_hidden, &mut model.input_hidden)?;
        copy_var(&self.hidden_bias, &mut model.hidden_bias)?;
        for layer in 0..model.trunk_depth {
            let hidden = model.hidden_size;
            let weight_offset = layer * hidden * hidden;
            let bias_offset = layer * hidden;
            let global_offset = layer * hidden * GLOBAL_CONTEXT_SIZE;
            copy_var(
                &self.trunk_weights[layer],
                &mut model.trunk_weights[weight_offset..weight_offset + hidden * hidden],
            )?;
            copy_var(
                &self.trunk_biases[layer],
                &mut model.trunk_biases[bias_offset..bias_offset + hidden],
            )?;
            copy_var(
                &self.trunk_global_weights[layer],
                &mut model.trunk_global_weights
                    [global_offset..global_offset + hidden * GLOBAL_CONTEXT_SIZE],
            )?;
        }
        copy_var(&self.board_conv1_weights, &mut model.board_conv1_weights)?;
        copy_var(&self.board_conv1_bias, &mut model.board_conv1_bias)?;
        copy_var(&self.board_conv2_weights, &mut model.board_conv2_weights)?;
        copy_var(&self.board_conv2_bias, &mut model.board_conv2_bias)?;
        copy_var(
            &self.board_attention_query,
            &mut model.board_attention_query,
        )?;
        copy_var(&self.board_hidden, &mut model.board_hidden)?;
        copy_var(&self.board_hidden_bias, &mut model.board_hidden_bias)?;
        copy_var(&self.board_global, &mut model.board_global)?;
        copy_var(&self.global_hidden, &mut model.global_hidden)?;
        copy_var(&self.global_bias, &mut model.global_bias)?;
        copy_var(
            &self.value_intermediate_hidden,
            &mut model.value_intermediate_hidden,
        )?;
        copy_var(
            &self.value_intermediate_bias,
            &mut model.value_intermediate_bias,
        )?;
        copy_var(&self.value_logits_weights, &mut model.value_logits_weights)?;
        copy_var(&self.value_logits_bias, &mut model.value_logits_bias)?;
        copy_var(&self.policy_move_hidden, &mut model.policy_move_hidden)?;
        copy_var(&self.policy_move_cnn, &mut model.policy_move_cnn)?;
        copy_var(&self.policy_move_bias, &mut model.policy_move_bias)?;
        Ok(())
    }
}

fn cuda_device_index() -> usize {
    std::env::var("CHINESEAI_CUDA_DEVICE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0)
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
        let swap_with = (rng.next() as usize) % (index + 1);
        values.swap(index, swap_with);
    }
}
