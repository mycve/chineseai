use matrixmultiply::sgemm;

use super::optim::{ADAMW_WEIGHT_DECAY, AdamWState, adamw_update};
use super::{
    AzGrad, AzNnue, AzTrainStats, AzTrainingSample, BOARD_CHANNELS, BOARD_PLANES_SIZE,
    CNN_CHANNELS, CNN_KERNEL_AREA, CNN_POOLED_SIZE, GLOBAL_CONTEXT_SIZE, RESIDUAL_TRUNK_SCALE,
    SplitMix64, VALUE_HIDDEN_SIZE, VALUE_LOGITS, add_scaled, scalar_to_wdl_target, softmax_fixed,
    softmax_slice,
};
use crate::xiangqi::BOARD_FILES;

pub fn train_samples(
    model: &mut AzNnue,
    samples: &[AzTrainingSample],
    epochs: usize,
    lr: f32,
    batch_size: usize,
    rng: &mut SplitMix64,
) -> AzTrainStats {
    if samples.is_empty() || epochs == 0 || lr <= 0.0 {
        return AzTrainStats::default();
    }

    let mut optimizer = match model.optimizer.take() {
        Some(optimizer) if optimizer.matches(model) => optimizer,
        _ => Box::new(AdamWState::new(model)),
    };
    let mut gradient = AzGrad::new(model);
    let mut order = (0..samples.len()).collect::<Vec<_>>();
    let mut stats = AzTrainStats::default();
    let batch_size = batch_size.max(1);
    for _ in 0..epochs {
        shuffle(&mut order, rng);
        stats = AzTrainStats::default();
        for batch in order.chunks(batch_size) {
            let batch_stats = train_batch(model, &mut optimizer, &mut gradient, samples, batch, lr);
            stats.add_assign(&batch_stats);
        }
    }
    model.optimizer = Some(optimizer);
    if stats.samples > 0 {
        let denom = stats.samples as f32;
        stats.loss /= denom;
        stats.value_loss /= denom;
        stats.policy_ce /= denom;
    }
    stats
}

fn train_batch(
    model: &mut AzNnue,
    optimizer: &mut AdamWState,
    gradient: &mut AzGrad,
    samples: &[AzTrainingSample],
    batch: &[usize],
    lr: f32,
) -> AzTrainStats {
    gradient.clear();
    let cache = train_batch_forward_cache(model, samples, batch);
    let stats = accumulate_batch_cached(model, gradient, samples, batch, &cache);
    if stats.samples > 0 {
        apply_adamw_gradient(model, optimizer, gradient, lr, stats.samples as f32);
    }
    stats
}

fn accumulate_batch_cached(
    model: &AzNnue,
    gradient: &mut AzGrad,
    samples: &[AzTrainingSample],
    batch: &[usize],
    cache: &TrainBatchCache,
) -> AzTrainStats {
    let batch_size = batch.len();
    let mut stats = AzTrainStats::default();
    let hidden_all = cache
        .activations
        .last()
        .expect("at least one activation exists");
    let cnn_global_all = &cache.cnn_global;
    let global_all = &cache.global;
    let mut activation_grads = vec![0.0; batch_size * model.hidden_size];
    let mut cnn_global_grads = vec![0.0; batch_size * CNN_POOLED_SIZE];
    let mut global_grads = vec![0.0; batch_size * GLOBAL_CONTEXT_SIZE];
    let mut value_logit_grads = vec![0.0; batch_size * VALUE_LOGITS];
    let policy_layout = build_policy_batch_layout(samples, batch);
    let mut policy_logits = vec![0.0; policy_layout.move_indices.len()];
    let mut policy_probs = vec![0.0; policy_layout.move_indices.len()];

    for (row, &sample_index) in batch.iter().enumerate() {
        let sample = &samples[sample_index];
        let value_probs: [f32; VALUE_LOGITS] = softmax_fixed(cache.row_value_logits(row));
        let value = if value_probs.len() >= VALUE_LOGITS {
            value_probs[0] - value_probs[2]
        } else {
            0.0
        };
        let value_target = scalar_to_wdl_target(sample.value);
        let value_error = value - sample.value;
        let value_loss = value_error * value_error;
        let value_train_loss = value_probs
            .iter()
            .zip(value_target.iter())
            .map(|(predicted, target)| -target * predicted.max(1e-9).ln())
            .sum::<f32>();

        stats.loss += value_train_loss;
        stats.value_loss += value_loss;
        stats.value_pred_sum += value;
        stats.value_pred_sq_sum += value * value;
        stats.value_target_sum += sample.value;
        stats.value_target_sq_sum += sample.value * sample.value;
        stats.samples += 1;

        for out in 0..VALUE_LOGITS {
            value_logit_grads[row * VALUE_LOGITS + out] = value_probs[out] - value_target[out];
        }
    }

    grad_weights_batch(
        &mut gradient.value_logits_weights,
        &value_logit_grads,
        VALUE_LOGITS,
        &cache.value_intermediate,
        VALUE_HIDDEN_SIZE,
        batch_size,
    );
    add_bias_grad(
        &mut gradient.value_logits_bias,
        &value_logit_grads,
        batch_size,
        VALUE_LOGITS,
    );

    let mut intermediate_grads = batch_times_weights(
        &value_logit_grads,
        batch_size,
        VALUE_LOGITS,
        &model.value_logits_weights,
        VALUE_HIDDEN_SIZE,
    );
    apply_relu_mask_and_clamp(
        &mut intermediate_grads,
        &cache.value_intermediate_pre,
        -4.0,
        4.0,
    );

    grad_weights_batch(
        &mut gradient.value_intermediate_hidden,
        &intermediate_grads,
        VALUE_HIDDEN_SIZE,
        hidden_all,
        model.hidden_size,
        batch_size,
    );
    add_bias_grad(
        &mut gradient.value_intermediate_bias,
        &intermediate_grads,
        batch_size,
        VALUE_HIDDEN_SIZE,
    );
    add_batch_times_weights(
        &mut activation_grads,
        &intermediate_grads,
        batch_size,
        VALUE_HIDDEN_SIZE,
        &model.value_intermediate_hidden,
        model.hidden_size,
    );
    compute_policy_batch_logits(
        model,
        hidden_all,
        cnn_global_all,
        &policy_layout,
        &mut policy_logits,
    );
    compute_policy_batch_probs(&policy_layout, &policy_logits, &mut policy_probs);

    for row in 0..batch_size {
        let policy_range = policy_layout.sample_range(row);
        let policy_ce = policy_probs[policy_range.clone()]
            .iter()
            .zip(policy_layout.targets[policy_range.clone()].iter())
            .map(|(predicted, target)| -target * predicted.max(1e-9).ln())
            .sum::<f32>();
        stats.loss += policy_ce;
        stats.policy_ce += policy_ce;

        let activation_grad = row_slice_mut(&mut activation_grads, row, model.hidden_size);
        let hidden = row_slice(hidden_all, row, model.hidden_size);
        let cnn_global = row_slice(cnn_global_all, row, CNN_POOLED_SIZE);
        for flat_index in policy_range {
            let move_index = policy_layout.move_indices[flat_index];
            let policy_grad =
                (policy_probs[flat_index] - policy_layout.targets[flat_index]).clamp(-4.0, 4.0);
            let hidden_offset = move_index * model.hidden_size;
            let hidden_row =
                &model.policy_move_hidden[hidden_offset..hidden_offset + model.hidden_size];
            let hidden_grad_row =
                &mut gradient.policy_move_hidden[hidden_offset..hidden_offset + model.hidden_size];
            add_scaled(activation_grad, hidden_row, policy_grad);
            add_scaled(hidden_grad_row, hidden, policy_grad);
            let cnn_offset = move_index * CNN_POOLED_SIZE;
            let cnn_row = &model.policy_move_cnn[cnn_offset..cnn_offset + CNN_POOLED_SIZE];
            let cnn_grad_row =
                &mut gradient.policy_move_cnn[cnn_offset..cnn_offset + CNN_POOLED_SIZE];
            add_scaled(
                &mut cnn_global_grads[row * CNN_POOLED_SIZE..(row + 1) * CNN_POOLED_SIZE],
                cnn_row,
                policy_grad,
            );
            add_scaled(cnn_grad_row, cnn_global, policy_grad);
            gradient.policy_move_bias[move_index] += policy_grad;
        }
    }

    let mut input_grads = activation_grads;
    for layer in (0..model.trunk_depth).rev() {
        let input = &cache.activations[layer];
        let output = &cache.activations[layer + 1];
        let weight_offset = layer * model.hidden_size * model.hidden_size;
        let global_weight_offset = layer * model.hidden_size * GLOBAL_CONTEXT_SIZE;
        let bias_offset = layer * model.hidden_size;

        clamp_inplace(&mut input_grads, -4.0, 4.0);
        let mut residual_grads = vec![0.0; batch_size * model.hidden_size];
        for idx in 0..residual_grads.len() {
            if output[idx] > input[idx] {
                residual_grads[idx] = input_grads[idx] * RESIDUAL_TRUNK_SCALE;
            }
        }
        let mut previous_grads = input_grads.clone();
        grad_weights_batch(
            &mut gradient.trunk_weights
                [weight_offset..weight_offset + model.hidden_size * model.hidden_size],
            &residual_grads,
            model.hidden_size,
            input,
            model.hidden_size,
            batch_size,
        );
        grad_weights_batch(
            &mut gradient.trunk_global_weights[global_weight_offset
                ..global_weight_offset + model.hidden_size * GLOBAL_CONTEXT_SIZE],
            &residual_grads,
            model.hidden_size,
            global_all,
            GLOBAL_CONTEXT_SIZE,
            batch_size,
        );
        add_bias_grad(
            &mut gradient.trunk_biases[bias_offset..bias_offset + model.hidden_size],
            &residual_grads,
            batch_size,
            model.hidden_size,
        );
        add_batch_times_weights(
            &mut previous_grads,
            &residual_grads,
            batch_size,
            model.hidden_size,
            &model.trunk_weights
                [weight_offset..weight_offset + model.hidden_size * model.hidden_size],
            model.hidden_size,
        );
        add_batch_times_weights(
            &mut global_grads,
            &residual_grads,
            batch_size,
            model.hidden_size,
            &model.trunk_global_weights[global_weight_offset
                ..global_weight_offset + model.hidden_size * GLOBAL_CONTEXT_SIZE],
            GLOBAL_CONTEXT_SIZE,
        );
        input_grads = previous_grads;
    }

    let initial_hidden = &cache.activations[0];
    apply_relu_mask_and_clamp(&mut global_grads, &cache.global_pre, -4.0, 4.0);
    grad_weights_batch(
        &mut gradient.board_global,
        &global_grads,
        GLOBAL_CONTEXT_SIZE,
        cnn_global_all,
        CNN_POOLED_SIZE,
        batch_size,
    );
    grad_weights_batch(
        &mut gradient.global_hidden,
        &global_grads,
        GLOBAL_CONTEXT_SIZE,
        initial_hidden,
        model.hidden_size,
        batch_size,
    );
    add_bias_grad(
        &mut gradient.global_bias,
        &global_grads,
        batch_size,
        GLOBAL_CONTEXT_SIZE,
    );
    add_batch_times_weights(
        &mut cnn_global_grads,
        &global_grads,
        batch_size,
        GLOBAL_CONTEXT_SIZE,
        &model.board_global,
        CNN_POOLED_SIZE,
    );
    add_batch_times_weights(
        &mut input_grads,
        &global_grads,
        batch_size,
        GLOBAL_CONTEXT_SIZE,
        &model.global_hidden,
        model.hidden_size,
    );

    backprop_board_branch(model, gradient, samples, batch, cache, &cnn_global_grads);

    for (row, &sample_index) in batch.iter().enumerate() {
        let sample = &samples[sample_index];
        let initial_hidden_row = row_slice(initial_hidden, row, model.hidden_size);
        let input_grad_row = row_slice(&input_grads, row, model.hidden_size);
        let input_scale = 1.0 / (sample.features.len() as f32).sqrt().max(1.0);
        for idx in 0..model.hidden_size {
            if initial_hidden_row[idx] <= 0.0 {
                continue;
            }
            let grad = input_grad_row[idx].clamp(-4.0, 4.0);
            gradient.hidden_bias[idx] += grad;
            for &feature in &sample.features {
                gradient.input_hidden[feature * model.hidden_size + idx] += grad * input_scale;
            }
        }
    }

    stats
}

pub(super) struct TrainBatchCache {
    pub(super) activations: Vec<Vec<f32>>,
    pub(super) conv1_pre: Vec<f32>,
    pub(super) conv1: Vec<f32>,
    pub(super) conv2_pre: Vec<f32>,
    pub(super) conv2: Vec<f32>,
    pub(super) cnn_global: Vec<f32>,
    pub(super) cnn_max_indices: Vec<u16>,
    pub(super) cnn_attn_weights: Vec<f32>,
    pub(super) global_pre: Vec<f32>,
    pub(super) global: Vec<f32>,
    pub(super) value_intermediate_pre: Vec<f32>,
    pub(super) value_intermediate: Vec<f32>,
    pub(super) value_logits: Vec<f32>,
}

pub(super) struct PolicyBatchLayout {
    pub(super) sample_offsets: Vec<usize>,
    pub(super) move_indices: Vec<usize>,
    pub(super) targets: Vec<f32>,
}

impl TrainBatchCache {
    fn row_value_logits(&self, row: usize) -> &[f32] {
        row_slice(&self.value_logits, row, VALUE_LOGITS)
    }
}

impl PolicyBatchLayout {
    fn sample_range(&self, row: usize) -> std::ops::Range<usize> {
        self.sample_offsets[row]..self.sample_offsets[row + 1]
    }
}

fn row_slice(values: &[f32], row: usize, width: usize) -> &[f32] {
    let start = row * width;
    &values[start..start + width]
}

fn row_slice_mut(values: &mut [f32], row: usize, width: usize) -> &mut [f32] {
    let start = row * width;
    &mut values[start..start + width]
}

fn build_policy_batch_layout(samples: &[AzTrainingSample], batch: &[usize]) -> PolicyBatchLayout {
    let total_moves = batch
        .iter()
        .map(|&sample_index| samples[sample_index].move_indices.len())
        .sum();
    let mut sample_offsets = Vec::with_capacity(batch.len() + 1);
    let mut move_indices = Vec::with_capacity(total_moves);
    let mut targets = Vec::with_capacity(total_moves);
    sample_offsets.push(0);
    for &sample_index in batch {
        let sample = &samples[sample_index];
        move_indices.extend(sample.move_indices.iter().copied());
        targets.extend(sample.policy.iter().copied());
        sample_offsets.push(move_indices.len());
    }
    PolicyBatchLayout {
        sample_offsets,
        move_indices,
        targets,
    }
}

fn compute_policy_batch_logits(
    model: &AzNnue,
    hidden_all: &[f32],
    cnn_global_all: &[f32],
    layout: &PolicyBatchLayout,
    logits: &mut [f32],
) {
    for row in 0..(layout.sample_offsets.len() - 1) {
        let range = layout.sample_range(row);
        let hidden = row_slice(hidden_all, row, model.hidden_size);
        let cnn_global = row_slice(cnn_global_all, row, CNN_POOLED_SIZE);
        for (slot, &move_index) in logits[range.clone()]
            .iter_mut()
            .zip(layout.move_indices[range].iter())
        {
            *slot = model.policy_logit_from_hidden_index(hidden, cnn_global, move_index);
        }
    }
}

fn compute_policy_batch_probs(layout: &PolicyBatchLayout, logits: &[f32], probs: &mut [f32]) {
    for row in 0..(layout.sample_offsets.len() - 1) {
        let range = layout.sample_range(row);
        softmax_slice(logits, probs, range);
    }
}

fn clamp_inplace(values: &mut [f32], min: f32, max: f32) {
    for value in values {
        *value = value.clamp(min, max);
    }
}

fn apply_relu_mask_and_clamp(grads: &mut [f32], pre_activation: &[f32], min: f32, max: f32) {
    for idx in 0..grads.len() {
        if pre_activation[idx] <= 0.0 {
            grads[idx] = 0.0;
        } else {
            grads[idx] = grads[idx].clamp(min, max);
        }
    }
}

fn add_bias_grad(dst: &mut [f32], grads: &[f32], batch_size: usize, width: usize) {
    for row in 0..batch_size {
        for col in 0..width {
            dst[col] += grads[row * width + col];
        }
    }
}

fn grad_weights_batch(
    dst_out_in: &mut [f32],
    grads_b_out: &[f32],
    out_dim: usize,
    input_b_in: &[f32],
    in_dim: usize,
    batch_size: usize,
) {
    unsafe {
        sgemm(
            out_dim,
            batch_size,
            in_dim,
            1.0,
            grads_b_out.as_ptr(),
            1,
            out_dim as isize,
            input_b_in.as_ptr(),
            in_dim as isize,
            1,
            1.0,
            dst_out_in.as_mut_ptr(),
            in_dim as isize,
            1,
        );
    }
}

fn batch_times_weights(
    input_b_in: &[f32],
    batch_size: usize,
    in_dim: usize,
    weights_out_in: &[f32],
    out_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; batch_size * out_dim];
    add_batch_times_weights(
        &mut output,
        input_b_in,
        batch_size,
        in_dim,
        weights_out_in,
        out_dim,
    );
    output
}

fn add_batch_times_weights(
    output_b_out: &mut [f32],
    input_b_in: &[f32],
    batch_size: usize,
    in_dim: usize,
    weights_out_in: &[f32],
    out_dim: usize,
) {
    unsafe {
        sgemm(
            batch_size,
            in_dim,
            out_dim,
            1.0,
            input_b_in.as_ptr(),
            in_dim as isize,
            1,
            weights_out_in.as_ptr(),
            1,
            in_dim as isize,
            1.0,
            output_b_out.as_mut_ptr(),
            out_dim as isize,
            1,
        );
    }
}

fn train_batch_forward_cache(
    model: &AzNnue,
    samples: &[AzTrainingSample],
    batch: &[usize],
) -> TrainBatchCache {
    let batch_size = batch.len();
    let (conv1_pre, conv1, conv2_pre, conv2, cnn_global, cnn_max_indices, cnn_attn_weights) =
        board_forward_batch(model, samples, batch);
    let mut hidden = vec![0.0; batch_size * model.hidden_size];
    for row in 0..batch_size {
        let start = row * model.hidden_size;
        hidden[start..start + model.hidden_size].copy_from_slice(&model.hidden_bias);
    }
    for (row, &index) in batch.iter().enumerate() {
        let sample = &samples[index];
        let row_offset = row * model.hidden_size;
        for &feature in &sample.features {
            let weight_row =
                &model.input_hidden[feature * model.hidden_size..(feature + 1) * model.hidden_size];
            for idx in 0..model.hidden_size {
                hidden[row_offset + idx] += weight_row[idx];
            }
        }
    }
    relu_inplace(&mut hidden);

    let mut global_pre = affine_batch(
        &hidden,
        batch_size,
        model.hidden_size,
        &model.global_hidden,
        GLOBAL_CONTEXT_SIZE,
        &model.global_bias,
    );
    add_affine_batch(
        &mut global_pre,
        &cnn_global,
        batch_size,
        CNN_CHANNELS,
        &model.board_global,
        GLOBAL_CONTEXT_SIZE,
    );
    let mut global = global_pre.clone();
    relu_inplace(&mut global);

    let mut activations = Vec::with_capacity(model.trunk_depth + 1);
    activations.push(hidden);
    for layer in 0..model.trunk_depth {
        let previous = activations.last().expect("previous activation exists");
        let weight_offset = layer * model.hidden_size * model.hidden_size;
        let global_weight_offset = layer * model.hidden_size * GLOBAL_CONTEXT_SIZE;
        let bias_offset = layer * model.hidden_size;
        let mut next = affine_batch(
            previous,
            batch_size,
            model.hidden_size,
            &model.trunk_weights
                [weight_offset..weight_offset + model.hidden_size * model.hidden_size],
            model.hidden_size,
            &model.trunk_biases[bias_offset..bias_offset + model.hidden_size],
        );
        add_affine_batch(
            &mut next,
            &global,
            batch_size,
            GLOBAL_CONTEXT_SIZE,
            &model.trunk_global_weights[global_weight_offset
                ..global_weight_offset + model.hidden_size * GLOBAL_CONTEXT_SIZE],
            model.hidden_size,
        );
        relu_inplace(&mut next);
        for idx in 0..next.len() {
            next[idx] = previous[idx] + RESIDUAL_TRUNK_SCALE * next[idx];
        }
        activations.push(next);
    }

    let hidden = activations.last().expect("at least one activation exists");
    let value_intermediate_pre = affine_batch(
        hidden,
        batch_size,
        model.hidden_size,
        &model.value_intermediate_hidden,
        VALUE_HIDDEN_SIZE,
        &model.value_intermediate_bias,
    );
    let mut value_intermediate = value_intermediate_pre.clone();
    relu_inplace(&mut value_intermediate);
    let value_logits = affine_batch(
        &value_intermediate,
        batch_size,
        VALUE_HIDDEN_SIZE,
        &model.value_logits_weights,
        VALUE_LOGITS,
        &model.value_logits_bias,
    );

    TrainBatchCache {
        activations,
        conv1_pre,
        conv1,
        conv2_pre,
        conv2,
        cnn_global,
        cnn_max_indices,
        cnn_attn_weights,
        global_pre,
        global,
        value_intermediate_pre,
        value_intermediate,
        value_logits,
    }
}

fn board_forward_batch(
    model: &AzNnue,
    samples: &[AzTrainingSample],
    batch: &[usize],
) -> (
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<u16>,
    Vec<f32>,
) {
    let batch_size = batch.len();
    let mut conv1_pre = vec![0.0; batch_size * CNN_CHANNELS * BOARD_PLANES_SIZE];
    let mut conv1 = vec![0.0; batch_size * CNN_CHANNELS * BOARD_PLANES_SIZE];
    let mut conv2_pre = vec![0.0; batch_size * CNN_CHANNELS * BOARD_PLANES_SIZE];
    let mut conv2 = vec![0.0; batch_size * CNN_CHANNELS * BOARD_PLANES_SIZE];
    let mut cnn_global = vec![0.0; batch_size * CNN_POOLED_SIZE];
    let mut cnn_max_indices = vec![0u16; batch_size * CNN_CHANNELS];
    let mut cnn_attn_weights = vec![0.0; batch_size * BOARD_PLANES_SIZE];
    for (row, &sample_index) in batch.iter().enumerate() {
        let sample = &samples[sample_index];
        let row_conv1_pre = row_slice_mut(&mut conv1_pre, row, CNN_CHANNELS * BOARD_PLANES_SIZE);
        conv_sparse_pre(
            &sample.board,
            BOARD_CHANNELS,
            &model.board_conv1_weights,
            &model.board_conv1_bias,
            row_conv1_pre,
        );
        let row_conv1 = row_slice_mut(&mut conv1, row, CNN_CHANNELS * BOARD_PLANES_SIZE);
        row_conv1.copy_from_slice(row_conv1_pre);
        relu_inplace(row_conv1);

        let row_conv2_pre = row_slice_mut(&mut conv2_pre, row, CNN_CHANNELS * BOARD_PLANES_SIZE);
        conv_dense_pre(
            row_conv1,
            CNN_CHANNELS,
            &model.board_conv2_weights,
            &model.board_conv2_bias,
            row_conv2_pre,
        );
        let row_conv2 = row_slice_mut(&mut conv2, row, CNN_CHANNELS * BOARD_PLANES_SIZE);
        row_conv2.copy_from_slice(row_conv2_pre);
        relu_inplace(row_conv2);

        let row_global = row_slice_mut(&mut cnn_global, row, CNN_POOLED_SIZE);
        let row_attn = row_slice_mut(&mut cnn_attn_weights, row, BOARD_PLANES_SIZE);
        let mut max_logit = f32::NEG_INFINITY;
        for sq in 0..BOARD_PLANES_SIZE {
            let mut logit = 0.0;
            for channel in 0..CNN_CHANNELS {
                logit += row_conv2[channel * BOARD_PLANES_SIZE + sq]
                    * model.board_attention_query[channel];
            }
            row_attn[sq] = logit;
            max_logit = max_logit.max(logit);
        }
        let mut denom = 0.0;
        for weight in row_attn.iter_mut() {
            *weight = (*weight - max_logit).exp();
            denom += *weight;
        }
        let inv_denom = 1.0 / denom.max(1e-12);
        for weight in row_attn.iter_mut() {
            *weight *= inv_denom;
        }
        let scale = 1.0 / BOARD_PLANES_SIZE as f32;
        for channel in 0..CNN_CHANNELS {
            let start = channel * BOARD_PLANES_SIZE;
            let row_slice = &row_conv2[start..start + BOARD_PLANES_SIZE];
            let mut sum = 0.0;
            let mut max_value = 0.0;
            let mut max_index = 0usize;
            let mut attn_sum = 0.0;
            for (offset, value) in row_slice.iter().enumerate() {
                sum += *value;
                if offset == 0 || *value > max_value {
                    max_value = *value;
                    max_index = offset;
                }
                attn_sum += *value * row_attn[offset];
            }
            row_global[channel] = sum * scale;
            row_global[CNN_CHANNELS + channel] = max_value;
            row_global[CNN_CHANNELS * 2 + channel] = attn_sum;
            cnn_max_indices[row * CNN_CHANNELS + channel] = max_index as u16;
        }
    }
    (
        conv1_pre,
        conv1,
        conv2_pre,
        conv2,
        cnn_global,
        cnn_max_indices,
        cnn_attn_weights,
    )
}

fn backprop_board_branch(
    model: &AzNnue,
    gradient: &mut AzGrad,
    samples: &[AzTrainingSample],
    batch: &[usize],
    cache: &TrainBatchCache,
    cnn_global_grads: &[f32],
) {
    let pool_scale = 1.0 / BOARD_PLANES_SIZE as f32;
    for (row, &sample_index) in batch.iter().enumerate() {
        let sample = &samples[sample_index];
        let mut conv2_grads = vec![0.0; CNN_CHANNELS * BOARD_PLANES_SIZE];
        let row_cnn_global_grads = row_slice(cnn_global_grads, row, CNN_POOLED_SIZE);
        let row_conv2 = row_slice(&cache.conv2, row, CNN_CHANNELS * BOARD_PLANES_SIZE);
        let row_conv2_pre = row_slice(&cache.conv2_pre, row, CNN_CHANNELS * BOARD_PLANES_SIZE);
        let row_conv1 = row_slice(&cache.conv1, row, CNN_CHANNELS * BOARD_PLANES_SIZE);
        let row_conv1_pre = row_slice(&cache.conv1_pre, row, CNN_CHANNELS * BOARD_PLANES_SIZE);
        let row_max_indices = &cache.cnn_max_indices[row * CNN_CHANNELS..(row + 1) * CNN_CHANNELS];
        let row_attn = row_slice(&cache.cnn_attn_weights, row, BOARD_PLANES_SIZE);
        let attn_grads = &row_cnn_global_grads[CNN_CHANNELS * 2..CNN_CHANNELS * 3];
        let mut score_grads = vec![0.0; BOARD_PLANES_SIZE];
        let mut attn_query_grads = vec![0.0; CNN_CHANNELS];
        let mut attn_weight_grads_weighted_sum = 0.0;
        for sq in 0..BOARD_PLANES_SIZE {
            let mut grad = 0.0;
            for channel in 0..CNN_CHANNELS {
                let value = row_conv2[channel * BOARD_PLANES_SIZE + sq];
                grad += attn_grads[channel] * value;
                conv2_grads[channel * BOARD_PLANES_SIZE + sq] += row_attn[sq] * attn_grads[channel];
            }
            score_grads[sq] = grad;
            attn_weight_grads_weighted_sum += row_attn[sq] * grad;
        }
        for sq in 0..BOARD_PLANES_SIZE {
            let score_grad = row_attn[sq] * (score_grads[sq] - attn_weight_grads_weighted_sum);
            for channel in 0..CNN_CHANNELS {
                let idx = channel * BOARD_PLANES_SIZE + sq;
                conv2_grads[idx] += score_grad * model.board_attention_query[channel];
                attn_query_grads[channel] += score_grad * row_conv2[idx];
            }
        }
        for channel in 0..CNN_CHANNELS {
            gradient.board_attention_query[channel] += attn_query_grads[channel];
        }
        for channel in 0..CNN_CHANNELS {
            let start = channel * BOARD_PLANES_SIZE;
            for offset in 0..BOARD_PLANES_SIZE {
                let idx = start + offset;
                let mut grad = 0.0;
                grad += row_cnn_global_grads[channel] * pool_scale;
                if offset == row_max_indices[channel] as usize {
                    grad += row_cnn_global_grads[CNN_CHANNELS + channel];
                }
                conv2_grads[idx] += grad;
                if row_conv2_pre[idx] > 0.0 && row_conv2[idx] > 0.0 {
                    conv2_grads[idx] = conv2_grads[idx].clamp(-4.0, 4.0);
                } else {
                    conv2_grads[idx] = 0.0;
                }
            }
        }

        let mut conv1_grads = vec![0.0; CNN_CHANNELS * BOARD_PLANES_SIZE];
        backprop_conv_dense(
            row_conv1,
            &conv2_grads,
            &mut conv1_grads,
            &model.board_conv2_weights,
            &mut gradient.board_conv2_weights,
            &mut gradient.board_conv2_bias,
        );
        for idx in 0..conv1_grads.len() {
            if row_conv1_pre[idx] <= 0.0 {
                conv1_grads[idx] = 0.0;
            } else {
                conv1_grads[idx] = conv1_grads[idx].clamp(-4.0, 4.0);
            }
        }
        backprop_conv_sparse(
            &sample.board,
            &conv1_grads,
            &model.board_conv1_weights,
            &mut gradient.board_conv1_weights,
            &mut gradient.board_conv1_bias,
        );
    }
}

fn conv_sparse_pre(
    board: &[u8],
    input_channels: usize,
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    for out_channel in 0..CNN_CHANNELS {
        let out_start = out_channel * BOARD_PLANES_SIZE;
        for sq in 0..BOARD_PLANES_SIZE {
            let mut value = bias[out_channel];
            let file = sq % BOARD_FILES;
            let rank = sq / BOARD_FILES;
            for kr in 0..3 {
                let nr = rank as isize + kr as isize - 1;
                if !(0..10).contains(&nr) {
                    continue;
                }
                for kf in 0..3 {
                    let nf = file as isize + kf as isize - 1;
                    if !(0..BOARD_FILES as isize).contains(&nf) {
                        continue;
                    }
                    let board_index = nr as usize * BOARD_FILES + nf as usize;
                    let plane = board[board_index];
                    if plane == 0 {
                        continue;
                    }
                    let in_channel = plane as usize - 1;
                    debug_assert!(in_channel < input_channels);
                    let weight_index = ((out_channel * input_channels + in_channel)
                        * CNN_KERNEL_AREA)
                        + kr * 3
                        + kf;
                    value += weights[weight_index];
                }
            }
            output[out_start + sq] = value;
        }
    }
}

fn conv_dense_pre(
    input: &[f32],
    input_channels: usize,
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    for out_channel in 0..CNN_CHANNELS {
        let out_start = out_channel * BOARD_PLANES_SIZE;
        for sq in 0..BOARD_PLANES_SIZE {
            let mut value = bias[out_channel];
            let file = sq % BOARD_FILES;
            let rank = sq / BOARD_FILES;
            for kr in 0..3 {
                let nr = rank as isize + kr as isize - 1;
                if !(0..10).contains(&nr) {
                    continue;
                }
                for kf in 0..3 {
                    let nf = file as isize + kf as isize - 1;
                    if !(0..BOARD_FILES as isize).contains(&nf) {
                        continue;
                    }
                    let board_index = nr as usize * BOARD_FILES + nf as usize;
                    for in_channel in 0..input_channels {
                        let weight_index = ((out_channel * input_channels + in_channel)
                            * CNN_KERNEL_AREA)
                            + kr * 3
                            + kf;
                        value += input[in_channel * BOARD_PLANES_SIZE + board_index]
                            * weights[weight_index];
                    }
                }
            }
            output[out_start + sq] = value;
        }
    }
}

fn backprop_conv_dense(
    input: &[f32],
    output_grads: &[f32],
    input_grads: &mut [f32],
    weights: &[f32],
    weight_grads: &mut [f32],
    bias_grads: &mut [f32],
) {
    for out_channel in 0..CNN_CHANNELS {
        let out_start = out_channel * BOARD_PLANES_SIZE;
        for sq in 0..BOARD_PLANES_SIZE {
            let grad = output_grads[out_start + sq];
            if grad == 0.0 {
                continue;
            }
            bias_grads[out_channel] += grad;
            let file = sq % BOARD_FILES;
            let rank = sq / BOARD_FILES;
            for kr in 0..3 {
                let nr = rank as isize + kr as isize - 1;
                if !(0..10).contains(&nr) {
                    continue;
                }
                for kf in 0..3 {
                    let nf = file as isize + kf as isize - 1;
                    if !(0..BOARD_FILES as isize).contains(&nf) {
                        continue;
                    }
                    let board_index = nr as usize * BOARD_FILES + nf as usize;
                    for in_channel in 0..CNN_CHANNELS {
                        let input_index = in_channel * BOARD_PLANES_SIZE + board_index;
                        let weight_index = ((out_channel * CNN_CHANNELS + in_channel)
                            * CNN_KERNEL_AREA)
                            + kr * 3
                            + kf;
                        weight_grads[weight_index] += grad * input[input_index];
                        input_grads[input_index] += grad * weights[weight_index];
                    }
                }
            }
        }
    }
}

fn backprop_conv_sparse(
    board: &[u8],
    output_grads: &[f32],
    weights: &[f32],
    weight_grads: &mut [f32],
    bias_grads: &mut [f32],
) {
    for out_channel in 0..CNN_CHANNELS {
        let out_start = out_channel * BOARD_PLANES_SIZE;
        for sq in 0..BOARD_PLANES_SIZE {
            let grad = output_grads[out_start + sq];
            if grad == 0.0 {
                continue;
            }
            bias_grads[out_channel] += grad;
            let file = sq % BOARD_FILES;
            let rank = sq / BOARD_FILES;
            for kr in 0..3 {
                let nr = rank as isize + kr as isize - 1;
                if !(0..10).contains(&nr) {
                    continue;
                }
                for kf in 0..3 {
                    let nf = file as isize + kf as isize - 1;
                    if !(0..BOARD_FILES as isize).contains(&nf) {
                        continue;
                    }
                    let board_index = nr as usize * BOARD_FILES + nf as usize;
                    let plane = board[board_index];
                    if plane == 0 {
                        continue;
                    }
                    let in_channel = plane as usize - 1;
                    let weight_index = ((out_channel * BOARD_CHANNELS + in_channel)
                        * CNN_KERNEL_AREA)
                        + kr * 3
                        + kf;
                    let _ = weights;
                    weight_grads[weight_index] += grad;
                }
            }
        }
    }
}

fn relu_inplace(values: &mut [f32]) {
    for value in values {
        *value = value.max(0.0);
    }
}

fn affine_batch(
    input: &[f32],
    batch_size: usize,
    in_dim: usize,
    weights_out_in: &[f32],
    out_dim: usize,
    bias: &[f32],
) -> Vec<f32> {
    let mut output = vec![0.0; batch_size * out_dim];
    for row in 0..batch_size {
        output[row * out_dim..(row + 1) * out_dim].copy_from_slice(bias);
    }
    unsafe {
        sgemm(
            batch_size,
            in_dim,
            out_dim,
            1.0,
            input.as_ptr(),
            in_dim as isize,
            1,
            weights_out_in.as_ptr(),
            1,
            in_dim as isize,
            1.0,
            output.as_mut_ptr(),
            out_dim as isize,
            1,
        );
    }
    output
}

fn add_affine_batch(
    output: &mut [f32],
    input: &[f32],
    batch_size: usize,
    in_dim: usize,
    weights_out_in: &[f32],
    out_dim: usize,
) {
    unsafe {
        sgemm(
            batch_size,
            in_dim,
            out_dim,
            1.0,
            input.as_ptr(),
            in_dim as isize,
            1,
            weights_out_in.as_ptr(),
            1,
            in_dim as isize,
            1.0,
            output.as_mut_ptr(),
            out_dim as isize,
            1,
        );
    }
}

fn apply_adamw_gradient(
    model: &mut AzNnue,
    optimizer: &mut AdamWState,
    gradient: &super::AzGrad,
    lr: f32,
    batch_len: f32,
) {
    let inv_batch = 1.0 / batch_len.max(1.0);
    let (bias_correction1, bias_correction2) = optimizer.advance();

    for idx in 0..model.board_conv1_weights.len() {
        adamw_update(
            &mut model.board_conv1_weights[idx],
            &mut optimizer.board_conv1_weights_m[idx],
            &mut optimizer.board_conv1_weights_v[idx],
            gradient.board_conv1_weights[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            ADAMW_WEIGHT_DECAY,
        );
    }
    for idx in 0..model.board_conv1_bias.len() {
        adamw_update(
            &mut model.board_conv1_bias[idx],
            &mut optimizer.board_conv1_bias_m[idx],
            &mut optimizer.board_conv1_bias_v[idx],
            gradient.board_conv1_bias[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            0.0,
        );
    }
    for idx in 0..model.board_conv2_weights.len() {
        adamw_update(
            &mut model.board_conv2_weights[idx],
            &mut optimizer.board_conv2_weights_m[idx],
            &mut optimizer.board_conv2_weights_v[idx],
            gradient.board_conv2_weights[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            ADAMW_WEIGHT_DECAY,
        );
    }
    for idx in 0..model.board_conv2_bias.len() {
        adamw_update(
            &mut model.board_conv2_bias[idx],
            &mut optimizer.board_conv2_bias_m[idx],
            &mut optimizer.board_conv2_bias_v[idx],
            gradient.board_conv2_bias[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            0.0,
        );
    }
    for idx in 0..model.board_attention_query.len() {
        adamw_update(
            &mut model.board_attention_query[idx],
            &mut optimizer.board_attention_query_m[idx],
            &mut optimizer.board_attention_query_v[idx],
            gradient.board_attention_query[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            ADAMW_WEIGHT_DECAY,
        );
    }
    for idx in 0..model.board_global.len() {
        adamw_update(
            &mut model.board_global[idx],
            &mut optimizer.board_global_m[idx],
            &mut optimizer.board_global_v[idx],
            gradient.board_global[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            ADAMW_WEIGHT_DECAY,
        );
    }

    for idx in 0..model.global_hidden.len() {
        adamw_update(
            &mut model.global_hidden[idx],
            &mut optimizer.global_hidden_m[idx],
            &mut optimizer.global_hidden_v[idx],
            gradient.global_hidden[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            ADAMW_WEIGHT_DECAY,
        );
    }
    for idx in 0..model.global_bias.len() {
        adamw_update(
            &mut model.global_bias[idx],
            &mut optimizer.global_bias_m[idx],
            &mut optimizer.global_bias_v[idx],
            gradient.global_bias[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            0.0,
        );
    }
    for idx in 0..model.value_intermediate_hidden.len() {
        adamw_update(
            &mut model.value_intermediate_hidden[idx],
            &mut optimizer.value_intermediate_hidden_m[idx],
            &mut optimizer.value_intermediate_hidden_v[idx],
            gradient.value_intermediate_hidden[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            ADAMW_WEIGHT_DECAY,
        );
    }
    for idx in 0..model.value_intermediate_bias.len() {
        adamw_update(
            &mut model.value_intermediate_bias[idx],
            &mut optimizer.value_intermediate_bias_m[idx],
            &mut optimizer.value_intermediate_bias_v[idx],
            gradient.value_intermediate_bias[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            0.0,
        );
    }
    for idx in 0..model.value_logits_weights.len() {
        adamw_update(
            &mut model.value_logits_weights[idx],
            &mut optimizer.value_logits_weights_m[idx],
            &mut optimizer.value_logits_weights_v[idx],
            gradient.value_logits_weights[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            ADAMW_WEIGHT_DECAY,
        );
    }
    for idx in 0..model.value_logits_bias.len() {
        adamw_update(
            &mut model.value_logits_bias[idx],
            &mut optimizer.value_logits_bias_m[idx],
            &mut optimizer.value_logits_bias_v[idx],
            gradient.value_logits_bias[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            0.0,
        );
    }

    for idx in 0..model.policy_move_hidden.len() {
        adamw_update(
            &mut model.policy_move_hidden[idx],
            &mut optimizer.policy_move_hidden_m[idx],
            &mut optimizer.policy_move_hidden_v[idx],
            gradient.policy_move_hidden[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            ADAMW_WEIGHT_DECAY,
        );
    }
    for idx in 0..model.policy_move_cnn.len() {
        adamw_update(
            &mut model.policy_move_cnn[idx],
            &mut optimizer.policy_move_cnn_m[idx],
            &mut optimizer.policy_move_cnn_v[idx],
            gradient.policy_move_cnn[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            ADAMW_WEIGHT_DECAY,
        );
    }
    for idx in 0..model.policy_move_bias.len() {
        adamw_update(
            &mut model.policy_move_bias[idx],
            &mut optimizer.policy_move_bias_m[idx],
            &mut optimizer.policy_move_bias_v[idx],
            gradient.policy_move_bias[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            0.0,
        );
    }

    for idx in 0..model.trunk_weights.len() {
        adamw_update(
            &mut model.trunk_weights[idx],
            &mut optimizer.trunk_weights_m[idx],
            &mut optimizer.trunk_weights_v[idx],
            gradient.trunk_weights[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            ADAMW_WEIGHT_DECAY,
        );
    }
    for idx in 0..model.trunk_biases.len() {
        adamw_update(
            &mut model.trunk_biases[idx],
            &mut optimizer.trunk_biases_m[idx],
            &mut optimizer.trunk_biases_v[idx],
            gradient.trunk_biases[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            0.0,
        );
    }
    for idx in 0..model.trunk_global_weights.len() {
        adamw_update(
            &mut model.trunk_global_weights[idx],
            &mut optimizer.trunk_global_weights_m[idx],
            &mut optimizer.trunk_global_weights_v[idx],
            gradient.trunk_global_weights[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            ADAMW_WEIGHT_DECAY,
        );
    }

    for idx in 0..model.hidden_bias.len() {
        adamw_update(
            &mut model.hidden_bias[idx],
            &mut optimizer.hidden_bias_m[idx],
            &mut optimizer.hidden_bias_v[idx],
            gradient.hidden_bias[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            0.0,
        );
    }
    for idx in 0..model.input_hidden.len() {
        adamw_update(
            &mut model.input_hidden[idx],
            &mut optimizer.input_hidden_m[idx],
            &mut optimizer.input_hidden_v[idx],
            gradient.input_hidden[idx] * inv_batch,
            lr,
            bias_correction1,
            bias_correction2,
            ADAMW_WEIGHT_DECAY,
        );
    }
}

fn shuffle(values: &mut [usize], rng: &mut SplitMix64) {
    for index in (1..values.len()).rev() {
        let swap_with = (rng.next() as usize) % (index + 1);
        values.swap(index, swap_with);
    }
}
