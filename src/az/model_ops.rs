use super::{
    BOARD_CHANNELS, BOARD_HISTORY_FRAMES, BOARD_INPUT_KERNEL_AREA, BOARD_PLANES_SIZE,
    CNN_KERNEL_AREA, CNN_POOL_BLOCKS, PIECE_BOARD_CHANNELS, VALUE_LOGITS,
};
use crate::xiangqi::{BOARD_FILES, BOARD_RANKS};

pub(super) fn conv_relu_layer_generic(
    board: &[u8],
    output_channels: usize,
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    debug_assert_eq!(BOARD_INPUT_KERNEL_AREA, 1);
    for (out_channel, bias_value) in bias.iter().copied().enumerate().take(output_channels) {
        let out_start = out_channel * BOARD_PLANES_SIZE;
        for sq in 0..BOARD_PLANES_SIZE {
            let mut value = bias_value;
            for frame in 0..BOARD_HISTORY_FRAMES {
                let plane = board[frame * BOARD_PLANES_SIZE + sq];
                if plane == 0 {
                    continue;
                }
                let in_channel = frame * PIECE_BOARD_CHANNELS + plane as usize - 1;
                debug_assert!(in_channel < BOARD_CHANNELS);
                value += weights[out_channel * BOARD_CHANNELS + in_channel];
            }
            output[out_start + sq] = value.max(0.0);
        }
    }
}

pub(super) fn conv_relu_layer_dense_generic(
    input: &[f32],
    input_channels: usize,
    output_channels: usize,
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    for (out_channel, bias_value) in bias.iter().copied().enumerate().take(output_channels) {
        let out_start = out_channel * BOARD_PLANES_SIZE;
        for sq in 0..BOARD_PLANES_SIZE {
            let mut value = bias_value;
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
            output[out_start + sq] = value.max(0.0);
        }
    }
}

pub(super) fn pool_cnn_features(
    input: &[f32],
    channels: usize,
    attention_query: &[f32],
    output: &mut [f32],
) {
    debug_assert_eq!(input.len(), channels * BOARD_PLANES_SIZE);
    debug_assert_eq!(attention_query.len(), channels);
    debug_assert_eq!(output.len(), channels * CNN_POOL_BLOCKS);
    let mut attention_logits = [0.0f32; BOARD_PLANES_SIZE];
    let mut max_logit = f32::NEG_INFINITY;
    for sq in 0..BOARD_PLANES_SIZE {
        let mut logit = 0.0;
        for channel in 0..channels {
            logit += input[channel * BOARD_PLANES_SIZE + sq] * attention_query[channel];
        }
        attention_logits[sq] = logit;
        max_logit = max_logit.max(logit);
    }
    let mut denom = 0.0;
    for logit in &mut attention_logits {
        *logit = (*logit - max_logit).exp();
        denom += *logit;
    }
    let scale = 1.0 / BOARD_PLANES_SIZE as f32;
    for channel in 0..channels {
        let start = channel * BOARD_PLANES_SIZE;
        let row = &input[start..start + BOARD_PLANES_SIZE];
        let mut sum = 0.0;
        let mut max_value = 0.0;
        let mut attn_sum = 0.0;
        for (idx, value) in row.iter().enumerate() {
            sum += *value;
            if idx == 0 || *value > max_value {
                max_value = *value;
            }
            attn_sum += (*value) * attention_logits[idx] / denom.max(1e-12);
        }
        output[channel] = sum * scale;
        output[channels + channel] = max_value;
        output[channels * 2 + channel] = attn_sum;

        let mut best_row_sum = 0.0f32;
        for rank in 0..BOARD_RANKS {
            let mut row_sum = 0.0f32;
            for file in 0..BOARD_FILES {
                row_sum += row[rank * BOARD_FILES + file];
            }
            best_row_sum = best_row_sum.max(row_sum);
        }
        let mut best_col_sum = 0.0f32;
        for file in 0..BOARD_FILES {
            let mut col_sum = 0.0f32;
            for rank in 0..BOARD_RANKS {
                col_sum += row[rank * BOARD_FILES + file];
            }
            best_col_sum = best_col_sum.max(col_sum);
        }
        output[channels * 3 + channel] = best_row_sum / BOARD_FILES as f32;
        output[channels * 4 + channel] = best_col_sum / BOARD_RANKS as f32;
    }
}

pub(super) fn apply_global_attention_feedback(
    features: &mut [f32],
    channels: usize,
    attention_query: &[f32],
    context_weights: &[f32],
    context_bias: &[f32],
    context: &mut [f32],
) {
    debug_assert_eq!(features.len(), channels * BOARD_PLANES_SIZE);
    debug_assert_eq!(attention_query.len(), channels);
    debug_assert_eq!(context_weights.len(), channels * channels);
    debug_assert_eq!(context_bias.len(), channels);
    debug_assert_eq!(context.len(), channels);

    let mut attention_logits = [0.0f32; BOARD_PLANES_SIZE];
    let mut max_logit = f32::NEG_INFINITY;
    for sq in 0..BOARD_PLANES_SIZE {
        let mut logit = 0.0;
        for channel in 0..channels {
            logit += features[channel * BOARD_PLANES_SIZE + sq] * attention_query[channel];
        }
        attention_logits[sq] = logit;
        max_logit = max_logit.max(logit);
    }

    let mut denom = 0.0;
    for logit in &mut attention_logits {
        *logit = (*logit - max_logit).exp();
        denom += *logit;
    }
    let inv_denom = 1.0 / denom.max(1e-12);
    context.fill(0.0);
    for channel in 0..channels {
        let row = &features[channel * BOARD_PLANES_SIZE..(channel + 1) * BOARD_PLANES_SIZE];
        let mut sum = 0.0;
        for (value, weight) in row.iter().zip(attention_logits.iter()) {
            sum += *value * *weight * inv_denom;
        }
        context[channel] = sum;
    }

    for out_channel in 0..channels {
        let mut delta = context_bias[out_channel];
        let row = &context_weights[out_channel * channels..(out_channel + 1) * channels];
        for (ctx, weight) in context.iter().zip(row) {
            delta += ctx * weight;
        }
        let start = out_channel * BOARD_PLANES_SIZE;
        for value in &mut features[start..start + BOARD_PLANES_SIZE] {
            *value = (*value + delta).max(0.0);
        }
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let mut values = Vec::with_capacity(logits.len());
    softmax_into(logits, &mut values);
    values
}

fn softmax_into(logits: &[f32], output: &mut Vec<f32>) {
    output.clear();
    if logits.is_empty() {
        return;
    }
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    output.reserve(logits.len());
    for &logit in logits {
        let value = (logit - max_logit).exp();
        sum += value;
        output.push(value);
    }
    let inv_sum = sum.max(1e-12).recip();
    for value in output {
        *value *= inv_sum;
    }
}

pub(super) fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;
    let chunks = left.len() / 4;
    for chunk in 0..chunks {
        let index = chunk * 4;
        sum0 += left[index] * right[index];
        sum1 += left[index + 1] * right[index + 1];
        sum2 += left[index + 2] * right[index + 2];
        sum3 += left[index + 3] * right[index + 3];
    }
    let mut sum = (sum0 + sum1) + (sum2 + sum3);
    for index in (chunks * 4)..left.len() {
        sum += left[index] * right[index];
    }
    sum
}

pub(super) fn scalar_value_from_logits(logits: &[f32]) -> (f32, Vec<f32>) {
    let probs = softmax(logits);
    if probs.len() < VALUE_LOGITS {
        return (0.0, probs);
    }
    (probs[0] - probs[2], probs)
}
