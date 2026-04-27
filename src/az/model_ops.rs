use super::{
    BOARD_CHANNELS, BOARD_HISTORY_FRAMES, BOARD_PLANES_SIZE, CNN_KERNEL_AREA, CNN_POOL_BLOCKS,
    LINE_CONTEXT_SCALE, PIECE_BOARD_CHANNELS, VALUE_LOGIT_SCALE, VALUE_LOGITS,
};
use crate::xiangqi::{BOARD_FILES, BOARD_RANKS};

pub(super) fn conv_relu_board_layer_sparse_3x3(
    board: &[u8],
    output_channels: usize,
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    debug_assert_eq!(CNN_KERNEL_AREA, 9);
    debug_assert_eq!(
        weights.len(),
        output_channels * BOARD_CHANNELS * CNN_KERNEL_AREA
    );
    for (out_channel, bias_value) in bias.iter().copied().enumerate().take(output_channels) {
        let out_start = out_channel * BOARD_PLANES_SIZE;
        for sq in 0..BOARD_PLANES_SIZE {
            let file = sq % BOARD_FILES;
            let rank = sq / BOARD_FILES;
            let mut value = bias_value;
            for kr in 0..3 {
                let nr = rank as isize + kr as isize - 1;
                if !(0..BOARD_RANKS as isize).contains(&nr) {
                    continue;
                }
                for kf in 0..3 {
                    let nf = file as isize + kf as isize - 1;
                    if !(0..BOARD_FILES as isize).contains(&nf) {
                        continue;
                    }
                    let input_sq = nr as usize * BOARD_FILES + nf as usize;
                    let kernel_index = kr * 3 + kf;
                    for frame in 0..BOARD_HISTORY_FRAMES {
                        let plane = board[frame * BOARD_PLANES_SIZE + input_sq];
                        if plane == 0 {
                            continue;
                        }
                        let in_channel = frame * PIECE_BOARD_CHANNELS + plane as usize - 1;
                        debug_assert!(in_channel < BOARD_CHANNELS);
                        let weight_index = ((out_channel * BOARD_CHANNELS + in_channel)
                            * CNN_KERNEL_AREA)
                            + kernel_index;
                        value += weights[weight_index];
                    }
                }
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
    conv_layer_dense_generic(
        input,
        input_channels,
        output_channels,
        weights,
        bias,
        output,
        true,
    );
}

pub(super) fn conv_linear_layer_dense_generic(
    input: &[f32],
    input_channels: usize,
    output_channels: usize,
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    conv_layer_dense_generic(
        input,
        input_channels,
        output_channels,
        weights,
        bias,
        output,
        false,
    );
}

fn conv_layer_dense_generic(
    input: &[f32],
    input_channels: usize,
    output_channels: usize,
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
    relu: bool,
) {
    debug_assert_eq!(CNN_KERNEL_AREA, 9);
    for (out_channel, bias_value) in bias.iter().copied().enumerate().take(output_channels) {
        let out_start = out_channel * BOARD_PLANES_SIZE;
        for sq in 0..BOARD_PLANES_SIZE {
            let mut value = bias_value;
            let file = sq % BOARD_FILES;
            let rank = sq / BOARD_FILES;
            for kr in 0..3 {
                let nr = rank as isize + kr as isize - 1;
                if !(0..BOARD_RANKS as isize).contains(&nr) {
                    continue;
                }
                for kf in 0..3 {
                    let nf = file as isize + kf as isize - 1;
                    if !(0..BOARD_FILES as isize).contains(&nf) {
                        continue;
                    }
                    let board_index = nr as usize * BOARD_FILES + nf as usize;
                    let kernel_index = kr * 3 + kf;
                    for in_channel in 0..input_channels {
                        let weight_index = ((out_channel * input_channels + in_channel)
                            * CNN_KERNEL_AREA)
                            + kernel_index;
                        value += input[in_channel * BOARD_PLANES_SIZE + board_index]
                            * weights[weight_index];
                    }
                }
            }
            output[out_start + sq] = if relu { value.max(0.0) } else { value };
        }
    }
}

pub(super) fn conv1x1_linear_layer_dense_generic(
    input: &[f32],
    input_channels: usize,
    output_channels: usize,
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    conv1x1_layer_dense_generic(
        input,
        input_channels,
        output_channels,
        weights,
        bias,
        output,
    );
}

fn conv1x1_layer_dense_generic(
    input: &[f32],
    input_channels: usize,
    output_channels: usize,
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    debug_assert_eq!(weights.len(), output_channels * input_channels);
    for (out_channel, bias_value) in bias.iter().copied().enumerate().take(output_channels) {
        let out_start = out_channel * BOARD_PLANES_SIZE;
        let weight_row = &weights[out_channel * input_channels..(out_channel + 1) * input_channels];
        for sq in 0..BOARD_PLANES_SIZE {
            let mut value = bias_value;
            for in_channel in 0..input_channels {
                value += input[in_channel * BOARD_PLANES_SIZE + sq] * weight_row[in_channel];
            }
            output[out_start + sq] = value;
        }
    }
}

pub(super) fn residual_line_gnn_block_generic(
    features: &mut [f32],
    tmp: &mut [f32],
    delta: &mut [f32],
    channels: usize,
    weights: &[f32],
    bias: &[f32],
    line_gates: &[f32],
) {
    let layer_len = channels * channels * CNN_KERNEL_AREA;
    debug_assert_eq!(weights.len(), layer_len * 2);
    debug_assert_eq!(bias.len(), channels * 2);
    conv_relu_layer_dense_generic(
        features,
        channels,
        channels,
        &weights[..layer_len],
        &bias[..channels],
        tmp,
    );
    add_row_col_context(tmp, channels, line_gates);
    conv_linear_layer_dense_generic(
        tmp,
        channels,
        channels,
        &weights[layer_len..layer_len * 2],
        &bias[channels..channels * 2],
        delta,
    );
    for (feature, add) in features.iter_mut().zip(delta.iter()) {
        *feature = (*feature + *add).max(0.0);
    }
}

fn add_row_col_context(features: &mut [f32], channels: usize, line_gates: &[f32]) {
    debug_assert_eq!(features.len(), channels * BOARD_PLANES_SIZE);
    debug_assert_eq!(line_gates.len(), channels * 2);
    for channel in 0..channels {
        let start = channel * BOARD_PLANES_SIZE;
        let mut row_sum = [0.0f32; BOARD_RANKS];
        let mut col_sum = [0.0f32; BOARD_FILES];
        for rank in 0..BOARD_RANKS {
            for file in 0..BOARD_FILES {
                let value = features[start + rank * BOARD_FILES + file];
                row_sum[rank] += value;
                col_sum[file] += value;
            }
        }
        for rank in 0..BOARD_RANKS {
            let row_mean = row_sum[rank] / BOARD_FILES as f32;
            for file in 0..BOARD_FILES {
                let col_mean = col_sum[file] / BOARD_RANKS as f32;
                let idx = start + rank * BOARD_FILES + file;
                features[idx] = (features[idx]
                    + LINE_CONTEXT_SCALE
                        * (line_gates[channel] * row_mean
                            + line_gates[channels + channel] * col_mean))
                    .max(0.0);
            }
        }
    }
}

pub(super) fn pool_cnn_features(input: &[f32], channels: usize, output: &mut [f32]) {
    debug_assert_eq!(input.len(), channels * BOARD_PLANES_SIZE);
    debug_assert_eq!(output.len(), channels * CNN_POOL_BLOCKS);
    let scale = 1.0 / BOARD_PLANES_SIZE as f32;
    for channel in 0..channels {
        let start = channel * BOARD_PLANES_SIZE;
        let row = &input[start..start + BOARD_PLANES_SIZE];
        let mut sum = 0.0;
        let mut max_value = 0.0;
        for (idx, value) in row.iter().enumerate() {
            sum += *value;
            if idx == 0 || *value > max_value {
                max_value = *value;
            }
        }
        output[channel] = sum * scale;
        output[channels + channel] = max_value;

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
        output[channels * 2 + channel] = best_row_sum / BOARD_FILES as f32;
        output[channels * 3 + channel] = best_col_sum / BOARD_RANKS as f32;
    }
}

pub(super) fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}

pub(super) fn scalar_value_from_logits(logits: &[f32]) -> (f32, f32) {
    if logits.len() < VALUE_LOGITS {
        return (0.0, 0.0);
    }
    let raw = (logits[0] - logits[2]) * VALUE_LOGIT_SCALE;
    let value = raw.tanh();
    (value, value.abs())
}
