pub fn inference_simd_backend() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            if std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512dq")
            {
                return "avx512+avx2+fma-4acc";
            }
            return "avx2+fma-4acc";
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return "avx2";
        }
    }
    #[cfg(target_arch = "x86")]
    if std::arch::is_x86_feature_detected!("avx2") {
        return "avx2";
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "neon";
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        "scalar"
    }
}

pub(crate) fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    #[cfg(target_arch = "aarch64")]
    if left.len() >= 16 {
        // AArch64 guarantees NEON; avoid scalar floating-point dependency chains.
        return unsafe { dot_product_neon(left, right) };
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(target_arch = "x86_64")]
        if left.len() == 64
            && std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512dq")
            && std::arch::is_x86_feature_detected!("avx2")
        {
            // SAFETY: runtime detection above guarantees the required features.
            return unsafe { dot_product_avx512(left, right) };
        }
        #[cfg(target_arch = "x86_64")]
        if left.len() >= 64
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: runtime detection above guarantees AVX2 and FMA support.
            return unsafe { dot_product_avx2_fma(left, right) };
        }
        if left.len() >= 64 && std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime detection above guarantees AVX2 support.
            return unsafe { dot_product_avx2(left, right) };
        }
    }
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx2")]
unsafe fn dot_product_avx512(left: &[f32], right: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let chunks = left.len() / 32;
    let mut first = _mm512_setzero_ps();
    let mut second = _mm512_setzero_ps();
    for chunk in 0..chunks {
        let index = chunk * 32;
        unsafe {
            first = _mm512_fmadd_ps(
                _mm512_loadu_ps(left.as_ptr().add(index)),
                _mm512_loadu_ps(right.as_ptr().add(index)),
                first,
            );
            second = _mm512_fmadd_ps(
                _mm512_loadu_ps(left.as_ptr().add(index + 16)),
                _mm512_loadu_ps(right.as_ptr().add(index + 16)),
                second,
            );
        }
    }
    // 保留 AVX2 四路累加和八通道顺序求和，确保推理结果逐位一致。
    let sum = _mm256_add_ps(
        _mm256_add_ps(
            _mm512_castps512_ps256(first),
            _mm512_extractf32x8_ps::<1>(first),
        ),
        _mm256_add_ps(
            _mm512_castps512_ps256(second),
            _mm512_extractf32x8_ps::<1>(second),
        ),
    );
    let mut lanes = [0.0; 8];
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), sum) };
    let mut result = lanes.iter().sum::<f32>();
    for index in chunks * 32..left.len() {
        result += left[index] * right[index];
    }
    result
}

pub(crate) fn add_scaled_feature_row(
    hidden: &mut [f32],
    input_hidden: &[f32],
    hidden_size: usize,
    feature: usize,
    scale: f32,
) {
    let row = &input_hidden[feature * hidden_size..(feature + 1) * hidden_size];
    debug_assert_eq!(hidden.len(), row.len());
    #[cfg(target_arch = "aarch64")]
    if hidden_size >= 32 {
        // AArch64 guarantees NEON.
        unsafe { add_scaled_feature_row_neon(hidden, row, scale) };
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(target_arch = "x86_64")]
        if hidden_size >= 64
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: runtime detection above guarantees AVX2 and FMA support.
            unsafe { add_scaled_feature_row_avx2_fma(hidden, row, scale) };
            return;
        }
        if hidden_size >= 64 && std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime detection above guarantees AVX2 support.
            unsafe {
                add_scaled_feature_row_avx2(hidden, row, scale);
            }
            return;
        }
    }
    for (left, &right) in hidden.iter_mut().zip(row) {
        *left += scale * right;
    }
}

/// 多个特征行按原顺序更新，每个隐藏向量只读写一次。

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn dot_product_neon(left: &[f32], right: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let chunks = left.len() / 16;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    for chunk in 0..chunks {
        let index = chunk * 16;
        unsafe {
            acc0 = vfmaq_f32(
                acc0,
                vld1q_f32(left.as_ptr().add(index)),
                vld1q_f32(right.as_ptr().add(index)),
            );
            acc1 = vfmaq_f32(
                acc1,
                vld1q_f32(left.as_ptr().add(index + 4)),
                vld1q_f32(right.as_ptr().add(index + 4)),
            );
            acc2 = vfmaq_f32(
                acc2,
                vld1q_f32(left.as_ptr().add(index + 8)),
                vld1q_f32(right.as_ptr().add(index + 8)),
            );
            acc3 = vfmaq_f32(
                acc3,
                vld1q_f32(left.as_ptr().add(index + 12)),
                vld1q_f32(right.as_ptr().add(index + 12)),
            );
        }
    }
    let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    for index in (chunks * 16)..left.len() {
        sum += left[index] * right[index];
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn add_scaled_feature_row_neon(hidden: &mut [f32], row: &[f32], scale: f32) {
    use std::arch::aarch64::*;
    let scale_vector = vdupq_n_f32(scale);
    let chunks = hidden.len() / 4;
    for chunk in 0..chunks {
        let index = chunk * 4;
        unsafe {
            let left = vld1q_f32(hidden.as_ptr().add(index));
            let right = vld1q_f32(row.as_ptr().add(index));
            vst1q_f32(
                hidden.as_mut_ptr().add(index),
                vfmaq_f32(left, right, scale_vector),
            );
        }
    }
    for index in (chunks * 4)..hidden.len() {
        hidden[index] += row[index] * scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot_product_avx2_fma(left: &[f32], right: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let chunks = left.len() / 32;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    for chunk in 0..chunks {
        let index = chunk * 32;
        unsafe {
            acc0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(left.as_ptr().add(index)),
                _mm256_loadu_ps(right.as_ptr().add(index)),
                acc0,
            );
            acc1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(left.as_ptr().add(index + 8)),
                _mm256_loadu_ps(right.as_ptr().add(index + 8)),
                acc1,
            );
            acc2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(left.as_ptr().add(index + 16)),
                _mm256_loadu_ps(right.as_ptr().add(index + 16)),
                acc2,
            );
            acc3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(left.as_ptr().add(index + 24)),
                _mm256_loadu_ps(right.as_ptr().add(index + 24)),
                acc3,
            );
        }
    }
    let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let mut lanes = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
    let mut sum = lanes.iter().sum::<f32>();
    for index in (chunks * 32)..left.len() {
        sum += left[index] * right[index];
    }
    sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn dot_product_avx2(left: &[f32], right: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let chunks = left.len() / 8;
    let mut acc = _mm256_setzero_ps();
    for chunk in 0..chunks {
        let index = chunk * 8;
        unsafe {
            let l = _mm256_loadu_ps(left.as_ptr().add(index));
            let r = _mm256_loadu_ps(right.as_ptr().add(index));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(l, r));
        }
    }
    let mut lanes = [0.0f32; 8];
    unsafe {
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    }
    let mut sum = lanes.iter().sum::<f32>();
    for index in (chunks * 8)..left.len() {
        sum += left[index] * right[index];
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn add_scaled_feature_row_avx2_fma(hidden: &mut [f32], row: &[f32], scale: f32) {
    use std::arch::x86_64::*;
    let scale_scalar = scale;
    let scale = _mm256_set1_ps(scale_scalar);
    let chunks = hidden.len() / 8;
    for chunk in 0..chunks {
        let index = chunk * 8;
        unsafe {
            let left = _mm256_loadu_ps(hidden.as_ptr().add(index));
            let right = _mm256_loadu_ps(row.as_ptr().add(index));
            _mm256_storeu_ps(
                hidden.as_mut_ptr().add(index),
                _mm256_fmadd_ps(right, scale, left),
            );
        }
    }
    for index in (chunks * 8)..hidden.len() {
        hidden[index] += row[index] * scale_scalar;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn add_scaled_feature_row_avx2(hidden: &mut [f32], row: &[f32], scale: f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let scale_scalar = scale;
    let scale = _mm256_set1_ps(scale_scalar);
    let chunks = hidden.len() / 8;
    for chunk in 0..chunks {
        let index = chunk * 8;
        unsafe {
            let left = _mm256_loadu_ps(hidden.as_ptr().add(index));
            let right = _mm256_loadu_ps(row.as_ptr().add(index));
            _mm256_storeu_ps(
                hidden.as_mut_ptr().add(index),
                _mm256_add_ps(left, _mm256_mul_ps(scale, right)),
            );
        }
    }
    for index in (chunks * 8)..hidden.len() {
        hidden[index] += row[index] * scale_scalar;
    }
}

pub(crate) fn relu_in_place(values: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    if values.len() >= 32 {
        // AArch64 guarantees NEON.
        unsafe { relu_in_place_neon(values) };
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if values.len() >= 64 && std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime detection above guarantees AVX2 support.
            unsafe {
                relu_in_place_avx2(values);
            }
            return;
        }
    }
    for value in values {
        *value = value.max(0.0);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn relu_in_place_neon(values: &mut [f32]) {
    use std::arch::aarch64::*;
    let zero = vdupq_n_f32(0.0);
    let chunks = values.len() / 4;
    for chunk in 0..chunks {
        let index = chunk * 4;
        unsafe {
            let value = vld1q_f32(values.as_ptr().add(index));
            vst1q_f32(values.as_mut_ptr().add(index), vmaxq_f32(value, zero));
        }
    }
    for value in &mut values[(chunks * 4)..] {
        *value = value.max(0.0);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn relu_in_place_avx2(values: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_ps();
    let chunks = values.len() / 8;
    for chunk in 0..chunks {
        let index = chunk * 8;
        unsafe {
            let value = _mm256_loadu_ps(values.as_ptr().add(index));
            _mm256_storeu_ps(values.as_mut_ptr().add(index), _mm256_max_ps(value, zero));
        }
    }
    for value in &mut values[(chunks * 8)..] {
        *value = value.max(0.0);
    }
}

pub(crate) fn matrix_dot_into(input: &[f32], weights: &[f32], output: &mut [f32]) {
    assert_eq!(weights.len(), input.len() * output.len());
    #[cfg(target_arch = "x86_64")]
    if input.len() >= 64
        && std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512dq")
        && std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
    {
        // SAFETY: matching dimensions and CPU features are checked above.
        unsafe { matrix_dot_avx512(input, weights, output) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if input.len() >= 64
        && std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
    {
        // SAFETY: feature detection and matching matrix dimensions above.
        unsafe {
            matrix_dot_avx2_fma(input, weights, output);
        }
        return;
    }
    for (row, value) in output.iter_mut().enumerate() {
        *value = dot_product(input, &weights[row * input.len()..(row + 1) * input.len()]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx2,fma")]
unsafe fn matrix_dot_avx512(input: &[f32], weights: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;
    let width = input.len();
    let groups = output.len() / 4;
    for group in 0..groups {
        let row = group * 4;
        let mut first = [_mm512_setzero_ps(); 4];
        let mut second = [_mm512_setzero_ps(); 4];
        for chunk in 0..width / 32 {
            let index = chunk * 32;
            unsafe {
                let a = _mm512_loadu_ps(input.as_ptr().add(index));
                let b = _mm512_loadu_ps(input.as_ptr().add(index + 16));
                for offset in 0..4 {
                    let coefficients = weights.as_ptr().add((row + offset) * width + index);
                    first[offset] =
                        _mm512_fmadd_ps(a, _mm512_loadu_ps(coefficients), first[offset]);
                    second[offset] =
                        _mm512_fmadd_ps(b, _mm512_loadu_ps(coefficients.add(16)), second[offset]);
                }
            }
        }
        for offset in 0..4 {
            // 每个半区保留原 AVX2 的一路累加，再按原来的四路顺序归约。
            let a = first[offset];
            let b = second[offset];
            let sum = _mm256_add_ps(
                _mm256_add_ps(_mm512_castps512_ps256(a), _mm512_extractf32x8_ps::<1>(a)),
                _mm256_add_ps(_mm512_castps512_ps256(b), _mm512_extractf32x8_ps::<1>(b)),
            );
            let mut lanes = [0.0; 8];
            unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), sum) };
            let mut value = lanes.iter().sum::<f32>();
            for index in width / 32 * 32..width {
                value += input[index] * weights[(row + offset) * width + index];
            }
            output[row + offset] = value;
        }
    }
    for row in groups * 4..output.len() {
        output[row] =
            unsafe { dot_product_avx2_fma(input, &weights[row * width..(row + 1) * width]) };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matrix_dot_avx2_fma(input: &[f32], weights: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;
    // 两行共用输入，保留原点积的四路累加、归约和尾部计算顺序。
    let width = input.len();
    let pairs = output.len() / 2;
    for pair in 0..pairs {
        let row = pair * 2;
        let mut accumulators = [[_mm256_setzero_ps(); 4]; 2];
        for chunk in 0..width / 32 {
            for block in 0..4 {
                let index = chunk * 32 + block * 8;
                unsafe {
                    let values = _mm256_loadu_ps(input.as_ptr().add(index));
                    for offset in 0..2 {
                        let coefficients =
                            _mm256_loadu_ps(weights.as_ptr().add((row + offset) * width + index));
                        accumulators[offset][block] =
                            _mm256_fmadd_ps(values, coefficients, accumulators[offset][block]);
                    }
                }
            }
        }
        for offset in 0..2 {
            let a = accumulators[offset];
            let sum = _mm256_add_ps(_mm256_add_ps(a[0], a[1]), _mm256_add_ps(a[2], a[3]));
            let mut lanes = [0.0; 8];
            unsafe {
                _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
            }
            let mut result = lanes.iter().sum::<f32>();
            for index in width / 32 * 32..width {
                result += input[index] * weights[(row + offset) * width + index];
            }
            output[row + offset] = result;
        }
    }
    if output.len() % 2 != 0 {
        let row = output.len() - 1;
        output[row] =
            unsafe { dot_product_avx2_fma(input, &weights[row * width..(row + 1) * width]) };
    }
}

#[cfg(test)]
#[test]
fn matrix_dot_matches_existing_dot_bitwise() {
    for width in [16, 63, 64, 65, 80, 95, 96, 128, 192] {
        for rows in [1, 2, 3, 16, 96] {
            let input: Vec<f32> = (0..width)
                .map(|i| ((i * 31 % 97) as f32 - 48.0) * 0.013)
                .collect();
            let weights: Vec<f32> = (0..width * rows)
                .map(|i| ((i * 53 % 199) as f32 - 99.0) * 0.017)
                .collect();
            let mut output = vec![0.0; rows];
            matrix_dot_into(&input, &weights, &mut output);
            for row in 0..rows {
                assert_eq!(
                    output[row].to_bits(),
                    dot_product(&input, &weights[row * width..(row + 1) * width]).to_bits(),
                    "width {width} row {row}"
                );
            }
        }
    }
}
