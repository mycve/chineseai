use std::sync::OnceLock;

use candle_core::{CpuStorage, CustomOp3, Layout, Result, Shape, Tensor};

use super::{
    DENSE_MOVE_SPACE, POLICY_ACCUMULATOR_RANK, POLICY_CONSEQUENCE_SIZE, POLICY_MOVE_CONTEXT_SIZE,
};
use crate::nnue::AZ_NNUE_INPUT_SIZE;

const MOVE_BITS: u32 = 12;
const FEATURE_BITS: u32 = 11;
const FEATURE_MASK: u64 = (1 << FEATURE_BITS) - 1;
const MOVE_VALID_BIT: u32 = MOVE_BITS + FEATURE_BITS * 3;
const CAPTURE_VALID_BIT: u32 = MOVE_VALID_BIT + 1;
const POLICY_VALID_BIT: u32 = CAPTURE_VALID_BIT + 1;
const INPUT_LEN: usize = AZ_NNUE_INPUT_SIZE * POLICY_CONSEQUENCE_SIZE;
const CONSEQUENCE_OFFSET: usize = INPUT_LEN;
const BIAS_OFFSET: usize = CONSEQUENCE_OFFSET + POLICY_CONSEQUENCE_SIZE;
const CONTEXT_OFFSET: usize = BIAS_OFFSET + DENSE_MOVE_SPACE;
const ACCUMULATOR_FEATURE_OFFSET: usize =
    CONTEXT_OFFSET + DENSE_MOVE_SPACE * POLICY_MOVE_CONTEXT_SIZE;
const ACCUMULATOR_MOVE_OFFSET: usize =
    ACCUMULATOR_FEATURE_OFFSET + AZ_NNUE_INPUT_SIZE * POLICY_ACCUMULATOR_RANK;
const TABLE_LEN: usize = ACCUMULATOR_MOVE_OFFSET + DENSE_MOVE_SPACE * POLICY_ACCUMULATOR_RANK;
const POLICY_CONTEXT_TOTAL: usize = POLICY_MOVE_CONTEXT_SIZE + POLICY_ACCUMULATOR_RANK;
const _: () = assert!(AZ_NNUE_INPUT_SIZE == 1260);
const _: () = assert!(
    POLICY_CONSEQUENCE_SIZE == 32
        && POLICY_MOVE_CONTEXT_SIZE == 16
        && POLICY_ACCUMULATOR_RANK == 64
);
const _: () = assert!(DENSE_MOVE_SPACE == 2086);
const _: () = assert!(CONSEQUENCE_OFFSET == 40320 && BIAS_OFFSET == 40352);
const _: () = assert!(CONTEXT_OFFSET == 42438 && ACCUMULATOR_FEATURE_OFFSET == 75814);
const _: () = assert!(ACCUMULATOR_MOVE_OFFSET == 156454 && TABLE_LEN == 289958);

const CUDA_SOURCE: &str = r#"
extern "C" __global__ void fused_policy_fwd(
    const float* tables, const float* context, const long long* items, float* output,
    unsigned int batch, unsigned int moves
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch * moves) return;
    unsigned long long packed = (unsigned long long)items[index];
    if (((packed >> 47) & 1u) == 0u) { output[index] = 0.0f; return; }
    unsigned int move_index = packed & 0xfffu;
    float value = tables[40352u + move_index];
    if (((packed >> 45) & 1u) != 0u) {
        unsigned int from = (packed >> 12) & 0x7ffu;
        unsigned int to = (packed >> 23) & 0x7ffu;
        unsigned int captured = (packed >> 34) & 0x7ffu;
        bool has_capture = ((packed >> 46) & 1u) != 0u;
        for (unsigned int h = 0; h < 32u; ++h) {
            float delta = tables[to * 32u + h] - tables[from * 32u + h];
            if (has_capture) delta -= tables[captured * 32u + h];
            value += delta * tables[40320u + h];
        }
        for (unsigned int h = 0; h < 16u; ++h) {
            value += context[(index / moves) * 80u + h]
                   * tables[42438u + move_index * 16u + h];
        }
        for (unsigned int h = 0; h < 64u; ++h) {
            float after = context[(index / moves) * 80u + 16u + h]
                        + tables[75814u + to * 64u + h]
                        - tables[75814u + from * 64u + h];
            if (has_capture) after -= tables[75814u + captured * 64u + h];
            value += after * tables[156454u + move_index * 64u + h];
        }
    }
    output[index] = value;
}

extern "C" __global__ void fused_policy_grad(
    const float* tables, const long long* items, const float* context_grad,
    float* output,
    unsigned int batch, unsigned int moves
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch * moves) return;
    unsigned long long packed = (unsigned long long)items[index];
    if (((packed >> 47) & 1u) == 0u) return;
    unsigned int move_index = packed & 0xfffu;
    const float* context = context_grad;
    const float* grad_output = context_grad + batch * 80u;
    float* grad_tables = output;
    float* grad_context = output + 289958u;
    float g = grad_output[index];
    atomicAdd(grad_tables + 40352u + move_index, g);
    if (((packed >> 45) & 1u) == 0u) return;
    unsigned int b = index / moves;
    unsigned int from = (packed >> 12) & 0x7ffu;
    unsigned int to = (packed >> 23) & 0x7ffu;
    unsigned int captured = (packed >> 34) & 0x7ffu;
    bool has_capture = ((packed >> 46) & 1u) != 0u;
    for (unsigned int h = 0; h < 32u; ++h) {
        float w = tables[40320u + h];
        float delta = tables[to * 32u + h] - tables[from * 32u + h];
        atomicAdd(grad_tables + to * 32u + h, g * w);
        atomicAdd(grad_tables + from * 32u + h, -g * w);
        if (has_capture) {
            delta -= tables[captured * 32u + h];
            atomicAdd(grad_tables + captured * 32u + h, -g * w);
        }
        atomicAdd(grad_tables + 40320u + h, g * delta);
    }
    for (unsigned int h = 0; h < 16u; ++h) {
        unsigned int context_index = 42438u + move_index * 16u + h;
        atomicAdd(grad_tables + context_index, g * context[b * 80u + h]);
        atomicAdd(grad_context + b * 80u + h, g * tables[context_index]);
    }
    for (unsigned int h = 0; h < 64u; ++h) {
        unsigned int move_factor = 156454u + move_index * 64u + h;
        float w = tables[move_factor];
        float after = context[b * 80u + 16u + h]
                    + tables[75814u + to * 64u + h]
                    - tables[75814u + from * 64u + h];
        atomicAdd(grad_context + b * 80u + 16u + h, g * w);
        atomicAdd(grad_tables + 75814u + to * 64u + h, g * w);
        atomicAdd(grad_tables + 75814u + from * 64u + h, -g * w);
        if (has_capture) {
            after -= tables[75814u + captured * 64u + h];
            atomicAdd(grad_tables + 75814u + captured * 64u + h, -g * w);
        }
        atomicAdd(grad_tables + move_factor, g * after);
    }
}
"#;

#[derive(Clone, Debug)]
struct FusedPolicy {
    batch: usize,
    moves: usize,
}

#[derive(Clone, Debug)]
struct FusedPolicyGrad(FusedPolicy);

pub(super) const fn padding_item() -> i64 {
    0
}

pub(super) const fn pack_policy_item(
    move_index: usize,
    from: usize,
    to: usize,
    captured: usize,
    move_valid: bool,
    capture_valid: bool,
) -> i64 {
    (move_index as u64
        | (from as u64) << MOVE_BITS
        | (to as u64) << (MOVE_BITS + FEATURE_BITS)
        | (captured as u64) << (MOVE_BITS + FEATURE_BITS * 2)
        | (move_valid as u64) << MOVE_VALID_BIT
        | (capture_valid as u64) << CAPTURE_VALID_BIT
        | 1u64 << POLICY_VALID_BIT) as i64
}

pub(super) fn fused_policy(tables: &Tensor, context: &Tensor, items: &Tensor) -> Result<Tensor> {
    let table_len = tables.dims1()?;
    let (batch, context_size) = context.dims2()?;
    let (item_batch, moves) = items.dims2()?;
    if table_len != TABLE_LEN || context_size != POLICY_CONTEXT_TOTAL || item_batch != batch {
        candle_core::bail!(
            "fused policy shape mismatch: tables={table_len}, context={:?}, items={:?}",
            context.dims(),
            items.dims()
        )
    }
    tables.apply_op3(context, items, FusedPolicy { batch, moves })
}

impl CustomOp3 for FusedPolicy {
    fn name(&self) -> &'static str {
        "az-fused-policy"
    }

    fn cpu_fwd(
        &self,
        table_storage: &CpuStorage,
        table_layout: &Layout,
        context_storage: &CpuStorage,
        context_layout: &Layout,
        item_storage: &CpuStorage,
        item_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (CpuStorage::F32(tables), CpuStorage::F32(context), CpuStorage::I64(items)) =
            (table_storage, context_storage, item_storage)
        else {
            candle_core::bail!("fused policy expects f32 tables/context and i64 items")
        };
        let tables = contiguous_slice(tables, table_layout, "policy tables")?;
        let context = contiguous_slice(context, context_layout, "policy context")?;
        let items = contiguous_slice(items, item_layout, "policy items")?;
        let mut output = vec![0.0f32; self.batch * self.moves];
        for index in 0..output.len() {
            let packed = items[index] as u64;
            if !flag(packed, POLICY_VALID_BIT) {
                continue;
            }
            let move_index = (packed & ((1 << MOVE_BITS) - 1)) as usize;
            let mut value = tables[BIAS_OFFSET + move_index];
            if flag(packed, MOVE_VALID_BIT) {
                let from = feature(packed, MOVE_BITS);
                let to = feature(packed, MOVE_BITS + FEATURE_BITS);
                let captured = feature(packed, MOVE_BITS + FEATURE_BITS * 2);
                let b = index / self.moves;
                for h in 0..POLICY_CONSEQUENCE_SIZE {
                    let mut delta = tables[to * POLICY_CONSEQUENCE_SIZE + h]
                        - tables[from * POLICY_CONSEQUENCE_SIZE + h];
                    if flag(packed, CAPTURE_VALID_BIT) {
                        delta -= tables[captured * POLICY_CONSEQUENCE_SIZE + h];
                    }
                    value += delta * tables[CONSEQUENCE_OFFSET + h];
                }
                for h in 0..POLICY_MOVE_CONTEXT_SIZE {
                    value += context[b * POLICY_CONTEXT_TOTAL + h]
                        * tables[CONTEXT_OFFSET + move_index * POLICY_MOVE_CONTEXT_SIZE + h];
                }
                for h in 0..POLICY_ACCUMULATOR_RANK {
                    let mut after = context
                        [b * POLICY_CONTEXT_TOTAL + POLICY_MOVE_CONTEXT_SIZE + h]
                        + tables[ACCUMULATOR_FEATURE_OFFSET + to * POLICY_ACCUMULATOR_RANK + h]
                        - tables[ACCUMULATOR_FEATURE_OFFSET + from * POLICY_ACCUMULATOR_RANK + h];
                    if flag(packed, CAPTURE_VALID_BIT) {
                        after -= tables
                            [ACCUMULATOR_FEATURE_OFFSET + captured * POLICY_ACCUMULATOR_RANK + h];
                    }
                    value += after
                        * tables
                            [ACCUMULATOR_MOVE_OFFSET + move_index * POLICY_ACCUMULATOR_RANK + h];
                }
            }
            output[index] = value;
        }
        Ok((CpuStorage::F32(output), (self.batch, self.moves).into()))
    }

    #[cfg(any(
        target_os = "windows",
        all(target_os = "linux", not(target_env = "musl"))
    ))]
    fn cuda_fwd(
        &self,
        tables: &candle_core::CudaStorage,
        table_layout: &Layout,
        context: &candle_core::CudaStorage,
        context_layout: &Layout,
        items: &candle_core::CudaStorage,
        item_layout: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        cuda_forward(
            tables,
            table_layout,
            context,
            context_layout,
            items,
            item_layout,
            self,
        )
    }

    fn bwd(
        &self,
        tables: &Tensor,
        context: &Tensor,
        items: &Tensor,
        _result: &Tensor,
        grad_result: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        let context_grad = Tensor::cat(&[&context.flatten_all()?, &grad_result.flatten_all()?], 0)?;
        let grads =
            tables.apply_op3_no_bwd(items, &context_grad, &FusedPolicyGrad(self.clone()))?;
        let grad_tables = grads.narrow(0, 0, TABLE_LEN)?;
        let grad_context = grads
            .narrow(0, TABLE_LEN, self.batch * POLICY_CONTEXT_TOTAL)?
            .reshape((self.batch, POLICY_CONTEXT_TOTAL))?;
        Ok((Some(grad_tables), Some(grad_context), None))
    }
}

impl CustomOp3 for FusedPolicyGrad {
    fn name(&self) -> &'static str {
        "az-fused-policy-grad"
    }

    fn cpu_fwd(
        &self,
        table_storage: &CpuStorage,
        table_layout: &Layout,
        item_storage: &CpuStorage,
        item_layout: &Layout,
        context_grad_storage: &CpuStorage,
        context_grad_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (CpuStorage::F32(tables), CpuStorage::I64(items), CpuStorage::F32(context_grad)) =
            (table_storage, item_storage, context_grad_storage)
        else {
            candle_core::bail!("fused policy gradient expects f32/i64/f32 inputs")
        };
        let tables = contiguous_slice(tables, table_layout, "policy tables")?;
        let items = contiguous_slice(items, item_layout, "policy items")?;
        let context_grad = contiguous_slice(
            context_grad,
            context_grad_layout,
            "policy context and gradient",
        )?;
        let op = &self.0;
        let context_len = op.batch * POLICY_CONTEXT_TOTAL;
        let (context, grad_output) = context_grad.split_at(context_len);
        let mut output = vec![0.0f32; TABLE_LEN + context_len];
        let (grad_tables, grad_context) = output.split_at_mut(TABLE_LEN);
        for index in 0..items.len() {
            let packed = items[index] as u64;
            if !flag(packed, POLICY_VALID_BIT) {
                continue;
            }
            let move_index = (packed & ((1 << MOVE_BITS) - 1)) as usize;
            let g = grad_output[index];
            grad_tables[BIAS_OFFSET + move_index] += g;
            if !flag(packed, MOVE_VALID_BIT) {
                continue;
            }
            let b = index / op.moves;
            let from = feature(packed, MOVE_BITS);
            let to = feature(packed, MOVE_BITS + FEATURE_BITS);
            let captured = feature(packed, MOVE_BITS + FEATURE_BITS * 2);
            for h in 0..POLICY_CONSEQUENCE_SIZE {
                let w = tables[CONSEQUENCE_OFFSET + h];
                let mut delta = tables[to * POLICY_CONSEQUENCE_SIZE + h]
                    - tables[from * POLICY_CONSEQUENCE_SIZE + h];
                grad_tables[to * POLICY_CONSEQUENCE_SIZE + h] += g * w;
                grad_tables[from * POLICY_CONSEQUENCE_SIZE + h] -= g * w;
                if flag(packed, CAPTURE_VALID_BIT) {
                    delta -= tables[captured * POLICY_CONSEQUENCE_SIZE + h];
                    grad_tables[captured * POLICY_CONSEQUENCE_SIZE + h] -= g * w;
                }
                grad_tables[CONSEQUENCE_OFFSET + h] += g * delta;
            }
            for h in 0..POLICY_MOVE_CONTEXT_SIZE {
                let context_index = CONTEXT_OFFSET + move_index * POLICY_MOVE_CONTEXT_SIZE + h;
                grad_tables[context_index] += g * context[b * POLICY_CONTEXT_TOTAL + h];
                grad_context[b * POLICY_CONTEXT_TOTAL + h] += g * tables[context_index];
            }
            for h in 0..POLICY_ACCUMULATOR_RANK {
                let move_factor =
                    ACCUMULATOR_MOVE_OFFSET + move_index * POLICY_ACCUMULATOR_RANK + h;
                let w = tables[move_factor];
                let context_index = b * POLICY_CONTEXT_TOTAL + POLICY_MOVE_CONTEXT_SIZE + h;
                let to_index = ACCUMULATOR_FEATURE_OFFSET + to * POLICY_ACCUMULATOR_RANK + h;
                let from_index = ACCUMULATOR_FEATURE_OFFSET + from * POLICY_ACCUMULATOR_RANK + h;
                let mut after = context[context_index] + tables[to_index] - tables[from_index];
                grad_context[context_index] += g * w;
                grad_tables[to_index] += g * w;
                grad_tables[from_index] -= g * w;
                if flag(packed, CAPTURE_VALID_BIT) {
                    let captured_index =
                        ACCUMULATOR_FEATURE_OFFSET + captured * POLICY_ACCUMULATOR_RANK + h;
                    after -= tables[captured_index];
                    grad_tables[captured_index] -= g * w;
                }
                grad_tables[move_factor] += g * after;
            }
        }
        Ok((CpuStorage::F32(output), (TABLE_LEN + context_len).into()))
    }

    #[cfg(any(
        target_os = "windows",
        all(target_os = "linux", not(target_env = "musl"))
    ))]
    fn cuda_fwd(
        &self,
        tables: &candle_core::CudaStorage,
        table_layout: &Layout,
        items: &candle_core::CudaStorage,
        item_layout: &Layout,
        context_grad: &candle_core::CudaStorage,
        context_grad_layout: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        cuda_gradient(
            tables,
            table_layout,
            items,
            item_layout,
            context_grad,
            context_grad_layout,
            &self.0,
        )
    }
}

fn feature(packed: u64, shift: u32) -> usize {
    ((packed >> shift) & FEATURE_MASK) as usize
}

fn flag(packed: u64, bit: u32) -> bool {
    ((packed >> bit) & 1) != 0
}

fn contiguous_slice<'a, T>(values: &'a [T], layout: &Layout, name: &str) -> Result<&'a [T]> {
    let Some((start, end)) = layout.contiguous_offsets() else {
        candle_core::bail!("{name} must be contiguous")
    };
    Ok(&values[start..end])
}

#[cfg(any(
    target_os = "windows",
    all(target_os = "linux", not(target_env = "musl"))
))]
fn policy_ptx() -> Result<&'static str> {
    static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    match PTX.get_or_init(|| {
        candle_core::cuda_backend::cudarc::nvrtc::safe::compile_ptx(CUDA_SOURCE)
            .map(|ptx| ptx.to_src())
            .map_err(|error| format!("failed to compile fused policy CUDA kernels: {error}"))
    }) {
        Ok(ptx) => Ok(ptx),
        Err(error) => candle_core::bail!("{error}"),
    }
}

#[cfg(any(
    target_os = "windows",
    all(target_os = "linux", not(target_env = "musl"))
))]
fn cuda_forward(
    table_storage: &candle_core::CudaStorage,
    table_layout: &Layout,
    context_storage: &candle_core::CudaStorage,
    context_layout: &Layout,
    item_storage: &candle_core::CudaStorage,
    item_layout: &Layout,
    op: &FusedPolicy,
) -> Result<(candle_core::CudaStorage, Shape)> {
    use candle_core::cuda_backend::CudaStorageSlice::{F32, I64};
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    let (F32(tables), F32(context), I64(items)) = (
        &table_storage.slice,
        &context_storage.slice,
        &item_storage.slice,
    ) else {
        candle_core::bail!("fused policy CUDA kernel expects f32/f32/i64 inputs")
    };
    let tables = cuda_view(tables, table_layout, "policy tables")?;
    let context = cuda_view(context, context_layout, "policy context")?;
    let items = cuda_view(items, item_layout, "policy items")?;
    let device = &table_storage.device;
    let output = unsafe { device.alloc::<f32>(op.batch * op.moves)? };
    let function = device.get_or_load_custom_func(
        "fused_policy_fwd",
        "chineseai_fused_policy_v2",
        policy_ptx()?,
    )?;
    let mut builder = function.builder();
    builder.arg(&tables).arg(&context).arg(&items).arg(&output);
    candle_core::builder_arg!(builder, op.batch as u32, op.moves as u32);
    unsafe { builder.launch(LaunchConfig::for_num_elems((op.batch * op.moves) as u32)) }
        .map_err(|error| candle_core::Error::Msg(format!("fused policy launch failed: {error}")))?;
    Ok((
        candle_core::CudaStorage {
            slice: F32(output),
            device: device.clone(),
        },
        (op.batch, op.moves).into(),
    ))
}

#[cfg(any(
    target_os = "windows",
    all(target_os = "linux", not(target_env = "musl"))
))]
fn cuda_gradient(
    table_storage: &candle_core::CudaStorage,
    table_layout: &Layout,
    item_storage: &candle_core::CudaStorage,
    item_layout: &Layout,
    context_grad_storage: &candle_core::CudaStorage,
    context_grad_layout: &Layout,
    op: &FusedPolicy,
) -> Result<(candle_core::CudaStorage, Shape)> {
    use candle_core::cuda_backend::CudaStorageSlice::{F32, I64};
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    let (F32(tables), I64(items), F32(context_grad)) = (
        &table_storage.slice,
        &item_storage.slice,
        &context_grad_storage.slice,
    ) else {
        candle_core::bail!("fused policy gradient CUDA kernel expects f32/i64/f32 inputs")
    };
    let tables = cuda_view(tables, table_layout, "policy tables")?;
    let items = cuda_view(items, item_layout, "policy items")?;
    let context_grad = cuda_view(context_grad, context_grad_layout, "policy context/gradient")?;
    let device = &table_storage.device;
    let output_len = TABLE_LEN + op.batch * POLICY_CONTEXT_TOTAL;
    let output = device.alloc_zeros::<f32>(output_len)?;
    let function = device.get_or_load_custom_func(
        "fused_policy_grad",
        "chineseai_fused_policy_v2",
        policy_ptx()?,
    )?;
    let mut builder = function.builder();
    builder
        .arg(&tables)
        .arg(&items)
        .arg(&context_grad)
        .arg(&output);
    candle_core::builder_arg!(builder, op.batch as u32, op.moves as u32);
    unsafe { builder.launch(LaunchConfig::for_num_elems((op.batch * op.moves) as u32)) }.map_err(
        |error| candle_core::Error::Msg(format!("fused policy gradient launch failed: {error}")),
    )?;
    Ok((
        candle_core::CudaStorage {
            slice: F32(output),
            device: device.clone(),
        },
        output_len.into(),
    ))
}

#[cfg(any(
    target_os = "windows",
    all(target_os = "linux", not(target_env = "musl"))
))]
fn cuda_view<'a, T: candle_core::cuda_backend::cudarc::driver::DeviceRepr>(
    values: &'a candle_core::cuda_backend::cudarc::driver::CudaSlice<T>,
    layout: &Layout,
    name: &str,
) -> Result<candle_core::cuda_backend::cudarc::driver::CudaView<'a, T>> {
    let Some((start, end)) = layout.contiguous_offsets() else {
        candle_core::bail!("{name} must be contiguous")
    };
    Ok(values.slice(start..end))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    fn output_and_grad(device: &Device) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let table_values = (0..TABLE_LEN)
            .map(|index| ((index % 97) as f32 - 48.0) * 0.001)
            .collect::<Vec<_>>();
        let context_values = (0..2 * POLICY_CONTEXT_TOTAL)
            .map(|index| (index as f32 - 12.0) * 0.01)
            .collect::<Vec<_>>();
        let tables = Var::from_slice(&table_values, TABLE_LEN, device)?;
        let context = Var::from_slice(&context_values, (2, POLICY_CONTEXT_TOTAL), device)?;
        let items = [
            pack_policy_item(3, 10, 11, 20, true, true),
            pack_policy_item(7, 30, 40, 0, true, false),
            padding_item(),
            pack_policy_item(3, 0, 0, 0, false, false),
            pack_policy_item(100, 500, 501, 0, true, false),
            pack_policy_item(7, 30, 40, 20, true, true),
        ];
        let items = Tensor::from_slice(&items, (2, 3), device)?;
        let output = fused_policy(&tables, &context, &items)?;
        let grads = output.sum_all()?.backward()?;
        Ok((
            output.flatten_all()?.to_vec1()?,
            grads.get(&tables).unwrap().flatten_all()?.to_vec1()?,
            grads.get(&context).unwrap().flatten_all()?.to_vec1()?,
        ))
    }

    fn assert_close(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        for (&left, &right) in left.iter().zip(right) {
            assert!((left - right).abs() < 3.0e-5, "left={left} right={right}");
        }
    }

    #[test]
    fn fused_policy_cuda_matches_cpu_forward_and_gradients() -> Result<()> {
        let cpu = output_and_grad(&Device::Cpu)?;
        let Ok(cuda) = Device::new_cuda(0) else {
            return Ok(());
        };
        let gpu = output_and_grad(&cuda)?;
        assert_close(&cpu.0, &gpu.0);
        assert_close(&cpu.1, &gpu.1);
        assert_close(&cpu.2, &gpu.2);
        Ok(())
    }
}
