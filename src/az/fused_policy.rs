//! 仅对候选走法计算全局上下文点积；走子、吃子增量已经并入策略表。
use super::{DENSE_MOVE_SPACE, POLICY_ACCUMULATOR_RANK, POLICY_MOVE_CONTEXT_SIZE};
use candle_core::{CpuStorage, CustomOp3, Layout, Result, Shape, Tensor};
use std::sync::OnceLock;

const MOVE_BITS: u32 = 12;
const MOVE_MASK: i64 = (1 << MOVE_BITS) - 1;
const CONTEXT_FLAG: i64 = 1 << MOVE_BITS;
const POLICY_CONTEXT_TOTAL: usize = POLICY_MOVE_CONTEXT_SIZE + POLICY_ACCUMULATOR_RANK;
const TABLE_LEN: usize = DENSE_MOVE_SPACE * POLICY_CONTEXT_TOTAL;
const _: () = assert!(POLICY_MOVE_CONTEXT_SIZE == 32 && POLICY_ACCUMULATOR_RANK == 64);
const _: () = assert!(POLICY_CONTEXT_TOTAL == 96 && TABLE_LEN == 197952);

const CUDA_SOURCE: &str = r#"
extern "C" __global__ void fused_policy_fwd(
    const float* tables, const float* context, const long long* items, float* output,
    unsigned int batch, unsigned int moves
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch * moves) return;
    unsigned long long packed = (unsigned long long)items[index];
    if (packed == 0u) { output[index] = 0.0f; return; }
    unsigned int move_index = (packed & 0xfffu) - 1u;
    float value = 0.0f;
    if ((packed & 0x1000u) != 0u) {
        for (unsigned int h = 0; h < 96u; ++h) {
            value += context[(index / moves) * 96u + h]
                   * tables[move_index * 96u + h];
        }
    }
    output[index] = value;
}

extern "C" __global__ void fused_policy_grad(
    const float* tables, const long long* items, const float* context_grad,
    float* output, unsigned int batch, unsigned int moves
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch * moves) return;
    unsigned long long packed = (unsigned long long)items[index];
    if (packed == 0u) return;
    unsigned int move_index = (packed & 0xfffu) - 1u;
    const float* context = context_grad;
    const float* grad_output = context_grad + batch * 96u;
    float* grad_tables = output;
    float* grad_context = output + 197952u;
    float g = grad_output[index];
    if ((packed & 0x1000u) == 0u) return;
    unsigned int b = index / moves;
    for (unsigned int h = 0; h < 96u; ++h) {
        unsigned int weight = move_index * 96u + h;
        unsigned int activation = b * 96u + h;
        atomicAdd(grad_tables + weight, g * context[activation]);
        atomicAdd(grad_context + activation, g * tables[weight]);
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

/// 低12位存走法编号加一，额外一位标记该走法是否有合法走子。
pub(super) const fn pack_policy_item(move_index: usize, move_valid: bool) -> i64 {
    assert!(move_index < DENSE_MOVE_SPACE);
    (move_index as i64 + 1) | ((move_valid as i64) << MOVE_BITS)
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
        for (index, &packed) in items.iter().enumerate() {
            if packed == padding_item() {
                continue;
            }
            let mv = (packed & MOVE_MASK) as usize - 1;
            let mut value = 0.0;
            if packed & CONTEXT_FLAG != 0 {
                let b = index / self.moves;
                for h in 0..POLICY_CONTEXT_TOTAL {
                    value += context[b * POLICY_CONTEXT_TOTAL + h]
                        * tables[mv * POLICY_CONTEXT_TOTAL + h];
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
        for (index, &packed) in items.iter().enumerate() {
            if packed == padding_item() {
                continue;
            }
            let mv = (packed & MOVE_MASK) as usize - 1;
            let g = grad_output[index];
            if packed & CONTEXT_FLAG == 0 {
                continue;
            }
            let b = index / op.moves;
            for h in 0..POLICY_CONTEXT_TOTAL {
                let weight = mv * POLICY_CONTEXT_TOTAL + h;
                let activation = b * POLICY_CONTEXT_TOTAL + h;
                grad_tables[weight] += g * context[activation];
                grad_context[activation] += g * tables[weight];
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
        "chineseai_fused_policy_v3",
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
        "chineseai_fused_policy_v3",
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
            pack_policy_item(3, true),
            pack_policy_item(7, true),
            padding_item(),
            pack_policy_item(3, false),
            pack_policy_item(100, true),
            pack_policy_item(7, true),
        ];
        let items = Tensor::from_slice(&items, (2, 3), device)?;
        let output = fused_policy(&tables, &context, &items)?;
        let grads = output.sum_all()?.backward()?;
        let actual = (
            output.flatten_all()?.to_vec1()?,
            grads.get(&tables).unwrap().flatten_all()?.to_vec1()?,
            grads.get(&context).unwrap().flatten_all()?.to_vec1()?,
        );
        // 用通用张量算子作独立参照，核对无偏置布局、无效项和重复走法梯度累加。
        let indices = Tensor::new(&[3u32, 7, 0, 3, 100, 7], device)?;
        let valid = Tensor::new(&[[1f32, 1., 0.], [0., 1., 1.]], device)?;
        let selected = tables
            .reshape((DENSE_MOVE_SPACE, POLICY_CONTEXT_TOTAL))?
            .index_select(&indices, 0)?
            .reshape((2, 3, POLICY_CONTEXT_TOTAL))?;
        let reference = selected
            .broadcast_mul(&context.unsqueeze(1)?)?
            .sum(2)?
            .mul(&valid)?;
        let reference_grads = reference.sum_all()?.backward()?;
        assert_close(&actual.0, &reference.flatten_all()?.to_vec1()?);
        assert_close(
            &actual.1,
            &reference_grads
                .get(&tables)
                .unwrap()
                .flatten_all()?
                .to_vec1()?,
        );
        assert_close(
            &actual.2,
            &reference_grads
                .get(&context)
                .unwrap()
                .flatten_all()?
                .to_vec1()?,
        );
        Ok(actual)
    }

    fn assert_close(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        for (&left, &right) in left.iter().zip(right) {
            assert!((left - right).abs() < 3.0e-5, "left={left} right={right}");
        }
    }

    #[test]
    fn fused_policy_cpu_matches_unfused_forward_and_gradients() -> Result<()> {
        output_and_grad(&Device::Cpu)?;
        Ok(())
    }

    #[cfg(feature = "slow-tests")]
    #[test]
    fn fused_policy_cuda_matches_cpu_forward_and_gradients() -> Result<()> {
        let cpu = output_and_grad(&Device::Cpu)?;
        let Some(cuda) = crate::az::cuda_test_device::shared_cuda_device() else {
            return Ok(());
        };
        let gpu = output_and_grad(cuda)?;
        assert_close(&cpu.0, &gpu.0);
        assert_close(&cpu.1, &gpu.1);
        assert_close(&cpu.2, &gpu.2);
        Ok(())
    }
}
