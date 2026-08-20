use std::sync::OnceLock;

use candle_core::{CpuStorage, CustomOp2, CustomOp3, Layout, Result, Shape, Tensor};

const TERMS: usize = 7;
const TACTICAL_TERMS: usize = 2;
const CUDA_SOURCE: &str = r#"
extern "C" __global__ void sparse_policy_fwd(
    const float* tables, const long long* indices, float* output, unsigned int outputs
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= outputs) return;
    float sum = 0.0f;
    #pragma unroll
    for (unsigned int i = 0; i < 7u; ++i) sum += tables[(unsigned long long)indices[index * 7u + i]];
    output[index] = sum;
}

extern "C" __global__ void sparse_policy_grad(
    const long long* indices, const float* grad_output, float* grad_tables, unsigned int entries
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= entries) return;
    atomicAdd(grad_tables + (unsigned long long)indices[index], grad_output[index / 7u]);
}

extern "C" __global__ void tactical_policy_fwd(
    const float* tables, const long long* indices, float* output, unsigned int outputs
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= outputs) return;
    output[index] = tables[(unsigned long long)indices[index * 2u]]
                  + tables[(unsigned long long)indices[index * 2u + 1u]];
}

extern "C" __global__ void tactical_policy_grad(
    const long long* indices, const float* grad_output, float* grad_tables, unsigned int entries
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= entries) return;
    atomicAdd(grad_tables + (unsigned long long)indices[index], grad_output[index / 2u]);
}
"#;

#[derive(Clone, Debug)]
struct SparsePolicy {
    table_len: usize,
    batch: usize,
    moves: usize,
    terms: usize,
}

#[derive(Clone, Debug)]
struct SparsePolicyGrad(SparsePolicy);

pub(super) fn sparse_policy(tables: &Tensor, indices: &Tensor) -> Result<Tensor> {
    sparse_table_policy(tables, indices, TERMS)
}

pub(super) fn tactical_policy(tables: &Tensor, indices: &Tensor) -> Result<Tensor> {
    sparse_table_policy(tables, indices, TACTICAL_TERMS)
}

fn sparse_table_policy(tables: &Tensor, indices: &Tensor, expected_terms: usize) -> Result<Tensor> {
    let table_len = tables.dims1()?;
    let (batch, moves, terms) = indices.dims3()?;
    if terms != expected_terms {
        candle_core::bail!("sparse policy expects {expected_terms} indices per move, got {terms}")
    }
    tables.apply_op2(
        indices,
        SparsePolicy {
            table_len,
            batch,
            moves,
            terms,
        },
    )
}

impl CustomOp2 for SparsePolicy {
    fn name(&self) -> &'static str {
        "az-sparse-policy"
    }

    fn cpu_fwd(
        &self,
        table_storage: &CpuStorage,
        table_layout: &Layout,
        index_storage: &CpuStorage,
        index_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (CpuStorage::F32(tables), CpuStorage::I64(indices)) = (table_storage, index_storage)
        else {
            candle_core::bail!("sparse policy expects f32 tables and i64 indices")
        };
        let tables = contiguous(tables, table_layout, "sparse policy tables")?;
        let indices = contiguous(indices, index_layout, "sparse policy indices")?;
        let output = indices
            .chunks_exact(self.terms)
            .map(|terms| terms.iter().map(|&index| tables[index as usize]).sum())
            .collect();
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
        indices: &candle_core::CudaStorage,
        index_layout: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        cuda_forward(tables, table_layout, indices, index_layout, self)
    }

    fn bwd(
        &self,
        tables: &Tensor,
        indices: &Tensor,
        _result: &Tensor,
        grad_result: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let grad =
            tables.apply_op3_no_bwd(indices, grad_result, &SparsePolicyGrad(self.clone()))?;
        Ok((Some(grad), None))
    }
}

impl CustomOp3 for SparsePolicyGrad {
    fn name(&self) -> &'static str {
        "az-sparse-policy-grad"
    }

    fn cpu_fwd(
        &self,
        _table_storage: &CpuStorage,
        _table_layout: &Layout,
        index_storage: &CpuStorage,
        index_layout: &Layout,
        grad_storage: &CpuStorage,
        grad_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (CpuStorage::I64(indices), CpuStorage::F32(grad)) = (index_storage, grad_storage)
        else {
            candle_core::bail!("sparse policy gradient expects i64 indices and f32 gradient")
        };
        let indices = contiguous(indices, index_layout, "sparse policy indices")?;
        let grad = contiguous(grad, grad_layout, "sparse policy output gradient")?;
        let mut output = vec![0.0f32; self.0.table_len];
        for (output_index, terms) in indices.chunks_exact(self.0.terms).enumerate() {
            for &table_index in terms {
                output[table_index as usize] += grad[output_index];
            }
        }
        Ok((CpuStorage::F32(output), self.0.table_len.into()))
    }

    #[cfg(any(
        target_os = "windows",
        all(target_os = "linux", not(target_env = "musl"))
    ))]
    fn cuda_fwd(
        &self,
        _tables: &candle_core::CudaStorage,
        _table_layout: &Layout,
        indices: &candle_core::CudaStorage,
        index_layout: &Layout,
        grad: &candle_core::CudaStorage,
        grad_layout: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        cuda_gradient(indices, index_layout, grad, grad_layout, &self.0)
    }
}

fn contiguous<'a, T>(values: &'a [T], layout: &Layout, name: &str) -> Result<&'a [T]> {
    let Some((start, end)) = layout.contiguous_offsets() else {
        candle_core::bail!("{name} must be contiguous")
    };
    Ok(&values[start..end])
}

#[cfg(any(
    target_os = "windows",
    all(target_os = "linux", not(target_env = "musl"))
))]
fn ptx() -> Result<&'static str> {
    static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    match PTX.get_or_init(|| {
        candle_core::cuda_backend::cudarc::nvrtc::safe::compile_ptx(CUDA_SOURCE)
            .map(|ptx| ptx.to_src())
            .map_err(|error| format!("failed to compile sparse policy CUDA kernels: {error}"))
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
    index_storage: &candle_core::CudaStorage,
    index_layout: &Layout,
    op: &SparsePolicy,
) -> Result<(candle_core::CudaStorage, Shape)> {
    use candle_core::cuda_backend::CudaStorageSlice::{F32, I64};
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    let (F32(tables), I64(indices)) = (&table_storage.slice, &index_storage.slice) else {
        candle_core::bail!("sparse policy CUDA kernel expects f32 tables and i64 indices")
    };
    let tables = cuda_view(tables, table_layout, "sparse policy tables")?;
    let indices = cuda_view(indices, index_layout, "sparse policy indices")?;
    let device = &table_storage.device;
    let outputs = op.batch * op.moves;
    let output = unsafe { device.alloc::<f32>(outputs)? };
    let name = if op.terms == TERMS {
        "sparse_policy_fwd"
    } else {
        "tactical_policy_fwd"
    };
    let function = device.get_or_load_custom_func(name, "chineseai_sparse_policy_v2", ptx()?)?;
    let mut builder = function.builder();
    builder.arg(&tables).arg(&indices).arg(&output);
    candle_core::builder_arg!(builder, outputs as u32);
    unsafe { builder.launch(LaunchConfig::for_num_elems(outputs as u32)) }.map_err(|error| {
        candle_core::Error::Msg(format!("sparse policy launch failed: {error}"))
    })?;
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
    index_storage: &candle_core::CudaStorage,
    index_layout: &Layout,
    grad_storage: &candle_core::CudaStorage,
    grad_layout: &Layout,
    op: &SparsePolicy,
) -> Result<(candle_core::CudaStorage, Shape)> {
    use candle_core::cuda_backend::CudaStorageSlice::{F32, I64};
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    let (I64(indices), F32(grad)) = (&index_storage.slice, &grad_storage.slice) else {
        candle_core::bail!(
            "sparse policy gradient CUDA kernel expects i64 indices and f32 gradient"
        )
    };
    let indices = cuda_view(indices, index_layout, "sparse policy indices")?;
    let grad = cuda_view(grad, grad_layout, "sparse policy output gradient")?;
    let device = &index_storage.device;
    let output = device.alloc_zeros::<f32>(op.table_len)?;
    let entries = op.batch * op.moves * op.terms;
    let name = if op.terms == TERMS {
        "sparse_policy_grad"
    } else {
        "tactical_policy_grad"
    };
    let function = device.get_or_load_custom_func(name, "chineseai_sparse_policy_v2", ptx()?)?;
    let mut builder = function.builder();
    builder.arg(&indices).arg(&grad).arg(&output);
    candle_core::builder_arg!(builder, entries as u32);
    unsafe { builder.launch(LaunchConfig::for_num_elems(entries as u32)) }.map_err(|error| {
        candle_core::Error::Msg(format!("sparse policy gradient launch failed: {error}"))
    })?;
    Ok((
        candle_core::CudaStorage {
            slice: F32(output),
            device: device.clone(),
        },
        op.table_len.into(),
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

    fn output_and_grad(device: &Device) -> Result<(Vec<f32>, Vec<f32>)> {
        let tables = Var::from_slice(&[0.5f32, -1.0, 2.0, 3.0, -0.25], 5, device)?;
        let indices = Tensor::from_slice(
            &[0i64, 1, 2, 2, 3, 4, 0, 4, 4, 4, 1, 1, 0, 3],
            (1, 2, TERMS),
            device,
        )?;
        let output = sparse_policy(&tables, &indices)?;
        let weights = Tensor::from_slice(&[2.0f32, -0.5], (1, 2), device)?;
        let grads = output.broadcast_mul(&weights)?.sum_all()?.backward()?;
        Ok((
            output.flatten_all()?.to_vec1()?,
            grads.get(&tables).unwrap().flatten_all()?.to_vec1()?,
        ))
    }

    fn assert_close(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        for (&left, &right) in left.iter().zip(right) {
            assert!((left - right).abs() < 1.0e-5, "left={left} right={right}");
        }
    }

    fn tactical_output_and_grad(device: &Device) -> Result<(Vec<f32>, Vec<f32>)> {
        let tables = Var::from_slice(&[0.5f32, -1.0, 2.0, 3.0, -0.25], 5, device)?;
        let indices = Tensor::from_slice(&[0i64, 2, 3, 4], (1, 2, TACTICAL_TERMS), device)?;
        let output = tactical_policy(&tables, &indices)?;
        let weights = Tensor::from_slice(&[2.0f32, -0.5], (1, 2), device)?;
        let grads = output.broadcast_mul(&weights)?.sum_all()?.backward()?;
        Ok((
            output.flatten_all()?.to_vec1()?,
            grads.get(&tables).unwrap().flatten_all()?.to_vec1()?,
        ))
    }

    #[test]
    fn fused_sparse_policy_matches_expected_and_cuda() -> Result<()> {
        let cpu = output_and_grad(&Device::Cpu)?;
        assert_close(&cpu.0, &[6.75, 0.75]);
        assert_close(&cpu.1, &[3.5, 1.0, 4.0, 1.5, 0.5]);
        if let Ok(device) = Device::new_cuda(0) {
            let cuda = output_and_grad(&device)?;
            assert_close(&cpu.0, &cuda.0);
            assert_close(&cpu.1, &cuda.1);
        }
        Ok(())
    }

    #[test]
    fn fused_tactical_policy_matches_expected_and_cuda() -> Result<()> {
        let cpu = tactical_output_and_grad(&Device::Cpu)?;
        assert_close(&cpu.0, &[2.5, 2.75]);
        assert_close(&cpu.1, &[2.0, 0.0, 2.0, -0.5, -0.5]);
        if let Ok(device) = Device::new_cuda(0) {
            let cuda = tactical_output_and_grad(&device)?;
            assert_close(&cpu.0, &cuda.0);
            assert_close(&cpu.1, &cuda.1);
        }
        Ok(())
    }
}
