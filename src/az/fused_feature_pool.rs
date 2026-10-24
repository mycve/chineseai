use candle_core::{CpuStorage, CustomOp2, CustomOp3, Layout, Result, Shape, Tensor};
use std::sync::OnceLock;
pub(super) const PADDING_ITEM: u32 = u32::MAX;
const CUDA_SOURCE: &str = r#"
extern "C" __global__ void sparse_pool_fwd(
    const float* table, const unsigned int* items, float* output,
    unsigned int batch, unsigned int item_count, unsigned int hidden
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch * hidden) return;
    unsigned int b = index / hidden;
    unsigned int h = index - b * hidden;
    float sum = 0.0f;
    for (unsigned int i = 0; i < item_count; ++i) {
        unsigned int item = items[b * item_count + i];
        if (item != 0xffffffffu) sum += table[item * hidden + h];
    }
    output[index] = sum;
}

extern "C" __global__ void sparse_pool_grad(
    const unsigned int* items, const float* grad_output, float* grad_table,
    unsigned int batch, unsigned int item_count, unsigned int hidden
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch * item_count * hidden) return;
    unsigned int h = index % hidden;
    unsigned int item_index = index / hidden;
    unsigned int item = items[item_index];
    if (item == 0xffffffffu) return;
    unsigned int b = item_index / item_count;
    atomicAdd(grad_table + item * hidden + h, grad_output[b * hidden + h]);
}
"#;
#[derive(Clone, Debug)]
struct SparsePool {
    rows: usize,
    batch: usize,
    items: usize,
    hidden: usize,
}

#[derive(Clone, Debug)]
struct SparsePoolGrad(SparsePool);

pub(super) fn sparse_pool(table: &Tensor, items: &Tensor) -> Result<Tensor> {
    let (rows, hidden) = table.dims2()?;
    let (batch, item_count) = items.dims2()?;
    table.apply_op2(
        items,
        SparsePool {
            rows,
            batch,
            items: item_count,
            hidden,
        },
    )
}

/// 训练查表统一走稀疏池化。不要改成 `Tensor::index_select`：Candle 0.10.2
/// 的反传 index_add 按表列数分配线程，再串行遍历索引；标量表只有一个线程。
pub(super) fn gather_rows(table: &Tensor, indices: &Tensor) -> Result<Tensor> {
    let rows = table.dim(0)?;
    let width = if table.rank() == 1 {
        1
    } else {
        table.dims2()?.1
    };
    let count = indices.dims1()?;
    let output = sparse_pool(
        &table.reshape((rows, width))?,
        &indices.reshape((count, 1))?,
    )?;
    if table.rank() == 1 {
        output.squeeze(1)
    } else {
        Ok(output)
    }
}

impl CustomOp2 for SparsePool {
    fn name(&self) -> &'static str {
        "az-sparse-pool"
    }

    fn cpu_fwd(
        &self,
        table_storage: &CpuStorage,
        table_layout: &Layout,
        item_storage: &CpuStorage,
        item_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (CpuStorage::F32(table), CpuStorage::U32(items)) = (table_storage, item_storage) else {
            candle_core::bail!("sparse pool expects f32 table and u32 items")
        };
        let table = contiguous_slice(table, table_layout, "sparse table")?;
        let items = contiguous_slice(items, item_layout, "sparse items")?;
        let mut output = vec![0.0f32; self.batch * self.hidden];
        for batch in 0..self.batch {
            for &item in &items[batch * self.items..(batch + 1) * self.items] {
                if item == PADDING_ITEM {
                    continue;
                }
                let item = item as usize;
                for hidden in 0..self.hidden {
                    output[batch * self.hidden + hidden] += table[item * self.hidden + hidden];
                }
            }
        }
        Ok((CpuStorage::F32(output), (self.batch, self.hidden).into()))
    }

    #[cfg(any(
        target_os = "windows",
        all(target_os = "linux", not(target_env = "musl"))
    ))]
    fn cuda_fwd(
        &self,
        table: &candle_core::CudaStorage,
        table_layout: &Layout,
        items: &candle_core::CudaStorage,
        item_layout: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        cuda_sparse_forward(table, table_layout, items, item_layout, self)
    }

    fn bwd(
        &self,
        table: &Tensor,
        items: &Tensor,
        _result: &Tensor,
        grad_result: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let grad = table.apply_op3_no_bwd(items, grad_result, &SparsePoolGrad(self.clone()))?;
        Ok((Some(grad), None))
    }
}

impl CustomOp3 for SparsePoolGrad {
    fn name(&self) -> &'static str {
        "az-sparse-pool-grad"
    }

    fn cpu_fwd(
        &self,
        _table_storage: &CpuStorage,
        _table_layout: &Layout,
        item_storage: &CpuStorage,
        item_layout: &Layout,
        grad_storage: &CpuStorage,
        grad_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (CpuStorage::U32(items), CpuStorage::F32(grad)) = (item_storage, grad_storage) else {
            candle_core::bail!("sparse pool gradient expects u32 items and f32 gradient")
        };
        let items = contiguous_slice(items, item_layout, "sparse items")?;
        let grad = contiguous_slice(grad, grad_layout, "sparse gradient")?;
        let op = &self.0;
        let mut output = vec![0.0f32; op.rows * op.hidden];
        for batch in 0..op.batch {
            for &item in &items[batch * op.items..(batch + 1) * op.items] {
                if item == PADDING_ITEM {
                    continue;
                }
                let item = item as usize;
                for hidden in 0..op.hidden {
                    output[item * op.hidden + hidden] += grad[batch * op.hidden + hidden];
                }
            }
        }
        Ok((CpuStorage::F32(output), (op.rows, op.hidden).into()))
    }

    #[cfg(any(
        target_os = "windows",
        all(target_os = "linux", not(target_env = "musl"))
    ))]
    fn cuda_fwd(
        &self,
        _table: &candle_core::CudaStorage,
        _table_layout: &Layout,
        items: &candle_core::CudaStorage,
        item_layout: &Layout,
        grad: &candle_core::CudaStorage,
        grad_layout: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        cuda_sparse_gradient(items, item_layout, grad, grad_layout, &self.0)
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
fn feature_ptx() -> Result<&'static str> {
    static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    match PTX.get_or_init(|| {
        candle_core::cuda_backend::cudarc::nvrtc::safe::compile_ptx(CUDA_SOURCE)
            .map(|ptx| ptx.to_src())
            .map_err(|error| format!("failed to compile feature pool CUDA kernels: {error}"))
    }) {
        Ok(ptx) => Ok(ptx),
        Err(error) => candle_core::bail!("{error}"),
    }
}

#[cfg(any(
    target_os = "windows",
    all(target_os = "linux", not(target_env = "musl"))
))]
fn cuda_sparse_forward(
    table_storage: &candle_core::CudaStorage,
    table_layout: &Layout,
    item_storage: &candle_core::CudaStorage,
    item_layout: &Layout,
    op: &SparsePool,
) -> Result<(candle_core::CudaStorage, Shape)> {
    use candle_core::cuda_backend::CudaStorageSlice::{F32, U32};
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    let (F32(table), U32(items)) = (&table_storage.slice, &item_storage.slice) else {
        candle_core::bail!("sparse pool CUDA kernel expects f32 table and u32 items")
    };
    let table = cuda_view(table, table_layout, "sparse table")?;
    let items = cuda_view(items, item_layout, "sparse items")?;
    let device = &table_storage.device;
    let output = unsafe { device.alloc::<f32>(op.batch * op.hidden)? };
    let function = device.get_or_load_custom_func(
        "sparse_pool_fwd",
        "chineseai_feature_pool_v2",
        feature_ptx()?,
    )?;
    let mut builder = function.builder();
    builder.arg(&table).arg(&items).arg(&output);
    candle_core::builder_arg!(builder, op.batch as u32, op.items as u32, op.hidden as u32);
    unsafe { builder.launch(LaunchConfig::for_num_elems((op.batch * op.hidden) as u32)) }
        .map_err(|error| candle_core::Error::Msg(format!("sparse pool launch failed: {error}")))?;
    Ok((
        candle_core::CudaStorage {
            slice: F32(output),
            device: device.clone(),
        },
        (op.batch, op.hidden).into(),
    ))
}

#[cfg(any(
    target_os = "windows",
    all(target_os = "linux", not(target_env = "musl"))
))]
fn cuda_sparse_gradient(
    item_storage: &candle_core::CudaStorage,
    item_layout: &Layout,
    grad_storage: &candle_core::CudaStorage,
    grad_layout: &Layout,
    op: &SparsePool,
) -> Result<(candle_core::CudaStorage, Shape)> {
    use candle_core::cuda_backend::CudaStorageSlice::{F32, U32};
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    let (U32(items), F32(grad)) = (&item_storage.slice, &grad_storage.slice) else {
        candle_core::bail!("sparse pool gradient CUDA kernel expects u32 items and f32 gradient")
    };
    let items = cuda_view(items, item_layout, "sparse items")?;
    let grad = cuda_view(grad, grad_layout, "sparse gradient")?;
    let device = &item_storage.device;
    let output = device.alloc_zeros::<f32>(op.rows * op.hidden)?;
    let function = device.get_or_load_custom_func(
        "sparse_pool_grad",
        "chineseai_feature_pool_v2",
        feature_ptx()?,
    )?;
    let mut builder = function.builder();
    builder.arg(&items).arg(&grad).arg(&output);
    candle_core::builder_arg!(builder, op.batch as u32, op.items as u32, op.hidden as u32);
    unsafe {
        builder.launch(LaunchConfig::for_num_elems(
            (op.batch * op.items * op.hidden) as u32,
        ))
    }
    .map_err(|error| {
        candle_core::Error::Msg(format!("sparse pool gradient launch failed: {error}"))
    })?;
    Ok((
        candle_core::CudaStorage {
            slice: F32(output),
            device: device.clone(),
        },
        (op.rows, op.hidden).into(),
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

    fn check_gather(device: &Device) -> Result<()> {
        for width in [1, 16, 288] {
            let rows = 37;
            let count = 8192;
            let data = (0..rows * width)
                .map(|i| (i % 29) as f32 / 8.)
                .collect::<Vec<_>>();
            // 大量重复同一索引，覆盖 CUDA 原子梯度累加与首尾边界。
            let ids = (0..count)
                .map(|i| if i % 3 == 0 { (i % rows) as u32 } else { 0 })
                .collect::<Vec<_>>();
            let weights = (0..count * width)
                .map(|i| (i % 7) as f32 / 16. - 0.1875)
                .collect::<Vec<_>>();
            let shape = if width == 1 {
                vec![rows]
            } else {
                vec![rows, width]
            };
            let table = Var::from_slice(&data, shape.as_slice(), device)?;
            let indices = Tensor::new(ids.as_slice(), device)?;
            let selected = gather_rows(table.as_tensor(), &indices)?;
            let expected = ids
                .iter()
                .flat_map(|&id| {
                    data[id as usize * width..(id as usize + 1) * width]
                        .iter()
                        .copied()
                })
                .collect::<Vec<_>>();
            assert_eq!(selected.flatten_all()?.to_vec1::<f32>()?, expected);
            let weighted = selected
                .flatten_all()?
                .mul(&Tensor::new(weights.as_slice(), device)?)?
                .sum_all()?;
            let grads = weighted.backward()?;
            let actual = grads.get(&table).unwrap().flatten_all()?.to_vec1::<f32>()?;
            let mut expected = vec![0.; rows * width];
            for (i, &id) in ids.iter().enumerate() {
                for h in 0..width {
                    expected[id as usize * width + h] += weights[i * width + h];
                }
            }
            assert_eq!(actual, expected);
        }
        Ok(())
    }

    #[test]
    fn row_gather_repeated_indices_forward_and_backward() -> Result<()> {
        check_gather(&Device::Cpu)
    }

    #[cfg(all(
        feature = "slow-tests",
        any(
            target_os = "windows",
            all(target_os = "linux", not(target_env = "musl"))
        )
    ))]
    #[test]
    fn cuda_row_gather_repeated_indices_forward_and_backward() -> Result<()> {
        if let Some(device) = crate::az::cuda_test_device::shared_cuda_device() {
            check_gather(device)?;
        }
        Ok(())
    }
}
