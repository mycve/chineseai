use std::sync::OnceLock;

use candle_core::{CpuStorage, CustomOp2, CustomOp3, Layout, Result, Shape, Tensor};

use super::{
    STRUCTURAL_FILE_SIZE, STRUCTURAL_KING_PIECE_SIZE, STRUCTURAL_PIECE_SIZE, STRUCTURAL_RANK_SIZE,
};
use crate::nnue::AZ_NNUE_INPUT_SIZE;
use crate::xiangqi::BOARD_SIZE;

pub(super) const PADDING_ITEM: u32 = u32::MAX;
const PIECE_OFFSET: usize = AZ_NNUE_INPUT_SIZE;
const RANK_OFFSET: usize = PIECE_OFFSET + STRUCTURAL_PIECE_SIZE;
const FILE_OFFSET: usize = RANK_OFFSET + STRUCTURAL_RANK_SIZE;
const KING_OFFSET: usize = FILE_OFFSET + STRUCTURAL_FILE_SIZE;
const TABLE_ROWS: usize = KING_OFFSET + STRUCTURAL_KING_PIECE_SIZE;
const _: () = assert!(AZ_NNUE_INPUT_SIZE == 1260);
const _: () = assert!(PIECE_OFFSET == 1260 && RANK_OFFSET == 1274);
const _: () = assert!(FILE_OFFSET == 1284 && KING_OFFSET == 1293 && TABLE_ROWS == 1545);

const CUDA_SOURCE: &str = r#"
extern "C" __global__ void feature_pool_fwd(
    const float* tables, const unsigned int* items, float* output,
    unsigned int batch, unsigned int item_count, unsigned int hidden
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch * hidden) return;
    unsigned int b = index / hidden;
    unsigned int h = index - b * hidden;
    float sum = 0.0f;
    for (unsigned int i = 0; i < item_count; ++i) {
        unsigned int packed = items[b * item_count + i];
        if (packed == 0xffffffffu) continue;
        unsigned int feature = packed & 0x7ffu;
        unsigned int us_bucket = (packed >> 11) & 0xfu;
        unsigned int them_bucket = (packed >> 15) & 0xfu;
        unsigned int piece = feature / 90u;
        unsigned int square = feature - piece * 90u;
        unsigned int rank = square / 9u;
        unsigned int file = square - rank * 9u;
        sum += tables[feature * hidden + h];
        sum += tables[(1260u + piece) * hidden + h];
        sum += tables[(1274u + rank) * hidden + h];
        sum += tables[(1284u + file) * hidden + h];
        sum += tables[(1293u + us_bucket * 14u + piece) * hidden + h];
        sum += tables[(1293u + (9u + them_bucket) * 14u + piece) * hidden + h];
    }
    output[index] = sum;
}

extern "C" __global__ void feature_pool_grad(
    const unsigned int* items, const float* grad_output, float* grad_tables,
    unsigned int batch, unsigned int item_count, unsigned int hidden
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch * item_count * hidden) return;
    unsigned int h = index % hidden;
    unsigned int item_index = index / hidden;
    unsigned int packed = items[item_index];
    if (packed == 0xffffffffu) return;
    unsigned int b = item_index / item_count;
    unsigned int feature = packed & 0x7ffu;
    unsigned int us_bucket = (packed >> 11) & 0xfu;
    unsigned int them_bucket = (packed >> 15) & 0xfu;
    unsigned int piece = feature / 90u;
    unsigned int square = feature - piece * 90u;
    unsigned int rank = square / 9u;
    unsigned int file = square - rank * 9u;
    float g = grad_output[b * hidden + h];
    atomicAdd(grad_tables + feature * hidden + h, g);
    atomicAdd(grad_tables + (1260u + piece) * hidden + h, g);
    atomicAdd(grad_tables + (1274u + rank) * hidden + h, g);
    atomicAdd(grad_tables + (1284u + file) * hidden + h, g);
    atomicAdd(grad_tables + (1293u + us_bucket * 14u + piece) * hidden + h, g);
    atomicAdd(grad_tables + (1293u + (9u + them_bucket) * 14u + piece) * hidden + h, g);
}
"#;

#[derive(Clone, Debug)]
struct FeaturePool {
    batch: usize,
    items: usize,
    hidden: usize,
}

#[derive(Clone, Debug)]
struct FeaturePoolGrad(FeaturePool);

pub(super) const fn pack_feature(feature: usize, us_bucket: usize, them_bucket: usize) -> u32 {
    feature as u32 | (us_bucket as u32) << 11 | (them_bucket as u32) << 15
}

pub(super) fn feature_pool(tables: &Tensor, items: &Tensor) -> Result<Tensor> {
    let (rows, hidden) = tables.dims2()?;
    let (batch, item_count) = items.dims2()?;
    if rows != TABLE_ROWS {
        candle_core::bail!("feature pool row mismatch: got {rows}, expected {TABLE_ROWS}")
    }
    tables.apply_op2(
        items,
        FeaturePool {
            batch,
            items: item_count,
            hidden,
        },
    )
}

impl CustomOp2 for FeaturePool {
    fn name(&self) -> &'static str {
        "az-feature-pool"
    }

    fn cpu_fwd(
        &self,
        table_storage: &CpuStorage,
        table_layout: &Layout,
        item_storage: &CpuStorage,
        item_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (CpuStorage::F32(tables), CpuStorage::U32(items)) = (table_storage, item_storage)
        else {
            candle_core::bail!("feature pool expects f32 tables and u32 items")
        };
        let tables = contiguous_slice(tables, table_layout, "feature tables")?;
        let items = contiguous_slice(items, item_layout, "feature items")?;
        let mut output = vec![0.0f32; self.batch * self.hidden];
        for b in 0..self.batch {
            for &packed in &items[b * self.items..(b + 1) * self.items] {
                if packed == PADDING_ITEM {
                    continue;
                }
                let rows = item_rows(packed);
                for h in 0..self.hidden {
                    let out = &mut output[b * self.hidden + h];
                    for row in rows {
                        *out += tables[row * self.hidden + h];
                    }
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
        tables: &candle_core::CudaStorage,
        table_layout: &Layout,
        items: &candle_core::CudaStorage,
        item_layout: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        cuda_forward(tables, table_layout, items, item_layout, self)
    }

    fn bwd(
        &self,
        tables: &Tensor,
        items: &Tensor,
        _result: &Tensor,
        grad_result: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let grad = tables.apply_op3_no_bwd(items, grad_result, &FeaturePoolGrad(self.clone()))?;
        Ok((Some(grad), None))
    }
}

impl CustomOp3 for FeaturePoolGrad {
    fn name(&self) -> &'static str {
        "az-feature-pool-grad"
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
            candle_core::bail!("feature pool gradient expects u32 items and f32 gradient")
        };
        let items = contiguous_slice(items, item_layout, "feature items")?;
        let grad = contiguous_slice(grad, grad_layout, "feature output gradient")?;
        let op = &self.0;
        let mut output = vec![0.0f32; TABLE_ROWS * op.hidden];
        for b in 0..op.batch {
            for &packed in &items[b * op.items..(b + 1) * op.items] {
                if packed == PADDING_ITEM {
                    continue;
                }
                for row in item_rows(packed) {
                    for h in 0..op.hidden {
                        output[row * op.hidden + h] += grad[b * op.hidden + h];
                    }
                }
            }
        }
        Ok((CpuStorage::F32(output), (TABLE_ROWS, op.hidden).into()))
    }

    #[cfg(any(
        target_os = "windows",
        all(target_os = "linux", not(target_env = "musl"))
    ))]
    fn cuda_fwd(
        &self,
        _tables: &candle_core::CudaStorage,
        _table_layout: &Layout,
        items: &candle_core::CudaStorage,
        item_layout: &Layout,
        grad: &candle_core::CudaStorage,
        grad_layout: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        cuda_gradient(items, item_layout, grad, grad_layout, &self.0)
    }
}

fn item_rows(packed: u32) -> [usize; 6] {
    let feature = (packed & 0x7ff) as usize;
    let us_bucket = ((packed >> 11) & 0xf) as usize;
    let them_bucket = ((packed >> 15) & 0xf) as usize;
    let piece = feature / BOARD_SIZE;
    let square = feature % BOARD_SIZE;
    [
        feature,
        PIECE_OFFSET + piece,
        RANK_OFFSET + square / 9,
        FILE_OFFSET + square % 9,
        KING_OFFSET + us_bucket * 14 + piece,
        KING_OFFSET + (9 + them_bucket) * 14 + piece,
    ]
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
fn cuda_forward(
    table_storage: &candle_core::CudaStorage,
    table_layout: &Layout,
    item_storage: &candle_core::CudaStorage,
    item_layout: &Layout,
    op: &FeaturePool,
) -> Result<(candle_core::CudaStorage, Shape)> {
    use candle_core::cuda_backend::CudaStorageSlice::{F32, U32};
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    let (F32(tables), U32(items)) = (&table_storage.slice, &item_storage.slice) else {
        candle_core::bail!("feature pool CUDA kernel expects f32 tables and u32 items")
    };
    let tables = cuda_view(tables, table_layout, "feature tables")?;
    let items = cuda_view(items, item_layout, "feature items")?;
    let device = &table_storage.device;
    let output = unsafe { device.alloc::<f32>(op.batch * op.hidden)? };
    let function = device.get_or_load_custom_func(
        "feature_pool_fwd",
        "chineseai_feature_pool_v1",
        feature_ptx()?,
    )?;
    let mut builder = function.builder();
    builder.arg(&tables).arg(&items).arg(&output);
    candle_core::builder_arg!(builder, op.batch as u32, op.items as u32, op.hidden as u32);
    unsafe { builder.launch(LaunchConfig::for_num_elems((op.batch * op.hidden) as u32)) }
        .map_err(|error| candle_core::Error::Msg(format!("feature pool launch failed: {error}")))?;
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
fn cuda_gradient(
    item_storage: &candle_core::CudaStorage,
    item_layout: &Layout,
    grad_storage: &candle_core::CudaStorage,
    grad_layout: &Layout,
    op: &FeaturePool,
) -> Result<(candle_core::CudaStorage, Shape)> {
    use candle_core::cuda_backend::CudaStorageSlice::{F32, U32};
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    let (U32(items), F32(grad)) = (&item_storage.slice, &grad_storage.slice) else {
        candle_core::bail!("feature pool gradient CUDA kernel expects u32 items and f32 gradient")
    };
    let items = cuda_view(items, item_layout, "feature items")?;
    let grad = cuda_view(grad, grad_layout, "feature output gradient")?;
    let device = &item_storage.device;
    let output = device.alloc_zeros::<f32>(TABLE_ROWS * op.hidden)?;
    let function = device.get_or_load_custom_func(
        "feature_pool_grad",
        "chineseai_feature_pool_v1",
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
        candle_core::Error::Msg(format!("feature pool gradient launch failed: {error}"))
    })?;
    Ok((
        candle_core::CudaStorage {
            slice: F32(output),
            device: device.clone(),
        },
        (TABLE_ROWS, op.hidden).into(),
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
        let hidden = 7;
        let values = (0..TABLE_ROWS * hidden)
            .map(|index| (index as f32 - 4000.0) * 0.0001)
            .collect::<Vec<_>>();
        let tables = Var::from_slice(&values, (TABLE_ROWS, hidden), device)?;
        let items = [
            pack_feature(0, 1, 8),
            pack_feature(95, 3, 4),
            PADDING_ITEM,
            pack_feature(1259, 8, 0),
            pack_feature(95, 3, 4),
            pack_feature(611, 2, 6),
        ];
        let items = Tensor::from_slice(&items, (2, 3), device)?;
        let output = feature_pool(&tables, &items)?;
        let grads = output.sum_all()?.backward()?;
        Ok((
            output.flatten_all()?.to_vec1()?,
            grads.get(&tables).unwrap().flatten_all()?.to_vec1()?,
        ))
    }

    fn assert_close(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        for (&left, &right) in left.iter().zip(right) {
            assert!((left - right).abs() < 2.0e-5, "left={left} right={right}");
        }
    }

    #[test]
    fn fused_feature_pool_cuda_matches_cpu_forward_and_gradient() -> Result<()> {
        let cpu = output_and_grad(&Device::Cpu)?;
        let Ok(cuda) = Device::new_cuda(0) else {
            return Ok(());
        };
        let gpu = output_and_grad(&cuda)?;
        assert_close(&cpu.0, &gpu.0);
        assert_close(&cpu.1, &gpu.1);
        Ok(())
    }
}
