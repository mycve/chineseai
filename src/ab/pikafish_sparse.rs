//! Ragged feature sums: stream rows directly, without a gathered feature matrix.
use candle_core::{CpuStorage, CustomOp2, CustomOp3, Layout, Result, Shape, Tensor};
const PAD: u32 = u32::MAX;
#[derive(Clone, Debug)]
struct Sum {
    rows: usize,
    width: usize,
    batch: usize,
    items: usize,
}
struct Grad(Sum);
pub(super) fn sums(table: &Tensor, indices: &[&[usize]]) -> Result<Tensor> {
    if indices.is_empty() {
        candle_core::bail!("empty sparse batch")
    }
    let (rows, width) = table.dims2()?;
    let items = indices
        .iter()
        .map(|ids| ids.len())
        .max()
        .unwrap_or(0)
        .max(1);
    let mut ids = vec![PAD; indices.len() * items];
    for (b, source) in indices.iter().enumerate() {
        for (i, &index) in source.iter().enumerate() {
            if index >= rows {
                candle_core::bail!("Pikafish feature out of range")
            }
            ids[b * items + i] = index as u32;
        }
    }
    table.apply_op2(
        &Tensor::from_vec(ids, (indices.len(), items), table.device())?,
        Sum {
            rows,
            width,
            batch: indices.len(),
            items,
        },
    )
}
fn slice<'a, T>(values: &'a [T], layout: &Layout) -> Result<&'a [T]> {
    let (start, end) = layout
        .contiguous_offsets()
        .ok_or_else(|| candle_core::Error::Msg("sparse sum requires contiguous storage".into()))?;
    Ok(&values[start..end])
}
impl CustomOp2 for Sum {
    fn name(&self) -> &'static str {
        "pikafish-sparse-sum"
    }
    fn cpu_fwd(
        &self,
        table: &CpuStorage,
        tl: &Layout,
        ids: &CpuStorage,
        il: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (CpuStorage::F32(table), CpuStorage::U32(ids)) = (table, ids) else {
            candle_core::bail!("sparse sum expects f32/u32")
        };
        let table = slice(table, tl)?;
        let ids = slice(ids, il)?;
        let mut output = vec![0f32; self.batch * self.width];
        for b in 0..self.batch {
            for &id in &ids[b * self.items..(b + 1) * self.items] {
                if id == PAD {
                    continue;
                }
                for (out, &value) in output[b * self.width..(b + 1) * self.width]
                    .iter_mut()
                    .zip(&table[id as usize * self.width..(id as usize + 1) * self.width])
                {
                    *out += value;
                }
            }
        }
        Ok((CpuStorage::F32(output), (self.batch, self.width).into()))
    }
    fn bwd(
        &self,
        table: &Tensor,
        ids: &Tensor,
        _: &Tensor,
        grad: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        Ok((
            Some(table.apply_op3_no_bwd(ids, &grad.contiguous()?, &Grad(self.clone()))?),
            None,
        ))
    }
    #[cfg(any(
        target_os = "windows",
        all(target_os = "linux", not(target_env = "musl"))
    ))]
    fn cuda_fwd(
        &self,
        table: &candle_core::CudaStorage,
        tl: &Layout,
        ids: &candle_core::CudaStorage,
        il: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        cuda::launch(self, table, tl, ids, il, None)
    }
}
impl CustomOp3 for Grad {
    fn name(&self) -> &'static str {
        "pikafish-sparse-sum-grad"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        ids: &CpuStorage,
        il: &Layout,
        grad: &CpuStorage,
        gl: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (CpuStorage::U32(ids), CpuStorage::F32(grad)) = (ids, grad) else {
            candle_core::bail!("sparse gradient expects u32/f32")
        };
        let ids = slice(ids, il)?;
        let grad = slice(grad, gl)?;
        let op = &self.0;
        let mut output = vec![0f32; op.rows * op.width];
        for b in 0..op.batch {
            for &id in &ids[b * op.items..(b + 1) * op.items] {
                if id == PAD {
                    continue;
                }
                for (out, &value) in output[id as usize * op.width..(id as usize + 1) * op.width]
                    .iter_mut()
                    .zip(&grad[b * op.width..(b + 1) * op.width])
                {
                    *out += value;
                }
            }
        }
        Ok((CpuStorage::F32(output), (op.rows, op.width).into()))
    }
    #[cfg(any(
        target_os = "windows",
        all(target_os = "linux", not(target_env = "musl"))
    ))]
    fn cuda_fwd(
        &self,
        table: &candle_core::CudaStorage,
        tl: &Layout,
        ids: &candle_core::CudaStorage,
        il: &Layout,
        grad: &candle_core::CudaStorage,
        gl: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        cuda::launch(&self.0, table, tl, ids, il, Some((grad, gl)))
    }
}
#[cfg(any(
    target_os = "windows",
    all(target_os = "linux", not(target_env = "musl"))
))]
mod cuda {
    use super::*;
    use candle_core::cuda_backend::{
        CudaStorage,
        CudaStorageSlice::{F32, U32},
        cudarc::driver::{LaunchConfig, PushKernelArg},
    };
    use std::sync::OnceLock;
    const SOURCE: &str = r#"
extern "C" __global__ void sparse_sum(const float* table,const unsigned int* ids,float* out,unsigned int batch,unsigned int items,unsigned int width) {
    unsigned int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=batch*width)return;
    unsigned int b=j/width,h=j%width; float value=0.0f;
    for(unsigned int i=0;i<items;i++){unsigned int id=ids[b*items+i];if(id!=0xffffffffu)value+=table[id*width+h];}
    out[j]=value;
}
extern "C" __global__ void sparse_grad(const float* grad,const unsigned int* ids,float* out,unsigned int batch,unsigned int items,unsigned int width) {
    unsigned int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=batch*items*width)return;
    unsigned int item=j/width,h=j%width,id=ids[item];
    if(id!=0xffffffffu)atomicAdd(out+id*width+h,grad[(item/items)*width+h]);
}
"#;
    fn ptx() -> Result<&'static str> {
        static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
        match PTX.get_or_init(|| {
            candle_core::cuda_backend::cudarc::nvrtc::safe::compile_ptx(SOURCE)
                .map(|p| p.to_src())
                .map_err(|e| e.to_string())
        }) {
            Ok(s) => Ok(s),
            Err(e) => candle_core::bail!("{e}"),
        }
    }
    fn view<'a, T: candle_core::cuda_backend::cudarc::driver::DeviceRepr>(
        data: &'a candle_core::cuda_backend::cudarc::driver::CudaSlice<T>,
        layout: &Layout,
    ) -> Result<candle_core::cuda_backend::cudarc::driver::CudaView<'a, T>> {
        let (a, b) = layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("sparse CUDA storage must be contiguous".into())
        })?;
        Ok(data.slice(a..b))
    }
    pub(super) fn launch(
        op: &Sum,
        table: &CudaStorage,
        tl: &Layout,
        ids: &CudaStorage,
        il: &Layout,
        grad: Option<(&CudaStorage, &Layout)>,
    ) -> Result<(CudaStorage, Shape)> {
        let (F32(table_data), U32(ids_data)) = (&table.slice, &ids.slice) else {
            candle_core::bail!("sparse CUDA expects f32/u32")
        };
        let ids_data = view(ids_data, il)?;
        let device = &table.device;
        let (shape, count, name) = if grad.is_some() {
            (
                (op.rows, op.width),
                op.batch * op.items * op.width,
                "sparse_grad",
            )
        } else {
            ((op.batch, op.width), op.batch * op.width, "sparse_sum")
        };
        let mut output = device.alloc_zeros::<f32>(shape.0 * shape.1)?;
        let input = if let Some((grad, layout)) = grad {
            let F32(values) = &grad.slice else {
                candle_core::bail!("sparse CUDA gradient expects f32")
            };
            view(values, layout)?
        } else {
            view(table_data, tl)?
        };
        let function = device.get_or_load_custom_func(name, "pikafish_sparse_v1", ptx()?)?;
        let mut builder = function.builder();
        builder.arg(&input).arg(&ids_data).arg(&mut output);
        candle_core::builder_arg!(builder, op.batch as u32, op.items as u32, op.width as u32);
        unsafe { builder.launch(LaunchConfig::for_num_elems(count as u32)) }
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        Ok((
            CudaStorage {
                slice: F32(output),
                device: device.clone(),
            },
            shape.into(),
        ))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};
    fn check(device: &Device) -> Result<()> {
        let table = Var::from_slice(
            &(0..35).map(|i| i as f32 * 0.03).collect::<Vec<_>>(),
            (5, 7),
            device,
        )?;
        let groups: [&[usize]; 3] = [&[3, 1, 3], &[], &[4, 0]];
        let result = sums(&table, &groups)?;
        let values = result.to_vec2::<f32>()?;
        for (b, ids) in groups.iter().enumerate() {
            for h in 0..7 {
                let expected = ids.iter().map(|&i| (i * 7 + h) as f32 * 0.03).sum::<f32>();
                assert!((values[b][h] - expected).abs() < 1e-5);
            }
        }
        let weights = Tensor::from_vec((0..21).map(|i| i as f32 * 0.1).collect(), (3, 7), device)?;
        let gradients = (result * weights)?.sum_all()?.backward()?;
        let actual = gradients.get(&table).unwrap().to_vec2::<f32>()?;
        for row in 0..5 {
            for h in 0..7 {
                let expected = groups
                    .iter()
                    .enumerate()
                    .map(|(b, ids)| {
                        ids.iter().filter(|&&i| i == row).count() as f32 * (b * 7 + h) as f32 * 0.1
                    })
                    .sum::<f32>();
                assert!((actual[row][h] - expected).abs() < 1e-5);
            }
        }
        assert!(sums(&table, &[&[5]]).is_err());
        Ok(())
    }
    #[test]
    fn cpu_forward_and_duplicate_gradients() -> Result<()> {
        check(&Device::Cpu)
    }
    #[test]
    fn cuda_forward_and_duplicate_gradients() -> Result<()> {
        if let Ok(device) = Device::new_cuda(0) {
            check(&device)?;
        }
        Ok(())
    }
}
