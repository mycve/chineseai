use std::sync::OnceLock;

use candle_core::{CpuStorage, CustomOp3, Layout, Result, Shape, Tensor};

const CUDA_SOURCE: &str = r#"
extern "C" __global__ void bucket_linear_fwd(
    const float* params, const float* input, const unsigned int* buckets, float* output,
    unsigned int batch, unsigned int stacks, unsigned int outputs, unsigned int inputs
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch * outputs) return;
    unsigned int b = index / outputs;
    unsigned int o = index - b * outputs;
    unsigned int bucket = buckets[b];
    if (bucket >= stacks) { output[index] = 0.0f; return; }
    const float* row = params + (bucket * outputs + o) * inputs;
    const float* x = input + b * inputs;
    float sum = 0.0f;
    for (unsigned int i = 0; i < inputs; ++i) sum += row[i] * x[i];
    output[index] = sum;
}

extern "C" __global__ void bucket_linear_grad_params(
    const float* input, const unsigned int* buckets, const float* grad, float* output,
    unsigned int batch, unsigned int stacks, unsigned int outputs, unsigned int inputs
) {
    unsigned long long count = (unsigned long long)batch * outputs * inputs;
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    unsigned int i = index % inputs;
    unsigned long long bo = index / inputs;
    unsigned int o = bo % outputs;
    unsigned int b = bo / outputs;
    unsigned int bucket = buckets[b];
    if (bucket >= stacks) return;
    atomicAdd(output + ((bucket * outputs + o) * inputs + i),
              grad[b * outputs + o] * input[b * inputs + i]);
}

extern "C" __global__ void bucket_linear_grad_input(
    const float* params, const unsigned int* buckets, const float* grad, float* output,
    unsigned int batch, unsigned int stacks, unsigned int outputs, unsigned int inputs,
    unsigned long long param_len
) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (unsigned long long)batch * inputs) return;
    unsigned int i = index % inputs;
    unsigned int b = index / inputs;
    unsigned int bucket = buckets[b];
    if (bucket >= stacks) { output[param_len + index] = 0.0f; return; }
    float sum = 0.0f;
    for (unsigned int o = 0; o < outputs; ++o) {
        sum += grad[b * outputs + o]
             * params[(bucket * outputs + o) * inputs + i];
    }
    output[param_len + index] = sum;
}
"#;

#[derive(Clone, Debug)]
struct BucketLinear {
    batch: usize,
    stacks: usize,
    outputs: usize,
    inputs: usize,
}

#[derive(Clone, Debug)]
struct BucketLinearGrad(BucketLinear);

/// Per-sample affine transform selected by a small integer bucket. Bias is folded into an
/// appended constant input so the custom op needs only three differentiable Candle operands.
pub(super) fn bucket_affine(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    buckets: &Tensor,
    stacks: usize,
) -> Result<Tensor> {
    let (batch, input_size) = input.dims2()?;
    let (weight_rows, weight_inputs) = weight.dims2()?;
    let bias_rows = bias.dims1()?;
    let bucket_batch = buckets.dims1()?;
    if weight_inputs != input_size
        || weight_rows != bias_rows
        || weight_rows % stacks != 0
        || bucket_batch != batch
    {
        candle_core::bail!(
            "bucket affine shape mismatch: input={:?} weight={:?} bias={:?} buckets={:?} stacks={stacks}",
            input.dims(),
            weight.dims(),
            bias.dims(),
            buckets.dims()
        )
    }
    let outputs = weight_rows / stacks;
    let params = Tensor::cat(&[weight, &bias.unsqueeze(1)?], 1)?.contiguous()?;
    let ones = Tensor::ones((batch, 1), input.dtype(), input.device())?;
    let augmented = Tensor::cat(&[input, &ones], 1)?.contiguous()?;
    params.apply_op3(
        &augmented,
        buckets,
        BucketLinear {
            batch,
            stacks,
            outputs,
            inputs: input_size + 1,
        },
    )
}

impl CustomOp3 for BucketLinear {
    fn name(&self) -> &'static str {
        "az-bucket-linear"
    }

    fn cpu_fwd(
        &self,
        param_storage: &CpuStorage,
        param_layout: &Layout,
        input_storage: &CpuStorage,
        input_layout: &Layout,
        bucket_storage: &CpuStorage,
        bucket_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (CpuStorage::F32(params), CpuStorage::F32(input), CpuStorage::U32(buckets)) =
            (param_storage, input_storage, bucket_storage)
        else {
            candle_core::bail!("bucket affine expects f32/f32/u32 inputs")
        };
        let params = contiguous(params, param_layout, "bucket affine params")?;
        let input = contiguous(input, input_layout, "bucket affine input")?;
        let buckets = contiguous(buckets, bucket_layout, "bucket affine buckets")?;
        let mut output = vec![0.0f32; self.batch * self.outputs];
        for b in 0..self.batch {
            let bucket = buckets[b] as usize;
            if bucket >= self.stacks {
                continue;
            }
            for o in 0..self.outputs {
                let row = &params[(bucket * self.outputs + o) * self.inputs
                    ..(bucket * self.outputs + o + 1) * self.inputs];
                let x = &input[b * self.inputs..(b + 1) * self.inputs];
                output[b * self.outputs + o] = row.iter().zip(x).map(|(&w, &v)| w * v).sum();
            }
        }
        Ok((CpuStorage::F32(output), (self.batch, self.outputs).into()))
    }

    #[cfg(any(
        target_os = "windows",
        all(target_os = "linux", not(target_env = "musl"))
    ))]
    fn cuda_fwd(
        &self,
        params: &candle_core::CudaStorage,
        param_layout: &Layout,
        input: &candle_core::CudaStorage,
        input_layout: &Layout,
        buckets: &candle_core::CudaStorage,
        bucket_layout: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        cuda_forward(
            params,
            param_layout,
            input,
            input_layout,
            buckets,
            bucket_layout,
            self,
        )
    }

    fn bwd(
        &self,
        params: &Tensor,
        input: &Tensor,
        buckets: &Tensor,
        _result: &Tensor,
        grad_result: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        let input_grad = Tensor::cat(&[&input.flatten_all()?, &grad_result.flatten_all()?], 0)?;
        let grads =
            params.apply_op3_no_bwd(buckets, &input_grad, &BucketLinearGrad(self.clone()))?;
        let param_len = self.stacks * self.outputs * self.inputs;
        let input_len = self.batch * self.inputs;
        Ok((
            Some(grads.narrow(0, 0, param_len)?.reshape(params.shape())?),
            Some(
                grads
                    .narrow(0, param_len, input_len)?
                    .reshape(input.shape())?,
            ),
            None,
        ))
    }
}

impl CustomOp3 for BucketLinearGrad {
    fn name(&self) -> &'static str {
        "az-bucket-linear-grad"
    }

    fn cpu_fwd(
        &self,
        param_storage: &CpuStorage,
        param_layout: &Layout,
        bucket_storage: &CpuStorage,
        bucket_layout: &Layout,
        input_grad_storage: &CpuStorage,
        input_grad_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (CpuStorage::F32(params), CpuStorage::U32(buckets), CpuStorage::F32(input_grad)) =
            (param_storage, bucket_storage, input_grad_storage)
        else {
            candle_core::bail!("bucket affine gradient expects f32/u32/f32 inputs")
        };
        let op = &self.0;
        let params = contiguous(params, param_layout, "bucket affine params")?;
        let buckets = contiguous(buckets, bucket_layout, "bucket affine buckets")?;
        let input_grad = contiguous(input_grad, input_grad_layout, "bucket affine input/grad")?;
        let input_len = op.batch * op.inputs;
        let (input, grad) = input_grad.split_at(input_len);
        let param_len = op.stacks * op.outputs * op.inputs;
        let mut output = vec![0.0f32; param_len + input_len];
        let (grad_params, grad_input) = output.split_at_mut(param_len);
        for b in 0..op.batch {
            let bucket = buckets[b] as usize;
            if bucket >= op.stacks {
                continue;
            }
            for o in 0..op.outputs {
                let g = grad[b * op.outputs + o];
                let row = (bucket * op.outputs + o) * op.inputs;
                for i in 0..op.inputs {
                    grad_params[row + i] += g * input[b * op.inputs + i];
                    grad_input[b * op.inputs + i] += g * params[row + i];
                }
            }
        }
        Ok((CpuStorage::F32(output), (param_len + input_len).into()))
    }

    #[cfg(any(
        target_os = "windows",
        all(target_os = "linux", not(target_env = "musl"))
    ))]
    fn cuda_fwd(
        &self,
        params: &candle_core::CudaStorage,
        param_layout: &Layout,
        buckets: &candle_core::CudaStorage,
        bucket_layout: &Layout,
        input_grad: &candle_core::CudaStorage,
        input_grad_layout: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        cuda_gradient(
            params,
            param_layout,
            buckets,
            bucket_layout,
            input_grad,
            input_grad_layout,
            &self.0,
        )
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
fn bucket_ptx() -> Result<&'static str> {
    static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    match PTX.get_or_init(|| {
        candle_core::cuda_backend::cudarc::nvrtc::safe::compile_ptx(CUDA_SOURCE)
            .map(|ptx| ptx.to_src())
            .map_err(|error| format!("failed to compile bucket affine CUDA kernels: {error}"))
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
    param_storage: &candle_core::CudaStorage,
    param_layout: &Layout,
    input_storage: &candle_core::CudaStorage,
    input_layout: &Layout,
    bucket_storage: &candle_core::CudaStorage,
    bucket_layout: &Layout,
    op: &BucketLinear,
) -> Result<(candle_core::CudaStorage, Shape)> {
    use candle_core::cuda_backend::CudaStorageSlice::{F32, U32};
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    let (F32(params), F32(input), U32(buckets)) = (
        &param_storage.slice,
        &input_storage.slice,
        &bucket_storage.slice,
    ) else {
        candle_core::bail!("bucket affine CUDA kernel expects f32/f32/u32 inputs")
    };
    let params = cuda_view(params, param_layout, "bucket affine params")?;
    let input = cuda_view(input, input_layout, "bucket affine input")?;
    let buckets = cuda_view(buckets, bucket_layout, "bucket affine buckets")?;
    let device = &param_storage.device;
    let output = unsafe { device.alloc::<f32>(op.batch * op.outputs)? };
    let function = device.get_or_load_custom_func(
        "bucket_linear_fwd",
        "chineseai_bucket_linear_v1",
        bucket_ptx()?,
    )?;
    let mut builder = function.builder();
    builder.arg(&params).arg(&input).arg(&buckets).arg(&output);
    candle_core::builder_arg!(
        builder,
        op.batch as u32,
        op.stacks as u32,
        op.outputs as u32,
        op.inputs as u32
    );
    unsafe { builder.launch(LaunchConfig::for_num_elems((op.batch * op.outputs) as u32)) }
        .map_err(|error| {
            candle_core::Error::Msg(format!("bucket affine launch failed: {error}"))
        })?;
    Ok((
        candle_core::CudaStorage {
            slice: F32(output),
            device: device.clone(),
        },
        (op.batch, op.outputs).into(),
    ))
}

#[cfg(any(
    target_os = "windows",
    all(target_os = "linux", not(target_env = "musl"))
))]
fn cuda_gradient(
    param_storage: &candle_core::CudaStorage,
    param_layout: &Layout,
    bucket_storage: &candle_core::CudaStorage,
    bucket_layout: &Layout,
    input_grad_storage: &candle_core::CudaStorage,
    input_grad_layout: &Layout,
    op: &BucketLinear,
) -> Result<(candle_core::CudaStorage, Shape)> {
    use candle_core::cuda_backend::CudaStorageSlice::{F32, U32};
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    let (F32(params), U32(buckets), F32(input_grad)) = (
        &param_storage.slice,
        &bucket_storage.slice,
        &input_grad_storage.slice,
    ) else {
        candle_core::bail!("bucket affine gradient CUDA kernel expects f32/u32/f32 inputs")
    };
    let params = cuda_view(params, param_layout, "bucket affine params")?;
    let buckets = cuda_view(buckets, bucket_layout, "bucket affine buckets")?;
    let input_grad = cuda_view(input_grad, input_grad_layout, "bucket affine input/grad")?;
    let device = &param_storage.device;
    let param_len = op.stacks * op.outputs * op.inputs;
    let input_len = op.batch * op.inputs;
    let output = device.alloc_zeros::<f32>(param_len + input_len)?;
    let grad_output = input_grad.slice((op.batch * op.inputs)..);
    let grad_params = device.get_or_load_custom_func(
        "bucket_linear_grad_params",
        "chineseai_bucket_linear_v1",
        bucket_ptx()?,
    )?;
    let mut builder = grad_params.builder();
    builder
        .arg(&input_grad)
        .arg(&buckets)
        .arg(&grad_output)
        .arg(&output);
    candle_core::builder_arg!(
        builder,
        op.batch as u32,
        op.stacks as u32,
        op.outputs as u32,
        op.inputs as u32
    );
    let param_work = op.batch * op.outputs * op.inputs;
    unsafe { builder.launch(LaunchConfig::for_num_elems(param_work as u32)) }.map_err(|error| {
        candle_core::Error::Msg(format!(
            "bucket affine parameter gradient launch failed: {error}"
        ))
    })?;
    let grad_input = device.get_or_load_custom_func(
        "bucket_linear_grad_input",
        "chineseai_bucket_linear_v1",
        bucket_ptx()?,
    )?;
    let mut builder = grad_input.builder();
    builder
        .arg(&params)
        .arg(&buckets)
        .arg(&grad_output)
        .arg(&output);
    candle_core::builder_arg!(
        builder,
        op.batch as u32,
        op.stacks as u32,
        op.outputs as u32,
        op.inputs as u32,
        param_len as u64
    );
    unsafe { builder.launch(LaunchConfig::for_num_elems(input_len as u32)) }.map_err(|error| {
        candle_core::Error::Msg(format!(
            "bucket affine input gradient launch failed: {error}"
        ))
    })?;
    Ok((
        candle_core::CudaStorage {
            slice: F32(output),
            device: device.clone(),
        },
        (param_len + input_len).into(),
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

    type ForwardAndGrads = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

    fn output_and_grads(device: &Device) -> Result<ForwardAndGrads> {
        let input = Var::from_slice(
            &[
                0.1f32, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8, 0.9, 1.0, -1.1, 1.2,
            ],
            (3, 4),
            device,
        )?;
        let weight_values = (0..2 * 3 * 4)
            .map(|index| (index as f32 - 11.0) * 0.013)
            .collect::<Vec<_>>();
        let weight = Var::from_slice(&weight_values, (2 * 3, 4), device)?;
        let bias = Var::from_slice(&[0.03f32, -0.04, 0.05, -0.06, 0.07, -0.08], 2 * 3, device)?;
        let buckets = Tensor::from_slice(&[1u32, 0, 1], 3, device)?;
        let output = bucket_affine(&input, &weight, &bias, &buckets, 2)?;
        let grads = output.sqr()?.sum_all()?.backward()?;
        Ok((
            output.flatten_all()?.to_vec1()?,
            grads.get(&input).unwrap().flatten_all()?.to_vec1()?,
            grads.get(&weight).unwrap().flatten_all()?.to_vec1()?,
            grads.get(&bias).unwrap().flatten_all()?.to_vec1()?,
        ))
    }

    fn assert_close(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        for (&left, &right) in left.iter().zip(right) {
            assert!((left - right).abs() < 3.0e-5, "left={left} right={right}");
        }
    }

    #[test]
    fn bucket_affine_cuda_matches_cpu_forward_and_gradients() -> Result<()> {
        let cpu = output_and_grads(&Device::Cpu)?;
        let Ok(cuda) = Device::new_cuda(0) else {
            return Ok(());
        };
        let gpu = output_and_grads(&cuda)?;
        assert_close(&cpu.0, &gpu.0);
        assert_close(&cpu.1, &gpu.1);
        assert_close(&cpu.2, &gpu.2);
        assert_close(&cpu.3, &gpu.3);
        Ok(())
    }
}
