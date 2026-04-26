use super::{AzNnue, AzTrainLossWeights, AzTrainStats, AzTrainingSample, SplitMix64};

/// 单步训练在全部 GPU 上合计覆盖的样本数（= 配置中 `batch_size` × 可见 CUDA 卡数，至少 1 卡）
pub fn global_training_step_sample_count(batch_size_per_gpu: usize) -> usize {
    batch_size_per_gpu.max(1) * super::train_gpu::training_cuda_device_count()
}

pub fn train_samples(
    model: &mut AzNnue,
    samples: &[AzTrainingSample],
    epochs: usize,
    lr: f32,
    // 每卡、每步 micro-batch 样本数；整 epoch 的步数约 ceil(样本数 / (每卡 batch×GPU 数))
    batch_size: usize,
    rng: &mut SplitMix64,
) -> AzTrainStats {
    train_samples_weighted(
        model,
        samples,
        epochs,
        lr,
        batch_size,
        rng,
        AzTrainLossWeights::default(),
    )
}

pub fn train_samples_weighted(
    model: &mut AzNnue,
    samples: &[AzTrainingSample],
    epochs: usize,
    lr: f32,
    batch_size: usize,
    rng: &mut SplitMix64,
    loss_weights: AzTrainLossWeights,
) -> AzTrainStats {
    match super::train_gpu::train_samples_gpu(
        model,
        samples,
        epochs,
        lr,
        batch_size,
        rng,
        loss_weights,
    ) {
        Ok(stats) => stats,
        Err(err) => panic!("Candle CUDA training failed: {err}"),
    }
}
