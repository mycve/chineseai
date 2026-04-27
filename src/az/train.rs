use super::{AzModel, AzTrainLossWeights, AzTrainStats, AzTrainingSample, SplitMix64};

/// Number of samples covered by one global training step across visible GPUs.
pub fn global_training_step_sample_count(batch_size_per_gpu: usize) -> usize {
    batch_size_per_gpu.max(1) * super::train_gpu::training_cuda_device_count()
}

pub fn train_samples(
    model: &mut AzModel,
    samples: &[AzTrainingSample],
    epochs: usize,
    lr: f32,
    // Per-GPU micro-batch size for one training step.
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
    model: &mut AzModel,
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
