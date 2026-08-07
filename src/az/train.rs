use std::sync::Arc;

use super::{AzNnue, AzTrainLossWeights, AzTrainStats, AzTrainingSample, SplitMix64};

/// ????????????????????? batch size?????????
pub fn global_training_step_sample_count(global_batch_size: usize) -> usize {
    global_batch_size.max(1)
}

pub fn train_samples(
    model: &mut AzNnue,
    samples: &[AzTrainingSample],
    epochs: usize,
    lr: f32,
    batch_size: usize,
    rng: &mut SplitMix64,
) -> Result<AzTrainStats, String> {
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
) -> Result<AzTrainStats, String> {
    train_samples_weighted_shared(
        model,
        Arc::new(samples.to_vec()),
        epochs,
        lr,
        batch_size,
        rng,
        loss_weights,
    )
}

pub fn train_samples_weighted_owned(
    model: &mut AzNnue,
    samples: Vec<AzTrainingSample>,
    epochs: usize,
    lr: f32,
    batch_size: usize,
    rng: &mut SplitMix64,
    loss_weights: AzTrainLossWeights,
) -> Result<AzTrainStats, String> {
    train_samples_weighted_shared(
        model,
        Arc::new(samples),
        epochs,
        lr,
        batch_size,
        rng,
        loss_weights,
    )
}

fn train_samples_weighted_shared(
    model: &mut AzNnue,
    samples: Arc<Vec<AzTrainingSample>>,
    epochs: usize,
    lr: f32,
    batch_size: usize,
    rng: &mut SplitMix64,
    loss_weights: AzTrainLossWeights,
) -> Result<AzTrainStats, String> {
    super::train_gpu::train_samples_gpu(
        model,
        samples,
        epochs,
        lr,
        batch_size,
        rng,
        loss_weights,
    )
}
