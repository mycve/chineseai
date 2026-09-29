use std::sync::Arc;

use super::{AbNnue, AbTrainLossWeights, AbTrainStats, AbTrainingSample, SplitMix64};

pub fn train_samples(
    model: &mut AbNnue,
    samples: &[AbTrainingSample],
    epochs: usize,
    lr: f32,
    batch_size: usize,
    rng: &mut SplitMix64,
) -> Result<AbTrainStats, String> {
    train_samples_weighted(
        model,
        samples,
        epochs,
        lr,
        batch_size,
        rng,
        AbTrainLossWeights::default(),
    )
}

pub fn train_samples_weighted(
    model: &mut AbNnue,
    samples: &[AbTrainingSample],
    epochs: usize,
    lr: f32,
    batch_size: usize,
    rng: &mut SplitMix64,
    loss_weights: AbTrainLossWeights,
) -> Result<AbTrainStats, String> {
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
    model: &mut AbNnue,
    samples: Vec<AbTrainingSample>,
    epochs: usize,
    lr: f32,
    batch_size: usize,
    rng: &mut SplitMix64,
    loss_weights: AbTrainLossWeights,
) -> Result<AbTrainStats, String> {
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

pub fn train_samples_weighted_shared(
    model: &mut AbNnue,
    samples: Arc<Vec<AbTrainingSample>>,
    epochs: usize,
    lr: f32,
    batch_size: usize,
    rng: &mut SplitMix64,
    loss_weights: AbTrainLossWeights,
) -> Result<AbTrainStats, String> {
    super::train_gpu::train_samples_gpu(model, samples, epochs, lr, batch_size, rng, loss_weights)
}
