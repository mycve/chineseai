use std::sync::Arc;

use super::{AzNnue, AzTrainLossWeights, AzTrainStats, AzTrainingSample, SplitMix64};

#[derive(Clone, Copy, Debug, Default)]
pub struct AzValueRankingStats {
    pub pairs: usize,
    pub loss: f32,
    pub accuracy: f32,
    pub mean_margin: f32,
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
    super::train_gpu::train_samples_gpu(model, samples, epochs, lr, batch_size, rng, loss_weights)
}

/// Trains only the relative ordering of sibling child positions. Each tuple is
/// `(preferred_child, rejected_child)`. Since both children are evaluated from
/// the opponent's perspective, the desired margin is Q(rejected) - Q(preferred).
pub fn train_value_ranking_pairs(
    model: &mut AzNnue,
    pairs: &[(AzTrainingSample, AzTrainingSample)],
    epochs: usize,
    lr: f32,
    batch_size: usize,
    scale: f32,
    rng: &mut SplitMix64,
) -> Result<AzValueRankingStats, String> {
    super::train_gpu::train_value_ranking_pairs_gpu(
        model, pairs, epochs, lr, batch_size, scale, rng,
    )
}
