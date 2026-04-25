use super::{AzNnue, AzTrainStats, AzTrainingSample, SplitMix64};

pub fn train_samples(
    model: &mut AzNnue,
    samples: &[AzTrainingSample],
    epochs: usize,
    lr: f32,
    batch_size: usize,
    rng: &mut SplitMix64,
) -> AzTrainStats {
    match super::train_gpu::train_samples_gpu(model, samples, epochs, lr, batch_size, rng) {
        Ok(stats) => stats,
        Err(err) => panic!("Candle CUDA training failed: {err}"),
    }
}
