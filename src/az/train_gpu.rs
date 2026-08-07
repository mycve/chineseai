#[cfg(any(
    all(feature = "gpu-train", not(target_os = "macos")),
    all(target_os = "linux", not(target_env = "musl")),
    target_os = "windows",
))]
use super::train_gpu_candle as candle;

#[cfg(any(
    all(feature = "gpu-train", not(target_os = "macos")),
    all(target_os = "linux", not(target_env = "musl")),
    target_os = "windows",
))]
pub(super) use candle::GpuTrainer;

/// ??? GPU ???????????? `String` ??????????
#[cfg(any(
    all(feature = "gpu-train", not(target_os = "macos")),
    all(target_os = "linux", not(target_env = "musl")),
    target_os = "windows",
))]
pub(super) fn train_samples_gpu(
    model: &mut super::AzNnue,
    samples: std::sync::Arc<Vec<super::AzTrainingSample>>,
    epochs: usize,
    lr: f32,
    batch_size: usize,
    rng: &mut super::SplitMix64,
    loss_weights: super::AzTrainLossWeights,
) -> Result<super::AzTrainStats, String> {
    candle::train_samples_gpu(model, samples, epochs, lr, batch_size, rng, loss_weights)
        .map_err(|err| err.to_string())
}

#[cfg(not(any(
    all(feature = "gpu-train", not(target_os = "macos")),
    all(target_os = "linux", not(target_env = "musl")),
    target_os = "windows",
)))]
#[derive(Debug)]
pub(super) struct GpuTrainer;

#[cfg(not(any(
    all(feature = "gpu-train", not(target_os = "macos")),
    all(target_os = "linux", not(target_env = "musl")),
    target_os = "windows",
)))]
pub(super) fn train_samples_gpu(
    _model: &mut super::AzNnue,
    _samples: std::sync::Arc<Vec<super::AzTrainingSample>>,
    _epochs: usize,
    _lr: f32,
    _batch_size: usize,
    _rng: &mut super::SplitMix64,
    _loss_weights: super::AzTrainLossWeights,
) -> Result<super::AzTrainStats, String> {
    Err("GPU training is disabled; rebuild with `--features gpu-train`".to_string())
}
