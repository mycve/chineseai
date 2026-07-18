#[cfg(any(
    all(feature = "gpu-train", not(target_os = "macos")),
    all(target_os = "linux", not(target_env = "musl"))
))]
use super::train_gpu_candle as candle;

#[cfg(any(
    all(feature = "gpu-train", not(target_os = "macos")),
    all(target_os = "linux", not(target_env = "musl"))
))]
pub(super) use candle::{GpuTrainer, train_samples_gpu};

#[cfg(any(
    all(feature = "gpu-train", not(target_os = "macos")),
    all(target_os = "linux", not(target_env = "musl"))
))]
pub(crate) use candle::training_cuda_device_count;

#[cfg(not(any(
    all(feature = "gpu-train", not(target_os = "macos")),
    all(target_os = "linux", not(target_env = "musl"))
)))]
#[derive(Debug)]
pub(super) struct GpuTrainer;

#[cfg(not(any(
    all(feature = "gpu-train", not(target_os = "macos")),
    all(target_os = "linux", not(target_env = "musl"))
)))]
pub(crate) fn training_cuda_device_count() -> usize {
    1
}

#[cfg(not(any(
    all(feature = "gpu-train", not(target_os = "macos")),
    all(target_os = "linux", not(target_env = "musl"))
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
