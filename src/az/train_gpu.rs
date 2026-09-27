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
impl GpuTrainer {
    pub(super) fn new(_: &super::AzNnue, _: f32) -> candle_core::Result<Self> {
        candle_core::bail!("GPU training is disabled")
    }
    pub(super) fn save_state(&self, _: &std::path::Path, _: usize) -> candle_core::Result<()> {
        candle_core::bail!("GPU training is disabled")
    }
    pub(super) fn restore_state(
        &mut self,
        _: &std::path::Path,
        _: usize,
    ) -> candle_core::Result<()> {
        candle_core::bail!("GPU training is disabled")
    }
    pub(super) fn steps(&self) -> usize {
        0
    }
    pub(super) fn set_holdout(&mut self, _: Vec<super::AzTrainingSample>) {}
    pub(super) fn take_checks(&mut self) -> Vec<super::AzHoldoutReport> {
        Vec::new()
    }
    pub(super) fn last_learning_rate(&self) -> f32 {
        0.0
    }
}

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
    Err("GPU training is disabled by `--no-default-features`".to_string())
}
