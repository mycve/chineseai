use std::sync::Arc;

use super::{AzNnue, AzTrainLossWeights, AzTrainStats, AzTrainingSample, SplitMix64};

/// 训练统一使用 AdamW；学习率按累计优化器步数预热和衰减。
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum AzTrainOptimizer {
    /// 支持状态恢复、学习率调度和逐张量 weight decay 分组。
    #[serde(rename = "adamw")]
    #[default]
    AdamW,
}

impl AzTrainOptimizer {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AdamW => "adamw",
        }
    }
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
        AzTrainOptimizer::default(),
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
    train_samples_weighted_owned_with_optimizer(
        model,
        samples,
        epochs,
        lr,
        batch_size,
        rng,
        loss_weights,
        AzTrainOptimizer::default(),
    )
}

/// 自博弈训练入口；配置仅接受 AdamW。
#[allow(clippy::too_many_arguments)]
pub fn train_samples_weighted_owned_with_optimizer(
    model: &mut AzNnue,
    samples: Vec<AzTrainingSample>,
    epochs: usize,
    lr: f32,
    batch_size: usize,
    rng: &mut SplitMix64,
    loss_weights: AzTrainLossWeights,
    optimizer: AzTrainOptimizer,
) -> Result<AzTrainStats, String> {
    train_samples_weighted_shared(
        model,
        Arc::new(samples),
        epochs,
        lr,
        batch_size,
        rng,
        loss_weights,
        optimizer,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn train_samples_weighted_shared(
    model: &mut AzNnue,
    samples: Arc<Vec<AzTrainingSample>>,
    epochs: usize,
    lr: f32,
    batch_size: usize,
    rng: &mut SplitMix64,
    loss_weights: AzTrainLossWeights,
    optimizer: AzTrainOptimizer,
) -> Result<AzTrainStats, String> {
    super::train_gpu::train_samples_gpu(
        model,
        samples,
        epochs,
        lr,
        batch_size,
        rng,
        loss_weights,
        optimizer,
    )
}
