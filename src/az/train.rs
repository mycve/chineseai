use std::sync::Arc;

use super::{AzNnue, AzTrainLossWeights, AzTrainStats, AzTrainingSample, SplitMix64};

/// 训练用的优化器内核算子。
///
/// 两个臂的差别不只是"算子"：`AdamW` 用常数 `lr=4e-4`，`Px0Sgd` 用 `config.lr`（默认 0.02）
/// 加 250 步 warmup 与按累计步数分段的阶梯。两者的学习率尺度不可直接比较（逐坐标自适应
/// 的步长 ≈ lr，带动量 SGD 的步长 ∝ lr·g），所以消融时"优化器 + 其学习率"要作为一个处理
/// 一起变、一起记录。
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum AzTrainOptimizer {
    /// Px0 `tfprocess.py` 的 SGD(momentum=0.9, nesterov=True)，原样移植。
    #[serde(rename = "px0-sgd")]
    Px0Sgd,
    /// 手写 AdamW（见 `az/adamw.rs`）：常数 lr、逐张量分组决定是否施加 weight decay。
    #[serde(rename = "adamw")]
    #[default]
    AdamW,
}

impl AzTrainOptimizer {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Px0Sgd => "px0-sgd",
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

/// 与 [`train_samples_weighted_owned`] 相同，但显式指定优化器内核。自博弈循环用这个入口，
/// 让 `train_optimizer` 这个 TOML 选项真正生效。
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
