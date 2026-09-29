//! 可训练的 Pikafish 2026 NNUE 张量布局与分桶前向。
//! 本模块使用浮点训练权重；量化、特征抽取及 `.nnue` 序列化由独立路径负责。

use candle_core::{DType, Device, Result, Tensor, Var};

pub const PSQ_FEATURES: usize = 16_536;
pub const THREAT_FEATURES: usize = 45_547;
pub const TRANSFORMER_WIDTH: usize = 1_024;
pub const PSQT_BUCKETS: usize = 16;
pub const LAYER_STACKS: usize = 16;
pub const FC_WIDTH: usize = 32;

/// 形状与 Pikafish 的 FeatureTransformer 和 16 个 NetworkArchitecture 对齐。
/// 训练态权重统一使用 f32；PSQT 与跳连输出保持标量价值目标。
pub const PARAMETER_LAYOUT: &[(&str, &[usize])] = &[
    ("transformer.bias", &[TRANSFORMER_WIDTH]),
    ("transformer.psq", &[PSQ_FEATURES, TRANSFORMER_WIDTH]),
    ("transformer.threat", &[THREAT_FEATURES, TRANSFORMER_WIDTH]),
    ("transformer.psqt", &[PSQ_FEATURES, PSQT_BUCKETS]),
    ("transformer.threat_psqt", &[THREAT_FEATURES, PSQT_BUCKETS]),
    (
        "stacks.fc0.weight",
        &[LAYER_STACKS, TRANSFORMER_WIDTH, FC_WIDTH],
    ),
    ("stacks.fc0.bias", &[LAYER_STACKS, FC_WIDTH]),
    ("stacks.fc1.weight", &[LAYER_STACKS, FC_WIDTH * 2, FC_WIDTH]),
    ("stacks.fc1.bias", &[LAYER_STACKS, FC_WIDTH]),
    ("stacks.fc2.weight", &[LAYER_STACKS, FC_WIDTH * 4, 1]),
    ("stacks.fc2.bias", &[LAYER_STACKS, 1]),
];

pub fn parameter_count() -> usize {
    PARAMETER_LAYOUT
        .iter()
        .map(|(_, shape)| shape.iter().product::<usize>())
        .sum()
}

#[derive(Debug)]
pub struct PikafishStack {
    fc0_weight: Var,
    fc0_bias: Var,
    fc1_weight: Var,
    fc1_bias: Var,
    fc2_weight: Var,
    fc2_bias: Var,
}

impl PikafishStack {
    pub fn new(device: &Device) -> Result<Self> {
        Ok(Self {
            fc0_weight: Var::randn(0f32, 0.01, (TRANSFORMER_WIDTH, FC_WIDTH), device)?,
            fc0_bias: Var::zeros(FC_WIDTH, DType::F32, device)?,
            fc1_weight: Var::randn(0f32, 0.01, (FC_WIDTH * 2, FC_WIDTH), device)?,
            fc1_bias: Var::zeros(FC_WIDTH, DType::F32, device)?,
            fc2_weight: Var::randn(0f32, 0.01, (FC_WIDTH * 4, 1), device)?,
            fc2_bias: Var::zeros(1, DType::F32, device)?,
        })
    }

    /// 输入为两个视角各 512 维的乘积变换结果，形状 `[batch, 1024]`。
    /// 双激活分支和 fc0 的末两维差值跳连跟随当前 Pikafish 架构。
    pub fn forward(&self, transformed: &Tensor) -> Result<Tensor> {
        let fc0 = transformed
            .matmul(self.fc0_weight.as_tensor())?
            .broadcast_add(self.fc0_bias.as_tensor())?;
        let ac0 = paired_activation(&fc0, 1.0)?;
        let fc1 = ac0
            .matmul(self.fc1_weight.as_tensor())?
            .broadcast_add(self.fc1_bias.as_tensor())?;
        let ac1 = paired_activation(&fc1, 1.0)?;
        let features = Tensor::cat(&[&ac0, &ac1], 1)?;
        let output = features
            .matmul(self.fc2_weight.as_tensor())?
            .broadcast_add(self.fc2_bias.as_tensor())?;
        let skip = (fc0.narrow(1, FC_WIDTH - 2, 1)? - fc0.narrow(1, FC_WIDTH - 1, 1)?)?;
        output + skip
    }

    pub fn vars(&self) -> [&Var; 6] {
        [
            &self.fc0_weight,
            &self.fc0_bias,
            &self.fc1_weight,
            &self.fc1_bias,
            &self.fc2_weight,
            &self.fc2_bias,
        ]
    }
}

fn paired_activation(x: &Tensor, scale: f64) -> Result<Tensor> {
    let clipped = x.clamp(0.0, scale)?;
    let squared = clipped.sqr()?.affine(1.0 / scale, 0.0)?;
    Tensor::cat(&[&squared, &clipped], 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_matches_current_pikafish_dimensions() {
        assert_eq!(PSQ_FEATURES + THREAT_FEATURES, 62_083);
        assert_eq!(PARAMETER_LAYOUT.len(), 11);
        assert!(parameter_count() > 63_000_000);
    }

    #[test]
    fn stack_forward_is_trainable() -> Result<()> {
        let stack = PikafishStack::new(&Device::Cpu)?;
        let input = Tensor::ones((2, TRANSFORMER_WIDTH), DType::F32, &Device::Cpu)?;
        let output = stack.forward(&input)?;
        assert_eq!(output.dims(), &[2, 1]);
        let gradients = output.sum_all()?.backward()?;
        assert!(stack.vars().iter().all(|var| gradients.get(var).is_some()));
        Ok(())
    }
}
