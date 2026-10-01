//! 可训练的 Pikafish 2026 NNUE 张量布局与分桶前向。
//! 本模块使用浮点训练权重；量化、特征抽取及 `.nnue` 序列化由独立路径负责。

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use crate::xiangqi::Position;
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

/// 每一局面的两个视角，顺序为行棋方、对手。
/// 输入必须使用当前 Pikafish 特征编号，不能使用旧 AB 稀疏特征。
#[derive(Clone, Debug)]
pub struct PikafishExample {
    pub psq: [Vec<usize>; 2],
    pub threats: [Vec<usize>; 2],
    pub psqt_bucket: usize,
    pub layer_stack: usize,
}

impl PikafishExample {
    pub fn from_position(position: &Position) -> Option<Self> {
        let side = position.side_to_move();
        let mut psq = [Vec::new(), Vec::new()];
        let mut threats = [Vec::new(), Vec::new()];
        for (index, perspective) in [side, side.opposite()].into_iter().enumerate() {
            crate::nnue::pikafish::fill_psq_features(position, perspective, &mut psq[index])?;
            crate::nnue::pikafish::fill_threat_features(
                position,
                perspective,
                &mut threats[index],
            )?;
        }
        let material_bucket = crate::nnue::pikafish::layer_stack_bucket(position);
        Some(Self {
            psq,
            threats,
            psqt_bucket: material_bucket,
            layer_stack: material_bucket,
        })
    }
}

/// 缩小特征表可用于单元测试；生产路径使用 `production` 的精确形状。
#[derive(Clone, Copy, Debug)]
pub struct PikafishShape {
    pub psq_features: usize,
    pub threat_features: usize,
}

impl PikafishShape {
    pub const fn production() -> Self {
        Self {
            psq_features: PSQ_FEATURES,
            threat_features: THREAT_FEATURES,
        }
    }
}

#[derive(Debug)]
pub struct PikafishModel {
    shape: PikafishShape,
    transformer_bias: Var,
    transformer_psq: Var,
    transformer_threat: Var,
    psqt: Var,
    threat_psqt: Var,
    stacks: Vec<PikafishStack>,
}

impl PikafishModel {
    pub fn new(device: &Device) -> Result<Self> {
        Self::with_shape(PikafishShape::production(), device)
    }

    pub fn with_shape(shape: PikafishShape, device: &Device) -> Result<Self> {
        let mut stacks = Vec::with_capacity(LAYER_STACKS);
        for _ in 0..LAYER_STACKS {
            stacks.push(PikafishStack::new(device)?);
        }
        Ok(Self {
            shape,
            transformer_bias: Var::from_tensor(&Tensor::full(128f32, TRANSFORMER_WIDTH, device)?)?,
            transformer_psq: Var::randn(
                0f32,
                0.01,
                (shape.psq_features, TRANSFORMER_WIDTH),
                device,
            )?,
            transformer_threat: Var::randn(
                0f32,
                0.01,
                (shape.threat_features, TRANSFORMER_WIDTH),
                device,
            )?,
            psqt: Var::zeros((shape.psq_features, PSQT_BUCKETS), DType::F32, device)?,
            threat_psqt: Var::zeros((shape.threat_features, PSQT_BUCKETS), DType::F32, device)?,
            stacks,
        })
    }

    /// 返回 `[batch, 1]` 价值分数。乘积门按量化模型的
    /// `clamp(a,0,255) * clamp(b,0,255) / 512` 定义；进入浮点 FC 前
    /// 再除以 128，避免随机初始化时双激活分支饱和。
    pub fn forward(&self, examples: &[PikafishExample]) -> Result<Tensor> {
        self.forward_mode(examples, true)
    }

    /// 搜索不构建反向图，避免每个叶节点创建训练图和变量依赖。
    pub fn forward_inference(&self, examples: &[PikafishExample]) -> Result<Tensor> {
        self.forward_mode(examples, false)
    }

    fn forward_mode(&self, examples: &[PikafishExample], training: bool) -> Result<Tensor> {
        if examples.is_empty() {
            candle_core::bail!("empty Pikafish batch");
        }
        let mut outputs = Vec::with_capacity(examples.len());
        for example in examples {
            if example.psqt_bucket >= PSQT_BUCKETS || example.layer_stack >= LAYER_STACKS {
                candle_core::bail!("Pikafish bucket out of range");
            }
            let mut perspectives = Vec::with_capacity(2);
            let mut psqt_values = Vec::with_capacity(2);
            for side in 0..2 {
                let psq_acc = sparse_sum(
                    &weight(&self.transformer_psq, training),
                    &example.psq[side],
                    self.shape.psq_features,
                )?;
                let threat_acc = sparse_sum(
                    &weight(&self.transformer_threat, training),
                    &example.threats[side],
                    self.shape.threat_features,
                )?;
                let acc = (weight(&self.transformer_bias, training) + psq_acc + threat_acc)?;
                let a = acc
                    .narrow(0, 0, TRANSFORMER_WIDTH / 2)?
                    .clamp(0f32, 255f32)?;
                let b = acc
                    .narrow(0, TRANSFORMER_WIDTH / 2, TRANSFORMER_WIDTH / 2)?
                    .clamp(0f32, 255f32)?;
                perspectives.push((a * b)?.affine(1.0 / (512.0 * 128.0), 0.0)?);
                let psqt = sparse_sum(
                    &weight(&self.psqt, training),
                    &example.psq[side],
                    self.shape.psq_features,
                )?;
                let threat_psqt = sparse_sum(
                    &weight(&self.threat_psqt, training),
                    &example.threats[side],
                    self.shape.threat_features,
                )?;
                let bucket_value = (psqt + threat_psqt)?.narrow(0, example.psqt_bucket, 1)?;
                psqt_values.push(bucket_value);
            }
            let transformed = Tensor::cat(&[&perspectives[0], &perspectives[1]], 0)?
                .reshape((1, TRANSFORMER_WIDTH))?;
            let stack_out =
                self.stacks[example.layer_stack].forward_mode(&transformed, training)?;
            let psqt_delta = ((&psqt_values[0] - &psqt_values[1])? / 2.0)?.reshape((1, 1))?;
            outputs.push((stack_out + psqt_delta)?);
        }
        Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 0)
    }

    pub fn vars(&self) -> Vec<Var> {
        let mut vars = vec![
            self.transformer_bias.clone(),
            self.transformer_psq.clone(),
            self.transformer_threat.clone(),
            self.psqt.clone(),
            self.threat_psqt.clone(),
        ];
        for stack in &self.stacks {
            vars.extend(stack.vars().into_iter().cloned());
        }
        vars
    }

    /// 创建独立变量的不可变快照，不重复随机初始化；多个搜索 worker 共享这份快照。
    pub fn snapshot(&self, device: &Device) -> Result<Self> {
        let copy = |value: &Var| Var::from_tensor(&value.as_tensor().detach().to_device(device)?);
        let stacks = self
            .stacks
            .iter()
            .map(|stack| {
                Ok(PikafishStack {
                    fc0_weight: copy(&stack.fc0_weight)?,
                    fc0_bias: copy(&stack.fc0_bias)?,
                    fc1_weight: copy(&stack.fc1_weight)?,
                    fc1_bias: copy(&stack.fc1_bias)?,
                    fc2_weight: copy(&stack.fc2_weight)?,
                    fc2_bias: copy(&stack.fc2_bias)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            shape: self.shape,
            transformer_bias: copy(&self.transformer_bias)?,
            transformer_psq: copy(&self.transformer_psq)?,
            transformer_threat: copy(&self.transformer_threat)?,
            psqt: copy(&self.psqt)?,
            threat_psqt: copy(&self.threat_psqt)?,
            stacks,
        })
    }

    fn named_vars(&self) -> BTreeMap<String, Var> {
        let mut vars = BTreeMap::new();
        for (name, var) in [
            ("transformer.bias", &self.transformer_bias),
            ("transformer.psq", &self.transformer_psq),
            ("transformer.threat", &self.transformer_threat),
            ("transformer.psqt", &self.psqt),
            ("transformer.threat_psqt", &self.threat_psqt),
        ] {
            vars.insert(name.to_owned(), var.clone());
        }
        for (index, stack) in self.stacks.iter().enumerate() {
            for (name, var) in [
                "fc0.weight",
                "fc0.bias",
                "fc1.weight",
                "fc1.bias",
                "fc2.weight",
                "fc2.bias",
            ]
            .into_iter()
            .zip(stack.vars())
            {
                vars.insert(format!("stacks.{index}.{name}"), var.clone());
            }
        }
        vars
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let tensors: HashMap<_, _> = self
            .named_vars()
            .into_iter()
            .map(|(name, var)| (name, var.as_tensor().clone()))
            .collect();
        candle_core::safetensors::save(&tensors, path)
    }

    pub fn load(&self, path: &Path) -> Result<()> {
        let device = self.transformer_bias.device();
        let tensors = candle_core::safetensors::load(path, device)?;
        let named = self.named_vars();
        if tensors.len() != named.len() {
            candle_core::bail!("Pikafish checkpoint tensor count mismatch");
        }
        for (name, var) in named {
            let tensor = tensors
                .get(&name)
                .ok_or_else(|| candle_core::Error::Msg(format!("missing {name}")))?;
            if tensor.shape() != var.shape() {
                candle_core::bail!("shape mismatch for {name}");
            }
            if tensor.dtype() != DType::F32 {
                candle_core::bail!("dtype mismatch for {name}");
            }
            var.set(tensor)?;
        }
        Ok(())
    }
}

fn weight(value: &Var, training: bool) -> Tensor {
    if training {
        value.as_tensor().clone()
    } else {
        value.as_tensor().detach()
    }
}

fn sparse_sum(table: &Tensor, indices: &[usize], rows: usize) -> Result<Tensor> {
    if indices.iter().any(|&index| index >= rows) {
        candle_core::bail!("Pikafish feature out of range");
    }
    if indices.is_empty() {
        return Tensor::zeros(table.dim(1)?, DType::F32, table.device());
    }
    let ids: Vec<u32> = indices.iter().map(|&index| index as u32).collect();
    table
        .index_select(&Tensor::from_vec(ids, indices.len(), table.device())?, 0)?
        .sum(0)
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
        self.forward_mode(transformed, true)
    }

    fn forward_mode(&self, transformed: &Tensor, training: bool) -> Result<Tensor> {
        let fc0 = transformed
            .matmul(&weight(&self.fc0_weight, training))?
            .broadcast_add(&weight(&self.fc0_bias, training))?;
        let ac0 = paired_activation(&fc0, 1.0)?;
        let fc1 = ac0
            .matmul(&weight(&self.fc1_weight, training))?
            .broadcast_add(&weight(&self.fc1_bias, training))?;
        let ac1 = paired_activation(&fc1, 1.0)?;
        let features = Tensor::cat(&[&ac0, &ac1], 1)?;
        let output = features
            .matmul(&weight(&self.fc2_weight, training))?
            .broadcast_add(&weight(&self.fc2_bias, training))?;
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
        let example = PikafishExample::from_position(&Position::startpos()).unwrap();
        assert_eq!(example.psq[0].len(), 32);
        assert_eq!(example.psq[1].len(), 32);
        assert_eq!(example.layer_stack, 11);
        assert_eq!(example.psqt_bucket, 11);
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

    #[test]
    fn full_forward_backpropagates_to_sparse_tables_and_stack() -> Result<()> {
        let model = PikafishModel::with_shape(
            PikafishShape {
                psq_features: 8,
                threat_features: 8,
            },
            &Device::Cpu,
        )?;
        let examples = [PikafishExample {
            psq: [vec![1, 2], vec![3]],
            threats: [vec![1], vec![2]],
            psqt_bucket: 4,
            layer_stack: 7,
        }];
        let output = model.forward(&examples)?;
        assert_eq!(output.dims(), &[1, 1]);
        let gradients = output.sqr()?.sum_all()?.backward()?;
        let nonzero_gradient = |var: &Var| -> Result<bool> {
            Ok(gradients
                .get(var)
                .unwrap()
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .any(|&value| value.abs() > 1.0e-12))
        };
        assert!(nonzero_gradient(&model.transformer_psq)?);
        assert!(nonzero_gradient(&model.transformer_threat)?);
        assert!(gradients.get(&model.psqt).is_some());
        assert!(gradients.get(&model.threat_psqt).is_some());
        assert!(
            model.stacks[7]
                .vars()
                .iter()
                .all(|var| gradients.get(var).is_some())
        );
        assert!(nonzero_gradient(&model.stacks[7].fc0_weight)?);
        Ok(())
    }

    #[test]
    fn inference_matches_training_and_snapshot_survives_updates() -> Result<()> {
        let model = PikafishModel::with_shape(
            PikafishShape {
                psq_features: 8,
                threat_features: 8,
            },
            &Device::Cpu,
        )?;
        let example = PikafishExample {
            psq: [vec![1, 2], vec![3]],
            threats: [vec![1], vec![2]],
            psqt_bucket: 4,
            layer_stack: 7,
        };
        let examples = [example];
        let trained = model.forward(&examples)?.to_vec2::<f32>()?[0][0];
        let inference = model.forward_inference(&examples)?;
        assert!((trained - inference.to_vec2::<f32>()?[0][0]).abs() < 1e-6);
        let gradients = inference.sum_all()?.backward()?;
        assert!(model.vars().iter().all(|var| gradients.get(var).is_none()));
        let snapshot = model.snapshot(&Device::Cpu)?;
        model.stacks[7]
            .fc2_bias
            .set(&Tensor::full(0.7f32, 1, &Device::Cpu)?)?;
        assert!(
            (snapshot.forward_inference(&examples)?.to_vec2::<f32>()?[0][0] - trained).abs() < 1e-6
        );
        assert!(
            (model.forward_inference(&examples)?.to_vec2::<f32>()?[0][0] - trained).abs() > 0.6
        );
        Ok(())
    }

    #[test]
    fn floating_checkpoint_roundtrip() -> Result<()> {
        let model = PikafishModel::with_shape(
            PikafishShape {
                psq_features: 2,
                threat_features: 2,
            },
            &Device::Cpu,
        )?;
        let path = std::path::PathBuf::from("target/fast/pikafish_candle_roundtrip.safetensors");
        std::fs::create_dir_all(path.parent().unwrap())?;
        model.save(&path)?;
        model
            .transformer_bias
            .set(&Tensor::zeros(TRANSFORMER_WIDTH, DType::F32, &Device::Cpu)?)?;
        model.load(&path)?;
        assert_eq!(
            model.transformer_bias.as_tensor().to_vec1::<f32>()?[0],
            128.0
        );
        std::fs::remove_file(path)?;
        Ok(())
    }
}
