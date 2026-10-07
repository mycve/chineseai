use candle_core::{Result, Tensor, Var, backprop::GradStore};
use std::{collections::HashMap, path::Path};

// 手写 AdamW，而不是直接用 `candle_nn::optim::AdamW`，原因有三条（都在 candle-nn 0.10.2
// 的 `optim.rs` 里可以核对）：
//   1. 它的 `AdamW` 没有 `save`/`load`，优化器状态无法持久化；
//   2. 它的 `step_t` 是私有字段且没有 getter，resume 后偏差校正会从头算 —— 恢复后的头
//      几十步 `scale_m/sqrt(scale_v)` 偏小（t=1 时约 0.32×，之后渐近到 1×），即欠更新；
//   3. 它不支持按张量分组决定 weight decay。
// 本实现照 `px0_sgd.rs` 的模板：自持步数计数器、m/v 用同一套 safetensors 布局序列化、
// decay 按张量分组。
const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;
const EPS: f64 = 1e-8;
/// 与 Px0Sgd 用同一个全局梯度范数上限，保证两臂的裁剪口径一致。
/// 实测这个阈值几乎不会触发（正常范数是 O(1)），所以它实际是个 no-op。
const MAX_GRAD_NORM: f64 = 10_000.0;
/// 施加 decay 的稠密权重用的系数。
///
/// 注意量级：`lr=4e-4` 时每步衰减因子是 `1 - lr*wd = 1 - 4e-8`，十万步累计只收缩
/// **0.4%**。所以这个旋钮在 Adam 的学习率尺度下几乎不动结果 —— 分组与否不会混淆消融。
pub(super) const DENSE_WEIGHT_DECAY: f64 = 1e-4;
/// 优化器状态文件的格式标签。与 Px0Sgd 的 `1` 区分开，禁止两种优化器的状态互载。
const STATE_TAG: i64 = 2;
/// `state` 张量的元素个数：tag / 模型格式版本 / 步数 / 下次更新的序号。
const STATE_LEN: usize = 4;
/// `hyper` 张量：base_lr / last_lr / beta1 / beta2 / eps / dense_weight_decay。
const HYPER_LEN: usize = 6;

#[derive(Debug)]
pub(super) struct AzAdamW {
    vars: Vec<Var>,
    first_moment: Vec<Var>,
    second_moment: Vec<Var>,
    /// 与 `vars` 同序：该张量是否施加 weight decay。
    decay: Vec<bool>,
    pub steps: usize,
    pub last_lr: f64,
    pub base_lr: f64,
}

impl AzAdamW {
    pub fn new(vars: Vec<Var>, decay: Vec<bool>, base_lr: f64) -> Result<Self> {
        if vars.len() != decay.len() {
            candle_core::bail!(
                "AdamW decay mask length {} does not match {} vars",
                decay.len(),
                vars.len()
            );
        }
        let first_moment = zeros_like(&vars)?;
        let second_moment = zeros_like(&vars)?;
        Ok(Self {
            vars,
            first_moment,
            second_moment,
            decay,
            steps: 0,
            last_lr: 0.0,
            base_lr,
        })
    }

    pub fn step(&mut self, grads: &GradStore) -> Result<()> {
        let scale = gradient_scale(&self.vars, grads)?;
        // 常数学习率：没有 warmup，也没有按步数分段的阶梯。
        let lr = self.base_lr;
        self.steps += 1;
        let step = self.steps as i32;
        let scale_m = 1.0 / (1.0 - BETA1.powi(step));
        let scale_v = 1.0 / (1.0 - BETA2.powi(step));
        for (index, ((var, m), v)) in self
            .vars
            .iter()
            .zip(&self.first_moment)
            .zip(&self.second_moment)
            .enumerate()
        {
            let Some(grad) = grads.get(var) else {
                continue;
            };
            let grad = if scale == 1.0 {
                grad.clone()
            } else {
                (grad * scale)?
            };
            let next_m = ((m.as_tensor() * BETA1)? + (&grad * (1.0 - BETA1))?)?;
            let next_v = ((v.as_tensor() * BETA2)? + (grad.sqr()? * (1.0 - BETA2))?)?;
            let m_hat = (&next_m * scale_m)?;
            let v_hat = (&next_v * scale_v)?;
            let update = ((m_hat / (v_hat.sqrt()? + EPS)?)? * lr)?;
            // Decoupled weight decay：只在该张量被标记为需要 decay、且本步存在梯度时施加，
            // 与 candle 的实现口径一致。
            let next_var = if self.decay[index] {
                ((var.as_tensor() * (1.0 - lr * DENSE_WEIGHT_DECAY))? - update)?
            } else {
                (var.as_tensor() - update)?
            };
            m.set(&next_m)?;
            v.set(&next_v)?;
            var.set(&next_var)?;
        }
        self.last_lr = lr;
        Ok(())
    }

    pub fn save(&self, path: &Path, next_update: usize) -> Result<()> {
        let cpu = candle_core::Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "state".to_owned(),
            Tensor::new(
                &[
                    STATE_TAG,
                    crate::infra::version::MODEL_FORMAT_VERSION as i64,
                    self.steps as i64,
                    next_update as i64,
                ],
                &cpu,
            )?,
        );
        tensors.insert(
            "hyper".to_owned(),
            Tensor::new(
                &[
                    self.base_lr,
                    self.last_lr,
                    BETA1,
                    BETA2,
                    EPS,
                    DENSE_WEIGHT_DECAY,
                ],
                &cpu,
            )?,
        );
        let mask: Vec<i64> = self.decay.iter().map(|&flag| i64::from(flag)).collect();
        tensors.insert("decay_mask".to_owned(), Tensor::new(mask, &cpu)?);
        for (i, ((var, m), v)) in self
            .vars
            .iter()
            .zip(&self.first_moment)
            .zip(&self.second_moment)
            .enumerate()
        {
            tensors.insert(format!("weight_{i}"), var.as_tensor().clone());
            tensors.insert(format!("first_moment_{i}"), m.as_tensor().clone());
            tensors.insert(format!("second_moment_{i}"), v.as_tensor().clone());
        }
        let temp = path.with_extension("safetensors.tmp");
        candle_core::safetensors::save(&tensors, &temp)?;
        std::fs::rename(temp, path)?;
        Ok(())
    }

    pub fn restore(&mut self, path: &Path, next_update: usize) -> Result<()> {
        let tensors = candle_core::safetensors::load(path, &candle_core::Device::Cpu)?;
        let get = |name: &str| {
            tensors
                .get(name)
                .ok_or_else(|| candle_core::Error::Msg(format!("missing AdamW tensor {name}")))
        };
        let state = get("state")?.to_vec1::<i64>()?;
        let hyper = get("hyper")?.to_vec1::<f64>()?;
        let mask = get("decay_mask")?.to_vec1::<i64>()?;
        let expected_mask: Vec<i64> = self.decay.iter().map(|&flag| i64::from(flag)).collect();
        if state.len() != STATE_LEN
            || state[0] != STATE_TAG
            || state[1] != crate::infra::version::MODEL_FORMAT_VERSION as i64
            || state[2] < 0
            || state[3] != next_update as i64
            || hyper.len() != HYPER_LEN
            || hyper[0] != self.base_lr
            || hyper[2] != BETA1
            || hyper[3] != BETA2
            || hyper[4] != EPS
            || hyper[5] != DENSE_WEIGHT_DECAY
            || mask != expected_mask
            || tensors.len() != 3 + self.vars.len() * 3
        {
            candle_core::bail!(
                "AdamW state does not match resume metadata: format={:?}/{} next_update={:?}/{next_update} base_lr={:?}/{} decay_mask_match={} tensors={}/{}",
                state.get(1),
                crate::infra::version::MODEL_FORMAT_VERSION,
                state.get(3),
                hyper.first(),
                self.base_lr,
                mask == expected_mask,
                tensors.len(),
                3 + self.vars.len() * 3
            );
        }
        // 先完整验证；不允许旧 checkpoint 的动量误配到不同权重。
        for (i, var) in self.vars.iter().enumerate() {
            for name in [
                format!("weight_{i}"),
                format!("first_moment_{i}"),
                format!("second_moment_{i}"),
            ] {
                let saved = get(&name)?;
                if saved.shape() != var.shape()
                    || saved.dtype() != var.dtype()
                    || (name.starts_with("weight_")
                        && saved
                            .flatten_all()?
                            .to_vec1::<f32>()?
                            .iter()
                            .map(|x| x.to_bits())
                            .ne(var
                                .flatten_all()?
                                .to_vec1::<f32>()?
                                .iter()
                                .map(|x| x.to_bits())))
                {
                    candle_core::bail!("AdamW state mismatch at `{name}`");
                }
            }
        }
        for i in 0..self.vars.len() {
            self.first_moment[i]
                .set(&get(&format!("first_moment_{i}"))?.to_device(self.vars[i].device())?)?;
            self.second_moment[i]
                .set(&get(&format!("second_moment_{i}"))?.to_device(self.vars[i].device())?)?;
        }
        self.steps = state[2] as usize;
        self.last_lr = hyper[1];
        Ok(())
    }
}

fn zeros_like(vars: &[Var]) -> Result<Vec<Var>> {
    vars.iter()
        .map(|v| Var::zeros(v.shape(), v.dtype(), v.device()))
        .collect()
}

fn gradient_scale(vars: &[Var], grads: &GradStore) -> Result<f64> {
    let mut norm_sq = None;
    for var in vars {
        if let Some(g) = grads.get(var) {
            let sq = g.sqr()?.sum_all()?;
            norm_sq = Some(match norm_sq {
                None => sq,
                Some(sum) => (sum + sq)?,
            });
        }
    }
    let norm = match norm_sq {
        Some(sum) => sum.sqrt()?.to_scalar::<f32>()? as f64,
        None => 0.0,
    };
    if !norm.is_finite() {
        candle_core::bail!("AdamW gradient norm is not finite");
    }
    Ok(if norm > MAX_GRAD_NORM {
        MAX_GRAD_NORM / norm
    } else {
        1.0
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn step(opt: &mut AzAdamW, grad: &[f32]) {
        let grads = opt.vars[0]
            .broadcast_mul(&Tensor::new(grad, &Device::Cpu).unwrap())
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        opt.step(&grads).unwrap();
    }

    /// 逐步对照一个纯 f64 的参考实现，覆盖 bias correction 与 decay 分组。
    #[test]
    fn adamw_matches_reference_including_decay_group() {
        let decayed = Var::new(&[1f32, -2.], &Device::Cpu).unwrap();
        let plain = Var::new(&[3f32, 0.5], &Device::Cpu).unwrap();
        let lr = 4e-4;
        let mut opt =
            AzAdamW::new(vec![decayed.clone(), plain.clone()], vec![true, false], lr).unwrap();
        let mut reference = [1f64, -2., 3., 0.5];
        let mut m = [0f64; 4];
        let mut v = [0f64; 4];
        for grad in [[0.25f32, -0.5], [1.5, 2.0], [-0.75, 0.125]] {
            let grads = decayed
                .broadcast_mul(&Tensor::new(grad.as_slice(), &Device::Cpu).unwrap())
                .unwrap()
                .sum_all()
                .unwrap()
                .add(
                    &plain
                        .broadcast_mul(&Tensor::new(grad.as_slice(), &Device::Cpu).unwrap())
                        .unwrap()
                        .sum_all()
                        .unwrap(),
                )
                .unwrap()
                .backward()
                .unwrap();
            opt.step(&grads).unwrap();
            let t = opt.steps as i32;
            let scale_m = 1.0 / (1.0 - BETA1.powi(t));
            let scale_v = 1.0 / (1.0 - BETA2.powi(t));
            for i in 0..4 {
                let g = grad[i % 2] as f64;
                m[i] = BETA1 * m[i] + (1.0 - BETA1) * g;
                v[i] = BETA2 * v[i] + (1.0 - BETA2) * g * g;
                let update = (m[i] * scale_m) / ((v[i] * scale_v).sqrt() + EPS) * lr;
                // 前两个下标走 decay 组，后两个不走。
                reference[i] = if i < 2 {
                    reference[i] * (1.0 - lr * DENSE_WEIGHT_DECAY) - update
                } else {
                    reference[i] - update
                };
            }
            for (index, (var, expected)) in [(&decayed, 0usize), (&plain, 2usize)]
                .into_iter()
                .flat_map(|(var, base)| {
                    var.to_vec1::<f32>()
                        .unwrap()
                        .into_iter()
                        .enumerate()
                        .map(move |(offset, actual)| (actual, reference[base + offset]))
                })
                .enumerate()
            {
                assert!(
                    (var as f64 - expected).abs() < 1e-6,
                    "tensor {index} diverged: {var} vs {expected}"
                );
            }
        }
        // 常数学习率：last_lr 必须始终等于 base_lr。
        assert!((opt.last_lr - lr).abs() < 1e-12);
    }

    #[test]
    fn decay_group_can_be_disabled_entirely_for_lookup_tables() {
        let table = Var::new(&[0.5f32], &Device::Cpu).unwrap();
        let mut opt = AzAdamW::new(vec![table.clone()], vec![false], 4e-4).unwrap();
        // 梯度恒为 0 时，m/v 保持 0，更新量是 0/(0+eps)=0，权重必须一动不动 ——
        // 这正是"查表组 wd=0"想要的行为：没有 decay 就不会慢慢吃掉未激活的行。
        let grads = table
            .broadcast_mul(&Tensor::new(&[0f32], &Device::Cpu).unwrap())
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        for _ in 0..1000 {
            opt.step(&grads).unwrap();
        }
        assert_eq!(table.to_vec1::<f32>().unwrap(), vec![0.5]);
    }

    #[test]
    fn restored_training_matches_uninterrupted_and_rejects_mismatches() {
        let var = Var::new(&[1f32, -2.], &Device::Cpu).unwrap();
        let mut opt = AzAdamW::new(vec![var.clone()], vec![true], 4e-4).unwrap();
        step(&mut opt, &[3., -4.]);
        let dir = std::env::current_dir().unwrap().join("tmp");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!(
            "chineseai-adamw-test-{}.safetensors",
            std::process::id()
        ));
        opt.save(&path, 42).unwrap();
        let restored_var =
            Var::new(var.to_vec1::<f32>().unwrap().as_slice(), &Device::Cpu).unwrap();
        let mut restored = AzAdamW::new(vec![restored_var.clone()], vec![true], 4e-4).unwrap();
        for attempts in [43usize, 41] {
            assert!(restored.restore(&path, attempts).is_err());
        }
        // 换一个 decay 分组的优化器不能载入这份状态。
        let mut wrong_group = AzAdamW::new(vec![restored_var.clone()], vec![false], 4e-4).unwrap();
        assert!(wrong_group.restore(&path, 42).is_err());
        restored.restore(&path, 42).unwrap();
        assert_eq!(restored.steps, opt.steps);
        // 恢复后继续走两步，必须与不中断的优化器逐位一致（步数计数器是状态的一部分，
        // 所以 bias correction 不会像 candle 内置 AdamW 那样在 resume 后算错）。
        step(&mut opt, &[2., 5.]);
        step(&mut restored, &[2., 5.]);
        assert_eq!(opt.steps, restored.steps);
        assert_eq!(
            var.to_vec1::<f32>().unwrap(),
            restored_var.to_vec1::<f32>().unwrap()
        );
        std::fs::remove_file(path).unwrap();
    }
}
