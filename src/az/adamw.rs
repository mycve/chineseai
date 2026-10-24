use candle_core::{Result, Tensor, Var, backprop::GradStore};
use std::{collections::HashMap, path::Path};

// 手写 AdamW，而不是直接用 `candle_nn::optim::AdamW`，原因有三条（都在 candle-nn 0.10.2
// 的 `optim.rs` 里可以核对）：
//   1. 它的 `AdamW` 没有 `save`/`load`，优化器状态无法持久化；
//   2. 它的 `step_t` 是私有字段且没有 getter，resume 后偏差校正会从头算 —— 恢复后的头
//      几十步 `scale_m/sqrt(scale_v)` 偏小（t=1 时约 0.32×，之后渐近到 1×），即欠更新；
//   3. 它不支持按张量分组决定 weight decay。
// 自持累计步数，持久化 m/v、调度参数及 decay 分组。
const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;
const EPS: f64 = 1e-8;
/// 全局梯度范数上限。
const MAX_GRAD_NORM: f64 = 10_000.0;
/// 施加 decay 的稠密权重用的系数。
///
/// 注意量级：`lr=4e-4` 时每步衰减因子是 `1 - lr*wd = 1 - 4e-8`，十万步累计只收缩
/// **0.4%**。所以这个旋钮在 Adam 的学习率尺度下几乎不动结果 —— 分组与否不会混淆消融。
pub(super) const DENSE_WEIGHT_DECAY: f64 = 1e-4;
/// 调度参数成为恢复契约的一部分；拒绝旧的固定学习率状态。
// 共享参数、独立动量和各张量更新倍率一起构成恢复契约。
pub(super) const STATE_TAG: i64 = 7;
/// `state` 张量的元素个数：tag / 模型格式版本 / 步数 / 下次更新的序号。
const STATE_LEN: usize = 4;
/// `hyper` 张量还保存预热步数、衰减时间尺度和最低倍率。
const HYPER_LEN: usize = 9;
const WARMUP_STEPS: usize = 250;
const DECAY_STEPS: usize = 10_000;
const MIN_LR_RATIO: f64 = 0.1;

#[cfg(test)]
#[test]
fn rejects_optimizer_state_without_update_groups() {
    let device = candle_core::Device::Cpu;
    let variable = Var::new(&[1f32], &device).unwrap();
    let mut optimizer = AzAdamW::new(vec![variable], vec![false], vec![1.; 1], 0.001).unwrap();
    let path = std::env::temp_dir().join(format!(
        "optimizer-sharing-version-{}.safetensors",
        std::process::id()
    ));
    optimizer.save(&path, 1).unwrap();
    assert_eq!(super::AzNnue::training_state_next_update(&path).unwrap(), 1);
    let mut tensors = candle_core::safetensors::load(&path, &device).unwrap();
    let mut state = tensors["state"].to_vec1::<i64>().unwrap();
    for old_tag in [4, 5, 6] {
        state[0] = old_tag;
        tensors.insert(
            "state".to_owned(),
            Tensor::new(state.as_slice(), &device).unwrap(),
        );
        candle_core::safetensors::save(&tensors, &path).unwrap();
        assert!(optimizer.restore(&path, 1).is_err());
        assert!(super::AzNnue::training_state_next_update(&path).is_err());
    }
    std::fs::remove_file(path).unwrap();
}

/// steps 为已经完成的更新次数。持续自博弈无需设定总训练步数。
fn learning_rate(base_lr: f64, steps: usize) -> f64 {
    let update = steps.saturating_add(1);
    let ratio = if update <= WARMUP_STEPS {
        update as f64 / WARMUP_STEPS as f64
    } else {
        (1.0 + (update - WARMUP_STEPS) as f64 / DECAY_STEPS as f64)
            .sqrt()
            .recip()
            .max(MIN_LR_RATIO)
    };
    base_lr * ratio
}

#[derive(Debug)]
pub(super) struct AzAdamW {
    vars: Vec<Var>,
    first_moment: Vec<Var>,
    second_moment: Vec<Var>,
    /// 与 `vars` 同序：该张量是否施加 weight decay。
    decay: Vec<bool>,
    /// 在 Adam 归一化后缩放更新量；不改变 decay 系数或梯度动量。
    update_scale: Vec<f64>,
    pub steps: usize,
    pub last_lr: f64,
    pub base_lr: f64,
}

impl AzAdamW {
    pub fn new(
        vars: Vec<Var>,
        decay: Vec<bool>,
        update_scale: Vec<f64>,
        base_lr: f64,
    ) -> Result<Self> {
        if !base_lr.is_finite() || base_lr <= 0.0 {
            candle_core::bail!("AdamW peak learning rate must be finite and positive");
        }
        if vars.len() != decay.len() {
            candle_core::bail!(
                "AdamW decay mask length {} does not match {} vars",
                decay.len(),
                vars.len()
            );
        }
        if update_scale.len() != vars.len()
            || update_scale.iter().any(|s| !s.is_finite() || *s <= 0.)
        {
            candle_core::bail!("AdamW update scales must match vars and be finite and positive");
        }
        let first_moment = zeros_like(&vars)?;
        let second_moment = zeros_like(&vars)?;
        Ok(Self {
            vars,
            first_moment,
            second_moment,
            decay,
            update_scale,
            steps: 0,
            last_lr: 0.0,
            base_lr,
        })
    }

    pub fn step(&mut self, grads: &GradStore) -> Result<()> {
        let scale = gradient_scale(&self.vars, grads)?;
        let lr = learning_rate(self.base_lr, self.steps);
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
            // 缩放须放在 Adam 归一化之后；预先缩放梯度会被一、二阶矩基本抵消。
            let update = ((m_hat / (v_hat.sqrt()? + EPS)?)? * (lr * self.update_scale[index]))?;
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

    #[cfg(test)]
    pub fn save(&self, path: &Path, next_update: usize) -> Result<()> {
        self.save_with_extra(path, next_update, HashMap::new())
    }

    pub fn save_with_extra(
        &self,
        path: &Path,
        next_update: usize,
        mut tensors: HashMap<String, Tensor>,
    ) -> Result<()> {
        let cpu = candle_core::Device::Cpu;
        tensors.insert(
            "feature_schema".to_owned(),
            Tensor::new(super::features::feature_schema().as_slice(), &cpu)?,
        );
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
                    WARMUP_STEPS as f64,
                    DECAY_STEPS as f64,
                    MIN_LR_RATIO,
                ],
                &cpu,
            )?,
        );
        let mask: Vec<i64> = self.decay.iter().map(|&flag| i64::from(flag)).collect();
        tensors.insert("decay_mask".to_owned(), Tensor::new(mask, &cpu)?);
        tensors.insert(
            "update_scale".to_owned(),
            Tensor::new(self.update_scale.as_slice(), &cpu)?,
        );
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

    #[cfg(test)]
    pub fn restore(&mut self, path: &Path, next_update: usize) -> Result<()> {
        self.restore_projected(
            path,
            next_update,
            |weights| Ok(weights.to_vec()),
            |_, _| Ok(0),
        )
    }

    /// 先严格校验推理权重，再恢复原始训练参数及动量。
    pub fn restore_projected(
        &mut self,
        path: &Path,
        next_update: usize,
        project: impl Fn(&[Tensor]) -> Result<Vec<Tensor>>,
        restore_extra: impl FnOnce(&HashMap<String, Tensor>, usize) -> Result<usize>,
    ) -> Result<()> {
        let tensors = candle_core::safetensors::load(path, &candle_core::Device::Cpu)?;
        let get = |name: &str| {
            tensors
                .get(name)
                .ok_or_else(|| candle_core::Error::Msg(format!("missing AdamW tensor {name}")))
        };
        let state = get("state")?.to_vec1::<i64>()?;
        if get("feature_schema")?.to_vec1::<f32>()? != super::features::feature_schema() {
            candle_core::bail!("AdamW feature schema mismatch");
        }
        let hyper = get("hyper")?.to_vec1::<f64>()?;
        let mask = get("decay_mask")?.to_vec1::<i64>()?;
        let update_scale = get("update_scale")?.to_vec1::<f64>()?;
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
            || hyper[6] != WARMUP_STEPS as f64
            || hyper[7] != DECAY_STEPS as f64
            || hyper[8] != MIN_LR_RATIO
            || !hyper[1].is_finite()
            || update_scale != self.update_scale
            || mask != expected_mask
        {
            candle_core::bail!(
                "AdamW state does not match resume metadata: format={:?}/{} next_update={:?}/{next_update} base_lr={:?}/{} decay_mask_match={} update_scale_match={} tensors={}/{}",
                state.get(1),
                crate::infra::version::MODEL_FORMAT_VERSION,
                state.get(3),
                hyper.first(),
                self.base_lr,
                mask == expected_mask,
                update_scale == self.update_scale,
                tensors.len(),
                5 + self.vars.len() * 3
            );
        }
        let extra = restore_extra(&tensors, state[2] as usize)?;
        if tensors.len() != 5 + self.vars.len() * 3 + extra {
            candle_core::bail!("unexpected AdamW checkpoint tensors");
        }
        // 先完整验证；不允许旧 checkpoint 的动量误配到不同权重。
        for (i, var) in self.vars.iter().enumerate() {
            for name in [
                format!("weight_{i}"),
                format!("first_moment_{i}"),
                format!("second_moment_{i}"),
            ] {
                let saved = get(&name)?;
                if saved.shape() != var.shape() || saved.dtype() != var.dtype() {
                    candle_core::bail!("AdamW state mismatch at `{name}`");
                }
            }
        }
        let saved_weights = (0..self.vars.len())
            .map(|i| get(&format!("weight_{i}")).cloned())
            .collect::<Result<Vec<_>>>()?;
        let current_weights = self
            .vars
            .iter()
            .map(|var| var.to_device(&candle_core::Device::Cpu))
            .collect::<Result<Vec<_>>>()?;
        let saved_inference = project(&saved_weights)?;
        let current_inference = project(&current_weights)?;
        if saved_inference.len() != self.vars.len() || current_inference.len() != self.vars.len() {
            candle_core::bail!("invalid optimizer weight projection");
        }
        for (i, (saved, current)) in saved_inference.iter().zip(&current_inference).enumerate() {
            if saved.shape() != current.shape()
                || saved.dtype() != current.dtype()
                || saved
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .map(|x| x.to_bits())
                    .ne(current
                        .flatten_all()?
                        .to_vec1::<f32>()?
                        .iter()
                        .map(|x| x.to_bits()))
            {
                candle_core::bail!("AdamW state mismatch at `weight_{i}`");
            }
        }
        for i in 0..self.vars.len() {
            self.vars[i].set(&saved_weights[i].to_device(self.vars[i].device())?)?;
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

    #[test]
    fn schedule_warms_up_decays_smoothly_and_keeps_a_floor() {
        let peak = 4e-4;
        assert_eq!(learning_rate(peak, 0), peak / 250.0);
        assert_eq!(learning_rate(peak, 249), peak);
        assert!(learning_rate(peak, 250) < peak);
        assert!((learning_rate(peak, 10_249) - peak / 2f64.sqrt()).abs() < 1e-15);
        assert_eq!(learning_rate(peak, 990_249), peak * MIN_LR_RATIO);
        assert_eq!(learning_rate(peak, usize::MAX), peak * MIN_LR_RATIO);
        for steps in 1..250 {
            assert!(learning_rate(peak, steps) > learning_rate(peak, steps - 1));
        }
        for steps in 250..20_000 {
            assert!(learning_rate(peak, steps) <= learning_rate(peak, steps - 1));
        }
    }

    #[test]
    fn resume_preserves_schedule_at_warmup_decay_and_floor_boundaries() {
        let path = std::env::temp_dir().join(format!(
            "chineseai-adamw-schedule-{}.safetensors",
            std::process::id()
        ));
        for steps in [249, 10_249, 990_249] {
            let var = Var::new(&[1f32], &Device::Cpu).unwrap();
            let mut opt = AzAdamW::new(vec![var.clone()], vec![true], vec![1.; 1], 4e-4).unwrap();
            step(&mut opt, &[0.5]);
            opt.steps = steps;
            opt.last_lr = learning_rate(opt.base_lr, steps - 1);
            opt.save(&path, 42).unwrap();
            let saved_var =
                Var::new(var.to_vec1::<f32>().unwrap().as_slice(), &Device::Cpu).unwrap();
            let mut restored =
                AzAdamW::new(vec![saved_var.clone()], vec![true], vec![1.; 1], 4e-4).unwrap();
            restored.restore(&path, 42).unwrap();
            for _ in 0..3 {
                step(&mut opt, &[0.25]);
                step(&mut restored, &[0.25]);
                assert_eq!(restored.steps, opt.steps);
                assert_eq!(restored.last_lr, learning_rate(opt.base_lr, opt.steps - 1));
                assert_eq!(
                    var.to_vec1::<f32>().unwrap(),
                    saved_var.to_vec1::<f32>().unwrap()
                );
            }
        }
        // 调度不同或旧格式的状态必须拒绝，不能悄悄改变恢复后的训练轨迹。
        let tensors = candle_core::safetensors::load(&path, &Device::Cpu).unwrap();
        for legacy in [false, true] {
            let mut altered = tensors.clone();
            if legacy {
                let mut state = altered["state"].to_vec1::<i64>().unwrap();
                state[0] = 2;
                altered.insert("state".into(), Tensor::new(state, &Device::Cpu).unwrap());
            } else {
                let mut hyper = altered["hyper"].to_vec1::<f64>().unwrap();
                hyper[6] += 1.0;
                altered.insert("hyper".into(), Tensor::new(hyper, &Device::Cpu).unwrap());
            }
            candle_core::safetensors::save(&altered, &path).unwrap();
            let var = Var::new(
                tensors["weight_0"].to_vec1::<f32>().unwrap().as_slice(),
                &Device::Cpu,
            )
            .unwrap();
            let mut opt = AzAdamW::new(vec![var], vec![true], vec![1.; 1], 4e-4).unwrap();
            assert!(opt.restore(&path, 42).is_err());
        }
        std::fs::remove_file(path).unwrap();
    }

    /// 逐步对照一个纯 f64 的参考实现，覆盖 bias correction 与 decay 分组。
    #[test]
    fn adamw_matches_reference_including_decay_group() {
        let decayed = Var::new(&[1f32, -2.], &Device::Cpu).unwrap();
        let plain = Var::new(&[3f32, 0.5], &Device::Cpu).unwrap();
        let lr = 4e-4;
        let mut opt = AzAdamW::new(
            vec![decayed.clone(), plain.clone()],
            vec![true, false],
            vec![1.; 2],
            lr,
        )
        .unwrap();
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
                let step_lr = learning_rate(lr, opt.steps - 1);
                let update = (m[i] * scale_m) / ((v[i] * scale_v).sqrt() + EPS) * step_lr;
                // 前两个下标走 decay 组，后两个不走。
                reference[i] = if i < 2 {
                    reference[i] * (1.0 - step_lr * DENSE_WEIGHT_DECAY) - update
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
        assert_eq!(opt.last_lr, learning_rate(lr, opt.steps - 1));
    }

    #[test]
    fn decay_group_can_be_disabled_entirely_for_lookup_tables() {
        let table = Var::new(&[0.5f32], &Device::Cpu).unwrap();
        let mut opt = AzAdamW::new(vec![table.clone()], vec![false], vec![1.; 1], 4e-4).unwrap();
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
    fn update_scales_apply_after_adam_normalization() {
        let normal = Var::new(&[0f32], &Device::Cpu).unwrap();
        let slow = Var::new(&[0f32], &Device::Cpu).unwrap();
        let mut opt = AzAdamW::new(
            vec![normal.clone(), slow.clone()],
            vec![false, false],
            vec![1., 0.1],
            4e-4,
        )
        .unwrap();
        let grads = (normal.as_tensor() + slow.as_tensor())
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        opt.step(&grads).unwrap();
        let normal_update = normal.to_vec1::<f32>().unwrap()[0];
        let slow_update = slow.to_vec1::<f32>().unwrap()[0];
        assert!((slow_update / normal_update - 0.1).abs() < 1e-6);
        assert_eq!(
            opt.first_moment[0].to_vec1::<f32>().unwrap(),
            opt.first_moment[1].to_vec1::<f32>().unwrap()
        );
        assert_eq!(
            opt.second_moment[0].to_vec1::<f32>().unwrap(),
            opt.second_moment[1].to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn restored_training_matches_uninterrupted_and_rejects_mismatches() {
        let var = Var::new(&[1f32, -2.], &Device::Cpu).unwrap();
        let mut opt = AzAdamW::new(vec![var.clone()], vec![true], vec![0.1], 4e-4).unwrap();
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
        let mut restored =
            AzAdamW::new(vec![restored_var.clone()], vec![true], vec![0.1], 4e-4).unwrap();
        for attempts in [43usize, 41] {
            assert!(restored.restore(&path, attempts).is_err());
        }
        // 换一个 decay 分组的优化器不能载入这份状态。
        let mut wrong_group =
            AzAdamW::new(vec![restored_var.clone()], vec![false], vec![1.; 1], 4e-4).unwrap();
        assert!(wrong_group.restore(&path, 42).is_err());
        let mut wrong_scale =
            AzAdamW::new(vec![restored_var.clone()], vec![true], vec![1.], 4e-4).unwrap();
        assert!(wrong_scale.restore(&path, 42).is_err());
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
