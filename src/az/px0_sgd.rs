use candle_core::{Result, Tensor, Var, backprop::GradStore};
use std::{collections::HashMap, path::Path};

// Px0 tfprocess.py 的 SGD(momentum=0.9, nesterov=True)。Candle SGD 无动量接口。
const MOMENTUM: f64 = 0.9;
const MAX_GRAD_NORM: f64 = 10_000.0;
const WARMUP_STEPS: usize = 250;

#[derive(Debug)]
pub(super) struct Px0Sgd {
    vars: Vec<Var>,
    velocity: Vec<Var>,
    pub steps: usize,
    pub last_lr: f64,
    pub base_lr: f64,
}

pub(super) fn learning_rate(base_lr: f64, steps: usize) -> f64 {
    // 按累计优化器步数衰减；保存周期不重启学习率或warmup。
    let factor = match steps {
        0..100_000 => 1.0,
        100_000..130_000 => 0.1,
        _ => 0.025,
    };
    let warmup = ((steps + 1) as f64 / WARMUP_STEPS as f64).min(1.0);
    base_lr * factor * warmup
}

impl Px0Sgd {
    pub fn new(vars: Vec<Var>, base_lr: f64) -> Result<Self> {
        let velocity = vars
            .iter()
            .map(|v| Var::zeros(v.shape(), v.dtype(), v.device()))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            velocity,
            steps: 0,
            last_lr: 0.0,
            base_lr,
        })
    }

    pub fn step(&mut self, grads: &GradStore) -> Result<()> {
        let mut norm_sq = None;
        for var in &self.vars {
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
            candle_core::bail!("SGD gradient norm is not finite");
        }
        let scale = if norm > MAX_GRAD_NORM {
            MAX_GRAD_NORM / norm
        } else {
            1.0
        };
        let lr = learning_rate(self.base_lr, self.steps);
        for (var, velocity) in self.vars.iter().zip(&self.velocity) {
            if let Some(g) = grads.get(var) {
                let scaled_grad = (g * (scale * lr))?;
                // Keras 保存的是含学习率的 velocity，变学习率时不能改用裸梯度动量。
                let next_velocity = ((velocity.as_tensor() * MOMENTUM)? - &scaled_grad)?;
                let next_var = ((var.as_tensor() + (&next_velocity * MOMENTUM)?)? - scaled_grad)?;
                velocity.set(&next_velocity)?;
                var.set(&next_var)?;
            }
        }
        self.steps += 1;
        self.last_lr = lr;
        Ok(())
    }

    pub fn save(&self, path: &Path, next_update: usize) -> Result<()> {
        let mut tensors = HashMap::new();
        tensors.insert(
            "state".to_owned(),
            Tensor::new(
                &[
                    1i64,
                    crate::infra::version::MODEL_FORMAT_VERSION as i64,
                    self.steps as i64,
                    next_update as i64,
                ],
                &candle_core::Device::Cpu,
            )?,
        );
        tensors.insert(
            "base_lr".to_owned(),
            Tensor::new(&[self.base_lr, self.last_lr], &candle_core::Device::Cpu)?,
        );
        for (i, (var, velocity)) in self.vars.iter().zip(&self.velocity).enumerate() {
            tensors.insert(format!("weight_{i}"), var.as_tensor().clone());
            tensors.insert(format!("velocity_{i}"), velocity.as_tensor().clone());
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
                .ok_or_else(|| candle_core::Error::Msg(format!("missing SGD tensor {name}")))
        };
        let state = get("state")?.to_vec1::<i64>()?;
        let rates = get("base_lr")?.to_vec1::<f64>()?;
        if state.len() != 4
            || state[0] != 1
            || state[1] != crate::infra::version::MODEL_FORMAT_VERSION as i64
            || state[2] < 0
            || state[3] != next_update as i64
            || rates.len() != 2
            || rates[0] != self.base_lr
            || tensors.len() != 2 + self.vars.len() * 2
        {
            candle_core::bail!(
                "SGD state does not match resume metadata: format={:?}/{} next_update={:?}/{next_update} base_lr={:?}/{} tensors={}/{}",
                state.get(1),
                crate::infra::version::MODEL_FORMAT_VERSION,
                state.get(3),
                rates.first(),
                self.base_lr,
                tensors.len(),
                2 + self.vars.len() * 2
            );
        }
        // 先完整验证；不允许旧 checkpoint 的动量误配到不同权重。
        for (i, var) in self.vars.iter().enumerate() {
            let saved = get(&format!("weight_{i}"))?;
            let velocity = get(&format!("velocity_{i}"))?;
            if saved.shape() != var.shape()
                || velocity.shape() != var.shape()
                || saved.dtype() != var.dtype()
                || velocity.dtype() != var.dtype()
                || saved
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .map(|x| x.to_bits())
                    .ne(var
                        .flatten_all()?
                        .to_vec1::<f32>()?
                        .iter()
                        .map(|x| x.to_bits()))
            {
                candle_core::bail!("SGD state weight mismatch at tensor {i}");
            }
        }
        for (i, velocity) in self.velocity.iter().enumerate() {
            velocity.set(&get(&format!("velocity_{i}"))?.to_device(velocity.device())?)?;
        }
        self.steps = state[2] as usize;
        self.last_lr = rates[1];
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn step(opt: &mut Px0Sgd, grad: &[f32]) {
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
    fn nesterov_matches_keras_variable_learning_rate_and_clipping() {
        let var = Var::new(&[1f32, -2.], &Device::Cpu).unwrap();
        let mut opt = Px0Sgd::new(vec![var.clone()], 0.02).unwrap();
        let mut expected = [1f64, -2.];
        let mut velocity = [0f64; 2];
        for (n, grad) in [
            (0, [3f32, -4.]),
            (1, [30_000., 40_000.]),
            (100_000, [2., -1.]),
            (130_000, [-1., 2.]),
        ] {
            opt.steps = n;
            let lr = learning_rate(0.02, n);
            let norm = (grad[0] as f64).hypot(grad[1] as f64);
            let scale = (MAX_GRAD_NORM / norm).min(1.0);
            for i in 0..2 {
                velocity[i] = 0.9 * velocity[i] - lr * scale * grad[i] as f64;
                expected[i] += 0.9 * velocity[i] - lr * scale * grad[i] as f64;
            }
            step(&mut opt, &grad);
            for (actual, expected) in var.to_vec1::<f32>().unwrap().iter().zip(expected) {
                assert!((*actual as f64 - expected).abs() < 1e-4);
            }
        }
        assert!((learning_rate(0.02, 249) - 0.02).abs() < 1e-12);
        for steps in [130_000, 139_999, 140_000, 150_000, 280_000, 1_000_000] {
            assert!((learning_rate(0.02, steps) - 0.0005).abs() < 1e-12);
        }
        assert!((learning_rate(0.02, 99_999) - 0.02).abs() < 1e-12);
        assert!((learning_rate(0.02, 100_000) - 0.002).abs() < 1e-12);
        assert!((learning_rate(0.02, 129_999) - 0.002).abs() < 1e-12);
    }

    #[test]
    fn restored_training_matches_uninterrupted_and_rejects_wrong_weights() {
        let var = Var::new(&[1f32, -2.], &Device::Cpu).unwrap();
        let mut opt = Px0Sgd::new(vec![var.clone()], 0.02).unwrap();
        step(&mut opt, &[3., -4.]);
        let dir = std::env::current_dir().unwrap().join("tmp");
        std::fs::create_dir_all(&dir).unwrap();
        // 旧版cycle 2检查点可正常恢复，下一步采用修正后的低学习率。
        opt.steps = 150_000;
        opt.last_lr = 0.02;
        let path = dir.join(format!(
            "chineseai-sgd-test-{}.safetensors",
            std::process::id()
        ));
        opt.save(&path, 42).unwrap();
        opt.save(&path, 42).unwrap();
        let restored_var =
            Var::new(var.to_vec1::<f32>().unwrap().as_slice(), &Device::Cpu).unwrap();
        let mut restored = Px0Sgd::new(vec![restored_var.clone()], 0.02).unwrap();
        assert!(restored.restore(&path, 43).is_err());
        restored.restore(&path, 42).unwrap();
        assert_eq!(restored.steps, 150_000);
        step(&mut opt, &[2., 5.]);
        step(&mut restored, &[2., 5.]);
        assert!((restored.last_lr - 0.0005).abs() < 1e-12);
        assert_eq!(restored.steps, 150_001);
        assert_eq!(
            var.to_vec1::<f32>().unwrap(),
            restored_var.to_vec1::<f32>().unwrap()
        );
        assert_eq!(opt.steps, restored.steps);
        assert!(restored.restore(&path, 42).is_err());
        std::fs::remove_file(path).unwrap();
    }
}
