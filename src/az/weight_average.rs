//! 发布最近五个晚期检查点的主干/价值头平均，学习器始终继续训练原始权重。
use super::AzNnue;
use candle_core::{Device, Result, Tensor};
use std::collections::{HashMap, VecDeque};

const START: usize = 1024;
const INTERVAL: usize = 256;
const WINDOW: usize = 5;

pub(super) fn includes(name: &str) -> bool {
    !name.starts_with("policy_") && !name.starts_with("moves_left_")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_average_preserves_latest_heads_and_resumes_exactly() {
        let current = AzNnue::random(8, 72);
        let mut average = WeightAverage::new(&current);
        for (i, step) in [1024, 1280, 1536, 1792, 2048].into_iter().enumerate() {
            average
                .capture(step, |model| {
                    model.apply_averaged_weights(&vec![
                        (i + 1) as f32 * 0.001;
                        model.averaged_weight_count()
                    ]);
                    Ok(())
                })
                .unwrap();
            if i < 4 {
                assert_eq!(average.model(&current).input_hidden, current.input_hidden);
            }
        }
        let published = average.model(&current);
        assert!(published.averaged_weights().iter().all(|&x| x == 0.003));
        assert_eq!(
            published.policy_context_hidden,
            current.policy_context_hidden
        );
        assert_eq!(
            published.policy_repetition_bias,
            current.policy_repetition_bias
        );
        assert_eq!(published.moves_left_output, current.moves_left_output);
        let state = average.state().unwrap();
        let mut restored = WeightAverage::new(&current);
        assert_eq!(restored.restore(&state, 2048).unwrap(), 6);
        assert_eq!(
            restored.model(&current).averaged_weights(),
            published.averaged_weights()
        );
        assert!(restored.restore(&state, 1792).is_err());
        for average in [&mut average, &mut restored] {
            average
                .capture(2304, |model| {
                    model.apply_averaged_weights(&vec![0.006; model.averaged_weight_count()]);
                    Ok(())
                })
                .unwrap();
        }
        assert_eq!(
            average.model(&current).averaged_weights(),
            restored.model(&current).averaged_weights()
        );
        assert!(
            average
                .model(&current)
                .averaged_weights()
                .iter()
                .all(|&x| (x - 0.004).abs() < 1e-9)
        );
        let mut corrupt = average.state().unwrap();
        corrupt.insert(
            "weight_average_0".into(),
            Tensor::new(&[f32::NAN], &Device::Cpu).unwrap(),
        );
        assert!(restored.restore(&corrupt, 2304).is_err());
    }
}

#[derive(Debug)]
pub(super) struct WeightAverage {
    template: AzNnue,
    snapshots: VecDeque<(usize, Vec<f32>)>,
    mean: Vec<f32>,
}

impl WeightAverage {
    pub fn new(model: &AzNnue) -> Self {
        Self {
            template: model.clone(),
            snapshots: VecDeque::new(),
            mean: Vec::new(),
        }
    }

    pub fn capture(
        &mut self,
        step: usize,
        copy: impl FnOnce(&mut AzNnue) -> Result<()>,
    ) -> Result<()> {
        if step >= START && step.is_multiple_of(INTERVAL) {
            copy(&mut self.template)?;
            self.snapshots
                .push_back((step, self.template.averaged_weights()));
            if self.snapshots.len() > WINDOW {
                self.snapshots.pop_front();
            }
            self.rebuild_mean();
        }
        Ok(())
    }

    fn rebuild_mean(&mut self) {
        self.mean.clear();
        if self.snapshots.len() != WINDOW {
            return;
        }
        let mut sum = vec![0f64; self.template.averaged_weight_count()];
        for (_, weights) in &self.snapshots {
            for (total, &value) in sum.iter_mut().zip(weights) {
                *total += value as f64;
            }
        }
        self.mean
            .extend(sum.into_iter().map(|v| (v / WINDOW as f64) as f32));
    }

    pub fn model(&self, current: &AzNnue) -> AzNnue {
        let mut model = current.clone();
        if !self.mean.is_empty() {
            model.apply_averaged_weights(&self.mean);
        }
        model
    }

    pub fn state(&self) -> Result<HashMap<String, Tensor>> {
        let mut tensors = HashMap::new();
        let mut meta = vec![1i64, START as i64, INTERVAL as i64, WINDOW as i64];
        meta.extend(self.snapshots.iter().map(|(step, _)| *step as i64));
        tensors.insert(
            "weight_average_meta".into(),
            Tensor::new(meta, &Device::Cpu)?,
        );
        for (i, (_, weights)) in self.snapshots.iter().enumerate() {
            tensors.insert(
                format!("weight_average_{i}"),
                Tensor::new(weights.as_slice(), &Device::Cpu)?,
            );
        }
        Ok(tensors)
    }

    pub fn restore(&mut self, tensors: &HashMap<String, Tensor>, step: usize) -> Result<usize> {
        let get = |name: &str| {
            tensors
                .get(name)
                .ok_or_else(|| candle_core::Error::Msg(format!("missing average tensor {name}")))
        };
        let last = step / INTERVAL * INTERVAL;
        let mut expected = vec![1i64, START as i64, INTERVAL as i64, WINDOW as i64];
        if last >= START {
            expected.extend(
                (last.saturating_sub((WINDOW - 1) * INTERVAL).max(START)..=last)
                    .step_by(INTERVAL)
                    .map(|s| s as i64),
            );
        }
        if get("weight_average_meta")?.to_vec1::<i64>()? != expected {
            candle_core::bail!("weight average history does not match optimizer step {step}");
        }
        let mut snapshots = VecDeque::new();
        for (i, &step) in expected[4..].iter().enumerate() {
            let weights = get(&format!("weight_average_{i}"))?.to_vec1::<f32>()?;
            if weights.len() != self.template.averaged_weight_count()
                || weights.iter().any(|v| !v.is_finite())
            {
                candle_core::bail!("invalid weight average snapshot {i}");
            }
            snapshots.push_back((step as usize, weights));
        }
        self.snapshots = snapshots;
        self.rebuild_mean();
        Ok(1 + self.snapshots.len())
    }
}
