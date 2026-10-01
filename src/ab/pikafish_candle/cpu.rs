//! 搜索专用只读权重与线程私有增量累加器；热路径不创建 Candle 张量或获取存储锁。
use super::*;
#[derive(Debug)]
pub struct PikafishCpuModel {
    identity: usize,
    shape: PikafishShape,
    bias: Vec<f32>,
    psq: Vec<f32>,
    threats: Vec<f32>,
    psqt: Vec<f32>,
    threat_psqt: Vec<f32>,
    stacks: Vec<Stack>,
}
#[derive(Debug)]
struct Stack {
    w0: Vec<f32>,
    b0: Vec<f32>,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: f32,
}
pub struct PikafishCpuCache {
    next: PikafishExample,
    previous: PikafishExample,
    acc: [[f32; TRANSFORMER_WIDTH]; 2],
    psqt: [[f32; PSQT_BUCKETS]; 2],
    evaluations: usize,
    model: usize,
    side: Option<crate::xiangqi::Color>,
}
impl Default for PikafishCpuCache {
    fn default() -> Self {
        let empty = || PikafishExample {
            psq: [Vec::new(), Vec::new()],
            threats: [Vec::new(), Vec::new()],
            psqt_bucket: 0,
            layer_stack: 0,
        };
        Self {
            next: empty(),
            previous: empty(),
            acc: [[0.; TRANSFORMER_WIDTH]; 2],
            psqt: [[0.; PSQT_BUCKETS]; 2],
            evaluations: 0,
            model: 0,
            side: None,
        }
    }
}
impl PikafishCpuModel {
    pub(super) fn from_model(model: &PikafishModel) -> Result<Self> {
        let copy = |v: &Var| {
            v.as_tensor()
                .detach()
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()
        };
        let stacks = model
            .stacks
            .iter()
            .map(|s| {
                Ok(Stack {
                    w0: copy(&s.fc0_weight)?,
                    b0: copy(&s.fc0_bias)?,
                    w1: copy(&s.fc1_weight)?,
                    b1: copy(&s.fc1_bias)?,
                    w2: copy(&s.fc2_weight)?,
                    b2: copy(&s.fc2_bias)?[0],
                })
            })
            .collect::<Result<_>>()?;
        static NEXT_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);
        Ok(Self {
            identity: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            shape: model.shape,
            bias: copy(&model.transformer_bias)?,
            psq: copy(&model.transformer_psq)?,
            threats: copy(&model.transformer_threat)?,
            psqt: copy(&model.psqt)?,
            threat_psqt: copy(&model.threat_psqt)?,
            stacks,
        })
    }
    pub fn load(path: &Path) -> Result<Self> {
        let model = PikafishModel::new(&Device::Cpu)?;
        model.load(path)?;
        model.cpu_snapshot()
    }
    pub fn evaluate(&self, position: &Position, cache: &mut PikafishCpuCache) -> Result<f32> {
        let side = position.side_to_move();
        if cache.side.is_some_and(|previous| previous != side) {
            cache.previous.psq.swap(0, 1);
            cache.previous.threats.swap(0, 1);
            cache.acc.swap(0, 1);
            cache.psqt.swap(0, 1);
        }
        cache.side = Some(side);
        for (i, perspective) in [side, side.opposite()].into_iter().enumerate() {
            crate::nnue::pikafish::fill_psq_features(position, perspective, &mut cache.next.psq[i])
                .ok_or_else(|| candle_core::Error::Msg("invalid PSQ features".into()))?;
        }
        let [red, black] = &mut cache.next.threats;
        crate::nnue::full_threats::fill_threat_features_both(position, red, black)
            .ok_or_else(|| candle_core::Error::Msg("invalid threat features".into()))?;
        if side == crate::xiangqi::Color::Black {
            cache.next.threats.swap(0, 1);
        }
        cache.next.layer_stack = crate::nnue::pikafish::layer_stack_bucket(position);
        cache.next.psqt_bucket = cache.next.layer_stack;
        self.finish(cache)
    }
    pub fn evaluate_example(
        &self,
        example: &PikafishExample,
        cache: &mut PikafishCpuCache,
    ) -> Result<f32> {
        cache.side = None;
        for i in 0..2 {
            cache.next.psq[i].clone_from(&example.psq[i]);
            cache.next.threats[i].clone_from(&example.threats[i]);
        }
        cache.next.layer_stack = example.layer_stack;
        cache.next.psqt_bucket = example.psqt_bucket;
        self.finish(cache)
    }
    fn finish(&self, cache: &mut PikafishCpuCache) -> Result<f32> {
        let identity = self.identity;
        if cache.model != identity {
            cache.evaluations = 0;
            cache.model = identity;
        }
        if cache.next.layer_stack >= LAYER_STACKS || cache.next.psqt_bucket >= PSQT_BUCKETS {
            candle_core::bail!("invalid Pikafish bucket")
        }
        for side in 0..2 {
            if cache.next.psq[side]
                .iter()
                .any(|&i| i >= self.shape.psq_features)
                || cache.next.threats[side]
                    .iter()
                    .any(|&i| i >= self.shape.threat_features)
            {
                candle_core::bail!("Pikafish feature out of range")
            }
        }
        for side in 0..2 {
            cache.next.psq[side].sort_unstable();
            cache.next.threats[side].sort_unstable();
            let refresh = cache.evaluations % 32 == 0;
            if refresh {
                cache.acc[side].copy_from_slice(&self.bias);
                cache.psqt[side].fill(0.);
            }
            update(
                &self.psq,
                &self.psqt,
                &cache.previous.psq[side],
                &cache.next.psq[side],
                refresh,
                &mut cache.acc[side],
                &mut cache.psqt[side],
            );
            update(
                &self.threats,
                &self.threat_psqt,
                &cache.previous.threats[side],
                &cache.next.threats[side],
                refresh,
                &mut cache.acc[side],
                &mut cache.psqt[side],
            );
        }
        let mut transformed = [0f32; TRANSFORMER_WIDTH];
        for side in 0..2 {
            for h in 0..TRANSFORMER_WIDTH / 2 {
                transformed[side * TRANSFORMER_WIDTH / 2 + h] = cache.acc[side][h].clamp(0., 255.)
                    * cache.acc[side][h + TRANSFORMER_WIDTH / 2].clamp(0., 255.)
                    / (512. * 128.);
            }
        }
        let stack = &self.stacks[cache.next.layer_stack];
        let mut fc0 = [0f32; FC_WIDTH];
        matvec(&transformed, &stack.w0, &stack.b0, &mut fc0);
        let mut ac0 = [0f32; FC_WIDTH * 2];
        activation(&fc0, &mut ac0);
        let mut fc1 = [0f32; FC_WIDTH];
        matvec(&ac0, &stack.w1, &stack.b1, &mut fc1);
        let mut features = [0f32; FC_WIDTH * 4];
        features[..FC_WIDTH * 2].copy_from_slice(&ac0);
        activation(&fc1, &mut features[FC_WIDTH * 2..]);
        let dense = features
            .iter()
            .zip(&stack.w2)
            .map(|(a, b)| a * b)
            .sum::<f32>()
            + stack.b2
            + fc0[FC_WIDTH - 2]
            - fc0[FC_WIDTH - 1];
        let bucket = cache.next.psqt_bucket;
        let result = dense + (cache.psqt[0][bucket] - cache.psqt[1][bucket]) * 0.5;
        std::mem::swap(&mut cache.next, &mut cache.previous);
        cache.evaluations += 1;
        if !result.is_finite() {
            candle_core::bail!("Pikafish evaluation is non-finite")
        }
        Ok(result.clamp(-1., 1.))
    }
}
fn update(
    table: &[f32],
    psqt: &[f32],
    old: &[usize],
    new: &[usize],
    refresh: bool,
    acc: &mut [f32],
    value: &mut [f32],
) {
    let add = |index: usize, sign: f32, acc: &mut [f32], value: &mut [f32]| {
        for (a, &w) in acc
            .iter_mut()
            .zip(&table[index * TRANSFORMER_WIDTH..(index + 1) * TRANSFORMER_WIDTH])
        {
            *a += sign * w;
        }
        for (a, &w) in value
            .iter_mut()
            .zip(&psqt[index * PSQT_BUCKETS..(index + 1) * PSQT_BUCKETS])
        {
            *a += sign * w;
        }
    };
    if refresh {
        for &index in new {
            add(index, 1., acc, value);
        }
        return;
    }
    let (mut a, mut b) = (0, 0);
    while a < old.len() || b < new.len() {
        if b == new.len() || a < old.len() && old[a] < new[b] {
            add(old[a], -1., acc, value);
            a += 1;
        } else if a == old.len() || new[b] < old[a] {
            add(new[b], 1., acc, value);
            b += 1;
        } else {
            a += 1;
            b += 1;
        }
    }
}
fn matvec(input: &[f32], weights: &[f32], bias: &[f32], out: &mut [f32]) {
    out.fill(0.);
    for (&value, row) in input.iter().zip(weights.chunks_exact(out.len())) {
        for (o, &w) in out.iter_mut().zip(row) {
            *o += value * w;
        }
    }
    for (o, &b) in out.iter_mut().zip(bias) {
        *o += b;
    }
}
fn activation(input: &[f32], out: &mut [f32]) {
    for (i, &x) in input.iter().enumerate() {
        let x = x.clamp(0., 1.);
        out[i] = x * x;
        out[i + input.len()] = x;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn incremental_all_buckets_duplicates_empty_and_snapshot_change() -> Result<()> {
        let model = PikafishModel::with_shape(
            PikafishShape {
                psq_features: 8,
                threat_features: 8,
            },
            &Device::Cpu,
        )?;
        let snapshot = model.cpu_snapshot()?;
        let mut cache = PikafishCpuCache::default();
        for i in 0..96 {
            let example = PikafishExample {
                psq: if i % 7 == 0 {
                    [vec![], vec![]]
                } else {
                    [vec![i % 8, (i + 1) % 8, i % 8], vec![(i + 3) % 8]]
                },
                threats: [vec![i % 8], vec![(i + 5) % 8, (i + 5) % 8]],
                psqt_bucket: i % 16,
                layer_stack: (i * 3) % 16,
            };
            let expected = model
                .forward_inference(std::slice::from_ref(&example))?
                .to_vec2::<f32>()?[0][0]
                .clamp(-1., 1.);
            let actual = snapshot.evaluate_example(&example, &mut cache)?;
            assert!(
                (actual - expected).abs() < 2e-5,
                "step {i}: {actual} vs {expected}"
            );
        }
        let example = PikafishExample {
            psq: [vec![1], vec![2]],
            threats: [vec![], vec![]],
            psqt_bucket: 0,
            layer_stack: 0,
        };
        let before = snapshot.evaluate_example(&example, &mut cache)?;
        model.stacks[0]
            .fc2_bias
            .set(&Tensor::full(0.7f32, 1, &Device::Cpu)?)?;
        assert!((snapshot.evaluate_example(&example, &mut cache)? - before).abs() < 2e-5);
        let newer = model.cpu_snapshot()?;
        assert!((newer.evaluate_example(&example, &mut cache)? - before - 0.7).abs() < 2e-5);
        assert!((snapshot.evaluate_example(&example, &mut cache)? - before).abs() < 2e-5);
        Ok(())
    }
}
