//! MLH 时序监督的抽样单元；只保留同局真实终局距离的配对。
use std::collections::HashMap;

use super::{AzTrainingSample, SplitMix64};

pub(super) const NO_PARTNER: usize = usize::MAX;

fn sample_key(sample: &AzTrainingSample) -> Option<(u32, u64, u16)> {
    (sample.meta.game_id != 0
        && sample.moves_left_weight.is_finite()
        && sample.moves_left_weight > 0.0
        && sample.moves_left.is_finite()
        && sample.moves_left >= 0.0)
        .then_some((
            sample.meta.generation_update,
            sample.meta.game_id,
            sample.meta.ply,
        ))
}

/// 在 SKIP 抽样前组成单元；保留每个原样本的入选概率，不改变终局标签。
pub(super) fn sampling_units(samples: &[&AzTrainingSample], seed: u64) -> Vec<[usize; 2]> {
    let mut units = Vec::with_capacity(samples.len().div_ceil(2));
    let mut used = vec![false; samples.len()];
    let positions = samples
        .iter()
        .enumerate()
        .filter_map(|(index, sample)| sample_key(sample).map(|key| (key, index)))
        .collect::<HashMap<_, _>>();
    let mut rng = SplitMix64::new(seed);
    for (index, sample) in samples.iter().enumerate() {
        if used[index] {
            continue;
        }
        let Some((generation, game, ply)) = sample_key(sample) else {
            continue;
        };
        let start = (rng.next_u64() % 6) as usize;
        let partner = (0..6).find_map(|offset| {
            let choice = (start + offset) % 6;
            let gap = [2, 8, 16][choice / 2];
            let next = if choice % 2 == 0 {
                ply.checked_add(gap)
            } else {
                ply.checked_sub(gap)
            }?;
            let &other = positions.get(&(generation, game, next))?;
            let expected = f32::from(next) - f32::from(ply);
            (!used[other]
                && (sample.moves_left - samples[other].moves_left - expected).abs() < 0.01)
                .then_some(other)
        });
        if let Some(other) = partner {
            used[index] = true;
            used[other] = true;
            units.push([index, other]);
        }
    }
    let remaining = (0..samples.len()).filter(|&i| !used[i]).collect::<Vec<_>>();
    for chunk in remaining.chunks(2) {
        units.push([chunk[0], chunk.get(1).copied().unwrap_or(NO_PARTNER)]);
    }
    for i in (1..units.len()).rev() {
        let j = rng.next_u64() as usize % (i + 1);
        units.swap(i, j);
    }
    units
}
