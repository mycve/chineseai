use chineseai::az::{AzTrainingSample, px0_data};
use std::{collections::HashMap, path::Path};

fn key(sample: &AzTrainingSample, full: bool) -> Vec<u8> {
    let mut features = sample.features.clone();
    features.sort_unstable();
    let mut bytes = Vec::with_capacity(features.len() * 2 + 32 + sample.move_indices.len() * 3);
    bytes.extend_from_slice(&(features.len() as u16).to_le_bytes());
    for feature in features {
        bytes.extend_from_slice(&(feature as u16).to_le_bytes());
    }
    if full {
        for value in sample.rule_context {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        let mut moves = sample
            .move_indices
            .iter()
            .enumerate()
            .map(|(i, &mv)| (mv, sample.repetition_flags[i]))
            .collect::<Vec<_>>();
        moves.sort_unstable();
        for (mv, repeat) in moves {
            bytes.extend_from_slice(&(mv as u16).to_le_bytes());
            bytes.push(repeat);
        }
    } else {
        bytes.extend_from_slice(&sample.rule_context[0].to_bits().to_le_bytes());
    }
    bytes
}

fn policy(sample: &AzTrainingSample) -> Vec<(usize, f64)> {
    let mut values = sample
        .move_indices
        .iter()
        .copied()
        .zip(sample.policy.iter().map(|&v| v as f64))
        .collect::<Vec<_>>();
    values.sort_unstable_by_key(|v| v.0);
    values
}

fn audit(samples: &[AzTrainingSample], full: bool) {
    let mut first = HashMap::new();
    let mut duplicates: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, sample) in samples.iter().enumerate() {
        let k = key(sample, full);
        if let Some(&j) = first.get(&k) {
            duplicates.entry(j).or_insert_with(|| vec![j]).push(i);
        } else {
            first.insert(k, i);
        }
    }
    let repeated_samples = duplicates.values().map(Vec::len).sum::<usize>();
    let mut q_sse = 0.0;
    let mut policy_floor_sum = 0.0;
    let mut max_q_range: f64 = 0.0;
    let mut max_js: f64 = 0.0;
    let mut q_gt_01 = 0;
    let mut q_gt_025 = 0;
    let mut policy_gt_01 = 0;
    let mut top1_disagree = 0;
    let mut mismatch_move_sets = 0;
    let mut worst = vec![];
    for ids in duplicates.values() {
        let mean_q = ids.iter().map(|&i| samples[i].value as f64).sum::<f64>() / ids.len() as f64;
        let qs = ids
            .iter()
            .map(|&i| samples[i].value as f64)
            .collect::<Vec<_>>();
        let range = qs.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - qs.iter().copied().fold(f64::INFINITY, f64::min);
        max_q_range = max_q_range.max(range);
        if range > 0.1 {
            q_gt_01 += 1;
        }
        if range > 0.25 {
            q_gt_025 += 1;
        }
        q_sse += qs.iter().map(|&q| (q - mean_q).powi(2)).sum::<f64>();
        let ps = ids.iter().map(|&i| policy(&samples[i])).collect::<Vec<_>>();
        if ps
            .iter()
            .any(|p| p.len() != ps[0].len() || p.iter().zip(&ps[0]).any(|(a, b)| a.0 != b.0))
        {
            mismatch_move_sets += 1;
            continue;
        }
        let mut mean_p = vec![0.0; ps[0].len()];
        for p in &ps {
            for (j, &(_, v)) in p.iter().enumerate() {
                mean_p[j] += v / ids.len() as f64;
            }
        }
        let js = ps
            .iter()
            .map(|p| {
                p.iter()
                    .zip(&mean_p)
                    .filter(|(v, _)| v.1 > 0.0)
                    .map(|(v, &m)| v.1 * (v.1 / m).ln())
                    .sum::<f64>()
            })
            .sum::<f64>()
            / ids.len() as f64;
        policy_floor_sum += js * ids.len() as f64;
        max_js = max_js.max(js);
        if js > 0.1 {
            policy_gt_01 += 1;
        }
        let top = |p: &Vec<(usize, f64)>| p.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
        if ps.iter().any(|p| top(p) != top(&ps[0])) {
            top1_disagree += 1;
        }
        worst.push((
            range,
            js,
            ids.len(),
            samples[ids[0]].meta.game_id,
            samples[ids[0]].meta.ply,
        ));
    }
    worst.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    println!(
        "key={} total_samples={} unique_inputs={} repeated_groups={} repeated_samples={} repeated_fraction={:.8} q_group_range_gt_01={} q_group_range_gt_025={} max_q_range={:.6} duplicate_Q_RMSE_floor={:.6} all_sample_Q_RMSE_floor={:.6} duplicate_policy_KL_floor={:.6} all_sample_policy_KL_floor={:.8} max_group_policy_JS={:.6} policy_JS_gt_01={} top1_disagree_groups={} mismatch_move_sets={}",
        if full {
            "complete_model_input"
        } else {
            "board_ruleclock"
        },
        samples.len(),
        first.len(),
        duplicates.len(),
        repeated_samples,
        repeated_samples as f64 / samples.len() as f64,
        q_gt_01,
        q_gt_025,
        max_q_range,
        (q_sse / repeated_samples.max(1) as f64).sqrt(),
        (q_sse / samples.len() as f64).sqrt(),
        policy_floor_sum / repeated_samples.max(1) as f64,
        policy_floor_sum / samples.len() as f64,
        max_js,
        policy_gt_01,
        top1_disagree,
        mismatch_move_sets
    );
    println!(
        "worst_Q_groups(range,JS,count,game_id,ply)={:?}",
        &worst[..worst.len().min(10)]
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut data = px0_data::load(Path::new("data/data.bin"), 4096, 4096)?;
    data.train.append(&mut data.validation);
    println!(
        "games={} samples={} deleted={}",
        data.games,
        data.train.len(),
        data.deleted
    );
    audit(&data.train, false);
    audit(&data.train, true);
    Ok(())
}
