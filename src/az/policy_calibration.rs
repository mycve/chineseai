//! 冻结主网络，用完整候选策略分布校准已有的 448 项战术因子。
use std::io;

use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};

use super::*;
use crate::xiangqi::BOARD_SIZE;

/// 交叉熵均在提供的训练样本上计算，不代表独立验证集收益。
#[derive(Debug)]
pub struct PolicyCalibrationReport {
    pub samples: usize,
    pub before_ce: f64,
    pub after_ce: f64,
    pub max_fold_error: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> AzTrainingSample {
        let position = Position::startpos();
        let moves = position.legal_moves();
        let mut policy = vec![0.0; moves.len()];
        policy[0] = 1.0;
        AzTrainingSample {
            features: nnue::extract_sparse_features_az(&position),
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            move_indices: moves.iter().map(|mv| dense_move_index(*mv)).collect(),
            repetition_flags: Vec::new(),
            policy,
            value_wdl: [0.0, 1.0, 0.0],
            root_search_wdl: [0.0, 1.0, 0.0],
            value: 0.0,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            moves_left: 0.0,
            moves_left_weight: 0.0,
            search_simulations: 0,
            meta: AzSampleMeta::default(),
        }
    }

    #[test]
    fn small_batch_calibration_folds_and_preserves_all_other_weights() {
        let mut model = AzNnue::random(8, 17);
        model.rebuild_policy_tactical();
        let sample = fixture();
        let original_wdl = outputs_for_training_sample(&model, &sample).unwrap().0;
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tmp")
            .join(format!("calibration-test-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let before = directory.join("before.safetensors");
        let after = directory.join("after.safetensors");
        model.save(&before).unwrap();
        let report = calibrate_policy(&mut model, std::slice::from_ref(&sample), 20, 17).unwrap();
        assert_eq!(report.samples, 1);
        assert!(report.after_ce < report.before_ce);
        assert!(report.max_fold_error < 1e-4);
        assert_eq!(
            outputs_for_training_sample(&model, &sample)
                .unwrap()
                .0
                .map(f32::to_bits),
            original_wdl.map(f32::to_bits)
        );
        model.save(&after).unwrap();
        let tensors_before = candle_core::safetensors::load(&before, &Device::Cpu).unwrap();
        let tensors_after = candle_core::safetensors::load(&after, &Device::Cpu).unwrap();
        assert_eq!(tensors_before.len(), tensors_after.len());
        for (name, tensor) in tensors_before {
            let a = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let b = tensors_after[&name]
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            if name == "policy_tactical" {
                let start = POLICY_TACTICAL_EXACT_SIZE;
                let end = start + POLICY_TACTICAL_FACTOR_SIZE;
                assert_eq!(a[..start], b[..start]);
                assert_eq!(a[end..], b[end..]);
                assert_ne!(a[start..end], b[start..end]);
            } else {
                assert_eq!(a, b, "校准意外修改参数 {name}");
            }
        }
        let reloaded = AzNnue::load(&after).unwrap();
        assert_eq!(
            outputs_for_training_sample(&model, &sample),
            outputs_for_training_sample(&reloaded, &sample)
        );
        std::fs::remove_file(before).unwrap();
        std::fs::remove_file(after).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn invalid_calibration_inputs_leave_model_unchanged() {
        let mut model = AzNnue::random(8, 19);
        model.rebuild_policy_tactical();
        let original = model.policy_tactical.clone();
        assert!(calibrate_policy(&mut model, &[], 1, 17).is_err());
        let mut sample = fixture();
        assert!(calibrate_policy(&mut model, std::slice::from_ref(&sample), 0, 17).is_err());
        sample.move_indices[0] = DENSE_MOVE_SPACE;
        assert!(calibrate_policy(&mut model, std::slice::from_ref(&sample), 1, 17).is_err());
        assert_eq!(model.policy_tactical, original);
    }
}

struct Row {
    sample: usize,
    logits: Vec<f32>,
    target: Vec<f32>,
    classes: Vec<usize>,
    weight: f64,
    wdl: [f32; 3],
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn prepare(model: &AzNnue, samples: &[AzTrainingSample]) -> io::Result<Vec<Row>> {
    let mut rows = Vec::new();
    for (i, sample) in samples.iter().enumerate() {
        let fail = || invalid(format!("策略校准样本 {i} 无效"));
        if !sample.policy_weight.is_finite() || sample.policy_weight < 0.0 {
            return Err(fail());
        }
        // 纯价值样本不参与策略校准。
        if sample.policy_weight == 0.0 {
            continue;
        }
        if sample.move_indices.is_empty()
            || sample.move_indices.len() != sample.policy.len()
            || (!sample.repetition_flags.is_empty()
                && sample.repetition_flags.len() != sample.move_indices.len())
            || sample.rule_context.iter().any(|x| !x.is_finite())
            || sample.policy.iter().any(|x| !x.is_finite() || *x < 0.0)
            || (sample.policy.iter().map(|x| f64::from(*x)).sum::<f64>() - 1.0).abs() > 1e-3
        {
            return Err(fail());
        }
        let mut occupied = [false; BOARD_SIZE];
        let mut kings = [0; 2];
        for &feature in &sample.features {
            if feature >= nnue::AZ_NNUE_INPUT_SIZE || occupied[feature % BOARD_SIZE] {
                return Err(fail());
            }
            occupied[feature % BOARD_SIZE] = true;
            match feature / BOARD_SIZE {
                0 => kings[0] += 1,
                7 => kings[1] += 1,
                _ => {}
            }
        }
        if kings != [1, 1] {
            return Err(fail());
        }
        let position = position_for_training_sample(sample).ok_or_else(fail)?;
        let side = position.side_to_move();
        let masks = position.attacked_squares_masks();
        let legal = position.legal_moves();
        let mut seen = vec![false; DENSE_MOVE_SPACE];
        let mut classes = Vec::with_capacity(sample.move_indices.len());
        for &index in &sample.move_indices {
            let (from, to) = dense_move_squares(index).ok_or_else(fail)?;
            if seen[index] {
                return Err(fail());
            }
            seen[index] = true;
            let mv = Move {
                from: from as u8,
                to: to as u8,
            };
            if !legal.contains(&mv) {
                return Err(fail());
            }
            let (source, _, captured) =
                policy_consequence_features(&position, side, mv).ok_or_else(fail)?;
            let (sa, da, sd, dd) = policy_move_tactical_flags(
                mv,
                masks[1 - color_index(side)],
                masks[color_index(side)],
            );
            classes.push(
                policy_tactical_indices(
                    index,
                    source / BOARD_SIZE,
                    sa,
                    da,
                    sd,
                    dd,
                    captured.map(|x| x / BOARD_SIZE),
                    position.gives_check_after_move_fast(mv),
                )[1] - POLICY_TACTICAL_EXACT_SIZE,
            );
        }
        let (wdl, logits) = outputs_for_training_sample(model, sample).ok_or_else(fail)?;
        if logits.iter().chain(wdl.iter()).any(|x| !x.is_finite()) {
            return Err(fail());
        }
        rows.push(Row {
            sample: i,
            logits,
            target: sample.policy.clone(),
            classes,
            weight: f64::from(sample.policy_weight),
            wdl,
        });
    }
    if rows.is_empty() {
        return Err(invalid("策略校准需要至少一个有效的策略训练样本"));
    }
    Ok(rows)
}

fn cross_entropy(rows: &[Row], delta: &[f32]) -> f64 {
    let mut total = 0.0;
    let mut weights = 0.0;
    for row in rows {
        let values: Vec<f64> = row
            .logits
            .iter()
            .zip(&row.classes)
            .map(|(&base, &class)| f64::from(base) + f64::from(delta[class]))
            .collect();
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let log_z = max + values.iter().map(|x| (x - max).exp()).sum::<f64>().ln();
        total += row.weight
            * values
                .iter()
                .zip(&row.target)
                .map(|(x, &p)| f64::from(p) * (log_z - x))
                .sum::<f64>();
        weights += row.weight;
    }
    total / weights
}

/// 只更新战术因子；其余参数与 WDL 输出保持不变。失败时保留原模型。
/// 使用 CPU AdamW，学习率与权重衰减均为 0.01，批大小上限为 256。
pub fn calibrate_policy(
    model: &mut AzNnue,
    samples: &[AzTrainingSample],
    steps: usize,
    seed: u64,
) -> io::Result<PolicyCalibrationReport> {
    if steps == 0 {
        return Err(invalid("策略校准步数必须大于零"));
    }
    model.validate()?;
    let rows = prepare(model, samples)?;
    let device = Device::Cpu;
    let weight =
        Var::zeros(POLICY_TACTICAL_FACTOR_SIZE, DType::F32, &device).map_err(candle_io_error)?;
    let mut optimizer = AdamW::new(
        vec![weight.clone()],
        ParamsAdamW {
            lr: 0.01,
            weight_decay: 0.01,
            ..Default::default()
        },
    )
    .map_err(candle_io_error)?;
    let mut rng = SplitMix64::new(seed);
    let mut order: Vec<usize> = (0..rows.len()).collect();
    let batch_size = rows.len().min(256);
    let mut cursor = rows.len();
    for _ in 0..steps {
        if cursor + batch_size > order.len() {
            for i in (1..order.len()).rev() {
                order.swap(i, (rng.next_u64() % (i as u64 + 1)) as usize);
            }
            cursor = 0;
        }
        let batch = &order[cursor..cursor + batch_size];
        cursor += batch_size;
        let width = batch.iter().map(|&i| rows[i].logits.len()).max().unwrap();
        let weight_sum: f64 = batch.iter().map(|&i| rows[i].weight).sum();
        let mut indices = vec![POLICY_TACTICAL_FACTOR_SIZE as u32; batch_size * width];
        let mut base = vec![-1e9f32; indices.len()];
        let mut target = vec![0.0f32; indices.len()];
        for (b, &i) in batch.iter().enumerate() {
            let row = &rows[i];
            for j in 0..row.logits.len() {
                let k = b * width + j;
                indices[k] = row.classes[j] as u32;
                base[k] = row.logits[j];
                target[k] = (f64::from(row.target[j]) * row.weight / weight_sum) as f32;
            }
        }
        let result = (|| -> candle_core::Result<()> {
            let zero = Tensor::zeros(1, DType::F32, &device)?;
            let padded = Tensor::cat(&[weight.as_tensor(), &zero], 0)?;
            let ids = Tensor::from_vec(indices, batch_size * width, &device)?;
            let residual = padded.index_select(&ids, 0)?.reshape((batch_size, width))?;
            let logits = (Tensor::from_vec(base, (batch_size, width), &device)? + residual)?;
            let targets = Tensor::from_vec(target, (batch_size, width), &device)?;
            let loss = (candle_nn::ops::log_softmax(&logits, 1)? * targets)?
                .sum_all()?
                .neg()?;
            optimizer.backward_step(&loss)
        })();
        result.map_err(candle_io_error)?;
    }
    let delta = weight.to_vec1::<f32>().map_err(candle_io_error)?;
    if delta.iter().any(|x| !x.is_finite()) {
        return Err(invalid("策略校准产生非有限参数"));
    }
    let before_ce = cross_entropy(&rows, &vec![0.0; POLICY_TACTICAL_FACTOR_SIZE]);
    let after_ce = cross_entropy(&rows, &delta);
    let range =
        POLICY_TACTICAL_EXACT_SIZE..POLICY_TACTICAL_EXACT_SIZE + POLICY_TACTICAL_FACTOR_SIZE;
    let original = model.policy_tactical[range.clone()].to_vec();
    for ((value, old), addition) in model.policy_tactical[range.clone()]
        .iter_mut()
        .zip(&original)
        .zip(&delta)
    {
        *value = old + addition;
    }
    model.rebuild_policy_tactical();
    let audit = (|| -> io::Result<f32> {
        let mut error = 0.0f32;
        for row in rows.iter().take(64) {
            let (wdl, logits) = outputs_for_training_sample(model, &samples[row.sample])
                .ok_or_else(|| invalid("策略校准折叠审计无法重建样本"))?;
            if wdl.map(f32::to_bits) != row.wdl.map(f32::to_bits) {
                return Err(invalid("策略校准改变了 WDL 输出"));
            }
            let shift = logits[0] - row.logits[0] - delta[row.classes[0]];
            for ((actual, base), &class) in logits.iter().zip(&row.logits).zip(&row.classes) {
                let difference = (actual - base - delta[class] - shift).abs();
                if !difference.is_finite() {
                    return Err(invalid("策略校准折叠输出非有限"));
                }
                error = error.max(difference);
            }
        }
        if error > 1e-4 {
            return Err(invalid(format!("策略校准折叠误差过大: {error}")));
        }
        Ok(error)
    })();
    match audit {
        Ok(max_fold_error) => {
            model.gpu_trainer = None;
            Ok(PolicyCalibrationReport {
                samples: rows.len(),
                before_ce,
                after_ce,
                max_fold_error,
            })
        }
        Err(error) => {
            model.policy_tactical[range].copy_from_slice(&original);
            model.rebuild_policy_tactical();
            Err(error)
        }
    }
}
