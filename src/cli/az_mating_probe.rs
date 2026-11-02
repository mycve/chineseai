use crate::cli::args::AzMatingProbeArgs;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use chineseai::az::{
    AzExperiencePool, AzNnue, AzTrainingSample, MATING_HEAD_SIZE, SplitMix64,
    mating_outputs_for_training_sample, replay_game_is_test,
};
use chineseai::xiangqi::MatingPattern;
use std::{error::Error, io, path::Path};

type ProbeResult<T> = Result<T, Box<dyn Error>>;
type Targets = [f32; MATING_HEAD_SIZE];

pub(crate) fn run(cmd: AzMatingProbeArgs) {
    if let Err(error) = execute(&cmd) {
        eprintln!("杀法探针失败：{error}");
        std::process::exit(2);
    }
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn execute(cmd: &AzMatingProbeArgs) -> ProbeResult<()> {
    if cmd.train_samples == 0
        || cmd.test_samples == 0
        || cmd.steps == 0
        || cmd.capacity == 0
        || !cmd.lr.is_finite()
        || cmd.lr <= 0.0
    {
        return Err(invalid("样本数、steps、capacity 和有限学习率必须为正").into());
    }
    let pool = AzExperiencePool::load_snapshot_lz4(Path::new(&cmd.replay), cmd.capacity)?;
    if !pool.has_grouped_game_ids() {
        return Err(invalid(
            "回放 chunk 混装多个 game_id，无法保证探针与正式训练使用相同整盘留出划分",
        )
        .into());
    }
    let (train, test) = split_reservoir(
        pool.iter_samples()
            .filter(|s| s.mating_weight > 0.0 && s.meta.game_id != 0),
        |s| s.meta.game_id,
        cmd.split_seed,
        cmd.train_samples,
        cmd.test_samples,
    );
    if train.is_empty() || test.is_empty() {
        return Err(invalid(format!(
            "有效训练/测试样本不足：train={} test={}；需要有终局标签和非零 game_id，并使用训练时相同 seed",
            train.len(), test.len(),
        )).into());
    }
    let train_y = targets(&train)?;
    let test_y = targets(&test)?;
    println!(
        "冻结主干探针：train={} test={} split_seed={} steps={} lr={}；按整盘留出、无放回抽样",
        train.len(),
        test.len(),
        cmd.split_seed,
        cmd.steps,
        cmd.lr,
    );
    let supported = (0..MATING_HEAD_SIZE)
        .filter(|&k| support(&train_y, k) && support(&test_y, k))
        .count();
    if supported == 0 {
        print_support(&train_y, &test_y);
        return Err(invalid("全部杀法头均缺少训练或测试正负样本，无法判断表征能力").into());
    }
    evaluate_model("模型", &cmd.model, &train, &test, &train_y, &test_y, cmd)?;
    if let Some(path) = &cmd.baseline {
        evaluate_model("对照", path, &train, &test, &train_y, &test_y, cmd)?;
    }
    println!(
        "探针只在内存拟合，模型与回放未写入；仅正负样本齐备的头可用于比较表征。原头 BCE 概率裁剪至 [1e-7, 1-1e-7]。"
    );
    Ok(())
}

/// 两个 reservoir 各自不重复取条目；整盘 hash 决定唯一分区。
fn split_reservoir<T>(
    items: impl IntoIterator<Item = T>,
    game_id: impl Fn(&T) -> u64,
    seed: u64,
    train_limit: usize,
    test_limit: usize,
) -> (Vec<T>, Vec<T>) {
    let mut train = Reservoir::new(train_limit, seed ^ 0x9E37_79B9_7F4A_7C15);
    let mut test = Reservoir::new(test_limit, seed ^ 0xD1B5_4A32_D192_ED03);
    for item in items {
        if replay_game_is_test(game_id(&item), seed) {
            test.push(item);
        } else {
            train.push(item);
        }
    }
    (train.items, test.items)
}

struct Reservoir<T> {
    items: Vec<T>,
    seen: u64,
    limit: usize,
    rng: SplitMix64,
}

impl<T> Reservoir<T> {
    fn new(limit: usize, seed: u64) -> Self {
        Self {
            items: Vec::new(),
            seen: 0,
            limit,
            rng: SplitMix64::new(seed),
        }
    }

    fn push(&mut self, item: T) {
        self.seen += 1;
        if self.items.len() < self.limit {
            self.items.push(item);
        } else {
            let index = (self.rng.next_u64() % self.seen) as usize;
            if index < self.limit {
                self.items[index] = item;
            }
        }
    }
}

fn targets(samples: &[&AzTrainingSample]) -> ProbeResult<Vec<Targets>> {
    samples
        .iter()
        .map(|sample| {
            if !sample.mating_weight.is_finite()
                || sample.mating_targets.iter().any(|&v| v != 0.0 && v != 1.0)
            {
                return Err(invalid(format!(
                    "game_id={} 的杀法监督不是有效二值标签",
                    sample.meta.game_id
                ))
                .into());
            }
            Ok(sample.mating_targets)
        })
        .collect()
}

fn positive(labels: &[Targets], head: usize) -> usize {
    labels.iter().filter(|row| row[head] > 0.5).count()
}

fn support(labels: &[Targets], head: usize) -> bool {
    let count = positive(labels, head);
    count > 0 && count < labels.len()
}

fn print_support(train: &[Targets], test: &[Targets]) {
    for (k, pattern) in MatingPattern::ALL.into_iter().enumerate() {
        println!(
            "{}：训练正={}/{} 测试正={}/{}，样本不足（需两组均有正负样本）",
            pattern.name(),
            positive(train, k),
            train.len(),
            positive(test, k),
            test.len()
        );
    }
}

fn encode(model: &AzNnue, samples: &[&AzTrainingSample]) -> ProbeResult<(Tensor, Vec<Vec<f32>>)> {
    let mut hidden = Vec::new();
    let mut outputs = Vec::with_capacity(samples.len());
    let mut width = None;
    for sample in samples {
        let (probabilities, representation) = mating_outputs_for_training_sample(model, sample)
            .ok_or_else(|| invalid(format!("game_id={} 的局面无法恢复", sample.meta.game_id)))?;
        if representation.is_empty()
            || representation
                .iter()
                .chain(probabilities.iter())
                .any(|v| !v.is_finite())
            || width.is_some_and(|width| width != representation.len())
        {
            return Err(invalid("模型产生无效共享表示或概率").into());
        }
        width = Some(representation.len());
        hidden.extend(representation);
        outputs.push(probabilities.to_vec());
    }
    Ok((
        Tensor::from_vec(hidden, (samples.len(), width.unwrap()), &Device::Cpu)?,
        outputs,
    ))
}

/// 训练集统计归一化是固定仿射变换；探针始终是共享表示上的线性 sigmoid。
fn fit_probe(
    train: &Tensor,
    labels: &Tensor,
    test: &Tensor,
    steps: usize,
    lr: f64,
) -> candle_core::Result<Tensor> {
    let device = train.device();
    let mean = train.mean_keepdim(0)?;
    let centered = train.broadcast_sub(&mean)?;
    let scale = centered
        .sqr()?
        .mean_keepdim(0)?
        .clamp(1e-6, f64::INFINITY)?
        .sqrt()?;
    let train = centered.broadcast_div(&scale)?;
    let test = test.broadcast_sub(&mean)?.broadcast_div(&scale)?;
    let width = train.dim(1)?;
    let weights = Var::zeros((width, MATING_HEAD_SIZE), DType::F32, device)?;
    let bias = Var::zeros(MATING_HEAD_SIZE, DType::F32, device)?;
    let mut optimizer = AdamW::new(
        vec![weights.clone(), bias.clone()],
        ParamsAdamW {
            lr,
            weight_decay: 0.0,
            ..Default::default()
        },
    )?;
    for _ in 0..steps {
        let logits = train
            .matmul(weights.as_tensor())?
            .broadcast_add(bias.as_tensor())?;
        optimizer.backward_step(&stable_bce(&logits, labels)?)?;
    }
    test.matmul(weights.as_tensor())?
        .broadcast_add(bias.as_tensor())
}

fn stable_bce(logits: &Tensor, labels: &Tensor) -> candle_core::Result<Tensor> {
    let zeros = logits.zeros_like()?;
    (Tensor::stack(&[&zeros, logits], 2)?.log_sum_exp(2)? - (logits * labels)?)?.mean_all()
}

fn metrics(predictions: &[Vec<f32>], labels: &[Targets], head: usize, logits: bool) -> (f64, f64) {
    let mut bce = 0.0;
    let mut brier = 0.0;
    for (row, target) in predictions.iter().zip(labels) {
        let value = f64::from(row[head]);
        let y = f64::from(target[head]);
        let probability = if logits {
            1.0 / (1.0 + (-value).exp())
        } else {
            value
        };
        brier += (probability - y).powi(2);
        bce += if logits {
            value.max(0.0) - value * y + (-value.abs()).exp().ln_1p()
        } else {
            let p = probability.clamp(1e-7, 1.0 - 1e-7);
            -y * p.ln() - (1.0 - y) * (-p).ln_1p()
        };
    }
    (bce / labels.len() as f64, brier / labels.len() as f64)
}

fn evaluate_model(
    name: &str,
    path: &str,
    train: &[&AzTrainingSample],
    test: &[&AzTrainingSample],
    train_y: &[Targets],
    test_y: &[Targets],
    cmd: &AzMatingProbeArgs,
) -> ProbeResult<()> {
    let model = AzNnue::load(path)?;
    let (train_x, _) = encode(&model, train)?;
    let (test_x, original) = encode(&model, test)?;
    let labels = Tensor::from_vec(
        train_y.iter().flatten().copied().collect::<Vec<_>>(),
        (train.len(), MATING_HEAD_SIZE),
        &Device::Cpu,
    )?;
    let probe = fit_probe(&train_x, &labels, &test_x, cmd.steps, cmd.lr)?.to_vec2::<f32>()?;
    if probe.iter().flatten().any(|v| !v.is_finite()) {
        return Err(invalid("探针拟合出现非有限结果").into());
    }
    println!("{name}：{path}");
    for (k, pattern) in MatingPattern::ALL.into_iter().enumerate() {
        let train_positive = positive(train_y, k);
        let test_positive = positive(test_y, k);
        let counts = format!(
            "训练正={train_positive}/{} 测试正={test_positive}/{}",
            train.len(),
            test.len()
        );
        if !support(train_y, k) || !support(test_y, k) {
            println!(
                "{}：{counts}，样本不足（需两组均有正负样本）",
                pattern.name()
            );
            continue;
        }
        let (original_bce, original_brier) = metrics(&original, test_y, k, false);
        let (probe_bce, probe_brier) = metrics(&probe, test_y, k, true);
        let frequency = train_positive as f64 / train.len() as f64;
        let constant_brier = test_y
            .iter()
            .map(|row| (frequency - f64::from(row[k])).powi(2))
            .sum::<f64>()
            / test.len() as f64;
        println!(
            "{}：{counts} 原头 BCE={original_bce:.5} Brier={original_brier:.5} 冻结探针 BCE={probe_bce:.5} Brier={probe_brier:.5} 常数频率 Brier={constant_brier:.5}",
            pattern.name()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::args::{Cli, CliCommand};
    use clap::Parser;
    use std::collections::HashSet;

    #[test]
    fn requires_explicit_training_split_seed() {
        let args = [
            "chineseai",
            "az-mating-probe",
            "--model",
            "a",
            "--replay",
            "b",
        ];
        assert!(Cli::try_parse_from(args).is_err());
        let cli = Cli::try_parse_from(args.into_iter().chain(["--split-seed", "42"])).unwrap();
        assert!(matches!(cli.command, Some(CliCommand::AzMatingProbe(_))));
    }

    #[test]
    fn game_reservoirs_are_deterministic_disjoint_and_without_replacement() {
        let items = (0..2000)
            .map(|index| (index, 1 + index / 5))
            .collect::<Vec<_>>();
        let (train, test) = split_reservoir(items.clone(), |item| item.1, 42, 100, 30);
        assert_eq!(train.len(), 100);
        assert_eq!(test.len(), 30);
        assert_eq!(
            (train.clone(), test.clone()),
            split_reservoir(items, |item| item.1, 42, 100, 30)
        );
        assert_eq!(
            train.iter().map(|x| x.0).collect::<HashSet<_>>().len(),
            train.len()
        );
        assert_eq!(
            test.iter().map(|x| x.0).collect::<HashSet<_>>().len(),
            test.len()
        );
        let train_games = train.iter().map(|x| x.1).collect::<HashSet<_>>();
        assert!(
            test.iter()
                .all(|x| !train_games.contains(&x.1) && replay_game_is_test(x.1, 42))
        );
        assert!(train.iter().all(|x| !replay_game_is_test(x.1, 42)));
    }

    #[test]
    fn frozen_linear_probe_learns_separable_representation_without_changing_inputs() {
        let train = Tensor::new(&[[-2f32], [-1.0], [1.0], [2.0]], &Device::Cpu).unwrap();
        let test = Tensor::new(&[[-1.5f32], [1.5]], &Device::Cpu).unwrap();
        let targets = Tensor::from_vec(
            [0f32, 0.0, 1.0, 1.0]
                .into_iter()
                .flat_map(|y| [y; MATING_HEAD_SIZE])
                .collect::<Vec<_>>(),
            (4, MATING_HEAD_SIZE),
            &Device::Cpu,
        )
        .unwrap();
        let test_targets = Tensor::from_vec(
            [0f32, 1.0]
                .into_iter()
                .flat_map(|y| [y; MATING_HEAD_SIZE])
                .collect::<Vec<_>>(),
            (2, MATING_HEAD_SIZE),
            &Device::Cpu,
        )
        .unwrap();
        let before = train.to_vec2::<f32>().unwrap();
        let logits = fit_probe(&train, &targets, &test, 100, 0.05).unwrap();
        assert!(
            stable_bce(&logits, &test_targets)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                < 0.2
        );
        assert_eq!(train.to_vec2::<f32>().unwrap(), before);
        let extreme = Tensor::new(&[[1000f32, -1000.0]], &Device::Cpu).unwrap();
        let target = Tensor::new(&[[0f32, 1.0]], &Device::Cpu).unwrap();
        assert!(
            stable_bce(&extreme, &target)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                .is_finite()
        );
    }
}
