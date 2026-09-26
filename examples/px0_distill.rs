use chineseai::az::{AzNnue, AzTrainLossWeights, SplitMix64, px0_data, train_samples_weighted};
use clap::Parser;
use std::{path::Path, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(default_value = "data/data.bin")]
    data: String,
    #[arg(long, default_value_t = 256)]
    games: usize,
    #[arg(long, default_value_t = 128)]
    hidden: usize,
    #[arg(long, default_value_t = 5)]
    epochs: usize,
    #[arg(long, default_value_t = 512)]
    batch: usize,
    #[arg(long, default_value_t = 0.0004)]
    lr: f32,
    #[arg(long, default_value_t = 20260927)]
    seed: u64,
    #[arg(long, default_value = "tmp/px0-distill.safetensors")]
    output: String,
    /// 从已有检查点继续监督训练；优化器状态重新初始化。
    #[arg(long)]
    initial_model: Option<String>,
    /// 小样本记忆实验，保留独立的整局验证集。
    #[arg(long)]
    train_limit: Option<usize>,
    /// 固定验证集来自归档前多少局；扩大训练集时保持此值。
    #[arg(long, default_value_t = 1024)]
    validation_games: usize,
    #[arg(long, default_value_t = 1.0)]
    policy_weight: f32,
    #[arg(long, default_value_t = 1.0)]
    value_weight: f32,
    #[arg(long, default_value_t = 5)]
    eval_every: usize,
    #[arg(long, default_value_t = 8192)]
    eval_train_samples: usize,
    /// 在整个归档中均匀抽样；games 此时指定训练对局数。
    #[arg(long)]
    reservoir: bool,
    /// 完成指定轮数后降低学习率。
    #[arg(long)]
    lr_drop_after: Option<usize>,
    #[arg(long, default_value_t = 0.1)]
    lr_drop_factor: f32,
    /// 对已有网络只做前向评测，可重复提供多个网络。
    #[arg(long)]
    evaluate_model: Vec<String>,
    /// 跳过已用于选优的验证前缀，评测后续独立留出对局。
    #[arg(long, default_value_t = 0)]
    validation_skip_samples: usize,
    /// 连续多少次评估没有达到最低相对改善时停止；0 不提前停止。
    #[arg(long, default_value_t = 0)]
    early_stop_patience: usize,
    #[arg(long, default_value_t = 0.005)]
    min_improvement: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let mut data = if args.reservoir {
        px0_data::load_reservoir(
            Path::new(&args.data),
            args.games,
            args.validation_games,
            args.seed,
        )?
    } else {
        px0_data::load(Path::new(&args.data), args.games, args.validation_games)?
    };
    if let Some(limit) = args.train_limit {
        data.train.truncate(limit);
    }
    println!(
        "split games={} train={} validation={} deleted={} target=best_search_wdl short_loss=0",
        data.games,
        data.train.len(),
        data.validation.len(),
        data.deleted
    );
    if !args.evaluate_model.is_empty() {
        let validation = data
            .validation
            .get(args.validation_skip_samples..)
            .filter(|samples| !samples.is_empty())
            .ok_or("empty evaluation holdout")?;
        for path in &args.evaluate_model {
            let model = AzNnue::load(path)?;
            println!(
                "evaluation model={path} metrics={:?}",
                px0_data::evaluate(&model, validation)
            );
        }
        return Ok(());
    }
    let mut model = match args.initial_model.as_ref() {
        Some(path) => AzNnue::load(path)?,
        None => AzNnue::random(args.hidden, args.seed),
    };
    let evaluated_train = &data.train[..data.train.len().min(args.eval_train_samples)];
    println!(
        "experiment hidden={} seed={} lr={} batch={} policy_weight={} value_weight={} train_eval={}",
        model.hidden_size,
        args.seed,
        args.lr,
        args.batch,
        args.policy_weight,
        args.value_weight,
        evaluated_train.len()
    );
    let baseline_validation = px0_data::evaluate(&model, &data.validation);
    println!(
        "baseline train={:?} validation={:?}",
        px0_data::evaluate(&model, evaluated_train),
        baseline_validation
    );
    let mut rng = SplitMix64::new(args.seed);
    let weights = AzTrainLossWeights {
        value: args.value_weight,
        policy: args.policy_weight,
        short_value: 0.0,
    };
    let mut best_score = args.policy_weight as f64 * baseline_validation.policy_kl
        + args.value_weight as f64 * baseline_validation.value_ce;
    let best_path = format!(
        "{}.best.safetensors",
        args.output.trim_end_matches(".safetensors")
    );
    model.save(&best_path)?;
    println!("best epoch=0 score={best_score:.6} path={best_path}");
    let mut progress_score = best_score;
    let mut stalled = 0;
    for epoch in 1..=args.epochs {
        let lr = if args.lr_drop_after.is_some_and(|after| epoch > after) {
            args.lr * args.lr_drop_factor
        } else {
            args.lr
        };
        let start = Instant::now();
        train_samples_weighted(
            &mut model,
            &data.train,
            1,
            lr,
            args.batch,
            &mut rng,
            weights,
        )?;
        let train_seconds = start.elapsed().as_secs_f32();
        if epoch % args.eval_every.max(1) == 0 || epoch == args.epochs {
            let validation = px0_data::evaluate(&model, &data.validation);
            let score = args.policy_weight as f64 * validation.policy_kl
                + args.value_weight as f64 * validation.value_ce;
            let epoch_path = format!(
                "{}.epoch-{epoch}.safetensors",
                args.output.trim_end_matches(".safetensors")
            );
            model.save(&epoch_path)?;
            if score < best_score {
                best_score = score;
                model.save(&best_path)?;
                println!("best epoch={epoch} score={score:.6} path={best_path}");
            }
            println!(
                "epoch={epoch} lr={lr:.6} train_seconds={train_seconds:.3} train={:?} validation={:?}",
                px0_data::evaluate(&model, evaluated_train),
                validation
            );
            if score < progress_score * (1.0 - args.min_improvement) {
                progress_score = score;
                stalled = 0;
            } else {
                stalled += 1;
            }
            if args.early_stop_patience > 0 && stalled >= args.early_stop_patience {
                println!("early_stop epoch={epoch} stalled={stalled} best_score={best_score:.6}");
                break;
            }
        }
    }
    model.save(&args.output)?;
    println!("saved {}", args.output);
    Ok(())
}
