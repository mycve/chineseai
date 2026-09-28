//! 使用 Px0 整局留出数据预训练，导出可直接续训的网络和优化器状态。
use chineseai::az::{
    AzNnue, AzTrainLossWeights, SplitMix64, px0_data, train_samples_weighted_shared,
};
use clap::Parser;
use std::{path::Path, sync::Arc, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(default_value = "data/data.bin")]
    data: String,
    #[arg(long, default_value_t = 16_384)]
    games: usize,
    #[arg(long, default_value_t = 1_024)]
    validation_games: usize,
    #[arg(long, default_value_t = 128)]
    hidden: usize,
    #[arg(long, default_value_t = 12)]
    epochs: usize,
    #[arg(long, default_value_t = 8_192)]
    validation_samples: usize,
    #[arg(long, default_value_t = 20260927)]
    seed: u64,
    #[arg(long, default_value = "data/px0-pretrained.safetensors")]
    output: String,
    /// 只验证输出网络与 SGD 状态可以完整恢复。
    #[arg(long)]
    verify_only: bool,
}

fn save(model: &AzNnue, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    model.save(path)?;
    if !model.save_training_state(format!("{path}.sgd.safetensors"), 1)? {
        return Err("pretraining optimizer state is missing".into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.verify_only {
        return verify(&args.output);
    }
    if args.epochs == 0 || args.validation_samples == 0 {
        return Err("epochs and validation_samples must be positive".into());
    }
    let mut data = px0_data::load_reservoir(
        Path::new(&args.data),
        args.games,
        args.validation_games,
        args.seed,
    )?;
    data.validation.truncate(args.validation_samples);
    let train = Arc::new(data.train);
    println!(
        "dataset games={} train={} validation={} deleted={} hidden={} batch=2048 base_lr=0.02 target=best_search_wdl",
        data.games,
        train.len(),
        data.validation.len(),
        data.deleted,
        args.hidden
    );
    let mut model = AzNnue::random(args.hidden, args.seed);
    let baseline = px0_data::evaluate(&model, &data.validation);
    println!("baseline {baseline:?}");
    model.set_training_holdout(data.validation.clone(), 0.02)?;
    let mut rng = SplitMix64::new(args.seed);
    let weights = AzTrainLossWeights {
        value: 1.0,
        policy: 1.0,
    };
    let mut best = f64::INFINITY;
    let mut best_epoch = 0;
    let mut best_step = 0;
    let mut stalled = 0;
    for epoch in 1..=args.epochs {
        let start = Instant::now();
        train_samples_weighted_shared(&mut model, train.clone(), 1, 0.02, 2048, &mut rng, weights)?;
        let metrics = px0_data::evaluate(&model, &data.validation);
        let score = metrics.policy_kl + metrics.value_ce;
        println!(
            "epoch={epoch} step={} seconds={:.1} lr={:.6} validation={metrics:?}",
            model.training_steps(),
            start.elapsed().as_secs_f32(),
            model.last_training_learning_rate().unwrap_or(0.0)
        );
        for check in model.take_training_checks() {
            println!(
                "gpu_check step={} policy_kl={:.6} value_ce={:.6} q_rmse={:.6}",
                check.step, check.policy_kl, check.value_loss, check.value_rmse
            );
        }
        save(
            &model,
            &format!(
                "{}.latest.safetensors",
                args.output.trim_end_matches(".safetensors")
            ),
        )?;
        if score < best {
            stalled = if score < best * 0.999 { 0 } else { stalled + 1 };
            best = score;
            best_epoch = epoch;
            best_step = model.training_steps();
            save(&model, &args.output)?;
            println!("selected epoch={best_epoch} step={best_step} score={best:.6}");
        } else {
            stalled += 1;
        }
        if stalled >= 3 {
            println!("stop: validation has not improved materially for three passes");
            break;
        }
    }
    println!(
        "ready model={} optimizer={}.sgd.safetensors selected_epoch={best_epoch} selected_step={best_step} score={best:.6}",
        args.output, args.output
    );
    verify(&args.output)
}

fn verify(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut model = AzNnue::load(path)?;
    model.restore_training_state(format!("{path}.sgd.safetensors"), 1, 0.02)?;
    println!(
        "verified model={path} hidden={} optimizer_step={} next_update=1",
        model.hidden_size,
        model.training_steps()
    );
    Ok(())
}
