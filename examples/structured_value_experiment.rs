//! 小规模结构消融：固定数据、初始化和训练预算，输出留出指标及测量局面。
use chineseai::az::{AzNnue, SplitMix64, position_for_training_sample, px0_data, train_samples};
use std::{env, fs, path::Path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    assert_eq!(args.len(), 6, "DATA INPUT_OR_- OUTPUT EPOCHS GAMES");
    let epochs = args[4].parse()?;
    let games = args[5].parse()?;
    let data = px0_data::load(Path::new(&args[1]), games, 32)?;
    let mut model = if args[2] == "-" {
        AzNnue::random(96, 20261019)
    } else {
        AzNnue::load(&args[2])?
    };
    eprintln!(
        "train={} validation={} epochs={epochs}",
        data.train.len(),
        data.validation.len()
    );
    train_samples(
        &mut model,
        &data.train,
        epochs,
        0.0004,
        256,
        &mut SplitMix64::new(17),
    )?;
    model.save(&args[3])?;
    let metrics = px0_data::evaluate(&model, &data.validation);
    println!(
        "samples={} value_ce={:.6} q_rmse={:.6} policy_kl={:.6} top1={:.6}",
        metrics.samples, metrics.value_ce, metrics.q_rmse, metrics.policy_kl, metrics.top1
    );
    let fens: Vec<_> = data
        .validation
        .iter()
        .step_by((data.validation.len() / 64).max(1))
        .take(64)
        .filter_map(position_for_training_sample)
        .map(|p| p.to_fen())
        .collect();
    fs::write(format!("{}.fens", args[3]), fens.join("\n"))?;
    Ok(())
}
