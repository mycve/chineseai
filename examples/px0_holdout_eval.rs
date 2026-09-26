use chineseai::az::{AzNnue, outputs_for_training_sample, px0_data};
use clap::Parser;
use std::{
    collections::BTreeMap,
    fs::OpenOptions,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Parser)]
struct Args {
    /// 待比较网络文件，按给定顺序评测。
    #[arg(required = true)]
    models: Vec<String>,
    #[arg(long, default_value = "data/data.bin")]
    data: String,
    #[arg(long, default_value_t = 1024)]
    games: usize,
    #[arg(long, default_value_t = 20261001)]
    seed: u64,
    /// 创建新的逐局指标TSV，禁止覆盖已有结果。
    #[arg(long)]
    per_game_output: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let mut per_game = args
        .per_game_output
        .as_ref()
        .map(|path| {
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .map(BufWriter::new)
        })
        .transpose()?;
    if let Some(output) = per_game.as_mut() {
        writeln!(
            output,
            "model\tgame_id\tsamples\tq_error_sq_sum\tvalue_ce_sum\tpolicy_kl_sum"
        )?;
    }
    let started = Instant::now();
    let dataset = px0_data::load_holdout_reservoir(Path::new(&args.data), args.games, args.seed)?;
    println!(
        "holdout seed={} games={} samples={} deleted={} load_seconds={:.3}",
        args.seed,
        dataset.games,
        dataset.validation.len(),
        dataset.deleted,
        started.elapsed().as_secs_f64()
    );
    assert!(dataset.train.is_empty());
    for path in &args.models {
        let model = AzNnue::load(path)?;
        let started = Instant::now();
        println!(
            "model={path} metrics={:?} seconds={:.3}",
            px0_data::evaluate(&model, &dataset.validation),
            started.elapsed().as_secs_f64()
        );
        if let Some(output) = per_game.as_mut() {
            let mut games = BTreeMap::<u64, (usize, f64, f64, f64)>::new();
            for sample in &dataset.validation {
                let (wdl, _, logits) =
                    outputs_for_training_sample(&model, sample).ok_or("invalid holdout sample")?;
                if logits.len() != sample.policy.len() || logits.len() != sample.move_indices.len()
                {
                    return Err("holdout policy length mismatch".into());
                }
                let target_q = sample.value_wdl[0] as f64 - sample.value_wdl[2] as f64;
                let q = wdl[0] as f64 - wdl[2] as f64;
                let value_ce = -sample
                    .value_wdl
                    .iter()
                    .zip(wdl)
                    .map(|(&t, p)| t as f64 * (p as f64).max(1e-12).ln())
                    .sum::<f64>();
                let max = logits
                    .iter()
                    .map(|&x| x as f64)
                    .fold(f64::NEG_INFINITY, f64::max);
                let log_sum = logits
                    .iter()
                    .map(|&x| (x as f64 - max).exp())
                    .sum::<f64>()
                    .ln()
                    + max;
                let policy_kl = sample
                    .policy
                    .iter()
                    .zip(&logits)
                    .filter(|(t, _)| **t > 0.0)
                    .map(|(&t, &logit)| t as f64 * ((t as f64).ln() - logit as f64 + log_sum))
                    .sum::<f64>();
                let sums = games.entry(sample.meta.game_id).or_default();
                sums.0 += 1;
                sums.1 += (q - target_q).powi(2);
                sums.2 += value_ce;
                sums.3 += policy_kl;
            }
            for (game, (samples, q_error, value_ce, policy_kl)) in games {
                writeln!(
                    output,
                    "{path}\t{game}\t{samples}\t{q_error:.17}\t{value_ce:.17}\t{policy_kl:.17}"
                )?;
            }
        }
    }
    if let Some(output) = per_game.as_mut() {
        output.flush()?;
    }
    Ok(())
}
