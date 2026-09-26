use chineseai::az::{AzNnue, px0_data};
use clap::Parser;
use std::{path::Path, time::Instant};

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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
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
    }
    Ok(())
}
