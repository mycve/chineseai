use chineseai::az::{AzNnue, AzNnueArch};
use clap::Parser;
use std::path::PathBuf;

#[cfg(all(target_os = "linux", not(target_env = "musl")))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(target_os = "windows")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser)]
#[command(about = "验证新特征的实际CPU前向、增量和完整MCTS成本")]
struct Args {
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long)]
    save_model: Option<PathBuf>,
    #[arg(long, default_value = "NNUE_FEATURE_BENCHMARK.csv")]
    output: PathBuf,
    #[arg(long, default_value_t = 100)]
    repeat: usize,
    #[arg(long, default_value_t = 512)]
    simulations: usize,
    #[arg(long, default_value_t = 64)]
    positions: usize,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let model = if let Some(path) = args.model {
        AzNnue::load(path)?
    } else {
        AzNnue::random_with_arch(AzNnueArch::default(), 17)
    };
    if let Some(path) = args.save_model {
        model.save(path)?;
    }
    println!(
        "{}",
        chineseai::az::feature_audit::write_feature_audit(
            &model,
            &args.output,
            args.repeat,
            args.simulations,
            args.positions
        )?
    );
    Ok(())
}
