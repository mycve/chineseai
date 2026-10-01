//! 固定权重和局面，测量预提取特征评估及单 worker 搜索吞吐。
use chineseai::{
    ab::{
        AbSearchLimits,
        pikafish_candle::{PikafishCpuCache, PikafishExample, PikafishModel},
        search_pikafish_model,
    },
    xiangqi::Position,
};
use clap::Parser;
use std::{
    hint::black_box,
    sync::{Arc, Barrier},
    time::Instant,
};
#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 1)]
    workers: usize,
    #[arg(long, default_value_t = 200)]
    iterations: usize,
    #[arg(long, default_value = "target/pikafish-perf.safetensors")]
    model: std::path::PathBuf,
    #[arg(long, default_value_t = 8192)]
    nodes: usize,
    #[arg(long, default_value_t = 64)]
    max_depth: usize,
    #[arg(long, default_value_t = 5)]
    search_repeats: usize,
    /// 每行一个 FEN；用于同局面对照成熟引擎。
    #[arg(long)]
    positions: Option<std::path::PathBuf>,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    assert!(args.workers > 0 && args.iterations > 0 && args.nodes > 0 && args.search_repeats > 0);
    let model = PikafishModel::new(&candle_core::Device::Cpu)?;
    if args.model.exists() {
        model.load(&args.model)?;
    } else {
        model.save(&args.model)?;
    }
    let model = Arc::new(model.cpu_snapshot()?);
    let mut positions = vec![Position::startpos()];
    for i in 0..7 {
        let mut position = positions.last().unwrap().clone();
        let moves = position.legal_moves();
        position.make_move(moves[(i * 7 + 3) % moves.len()]);
        positions.push(position);
    }
    if let Some(path) = args.positions {
        positions = std::fs::read_to_string(path)?
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(Position::from_fen)
            .collect::<Result<Vec<_>, _>>()?;
        assert!(!positions.is_empty());
    }
    let examples: Vec<_> = positions
        .iter()
        .map(|p| PikafishExample::from_position(p).unwrap())
        .collect();
    let mut cache = PikafishCpuCache::default();
    for example in &examples {
        black_box(model.evaluate_example(example, &mut cache)?);
    }
    let barrier = Barrier::new(args.workers + 1);
    let start = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for worker in 0..args.workers {
            let model = &model;
            let examples = &examples;
            let barrier = &barrier;
            handles.push(scope.spawn(move || {
                let mut cache = PikafishCpuCache::default();
                barrier.wait();
                let mut sum = 0.0f32;
                for i in 0..args.iterations {
                    sum += model
                        .evaluate_example(&examples[(i + worker) % examples.len()], &mut cache)
                        .unwrap();
                }
                black_box(sum);
                chineseai::profile::flush_thread();
            }));
        }
        let start = Instant::now();
        barrier.wait();
        for handle in handles {
            handle.join().unwrap();
        }
        start
    });
    let eval_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let mut nodes = 0;
    let mut searches = Vec::new();
    for position in positions
        .iter()
        .cycle()
        .take(positions.len() * args.search_repeats)
    {
        let result = search_pikafish_model(
            position,
            &position.initial_rule_history(),
            &model,
            AbSearchLimits {
                nodes: args.nodes,
                max_depth: args.max_depth,
            },
        )?;
        nodes += result.nodes;
        searches.push((result.best_move.map(|m| m.to_uci()), result.value_q));
    }
    let search_seconds = start.elapsed().as_secs_f64();
    println!(
        "{}",
        serde_json::json!({"workers":args.workers,"evals":args.workers*args.iterations,"eval_seconds":eval_seconds,"evals_per_second":args.workers as f64*args.iterations as f64/eval_seconds,"search_seconds":search_seconds,"search_nodes":nodes,"search_nps":nodes as f64/search_seconds,"node_budget":args.nodes,"searches":searches})
    );
    chineseai::profile::print_report();
    Ok(())
}
