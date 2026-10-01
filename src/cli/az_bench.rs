use crate::cli::args::*;
use crate::cli::az_search::{fixed_az_search_limits, parse_position};
use chineseai::az::{AzNnue, alphazero_search};

pub(crate) fn run(cmd: AzBenchArgs) {
    let model_path = cmd.model;
    let simulations = cmd.simulations.max(1);
    let repeat = cmd.repeat.max(1);
    let cpuct = cmd.cpuct.max(0.0);
    let fen = cmd.fen.join(" ");
    let position = parse_position(&fen);
    let model = AzNnue::load(&model_path).unwrap_or_else(|err| {
        panic!("failed to load `{model_path}`: {err}");
    });

    let _ = alphazero_search(
        &position,
        &model,
        fixed_az_search_limits(simulations, 0, cpuct, cpuct, 0, 1.4),
    );

    let started = std::time::Instant::now();
    let mut total_sims = 0usize;
    let mut best_move = None;
    for iteration in 0..repeat {
        let result = alphazero_search(
            &position,
            &model,
            fixed_az_search_limits(simulations, iteration as u64, cpuct, cpuct, 0, 1.4),
        );
        total_sims += result.simulations;
        best_move = result.best_move;
    }
    let elapsed = started.elapsed();
    let elapsed_secs = elapsed.as_secs_f64().max(f64::EPSILON);
    println!("bench        : fixed-search");
    println!("model        : {model_path}");
    println!("arch         : hidden={}", model.arch.hidden_size);
    println!("fen          : {}", position.to_fen());
    println!("sims/search  : {simulations}");
    println!("repeat       : {repeat}");
    println!("search       : alphazero");
    println!("simd         : {}", chineseai::az::inference_simd_backend());
    println!("cpuct        : {cpuct}");
    println!("total_sims   : {total_sims}");
    println!("elapsed_ms   : {:.3}", elapsed.as_secs_f64() * 1000.0);
    println!(
        "ms/search    : {:.3}",
        elapsed.as_secs_f64() * 1000.0 / repeat as f64
    );
    println!("sims/sec     : {:.0}", total_sims as f64 / elapsed_secs);
    println!(
        "last_bestmove: {}",
        best_move
            .map(|mv| mv.to_string())
            .unwrap_or_else(|| "(none)".into())
    );
}
