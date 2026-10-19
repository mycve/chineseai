//! 同权重比较全量和节点增量搜索，报告估值、先验、访问数和最佳走法差异。
use chineseai::{
    az::{AzNnue, AzSearchLimits, alphazero_search},
    xiangqi::Position,
};
use std::{env, fs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    assert!(
        (4..=5).contains(&args.len()),
        "MODEL FENS NODES [king|threat]"
    );
    let mut full = AzNnue::load(&args[1])?;
    full.node_incremental = false;
    let mut incremental = full.clone();
    incremental.threat_incremental = args.get(4).is_none_or(|s| s == "threat");
    incremental.node_incremental = true;
    let nodes = args[3].parse()?;
    let mut different_best = 0;
    let mut different_visits = 0;
    let mut max_wdl = 0.0f32;
    let mut max_prior = 0.0f32;
    let mut max_q = 0.0f32;
    let mut count = 0;
    for fen in fs::read_to_string(&args[2])?.lines() {
        let position = Position::from_fen(fen)?;
        let limits = AzSearchLimits {
            simulations: nodes,
            seed: 20261019,
            ..Default::default()
        };
        let a = alphazero_search(&position, &full, limits);
        let b = alphazero_search(&position, &incremental, limits);
        if a.best_move != b.best_move {
            different_best += 1;
            println!("different best: {fen}");
        }
        max_q = max_q.max((a.value_q - b.value_q).abs());
        for (a, b) in a.network_value_wdl.iter().zip(b.network_value_wdl) {
            max_wdl = max_wdl.max((a - b).abs());
        }
        for a in &a.candidates {
            let b = b.candidates.iter().find(|b| a.mv == b.mv).unwrap();
            max_prior = max_prior.max((a.raw_prior - b.raw_prior).abs());
            if a.visits != b.visits {
                different_visits += 1;
            }
        }
        count += 1;
    }
    println!(
        "positions={count} different_best={different_best} different_candidate_visits={different_visits} max_root_wdl={max_wdl:.9} max_prior={max_prior:.9} max_search_q={max_q:.9}"
    );
    assert_eq!(different_best, 0);
    assert!(max_wdl < 1e-5 && max_prior < 1e-5);
    Ok(())
}
