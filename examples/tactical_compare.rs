//! 相同时间预算下比较 MCTS 与内部战术延伸；只加载一次模型。
use chineseai::{
    az::{AzNnue, AzSearchControl, AzSearchLimits, alphazero_search_with_rules_controlled},
    xiangqi::{Move, Position},
};
use std::{
    sync::{Arc, atomic::AtomicBool},
    time::{Duration, Instant},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model = AzNnue::load(
        args.get(1)
            .map(String::as_str)
            .unwrap_or("best.safetensors"),
    )?;
    let nodes = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(4096);
    let fixtures = [
        (
            "blindspot",
            Position::from_fen(
                "Cn1akab2/5R3/2n1b4/p2Rp1P1p/2p3r2/5N3/P1c1P4/4B4/9/1r1AKAB2 b - - 0 1",
            )?,
        ),
        ("startpos", Position::startpos()),
        (
            "checked",
            Position::from_fen("4k4/9/4R4/9/9/9/9/9/9/4K4 b")?,
        ),
    ];
    println!(
        "fixture,tactical_nodes_per_leaf,budget_ms,elapsed_ms,simulations,best,q,f9e8_q,f9e8_visits,tactical_nodes,completed,aborted"
    );
    for (label, p) in fixtures {
        for ms in [100, 500, 1000] {
            for budget in [0, nodes] {
                model.tactical_search_nodes = budget;
                let start = Instant::now();
                let control = AzSearchControl::new(
                    Arc::new(AtomicBool::new(false)),
                    Some(start + Duration::from_millis(ms)),
                );
                let result = alphazero_search_with_rules_controlled(
                    &p,
                    None,
                    None,
                    &model,
                    AzSearchLimits {
                        simulations: 1_000_000,
                        ..AzSearchLimits::default()
                    },
                    Some(&control),
                );
                let bad = result
                    .candidates
                    .iter()
                    .find(|c| c.mv == Move::from_uci("f9e8").unwrap());
                println!(
                    "{label},{budget},{ms},{:.3},{},{},{:.5},{},{},{},{},{}",
                    start.elapsed().as_secs_f64() * 1000.0,
                    result.simulations,
                    result.best_move.map(|m| m.to_uci()).unwrap_or_default(),
                    result.value_q,
                    bad.map(|c| format!("{:.5}", c.q)).unwrap_or_default(),
                    bad.map(|c| c.visits.to_string()).unwrap_or_default(),
                    result.tactical_nodes,
                    result.tactical_completed,
                    result.tactical_aborted
                );
            }
        }
    }
    Ok(())
}
