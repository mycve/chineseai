//! 固定节点、固定书局面、逐局交换红黑的小规模架构对比。
use chineseai::{
    az::{AzArenaConfig, AzArenaReport, AzNnue, AzSearchLimits, play_arena_games_from_snapshots},
    pikafish::opening_book::Px0OpeningBook,
};
use std::{env, fs, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    assert!(
        (6..=7).contains(&args.len()),
        "CANDIDATE BASELINE BOOK PAIRS NODES [START_PAIR]"
    );
    let candidate = AzNnue::load(&args[1])?;
    let baseline = AzNnue::load(&args[2])?;
    let pairs: usize = args[4].parse()?;
    let nodes = args[5].parse()?;
    let offset: usize = args.get(6).map(|s| s.parse()).transpose()?.unwrap_or(0);
    let snapshots = Px0OpeningBook::load(&args[3], 20261019)?.next_batch(offset + pairs, 0)?;
    fs::write(
        format!("{}.arena.fens", args[1]),
        snapshots
            .iter()
            .map(|s| s.position.to_fen())
            .collect::<Vec<_>>()
            .join("\n"),
    )?;
    let limits = AzSearchLimits::default();
    let mut total = AzArenaReport::default();
    let started = Instant::now();
    for pair in 0..pairs {
        let result = play_arena_games_from_snapshots(
            &candidate,
            &baseline,
            &snapshots,
            AzArenaConfig {
                simulations: nodes,
                max_plies: 400,
                rule60_max_ply: Some(120),
                games_as_red: 1,
                games_as_black: 1,
                start_index: offset + pair,
                seed: 20261019,
                cpuct: limits.cpuct,
                cpuct_at_root: limits.cpuct_at_root,
                cpuct_base: limits.cpuct_base,
                cpuct_factor: limits.cpuct_factor,
                cpuct_base_at_root: limits.cpuct_base_at_root,
                cpuct_factor_at_root: limits.cpuct_factor_at_root,
                fpu_value: limits.fpu_value,
                fpu_value_at_root: limits.fpu_value_at_root,
                fpu_absolute_at_root: limits.fpu_absolute_at_root,
                minimum_kldgain_per_node: 0.0,
                draw_score: 0.0,
                policy_softmax_temp: limits.policy_softmax_temp,
            },
        );
        total.add_assign(&result);
        println!(
            "pairs={} W={} D={} L={} score={:.4} CI95=[{:.4},{:.4}] seconds={:.1}",
            pair + 1,
            total.wins,
            total.draws,
            total.losses,
            total.score_rate(),
            total.score_rate_lower_bound(1.96).max(0.0),
            total.score_rate_upper_bound(1.96).min(1.0),
            started.elapsed().as_secs_f64()
        );
    }
    Ok(())
}
