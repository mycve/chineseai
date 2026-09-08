//! 用固定模型和 Pikafish 标签比较根探索参数；不更新模型。
use chineseai::az::{AzNnue, AzSearchLimits, alphazero_search};
use chineseai::xiangqi::Position;
use clap::Parser;
use rayon::prelude::*;
use rusqlite::{Connection, OpenFlags};
use std::{fs, io::Write, path::PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    sqlite: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 128)]
    positions: usize,
    #[arg(long, default_value_t = 4)]
    seeds: usize,
    #[arg(long, default_value_t = 400)]
    simulations: usize,
    #[arg(long, default_value_t = 4)]
    workers: usize,
    #[arg(long, default_value_t = 0)]
    skip_per_group: usize,
    #[arg(long, value_delimiter = ',')]
    arms: Vec<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let model = AzNnue::load(&args.model)?;
    let db = Connection::open_with_flags(&args.sqlite, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    // 固定打散 ID，并让约一半局面来自杀棋标签，避免只测连续开局。
    let mut query = db.prepare("SELECT id, fen, bestmove, best_mate FROM pikafish_labels ORDER BY (id * 2654435761) % 4294967296")?;
    let rows = query.query_map([], |r| {
        Ok((
            r.get::<_, u64>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, String>(2)?,
            r.get::<_, Option<i32>>(3)?,
        ))
    })?;
    let mut positions = Vec::new();
    let mut counts = [0; 2];
    let mut skipped = [0; 2];
    for row in rows {
        let (id, fen, best, mate) = row?;
        let group = usize::from(mate.is_some());
        let quota = args.positions / 2 + usize::from(group == 0) * (args.positions % 2);
        if counts[group] >= quota {
            continue;
        }
        let position = Position::from_fen(&fen)?;
        let Some(best) = position.parse_uci_move(&best) else {
            continue;
        };
        let history = position.initial_rule_history();
        if position.rule_outcome_with_history(&history).is_some()
            || !position.legal_moves_with_rules(&history).contains(&best)
        {
            continue;
        }
        if skipped[group] < args.skip_per_group {
            skipped[group] += 1;
            continue;
        }
        positions.push((id, position, best, group));
        counts[group] += 1;
        if positions.len() == args.positions {
            break;
        }
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.workers)
        .build()?;
    fs::create_dir_all(&args.output)?;
    let mut details = fs::File::create(args.output.join("details.csv"))?;
    writeln!(
        details,
        "arm,id,split,mate,seed,label_visits,label_prior,top1,depth"
    )?;
    let mut summary = fs::File::create(args.output.join("summary.csv"))?;
    writeln!(
        summary,
        "arm,split,n,top1_pct,unvisited_pct,label_visit_pct,mate_top1_pct,mate_unvisited_pct"
    )?;
    // alpha, 总浓度, 噪声比例, FPU, 根 FPU, 策略温度。
    let arms = [
        ("baseline", 0.12, 0.0, 0.08, 0.20, 0.10, 1.15),
        ("proposed_clamped", 0.20, 0.0, 0.25, 0.0, 0.0, 1.30),
        ("fixed_corrected", 0.20, 0.0, 0.25, 0.15, 0.05, 1.30),
        ("dynamic4", 0.0, 4.0, 0.25, 0.15, 0.05, 1.30),
        ("dynamic8", 0.0, 8.0, 0.25, 0.15, 0.05, 1.30),
        ("dynamic12", 0.0, 12.0, 0.25, 0.15, 0.05, 1.30),
        ("dynamic8_noise15", 0.0, 8.0, 0.15, 0.15, 0.05, 1.30),
        ("dynamic8_noise08", 0.0, 8.0, 0.08, 0.15, 0.05, 1.30),
        ("dynamic8_baseline", 0.0, 8.0, 0.08, 0.20, 0.10, 1.15),
        ("dynamic12_baseline", 0.0, 12.0, 0.08, 0.20, 0.10, 1.15),
        ("dynamic8_temp115", 0.0, 8.0, 0.08, 0.15, 0.05, 1.15),
        ("dynamic8_fpu0", 0.0, 8.0, 0.08, 0.0, 0.0, 1.15),
        ("dynamic8_cpuct25", 0.0, 8.0, 0.08, 0.15, 0.05, 1.15),
        ("dynamic8_cpuct30", 0.0, 8.0, 0.08, 0.15, 0.05, 1.15),
    ];
    for (name, alpha, total, noise, fpu, root_fpu, temp) in arms {
        if !args.arms.is_empty() && !args.arms.iter().any(|arm| arm == name) {
            continue;
        }
        let results = pool.install(|| {
            positions
                .par_iter()
                .enumerate()
                .flat_map_iter(|(index, (id, position, best, mate))| {
                    let model = &model;
                    (0..args.seeds).map(move |seed| {
                        let result = alphazero_search(
                            position,
                            model,
                            AzSearchLimits {
                                simulations: args.simulations,
                                seed: 20260420 + seed as u64,
                                cpuct: 0.9,
                                cpuct_at_root: match name {
                                    "dynamic8_cpuct25" => 2.5,
                                    "dynamic8_cpuct30" => 3.0,
                                    _ => 2.0,
                                },
                                cpuct_factor: 1.5,
                                cpuct_factor_at_root: 1.5,
                                root_dirichlet_total_concentration: if total > 0.0 {
                                    total
                                } else {
                                    alpha
                                        * position
                                            .legal_moves_with_rules(
                                                &position.initial_rule_history(),
                                            )
                                            .len() as f32
                                },
                                root_exploration_fraction: noise,
                                fpu_value: fpu,
                                fpu_value_at_root: root_fpu,
                                policy_softmax_temp: temp,
                                ..AzSearchLimits::default()
                            },
                        );
                        let label = result
                            .candidates
                            .iter()
                            .find(|c| c.mv == *best)
                            .expect("legal label must be reported");
                        (
                            *id,
                            index % 2,
                            *mate,
                            seed,
                            label.visits,
                            label.raw_prior,
                            result.best_move == Some(*best),
                            result.search_depth_avg,
                        )
                    })
                })
                .collect::<Vec<_>>()
        });
        for (id, split, mate, seed, visits, prior, hit, depth) in &results {
            writeln!(
                details,
                "{name},{id},{split},{mate},{seed},{visits},{prior},{hit},{depth}"
            )?;
        }
        for split in 0..2 {
            let rows: Vec<_> = results.iter().filter(|r| r.1 == split).collect();
            let n = rows.len() as f64;
            let mates: Vec<_> = rows.iter().filter(|r| r.2 == 1).collect();
            let mn = mates.len() as f64;
            let line = format!(
                "{name},{split},{},{:.3},{:.3},{:.3},{:.3},{:.3}",
                rows.len(),
                rows.iter().filter(|r| r.6).count() as f64 / n * 100.0,
                rows.iter().filter(|r| r.4 == 0).count() as f64 / n * 100.0,
                rows.iter().map(|r| r.4 as f64).sum::<f64>() / n / args.simulations as f64 * 100.0,
                mates.iter().filter(|r| r.6).count() as f64 / mn * 100.0,
                mates.iter().filter(|r| r.4 == 0).count() as f64 / mn * 100.0
            );
            writeln!(summary, "{line}")?;
            println!("{line}");
        }
        summary.flush()?;
        details.flush()?;
    }
    Ok(())
}
