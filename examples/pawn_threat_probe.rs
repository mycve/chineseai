//! 检查兵车联手的一步杀威胁：区分原网络先验与搜索后的选择。
use chineseai::{
    az::{AzMatingGoal, AzNnue, AzSearchLimits, alphazero_search_with_rules},
    xiangqi::{MateKind, Position},
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::{error::Error, fs, io::Write, path::PathBuf};

#[derive(Parser)]
struct Args {
    model: PathBuf,
    #[arg(long, default_value = "eval/pawn-coordination.toml")]
    cases: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 20000)]
    simulations: usize,
}
#[derive(Deserialize)]
struct Suite {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    fen: String,
}
#[derive(Serialize)]
struct Row {
    name: String,
    fen: String,
    legal_moves: usize,
    allows_mate_in_one: Vec<String>,
    network_wdl: [f32; 3],
    unsafe_prior_mass: f32,
    network_top_move: String,
    network_top_allows_mate: bool,
    search_move: String,
    search_allows_mate: bool,
    search_wdl: [f32; 3],
    simulations: usize,
}
#[derive(Serialize)]
struct Report {
    model: String,
    cases: Vec<Row>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let model = AzNnue::load(&args.model)?;
    let suite: Suite = toml::from_str(&fs::read_to_string(args.cases)?)?;
    let mut rows = Vec::new();
    for case in suite.cases {
        let position = Position::from_fen(&case.fen)?;
        let history = position.initial_rule_history();
        let legal = position.legal_moves_with_rules(&history);
        if legal.is_empty() {
            return Err("构造局面已经终局".into());
        }
        let mut unsafe_moves = Vec::new();
        for &mv in &legal {
            let mut next = position.clone();
            let mover = position.side_to_move();
            let captured = position.piece_at(mv.to as usize);
            next.make_move(mv);
            let mut replies_history = history.clone();
            replies_history.push(next.rule_history_entry_after_moved(mover, mv, captured));
            let attacker = next.side_to_move();
            if next
                .legal_moves_with_rules(&replies_history)
                .into_iter()
                .any(|reply| {
                    let mut terminal = next.clone();
                    terminal.make_move(reply);
                    terminal.classify_mate().is_some_and(|mate| {
                        mate.kind == MateKind::Checkmate && mate.winner == attacker
                    })
                })
            {
                unsafe_moves.push(mv);
            }
        }
        let (_, wdl, logits) =
            model.evaluate_with_goal(&position, &history, &legal, AzMatingGoal::NONE);
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let priors = logits
            .iter()
            .map(|&value| (value - maximum).exp())
            .collect::<Vec<_>>();
        let sum: f32 = priors.iter().sum();
        let top = (0..legal.len())
            .max_by(|&a, &b| priors[a].total_cmp(&priors[b]))
            .unwrap();
        let unsafe_prior_mass = legal
            .iter()
            .zip(&priors)
            .filter(|(mv, _)| unsafe_moves.contains(mv))
            .map(|(_, p)| p)
            .sum::<f32>()
            / sum;
        let result = alphazero_search_with_rules(
            &position,
            Some(history),
            Some(legal.clone()),
            &model,
            AzSearchLimits {
                simulations: args.simulations,
                ..Default::default()
            },
        );
        let selected = result.best_move.ok_or("搜索未返回走法")?;
        let row = Row {
            name: case.name,
            fen: case.fen,
            legal_moves: legal.len(),
            allows_mate_in_one: unsafe_moves.iter().map(|mv| mv.to_uci()).collect(),
            network_wdl: wdl,
            unsafe_prior_mass,
            network_top_move: legal[top].to_uci(),
            network_top_allows_mate: unsafe_moves.contains(&legal[top]),
            search_move: selected.to_uci(),
            search_allows_mate: unsafe_moves.contains(&selected),
            search_wdl: result.value_wdl,
            simulations: result.simulations,
        };
        println!(
            "{}: legal={} allows_mate={} unsafe_prior={:.1}% raw={}/{} search={}/{}",
            row.name,
            row.legal_moves,
            row.allows_mate_in_one.len(),
            row.unsafe_prior_mass * 100.0,
            row.network_top_move,
            row.network_top_allows_mate,
            row.search_move,
            row.search_allows_mate
        );
        rows.push(row);
    }
    let output = toml::to_string_pretty(&Report {
        model: args.model.display().to_string(),
        cases: rows,
    })?;
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(args.output)?
        .write_all(output.as_bytes())?;
    Ok(())
}
