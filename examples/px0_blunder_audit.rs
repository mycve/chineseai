//! 复盘已保存对局，在完整规则历史上比较真实落子与提高模拟数后的落子。
use chineseai::{
    az::{AzNnue, AzSearchLimits, alphazero_search_trace_with_rules, alphazero_search_with_rules},
    xiangqi::{Color, Position},
};
use clap::Parser;
use serde::Deserialize;
use std::fs;
#[path = "support/pikafish.rs"]
mod pikafish;
use pikafish::Engine;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: String,
    #[arg(long, default_value = "tmp/px0-65536-vs-pikafish-trajectories.log")]
    trajectories: String,
    /// 固定病例来自旧网络实际对局，禁止根据新网络重新选择失误。
    #[arg(long, default_value = "tmp/px0-blunder-cases.toml")]
    cases: String,
    #[arg(long, default_value = "tools/pikafish-avx2.exe")]
    pikafish: String,
    #[arg(long, value_delimiter = ',', default_value = "400,1600")]
    simulations: Vec<usize>,
}

#[derive(Deserialize)]
struct Cases {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    game: usize,
    ply: usize,
    fen: String,
    played: String,
}

fn command(fen: &str, moves: &[&str]) -> String {
    format!(
        "position fen {fen}{}",
        if moves.is_empty() {
            String::new()
        } else {
            format!(" moves {}", moves.join(" "))
        }
    )
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let model = AzNnue::load(&args.model)?;
    let text = fs::read_to_string(&args.trajectories)?;
    let cases: Cases = toml::from_str(&fs::read_to_string(&args.cases)?)?;
    let mut engine = Engine::new(&args.pikafish)?;
    let mut evaluated = 0;
    println!(
        "audit model={} fixed_cases={} simulations={:?}",
        args.model,
        cases.cases.len(),
        args.simulations
    );
    for line in text
        .lines()
        .filter(|line| line.starts_with("vs-pikafish-final:"))
    {
        let game = line
            .split("game=")
            .nth(1)
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .parse::<usize>()?;
        let Some(case) = cases.cases.iter().find(|case| case.game == game) else {
            continue;
        };
        let chinese = if line.contains("chinese=red") {
            Color::Red
        } else {
            Color::Black
        };
        let trajectory = line.split(" position fen ").nth(1).unwrap();
        let (fen, moves) = trajectory.split_once(" moves ").unwrap();
        let moves = moves.split_whitespace().collect::<Vec<_>>();
        let mut position = Position::from_fen(fen)?;
        let mut history = position.initial_rule_history();
        if case.ply >= moves.len() {
            return Err("cached ply exceeds trajectory".into());
        }
        for &played in &moves[..case.ply] {
            let mv = position
                .parse_uci_move(played)
                .ok_or("invalid trajectory move")?;
            history.push(position.rule_history_entry_after_move(mv));
            position.make_move(mv);
        }
        if position.to_fen() != case.fen
            || moves[case.ply] != case.played
            || position.side_to_move() != chinese
        {
            return Err(
                format!("fixed case does not match original trajectory: game={game}").into(),
            );
        }
        evaluated += 1;
        let ply = case.ply;
        let played = case.played.as_str();
        let cmd = command(fen, &moves[..ply]);
        let best = engine.score(&cmd, 10, None)?;
        let actual = engine.score(&cmd, 10, Some(played))?;
        println!(
            "CASE game={game} ply={ply} side={chinese:?} fen=\"{}\" played={played} pf_best={best:?} pf_played={actual:?}",
            position.to_fen()
        );
        for name in [played, best.best.as_str()] {
            let mut child = position.clone();
            let mv = child.parse_uci_move(name).ok_or("PF invalid move")?;
            let mut child_history = history.clone();
            child_history.push(child.rule_history_entry_after_move(mv));
            child.make_move(mv);
            let wdl = model.evaluate_wdl_with_rules(
                &child,
                &child_history,
                &child.legal_moves_with_rules(&child_history),
            );
            println!(
                "  immediate_child move={name} network_q_root_perspective={:.4} child_wdl={wdl:?}",
                wdl[2] - wdl[0]
            );
        }
        if game == 5 {
            let mut child = position.clone();
            let mut child_history = history.clone();
            for name in &actual.pv {
                let mv = child.parse_uci_move(name).ok_or("PF PV invalid move")?;
                if !child.legal_moves_with_rules(&child_history).contains(&mv) {
                    return Err("PF PV violates our rules".into());
                }
                child_history.push(child.rule_history_entry_after_move(mv));
                child.make_move(mv);
            }
            println!(
                "  mate_pv_check legal_replies={} outcome={:?}",
                child.legal_moves_with_rules(&child_history).len(),
                child.rule_outcome_with_history(&child_history)
            );
        }
        if game == 5 {
            for name in ["a9b9", "g5g4"] {
                let mut child = position.clone();
                let mv = child.parse_uci_move(name).ok_or("invalid branch move")?;
                let mut child_history = history.clone();
                child_history.push(child.rule_history_entry_after_move(mv));
                child.make_move(mv);
                let reply = alphazero_search_with_rules(
                    &child,
                    Some(child_history),
                    None,
                    &model,
                    AzSearchLimits {
                        simulations: 400,
                        ..AzSearchLimits::default()
                    },
                );
                let mut priors = reply.candidates.clone();
                priors.sort_by(|a, b| b.raw_prior.total_cmp(&a.raw_prior));
                println!(
                    "  mate_reply parent_move={name} selected={:?} winning_reply_prior_rank={:?} winning_reply_candidate={:?}",
                    reply.best_move,
                    priors
                        .iter()
                        .position(|c| c.mv.to_string() == "f4f9")
                        .map(|i| i + 1),
                    reply.candidates.iter().find(|c| c.mv.to_string() == "f4f9")
                );
            }
        }
        for &simulations in &args.simulations {
            let search = alphazero_search_with_rules(
                &position,
                Some(history.clone()),
                None,
                &model,
                AzSearchLimits {
                    simulations,
                    ..AzSearchLimits::default()
                },
            );
            let selected = search.best_move.unwrap().to_string();
            let score = engine.score(&cmd, 10, Some(&selected))?;
            let mut order = search.candidates.clone();
            order.sort_by(|a, b| b.raw_prior.total_cmp(&a.raw_prior));
            let best_rank = order
                .iter()
                .position(|c| c.mv.to_string() == best.best)
                .map(|i| i + 1);
            let candidate = search
                .candidates
                .iter()
                .find(|c| c.mv.to_string() == best.best);
            let selected_candidate = search
                .candidates
                .iter()
                .find(|c| c.mv.to_string() == selected);
            println!(
                "  sims={simulations} used={} selected={selected} pf_selected={score:?} network_wdl={:?} search_q={:.4} pf_best_prior_rank={best_rank:?} pf_best_candidate={candidate:?} selected_candidate={selected_candidate:?} depth={}",
                search.simulations,
                search.network_value_wdl,
                search.value_q,
                search.search_depth_max
            );
            if game == 5 && simulations == 6400 {
                let (_, trace) = alphazero_search_trace_with_rules(
                    &position,
                    Some(history.clone()),
                    None,
                    &model,
                    AzSearchLimits {
                        simulations,
                        ..AzSearchLimits::default()
                    },
                    search.best_move.unwrap(),
                );
                println!("  selected_branch_trace={:?}", &trace[..trace.len().min(3)]);
            }
        }
    }
    if evaluated != cases.cases.len() {
        return Err("fixed cases missing from trajectories".into());
    }
    Ok(())
}
