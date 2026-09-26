//! 复盘已保存对局，在完整规则历史上比较真实落子与提高模拟数后的落子。
use chineseai::{
    az::{AzNnue, AzSearchLimits, alphazero_search_trace_with_rules, alphazero_search_with_rules},
    xiangqi::{Color, Position},
};
use clap::Parser;
use serde::{Deserialize, Serialize};
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
    #[arg(long, default_value_t = 10)]
    depth: usize,
    #[arg(long)]
    trace: bool,
    /// 从真实对局前若干步选出每局损失最大的落子，保存为固定病例。
    #[arg(long)]
    discover_output: Option<String>,
    #[arg(long, default_value_t = 30)]
    discover_plies: usize,
    /// 优先检查尚可防守的局面，避免在已败局中比较两个败着。
    #[arg(long, default_value_t = -200)]
    discover_min_best_cp: i32,
    #[arg(long)]
    search_config: Option<String>,
    #[arg(long)]
    minimum_kldgain_per_node: Option<f32>,
}

#[derive(Deserialize, Serialize)]
struct Cases {
    cases: Vec<Case>,
}
#[derive(Deserialize, Serialize)]
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
    let mut base_limits = AzSearchLimits::default();
    if let Some(path) = args.search_config.as_ref() {
        let config: toml::Value = toml::from_str(&fs::read_to_string(path)?)?;
        macro_rules! load_fields {
            ($($field:ident),*) => { $(if let Some(value) = config.get(stringify!($field)) {
                base_limits.$field = value.clone().try_into()?;
            })* };
        }
        load_fields!(
            seed,
            cpuct,
            cpuct_at_root,
            cpuct_base,
            cpuct_factor,
            cpuct_base_at_root,
            cpuct_factor_at_root,
            root_dirichlet_alpha,
            root_exploration_fraction,
            fpu_value,
            fpu_value_at_root,
            fpu_absolute_at_root,
            minimum_kldgain_per_node,
            policy_softmax_temp,
            draw_score
        );
    }
    if let Some(value) = args.minimum_kldgain_per_node {
        base_limits.minimum_kldgain_per_node = value;
    }
    let model = AzNnue::load(&args.model)?;
    let text = fs::read_to_string(&args.trajectories)?;
    if let Some(output) = args.discover_output.as_ref() {
        let mut engine = Engine::new(&args.pikafish)?;
        let mut cases = Vec::new();
        for line in text
            .lines()
            .filter(|line| line.starts_with("vs-pikafish-final:"))
        {
            let game: usize = line
                .split("game=")
                .nth(1)
                .unwrap()
                .split_whitespace()
                .next()
                .unwrap()
                .parse()?;
            let chinese = if line.contains("chinese=red") {
                Color::Red
            } else {
                Color::Black
            };
            let (fen, moves) = line
                .split(" position fen ")
                .nth(1)
                .unwrap()
                .split_once(" moves ")
                .unwrap();
            let moves = moves.split_whitespace().collect::<Vec<_>>();
            let mut position = Position::from_fen(fen)?;
            let mut worst = None;
            let mut worst_gap = 0;
            for (ply, &played) in moves.iter().take(args.discover_plies).enumerate() {
                if position.side_to_move() == chinese {
                    let cmd = command(fen, &moves[..ply]);
                    let best = engine.score(&cmd, args.depth, None)?;
                    let actual = engine.score(&cmd, args.depth, Some(played))?;
                    let gap = best.cp - actual.cp;
                    if best.cp >= args.discover_min_best_cp && gap > worst_gap {
                        worst_gap = gap;
                        worst = Some(Case {
                            game,
                            ply,
                            fen: position.to_fen(),
                            played: played.into(),
                        });
                    }
                }
                let mv = position
                    .parse_uci_move(played)
                    .ok_or("invalid trajectory move")?;
                position.make_move(mv);
            }
            if let Some(case) = worst {
                println!(
                    "discovered game={game} ply={} played={} cp_gap={worst_gap}",
                    case.ply, case.played
                );
                cases.push(case);
            }
        }
        use std::io::Write;
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)?;
        file.write_all(toml::to_string(&Cases { cases })?.as_bytes())?;
        return Ok(());
    }
    let cases: Cases = toml::from_str(&fs::read_to_string(&args.cases)?)?;
    let mut engine = Engine::new(&args.pikafish)?;
    let mut evaluated = 0;
    println!(
        "audit model={} fixed_cases={} simulations={:?} limits={base_limits:?}",
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
        let best = engine.score(&cmd, args.depth, None)?;
        let actual = engine.score(&cmd, args.depth, Some(played))?;
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
        if actual.mate == Some(-1) {
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
        if actual.mate == Some(-1) && played == "a9b9" {
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
                        ..base_limits
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
                    ..base_limits
                },
            );
            let selected = search.best_move.unwrap().to_string();
            let score = engine.score(&cmd, args.depth, Some(&selected))?;
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
            if args.trace || (game == 5 && simulations == 6400) {
                if args.trace {
                    let mut child = position.clone();
                    let mut child_history = history.clone();
                    let mv = search.best_move.unwrap();
                    child_history.push(child.rule_history_entry_after_move(mv));
                    child.make_move(mv);
                    let mut child_moves = moves[..ply].to_vec();
                    child_moves.push(&selected);
                    let pf_reply = engine.score(&command(fen, &child_moves), args.depth, None)?;
                    let reply = alphazero_search_with_rules(
                        &child,
                        Some(child_history),
                        None,
                        &model,
                        AzSearchLimits {
                            simulations,
                            ..base_limits
                        },
                    );
                    println!(
                        "  opponent_probe sims={simulations} pf_reply={pf_reply:?} selected={:?} pf_candidate={:?} search_q={:.4}",
                        reply.best_move,
                        reply
                            .candidates
                            .iter()
                            .find(|c| c.mv.to_string() == pf_reply.best),
                        reply.value_q,
                    );
                }
                let (_, trace) = alphazero_search_trace_with_rules(
                    &position,
                    Some(history.clone()),
                    None,
                    &model,
                    AzSearchLimits {
                        simulations,
                        ..base_limits
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
