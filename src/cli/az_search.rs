use crate::cli::args::*;
use chineseai::{
    az::{
        AzMatingGoal, AzNnue, AzSearchLimits, alphazero_search_trace_with_rules,
        alphazero_search_with_rules,
    },
    xiangqi::{Move, Position},
};
use std::time::Instant;

pub(crate) fn fixed_az_search_limits(
    simulations: usize,
    seed: u64,
    cpuct: f32,
    cpuct_at_root: f32,
    max_depth: usize,
    policy_softmax_temp: f32,
) -> AzSearchLimits {
    AzSearchLimits {
        simulations,
        seed,
        cpuct,
        cpuct_at_root,
        cpuct_base: 38739.0,
        cpuct_factor: 3.894,
        cpuct_base_at_root: 38739.0,
        cpuct_factor_at_root: 3.894,
        max_depth,
        root_dirichlet_alpha: 0.0,
        root_exploration_fraction: 0.0,
        fpu_value: 0.23,
        fpu_value_at_root: 1.0,
        fpu_absolute_at_root: true,
        minimum_kldgain_per_node: 0.0,
        policy_softmax_temp: policy_softmax_temp.max(1.0e-3),
        draw_score: 0.0,
        value_scale: 1.0,
        mating_goal: AzMatingGoal::default(),
    }
}

pub(crate) fn print_az_search_candidates(result: &chineseai::az::AzSearchResult, top: usize) {
    let mut candidates = result.candidates.iter().collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        right
            .visits
            .cmp(&left.visits)
            .then_with(|| right.policy.total_cmp(&left.policy))
            .then_with(|| right.q.total_cmp(&left.q))
    });
    let shown = if top == 0 {
        candidates.len()
    } else {
        top.min(candidates.len())
    };
    println!(
        "\nCANDIDATES — visits descending ({shown}/{})",
        candidates.len()
    );
    println!(
        "    #  B  MOVE      VISITS  VISIT P       Q      CP       NET P      TREE P   MLH(ply)  GOAL(owner)"
    );
    println!(
        "  ---- --  -------  --------  -------  ------  ------  ----------  ----------  ---------"
    );
    for (rank, candidate) in candidates.into_iter().take(shown).enumerate() {
        let best = if Some(candidate.mv) == result.best_move {
            "*"
        } else {
            " "
        };
        println!(
            "  {:>4}  {}  {:<7}  {:>8}  {:>6.2}%  {:>+6.3}  {:>+6}  {:>9.5}  {:>9.5}  {:>9}  {:>11.4}",
            rank + 1,
            best,
            candidate.mv,
            candidate.visits,
            candidate.policy * 100.0,
            candidate.q,
            chineseai::az::cp_from_q(candidate.q),
            candidate.raw_prior,
            candidate.prior,
            candidate
                .moves_left
                .map(|m| format!("{m:.1}"))
                .unwrap_or_else(|| "-".into()),
            candidate.goal_value,
        );
    }
}

pub(crate) fn print_az_search_trace(trace_move: Move, trace: &[chineseai::az::AzSearchTraceStep]) {
    println!(
        "\nPRINCIPAL TRACE — root {trace_move}, {} plies",
        trace.len()
    );
    if trace.is_empty() {
        println!("  Root move was not expanded.");
        return;
    }
    println!(
        "  PLY  MOVE      VISITS       Q     PRIOR  CHECK  EXPANDED    CHILD Q       CHILD W/D/L"
    );
    println!(
        "  ---  -------  --------  ------  --------  -----  --------  --------  ------------------"
    );
    for step in trace {
        println!(
            "  {:>3}  {:<7}  {:>8}  {:>+6.3}  {:>7.5}  {:>5}  {:>8}  {:>+8.3}  {:>5.1}%/{:>5.1}%/{:>5.1}%",
            step.ply,
            step.mv,
            step.visits,
            step.q,
            step.prior,
            if step.gives_check { "yes" } else { "no" },
            if step.child_expanded { "yes" } else { "no" },
            step.child_value,
            step.child_value_wdl[0] * 100.0,
            step.child_value_wdl[1] * 100.0,
            step.child_value_wdl[2] * 100.0,
        );
        println!("       fen: {}", step.child_fen);
    }
}

pub(crate) fn parse_position(text: &str) -> Position {
    if text.trim().is_empty() || text == "startpos" {
        Position::startpos()
    } else {
        Position::from_fen(text).unwrap_or_else(|err| {
            panic!("invalid FEN `{text}`: {err}");
        })
    }
}

fn search_mating_goal(cmd: &AzSearchArgs, position: &Position) -> AzMatingGoal {
    let goal = AzMatingGoal {
        target: cmd.mating_target,
        side: cmd
            .mating_side
            .map(MatingSide::color)
            .unwrap_or(position.side_to_move()),
        strength: cmd.mating_strength,
    };
    goal.validate().expect("invalid mating goal");
    goal
}

fn verification_limits(mut limits: AzSearchLimits, simulations: usize) -> AzSearchLimits {
    limits.simulations = simulations.max(1);
    limits.seed = 0;
    limits
}

pub(crate) fn run(cmd: AzSearchArgs) {
    let model_path = &cmd.model;
    let simulations = cmd.simulations.max(1);
    let cpuct = cmd.cpuct.max(0.0);
    let cpuct_at_root = cmd.cpuct_at_root.max(0.0);
    let fen = cmd.fen.join(" ");
    let mut position = parse_position(&fen);
    let mut rule_history = position.initial_rule_history();
    for text in &cmd.moves {
        let mv = position
            .parse_uci_move(text)
            .unwrap_or_else(|| panic!("invalid or illegal --move `{text}` for this position"));
        rule_history.push(position.rule_history_entry_after_move(mv));
        position.make_move(mv);
    }
    let mating_goal = search_mating_goal(&cmd, &position);
    let model = AzNnue::load(model_path).unwrap_or_else(|err| {
        panic!("failed to load `{model_path}`: {err}");
    });
    let search_limits = AzSearchLimits {
        simulations,
        seed: 0,
        cpuct,
        cpuct_at_root,
        cpuct_base: cmd.cpuct_base.max(1.0),
        cpuct_factor: cmd.cpuct_factor.max(0.0),
        cpuct_base_at_root: cmd.cpuct_base_at_root.max(1.0),
        cpuct_factor_at_root: cmd.cpuct_factor_at_root.max(0.0),
        max_depth: cmd.max_depth,
        root_dirichlet_alpha: 0.0,
        root_exploration_fraction: 0.0,
        fpu_value: cmd.fpu_value.max(0.0),
        fpu_value_at_root: cmd.fpu_value_at_root.max(0.0),
        fpu_absolute_at_root: true,
        minimum_kldgain_per_node: 0.0,
        policy_softmax_temp: cmd.policy_softmax_temp.max(1.0e-3),
        draw_score: cmd.draw_score.clamp(-1.0, 1.0),
        value_scale: cmd.value_scale.clamp(0.0, 1.0),
        mating_goal,
    };
    let root_moves = if cmd.root_moves.is_empty() {
        None
    } else {
        Some(
            cmd.root_moves
                .iter()
                .map(|text| {
                    position.parse_uci_move(text).unwrap_or_else(|| {
                        panic!("invalid or illegal --root-move `{text}` for this position")
                    })
                })
                .collect::<Vec<_>>(),
        )
    };
    let trace_move = cmd.trace_move.as_deref().map(|text| {
        position
            .parse_uci_move(text)
            .unwrap_or_else(|| panic!("invalid or illegal --trace-move `{text}`"))
    });
    let search_started = Instant::now();
    let (result, trace) = if let Some(trace_move) = trace_move {
        alphazero_search_trace_with_rules(
            &position,
            Some(rule_history.clone()),
            root_moves,
            &model,
            search_limits,
            trace_move,
        )
    } else {
        (
            alphazero_search_with_rules(
                &position,
                Some(rule_history.clone()),
                root_moves,
                &model,
                search_limits,
            ),
            Vec::new(),
        )
    };
    let search_elapsed = search_started.elapsed();
    let mut by_visits = result.candidates.clone();
    by_visits.sort_by(|left, right| {
        right
            .visits
            .cmp(&left.visits)
            .then_with(|| right.policy.total_cmp(&left.policy))
            .then_with(|| right.q.total_cmp(&left.q))
    });
    let visited_actions = by_visits
        .iter()
        .filter(|candidate| candidate.visits > 0)
        .count();
    let elapsed_seconds = search_elapsed.as_secs_f64().max(f64::EPSILON);
    let best_move = result
        .best_move
        .map(|mv| mv.to_string())
        .unwrap_or_else(|| "(none)".into());
    println!("AZ SEARCH");
    println!("=========");
    println!("\nPOSITION");
    println!("  FEN          {}", position.to_fen());
    println!("  Side         {:?}", position.side_to_move());
    println!(
        "  Applied      {}",
        if cmd.moves.is_empty() {
            "(none)".to_string()
        } else {
            cmd.moves.join(" ")
        }
    );
    println!(
        "  Root moves   {}",
        if cmd.root_moves.is_empty() {
            "all legal".to_string()
        } else {
            cmd.root_moves.join(" ")
        }
    );
    println!("\nCONFIGURATION");
    println!("  Model        {model_path}");
    println!("  Simulations  {simulations}");
    println!(
        "  PUCT         non-root={cpuct:.3} root={cpuct_at_root:.3} base={:.1}/{:.1} factor={:.3}/{:.3}",
        search_limits.cpuct_base,
        search_limits.cpuct_base_at_root,
        search_limits.cpuct_factor,
        search_limits.cpuct_factor_at_root
    );
    println!(
        "  FPU reduce   non-root={:.3} root={:.3}",
        search_limits.fpu_value, search_limits.fpu_value_at_root
    );
    println!("  Policy temp  {:.3}", search_limits.policy_softmax_temp);
    println!("  Draw score   {:.3}", search_limits.draw_score);
    println!(
        "  杀法目标     {} owner={:?} strength={:.3} 当前有效={}",
        mating_goal.target.name(),
        mating_goal.side,
        mating_goal.strength,
        mating_goal.effective(&position).enabled()
    );
    println!("\nRESULT");
    println!("  Best move    {best_move}");
    println!(
        "  目标达成值   {:.4}（固定执行方 {:?} 视角）",
        result.goal_value, mating_goal.side
    );
    println!(
        "  Search value Q={:+.4}  CP={:+}  W/D/L={:.2}%/{:.2}%/{:.2}%",
        result.value_q,
        result.value_cp,
        result.value_wdl[0] * 100.0,
        result.value_wdl[1] * 100.0,
        result.value_wdl[2] * 100.0
    );
    println!(
        "  Network WDL  {:.2}%/{:.2}%/{:.2}%",
        result.network_value_wdl[0] * 100.0,
        result.network_value_wdl[1] * 100.0,
        result.network_value_wdl[2] * 100.0
    );
    println!(
        "  Root actions {} legal, {} visited",
        result.candidates.len(),
        visited_actions
    );
    println!(
        "  Depth        avg={:.2} max={} limit={} cutoffs={}",
        result.search_depth_avg,
        result.search_depth_max,
        result.search_depth_limit,
        result.search_depth_cutoffs
    );
    println!(
        "  Performance  {:.3} ms, {:.0} simulations/s",
        search_elapsed.as_secs_f64() * 1000.0,
        result.simulations as f64 / elapsed_seconds
    );
    if let Some(distance) = result
        .candidates
        .iter()
        .find(|c| Some(c.mv) == result.best_move)
        .and_then(|c| c.moves_left)
    {
        println!(
            "  Moves left   {:.1} plies; MLH enabled={} threshold |Q|>{:.3} / |CP|>{:.0}",
            distance,
            model.moves_left_params.enabled,
            model.moves_left_params.threshold,
            model.moves_left_params.threshold * 1000.0
        );
    }
    print_az_search_candidates(&result, cmd.top);
    if let Some(trace_move) = trace_move {
        print_az_search_trace(trace_move, &trace);
    }
    let verify_sims = if cmd.verify_sims == 0 {
        simulations
    } else {
        cmd.verify_sims
    };
    let mut verify_moves = by_visits
        .iter()
        .take(cmd.verify_top)
        .map(|candidate| candidate.mv)
        .collect::<Vec<_>>();
    for text in &cmd.verify_moves {
        let mv = position.parse_uci_move(text).unwrap_or_else(|| {
            panic!("invalid or illegal --verify-move `{text}` for this position")
        });
        if !verify_moves.contains(&mv) {
            verify_moves.push(mv);
        }
    }
    if !verify_moves.is_empty() {
        println!("\nCHILD VERIFICATION — {verify_sims} simulations each");
        println!(
            "  MOVE     VISITS   ROOT Q     NN Q   DEEP Q       ΔQ      CP  GOAL(owner) OPPONENT REPLY"
        );
        println!("  -------  -------  -------  -------  -------  -------  ------  --------------");
    }
    for mv in verify_moves {
        let Some(root_candidate) = result.candidates.iter().find(|item| item.mv == mv) else {
            println!("  {mv:<7}  unavailable at root");
            continue;
        };
        let mut child_rule_history = rule_history.clone();
        child_rule_history.push(position.rule_history_entry_after_move(mv));
        let mut child = position.clone();
        child.make_move(mv);
        let child_legal = child.legal_moves_with_rules(&child_rule_history);
        let child_nn_q = model.evaluate_value_with_rules(&child, &child_rule_history, &child_legal);
        let verify_limits = verification_limits(search_limits, verify_sims);
        let verified = alphazero_search_with_rules(
            &child,
            Some(child_rule_history),
            Some(child_legal),
            &model,
            verify_limits,
        );
        let verified_root_q = -verified.value_q;
        let verified_root_cp = -verified.value_cp;
        println!(
            "  {:<7}  {:>7}  {:>+7.3}  {:>+7.3}  {:>+7.3}  {:>+7.3}  {:>+6}  {:>11.4} {}",
            mv,
            root_candidate.visits,
            root_candidate.q,
            -child_nn_q,
            verified_root_q,
            verified_root_q - root_candidate.q,
            verified_root_cp,
            verified.goal_value,
            verified
                .best_move
                .map(|best| best.to_string())
                .unwrap_or_else(|| "(none)".into())
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chineseai::{az::MatingTarget, xiangqi::Color};
    use clap::Parser;

    fn parse_args(extra: &[&str]) -> AzSearchArgs {
        let cli = Cli::try_parse_from(
            ["chineseai", "az-search", "model.safetensors"]
                .into_iter()
                .chain(extra.iter().copied()),
        )
        .unwrap();
        let Some(CliCommand::AzSearch(args)) = cli.command else {
            panic!("az-search args");
        };
        args
    }

    #[test]
    fn mating_cli_validates_target_and_strength_bounds() {
        for name in ["double-cannon", "double_cannon", "重炮"] {
            assert_eq!(
                parse_args(&["--mating-target", name]).mating_target,
                MatingTarget::DoubleCannon
            );
        }
        assert_eq!(
            parse_args(&["--mating-target", "铁门栓"]).mating_target,
            MatingTarget::IronBolt
        );
        for value in ["0", "1"] {
            assert!(
                parse_args(&["--mating-strength", value])
                    .mating_strength
                    .is_finite()
            );
        }
        for value in ["NaN", "inf", "1.1", "-0.1"] {
            assert!(
                Cli::try_parse_from([
                    "chineseai",
                    "az-search",
                    "model",
                    "--mating-strength",
                    value
                ])
                .is_err()
            );
        }
        assert!(
            Cli::try_parse_from([
                "chineseai",
                "az-search",
                "model",
                "--mating-target",
                "unknown"
            ])
            .is_err()
        );
    }

    #[test]
    fn search_goal_owner_is_set_after_moves_and_inherited_by_child_search() {
        let args = parse_args(&["--mating-target", "重炮", "--move", "b0c2"]);
        let mut position = Position::startpos();
        for text in &args.moves {
            position.make_move(position.parse_uci_move(text).unwrap());
        }
        let goal = search_mating_goal(&args, &position);
        assert_eq!(goal.side, Color::Black);
        let limits = AzSearchLimits {
            mating_goal: goal,
            ..Default::default()
        };
        let child_limits = verification_limits(limits, 7);
        position.make_move(position.legal_moves()[0]);
        assert_eq!(position.side_to_move(), Color::Red);
        assert_eq!(child_limits.mating_goal, goal);
        assert_eq!(child_limits.simulations, 7);
        let explicit = parse_args(&["--mating-target", "重炮", "--mating-side", "red"]);
        assert_eq!(search_mating_goal(&explicit, &position).side, Color::Red);
    }
}
