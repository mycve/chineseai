use crate::cli::args::*;
use chineseai::{
    az::{AzNnue, AzSearchLimits, alphazero_search_trace_with_rules, alphazero_search_with_rules},
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
        "    #  B  MOVE      VISITS  VISIT P       Q      CP       NET P      TREE P   MLH(ply)"
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
            "  {:>4}  {}  {:<7}  {:>8}  {:>6.2}%  {:>+6.3}  {:>+6}  {:>9.5}  {:>9.5}  {:>9}",
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

pub(crate) fn run(cmd: AzSearchArgs) {
    let model_path = cmd.model;
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
    let model = AzNnue::load(&model_path).unwrap_or_else(|err| {
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
    println!("\nRESULT");
    println!("  Best move    {best_move}");
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
        println!("  MOVE     VISITS   ROOT Q     NN Q   DEEP Q       ΔQ      CP  OPPONENT REPLY");
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
        let mut verify_limits = search_limits;
        verify_limits.simulations = verify_sims.max(1);
        verify_limits.seed = 0;
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
            "  {:<7}  {:>7}  {:>+7.3}  {:>+7.3}  {:>+7.3}  {:>+7.3}  {:>+6}  {}",
            mv,
            root_candidate.visits,
            root_candidate.q,
            -child_nn_q,
            verified_root_q,
            verified_root_q - root_candidate.q,
            verified_root_cp,
            verified
                .best_move
                .map(|best| best.to_string())
                .unwrap_or_else(|| "(none)".into())
        );
    }
}
