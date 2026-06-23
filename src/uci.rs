use crate::az::{
    AzNnue, AzSearchLimits, GumbelSearchConfig, alphazero_search_with_history_and_rules,
    gumbel_search_with_history_and_rules,
};
use crate::nnue::{HISTORY_PLIES, HistoryMove};
use crate::xiangqi::{Position, RuleHistoryEntry, RuleOutcome};
use std::io::{self, BufRead, Write};

#[derive(Clone, Debug)]
struct UciState {
    position: Position,
    history: Vec<HistoryMove>,
    rule_history: Vec<RuleHistoryEntry>,
    eval_file: String,
    model: Option<AzNnue>,
    simulations: usize,
    threads: usize,
    cpuct: f32,
    cpuct_at_root: f32,
    search_gumbel: bool,
    gumbel_actions: usize,
    gumbel_scale: f32,
    gumbel_value_scale: f32,
    gumbel_maxvisit_init: f32,
    cpuct_base: f32,
    cpuct_factor: f32,
    cpuct_base_at_root: f32,
    cpuct_factor_at_root: f32,
    fpu_value: f32,
    fpu_value_at_root: f32,
    draw_score: f32,
    moves_left_max_effect: f32,
    moves_left_slope: f32,
    moves_left_threshold: f32,
    moves_left_constant_factor: f32,
    moves_left_scaled_factor: f32,
    moves_left_quadratic_factor: f32,
    seed: u64,
}

impl Default for UciState {
    fn default() -> Self {
        Self {
            position: Position::startpos(),
            history: Vec::new(),
            rule_history: Position::startpos().initial_rule_history(),
            eval_file: "model.safetensors".into(),
            model: None,
            simulations: 10_000,
            threads: 1,
            cpuct: 1.5,
            cpuct_at_root: 3.0,
            search_gumbel: false,
            gumbel_actions: 16,
            gumbel_scale: 0.0,
            gumbel_value_scale: 0.02,
            gumbel_maxvisit_init: 50.0,
            cpuct_base: 19652.0,
            cpuct_factor: 2.0,
            cpuct_base_at_root: 19652.0,
            cpuct_factor_at_root: 2.0,
            fpu_value: 0.23,
            fpu_value_at_root: 1.0,
            draw_score: 0.0,
            moves_left_max_effect: 0.25,
            moves_left_slope: 0.002,
            moves_left_threshold: 0.6,
            moves_left_constant_factor: 0.0,
            moves_left_scaled_factor: 0.15,
            moves_left_quadratic_factor: 0.85,
            seed: 20260409,
        }
    }
}

pub fn run_uci() {
    let stdin = io::stdin();
    let mut state = UciState::default();
    for line in stdin.lock().lines() {
        let Ok(line) = line else {
            break;
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match line.split_whitespace().next() {
            Some("uci") => print_uci_id(),
            Some("isready") => {
                ensure_model(&mut state);
                println!("readyok");
                flush();
            }
            Some("ucinewgame") => {
                state.position = Position::startpos();
                state.history.clear();
                state.rule_history = state.position.initial_rule_history();
                state.seed = 20260409;
            }
            Some("setoption") => handle_setoption(line, &mut state),
            Some("position") => handle_position(line, &mut state),
            Some("go") => handle_go(line, &mut state),
            Some("stop") => {}
            Some("quit") => break,
            _ => {}
        }
    }
}

fn print_uci_id() {
    println!("id name ChineseAI AZ-NNUE");
    println!("id author ChineseAI");
    println!("option name EvalFile type string default model.safetensors");
    println!("option name Simulations type spin default 10000 min 1 max 100000000");
    println!("option name Threads type spin default 1 min 1 max 1");
    println!("option name Cpuct type string default 1.5");
    println!("option name CpuctAtRoot type string default 3.0");
    println!(
        "option name SearchAlgorithm type combo default alphazero var alphazero var gumbel"
    );
    println!("option name GumbelActions type spin default 16 min 1 max 128");
    println!("option name GumbelScale type string default 0.0");
    println!("option name GumbelValueScale type string default 0.02");
    println!("option name GumbelMaxVisitInit type string default 50.0");
    println!("option name CpuctBase type string default 19652.0");
    println!("option name CpuctFactor type string default 2.0");
    println!("option name CpuctBaseAtRoot type string default 19652.0");
    println!("option name CpuctFactorAtRoot type string default 2.0");
    println!("option name FpuValue type string default 0.23");
    println!("option name FpuValueAtRoot type string default 1.0");
    println!("option name DrawScore type string default 0.0");
    println!("option name MovesLeftMaxEffect type string default 0.25");
    println!("option name MovesLeftSlope type string default 0.002");
    println!("option name MovesLeftThreshold type string default 0.6");
    println!("option name MovesLeftConstantFactor type string default 0.0");
    println!("option name MovesLeftScaledFactor type string default 0.15");
    println!("option name MovesLeftQuadraticFactor type string default 0.85");
    println!("uciok");
    flush();
}

fn ensure_model(state: &mut UciState) {
    if state.model.is_some() {
        return;
    }
    state.model = Some(AzNnue::load(&state.eval_file).unwrap_or_else(|err| {
        println!(
            "info string failed to load {}, using random model: {}",
            state.eval_file, err
        );
        flush();
        AzNnue::random(128, state.seed)
    }));
}

fn handle_setoption(line: &str, state: &mut UciState) {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    let Some(name_index) = tokens.iter().position(|token| *token == "name") else {
        return;
    };
    let value_index = tokens.iter().position(|token| *token == "value");
    let name_end = value_index.unwrap_or(tokens.len());
    let name = tokens[name_index + 1..name_end]
        .join(" ")
        .to_ascii_lowercase();
    let value = value_index
        .map(|index| tokens[index + 1..].join(" "))
        .unwrap_or_default();

    match name.as_str() {
        "evalfile" => {
            state.eval_file = value;
            state.model = None;
        }
        "simulations" => {
            state.simulations = value.parse::<usize>().unwrap_or(state.simulations).max(1);
        }
        "threads" => {
            let _ = value;
            state.threads = 1;
        }
        "cpuct" => {
            state.cpuct = value.parse::<f32>().unwrap_or(state.cpuct).max(0.0);
        }
        "cpuctatroot" => {
            state.cpuct_at_root = value.parse::<f32>().unwrap_or(state.cpuct_at_root).max(0.0);
        }
        "searchalgorithm" => {
            state.search_gumbel = value.trim().eq_ignore_ascii_case("gumbel");
        }
        "gumbelactions" => {
            state.gumbel_actions = value.parse::<usize>().unwrap_or(state.gumbel_actions).max(1);
        }
        "gumbelscale" => {
            state.gumbel_scale = value.parse::<f32>().unwrap_or(state.gumbel_scale).max(0.0);
        }
        "gumbelvaluescale" => {
            state.gumbel_value_scale = value
                .parse::<f32>()
                .unwrap_or(state.gumbel_value_scale)
                .max(0.0);
        }
        "gumbelmaxvisitinit" => {
            state.gumbel_maxvisit_init = value
                .parse::<f32>()
                .unwrap_or(state.gumbel_maxvisit_init)
                .max(0.0);
        }
        "cpuctbase" => {
            state.cpuct_base = value.parse::<f32>().unwrap_or(state.cpuct_base).max(1.0);
        }
        "cpuctfactor" => {
            state.cpuct_factor = value.parse::<f32>().unwrap_or(state.cpuct_factor).max(0.0);
        }
        "cpuctbaseatroot" => {
            state.cpuct_base_at_root = value
                .parse::<f32>()
                .unwrap_or(state.cpuct_base_at_root)
                .max(1.0);
        }
        "cpuctfactoratroot" => {
            state.cpuct_factor_at_root = value
                .parse::<f32>()
                .unwrap_or(state.cpuct_factor_at_root)
                .max(0.0);
        }
        "fpuvalue" => {
            state.fpu_value = value.parse::<f32>().unwrap_or(state.fpu_value).max(0.0);
        }
        "fpuvalueatroot" => {
            state.fpu_value_at_root = value
                .parse::<f32>()
                .unwrap_or(state.fpu_value_at_root)
                .clamp(-1.0, 1.0);
        }
        "drawscore" => {
            state.draw_score = value
                .parse::<f32>()
                .unwrap_or(state.draw_score)
                .clamp(-1.0, 1.0);
        }
        "movesleftmaxeffect" => {
            state.moves_left_max_effect = value
                .parse::<f32>()
                .unwrap_or(state.moves_left_max_effect)
                .max(0.0);
        }
        "movesleftslope" => {
            state.moves_left_slope = value
                .parse::<f32>()
                .unwrap_or(state.moves_left_slope)
                .max(0.0);
        }
        "movesleftthreshold" => {
            state.moves_left_threshold = value
                .parse::<f32>()
                .unwrap_or(state.moves_left_threshold)
                .clamp(0.0, 1.0);
        }
        "movesleftconstantfactor" => {
            state.moves_left_constant_factor = value
                .parse::<f32>()
                .unwrap_or(state.moves_left_constant_factor);
        }
        "movesleftscaledfactor" => {
            state.moves_left_scaled_factor = value
                .parse::<f32>()
                .unwrap_or(state.moves_left_scaled_factor);
        }
        "movesleftquadraticfactor" => {
            state.moves_left_quadratic_factor = value
                .parse::<f32>()
                .unwrap_or(state.moves_left_quadratic_factor);
        }
        _ => {}
    }
}

fn handle_position(line: &str, state: &mut UciState) {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    if tokens.get(1) == Some(&"startpos") {
        state.position = Position::startpos();
        state.history.clear();
        state.rule_history = state.position.initial_rule_history();
        if let Some(moves_index) = tokens.iter().position(|token| *token == "moves") {
            let move_list = &tokens[moves_index + 1..];
            apply_uci_moves(
                &mut state.position,
                &mut state.history,
                &mut state.rule_history,
                move_list,
            );
        }
        return;
    }

    if tokens.get(1) == Some(&"fen") {
        let moves_index = tokens.iter().position(|token| *token == "moves");
        let fen_end = moves_index.unwrap_or(tokens.len());
        let fen = tokens[2..fen_end].join(" ");
        if let Ok(position) = Position::from_fen(&fen) {
            state.position = position;
            state.history.clear();
            state.rule_history = state.position.initial_rule_history();
            if let Some(moves_index) = moves_index {
                let move_list = &tokens[moves_index + 1..];
                apply_uci_moves(
                    &mut state.position,
                    &mut state.history,
                    &mut state.rule_history,
                    move_list,
                );
            }
        }
    }
}

fn apply_uci_moves(
    position: &mut Position,
    history: &mut Vec<HistoryMove>,
    rule_history: &mut Vec<RuleHistoryEntry>,
    moves: &[&str],
) {
    for text in moves {
        let Some(mv) = position.parse_uci_move(text) else {
            break;
        };
        if !position.legal_moves_with_rules(rule_history).contains(&mv) {
            break;
        }
        if let Some(piece) = position.piece_at(mv.from as usize) {
            history.push(HistoryMove {
                piece,
                captured: position.piece_at(mv.to as usize),
                mv,
            });
            let overflow = history.len().saturating_sub(HISTORY_PLIES);
            if overflow > 0 {
                history.drain(0..overflow);
            }
        }
        rule_history.push(position.rule_history_entry_after_move(mv));
        position.make_move(mv);
    }
}

fn position_is_rule_draw(position: &Position, rule_history: &[RuleHistoryEntry]) -> bool {
    matches!(
        position.rule_outcome_with_history(rule_history),
        Some(RuleOutcome::Draw(_))
    )
}

fn handle_go(_line: &str, state: &mut UciState) {
    ensure_model(state);
    let simulations = state.simulations.max(1);
    let model = state.model.as_ref().expect("model was loaded");

    if position_is_rule_draw(&state.position, &state.rule_history) {
        println!("info depth 0 nodes 0 time 0 score cp 0 pv draw");
        println!("bestmove draw");
        flush();
        return;
    }

    let legal = state.position.legal_moves_with_rules(&state.rule_history);

    if legal.is_empty() {
        println!("info depth 1 nodes 0 time 0 score cp -32000");
        flush();
        return;
    }

    let started = std::time::Instant::now();
    let limits = AzSearchLimits {
            simulations,
            seed: state.seed,
            cpuct: state.cpuct,
            cpuct_at_root: state.cpuct_at_root,
            cpuct_base: state.cpuct_base,
            cpuct_factor: state.cpuct_factor,
            cpuct_base_at_root: state.cpuct_base_at_root,
            cpuct_factor_at_root: state.cpuct_factor_at_root,
            max_depth: 0,
            root_dirichlet_alpha: 0.0,
            root_exploration_fraction: 0.0,
            fpu_value: state.fpu_value,
            fpu_value_at_root: state.fpu_value_at_root,
            draw_score: state.draw_score,
            moves_left_max_effect: state.moves_left_max_effect,
            moves_left_slope: state.moves_left_slope,
            moves_left_threshold: state.moves_left_threshold,
            moves_left_constant_factor: state.moves_left_constant_factor,
            moves_left_scaled_factor: state.moves_left_scaled_factor,
            moves_left_quadratic_factor: state.moves_left_quadratic_factor,
            value_scale: 1.0,
        };
    let result = if state.search_gumbel {
        gumbel_search_with_history_and_rules(
            &state.position,
            &state.history,
            Some(state.rule_history.clone()),
            Some(legal),
            model,
            limits,
            GumbelSearchConfig {
                max_num_considered_actions: state.gumbel_actions,
                gumbel_scale: state.gumbel_scale,
                value_scale: state.gumbel_value_scale,
                maxvisit_init: state.gumbel_maxvisit_init,
            },
        )
    } else {
        alphazero_search_with_history_and_rules(
            &state.position,
            &state.history,
            Some(state.rule_history.clone()),
            Some(legal),
            model,
            limits,
        )
    };
    state.seed = state.seed.wrapping_add(1);
    match result.best_move {
        Some(mv) => {
            let best_text = mv.to_string();
            let elapsed_ms = started.elapsed().as_millis();
            let nps = (result.simulations as u128 * 1000 / elapsed_ms.max(1)) as usize;
            let wdl = uci_wdl(result.value_wdl);
            println!(
                "info depth {} seldepth {} nodes {} nps {} time {} score cp {} wdl {} {} {} pv {}",
                result.search_depth_avg.round() as usize,
                result.search_depth_max,
                result.simulations,
                nps,
                elapsed_ms,
                result.value_cp,
                wdl[0],
                wdl[1],
                wdl[2],
                best_text
            );
            println!("bestmove {best_text}");
        }
        None => {
            println!(
                "info depth 1 nodes {} time {} score cp {}",
                result.simulations,
                started.elapsed().as_millis(),
                result.value_cp
            );
            if position_is_rule_draw(&state.position, &state.rule_history) {
                println!("bestmove draw");
            }
        }
    }
    flush();
}

fn uci_wdl(probabilities: [f32; 3]) -> [u16; 3] {
    let mut wdl = probabilities.map(|value| (value.clamp(0.0, 1.0) * 1000.0).round() as u16);
    let sum = wdl.iter().copied().map(i32::from).sum::<i32>();
    let draw = (i32::from(wdl[1]) + 1000 - sum).clamp(0, 1000) as u16;
    wdl[1] = draw;
    wdl
}

fn flush() {
    let _ = io::stdout().flush();
}
