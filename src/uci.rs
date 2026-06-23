use crate::az::{
    AzNnue, AzSearchLimits, GumbelSearchConfig, cp_from_q, gumbel_search_with_history_and_rules,
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
    gumbel_actions: usize,
    gumbel_scale: f32,
    gumbel_value_scale: f32,
    gumbel_maxvisit_init: f32,
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
            gumbel_actions: 24,
            gumbel_scale: 0.0,
            gumbel_value_scale: 0.1,
            gumbel_maxvisit_init: 50.0,
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
    println!("option name GumbelActions type spin default 24 min 1 max 128");
    println!("option name GumbelScale type string default 0.0");
    println!("option name GumbelValueScale type string default 0.1");
    println!("option name GumbelMaxVisitInit type string default 50.0");
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
        "gumbelactions" => {
            state.gumbel_actions = value
                .parse::<usize>()
                .unwrap_or(state.gumbel_actions)
                .max(1);
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
        println!("bestmove 0000");
        flush();
        return;
    }

    let legal = state.position.legal_moves_with_rules(&state.rule_history);

    if legal.is_empty() {
        println!("info depth 1 nodes 0 time 0 score cp -32000");
        println!("bestmove 0000");
        flush();
        return;
    }

    let started = std::time::Instant::now();
    let limits = AzSearchLimits {
        simulations,
        seed: state.seed,
        ..AzSearchLimits::default()
    };
    let result = gumbel_search_with_history_and_rules(
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
    );
    state.seed = state.seed.wrapping_add(1);
    match result.best_move {
        Some(mv) => {
            let best_text = mv.to_string();
            let elapsed_ms = started.elapsed().as_millis();
            let nps = (result.simulations as u128 * 1000 / elapsed_ms.max(1)) as usize;
            let selected = result
                .candidates
                .iter()
                .find(|candidate| candidate.mv == mv);
            let selected_q = selected
                .map(|candidate| candidate.q)
                .unwrap_or(result.value_q);
            let selected_wdl = selected
                .map(|candidate| candidate.value_wdl)
                .unwrap_or(result.value_wdl);
            let wdl = uci_wdl(selected_wdl);
            println!(
                "info depth {} seldepth {} nodes {} nps {} time {} score cp {} wdl {} {} {} pv {}",
                result.search_depth_avg.round() as usize,
                result.search_depth_max,
                result.simulations,
                nps,
                elapsed_ms,
                cp_from_q(selected_q),
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
            println!("bestmove 0000");
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
