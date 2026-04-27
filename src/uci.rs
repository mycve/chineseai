use crate::az::{
    AzGumbelConfig, AzNnue, AzSearchAlgorithm, AzSearchLimits,
    alphazero_search_with_history_and_rules,
};
use crate::board_transform::{HISTORY_PLIES, HistoryMove};
use crate::xiangqi::{Position, RuleHistoryEntry, RuleOutcome};
use std::fs;
use std::io::{self, BufRead, BufWriter, Write};

struct UciLogger {
    file: Option<BufWriter<fs::File>>,
    elapsed: std::time::Instant,
}

impl UciLogger {
    fn new() -> Self {
        let path = std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(|dir| dir.join("chineseai-uci.log")));

        let file = path.and_then(|p| {
            fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
                .ok()
                .map(BufWriter::new)
        });

        Self {
            file,
            elapsed: std::time::Instant::now(),
        }
    }

    fn log(&mut self, msg: &str) {
        let Some(f) = self.file.as_mut() else { return };
        let ms = self.elapsed.elapsed().as_millis();
        let _ = writeln!(f, "[{ms:>8}ms] {msg}");
        let _ = f.flush();
    }
}

macro_rules! ulog {
    ($logger:expr, $($arg:tt)*) => {
        $logger.log(&format!($($arg)*))
    };
}

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
    search_algorithm: AzSearchAlgorithm,
    gumbel: AzGumbelConfig,
    policy_debug: bool,
    policy_debug_limit: usize,
    seed: u64,
}

impl Default for UciState {
    fn default() -> Self {
        Self {
            position: Position::startpos(),
            history: Vec::new(),
            rule_history: Position::startpos().initial_rule_history(),
            eval_file: "chineseai.nnue".into(),
            model: None,
            simulations: 10_000,
            threads: 1,
            cpuct: 1.5,
            search_algorithm: AzSearchAlgorithm::AlphaZero,
            gumbel: AzGumbelConfig::default(),
            policy_debug: false,
            policy_debug_limit: 16,
            seed: 20260409,
        }
    }
}

pub fn run_uci() {
    let stdin = io::stdin();
    let mut state = UciState::default();
    let mut logger = UciLogger::new();
    ulog!(logger, "=== UCI session started ===");
    for line in stdin.lock().lines() {
        let Ok(line) = line else {
            break;
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        ulog!(logger, ">> {line}");
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
                ulog!(logger, "[ucinewgame] reset to startpos");
            }
            Some("setoption") => handle_setoption(line, &mut state),
            Some("position") => handle_position(line, &mut state, &mut logger),
            Some("go") => handle_go(&mut state, &mut logger),
            Some("stop") => {}
            Some("quit") => {
                ulog!(logger, "=== UCI session ended ===");
                break;
            }
            _ => {}
        }
    }
}

fn print_uci_id() {
    println!("id name ChineseAI AZ");
    println!("id author ChineseAI");
    println!("option name EvalFile type string default chineseai.nnue");
    println!("option name Simulations type spin default 10000 min 1 max 100000000");
    println!("option name Threads type spin default 1 min 1 max 1");
    println!("option name Cpuct type string default 1.5");
    println!(
        "option name SearchAlgorithm type combo default alphazero var alphazero var gumbel_alphazero"
    );
    println!("option name GumbelMaxActions type spin default 16 min 1 max 512");
    println!("option name GumbelScale type string default 1.0");
    println!("option name PolicyDebug type check default false");
    println!("option name PolicyDebugLimit type spin default 16 min 1 max 256");
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
        "searchalgorithm" => {
            if let Some(algorithm) = AzSearchAlgorithm::parse(&value) {
                state.search_algorithm = algorithm;
            }
        }
        "gumbelmaxactions" => {
            state.gumbel.max_num_considered_actions = value
                .parse::<usize>()
                .unwrap_or(state.gumbel.max_num_considered_actions)
                .clamp(1, 512);
        }
        "gumbelscale" => {
            state.gumbel.gumbel_scale = value
                .parse::<f32>()
                .unwrap_or(state.gumbel.gumbel_scale)
                .max(0.0);
        }
        "policydebug" => {
            state.policy_debug = matches!(value.to_ascii_lowercase().as_str(), "true" | "1" | "on");
        }
        "policydebuglimit" => {
            state.policy_debug_limit = value
                .parse::<usize>()
                .unwrap_or(state.policy_debug_limit)
                .clamp(1, 256);
        }
        _ => {}
    }
}

fn handle_position(line: &str, state: &mut UciState, logger: &mut UciLogger) {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    if tokens.get(1) == Some(&"startpos") {
        state.position = Position::startpos();
        state.history.clear();
        state.rule_history = state.position.initial_rule_history();
        if let Some(moves_index) = tokens.iter().position(|token| *token == "moves") {
            let move_list = &tokens[moves_index + 1..];
            ulog!(logger, "[position] startpos moves={}", move_list.join(" "));
            apply_uci_moves(
                &mut state.position,
                &mut state.history,
                &mut state.rule_history,
                move_list,
                logger,
            );
        } else {
            ulog!(logger, "[position] startpos (no moves)");
        }
        ulog!(
            logger,
            "[position] result fen={} halfmove_clock={}",
            state.position.to_fen(),
            state.position.halfmove_clock()
        );
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
                ulog!(logger, "[position] fen={fen} moves={}", move_list.join(" "));
                apply_uci_moves(
                    &mut state.position,
                    &mut state.history,
                    &mut state.rule_history,
                    move_list,
                    logger,
                );
            } else {
                ulog!(logger, "[position] fen={fen} (no moves)");
            }
            ulog!(
                logger,
                "[position] result fen={} halfmove_clock={}",
                state.position.to_fen(),
                state.position.halfmove_clock()
            );
        } else {
            eprintln!("info string invalid fen: {fen}");
            ulog!(logger, "[position] ERROR invalid fen={fen}");
        }
    }
}

fn apply_uci_moves(
    position: &mut Position,
    history: &mut Vec<HistoryMove>,
    rule_history: &mut Vec<RuleHistoryEntry>,
    moves: &[&str],
    logger: &mut UciLogger,
) {
    for (i, text) in moves.iter().enumerate() {
        let Some(mv) = position.parse_uci_move(text) else {
            eprintln!("info string illegal move ignored: {text}");
            ulog!(
                logger,
                "[apply_move] #{i} {text} ILLEGAL fen={} halfmove_clock={}",
                position.to_fen(),
                position.halfmove_clock()
            );
            break;
        };
        if let Some(piece) = position.piece_at(mv.from as usize) {
            history.push(HistoryMove { piece, mv });
            let overflow = history.len().saturating_sub(HISTORY_PLIES);
            if overflow > 0 {
                history.drain(0..overflow);
            }
        }
        rule_history.push(position.rule_history_entry_after_move(mv));
        position.make_move(mv);
        ulog!(
            logger,
            "[apply_move] #{i} {text} ok fen={}",
            position.to_fen()
        );
    }
}

fn position_is_rule_draw(position: &Position, rule_history: &[RuleHistoryEntry]) -> bool {
    matches!(
        position.rule_outcome_with_history(rule_history),
        Some(RuleOutcome::Draw(_))
    )
}

fn handle_go(state: &mut UciState, logger: &mut UciLogger) {
    ensure_model(state);
    let simulations = state.simulations.max(1);
    let model = state.model.as_ref().expect("model was loaded");
    let fen = state.position.to_fen();

    if position_is_rule_draw(&state.position, &state.rule_history) {
        ulog!(logger, "[go] rule draw fen={fen} (no search)");
        println!("info depth 0 nodes 0 time 0 score cp 0 pv draw");
        println!("bestmove draw");
        flush();
        return;
    }

    let raw_legal = state.position.legal_moves();
    let legal = state.position.legal_moves_with_rules(&state.rule_history);

    ulog!(
        logger,
        "[go] fen={fen} halfmove_clock={} rule_history_len={} raw_legal={} filtered_legal={}",
        state.position.halfmove_clock(),
        state.rule_history.len(),
        raw_legal.len(),
        legal.len()
    );

    if legal.is_empty() {
        ulog!(logger, "[no_legal_moves] fen={fen}");
        ulog!(
            logger,
            "[no_legal_moves] raw_legal_moves({})={}",
            raw_legal.len(),
            raw_legal
                .iter()
                .map(|mv| mv.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        );
        ulog!(
            logger,
            "[no_legal_moves] rule_history ({} entries):",
            state.rule_history.len()
        );
        for (i, entry) in state.rule_history.iter().enumerate() {
            ulog!(
                logger,
                "[no_legal_moves]   [{i:>3}] hash={:#018x} side={:?} mover={:?} check={} chase_mask={:#034x}",
                entry.hash,
                entry.side_to_move,
                entry.mover,
                entry.gives_check,
                entry.chased_mask
            );
        }
        println!("info depth 1 nodes 0 time 0 score cp -32000");
        flush();
        return;
    }

    let started = std::time::Instant::now();
    let result = alphazero_search_with_history_and_rules(
        &state.position,
        &state.history,
        Some(state.rule_history.clone()),
        Some(legal),
        model,
        AzSearchLimits {
            simulations,
            seed: state.seed,
            cpuct: state.cpuct,
            root_dirichlet_alpha: 0.0,
            root_exploration_fraction: 0.0,
            algorithm: state.search_algorithm,
            gumbel: state.gumbel,
        },
    );
    state.seed = state.seed.wrapping_add(1);
    if state.policy_debug {
        print_policy_debug(&result, state.policy_debug_limit);
    }
    match result.best_move {
        Some(mv) => {
            let best_text = mv.to_string();
            println!(
                "info depth 1 nodes {} time {} score cp {} pv {}",
                result.simulations,
                started.elapsed().as_millis(),
                result.value_cp,
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

fn print_policy_debug(result: &crate::az::AzSearchResult, limit: usize) {
    let visited_actions = result
        .candidates
        .iter()
        .filter(|candidate| candidate.visits > 0)
        .count();
    println!("info string policy_debug visited_actions {visited_actions}");
    println!("info string policy_debug columns move visits q raw_policy searched_policy");
    for candidate in result.candidates.iter().take(limit) {
        println!(
            "info string policy_debug {} {} {:.4} {:.6} {:.6}",
            candidate.mv, candidate.visits, candidate.q, candidate.prior, candidate.policy
        );
    }
    println!("info string policy_debug_by_visits columns move visits q raw_policy searched_policy");
    let mut by_visits = result.candidates.clone();
    by_visits.sort_by(|left, right| {
        right
            .visits
            .cmp(&left.visits)
            .then_with(|| right.policy.total_cmp(&left.policy))
            .then_with(|| right.q.total_cmp(&left.q))
    });
    for candidate in by_visits.iter().take(limit) {
        println!(
            "info string policy_debug_by_visits {} {} {:.4} {:.6} {:.6}",
            candidate.mv, candidate.visits, candidate.q, candidate.prior, candidate.policy
        );
    }
}

fn flush() {
    let _ = io::stdout().flush();
}
