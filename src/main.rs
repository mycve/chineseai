use chineseai::{
    az::{AzLoopConfig, AzNnue, AzSearchLimits, gumbel_search, selfplay_train_iteration},
    nnue::{HISTORY_PLIES, HistoryMove},
    xiangqi::{Position, STARTPOS_FEN},
};
use std::io::{self, BufRead, Write};

fn main() {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        None => run_uci(),
        Some("uci") => run_uci(),
        Some("az-init") => {
            let hidden = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(128);
            let output = args.next().unwrap_or_else(|| "chineseai.nnue.txt".into());
            let seed = args
                .next()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(20260409);
            let trunk_depth = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(2);
            let model = AzNnue::random_with_depth(hidden, trunk_depth, seed);
            model.save_text(&output).unwrap_or_else(|err| {
                panic!("failed to write `{output}`: {err}");
            });
            println!("aznnue   : initialized");
            println!("hidden   : {hidden}");
            println!("depth    : {trunk_depth}");
            println!("seed     : {seed}");
            println!("format   : aznnue-v3");
            println!("output   : {output}");
        }
        Some("az-gumbel") => {
            let model_path = args
                .next()
                .unwrap_or_else(|| panic!("usage: az-gumbel <model> [simulations] [top_k] [fen]"));
            let simulations = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(10_000);
            let top_k = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(32);
            let fen = args.collect::<Vec<_>>().join(" ");
            let position = parse_position(&fen);
            let model = AzNnue::load_text(&model_path).unwrap_or_else(|err| {
                panic!("failed to load `{model_path}`: {err}");
            });
            let result = gumbel_search(
                &position,
                &model,
                AzSearchLimits {
                    simulations,
                    top_k,
                    seed: 0,
                    gumbel_scale: 1.0,
                    workers: 1,
                },
            );
            println!("fen      : {}", position.to_fen());
            println!("model    : {model_path}");
            println!("sims     : {}", result.simulations);
            println!("top_k    : {top_k}");
            println!("value_cp : {}", result.value_cp);
            println!(
                "bestmove : {}",
                result
                    .best_move
                    .map(|mv| mv.to_string())
                    .unwrap_or_else(|| "(none)".into())
            );
            for candidate in result.candidates.iter().take(top_k) {
                println!(
                    "candidate: {} visits={} q={:.3} prior={:.5} policy={:.5}",
                    candidate.mv, candidate.visits, candidate.q, candidate.prior, candidate.policy
                );
            }
        }
        Some("az-loop") => {
            let model_path = args.next().unwrap_or_else(|| "chineseai.nnue.txt".into());
            let iterations = parse_next(&mut args, 100usize);
            let games = parse_next(&mut args, 16usize);
            let simulations = parse_next(&mut args, 256usize);
            let top_k = parse_next(&mut args, 16usize);
            let epochs = parse_next(&mut args, 1usize);
            let lr = parse_next(&mut args, 0.001f32);
            let max_plies = parse_next(&mut args, 160usize);
            let hidden = parse_next(&mut args, 128usize);
            let seed = parse_next(&mut args, 20260409u64);
            let workers = parse_next(&mut args, 1usize).max(1);
            let temperature_start = parse_next(&mut args, 1.0f32);
            let temperature_end = parse_next(&mut args, 0.2f32);
            let temperature_decay_plies = parse_next(&mut args, 40usize);
            let gumbel_scale = parse_next(&mut args, 1.0f32);
            let trunk_depth = parse_next(&mut args, 2usize);

            let mut model = if std::path::Path::new(&model_path).exists() {
                println!("model    : load {model_path}");
                AzNnue::load_text(&model_path).unwrap_or_else(|err| {
                    panic!("failed to load `{model_path}`: {err}");
                })
            } else {
                println!("model    : init {model_path}");
                AzNnue::random_with_depth(hidden, trunk_depth, seed)
            };

            println!(
                "loop     : iterations={iterations} games={games} sims={simulations} top_k={top_k} epochs={epochs} lr={lr} max_plies={max_plies} workers={workers} temp={temperature_start}->{temperature_end}/{temperature_decay_plies}ply gumbel_scale={gumbel_scale} depth={trunk_depth}"
            );
            for iteration in 1..=iterations {
                let started = std::time::Instant::now();
                let report = selfplay_train_iteration(
                    &mut model,
                    &AzLoopConfig {
                        games,
                        max_plies,
                        simulations,
                        top_k,
                        epochs,
                        lr,
                        seed: seed ^ iteration as u64,
                        workers,
                        temperature_start,
                        temperature_end,
                        temperature_decay_plies,
                        gumbel_scale,
                    },
                );
                model.save_text(&model_path).unwrap_or_else(|err| {
                    panic!("failed to write `{model_path}`: {err}");
                });
                println!(
                    "iter {iteration:04}: games={} samples={} W/B/D={}/{}/{} avg_plies={:.1} loss={:.4} value_mse={:.4} policy_ce={:.4} elapsed={:.1}s saved={}",
                    report.games,
                    report.samples,
                    report.red_wins,
                    report.black_wins,
                    report.draws,
                    report.avg_plies,
                    report.loss,
                    report.value_mse,
                    report.policy_ce,
                    started.elapsed().as_secs_f32(),
                    model_path
                );
            }
        }
        Some("perft") => {
            let depth = args
                .next()
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or(1);
            let fen = args.collect::<Vec<_>>().join(" ");
            let position = parse_position(&fen);
            println!("fen   : {}", position.to_fen());
            println!("depth : {depth}");
            println!("nodes : {}", position.perft(depth));
        }
        Some("help") | _ => print_help(),
    }
}

fn print_help() {
    let position = Position::startpos();
    println!("ChineseAI AZ-NNUE Gumbel core");
    println!("start : {STARTPOS_FEN}");
    println!("moves : {}", position.legal_moves().len());
    println!("hint  : cargo run --release -- az-init 128 chineseai.nnue.txt 20260409 2");
    println!("hint  : cargo run --release -- uci");
    println!("hint  : cargo run --release -- az-gumbel chineseai.nnue.txt 10000 32 startpos");
    println!(
        "hint  : cargo run --release -- az-loop chineseai.nnue.txt 100 16 256 16 1 0.001 160 128 20260409 4 1.0 0.2 40 1.0 2"
    );
}

#[derive(Clone, Debug)]
struct UciState {
    position: Position,
    history: Vec<HistoryMove>,
    eval_file: String,
    model: Option<AzNnue>,
    simulations: usize,
    top_k: usize,
    threads: usize,
    gumbel_scale: f32,
    seed: u64,
}

impl Default for UciState {
    fn default() -> Self {
        Self {
            position: Position::startpos(),
            history: Vec::new(),
            eval_file: "chineseai.nnue.txt".into(),
            model: None,
            simulations: 10_000,
            top_k: 32,
            threads: 1,
            gumbel_scale: 0.0,
            seed: 20260409,
        }
    }
}

fn run_uci() {
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
                ensure_uci_model(&mut state);
                println!("readyok");
                uci_flush();
            }
            Some("ucinewgame") => {
                state.position = Position::startpos();
                state.history.clear();
            }
            Some("setoption") => handle_setoption(line, &mut state),
            Some("position") => handle_position(line, &mut state),
            Some("go") => handle_go(line, &mut state),
            Some("stop") => {
                println!("bestmove 0000");
                uci_flush();
            }
            Some("quit") => break,
            _ => {}
        }
    }
}

fn print_uci_id() {
    println!("id name ChineseAI AZ-NNUE");
    println!("id author ChineseAI");
    println!("option name EvalFile type string default chineseai.nnue.txt");
    println!("option name Simulations type spin default 10000 min 1 max 100000000");
    println!("option name TopK type spin default 32 min 1 max 256");
    println!("option name Threads type spin default 1 min 1 max 1024");
    println!("option name GumbelScale type string default 0.0");
    println!("uciok");
    uci_flush();
}

fn ensure_uci_model(state: &mut UciState) {
    if state.model.is_some() {
        return;
    }
    state.model = Some(AzNnue::load_text(&state.eval_file).unwrap_or_else(|err| {
        println!(
            "info string failed to load {}, using random model: {}",
            state.eval_file, err
        );
        uci_flush();
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
        "topk" => {
            state.top_k = value.parse::<usize>().unwrap_or(state.top_k).max(1);
        }
        "threads" => {
            state.threads = value.parse::<usize>().unwrap_or(state.threads).max(1);
        }
        "gumbelscale" => {
            state.gumbel_scale = value.parse::<f32>().unwrap_or(state.gumbel_scale).max(0.0);
        }
        _ => {}
    }
}

fn handle_position(line: &str, state: &mut UciState) {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    if tokens.get(1) == Some(&"startpos") {
        state.position = Position::startpos();
        state.history.clear();
        if let Some(moves_index) = tokens.iter().position(|token| *token == "moves") {
            apply_uci_moves(
                &mut state.position,
                &mut state.history,
                &tokens[moves_index + 1..],
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
            if let Some(moves_index) = moves_index {
                apply_uci_moves(
                    &mut state.position,
                    &mut state.history,
                    &tokens[moves_index + 1..],
                );
            }
        } else {
            eprintln!("info string invalid fen: {fen}");
        }
    }
}

fn apply_uci_moves(position: &mut Position, history: &mut Vec<HistoryMove>, moves: &[&str]) {
    for text in moves {
        let Some(mv) = position.parse_uci_move(text) else {
            eprintln!("info string illegal move ignored: {text}");
            break;
        };
        if let Some(piece) = position.piece_at(mv.from as usize) {
            history.push(HistoryMove { piece, mv });
            let overflow = history.len().saturating_sub(HISTORY_PLIES);
            if overflow > 0 {
                history.drain(0..overflow);
            }
        }
        position.make_move(mv);
    }
}

fn handle_go(line: &str, state: &mut UciState) {
    ensure_uci_model(state);
    let simulations = go_simulations(line, state.simulations);
    let model = state.model.as_ref().expect("model was loaded");
    let started = std::time::Instant::now();
    let result = chineseai::az::gumbel_search_with_history(
        &state.position,
        &state.history,
        model,
        AzSearchLimits {
            simulations,
            top_k: state.top_k,
            seed: state.seed,
            gumbel_scale: state.gumbel_scale,
            workers: state.threads,
        },
    );
    state.seed = state.seed.wrapping_add(1);
    println!(
        "info depth 1 nodes {} time {} score cp {} pv {}",
        result.simulations,
        started.elapsed().as_millis(),
        result.value_cp,
        result
            .best_move
            .map(|mv| mv.to_string())
            .unwrap_or_else(|| "0000".into())
    );
    println!(
        "bestmove {}",
        result
            .best_move
            .map(|mv| mv.to_string())
            .unwrap_or_else(|| "0000".into())
    );
    uci_flush();
}

fn go_simulations(line: &str, default_simulations: usize) -> usize {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    if let Some(nodes_index) = tokens.iter().position(|token| *token == "nodes") {
        return tokens
            .get(nodes_index + 1)
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(default_simulations)
            .max(1);
    }
    if let Some(movetime_index) = tokens.iter().position(|token| *token == "movetime") {
        let movetime = tokens
            .get(movetime_index + 1)
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1000);
        return default_simulations.max((movetime * 64).max(1));
    }
    default_simulations.max(1)
}

fn uci_flush() {
    let _ = io::stdout().flush();
}

fn parse_position(text: &str) -> Position {
    if text.trim().is_empty() || text == "startpos" {
        Position::startpos()
    } else {
        Position::from_fen(text).unwrap_or_else(|err| {
            panic!("invalid FEN `{text}`: {err}");
        })
    }
}

fn parse_next<T: std::str::FromStr>(args: &mut impl Iterator<Item = String>, default: T) -> T {
    args.next()
        .and_then(|value| value.parse::<T>().ok())
        .unwrap_or(default)
}
