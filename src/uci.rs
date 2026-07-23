use crate::az::{
    AzNnue, AzSearchControl, AzSearchLimits, AzSearchResult,
    alphazero_search_with_rules_controlled_with_progress,
};
use crate::xiangqi::{Color, Position, RuleHistoryEntry, RuleOutcome};
use std::io::{self, BufRead, Write};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const MAX_UCI_SIMULATIONS: usize = u32::MAX as usize - 1;
// MCTS 会保留整棵搜索树，`go infinite` 必须限制单棵树规模以免 GUI 长时间分析 OOM。
const MAX_UCI_INFINITE_TREE_SIMULATIONS: usize = 100_000;
const MAX_UCI_TIME_MS: u64 = 7 * 24 * 60 * 60 * 1_000;

#[derive(Clone, Debug)]
struct UciState {
    position: Position,
    rule_history: Vec<RuleHistoryEntry>,
    eval_file: String,
    model: Option<Arc<AzNnue>>,
    simulations: usize,
    threads: usize,
    cpuct: f32,
    cpuct_at_root: f32,
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
            rule_history: Position::startpos().initial_rule_history(),
            eval_file: "model.safetensors".into(),
            model: None,
            simulations: 10_000,
            threads: 1,
            cpuct: 1.5,
            cpuct_at_root: 3.0,
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

struct ActiveSearch {
    stop: Arc<AtomicBool>,
    handle: JoinHandle<()>,
}

impl ActiveSearch {
    fn stop_and_join(self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.handle.join();
    }
}

pub fn run_uci() {
    let stdin = io::stdin();
    let mut state = UciState::default();
    let mut active_search: Option<ActiveSearch> = None;
    for line in stdin.lock().lines() {
        let Ok(line) = line else {
            break;
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if active_search
            .as_ref()
            .is_some_and(|search| search.handle.is_finished())
        {
            let _ = active_search.take().unwrap().handle.join();
        }
        match line.split_whitespace().next() {
            Some("uci") => print_uci_id(),
            Some("isready") => {
                ensure_model(&mut state);
                println!("readyok");
                flush();
            }
            Some("ucinewgame") => {
                stop_active_search(&mut active_search);
                state.position = Position::startpos();
                state.rule_history = state.position.initial_rule_history();
                state.seed = 20260409;
            }
            Some("setoption") => {
                stop_active_search(&mut active_search);
                handle_setoption(line, &mut state);
            }
            Some("position") => {
                stop_active_search(&mut active_search);
                handle_position(line, &mut state);
            }
            Some("go") => {
                stop_active_search(&mut active_search);
                active_search = Some(start_go(line, &mut state));
            }
            Some("stop") => stop_active_search(&mut active_search),
            Some("quit") => {
                stop_active_search(&mut active_search);
                break;
            }
            _ => {}
        }
    }
    stop_active_search(&mut active_search);
}

fn stop_active_search(active_search: &mut Option<ActiveSearch>) {
    if let Some(search) = active_search.take() {
        search.stop_and_join();
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
    state.model = Some(Arc::new(AzNnue::load(&state.eval_file).unwrap_or_else(
        |err| {
            println!(
                "info string failed to load {}, using random model: {}",
                state.eval_file, err
            );
            flush();
            AzNnue::random(128, state.seed)
        },
    )));
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
        state.rule_history = state.position.initial_rule_history();
        if let Some(moves_index) = tokens.iter().position(|token| *token == "moves") {
            let move_list = &tokens[moves_index + 1..];
            apply_uci_moves(&mut state.position, &mut state.rule_history, move_list);
        }
        return;
    }

    if tokens.get(1) == Some(&"fen") {
        let moves_index = tokens.iter().position(|token| *token == "moves");
        let fen_end = moves_index.unwrap_or(tokens.len());
        let fen = tokens[2..fen_end].join(" ");
        if let Ok(position) = Position::from_fen(&fen) {
            state.position = position;
            state.rule_history = state.position.initial_rule_history();
            if let Some(moves_index) = moves_index {
                let move_list = &tokens[moves_index + 1..];
                apply_uci_moves(&mut state.position, &mut state.rule_history, move_list);
            }
        }
    }
}

fn apply_uci_moves(
    position: &mut Position,
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

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct GoParams {
    searchmoves: Vec<String>,
    wtime_ms: Option<u64>,
    btime_ms: Option<u64>,
    winc_ms: u64,
    binc_ms: u64,
    moves_to_go: Option<u64>,
    move_time_ms: Option<u64>,
    nodes: Option<usize>,
    depth: Option<usize>,
    infinite: bool,
}

fn parse_go(line: &str) -> GoParams {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    let mut params = GoParams::default();
    let mut index = 1usize;
    while index < tokens.len() {
        let token = tokens[index];
        match token {
            "searchmoves" => {
                index += 1;
                while index < tokens.len() && !is_go_keyword(tokens[index]) {
                    params.searchmoves.push(tokens[index].to_owned());
                    index += 1;
                }
                continue;
            }
            "wtime" => params.wtime_ms = parse_next(&tokens, index),
            "btime" => params.btime_ms = parse_next(&tokens, index),
            "winc" => params.winc_ms = parse_next(&tokens, index).unwrap_or(0),
            "binc" => params.binc_ms = parse_next(&tokens, index).unwrap_or(0),
            "movestogo" => params.moves_to_go = parse_next(&tokens, index),
            "movetime" => params.move_time_ms = parse_next(&tokens, index),
            "nodes" => params.nodes = parse_next(&tokens, index),
            "depth" => params.depth = parse_next(&tokens, index),
            "infinite" => {
                params.infinite = true;
                index += 1;
                continue;
            }
            _ => {
                index += 1;
                continue;
            }
        }
        index += 2;
    }
    params
}

fn parse_next<T: std::str::FromStr>(tokens: &[&str], index: usize) -> Option<T> {
    tokens.get(index + 1)?.parse().ok()
}

fn is_go_keyword(token: &str) -> bool {
    matches!(
        token,
        "searchmoves"
            | "ponder"
            | "wtime"
            | "btime"
            | "winc"
            | "binc"
            | "movestogo"
            | "depth"
            | "nodes"
            | "mate"
            | "movetime"
            | "infinite"
    )
}

fn time_budget_ms(params: &GoParams, side: Color) -> Option<u64> {
    if let Some(move_time_ms) = params.move_time_ms {
        return Some(move_time_ms.clamp(1, MAX_UCI_TIME_MS));
    }
    if params.infinite {
        return None;
    }
    let (remaining_ms, increment_ms) = match side {
        Color::Red => (params.wtime_ms?, params.winc_ms),
        Color::Black => (params.btime_ms?, params.binc_ms),
    };
    let usable_ms = remaining_ms.max(1);
    let moves = params.moves_to_go.unwrap_or(24).max(1);
    let target_ms = usable_ms / moves + increment_ms.saturating_mul(3) / 4;
    let maximum_ms = (usable_ms / 5).max(1);
    Some(target_ms.clamp(1, maximum_ms).min(MAX_UCI_TIME_MS))
}

fn start_go(line: &str, state: &mut UciState) -> ActiveSearch {
    ensure_model(state);
    let params = parse_go(line);
    let snapshot = state.clone();
    state.seed = state.seed.wrapping_add(1);
    let stop = Arc::new(AtomicBool::new(false));
    let search_stop = Arc::clone(&stop);
    let handle = thread::spawn(move || run_go_search(snapshot, params, search_stop));
    ActiveSearch { stop, handle }
}

fn run_go_search(state: UciState, params: GoParams, stop: Arc<AtomicBool>) {
    let model = state.model.as_ref().expect("model was loaded");

    if position_is_rule_draw(&state.position, &state.rule_history) {
        println!("info depth 0 nodes 0 time 0 score cp 0");
        println!("bestmove 0000");
        flush();
        return;
    }

    let mut legal = state.position.legal_moves_with_rules(&state.rule_history);
    if !params.searchmoves.is_empty() {
        legal.retain(|mv| {
            params
                .searchmoves
                .iter()
                .any(|text| state.position.parse_uci_move(text) == Some(*mv))
        });
    }

    if legal.is_empty() {
        println!("info depth 1 nodes 0 time 0 score cp -32000");
        println!("bestmove 0000");
        flush();
        return;
    }

    let budget_ms = time_budget_ms(&params, state.position.side_to_move());
    let has_time_control = budget_ms.is_some() || params.infinite;
    let simulations = uci_simulation_limit(&params, state.simulations, has_time_control);
    let started = Instant::now();
    let deadline = budget_ms.map(|budget| started + Duration::from_millis(budget));
    let control = AzSearchControl::new(Arc::clone(&stop), deadline);
    let mut report_progress = |progress: &AzSearchResult| {
        print_search_info(progress, started);
        flush();
    };
    let result = alphazero_search_with_rules_controlled_with_progress(
        &state.position,
        Some(state.rule_history.clone()),
        Some(legal),
        model,
        AzSearchLimits {
            simulations,
            seed: state.seed,
            cpuct: state.cpuct,
            cpuct_at_root: state.cpuct_at_root,
            cpuct_base: state.cpuct_base,
            cpuct_factor: state.cpuct_factor,
            cpuct_base_at_root: state.cpuct_base_at_root,
            cpuct_factor_at_root: state.cpuct_factor_at_root,
            max_depth: params.depth.unwrap_or(0),
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
        },
        Some(&control),
        Some(&mut report_progress),
    );
    // UCI 规定无限分析在收到 `stop` 前不发送 bestmove。树达到安全上限后保留结果等待，
    // 不再继续无界分配节点。
    while params.infinite && !stop.load(Ordering::Relaxed) {
        thread::park_timeout(Duration::from_millis(10));
    }
    match result.best_move {
        Some(mv) => {
            let best_text = mv.to_string();
            print_search_info(&result, started);
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

fn uci_simulation_limit(params: &GoParams, configured: usize, has_time_control: bool) -> usize {
    let requested = params.nodes.unwrap_or(if params.infinite {
        configured.max(1)
    } else if has_time_control {
        MAX_UCI_SIMULATIONS
    } else {
        configured.max(1)
    });
    if params.infinite {
        requested.clamp(1, MAX_UCI_INFINITE_TREE_SIMULATIONS)
    } else {
        requested.clamp(1, MAX_UCI_SIMULATIONS)
    }
}

fn print_search_info(result: &AzSearchResult, started: Instant) {
    let elapsed_ms = started.elapsed().as_millis();
    let nps = (result.simulations as u128 * 1000 / elapsed_ms.max(1)) as usize;
    let wdl = uci_wdl(result.value_wdl);
    match result.best_move {
        Some(mv) => println!(
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
            mv
        ),
        None => println!(
            "info depth {} seldepth {} nodes {} nps {} time {} score cp {} wdl {} {} {}",
            result.search_depth_avg.round() as usize,
            result.search_depth_max,
            result.simulations,
            nps,
            elapsed_ms,
            result.value_cp,
            wdl[0],
            wdl[1],
            wdl[2]
        ),
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_standard_go_time_and_search_limits() {
        let params = parse_go(
            "go searchmoves a0a1 b0b1 wtime 60000 btime 50000 winc 1000 binc 500 \
             movestogo 20 nodes 1234 depth 12",
        );

        assert_eq!(params.searchmoves, ["a0a1", "b0b1"]);
        assert_eq!(params.wtime_ms, Some(60_000));
        assert_eq!(params.btime_ms, Some(50_000));
        assert_eq!(params.winc_ms, 1_000);
        assert_eq!(params.binc_ms, 500);
        assert_eq!(params.moves_to_go, Some(20));
        assert_eq!(params.nodes, Some(1_234));
        assert_eq!(params.depth, Some(12));
    }

    #[test]
    fn movetime_uses_exact_budget_and_clock_budget_is_bounded() {
        let move_time = parse_go("go movetime 1000");
        assert_eq!(time_budget_ms(&move_time, Color::Red), Some(1_000));

        let clock = parse_go("go wtime 60000 btime 30000 winc 1000 binc 0 movestogo 20");
        assert_eq!(time_budget_ms(&clock, Color::Red), Some(3_750));
        assert_eq!(time_budget_ms(&clock, Color::Black), Some(1_500));

        let infinite = parse_go("go infinite");
        assert_eq!(time_budget_ms(&infinite, Color::Red), None);
    }

    #[test]
    fn infinite_analysis_uses_a_bounded_tree() {
        let infinite = parse_go("go infinite");
        assert_eq!(uci_simulation_limit(&infinite, 10_000, true), 10_000);

        let oversized = parse_go("go infinite nodes 100000000");
        assert_eq!(
            uci_simulation_limit(&oversized, 10_000, true),
            MAX_UCI_INFINITE_TREE_SIMULATIONS
        );

        let timed = parse_go("go movetime 1000");
        assert_eq!(
            uci_simulation_limit(&timed, 10_000, true),
            MAX_UCI_SIMULATIONS
        );
    }
}
