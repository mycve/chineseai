use crate::az::{
    AzNnue, AzSearchControl, AzSearchLimits, AzSearchResult,
    alphazero_search_external_root_controlled_with_progress,
};
use crate::xiangqi::{Color, Move, Position, RuleHistoryEntry};
use std::io::{self, BufRead, Write};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const MAX_UCI_SIMULATIONS: usize = u32::MAX as usize - 1;
// MCTS 会保留整棵搜索树，`go infinite` 必须限制单棵树规模以免 GUI 长时间分析 OOM。
const MAX_UCI_TIME_MS: u64 = 7 * 24 * 60 * 60 * 1_000;
const DEFAULT_SIMULATIONS: usize = 10_000;
const DEFAULT_CPUCT: f32 = 0.9;
const DEFAULT_CPUCT_AT_ROOT: f32 = 2.0;
const DEFAULT_CPUCT_BASE: f32 = 19_652.0;
const DEFAULT_CPUCT_FACTOR: f32 = 1.5;
const DEFAULT_FPU_VALUE: f32 = 0.20;
const DEFAULT_FPU_VALUE_AT_ROOT: f32 = 0.10;
const DEFAULT_POLICY_SOFTMAX_TEMP: f32 = 1.2;

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
    policy_softmax_temp: f32,
    draw_score: f32,
    sixty_move_rule: bool,
    rule60_max_ply: u16,
    seed: u64,
}

impl Default for UciState {
    fn default() -> Self {
        Self {
            position: Position::startpos(),
            rule_history: Position::startpos().initial_rule_history(),
            eval_file: "model.safetensors".into(),
            model: None,
            simulations: DEFAULT_SIMULATIONS,
            threads: 1,
            cpuct: DEFAULT_CPUCT,
            cpuct_at_root: DEFAULT_CPUCT_AT_ROOT,
            cpuct_base: DEFAULT_CPUCT_BASE,
            cpuct_factor: DEFAULT_CPUCT_FACTOR,
            cpuct_base_at_root: DEFAULT_CPUCT_BASE,
            cpuct_factor_at_root: DEFAULT_CPUCT_FACTOR,
            fpu_value: DEFAULT_FPU_VALUE,
            fpu_value_at_root: DEFAULT_FPU_VALUE_AT_ROOT,
            policy_softmax_temp: DEFAULT_POLICY_SOFTMAX_TEMP,
            draw_score: 0.0,
            sixty_move_rule: true,
            rule60_max_ply: 120,
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
                apply_rule_options(&mut state);
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
    println!("option name Simulations type spin default {DEFAULT_SIMULATIONS} min 1 max 100000000");
    println!("option name Threads type spin default 1 min 1 max 1");
    println!("option name Cpuct type string default {DEFAULT_CPUCT}");
    println!("option name CpuctAtRoot type string default {DEFAULT_CPUCT_AT_ROOT}");
    println!("option name CpuctBase type string default {DEFAULT_CPUCT_BASE}");
    println!("option name CpuctFactor type string default {DEFAULT_CPUCT_FACTOR}");
    println!("option name CpuctBaseAtRoot type string default {DEFAULT_CPUCT_BASE}");
    println!("option name CpuctFactorAtRoot type string default {DEFAULT_CPUCT_FACTOR}");
    println!("option name FpuValue type string default {DEFAULT_FPU_VALUE}");
    println!("option name FpuValueAtRoot type string default {DEFAULT_FPU_VALUE_AT_ROOT}");
    println!("option name PolicySoftmaxTemp type string default {DEFAULT_POLICY_SOFTMAX_TEMP}");
    println!("option name DrawScore type string default 0.0");
    println!("option name Sixty Move Rule type check default true");
    println!("option name Rule60MaxPly type spin default 120 min 1 max 150");
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
                .max(0.0);
        }
        "policysoftmaxtemp" => {
            state.policy_softmax_temp = value
                .parse::<f32>()
                .unwrap_or(state.policy_softmax_temp)
                .max(1.0e-3);
        }
        "drawscore" => {
            state.draw_score = value
                .parse::<f32>()
                .unwrap_or(state.draw_score)
                .clamp(-1.0, 1.0);
        }
        "sixty move rule" => {
            state.sixty_move_rule = value.eq_ignore_ascii_case("true");
            apply_rule_options(state);
        }
        "rule60maxply" => {
            state.rule60_max_ply = value
                .parse::<u16>()
                .unwrap_or(state.rule60_max_ply)
                .clamp(1, 150);
            apply_rule_options(state);
        }
        _ => {}
    }
}

fn apply_rule_options(state: &mut UciState) {
    state
        .position
        .set_rule60_max_ply(state.sixty_move_rule.then_some(state.rule60_max_ply));
}

fn handle_position(line: &str, state: &mut UciState) {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    if tokens.get(1) == Some(&"startpos") {
        state.position = Position::startpos();
        apply_rule_options(state);
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
            apply_rule_options(state);
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
        // `position ... moves` is the external controller's authoritative game
        // history. Accept every board-legal move even if its tournament rule set
        // differs from ours; our repetition rules only guide future search moves.
        if !position.legal_moves().contains(&mv) {
            break;
        }
        rule_history.push(position.rule_history_entry_after_move(mv));
        position.make_move(mv);
    }
}

fn uci_root_moves(position: &Position, rule_history: &[RuleHistoryEntry]) -> Vec<Move> {
    let filtered = position.legal_moves_with_rules(rule_history);
    if filtered.is_empty() {
        position.legal_moves()
    } else {
        filtered
    }
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

    let mut legal = uci_root_moves(&state.position, &state.rule_history);
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
    println!(
        "info string searchparams cpuct={:.4}/{:.4} base={:.1}/{:.1} factor={:.4}/{:.4} fpu={:.4}/{:.4} policytemp={:.4}",
        state.cpuct,
        state.cpuct_at_root,
        state.cpuct_base,
        state.cpuct_base_at_root,
        state.cpuct_factor,
        state.cpuct_factor_at_root,
        state.fpu_value,
        state.fpu_value_at_root,
        state.policy_softmax_temp
    );
    flush();
    let mut report_progress = |progress: &AzSearchResult| {
        print_search_info(progress, started);
        flush();
    };
    let result = alphazero_search_external_root_controlled_with_progress(
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
            policy_softmax_temp: state.policy_softmax_temp,
            draw_score: state.draw_score,
            value_scale: 1.0,
        },
        Some(&control),
        Some(&mut report_progress),
    );
    // UCI 规定无限分析在收到 `stop` 前不发送 bestmove。只有达到内部 u32
    // 节点索引的表示上限时才会进入等待；正常分析不会受配置节点数限制。
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
        MAX_UCI_SIMULATIONS
    } else if has_time_control {
        MAX_UCI_SIMULATIONS
    } else {
        configured.max(1)
    });
    requested.clamp(1, MAX_UCI_SIMULATIONS)
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
    fn state_defaults_use_the_single_uci_default_source() {
        let state = UciState::default();
        assert_eq!(state.simulations, DEFAULT_SIMULATIONS);
        assert_eq!(state.cpuct, DEFAULT_CPUCT);
        assert_eq!(state.cpuct_at_root, DEFAULT_CPUCT_AT_ROOT);
        assert_eq!(state.cpuct_base, DEFAULT_CPUCT_BASE);
        assert_eq!(state.cpuct_factor, DEFAULT_CPUCT_FACTOR);
        assert_eq!(state.cpuct_base_at_root, DEFAULT_CPUCT_BASE);
        assert_eq!(state.cpuct_factor_at_root, DEFAULT_CPUCT_FACTOR);
        assert_eq!(state.fpu_value, DEFAULT_FPU_VALUE);
        assert_eq!(state.fpu_value_at_root, DEFAULT_FPU_VALUE_AT_ROOT);
        assert_eq!(state.policy_softmax_temp, DEFAULT_POLICY_SOFTMAX_TEMP);
    }

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
    fn infinite_analysis_runs_until_stop_or_explicit_nodes() {
        let infinite = parse_go("go infinite");
        assert_eq!(
            uci_simulation_limit(&infinite, 10_000, true),
            MAX_UCI_SIMULATIONS
        );

        let explicit = parse_go("go infinite nodes 100000000");
        assert_eq!(uci_simulation_limit(&explicit, 10_000, true), 100_000_000);

        let timed = parse_go("go movetime 1000");
        assert_eq!(
            uci_simulation_limit(&timed, 10_000, true),
            MAX_UCI_SIMULATIONS
        );
    }

    #[test]
    fn policy_softmax_temperature_is_configurable() {
        let mut state = UciState::default();
        handle_setoption("setoption name PolicySoftmaxTemp value 1.5", &mut state);
        assert_eq!(state.policy_softmax_temp, 1.5);
    }

    #[test]
    fn natural_move_limit_is_configurable() {
        let mut state = UciState::default();
        handle_setoption("setoption name Rule60MaxPly value 80", &mut state);
        assert_eq!(state.position.rule60_max_ply(), Some(80));
        handle_setoption("setoption name Sixty Move Rule value false", &mut state);
        assert_eq!(state.position.rule60_max_ply(), None);
        handle_position("position startpos", &mut state);
        assert_eq!(state.position.rule60_max_ply(), None);
    }

    #[test]
    fn uci_import_accepts_external_repeated_long_check() {
        let mut position = Position::from_fen(
            "2Rakab2/8r/4c1n2/p3p1p1p/2p6/9/P3P3P/1CN1NC3/9/1RBAKArc1 b - - 0 1",
        )
        .unwrap();
        let mut history = position.initial_rule_history();
        let moves = ["g0g1", "f0e1", "g1g0", "e1f0", "g0g1"];
        apply_uci_moves(&mut position, &mut history, &moves);

        assert_eq!(history.len(), moves.len() + 1);
        assert_eq!(
            position.rule_outcome_with_history(&history),
            Some(crate::xiangqi::RuleOutcome::Win(Color::Red))
        );
        assert_eq!(position.side_to_move(), Color::Red);
        assert!(!uci_root_moves(&position, &history).is_empty());
    }

    #[test]
    fn uci_import_accepts_external_repeated_long_chase() {
        let mut position =
            Position::from_fen("2bak4/4a4/2ncb2c1/p3p2CP/9/1N1RP4/P5r2/4C4/9/2BAKA3 b - - 0 1")
                .unwrap();
        let mut history = position.initial_rule_history();
        let moves = ["c7b5", "d4d5", "b5c7", "d5d4", "c7b5"];
        apply_uci_moves(&mut position, &mut history, &moves);

        assert_eq!(history.len(), moves.len() + 1);
        assert_eq!(position.side_to_move(), Color::Red);
        assert!(!uci_root_moves(&position, &history).is_empty());
    }
}
