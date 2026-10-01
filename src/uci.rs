use crate::ab::pikafish_candle::{PikafishCpuCache, PikafishCpuModel};
use crate::ab::{
    AbNnue, AbSearchControl, AbSearchLimits, AbUciSearchResult, cp_from_q, search_uci,
    search_uci_pikafish, search_uci_pikafish_float,
};
use crate::nnue::pikafish_file::{PikafishNet, internal_units_from_q};
use crate::xiangqi::{Color, Move, Position, RuleHistoryEntry};
use std::io::{self, BufRead, Read, Write};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const MAX_UCI_NODES: usize = u32::MAX as usize - 1;
const MAX_UCI_TIME_MS: u64 = 7 * 24 * 60 * 60 * 1_000;
const DEFAULT_NODES: usize = 10_000;
const MAX_SEARCH_NODES_OPTION: usize = 100_000_000;
const DEFAULT_MOVE_OVERHEAD_MS: u64 = 10;

#[derive(Clone)]
enum UciModel {
    Native(Arc<AbNnue>),
    Pikafish(Arc<PikafishNet>),
    PikafishFloat(Arc<PikafishCpuModel>),
}

#[derive(Clone)]
struct UciState {
    position: Position,
    rule_history: Vec<RuleHistoryEntry>,
    eval_file: String,
    model: Option<UciModel>,
    nodes: usize,
    sixty_move_rule: bool,
    rule60_max_ply: u16,
    multipv: usize,
    show_wdl: bool,
    move_overhead_ms: u64,
}

impl Default for UciState {
    fn default() -> Self {
        Self {
            position: Position::startpos(),
            rule_history: Position::startpos().initial_rule_history(),
            eval_file: "best.safetensors".into(),
            model: None,
            nodes: DEFAULT_NODES,
            sixty_move_rule: true,
            rule60_max_ply: 120,
            multipv: 1,
            show_wdl: false,
            move_overhead_ms: DEFAULT_MOVE_OVERHEAD_MS,
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
            }
            Some("setoption") => {
                stop_active_search(&mut active_search);
                handle_setoption(line, &mut state);
            }
            Some("position") => {
                stop_active_search(&mut active_search);
                handle_position(line, &mut state);
            }
            Some("eval") => {
                stop_active_search(&mut active_search);
                print_static_eval(&mut state);
            }
            Some("go") => {
                stop_active_search(&mut active_search);
                active_search = start_go(line, &mut state);
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
    println!("id name ChineseAI AB-NNUE");
    println!("id author ChineseAI");
    println!("option name EvalFile type string default best.safetensors");
    println!("option name SearchNodes type spin default {DEFAULT_NODES} min 1 max 100000000");
    println!("option name MultiPV type spin default 1 min 1 max 64");
    println!(
        "option name Move Overhead type spin default {DEFAULT_MOVE_OVERHEAD_MS} min 0 max 5000"
    );
    println!("option name UCI_ShowWDL type check default false");
    println!("option name Sixty Move Rule type check default true");
    println!("option name Rule60MaxPly type spin default 120 min 1 max 150");
    println!("uciok");
    flush();
}

fn load_float_or_native(path: &str) -> Result<UciModel, String> {
    let mut file = std::fs::File::open(path).map_err(|error| error.to_string())?;
    let mut length = [0; 8];
    file.read_exact(&mut length)
        .map_err(|error| error.to_string())?;
    let size = u64::from_le_bytes(length);
    if size > 1_048_576 {
        return Err("safetensors header too large".into());
    }
    let mut header = vec![0; size as usize];
    file.read_exact(&mut header)
        .map_err(|error| error.to_string())?;
    let metadata: serde_json::Value =
        serde_json::from_slice(&header).map_err(|error| error.to_string())?;
    if metadata.get("transformer.psq").is_some() {
        let model = PikafishCpuModel::load(std::path::Path::new(path))
            .map_err(|error| error.to_string())?;
        Ok(UciModel::PikafishFloat(Arc::new(model)))
    } else {
        AbNnue::load(path)
            .map(|model| UciModel::Native(Arc::new(model)))
            .map_err(|error| error.to_string())
    }
}

fn ensure_model(state: &mut UciState) -> bool {
    if state.model.is_some() {
        return true;
    }
    let loaded = if state.eval_file.to_ascii_lowercase().ends_with(".nnue") {
        PikafishNet::load(std::path::Path::new(&state.eval_file))
            .map(|model| UciModel::Pikafish(Arc::new(model)))
    } else {
        load_float_or_native(&state.eval_file)
    };
    match loaded {
        Ok(model) => {
            println!("info string loaded {}", state.eval_file);
            state.model = Some(model);
            flush();
            true
        }
        Err(err) => {
            println!("info string failed to load {}: {}", state.eval_file, err);
            flush();
            false
        }
    }
}

fn print_static_eval(state: &mut UciState) {
    if !ensure_model(state) {
        return;
    }
    match state.model.as_ref().unwrap() {
        UciModel::Pikafish(model) => {
            if state.position.in_check(state.position.side_to_move()) {
                println!("Final evaluation: none (in check)");
            } else {
                match model.evaluate(&state.position) {
                    Ok(raw) => println!("NNUE evaluation: {raw:+} (internal units)"),
                    Err(error) => println!("info string eval failed: {error}"),
                }
            }
        }
        UciModel::Native(model) => {
            let wdl = model.evaluate_wdl_with_rules(&state.position, &state.rule_history);
            let cp = (((wdl[0] - wdl[2]).clamp(-1.0, 1.0)) * 1000.0).round() as i32;
            println!("info string ChineseAI native static value: {cp:+} project cp");
        }
        UciModel::PikafishFloat(model) => {
            let value = model
                .evaluate(&state.position, &mut PikafishCpuCache::default())
                .map_err(|error| error.to_string());
            match value {
                Ok(value) => println!(
                    "info string ChineseAI Pikafish float value: {:+} project cp",
                    cp_from_q(value)
                ),
                Err(error) => println!("info string eval failed: {error}"),
            }
        }
    }
    flush();
}

fn handle_setoption(line: &str, state: &mut UciState) {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    let Some(name_index) = tokens
        .iter()
        .position(|token| token.eq_ignore_ascii_case("name"))
    else {
        return;
    };
    let value_index = tokens
        .iter()
        .position(|token| token.eq_ignore_ascii_case("value"));
    let name_end = value_index.unwrap_or(tokens.len());
    let name = tokens[name_index + 1..name_end]
        .join(" ")
        .to_ascii_lowercase();
    let value = value_index
        .map(|index| tokens[index + 1..].join(" "))
        .unwrap_or_default();

    match name.as_str() {
        "multipv" => {
            if let Ok(value) = value.parse::<usize>() {
                state.multipv = value.clamp(1, 64);
            }
        }
        "evalfile" => {
            state.eval_file = value;
            state.model = None;
        }
        "searchnodes" => {
            if let Ok(nodes) = value.parse::<usize>() {
                state.nodes = nodes.clamp(1, MAX_SEARCH_NODES_OPTION);
            }
        }
        "move overhead" => {
            if let Ok(milliseconds) = value.parse::<u64>() {
                state.move_overhead_ms = milliseconds.min(5_000);
            }
        }
        "uci_showwdl" => {
            if let Some(enabled) = parse_uci_bool(&value) {
                state.show_wdl = enabled;
            }
        }
        "sixty move rule" => {
            if let Some(enabled) = parse_uci_bool(&value) {
                state.sixty_move_rule = enabled;
                apply_rule_options(state);
            }
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

fn parse_uci_bool(value: &str) -> Option<bool> {
    if value.eq_ignore_ascii_case("true") {
        Some(true)
    } else if value.eq_ignore_ascii_case("false") {
        Some(false)
    } else {
        None
    }
}

fn apply_rule_options(state: &mut UciState) {
    state
        .position
        .set_rule60_max_ply(state.sixty_move_rule.then_some(state.rule60_max_ply));
}

fn handle_position(line: &str, state: &mut UciState) {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    let moves_index = tokens.iter().position(|token| *token == "moves");
    let mut position = match tokens.get(1) {
        Some(&"startpos") => Position::startpos(),
        Some(&"fen") => {
            let fen = tokens[2..moves_index.unwrap_or(tokens.len())].join(" ");
            let Ok(position) = Position::from_fen(&fen) else {
                println!("info string invalid position FEN");
                return;
            };
            position
        }
        _ => return,
    };
    position.set_rule60_max_ply(state.sixty_move_rule.then_some(state.rule60_max_ply));
    let mut history = position.initial_rule_history();
    if let Some(index) = moves_index {
        if let Err(error) = apply_uci_moves(&mut position, &mut history, &tokens[index + 1..]) {
            println!("info string {error}; position rejected");
            return;
        }
    }
    state.position = position;
    state.rule_history = history;
}

fn apply_uci_moves(
    position: &mut Position,
    rule_history: &mut Vec<RuleHistoryEntry>,
    moves: &[&str],
) -> Result<(), String> {
    for (ply, text) in moves.iter().enumerate() {
        let mv = position
            .parse_uci_move(text)
            .ok_or_else(|| format!("invalid history move {} at ply {}", text, ply + 1))?;
        // `position ... moves` is the external controller's authoritative game
        // history. Accept every board-legal move even if its tournament rule set
        // differs from ours; our repetition rules only guide future search moves.
        if !position.legal_moves().contains(&mv) {
            return Err(format!("illegal history move {} at ply {}", text, ply + 1));
        }
        rule_history.push(position.rule_history_entry_after_move(mv));
        position.make_move(mv);
    }
    Ok(())
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

fn time_budget_ms(params: &GoParams, side: Color, overhead_ms: u64) -> Option<u64> {
    if let Some(move_time_ms) = params.move_time_ms {
        return Some(
            move_time_ms
                .saturating_sub(overhead_ms)
                .clamp(1, MAX_UCI_TIME_MS),
        );
    }
    if params.infinite {
        return None;
    }
    let (remaining_ms, increment_ms) = match side {
        Color::Red => (params.wtime_ms?, params.winc_ms),
        Color::Black => (params.btime_ms?, params.binc_ms),
    };
    let usable_ms = remaining_ms.saturating_sub(overhead_ms).max(1);
    let moves = params.moves_to_go.unwrap_or(24).max(1);
    let target_ms = usable_ms / moves + increment_ms.saturating_mul(3) / 4;
    let maximum_ms = (usable_ms / 5).max(1);
    Some(target_ms.clamp(1, maximum_ms).min(MAX_UCI_TIME_MS))
}

fn start_go(line: &str, state: &mut UciState) -> Option<ActiveSearch> {
    if !ensure_model(state) {
        println!("bestmove 0000");
        flush();
        return None;
    }
    let params = parse_go(line);
    let snapshot = state.clone();
    let stop = Arc::new(AtomicBool::new(false));
    let search_stop = Arc::clone(&stop);
    let handle = thread::spawn(move || run_go_search(snapshot, params, search_stop));
    Some(ActiveSearch { stop, handle })
}

fn run_go_search(state: UciState, params: GoParams, stop: Arc<AtomicBool>) {
    let model = state.model.as_ref().expect("model was loaded");

    let mut legal = uci_root_moves(&state.position, &state.rule_history);
    let root_has_legal_moves = !legal.is_empty();
    if !params.searchmoves.is_empty() {
        legal.retain(|mv| {
            params
                .searchmoves
                .iter()
                .any(|text| state.position.parse_uci_move(text) == Some(*mv))
        });
    }

    if legal.is_empty() {
        let score = if root_has_legal_moves {
            "cp 0"
        } else {
            "cp -1000"
        };
        println!("info depth 1 nodes 0 time 0 score {score}");
        println!("bestmove 0000");
        flush();
        return;
    }

    let budget_ms = time_budget_ms(
        &params,
        state.position.side_to_move(),
        state.move_overhead_ms,
    );
    let has_time_control = budget_ms.is_some() || params.infinite;
    let nodes = uci_node_limit(&params, state.nodes, has_time_control);
    let started = Instant::now();
    let deadline = budget_ms.map(|budget| started + Duration::from_millis(budget));
    let control = AbSearchControl::new(Arc::clone(&stop), deadline);
    let pikafish_score = matches!(model, UciModel::Pikafish(_));
    println!(
        "info string searchparams mode=alphabeta nodes={nodes} score_scale={}",
        if pikafish_score {
            "Pikafish-internal"
        } else {
            "ChineseAI-q1000"
        }
    );
    flush();
    let mut last_score_source = None;
    let mut report_progress = |progress: &AbUciSearchResult| {
        if let Some(proven) = high_score_source(progress) {
            if last_score_source != Some(proven) {
                print_high_score_source(progress, proven);
                last_score_source = Some(proven);
            }
        }
        print_search_info(progress, started, state.show_wdl, pikafish_score);
        flush();
    };
    let limits = AbSearchLimits {
        nodes,
        max_depth: params.depth.unwrap_or(0),
        ..AbSearchLimits::default()
    };
    let report = match model {
        UciModel::Native(model) => search_uci(
            &state.position,
            state.rule_history.clone(),
            legal,
            model,
            limits,
            &control,
            state.multipv,
            &mut report_progress,
        ),
        UciModel::Pikafish(model) => search_uci_pikafish(
            &state.position,
            state.rule_history.clone(),
            legal,
            model,
            limits,
            &control,
            state.multipv,
            &mut report_progress,
        ),
        UciModel::PikafishFloat(model) => search_uci_pikafish_float(
            &state.position,
            state.rule_history.clone(),
            legal,
            model,
            limits,
            &control,
            state.multipv,
            &mut report_progress,
        ),
    };
    let result = &report.search;
    if let Some(proven) = high_score_source(&report) {
        if last_score_source != Some(proven) {
            print_high_score_source(&report, proven);
        }
    }
    // 无限分析在收到 stop 前不发 bestmove；证明终局后等待。
    while params.infinite && !stop.load(Ordering::Relaxed) {
        thread::park_timeout(Duration::from_millis(10));
    }
    match result.best_move {
        Some(mv) => {
            print_search_info(&report, started, state.show_wdl, pikafish_score);
            println!("bestmove {mv}");
        }
        None => {
            println!(
                "info depth 1 nodes {} time {} score cp {}",
                result.nodes,
                started.elapsed().as_millis(),
                uci_score_from_q(result.value_q, pikafish_score)
            );
            println!("bestmove 0000");
        }
    }
    flush();
}

fn uci_node_limit(params: &GoParams, configured: usize, has_time_control: bool) -> usize {
    let requested = params.nodes.unwrap_or(if params.infinite {
        MAX_UCI_NODES
    } else if has_time_control {
        MAX_UCI_NODES
    } else {
        configured.max(1)
    });
    requested.clamp(1, MAX_UCI_NODES)
}

fn high_score_source(report: &AbUciSearchResult) -> Option<bool> {
    report
        .variations
        .first()
        .filter(|pv| cp_from_q(pv.q).abs() >= 900)
        .map(|pv| pv.proven.is_some())
}

fn print_high_score_source(report: &AbUciSearchResult, proven: bool) {
    println!(
        "info string high score q={:.3} source={}",
        report.variations[0].q,
        if proven {
            "search-proof"
        } else {
            "value-estimate"
        }
    );
}

fn uci_score_from_q(q: f32, pikafish_score: bool) -> i32 {
    if pikafish_score {
        internal_units_from_q(q)
    } else {
        cp_from_q(q)
    }
}

fn print_search_info(
    report: &AbUciSearchResult,
    started: Instant,
    show_wdl: bool,
    pikafish_score: bool,
) {
    let result = &report.search;
    let elapsed_ms = started.elapsed().as_millis();
    let nps = result.nodes as u128 * 1000 / elapsed_ms.max(1);
    for (index, pv) in report.variations.iter().enumerate() {
        let wdl = uci_wdl(pv.wdl);
        let moves = pv
            .moves
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" ");
        let wdl_text = if show_wdl {
            format!(" wdl {} {} {}", wdl[0], wdl[1], wdl[2])
        } else {
            String::new()
        };
        println!(
            "info depth {} seldepth {} multipv {} nodes {} nps {} time {} score cp {}{} pv {}",
            result.search_depth_avg.round() as usize,
            result.selective_depth,
            index + 1,
            result.nodes,
            nps,
            elapsed_ms,
            uci_score_from_q(pv.q, pikafish_score),
            wdl_text,
            moves,
        );
    }
    if report.variations.is_empty() {
        println!(
            "info depth 0 nodes {} time {} score cp {}",
            result.nodes,
            elapsed_ms,
            uci_score_from_q(result.value_q, pikafish_score)
        );
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
        assert_eq!(state.nodes, DEFAULT_NODES);
        assert_eq!(state.eval_file, "best.safetensors");
    }

    #[test]
    fn missing_model_never_falls_back_to_random_weights() {
        let mut state = UciState::default();
        state.eval_file = "missing-ab-model.safetensors".into();
        assert!(!ensure_model(&mut state));
        assert!(state.model.is_none());
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
        assert_eq!(time_budget_ms(&move_time, Color::Red, 0), Some(1_000));
        assert_eq!(time_budget_ms(&move_time, Color::Red, 10), Some(990));

        let clock = parse_go("go wtime 60000 btime 30000 winc 1000 binc 0 movestogo 20");
        assert_eq!(time_budget_ms(&clock, Color::Red, 0), Some(3_750));
        assert_eq!(time_budget_ms(&clock, Color::Black, 0), Some(1_500));

        let infinite = parse_go("go infinite");
        assert_eq!(time_budget_ms(&infinite, Color::Red, 10), None);
    }

    #[test]
    fn infinite_analysis_runs_until_stop_or_explicit_nodes() {
        let infinite = parse_go("go infinite");
        assert_eq!(uci_node_limit(&infinite, 10_000, true), MAX_UCI_NODES);

        let explicit = parse_go("go infinite nodes 100000000");
        assert_eq!(uci_node_limit(&explicit, 10_000, true), 100_000_000);

        let timed = parse_go("go movetime 1000");
        assert_eq!(uci_node_limit(&timed, 10_000, true), MAX_UCI_NODES);
    }

    #[test]
    fn search_node_limit_is_configurable() {
        let mut state = UciState::default();
        handle_setoption("setoption name SearchNodes value 4096", &mut state);
        assert_eq!(state.nodes, 4096);
        handle_setoption("setoption name SearchNodes value 999999999", &mut state);
        assert_eq!(state.nodes, MAX_SEARCH_NODES_OPTION);
    }

    #[test]
    fn pikafish_style_uci_options_are_parsed_and_validated() {
        let mut state = UciState::default();
        assert!(!state.show_wdl);
        assert_eq!(state.move_overhead_ms, 10);
        handle_setoption("setoption name UCI_ShowWDL value true", &mut state);
        handle_setoption("setoption name Move Overhead value 23", &mut state);
        assert!(state.show_wdl);
        assert_eq!(state.move_overhead_ms, 23);
        handle_setoption("setoption NAME UCI_ShowWDL VALUE invalid", &mut state);
        assert!(state.show_wdl);
        handle_setoption("setoption name UCI_ShowWDL value false", &mut state);
        assert!(!state.show_wdl);
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
    fn invalid_position_history_does_not_replace_current_state() {
        let mut state = UciState::default();
        let before = state.position.to_fen();
        handle_position(
            "position fen 2baka3/9/9/2p1r4/P8/4p1R1R/1n1cc4/B8/4A4/3K1AB2 w - - 0 1 moves h7d7",
            &mut state,
        );
        assert_eq!(state.position.to_fen(), before);
        assert_eq!(state.rule_history.len(), 1);
    }

    #[test]
    fn uci_import_accepts_external_repeated_long_check() {
        let mut position = Position::from_fen(
            "2Rakab2/8r/4c1n2/p3p1p1p/2p6/9/P3P3P/1CN1NC3/9/1RBAKArc1 b - - 0 1",
        )
        .unwrap();
        let mut history = position.initial_rule_history();
        let moves = ["g0g1", "f0e1", "g1g0", "e1f0", "g0g1"];
        apply_uci_moves(&mut position, &mut history, &moves).unwrap();

        assert_eq!(history.len(), moves.len() + 1);
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
        apply_uci_moves(&mut position, &mut history, &moves).unwrap();

        assert_eq!(history.len(), moves.len() + 1);
        assert_eq!(position.side_to_move(), Color::Red);
        assert!(!uci_root_moves(&position, &history).is_empty());
    }

    #[test]
    fn uci_import_preserves_repetition_history_for_value_evaluation() {
        let mut position = Position::from_fen(
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1",
        )
        .unwrap();
        let mut history = position.initial_rule_history();
        let moves = "b2e2 b9c7 b0c2 h9g7 c3c4 g6g5 a0b0 a9b9 h0i2 i6i5 h2f2 b7b5 b0b4 i9i6 c4c5 c6c5 i0h0 c5c4 b4c4 g7h5 f2h2 h5g7 h2f2 g7h5 f2h2 h5g7 h2f2 g7h5 f2h2";
        let moves = moves.split_whitespace().collect::<Vec<_>>();

        apply_uci_moves(&mut position, &mut history, &moves).unwrap();

        assert_eq!(history.len(), moves.len() + 1);
        assert_eq!(position.side_to_move(), Color::Black);
        assert!(crate::ab::rule_context_features(&position, &history)[1] > 0.0);
    }
}
