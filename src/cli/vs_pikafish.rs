use crate::cli::args::*;
use chineseai::pikafish::{
    PikafishEngineConfig, VsPikafishConfig, opening_book::Px0OpeningBook, run_vs_pikafish,
};
use std::path::Path;
use std::path::PathBuf;

pub(crate) fn run(cmd: VsPikafishArgs) {
    let pikafish_exe = cmd.pikafish_exe;
    let model_path = cmd.model;
    let simulations = cmd.simulations.unwrap_or(800).max(1);
    let cpuct = cmd.cpuct.max(0.0);
    let cpuct_at_root = cmd.cpuct_at_root.max(0.0);
    let cpuct_base = cmd.cpuct_base.max(1.0);
    let cpuct_factor = cmd.cpuct_factor.max(0.0);
    let cpuct_base_at_root = cmd.cpuct_base_at_root.max(1.0);
    let cpuct_factor_at_root = cmd.cpuct_factor_at_root.max(0.0);
    let fpu_value = cmd.fpu_value.max(0.0);
    let fpu_value_at_root = cmd.fpu_value_at_root.max(0.0);
    let policy_softmax_temp = cmd.policy_softmax_temp.max(1.0e-3);
    let max_plies = cmd.max_plies.max(1);
    let pikafish_depth = cmd.pikafish_depth.max(1);
    let games = cmd.games.max(1);
    let parallel_games = cmd.parallel_games.max(1);
    let (start_positions, opening_mode) = if cmd.opening_book.trim().is_empty() {
        (Vec::new(), "startpos_fallback".to_string())
    } else {
        let mut book = Px0OpeningBook::load(&cmd.opening_book, cmd.seed)
            .unwrap_or_else(|err| panic!("failed to load Px0 opening book: {err}"));
        let positions = book
            .next_batch(cmd.opening_positions.max(1), 0)
            .unwrap_or_else(|err| panic!("invalid opening FEN: {err}"))
            .into_iter()
            .map(|s| s.position)
            .collect();
        (
            positions,
            format!("px0-shuffled(book={})", cmd.opening_book),
        )
    };
    let summary = run_vs_pikafish(
        Path::new(&pikafish_exe),
        Path::new(&model_path),
        &start_positions,
        VsPikafishConfig {
            pikafish_depth,
            total_games: games,
            max_plies,
            simulations,
            seed: cmd.seed,
            parallel_games,
            cpuct,
            cpuct_at_root,
            cpuct_base,
            cpuct_factor,
            cpuct_base_at_root,
            cpuct_factor_at_root,
            fpu_value,
            fpu_value_at_root,
            policy_softmax_temp,
            report_games: cmd.report_games,
            engine: PikafishEngineConfig {
                nnue: (!cmd.pikafish_nnue.trim().is_empty())
                    .then(|| PathBuf::from(&cmd.pikafish_nnue)),
                threads: cmd.pikafish_threads.max(1),
                hash_mb: cmd.pikafish_hash_mb,
                use_book: Some(cmd.pikafish_use_book),
            },
            skip_games: cmd.skip_games,
            stop_after: cmd.stop_after,
        },
    )
    .unwrap_or_else(|err| panic!("vs-pikafish failed: {err}"));
    for item in &summary.abnormal_ends {
        println!(
            "vs-pikafish-final: game={} chinese={} end={} final_fen=\"{}\" {}",
            item.game_index,
            if item.chinese_plays_red {
                "red"
            } else {
                "black"
            },
            item.end,
            item.final_fen,
            item.position_command
        );
    }
    println!(
        "vs-pikafish: model={} search=alphazero games={} fens={} opening={} parallel={} chinese W/L/D={}/{}/{} (as_red={} as_black={}) win_reasons(general_capture={} checkmate_no_legal_moves={} rule={} pikafish_no_bestmove={} pikafish_invalid_move={} pikafish_illegal_move={}) | pikafish_depth={} max_plies={} sims={} cpuct={}/{} base={}/{} factor={}/{} fpu={}/{} policy_temp={}",
        model_path,
        summary.total_games,
        start_positions.len(),
        opening_mode,
        parallel_games.min(games),
        summary.chinese_wins,
        summary.chinese_losses,
        summary.draws,
        summary.chinese_wins_as_red,
        summary.chinese_wins_as_black,
        summary.chinese_win_by_general_capture,
        summary.chinese_win_by_no_legal_moves,
        summary.chinese_win_by_rule,
        summary.chinese_win_by_pikafish_no_bestmove,
        summary.chinese_win_by_pikafish_invalid_move,
        summary.chinese_win_by_pikafish_illegal_move,
        pikafish_depth,
        max_plies,
        simulations,
        cpuct,
        cpuct_at_root,
        cpuct_base,
        cpuct_base_at_root,
        cpuct_factor,
        cpuct_factor_at_root,
        fpu_value,
        fpu_value_at_root,
        policy_softmax_temp
    );
}
