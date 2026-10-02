//! `dive-games`：用开局库开局，与 Pikafish 交换对弈，抽出"跳水局面"写进热身库。

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use chineseai::pikafish::opening_book::Px0OpeningBook;
use chineseai::pikafish::dive::{DiveConfig, DiveSink, SqliteDiveSink};
use chineseai::pikafish::{
    PikafishDiveConfig, PikafishEngineConfig, VsPikafishConfig, run_vs_pikafish,
};

use crate::cli::args::DiveGamesArgs;

pub(crate) fn run(cmd: DiveGamesArgs) {
    let pikafish_exe = PathBuf::from(&cmd.pikafish_exe);
    if !pikafish_exe.exists() {
        panic!("pikafish executable `{}` does not exist", pikafish_exe.display());
    }
    let model_path = PathBuf::from(&cmd.model);
    if !model_path.exists() {
        panic!("model `{}` does not exist", model_path.display());
    }
    let output = Path::new(&cmd.output);

    if cmd.clear {
        let removed = SqliteDiveSink::clear(output)
            .unwrap_or_else(|err| panic!("failed to clear `{}`: {err}", output.display()));
        println!("dive-games: cleared {removed} existing rows from {}", output.display());
    }

    let dive = DiveConfig {
        lost_below: cmd.lost_below.clamp(-1.0, 0.0),
        delta_q: cmd.delta_q.clamp(0.0, 2.0),
        delta_cp: cmd.delta_cp.max(0),
        drop_q: cmd.drop_q.clamp(0.0, 2.0),
        drop_cp: cmd.drop_cp.max(0),
        min_pieces: cmd.min_pieces.max(2),
        max_ply: cmd.max_dive_ply.max(1),
        max_frames_per_game: cmd.max_frames_per_game,
    };
    let sink: Arc<Mutex<Box<dyn DiveSink>>> = Arc::new(Mutex::new(
        Box::new(SqliteDiveSink::open(output).unwrap_or_else(|err| {
            panic!("failed to open `{}`: {err}", output.display())
        })) as Box<dyn DiveSink>,
    ));

    let (start_positions, opening_mode) = load_openings(&cmd);
    let started = Instant::now();
    let summary = run_vs_pikafish(
        &pikafish_exe,
        &model_path,
        &start_positions,
        VsPikafishConfig {
            pikafish_depth: cmd.pikafish_depth.max(1),
            total_games: cmd.games.max(1),
            max_plies: cmd.max_plies.max(1),
            simulations: cmd.simulations.max(1),
            seed: cmd.seed,
            parallel_games: cmd.parallel_games.max(1),
            cpuct: cmd.cpuct.max(0.0),
            cpuct_at_root: cmd.cpuct_at_root.max(0.0),
            cpuct_base: cmd.cpuct_base.max(1.0),
            cpuct_factor: cmd.cpuct_factor.max(0.0),
            cpuct_base_at_root: cmd.cpuct_base_at_root.max(1.0),
            cpuct_factor_at_root: cmd.cpuct_factor_at_root.max(0.0),
            fpu_value: cmd.fpu_value.max(0.0),
            fpu_value_at_root: cmd.fpu_value_at_root.max(0.0),
            policy_softmax_temp: cmd.policy_softmax_temp.max(1.0e-3),
            report_games: cmd.report_games,
            engine: PikafishEngineConfig {
                nnue: (!cmd.pikafish_nnue.trim().is_empty())
                    .then(|| PathBuf::from(&cmd.pikafish_nnue)),
                threads: cmd.pikafish_threads.max(1),
                hash_mb: cmd.pikafish_hash_mb,
                use_book: Some(cmd.pikafish_use_book),
            },
            dive: Some(PikafishDiveConfig {
                dive,
                analyze_depth: cmd.analyze_depth,
            }),
            skip_games: cmd.skip_games,
            stop_after: cmd.stop_after,
        },
        Some(Arc::clone(&sink)),
    )
    .unwrap_or_else(|err| panic!("dive-games failed: {err}"));
    let seconds = started.elapsed().as_secs_f64().max(1.0e-6);

    if cmd.report_games {
        for item in &summary.abnormal_ends {
            println!(
                "dive-games-final: game={} chinese={} end={} final_fen=\"{}\" {}",
                item.game_index,
                if item.chinese_plays_red { "red" } else { "black" },
                item.end,
                item.final_fen,
                item.position_command
            );
        }
    }
    let total_rows = SqliteDiveSink::count(output).unwrap_or(0);
    println!(
        "dive-games: played={}/{}{}{} model={} opening={} fens={} parallel={} W/L/D={}/{}/{} frames_capped={} dives_this_run={} dives_in_db={} | pikafish_depth={} analyze_depth={} sims={} delta_q={} lost_below={} drop_q={} max_frames_per_game={} min_pieces={} max_ply={} elapsed={:.1}s games_per_second={:.2}",
        summary.played_games,
        summary.total_games,
        if summary.skipped_games > 0 {
            format!(" skipped={}", summary.skipped_games)
        } else {
            String::new()
        },
        if summary.interrupted && cmd.stop_after == 0 {
            " interrupted"
        } else {
            ""
        },
        model_path.display(),
        opening_mode,
        start_positions.len(),
        cmd.parallel_games.min(cmd.games).max(1),
        summary.chinese_wins,
        summary.chinese_losses,
        summary.draws,
        summary.frames_collected_games,
        summary.dives.len(),
        total_rows,
        cmd.pikafish_depth.max(1),
        cmd.analyze_depth,
        cmd.simulations.max(1),
        cmd.delta_q.clamp(0.0, 2.0),
        cmd.lost_below.clamp(-1.0, 0.0),
        cmd.drop_q.clamp(0.0, 2.0),
        cmd.max_frames_per_game,
        cmd.min_pieces.max(2),
        cmd.max_dive_ply.max(1),
        seconds,
        summary.played_games as f64 / seconds
    );
}

/// 开局局面一律从 `book.pgn.gz` 读。
fn load_openings(cmd: &DiveGamesArgs) -> (Vec<chineseai::xiangqi::Position>, String) {
    if cmd.opening_book.trim().is_empty() {
        return (Vec::new(), "startpos_fallback".to_string());
    }
    let mut book = Px0OpeningBook::load(&cmd.opening_book, cmd.seed)
        .unwrap_or_else(|err| panic!("failed to load Px0 opening book `{}`: {err}", cmd.opening_book));
    let positions = book
        .next_batch(cmd.opening_positions.max(1), 0)
        .unwrap_or_else(|err| panic!("invalid opening FEN: {err}"))
        .into_iter()
        .map(|snapshot| snapshot.position)
        .collect();
    (
        positions,
        format!("px0-shuffled(book={})", cmd.opening_book),
    )
}
