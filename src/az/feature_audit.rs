//! 固定局面上的特征、前向和完整搜索成本；不把微基准当作棋力。
use super::{features::*, *};
use crate::xiangqi::Position;
use std::{hint::black_box, io, path::Path, time::Instant};

fn measure(mut action: impl FnMut(), repeat: usize) -> f64 {
    action();
    let start = Instant::now();
    for _ in 0..repeat {
        action();
    }
    start.elapsed().as_secs_f64() * 1e9 / repeat as f64
}
fn median(mut numbers: Vec<f64>) -> f64 {
    numbers.sort_by(f64::total_cmp);
    let middle = numbers.len() / 2;
    if numbers.len() % 2 == 0 {
        (numbers[middle - 1] + numbers[middle]) * 0.5
    } else {
        numbers[middle]
    }
}

pub fn write_feature_audit(
    model: &AzNnue,
    path: &Path,
    repeat: usize,
    simulations: usize,
    position_count: usize,
) -> io::Result<String> {
    let repeat = repeat.max(1);
    let simulations = simulations.max(1);
    let mut positions = vec![("opening".to_owned(), Position::startpos())];
    for (name, fen) in [
        ("cannon_screen", "c3k4/9/9/9/R3P4/9/9/9/9/R3K4 w"),
        ("cannon_two_gates", "c3k4/9/9/9/R3P4/9/P8/9/9/R3K4 w"),
        ("horse_gate", "3k5/9/9/5r3/4P4/4N4/9/9/9/4K4 w"),
        ("double_attack", "3k5/9/9/3r1c3/9/4N4/9/9/9/4K4 w"),
        ("guard_chain", "3k5/9/9/3P5/9/4Nr3/9/9/9/4K4 w"),
    ] {
        positions.push((
            name.to_owned(),
            Position::from_fen(fen).map_err(candle_io_error)?,
        ));
    }
    let mut rng = SplitMix64::new(0x46454154555245);
    let mut game = 0;
    while positions.len() < position_count.max(14) {
        let mut p = Position::startpos();
        for ply in 1..=96 {
            let moves = p.legal_moves();
            if moves.is_empty() {
                break;
            }
            p.make_move(moves[rng.next_u64() as usize % moves.len()]);
            if ply % 11 == 0 {
                positions.push((format!("walk_{game:02}_{ply:02}"), p.clone()));
                if positions.len() >= position_count.max(14) {
                    break;
                }
            }
        }
        game += 1;
    }
    let mut csv = String::from(
        "case,fen,pieces,moves,relation_rows,motif_rows,dirty_sources,dirty_roots,graph_ns,king_ns,move_features_ns,full_features_ns,incremental_features_ns,full_eval_ns,cached_eval_ns,search_full_nps,search_incremental_nps,search_full_bytes,search_incremental_bytes\n",
    );
    let mut full_times = Vec::new();
    let mut cached_times = Vec::new();
    let mut nps_ratios = Vec::new();
    for (name, p) in &positions {
        let moves = p.legal_moves();
        if moves.is_empty() {
            continue;
        }
        let graph = Graph::new(Board::new(p));
        let rows = extract(&graph, p.side_to_move(), u128::MAX);
        let sums = model.feature_sums(&graph, p.side_to_move());
        let context = rule_context_features(p, &p.initial_rule_history());
        let features = nnue::extract_sparse_features_az(p);
        let mut board = vec![0.; model.hidden_size];
        model.input_embedding_linear_into_slice(&features, &mut board);
        let mut full = AzEvalScratch::new(model.arch);
        let mut cached = AzEvalScratch::new(model.arch);
        let full_eval = measure(
            || {
                black_box(model.evaluate_with_scratch_output(p, &moves, &context, &mut full));
            },
            repeat,
        );
        let cached_eval = measure(
            || {
                cached.feature_precomputed = Some(sums);
                black_box(model.evaluate_incremental_with_scratch_output(
                    p,
                    &board,
                    &moves,
                    &[],
                    &context,
                    &mut cached,
                ));
            },
            repeat,
        );
        let full_features = measure(
            || {
                black_box(model.feature_sums(&Graph::new(Board::new(p)), p.side_to_move()));
            },
            repeat,
        );
        let graph_ns = measure(
            || {
                black_box(Graph::new(Board::new(p)));
            },
            repeat,
        );
        let king_ns = measure(
            || {
                black_box(king_context(&graph, p.side_to_move()));
            },
            repeat,
        );
        let move_ns = measure(
            || {
                let mut cache = MoveFeatureCache::new(&graph, moves.iter().copied());
                for &m in &moves {
                    black_box(cache.features(m, p.side_to_move(), None));
                }
            },
            repeat,
        );
        let mv = moves[0];
        let mut child = p.clone();
        child.make_move(mv);
        let new_graph = Graph::new(Board::new(&child));
        let (sources, roots) =
            dirty_roots(&graph, &new_graph, (1u128 << mv.from) | (1u128 << mv.to));
        let mut arena = node_features::NodeFeatureArena::default();
        let base = arena.root(model, p, p.side_to_move());
        let delta_features = measure(
            || {
                black_box(arena.update(
                    model,
                    base,
                    p,
                    &child,
                    p.side_to_move(),
                    (1u128 << mv.from) | (1u128 << mv.to),
                ));
            },
            repeat,
        );
        let mut search_rates = [0.; 2];
        let mut search_bytes = [0; 2];
        for (i, incremental) in [false, true].into_iter().enumerate() {
            let mut m = model.clone();
            m.node_incremental = incremental;
            let limits = AzSearchLimits {
                simulations,
                cpuct: 1.5,
                cpuct_at_root: 1.5,
                ..Default::default()
            };
            black_box(alphazero_search(p, &m, limits));
            let mut count = 0;
            let start = Instant::now();
            for iteration in 0..3 {
                count += alphazero_search(
                    p,
                    &m,
                    AzSearchLimits {
                        seed: iteration,
                        ..limits
                    },
                )
                .simulations;
            }
            search_rates[i] = count as f64 / start.elapsed().as_secs_f64();
            let mut workspace = alphazero::AzSearchWorkspace::new(&m);
            black_box(alphazero::alphazero_search_with_rules_reusing(
                p,
                &p.initial_rule_history(),
                p.legal_moves(),
                limits,
                &mut workspace,
            ));
            search_bytes[i] = workspace.retained_bytes();
        }
        full_times.push(full_eval);
        cached_times.push(cached_eval);
        nps_ratios.push(search_rates[1] / search_rates[0]);
        csv.push_str(&format!("{name},{},{},{},{},{},{},{},{graph_ns:.1},{king_ns:.1},{move_ns:.1},{full_features:.1},{delta_features:.1},{full_eval:.1},{cached_eval:.1},{:.1},{:.1},{},{}\n",p.to_fen(),graph.board.occupied.count_ones(),moves.len(),rows.relations.len(),rows.motifs.len(),sources.count_ones(),(roots&(graph.board.occupied|new_graph.board.occupied)).count_ones(),search_rates[0],search_rates[1],search_bytes[0],search_bytes[1]));
    }
    std::fs::write(path, csv)?;
    let table_bytes = (model.input_hidden.len()
        + model.king_piece.len()
        + model.relation_embedding.len()
        + model.motif_embedding.len())
        * 4;
    Ok(format!(
        "cases={} repeat={} simulations={}\ninput_tables_mib={:.3}\nmedian_full_eval_us={:.3}\nmedian_cached_eval_us={:.3}\nmedian_search_incremental_ratio={:.3}\nreport={}",
        positions.len(),
        repeat,
        simulations,
        table_bytes as f64 / 1048576.,
        median(full_times) / 1000.,
        median(cached_times) / 1000.,
        median(nps_ratios),
        path.display()
    ))
}
