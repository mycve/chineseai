use chineseai::{
    nnue::{FeatureSet, NnueModel, extract_sparse_features, extract_sparse_features_v2},
    rules::{MoveRecord, make_record},
    search::{Engine, SearchLimits, render_principal_variation},
    uci,
    xiangqi::{Color, Position, STARTPOS_FEN},
};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};

fn main() {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("uci") | None => {
            if let Err(err) = uci::run_loop() {
                eprintln!("uci error: {err}");
                std::process::exit(1);
            }
        }
        Some("bench") => {
            let depth = args
                .next()
                .and_then(|value| value.parse::<u8>().ok())
                .unwrap_or(6);
            let position = Position::startpos();
            let started = std::time::Instant::now();
            let mut engine = Engine::default();
            let result = engine.search(
                &position,
                SearchLimits {
                    depth,
                    movetime: None,
                    nodes: None,
                },
            );
            let elapsed = started.elapsed();
            let seconds = elapsed.as_secs_f64().max(0.000_001);
            let nps = (result.nodes as f64 / seconds) as u64;

            println!("bench    : startpos");
            println!("depth    : {}", result.depth);
            println!("score    : {}", result.score_string());
            println!("nodes    : {}", result.nodes);
            println!("time_ms  : {}", elapsed.as_millis());
            println!("nps      : {nps}");
            println!(
                "bestmove : {}",
                result
                    .best_move
                    .map(|mv| mv.to_string())
                    .unwrap_or_else(|| "(none)".into())
            );
            println!("pv       : {}", render_principal_variation(&result.pv));
        }
        Some("nnue-init") => {
            let hidden = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(32);
            let output = args.next().unwrap_or_else(|| "model.nnue.txt".into());
            let feature_set = args
                .next()
                .map(|value| parse_feature_set(&value))
                .unwrap_or(FeatureSet::V1);
            let model = NnueModel::zeroed_with_feature_set(hidden, feature_set);
            model.save_text(&output).unwrap_or_else(|err| {
                panic!("failed to write `{output}`: {err}");
            });
            println!("nnue     : initialized");
            println!("features : {:?}", feature_set);
            println!("hidden   : {hidden}");
            println!("inputs   : {}", model.input_size);
            println!("output   : {output}");
        }
        Some("nnue-features") => {
            let values = args.collect::<Vec<_>>();
            let (feature_set, fen_parts) = if values.first().is_some_and(|value| value == "--v2") {
                (FeatureSet::V2, &values[1..])
            } else {
                (FeatureSet::V1, &values[..])
            };
            let fen = fen_parts.join(" ");
            let position = if fen.trim().is_empty() || fen == "startpos" {
                Position::startpos()
            } else {
                Position::from_fen(&fen).unwrap_or_else(|err| {
                    panic!("invalid FEN `{fen}`: {err}");
                })
            };
            let features = match feature_set {
                FeatureSet::V1 => extract_sparse_features(&position),
                FeatureSet::V2 => extract_sparse_features_v2(&position),
            };
            println!("schema   : {:?}", feature_set);
            println!("features : {}", features.len());
            println!(
                "indices  : {}",
                features
                    .iter()
                    .map(|value| value.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
        Some("nnue-eval") => {
            let model_path = args
                .next()
                .unwrap_or_else(|| panic!("usage: nnue-eval <model> [fen]"));
            let fen = args.collect::<Vec<_>>().join(" ");
            let position = if fen.trim().is_empty() || fen == "startpos" {
                Position::startpos()
            } else {
                Position::from_fen(&fen).unwrap_or_else(|err| {
                    panic!("invalid FEN `{fen}`: {err}");
                })
            };
            let model = NnueModel::load_text(&model_path).unwrap_or_else(|err| {
                panic!("failed to load `{model_path}`: {err}");
            });
            println!("nnue     : {}", model.evaluate(&position));
        }
        Some("nnue-dump") => {
            let input = args
                .next()
                .unwrap_or_else(|| panic!("usage: nnue-dump <input> <output>"));
            let output = args
                .next()
                .unwrap_or_else(|| panic!("usage: nnue-dump <input> <output>"));
            let raw = std::fs::read_to_string(&input).unwrap_or_else(|err| {
                panic!("failed to read `{input}`: {err}");
            });
            let mut rows = Vec::new();
            for (line_no, line) in raw.lines().enumerate() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let Some((label, fen)) = line.split_once(' ') else {
                    panic!("line {} missing `<label> <fen>`", line_no + 1);
                };
                let position = Position::from_fen(fen).unwrap_or_else(|err| {
                    panic!("invalid FEN on line {}: {err}", line_no + 1);
                });
                let features = extract_sparse_features(&position);
                rows.push(format!(
                    "{}\t{}",
                    label,
                    features
                        .iter()
                        .map(|value| value.to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                ));
            }
            std::fs::write(&output, rows.join("\n") + "\n").unwrap_or_else(|err| {
                panic!("failed to write `{output}`: {err}");
            });
            println!("dumped   : {}", rows.len());
            println!("output   : {output}");
        }
        Some("selfplay-dump") => {
            let games = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(1);
            let depth = args
                .next()
                .and_then(|value| value.parse::<u8>().ok())
                .unwrap_or(4);
            let output = args.next().unwrap_or_else(|| "selfplay.txt".into());
            let max_plies = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(160);

            let mut rows = Vec::new();
            let mut engine = Engine::default();

            for game in 0..games {
                let mut position = Position::startpos();
                let mut history_hashes = Vec::new();
                let mut records: Vec<MoveRecord> = Vec::new();

                for ply in 0..max_plies {
                    let result = engine.search_with_history_and_records(
                        &position,
                        &history_hashes,
                        &records,
                        SearchLimits {
                            depth,
                            movetime: None,
                            nodes: Some(4_000),
                        },
                    );

                    rows.push(format!("{} {}", result.score, position.to_fen()));

                    let legal_moves = position.legal_moves();
                    if legal_moves.is_empty() {
                        break;
                    }

                    let chosen = if ply < 6 {
                        let index = ((position.hash() as usize) ^ (game * 17 + ply * 31))
                            % legal_moves.len();
                        legal_moves[index]
                    } else {
                        result.best_move.unwrap_or(legal_moves[0])
                    };

                    history_hashes.push(position.hash());
                    records.push(make_record(&position, chosen));
                    position.make_move(chosen);

                    if position.halfmove_clock() >= 120 {
                        rows.push(format!("0 {}", position.to_fen()));
                        break;
                    }
                }
            }

            std::fs::write(&output, rows.join("\n") + "\n").unwrap_or_else(|err| {
                panic!("failed to write `{output}`: {err}");
            });
            println!("games    : {games}");
            println!("depth    : {depth}");
            println!("samples  : {}", rows.len());
            println!("output   : {output}");
        }
        Some("selfplay-nnue-dump") => {
            let games = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(10);
            let depth = args
                .next()
                .and_then(|value| value.parse::<u8>().ok())
                .unwrap_or(4);
            let output = args
                .next()
                .unwrap_or_else(|| panic!("usage: selfplay-nnue-dump <games> <depth> <output> [max_plies] [nodes] [random_plies] [result_scale]"));
            let max_plies = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(160);
            let nodes = args
                .next()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(4_000);
            let random_plies = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(8);
            let result_scale = args
                .next()
                .and_then(|value| value.parse::<i32>().ok())
                .unwrap_or(600);

            let mut writer = BufWriter::new(File::create(&output).unwrap_or_else(|err| {
                panic!("failed to create `{output}`: {err}");
            }));
            let mut engine = Engine::default();
            let mut total_samples = 0usize;
            let mut red_wins = 0usize;
            let mut black_wins = 0usize;
            let mut draws = 0usize;

            for game in 0..games {
                let mut position = Position::startpos();
                let mut history_hashes = Vec::new();
                let mut records: Vec<MoveRecord> = Vec::new();
                let mut seen_hashes = HashMap::new();
                seen_hashes.insert(position.hash(), 1usize);
                let mut samples = Vec::new();
                let outcome = loop {
                    let legal_moves = position.legal_moves();
                    if legal_moves.is_empty() {
                        break match position.side_to_move() {
                            Color::Red => GameOutcome::BlackWin,
                            Color::Black => GameOutcome::RedWin,
                        };
                    }
                    if samples.len() >= max_plies {
                        break GameOutcome::Draw;
                    }

                    let result = engine.search_with_history_and_records(
                        &position,
                        &history_hashes,
                        &records,
                        SearchLimits {
                            depth,
                            movetime: None,
                            nodes: Some(nodes),
                        },
                    );
                    samples.push(SelfplaySample {
                        side_to_move: position.side_to_move(),
                        search_score: clamp_training_score(result.score),
                        features: extract_sparse_features(&position),
                    });

                    let chosen = if samples.len() <= random_plies {
                        let index = ((position.hash() as usize) ^ (game * 17 + samples.len() * 31))
                            % legal_moves.len();
                        legal_moves[index]
                    } else {
                        result.best_move.unwrap_or(legal_moves[0])
                    };

                    history_hashes.push(position.hash());
                    records.push(make_record(&position, chosen));
                    position.make_move(chosen);

                    let next_hash = position.hash();
                    let visits = seen_hashes.entry(next_hash).or_insert(0);
                    *visits += 1;
                    if *visits >= 3 || position.halfmove_clock() >= 120 {
                        break GameOutcome::Draw;
                    }
                };

                match outcome {
                    GameOutcome::RedWin => red_wins += 1,
                    GameOutcome::BlackWin => black_wins += 1,
                    GameOutcome::Draw => draws += 1,
                }

                for sample in samples {
                    let target = blended_training_target(
                        sample.search_score,
                        sample.side_to_move,
                        outcome,
                        result_scale,
                    );
                    write!(writer, "{}\t", target)
                        .unwrap_or_else(|err| panic!("failed to write `{output}`: {err}"));
                    for (index, feature) in sample.features.iter().enumerate() {
                        if index > 0 {
                            write!(writer, " ")
                                .unwrap_or_else(|err| panic!("failed to write `{output}`: {err}"));
                        }
                        write!(writer, "{feature}")
                            .unwrap_or_else(|err| panic!("failed to write `{output}`: {err}"));
                    }
                    writeln!(writer)
                        .unwrap_or_else(|err| panic!("failed to write `{output}`: {err}"));
                    total_samples += 1;
                }
            }

            writer.flush().unwrap_or_else(|err| {
                panic!("failed to flush `{output}`: {err}");
            });
            println!("games       : {games}");
            println!("depth       : {depth}");
            println!("nodes       : {nodes}");
            println!("randomplies : {random_plies}");
            println!("scale       : {result_scale}");
            println!("samples     : {total_samples}");
            println!("red wins    : {red_wins}");
            println!("black wins  : {black_wins}");
            println!("draws       : {draws}");
            println!("output      : {output}");
        }
        Some("perft") => {
            let depth = args
                .next()
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or(1);
            let fen = args.collect::<Vec<_>>().join(" ");
            let position = if fen.trim().is_empty() || fen == "startpos" {
                Position::startpos()
            } else {
                Position::from_fen(&fen).unwrap_or_else(|err| {
                    panic!("invalid FEN `{fen}`: {err}");
                })
            };
            println!("fen   : {}", position.to_fen());
            println!("depth : {depth}");
            println!("nodes : {}", position.perft(depth));
        }
        Some("search") => {
            let depth = args
                .next()
                .and_then(|value| value.parse::<u8>().ok())
                .unwrap_or(5);
            let fen = args.collect::<Vec<_>>().join(" ");
            let position = if fen.trim().is_empty() || fen == "startpos" {
                Position::startpos()
            } else {
                Position::from_fen(&fen).unwrap_or_else(|err| {
                    panic!("invalid FEN `{fen}`: {err}");
                })
            };

            let mut engine = Engine::default();
            let result = engine.search(
                &position,
                SearchLimits {
                    depth,
                    movetime: None,
                    nodes: None,
                },
            );
            println!("fen      : {}", position.to_fen());
            println!("depth    : {}", result.depth);
            println!("score    : {}", result.score_string());
            println!("nodes    : {}", result.nodes);
            println!(
                "bestmove : {}",
                result
                    .best_move
                    .map(|mv| mv.to_string())
                    .unwrap_or_else(|| "(none)".into())
            );
            println!("pv       : {}", render_principal_variation(&result.pv));
        }
        _ => {
            let position = Position::startpos();
            println!("ChineseAI Xiangqi core");
            println!("start : {}", STARTPOS_FEN);
            println!("moves : {}", position.legal_moves().len());
            println!("hint  : cargo run -- uci");
        }
    }
}

#[derive(Clone)]
struct SelfplaySample {
    side_to_move: Color,
    search_score: i32,
    features: Vec<usize>,
}

#[derive(Clone, Copy)]
enum GameOutcome {
    RedWin,
    BlackWin,
    Draw,
}

fn clamp_training_score(score: i32) -> i32 {
    score.clamp(-1_200, 1_200)
}

fn outcome_target(side_to_move: Color, outcome: GameOutcome, result_scale: i32) -> i32 {
    match outcome {
        GameOutcome::Draw => 0,
        GameOutcome::RedWin => {
            if side_to_move == Color::Red {
                result_scale
            } else {
                -result_scale
            }
        }
        GameOutcome::BlackWin => {
            if side_to_move == Color::Black {
                result_scale
            } else {
                -result_scale
            }
        }
    }
}

fn blended_training_target(
    search_score: i32,
    side_to_move: Color,
    outcome: GameOutcome,
    result_scale: i32,
) -> i32 {
    let result_target = outcome_target(side_to_move, outcome, result_scale);
    (search_score + result_target * 3) / 4
}

fn parse_feature_set(value: &str) -> FeatureSet {
    match value {
        "v1" | "V1" => FeatureSet::V1,
        "v2" | "V2" => FeatureSet::V2,
        _ => panic!("unsupported feature set `{value}`; expected v1 or v2"),
    }
}
