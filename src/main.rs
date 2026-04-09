use chineseai::{
    az::{AzNnue, AzSearchLimits, gumbel_search},
    xiangqi::{Position, STARTPOS_FEN},
};

fn main() {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("az-init") => {
            let hidden = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(128);
            let output = args.next().unwrap_or_else(|| "az.nnue.txt".into());
            let seed = args
                .next()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(20260409);
            let model = AzNnue::random(hidden, seed);
            model.save_text(&output).unwrap_or_else(|err| {
                panic!("failed to write `{output}`: {err}");
            });
            println!("aznnue   : initialized");
            println!("hidden   : {hidden}");
            println!("seed     : {seed}");
            println!("format   : aznnue-v1");
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
            let result = gumbel_search(&position, &model, AzSearchLimits { simulations, top_k });
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
                    "candidate: {} visits={} q={:.3} prior={:.5}",
                    candidate.mv, candidate.visits, candidate.q, candidate.prior
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
        _ => {
            let position = Position::startpos();
            println!("ChineseAI AZ-NNUE Gumbel core");
            println!("start : {STARTPOS_FEN}");
            println!("moves : {}", position.legal_moves().len());
            println!("hint  : cargo run --release -- az-init 128 az.nnue.txt");
            println!("hint  : cargo run --release -- az-gumbel az.nnue.txt 10000 32 startpos");
        }
    }
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
