//! 只读检查经验标签和战术病例覆盖；不复制整个经验池。
use chineseai::{
    az::{
        AzExperiencePool, AzNnue, AzTrainingSample, SplitMix64, outputs_for_training_sample,
        position_for_training_sample,
    },
    nnue::{
        AZ_NNUE_INPUT_SIZE, extract_sparse_features_az, mirror_sparse_features_az_canonical_file,
    },
    xiangqi::Position,
};
use clap::Parser;
use serde::Deserialize;
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::Path,
};
#[path = "support/pikafish.rs"]
mod pikafish;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    replay: String,
    #[arg(long)]
    model: String,
    #[arg(long)]
    cases: String,
    /// 病例编号:关键应手，可重复提供。
    #[arg(long)]
    reply: Vec<String>,
    #[arg(long, default_value_t = 2400000)]
    capacity: usize,
    #[arg(long, default_value_t = 4096)]
    evaluate_samples: usize,
    #[arg(long)]
    teacher_output: Option<String>,
    #[arg(long, default_value = "tools/pikafish-avx2.exe")]
    pikafish: String,
    #[arg(long, default_value_t = 64)]
    teacher_samples: usize,
    #[arg(long, default_value_t = 16)]
    teacher_depth: usize,
    #[arg(long)]
    input_collisions: bool,
    #[arg(long)]
    pair_coverage: bool,
}

#[derive(Deserialize)]
struct Cases {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    game: usize,
    fen: String,
    played: String,
}

fn bits(features: &[usize]) -> [u64; 20] {
    let mut out = [0; 20];
    for &f in features {
        out[f / 64] |= 1 << (f % 64);
    }
    out
}
struct Probe {
    name: String,
    original: [u64; 20],
    mirrored: [u64; 20],
    exact: usize,
    labels: [usize; 3],
    nearest: Vec<(u32, usize)>,
}
fn probe(name: String, position: &Position) -> Probe {
    let mut features = extract_sparse_features_az(position);
    let original = bits(&features);
    mirror_sparse_features_az_canonical_file(&mut features);
    Probe {
        name,
        original,
        mirrored: bits(&features),
        exact: 0,
        labels: [0; 3],
        nearest: vec![],
    }
}
fn distance(a: &[u64; 20], b: &[u64; 20]) -> u32 {
    a.iter().zip(b).map(|(a, b)| (a ^ b).count_ones()).sum()
}
fn class(sample: &AzTrainingSample) -> usize {
    (0..3)
        .max_by(|&a, &b| sample.value_wdl[a].total_cmp(&sample.value_wdl[b]))
        .unwrap()
}
#[derive(Default)]
struct Game {
    max_ply: u16,
    samples: usize,
    all_draw: bool,
}
#[derive(Default)]
struct Moments {
    labels: [u32; 3],
    sum: f64,
    squares: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let pool = AzExperiencePool::load_snapshot_lz4(Path::new(&args.replay), args.capacity)?;
    let model = AzNnue::load(&args.model)?;
    let cases: Cases = toml::from_str(&fs::read_to_string(&args.cases)?)?;
    let replies = args
        .reply
        .iter()
        .map(|s| {
            let (game, mv) = s.split_once(':').ok_or("reply must be game:move")?;
            Ok((game.parse::<usize>()?, mv.to_string()))
        })
        .collect::<Result<BTreeMap<_, _>, Box<dyn std::error::Error>>>()?;
    let mut probes = vec![];
    for case in cases.cases {
        let mut p = Position::from_fen(&case.fen)?;
        probes.push(probe(format!("game{}:root", case.game), &p));
        let mv = p
            .parse_uci_move(&case.played)
            .ok_or("invalid played move")?;
        p.make_move(mv);
        probes.push(probe(
            format!("game{}:after_{}", case.game, case.played),
            &p,
        ));
        if let Some(reply) = replies.get(&case.game) {
            let mv = p.parse_uci_move(reply).ok_or("invalid reply move")?;
            p.make_move(mv);
            probes.push(probe(
                format!("game{}:after_reply_{}", case.game, reply),
                &p,
            ));
        }
    }
    let mut labels = [0usize; 3];
    let mut zero_weight = 0;
    let mut non_one_hot = 0;
    let mut simulations = 0u64;
    let mut generations = (u32::MAX, 0);
    let mut confident = [0usize; 3]; // 总数，反向结果，和棋
    let mut games = BTreeMap::<(u32, u64), Game>::new();
    let mut pairs = if args.pair_coverage {
        vec![0u32; AZ_NNUE_INPUT_SIZE * AZ_NNUE_INPUT_SIZE]
    } else {
        vec![]
    };
    let mut inputs = HashMap::<([u64; 20], [u32; 7]), Moments>::new();
    for (index, s) in pool.iter_samples().enumerate() {
        let c = class(s);
        labels[c] += 1;
        zero_weight += usize::from(s.value_weight == 0.0);
        non_one_hot += usize::from(s.value_wdl[c] < 0.999);
        simulations += u64::from(s.search_simulations);
        generations.0 = generations.0.min(s.meta.generation_update);
        generations.1 = generations.1.max(s.meta.generation_update);
        if s.meta.root_q.abs() >= 0.7 {
            confident[0] += 1;
            confident[1] += usize::from(s.meta.root_q * s.value < -0.5);
            confident[2] += usize::from(c == 1);
        }
        let game = games
            .entry((s.meta.generation_update, s.meta.game_id))
            .or_insert_with(|| Game {
                all_draw: true,
                ..Game::default()
            });
        game.max_ply = game.max_ply.max(s.meta.ply);
        game.samples += 1;
        game.all_draw &= c == 1;
        let b = bits(&s.features);
        if args.pair_coverage {
            for (i, &a) in s.features.iter().enumerate() {
                for &b in &s.features[i + 1..] {
                    pairs[a.min(b) * AZ_NNUE_INPUT_SIZE + a.max(b)] += 1;
                }
            }
        }
        if args.input_collisions && s.value_weight > 0.0 {
            let m = inputs
                .entry((b, s.rule_context.map(f32::to_bits)))
                .or_default();
            m.labels[c] += 1;
            m.sum += f64::from(s.value);
            m.squares += f64::from(s.value).powi(2);
        }
        for p in &mut probes {
            let d = distance(&b, &p.original).min(distance(&b, &p.mirrored));
            if d == 0 {
                p.exact += 1;
                p.labels[c] += 1;
            }
            if p.nearest.len() < 4 || d < p.nearest.last().unwrap().0 {
                p.nearest.push((d, index));
                p.nearest.sort_unstable();
                p.nearest.truncate(4);
            }
        }
    }
    println!(
        "replay samples={} games={} generation={generations:?} labels_WDL={labels:?} zero_value_weight={zero_weight} non_one_hot={non_one_hot} avg_sims={:.2} confident_root(total/opposite/draw)={confident:?}",
        pool.sample_count(),
        games.len(),
        simulations as f64 / pool.sample_count().max(1) as f64
    );
    let cap_draws = games
        .values()
        .filter(|g| g.all_draw && g.max_ply == 199)
        .collect::<Vec<_>>();
    println!(
        "draw_tail_at_ply199 games={} samples={} (cutoff_candidates_only_not_confirmed)",
        cap_draws.len(),
        cap_draws.iter().map(|g| g.samples).sum::<usize>()
    );
    if args.input_collisions {
        let mut repeated = 0usize;
        let mut contradictory = 0usize;
        let mut repeated_samples = 0usize;
        let mut variance = 0.0f64;
        for m in inputs.values() {
            let n = m.labels.iter().sum::<u32>() as usize;
            if n > 1 {
                repeated += 1;
                repeated_samples += n;
                contradictory += usize::from(m.labels.iter().filter(|&&v| v > 0).count() > 1);
                variance += (m.squares - m.sum * m.sum / n as f64).max(0.0);
            }
        }
        println!(
            "input_collisions unique={} repeated_inputs={repeated} contradictory_inputs={contradictory} repeated_samples={repeated_samples} empirical_min_Q_rmse_all={:.6} empirical_min_Q_rmse_repeated={:.6}; value_input=canonical_board_plus_rule_context",
            inputs.len(),
            (variance / (pool.sample_count() - zero_weight).max(1) as f64).sqrt(),
            (variance / repeated_samples.max(1) as f64).sqrt()
        );
        drop(inputs);
    }
    // 仅保留各病例最邻近的少量样本。
    let wanted = probes
        .iter()
        .flat_map(|p| p.nearest.iter().map(|(_, i)| *i))
        .collect::<std::collections::BTreeSet<_>>();
    let selected = pool
        .iter_samples()
        .enumerate()
        .filter(|(i, _)| wanted.contains(i))
        .map(|(i, s)| (i, s))
        .collect::<BTreeMap<_, _>>();
    for p in probes {
        println!(
            "PROBE {} exact_board_or_mirror={} labels_WDL={:?} (rule_context_not_matched)",
            p.name, p.exact, p.labels
        );
        if args.pair_coverage {
            let features = (0..AZ_NNUE_INPUT_SIZE)
                .filter(|&f| p.original[f / 64] & (1 << (f % 64)) != 0)
                .collect::<Vec<_>>();
            let mut frequencies = vec![];
            for (i, &a) in features.iter().enumerate() {
                for &b in &features[i + 1..] {
                    let mut mirrored = [a, b];
                    mirror_sparse_features_az_canonical_file(&mut mirrored);
                    let mut count = pairs[a * AZ_NNUE_INPUT_SIZE + b];
                    if mirrored != [a, b] {
                        count += pairs[mirrored[0] * AZ_NNUE_INPUT_SIZE + mirrored[1]];
                    }
                    frequencies.push(count);
                }
            }
            frequencies.sort_unstable();
            println!(
                "  pair_coverage pairs={} unseen={} min={} median={} (piece_square_cooccurrence_only)",
                frequencies.len(),
                frequencies.iter().filter(|&&n| n == 0).count(),
                frequencies.first().unwrap_or(&0),
                frequencies.get(frequencies.len() / 2).unwrap_or(&0)
            );
        }
        for (distance, index) in p.nearest {
            let s = selected[&index];
            println!(
                "  nearest feature_difference={distance} generation={} game={} ply={} target={:?} root_target={:?} root_q={:.4} value_weight={} rule_context={:?}",
                s.meta.generation_update,
                s.meta.game_id,
                s.meta.ply,
                s.value_wdl,
                s.root_search_wdl,
                s.meta.root_q,
                s.value_weight,
                s.rule_context
            );
        }
    }
    let mut rng = SplitMix64::new(20260927);
    let sample = pool.sample_uniform(args.evaluate_samples, &mut rng);
    let mut error = 0.0f64;
    let mut root_error = 0.0f64;
    for s in &sample {
        let (wdl, _, _) = outputs_for_training_sample(&model, s).ok_or("invalid sample")?;
        let q = wdl[0] - wdl[2];
        error += f64::from(q - s.value).powi(2);
        root_error += f64::from(q - (s.root_search_wdl[0] - s.root_search_wdl[2])).powi(2);
    }
    println!(
        "sample_eval n={} terminal_Q_rmse={:.6} root_search_Q_rmse={:.6}",
        sample.len(),
        (error / sample.len().max(1) as f64).sqrt(),
        (root_error / sample.len().max(1) as f64).sqrt()
    );
    if let Some(path) = args.teacher_output.as_ref() {
        use std::io::Write;
        let mut output = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        writeln!(
            output,
            "generation\tgame\tply\tfen\tterminal_q\tstored_search_q\tmodel_q\tpf_cp\tpf_mate"
        )?;
        let mut engine = pikafish::Engine::new(&args.pikafish)?;
        let mut count = 0;
        let mut strong = [0usize; 4]; // 强评估总数、反向终局、和棋终局、反向网络
        for s in sample.iter().filter(|s| {
            (20..80).contains(&s.meta.ply)
                && s.rule_context[0] < 0.5
                && s.rule_context[1..].iter().all(|&v| v == 0.0)
                && s.repetition_flags.iter().all(|&v| v == 0)
        }) {
            if count >= args.teacher_samples {
                break;
            }
            let p = position_for_training_sample(s).ok_or("invalid teacher sample")?;
            let fen = p.to_fen();
            let mut fields = fen
                .split_whitespace()
                .map(str::to_string)
                .collect::<Vec<_>>();
            fields[4] = (s.rule_context[0] * 120.0).round().to_string();
            let fen = fields.join(" ");
            let pf = engine.score(&format!("position fen {fen}"), args.teacher_depth, None)?;
            let (wdl, _, _) = outputs_for_training_sample(&model, s).unwrap();
            let q = wdl[0] - wdl[2];
            if pf.cp.abs() >= 700 {
                strong[0] += 1;
                strong[1] += usize::from(pf.cp as f32 * s.value < 0.0);
                strong[2] += usize::from(s.value == 0.0);
                strong[3] += usize::from(pf.cp as f32 * q < 0.0);
            }
            writeln!(
                output,
                "{}\t{}\t{}\t{fen}\t{}\t{}\t{q}\t{}\t{:?}",
                s.meta.generation_update,
                s.meta.game_id,
                s.meta.ply,
                s.value,
                s.meta.root_q,
                pf.cp,
                pf.mate
            )?;
            count += 1;
        }
        println!(
            "teacher_probe n={count} depth={} strong(total/opposite_terminal/draw_terminal/opposite_model)={strong:?}; no_repetition_samples_only,full_history_not_reconstructed",
            args.teacher_depth
        );
    }
    Ok(())
}
