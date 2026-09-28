use chineseai::az::{AzNnue, outputs_for_training_sample, px0_data};
use clap::Parser;
use std::{io::Write, path::Path};
#[path = "support/pikafish.rs"]
mod pikafish;
use pikafish::Engine;

#[derive(Parser)]
struct Args {
    #[arg(long, required = true)]
    model: Vec<String>,
    /// 新的 TSV 输出路径；已有文件不覆盖。
    #[arg(long)]
    output: String,
    #[arg(long, default_value_t = 256)]
    games: usize,
    #[arg(long, default_value_t = 20261002)]
    seed: u64,
    #[arg(long, default_value_t = 10)]
    depth: usize,
    #[arg(long, default_value = "data/data.bin")]
    data: String,
    #[arg(long, default_value = "tools/pikafish-avx2.exe")]
    pikafish: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let mut rows = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?;
    let probes = px0_data::load_teacher_probes(Path::new(&args.data), args.games, args.seed)?;
    let models = args
        .model
        .iter()
        .map(|p| AzNnue::load(p))
        .collect::<Result<Vec<_>, _>>()?;
    let mut engine = Engine::new(&args.pikafish)?;
    write!(
        rows,
        "game_id\tply\tfen\tteacherQ\tteacherD\tresultQ\tpfCP\tpfMate"
    )?;
    for path in &args.model {
        write!(rows, "\tmodelQ:{path}")?;
    }
    writeln!(rows)?;
    println!(
        "probe games={} seed={} depth={} models={:?} output={}",
        args.games, args.seed, args.depth, args.model, args.output
    );
    let mut strong = 0;
    let mut reverse = vec![0; models.len() + 1];
    let mut weak = reverse.clone();
    let mut neutral = reverse.clone();
    for (i, p) in probes.iter().enumerate() {
        let score = engine.score(&format!("position fen {}", p.fen), args.depth, None)?;
        let mut qs = vec![p.teacher_q];
        for model in &models {
            let (wdl, _) =
                outputs_for_training_sample(model, &p.sample).ok_or("sample eval failed")?;
            qs.push(wdl[0] - wdl[2]);
        }
        write!(
            rows,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:?}",
            p.sample.meta.game_id,
            p.sample.meta.ply,
            p.fen,
            p.teacher_q,
            p.teacher_d,
            p.result_q,
            if score.mate.is_none() {
                score.cp.to_string()
            } else {
                String::new()
            },
            score.mate
        )?;
        for q in &qs[1..] {
            write!(rows, "\t{q}")?;
        }
        writeln!(rows)?;
        if score.mate.is_some() || score.cp.abs() >= 600 {
            strong += 1;
            let sign = score.mate.unwrap_or(score.cp).signum() as f32;
            for (j, &q) in qs.iter().enumerate() {
                if q.abs() < 0.2 {
                    neutral[j] += 1;
                } else if q * sign < 0.0 {
                    reverse[j] += 1;
                } else if q.abs() < 0.5 {
                    weak[j] += 1;
                }
            }
        }
        if (i + 1) % 32 == 0 {
            println!(
                "scored={} strong={} reverse_teacher_then_models={:?} neutral_absQ_lt02={:?} aligned_weak_absQ_02_to05={:?}",
                i + 1,
                strong,
                reverse,
                neutral,
                weak
            );
        }
    }
    println!(
        "final probes={} depth={} strong={} reverse={:?} neutral={:?} aligned_weak={:?}",
        probes.len(),
        args.depth,
        strong,
        reverse,
        neutral,
        weak
    );
    Ok(())
}
