//! 用真实终局轨迹微调迁移模型的目标分支，普通参数保持冻结。
use chineseai::{
    az::{
        AzMatingGoal, AzNnue, AzTrainLossWeights, SplitMix64, samples_from_mating_trajectory,
        train_samples_weighted,
    },
    xiangqi::{Color, Move, Position},
};
use clap::Parser;
use serde::Deserialize;
use std::{error::Error, fs, io, path::PathBuf};

#[derive(Parser)]
struct Args {
    model: PathBuf,
    trajectories: PathBuf,
    output: PathBuf,
    #[arg(long, default_value_t = 20)]
    epochs: usize,
    #[arg(long, default_value_t = 512)]
    batch_size: usize,
    #[arg(long, default_value_t = 0.0004)]
    lr: f32,
}
#[derive(Deserialize)]
struct Input {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    fen: String,
    #[serde(default)]
    prefix_moves: Vec<String>,
    moves: Vec<String>,
    target: chineseai::az::MatingTarget,
    side: Color,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if args.output.exists() {
        return Err("拒绝覆盖已有模型".into());
    }
    let mut model = AzNnue::load(&args.model)?;
    if !model.is_goal_adaptation_only() {
        return Err("请先迁移为目标微调模型，禁止此工具更新普通棋力参数".into());
    }
    let input: Input = toml::from_str(&fs::read_to_string(args.trajectories)?)?;
    let mut samples = Vec::new();
    let mut shortest = std::collections::BTreeMap::new();
    let mut accepted = 0;
    for (game_id, case) in input.cases.into_iter().enumerate() {
        let mut position = Position::from_fen(&case.fen)?;
        let mut history = position.initial_rule_history();
        for text in &case.prefix_moves {
            let mv = Move::from_uci(text).ok_or("非法走子格式")?;
            if !position.legal_moves_with_rules(&history).contains(&mv) {
                return Err(format!("{} 的前缀非法：{text}", case.name).into());
            }
            history.push(position.rule_history_entry_after_move(mv));
            position.make_move(mv);
        }
        let moves = case
            .moves
            .iter()
            .map(|text| Move::from_uci(text).ok_or("非法走子格式"))
            .collect::<Result<Vec<_>, _>>()?;
        let goal = AzMatingGoal {
            target: case.target,
            side: case.side,
            strength: 0.4,
        };
        let key = format!(
            "{}|{:?}|{:?}|{:?}",
            case.fen, case.prefix_moves, goal.target, goal.side
        );
        match samples_from_mating_trajectory(position, history, &moves, goal, game_id as u64 + 1) {
            Ok(rows) => {
                if rows.iter().any(|sample| sample.goal_target == 1.0) {
                    let previous = shortest.entry(key).or_insert_with(Vec::new);
                    if previous.is_empty() || rows.len() < previous.len() {
                        *previous = rows;
                    }
                } else {
                    samples.extend(rows);
                    accepted += 1;
                }
            }
            Err(error) => return Err(format!("{}: {error}", case.name).into()),
        }
    }
    for rows in shortest.into_values() {
        samples.extend(rows);
        accepted += 1;
    }
    if samples.is_empty() {
        return Err("没有有效终局轨迹".into());
    }
    let positives = samples
        .iter()
        .filter(|sample| sample.goal_target > 0.0)
        .count();
    println!(
        "目标微调：轨迹={accepted} 样本={} 成功样本={positives} epochs={}",
        samples.len(),
        args.epochs
    );
    let stats = train_samples_weighted(
        &mut model,
        &samples,
        args.epochs,
        args.lr,
        args.batch_size,
        &mut SplitMix64::new(20261103),
        AzTrainLossWeights::default(),
    )
    .map_err(io::Error::other)?;
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?;
    if let Err(error) = model.save(&args.output) {
        let _ = fs::remove_file(&args.output);
        return Err(error.into());
    }
    println!(
        "目标CE={:.4} 策略CE={:.4} 输出={}",
        stats.goal_loss,
        stats.policy_ce,
        args.output.display()
    );
    Ok(())
}
