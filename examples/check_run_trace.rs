//! 从评测文件的完整棋谱检查连续将军后变招：区分可继续将军与被规则禁止。
use chineseai::xiangqi::{Color, Position};
use serde::Deserialize;
use std::{error::Error, fs, io};

#[derive(Deserialize)]
struct Input {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    fen: String,
    side: Color,
    #[serde(default)]
    moves: Vec<String>,
}

fn run() -> Result<(), Box<dyn Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "eval/mating-goal-endgame.toml".into());
    let input: Input = toml::from_str(&fs::read_to_string(path)?)?;
    let case = input
        .cases
        .into_iter()
        .max_by_key(|case| case.moves.len())
        .ok_or_else(|| io::Error::other("没有棋谱"))?;
    let mut position = Position::from_fen(&case.fen)?;
    let mut history = position.initial_rule_history();
    let mut streak = 0;
    let mut longest = 0;
    println!(
        "棋谱={} 指令方={:?} 半回合数={}",
        case.name,
        case.side,
        case.moves.len()
    );
    for (ply, text) in case.moves.iter().enumerate() {
        let legal = position.legal_moves_with_rules(&history);
        let mv = position
            .parse_uci_move(text)
            .filter(|mv| legal.contains(mv))
            .ok_or_else(|| io::Error::other(format!("第{}半回合非法：{text}", ply + 1)))?;
        let mover = position.side_to_move();
        let gives_check = position.gives_check_after_move_fast(mv);
        if mover == case.side {
            if !gives_check && streak >= 2 {
                let raw_checks = position
                    .legal_moves()
                    .into_iter()
                    .filter(|&mv| position.gives_check_after_move_fast(mv))
                    .count();
                let legal_checks = legal
                    .iter()
                    .filter(|&&mv| position.gives_check_after_move_fast(mv))
                    .count();
                println!(
                    "ply={} 连将={} 改走={} 可继续将军={} 被规则过滤的将军={} FEN={}",
                    ply + 1,
                    streak,
                    mv,
                    legal_checks,
                    raw_checks - legal_checks,
                    position.to_fen()
                );
            }
            streak = if gives_check { streak + 1 } else { 0 };
            longest = longest.max(streak);
        }
        let captured = position.piece_at(mv.to as usize);
        position.make_move(mv);
        history.push(position.rule_history_entry_after_moved(mover, mv, captured));
    }
    println!("最长连续将军={}；有合法将军可走不代表存在强制杀。", longest);
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("检查失败：{error}");
        std::process::exit(2);
    }
}
