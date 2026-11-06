//! 用本引擎规则验证残局候选，生成与完整历史绑定的小型标签库；不是完整残局库生成器。
use chineseai::{
    az::{
        MateSearchLimits, TablebaseLabel, search_root_mate_profiled, write_verified_tablebase_label,
    },
    xiangqi::{Position, StepOutcome, XiangqiEnv},
};
use clap::Parser;
use rusqlite::Connection;
use std::{
    fs::File,
    io::{self, BufRead, BufReader},
    path::PathBuf,
};

#[derive(Parser)]
#[command(about = "验证终局/连将强制胜并写入本地训练标签；unknown绝不标和")]
struct Args {
    /// UTF-8，每行 FEN；可在 TAB 后附空格分隔的 UCI 历史着法。
    input: PathBuf,
    database: PathBuf,
    #[arg(long, default_value_t = 9)]
    max_plies: usize,
    #[arg(long, default_value_t = 200_000)]
    max_nodes: usize,
    /// 自然限着半回合数，0 表示禁用。必须与训练配置一致。
    #[arg(long, default_value_t = 120)]
    rule60_max_ply: u16,
    #[arg(long)]
    no_repetition_draw: bool,
}

fn environment(line: &str, rule60: u16, repetition_draw: bool) -> io::Result<XiangqiEnv> {
    let (fen, moves) = line.split_once('\t').unwrap_or((line, ""));
    let mut position = Position::from_fen(fen.trim())
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    position.set_rule60_max_ply((rule60 > 0).then_some(rule60));
    position.set_repetition_draw_enabled(repetition_draw);
    let mut environment = XiangqiEnv::from_position(position);
    for text in moves.split_whitespace() {
        environment.step_uci(text).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("历史中含非法着法：{text}"),
            )
        })?;
    }
    Ok(environment)
}

fn verify(environment: &XiangqiEnv, limits: MateSearchLimits) -> (Option<TablebaseLabel>, bool) {
    let value = match environment.current_outcome() {
        StepOutcome::Draw(_) => Some(0),
        StepOutcome::Win(winner) => Some(if winner == environment.position().side_to_move() {
            1
        } else {
            -1
        }),
        StepOutcome::Ongoing => None,
    };
    if let Some(value) = value {
        return (
            Some(TablebaseLabel {
                value,
                moves_left: Some(0),
            }),
            false,
        );
    }
    let report =
        search_root_mate_profiled(environment.position(), environment.rule_history(), limits);
    // check-only 最短杀距不是全宽 DTM；只写入已经证明的胜负，不用于 MLH。
    (
        report.solution.map(|_| TablebaseLabel {
            value: 1,
            moves_left: None,
        }),
        report.budget_exhausted,
    )
}

fn run(args: Args) -> io::Result<()> {
    if !(1..=63).contains(&args.max_plies) || !(1..=100_000_000).contains(&args.max_nodes) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "max-plies须为1..63，max-nodes须为1..100000000",
        ));
    }
    let input = BufReader::new(File::open(&args.input)?);
    let mut connection = Connection::open(&args.database).map_err(io::Error::other)?;
    let transaction = connection.transaction().map_err(io::Error::other)?;
    transaction
        .execute_batch(chineseai::az::TABLEBASE_SCHEMA)
        .map_err(io::Error::other)?;
    let limits = MateSearchLimits {
        max_plies: args.max_plies,
        max_nodes: args.max_nodes,
    };
    let (mut verified, mut unknown, mut exhausted) = (0, 0, 0);
    for (index, line) in input.lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let environment = environment(line, args.rule60_max_ply, !args.no_repetition_draw)
            .map_err(|err| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("第{}行：{err}", index + 1),
                )
            })?;
        let (label, budget_exhausted) = verify(&environment, limits);
        if let Some(label) = label {
            write_verified_tablebase_label(
                &transaction,
                environment.position(),
                environment.rule_history(),
                label,
                if label.moves_left == Some(0) {
                    "chineseai-terminal-v1"
                } else {
                    "chineseai-check-mate-v1"
                },
            )?;
            verified += 1;
        } else {
            unknown += 1;
            exhausted += usize::from(budget_exhausted);
        }
    }
    transaction.commit().map_err(io::Error::other)?;
    println!("verified={verified} unknown={unknown} node_budget_exhausted={exhausted}");
    Ok(())
}

fn main() -> io::Result<()> {
    run(Args::parse())
}

#[cfg(test)]
mod tests {
    use super::*;
    const LIMITS: MateSearchLimits = MateSearchLimits {
        max_plies: 5,
        max_nodes: 20_000,
    };

    #[test]
    fn verifies_terminal_draw_but_never_converts_unknown_to_draw() {
        let draw = environment("4k4/9/9/9/9/9/9/9/9/3K5 w", 120, true).unwrap();
        assert_eq!(
            verify(&draw, LIMITS).0,
            Some(TablebaseLabel {
                value: 0,
                moves_left: Some(0)
            })
        );
        let opening = XiangqiEnv::startpos();
        assert!(verify(&opening, LIMITS).0.is_none());
    }

    #[test]
    fn restores_real_history_and_rejects_illegal_moves() {
        let start = Position::startpos().to_fen();
        let continued = environment(&format!("{start}\ta3a4 a6a5"), 120, true).unwrap();
        assert_eq!(continued.rule_history().len(), 3);
        assert!(environment(&format!("{start}\ta3a9"), 120, true).is_err());
    }

    #[test]
    fn proves_win_without_claiming_exact_dtm() {
        let mate = environment(
            "2bakab2/9/5r1c1/p1PRC1p2/4P2nP/6P2/4N1r2/7c1/4A4/2BAK1B1R b - - 0 1",
            120,
            true,
        )
        .unwrap();
        assert_eq!(
            verify(
                &mate,
                MateSearchLimits {
                    max_plies: 15,
                    max_nodes: 200_000
                }
            )
            .0,
            Some(TablebaseLabel {
                value: 1,
                moves_left: None
            })
        );
    }
}
