use crate::cli::args::MateClassifyArgs;
use chineseai::xiangqi::{Color, MateKind, Position, square_name};

pub(crate) fn run(cmd: MateClassifyArgs) {
    let fen = cmd.fen.join(" ");
    let position = Position::from_fen(&fen).unwrap_or_else(|err| {
        eprintln!("FEN 无效：{err}");
        std::process::exit(2);
    });
    println!("{}", describe(&position));
}

fn describe(position: &Position) -> String {
    let Some(result) = position.classify_mate() else {
        return "终局：未识别到将死或困毙\n胜方：未判定".into();
    };
    let kind = match result.kind {
        MateKind::Checkmate => "将死",
        MateKind::Stalemate => "困毙",
    };
    let winner = match result.winner {
        Color::Red => "红方",
        Color::Black => "黑方",
    };
    let patterns = result
        .patterns
        .iter()
        .map(|pattern| pattern.name())
        .collect::<Vec<_>>()
        .join("、");
    let checkers = result
        .checkers
        .into_iter()
        .map(square_name)
        .collect::<Vec<_>>()
        .join("、");
    format!(
        "终局：{kind}\n胜方：{winner}\n杀法：{}\n将军棋子：{}",
        if patterns.is_empty() {
            "无匹配标签"
        } else {
            &patterns
        },
        if checkers.is_empty() {
            "无"
        } else {
            &checkers
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::args::{Cli, CliCommand};
    use clap::Parser;

    const DOUBLE_CANNON_FEN: &str = "3aka3/9/9/4C4/9/4C4/9/9/9/4K4 b";

    #[test]
    fn accepts_quoted_and_split_full_fen() {
        let full_fen = format!("{DOUBLE_CANNON_FEN} - - 0 1");
        for fen_args in [
            vec![full_fen.as_str()],
            full_fen.split_whitespace().collect(),
        ] {
            let mut argv = vec!["chineseai", "mate-classify"];
            argv.extend(fen_args);
            let cli = Cli::try_parse_from(argv).unwrap();
            let Some(CliCommand::MateClassify(cmd)) = cli.command else {
                panic!("应解析为 mate-classify 命令");
            };
            assert_eq!(cmd.fen.join(" "), full_fen);
            let output = describe(&Position::from_fen(&cmd.fen.join(" ")).unwrap());
            assert!(output.contains("终局：将死"));
            assert!(output.contains("胜方：红方"));
            assert!(output.contains("重炮"));
            assert!(output.contains("将军棋子：e4"));
        }
    }

    #[test]
    fn requires_fen_and_does_not_assign_winner_to_start_position() {
        assert!(Cli::try_parse_from(["chineseai", "mate-classify"]).is_err());
        assert_eq!(
            describe(&Position::startpos()),
            "终局：未识别到将死或困毙\n胜方：未判定"
        );
    }
}
