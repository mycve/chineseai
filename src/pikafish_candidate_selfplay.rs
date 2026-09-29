//! 使用项目自己的 Pikafish 形状价值网络和共用 AB 搜索生成自博弈数据。

use std::{
    fs::File,
    io::{self, BufWriter, Write},
    path::Path,
};

use crate::{
    ab::{AbSearchLimits, pikafish_candle::PikafishModel, search_pikafish_model},
    xiangqi::{Color, Position, RuleOutcome},
};

#[derive(Clone, Copy, Debug)]
pub struct CandidateSelfplayConfig {
    pub games: usize,
    pub nodes: usize,
    pub max_depth: usize,
    pub max_plies: usize,
    pub opening_plies: usize,
    pub seed: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CandidateSelfplayReport {
    pub games: usize,
    pub decisive: usize,
    pub draws: usize,
    pub truncated: usize,
    pub positions: usize,
}

fn next_random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn q_to_training_cp(q: f32) -> i32 {
    ((q.clamp(-0.95, 0.95).atanh() * 600.0).round()) as i32
}

/// 输出与 `pikafish-selfplay` 相同的评分列，供项目自己的浮点模型继续训练。
/// `score_cp` 是 `tanh(cp/600)` 的逆映射，不是 Pikafish 引擎给出的真实厘兵分。
pub fn generate(
    model: &PikafishModel,
    output: &Path,
    config: CandidateSelfplayConfig,
) -> io::Result<CandidateSelfplayReport> {
    if config.games == 0 || config.nodes == 0 || config.max_depth == 0 || config.max_plies == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "games, nodes, max_depth and max_plies must be positive",
        ));
    }
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(File::create(output)?);
    writeln!(
        writer,
        "game\tply\tfen\tbestmove\tscore_cp\tred_result\ttermination"
    )?;
    let mut report = CandidateSelfplayReport::default();
    let mut random = config.seed;
    for game in 0..config.games {
        let mut position = Position::startpos();
        let mut history = position.initial_rule_history();
        let mut samples = Vec::new();
        let (result, termination) = loop {
            if !position.has_general(Color::Red) {
                break ("0", "general");
            }
            if !position.has_general(Color::Black) {
                break ("1", "general");
            }
            if let Some(outcome) = position.rule_outcome_with_history(&history) {
                break match outcome {
                    RuleOutcome::Draw(_) => ("1/2", "rule"),
                    RuleOutcome::Win(Color::Red) => ("1", "rule"),
                    RuleOutcome::Win(Color::Black) => ("0", "rule"),
                };
            }
            let legal = position.legal_moves_with_rules(&history);
            if legal.is_empty() {
                break (
                    if position.side_to_move() == Color::Red {
                        "0"
                    } else {
                        "1"
                    },
                    "no_legal_moves",
                );
            }
            if history.len().saturating_sub(1) >= config.max_plies {
                break ("?", "max_plies");
            }
            let mv = if history.len().saturating_sub(1) < config.opening_plies {
                legal[next_random(&mut random) as usize % legal.len()]
            } else {
                let search = search_pikafish_model(
                    &position,
                    &history,
                    model,
                    AbSearchLimits {
                        nodes: config.nodes,
                        max_depth: config.max_depth,
                    },
                )
                .map_err(io::Error::other)?;
                let mv = search
                    .best_move
                    .ok_or_else(|| io::Error::other("AB search returned no bestmove"))?;
                if !legal.contains(&mv) {
                    return Err(io::Error::other("AB search returned illegal bestmove"));
                }
                samples.push((
                    history.len() - 1,
                    position.to_fen(),
                    mv.to_uci(),
                    q_to_training_cp(search.value_q),
                ));
                mv
            };
            history.push(position.rule_history_entry_after_move(mv));
            position.make_move(mv);
        };
        for (ply, fen, bestmove, score_cp) in &samples {
            writeln!(
                writer,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}",
                game + 1,
                ply,
                fen,
                bestmove,
                score_cp,
                result,
                termination
            )?;
        }
        report.games += 1;
        report.positions += samples.len();
        match result {
            "?" => report.truncated += 1,
            "1/2" => report.draws += 1,
            _ => report.decisive += 1,
        }
    }
    writer.flush()?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::q_to_training_cp;
    use crate::pikafish_pretrain::cp_to_value;

    #[test]
    fn search_q_roundtrips_through_training_column() {
        for q in [-0.8, -0.2, 0.0, 0.2, 0.8] {
            assert!((cp_to_value(q_to_training_cp(q)) - q).abs() < 0.001);
        }
    }
}
