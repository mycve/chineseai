//! 使用项目自己的 Pikafish 形状价值网络和共用 AB 搜索生成自博弈数据。

use std::{
    fs::File,
    io::{self, BufWriter, Write},
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

use crate::{
    ab::{AbSearchLimits, pikafish_candle::PikafishCpuModel, search_pikafish_model},
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

#[derive(Debug)]
pub struct SelfplaySample {
    pub ply: usize,
    pub fen: String,
    pub bestmove: String,
    pub score_cp: i32,
    pub side: Color,
}

#[derive(Debug)]
pub struct SelfplayGame {
    pub samples: Vec<SelfplaySample>,
    pub red_result: Option<f32>,
    pub termination: &'static str,
    pub search_nodes: usize,
}

impl SelfplayGame {
    pub fn labels(&self) -> impl Iterator<Item = (&str, f32)> {
        self.samples.iter().filter_map(|sample| {
            self.red_result.map(|value| {
                (
                    sample.fen.as_str(),
                    if sample.side == Color::Red {
                        value
                    } else {
                        -value
                    },
                )
            })
        })
    }

    pub fn result_text(&self) -> &'static str {
        match self.red_result {
            Some(value) if value > 0.0 => "1",
            Some(value) if value < 0.0 => "0",
            Some(_) => "1/2",
            None => "?",
        }
    }
}

/// 对局在内存中完成后才发送给训练器，不经 TSV 往返，也不发布半局标签。
pub fn play_game(
    model: &PikafishCpuModel,
    start: Position,
    config: CandidateSelfplayConfig,
    stop: &AtomicBool,
    mut on_search: impl FnMut(usize),
) -> io::Result<SelfplayGame> {
    let mut position = start;
    let mut history = position.initial_rule_history();
    let mut samples = Vec::new();
    let mut random = config.seed;
    let mut search_nodes = 0;
    let (red_result, termination) = loop {
        if stop.load(Ordering::Relaxed) {
            break (None, "interrupted");
        }
        if !position.has_general(Color::Red) {
            break (Some(-1.0), "general");
        }
        if !position.has_general(Color::Black) {
            break (Some(1.0), "general");
        }
        if let Some(outcome) = position.rule_outcome_with_history(&history) {
            break (
                Some(match outcome {
                    RuleOutcome::Draw(_) => 0.0,
                    RuleOutcome::Win(Color::Red) => 1.0,
                    RuleOutcome::Win(Color::Black) => -1.0,
                }),
                "rule",
            );
        }
        let legal = position.legal_moves_with_rules(&history);
        if legal.is_empty() {
            break (
                Some(if position.side_to_move() == Color::Red {
                    -1.0
                } else {
                    1.0
                }),
                "no_legal_moves",
            );
        }
        if history.len().saturating_sub(1) >= config.max_plies {
            break (None, "max_plies");
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
            search_nodes += search.nodes;
            on_search(search.nodes);
            let mv = search
                .best_move
                .ok_or_else(|| io::Error::other("AB search returned no bestmove"))?;
            if !legal.contains(&mv) {
                return Err(io::Error::other("AB search returned illegal bestmove"));
            }
            samples.push(SelfplaySample {
                ply: history.len() - 1,
                fen: position.to_fen(),
                bestmove: mv.to_uci(),
                score_cp: q_to_training_cp(search.value_q),
                side: position.side_to_move(),
            });
            mv
        };
        history.push(position.rule_history_entry_after_move(mv));
        position.make_move(mv);
    };
    Ok(SelfplayGame {
        samples,
        red_result,
        termination,
        search_nodes,
    })
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
    model: &PikafishCpuModel,
    output: &Path,
    config: CandidateSelfplayConfig,
) -> io::Result<CandidateSelfplayReport> {
    generate_from_openings(model, output, config, &[], &AtomicBool::new(false), |_| {})
}

/// 单独导出 TSV 时复用内存对局接口；中断对局没有训练标签。
pub fn generate_from_openings(
    model: &PikafishCpuModel,
    output: &Path,
    config: CandidateSelfplayConfig,
    openings: &[Position],
    stop: &AtomicBool,
    mut progress: impl FnMut(&CandidateSelfplayReport),
) -> io::Result<CandidateSelfplayReport> {
    if config.games == 0 || config.nodes == 0 || config.max_depth == 0 || config.max_plies == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "games, nodes, max_depth and max_plies must be positive",
        ));
    }
    if !openings.is_empty() && openings.len() != config.games {
        return Err(io::Error::other("开局数与自博弈局数不一致"));
    }
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(File::create(output)?);
    writeln!(
        writer,
        "game\tply\tfen\tbestmove\tscore_cp\tred_result\ttermination\tsource"
    )?;
    let mut report = CandidateSelfplayReport::default();
    for game in 0..config.games {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let position = openings
            .get(game)
            .cloned()
            .unwrap_or_else(Position::startpos);
        let game_data = play_game(
            model,
            position,
            CandidateSelfplayConfig {
                seed: config.seed ^ game as u64,
                ..config
            },
            stop,
            |_| {},
        )?;
        let result = game_data.result_text();
        for sample in &game_data.samples {
            writeln!(
                writer,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\tcandidate",
                game + 1,
                sample.ply,
                sample.fen,
                sample.bestmove,
                sample.score_cp,
                result,
                game_data.termination
            )?;
        }
        report.games += 1;
        report.positions += game_data.samples.len();
        match result {
            "?" => report.truncated += 1,
            "1/2" => report.draws += 1,
            _ => report.decisive += 1,
        }
        writer.flush()?;
        progress(&report);
    }
    writer.flush()?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::q_to_training_cp;
    use crate::pikafish_pretrain::cp_to_value;

    #[test]
    fn incomplete_games_publish_no_labels_and_completed_labels_follow_color() {
        use super::{SelfplayGame, SelfplaySample};
        use crate::xiangqi::{Color, Position};
        let mut game = SelfplayGame {
            samples: vec![
                SelfplaySample {
                    ply: 0,
                    fen: Position::startpos().to_fen(),
                    bestmove: "h2e2".into(),
                    score_cp: 100,
                    side: Color::Red,
                },
                SelfplaySample {
                    ply: 1,
                    fen: String::new(),
                    bestmove: "h7e7".into(),
                    score_cp: -100,
                    side: Color::Black,
                },
            ],
            red_result: None,
            termination: "interrupted",
            search_nodes: 10,
        };
        assert_eq!(game.labels().count(), 0);
        game.red_result = Some(1.0);
        assert_eq!(
            game.labels().map(|(_, value)| value).collect::<Vec<_>>(),
            vec![1.0, -1.0]
        );
    }

    #[test]
    fn search_q_roundtrips_through_training_column() {
        for q in [-0.8, -0.2, 0.0, 0.2, 0.8] {
            assert!((cp_to_value(q_to_training_cp(q)) - q).abs() < 0.001);
        }
    }
}
