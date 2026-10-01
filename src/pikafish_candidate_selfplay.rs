//! 使用项目自己的 Pikafish 形状价值网络和共用 AB 搜索生成自博弈数据。

use std::{
    fs::File,
    io::{self, BufWriter, Write},
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

use rand::{
    SeedableRng,
    distr::{Distribution, weighted::WeightedIndex},
    rngs::StdRng,
};
use serde::{Deserialize, Serialize};

use crate::{
    ab::{AbCandidate, AbSearchLimits, pikafish_candle::PikafishCpuModel, search_pikafish_model},
    xiangqi::{Color, Move, Position, RuleOutcome},
};

/// 温度单位是网络 q，概率为 exp((q - q_max) / T)。plies 从开局库起点算半回合。
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, clap::Args)]
#[serde(default, deny_unknown_fields)]
pub struct SelfplayTemperature {
    #[arg(long = "temperature-start", default_value_t = 0.05)]
    pub start: f32,
    #[arg(long = "temperature-end", default_value_t = 0.005)]
    pub end: f32,
    #[arg(long = "temperature-plies", default_value_t = 60)]
    pub plies: usize,
}
impl Default for SelfplayTemperature {
    fn default() -> Self {
        Self {
            start: 0.05,
            end: 0.005,
            plies: 60,
        }
    }
}
impl SelfplayTemperature {
    pub fn validate(&self) -> io::Result<()> {
        if !self.start.is_finite()
            || !self.end.is_finite()
            || self.end < 0.0
            || self.start < self.end
            || self.plies == 0
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "温度必须有限且 0 <= end <= start；plies 必须为正",
            ));
        }
        Ok(())
    }
    fn at(&self, ply: usize) -> f32 {
        let progress = ply.min(self.plies) as f64 / self.plies as f64;
        (self.start as f64 + (self.end as f64 - self.start as f64) * progress) as f32
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CandidateSelfplayConfig {
    pub games: usize,
    pub nodes: usize,
    pub max_depth: usize,
    pub max_plies: usize,
    pub temperature: SelfplayTemperature,
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
    pub playedmove: String,
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
    config.temperature.validate()?;
    let mut position = start;
    let mut history = position.initial_rule_history();
    let mut samples = Vec::new();
    let mut random = StdRng::seed_from_u64(config.seed);
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
        let ply = history.len() - 1;
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
        let candidate = sample_candidate(
            &search.candidates,
            search.best_move,
            config.temperature.at(ply),
            &mut random,
        )?;
        let mv = candidate.mv;
        if !legal.contains(&mv) {
            return Err(io::Error::other("AB search returned illegal move"));
        }
        samples.push(SelfplaySample {
            ply,
            fen: position.to_fen(),
            playedmove: mv.to_uci(),
            score_cp: q_to_training_cp(candidate.q),
            side: position.side_to_move(),
        });
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

fn sample_candidate<'a>(
    candidates: &'a [AbCandidate],
    best_move: Option<Move>,
    temperature: f32,
    random: &mut StdRng,
) -> io::Result<&'a AbCandidate> {
    if temperature == 0.0 {
        return candidates
            .iter()
            .find(|candidate| Some(candidate.mv) == best_move)
            .ok_or_else(|| io::Error::other("AB search returned no bestmove"));
    }
    let has_win = candidates
        .iter()
        .any(|candidate| candidate.solved == Some(1));
    let has_non_loss = candidates
        .iter()
        .any(|candidate| candidate.solved != Some(-1));
    let eligible: Vec<_> = candidates
        .iter()
        .filter(|candidate| {
            if has_win {
                candidate.solved == Some(1)
            } else {
                !has_non_loss || candidate.solved != Some(-1)
            }
        })
        .collect();
    let max = eligible
        .iter()
        .map(|candidate| candidate.q as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    let distribution = WeightedIndex::new(
        eligible
            .iter()
            .map(|candidate| ((candidate.q as f64 - max) / temperature as f64).exp()),
    )
    .map_err(io::Error::other)?;
    Ok(eligible[distribution.sample(random)])
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
        "game\tply\tfen\tplayedmove\tscore_cp\tred_result\ttermination\tsource"
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
                sample.playedmove,
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

    use super::*;
    fn choices(scores: &[(f32, Option<i8>)]) -> Vec<AbCandidate> {
        scores
            .iter()
            .enumerate()
            .map(|(i, &(q, solved))| AbCandidate {
                mv: Move {
                    from: 0,
                    to: (i + 1) as u8,
                },
                q,
                selection_weight: 0.0,
                solved,
            })
            .collect()
    }
    #[test]
    fn temperature_schedule_validates_and_anneals() {
        let schedule = SelfplayTemperature::default();
        schedule.validate().unwrap();
        assert_eq!(schedule.at(0), 0.05);
        assert!((schedule.at(30) - 0.0275).abs() < 1e-7);
        assert_eq!(schedule.at(60), 0.005);
        assert_eq!(schedule.at(1000), 0.005);
        for (start, end, plies) in [
            (f32::NAN, 0., 1),
            (f32::INFINITY, 0., 1),
            (0.1, -0.1, 1),
            (0.1, 0.2, 1),
            (0.1, 0., 0),
        ] {
            assert!(
                SelfplayTemperature { start, end, plies }
                    .validate()
                    .is_err()
            );
        }
    }
    #[test]
    fn softmax_frequencies_match_scores_and_seed_is_repeatable() {
        let candidates = choices(&[(0.2, None), (0.1, None)]);
        let best = Some(candidates[0].mv);
        let mut random = StdRng::seed_from_u64(123);
        let mut worse = 0;
        for _ in 0..50000 {
            if sample_candidate(&candidates, best, 0.1, &mut random)
                .unwrap()
                .mv
                == candidates[1].mv
            {
                worse += 1;
            }
        }
        let expected = 1.0 / (1.0 + 1f64.exp());
        assert!((worse as f64 / 50000.0 - expected).abs() < 0.01);
        let sequence = |seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            (0..100)
                .map(|_| {
                    sample_candidate(&candidates, best, 0.1, &mut rng)
                        .unwrap()
                        .mv
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(sequence(123), sequence(123));
        assert_ne!(sequence(123), sequence(456));
    }
    #[test]
    fn temperature_respects_proofs_and_cold_sampling_is_stable() {
        let mut random = StdRng::seed_from_u64(9);
        let candidates = choices(&[(0.1, None), (0.9, Some(-1))]);
        for _ in 0..100 {
            assert_eq!(
                sample_candidate(&candidates, Some(candidates[0].mv), 1000., &mut random)
                    .unwrap()
                    .mv,
                candidates[0].mv
            );
        }
        let candidates = choices(&[(0.1, Some(1)), (0.9, None), (0.8, Some(-1))]);
        for _ in 0..100 {
            assert_eq!(
                sample_candidate(&candidates, Some(candidates[0].mv), 1000., &mut random)
                    .unwrap()
                    .mv,
                candidates[0].mv
            );
        }
        let candidates = choices(&[(-0.95, None), (0.95, None)]);
        for temperature in [0., 1e-30] {
            assert_eq!(
                sample_candidate(
                    &candidates,
                    Some(candidates[1].mv),
                    temperature,
                    &mut random
                )
                .unwrap()
                .mv,
                candidates[1].mv
            );
        }
        let candidates = choices(&[(-1., Some(-1)), (-1., Some(-1))]);
        assert!(sample_candidate(&candidates, Some(candidates[0].mv), 0.05, &mut random).is_ok());
        assert!(sample_candidate(&[], None, 0.05, &mut random).is_err());
    }
    #[test]
    fn every_selfplay_move_is_searched_and_truncation_has_no_mc_labels() -> io::Result<()> {
        let model = crate::ab::pikafish_candle::PikafishModel::new(&candle_core::Device::Cpu)
            .map_err(io::Error::other)?
            .cpu_snapshot()
            .map_err(io::Error::other)?;
        let config = CandidateSelfplayConfig {
            games: 1,
            nodes: 64,
            max_depth: 2,
            max_plies: 2,
            temperature: SelfplayTemperature::default(),
            seed: 77,
        };
        let mut searched = Vec::new();
        let game = play_game(
            &model,
            Position::startpos(),
            config,
            &AtomicBool::new(false),
            |nodes| searched.push(nodes),
        )?;
        assert_eq!(searched.len(), 2);
        assert!(searched.iter().all(|&nodes| nodes > 0));
        assert_eq!(game.samples.len(), 2);
        assert_eq!(game.termination, "max_plies");
        assert_eq!(game.labels().count(), 0);
        for sample in &game.samples {
            let position = Position::from_fen(&sample.fen).map_err(io::Error::other)?;
            assert!(
                position
                    .legal_moves()
                    .iter()
                    .any(|mv| mv.to_uci() == sample.playedmove)
            );
        }
        let repeated = play_game(
            &model,
            Position::startpos(),
            config,
            &AtomicBool::new(false),
            |_| {},
        )?;
        let moves = |game: &SelfplayGame| {
            game.samples
                .iter()
                .map(|sample| sample.playedmove.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(moves(&game), moves(&repeated));
        Ok(())
    }

    #[test]
    fn incomplete_games_publish_no_labels_and_completed_labels_follow_color() {
        use super::{SelfplayGame, SelfplaySample};
        use crate::xiangqi::{Color, Position};
        let mut game = SelfplayGame {
            samples: vec![
                SelfplaySample {
                    ply: 0,
                    fen: Position::startpos().to_fen(),
                    playedmove: "h2e2".into(),
                    score_cp: 100,
                    side: Color::Red,
                },
                SelfplaySample {
                    ply: 1,
                    fen: String::new(),
                    playedmove: "h7e7".into(),
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
