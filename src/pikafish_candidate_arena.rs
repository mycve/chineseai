//! 项目自有 Pikafish 形状网络的配对开局晋级赛。

use crate::{
    ab::{AbArenaReport, AbSearchLimits, pikafish_candle::PikafishModel, search_pikafish_model},
    xiangqi::{Color, Position, RuleOutcome},
};

#[derive(Clone, Copy, Debug)]
pub struct CandidateArenaConfig {
    /// 每个开局各执红、黑一局，故总局数是此值的两倍。
    pub pairs: usize,
    pub nodes: usize,
    pub max_depth: usize,
    pub max_plies: usize,
    pub promotion_rate: f32,
    pub confidence_z: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateArenaDecision {
    Promote,
    Reject,
    Inconclusive,
}

#[derive(Clone, Copy, Debug)]
pub struct CandidateArenaResult {
    pub report: AbArenaReport,
    pub lower_bound: f32,
    pub upper_bound: f32,
    pub decision: CandidateArenaDecision,
}

/// `openings` 中的每个局面先由候选执红、再由候选执黑。
/// 每对使用不同开局，未终局的截断对局返回错误，避免当成和棋晋级。
pub fn play_paired(
    candidate: &PikafishModel,
    champion: &PikafishModel,
    openings: &[Position],
    config: CandidateArenaConfig,
) -> Result<CandidateArenaResult, String> {
    if config.pairs == 0 || config.nodes == 0 || config.max_depth == 0 || config.max_plies == 0 {
        return Err("pairs, nodes, max_depth and max_plies must be positive".into());
    }
    if !config.promotion_rate.is_finite()
        || !(0.0..=1.0).contains(&config.promotion_rate)
        || !config.confidence_z.is_finite()
        || config.confidence_z < 0.0
    {
        return Err("invalid promotion rate or confidence z".into());
    }
    validate_openings(openings, config.pairs)?;
    let mut report = AbArenaReport::default();
    for pair in 0..config.pairs {
        let opening = openings[pair].clone();
        let red = play_game(&opening, candidate, champion, config)?;
        let black = play_game(&opening, champion, candidate, config)?;
        add_game(&mut report, red, true);
        add_game(&mut report, -black, false);
        let pair_score = (outcome_score(red) + outcome_score(-black)) / 2.0;
        report.paired_openings += 1;
        report.paired_score_sum += pair_score;
        report.paired_score_sq_sum += pair_score * pair_score;
    }
    let decision = decide(&report, config.promotion_rate, config.confidence_z);
    let (lower_bound, upper_bound) = paired_score_bounds(&report, config.confidence_z);
    Ok(CandidateArenaResult {
        report,
        lower_bound,
        upper_bound,
        decision,
    })
}

fn validate_openings(openings: &[Position], pairs: usize) -> Result<(), String> {
    if openings.len() < pairs {
        return Err(format!(
            "need {pairs} distinct openings, got {}",
            openings.len()
        ));
    }
    let mut seen = std::collections::HashSet::new();
    for opening in &openings[..pairs] {
        if !opening.has_general(Color::Red) || !opening.has_general(Color::Black) {
            return Err("opening lacks a general".into());
        }
        if opening
            .rule_outcome_with_history(&opening.initial_rule_history())
            .is_some()
            || opening.legal_moves().is_empty()
        {
            return Err("opening is already terminal".into());
        }
        if !seen.insert(opening.to_fen()) {
            return Err("duplicate opening position".into());
        }
    }
    Ok(())
}

pub fn decide(
    report: &AbArenaReport,
    promotion_rate: f32,
    confidence_z: f32,
) -> CandidateArenaDecision {
    if report.paired_openings < 2 || report.total_games() != report.paired_openings * 2 {
        return CandidateArenaDecision::Inconclusive;
    }
    let (lower, upper) = paired_score_bounds(report, confidence_z);
    if lower > promotion_rate {
        CandidateArenaDecision::Promote
    } else if upper < promotion_rate {
        CandidateArenaDecision::Reject
    } else {
        CandidateArenaDecision::Inconclusive
    }
}

/// 每个红黑交换的开局视为一个 [0,1] 样本；Wilson 界在全胜、全负时仍保留不确定性。
pub fn paired_score_bounds(report: &AbArenaReport, z: f32) -> (f32, f32) {
    let n = report.paired_openings as f32;
    if n == 0.0 {
        return (0.0, 1.0);
    }
    let z = z.max(0.0);
    let p = (report.paired_score_sum / n).clamp(0.0, 1.0);
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denom;
    let radius = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denom;
    ((center - radius).max(0.0), (center + radius).min(1.0))
}

fn add_game(report: &mut AbArenaReport, red_value: f32, candidate_red: bool) {
    if red_value > 0.0 {
        report.wins += 1;
        if candidate_red {
            report.wins_as_red += 1
        } else {
            report.wins_as_black += 1
        }
    } else if red_value < 0.0 {
        report.losses += 1;
        if candidate_red {
            report.losses_as_red += 1
        } else {
            report.losses_as_black += 1
        }
    } else {
        report.draws += 1;
    }
}

fn outcome_score(value: f32) -> f32 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        0.0
    } else {
        0.5
    }
}

fn play_game(
    opening: &Position,
    red: &PikafishModel,
    black: &PikafishModel,
    config: CandidateArenaConfig,
) -> Result<f32, String> {
    let mut position = opening.clone();
    let mut history = position.initial_rule_history();
    for _ in 0..config.max_plies {
        if let Some(result) = position.rule_outcome_with_history(&history) {
            return Ok(outcome_value(result));
        }
        let legal = position.legal_moves_with_rules(&history);
        if legal.is_empty() {
            return Ok(if position.side_to_move() == Color::Red {
                -1.0
            } else {
                1.0
            });
        }
        let model = if position.side_to_move() == Color::Red {
            red
        } else {
            black
        };
        let result = search_pikafish_model(
            &position,
            &history,
            model,
            AbSearchLimits {
                nodes: config.nodes,
                max_depth: config.max_depth,
            },
        )?;
        let mv = result.best_move.ok_or("AB search returned no bestmove")?;
        if !legal.contains(&mv) {
            return Err("AB search returned illegal bestmove".into());
        }
        history.push(position.rule_history_entry_after_move(mv));
        position.make_move(mv);
        if !position.has_general(Color::Red) {
            return Ok(-1.0);
        }
        if !position.has_general(Color::Black) {
            return Ok(1.0);
        }
    }
    if let Some(outcome) = position.rule_outcome_with_history(&history) {
        return Ok(outcome_value(outcome));
    }
    if position.legal_moves_with_rules(&history).is_empty() {
        return Ok(if position.side_to_move() == Color::Red {
            -1.0
        } else {
            1.0
        });
    }
    Err("arena game reached max_plies without result".into())
}

fn outcome_value(outcome: RuleOutcome) -> f32 {
    match outcome {
        RuleOutcome::Draw(_) => 0.0,
        RuleOutcome::Win(Color::Red) => 1.0,
        RuleOutcome::Win(Color::Black) => -1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_gate_requires_evidence() {
        let mut report = AbArenaReport::default();
        assert_eq!(
            decide(&report, 0.5, 1.96),
            CandidateArenaDecision::Inconclusive
        );
        report.wins = 4;
        report.paired_openings = 2;
        report.paired_score_sum = 2.0;
        report.paired_score_sq_sum = 2.0;
        assert_eq!(
            decide(&report, 0.5, 1.96),
            CandidateArenaDecision::Inconclusive
        );
        report.wins = 16;
        report.paired_openings = 8;
        report.paired_score_sum = 8.0;
        report.paired_score_sq_sum = 8.0;
        assert_eq!(decide(&report, 0.5, 1.96), CandidateArenaDecision::Promote);
        report.wins = 0;
        report.losses = 16;
        report.paired_score_sum = 0.0;
        report.paired_score_sq_sum = 0.0;
        assert_eq!(decide(&report, 0.5, 1.96), CandidateArenaDecision::Reject);
    }

    #[test]
    fn candidate_colors_are_counted_correctly() {
        let mut report = AbArenaReport::default();
        add_game(&mut report, 1.0, true);
        add_game(&mut report, -1.0, false);
        assert_eq!((report.wins_as_red, report.losses_as_black), (1, 1));
    }

    #[test]
    fn duplicate_openings_are_rejected() {
        let opening = Position::startpos();
        assert!(validate_openings(&[opening.clone(), opening], 2).is_err());
        assert!(validate_openings(&[], 1).is_err());
    }
}
