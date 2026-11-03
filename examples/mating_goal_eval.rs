//! 固定模型、局面、预算的配对实战评测；不改写权重，不生成训练标签。
use chineseai::{
    az::{AzMatingGoal, AzNnue, AzSearchLimits, MatingTarget, alphazero_search_with_rules},
    xiangqi::{Color, Position, RuleHistoryEntry, RuleOutcome},
};
use clap::Parser;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    error::Error,
    fs,
    io::{self, Write},
    path::PathBuf,
    sync::OnceLock,
};

#[derive(Parser)]
struct Args {
    model: PathBuf,
    #[arg(long, default_value = "eval/mating-goal-cases.toml")]
    cases: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 800)]
    simulations: usize,
    #[arg(long, default_value_t = 200)]
    max_plies: usize,
    #[arg(long, default_value_t = 0.8)]
    strength: f32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Suite {
    #[serde(default)]
    targets: Vec<MatingTarget>,
    cases: Vec<Case>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    stage: String,
    fen: String,
    #[serde(default)]
    target: MatingTarget,
    side: Color,
    /// 从 FEN 起顺序执行，保留中局的真实规则历史。
    #[serde(default)]
    moves: Vec<String>,
    /// 仅限制首步，用于隔离“普通一手杀”和“指定杀法路线”的冲突。
    #[serde(default)]
    root_moves: Vec<String>,
}

#[derive(Clone, Serialize)]
struct Game {
    mode: &'static str,
    outcome: &'static str,
    hit: bool,
    simulations: usize,
    moves: Vec<String>,
    final_fen: String,
    patterns: Vec<String>,
}

type BaselineKey = (String, Vec<String>, Vec<String>, u8);
type Baseline = OnceLock<Result<Game, String>>;

fn baseline_key(case: &Case) -> BaselineKey {
    (
        case.fen.clone(),
        case.moves.clone(),
        case.root_moves.clone(),
        case.side as u8,
    )
}

#[derive(Serialize)]
struct Pair {
    name: String,
    stage: String,
    target: MatingTarget,
    side: Color,
    initial_fen: String,
    goal_available: bool,
    /// 包含现有几何先验，不能单凭此值声称网络学到了目标表征。
    prior_total_variation: f32,
    learned_prior_total_variation: f32,
    network_goal_probability: f32,
    ordinary_wdl: [f32; 3],
    conditioned_wdl: [f32; 3],
    ordinary: Game,
    conditioned: Game,
}

#[derive(Serialize)]
struct Report {
    model: String,
    model_bytes: u64,
    simulations_per_move: usize,
    max_plies: usize,
    strength: f32,
    note: &'static str,
    pairs: Vec<Pair>,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

impl Suite {
    fn expand(self) -> Result<Vec<Case>, io::Error> {
        let mut cases = Vec::new();
        for case in self.cases {
            if case.target != MatingTarget::None {
                cases.push(case);
            } else {
                if self.targets.is_empty() || self.targets.contains(&MatingTarget::None) {
                    return Err(invalid("未指定目标的局面必须提供非空具名 targets"));
                }
                for &target in &self.targets {
                    cases.push(Case {
                        target,
                        ..case.clone()
                    });
                }
            }
        }
        Ok(cases)
    }
}

fn probabilities(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let values = logits
        .iter()
        .map(|x| ((x - max) / 1.4).exp())
        .collect::<Vec<_>>();
    let sum = values.iter().sum::<f32>();
    values.into_iter().map(|x| x / sum).collect()
}

fn terminal_game(
    position: &Position,
    outcome: Option<RuleOutcome>,
    goal: AzMatingGoal,
    mode: &'static str,
    simulations: usize,
    moves: Vec<String>,
) -> Game {
    let mate = position.classify_mate();
    let won = matches!(outcome, Some(RuleOutcome::Win(side)) if side == goal.side);
    Game {
        mode,
        outcome: match outcome {
            Some(RuleOutcome::Win(_)) if won => "win",
            Some(RuleOutcome::Win(_)) => "loss",
            Some(RuleOutcome::Draw(_)) => "draw",
            None => "cutoff",
        },
        // 两组都按同一个指定目标识别，普通组不能因为关闭指令就永远算失败。
        hit: won && mate.as_ref().is_some_and(|mate| goal.matches_mate(mate)),
        simulations,
        moves,
        final_fen: position.to_fen(),
        patterns: mate.map_or_else(Vec::new, |mate| {
            mate.patterns
                .into_iter()
                .map(|p| p.name().to_owned())
                .collect()
        }),
    }
}

fn play(
    model: &AzNnue,
    initial: &Position,
    initial_history: &[RuleHistoryEntry],
    case: &Case,
    args: &Args,
    enabled: bool,
) -> Result<Game, io::Error> {
    let mut position = initial.clone();
    let mut history = initial_history.to_vec();
    let goal = AzMatingGoal {
        target: case.target,
        side: case.side,
        strength: args.strength,
    };
    let mode = if enabled { "conditioned" } else { "ordinary" };
    let mut moves = Vec::new();
    let mut simulations = 0;
    for ply in 0..=args.max_plies {
        let outcome = position.rule_outcome_with_history(&history);
        if outcome.is_some() {
            return Ok(terminal_game(
                &position,
                outcome,
                goal,
                mode,
                simulations,
                moves,
            ));
        }
        let legal = position.legal_moves_with_rules(&history);
        if legal.is_empty() {
            return Ok(terminal_game(
                &position,
                Some(RuleOutcome::Win(position.side_to_move().opposite())),
                goal,
                mode,
                simulations,
                moves,
            ));
        }
        // 最后一手恰好将死也必须计胜，不能在检查无合法着之前按步数截断。
        if ply == args.max_plies {
            return Ok(terminal_game(
                &position,
                None,
                goal,
                mode,
                simulations,
                moves,
            ));
        }
        let root_moves = if ply == 0 && !case.root_moves.is_empty() {
            case.root_moves
                .iter()
                .map(|text| {
                    position
                        .parse_uci_move(text)
                        .filter(|mv| legal.contains(mv))
                        .ok_or_else(|| invalid(format!("{}：非法首步 {text}", case.name)))
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            legal
        };
        let result = alphazero_search_with_rules(
            &position,
            Some(history.clone()),
            Some(root_moves.clone()),
            model,
            AzSearchLimits {
                simulations: args.simulations,
                seed: ply as u64,
                // 普通组与目标组共用预算，关闭噪声、温度采样、KLD 提前收敛。
                // 规则终局及已证明的结果仍按现有搜索正常停止。
                mating_goal: if enabled { goal } else { AzMatingGoal::NONE },
                ..AzSearchLimits::default()
            },
        );
        simulations += result.simulations;
        let mv = result
            .best_move
            .filter(|mv| root_moves.contains(mv))
            .ok_or_else(|| invalid(format!("{}：{mode} 搜索未返回合法走法", case.name)))?;
        let mover = position.side_to_move();
        let captured = position.piece_at(mv.to as usize);
        position.make_move(mv);
        history.push(position.rule_history_entry_after_moved(mover, mv, captured));
        moves.push(mv.to_string());
    }
    unreachable!()
}

fn evaluate(
    model: &AzNnue,
    case: &Case,
    args: &Args,
    baseline: &Baseline,
) -> Result<Pair, io::Error> {
    let (position, history) = initial_state(case)?;
    if case.target == MatingTarget::None || position.rule_outcome_with_history(&history).is_some() {
        return Err(invalid(format!(
            "{}：必须指定具名目标且从未结束局面开始",
            case.name
        )));
    }
    let goal = AzMatingGoal {
        target: case.target,
        side: case.side,
        strength: args.strength,
    };
    let legal = position.legal_moves_with_rules(&history);
    let (_, ordinary_wdl, ordinary_logits) =
        model.evaluate_with_goal(&position, &history, &legal, AzMatingGoal::NONE);
    let (network_goal_probability, conditioned_wdl, conditioned_logits) =
        model.evaluate_with_goal(&position, &history, &legal, goal);
    let prior_total_variation = probabilities(&ordinary_logits)
        .iter()
        .zip(probabilities(&conditioned_logits))
        .map(|(left, right)| (left - right).abs())
        .sum::<f32>()
        * 0.5;
    let learned_logits = conditioned_logits
        .iter()
        .zip(&legal)
        .map(|(logit, mv)| logit - goal.policy_bias(&position, *mv))
        .collect::<Vec<_>>();
    let learned_prior_total_variation = probabilities(&ordinary_logits)
        .iter()
        .zip(probabilities(&learned_logits))
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        * 0.5;
    let mut ordinary = baseline
        .get_or_init(|| {
            play(model, &position, &history, case, args, false).map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|e| invalid(e.clone()))?
        .clone();
    // 共用普通棋谱，按当前目标重新计分；不能沿用首次缓存时的目标命中标签。
    ordinary.hit = ordinary.outcome == "win"
        && Position::from_fen(&ordinary.final_fen)
            .map_err(invalid)?
            .classify_mate()
            .is_some_and(|mate| goal.matches_mate(&mate));
    let conditioned = play(model, &position, &history, case, args, true)?;
    Ok(Pair {
        name: case.name.clone(),
        stage: case.stage.clone(),
        target: case.target,
        side: case.side,
        initial_fen: position.to_fen_with_history(&history),
        goal_available: goal.effective(&position).enabled(),
        prior_total_variation,
        learned_prior_total_variation,
        network_goal_probability,
        ordinary_wdl,
        conditioned_wdl,
        ordinary,
        conditioned,
    })
}

fn initial_state(case: &Case) -> Result<(Position, Vec<RuleHistoryEntry>), io::Error> {
    let mut position = Position::from_fen(&case.fen).map_err(invalid)?;
    let mut history = position.initial_rule_history();
    for text in &case.moves {
        let mv = position
            .parse_uci_move(text)
            .filter(|mv| {
                position.rule_outcome_with_history(&history).is_none()
                    && position.legal_moves_with_rules(&history).contains(mv)
            })
            .ok_or_else(|| invalid(format!("{}：非法前缀走法 {text}", case.name)))?;
        let mover = position.side_to_move();
        let captured = position.piece_at(mv.to as usize);
        position.make_move(mv);
        history.push(position.rule_history_entry_after_moved(mover, mv, captured));
    }
    Ok((position, history))
}

fn run(args: Args) -> Result<(), Box<dyn Error + Send + Sync>> {
    if args.simulations == 0
        || args.max_plies == 0
        || !args.strength.is_finite()
        || !(0.0 < args.strength && args.strength <= 1.0)
    {
        return Err(invalid("预算必须为正，强度必须在 (0,1] 内").into());
    }
    if args.output.exists() {
        return Err(invalid("拒绝覆盖已有评测报告").into());
    }
    let suite: Suite = toml::from_str(&fs::read_to_string(&args.cases)?)?;
    let cases = suite.expand()?;
    if cases.is_empty() {
        return Err(invalid("评测局面为空").into());
    }
    let model = AzNnue::load(&args.model)?;
    let baselines = cases
        .iter()
        .map(|case| (baseline_key(case), OnceLock::new()))
        .collect::<BTreeMap<_, _>>();
    let pairs = cases
        .par_iter()
        .map(|case| {
            let result = evaluate(&model, case, &args, &baselines[&baseline_key(case)])?;
            eprintln!(
                "{} [{}] 普通={}/{} 目标={}/{} 步数={}/{}",
                case.name,
                case.target.name(),
                result.ordinary.outcome,
                result.ordinary.hit,
                result.conditioned.outcome,
                result.conditioned.hit,
                result.ordinary.moves.len(),
                result.conditioned.moves.len()
            );
            Ok(result)
        })
        .collect::<Result<Vec<_>, io::Error>>()?;
    // 分阶段报告，不能把容易的一手杀兑现率混成全局控制成功率。
    let mut stages = pairs.iter().map(|p| p.stage.as_str()).collect::<Vec<_>>();
    stages.sort_unstable();
    stages.dedup();
    for stage in stages {
        let rows = pairs
            .iter()
            .filter(|p| p.stage == stage)
            .collect::<Vec<_>>();
        let count = |enabled: bool, test: fn(&Game) -> bool| {
            rows.iter()
                .filter(|p| test(if enabled { &p.conditioned } else { &p.ordinary }))
                .count()
        };
        println!(
            "{stage}: positions={} pairs={} 普通命中={} 目标命中={} 普通胜={} 目标胜={} 普通截断={} 目标截断={}",
            rows.iter()
                .map(|p| p.initial_fen.as_str())
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            rows.len(),
            count(false, |g| g.hit),
            count(true, |g| g.hit),
            count(false, |g| g.outcome == "win"),
            count(true, |g| g.outcome == "win"),
            count(false, |g| g.outcome == "cutoff"),
            count(true, |g| g.outcome == "cutoff")
        );
    }
    let report = Report {
        model: args.model.canonicalize()?.display().to_string(),
        model_bytes: fs::metadata(&args.model)?.len(),
        simulations_per_move: args.simulations,
        max_plies: args.max_plies,
        strength: args.strength,
        note: "固定模型、同一初始局面及搜索预算；无噪声、无温度采样；对手同权重、按现有普通争胜搜索。逐局保留全部着法；截断单列。prior_total_variation 含几何偏置，learned_prior_total_variation 扣除了几何偏置；两者均不单独证明全局控制能力。教学杀局只诊断兑现能力，不代表开局成功率。相同局面的普通基线可被多个目标复用，pairs 不是独立开局数。",
        pairs,
    };
    let body = toml::to_string_pretty(&report)?;
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?;
    output.write_all(body.as_bytes())?;
    println!("报告：{}", args.output.display());
    Ok(())
}

fn main() {
    if let Err(error) = run(Args::parse()) {
        eprintln!("评测失败：{error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_ordinary_transcript_is_rescored_for_each_target() {
        let suite: Suite = toml::from_str(include_str!("../eval/mating-goal-cases.toml")).unwrap();
        let mut case = suite.cases[0].clone();
        case.root_moves = vec!["h4e4".to_owned()];
        let mut other = case.clone();
        other.target = MatingTarget::HorseCannon;
        assert_eq!(baseline_key(&case), baseline_key(&other));
        let args = Args {
            model: PathBuf::new(),
            cases: PathBuf::new(),
            output: PathBuf::new(),
            simulations: 128,
            max_plies: 1,
            strength: 0.8,
        };
        let model = AzNnue::random(8, 20261103);
        let baseline = OnceLock::new();
        let first = evaluate(&model, &case, &args, &baseline).unwrap();
        let second = evaluate(&model, &other, &args, &baseline).unwrap();
        assert!(first.ordinary.hit);
        assert!(!second.ordinary.hit);
        assert_eq!(first.ordinary.moves, second.ordinary.moves);
        assert_eq!(first.ordinary.final_fen, second.ordinary.final_fen);
    }

    #[test]
    fn paired_suites_expand_without_illegal_or_terminal_starts() {
        for (text, expected) in [
            (include_str!("../eval/mating-goal-paired.toml"), 56),
            (include_str!("../eval/mating-goal-book.toml"), 224),
            (include_str!("../eval/mating-goal-endgame.toml"), 43),
        ] {
            let cases = toml::from_str::<Suite>(text).unwrap().expand().unwrap();
            assert_eq!(cases.len(), expected);
            for case in cases {
                let (position, history) = initial_state(&case).unwrap();
                assert!(position.rule_outcome_with_history(&history).is_none());
                assert_ne!(case.target, MatingTarget::None);
            }
        }
    }

    #[test]
    fn every_named_target_has_a_legal_one_move_fixture() {
        let suite: Suite = toml::from_str(include_str!("../eval/mating-goal-cases.toml")).unwrap();
        let cases = suite
            .cases
            .iter()
            .filter(|case| case.stage == "mate_in_one")
            .collect::<Vec<_>>();
        assert_eq!(cases.len(), MatingTarget::ALL.len());
        for (case, target) in cases.into_iter().zip(MatingTarget::ALL) {
            assert_eq!(case.target, target);
            let position = Position::from_fen(&case.fen).unwrap();
            assert!(!position.in_check(case.side.opposite()), "{}", case.name);
            let goal = AzMatingGoal {
                target,
                side: case.side,
                strength: 0.8,
            };
            assert!(
                position.legal_moves().into_iter().any(|mv| {
                    let mut after = position.clone();
                    after.make_move(mv);
                    after
                        .classify_mate()
                        .is_some_and(|mate| goal.matches_mate(&mate))
                }),
                "{}",
                case.name
            );
        }
    }

    #[test]
    fn ordinary_game_is_scored_against_the_requested_pattern() {
        let position = Position::from_fen("2bakab2/9/9/4C4/9/4C4/9/9/9/4K4 b").unwrap();
        let goal = AzMatingGoal {
            target: MatingTarget::DoubleCannon,
            side: Color::Red,
            strength: 0.8,
        };
        let outcome = Some(RuleOutcome::Win(Color::Red));
        assert!(terminal_game(&position, outcome, goal, "ordinary", 0, vec![]).hit);
        assert!(!terminal_game(&position, None, goal, "ordinary", 0, vec![]).hit);
        assert!(
            !terminal_game(
                &position,
                outcome,
                AzMatingGoal {
                    side: Color::Black,
                    ..goal
                },
                "ordinary",
                0,
                vec![]
            )
            .hit
        );
    }
}
