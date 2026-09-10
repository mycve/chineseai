//! 独立的自举困难样本实验；不修改 az-loop 的数据或模型。
use chineseai::{
    az::{
        AzExperiencePool, AzNnue, AzSampleMeta, AzSearchLimits, AzSearchResult, AzTrainingSample,
        SHORT_VALUE_HEADS, SplitMix64, alphazero_search_with_rules, dense_move_index,
        rule_context_features, train_samples,
    },
    nnue::{canonical_move, extract_sparse_features_az},
    xiangqi::{Move, Position, RuleHistoryEntry, RuleOutcome},
};
use clap::{Parser, Subcommand};
use rayon::prelude::*;
use rusqlite::{Connection, params};
use std::{
    collections::HashSet,
    error::Error,
    fs,
    path::{Path, PathBuf},
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
#[derive(Parser)]
struct Cli {
    #[arg(long, default_value_t = 16)]
    threads: usize,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// 从自身对局采集、复核并保存独立数据集。输出目录必须不存在。
    Mine {
        model: PathBuf,
        output: PathBuf,
        #[arg(long, default_value_t = 1024)]
        games: usize,
        #[arg(long, default_value_t = 16)]
        rollouts: usize,
        #[arg(long, default_value_t = 20260912)]
        seed: u64,
    },
    /// 从已接受位置的两侧分支重新生成完整对局，导出真实终局价值样本。
    Expand {
        dataset: PathBuf,
        output: PathBuf,
        #[arg(long, default_value_t = 8)]
        games_per_branch: usize,
        #[arg(long, default_value_t = 20261021)]
        seed: u64,
    },
    /// 在指定数据集上评估模型，不训练或修改权重。
    Check {
        dataset: PathBuf,
        /// 仅用于整批均未参与训练的独立数据；评估所有接受位置。
        #[arg(long, requires = "exclude_training")]
        independent_all: bool,
        /// 剔除与补训数据池棋盘特征相同的测试位置。
        #[arg(long)]
        exclude_training: Option<PathBuf>,
        #[arg(required = true)]
        models: Vec<PathBuf>,
    },
    /// 固定样本预算对照：普通样本 vs 普通样本加少量成对策略监督。
    Fit {
        dataset: PathBuf,
        output: PathBuf,
        #[arg(long, default_value_t = 100_000)]
        samples: usize,
        #[arg(long, default_value_t = 0.05)]
        hard_fraction: f32,
        #[arg(long, default_value_t = 32)]
        max_repeats: usize,
        #[arg(long, default_value_t = 0.00001)]
        lr: f32,
        #[arg(long, default_value_t = 20260913)]
        seed: u64,
    },
}
#[derive(Clone)]
struct State {
    position: Position,
    history: Vec<RuleHistoryEntry>,
    moves: Vec<Move>,
    game: usize,
}
impl State {
    fn start(game: usize) -> Self {
        let mut position = Position::startpos();
        position.set_rule60_max_ply(Some(120));
        Self {
            history: position.initial_rule_history(),
            position,
            moves: Vec::new(),
            game,
        }
    }
    fn after(&self, mv: Move) -> Self {
        let mut s = self.clone();
        let who = s.position.side_to_move();
        let captured = s.position.piece_at(mv.to as usize);
        s.position.make_move(mv);
        s.history
            .push(s.position.rule_history_entry_after_moved(who, mv, captured));
        s.moves.push(mv);
        s
    }
    fn outcome(&self) -> Option<RuleOutcome> {
        self.position
            .rule_outcome_with_history(&self.history)
            .or_else(|| {
                self.position
                    .legal_moves_with_rules(&self.history)
                    .is_empty()
                    .then(|| RuleOutcome::Win(self.position.side_to_move().opposite()))
            })
    }
    fn heldout(&self) -> bool {
        self.game % 5 == 0
    }
}
fn search(model: &AzNnue, s: &State, simulations: usize, seed: u64, noise: f32) -> AzSearchResult {
    alphazero_search_with_rules(
        &s.position,
        Some(s.history.clone()),
        None,
        model,
        AzSearchLimits {
            simulations,
            seed,
            cpuct: 0.9,
            cpuct_at_root: 2.0,
            cpuct_factor: 1.5,
            cpuct_factor_at_root: 1.5,
            fpu_value: 0.15,
            fpu_value_at_root: 0.05,
            policy_softmax_temp: 1.3,
            root_dirichlet_total_concentration: 8.0,
            root_exploration_fraction: noise,
            ..Default::default()
        },
    )
}
fn verify(model: &AzNnue, s: &State, mv: Move, n: usize, seed: u64, noise: f32) -> f32 {
    let r = search(model, &s.after(mv), n, seed, noise);
    -r.best_move
        .and_then(|mv| r.candidates.iter().find(|c| c.mv == mv).map(|c| c.q))
        .unwrap_or(r.value_q)
}
fn sample(
    s: &State,
    moves: Vec<Move>,
    policy: Vec<f32>,
    value_weight: f32,
    n: usize,
) -> AzTrainingSample {
    AzTrainingSample {
        features: extract_sparse_features_az(&s.position),
        rule_context: rule_context_features(&s.position, &s.history),
        move_indices: moves
            .into_iter()
            .map(|m| dense_move_index(canonical_move(s.position.side_to_move(), m)))
            .collect(),
        policy,
        value_wdl: [0., 1., 0.],
        root_search_wdl: [0., 1., 0.],
        short_value_wdl: [[0., 1., 0.]; SHORT_VALUE_HEADS],
        value: 0.,
        side_sign: 1.,
        policy_weight: 1.,
        value_weight,
        search_simulations: n as u32,
        meta: AzSampleMeta {
            game_id: s.game as u64,
            ply: s.moves.len() as u16,
            ..Default::default()
        },
    }
}
fn play(model: &AzNnue, game: usize, seed: u64) -> (Vec<State>, Vec<AzTrainingSample>) {
    let mut s = State::start(game);
    let mut rng = SplitMix64::new(seed);
    let mut positions = Vec::new();
    let mut samples = Vec::new();
    let mut sides = Vec::new();
    for ply in 0..240 {
        if s.outcome().is_some() {
            break;
        }
        if ply >= 12 && ply % 12 == 0 {
            positions.push(s.clone());
        }
        let r = search(model, &s, 400, rng.next_u64(), 0.08);
        let mut mv = r.best_move.expect("nonterminal search");
        if ply < 24 {
            let mut x = rng.unit_f32();
            for c in &r.candidates {
                x -= c.policy;
                if x <= 0. {
                    mv = c.mv;
                    break;
                }
            }
        }
        let mut row = sample(
            &s,
            r.candidates.iter().map(|c| c.mv).collect(),
            r.candidates.iter().map(|c| c.policy).collect(),
            1.,
            400,
        );
        row.root_search_wdl = r.value_wdl;
        samples.push(row);
        sides.push(s.position.side_to_move());
        s = s.after(mv);
    }
    // 达到步数上限不伪造和棋标签。
    if let Some(outcome) = s.outcome() {
        for (row, side) in samples.iter_mut().zip(sides) {
            let wdl = match outcome {
                RuleOutcome::Draw(_) => [0., 1., 0.],
                RuleOutcome::Win(c) if c == side => [1., 0., 0.],
                _ => [0., 0., 1.],
            };
            row.value_wdl = wdl;
            row.value = wdl[0] - wdl[2];
            row.short_value_wdl = [wdl; SHORT_VALUE_HEADS];
        }
    } else {
        samples.clear();
    }
    (positions, samples)
}
fn rollout(model: &AzNnue, s: &State, mv: Move, seed: u64) -> Option<f64> {
    let who = s.position.side_to_move();
    let mut s = s.after(mv);
    let mut rng = SplitMix64::new(seed);
    for _ in 0..800 {
        if let Some(o) = s.outcome() {
            return Some(match o {
                RuleOutcome::Draw(_) => 0.5,
                RuleOutcome::Win(c) if c == who => 1.,
                _ => 0.,
            });
        }
        let r = search(model, &s, 1600, rng.next_u64(), 0.08);
        s = s.after(r.best_move.expect("nonterminal search"));
    }
    s.outcome().map(|o| match o {
        RuleOutcome::Draw(_) => 0.5,
        RuleOutcome::Win(c) if c == who => 1.,
        _ => 0.,
    })
}
fn paired_lower(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    if a.len() < 2 {
        return f64::NEG_INFINITY;
    }
    let n = a.len() as f64;
    let mean = a.iter().zip(b).map(|(x, y)| x - y).sum::<f64>() / n;
    let variance = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y - mean).powi(2))
        .sum::<f64>()
        / (n - 1.);
    mean - 1.96 * (variance / n).sqrt()
}
struct Finding {
    state: State,
    baseline: Move,
    candidate: Move,
    gaps: [f32; 3],
    scores: Vec<(Option<f64>, Option<f64>)>,
    lower: Option<f64>,
}
fn mine_position(model: &AzNnue, s: &State, rollouts: usize, seed: u64) -> Vec<Finding> {
    let root = search(model, s, 400, 0, 0.);
    let baseline = root.best_move.unwrap();
    let mut ranked = root.candidates;
    ranked.sort_by(|a, b| b.visits.cmp(&a.visits).then_with(|| b.q.total_cmp(&a.q)));
    let mut rest: Vec<_> = ranked.into_iter().skip(4).collect();
    let mut rng = SplitMix64::new(seed);
    let mut candidates = Vec::new();
    for _ in 0..4 {
        if rest.is_empty() {
            break;
        }
        let i = rng.next_u64() as usize % rest.len();
        candidates.push(rest.remove(i).mv);
    }
    let base256 = verify(model, s, baseline, 256, 0, 0.);
    let mut out = Vec::new();
    for candidate in candidates {
        let gap256 = verify(model, s, candidate, 256, 0, 0.) - base256;
        if gap256 < 0.2 {
            continue;
        }
        let gaps = [
            gap256,
            verify(model, s, candidate, 1600, 0, 0.) - verify(model, s, baseline, 1600, 0, 0.),
            verify(model, s, candidate, 6400, 0, 0.) - verify(model, s, baseline, 6400, 0, 0.),
        ];
        let stable = gaps[1..].iter().all(|&v| v > 0.1)
            && (0..3).all(|j| {
                verify(model, s, candidate, 1600, seed + j, 0.08)
                    - verify(model, s, baseline, 1600, seed + j, 0.08)
                    > 0.1
            });
        let scores: Vec<_> = if stable {
            (0..rollouts)
                .map(|i| {
                    (
                        rollout(model, s, baseline, seed + 100 + i as u64),
                        rollout(model, s, candidate, seed + 100 + i as u64),
                    )
                })
                .collect()
        } else {
            Vec::new()
        };
        let lower =
            if scores.len() == rollouts && scores.iter().all(|(a, b)| a.is_some() && b.is_some()) {
                Some(paired_lower(
                    &scores.iter().map(|(_, b)| b.unwrap()).collect::<Vec<_>>(),
                    &scores.iter().map(|(a, _)| a.unwrap()).collect::<Vec<_>>(),
                ))
            } else {
                None
            };
        out.push(Finding {
            state: s.clone(),
            baseline,
            candidate,
            gaps,
            scores,
            lower,
        });
    }
    out
}
fn save_pool(path: &Path, samples: Vec<AzTrainingSample>) -> Result<()> {
    let mut pool = AzExperiencePool::new(samples.len().max(1));
    pool.add_samples(samples);
    pool.save_snapshot_lz4(path)?;
    Ok(())
}
fn mine(
    model_path: PathBuf,
    output: PathBuf,
    games: usize,
    rollouts: usize,
    seed: u64,
) -> Result<()> {
    if games < 5 || rollouts < 16 {
        return Err("games >= 5 and rollouts >= 16 required".into());
    }
    fs::create_dir(&output)?;
    fs::copy(&model_path, output.join("source.safetensors"))?;
    let model = AzNnue::load(output.join("source.safetensors"))?;
    let mut db = Connection::open(output.join("discovery.sqlite"))?;
    db.execute_batch("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL); CREATE TABLE positions(id INTEGER PRIMARY KEY,game INTEGER,heldout INTEGER,fen TEXT,moves TEXT); CREATE TABLE pairs(position_id INTEGER,baseline TEXT,candidate TEXT,gap256 REAL,gap1600 REAL,gap6400 REAL,lower_bound REAL,accepted INTEGER); CREATE TABLE rollouts(position_id INTEGER,candidate TEXT,seed_index INTEGER,baseline_score REAL,candidate_score REAL);")?;
    for (key, value) in [
        ("format", "1".to_string()),
        ("seed", seed.to_string()),
        ("games", games.to_string()),
        ("rollouts", rollouts.to_string()),
        ("source_path", model_path.display().to_string()),
        ("status", "running".to_string()),
    ] {
        db.execute("INSERT INTO metadata VALUES(?1,?2)", params![key, value])?;
    }
    println!("generating {games} games");
    let generated: Vec<_> = (0..games)
        .into_par_iter()
        .map(|g| play(&model, g, seed + g as u64))
        .collect();
    let mut states = Vec::new();
    let mut normal = Vec::new();
    let mut validation = Vec::new();
    let mut seen = HashSet::new();
    for (snapshots, rows) in generated {
        for s in snapshots {
            if seen.insert(s.position.hash()) {
                states.push(s);
            }
        }
        for row in rows {
            if row.meta.game_id % 5 == 0 {
                validation.push(row);
            } else {
                normal.push(row);
            }
        }
    }
    // 保守移除训练集中与验证集相同的棋盘特征；完整规则历史仍保存用于复核。
    let heldout_features: HashSet<_> = validation
        .iter()
        .map(|r| r.features.clone())
        .chain(
            states
                .iter()
                .filter(|s| s.heldout())
                .map(|s| extract_sparse_features_az(&s.position)),
        )
        .collect();
    normal.retain(|r| !heldout_features.contains(&r.features));
    println!(
        "positions={} normal={} validation={}",
        states.len(),
        normal.len(),
        validation.len()
    );
    save_pool(&output.join("normal.lz4"), normal)?;
    save_pool(&output.join("validation.lz4"), validation)?;
    let findings: Vec<_> = states
        .par_iter()
        .enumerate()
        .map(|(i, s)| mine_position(&model, s, rollouts, seed + 1_000_000 + i as u64 * 1000))
        .collect();
    let transaction = db.transaction()?;
    let mut hard = Vec::new();
    let mut heldout = Vec::new();
    let mut screened = 0;
    for (id, (s, rows)) in states.iter().zip(findings).enumerate() {
        transaction.execute(
            "INSERT INTO positions VALUES(?1,?2,?3,?4,?5)",
            params![
                id,
                s.game,
                s.heldout(),
                s.position.to_fen(),
                s.moves
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(" ")
            ],
        )?;
        for r in rows {
            screened += 1;
            let accepted = r.lower.is_some_and(|v| v > 0.1);
            transaction.execute(
                "INSERT INTO pairs VALUES(?1,?2,?3,?4,?5,?6,?7,?8)",
                params![
                    id,
                    r.baseline.to_string(),
                    r.candidate.to_string(),
                    r.gaps[0],
                    r.gaps[1],
                    r.gaps[2],
                    r.lower,
                    accepted
                ],
            )?;
            for (i, (a, b)) in r.scores.iter().enumerate() {
                transaction.execute(
                    "INSERT INTO rollouts VALUES(?1,?2,?3,?4,?5)",
                    params![id, r.candidate.to_string(), i, a, b],
                )?;
            }
            if accepted {
                // 只监督已经比较过的两招的相对偏好，不声称候选是全局最佳，也不伪造价值标签。
                let row = sample(
                    &r.state,
                    vec![r.baseline, r.candidate],
                    vec![0., 1.],
                    0.,
                    6400,
                );
                if r.state.heldout() {
                    heldout.push(row);
                } else if !heldout_features.contains(&row.features) {
                    hard.push(row);
                }
            }
        }
    }
    // 困难验证集的棋盘也不能出现在困难训练集中。
    let heldout_hard: HashSet<_> = heldout.iter().map(|r| r.features.clone()).collect();
    hard.retain(|r| !heldout_hard.contains(&r.features));
    let summary = format!(
        "screened={screened} hard_train={} hard_validation={}",
        hard.len(),
        heldout.len()
    );
    save_pool(&output.join("hard.lz4"), hard)?;
    save_pool(&output.join("hard-validation.lz4"), heldout)?;
    transaction.execute(
        "UPDATE metadata SET value='complete' WHERE key='status'",
        [],
    )?;
    transaction.commit()?;
    println!("{summary}");
    Ok(())
}
fn load_pool(path: &Path) -> Result<Vec<AzTrainingSample>> {
    Ok(AzExperiencePool::load_snapshot_lz4(path, usize::MAX)?.all_samples())
}
fn policy_ce(model: &AzNnue, rows: &[AzTrainingSample]) -> f64 {
    let stats = chineseai::az::evaluate_policy_groups(model, rows);
    let n = stats.quiet_samples + stats.tactical_samples;
    if n == 0 {
        return f64::NAN;
    }
    (stats.quiet_ce as f64 * stats.quiet_samples as f64
        + stats.tactical_ce as f64 * stats.tactical_samples as f64)
        / n as f64
}

fn heldout_search_report(db: &Connection, model: &AzNnue) -> Result<String> {
    search_report(db, model, false, &HashSet::new())
}
fn search_report(
    db: &Connection,
    model: &AzNnue,
    independent_all: bool,
    excluded: &HashSet<Vec<usize>>,
) -> Result<String> {
    let mut statement = db.prepare("SELECT game,moves,fen,baseline,candidate FROM positions JOIN pairs ON positions.id=pairs.position_id WHERE (heldout=1 OR ?1) AND accepted=1")?;
    let rows = statement.query_map([independent_all], |r| {
        Ok((
            r.get::<_, usize>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, String>(2)?,
            r.get::<_, String>(3)?,
            r.get::<_, String>(4)?,
        ))
    })?;
    let mut total = 0;
    let mut selected = [0, 0];
    let mut overlap = 0;
    for row in rows {
        let (game, moves, fen, _, candidate) = row?;
        let mut state = State::start(game);
        for text in moves.split_whitespace() {
            let mv = state
                .position
                .parse_uci_move(text)
                .ok_or("invalid stored move")?;
            if !state
                .position
                .legal_moves_with_rules(&state.history)
                .contains(&mv)
            {
                return Err("illegal stored move".into());
            }
            state = state.after(mv);
        }
        if state.position.to_fen() != fen {
            return Err("stored history/FEN mismatch".into());
        }
        if excluded.contains(&extract_sparse_features_az(&state.position)) {
            overlap += 1;
            continue;
        }
        for (i, budget) in [400, 1600].into_iter().enumerate() {
            let result = search(model, &state, budget, 0, 0.);
            selected[i] += usize::from(
                result
                    .best_move
                    .is_some_and(|mv| mv.to_string() == candidate),
            );
        }
        total += 1;
    }
    Ok(format!(
        "independent_all={independent_all} excluded_pairs={overlap} candidate_selected_400={}/{} 1600={}/{}",
        selected[0], total, selected[1], total
    ))
}

fn replay_state(game: usize, moves: &str) -> Result<State> {
    let mut state = State::start(game);
    for text in moves.split_whitespace() {
        let mv = state
            .position
            .parse_uci_move(text)
            .ok_or("invalid stored move")?;
        if !state
            .position
            .legal_moves_with_rules(&state.history)
            .contains(&mv)
        {
            return Err("illegal stored move".into());
        }
        state = state.after(mv);
    }
    Ok(state)
}
fn terminal_target(outcome: RuleOutcome, side: chineseai::xiangqi::Color) -> [f32; 3] {
    match outcome {
        RuleOutcome::Draw(_) => [0., 1., 0.],
        RuleOutcome::Win(winner) if winner == side => [1., 0., 0.],
        _ => [0., 0., 1.],
    }
}
fn outcome_code(outcome: RuleOutcome) -> i32 {
    match outcome {
        RuleOutcome::Draw(_) => 0,
        RuleOutcome::Win(chineseai::xiangqi::Color::Red) => 1,
        _ => -1,
    }
}
struct Trajectory {
    root_id: usize,
    branch: String,
    heldout: bool,
    game: usize,
    seed: u64,
    start_ply: usize,
    moves: String,
    outcome: Option<RuleOutcome>,
    samples: Vec<AzTrainingSample>,
}
fn trajectory(model: &AzNnue, root_id: usize, root: &State, mv: Move, seed: u64) -> Trajectory {
    let mut state = root.after(mv);
    let start_ply = state.moves.len();
    let mut rng = SplitMix64::new(seed);
    let mut rows = Vec::new();
    let mut sides = Vec::new();
    for ply in 0..800 {
        if state.outcome().is_some() {
            break;
        }
        let result = search(model, &state, 1600, rng.next_u64(), 0.08);
        if ply % 4 == 0 {
            let mut row = sample(
                &state,
                result.candidates.iter().map(|c| c.mv).collect(),
                result.candidates.iter().map(|c| c.policy).collect(),
                1.,
                1600,
            );
            row.policy_weight = 0.;
            row.root_search_wdl = result.value_wdl;
            rows.push(row);
            sides.push(state.position.side_to_move());
        }
        state = state.after(result.best_move.expect("nonterminal search"));
    }
    let outcome = state.outcome();
    if let Some(o) = outcome {
        for (row, side) in rows.iter_mut().zip(sides) {
            let wdl = terminal_target(o, side);
            row.value_wdl = wdl;
            row.value = wdl[0] - wdl[2];
            row.short_value_wdl = [wdl; SHORT_VALUE_HEADS];
        }
    } else {
        rows.clear();
    }
    Trajectory {
        root_id,
        branch: mv.to_string(),
        heldout: root.heldout(),
        game: root.game,
        seed,
        start_ply,
        moves: state
            .moves
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" "),
        outcome,
        samples: rows,
    }
}
fn expand(dataset: PathBuf, output: PathBuf, games: usize, seed: u64) -> Result<()> {
    if games == 0 {
        return Err("games_per_branch must be positive".into());
    }
    let input = Connection::open(dataset.join("discovery.sqlite"))?;
    let status: String =
        input.query_row("SELECT value FROM metadata WHERE key='status'", [], |r| {
            r.get(0)
        })?;
    if status != "complete" {
        return Err("dataset incomplete".into());
    }
    let model = AzNnue::load(dataset.join("source.safetensors"))?;
    let mut statement=input.prepare("SELECT DISTINCT position_id,game,moves,baseline,candidate FROM positions JOIN pairs ON positions.id=position_id WHERE accepted=1 ORDER BY position_id,candidate")?;
    let mut jobs = Vec::new();
    let mut seen = HashSet::new();
    for row in statement.query_map([], |r| {
        Ok((
            r.get::<_, usize>(0)?,
            r.get::<_, usize>(1)?,
            r.get::<_, String>(2)?,
            r.get::<_, String>(3)?,
            r.get::<_, String>(4)?,
        ))
    })? {
        let (id, game, moves, base, candidate) = row?;
        let state = replay_state(game, &moves)?;
        for text in [base, candidate] {
            if seen.insert((id, text.clone())) {
                let mv = state
                    .position
                    .parse_uci_move(&text)
                    .ok_or("invalid branch")?;
                for index in 0..games {
                    jobs.push((
                        id,
                        state.clone(),
                        mv,
                        seed + id as u64 * 10000 + index as u64,
                    ));
                }
            }
        }
    }
    fs::create_dir(&output)?;
    for file in [
        "source.safetensors",
        "discovery.sqlite",
        "normal.lz4",
        "validation.lz4",
        "hard.lz4",
        "hard-validation.lz4",
    ] {
        fs::copy(dataset.join(file), output.join(file))?;
    }
    let mut db = Connection::open(output.join("discovery.sqlite"))?;
    db.execute(
        "UPDATE metadata SET value='expanding' WHERE key='status'",
        [],
    )?;
    db.execute_batch("CREATE TABLE trajectories(root_id INTEGER,game INTEGER,heldout INTEGER,branch TEXT,seed INTEGER,start_ply INTEGER,moves TEXT,outcome INTEGER)")?;
    println!("trajectory_games={}", jobs.len());
    let results: Vec<_> = jobs
        .par_iter()
        .map(|(id, s, mv, seed)| trajectory(&model, *id, s, *mv, *seed))
        .collect();
    let mut train = Vec::new();
    let mut val = Vec::new();
    let mut finished = 0;
    let tx = db.transaction()?;
    for r in results {
        tx.execute(
            "INSERT INTO trajectories VALUES(?1,?2,?3,?4,?5,?6,?7,?8)",
            params![
                r.root_id,
                r.game,
                r.heldout,
                r.branch,
                r.seed as i64,
                r.start_ply,
                r.moves,
                r.outcome.map(outcome_code)
            ],
        )?;
        if r.outcome.is_some() {
            finished += 1;
        }
        if r.heldout {
            val.extend(r.samples);
        } else {
            train.extend(r.samples);
        }
    }
    let mut normal = load_pool(&output.join("normal.lz4"))?;
    let validation = load_pool(&output.join("validation.lz4"))?;
    let mut hard = load_pool(&output.join("hard.lz4"))?;
    let hard_val = load_pool(&output.join("hard-validation.lz4"))?;
    let excluded: HashSet<_> = validation
        .iter()
        .chain(&hard_val)
        .chain(&val)
        .map(|r| r.features.clone())
        .collect();
    normal.retain(|r| !excluded.contains(&r.features));
    hard.retain(|r| !excluded.contains(&r.features));
    train.retain(|r| !excluded.contains(&r.features));
    println!(
        "completed={finished}/{} value_train={} value_validation={}",
        jobs.len(),
        train.len(),
        val.len()
    );
    save_pool(&output.join("normal.lz4"), normal)?;
    save_pool(&output.join("hard.lz4"), hard)?;
    save_pool(&output.join("value.lz4"), train)?;
    save_pool(&output.join("value-validation.lz4"), val)?;
    tx.execute(
        "UPDATE metadata SET value='complete' WHERE key='status'",
        [],
    )?;
    tx.commit()?;
    Ok(())
}
fn trajectory_value_report(db: &Connection, model: &AzNnue) -> Result<String> {
    let exists: bool = db.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE name='trajectories')",
        [],
        |r| r.get(0),
    )?;
    if !exists {
        return Ok("no_trajectory_validation".into());
    }
    let mut stmt=db.prepare("SELECT game,start_ply,moves,outcome FROM trajectories WHERE heldout=1 AND outcome IS NOT NULL")?;
    let rows = stmt
        .query_map([], |r| {
            Ok((
                r.get::<_, usize>(0)?,
                r.get::<_, usize>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, i32>(3)?,
            ))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let values: Vec<_> = rows
        .par_iter()
        .map(|(game, start, moves, outcome)| {
            let mut state = State::start(*game);
            let mut count = 0;
            let mut error = 0.;
            for (ply, text) in moves.split_whitespace().enumerate() {
                if ply >= *start && (ply - start) % 4 == 0 {
                    let target = if *outcome == 0 {
                        0.
                    } else if (*outcome == 1)
                        == (state.position.side_to_move() == chineseai::xiangqi::Color::Red)
                    {
                        1.
                    } else {
                        -1.
                    };
                    let r = search(model, &state, 0, 0, 0.);
                    let q = r.network_value_wdl[0] - r.network_value_wdl[2];
                    error += (q - target).abs() as f64;
                    count += 1;
                }
                state = state.after(
                    state
                        .position
                        .parse_uci_move(text)
                        .expect("stored legal trajectory"),
                );
            }
            (count, error)
        })
        .collect();
    let (count, error) = values.iter().fold((0, 0.), |a, b| (a.0 + b.0, a.1 + b.1));
    Ok(format!(
        "trajectory_value_rows={count} q_mae={}",
        error / count.max(1) as f64
    ))
}
fn fit(
    dataset: PathBuf,
    output: PathBuf,
    count: usize,
    fraction: f32,
    repeats: usize,
    lr: f32,
    seed: u64,
) -> Result<()> {
    if count < 1024
        || !fraction.is_finite()
        || !(0.0..=0.05).contains(&fraction)
        || !lr.is_finite()
        || lr <= 0.
        || repeats == 0
    {
        return Err("invalid fit budget/fraction/lr/repeat cap".into());
    }
    let db = Connection::open(dataset.join("discovery.sqlite"))?;
    let status: String =
        db.query_row("SELECT value FROM metadata WHERE key='status'", [], |r| {
            r.get(0)
        })?;
    if status != "complete" {
        return Err("dataset is incomplete".into());
    }
    let normal = load_pool(&dataset.join("normal.lz4"))?;
    let hard = load_pool(&dataset.join("hard.lz4"))?;
    let validation = load_pool(&dataset.join("validation.lz4"))?;
    let hard_validation = load_pool(&dataset.join("hard-validation.lz4"))?;
    if normal.is_empty() || hard.is_empty() || validation.is_empty() || hard_validation.is_empty() {
        return Err(
            "need nonempty normal/hard training and validation splits; mine more games".into(),
        );
    }
    fs::create_dir(&output)?;
    let hard_count = ((count as f32 * fraction) as usize).min(hard.len().saturating_mul(repeats));
    let mut rng = SplitMix64::new(seed);
    let control: Vec<_> = (0..count)
        .map(|_| normal[rng.next_u64() as usize % normal.len()].clone())
        .collect();
    let mut experiment = control.clone();
    let mut slots: Vec<_> = (0..count).collect();
    for i in (1..slots.len()).rev() {
        let j = rng.next_u64() as usize % (i + 1);
        slots.swap(i, j);
    }
    for i in 0..hard_count {
        experiment[slots[i]] = hard[i % hard.len()].clone();
    }
    let mut text = format!(
        "samples={count} hard={hard_count} fraction={} lr={lr} repeat_cap={repeats}\n",
        hard_count as f64 / count as f64
    );

    let mut arms = vec![("control", control), ("experiment", experiment)];
    if dataset.join("value.lz4").exists() {
        let mut value = load_pool(&dataset.join("value.lz4"))?;
        // 防止重复的轨迹棋盘占满价值配额，每个棋盘最多保留八条终局观测。
        let mut counts = std::collections::HashMap::new();
        value.retain(|r| {
            let n = counts.entry(r.features.clone()).or_insert(0usize);
            *n += 1;
            *n <= 8
        });
        let value_count = ((count as f32 * fraction) as usize)
            .saturating_sub(hard_count)
            .min(value.len());
        if value_count == 0 {
            return Err("no trajectory budget/data".into());
        }
        for i in (1..value.len()).rev() {
            let j = rng.next_u64() as usize % (i + 1);
            value.swap(i, j);
        }
        let mut joint = arms[1].1.clone();
        for i in 0..value_count {
            joint[slots[hard_count + i]] = value[i].clone();
        }
        arms.push(("joint", joint));
        text += &format!(
            "value_samples={value_count} total_extra={}\n",
            hard_count + value_count
        );
    }
    for (name, rows) in arms {
        let mut model = AzNnue::load(dataset.join("source.safetensors"))?;
        let mut rng = SplitMix64::new(seed);
        let before = policy_ce(&model, &hard_validation);
        let normal_before = policy_ce(&model, &validation);
        let search_before = heldout_search_report(&db, &model)?;
        let value_before = trajectory_value_report(&db, &model)?;
        let stats = train_samples(&mut model, &rows, 1, lr, 1024, &mut rng)
            .map_err(std::io::Error::other)?;
        model.save(output.join(format!("{name}.safetensors")))?;
        let after = policy_ce(&model, &hard_validation);
        let normal_after = policy_ce(&model, &validation);
        text += &format!(
            "{name}: loss={} heldout_pair_ce_before={before} after={after} normal_ce_before={normal_before} after={normal_after}\n",
            stats.loss
        );
        text += &format!(
            "{name} before: {search_before}\n{name} after: {}\n",
            heldout_search_report(&db, &model)?
        );
        text += &format!(
            "{name} value_before: {value_before}\n{name} value_after: {}\n",
            trajectory_value_report(&db, &model)?
        );
        fs::write(output.join("report.txt"), &text)?;
    }
    fs::write(output.join("report.txt"), &text)?;
    println!("{text}");
    Ok(())
}
fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.threads == 0 {
        return Err("threads must be positive".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(cli.threads)
        .build_global()?;
    match cli.command {
        Command::Expand {
            dataset,
            output,
            games_per_branch,
            seed,
        } => expand(dataset, output, games_per_branch, seed),
        Command::Check {
            dataset,
            models,
            independent_all,
            exclude_training,
        } => {
            let db = Connection::open(dataset.join("discovery.sqlite"))?;
            let mut excluded = HashSet::new();
            if let Some(training) = exclude_training {
                for file in ["normal.lz4", "hard.lz4", "value.lz4"] {
                    let path = training.join(file);
                    if file == "value.lz4" && !path.exists() {
                        continue;
                    }
                    for row in load_pool(&path)? {
                        excluded.insert(row.features);
                    }
                }
            }
            for path in models {
                let model = AzNnue::load(&path)?;
                println!(
                    "{}: {} {}",
                    path.display(),
                    search_report(&db, &model, independent_all, &excluded)?,
                    trajectory_value_report(&db, &model)?
                );
            }
            Ok(())
        }
        Command::Mine {
            model,
            output,
            games,
            rollouts,
            seed,
        } => mine(model, output, games, rollouts, seed),
        Command::Fit {
            dataset,
            output,
            samples,
            hard_fraction,
            max_repeats,
            lr,
            seed,
        } => fit(
            dataset,
            output,
            samples,
            hard_fraction,
            max_repeats,
            lr,
            seed,
        ),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mate_is_a_loss_not_an_unfinished_game() {
        let position = Position::from_fen(
            "3aka3/3nc4/4R4/p3n1p1p/2p1C4/2P3r2/P1c1P3P/6N2/4N4/2BAKAB2 w - - 0 1",
        )
        .unwrap();
        let winner = position.side_to_move();
        let mut state = State {
            history: position.initial_rule_history(),
            position,
            moves: Vec::new(),
            game: 0,
        };
        for text in ["e5h5", "c5c4", "h5h9"] {
            let mv = state.position.parse_uci_move(text).unwrap();
            assert!(
                state
                    .position
                    .legal_moves_with_rules(&state.history)
                    .contains(&mv)
            );
            state = state.after(mv);
        }
        assert_eq!(state.outcome(), Some(RuleOutcome::Win(winner)));
        // 续弈价值标签必须随执棋方换符号，将死后的待走方为负。
        let outcome = state.outcome().unwrap();
        assert_eq!(terminal_target(outcome, winner), [1., 0., 0.]);
        assert_eq!(
            terminal_target(outcome, state.position.side_to_move()),
            [0., 0., 1.]
        );
    }
    #[test]
    fn paired_evidence_rejects_equal_results() {
        assert!(paired_lower(&[1.; 16], &[1.; 16]) <= 0.);
        assert!(paired_lower(&[1.; 16], &[0.; 16]) > 0.1);
        assert!(paired_lower(&[0.5; 16], &[0.; 16]) > 0.1);
    }
    #[test]
    fn replay_preserves_rule_history() {
        let mut s = State::start(7);
        for text in ["b0c2", "b9c7", "c2b0", "c7b9"] {
            let mv = s.position.parse_uci_move(text).unwrap();
            s = s.after(mv);
        }
        let mut replay = State::start(7);
        for mv in &s.moves {
            replay = replay.after(*mv);
        }
        assert_eq!(s.position, replay.position);
        assert_eq!(
            rule_context_features(&s.position, &s.history),
            rule_context_features(&replay.position, &replay.history)
        );
        assert_eq!(s.history.len(), 5);
    }
}
