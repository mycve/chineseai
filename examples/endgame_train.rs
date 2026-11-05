//! 离线普通残局监督与留出评估，不接入搜索热路径、不自动替换best。
use chineseai::{
    az::{
        self, AzNnue, AzTrainingSample, SplitMix64,
        endgame_data::{EndgameCorpus, EndgameMetrics as Metrics, evaluate},
        tablebase::{EndgameTablebases, certified_sample, certified_trajectory, draw_rollout},
    },
    xiangqi::Position,
};
use clap::Parser;
use serde::Serialize;
use std::{
    collections::HashSet,
    fs,
    io::{self, Write},
    path::PathBuf,
};
#[derive(Parser, Clone, Serialize)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "data/tablebases/chessdb")]
    tables: PathBuf,
    /// 每行一个新残局FEN；按完整棋盘及镜像去重划分，历史不从FEN伪造。
    #[arg(long, conflicts_with = "corpus", required_unless_present = "corpus")]
    seeds: Option<PathBuf>,
    /// 复用已导出的固定数据集，跳过查表生成；与--seeds二选一。
    #[arg(long)]
    corpus: Option<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    /// 普通Px0对局锚点，防止只拟合残局后遗忘普通局面。
    #[arg(long)]
    anchors: Option<PathBuf>,
    #[arg(long, default_value_t = 8)]
    per_seed: usize,
    #[arg(long, default_value_t = 4)]
    epochs: usize,
    #[arg(long, default_value_t = 0.00002)]
    lr: f32,
    #[arg(long, default_value_t = 512)]
    batch_size: usize,
    #[arg(long, default_value_t = 4096)]
    /// 胜负证明每步的节点预算；三子保和图另有固定20000局面上限。
    proof_nodes: usize,
    #[arg(long)]
    evaluate_only: bool,
    /// 固定对手权重；只读评估候选时使用，与基线复用同一个对手。
    #[arg(long)]
    opponent: Option<PathBuf>,
}
#[derive(Default, Serialize)]
struct Conversion {
    games: usize,
    wins: usize,
    losses: usize,
    draws: usize,
    cutoffs: usize,
    winning_plies: usize,
    search_failed: usize,
    played_plies: usize,
    /// 每局FEN及实际长度，方便复现，避免相邻同类残局掩盖弱点。
    starts: Vec<String>,
    lengths: Vec<usize>,
    outcomes: Vec<String>,
    table_replies: usize,
    fallback_replies: usize,
}
fn table_reply(
    tables: &mut EndgameTablebases,
    p: &Position,
    history: &[chineseai::xiangqi::RuleHistoryEntry],
) -> io::Result<Option<chineseai::xiangqi::Move>> {
    use chineseai::xiangqi::RuleOutcome;
    let mut best = None;
    let mut missing = false;
    for mv in p.legal_moves_with_rules(history) {
        let mut h = history.to_vec();
        h.push(p.rule_history_entry_after_move(mv));
        let mut child = p.clone();
        child.make_move(mv);
        let terminal = child.rule_outcome_with_history(&h).or_else(|| {
            child
                .legal_moves_with_rules(&h)
                .is_empty()
                .then_some(RuleOutcome::Win(p.side_to_move()))
        });
        let score = if let Some(outcome) = terminal {
            match outcome {
                RuleOutcome::Win(c) if c == p.side_to_move() => return Ok(Some(mv)),
                RuleOutcome::Win(_) => (-2, 0),
                RuleOutcome::Draw(_) => (0, 0),
            }
        } else if let Some(v) = tables.probe(&child)? {
            let wdl = -v.wdl;
            (
                wdl,
                if wdl > 0 {
                    -i32::from(v.dtm_rank)
                } else {
                    i32::from(v.dtm_rank)
                },
            )
        } else {
            missing = true;
            continue;
        };
        if best.is_none_or(|(s, _)| score > s) {
            best = Some((score, mv));
        }
    }
    // 查询仅用于选择一个实际合法的对手动作，不作为胜负标签。缺表回退冻结网络。
    Ok(if missing { None } else { best.map(|(_, m)| m) })
}
fn conversion(
    tables: &mut EndgameTablebases,
    model: &AzNnue,
    opponent: &AzNnue,
    rows: &[AzTrainingSample],
) -> io::Result<Conversion> {
    use chineseai::{
        az::{AzSearchLimits, alphazero_search_with_rules},
        xiangqi::{Color, RuleOutcome},
    };
    let mut result = Conversion::default();
    let mut materials = HashSet::new();
    // 每种子力至多一局，固定前16类表距离至少8的留出胜局，不按网络表现筛选。
    for row in rows.iter().filter(|r| r.value > 0.) {
        let mut p = az::position_for_training_sample(row).unwrap();
        if tables.probe(&p)?.is_none_or(|v| v.dtm_rank < 8) {
            continue;
        }
        // 轨迹留出行原本带历史；兑现从新残局起步，必须在空历史下重新证明。
        if certified_sample(tables, &p, 4096)?.is_none_or(|r| r.value != 1.) {
            continue;
        }
        let mut material: Vec<_> = p
            .to_fen()
            .split_whitespace()
            .next()
            .unwrap()
            .chars()
            .filter(|c| c.is_ascii_alphabetic())
            .collect();
        material.sort_unstable();
        if !materials.insert(material) {
            continue;
        }
        if result.games == 16 {
            break;
        }
        result.starts.push(p.to_fen());
        let played_before = result.played_plies;
        let mut history = p.initial_rule_history();
        result.games += 1;
        let mut finished = false;
        let mut ending = "cutoff";
        for ply in 0..=160 {
            let outcome = if !p.has_general(Color::Red) {
                Some(RuleOutcome::Win(Color::Black))
            } else if !p.has_general(Color::Black) {
                Some(RuleOutcome::Win(Color::Red))
            } else {
                p.rule_outcome_with_history(&history).or_else(|| {
                    p.legal_moves_with_rules(&history)
                        .is_empty()
                        .then_some(RuleOutcome::Win(p.side_to_move().opposite()))
                })
            };
            if let Some(outcome) = outcome {
                ending = match outcome {
                    RuleOutcome::Win(Color::Red) => "win",
                    RuleOutcome::Win(Color::Black) => "loss",
                    _ => "draw",
                };
                match outcome {
                    RuleOutcome::Win(Color::Red) => {
                        result.wins += 1;
                        result.winning_plies += ply;
                    }
                    RuleOutcome::Win(Color::Black) => result.losses += 1,
                    _ => result.draws += 1,
                }
                finished = true;
                break;
            }
            if ply == 160 {
                break;
            }
            if p.side_to_move() == Color::Black {
                if let Some(mv) = table_reply(tables, &p, &history)? {
                    result.table_replies += 1;
                    history.push(p.rule_history_entry_after_move(mv));
                    p.make_move(mv);
                    result.played_plies += 1;
                    continue;
                }
                result.fallback_replies += 1;
            }
            let network = if p.side_to_move() == Color::Red {
                model
            } else {
                opponent
            };
            let search = alphazero_search_with_rules(
                &p,
                Some(history.clone()),
                None,
                network,
                AzSearchLimits {
                    simulations: 512,
                    minimum_kldgain_per_node: 0.,
                    ..AzSearchLimits::default()
                },
            );
            let Some(mv) = search.best_move else {
                ending = "search_failed";
                result.search_failed += 1;
                eprintln!("残局验收搜索失败：{} q={}", p.to_fen(), search.value_q);
                finished = true;
                break;
            };
            history.push(p.rule_history_entry_after_move(mv));
            p.make_move(mv);
            result.played_plies += 1;
        }
        if !finished {
            result.cutoffs += 1;
        }
        result.lengths.push(result.played_plies - played_before);
        result.outcomes.push(ending.to_owned());
    }
    Ok(result)
}
fn orbit(p: &Position) -> String {
    let key = |p: &Position| {
        let mut f = az::nnue::extract_sparse_features_az(p);
        f.sort_unstable();
        format!("{f:?}")
    };
    key(p).min(key(&p.mirror_files()))
}
#[derive(Serialize)]
struct OrdinaryArena {
    wins: usize,
    losses: usize,
    draws: usize,
    score: f32,
    ci95: [f32; 2],
    starts: Vec<String>,
}
fn ordinary_arena(
    model: &AzNnue,
    baseline: &AzNnue,
    rows: &[AzTrainingSample],
) -> Option<OrdinaryArena> {
    let mut seen = HashSet::new();
    let all: Vec<_> = rows
        .iter()
        .filter_map(az::position_for_training_sample)
        .filter(|p| {
            (0..90).filter(|&s| p.piece_at(s).is_some()).count() >= 24 && seen.insert(orbit(p))
        })
        .collect();
    if all.is_empty() {
        return None;
    }
    let n = all.len().min(8);
    let positions: Vec<_> = (0..n).map(|i| all[i * all.len() / n].clone()).collect();
    let s = az::AzSearchLimits::default();
    let r = az::play_arena_games_from_positions(
        model,
        baseline,
        &positions,
        az::AzArenaConfig {
            simulations: 512,
            max_plies: 256,
            rule60_max_ply: Some(120),
            games_as_red: positions.len(),
            games_as_black: positions.len(),
            start_index: 0,
            seed: 20261105,
            cpuct: s.cpuct,
            cpuct_at_root: s.cpuct_at_root,
            cpuct_base: s.cpuct_base,
            cpuct_factor: s.cpuct_factor,
            cpuct_base_at_root: s.cpuct_base_at_root,
            cpuct_factor_at_root: s.cpuct_factor_at_root,
            fpu_value: s.fpu_value,
            fpu_value_at_root: s.fpu_value_at_root,
            fpu_absolute_at_root: s.fpu_absolute_at_root,
            minimum_kldgain_per_node: 0.,
            draw_score: s.draw_score,
            policy_softmax_temp: s.policy_softmax_temp,
        },
    );
    Some(OrdinaryArena {
        wins: r.wins,
        losses: r.losses,
        draws: r.draws,
        score: r.score_rate(),
        ci95: [
            r.score_rate_lower_bound(1.96),
            r.score_rate_upper_bound(1.96),
        ],
        starts: positions.iter().map(Position::to_fen).collect(),
    })
}
fn hash(s: &str) -> u64 {
    s.bytes().fold(0xcbf29ce484222325, |h, c| {
        (h ^ u64::from(c)).wrapping_mul(0x100000001b3)
    })
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Args::parse();
    if a.output.exists()
        || a.output.with_extension("report.toml").exists()
        || a.output.with_extension("endgame").exists()
    {
        return Err("输出路径已存在，拒绝覆盖".into());
    }
    if a.per_seed == 0 || a.epochs == 0 || a.batch_size == 0 || !a.lr.is_finite() || a.lr <= 0. {
        return Err("训练参数无效".into());
    }
    let attempts = a.per_seed.checked_mul(8).ok_or("采样数量溢出")?;
    let mut model = AzNnue::load(&a.model)?;
    let mut tables = EndgameTablebases::load(&a.tables)?;
    let mut train = Vec::new();
    let mut held = Vec::new();
    let mut seen = HashSet::new();
    let mut counts = [0usize; 3];
    let mut raw = [0usize; 3];
    let mut unknown = 0;
    let mut table_misses = 0;
    let mut accepted_per_seed = Vec::new();
    let mut draw_rollout_games = 0;
    let mut rng = SplitMix64::new(20261105);
    if let Some(path) = &a.corpus {
        (train, held) = EndgameCorpus::load(path)?.into_parts();
    } else {
        let seeds: Vec<_> = fs::read_to_string(a.seeds.as_ref().unwrap())?
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
            .map(Position::from_fen)
            .collect::<Result<_, _>>()?;
        if seeds.is_empty() {
            return Err("起点文件为空".into());
        }
        for (si, seed) in seeds.iter().enumerate() {
            let mut p = seed.clone();
            let mut accepted = 0;
            for step in 0..attempts {
                if step % 48 == 0
                    || p.rule_outcome_with_history(&p.initial_rule_history())
                        .is_some()
                {
                    p = seed.clone();
                }
                let key = orbit(&p);
                if !seen.contains(&key) {
                    seen.insert(key.clone());
                    if let Some(v) = tables.probe(&p)? {
                        raw[if v.wdl > 0 {
                            0
                        } else if v.wdl == 0 {
                            1
                        } else {
                            2
                        }] += 1;
                        let rows = if v.wdl == 0 {
                            // 随机游走仅生成棋盘；作为独立新残局重置自然限着计数。
                            let fen = p.to_fen();
                            let mut fields: Vec<_> = fen.split_whitespace().collect();
                            fields[4] = "0";
                            let fresh = Position::from_fen(&fields.join(" ")).unwrap();
                            if let Some(row) = certified_sample(&mut tables, &fresh, a.proof_nodes)?
                            {
                                Some(vec![row])
                            } else {
                                let result = draw_rollout(&mut tables, &fresh)?;
                                draw_rollout_games += usize::from(result.is_some());
                                result
                            }
                        } else {
                            certified_trajectory(&mut tables, &p, a.proof_nodes)?
                        };
                        if let Some(rows) = rows {
                            let test = hash(&key) % 5 == 0;
                            for row in rows {
                                counts[if row.value > 0. {
                                    0
                                } else if row.value == 0. {
                                    1
                                } else {
                                    2
                                }] += 1;
                                if test {
                                    held.push(row);
                                } else {
                                    train.push(row);
                                }
                            }
                            accepted += 1;
                            if accepted >= a.per_seed {
                                break;
                            }
                        } else {
                            unknown += 1;
                        }
                    } else {
                        table_misses += 1;
                    }
                }
                let moves = p.legal_moves();
                if moves.is_empty() {
                    p = seed.clone();
                } else {
                    p.make_move(moves[rng.next_u64() as usize % moves.len()]);
                }
            }
            println!(
                "起点 {}/{} 已接收={} 累计训练={} 留出={}",
                si + 1,
                seeds.len(),
                accepted,
                train.len(),
                held.len()
            );
            accepted_per_seed.push(accepted);
        }
        let held_keys: HashSet<_> = held
            .iter()
            .map(|r| orbit(&az::position_for_training_sample(r).unwrap()))
            .collect();
        train
            .retain(|r| !held_keys.contains(&orbit(&az::position_for_training_sample(r).unwrap())));
        let mut unique_train = HashSet::new();
        train.retain(|r| unique_train.insert(orbit(&az::position_for_training_sample(r).unwrap())));
        let mut unique_test = HashSet::new();
        held.retain(|r| unique_test.insert(orbit(&az::position_for_training_sample(r).unwrap())));
        // 和棋收尾通常很长，按固定棋盘键抽取，训练行中和棋至多约1/3。
        let draw_limit = train.iter().filter(|r| r.value != 0.).count() / 2;
        train.sort_by_key(|r| hash(&orbit(&az::position_for_training_sample(r).unwrap())));
        let mut draws = 0;
        train.retain(|r| {
            if r.value == 0. {
                draws += 1;
                draws <= draw_limit
            } else {
                true
            }
        });
    }
    if train.is_empty() || held.is_empty() {
        return Err("可证明训练/留出不足，未训练、未保存模型".into());
    }
    let corpus_path = a.output.with_extension("endgame");
    EndgameCorpus::save(
        &corpus_path,
        &train,
        &held,
        "胜负轨迹及三子保和经证明；其他和棋为项目完整棋规下实际完成的教师回报，只监督WDL。",
    )?;
    println!(
        "固定残局监督={} train={} held={}",
        corpus_path.display(),
        train.len(),
        held.len()
    );
    let before = evaluate(&model, &held);
    let baseline = a
        .opponent
        .as_ref()
        .map(AzNnue::load)
        .transpose()?
        .unwrap_or_else(|| model.clone());
    let conversion_before = conversion(&mut tables, &model, &baseline, &held)?;
    let mut anchors = Vec::new();
    let mut ordinary_test = Vec::new();
    if let Some(path) = &a.anchors {
        let dataset = az::px0_data::load(path, 64, 64)?;
        anchors = dataset.train;
        for row in &mut anchors {
            if let Some((wdl, logits)) = az::outputs_for_training_sample(&model, row) {
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut policy: Vec<_> = logits.iter().map(|&x| (x - max).exp()).collect();
                let sum = policy.iter().sum::<f32>();
                for p in &mut policy {
                    *p /= sum;
                }
                row.policy = policy;
                row.value_wdl = wdl;
                row.value = wdl[0] - wdl[2];
                row.moves_left_weight = 0.;
            }
        }
        ordinary_test = dataset.validation;
    }
    if !a.evaluate_only && anchors.is_empty() {
        return Err("训练必须提供普通对局锚点 --anchors；可先 --evaluate-only".into());
    }
    let ordinary_before =
        (!ordinary_test.is_empty()).then(|| az::px0_data::evaluate(&model, &ordinary_test));
    let supervised = train.len();
    let label_counts = |rows: &[AzTrainingSample]| {
        rows.iter().fold([0usize; 3], |mut a, r| {
            a[if r.value > 0. {
                0
            } else if r.value == 0. {
                1
            } else {
                2
            }] += 1;
            a
        })
    };
    let train_wdl = label_counts(&train);
    let held_wdl = label_counts(&held);
    // 等量普通锚点，蒸馏输入模型自己的策略与WDL，减少普通局面遗忘。
    for i in 0..supervised.min(anchors.len()) {
        train.push(anchors[i * anchors.len() / supervised.min(anchors.len())].clone());
    }
    let mut loss = None;
    let mut moves_left_pairs = 0;
    if !a.evaluate_only {
        let stats = az::tablebase::train_endgame(
            &mut model,
            &train,
            a.epochs,
            a.lr,
            a.batch_size,
            &mut rng,
        )
        .map_err(io::Error::other)?;
        loss = Some(stats.loss);
        moves_left_pairs = stats.moves_left_pairs;
    }
    let after = evaluate(&model, &held);
    let conversion_after = if a.evaluate_only {
        None
    } else {
        Some(conversion(&mut tables, &model, &baseline, &held)?)
    };
    let ordinary_after =
        (!ordinary_test.is_empty()).then(|| az::px0_data::evaluate(&model, &ordinary_test));
    let common_wins = conversion_after.as_ref().map(|after| {
        assert_eq!(
            conversion_before.starts, after.starts,
            "兑现前后起点必须一致"
        );
        let mut common = CommonWins {
            games: 0,
            before_plies: 0,
            after_plies: 0,
        };
        for i in 0..after.games {
            if conversion_before.outcomes[i] == "win" && after.outcomes[i] == "win" {
                common.games += 1;
                common.before_plies += conversion_before.lengths[i];
                common.after_plies += after.lengths[i];
            }
        }
        common
    });
    let ordinary_arena = if !a.evaluate_only || a.opponent.is_some() {
        ordinary_arena(&model, &baseline, &ordinary_test)
    } else {
        None
    };
    // TOML报告只记录本轮验证，不自动批准候选。原模型始终保留。
    let report = toml::to_string_pretty(&Report {
        parameters: a.clone(),
        source_model: a.model.display().to_string(),
        corpus_path: corpus_path.display().to_string(),
        rules: "胜负轨迹采用完整历史证明；三子和棋采用安全闭包证明。其余和棋只接收表引导且由项目完整棋规实际判和的RL回报，策略权重为0；不是极小极大和棋证明。缺表、截断均跳过。"
            .into(),
        materials: tables.materials(),
        table_bytes: tables.image_bytes,
        train: supervised,
        train_wdl,
        held_wdl,
        anchors: train.len() - supervised,
        raw_wdl: raw,
        generated_wdl: counts,
        unverified: unknown,
        table_misses,
        accepted_per_seed,
        draw_rollout_games,
        before,
        after,
        conversion_before,
        conversion_after,
        common_wins,
        loss,
        moves_left_pairs,
        ordinary_before: ordinary_before
            .as_ref()
            .map(|x| [x.value_ce as f64, x.q_rmse as f64, x.policy_kl as f64]),
        ordinary_after: ordinary_after
            .as_ref()
            .map(|x| [x.value_ce as f64, x.q_rmse as f64, x.policy_kl as f64]),
        ordinary_arena,
    })?;
    let report_path = a.output.with_extension("report.toml");
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&report_path)?
        .write_all(report.as_bytes())?;
    if !a.evaluate_only {
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&a.output)?;
        if let Err(e) = model.save(&a.output) {
            let _ = fs::remove_file(&a.output);
            return Err(e.into());
        }
    }
    println!("报告={}；候选不自动替换best", report_path.display());
    Ok(())
}
#[derive(Serialize)]
struct Report {
    parameters: Args,
    source_model: String,
    corpus_path: String,
    rules: String,
    materials: Vec<String>,
    table_bytes: usize,
    train: usize,
    train_wdl: [usize; 3],
    held_wdl: [usize; 3],
    anchors: usize,
    raw_wdl: [usize; 3],
    generated_wdl: [usize; 3],
    unverified: usize,
    table_misses: usize,
    accepted_per_seed: Vec<usize>,
    draw_rollout_games: usize,
    before: Metrics,
    after: Metrics,
    conversion_before: Conversion,
    conversion_after: Option<Conversion>,
    common_wins: Option<CommonWins>,
    loss: Option<f32>,
    moves_left_pairs: usize,
    ordinary_before: Option<[f64; 3]>,
    ordinary_after: Option<[f64; 3]>,
    ordinary_arena: Option<OrdinaryArena>,
}
#[derive(Serialize)]
struct CommonWins {
    games: usize,
    before_plies: usize,
    after_plies: usize,
}
