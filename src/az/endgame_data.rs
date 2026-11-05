//! 固定残局监督数据：复用经验池快照，训练与留出严格分开；不依赖查表器。
use super::{AzExperiencePool, AzNnue, AzTrainingSample, SplitMix64};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    fs, io,
    path::Path,
};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    format: u32,
    kind: String,
    train_rows: usize,
    held_rows: usize,
    train_sha256: String,
    held_sha256: String,
    rules: String,
}

pub struct EndgameCorpus {
    train: Vec<AzTrainingSample>,
    held: Vec<AzTrainingSample>,
    held_boards: HashSet<Vec<usize>>,
    units: Vec<[usize; 2]>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct EndgameMetrics {
    pub samples: usize,
    pub wdl: [usize; 3],
    pub value_ce: f64,
    pub value_rmse: f64,
    pub value_accuracy: f64,
    pub policy_samples: usize,
    pub policy_teacher_mass: f64,
    pub policy_top1: f64,
    pub moves_left_samples: usize,
    pub moves_left_rmse: f64,
    pub moves_left_pairs: usize,
    pub moves_left_pair_accuracy: f64,
}

#[derive(Clone, Debug, Default)]
pub struct EndgameTrainingReport {
    pub mixed_samples: usize,
    pub holdout: Option<EndgameMetrics>,
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
fn sha(path: &Path) -> io::Result<String> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

fn key(row: &AzTrainingSample) -> io::Result<Vec<usize>> {
    let p = super::position_for_training_sample(row).ok_or_else(|| invalid("残局棋盘编码无效"))?;
    if p.occupied_squares().count() > 13
        || row.mating_goal.enabled()
        || row.goal_weight != 0.
        || row.mating_weight != 0.
        || row.search_simulations != 0
        || row.value_weight != 1.
        || ![0., 1.].contains(&row.policy_weight)
        || row.move_indices.is_empty()
        || row.move_indices.len() != row.policy.len()
        || row.repetition_flags.len() != row.policy.len()
        || row
            .move_indices
            .iter()
            .any(|&i| super::dense_move_squares(i).is_none())
        || row.policy.iter().any(|p| !p.is_finite() || *p < 0.)
        || (row.policy.iter().sum::<f32>() - 1.).abs() > 1e-5
        || row
            .rule_context
            .iter()
            .chain(&row.history_features)
            .any(|v| !v.is_finite())
        || ![[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]].contains(&row.value_wdl)
        || row.value != row.value_wdl[0] - row.value_wdl[2]
        || ![0., 1.].contains(&row.moves_left_weight)
        || !row.moves_left.is_finite()
        || row.moves_left < 0.
        || (row.moves_left_weight > 0. && (row.value == 0. || row.meta.game_id == 0))
    {
        return Err(invalid("不是有效的普通残局教师样本"));
    }
    let features = |p: &crate::xiangqi::Position| {
        let mut f = super::nnue::extract_sparse_features_az(p);
        f.sort_unstable();
        f
    };
    Ok(features(&p).min(features(&p.mirror_files())))
}

fn validate(train: &[AzTrainingSample], held: &[AzTrainingSample]) -> io::Result<()> {
    if train.is_empty() || held.is_empty() || train.len() > 1_000_000 || held.len() > 1_000_000 {
        return Err(invalid("残局数据集为空或超过100万行上限"));
    }
    let train_keys = train.iter().map(key).collect::<io::Result<HashSet<_>>>()?;
    for row in held {
        if train_keys.contains(&key(row)?) {
            return Err(invalid("残局训练与留出集有等价棋盘重叠"));
        }
    }
    Ok(())
}

impl EndgameCorpus {
    fn from_rows(train: Vec<AzTrainingSample>, held: Vec<AzTrainingSample>) -> io::Result<Self> {
        validate(&train, &held)?;
        let mut held_boards = HashSet::new();
        for row in &held {
            let p = super::position_for_training_sample(row).unwrap();
            held_boards.insert(super::nnue::extract_sparse_features_az(&p));
            held_boards.insert(super::nnue::extract_sparse_features_az(&p.mirror_files()));
        }
        let units =
            super::mlh::sampling_units(&train.iter().collect::<Vec<_>>(), 0xD1B5_4A32_D192_ED03);
        Ok(Self {
            train,
            held,
            held_boards,
            units,
        })
    }
    /// 仅创建新目录；快照格式沿用当前引擎格式，不兼容旧版本时明确报错。
    pub fn save(
        root: &Path,
        train: &[AzTrainingSample],
        held: &[AzTrainingSample],
        rules: &str,
    ) -> io::Result<()> {
        validate(train, held)?;
        fs::create_dir(root)?;
        for (name, rows) in [("train.replay.lz4", train), ("held.replay.lz4", held)] {
            let mut pool = AzExperiencePool::new(rows.len());
            pool.add_samples(rows.iter().cloned());
            pool.save_snapshot_lz4(&root.join(name))?;
        }
        let manifest = Manifest {
            format: 1,
            kind: "chineseai-endgame".into(),
            train_rows: train.len(),
            held_rows: held.len(),
            train_sha256: sha(&root.join("train.replay.lz4"))?,
            held_sha256: sha(&root.join("held.replay.lz4"))?,
            rules: rules.into(),
        };
        fs::write(
            root.join("manifest.toml"),
            toml::to_string_pretty(&manifest).map_err(io::Error::other)?,
        )
    }
    pub fn load(root: &Path) -> io::Result<Self> {
        let m: Manifest = toml::from_str(&fs::read_to_string(root.join("manifest.toml"))?)
            .map_err(io::Error::other)?;
        if m.format != 1
            || m.kind != "chineseai-endgame"
            || m.train_rows == 0
            || m.held_rows == 0
            || m.train_rows > 1_000_000
            || m.held_rows > 1_000_000
        {
            return Err(invalid("残局监督清单版本或行数无效"));
        }
        let load = |name: &str, count, checksum: &str| -> io::Result<Vec<AzTrainingSample>> {
            let path = root.join(name);
            if sha(&path)? != checksum {
                return Err(invalid("残局快照摘要不匹配"));
            }
            let pool = AzExperiencePool::load_snapshot_lz4(&path, 1_000_000)?;
            if pool.sample_count() != count {
                return Err(invalid("残局快照行数不匹配"));
            }
            Ok(pool.all_samples())
        };
        let train = load("train.replay.lz4", m.train_rows, &m.train_sha256)?;
        let held = load("held.replay.lz4", m.held_rows, &m.held_sha256)?;
        Self::from_rows(train, held)
    }
    pub fn train_len(&self) -> usize {
        self.train.len()
    }
    pub fn held_len(&self) -> usize {
        self.held.len()
    }
    pub fn into_parts(self) -> (Vec<AzTrainingSample>, Vec<AzTrainingSample>) {
        (self.train, self.held)
    }
    /// 每次更新替换floor(total/10)行，优化器随后照常洗牌；不增加步数、不抽取留出行。
    pub fn mix(&self, rows: &mut [AzTrainingSample], rng: &mut SplitMix64) -> io::Result<usize> {
        let forbidden: Vec<_> = rows
            .iter()
            .map(|r| {
                if r.features.len() > 13 {
                    return false;
                }
                let mut features = r.features.clone();
                features.sort_unstable();
                self.held_boards.contains(&features)
            })
            .collect();
        if forbidden.iter().any(|b| *b) {
            let allowed: Vec<_> = (0..rows.len()).filter(|&i| !forbidden[i]).collect();
            if allowed.is_empty() {
                return Err(invalid("在线批次全部命中残局留出集，跳过本次训练"));
            }
            for (i, blocked) in forbidden.iter().enumerate() {
                if *blocked {
                    rows[i] = rows[allowed[rng.next_u64() as usize % allowed.len()]].clone();
                }
            }
        }
        let mut slots: Vec<_> = (0..rows.len()).collect();
        let count = rows.len() / 10;
        for i in 0..count {
            let j = i + rng.next_u64() as usize % (rows.len() - i);
            slots.swap(i, j);
        }
        let mut written = 0;
        while written < count {
            let mut unit = self.units[rng.next_u64() as usize % self.units.len()];
            if rng.next_u64() & 1 != 0 {
                unit.swap(0, 1);
            }
            for index in unit {
                if index == super::mlh::NO_PARTNER || written == count {
                    continue;
                }
                let mut row = self.train[index].clone();
                // 复用既有MLH采样单元，保留同局配对；隔离在线自博弈同局编号。
                row.meta.generation_update = u32::MAX;
                rows[slots[written]] = row;
                written += 1;
            }
        }
        Ok(count)
    }
    pub fn evaluate(&self, model: &AzNnue) -> EndgameMetrics {
        evaluate(model, &self.held)
    }
}

pub fn evaluate(model: &AzNnue, rows: &[AzTrainingSample]) -> EndgameMetrics {
    let mut m = EndgameMetrics::default();
    let mut temporal = BTreeMap::new();
    for row in rows {
        let Some((out, scratch)) = super::inference::evaluate_training_sample(model, row) else {
            continue;
        };
        m.samples += 1;
        let target = row
            .value_wdl
            .iter()
            .position(|&p| p == 1.)
            .expect("validated hard WDL");
        m.wdl[target] += 1;
        let predicted = (0..3)
            .max_by(|&a, &b| out.value_wdl[a].total_cmp(&out.value_wdl[b]))
            .unwrap();
        m.value_accuracy += f64::from(target == predicted);
        m.value_ce -= f64::from(out.value_wdl[target].max(1e-12)).ln();
        m.value_rmse += f64::from((out.value_wdl[0] - out.value_wdl[2] - row.value).powi(2));
        if row.moves_left_weight > 0. {
            m.moves_left_samples += 1;
            m.moves_left_rmse += f64::from((out.moves_left - row.moves_left).powi(2));
            temporal.insert(
                (row.meta.generation_update, row.meta.game_id, row.meta.ply),
                (row.moves_left, out.moves_left),
            );
        }
        if row.policy_weight == 0. {
            continue;
        }
        m.policy_samples += 1;
        let logits = &scratch.logits;
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let total: f32 = logits.iter().map(|&v| (v - max).exp()).sum();
        m.policy_teacher_mass += f64::from(
            logits
                .iter()
                .zip(&row.policy)
                .filter(|(_, t)| **t > 0.)
                .map(|(&v, _)| (v - max).exp())
                .sum::<f32>()
                / total,
        );
        let top = (0..logits.len())
            .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
            .unwrap();
        m.policy_top1 += f64::from(row.policy[top] > 0.);
    }
    for (&(generation, game, ply), &(target, predicted)) in &temporal {
        for gap in [2u16, 8, 16] {
            let Some(next) = ply.checked_add(gap) else {
                continue;
            };
            if let Some(&(next_target, next_predicted)) = temporal.get(&(generation, game, next))
                && (target - next_target - f32::from(gap)).abs() < 1e-4
            {
                m.moves_left_pairs += 1;
                m.moves_left_pair_accuracy += f64::from(predicted > next_predicted);
            }
        }
    }
    let n = m.samples.max(1) as f64;
    m.value_ce /= n;
    m.value_accuracy /= n;
    m.value_rmse = (m.value_rmse / n).sqrt();
    m.policy_teacher_mass /= m.policy_samples.max(1) as f64;
    m.policy_top1 /= m.policy_samples.max(1) as f64;
    m.moves_left_rmse = (m.moves_left_rmse / m.moves_left_samples.max(1) as f64).sqrt();
    m.moves_left_pair_accuracy /= m.moves_left_pairs.max(1) as f64;
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        az::{AzMatingGoal, AzSampleMeta},
        xiangqi::Position,
    };
    fn fixture(fen: &str) -> AzTrainingSample {
        let p = Position::from_fen(fen).unwrap();
        let moves = p.legal_moves();
        let mut policy = vec![0.; moves.len()];
        policy[0] = 1.;
        AzTrainingSample {
            features: super::super::nnue::extract_sparse_features_az(&p),
            rule_context: super::super::rule_context_features(&p, &p.initial_rule_history()),
            history_features: super::super::history_features(&p, &p.initial_rule_history()),
            move_indices: moves
                .iter()
                .map(|&mv| super::super::dense_move_index(mv))
                .collect(),
            repetition_flags: vec![0; moves.len()],
            policy,
            value_wdl: [1., 0., 0.],
            root_search_wdl: [1., 0., 0.],
            value: 1.,
            side_sign: 1.,
            policy_weight: 1.,
            value_weight: 1.,
            moves_left: 10.,
            moves_left_weight: 1.,
            mating_targets: [0.; super::super::MATING_HEAD_SIZE],
            mating_weight: 0.,
            mating_goal: AzMatingGoal::NONE,
            goal_target: 0.,
            goal_weight: 0.,
            search_simulations: 0,
            meta: AzSampleMeta {
                game_id: 11,
                ..Default::default()
            },
        }
    }
    fn rows() -> (AzTrainingSample, AzTrainingSample) {
        (
            fixture("3k5/9/9/9/9/9/9/2R6/9/5K3 w - - 0 1"),
            fixture("3k5/9/9/9/9/9/9/3R5/9/5K3 w - - 0 1"),
        )
    }
    #[test]
    fn refuses_mirrored_holdout_overlap_and_non_teacher_targets() {
        let (train, held) = rows();
        assert!(validate(std::slice::from_ref(&train), std::slice::from_ref(&held)).is_ok());
        let mirror = fixture("5k3/9/9/9/9/9/9/6R2/9/3K5 w - - 0 1");
        assert!(validate(std::slice::from_ref(&train), &[mirror]).is_err());
        let mut bad = train;
        bad.search_simulations = 1;
        assert!(validate(&[bad], &[held]).is_err());
    }
    #[test]
    fn mixing_preserves_size_and_does_not_use_holdout_or_change_labels() {
        let (train, held) = rows();
        let corpus = EndgameCorpus::from_rows(vec![train.clone()], vec![held.clone()]).unwrap();
        let mut batch = vec![fixture(&Position::startpos().to_fen()); 103];
        batch[1] = held.clone();
        batch[2] = fixture("5k3/9/9/9/9/9/9/5R3/9/3K5 w - - 0 1");
        assert_eq!(corpus.mix(&mut batch, &mut SplitMix64::new(7)).unwrap(), 10);
        assert_eq!(batch.len(), 103);
        assert!(
            batch
                .iter()
                .all(|r| r.features.len() > 13 || key(r).unwrap() != key(&held).unwrap())
        );
        let mixed: Vec<_> = batch
            .iter()
            .filter(|r| r.meta.generation_update == u32::MAX)
            .collect();
        assert_eq!(mixed.len(), 10);
        for r in mixed {
            assert_eq!(r.features, train.features);
            assert_eq!(r.policy, train.policy);
            assert_eq!(r.moves_left, train.moves_left);
            assert_eq!(r.meta.game_id, train.meta.game_id);
        }
        assert_eq!(corpus.mix(&mut [], &mut SplitMix64::new(3)).unwrap(), 0);
        assert!(corpus.mix(&mut [held], &mut SplitMix64::new(3)).is_err());
    }
    #[test]
    fn corpus_roundtrip_checks_checksum_and_preserves_teacher_history() {
        let (train, held) = rows();
        let root = std::env::temp_dir().join(format!(
            "chineseai-endgame-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        EndgameCorpus::save(&root, std::slice::from_ref(&train), &[held], "test").unwrap();
        let loaded = EndgameCorpus::load(&root).unwrap();
        assert_eq!(loaded.train[0].history_features, train.history_features);
        assert_eq!(loaded.train[0].moves_left_weight, 1.);
        assert_eq!(loaded.train[0].meta.game_id, 11);
        let path = root.join("train.replay.lz4");
        let mut bytes = fs::read(&path).unwrap();
        bytes[0] ^= 1;
        fs::write(&path, bytes).unwrap();
        assert!(
            EndgameCorpus::load(&root)
                .err()
                .unwrap()
                .to_string()
                .contains("摘要")
        );
        for name in ["train.replay.lz4", "held.replay.lz4", "manifest.toml"] {
            fs::remove_file(root.join(name)).unwrap();
        }
        fs::remove_dir(root).unwrap();
    }
    #[test]
    fn draw_value_only_rows_do_not_pollute_policy_or_distance_metrics() {
        let (mut a, mut b) = rows();
        b.meta.ply = 2;
        b.moves_left = 8.;
        let model = AzNnue::random(8, 3);
        let metrics = evaluate(&model, &[a.clone(), b]);
        assert_eq!(metrics.moves_left_pairs, 1);
        a.value_wdl = [0., 1., 0.];
        a.value = 0.;
        a.policy_weight = 0.;
        a.moves_left_weight = 0.;
        let metrics = evaluate(&model, &[a]);
        assert_eq!(metrics.wdl, [0, 1, 0]);
        assert_eq!(metrics.policy_samples, 0);
        assert_eq!(metrics.moves_left_samples, 0);
        assert!(metrics.value_ce.is_finite());
    }
    #[test]
    fn persistent_sampling_keeps_known_temporal_partners_together() {
        let (a, mut b) = rows();
        b.meta.ply = 2;
        b.moves_left = 8.;
        let held = fixture("3k5/9/9/9/9/9/9/4R4/9/5K3 w - - 0 1");
        let corpus = EndgameCorpus::from_rows(vec![a, b], vec![held]).unwrap();
        let mut batch = vec![fixture(&Position::startpos().to_fen()); 100];
        corpus.mix(&mut batch, &mut SplitMix64::new(27)).unwrap();
        let rows: Vec<_> = batch
            .iter()
            .filter(|r| r.meta.generation_update == u32::MAX)
            .collect();
        assert_eq!(rows.iter().filter(|r| r.meta.ply == 0).count(), 5);
        assert_eq!(rows.iter().filter(|r| r.meta.ply == 2).count(), 5);
    }
    #[test]
    fn enabling_ordinary_training_preserves_existing_predictions() {
        let (sample, _) = rows();
        let mut model = AzNnue::random(8, 49);
        model.goal_adaptation_only = true;
        let before = super::super::outputs_for_training_sample(&model, &sample).unwrap();
        model.enable_ordinary_training();
        assert!(!model.is_goal_adaptation_only());
        assert_eq!(
            before,
            super::super::outputs_for_training_sample(&model, &sample).unwrap()
        );
    }
}
