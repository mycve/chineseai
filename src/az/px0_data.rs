//! Px0 V6 / input format 1 数据读取与固定验证集蒸馏评测。
use super::*;
use crate::nnue::extract_sparse_features_az;
use crate::xiangqi::RuleHistoryEntry;
use flate2::read::GzDecoder;
use std::{
    fs::File,
    io::{self, Read},
    path::Path,
};

const RECORD_SIZE: usize = 10256;
const PLANES: usize = 8256;
const VALUES: usize = 10180;

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}
fn float(record: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(record[offset..offset + 4].try_into().unwrap())
}
fn index(record: &[u8], offset: usize) -> usize {
    u16::from_le_bytes(record[offset..offset + 2].try_into().unwrap()) as usize
}

fn position(record: &[u8]) -> io::Result<Position> {
    if record[10176] > 1 {
        return Err(invalid("invalid side to move"));
    }
    let black = record[10176] != 0;
    let mut board = [None; 90];
    for plane in 0..14 {
        let offset = PLANES + plane * 16;
        let mask = u128::from_le_bytes(record[offset..offset + 16].try_into().unwrap());
        if mask >> 90 != 0 {
            return Err(invalid("piece plane exceeds board"));
        }
        let ours = plane < 7;
        let red = ours != black;
        let mut ch = b"racpnbk"[plane % 7];
        if red {
            ch = ch.to_ascii_uppercase();
        }
        for bit in 0..90 {
            if mask & (1u128 << bit) == 0 {
                continue;
            }
            let rank = if black { bit / 9 } else { 9 - bit / 9 };
            let square = rank * 9 + bit % 9;
            if board[square].replace(ch as char).is_some() {
                return Err(invalid("overlapping pieces"));
            }
        }
    }
    let mut fen = String::new();
    for rank in 0..10 {
        if rank > 0 {
            fen.push('/');
        }
        let mut empty = 0;
        for file in 0..9 {
            if let Some(ch) = board[rank * 9 + file] {
                if empty > 0 {
                    fen.push(char::from(b'0' + empty));
                    empty = 0;
                }
                fen.push(ch);
            } else {
                empty += 1;
            }
        }
        if empty > 0 {
            fen.push(char::from(b'0' + empty));
        }
    }
    fen.push_str(&format!(
        " {} - - {} 1",
        if black { "b" } else { "w" },
        record[10177]
    ));
    Position::from_fen(&fen).map_err(invalid)
}

fn px0_move(move_index: usize, side: Color) -> io::Result<Move> {
    let (from, to) =
        dense_move_squares(move_index).ok_or_else(|| invalid("policy index out of range"))?;
    let square = |sq: usize| {
        if side == Color::Black {
            (9 - sq / 9) * 9 + sq % 9
        } else {
            sq
        }
    };
    Ok(Move::new(square(from), square(to)))
}

fn wdl(q: f32, d: f32) -> io::Result<[f32; 3]> {
    if !q.is_finite() || !d.is_finite() || d < -1e-4 || d > 1.0001 || q.abs() + d > 1.0001 {
        return Err(invalid(format!("invalid Q/D: {q}/{d}")));
    }
    Ok(normalize_wdl_target([
        (1.0 - d + q) / 2.0,
        d,
        (1.0 - d - q) / 2.0,
    ]))
}

pub struct Dataset {
    pub train: Vec<AzTrainingSample>,
    pub validation: Vec<AzTrainingSample>,
    pub games: usize,
    pub deleted: usize,
}

/// 按完整 gzip 对局名称散列划分，约 10% 对局固定用于验证。
/// 验证只取前 validation_games 局，后续验证分组跳过，扩大训练时验证集不变。
/// 首条记录之前的规则历史不可完整恢复，从该条开始累计真实走法历史。
pub fn load(path: &Path, max_games: usize, validation_games: usize) -> io::Result<Dataset> {
    if max_games < 10 {
        return Err(invalid("at least 10 games required"));
    }
    let mut archive = tar::Archive::new(File::open(path)?);
    let mut dataset = empty_dataset();
    for entry in archive.entries()? {
        let entry = entry?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let name = entry.path()?.to_string_lossy().into_owned();
        let validation = dataset.games < validation_games;
        decode_game(&mut dataset, &name, entry, validation)?;
        if dataset.games >= max_games {
            break;
        }
    }
    finish_dataset(dataset)
}

fn empty_dataset() -> Dataset {
    Dataset {
        train: vec![],
        validation: vec![],
        games: 0,
        deleted: 0,
    }
}

fn finish_dataset(dataset: Dataset) -> io::Result<Dataset> {
    if dataset.train.is_empty() || dataset.validation.is_empty() {
        return Err(invalid("empty training/validation split"));
    }
    Ok(dataset)
}

fn game_id(name: &str) -> u64 {
    name.bytes().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ byte as u64).wrapping_mul(0x100000001b3)
    })
}

// Algorithm R: 第 seen 个候选进入每个槽的概率均为 1 / seen。
fn reservoir_slot(seen: usize, capacity: usize, rng: &mut SplitMix64) -> Option<usize> {
    if seen <= capacity {
        return Some(seen - 1);
    }
    let bound = seen as u64;
    let threshold = bound.wrapping_neg() % bound;
    let slot = loop {
        let value = rng.next_u64();
        if value >= threshold {
            break (value % bound) as usize;
        }
    };
    (slot < capacity).then_some(slot)
}

/// 扫描整个 TAR，按整局均匀抽取训练对局；固定验证来自归档前 validation_games 局。
/// 扫描时仅保留抽中的 gzip 字节，抽样结束后才解压及转换。
pub fn load_reservoir(
    path: &Path,
    train_games: usize,
    validation_games: usize,
    seed: u64,
) -> io::Result<Dataset> {
    if train_games == 0 || validation_games == 0 {
        return Err(invalid(
            "training and validation game counts must be positive",
        ));
    }
    let mut archive = tar::Archive::new(File::open(path)?);
    let mut rng = SplitMix64::new(seed);
    let mut train: Vec<(String, Vec<u8>)> = vec![];
    let mut validation = vec![];
    let mut scanned = 0;
    let mut eligible = 0;
    for entry in archive.entries()? {
        let mut entry = entry?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let name = entry.path()?.to_string_lossy().into_owned();
        scanned += 1;
        if game_id(&name) % 10 == 0 {
            if scanned <= validation_games {
                let mut compressed = vec![];
                entry.read_to_end(&mut compressed)?;
                validation.push((name, compressed));
            }
        } else {
            eligible += 1;
            if let Some(slot) = reservoir_slot(eligible, train_games, &mut rng) {
                retain_compressed(slot, name, &mut entry, &mut train)?;
            }
        }
        if scanned % 10000 == 0 {
            eprintln!(
                "reservoir scanned={scanned} eligible_train={eligible} selected_train={} selected_validation={}",
                train.len(),
                validation.len()
            );
        }
    }
    eprintln!(
        "reservoir scan_complete scanned={scanned} eligible_train={eligible} selected_train={} selected_validation={}",
        train.len(),
        validation.len()
    );
    if eligible < train_games {
        return Err(invalid(format!(
            "requested {train_games} training games, only {eligible} eligible"
        )));
    }
    let mut dataset = empty_dataset();
    for (name, compressed) in validation.into_iter().chain(train) {
        decode_game(&mut dataset, &name, compressed.as_slice(), true)?;
        if dataset.games % 256 == 0 {
            eprintln!(
                "reservoir decoded_games={} train_samples={} validation_samples={}",
                dataset.games,
                dataset.train.len(),
                dataset.validation.len()
            );
        }
    }
    eprintln!(
        "reservoir decode_complete games={} train_samples={} validation_samples={}",
        dataset.games,
        dataset.train.len(),
        dataset.validation.len()
    );
    finish_dataset(dataset)
}

fn retain_compressed(
    slot: usize,
    name: String,
    mut reader: impl Read,
    selected: &mut Vec<(String, Vec<u8>)>,
) -> io::Result<()> {
    let mut compressed = vec![];
    reader.read_to_end(&mut compressed)?;
    if slot == selected.len() {
        selected.push((name, compressed));
    } else {
        selected[slot] = (name, compressed);
    }
    Ok(())
}

/// 从全归档永不用于训练的 FNV%10==0 分组均匀抽取完整验证对局。
/// 只做读取和转换，返回 train 为空的 Dataset。
pub fn load_holdout_reservoir(path: &Path, games: usize, seed: u64) -> io::Result<Dataset> {
    if games == 0 {
        return Err(invalid("holdout game count must be positive"));
    }
    let selected = select_holdout_compressed(path, games, seed)?;
    let mut dataset = empty_dataset();
    for (name, compressed) in selected {
        decode_game(&mut dataset, &name, compressed.as_slice(), true)?;
        if dataset.games % 256 == 0 {
            eprintln!(
                "holdout_reservoir decoded_games={} validation_samples={}",
                dataset.games,
                dataset.validation.len()
            );
        }
    }
    if dataset.validation.is_empty() {
        return Err(invalid("empty holdout samples"));
    }
    Ok(dataset)
}

fn select_holdout_compressed(
    path: &Path,
    games: usize,
    seed: u64,
) -> io::Result<Vec<(String, Vec<u8>)>> {
    let mut archive = tar::Archive::new(File::open(path)?);
    let mut rng = SplitMix64::new(seed);
    let mut selected = vec![];
    let mut scanned = 0;
    let mut eligible = 0;
    for entry in archive.entries()? {
        let mut entry = entry?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let name = entry.path()?.to_string_lossy().into_owned();
        scanned += 1;
        if game_id(&name) % 10 == 0 {
            eligible += 1;
            if let Some(slot) = reservoir_slot(eligible, games, &mut rng) {
                retain_compressed(slot, name, &mut entry, &mut selected)?;
            }
        }
        if scanned % 10000 == 0 {
            eprintln!(
                "holdout_reservoir scanned={scanned} eligible={eligible} selected={}",
                selected.len()
            );
        }
    }
    eprintln!(
        "holdout_reservoir scan_complete scanned={scanned} eligible={eligible} selected={}",
        selected.len()
    );
    if eligible < games {
        return Err(invalid(format!(
            "requested {games} holdout games, only {eligible} eligible"
        )));
    }
    Ok(selected)
}

pub struct TeacherProbe {
    pub fen: String,
    pub teacher_q: f32,
    pub teacher_d: f32,
    pub result_q: f32,
    pub sample: AzTrainingSample,
}

/// 留出分组均匀抽整局，每局再均匀抽一条未删除记录，用于只读标签诊断。
pub fn load_teacher_probes(path: &Path, games: usize, seed: u64) -> io::Result<Vec<TeacherProbe>> {
    if games == 0 {
        return Err(invalid("probe game count must be positive"));
    }
    let selected = select_holdout_compressed(path, games, seed)?;
    let mut rng = SplitMix64::new(seed ^ 0x54454143484552);
    let mut probes = vec![];
    for (name, compressed) in selected {
        let mut dataset = empty_dataset();
        decode_game(&mut dataset, &name, compressed.as_slice(), true)?;
        if dataset.validation.is_empty() {
            return Err(invalid("selected probe game has no retained records"));
        }
        let mut selected_index = 0;
        for seen in 1..=dataset.validation.len() {
            if reservoir_slot(seen, 1, &mut rng).is_some() {
                selected_index = seen - 1;
            }
        }
        let sample = dataset.validation.swap_remove(selected_index);
        let mut decoded = vec![];
        GzDecoder::new(compressed.as_slice()).read_to_end(&mut decoded)?;
        let offset = sample.meta.ply as usize * RECORD_SIZE;
        let record = &decoded[offset..offset + RECORD_SIZE];
        probes.push(TeacherProbe {
            fen: position(record)?.to_fen(),
            teacher_q: float(record, VALUES + 4),
            teacher_d: float(record, VALUES + 12),
            result_q: float(record, VALUES + 28),
            sample,
        });
    }
    Ok(probes)
}

fn decode_game(
    dataset: &mut Dataset,
    name: &str,
    reader: impl Read,
    validation: bool,
) -> io::Result<()> {
    let mut decoded = vec![];
    GzDecoder::new(reader).read_to_end(&mut decoded)?;
    if decoded.is_empty() || decoded.len() % RECORD_SIZE != 0 {
        return Err(invalid(format!("invalid game size: {name}")));
    }
    let game_id = game_id(name);
    let mut history: Vec<RuleHistoryEntry> = vec![];
    let mut expected_hash = None;
    for (ply, record) in decoded.chunks_exact(RECORD_SIZE).enumerate() {
        if u32::from_le_bytes(record[..4].try_into().unwrap()) != 6
            || u32::from_le_bytes(record[4..8].try_into().unwrap()) != 1
        {
            return Err(invalid("only Px0 V6 input format 1 supported"));
        }
        let position =
            position(record).map_err(|err| invalid(format!("{name} ply={ply}: {err}")))?;
        if expected_hash.is_some_and(|hash| hash != position.hash()) {
            return Err(invalid(format!("trajectory mismatch: {name} ply={ply}")));
        }
        if history.is_empty() {
            history = position.initial_rule_history();
        }
        let side = position.side_to_move();
        let legal = position.legal_moves();
        let played = px0_move(index(record, 10244), side)?;
        let best = px0_move(index(record, 10246), side)?;
        if !legal.contains(&played) || !legal.contains(&best) {
            return Err(invalid(format!("illegal played/best: {name} ply={ply}")));
        }
        let mut moves = vec![];
        let mut policy = vec![];
        for i in 0..DENSE_MOVE_SPACE {
            let probability = float(record, 8 + i * 4);
            if !probability.is_finite() {
                return Err(invalid("non-finite policy"));
            }
            if probability < 0.0 {
                continue;
            }
            let mv = px0_move(i, side)?;
            if !legal.contains(&mv) {
                return Err(invalid(format!(
                    "illegal policy move: {name} ply={ply} move={}",
                    mv.to_uci()
                )));
            }
            moves.push(mv);
            policy.push(probability);
        }
        let sum: f32 = policy.iter().sum();
        if (sum - 1.0).abs() > 0.001 {
            return Err(invalid("policy mass differs from 1"));
        }
        for probability in &mut policy {
            *probability /= sum;
        }
        let visits = u32::from_le_bytes(record[10240..10244].try_into().unwrap());
        let locate = |mv: Move| {
            moves
                .iter()
                .position(|&candidate| candidate == mv)
                .ok_or_else(|| invalid("played/best missing from policy"))
        };
        let played_index = locate(played)?;
        let best_index = locate(best)?;
        let root = wdl(float(record, VALUES), float(record, VALUES + 8))?;
        let target = wdl(float(record, VALUES + 4), float(record, VALUES + 12))?;
        let sample = AzTrainingSample {
            features: extract_sparse_features_az(&position),
            rule_context: rule_context_features(&position, &history),
            move_indices: moves
                .iter()
                .map(|&mv| dense_move_index(canonical_move(side, mv)))
                .collect(),
            repetition_flags: moves
                .iter()
                .map(|&mv| u8::from(position.move_repeats_history(&history, mv)))
                .collect(),
            policy,
            value_wdl: target,
            root_search_wdl: root,
            short_value_wdl: [target; SHORT_VALUE_HEADS],
            value: target[0] - target[2],
            side_sign: if side == Color::Red { 1.0 } else { -1.0 },
            policy_weight: 1.0,
            value_weight: 1.0,
            search_simulations: visits,
            meta: AzSampleMeta {
                game_id,
                ply: ply.min(u16::MAX as usize) as u16,
                root_q: float(record, VALUES),
                best_q: float(record, VALUES + 4),
                played_q: float(record, VALUES + 36),
                best_index: best_index as u16,
                played_index: played_index as u16,
                best_visits: (visits as f32 * float(record, 8 + index(record, 10246) * 4)).round()
                    as u32,
                played_visits: (visits as f32 * float(record, 8 + index(record, 10244) * 4)).round()
                    as u32,
                ..AzSampleMeta::default()
            },
        };
        if record[10178] & 64 != 0 {
            dataset.deleted += 1;
        } else if game_id % 10 == 0 {
            if validation {
                dataset.validation.push(sample);
            }
        } else {
            dataset.train.push(sample);
        }
        let next = position.rule_history_entry_after_move(played);
        expected_hash = Some(next.hash);
        history.push(next);
    }
    dataset.games += 1;
    Ok(())
}

#[derive(Debug)]
pub struct Metrics {
    pub samples: usize,
    pub policy_kl: f64,
    pub top1: f64,
    pub value_ce: f64,
    pub q_rmse: f64,
}

/// 纯 CPU 前向评测，无优化器和参数修改。
pub fn evaluate(model: &AzNnue, samples: &[AzTrainingSample]) -> Metrics {
    let mut metrics = Metrics {
        samples: samples.len(),
        policy_kl: 0.0,
        top1: 0.0,
        value_ce: 0.0,
        q_rmse: 0.0,
    };
    let mut scratch = AzEvalScratch::new(model.arch);
    for sample in samples {
        let position = position_for_training_sample(sample).expect("validated dataset position");
        let moves = sample
            .move_indices
            .iter()
            .map(|&i| {
                let (from, to) = dense_move_squares(i).unwrap();
                Move::new(from, to)
            })
            .collect::<Vec<_>>();
        let output = model.evaluate_with_scratch_output_with_repetition(
            &position,
            &moves,
            &sample.repetition_flags,
            &sample.rule_context,
            &mut scratch,
        );
        let max = scratch
            .logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let log_sum = scratch
            .logits
            .iter()
            .map(|&x| ((x - max) as f64).exp())
            .sum::<f64>()
            .ln()
            + max as f64;
        for (&target, &logit) in sample.policy.iter().zip(&scratch.logits) {
            if target > 0.0 {
                metrics.policy_kl +=
                    target as f64 * ((target as f64).ln() - logit as f64 + log_sum);
            }
        }
        let top = |values: &[f32]| {
            values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        metrics.top1 += f64::from(top(&sample.policy) == top(&scratch.logits));
        metrics.value_ce -= sample
            .value_wdl
            .iter()
            .zip(output.value_wdl)
            .map(|(&t, p)| t as f64 * (p.max(1e-12) as f64).ln())
            .sum::<f64>();
        metrics.q_rmse +=
            ((output.value_wdl[0] - output.value_wdl[2] - sample.value) as f64).powi(2);
    }
    let count = metrics.samples.max(1) as f64;
    metrics.policy_kl /= count;
    metrics.top1 /= count;
    metrics.value_ce /= count;
    metrics.q_rmse = (metrics.q_rmse / count).sqrt();
    metrics
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservoir_is_reproducible_and_samples_the_entire_stream() {
        let select = |seed| {
            let mut rng = SplitMix64::new(seed);
            let mut selected = vec![];
            for seen in 1..=20 {
                if let Some(slot) = reservoir_slot(seen, 4, &mut rng) {
                    if slot == selected.len() {
                        selected.push(seen);
                    } else {
                        selected[slot] = seen;
                    }
                }
            }
            selected
        };
        assert_eq!(select(7), select(7));
        let mut counts = [0; 20];
        for seed in 0..2000 {
            let selected = select(seed);
            assert_eq!(selected.len(), 4);
            for candidate in selected {
                counts[candidate - 1] += 1;
            }
        }
        // 每局预期约 400 次；包括开头和归档尾部，排除仅采样前缀。
        assert!(
            counts.iter().all(|&count| (300..500).contains(&count)),
            "{counts:?}"
        );
    }

    #[test]
    fn black_px0_moves_are_rank_flipped_then_file_flipped_for_our_network() {
        // Px0 a0a1 是黑方 a9a8；我们的黑方规范坐标为 i0i1。
        let mv = px0_move(0, Color::Black).unwrap();
        assert_eq!(mv.to_uci(), "a9a8");
        let canonical = canonical_move(Color::Black, mv);
        assert_eq!(canonical.to_uci(), "i0i1");
        assert_ne!(dense_move_index(canonical), 0);
        assert_eq!(px0_move(0, Color::Red).unwrap().to_uci(), "a0a1");
    }

    #[test]
    fn v6_piece_planes_recover_both_colors_and_asymmetric_positions() {
        let mut source = Position::startpos();
        for ply in 0..6 {
            let mut record = vec![0u8; RECORD_SIZE];
            let black = source.side_to_move() == Color::Black;
            record[10176] = u8::from(black);
            for sq in 0..90 {
                let Some(piece) = source.piece_at(sq) else {
                    continue;
                };
                let plane = match piece.kind {
                    PieceKind::Rook => 0,
                    PieceKind::Advisor => 1,
                    PieceKind::Cannon => 2,
                    PieceKind::Soldier => 3,
                    PieceKind::Horse => 4,
                    PieceKind::Elephant => 5,
                    PieceKind::General => 6,
                } + if piece.color == source.side_to_move() {
                    0
                } else {
                    7
                };
                let bit = (if black { sq / 9 } else { 9 - sq / 9 }) * 9 + sq % 9;
                record[PLANES + plane * 16 + bit / 8] |= 1 << (bit % 8);
            }
            let decoded = position(&record).unwrap();
            assert_eq!(decoded.hash(), source.hash());
            assert_eq!(decoded.side_to_move(), source.side_to_move());
            let moves = source.legal_moves();
            source.make_move(moves[(ply * 7 + 3) % moves.len()]);
        }
    }
}
