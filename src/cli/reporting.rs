use chineseai::az::{AzNnue, AzSearchLimits, SplitMix64, alphazero_search};
use chineseai::xiangqi::Position;
use rusqlite::Connection;
use std::io;
use std::sync::Arc;
use std::thread;

#[derive(Clone, Debug)]
pub(crate) struct PikafishLabelRow {
    pub(crate) id: i64,
    pub(crate) fen: String,
    pub(crate) bestmove: String,
    pub(crate) best_wdl: [u16; 3],
}

#[derive(Default)]
pub(crate) struct LabelEvalStats {
    pub(crate) count: usize,
    pub(crate) legal_bestmove: usize,
    pub(crate) top1_hits: usize,
    pub(crate) top2_hits: usize,
    pub(crate) top4_hits: usize,
    pub(crate) top8_hits: usize,
    pub(crate) prior_top1_hits: usize,
    pub(crate) value_pairs: usize,
    pub(crate) value_q_sum: f64,
    pub(crate) target_q_sum: f64,
    pub(crate) value_q_sq_sum: f64,
    pub(crate) target_q_sq_sum: f64,
    pub(crate) value_target_cross_sum: f64,
    pub(crate) abs_value_error_sum: f64,
    pub(crate) raw_value_pairs: usize,
    pub(crate) raw_value_q_sum: f64,
    pub(crate) raw_target_q_sum: f64,
    pub(crate) raw_value_q_sq_sum: f64,
    pub(crate) raw_target_q_sq_sum: f64,
    pub(crate) raw_value_target_cross_sum: f64,
    pub(crate) raw_abs_value_error_sum: f64,
}

impl LabelEvalStats {
    pub(crate) fn merge(&mut self, other: LabelEvalStats) {
        self.count += other.count;
        self.legal_bestmove += other.legal_bestmove;
        self.top1_hits += other.top1_hits;
        self.top2_hits += other.top2_hits;
        self.top4_hits += other.top4_hits;
        self.top8_hits += other.top8_hits;
        self.prior_top1_hits += other.prior_top1_hits;
        self.value_pairs += other.value_pairs;
        self.value_q_sum += other.value_q_sum;
        self.target_q_sum += other.target_q_sum;
        self.value_q_sq_sum += other.value_q_sq_sum;
        self.target_q_sq_sum += other.target_q_sq_sum;
        self.value_target_cross_sum += other.value_target_cross_sum;
        self.abs_value_error_sum += other.abs_value_error_sum;
        self.raw_value_pairs += other.raw_value_pairs;
        self.raw_value_q_sum += other.raw_value_q_sum;
        self.raw_target_q_sum += other.raw_target_q_sum;
        self.raw_value_q_sq_sum += other.raw_value_q_sq_sum;
        self.raw_target_q_sq_sum += other.raw_target_q_sq_sum;
        self.raw_value_target_cross_sum += other.raw_value_target_cross_sum;
        self.raw_abs_value_error_sum += other.raw_abs_value_error_sum;
    }

    pub(crate) fn denom(&self) -> f32 {
        self.count.max(1) as f32
    }

    pub(crate) fn top1_rate(&self) -> f32 {
        self.top1_hits as f32 / self.denom()
    }

    pub(crate) fn top2_rate(&self) -> f32 {
        self.top2_hits as f32 / self.denom()
    }

    pub(crate) fn top4_rate(&self) -> f32 {
        self.top4_hits as f32 / self.denom()
    }

    pub(crate) fn top8_rate(&self) -> f32 {
        self.top8_hits as f32 / self.denom()
    }

    pub(crate) fn prior_top1_rate(&self) -> f32 {
        self.prior_top1_hits as f32 / self.denom()
    }

    pub(crate) fn value_mae_wdl_q(&self) -> f32 {
        (self.abs_value_error_sum / self.value_count().max(1) as f64) as f32
    }

    pub(crate) fn target_q(wdl: [u16; 3]) -> f64 {
        (f64::from(wdl[0]) - f64::from(wdl[2])) / 1000.0
    }

    pub(crate) fn push_value_pair(&mut self, value_q: f32, wdl: [u16; 3]) {
        let target = Self::target_q(wdl);
        let value = value_q as f64;
        self.value_pairs += 1;
        self.value_q_sum += value;
        self.target_q_sum += target;
        self.value_q_sq_sum += value * value;
        self.target_q_sq_sum += target * target;
        self.value_target_cross_sum += value * target;
        self.abs_value_error_sum += (value - target).abs();
    }

    pub(crate) fn value_count(&self) -> usize {
        self.value_pairs
    }

    pub(crate) fn push_raw_value_pair(&mut self, value_q: f32, wdl: [u16; 3]) {
        let target = Self::target_q(wdl);
        let value = value_q as f64;
        self.raw_value_pairs += 1;
        self.raw_value_q_sum += value;
        self.raw_target_q_sum += target;
        self.raw_value_q_sq_sum += value * value;
        self.raw_target_q_sq_sum += target * target;
        self.raw_value_target_cross_sum += value * target;
        self.raw_abs_value_error_sum += (value - target).abs();
    }

    pub(crate) fn raw_value_mae_wdl_q(&self) -> f32 {
        (self.raw_abs_value_error_sum / self.raw_value_pairs.max(1) as f64) as f32
    }

    pub(crate) fn raw_value_corr(&self) -> f64 {
        let n = self.raw_value_pairs as f64;
        if n <= 1.0 {
            return 0.0;
        }
        let cov =
            self.raw_value_target_cross_sum - self.raw_value_q_sum * self.raw_target_q_sum / n;
        let left = self.raw_value_q_sq_sum - self.raw_value_q_sum * self.raw_value_q_sum / n;
        let right = self.raw_target_q_sq_sum - self.raw_target_q_sum * self.raw_target_q_sum / n;
        if left <= 0.0 || right <= 0.0 {
            0.0
        } else {
            cov / (left * right).sqrt()
        }
    }

    pub(crate) fn value_corr(&self) -> f64 {
        let n = self.value_count() as f64;
        if n <= 1.0 {
            return 0.0;
        }
        let cov = self.value_target_cross_sum - self.value_q_sum * self.target_q_sum / n;
        let left = self.value_q_sq_sum - self.value_q_sum * self.value_q_sum / n;
        let right = self.target_q_sq_sum - self.target_q_sum * self.target_q_sum / n;
        if left <= 0.0 || right <= 0.0 {
            0.0
        } else {
            cov / (left * right).sqrt()
        }
    }
}

pub(crate) fn evaluate_pikafish_labels(
    model: &AzNnue,
    rows: &[PikafishLabelRow],
    search_limits: AzSearchLimits,
    mut progress: impl FnMut(usize, usize),
) -> io::Result<LabelEvalStats> {
    let mut stats = LabelEvalStats::default();
    for (offset, row) in rows.iter().enumerate() {
        let position = Position::from_fen(&row.fen).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid FEN id={}: {err}", row.id),
            )
        })?;
        let rule_history = position.initial_rule_history();
        if position.rule_outcome_with_history(&rule_history).is_some() {
            continue;
        }
        let Some(label_move) = position.parse_uci_move(&row.bestmove) else {
            continue;
        };
        let legal_moves = position.legal_moves_with_rules(&rule_history);
        if !legal_moves.contains(&label_move) {
            continue;
        }
        stats.legal_bestmove += 1;
        let raw_value = model.evaluate_value_with_rules(&position, &rule_history, &legal_moves);
        stats.push_raw_value_pair(raw_value, row.best_wdl);
        let result = alphazero_search(
            &position,
            model,
            AzSearchLimits {
                seed: search_limits.seed ^ row.id as u64,
                ..search_limits
            },
        );
        stats.count += 1;
        if result.best_move == Some(label_move) {
            stats.top1_hits += 1;
        }
        let mut by_visits = result.candidates.clone();
        by_visits.sort_by(|left, right| {
            right
                .visits
                .cmp(&left.visits)
                .then_with(|| right.policy.total_cmp(&left.policy))
        });
        if by_visits
            .iter()
            .take(2)
            .any(|candidate| candidate.mv == label_move)
        {
            stats.top2_hits += 1;
        }
        if by_visits
            .iter()
            .take(4)
            .any(|candidate| candidate.mv == label_move)
        {
            stats.top4_hits += 1;
        }
        if by_visits
            .iter()
            .take(8)
            .any(|candidate| candidate.mv == label_move)
        {
            stats.top8_hits += 1;
        }
        if result
            .candidates
            .iter()
            .max_by(|left, right| left.raw_prior.total_cmp(&right.raw_prior))
            .is_some_and(|candidate| candidate.mv == label_move)
        {
            stats.prior_top1_hits += 1;
        }
        stats.push_value_pair(result.value_q, row.best_wdl);
        progress(offset + 1, rows.len());
    }
    Ok(stats)
}

pub(crate) fn evaluate_pikafish_labels_parallel(
    model: Arc<AzNnue>,
    rows: Vec<PikafishLabelRow>,
    search_limits: AzSearchLimits,
    thread_count: usize,
) -> io::Result<LabelEvalStats> {
    if rows.is_empty() {
        return Ok(LabelEvalStats::default());
    }
    let thread_count = thread_count.max(1).min(rows.len());
    let rows = Arc::new(rows);
    let mut handles = Vec::with_capacity(thread_count);
    for thread_id in 0..thread_count {
        let model = Arc::clone(&model);
        let rows = Arc::clone(&rows);
        handles.push(thread::spawn(move || {
            let shard: Vec<_> = rows
                .iter()
                .enumerate()
                .filter(|(index, _)| index % thread_count == thread_id)
                .map(|(_, row)| row.clone())
                .collect();
            evaluate_pikafish_labels(&model, &shard, search_limits, |_, _| {})
        }));
    }

    let mut merged = LabelEvalStats::default();
    for handle in handles {
        let stats = handle
            .join()
            .map_err(|_| io::Error::other("pikafish label eval thread panicked"))??;
        merged.merge(stats);
    }
    Ok(merged)
}

pub(crate) fn load_pikafish_label_rows(
    conn: &Connection,
    limit: usize,
    seed: u64,
) -> rusqlite::Result<Vec<PikafishLabelRow>> {
    let mut stmt = conn.prepare(
        "SELECT id, fen, bestmove, wdl_win, wdl_draw, wdl_loss FROM pikafish_labels ORDER BY id",
    )?;
    let mut rows: Vec<_> = stmt
        .query_map([], |row| {
            Ok(PikafishLabelRow {
                id: row.get(0)?,
                fen: row.get(1)?,
                bestmove: row.get(2)?,
                best_wdl: [row.get(3)?, row.get(4)?, row.get(5)?],
            })
        })?
        .collect::<rusqlite::Result<_>>()?;
    if limit > 0 && limit < rows.len() {
        let mut rng = SplitMix64::new(seed ^ 0xA076_1D64_78BD_642F);
        for index in (1..rows.len()).rev() {
            rows.swap(index, rng.next_u64() as usize % (index + 1));
        }
        rows.truncate(limit);
    }
    Ok(rows)
}

pub(crate) fn sqlite_io_error(err: rusqlite::Error) -> io::Error {
    io::Error::other(err.to_string())
}

