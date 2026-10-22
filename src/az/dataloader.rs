use super::{features::*, fused_feature_pool::PADDING_ITEM, *};
use std::{
    collections::{BTreeMap, HashSet},
    sync::{Arc, Mutex, mpsc},
    thread,
    time::Instant,
};
const POLICY_MASK_VALUE: f32 = -1e9;
#[derive(Clone, Debug, Default)]
pub struct AzSparseActivationStats {
    pub inspected_samples: usize,
    pub inspected_policy_moves: usize,
    pub relation_unique: usize,
    pub motif_unique: usize,
    pub policy_joint_unique: usize,
    pub policy_event_unique: usize,
    pub relation_coverage: f32,
    pub motif_coverage: f32,
    pub policy_joint_coverage: f32,
    pub policy_event_coverage: f32,
}
pub fn sparse_activation_stats(
    samples: &[AzTrainingSample],
    limit: usize,
    seed: u64,
) -> AzSparseActivationStats {
    let mut order = (0..samples.len()).collect::<Vec<_>>();
    shuffle_indices(&mut order, seed);
    order.truncate(limit.min(order.len()));
    let mut stats = AzSparseActivationStats::default();
    let mut relations = HashSet::new();
    let mut motifs = HashSet::new();
    let mut joints = HashSet::new();
    let mut events = HashSet::new();
    for i in order {
        let sample = &samples[i];
        let Some(p) = position_for_training_sample(sample) else {
            continue;
        };
        let g = Graph::new(Board::new(&p));
        let rows = extract(&g, Color::Red, u128::MAX);
        relations.extend(rows.relations);
        motifs.extend(rows.motifs);
        motifs.extend(king_context(&g, Color::Red).1);
        stats.inspected_samples += 1;
        if sample.policy_weight > 0. {
            for &id in &sample.move_indices {
                if let Some((a, b)) = dense_move_squares(id) {
                    if g.board.cells[a].is_some_and(|p| p.color == Color::Red) {
                        let m = move_features(&g, Move::new(a, b), Color::Red, None);
                        joints.insert(m.joint);
                        events.extend(m.events);
                        stats.inspected_policy_moves += 1;
                    }
                }
            }
        }
    }
    stats.relation_unique = relations.len();
    stats.motif_unique = motifs.len();
    stats.policy_joint_unique = joints.len();
    stats.policy_event_unique = events.len();
    stats.relation_coverage = relations.len() as f32 / RELATION_ROWS as f32;
    stats.motif_coverage = motifs.len() as f32 / MOTIF_ROWS as f32;
    stats.policy_joint_coverage = joints.len() as f32 / POLICY_JOINT_ROWS as f32;
    stats.policy_event_coverage = events.len() as f32 / POLICY_EVENT_ROWS as f32;
    stats
}
#[derive(Clone, Debug)]
pub(super) struct DataLoaderConfig {
    pub batch_size: usize,
    pub shuffle: bool,
    pub drop_last: bool,
    pub num_workers: usize,
    pub prefetch_batches: usize,
    pub seed: u64,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 4096,
            shuffle: true,
            drop_last: false,
            num_workers: 1,
            prefetch_batches: 2,
            seed: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct BatchPlan {
    steps: Vec<BatchStep>,
}

#[derive(Clone, Debug)]
struct BatchStep {
    indices: Vec<usize>,
}

impl BatchPlan {
    pub(super) fn epoch(sample_count: usize, config: &DataLoaderConfig) -> Self {
        let batch_size = config.batch_size.max(1);
        let mut order = (0..sample_count).collect::<Vec<_>>();
        if config.shuffle {
            shuffle_indices(&mut order, config.seed);
        }

        let mut steps = Vec::with_capacity(sample_count.div_ceil(batch_size));
        for chunk in order.chunks(batch_size) {
            if config.drop_last && chunk.len() < batch_size {
                break;
            }
            steps.push(BatchStep {
                indices: chunk.to_vec(),
            });
        }
        Self { steps }
    }

    pub(super) fn len(&self) -> usize {
        self.steps.len()
    }
}

#[derive(Clone, Debug)]
pub(super) struct PackedStepBatch {
    pub(super) batch: PackedBatch,
    pub(super) pack_seconds: f64,
}

#[derive(Clone, Debug)]
pub(super) struct PackedBatch {
    pub batch_size: usize,
    pub max_features: usize,
    pub max_king: usize,
    pub max_relations: usize,
    pub max_motifs: usize,
    pub max_policy_moves: usize,
    pub max_policy_terms: usize,
    pub value_buckets: Vec<u32>,
    pub feature_items: Vec<u32>,
    pub king_indices: Vec<u32>,
    pub relation_indices: Vec<u32>,
    pub motif_indices: Vec<u32>,
    pub move_indices: Vec<u32>,
    pub policy_indices: Vec<u32>,
    pub policy_from: Vec<u32>,
    pub policy_to: Vec<u32>,
    pub policy_capture: Vec<u32>,
    pub policy_local: Vec<f32>,
    pub policy_targets: Vec<f32>,
    pub policy_mask: Vec<f32>,
    pub policy_repetition: Vec<f32>,
    pub value_wdl: Vec<f32>,
    pub values: Vec<f32>,
    pub rule_context: Vec<f32>,
    pub check_context: Vec<f32>,
    pub policy_weights: Vec<f32>,
    pub value_weights: Vec<f32>,
    pub moves_left_targets: Vec<f32>,
    pub moves_left_weights: Vec<f32>,
    pub value_phase_masks: Vec<f32>,
    pub value_source_phase_masks: Vec<f32>,
}
impl PackedBatch {
    pub fn from_indices(samples: &[AzTrainingSample], batch: &[usize]) -> Self {
        let batch_size = batch.len();
        let mut rows = Vec::with_capacity(batch_size);
        for &index in batch {
            let sample = &samples[index];
            let p = position_for_training_sample(sample)
                .expect("training sample requires both generals");
            let graph = Graph::new(Board::new(&p));
            let mut features = extract(&graph, Color::Red, u128::MAX);
            let (checks, seals) = king_context(&graph, Color::Red);
            features.motifs.extend(seals);
            let kings = king_piece_rows(&graph.board, Color::Red);
            let buckets = canonical_buckets_for_perspective(&p, Color::Red);
            let mut moves = Vec::new();
            let mut move_cache = MoveFeatureCache::new(
                &graph,
                sample
                    .move_indices
                    .iter()
                    .filter_map(|&id| dense_move_squares(id))
                    .map(|(a, b)| Move::new(a, b)),
            );
            for (offset, &id) in sample.move_indices.iter().enumerate() {
                if let Some((a, b)) = dense_move_squares(id) {
                    assert!(
                        graph.board.cells[a].is_some_and(|p| p.color == Color::Red),
                        "invalid policy source"
                    );
                    let local = move_cache.features(
                        Move::new(a, b),
                        Color::Red,
                        sample
                            .repetition_flags
                            .get(offset)
                            .and_then(|&v| if v < 2 { Some(v != 0) } else { None }),
                    );
                    let terms = policy_terms(id, &local, buckets);
                    moves.push((offset, id, local, terms));
                }
            }
            rows.push((
                features,
                kings,
                checks,
                moves,
                value_material_bucket(&p, Color::Red),
            ));
        }
        let max_features = batch
            .iter()
            .map(|&i| samples[i].features.len())
            .max()
            .unwrap_or(0)
            .max(1);
        let max_king = rows.iter().map(|r| r.1.len()).max().unwrap_or(0).max(1);
        let max_relations = rows
            .iter()
            .map(|r| r.0.relations.len())
            .max()
            .unwrap_or(0)
            .max(1);
        let max_motifs = rows
            .iter()
            .map(|r| r.0.motifs.len())
            .max()
            .unwrap_or(0)
            .max(1);
        let max_policy_moves = rows.iter().map(|r| r.3.len()).max().unwrap_or(0).max(1);
        let max_policy_terms = rows
            .iter()
            .flat_map(|r| r.3.iter().map(|m| m.3.len()))
            .max()
            .unwrap_or(0)
            .max(1);
        let n = batch_size * max_policy_moves;
        let mut result = Self {
            batch_size,
            max_features,
            max_king,
            max_relations,
            max_motifs,
            max_policy_moves,
            max_policy_terms,
            value_buckets: vec![0; batch_size],
            feature_items: vec![PADDING_ITEM; batch_size * max_features],
            king_indices: vec![PADDING_ITEM; batch_size * max_king],
            relation_indices: vec![PADDING_ITEM; batch_size * max_relations],
            motif_indices: vec![PADDING_ITEM; batch_size * max_motifs],
            move_indices: vec![0; n],
            policy_indices: vec![PADDING_ITEM; n * max_policy_terms],
            policy_from: vec![1260; n],
            policy_to: vec![1260; n],
            policy_capture: vec![1260; n],
            policy_local: vec![0.; n * MOVE_LOCAL_SIZE],
            policy_targets: vec![0.; n],
            policy_mask: vec![POLICY_MASK_VALUE; n],
            policy_repetition: vec![0.; n],
            value_wdl: vec![0.; batch_size * 3],
            values: vec![0.; batch_size],
            rule_context: vec![0.; batch_size * RULE_CONTEXT_SIZE],
            check_context: vec![0.; batch_size * CHECK_CONTEXT_SIZE],
            policy_weights: vec![0.; batch_size],
            value_weights: vec![0.; batch_size],
            moves_left_targets: vec![0.; batch_size],
            moves_left_weights: vec![0.; batch_size],
            value_phase_masks: vec![0.; batch_size * 3],
            value_source_phase_masks: vec![0.; batch_size * 9],
        };
        for (row, (&index, (features, kings, checks, moves, bucket))) in
            batch.iter().zip(rows).enumerate()
        {
            let sample = &samples[index];
            result.value_buckets[row] = bucket as u32;
            for (i, &feature) in sample.features.iter().enumerate() {
                assert!(feature < 1260);
                result.feature_items[row * max_features + i] = feature as u32;
            }
            result.king_indices[row * max_king..row * max_king + kings.len()]
                .copy_from_slice(&kings);
            result.relation_indices
                [row * max_relations..row * max_relations + features.relations.len()]
                .copy_from_slice(&features.relations);
            result.motif_indices[row * max_motifs..row * max_motifs + features.motifs.len()]
                .copy_from_slice(&features.motifs);
            result.check_context[row * CHECK_CONTEXT_SIZE..(row + 1) * CHECK_CONTEXT_SIZE]
                .copy_from_slice(&checks);
            let active = moves.len();
            for (column, (offset, id, local, terms)) in moves.into_iter().enumerate() {
                let i = row * max_policy_moves + column;
                result.move_indices[i] = id as u32;
                result.policy_targets[i] = sample.policy[offset].max(0.);
                result.policy_mask[i] = 0.;
                result.policy_from[i] = local.from_feature as u32;
                result.policy_to[i] = local.to_feature as u32;
                result.policy_capture[i] = local.captured_feature.unwrap_or(1260) as u32;
                result.policy_local[i * MOVE_LOCAL_SIZE..(i + 1) * MOVE_LOCAL_SIZE]
                    .copy_from_slice(&local.local);
                result.policy_repetition[i] = local.local[10];
                result.policy_indices[i * max_policy_terms..i * max_policy_terms + terms.len()]
                    .copy_from_slice(&terms);
            }
            normalize_policy_targets(
                &mut result.policy_targets[row * max_policy_moves..(row + 1) * max_policy_moves],
                active,
            );
            result.value_wdl[row * 3..row * 3 + 3]
                .copy_from_slice(&normalize_wdl_target(sample.value_wdl));
            result.values[row] = sample.value.clamp(-1., 1.);
            result.rule_context[row * RULE_CONTEXT_SIZE..(row + 1) * RULE_CONTEXT_SIZE]
                .copy_from_slice(&sample.rule_context);
            result.policy_weights[row] = sample.policy_weight.max(0.);
            result.value_weights[row] = sample.value_weight.max(0.);
            if sample.moves_left.is_finite()
                && sample.moves_left >= 0.
                && sample.moves_left_weight.is_finite()
            {
                result.moves_left_targets[row] = sample.moves_left / MOVES_LEFT_SCALE;
                result.moves_left_weights[row] = sample.moves_left_weight.max(0.);
            }
            let phase = if sample.meta.ply < 40 {
                0
            } else if sample.meta.ply < 120 {
                1
            } else {
                2
            };
            result.value_phase_masks[row * 3 + phase] = 1.;
            result.value_source_phase_masks
                [row * 9 + sample.meta.start_source.index() * 3 + phase] = 1.;
        }
        result
    }
}
fn policy_terms(id: usize, m: &MoveFeatures, kings: (usize, usize)) -> Vec<u32> {
    let piece = DENSE_MOVE_SPACE;
    let king = piece + DENSE_MOVE_SPACE * 7;
    let capture = king + DENSE_MOVE_SPACE * 18;
    let joint = capture + DENSE_MOVE_SPACE * 8;
    let events = joint + POLICY_JOINT_ROWS;
    let mut terms = vec![
        id as u32,
        (piece + id * 7 + m.moved) as u32,
        (king + id * 9 + kings.0) as u32,
        (king + (DENSE_MOVE_SPACE + id) * 9 + kings.1) as u32,
        (capture + id * 8 + m.captured) as u32,
        (joint + m.joint as usize) as u32,
    ];
    terms.extend(m.events.iter().map(|&e| (events + e as usize) as u32));
    terms
}
#[derive(Debug)]
pub(super) enum DataLoaderError {
    WorkerPanic,
    Closed,
}

pub(super) struct PrefetchDataLoader {
    rx: mpsc::Receiver<(usize, PackedStepBatch)>,
    workers: Vec<thread::JoinHandle<()>>,
    next_batch_id: usize,
    total_batches: usize,
    pending: BTreeMap<usize, PackedStepBatch>,
}

impl PrefetchDataLoader {
    pub(super) fn new(
        samples: Arc<Vec<AzTrainingSample>>,
        plan: BatchPlan,
        config: &DataLoaderConfig,
    ) -> Self {
        let total_batches = plan.len();
        let workers = config.num_workers.max(1);
        let channel_depth = config.prefetch_batches.max(1) * workers;
        let (tx, rx) = mpsc::sync_channel(channel_depth);
        let plan = Arc::new(plan.steps);
        let cursor = Arc::new(Mutex::new(0usize));
        let mut handles = Vec::with_capacity(workers);

        for _ in 0..workers {
            let tx = tx.clone();
            let samples = Arc::clone(&samples);
            let plan = Arc::clone(&plan);
            let cursor = Arc::clone(&cursor);
            handles.push(thread::spawn(move || {
                loop {
                    let batch_id = {
                        let mut cursor = cursor.lock().expect("dataloader cursor poisoned");
                        if *cursor >= plan.len() {
                            return;
                        }
                        let batch_id = *cursor;
                        *cursor += 1;
                        batch_id
                    };
                    let started = Instant::now();
                    let step = &plan[batch_id];
                    let packed = PackedStepBatch {
                        batch: PackedBatch::from_indices(&samples, &step.indices),
                        pack_seconds: started.elapsed().as_secs_f64(),
                    };
                    if tx.send((batch_id, packed)).is_err() {
                        return;
                    }
                }
            }));
        }
        drop(tx);

        Self {
            rx,
            workers: handles,
            next_batch_id: 0,
            total_batches,
            pending: BTreeMap::new(),
        }
    }

    pub(super) fn next_packed(&mut self) -> Result<Option<PackedStepBatch>, DataLoaderError> {
        if self.next_batch_id >= self.total_batches {
            return Ok(None);
        }
        if let Some(batch) = self.pending.remove(&self.next_batch_id) {
            self.next_batch_id += 1;
            return Ok(Some(batch));
        }

        while let Ok((batch_id, batch)) = self.rx.recv() {
            if batch_id == self.next_batch_id {
                self.next_batch_id += 1;
                return Ok(Some(batch));
            }
            self.pending.insert(batch_id, batch);
        }
        Err(DataLoaderError::Closed)
    }

    pub(super) fn join(self) -> Result<(), DataLoaderError> {
        for worker in self.workers {
            worker.join().map_err(|_| DataLoaderError::WorkerPanic)?;
        }
        Ok(())
    }
}

fn normalize_policy_targets(targets: &mut [f32], active: usize) {
    if active == 0 {
        return;
    }
    let active_targets = &mut targets[..active];
    let sum = active_targets.iter().copied().sum::<f32>();
    if sum.is_finite() && sum > 1.0e-12 {
        for target in active_targets.iter_mut() {
            *target = (*target / sum).max(0.0);
        }
    } else {
        let uniform = 1.0 / active as f32;
        active_targets.fill(uniform);
    }
}

fn shuffle_indices(values: &mut [usize], seed: u64) {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    for index in (1..values.len()).rev() {
        state = splitmix_next(&mut state);
        values.swap(index, (state as usize) % (index + 1));
    }
}

fn splitmix_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}
