#![allow(dead_code)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::Instant;

use crate::nnue::AZ_NNUE_INPUT_SIZE;
use crate::xiangqi::{BOARD_SIZE, Color, Position};

use super::{
    AzTrainingSample, DENSE_MOVE_SPACE, POLICY_SPARSE_TABLE_SIZE, POLICY_TACTICAL_SIZE,
    RULE_CONTEXT_SIZE, VALUE_THREAT_MAX_ACTIVE, VALUE_THREAT_VOCAB, WDL_HEAD_SIZE,
    canonical_general_buckets_from_features, decode_current_piece_square_feature,
    dense_move_squares,
    fused_feature_pool::{PADDING_ITEM, pack_feature},
    fused_policy::{pack_policy_item, padding_item as policy_padding_item},
    normalize_wdl_target, policy_sparse_capture_index, policy_sparse_factor_indices,
    policy_sparse_main_index, policy_tactical_indices, value_threat_index,
};

const POLICY_MASK_VALUE: f32 = -1.0e9;

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

    pub(super) fn is_empty(&self) -> bool {
        self.steps.is_empty()
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
    pub max_policy_moves: usize,
    pub max_value_threats: usize,
    pub feature_items: Vec<u32>,
    pub value_threat_indices: Vec<u32>,
    pub policy_items: Vec<i64>,
    pub policy_sparse_indices: Vec<i64>,
    pub policy_tactical_indices: Vec<i64>,
    pub policy_targets: Vec<f32>,
    pub policy_mask: Vec<f32>,
    pub value_wdl: Vec<f32>,
    pub values: Vec<f32>,
    pub rule_context: Vec<f32>,
    pub policy_weights: Vec<f32>,
    pub value_weights: Vec<f32>,
    pub value_phase_masks: Vec<f32>,
    pub value_source_phase_masks: Vec<f32>,
}

impl PackedBatch {
    pub(super) fn from_indices(samples: &[AzTrainingSample], batch: &[usize]) -> Self {
        let batch_size = batch.len();
        let max_features = batch
            .iter()
            .map(|&sample_index| samples[sample_index].features.len())
            .max()
            .unwrap_or(0)
            .max(1);
        let max_policy_moves = batch
            .iter()
            .map(|&sample_index| {
                samples[sample_index]
                    .move_indices
                    .iter()
                    .filter(|&&move_index| move_index < DENSE_MOVE_SPACE)
                    .count()
            })
            .max()
            .unwrap_or(0)
            .max(1);
        let value_threats = batch
            .iter()
            .map(|&sample_index| value_threat_features(&samples[sample_index]))
            .collect::<Vec<_>>();
        let max_value_threats = value_threats.iter().map(Vec::len).max().unwrap_or(0).max(1);

        let mut packed = Self {
            batch_size,
            max_features,
            max_policy_moves,
            max_value_threats,
            feature_items: vec![PADDING_ITEM; batch_size * max_features],
            value_threat_indices: vec![PADDING_ITEM; batch_size * max_value_threats],
            policy_items: vec![policy_padding_item(); batch_size * max_policy_moves],
            policy_sparse_indices: vec![
                (POLICY_SPARSE_TABLE_SIZE - 1) as i64;
                batch_size * max_policy_moves * 7
            ],
            policy_tactical_indices: vec![
                POLICY_TACTICAL_SIZE as i64;
                batch_size * max_policy_moves * 2
            ],
            policy_targets: vec![0.0f32; batch_size * max_policy_moves],
            policy_mask: vec![POLICY_MASK_VALUE; batch_size * max_policy_moves],
            value_wdl: vec![0.0f32; batch_size * WDL_HEAD_SIZE],
            values: vec![0.0f32; batch_size],
            rule_context: vec![0.0f32; batch_size * RULE_CONTEXT_SIZE],
            policy_weights: vec![1.0f32; batch_size],
            value_weights: vec![1.0f32; batch_size],
            value_phase_masks: vec![0.0f32; batch_size * 3],
            value_source_phase_masks: vec![0.0f32; batch_size * 9],
        };

        for (row, (&sample_index, threats)) in batch.iter().zip(&value_threats).enumerate() {
            let sample = &samples[sample_index];
            packed.pack_features(row, sample);
            let threat_base = row * max_value_threats;
            packed.value_threat_indices[threat_base..threat_base + threats.len()]
                .copy_from_slice(threats);
            packed.pack_policy(row, sample);
            let wdl = normalize_wdl_target(sample.value_wdl);
            packed.value_wdl[row * WDL_HEAD_SIZE..(row + 1) * WDL_HEAD_SIZE].copy_from_slice(&wdl);
            packed.values[row] = sample.value.clamp(-1.0, 1.0);
            packed.rule_context[row * RULE_CONTEXT_SIZE..(row + 1) * RULE_CONTEXT_SIZE]
                .copy_from_slice(&sample.rule_context);
            packed.policy_weights[row] = sample.policy_weight.max(0.0);
            packed.value_weights[row] = sample.value_weight.max(0.0);
            let phase = if sample.meta.ply < 40 {
                0
            } else if sample.meta.ply < 120 {
                1
            } else {
                2
            };
            packed.value_phase_masks[row * 3 + phase] = 1.0;
            packed.value_source_phase_masks
                [row * 9 + sample.meta.start_source.index() * 3 + phase] = 1.0;
        }
        packed
    }

    fn pack_features(&mut self, row: usize, sample: &AzTrainingSample) {
        let (us_king_bucket, them_king_bucket) =
            canonical_general_buckets_from_features(&sample.features);
        let feature_base = row * self.max_features;
        for (feature_offset, &feature) in sample.features.iter().enumerate() {
            if feature >= AZ_NNUE_INPUT_SIZE {
                continue;
            }
            let batch_feature_index = feature_base + feature_offset;
            self.feature_items[batch_feature_index] =
                pack_feature(feature, us_king_bucket, them_king_bucket);
        }
    }

    fn pack_policy(&mut self, row: usize, sample: &AzTrainingSample) {
        let policy_base = row * self.max_policy_moves;
        let king_buckets = canonical_general_buckets_from_features(&sample.features);
        let mut board_features = [usize::MAX; BOARD_SIZE];
        let mut pieces = Vec::with_capacity(sample.features.len());
        for &feature in &sample.features {
            if let Some(structural) = decode_current_piece_square_feature(feature) {
                let square = structural.rank * 9 + structural.file;
                board_features[square] = feature;
                pieces.push((structural.piece_index, square));
            }
        }
        let position = Position::from_canonical_piece_squares(&pieces);
        let opponent_attacks = position.attacked_squares_mask(crate::xiangqi::Color::Black);
        let own_attacks = position.attacked_squares_mask(crate::xiangqi::Color::Red);
        let mut policy_offset = 0usize;
        for (&move_index, &target) in sample.move_indices.iter().zip(sample.policy.iter()) {
            if move_index < DENSE_MOVE_SPACE {
                self.policy_targets[policy_base + policy_offset] = target.max(0.0);
                self.policy_mask[policy_base + policy_offset] = 0.0;
                let mut consequence_from = 0usize;
                let mut consequence_to = 0usize;
                let mut consequence_captured = 0usize;
                let mut move_valid = false;
                let mut capture_valid = false;
                if let Some((from, to)) = dense_move_squares(move_index) {
                    let moved_feature = board_features[from];
                    if moved_feature != usize::MAX
                        && moved_feature / BOARD_SIZE < super::STRUCTURAL_PIECE_SIZE / 2
                    {
                        let piece_index = moved_feature / BOARD_SIZE;
                        consequence_from = moved_feature;
                        consequence_to = piece_index * BOARD_SIZE + to;
                        move_valid = true;
                        let captured_feature = board_features[to];
                        if captured_feature != usize::MAX {
                            consequence_captured = captured_feature;
                            capture_valid = true;
                        }
                    }
                }
                let item_index = policy_base + policy_offset;
                self.policy_items[item_index] = pack_policy_item(
                    move_index,
                    consequence_from,
                    consequence_to,
                    consequence_captured,
                    move_valid,
                    capture_valid,
                );
                if move_valid {
                    let moved_piece = consequence_from / BOARD_SIZE;
                    let captured_piece = capture_valid.then_some(consequence_captured / BOARD_SIZE);
                    let sparse_base = item_index * 7;
                    self.policy_sparse_indices[sparse_base] = policy_sparse_main_index(
                        move_index,
                        moved_piece,
                        king_buckets.0,
                        king_buckets.1,
                    ) as i64;
                    self.policy_sparse_indices[sparse_base + 1] =
                        policy_sparse_capture_index(move_index, captured_piece) as i64;
                    for (offset, factor) in policy_sparse_factor_indices(
                        move_index,
                        moved_piece,
                        king_buckets.0,
                        king_buckets.1,
                    )
                    .into_iter()
                    .enumerate()
                    {
                        self.policy_sparse_indices[sparse_base + 2 + offset] =
                            (POLICY_SPARSE_TABLE_SIZE + factor) as i64;
                    }
                    let mv = crate::xiangqi::Move::new(
                        consequence_from % BOARD_SIZE,
                        consequence_to % BOARD_SIZE,
                    );
                    let check = position.gives_check_after_move_fast(mv);
                    let source_attacked = opponent_attacks & (1u128 << mv.from as usize) != 0;
                    let destination_attacked = opponent_attacks & (1u128 << mv.to as usize) != 0;
                    let source_defended = own_attacks & (1u128 << mv.from as usize) != 0;
                    let destination_defended = own_attacks & (1u128 << mv.to as usize) != 0;
                    let tactical_base = item_index * 2;
                    for (offset, tactical) in policy_tactical_indices(
                        move_index,
                        moved_piece,
                        source_attacked,
                        destination_attacked,
                        source_defended,
                        destination_defended,
                        capture_valid,
                        check,
                    )
                    .into_iter()
                    .enumerate()
                    {
                        self.policy_tactical_indices[tactical_base + offset] = tactical as i64;
                    }
                }
                policy_offset += 1;
            }
        }
        normalize_policy_targets(
            &mut self.policy_targets[policy_base..policy_base + self.max_policy_moves],
            policy_offset,
        );
    }
}

fn value_threat_features(sample: &AzTrainingSample) -> Vec<u32> {
    let pieces = sample
        .features
        .iter()
        .filter_map(|&feature| decode_current_piece_square_feature(feature))
        .map(|piece| (piece.piece_index, piece.rank * 9 + piece.file))
        .collect::<Vec<_>>();
    let position = Position::from_canonical_piece_squares(&pieces);
    let mut features = Vec::with_capacity(32);
    position.visit_occupied_relations(|source, attacker, target, attacked| {
        let feature = value_threat_index(Color::Red, source, attacker, target, attacked);
        if feature != VALUE_THREAT_VOCAB {
            features.push(feature as u32);
        }
    });
    assert!(
        features.len() <= VALUE_THREAT_MAX_ACTIVE,
        "too many active value threats"
    );
    features
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{az::AzSampleMeta, xiangqi::Move};

    use super::*;

    fn sample(index: usize) -> AzTrainingSample {
        AzTrainingSample {
            features: vec![index % AZ_NNUE_INPUT_SIZE],
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            move_indices: vec![0, 1],
            policy: vec![1.0 + index as f32, 1.0],
            value_wdl: [1.0, 0.0, 0.0],
            value: 2.0,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            search_simulations: 0,
            meta: AzSampleMeta::default(),
        }
    }

    #[test]
    fn batch_plan_respects_drop_last() {
        let config = DataLoaderConfig {
            batch_size: 3,
            shuffle: false,
            drop_last: true,
            ..DataLoaderConfig::default()
        };
        let plan = BatchPlan::epoch(8, &config);
        assert_eq!(plan.len(), 2);
        assert_eq!(plan.steps[0].indices, vec![0, 1, 2]);
        assert_eq!(plan.steps[1].indices, vec![3, 4, 5]);
    }

    #[test]
    fn packed_batch_normalizes_policy_and_clamps_targets() {
        let mut samples = vec![sample(0), sample(1)];
        samples[1].meta.start_source = crate::az::AzStartSource::Midgame;
        samples[1].meta.ply = 130;
        let packed = PackedBatch::from_indices(&samples, &[0, 1]);
        assert_eq!(packed.batch_size, 2);
        assert_eq!(packed.max_policy_moves, 2);
        assert_eq!(packed.policy_targets[0], 0.5);
        assert_eq!(packed.policy_targets[1], 0.5);
        assert!((packed.policy_targets[2] - 2.0 / 3.0).abs() < 1.0e-6);
        assert!((packed.policy_targets[3] - 1.0 / 3.0).abs() < 1.0e-6);
        assert_eq!(&packed.value_wdl[0..3], &[1.0, 0.0, 0.0]);
        assert_eq!(packed.values, vec![1.0, 1.0]);
        assert_eq!(packed.value_source_phase_masks[0], 1.0);
        assert_eq!(packed.value_source_phase_masks[9 + 8], 1.0);
        assert_eq!(packed.value_source_phase_masks.iter().sum::<f32>(), 2.0);
    }

    #[test]
    fn packed_policy_consequence_encodes_move_and_capture() {
        let moved_feature = 6 * BOARD_SIZE;
        let captured_feature = 10 * BOARD_SIZE + 1;
        let move_index = super::super::dense_move_index(Move::new(0, 1));
        let mut training_sample = sample(0);
        training_sample.features = vec![moved_feature, captured_feature];
        training_sample.move_indices = vec![move_index];
        training_sample.policy = vec![1.0];

        let packed = PackedBatch::from_indices(&[training_sample], &[0]);
        assert_eq!(
            packed.policy_items,
            vec![pack_policy_item(
                move_index,
                moved_feature,
                6 * BOARD_SIZE + 1,
                captured_feature,
                true,
                true,
            )]
        );
    }

    #[test]
    fn prefetch_loader_preserves_batch_order() {
        let samples = Arc::new((0..7).map(sample).collect::<Vec<_>>());
        let config = DataLoaderConfig {
            batch_size: 2,
            shuffle: false,
            drop_last: false,
            num_workers: 2,
            prefetch_batches: 2,
            ..DataLoaderConfig::default()
        };
        let plan = BatchPlan::epoch(samples.len(), &config);
        let mut loader = PrefetchDataLoader::new(Arc::clone(&samples), plan, &config);
        let mut sizes = Vec::new();
        while let Some(batch) = loader.next_packed().unwrap() {
            sizes.push(batch.batch.batch_size);
        }
        loader.join().unwrap();
        assert_eq!(sizes, vec![2, 2, 2, 1]);
    }
}
