#![allow(dead_code)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::Instant;

use crate::nnue::AZ_NNUE_INPUT_SIZE;
use crate::xiangqi::{BOARD_FILES, BOARD_SIZE};

use super::{
    AzTrainingSample, DENSE_MOVE_SPACE, WDL_HEAD_SIZE, canonical_general_buckets_from_features,
    decode_current_piece_square_feature, normalize_wdl_target, structural_king_piece_index,
};

const POLICY_MASK_VALUE: f32 = -1.0e9;

#[derive(Clone, Debug)]
pub(super) struct DataLoaderConfig {
    pub batch_size: usize,
    pub shuffle: bool,
    pub drop_last: bool,
    pub num_workers: usize,
    pub prefetch_batches: usize,
    pub shard_count: usize,
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
            shard_count: 1,
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
    shards: Vec<Vec<usize>>,
    sample_count: usize,
}

impl BatchPlan {
    pub(super) fn epoch(sample_count: usize, config: &DataLoaderConfig) -> Self {
        let batch_size = config.batch_size.max(1);
        let mut order = (0..sample_count).collect::<Vec<_>>();
        if config.shuffle {
            shuffle_indices(&mut order, config.seed);
        }

        let shard_count = config.shard_count.max(1);
        let mut steps = Vec::with_capacity(sample_count.div_ceil(batch_size));
        for chunk in order.chunks(batch_size) {
            if config.drop_last && chunk.len() < batch_size {
                break;
            }
            let shard_size = chunk.len().div_ceil(shard_count);
            let shards = chunk
                .chunks(shard_size.max(1))
                .map(|shard| shard.to_vec())
                .collect::<Vec<_>>();
            steps.push(BatchStep {
                shards,
                sample_count: chunk.len(),
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
    pub(super) batch_size: usize,
    pub(super) shards: Vec<PackedBatch>,
    pub(super) pack_seconds: f64,
}

#[derive(Clone, Debug)]
pub(super) struct PackedBatch {
    pub batch_size: usize,
    pub max_features: usize,
    pub max_policy_moves: usize,
    pub feature_indices: Vec<u32>,
    pub feature_mask: Vec<f32>,
    pub structural_piece_indices: Vec<u32>,
    pub structural_rank_indices: Vec<u32>,
    pub structural_file_indices: Vec<u32>,
    pub structural_us_king_piece_indices: Vec<u32>,
    pub structural_them_king_piece_indices: Vec<u32>,
    pub structural_mask: Vec<f32>,
    pub square_token_feature_indices: Vec<u32>,
    pub square_token_mask: Vec<f32>,
    pub policy_indices: Vec<u32>,
    pub policy_targets: Vec<f32>,
    pub policy_mask: Vec<f32>,
    pub value_wdl: Vec<f32>,
    pub values: Vec<f32>,
    pub moves_left: Vec<f32>,
    pub policy_weights: Vec<f32>,
    pub value_weights: Vec<f32>,
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

        let mut packed = Self {
            batch_size,
            max_features,
            max_policy_moves,
            feature_indices: vec![0u32; batch_size * max_features],
            feature_mask: vec![0.0f32; batch_size * max_features],
            structural_piece_indices: vec![0u32; batch_size * max_features],
            structural_rank_indices: vec![0u32; batch_size * max_features],
            structural_file_indices: vec![0u32; batch_size * max_features],
            structural_us_king_piece_indices: vec![0u32; batch_size * max_features],
            structural_them_king_piece_indices: vec![0u32; batch_size * max_features],
            structural_mask: vec![0.0f32; batch_size * max_features],
            square_token_feature_indices: vec![0u32; batch_size * BOARD_SIZE],
            square_token_mask: vec![0.0f32; batch_size * BOARD_SIZE],
            policy_indices: vec![0u32; batch_size * max_policy_moves],
            policy_targets: vec![0.0f32; batch_size * max_policy_moves],
            policy_mask: vec![POLICY_MASK_VALUE; batch_size * max_policy_moves],
            value_wdl: vec![0.0f32; batch_size * WDL_HEAD_SIZE],
            values: vec![0.0f32; batch_size],
            moves_left: vec![0.0f32; batch_size],
            policy_weights: vec![1.0f32; batch_size],
            value_weights: vec![1.0f32; batch_size],
        };

        for (row, &sample_index) in batch.iter().enumerate() {
            let sample = &samples[sample_index];
            packed.pack_features(row, sample);
            packed.pack_policy(row, sample);
            let wdl = normalize_wdl_target(sample.value_wdl);
            packed.value_wdl[row * WDL_HEAD_SIZE..(row + 1) * WDL_HEAD_SIZE].copy_from_slice(&wdl);
            packed.values[row] = sample.value.clamp(-1.0, 1.0);
            packed.moves_left[row] = sample.moves_left.max(0.0);
            packed.policy_weights[row] = sample.policy_weight.max(0.0);
            packed.value_weights[row] = sample.value_weight.max(0.0);
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
            self.feature_indices[batch_feature_index] = feature as u32;
            self.feature_mask[batch_feature_index] = 1.0;
            if let Some(structural) = decode_current_piece_square_feature(feature) {
                self.structural_piece_indices[batch_feature_index] = structural.piece_index as u32;
                self.structural_rank_indices[batch_feature_index] = structural.rank as u32;
                self.structural_file_indices[batch_feature_index] = structural.file as u32;
                self.structural_us_king_piece_indices[batch_feature_index] =
                    structural_king_piece_index(0, us_king_bucket, structural.piece_index) as u32;
                self.structural_them_king_piece_indices[batch_feature_index] =
                    structural_king_piece_index(1, them_king_bucket, structural.piece_index) as u32;
                self.structural_mask[batch_feature_index] = 1.0;
                let sq = structural.rank * BOARD_FILES + structural.file;
                self.square_token_feature_indices[row * BOARD_SIZE + sq] =
                    batch_feature_index as u32;
                self.square_token_mask[row * BOARD_SIZE + sq] = 1.0;
            }
        }
    }

    fn pack_policy(&mut self, row: usize, sample: &AzTrainingSample) {
        let policy_base = row * self.max_policy_moves;
        let mut policy_offset = 0usize;
        for (&move_index, &target) in sample.move_indices.iter().zip(sample.policy.iter()) {
            if move_index < DENSE_MOVE_SPACE {
                self.policy_indices[policy_base + policy_offset] = move_index as u32;
                self.policy_targets[policy_base + policy_offset] = target.max(0.0);
                self.policy_mask[policy_base + policy_offset] = 0.0;
                policy_offset += 1;
            }
        }
        normalize_policy_targets(
            &mut self.policy_targets[policy_base..policy_base + self.max_policy_moves],
            policy_offset,
        );
    }
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
                        batch_size: step.sample_count,
                        shards: step
                            .shards
                            .iter()
                            .map(|shard| PackedBatch::from_indices(&samples, shard))
                            .collect(),
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

    use crate::az::AzSampleMeta;

    use super::*;

    fn sample(index: usize) -> AzTrainingSample {
        AzTrainingSample {
            features: vec![index % AZ_NNUE_INPUT_SIZE],
            move_indices: vec![0, 1],
            policy: vec![1.0 + index as f32, 1.0],
            value_wdl: [1.0, 0.0, 0.0],
            value: 2.0,
            side_sign: 1.0,
            moves_left: -1.0,
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
        assert_eq!(plan.steps[0].shards[0], vec![0, 1, 2]);
        assert_eq!(plan.steps[1].shards[0], vec![3, 4, 5]);
    }

    #[test]
    fn batch_plan_splits_steps_into_shards() {
        let config = DataLoaderConfig {
            batch_size: 5,
            shuffle: false,
            shard_count: 2,
            ..DataLoaderConfig::default()
        };
        let plan = BatchPlan::epoch(7, &config);
        assert_eq!(plan.len(), 2);
        assert_eq!(plan.steps[0].sample_count, 5);
        assert_eq!(plan.steps[0].shards, vec![vec![0, 1, 2], vec![3, 4]]);
        assert_eq!(plan.steps[1].sample_count, 2);
        assert_eq!(plan.steps[1].shards, vec![vec![5], vec![6]]);
    }

    #[test]
    fn packed_batch_normalizes_policy_and_clamps_targets() {
        let samples = vec![sample(0), sample(1)];
        let packed = PackedBatch::from_indices(&samples, &[0, 1]);
        assert_eq!(packed.batch_size, 2);
        assert_eq!(packed.max_policy_moves, 2);
        assert_eq!(packed.policy_targets[0], 0.5);
        assert_eq!(packed.policy_targets[1], 0.5);
        assert!((packed.policy_targets[2] - 2.0 / 3.0).abs() < 1.0e-6);
        assert!((packed.policy_targets[3] - 1.0 / 3.0).abs() < 1.0e-6);
        assert_eq!(&packed.value_wdl[0..3], &[1.0, 0.0, 0.0]);
        assert_eq!(packed.values, vec![1.0, 1.0]);
        assert_eq!(packed.moves_left, vec![0.0, 0.0]);
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
            sizes.push(batch.batch_size);
            assert_eq!(batch.shards.len(), 1);
        }
        loader.join().unwrap();
        assert_eq!(sizes, vec![2, 2, 2, 1]);
    }

    #[test]
    fn prefetch_loader_packs_shards() {
        let samples = Arc::new((0..5).map(sample).collect::<Vec<_>>());
        let config = DataLoaderConfig {
            batch_size: 5,
            shuffle: false,
            shard_count: 2,
            ..DataLoaderConfig::default()
        };
        let plan = BatchPlan::epoch(samples.len(), &config);
        let mut loader = PrefetchDataLoader::new(Arc::clone(&samples), plan, &config);
        let batch = loader.next_packed().unwrap().unwrap();
        loader.join().unwrap();
        assert_eq!(batch.batch_size, 5);
        assert_eq!(batch.shards.len(), 2);
        assert_eq!(batch.shards[0].batch_size, 3);
        assert_eq!(batch.shards[1].batch_size, 2);
    }
}
