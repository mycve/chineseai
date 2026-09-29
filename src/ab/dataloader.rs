#![allow(dead_code)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::Instant;

use super::{
    AbTrainingSample, RULE_CONTEXT_SIZE, WDL_HEAD_SIZE, canonical_general_buckets_from_features,
    fused_feature_pool::{PADDING_ITEM, pack_feature},
    normalize_wdl_target,
};
use crate::nnue::AB_NNUE_INPUT_SIZE;

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
    pub feature_items: Vec<u32>,
    pub value_wdl: Vec<f32>,
    pub values: Vec<f32>,
    pub rule_context: Vec<f32>,
    pub value_weights: Vec<f32>,
    pub value_phase_masks: Vec<f32>,
    pub value_source_phase_masks: Vec<f32>,
}

impl PackedBatch {
    pub(super) fn from_indices(samples: &[AbTrainingSample], batch: &[usize]) -> Self {
        let batch_size = batch.len();
        let max_features = batch
            .iter()
            .map(|&sample_index| samples[sample_index].features.len())
            .max()
            .unwrap_or(0)
            .max(1);
        let mut packed = Self {
            batch_size,
            max_features,
            feature_items: vec![PADDING_ITEM; batch_size * max_features],
            value_wdl: vec![0.0f32; batch_size * WDL_HEAD_SIZE],
            values: vec![0.0f32; batch_size],
            rule_context: vec![0.0f32; batch_size * RULE_CONTEXT_SIZE],
            value_weights: vec![1.0f32; batch_size],
            value_phase_masks: vec![0.0f32; batch_size * 3],
            value_source_phase_masks: vec![0.0f32; batch_size * 9],
        };

        for (row, &sample_index) in batch.iter().enumerate() {
            let sample = &samples[sample_index];
            packed.pack_features(row, sample);
            let wdl = normalize_wdl_target(sample.value_wdl);
            packed.value_wdl[row * WDL_HEAD_SIZE..(row + 1) * WDL_HEAD_SIZE].copy_from_slice(&wdl);
            packed.values[row] = sample.value.clamp(-1.0, 1.0);
            packed.rule_context[row * RULE_CONTEXT_SIZE..(row + 1) * RULE_CONTEXT_SIZE]
                .copy_from_slice(&sample.rule_context);
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

    fn pack_features(&mut self, row: usize, sample: &AbTrainingSample) {
        let (us_king_bucket, them_king_bucket) =
            canonical_general_buckets_from_features(&sample.features);
        let feature_base = row * self.max_features;
        for (feature_offset, &feature) in sample.features.iter().enumerate() {
            if feature >= AB_NNUE_INPUT_SIZE {
                continue;
            }
            let batch_feature_index = feature_base + feature_offset;
            self.feature_items[batch_feature_index] =
                pack_feature(feature, us_king_bucket, them_king_bucket);
        }
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
        samples: Arc<Vec<AbTrainingSample>>,
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

    use crate::ab::AbSampleMeta;

    use super::*;

    fn sample(index: usize) -> AbTrainingSample {
        AbTrainingSample {
            features: vec![index % AB_NNUE_INPUT_SIZE],
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            value_wdl: [1.0, 0.0, 0.0],
            root_search_wdl: [1.0, 0.0, 0.0],
            value: 2.0,
            side_sign: 1.0,
            value_weight: 1.0,
            search_nodes: 0,
            meta: AbSampleMeta::default(),
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
    fn packed_batch_clamps_value_targets() {
        let mut samples = vec![sample(0), sample(1)];
        samples[1].meta.start_source = crate::ab::AbStartSource::Midgame;
        samples[1].meta.ply = 130;
        let packed = PackedBatch::from_indices(&samples, &[0, 1]);
        assert_eq!(packed.batch_size, 2);
        assert_eq!(&packed.value_wdl[0..3], &[1.0, 0.0, 0.0]);
        assert_eq!(packed.values, vec![1.0, 1.0]);
        assert_eq!(packed.value_source_phase_masks[0], 1.0);
        assert_eq!(packed.value_source_phase_masks[9 + 8], 1.0);
        assert_eq!(packed.value_source_phase_masks.iter().sum::<f32>(), 2.0);
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
