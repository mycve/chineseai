use crate::infra::version::REPLAY_FILE_VERSION;

use std::collections::VecDeque;
use std::fs;
use std::io::{self, Cursor, Read};
use std::path::{Path, PathBuf};

use byteorder::{ByteOrder, LittleEndian, ReadBytesExt};
use lz4_flex::block::{compress_prepend_size, decompress_size_prepended};

use super::{
    AzSampleMeta, AzStartSource, AzTrainingSample, DENSE_MOVE_SPACE, SplitMix64, WDL_HEAD_SIZE,
    normalize_wdl_target,
};

/// 经验池磁盘快照（与 `AzExperiencePool::save_snapshot_lz4` 对应）。
const REPLAY_MAGIC: &[u8] = b"AZRP";
/// 经验池快照内 `encode_az_training_sample` 布局版本（与旧版不兼容时递增）。
// v32 开始使用干净主搜索与独立战术教师；旧访问目标语义不同，禁止混入。
/// 分块快照解压后体积极限（防恶意或损坏文件占满内存）。
const REPLAY_MAX_DECOMPRESSED_BYTES: usize = 16usize << 30;
const REPLAY_CHUNKED_MARKER: &[u8] = b"CHNK";
#[cfg(not(test))]
const REPLAY_COMPRESS_CHUNK_BYTES: usize = 64 * 1024 * 1024;
#[cfg(test)]
const REPLAY_COMPRESS_CHUNK_BYTES: usize = 512;
const REPLAY_MAX_FEATURES_PER_SAMPLE: u32 = 16_384;
const REPLAY_MAX_MOVES_PER_SAMPLE: u32 = (DENSE_MOVE_SPACE as u32).saturating_add(128);
fn replay_push_u32(out: &mut Vec<u8>, v: u32) {
    let mut buf = [0u8; 4];
    LittleEndian::write_u32(&mut buf, v);
    out.extend_from_slice(&buf);
}

fn replay_push_u64(out: &mut Vec<u8>, v: u64) {
    let mut buf = [0u8; 8];
    LittleEndian::write_u64(&mut buf, v);
    out.extend_from_slice(&buf);
}

fn replay_push_f32(out: &mut Vec<u8>, v: f32) {
    let mut buf = [0u8; 4];
    LittleEndian::write_f32(&mut buf, v);
    out.extend_from_slice(&buf);
}

fn replay_read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    reader.read_u32::<LittleEndian>()
}

fn replay_read_u64<R: Read>(reader: &mut R) -> io::Result<u64> {
    reader.read_u64::<LittleEndian>()
}

fn replay_read_f32<R: Read>(reader: &mut R) -> io::Result<f32> {
    reader.read_f32::<LittleEndian>()
}

fn encode_az_training_sample(out: &mut Vec<u8>, sample: &AzTrainingSample) -> io::Result<()> {
    if sample.features.len() > REPLAY_MAX_FEATURES_PER_SAMPLE as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "replay encode: too many features",
        ));
    }
    if sample.move_indices.len() > REPLAY_MAX_MOVES_PER_SAMPLE as usize
        || sample.policy.len() != sample.move_indices.len()
        || (!sample.repetition_flags.is_empty()
            && sample.repetition_flags.len() != sample.move_indices.len())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "replay encode: move_indices/policy mismatch or too long",
        ));
    }
    if !sample.moves_left.is_finite()
        || sample.moves_left < 0.0
        || !sample.moves_left_weight.is_finite()
        || sample.moves_left_weight < 0.0
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid moves-left target",
        ));
    }
    replay_push_u32(out, sample.features.len() as u32);
    for &f in &sample.features {
        replay_push_u32(out, f as u32);
    }
    for &value in &sample.rule_context {
        replay_push_f32(out, value);
    }
    replay_push_u32(out, sample.move_indices.len() as u32);
    for &m in &sample.move_indices {
        replay_push_u32(out, m as u32);
    }
    if sample.repetition_flags.is_empty() {
        out.resize(out.len() + sample.move_indices.len(), 0);
    } else {
        out.extend_from_slice(&sample.repetition_flags);
    }
    for &p in &sample.policy {
        replay_push_f32(out, p);
    }
    for &value in &normalize_wdl_target(sample.value_wdl) {
        replay_push_f32(out, value);
    }
    for &value in &normalize_wdl_target(sample.root_search_wdl) {
        replay_push_f32(out, value);
    }
    replay_push_f32(out, sample.value);
    replay_push_f32(out, sample.side_sign);
    replay_push_f32(out, sample.policy_weight);
    replay_push_f32(out, sample.value_weight);
    replay_push_f32(out, sample.moves_left);
    replay_push_f32(out, sample.moves_left_weight);
    replay_push_u32(out, sample.search_simulations);
    replay_push_u32(out, sample.meta.generation_update);
    replay_push_u64(out, sample.meta.game_id);
    replay_push_u32(out, sample.meta.ply as u32);
    replay_push_f32(out, sample.meta.root_q);
    replay_push_f32(out, sample.meta.best_q);
    replay_push_f32(out, sample.meta.played_q);
    replay_push_u32(out, sample.meta.best_visits);
    replay_push_u32(out, sample.meta.played_visits);
    replay_push_u32(out, sample.meta.best_index as u32);
    replay_push_u32(out, sample.meta.played_index as u32);
    out.push(sample.meta.start_source as u8);
    Ok(())
}

#[derive(Clone, Debug)]
struct ReplayEntry {
    sample: AzTrainingSample,
}

#[derive(Clone, Debug)]
struct ReplayChunk {
    generation_update: u32,
    entries: Vec<ReplayEntry>,
}

impl ReplayChunk {
    fn new(samples: Vec<AzTrainingSample>) -> Self {
        let generation_update = samples
            .first()
            .map(|sample| sample.meta.generation_update)
            .unwrap_or(0);
        let entries = samples
            .into_iter()
            .map(|sample| ReplayEntry { sample })
            .collect();
        Self {
            generation_update,
            entries,
        }
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AzReplayWindowStats {
    pub chunks: usize,
    pub samples: usize,
    pub oldest_generation_update: u32,
    pub newest_generation_update: u32,
    pub avg_generation_update: f32,
    pub window_games: u32,
    pub recent_window_sample_fraction: f32,
}

#[derive(Clone, Debug, Default)]
pub struct AzReplaySampleBatch {
    pub samples: Vec<AzTrainingSample>,
    pub recent_samples: usize,
    pub actual_recent_samples: usize,
    pub full_window_samples: usize,
    pub source_samples: [usize; AzStartSource::COUNT],
}

fn encode_replay_entry(out: &mut Vec<u8>, entry: &ReplayEntry) -> io::Result<()> {
    encode_az_training_sample(out, &entry.sample)
}

fn decode_az_training_sample<R: Read>(reader: &mut R) -> io::Result<AzTrainingSample> {
    let nf = replay_read_u32(reader)?;
    if nf > REPLAY_MAX_FEATURES_PER_SAMPLE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "replay decode: feature count out of range",
        ));
    }
    let mut features = Vec::with_capacity(nf as usize);
    for _ in 0..nf {
        features.push(replay_read_u32(reader)? as usize);
    }
    let mut rule_context = [0.0; super::RULE_CONTEXT_SIZE];
    for value in &mut rule_context {
        *value = replay_read_f32(reader)?;
    }
    let nm = replay_read_u32(reader)?;
    if nm > REPLAY_MAX_MOVES_PER_SAMPLE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "replay decode: move count out of range",
        ));
    }
    let mut move_indices = Vec::with_capacity(nm as usize);
    for _ in 0..nm {
        move_indices.push(replay_read_u32(reader)? as usize);
    }
    let mut repetition_flags = vec![0; nm as usize];
    reader.read_exact(&mut repetition_flags)?;
    if repetition_flags.iter().any(|&flag| flag > 1) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "replay decode: invalid repetition flag",
        ));
    }
    let mut policy = Vec::with_capacity(nm as usize);
    for _ in 0..nm {
        policy.push(replay_read_f32(reader)?);
    }
    let mut value_wdl = [0.0f32; WDL_HEAD_SIZE];
    for value in &mut value_wdl {
        *value = replay_read_f32(reader)?;
    }
    value_wdl = normalize_wdl_target(value_wdl);
    let mut root_search_wdl = [0.0f32; WDL_HEAD_SIZE];
    for value in &mut root_search_wdl {
        *value = replay_read_f32(reader)?;
    }
    root_search_wdl = normalize_wdl_target(root_search_wdl);
    let value = replay_read_f32(reader)?;
    let side_sign = replay_read_f32(reader)?;
    let policy_weight = replay_read_f32(reader)?;
    let value_weight = replay_read_f32(reader)?;
    let moves_left = replay_read_f32(reader)?;
    let moves_left_weight = replay_read_f32(reader)?;
    if !moves_left.is_finite()
        || moves_left < 0.0
        || !moves_left_weight.is_finite()
        || moves_left_weight < 0.0
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid moves-left target",
        ));
    }
    let search_simulations = replay_read_u32(reader)?;
    let meta = AzSampleMeta {
        generation_update: replay_read_u32(reader)?,
        game_id: replay_read_u64(reader)?,
        ply: replay_read_u32(reader)?.min(u16::MAX as u32) as u16,
        root_q: replay_read_f32(reader)?,
        best_q: replay_read_f32(reader)?,
        played_q: replay_read_f32(reader)?,
        best_visits: replay_read_u32(reader)?,
        played_visits: replay_read_u32(reader)?,
        best_index: replay_read_u32(reader)?.min(u16::MAX as u32) as u16,
        played_index: replay_read_u32(reader)?.min(u16::MAX as u32) as u16,
        start_source: AzStartSource::from_u8(reader.read_u8()?).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "replay decode: invalid start source",
            )
        })?,
    };
    Ok(AzTrainingSample {
        features,
        rule_context,
        move_indices,
        repetition_flags,
        policy,
        value_wdl,
        root_search_wdl,
        value,
        side_sign,
        policy_weight,
        value_weight,
        moves_left,
        moves_left_weight,
        search_simulations,
        meta,
    })
}

fn decode_replay_entry<R: Read>(reader: &mut R) -> io::Result<ReplayEntry> {
    let sample = decode_az_training_sample(reader)?;
    Ok(ReplayEntry { sample })
}

#[derive(Clone, Debug)]
pub struct AzExperiencePool {
    capacity: usize,
    chunks: VecDeque<ReplayChunk>,
    sample_count: usize,
}

/// Px0 train.py SKIP=32 与 shufflebuffer.py insert_or_replace。
/// 降采样发生在每次读取对局时；不永久删除经验池里的其余局面。
pub struct Px0ReplaySampler {
    capacity: usize,
    buffer: Vec<AzTrainingSample>,
    pending: VecDeque<AzTrainingSample>,
    partition: Option<(u64, bool)>,
}

impl Px0ReplaySampler {
    pub fn new(shuffle_size: usize) -> Self {
        Self {
            capacity: shuffle_size.max(1),
            buffer: Vec::new(),
            pending: VecDeque::new(),
            partition: None,
        }
    }

    /// 整盘固定分组，避免滚动经验池或续训把测试对局重新送入训练。
    pub fn partitioned(shuffle_size: usize, seed: u64, test: bool) -> Self {
        let mut sampler = Self::new(shuffle_size);
        sampler.partition = Some((seed, test));
        sampler
    }

    pub fn sample(
        &mut self,
        pool: &AzExperiencePool,
        count: usize,
        recent_games: u32,
        rng: &mut SplitMix64,
    ) -> AzReplaySampleBatch {
        if pool.sample_count == 0
            || count == 0
            || !pool.chunks.iter().any(|chunk| self.includes(chunk))
        {
            return AzReplaySampleBatch::default();
        }
        let recent_ids: std::collections::HashSet<_> = pool
            .chunks
            .iter()
            .rev()
            .take(recent_games as usize)
            .flat_map(|chunk| chunk.entries.first())
            .map(|entry| entry.sample.meta.game_id)
            .collect();
        let mut batch = AzReplaySampleBatch::default();
        batch.samples.reserve(count);
        while batch.samples.len() < count {
            if self.pending.is_empty() {
                let mut order: Vec<_> = pool
                    .chunks
                    .iter()
                    .enumerate()
                    .filter_map(|(i, chunk)| self.includes(chunk).then_some(i))
                    .collect();
                for i in (1..order.len()).rev() {
                    let j = rng.next_u64() as usize % (i + 1);
                    order.swap(i, j);
                }
                for index in order {
                    for entry in &pool.chunks[index].entries {
                        if rng.next_u64() % 32 == 0 {
                            self.pending.push_back(entry.sample.clone());
                        }
                    }
                }
            }
            let Some(mut item) = self.pending.pop_front() else {
                continue;
            };
            if !self.buffer.is_empty() {
                let index = rng.next_u64() as usize % self.buffer.len();
                item = std::mem::replace(&mut self.buffer[index], item);
            }
            if self.buffer.len() < self.capacity {
                self.buffer.push(item);
                continue;
            }
            batch.actual_recent_samples += usize::from(recent_ids.contains(&item.meta.game_id));
            batch.source_samples[item.meta.start_source.index()] += 1;
            batch.samples.push(item);
        }
        batch.full_window_samples = count;
        batch
    }

    fn includes(&self, chunk: &ReplayChunk) -> bool {
        self.partition
            .is_none_or(|(seed, test)| chunk_is_test(chunk, seed) == test)
    }
}

fn chunk_is_test(chunk: &ReplayChunk, seed: u64) -> bool {
    let game_id = chunk.entries[0].sample.meta.game_id;
    SplitMix64::new(game_id ^ seed ^ 0xA076_1D64_78BD_642F).next_u64() % 10 == 0
}

impl AzExperiencePool {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            chunks: VecDeque::new(),
            sample_count: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    pub fn partition_chunks(&self, seed: u64) -> (usize, usize) {
        let test = self
            .chunks
            .iter()
            .filter(|chunk| chunk_is_test(chunk, seed))
            .count();
        (self.chunks.len() - test, test)
    }

    pub fn add_samples<I>(&mut self, samples: I)
    where
        I: IntoIterator<Item = AzTrainingSample>,
    {
        self.add_chunk(samples.into_iter().collect());
    }

    fn add_chunk(&mut self, samples: Vec<AzTrainingSample>) {
        if self.capacity == 0 {
            return;
        }
        if samples.is_empty() {
            return;
        }
        let mut chunk = ReplayChunk::new(samples);
        if chunk.len() > self.capacity {
            let start = chunk.len() - self.capacity;
            chunk.entries.drain(0..start);
        }
        self.sample_count += chunk.len();
        self.chunks.push_back(chunk);
        self.prune_to_capacity();
    }

    fn prune_to_capacity(&mut self) {
        while self.sample_count > self.capacity {
            let Some(chunk) = self.chunks.pop_front() else {
                self.sample_count = 0;
                return;
            };
            self.sample_count = self.sample_count.saturating_sub(chunk.len());
        }
    }

    pub fn add_games(&mut self, games: Vec<Vec<AzTrainingSample>>) {
        for game in games {
            self.add_chunk(game);
        }
    }

    pub fn sample_uniform(&self, count: usize, rng: &mut SplitMix64) -> Vec<AzTrainingSample> {
        if self.sample_count == 0 || count == 0 {
            return Vec::new();
        }
        let chunk_ends = self.chunk_ends();
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let index = (rng.next_u64() as usize) % self.sample_count;
            out.push(self.sample_by_flat_index(index, &chunk_ends));
        }
        out
    }

    pub fn sample_mixed_recent(
        &self,
        count: usize,
        recent_fraction: f32,
        recent_games: u32,
        rng: &mut SplitMix64,
    ) -> AzReplaySampleBatch {
        if self.sample_count == 0 || count == 0 {
            return AzReplaySampleBatch::default();
        }
        let Some(recent_start) = self.recent_start_flat(recent_games.max(1)) else {
            return AzReplaySampleBatch {
                samples: self.sample_uniform(count, rng),
                recent_samples: 0,
                actual_recent_samples: 0,
                full_window_samples: count,
                source_samples: [0; AzStartSource::COUNT],
            };
        };
        let recent_count = self.sample_count - recent_start;
        let recent_target = ((count as f32) * recent_fraction.clamp(0.0, 1.0)).round() as usize;
        let recent_target = recent_target.min(count);
        let chunk_ends = self.chunk_ends();
        let mut samples = Vec::with_capacity(count);
        for _ in 0..recent_target {
            let flat = recent_start + (rng.next_u64() as usize) % recent_count;
            samples.push(self.sample_by_flat_index(flat, &chunk_ends));
        }
        let full_count = count - recent_target;
        let mut actual_recent_samples = recent_target;
        for _ in 0..full_count {
            let flat = (rng.next_u64() as usize) % self.sample_count;
            actual_recent_samples += usize::from(flat >= recent_start);
            samples.push(self.sample_by_flat_index(flat, &chunk_ends));
        }
        AzReplaySampleBatch {
            samples,
            recent_samples: recent_target,
            actual_recent_samples,
            full_window_samples: full_count,
            source_samples: [0; AzStartSource::COUNT],
        }
    }

    fn recent_start_flat(&self, recent_games: u32) -> Option<usize> {
        if self.chunks.is_empty() {
            return None;
        }
        let old_games = self
            .chunks
            .len()
            .saturating_sub((recent_games as usize).max(1));
        Some(
            self.chunks
                .iter()
                .take(old_games)
                .map(ReplayChunk::len)
                .sum(),
        )
    }

    fn chunk_ends(&self) -> Vec<usize> {
        let mut total = 0usize;
        self.chunks
            .iter()
            .map(|chunk| {
                total += chunk.len();
                total
            })
            .collect()
    }

    fn sample_by_flat_index(&self, index: usize, chunk_ends: &[usize]) -> AzTrainingSample {
        debug_assert!(index < self.sample_count);
        let chunk_index = chunk_ends.partition_point(|&end| end <= index);
        let chunk_start = chunk_index
            .checked_sub(1)
            .map_or(0, |previous| chunk_ends[previous]);
        self.chunks[chunk_index].entries[index - chunk_start]
            .sample
            .clone()
    }

    pub fn all_samples(&self) -> Vec<AzTrainingSample> {
        self.iter_samples().cloned().collect()
    }

    pub fn iter_samples(&self) -> impl Iterator<Item = &AzTrainingSample> {
        self.chunks
            .iter()
            .flat_map(|chunk| chunk.entries.iter().map(|entry| &entry.sample))
    }

    pub fn all_sample_groups(&self) -> Vec<Vec<AzTrainingSample>> {
        self.chunks
            .iter()
            .map(|chunk| {
                chunk
                    .entries
                    .iter()
                    .map(|entry| entry.sample.clone())
                    .collect()
            })
            .collect()
    }

    pub fn window_stats(&self, recent_games: u32) -> AzReplayWindowStats {
        if self.sample_count == 0 {
            return AzReplayWindowStats::default();
        }
        let oldest = self
            .chunks
            .iter()
            .map(|chunk| chunk.generation_update)
            .min()
            .unwrap_or(0);
        let newest = self.max_generation_update();
        let mut weighted_sum = 0u64;
        for chunk in &self.chunks {
            weighted_sum += chunk.generation_update as u64 * chunk.len() as u64;
        }
        let recent_samples = self
            .chunks
            .iter()
            .rev()
            .take(recent_games.max(1) as usize)
            .map(ReplayChunk::len)
            .sum::<usize>();
        AzReplayWindowStats {
            chunks: self.chunks.len(),
            samples: self.sample_count,
            oldest_generation_update: oldest,
            newest_generation_update: newest,
            avg_generation_update: weighted_sum as f32 / self.sample_count as f32,
            window_games: self.chunks.len().min(u32::MAX as usize) as u32,
            recent_window_sample_fraction: recent_samples as f32 / self.sample_count as f32,
        }
    }

    pub fn max_generation_update(&self) -> u32 {
        self.chunks
            .iter()
            .map(|chunk| chunk.generation_update)
            .max()
            .unwrap_or(0)
    }

    fn encode_replay_payload(&self) -> io::Result<Vec<u8>> {
        let mut out = Vec::new();
        replay_push_u64(&mut out, self.capacity as u64);
        replay_push_u64(&mut out, self.chunks.len() as u64);
        for chunk in &self.chunks {
            replay_push_u32(&mut out, chunk.generation_update);
            replay_push_u64(&mut out, chunk.entries.len() as u64);
            for entry in &chunk.entries {
                encode_replay_entry(&mut out, entry)?;
            }
        }
        Ok(out)
    }

    fn decode_replay_payload(data: &[u8], capacity: usize) -> io::Result<Self> {
        let mut reader = Cursor::new(data);
        let _stored_capacity = replay_read_u64(&mut reader)? as usize;
        let n_chunks = replay_read_u64(&mut reader)? as usize;
        if n_chunks > 10_000_000 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "replay decode: absurd chunk count",
            ));
        }

        let mut pool = Self::new(capacity);
        for _ in 0..n_chunks {
            let generation_update = replay_read_u32(&mut reader)?;
            let n_entries = replay_read_u64(&mut reader)? as usize;
            if n_entries > 10_000_000 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "replay decode: absurd chunk size",
                ));
            }
            let mut entries = Vec::with_capacity(n_entries.min(capacity));
            for _ in 0..n_entries {
                let entry = decode_replay_entry(&mut reader)?;
                if capacity > 0 {
                    entries.push(entry);
                }
            }
            if capacity > 0 && !entries.is_empty() {
                pool.sample_count += entries.len();
                pool.chunks.push_back(ReplayChunk {
                    generation_update,
                    entries,
                });
                pool.prune_to_capacity();
            }
        }

        Ok(pool)
    }

    pub fn save_snapshot_lz4(&self, path: &Path) -> io::Result<()> {
        if self.capacity == 0 || self.sample_count == 0 {
            let _ = fs::remove_file(path);
            return Ok(());
        }
        let inner = self.encode_replay_payload()?;
        let mut file_blob = Vec::new();
        file_blob.extend_from_slice(REPLAY_MAGIC);
        replay_push_u32(&mut file_blob, REPLAY_FILE_VERSION);
        file_blob.extend_from_slice(REPLAY_CHUNKED_MARKER);
        replay_push_u64(&mut file_blob, inner.len() as u64);
        let chunk_count = inner.len().div_ceil(REPLAY_COMPRESS_CHUNK_BYTES);
        replay_push_u64(&mut file_blob, chunk_count as u64);
        for chunk in inner.chunks(REPLAY_COMPRESS_CHUNK_BYTES) {
            let compressed = compress_prepend_size(chunk);
            replay_push_u32(&mut file_blob, chunk.len() as u32);
            replay_push_u64(&mut file_blob, compressed.len() as u64);
            file_blob.extend_from_slice(&compressed);
        }
        let tmp = PathBuf::from(format!("{}.tmp", path.display()));
        fs::write(&tmp, &file_blob)?;
        if path.exists() {
            fs::remove_file(path)?;
        }
        fs::rename(&tmp, path)?;
        Ok(())
    }

    pub fn load_snapshot_lz4(path: &Path, capacity: usize) -> io::Result<Self> {
        let file_blob = fs::read(path)?;
        if file_blob.len() < 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "replay file too small",
            ));
        }
        if &file_blob[0..4] != REPLAY_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "replay bad magic",
            ));
        }
        let ver = LittleEndian::read_u32(&file_blob[4..8]);
        if ver != REPLAY_FILE_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("replay unsupported version {ver} (expected v{REPLAY_FILE_VERSION})"),
            ));
        }
        if file_blob.len() < 12 || &file_blob[8..12] != REPLAY_CHUNKED_MARKER {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "replay missing chunked snapshot marker",
            ));
        }
        let inner = Self::decompress_chunked_snapshot(&file_blob[12..])?;
        Self::decode_replay_payload(&inner, capacity)
    }

    fn decompress_chunked_snapshot(data: &[u8]) -> io::Result<Vec<u8>> {
        let mut reader = Cursor::new(data);
        let total_len = replay_read_u64(&mut reader)? as usize;
        if total_len > REPLAY_MAX_DECOMPRESSED_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "replay chunked snapshot: decompressed size over cap",
            ));
        }
        let chunk_count = replay_read_u64(&mut reader)? as usize;
        if chunk_count > 1_000_000 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "replay chunked snapshot: absurd chunk count",
            ));
        }
        let mut inner = Vec::with_capacity(total_len);
        for _ in 0..chunk_count {
            let raw_len = replay_read_u32(&mut reader)? as usize;
            let compressed_len = replay_read_u64(&mut reader)? as usize;
            if raw_len > REPLAY_COMPRESS_CHUNK_BYTES || compressed_len > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "replay chunked snapshot: invalid chunk size",
                ));
            }
            let start = reader.position() as usize;
            let end = start.checked_add(compressed_len).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "replay chunked snapshot: chunk size overflow",
                )
            })?;
            if end > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "replay chunked snapshot: truncated chunk",
                ));
            }
            let chunk = decompress_size_prepended(&data[start..end]).map_err(|err| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("replay chunked lz4 decompress: {err:?}"),
                )
            })?;
            if chunk.len() != raw_len {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "replay chunked snapshot: raw chunk size mismatch",
                ));
            }
            inner.extend_from_slice(&chunk);
            reader.set_position(end as u64);
        }
        if inner.len() != total_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "replay chunked snapshot: total size mismatch",
            ));
        }
        Ok(inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(source: AzStartSource, generation: u32, id: u64) -> AzTrainingSample {
        AzTrainingSample {
            repetition_flags: vec![0],
            features: vec![0],
            rule_context: [0.0; super::super::RULE_CONTEXT_SIZE],
            move_indices: vec![0],
            policy: vec![1.0],
            value_wdl: [0.0, 1.0, 0.0],
            root_search_wdl: [0.0, 1.0, 0.0],
            value: 0.0,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            moves_left: 0.0,
            moves_left_weight: 0.0,
            search_simulations: 400,
            meta: AzSampleMeta {
                generation_update: generation,
                game_id: id,
                start_source: source,
                ..AzSampleMeta::default()
            },
        }
    }

    #[test]
    fn px0_sampler_keeps_pool_and_natural_phase_distribution_across_batches() {
        let mut pool = AzExperiencePool::new(1000);
        pool.add_games(
            (0..100)
                .map(|game| {
                    (0..10)
                        .map(|_| {
                            let mut item = sample(AzStartSource::OpeningBook, 1, game);
                            item.meta.ply = if game < 90 { 15 } else { 150 };
                            item
                        })
                        .collect()
                })
                .collect(),
        );
        let mut sampler = Px0ReplaySampler::new(128);
        let mut rng = SplitMix64::new(42);
        let first = sampler.sample(&pool, 2500, 10, &mut rng);
        let second = sampler.sample(&pool, 2500, 10, &mut rng);
        let early = first
            .samples
            .iter()
            .chain(&second.samples)
            .filter(|s| s.meta.ply == 15)
            .count();
        assert!(
            early > 4000 && early < 4900,
            "unexpected phase bias: {early}/5000"
        );
        assert_eq!(sampler.buffer.len(), 128);
        assert_eq!(pool.sample_count(), 1000);
        assert_eq!(first.recent_samples, 0);
        assert_eq!(first.full_window_samples, 2500);
        assert!(first.actual_recent_samples < 500);
        assert_eq!(second.samples.len(), 2500);
    }

    #[test]
    fn replay_rejects_old_format() {
        let mut pool = AzExperiencePool::new(10);
        pool.add_games(vec![vec![sample(AzStartSource::OpeningBook, 7, 11)]]);
        let path = std::env::temp_dir().join(format!(
            "chineseai-strict-replay-{}.lz4",
            std::process::id()
        ));
        pool.save_snapshot_lz4(&path).unwrap();
        let current = fs::read(&path).unwrap();
        for version in [40u32, 41] {
            let mut old = current.clone();
            old[4..8].copy_from_slice(&version.to_le_bytes());
            fs::write(&path, &old).unwrap();
            assert!(AzExperiencePool::load_snapshot_lz4(&path, 10).is_err());
        }
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn replay_roundtrip_preserves_start_source() {
        let mut encoded = Vec::new();
        let mut original = sample(AzStartSource::OpeningBook, 7, 11);
        original.root_search_wdl = [0.6, 0.3, 0.1];
        original.moves_left = 37.0;
        original.moves_left_weight = 1.0;
        original.repetition_flags[0] = 1;
        encode_az_training_sample(&mut encoded, &original).unwrap();
        let decoded = decode_az_training_sample(&mut Cursor::new(encoded)).unwrap();
        assert_eq!(decoded.meta.start_source, AzStartSource::OpeningBook);
        assert_eq!(decoded.meta.generation_update, 7);
        assert_eq!(decoded.meta.game_id, 11);
        assert_eq!(decoded.root_search_wdl, original.root_search_wdl);
        assert_eq!(decoded.moves_left, 37.0);
        assert_eq!(decoded.moves_left_weight, 1.0);
        assert_eq!(decoded.repetition_flags, original.repetition_flags);
    }

    #[test]
    fn whole_game_holdout_is_disjoint_and_stable_after_decode() {
        let mut pool = AzExperiencePool::new(20_000);
        pool.add_games(
            (0..1000)
                .map(|id| {
                    (0..10)
                        .map(|_| sample(AzStartSource::OpeningBook, 1, id))
                        .collect()
                })
                .collect(),
        );
        let (train_chunks, test_chunks) = pool.partition_chunks(77);
        assert_eq!(train_chunks + test_chunks, 1000);
        assert!(
            (70..130).contains(&test_chunks),
            "test chunks={test_chunks}"
        );
        let mut train = Px0ReplaySampler::partitioned(64, 77, false);
        let mut test = Px0ReplaySampler::partitioned(6, 77, true);
        let train = train.sample(&pool, 1000, 0, &mut SplitMix64::new(1));
        let test = test.sample(&pool, 1000, 0, &mut SplitMix64::new(2));
        let train_ids: std::collections::HashSet<_> =
            train.samples.iter().map(|s| s.meta.game_id).collect();
        assert!(
            test.samples
                .iter()
                .all(|s| !train_ids.contains(&s.meta.game_id))
        );
        for chunk in &pool.chunks {
            let mut encoded = Vec::new();
            encode_replay_entry(&mut encoded, &chunk.entries[0]).unwrap();
            let decoded = decode_replay_entry(&mut Cursor::new(encoded)).unwrap();
            let restored = ReplayChunk::new(vec![decoded.sample]);
            assert_eq!(chunk_is_test(chunk, 77), chunk_is_test(&restored, 77));
        }
        let empty = AzExperiencePool::new(10);
        assert!(
            Px0ReplaySampler::partitioned(6, 77, true)
                .sample(&empty, 8, 0, &mut SplitMix64::new(3))
                .samples
                .is_empty()
        );
    }
}
