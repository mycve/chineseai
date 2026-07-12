use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::io::{self, Cursor, Read};
use std::path::{Path, PathBuf};

use byteorder::{ByteOrder, LittleEndian, ReadBytesExt};
use lz4_flex::block::{compress_prepend_size, decompress_size_prepended};

use super::{
    AzSampleMeta, AzTrainingSample, DENSE_MOVE_SPACE, SplitMix64, WDL_HEAD_SIZE,
    normalize_wdl_target,
};

/// 经验池磁盘快照（与 `AzExperiencePool::save_snapshot_lz4` 对应）。
const REPLAY_MAGIC: &[u8] = b"AZRP";
/// 经验池快照内 `encode_az_training_sample` 布局版本（与旧版不兼容时递增）。
const REPLAY_FILE_VERSION: u32 = 26;
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
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "replay encode: move_indices/policy mismatch or too long",
        ));
    }
    replay_push_u32(out, sample.features.len() as u32);
    for &f in &sample.features {
        replay_push_u32(out, f as u32);
    }
    replay_push_u32(out, sample.move_indices.len() as u32);
    for &m in &sample.move_indices {
        replay_push_u32(out, m as u32);
    }
    for &p in &sample.policy {
        replay_push_f32(out, p);
    }
    for &value in &normalize_wdl_target(sample.value_wdl) {
        replay_push_f32(out, value);
    }
    replay_push_f32(out, sample.value);
    replay_push_f32(out, sample.side_sign);
    replay_push_f32(out, sample.moves_left);
    replay_push_f32(out, sample.policy_weight);
    replay_push_f32(out, sample.value_weight);
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
    replay_push_u32(out, u32::from(sample.meta.deblundered));
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
    pub window_updates: u32,
    pub recent_window_sample_fraction: f32,
}

#[derive(Clone, Debug, Default)]
pub struct AzReplaySampleBatch {
    pub samples: Vec<AzTrainingSample>,
    pub recent_samples: usize,
    pub full_window_samples: usize,
}

fn encode_replay_entry(out: &mut Vec<u8>, entry: &ReplayEntry) -> io::Result<()> {
    encode_az_training_sample(out, &entry.sample)
}

fn decode_az_training_sample<R: Read>(
    reader: &mut R,
    version: u32,
) -> io::Result<AzTrainingSample> {
    if version != REPLAY_FILE_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "replay decode: incompatible replay sample version",
        ));
    }
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
    let mut policy = Vec::with_capacity(nm as usize);
    for _ in 0..nm {
        policy.push(replay_read_f32(reader)?);
    }
    let mut value_wdl = [0.0f32; WDL_HEAD_SIZE];
    for value in &mut value_wdl {
        *value = replay_read_f32(reader)?;
    }
    value_wdl = normalize_wdl_target(value_wdl);
    let value = replay_read_f32(reader)?;
    let side_sign = replay_read_f32(reader)?;
    let moves_left = replay_read_f32(reader)?;
    let policy_weight = replay_read_f32(reader)?;
    let value_weight = replay_read_f32(reader)?;
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
        deblundered: replay_read_u32(reader)? != 0,
    };
    Ok(AzTrainingSample {
        features,
        move_indices,
        policy,
        value_wdl,
        value,
        side_sign,
        moves_left,
        policy_weight,
        value_weight,
        search_simulations,
        meta,
    })
}

fn decode_replay_entry<R: Read>(reader: &mut R, version: u32) -> io::Result<ReplayEntry> {
    let sample = decode_az_training_sample(reader, version)?;
    Ok(ReplayEntry { sample })
}

#[derive(Clone, Debug)]
pub struct AzExperiencePool {
    capacity: usize,
    chunks: VecDeque<ReplayChunk>,
    sample_count: usize,
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
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let mut index = (rng.next_u64() as usize) % self.sample_count;
            for chunk in &self.chunks {
                if index < chunk.len() {
                    out.push(chunk.entries[index].sample.clone());
                    break;
                }
                index -= chunk.len();
            }
        }
        out
    }

    pub fn sample_mixed_recent(
        &self,
        count: usize,
        recent_fraction: f32,
        recent_window_updates: u32,
        rng: &mut SplitMix64,
    ) -> AzReplaySampleBatch {
        if self.sample_count == 0 || count == 0 {
            return AzReplaySampleBatch::default();
        }
        let recent_indices = self.recent_flat_indices(recent_window_updates.max(1));
        if recent_indices.is_empty() {
            return AzReplaySampleBatch {
                samples: self.sample_uniform(count, rng),
                recent_samples: 0,
                full_window_samples: count,
            };
        }
        let recent_target = ((count as f32) * recent_fraction.clamp(0.0, 1.0)).round() as usize;
        let recent_target = recent_target.min(count);
        let mut samples = Vec::with_capacity(count);
        for _ in 0..recent_target {
            let flat = recent_indices[(rng.next_u64() as usize) % recent_indices.len()];
            samples.push(self.sample_by_flat_index(flat));
        }
        let full_count = count - recent_target;
        samples.extend(self.sample_uniform(full_count, rng));
        AzReplaySampleBatch {
            samples,
            recent_samples: recent_target,
            full_window_samples: full_count,
        }
    }

    fn recent_flat_indices(&self, recent_window_updates: u32) -> Vec<usize> {
        let newest = self
            .chunks
            .iter()
            .map(|chunk| chunk.generation_update)
            .max()
            .unwrap_or(0);
        let oldest_recent = newest.saturating_sub(recent_window_updates.saturating_sub(1));
        let mut out = Vec::new();
        let mut flat = 0usize;
        for chunk in &self.chunks {
            if chunk.generation_update >= oldest_recent {
                out.extend(flat..flat + chunk.len());
            }
            flat += chunk.len();
        }
        out
    }

    fn sample_by_flat_index(&self, mut index: usize) -> AzTrainingSample {
        for chunk in &self.chunks {
            if index < chunk.len() {
                return chunk.entries[index].sample.clone();
            }
            index -= chunk.len();
        }
        self.chunks
            .back()
            .and_then(|chunk| chunk.entries.last())
            .expect("non-empty replay pool")
            .sample
            .clone()
    }

    pub fn all_samples(&self) -> Vec<AzTrainingSample> {
        self.chunks
            .iter()
            .flat_map(|chunk| chunk.entries.iter())
            .map(|entry| entry.sample.clone())
            .collect()
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

    pub fn window_stats(&self, recent_window_updates: u32) -> AzReplayWindowStats {
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
        let mut by_update = BTreeMap::<u32, usize>::new();
        for chunk in &self.chunks {
            weighted_sum += chunk.generation_update as u64 * chunk.len() as u64;
            *by_update.entry(chunk.generation_update).or_default() += chunk.len();
        }
        let recent_oldest = newest.saturating_sub(recent_window_updates.max(1).saturating_sub(1));
        let recent_samples = by_update
            .iter()
            .filter_map(|(&update, &count)| (update >= recent_oldest).then_some(count))
            .sum::<usize>();
        AzReplayWindowStats {
            chunks: self.chunks.len(),
            samples: self.sample_count,
            oldest_generation_update: oldest,
            newest_generation_update: newest,
            avg_generation_update: weighted_sum as f32 / self.sample_count as f32,
            window_updates: newest.saturating_sub(oldest).saturating_add(1),
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

    fn decode_replay_payload(data: &[u8], capacity: usize, version: u32) -> io::Result<Self> {
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
                let entry = decode_replay_entry(&mut reader, version)?;
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
        Self::decode_replay_payload(&inner, capacity, ver)
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
