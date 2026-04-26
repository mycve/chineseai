use std::collections::VecDeque;
use std::fs;
use std::io::{self, Cursor, Read};
use std::path::{Path, PathBuf};

use lz4_flex::block::{compress_prepend_size, decompress_size_prepended};

use super::{AzTrainingSample, BOARD_HISTORY_SIZE, DENSE_MOVE_SPACE, SplitMix64};

/// 经验池磁盘快照（与 `AzExperiencePool::save_snapshot_lz4` 对应）。
const REPLAY_MAGIC: &[u8] = b"AZRP";
/// 经验池快照内 `encode_az_training_sample` 布局版本（与旧版不兼容时递增）。
const REPLAY_FILE_VERSION: u32 = 5;
/// 解压后体积极限（防恶意或损坏文件占满内存）。
const REPLAY_MAX_DECOMPRESSED_BYTES: usize = 2usize << 30;
const REPLAY_MAX_FEATURES_PER_SAMPLE: u32 = 16_384;
const REPLAY_MAX_MOVES_PER_SAMPLE: u32 = (DENSE_MOVE_SPACE as u32).saturating_add(128);

fn replay_push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn replay_push_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn replay_push_f32(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&v.to_bits().to_le_bytes());
}

fn replay_read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn replay_read_u64<R: Read>(reader: &mut R) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn replay_read_f32<R: Read>(reader: &mut R) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_bits(u32::from_le_bytes(buf)))
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
    if sample.board.len() != BOARD_HISTORY_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "replay encode: invalid board history length",
        ));
    }
    out.extend_from_slice(&sample.board);
    replay_push_u32(out, sample.move_indices.len() as u32);
    for &m in &sample.move_indices {
        replay_push_u32(out, m as u32);
    }
    for &p in &sample.policy {
        replay_push_f32(out, p);
    }
    replay_push_f32(out, sample.value);
    replay_push_f32(out, sample.side_sign);
    Ok(())
}

#[derive(Clone, Debug)]
struct ReplayEntry {
    sample: AzTrainingSample,
    train_count: u32,
}

fn encode_replay_entry(out: &mut Vec<u8>, entry: &ReplayEntry) -> io::Result<()> {
    encode_az_training_sample(out, &entry.sample)?;
    replay_push_u32(out, entry.train_count);
    Ok(())
}

fn decode_az_training_sample<R: Read>(
    reader: &mut R,
    version: u32,
) -> io::Result<AzTrainingSample> {
    if version < REPLAY_FILE_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "replay decode: incompatible board history version",
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
    let mut board = vec![0u8; BOARD_HISTORY_SIZE];
    reader.read_exact(&mut board)?;
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
    let value = replay_read_f32(reader)?;
    let side_sign = replay_read_f32(reader)?;
    Ok(AzTrainingSample {
        features,
        board,
        move_indices,
        policy,
        value,
        side_sign,
    })
}

fn decode_replay_entry<R: Read>(reader: &mut R, version: u32) -> io::Result<ReplayEntry> {
    let sample = decode_az_training_sample(reader, version)?;
    let train_count = if version >= 3 {
        replay_read_u32(reader)?
    } else {
        0
    };
    Ok(ReplayEntry {
        sample,
        train_count,
    })
}

#[derive(Clone, Debug)]
pub struct AzExperiencePool {
    game_capacity: usize,
    games: VecDeque<Vec<ReplayEntry>>,
    samples: usize,
    train_count_sum: u64,
}

impl AzExperiencePool {
    pub fn new(game_capacity: usize) -> Self {
        Self {
            game_capacity,
            games: VecDeque::new(),
            samples: 0,
            train_count_sum: 0,
        }
    }

    pub fn game_count(&self) -> usize {
        self.games.len()
    }

    pub fn sample_count(&self) -> usize {
        self.samples
    }

    pub fn average_train_count(&self) -> f32 {
        if self.samples == 0 {
            0.0
        } else {
            self.train_count_sum as f32 / self.samples as f32
        }
    }

    pub fn add_games(&mut self, games: Vec<Vec<AzTrainingSample>>) {
        if self.game_capacity == 0 {
            return;
        }
        for game in games.into_iter().filter(|game| !game.is_empty()) {
            self.samples += game.len();
            self.games.push_back(
                game.into_iter()
                    .map(|sample| ReplayEntry {
                        sample,
                        train_count: 0,
                    })
                    .collect(),
            );
            while self.games.len() > self.game_capacity {
                if let Some(removed) = self.games.pop_front() {
                    self.samples = self.samples.saturating_sub(removed.len());
                    self.train_count_sum = self
                        .train_count_sum
                        .saturating_sub(removed.iter().map(|entry| entry.train_count as u64).sum());
                }
            }
        }
    }

    pub fn sample_uniform_games_marked(
        &mut self,
        count: usize,
        max_train_count: u32,
        rng: &mut SplitMix64,
    ) -> Vec<AzTrainingSample> {
        self.sample_uniform_games_marked_excluding_newest(count, max_train_count, 0, rng)
    }

    pub fn sample_uniform_games_marked_excluding_newest(
        &mut self,
        count: usize,
        max_train_count: u32,
        exclude_newest_games: usize,
        rng: &mut SplitMix64,
    ) -> Vec<AzTrainingSample> {
        let eligible_games = self.games.len().saturating_sub(exclude_newest_games);
        if eligible_games == 0 || count == 0 {
            return Vec::new();
        }
        let mut samples = Vec::with_capacity(count);
        for _ in 0..count {
            let game_index = (rng.next_u64() as usize) % eligible_games;
            let game = &mut self.games[game_index];
            if game.is_empty() {
                continue;
            }
            let sample_index = (rng.next_u64() as usize) % game.len();
            let entry = &mut game[sample_index];
            entry.train_count = entry.train_count.saturating_add(1);
            self.train_count_sum = self.train_count_sum.saturating_add(1);
            samples.push(entry.sample.clone());
        }
        if max_train_count > 0 {
            self.prune_overtrained(max_train_count);
        }
        samples
    }

    pub fn sample_uniform_games(
        &self,
        count: usize,
        rng: &mut SplitMix64,
    ) -> Vec<AzTrainingSample> {
        if self.games.is_empty() || count == 0 {
            return Vec::new();
        }
        let mut samples = Vec::with_capacity(count);
        for _ in 0..count {
            let game_index = (rng.next_u64() as usize) % self.games.len();
            let game = &self.games[game_index];
            if game.is_empty() {
                continue;
            }
            let sample_index = (rng.next_u64() as usize) % game.len();
            samples.push(game[sample_index].sample.clone());
        }
        samples
    }

    fn prune_overtrained(&mut self, max_train_count: u32) {
        if max_train_count == 0 {
            return;
        }
        let mut rebuilt = VecDeque::with_capacity(self.games.len());
        let mut samples = 0usize;
        let mut train_count_sum = 0u64;
        while let Some(mut game) = self.games.pop_front() {
            game.retain(|entry| entry.train_count <= max_train_count);
            if !game.is_empty() {
                samples += game.len();
                train_count_sum += game
                    .iter()
                    .map(|entry| entry.train_count as u64)
                    .sum::<u64>();
                rebuilt.push_back(game);
            }
        }
        self.games = rebuilt;
        self.samples = samples;
        self.train_count_sum = train_count_sum;
    }

    fn encode_replay_payload(&self) -> io::Result<Vec<u8>> {
        let mut out = Vec::new();
        replay_push_u64(&mut out, self.game_capacity as u64);
        replay_push_u64(&mut out, self.games.len() as u64);
        for game in &self.games {
            replay_push_u64(&mut out, game.len() as u64);
            for entry in game {
                encode_replay_entry(&mut out, entry)?;
            }
        }
        Ok(out)
    }

    fn decode_replay_payload(data: &[u8], game_capacity: usize, version: u32) -> io::Result<Self> {
        let mut reader = Cursor::new(data);
        let _stored_capacity = replay_read_u64(&mut reader)? as usize;
        let n_games = replay_read_u64(&mut reader)? as usize;
        if n_games > 10_000_000 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "replay decode: absurd game count",
            ));
        }
        let mut games = VecDeque::with_capacity(n_games.min(game_capacity.max(1)));
        let mut samples = 0usize;
        let mut train_count_sum = 0u64;
        for _ in 0..n_games {
            let n_s = replay_read_u64(&mut reader)? as usize;
            if n_s > 50_000 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "replay decode: absurd samples per game",
                ));
            }
            let mut game = Vec::with_capacity(n_s);
            for _ in 0..n_s {
                let entry = decode_replay_entry(&mut reader, version)?;
                train_count_sum += entry.train_count as u64;
                game.push(entry);
            }
            samples += game.len();
            games.push_back(game);
        }
        let mut pool = Self {
            game_capacity,
            games,
            samples,
            train_count_sum,
        };
        while pool.games.len() > pool.game_capacity {
            if let Some(removed) = pool.games.pop_front() {
                pool.samples = pool.samples.saturating_sub(removed.len());
                pool.train_count_sum = pool
                    .train_count_sum
                    .saturating_sub(removed.iter().map(|entry| entry.train_count as u64).sum());
            }
        }
        Ok(pool)
    }

    pub fn save_snapshot_lz4(&self, path: &Path) -> io::Result<()> {
        if self.game_capacity == 0 || self.games.is_empty() {
            let _ = fs::remove_file(path);
            return Ok(());
        }
        let inner = self.encode_replay_payload()?;
        if inner.len() > REPLAY_MAX_DECOMPRESSED_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "replay snapshot raw size {} exceeds {}",
                    inner.len(),
                    REPLAY_MAX_DECOMPRESSED_BYTES
                ),
            ));
        }
        let compressed = compress_prepend_size(&inner);
        let mut file_blob = Vec::with_capacity(8 + compressed.len());
        file_blob.extend_from_slice(REPLAY_MAGIC);
        replay_push_u32(&mut file_blob, REPLAY_FILE_VERSION);
        file_blob.extend_from_slice(&compressed);
        let tmp = PathBuf::from(format!("{}.tmp", path.display()));
        fs::write(&tmp, &file_blob)?;
        if path.exists() {
            fs::remove_file(path)?;
        }
        fs::rename(&tmp, path)?;
        Ok(())
    }

    pub fn load_snapshot_lz4(path: &Path, game_capacity: usize) -> io::Result<Self> {
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
        let ver = u32::from_le_bytes(file_blob[4..8].try_into().unwrap());
        if !(2..=REPLAY_FILE_VERSION).contains(&ver) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "replay unsupported version {ver} (supported: v2..=v{REPLAY_FILE_VERSION})"
                ),
            ));
        }
        let inner = decompress_size_prepended(&file_blob[8..]).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("replay lz4 decompress: {err:?}"),
            )
        })?;
        if inner.len() > REPLAY_MAX_DECOMPRESSED_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "replay decompressed size over cap",
            ));
        }
        Self::decode_replay_payload(&inner, game_capacity, ver)
    }
}
