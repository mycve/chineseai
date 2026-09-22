use std::{
    collections::HashSet,
    fs, io,
    io::{Cursor, Read},
    path::Path,
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use lz4_flex::block::{compress_prepend_size, decompress_size_prepended};

use crate::xiangqi::{Color, Move, Piece, PieceKind, Position, RuleHistoryEntry};

use super::SplitMix64;

const MIDGAME_MAGIC: &[u8; 5] = b"AZMG2";
const MAX_SNAPSHOTS: usize = 1_000_000;
const MAX_HISTORY: usize = 4096;
const MAX_FEN_BYTES: usize = 1024;

#[derive(Clone, Debug)]
pub struct AzStartSnapshot {
    pub position: Position,
    pub rule_history: Vec<RuleHistoryEntry>,
    pub phase_ply: u16,
    pub generation: u32,
}

#[derive(Clone, Debug)]
pub struct AzMidgamePool {
    capacity: usize,
    snapshots: Vec<AzStartSnapshot>,
    identities: HashSet<u64>,
    seen: u64,
}

impl AzMidgamePool {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            snapshots: Vec::with_capacity(capacity.min(65_536)),
            identities: HashSet::with_capacity(capacity.min(65_536)),
            seen: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    pub fn add_snapshots<I>(&mut self, snapshots: I, seed: u64) -> usize
    where
        I: IntoIterator<Item = AzStartSnapshot>,
    {
        let mut added = 0usize;
        for snapshot in snapshots {
            if self.capacity == 0 || !snapshot_is_consistent(&snapshot) {
                continue;
            }
            let identity = snapshot_identity(&snapshot);
            if self.identities.contains(&identity) {
                continue;
            }
            self.seen = self.seen.saturating_add(1);
            if self.snapshots.len() < self.capacity {
                self.identities.insert(identity);
                self.snapshots.push(snapshot);
                added += 1;
                continue;
            }
            let mut rng = SplitMix64::new(seed ^ self.seen.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let slot = (rng.next_u64() % self.seen.max(1)) as usize;
            if slot >= self.capacity {
                continue;
            }
            self.identities
                .remove(&snapshot_identity(&self.snapshots[slot]));
            self.identities.insert(identity);
            self.snapshots[slot] = snapshot;
            added += 1;
        }
        added
    }

    pub fn sample(&self, count: usize, rng: &mut SplitMix64) -> Vec<AzStartSnapshot> {
        let count = count.min(self.snapshots.len());
        let mut selected = HashSet::with_capacity(count);
        let mut out = Vec::with_capacity(count);
        while out.len() < count {
            let index = (rng.next_u64() as usize) % self.snapshots.len();
            if selected.insert(index) {
                out.push(self.snapshots[index].clone());
            }
        }
        out
    }

    /// Samples a requested share from recent generations and fills the remainder
    /// from older snapshots. If either side lacks enough entries, the other side
    /// supplies the deficit without duplicating a snapshot within the batch.
    pub fn sample_recent_mixed(
        &self,
        count: usize,
        recent_fraction: f32,
        recent_generations: u32,
        current_generation: u32,
        rng: &mut SplitMix64,
    ) -> Vec<AzStartSnapshot> {
        let count = count.min(self.snapshots.len());
        if count == 0 {
            return Vec::new();
        }
        let cutoff = current_generation.saturating_sub(recent_generations.max(1) - 1);
        let (recent, old): (Vec<_>, Vec<_>) = (0..self.snapshots.len())
            .partition(|&index| self.snapshots[index].generation >= cutoff);
        let recent_target =
            ((count as f32 * recent_fraction.clamp(0.0, 1.0)).round() as usize).min(recent.len());
        let old_target = (count - recent_target).min(old.len());
        let mut selected = HashSet::with_capacity(count);
        sample_indices(&recent, recent_target, &mut selected, rng);
        sample_indices(&old, old_target, &mut selected, rng);
        if selected.len() < count {
            let all = (0..self.snapshots.len()).collect::<Vec<_>>();
            sample_indices(&all, count - selected.len(), &mut selected, rng);
        }
        selected
            .into_iter()
            .map(|index| self.snapshots[index].clone())
            .collect()
    }

    pub fn save_lz4(&self, path: &Path) -> io::Result<()> {
        let mut raw = Vec::new();
        raw.extend_from_slice(MIDGAME_MAGIC);
        raw.write_u64::<LittleEndian>(self.seen)?;
        raw.write_u32::<LittleEndian>(self.snapshots.len() as u32)?;
        for snapshot in &self.snapshots {
            encode_snapshot(&mut raw, snapshot)?;
        }
        if let Some(parent) = path.parent().filter(|path| !path.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        let compressed = compress_prepend_size(&raw);
        let temporary = path.with_extension("lz4.tmp");
        fs::write(&temporary, compressed)?;
        #[cfg(windows)]
        if path.exists() {
            fs::remove_file(path)?;
        }
        fs::rename(temporary, path)
    }

    pub fn load_lz4(path: &Path, capacity: usize) -> io::Result<Self> {
        let compressed = fs::read(path)?;
        let raw = decompress_size_prepended(&compressed)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err.to_string()))?;
        let mut reader = Cursor::new(raw);
        let mut magic = [0u8; MIDGAME_MAGIC.len()];
        reader.read_exact(&mut magic)?;
        if &magic != MIDGAME_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid midgame pool magic",
            ));
        }
        let seen = reader.read_u64::<LittleEndian>()?;
        let count = reader.read_u32::<LittleEndian>()? as usize;
        if count > MAX_SNAPSHOTS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "midgame pool snapshot count exceeds limit",
            ));
        }
        let mut pool = Self::new(capacity);
        for _ in 0..count {
            let snapshot = decode_snapshot(&mut reader)?;
            if pool.snapshots.len() < capacity {
                pool.identities.insert(snapshot_identity(&snapshot));
                pool.snapshots.push(snapshot);
            }
        }
        pool.seen = seen.max(pool.snapshots.len() as u64);
        Ok(pool)
    }
}

fn sample_indices(
    choices: &[usize],
    count: usize,
    selected: &mut HashSet<usize>,
    rng: &mut SplitMix64,
) {
    let available = choices
        .iter()
        .filter(|index| !selected.contains(index))
        .count();
    let target = count.min(available);
    let initial_len = selected.len();
    while selected.len() < initial_len + target {
        selected.insert(choices[rng.next_u64() as usize % choices.len()]);
    }
}

/// Board hashes alone are insufficient in Xiangqi: the same placement can have
/// different repetition, long-check, long-chase, and natural-move-limit state.
/// Keep a compact deterministic digest of the rule-relevant recent history.
fn snapshot_identity(snapshot: &AzStartSnapshot) -> u64 {
    fn mix(state: u64, value: u64) -> u64 {
        let mut value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        state.rotate_left(17) ^ value ^ (value >> 31)
    }

    let mut state = mix(0xA076_1D64_78BD_642F, snapshot.position.hash());
    state = mix(state, snapshot.rule_history.len() as u64);
    for entry in snapshot.rule_history.iter().rev().take(32).rev() {
        state = mix(state, entry.hash);
        let flags = encode_color(entry.side_to_move) as u64
            | (entry.mover.map(encode_color).unwrap_or(2) as u64) << 2
            | u64::from(entry.gives_check) << 4
            | (entry.rule60_clock as u64) << 5;
        state = mix(state, flags);
        state = mix(state, entry.chased_mask as u64);
        state = mix(state, (entry.chased_mask >> 64) as u64);
        let move_code = entry
            .mv
            .map_or(u16::MAX, |mv| u16::from(mv.from) | (u16::from(mv.to) << 8));
        let captured = entry.captured.map(encode_piece).unwrap_or(u8::MAX);
        state = mix(state, u64::from(move_code) | (u64::from(captured) << 16));
    }
    state
}

fn snapshot_is_consistent(snapshot: &AzStartSnapshot) -> bool {
    snapshot.position.has_general(Color::Red)
        && snapshot.position.has_general(Color::Black)
        && snapshot
            .rule_history
            .last()
            .is_some_and(|entry| entry.hash == snapshot.position.hash())
        && snapshot
            .position
            .rule_outcome_with_history(&snapshot.rule_history)
            .is_none()
}

fn encode_snapshot(out: &mut Vec<u8>, snapshot: &AzStartSnapshot) -> io::Result<()> {
    let fen = snapshot
        .position
        .to_fen_with_history(&snapshot.rule_history);
    if fen.len() > MAX_FEN_BYTES || snapshot.rule_history.len() > MAX_HISTORY {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "midgame snapshot exceeds encoding limits",
        ));
    }
    out.write_u16::<LittleEndian>(snapshot.phase_ply)?;
    out.write_u32::<LittleEndian>(snapshot.generation)?;
    out.write_u16::<LittleEndian>(fen.len() as u16)?;
    out.extend_from_slice(fen.as_bytes());
    out.write_u16::<LittleEndian>(snapshot.rule_history.len() as u16)?;
    for entry in &snapshot.rule_history {
        out.write_u64::<LittleEndian>(entry.hash)?;
        out.write_u8(encode_color(entry.side_to_move))?;
        out.write_u8(entry.mover.map(encode_color).unwrap_or(2))?;
        out.write_u8(u8::from(entry.gives_check))?;
        out.extend_from_slice(&entry.chased_mask.to_le_bytes());
        out.write_u8(entry.mv.map(|mv| mv.from).unwrap_or(u8::MAX))?;
        out.write_u8(entry.mv.map(|mv| mv.to).unwrap_or(u8::MAX))?;
        out.write_u8(entry.captured.map(encode_piece).unwrap_or(u8::MAX))?;
        out.write_u16::<LittleEndian>(entry.rule60_clock)?;
    }
    Ok(())
}

fn decode_snapshot(reader: &mut Cursor<Vec<u8>>) -> io::Result<AzStartSnapshot> {
    let phase_ply = reader.read_u16::<LittleEndian>()?;
    let generation = reader.read_u32::<LittleEndian>()?;
    let fen_len = reader.read_u16::<LittleEndian>()? as usize;
    if fen_len > MAX_FEN_BYTES {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "FEN too long"));
    }
    let mut fen = vec![0u8; fen_len];
    reader.read_exact(&mut fen)?;
    let fen =
        std::str::from_utf8(&fen).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    let position =
        Position::from_fen(fen).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    let history_len = reader.read_u16::<LittleEndian>()? as usize;
    if history_len > MAX_HISTORY {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "rule history too long",
        ));
    }
    let mut rule_history = Vec::with_capacity(history_len);
    for _ in 0..history_len {
        let hash = reader.read_u64::<LittleEndian>()?;
        let side_to_move = decode_color(reader.read_u8()?)?;
        let mover = match reader.read_u8()? {
            0 => Some(Color::Red),
            1 => Some(Color::Black),
            2 => None,
            _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid mover")),
        };
        let gives_check = reader.read_u8()? != 0;
        let mut chased = [0u8; 16];
        reader.read_exact(&mut chased)?;
        let chased_mask = u128::from_le_bytes(chased);
        let from = reader.read_u8()?;
        let to = reader.read_u8()?;
        let mv = (from != u8::MAX && to != u8::MAX).then(|| Move { from, to });
        let captured = match reader.read_u8()? {
            u8::MAX => None,
            value => Some(decode_piece(value)?),
        };
        let rule60_clock = reader.read_u16::<LittleEndian>()?;
        rule_history.push(RuleHistoryEntry {
            hash,
            side_to_move,
            mover,
            gives_check,
            chased_mask,
            mv,
            captured,
            rule60_clock,
        });
    }
    let snapshot = AzStartSnapshot {
        position,
        rule_history,
        phase_ply,
        generation,
    };
    if !snapshot_is_consistent(&snapshot) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "inconsistent midgame snapshot",
        ));
    }
    Ok(snapshot)
}

fn encode_color(color: Color) -> u8 {
    match color {
        Color::Red => 0,
        Color::Black => 1,
    }
}

fn decode_color(value: u8) -> io::Result<Color> {
    match value {
        0 => Ok(Color::Red),
        1 => Ok(Color::Black),
        _ => Err(io::Error::new(io::ErrorKind::InvalidData, "invalid color")),
    }
}

fn encode_piece(piece: Piece) -> u8 {
    encode_color(piece.color) * 7
        + match piece.kind {
            PieceKind::General => 0,
            PieceKind::Advisor => 1,
            PieceKind::Elephant => 2,
            PieceKind::Horse => 3,
            PieceKind::Rook => 4,
            PieceKind::Cannon => 5,
            PieceKind::Soldier => 6,
        }
}

fn decode_piece(value: u8) -> io::Result<Piece> {
    if value >= 14 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid piece"));
    }
    let color = decode_color(value / 7)?;
    let kind = match value % 7 {
        0 => PieceKind::General,
        1 => PieceKind::Advisor,
        2 => PieceKind::Elephant,
        3 => PieceKind::Horse,
        4 => PieceKind::Rook,
        5 => PieceKind::Cannon,
        6 => PieceKind::Soldier,
        _ => unreachable!(),
    };
    Ok(Piece { color, kind })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(phase_ply: u16) -> AzStartSnapshot {
        let position = Position::startpos();
        AzStartSnapshot {
            rule_history: position.initial_rule_history(),
            position,
            phase_ply,
            generation: 7,
        }
    }

    #[test]
    fn midgame_pool_roundtrips_full_rule_history() {
        let mut pool = AzMidgamePool::new(8);
        assert_eq!(pool.add_snapshots([snapshot(42)], 1), 1);
        let path = std::env::temp_dir().join(format!(
            "chineseai-midgame-{}-{}.lz4",
            std::process::id(),
            42
        ));
        pool.save_lz4(&path).unwrap();
        let loaded = AzMidgamePool::load_lz4(&path, 8).unwrap();
        fs::remove_file(path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.snapshots[0].phase_ply, 42);
        assert_eq!(
            loaded.snapshots[0].rule_history,
            pool.snapshots[0].rule_history
        );
    }

    #[test]
    fn midgame_pool_deduplicates_and_respects_capacity() {
        let mut pool = AzMidgamePool::new(1);
        assert_eq!(pool.add_snapshots([snapshot(30), snapshot(31)], 2), 1);
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn pool_keeps_same_board_with_distinct_rule_history() {
        let first = snapshot(42);
        let mut second = first.clone();
        second.rule_history.last_mut().unwrap().rule60_clock = 1;
        assert_eq!(first.position.hash(), second.position.hash());
        assert_ne!(snapshot_identity(&first), snapshot_identity(&second));

        let mut pool = AzMidgamePool::new(8);
        assert_eq!(pool.add_snapshots([first, second], 3), 2);
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn mixed_sampling_prioritizes_recent_generations() {
        let mut pool = AzMidgamePool::new(8);
        pool.snapshots = (0..8)
            .map(|generation| AzStartSnapshot {
                generation,
                ..snapshot(generation as u16)
            })
            .collect();
        let sampled = pool.sample_recent_mixed(4, 0.75, 4, 7, &mut SplitMix64::new(9));
        assert_eq!(sampled.len(), 4);
        assert_eq!(
            sampled
                .iter()
                .filter(|sample| sample.generation >= 4)
                .count(),
            3
        );
    }
}
