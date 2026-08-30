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
    hashes: HashSet<u64>,
    seen: u64,
}

impl AzMidgamePool {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            snapshots: Vec::with_capacity(capacity.min(65_536)),
            hashes: HashSet::with_capacity(capacity.min(65_536)),
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
            let hash = snapshot.position.hash();
            if self.hashes.contains(&hash) {
                continue;
            }
            self.seen = self.seen.saturating_add(1);
            if self.snapshots.len() < self.capacity {
                self.hashes.insert(hash);
                self.snapshots.push(snapshot);
                added += 1;
                continue;
            }
            let mut rng = SplitMix64::new(seed ^ self.seen.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let slot = (rng.next_u64() % self.seen.max(1)) as usize;
            if slot >= self.capacity {
                continue;
            }
            self.hashes.remove(&self.snapshots[slot].position.hash());
            self.hashes.insert(hash);
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
                pool.hashes.insert(snapshot.position.hash());
                pool.snapshots.push(snapshot);
            }
        }
        pool.seen = seen.max(pool.snapshots.len() as u64);
        Ok(pool)
    }
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
}
