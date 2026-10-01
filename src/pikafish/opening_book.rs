//! Px0 官方 book.pgn.gz 是 FEN 标签加空注释的局面集合。
use crate::{
    az::{AzStartSnapshot, SplitMix64},
    xiangqi::Position,
};
use flate2::read::GzDecoder;
use std::{
    fs::File,
    io::{self, BufRead, BufReader},
    path::Path,
};

pub struct Px0OpeningBook {
    fens: Vec<String>,
    order: Vec<usize>,
    cursor: usize,
}

impl Px0OpeningBook {
    pub fn load(path: impl AsRef<Path>, seed: u64) -> io::Result<Self> {
        Self::read(BufReader::new(GzDecoder::new(File::open(path)?)), seed)
    }

    fn read(reader: impl BufRead, seed: u64) -> io::Result<Self> {
        let mut fens = Vec::new();
        for line in reader.lines() {
            let line = line?;
            let line = line.trim().trim_start_matches('\u{feff}');
            if let Some(fen) = line
                .strip_prefix("[FEN \"")
                .and_then(|s| s.strip_suffix("\"]"))
            {
                fens.push(fen.to_owned());
            } else if !line.is_empty() && line != "{}" {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Px0 FEN 开局库包含不支持的内容",
                ));
            }
        }
        if fens.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Px0 开局库为空"));
        }
        let mut order: Vec<_> = (0..fens.len()).collect();
        let mut rng = SplitMix64::new(seed);
        for i in (1..order.len()).rev() {
            let j = rng.next_u64() as usize % (i + 1);
            order.swap(i, j);
        }
        Ok(Self {
            fens,
            order,
            cursor: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.fens.len()
    }

    pub fn next_batch(
        &mut self,
        count: usize,
        generation: u32,
    ) -> io::Result<Vec<AzStartSnapshot>> {
        let mut snapshots = Vec::with_capacity(count);
        for _ in 0..count {
            if self.cursor == self.order.len() {
                self.cursor = 0;
            }
            let fen = &self.fens[self.order[self.cursor]];
            let position = Position::from_fen(fen)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            let rule_history = position.initial_rule_history();
            snapshots.push(AzStartSnapshot {
                position,
                rule_history,
                phase_ply: 0,
                generation,
            });
            self.cursor += 1;
        }
        Ok(snapshots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "需要官方 book.pgn.gz"]
    fn official_book_loads_and_supplies_valid_fens() {
        let mut book = Px0OpeningBook::load("book.pgn.gz", 7).unwrap();
        assert_eq!(book.len(), 3_210_663);
        for start in book.next_batch(1024, 0).unwrap() {
            assert!(!start.position.legal_moves().is_empty());
            assert_eq!(start.rule_history.len(), 1);
        }
    }

    #[test]
    fn shuffled_fens_are_used_once_per_cycle_without_invented_history() {
        let input = b"[FEN \"4k4/9/9/9/9/9/9/9/4P4/4K4 w\"]\n{}\n[FEN \"4k4/9/9/9/9/9/9/4P4/9/4K4 b\"]\n{}\n";
        let mut book = Px0OpeningBook::read(&input[..], 7).unwrap();
        let batch = book.next_batch(2, 3).unwrap();
        assert_ne!(batch[0].position.to_fen(), batch[1].position.to_fen());
        assert!(
            batch
                .iter()
                .all(|s| s.rule_history.len() == 1 && s.phase_ply == 0 && s.generation == 3)
        );
        let next = book.next_batch(2, 4).unwrap();
        for (first, repeated) in batch.iter().zip(next) {
            assert_eq!(first.position.to_fen(), repeated.position.to_fen());
        }
    }
}
