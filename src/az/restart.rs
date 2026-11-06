//! Go-Exploit Visited States Circular Archive（GEVC）。
//!
//! 完成轨迹后发布非终局局面，从最近访问的有界档案中均匀抽取重启点。
//! 对应论文第 3.1 节：https://arxiv.org/abs/2302.12359。
//! 保留完整规则历史；温度与单次轨迹长度由调用方按重启后的局部步数处理。

use std::{collections::VecDeque, sync::Mutex};

use crate::xiangqi::{Position, RuleHistoryEntry};

use super::{AzStartSnapshot, SplitMix64};

#[derive(Debug)]
pub struct AzRestartArchive {
    capacity: usize,
    state: Mutex<ArchiveState>,
}

#[derive(Debug, Default)]
struct ArchiveState {
    snapshots: VecDeque<AzStartSnapshot>,
    seen: u64,
}

/// 单条轨迹暂存区，结束前不向其他自博弈线程发布。
pub struct AzRestartGame {
    capacity: usize,
    snapshots: VecDeque<AzStartSnapshot>,
    seen: u64,
}

impl AzRestartArchive {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            state: Mutex::new(ArchiveState::default()),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.state
            .lock()
            .expect("restart archive poisoned")
            .snapshots
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 已发布轨迹的非终局局面数，包括因容量限制被淘汰的局面。
    pub fn seen(&self) -> u64 {
        self.state.lock().expect("restart archive poisoned").seen
    }

    pub fn start_recording(&self) -> AzRestartGame {
        AzRestartGame {
            capacity: self.capacity,
            snapshots: VecDeque::new(),
            seen: 0,
        }
    }

    pub fn add_game(&self, mut game: AzRestartGame) {
        assert_eq!(game.capacity, self.capacity);
        if self.capacity == 0 || game.seen == 0 {
            return;
        }
        let mut state = self.state.lock().expect("restart archive poisoned");
        state.seen = state.seen.saturating_add(game.seen);
        let keep_old = self.capacity - game.snapshots.len();
        let remove = state.snapshots.len().saturating_sub(keep_old);
        state.snapshots.drain(..remove);
        state.snapshots.append(&mut game.snapshots);
    }

    pub fn sample(&self, rng: &mut SplitMix64) -> Option<AzStartSnapshot> {
        let state = self.state.lock().expect("restart archive poisoned");
        let count = state.snapshots.len() as u64;
        if count == 0 {
            return None;
        }
        // 拒绝不完整区间，避免直接取模产生偏差。
        let reject_below = count.wrapping_neg() % count;
        let index = loop {
            let draw = rng.next_u64();
            if draw >= reject_below {
                break (draw % count) as usize;
            }
        };
        Some(state.snapshots[index].clone())
    }
}

impl AzRestartGame {
    /// 调用方只提交规则检查通过且仍有合法走法的局面。
    pub fn observe(
        &mut self,
        position: &Position,
        rule_history: &[RuleHistoryEntry],
        phase_ply: usize,
        generation: u32,
    ) {
        if self.capacity == 0 {
            return;
        }
        self.seen = self.seen.saturating_add(1);
        if self.snapshots.len() == self.capacity {
            self.snapshots.pop_front();
        }
        self.snapshots.push_back(AzStartSnapshot {
            position: position.clone(),
            rule_history: rule_history.to_vec(),
            phase_ply: phase_ply.min(u16::MAX as usize) as u16,
            generation,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xiangqi::{RuleDrawReason, RuleOutcome};

    fn publish(archive: &AzRestartArchive, range: std::ops::Range<usize>) {
        let position = Position::startpos();
        let history = position.initial_rule_history();
        let mut game = archive.start_recording();
        for ply in range {
            game.observe(&position, &history, ply, 7);
        }
        archive.add_game(game);
    }

    #[test]
    fn recording_is_private_until_game_finishes_and_retains_recent_states() {
        let archive = AzRestartArchive::new(3);
        publish(&archive, 0..2);
        let position = Position::startpos();
        let history = position.initial_rule_history();
        let mut game = archive.start_recording();
        for ply in 2..8 {
            game.observe(&position, &history, ply, 7);
        }
        assert_eq!(archive.seen(), 2);
        assert_eq!(archive.len(), 2);
        assert_eq!(game.snapshots.len(), 3);
        archive.add_game(game);
        assert_eq!(archive.seen(), 8);
        let state = archive.state.lock().unwrap();
        assert_eq!(
            state
                .snapshots
                .iter()
                .map(|s| s.phase_ply)
                .collect::<Vec<_>>(),
            [5, 6, 7]
        );
    }

    #[test]
    fn games_contribute_positions_instead_of_one_vote_per_game() {
        let archive = AzRestartArchive::new(5);
        publish(&archive, 0..4);
        publish(&archive, 4..5);
        let mut rng = SplitMix64::new(13);
        let mut counts = [0usize; 5];
        for _ in 0..10_000 {
            counts[archive.sample(&mut rng).unwrap().phase_ply as usize] += 1;
        }
        assert!(
            counts.iter().all(|&count| (1700..2300).contains(&count)),
            "{counts:?}"
        );
    }

    #[test]
    fn snapshot_keeps_rule_history_clock_and_generation() {
        let archive = AzRestartArchive::new(1);
        let mut position = Position::startpos();
        position.set_rule60_max_ply(Some(8));
        let mut history = position.initial_rule_history();
        for text in ["b0c2", "b9c7", "c2b0", "c7b9"] {
            let mv = position.parse_uci_move(text).unwrap();
            history.push(position.rule_history_entry_after_move(mv));
            position.make_move(mv);
        }
        let mut game = archive.start_recording();
        game.observe(&position, &history, 404, 31);
        archive.add_game(game);
        let mut sample = archive.sample(&mut SplitMix64::new(1)).unwrap();
        assert_eq!(sample.position, position);
        assert_eq!(sample.rule_history, history);
        assert_eq!(sample.phase_ply, 404);
        assert_eq!(sample.generation, 31);
        assert_eq!(
            sample
                .position
                .rule60_count_with_history(&sample.rule_history),
            4
        );
        for text in ["b0c2", "b9c7", "c2b0", "c7b9"] {
            let mv = sample.position.parse_uci_move(text).unwrap();
            sample
                .rule_history
                .push(sample.position.rule_history_entry_after_move(mv));
            sample.position.make_move(mv);
        }
        assert_eq!(
            sample
                .position
                .rule_outcome_with_history(&sample.rule_history),
            Some(RuleOutcome::Draw(RuleDrawReason::Repetition))
        );
    }

    #[test]
    fn capacity_zero_has_no_recording_or_rng_side_effects() {
        let archive = AzRestartArchive::new(0);
        publish(&archive, 0..100);
        let mut rng = SplitMix64::new(19);
        let before = rng.state;
        assert!(archive.sample(&mut rng).is_none());
        assert_eq!(rng.state, before);
        assert!(archive.is_empty());
        assert_eq!(archive.seen(), 0);
    }

    #[test]
    fn concurrent_completed_games_preserve_capacity_and_observation_count() {
        let archive = AzRestartArchive::new(11);
        std::thread::scope(|scope| {
            for worker in 0..4 {
                let archive = &archive;
                scope.spawn(move || publish(archive, worker * 40..(worker + 1) * 40));
            }
        });
        assert_eq!(archive.len(), 11);
        assert_eq!(archive.seen(), 160);
    }
}
