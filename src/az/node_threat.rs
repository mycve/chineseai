//! 每个已评估的 MCTS 节点保存当前视角的威胁状态，由同视角祖先更新。
use super::{
    AzNnue, VALUE_THREAT_RANK, ValueKingState, inference::visit_value_threat_entries_from,
};
use crate::xiangqi::{BOARD_SIZE, Color, Position, color_index, piece_kind_index};

const NONE: u32 = u32::MAX;
const _: () = assert!(super::VALUE_THREAT_VOCAB <= u16::MAX as usize + 1);

#[derive(Default)]
pub(crate) struct NodeThreatArena {
    states: Vec<ThreatState>,
    sums: Vec<[f32; VALUE_THREAT_RANK]>,
    entries: Vec<u32>,
    pending: Vec<u32>,
    merged: Vec<u32>,
    roots: [Option<u32>; 2],
}

struct ThreatState {
    start: u32,
    len: u16,
    age: u8,
    sum: u32,
    king: ValueKingState,
}

impl NodeThreatArena {
    pub(crate) fn clear(&mut self) {
        self.states.clear();
        self.sums.clear();
        self.entries.clear();
        self.roots = [None; 2];
    }

    pub(crate) fn root(&mut self, model: &AzNnue, position: &Position, side: Color) -> u32 {
        if let Some(index) = self.roots[color_index(side)] {
            return index;
        }
        self.merged.clear();
        if model.threat_incremental {
            visit_value_threat_entries_from(position, side, u128::MAX, |e| self.merged.push(e));
        }
        let mut sum = [0.0; VALUE_THREAT_RANK];
        for &entry in &self.merged {
            add(&mut sum, model, entry, 1.0);
        }
        let king = ValueKingState::new(model, position, side);
        let index = self.store(model.threat_incremental.then_some(sum), 0, king);
        self.roots[color_index(side)] = Some(index);
        index
    }

    pub(crate) fn update(
        &mut self,
        model: &AzNnue,
        base: u32,
        before: &Position,
        after: &Position,
        side: Color,
        changed: u128,
    ) -> u32 {
        let base = &self.states[base as usize];
        let mut age = base.age + 2;
        let king = if age >= 32 {
            ValueKingState::new(model, after, side)
        } else {
            base.king.update(model, before, after, side, changed)
        };
        if !model.threat_incremental {
            self.merged.clear();
            return self.store(None, if age >= 32 { 0 } else { age }, king);
        }
        let mut sum = self.sums[base.sum as usize];
        let previous = &self.entries[base.start as usize..base.start as usize + base.len as usize];
        let mut dirty = changed;
        let mut occupancy_changed = 0;
        let mut changes = changed;
        while changes != 0 {
            let square = changes.trailing_zeros() as usize;
            changes &= changes - 1;
            if before.piece_at(square).is_some() != after.piece_at(square).is_some() {
                occupancy_changed |= 1u128 << square;
            }
        }
        for source in after.occupied_squares() {
            let piece = after.piece_at(source).unwrap();
            let class = color_index(piece.color) * 7 + piece_kind_index(piece.kind);
            let (targets, blockers) = dependency_masks()[class][source];
            if targets & changed != 0 || blockers & occupancy_changed != 0 {
                dirty |= 1u128 << source;
            }
        }
        self.pending.clear();
        visit_value_threat_entries_from(after, side, dirty, |e| self.pending.push(e));
        self.merged.clear();
        let (mut old, mut new) = (0, 0);
        while old < previous.len() || new < self.pending.len() {
            let old_key = previous.get(old).map_or(NONE, |e| e >> 16);
            let new_key = self.pending.get(new).map_or(NONE, |e| e >> 16);
            if old_key < new_key {
                let end = group_end(previous, old);
                if dirty & (1u128 << (old_key & 127)) == 0 {
                    self.merged.extend_from_slice(&previous[old..end]);
                } else {
                    for &e in &previous[old..end] {
                        add(&mut sum, model, e, -1.0);
                    }
                }
                old = end;
            } else if new_key < old_key {
                let end = group_end(&self.pending, new);
                for &e in &self.pending[new..end] {
                    add(&mut sum, model, e, 1.0);
                }
                self.merged.extend_from_slice(&self.pending[new..end]);
                new = end;
            } else {
                let old_end = group_end(previous, old);
                let new_end = group_end(&self.pending, new);
                let a = &previous[old..old_end];
                let b = &self.pending[new..new_end];
                if a != b {
                    apply_difference(&mut sum, model, a, b);
                }
                self.merged.extend_from_slice(b);
                old = old_end;
                new = new_end;
            }
        }
        // 限制浮点加减误差；重算沿用完整特征遍历的累加顺序。
        if age >= 32 {
            sum.fill(0.0);
            for &entry in &self.merged {
                add(&mut sum, model, entry, 1.0);
            }
            age = 0;
        }
        self.store(Some(sum), age, king)
    }

    fn store(
        &mut self,
        sum: Option<[f32; VALUE_THREAT_RANK]>,
        age: u8,
        king: ValueKingState,
    ) -> u32 {
        let sum = sum.map_or(NONE, |sum| {
            let index = u32::try_from(self.sums.len()).expect("threat sum index overflow");
            self.sums.push(sum);
            index
        });
        let index = u32::try_from(self.states.len()).expect("threat state index overflow");
        self.states.push(ThreatState {
            start: u32::try_from(self.entries.len()).expect("threat feature arena overflow"),
            len: u16::try_from(self.merged.len()).expect("threat feature count overflow"),
            age,
            sum,
            king,
        });
        self.entries.extend_from_slice(&self.merged);
        index
    }

    pub(crate) fn king(&self, state: u32) -> ValueKingState {
        self.states[state as usize].king
    }

    pub(crate) fn value(&self, state: u32) -> ([f32; VALUE_THREAT_RANK], usize) {
        let state = &self.states[state as usize];
        (self.sums[state.sum as usize], state.len as usize)
    }
}

fn group_end(entries: &[u32], start: usize) -> usize {
    let key = entries[start] >> 16;
    let mut end = start + 1;
    while end < entries.len() && entries[end] >> 16 == key {
        end += 1;
    }
    end
}

fn apply_difference(
    sum: &mut [f32; VALUE_THREAT_RANK],
    model: &AzNnue,
    previous: &[u32],
    next: &[u32],
) {
    // 每个来源最多 12 项：四条车炮射线各两个阻挡项加一个炮架项。
    let mut a = [0u16; 12];
    let mut b = [0u16; 12];
    for (dst, &src) in a[..previous.len()].iter_mut().zip(previous) {
        *dst = src as u16;
    }
    for (dst, &src) in b[..next.len()].iter_mut().zip(next) {
        *dst = src as u16;
    }
    let a = &mut a[..previous.len()];
    let b = &mut b[..next.len()];
    a.sort_unstable();
    b.sort_unstable();
    let (mut old, mut new) = (0, 0);
    while old < a.len() || new < b.len() {
        if old < a.len() && new < b.len() && a[old] == b[new] {
            old += 1;
            new += 1;
        } else if new == b.len() || old < a.len() && a[old] < b[new] {
            add(sum, model, u32::from(a[old]), -1.0);
            old += 1;
        } else {
            add(sum, model, u32::from(b[new]), 1.0);
            new += 1;
        }
    }
}

#[inline(always)]
fn add(sum: &mut [f32; VALUE_THREAT_RANK], model: &AzNnue, entry: u32, sign: f32) {
    let start = (entry as u16 as usize) * VALUE_THREAT_RANK;
    for (sum, &weight) in sum
        .iter_mut()
        .zip(&model.threat_embedding[start..start + VALUE_THREAT_RANK])
    {
        *sum += sign * weight;
    }
}

fn dependency_masks() -> &'static [[(u128, u128); BOARD_SIZE]; 14] {
    static MASKS: std::sync::OnceLock<[[(u128, u128); BOARD_SIZE]; 14]> =
        std::sync::OnceLock::new();
    MASKS.get_or_init(|| {
        std::array::from_fn(|class| {
            std::array::from_fn(|source| {
                let color = if class / 7 == color_index(Color::Red) {
                    Color::Red
                } else {
                    Color::Black
                };
                let (mut targets, mut blockers) = (0, 0);
                for target in 0..BOARD_SIZE {
                    let df = target as i32 % 9 - source as i32 % 9;
                    let dr = target as i32 / 9 - source as i32 / 9;
                    let (f, r) = (df.abs(), dr.abs());
                    let (endpoint, blocker) = match class % 7 {
                        0 => (f + r == 1, false),
                        1 => (f == 1 && r == 1, false),
                        2 => (f == 2 && r == 2, f == 1 && r == 1),
                        3 => ((f == 1 && r == 2) || (f == 2 && r == 1), f + r == 1),
                        4 | 5 => (f == 0 || r == 0, false),
                        6 => (
                            df == 0 && dr == color.forward_step()
                                || dr == 0
                                    && f == 1
                                    && if color == Color::Red {
                                        source / 9 <= 4
                                    } else {
                                        source / 9 >= 5
                                    },
                            false,
                        ),
                        _ => unreachable!(),
                    };
                    if endpoint {
                        targets |= 1u128 << target;
                    }
                    if blocker {
                        blockers |= 1u128 << target;
                    }
                }
                (targets, blockers)
            })
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::az::SplitMix64;

    #[test]
    fn persistent_states_match_full_features_across_branches_and_refreshes() {
        let mut model = AzNnue::random(16, 123);
        model.threat_incremental = true;
        for (i, weight) in model.value_king_piece_hidden.iter_mut().enumerate() {
            *weight = (i % 23) as f32 * 0.001 - 0.011;
        }
        let mut arena = NodeThreatArena::default();
        let mut rng = SplitMix64::new(35);
        let mut root = Position::startpos();
        let first = root.legal_moves()[0];
        root.make_move(first);
        let mut captures = 0;
        for initial in [Position::startpos(), root] {
            let side = initial.side_to_move();
            let initial_state = arena.root(&model, &initial, side);
            let mut positions = vec![(initial, initial_state)];
            for step in 0..400 {
                let index = if step % 5 == 0 {
                    rng.next_u64() as usize % positions.len()
                } else {
                    positions.len() - 1
                };
                let (before, base) = &positions[index];
                let mut after = before.clone();
                let mut changed = 0;
                let mut valid = true;
                for _ in 0..2 {
                    let moves = after.legal_moves();
                    if moves.is_empty() {
                        valid = false;
                        break;
                    }
                    let preferred: Vec<_> = moves
                        .iter()
                        .copied()
                        .filter(|m| after.piece_at(m.to as usize).is_some())
                        .collect();
                    let choices = if step % 3 == 0 && !preferred.is_empty() {
                        &preferred
                    } else {
                        &moves
                    };
                    let mv = choices[rng.next_u64() as usize % choices.len()];
                    captures += usize::from(after.piece_at(mv.to as usize).is_some());
                    changed |= (1u128 << mv.from) | (1u128 << mv.to);
                    after.make_move(mv);
                }
                if !valid {
                    positions.truncate(1);
                    continue;
                }
                let state = arena.update(&model, *base, before, &after, side, changed);
                let saved = &arena.states[state as usize];
                let expected_king = ValueKingState::new(&model, &after, side);
                assert_eq!(saved.king.bucket(), expected_king.bucket());
                assert_eq!(saved.king.active(), expected_king.active());
                for (a, b) in saved.king.pool.iter().zip(expected_king.pool) {
                    assert!((a - b).abs() < 2e-5, "king piece pool drift at step {step}");
                }
                let mut expected_entries = Vec::new();
                let mut expected_sum = [0.0; VALUE_THREAT_RANK];
                visit_value_threat_entries_from(&after, side, u128::MAX, |e| {
                    expected_entries.push(e);
                    add(&mut expected_sum, &model, e, 1.0);
                });
                assert_eq!(
                    &arena.entries[saved.start as usize..saved.start as usize + saved.len as usize],
                    expected_entries
                );
                for (a, b) in arena.sums[saved.sum as usize].iter().zip(expected_sum) {
                    assert!(
                        (a - b).abs() < 2e-5,
                        "step={step} error={} fen={}",
                        (a - b).abs(),
                        after.to_fen()
                    );
                }
                positions.push((after, state));
            }
        }
        assert!(captures > 20);
        arena.clear();
        assert!(arena.states.is_empty() && arena.entries.is_empty());
    }
}
