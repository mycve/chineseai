//! 节点只保存低维和；有界工作区缓存复用祖先图及按根编码的模式行。
use super::{features::*, *};
use crate::xiangqi::{Color, Position};
use smallvec::SmallVec;
use std::sync::Arc;

const GRAPH_CACHE_SIZE: usize = 32;

struct RootRows {
    offsets: [u32; 91],
    rows: Vec<u32>,
}
impl RootRows {
    fn extract(graph: &Graph, side: Color, roots: u128) -> Self {
        let events = motifs(graph, roots, side);
        let mut records = SmallVec::<[(u8, [u32; 3]); 64]>::new();
        let mut offsets = [0; 91];
        emit_motif_rows(graph, &events, side, |root, rows| {
            offsets[root + 1] += 3;
            records.push((root as u8, rows));
        });
        for i in 0..90 {
            offsets[i + 1] += offsets[i];
        }
        let mut rows = vec![0; offsets[90] as usize];
        let mut cursor = offsets;
        for (root, record) in records {
            let root = root as usize;
            let start = cursor[root] as usize;
            rows[start..start + 3].copy_from_slice(&record);
            cursor[root] += 3;
        }
        Self { offsets, rows }
    }
    fn at(&self, root: usize) -> &[u32] {
        &self.rows[self.offsets[root] as usize..self.offsets[root + 1] as usize]
    }
    fn patched(&self, patch: &Self, roots: u128) -> Self {
        let mut offsets = [0; 91];
        let mut rows = Vec::with_capacity(self.rows.len() + patch.rows.len());
        let mut cursor = 0;
        for root in bits(roots) {
            let delta = (rows.len() as u32).wrapping_sub(self.offsets[cursor]);
            rows.extend_from_slice(
                &self.rows[self.offsets[cursor] as usize..self.offsets[root] as usize],
            );
            for (out, &old) in offsets[cursor..=root]
                .iter_mut()
                .zip(&self.offsets[cursor..=root])
            {
                *out = old.wrapping_add(delta);
            }
            rows.extend_from_slice(patch.at(root));
            cursor = root + 1;
        }
        let delta = (rows.len() as u32).wrapping_sub(self.offsets[cursor]);
        rows.extend_from_slice(&self.rows[self.offsets[cursor] as usize..]);
        for (out, &old) in offsets[cursor..].iter_mut().zip(&self.offsets[cursor..]) {
            *out = old.wrapping_add(delta);
        }
        Self { offsets, rows }
    }
}

#[derive(Default)]
pub(crate) struct NodeFeatureArena {
    states: Vec<State>,
    working: Option<Arc<Graph>>,
    roots: [Option<u32>; 2],
    root_graph: Option<Arc<Graph>>,
    root_rows: [Option<RootRows>; 2],
    cache: Vec<Cached>,
    clock: u64,
    pub updates: usize,
    pub cached_updates: usize,
    pub refreshes: usize,
}
struct Cached {
    state: u32,
    graph: Arc<Graph>,
    rows: RootRows,
    touched: u64,
    depth: u32,
}
struct State {
    sums: FeatureSums,
    age: u8,
    depth: u32,
}
impl NodeFeatureArena {
    pub fn heap_bytes(&self) -> usize {
        self.states.capacity() * std::mem::size_of::<State>()
            + self.cache.capacity() * std::mem::size_of::<Cached>()
            + self
                .cache
                .iter()
                .map(|c| {
                    std::mem::size_of::<Graph>()
                        + 2 * std::mem::size_of::<usize>()
                        + c.graph.heap_bytes()
                        + c.rows.rows.capacity() * 4
                })
                .sum::<usize>()
            + self.root_graph.as_ref().map_or(0, |g| {
                std::mem::size_of::<Graph>() + 2 * std::mem::size_of::<usize>() + g.heap_bytes()
            })
            + self
                .root_rows
                .iter()
                .flatten()
                .map(|r| r.rows.capacity() * 4)
                .sum::<usize>()
    }
    pub fn clear(&mut self) {
        self.states.clear();
        self.working = None;
        self.roots = [None; 2];
        self.root_graph = None;
        self.root_rows = [None, None];
        self.cache.clear();
        self.clock = 0;
        self.updates = 0;
        self.cached_updates = 0;
        self.refreshes = 0;
    }
    pub fn take_graph(&mut self) -> Option<Arc<Graph>> {
        self.working.take()
    }
    pub fn root(&mut self, model: &AzNnue, p: &Position, side: Color) -> u32 {
        let index = crate::xiangqi::color_index(side);
        if let Some(id) = self.roots[index] {
            self.working = self.root_graph.clone();
            return id;
        }
        let graph = self
            .root_graph
            .get_or_insert_with(|| Arc::new(Graph::new(Board::new(p))));
        let rows = RootRows::extract(graph, side, u128::MAX);
        let sums = model.feature_sums_from_rows(
            graph,
            side,
            relation_rows(graph, side, u128::MAX),
            rows.rows.iter().copied(),
        );
        self.root_rows[index] = Some(rows);
        let pending = graph.clone();
        let id = self.store(sums, 0, 0);
        self.roots[index] = Some(id);
        self.working = Some(pending);
        id
    }
    pub fn value(&self, index: u32) -> FeatureSums {
        self.states[index as usize].sums
    }
    fn store(&mut self, sums: FeatureSums, age: u8, depth: u32) -> u32 {
        let index = self.states.len() as u32;
        self.states.push(State { sums, age, depth });
        index
    }
    fn cache(&mut self, state: u32, graph: Graph, rows: RootRows) {
        self.clock += 1;
        let depth = self.states[state as usize].depth;
        if self.cache.len() < GRAPH_CACHE_SIZE {
            let graph = Arc::new(graph);
            self.working = Some(graph.clone());
            self.cache.push(Cached {
                state,
                graph,
                rows,
                touched: self.clock,
                depth,
            });
        } else {
            let slot = self
                .cache
                .iter_mut()
                .max_by_key(|c| (c.depth, std::cmp::Reverse(c.touched)))
                .unwrap();
            slot.state = state;
            if let Some(cached_graph) = Arc::get_mut(&mut slot.graph) {
                *cached_graph = graph;
            } else {
                slot.graph = Arc::new(graph);
            }
            slot.rows = rows;
            slot.touched = self.clock;
            slot.depth = depth;
            self.working = Some(slot.graph.clone());
        }
    }
    pub fn update(
        &mut self,
        model: &AzNnue,
        base: u32,
        before: &Position,
        after: &Position,
        side: Color,
        changed: u128,
    ) -> u32 {
        self.working = None;
        self.updates += 1;
        let age = self.states[base as usize].age + 2;
        let depth = self.states[base as usize].depth
            + if before.side_to_move() == after.side_to_move() {
                2
            } else {
                1
            };
        let board = Board::new(after);
        // 小残局的完整模式很少，维护根行缓存比准确重提更贵。
        if board.occupied.count_ones() < 12 {
            let graph = Graph::new(board);
            let sums = model.feature_sums(&graph, side);
            self.refreshes += 1;
            self.working = Some(Arc::new(graph));
            return self.store(sums, 0, depth);
        }
        let root_index = self.roots.iter().position(|&root| root == Some(base));
        let cache_index = self.cache.iter().position(|c| c.state == base);
        if let Some(index) = cache_index {
            self.clock += 1;
            self.cache[index].touched = self.clock;
        }
        let cached = if let Some(index) = root_index {
            Some((
                self.root_graph.as_deref().unwrap(),
                self.root_rows[index].as_ref().unwrap(),
            ))
        } else {
            cache_index.map(|i| (&*self.cache[i].graph, &self.cache[i].rows))
        };
        let new_graph = if let Some((old, _)) = cached {
            old.updated(board, changed)
        } else {
            let root = self.root_graph.as_ref().unwrap();
            root.updated(board, root.board.changed_contents(&board))
        };
        let (sums, rows, age) = if let Some((old_graph, old_rows)) = cached.filter(|_| age < 32) {
            self.cached_updates += 1;
            let (sources, roots) = dirty_roots(old_graph, &new_graph, changed);
            let mut sums = self.states[base as usize].sums;
            update_kings(
                model,
                &mut sums,
                &old_graph.board,
                &new_graph.board,
                before,
                after,
                side,
                changed,
            );
            let (mut removed, mut added) =
                relation_changes(old_graph, &new_graph, side, sources, changed);
            difference(
                &mut sums.relations,
                &model.relation_embedding,
                &mut removed,
                &mut added,
            );
            let mut patch = RootRows::extract(&new_graph, side, roots);
            let rows = old_rows.patched(&patch, roots);
            let mut removed = SmallVec::<[u32; 128]>::new();
            for root in bits(roots) {
                removed.extend_from_slice(old_rows.at(root));
            }
            difference(
                &mut sums.motifs,
                &model.motif_embedding,
                &mut removed,
                &mut patch.rows,
            );
            (sums, rows, age)
        } else {
            self.refreshes += 1;
            let rows = RootRows::extract(&new_graph, side, u128::MAX);
            let sums = model.feature_sums_from_rows(
                &new_graph,
                side,
                relation_rows(&new_graph, side, u128::MAX),
                rows.rows.iter().copied(),
            );
            (sums, rows, 0)
        };
        let id = self.store(sums, age, depth);
        self.cache(id, new_graph, rows);
        id
    }
}

#[allow(clippy::too_many_arguments)]
fn update_kings(
    model: &AzNnue,
    sums: &mut FeatureSums,
    old: &Board,
    new: &Board,
    before: &Position,
    after: &Position,
    side: Color,
    changed: u128,
) {
    let old_buckets = canonical_buckets_for_perspective(before, side);
    let new_buckets = canonical_buckets_for_perspective(after, side);
    for (half, (old_bucket, new_bucket)) in [
        (old_buckets.0, new_buckets.0),
        (old_buckets.1, new_buckets.1),
    ]
    .into_iter()
    .enumerate()
    {
        let mask = if old_bucket == new_bucket {
            changed
        } else {
            u128::MAX
        };
        for (board, bucket, scale) in [(old, old_bucket, -1.), (new, new_bucket, 1.)] {
            for square in bits(board.occupied & mask) {
                let piece = board.cells[square].unwrap();
                let class = nnue::piece_absolute_feature_index(side, piece);
                let row =
                    ((half * 9 + bucket) * 14 + class) * 90 + nnue::canonical_square(side, square);
                add_scaled_row(&mut sums.king, &model.king_piece, row, scale);
            }
        }
    }
}
fn difference(output: &mut [f32], weights: &[f32], old: &mut [u32], new: &mut [u32]) {
    old.sort_unstable();
    new.sort_unstable();
    let (mut a, mut b) = (0, 0);
    while a < old.len() || b < new.len() {
        let x = old.get(a).copied().unwrap_or(u32::MAX);
        let y = new.get(b).copied().unwrap_or(u32::MAX);
        if x == y {
            a += 1;
            b += 1;
        } else if x < y {
            add_scaled_row(output, weights, x as usize, -1.);
            a += 1;
        } else {
            add_scaled_row(output, weights, y as usize, 1.);
            b += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evicted_branch_rebuilds_exactly_without_growing_graph_cache() {
        let model = AzNnue::random(96, 8031);
        let mut rng = SplitMix64::new(8125);
        let mut p = Position::startpos();
        let mut arena = NodeFeatureArena::default();
        let mut id = arena.root(&model, &p, Color::Red);
        let mut branch = None;
        for ply in 0..60 {
            let before = p.clone();
            let moves = p.legal_moves();
            let mv = moves[rng.next_u64() as usize % moves.len()];
            p.make_move(mv);
            id = arena.update(
                &model,
                id,
                &before,
                &p,
                Color::Red,
                (1u128 << mv.from) | (1u128 << mv.to),
            );
            if ply == 40 {
                branch = Some((id, p.clone()));
            }
            assert!(arena.cache.len() <= GRAPH_CACHE_SIZE);
            if let Some(cached) = arena.cache.iter().find(|entry| entry.state == id) {
                let full = Graph::new(Board::new(&p));
                let mut expected =
                    encode_motifs(&full, &motifs(&full, u128::MAX, Color::Red), Color::Red);
                let mut actual = cached.rows.rows.clone();
                expected.sort_unstable();
                actual.sort_unstable();
                assert_eq!(actual, expected, "cached motif multiset at ply {ply}");
            }
        }
        let (base, before) = branch.unwrap();
        assert!(!arena.cache.iter().any(|entry| entry.state == base));
        let mut after = before.clone();
        let mv = after.legal_moves()[0];
        after.make_move(mv);
        let refreshes = arena.refreshes;
        let id = arena.update(
            &model,
            base,
            &before,
            &after,
            Color::Red,
            (1u128 << mv.from) | (1u128 << mv.to),
        );
        assert_eq!(arena.refreshes, refreshes + 1);
        let actual = arena.value(id);
        let full_graph = Graph::new(Board::new(&after));
        let expected = model.feature_sums(&full_graph, Color::Red);
        for (a, b) in actual
            .king
            .iter()
            .chain(&actual.relations)
            .chain(&actual.motifs)
            .zip(
                expected
                    .king
                    .iter()
                    .chain(&expected.relations)
                    .chain(&expected.motifs),
            )
        {
            assert!((a - b).abs() <= 1e-4 + 1e-5 * b.abs());
        }
        let cached_graph = arena.take_graph().unwrap();
        assert_eq!(cached_graph.controls, full_graph.controls);
        assert_eq!(cached_graph.incoming, full_graph.incoming);
        assert_eq!(arena.cache.len(), GRAPH_CACHE_SIZE);
        assert_eq!(arena.updates, arena.cached_updates + arena.refreshes);
    }
}
