//! MCTS节点仅保存低维累加和；关系工作图不复制到节点。
use super::{features::*, *};
use crate::xiangqi::{Color, Position};

#[derive(Default)]
pub(crate) struct NodeFeatureArena {
    states: Vec<State>,
    working: Option<Graph>,
    roots: [Option<u32>; 2],
    root_graph: Option<Graph>,
    root_rows: [Option<Box<[FeatureRows; 90]>>; 2],
}
struct State {
    sums: FeatureSums,
    age: u8,
}
impl NodeFeatureArena {
    pub fn heap_bytes(&self) -> usize {
        self.states.capacity() * std::mem::size_of::<State>()
            + self
                .root_rows
                .iter()
                .flatten()
                .map(|rows| {
                    std::mem::size_of::<[FeatureRows; 90]>()
                        + rows
                            .iter()
                            .map(|r| (r.relations.capacity() + r.motifs.capacity()) * 4)
                            .sum::<usize>()
                })
                .sum::<usize>()
    }
    pub fn clear(&mut self) {
        self.states.clear();
        self.working = None;
        self.roots = [None; 2];
        self.root_graph = None;
        self.root_rows = [None, None];
    }
    pub fn take_graph(&mut self) -> Option<Graph> {
        self.working.take()
    }
    pub fn root(&mut self, model: &AzNnue, p: &Position, side: Color) -> u32 {
        let index = crate::xiangqi::color_index(side);
        if let Some(id) = self.roots[index] {
            return id;
        }
        let graph = self
            .root_graph
            .get_or_insert_with(|| Graph::new(Board::new(p)));
        let sums = model.feature_sums(graph, side);
        let events = motifs(graph, u128::MAX, side);
        let rows = Box::new(std::array::from_fn(|source| {
            let events = events
                .iter()
                .filter(|e| e.root as usize == source)
                .copied()
                .collect::<Vec<_>>();
            FeatureRows {
                relations: relation_rows(graph, side, 1u128 << source),
                motifs: encode_motifs(graph, &events, side),
            }
        }));
        self.root_rows[index] = Some(rows);
        self.working = Some(graph.clone());
        let id = self.store(sums, 0);
        self.roots[index] = Some(id);
        id
    }
    pub fn value(&self, index: u32) -> FeatureSums {
        self.states[index as usize].sums
    }
    fn store(&mut self, sums: FeatureSums, age: u8) -> u32 {
        let index = self.states.len() as u32;
        self.states.push(State { sums, age });
        index
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
        let state = &self.states[base as usize];
        let age = state.age + 2;
        let new_graph = Graph::new(Board::new(after));
        let pieces = new_graph.board.occupied.count_ones();
        let estimated =
            (new_graph.estimated_dirty_sources(changed) & new_graph.board.occupied).count_ones();
        // 跨两步导致大量潜在源失效时，直接刷新可避免为一次差分重建祖先图。
        if age >= 32 || pieces < 12 || (changed.count_ones() > 2 && estimated * 6 >= pieces) {
            let sums = model.feature_sums(&new_graph, side);
            self.working = Some(new_graph);
            return self.store(sums, 0);
        }
        let root_index = self.roots.iter().position(|&root| root == Some(base));
        let owned_old = if root_index.is_some() {
            None
        } else {
            Some(Graph::new(Board::new(before)))
        };
        let old_graph = owned_old
            .as_ref()
            .unwrap_or_else(|| self.root_graph.as_ref().unwrap());
        let cached = root_index.map(|i| self.root_rows[i].as_ref().unwrap());
        let (sources, roots) = dirty_roots(old_graph, &new_graph, changed);
        let mut sums = state.sums;
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
            for (board, bucket, scale) in [
                (&old_graph.board, old_bucket, -1.),
                (&new_graph.board, new_bucket, 1.),
            ] {
                for square in bits(board.occupied & mask) {
                    let piece = board.cells[square].unwrap();
                    let class = nnue::piece_absolute_feature_index(side, piece);
                    let row = ((half * 9 + bucket) * 14 + class) * 90
                        + nnue::canonical_square(side, square);
                    add_scaled_row(&mut sums.king, &model.king_piece, row, scale);
                }
            }
        }
        let old_relations = if let Some(rows) = cached {
            bits(sources)
                .flat_map(|s| rows[s].relations.iter().copied())
                .collect()
        } else {
            relation_rows(old_graph, side, sources)
        };
        difference(
            &mut sums.relations,
            &model.relation_embedding,
            old_relations,
            relation_rows(&new_graph, side, sources),
        );
        let ratio = if cached.is_some() { 4 } else { 2 };
        if (roots & new_graph.board.occupied).count_ones() * ratio
            > new_graph.board.occupied.count_ones() * (ratio - 1)
        {
            sums.motifs.fill(0.);
            for row in encode_motifs(&new_graph, &motifs(&new_graph, u128::MAX, side), side) {
                add_scaled_row(&mut sums.motifs, &model.motif_embedding, row as usize, 1.);
            }
        } else {
            let old_motifs = if let Some(rows) = cached {
                bits(roots)
                    .flat_map(|s| rows[s].motifs.iter().copied())
                    .collect()
            } else {
                encode_motifs(old_graph, &motifs(old_graph, roots, side), side)
            };
            difference(
                &mut sums.motifs,
                &model.motif_embedding,
                old_motifs,
                encode_motifs(&new_graph, &motifs(&new_graph, roots, side), side),
            );
        }
        self.working = Some(new_graph);
        self.store(sums, age)
    }
}
fn difference(output: &mut [f32], weights: &[f32], mut old: Vec<u32>, mut new: Vec<u32>) {
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
