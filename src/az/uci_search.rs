//! UCI 独占的树保留和 PV 提取，不改变自博弈搜索路径。
use super::*;
use std::collections::BinaryHeap;

const MAX_RETAINED_NODES: usize = 100_000;

#[derive(Default)]
pub(crate) struct AzUciSearchCache {
    retained: Option<RetainedTree>,
}

struct RetainedTree {
    nodes: Vec<AzNode>,
    children: Vec<AzChild>,
    accumulators: Vec<f32>,
    history: Vec<RuleHistoryEntry>,
    limits: AzSearchLimits,
    model_identity: usize,
}

pub(crate) struct AzUciPv {
    pub moves: Vec<Move>,
    pub wdl: [f32; 3],
    pub q: f32,
    pub proven: Option<i8>,
}

pub(crate) struct AzUciSearchResult {
    pub search: AzSearchResult,
    pub variations: Vec<AzUciPv>,
    pub reused_visits: u32,
}

impl AzUciSearchCache {
    pub(crate) fn clear(&mut self) {
        self.retained = None;
    }

    fn restore(&mut self, tree: &mut AzTree<'_>, limits: AzSearchLimits) -> u32 {
        let Some(old) = self.retained.take() else {
            return 0;
        };
        let mut previous_limits = old.limits;
        previous_limits.simulations = limits.simulations;
        previous_limits.seed = limits.seed;
        if previous_limits != limits
            || old.model_identity != tree.model as *const AzNnue as usize
            || !tree.rule_history_scratch.starts_with(&old.history)
        {
            return 0;
        }
        let mut root = 0;
        for entry in &tree.rule_history_scratch[old.history.len()..] {
            let node = &old.nodes[root];
            let Some(next) = old.children[node.children_offset as usize
                ..node.children_offset as usize + node.children_len as usize]
                .iter()
                .filter_map(AzChild::child_node)
                .find(|&index| old.nodes[index].rule_entry.as_ref() == Some(entry))
            else {
                return 0;
            };
            root = next;
        }
        let node = &old.nodes[root];
        if node.position != tree.nodes[0].position
            || !node.expanded
            || terminal_value(&node.position, &tree.rule_history_scratch).is_some()
            || node.visits
                >= u32::MAX.saturating_sub(limits.simulations.min(u32::MAX as usize - 1) as u32)
        {
            return 0;
        }
        let old_children = &old.children[node.children_offset as usize
            ..node.children_offset as usize + node.children_len as usize];
        let legal = tree.root_moves.as_ref().expect("UCI supplies root moves");
        if old_children.len() != legal.len()
            || old_children.iter().any(|child| !legal.contains(&child.mv))
        {
            return 0;
        }

        // 仅复制可达子树，重排父子索引；新根双视角累加器由当前局面重建。
        let root_offset = tree.nodes[0].accumulator_offset;
        let root_policy = tree.nodes[0].policy_accumulator;
        tree.nodes.clear();
        let mut queue = vec![(root, NO_CHILD)];
        let mut cursor = 0;
        while cursor < queue.len() {
            let (old_index, parent) = queue[cursor];
            let mut node = old.nodes[old_index].clone();
            node.parent = parent;
            if cursor == 0 {
                node.accumulator_offset = root_offset;
                node.policy_accumulator = root_policy;
                node.incoming_move = None;
                node.rule_entry = None;
            } else {
                let start = node.accumulator_offset as usize;
                node.accumulator_offset = tree.accumulator_arena.len() as u32;
                tree.accumulator_arena
                    .extend_from_slice(&old.accumulators[start..start + tree.model.hidden_size]);
            }
            let start = node.children_offset as usize;
            node.children_offset = tree.children.len() as u32;
            for child in &old.children[start..start + node.children_len as usize] {
                let mut child = child.clone();
                if let Some(index) = child.child_node() {
                    child.set_child_node(queue.len());
                    queue.push((index, cursor as u32));
                }
                tree.children.push(child);
            }
            tree.nodes.push(node);
            cursor += 1;
        }
        let temperature = tree.policy_softmax_temp;
        tree.root_raw_priors = tree
            .node_children(0)
            .iter()
            .map(|child| child.prior.powf(temperature))
            .collect();
        let sum: f32 = tree.root_raw_priors.iter().sum();
        for prior in &mut tree.root_raw_priors {
            *prior /= sum.max(f32::MIN_POSITIVE);
        }
        tree.root_moves = None;
        tree.nodes[0].visits
    }
}

impl AzTree<'_> {
    fn compact_uci_tree(&mut self, keep: usize) {
        let mut selected = Vec::with_capacity(keep);
        let mut map = vec![NO_CHILD; self.nodes.len()];
        let mut queue = BinaryHeap::new();
        queue.push((u32::MAX, u32::MAX, 0usize));
        while selected.len() < keep {
            let Some((_, depth_priority, index)) = queue.pop() else {
                break;
            };
            map[index] = selected.len() as u32;
            selected.push(index);
            for child in self.node_children(index) {
                if let Some(next) = child.child_node() {
                    let priority = if self.child_solved(child).is_some() {
                        u32::MAX
                    } else {
                        child.visits
                    };
                    queue.push((priority, depth_priority - 1, next));
                }
            }
        }
        let old_nodes = std::mem::take(&mut self.nodes);
        let old_children = std::mem::take(&mut self.children);
        let old_accumulators = std::mem::take(&mut self.accumulator_arena);
        self.nodes = Vec::with_capacity(selected.len());
        self.children = Vec::new();
        self.accumulator_arena = old_accumulators[..2 * self.model.hidden_size].to_vec();
        for (new_index, old_index) in selected.into_iter().enumerate() {
            let mut node = old_nodes[old_index].clone();
            node.parent = if new_index == 0 {
                NO_CHILD
            } else {
                map[node.parent as usize]
            };
            if new_index > 0 {
                let start = node.accumulator_offset as usize;
                node.accumulator_offset = self.accumulator_arena.len() as u32;
                self.accumulator_arena
                    .extend_from_slice(&old_accumulators[start..start + self.model.hidden_size]);
            }
            let start = node.children_offset as usize;
            node.children_offset = self.children.len() as u32;
            for child in &old_children[start..start + node.children_len as usize] {
                let mut child = child.clone();
                if let Some(next) = child.child_node() {
                    child.child = map[next];
                }
                self.children.push(child);
            }
            self.nodes.push(node);
        }
    }

    fn uci_snapshot(&self, used: usize, multipv: usize, reused_visits: u32) -> AzUciSearchResult {
        let search = self.search_result(used);
        let mut ranked = search.candidates.iter().collect::<Vec<_>>();
        ranked.sort_by(|a, b| {
            b.proof_priority()
                .cmp(&a.proof_priority())
                .then_with(|| b.visits.cmp(&a.visits))
                .then_with(|| b.q.total_cmp(&a.q))
        });
        if let Some(best) = search.best_move {
            if let Some(index) = ranked.iter().position(|candidate| candidate.mv == best) {
                let first = ranked.remove(index);
                ranked.insert(0, first);
            }
        }
        let variations = ranked
            .into_iter()
            .take(multipv.max(1))
            .map(|candidate| {
                let child = self
                    .node_children(self.root)
                    .iter()
                    .find(|child| child.mv == candidate.mv)
                    .unwrap();
                let wdl = if let Some(value) = self.child_solved(child) {
                    scalar_terminal_wdl(value as f32)
                } else if child.visits > 0 {
                    child.value_wdl_sum.map(|v| v / child.visits as f32)
                } else {
                    search.value_wdl
                };
                let mut moves = vec![child.mv];
                let mut next = child.child_node();
                while moves.len() < self.max_depth.min(128) {
                    let Some(node) = next else { break };
                    let Some(index) = self.best_root_child(node) else {
                        break;
                    };
                    let child = &self.node_children(node)[index];
                    moves.push(child.mv);
                    next = child.child_node();
                }
                AzUciPv {
                    moves,
                    wdl,
                    q: wdl_utility(wdl, self.draw_score),
                    proven: self.child_solved(child),
                }
            })
            .collect();
        AzUciSearchResult {
            search,
            variations,
            reused_visits,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn search_uci(
    position: &Position,
    history: Vec<RuleHistoryEntry>,
    root_moves: Vec<Move>,
    model: &AzNnue,
    limits: AzSearchLimits,
    control: &AzSearchControl,
    cache: &mut AzUciSearchCache,
    multipv: usize,
    retain: bool,
    mut progress: impl FnMut(&AzUciSearchResult),
) -> AzUciSearchResult {
    let mut tree = AzTree::new(
        position.clone(),
        history.clone(),
        Some(root_moves),
        model,
        limits,
    );
    tree.adjudicate_root_rules = false;
    if !retain {
        cache.clear();
    }
    let reused = cache.restore(&mut tree, limits);
    if !tree.nodes[0].expanded {
        tree.expand(0);
    }
    let mut used = 0;
    let mut last_progress = Instant::now();
    if tree.nodes[0].children_len > 0 {
        for _ in 0..limits.simulations {
            if control.should_stop() {
                break;
            }
            if tree.nodes.len() >= MAX_RETAINED_NODES {
                tree.compact_uci_tree(MAX_RETAINED_NODES / 2);
            }
            tree.simulate(0, 0);
            used += 1;
            if tree.nodes[0].solved.is_some() {
                break;
            }
            if used % SEARCH_PROGRESS_POLL_SIMULATIONS == 0
                && last_progress.elapsed() >= SEARCH_PROGRESS_INTERVAL
            {
                progress(&tree.uci_snapshot(used, multipv, reused));
                last_progress = Instant::now();
            }
        }
    }
    let result = tree.uci_snapshot(used, multipv, reused);
    if tree.nodes.len() >= MAX_RETAINED_NODES {
        tree.compact_uci_tree(MAX_RETAINED_NODES / 2);
    }
    if retain {
        cache.retained = Some(RetainedTree {
            nodes: tree.nodes,
            children: tree.children,
            accumulators: tree.accumulator_arena,
            history,
            limits,
            model_identity: model as *const AzNnue as usize,
        });
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(
        position: &Position,
        history: Vec<RuleHistoryEntry>,
        model: &AzNnue,
        limits: AzSearchLimits,
        cache: &mut AzUciSearchCache,
    ) -> AzUciSearchResult {
        let control = AzSearchControl::new(Arc::new(AtomicBool::new(false)), None);
        search_uci(
            position,
            history.clone(),
            position.legal_moves_with_rules(&history),
            model,
            limits,
            &control,
            cache,
            4,
            true,
            |_| {},
        )
    }

    fn validate_pvs(position: &Position, history: &[RuleHistoryEntry], report: &AzUciSearchResult) {
        assert_eq!(report.variations.len(), 4);
        assert_eq!(
            report.variations[0].moves[0],
            report.search.best_move.unwrap()
        );
        let mut first_moves = Vec::new();
        for pv in &report.variations {
            assert!(!first_moves.contains(&pv.moves[0]));
            first_moves.push(pv.moves[0]);
            let mut position = position.clone();
            let mut history = history.to_vec();
            for &mv in &pv.moves {
                assert!(position.legal_moves_with_rules(&history).contains(&mv));
                let entry = position.rule_history_entry_after_move(mv);
                position.make_move(mv);
                history.push(entry);
            }
        }
    }

    #[test]
    fn compacted_tree_keeps_legal_pvs_and_can_continue_searching() {
        let model = AzNnue::random(16, 20260928);
        let position = Position::startpos();
        let history = position.initial_rule_history();
        let limits = AzSearchLimits {
            simulations: 1024,
            ..Default::default()
        };
        let mut tree = AzTree::new(
            position.clone(),
            history.clone(),
            Some(position.legal_moves_with_rules(&history)),
            &model,
            limits,
        );
        tree.adjudicate_root_rules = false;
        tree.expand(0);
        for _ in 0..1024 {
            tree.simulate(0, 0);
        }
        let before = tree.nodes[0].visits;
        let original_nodes = tree.nodes.len();
        assert!(original_nodes > 32);
        tree.compact_uci_tree(original_nodes / 2);
        assert_eq!(tree.nodes.len(), original_nodes / 2);
        assert_eq!(tree.nodes[0].visits, before);
        for (index, node) in tree.nodes.iter().enumerate() {
            if index > 0 {
                assert!((node.parent as usize) < index);
            }
            let expected = AzEvalAccumulator::new(&model, &node.position).into_hidden_sum();
            let perspective = color_index(node.position.side_to_move()) * model.hidden_size;
            let offset = node.accumulator_offset as usize;
            for (actual, expected) in tree.accumulator_arena[offset..offset + model.hidden_size]
                .iter()
                .zip(&expected[perspective..perspective + model.hidden_size])
            {
                assert!((actual - expected).abs() < 1e-5);
            }
            for child in tree.node_children(index) {
                if let Some(next) = child.child_node() {
                    assert_eq!(tree.nodes[next].parent as usize, index);
                }
            }
        }
        for _ in 0..128 {
            tree.simulate(0, 0);
        }
        assert!(tree.nodes[0].visits > before);
        validate_pvs(&position, &history, &tree.uci_snapshot(128, 4, 0));
    }

    #[test]
    fn multipv_and_reuse_preserve_history_accumulators_and_new_budget() {
        let model = AzNnue::random(16, 20260928);
        let mut position = Position::startpos();
        let mut history = position.initial_rule_history();
        let mut cache = AzUciSearchCache::default();
        let mut limits = AzSearchLimits {
            simulations: 512,
            ..Default::default()
        };
        let first = run(&position, history.clone(), &model, limits, &mut cache);
        let fresh = alphazero_search_external_root_controlled_with_progress(
            &position,
            Some(history.clone()),
            Some(position.legal_moves_with_rules(&history)),
            &model,
            limits,
            None,
            None,
        );
        assert_eq!(first.search.best_move, fresh.best_move);
        assert_eq!(first.search.value_wdl, fresh.value_wdl);
        for (actual, expected) in first.search.candidates.iter().zip(&fresh.candidates) {
            assert_eq!(actual.mv, expected.mv);
            assert_eq!(actual.visits, expected.visits);
            assert_eq!(actual.q, expected.q);
        }
        validate_pvs(&position, &history, &first);
        assert_eq!(first.reused_visits, 0);
        assert_eq!(first.search.simulations, 512);
        assert!(first.variations[0].moves.len() >= 2);
        limits.simulations = 32;
        let repeated = run(&position, history.clone(), &model, limits, &mut cache);
        assert_eq!(repeated.reused_visits, 512);
        assert_eq!(repeated.search.simulations, 32);
        let continuation = repeated.variations[0]
            .moves
            .iter()
            .take(2)
            .copied()
            .collect::<Vec<_>>();
        for mv in continuation {
            history.push(position.rule_history_entry_after_move(mv));
            position.make_move(mv);
            let step = run(&position, history.clone(), &model, limits, &mut cache);
            assert!(step.reused_visits > 0);
            assert_eq!(step.search.simulations, 32);
            validate_pvs(&position, &history, &step);
        }
        let promoted = run(&position, history.clone(), &model, limits, &mut cache);
        assert!(promoted.reused_visits > 0);
        assert_eq!(promoted.search.simulations, 32);
        validate_pvs(&position, &history, &promoted);
        let retained = cache.retained.as_ref().unwrap();
        for node in &retained.nodes {
            let full = AzEvalAccumulator::new(&model, &node.position).into_hidden_sum();
            let perspective = color_index(node.position.side_to_move()) * model.hidden_size;
            let offset = node.accumulator_offset as usize;
            for (actual, expected) in retained.accumulators[offset..offset + model.hidden_size]
                .iter()
                .zip(&full[perspective..perspective + model.hidden_size])
            {
                assert!((actual - expected).abs() < 1e-5);
            }
        }
        // 同一棋盘缺少走子历史，不得复用；参数改变也必须重建。
        let no_history = run(
            &position,
            position.initial_rule_history(),
            &model,
            limits,
            &mut cache,
        );
        assert_eq!(no_history.reused_visits, 0);
        limits.cpuct += 0.1;
        let changed = run(
            &position,
            position.initial_rule_history(),
            &model,
            limits,
            &mut cache,
        );
        assert_eq!(changed.reused_visits, 0);
        let other_model = model.clone();
        let changed_model = run(
            &position,
            position.initial_rule_history(),
            &other_model,
            limits,
            &mut cache,
        );
        assert_eq!(changed_model.reused_visits, 0);
        let control = AzSearchControl::new(Arc::new(AtomicBool::new(false)), None);
        let restricted = search_uci(
            &position,
            position.initial_rule_history(),
            position.legal_moves()[..2].to_vec(),
            &other_model,
            limits,
            &control,
            &mut cache,
            2,
            false,
            |_| {},
        );
        assert_eq!(restricted.reused_visits, 0);
        assert_eq!(restricted.variations.len(), 2);
        assert!(cache.retained.is_none());
        cache.clear();
        assert!(cache.retained.is_none());
    }
}
