//! 同一棵树收集叶子；虚拟统计独立存放，仅用于尚未完成的选路。
use super::*;

#[derive(Default)]
pub(super) struct LeafBatchScratch {
    pending: Vec<PendingSimulation>,
    hidden: Vec<f32>,
    contexts: Vec<f32>,
}

#[derive(Default)]
struct PendingSimulation {
    node: usize,
    path: Vec<(usize, usize)>,
    cutoff: bool,
    row: Option<usize>,
}

impl AzTree<'_> {
    pub(super) fn simulate_batch(
        &mut self,
        remaining: usize,
        control: Option<&AzSearchControl>,
    ) -> usize {
        if remaining == 0 || control.is_some_and(AzSearchControl::should_stop) {
            return 0;
        }
        if !self.root_tactics_ready {
            self.prepare_root_tactics(control);
            if control.is_some_and(AzSearchControl::should_stop) {
                return 0;
            }
        }
        if self.inference_batch_size == 1 || self.nodes[self.root].solved.is_some() {
            self.simulate(self.root, 0);
            return 1;
        }
        let mut batch = std::mem::take(&mut self.leaf_batch);
        batch.hidden.clear();
        batch.contexts.clear();
        let mut count = 0;
        for _ in 0..remaining.min(self.inference_batch_size) {
            if control.is_some_and(AzSearchControl::should_stop) {
                break;
            }
            if count == batch.pending.len() {
                batch.pending.push(PendingSimulation::default());
            }
            let pending = &mut batch.pending[count];
            if !self.collect_pending_leaf(pending, &mut batch.hidden, &mut batch.contexts) {
                break;
            }
            count += 1;
            // 终局立即完成回传，避免把同一证明重复塞满 batch。
            if self.nodes[pending.node].solved.is_some() || self.nodes[self.root].solved.is_some() {
                break;
            }
        }
        let outputs = self
            .model
            .evaluate_incremental_value_batch(&mut batch.hidden, &batch.contexts);
        for pending in &batch.pending[..count] {
            if let Some(row) = pending.row {
                let eval = outputs[row];
                let node = &mut self.nodes[pending.node];
                node.value = eval.value * self.value_scale;
                node.value_wdl = scale_wdl_value(eval.value_wdl, self.value_scale);
                node.value_cached = true;
            }
        }
        // 在任何真实回传、终局修正与快照之前，先撤销整批虚拟统计。
        for pending in &batch.pending[..count] {
            self.reserve_pending(pending, false);
        }
        let history_len = self.rule_history_scratch.len();
        for pending in &batch.pending[..count] {
            for &(parent, edge) in &pending.path {
                let child = self.node_children(parent)[edge].child_node().unwrap();
                self.rule_history_scratch
                    .push(self.nodes[child].rule_entry.unwrap());
            }
            let mut eval = if pending.cutoff {
                self.node_eval(pending.node)
            } else {
                let hidden = pending.row.map(|row| {
                    let start = row * self.model.hidden_size;
                    &batch.hidden[start..start + self.model.hidden_size]
                });
                self.expand_with_hidden(pending.node, hidden)
            };
            self.add_node_visit(pending.node, eval);
            self.record_leaf_depth(pending.path.len(), pending.cutoff);
            for &(parent, edge) in pending.path.iter().rev() {
                eval = AzEvalOutput {
                    value: -eval.value,
                    value_wdl: flip_wdl(eval.value_wdl),
                };
                let child = &mut self.node_children_mut(parent)[edge];
                child.visits += 1;
                add_wdl(&mut child.value_wdl_sum, eval.value_wdl);
                self.update_solved(parent);
                if self.nodes[parent].solved.is_some() {
                    eval = self.node_eval(parent);
                }
                self.add_node_visit(parent, eval);
            }
            self.rule_history_scratch.truncate(history_len);
        }
        self.leaf_batch = batch;
        count
    }

    fn collect_pending_leaf(
        &mut self,
        pending: &mut PendingSimulation,
        hidden: &mut Vec<f32>,
        contexts: &mut Vec<f32>,
    ) -> bool {
        pending.path.clear();
        pending.row = None;
        pending.cutoff = false;
        let history_len = self.rule_history_scratch.len();
        let mut node = self.root;
        loop {
            let depth = pending.path.len();
            if self.nodes[node].solved.is_some() {
                break;
            }
            let cutoff = depth >= self.max_depth;
            if cutoff || !self.nodes[node].expanded {
                if !self.nodes[node].value_cached {
                    self.prepare_batch_leaf(node, cutoff);
                }
                if self.nodes[node].solved.is_some() {
                    break;
                }
                // 与原始 simulate 保持一致：被将军和唯一应手继续搜索。
                let checked = self.nodes[node]
                    .rule_entry
                    .is_some_and(|entry| entry.gives_check);
                if !cutoff && (checked || self.nodes[node].children_len == 1) {
                    self.expand(node);
                    if self.nodes[node].solved.is_some() {
                        break;
                    }
                } else {
                    pending.cutoff = cutoff;
                    if !self.nodes[node].value_cached {
                        pending.row = Some(hidden.len() / self.model.hidden_size);
                        let start = self.nodes[node].accumulator_offset as usize;
                        hidden.extend_from_slice(
                            &self.accumulator_arena[start..start + self.model.hidden_size],
                        );
                        contexts.extend_from_slice(&rule_context_features(
                            &self.nodes[node].position,
                            &self.rule_history_scratch,
                        ));
                    }
                    break;
                }
            }
            if self.nodes[node].children_len == 0 {
                break;
            }
            let Some(edge) = self.select_available_child(node, true) else {
                self.rule_history_scratch.truncate(history_len);
                return false;
            };
            let child = self.ensure_child_node(node, edge);
            pending.path.push((node, edge));
            self.rule_history_scratch
                .push(self.nodes[child].rule_entry.unwrap());
            node = child;
        }
        pending.node = node;
        self.reserve_pending(pending, true);
        self.rule_history_scratch.truncate(history_len);
        true
    }

    fn reserve_pending(&mut self, pending: &PendingSimulation, add: bool) {
        self.nodes[pending.node].pending = add;
        let update = |visits: &mut u32| {
            if add {
                *visits += 1;
            } else {
                *visits -= 1;
            }
        };
        update(&mut self.nodes[pending.node].virtual_visits);
        for &(parent, edge) in &pending.path {
            update(&mut self.nodes[parent].virtual_visits);
            update(&mut self.node_children_mut(parent)[edge].virtual_visits);
        }
    }

    fn prepare_batch_leaf(&mut self, node: usize, cutoff: bool) {
        let terminal = terminal_value(&self.nodes[node].position, &self.rule_history_scratch);
        let moves = if terminal.is_none() {
            self.nodes[node]
                .position
                .legal_moves_with_rules_and_repetition(&self.rule_history_scratch)
        } else {
            Vec::new()
        };
        if terminal.is_some() || moves.is_empty() {
            let value = terminal.unwrap_or(-1.0);
            let node = &mut self.nodes[node];
            node.value = value;
            node.value_wdl = scalar_terminal_wdl(value);
            node.value_cached = true;
            node.expanded = true;
            node.solved = Some(value as i8);
        } else if !cutoff {
            self.set_node_children(
                node,
                moves.into_iter().map(|(mv, _)| AzChild {
                    mv,
                    prior: 0.0,
                    visits: 0,
                    virtual_visits: 0,
                    value_wdl_sum: [0.0; 3],
                    child: NO_CHILD,
                }),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaf_batch_removes_virtual_statistics_and_respects_real_budget() {
        let board = Position::startpos();
        let mut model = AzNnue::random(16, 911);
        for (i, w) in model.value_head_output.iter_mut().enumerate() {
            *w = (i as f32 % 11.0 - 5.0) * 0.025;
        }
        for (i, w) in model.rule_context_hidden.iter_mut().enumerate() {
            *w = (i as f32 % 9.0 - 4.0) * 0.013;
        }
        for (i, w) in model.policy_move_context.iter_mut().enumerate() {
            *w = (i as f32 % 17.0 - 8.0) * 0.011;
        }
        for size in [1, 4, 8, 16] {
            let limits = AzSearchLimits {
                simulations: 131,
                inference_batch_size: size,
                value_scale: 0.65,
                draw_score: 0.3,
                ..Default::default()
            };
            let mut tree = AzTree::new(
                board.clone(),
                board.initial_rule_history(),
                None,
                &model,
                limits,
            );
            tree.expand(tree.root);
            let mut used = 0;
            while used < limits.simulations {
                let completed = tree.simulate_batch(limits.simulations - used, None);
                assert!(completed > 0 && completed <= size);
                used += completed;
                assert!(
                    tree.nodes
                        .iter()
                        .all(|node| !node.pending && node.virtual_visits == 0)
                );
                assert!(tree.children.iter().all(|edge| edge.virtual_visits == 0));
                assert_eq!(tree.nodes[tree.root].visits as usize, used);
                assert_eq!(
                    tree.node_children(tree.root)
                        .iter()
                        .map(|edge| edge.visits as usize)
                        .sum::<usize>(),
                    used
                );
                assert_eq!(tree.rule_history_scratch.len(), 1);
            }
            let result = tree.search_result(used);
            assert_eq!(result.simulations, 131);
            assert!((result.candidates.iter().map(|c| c.policy).sum::<f32>() - 1.0).abs() < 1e-5);
            for (node, state) in tree.nodes.iter().enumerate().skip(1) {
                if !state.value_cached || state.solved.is_some() {
                    continue;
                }
                let mut history = Vec::new();
                let mut current = node;
                while tree.nodes[current].parent != NO_CHILD {
                    history.push(tree.nodes[current].rule_entry.unwrap());
                    current = tree.nodes[current].parent as usize;
                }
                history.push(board.initial_rule_history()[0]);
                history.reverse();
                let moves = state.position.legal_moves_with_rules(&history);
                let expected = scale_wdl_value(
                    model.evaluate_wdl_with_rules(&state.position, &history, &moves),
                    limits.value_scale,
                );
                for i in 0..3 {
                    assert!((state.value_wdl[i] - expected[i]).abs() < 1e-4);
                }
                if state.expanded {
                    let flags: Vec<_> = moves
                        .iter()
                        .map(|&mv| u8::from(state.position.move_repeats_history(&history, mv)))
                        .collect();
                    let mut scratch = AzEvalScratch::new(model.arch);
                    model.evaluate_with_scratch_output_with_repetition(
                        &state.position,
                        &moves,
                        &flags,
                        &rule_context_features(&state.position, &history),
                        &mut scratch,
                    );
                    let mut priors = Vec::new();
                    softmax_into(&scratch.logits, limits.policy_softmax_temp, &mut priors);
                    for (edge, prior) in tree.node_children(node).iter().zip(priors) {
                        assert!((edge.prior - prior).abs() < 1e-4);
                    }
                }
            }
        }
    }

    #[test]
    fn leaf_batch_stop_has_no_extra_visits_and_mate_proof_is_exact() {
        let model = AzNnue::random(16, 912);
        let board = Position::startpos();
        let control = AzSearchControl::new(Arc::new(AtomicBool::new(true)), None);
        let result = alphazero_search_with_rules_controlled(
            &board,
            None,
            None,
            &model,
            AzSearchLimits {
                inference_batch_size: 16,
                ..Default::default()
            },
            Some(&control),
        );
        assert_eq!(result.simulations, 0);
        assert_eq!(result.candidates.iter().map(|c| c.visits).sum::<u32>(), 0);
        let mate = Position::from_fen(
            "2bak2r1/4a4/4b4/p2R4p/4C1n2/2P1c3P/P1r3P2/4B4/4A4/2BK1A2R w - - 1 1",
        )
        .unwrap();
        let result = alphazero_search(
            &mate,
            &model,
            AzSearchLimits {
                inference_batch_size: 16,
                ..Default::default()
            },
        );
        assert_eq!(result.best_move, Some(Move::from_uci("d6d9").unwrap()));
        assert_eq!(result.value_wdl, [1.0, 0.0, 0.0]);
    }
}
