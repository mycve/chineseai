//! 独立根战术树：仅返回评分，不创建 MCTS 节点、访问或 policy。
use super::*;

const VALUE_BATCH: usize = 256;

struct TacticalNode {
    parent: Option<usize>,
    best: Option<[f32; 3]>,
}

struct TacticalSearch<'a> {
    model: &'a AzNnue,
    control: Option<&'a AzSearchControl>,
    depth: usize,
    draw_score: f32,
    value_scale: f32,
    nodes: Vec<TacticalNode>,
    rows: Vec<usize>,
    hidden: Vec<f32>,
    contexts: Vec<f32>,
}

impl TacticalSearch<'_> {
    fn flush(&mut self) {
        let outputs = self
            .model
            .evaluate_incremental_value_batch(&mut self.hidden, &self.contexts);
        for (&node, eval) in self.rows.iter().zip(outputs) {
            self.nodes[node].best = Some(scale_wdl_value(eval.value_wdl, self.value_scale));
        }
        self.rows.clear();
        self.hidden.clear();
        self.contexts.clear();
    }

    fn collect(
        &mut self,
        position: &Position,
        hidden: &[f32],
        history: &mut Vec<RuleHistoryEntry>,
        parent: Option<usize>,
        ply: usize,
    ) -> Option<usize> {
        if self.control.is_some_and(AzSearchControl::should_stop) {
            return None;
        }
        let node = self.nodes.len();
        self.nodes.push(TacticalNode { parent, best: None });
        if let Some(value) = terminal_value(position, history) {
            self.nodes[node].best = Some(scalar_terminal_wdl(value));
            return Some(node);
        }
        let moves = position.legal_moves_with_rules_and_repetition(history);
        if moves.is_empty() {
            self.nodes[node].best = Some(scalar_terminal_wdl(-1.0));
            return Some(node);
        }
        let checked = position.in_check(position.side_to_move());
        // 非应将节点允许静态评价（stand pat）；应将必须选择合法应手。
        // 深度截断即使正在被将军也只是估值，绝不标记成杀棋证明。
        if !checked || ply >= self.depth {
            self.rows.push(node);
            self.hidden
                .extend_from_slice(AzEvalAccumulator::hidden_for_slice(
                    hidden,
                    self.model.hidden_size,
                    position.side_to_move(),
                ));
            self.contexts
                .extend_from_slice(&rule_context_features(position, history));
            if self.rows.len() == VALUE_BATCH {
                self.flush();
            }
        }
        if ply < self.depth {
            for (mv, _) in moves {
                let moved = position.piece_at(mv.from as usize).unwrap();
                let captured = position.piece_at(mv.to as usize);
                if !checked && captured.is_none() && !position.gives_check_after_move_fast(mv) {
                    continue;
                }
                let mut after = position.clone();
                after.make_move(mv);
                let mut next_hidden = hidden.to_vec();
                AzEvalAccumulator::apply_transition_to_hidden(
                    self.model,
                    position,
                    &after,
                    mv,
                    moved,
                    captured,
                    &mut next_hidden,
                );
                history.push(after.rule_history_entry_after_moved(
                    position.side_to_move(),
                    mv,
                    captured,
                ));
                let completed = self.collect(&after, &next_hidden, history, Some(node), ply + 1);
                history.pop();
                completed?;
            }
        }
        Some(node)
    }

    fn finish(&mut self) {
        self.flush();
        // 后序归约，双方各选最有利的战术回复；不平均对手的反击。
        for index in (0..self.nodes.len()).rev() {
            if let Some(parent) = self.nodes[index].parent {
                let candidate = flip_wdl(self.nodes[index].best.expect("completed tactical node"));
                let best = &mut self.nodes[parent].best;
                if best.is_none_or(|wdl| {
                    wdl_utility(candidate, self.draw_score) > wdl_utility(wdl, self.draw_score)
                }) {
                    *best = Some(candidate);
                }
            }
        }
    }
}

impl AzTree<'_> {
    pub(super) fn prepare_root_tactics(&mut self, control: Option<&AzSearchControl>) {
        self.root_tactics_ready = true;
        if self.root_tactics_depth == 0
            || self.root_tactics_weight == 0.0
            || self.nodes[self.root].solved.is_some()
        {
            return;
        }
        let position = &self.nodes[self.root].position;
        let root_hidden = AzEvalAccumulator::new(self.model, position).into_hidden_sum();
        let mut search = TacticalSearch {
            model: self.model,
            control,
            depth: self.root_tactics_depth,
            draw_score: self.draw_score,
            value_scale: self.value_scale,
            nodes: Vec::new(),
            rows: Vec::with_capacity(VALUE_BATCH),
            hidden: Vec::with_capacity(VALUE_BATCH * self.model.hidden_size),
            contexts: Vec::new(),
        };
        let mut history = self.rule_history_scratch.clone();
        let mut roots = Vec::with_capacity(self.nodes[self.root].children_len as usize);
        for child in self.node_children(self.root) {
            let mv = child.mv;
            let moved = position.piece_at(mv.from as usize).unwrap();
            let captured = position.piece_at(mv.to as usize);
            let mut after = position.clone();
            after.make_move(mv);
            let mut hidden = root_hidden.clone();
            AzEvalAccumulator::apply_transition_to_hidden(
                self.model,
                position,
                &after,
                mv,
                moved,
                captured,
                &mut hidden,
            );
            history.push(after.rule_history_entry_after_moved(
                position.side_to_move(),
                mv,
                captured,
            ));
            let result = search.collect(&after, &hidden, &mut history, None, 1);
            history.pop();
            let Some(root) = result else {
                return;
            }; // 中断整批作废，避免偏向先处理的根走法。
            roots.push(root);
        }
        search.finish();
        if control.is_some_and(AzSearchControl::should_stop) {
            return;
        }
        self.root_tactics_q = roots
            .into_iter()
            .map(|root| wdl_utility(flip_wdl(search.nodes[root].best.unwrap()), self.draw_score))
            .collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_tactics_preserves_statistics_and_real_search_budget() {
        let board = Position::from_fen("4k4/9/9/9/9/4P4/9/9/9/4K4 w").unwrap();
        let model = AzNnue::random(16, 913);
        for size in [1, 16] {
            let limits = AzSearchLimits {
                root_tactics_depth: 8,
                root_tactics_weight: 0.5,
                inference_batch_size: size,
                simulations: 131,
                ..Default::default()
            };
            let mut tree = AzTree::new(
                board.clone(),
                board.initial_rule_history(),
                None,
                &model,
                limits,
            );
            tree.expand(0);
            let priors: Vec<_> = tree.children.iter().map(|edge| edge.prior).collect();
            let policy = tree.root_policy(0);
            let hidden = tree.accumulator_arena.clone();
            let node_count = tree.nodes.len();
            let history = tree.rule_history_scratch.clone();
            tree.prepare_root_tactics(None);
            assert_eq!(tree.nodes.len(), node_count);
            assert_eq!(tree.accumulator_arena, hidden);
            assert_eq!(tree.root_tactics_q.len(), tree.node_children(0).len());
            assert_eq!(tree.root_policy(0), policy);
            assert_eq!(
                tree.children
                    .iter()
                    .map(|edge| edge.prior)
                    .collect::<Vec<_>>(),
                priors
            );
            assert!(tree.children.iter().all(|edge| edge.visits == 0
                && edge.virtual_visits == 0
                && edge.value_wdl_sum == [0.0; 3]));
            assert!(
                tree.nodes
                    .iter()
                    .all(|node| node.visits == 0 && node.value_wdl_sum == [0.0; 3])
            );
            assert_eq!(tree.rule_history_scratch, history);
            let mut used = 0;
            while used < limits.simulations {
                used += tree.simulate_batch(limits.simulations - used, None);
            }
            assert_eq!(tree.nodes[0].visits as usize, used);
            assert_eq!(
                tree.node_children(0)
                    .iter()
                    .map(|edge| edge.visits as usize)
                    .sum::<usize>(),
                used
            );
            assert!(
                tree.nodes
                    .iter()
                    .all(|node| !node.pending && node.virtual_visits == 0)
            );
        }
    }

    #[test]
    fn root_tactics_depth_one_matches_scalar_value_with_rule_context() {
        let board = Position::startpos();
        let mut model = AzNnue::random(16, 912);
        for (i, w) in model.value_head_output.iter_mut().enumerate() {
            *w = (i as f32 % 11.0 - 5.0) * 0.025;
        }
        for (i, w) in model.rule_context_hidden.iter_mut().enumerate() {
            *w = (i as f32 % 9.0 - 4.0) * 0.013;
        }
        let limits = AzSearchLimits {
            root_tactics_depth: 1,
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
        tree.expand(0);
        tree.prepare_root_tactics(None);
        for edge in 0..tree.node_children(0).len() {
            let child = tree.ensure_child_node(0, edge);
            tree.rule_history_scratch
                .push(tree.nodes[child].rule_entry.unwrap());
            let scalar = tree.cutoff_value(child);
            tree.rule_history_scratch.pop();
            let expected = wdl_utility(flip_wdl(scalar.value_wdl), limits.draw_score);
            assert!((tree.root_tactics_q[edge] - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn root_tactics_mate_and_cancellation_do_not_create_mcts_proofs() {
        let board =
            Position::from_fen("2bak2r1/4a4/4b4/p2R4p/4C1n2/2P1c3P/P1r3P2/4B4/4A4/2BK1A2R w")
                .unwrap();
        let model = AzNnue::random(16, 914);
        let mv = board
            .legal_moves()
            .into_iter()
            .find(|mv| mv.to_string() == "d6d9")
            .unwrap();
        let captured = board.piece_at(mv.to as usize);
        let mut after = board.clone();
        after.make_move(mv);
        let mut history = board.initial_rule_history();
        history.push(after.rule_history_entry_after_moved(board.side_to_move(), mv, captured));
        let mut search = TacticalSearch {
            model: &model,
            control: None,
            depth: 8,
            draw_score: 0.3,
            value_scale: 0.65,
            nodes: Vec::new(),
            rows: Vec::new(),
            hidden: Vec::new(),
            contexts: Vec::new(),
        };
        let hidden = AzEvalAccumulator::new(&model, &after).into_hidden_sum();
        search
            .collect(&after, &hidden, &mut history, None, 1)
            .unwrap();
        search.finish();
        assert_eq!(search.nodes[0].best, Some([0.0, 0.0, 1.0]));
        assert_eq!(search.nodes.len(), 1);
        let board = Position::startpos();
        let limits = AzSearchLimits {
            root_tactics_depth: 2,
            ..Default::default()
        };
        let mut tree = AzTree::new(
            board.clone(),
            board.initial_rule_history(),
            None,
            &model,
            limits,
        );
        tree.expand(0);
        let control = AzSearchControl::new(Arc::new(AtomicBool::new(true)), None);
        tree.prepare_root_tactics(Some(&control));
        assert!(tree.root_tactics_q.is_empty());
        assert_eq!(tree.nodes[0].visits, 0);
        assert_eq!(tree.nodes.len(), 1);
        assert!(
            tree.node_children(0)
                .iter()
                .all(|edge| tree.child_solved(edge).is_none())
        );
    }
}
