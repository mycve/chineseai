//! 只复用实际已走分支的网络结果，每步搜索统计与证明均重新建立。
use super::*;

pub(super) struct CachedTree {
    nodes: Vec<AzNode>,
    edges: Vec<AzChild>,
    pub(super) root: usize,
}

fn signature(mut limits: AzSearchLimits) -> AzSearchLimits {
    if limits.max_depth == 0 {
        limits.max_depth = limits.simulations;
    }
    limits.seed = 0;
    limits.simulations = 0;
    limits.root_dirichlet_alpha = 0.0;
    limits.root_exploration_fraction = 0.0;
    limits
}

pub(super) fn take_cached(
    position: &Position,
    history: &[RuleHistoryEntry],
    limits: AzSearchLimits,
    workspace: &mut AzSearchWorkspace<'_>,
) -> Option<CachedTree> {
    let previous = workspace.previous_limits.replace(signature(limits));
    if previous != Some(signature(limits)) || limits.draw_score != 0.0 || workspace.nodes.is_empty()
    {
        return None;
    }
    let old_history = &workspace.rule_history_scratch;
    if history.len() != old_history.len() + 1 || !history.starts_with(old_history) {
        return None;
    }
    let n = &workspace.nodes[0];
    let root = workspace.children
        [n.children_offset as usize..n.children_offset as usize + n.children_len as usize]
        .iter()
        .filter_map(AzChild::child_node)
        .find(|&id| {
            let n = &workspace.nodes[id];
            n.position == *position && n.rule_entry == history.last().copied()
        })?;
    Some(CachedTree {
        nodes: std::mem::take(&mut workspace.nodes),
        edges: std::mem::take(&mut workspace.children),
        root,
    })
}

impl CachedTree {
    pub(super) fn matching_child(
        &self,
        parent: u32,
        child_index: usize,
        mv: Move,
        position: &Position,
        entry: RuleHistoryEntry,
    ) -> u32 {
        let Some(n) = self.nodes.get(parent as usize) else {
            return NO_CHILD;
        };
        if child_index >= n.children_len as usize {
            return NO_CHILD;
        }
        self.edges
            .get(n.children_offset as usize + child_index)
            .filter(|edge| edge.mv == mv)
            .and_then(AzChild::child_node)
            .filter(|&id| {
                self.nodes[id].position == *position && self.nodes[id].rule_entry == Some(entry)
            })
            .map_or(NO_CHILD, |id| id as u32)
    }
    pub(super) fn network(
        &self,
        id: u32,
        position: &Position,
        moves: &[Move],
    ) -> Option<(AzEvalOutput, &[AzChild])> {
        let n = self.nodes.get(id as usize)?;
        // 模型、完整路径历史与价值缩放均相同，启用战术搜索时不使用缓存。
        // 已证明节点的值可能被证明覆盖，不能当成原网络结果复用。
        if !n.expanded
            || n.solved.is_some()
            || n.position != *position
            || n.children_len as usize != moves.len()
        {
            return None;
        }
        let edges = &self.edges
            [n.children_offset as usize..n.children_offset as usize + n.children_len as usize];
        if !edges.iter().zip(moves).all(|(edge, mv)| edge.mv == *mv) {
            return None;
        }
        Some((
            AzEvalOutput {
                value: n.value,
                value_wdl: n.value_wdl,
                moves_left: n.moves_left,
            },
            edges,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compare(
        p: &Position,
        h: &[RuleHistoryEntry],
        limits: AzSearchLimits,
        workspace: &mut AzSearchWorkspace<'_>,
    ) -> AzSearchResult {
        let moves = p.legal_moves_with_rules(h);
        let cold = alphazero_search_with_rules(
            p,
            Some(h.to_vec()),
            Some(moves.clone()),
            workspace.model,
            limits,
        );
        let cached = alphazero_search_with_rules_reusing(p, h, moves, limits, workspace);
        assert_eq!(cached.best_move, cold.best_move);
        assert_eq!(cached.simulations, cold.simulations);
        assert_eq!(cached.candidates.len(), cold.candidates.len());
        for (a, b) in cached.network_value_wdl.iter().zip(cold.network_value_wdl) {
            assert!((a - b).abs() <= 1e-6);
        }
        for (a, b) in cached.candidates.iter().zip(&cold.candidates) {
            assert_eq!((a.mv, a.visits, a.policy), (b.mv, b.visits, b.policy));
            assert!((a.raw_prior - b.raw_prior).abs() <= 1e-6);
            assert!((a.q - b.q).abs() <= 1e-6);
        }
        cached
    }

    #[test]
    fn eval_cache_keeps_cold_search_with_noise_kld_mate_and_history() {
        let mut model = AzNnue::random(8, 31);
        model.mate_search_plies = 9;
        for i in 48..96 {
            model.value_history_output[i] = 0.7;
        }
        model.rebuild_value_history();
        let mut workspace = AzSearchWorkspace::new(&model);
        let mut p = Position::startpos();
        let mut h = p.initial_rule_history();
        let mut hits = 0;
        for ply in 0..4 {
            let result = compare(
                &p,
                &h,
                AzSearchLimits {
                    simulations: 400,
                    policy_softmax_temp: 1.45,
                    root_dirichlet_alpha: 0.12,
                    root_exploration_fraction: 0.1,
                    minimum_kldgain_per_node: 5e-5,
                    seed: 17 + ply,
                    ..Default::default()
                },
                &mut workspace,
            );
            hits += workspace.last_cache_hits;
            let mv = result.best_move.unwrap();
            h.push(p.rule_history_entry_after_move(mv));
            p.make_move(mv);
        }
        assert!(hits > 0, "必须实际命中网络缓存");
        assert_eq!(std::mem::size_of::<AzChild>(), 32);
    }

    #[test]
    fn eval_cache_capture_king_and_repeated_paths_match_cold() {
        let model = AzNnue::random(8, 31);
        let limits = AzSearchLimits {
            simulations: 128,
            ..Default::default()
        };
        for sequence in [
            vec!["b2b9"],
            vec!["e0e1"],
            vec!["b0c2", "b9c7", "c2b0", "c7b9"],
        ] {
            let mut workspace = AzSearchWorkspace::new(&model);
            let mut p = Position::startpos();
            let mut h = p.initial_rule_history();
            compare(&p, &h, limits, &mut workspace);
            for text in sequence {
                let mv = p.parse_uci_move(text).unwrap();
                assert!(p.legal_moves_with_rules(&h).contains(&mv));
                h.push(p.rule_history_entry_after_move(mv));
                p.make_move(mv);
                compare(&p, &h, limits, &mut workspace);
            }
        }
    }

    #[test]
    fn eval_cache_rejects_history_and_limits_mismatch() {
        let model = AzNnue::random(8, 31);
        let mut workspace = AzSearchWorkspace::new(&model);
        let mut p = Position::startpos();
        let mut h = p.initial_rule_history();
        let limits = AzSearchLimits {
            simulations: 128,
            ..Default::default()
        };
        let result = compare(&p, &h, limits, &mut workspace);
        let mv = result.best_move.unwrap();
        h.push(p.rule_history_entry_after_move(mv));
        p.make_move(mv);
        let mut wrong = h.clone();
        wrong[0].rule60_clock += 1;
        assert!(take_cached(&p, &wrong, limits, &mut workspace).is_none());
        assert!(!workspace.nodes.is_empty());
        assert!(
            take_cached(
                &p,
                &h,
                AzSearchLimits {
                    value_scale: 0.5,
                    ..limits
                },
                &mut workspace
            )
            .is_none()
        );
        assert!(!workspace.nodes.is_empty());
    }
}
