use super::*;

#[derive(Clone)]
pub(crate) struct GoalPlan {
    goal: AzMatingGoal,
    history: Vec<RuleHistoryEntry>,
    root: Arc<GoalPlanNode>,
}

struct GoalPlanNode {
    position: Position,
    plies: u32,
    children: Vec<(Move, RuleHistoryEntry, Arc<GoalPlanNode>)>,
}

impl GoalPlan {
    fn at(
        &self,
        position: &Position,
        history: &[RuleHistoryEntry],
        goal: AzMatingGoal,
    ) -> Option<&GoalPlanNode> {
        if !goal.effective(position).enabled()
            || goal.target != self.goal.target
            || goal.side != self.goal.side
            || position.side_to_move() != goal.side
            || !history.starts_with(&self.history)
        {
            return None;
        }
        let mut node = self.root.as_ref();
        for entry in &history[self.history.len()..] {
            node = node
                .children
                .iter()
                .find(|(_, expected, _)| expected == entry)?
                .2
                .as_ref();
        }
        (node.position == *position && !node.children.is_empty()).then_some(node)
    }

    pub(crate) fn apply(
        &self,
        position: &Position,
        history: &[RuleHistoryEntry],
        goal: AzMatingGoal,
        result: &mut AzSearchResult,
    ) -> Option<Vec<Move>> {
        let node = self.at(position, history, goal)?;
        let mv = node.children.first()?.0;
        if !result.candidates.iter().any(|candidate| candidate.mv == mv) {
            return None;
        }
        result.best_move = Some(mv);
        result.value_q = 1.0;
        result.value_wdl = [1.0, 0.0, 0.0];
        result.best_value_wdl = result.value_wdl;
        result.value_cp = cp_from_q(1.0);
        result.goal_value = 1.0;
        result.proven_moves_left = Some(node.plies as usize);
        for candidate in &mut result.candidates {
            candidate.policy = f32::from(u8::from(candidate.mv == mv));
            if candidate.mv == mv {
                candidate.q = 1.0;
                candidate.solved = Some(1);
                candidate.goal_value = 1.0;
                candidate.moves_left = Some(node.plies as f32);
            }
        }
        let mut pv = Vec::new();
        let mut current = node;
        while let Some((mv, _, next)) = current
            .children
            .iter()
            .max_by_key(|(_, _, next)| next.plies)
        {
            pv.push(*mv);
            current = next;
        }
        Some(pv)
    }

    pub(crate) fn retain_shorter(
        cache: &mut Option<Self>,
        fresh: Option<Self>,
        position: &Position,
        history: &[RuleHistoryEntry],
        goal: AzMatingGoal,
    ) {
        let previous = cache
            .as_ref()
            .and_then(|plan| plan.at(position, history, goal))
            .map(|node| node.plies);
        if let Some(plan) = fresh {
            if previous.is_none_or(|bound| plan.root.plies < bound) {
                *cache = Some(plan);
            }
        }
    }
}

impl AzTree<'_> {
    pub(super) fn goal_plan(&self) -> Option<GoalPlan> {
        if !self.goal_turn(self.root) {
            return None;
        }
        fn collect(tree: &AzTree<'_>, index: usize) -> Option<Arc<GoalPlanNode>> {
            let node = &tree.nodes[index];
            if node.goal_outcome != Some(true) {
                return None;
            }
            let mut children = Vec::new();
            if node.goal_plies > 0 {
                let edges: Vec<_> = if node.position.side_to_move() == tree.mating_goal.side {
                    vec![&tree.node_children(index)[tree.best_goal_root_child(index)?]]
                } else {
                    tree.node_children(index).iter().collect()
                };
                if edges.is_empty() {
                    return None;
                }
                for edge in edges {
                    let child = edge.child_node()?;
                    let next = collect(tree, child)?;
                    if next.plies + 1 > node.goal_plies {
                        return None;
                    }
                    children.push((edge.mv, tree.nodes[child].rule_entry?, next));
                }
            } else if !tree.terminal_goal_success(&node.position) {
                return None;
            }
            Some(Arc::new(GoalPlanNode {
                position: node.position.clone(),
                plies: node.goal_plies,
                children,
            }))
        }
        Some(GoalPlan {
            goal: self.mating_goal,
            history: self.rule_history_scratch.clone(),
            root: collect(self, self.root)?,
        })
    }
    /// 普通必胜证明不等于指定杀法已经完成；保留该条件策略的耗时估计。
    pub(super) fn goal_pending(&self, node: usize) -> bool {
        self.mating_goal.enabled() && self.nodes[node].goal_outcome.is_none()
    }

    pub(super) fn evaluation_moves_left(&self, node: usize, estimate: f32) -> f32 {
        if self.nodes[node].goal_outcome == Some(true) {
            self.nodes[node].goal_plies as f32
        } else if self.goal_pending(node) {
            estimate
        } else {
            self.nodes[node]
                .proven_moves_left
                .map_or(estimate, |distance| distance as f32)
        }
    }

    pub(super) fn child_goal_proven_distance(&self, child: &AzChild) -> Option<usize> {
        let node = &self.nodes[child.child_node()?];
        (node.goal_outcome == Some(true)).then_some(node.goal_plies as usize + 1)
    }

    pub(super) fn update_goal_proof(&mut self, node: usize) {
        if !self.mating_goal.enabled() || self.node_children(node).is_empty() {
            return;
        }
        if self.nodes[node].goal_outcome == Some(false) {
            self.nodes[node].goal_min_plies = u32::MAX;
            return;
        }
        let owner = self.nodes[node].position.side_to_move() == self.mating_goal.side;
        let lower = self
            .node_children(node)
            .iter()
            .map(|child| self.goal_child_lower_bound(node, child).saturating_add(1));
        let lower = if owner { lower.min() } else { lower.max() }.unwrap_or(0);
        self.nodes[node].goal_min_plies = self.nodes[node].goal_min_plies.max(lower);
        let mut distances = self
            .node_children(node)
            .iter()
            .map(|child| self.child_goal_proven_distance(child));
        let distance = if owner {
            distances.flatten().min()
        } else {
            distances.try_fold(0usize, |bound, distance| Some(bound.max(distance?)))
        };
        if let Some(distance) = distance {
            let entry = &mut self.nodes[node];
            entry.goal_outcome = Some(true);
            entry.goal_plies = distance as u32;
            entry.goal_value = 1.0;
            entry.moves_left = distance as f32;
        }
    }
    pub(super) fn goal_at(&self, node: usize) -> AzMatingGoal {
        if self.nodes[node].goal_outcome == Some(false) {
            return AzMatingGoal::NONE;
        }
        self.mating_goal.effective(&self.nodes[node].position)
    }

    pub(super) fn goal_child_lower_bound(&self, parent: usize, child: &AzChild) -> u32 {
        child.child_node().map_or_else(
            || {
                if self.nodes[parent].position.side_to_move() == self.mating_goal.side {
                    2
                } else {
                    1
                }
            },
            |index| self.nodes[index].goal_min_plies,
        )
    }

    pub(super) fn goal_turn(&self, node: usize) -> bool {
        let goal = self.goal_at(node);
        goal.enabled() && self.nodes[node].position.side_to_move() == goal.side
    }

    pub(super) fn search_finished(&self, node: usize) -> bool {
        if self.nodes[node].solved.is_none() {
            return false;
        }
        if !self.goal_turn(node) || self.nodes[node].goal_outcome == Some(false) {
            return true;
        }
        if self.nodes[node].goal_outcome == Some(true)
            && self.nodes[node].goal_min_plies >= self.nodes[node].goal_plies
        {
            return true;
        }
        if self.node_children(node).iter().any(|child| {
            child.child_node().is_some_and(|index| {
                self.nodes[index].goal_outcome == Some(true)
                    && self.nodes[index].goal_plies == 0
                    && self.child_solved(child) == Some(1)
            })
        }) {
            return true;
        }
        // 普通必胜不终止指定杀法探索；仍服从原 simulations、KLD 和时间预算。
        false
    }

    pub(super) fn node_goal_value(&self, node: usize) -> f32 {
        let node = &self.nodes[node];
        node.goal_outcome
            .map_or_else(
                || {
                    if node.visits > 0 {
                        node.goal_value_sum / node.visits as f32
                    } else {
                        node.goal_value
                    }
                },
                |success| f32::from(u8::from(success)),
            )
            .clamp(0.0, 1.0)
    }

    /// 普通胜负证明继续有效；只允许目标概率在后续探索中更新。
    pub(super) fn with_proven_wdl(&self, node: usize, mut eval: AzEvalOutput) -> AzEvalOutput {
        if let Some(value) = self.nodes[node].solved {
            eval.value = value as f32;
            eval.value_wdl = scalar_terminal_wdl(value as f32);
            eval.moves_left = self.evaluation_moves_left(node, eval.moves_left);
        }
        if let Some(success) = self.nodes[node].goal_outcome {
            eval.goal_value = f32::from(u8::from(success));
        }
        eval
    }

    pub(super) fn child_goal_value(&self, child: &AzChild) -> f32 {
        if let Some(index) = child.child_node() {
            if let Some(success) = self.nodes[index].goal_outcome {
                return f32::from(u8::from(success));
            }
            if child.visits == 0 {
                return self.nodes[index].goal_value;
            }
        }
        if child.visits == 0 {
            0.0
        } else {
            (child.goal_value_sum / child.visits as f32).clamp(0.0, 1.0)
        }
    }

    /// 目标成功为 +1，非目标终局为 -1；WDL 作为强度之外的争胜分量。
    pub(super) fn goal_utility(&self, q: f32, probability: f32) -> f32 {
        let strength = self.mating_goal.strength;
        (1.0 - strength) * q + strength * (2.0 * probability - 1.0)
    }

    /// 胜势明确后，比较单位预计耗时的目标收益；前期维持原目标效用。
    /// MLH 是条件策略的对局长度估计，不是指定杀法的严格距离证明。
    pub(super) fn goal_child_utility(&self, child: &AzChild, q: f32) -> f32 {
        let probability = self.child_goal_value(child);
        let plies = self.child_moves_left(child);
        let params = self.model.moves_left_params;
        let confidence = if params.threshold < 1.0 {
            ((q - params.threshold) / (1.0 - params.threshold)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let timely = if self.model.moves_left_active
            && params.enabled
            && params.max_effect > 0.0
            && plies.is_finite()
        {
            probability / (1.0 + confidence * params.slope * plies.max(0.0))
        } else {
            probability
        };
        self.goal_utility(q, timely)
    }

    pub(super) fn goal_child_priority(&self, child: &AzChild, solved: Option<i8>) -> u8 {
        if solved == Some(1)
            && child
                .child_node()
                .is_some_and(|index| self.nodes[index].goal_outcome == Some(true))
        {
            2 // 真实完成目标；不能由网络预测冒充。
        } else {
            u8::from(solved != Some(-1))
        }
    }

    pub(super) fn best_goal_root_child(&self, node: usize) -> Option<usize> {
        let draw_score = self.node_draw_score(node);
        self.node_children(node)
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                self.goal_child_priority(left, self.child_solved(left))
                    .cmp(&self.goal_child_priority(right, self.child_solved(right)))
                    .then_with(|| {
                        self.child_goal_proven_distance(right)
                            .unwrap_or(usize::MAX)
                            .cmp(&self.child_goal_proven_distance(left).unwrap_or(usize::MAX))
                    })
                    .then_with(|| left.visits.cmp(&right.visits))
                    .then_with(|| {
                        self.goal_child_utility(left, self.child_q(left, draw_score))
                            .total_cmp(
                                &self.goal_child_utility(right, self.child_q(right, draw_score)),
                            )
                    })
                    .then_with(|| {
                        self.child_q(left, draw_score)
                            .total_cmp(&self.child_q(right, draw_score))
                    })
                    .then_with(|| left.prior.total_cmp(&right.prior))
            })
            .map(|(index, _)| index)
    }

    pub(super) fn terminal_goal_success(&self, position: &Position) -> bool {
        if !self.mating_goal.enabled() {
            return false;
        }
        position
            .classify_mate()
            .is_some_and(|mate| self.mating_goal.matches_mate(&mate))
    }

    pub(super) fn set_terminal_goal(&mut self, node: usize) {
        let owner_won = if self.nodes[node].position.side_to_move() == self.mating_goal.side {
            self.nodes[node].value > 0.0
        } else {
            self.nodes[node].value < 0.0
        };
        let success = owner_won && self.terminal_goal_success(&self.nodes[node].position);
        self.nodes[node].goal_value = f32::from(u8::from(success));
        self.nodes[node].goal_outcome = self.mating_goal.enabled().then_some(success);
        self.nodes[node].goal_min_plies = if success { 0 } else { u32::MAX };
    }

    pub(super) fn prepare_goal(&mut self, node: usize) {
        let effective = self.goal_at(node);
        self.eval_scratch.mating_goal = effective;
        if effective.enabled() {
            let minimum = effective.minimum_finish_plies(&self.nodes[node].position);
            self.nodes[node].goal_min_plies = self.nodes[node].goal_min_plies.max(minimum);
        }
        if self.mating_goal.enabled() && !effective.enabled() {
            self.nodes[node].goal_value = 0.0;
            self.nodes[node].goal_outcome = Some(false);
            self.nodes[node].goal_min_plies = u32::MAX;
        }
    }

    pub(super) fn reject_unwinnable_goal(&mut self, node: usize, outcome: i8) {
        if !self.mating_goal.enabled() {
            return;
        }
        let owner_outcome = if self.nodes[node].position.side_to_move() == self.mating_goal.side {
            outcome
        } else {
            -outcome
        };
        if owner_outcome <= 0 {
            self.nodes[node].goal_outcome = Some(false);
            self.nodes[node].goal_value = 0.0;
            self.nodes[node].goal_value_sum = 0.0;
            self.nodes[node].goal_min_plies = u32::MAX;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::az::MatingTarget;
    use crate::xiangqi::MateKind;

    #[test]
    #[ignore = "手动性能基准：固定局面、模型和预算，不断言耗时"]
    fn goal_hot_path_benchmark() {
        use std::hint::black_box;
        let mut model = AzNnue::random(64, 1902);
        for (i, weight) in model.goal_hidden.iter_mut().enumerate() {
            *weight = (i % 17) as f32 * 0.01 - 0.08;
        }
        let position = Position::startpos();
        let history = position.initial_rule_history();
        let moves = position.legal_moves();
        let goals = MatingTarget::ALL.map(|target| goal(target, Color::Red, 0.8));
        for &instruction in &goals {
            black_box(model.evaluate_with_goal(&position, &history, &moves, instruction));
        }
        let mut checksum = 0.0f64;
        for (label, instructions) in [
            ("ordinary_eval", vec![AzMatingGoal::NONE; goals.len()]),
            ("goal_eval", goals.to_vec()),
        ] {
            let started = Instant::now();
            for _ in 0..100 {
                for &instruction in &instructions {
                    let (probability, wdl, logits) =
                        model.evaluate_with_goal(&position, &history, &moves, instruction);
                    checksum += f64::from(probability + wdl[0] + logits.iter().sum::<f32>());
                    black_box(logits);
                }
            }
            eprintln!(
                "{label}: evaluations={} ms={:.3}",
                100 * instructions.len(),
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
        let started = Instant::now();
        let mut simulations = 0;
        for _ in 0..4 {
            for &instruction in &goals {
                let result = alphazero_search_with_rules(
                    &position,
                    Some(history.clone()),
                    None,
                    &model,
                    AzSearchLimits {
                        simulations: 128,
                        max_depth: 4,
                        mating_goal: instruction,
                        ..Default::default()
                    },
                );
                simulations += result.simulations;
                let mv = result.best_move.unwrap();
                checksum += f64::from(mv.from) + f64::from(mv.to);
                black_box(result);
            }
        }
        eprintln!(
            "goal_search: searches={} simulations={simulations} ms={:.3} checksum={checksum:.9}",
            4 * goals.len(),
            started.elapsed().as_secs_f64() * 1000.0
        );
    }

    fn goal(target: MatingTarget, side: Color, strength: f32) -> AzMatingGoal {
        AzMatingGoal {
            target,
            side,
            strength,
        }
    }

    fn limits(mating_goal: AzMatingGoal) -> AzSearchLimits {
        AzSearchLimits {
            simulations: 32,
            max_depth: 4,
            mating_goal,
            ..Default::default()
        }
    }

    fn assert_identical(left: &AzSearchResult, right: &AzSearchResult) {
        assert_eq!(left.best_move, right.best_move);
        assert_eq!(left.value_wdl, right.value_wdl);
        assert_eq!(left.network_value_wdl, right.network_value_wdl);
        assert_eq!(left.value_q, right.value_q);
        assert_eq!(left.goal_value, right.goal_value);
        assert_eq!(left.proven_moves_left, right.proven_moves_left);
        assert_eq!(left.simulations, right.simulations);
        assert_eq!(left.search_depth_avg, right.search_depth_avg);
        assert_eq!(left.search_depth_max, right.search_depth_max);
        assert_eq!(left.search_depth_cutoffs, right.search_depth_cutoffs);
        assert_eq!(left.candidates.len(), right.candidates.len());
        for (a, b) in left.candidates.iter().zip(&right.candidates) {
            assert_eq!(
                (
                    a.mv,
                    a.visits,
                    a.q,
                    a.prior,
                    a.raw_prior,
                    a.policy,
                    a.solved,
                    a.moves_left,
                    a.goal_value
                ),
                (
                    b.mv,
                    b.visits,
                    b.q,
                    b.prior,
                    b.raw_prior,
                    b.policy,
                    b.solved,
                    b.moves_left,
                    b.goal_value
                )
            );
        }
    }

    #[test]
    fn terminal_goal_requires_the_rules_to_award_the_owner_a_win() {
        let position = Position::from_fen("3aka3/9/9/4C4/9/4C4/9/9/9/4K4 b").unwrap();
        let model = AzNnue::random(8, 103);
        let mut tree = AzTree::new(
            position.clone(),
            position.initial_rule_history(),
            None,
            &model,
            limits(goal(MatingTarget::DoubleCannon, Color::Red, 0.8)),
        );
        for (value, expected) in [(1.0, false), (0.0, false), (-1.0, true)] {
            tree.nodes[0].value = value;
            tree.set_terminal_goal(0);
            assert_eq!(tree.nodes[0].goal_outcome, Some(expected));
            assert_eq!(tree.nodes[0].goal_value, f32::from(u8::from(expected)));
        }
    }

    #[test]
    fn target_selects_its_actual_mate_over_another_immediate_win() {
        for (fen, ordinary, target_move, target) in [
            (
                "3aka3/R8/6N2/4C4/9/7C1/9/9/9/4K4 w",
                "a8e8",
                "h4e4",
                MatingTarget::DoubleCannon,
            ),
            (
                "2bak4/4a4/4b4/1N2C4/9/9/5R3/9/9/5K3 w",
                "b6c8",
                "f3f9",
                MatingTarget::IronBolt,
            ),
        ] {
            let position = Position::from_fen(fen).unwrap();
            let moves = [ordinary, target_move].map(|text| position.parse_uci_move(text).unwrap());
            for mv in moves {
                let mut terminal = position.clone();
                terminal.make_move(mv);
                assert_eq!(terminal.classify_mate().unwrap().kind, MateKind::Checkmate);
            }
            let model = AzNnue::random(8, 91);
            let baseline = alphazero_search_with_rules(
                &position,
                None,
                Some(moves.to_vec()),
                &model,
                limits(AzMatingGoal::NONE),
            );
            assert_eq!(baseline.best_move, Some(moves[0]));
            let controlled = alphazero_search_with_rules(
                &position,
                None,
                Some(moves.to_vec()),
                &model,
                limits(goal(target, Color::Red, 1.0)),
            );
            assert_eq!(controlled.best_move, Some(moves[1]));
            assert_eq!(controlled.value_wdl, [1.0, 0.0, 0.0]);
            let selected = controlled
                .candidates
                .iter()
                .find(|c| c.mv == moves[1])
                .unwrap();
            assert_eq!(selected.goal_value, 1.0);
            assert!(controlled.goal_value > 0.0 && controlled.goal_value <= 1.0);
            assert_eq!(controlled.proven_moves_left, Some(1));
        }
    }

    #[test]
    fn opening_instruction_changes_policy_and_search_before_any_mate_exists() {
        let position = Position::startpos();
        let central = position.parse_uci_move("b2e2").unwrap();
        let flank = position.parse_uci_move("b2a2").unwrap();
        let mut model = AzNnue::random(8, 109);
        model.value_head_output.fill(0.0);
        model.value_head_bias.fill(0.0);
        model.goal_output.fill(0.0);
        let run = |mating_goal| {
            alphazero_search_with_rules(
                &position,
                None,
                Some(vec![central, flank]),
                &model,
                AzSearchLimits {
                    simulations: 128,
                    max_depth: 1,
                    mating_goal,
                    ..Default::default()
                },
            )
        };
        let ordinary = run(AzMatingGoal::NONE);
        let base = ordinary
            .candidates
            .iter()
            .find(|c| c.mv == central)
            .unwrap();
        for target in [MatingTarget::DoubleCannon, MatingTarget::IronBolt] {
            let controlled = run(goal(target, Color::Red, 1.0));
            let directed = controlled
                .candidates
                .iter()
                .find(|c| c.mv == central)
                .unwrap();
            assert!(controlled.candidates.iter().all(|c| c.solved.is_none()));
            assert!(
                directed.raw_prior > base.raw_prior,
                "目标从第一步改变布局先验"
            );
            assert!(directed.visits > base.visits, "MCTS 给目标布局路线更多探索");
            assert!(directed.policy > base.policy, "目标布局方向写入训练策略");
            assert_eq!(controlled.network_value_wdl, ordinary.network_value_wdl);
        }
    }

    #[test]
    fn zero_strength_and_unreachable_material_are_exact_baselines() {
        let model = AzNnue::random(8, 93);
        for (fen, requested) in [
            (
                crate::xiangqi::STARTPOS_FEN,
                goal(MatingTarget::DoubleCannon, Color::Red, 0.0),
            ),
            (
                "3aka3/R8/6N2/4C4/9/9/9/9/9/4K4 w",
                goal(MatingTarget::DoubleCannon, Color::Red, 1.0),
            ),
        ] {
            let position = Position::from_fen(fen).unwrap();
            let ordinary = alphazero_search(&position, &model, limits(AzMatingGoal::NONE));
            let fallback = alphazero_search(&position, &model, limits(requested));
            assert_identical(&ordinary, &fallback);
        }
    }

    fn two_move_tree<'a>(position: &Position, model: &'a AzNnue, side: Color) -> AzTree<'a> {
        let moves = position.legal_moves().into_iter().take(2).collect();
        let mut tree = AzTree::new(
            position.clone(),
            position.initial_rule_history(),
            Some(moves),
            model,
            limits(goal(MatingTarget::DoubleCannon, side, 1.0)),
        );
        tree.expand(0);
        tree
    }

    #[test]
    fn fixed_owner_probability_is_backed_up_without_negation_or_proof() {
        let mut model = AzNnue::random(8, 94);
        model.goal_output.fill(0.0);
        model.goal_bias[0] = (7.0f32 / 3.0).ln();
        let mut tree = two_move_tree(&Position::startpos(), &model, Color::Red);
        tree.max_depth = 1;
        tree.simulate_child(0, 0, 1);
        let child = tree.node_children(0)[0].child_node().unwrap();
        tree.nodes[child].solved = Some(-1);
        tree.nodes[child].goal_outcome = None;
        let backed = tree.simulate_child(0, 0, 1);
        assert_eq!(backed.value, 1.0);
        assert!((backed.goal_value - 0.7).abs() < 1e-6);
        assert_eq!(
            tree.nodes[0].goal_outcome, None,
            "普通胜负证明不能变成目标证明"
        );
    }

    #[test]
    fn search_raw_priors_use_the_same_single_bootstrap_bias_as_cpu_policy() {
        let position = Position::startpos();
        let model = AzNnue::random(8, 98);
        let goal = goal(MatingTarget::DoubleCannon, Color::Red, 1.0);
        let moves = position.legal_moves();
        let mut scratch = AzEvalScratch::new(model.arch);
        scratch.mating_goal = goal;
        model.evaluate_with_scratch_output_with_repetition_and_history(
            &position,
            &moves,
            &[],
            &rule_context_features(&position, &position.initial_rule_history()),
            &position.initial_rule_history(),
            &mut scratch,
        );
        let mut expected = Vec::new();
        softmax_into(&scratch.logits, 1.0, &mut expected);
        let mut tree = AzTree::new(
            position.clone(),
            position.initial_rule_history(),
            Some(moves),
            &model,
            limits(goal),
        );
        tree.expand(0);
        assert_eq!(tree.root_raw_priors.len(), expected.len());
        for (actual, expected) in tree.root_raw_priors.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn first_ordinary_proof_does_not_stop_other_goal_routes() {
        let model = AzNnue::random(8, 95);
        let mut tree = two_move_tree(&Position::startpos(), &model, Color::Red);
        tree.simulate_child(0, 0, 1);
        let child = tree.node_children(0)[0].child_node().unwrap();
        tree.set_proven(child, -1);
        tree.update_solved(0);
        assert_eq!(tree.nodes[0].solved, Some(1));
        assert!(!tree.search_finished(0));
        for _ in 0..32 {
            tree.simulate(0, 0);
        }
        assert!(
            tree.node_children(0)[1].visits > 0,
            "已证明普通胜法不能垄断所有搜索预算"
        );
    }

    #[test]
    fn opponent_uses_wdl_and_does_not_cooperate_with_the_goal() {
        let model = AzNnue::random(8, 96);
        let position =
            Position::from_fen(&crate::xiangqi::STARTPOS_FEN.replace(" w", " b")).unwrap();
        let mut tree = two_move_tree(&position, &model, Color::Red);
        for (index, child) in tree.node_children_mut(0).iter_mut().enumerate() {
            child.visits = 10;
            child.prior = 0.5;
            child.goal_value_sum = if index == 0 { 10.0 } else { 0.0 };
            child.value_wdl_sum = if index == 0 {
                [0.0, 10.0, 0.0]
            } else {
                [8.0, 2.0, 0.0]
            };
        }
        assert!(!tree.goal_turn(0));
        assert_eq!(tree.select_child(0), 1);
        assert_eq!(tree.best_root_child(0), Some(1));
    }

    #[test]
    fn target_confidence_cannot_override_a_proven_loss() {
        let model = AzNnue::random(8, 97);
        let mut tree = two_move_tree(&Position::startpos(), &model, Color::Red);
        tree.simulate_child(0, 0, 1);
        let child = tree.node_children(0)[0].child_node().unwrap();
        tree.nodes[child].solved = Some(1);
        tree.nodes[child].goal_value = 1.0;
        tree.node_children_mut(0)[0].goal_value_sum = 1000.0;
        tree.node_children_mut(0)[0].prior = 1.0;
        tree.node_children_mut(0)[1].prior = 0.0;
        assert_eq!(tree.select_child(0), 1);
        assert_eq!(tree.best_root_child(0), Some(1));
    }

    #[test]
    fn proven_owner_loss_or_draw_zeroes_goal_in_both_node_perspectives() {
        let model = AzNnue::random(8, 99);
        for node_side in [Color::Red, Color::Black] {
            let fen = if node_side == Color::Red {
                crate::xiangqi::STARTPOS_FEN.to_owned()
            } else {
                crate::xiangqi::STARTPOS_FEN.replace(" w", " b")
            };
            let position = Position::from_fen(&fen).unwrap();
            for owner in [Color::Red, Color::Black] {
                for owner_outcome in [-1, 0] {
                    let mut tree = AzTree::new(
                        position.clone(),
                        position.initial_rule_history(),
                        None,
                        &model,
                        limits(goal(MatingTarget::DoubleCannon, owner, 1.0)),
                    );
                    tree.nodes[0].goal_value = 0.9;
                    tree.nodes[0].goal_value_sum = 9.0;
                    tree.nodes[0].visits = 10;
                    let outcome = if node_side == owner {
                        owner_outcome
                    } else {
                        -owner_outcome
                    };
                    tree.set_proven(0, outcome);
                    assert_eq!(tree.nodes[0].goal_outcome, Some(false));
                    assert_eq!(tree.nodes[0].goal_value, 0.0);
                    assert_eq!(tree.node_eval(0).goal_value, 0.0);
                    assert!(!tree.goal_at(0).enabled(), "目标已证不可达后应恢复普通争胜");
                    assert!(tree.search_finished(0));
                }
            }
        }
    }

    #[test]
    fn ordinary_win_respects_original_search_budget_and_kld_still_works() {
        let model = AzNnue::random(8, 100);
        let mut tree = two_move_tree(&Position::startpos(), &model, Color::Red);
        tree.set_proven(0, 1);
        for visits in [0, 63, 64, 1000] {
            tree.nodes[0].visits = visits;
            assert!(!tree.search_finished(0), "普通必胜不能截断目标探索");
        }
        tree.set_proven(0, 1);
        assert!(!tree.search_finished(0));

        let position = Position::startpos();
        let history = position.initial_rule_history();
        let moves = position.legal_moves()[..2].to_vec();
        let limits = AzSearchLimits {
            simulations: 4096,
            max_depth: 1,
            minimum_kldgain_per_node: 5e-5,
            mating_goal: goal(MatingTarget::DoubleCannon, Color::Red, 1.0),
            ..Default::default()
        };
        let plain = alphazero_search_with_rules(
            &position,
            Some(history.clone()),
            Some(moves.clone()),
            &model,
            limits,
        );
        let mut workspace = AzSearchWorkspace::new(&model);
        let reused = alphazero_search_with_rules_reusing(
            &position,
            &history,
            moves.clone(),
            limits,
            &mut workspace,
        );
        let (traced, _) = alphazero_search_trace_with_rules(
            &position,
            Some(history.clone()),
            Some(moves.clone()),
            &model,
            limits,
            moves[0],
        );
        let control = AzSearchControl::new(Arc::new(AtomicBool::new(false)), None);
        let controlled = alphazero_search_with_rules_controlled(
            &position,
            Some(history),
            Some(moves),
            &model,
            limits,
            Some(&control),
        );
        for result in [plain, reused, traced, controlled] {
            assert_eq!(result.simulations, 400, "带目标也应在策略收敛后停止");
        }
    }

    #[test]
    fn ordinary_mate_in_one_does_not_hide_target_mate_in_three_with_two_defenses() {
        let position = Position::from_fen("3aka3/R8/6N2/p3C3p/9/6PC1/9/9/9/4K4 w").unwrap();
        let ordinary = position.parse_uci_move("a8e8").unwrap();
        let setup = position.parse_uci_move("h4h3").unwrap();
        let mut after_setup = position.clone();
        after_setup.make_move(setup);
        let defenses = after_setup.legal_moves();
        assert_eq!(defenses.len(), 2, "目标路线需要覆盖不唯一的合法防守");
        let requested = goal(MatingTarget::DoubleCannon, Color::Red, 1.0);
        for defense in defenses {
            let mut reply = after_setup.clone();
            reply.make_move(defense);
            let mate = reply.parse_uci_move("h3e3").unwrap();
            reply.make_move(mate);
            assert!(requested.matches_mate(&reply.classify_mate().unwrap()));
        }
        let model = AzNnue::random(8, 101);
        let result = alphazero_search_with_rules(
            &position,
            None,
            Some(vec![ordinary, setup]),
            &model,
            AzSearchLimits {
                simulations: 4096,
                max_depth: 5,
                mating_goal: requested,
                ..Default::default()
            },
        );
        assert_eq!(result.best_move, Some(setup));
        assert_eq!(result.value_wdl, [1.0, 0.0, 0.0]);
        assert_eq!(result.proven_moves_left, Some(3));
        assert!(result.simulations > 1 && result.simulations <= 4096);
        let target = result
            .candidates
            .iter()
            .find(|child| child.mv == setup)
            .unwrap();
        let generic = result
            .candidates
            .iter()
            .find(|child| child.mv == ordinary)
            .unwrap();
        assert_eq!(target.solved, Some(1));
        assert!(target.goal_value > generic.goal_value);
        assert!(target.policy > generic.policy);
    }
    #[test]
    fn strong_goal_can_choose_an_unproven_route_over_a_proven_ordinary_win() {
        let model = AzNnue::random(8, 1910);
        let mut tree = two_move_tree(&Position::startpos(), &model, Color::Red);
        tree.simulate_child(0, 0, 1);
        tree.simulate_child(0, 1, 1);
        let ordinary = tree.node_children(0)[0].child_node().unwrap();
        tree.nodes[ordinary].solved = Some(-1);
        tree.nodes[ordinary].goal_outcome = Some(false);
        tree.nodes[0].visits = 300;
        for (index, child) in tree.node_children_mut(0).iter_mut().enumerate() {
            child.visits = if index == 0 { 100 } else { 200 };
            child.prior = 0.5;
            child.value_wdl_sum = if index == 0 {
                [100.0, 0.0, 0.0]
            } else {
                [120.0, 0.0, 80.0]
            };
            child.goal_value_sum = if index == 0 { 0.0 } else { 120.0 };
        }
        tree.mating_goal.strength = 0.1;
        assert_eq!(tree.select_child(0), 0, "弱偏好仍偏向普通必胜");
        tree.mating_goal.strength = 0.8;
        assert_eq!(tree.select_child(0), 1, "强偏好必须改变搜索效用");
        assert_eq!(
            tree.best_root_child(0),
            Some(1),
            "普通胜负证明不能覆盖目标路线"
        );
        let policy = tree.root_policy(0);
        assert!((policy[1] - 2.0 / 3.0).abs() < 1e-6);
        tree.nodes[ordinary].proven_moves_left = Some(0);
        tree.set_proven(0, 1);
        let report = tree.search_result(300);
        assert_eq!(
            report.proven_moves_left, None,
            "不能把放弃的普通一手杀距离报告为目标路线的证明"
        );
        assert_eq!(report.best_move, Some(tree.node_children(0)[1].mv));
        assert!(
            policy[0] > 0.0,
            "预测不能伪装成确定的目标证明并压成单点标签"
        );
    }

    #[test]
    fn goal_keeps_bounded_finish_preference_without_overriding_goal_utility() {
        let mut model = AzNnue::random(8, 1915);
        model.moves_left_active = true;
        let mut tree = two_move_tree(&Position::startpos(), &model, Color::Red);
        tree.mating_goal.strength = 0.4;
        tree.nodes[0].visits = 200;
        tree.nodes[0].value_wdl_sum = [200.0, 0.0, 0.0];
        tree.nodes[0].moves_left_sum = 200.0 * 30.0;
        for (index, child) in tree.node_children_mut(0).iter_mut().enumerate() {
            child.visits = 100;
            child.prior = 0.5;
            child.value_wdl_sum = [100.0, 0.0, 0.0];
            child.goal_value_sum = 50.0;
            child.moves_left_sum = 100.0 * if index == 0 { 80.0 } else { 4.0 };
        }
        assert_eq!(tree.select_child(0), 1, "同目标效用下应保留收尾偏好");
        tree.node_children_mut(0)[0].goal_value_sum = 90.0;
        tree.node_children_mut(0)[1].goal_value_sum = 10.0;
        assert_eq!(
            tree.select_child(0),
            0,
            "有界距离加分不能压过明显更好的目标路线"
        );
        tree.nodes[0].value_wdl_sum = [100.0, 0.0, 100.0];
        assert!(
            tree.moves_left_context(0).is_none(),
            "胜负不明确时仍关闭距离加分"
        );
    }

    #[test]
    fn ordinary_win_proof_keeps_goal_rollout_time_until_the_goal_is_settled() {
        let model = AzNnue::random(8, 1920);
        let mut tree = two_move_tree(&Position::startpos(), &model, Color::Red);
        tree.nodes[0].moves_left = 80.0;
        tree.nodes[0].proven_moves_left = Some(3);
        tree.set_proven(0, 1);
        assert_eq!(tree.nodes[0].proven_moves_left, Some(3));
        assert_eq!(tree.nodes[0].moves_left, 80.0);
        let neural = AzEvalOutput {
            moves_left: 90.0,
            value_wdl: [0.7, 0.1, 0.2],
            value: 0.5,
            goal_value: 0.6,
        };
        let proven = tree.with_proven_wdl(0, neural);
        assert_eq!(proven.value, 1.0);
        assert_eq!(proven.moves_left, 90.0);
        tree.max_depth = 1;
        let backed = tree.simulate_child(0, 0, 1);
        assert_eq!(
            backed.moves_left, 101.0,
            "普通三步杀不能截短目标路线的距离回传"
        );
        tree.nodes[0].goal_outcome = Some(false);
        tree.set_proven(0, 1);
        assert_eq!(tree.nodes[0].moves_left, 3.0);
        assert_eq!(tree.with_proven_wdl(0, neural).moves_left, 3.0);
    }

    #[test]
    fn proven_ordinary_wins_still_prefer_a_faster_goal_route() {
        let mut model = AzNnue::random(8, 1921);
        model.moves_left_active = true;
        let mut tree = two_move_tree(&Position::startpos(), &model, Color::Red);
        tree.mating_goal.strength = 0.4;
        for i in 0..2 {
            tree.simulate_child(0, i, 1);
            let index = tree.node_children(0)[i].child_node().unwrap();
            tree.set_proven(index, -1);
            tree.nodes[index].proven_moves_left = Some(3);
        }
        tree.nodes[0].visits = 200;
        tree.nodes[0].value_wdl_sum = [200.0, 0.0, 0.0];
        tree.nodes[0].moves_left_sum = 200.0 * 80.0;
        for (i, child) in tree.node_children_mut(0).iter_mut().enumerate() {
            child.visits = 100;
            child.prior = 0.5;
            child.value_wdl_sum = [100.0, 0.0, 0.0];
            child.goal_value_sum = 80.0;
            child.moves_left_sum = 100.0 * if i == 0 { 150.0 } else { 10.0 };
        }
        assert_eq!(tree.child_moves_left(&tree.node_children(0)[0]), 150.0);
        assert_eq!(tree.select_child(0), 1);
        tree.node_children_mut(0)[0].goal_value_sum = 95.0;
        tree.node_children_mut(0)[1].goal_value_sum = 10.0;
        assert_eq!(tree.select_child(0), 0, "明显更可能完成目标的路线仍可优先");
    }

    #[test]
    fn cached_goal_plan_covers_all_defenses_and_rejects_different_instructions_or_history() {
        let position = Position::from_fen("3aka3/R8/6N2/p3C3p/9/6PC1/9/9/9/4K4 w").unwrap();
        let history = position.initial_rule_history();
        let model = AzNnue::random(8, 1923);
        let requested = goal(MatingTarget::DoubleCannon, Color::Red, 1.0);
        let roots = ["a8e8", "h4h3"]
            .map(|text| position.parse_uci_move(text).unwrap())
            .to_vec();
        let mut tree = AzTree::new(
            position.clone(),
            history.clone(),
            Some(roots),
            &model,
            AzSearchLimits {
                simulations: 4096,
                max_depth: 5,
                mating_goal: requested,
                ..Default::default()
            },
        );
        tree.expand(0);
        for _ in 0..4096 {
            tree.simulate(0, 0);
        }
        let plan = tree.goal_plan().expect("真实三步重炮必须形成完整应对证明");
        assert_eq!(plan.root.plies, 3);
        let mut result = tree.search_result(4096);
        assert!(
            plan.apply(&position, &history, requested, &mut result)
                .is_some()
        );
        assert_eq!(result.best_move.unwrap().to_uci(), "h4h3");
        assert_eq!(result.proven_moves_left, Some(3));
        assert!(
            plan.at(
                &position,
                &history,
                goal(MatingTarget::IronBolt, Color::Red, 1.0)
            )
            .is_none()
        );
        let mut after = position.clone();
        let mut moves = history.clone();
        let setup = result.best_move.unwrap();
        let mover = after.side_to_move();
        let captured = after.piece_at(setup.to as usize);
        after.make_move(setup);
        moves.push(after.rule_history_entry_after_moved(mover, setup, captured));
        assert!(
            plan.at(&after, &moves, requested).is_none(),
            "不能替对手选择配合目标的走法"
        );
        let replies = after.legal_moves_with_rules(&moves);
        assert_eq!(replies.len(), 2);
        for reply in replies {
            let mut next = after.clone();
            let mut next_history = moves.clone();
            let mover = next.side_to_move();
            let captured = next.piece_at(reply.to as usize);
            next.make_move(reply);
            next_history.push(next.rule_history_entry_after_moved(mover, reply, captured));
            assert_eq!(plan.at(&next, &next_history, requested).unwrap().plies, 1);
            assert!(
                plan.at(&next, &next.initial_rule_history(), requested)
                    .is_none()
            );
            let mut fresh = alphazero_search_with_rules(
                &next,
                Some(next_history.clone()),
                None,
                &model,
                AzSearchLimits {
                    simulations: 1,
                    mating_goal: requested,
                    ..Default::default()
                },
            );
            assert!(
                plan.apply(&next, &next_history, requested, &mut fresh)
                    .is_some()
            );
            let last = fresh.best_move.unwrap();
            next.make_move(last);
            assert!(requested.matches_mate(&next.classify_mate().unwrap()));
        }
        let ordinary = position.parse_uci_move("a8e8").unwrap();
        let mut constrained = alphazero_search_with_rules(
            &position,
            Some(history.clone()),
            Some(vec![ordinary]),
            &model,
            AzSearchLimits {
                simulations: 1,
                mating_goal: requested,
                ..Default::default()
            },
        );
        assert!(
            plan.apply(&position, &history, requested, &mut constrained)
                .is_none(),
            "尊重 GUI searchmoves 限制"
        );
        assert_eq!(constrained.best_move, Some(ordinary));
    }

    #[test]
    fn goal_urgency_is_off_before_a_clear_advantage_and_when_mlh_is_disabled() {
        let mut model = AzNnue::random(8, 1922);
        model.moves_left_active = true;
        let mut tree = two_move_tree(&Position::startpos(), &model, Color::Red);
        for (i, child) in tree.node_children_mut(0).iter_mut().enumerate() {
            child.visits = 100;
            child.goal_value_sum = 80.0;
            child.moves_left_sum = 100.0 * if i == 0 { 150.0 } else { 10.0 };
        }
        let children = tree.node_children(0);
        for q in [-1.0, 0.0, 0.6, model.moves_left_params.threshold] {
            for child in children {
                assert_eq!(tree.goal_child_utility(child, q), tree.goal_utility(q, 0.8));
            }
        }
        assert!(
            tree.goal_child_utility(&children[1], 1.0) > tree.goal_child_utility(&children[0], 1.0)
        );
        drop(tree);
        for disabled in [0, 1, 2] {
            let mut no_mlh = model.clone();
            match disabled {
                0 => no_mlh.moves_left_active = false,
                1 => no_mlh.moves_left_params.enabled = false,
                _ => no_mlh.moves_left_params.max_effect = 0.0,
            }
            let mut tree = two_move_tree(&Position::startpos(), &no_mlh, Color::Red);
            let child = &mut tree.node_children_mut(0)[0];
            child.visits = 100;
            child.goal_value_sum = 80.0;
            child.moves_left_sum = 15000.0;
            assert_eq!(
                tree.goal_child_utility(&tree.node_children(0)[0], 1.0),
                tree.goal_utility(1.0, 0.8)
            );
        }
    }

    #[test]
    fn tiny_goal_prediction_differences_do_not_erase_root_visit_evidence() {
        let model = AzNnue::random(8, 1911);
        let mut tree = two_move_tree(&Position::startpos(), &model, Color::Red);
        for index in 0..2 {
            tree.simulate_child(0, index, 1);
            let node = tree.node_children(0)[index].child_node().unwrap();
            tree.nodes[node].solved = Some(-1);
        }
        for (index, child) in tree.node_children_mut(0).iter_mut().enumerate() {
            child.visits = if index == 0 { 10 } else { 100 };
            child.goal_value_sum = child.visits as f32 * if index == 0 { 0.501 } else { 0.5 };
        }
        assert_eq!(tree.best_root_child(0), Some(1));
        let policy = tree.root_policy(0);
        assert!(policy[0] > 0.0 && policy[1] > policy[0]);
    }
    #[test]
    fn all_confirmed_failed_goal_routes_restore_ordinary_winning_selection() {
        let model = AzNnue::random(8, 1914);
        let mut tree = two_move_tree(&Position::startpos(), &model, Color::Red);
        for index in 0..2 {
            tree.simulate_child(0, index, 1);
            let node = tree.node_children(0)[index].child_node().unwrap();
            tree.nodes[node].goal_outcome = Some(false);
            tree.nodes[node].solved = Some(if index == 0 { -1 } else { 0 });
            tree.node_children_mut(0)[index].visits = if index == 0 { 10 } else { 100 };
        }
        tree.update_solved(0);
        assert_eq!(tree.nodes[0].goal_outcome, Some(false));
        assert!(!tree.goal_turn(0));
        assert_eq!(tree.best_root_child(0), Some(0));
        assert_eq!(tree.root_policy(0), vec![1.0, 0.0]);
        assert!(tree.search_finished(0));
    }
}
