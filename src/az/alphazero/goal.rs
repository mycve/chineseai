use super::*;

impl AzTree<'_> {
    pub(super) fn goal_at(&self, node: usize) -> AzMatingGoal {
        if self.nodes[node].goal_outcome == Some(false) {
            return AzMatingGoal::NONE;
        }
        self.mating_goal.effective(&self.nodes[node].position)
    }

    pub(super) fn goal_turn(&self, node: usize) -> bool {
        let goal = self.goal_at(node);
        goal.enabled() && self.nodes[node].position.side_to_move() == goal.side
    }

    pub(super) fn search_finished(&self, node: usize) -> bool {
        if self.nodes[node].solved.is_none() {
            return false;
        }
        if !self.goal_turn(node) || self.nodes[node].goal_outcome.is_some() {
            return true;
        }
        if self.node_children(node).iter().any(|child| {
            child.child_node().is_some_and(|index| {
                self.nodes[index].goal_outcome == Some(true) && self.child_solved(child) == Some(1)
            })
        }) {
            return true;
        }
        // 不证明指定杀法。普通必胜后只留一小段探索，且不突破调用方总预算。
        self.goal_search_until
            .is_none_or(|until| self.nodes[node].visits >= until)
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
            eval.moves_left = self.nodes[node].moves_left;
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

    pub(super) fn goal_bonus_weight(&self, node: usize) -> f32 {
        if self.goal_turn(node) {
            0.25 * self.mating_goal.strength
        } else {
            0.0
        }
    }

    pub(super) fn best_goal_root_child(&self, node: usize) -> Option<usize> {
        let children = self.node_children(node);
        let draw_score = self.node_draw_score(node);
        let goal_weight = self.goal_bonus_weight(node);
        children
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                self.child_priority(left)
                    .cmp(&self.child_priority(right))
                    .then_with(|| {
                        if self.child_solved(left) == Some(1) {
                            self.child_goal_value(left)
                                .total_cmp(&self.child_goal_value(right))
                                .then_with(|| {
                                    let distance = |child: &AzChild| {
                                        child
                                            .child_node()
                                            .and_then(|index| self.nodes[index].proven_moves_left)
                                            .unwrap_or(usize::MAX)
                                    };
                                    distance(right).cmp(&distance(left))
                                })
                        } else {
                            left.visits.cmp(&right.visits).then_with(|| {
                                (self.child_q(left, draw_score)
                                    + goal_weight * self.child_goal_value(left))
                                .total_cmp(
                                    &(self.child_q(right, draw_score)
                                        + goal_weight * self.child_goal_value(right)),
                                )
                            })
                        }
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
    }

    pub(super) fn prepare_goal(&mut self, node: usize) {
        let effective = self.goal_at(node);
        self.eval_scratch.mating_goal = effective;
        if self.mating_goal.enabled() && !effective.enabled() {
            self.nodes[node].goal_value = 0.0;
            self.nodes[node].goal_outcome = Some(false);
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
            ("ordinary_eval", vec![AzMatingGoal::NONE; 14]),
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
                "{label}: evaluations=1400 ms={:.3}",
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
            "goal_search: searches=56 simulations={simulations} ms={:.3} checksum={checksum:.9}",
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
        assert_eq!(tree.goal_bonus_weight(0), 0.0);
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
    fn ordinary_win_has_bounded_goal_exploration_and_kld_works_with_goals() {
        let model = AzNnue::random(8, 100);
        let mut tree = two_move_tree(&Position::startpos(), &model, Color::Red);
        tree.set_proven(0, 1);
        let until = tree.goal_search_until.unwrap();
        assert_eq!(until, 64);
        tree.nodes[0].visits = until - 1;
        assert!(!tree.search_finished(0));
        tree.nodes[0].visits = until;
        assert!(tree.search_finished(0));
        // 重复胜负更新不能重置额外预算。
        tree.set_proven(0, 1);
        assert_eq!(tree.goal_search_until, Some(until));

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
        assert!(result.simulations > 1 && result.simulations <= 65);
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
        assert_eq!(target.policy, 1.0);
        assert_eq!(generic.policy, 0.0);
    }
}
