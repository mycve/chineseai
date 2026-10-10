use crate::az::nnue::{canonical_square, piece_absolute_feature_index};
use crate::xiangqi::{
    BOARD_FILES, BOARD_SIZE, Color, Move, Piece, Position, color_index, piece_kind_index,
};

use super::*;

pub(crate) struct AzEvalScratch {
    // NNUE 热路径复用特征存储，避免每个 MCTS 叶节点分配并排序 Vec。
    pub(crate) features: Vec<usize>,
    pub(crate) reflected_moves: Vec<Move>,
    pub(crate) hidden: Vec<f32>,
    pub(crate) shared_hidden: Vec<f32>,
    pub(crate) policy_context: Vec<f32>,
    pub(crate) policy_accumulator_context: [f32; POLICY_ACCUMULATOR_RANK],
    pub(crate) policy_piece_square_scores: Vec<f32>,
    pub(crate) value_head: Vec<f32>,
    pub(crate) value_king_piece_accumulator: Vec<f32>,
    pub(crate) value_threat_accumulator: Vec<f32>,
    pub(crate) value_threat_activation: Vec<f32>,
    pub(crate) policy_gives_check: Vec<f32>,
    /// 本节点的双方攻击位板。主干之前的标量块与策略头的战术块共用同一份，
    /// 避免为同一个局面算两遍。
    pub(crate) attack_masks: [u128; 2],
    /// `policy_gives_check` / `attack_masks` 是否已经是**当前局面**的值。
    /// 与 `scratch` 跨节点复用的做法配合：每次评估开头清零。
    pub(crate) policy_inputs_ready: bool,
    pub(crate) logits: Vec<f32>,
    pub(crate) priors: Vec<f32>,
}

impl AzEvalScratch {
    pub(crate) fn new(arch: AzNnueArch) -> Self {
        let hidden_size = arch.hidden_size;
        Self {
            features: Vec::with_capacity(48),
            reflected_moves: Vec::with_capacity(192),
            hidden: vec![0.0; hidden_size],
            shared_hidden: vec![0.0; hidden_size],
            policy_context: vec![0.0; POLICY_MOVE_CONTEXT_SIZE],
            policy_accumulator_context: [0.0; POLICY_ACCUMULATOR_RANK],
            policy_piece_square_scores: Vec::new(),
            value_head: vec![0.0; VALUE_HEAD_SIZE],
            value_king_piece_accumulator: vec![0.0; VALUE_HEAD_SIZE],
            value_threat_accumulator: vec![0.0; VALUE_THREAT_RANK],
            value_threat_activation: vec![0.0; VALUE_THREAT_RANK * 2],
            policy_gives_check: Vec::with_capacity(192),
            attack_masks: [0u128; 2],
            policy_inputs_ready: false,
            logits: Vec::with_capacity(192),
            priors: Vec::with_capacity(192),
        }
    }

    pub(crate) fn empty() -> Self {
        Self {
            features: Vec::new(),
            reflected_moves: Vec::new(),
            hidden: Vec::new(),
            shared_hidden: Vec::new(),
            policy_context: Vec::new(),
            policy_accumulator_context: [0.0; POLICY_ACCUMULATOR_RANK],
            policy_piece_square_scores: Vec::new(),
            value_head: Vec::new(),
            value_king_piece_accumulator: Vec::new(),
            value_threat_accumulator: Vec::new(),
            value_threat_activation: Vec::new(),
            policy_gives_check: Vec::new(),
            attack_masks: [0u128; 2],
            policy_inputs_ready: false,
            logits: Vec::new(),
            priors: Vec::new(),
        }
    }
}

/// 一次走子的规范坐标上下文，供主干与策略累加器共享。
pub(crate) struct CanonicalTransition {
    pub(crate) perspective: Color,
    pub(crate) reflected: bool,
    pub(crate) after_reflected: bool,
    pub(crate) before_buckets: (usize, usize),
    pub(crate) after_buckets: (usize, usize),
}

impl CanonicalTransition {
    pub(crate) fn new(before: &Position, after: &Position, perspective: Color) -> Self {
        let reflected = super::reflection::board_orientation_for(before, perspective)
            == std::cmp::Ordering::Greater;
        let after_reflected = super::reflection::board_orientation_for(after, perspective)
            == std::cmp::Ordering::Greater;
        Self {
            perspective,
            reflected,
            after_reflected,
            before_buckets: canonical_buckets_for_reflection(before, perspective, reflected),
            after_buckets: canonical_buckets_for_reflection(after, perspective, after_reflected),
        }
    }

    pub(crate) fn needs_refresh(&self) -> bool {
        self.reflected != self.after_reflected || self.before_buckets != self.after_buckets
    }
}

/// 搜索节点使用的双视角 NNUE 累加器，不包含随每步老化的历史特征。
#[derive(Clone, Debug)]
pub(crate) struct AzEvalAccumulator {
    pub(crate) hidden_sum: Vec<f32>,
}

impl AzEvalAccumulator {
    pub(crate) fn new(model: &AzNnue, position: &Position) -> Self {
        let mut accumulator = Self {
            hidden_sum: vec![0.0; model.hidden_size * 2],
        };
        accumulator.refresh(model, position);
        accumulator
    }

    pub(crate) fn refresh(&mut self, model: &AzNnue, position: &Position) {
        for perspective in [Color::Red, Color::Black] {
            let index = color_index(perspective);
            let start = index * model.hidden_size;
            Self::refresh_perspective(
                model,
                position,
                perspective,
                &mut self.hidden_sum[start..start + model.hidden_size],
            );
        }
    }

    pub(crate) fn refresh_perspective(
        model: &AzNnue,
        position: &Position,
        perspective: Color,
        hidden: &mut [f32],
    ) {
        let reflected = super::reflection::board_orientation_for(position, perspective)
            == std::cmp::Ordering::Greater;
        Self::refresh_oriented(model, position, perspective, reflected, hidden);
    }

    fn refresh_oriented(
        model: &AzNnue,
        position: &Position,
        perspective: Color,
        reflected: bool,
        hidden: &mut [f32],
    ) {
        let mut features = Vec::with_capacity(32);
        for sq in 0..BOARD_SIZE {
            let source = if reflected {
                crate::az::nnue::mirror_file_square(sq)
            } else {
                sq
            };
            if let Some(piece) = position.piece_at(source) {
                let piece_index = piece_absolute_feature_index(perspective, piece);
                features.push(piece_index * BOARD_SIZE + canonical_square(perspective, sq));
            }
        }
        model.input_embedding_linear_into_slice(&features, hidden);
    }

    #[cfg(test)]
    pub(crate) fn apply_transition_to_hidden(
        model: &AzNnue,
        before: &Position,
        after: &Position,
        mv: Move,
        moved: Piece,
        captured: Option<Piece>,
        hidden_sum: &mut [f32],
    ) {
        debug_assert_eq!(hidden_sum.len(), model.hidden_size * 2);
        for perspective in [Color::Red, Color::Black] {
            let start = color_index(perspective) * model.hidden_size;
            Self::apply_transition_for_perspective(
                model,
                before,
                after,
                mv,
                moved,
                captured,
                perspective,
                &mut hidden_sum[start..start + model.hidden_size],
            );
        }
    }

    #[cfg(test)]
    pub(crate) fn apply_transition_for_perspective(
        model: &AzNnue,
        before: &Position,
        after: &Position,
        mv: Move,
        moved: Piece,
        captured: Option<Piece>,
        perspective: Color,
        hidden: &mut [f32],
    ) {
        let context = CanonicalTransition::new(before, after, perspective);
        Self::apply_canonical_transition(model, after, mv, moved, captured, &context, hidden);
    }

    pub(crate) fn apply_canonical_transition(
        model: &AzNnue,
        after: &Position,
        mv: Move,
        moved: Piece,
        captured: Option<Piece>,
        context: &CanonicalTransition,
        hidden: &mut [f32],
    ) {
        let perspective = context.perspective;
        let before_buckets = context.before_buckets;
        let after_buckets = context.after_buckets;
        if context.needs_refresh() {
            Self::refresh_oriented(model, after, perspective, context.after_reflected, hidden);
            return;
        }
        let mv = if context.reflected {
            crate::az::nnue::mirror_file_move(mv)
        } else {
            mv
        };
        add_canonical_piece_contribution(
            model,
            hidden,
            perspective,
            before_buckets,
            mv.from as usize,
            moved,
            -1.0,
        );
        if let Some(captured) = captured {
            add_canonical_piece_contribution(
                model,
                hidden,
                perspective,
                before_buckets,
                mv.to as usize,
                captured,
                -1.0,
            );
        }
        add_canonical_piece_contribution(
            model,
            hidden,
            perspective,
            after_buckets,
            mv.to as usize,
            moved,
            1.0,
        );
    }

    pub(crate) fn hidden_for_slice(hidden_sum: &[f32], hidden_size: usize, side: Color) -> &[f32] {
        let start = color_index(side) * hidden_size;
        &hidden_sum[start..start + hidden_size]
    }

    pub(crate) fn into_hidden_sum(self) -> Vec<f32> {
        self.hidden_sum
    }
}

pub(crate) fn canonical_buckets_for_perspective(
    position: &Position,
    perspective: Color,
) -> (usize, usize) {
    let us = position
        .general_square(perspective)
        .map(|sq| canonical_general_bucket(0, canonical_square_for(perspective, sq)))
        .unwrap_or(4);
    let them = position
        .general_square(perspective.opposite())
        .map(|sq| canonical_general_bucket(7, canonical_square_for(perspective, sq)))
        .unwrap_or(4);
    (us, them)
}

pub(crate) fn canonical_buckets_for_reflection(
    position: &Position,
    perspective: Color,
    reflected: bool,
) -> (usize, usize) {
    let orient = |sq| {
        canonical_square_for(
            perspective,
            if reflected {
                crate::az::nnue::mirror_file_square(sq)
            } else {
                sq
            },
        )
    };
    (
        position
            .general_square(perspective)
            .map(|sq| canonical_general_bucket(0, orient(sq)))
            .unwrap_or(4),
        position
            .general_square(perspective.opposite())
            .map(|sq| canonical_general_bucket(7, orient(sq)))
            .unwrap_or(4),
    )
}

pub(crate) fn add_canonical_piece_contribution(
    model: &AzNnue,
    hidden: &mut [f32],
    perspective: Color,
    buckets: (usize, usize),
    sq: usize,
    piece: Piece,
    scale: f32,
) {
    let relative_color = if piece.color == perspective { 0 } else { 7 };
    let piece_index = relative_color + piece_kind_index(piece.kind);
    let relative_square = canonical_square_for(perspective, sq);
    let feature = piece_index * BOARD_SIZE + relative_square;
    let rank = relative_square / BOARD_FILES;
    let file = relative_square % BOARD_FILES;
    add_scaled_feature_row(
        hidden,
        &model.input_hidden,
        model.hidden_size,
        feature,
        scale,
    );
    add_scaled_feature_row(
        hidden,
        &model.input_piece_hidden,
        model.hidden_size,
        piece_index,
        scale,
    );
    add_scaled_feature_row(
        hidden,
        &model.input_rank_hidden,
        model.hidden_size,
        rank,
        scale,
    );
    add_scaled_feature_row(
        hidden,
        &model.input_file_hidden,
        model.hidden_size,
        file,
        scale,
    );
    add_scaled_feature_row(
        hidden,
        &model.input_king_piece_hidden,
        model.hidden_size,
        structural_king_piece_index(0, buckets.0, piece_index),
        scale,
    );
    add_scaled_feature_row(
        hidden,
        &model.input_king_piece_hidden,
        model.hidden_size,
        structural_king_piece_index(1, buckets.1, piece_index),
        scale,
    );
}

#[cfg(test)]
mod reflection_tests {
    use super::*;

    #[test]
    fn reflection_full_and_incremental_follow_real_moves() {
        let mut model = AzNnue::random(96, 20261010);
        for (i, w) in model.value_history_output.iter_mut().enumerate() {
            *w = ((i % 19) as f32 - 9.0) * 0.002;
        }
        model.rebuild_value_history();
        for (i, w) in model.policy_move_bias.iter_mut().enumerate() {
            *w = ((i % 31) as f32 - 15.0) * 0.01;
        }
        for weights in [
            &mut model.policy_accumulator_move,
            &mut model.policy_move_context,
            &mut model.policy_consequence_output,
            &mut model.policy_threat_context,
            &mut model.policy_sparse_table,
            &mut model.policy_sparse_factor,
            &mut model.policy_tactical,
            &mut model.value_threat_output,
            &mut model.check_context_hidden,
        ] {
            for (i, w) in weights.iter_mut().enumerate() {
                *w = ((i % 23) as f32 - 11.0) * 0.001;
            }
        }
        model.rebuild_policy_cache();
        model.rebuild_policy_tactical();
        model.rebuild_value_threat();
        model.rebuild_check_context();
        let mut p = Position::startpos();
        let mut history = p.initial_rule_history();
        let mut accumulator = AzEvalAccumulator::new(&model, &p);
        let mut mirror_accumulator = AzEvalAccumulator::new(&model, &p.mirror_files());
        let mut policy = [
            model.policy_accumulator(&p, Color::Red),
            model.policy_accumulator(&p, Color::Black),
        ];
        for text in [
            "b0c2", "b9c7", "c3c4", "c6c5", "c4c5", "a6a5", "e0e1", "e9e8",
        ] {
            let (moves, flags): (Vec<_>, Vec<_>) = p
                .legal_moves_with_rules_and_repetition(&history)
                .into_iter()
                .map(|(m, r)| (m, u8::from(r)))
                .unzip();
            let context = rule_context_features(&p, &history);
            let mut full = AzEvalScratch::new(model.arch);
            let evaluated = model.evaluate_with_scratch_output_with_repetition_and_history(
                &p, &moves, &flags, &context, &history, &mut full,
            );
            let mp = p.mirror_files();
            let mm: Vec<_> = moves
                .iter()
                .copied()
                .map(crate::az::nnue::mirror_file_move)
                .collect();
            let mh: Vec<_> = history
                .iter()
                .copied()
                .map(|mut e| {
                    e.mv = e.mv.map(crate::az::nnue::mirror_file_move);
                    e
                })
                .collect();
            let mut mirrored = AzEvalScratch::new(model.arch);
            let other = model.evaluate_with_scratch_output_with_repetition_and_history(
                &mp,
                &mm,
                &flags,
                &context,
                &mh,
                &mut mirrored,
            );
            assert_eq!(
                evaluated.value_wdl, other.value_wdl,
                "mirror WDL before {text}"
            );
            assert_eq!(full.logits, mirrored.logits, "mirror logits before {text}");
            if history.len() == 1 {
                for (i, mv) in moves.iter().enumerate() {
                    let j = moves
                        .iter()
                        .position(|m| *m == crate::az::nnue::mirror_file_move(*mv))
                        .unwrap();
                    assert_eq!(full.logits[i], full.logits[j], "fixed-point pair");
                }
            }
            let mut mi = AzEvalScratch::new(model.arch);
            let mie = model.evaluate_incremental_with_scratch_output_with_history(
                &mp,
                &mirror_accumulator.hidden_sum,
                &model.policy_accumulator(&mp, mp.side_to_move()),
                &mm,
                &flags,
                &context,
                &mh,
                &mut mi,
            );
            for (a, b) in other.value_wdl.iter().zip(mie.value_wdl) {
                assert!((a - b).abs() < 2e-5);
            }
            for (a, b) in mirrored.logits.iter().zip(&mi.logits) {
                assert!((a - b).abs() < 2e-5);
            }

            let mut incremental = AzEvalScratch::new(model.arch);
            let inc = model.evaluate_incremental_with_scratch_output_with_history(
                &p,
                &accumulator.hidden_sum,
                &policy[color_index(p.side_to_move())],
                &moves,
                &flags,
                &context,
                &history,
                &mut incremental,
            );
            for (a, b) in evaluated.value_wdl.iter().zip(inc.value_wdl) {
                assert!((a - b).abs() < 2e-5, "incremental WDL {text}: {a} {b}");
            }
            for (a, b) in full.logits.iter().zip(&incremental.logits) {
                assert!((a - b).abs() < 2e-5, "incremental logits {text}: {a} {b}");
            }
            let features = super::super::history::history_features(&p, &history);
            let mut explicit = AzEvalScratch::new(model.arch);
            let exp = model.evaluate_with_scratch_output_with_repetition_and_history_features(
                &p,
                &moves,
                &flags,
                &context,
                &features,
                &mut explicit,
            );
            for (a, b) in evaluated.value_wdl.iter().zip(exp.value_wdl) {
                assert!((a - b).abs() < 2e-5);
            }
            let mv = p.parse_uci_move(text).unwrap();
            assert!(moves.contains(&mv), "fixture move {text}");
            let moved = p.piece_at(mv.from as usize).unwrap();
            let captured = p.piece_at(mv.to as usize);
            let mut next = p.clone();
            next.make_move(mv);
            history.push(p.rule_history_entry_after_move(mv));
            AzEvalAccumulator::apply_transition_to_hidden(
                &model,
                &p,
                &next,
                mv,
                moved,
                captured,
                &mut accumulator.hidden_sum,
            );
            for side in [Color::Red, Color::Black] {
                model.apply_policy_transition(
                    &p,
                    &next,
                    mv,
                    moved,
                    captured,
                    side,
                    &mut policy[color_index(side)],
                );
            }
            let mmv = crate::az::nnue::mirror_file_move(mv);
            AzEvalAccumulator::apply_transition_to_hidden(
                &model,
                &p.mirror_files(),
                &next.mirror_files(),
                mmv,
                moved,
                captured,
                &mut mirror_accumulator.hidden_sum,
            );
            p = next;
        }
    }
}
