use crate::az::nnue::{canonical_square, piece_absolute_feature_index};
use crate::xiangqi::{
    BOARD_FILES, BOARD_SIZE, Color, Move, Piece, Position, color_index, piece_kind_index,
};

use super::*;

pub(crate) struct AzEvalScratch {
    // NNUE 热路径复用特征存储，避免每个 MCTS 叶节点分配并排序 Vec。
    pub(crate) features: Vec<usize>,
    pub(crate) hidden: Vec<f32>,
    pub(crate) policy_context: Vec<f32>,
    pub(crate) policy_accumulator_context: [f32; POLICY_ACCUMULATOR_RANK],
    pub(crate) policy_piece_square_scores: Vec<f32>,
    pub(crate) value_head: Vec<f32>,
    pub(crate) value_king_piece_accumulator: Vec<f32>,
    pub(crate) value_threat_accumulator: Vec<f32>,
    pub(crate) value_threat_activation: Vec<f32>,
    pub(crate) policy_gives_check: Vec<f32>,
    pub(crate) logits: Vec<f32>,
    pub(crate) priors: Vec<f32>,
}

impl AzEvalScratch {
    pub(crate) fn new(arch: AzNnueArch) -> Self {
        let hidden_size = arch.hidden_size;
        Self {
            features: Vec::with_capacity(48),
            hidden: vec![0.0; hidden_size],
            policy_context: vec![0.0; POLICY_MOVE_CONTEXT_SIZE],
            policy_accumulator_context: [0.0; POLICY_ACCUMULATOR_RANK],
            policy_piece_square_scores: Vec::new(),
            value_head: vec![0.0; VALUE_HEAD_SIZE],
            value_king_piece_accumulator: vec![0.0; VALUE_HEAD_SIZE],
            value_threat_accumulator: vec![0.0; VALUE_THREAT_RANK],
            value_threat_activation: vec![0.0; VALUE_THREAT_RANK * 2],
            policy_gives_check: Vec::with_capacity(192),
            logits: Vec::with_capacity(192),
            priors: Vec::with_capacity(192),
        }
    }

    pub(crate) fn empty() -> Self {
        Self {
            features: Vec::new(),
            hidden: Vec::new(),
            policy_context: Vec::new(),
            policy_accumulator_context: [0.0; POLICY_ACCUMULATOR_RANK],
            policy_piece_square_scores: Vec::new(),
            value_head: Vec::new(),
            value_king_piece_accumulator: Vec::new(),
            value_threat_accumulator: Vec::new(),
            value_threat_activation: Vec::new(),
            policy_gives_check: Vec::new(),
            logits: Vec::new(),
            priors: Vec::new(),
        }
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
        let mut features = Vec::with_capacity(32);
        for sq in 0..BOARD_SIZE {
            if let Some(piece) = position.piece_at(sq) {
                let piece_index = piece_absolute_feature_index(perspective, piece);
                features.push(piece_index * BOARD_SIZE + canonical_square(perspective, sq));
            }
        }
        model.input_embedding_linear_into_slice(&features, hidden);
    }

    #[cfg_attr(not(test), allow(dead_code))]
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
        let before_buckets = canonical_buckets_for_perspective(before, perspective);
        let after_buckets = canonical_buckets_for_perspective(after, perspective);
        if before_buckets != after_buckets {
            // 将帅移动会改变所有棋子的王桶结构项，少见且必须完整刷新。
            Self::refresh_perspective(model, after, perspective, hidden);
            return;
        }
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

pub(crate) fn canonical_buckets_for_perspective(position: &Position, perspective: Color) -> (usize, usize) {
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
