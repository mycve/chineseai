use crate::az::nnue::{canonical_square, piece_absolute_feature_index};
use crate::xiangqi::{
    BOARD_FILES, BOARD_SIZE, Color, Move, Piece, PieceKind, Position, color_index, piece_kind_index,
};

use super::*;

pub(crate) struct AzEvalScratch {
    // NNUE 热路径复用特征存储，避免每个 MCTS 叶节点分配并排序 Vec。
    pub(crate) features: Vec<usize>,
    pub(crate) hidden: Vec<f32>,
    pub(crate) shared_hidden: Vec<f32>,
    pub(crate) policy_context: Vec<f32>,
    pub(crate) policy_accumulator_context: [f32; POLICY_ACCUMULATOR_RANK],
    pub(crate) policy_piece_square_scores: Vec<f32>,
    pub(crate) value_head: Vec<f32>,
    pub(crate) value_king_piece_accumulator: Vec<f32>,
    pub(crate) value_king_piece_precomputed: Option<ValueKingState>,
    pub(crate) value_threat_accumulator: Vec<f32>,
    pub(crate) value_threat_activation: Vec<f32>,
    pub(crate) value_threat_precomputed: Option<([f32; VALUE_THREAT_RANK], usize)>,
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
            hidden: vec![0.0; hidden_size],
            shared_hidden: vec![0.0; hidden_size],
            policy_context: vec![0.0; POLICY_MOVE_CONTEXT_SIZE],
            policy_accumulator_context: [0.0; POLICY_ACCUMULATOR_RANK],
            policy_piece_square_scores: Vec::new(),
            value_head: vec![0.0; VALUE_HEAD_SIZE],
            value_king_piece_accumulator: vec![0.0; VALUE_HEAD_SIZE],
            value_king_piece_precomputed: None,
            value_threat_accumulator: vec![0.0; VALUE_THREAT_RANK],
            value_threat_precomputed: None,
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
            hidden: Vec::new(),
            shared_hidden: Vec::new(),
            policy_context: Vec::new(),
            policy_accumulator_context: [0.0; POLICY_ACCUMULATOR_RANK],
            policy_piece_square_scores: Vec::new(),
            value_head: Vec::new(),
            value_king_piece_accumulator: Vec::new(),
            value_king_piece_precomputed: None,
            value_threat_accumulator: Vec::new(),
            value_threat_precomputed: None,
            value_threat_activation: Vec::new(),
            policy_gives_check: Vec::new(),
            attack_masks: [0u128; 2],
            policy_inputs_ready: false,
            logits: Vec::new(),
            priors: Vec::new(),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ValueKingState {
    pub(crate) pool: [f32; VALUE_KING_PIECE_RANK],
    rooks: [u8; 2],
    minors: [u8; 2],
    pieces: u8,
}

impl ValueKingState {
    pub(crate) fn new(model: &AzNnue, position: &Position, side: Color) -> Self {
        let mut state = Self {
            pool: [0.0; VALUE_KING_PIECE_RANK],
            rooks: [0; 2],
            minors: [0; 2],
            pieces: 0,
        };
        visit_value_king_piece_features(position, side, |feature| {
            let start = feature * VALUE_KING_PIECE_RANK;
            for (sum, &weight) in state
                .pool
                .iter_mut()
                .zip(&model.value_king_piece_hidden[start..start + VALUE_KING_PIECE_RANK])
            {
                *sum += weight;
            }
        });
        for square in position.occupied_squares() {
            state.material(position.piece_at(square).unwrap(), side, 1);
        }
        state
    }

    pub(crate) fn update(
        mut self,
        model: &AzNnue,
        before: &Position,
        after: &Position,
        side: Color,
        mut changed: u128,
    ) -> Self {
        let buckets = canonical_buckets_for_perspective(after, side);
        if canonical_buckets_for_perspective(before, side) != buckets {
            return Self::new(model, after, side);
        }
        while changed != 0 {
            let square = changed.trailing_zeros() as usize;
            changed &= changed - 1;
            for (piece, sign) in [(before.piece_at(square), -1), (after.piece_at(square), 1)] {
                let Some(piece) = piece else {
                    continue;
                };
                self.material(piece, side, sign);
                let index = piece_absolute_feature_index(side, piece);
                let canonical = canonical_square(side, square);
                for (king, bucket) in [(0, buckets.0), (1, buckets.1)] {
                    let feature = ((king * crate::az::nnue::V2_KING_BUCKETS + bucket) * 14 + index)
                        * BOARD_SIZE
                        + canonical;
                    let start = feature * VALUE_KING_PIECE_RANK;
                    for (sum, &weight) in self
                        .pool
                        .iter_mut()
                        .zip(&model.value_king_piece_hidden[start..start + VALUE_KING_PIECE_RANK])
                    {
                        *sum += f32::from(sign) * weight;
                    }
                }
            }
        }
        self
    }

    fn material(&mut self, piece: Piece, side: Color, delta: i8) {
        let index = usize::from(piece.color != side);
        self.pieces = (self.pieces as i16 + i16::from(delta)) as u8;
        match piece.kind {
            PieceKind::Rook => {
                self.rooks[index] = (self.rooks[index] as i16 + i16::from(delta)) as u8
            }
            PieceKind::Horse | PieceKind::Cannon => {
                self.minors[index] = (self.minors[index] as i16 + i16::from(delta)) as u8
            }
            _ => {}
        }
    }

    pub(crate) fn bucket(&self) -> usize {
        super::inference::value_material_bucket_counts(
            self.rooks.map(usize::from),
            self.minors.map(usize::from),
        )
    }
    pub(crate) fn active(&self) -> usize {
        usize::from(self.pieces) * 2
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
    let h = model.hidden_size;

    add_scaled_rows(
        hidden,
        [
            super::simd::feature_row(&model.input_hidden, h, feature),
            super::simd::feature_row(&model.input_piece_hidden, h, piece_index),
            super::simd::feature_row(&model.input_rank_hidden, h, rank),
            super::simd::feature_row(&model.input_file_hidden, h, file),
            super::simd::feature_row(
                &model.input_king_piece_hidden,
                h,
                structural_king_piece_index(0, buckets.0, piece_index),
            ),
            super::simd::feature_row(
                &model.input_king_piece_hidden,
                h,
                structural_king_piece_index(1, buckets.1, piece_index),
            ),
        ],
        scale,
    );
}
