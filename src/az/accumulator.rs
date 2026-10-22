use super::features::{MOTIF_RANK, RELATION_RANK};
use super::nnue::{canonical_square, piece_absolute_feature_index};
use super::*;
use crate::xiangqi::{Color, Move, Piece, Position, color_index};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FeatureSums {
    pub king: [f32; VALUE_KING_PIECE_RANK],
    pub relations: [f32; RELATION_RANK],
    pub motifs: [f32; MOTIF_RANK],
}
pub(crate) struct AzEvalScratch {
    pub hidden: Vec<f32>,
    pub value_head: Vec<f32>,
    pub logits: Vec<f32>,
    pub priors: Vec<f32>,
    pub policy_gives_check: Vec<f32>,
    pub feature_precomputed: Option<FeatureSums>,
    pub graph_precomputed: Option<super::features::Graph>,
}
impl AzEvalScratch {
    pub fn new(arch: AzNnueArch) -> Self {
        Self {
            hidden: vec![0.; arch.hidden_size],
            value_head: vec![0.; VALUE_HEAD_SIZE],
            logits: Vec::with_capacity(128),
            priors: Vec::with_capacity(128),
            policy_gives_check: Vec::with_capacity(128),
            feature_precomputed: None,
            graph_precomputed: None,
        }
    }
    pub fn empty() -> Self {
        Self::new(AzNnueArch::with_hidden_size(1))
    }
}
#[derive(Clone, Debug)]
pub(crate) struct AzEvalAccumulator {
    pub hidden_sum: Vec<f32>,
}
impl AzEvalAccumulator {
    pub fn new(model: &AzNnue, p: &Position) -> Self {
        let mut result = Self {
            hidden_sum: vec![0.; model.hidden_size * 2],
        };
        result.refresh(model, p);
        result
    }
    pub fn refresh(&mut self, model: &AzNnue, p: &Position) {
        for side in [Color::Red, Color::Black] {
            let start = color_index(side) * model.hidden_size;
            Self::refresh_perspective(
                model,
                p,
                side,
                &mut self.hidden_sum[start..start + model.hidden_size],
            );
        }
    }
    pub fn refresh_perspective(model: &AzNnue, p: &Position, side: Color, output: &mut [f32]) {
        output.fill(0.);
        for square in p.occupied_squares() {
            let piece = p.piece_at(square).unwrap();
            add_canonical_piece_contribution(model, output, side, square, piece, 1.);
        }
    }
    pub fn apply_transition_for_perspective(
        model: &AzNnue,
        _before: &Position,
        _after: &Position,
        mv: Move,
        moved: Piece,
        captured: Option<Piece>,
        side: Color,
        output: &mut [f32],
    ) {
        add_canonical_piece_contribution(model, output, side, mv.from as usize, moved, -1.);
        if let Some(piece) = captured {
            add_canonical_piece_contribution(model, output, side, mv.to as usize, piece, -1.);
        }
        add_canonical_piece_contribution(model, output, side, mv.to as usize, moved, 1.);
    }
    #[cfg(test)]
    pub fn apply_transition_to_hidden(
        model: &AzNnue,
        before: &Position,
        after: &Position,
        mv: Move,
        moved: Piece,
        captured: Option<Piece>,
        output: &mut [f32],
    ) {
        for side in [Color::Red, Color::Black] {
            let start = color_index(side) * model.hidden_size;
            Self::apply_transition_for_perspective(
                model,
                before,
                after,
                mv,
                moved,
                captured,
                side,
                &mut output[start..start + model.hidden_size],
            );
        }
    }
    pub fn hidden_for_slice(values: &[f32], h: usize, side: Color) -> &[f32] {
        let start = color_index(side) * h;
        &values[start..start + h]
    }
    pub fn into_hidden_sum(self) -> Vec<f32> {
        self.hidden_sum
    }
}
pub(crate) fn canonical_buckets_for_perspective(p: &Position, side: Color) -> (usize, usize) {
    let us = p.general_square(side).map_or(4, |s| {
        canonical_general_bucket(0, canonical_square(side, s))
    });
    let them = p.general_square(side.opposite()).map_or(4, |s| {
        canonical_general_bucket(7, canonical_square(side, s))
    });
    (us, them)
}
fn add_canonical_piece_contribution(
    model: &AzNnue,
    output: &mut [f32],
    side: Color,
    square: usize,
    piece: Piece,
    scale: f32,
) {
    let feature = piece_absolute_feature_index(side, piece) * 90 + canonical_square(side, square);
    add_scaled_row(output, &model.input_hidden, feature, scale);
}
