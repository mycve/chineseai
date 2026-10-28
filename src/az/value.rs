//! 价值专用关系、子力专家和剩余步数输出，消费共享局面编码。
use super::*;

impl AzNnue {
    /// 价值头消费共享编码，再加入价值专用的王棋子关系与子力专家。
    pub(super) fn evaluate_value_head(
        &self,
        position: &Position,
        scratch: &mut AzEvalScratch,
    ) -> ([f32; WDL_HEAD_SIZE], f32) {
        let bucket = self.value_king_piece_accumulate(
            position,
            &mut scratch.value_king_piece_accumulator,
            scratch.value_king_piece_precomputed.take(),
        );
        let threat_logits = std::array::from_fn(|output| {
            let start = output * VALUE_THREAT_RANK * 2;
            dot_product(
                &scratch.threat_activation,
                &self.value_threat_output[start..start + VALUE_THREAT_RANK * 2],
            )
        });
        self.value_wdl_from_hidden_into(
            bucket,
            &scratch.hidden,
            &scratch.value_king_piece_accumulator,
            &mut scratch.value_head,
            threat_logits,
        )
    }

    pub(crate) fn value_king_piece_accumulate(
        &self,
        position: &Position,
        accumulator: &mut Vec<f32>,
        precomputed: Option<ValueKingState>,
    ) -> usize {
        crate::scope_profile!("az.eval.value_king_piece");
        let (pooled, active, bucket) = if let Some(state) = precomputed {
            (state.pool, state.active(), state.bucket())
        } else {
            let mut pooled = [0.0; VALUE_KING_PIECE_RANK];
            let mut active = 0;
            let bucket =
                visit_value_king_piece_features(position, position.side_to_move(), |feature| {
                    active += 1;
                    let row = &self.value_king_piece_hidden
                        [feature * VALUE_KING_PIECE_RANK..(feature + 1) * VALUE_KING_PIECE_RANK];
                    for (sum, &weight) in pooled.iter_mut().zip(row) {
                        *sum += weight;
                    }
                });
            (pooled, active, bucket)
        };
        let scale = 1.0 / (active.max(1) as f32).sqrt();
        accumulator.resize(VALUE_HEAD_SIZE, 0.0);
        accumulator.fill(0.0);
        for (channel, value) in pooled.into_iter().enumerate() {
            let value = value * scale;
            let row = &self.value_king_piece_projection
                [channel * VALUE_HEAD_SIZE..(channel + 1) * VALUE_HEAD_SIZE];
            for (sum, &weight) in accumulator.iter_mut().zip(row) {
                *sum += value * weight;
            }
        }
        bucket
    }

    pub(crate) fn value_wdl_from_hidden_into(
        &self,
        bucket: usize,
        hidden: &[f32],
        king_piece: &[f32],
        value_head: &mut Vec<f32>,
        threat_logits: [f32; WDL_HEAD_SIZE],
    ) -> ([f32; WDL_HEAD_SIZE], f32) {
        value_head.resize(VALUE_HEAD_SIZE, 0.0);
        matrix_dot_into(hidden, &self.value_head_hidden, value_head);
        for (feature, value) in value_head.iter_mut().enumerate() {
            *value = (self.value_head_bias[feature] + (king_piece[feature] + *value)).max(0.0);
        }
        let mut logits = [0.0f32; WDL_HEAD_SIZE];
        for (out, logit) in logits.iter_mut().enumerate() {
            let start = (bucket * WDL_HEAD_SIZE + out) * VALUE_HEAD_SIZE;
            let row = &self.value_head_output[start..start + VALUE_HEAD_SIZE];
            *logit = dot_product(value_head, row) + threat_logits[out];
        }
        let wdl = softmax_fixed3(logits);
        let q = wdl[0] - wdl[2];
        (wdl, q)
    }

    /// 从共享隐藏层计算剩余步数预测。
    pub(crate) fn moves_left_from_hidden(&self, hidden: &[f32]) -> f32 {
        ((dot_product(hidden, &self.moves_left_output) + self.moves_left_bias[0] + 1.0).max(0.0)
            * MOVES_LEFT_SCALE)
            .min(4096.0)
    }

    pub(crate) fn rebuild_moves_left(&mut self) {
        self.moves_left_active =
            self.moves_left_output.iter().any(|&v| v != 0.0) || self.moves_left_bias[0] != 0.0;
    }
}
