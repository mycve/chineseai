//! 共享局面编码：双视角主干、统一上下文和供两头复用的威胁关系。
use super::*;

impl AzNnue {
    /// 共享编码只负责局面：双视角棋子表示、统一上下文和威胁关系。
    pub(super) fn encode_shared_position(
        &self,
        position: &Position,
        moves: &[Move],
        rules: &[f32; RULE_CONTEXT_SIZE],
        history: &[f32; HISTORY_CONTEXT_SIZE],
        scratch: &mut AzEvalScratch,
    ) {
        scratch.policy_inputs_ready = false;
        if self.context_active {
            self.fill_policy_inputs(position, moves, scratch, true);
            let checks = check_context_features(
                position,
                moves,
                &scratch.policy_gives_check,
                scratch.attack_masks,
            );
            let context = super::context::position_context_features(rules, &checks, history);
            self.add_context_to_hidden(&context, &mut scratch.hidden);
        }
        relu_in_place(&mut scratch.hidden);
        rms_norm_in_place(&mut scratch.hidden);
        self.shared_hidden_into(&scratch.hidden, &mut scratch.shared_hidden);
        std::mem::swap(&mut scratch.hidden, &mut scratch.shared_hidden);
        self.encode_threat_relations(
            position,
            &mut scratch.threat_accumulator,
            &mut scratch.threat_activation,
            scratch.threat_precomputed.take(),
        );
    }

    fn encode_threat_relations(
        &self,
        position: &Position,
        accumulator: &mut Vec<f32>,
        activation: &mut Vec<f32>,
        precomputed: Option<super::node_threat::ThreatEncoding>,
    ) {
        activation.resize(VALUE_THREAT_RANK * 2, 0.0);
        if !self.threat_active {
            activation.fill(0.0);
            return;
        }
        crate::scope_profile!("az.eval.shared_threat");
        accumulator.resize(VALUE_THREAT_RANK, 0.0);
        accumulator.fill(0.0);
        let perspective = position.side_to_move();
        {
            crate::scope_profile!("az.eval.shared_threat.accumulate");
            let mut active = 0usize;
            if let Some(cached) = precomputed {
                accumulator.copy_from_slice(&cached.sum);
                active = cached.active;
            } else {
                visit_value_threat_entries_from(position, perspective, u128::MAX, |entry| {
                    let feature = entry as u16 as usize;
                    active += 1;
                    let row = &self.threat_embedding
                        [feature * VALUE_THREAT_RANK..(feature + 1) * VALUE_THREAT_RANK];
                    for (sum, weight) in accumulator.iter_mut().zip(row) {
                        *sum += weight;
                    }
                });
            }
            let scale = 1.0 / (active.max(1) as f32).sqrt();
            for value in accumulator.iter_mut() {
                *value *= scale;
            }
        }
        for rank in 0..VALUE_THREAT_RANK {
            let value = accumulator[rank];
            activation[rank] = value;
            activation[VALUE_THREAT_RANK + rank] = value * value;
        }
    }

    pub(crate) fn shared_hidden_into(&self, hidden: &[f32], output: &mut Vec<f32>) {
        output.resize(self.hidden_size, 0.0);
        matrix_dot_into(hidden, &self.shared_hidden, output);
        for (value, &bias) in output.iter_mut().zip(&self.shared_bias) {
            *value = (*value + bias).max(0.0);
        }
    }

    fn add_context_to_hidden(&self, context: &[f32; POSITION_CONTEXT_SIZE], hidden: &mut [f32]) {
        for (&value, row) in context
            .iter()
            .zip(self.context_hidden.chunks_exact(2 * self.hidden_size))
        {
            if value != 0.0 {
                for (target, &weight) in hidden.iter_mut().zip(row) {
                    *target += value * weight;
                }
            }
        }
    }
}
