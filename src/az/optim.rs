use super::AzNnue;

pub(super) const ADAMW_BETA1: f32 = 0.9;
pub(super) const ADAMW_BETA2: f32 = 0.999;
pub(super) const ADAMW_EPSILON: f32 = 1e-8;
pub(super) const ADAMW_WEIGHT_DECAY: f32 = 1e-4;

#[derive(Debug)]
pub(super) struct AdamWState {
    pub(super) step: u64,
    pub(super) beta1_power: f32,
    pub(super) beta2_power: f32,
    pub(super) input_hidden_m: Vec<f32>,
    pub(super) input_hidden_v: Vec<f32>,
    pub(super) hidden_bias_m: Vec<f32>,
    pub(super) hidden_bias_v: Vec<f32>,
    pub(super) trunk_weights_m: Vec<f32>,
    pub(super) trunk_weights_v: Vec<f32>,
    pub(super) trunk_biases_m: Vec<f32>,
    pub(super) trunk_biases_v: Vec<f32>,
    pub(super) trunk_global_weights_m: Vec<f32>,
    pub(super) trunk_global_weights_v: Vec<f32>,
    pub(super) board_conv1_weights_m: Vec<f32>,
    pub(super) board_conv1_weights_v: Vec<f32>,
    pub(super) board_conv1_bias_m: Vec<f32>,
    pub(super) board_conv1_bias_v: Vec<f32>,
    pub(super) board_conv2_weights_m: Vec<f32>,
    pub(super) board_conv2_weights_v: Vec<f32>,
    pub(super) board_conv2_bias_m: Vec<f32>,
    pub(super) board_conv2_bias_v: Vec<f32>,
    pub(super) board_global_m: Vec<f32>,
    pub(super) board_global_v: Vec<f32>,
    pub(super) global_hidden_m: Vec<f32>,
    pub(super) global_hidden_v: Vec<f32>,
    pub(super) global_bias_m: Vec<f32>,
    pub(super) global_bias_v: Vec<f32>,
    pub(super) value_intermediate_hidden_m: Vec<f32>,
    pub(super) value_intermediate_hidden_v: Vec<f32>,
    pub(super) value_intermediate_bias_m: Vec<f32>,
    pub(super) value_intermediate_bias_v: Vec<f32>,
    pub(super) value_logits_weights_m: Vec<f32>,
    pub(super) value_logits_weights_v: Vec<f32>,
    pub(super) value_logits_bias_m: Vec<f32>,
    pub(super) value_logits_bias_v: Vec<f32>,
    pub(super) policy_move_hidden_m: Vec<f32>,
    pub(super) policy_move_hidden_v: Vec<f32>,
    pub(super) policy_move_bias_m: Vec<f32>,
    pub(super) policy_move_bias_v: Vec<f32>,
}

impl AdamWState {
    pub(super) fn new(model: &AzNnue) -> Self {
        Self {
            step: 0,
            beta1_power: 1.0,
            beta2_power: 1.0,
            input_hidden_m: vec![0.0; model.input_hidden.len()],
            input_hidden_v: vec![0.0; model.input_hidden.len()],
            hidden_bias_m: vec![0.0; model.hidden_bias.len()],
            hidden_bias_v: vec![0.0; model.hidden_bias.len()],
            trunk_weights_m: vec![0.0; model.trunk_weights.len()],
            trunk_weights_v: vec![0.0; model.trunk_weights.len()],
            trunk_biases_m: vec![0.0; model.trunk_biases.len()],
            trunk_biases_v: vec![0.0; model.trunk_biases.len()],
            trunk_global_weights_m: vec![0.0; model.trunk_global_weights.len()],
            trunk_global_weights_v: vec![0.0; model.trunk_global_weights.len()],
            board_conv1_weights_m: vec![0.0; model.board_conv1_weights.len()],
            board_conv1_weights_v: vec![0.0; model.board_conv1_weights.len()],
            board_conv1_bias_m: vec![0.0; model.board_conv1_bias.len()],
            board_conv1_bias_v: vec![0.0; model.board_conv1_bias.len()],
            board_conv2_weights_m: vec![0.0; model.board_conv2_weights.len()],
            board_conv2_weights_v: vec![0.0; model.board_conv2_weights.len()],
            board_conv2_bias_m: vec![0.0; model.board_conv2_bias.len()],
            board_conv2_bias_v: vec![0.0; model.board_conv2_bias.len()],
            board_global_m: vec![0.0; model.board_global.len()],
            board_global_v: vec![0.0; model.board_global.len()],
            global_hidden_m: vec![0.0; model.global_hidden.len()],
            global_hidden_v: vec![0.0; model.global_hidden.len()],
            global_bias_m: vec![0.0; model.global_bias.len()],
            global_bias_v: vec![0.0; model.global_bias.len()],
            value_intermediate_hidden_m: vec![0.0; model.value_intermediate_hidden.len()],
            value_intermediate_hidden_v: vec![0.0; model.value_intermediate_hidden.len()],
            value_intermediate_bias_m: vec![0.0; model.value_intermediate_bias.len()],
            value_intermediate_bias_v: vec![0.0; model.value_intermediate_bias.len()],
            value_logits_weights_m: vec![0.0; model.value_logits_weights.len()],
            value_logits_weights_v: vec![0.0; model.value_logits_weights.len()],
            value_logits_bias_m: vec![0.0; model.value_logits_bias.len()],
            value_logits_bias_v: vec![0.0; model.value_logits_bias.len()],
            policy_move_hidden_m: vec![0.0; model.policy_move_hidden.len()],
            policy_move_hidden_v: vec![0.0; model.policy_move_hidden.len()],
            policy_move_bias_m: vec![0.0; model.policy_move_bias.len()],
            policy_move_bias_v: vec![0.0; model.policy_move_bias.len()],
        }
    }

    pub(super) fn matches(&self, model: &AzNnue) -> bool {
        self.input_hidden_m.len() == model.input_hidden.len()
            && self.input_hidden_v.len() == model.input_hidden.len()
            && self.hidden_bias_m.len() == model.hidden_bias.len()
            && self.hidden_bias_v.len() == model.hidden_bias.len()
            && self.trunk_weights_m.len() == model.trunk_weights.len()
            && self.trunk_weights_v.len() == model.trunk_weights.len()
            && self.trunk_biases_m.len() == model.trunk_biases.len()
            && self.trunk_biases_v.len() == model.trunk_biases.len()
            && self.trunk_global_weights_m.len() == model.trunk_global_weights.len()
            && self.trunk_global_weights_v.len() == model.trunk_global_weights.len()
            && self.board_conv1_weights_m.len() == model.board_conv1_weights.len()
            && self.board_conv1_weights_v.len() == model.board_conv1_weights.len()
            && self.board_conv1_bias_m.len() == model.board_conv1_bias.len()
            && self.board_conv1_bias_v.len() == model.board_conv1_bias.len()
            && self.board_conv2_weights_m.len() == model.board_conv2_weights.len()
            && self.board_conv2_weights_v.len() == model.board_conv2_weights.len()
            && self.board_conv2_bias_m.len() == model.board_conv2_bias.len()
            && self.board_conv2_bias_v.len() == model.board_conv2_bias.len()
            && self.board_global_m.len() == model.board_global.len()
            && self.board_global_v.len() == model.board_global.len()
            && self.global_hidden_m.len() == model.global_hidden.len()
            && self.global_hidden_v.len() == model.global_hidden.len()
            && self.global_bias_m.len() == model.global_bias.len()
            && self.global_bias_v.len() == model.global_bias.len()
            && self.value_intermediate_hidden_m.len() == model.value_intermediate_hidden.len()
            && self.value_intermediate_hidden_v.len() == model.value_intermediate_hidden.len()
            && self.value_intermediate_bias_m.len() == model.value_intermediate_bias.len()
            && self.value_intermediate_bias_v.len() == model.value_intermediate_bias.len()
            && self.value_logits_weights_m.len() == model.value_logits_weights.len()
            && self.value_logits_weights_v.len() == model.value_logits_weights.len()
            && self.value_logits_bias_m.len() == model.value_logits_bias.len()
            && self.value_logits_bias_v.len() == model.value_logits_bias.len()
            && self.policy_move_hidden_m.len() == model.policy_move_hidden.len()
            && self.policy_move_hidden_v.len() == model.policy_move_hidden.len()
            && self.policy_move_bias_m.len() == model.policy_move_bias.len()
            && self.policy_move_bias_v.len() == model.policy_move_bias.len()
    }

    pub(super) fn advance(&mut self) -> (f32, f32) {
        self.step += 1;
        self.beta1_power *= ADAMW_BETA1;
        self.beta2_power *= ADAMW_BETA2;
        (1.0 - self.beta1_power, 1.0 - self.beta2_power)
    }
}

pub(super) fn adamw_update(
    parameter: &mut f32,
    first_moment: &mut f32,
    second_moment: &mut f32,
    gradient: f32,
    lr: f32,
    bias_correction1: f32,
    bias_correction2: f32,
    weight_decay: f32,
) {
    let gradient = gradient.clamp(-4.0, 4.0);
    *first_moment = ADAMW_BETA1 * *first_moment + (1.0 - ADAMW_BETA1) * gradient;
    *second_moment = ADAMW_BETA2 * *second_moment + (1.0 - ADAMW_BETA2) * gradient * gradient;
    let first_unbiased = *first_moment / bias_correction1.max(1e-12);
    let second_unbiased = *second_moment / bias_correction2.max(1e-12);
    if weight_decay > 0.0 {
        *parameter -= lr * weight_decay * *parameter;
    }
    *parameter -= lr * first_unbiased / (second_unbiased.sqrt() + ADAMW_EPSILON);
}
