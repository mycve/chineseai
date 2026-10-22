use crate::cli::args::*;
use chineseai::az::AzNnue;

pub(crate) fn run(cmd: AzPolicyScaleArgs) {
    assert!(
        cmd.scale.is_finite() && cmd.scale >= 0.0,
        "policy component scale must be finite and non-negative"
    );
    let mut model = AzNnue::load(&cmd.input)
        .unwrap_or_else(|err| panic!("failed to load `{}`: {err}", cmd.input));
    let weights: &mut [f32] = match cmd.component {
        PolicyComponent::Capture => &mut model.policy_move_capture,
        PolicyComponent::MovePiece => &mut model.policy_move_piece,
        PolicyComponent::King => &mut model.policy_move_king,
        PolicyComponent::Local => &mut model.policy_local,
        PolicyComponent::Joint => &mut model.policy_joint,
        PolicyComponent::Events => &mut model.policy_events,
        PolicyComponent::Context => &mut model.policy_move_context,
        PolicyComponent::Consequence => &mut model.policy_piece_square,
        PolicyComponent::MoveBias => &mut model.policy_move_bias,
    };
    let (rows, nonzero, before_l2) = {
        let nonzero = weights.iter().filter(|&&weight| weight != 0.0).count();
        let before_l2 = weights
            .iter()
            .map(|&weight| f64::from(weight) * f64::from(weight))
            .sum::<f64>()
            .sqrt();
        for weight in weights.iter_mut() {
            *weight *= cmd.scale;
        }
        (weights.len(), nonzero, before_l2)
    };
    let after_l2 = before_l2 * f64::from(cmd.scale);
    model
        .save(&cmd.output)
        .unwrap_or_else(|err| panic!("failed to write `{}`: {err}", cmd.output));
    println!("input    : {}", cmd.input);
    println!("output   : {}", cmd.output);
    println!("component: {:?}", cmd.component);
    println!("scale    : {}", cmd.scale);
    println!("weights  : rows={} nonzero={}", rows, nonzero);
    println!("l2       : {:.6} -> {:.6}", before_l2, after_l2);
}
