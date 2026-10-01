use crate::cli::args::*;
use chineseai::az::{AzNnue, POLICY_SPARSE_MAIN_SIZE, POLICY_TACTICAL_EXACT_SIZE};

pub(crate) fn run(cmd: AzPolicyScaleArgs) {
    assert!(
        cmd.scale.is_finite() && cmd.scale >= 0.0,
        "policy exact scale must be finite and non-negative"
    );
    let mut model = AzNnue::load(&cmd.input)
        .unwrap_or_else(|err| panic!("failed to load `{}`: {err}", cmd.input));
    let sparse_len = model.policy_sparse_table.len();
    let weights: &mut [f32] = match cmd.component {
        PolicyComponent::Exact => &mut model.policy_sparse_table[..POLICY_SPARSE_MAIN_SIZE],
        PolicyComponent::Capture => {
            &mut model.policy_sparse_table[POLICY_SPARSE_MAIN_SIZE..sparse_len - 1]
        }
        PolicyComponent::Factor => &mut model.policy_sparse_factor,
        PolicyComponent::Tactical => &mut model.policy_tactical,
        PolicyComponent::TacticalExact => {
            &mut model.policy_tactical[..POLICY_TACTICAL_EXACT_SIZE]
        }
        PolicyComponent::TacticalFactor => {
            &mut model.policy_tactical[POLICY_TACTICAL_EXACT_SIZE..]
        }
        PolicyComponent::Accumulator => &mut model.policy_accumulator_move,
        PolicyComponent::Context => &mut model.policy_move_context,
        PolicyComponent::Consequence => &mut model.policy_consequence_output,
        PolicyComponent::MoveBias => &mut model.policy_move_bias,
        PolicyComponent::ThreatContext => &mut model.policy_threat_context,
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
