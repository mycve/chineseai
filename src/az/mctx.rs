use super::SplitMix64;

#[derive(Clone, Copy, Debug)]
pub struct AzGumbelConfig {
    pub max_num_considered_actions: usize,
    pub gumbel_scale: f32,
    pub value_scale: f32,
    pub maxvisit_init: f32,
    pub rescale_values: bool,
    pub use_mixed_value: bool,
}

impl Default for AzGumbelConfig {
    fn default() -> Self {
        Self {
            max_num_considered_actions: 16,
            gumbel_scale: 1.0,
            value_scale: 0.1,
            maxvisit_init: 50.0,
            rescale_values: true,
            use_mixed_value: true,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ActionStats {
    pub logit: f32,
    pub visit_count: u32,
    pub qvalue: f32,
}

pub(super) fn sample_gumbels(num_actions: usize, scale: f32, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed ^ 0x9A2D_D6ED_2B7F_5A13u64 ^ num_actions as u64);
    (0..num_actions)
        .map(|_| {
            let u = rng.unit_f32().clamp(1e-12, 1.0 - 1e-7);
            (-(-(u as f64).ln()).ln()) as f32 * scale
        })
        .collect()
}

pub(super) fn gumbel_muzero_root_action_selection(
    actions: &[ActionStats],
    root_gumbel: &[f32],
    considered_visit_sequence: &[u32],
    config: AzGumbelConfig,
    raw_value: f32,
) -> usize {
    let simulation_index = actions
        .iter()
        .map(|action| action.visit_count as usize)
        .sum::<usize>();
    let considered_visit = considered_visit_sequence
        .get(simulation_index)
        .copied()
        .unwrap_or_else(|| {
            actions
                .iter()
                .map(|action| action.visit_count)
                .max()
                .unwrap_or(0)
        });
    let completed_qvalues = qtransform_completed_by_mix_value(actions, raw_value, config);
    let logits = normalized_logits(actions);

    let mut best_index = None;
    let mut best_score = f32::NEG_INFINITY;
    for (index, action) in actions.iter().enumerate() {
        if action.visit_count != considered_visit {
            continue;
        }
        let score = score_considered(
            root_gumbel.get(index).copied().unwrap_or(0.0),
            logits[index],
            completed_qvalues[index],
        );
        if best_index.is_none() || score > best_score {
            best_index = Some(index);
            best_score = score;
        }
    }
    best_index.unwrap_or(0)
}

pub(super) fn gumbel_muzero_interior_action_selection(
    actions: &[ActionStats],
    config: AzGumbelConfig,
    raw_value: f32,
) -> usize {
    let completed_qvalues = qtransform_completed_by_mix_value(actions, raw_value, config);
    let probs = softmax_logits_plus_values(actions, &completed_qvalues);
    let total_visits = actions
        .iter()
        .map(|action| action.visit_count as f32)
        .sum::<f32>();

    let mut best_index = None;
    let mut best_score = f32::NEG_INFINITY;
    for (index, action) in actions.iter().enumerate() {
        let score = probs[index] - action.visit_count as f32 / (1.0 + total_visits);
        if best_index.is_none() || score > best_score {
            best_index = Some(index);
            best_score = score;
        }
    }
    best_index.unwrap_or(0)
}

pub(super) fn gumbel_muzero_root_best_action(
    actions: &[ActionStats],
    root_gumbel: &[f32],
    config: AzGumbelConfig,
    raw_value: f32,
) -> Option<usize> {
    let considered_visit = actions
        .iter()
        .map(|action| action.visit_count)
        .max()
        .unwrap_or(0);
    let completed_qvalues = qtransform_completed_by_mix_value(actions, raw_value, config);
    let logits = normalized_logits(actions);

    let mut best_index = None;
    let mut best_score = f32::NEG_INFINITY;
    for (index, action) in actions.iter().enumerate() {
        if action.visit_count != considered_visit {
            continue;
        }
        let score = score_considered(
            root_gumbel.get(index).copied().unwrap_or(0.0),
            logits[index],
            completed_qvalues[index],
        );
        if best_index.is_none() || score > best_score {
            best_index = Some(index);
            best_score = score;
        }
    }
    best_index
}

pub(super) fn gumbel_muzero_root_policy(
    actions: &[ActionStats],
    config: AzGumbelConfig,
    raw_value: f32,
) -> Vec<f32> {
    let completed_qvalues = qtransform_completed_by_mix_value(actions, raw_value, config);
    softmax_logits_plus_values(actions, &completed_qvalues)
}

pub(super) fn qtransform_completed_by_mix_value(
    actions: &[ActionStats],
    raw_value: f32,
    config: AzGumbelConfig,
) -> Vec<f32> {
    if actions.is_empty() {
        return Vec::new();
    }

    let mut prior_probs = softmax_logits(actions);
    for prior in &mut prior_probs {
        *prior = prior.max(f32::MIN_POSITIVE);
    }
    let sum_visits = actions
        .iter()
        .map(|action| action.visit_count as f32)
        .sum::<f32>();
    let sum_visited_prior = actions
        .iter()
        .zip(&prior_probs)
        .filter(|(action, _)| action.visit_count > 0)
        .map(|(_, prior)| *prior)
        .sum::<f32>()
        .max(f32::MIN_POSITIVE);
    let weighted_q = actions
        .iter()
        .zip(&prior_probs)
        .filter(|(action, _)| action.visit_count > 0)
        .map(|(action, prior)| *prior * action.qvalue / sum_visited_prior)
        .sum::<f32>();
    let mixed_value = if config.use_mixed_value {
        (raw_value + sum_visits * weighted_q) / (sum_visits + 1.0)
    } else {
        raw_value
    };

    let mut completed = actions
        .iter()
        .map(|action| {
            if action.visit_count > 0 {
                action.qvalue
            } else {
                mixed_value
            }
        })
        .collect::<Vec<_>>();
    if config.rescale_values {
        let min_value = completed.iter().copied().fold(f32::INFINITY, f32::min);
        let max_value = completed.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denom = (max_value - min_value).max(1e-8);
        for value in &mut completed {
            *value = (*value - min_value) / denom;
        }
    }

    let max_visit = actions
        .iter()
        .map(|action| action.visit_count)
        .max()
        .unwrap_or(0) as f32;
    let scale = (config.maxvisit_init + max_visit) * config.value_scale;
    for value in &mut completed {
        *value *= scale;
    }
    completed
}

fn score_considered(gumbel: f32, logit: f32, completed_qvalue: f32) -> f32 {
    (gumbel + logit + completed_qvalue).max(-1.0e9)
}

fn normalized_logits(actions: &[ActionStats]) -> Vec<f32> {
    let max_logit = actions
        .iter()
        .map(|action| action.logit)
        .fold(f32::NEG_INFINITY, f32::max);
    actions
        .iter()
        .map(|action| action.logit - max_logit)
        .collect()
}

fn softmax_logits(actions: &[ActionStats]) -> Vec<f32> {
    if actions.is_empty() {
        return Vec::new();
    }
    let max_logit = actions
        .iter()
        .map(|action| action.logit)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs = Vec::with_capacity(actions.len());
    let mut sum = 0.0f32;
    for action in actions {
        let value = (action.logit - max_logit).exp();
        probs.push(value);
        sum += value;
    }
    let inv_sum = sum.max(1e-12).recip();
    for value in &mut probs {
        *value *= inv_sum;
    }
    probs
}

fn softmax_logits_plus_values(actions: &[ActionStats], values: &[f32]) -> Vec<f32> {
    if actions.is_empty() {
        return Vec::new();
    }
    let max_score = actions
        .iter()
        .zip(values)
        .map(|(action, value)| action.logit + *value)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut out = Vec::with_capacity(actions.len());
    let mut sum = 0.0f32;
    for (action, value) in actions.iter().zip(values) {
        let probability = (action.logit + *value - max_score).exp();
        out.push(probability);
        sum += probability;
    }
    let inv_sum = sum.max(1e-12).recip();
    for value in &mut out {
        *value *= inv_sum;
    }
    out
}

pub(super) fn get_sequence_of_considered_visits(
    max_num_considered_actions: usize,
    num_simulations: usize,
) -> Vec<u32> {
    if max_num_considered_actions <= 1 {
        return (0..num_simulations).map(|visit| visit as u32).collect();
    }
    let log2max = (max_num_considered_actions as f32).log2().ceil().max(1.0) as usize;
    let mut sequence = Vec::with_capacity(num_simulations);
    let mut visits = vec![0u32; max_num_considered_actions];
    let mut num_considered = max_num_considered_actions;
    while sequence.len() < num_simulations {
        let num_extra_visits = (num_simulations / (log2max * num_considered)).max(1);
        for _ in 0..num_extra_visits {
            for visit in visits.iter_mut().take(num_considered) {
                sequence.push(*visit);
                *visit += 1;
                if sequence.len() == num_simulations {
                    break;
                }
            }
            if sequence.len() == num_simulations {
                break;
            }
        }
        num_considered = (num_considered / 2).max(2);
    }
    sequence
}
