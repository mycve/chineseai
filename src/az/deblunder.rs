//! LC0 式结果重标注：校正搜索已识别、但探索选步仍然造成的失误。
//!
//! 方法来源：LC0 `trainingdata/rescorer.cc` 的 `ApplyDeblunder`。
//! https://github.com/LeelaChessZero/lc0/blob/master/src/trainingdata/rescorer.cc
//! 本模块只处理完整对局的 WDL；证明标签作为边界，未知替代策略的步数不参与 MLH。

use super::{AzSearchResult, AzTrainingSample, Move};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeblunderConfig {
    pub enabled: bool,
    pub q_gap_threshold: f32,
    pub q_gap_width: f32,
}

impl Default for DeblunderConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            // LC0 T60 公开训练记录中的启用阈值；默认不改变现有训练。
            q_gap_threshold: 0.1,
            q_gap_width: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DeblunderStep {
    pub best_wdl: [f32; 3],
    pub best_q: f32,
    pub played_q: f32,
    pub best_visits: u32,
    pub played_visits: u32,
    pub best_index: usize,
    pub played_index: usize,
}

impl DeblunderStep {
    /// 与实际部署的最佳着法保持一致，不从未访问候选的 FPU 中取最大 Q。
    pub(crate) fn from_search(search: &AzSearchResult, played: Move) -> Option<Self> {
        let best_move = search.best_move?;
        let (best_index, best) = search
            .candidates
            .iter()
            .enumerate()
            .find(|(_, candidate)| candidate.mv == best_move)?;
        let (played_index, played) = search
            .candidates
            .iter()
            .enumerate()
            .find(|(_, candidate)| candidate.mv == played)?;
        let step = Self {
            best_wdl: search.best_value_wdl,
            best_q: best.q,
            played_q: played.q,
            best_visits: best.visits,
            played_visits: played.visits,
            best_index,
            played_index,
        };
        step.valid().then_some(step)
    }

    fn valid(self) -> bool {
        self.best_visits > 0
            && self.played_visits > 0
            && (-1.0..=1.0).contains(&self.best_q)
            && (-1.0..=1.0).contains(&self.played_q)
            && checked_wdl(self.best_wdl).is_some()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DeblunderStats {
    pub triggers: usize,
    pub corrected_samples: usize,
}

/// 在终局标签和 MLH 标签生成后调用。`proofs` 必须包括所有精确重标注标签。
///
/// 较晚的估计校正不能越过证明或无效标签。这里只屏蔽受影响的 MLH 权重，
/// 不用网络估计杜撰新的终局距离，也不为被截断的对局制造价值标签。
pub(crate) fn apply_deblunder(
    samples: &mut [AzTrainingSample],
    steps: &[Option<DeblunderStep>],
    proofs: &[Option<i8>],
    terminal_known: bool,
    config: DeblunderConfig,
) -> DeblunderStats {
    let mut stats = DeblunderStats::default();
    if !config.enabled || !terminal_known {
        return stats;
    }
    assert_eq!(samples.len(), steps.len());
    assert_eq!(samples.len(), proofs.len());
    assert!((0.0..=2.0).contains(&config.q_gap_threshold));
    assert!((0.0..=2.0).contains(&config.q_gap_width));

    // 用固定红方视角存储回传目标，兼容非连续采样、重复镜像样本等情况。
    let mut carried: Option<[f32; 3]> = None;
    let mut correcting = false;
    for ((sample, step), proof) in samples.iter_mut().zip(steps).zip(proofs).rev() {
        let original = checked_wdl(sample.value_wdl);
        if proof.is_some()
            || original.is_none()
            || !sample.value_weight.is_finite()
            || sample.value_weight <= 0.0
            || !matches!(sample.side_sign, -1.0 | 1.0)
        {
            carried = None;
            correcting = false;
            continue;
        }
        let target = carried.get_or_insert_with(|| orient(original.unwrap(), sample.side_sign));
        if let Some(step) = step.filter(|step| step.valid())
            && step.best_index != step.played_index
        {
            let gap = step.best_q - step.played_q;
            let mix = if config.q_gap_width > 0.0 {
                ((gap - config.q_gap_threshold) / config.q_gap_width + 0.5).clamp(0.0, 1.0)
            } else {
                f32::from(gap > config.q_gap_threshold)
            };
            if mix > 0.0 {
                let best = orient(checked_wdl(step.best_wdl).unwrap(), sample.side_sign);
                for (value, best_value) in target.iter_mut().zip(best) {
                    *value = (1.0 - mix) * *value + mix * best_value;
                }
                correcting = true;
                stats.triggers += 1;
            }
        }
        if correcting {
            let target = orient(*target, sample.side_sign);
            if target != sample.value_wdl || sample.value != target[0] - target[2] {
                stats.corrected_samples += 1;
            }
            sample.value_wdl = target;
            sample.value = target[0] - target[2];
            sample.moves_left_weight = 0.0;
        }
    }
    stats
}

fn checked_wdl(wdl: [f32; 3]) -> Option<[f32; 3]> {
    let sum: f32 = wdl.iter().sum();
    (wdl.iter().all(|value| (0.0..=1.0).contains(value)) && (sum - 1.0).abs() < 0.001)
        .then(|| wdl.map(|value| value / sum))
}

fn orient(wdl: [f32; 3], side_sign: f32) -> [f32; 3] {
    if side_sign < 0.0 {
        [wdl[2], wdl[1], wdl[0]]
    } else {
        wdl
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::az::{AzCandidate, AzSampleMeta, HISTORY_CONTEXT_SIZE, RULE_CONTEXT_SIZE};

    fn enabled() -> DeblunderConfig {
        DeblunderConfig {
            enabled: true,
            ..Default::default()
        }
    }

    fn sample(side_sign: f32) -> AzTrainingSample {
        AzTrainingSample {
            features: Vec::new(),
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            history_features: [0.0; HISTORY_CONTEXT_SIZE],
            move_indices: vec![0, 1],
            repetition_flags: vec![0, 0],
            policy: vec![0.8, 0.2],
            value_wdl: [0.0, 1.0, 0.0],
            root_search_wdl: [0.0, 1.0, 0.0],
            value: 0.0,
            side_sign,
            policy_weight: 1.0,
            value_weight: 1.0,
            moves_left: 10.0,
            moves_left_weight: 1.0,
            search_simulations: 32,
            meta: AzSampleMeta::default(),
        }
    }

    fn step() -> DeblunderStep {
        DeblunderStep {
            best_wdl: [0.8, 0.1, 0.1],
            best_q: 0.7,
            played_q: 0.1,
            best_visits: 20,
            played_visits: 12,
            best_index: 0,
            played_index: 1,
        }
    }

    fn assert_wdl(actual: [f32; 3], expected: [f32; 3]) {
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn propagates_before_blunder_in_each_samples_perspective() {
        let mut samples = [sample(1.0), sample(-1.0), sample(1.0)];
        let stats = apply_deblunder(
            &mut samples,
            &[None, Some(step()), None],
            &[None; 3],
            true,
            enabled(),
        );
        assert_eq!(
            stats,
            DeblunderStats {
                triggers: 1,
                corrected_samples: 2
            }
        );
        assert_wdl(samples[0].value_wdl, [0.1, 0.1, 0.8]);
        assert_wdl(samples[1].value_wdl, [0.8, 0.1, 0.1]);
        assert_wdl(samples[2].value_wdl, [0.0, 1.0, 0.0]);
        assert!((samples[0].value + 0.7).abs() < 1e-6);
        assert_eq!(samples[0].policy, vec![0.8, 0.2]);
        assert_eq!(samples[0].moves_left_weight, 0.0);
        assert_eq!(samples[1].moves_left_weight, 0.0);
        assert_eq!(samples[2].moves_left_weight, 1.0);
    }

    #[test]
    fn soft_threshold_blends_full_wdl_including_draw_probability() {
        let mut samples = [sample(1.0)];
        let mut config = enabled();
        config.q_gap_threshold = 0.5;
        config.q_gap_width = 0.2;
        let mut step = step();
        step.played_q = 0.2;
        apply_deblunder(&mut samples, &[Some(step)], &[None], true, config);
        assert_wdl(samples[0].value_wdl, [0.4, 0.55, 0.05]);
    }

    #[test]
    fn proof_is_preserved_and_blocks_later_heuristic_correction() {
        let mut samples = [sample(1.0), sample(-1.0), sample(1.0)];
        samples[1].value_wdl = [0.0, 0.0, 1.0];
        samples[1].value = -1.0;
        let stats = apply_deblunder(
            &mut samples,
            &[None, Some(step()), Some(step())],
            &[None, Some(-1), None],
            true,
            enabled(),
        );
        assert_eq!(stats.triggers, 1);
        assert_eq!(stats.corrected_samples, 1);
        assert_wdl(samples[0].value_wdl, [0.0, 1.0, 0.0]);
        assert_wdl(samples[1].value_wdl, [0.0, 0.0, 1.0]);
        assert_eq!(samples[1].value, -1.0);
        assert_eq!(samples[1].moves_left_weight, 1.0);
        assert_wdl(samples[2].value_wdl, [0.8, 0.1, 0.1]);
    }

    #[test]
    fn tablebase_anchor_allows_correction_only_inside_known_truncated_prefix() {
        let mut samples = [sample(1.0), sample(-1.0), sample(1.0), sample(-1.0)];
        for (ply, sample) in samples.iter_mut().enumerate() {
            sample.meta.ply = ply as u16;
            sample.value_weight = 0.0;
            sample.moves_left_weight = 0.0;
        }
        let mut proofs = [None; 4];
        let labels = [
            None,
            None,
            Some(crate::az::TablebaseLabel {
                value: 1,
                moves_left: Some(8),
            }),
            None,
        ];
        crate::az::rescore::apply_tablebase_targets(&mut samples, &mut proofs, &labels);
        let stats = apply_deblunder(
            &mut samples[..3],
            &[None, Some(step()), Some(step())],
            &proofs[..3],
            true,
            enabled(),
        );
        assert_eq!(stats.triggers, 1);
        assert_eq!(stats.corrected_samples, 2);
        assert_wdl(samples[0].value_wdl, [0.1, 0.1, 0.8]);
        assert_wdl(samples[1].value_wdl, [0.8, 0.1, 0.1]);
        assert_eq!(samples[0].moves_left_weight, 0.0);
        assert_eq!(samples[1].moves_left_weight, 0.0);
        assert_wdl(samples[2].value_wdl, [1.0, 0.0, 0.0]);
        assert_eq!(samples[2].moves_left, 8.0);
        assert_eq!(samples[2].moves_left_weight, 1.0);
        assert_eq!(samples[3].value_weight, 0.0);
        assert_eq!(samples[3].moves_left_weight, 0.0);
    }

    #[test]
    fn successive_opposite_side_blunders_mix_with_the_carried_target() {
        let mut samples = [sample(1.0), sample(-1.0), sample(1.0), sample(-1.0)];
        let mut earlier = step();
        earlier.played_q = 0.6;
        let config = DeblunderConfig {
            q_gap_width: 0.2,
            ..enabled()
        };
        let stats = apply_deblunder(
            &mut samples,
            &[None, Some(earlier), Some(step()), None],
            &[None; 4],
            true,
            config,
        );
        assert_eq!(stats.triggers, 2);
        assert_eq!(stats.corrected_samples, 3);
        assert_wdl(samples[0].value_wdl, [0.45, 0.1, 0.45]);
        assert_wdl(samples[1].value_wdl, [0.45, 0.1, 0.45]);
        assert_wdl(samples[2].value_wdl, [0.8, 0.1, 0.1]);
        assert_wdl(samples[3].value_wdl, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn disabled_and_truncated_games_keep_existing_targets() {
        for (config, terminal_known) in [(DeblunderConfig::default(), true), (enabled(), false)] {
            let mut samples = [sample(1.0)];
            if !terminal_known {
                samples[0].value_weight = 0.0;
            }
            let stats = apply_deblunder(
                &mut samples,
                &[Some(step())],
                &[None],
                terminal_known,
                config,
            );
            assert_eq!(stats, DeblunderStats::default());
            assert_wdl(samples[0].value_wdl, [0.0, 1.0, 0.0]);
            assert_eq!(samples[0].value_weight, f32::from(terminal_known));
            assert_eq!(samples[0].moves_left_weight, 1.0);
        }
    }

    #[test]
    fn missing_unvisited_and_invalid_search_data_do_not_trigger() {
        let mut variants = vec![None];
        let mut invalid = step();
        invalid.best_visits = 0;
        variants.push(Some(invalid));
        invalid = step();
        invalid.played_visits = 0;
        variants.push(Some(invalid));
        invalid = step();
        invalid.best_q = f32::NAN;
        variants.push(Some(invalid));
        invalid = step();
        invalid.best_wdl = [f32::NAN, 1.0, 0.0];
        variants.push(Some(invalid));
        invalid = step();
        invalid.best_wdl = [0.0; 3];
        variants.push(Some(invalid));
        invalid = step();
        invalid.played_index = invalid.best_index;
        variants.push(Some(invalid));
        for step in variants {
            let mut samples = [sample(1.0)];
            let stats = apply_deblunder(&mut samples, &[step], &[None], true, enabled());
            assert_eq!(stats, DeblunderStats::default());
            assert_wdl(samples[0].value_wdl, [0.0, 1.0, 0.0]);
            assert_eq!(samples[0].moves_left_weight, 1.0);
        }
    }

    #[test]
    fn records_deployed_best_move_and_its_wdl_instead_of_max_q() {
        let moves = [Move::new(0, 1), Move::new(0, 2), Move::new(0, 3)];
        let search = AzSearchResult {
            best_move: Some(moves[1]),
            value_q: 0.0,
            value_cp: 0,
            value_wdl: [0.0, 1.0, 0.0],
            network_value_wdl: [0.0, 1.0, 0.0],
            best_value_wdl: [0.8, 0.1, 0.1],
            proven_moves_left: None,
            simulations: 32,
            search_depth_avg: 0.0,
            search_depth_max: 0,
            search_depth_limit: 0,
            search_depth_cutoffs: 0,
            mate_search: None,
            candidates: moves
                .into_iter()
                .zip([(0, 1.0), (20, 0.7), (12, 0.1)])
                .map(|(mv, (visits, q))| AzCandidate {
                    mv,
                    visits,
                    q,
                    raw_prior: 0.0,
                    prior: 0.0,
                    policy: 0.0,
                    solved: None,
                    moves_left: None,
                })
                .collect(),
        };
        let recorded = DeblunderStep::from_search(&search, moves[2]).unwrap();
        assert_eq!(recorded.best_index, 1);
        assert_eq!(recorded.best_q, 0.7);
        assert_eq!(recorded.best_wdl, [0.8, 0.1, 0.1]);
        assert_eq!(recorded.played_index, 2);
        assert!(DeblunderStep::from_search(&search, moves[0]).is_none());
    }
}
