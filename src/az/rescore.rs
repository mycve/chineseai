//! 已验证残局标签向轨迹前缀传播，对照 LC0 rescorer；不修改搜索的 solved 状态。
use super::{AzTrainingSample, TablebaseLabel};

pub(super) fn apply_tablebase_targets(
    samples: &mut [AzTrainingSample],
    proofs: &mut [Option<i8>],
    labels: &[Option<TablebaseLabel>],
) -> usize {
    assert_eq!(samples.len(), proofs.len());
    assert_eq!(samples.len(), labels.len());
    let mut active: Option<(TablebaseLabel, f32, u16)> = None;
    let mut labeled = 0;
    for index in (0..samples.len()).rev() {
        // 当前搜索的规则证明优先，也阻止后续轨迹的标签越过该节点。
        if proofs[index].is_some() {
            active = None;
            continue;
        }
        let sample = &mut samples[index];
        if let Some(label) = labels[index] {
            active = Some((label, sample.side_sign, sample.meta.ply));
            // 仅锚点本身具有精确标签；其前缀仍是沿本次轨迹传播的训练目标。
            proofs[index] = Some(label.value);
        }
        let Some((label, side, anchor_ply)) = active else {
            continue;
        };
        let value = if side == sample.side_sign {
            label.value
        } else {
            -label.value
        };
        sample.value = f32::from(value);
        sample.value_wdl = TablebaseLabel {
            value,
            moves_left: None,
        }
        .wdl();
        sample.value_weight = 1.0;
        sample.moves_left = 0.0;
        sample.moves_left_weight = 0.0;
        if let Some(distance) = label.moves_left {
            sample.moves_left =
                f32::from(distance) + f32::from(anchor_ply.saturating_sub(sample.meta.ply));
            sample.moves_left_weight = 1.0;
        }
        labeled += 1;
    }
    labeled
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::az::{AzSampleMeta, HISTORY_CONTEXT_SIZE, RULE_CONTEXT_SIZE};

    fn sample(ply: u16) -> AzTrainingSample {
        AzTrainingSample {
            features: vec![],
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            history_features: [0.0; HISTORY_CONTEXT_SIZE],
            move_indices: vec![],
            repetition_flags: vec![],
            policy: vec![],
            value_wdl: [0.0, 1.0, 0.0],
            root_search_wdl: [0.0, 1.0, 0.0],
            value: 0.0,
            side_sign: if ply % 2 == 0 { 1.0 } else { -1.0 },
            policy_weight: 1.0,
            value_weight: 0.0,
            moves_left: 0.0,
            moves_left_weight: 0.0,
            search_simulations: 10,
            meta: AzSampleMeta {
                ply,
                ..Default::default()
            },
        }
    }

    #[test]
    fn tablebase_anchor_rescores_only_known_prefix_of_truncation() {
        let mut samples = (0..4).map(sample).collect::<Vec<_>>();
        let mut proofs = vec![None; 4];
        let labels = [
            None,
            None,
            Some(TablebaseLabel {
                value: 1,
                moves_left: Some(7),
            }),
            None,
        ];
        assert_eq!(
            apply_tablebase_targets(&mut samples, &mut proofs, &labels),
            3
        );
        assert_eq!(
            samples.iter().map(|s| s.value).collect::<Vec<_>>(),
            [1.0, -1.0, 1.0, 0.0]
        );
        assert_eq!(
            samples.iter().map(|s| s.moves_left).collect::<Vec<_>>(),
            [9.0, 8.0, 7.0, 0.0]
        );
        assert_eq!(samples[3].value_weight, 0.0);
        assert_eq!(proofs, [None, None, Some(1), None]);
    }

    #[test]
    fn search_proof_wins_over_conflicting_cache_and_blocks_propagation() {
        let mut samples = (0..3).map(sample).collect::<Vec<_>>();
        samples[1].value = -1.0;
        samples[1].value_wdl = [0.0, 0.0, 1.0];
        samples[1].value_weight = 1.0;
        let mut proofs = [None, Some(-1), None];
        let labels = [
            None,
            Some(TablebaseLabel {
                value: 1,
                moves_left: None,
            }),
            Some(TablebaseLabel {
                value: -1,
                moves_left: None,
            }),
        ];
        assert_eq!(
            apply_tablebase_targets(&mut samples, &mut proofs, &labels),
            1
        );
        assert_eq!(samples[1].value, -1.0);
        assert_eq!(samples[0].value_weight, 0.0);
        assert_eq!(samples[2].moves_left_weight, 0.0);
    }

    #[test]
    fn earlier_anchor_replaces_later_trajectory_result() {
        let mut samples = (0..3).map(sample).collect::<Vec<_>>();
        let mut proofs = [None; 3];
        let labels = [
            None,
            Some(TablebaseLabel {
                value: 0,
                moves_left: None,
            }),
            Some(TablebaseLabel {
                value: 1,
                moves_left: Some(5),
            }),
        ];
        apply_tablebase_targets(&mut samples, &mut proofs, &labels);
        assert_eq!(samples[0].value_wdl, [0.0, 1.0, 0.0]);
        assert_eq!(samples[0].moves_left_weight, 0.0);
        assert_eq!(samples[2].value, 1.0);
    }
}
