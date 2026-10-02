use candle_core::{DType, Device, Result as CandleResult, Tensor, Var};

use super::arch::CHECK_CONTEXT_SIZE;
use super::{
    AzNnue, AzNnueArch, DENSE_MOVE_SPACE, POLICY_ACCUMULATOR_RANK, POLICY_CONSEQUENCE_SIZE,
    POLICY_MOVE_CONTEXT_SIZE, POLICY_SPARSE_FACTOR_SIZE, POLICY_SPARSE_TABLE_SIZE,
    POLICY_TACTICAL_SIZE, POLICY_TACTICAL_TERMS, POLICY_THREAT_CONTEXT_SIZE, RULE_CONTEXT_SIZE,
    STRUCTURAL_FILE_SIZE, STRUCTURAL_KING_PIECE_SIZE, STRUCTURAL_PIECE_SIZE, STRUCTURAL_RANK_SIZE,
    VALUE_HEAD_SIZE, VALUE_KING_PIECE_VOCAB, VALUE_THREAT_RANK, VALUE_THREAT_VOCAB, WDL_HEAD_SIZE,
    dataloader::PackedBatch,
    fused_feature_pool::{PADDING_ITEM, feature_pool, sparse_pool},
    fused_policy::fused_policy,
    fused_sparse_policy::{sparse_policy, tactical_policy},
};
use crate::az::nnue::AZ_NNUE_INPUT_SIZE;

const RMS_NORM_EPS: f64 = 1.0e-6;

#[derive(Debug)]
pub(super) struct AzCandleModel {
    arch: AzNnueArch,
    input_hidden: Var,
    input_piece_hidden: Var,
    input_rank_hidden: Var,
    input_file_hidden: Var,
    input_king_piece_hidden: Var,
    rule_context_hidden: Var,
    check_context_hidden: Var,
    hidden_bias: Var,
    value_head_hidden: Var,
    value_head_bias: Var,
    value_king_piece_hidden: Var,
    value_head_output: Var,
    value_threat_embedding: Var,
    value_threat_output: Var,
    policy_threat_context: Var,
    policy_move_bias: Var,
    policy_consequence_output: Var,
    policy_context_hidden: Var,
    policy_move_context: Var,
    policy_accumulator_hidden: Var,
    policy_accumulator_move: Var,
    policy_sparse_table: Var,
    policy_sparse_factor: Var,
    policy_tactical: Var,
    policy_repetition_hidden: Var,
    policy_repetition_bias: Var,
}

impl AzCandleModel {
    pub(super) fn forward(&self, batch: &BatchTensors) -> CandleResult<ForwardOutput> {
        let hidden_size = self.arch.hidden_size;
        let policy_consequence_size = POLICY_CONSEQUENCE_SIZE.min(hidden_size);
        let feature_tables = Tensor::cat(
            &[
                self.input_hidden.as_tensor(),
                self.input_piece_hidden.as_tensor(),
                self.input_rank_hidden.as_tensor(),
                self.input_file_hidden.as_tensor(),
                self.input_king_piece_hidden.as_tensor(),
            ],
            0,
        )?;
        let board_pre = feature_pool(&feature_tables, &batch.feature_items)?
            .broadcast_add(&self.hidden_bias)?;
        let accumulator_context = board_pre.matmul(&self.policy_accumulator_hidden.t()?)?;
        let rule_pre = batch.rule_context.matmul(&self.rule_context_hidden)?;
        let check_pre = batch.check_context.matmul(&self.check_context_hidden)?;
        let sparse_pre = ((board_pre + rule_pre)? + check_pre)?;
        let sparse_hidden = sparse_pre.relu()?;
        let rms = sparse_hidden
            .sqr()?
            .mean_keepdim(1)?
            .affine(1.0, RMS_NORM_EPS)?
            .sqrt()?;
        let hidden = sparse_hidden.broadcast_div(&rms)?;
        let value_king_piece = sparse_pool(
            self.value_king_piece_hidden.as_tensor(),
            &batch.value_king_piece_indices,
        )?
        .broadcast_mul(&batch.value_king_piece_scales)?;
        let value_head = hidden
            .matmul(&self.value_head_hidden.t()?)?
            .broadcast_add(&self.value_head_bias)?
            .add(&value_king_piece)?
            .relu()?;
        let value_logits = value_head.matmul(&self.value_head_output.t()?)?;
        let threat_accumulator = sparse_pool(
            self.value_threat_embedding.as_tensor(),
            &batch.value_threat_indices,
        )?
        .broadcast_mul(&batch.value_threat_scales)?;
        let threat_activation = threat_accumulator;
        let threat_pair = Tensor::cat(&[&threat_activation, &threat_activation.sqr()?], 1)?;
        let value_logits = (value_logits + threat_pair.matmul(&self.value_threat_output.t()?)?)?;
        let piece_square_policy = self
            .input_hidden
            .narrow(1, 0, policy_consequence_size)?
            .contiguous()?;
        let piece_square_policy = if policy_consequence_size < POLICY_CONSEQUENCE_SIZE {
            Tensor::cat(
                &[
                    &piece_square_policy,
                    &Tensor::zeros(
                        (
                            AZ_NNUE_INPUT_SIZE,
                            POLICY_CONSEQUENCE_SIZE - policy_consequence_size,
                        ),
                        DType::F32,
                        self.input_hidden.device(),
                    )?,
                ],
                1,
            )?
        } else {
            piece_square_policy
        };
        let piece_square_policy = piece_square_policy.flatten_all()?;
        let policy_context = (hidden.matmul(&self.policy_context_hidden.t()?)?
            + threat_pair.matmul(&self.policy_threat_context.t()?)?)?;
        let policy_context = Tensor::cat(&[&policy_context, &accumulator_context], 1)?;
        let accumulator_feature = self
            .input_hidden
            .matmul(&self.policy_accumulator_hidden.t()?)?
            .flatten_all()?;
        let policy_tables = Tensor::cat(
            &[
                &piece_square_policy,
                &self.policy_consequence_output.flatten_all()?,
                &self.policy_move_bias.flatten_all()?,
                &self.policy_move_context.flatten_all()?,
                &accumulator_feature,
                &self.policy_accumulator_move.flatten_all()?,
            ],
            0,
        )?;
        let policy_logits = fused_policy(&policy_tables, &policy_context, &batch.policy_items)?;
        let sparse_tables = Tensor::cat(
            &[
                self.policy_sparse_table.as_tensor(),
                self.policy_sparse_factor.as_tensor(),
            ],
            0,
        )?;
        let sparse_logits = sparse_policy(&sparse_tables, &batch.policy_sparse_indices)?;
        let tactical_table = Tensor::cat(
            &[
                self.policy_tactical.as_tensor(),
                &Tensor::zeros(1, DType::F32, self.input_hidden.device())?,
            ],
            0,
        )?;
        let tactical_logits = tactical_policy(&tactical_table, &batch.policy_tactical_indices)?;
        let repetition_logit = hidden
            .matmul(
                &self
                    .policy_repetition_hidden
                    .as_tensor()
                    .reshape((hidden_size, 1))?,
            )?
            .broadcast_add(&self.policy_repetition_bias)?;
        let repetition_logits = batch.policy_repetition.broadcast_mul(&repetition_logit)?;
        let policy_logits =
            ((policy_logits + sparse_logits + tactical_logits)? + repetition_logits)?;

        Ok(ForwardOutput {
            value_logits,
            policy_logits,
        })
    }
}

pub(super) struct ForwardOutput {
    pub(super) value_logits: Tensor,
    pub(super) policy_logits: Tensor,
}

pub(super) struct BatchTensors {
    pub(super) batch_size: usize,
    pub(super) feature_items: Tensor,
    pub(super) value_threat_indices: Tensor,
    pub(super) value_threat_scales: Tensor,
    pub(super) value_king_piece_indices: Tensor,
    pub(super) value_king_piece_scales: Tensor,
    pub(super) policy_items: Tensor,
    pub(super) policy_sparse_indices: Tensor,
    pub(super) policy_tactical_indices: Tensor,
    pub(super) policy_targets: Tensor,
    pub(super) policy_mask: Tensor,
    pub(super) policy_repetition: Tensor,
    pub(super) value_wdl: Tensor,
    pub(super) values: Tensor,
    pub(super) rule_context: Tensor,
    pub(super) check_context: Tensor,
    pub(super) policy_weights: Tensor,
    pub(super) value_weights: Tensor,
    pub(super) value_phase_masks: Tensor,
    pub(super) value_source_phase_masks: Tensor,
}

impl BatchTensors {
    pub(super) fn from_packed(packed: PackedBatch, device: &Device) -> CandleResult<Self> {
        let batch_size = packed.batch_size;
        let max_features = packed.max_features;
        let max_policy_moves = packed.max_policy_moves;
        let max_value_threats = packed.max_value_threats;
        let max_value_king_pieces = packed.max_value_king_pieces;
        assert!(
            packed
                .value_threat_indices
                .iter()
                .all(|&index| index == PADDING_ITEM || index < VALUE_THREAT_VOCAB as u32),
            "value threat index exceeds vocabulary"
        );
        assert!(
            packed
                .value_king_piece_indices
                .iter()
                .all(|&index| index == PADDING_ITEM || index < VALUE_KING_PIECE_VOCAB as u32)
        );
        Ok(Self {
            batch_size,
            feature_items: Tensor::from_vec(
                packed.feature_items,
                (batch_size, max_features),
                device,
            )?,
            value_threat_indices: Tensor::from_vec(
                packed.value_threat_indices,
                (batch_size, max_value_threats),
                device,
            )?,
            value_threat_scales: Tensor::from_vec(
                packed.value_threat_scales,
                (batch_size, 1),
                device,
            )?,
            value_king_piece_indices: Tensor::from_vec(
                packed.value_king_piece_indices,
                (batch_size, max_value_king_pieces),
                device,
            )?,
            value_king_piece_scales: Tensor::from_vec(
                packed.value_king_piece_scales,
                (batch_size, 1),
                device,
            )?,
            policy_items: Tensor::from_vec(
                packed.policy_items,
                (batch_size, max_policy_moves),
                device,
            )?,
            policy_sparse_indices: Tensor::from_vec(
                packed.policy_sparse_indices,
                (batch_size, max_policy_moves, 7),
                device,
            )?,
            policy_tactical_indices: Tensor::from_vec(
                packed.policy_tactical_indices,
                (batch_size, max_policy_moves, POLICY_TACTICAL_TERMS),
                device,
            )?,
            policy_targets: Tensor::from_vec(
                packed.policy_targets,
                (batch_size, max_policy_moves),
                device,
            )?,
            policy_mask: Tensor::from_vec(
                packed.policy_mask,
                (batch_size, max_policy_moves),
                device,
            )?,
            policy_repetition: Tensor::from_vec(
                packed.policy_repetition,
                (batch_size, max_policy_moves),
                device,
            )?,
            value_wdl: Tensor::from_vec(packed.value_wdl, (batch_size, WDL_HEAD_SIZE), device)?,
            values: Tensor::from_vec(packed.values, batch_size, device)?,
            rule_context: Tensor::from_vec(
                packed.rule_context,
                (batch_size, RULE_CONTEXT_SIZE),
                device,
            )?,
            policy_weights: Tensor::from_vec(packed.policy_weights, batch_size, device)?,
            check_context: Tensor::from_vec(
                packed.check_context,
                (batch_size, CHECK_CONTEXT_SIZE),
                device,
            )?,
            value_weights: Tensor::from_vec(packed.value_weights, batch_size, device)?,
            value_phase_masks: Tensor::from_vec(packed.value_phase_masks, (batch_size, 3), device)?,
            value_source_phase_masks: Tensor::from_vec(
                packed.value_source_phase_masks,
                (batch_size, 9),
                device,
            )?,
        })
    }
}

impl AzCandleModel {
    pub(super) fn from_model(model: &AzNnue, device: &Device) -> CandleResult<Self> {
        let arch = model.arch;
        let hidden = arch.hidden_size;
        Ok(Self {
            arch,
            input_hidden: var_from_slice(
                &model.input_hidden,
                (AZ_NNUE_INPUT_SIZE, hidden),
                device,
            )?,
            input_piece_hidden: var_from_slice(
                &model.input_piece_hidden,
                (STRUCTURAL_PIECE_SIZE, hidden),
                device,
            )?,
            input_rank_hidden: var_from_slice(
                &model.input_rank_hidden,
                (STRUCTURAL_RANK_SIZE, hidden),
                device,
            )?,
            input_file_hidden: var_from_slice(
                &model.input_file_hidden,
                (STRUCTURAL_FILE_SIZE, hidden),
                device,
            )?,
            input_king_piece_hidden: var_from_slice(
                &model.input_king_piece_hidden,
                (STRUCTURAL_KING_PIECE_SIZE, hidden),
                device,
            )?,
            rule_context_hidden: var_from_slice(
                &model.rule_context_hidden,
                (RULE_CONTEXT_SIZE, hidden),
                device,
            )?,
            hidden_bias: var_from_slice(&model.hidden_bias, hidden, device)?,
            check_context_hidden: var_from_slice(
                &model.check_context_hidden,
                (CHECK_CONTEXT_SIZE, hidden),
                device,
            )?,
            value_head_hidden: var_from_slice(
                &model.value_head_hidden,
                (VALUE_HEAD_SIZE, hidden),
                device,
            )?,
            value_head_bias: var_from_slice(&model.value_head_bias, VALUE_HEAD_SIZE, device)?,
            value_king_piece_hidden: var_from_slice(
                &model.value_king_piece_hidden,
                (VALUE_KING_PIECE_VOCAB, VALUE_HEAD_SIZE),
                device,
            )?,
            value_head_output: var_from_slice(
                &model.value_head_output,
                (WDL_HEAD_SIZE, VALUE_HEAD_SIZE),
                device,
            )?,
            value_threat_embedding: var_from_slice(
                &model.value_threat_embedding,
                (VALUE_THREAT_VOCAB, VALUE_THREAT_RANK),
                device,
            )?,
            value_threat_output: var_from_slice(
                &model.value_threat_output,
                (WDL_HEAD_SIZE, VALUE_THREAT_RANK * 2),
                device,
            )?,
            policy_threat_context: var_from_slice(
                &model.policy_threat_context,
                (POLICY_THREAT_CONTEXT_SIZE, VALUE_THREAT_RANK * 2),
                device,
            )?,
            policy_move_bias: var_from_slice(&model.policy_move_bias, DENSE_MOVE_SPACE, device)?,
            policy_consequence_output: var_from_slice(
                &model.policy_consequence_output,
                POLICY_CONSEQUENCE_SIZE,
                device,
            )?,
            policy_context_hidden: var_from_slice(
                &model.policy_context_hidden,
                (POLICY_MOVE_CONTEXT_SIZE, hidden),
                device,
            )?,
            policy_move_context: var_from_slice(
                &model.policy_move_context,
                (DENSE_MOVE_SPACE, POLICY_MOVE_CONTEXT_SIZE),
                device,
            )?,
            policy_accumulator_hidden: var_from_slice(
                &model.policy_accumulator_hidden,
                (POLICY_ACCUMULATOR_RANK, hidden),
                device,
            )?,
            policy_accumulator_move: var_from_slice(
                &model.policy_accumulator_move,
                (DENSE_MOVE_SPACE, POLICY_ACCUMULATOR_RANK),
                device,
            )?,
            policy_sparse_table: var_from_slice(
                &model.policy_sparse_table,
                POLICY_SPARSE_TABLE_SIZE,
                device,
            )?,
            policy_sparse_factor: var_from_slice(
                &model.policy_sparse_factor,
                POLICY_SPARSE_FACTOR_SIZE,
                device,
            )?,
            policy_tactical: var_from_slice(&model.policy_tactical, POLICY_TACTICAL_SIZE, device)?,
            policy_repetition_hidden: var_from_slice(
                &model.policy_repetition_hidden,
                hidden,
                device,
            )?,
            policy_repetition_bias: var_from_slice(&model.policy_repetition_bias, 1, device)?,
        })
    }

    pub(super) fn all_vars(&self) -> Vec<Var> {
        let mut vars = Vec::new();
        vars.push(self.input_hidden.clone());
        vars.push(self.input_piece_hidden.clone());
        vars.push(self.input_rank_hidden.clone());
        vars.push(self.input_file_hidden.clone());
        vars.push(self.input_king_piece_hidden.clone());
        vars.push(self.rule_context_hidden.clone());
        vars.push(self.check_context_hidden.clone());
        vars.push(self.hidden_bias.clone());
        vars.push(self.value_head_hidden.clone());
        vars.push(self.value_head_bias.clone());
        vars.push(self.value_king_piece_hidden.clone());
        vars.push(self.value_head_output.clone());
        vars.push(self.value_threat_embedding.clone());
        vars.push(self.value_threat_output.clone());
        vars.push(self.policy_threat_context.clone());
        vars.push(self.policy_move_bias.clone());
        vars.push(self.policy_consequence_output.clone());
        vars.push(self.policy_context_hidden.clone());
        vars.push(self.policy_move_context.clone());
        vars.push(self.policy_accumulator_hidden.clone());
        vars.push(self.policy_accumulator_move.clone());
        vars.push(self.policy_sparse_table.clone());
        vars.push(self.policy_sparse_factor.clone());
        vars.push(self.policy_tactical.clone());
        vars.push(self.policy_repetition_hidden.clone());
        vars.push(self.policy_repetition_bias.clone());
        vars
    }

    pub(super) fn copy_to_model(&self, model: &mut AzNnue) -> CandleResult<()> {
        copy_var(&self.input_hidden, &mut model.input_hidden)?;
        copy_var(&self.input_piece_hidden, &mut model.input_piece_hidden)?;
        copy_var(&self.input_rank_hidden, &mut model.input_rank_hidden)?;
        copy_var(&self.input_file_hidden, &mut model.input_file_hidden)?;
        copy_var(
            &self.input_king_piece_hidden,
            &mut model.input_king_piece_hidden,
        )?;
        copy_var(&self.rule_context_hidden, &mut model.rule_context_hidden)?;
        copy_var(&self.check_context_hidden, &mut model.check_context_hidden)?;
        copy_var(&self.hidden_bias, &mut model.hidden_bias)?;
        copy_var(&self.value_head_hidden, &mut model.value_head_hidden)?;
        copy_var(&self.value_head_bias, &mut model.value_head_bias)?;
        copy_var(
            &self.value_king_piece_hidden,
            &mut model.value_king_piece_hidden,
        )?;
        copy_var(&self.value_head_output, &mut model.value_head_output)?;
        copy_var(
            &self.value_threat_embedding,
            &mut model.value_threat_embedding,
        )?;
        copy_var(&self.value_threat_output, &mut model.value_threat_output)?;
        copy_var(
            &self.policy_threat_context,
            &mut model.policy_threat_context,
        )?;
        copy_var(&self.policy_move_bias, &mut model.policy_move_bias)?;
        copy_var(
            &self.policy_consequence_output,
            &mut model.policy_consequence_output,
        )?;
        copy_var(
            &self.policy_context_hidden,
            &mut model.policy_context_hidden,
        )?;
        copy_var(&self.policy_move_context, &mut model.policy_move_context)?;
        copy_var(
            &self.policy_accumulator_hidden,
            &mut model.policy_accumulator_hidden,
        )?;
        copy_var(
            &self.policy_accumulator_move,
            &mut model.policy_accumulator_move,
        )?;
        copy_var(&self.policy_sparse_table, &mut model.policy_sparse_table)?;
        copy_var(&self.policy_sparse_factor, &mut model.policy_sparse_factor)?;
        copy_var(&self.policy_tactical, &mut model.policy_tactical)?;
        copy_var(
            &self.policy_repetition_hidden,
            &mut model.policy_repetition_hidden,
        )?;
        copy_var(
            &self.policy_repetition_bias,
            &mut model.policy_repetition_bias,
        )?;
        model.rebuild_value_threat();
        model.rebuild_check_context();
        model.rebuild_policy_tactical();
        model.rebuild_policy_cache();
        Ok(())
    }
}

fn var_from_slice<S: Into<candle_core::Shape>>(
    values: &[f32],
    shape: S,
    device: &Device,
) -> CandleResult<Var> {
    Var::from_slice(values, shape, device)
}

fn copy_var(var: &Var, dst: &mut [f32]) -> CandleResult<()> {
    let values = var.as_detached_tensor().flatten_all()?.to_vec1::<f32>()?;
    dst.copy_from_slice(&values);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        az::nnue::extract_sparse_features_az,
        az::{
            AzEvalScratch, AzSampleMeta, AzTrainingSample, POLICY_SPARSE_TABLE_SIZE,
            RULE_CONTEXT_SIZE, canonical_buckets_for_perspective, dense_move_index,
            policy_consequence_features, policy_sparse_capture_index, policy_sparse_factor_indices,
            policy_sparse_main_index,
        },
        xiangqi::Position,
    };

    #[test]
    #[ignore = "需要本地已训练检查点、Px0数据及CUDA；只核对前向"]
    fn trained_cuda_forward_matches_search_cpu() {
        let path = std::env::var("CHINESEAI_AUDIT_MODEL")
            .unwrap_or_else(|_| "tmp/px0-reservoir-131072.epoch-3.safetensors".into());
        let model = AzNnue::load(path).unwrap();
        let dataset =
            crate::az::px0_data::load(std::path::Path::new("data/data.bin"), 1024, 1024).unwrap();
        let stride = (dataset.train.len() / 128).max(1);
        let samples: Vec<_> = dataset
            .train
            .iter()
            .step_by(stride)
            .take(128)
            .cloned()
            .collect();
        assert!(samples.len() >= 64);
        let device = Device::new_cuda(0).unwrap();
        let candle = AzCandleModel::from_model(&model, &device).unwrap();
        let ids: Vec<_> = (0..samples.len()).collect();
        let batch =
            BatchTensors::from_packed(PackedBatch::from_indices(&samples, &ids), &device).unwrap();
        let forward = candle.forward(&batch).unwrap();
        let wdl = candle_nn::ops::softmax(&forward.value_logits, 1)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let policy = forward.policy_logits.to_vec2::<f32>().unwrap();
        let mut max_value = 0f32;
        let mut max_policy = 0f32;
        let mut max_wdl = 0f32;
        let mut moves_checked = 0;
        for (row, sample) in samples.iter().enumerate() {
            let position = crate::az::position_for_training_sample(sample).unwrap();
            let moves: Vec<_> = sample
                .move_indices
                .iter()
                .map(|&index| {
                    let (from, to) = crate::az::dense_move_squares(index).unwrap();
                    crate::xiangqi::Move::new(from, to)
                })
                .collect();
            let mut scratch = AzEvalScratch::new(model.arch);
            let cpu = model.evaluate_with_scratch_output_with_repetition(
                &position,
                &moves,
                &sample.repetition_flags,
                &sample.rule_context,
                &mut scratch,
            );
            max_value = max_value.max((cpu.value - (wdl[row][0] - wdl[row][2])).abs());
            for (a, b) in cpu.value_wdl.iter().zip(&wdl[row]) {
                max_wdl = max_wdl.max((a - b).abs());
            }
            for (a, b) in scratch.logits.iter().zip(&policy[row]) {
                max_policy = max_policy.max((a - b).abs());
            }
            moves_checked += moves.len();
            assert!(
                (cpu.value - (wdl[row][0] - wdl[row][2])).abs() < 1e-4,
                "row={row} fen={}",
                position.to_fen()
            );
        }
        println!(
            "trained GPU/CPU audit: samples={} moves={moves_checked} max_value={max_value:.8} max_wdl={max_wdl:.8} max_policy={max_policy:.8}",
            samples.len()
        );
        assert!(max_wdl < 1e-4 && max_policy < 1e-3);
    }

    #[test]
    fn candle_and_cpu_policy_consequence_logits_match() {
        let position =
            Position::from_fen("1rbakab1r/9/4c3n/p3p3P/2p6/1C2c1pN1/P1P6/4B2C1/4A4/1RBAK3R w")
                .unwrap();
        let moves = position.legal_moves();
        let mut model = AzNnue::random(32, 20260730);
        for (index, weight) in model.policy_consequence_output.iter_mut().enumerate() {
            *weight = (index as f32 + 1.0) * 0.003;
        }
        for (index, weight) in model.policy_move_context.iter_mut().enumerate() {
            *weight = ((index % POLICY_MOVE_CONTEXT_SIZE) as f32 + 1.0) * 0.002;
        }
        for (index, weight) in model.policy_accumulator_move.iter_mut().enumerate() {
            *weight = ((index % POLICY_ACCUMULATOR_RANK) as f32 + 1.0) * 0.0002;
        }
        for (index, weight) in model.value_threat_output.iter_mut().enumerate() {
            *weight = (index % 17) as f32 * 0.0003 - 0.002;
        }
        for (index, weight) in model.value_king_piece_hidden.iter_mut().enumerate() {
            *weight = (index % 19) as f32 * 0.001 - 0.009;
        }
        for (index, weight) in model.value_head_output.iter_mut().enumerate() {
            *weight = (index % 11) as f32 * 0.003 - 0.015;
        }
        for (index, weight) in model.check_context_hidden.iter_mut().enumerate() {
            *weight = (index % 17) as f32 * 0.01 - 0.08;
        }
        model.rebuild_check_context();
        model.value_head_bias.fill(1.0);
        for (index, weight) in model.policy_threat_context.iter_mut().enumerate() {
            *weight = (index % 13) as f32 * 0.0001 - 0.0005;
        }
        for (index, weight) in model.policy_tactical.iter_mut().enumerate() {
            *weight = (index % 11) as f32 * 0.0002 - 0.001;
        }
        model.policy_sparse_table[POLICY_SPARSE_TABLE_SIZE - 1] = 0.127;
        model.policy_repetition_hidden.fill(0.01);
        model.policy_repetition_bias[0] = 0.25;
        let side = position.side_to_move();
        let buckets = canonical_buckets_for_perspective(&position, side);
        for (index, &mv) in moves.iter().enumerate() {
            let move_index = dense_move_index(mv);
            let (from, _, captured) = policy_consequence_features(&position, side, mv).unwrap();
            let main = policy_sparse_main_index(move_index, from / 90, buckets.0, buckets.1);
            let capture =
                policy_sparse_capture_index(move_index, captured.map(|feature| feature / 90));
            model.policy_sparse_table[main] = (index as f32 % 101.0) * 0.001;
            model.policy_sparse_table[capture] = -((index as f32 % 53.0) * 0.001);
            for (factor_offset, factor) in
                policy_sparse_factor_indices(move_index, from / 90, buckets.0, buckets.1)
                    .into_iter()
                    .enumerate()
            {
                model.policy_sparse_factor[factor] =
                    ((index + factor_offset) as f32 % 37.0) * 0.001;
            }
        }
        model.rebuild_policy_cache();
        model.rebuild_policy_tactical();
        model.rebuild_value_threat();

        let mut cpu = AzEvalScratch::new(model.arch);
        let mut repetition_flags = vec![0; moves.len()];
        repetition_flags[0] = 1;
        let cpu_output = model.evaluate_with_scratch_output_with_repetition(
            &position,
            &moves,
            &repetition_flags,
            &[0.0; RULE_CONTEXT_SIZE],
            &mut cpu,
        );

        let sample = AzTrainingSample {
            repetition_flags,
            features: extract_sparse_features_az(&position),
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            move_indices: moves.iter().map(|&mv| dense_move_index(mv)).collect(),
            policy: vec![1.0; moves.len()],
            value_wdl: [0.0, 1.0, 0.0],
            root_search_wdl: [0.0, 1.0, 0.0],
            value: 0.0,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            search_simulations: 1,
            meta: AzSampleMeta::default(),
        };
        let (sample_wdl, sample_logits) =
            crate::az::outputs_for_training_sample(&model, &sample).unwrap();
        assert_eq!(sample_wdl, cpu_output.value_wdl);
        assert_eq!(sample_logits, cpu.logits);
        let packed = PackedBatch::from_indices(&[sample], &[0]);
        let batch = BatchTensors::from_packed(packed, &Device::Cpu).unwrap();
        let candle = AzCandleModel::from_model(&model, &Device::Cpu).unwrap();
        let forward = candle.forward(&batch).unwrap();
        let legal = forward.policy_logits.to_vec2::<f32>().unwrap();
        let value_logits = forward.value_logits.to_vec2::<f32>().unwrap();
        let candle_wdl = crate::az::softmax_fixed3(value_logits[0].clone().try_into().unwrap());

        for (candle_value, cpu_value) in candle_wdl.iter().zip(cpu_output.value_wdl) {
            assert!(
                (candle_value - cpu_value).abs() < 2.0e-5,
                "candle={candle_value} cpu={cpu_value}"
            );
        }
        let gradients = forward.value_logits.sum_all().unwrap().backward().unwrap();
        let king_gradient = gradients
            .get(&candle.value_king_piece_hidden)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(king_gradient.iter().any(|&gradient| gradient != 0.0));

        assert_eq!(legal[0].len(), cpu.logits.len());
        for (candle_logit, cpu_logit) in legal[0].iter().zip(&cpu.logits) {
            assert!(
                (candle_logit - cpu_logit).abs() < 2.0e-3,
                "candle={candle_logit} cpu={cpu_logit}"
            );
        }

        let mut gradient_model = AzNnue::random(32, 20260731);
        gradient_model.policy_move_context.fill(0.01);
        let gradient_candle = AzCandleModel::from_model(&gradient_model, &Device::Cpu).unwrap();
        let gradient_forward = gradient_candle.forward(&batch).unwrap();
        let gradients = gradient_forward
            .policy_logits
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let tactical_gradient = gradients
            .get(&gradient_candle.policy_tactical)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            tactical_gradient[super::super::POLICY_CAPTURE_RELATION_OFFSET..]
                .iter()
                .any(|&gradient| gradient != 0.0)
        );
        let repetition_gradient = gradients
            .get(&gradient_candle.policy_repetition_hidden)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            repetition_gradient
                .iter()
                .any(|&value| value.abs() > 1.0e-8)
        );
        let output_gradient = gradients
            .get(&gradient_candle.policy_consequence_output)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            output_gradient
                .iter()
                .any(|gradient| gradient.abs() > 1.0e-8)
        );
        let move_context_gradient = gradients
            .get(&gradient_candle.policy_move_context)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            move_context_gradient
                .iter()
                .any(|gradient| gradient.abs() > 1.0e-8)
        );
        let accumulator_move_gradient = gradients
            .get(&gradient_candle.policy_accumulator_move)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            accumulator_move_gradient
                .iter()
                .any(|gradient| gradient.abs() > 1.0e-8)
        );
    }

    #[test]
    fn gpu_and_cpu_weight_tensors_roundtrip() {
        let mut model = AzNnue::random(16, 12345);
        model.check_context_hidden.fill(0.125);
        model.rebuild_check_context();
        let candle = AzCandleModel::from_model(&model, &Device::Cpu).unwrap();
        let mut back = AzNnue::random_with_arch(model.arch, 54321);
        candle.copy_to_model(&mut back).unwrap();

        macro_rules! assert_weight_parity {
            ($($field:ident),* $(,)?) => {
                $(
                    assert_eq!(
                        model.$field, back.$field,
                        "weight tensor `{}` drifted between CPU and GPU paths",
                        stringify!($field)
                    );
                )*
            };
        }
        assert_weight_parity!(
            input_hidden,
            input_piece_hidden,
            input_rank_hidden,
            input_file_hidden,
            input_king_piece_hidden,
            rule_context_hidden,
            check_context_hidden,
            hidden_bias,
            value_head_hidden,
            value_head_bias,
            value_king_piece_hidden,
            value_head_output,
            policy_threat_context,
            policy_move_bias,
            policy_consequence_output,
            policy_context_hidden,
            policy_move_context,
            policy_accumulator_hidden,
            policy_accumulator_move,
            policy_sparse_table,
            policy_sparse_factor,
            policy_tactical,
        );
        assert!(back.check_context_active);
    }

    fn check_context_training_matches_inference(device: &Device) {
        use crate::az::{nnue::canonical_move, px0_sgd::Px0Sgd};

        let positions = [
            Position::startpos(),
            Position::from_fen("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b")
                .unwrap(),
            Position::from_fen(
                "2bakab2/9/5r1c1/p1PRC1p2/4P2nP/6P2/4N1r2/7c1/4A4/2BAK1B1R b - - 0 1",
            )
            .unwrap(),
            Position::from_fen("3k5/9/9/9/9/9/9/4r4/9/4K4 w").unwrap(),
        ];
        let positions = positions
            .iter()
            .flat_map(|p| [p.clone(), p.mirror_files()])
            .collect::<Vec<_>>();
        let samples = positions
            .iter()
            .map(|position| {
                let moves = position.legal_moves();
                AzTrainingSample {
                    features: extract_sparse_features_az(position),
                    move_indices: moves
                        .iter()
                        .map(|&mv| dense_move_index(canonical_move(position.side_to_move(), mv)))
                        .collect(),
                    policy: vec![1.0 / moves.len() as f32; moves.len()],
                    repetition_flags: vec![0; moves.len()],
                    rule_context: [0.0; RULE_CONTEXT_SIZE],
                    value_wdl: [1.0, 0.0, 0.0],
                    root_search_wdl: [1.0, 0.0, 0.0],
                    value: 1.0,
                    side_sign: 1.0,
                    policy_weight: 1.0,
                    value_weight: 1.0,
                    search_simulations: 1,
                    meta: AzSampleMeta::default(),
                }
            })
            .collect::<Vec<_>>();
        let ids = (0..samples.len()).collect::<Vec<_>>();
        let packed = PackedBatch::from_indices(&samples, &ids);
        for (row, position) in positions.iter().enumerate() {
            let moves = position.legal_moves();
            let checks = moves
                .iter()
                .map(|&mv| f32::from(position.gives_check_after_move_fast(mv)))
                .collect::<Vec<_>>();
            let context = crate::az::inference::check_context_features(
                position,
                &moves,
                &checks,
                position.attacked_squares_masks(),
            );
            assert_eq!(
                &packed.check_context[row * CHECK_CONTEXT_SIZE..(row + 1) * CHECK_CONTEXT_SIZE],
                &context
            );
        }
        let batch = BatchTensors::from_packed(packed, device).unwrap();
        let mut model = AzNnue::random(32, 20261002);
        model.value_head_bias.fill(1.0);
        // 新标量权重为零，已有价值头需能把梯度传回主干。
        for (index, weight) in model.value_head_output.iter_mut().enumerate() {
            *weight = (index % 11) as f32 * 0.003 - 0.015;
        }
        let candle = AzCandleModel::from_model(&model, device).unwrap();
        let forward = candle.forward(&batch).unwrap();
        let loss = candle_nn::ops::log_softmax(&forward.value_logits, 1)
            .unwrap()
            .mul(&batch.value_wdl)
            .unwrap()
            .sum_all()
            .unwrap()
            .neg()
            .unwrap();
        let gradients = loss.backward().unwrap();
        let gradient = gradients
            .get(&candle.check_context_hidden)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(gradient.iter().all(|g| g.is_finite()));
        assert!(gradient.iter().any(|g| g.abs() > 1e-8));
        let mut optimizer = Px0Sgd::new(candle.all_vars(), 0.02).unwrap();
        optimizer.step(&gradients).unwrap();
        candle.copy_to_model(&mut model).unwrap();
        assert!(model.check_context_hidden.iter().any(|&w| w != 0.0));
        assert!(model.check_context_active);
        let forward = candle.forward(&batch).unwrap();
        let wdl = candle_nn::ops::softmax(&forward.value_logits, 1)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let logits = forward.policy_logits.to_vec2::<f32>().unwrap();
        for (row, sample) in samples.iter().enumerate() {
            let position = &positions[row];
            let moves = position.legal_moves();
            let mut scratch = AzEvalScratch::new(model.arch);
            let cpu = model.evaluate_with_scratch_output_with_repetition(
                position,
                &moves,
                &sample.repetition_flags,
                &sample.rule_context,
                &mut scratch,
            );
            let (sample_wdl, sample_logits) =
                crate::az::outputs_for_training_sample(&model, sample).unwrap();
            for (a, b) in cpu.value_wdl.iter().zip(&wdl[row]) {
                assert!((a - b).abs() < 1e-4, "row={row} value {a} != {b}");
            }
            for (a, b) in sample_wdl.iter().zip(&wdl[row]) {
                assert!((a - b).abs() < 1e-4);
            }
            for (a, b) in sample_logits.iter().zip(&logits[row]) {
                assert!((a - b).abs() < 2e-3, "row={row} policy {a} != {b}");
            }
        }
    }

    #[test]
    fn check_context_training_updates_zero_weights_and_matches_cpu() {
        check_context_training_matches_inference(&Device::Cpu);
    }

    #[cfg(feature = "slow-tests")]
    #[test]
    fn check_context_training_updates_zero_weights_and_matches_cuda() {
        let device =
            crate::az::cuda_test_device::shared_cuda_device().expect("CUDA device required");
        check_context_training_matches_inference(device);
    }
}
