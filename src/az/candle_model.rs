use candle_core::{DType, Device, Result as CandleResult, Tensor, Var};

use super::{
    AzNnue, AzNnueArch, DENSE_MOVE_SPACE, PIKAFISH_LAYER_STACKS, PIKAFISH_PSQ_DIMENSIONS,
    PIKAFISH_PSQT_BUCKETS, PIKAFISH_TRANSFORMED_DIMENSIONS, PIKAFISH_TRANSFORMER_DIMENSIONS,
    PIKAFISH_TRANSFORMER_HALF, PIKAFISH_VALUE_FC0, PIKAFISH_VALUE_FC1, PIKAFISH_VALUE_TAIL,
    PIKAFISH_VALUE_THREAT_DIMENSIONS, POLICY_ACCUMULATOR_RANK, POLICY_CONSEQUENCE_SIZE,
    POLICY_MOVE_CONTEXT_SIZE, POLICY_SPARSE_FACTOR_SIZE, POLICY_SPARSE_TABLE_SIZE,
    POLICY_TACTICAL_SIZE, POLICY_THREAT_CONTEXT_SIZE, RULE_CONTEXT_SIZE, STRUCTURAL_FILE_SIZE,
    STRUCTURAL_KING_PIECE_SIZE, STRUCTURAL_PIECE_SIZE, STRUCTURAL_RANK_SIZE, VALUE_THREAT_RANK,
    VALUE_THREAT_VOCAB, WDL_HEAD_SIZE,
    dataloader::PackedBatch,
    fused_feature_pool::{PADDING_ITEM, feature_pool, sparse_pool},
    fused_policy::fused_policy,
    fused_sparse_policy::{sparse_policy, tactical_policy},
};
use crate::nnue::AZ_NNUE_INPUT_SIZE;

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
    hidden_bias: Var,
    pikafish_psq_embedding: Var,
    pikafish_threat_embedding: Var,
    pikafish_psqt: Var,
    pikafish_value_fc0: Var,
    pikafish_value_fc0_bias: Var,
    pikafish_value_rule_fc0: Var,
    pikafish_value_fc1: Var,
    pikafish_value_fc1_bias: Var,
    pikafish_value_output: Var,
    pikafish_value_output_bias: Var,
    pikafish_short_value_output: Var,
    pikafish_short_value_bias: Var,
    policy_threat_embedding: Var,
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
        let sparse_pre = (board_pre + rule_pre)?;
        let sparse_hidden = sparse_pre.relu()?;
        let rms = sparse_hidden
            .sqr()?
            .mean_keepdim(1)?
            .affine(1.0, RMS_NORM_EPS)?
            .sqrt()?;
        let hidden = sparse_hidden.broadcast_div(&rms)?;
        let threat_accumulator = sparse_pool(
            self.policy_threat_embedding.as_tensor(),
            &batch.value_threat_indices,
        )?;
        let threat_activation = threat_accumulator.clamp(0.0f64, 1.0f64)?;
        let threat_pair = Tensor::cat(&[&threat_activation, &threat_activation.sqr()?], 1)?;
        let value_accumulator = (sparse_pool(
            self.pikafish_psq_embedding.as_tensor(),
            &batch.pikafish_psq_indices,
        )? + sparse_pool(
            self.pikafish_threat_embedding.as_tensor(),
            &batch.pikafish_threat_indices,
        )?)?;
        let left = value_accumulator
            .narrow(1, 0, PIKAFISH_TRANSFORMER_HALF)?
            .clamp(0.0f64, 1.0f64)?;
        let right = value_accumulator
            .narrow(1, PIKAFISH_TRANSFORMER_HALF, PIKAFISH_TRANSFORMER_HALF)?
            .clamp(0.0f64, 1.0f64)?;
        let value_views =
            (left * right)?.reshape((batch.batch_size, 2, PIKAFISH_TRANSFORMER_HALF))?;
        let value_input = Tensor::cat(
            &[
                &value_views.narrow(1, 0, 1)?.squeeze(1)?,
                &value_views.narrow(1, 1, 1)?.squeeze(1)?,
            ],
            1,
        )?;
        let fc0 = value_input
            .matmul(&self.pikafish_value_fc0.t()?)?
            .broadcast_add(&self.pikafish_value_fc0_bias)?
            .broadcast_add(
                &batch
                    .rule_context
                    .matmul(&self.pikafish_value_rule_fc0.t()?)?,
            )?
            .reshape((batch.batch_size, PIKAFISH_LAYER_STACKS, PIKAFISH_VALUE_FC0))?;
        let fc0_clip = fc0.clamp(0.0f64, 1.0f64)?;
        let fc0_pair = Tensor::cat(&[&fc0_clip.sqr()?, &fc0_clip], 2)?;
        let fc1_weight = self
            .pikafish_value_fc1
            .reshape((
                1,
                PIKAFISH_LAYER_STACKS,
                PIKAFISH_VALUE_FC1,
                PIKAFISH_VALUE_FC0 * 2,
            ))?
            .transpose(2, 3)?;
        let fc1_all = fc0_pair
            .unsqueeze(2)?
            .broadcast_matmul(&fc1_weight)?
            .squeeze(2)?
            .broadcast_add(
                &self
                    .pikafish_value_fc1_bias
                    .reshape((PIKAFISH_LAYER_STACKS, PIKAFISH_VALUE_FC1))?,
            )?;
        let fc1_clip = fc1_all.clamp(0.0f64, 1.0f64)?;
        let tail = Tensor::cat(&[&fc0_pair, &fc1_clip.sqr()?, &fc1_clip], 2)?;
        let stack_mask = batch.layer_stack_one_hot.unsqueeze(2)?;
        let selected_tail = tail.broadcast_mul(&stack_mask)?.sum(1)?;
        let selected_fc0 = fc0.broadcast_mul(&stack_mask)?.sum(1)?;
        let output_all = selected_tail
            .matmul(&self.pikafish_value_output.t()?)?
            .broadcast_add(&self.pikafish_value_output_bias)?
            .reshape((batch.batch_size, PIKAFISH_LAYER_STACKS, WDL_HEAD_SIZE))?;
        let mut value_logits = output_all.broadcast_mul(&stack_mask)?.sum(1)?;
        let skip = (selected_fc0.narrow(1, PIKAFISH_VALUE_FC0 - 2, 1)?
            - selected_fc0.narrow(1, PIKAFISH_VALUE_FC0 - 1, 1)?)?;
        let signed = Tensor::cat(&[&skip, &Tensor::zeros_like(&skip)?, &skip.neg()?], 1)?;
        let psqt_views = sparse_pool(self.pikafish_psqt.as_tensor(), &batch.pikafish_psq_indices)?
            .reshape((batch.batch_size, 2, PIKAFISH_PSQT_BUCKETS))?;
        let psqt_diff =
            (psqt_views.narrow(1, 0, 1)?.squeeze(1)? - psqt_views.narrow(1, 1, 1)?.squeeze(1)?)?;
        let psqt = (psqt_diff * batch.layer_stack_one_hot.clone())?.sum_keepdim(1)?;
        let psqt_signed = Tensor::cat(&[&psqt, &Tensor::zeros_like(&psqt)?, &psqt.neg()?], 1)?;
        value_logits = (value_logits + signed + psqt_signed)?;
        let short_all = selected_tail
            .matmul(&self.pikafish_short_value_output.t()?)?
            .broadcast_add(&self.pikafish_short_value_bias)?
            .reshape((
                batch.batch_size,
                PIKAFISH_LAYER_STACKS,
                super::SHORT_VALUE_HEADS,
                WDL_HEAD_SIZE,
            ))?;
        let short_value_logits = short_all
            .broadcast_mul(&batch.layer_stack_one_hot.unsqueeze(2)?.unsqueeze(3)?)?
            .sum(1)?;
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
        let policy_logits = (policy_logits + sparse_logits + tactical_logits)?;

        Ok(ForwardOutput {
            value_logits,
            short_value_logits,
            policy_logits,
        })
    }
}

pub(super) struct ForwardOutput {
    pub(super) value_logits: Tensor,
    pub(super) short_value_logits: Tensor,
    pub(super) policy_logits: Tensor,
}

pub(super) struct BatchTensors {
    pub(super) batch_size: usize,
    pub(super) feature_items: Tensor,
    pub(super) value_threat_indices: Tensor,
    pub(super) pikafish_psq_indices: Tensor,
    pub(super) pikafish_threat_indices: Tensor,
    pub(super) layer_stack_one_hot: Tensor,
    pub(super) policy_items: Tensor,
    pub(super) policy_sparse_indices: Tensor,
    pub(super) policy_tactical_indices: Tensor,
    pub(super) policy_targets: Tensor,
    pub(super) policy_mask: Tensor,
    pub(super) value_wdl: Tensor,
    pub(super) short_value_wdl: Tensor,
    pub(super) values: Tensor,
    pub(super) rule_context: Tensor,
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
        let max_pikafish_psq = packed.max_pikafish_psq;
        let max_pikafish_threats = packed.max_pikafish_threats;
        let mut layer_stack_one_hot = vec![0.0f32; batch_size * PIKAFISH_LAYER_STACKS];
        for (row, &stack) in packed.pikafish_layer_stacks.iter().enumerate() {
            layer_stack_one_hot[row * PIKAFISH_LAYER_STACKS + stack as usize] = 1.0;
        }
        assert!(
            packed
                .value_threat_indices
                .iter()
                .all(|&index| index == PADDING_ITEM || index < VALUE_THREAT_VOCAB as u32),
            "value threat index exceeds vocabulary"
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
            pikafish_psq_indices: Tensor::from_vec(
                packed.pikafish_psq_indices,
                (batch_size * 2, max_pikafish_psq),
                device,
            )?,
            pikafish_threat_indices: Tensor::from_vec(
                packed.pikafish_threat_indices,
                (batch_size * 2, max_pikafish_threats),
                device,
            )?,
            layer_stack_one_hot: Tensor::from_vec(
                layer_stack_one_hot,
                (batch_size, PIKAFISH_LAYER_STACKS),
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
                (batch_size, max_policy_moves, 2),
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
            value_wdl: Tensor::from_vec(packed.value_wdl, (batch_size, WDL_HEAD_SIZE), device)?,
            short_value_wdl: Tensor::from_vec(
                packed.short_value_wdl,
                (batch_size, super::SHORT_VALUE_HEADS, WDL_HEAD_SIZE),
                device,
            )?,
            values: Tensor::from_vec(packed.values, batch_size, device)?,
            rule_context: Tensor::from_vec(
                packed.rule_context,
                (batch_size, RULE_CONTEXT_SIZE),
                device,
            )?,
            policy_weights: Tensor::from_vec(packed.policy_weights, batch_size, device)?,
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
            pikafish_psq_embedding: var_from_slice(
                &model.pikafish_psq_embedding,
                (PIKAFISH_PSQ_DIMENSIONS, PIKAFISH_TRANSFORMER_DIMENSIONS),
                device,
            )?,
            pikafish_threat_embedding: var_from_slice(
                &model.pikafish_threat_embedding,
                (
                    PIKAFISH_VALUE_THREAT_DIMENSIONS,
                    PIKAFISH_TRANSFORMER_DIMENSIONS,
                ),
                device,
            )?,
            pikafish_psqt: var_from_slice(
                &model.pikafish_psqt,
                (PIKAFISH_PSQ_DIMENSIONS, PIKAFISH_PSQT_BUCKETS),
                device,
            )?,
            pikafish_value_fc0: var_from_slice(
                &model.pikafish_value_fc0,
                (
                    PIKAFISH_LAYER_STACKS * PIKAFISH_VALUE_FC0,
                    PIKAFISH_TRANSFORMED_DIMENSIONS,
                ),
                device,
            )?,
            pikafish_value_fc0_bias: var_from_slice(
                &model.pikafish_value_fc0_bias,
                PIKAFISH_LAYER_STACKS * PIKAFISH_VALUE_FC0,
                device,
            )?,
            pikafish_value_rule_fc0: var_from_slice(
                &model.pikafish_value_rule_fc0,
                (
                    PIKAFISH_LAYER_STACKS * PIKAFISH_VALUE_FC0,
                    RULE_CONTEXT_SIZE,
                ),
                device,
            )?,
            pikafish_value_fc1: var_from_slice(
                &model.pikafish_value_fc1,
                (
                    PIKAFISH_LAYER_STACKS * PIKAFISH_VALUE_FC1,
                    PIKAFISH_VALUE_FC0 * 2,
                ),
                device,
            )?,
            pikafish_value_fc1_bias: var_from_slice(
                &model.pikafish_value_fc1_bias,
                PIKAFISH_LAYER_STACKS * PIKAFISH_VALUE_FC1,
                device,
            )?,
            pikafish_value_output: var_from_slice(
                &model.pikafish_value_output,
                (PIKAFISH_LAYER_STACKS * WDL_HEAD_SIZE, PIKAFISH_VALUE_TAIL),
                device,
            )?,
            pikafish_value_output_bias: var_from_slice(
                &model.pikafish_value_output_bias,
                PIKAFISH_LAYER_STACKS * WDL_HEAD_SIZE,
                device,
            )?,
            pikafish_short_value_output: var_from_slice(
                &model.pikafish_short_value_output,
                (
                    PIKAFISH_LAYER_STACKS * super::SHORT_VALUE_HEADS * WDL_HEAD_SIZE,
                    PIKAFISH_VALUE_TAIL,
                ),
                device,
            )?,
            pikafish_short_value_bias: var_from_slice(
                &model.pikafish_short_value_bias,
                PIKAFISH_LAYER_STACKS * super::SHORT_VALUE_HEADS * WDL_HEAD_SIZE,
                device,
            )?,
            policy_threat_embedding: var_from_slice(
                &model.policy_threat_embedding,
                (VALUE_THREAT_VOCAB, VALUE_THREAT_RANK),
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
        vars.push(self.hidden_bias.clone());
        vars.push(self.pikafish_psq_embedding.clone());
        vars.push(self.pikafish_threat_embedding.clone());
        vars.push(self.pikafish_psqt.clone());
        vars.push(self.pikafish_value_fc0.clone());
        vars.push(self.pikafish_value_fc0_bias.clone());
        vars.push(self.pikafish_value_rule_fc0.clone());
        vars.push(self.pikafish_value_fc1.clone());
        vars.push(self.pikafish_value_fc1_bias.clone());
        vars.push(self.pikafish_value_output.clone());
        vars.push(self.pikafish_value_output_bias.clone());
        vars.push(self.pikafish_short_value_output.clone());
        vars.push(self.pikafish_short_value_bias.clone());
        vars.push(self.policy_threat_embedding.clone());
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
        copy_var(&self.hidden_bias, &mut model.hidden_bias)?;
        copy_var(
            &self.pikafish_psq_embedding,
            &mut model.pikafish_psq_embedding,
        )?;
        copy_var(
            &self.pikafish_threat_embedding,
            &mut model.pikafish_threat_embedding,
        )?;
        copy_var(&self.pikafish_psqt, &mut model.pikafish_psqt)?;
        copy_var(&self.pikafish_value_fc0, &mut model.pikafish_value_fc0)?;
        copy_var(
            &self.pikafish_value_fc0_bias,
            &mut model.pikafish_value_fc0_bias,
        )?;
        copy_var(
            &self.pikafish_value_rule_fc0,
            &mut model.pikafish_value_rule_fc0,
        )?;
        copy_var(&self.pikafish_value_fc1, &mut model.pikafish_value_fc1)?;
        copy_var(
            &self.pikafish_value_fc1_bias,
            &mut model.pikafish_value_fc1_bias,
        )?;
        copy_var(
            &self.pikafish_value_output,
            &mut model.pikafish_value_output,
        )?;
        copy_var(
            &self.pikafish_value_output_bias,
            &mut model.pikafish_value_output_bias,
        )?;
        copy_var(
            &self.pikafish_short_value_output,
            &mut model.pikafish_short_value_output,
        )?;
        copy_var(
            &self.pikafish_short_value_bias,
            &mut model.pikafish_short_value_bias,
        )?;
        copy_var(
            &self.policy_threat_embedding,
            &mut model.policy_threat_embedding,
        )?;
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
        model.rebuild_value_quantization();
        model.rebuild_policy_tactical();
        model.rebuild_policy_accumulator_quantization();
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
        az::{
            AzEvalScratch, AzSampleMeta, AzTrainingSample, POLICY_SPARSE_TABLE_SIZE,
            RULE_CONTEXT_SIZE, canonical_buckets_for_perspective, dense_move_index,
            policy_consequence_features, policy_sparse_capture_index, policy_sparse_factor_indices,
            policy_sparse_main_index,
        },
        nnue::extract_sparse_features_az,
        xiangqi::Position,
    };

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
        model.policy_sparse_table[POLICY_SPARSE_TABLE_SIZE - 1] = 0.127;
        for (index, weight) in model.pikafish_value_output.iter_mut().enumerate() {
            *weight = ((index % 29) as f32 - 14.0) * 0.0002;
        }
        for (index, bias) in model.pikafish_value_output_bias.iter_mut().enumerate() {
            *bias = ((index % 3) as f32 - 1.0) * 0.03;
        }
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
        model.rebuild_policy_accumulator_quantization();

        let mut cpu = AzEvalScratch::new(model.arch);
        let cpu_output = model.evaluate_with_scratch_output(
            &position,
            &moves,
            &[0.0; RULE_CONTEXT_SIZE],
            &mut cpu,
        );

        let sample = AzTrainingSample {
            features: extract_sparse_features_az(&position),
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            move_indices: moves.iter().map(|&mv| dense_move_index(mv)).collect(),
            policy: vec![1.0; moves.len()],
            value_wdl: [0.0, 1.0, 0.0],
            root_search_wdl: [0.0, 1.0, 0.0],
            short_value_wdl: [[0.0, 1.0, 0.0]; crate::az::SHORT_VALUE_HEADS],
            value: 0.0,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            search_simulations: 1,
            meta: AzSampleMeta::default(),
        };
        let packed = PackedBatch::from_indices(&[sample], &[0]);
        let batch = BatchTensors::from_packed(packed, &Device::Cpu).unwrap();
        let candle = AzCandleModel::from_model(&model, &Device::Cpu).unwrap();
        let forward = candle.forward(&batch).unwrap();
        let legal = forward.policy_logits.to_vec2::<f32>().unwrap();
        let value_logits = forward.value_logits.to_vec2::<f32>().unwrap();
        let candle_wdl = crate::az::softmax_fixed3(value_logits[0].as_slice().try_into().unwrap());
        for (left, right) in candle_wdl.iter().zip(cpu_output.value_wdl) {
            assert!(
                (left - right).abs() < 3.0e-3,
                "Candle/CPU value drift: {left} vs {right}"
            );
        }

        assert_eq!(legal[0].len(), cpu.logits.len());
        for (candle_logit, cpu_logit) in legal[0].iter().zip(&cpu.logits) {
            assert!(
                (candle_logit - cpu_logit).abs() < 2.0e-3,
                "candle={candle_logit} cpu={cpu_logit}"
            );
        }

        let gradient_model = AzNnue::random(32, 20260731);
        let gradient_candle = AzCandleModel::from_model(&gradient_model, &Device::Cpu).unwrap();
        let gradient_forward = gradient_candle.forward(&batch).unwrap();
        let gradients = gradient_forward
            .policy_logits
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
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

    /// CPU ???`AzNnue`?? Candle GPU ???????????
    /// `from_model` + `copy_to_model` ??????????????
    /// ?? CPU/GPU ???????????
    #[test]
    fn gpu_and_cpu_weight_tensors_roundtrip() {
        let model = AzNnue::random(16, 12345);
        let candle = AzCandleModel::from_model(&model, &Device::Cpu).unwrap();
        let mut back = AzNnue::random(16, 54321);
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
            hidden_bias,
            pikafish_psq_embedding,
            pikafish_threat_embedding,
            pikafish_psqt,
            pikafish_value_fc0,
            pikafish_value_fc1,
            pikafish_value_output,
            pikafish_short_value_output,
            policy_threat_context,
            policy_move_bias,
            policy_consequence_output,
            policy_context_hidden,
            policy_move_context,
            policy_accumulator_hidden,
            policy_accumulator_move,
        );
    }
}
