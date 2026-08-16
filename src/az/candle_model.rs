use candle_core::{DType, Device, Result as CandleResult, Tensor, Var, backprop::GradStore};

use super::{
    AzNnue, AzNnueArch, DENSE_MOVE_SPACE, POLICY_ACCUMULATOR_RANK, POLICY_CONSEQUENCE_SIZE,
    POLICY_MOVE_CONTEXT_SIZE, POLICY_SPARSE_FACTOR_SIZE, POLICY_SPARSE_TABLE_SIZE,
    RULE_CONTEXT_SIZE, STRUCTURAL_FILE_SIZE, STRUCTURAL_KING_PIECE_SIZE, STRUCTURAL_PIECE_SIZE,
    STRUCTURAL_RANK_SIZE, VALUE_HEAD_SIZE, WDL_HEAD_SIZE, dataloader::PackedBatch,
    fused_feature_pool::feature_pool, fused_policy::fused_policy,
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
    value_head_hidden: Var,
    value_head_bias: Var,
    value_head_output: Var,
    policy_move_bias: Var,
    policy_consequence_output: Var,
    policy_context_hidden: Var,
    policy_move_context: Var,
    policy_accumulator_hidden: Var,
    policy_accumulator_move: Var,
    policy_sparse_table: Var,
    policy_sparse_factor: Var,
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
        let value_head = hidden
            .matmul(&self.value_head_hidden.t()?)?
            .broadcast_add(&self.value_head_bias)?
            .relu()?;
        let value_logits = value_head.matmul(&self.value_head_output.t()?)?;
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
        let policy_context = hidden.matmul(&self.policy_context_hidden.t()?)?;
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
        let sparse_logits = sparse_tables
            .index_select(&batch.policy_sparse_indices.flatten_all()?, 0)?
            .reshape((batch.batch_size, batch.max_policy_moves, 7))?
            .sum(2)?;
        let policy_logits = (policy_logits + sparse_logits)?;

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
    pub(super) policy_items: Tensor,
    pub(super) policy_sparse_indices: Tensor,
    pub(super) max_policy_moves: usize,
    pub(super) policy_targets: Tensor,
    pub(super) policy_mask: Tensor,
    pub(super) value_wdl: Tensor,
    pub(super) values: Tensor,
    pub(super) rule_context: Tensor,
    pub(super) policy_weights: Tensor,
    pub(super) value_weights: Tensor,
    pub(super) value_phase_masks: Tensor,
}

impl BatchTensors {
    pub(super) fn from_packed(packed: PackedBatch, device: &Device) -> CandleResult<Self> {
        let batch_size = packed.batch_size;
        let max_features = packed.max_features;
        let max_policy_moves = packed.max_policy_moves;
        Ok(Self {
            batch_size,
            feature_items: Tensor::from_vec(
                packed.feature_items,
                (batch_size, max_features),
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
            max_policy_moves,
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
            values: Tensor::from_vec(packed.values, batch_size, device)?,
            rule_context: Tensor::from_vec(
                packed.rule_context,
                (batch_size, RULE_CONTEXT_SIZE),
                device,
            )?,
            policy_weights: Tensor::from_vec(packed.policy_weights, batch_size, device)?,
            value_weights: Tensor::from_vec(packed.value_weights, batch_size, device)?,
            value_phase_masks: Tensor::from_vec(packed.value_phase_masks, (batch_size, 3), device)?,
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
            value_head_hidden: var_from_slice(
                &model.value_head_hidden,
                (VALUE_HEAD_SIZE, hidden),
                device,
            )?,
            value_head_bias: var_from_slice(&model.value_head_bias, VALUE_HEAD_SIZE, device)?,
            value_head_output: var_from_slice(
                &model.value_head_output,
                (WDL_HEAD_SIZE, VALUE_HEAD_SIZE),
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
        vars.push(self.value_head_hidden.clone());
        vars.push(self.value_head_bias.clone());
        vars.push(self.value_head_output.clone());
        vars.push(self.policy_move_bias.clone());
        vars.push(self.policy_consequence_output.clone());
        vars.push(self.policy_context_hidden.clone());
        vars.push(self.policy_move_context.clone());
        vars.push(self.policy_accumulator_hidden.clone());
        vars.push(self.policy_accumulator_move.clone());
        vars.push(self.policy_sparse_table.clone());
        vars.push(self.policy_sparse_factor.clone());
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
        copy_var(&self.value_head_hidden, &mut model.value_head_hidden)?;
        copy_var(&self.value_head_bias, &mut model.value_head_bias)?;
        copy_var(&self.value_head_output, &mut model.value_head_output)?;
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
        model.rebuild_policy_accumulator_quantization();
        Ok(())
    }

    pub(super) fn cpu_grads(&self, grads: &GradStore) -> CandleResult<Vec<Option<Vec<f32>>>> {
        let mut out = Vec::new();
        for var in self.all_vars() {
            let grad = grads
                .get(&var)
                .map(|grad| grad.flatten_all()?.to_vec1::<f32>())
                .transpose()?;
            out.push(grad);
        }
        Ok(out)
    }

    pub(super) fn to_cpu_values(&self) -> CandleResult<Vec<Vec<f32>>> {
        let mut values = Vec::new();
        for var in self.all_vars() {
            values.push(var.as_detached_tensor().flatten_all()?.to_vec1::<f32>()?);
        }
        Ok(values)
    }

    pub(super) fn set_from_cpu_values(&self, values: &[Vec<f32>]) -> CandleResult<()> {
        for (var, values) in self.all_vars().iter().zip(values.iter()) {
            let tensor = Tensor::from_vec(values.clone(), var.shape().clone(), var.device())?;
            var.set(&tensor)?;
        }
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
        model.evaluate_with_scratch_output(&position, &moves, &[0.0; RULE_CONTEXT_SIZE], &mut cpu);

        let sample = AzTrainingSample {
            features: extract_sparse_features_az(&position),
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            move_indices: moves.iter().map(|&mv| dense_move_index(mv)).collect(),
            policy: vec![1.0; moves.len()],
            value_wdl: [0.0, 1.0, 0.0],
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
            value_head_hidden,
            value_head_bias,
            value_head_output,
            policy_move_bias,
            policy_consequence_output,
            policy_context_hidden,
            policy_move_context,
            policy_accumulator_hidden,
            policy_accumulator_move,
        );
    }
}
