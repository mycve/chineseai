use candle_core::{Device, Result as CandleResult, Tensor, Var};

use super::{
    AbNnue, RULE_CONTEXT_SIZE, STRUCTURAL_FILE_SIZE, STRUCTURAL_KING_PIECE_SIZE,
    STRUCTURAL_PIECE_SIZE, STRUCTURAL_RANK_SIZE, VALUE_HEAD_SIZE, WDL_HEAD_SIZE,
    dataloader::PackedBatch, fused_feature_pool::feature_pool,
};
use crate::nnue::AB_NNUE_INPUT_SIZE;

const RMS_NORM_EPS: f64 = 1.0e-6;

#[derive(Debug)]
pub(super) struct AbCandleModel {
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
}

impl AbCandleModel {
    pub(super) fn forward(&self, batch: &BatchTensors) -> CandleResult<ForwardOutput> {
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
        let rule_pre = batch.rule_context.matmul(&self.rule_context_hidden)?;
        let hidden = (board_pre + rule_pre)?.relu()?;
        let rms = hidden
            .sqr()?
            .mean_keepdim(1)?
            .affine(1.0, RMS_NORM_EPS)?
            .sqrt()?;
        let hidden = hidden.broadcast_div(&rms)?;
        let value_head = hidden
            .matmul(&self.value_head_hidden.t()?)?
            .broadcast_add(&self.value_head_bias)?
            .relu()?;
        Ok(ForwardOutput {
            value_logits: value_head.matmul(&self.value_head_output.t()?)?,
        })
    }

    pub(super) fn from_model(model: &AbNnue, device: &Device) -> CandleResult<Self> {
        let arch = model.arch;
        let hidden = arch.hidden_size;
        Ok(Self {
            input_hidden: var_from_slice(
                &model.input_hidden,
                (AB_NNUE_INPUT_SIZE, hidden),
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
        })
    }

    pub(super) fn all_vars(&self) -> Vec<Var> {
        vec![
            self.input_hidden.clone(),
            self.input_piece_hidden.clone(),
            self.input_rank_hidden.clone(),
            self.input_file_hidden.clone(),
            self.input_king_piece_hidden.clone(),
            self.rule_context_hidden.clone(),
            self.hidden_bias.clone(),
            self.value_head_hidden.clone(),
            self.value_head_bias.clone(),
            self.value_head_output.clone(),
        ]
    }

    pub(super) fn copy_to_model(&self, model: &mut AbNnue) -> CandleResult<()> {
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
        Ok(())
    }
}

pub(super) struct ForwardOutput {
    pub(super) value_logits: Tensor,
}

pub(super) struct BatchTensors {
    pub(super) batch_size: usize,
    pub(super) feature_items: Tensor,
    pub(super) value_wdl: Tensor,
    pub(super) values: Tensor,
    pub(super) rule_context: Tensor,
    pub(super) value_weights: Tensor,
    pub(super) value_phase_masks: Tensor,
    pub(super) value_source_phase_masks: Tensor,
}

impl BatchTensors {
    pub(super) fn from_packed(packed: PackedBatch, device: &Device) -> CandleResult<Self> {
        let batch_size = packed.batch_size;
        Ok(Self {
            batch_size,
            feature_items: Tensor::from_vec(
                packed.feature_items,
                (batch_size, packed.max_features),
                device,
            )?,
            value_wdl: Tensor::from_vec(packed.value_wdl, (batch_size, WDL_HEAD_SIZE), device)?,
            values: Tensor::from_vec(packed.values, batch_size, device)?,
            rule_context: Tensor::from_vec(
                packed.rule_context,
                (batch_size, RULE_CONTEXT_SIZE),
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
