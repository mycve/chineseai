use super::{
    dataloader::PackedBatch,
    features::*,
    fused_feature_pool::{gather_rows, sparse_pool},
    nnue::AZ_NNUE_INPUT_SIZE,
    *,
};
use candle_core::{DType, Device, Result as CandleResult, Tensor, Var};
fn fold_input_weights(weights: &mut [f32], factors: &[f32], hidden: usize) {
    for (feature, row) in weights.chunks_exact_mut(hidden).enumerate() {
        for (factor, scale) in input_factor_terms(feature) {
            for (value, shared) in row
                .iter_mut()
                .zip(&factors[factor * hidden..(factor + 1) * hidden])
            {
                *value += scale * shared;
            }
        }
    }
}
fn fold_value_weights(weights: &mut [f32], shared: &[f32]) {
    for bucket in weights.chunks_exact_mut(WDL_HEAD_SIZE * VALUE_HEAD_SIZE) {
        for (value, shared) in bucket.iter_mut().zip(shared) {
            *value += shared;
        }
    }
}
#[derive(Debug)]
pub(super) struct AzCandleModel {
    arch: AzNnueArch,
    input_hidden: Var,
    // 训练专用；同步到 CPU 时折叠，推理不增加查表或矩阵运算。
    input_factors: Var,
    value_shared_output: Var,
    hidden_bias: Var,
    king_piece: Var,
    king_projection: Var,
    relation_embedding: Var,
    relation_projection: Var,
    motif_embedding: Var,
    motif_projection: Var,
    rule_context_hidden: Var,
    check_context_hidden: Var,
    value_head_hidden: Var,
    value_head_bias: Var,
    value_head_output: Var,
    moves_left_output: Var,
    moves_left_bias: Var,
    policy_move_bias: Var,
    policy_piece_square: Var,
    policy_context_hidden: Var,
    policy_move_context: Var,
    policy_move_piece: Var,
    policy_move_king: Var,
    policy_move_capture: Var,
    policy_local: Var,
    policy_joint: Var,
    policy_events: Var,
    policy_repetition_hidden: Var,
    policy_repetition_bias: Var,
}
impl AzCandleModel {
    pub(super) fn from_model(model: &AzNnue, device: &Device) -> CandleResult<Self> {
        let h = model.hidden_size;
        Ok(Self {
            arch: model.arch,
            input_hidden: Var::from_slice(&model.input_hidden, &[AZ_NNUE_INPUT_SIZE, h], device)?,
            input_factors: Var::zeros((INPUT_FACTOR_ROWS, h), DType::F32, device)?,
            value_shared_output: Var::zeros((WDL_HEAD_SIZE, VALUE_HEAD_SIZE), DType::F32, device)?,
            hidden_bias: Var::from_slice(&model.hidden_bias, &[h], device)?,
            king_piece: Var::from_slice(
                &model.king_piece,
                &[VALUE_KING_PIECE_VOCAB, VALUE_KING_PIECE_RANK],
                device,
            )?,
            king_projection: Var::from_slice(
                &model.king_projection,
                &[VALUE_KING_PIECE_RANK, h],
                device,
            )?,
            relation_embedding: Var::from_slice(
                &model.relation_embedding,
                &[RELATION_ROWS, RELATION_RANK],
                device,
            )?,
            relation_projection: Var::from_slice(
                &model.relation_projection,
                &[RELATION_RANK, h],
                device,
            )?,
            motif_embedding: Var::from_slice(
                &model.motif_embedding,
                &[MOTIF_ROWS, MOTIF_RANK],
                device,
            )?,
            motif_projection: Var::from_slice(&model.motif_projection, &[MOTIF_RANK, h], device)?,
            rule_context_hidden: Var::from_slice(
                &model.rule_context_hidden,
                &[RULE_CONTEXT_SIZE, h],
                device,
            )?,
            check_context_hidden: Var::from_slice(
                &model.check_context_hidden,
                &[CHECK_CONTEXT_SIZE, h],
                device,
            )?,
            value_head_hidden: Var::from_slice(
                &model.value_head_hidden,
                &[VALUE_HEAD_SIZE, h],
                device,
            )?,
            value_head_bias: Var::from_slice(&model.value_head_bias, &[VALUE_HEAD_SIZE], device)?,
            value_head_output: Var::from_slice(
                &model.value_head_output,
                &[VALUE_BUCKETS, WDL_HEAD_SIZE, VALUE_HEAD_SIZE],
                device,
            )?,
            moves_left_output: Var::from_slice(&model.moves_left_output, &[1, h], device)?,
            moves_left_bias: Var::from_slice(&model.moves_left_bias, &[1], device)?,
            policy_move_bias: Var::from_slice(
                &model.policy_move_bias,
                &[DENSE_MOVE_SPACE],
                device,
            )?,
            policy_piece_square: Var::from_slice(
                &model.policy_piece_square,
                &[AZ_NNUE_INPUT_SIZE],
                device,
            )?,
            policy_context_hidden: Var::from_slice(
                &model.policy_context_hidden,
                &[POLICY_MOVE_CONTEXT_SIZE, h],
                device,
            )?,
            policy_move_context: Var::from_slice(
                &model.policy_move_context,
                &[DENSE_MOVE_SPACE, POLICY_MOVE_CONTEXT_SIZE],
                device,
            )?,
            policy_move_piece: Var::from_slice(
                &model.policy_move_piece,
                &[DENSE_MOVE_SPACE, 7],
                device,
            )?,
            policy_move_king: Var::from_slice(
                &model.policy_move_king,
                &[2, DENSE_MOVE_SPACE, 9],
                device,
            )?,
            policy_move_capture: Var::from_slice(
                &model.policy_move_capture,
                &[DENSE_MOVE_SPACE, 8],
                device,
            )?,
            policy_local: Var::from_slice(&model.policy_local, &[MOVE_LOCAL_SIZE], device)?,
            policy_joint: Var::from_slice(&model.policy_joint, &[POLICY_JOINT_ROWS], device)?,
            policy_events: Var::from_slice(&model.policy_events, &[POLICY_EVENT_ROWS], device)?,
            policy_repetition_hidden: Var::from_slice(
                &model.policy_repetition_hidden,
                &[h],
                device,
            )?,
            policy_repetition_bias: Var::from_slice(&model.policy_repetition_bias, &[1], device)?,
        })
    }
    pub(super) fn all_vars(&self) -> Vec<Var> {
        vec![
            self.input_hidden.clone(),
            self.hidden_bias.clone(),
            self.king_piece.clone(),
            self.king_projection.clone(),
            self.relation_embedding.clone(),
            self.relation_projection.clone(),
            self.motif_embedding.clone(),
            self.motif_projection.clone(),
            self.rule_context_hidden.clone(),
            self.check_context_hidden.clone(),
            self.value_head_hidden.clone(),
            self.value_head_bias.clone(),
            self.value_head_output.clone(),
            self.moves_left_output.clone(),
            self.moves_left_bias.clone(),
            self.policy_move_bias.clone(),
            self.policy_piece_square.clone(),
            self.policy_context_hidden.clone(),
            self.policy_move_context.clone(),
            self.policy_move_piece.clone(),
            self.policy_move_king.clone(),
            self.policy_move_capture.clone(),
            self.policy_local.clone(),
            self.policy_joint.clone(),
            self.policy_events.clone(),
            self.policy_repetition_hidden.clone(),
            self.policy_repetition_bias.clone(),
            self.input_factors.clone(),
            self.value_shared_output.clone(),
        ]
    }
    pub(super) fn optimizer_parameters(&self) -> (Vec<Var>, Vec<bool>, Vec<f64>) {
        let vars = self.all_vars();
        // K/M 条件表及其投影容易在小样本中共同过拟合；保留前向和全部特征，
        // 只降低这些张量的 Adam 更新幅度，weight decay 仍沿用原分组。
        let scales = vars
            .iter()
            .map(|var| {
                if var.id() == self.king_piece.id()
                    || var.id() == self.king_projection.id()
                    || var.id() == self.motif_embedding.id()
                    || var.id() == self.motif_projection.id()
                {
                    0.1
                } else {
                    1.
                }
            })
            .collect();
        (
            vars,
            vec![
                false, false, false, true, false, true, false, true, true, true, true, false, true,
                true, false, false, false, true, false, false, false, false, true, false, false,
                true, false, false, true,
            ],
            scales,
        )
    }
    pub(super) fn copy_to_model(&self, model: &mut AzNnue) -> CandleResult<()> {
        model.input_hidden = self.input_hidden.flatten_all()?.to_vec1::<f32>()?;
        let factors = self.input_factors.flatten_all()?.to_vec1::<f32>()?;
        fold_input_weights(&mut model.input_hidden, &factors, model.hidden_size);
        model.hidden_bias = self.hidden_bias.flatten_all()?.to_vec1::<f32>()?;
        model.king_piece = self.king_piece.flatten_all()?.to_vec1::<f32>()?;
        model.king_projection = self.king_projection.flatten_all()?.to_vec1::<f32>()?;
        model.relation_embedding = self.relation_embedding.flatten_all()?.to_vec1::<f32>()?;
        model.relation_projection = self.relation_projection.flatten_all()?.to_vec1::<f32>()?;
        model.motif_embedding = self.motif_embedding.flatten_all()?.to_vec1::<f32>()?;
        model.motif_projection = self.motif_projection.flatten_all()?.to_vec1::<f32>()?;
        model.rule_context_hidden = self.rule_context_hidden.flatten_all()?.to_vec1::<f32>()?;
        model.check_context_hidden = self.check_context_hidden.flatten_all()?.to_vec1::<f32>()?;
        model.value_head_hidden = self.value_head_hidden.flatten_all()?.to_vec1::<f32>()?;
        model.value_head_bias = self.value_head_bias.flatten_all()?.to_vec1::<f32>()?;
        model.value_head_output = self.value_head_output.flatten_all()?.to_vec1::<f32>()?;
        let shared_output = self.value_shared_output.flatten_all()?.to_vec1::<f32>()?;
        fold_value_weights(&mut model.value_head_output, &shared_output);
        model.moves_left_output = self.moves_left_output.flatten_all()?.to_vec1::<f32>()?;
        model.moves_left_bias = self.moves_left_bias.flatten_all()?.to_vec1::<f32>()?;
        model.policy_move_bias = self.policy_move_bias.flatten_all()?.to_vec1::<f32>()?;
        model.policy_piece_square = self.policy_piece_square.flatten_all()?.to_vec1::<f32>()?;
        model.policy_context_hidden = self.policy_context_hidden.flatten_all()?.to_vec1::<f32>()?;
        model.policy_move_context = self.policy_move_context.flatten_all()?.to_vec1::<f32>()?;
        model.policy_move_piece = self.policy_move_piece.flatten_all()?.to_vec1::<f32>()?;
        model.policy_move_king = self.policy_move_king.flatten_all()?.to_vec1::<f32>()?;
        model.policy_move_capture = self.policy_move_capture.flatten_all()?.to_vec1::<f32>()?;
        model.policy_local = self.policy_local.flatten_all()?.to_vec1::<f32>()?;
        model.policy_joint = self.policy_joint.flatten_all()?.to_vec1::<f32>()?;
        model.policy_events = self.policy_events.flatten_all()?.to_vec1::<f32>()?;
        model.policy_repetition_hidden = self
            .policy_repetition_hidden
            .flatten_all()?
            .to_vec1::<f32>()?;
        model.policy_repetition_bias = self
            .policy_repetition_bias
            .flatten_all()?
            .to_vec1::<f32>()?;
        model.rebuild_moves_left();
        Ok(())
    }
    pub(super) fn folded_training_weights(&self, weights: &[Tensor]) -> CandleResult<Vec<Tensor>> {
        let vars = self.all_vars();
        let index = |target: &Var| {
            vars.iter()
                .position(|var| var.id() == target.id())
                .expect("trainable tensor")
        };
        let input = index(&self.input_hidden);
        let factors = index(&self.input_factors);
        let output = index(&self.value_head_output);
        let shared = index(&self.value_shared_output);
        let mut result = weights.to_vec();
        let mut input_weights = weights[input].flatten_all()?.to_vec1::<f32>()?;
        fold_input_weights(
            &mut input_weights,
            &weights[factors].flatten_all()?.to_vec1::<f32>()?,
            self.arch.hidden_size,
        );
        result[input] = Tensor::from_vec(input_weights, weights[input].shape(), &Device::Cpu)?;
        let mut output_weights = weights[output].flatten_all()?.to_vec1::<f32>()?;
        fold_value_weights(
            &mut output_weights,
            &weights[shared].flatten_all()?.to_vec1::<f32>()?,
        );
        result[output] = Tensor::from_vec(output_weights, weights[output].shape(), &Device::Cpu)?;
        result[factors] = weights[factors].zeros_like()?;
        result[shared] = weights[shared].zeros_like()?;
        Ok(result)
    }
    pub(super) fn forward(&self, b: &BatchTensors) -> CandleResult<ForwardOutput> {
        let board = (sparse_pool(self.input_hidden.as_tensor(), &b.feature_items)?
            + b.input_factors.matmul(self.input_factors.as_tensor())?)?;
        let king = sparse_pool(self.king_piece.as_tensor(), &b.king_indices)?
            .matmul(self.king_projection.as_tensor())?;
        let relation = sparse_pool(self.relation_embedding.as_tensor(), &b.relation_indices)?
            .matmul(self.relation_projection.as_tensor())?;
        let motif = sparse_pool(self.motif_embedding.as_tensor(), &b.motif_indices)?
            .matmul(self.motif_projection.as_tensor())?;
        let hidden = ((board + king)? + relation)?;
        let hidden = (hidden + motif)?;
        let hidden = (hidden
            + b.rule_context
                .matmul(self.rule_context_hidden.as_tensor())?)?;
        let hidden = (hidden
            + b.check_context
                .matmul(self.check_context_hidden.as_tensor())?)?
        .broadcast_add(&self.hidden_bias)?
        .relu()?;
        let head = hidden
            .matmul(&self.value_head_hidden.t()?)?
            .broadcast_add(&self.value_head_bias)?
            .relu()?;
        let selected = gather_rows(
            &self
                .value_head_output
                .broadcast_add(self.value_shared_output.as_tensor())?
                .reshape((VALUE_BUCKETS, 3 * VALUE_HEAD_SIZE))?,
            &b.value_buckets,
        )?
        .reshape((b.batch_size, 3, VALUE_HEAD_SIZE))?;
        let value_logits = selected.matmul(&head.unsqueeze(2)?)?.squeeze(2)?;
        let context = hidden.matmul(&self.policy_context_hidden.t()?)?;
        let move_context = gather_rows(
            self.policy_move_context.as_tensor(),
            &b.move_indices.flatten_all()?,
        )?
        .reshape((b.batch_size, b.max_policy_moves, POLICY_MOVE_CONTEXT_SIZE))?;
        let policy_context = move_context.broadcast_mul(&context.unsqueeze(1)?)?.sum(2)?;
        let tables = Tensor::cat(
            &[
                &self.policy_move_bias.flatten_all()?,
                &self.policy_move_piece.flatten_all()?,
                &self.policy_move_king.flatten_all()?,
                &self.policy_move_capture.flatten_all()?,
                &self.policy_joint.flatten_all()?,
                &self.policy_events.flatten_all()?,
            ],
            0,
        )?
        .unsqueeze(1)?;
        let sparse =
            sparse_pool(&tables, &b.policy_indices)?.reshape((b.batch_size, b.max_policy_moves))?;
        let zero = Tensor::zeros(1, DType::F32, self.policy_piece_square.device())?;
        let piece = Tensor::cat(&[self.policy_piece_square.as_tensor(), &zero], 0)?;
        let from = gather_rows(&piece, &b.policy_from.flatten_all()?)?
            .reshape((b.batch_size, b.max_policy_moves))?;
        let to = gather_rows(&piece, &b.policy_to.flatten_all()?)?
            .reshape((b.batch_size, b.max_policy_moves))?;
        let capture = gather_rows(&piece, &b.policy_capture.flatten_all()?)?
            .reshape((b.batch_size, b.max_policy_moves))?;
        let local = b
            .policy_local
            .reshape((b.batch_size * b.max_policy_moves, MOVE_LOCAL_SIZE))?
            .matmul(&self.policy_local.reshape((MOVE_LOCAL_SIZE, 1))?)?
            .reshape((b.batch_size, b.max_policy_moves))?;
        let repetition = hidden
            .matmul(
                &self
                    .policy_repetition_hidden
                    .reshape((self.arch.hidden_size, 1))?,
            )?
            .broadcast_add(&self.policy_repetition_bias)?;
        let policy_logits = ((((((policy_context + sparse)? + to)? - from)? - capture)? + local)?
            + b.policy_repetition.broadcast_mul(&repetition)?)?
        .add(&b.policy_mask)?;
        let moves_left = hidden
            .matmul(&self.moves_left_output.t()?)?
            .broadcast_add(&self.moves_left_bias)?
            .affine(1., 1.)?
            .relu()?;
        Ok(ForwardOutput {
            moves_left,
            value_logits,
            policy_logits,
        })
    }
}
pub(super) struct ForwardOutput {
    pub moves_left: Tensor,
    pub value_logits: Tensor,
    pub policy_logits: Tensor,
}
pub(super) struct BatchTensors {
    pub batch_size: usize,
    pub max_policy_moves: usize,
    pub value_buckets: Tensor,
    pub feature_items: Tensor,
    pub input_factors: Tensor,
    pub king_indices: Tensor,
    pub relation_indices: Tensor,
    pub motif_indices: Tensor,
    pub move_indices: Tensor,
    pub policy_indices: Tensor,
    pub policy_from: Tensor,
    pub policy_to: Tensor,
    pub policy_capture: Tensor,
    pub policy_local: Tensor,
    pub policy_targets: Tensor,
    pub policy_mask: Tensor,
    pub policy_repetition: Tensor,
    pub value_wdl: Tensor,
    pub values: Tensor,
    pub rule_context: Tensor,
    pub check_context: Tensor,
    pub policy_weights: Tensor,
    pub value_weights: Tensor,
    pub moves_left_targets: Tensor,
    pub moves_left_weights: Tensor,
    pub value_phase_masks: Tensor,
    pub value_source_phase_masks: Tensor,
}
impl BatchTensors {
    pub(super) fn from_packed(p: PackedBatch, device: &Device) -> CandleResult<Self> {
        let batch_size = p.batch_size;
        let max_policy_moves = p.max_policy_moves;
        Ok(Self {
            batch_size,
            max_policy_moves,
            value_buckets: Tensor::from_vec(p.value_buckets, batch_size, device)?,
            feature_items: Tensor::from_vec(p.feature_items, (batch_size, p.max_features), device)?,
            input_factors: Tensor::from_vec(
                p.input_factors,
                (batch_size, INPUT_FACTOR_ROWS),
                device,
            )?,
            king_indices: Tensor::from_vec(p.king_indices, (batch_size, p.max_king), device)?,
            relation_indices: Tensor::from_vec(
                p.relation_indices,
                (batch_size, p.max_relations),
                device,
            )?,
            motif_indices: Tensor::from_vec(p.motif_indices, (batch_size, p.max_motifs), device)?,
            move_indices: Tensor::from_vec(p.move_indices, (batch_size, max_policy_moves), device)?,
            policy_indices: Tensor::from_vec(
                p.policy_indices,
                (batch_size * max_policy_moves, p.max_policy_terms),
                device,
            )?,
            policy_from: Tensor::from_vec(p.policy_from, (batch_size, max_policy_moves), device)?,
            policy_to: Tensor::from_vec(p.policy_to, (batch_size, max_policy_moves), device)?,
            policy_capture: Tensor::from_vec(
                p.policy_capture,
                (batch_size, max_policy_moves),
                device,
            )?,
            policy_local: Tensor::from_vec(
                p.policy_local,
                (batch_size, max_policy_moves, MOVE_LOCAL_SIZE),
                device,
            )?,
            policy_targets: Tensor::from_vec(
                p.policy_targets,
                (batch_size, max_policy_moves),
                device,
            )?,
            policy_mask: Tensor::from_vec(p.policy_mask, (batch_size, max_policy_moves), device)?,
            policy_repetition: Tensor::from_vec(
                p.policy_repetition,
                (batch_size, max_policy_moves),
                device,
            )?,
            value_wdl: Tensor::from_vec(p.value_wdl, (batch_size, 3), device)?,
            values: Tensor::from_vec(p.values, batch_size, device)?,
            rule_context: Tensor::from_vec(
                p.rule_context,
                (batch_size, RULE_CONTEXT_SIZE),
                device,
            )?,
            check_context: Tensor::from_vec(
                p.check_context,
                (batch_size, CHECK_CONTEXT_SIZE),
                device,
            )?,
            policy_weights: Tensor::from_vec(p.policy_weights, batch_size, device)?,
            value_weights: Tensor::from_vec(p.value_weights, batch_size, device)?,
            moves_left_targets: Tensor::from_vec(p.moves_left_targets, batch_size, device)?,
            moves_left_weights: Tensor::from_vec(p.moves_left_weights, batch_size, device)?,
            value_phase_masks: Tensor::from_vec(p.value_phase_masks, (batch_size, 3), device)?,
            value_source_phase_masks: Tensor::from_vec(
                p.value_source_phase_masks,
                (batch_size, 9),
                device,
            )?,
        })
    }
}
