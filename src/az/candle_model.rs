use candle_core::{DType, Device, Result as CandleResult, Tensor, Var};

use super::arch::{POSITION_CONTEXT_SIZE, az_weight_tensors};
use super::{
    AzNnue, AzNnueArch, DENSE_MOVE_SPACE, POLICY_ACCUMULATOR_RANK, POLICY_MOVE_CONTEXT_SIZE,
    POLICY_SPARSE_TABLE_SIZE, POLICY_SPARSE_TERMS, POLICY_TACTICAL_SIZE, POLICY_TACTICAL_TERMS,
    POLICY_THREAT_CONTEXT_SIZE, VALUE_BUCKETS, VALUE_HEAD_SIZE, VALUE_KING_PIECE_RANK,
    VALUE_KING_PIECE_VOCAB, VALUE_THREAT_RANK, VALUE_THREAT_VOCAB, WDL_HEAD_SIZE,
    dataloader::PackedBatch,
    fused_policy::fused_policy,
    fused_sparse_policy::sparse_policy,
    sparse_pool::{PADDING_ITEM, sparse_pool},
};
use crate::az::nnue::AZ_NNUE_INPUT_SIZE;

const RMS_NORM_EPS: f64 = 1.0e-6;

#[derive(Debug)]
pub(super) struct AzCandleModel {
    arch: AzNnueArch,
    input_hidden: Var,
    context_hidden: Var,
    hidden_bias: Var,
    shared_hidden: Var,
    shared_bias: Var,
    value_head_hidden: Var,
    value_head_bias: Var,
    value_king_piece_hidden: Var,
    value_king_piece_projection: Var,
    value_head_output: Var,
    moves_left_output: Var,
    moves_left_bias: Var,
    threat_embedding: Var,
    value_threat_output: Var,
    policy_threat_context: Var,
    policy_context_hidden: Var,
    policy_move_context: Var,
    policy_accumulator_hidden: Var,
    policy_accumulator_move: Var,
    policy_sparse_table: Var,
    policy_tactical: Var,
    policy_repetition_hidden: Var,
    policy_repetition_bias: Var,
}

/// 两个输出头消费同一份局面编码，策略头自行投影线性棋子表示。
struct SharedEncoding {
    hidden: Tensor,
    board: Tensor,
    threat: Tensor,
}

impl AzCandleModel {
    pub(super) fn forward(&self, batch: &BatchTensors) -> CandleResult<ForwardOutput> {
        let encoded = self.encode_shared(batch)?;
        let value_logits = self.value_head(batch, &encoded)?;
        let policy_logits = self.policy_head(batch, &encoded)?;
        let moves_left = encoded
            .hidden
            .matmul(&self.moves_left_output.t()?)?
            .broadcast_add(&self.moves_left_bias)?
            .affine(1.0, 1.0)?
            .relu()?;
        Ok(ForwardOutput {
            moves_left,
            value_logits,
            policy_logits,
        })
    }

    fn encode_shared(&self, batch: &BatchTensors) -> CandleResult<SharedEncoding> {
        let hidden_size = self.arch.hidden_size;
        let board_pre = sparse_pool(self.input_hidden.as_tensor(), &batch.feature_items)?
            .broadcast_add(&self.hidden_bias)?
            .reshape((batch.batch_size, 2 * hidden_size))?;
        let board = board_pre.narrow(1, 0, hidden_size)?.contiguous()?;
        let sparse_hidden = (board_pre + batch.context.matmul(&self.context_hidden)?)?.relu()?;
        let rms = sparse_hidden
            .sqr()?
            .mean_keepdim(1)?
            .affine(1.0, RMS_NORM_EPS)?
            .sqrt()?;
        let hidden = sparse_hidden
            .broadcast_div(&rms)?
            .matmul(&self.shared_hidden.t()?)?
            .broadcast_add(&self.shared_bias)?
            .relu()?;
        let threat = sparse_pool(
            self.threat_embedding.as_tensor(),
            &batch.value_threat_indices,
        )?
        .broadcast_mul(&batch.value_threat_scales)?;
        let threat = Tensor::cat(&[&threat, &threat.sqr()?], 1)?;
        Ok(SharedEncoding {
            hidden,
            board,
            threat,
        })
    }

    fn value_head(&self, batch: &BatchTensors, encoded: &SharedEncoding) -> CandleResult<Tensor> {
        let king_piece = sparse_pool(
            self.value_king_piece_hidden.as_tensor(),
            &batch.value_king_piece_indices,
        )?
        .matmul(self.value_king_piece_projection.as_tensor())?
        .broadcast_mul(&batch.value_king_piece_scales)?;
        let value = encoded
            .hidden
            .matmul(&self.value_head_hidden.t()?)?
            .broadcast_add(&self.value_head_bias)?
            .add(&king_piece)?
            .relu()?;
        let selected = self
            .value_head_output
            .reshape((VALUE_BUCKETS, WDL_HEAD_SIZE * VALUE_HEAD_SIZE))?
            .index_select(&batch.value_buckets, 0)?
            .reshape((batch.batch_size, WDL_HEAD_SIZE, VALUE_HEAD_SIZE))?;
        let logits = selected.matmul(&value.unsqueeze(2)?)?.squeeze(2)?;
        logits + encoded.threat.matmul(&self.value_threat_output.t()?)?
    }

    fn policy_head(&self, batch: &BatchTensors, encoded: &SharedEncoding) -> CandleResult<Tensor> {
        let context = (encoded.hidden.matmul(&self.policy_context_hidden.t()?)?
            + encoded.threat.matmul(&self.policy_threat_context.t()?)?)?;
        let linear = encoded.board.matmul(&self.policy_accumulator_hidden.t()?)?;
        let context = Tensor::cat(&[&context, &linear], 1)?;
        let moves = Tensor::cat(
            &[
                self.policy_move_context.as_tensor(),
                self.policy_accumulator_move.as_tensor(),
            ],
            1,
        )?
        .flatten_all()?;
        let dynamic = fused_policy(&moves, &context, &batch.policy_items)?;
        let repetition = encoded
            .hidden
            .matmul(
                &self
                    .policy_repetition_hidden
                    .as_tensor()
                    .reshape((self.arch.hidden_size, 1))?,
            )?
            .broadcast_add(&self.policy_repetition_bias)?;
        (dynamic + self.policy_prior(batch)?)?
            + batch.policy_repetition.broadcast_mul(&repetition)?
    }

    /// 几何、将位、吃子和战术表共同组成候选走法先验。
    fn policy_prior(&self, batch: &BatchTensors) -> CandleResult<Tensor> {
        let sparse = sparse_policy(
            self.policy_sparse_table.as_tensor(),
            &batch.policy_sparse_indices,
        )?;
        let tactical_table = Tensor::cat(
            &[
                self.policy_tactical.as_tensor(),
                &Tensor::zeros(1, DType::F32, self.input_hidden.device())?,
            ],
            0,
        )?;
        sparse + sparse_policy(&tactical_table, &batch.policy_tactical_indices)?
    }
}

pub(super) struct ForwardOutput {
    /// 已除以 MOVES_LEFT_SCALE 的半回合数。
    pub(super) moves_left: Tensor,
    pub(super) value_logits: Tensor,
    pub(super) policy_logits: Tensor,
}

pub(super) struct MovesLeftPairs {
    pub(super) from: Tensor,
    pub(super) to: Tensor,
    pub(super) weights: Tensor,
    pub(super) total_weight: f64,
}

pub(super) struct BatchTensors {
    pub(super) batch_size: usize,
    pub(super) value_buckets: Tensor,
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
    pub(super) context: Tensor,
    pub(super) policy_weights: Tensor,
    pub(super) value_weights: Tensor,
    pub(super) moves_left_targets: Tensor,
    pub(super) moves_left_weights: Tensor,
    pub(super) moves_left_pairs: Option<MovesLeftPairs>,
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
        let pair_count = packed.moves_left_pair_from.len();
        let moves_left_pairs = if pair_count == 0 {
            None
        } else {
            let total_weight = packed
                .moves_left_pair_weights
                .iter()
                .map(|&w| f64::from(w))
                .sum();
            Some(MovesLeftPairs {
                from: Tensor::from_vec(packed.moves_left_pair_from, pair_count, device)?,
                to: Tensor::from_vec(packed.moves_left_pair_to, pair_count, device)?,
                weights: Tensor::from_vec(packed.moves_left_pair_weights, pair_count, device)?,
                total_weight,
            })
        };
        Ok(Self {
            batch_size,
            moves_left_pairs,
            value_buckets: Tensor::from_vec(packed.value_buckets, batch_size, device)?,
            feature_items: Tensor::from_vec(
                packed.feature_items,
                (batch_size * 2, max_features),
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
                (batch_size, max_policy_moves, POLICY_SPARSE_TERMS),
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
            context: Tensor::from_vec(packed.context, (batch_size, POSITION_CONTEXT_SIZE), device)?,
            policy_weights: Tensor::from_vec(packed.policy_weights, batch_size, device)?,
            value_weights: Tensor::from_vec(packed.value_weights, batch_size, device)?,
            moves_left_targets: Tensor::from_vec(packed.moves_left_targets, batch_size, device)?,
            moves_left_weights: Tensor::from_vec(packed.moves_left_weights, batch_size, device)?,
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
            context_hidden: var_from_slice(
                &model.context_hidden,
                (POSITION_CONTEXT_SIZE, 2 * hidden),
                device,
            )?,
            hidden_bias: var_from_slice(&model.hidden_bias, hidden, device)?,
            shared_hidden: var_from_slice(&model.shared_hidden, (hidden, 2 * hidden), device)?,
            shared_bias: var_from_slice(&model.shared_bias, hidden, device)?,
            value_head_hidden: var_from_slice(
                &model.value_head_hidden,
                (VALUE_HEAD_SIZE, hidden),
                device,
            )?,
            value_head_bias: var_from_slice(&model.value_head_bias, VALUE_HEAD_SIZE, device)?,
            value_king_piece_hidden: var_from_slice(
                &model.value_king_piece_hidden,
                (VALUE_KING_PIECE_VOCAB, VALUE_KING_PIECE_RANK),
                device,
            )?,
            value_king_piece_projection: var_from_slice(
                &model.value_king_piece_projection,
                (VALUE_KING_PIECE_RANK, VALUE_HEAD_SIZE),
                device,
            )?,
            value_head_output: var_from_slice(
                &model.value_head_output,
                (VALUE_BUCKETS, WDL_HEAD_SIZE, VALUE_HEAD_SIZE),
                device,
            )?,
            moves_left_output: var_from_slice(&model.moves_left_output, (1, hidden), device)?,
            moves_left_bias: var_from_slice(&model.moves_left_bias, 1, device)?,
            threat_embedding: var_from_slice(
                &model.threat_embedding,
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
            policy_tactical: var_from_slice(&model.policy_tactical, POLICY_TACTICAL_SIZE, device)?,
            policy_repetition_hidden: var_from_slice(
                &model.policy_repetition_hidden,
                hidden,
                device,
            )?,
            policy_repetition_bias: var_from_slice(&model.policy_repetition_bias, 1, device)?,
        })
    }

    #[cfg(test)]
    pub(super) fn all_vars(&self) -> Vec<Var> {
        self.all_vars_with_decay().0
    }

    /// 查表参数和偏置不衰减；分组与张量形状共用结构清单。
    pub(super) fn all_vars_with_decay(&self) -> (Vec<Var>, Vec<bool>) {
        let mut vars = Vec::new();
        let mut decay = Vec::new();
        macro_rules! collect {
            ($field:ident, [$($dim:expr),+], $decay:expr) => {
                vars.push(self.$field.clone());
                decay.push($decay);
            };
        }
        az_weight_tensors!(collect, self.arch.hidden_size);
        (vars, decay)
    }

    pub(super) fn copy_to_model(&self, model: &mut AzNnue) -> CandleResult<()> {
        macro_rules! copy {
            ($field:ident, [$($dim:expr),+], $decay:expr) => {
                copy_var(&self.$field, &mut model.$field)?;
            };
        }
        az_weight_tensors!(copy, self.arch.hidden_size);
        model.rebuild_moves_left();
        model.rebuild_threat();
        model.rebuild_context();
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
            policy_move_pieces, policy_sparse_capture_index, policy_sparse_main_index,
        },
        xiangqi::Position,
    };

    #[test]
    fn king_piece_factorization_initialization_trains_both_factors() {
        king_piece_factorization_training_matches_cpu(&Device::Cpu);
    }

    #[cfg(feature = "slow-tests")]
    #[test]
    fn king_piece_factorization_cuda_training_matches_cpu() {
        let device = crate::az::cuda_test_device::shared_cuda_device()
            .expect("CUDA is required for this test");
        king_piece_factorization_training_matches_cpu(device);
    }

    fn king_piece_factorization_training_matches_cpu(device: &Device) {
        let position = Position::startpos();
        let moves = position.legal_moves();
        let sample = AzTrainingSample {
            features: extract_sparse_features_az(&position),
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            history_features: crate::az::history_features(&position, &[]),
            move_indices: moves.iter().map(|&mv| dense_move_index(mv)).collect(),
            repetition_flags: vec![0; moves.len()],
            policy: vec![1.0 / moves.len() as f32; moves.len()],
            value_wdl: [0.0, 1.0, 0.0],
            root_search_wdl: [0.0, 1.0, 0.0],
            value: 0.0,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            moves_left: 0.0,
            moves_left_weight: 0.0,
            search_simulations: 0,
            meta: Default::default(),
        };
        let mut model = AzNnue::random(96, 20261008);
        assert_eq!(
            model.value_king_piece_hidden.len(),
            VALUE_KING_PIECE_VOCAB * VALUE_KING_PIECE_RANK
        );
        assert!(model.value_king_piece_hidden.iter().all(|&x| x == 0.0));
        for channel in 0..VALUE_KING_PIECE_RANK {
            let row = &model.value_king_piece_projection
                [channel * VALUE_HEAD_SIZE..(channel + 1) * VALUE_HEAD_SIZE];
            assert_eq!(row[channel], 1.0);
            assert_eq!(row.iter().filter(|&&x| x != 0.0).count(), 1);
        }
        // 保持每个价值方向的ReLU开放，专门核验两个因子的梯度启动。
        model.value_head_hidden.fill(0.0);
        model.value_head_bias.fill(1.0);
        let initial_projection = model.value_king_piece_projection.clone();
        let candle = AzCandleModel::from_model(&model, device).unwrap();
        let (all_vars, decay) = candle.all_vars_with_decay();
        assert_eq!(all_vars.len(), decay.len());
        let index = all_vars
            .iter()
            .position(|v| v.id() == candle.value_king_piece_projection.id())
            .unwrap();
        assert!(decay[index]);
        let batch = BatchTensors::from_packed(
            PackedBatch::from_indices(std::slice::from_ref(&sample), &[0]),
            device,
        )
        .unwrap();
        let mut optimizer = crate::az::adamw::AzAdamW::new(all_vars, decay, 0.001).unwrap();
        let mut live_channels = [false; VALUE_KING_PIECE_RANK];
        let mut projection_live = false;
        for step in 0..6 {
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
            let output_gradient = gradients
                .get(&candle.value_head_output)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let bucket = crate::az::value_material_bucket(&position, position.side_to_move());
            for (index, row) in output_gradient
                .chunks_exact(WDL_HEAD_SIZE * VALUE_HEAD_SIZE)
                .enumerate()
            {
                if index == bucket {
                    assert!(row.iter().any(|&g| g != 0.0));
                } else {
                    assert!(
                        row.iter().all(|&g| g == 0.0),
                        "unselected bucket {index} received gradients"
                    );
                }
            }
            let table_gradient = gradients
                .get(&candle.value_king_piece_hidden)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert!(table_gradient.iter().all(|x| x.is_finite()));
            for (channel, live) in live_channels.iter_mut().enumerate() {
                *live |= table_gradient
                    .iter()
                    .skip(channel)
                    .step_by(VALUE_KING_PIECE_RANK)
                    .any(|x| x.abs() > 1e-8);
            }
            let projection_gradient = gradients
                .get(&candle.value_king_piece_projection)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert!(projection_gradient.iter().all(|x| x.is_finite()));
            if step == 0 {
                assert!(projection_gradient.iter().all(|&x| x == 0.0));
            }
            projection_live |= projection_gradient.iter().any(|x| x.abs() > 1e-8);
            optimizer.step(&gradients).unwrap();
        }
        assert!(live_channels.into_iter().all(|live| live));
        assert!(projection_live);
        candle.copy_to_model(&mut model).unwrap();
        assert!(model.value_king_piece_hidden.iter().any(|x| x.abs() > 1e-6));
        assert!(
            model
                .value_king_piece_projection
                .iter()
                .zip(initial_projection)
                .any(|(&x, y)| (x - y).abs() > 1e-6)
        );
        let wdl = candle_nn::ops::softmax(&candle.forward(&batch).unwrap().value_logits, 1)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let (cpu, _) = crate::az::outputs_for_training_sample(&model, &sample).unwrap();
        assert!(cpu.iter().zip(&wdl[0]).all(|(&x, &y)| (x - y).abs() < 2e-5));
    }

    fn history_context_training_matches(device: &Device) {
        let mut position = Position::startpos();
        let mut history = position.initial_rule_history();
        for _ in 0..2 {
            let mv = position.legal_moves()[0];
            history.push(position.rule_history_entry_after_move(mv));
            position.make_move(mv);
        }
        let moves = position.legal_moves();
        let sample = AzTrainingSample {
            features: extract_sparse_features_az(&position),
            rule_context: [0.0; RULE_CONTEXT_SIZE],
            history_features: crate::az::history_features(&position, &history),
            move_indices: moves.iter().map(|&mv| dense_move_index(mv)).collect(),
            repetition_flags: vec![0; moves.len()],
            policy: vec![1.0 / moves.len() as f32; moves.len()],
            value_wdl: [1.0, 0.0, 0.0],
            root_search_wdl: [1.0, 0.0, 0.0],
            value: 1.0,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            moves_left: 0.0,
            moves_left_weight: 0.0,
            search_simulations: 0,
            meta: Default::default(),
        };
        assert!(sample.history_features[48..].iter().any(|x| *x != 0.0));
        let mut model = AzNnue::random(16, 7);
        model.value_head_bias.fill(1.0);
        for (i, w) in model.value_head_output.iter_mut().enumerate() {
            *w = (i % 17) as f32 * 0.005 - 0.04;
        }
        for (i, w) in model.policy_move_context.iter_mut().enumerate() {
            *w = (i % 23) as f32 * 0.003 - 0.03;
        }
        let candle = AzCandleModel::from_model(&model, device).unwrap();
        let (vars, decay) = candle.all_vars_with_decay();
        let context_index = vars
            .iter()
            .position(|var| var.id() == candle.context_hidden.id())
            .unwrap();
        assert!(decay[context_index]);
        let batch = BatchTensors::from_packed(
            PackedBatch::from_indices(std::slice::from_ref(&sample), &[0]),
            device,
        )
        .unwrap();
        let output = candle.forward(&batch).unwrap();
        let value_loss = candle_nn::ops::log_softmax(&output.value_logits, 1)
            .unwrap()
            .mul(&batch.value_wdl)
            .unwrap()
            .sum_all()
            .unwrap()
            .neg()
            .unwrap();
        let policy_loss = output.policy_logits.sqr().unwrap().sum_all().unwrap();
        let value_grads = value_loss.backward().unwrap();
        let policy_grads = policy_loss.backward().unwrap();
        let past_offset = (crate::az::HISTORY_CONTEXT_OFFSET + 48) * 2 * model.hidden_size;
        for grads in [&value_grads, &policy_grads] {
            let gradient = grads
                .get(&candle.context_hidden)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert!(gradient.iter().all(|x| x.is_finite()));
            assert!(gradient[past_offset..].iter().any(|x| x.abs() > 1e-8));
        }
        let mut optimizer =
            crate::az::adamw::AzAdamW::new(vec![candle.context_hidden.clone()], vec![true], 0.001)
                .unwrap();
        optimizer.step(&value_grads).unwrap();
        candle.copy_to_model(&mut model).unwrap();
        let forward = candle.forward(&batch).unwrap();
        let wdl = candle_nn::ops::softmax(&forward.value_logits, 1)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let logits = forward.policy_logits.to_vec2::<f32>().unwrap();
        let (cpu_wdl, cpu_logits) =
            crate::az::outputs_for_training_sample(&model, &sample).unwrap();
        assert!(
            cpu_wdl
                .iter()
                .zip(&wdl[0])
                .all(|(a, b)| (a - b).abs() < 1e-5)
        );
        assert!(
            cpu_logits
                .iter()
                .zip(&logits[0])
                .all(|(a, b)| (a - b).abs() < 1e-5)
        );
    }

    #[test]
    fn history_context_training_matches_cpu() {
        history_context_training_matches(&Device::Cpu);
    }

    #[cfg(feature = "slow-tests")]
    #[test]
    fn history_context_training_matches_cuda() {
        let device =
            crate::az::cuda_test_device::shared_cuda_device().expect("CUDA device required");
        history_context_training_matches(device);
    }

    fn assert_shared_gradients(
        model: &AzCandleModel,
        gradients: &candle_core::backprop::GradStore,
    ) {
        for var in [&model.shared_hidden, &model.shared_bias] {
            let values = gradients
                .get(var)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert!(values.iter().all(|g| g.is_finite()));
            assert!(values.iter().any(|g| g.abs() > 1e-8));
            if var.id() == model.shared_hidden.id() {
                let width = model.arch.hidden_size;
                for perspective in 0..2 {
                    assert!(
                        values.chunks_exact(2 * width).any(|row| {
                            row[perspective * width..(perspective + 1) * width]
                                .iter()
                                .any(|g| g.abs() > 1e-8)
                        }),
                        "视角 {perspective} 没有参与主干梯度计算"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "需要本地已训练检查点、Px0数据及CUDA；只核对前向"]
    fn trained_cuda_forward_matches_search_cpu() {
        let path = std::env::var("CHINESEAI_AUDIT_MODEL")
            .unwrap_or_else(|_| "best-v42.safetensors".into());
        let model = AzNnue::load(path).unwrap();
        let data_path =
            std::env::var("CHINESEAI_AUDIT_DATA").unwrap_or_else(|_| "data/data.bin".into());
        // 前16个有效训练对局足够检查前向，避免扫描大型数据集。
        let training =
            crate::az::px0_data::load_training(std::path::Path::new(&data_path), 16).unwrap();
        let stride = (training.len() / 128).max(1);
        let samples = training
            .into_iter()
            .step_by(stride)
            .take(128)
            .collect::<Vec<_>>();
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
            let cpu = model.evaluate_with_scratch_output_with_repetition_and_history_features(
                &position,
                &moves,
                &sample.repetition_flags,
                &sample.rule_context,
                &sample.history_features,
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
    fn candle_and_cpu_policy_heads_and_gradients_match() {
        policy_heads_and_gradients_match(&Device::Cpu);
    }

    #[cfg(feature = "slow-tests")]
    #[test]
    fn compact_feature_heads_and_gradients_match_cuda() {
        let device = crate::az::cuda_test_device::shared_cuda_device()
            .expect("CUDA is required for this test");
        policy_heads_and_gradients_match(device);
    }

    fn policy_heads_and_gradients_match(device: &Device) {
        let position =
            Position::from_fen("1rbakab1r/9/4c3n/p3p3P/2p6/1C2p1cN1/P1P6/4B2C1/4A4/1RBAK3R w")
                .unwrap();
        let moves = position.legal_moves();
        let mut model = AzNnue::random(32, 20260730);
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
        for (index, weight) in model.context_hidden.iter_mut().enumerate() {
            *weight = (index % 17) as f32 * 0.01 - 0.08;
        }
        model.rebuild_context();
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
            let (moved_piece, captured_piece) = policy_move_pieces(&position, side, mv).unwrap();
            let main = policy_sparse_main_index(move_index, moved_piece, buckets.0, buckets.1);
            let capture = policy_sparse_capture_index(move_index, captured_piece);
            model.policy_sparse_table[main] = (index as f32 % 101.0) * 0.001;
            model.policy_sparse_table[capture] = -((index as f32 % 53.0) * 0.001);
        }
        model.rebuild_policy_cache();
        model.rebuild_policy_tactical();
        model.rebuild_moves_left();
        model.rebuild_threat();

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
            history_features: crate::az::history_features(&position, &[]),
            move_indices: moves.iter().map(|&mv| dense_move_index(mv)).collect(),
            policy: vec![1.0; moves.len()],
            value_wdl: [0.0, 1.0, 0.0],
            root_search_wdl: [0.0, 1.0, 0.0],
            value: 0.0,
            side_sign: 1.0,
            policy_weight: 1.0,
            value_weight: 1.0,
            moves_left: 0.0,
            moves_left_weight: 0.0,
            search_simulations: 1,
            meta: AzSampleMeta::default(),
        };
        let (sample_wdl, sample_logits) =
            crate::az::outputs_for_training_sample(&model, &sample).unwrap();
        assert_eq!(sample_wdl, cpu_output.value_wdl);
        assert_eq!(sample_logits, cpu.logits);
        let packed = PackedBatch::from_indices(&[sample], &[0]);
        let batch = BatchTensors::from_packed(packed, device).unwrap();
        let candle = AzCandleModel::from_model(&model, device).unwrap();
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
        let threat_gradient = gradients
            .get(&candle.threat_embedding)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let ray_start = crate::az::VALUE_THREAT_PAIR_VOCAB * VALUE_THREAT_RANK;
        let cannon_start = ray_start + crate::az::VALUE_RAY_VOCAB * VALUE_THREAT_RANK;
        assert!(
            threat_gradient[ray_start..cannon_start]
                .iter()
                .any(|&g| g != 0.0)
        );
        assert!(threat_gradient[cannon_start..].iter().any(|&g| g != 0.0));

        assert_eq!(legal[0].len(), cpu.logits.len());
        for (candle_logit, cpu_logit) in legal[0].iter().zip(&cpu.logits) {
            assert!(
                (candle_logit - cpu_logit).abs() < 2.0e-3,
                "candle={candle_logit} cpu={cpu_logit}"
            );
        }

        let mut gradient_model = AzNnue::random(32, 20260731);
        gradient_model.policy_move_context.fill(0.01);
        for row in gradient_model
            .policy_move_context
            .chunks_exact_mut(POLICY_MOVE_CONTEXT_SIZE)
        {
            row[16..].fill(0.0);
        }
        let gradient_candle = AzCandleModel::from_model(&gradient_model, device).unwrap();
        let gradient_forward = gradient_candle.forward(&batch).unwrap();
        let gradients = gradient_forward
            .policy_logits
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        assert_shared_gradients(&gradient_candle, &gradients);
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
        assert!(
            move_context_gradient
                .chunks_exact(POLICY_MOVE_CONTEXT_SIZE)
                .any(|row| row[16..].iter().any(|gradient| gradient.abs() > 1.0e-8)),
            "扩宽通道即使初始输出为零，也必须能收到梯度"
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
        model.context_hidden.fill(0.125);
        model.rebuild_context();
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
            context_hidden,
            hidden_bias,
            shared_hidden,
            shared_bias,
            value_head_hidden,
            value_head_bias,
            value_king_piece_hidden,
            value_king_piece_projection,
            value_head_output,
            policy_threat_context,
            policy_context_hidden,
            policy_move_context,
            policy_accumulator_hidden,
            policy_accumulator_move,
            policy_sparse_table,
            policy_tactical,
        );
        assert!(back.context_active);
    }

    fn position_context_training_matches_inference(device: &Device) {
        use crate::az::{adamw::AzAdamW, nnue::canonical_move};

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
                    history_features: crate::az::history_features(position, &[]),
                    value_wdl: [1.0, 0.0, 0.0],
                    root_search_wdl: [1.0, 0.0, 0.0],
                    value: 1.0,
                    side_sign: 1.0,
                    policy_weight: 1.0,
                    value_weight: 1.0,
                    moves_left: 0.0,
                    moves_left_weight: 0.0,
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
            let context = crate::az::context::check_context_features(
                position,
                &moves,
                &checks,
                position.attacked_squares_masks(),
            );
            assert_eq!(
                &packed.context[row * POSITION_CONTEXT_SIZE + crate::az::CHECK_CONTEXT_OFFSET
                    ..row * POSITION_CONTEXT_SIZE + crate::az::HISTORY_CONTEXT_OFFSET],
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
        model.moves_left_output.fill(0.02);
        model.moves_left_bias[0] = 0.1;
        model.rebuild_moves_left();
        let candle = AzCandleModel::from_model(&model, device).unwrap();
        let forward = candle.forward(&batch).unwrap();
        assert_shared_gradients(
            &candle,
            &forward.moves_left.sum_all().unwrap().backward().unwrap(),
        );
        let loss = candle_nn::ops::log_softmax(&forward.value_logits, 1)
            .unwrap()
            .mul(&batch.value_wdl)
            .unwrap()
            .sum_all()
            .unwrap()
            .neg()
            .unwrap();
        let gradients = loss.backward().unwrap();
        assert_shared_gradients(&candle, &gradients);
        let gradient = gradients
            .get(&candle.context_hidden)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(gradient.iter().all(|g| g.is_finite()));
        assert!(gradient.iter().any(|g| g.abs() > 1e-8));
        let (vars, decay) = candle.all_vars_with_decay();
        let mut optimizer = AzAdamW::new(vars, decay, 4e-4).unwrap();
        optimizer.step(&gradients).unwrap();
        candle.copy_to_model(&mut model).unwrap();
        assert!(model.context_hidden.iter().any(|&w| w != 0.0));
        assert!(model.context_active);
        let forward = candle.forward(&batch).unwrap();
        let wdl = candle_nn::ops::softmax(&forward.value_logits, 1)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let logits = forward.policy_logits.to_vec2::<f32>().unwrap();
        let moves_left = forward.moves_left.to_vec2::<f32>().unwrap();
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
            assert!(
                (cpu.moves_left - moves_left[row][0] * crate::az::MOVES_LEFT_SCALE).abs() < 1e-3
            );
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
    fn position_context_training_updates_zero_weights_and_matches_cpu() {
        position_context_training_matches_inference(&Device::Cpu);
    }

    #[cfg(feature = "slow-tests")]
    #[test]
    fn position_context_training_updates_zero_weights_and_matches_cuda() {
        let device =
            crate::az::cuda_test_device::shared_cuda_device().expect("CUDA device required");
        position_context_training_matches_inference(device);
    }
}
