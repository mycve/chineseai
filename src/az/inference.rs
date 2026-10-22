use super::{
    features::*,
    nnue::{AZ_NNUE_INPUT_SIZE, canonical_square, piece_absolute_feature_index},
    *,
};
use crate::infra::version::MODEL_FORMAT_VERSION;
use crate::xiangqi::{Color, Move, PieceKind, Position, color_index};
use candle_core::{DType, Device, Var};
use candle_nn::VarMap;
use std::{io, path::Path};

pub(crate) fn candle_io_error(err: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err.to_string())
}
pub(crate) fn insert_candle_var(
    varmap: &VarMap,
    name: &str,
    data: &[f32],
    shape: &[usize],
) -> io::Result<()> {
    let var = Var::from_slice(data, shape, &Device::Cpu).map_err(candle_io_error)?;
    varmap
        .data()
        .lock()
        .expect("varmap poisoned")
        .insert(name.to_owned(), var);
    Ok(())
}
pub(crate) fn load_candle_f32_tensor(
    tensors: &candle_core::safetensors::MmapedSafetensors,
    name: &str,
) -> io::Result<Vec<f32>> {
    let tensor = tensors.load(name, &Device::Cpu).map_err(candle_io_error)?;
    if tensor.dtype() != DType::F32 {
        return Err(candle_io_error(format!("{name}: expected f32")));
    }
    tensor
        .flatten_all()
        .and_then(|t| t.to_vec1())
        .map_err(candle_io_error)
}
macro_rules! az_weight_tensors {
    ($visit:ident, $h:expr) => {{
        let h = $h;
        $visit!(input_hidden, [AZ_NNUE_INPUT_SIZE, h]);
        $visit!(hidden_bias, [h]);
        $visit!(king_piece, [VALUE_KING_PIECE_VOCAB, VALUE_KING_PIECE_RANK]);
        $visit!(king_projection, [VALUE_KING_PIECE_RANK, h]);
        $visit!(relation_embedding, [RELATION_ROWS, RELATION_RANK]);
        $visit!(relation_projection, [RELATION_RANK, h]);
        $visit!(motif_embedding, [MOTIF_ROWS, MOTIF_RANK]);
        $visit!(motif_projection, [MOTIF_RANK, h]);
        $visit!(rule_context_hidden, [RULE_CONTEXT_SIZE, h]);
        $visit!(check_context_hidden, [CHECK_CONTEXT_SIZE, h]);
        $visit!(value_head_hidden, [VALUE_HEAD_SIZE, h]);
        $visit!(value_head_bias, [VALUE_HEAD_SIZE]);
        $visit!(
            value_head_output,
            [VALUE_BUCKETS, WDL_HEAD_SIZE, VALUE_HEAD_SIZE]
        );
        $visit!(moves_left_output, [1, h]);
        $visit!(moves_left_bias, [1]);
        $visit!(policy_move_bias, [DENSE_MOVE_SPACE]);
        $visit!(policy_piece_square, [AZ_NNUE_INPUT_SIZE]);
        $visit!(policy_context_hidden, [POLICY_MOVE_CONTEXT_SIZE, h]);
        $visit!(
            policy_move_context,
            [DENSE_MOVE_SPACE, POLICY_MOVE_CONTEXT_SIZE]
        );
        $visit!(policy_move_piece, [DENSE_MOVE_SPACE, 7]);
        $visit!(policy_move_king, [2, DENSE_MOVE_SPACE, 9]);
        $visit!(policy_move_capture, [DENSE_MOVE_SPACE, 8]);
        $visit!(policy_local, [MOVE_LOCAL_SIZE]);
        $visit!(policy_joint, [POLICY_JOINT_ROWS]);
        $visit!(policy_events, [POLICY_EVENT_ROWS]);
        $visit!(policy_repetition_hidden, [h]);
        $visit!(policy_repetition_bias, [1]);
    }};
}

pub(crate) fn value_material_bucket(position: &Position, perspective: Color) -> usize {
    let mut rooks = [0usize; 2];
    let mut minors = [0usize; 2];
    for square in position.occupied_squares() {
        let piece = position.piece_at(square).unwrap();
        let side = usize::from(piece.color != perspective);
        match piece.kind {
            PieceKind::Rook => rooks[side] += 1,
            PieceKind::Horse | PieceKind::Cannon => minors[side] += 1,
            _ => {}
        }
    }
    value_material_bucket_counts(rooks, minors)
}

pub(crate) fn value_material_bucket_counts(rooks: [usize; 2], minors: [usize; 2]) -> usize {
    let [us, them] = rooks.map(|n| n.min(2));
    let total = minors[0] + minors[1];
    if us == them {
        us * 8 + usize::from(minors[0] == minors[1]) * 4 + (total / 2).min(3)
    } else if us == 2 && them == 1 {
        24 + usize::from(total >= 4)
    } else if us == 1 && them == 2 {
        26 + usize::from(total >= 4)
    } else if us > 0 && them == 0 {
        28 + usize::from(total >= 4)
    } else {
        30 + usize::from(total >= 4)
    }
}

#[derive(Debug)]
pub struct AzNnue {
    pub hidden_size: usize,
    pub arch: AzNnueArch,
    pub input_hidden: Vec<f32>,
    pub hidden_bias: Vec<f32>,
    pub king_piece: Vec<f32>,
    pub king_projection: Vec<f32>,
    pub relation_embedding: Vec<f32>,
    pub relation_projection: Vec<f32>,
    pub motif_embedding: Vec<f32>,
    pub motif_projection: Vec<f32>,
    pub rule_context_hidden: Vec<f32>,
    pub check_context_hidden: Vec<f32>,
    pub value_head_hidden: Vec<f32>,
    pub value_head_bias: Vec<f32>,
    pub value_head_output: Vec<f32>,
    pub moves_left_output: Vec<f32>,
    pub moves_left_bias: Vec<f32>,
    pub policy_move_bias: Vec<f32>,
    pub policy_piece_square: Vec<f32>,
    pub policy_context_hidden: Vec<f32>,
    pub policy_move_context: Vec<f32>,
    pub policy_move_piece: Vec<f32>,
    pub policy_move_king: Vec<f32>,
    pub policy_move_capture: Vec<f32>,
    pub policy_local: Vec<f32>,
    pub policy_joint: Vec<f32>,
    pub policy_events: Vec<f32>,
    pub policy_repetition_hidden: Vec<f32>,
    pub policy_repetition_bias: Vec<f32>,
    pub moves_left_active: bool,
    pub moves_left_params: AzMovesLeftParams,
    pub mate_search_plies: usize,
    pub mate_search_nodes: usize,
    pub node_incremental: bool,
    pub(super) gpu_trainer: Option<Box<train_gpu::GpuTrainer>>,
}
impl Clone for AzNnue {
    fn clone(&self) -> Self {
        Self {
            hidden_size: self.hidden_size,
            arch: self.arch,
            input_hidden: self.input_hidden.clone(),
            hidden_bias: self.hidden_bias.clone(),
            king_piece: self.king_piece.clone(),
            king_projection: self.king_projection.clone(),
            relation_embedding: self.relation_embedding.clone(),
            relation_projection: self.relation_projection.clone(),
            motif_embedding: self.motif_embedding.clone(),
            motif_projection: self.motif_projection.clone(),
            rule_context_hidden: self.rule_context_hidden.clone(),
            check_context_hidden: self.check_context_hidden.clone(),
            value_head_hidden: self.value_head_hidden.clone(),
            value_head_bias: self.value_head_bias.clone(),
            value_head_output: self.value_head_output.clone(),
            moves_left_output: self.moves_left_output.clone(),
            moves_left_bias: self.moves_left_bias.clone(),
            policy_move_bias: self.policy_move_bias.clone(),
            policy_piece_square: self.policy_piece_square.clone(),
            policy_context_hidden: self.policy_context_hidden.clone(),
            policy_move_context: self.policy_move_context.clone(),
            policy_move_piece: self.policy_move_piece.clone(),
            policy_move_king: self.policy_move_king.clone(),
            policy_move_capture: self.policy_move_capture.clone(),
            policy_local: self.policy_local.clone(),
            policy_joint: self.policy_joint.clone(),
            policy_events: self.policy_events.clone(),
            policy_repetition_hidden: self.policy_repetition_hidden.clone(),
            policy_repetition_bias: self.policy_repetition_bias.clone(),
            moves_left_active: self.moves_left_active,
            moves_left_params: self.moves_left_params,
            mate_search_plies: self.mate_search_plies,
            mate_search_nodes: self.mate_search_nodes,
            node_incremental: self.node_incremental,
            gpu_trainer: None,
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct AzEvalOutput {
    pub value_wdl: [f32; 3],
    pub value: f32,
    pub moves_left: f32,
}

impl AzNnue {
    pub fn random(hidden_size: usize, seed: u64) -> Self {
        Self::random_with_arch(AzNnueArch::with_hidden_size(hidden_size), seed)
    }
    pub fn random_with_arch(arch: AzNnueArch, seed: u64) -> Self {
        arch.validate().expect("invalid architecture");
        let h = arch.hidden_size;
        let mut rng = SplitMix64::new(seed);
        let mut random = |len: usize, scale: f32| (0..len).map(|_| rng.weight(scale)).collect();
        Self {
            hidden_size: h,
            arch,
            input_hidden: random([AZ_NNUE_INPUT_SIZE, h].iter().product(), 0.02),
            hidden_bias: vec![0.; [h].iter().product()],
            king_piece: random(
                [VALUE_KING_PIECE_VOCAB, VALUE_KING_PIECE_RANK]
                    .iter()
                    .product(),
                0.02,
            ),
            king_projection: random([VALUE_KING_PIECE_RANK, h].iter().product(), 0.1),
            relation_embedding: random([RELATION_ROWS, RELATION_RANK].iter().product(), 0.02),
            relation_projection: random([RELATION_RANK, h].iter().product(), 0.1),
            motif_embedding: random([MOTIF_ROWS, MOTIF_RANK].iter().product(), 0.02),
            motif_projection: random([MOTIF_RANK, h].iter().product(), 0.1),
            rule_context_hidden: random([RULE_CONTEXT_SIZE, h].iter().product(), 0.1),
            check_context_hidden: random([CHECK_CONTEXT_SIZE, h].iter().product(), 0.1),
            value_head_hidden: random([VALUE_HEAD_SIZE, h].iter().product(), 0.1),
            value_head_bias: vec![0.; [VALUE_HEAD_SIZE].iter().product()],
            value_head_output: random(
                [VALUE_BUCKETS, WDL_HEAD_SIZE, VALUE_HEAD_SIZE]
                    .iter()
                    .product(),
                0.1,
            ),
            moves_left_output: vec![0.; h],
            moves_left_bias: vec![0.; [1].iter().product()],
            policy_move_bias: vec![0.; [DENSE_MOVE_SPACE].iter().product()],
            policy_piece_square: random([AZ_NNUE_INPUT_SIZE].iter().product(), 0.01),
            policy_context_hidden: random([POLICY_MOVE_CONTEXT_SIZE, h].iter().product(), 0.1),
            policy_move_context: random(
                [DENSE_MOVE_SPACE, POLICY_MOVE_CONTEXT_SIZE]
                    .iter()
                    .product(),
                0.01,
            ),
            policy_move_piece: random([DENSE_MOVE_SPACE, 7].iter().product(), 0.01),
            policy_move_king: random([2, DENSE_MOVE_SPACE, 9].iter().product(), 0.01),
            policy_move_capture: random([DENSE_MOVE_SPACE, 8].iter().product(), 0.01),
            policy_local: random([MOVE_LOCAL_SIZE].iter().product(), 0.01),
            policy_joint: random([POLICY_JOINT_ROWS].iter().product(), 0.01),
            policy_events: random([POLICY_EVENT_ROWS].iter().product(), 0.01),
            policy_repetition_hidden: random([h].iter().product(), 0.1),
            policy_repetition_bias: vec![0.; [1].iter().product()],
            moves_left_active: false,
            moves_left_params: AzMovesLeftParams::default(),
            mate_search_plies: 0,
            mate_search_nodes: 200_000,
            node_incremental: true,
            gpu_trainer: None,
        }
    }
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        self.validate()?;
        let map = VarMap::new();
        insert_candle_var(
            &map,
            "az_model_format_version",
            &[MODEL_FORMAT_VERSION],
            &[1],
        )?;
        let schema = feature_schema();
        insert_candle_var(&map, "feature_schema", &schema, &[schema.len()])?;
        macro_rules! save {
            ($field:ident,$shape:expr) => {
                insert_candle_var(&map, stringify!($field), &self.$field, &$shape)?;
            };
        }
        az_weight_tensors!(save, self.hidden_size);
        map.save(path).map_err(candle_io_error)
    }
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }
            .map_err(candle_io_error)?;
        if load_candle_f32_tensor(&tensors, "az_model_format_version")? != [MODEL_FORMAT_VERSION] {
            return Err(candle_io_error(
                "unsupported model format; retrain for tactical features",
            ));
        }
        if load_candle_f32_tensor(&tensors, "feature_schema")? != feature_schema() {
            return Err(candle_io_error("feature schema mismatch"));
        }
        let (_, h) = tensors
            .load("input_hidden", &Device::Cpu)
            .and_then(|t| t.dims2())
            .map_err(candle_io_error)?;
        let mut model = Self::random(h, 0);
        macro_rules! load {
            ($field:ident,$shape:expr) => {{
                let t = tensors
                    .load(stringify!($field), &Device::Cpu)
                    .map_err(candle_io_error)?;
                if t.dims() != &$shape {
                    return Err(candle_io_error(concat!(
                        "shape mismatch: ",
                        stringify!($field)
                    )));
                }
                model.$field = load_candle_f32_tensor(&tensors, stringify!($field))?;
            }};
        }
        az_weight_tensors!(load, h);
        model.rebuild_moves_left();
        model.validate()?;
        Ok(model)
    }
    pub(crate) fn validate(&self) -> io::Result<()> {
        self.arch.validate().map_err(candle_io_error)?;
        if self.hidden_size != self.arch.hidden_size {
            return Err(candle_io_error("hidden size mismatch"));
        }
        macro_rules! check {
            ($field:ident,$shape:expr) => {
                if self.$field.len() != $shape.iter().product::<usize>()
                    || self.$field.iter().any(|x| !x.is_finite())
                {
                    return Err(candle_io_error(concat!(
                        "invalid tensor ",
                        stringify!($field)
                    )));
                }
            };
        }
        az_weight_tensors!(check, self.hidden_size);
        Ok(())
    }
    pub fn save_training_state(
        &self,
        path: impl AsRef<Path>,
        next_update: usize,
    ) -> io::Result<bool> {
        let Some(trainer) = self.gpu_trainer.as_ref() else {
            return Ok(false);
        };
        trainer
            .save_state(path.as_ref(), next_update)
            .map_err(candle_io_error)?;
        Ok(true)
    }

    /// 优化器检查点同时记录更新序号；缺失独立进度文件时用它恢复。
    pub fn training_state_next_update(path: impl AsRef<Path>) -> io::Result<usize> {
        let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }
            .map_err(candle_io_error)?;
        let state = tensors
            .load("state", &Device::Cpu)
            .and_then(|state| state.to_vec1::<i64>())
            .map_err(candle_io_error)?;
        if load_candle_f32_tensor(&tensors, "feature_schema")? != feature_schema() {
            return Err(candle_io_error("optimizer feature schema mismatch"));
        }
        if state.len() != 4
            || state[0] != super::adamw::STATE_TAG
            || state[1] != MODEL_FORMAT_VERSION as i64
            || state[2] < 0
            || state[3] < 1
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid optimizer resume metadata",
            ));
        }
        usize::try_from(state[3]).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
    }

    pub fn restore_training_state(
        &mut self,
        path: impl AsRef<Path>,
        next_update: usize,
        lr: f32,
        optimizer: AzTrainOptimizer,
    ) -> io::Result<()> {
        let mut trainer =
            train_gpu::GpuTrainer::new(self, lr, optimizer).map_err(candle_io_error)?;
        trainer
            .restore_state(path.as_ref(), next_update)
            .map_err(candle_io_error)?;
        self.gpu_trainer = Some(Box::new(trainer));
        Ok(())
    }

    pub fn training_steps(&self) -> usize {
        self.gpu_trainer
            .as_ref()
            .map_or(0, |trainer| trainer.steps())
    }

    pub fn set_training_holdout(
        &mut self,
        samples: Vec<AzTrainingSample>,
        lr: f32,
        optimizer: AzTrainOptimizer,
    ) -> io::Result<()> {
        if self.gpu_trainer.is_none() {
            self.gpu_trainer = Some(Box::new(
                train_gpu::GpuTrainer::new(self, lr, optimizer).map_err(candle_io_error)?,
            ));
        }
        self.gpu_trainer.as_mut().unwrap().set_holdout(samples);
        Ok(())
    }

    pub fn take_training_checks(&mut self) -> Vec<AzHoldoutReport> {
        self.gpu_trainer
            .as_mut()
            .map_or_else(Vec::new, |trainer| trainer.take_checks())
    }

    pub fn last_training_learning_rate(&self) -> Option<f32> {
        self.gpu_trainer
            .as_ref()
            .map(|trainer| trainer.last_learning_rate())
    }

    pub(crate) fn input_embedding_linear_into_slice(&self, features: &[usize], output: &mut [f32]) {
        output.fill(0.);
        for &feature in features {
            add_scaled_row(output, &self.input_hidden, feature, 1.);
        }
    }
    pub(crate) fn feature_sums(&self, graph: &Graph, side: Color) -> FeatureSums {
        let rows = features::extract(graph, side, u128::MAX);
        let mut sums = FeatureSums::default();
        for row in king_piece_rows(&graph.board, side) {
            add_scaled_row(&mut sums.king, &self.king_piece, row as usize, 1.);
        }
        for row in rows.relations {
            add_scaled_row(
                &mut sums.relations,
                &self.relation_embedding,
                row as usize,
                1.,
            );
        }
        for row in rows.motifs {
            add_scaled_row(&mut sums.motifs, &self.motif_embedding, row as usize, 1.);
        }
        sums
    }
    pub(crate) fn evaluate_with_scratch_output(
        &self,
        p: &Position,
        m: &[Move],
        r: &[f32; RULE_CONTEXT_SIZE],
        s: &mut AzEvalScratch,
    ) -> AzEvalOutput {
        self.evaluate_with_scratch_output_with_repetition(p, m, &[], r, s)
    }
    pub(crate) fn evaluate_with_scratch_output_with_repetition(
        &self,
        p: &Position,
        m: &[Move],
        flags: &[u8],
        r: &[f32; RULE_CONTEXT_SIZE],
        s: &mut AzEvalScratch,
    ) -> AzEvalOutput {
        let mut board = vec![0.; self.hidden_size];
        let f = nnue::extract_sparse_features_az(p);
        self.input_embedding_linear_into_slice(&f, &mut board);
        self.evaluate_core(p, &board, m, flags, r, s)
    }
    pub(crate) fn evaluate_incremental_with_scratch_output(
        &self,
        p: &Position,
        b: &[f32],
        m: &[Move],
        flags: &[u8],
        r: &[f32; RULE_CONTEXT_SIZE],
        s: &mut AzEvalScratch,
    ) -> AzEvalOutput {
        let b = if b.len() == self.hidden_size {
            b
        } else {
            AzEvalAccumulator::hidden_for_slice(b, self.hidden_size, p.side_to_move())
        };
        self.evaluate_core(p, b, m, flags, r, s)
    }
    pub(crate) fn evaluate_search(
        &self,
        p: &Position,
        b: &[f32],
        m: &[Move],
        flags: &[u8],
        r: &[f32; RULE_CONTEXT_SIZE],
        s: &mut AzEvalScratch,
    ) -> AzEvalOutput {
        let missing_history: smallvec::SmallVec<[u8; 64]>;
        let flags = if r[12] == 0.0 {
            missing_history = flags
                .iter()
                .map(|&flag| if flag == 0 { 2 } else { flag })
                .collect();
            missing_history.as_slice()
        } else {
            flags
        };
        if self.node_incremental {
            self.evaluate_incremental_with_scratch_output(p, b, m, flags, r, s)
        } else {
            self.evaluate_with_scratch_output_with_repetition(p, m, flags, r, s)
        }
    }
    pub(crate) fn evaluate_core(
        &self,
        p: &Position,
        b: &[f32],
        moves: &[Move],
        flags: &[u8],
        rules: &[f32; RULE_CONTEXT_SIZE],
        s: &mut AzEvalScratch,
    ) -> AzEvalOutput {
        let side = p.side_to_move();
        let graph = s
            .graph_precomputed
            .take()
            .unwrap_or_else(|| Graph::new(Board::new(p)));
        let sums = s
            .feature_precomputed
            .take()
            .unwrap_or_else(|| self.feature_sums(&graph, side));
        let (context, seals) = king_context(&graph, side);
        let mut motif = sums.motifs;
        for row in seals {
            add_scaled_row(&mut motif, &self.motif_embedding, row as usize, 1.);
        }
        s.hidden.clear();
        s.hidden.extend_from_slice(b);
        for (value, bias) in s.hidden.iter_mut().zip(&self.hidden_bias) {
            *value += bias;
        }
        project_add(&sums.king, &self.king_projection, &mut s.hidden);
        project_add(&sums.relations, &self.relation_projection, &mut s.hidden);
        project_add(&motif, &self.motif_projection, &mut s.hidden);
        project_add(rules, &self.rule_context_hidden, &mut s.hidden);
        project_add(&context, &self.check_context_hidden, &mut s.hidden);
        relu_in_place(&mut s.hidden);
        s.value_head.resize(VALUE_HEAD_SIZE, 0.);
        matrix_dot_into(&s.hidden, &self.value_head_hidden, &mut s.value_head);
        for (v, bias) in s.value_head.iter_mut().zip(&self.value_head_bias) {
            *v = (*v + bias).max(0.);
        }
        let bucket = value_material_bucket(p, side);
        let mut logits = [0.; 3];
        for (i, v) in logits.iter_mut().enumerate() {
            let start = (bucket * 3 + i) * VALUE_HEAD_SIZE;
            *v = dot_product(
                &s.value_head,
                &self.value_head_output[start..start + VALUE_HEAD_SIZE],
            );
        }
        let wdl = softmax_fixed3(logits);
        let mut policy_context = [0.; POLICY_MOVE_CONTEXT_SIZE];
        matrix_dot_into(&s.hidden, &self.policy_context_hidden, &mut policy_context);
        let repeat =
            dot_product(&s.hidden, &self.policy_repetition_hidden) + self.policy_repetition_bias[0];
        let buckets = canonical_buckets_for_perspective(p, side);
        s.logits.clear();
        s.policy_gives_check.clear();
        let mut move_cache = MoveFeatureCache::new(&graph, moves.iter().copied());
        for (i, &mv) in moves.iter().enumerate() {
            let local = move_cache.features(
                mv,
                side,
                flags
                    .get(i)
                    .and_then(|&v| if v < 2 { Some(v != 0) } else { None }),
            );
            let id = dense_move_index(nnue::canonical_move(side, mv));
            let start = id * POLICY_MOVE_CONTEXT_SIZE;
            let mut score = self.policy_move_bias[id]
                + dot_product(
                    &policy_context,
                    &self.policy_move_context[start..start + POLICY_MOVE_CONTEXT_SIZE],
                );
            score += self.policy_piece_square[local.to_feature]
                - self.policy_piece_square[local.from_feature]
                - local
                    .captured_feature
                    .map_or(0., |f| self.policy_piece_square[f]);
            score += self.policy_move_piece[id * 7 + local.moved]
                + self.policy_move_capture[id * 8 + local.captured];
            score += self.policy_move_king[id * 9 + buckets.0]
                + self.policy_move_king[(DENSE_MOVE_SPACE + id) * 9 + buckets.1];
            score += dot_product(&local.local, &self.policy_local)
                + self.policy_joint[local.joint as usize]
                + local
                    .events
                    .iter()
                    .map(|&e| self.policy_events[e as usize])
                    .sum::<f32>();
            score += local.local[10] * repeat;
            s.logits.push(score);
            s.policy_gives_check.push(local.local[1]);
        }
        AzEvalOutput {
            value_wdl: wdl,
            value: wdl[0] - wdl[2],
            moves_left: self.moves_left_from_hidden(&s.hidden),
        }
    }
    pub fn evaluate_value(&self, p: &Position, m: &[Move]) -> f32 {
        let mut s = AzEvalScratch::new(self.arch);
        self.evaluate_with_scratch_output(p, m, &[0.; RULE_CONTEXT_SIZE], &mut s)
            .value
    }
    pub fn evaluate_value_with_rules(
        &self,
        p: &Position,
        h: &[crate::xiangqi::RuleHistoryEntry],
        m: &[Move],
    ) -> f32 {
        let mut s = AzEvalScratch::new(self.arch);
        self.evaluate_with_scratch_output(p, m, &rule_context_features(p, h), &mut s)
            .value
    }
    pub fn evaluate_wdl_with_rules(
        &self,
        p: &Position,
        h: &[crate::xiangqi::RuleHistoryEntry],
        m: &[Move],
    ) -> [f32; 3] {
        let mut s = AzEvalScratch::new(self.arch);
        self.evaluate_with_scratch_output(p, m, &rule_context_features(p, h), &mut s)
            .value_wdl
    }
    pub(crate) fn moves_left_from_hidden(&self, h: &[f32]) -> f32 {
        (dot_product(h, &self.moves_left_output) + self.moves_left_bias[0] + 1.).max(0.)
            * MOVES_LEFT_SCALE
    }
    pub(crate) fn rebuild_moves_left(&mut self) {
        self.moves_left_active = self
            .moves_left_output
            .iter()
            .chain(&self.moves_left_bias)
            .any(|&v| v != 0.);
    }
}

pub(crate) fn add_scaled_row(output: &mut [f32], table: &[f32], row: usize, scale: f32) {
    super::simd::add_scaled_feature_row(output, table, output.len(), row, scale);
}
fn project_add(input: &[f32], weights: &[f32], output: &mut [f32]) {
    for (i, &v) in input.iter().enumerate() {
        add_scaled_row(output, weights, i, v);
    }
}
pub(crate) fn king_piece_rows(board: &Board, side: Color) -> Vec<u32> {
    let buckets = board.kings.map(|s| {
        let p = board.cells[s].unwrap();
        canonical_general_bucket(
            if p.color == side { 0 } else { 7 },
            canonical_square(side, s),
        )
    });
    let us = buckets[color_index(side)];
    let them = buckets[color_index(side.opposite())];
    let mut rows = Vec::with_capacity(64);
    for square in bits(board.occupied) {
        let piece = board.cells[square].unwrap();
        let p = piece_absolute_feature_index(side, piece);
        let s = canonical_square(side, square);
        rows.push(((us * 14 + p) * 90 + s) as u32);
        rows.push((((9 + them) * 14 + p) * 90 + s) as u32);
    }
    rows
}
pub(crate) fn softmax_fixed3(mut x: [f32; 3]) -> [f32; 3] {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for v in &mut x {
        *v = (*v - m).exp();
    }
    let sum = x.iter().sum::<f32>();
    x.map(|v| v / sum)
}
pub fn position_for_training_sample(sample: &AzTrainingSample) -> Option<Position> {
    let pieces = sample
        .features
        .iter()
        .filter_map(|&f| decode_current_piece_square_feature(f))
        .map(|p| (p.piece_index, p.rank * 9 + p.file))
        .collect::<Vec<_>>();
    let p = Position::from_canonical_piece_squares(&pieces);
    (p.has_general(Color::Red) && p.has_general(Color::Black)).then_some(p)
}
pub fn outputs_for_training_sample(
    model: &AzNnue,
    sample: &AzTrainingSample,
) -> Option<([f32; 3], Vec<f32>)> {
    let p = position_for_training_sample(sample)?;
    let moves = sample
        .move_indices
        .iter()
        .map(|&i| dense_move_squares(i).map(|(a, b)| Move::new(a, b)))
        .collect::<Option<Vec<_>>>()?;
    let mut s = AzEvalScratch::new(model.arch);
    let out = model.evaluate_with_scratch_output_with_repetition(
        &p,
        &moves,
        &sample.repetition_flags,
        &sample.rule_context,
        &mut s,
    );
    Some((out.value_wdl, s.logits))
}
pub(crate) fn scalar_value_to_wdl_target(value: f32) -> [f32; 3] {
    let value = value.clamp(-1.0, 1.0);
    if value >= 0.0 {
        [value, 1.0 - value, 0.0]
    } else {
        [0.0, 1.0 + value, -value]
    }
}

pub(crate) fn normalize_wdl_target(mut wdl: [f32; WDL_HEAD_SIZE]) -> [f32; WDL_HEAD_SIZE] {
    for value in &mut wdl {
        *value = value.max(0.0);
    }
    let sum = wdl.iter().sum::<f32>();
    if sum.is_finite() && sum > 1.0e-6 {
        for value in &mut wdl {
            *value /= sum;
        }
        wdl
    } else {
        [0.0, 1.0, 0.0]
    }
}
