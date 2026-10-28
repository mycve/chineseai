use std::io;
use std::path::Path;

use candle_core::{DType, Device, Shape, Var};
use candle_nn::VarMap;

use crate::az::nnue::{
    AZ_NNUE_INPUT_SIZE, canonical_piece_square_reachable, canonical_square,
    fill_sparse_features_az, piece_absolute_feature_index,
};
use crate::infra::version::MODEL_FORMAT_VERSION;
use crate::xiangqi::{
    BOARD_FILES, BOARD_RANKS, BOARD_SIZE, Color, Move, Piece, PieceKind, Position, nearest_on_ray,
    piece_kind_index,
};

use super::arch::az_weight_tensors;
use super::feature_map::{value_king_piece_index, value_ray_index};
use super::*;

pub(crate) fn candle_io_error(err: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err.to_string())
}

pub(crate) fn insert_candle_var(
    varmap: &VarMap,
    name: &str,
    data: &[f32],
    shape: impl Into<Shape>,
) -> io::Result<()> {
    let var = Var::from_slice(data, shape, &Device::Cpu).map_err(candle_io_error)?;
    varmap
        .data()
        .lock()
        .unwrap_or_else(|_| panic!("candle varmap poisoned"))
        .insert(name.to_string(), var);
    Ok(())
}

pub(crate) fn load_candle_f32_tensor(
    tensors: &candle_core::safetensors::MmapedSafetensors,
    name: &str,
) -> io::Result<Vec<f32>> {
    let tensor = tensors.load(name, &Device::Cpu).map_err(candle_io_error)?;
    if tensor.dtype() != DType::F32 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("tensor `{name}` is {:?}, expected F32", tensor.dtype()),
        ));
    }
    tensor
        .flatten_all()
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(candle_io_error)
}

/// 按双方车与马炮数量选择价值输出专家；走棋方始终为第一视角。
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

pub(crate) fn visit_value_king_piece_features(
    position: &Position,
    perspective: Color,
    mut visitor: impl FnMut(usize),
) -> usize {
    let mut rooks = [0usize; 2];
    let mut minors = [0usize; 2];
    let buckets = canonical_buckets_for_perspective(position, perspective);
    for square in position.occupied_squares() {
        let piece = position
            .piece_at(square)
            .expect("occupancy and board must agree");
        let side = usize::from(piece.color != perspective);
        match piece.kind {
            PieceKind::Rook => rooks[side] += 1,
            PieceKind::Horse | PieceKind::Cannon => minors[side] += 1,
            _ => {}
        }
        let piece_index = piece_absolute_feature_index(perspective, piece);
        let canonical = canonical_square_for(perspective, square);
        for (king_side, bucket) in [(0, buckets.0), (1, buckets.1)] {
            if let Some(feature) = value_king_piece_index(king_side, bucket, piece_index, canonical)
            {
                visitor(feature);
            }
        }
    }
    value_material_bucket_counts(rooks, minors)
}

struct ThreatRelationMap {
    bases: Vec<u16>,
    compact: Vec<u16>,
}

fn threat_relation_map() -> &'static ThreatRelationMap {
    use std::sync::OnceLock;
    static MAP: OnceLock<ThreatRelationMap> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut map = vec![u16::MAX; STRUCTURAL_PIECE_SIZE * BOARD_SIZE * BOARD_SIZE];
        let mut next = 0u16;
        let mut keys = Vec::new();
        for attacker in 0..STRUCTURAL_PIECE_SIZE {
            let ours = attacker < 7;
            let kind = attacker % 7;
            // 车炮已由射线与炮架特征编码，不再分配从未激活的配对行。
            if kind == 4 || kind == 5 {
                continue;
            }
            for source in 0..BOARD_SIZE {
                if !canonical_piece_square_reachable(attacker, source) {
                    continue;
                }
                let rank = source / BOARD_FILES;
                let file = source % BOARD_FILES;
                let mut targets = Vec::with_capacity(18);
                let steps: &[(isize, isize)] = match kind {
                    0 => &[(0, -1), (0, 1), (-1, 0), (1, 0)],
                    1 => &[(-1, -1), (1, -1), (-1, 1), (1, 1)],
                    2 => &[(-2, -2), (2, -2), (-2, 2), (2, 2)],
                    3 => &[
                        (-1, -2),
                        (1, -2),
                        (-1, 2),
                        (1, 2),
                        (-2, -1),
                        (-2, 1),
                        (2, -1),
                        (2, 1),
                    ],
                    6 if ours && rank <= 4 => &[(0, -1), (-1, 0), (1, 0)],
                    6 if !ours && rank >= 5 => &[(0, 1), (-1, 0), (1, 0)],
                    6 if ours => &[(0, -1)],
                    6 => &[(0, 1)],
                    _ => unreachable!(),
                };
                for &(df, dr) in steps {
                    let target_file = file as isize + df;
                    let target_rank = rank as isize + dr;
                    if !(0..BOARD_FILES as isize).contains(&target_file)
                        || !(0..BOARD_RANKS as isize).contains(&target_rank)
                    {
                        continue;
                    }
                    if (kind == 0 || kind == 1)
                        && (!(3..=5).contains(&(target_file as usize))
                            || if ours {
                                target_rank < 7
                            } else {
                                target_rank > 2
                            })
                    {
                        continue;
                    }
                    if kind == 2
                        && if ours {
                            target_rank < 5
                        } else {
                            target_rank > 4
                        }
                    {
                        continue;
                    }
                    targets.push(target_rank as usize * BOARD_FILES + target_file as usize);
                }
                for target in targets {
                    let relation = (attacker * BOARD_SIZE + source) * BOARD_SIZE + target;
                    map[relation] = next;
                    for victim in 0..STRUCTURAL_PIECE_SIZE {
                        if canonical_piece_square_reachable(victim, target) {
                            keys.push((attacker, source, target, victim));
                            next += 1;
                        }
                    }
                }
            }
        }
        let mut compact = vec![u16::MAX; keys.len()];
        let mut count = 0;
        for (raw, &(attacker, source, target, victim)) in keys.iter().enumerate() {
            // 同色同将配对需要两个己方将，不可能出现在标准棋局中。
            if attacker == victim && attacker % 7 == 0 {
                continue;
            }
            // 马的往返腿格不同，不能共享；其他互反关系的阻挡条件相同。
            let reverse_base = map[(victim * BOARD_SIZE + target) * BOARD_SIZE + source];
            let reverse_offset =
                threat_attacked_offsets()[source * STRUCTURAL_PIECE_SIZE + attacker];
            let reverse = usize::from(reverse_base) + usize::from(reverse_offset);
            compact[raw] = if attacker % 7 != 3
                && victim % 7 != 3
                && reverse_base != u16::MAX
                && reverse_offset != u8::MAX
                && reverse < raw
            {
                compact[reverse]
            } else {
                let index = count;
                count += 1;
                index
            };
        }
        assert_eq!(count as usize, VALUE_THREAT_PAIR_VOCAB);
        ThreatRelationMap {
            bases: map,
            compact,
        }
    })
}

pub(crate) fn threat_attacked_offsets() -> &'static [u8] {
    use std::sync::OnceLock;
    static OFFSETS: OnceLock<Vec<u8>> = OnceLock::new();
    OFFSETS.get_or_init(|| {
        let mut offsets = vec![u8::MAX; BOARD_SIZE * STRUCTURAL_PIECE_SIZE];
        for square in 0..BOARD_SIZE {
            let mut next = 0u8;
            for piece in 0..STRUCTURAL_PIECE_SIZE {
                if canonical_piece_square_reachable(piece, square) {
                    offsets[square * STRUCTURAL_PIECE_SIZE + piece] = next;
                    next += 1;
                }
            }
        }
        offsets
    })
}

#[inline]
pub(crate) fn canonical_threat_pair_index(
    attacker: usize,
    source: usize,
    target: usize,
    attacked: usize,
) -> Option<usize> {
    let relation = (attacker * BOARD_SIZE + source) * BOARD_SIZE + target;
    let map = threat_relation_map();
    let base = map.bases[relation];
    if base == u16::MAX {
        return None;
    }
    let offset = threat_attacked_offsets()[target * STRUCTURAL_PIECE_SIZE + attacked];
    if offset == u8::MAX {
        return None;
    }
    let compact = map.compact[base as usize + usize::from(offset)];
    (compact != u16::MAX).then_some(usize::from(compact))
}

#[inline]
pub(crate) fn value_threat_index(
    perspective: Color,
    source: usize,
    attacker: Piece,
    target: usize,
    attacked: Piece,
) -> usize {
    let attacker =
        (if attacker.color == perspective { 0 } else { 7 }) + piece_kind_index(attacker.kind);
    let attacked =
        (if attacked.color == perspective { 0 } else { 7 }) + piece_kind_index(attacked.kind);
    let source = canonical_square_for(perspective, source);
    let target = canonical_square_for(perspective, target);
    canonical_threat_pair_index(attacker, source, target, attacked)
        .unwrap_or(VALUE_THREAT_PAIR_VOCAB)
}

#[cfg(test)]
pub(crate) fn visit_value_threat_features(
    position: &Position,
    perspective: Color,
    visitor: impl FnMut(usize),
) {
    visit_value_threat_features_from(position, perspective, u128::MAX, visitor);
}

#[cfg(test)]
pub(crate) fn visit_value_threat_features_from(
    position: &Position,
    perspective: Color,
    sources: u128,
    mut visitor: impl FnMut(usize),
) {
    visit_value_threat_entries_from(position, perspective, sources, |entry| {
        visitor((entry & 0xffff) as usize)
    });
}

#[cfg(test)]
pub(crate) fn visit_value_threat_entries_from(
    position: &Position,
    perspective: Color,
    sources: u128,
    visitor: impl FnMut(u32),
) {
    visit_threat_entries_with_masks(
        position,
        perspective,
        sources,
        sources,
        position.attacked_squares_masks(),
        visitor,
    );
}

pub(crate) fn visit_threat_entries_with_masks(
    position: &Position,
    perspective: Color,
    sources: u128,
    mobility_sources: u128,
    attacks: [u128; 2],
    mut visitor: impl FnMut(u32),
) {
    position.visit_occupied_leaper_relations_from(sources, |source, attacker, target, attacked| {
        let feature = value_threat_index(perspective, source, attacker, target, attacked);
        if feature != VALUE_THREAT_PAIR_VOCAB {
            visitor(((source as u32) << 16) | feature as u32);
        }
    });

    for source in position.occupied_squares_from(sources) {
        let attacker = position
            .piece_at(source)
            .expect("occupancy and board must agree");
        if !matches!(attacker.kind, PieceKind::Rook | PieceKind::Cannon) {
            continue;
        }
        for (df, dr) in [(0, -1), (0, 1), (-1, 0), (1, 0)] {
            let direction_native = if df == 0 {
                2 + usize::from(dr < 0)
            } else {
                usize::from(df < 0)
            };
            let blockers = position.orthogonal_blockers(source, direction_native);
            let blocker_count = blockers.count_ones() as usize;
            let increasing = dr * BOARD_FILES as i32 + df > 0;
            let first = if blockers == 0 {
                None
            } else {
                let sq = nearest_on_ray(blockers, increasing);
                Some((sq, position.piece_at(sq).unwrap()))
            };
            let rest = first.map_or(0, |(sq, _)| blockers & !(1u128 << sq));
            let second = if rest == 0 {
                None
            } else {
                let sq = nearest_on_ray(rest, increasing);
                Some((sq, position.piece_at(sq).unwrap()))
            };
            let owner = usize::from(attacker.color != perspective);
            let slider = usize::from(attacker.kind == PieceKind::Cannon);
            let attacker_class = owner * 2 + slider;
            if blocker_count == 0 {
                let (canonical_df, canonical_dr) = if perspective == Color::Red {
                    (df, dr)
                } else {
                    (-df, -dr)
                };
                let direction = if canonical_df == 0 {
                    usize::from(canonical_dr > 0)
                } else {
                    2 + usize::from(canonical_df > 0)
                };
                let ray = value_ray_index(attacker_class, 0, direction, 0, 14, 0)
                    .expect("empty ray must have a feature");
                visitor(
                    (1 << 23) | ((source as u32) << 16) | (VALUE_THREAT_PAIR_VOCAB + ray) as u32,
                );
                continue;
            }
            let ray_state = blocker_count.min(3);
            let canonical_source = canonical_square(perspective, source);
            for (ordinal, (square, blocked)) in [first, second].into_iter().flatten().enumerate() {
                let canonical_target = canonical_square(perspective, square);
                let sf = canonical_source % BOARD_FILES;
                let sr = canonical_source / BOARD_FILES;
                let tf = canonical_target % BOARD_FILES;
                let tr = canonical_target / BOARD_FILES;
                let direction = if tf == sf {
                    usize::from(tr > sr)
                } else {
                    2 + usize::from(tf > sf)
                };
                let distance = sf.abs_diff(tf).max(sr.abs_diff(tr)) - 1;
                let blocked_class = (if blocked.color == perspective { 0 } else { 7 })
                    + piece_kind_index(blocked.kind);
                if let Some(ray) = value_ray_index(
                    attacker_class,
                    ordinal,
                    direction,
                    distance,
                    blocked_class,
                    ray_state,
                ) {
                    visitor(
                        (1 << 23)
                            | ((source as u32) << 16)
                            | (VALUE_THREAT_PAIR_VOCAB + ray) as u32,
                    );
                }

                if attacker.kind == PieceKind::Cannon && ordinal == 1 {
                    let (screen_source, screen) = first.expect("second blocker requires first");
                    let screen_square = canonical_square(perspective, screen_source);
                    let screen_distance = (canonical_source % BOARD_FILES)
                        .abs_diff(screen_square % BOARD_FILES)
                        .max(
                            (canonical_source / BOARD_FILES).abs_diff(screen_square / BOARD_FILES),
                        )
                        - 1;
                    let screen_class = (if screen.color == perspective { 0 } else { 7 })
                        + piece_kind_index(screen.kind);
                    let target_class = blocked_class;
                    let mut triple = canonical_source as u64;
                    for field in [
                        owner,
                        direction,
                        screen_class,
                        screen_distance,
                        target_class,
                        distance,
                    ] {
                        triple = triple
                            .wrapping_mul(0x9E37_79B1_85EB_CA87)
                            .wrapping_add(field as u64 + 0xC2B2_AE3D_27D4_EB4F);
                        triple ^= triple >> 29;
                    }
                    let triple = triple as usize & (VALUE_CANNON_TRIPLE_VOCAB - 1);
                    visitor(
                        (1 << 23)
                            | ((source as u32) << 16)
                            | (VALUE_THREAT_PAIR_VOCAB + VALUE_RAY_VOCAB + triple) as u32,
                    );
                }
            }
        }
    }
    super::horse::visit_horse_threat_entries_from(position, perspective, sources, &mut visitor);
    super::mobility::visit_mobility_entries_from(
        position,
        perspective,
        mobility_sources,
        attacks,
        visitor,
    );
}

#[derive(Debug)]
pub struct AzNnue {
    pub hidden_size: usize,
    pub arch: AzNnueArch,
    pub input_hidden: Vec<f32>,
    pub context_hidden: Vec<f32>,
    pub hidden_bias: Vec<f32>,
    pub shared_hidden: Vec<f32>,
    pub shared_bias: Vec<f32>,
    pub value_head_hidden: Vec<f32>,
    pub value_head_bias: Vec<f32>,
    pub value_king_piece_hidden: Vec<f32>,
    pub value_king_piece_projection: Vec<f32>,
    pub value_head_output: Vec<f32>,
    pub moves_left_output: Vec<f32>,
    pub moves_left_bias: Vec<f32>,
    pub(crate) moves_left_active: bool,
    pub moves_left_params: AzMovesLeftParams,
    pub threat_embedding: Vec<f32>,
    pub value_threat_output: Vec<f32>,
    pub policy_threat_context: Vec<f32>,
    pub policy_context_hidden: Vec<f32>,
    pub policy_move_context: Vec<f32>,
    pub policy_accumulator_hidden: Vec<f32>,
    pub policy_accumulator_move: Vec<f32>,
    pub policy_sparse_table: Vec<f32>,
    pub policy_tactical: Vec<f32>,
    pub policy_repetition_hidden: Vec<f32>,
    pub policy_repetition_bias: Vec<f32>,
    pub(crate) policy_accumulator_features: Vec<f32>,
    pub(crate) threat_active: bool,
    pub(crate) policy_tactical_active: bool,
    /// 统一上下文投影是否非零（全零时跳过特征提取）。
    pub(crate) context_active: bool,
    /// 根节点 check-only 连杀证明搜索的最大半回合数（0 = 关闭）。
    ///
    /// 挂在模型上而不是 `AzSearchLimits` 上，是因为后者的结构体字面量全仓有 41 处，
    /// 而它只在根节点用一次、与网络质量无关，属于"这一次搜索肯花多少额外预算"。
    /// 打开后：证明出连杀就直接把该子局面标成对方必败，`root_policy` 会把策略目标
    /// 压到杀着上、`proven_root_value` 会把价值目标设为必胜。
    ///
    /// 代价：每次根搜索多一次受限搜索。**实测**（`az-bench best.safetensors 800 5 1.4`，
    /// ms/search 取 5 次均值）：
    /// - 根局面**没有**将军着法（startpos）：4.08 → 3.99ms，无可测代价（只多一遍 check 过滤）；
    /// - 有将军但**没杀**（中局）：7.31 → 7.59ms（plies=9，**+4%**）、12.52ms（plies=15，**+71%**）；
    /// - **真出杀**（mate-in-8 的局面）：5.4 → 36.5ms，一次证明约 **31ms / 58,178 节点**；
    ///   证完根节点已 solved，`simulate` 立即返回，所以每次搜索只付一次。
    ///
    /// 建议：**自博弈/数据生成用 9**（把可证射程从 mate-in-4 推到 mate-in-5，成本被深度上限兜住）；
    /// **分析、UCI `go mate N`、高预算评测用 15~17**（那个 mate-in-8 需要 15）。
    ///
    /// 想证更长的杀只能加预算（[`Self::mate_search_nodes`]）：置换表和着法排序都实测过，
    /// 在 check-only 树上没有收益，原因记在 `mate.rs` 的模块注释里。
    pub mate_search_plies: usize,
    /// 实验性威胁增量缓存；当前 MCTS 默认使用更快的全量遍历。
    pub threat_incremental: bool,
    pub node_incremental: bool,
    /// 连杀证明搜索的全局节点预算上限。默认 20 万覆盖实测的 mate-in-8（58,178 节点），
    /// 同时是"最坏情况花多少时间"的硬兜底。
    ///
    /// UCI 侧对应 `MateSearchNodes`。历史压力测试中最坏的中局在 depth=31 下要 **656,431**
    /// 节点才能判"无杀"（约 380ms）；把预算开到 2,000,000 就能让它走到结论，代价是这一手
    /// 慢 10 倍。预算撞墙时会打印 `info string mate ... source=node-budget`。
    pub mate_search_nodes: usize,
    #[cfg_attr(not(feature = "gpu-train"), allow(dead_code))]
    pub(super) gpu_trainer: Option<Box<train_gpu::GpuTrainer>>,
}

impl Clone for AzNnue {
    fn clone(&self) -> Self {
        Self {
            hidden_size: self.hidden_size,
            arch: self.arch,
            input_hidden: self.input_hidden.clone(),
            context_hidden: self.context_hidden.clone(),
            hidden_bias: self.hidden_bias.clone(),
            shared_hidden: self.shared_hidden.clone(),
            shared_bias: self.shared_bias.clone(),
            value_head_hidden: self.value_head_hidden.clone(),
            value_head_bias: self.value_head_bias.clone(),
            value_king_piece_hidden: self.value_king_piece_hidden.clone(),
            value_king_piece_projection: self.value_king_piece_projection.clone(),
            value_head_output: self.value_head_output.clone(),
            moves_left_output: self.moves_left_output.clone(),
            moves_left_bias: self.moves_left_bias.clone(),
            moves_left_active: self.moves_left_active,
            moves_left_params: self.moves_left_params,
            threat_embedding: self.threat_embedding.clone(),
            value_threat_output: self.value_threat_output.clone(),
            policy_threat_context: self.policy_threat_context.clone(),
            policy_context_hidden: self.policy_context_hidden.clone(),
            policy_move_context: self.policy_move_context.clone(),
            policy_accumulator_hidden: self.policy_accumulator_hidden.clone(),
            policy_accumulator_move: self.policy_accumulator_move.clone(),
            policy_sparse_table: self.policy_sparse_table.clone(),
            policy_tactical: self.policy_tactical.clone(),
            policy_repetition_hidden: self.policy_repetition_hidden.clone(),
            policy_repetition_bias: self.policy_repetition_bias.clone(),
            policy_accumulator_features: self.policy_accumulator_features.clone(),
            threat_active: self.threat_active,
            policy_tactical_active: self.policy_tactical_active,
            context_active: self.context_active,
            mate_search_plies: self.mate_search_plies,
            threat_incremental: self.threat_incremental,
            node_incremental: self.node_incremental,
            mate_search_nodes: self.mate_search_nodes,
            gpu_trainer: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct AzEvalOutput {
    pub value_wdl: [f32; WDL_HEAD_SIZE],
    pub value: f32,
    pub moves_left: f32,
}

impl AzNnue {
    pub fn random_with_arch(arch: AzNnueArch, seed: u64) -> Self {
        if let Err(err) = arch.validate() {
            panic!("AzNnue::random_with_arch: invalid arch ({err})");
        }
        let hidden_size = arch.hidden_size;
        let mut rng = SplitMix64::new(seed);
        let input_hidden: Vec<f32> = (0..AZ_NNUE_INPUT_SIZE * hidden_size)
            .map(|_| rng.weight(0.015))
            .collect();
        // 局面上下文从零开始，通过共享主干学习策略与价值的共同需求。
        let context_hidden = vec![0.0; POSITION_CONTEXT_SIZE * 2 * hidden_size];
        let hidden_bias = vec![0.0; hidden_size];
        // Start value-neutral. A random value head can evaluate startpos as a
        // large red/black advantage before any training, and MCTS amplifies
        // that noise into the first self-play dataset.
        let value_head_hidden = (0..VALUE_HEAD_SIZE * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt() * 0.5))
            .collect();
        let value_head_bias = vec![0.0; VALUE_HEAD_SIZE];
        let value_king_piece_hidden = vec![0.0; VALUE_KING_PIECE_VOCAB * VALUE_KING_PIECE_RANK];
        // 零表保持初始旁路输出为零；非零投影使价值输出头开始学习后，表即可收到梯度。
        let mut value_king_piece_projection = vec![0.0; VALUE_KING_PIECE_RANK * VALUE_HEAD_SIZE];
        for channel in 0..VALUE_KING_PIECE_RANK {
            value_king_piece_projection[channel * VALUE_HEAD_SIZE + channel] = 1.0;
        }
        // Keep the value head output-neutral at initialization. This preserves
        // stable first self-play while giving value its own nonlinear capacity.
        let value_head_output = vec![0.0; VALUE_BUCKETS * WDL_HEAD_SIZE * VALUE_HEAD_SIZE];
        let threat_embedding = (0..VALUE_THREAT_VOCAB * VALUE_THREAT_RANK)
            .map(|_| rng.weight(0.02))
            .collect();
        let value_threat_output = vec![0.0; WDL_HEAD_SIZE * VALUE_THREAT_RANK * 2];
        let policy_threat_context = vec![0.0; POLICY_THREAT_CONTEXT_SIZE * VALUE_THREAT_RANK * 2];
        // Zero output preserves the exact policy distribution until this branch is trained.
        // One factor starts random and the other at zero: the new branch is
        // exactly policy-neutral at initialization, while gradients can update
        // move embeddings on the first optimization step.
        let policy_context_hidden = (0..POLICY_MOVE_CONTEXT_SIZE * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt() * 0.5))
            .collect();
        let policy_move_context = vec![0.0; DENSE_MOVE_SPACE * POLICY_MOVE_CONTEXT_SIZE];
        let policy_accumulator_hidden = (0..POLICY_ACCUMULATOR_RANK * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt() * 0.5))
            .collect();
        let policy_accumulator_move = vec![0.0; DENSE_MOVE_SPACE * POLICY_ACCUMULATOR_RANK];
        let policy_sparse_table = vec![0.0; POLICY_SPARSE_TABLE_SIZE];
        let policy_tactical = vec![0.0; POLICY_TACTICAL_SIZE];
        let policy_repetition_hidden = vec![0.0; hidden_size];
        let policy_repetition_bias = vec![0.0; 1];
        let shared_hidden = (0..hidden_size * 2 * hidden_size)
            .map(|_| rng.weight((6.0 / (2 * hidden_size) as f32).sqrt()))
            .collect();
        let mut model = Self {
            hidden_size,
            arch,
            input_hidden,
            context_hidden,
            hidden_bias,
            shared_hidden,
            shared_bias: vec![0.0; hidden_size],
            value_head_hidden,
            value_head_bias,
            value_king_piece_hidden,
            value_king_piece_projection,
            value_head_output,
            moves_left_output: vec![0.0; hidden_size],
            moves_left_bias: vec![0.0],
            moves_left_active: false,
            moves_left_params: AzMovesLeftParams::default(),
            threat_embedding,
            value_threat_output,
            policy_threat_context,
            policy_context_hidden,
            policy_move_context,
            policy_accumulator_hidden,
            policy_accumulator_move,
            policy_sparse_table,
            policy_tactical,
            policy_repetition_hidden,
            policy_repetition_bias,
            policy_accumulator_features: Vec::new(),
            threat_active: false,
            policy_tactical_active: false,
            context_active: false,
            mate_search_plies: 0,
            threat_incremental: false,
            node_incremental: true,
            mate_search_nodes: 200_000,
            gpu_trainer: None,
        };
        model.rebuild_policy_cache();
        model.rebuild_threat();
        model.rebuild_context();
        model.rebuild_moves_left();
        model
    }

    pub fn random(hidden_size: usize, seed: u64) -> Self {
        Self::random_with_arch(AzNnueArch::with_hidden_size(hidden_size), seed)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let h = self.hidden_size;
        let varmap = VarMap::new();
        insert_candle_var(
            &varmap,
            "az_model_format_version",
            &[MODEL_FORMAT_VERSION],
            (1,),
        )?;
        macro_rules! save_tensor {
            ($field:ident, [$($dim:expr),+], $decay:expr) => {
                insert_candle_var(&varmap, stringify!($field), &self.$field, ($($dim),+))?;
            };
        }
        az_weight_tensors!(save_tensor, h);
        varmap.save(path).map_err(candle_io_error)
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

    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let tensors = unsafe {
            candle_core::safetensors::MmapedSafetensors::new(path.as_ref())
                .map_err(candle_io_error)?
        };
        let mut expected_tensors = vec!["az_model_format_version"];
        macro_rules! expect_tensor {
            ($field:ident, [$($dim:expr),+], $decay:expr) => {
                expected_tensors.push(stringify!($field));
            };
        }
        az_weight_tensors!(expect_tensor, 0);
        if let Some((name, _)) = tensors
            .tensors()
            .iter()
            .find(|(name, _)| !expected_tensors.contains(&name.as_str()))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported AZ model tensor `{name}`"),
            ));
        }
        let format_version = load_candle_f32_tensor(&tensors, "az_model_format_version")?;
        let Some(&format_version) = format_version.first() else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "missing AZ model format",
            ));
        };
        if format_version != MODEL_FORMAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported AZ model format {:?}; expected v{}",
                    format_version, MODEL_FORMAT_VERSION
                ),
            ));
        }
        let hidden_bias = load_candle_f32_tensor(&tensors, "hidden_bias")?;
        let hidden_size = hidden_bias.len();
        let arch = AzNnueArch { hidden_size };
        let policy_accumulator_hidden =
            load_candle_f32_tensor(&tensors, "policy_accumulator_hidden")?;
        let policy_accumulator_move = load_candle_f32_tensor(&tensors, "policy_accumulator_move")?;
        let mut model = Self {
            hidden_size,
            arch,
            input_hidden: load_candle_f32_tensor(&tensors, "input_hidden")?,
            context_hidden: load_candle_f32_tensor(&tensors, "context_hidden")?,
            hidden_bias,
            shared_hidden: load_candle_f32_tensor(&tensors, "shared_hidden")?,
            shared_bias: load_candle_f32_tensor(&tensors, "shared_bias")?,
            value_head_hidden: load_candle_f32_tensor(&tensors, "value_head_hidden")?,
            value_head_bias: load_candle_f32_tensor(&tensors, "value_head_bias")?,
            value_king_piece_hidden: load_candle_f32_tensor(&tensors, "value_king_piece_hidden")?,
            value_king_piece_projection: load_candle_f32_tensor(
                &tensors,
                "value_king_piece_projection",
            )?,
            value_head_output: load_candle_f32_tensor(&tensors, "value_head_output")?,
            moves_left_output: load_candle_f32_tensor(&tensors, "moves_left_output")?,
            moves_left_bias: load_candle_f32_tensor(&tensors, "moves_left_bias")?,
            moves_left_active: false,
            moves_left_params: AzMovesLeftParams::default(),
            threat_embedding: load_candle_f32_tensor(&tensors, "threat_embedding")?,
            value_threat_output: load_candle_f32_tensor(&tensors, "value_threat_output")?,
            policy_threat_context: load_candle_f32_tensor(&tensors, "policy_threat_context")?,
            policy_context_hidden: load_candle_f32_tensor(&tensors, "policy_context_hidden")?,
            policy_move_context: load_candle_f32_tensor(&tensors, "policy_move_context")?,
            policy_accumulator_hidden,
            policy_accumulator_move,
            policy_sparse_table: load_candle_f32_tensor(&tensors, "policy_sparse_table")?,
            policy_repetition_hidden: load_candle_f32_tensor(&tensors, "policy_repetition_hidden")?,
            policy_repetition_bias: load_candle_f32_tensor(&tensors, "policy_repetition_bias")?,
            policy_tactical: load_candle_f32_tensor(&tensors, "policy_tactical")?,
            policy_accumulator_features: Vec::new(),
            threat_active: false,
            policy_tactical_active: false,
            context_active: false,
            mate_search_plies: 0,
            threat_incremental: false,
            node_incremental: true,
            mate_search_nodes: 200_000,
            gpu_trainer: None,
        };
        model.rebuild_policy_cache();
        model.rebuild_threat();
        model.rebuild_context();
        model.rebuild_moves_left();
        model.rebuild_policy_tactical();
        model.validate()?;
        Ok(model)
    }

    pub fn evaluate_value(&self, position: &Position, moves: &[Move]) -> f32 {
        let mut scratch = AzEvalScratch::new(self.arch);
        self.evaluate_with_scratch(position, moves, &mut scratch)
    }

    pub fn evaluate_value_with_rules(
        &self,
        position: &Position,
        history: &[crate::xiangqi::RuleHistoryEntry],
        moves: &[Move],
    ) -> f32 {
        let mut scratch = AzEvalScratch::new(self.arch);
        self.evaluate_with_scratch_output_with_repetition_and_history(
            position,
            moves,
            &[],
            &rule_context_features(position, history),
            history,
            &mut scratch,
        )
        .value
    }

    pub fn evaluate_wdl_with_rules(
        &self,
        position: &Position,
        history: &[crate::xiangqi::RuleHistoryEntry],
        moves: &[Move],
    ) -> [f32; WDL_HEAD_SIZE] {
        let mut scratch = AzEvalScratch::new(self.arch);
        self.evaluate_with_scratch_output_with_repetition_and_history(
            position,
            moves,
            &[],
            &rule_context_features(position, history),
            history,
            &mut scratch,
        )
        .value_wdl
    }

    pub(crate) fn evaluate_with_scratch(
        &self,
        position: &Position,
        moves: &[Move],
        scratch: &mut AzEvalScratch,
    ) -> f32 {
        self.evaluate_with_scratch_output(position, moves, &[0.0; RULE_CONTEXT_SIZE], scratch)
            .value
    }

    pub(crate) fn evaluate_with_scratch_output(
        &self,
        position: &Position,
        moves: &[Move],
        rule_context: &[f32; RULE_CONTEXT_SIZE],
        scratch: &mut AzEvalScratch,
    ) -> AzEvalOutput {
        self.evaluate_with_scratch_output_with_repetition(
            position,
            moves,
            &[],
            rule_context,
            scratch,
        )
    }

    pub(crate) fn evaluate_with_scratch_output_with_repetition(
        &self,
        position: &Position,
        moves: &[Move],
        repetition_flags: &[u8],
        rule_context: &[f32; RULE_CONTEXT_SIZE],
        scratch: &mut AzEvalScratch,
    ) -> AzEvalOutput {
        self.evaluate_with_scratch_output_with_repetition_and_history(
            position,
            moves,
            repetition_flags,
            rule_context,
            &[],
            scratch,
        )
    }

    pub(crate) fn evaluate_with_scratch_output_with_repetition_and_history(
        &self,
        position: &Position,
        moves: &[Move],
        repetition_flags: &[u8],
        rule_context: &[f32; RULE_CONTEXT_SIZE],
        history: &[crate::xiangqi::RuleHistoryEntry],
        scratch: &mut AzEvalScratch,
    ) -> AzEvalOutput {
        let history = if self.context_active {
            super::history::history_features(position, history)
        } else {
            [0.0; HISTORY_CONTEXT_SIZE]
        };
        self.evaluate_with_scratch_output_with_repetition_and_history_features(
            position,
            moves,
            repetition_flags,
            rule_context,
            &history,
            scratch,
        )
    }

    pub(crate) fn evaluate_with_scratch_output_with_repetition_and_history_features(
        &self,
        position: &Position,
        moves: &[Move],
        repetition_flags: &[u8],
        rule_context: &[f32; RULE_CONTEXT_SIZE],
        history_features: &[f32; HISTORY_CONTEXT_SIZE],
        scratch: &mut AzEvalScratch,
    ) -> AzEvalOutput {
        crate::scope_profile!("az.evaluate_with_scratch");
        let mut features = std::mem::take(&mut scratch.features);
        fill_sparse_features_az(position, &mut features);
        scratch.hidden.resize(2 * self.hidden_size, 0.0);
        self.input_embedding_linear_into_slice(&features, &mut scratch.hidden[..self.hidden_size]);
        for feature in &mut features {
            *feature = nnue::opposite_feature(*feature);
        }
        self.input_embedding_linear_into_slice(&features, &mut scratch.hidden[self.hidden_size..]);
        for feature in &mut features {
            *feature = nnue::opposite_feature(*feature);
        }
        scratch.features = features;
        scratch.policy_accumulator_context =
            self.policy_accumulator(position, position.side_to_move());
        self.evaluate_encoded_position(
            position,
            moves,
            repetition_flags,
            rule_context,
            history_features,
            scratch,
        )
    }

    #[cfg(test)]
    pub(crate) fn evaluate_incremental_with_scratch_output(
        &self,
        position: &Position,
        accumulator_hidden: &[f32],
        policy_accumulator: &[f32; POLICY_ACCUMULATOR_RANK],
        moves: &[Move],
        repetition_flags: &[u8],
        rule_context: &[f32; RULE_CONTEXT_SIZE],
        scratch: &mut AzEvalScratch,
    ) -> AzEvalOutput {
        self.evaluate_incremental_with_scratch_output_with_history(
            position,
            accumulator_hidden,
            policy_accumulator,
            moves,
            repetition_flags,
            rule_context,
            &[],
            scratch,
        )
    }

    pub(crate) fn evaluate_incremental_with_scratch_output_with_history(
        &self,
        position: &Position,
        accumulator_hidden: &[f32],
        policy_accumulator: &[f32; POLICY_ACCUMULATOR_RANK],
        moves: &[Move],
        repetition_flags: &[u8],
        rule_context: &[f32; RULE_CONTEXT_SIZE],
        history: &[crate::xiangqi::RuleHistoryEntry],
        scratch: &mut AzEvalScratch,
    ) -> AzEvalOutput {
        crate::scope_profile!("az.evaluate_incremental_with_scratch");
        let side = position.side_to_move();
        scratch.hidden.resize(2 * self.hidden_size, 0.0);
        for (slot, perspective) in [side, side.opposite()].into_iter().enumerate() {
            let source = AzEvalAccumulator::hidden_for_slice(
                accumulator_hidden,
                self.hidden_size,
                perspective,
            );
            scratch.hidden[slot * self.hidden_size..(slot + 1) * self.hidden_size]
                .copy_from_slice(source);
        }
        scratch
            .policy_accumulator_context
            .copy_from_slice(policy_accumulator);
        let history = if self.context_active {
            super::history::history_features(position, history)
        } else {
            [0.0; HISTORY_CONTEXT_SIZE]
        };
        self.evaluate_encoded_position(
            position,
            moves,
            repetition_flags,
            rule_context,
            &history,
            scratch,
        )
    }

    /// 全量和增量路径在棋子累加器准备完毕后汇合，后续编码与输出完全共用。
    fn evaluate_encoded_position(
        &self,
        position: &Position,
        moves: &[Move],
        repetition_flags: &[u8],
        rules: &[f32; RULE_CONTEXT_SIZE],
        history: &[f32; HISTORY_CONTEXT_SIZE],
        scratch: &mut AzEvalScratch,
    ) -> AzEvalOutput {
        self.encode_shared_position(position, moves, rules, history, scratch);
        let (value_wdl, value) = self.evaluate_value_head(position, scratch);
        self.evaluate_policy_with_scratch(position, moves, repetition_flags, scratch);
        AzEvalOutput {
            value_wdl,
            value,
            moves_left: self.moves_left_from_hidden(&scratch.hidden),
        }
    }

    pub(crate) fn input_embedding_linear_into_slice(&self, features: &[usize], hidden: &mut [f32]) {
        debug_assert_eq!(hidden.len(), self.hidden_size);
        hidden.copy_from_slice(&self.hidden_bias);
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if self.hidden_size >= 64 && std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: runtime detection above guarantees AVX2 support.
                unsafe {
                    input_embedding_add_features_avx2(
                        &self.input_hidden,
                        self.hidden_size,
                        features,
                        hidden,
                    );
                }
                return;
            }
        }
        for &feature in features {
            let Some(feature) = nnue::input_feature_index(feature) else {
                continue;
            };
            let row =
                &self.input_hidden[feature * self.hidden_size..(feature + 1) * self.hidden_size];
            for (left, &right) in hidden.iter_mut().zip(row) {
                *left += right;
            }
        }
    }

    pub(crate) fn rebuild_threat(&mut self) {
        self.threat_active = self.value_threat_output.iter().any(|&weight| weight != 0.0)
            || self
                .policy_threat_context
                .iter()
                .any(|&weight| weight != 0.0);
    }

    pub(crate) fn validate(&self) -> io::Result<()> {
        let arch = &self.arch;
        if arch.hidden_size != self.hidden_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "aznnue arch.hidden_size does not match the cached hidden_size field",
            ));
        }
        if let Err(err) = arch.validate() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("aznnue arch invalid: {err}"),
            ));
        }
        let hidden = arch.hidden_size;
        macro_rules! validate_tensor {
            ($field:ident, [$($dim:expr),+], $decay:expr) => {
                let expected = [$($dim),+].into_iter().product::<usize>();
                if self.$field.len() != expected {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "az model tensor `{}` length mismatch: got {}, expected {}",
                            stringify!($field),
                            self.$field.len(),
                            expected
                        ),
                    ));
                }
            };
        }
        az_weight_tensors!(validate_tensor, hidden);
        if self
            .value_king_piece_hidden
            .iter()
            .chain(&self.value_king_piece_projection)
            .any(|weight| !weight.is_finite())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "az king-piece factors have invalid parameters",
            ));
        }
        if self.context_hidden.iter().any(|x| !x.is_finite()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "az context has invalid parameters",
            ));
        }
        if self.policy_accumulator_features.len()
            != POLICY_ACCUMULATOR_ROWS * POLICY_ACCUMULATOR_RANK
            || self.policy_accumulator_move.len() != DENSE_MOVE_SPACE * POLICY_ACCUMULATOR_RANK
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "az model derived policy accumulator cache length mismatch",
            ));
        }
        Ok(())
    }
}

pub(crate) fn softmax_fixed3(logits: [f32; 3]) -> [f32; 3] {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out = [
        (logits[0] - max_logit).exp(),
        (logits[1] - max_logit).exp(),
        (logits[2] - max_logit).exp(),
    ];
    let sum = (out[0] + out[1] + out[2]).max(f32::MIN_POSITIVE);
    out[0] /= sum;
    out[1] /= sum;
    out[2] /= sum;
    out
}

pub fn outputs_for_training_sample(
    model: &AzNnue,
    sample: &AzTrainingSample,
) -> Option<([f32; WDL_HEAD_SIZE], Vec<f32>)> {
    let position = position_for_training_sample(sample)?;
    let moves = sample
        .move_indices
        .iter()
        .filter_map(|&index| dense_move_squares(index))
        .map(|(from, to)| Move::new(from, to))
        .collect::<Vec<_>>();
    if moves.len() != sample.move_indices.len() {
        return None;
    }
    let mut scratch = AzEvalScratch::new(model.arch);
    let evaluated = model.evaluate_with_scratch_output_with_repetition_and_history_features(
        &position,
        &moves,
        &sample.repetition_flags,
        &sample.rule_context,
        &sample.history_features,
        &mut scratch,
    );
    Some((evaluated.value_wdl, scratch.logits))
}

pub fn position_for_training_sample(sample: &AzTrainingSample) -> Option<Position> {
    let pieces = sample
        .features
        .iter()
        .filter_map(|&feature| decode_current_piece_square_feature(feature))
        .map(|piece| (piece.piece_index, piece.rank * BOARD_FILES + piece.file))
        .collect::<Vec<_>>();
    let position = Position::from_canonical_piece_squares(&pieces);
    (position.has_general(Color::Red) && position.has_general(Color::Black)).then_some(position)
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

#[cfg(test)]
mod pair_compaction_tests {
    use super::*;

    #[test]
    fn pair_rows_only_share_strict_reciprocals_and_exclude_duplicate_generals() {
        let map = threat_relation_map();
        let mut keys = vec![Vec::new(); VALUE_THREAT_PAIR_VOCAB];
        let mut dead = 0;
        for attacker in 0..14 {
            for source in 0..BOARD_SIZE {
                for target in 0..BOARD_SIZE {
                    let base = map.bases[(attacker * BOARD_SIZE + source) * BOARD_SIZE + target];
                    if base == u16::MAX {
                        continue;
                    }
                    for victim in 0..14 {
                        let offset = threat_attacked_offsets()[target * 14 + victim];
                        if offset == u8::MAX {
                            continue;
                        }
                        let index = map.compact[usize::from(base) + usize::from(offset)];
                        if index == u16::MAX {
                            assert_eq!(attacker, victim);
                            assert_eq!(attacker % 7, 0);
                            dead += 1;
                        } else {
                            keys[index as usize].push((attacker, source, target, victim));
                        }
                    }
                }
            }
        }
        assert_eq!(dead, 48);
        let mut shared = 0;
        for group in keys {
            match group.as_slice() {
                [_] => {}
                &[(a, s, t, v), (b, u, w, p)] => {
                    assert_eq!((a, s, t, v), (p, w, u, b));
                    assert_ne!(a % 7, 3);
                    assert_ne!(v % 7, 3);
                    shared += 1;
                }
                _ => panic!("配对表有空行或非互反别名: {group:?}"),
            }
        }
        assert_eq!(shared, 155);
    }

    #[test]
    fn shared_pairs_preserve_pool_sum_multiplicity_and_source_tags() {
        let map = threat_relation_map();
        // 合成未共享表，再取各共享组的平均。仅用于检验代数等价，不涉及权重文件。
        let original: Vec<_> = (0..map.compact.len())
            .map(|i| (i % 41) as f64 - 20.0)
            .collect();
        let mut merged = vec![(0.0, 0); VALUE_THREAT_PAIR_VOCAB];
        for (&weight, &row) in original.iter().zip(&map.compact) {
            if row != u16::MAX {
                merged[row as usize].0 += weight;
                merged[row as usize].1 += 1;
            }
        }
        let weights: Vec<_> = merged
            .iter()
            .map(|&(sum, count)| sum / f64::from(count))
            .collect();
        let mut rng = SplitMix64::new(20261028);
        let mut duplicated = 0;
        let horse =
            Position::from_fen("2baka3/4n4/4b4/p2np3p/6P2/2B6/2r1P3P/C3BA3/4N4/4KAR2 w - - 0 1")
                .unwrap();
        for mut position in [Position::startpos(), horse] {
            for _ in 0..200 {
                for side in [Color::Red, Color::Black] {
                    let mut expected = Vec::new();
                    let mut sum = 0.0;
                    position.visit_occupied_leaper_relations_from(
                        u128::MAX,
                        |source, attacker, target, victim| {
                            let a = piece_absolute_feature_index(side, attacker);
                            let v = piece_absolute_feature_index(side, victim);
                            let s = canonical_square_for(side, source);
                            let t = canonical_square_for(side, target);
                            let base = map.bases[(a * BOARD_SIZE + s) * BOARD_SIZE + t];
                            assert_ne!(base, u16::MAX);
                            let raw = usize::from(base)
                                + usize::from(threat_attacked_offsets()[t * 14 + v]);
                            let row = map.compact[raw];
                            assert_ne!(row, u16::MAX);
                            expected.push(((source as u32) << 16) | u32::from(row));
                            sum += original[raw];
                        },
                    );
                    let mut actual = Vec::new();
                    visit_value_threat_entries_from(&position, side, u128::MAX, |entry| {
                        if entry >> 23 == 0 {
                            actual.push(entry);
                        }
                    });
                    assert_eq!(actual, expected);
                    let mut count = vec![0; VALUE_THREAT_PAIR_VOCAB];
                    let mut compact_sum = 0.0;
                    for entry in &actual {
                        let row = *entry as u16 as usize;
                        count[row] += 1;
                        compact_sum += weights[row];
                    }
                    for (row, &(.., group_size)) in merged.iter().enumerate() {
                        if group_size == 2 {
                            assert!(count[row] == 0 || count[row] == 2);
                            duplicated += usize::from(count[row] == 2);
                        }
                    }
                    // 和与数量都相同，因此 /sqrt(N) 及后续平方特征也相同。
                    assert_eq!(compact_sum, sum, "{} {side:?}", position.to_fen());
                }
                let moves = position.legal_moves();
                if moves.is_empty() {
                    break;
                }
                position.make_move(moves[rng.next_u64() as usize % moves.len()]);
            }
        }
        assert!(duplicated > 100);
    }
}
