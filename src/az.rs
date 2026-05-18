use std::fs;
use std::io::{self, BufWriter, Cursor, Read, Write};
use std::path::Path;

mod alphazero;
#[cfg(feature = "distill")]
mod distill;
mod mctx;
mod play;
mod replay;
mod train;
mod train_gpu;

use crate::nnue::{
    HISTORY_PLIES, HistoryMove, V4_INPUT_SIZE, canonical_move, canonical_square,
    extract_sparse_features_v4_canonical,
};
use crate::xiangqi::{BOARD_FILES, BOARD_RANKS, BOARD_SIZE, Color, Move, PieceKind, Position};

pub use alphazero::{
    AzCandidate, AzSearchAlgorithm, AzSearchLimits, AzSearchResult, alphazero_search,
    alphazero_search_with_history_and_rules,
};
#[cfg(feature = "distill")]
pub use distill::{AzDistillLoadOptions, AzDistillLoadStats, load_distill_npz_samples};
pub use mctx::AzGumbelConfig;
pub use play::{
    AzArenaConfig, AzArenaReport, AzSelfplayData, AzTerminalStats, generate_selfplay_data,
    play_arena_games_from_positions,
};
pub use replay::AzExperiencePool;
pub use train::{global_training_step_sample_count, train_samples, train_samples_weighted};

pub const AZNNUE_BINARY_MAGIC: &[u8] = b"AZB1";
// v37：value head 增加 value adapter + 第二层 MLP。Value 仍只读取共享
//   hidden/node_global，不引入任何手工战术特征。
// v36：policy global condition 增加一层小 MLP，上游共享表征先映射到
//   policy_context，再输出到 move-feature condition。
// v35：给每层 Local/Row/Col 图聚合增加 per-channel 可训练 gate/bias。
// v34：把 trunk 4 个容量旋钮（gnn_node_channels/layers, value_hidden_size, policy_node_proj_size）
//   从编译期常量改为 nnue 二进制头里携带的运行时字段。旧 v33 文件不再兼容。
const AZNNUE_BINARY_VERSION: u32 = 37;
// 头部布局（小端 u32 依次）：
//   magic(4 字节) | version | input_size | hidden_size | gnn_node_channels |
//   gnn_node_layers | value_hidden_size | policy_node_proj_size
const AZNNUE_BINARY_HEADER_LEN: usize = 4 + 4 * 7;

fn write_f32_slice_le<W: Write>(writer: &mut W, slice: &[f32]) -> io::Result<()> {
    for &value in slice {
        writer.write_all(&value.to_bits().to_le_bytes())?;
    }
    Ok(())
}

fn read_u32_le(reader: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_f32_vec_le(reader: &mut impl Read, len: usize) -> io::Result<Vec<f32>> {
    let mut out = vec![0.0f32; len];
    let mut buf = [0u8; 4];
    for slot in &mut out {
        reader.read_exact(&mut buf)?;
        *slot = f32::from_bits(u32::from_le_bytes(buf));
    }
    Ok(out)
}

const SPARSE_MOVE_SPACE: usize = BOARD_SIZE * BOARD_SIZE;
const DENSE_MOVE_SPACE: usize = compute_dense_move_count();
// v34 起，下列原 trunk 常量已改为 AzNnue 上的运行时字段，但仍提供同名 DEFAULT_*
//   作为 AzNnueArch::default() 与配置文件默认值的来源，保持与 v33 行为一致。
pub(super) const DEFAULT_GNN_NODE_CHANNELS: usize = 32;
pub(super) const DEFAULT_GNN_NODE_LAYERS: usize = 2;
pub(super) const DEFAULT_VALUE_HIDDEN_SIZE: usize = 256;
pub(super) const DEFAULT_POLICY_NODE_PROJ_SIZE: usize = 32;
const VALUE_LOGITS: usize = 3;
pub(super) const BOARD_PLANES_SIZE: usize = BOARD_SIZE;
pub(super) const BOARD_HISTORY_FRAMES: usize = HISTORY_PLIES + 1;
pub(super) const BOARD_HISTORY_SIZE: usize = BOARD_HISTORY_FRAMES * BOARD_PLANES_SIZE;
pub(super) const PIECE_BOARD_CHANNELS: usize = 14;
pub(super) const BOARD_CHANNELS: usize = PIECE_BOARD_CHANNELS * BOARD_HISTORY_FRAMES;
pub(super) const BOARD_INPUT_KERNEL_AREA: usize = 1;
// pool_node_features 内部硬编码 6 种聚合（mean/max/attn/row/col/std），
//   该常量与代码强耦合，不开放为配置项。
pub(super) const NODE_POOL_BLOCKS: usize = 6;
pub(super) const POLICY_CONTEXT_SIZE: usize = 64;
pub(super) const POLICY_CONDITION_SIZE: usize = 32;
const VALUE_SCALE_CP: f32 = 1000.0;
const RMS_NORM_EPS: f32 = 1.0e-6;

/// 模型 trunk 容量配置：与 hidden_size 一样，全部由配置/二进制头驱动。
/// 改这些值会改变模型形状，不能与已有 nnue 文件兼容，需要重新初始化。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AzNnueArch {
    /// 共享 hidden 向量宽度。
    pub hidden_size: usize,
    /// GNN 每层节点通道数（trunk 主要宽度旋钮）。
    pub gnn_node_channels: usize,
    /// GNN 聚合层数；每层带轻量 per-channel Local/Row/Col gate。
    pub gnn_node_layers: usize,
    /// value 头中间隐藏宽度。
    pub value_hidden_size: usize,
    /// policy node q/k 投影维度。
    pub policy_node_proj_size: usize,
}

impl AzNnueArch {
    /// 与 v33 行为一致的默认 trunk 容量。
    pub const fn default_const() -> Self {
        Self {
            hidden_size: 256,
            gnn_node_channels: DEFAULT_GNN_NODE_CHANNELS,
            gnn_node_layers: DEFAULT_GNN_NODE_LAYERS,
            value_hidden_size: DEFAULT_VALUE_HIDDEN_SIZE,
            policy_node_proj_size: DEFAULT_POLICY_NODE_PROJ_SIZE,
        }
    }

    /// 仅指定 hidden_size，其他 trunk 旋钮取默认。便于命令行/测试场景。
    pub const fn with_hidden_size(hidden_size: usize) -> Self {
        let mut arch = Self::default_const();
        arch.hidden_size = hidden_size;
        arch
    }

    /// 节点池尺寸（mean/max/attn/row/col/std 共 6 块）。
    pub const fn node_pool_size(&self) -> usize {
        self.gnn_node_channels * NODE_POOL_BLOCKS
    }

    /// 形状合法性自检：任一字段为 0 或层数 < 1 都视为非法配置。
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_size == 0 {
            return Err(format!("invalid hidden_size {}", self.hidden_size));
        }
        if self.gnn_node_channels == 0 {
            return Err(format!(
                "invalid gnn_node_channels {}",
                self.gnn_node_channels
            ));
        }
        if self.gnn_node_layers == 0 {
            return Err(format!(
                "invalid gnn_node_layers {} (must be >= 1)",
                self.gnn_node_layers
            ));
        }
        if self.value_hidden_size == 0 {
            return Err(format!(
                "invalid value_hidden_size {}",
                self.value_hidden_size
            ));
        }
        if self.policy_node_proj_size == 0 {
            return Err(format!(
                "invalid policy_node_proj_size {}",
                self.policy_node_proj_size
            ));
        }
        Ok(())
    }
}

impl Default for AzNnueArch {
    fn default() -> Self {
        Self::default_const()
    }
}
pub(super) struct AzEvalScratch {
    hidden: Vec<f32>,
    next: Vec<f32>,
    board: Vec<u8>,
    node_input: Vec<f32>,
    graph_tmp: Vec<f32>,
    policy_nodes: Vec<f32>,
    policy_q_nodes: Vec<f32>,
    policy_k_nodes: Vec<f32>,
    node_global: Vec<f32>,
    value_intermediate: Vec<f32>,
    value_hidden: Vec<f32>,
    value_logits: Vec<f32>,
    policy_context: Vec<f32>,
    policy_condition: Vec<f32>,
    logits: Vec<f32>,
    priors: Vec<f32>,
}

impl AzEvalScratch {
    /// 按当前 arch 一次性预分配前向缓冲。所有依赖 trunk 容量的尺寸都在此固化。
    pub(super) fn new(arch: AzNnueArch) -> Self {
        let hidden_size = arch.hidden_size;
        let gnn_channels = arch.gnn_node_channels;
        let proj = arch.policy_node_proj_size;
        let pool = arch.node_pool_size();
        Self {
            hidden: vec![0.0; hidden_size],
            next: vec![0.0; hidden_size],
            board: vec![0; BOARD_HISTORY_SIZE],
            node_input: vec![0.0; gnn_channels * BOARD_PLANES_SIZE],
            graph_tmp: vec![0.0; gnn_channels * BOARD_PLANES_SIZE],
            policy_nodes: vec![0.0; gnn_channels * BOARD_PLANES_SIZE],
            policy_q_nodes: vec![0.0; proj * BOARD_PLANES_SIZE],
            policy_k_nodes: vec![0.0; proj * BOARD_PLANES_SIZE],
            node_global: vec![0.0; pool],
            value_intermediate: vec![0.0; arch.value_hidden_size],
            value_hidden: vec![0.0; arch.value_hidden_size],
            value_logits: vec![0.0; VALUE_LOGITS],
            policy_context: vec![0.0; POLICY_CONTEXT_SIZE],
            policy_condition: vec![0.0; POLICY_CONDITION_SIZE],
            logits: Vec::with_capacity(192),
            priors: Vec::with_capacity(192),
        }
    }
}

#[derive(Debug)]
pub struct AzNnue {
    pub hidden_size: usize,
    /// trunk 的可配置容量旋钮（与 hidden_size 一起决定模型形状）。
    /// `hidden_size` 字段保留用于历史调用方（uci/main 等）的兼容访问，
    /// 与 `arch.hidden_size` 始终保持同步。
    pub arch: AzNnueArch,
    pub input_hidden: Vec<f32>,
    pub hidden_bias: Vec<f32>,
    pub node_input_weights: Vec<f32>,
    pub node_input_bias: Vec<f32>,
    pub graph_self_gate: Vec<f32>,
    pub graph_local_gate: Vec<f32>,
    pub graph_row_gate: Vec<f32>,
    pub graph_col_gate: Vec<f32>,
    pub graph_bias: Vec<f32>,
    pub node_attention_query: Vec<f32>,
    pub node_hidden: Vec<f32>,
    pub node_hidden_bias: Vec<f32>,
    pub value_intermediate_hidden: Vec<f32>,
    pub value_intermediate_nodes: Vec<f32>,
    pub value_intermediate_bias: Vec<f32>,
    pub value_hidden_weights: Vec<f32>,
    pub value_hidden_nodes: Vec<f32>,
    pub value_hidden_bias: Vec<f32>,
    pub value_logits_weights: Vec<f32>,
    pub value_logits_bias: Vec<f32>,
    pub policy_node_query: Vec<f32>,
    pub policy_node_key: Vec<f32>,
    pub policy_from_bias: Vec<f32>,
    pub policy_to_bias: Vec<f32>,
    pub policy_move_bias: Vec<f32>,
    pub policy_context_hidden: Vec<f32>,
    pub policy_context_nodes: Vec<f32>,
    pub policy_context_bias: Vec<f32>,
    pub policy_feature_hidden: Vec<f32>,
    pub policy_feature_nodes: Vec<f32>,
    pub policy_feature_bias: Vec<f32>,
    gpu_trainer: Option<Box<train_gpu::GpuTrainer>>,
}

// Architecture notes:
// - v27 removes the independent value CNN / value NNUE branch. Value now reads
//   the same shared hidden representation used by policy, then applies only a
//   small MLP head. This keeps the evaluator cheaper and prevents value from
//   learning a separate board model from policy.
// - v28 adds parameter-free RMSNorm on the shared hidden vector before both
//   value and policy heads. The GNN node path keeps only fixed /3 aggregation
//   scaling to stay cheap during CPU MCTS.
// - v29 removes the dense per-move hidden table from policy. Policy is now
//   driven by GNN from/to/pair node terms plus small shared move-geometry
//   conditioning, which shares better across similar moves.
// - v30 removes the remaining dense per-move node-weight table. Policy now uses
//   shared from/to node projections: q(from) dot k(to), plus shared from/to
//   node bias terms, move geometry conditioning, and a small dense move bias.
//   v31 removes the old trainable 3x3 board convolution from the model file
//   entirely. The board path follows the ZeroForge-style cheap GNN pattern:
//   a per-square input channel projection feeds Local8 + Row + Col graph
//   aggregation, then node pooling feeds the shared hidden/value path.
// - v35 keeps that cheap aggregation but gives each layer/channel trainable
//   self/local/row/col gates and a bias. This is intentionally much lighter
//   than a learned 3x3 convolution while letting training choose how much each
//   information path matters per channel.
// - Do not add a from/to-square factorized policy head on top of this absolute
//   board representation. A previous v14/v15 experiment mixed absolute board
//   features with partially relative action-square sharing and immediately
//   produced a severe red/black bias in self-play. Factorized policy should
//   wait until the whole model has a consistent per-square or canonical view.
// - The board summary deliberately keeps cheap row/column line pools. This is
//   the safe part borrowed from GNN-style row/column attention: it helps long
//   rook/cannon/general files without changing action semantics.
// - v18 removes the expensive first 3x3 input-board convolution. The first
//   board layer is now a 1x1 channel mixer, then the second layer handles local
//   3x3 patterns. This keeps local tactics while cutting the worst early CPU
//   eval cost.
// - v18 adds a static geometry-conditioned policy residual. It shares policy
//   knowledge across similar move shapes without resurrecting the bad v14/v15
//   from/to factorization or changing the absolute board view.
// - v20 removes the old policy residual trunk completely. It was not a useful
//   runtime knob: width, cheap board summaries, and move-conditioned policy
//   sharing were more valuable than hidden->hidden depth for this CPU MCTS net.
// - v21 tried a value-only relation/move encoder. It helped loss in places but
//   made the value experiment muddy: hand summaries can become shortcuts and
//   hide whether the board model itself understands positions.
// - v22 switches the net to side-to-move canonical inputs and canonical policy
//   actions: the mover is always represented as Red, and Black-to-move boards
//   are rotated 180 degrees with colors swapped. The old side-to-move board
//   channel is removed; do not feed canonical boards with absolute move labels,
//   because that recreates the red/black leakage bug.
// - v26 changes policy scoring to a GNN-style node head. The policy board CNN
//   becomes the node encoder, then Local/Row/Col graph aggregation builds
//   per-square features. Action logits use from-node, to-node, pair, and global
//   terms.

impl Clone for AzNnue {
    fn clone(&self) -> Self {
        Self {
            hidden_size: self.hidden_size,
            arch: self.arch,
            input_hidden: self.input_hidden.clone(),
            hidden_bias: self.hidden_bias.clone(),
            node_input_weights: self.node_input_weights.clone(),
            node_input_bias: self.node_input_bias.clone(),
            graph_self_gate: self.graph_self_gate.clone(),
            graph_local_gate: self.graph_local_gate.clone(),
            graph_row_gate: self.graph_row_gate.clone(),
            graph_col_gate: self.graph_col_gate.clone(),
            graph_bias: self.graph_bias.clone(),
            node_attention_query: self.node_attention_query.clone(),
            node_hidden: self.node_hidden.clone(),
            node_hidden_bias: self.node_hidden_bias.clone(),
            value_intermediate_hidden: self.value_intermediate_hidden.clone(),
            value_intermediate_nodes: self.value_intermediate_nodes.clone(),
            value_intermediate_bias: self.value_intermediate_bias.clone(),
            value_hidden_weights: self.value_hidden_weights.clone(),
            value_hidden_nodes: self.value_hidden_nodes.clone(),
            value_hidden_bias: self.value_hidden_bias.clone(),
            value_logits_weights: self.value_logits_weights.clone(),
            value_logits_bias: self.value_logits_bias.clone(),
            policy_node_query: self.policy_node_query.clone(),
            policy_node_key: self.policy_node_key.clone(),
            policy_from_bias: self.policy_from_bias.clone(),
            policy_to_bias: self.policy_to_bias.clone(),
            policy_move_bias: self.policy_move_bias.clone(),
            policy_context_hidden: self.policy_context_hidden.clone(),
            policy_context_nodes: self.policy_context_nodes.clone(),
            policy_context_bias: self.policy_context_bias.clone(),
            policy_feature_hidden: self.policy_feature_hidden.clone(),
            policy_feature_nodes: self.policy_feature_nodes.clone(),
            policy_feature_bias: self.policy_feature_bias.clone(),
            gpu_trainer: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AzLoopConfig {
    pub games: usize,
    pub max_plies: usize,
    pub simulations: usize,
    pub seed: u64,
    pub workers: usize,
    pub temperature_start: f32,
    pub temperature_end: f32,
    pub temperature_decay_plies: usize,
    pub search_algorithm: AzSearchAlgorithm,
    pub cpuct: f32,
    pub root_dirichlet_alpha: f32,
    pub root_exploration_fraction: f32,
    pub gumbel: AzGumbelConfig,
    pub td_lambda: f32,
    pub mirror_probability: f32,
    pub opening_policy_smoothing_plies: usize,
    pub opening_policy_smoothing: f32,
}

#[derive(Clone, Debug)]
pub struct AzLoopReport {
    pub games: usize,
    pub samples: usize,
    pub red_wins: usize,
    pub black_wins: usize,
    pub draws: usize,
    pub avg_plies: f32,
    pub loss: f32,
    pub value_loss: f32,
    pub value_mse: f32,
    pub value_pred_mean: f32,
    pub value_target_mean: f32,
    pub policy_ce: f32,
    /// 根 visit 策略分布熵（全局半步均值），TensorBoard：`stats/root_visit_entropy`。
    pub root_visit_entropy: f32,
    /// 开局阶段熵均值，`stats/entropy_opening`。
    pub entropy_opening: f32,
    /// 中后盘熵均值，`stats/entropy_mid`。
    pub entropy_mid: f32,
    pub selfplay_seconds: f32,
    pub train_seconds: f32,
    pub total_seconds: f32,
    pub games_per_second: f32,
    pub samples_per_second: f32,
    pub train_samples_per_second: f32,
    pub train_samples: usize,
    pub pool_games: usize,
    pub pool_samples: usize,
    pub terminal_no_legal_moves: usize,
    pub terminal_red_general_missing: usize,
    pub terminal_black_general_missing: usize,
    pub terminal_rule_draw: usize,
    pub terminal_rule_draw_halfmove120: usize,
    pub terminal_rule_draw_repetition: usize,
    pub terminal_rule_draw_mutual_long_check: usize,
    pub terminal_rule_draw_mutual_long_chase: usize,
    pub terminal_rule_win_red: usize,
    pub terminal_rule_win_black: usize,
    pub terminal_max_plies: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AzTrainBenchmark {
    pub loss: f32,
    pub value_loss: f32,
    pub policy_ce: f32,
}

#[derive(Clone, Debug)]
pub struct AzTrainingSample {
    pub features: Vec<usize>,
    pub board: Vec<u8>,
    pub move_indices: Vec<usize>,
    pub policy: Vec<f32>,
    pub value: f32,
    pub side_sign: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AzTrainStats {
    pub loss: f32,
    pub value_loss: f32,
    pub policy_ce: f32,
    pub value_pred_sum: f32,
    pub value_pred_sq_sum: f32,
    pub value_target_sum: f32,
    pub value_target_sq_sum: f32,
    pub value_error_sq_sum: f32,
    pub samples: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct AzTrainLossWeights {
    pub value: f32,
    pub policy: f32,
    pub train_shared: bool,
    pub train_value_head: bool,
    pub train_policy_head: bool,
}

impl Default for AzTrainLossWeights {
    fn default() -> Self {
        Self {
            value: 1.0,
            policy: 1.0,
            train_shared: true,
            train_value_head: true,
            train_policy_head: true,
        }
    }
}

impl AzTrainStats {
    fn add_assign(&mut self, other: &Self) {
        self.loss += other.loss;
        self.value_loss += other.value_loss;
        self.policy_ce += other.policy_ce;
        self.value_pred_sum += other.value_pred_sum;
        self.value_pred_sq_sum += other.value_pred_sq_sum;
        self.value_target_sum += other.value_target_sum;
        self.value_target_sq_sum += other.value_target_sq_sum;
        self.value_error_sq_sum += other.value_error_sq_sum;
        self.samples += other.samples;
    }
}

impl AzNnue {
    /// 用指定 trunk 容量随机初始化模型。所有形状均由 `arch` 驱动。
    pub fn random_with_arch(arch: AzNnueArch, seed: u64) -> Self {
        if let Err(err) = arch.validate() {
            panic!("AzNnue::random_with_arch: invalid arch ({err})");
        }
        let hidden_size = arch.hidden_size;
        let gnn_channels = arch.gnn_node_channels;
        let value_hidden = arch.value_hidden_size;
        let proj = arch.policy_node_proj_size;
        let pool = arch.node_pool_size();
        let mut rng = SplitMix64::new(seed);
        let input_hidden: Vec<f32> = (0..V4_INPUT_SIZE * hidden_size)
            .map(|_| rng.weight(0.015))
            .collect();
        let hidden_bias = vec![0.0; hidden_size];
        let node_input_weights: Vec<f32> =
            (0..gnn_channels * BOARD_CHANNELS * BOARD_INPUT_KERNEL_AREA)
                .map(|_| rng.weight(0.08))
                .collect();
        let node_input_bias = vec![0.0; gnn_channels];
        let graph_gate_len = arch.gnn_node_layers * gnn_channels;
        let graph_self_gate = vec![0.25; graph_gate_len];
        let graph_local_gate = vec![0.25; graph_gate_len];
        let graph_row_gate = vec![0.25; graph_gate_len];
        let graph_col_gate = vec![0.25; graph_gate_len];
        let graph_bias = vec![0.0; graph_gate_len];
        let node_attention_query: Vec<f32> = (0..gnn_channels).map(|_| rng.weight(0.08)).collect();
        let node_hidden: Vec<f32> = (0..hidden_size * pool)
            .map(|_| rng.weight((2.0 / pool as f32).sqrt()))
            .collect();
        let node_hidden_bias = vec![0.0; hidden_size];
        let value_intermediate_hidden = (0..value_hidden * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt()))
            .collect();
        let value_intermediate_nodes = (0..value_hidden * pool)
            .map(|_| rng.weight((2.0 / pool as f32).sqrt() * 0.5))
            .collect();
        let value_intermediate_bias = vec![0.0; value_hidden];
        let value_hidden_weights = (0..value_hidden * value_hidden)
            .map(|_| rng.weight((2.0 / value_hidden as f32).sqrt() * 0.5))
            .collect();
        let value_hidden_nodes = (0..value_hidden * pool)
            .map(|_| rng.weight((2.0 / pool as f32).sqrt() * 0.25))
            .collect();
        let value_hidden_bias = vec![0.0; value_hidden];
        let value_logits_weights = (0..VALUE_LOGITS * value_hidden)
            .map(|_| rng.weight((2.0 / value_hidden as f32).sqrt()))
            .collect();
        let value_logits_bias = vec![0.0; VALUE_LOGITS];
        let policy_node_query = (0..proj * gnn_channels)
            .map(|_| rng.weight((2.0 / gnn_channels as f32).sqrt() * 0.5))
            .collect();
        let policy_node_key = (0..proj * gnn_channels)
            .map(|_| rng.weight((2.0 / gnn_channels as f32).sqrt() * 0.5))
            .collect();
        let policy_from_bias = (0..gnn_channels).map(|_| rng.weight(0.01)).collect();
        let policy_to_bias = (0..gnn_channels).map(|_| rng.weight(0.01)).collect();
        let policy_move_bias = vec![0.0; DENSE_MOVE_SPACE];
        let policy_context_hidden = (0..POLICY_CONTEXT_SIZE * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt() * 0.5))
            .collect();
        let policy_context_nodes = (0..POLICY_CONTEXT_SIZE * pool)
            .map(|_| rng.weight((2.0 / pool as f32).sqrt() * 0.5))
            .collect();
        let policy_context_bias = vec![0.0; POLICY_CONTEXT_SIZE];
        let policy_feature_hidden = (0..POLICY_CONDITION_SIZE * POLICY_CONTEXT_SIZE)
            .map(|_| rng.weight((2.0 / POLICY_CONTEXT_SIZE as f32).sqrt() * 0.5))
            .collect();
        let policy_feature_nodes = (0..POLICY_CONDITION_SIZE * pool)
            .map(|_| rng.weight((2.0 / pool as f32).sqrt() * 0.5))
            .collect();
        let policy_feature_bias = vec![0.0; POLICY_CONDITION_SIZE];
        Self {
            hidden_size,
            arch,
            input_hidden,
            hidden_bias,
            node_input_weights,
            node_input_bias,
            graph_self_gate,
            graph_local_gate,
            graph_row_gate,
            graph_col_gate,
            graph_bias,
            node_attention_query,
            node_hidden,
            node_hidden_bias,
            value_intermediate_hidden,
            value_intermediate_nodes,
            value_intermediate_bias,
            value_hidden_weights,
            value_hidden_nodes,
            value_hidden_bias,
            value_logits_weights,
            value_logits_bias,
            policy_node_query,
            policy_node_key,
            policy_from_bias,
            policy_to_bias,
            policy_move_bias,
            policy_context_hidden,
            policy_context_nodes,
            policy_context_bias,
            policy_feature_hidden,
            policy_feature_nodes,
            policy_feature_bias,
            gpu_trainer: None,
        }
    }

    /// 仅指定 hidden_size 的便捷构造（其余 trunk 旋钮取默认）。
    pub fn random(hidden_size: usize, seed: u64) -> Self {
        Self::random_with_arch(AzNnueArch::with_hidden_size(hidden_size), seed)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let file = fs::File::create(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(AZNNUE_BINARY_MAGIC)?;
        writer.write_all(&AZNNUE_BINARY_VERSION.to_le_bytes())?;
        writer.write_all(&(V4_INPUT_SIZE as u32).to_le_bytes())?;
        writer.write_all(&(self.arch.hidden_size as u32).to_le_bytes())?;
        writer.write_all(&(self.arch.gnn_node_channels as u32).to_le_bytes())?;
        writer.write_all(&(self.arch.gnn_node_layers as u32).to_le_bytes())?;
        writer.write_all(&(self.arch.value_hidden_size as u32).to_le_bytes())?;
        writer.write_all(&(self.arch.policy_node_proj_size as u32).to_le_bytes())?;
        write_f32_slice_le(&mut writer, &self.input_hidden)?;
        write_f32_slice_le(&mut writer, &self.hidden_bias)?;
        write_f32_slice_le(&mut writer, &self.node_input_weights)?;
        write_f32_slice_le(&mut writer, &self.node_input_bias)?;
        write_f32_slice_le(&mut writer, &self.graph_self_gate)?;
        write_f32_slice_le(&mut writer, &self.graph_local_gate)?;
        write_f32_slice_le(&mut writer, &self.graph_row_gate)?;
        write_f32_slice_le(&mut writer, &self.graph_col_gate)?;
        write_f32_slice_le(&mut writer, &self.graph_bias)?;
        write_f32_slice_le(&mut writer, &self.node_attention_query)?;
        write_f32_slice_le(&mut writer, &self.node_hidden)?;
        write_f32_slice_le(&mut writer, &self.node_hidden_bias)?;
        write_f32_slice_le(&mut writer, &self.value_intermediate_hidden)?;
        write_f32_slice_le(&mut writer, &self.value_intermediate_nodes)?;
        write_f32_slice_le(&mut writer, &self.value_intermediate_bias)?;
        write_f32_slice_le(&mut writer, &self.value_hidden_weights)?;
        write_f32_slice_le(&mut writer, &self.value_hidden_nodes)?;
        write_f32_slice_le(&mut writer, &self.value_hidden_bias)?;
        write_f32_slice_le(&mut writer, &self.value_logits_weights)?;
        write_f32_slice_le(&mut writer, &self.value_logits_bias)?;
        write_f32_slice_le(&mut writer, &self.policy_node_query)?;
        write_f32_slice_le(&mut writer, &self.policy_node_key)?;
        write_f32_slice_le(&mut writer, &self.policy_from_bias)?;
        write_f32_slice_le(&mut writer, &self.policy_to_bias)?;
        write_f32_slice_le(&mut writer, &self.policy_move_bias)?;
        write_f32_slice_le(&mut writer, &self.policy_context_hidden)?;
        write_f32_slice_le(&mut writer, &self.policy_context_nodes)?;
        write_f32_slice_le(&mut writer, &self.policy_context_bias)?;
        write_f32_slice_le(&mut writer, &self.policy_feature_hidden)?;
        write_f32_slice_le(&mut writer, &self.policy_feature_nodes)?;
        write_f32_slice_le(&mut writer, &self.policy_feature_bias)?;
        writer.flush()?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let bytes = fs::read(path.as_ref())?;
        Self::decode_binary(&bytes)
    }

    fn decode_binary(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < AZNNUE_BINARY_HEADER_LEN || !bytes.starts_with(AZNNUE_BINARY_MAGIC) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "truncated or invalid AZNNUE binary",
            ));
        }
        let mut reader = Cursor::new(&bytes[AZNNUE_BINARY_MAGIC.len()..]);
        let version = read_u32_le(&mut reader)?;
        if version != AZNNUE_BINARY_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported AZNNUE binary version {version} (expected {AZNNUE_BINARY_VERSION})"
                ),
            ));
        }
        let input_size = read_u32_le(&mut reader)? as usize;
        let hidden_size = read_u32_le(&mut reader)? as usize;
        let gnn_node_channels = read_u32_le(&mut reader)? as usize;
        let gnn_node_layers = read_u32_le(&mut reader)? as usize;
        let value_hidden_size = read_u32_le(&mut reader)? as usize;
        let policy_node_proj_size = read_u32_le(&mut reader)? as usize;
        if input_size != V4_INPUT_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "binary input_size does not match this build (V4_INPUT_SIZE)",
            ));
        }
        let arch = AzNnueArch {
            hidden_size,
            gnn_node_channels,
            gnn_node_layers,
            value_hidden_size,
            policy_node_proj_size,
        };
        if let Err(err) = arch.validate() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("AZNNUE binary header arch invalid: {err}"),
            ));
        }
        let pool_size = arch.node_pool_size();
        let input_hidden_len = V4_INPUT_SIZE * hidden_size;
        let hidden_bias_len = hidden_size;
        let node_input_weights_len = gnn_node_channels * BOARD_CHANNELS * BOARD_INPUT_KERNEL_AREA;
        let node_input_bias_len = gnn_node_channels;
        let graph_gate_len = gnn_node_layers * gnn_node_channels;
        let node_attention_query_len = gnn_node_channels;
        let node_hidden_len = hidden_size * pool_size;
        let node_hidden_bias_len = hidden_size;
        let vih_len = value_hidden_size * hidden_size;
        let vin_len = value_hidden_size * pool_size;
        let vib_len = value_hidden_size;
        let vhw_len = value_hidden_size * value_hidden_size;
        let vhn_len = value_hidden_size * pool_size;
        let vhb_len = value_hidden_size;
        let vlw_len = VALUE_LOGITS * value_hidden_size;
        let vlb_len = VALUE_LOGITS;
        let pnq_len = policy_node_proj_size * gnn_node_channels;
        let pnk_len = policy_node_proj_size * gnn_node_channels;
        let pfbias_len = gnn_node_channels;
        let ptbias_len = gnn_node_channels;
        let pmb_len = DENSE_MOVE_SPACE;
        let pch_len = POLICY_CONTEXT_SIZE * hidden_size;
        let pcn_len = POLICY_CONTEXT_SIZE * pool_size;
        let pcb_len = POLICY_CONTEXT_SIZE;
        let pfh_len = POLICY_CONDITION_SIZE * POLICY_CONTEXT_SIZE;
        let pfc_len = POLICY_CONDITION_SIZE * pool_size;
        let pfb_len = POLICY_CONDITION_SIZE;
        let float_count = input_hidden_len
            + hidden_bias_len
            + node_input_weights_len
            + node_input_bias_len
            + graph_gate_len * 5
            + node_attention_query_len
            + node_hidden_len
            + node_hidden_bias_len
            + vih_len
            + vin_len
            + vib_len
            + vhw_len
            + vhn_len
            + vhb_len
            + vlw_len
            + vlb_len
            + pnq_len
            + pnk_len
            + pfbias_len
            + ptbias_len
            + pmb_len
            + pch_len
            + pcn_len
            + pcb_len
            + pfh_len
            + pfc_len
            + pfb_len;
        let expected_len = AZNNUE_BINARY_HEADER_LEN + float_count * 4;
        if bytes.len() != expected_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "AZNNUE binary size mismatch: got {} bytes, expected {} (floats {})",
                    bytes.len(),
                    expected_len,
                    float_count
                ),
            ));
        }
        let input_hidden = read_f32_vec_le(&mut reader, input_hidden_len)?;
        let hidden_bias = read_f32_vec_le(&mut reader, hidden_bias_len)?;
        let node_input_weights = read_f32_vec_le(&mut reader, node_input_weights_len)?;
        let node_input_bias = read_f32_vec_le(&mut reader, node_input_bias_len)?;
        let graph_self_gate = read_f32_vec_le(&mut reader, graph_gate_len)?;
        let graph_local_gate = read_f32_vec_le(&mut reader, graph_gate_len)?;
        let graph_row_gate = read_f32_vec_le(&mut reader, graph_gate_len)?;
        let graph_col_gate = read_f32_vec_le(&mut reader, graph_gate_len)?;
        let graph_bias = read_f32_vec_le(&mut reader, graph_gate_len)?;
        let node_attention_query = read_f32_vec_le(&mut reader, node_attention_query_len)?;
        let node_hidden = read_f32_vec_le(&mut reader, node_hidden_len)?;
        let node_hidden_bias = read_f32_vec_le(&mut reader, node_hidden_bias_len)?;
        let value_intermediate_hidden = read_f32_vec_le(&mut reader, vih_len)?;
        let value_intermediate_nodes = read_f32_vec_le(&mut reader, vin_len)?;
        let value_intermediate_bias = read_f32_vec_le(&mut reader, vib_len)?;
        let value_hidden_weights = read_f32_vec_le(&mut reader, vhw_len)?;
        let value_hidden_nodes = read_f32_vec_le(&mut reader, vhn_len)?;
        let value_hidden_bias = read_f32_vec_le(&mut reader, vhb_len)?;
        let value_logits_weights = read_f32_vec_le(&mut reader, vlw_len)?;
        let value_logits_bias = read_f32_vec_le(&mut reader, vlb_len)?;
        let policy_node_query = read_f32_vec_le(&mut reader, pnq_len)?;
        let policy_node_key = read_f32_vec_le(&mut reader, pnk_len)?;
        let policy_from_bias = read_f32_vec_le(&mut reader, pfbias_len)?;
        let policy_to_bias = read_f32_vec_le(&mut reader, ptbias_len)?;
        let policy_move_bias = read_f32_vec_le(&mut reader, pmb_len)?;
        let policy_context_hidden = read_f32_vec_le(&mut reader, pch_len)?;
        let policy_context_nodes = read_f32_vec_le(&mut reader, pcn_len)?;
        let policy_context_bias = read_f32_vec_le(&mut reader, pcb_len)?;
        let policy_feature_hidden = read_f32_vec_le(&mut reader, pfh_len)?;
        let policy_feature_nodes = read_f32_vec_le(&mut reader, pfc_len)?;
        let policy_feature_bias = read_f32_vec_le(&mut reader, pfb_len)?;
        let model = Self {
            hidden_size,
            arch,
            input_hidden,
            hidden_bias,
            node_input_weights,
            node_input_bias,
            graph_self_gate,
            graph_local_gate,
            graph_row_gate,
            graph_col_gate,
            graph_bias,
            node_attention_query,
            node_hidden,
            node_hidden_bias,
            value_intermediate_hidden,
            value_intermediate_nodes,
            value_intermediate_bias,
            value_hidden_weights,
            value_hidden_nodes,
            value_hidden_bias,
            value_logits_weights,
            value_logits_bias,
            policy_node_query,
            policy_node_key,
            policy_from_bias,
            policy_to_bias,
            policy_move_bias,
            policy_context_hidden,
            policy_context_nodes,
            policy_context_bias,
            policy_feature_hidden,
            policy_feature_nodes,
            policy_feature_bias,
            gpu_trainer: None,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn evaluate_value(
        &self,
        position: &Position,
        history: &[HistoryMove],
        moves: &[Move],
    ) -> f32 {
        let mut scratch = AzEvalScratch::new(self.arch);
        self.evaluate_with_scratch(position, history, moves, &mut scratch)
    }

    pub(super) fn evaluate_with_scratch(
        &self,
        position: &Position,
        history: &[HistoryMove],
        moves: &[Move],
        scratch: &mut AzEvalScratch,
    ) -> f32 {
        let side = position.side_to_move();
        let features = extract_sparse_features_v4_canonical(position, history);
        extract_board_planes(position, history, &mut scratch.board);
        self.input_embedding_into(&features, &mut scratch.hidden);
        self.board_forward_into(
            &scratch.board,
            &mut scratch.node_input,
            &mut scratch.graph_tmp,
            &mut scratch.policy_nodes,
            &mut scratch.node_global,
        );
        self.node_hidden_into(&scratch.node_global, &mut scratch.next);
        for idx in 0..self.hidden_size {
            scratch.hidden[idx] += scratch.next[idx];
        }
        rms_norm_in_place(&mut scratch.hidden);
        self.policy_node_projection_into(
            &scratch.policy_nodes,
            &self.policy_node_query,
            &mut scratch.policy_q_nodes,
        );
        self.policy_node_projection_into(
            &scratch.policy_nodes,
            &self.policy_node_key,
            &mut scratch.policy_k_nodes,
        );
        let value = self.value_from_hidden_into(
            &scratch.hidden,
            &scratch.node_global,
            &mut scratch.value_intermediate,
            &mut scratch.value_hidden,
            &mut scratch.value_logits,
        );
        self.policy_condition_into(
            &scratch.hidden,
            &scratch.node_global,
            &mut scratch.policy_context,
            &mut scratch.policy_condition,
        );
        scratch.logits.resize(moves.len(), 0.0);
        for (index, mv) in moves.iter().enumerate() {
            scratch.logits[index] = self.policy_logit_from_hidden_index(
                &scratch.policy_nodes,
                &scratch.policy_q_nodes,
                &scratch.policy_k_nodes,
                &scratch.policy_condition,
                dense_move_index(canonical_move(side, *mv)),
            );
        }
        value
    }

    fn input_embedding_into(&self, features: &[usize], hidden: &mut Vec<f32>) {
        hidden.resize(self.hidden_size, 0.0);
        hidden.copy_from_slice(&self.hidden_bias);
        for &feature in features {
            let row =
                &self.input_hidden[feature * self.hidden_size..(feature + 1) * self.hidden_size];
            for idx in 0..self.hidden_size {
                hidden[idx] += row[idx];
            }
        }
        for value in hidden.iter_mut() {
            *value = value.max(0.0);
        }
    }

    fn node_hidden_into(&self, node_global: &[f32], hidden: &mut Vec<f32>) {
        let pool = self.arch.node_pool_size();
        hidden.resize(self.hidden_size, 0.0);
        hidden.copy_from_slice(&self.node_hidden_bias);
        for (out, hidden_value) in hidden.iter_mut().enumerate().take(self.hidden_size) {
            let row = &self.node_hidden[out * pool..(out + 1) * pool];
            for (node_value, weight) in node_global.iter().zip(row) {
                *hidden_value += node_value * weight;
            }
            *hidden_value = (*hidden_value).max(0.0);
        }
    }

    fn board_forward_into(
        &self,
        board: &[u8],
        node_input: &mut Vec<f32>,
        graph_tmp: &mut Vec<f32>,
        policy_nodes: &mut Vec<f32>,
        node_global: &mut Vec<f32>,
    ) {
        let channels = self.arch.gnn_node_channels;
        let pool = self.arch.node_pool_size();
        let layers = self.arch.gnn_node_layers;
        node_input.resize(channels * BOARD_PLANES_SIZE, 0.0);
        graph_tmp.resize(channels * BOARD_PLANES_SIZE, 0.0);
        node_global.resize(pool, 0.0);
        node_input_layer_into(
            board,
            channels,
            &self.node_input_weights,
            &self.node_input_bias,
            node_input,
        );
        // 至少跑一次图聚合（layers >= 1，已在 arch.validate 校验）。
        graph_node_layer_into(node_input, graph_tmp, self, 0);
        if layers == 1 {
            policy_nodes.resize(channels * BOARD_PLANES_SIZE, 0.0);
            policy_nodes.copy_from_slice(graph_tmp);
        } else {
            graph_node_layer_into(graph_tmp, policy_nodes, self, 1);
            // 第 3 层及以后：在 policy_nodes 与 graph_tmp 之间反复聚合。
            for layer in 2..layers {
                graph_node_layer_into(policy_nodes, graph_tmp, self, layer);
                policy_nodes.resize(channels * BOARD_PLANES_SIZE, 0.0);
                policy_nodes.copy_from_slice(graph_tmp);
            }
        }
        pool_node_features(
            policy_nodes,
            channels,
            &self.node_attention_query,
            node_global,
        );
    }

    fn value_from_hidden_into(
        &self,
        hidden: &[f32],
        node_global: &[f32],
        value_intermediate: &mut Vec<f32>,
        value_hidden_out: &mut Vec<f32>,
        value_logits: &mut Vec<f32>,
    ) -> f32 {
        let value_hidden = self.arch.value_hidden_size;
        let pool = self.arch.node_pool_size();
        value_intermediate.copy_from_slice(&self.value_intermediate_bias);
        for (j, value) in value_intermediate.iter_mut().enumerate().take(value_hidden) {
            let h_row =
                &self.value_intermediate_hidden[j * self.hidden_size..(j + 1) * self.hidden_size];
            for (hidden_value, weight) in hidden.iter().zip(h_row) {
                *value += hidden_value * weight;
            }
            let node_row = &self.value_intermediate_nodes[j * pool..(j + 1) * pool];
            for (node_value, weight) in node_global.iter().zip(node_row) {
                *value += node_value * weight;
            }
            *value = (*value).max(0.0);
        }
        value_hidden_out.copy_from_slice(&self.value_hidden_bias);
        for (j, value) in value_hidden_out.iter_mut().enumerate().take(value_hidden) {
            let h_row = &self.value_hidden_weights[j * value_hidden..(j + 1) * value_hidden];
            for (intermediate, weight) in value_intermediate.iter().zip(h_row) {
                *value += intermediate * weight;
            }
            let node_row = &self.value_hidden_nodes[j * pool..(j + 1) * pool];
            for (node_value, weight) in node_global.iter().zip(node_row) {
                *value += node_value * weight;
            }
            *value = (*value).max(0.0);
        }
        value_logits.copy_from_slice(&self.value_logits_bias);
        for out in 0..VALUE_LOGITS {
            let row = &self.value_logits_weights[out * value_hidden..(out + 1) * value_hidden];
            for (value_hidden, weight) in value_hidden_out.iter().zip(row) {
                value_logits[out] += value_hidden * weight;
            }
        }
        scalar_value_from_logits(value_logits).0
    }

    fn policy_condition_into(
        &self,
        hidden: &[f32],
        node_global: &[f32],
        context: &mut Vec<f32>,
        out: &mut Vec<f32>,
    ) {
        let pool = self.arch.node_pool_size();
        context.resize(POLICY_CONTEXT_SIZE, 0.0);
        context.copy_from_slice(&self.policy_context_bias);
        for (feature, value) in context.iter_mut().enumerate().take(POLICY_CONTEXT_SIZE) {
            let hidden_row = &self.policy_context_hidden
                [feature * self.hidden_size..(feature + 1) * self.hidden_size];
            *value += dot_product(hidden, hidden_row);
            let node_row = &self.policy_context_nodes[feature * pool..(feature + 1) * pool];
            *value += dot_product(node_global, node_row);
            *value = (*value).max(0.0);
        }
        out.resize(POLICY_CONDITION_SIZE, 0.0);
        out.copy_from_slice(&self.policy_feature_bias);
        for (feature, value) in out.iter_mut().enumerate().take(POLICY_CONDITION_SIZE) {
            let hidden_row = &self.policy_feature_hidden
                [feature * POLICY_CONTEXT_SIZE..(feature + 1) * POLICY_CONTEXT_SIZE];
            *value += dot_product(context, hidden_row);
            let node_row = &self.policy_feature_nodes[feature * pool..(feature + 1) * pool];
            *value += dot_product(node_global, node_row);
        }
    }

    fn policy_node_projection_into(&self, nodes: &[f32], weights: &[f32], out: &mut Vec<f32>) {
        let proj = self.arch.policy_node_proj_size;
        let channels = self.arch.gnn_node_channels;
        out.resize(proj * BOARD_PLANES_SIZE, 0.0);
        for p in 0..proj {
            let weight_row = &weights[p * channels..(p + 1) * channels];
            for sq in 0..BOARD_PLANES_SIZE {
                let mut value = 0.0;
                for channel in 0..channels {
                    value += nodes[channel * BOARD_PLANES_SIZE + sq] * weight_row[channel];
                }
                out[p * BOARD_PLANES_SIZE + sq] = value;
            }
        }
    }

    fn validate(&self) -> io::Result<()> {
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
        let channels = arch.gnn_node_channels;
        let layers = arch.gnn_node_layers;
        let pool = arch.node_pool_size();
        let value_hidden = arch.value_hidden_size;
        let proj = arch.policy_node_proj_size;
        if self.input_hidden.len() != V4_INPUT_SIZE * hidden
            || self.hidden_bias.len() != hidden
            || self.node_input_weights.len() != channels * BOARD_CHANNELS * BOARD_INPUT_KERNEL_AREA
            || self.node_input_bias.len() != channels
            || self.graph_self_gate.len() != layers * channels
            || self.graph_local_gate.len() != layers * channels
            || self.graph_row_gate.len() != layers * channels
            || self.graph_col_gate.len() != layers * channels
            || self.graph_bias.len() != layers * channels
            || self.node_attention_query.len() != channels
            || self.node_hidden.len() != hidden * pool
            || self.node_hidden_bias.len() != hidden
            || self.value_intermediate_hidden.len() != value_hidden * hidden
            || self.value_intermediate_nodes.len() != value_hidden * pool
            || self.value_intermediate_bias.len() != value_hidden
            || self.value_hidden_weights.len() != value_hidden * value_hidden
            || self.value_hidden_nodes.len() != value_hidden * pool
            || self.value_hidden_bias.len() != value_hidden
            || self.value_logits_weights.len() != VALUE_LOGITS * value_hidden
            || self.value_logits_bias.len() != VALUE_LOGITS
            || self.policy_node_query.len() != proj * channels
            || self.policy_node_key.len() != proj * channels
            || self.policy_from_bias.len() != channels
            || self.policy_to_bias.len() != channels
            || self.policy_move_bias.len() != DENSE_MOVE_SPACE
            || self.policy_context_hidden.len() != POLICY_CONTEXT_SIZE * hidden
            || self.policy_context_nodes.len() != POLICY_CONTEXT_SIZE * pool
            || self.policy_context_bias.len() != POLICY_CONTEXT_SIZE
            || self.policy_feature_hidden.len() != POLICY_CONDITION_SIZE * POLICY_CONTEXT_SIZE
            || self.policy_feature_nodes.len() != POLICY_CONDITION_SIZE * pool
            || self.policy_feature_bias.len() != POLICY_CONDITION_SIZE
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "aznnue vector length mismatch",
            ));
        }
        Ok(())
    }

    fn policy_logit_from_hidden_index(
        &self,
        policy_nodes: &[f32],
        policy_q_nodes: &[f32],
        policy_k_nodes: &[f32],
        policy_condition: &[f32],
        move_index: usize,
    ) -> f32 {
        let proj = self.arch.policy_node_proj_size;
        let channels = self.arch.gnn_node_channels;
        let sparse = move_map().dense_to_sparse[move_index] as usize;
        let from = sparse / BOARD_SIZE;
        let to = sparse % BOARD_SIZE;
        let feature_offset = move_index * POLICY_CONDITION_SIZE;
        let move_features =
            &policy_move_features()[feature_offset..feature_offset + POLICY_CONDITION_SIZE];
        let mut logit =
            self.policy_move_bias[move_index] + dot_product(policy_condition, move_features);
        let mut pair = 0.0;
        for p in 0..proj {
            pair += policy_q_nodes[p * BOARD_PLANES_SIZE + from]
                * policy_k_nodes[p * BOARD_PLANES_SIZE + to];
        }
        logit += pair * (proj as f32).sqrt().recip();
        for channel in 0..channels {
            let from_value = policy_nodes[channel * BOARD_PLANES_SIZE + from];
            let to_value = policy_nodes[channel * BOARD_PLANES_SIZE + to];
            logit += from_value * self.policy_from_bias[channel];
            logit += to_value * self.policy_to_bias[channel];
        }
        logit
    }
}

pub(super) fn extract_board_planes(
    position: &Position,
    history: &[HistoryMove],
    board: &mut Vec<u8>,
) {
    let side = position.side_to_move();
    board.resize(BOARD_HISTORY_SIZE, 0);
    board.fill(0);
    let mut frame = [0u8; BOARD_PLANES_SIZE];
    extract_position_piece_planes(position, &mut frame);
    board[..BOARD_PLANES_SIZE].copy_from_slice(&frame);

    let mut rewound = frame;
    for (history_index, entry) in history.iter().rev().take(HISTORY_PLIES).enumerate() {
        let piece_plane =
            (canonical_piece_plane(side, entry.piece.color, entry.piece.kind) + 1) as u8;
        rewound[canonical_square(side, entry.mv.to as usize)] = entry.captured.map_or(0, |piece| {
            (canonical_piece_plane(side, piece.color, piece.kind) + 1) as u8
        });
        rewound[canonical_square(side, entry.mv.from as usize)] = piece_plane;
        let start = (history_index + 1) * BOARD_PLANES_SIZE;
        board[start..start + BOARD_PLANES_SIZE].copy_from_slice(&rewound);
    }
}

fn extract_position_piece_planes(position: &Position, board: &mut [u8; BOARD_PLANES_SIZE]) {
    board.fill(0);
    let side = position.side_to_move();
    for sq in 0..BOARD_SIZE {
        let Some(piece) = position.piece_at(sq) else {
            continue;
        };
        let plane = canonical_piece_plane(side, piece.color, piece.kind);
        board[canonical_square(side, sq)] = (plane + 1) as u8;
    }
}

fn absolute_piece_plane(piece_color: Color, kind: PieceKind) -> usize {
    let own_offset = if piece_color == Color::Red { 0 } else { 7 };
    own_offset
        + match kind {
            PieceKind::General => 0,
            PieceKind::Advisor => 1,
            PieceKind::Elephant => 2,
            PieceKind::Horse => 3,
            PieceKind::Rook => 4,
            PieceKind::Cannon => 5,
            PieceKind::Soldier => 6,
        }
}

fn canonical_piece_plane(side: Color, piece_color: Color, kind: PieceKind) -> usize {
    let canonical_color = if piece_color == side {
        Color::Red
    } else {
        Color::Black
    };
    absolute_piece_plane(canonical_color, kind)
}

fn node_input_layer_into(
    board: &[u8],
    output_channels: usize,
    weights: &[f32],
    bias: &[f32],
    output: &mut [f32],
) {
    debug_assert_eq!(BOARD_INPUT_KERNEL_AREA, 1);
    for (out_channel, bias_value) in bias.iter().copied().enumerate().take(output_channels) {
        let out_start = out_channel * BOARD_PLANES_SIZE;
        for sq in 0..BOARD_PLANES_SIZE {
            let mut value = bias_value;
            for frame in 0..BOARD_HISTORY_FRAMES {
                let plane = board[frame * BOARD_PLANES_SIZE + sq];
                if plane == 0 {
                    continue;
                }
                let in_channel = frame * PIECE_BOARD_CHANNELS + plane as usize - 1;
                debug_assert!(in_channel < BOARD_CHANNELS);
                value += weights[out_channel * BOARD_CHANNELS + in_channel];
            }
            output[out_start + sq] = value.max(0.0);
        }
    }
}

fn pool_node_features(input: &[f32], channels: usize, attention_query: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), channels * BOARD_PLANES_SIZE);
    debug_assert_eq!(attention_query.len(), channels);
    debug_assert_eq!(output.len(), channels * NODE_POOL_BLOCKS);
    let mut attention_logits = [0.0f32; BOARD_PLANES_SIZE];
    let mut max_logit = f32::NEG_INFINITY;
    for sq in 0..BOARD_PLANES_SIZE {
        let mut logit = 0.0;
        for channel in 0..channels {
            logit += input[channel * BOARD_PLANES_SIZE + sq] * attention_query[channel];
        }
        attention_logits[sq] = logit;
        max_logit = max_logit.max(logit);
    }
    let mut denom = 0.0;
    for logit in &mut attention_logits {
        *logit = (*logit - max_logit).exp();
        denom += *logit;
    }
    let scale = 1.0 / BOARD_PLANES_SIZE as f32;
    for channel in 0..channels {
        let start = channel * BOARD_PLANES_SIZE;
        let row = &input[start..start + BOARD_PLANES_SIZE];
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut max_value = 0.0;
        let mut attn_sum = 0.0;
        for (idx, value) in row.iter().enumerate() {
            sum += *value;
            sum_sq += *value * *value;
            if idx == 0 || *value > max_value {
                max_value = *value;
            }
            attn_sum += (*value) * attention_logits[idx] / denom.max(1e-12);
        }
        let mean = sum * scale;
        let var = (sum_sq * scale - mean * mean).max(0.0);
        output[channel] = mean;
        output[channels + channel] = max_value;
        output[channels * 2 + channel] = attn_sum;

        let mut best_row_sum = 0.0f32;
        for rank in 0..BOARD_RANKS {
            let mut row_sum = 0.0f32;
            for file in 0..BOARD_FILES {
                row_sum += row[rank * BOARD_FILES + file];
            }
            best_row_sum = best_row_sum.max(row_sum);
        }
        let mut best_col_sum = 0.0f32;
        for file in 0..BOARD_FILES {
            let mut col_sum = 0.0f32;
            for rank in 0..BOARD_RANKS {
                col_sum += row[rank * BOARD_FILES + file];
            }
            best_col_sum = best_col_sum.max(col_sum);
        }
        output[channels * 3 + channel] = best_row_sum / BOARD_FILES as f32;
        output[channels * 4 + channel] = best_col_sum / BOARD_RANKS as f32;
        output[channels * 5 + channel] = (var + RMS_NORM_EPS).sqrt();
    }
}

fn graph_node_layer_into(input: &[f32], output: &mut Vec<f32>, model: &AzNnue, layer: usize) {
    let channels = model.arch.gnn_node_channels;
    debug_assert_eq!(input.len(), channels * BOARD_PLANES_SIZE);
    debug_assert!(layer < model.arch.gnn_node_layers);
    output.resize(channels * BOARD_PLANES_SIZE, 0.0);
    let gate_base = layer * channels;
    for channel in 0..channels {
        let channel_start = channel * BOARD_PLANES_SIZE;
        let input_row = &input[channel_start..channel_start + BOARD_PLANES_SIZE];
        let output_row = &mut output[channel_start..channel_start + BOARD_PLANES_SIZE];
        let gate_index = gate_base + channel;
        let self_gate = model.graph_self_gate[gate_index];
        let local_gate = model.graph_local_gate[gate_index];
        let row_gate = model.graph_row_gate[gate_index];
        let col_gate = model.graph_col_gate[gate_index];
        let bias = model.graph_bias[gate_index];
        let mut row_mean = [0.0f32; BOARD_RANKS];
        let mut col_mean = [0.0f32; BOARD_FILES];
        let mut local_sum = [0.0f32; BOARD_PLANES_SIZE];
        let mut local_count = [0.0f32; BOARD_PLANES_SIZE];
        for rank in 0..BOARD_RANKS {
            for file in 0..BOARD_FILES {
                let sq = rank * BOARD_FILES + file;
                let value = input_row[sq];
                row_mean[rank] += value;
                col_mean[file] += value;
                for dr in -1i32..=1 {
                    for df in -1i32..=1 {
                        if dr == 0 && df == 0 {
                            continue;
                        }
                        let nr = rank as i32 + dr;
                        let nf = file as i32 + df;
                        if (0..BOARD_RANKS as i32).contains(&nr)
                            && (0..BOARD_FILES as i32).contains(&nf)
                        {
                            local_sum[sq] += input_row[nr as usize * BOARD_FILES + nf as usize];
                            local_count[sq] += 1.0;
                        }
                    }
                }
            }
        }
        for value in &mut row_mean {
            *value /= BOARD_FILES as f32;
        }
        for value in &mut col_mean {
            *value /= BOARD_RANKS as f32;
        }
        for sq in 0..BOARD_PLANES_SIZE {
            let file = sq % BOARD_FILES;
            let rank = sq / BOARD_FILES;
            let local_mean = local_sum[sq] / local_count[sq].max(1.0);
            let value = input_row[sq] * self_gate
                + local_mean * local_gate
                + row_mean[rank] * row_gate
                + col_mean[file] * col_gate
                + bias;
            output_row[sq] = value.max(0.0);
        }
    }
}

/// `batch_size` is the per-device training batch size.
pub fn benchmark_training(
    model: &mut AzNnue,
    sample_count: usize,
    epochs: usize,
    batch_size: usize,
    lr: f32,
    seed: u64,
) -> AzTrainBenchmark {
    let mut rng = SplitMix64::new(seed);
    let mut samples = Vec::with_capacity(sample_count);
    for index in 0..sample_count {
        let feature_count = 24 + (rng.next_u64() as usize % 16);
        let mut features = Vec::with_capacity(feature_count);
        for _ in 0..feature_count {
            features.push((rng.next_u64() as usize) % V4_INPUT_SIZE);
        }
        features.sort_unstable();
        features.dedup();

        let value = rng.unit_f32() * 2.0 - 1.0;
        let move_count = 12 + (rng.next_u64() as usize % 24);
        let mut move_indices = Vec::with_capacity(move_count);
        while move_indices.len() < move_count {
            let candidate = (rng.next_u64() as usize) % DENSE_MOVE_SPACE;
            if !move_indices.contains(&candidate) {
                move_indices.push(candidate);
            }
        }
        let mut policy = (0..move_count)
            .map(|_| rng.unit_f32().max(1e-6))
            .collect::<Vec<_>>();
        let policy_sum = policy.iter().sum::<f32>().max(1e-6);
        for value in &mut policy {
            *value /= policy_sum;
        }
        samples.push(AzTrainingSample {
            features,
            board: vec![0; BOARD_HISTORY_SIZE],
            move_indices,
            policy,
            value,
            side_sign: 1.0,
        });
        if index + 1 == sample_count {
            break;
        }
    }
    let stats = train_samples(model, &samples, epochs, lr, batch_size, &mut rng);
    AzTrainBenchmark {
        loss: stats.loss,
        value_loss: stats.value_loss,
        policy_ce: stats.policy_ce,
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let mut values = Vec::with_capacity(logits.len());
    softmax_into(logits, &mut values);
    values
}

fn softmax_into(logits: &[f32], output: &mut Vec<f32>) {
    output.clear();
    if logits.is_empty() {
        return;
    }
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    output.reserve(logits.len());
    for &logit in logits {
        let value = (logit - max_logit).exp();
        sum += value;
        output.push(value);
    }
    let inv_sum = sum.max(1e-12).recip();
    for value in output {
        *value *= inv_sum;
    }
}

fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    let mut sum0 = 0.0;
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;
    let chunks = left.len() / 4;
    for chunk in 0..chunks {
        let index = chunk * 4;
        sum0 += left[index] * right[index];
        sum1 += left[index + 1] * right[index + 1];
        sum2 += left[index + 2] * right[index + 2];
        sum3 += left[index + 3] * right[index + 3];
    }
    let mut sum = (sum0 + sum1) + (sum2 + sum3);
    for index in (chunks * 4)..left.len() {
        sum += left[index] * right[index];
    }
    sum
}

fn rms_norm_in_place(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }
    let mut sum_squares = 0.0f32;
    for &value in values.iter() {
        sum_squares += value * value;
    }
    let inv_rms = (sum_squares / values.len() as f32 + RMS_NORM_EPS)
        .sqrt()
        .recip();
    for value in values {
        *value *= inv_rms;
    }
}

fn scalar_value_from_logits(logits: &[f32]) -> (f32, Vec<f32>) {
    let probs = softmax(logits);
    if probs.len() < VALUE_LOGITS {
        return (0.0, probs);
    }
    (probs[0] - probs[2], probs)
}

pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = splitmix64(self.state);
        self.state
    }

    pub fn unit_f32(&mut self) -> f32 {
        let value = self.next_u64();
        (((value >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))) as f32
    }

    fn weight(&mut self, scale: f32) -> f32 {
        (self.unit_f32() * 2.0 - 1.0) * scale
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut mixed = value;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^ (mixed >> 31)
}

const fn is_advisor_pos(rank: usize, file: usize) -> bool {
    matches!(
        (rank, file),
        (0, 3) | (0, 5) | (1, 4) | (2, 3) | (2, 5) | (7, 3) | (7, 5) | (8, 4) | (9, 3) | (9, 5)
    )
}

const fn is_palace_pos(rank: usize, file: usize) -> bool {
    file >= 3 && file <= 5 && (rank <= 2 || rank >= 7)
}

const fn is_elephant_pos(rank: usize, file: usize) -> bool {
    matches!(
        (rank, file),
        (0, 2)
            | (0, 6)
            | (2, 0)
            | (2, 4)
            | (2, 8)
            | (4, 2)
            | (4, 6)
            | (5, 2)
            | (5, 6)
            | (7, 0)
            | (7, 4)
            | (7, 8)
            | (9, 2)
            | (9, 6)
    )
}

const fn is_valid_policy_move(from: usize, to: usize) -> bool {
    let from_file = from % BOARD_FILES;
    let from_rank = from / BOARD_FILES;
    let to_file = to % BOARD_FILES;
    let to_rank = to / BOARD_FILES;

    let df_signed = to_file as i32 - from_file as i32;
    let dr_signed = to_rank as i32 - from_rank as i32;
    let df = if df_signed < 0 { -df_signed } else { df_signed };
    let dr = if dr_signed < 0 { -dr_signed } else { dr_signed };

    if df == 0 || dr == 0 {
        return true;
    }
    if (df == 1 && dr == 2) || (df == 2 && dr == 1) {
        return true;
    }
    if df == 1
        && dr == 1
        && is_advisor_pos(from_rank, from_file)
        && is_advisor_pos(to_rank, to_file)
    {
        return true;
    }
    if df == 2
        && dr == 2
        && is_elephant_pos(from_rank, from_file)
        && is_elephant_pos(to_rank, to_file)
    {
        return true;
    }
    false
}

const fn compute_dense_move_count() -> usize {
    let mut count = 0;
    let mut from = 0;
    while from < BOARD_SIZE {
        let mut to = 0;
        while to < BOARD_SIZE {
            if from != to && is_valid_policy_move(from, to) {
                count += 1;
            }
            to += 1;
        }
        from += 1;
    }
    count
}

struct MoveMap {
    sparse_to_dense: [u16; SPARSE_MOVE_SPACE],
    #[allow(dead_code)]
    dense_to_sparse: [u16; DENSE_MOVE_SPACE],
}

fn move_map() -> &'static MoveMap {
    use std::sync::OnceLock;
    static MAP: OnceLock<MoveMap> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut sparse_to_dense = [u16::MAX; SPARSE_MOVE_SPACE];
        let mut dense_to_sparse = [0u16; DENSE_MOVE_SPACE];
        let mut idx = 0usize;
        for from in 0..BOARD_SIZE {
            for to in 0..BOARD_SIZE {
                if from != to && is_valid_policy_move(from, to) {
                    let sparse = from * BOARD_SIZE + to;
                    sparse_to_dense[sparse] = idx as u16;
                    dense_to_sparse[idx] = sparse as u16;
                    idx += 1;
                }
            }
        }
        assert_eq!(idx, DENSE_MOVE_SPACE);
        MoveMap {
            sparse_to_dense,
            dense_to_sparse,
        }
    })
}

pub(super) fn policy_move_from_indices() -> Vec<u32> {
    move_map()
        .dense_to_sparse
        .iter()
        .map(|&sparse| (sparse as usize / BOARD_SIZE) as u32)
        .collect()
}

pub(super) fn policy_move_to_indices() -> Vec<u32> {
    move_map()
        .dense_to_sparse
        .iter()
        .map(|&sparse| (sparse as usize % BOARD_SIZE) as u32)
        .collect()
}

pub(super) fn policy_move_features() -> &'static [f32] {
    use std::sync::OnceLock;
    static FEATURES: OnceLock<Vec<f32>> = OnceLock::new();
    FEATURES.get_or_init(|| {
        let mut features = Vec::with_capacity(DENSE_MOVE_SPACE * POLICY_CONDITION_SIZE);
        for &sparse in &move_map().dense_to_sparse {
            let from = sparse as usize / BOARD_SIZE;
            let to = sparse as usize % BOARD_SIZE;
            let from_file = from % BOARD_FILES;
            let from_rank = from / BOARD_FILES;
            let to_file = to % BOARD_FILES;
            let to_rank = to / BOARD_FILES;
            let df_signed = to_file as i32 - from_file as i32;
            let dr_signed = to_rank as i32 - from_rank as i32;
            let df = df_signed.unsigned_abs() as usize;
            let dr = dr_signed.unsigned_abs() as usize;
            let line = (df == 0 || dr == 0) as u8 as f32;
            let horse = ((df == 1 && dr == 2) || (df == 2 && dr == 1)) as u8 as f32;
            let advisor = (df == 1 && dr == 1) as u8 as f32;
            let elephant = (df == 2 && dr == 2) as u8 as f32;
            let distance = (df + dr) as f32;
            let from_palace = is_palace_pos(from_rank, from_file) as u8 as f32;
            let to_palace = is_palace_pos(to_rank, to_file) as u8 as f32;
            let crosses_river = ((from_rank < 5) != (to_rank < 5)) as u8 as f32;
            let forward_red = (to_rank as i32 - from_rank as i32).signum() as f32;
            let center_from = 1.0 - ((from_file as f32 - 4.0).abs() / 4.0);
            let center_to = 1.0 - ((to_file as f32 - 4.0).abs() / 4.0);
            let raw = [
                1.0,
                from_file as f32 / 8.0,
                from_rank as f32 / 9.0,
                to_file as f32 / 8.0,
                to_rank as f32 / 9.0,
                df as f32 / 8.0,
                dr as f32 / 9.0,
                df_signed as f32 / 8.0,
                dr_signed as f32 / 9.0,
                line,
                (df == 0) as u8 as f32,
                (dr == 0) as u8 as f32,
                horse,
                advisor,
                elephant,
                distance / 9.0,
                (distance <= 1.0) as u8 as f32,
                (distance <= 2.0) as u8 as f32,
                (distance >= 5.0) as u8 as f32,
                from_palace,
                to_palace,
                crosses_river,
                forward_red,
                center_from,
                center_to,
                ((from_rank <= 2) || (from_rank >= 7)) as u8 as f32,
                ((to_rank <= 2) || (to_rank >= 7)) as u8 as f32,
                ((from_file == 0) || (from_file == 8)) as u8 as f32,
                ((to_file == 0) || (to_file == 8)) as u8 as f32,
                (df == dr) as u8 as f32,
                ((df + dr) % 2 == 0) as u8 as f32,
                ((from + to) % 2 == 0) as u8 as f32,
            ];
            debug_assert_eq!(raw.len(), POLICY_CONDITION_SIZE);
            features.extend_from_slice(&raw);
        }
        features
    })
}

fn dense_move_index(mv: Move) -> usize {
    let sparse = mv.from as usize * BOARD_SIZE + mv.to as usize;
    let dense = move_map().sparse_to_dense[sparse];
    debug_assert!(
        dense != u16::MAX,
        "invalid policy move {}->{}",
        mv.from,
        mv.to
    );
    dense as usize
}

#[cfg_attr(not(feature = "distill"), allow(dead_code))]
pub(super) fn dense_move_to_move(index: usize) -> Option<Move> {
    let sparse = *move_map().dense_to_sparse.get(index)? as usize;
    Some(Move::new(sparse / BOARD_SIZE, sparse % BOARD_SIZE))
}

#[cfg(test)]
fn replay_pool_test_fixture() -> AzExperiencePool {
    let sample = AzTrainingSample {
        features: vec![1, 2, 3],
        board: vec![0; BOARD_HISTORY_SIZE],
        move_indices: vec![0, 1],
        policy: vec![0.6, 0.4],
        value: 0.1,
        side_sign: 1.0,
    };
    let mut pool = AzExperiencePool::new(100);
    pool.add_games(vec![vec![sample.clone()], vec![sample.clone(), sample]]);
    pool
}

#[cfg(test)]
mod tests {
    use super::play::{assign_td_lambda_value_targets, assign_terminal_value_targets};
    use super::*;

    #[test]
    fn dense_move_space_matches_enumeration() {
        let map = move_map();
        assert_eq!(DENSE_MOVE_SPACE, 2086);
        for i in 0..DENSE_MOVE_SPACE {
            let sparse = map.dense_to_sparse[i] as usize;
            assert_eq!(map.sparse_to_dense[sparse], i as u16);
        }
    }

    #[test]
    fn board_history_planes_include_rewound_previous_position() {
        let mut position = Position::startpos();
        let mv = position.legal_moves()[0];
        let moved_piece = position.piece_at(mv.from as usize).unwrap();
        let history = vec![HistoryMove {
            piece: moved_piece,
            captured: None,
            mv,
        }];
        position.make_move(mv);

        let mut board = Vec::new();
        extract_board_planes(&position, &history, &mut board);
        let side = position.side_to_move();
        let piece_plane =
            (canonical_piece_plane(side, moved_piece.color, moved_piece.kind) + 1) as u8;
        let from = canonical_square(side, mv.from as usize);
        let to = canonical_square(side, mv.to as usize);

        assert_eq!(board.len(), BOARD_HISTORY_SIZE);
        assert_eq!(board[to], piece_plane);
        assert_eq!(board[BOARD_PLANES_SIZE + from], piece_plane);
        assert_eq!(board[BOARD_PLANES_SIZE + to], 0);
    }

    #[test]
    fn board_history_planes_restore_captured_piece_when_rewound() {
        let mut position = Position::from_fen("4k4/9/9/9/r3c4/9/9/9/R8/4K4 w").unwrap();
        let mv = Move::new(72, 36);
        assert!(position.is_legal_move(mv));
        let moved_piece = position.piece_at(mv.from as usize).unwrap();
        let captured_piece = position.piece_at(mv.to as usize).unwrap();
        let history = vec![HistoryMove {
            piece: moved_piece,
            captured: Some(captured_piece),
            mv,
        }];
        position.make_move(mv);

        let mut board = Vec::new();
        extract_board_planes(&position, &history, &mut board);
        let side = position.side_to_move();
        let moved_plane =
            (canonical_piece_plane(side, moved_piece.color, moved_piece.kind) + 1) as u8;
        let captured_plane =
            (canonical_piece_plane(side, captured_piece.color, captured_piece.kind) + 1) as u8;
        let from = canonical_square(side, mv.from as usize);
        let to = canonical_square(side, mv.to as usize);

        assert_eq!(board[to], moved_plane);
        assert_eq!(board[BOARD_PLANES_SIZE + from], moved_plane);
        assert_eq!(board[BOARD_PLANES_SIZE + to], captured_plane);
    }

    #[test]
    fn terminal_value_targets_match_outcome_for_side_to_move() {
        let mut samples = vec![
            AzTrainingSample {
                features: Vec::new(),
                board: vec![0; BOARD_HISTORY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 0.0,
                side_sign: 1.0,
            },
            AzTrainingSample {
                features: Vec::new(),
                board: vec![0; BOARD_HISTORY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 0.0,
                side_sign: -1.0,
            },
        ];

        assign_terminal_value_targets(&mut samples, 1.0);

        assert!((samples[0].value - 1.0).abs() < 1e-6);
        assert!((samples[1].value + 1.0).abs() < 1e-6);
    }

    #[test]
    fn td_lambda_value_targets_mix_future_bootstrap_values() {
        let mut samples = vec![
            AzTrainingSample {
                features: Vec::new(),
                board: vec![0; BOARD_HISTORY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 0.2,
                side_sign: 1.0,
            },
            AzTrainingSample {
                features: Vec::new(),
                board: vec![0; BOARD_HISTORY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: -0.4,
                side_sign: -1.0,
            },
            AzTrainingSample {
                features: Vec::new(),
                board: vec![0; BOARD_HISTORY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 0.6,
                side_sign: 1.0,
            },
        ];

        assign_td_lambda_value_targets(&mut samples, 1.0, 0.5);

        assert!((samples[2].value - 1.0).abs() < 1e-6);
        assert!((samples[1].value + 0.8).abs() < 1e-6);
        assert!((samples[0].value - 0.6).abs() < 1e-6);
    }

    #[test]
    fn arena_report_anchored_elo_tracks_score_vs_reference() {
        let reference = 1500.0f32;
        let stronger = AzArenaReport {
            wins: 6,
            losses: 3,
            draws: 1,
            ..AzArenaReport::default()
        };
        let weaker = AzArenaReport {
            wins: 3,
            losses: 6,
            draws: 1,
            ..AzArenaReport::default()
        };

        assert!(stronger.score_rate() > 0.5);
        assert!(stronger.anchored_elo(reference) > reference);
        assert!(weaker.score_rate() < 0.5);
        assert!(weaker.anchored_elo(reference) < reference);
    }

    #[test]
    fn value_head_can_overfit_tiny_fixed_dataset() {
        let mut model = AzNnue::random(16, 7);
        model.hidden_bias.fill(0.1);
        model.hidden_bias.fill(0.1);
        let board_with = |sq: usize, plane: u8| {
            let mut board = vec![0; BOARD_HISTORY_SIZE];
            board[sq] = plane;
            board
        };

        let samples = vec![
            AzTrainingSample {
                features: vec![0],
                board: board_with(0, 1),
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 1.0,
                side_sign: 1.0,
            },
            AzTrainingSample {
                features: vec![1],
                board: board_with(10, 2),
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: -1.0,
                side_sign: 1.0,
            },
            AzTrainingSample {
                features: vec![2],
                board: board_with(40, 3),
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 0.75,
                side_sign: 1.0,
            },
            AzTrainingSample {
                features: vec![3],
                board: board_with(80, 4),
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: -0.75,
                side_sign: 1.0,
            },
        ];

        let mut rng = SplitMix64::new(17);
        let before = train_samples(&mut model, &samples, 1, 0.003, 4, &mut rng).value_loss;
        let after = train_samples(&mut model, &samples, 300, 0.003, 4, &mut rng).value_loss;

        assert!(after < before * 0.5, "before={before} after={after}");
        assert!(after < 0.35, "after={after}");
    }

    #[test]
    fn batched_training_is_deterministic() {
        let samples = vec![
            AzTrainingSample {
                features: vec![0, 4, 8],
                board: vec![0; BOARD_HISTORY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 1.0,
                side_sign: 1.0,
            },
            AzTrainingSample {
                features: vec![1, 5, 9],
                board: vec![0; BOARD_HISTORY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: -1.0,
                side_sign: 1.0,
            },
            AzTrainingSample {
                features: vec![2, 6, 10],
                board: vec![0; BOARD_HISTORY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 0.5,
                side_sign: 1.0,
            },
            AzTrainingSample {
                features: vec![3, 7, 11],
                board: vec![0; BOARD_HISTORY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: -0.5,
                side_sign: 1.0,
            },
        ];
        let mut single = AzNnue::random(16, 23);
        single.hidden_bias.fill(0.1);
        single.hidden_bias.fill(0.1);
        let mut repeated = single.clone();

        let mut rng_single = SplitMix64::new(99);
        let mut rng_repeated = SplitMix64::new(99);
        let single_stats = train_samples(&mut single, &samples, 5, 0.003, 4, &mut rng_single);
        let repeated_stats = train_samples(&mut repeated, &samples, 5, 0.003, 4, &mut rng_repeated);

        assert!((single_stats.loss - repeated_stats.loss).abs() < 1e-5);
        assert!((single_stats.value_loss - repeated_stats.value_loss).abs() < 1e-5);
        assert!((single_stats.value_pred_sum - repeated_stats.value_pred_sum).abs() < 1e-4);
        assert!((single_stats.value_target_sum - repeated_stats.value_target_sum).abs() < 1e-6);
        assert!(
            single
                .value_logits_bias
                .iter()
                .zip(&repeated.value_logits_bias)
                .all(|(left, right)| (*left - *right).abs() < 1e-5)
        );
    }

    #[test]
    fn aznnue_binary_roundtrip_matches_weights() {
        let model = AzNnue::random(16, 42);
        let path = std::env::temp_dir().join("chineseai_test_aznnue_roundtrip.nnue");
        let _ = fs::remove_file(&path);
        model.save(&path).unwrap();
        let loaded = AzNnue::load(&path).unwrap();
        let _ = fs::remove_file(&path);
        assert_eq!(model.hidden_size, loaded.hidden_size);
        assert_eq!(model.input_hidden, loaded.input_hidden);
        assert_eq!(model.policy_move_bias, loaded.policy_move_bias);
    }

    #[test]
    fn replay_pool_lz4_snapshot_roundtrip() {
        let path = std::env::temp_dir().join("chineseai_test_replay_roundtrip.replay.lz4");
        let _ = fs::remove_file(&path);
        let pool = super::replay_pool_test_fixture();
        pool.save_snapshot_lz4(&path).unwrap();
        let loaded = AzExperiencePool::load_snapshot_lz4(&path, 100).unwrap();
        let _ = fs::remove_file(&path);
        assert_eq!(loaded.game_count(), pool.game_count());
        assert_eq!(loaded.sample_count(), pool.sample_count());
    }
}
