mod accumulator;
mod adamw;
mod alphazero;
pub(crate) use alphazero::{AzUciSearchResult, search_uci};
mod arch;
#[cfg(any(
    all(feature = "gpu-train", not(target_os = "macos")),
    all(target_os = "linux", not(target_env = "musl")),
    all(test, target_os = "macos"),
    target_os = "windows",
))]
#[cfg_attr(all(test, target_os = "macos"), allow(dead_code))]
mod candle_model;
// slow-tests 的 CUDA 对照测试共用同一个设备，避免每个测试重复 JIT 编译 kernel。
#[cfg(all(test, feature = "slow-tests"))]
mod cuda_test_device;
mod dataloader;
mod fused_feature_pool;
mod fused_policy;
mod fused_sparse_policy;
mod history;
mod inference;
mod mate;
pub mod nnue;
mod play;
mod policy_calibration;
pub use policy_calibration::{PolicyCalibrationReport, calibrate_policy};
pub mod px0_data;
mod px0_policy_map;
#[path = "px0_sgd.rs"]
mod px0_sgd;
mod replay;
mod reflection;
mod sample;
mod simd;
mod start;
#[cfg(test)]
mod tests;
mod train;
mod train_gpu;
#[cfg(any(
    all(feature = "gpu-train", not(target_os = "macos")),
    all(target_os = "linux", not(target_env = "musl")),
    target_os = "windows",
))]
#[path = "train_gpu_candle.rs"]
mod train_gpu_candle;

pub use alphazero::{
    AzCandidate, AzSearchControl, AzSearchLimits, AzSearchResult, AzSearchTraceStep,
    alphazero_search, alphazero_search_external_root_controlled_with_progress,
    alphazero_search_trace_with_rules, alphazero_search_with_rules,
    alphazero_search_with_rules_controlled, alphazero_search_with_rules_controlled_with_progress,
    cp_from_q,
};
pub use dataloader::{AzSparseActivationStats, sparse_activation_stats};
pub use play::{
    AzArenaConfig, AzArenaReport, AzSelfplayData, AzTerminalStats, generate_selfplay_data,
    play_arena_games_from_positions, play_arena_games_from_snapshots,
};
pub use replay::{AzExperiencePool, AzReplaySampleBatch, AzReplayWindowStats, Px0ReplaySampler};
pub use start::AzStartSnapshot;
pub use train::{
    AzTrainOptimizer, train_samples, train_samples_weighted, train_samples_weighted_owned,
    train_samples_weighted_owned_with_optimizer, train_samples_weighted_shared,
};

// 门面重导出：把拆分到子模块的项保持在 `crate::az::*` 的原路径上。
pub(crate) use crate::az::nnue::canonical_move;
pub(crate) use crate::xiangqi::{Color, Move, Position, color_index};
// px0_data 的测试模块通过 `use super::*` 使用该名字。
#[cfg(test)]
pub(crate) use crate::xiangqi::PieceKind;

pub(crate) use accumulator::*;
pub(crate) use arch::*;
pub use history::{history_features, history_features_from_planes};
pub(crate) use inference::*;
pub(crate) use simd::*;

pub use arch::{
    AzMovesLeftParams, AzNnueArch, CHECK_CONTEXT_SIZE, DENSE_MOVE_SPACE, HISTORY_CONTEXT_SIZE,
    POLICY_SPARSE_MAIN_SIZE, POLICY_TACTICAL_EXACT_SIZE, POLICY_TACTICAL_FACTOR_SIZE,
    RULE_CONTEXT_SIZE, dense_move_index,
};
pub use inference::{AzNnue, outputs_for_training_sample, position_for_training_sample};
pub use mate::{
    MateSearchLimits, MateSearchOutcome, MateSearchReport, MateSolution, search_root_mate,
    search_root_mate_profiled,
};
pub use sample::{
    AzHoldoutReport, AzLoopConfig, AzLoopReport, AzPhaseValueReport, AzPolicyGroupStats,
    AzSampleMeta, AzStartSource, AzTrainLossWeights, AzTrainStats, AzTrainingSample,
    AzValueMomentStats, PX0_CYCLE_STEPS, PX0_TEST_STEPS, SplitMix64, evaluate_policy_groups,
    policy_target_entropy, rule_context_features,
};
pub use simd::inference_simd_backend;
