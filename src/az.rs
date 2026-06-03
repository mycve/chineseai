use std::fs;
use std::io::{self, BufWriter, Cursor, Read, Write};
use std::path::Path;

mod alphazero;
mod mctx;
mod play;
mod policy_features;
mod replay;
mod train;
mod train_gpu;
#[cfg(any(
    feature = "gpu-train",
    all(target_os = "linux", not(target_env = "musl"))
))]
#[path = "az/train_gpu_candle.rs"]
mod train_gpu_candle;

use crate::nnue::{
    AZ_NNUE_INPUT_SIZE, HistoryMove, V2_KING_BUCKETS, add_az_absolute_history_features,
    az_absolute_general_bucket, az_absolute_piece_king_feature,
    az_absolute_piece_non_king_features, az_absolute_side_to_move_feature,
    az_absolute_strategic_features, canonical_move, canonical_square,
    extract_sparse_features_az_absolute_current, extract_sparse_features_az_canonical,
};
use crate::xiangqi::{
    BOARD_FILES, BOARD_RANKS, BOARD_SIZE, Color, Move, Piece, PieceKind, Position,
};

pub use alphazero::{
    AzCandidate, AzSearchAlgorithm, AzSearchLimits, AzSearchResult, alphazero_search,
    alphazero_search_with_history_and_rules,
};
pub use mctx::AzGumbelConfig;
pub use play::{
    AzArenaConfig, AzArenaReport, AzSelfplayData, AzTerminalStats, generate_selfplay_data,
    play_arena_games_from_positions,
};
pub(super) use policy_features::{PolicyFeaturePositionCache, policy_dynamic_features_cached};
pub use replay::AzExperiencePool;
pub use train::{global_training_step_sample_count, train_samples, train_samples_weighted};

pub const AZNNUE_BINARY_MAGIC: &[u8] = b"AZB1";
const AZNNUE_BINARY_VERSION: u32 = 72;
const AZNNUE_BINARY_HEADER_LEN: usize = 4 + 4 * 3;

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
pub(super) const POLICY_CONTEXT_SIZE: usize = 64;
pub(super) const POLICY_CONDITION_SIZE: usize = 32;
pub(super) const POLICY_DYNAMIC_FEATURE_SIZE: usize = 32;
pub(super) const POLICY_ACTION_SIZE: usize = 32;
pub(super) const POLICY_PAIR_CONTEXT_SIZE: usize = 32;
pub(super) const POLICY_MOVE_EMBED_SIZE: usize = 16;
pub(super) const POLICY_RELATION_SIZE: usize = 64;
pub(super) const VALUE_HEAD_SIZE: usize = 64;
#[cfg_attr(not(feature = "gpu-train"), allow(dead_code))]
pub(super) const MOVES_LEFT_AUX_WEIGHT: f32 = 0.05;
pub(super) const AUTO_FEATURE_SIZE: usize = 64;
pub(super) const PIECE_ATTENTION_SIZE: usize = 32;
pub(super) const LINE_MIXER_RANK: usize = 32;
pub(super) const SQUARE_TOKEN_PIECES: usize = 15;
pub(super) const LINE_COUNT: usize = BOARD_RANKS + BOARD_FILES;
pub(super) const TRUNK_LAYERS: usize = 2;
const VALUE_SCALE_CP: f32 = 1000.0;
const RMS_NORM_EPS: f32 = 1.0e-6;
pub(super) const PIECE_SQUARE_INPUT_SIZE: usize = BOARD_SIZE * 14;
pub(super) const STRUCTURAL_PIECE_SIZE: usize = 14;
pub(super) const STRUCTURAL_RANK_SIZE: usize = BOARD_RANKS;
pub(super) const STRUCTURAL_FILE_SIZE: usize = BOARD_FILES;
pub(super) const STRUCTURAL_KING_PIECE_SIZE: usize = 2 * V2_KING_BUCKETS * 14;

#[derive(Clone, Copy, Debug)]
pub(super) struct StructuralPieceSquare {
    pub piece_index: usize,
    pub rank: usize,
    pub file: usize,
}

pub(super) fn decode_current_piece_square_feature(feature: usize) -> Option<StructuralPieceSquare> {
    if feature >= PIECE_SQUARE_INPUT_SIZE {
        return None;
    }
    let piece_index = feature / BOARD_SIZE;
    let sq = feature % BOARD_SIZE;
    Some(StructuralPieceSquare {
        piece_index,
        rank: sq / BOARD_FILES,
        file: sq % BOARD_FILES,
    })
}

pub(super) fn canonical_general_buckets_from_features(features: &[usize]) -> (usize, usize) {
    let mut us = 4;
    let mut them = 4;
    for &feature in features {
        if feature >= PIECE_SQUARE_INPUT_SIZE {
            continue;
        }
        let piece_index = feature / BOARD_SIZE;
        let sq = feature % BOARD_SIZE;
        match piece_index {
            0 => us = canonical_general_bucket(piece_index, sq),
            7 => them = canonical_general_bucket(piece_index, sq),
            _ => {}
        }
    }
    (us, them)
}

pub(super) fn structural_king_piece_index(
    perspective: usize,
    king_bucket: usize,
    piece_index: usize,
) -> usize {
    ((perspective * V2_KING_BUCKETS + king_bucket.min(V2_KING_BUCKETS - 1)) * 14) + piece_index
}

fn canonical_general_bucket(piece_index: usize, sq: usize) -> usize {
    let oriented_sq = if piece_index < 7 {
        sq
    } else {
        BOARD_SIZE - 1 - sq
    };
    let file = (oriented_sq % BOARD_FILES).clamp(3, 5) - 3;
    let rank = (oriented_sq / BOARD_FILES).clamp(7, 9) - 7;
    rank * 3 + file
}

fn square_token_piece_index(piece: Piece, side_to_move: Color) -> usize {
    let color_offset = if piece.color == side_to_move { 0 } else { 7 };
    color_offset
        + match piece.kind {
            PieceKind::General => 0,
            PieceKind::Advisor => 1,
            PieceKind::Elephant => 2,
            PieceKind::Horse => 3,
            PieceKind::Rook => 4,
            PieceKind::Cannon => 5,
            PieceKind::Soldier => 6,
        }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AzNnueArch {
    pub hidden_size: usize,
}

impl AzNnueArch {
    pub const fn default_const() -> Self {
        Self { hidden_size: 192 }
    }

    pub const fn with_hidden_size(hidden_size: usize) -> Self {
        let mut arch = Self::default_const();
        arch.hidden_size = hidden_size;
        arch
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_size == 0 {
            return Err(format!("invalid hidden_size {}", self.hidden_size));
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
    auto_features: Vec<f32>,
    line_repr: Vec<f32>,
    line_bottleneck: Vec<f32>,
    trunk_work: Vec<f32>,
    #[allow(dead_code)]
    history_features: Vec<usize>,
    value_head: Vec<f32>,
    policy_dynamic_condition: Vec<f32>,
    policy_action_context: Vec<f32>,
    policy_pair_context: Vec<f32>,
    policy_move_context: Vec<f32>,
    policy_relation_context: Vec<f32>,
    policy_from_scores: Vec<f32>,
    policy_to_scores: Vec<f32>,
    logits: Vec<f32>,
    priors: Vec<f32>,
}

impl AzEvalScratch {
    pub(super) fn new(arch: AzNnueArch) -> Self {
        let hidden_size = arch.hidden_size;
        Self {
            hidden: vec![0.0; hidden_size],
            auto_features: vec![0.0; AUTO_FEATURE_SIZE],
            line_repr: vec![0.0; hidden_size],
            line_bottleneck: vec![0.0; LINE_MIXER_RANK],
            trunk_work: vec![0.0; hidden_size],
            history_features: Vec::with_capacity(16),
            value_head: vec![0.0; VALUE_HEAD_SIZE],
            policy_dynamic_condition: vec![0.0; POLICY_DYNAMIC_FEATURE_SIZE],
            policy_action_context: vec![0.0; POLICY_ACTION_SIZE],
            policy_pair_context: vec![0.0; POLICY_PAIR_CONTEXT_SIZE],
            policy_move_context: vec![0.0; POLICY_MOVE_EMBED_SIZE],
            policy_relation_context: vec![0.0; POLICY_RELATION_SIZE],
            policy_from_scores: vec![0.0; BOARD_SIZE],
            policy_to_scores: vec![0.0; BOARD_SIZE],
            logits: Vec::with_capacity(192),
            priors: Vec::with_capacity(192),
        }
    }
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Debug)]
struct AzEvalAccumulator {
    hidden_sum: Vec<f32>,
}

#[cfg_attr(not(test), allow(dead_code))]
impl AzEvalAccumulator {
    fn new(model: &AzNnue, position: &Position) -> Self {
        let mut accumulator = Self {
            hidden_sum: vec![0.0; model.hidden_size],
        };
        accumulator.refresh_from_position(model, position);
        accumulator
    }

    fn refresh_from_position(&mut self, model: &AzNnue, position: &Position) {
        self.hidden_sum.resize(model.hidden_size, 0.0);
        self.hidden_sum.copy_from_slice(&model.hidden_bias);
        let features = extract_sparse_features_az_absolute_current(position);
        for feature in features {
            self.add_feature(model, feature);
        }
    }

    #[allow(dead_code)]
    fn apply_transition(
        &mut self,
        model: &AzNnue,
        before: &Position,
        after: &Position,
        mv: Move,
        moved: Piece,
        captured: Option<Piece>,
    ) {
        self.replace_side_feature(model, before, after);
        self.replace_strategic_features(model, before, after);

        let old_red_bucket = az_absolute_general_bucket(before, Color::Red);
        let new_red_bucket = az_absolute_general_bucket(after, Color::Red);
        let old_black_bucket = az_absolute_general_bucket(before, Color::Black);
        let new_black_bucket = az_absolute_general_bucket(after, Color::Black);
        let red_king_changed = old_red_bucket != new_red_bucket;
        let black_king_changed = old_black_bucket != new_black_bucket;

        self.remove_non_king_piece_features(model, mv.from as usize, moved);
        if let Some(captured) = captured {
            self.remove_non_king_piece_features(model, mv.to as usize, captured);
        }
        self.add_non_king_piece_features(model, mv.to as usize, moved);

        if red_king_changed {
            self.replace_all_king_features(
                model,
                before,
                after,
                Color::Red,
                old_red_bucket,
                new_red_bucket,
            );
        } else {
            self.remove_piece_king_feature(
                model,
                Color::Red,
                old_red_bucket,
                mv.from as usize,
                moved,
            );
            if let Some(captured) = captured {
                self.remove_piece_king_feature(
                    model,
                    Color::Red,
                    old_red_bucket,
                    mv.to as usize,
                    captured,
                );
            }
            self.add_piece_king_feature(model, Color::Red, new_red_bucket, mv.to as usize, moved);
        }

        if black_king_changed {
            self.replace_all_king_features(
                model,
                before,
                after,
                Color::Black,
                old_black_bucket,
                new_black_bucket,
            );
        } else {
            self.remove_piece_king_feature(
                model,
                Color::Black,
                old_black_bucket,
                mv.from as usize,
                moved,
            );
            if let Some(captured) = captured {
                self.remove_piece_king_feature(
                    model,
                    Color::Black,
                    old_black_bucket,
                    mv.to as usize,
                    captured,
                );
            }
            self.add_piece_king_feature(
                model,
                Color::Black,
                new_black_bucket,
                mv.to as usize,
                moved,
            );
        }
    }

    #[allow(dead_code)]
    fn replace_side_feature(&mut self, model: &AzNnue, before: &Position, after: &Position) {
        if let Some(feature) = az_absolute_side_to_move_feature(before) {
            self.sub_feature(model, feature);
        }
        if let Some(feature) = az_absolute_side_to_move_feature(after) {
            self.add_feature(model, feature);
        }
    }

    #[allow(dead_code)]
    fn replace_strategic_features(&mut self, model: &AzNnue, before: &Position, after: &Position) {
        let mut features = Vec::with_capacity(8);
        az_absolute_strategic_features(before, &mut features);
        for feature in features.drain(..) {
            self.sub_feature(model, feature);
        }
        az_absolute_strategic_features(after, &mut features);
        for feature in features {
            self.add_feature(model, feature);
        }
    }

    #[allow(dead_code)]
    fn add_non_king_piece_features(&mut self, model: &AzNnue, sq: usize, piece: Piece) {
        let mut features = Vec::with_capacity(3);
        az_absolute_piece_non_king_features(sq, piece, &mut features);
        for feature in features {
            self.add_feature(model, feature);
        }
    }

    #[allow(dead_code)]
    fn remove_non_king_piece_features(&mut self, model: &AzNnue, sq: usize, piece: Piece) {
        let mut features = Vec::with_capacity(3);
        az_absolute_piece_non_king_features(sq, piece, &mut features);
        for feature in features {
            self.sub_feature(model, feature);
        }
    }

    #[allow(dead_code)]
    fn replace_all_king_features(
        &mut self,
        model: &AzNnue,
        before: &Position,
        after: &Position,
        perspective: Color,
        old_bucket: usize,
        new_bucket: usize,
    ) {
        for sq in 0..BOARD_SIZE {
            if let Some(piece) = before.piece_at(sq) {
                self.remove_piece_king_feature(model, perspective, old_bucket, sq, piece);
            }
        }
        for sq in 0..BOARD_SIZE {
            if let Some(piece) = after.piece_at(sq) {
                self.add_piece_king_feature(model, perspective, new_bucket, sq, piece);
            }
        }
    }

    #[allow(dead_code)]
    fn add_piece_king_feature(
        &mut self,
        model: &AzNnue,
        perspective: Color,
        bucket: usize,
        sq: usize,
        piece: Piece,
    ) {
        self.add_feature(
            model,
            az_absolute_piece_king_feature(perspective, bucket, sq, piece),
        );
    }

    #[allow(dead_code)]
    fn remove_piece_king_feature(
        &mut self,
        model: &AzNnue,
        perspective: Color,
        bucket: usize,
        sq: usize,
        piece: Piece,
    ) {
        self.sub_feature(
            model,
            az_absolute_piece_king_feature(perspective, bucket, sq, piece),
        );
    }

    fn add_feature(&mut self, model: &AzNnue, feature: usize) {
        add_scaled_feature_row(
            &mut self.hidden_sum,
            &model.input_hidden,
            model.hidden_size,
            feature,
            1.0,
        );
    }

    #[allow(dead_code)]
    fn sub_feature(&mut self, model: &AzNnue, feature: usize) {
        add_scaled_feature_row(
            &mut self.hidden_sum,
            &model.input_hidden,
            model.hidden_size,
            feature,
            -1.0,
        );
    }
}

#[derive(Debug)]
pub struct AzNnue {
    pub hidden_size: usize,
    pub arch: AzNnueArch,
    pub input_hidden: Vec<f32>,
    pub input_piece_hidden: Vec<f32>,
    pub input_rank_hidden: Vec<f32>,
    pub input_file_hidden: Vec<f32>,
    pub input_king_piece_hidden: Vec<f32>,
    pub hidden_bias: Vec<f32>,
    pub input_quadratic_scale: Vec<f32>,
    pub piece_attention_query: Vec<f32>,
    pub piece_attention_value: Vec<f32>,
    pub piece_attention_output: Vec<f32>,
    pub square_piece_token: Vec<f32>,
    pub square_position_token: Vec<f32>,
    pub line_mixer_down: Vec<f32>,
    pub line_mixer_bias: Vec<f32>,
    pub line_mixer_up: Vec<f32>,
    pub line_context_bias: Vec<f32>,
    pub trunk_residual_hidden: Vec<f32>,
    pub trunk_residual_bias: Vec<f32>,
    pub auto_feature_hidden: Vec<f32>,
    pub auto_feature_bias: Vec<f32>,
    pub auto_feature_output: Vec<f32>,
    pub value_head_hidden: Vec<f32>,
    pub value_head_bias: Vec<f32>,
    pub value_head_output: Vec<f32>,
    pub moves_left_output: Vec<f32>,
    pub moves_left_bias: Vec<f32>,
    pub policy_move_bias: Vec<f32>,
    pub policy_context_hidden: Vec<f32>,
    pub policy_context_bias: Vec<f32>,
    pub policy_feature_hidden: Vec<f32>,
    pub policy_feature_bias: Vec<f32>,
    pub policy_dynamic_hidden: Vec<f32>,
    pub policy_action_context_hidden: Vec<f32>,
    pub policy_action_context_bias: Vec<f32>,
    pub policy_action_static_hidden: Vec<f32>,
    policy_action_static_projection: Vec<f32>,
    pub policy_action_dynamic_hidden: Vec<f32>,
    pub policy_action_output: Vec<f32>,
    pub policy_from_hidden: Vec<f32>,
    pub policy_to_hidden: Vec<f32>,
    pub policy_pair_context_hidden: Vec<f32>,
    pub policy_pair_context_bias: Vec<f32>,
    pub policy_pair_embedding: Vec<f32>,
    pub policy_move_context_hidden: Vec<f32>,
    pub policy_move_embedding: Vec<f32>,
    pub policy_relation_context_hidden: Vec<f32>,
    pub policy_relation_context_bias: Vec<f32>,
    pub policy_relation_from_embedding: Vec<f32>,
    pub policy_relation_to_embedding: Vec<f32>,
    policy_relation_pair_embedding: Vec<f32>,
    pub policy_relation_output: Vec<f32>,
    #[cfg_attr(not(feature = "gpu-train"), allow(dead_code))]
    gpu_trainer: Option<Box<train_gpu::GpuTrainer>>,
}

impl Clone for AzNnue {
    fn clone(&self) -> Self {
        Self {
            hidden_size: self.hidden_size,
            arch: self.arch,
            input_hidden: self.input_hidden.clone(),
            input_piece_hidden: self.input_piece_hidden.clone(),
            input_rank_hidden: self.input_rank_hidden.clone(),
            input_file_hidden: self.input_file_hidden.clone(),
            input_king_piece_hidden: self.input_king_piece_hidden.clone(),
            hidden_bias: self.hidden_bias.clone(),
            input_quadratic_scale: self.input_quadratic_scale.clone(),
            piece_attention_query: self.piece_attention_query.clone(),
            piece_attention_value: self.piece_attention_value.clone(),
            piece_attention_output: self.piece_attention_output.clone(),
            square_piece_token: self.square_piece_token.clone(),
            square_position_token: self.square_position_token.clone(),
            line_mixer_down: self.line_mixer_down.clone(),
            line_mixer_bias: self.line_mixer_bias.clone(),
            line_mixer_up: self.line_mixer_up.clone(),
            line_context_bias: self.line_context_bias.clone(),
            trunk_residual_hidden: self.trunk_residual_hidden.clone(),
            trunk_residual_bias: self.trunk_residual_bias.clone(),
            auto_feature_hidden: self.auto_feature_hidden.clone(),
            auto_feature_bias: self.auto_feature_bias.clone(),
            auto_feature_output: self.auto_feature_output.clone(),
            value_head_hidden: self.value_head_hidden.clone(),
            value_head_bias: self.value_head_bias.clone(),
            value_head_output: self.value_head_output.clone(),
            moves_left_output: self.moves_left_output.clone(),
            moves_left_bias: self.moves_left_bias.clone(),
            policy_move_bias: self.policy_move_bias.clone(),
            policy_context_hidden: self.policy_context_hidden.clone(),
            policy_context_bias: self.policy_context_bias.clone(),
            policy_feature_hidden: self.policy_feature_hidden.clone(),
            policy_feature_bias: self.policy_feature_bias.clone(),
            policy_dynamic_hidden: self.policy_dynamic_hidden.clone(),
            policy_action_context_hidden: self.policy_action_context_hidden.clone(),
            policy_action_context_bias: self.policy_action_context_bias.clone(),
            policy_action_static_hidden: self.policy_action_static_hidden.clone(),
            policy_action_static_projection: self.policy_action_static_projection.clone(),
            policy_action_dynamic_hidden: self.policy_action_dynamic_hidden.clone(),
            policy_action_output: self.policy_action_output.clone(),
            policy_from_hidden: self.policy_from_hidden.clone(),
            policy_to_hidden: self.policy_to_hidden.clone(),
            policy_pair_context_hidden: self.policy_pair_context_hidden.clone(),
            policy_pair_context_bias: self.policy_pair_context_bias.clone(),
            policy_pair_embedding: self.policy_pair_embedding.clone(),
            policy_move_context_hidden: self.policy_move_context_hidden.clone(),
            policy_move_embedding: self.policy_move_embedding.clone(),
            policy_relation_context_hidden: self.policy_relation_context_hidden.clone(),
            policy_relation_context_bias: self.policy_relation_context_bias.clone(),
            policy_relation_from_embedding: self.policy_relation_from_embedding.clone(),
            policy_relation_to_embedding: self.policy_relation_to_embedding.clone(),
            policy_relation_pair_embedding: self.policy_relation_pair_embedding.clone(),
            policy_relation_output: self.policy_relation_output.clone(),
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
    pub mirror_probability: f32,
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
    pub learning_rate: f32,
    pub value_loss: f32,
    pub value_mse: f32,
    pub value_pred_mean: f32,
    pub value_target_mean: f32,
    pub value_pred_rms: f32,
    pub value_target_rms: f32,
    pub value_corr: f32,
    pub value_calibration: f32,
    pub policy_ce: f32,
    pub policy_kl: f32,
    pub root_visit_entropy: f32,
    pub entropy_opening: f32,
    pub entropy_mid: f32,
    pub raw_prior_top1: f32,
    pub raw_prior_top2: f32,
    pub policy_top1: f32,
    pub policy_top2: f32,
    pub root_q_gap: f32,
    pub root_q_top1_abs: f32,
    pub visited_actions: f32,
    pub opening_raw_prior_top1: f32,
    pub opening_raw_prior_top2: f32,
    pub opening_policy_top1: f32,
    pub opening_policy_top2: f32,
    pub opening_q_gap: f32,
    pub opening_q_top1_abs: f32,
    pub opening_visited_actions: f32,
    pub selfplay_seconds: f32,
    pub train_seconds: f32,
    pub total_seconds: f32,
    pub games_per_second: f32,
    pub samples_per_second: f32,
    pub train_samples_per_second: f32,
    pub train_samples: usize,
    pub pool_samples: usize,
    pub pool_capacity: usize,
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

#[derive(Clone, Copy, Debug, Default)]
pub struct AzPolicyFitBenchmark {
    pub samples: usize,
    pub train_samples: usize,
    pub holdout_samples: usize,
    pub epochs_completed: usize,
    pub target_entropy: f32,
    pub initial_value_ce: f32,
    pub initial_value_mse: f32,
    pub initial_policy_ce: f32,
    pub initial_policy_kl: f32,
    pub final_value_ce: f32,
    pub final_value_mse: f32,
    pub final_policy_ce: f32,
    pub final_policy_kl: f32,
    pub holdout_target_entropy: f32,
    pub holdout_initial_value_ce: f32,
    pub holdout_initial_value_mse: f32,
    pub holdout_initial_policy_ce: f32,
    pub holdout_initial_policy_kl: f32,
    pub holdout_final_value_ce: f32,
    pub holdout_final_value_mse: f32,
    pub holdout_final_policy_ce: f32,
    pub holdout_final_policy_kl: f32,
    pub train_loss: f32,
    pub train_value_loss: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AzPolicyFitEpochReport {
    pub epoch: usize,
    pub train_target_entropy: f32,
    pub train_value_ce: f32,
    pub train_value_mse: f32,
    pub train_policy_ce: f32,
    pub train_policy_kl: f32,
    pub holdout_target_entropy: f32,
    pub holdout_value_ce: f32,
    pub holdout_value_mse: f32,
    pub holdout_policy_ce: f32,
    pub holdout_policy_kl: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AzSelfplayPolicyFitBenchmark {
    pub games: usize,
    pub samples: usize,
    pub epochs_completed: usize,
    pub red_wins: usize,
    pub black_wins: usize,
    pub draws: usize,
    pub avg_plies: f32,
    pub selfplay_seconds: f32,
    pub target_entropy: f32,
    pub initial_value_ce: f32,
    pub initial_value_mse: f32,
    pub initial_policy_ce: f32,
    pub initial_policy_kl: f32,
    pub final_value_ce: f32,
    pub final_value_mse: f32,
    pub final_policy_ce: f32,
    pub final_policy_kl: f32,
    pub train_loss: f32,
    pub train_value_loss: f32,
}

#[derive(Clone, Debug)]
pub struct AzTrainingSample {
    pub features: Vec<usize>,
    pub move_indices: Vec<usize>,
    pub policy_dynamic_features: Vec<f32>,
    pub policy: Vec<f32>,
    pub value: f32,
    pub side_sign: f32,
    pub moves_left: f32,
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
    pub value_pred_target_sum: f32,
    pub value_error_sq_sum: f32,
    pub samples: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct AzTrainLossWeights {
    pub value: f32,
    pub policy: f32,
    pub train_trunk: bool,
    pub train_value_head: bool,
    pub train_policy_head: bool,
}

impl Default for AzTrainLossWeights {
    fn default() -> Self {
        Self {
            value: 1.0,
            policy: 1.0,
            train_trunk: true,
            train_value_head: true,
            train_policy_head: true,
        }
    }
}

impl AzTrainStats {
    #[cfg_attr(not(feature = "gpu-train"), allow(dead_code))]
    fn add_assign(&mut self, other: &Self) {
        self.loss += other.loss;
        self.value_loss += other.value_loss;
        self.policy_ce += other.policy_ce;
        self.value_pred_sum += other.value_pred_sum;
        self.value_pred_sq_sum += other.value_pred_sq_sum;
        self.value_target_sum += other.value_target_sum;
        self.value_target_sq_sum += other.value_target_sq_sum;
        self.value_pred_target_sum += other.value_pred_target_sum;
        self.value_error_sq_sum += other.value_error_sq_sum;
        self.samples += other.samples;
    }
}

impl AzNnue {
    #[cfg_attr(not(feature = "gpu-train"), allow(dead_code))]
    pub(super) fn refresh_policy_derived_caches(&mut self) {
        self.policy_action_static_projection =
            build_policy_action_static_projection(&self.policy_action_static_hidden);
        self.policy_relation_pair_embedding = build_policy_relation_pair_embedding(
            &self.policy_relation_from_embedding,
            &self.policy_relation_to_embedding,
        );
    }

    pub fn random_with_arch(arch: AzNnueArch, seed: u64) -> Self {
        if let Err(err) = arch.validate() {
            panic!("AzNnue::random_with_arch: invalid arch ({err})");
        }
        let hidden_size = arch.hidden_size;
        let mut rng = SplitMix64::new(seed);
        let input_hidden: Vec<f32> = (0..AZ_NNUE_INPUT_SIZE * hidden_size)
            .map(|_| rng.weight(0.015))
            .collect();
        // Learned structural factors recover row/file/material/king context from
        // piece-square facts without reintroducing those handcrafted feature ids.
        let input_piece_hidden = vec![0.0; STRUCTURAL_PIECE_SIZE * hidden_size];
        let input_rank_hidden = vec![0.0; STRUCTURAL_RANK_SIZE * hidden_size];
        let input_file_hidden = vec![0.0; STRUCTURAL_FILE_SIZE * hidden_size];
        let input_king_piece_hidden = vec![0.0; STRUCTURAL_KING_PIECE_SIZE * hidden_size];
        let hidden_bias = vec![0.0; hidden_size];
        // Learned second-order pooling over the additive sparse trunk. It is
        // initialized as a no-op, then training can open cheap global
        // co-occurrence channels for tactical combinations.
        let input_quadratic_scale = vec![0.0; hidden_size];
        let piece_attention_query = (0..hidden_size)
            .map(|_| rng.weight((1.0 / hidden_size.max(1) as f32).sqrt()))
            .collect();
        let piece_attention_value = (0..PIECE_ATTENTION_SIZE * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt() * 0.25))
            .collect();
        // Sparse attention starts as an exact residual no-op. Training can then
        // open the output gate without perturbing the first self-play games.
        let piece_attention_output = vec![0.0; hidden_size * PIECE_ATTENTION_SIZE];
        let square_piece_token = (0..SQUARE_TOKEN_PIECES * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt() * 0.25))
            .collect();
        let square_position_token = (0..BOARD_SIZE * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt() * 0.25))
            .collect();
        let line_mixer_down = (0..LINE_MIXER_RANK * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt() * 0.25))
            .collect();
        let line_mixer_bias = vec![0.0; LINE_MIXER_RANK];
        let line_mixer_up = vec![0.0; hidden_size * LINE_MIXER_RANK];
        let line_context_bias = vec![0.0; hidden_size];
        let trunk_residual_hidden = vec![0.0; TRUNK_LAYERS * hidden_size * hidden_size];
        let trunk_residual_bias = vec![0.0; TRUNK_LAYERS * hidden_size];
        let auto_feature_hidden = (0..AUTO_FEATURE_SIZE * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt() * 0.25))
            .collect();
        let auto_feature_bias = vec![0.0; AUTO_FEATURE_SIZE];
        // Residual output starts at exact zero, so fresh models are behaviorally
        // identical to the additive sparse trunk until training discovers useful
        // feature interactions.
        let auto_feature_output = vec![0.0; hidden_size * AUTO_FEATURE_SIZE];
        // Start value-neutral. A random value head can evaluate startpos as a
        // large red/black advantage before any training, and MCTS amplifies
        // that noise into the first self-play dataset.
        let value_head_hidden = (0..VALUE_HEAD_SIZE * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt() * 0.5))
            .collect();
        let value_head_bias = vec![0.0; VALUE_HEAD_SIZE];
        // Keep the value head output-neutral at initialization. This preserves
        // stable first self-play while giving value its own nonlinear capacity.
        let value_head_output = vec![0.0; VALUE_HEAD_SIZE];
        let moves_left_output = vec![0.0; VALUE_HEAD_SIZE];
        let moves_left_bias = vec![0.0; 1];
        let policy_move_bias = vec![0.0; DENSE_MOVE_SPACE];
        let policy_context_hidden = (0..POLICY_CONTEXT_SIZE * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt() * 0.5))
            .collect();
        let policy_context_bias = vec![0.0; POLICY_CONTEXT_SIZE];
        let policy_feature_hidden = (0..POLICY_CONDITION_SIZE * POLICY_CONTEXT_SIZE)
            .map(|_| rng.weight((2.0 / POLICY_CONTEXT_SIZE as f32).sqrt() * 0.5))
            .collect();
        let policy_feature_bias = vec![0.0; POLICY_CONDITION_SIZE];
        let policy_dynamic_hidden = (0..POLICY_DYNAMIC_FEATURE_SIZE * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt() * 0.1))
            .collect();
        let policy_action_context_hidden = (0..POLICY_ACTION_SIZE * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt() * 0.25))
            .collect();
        let policy_action_context_bias = vec![0.0; POLICY_ACTION_SIZE];
        let policy_action_static_hidden: Vec<f32> = (0..POLICY_ACTION_SIZE * POLICY_CONDITION_SIZE)
            .map(|_| rng.weight((2.0 / POLICY_CONDITION_SIZE as f32).sqrt() * 0.25))
            .collect();
        let policy_action_static_projection =
            build_policy_action_static_projection(&policy_action_static_hidden);
        let policy_action_dynamic_hidden = (0..POLICY_ACTION_SIZE * POLICY_DYNAMIC_FEATURE_SIZE)
            .map(|_| rng.weight((2.0 / POLICY_DYNAMIC_FEATURE_SIZE as f32).sqrt() * 0.25))
            .collect();
        // Start the structural action branch as an exact residual no-op. The
        // branch can learn policy corrections, but fresh self-play should
        // behave like the simpler first-version policy head instead of adding
        // random per-move noise.
        let policy_action_output = vec![0.0; POLICY_ACTION_SIZE];
        let policy_from_hidden = (0..BOARD_SIZE * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt() * 0.25))
            .collect();
        let policy_to_hidden = (0..BOARD_SIZE * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt() * 0.25))
            .collect();
        let policy_pair_context_hidden = (0..POLICY_PAIR_CONTEXT_SIZE * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt() * 0.25))
            .collect();
        let policy_pair_context_bias = vec![0.0; POLICY_PAIR_CONTEXT_SIZE];
        let policy_pair_embedding = (0..DENSE_MOVE_SPACE * POLICY_PAIR_CONTEXT_SIZE)
            .map(|_| rng.weight((2.0 / POLICY_PAIR_CONTEXT_SIZE as f32).sqrt() * 0.1))
            .collect();
        let policy_move_context_hidden = (0..POLICY_MOVE_EMBED_SIZE * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt() * 0.25))
            .collect();
        let policy_move_embedding = vec![0.0; DENSE_MOVE_SPACE * POLICY_MOVE_EMBED_SIZE];
        let policy_relation_context_hidden = (0..POLICY_RELATION_SIZE * hidden_size)
            .map(|_| rng.weight((2.0 / hidden_size.max(1) as f32).sqrt() * 0.25))
            .collect();
        let policy_relation_context_bias = vec![0.0; POLICY_RELATION_SIZE];
        let policy_relation_from_embedding: Vec<f32> = (0..BOARD_SIZE * POLICY_RELATION_SIZE)
            .map(|_| rng.weight((2.0 / POLICY_RELATION_SIZE as f32).sqrt() * 0.1))
            .collect();
        let policy_relation_to_embedding: Vec<f32> = (0..BOARD_SIZE * POLICY_RELATION_SIZE)
            .map(|_| rng.weight((2.0 / POLICY_RELATION_SIZE as f32).sqrt() * 0.1))
            .collect();
        let policy_relation_pair_embedding = build_policy_relation_pair_embedding(
            &policy_relation_from_embedding,
            &policy_relation_to_embedding,
        );
        // Relation logits are a residual policy branch. Start at exact no-op so
        // the first self-play distribution matches the parent model structure.
        let policy_relation_output = vec![0.0; POLICY_RELATION_SIZE];
        Self {
            hidden_size,
            arch,
            input_hidden,
            input_piece_hidden,
            input_rank_hidden,
            input_file_hidden,
            input_king_piece_hidden,
            hidden_bias,
            input_quadratic_scale,
            piece_attention_query,
            piece_attention_value,
            piece_attention_output,
            square_piece_token,
            square_position_token,
            line_mixer_down,
            line_mixer_bias,
            line_mixer_up,
            line_context_bias,
            trunk_residual_hidden,
            trunk_residual_bias,
            auto_feature_hidden,
            auto_feature_bias,
            auto_feature_output,
            value_head_hidden,
            value_head_bias,
            value_head_output,
            moves_left_output,
            moves_left_bias,
            policy_move_bias,
            policy_context_hidden,
            policy_context_bias,
            policy_feature_hidden,
            policy_feature_bias,
            policy_dynamic_hidden,
            policy_action_context_hidden,
            policy_action_context_bias,
            policy_action_static_hidden,
            policy_action_static_projection,
            policy_action_dynamic_hidden,
            policy_action_output,
            policy_from_hidden,
            policy_to_hidden,
            policy_pair_context_hidden,
            policy_pair_context_bias,
            policy_pair_embedding,
            policy_move_context_hidden,
            policy_move_embedding,
            policy_relation_context_hidden,
            policy_relation_context_bias,
            policy_relation_from_embedding,
            policy_relation_to_embedding,
            policy_relation_pair_embedding,
            policy_relation_output,
            gpu_trainer: None,
        }
    }

    pub fn random(hidden_size: usize, seed: u64) -> Self {
        Self::random_with_arch(AzNnueArch::with_hidden_size(hidden_size), seed)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let file = fs::File::create(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(AZNNUE_BINARY_MAGIC)?;
        writer.write_all(&AZNNUE_BINARY_VERSION.to_le_bytes())?;
        writer.write_all(&(AZ_NNUE_INPUT_SIZE as u32).to_le_bytes())?;
        writer.write_all(&(self.arch.hidden_size as u32).to_le_bytes())?;
        write_f32_slice_le(&mut writer, &self.input_hidden)?;
        write_f32_slice_le(&mut writer, &self.input_piece_hidden)?;
        write_f32_slice_le(&mut writer, &self.input_rank_hidden)?;
        write_f32_slice_le(&mut writer, &self.input_file_hidden)?;
        write_f32_slice_le(&mut writer, &self.input_king_piece_hidden)?;
        write_f32_slice_le(&mut writer, &self.hidden_bias)?;
        write_f32_slice_le(&mut writer, &self.input_quadratic_scale)?;
        write_f32_slice_le(&mut writer, &self.piece_attention_query)?;
        write_f32_slice_le(&mut writer, &self.piece_attention_value)?;
        write_f32_slice_le(&mut writer, &self.piece_attention_output)?;
        write_f32_slice_le(&mut writer, &self.square_piece_token)?;
        write_f32_slice_le(&mut writer, &self.square_position_token)?;
        write_f32_slice_le(&mut writer, &self.line_mixer_down)?;
        write_f32_slice_le(&mut writer, &self.line_mixer_bias)?;
        write_f32_slice_le(&mut writer, &self.line_mixer_up)?;
        write_f32_slice_le(&mut writer, &self.line_context_bias)?;
        write_f32_slice_le(&mut writer, &self.trunk_residual_hidden)?;
        write_f32_slice_le(&mut writer, &self.trunk_residual_bias)?;
        write_f32_slice_le(&mut writer, &self.auto_feature_hidden)?;
        write_f32_slice_le(&mut writer, &self.auto_feature_bias)?;
        write_f32_slice_le(&mut writer, &self.auto_feature_output)?;
        write_f32_slice_le(&mut writer, &self.value_head_hidden)?;
        write_f32_slice_le(&mut writer, &self.value_head_bias)?;
        write_f32_slice_le(&mut writer, &self.value_head_output)?;
        write_f32_slice_le(&mut writer, &self.moves_left_output)?;
        write_f32_slice_le(&mut writer, &self.moves_left_bias)?;
        write_f32_slice_le(&mut writer, &self.policy_move_bias)?;
        write_f32_slice_le(&mut writer, &self.policy_context_hidden)?;
        write_f32_slice_le(&mut writer, &self.policy_context_bias)?;
        write_f32_slice_le(&mut writer, &self.policy_feature_hidden)?;
        write_f32_slice_le(&mut writer, &self.policy_feature_bias)?;
        write_f32_slice_le(&mut writer, &self.policy_dynamic_hidden)?;
        write_f32_slice_le(&mut writer, &self.policy_action_context_hidden)?;
        write_f32_slice_le(&mut writer, &self.policy_action_context_bias)?;
        write_f32_slice_le(&mut writer, &self.policy_action_static_hidden)?;
        write_f32_slice_le(&mut writer, &self.policy_action_dynamic_hidden)?;
        write_f32_slice_le(&mut writer, &self.policy_action_output)?;
        write_f32_slice_le(&mut writer, &self.policy_from_hidden)?;
        write_f32_slice_le(&mut writer, &self.policy_to_hidden)?;
        write_f32_slice_le(&mut writer, &self.policy_pair_context_hidden)?;
        write_f32_slice_le(&mut writer, &self.policy_pair_context_bias)?;
        write_f32_slice_le(&mut writer, &self.policy_pair_embedding)?;
        write_f32_slice_le(&mut writer, &self.policy_move_context_hidden)?;
        write_f32_slice_le(&mut writer, &self.policy_move_embedding)?;
        write_f32_slice_le(&mut writer, &self.policy_relation_context_hidden)?;
        write_f32_slice_le(&mut writer, &self.policy_relation_context_bias)?;
        write_f32_slice_le(&mut writer, &self.policy_relation_from_embedding)?;
        write_f32_slice_le(&mut writer, &self.policy_relation_to_embedding)?;
        write_f32_slice_le(&mut writer, &self.policy_relation_output)?;
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
        if input_size != AZ_NNUE_INPUT_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "binary input_size does not match this build (AZ_NNUE_INPUT_SIZE)",
            ));
        }
        let arch = AzNnueArch { hidden_size };
        if let Err(err) = arch.validate() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("AZNNUE binary header arch invalid: {err}"),
            ));
        }
        let input_hidden_len = AZ_NNUE_INPUT_SIZE * hidden_size;
        let iph_len = STRUCTURAL_PIECE_SIZE * hidden_size;
        let irh_len = STRUCTURAL_RANK_SIZE * hidden_size;
        let ifh_len = STRUCTURAL_FILE_SIZE * hidden_size;
        let ikph_len = STRUCTURAL_KING_PIECE_SIZE * hidden_size;
        let hidden_bias_len = hidden_size;
        let iqs_len = hidden_size;
        let attn_query_len = hidden_size;
        let attn_value_len = PIECE_ATTENTION_SIZE * hidden_size;
        let attn_output_len = hidden_size * PIECE_ATTENTION_SIZE;
        let spt_len = SQUARE_TOKEN_PIECES * hidden_size;
        let sqt_len = BOARD_SIZE * hidden_size;
        let lmd_len = LINE_MIXER_RANK * hidden_size;
        let lmb_len = LINE_MIXER_RANK;
        let lmu_len = hidden_size * LINE_MIXER_RANK;
        let lcb_len = hidden_size;
        let trh_len = TRUNK_LAYERS * hidden_size * hidden_size;
        let trb_len = TRUNK_LAYERS * hidden_size;
        let afh_len = AUTO_FEATURE_SIZE * hidden_size;
        let afb_len = AUTO_FEATURE_SIZE;
        let afo_len = hidden_size * AUTO_FEATURE_SIZE;
        let vhh_len = VALUE_HEAD_SIZE * hidden_size;
        let vhb_len = VALUE_HEAD_SIZE;
        let vho_len = VALUE_HEAD_SIZE;
        let mlo_len = VALUE_HEAD_SIZE;
        let mlb_len = 1;
        let pmb_len = DENSE_MOVE_SPACE;
        let pch_len = POLICY_CONTEXT_SIZE * hidden_size;
        let pcb_len = POLICY_CONTEXT_SIZE;
        let pfh_len = POLICY_CONDITION_SIZE * POLICY_CONTEXT_SIZE;
        let pfb_len = POLICY_CONDITION_SIZE;
        let pdh_len = POLICY_DYNAMIC_FEATURE_SIZE * hidden_size;
        let pach_len = POLICY_ACTION_SIZE * hidden_size;
        let pacb_len = POLICY_ACTION_SIZE;
        let pash_len = POLICY_ACTION_SIZE * POLICY_CONDITION_SIZE;
        let padyh_len = POLICY_ACTION_SIZE * POLICY_DYNAMIC_FEATURE_SIZE;
        let pao_len = POLICY_ACTION_SIZE;
        let pfrom_len = BOARD_SIZE * hidden_size;
        let pto_len = BOARD_SIZE * hidden_size;
        let ppch_len = POLICY_PAIR_CONTEXT_SIZE * hidden_size;
        let ppcb_len = POLICY_PAIR_CONTEXT_SIZE;
        let ppe_len = DENSE_MOVE_SPACE * POLICY_PAIR_CONTEXT_SIZE;
        let pmch_len = POLICY_MOVE_EMBED_SIZE * hidden_size;
        let pme_len = DENSE_MOVE_SPACE * POLICY_MOVE_EMBED_SIZE;
        let prch_len = POLICY_RELATION_SIZE * hidden_size;
        let prcb_len = POLICY_RELATION_SIZE;
        let prfe_len = BOARD_SIZE * POLICY_RELATION_SIZE;
        let prte_len = BOARD_SIZE * POLICY_RELATION_SIZE;
        let pro_len = POLICY_RELATION_SIZE;
        let float_count = input_hidden_len
            + iph_len
            + irh_len
            + ifh_len
            + ikph_len
            + hidden_bias_len
            + iqs_len
            + attn_query_len
            + attn_value_len
            + attn_output_len
            + spt_len
            + sqt_len
            + lmd_len
            + lmb_len
            + lmu_len
            + lcb_len
            + trh_len
            + trb_len
            + afh_len
            + afb_len
            + afo_len
            + vhh_len
            + vhb_len
            + vho_len
            + mlo_len
            + mlb_len
            + pmb_len
            + pch_len
            + pcb_len
            + pfh_len
            + pfb_len
            + pdh_len
            + pach_len
            + pacb_len
            + pash_len
            + padyh_len
            + pao_len
            + pfrom_len
            + pto_len
            + ppch_len
            + ppcb_len
            + ppe_len
            + pmch_len
            + pme_len
            + prch_len
            + prcb_len
            + prfe_len
            + prte_len
            + pro_len;
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
        let input_piece_hidden = read_f32_vec_le(&mut reader, iph_len)?;
        let input_rank_hidden = read_f32_vec_le(&mut reader, irh_len)?;
        let input_file_hidden = read_f32_vec_le(&mut reader, ifh_len)?;
        let input_king_piece_hidden = read_f32_vec_le(&mut reader, ikph_len)?;
        let hidden_bias = read_f32_vec_le(&mut reader, hidden_bias_len)?;
        let input_quadratic_scale = read_f32_vec_le(&mut reader, iqs_len)?;
        let piece_attention_query = read_f32_vec_le(&mut reader, attn_query_len)?;
        let piece_attention_value = read_f32_vec_le(&mut reader, attn_value_len)?;
        let piece_attention_output = read_f32_vec_le(&mut reader, attn_output_len)?;
        let square_piece_token = read_f32_vec_le(&mut reader, spt_len)?;
        let square_position_token = read_f32_vec_le(&mut reader, sqt_len)?;
        let line_mixer_down = read_f32_vec_le(&mut reader, lmd_len)?;
        let line_mixer_bias = read_f32_vec_le(&mut reader, lmb_len)?;
        let line_mixer_up = read_f32_vec_le(&mut reader, lmu_len)?;
        let line_context_bias = read_f32_vec_le(&mut reader, lcb_len)?;
        let trunk_residual_hidden = read_f32_vec_le(&mut reader, trh_len)?;
        let trunk_residual_bias = read_f32_vec_le(&mut reader, trb_len)?;
        let auto_feature_hidden = read_f32_vec_le(&mut reader, afh_len)?;
        let auto_feature_bias = read_f32_vec_le(&mut reader, afb_len)?;
        let auto_feature_output = read_f32_vec_le(&mut reader, afo_len)?;
        let value_head_hidden = read_f32_vec_le(&mut reader, vhh_len)?;
        let value_head_bias = read_f32_vec_le(&mut reader, vhb_len)?;
        let value_head_output = read_f32_vec_le(&mut reader, vho_len)?;
        let moves_left_output = read_f32_vec_le(&mut reader, mlo_len)?;
        let moves_left_bias = read_f32_vec_le(&mut reader, mlb_len)?;
        let policy_move_bias = read_f32_vec_le(&mut reader, pmb_len)?;
        let policy_context_hidden = read_f32_vec_le(&mut reader, pch_len)?;
        let policy_context_bias = read_f32_vec_le(&mut reader, pcb_len)?;
        let policy_feature_hidden = read_f32_vec_le(&mut reader, pfh_len)?;
        let policy_feature_bias = read_f32_vec_le(&mut reader, pfb_len)?;
        let policy_dynamic_hidden = read_f32_vec_le(&mut reader, pdh_len)?;
        let policy_action_context_hidden = read_f32_vec_le(&mut reader, pach_len)?;
        let policy_action_context_bias = read_f32_vec_le(&mut reader, pacb_len)?;
        let policy_action_static_hidden = read_f32_vec_le(&mut reader, pash_len)?;
        let policy_action_static_projection =
            build_policy_action_static_projection(&policy_action_static_hidden);
        let policy_action_dynamic_hidden = read_f32_vec_le(&mut reader, padyh_len)?;
        let policy_action_output = read_f32_vec_le(&mut reader, pao_len)?;
        let policy_from_hidden = read_f32_vec_le(&mut reader, pfrom_len)?;
        let policy_to_hidden = read_f32_vec_le(&mut reader, pto_len)?;
        let policy_pair_context_hidden = read_f32_vec_le(&mut reader, ppch_len)?;
        let policy_pair_context_bias = read_f32_vec_le(&mut reader, ppcb_len)?;
        let policy_pair_embedding = read_f32_vec_le(&mut reader, ppe_len)?;
        let policy_move_context_hidden = read_f32_vec_le(&mut reader, pmch_len)?;
        let policy_move_embedding = read_f32_vec_le(&mut reader, pme_len)?;
        let policy_relation_context_hidden = read_f32_vec_le(&mut reader, prch_len)?;
        let policy_relation_context_bias = read_f32_vec_le(&mut reader, prcb_len)?;
        let policy_relation_from_embedding = read_f32_vec_le(&mut reader, prfe_len)?;
        let policy_relation_to_embedding = read_f32_vec_le(&mut reader, prte_len)?;
        let policy_relation_pair_embedding = build_policy_relation_pair_embedding(
            &policy_relation_from_embedding,
            &policy_relation_to_embedding,
        );
        let policy_relation_output = read_f32_vec_le(&mut reader, pro_len)?;
        let model = Self {
            hidden_size,
            arch,
            input_hidden,
            input_piece_hidden,
            input_rank_hidden,
            input_file_hidden,
            input_king_piece_hidden,
            hidden_bias,
            input_quadratic_scale,
            piece_attention_query,
            piece_attention_value,
            piece_attention_output,
            square_piece_token,
            square_position_token,
            line_mixer_down,
            line_mixer_bias,
            line_mixer_up,
            line_context_bias,
            trunk_residual_hidden,
            trunk_residual_bias,
            auto_feature_hidden,
            auto_feature_bias,
            auto_feature_output,
            value_head_hidden,
            value_head_bias,
            value_head_output,
            moves_left_output,
            moves_left_bias,
            policy_move_bias,
            policy_context_hidden,
            policy_context_bias,
            policy_feature_hidden,
            policy_feature_bias,
            policy_dynamic_hidden,
            policy_action_context_hidden,
            policy_action_context_bias,
            policy_action_static_hidden,
            policy_action_static_projection,
            policy_action_dynamic_hidden,
            policy_action_output,
            policy_from_hidden,
            policy_to_hidden,
            policy_pair_context_hidden,
            policy_pair_context_bias,
            policy_pair_embedding,
            policy_move_context_hidden,
            policy_move_embedding,
            policy_relation_context_hidden,
            policy_relation_context_bias,
            policy_relation_from_embedding,
            policy_relation_to_embedding,
            policy_relation_pair_embedding,
            policy_relation_output,
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
        crate::scope_profile!("az.evaluate_with_scratch");
        let features = {
            crate::scope_profile!("az.eval.extract_features");
            extract_sparse_features_az_canonical(position, history)
        };
        {
            crate::scope_profile!("az.eval.input_embedding");
            self.input_embedding_into(&features, &mut scratch.hidden);
            self.line_aware_trunk_into(
                position,
                &mut scratch.hidden,
                &mut scratch.line_repr,
                &mut scratch.line_bottleneck,
                &mut scratch.trunk_work,
            );
            self.auto_feature_adapter_into(&mut scratch.hidden, &mut scratch.auto_features);
            rms_norm_in_place(&mut scratch.hidden);
        }
        let value = {
            crate::scope_profile!("az.eval.value_head");
            self.value_from_hidden_into(&scratch.hidden, &features, &mut scratch.value_head)
        };
        self.evaluate_prepared_hidden_with_scratch(position, value, moves, scratch)
    }

    #[allow(dead_code)]
    fn evaluate_accumulator_with_scratch(
        &self,
        position: &Position,
        accumulator: &AzEvalAccumulator,
        history: &[HistoryMove],
        moves: &[Move],
        scratch: &mut AzEvalScratch,
    ) -> f32 {
        crate::scope_profile!("az.evaluate_accumulator_with_scratch");
        {
            crate::scope_profile!("az.eval.input_embedding");
            scratch.hidden.resize(self.hidden_size, 0.0);
            scratch.hidden.copy_from_slice(&accumulator.hidden_sum);
            self.add_history_features_to_hidden(
                history,
                &mut scratch.history_features,
                &mut scratch.hidden,
            );
            let features = extract_sparse_features_az_canonical(position, history);
            self.add_sparse_attention_into(&features, &mut scratch.hidden);
            relu_in_place(&mut scratch.hidden);
            self.line_aware_trunk_into(
                position,
                &mut scratch.hidden,
                &mut scratch.line_repr,
                &mut scratch.line_bottleneck,
                &mut scratch.trunk_work,
            );
            self.auto_feature_adapter_into(&mut scratch.hidden, &mut scratch.auto_features);
            rms_norm_in_place(&mut scratch.hidden);
        }
        let value = {
            crate::scope_profile!("az.eval.value_head");
            let features = extract_sparse_features_az_canonical(position, history);
            self.value_from_hidden_into(&scratch.hidden, &features, &mut scratch.value_head)
        };
        self.evaluate_prepared_hidden_with_scratch(position, value, moves, scratch)
    }

    #[allow(dead_code)]
    fn add_history_features_to_hidden(
        &self,
        history: &[HistoryMove],
        features: &mut Vec<usize>,
        hidden: &mut [f32],
    ) {
        features.clear();
        add_az_absolute_history_features(history, features);
        for &feature in features.iter() {
            add_scaled_feature_row(hidden, &self.input_hidden, self.hidden_size, feature, 1.0);
        }
    }

    fn evaluate_prepared_hidden_with_scratch(
        &self,
        position: &Position,
        value: f32,
        moves: &[Move],
        scratch: &mut AzEvalScratch,
    ) -> f32 {
        {
            crate::scope_profile!("az.eval.policy_contexts");
            self.policy_dynamic_condition_into(
                &scratch.hidden,
                &mut scratch.policy_dynamic_condition,
            );
            self.policy_action_context_into(&scratch.hidden, &mut scratch.policy_action_context);
            self.policy_pair_context_into(&scratch.hidden, &mut scratch.policy_pair_context);
            self.policy_move_context_into(&scratch.hidden, &mut scratch.policy_move_context);
            self.policy_relation_context_into(
                &scratch.hidden,
                &mut scratch.policy_relation_context,
            );
        }
        scratch.logits.resize(moves.len(), 0.0);
        let move_map = move_map();
        let mut from_used = [false; BOARD_SIZE];
        let mut to_used = [false; BOARD_SIZE];
        let mut from_squares = [0usize; BOARD_SIZE];
        let mut to_squares = [0usize; BOARD_SIZE];
        let mut from_count = 0usize;
        let mut to_count = 0usize;
        let side = position.side_to_move();
        for mv in moves {
            let canonical = canonical_move(side, *mv);
            let from = canonical.from as usize;
            let to = canonical.to as usize;
            if !from_used[from] {
                from_used[from] = true;
                from_squares[from_count] = from;
                from_count += 1;
            }
            if !to_used[to] {
                to_used[to] = true;
                to_squares[to_count] = to;
                to_count += 1;
            }
        }
        {
            crate::scope_profile!("az.eval.policy_square_scores");
            self.policy_square_scores_for_squares_into(
                &scratch.hidden,
                &from_squares[..from_count],
                &to_squares[..to_count],
                &mut scratch.policy_from_scores,
                &mut scratch.policy_to_scores,
            );
        }
        {
            crate::scope_profile!("az.eval.policy_logits");
            let policy_feature_cache = PolicyFeaturePositionCache::new_for_moves(position, moves);
            for (index, mv) in moves.iter().enumerate() {
                let canonical = canonical_move(side, *mv);
                let sparse = canonical.from as usize * BOARD_SIZE + canonical.to as usize;
                let dense = move_map.sparse_to_dense[sparse];
                debug_assert!(
                    dense != u16::MAX,
                    "invalid policy move {}->{}",
                    mv.from,
                    mv.to
                );
                let move_index = dense as usize;
                scratch.logits[index] = self.policy_logit_from_hidden_index(
                    &scratch.policy_dynamic_condition,
                    &scratch.policy_action_context,
                    &scratch.policy_pair_context,
                    &scratch.policy_move_context,
                    &scratch.policy_relation_context,
                    &scratch.policy_from_scores,
                    &scratch.policy_to_scores,
                    move_index,
                    &policy_dynamic_features_cached(position, *mv, &policy_feature_cache),
                );
            }
        }
        value
    }

    fn add_factorized_structure_into(&self, features: &[usize], hidden: &mut [f32]) {
        let (us_king_bucket, them_king_bucket) = canonical_general_buckets_from_features(features);
        for &feature in features {
            let Some(structural) = decode_current_piece_square_feature(feature) else {
                continue;
            };
            add_scaled_feature_row(
                hidden,
                &self.input_piece_hidden,
                self.hidden_size,
                structural.piece_index,
                1.0,
            );
            add_scaled_feature_row(
                hidden,
                &self.input_rank_hidden,
                self.hidden_size,
                structural.rank,
                1.0,
            );
            add_scaled_feature_row(
                hidden,
                &self.input_file_hidden,
                self.hidden_size,
                structural.file,
                1.0,
            );
            add_scaled_feature_row(
                hidden,
                &self.input_king_piece_hidden,
                self.hidden_size,
                structural_king_piece_index(0, us_king_bucket, structural.piece_index),
                1.0,
            );
            add_scaled_feature_row(
                hidden,
                &self.input_king_piece_hidden,
                self.hidden_size,
                structural_king_piece_index(1, them_king_bucket, structural.piece_index),
                1.0,
            );
        }
    }

    fn input_embedding_into(&self, features: &[usize], hidden: &mut Vec<f32>) {
        hidden.resize(self.hidden_size, 0.0);
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
                self.add_factorized_structure_into(features, hidden);
                self.add_quadratic_trunk_into(hidden);
                self.add_sparse_attention_into(features, hidden);
                // SAFETY: runtime detection above guarantees AVX2 support.
                unsafe {
                    relu_in_place_avx2(hidden);
                }
                return;
            }
        }
        for &feature in features {
            let row =
                &self.input_hidden[feature * self.hidden_size..(feature + 1) * self.hidden_size];
            for (left, &right) in hidden.iter_mut().zip(row) {
                *left += right;
            }
        }
        self.add_factorized_structure_into(features, hidden);
        self.add_quadratic_trunk_into(hidden);
        self.add_sparse_attention_into(features, hidden);
        relu_in_place(hidden);
    }

    fn add_quadratic_trunk_into(&self, hidden: &mut [f32]) {
        for (value, &scale) in hidden.iter_mut().zip(&self.input_quadratic_scale) {
            *value += scale * *value * *value;
        }
    }

    fn add_sparse_attention_into(&self, features: &[usize], hidden: &mut [f32]) {
        if features.is_empty()
            || self
                .piece_attention_output
                .iter()
                .all(|&value| value == 0.0)
        {
            return;
        }
        let hidden_size = self.hidden_size;
        let mut scores = Vec::with_capacity(features.len());
        let mut values = Vec::with_capacity(features.len() * PIECE_ATTENTION_SIZE);
        let mut token = vec![0.0; hidden_size];
        for &feature in features {
            if feature >= AZ_NNUE_INPUT_SIZE {
                continue;
            }
            self.sparse_attention_token_into(feature, features, &mut token);
            scores.push(dot_product(&token, &self.piece_attention_query));
            for attention_feature in 0..PIECE_ATTENTION_SIZE {
                let row = &self.piece_attention_value
                    [attention_feature * hidden_size..(attention_feature + 1) * hidden_size];
                values.push(dot_product(&token, row));
            }
        }
        if scores.is_empty() {
            return;
        }
        let max_score = scores
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, |left, right| left.max(right));
        let mut weight_sum = 0.0f32;
        for score in &mut scores {
            *score = (*score - max_score).exp();
            weight_sum += *score;
        }
        if !weight_sum.is_finite() || weight_sum <= 0.0 {
            return;
        }
        let mut context = vec![0.0; PIECE_ATTENTION_SIZE];
        for (token_index, &weight) in scores.iter().enumerate() {
            let weight = weight / weight_sum;
            let value_base = token_index * PIECE_ATTENTION_SIZE;
            for attention_feature in 0..PIECE_ATTENTION_SIZE {
                context[attention_feature] += weight * values[value_base + attention_feature];
            }
        }
        for (hidden_index, hidden_value) in hidden.iter_mut().enumerate() {
            let row = &self.piece_attention_output
                [hidden_index * PIECE_ATTENTION_SIZE..(hidden_index + 1) * PIECE_ATTENTION_SIZE];
            *hidden_value += dot_product(&context, row);
        }
    }

    fn sparse_attention_token_into(&self, feature: usize, features: &[usize], token: &mut [f32]) {
        let hidden_size = self.hidden_size;
        token.copy_from_slice(
            &self.input_hidden[feature * hidden_size..(feature + 1) * hidden_size],
        );
        let Some(structural) = decode_current_piece_square_feature(feature) else {
            return;
        };
        let (us_king_bucket, them_king_bucket) = canonical_general_buckets_from_features(features);
        add_scaled_feature_row(
            token,
            &self.input_piece_hidden,
            hidden_size,
            structural.piece_index,
            1.0,
        );
        add_scaled_feature_row(
            token,
            &self.input_rank_hidden,
            hidden_size,
            structural.rank,
            1.0,
        );
        add_scaled_feature_row(
            token,
            &self.input_file_hidden,
            hidden_size,
            structural.file,
            1.0,
        );
        add_scaled_feature_row(
            token,
            &self.input_king_piece_hidden,
            hidden_size,
            structural_king_piece_index(0, us_king_bucket, structural.piece_index),
            1.0,
        );
        add_scaled_feature_row(
            token,
            &self.input_king_piece_hidden,
            hidden_size,
            structural_king_piece_index(1, them_king_bucket, structural.piece_index),
            1.0,
        );
    }

    fn line_aware_trunk_into(
        &self,
        position: &Position,
        hidden: &mut [f32],
        line_repr: &mut Vec<f32>,
        line_bottleneck: &mut Vec<f32>,
        trunk_work: &mut Vec<f32>,
    ) {
        self.add_line_context_into(position, hidden, line_repr, line_bottleneck);
        relu_in_place(hidden);
        self.apply_residual_trunk_into(hidden, trunk_work);
    }

    fn add_line_context_into(
        &self,
        position: &Position,
        hidden: &mut [f32],
        line_repr: &mut Vec<f32>,
        line_bottleneck: &mut Vec<f32>,
    ) {
        line_bottleneck.resize(LINE_MIXER_RANK, 0.0);
        line_bottleneck.fill(0.0);
        line_repr.resize(self.hidden_size, 0.0);

        let mut lines = 0usize;
        for rank in 0..BOARD_RANKS {
            line_repr.fill(0.0);
            for file in 0..BOARD_FILES {
                let sq = rank * BOARD_FILES + file;
                self.add_square_token_to_line(position, sq, line_repr);
            }
            self.add_line_bottleneck(line_repr, line_bottleneck);
            lines += 1;
        }
        for file in 0..BOARD_FILES {
            line_repr.fill(0.0);
            for rank in 0..BOARD_RANKS {
                let sq = rank * BOARD_FILES + file;
                self.add_square_token_to_line(position, sq, line_repr);
            }
            self.add_line_bottleneck(line_repr, line_bottleneck);
            lines += 1;
        }
        debug_assert_eq!(lines, LINE_COUNT);

        for hidden_index in 0..self.hidden_size {
            let row = &self.line_mixer_up
                [hidden_index * LINE_MIXER_RANK..(hidden_index + 1) * LINE_MIXER_RANK];
            hidden[hidden_index] +=
                self.line_context_bias[hidden_index] + dot_product(line_bottleneck, row);
        }
    }

    fn add_square_token_to_line(
        &self,
        position: &Position,
        canonical_sq: usize,
        line_repr: &mut [f32],
    ) {
        let actual_sq = canonical_square(position.side_to_move(), canonical_sq);
        let piece_index = position
            .piece_at(actual_sq)
            .map(|piece| square_token_piece_index(piece, position.side_to_move()))
            .unwrap_or(SQUARE_TOKEN_PIECES - 1);
        add_scaled_feature_row(
            line_repr,
            &self.square_piece_token,
            self.hidden_size,
            piece_index,
            1.0,
        );
        add_scaled_feature_row(
            line_repr,
            &self.square_position_token,
            self.hidden_size,
            canonical_sq,
            1.0,
        );
    }

    fn add_line_bottleneck(&self, line_repr: &[f32], line_bottleneck: &mut [f32]) {
        for rank_feature in 0..LINE_MIXER_RANK {
            let row = &self.line_mixer_down
                [rank_feature * self.hidden_size..(rank_feature + 1) * self.hidden_size];
            line_bottleneck[rank_feature] +=
                (self.line_mixer_bias[rank_feature] + dot_product(line_repr, row)).max(0.0);
        }
    }

    fn apply_residual_trunk_into(&self, hidden: &mut [f32], trunk_work: &mut Vec<f32>) {
        trunk_work.resize(self.hidden_size, 0.0);
        for layer in 0..TRUNK_LAYERS {
            let weight_base = layer * self.hidden_size * self.hidden_size;
            let bias_base = layer * self.hidden_size;
            let weights = &self.trunk_residual_hidden
                [weight_base..weight_base + self.hidden_size * self.hidden_size];
            let bias = &self.trunk_residual_bias[bias_base..bias_base + self.hidden_size];
            if weights.iter().all(|&value| value == 0.0) && bias.iter().all(|&value| value == 0.0) {
                continue;
            }
            for out in 0..self.hidden_size {
                let row = &weights[out * self.hidden_size..(out + 1) * self.hidden_size];
                trunk_work[out] = hidden[out] + bias[out] + dot_product(hidden, row);
            }
            hidden.copy_from_slice(trunk_work);
            relu_in_place(hidden);
        }
    }

    fn auto_feature_adapter_into(&self, hidden: &mut [f32], auto_features: &mut Vec<f32>) {
        auto_features.resize(AUTO_FEATURE_SIZE, 0.0);
        auto_features.copy_from_slice(&self.auto_feature_bias);
        for feature in 0..AUTO_FEATURE_SIZE {
            let row = &self.auto_feature_hidden
                [feature * self.hidden_size..(feature + 1) * self.hidden_size];
            auto_features[feature] += dot_product(hidden, row);
            auto_features[feature] = auto_features[feature].max(0.0);
        }
        for (hidden_index, value) in hidden.iter_mut().enumerate() {
            let row = &self.auto_feature_output
                [hidden_index * AUTO_FEATURE_SIZE..(hidden_index + 1) * AUTO_FEATURE_SIZE];
            *value += dot_product(auto_features, row);
        }
        relu_in_place(hidden);
    }

    fn value_from_hidden_into(
        &self,
        hidden: &[f32],
        features: &[usize],
        value_head: &mut Vec<f32>,
    ) -> f32 {
        value_head.resize(VALUE_HEAD_SIZE, 0.0);
        value_head.copy_from_slice(&self.value_head_bias);
        for (feature, value) in value_head.iter_mut().enumerate().take(VALUE_HEAD_SIZE) {
            let hidden_row = &self.value_head_hidden
                [feature * self.hidden_size..(feature + 1) * self.hidden_size];
            *value += dot_product(hidden, hidden_row);
            *value = (*value).max(0.0);
        }
        let _ = features;
        dot_product(value_head, &self.value_head_output).tanh()
    }

    fn policy_dynamic_condition_into(&self, hidden: &[f32], out: &mut Vec<f32>) {
        out.resize(POLICY_DYNAMIC_FEATURE_SIZE, 0.0);
        for (feature, value) in out.iter_mut().enumerate().take(POLICY_DYNAMIC_FEATURE_SIZE) {
            let hidden_row = &self.policy_dynamic_hidden
                [feature * self.hidden_size..(feature + 1) * self.hidden_size];
            *value = dot_product(hidden, hidden_row);
        }
    }

    fn policy_action_context_into(&self, hidden: &[f32], out: &mut Vec<f32>) {
        out.resize(POLICY_ACTION_SIZE, 0.0);
        out.copy_from_slice(&self.policy_action_context_bias);
        for (feature, value) in out.iter_mut().enumerate().take(POLICY_ACTION_SIZE) {
            let hidden_row = &self.policy_action_context_hidden
                [feature * self.hidden_size..(feature + 1) * self.hidden_size];
            *value += dot_product(hidden, hidden_row);
        }
    }

    fn policy_square_scores_for_squares_into(
        &self,
        hidden: &[f32],
        from_squares: &[usize],
        to_squares: &[usize],
        from_scores: &mut Vec<f32>,
        to_scores: &mut Vec<f32>,
    ) {
        from_scores.resize(BOARD_SIZE, 0.0);
        to_scores.resize(BOARD_SIZE, 0.0);
        for &square in from_squares {
            let start = square * self.hidden_size;
            let end = start + self.hidden_size;
            from_scores[square] = dot_product(hidden, &self.policy_from_hidden[start..end]);
        }
        for &square in to_squares {
            let start = square * self.hidden_size;
            let end = start + self.hidden_size;
            to_scores[square] = dot_product(hidden, &self.policy_to_hidden[start..end]);
        }
    }

    fn policy_pair_context_into(&self, hidden: &[f32], out: &mut Vec<f32>) {
        out.resize(POLICY_PAIR_CONTEXT_SIZE, 0.0);
        out.copy_from_slice(&self.policy_pair_context_bias);
        for (feature, value) in out.iter_mut().enumerate().take(POLICY_PAIR_CONTEXT_SIZE) {
            let hidden_row = &self.policy_pair_context_hidden
                [feature * self.hidden_size..(feature + 1) * self.hidden_size];
            *value += dot_product(hidden, hidden_row);
            *value = (*value).max(0.0);
        }
    }

    fn policy_move_context_into(&self, hidden: &[f32], out: &mut Vec<f32>) {
        out.resize(POLICY_MOVE_EMBED_SIZE, 0.0);
        for (feature, value) in out.iter_mut().enumerate().take(POLICY_MOVE_EMBED_SIZE) {
            let hidden_row = &self.policy_move_context_hidden
                [feature * self.hidden_size..(feature + 1) * self.hidden_size];
            *value = dot_product(hidden, hidden_row);
        }
    }

    fn policy_relation_context_into(&self, hidden: &[f32], out: &mut Vec<f32>) {
        out.resize(POLICY_RELATION_SIZE, 0.0);
        out.copy_from_slice(&self.policy_relation_context_bias);
        for (feature, value) in out.iter_mut().enumerate().take(POLICY_RELATION_SIZE) {
            let hidden_row = &self.policy_relation_context_hidden
                [feature * self.hidden_size..(feature + 1) * self.hidden_size];
            *value += dot_product(hidden, hidden_row);
            *value = (*value).max(0.0);
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
        if self.input_hidden.len() != AZ_NNUE_INPUT_SIZE * hidden
            || self.input_piece_hidden.len() != STRUCTURAL_PIECE_SIZE * hidden
            || self.input_rank_hidden.len() != STRUCTURAL_RANK_SIZE * hidden
            || self.input_file_hidden.len() != STRUCTURAL_FILE_SIZE * hidden
            || self.input_king_piece_hidden.len() != STRUCTURAL_KING_PIECE_SIZE * hidden
            || self.hidden_bias.len() != hidden
            || self.input_quadratic_scale.len() != hidden
            || self.piece_attention_query.len() != hidden
            || self.piece_attention_value.len() != PIECE_ATTENTION_SIZE * hidden
            || self.piece_attention_output.len() != hidden * PIECE_ATTENTION_SIZE
            || self.square_piece_token.len() != SQUARE_TOKEN_PIECES * hidden
            || self.square_position_token.len() != BOARD_SIZE * hidden
            || self.line_mixer_down.len() != LINE_MIXER_RANK * hidden
            || self.line_mixer_bias.len() != LINE_MIXER_RANK
            || self.line_mixer_up.len() != hidden * LINE_MIXER_RANK
            || self.line_context_bias.len() != hidden
            || self.trunk_residual_hidden.len() != TRUNK_LAYERS * hidden * hidden
            || self.trunk_residual_bias.len() != TRUNK_LAYERS * hidden
            || self.auto_feature_hidden.len() != AUTO_FEATURE_SIZE * hidden
            || self.auto_feature_bias.len() != AUTO_FEATURE_SIZE
            || self.auto_feature_output.len() != hidden * AUTO_FEATURE_SIZE
            || self.value_head_hidden.len() != VALUE_HEAD_SIZE * hidden
            || self.value_head_bias.len() != VALUE_HEAD_SIZE
            || self.value_head_output.len() != VALUE_HEAD_SIZE
            || self.moves_left_output.len() != VALUE_HEAD_SIZE
            || self.moves_left_bias.len() != 1
            || self.policy_move_bias.len() != DENSE_MOVE_SPACE
            || self.policy_context_hidden.len() != POLICY_CONTEXT_SIZE * hidden
            || self.policy_context_bias.len() != POLICY_CONTEXT_SIZE
            || self.policy_feature_hidden.len() != POLICY_CONDITION_SIZE * POLICY_CONTEXT_SIZE
            || self.policy_feature_bias.len() != POLICY_CONDITION_SIZE
            || self.policy_dynamic_hidden.len() != POLICY_DYNAMIC_FEATURE_SIZE * hidden
            || self.policy_action_context_hidden.len() != POLICY_ACTION_SIZE * hidden
            || self.policy_action_context_bias.len() != POLICY_ACTION_SIZE
            || self.policy_action_static_hidden.len() != POLICY_ACTION_SIZE * POLICY_CONDITION_SIZE
            || self.policy_action_static_projection.len() != DENSE_MOVE_SPACE * POLICY_ACTION_SIZE
            || self.policy_action_dynamic_hidden.len()
                != POLICY_ACTION_SIZE * POLICY_DYNAMIC_FEATURE_SIZE
            || self.policy_action_output.len() != POLICY_ACTION_SIZE
            || self.policy_from_hidden.len() != BOARD_SIZE * hidden
            || self.policy_to_hidden.len() != BOARD_SIZE * hidden
            || self.policy_pair_context_hidden.len() != POLICY_PAIR_CONTEXT_SIZE * hidden
            || self.policy_pair_context_bias.len() != POLICY_PAIR_CONTEXT_SIZE
            || self.policy_pair_embedding.len() != DENSE_MOVE_SPACE * POLICY_PAIR_CONTEXT_SIZE
            || self.policy_move_context_hidden.len() != POLICY_MOVE_EMBED_SIZE * hidden
            || self.policy_move_embedding.len() != DENSE_MOVE_SPACE * POLICY_MOVE_EMBED_SIZE
            || self.policy_relation_context_hidden.len() != POLICY_RELATION_SIZE * hidden
            || self.policy_relation_context_bias.len() != POLICY_RELATION_SIZE
            || self.policy_relation_from_embedding.len() != BOARD_SIZE * POLICY_RELATION_SIZE
            || self.policy_relation_to_embedding.len() != BOARD_SIZE * POLICY_RELATION_SIZE
            || self.policy_relation_pair_embedding.len() != DENSE_MOVE_SPACE * POLICY_RELATION_SIZE
            || self.policy_relation_output.len() != POLICY_RELATION_SIZE
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
        policy_dynamic_condition: &[f32],
        policy_action_context: &[f32],
        policy_pair_context: &[f32],
        policy_move_context: &[f32],
        policy_relation_context: &[f32],
        from_scores: &[f32],
        to_scores: &[f32],
        move_index: usize,
        dynamic_features: &[f32],
    ) -> f32 {
        let sparse = move_map().dense_to_sparse[move_index] as usize;
        let from = sparse / BOARD_SIZE;
        let to = sparse % BOARD_SIZE;
        self.policy_move_bias[move_index]
            + dot_product(policy_dynamic_condition, dynamic_features)
            + self.policy_action_mlp_logit(policy_action_context, move_index, dynamic_features)
            + from_scores[from]
            + to_scores[to]
            + dot_product(
                policy_pair_context,
                &self.policy_pair_embedding[move_index * POLICY_PAIR_CONTEXT_SIZE
                    ..(move_index + 1) * POLICY_PAIR_CONTEXT_SIZE],
            )
            + dot_product(
                policy_move_context,
                &self.policy_move_embedding[move_index * POLICY_MOVE_EMBED_SIZE
                    ..(move_index + 1) * POLICY_MOVE_EMBED_SIZE],
            )
            + self.policy_relation_logit(policy_relation_context, move_index)
    }

    fn policy_relation_logit(&self, policy_relation_context: &[f32], move_index: usize) -> f32 {
        let pair_row = &self.policy_relation_pair_embedding
            [move_index * POLICY_RELATION_SIZE..(move_index + 1) * POLICY_RELATION_SIZE];
        let mut sum = 0.0f32;
        for feature in 0..POLICY_RELATION_SIZE {
            let value = policy_relation_context[feature] + pair_row[feature];
            sum += value.max(0.0) * self.policy_relation_output[feature];
        }
        sum
    }

    fn policy_action_mlp_logit(
        &self,
        policy_action_context: &[f32],
        move_index: usize,
        dynamic_features: &[f32],
    ) -> f32 {
        let mut sum = 0.0f32;
        let static_projection = &self.policy_action_static_projection
            [move_index * POLICY_ACTION_SIZE..(move_index + 1) * POLICY_ACTION_SIZE];
        for feature in 0..POLICY_ACTION_SIZE {
            let dynamic_row = &self.policy_action_dynamic_hidden[feature
                * POLICY_DYNAMIC_FEATURE_SIZE
                ..(feature + 1) * POLICY_DYNAMIC_FEATURE_SIZE];
            let move_action =
                static_projection[feature] + dot_product(dynamic_features, dynamic_row);
            let additive = (policy_action_context[feature] + move_action).max(0.0);
            let interaction = (policy_action_context[feature] * move_action).clamp(-10.0, 10.0);
            sum += (additive + 0.25 * interaction) * self.policy_action_output[feature];
        }
        sum
    }
}

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
            features.push((rng.next_u64() as usize) % AZ_NNUE_INPUT_SIZE);
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
        let mut policy_dynamic_features =
            Vec::with_capacity(move_count * POLICY_DYNAMIC_FEATURE_SIZE);
        for _ in 0..move_count {
            policy_dynamic_features.extend_from_slice(&[0.0; POLICY_DYNAMIC_FEATURE_SIZE]);
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
            move_indices,
            policy_dynamic_features,
            policy,
            value,
            side_sign: 1.0,
            moves_left: 0.0,
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

pub fn benchmark_policy_fit(
    model: &mut AzNnue,
    sample_count: usize,
    epochs: usize,
    batch_size: usize,
    lr: f32,
    seed: u64,
    teacher_seed: u64,
    max_random_plies: usize,
    target_temperature: f32,
) -> AzPolicyFitBenchmark {
    let mut rng = SplitMix64::new(seed);
    let teacher = AzNnue::random_with_arch(model.arch, teacher_seed);
    let samples = generate_policy_fit_samples(
        &teacher,
        sample_count,
        seed ^ 0xA5A5_5A5A_D3C1_B2E0,
        max_random_plies,
        target_temperature,
    );
    let before = policy_fit_eval(model, &samples);
    let (stats, after, epochs_completed) = fit_samples_with_early_stop(
        model,
        &samples,
        epochs,
        lr,
        batch_size,
        &mut rng,
        0,
        0.0,
        AzTrainLossWeights {
            value: 0.0,
            policy: 1.0,
            train_trunk: true,
            train_value_head: false,
            train_policy_head: true,
        },
    );
    AzPolicyFitBenchmark {
        samples: samples.len(),
        train_samples: samples.len(),
        holdout_samples: 0,
        epochs_completed,
        target_entropy: before.target_entropy,
        initial_value_ce: before.value_ce,
        initial_value_mse: before.value_mse,
        initial_policy_ce: before.policy_ce,
        initial_policy_kl: before.policy_ce - before.target_entropy,
        final_value_ce: after.value_ce,
        final_value_mse: after.value_mse,
        final_policy_ce: after.policy_ce,
        final_policy_kl: after.policy_ce - after.target_entropy,
        holdout_target_entropy: 0.0,
        holdout_initial_value_ce: 0.0,
        holdout_initial_value_mse: 0.0,
        holdout_initial_policy_ce: 0.0,
        holdout_initial_policy_kl: 0.0,
        holdout_final_value_ce: 0.0,
        holdout_final_value_mse: 0.0,
        holdout_final_policy_ce: 0.0,
        holdout_final_policy_kl: 0.0,
        train_loss: stats.loss,
        train_value_loss: stats.value_loss,
    }
}

pub fn benchmark_selfplay_policy_fit(
    model: &mut AzNnue,
    config: &AzLoopConfig,
    epochs: usize,
    batch_size: usize,
    lr: f32,
    seed: u64,
    early_stop_patience: usize,
    min_delta: f32,
) -> AzSelfplayPolicyFitBenchmark {
    let started = std::time::Instant::now();
    let selfplay = generate_selfplay_data(model, config);
    let selfplay_seconds = started.elapsed().as_secs_f32();
    let before = policy_fit_eval(model, &selfplay.samples);
    let mut rng = SplitMix64::new(seed);
    let (stats, after, epochs_completed) = fit_samples_with_early_stop(
        model,
        &selfplay.samples,
        epochs,
        lr,
        batch_size,
        &mut rng,
        early_stop_patience,
        min_delta,
        AzTrainLossWeights {
            value: 1.0,
            policy: 1.0,
            train_trunk: true,
            train_value_head: true,
            train_policy_head: true,
        },
    );
    let games = selfplay.games.len();
    AzSelfplayPolicyFitBenchmark {
        games,
        samples: selfplay.samples.len(),
        epochs_completed,
        red_wins: selfplay.red_wins,
        black_wins: selfplay.black_wins,
        draws: selfplay.draws,
        avg_plies: if games == 0 {
            0.0
        } else {
            selfplay.plies_total as f32 / games as f32
        },
        selfplay_seconds,
        target_entropy: before.target_entropy,
        initial_value_ce: before.value_ce,
        initial_value_mse: before.value_mse,
        initial_policy_ce: before.policy_ce,
        initial_policy_kl: before.policy_ce - before.target_entropy,
        final_value_ce: after.value_ce,
        final_value_mse: after.value_mse,
        final_policy_ce: after.policy_ce,
        final_policy_kl: after.policy_ce - after.target_entropy,
        train_loss: stats.loss,
        train_value_loss: stats.value_loss,
    }
}

pub fn benchmark_fixed_policy_fit(
    model: &mut AzNnue,
    samples: &[AzTrainingSample],
    holdout_samples: &[AzTrainingSample],
    epochs: usize,
    batch_size: usize,
    lr: f32,
    seed: u64,
    early_stop_patience: usize,
    min_delta: f32,
    loss_weights: AzTrainLossWeights,
) -> AzPolicyFitBenchmark {
    benchmark_fixed_policy_fit_with_trace(
        model,
        samples,
        holdout_samples,
        epochs,
        batch_size,
        lr,
        seed,
        early_stop_patience,
        min_delta,
        loss_weights,
        0,
    )
    .0
}

pub fn benchmark_fixed_policy_fit_with_trace(
    model: &mut AzNnue,
    samples: &[AzTrainingSample],
    holdout_samples: &[AzTrainingSample],
    epochs: usize,
    batch_size: usize,
    lr: f32,
    seed: u64,
    early_stop_patience: usize,
    min_delta: f32,
    loss_weights: AzTrainLossWeights,
    trace_interval: usize,
) -> (AzPolicyFitBenchmark, Vec<AzPolicyFitEpochReport>) {
    let before = policy_fit_eval(model, samples);
    let holdout_before = policy_fit_eval(model, holdout_samples);
    let mut rng = SplitMix64::new(seed);
    let mut trace = Vec::new();
    if trace_interval > 0 {
        trace.push(policy_fit_epoch_report(0, before, holdout_before));
    }

    let mut last_stats = AzTrainStats::default();
    let mut after = before;
    let mut holdout_after = holdout_before;
    let mut best_loss = loss_weights.policy.max(0.0) * before.policy_ce
        + loss_weights.value.max(0.0) * before.value_ce;
    let mut stale_epochs = 0usize;
    let mut epochs_completed = 0usize;
    for epoch in 1..=epochs {
        last_stats =
            train_samples_weighted(model, samples, 1, lr, batch_size, &mut rng, loss_weights);
        epochs_completed = epoch;
        after = policy_fit_eval(model, samples);
        let should_trace = trace_interval > 0 && (epoch % trace_interval == 0 || epoch == epochs);
        if should_trace || !holdout_samples.is_empty() {
            holdout_after = policy_fit_eval(model, holdout_samples);
        }
        if should_trace {
            trace.push(policy_fit_epoch_report(epoch, after, holdout_after));
        }
        let current_loss = loss_weights.policy.max(0.0) * after.policy_ce
            + loss_weights.value.max(0.0) * after.value_ce;
        if best_loss - current_loss > min_delta.max(0.0) {
            best_loss = current_loss;
            stale_epochs = 0;
        } else {
            stale_epochs += 1;
            if early_stop_patience > 0 && stale_epochs >= early_stop_patience {
                if trace_interval > 0 && trace.last().is_none_or(|entry| entry.epoch != epoch) {
                    trace.push(policy_fit_epoch_report(epoch, after, holdout_after));
                }
                break;
            }
        }
    }
    if trace_interval == 0
        || trace
            .last()
            .is_none_or(|entry| entry.epoch != epochs_completed)
    {
        holdout_after = policy_fit_eval(model, holdout_samples);
    }
    let summary = AzPolicyFitBenchmark {
        samples: samples.len() + holdout_samples.len(),
        train_samples: samples.len(),
        holdout_samples: holdout_samples.len(),
        epochs_completed,
        target_entropy: before.target_entropy,
        initial_value_ce: before.value_ce,
        initial_value_mse: before.value_mse,
        initial_policy_ce: before.policy_ce,
        initial_policy_kl: before.policy_ce - before.target_entropy,
        final_value_ce: after.value_ce,
        final_value_mse: after.value_mse,
        final_policy_ce: after.policy_ce,
        final_policy_kl: after.policy_ce - after.target_entropy,
        holdout_target_entropy: holdout_before.target_entropy,
        holdout_initial_value_ce: holdout_before.value_ce,
        holdout_initial_value_mse: holdout_before.value_mse,
        holdout_initial_policy_ce: holdout_before.policy_ce,
        holdout_initial_policy_kl: holdout_before.policy_ce - holdout_before.target_entropy,
        holdout_final_value_ce: holdout_after.value_ce,
        holdout_final_value_mse: holdout_after.value_mse,
        holdout_final_policy_ce: holdout_after.policy_ce,
        holdout_final_policy_kl: holdout_after.policy_ce - holdout_after.target_entropy,
        train_loss: last_stats.loss,
        train_value_loss: last_stats.value_loss,
    };
    (summary, trace)
}

fn fit_samples_with_early_stop(
    model: &mut AzNnue,
    samples: &[AzTrainingSample],
    epochs: usize,
    lr: f32,
    batch_size: usize,
    rng: &mut SplitMix64,
    early_stop_patience: usize,
    min_delta: f32,
    loss_weights: AzTrainLossWeights,
) -> (AzTrainStats, PolicyFitEval, usize) {
    let mut last_stats = AzTrainStats::default();
    let mut last_eval = policy_fit_eval(model, samples);
    let mut best_loss = loss_weights.policy.max(0.0) * last_eval.policy_ce
        + loss_weights.value.max(0.0) * last_eval.value_ce;
    let mut stale_epochs = 0usize;
    let mut completed = 0usize;
    for _ in 0..epochs {
        last_stats = train_samples_weighted(model, samples, 1, lr, batch_size, rng, loss_weights);
        completed += 1;
        last_eval = policy_fit_eval(model, samples);
        let current_loss = loss_weights.policy.max(0.0) * last_eval.policy_ce
            + loss_weights.value.max(0.0) * last_eval.value_ce;
        if best_loss - current_loss > min_delta.max(0.0) {
            best_loss = current_loss;
            stale_epochs = 0;
        } else {
            stale_epochs += 1;
            if early_stop_patience > 0 && stale_epochs >= early_stop_patience {
                break;
            }
        }
    }
    (last_stats, last_eval, completed)
}

fn policy_fit_epoch_report(
    epoch: usize,
    train: PolicyFitEval,
    holdout: PolicyFitEval,
) -> AzPolicyFitEpochReport {
    AzPolicyFitEpochReport {
        epoch,
        train_target_entropy: train.target_entropy,
        train_value_ce: train.value_ce,
        train_value_mse: train.value_mse,
        train_policy_ce: train.policy_ce,
        train_policy_kl: train.policy_ce - train.target_entropy,
        holdout_target_entropy: holdout.target_entropy,
        holdout_value_ce: holdout.value_ce,
        holdout_value_mse: holdout.value_mse,
        holdout_policy_ce: holdout.policy_ce,
        holdout_policy_kl: holdout.policy_ce - holdout.target_entropy,
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct PolicyFitEval {
    target_entropy: f32,
    policy_ce: f32,
    value_ce: f32,
    value_mse: f32,
}

fn generate_policy_fit_samples(
    teacher: &AzNnue,
    sample_count: usize,
    seed: u64,
    max_random_plies: usize,
    target_temperature: f32,
) -> Vec<AzTrainingSample> {
    let mut rng = SplitMix64::new(seed);
    let mut samples = Vec::with_capacity(sample_count);
    let mut teacher_scratch = AzEvalScratch::new(teacher.arch);
    while samples.len() < sample_count {
        let mut position = Position::startpos();
        let mut history = Vec::new();
        let random_plies = (rng.next_u64() as usize) % (max_random_plies.max(1) + 1);
        for _ in 0..random_plies {
            let legal = position.legal_moves();
            if legal.is_empty() {
                break;
            }
            let mv = legal[(rng.next_u64() as usize) % legal.len()];
            alphazero::append_history(&mut history, &position, mv);
            position.make_move(mv);
            if !position.has_general(Color::Red) || !position.has_general(Color::Black) {
                break;
            }
        }
        if !position.has_general(Color::Red) || !position.has_general(Color::Black) {
            continue;
        }
        let moves = position.legal_moves();
        if moves.is_empty() {
            continue;
        }
        let value =
            teacher.evaluate_with_scratch(&position, &history, &moves, &mut teacher_scratch);
        let temperature = target_temperature.max(1e-3);
        let scaled_logits = teacher_scratch.logits[..moves.len()]
            .iter()
            .map(|&logit| logit / temperature)
            .collect::<Vec<_>>();
        let policy = softmax_values(&scaled_logits);
        let side = position.side_to_move();
        let move_indices = moves
            .iter()
            .copied()
            .map(|mv| dense_move_index(canonical_move(side, mv)))
            .collect();
        let policy_feature_cache = PolicyFeaturePositionCache::new_for_moves(&position, &moves);
        let policy_dynamic_features = moves
            .iter()
            .flat_map(|&mv| policy_dynamic_features_cached(&position, mv, &policy_feature_cache))
            .collect();
        samples.push(AzTrainingSample {
            features: extract_sparse_features_az_canonical(&position, &history),
            move_indices,
            policy_dynamic_features,
            policy,
            value: value.clamp(-1.0, 1.0),
            side_sign: if side == Color::Red { 1.0 } else { -1.0 },
            moves_left: 0.0,
        });
    }
    samples
}

fn policy_fit_eval(model: &AzNnue, samples: &[AzTrainingSample]) -> PolicyFitEval {
    if samples.is_empty() {
        return PolicyFitEval::default();
    }
    let mut scratch = AzEvalScratch::new(model.arch);
    let mut target_entropy = 0.0f32;
    let mut policy_ce = 0.0f32;
    let mut value_ce = 0.0f32;
    let mut value_mse = 0.0f32;
    for sample in samples {
        model.input_embedding_into(&sample.features, &mut scratch.hidden);
        model.auto_feature_adapter_into(&mut scratch.hidden, &mut scratch.auto_features);
        rms_norm_in_place(&mut scratch.hidden);
        let value_pred = model.value_from_hidden_into(
            &scratch.hidden,
            &sample.features,
            &mut scratch.value_head,
        );
        let value_target = sample.value.clamp(-1.0, 1.0);
        let error = value_pred - value_target;
        value_mse += error * error;
        value_ce += error * error;
        model.policy_dynamic_condition_into(&scratch.hidden, &mut scratch.policy_dynamic_condition);
        model.policy_action_context_into(&scratch.hidden, &mut scratch.policy_action_context);
        model.policy_pair_context_into(&scratch.hidden, &mut scratch.policy_pair_context);
        model.policy_move_context_into(&scratch.hidden, &mut scratch.policy_move_context);
        model.policy_relation_context_into(&scratch.hidden, &mut scratch.policy_relation_context);
        scratch.logits.resize(sample.move_indices.len(), 0.0);
        scratch.policy_from_scores.resize(BOARD_SIZE, 0.0);
        scratch.policy_to_scores.resize(BOARD_SIZE, 0.0);
        let mut from_used = [false; BOARD_SIZE];
        let mut to_used = [false; BOARD_SIZE];
        let mut from_squares = [0usize; BOARD_SIZE];
        let mut to_squares = [0usize; BOARD_SIZE];
        let mut from_count = 0usize;
        let mut to_count = 0usize;
        for &move_index in &sample.move_indices {
            let sparse = move_map().dense_to_sparse[move_index] as usize;
            let from = sparse / BOARD_SIZE;
            let to = sparse % BOARD_SIZE;
            if !from_used[from] {
                from_used[from] = true;
                from_squares[from_count] = from;
                from_count += 1;
            }
            if !to_used[to] {
                to_used[to] = true;
                to_squares[to_count] = to;
                to_count += 1;
            }
        }
        model.policy_square_scores_for_squares_into(
            &scratch.hidden,
            &from_squares[..from_count],
            &to_squares[..to_count],
            &mut scratch.policy_from_scores,
            &mut scratch.policy_to_scores,
        );
        for (index, &move_index) in sample.move_indices.iter().enumerate() {
            let dynamic_start = index * POLICY_DYNAMIC_FEATURE_SIZE;
            let dynamic_end = dynamic_start + POLICY_DYNAMIC_FEATURE_SIZE;
            let dynamic_features = sample
                .policy_dynamic_features
                .get(dynamic_start..dynamic_end)
                .unwrap_or(&[0.0; POLICY_DYNAMIC_FEATURE_SIZE]);
            scratch.logits[index] = model.policy_logit_from_hidden_index(
                &scratch.policy_dynamic_condition,
                &scratch.policy_action_context,
                &scratch.policy_pair_context,
                &scratch.policy_move_context,
                &scratch.policy_relation_context,
                &scratch.policy_from_scores,
                &scratch.policy_to_scores,
                move_index,
                dynamic_features,
            );
        }
        let log_probs = log_softmax_values(&scratch.logits);
        for (&target, &log_prob) in sample.policy.iter().zip(&log_probs) {
            let p = target.max(0.0);
            if p > 0.0 {
                target_entropy -= p * p.ln();
                policy_ce -= p * log_prob;
            }
        }
    }
    let denom = samples.len() as f32;
    PolicyFitEval {
        target_entropy: target_entropy / denom,
        policy_ce: policy_ce / denom,
        value_ce: value_ce / denom,
        value_mse: value_mse / denom,
    }
}

fn log_softmax_values(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum = logits
        .iter()
        .map(|&logit| (logit - max_logit).exp())
        .sum::<f32>()
        .max(f32::MIN_POSITIVE);
    let log_sum = max_logit + sum.ln();
    logits.iter().map(|&logit| logit - log_sum).collect()
}

fn softmax_values(logits: &[f32]) -> Vec<f32> {
    let log_probs = log_softmax_values(logits);
    log_probs.iter().map(|&value| value.exp()).collect()
}

fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if left.len() >= 64 && std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime detection above guarantees AVX2 support.
            return unsafe { dot_product_avx2(left, right) };
        }
    }
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

fn add_scaled_feature_row(
    hidden: &mut [f32],
    input_hidden: &[f32],
    hidden_size: usize,
    feature: usize,
    scale: f32,
) {
    let row = &input_hidden[feature * hidden_size..(feature + 1) * hidden_size];
    debug_assert_eq!(hidden.len(), row.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if hidden_size >= 64 && std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime detection above guarantees AVX2 support.
            unsafe {
                add_scaled_feature_row_avx2(hidden, row, scale);
            }
            return;
        }
    }
    for (left, &right) in hidden.iter_mut().zip(row) {
        *left += scale * right;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2(left: &[f32], right: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let chunks = left.len() / 8;
    let mut acc = _mm256_setzero_ps();
    for chunk in 0..chunks {
        let index = chunk * 8;
        unsafe {
            let l = _mm256_loadu_ps(left.as_ptr().add(index));
            let r = _mm256_loadu_ps(right.as_ptr().add(index));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(l, r));
        }
    }
    let mut lanes = [0.0f32; 8];
    unsafe {
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    }
    let mut sum = lanes.iter().sum::<f32>();
    for index in (chunks * 8)..left.len() {
        sum += left[index] * right[index];
    }
    sum
}

#[cfg(target_arch = "x86")]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2(left: &[f32], right: &[f32]) -> f32 {
    use std::arch::x86::*;
    let chunks = left.len() / 8;
    let mut acc = _mm256_setzero_ps();
    for chunk in 0..chunks {
        let index = chunk * 8;
        unsafe {
            let l = _mm256_loadu_ps(left.as_ptr().add(index));
            let r = _mm256_loadu_ps(right.as_ptr().add(index));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(l, r));
        }
    }
    let mut lanes = [0.0f32; 8];
    unsafe {
        _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    }
    let mut sum = lanes.iter().sum::<f32>();
    for index in (chunks * 8)..left.len() {
        sum += left[index] * right[index];
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_scaled_feature_row_avx2(hidden: &mut [f32], row: &[f32], scale: f32) {
    use std::arch::x86_64::*;
    let scale_scalar = scale;
    let scale = _mm256_set1_ps(scale_scalar);
    let chunks = hidden.len() / 8;
    for chunk in 0..chunks {
        let index = chunk * 8;
        unsafe {
            let left = _mm256_loadu_ps(hidden.as_ptr().add(index));
            let right = _mm256_loadu_ps(row.as_ptr().add(index));
            _mm256_storeu_ps(
                hidden.as_mut_ptr().add(index),
                _mm256_add_ps(left, _mm256_mul_ps(scale, right)),
            );
        }
    }
    for index in (chunks * 8)..hidden.len() {
        hidden[index] += row[index] * scale_scalar;
    }
}

#[cfg(target_arch = "x86")]
#[target_feature(enable = "avx2")]
unsafe fn add_scaled_feature_row_avx2(hidden: &mut [f32], row: &[f32], scale: f32) {
    use std::arch::x86::*;
    let scale_scalar = scale;
    let scale = _mm256_set1_ps(scale_scalar);
    let chunks = hidden.len() / 8;
    for chunk in 0..chunks {
        let index = chunk * 8;
        unsafe {
            let left = _mm256_loadu_ps(hidden.as_ptr().add(index));
            let right = _mm256_loadu_ps(row.as_ptr().add(index));
            _mm256_storeu_ps(
                hidden.as_mut_ptr().add(index),
                _mm256_add_ps(left, _mm256_mul_ps(scale, right)),
            );
        }
    }
    for index in (chunks * 8)..hidden.len() {
        hidden[index] += row[index] * scale_scalar;
    }
}

fn relu_in_place(values: &mut [f32]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if values.len() >= 64 && std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime detection above guarantees AVX2 support.
            unsafe {
                relu_in_place_avx2(values);
            }
            return;
        }
    }
    for value in values {
        *value = value.max(0.0);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn input_embedding_add_features_avx2(
    input_hidden: &[f32],
    hidden_size: usize,
    features: &[usize],
    hidden: &mut [f32],
) {
    use std::arch::x86_64::*;
    let chunks = hidden_size / 8;
    for &feature in features {
        let row = &input_hidden[feature * hidden_size..(feature + 1) * hidden_size];
        for chunk in 0..chunks {
            let index = chunk * 8;
            unsafe {
                let left = _mm256_loadu_ps(hidden.as_ptr().add(index));
                let right = _mm256_loadu_ps(row.as_ptr().add(index));
                _mm256_storeu_ps(hidden.as_mut_ptr().add(index), _mm256_add_ps(left, right));
            }
        }
        for index in (chunks * 8)..hidden_size {
            hidden[index] += row[index];
        }
    }
}

#[cfg(target_arch = "x86")]
#[target_feature(enable = "avx2")]
unsafe fn input_embedding_add_features_avx2(
    input_hidden: &[f32],
    hidden_size: usize,
    features: &[usize],
    hidden: &mut [f32],
) {
    use std::arch::x86::*;
    let chunks = hidden_size / 8;
    for &feature in features {
        let row = &input_hidden[feature * hidden_size..(feature + 1) * hidden_size];
        for chunk in 0..chunks {
            let index = chunk * 8;
            unsafe {
                let left = _mm256_loadu_ps(hidden.as_ptr().add(index));
                let right = _mm256_loadu_ps(row.as_ptr().add(index));
                _mm256_storeu_ps(hidden.as_mut_ptr().add(index), _mm256_add_ps(left, right));
            }
        }
        for index in (chunks * 8)..hidden_size {
            hidden[index] += row[index];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn relu_in_place_avx2(values: &mut [f32]) {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_ps();
    let chunks = values.len() / 8;
    for chunk in 0..chunks {
        let index = chunk * 8;
        unsafe {
            let value = _mm256_loadu_ps(values.as_ptr().add(index));
            _mm256_storeu_ps(values.as_mut_ptr().add(index), _mm256_max_ps(value, zero));
        }
    }
    for value in &mut values[(chunks * 8)..] {
        *value = value.max(0.0);
    }
}

#[cfg(target_arch = "x86")]
#[target_feature(enable = "avx2")]
unsafe fn relu_in_place_avx2(values: &mut [f32]) {
    use std::arch::x86::*;
    let zero = _mm256_setzero_ps();
    let chunks = values.len() / 8;
    for chunk in 0..chunks {
        let index = chunk * 8;
        unsafe {
            let value = _mm256_loadu_ps(values.as_ptr().add(index));
            _mm256_storeu_ps(values.as_mut_ptr().add(index), _mm256_max_ps(value, zero));
        }
    }
    for value in &mut values[(chunks * 8)..] {
        *value = value.max(0.0);
    }
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

fn build_policy_action_static_projection(policy_action_static_hidden: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; DENSE_MOVE_SPACE * POLICY_ACTION_SIZE];
    let move_features = policy_move_features();
    for move_index in 0..DENSE_MOVE_SPACE {
        let move_row = &move_features
            [move_index * POLICY_CONDITION_SIZE..(move_index + 1) * POLICY_CONDITION_SIZE];
        for feature in 0..POLICY_ACTION_SIZE {
            let static_row = &policy_action_static_hidden
                [feature * POLICY_CONDITION_SIZE..(feature + 1) * POLICY_CONDITION_SIZE];
            out[move_index * POLICY_ACTION_SIZE + feature] = dot_product(move_row, static_row);
        }
    }
    out
}

fn build_policy_relation_pair_embedding(
    relation_from_embedding: &[f32],
    relation_to_embedding: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0; DENSE_MOVE_SPACE * POLICY_RELATION_SIZE];
    for (move_index, &sparse) in move_map().dense_to_sparse.iter().enumerate() {
        let from = sparse as usize / BOARD_SIZE;
        let to = sparse as usize % BOARD_SIZE;
        let from_row = &relation_from_embedding
            [from * POLICY_RELATION_SIZE..(from + 1) * POLICY_RELATION_SIZE];
        let to_row =
            &relation_to_embedding[to * POLICY_RELATION_SIZE..(to + 1) * POLICY_RELATION_SIZE];
        let out_row =
            &mut out[move_index * POLICY_RELATION_SIZE..(move_index + 1) * POLICY_RELATION_SIZE];
        for feature in 0..POLICY_RELATION_SIZE {
            out_row[feature] = from_row[feature] + to_row[feature];
        }
    }
    out
}

#[cfg_attr(not(feature = "gpu-train"), allow(dead_code))]
pub(super) fn policy_move_from_features() -> &'static [f32] {
    use std::sync::OnceLock;
    static FEATURES: OnceLock<Vec<f32>> = OnceLock::new();
    FEATURES.get_or_init(|| {
        let mut features = vec![0.0; DENSE_MOVE_SPACE * BOARD_SIZE];
        for (move_index, &sparse) in move_map().dense_to_sparse.iter().enumerate() {
            let from = sparse as usize / BOARD_SIZE;
            features[move_index * BOARD_SIZE + from] = 1.0;
        }
        features
    })
}

#[cfg_attr(not(feature = "gpu-train"), allow(dead_code))]
pub(super) fn policy_move_to_features() -> &'static [f32] {
    use std::sync::OnceLock;
    static FEATURES: OnceLock<Vec<f32>> = OnceLock::new();
    FEATURES.get_or_init(|| {
        let mut features = vec![0.0; DENSE_MOVE_SPACE * BOARD_SIZE];
        for (move_index, &sparse) in move_map().dense_to_sparse.iter().enumerate() {
            let to = sparse as usize % BOARD_SIZE;
            features[move_index * BOARD_SIZE + to] = 1.0;
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

#[cfg(test)]
fn replay_pool_test_fixture() -> AzExperiencePool {
    let sample = AzTrainingSample {
        features: vec![1, 2, 3],
        move_indices: vec![0, 1],
        policy_dynamic_features: vec![0.0; 2 * POLICY_DYNAMIC_FEATURE_SIZE],
        policy: vec![0.6, 0.4],
        value: 0.1,
        side_sign: 1.0,
        moves_left: 0.0,
    };
    let mut pool = AzExperiencePool::new(100);
    pool.add_games(vec![vec![sample.clone()], vec![sample.clone(), sample]]);
    pool
}

#[cfg(test)]
mod tests {
    use super::play::assign_terminal_value_targets;
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
    fn random_initial_value_head_is_neutral() {
        let model = AzNnue::random_with_arch(AzNnueArch::with_hidden_size(512), 20260409);
        assert!(model.value_head_output.iter().all(|&weight| weight == 0.0));

        let position = Position::startpos();
        let moves = position.legal_moves();
        let value = model.evaluate_value(&position, &[], &moves);
        assert!(value.abs() < 1e-6, "initial startpos value={value}");
    }

    #[test]
    fn az_eval_accumulator_tracks_absolute_features_across_moves() {
        let model = AzNnue::random_with_arch(AzNnueArch::with_hidden_size(64), 7);
        let mut position = Position::startpos();
        let mut accumulator = AzEvalAccumulator::new(&model, &position);
        assert_accumulator_matches_full_features(&model, &position, &accumulator);

        let mut rng = SplitMix64::new(20260521);
        for _ in 0..60 {
            let moves = position.legal_moves();
            if moves.is_empty() {
                break;
            }
            let mv = moves[(rng.next_u64() as usize) % moves.len()];
            let before = position.clone();
            let moved = before.piece_at(mv.from as usize).unwrap();
            let captured = before.piece_at(mv.to as usize);
            position.make_move(mv);
            accumulator.apply_transition(&model, &before, &position, mv, moved, captured);
            assert_accumulator_matches_full_features(&model, &position, &accumulator);

            let legal = position.legal_moves();
            let mut full_scratch = AzEvalScratch::new(model.arch);
            let mut acc_scratch = AzEvalScratch::new(model.arch);
            let full = model.evaluate_with_scratch(&position, &[], &legal, &mut full_scratch);
            let via_acc = model.evaluate_accumulator_with_scratch(
                &position,
                &accumulator,
                &[],
                &legal,
                &mut acc_scratch,
            );
            assert!(
                (full - via_acc).abs() < 1e-5,
                "full={full} via_acc={via_acc} fen={}",
                position.to_fen()
            );
        }
    }

    fn assert_accumulator_matches_full_features(
        model: &AzNnue,
        position: &Position,
        accumulator: &AzEvalAccumulator,
    ) {
        let mut expected = model.hidden_bias.clone();
        for feature in extract_sparse_features_az_absolute_current(position) {
            add_scaled_feature_row(
                &mut expected,
                &model.input_hidden,
                model.hidden_size,
                feature,
                1.0,
            );
        }
        assert_eq!(expected.len(), accumulator.hidden_sum.len());
        for (index, (&left, &right)) in expected.iter().zip(&accumulator.hidden_sum).enumerate() {
            assert!(
                (left - right).abs() < 1e-4,
                "accumulator mismatch at {index}: expected={left} got={right} fen={}",
                position.to_fen()
            );
        }
    }

    #[test]
    fn terminal_value_targets_match_side_to_move_outcome() {
        let mut samples = vec![
            AzTrainingSample {
                features: Vec::new(),
                move_indices: Vec::new(),
                policy_dynamic_features: Vec::new(),
                policy: Vec::new(),
                value: -0.5,
                side_sign: 1.0,
                moves_left: 0.0,
            },
            AzTrainingSample {
                features: Vec::new(),
                move_indices: Vec::new(),
                policy_dynamic_features: Vec::new(),
                policy: Vec::new(),
                value: 0.5,
                side_sign: -1.0,
                moves_left: 0.0,
            },
        ];

        assign_terminal_value_targets(&mut samples, 1.0);

        assert!(samples[0].value > -0.5);
        assert!(samples[0].value < 1.0);
        assert!((samples[1].value + 1.0).abs() < 1e-6);
    }

    #[test]
    fn scalar_value_head_starts_neutral() {
        let model = AzNnue::random(16, 7);
        let mut scratch = AzEvalScratch::new(model.arch);
        let value = model.value_from_hidden_into(&scratch.hidden, &[], &mut scratch.value_head);

        assert!(value.abs() < 1e-6);
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

    #[cfg(feature = "gpu-train")]
    #[test]
    fn value_head_can_overfit_tiny_fixed_dataset() {
        let mut model = AzNnue::random(16, 7);
        model.hidden_bias.fill(0.1);
        model.hidden_bias.fill(0.1);

        let samples = vec![
            AzTrainingSample {
                features: vec![0],
                move_indices: Vec::new(),
                policy_dynamic_features: Vec::new(),
                policy: Vec::new(),
                value: 1.0,
                side_sign: 1.0,
                moves_left: 0.0,
            },
            AzTrainingSample {
                features: vec![1],
                move_indices: Vec::new(),
                policy_dynamic_features: Vec::new(),
                policy: Vec::new(),
                value: -1.0,
                side_sign: 1.0,
                moves_left: 0.0,
            },
            AzTrainingSample {
                features: vec![2],
                move_indices: Vec::new(),
                policy_dynamic_features: Vec::new(),
                policy: Vec::new(),
                value: 0.75,
                side_sign: 1.0,
                moves_left: 0.0,
            },
            AzTrainingSample {
                features: vec![3],
                move_indices: Vec::new(),
                policy_dynamic_features: Vec::new(),
                policy: Vec::new(),
                value: -0.75,
                side_sign: 1.0,
                moves_left: 0.0,
            },
        ];

        let mut rng = SplitMix64::new(17);
        let before = train_samples(&mut model, &samples, 1, 0.003, 4, &mut rng).value_loss;
        let after = train_samples(&mut model, &samples, 300, 0.003, 4, &mut rng).value_loss;

        assert!(after < before * 0.5, "before={before} after={after}");
        assert!(after < 0.35, "after={after}");
    }

    #[cfg(feature = "gpu-train")]
    #[test]
    fn batched_training_is_deterministic() {
        let samples = vec![
            AzTrainingSample {
                features: vec![0, 4, 8],
                move_indices: Vec::new(),
                policy_dynamic_features: Vec::new(),
                policy: Vec::new(),
                value: 1.0,
                side_sign: 1.0,
                moves_left: 0.0,
            },
            AzTrainingSample {
                features: vec![1, 5, 9],
                move_indices: Vec::new(),
                policy_dynamic_features: Vec::new(),
                policy: Vec::new(),
                value: -1.0,
                side_sign: 1.0,
                moves_left: 0.0,
            },
            AzTrainingSample {
                features: vec![2, 6, 10],
                move_indices: Vec::new(),
                policy_dynamic_features: Vec::new(),
                policy: Vec::new(),
                value: 0.5,
                side_sign: 1.0,
                moves_left: 0.0,
            },
            AzTrainingSample {
                features: vec![3, 7, 11],
                move_indices: Vec::new(),
                policy_dynamic_features: Vec::new(),
                policy: Vec::new(),
                value: -0.5,
                side_sign: 1.0,
                moves_left: 0.0,
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
                .value_head_output
                .iter()
                .zip(&repeated.value_head_output)
                .all(|(left, right)| (*left - *right).abs() < 1e-5)
        );
    }

    #[cfg(feature = "gpu-train")]
    #[test]
    fn value_only_training_updates_trunk_when_trunk_training_enabled() {
        let samples = vec![
            AzTrainingSample {
                features: vec![0, 4, 8],
                move_indices: Vec::new(),
                policy_dynamic_features: Vec::new(),
                policy: Vec::new(),
                value: 1.0,
                side_sign: 1.0,
                moves_left: 0.0,
            },
            AzTrainingSample {
                features: vec![1, 5, 9],
                move_indices: Vec::new(),
                policy_dynamic_features: Vec::new(),
                policy: Vec::new(),
                value: -1.0,
                side_sign: 1.0,
                moves_left: 0.0,
            },
            AzTrainingSample {
                features: vec![2, 6, 10],
                move_indices: Vec::new(),
                policy_dynamic_features: Vec::new(),
                policy: Vec::new(),
                value: 0.75,
                side_sign: 1.0,
                moves_left: 0.0,
            },
            AzTrainingSample {
                features: vec![3, 7, 11],
                move_indices: Vec::new(),
                policy_dynamic_features: Vec::new(),
                policy: Vec::new(),
                value: -0.75,
                side_sign: 1.0,
                moves_left: 0.0,
            },
        ];
        let mut model = AzNnue::random(8, 31);
        model.hidden_bias.fill(0.1);
        let before_input = model.input_hidden.clone();
        let before_bias = model.hidden_bias.clone();
        let before_quadratic = model.input_quadratic_scale.clone();

        let mut rng = SplitMix64::new(32);
        let weights = AzTrainLossWeights {
            value: 1.0,
            policy: 0.0,
            train_trunk: true,
            train_value_head: true,
            train_policy_head: false,
        };
        train_samples_weighted(&mut model, &samples, 20, 0.01, 4, &mut rng, weights);

        let input_changed = before_input
            .iter()
            .zip(&model.input_hidden)
            .any(|(left, right)| (*left - *right).abs() > 1e-7);
        let bias_changed = before_bias
            .iter()
            .zip(&model.hidden_bias)
            .any(|(left, right)| (*left - *right).abs() > 1e-7);
        let quadratic_changed = before_quadratic
            .iter()
            .zip(&model.input_quadratic_scale)
            .any(|(left, right)| (*left - *right).abs() > 1e-7);
        assert!(
            input_changed || bias_changed || quadratic_changed,
            "value-only training should update trunk when train_trunk=true"
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
        assert_eq!(model.input_piece_hidden, loaded.input_piece_hidden);
        assert_eq!(model.input_rank_hidden, loaded.input_rank_hidden);
        assert_eq!(model.input_file_hidden, loaded.input_file_hidden);
        assert_eq!(
            model.input_king_piece_hidden,
            loaded.input_king_piece_hidden
        );
        assert_eq!(model.input_quadratic_scale, loaded.input_quadratic_scale);
        assert_eq!(model.piece_attention_query, loaded.piece_attention_query);
        assert_eq!(model.piece_attention_value, loaded.piece_attention_value);
        assert_eq!(model.piece_attention_output, loaded.piece_attention_output);
        assert_eq!(model.square_piece_token, loaded.square_piece_token);
        assert_eq!(model.square_position_token, loaded.square_position_token);
        assert_eq!(model.line_mixer_down, loaded.line_mixer_down);
        assert_eq!(model.line_mixer_bias, loaded.line_mixer_bias);
        assert_eq!(model.line_mixer_up, loaded.line_mixer_up);
        assert_eq!(model.line_context_bias, loaded.line_context_bias);
        assert_eq!(model.trunk_residual_hidden, loaded.trunk_residual_hidden);
        assert_eq!(model.trunk_residual_bias, loaded.trunk_residual_bias);
        assert_eq!(model.auto_feature_hidden, loaded.auto_feature_hidden);
        assert_eq!(model.auto_feature_bias, loaded.auto_feature_bias);
        assert_eq!(model.auto_feature_output, loaded.auto_feature_output);
        assert_eq!(model.value_head_hidden, loaded.value_head_hidden);
        assert_eq!(model.value_head_bias, loaded.value_head_bias);
        assert_eq!(model.value_head_output, loaded.value_head_output);
        assert_eq!(model.moves_left_output, loaded.moves_left_output);
        assert_eq!(model.moves_left_bias, loaded.moves_left_bias);
        assert_eq!(model.policy_move_bias, loaded.policy_move_bias);
        assert_eq!(
            model.policy_pair_context_hidden,
            loaded.policy_pair_context_hidden
        );
        assert_eq!(model.policy_pair_embedding, loaded.policy_pair_embedding);
        assert_eq!(
            model.policy_move_context_hidden,
            loaded.policy_move_context_hidden
        );
        assert_eq!(model.policy_move_embedding, loaded.policy_move_embedding);
        assert_eq!(
            model.policy_relation_context_hidden,
            loaded.policy_relation_context_hidden
        );
        assert_eq!(
            model.policy_relation_context_bias,
            loaded.policy_relation_context_bias
        );
        assert_eq!(
            model.policy_relation_from_embedding,
            loaded.policy_relation_from_embedding
        );
        assert_eq!(
            model.policy_relation_to_embedding,
            loaded.policy_relation_to_embedding
        );
        assert_eq!(model.policy_relation_output, loaded.policy_relation_output);
    }

    #[test]
    fn replay_pool_lz4_snapshot_roundtrip() {
        let path = std::env::temp_dir().join("chineseai_test_replay_roundtrip.replay.lz4");
        let _ = fs::remove_file(&path);
        let pool = super::replay_pool_test_fixture();
        pool.save_snapshot_lz4(&path).unwrap();
        let loaded = AzExperiencePool::load_snapshot_lz4(&path, 100).unwrap();
        let _ = fs::remove_file(&path);
        assert_eq!(loaded.sample_count(), pool.sample_count());
        assert_eq!(loaded.capacity(), pool.capacity());
    }
}
