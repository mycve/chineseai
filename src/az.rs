#[cfg(test)]
use crate::board_transform::{HistoryMove, canonical_square};
#[cfg(test)]
use crate::xiangqi::{Move, Position};
#[cfg(test)]
use std::fs;

mod alphazero;
mod mctx;
mod model;
mod model_binary;
mod model_config;
mod model_ops;
mod play;
mod replay;
mod train;
mod train_gpu;

pub use alphazero::{
    AzCandidate, AzSearchAlgorithm, AzSearchLimits, AzSearchResult, alphazero_search,
    alphazero_search_with_history_and_rules,
};
pub use mctx::AzGumbelConfig;
use model::{
    AZ_MODEL_BINARY_HEADER_LEN, AZ_MODEL_BINARY_VERSION, AzEvalScratch, BOARD_CHANNELS,
    BOARD_HISTORY_FRAMES, BOARD_HISTORY_SIZE, BOARD_INPUT_KERNEL_AREA, BOARD_PLANES_SIZE,
    CNN_CHANNELS, CNN_KERNEL_AREA, CNN_POOL_BLOCKS, CNN_POOLED_SIZE, DENSE_MOVE_SPACE,
    MOBILE_BLOCK_BIAS_SIZE, MOBILE_BLOCK_WEIGHT_SIZE, PIECE_BOARD_CHANNELS, POLICY_CONDITION_SIZE,
    RESIDUAL_BLOCKS, VALUE_HEAD_CHANNELS, VALUE_HEAD_FEATURES, VALUE_HEAD_LEAK,
    VALUE_HEAD_MAP_SIZE, VALUE_HIDDEN_SIZE, VALUE_LOGIT_SCALE, VALUE_LOGITS, VALUE_SCALE_CP,
    dense_move_index, extract_board_planes, policy_move_features, policy_move_from_select,
    policy_move_to_select,
};
pub use model::{AZ_MODEL_BINARY_MAGIC, AzModel, SplitMix64};
#[cfg(test)]
use model::{canonical_piece_plane, move_map};
pub use model_config::AzModelConfig;
pub use play::{
    AzArenaConfig, AzArenaReport, AzSelfplayData, AzTerminalStats, generate_selfplay_data,
    play_arena_games_from_positions,
};
pub use replay::AzExperiencePool;
pub use train::{global_training_step_sample_count, train_samples};

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
    pub temperature_early_entropy: f32,
    pub temperature_mid_entropy: f32,
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

#[derive(Clone, Debug)]
pub struct AzTrainingSample {
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

#[cfg(test)]
fn replay_pool_test_fixture() -> AzExperiencePool {
    let sample = AzTrainingSample {
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
                board: vec![0; BOARD_HISTORY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 0.0,
                side_sign: 1.0,
            },
            AzTrainingSample {
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
                board: vec![0; BOARD_HISTORY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 0.2,
                side_sign: 1.0,
            },
            AzTrainingSample {
                board: vec![0; BOARD_HISTORY_SIZE],
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: -0.4,
                side_sign: -1.0,
            },
            AzTrainingSample {
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
    fn arena_report_elo_tracks_score_rate_direction() {
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
        assert!(stronger.elo() > 0.0);
        assert!(weaker.score_rate() < 0.5);
        assert!(weaker.elo() < 0.0);
    }

    #[test]
    fn value_head_can_overfit_tiny_fixed_dataset() {
        let mut model = AzModel::random(16, 7);
        let board_with = |sq: usize, plane: u8| {
            let mut board = vec![0; BOARD_HISTORY_SIZE];
            board[sq] = plane;
            board
        };

        let samples = vec![
            AzTrainingSample {
                board: board_with(0, 1),
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 1.0,
                side_sign: 1.0,
            },
            AzTrainingSample {
                board: board_with(10, 2),
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: -1.0,
                side_sign: 1.0,
            },
            AzTrainingSample {
                board: board_with(40, 3),
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 0.75,
                side_sign: 1.0,
            },
            AzTrainingSample {
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
        let board_with = |sq: usize, plane: u8| {
            let mut board = vec![0; BOARD_HISTORY_SIZE];
            board[sq] = plane;
            board
        };
        let samples = vec![
            AzTrainingSample {
                board: board_with(0, 1),
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 1.0,
                side_sign: 1.0,
            },
            AzTrainingSample {
                board: board_with(10, 2),
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: -1.0,
                side_sign: 1.0,
            },
            AzTrainingSample {
                board: board_with(40, 3),
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: 0.5,
                side_sign: 1.0,
            },
            AzTrainingSample {
                board: board_with(80, 4),
                move_indices: Vec::new(),
                policy: Vec::new(),
                value: -0.5,
                side_sign: 1.0,
            },
        ];
        let mut single = AzModel::random(16, 23);
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
    fn az_model_binary_roundtrip_matches_weights() {
        let model = AzModel::random(16, 42);
        let path = std::env::temp_dir().join("chineseai_test_az_model_roundtrip.azm");
        let _ = fs::remove_file(&path);
        model.save(&path).unwrap();
        let loaded = AzModel::load(&path).unwrap();
        let _ = fs::remove_file(&path);
        assert_eq!(model.hidden_size, loaded.hidden_size);
        assert_eq!(model.board_hidden, loaded.board_hidden);
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
