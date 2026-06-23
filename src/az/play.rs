use std::sync::Arc;

use rayon::prelude::*;

use crate::nnue::{
    HistoryMove, canonical_move, extract_sparse_features_az_canonical, mirror_file_move,
    mirror_sparse_features_az_absolute_file,
};
use crate::xiangqi::{Color, Move, Position, RuleDrawReason, RuleOutcome};

use super::alphazero::append_history;
use super::{
    AzCandidate, AzLoopConfig, AzNnue, AzSampleMeta, AzSearchLimits, AzTrainingSample,
    GumbelSearchConfig, SplitMix64, alphazero_search_with_history_and_rules, dense_move_index,
    gumbel_search_with_history_and_rules, scalar_value_to_wdl_target,
};

#[derive(Clone, Copy, Debug)]
struct DeblunderEvent {
    sample_index: usize,
    boundary_red: f32,
}

#[derive(Clone, Copy, Debug)]
struct MoveSearchMeta {
    sample: AzSampleMeta,
    deblunder_boundary_q: Option<f32>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AzTerminalStats {
    pub no_legal_moves: usize,
    pub red_general_missing: usize,
    pub black_general_missing: usize,
    pub rule_draw: usize,
    pub rule_draw_halfmove120: usize,
    pub rule_draw_repetition: usize,
    pub rule_draw_mutual_long_check: usize,
    pub rule_draw_mutual_long_chase: usize,
    pub rule_win_red: usize,
    pub rule_win_black: usize,
    pub resign_red: usize,
    pub resign_black: usize,
    pub max_plies: usize,
}

impl AzTerminalStats {
    pub fn add_assign(&mut self, other: &Self) {
        self.no_legal_moves += other.no_legal_moves;
        self.red_general_missing += other.red_general_missing;
        self.black_general_missing += other.black_general_missing;
        self.rule_draw += other.rule_draw;
        self.rule_draw_halfmove120 += other.rule_draw_halfmove120;
        self.rule_draw_repetition += other.rule_draw_repetition;
        self.rule_draw_mutual_long_check += other.rule_draw_mutual_long_check;
        self.rule_draw_mutual_long_chase += other.rule_draw_mutual_long_chase;
        self.rule_win_red += other.rule_win_red;
        self.rule_win_black += other.rule_win_black;
        self.resign_red += other.resign_red;
        self.resign_black += other.resign_black;
        self.max_plies += other.max_plies;
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AzArenaReport {
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
    pub wins_as_red: usize,
    pub losses_as_red: usize,
    pub wins_as_black: usize,
    pub losses_as_black: usize,
}

impl AzArenaReport {
    pub fn add_assign(&mut self, other: &Self) {
        self.wins += other.wins;
        self.losses += other.losses;
        self.draws += other.draws;
        self.wins_as_red += other.wins_as_red;
        self.losses_as_red += other.losses_as_red;
        self.wins_as_black += other.wins_as_black;
        self.losses_as_black += other.losses_as_black;
    }

    pub fn total_games(&self) -> usize {
        self.wins + self.losses + self.draws
    }

    pub fn score(&self) -> f32 {
        self.wins as f32 + 0.5 * self.draws as f32
    }

    pub fn score_rate(&self) -> f32 {
        self.score() / self.total_games().max(1) as f32
    }

    pub fn score_rate_standard_error(&self) -> f32 {
        let games = self.total_games();
        if games <= 1 {
            return 0.5;
        }
        let mean = self.score_rate();
        let mean_square = (self.wins as f32 + 0.25 * self.draws as f32) / games as f32;
        let variance = (mean_square - mean * mean).max(0.0);
        (variance / games as f32).sqrt()
    }

    pub fn score_rate_lower_bound(&self, z: f32) -> f32 {
        self.score_rate() - z.max(0.0) * self.score_rate_standard_error()
    }

    pub fn promotes_with_lower_bound(&self, threshold: f32, z: f32) -> bool {
        self.score_rate_lower_bound(z) >= threshold.clamp(0.0, 1.0)
    }

    pub fn anchored_elo(&self, ref_elo: f32) -> f32 {
        ref_elo + self.elo_diff_vs_even()
    }

    pub fn elo_diff_vs_even(&self) -> f32 {
        let total = self.total_games();
        if total == 0 {
            return 0.0;
        }
        let score = self.score() / total as f32;
        if score <= 0.0 {
            -400.0
        } else if score >= 1.0 {
            400.0
        } else {
            400.0 * (score / (1.0 - score)).log10()
        }
    }
}

#[derive(Clone, Default)]
pub struct AzSelfplayData {
    pub samples: Vec<AzTrainingSample>,
    pub games: Vec<Vec<AzTrainingSample>>,
    pub red_wins: usize,
    pub black_wins: usize,
    pub draws: usize,
    pub plies_total: usize,
    pub entropy_all_sum: f32,
    pub entropy_all_count: usize,
    pub entropy_opening_sum: f32,
    pub entropy_opening_count: usize,
    pub entropy_mid_sum: f32,
    pub entropy_mid_count: usize,
    pub raw_prior_top1_sum: f32,
    pub raw_prior_top2_sum: f32,
    pub policy_top1_sum: f32,
    pub policy_top2_sum: f32,
    pub q_gap_sum: f32,
    pub q_top1_abs_sum: f32,
    pub visited_actions_sum: usize,
    pub shape_count: usize,
    pub opening_raw_prior_top1_sum: f32,
    pub opening_raw_prior_top2_sum: f32,
    pub opening_policy_top1_sum: f32,
    pub opening_policy_top2_sum: f32,
    pub opening_q_gap_sum: f32,
    pub opening_q_top1_abs_sum: f32,
    pub opening_visited_actions_sum: usize,
    pub opening_shape_count: usize,
    pub sampled_moves: usize,
    pub sampled_best_moves: usize,
    pub deblundered_moves: usize,
    pub best_played_q_gap_sum: f32,
    pub played_top_visit_ratio_sum: f32,
    pub best_q_sum: f32,
    pub played_q_sum: f32,
    pub terminal: AzTerminalStats,
}

impl AzSelfplayData {
    pub fn add_assign(&mut self, other: &Self) {
        self.samples.extend(other.samples.iter().cloned());
        self.games.extend(other.games.iter().cloned());
        self.red_wins += other.red_wins;
        self.black_wins += other.black_wins;
        self.draws += other.draws;
        self.plies_total += other.plies_total;
        self.entropy_all_sum += other.entropy_all_sum;
        self.entropy_all_count += other.entropy_all_count;
        self.entropy_opening_sum += other.entropy_opening_sum;
        self.entropy_opening_count += other.entropy_opening_count;
        self.entropy_mid_sum += other.entropy_mid_sum;
        self.entropy_mid_count += other.entropy_mid_count;
        self.raw_prior_top1_sum += other.raw_prior_top1_sum;
        self.raw_prior_top2_sum += other.raw_prior_top2_sum;
        self.policy_top1_sum += other.policy_top1_sum;
        self.policy_top2_sum += other.policy_top2_sum;
        self.q_gap_sum += other.q_gap_sum;
        self.q_top1_abs_sum += other.q_top1_abs_sum;
        self.visited_actions_sum += other.visited_actions_sum;
        self.shape_count += other.shape_count;
        self.opening_raw_prior_top1_sum += other.opening_raw_prior_top1_sum;
        self.opening_raw_prior_top2_sum += other.opening_raw_prior_top2_sum;
        self.opening_policy_top1_sum += other.opening_policy_top1_sum;
        self.opening_policy_top2_sum += other.opening_policy_top2_sum;
        self.opening_q_gap_sum += other.opening_q_gap_sum;
        self.opening_q_top1_abs_sum += other.opening_q_top1_abs_sum;
        self.opening_visited_actions_sum += other.opening_visited_actions_sum;
        self.opening_shape_count += other.opening_shape_count;
        self.sampled_moves += other.sampled_moves;
        self.sampled_best_moves += other.sampled_best_moves;
        self.deblundered_moves += other.deblundered_moves;
        self.best_played_q_gap_sum += other.best_played_q_gap_sum;
        self.played_top_visit_ratio_sum += other.played_top_visit_ratio_sum;
        self.best_q_sum += other.best_q_sum;
        self.played_q_sum += other.played_q_sum;
        self.terminal.add_assign(&other.terminal);
    }
}

pub fn generate_selfplay_data(model: &AzNnue, config: &AzLoopConfig) -> AzSelfplayData {
    let workers = config.workers.max(1).min(config.games.max(1));
    if workers == 1 || config.games <= 1 {
        return generate_selfplay_chunk(model, config);
    }

    let shared_model = Arc::new(model.clone());
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .expect("failed to build selfplay rayon pool");
    let chunks = pool.install(|| {
        (0..workers)
            .into_par_iter()
            .map(|worker| {
                let games = config.games / workers + usize::from(worker < config.games % workers);
                let mut worker_config = config.clone();
                worker_config.games = games;
                worker_config.workers = 1;
                worker_config.seed ^= (worker as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                generate_selfplay_chunk(&shared_model, &worker_config)
            })
            .collect::<Vec<_>>()
    });
    let mut merged = AzSelfplayData::default();
    for chunk in chunks {
        merged.samples.extend(chunk.samples);
        merged.games.extend(chunk.games);
        merged.red_wins += chunk.red_wins;
        merged.black_wins += chunk.black_wins;
        merged.draws += chunk.draws;
        merged.plies_total += chunk.plies_total;
        merged.entropy_all_sum += chunk.entropy_all_sum;
        merged.entropy_all_count += chunk.entropy_all_count;
        merged.entropy_opening_sum += chunk.entropy_opening_sum;
        merged.entropy_opening_count += chunk.entropy_opening_count;
        merged.entropy_mid_sum += chunk.entropy_mid_sum;
        merged.entropy_mid_count += chunk.entropy_mid_count;
        merged.raw_prior_top1_sum += chunk.raw_prior_top1_sum;
        merged.raw_prior_top2_sum += chunk.raw_prior_top2_sum;
        merged.policy_top1_sum += chunk.policy_top1_sum;
        merged.policy_top2_sum += chunk.policy_top2_sum;
        merged.q_gap_sum += chunk.q_gap_sum;
        merged.q_top1_abs_sum += chunk.q_top1_abs_sum;
        merged.visited_actions_sum += chunk.visited_actions_sum;
        merged.shape_count += chunk.shape_count;
        merged.opening_raw_prior_top1_sum += chunk.opening_raw_prior_top1_sum;
        merged.opening_raw_prior_top2_sum += chunk.opening_raw_prior_top2_sum;
        merged.opening_policy_top1_sum += chunk.opening_policy_top1_sum;
        merged.opening_policy_top2_sum += chunk.opening_policy_top2_sum;
        merged.opening_q_gap_sum += chunk.opening_q_gap_sum;
        merged.opening_q_top1_abs_sum += chunk.opening_q_top1_abs_sum;
        merged.opening_visited_actions_sum += chunk.opening_visited_actions_sum;
        merged.opening_shape_count += chunk.opening_shape_count;
        merged.sampled_moves += chunk.sampled_moves;
        merged.sampled_best_moves += chunk.sampled_best_moves;
        merged.deblundered_moves += chunk.deblundered_moves;
        merged.best_played_q_gap_sum += chunk.best_played_q_gap_sum;
        merged.played_top_visit_ratio_sum += chunk.played_top_visit_ratio_sum;
        merged.best_q_sum += chunk.best_q_sum;
        merged.played_q_sum += chunk.played_q_sum;
        merged.terminal.add_assign(&chunk.terminal);
    }
    merged
}

fn generate_selfplay_chunk(model: &AzNnue, config: &AzLoopConfig) -> AzSelfplayData {
    let mut rng = SplitMix64::new(config.seed);
    let mut samples = Vec::new();
    let mut red_wins = 0usize;
    let mut black_wins = 0usize;
    let mut draws = 0usize;
    let mut plies_total = 0usize;
    let mut games = Vec::with_capacity(config.games);
    let mut entropy_all_sum = 0.0f32;
    let mut entropy_all_count = 0usize;
    let mut entropy_opening_sum = 0.0f32;
    let mut entropy_opening_count = 0usize;
    let mut entropy_mid_sum = 0.0f32;
    let mut entropy_mid_count = 0usize;
    let mut raw_prior_top1_sum = 0.0f32;
    let mut raw_prior_top2_sum = 0.0f32;
    let mut policy_top1_sum = 0.0f32;
    let mut policy_top2_sum = 0.0f32;
    let mut q_gap_sum = 0.0f32;
    let mut q_top1_abs_sum = 0.0f32;
    let mut visited_actions_sum = 0usize;
    let mut shape_count = 0usize;
    let mut opening_raw_prior_top1_sum = 0.0f32;
    let mut opening_raw_prior_top2_sum = 0.0f32;
    let mut opening_policy_top1_sum = 0.0f32;
    let mut opening_policy_top2_sum = 0.0f32;
    let mut opening_q_gap_sum = 0.0f32;
    let mut opening_q_top1_abs_sum = 0.0f32;
    let mut opening_visited_actions_sum = 0usize;
    let mut opening_shape_count = 0usize;
    let mut sampled_moves = 0usize;
    let mut sampled_best_moves = 0usize;
    let mut deblundered_moves = 0usize;
    let mut best_played_q_gap_sum = 0.0f32;
    let mut played_top_visit_ratio_sum = 0.0f32;
    let mut best_q_sum = 0.0f32;
    let mut played_q_sum = 0.0f32;
    let mut terminal = AzTerminalStats::default();

    for game_index in 0..config.games {
        let mut position = if config.opening_positions.is_empty() {
            Position::startpos()
        } else {
            let index = (rng.next_u64() as usize) % config.opening_positions.len();
            config.opening_positions[index].clone()
        };
        let mut history = Vec::new();
        let mut rule_history = position.initial_rule_history();
        let mut game_samples = Vec::new();
        let mut deblunder_events = Vec::new();
        let mut result = None;
        let mut plies = 0usize;
        let allow_resign = rng.unit_f32() * 100.0 >= config.resign_playthrough;

        for ply in 0..config.max_plies {
            plies = ply + 1;
            let legal = position.legal_moves_with_rules(&rule_history);
            if legal.is_empty() {
                result = Some(if position.side_to_move() == Color::Red {
                    -1.0
                } else {
                    1.0
                });
                terminal.no_legal_moves += 1;
                break;
            }

            let search_seed = rng.next_u64() ^ ((game_index as u64) << 32) ^ ply as u64;
            let limits = AzSearchLimits {
                simulations: config.simulations,
                seed: search_seed,
                cpuct: config.cpuct,
                cpuct_at_root: config.cpuct_at_root,
                cpuct_base: config.cpuct_base,
                cpuct_factor: config.cpuct_factor,
                cpuct_base_at_root: config.cpuct_base_at_root,
                cpuct_factor_at_root: config.cpuct_factor_at_root,
                max_depth: 0,
                root_dirichlet_alpha: if config.search == "gumbel" {
                    0.0
                } else {
                    config.root_dirichlet_alpha
                },
                root_exploration_fraction: if config.search == "gumbel" {
                    0.0
                } else {
                    config.root_exploration_fraction
                },
                fpu_value: config.fpu_value,
                fpu_value_at_root: config.fpu_value_at_root,
                draw_score: config.draw_score,
                moves_left_max_effect: config.moves_left_max_effect,
                moves_left_slope: config.moves_left_slope,
                moves_left_threshold: config.moves_left_threshold,
                moves_left_constant_factor: config.moves_left_constant_factor,
                moves_left_scaled_factor: config.moves_left_scaled_factor,
                moves_left_quadratic_factor: config.moves_left_quadratic_factor,
                value_scale: 1.0,
            };
            let search = if config.search == "gumbel" {
                gumbel_search_with_history_and_rules(
                    &position,
                    &history,
                    Some(rule_history.clone()),
                    Some(legal),
                    model,
                    limits,
                    GumbelSearchConfig {
                        max_num_considered_actions: config.gumbel_actions,
                        gumbel_scale: config.gumbel_scale,
                        value_scale: config.gumbel_value_scale,
                        maxvisit_init: config.gumbel_maxvisit_init,
                    },
                )
            } else {
                alphazero_search_with_history_and_rules(
                    &position,
                    &history,
                    Some(rule_history.clone()),
                    Some(legal),
                    model,
                    limits,
                )
            };
            let entropy = policy_entropy(&search.candidates);
            let shape = policy_shape_stats(&search.candidates);
            raw_prior_top1_sum += shape.raw_prior_top1;
            raw_prior_top2_sum += shape.raw_prior_top2;
            policy_top1_sum += shape.policy_top1;
            policy_top2_sum += shape.policy_top2;
            q_gap_sum += shape.q_gap;
            q_top1_abs_sum += shape.q_top1_abs;
            visited_actions_sum += shape.visited_actions;
            shape_count += 1;
            entropy_all_sum += entropy;
            entropy_all_count += 1;
            if ply < temperature_opening_plies(config) {
                entropy_opening_sum += entropy;
                entropy_opening_count += 1;
                opening_raw_prior_top1_sum += shape.raw_prior_top1;
                opening_raw_prior_top2_sum += shape.raw_prior_top2;
                opening_policy_top1_sum += shape.policy_top1;
                opening_policy_top2_sum += shape.policy_top2;
                opening_q_gap_sum += shape.q_gap;
                opening_q_top1_abs_sum += shape.q_top1_abs;
                opening_visited_actions_sum += shape.visited_actions;
                opening_shape_count += 1;
            } else {
                entropy_mid_sum += entropy;
                entropy_mid_count += 1;
            }
            if allow_resign && should_resign(search.value_q, config) {
                let meta = root_search_meta(
                    &search.candidates,
                    search.value_q,
                    config.generation_update,
                    config.seed ^ game_index as u64,
                    ply,
                );
                game_samples.push(make_training_sample(
                    &position,
                    &history,
                    &search.candidates,
                    search.value_q,
                    if config.search == "gumbel" {
                        1.0
                    } else {
                        config.policy_softmax_temp
                    },
                    rng.unit_f32() < config.mirror_probability.clamp(0.0, 1.0),
                    meta,
                ));
                result = Some(if position.side_to_move() == Color::Red {
                    terminal.resign_red += 1;
                    -1.0
                } else {
                    terminal.resign_black += 1;
                    1.0
                });
                break;
            }
            let temperature = temperature_for_ply(config, ply);
            let mv_opt = if config.search == "gumbel" {
                search.best_move
            } else if temperature <= 1e-6 {
                search.best_move.or_else(|| {
                    choose_selfplay_move(&search.candidates, temperature, 0.0, 0.0, &mut rng)
                })
            } else {
                choose_selfplay_move(
                    &search.candidates,
                    temperature,
                    config.temperature_value_cutoff,
                    config.temperature_visit_offset,
                    &mut rng,
                )
            };
            let Some(mv) = mv_opt else {
                result = Some(0.0);
                break;
            };
            let bootstrap_value = if config.search == "gumbel" {
                search
                    .candidates
                    .iter()
                    .find(|candidate| candidate.mv == mv)
                    .map(|candidate| candidate.q)
                    .unwrap_or(search.value_q)
            } else {
                search.value_q
            };
            let move_meta = move_search_meta(
                &search.candidates,
                mv,
                bootstrap_value,
                config.generation_update,
                config.seed ^ game_index as u64,
                ply,
                if config.search == "gumbel" {
                    0.0
                } else {
                    config.deblunder_q_gap
                },
            );
            sampled_moves += 1;
            sampled_best_moves +=
                usize::from(move_meta.sample.best_index == move_meta.sample.played_index);
            deblundered_moves += usize::from(move_meta.sample.deblundered);
            best_played_q_gap_sum += (move_meta.sample.best_q - move_meta.sample.played_q).max(0.0);
            let top_visits = search
                .candidates
                .iter()
                .map(|candidate| candidate.visits)
                .max()
                .unwrap_or(0);
            played_top_visit_ratio_sum += if top_visits == 0 {
                0.0
            } else {
                move_meta.sample.played_visits as f32 / top_visits as f32
            };
            best_q_sum += move_meta.sample.best_q;
            played_q_sum += move_meta.sample.played_q;
            let side_sign = if position.side_to_move() == Color::Red {
                1.0
            } else {
                -1.0
            };
            if let Some(best_q) = move_meta.deblunder_boundary_q {
                deblunder_events.push(DeblunderEvent {
                    sample_index: game_samples.len(),
                    boundary_red: (best_q * side_sign).clamp(-1.0, 1.0),
                });
            }
            game_samples.push(make_training_sample(
                &position,
                &history,
                &search.candidates,
                bootstrap_value,
                if config.search == "gumbel" {
                    1.0
                } else {
                    config.policy_softmax_temp
                },
                rng.unit_f32() < config.mirror_probability.clamp(0.0, 1.0),
                move_meta.sample,
            ));
            append_history(&mut history, &position, mv);
            rule_history.push(position.rule_history_entry_after_move(mv));
            position.make_move(mv);

            if !position.has_general(Color::Red) {
                result = Some(-1.0);
                terminal.red_general_missing += 1;
                break;
            }
            if !position.has_general(Color::Black) {
                result = Some(1.0);
                terminal.black_general_missing += 1;
                break;
            }
            if let Some(rule_outcome) = position.rule_outcome_with_history(&rule_history) {
                result = Some(match rule_outcome {
                    RuleOutcome::Draw(_) => 0.0,
                    RuleOutcome::Win(Color::Red) => 1.0,
                    RuleOutcome::Win(Color::Black) => -1.0,
                });
                match rule_outcome {
                    RuleOutcome::Draw(reason) => {
                        terminal.rule_draw += 1;
                        match reason {
                            RuleDrawReason::Halfmove120 => terminal.rule_draw_halfmove120 += 1,
                            RuleDrawReason::Repetition => terminal.rule_draw_repetition += 1,
                            RuleDrawReason::MutualLongCheck => {
                                terminal.rule_draw_mutual_long_check += 1
                            }
                            RuleDrawReason::MutualLongChase => {
                                terminal.rule_draw_mutual_long_chase += 1
                            }
                        }
                    }
                    RuleOutcome::Win(Color::Red) => terminal.rule_win_red += 1,
                    RuleOutcome::Win(Color::Black) => terminal.rule_win_black += 1,
                }
                break;
            }
        }
        if result.is_none() {
            terminal.max_plies += 1;
        }

        let result: f32 = result.unwrap_or(0.0);
        match result.total_cmp(&0.0) {
            std::cmp::Ordering::Greater => red_wins += 1,
            std::cmp::Ordering::Less => black_wins += 1,
            std::cmp::Ordering::Equal => draws += 1,
        }
        plies_total += plies;

        assign_deblundered_value_targets(&mut game_samples, result, &deblunder_events, config);
        assign_moves_left_targets(&mut game_samples, config.max_plies);
        samples.extend(game_samples.clone());
        games.push(game_samples);
    }

    AzSelfplayData {
        samples,
        games,
        red_wins,
        black_wins,
        draws,
        plies_total,
        entropy_all_sum,
        entropy_all_count,
        entropy_opening_sum,
        entropy_opening_count,
        entropy_mid_sum,
        entropy_mid_count,
        raw_prior_top1_sum,
        raw_prior_top2_sum,
        policy_top1_sum,
        policy_top2_sum,
        q_gap_sum,
        q_top1_abs_sum,
        visited_actions_sum,
        shape_count,
        opening_raw_prior_top1_sum,
        opening_raw_prior_top2_sum,
        opening_policy_top1_sum,
        opening_policy_top2_sum,
        opening_q_gap_sum,
        opening_q_top1_abs_sum,
        opening_visited_actions_sum,
        opening_shape_count,
        sampled_moves,
        sampled_best_moves,
        deblundered_moves,
        best_played_q_gap_sum,
        played_top_visit_ratio_sum,
        best_q_sum,
        played_q_sum,
        terminal,
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct PolicyShapeStats {
    raw_prior_top1: f32,
    raw_prior_top2: f32,
    policy_top1: f32,
    policy_top2: f32,
    q_gap: f32,
    q_top1_abs: f32,
    visited_actions: usize,
}

fn policy_shape_stats(candidates: &[AzCandidate]) -> PolicyShapeStats {
    let mut raw_top = [0.0f32; 2];
    let mut policy_top = [0.0f32; 2];
    let mut q_top = [f32::NEG_INFINITY; 2];
    let mut visited_actions = 0usize;
    for candidate in candidates {
        insert_top2(candidate.raw_prior.max(0.0), &mut raw_top);
        insert_top2(candidate.policy.max(0.0), &mut policy_top);
        if candidate.visits > 0 {
            insert_top2(candidate.q, &mut q_top);
            visited_actions += 1;
        }
    }
    let q_gap = if q_top[1].is_finite() {
        (q_top[0] - q_top[1]).max(0.0)
    } else {
        0.0
    };
    let q_top1_abs = if q_top[0].is_finite() {
        q_top[0].abs()
    } else {
        0.0
    };
    PolicyShapeStats {
        raw_prior_top1: raw_top[0],
        raw_prior_top2: raw_top[0] + raw_top[1],
        policy_top1: policy_top[0],
        policy_top2: policy_top[0] + policy_top[1],
        q_gap,
        q_top1_abs,
        visited_actions,
    }
}

fn insert_top2(value: f32, top: &mut [f32; 2]) {
    if value > top[0] {
        top[1] = top[0];
        top[0] = value;
    } else if value > top[1] {
        top[1] = value;
    }
}

fn make_training_sample(
    position: &Position,
    history: &[HistoryMove],
    candidates: &[AzCandidate],
    value: f32,
    policy_softmax_temp: f32,
    mirror_file: bool,
    meta: AzSampleMeta,
) -> AzTrainingSample {
    let side = position.side_to_move();
    let side_sign = if side == Color::Red { 1.0 } else { -1.0 };
    let policy_softmax_temp = policy_softmax_temp.max(1e-3);
    let mut features = extract_sparse_features_az_canonical(position, history);
    let mut moves = candidates
        .iter()
        .map(|candidate| candidate.mv)
        .collect::<Vec<_>>();
    if mirror_file {
        mirror_sparse_features_az_absolute_file(&mut features);
        for mv in &mut moves {
            *mv = mirror_file_move(*mv);
        }
    }
    let move_indices = moves
        .iter()
        .copied()
        .map(|mv| dense_move_index(canonical_move(side, mv)))
        .collect();
    let mut policy = candidates
        .iter()
        .map(|candidate| candidate.policy.max(1e-12).powf(1.0 / policy_softmax_temp))
        .collect::<Vec<_>>();
    let total_policy = policy.iter().sum::<f32>().max(1e-12);
    for value in &mut policy {
        *value /= total_policy;
    }

    AzTrainingSample {
        features,
        move_indices,
        policy,
        value_wdl: scalar_value_to_wdl_target(value),
        value: value.clamp(-1.0, 1.0),
        side_sign,
        moves_left: 0.0,
        meta,
    }
}

pub(super) fn assign_td_lambda_value_targets(
    samples: &mut [AzTrainingSample],
    game_result_red: f32,
    td_lambda: f32,
) {
    if samples.is_empty() {
        return;
    }
    let td_lambda = td_lambda.clamp(0.0, 1.0);
    let result_red = game_result_red.clamp(-1.0, 1.0);
    let search_values_red = samples
        .iter()
        .map(|sample| (sample.value * sample.side_sign).clamp(-1.0, 1.0))
        .collect::<Vec<_>>();
    let mut return_red = result_red;
    for index in (0..samples.len()).rev() {
        if index + 1 < samples.len() {
            return_red = (search_values_red[index + 1] * (1.0 - td_lambda)
                + return_red * td_lambda)
                .clamp(-1.0, 1.0);
        }
        let sample = &mut samples[index];
        let side_value = (return_red * sample.side_sign).clamp(-1.0, 1.0);
        let side_result = (game_result_red * sample.side_sign).clamp(-1.0, 1.0);
        sample.value_wdl = scalar_value_to_wdl_target(side_result);
        sample.value = side_value;
    }
}

fn root_search_meta(
    candidates: &[AzCandidate],
    root_q: f32,
    generation_update: u32,
    game_id: u64,
    ply: usize,
) -> AzSampleMeta {
    let mut meta = AzSampleMeta {
        generation_update,
        game_id,
        ply: ply.min(u16::MAX as usize) as u16,
        root_q,
        best_index: u16::MAX,
        played_index: u16::MAX,
        ..AzSampleMeta::default()
    };
    if let Some((best_index, best)) = candidates
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.q.total_cmp(&right.q))
    {
        meta.best_q = best.q;
        meta.best_visits = best.visits;
        meta.best_index = best_index.min(u16::MAX as usize) as u16;
    }
    meta
}

fn move_search_meta(
    candidates: &[AzCandidate],
    mv: Move,
    root_q: f32,
    generation_update: u32,
    game_id: u64,
    ply: usize,
    q_gap: f32,
) -> MoveSearchMeta {
    let mut meta = root_search_meta(candidates, root_q, generation_update, game_id, ply);
    if let Some((played_index, played)) = candidates
        .iter()
        .enumerate()
        .find(|(_, candidate)| candidate.mv == mv)
    {
        meta.played_q = played.q;
        meta.played_visits = played.visits;
        meta.played_index = played_index.min(u16::MAX as usize) as u16;
    }
    let deblunder_boundary_q = if q_gap > 0.0
        && meta.best_index != meta.played_index
        && meta.played_index != u16::MAX
        && meta.best_q - meta.played_q >= q_gap
    {
        meta.deblundered = true;
        Some(meta.best_q)
    } else {
        None
    };
    MoveSearchMeta {
        sample: meta,
        deblunder_boundary_q,
    }
}

fn assign_deblundered_value_targets(
    samples: &mut [AzTrainingSample],
    game_result_red: f32,
    deblunder_events: &[DeblunderEvent],
    config: &AzLoopConfig,
) {
    if deblunder_events.is_empty() {
        assign_value_targets(samples, game_result_red, config);
        return;
    }

    let mut start = 0usize;
    for event in deblunder_events {
        if event.sample_index < start || event.sample_index >= samples.len() {
            continue;
        }
        let end = event.sample_index + 1;
        assign_value_targets(&mut samples[start..end], event.boundary_red, config);
        start = end;
    }
    if start < samples.len() {
        assign_value_targets(&mut samples[start..], game_result_red, config);
    }
}

fn assign_value_targets(
    samples: &mut [AzTrainingSample],
    game_result_red: f32,
    config: &AzLoopConfig,
) {
    assign_td_lambda_value_targets(samples, game_result_red, config.td_lambda);
}

fn should_resign(root_q: f32, config: &AzLoopConfig) -> bool {
    if config.resign_percentage <= 0.0 {
        return false;
    }
    let threshold = -(1.0 - config.resign_percentage.clamp(0.0, 100.0) / 100.0);
    root_q <= threshold
}

pub(super) fn assign_moves_left_targets(samples: &mut [AzTrainingSample], _max_plies: usize) {
    let game_len = samples.len();
    for (index, sample) in samples.iter_mut().enumerate() {
        let remaining = game_len.saturating_sub(index).max(1) as f32;
        sample.moves_left = remaining;
    }
}

fn temperature_for_ply(config: &AzLoopConfig, ply: usize) -> f32 {
    if ply < config.temperature_decay_delay_plies {
        return config.temperature_start;
    }
    if config.temperature_decay_plies == 0 {
        return config.temperature_endgame;
    }
    let decay_ply = ply.saturating_sub(config.temperature_decay_delay_plies);
    if decay_ply >= config.temperature_decay_plies {
        return config.temperature_endgame;
    }
    let progress = decay_ply as f32 / config.temperature_decay_plies as f32;
    config.temperature_start + (config.temperature_endgame - config.temperature_start) * progress
}

fn temperature_opening_plies(config: &AzLoopConfig) -> usize {
    config
        .temperature_decay_delay_plies
        .saturating_add(config.temperature_decay_plies)
}

fn choose_selfplay_move(
    candidates: &[AzCandidate],
    temperature: f32,
    value_cutoff: f32,
    visit_offset: f32,
    rng: &mut SplitMix64,
) -> Option<Move> {
    if temperature <= 1e-6 {
        return candidates
            .iter()
            .max_by(|left, right| {
                left.policy
                    .total_cmp(&right.policy)
                    .then_with(|| left.visits.cmp(&right.visits))
            })
            .map(|candidate| candidate.mv);
    }

    let weights = temperature_move_weights(candidates, temperature, value_cutoff, visit_offset);
    let total = candidates
        .iter()
        .zip(&weights)
        .map(|(_, weight)| *weight)
        .sum::<f32>();
    if total <= 0.0 {
        return candidates.first().map(|candidate| candidate.mv);
    }

    let mut ticket = rng.unit_f32() * total;
    for (candidate, weight) in candidates.iter().zip(weights) {
        if ticket < weight {
            return Some(candidate.mv);
        }
        ticket -= weight;
    }
    candidates.first().map(|candidate| candidate.mv)
}

fn temperature_move_weights(
    candidates: &[AzCandidate],
    temperature: f32,
    value_cutoff: f32,
    visit_offset: f32,
) -> Vec<f32> {
    let best_q = candidates
        .iter()
        .map(|candidate| candidate.q)
        .fold(f32::NEG_INFINITY, f32::max);
    let inv_temperature = 1.0 / temperature.max(1e-3);
    let mut weights = candidates
        .iter()
        .map(|candidate| {
            (candidate.visits as f32 - visit_offset)
                .max(1e-9)
                .powf(inv_temperature)
        })
        .collect::<Vec<_>>();

    if value_cutoff <= 0.0 || value_cutoff >= 1.0 || !best_q.is_finite() {
        return weights;
    }

    const MIN_CUTOFF_CANDIDATES: usize = 8;
    const MIN_CUTOFF_WEIGHT_MASS: f32 = 0.98;

    let total_weight = weights.iter().sum::<f32>();
    if total_weight <= 0.0 {
        return weights;
    }

    let mut keep = candidates
        .iter()
        .map(|candidate| {
            let best_win = (best_q + 1.0) * 0.5;
            let win = (candidate.q + 1.0) * 0.5;
            best_win - win <= value_cutoff
        })
        .collect::<Vec<_>>();
    if keep.iter().filter(|&&kept| kept).count() < MIN_CUTOFF_CANDIDATES {
        let mut ranked = candidates
            .iter()
            .enumerate()
            .collect::<Vec<(usize, &AzCandidate)>>();
        ranked.sort_by(|(_, left), (_, right)| {
            right
                .visits
                .cmp(&left.visits)
                .then_with(|| right.q.total_cmp(&left.q))
        });
        for (index, _) in ranked.into_iter().take(MIN_CUTOFF_CANDIDATES) {
            keep[index] = true;
        }
    }

    let kept_weight = weights
        .iter()
        .zip(&keep)
        .filter_map(|(weight, kept)| kept.then_some(*weight))
        .sum::<f32>();
    if kept_weight / total_weight < MIN_CUTOFF_WEIGHT_MASS {
        let mut ranked = weights
            .iter()
            .copied()
            .enumerate()
            .collect::<Vec<(usize, f32)>>();
        ranked.sort_by(|(_, left), (_, right)| right.total_cmp(left));
        let mut mass = kept_weight;
        for (index, weight) in ranked {
            if keep[index] {
                continue;
            }
            keep[index] = true;
            mass += weight;
            if mass / total_weight >= MIN_CUTOFF_WEIGHT_MASS {
                break;
            }
        }
    }

    for (weight, kept) in weights.iter_mut().zip(keep) {
        if !kept {
            *weight = 0.0;
        }
    }
    weights
}

fn policy_entropy(candidates: &[AzCandidate]) -> f32 {
    const EPS: f32 = 1e-10;
    let total = candidates
        .iter()
        .map(|candidate| candidate.policy.max(0.0))
        .sum::<f32>();
    if total <= 0.0 {
        return 0.0;
    }
    candidates
        .iter()
        .map(|candidate| {
            let p = (candidate.policy.max(0.0) / total).max(0.0);
            if p <= 0.0 { 0.0 } else { -p * (p + EPS).ln() }
        })
        .sum()
}

#[derive(Clone, Copy, Debug)]
pub struct AzArenaConfig {
    pub simulations: usize,
    pub max_plies: usize,
    pub games_as_red: usize,
    pub games_as_black: usize,
    pub start_index: usize,
    pub seed: u64,
    pub cpuct: f32,
}

pub fn play_arena_games_from_positions(
    candidate: &AzNnue,
    baseline: &AzNnue,
    positions: &[Position],
    config: AzArenaConfig,
) -> AzArenaReport {
    let mut report = AzArenaReport::default();
    let mut game_seed = config.seed;
    for game_index in 0..config.games_as_red {
        let position = arena_start_position(positions, config.start_index + game_index);
        let outcome = play_arena_game(
            &position,
            candidate,
            baseline,
            config.simulations,
            config.max_plies,
            game_seed,
            config.cpuct,
        );
        match outcome.total_cmp(&0.0) {
            std::cmp::Ordering::Greater => {
                report.wins += 1;
                report.wins_as_red += 1;
            }
            std::cmp::Ordering::Less => {
                report.losses += 1;
                report.losses_as_red += 1;
            }
            std::cmp::Ordering::Equal => report.draws += 1,
        }
        game_seed = game_seed.wrapping_add(1);
    }
    for game_index in 0..config.games_as_black {
        let position = arena_start_position(positions, config.start_index + game_index);
        let outcome = play_arena_game(
            &position,
            baseline,
            candidate,
            config.simulations,
            config.max_plies,
            game_seed,
            config.cpuct,
        );
        match outcome.total_cmp(&0.0) {
            std::cmp::Ordering::Greater => {
                report.losses += 1;
                report.losses_as_black += 1;
            }
            std::cmp::Ordering::Less => {
                report.wins += 1;
                report.wins_as_black += 1;
            }
            std::cmp::Ordering::Equal => report.draws += 1,
        }
        game_seed = game_seed.wrapping_add(1);
    }
    report
}

fn arena_start_position(positions: &[Position], game_index: usize) -> Position {
    if positions.is_empty() {
        Position::startpos()
    } else {
        let index = game_index % positions.len();
        positions[index].clone()
    }
}

fn play_arena_game(
    initial_position: &Position,
    red_model: &AzNnue,
    black_model: &AzNnue,
    simulations: usize,
    max_plies: usize,
    seed: u64,
    cpuct: f32,
) -> f32 {
    let mut position = initial_position.clone();
    let mut history = Vec::new();
    let mut rule_history = position.initial_rule_history();
    for ply in 0..max_plies {
        let legal = position.legal_moves_with_rules(&rule_history);
        if legal.is_empty() {
            return if position.side_to_move() == Color::Red {
                -1.0
            } else {
                1.0
            };
        }
        let model = if position.side_to_move() == Color::Red {
            red_model
        } else {
            black_model
        };
        let result = alphazero_search_with_history_and_rules(
            &position,
            &history,
            Some(rule_history.clone()),
            Some(legal),
            model,
            AzSearchLimits {
                simulations,
                seed: seed ^ ((ply as u64) << 32),
                cpuct,
                cpuct_at_root: cpuct,
                cpuct_base: 19652.0,
                cpuct_factor: 2.0,
                cpuct_base_at_root: 19652.0,
                cpuct_factor_at_root: 2.0,
                max_depth: 0,
                root_dirichlet_alpha: 0.0,
                root_exploration_fraction: 0.0,
                fpu_value: 0.23,
                fpu_value_at_root: 1.0,
                draw_score: 0.0,
                moves_left_max_effect: 0.0,
                moves_left_slope: 0.0,
                moves_left_threshold: 0.6,
                moves_left_constant_factor: 0.0,
                moves_left_scaled_factor: 0.0,
                moves_left_quadratic_factor: 0.0,
                value_scale: 1.0,
            },
        );
        let Some(mv) = result.best_move else {
            return 0.0;
        };
        append_history(&mut history, &position, mv);
        rule_history.push(position.rule_history_entry_after_move(mv));
        position.make_move(mv);

        if !position.has_general(Color::Red) {
            return -1.0;
        }
        if !position.has_general(Color::Black) {
            return 1.0;
        }
        if let Some(rule_outcome) = position.rule_outcome_with_history(&rule_history) {
            return match rule_outcome {
                RuleOutcome::Draw(_) => 0.0,
                RuleOutcome::Win(Color::Red) => 1.0,
                RuleOutcome::Win(Color::Black) => -1.0,
            };
        }
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(mv: Move, policy: f32) -> AzCandidate {
        AzCandidate {
            mv,
            visits: (policy * 100.0) as u32,
            q: 0.0,
            value_wdl: [0.0, 1.0, 0.0],
            moves_left: 0.0,
            raw_prior: policy,
            prior: policy,
            policy,
        }
    }

    fn candidate_q(mv: Move, visits: u32, q: f32) -> AzCandidate {
        AzCandidate {
            mv,
            visits,
            q,
            value_wdl: [0.0, 1.0, 0.0],
            moves_left: 0.0,
            raw_prior: 0.0,
            prior: 0.0,
            policy: 0.0,
        }
    }

    fn sample(value: f32, side_sign: f32) -> AzTrainingSample {
        AzTrainingSample {
            features: Vec::new(),
            move_indices: Vec::new(),
            policy: Vec::new(),
            value_wdl: scalar_value_to_wdl_target(value),
            value,
            side_sign,
            moves_left: 0.0,
            meta: AzSampleMeta::default(),
        }
    }

    fn test_config() -> AzLoopConfig {
        AzLoopConfig {
            games: 1,
            search: "alphazero".into(),
            gumbel_actions: 16,
            gumbel_scale: 1.0,
            gumbel_value_scale: 0.02,
            gumbel_maxvisit_init: 50.0,
            max_plies: 100,
            simulations: 1,
            seed: 1,
            workers: 1,
            generation_update: 0,
            temperature_start: 1.0,
            temperature_endgame: 0.0,
            temperature_decay_delay_plies: 0,
            temperature_decay_plies: 0,
            temperature_value_cutoff: 0.0,
            temperature_visit_offset: 0.0,
            cpuct: 1.0,
            cpuct_at_root: 1.0,
            cpuct_base: 1.0,
            cpuct_factor: 0.0,
            cpuct_base_at_root: 1.0,
            cpuct_factor_at_root: 0.0,
            root_dirichlet_alpha: 0.0,
            root_exploration_fraction: 0.0,
            fpu_value: 0.0,
            fpu_value_at_root: 0.0,
            draw_score: 0.0,
            moves_left_max_effect: 0.0,
            moves_left_slope: 0.0,
            moves_left_threshold: 0.0,
            moves_left_constant_factor: 0.0,
            moves_left_scaled_factor: 0.0,
            moves_left_quadratic_factor: 0.0,
            policy_softmax_temp: 1.0,
            opening_positions: Vec::new(),
            resign_percentage: 0.0,
            resign_playthrough: 0.0,
            mirror_probability: 0.0,
            deblunder_q_gap: 0.25,
            td_lambda: 1.0,
        }
    }

    #[test]
    fn arena_promotion_uses_score_lower_bound() {
        let report = AzArenaReport {
            wins: 84,
            losses: 68,
            draws: 48,
            ..AzArenaReport::default()
        };

        assert!((report.score_rate() - 0.54).abs() < 1e-6);
        assert!(report.promotes_with_lower_bound(0.50, 1.0));
        assert!(!report.promotes_with_lower_bound(0.50, 1.64));
    }

    #[test]
    fn mirrored_training_sample_mirrors_move_indices() {
        let position =
            Position::from_fen("3ak4/9/2n1b4/p3p3p/4R4/2P6/P3P3P/2N1C4/4A4/2BAK3c b").unwrap();
        let moves = position.legal_moves();
        let candidates = moves
            .iter()
            .take(4)
            .enumerate()
            .map(|(index, &mv)| candidate(mv, 1.0 / (index + 2) as f32))
            .collect::<Vec<_>>();
        let sample = make_training_sample(
            &position,
            &[],
            &candidates,
            0.0,
            1.45,
            true,
            AzSampleMeta::default(),
        );

        let mirrored_position = position.mirror_files();
        let mirrored_moves = candidates
            .iter()
            .map(|candidate| mirror_file_move(candidate.mv))
            .collect::<Vec<_>>();
        let expected = mirrored_moves
            .iter()
            .copied()
            .map(|mv| dense_move_index(canonical_move(mirrored_position.side_to_move(), mv)))
            .collect::<Vec<_>>();

        assert_eq!(sample.move_indices, expected);
    }

    #[test]
    fn deblunder_value_targets_split_on_each_repair_event() {
        let mut samples = vec![
            sample(0.1, 1.0),
            sample(0.2, -1.0),
            sample(-0.3, 1.0),
            sample(-0.4, -1.0),
        ];
        let events = vec![
            DeblunderEvent {
                sample_index: 1,
                boundary_red: 0.6,
            },
            DeblunderEvent {
                sample_index: 2,
                boundary_red: -0.5,
            },
        ];

        assign_deblundered_value_targets(&mut samples, 1.0, &events, &test_config());

        assert!((samples[0].value - 0.6).abs() < 1e-6);
        assert!((samples[1].value + 0.6).abs() < 1e-6);
        assert!((samples[2].value + 0.5).abs() < 1e-6);
        assert!((samples[3].value + 1.0).abs() < 1e-6);
    }

    #[test]
    fn moves_left_targets_use_raw_remaining_plies() {
        let mut samples = vec![sample(0.0, 1.0), sample(0.0, -1.0), sample(0.0, 1.0)];

        assign_moves_left_targets(&mut samples, 300);

        assert_eq!(
            samples
                .iter()
                .map(|sample| sample.moves_left)
                .collect::<Vec<_>>(),
            vec![3.0, 2.0, 1.0]
        );
    }

    #[test]
    fn sampled_move_deblunder_boundary_uses_best_q() {
        let moves = [
            Move { from: 0, to: 1 },
            Move { from: 1, to: 2 },
            Move { from: 2, to: 3 },
        ];
        let mut candidates = vec![
            candidate(moves[0], 0.5),
            candidate(moves[1], 0.3),
            candidate(moves[2], 0.2),
        ];
        candidates[0].q = 0.7;
        candidates[1].q = 0.35;
        candidates[2].q = 0.65;

        let meta = move_search_meta(&candidates, moves[1], 0.2, 3, 99, 7, 0.25);
        assert_eq!(meta.deblunder_boundary_q, Some(0.7));
        assert!(meta.sample.deblundered);
        assert_eq!(meta.sample.generation_update, 3);
        assert_eq!(meta.sample.game_id, 99);
        assert_eq!(meta.sample.ply, 7);
        assert_eq!(meta.sample.best_index, 0);
        assert_eq!(meta.sample.played_index, 1);
        let quiet_meta = move_search_meta(&candidates, moves[2], 0.2, 3, 99, 7, 0.25);
        assert_eq!(quiet_meta.deblunder_boundary_q, None);
        assert!(!quiet_meta.sample.deblundered);
    }

    #[test]
    fn temperature_value_cutoff_uses_win_probability_gap() {
        let mut candidates = vec![
            candidate_q(Move::new(0, 1), 100, 0.80),
            candidate_q(Move::new(0, 2), 1, 0.60),
        ];
        for index in 2..10 {
            candidates.push(candidate_q(Move::new(index, index + 1), 10, 0.40));
        }

        let weights = temperature_move_weights(&candidates, 1.0, 0.15, 0.0);

        assert!(weights[1] > 0.0);
    }

    #[test]
    fn temperature_value_cutoff_keeps_exploration_floor() {
        let candidates = (0..12)
            .map(|index| {
                candidate_q(
                    Move::new(index, index + 1),
                    if index == 0 { 100 } else { 10 },
                    0.90 - index as f32 * 0.10,
                )
            })
            .collect::<Vec<_>>();

        let weights = temperature_move_weights(&candidates, 1.0, 0.15, 0.0);
        let kept = weights.iter().filter(|&&weight| weight > 0.0).count();
        let total_without_cutoff = 100.0 + 11.0 * 10.0;
        let kept_weight = weights.iter().sum::<f32>();

        assert!(kept >= 8);
        assert!(kept_weight / total_without_cutoff >= 0.98);
    }
}
