use std::sync::Arc;

use rayon::prelude::*;

use crate::nnue::{
    canonical_move, extract_sparse_features_az, mirror_file_move,
    mirror_sparse_features_az_canonical_file,
};
use crate::xiangqi::{Color, Move, Position, RuleDrawReason, RuleOutcome};

use super::{
    AzLoopConfig, AzNnue, AzSampleMeta, AzTrainingSample, GumbelCandidate, GumbelSearchLimits,
    SplitMix64, dense_move_index, gumbel_search_with_rules, scalar_value_to_wdl_target,
};

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

#[derive(Clone, Copy, Debug, Default)]
pub struct AzSearchSimulationStats {
    pub searches: usize,
    pub simulations_sum: usize,
}

impl AzSearchSimulationStats {
    pub fn add_assign(&mut self, other: &Self) {
        self.searches += other.searches;
        self.simulations_sum += other.simulations_sum;
    }
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
    pub best_played_q_gap_sum: f32,
    pub played_top_visit_ratio_sum: f32,
    pub best_q_sum: f32,
    pub played_q_sum: f32,
    pub terminal: AzTerminalStats,
    pub search_simulations: AzSearchSimulationStats,
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
        self.best_played_q_gap_sum += other.best_played_q_gap_sum;
        self.played_top_visit_ratio_sum += other.played_top_visit_ratio_sum;
        self.best_q_sum += other.best_q_sum;
        self.played_q_sum += other.played_q_sum;
        self.terminal.add_assign(&other.terminal);
        self.search_simulations
            .add_assign(&other.search_simulations);
    }
}

pub fn generate_selfplay_data(model: &AzNnue, config: &AzLoopConfig) -> AzSelfplayData {
    crate::scope_profile!("az.selfplay.generate");
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
                let chunk = generate_selfplay_chunk(&shared_model, &worker_config);
                crate::profile::flush_thread();
                chunk
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
        merged.best_played_q_gap_sum += chunk.best_played_q_gap_sum;
        merged.played_top_visit_ratio_sum += chunk.played_top_visit_ratio_sum;
        merged.best_q_sum += chunk.best_q_sum;
        merged.played_q_sum += chunk.played_q_sum;
        merged.terminal.add_assign(&chunk.terminal);
        merged
            .search_simulations
            .add_assign(&chunk.search_simulations);
    }
    merged
}

fn generate_selfplay_chunk(model: &AzNnue, config: &AzLoopConfig) -> AzSelfplayData {
    crate::scope_profile!("az.selfplay.chunk");
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
    let mut best_played_q_gap_sum = 0.0f32;
    let mut played_top_visit_ratio_sum = 0.0f32;
    let mut best_q_sum = 0.0f32;
    let mut played_q_sum = 0.0f32;
    let mut terminal = AzTerminalStats::default();
    let mut search_simulations = AzSearchSimulationStats::default();

    for game_index in 0..config.games {
        let mut position = if config.opening_positions.is_empty() {
            Position::startpos()
        } else {
            let index = (rng.next_u64() as usize) % config.opening_positions.len();
            config.opening_positions[index].clone()
        };
        let mut rule_history = position.initial_rule_history();
        let mut game_samples = Vec::new();
        let mut result = None;
        let mut plies = 0usize;
        let allow_resign = rng.unit_f32() * 100.0 >= config.resign_playthrough;

        for ply in 0..config.max_plies {
            plies = ply + 1;
            let legal = {
                crate::scope_profile!("az.selfplay.root_legal_moves");
                position.legal_moves_with_rules(&rule_history)
            };
            if legal.is_empty() {
                result = Some(if position.side_to_move() == Color::Red {
                    -1.0
                } else {
                    1.0
                });
                terminal.no_legal_moves += 1;
                break;
            }

            let search_simulation_count = config.simulations.max(1);
            search_simulations.searches += 1;
            search_simulations.simulations_sum += search_simulation_count;
            let search = {
                crate::scope_profile!("az.selfplay.search");
                gumbel_search_with_rules(
                    &position,
                    Some(rule_history.clone()),
                    Some(legal),
                    model,
                    GumbelSearchLimits {
                        simulations: search_simulation_count,
                        seed: rng.next_u64() ^ ((game_index as u64) << 32) ^ ply as u64,
                        max_num_considered_actions: config.gumbel_max_num_considered_actions,
                        gumbel_scale: config.gumbel_scale,
                        q_value_scale: config.gumbel_q_value_scale,
                        q_maxvisit_init: config.gumbel_q_maxvisit_init,
                        max_depth: 0,
                        draw_score: config.draw_score,
                        value_scale: 1.0,
                    },
                )
            };
            crate::scope_profile!("az.selfplay.post_search");
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
            if ply < 90 {
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
                let sample = make_training_sample(
                    &position,
                    &search.candidates,
                    search.value_q,
                    rng.unit_f32() < config.mirror_probability.clamp(0.0, 1.0),
                    meta,
                    search_simulation_count,
                    1.0,
                );
                game_samples.push(sample);
                result = Some(if position.side_to_move() == Color::Red {
                    terminal.resign_red += 1;
                    -1.0
                } else {
                    terminal.resign_black += 1;
                    1.0
                });
                break;
            }
            let mv_opt = search.best_move;
            let Some(mv) = mv_opt else {
                result = Some(0.0);
                break;
            };
            let move_meta = move_search_meta(
                &search.candidates,
                mv,
                search.value_q,
                config.generation_update,
                config.seed ^ game_index as u64,
                ply,
            );
            sampled_moves += 1;
            sampled_best_moves += usize::from(move_meta.best_index == move_meta.played_index);
            best_played_q_gap_sum += (move_meta.best_q - move_meta.played_q).max(0.0);
            let top_visits = search
                .candidates
                .iter()
                .map(|candidate| candidate.visits)
                .max()
                .unwrap_or(0);
            played_top_visit_ratio_sum += if top_visits == 0 {
                0.0
            } else {
                move_meta.played_visits as f32 / top_visits as f32
            };
            best_q_sum += move_meta.best_q;
            played_q_sum += move_meta.played_q;
            {
                crate::scope_profile!("az.selfplay.make_sample");
                let sample = make_training_sample(
                    &position,
                    &search.candidates,
                    search.value_q,
                    rng.unit_f32() < config.mirror_probability.clamp(0.0, 1.0),
                    move_meta,
                    search_simulation_count,
                    1.0,
                );
                game_samples.push(sample);
            }
            let mover = position.side_to_move();
            position.make_move(mv);
            rule_history.push(position.rule_history_entry_after_moved(mover, mv.to as usize));

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
            let rule_outcome = {
                crate::scope_profile!("az.selfplay.rule_outcome");
                position.rule_outcome_with_history(&rule_history)
            };
            if let Some(rule_outcome) = rule_outcome {
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

        {
            crate::scope_profile!("az.selfplay.finalize_game");
            assign_value_targets(&mut game_samples, result, config);
            assign_moves_left_targets(&mut game_samples, config.max_plies);
        }
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
        best_played_q_gap_sum,
        played_top_visit_ratio_sum,
        best_q_sum,
        played_q_sum,
        terminal,
        search_simulations,
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

fn policy_shape_stats(candidates: &[GumbelCandidate]) -> PolicyShapeStats {
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
    candidates: &[GumbelCandidate],
    value: f32,
    mirror_file: bool,
    meta: AzSampleMeta,
    search_simulations: usize,
    policy_weight: f32,
) -> AzTrainingSample {
    let side = position.side_to_move();
    let side_sign = if side == Color::Red { 1.0 } else { -1.0 };
    let mut features = extract_sparse_features_az(position);
    let mut moves = candidates
        .iter()
        .map(|candidate| candidate.mv)
        .collect::<Vec<_>>();
    if mirror_file {
        mirror_sparse_features_az_canonical_file(&mut features);
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
        .map(|candidate| candidate.policy.max(0.0))
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
        policy_weight: policy_weight.max(0.0),
        value_weight: 1.0,
        search_simulations: search_simulations.min(u32::MAX as usize) as u32,
        meta,
    }
}

fn root_search_meta(
    candidates: &[GumbelCandidate],
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
    candidates: &[GumbelCandidate],
    mv: Move,
    root_q: f32,
    generation_update: u32,
    game_id: u64,
    ply: usize,
) -> AzSampleMeta {
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
    meta
}

fn assign_value_targets(
    samples: &mut [AzTrainingSample],
    game_result_red: f32,
    _config: &AzLoopConfig,
) {
    for sample in samples {
        let side_result = (game_result_red * sample.side_sign).clamp(-1.0, 1.0);
        sample.value_wdl = scalar_value_to_wdl_target(side_result);
        sample.value = side_result;
    }
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

fn policy_entropy(candidates: &[GumbelCandidate]) -> f32 {
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
) -> f32 {
    let mut position = initial_position.clone();
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
        let result = gumbel_search_with_rules(
            &position,
            Some(rule_history.clone()),
            Some(legal),
            model,
            GumbelSearchLimits {
                simulations,
                seed: seed ^ ((ply as u64) << 32),
                max_num_considered_actions: 16,
                gumbel_scale: 0.0,
                q_value_scale: 0.1,
                q_maxvisit_init: 50.0,
                max_depth: 0,
                draw_score: 0.0,
                value_scale: 1.0,
            },
        );
        let Some(mv) = result.best_move else {
            return 0.0;
        };
        let mover = position.side_to_move();
        position.make_move(mv);
        rule_history.push(position.rule_history_entry_after_moved(mover, mv.to as usize));

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

    fn candidate(mv: Move, policy: f32) -> GumbelCandidate {
        GumbelCandidate {
            mv,
            visits: (policy * 100.0) as u32,
            q: 0.0,
            moves_left: 0.0,
            raw_prior: policy,
            prior: policy,
            policy,
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
            policy_weight: 1.0,
            value_weight: 1.0,
            search_simulations: 0,
            meta: AzSampleMeta::default(),
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
            &candidates,
            0.0,
            true,
            AzSampleMeta::default(),
            1,
            1.0,
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
        let expected_policy = candidates
            .iter()
            .map(|candidate| candidate.policy)
            .collect::<Vec<_>>();
        let expected_total = expected_policy.iter().sum::<f32>();
        for (actual, expected) in sample.policy.iter().zip(expected_policy) {
            assert!((actual - expected / expected_total).abs() < 1e-6);
        }
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
    fn sampled_move_metadata_tracks_best_and_played_moves() {
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

        let meta = move_search_meta(&candidates, moves[1], 0.2, 3, 99, 7);
        assert_eq!(meta.generation_update, 3);
        assert_eq!(meta.game_id, 99);
        assert_eq!(meta.ply, 7);
        assert_eq!(meta.best_index, 0);
        assert_eq!(meta.played_index, 1);
        assert_eq!(meta.best_q, 0.7);
        assert_eq!(meta.played_q, 0.35);
    }
}
