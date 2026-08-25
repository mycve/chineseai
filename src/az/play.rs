use std::sync::Arc;

use rayon::prelude::*;

use crate::nnue::{
    canonical_move, extract_sparse_features_az, mirror_file_move,
    mirror_sparse_features_az_canonical_file,
};
use crate::xiangqi::{Color, Move, Position, RuleDrawReason, RuleHistoryEntry, RuleOutcome};

use super::alphazero::{
    AzBatchSearchInput, AzBatchSearchWorkspace, AzSearchWorkspace, alphazero_search_batch4_reusing,
    alphazero_search_with_rules_reusing,
};
use super::{
    AzCandidate, AzLoopConfig, AzNnue, AzSampleMeta, AzSearchLimits, AzStartSnapshot,
    AzStartSource, AzTrainingSample, SplitMix64, alphazero_search_with_rules, dense_move_index,
    rule_context_features, scalar_value_to_wdl_target,
};

#[derive(Clone, Copy, Debug, Default)]
pub struct AzTerminalStats {
    pub no_legal_moves: usize,
    pub red_general_missing: usize,
    pub black_general_missing: usize,
    pub rule_draw: usize,
    pub rule_draw_natural_limit: usize,
    pub rule_draw_insufficient_material: usize,
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
        self.rule_draw_natural_limit += other.rule_draw_natural_limit;
        self.rule_draw_insufficient_material += other.rule_draw_insufficient_material;
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
    /// 每个开局交换红黑后的候选平均得分矩，用于消除开局先后手偏置。
    pub paired_openings: usize,
    pub paired_score_sum: f32,
    pub paired_score_sq_sum: f32,
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
        self.paired_openings += other.paired_openings;
        self.paired_score_sum += other.paired_score_sum;
        self.paired_score_sq_sum += other.paired_score_sq_sum;
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
        if self.paired_openings > 1 && self.paired_openings * 2 == self.total_games() {
            let count = self.paired_openings as f32;
            let mean = self.paired_score_sum / count;
            let sample_variance =
                ((self.paired_score_sq_sum - count * mean * mean) / (count - 1.0)).max(0.0);
            return (sample_variance / count).sqrt();
        }
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

    pub fn score_rate_upper_bound(&self, z: f32) -> f32 {
        self.score_rate() + z.max(0.0) * self.score_rate_standard_error()
    }

    pub fn elo_diff_vs_even(&self) -> f32 {
        let total = self.total_games();
        if total == 0 {
            return 0.0;
        }
        score_rate_to_elo(self.score() / total as f32)
    }

    pub fn elo_diff_bounds(&self, z: f32) -> (f32, f32) {
        (
            score_rate_to_elo(self.score_rate_lower_bound(z)),
            score_rate_to_elo(self.score_rate_upper_bound(z)),
        )
    }
}

fn score_rate_to_elo(score: f32) -> f32 {
    let score = score.clamp(0.0001, 0.9999);
    400.0 * (score / (1.0 - score)).log10()
}

#[derive(Clone, Default)]
pub struct AzSelfplayData {
    pub samples: Vec<AzTrainingSample>,
    pub games: Vec<Vec<AzTrainingSample>>,
    pub position_fens: Vec<String>,
    pub midgame_snapshots: Vec<AzStartSnapshot>,
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
        self.position_fens
            .extend(other.position_fens.iter().cloned());
        self.midgame_snapshots
            .extend(other.midgame_snapshots.iter().cloned());
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
        merged.position_fens.extend(chunk.position_fens);
        merged.midgame_snapshots.extend(chunk.midgame_snapshots);
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

fn selfplay_search_limits(config: &AzLoopConfig, _ply: usize, seed: u64) -> AzSearchLimits {
    AzSearchLimits {
        simulations: config.simulations.max(1),
        seed,
        cpuct: config.cpuct,
        cpuct_at_root: config.cpuct_at_root,
        cpuct_base: config.cpuct_base,
        cpuct_factor: config.cpuct_factor,
        cpuct_base_at_root: config.cpuct_base_at_root,
        cpuct_factor_at_root: config.cpuct_factor_at_root,
        max_depth: 0,
        root_dirichlet_alpha: config.root_dirichlet_alpha,
        root_exploration_fraction: config.root_exploration_fraction,
        fpu_value: config.fpu_value,
        fpu_value_at_root: config.fpu_value_at_root,
        policy_softmax_temp: config.policy_softmax_temp,
        draw_score: config.draw_score,
        value_scale: 1.0,
    }
}

fn configure_selfplay_rules(mut position: Position, config: &AzLoopConfig) -> Position {
    position.set_rule60_max_ply(config.rule60_max_ply);
    position
}

const OPENING_FEN_PHASE_PLY: usize = 8;

struct SelfplayStart {
    position: Position,
    rule_history: Vec<RuleHistoryEntry>,
    phase_ply: usize,
    harvest_ply: Option<usize>,
    source: AzStartSource,
}

fn choose_harvest_ply(rng: &mut SplitMix64, phase_ply: usize, max_plies: usize) -> Option<usize> {
    let roll = rng.unit_f32();
    let (start, width) = if roll < 0.50 {
        (30usize, 30usize)
    } else if roll < 0.85 {
        (60, 40)
    } else {
        (100, 40)
    };
    let target = start + (rng.next_u64() as usize % width);
    (target >= phase_ply && target < max_plies).then_some(target)
}

fn choose_selfplay_start(config: &AzLoopConfig, rng: &mut SplitMix64) -> SelfplayStart {
    let roll = rng.unit_f32();
    let configured_midgame_fraction = config.midgame_start_fraction.clamp(0.0, 1.0);
    let pool_empty = config.midgame_positions.is_empty();
    let midgame_fraction = if pool_empty {
        0.0
    } else {
        configured_midgame_fraction
    };
    if roll < midgame_fraction {
        let index = rng.next_u64() as usize % config.midgame_positions.len();
        let snapshot = &config.midgame_positions[index];
        return SelfplayStart {
            position: configure_selfplay_rules(snapshot.position.clone(), config),
            rule_history: snapshot.rule_history.clone(),
            phase_ply: snapshot.phase_ply as usize,
            harvest_ply: None,
            source: AzStartSource::Midgame,
        };
    }
    let use_opening = use_opening_fen(
        !config.opening_positions.is_empty(),
        config.opening_fen_game_fraction
            + if pool_empty {
                configured_midgame_fraction
            } else {
                0.0
            },
        roll - midgame_fraction,
    );
    let (position, phase_ply, source) = if use_opening {
        let index = rng.next_u64() as usize % config.opening_positions.len();
        (
            config.opening_positions[index].clone(),
            OPENING_FEN_PHASE_PLY,
            AzStartSource::OpeningFen,
        )
    } else {
        (Position::startpos(), 0, AzStartSource::Startpos)
    };
    let position = configure_selfplay_rules(position, config);
    let rule_history = position.initial_rule_history();
    SelfplayStart {
        position,
        rule_history,
        phase_ply,
        harvest_ply: choose_harvest_ply(rng, phase_ply, config.max_plies),
        source,
    }
}

fn generate_selfplay_chunk(model: &AzNnue, config: &AzLoopConfig) -> AzSelfplayData {
    crate::scope_profile!("az.selfplay.chunk");
    if config.games >= 4 {
        generate_selfplay_chunk_batch4(model, config)
    } else {
        generate_selfplay_chunk_scalar(model, config)
    }
}

fn generate_selfplay_chunk_scalar(model: &AzNnue, config: &AzLoopConfig) -> AzSelfplayData {
    crate::scope_profile!("az.selfplay.chunk_scalar");
    let mut rng = SplitMix64::new(config.seed);
    let mut samples = Vec::new();
    let mut position_fens = Vec::new();
    let mut midgame_snapshots = Vec::new();
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
    let mut search_workspace = AzSearchWorkspace::new(model);

    for game_index in 0..config.games {
        let start = choose_selfplay_start(config, &mut rng);
        let mut position = start.position;
        let mut rule_history = start.rule_history;
        let start_phase_ply = start.phase_ply;
        let harvest_ply = start.harvest_ply;
        let start_source = start.source;
        let mut game_samples = Vec::new();
        let mut game_bootstrap_wdls = Vec::new();
        let mut result = None;
        let mut plies = 0usize;
        let allow_resign = rng.unit_f32() * 100.0 >= config.resign_playthrough;

        for local_ply in 0..config.max_plies.saturating_sub(start_phase_ply) {
            let ply = start_phase_ply + local_ply;
            plies = local_ply + 1;
            if harvest_ply == Some(ply) {
                midgame_snapshots.push(AzStartSnapshot {
                    position: position.clone(),
                    rule_history: rule_history.clone(),
                    phase_ply: ply.min(u16::MAX as usize) as u16,
                    generation: config.generation_update,
                });
            }
            let legal = {
                crate::scope_profile!("az.selfplay.root_legal_moves");
                position
                    .legal_moves_with_rules_and_repetition(&rule_history)
                    .into_iter()
                    .map(|(mv, _)| mv)
                    .collect::<Vec<_>>()
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
            let limits = selfplay_search_limits(
                config,
                ply,
                rng.next_u64() ^ ((game_index as u64) << 32) ^ ply as u64,
            );
            let search = {
                crate::scope_profile!("az.selfplay.search");
                alphazero_search_with_rules_reusing(
                    &position,
                    &rule_history,
                    legal,
                    model,
                    limits,
                    &mut search_workspace,
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
                let mut meta = root_search_meta(
                    &search.candidates,
                    search.value_q,
                    config.generation_update,
                    config.seed ^ game_index as u64,
                    ply,
                );
                meta.start_source = start_source;
                let sample = make_training_sample(
                    &position,
                    &rule_history,
                    &search.candidates,
                    search.value_q,
                    rng.unit_f32() < config.mirror_probability.clamp(0.0, 1.0),
                    meta,
                    search_simulation_count,
                    1.0,
                );
                game_samples.push(sample);
                game_bootstrap_wdls.push(search.network_value_wdl);
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
            let mv_opt = if temperature <= 1e-6 {
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
            let mut move_meta = move_search_meta(
                &search.candidates,
                mv,
                search.value_q,
                config.generation_update,
                config.seed ^ game_index as u64,
                ply,
            );
            move_meta.start_source = start_source;
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
                if config.record_fens {
                    position_fens.push(position.to_fen_with_history(&rule_history));
                }
                let sample = make_training_sample(
                    &position,
                    &rule_history,
                    &search.candidates,
                    search.value_q,
                    rng.unit_f32() < config.mirror_probability.clamp(0.0, 1.0),
                    move_meta,
                    search_simulation_count,
                    1.0,
                );
                game_samples.push(sample);
                game_bootstrap_wdls.push(search.network_value_wdl);
            }
            let mover = position.side_to_move();
            let captured = position.piece_at(mv.to as usize);
            position.make_move(mv);
            rule_history.push(position.rule_history_entry_after_moved(mover, mv, captured));

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
                            RuleDrawReason::NaturalMoveLimit => {
                                terminal.rule_draw_natural_limit += 1
                            }
                            RuleDrawReason::InsufficientMaterial => {
                                terminal.rule_draw_insufficient_material += 1
                            }
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
            assign_td_lambda_value_targets(
                &mut game_samples,
                &game_bootstrap_wdls,
                result,
                config.value_td_lambda,
            );
        }
        samples.extend(game_samples.clone());
        games.push(game_samples);
    }

    AzSelfplayData {
        samples,
        games,
        position_fens,
        midgame_snapshots,
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

struct BatchedSelfplayGame {
    game_index: usize,
    position: Position,
    rule_history: Vec<RuleHistoryEntry>,
    samples: Vec<AzTrainingSample>,
    bootstrap_wdls: Vec<[f32; 3]>,
    result: Option<f32>,
    ply: usize,
    phase_ply: usize,
    harvest_ply: Option<usize>,
    reported_plies: usize,
    allow_resign: bool,
    start_source: AzStartSource,
    rng: SplitMix64,
}

fn new_batched_selfplay_game(game_index: usize, config: &AzLoopConfig) -> BatchedSelfplayGame {
    let mut rng =
        SplitMix64::new(config.seed ^ (game_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let start = choose_selfplay_start(config, &mut rng);
    let allow_resign = rng.unit_f32() * 100.0 >= config.resign_playthrough;
    BatchedSelfplayGame {
        game_index,
        position: start.position,
        rule_history: start.rule_history,
        samples: Vec::new(),
        bootstrap_wdls: Vec::new(),
        result: None,
        ply: 0,
        phase_ply: start.phase_ply,
        harvest_ply: start.harvest_ply,
        reported_plies: 0,
        allow_resign,
        start_source: start.source,
        rng,
    }
}

fn record_batched_search_stats(
    data: &mut AzSelfplayData,
    search: &super::AzSearchResult,
    ply: usize,
    config: &AzLoopConfig,
) {
    let entropy = policy_entropy(&search.candidates);
    let shape = policy_shape_stats(&search.candidates);
    data.raw_prior_top1_sum += shape.raw_prior_top1;
    data.raw_prior_top2_sum += shape.raw_prior_top2;
    data.policy_top1_sum += shape.policy_top1;
    data.policy_top2_sum += shape.policy_top2;
    data.q_gap_sum += shape.q_gap;
    data.q_top1_abs_sum += shape.q_top1_abs;
    data.visited_actions_sum += shape.visited_actions;
    data.shape_count += 1;
    data.entropy_all_sum += entropy;
    data.entropy_all_count += 1;
    if ply < temperature_opening_plies(config) {
        data.entropy_opening_sum += entropy;
        data.entropy_opening_count += 1;
        data.opening_raw_prior_top1_sum += shape.raw_prior_top1;
        data.opening_raw_prior_top2_sum += shape.raw_prior_top2;
        data.opening_policy_top1_sum += shape.policy_top1;
        data.opening_policy_top2_sum += shape.policy_top2;
        data.opening_q_gap_sum += shape.q_gap;
        data.opening_q_top1_abs_sum += shape.q_top1_abs;
        data.opening_visited_actions_sum += shape.visited_actions;
        data.opening_shape_count += 1;
    } else {
        data.entropy_mid_sum += entropy;
        data.entropy_mid_count += 1;
    }
}

fn inactive_batch_input(config: &AzLoopConfig) -> AzBatchSearchInput {
    let position = configure_selfplay_rules(Position::startpos(), config);
    let rule_history = position.initial_rule_history();
    let root_moves = position.legal_moves();
    let mut limits = selfplay_search_limits(config, 0, 0);
    limits.simulations = 0;
    AzBatchSearchInput {
        position,
        rule_history,
        root_moves,
        limits,
    }
}

fn generate_selfplay_chunk_batch4(model: &AzNnue, config: &AzLoopConfig) -> AzSelfplayData {
    crate::scope_profile!("az.selfplay.chunk_batch4");
    let mut data = AzSelfplayData::default();
    let mut workspace = AzBatchSearchWorkspace::new(model);
    for group_start in (0..config.games).step_by(4) {
        let mut states: [Option<BatchedSelfplayGame>; 4] = std::array::from_fn(|slot| {
            let game_index = group_start + slot;
            (game_index < config.games).then(|| new_batched_selfplay_game(game_index, config))
        });
        loop {
            let mut legal_moves: [Vec<Move>; 4] = std::array::from_fn(|_| Vec::new());
            let mut searched = [false; 4];
            for index in 0..4 {
                let Some(state) = states[index].as_mut() else {
                    continue;
                };
                if state.result.is_some() {
                    continue;
                }
                if state.phase_ply >= config.max_plies {
                    state.result = Some(0.0);
                    data.terminal.max_plies += 1;
                    continue;
                }
                state.reported_plies = state.ply + 1;
                if state.harvest_ply == Some(state.phase_ply) {
                    data.midgame_snapshots.push(AzStartSnapshot {
                        position: state.position.clone(),
                        rule_history: state.rule_history.clone(),
                        phase_ply: state.phase_ply.min(u16::MAX as usize) as u16,
                        generation: config.generation_update,
                    });
                    state.harvest_ply = None;
                }
                legal_moves[index] = state
                    .position
                    .legal_moves_with_rules_and_repetition(&state.rule_history)
                    .into_iter()
                    .map(|(mv, _)| mv)
                    .collect();
                if legal_moves[index].is_empty() {
                    state.result = Some(if state.position.side_to_move() == Color::Red {
                        -1.0
                    } else {
                        1.0
                    });
                    data.terminal.no_legal_moves += 1;
                } else {
                    searched[index] = true;
                }
            }
            if !searched.iter().any(|&active| active) {
                break;
            }
            let inputs = std::array::from_fn(|index| {
                if !searched[index] {
                    return inactive_batch_input(config);
                }
                let state = states[index].as_mut().unwrap();
                let seed = state.rng.next_u64()
                    ^ ((state.game_index as u64) << 32)
                    ^ state.phase_ply as u64;
                AzBatchSearchInput {
                    position: state.position.clone(),
                    rule_history: state.rule_history.clone(),
                    root_moves: std::mem::take(&mut legal_moves[index]),
                    limits: selfplay_search_limits(config, state.phase_ply, seed),
                }
            });
            let searches = alphazero_search_batch4_reusing(inputs, model, &mut workspace);
            for index in 0..4 {
                if !searched[index] {
                    continue;
                }
                let state = states[index].as_mut().unwrap();
                let search = &searches[index];
                data.search_simulations.searches += 1;
                data.search_simulations.simulations_sum += search.simulations;
                record_batched_search_stats(&mut data, search, state.phase_ply, config);
                if state.allow_resign && should_resign(search.value_q, config) {
                    let mut meta = root_search_meta(
                        &search.candidates,
                        search.value_q,
                        config.generation_update,
                        config.seed ^ state.game_index as u64,
                        state.phase_ply,
                    );
                    meta.start_source = state.start_source;
                    state.samples.push(make_training_sample(
                        &state.position,
                        &state.rule_history,
                        &search.candidates,
                        search.value_q,
                        state.rng.unit_f32() < config.mirror_probability.clamp(0.0, 1.0),
                        meta,
                        search.simulations,
                        1.0,
                    ));
                    state.bootstrap_wdls.push(search.network_value_wdl);
                    state.result = Some(if state.position.side_to_move() == Color::Red {
                        data.terminal.resign_red += 1;
                        -1.0
                    } else {
                        data.terminal.resign_black += 1;
                        1.0
                    });
                    continue;
                }
                let temperature = temperature_for_ply(config, state.phase_ply);
                let mv = if temperature <= 1e-6 {
                    search.best_move.or_else(|| {
                        choose_selfplay_move(
                            &search.candidates,
                            temperature,
                            0.0,
                            0.0,
                            &mut state.rng,
                        )
                    })
                } else {
                    choose_selfplay_move(
                        &search.candidates,
                        temperature,
                        config.temperature_value_cutoff,
                        config.temperature_visit_offset,
                        &mut state.rng,
                    )
                };
                let Some(mv) = mv else {
                    state.result = Some(0.0);
                    continue;
                };
                let mut meta = move_search_meta(
                    &search.candidates,
                    mv,
                    search.value_q,
                    config.generation_update,
                    config.seed ^ state.game_index as u64,
                    state.phase_ply,
                );
                meta.start_source = state.start_source;
                data.sampled_moves += 1;
                data.sampled_best_moves += usize::from(meta.best_index == meta.played_index);
                data.best_played_q_gap_sum += (meta.best_q - meta.played_q).max(0.0);
                let top_visits = search
                    .candidates
                    .iter()
                    .map(|candidate| candidate.visits)
                    .max()
                    .unwrap_or(0);
                data.played_top_visit_ratio_sum += if top_visits == 0 {
                    0.0
                } else {
                    meta.played_visits as f32 / top_visits as f32
                };
                data.best_q_sum += meta.best_q;
                data.played_q_sum += meta.played_q;
                if config.record_fens {
                    data.position_fens
                        .push(state.position.to_fen_with_history(&state.rule_history));
                }
                state.samples.push(make_training_sample(
                    &state.position,
                    &state.rule_history,
                    &search.candidates,
                    search.value_q,
                    state.rng.unit_f32() < config.mirror_probability.clamp(0.0, 1.0),
                    meta,
                    search.simulations,
                    1.0,
                ));
                state.bootstrap_wdls.push(search.network_value_wdl);
                let mover = state.position.side_to_move();
                let captured = state.position.piece_at(mv.to as usize);
                state.position.make_move(mv);
                state.rule_history.push(
                    state
                        .position
                        .rule_history_entry_after_moved(mover, mv, captured),
                );
                state.ply += 1;
                state.phase_ply += 1;
                if !state.position.has_general(Color::Red) {
                    state.result = Some(-1.0);
                    data.terminal.red_general_missing += 1;
                    continue;
                }
                if !state.position.has_general(Color::Black) {
                    state.result = Some(1.0);
                    data.terminal.black_general_missing += 1;
                    continue;
                }
                if let Some(outcome) = state
                    .position
                    .rule_outcome_with_history(&state.rule_history)
                {
                    state.result = Some(match outcome {
                        RuleOutcome::Draw(_) => 0.0,
                        RuleOutcome::Win(Color::Red) => 1.0,
                        RuleOutcome::Win(Color::Black) => -1.0,
                    });
                    match outcome {
                        RuleOutcome::Draw(reason) => {
                            data.terminal.rule_draw += 1;
                            match reason {
                                RuleDrawReason::NaturalMoveLimit => {
                                    data.terminal.rule_draw_natural_limit += 1
                                }
                                RuleDrawReason::InsufficientMaterial => {
                                    data.terminal.rule_draw_insufficient_material += 1
                                }
                                RuleDrawReason::Repetition => {
                                    data.terminal.rule_draw_repetition += 1
                                }
                                RuleDrawReason::MutualLongCheck => {
                                    data.terminal.rule_draw_mutual_long_check += 1
                                }
                                RuleDrawReason::MutualLongChase => {
                                    data.terminal.rule_draw_mutual_long_chase += 1
                                }
                            }
                        }
                        RuleOutcome::Win(Color::Red) => data.terminal.rule_win_red += 1,
                        RuleOutcome::Win(Color::Black) => data.terminal.rule_win_black += 1,
                    }
                }
            }
        }
        for state in states.into_iter().flatten() {
            let result = state.result.unwrap_or(0.0);
            match result.total_cmp(&0.0) {
                std::cmp::Ordering::Greater => data.red_wins += 1,
                std::cmp::Ordering::Less => data.black_wins += 1,
                std::cmp::Ordering::Equal => data.draws += 1,
            }
            data.plies_total += state.reported_plies;
            let mut game_samples = state.samples;
            assign_td_lambda_value_targets(
                &mut game_samples,
                &state.bootstrap_wdls,
                result,
                config.value_td_lambda,
            );
            data.samples.extend(game_samples.iter().cloned());
            data.games.push(game_samples);
        }
    }
    data
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
    rule_history: &[RuleHistoryEntry],
    candidates: &[AzCandidate],
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
        rule_context: rule_context_features(position, rule_history),
        move_indices,
        policy,
        value_wdl: scalar_value_to_wdl_target(value),
        value: value.clamp(-1.0, 1.0),
        side_sign,
        policy_weight: policy_weight.max(0.0),
        value_weight: 1.0,
        search_simulations: search_simulations.min(u32::MAX as usize) as u32,
        meta,
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

fn assign_td_lambda_value_targets(
    samples: &mut [AzTrainingSample],
    bootstrap_wdls: &[[f32; 3]],
    game_result_red: f32,
    td_lambda: f32,
) {
    assert_eq!(samples.len(), bootstrap_wdls.len());
    let Some(last) = samples.last_mut() else {
        return;
    };
    let lambda = td_lambda.clamp(0.0, 1.0);
    let terminal = scalar_value_to_wdl_target((game_result_red * last.side_sign).clamp(-1.0, 1.0));
    last.value_wdl = terminal;
    last.value = terminal[0] - terminal[2];
    let mut next_target = terminal;
    for index in (0..samples.len().saturating_sub(1)).rev() {
        let bootstrap = flip_wdl(bootstrap_wdls[index + 1]);
        let continuation = flip_wdl(next_target);
        let target = std::array::from_fn(|part| {
            (1.0 - lambda) * bootstrap[part] + lambda * continuation[part]
        });
        samples[index].value_wdl = target;
        samples[index].value = target[0] - target[2];
        next_target = target;
    }
}

fn flip_wdl(wdl: [f32; 3]) -> [f32; 3] {
    [wdl[2], wdl[1], wdl[0]]
}

fn should_resign(root_q: f32, config: &AzLoopConfig) -> bool {
    if config.resign_percentage <= 0.0 {
        return false;
    }
    let threshold = -(1.0 - config.resign_percentage.clamp(0.0, 100.0) / 100.0);
    root_q <= threshold
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

fn use_opening_fen(openings_available: bool, fraction: f32, random_unit: f32) -> bool {
    openings_available && random_unit < fraction.clamp(0.0, 1.0)
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
    let cutoff_anchor_q = candidates
        .iter()
        .max_by(|left, right| {
            (left.visits as f32 + visit_offset).total_cmp(&(right.visits as f32 + visit_offset))
        })
        .map(|candidate| candidate.q);
    let inv_temperature = 1.0 / temperature.max(1e-3);
    let mut weights = candidates
        .iter()
        .map(|candidate| {
            (candidate.visits as f32 + visit_offset)
                .max(1e-9)
                .powf(inv_temperature)
        })
        .collect::<Vec<_>>();

    let Some(cutoff_anchor_q) = cutoff_anchor_q else {
        return weights;
    };
    if value_cutoff <= 0.0 || value_cutoff >= 1.0 || !cutoff_anchor_q.is_finite() {
        return weights;
    }

    // 配置值表示胜率差；Q=W-L，因此胜率差 value_cutoff 对应 2*value_cutoff 的 Q 差。
    let min_q = cutoff_anchor_q - 2.0 * value_cutoff;
    for (weight, candidate) in weights.iter_mut().zip(candidates) {
        if candidate.q < min_q {
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
    pub rule60_max_ply: Option<u16>,
    pub games_as_red: usize,
    pub games_as_black: usize,
    pub start_index: usize,
    pub seed: u64,
    pub cpuct: f32,
    pub cpuct_at_root: f32,
    pub cpuct_base: f32,
    pub cpuct_factor: f32,
    pub cpuct_base_at_root: f32,
    pub cpuct_factor_at_root: f32,
    pub fpu_value: f32,
    pub fpu_value_at_root: f32,
    pub draw_score: f32,
    pub policy_softmax_temp: f32,
}

pub fn play_arena_games_from_positions(
    candidate: &AzNnue,
    baseline: &AzNnue,
    positions: &[Position],
    config: AzArenaConfig,
) -> AzArenaReport {
    let mut report = AzArenaReport::default();
    let mut red_scores = Vec::with_capacity(config.games_as_red);
    for game_index in 0..config.games_as_red {
        let mut position = arena_start_position(positions, config.start_index + game_index);
        position.set_rule60_max_ply(config.rule60_max_ply);
        let outcome = play_arena_game(
            &position,
            candidate,
            baseline,
            config.simulations,
            config.max_plies,
            config.seed ^ (config.start_index + game_index) as u64,
            config.cpuct,
            config.cpuct_at_root,
            config.cpuct_base,
            config.cpuct_factor,
            config.cpuct_base_at_root,
            config.cpuct_factor_at_root,
            config.fpu_value,
            config.fpu_value_at_root,
            config.draw_score,
            config.policy_softmax_temp,
        );
        match outcome.total_cmp(&0.0) {
            std::cmp::Ordering::Greater => {
                report.wins += 1;
                report.wins_as_red += 1;
                red_scores.push(1.0);
            }
            std::cmp::Ordering::Less => {
                report.losses += 1;
                report.losses_as_red += 1;
                red_scores.push(0.0);
            }
            std::cmp::Ordering::Equal => {
                report.draws += 1;
                red_scores.push(0.5);
            }
        }
    }
    for game_index in 0..config.games_as_black {
        let mut position = arena_start_position(positions, config.start_index + game_index);
        position.set_rule60_max_ply(config.rule60_max_ply);
        let outcome = play_arena_game(
            &position,
            baseline,
            candidate,
            config.simulations,
            config.max_plies,
            config.seed ^ (config.start_index + game_index) as u64,
            config.cpuct,
            config.cpuct_at_root,
            config.cpuct_base,
            config.cpuct_factor,
            config.cpuct_base_at_root,
            config.cpuct_factor_at_root,
            config.fpu_value,
            config.fpu_value_at_root,
            config.draw_score,
            config.policy_softmax_temp,
        );
        let black_score = match outcome.total_cmp(&0.0) {
            std::cmp::Ordering::Greater => {
                report.losses += 1;
                report.losses_as_black += 1;
                0.0
            }
            std::cmp::Ordering::Less => {
                report.wins += 1;
                report.wins_as_black += 1;
                1.0
            }
            std::cmp::Ordering::Equal => {
                report.draws += 1;
                0.5
            }
        };
        if let Some(&red_score) = red_scores.get(game_index) {
            let paired_score = 0.5 * (red_score + black_score);
            report.paired_openings += 1;
            report.paired_score_sum += paired_score;
            report.paired_score_sq_sum += paired_score * paired_score;
        }
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
    cpuct_at_root: f32,
    cpuct_base: f32,
    cpuct_factor: f32,
    cpuct_base_at_root: f32,
    cpuct_factor_at_root: f32,
    fpu_value: f32,
    fpu_value_at_root: f32,
    draw_score: f32,
    policy_softmax_temp: f32,
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
        let result = alphazero_search_with_rules(
            &position,
            Some(rule_history.clone()),
            Some(legal),
            model,
            AzSearchLimits {
                simulations,
                seed: seed ^ ((ply as u64) << 32),
                cpuct,
                cpuct_at_root,
                cpuct_base,
                cpuct_factor,
                cpuct_base_at_root,
                cpuct_factor_at_root,
                max_depth: 0,
                root_dirichlet_alpha: 0.0,
                root_exploration_fraction: 0.0,
                fpu_value,
                fpu_value_at_root,
                policy_softmax_temp,
                draw_score,
                value_scale: 1.0,
            },
        );
        let Some(mv) = result.best_move else {
            return 0.0;
        };
        let mover = position.side_to_move();
        let captured = position.piece_at(mv.to as usize);
        position.make_move(mv);
        rule_history.push(position.rule_history_entry_after_moved(mover, mv, captured));

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

    fn selfplay_test_config(games: usize) -> AzLoopConfig {
        AzLoopConfig {
            games,
            max_plies: 12,
            rule60_max_ply: Some(120),
            simulations: 64,
            seed: 20260817,
            workers: 1,
            generation_update: 0,
            temperature_start: 0.0,
            temperature_endgame: 0.0,
            temperature_decay_delay_plies: 0,
            temperature_decay_plies: 0,
            temperature_value_cutoff: 0.0,
            temperature_visit_offset: 0.0,
            cpuct: 0.65,
            cpuct_at_root: 1.5,
            cpuct_base: 19652.0,
            cpuct_factor: 1.5,
            cpuct_base_at_root: 19652.0,
            cpuct_factor_at_root: 1.5,
            root_dirichlet_alpha: 0.0,
            root_exploration_fraction: 0.0,
            fpu_value: 0.30,
            fpu_value_at_root: 0.20,
            draw_score: 0.0,
            policy_softmax_temp: 1.0,
            value_td_lambda: 0.9,
            opening_positions: Default::default(),
            opening_fen_game_fraction: 0.0,
            midgame_positions: Default::default(),
            midgame_start_fraction: 0.0,
            resign_percentage: 0.0,
            resign_playthrough: 100.0,
            mirror_probability: 0.0,
            record_fens: false,
        }
    }

    #[test]
    fn arena_uncertainty_uses_color_swapped_opening_pairs() {
        let report = AzArenaReport {
            wins: 2,
            losses: 2,
            paired_openings: 2,
            paired_score_sum: 1.0,
            paired_score_sq_sum: 0.5,
            ..AzArenaReport::default()
        };
        // 两个开局的配对得分为 0.5/0.5；红黑单盘虽各有胜负，先后手抵消后方差为零。
        assert_eq!(report.score_rate(), 0.5);
        assert_eq!(report.score_rate_standard_error(), 0.0);
    }

    #[test]
    fn midgame_start_preserves_phase_and_rule_history_without_reharvesting() {
        let mut config = selfplay_test_config(1);
        config.max_plies = 100;
        config.midgame_start_fraction = 1.0;
        let position = Position::startpos();
        let rule_history = position.initial_rule_history();
        config.midgame_positions = vec![AzStartSnapshot {
            position,
            rule_history: rule_history.clone(),
            phase_ply: 57,
            generation: 9,
        }]
        .into();
        let start = choose_selfplay_start(&config, &mut SplitMix64::new(3));
        assert_eq!(start.position.hash(), rule_history.last().unwrap().hash);
        assert_eq!(start.rule_history, rule_history);
        assert_eq!(start.phase_ply, 57);
        assert_eq!(start.harvest_ply, None);
        assert_eq!(start.source, AzStartSource::Midgame);
    }

    #[test]
    fn start_source_distinguishes_standard_and_opening_positions() {
        let config = selfplay_test_config(1);
        let start = choose_selfplay_start(&config, &mut SplitMix64::new(1));
        assert_eq!(start.source, AzStartSource::Startpos);

        let mut config = selfplay_test_config(1);
        config.opening_positions = vec![
            Position::from_fen("rnbakabnr/9/1c5c1/p1p1p1p1p/9/4P4/P1P3P1P/1C5C1/9/RNBAKABNR b")
                .unwrap(),
        ]
        .into();
        config.opening_fen_game_fraction = 1.0;
        let start = choose_selfplay_start(&config, &mut SplitMix64::new(1));
        assert_eq!(start.source, AzStartSource::OpeningFen);
    }

    #[test]
    fn empty_midgame_pool_falls_back_to_opening_share() {
        let mut config = selfplay_test_config(1);
        config.opening_positions = vec![
            Position::from_fen("rnbakabnr/9/1c5c1/p1p1p1p1p/9/4P4/P1P3P1P/1C5C1/9/RNBAKABNR b")
                .unwrap(),
        ]
        .into();
        config.opening_fen_game_fraction = 0.5;
        config.midgame_start_fraction = 0.3;
        let opening_hash = config.opening_positions[0].hash();
        let openings = (0..10_000)
            .filter(|seed| {
                choose_selfplay_start(&config, &mut SplitMix64::new(*seed))
                    .position
                    .hash()
                    == opening_hash
            })
            .count();
        assert!((7_700..8_300).contains(&openings));
    }

    #[test]
    fn batch4_selfplay_produces_complete_games_and_samples() {
        let model = AzNnue::random(32, 41);
        let config = selfplay_test_config(5);
        let data = generate_selfplay_chunk_batch4(&model, &config);
        assert_eq!(data.games.len(), 5);
        assert_eq!(data.red_wins + data.black_wins + data.draws, 5);
        assert_eq!(
            data.samples.len(),
            data.games.iter().map(Vec::len).sum::<usize>()
        );
        assert_eq!(data.search_simulations.searches, data.samples.len());
        assert!(
            data.games
                .iter()
                .all(|game| !game.is_empty() && game.len() <= config.max_plies)
        );
    }

    #[test]
    #[ignore = "manual fast-profile end-to-end selfplay benchmark"]
    fn benchmark_batch4_selfplay() {
        use std::time::Instant;

        let model = AzNnue::random(128, 41);
        let mut config = selfplay_test_config(8);
        config.max_plies = 24;
        config.simulations = 256;

        let scalar_started = Instant::now();
        let scalar = generate_selfplay_chunk_scalar(&model, &config);
        let scalar_elapsed = scalar_started.elapsed();
        let batch_started = Instant::now();
        let batch = generate_selfplay_chunk_batch4(&model, &config);
        let batch_elapsed = batch_started.elapsed();

        eprintln!(
            "selfplay: scalar={:.3}ms batch4={:.3}ms speedup={:.3}x scalar_sims/s={:.0} batch_sims/s={:.0}",
            scalar_elapsed.as_secs_f64() * 1e3,
            batch_elapsed.as_secs_f64() * 1e3,
            scalar_elapsed.as_secs_f64() / batch_elapsed.as_secs_f64(),
            scalar.search_simulations.simulations_sum as f64 / scalar_elapsed.as_secs_f64(),
            batch.search_simulations.simulations_sum as f64 / batch_elapsed.as_secs_f64(),
        );
        assert!(scalar.search_simulations.searches > 0);
        assert!(batch.search_simulations.searches > 0);
        crate::profile::print_report();
    }

    #[test]
    #[ignore = "manual fast-profile multi-worker selfplay benchmark"]
    fn benchmark_batch4_selfplay_workers() {
        use std::time::Instant;

        let workers = 4;
        let model = AzNnue::random(128, 41);
        let mut config = selfplay_test_config(4);
        config.max_plies = 8;
        config.simulations = 1600;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        let scalar_started = Instant::now();
        let scalar_sims = pool.install(|| {
            (0..workers)
                .into_par_iter()
                .map(|worker| {
                    let mut worker_config = config.clone();
                    worker_config.seed ^= worker as u64;
                    generate_selfplay_chunk_scalar(&model, &worker_config)
                        .search_simulations
                        .simulations_sum
                })
                .sum::<usize>()
        });
        let scalar_elapsed = scalar_started.elapsed();
        let batch_started = Instant::now();
        let batch_sims = pool.install(|| {
            (0..workers)
                .into_par_iter()
                .map(|worker| {
                    let mut worker_config = config.clone();
                    worker_config.seed ^= worker as u64;
                    generate_selfplay_chunk_batch4(&model, &worker_config)
                        .search_simulations
                        .simulations_sum
                })
                .sum::<usize>()
        });
        let batch_elapsed = batch_started.elapsed();
        eprintln!(
            "selfplay-workers={workers}: scalar={:.3}ms batch4={:.3}ms speedup={:.3}x scalar_sims/s={:.0} batch_sims/s={:.0}",
            scalar_elapsed.as_secs_f64() * 1e3,
            batch_elapsed.as_secs_f64() * 1e3,
            scalar_elapsed.as_secs_f64() / batch_elapsed.as_secs_f64(),
            scalar_sims as f64 / scalar_elapsed.as_secs_f64(),
            batch_sims as f64 / batch_elapsed.as_secs_f64(),
        );
    }

    #[test]
    fn opening_fen_sampling_respects_configured_fraction() {
        assert!(use_opening_fen(true, 0.75, 0.0));
        assert!(use_opening_fen(true, 0.75, 0.749_999));
        assert!(!use_opening_fen(true, 0.75, 0.75));
        assert!(!use_opening_fen(true, 0.75, 0.999_999));
        assert!(!use_opening_fen(false, 0.75, 0.0));
    }

    fn candidate(mv: Move, policy: f32) -> AzCandidate {
        AzCandidate {
            mv,
            visits: (policy * 100.0) as u32,
            q: 0.0,
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
            raw_prior: 0.0,
            prior: 0.0,
            policy: 0.0,
        }
    }

    fn sample(value: f32, side_sign: f32) -> AzTrainingSample {
        AzTrainingSample {
            features: Vec::new(),
            rule_context: [0.0; crate::az::RULE_CONTEXT_SIZE],
            move_indices: Vec::new(),
            policy: Vec::new(),
            value_wdl: scalar_value_to_wdl_target(value),
            value,
            side_sign,
            policy_weight: 1.0,
            value_weight: 1.0,
            search_simulations: 0,
            meta: AzSampleMeta::default(),
        }
    }

    #[test]
    fn td_lambda_one_is_terminal_mc() {
        let mut samples = [sample(0.0, 1.0), sample(0.0, -1.0)];
        samples[0].meta.root_q = -1.0;
        samples[1].meta.root_q = 1.0;

        assign_td_lambda_value_targets(&mut samples, &[[0.2, 0.3, 0.5], [0.6, 0.2, 0.2]], 1.0, 1.0);

        assert_eq!(samples[0].value_wdl, [1.0, 0.0, 0.0]);
        assert_eq!(samples[0].value, 1.0);
        assert_eq!(samples[1].value_wdl, [0.0, 0.0, 1.0]);
        assert_eq!(samples[1].value, -1.0);
    }

    #[test]
    fn td_lambda_mixes_wdl_bootstrap_and_terminal_return() {
        let mut samples = [sample(0.0, 1.0), sample(0.0, -1.0), sample(0.0, 1.0)];
        let bootstraps = [[0.4, 0.4, 0.2], [0.1, 0.6, 0.3], [0.6, 0.2, 0.2]];

        assign_td_lambda_value_targets(&mut samples, &bootstraps, 1.0, 0.9);

        let expected = [[0.894, 0.078, 0.028], [0.02, 0.02, 0.96], [1.0, 0.0, 0.0]];
        for (sample, expected) in samples.iter().zip(expected) {
            for (actual, expected) in sample.value_wdl.iter().zip(expected) {
                assert!((actual - expected).abs() < 1.0e-6);
            }
            assert!((sample.value - (expected[0] - expected[2])).abs() < 1.0e-6);
        }
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
            &position.initial_rule_history(),
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
    fn negative_visit_offset_is_added_like_lc0() {
        let candidates = vec![
            candidate_q(Move::new(0, 1), 1, 0.0),
            candidate_q(Move::new(0, 2), 10, 0.0),
        ];

        let weights = temperature_move_weights(&candidates, 1.0, 0.0, -0.8);

        assert!((weights[0] - 0.2).abs() < 1e-6);
        assert!((weights[1] - 9.2).abs() < 1e-6);
    }

    #[test]
    fn temperature_value_cutoff_is_anchored_to_most_visited_move() {
        let candidates = vec![
            candidate_q(Move::new(0, 1), 100, 0.40),
            candidate_q(Move::new(0, 2), 5, 0.90),
            candidate_q(Move::new(0, 3), 10, 0.05),
        ];

        let weights = temperature_move_weights(&candidates, 1.0, 0.15, 0.0);

        assert!(weights[0] > 0.0);
        assert!(weights[1] > 0.0);
        assert_eq!(weights[2], 0.0);
    }
}
