use std::sync::Arc;

use rayon::prelude::*;

use crate::az::nnue::{
    canonical_move, extract_sparse_features_az, mirror_file_move,
    mirror_sparse_features_az_canonical_file,
};
use crate::xiangqi::{Color, Move, Position, RuleDrawReason, RuleHistoryEntry, RuleOutcome};

use super::alphazero::{AzSearchWorkspace, alphazero_search_with_rules_reusing};
use super::{
    AzCandidate, AzLoopConfig, AzNnue, AzSampleMeta, AzSearchLimits, AzStartSnapshot,
    AzStartSource, AzTrainingSample, SplitMix64, alphazero_search_with_rules, dense_move_index,
    normalize_wdl_target, rule_context_features, scalar_value_to_wdl_target,
};

fn outcome_index(result_red: f32) -> usize {
    if result_red > 0.0 {
        0
    } else if result_red < 0.0 {
        2
    } else {
        1
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AzTerminalStats {
    /// 已由搜索证明的提前终局，按红胜、和棋、黑胜排列。
    pub search_proven: [usize; 3],
    pub no_legal_moves: usize,
    pub checkmate: usize,
    pub stalemate: usize,
    pub rule_blocked: usize,
    pub search_no_move: usize,
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
    fn record_no_legal_moves(&mut self, position: &Position) {
        self.no_legal_moves += 1;
        if !position.legal_moves().is_empty() {
            self.rule_blocked += 1;
        } else if position.in_check(position.side_to_move()) {
            self.checkmate += 1;
        } else {
            self.stalemate += 1;
        }
    }

    pub fn add_assign(&mut self, other: &Self) {
        for (count, other_count) in self.search_proven.iter_mut().zip(other.search_proven) {
            *count += other_count;
        }
        self.no_legal_moves += other.no_legal_moves;
        self.checkmate += other.checkmate;
        self.stalemate += other.stalemate;
        self.rule_blocked += other.rule_blocked;
        self.search_no_move += other.search_no_move;
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
    pub red_wins: usize,
    pub black_wins: usize,
    pub draws: usize,
    pub plies_total: usize,
    pub start_games: [usize; AzStartSource::COUNT],
    pub start_phase_ply_sum: [u64; AzStartSource::COUNT],
    pub start_age_sum: [u64; AzStartSource::COUNT],
    pub start_age_max: [u32; AzStartSource::COUNT],
    pub start_temperature_sum: [f32; AzStartSource::COUNT],
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
        self.red_wins += other.red_wins;
        self.black_wins += other.black_wins;
        self.draws += other.draws;
        self.plies_total += other.plies_total;
        for source in 0..AzStartSource::COUNT {
            self.start_games[source] += other.start_games[source];
            self.start_phase_ply_sum[source] += other.start_phase_ply_sum[source];
            self.start_age_sum[source] += other.start_age_sum[source];
            self.start_age_max[source] =
                self.start_age_max[source].max(other.start_age_max[source]);
            self.start_temperature_sum[source] += other.start_temperature_sum[source];
        }
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
    // 连杀预算挂在模型上（`AzNnue::mate_search_plies`）。这里统一打一次补丁，让单线程与
    // 多线程两条路径共用同一份设置；两边本来就一样时不克隆（关闭态零代价）。
    let patched = (model.mate_search_plies != config.mate_search_plies
        || model.tactical_search_nodes != config.tactical_search_nodes
        || model.tactical_search_plies != config.tactical_search_plies
        || model.tactical_quiet_plies != config.tactical_quiet_plies)
        .then(|| {
            let mut patched = model.clone();
            patched.mate_search_plies = config.mate_search_plies;
            patched.tactical_search_nodes = config.tactical_search_nodes;
            patched.tactical_search_plies = config.tactical_search_plies;
            patched.tactical_quiet_plies = config.tactical_quiet_plies;
            patched
        });
    let model = patched.as_ref().unwrap_or(model);
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
                if !config.opening_positions.is_empty() {
                    let offset =
                        worker * (config.games / workers) + worker.min(config.games % workers);
                    worker_config.opening_positions = (offset..offset + games)
                        .map(|i| {
                            config.opening_positions[i % config.opening_positions.len()].clone()
                        })
                        .collect::<Vec<_>>()
                        .into();
                }
                worker_config.workers = 1;
                worker_config.seed ^= (worker as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let chunk = generate_selfplay_chunk(&shared_model, &worker_config);
                crate::infra::profile::flush_thread();
                chunk
            })
            .collect::<Vec<_>>()
    });
    let mut merged = AzSelfplayData::default();
    for chunk in chunks {
        merged.add_assign(&chunk);
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
        fpu_absolute_at_root: config.fpu_absolute_at_root,
        minimum_kldgain_per_node: config.minimum_kldgain_per_node,
        policy_softmax_temp: config.policy_softmax_temp,
        draw_score: config.draw_score,
        value_scale: 1.0,
    }
}

fn configure_selfplay_rules(mut position: Position, config: &AzLoopConfig) -> Position {
    position.set_rule60_max_ply(config.rule60_max_ply);
    position
}

struct SelfplayStart {
    position: Position,
    rule_history: Vec<RuleHistoryEntry>,
    phase_ply: usize,
    source: AzStartSource,
    generation: u32,
}

fn choose_selfplay_start(
    config: &AzLoopConfig,
    _rng: &mut SplitMix64,
    game_index: usize,
) -> SelfplayStart {
    if !config.opening_positions.is_empty() {
        let snapshot = &config.opening_positions[game_index % config.opening_positions.len()];
        return SelfplayStart {
            position: configure_selfplay_rules(snapshot.position.clone(), config),
            rule_history: snapshot.rule_history.clone(),
            phase_ply: snapshot.phase_ply as usize,
            source: AzStartSource::OpeningBook,
            generation: snapshot.generation,
        };
    }
    let position = configure_selfplay_rules(Position::startpos(), config);
    let rule_history = position.initial_rule_history();
    SelfplayStart {
        position,
        rule_history,
        phase_ply: 0,
        source: AzStartSource::Startpos,
        generation: config.generation_update,
    }
}

fn generate_selfplay_chunk(model: &AzNnue, config: &AzLoopConfig) -> AzSelfplayData {
    crate::scope_profile!("az.selfplay.chunk");
    let mut rng = SplitMix64::new(config.seed);
    let mut samples = Vec::new();
    let mut position_fens = Vec::new();
    let mut red_wins = 0usize;
    let mut black_wins = 0usize;
    let mut draws = 0usize;
    let mut plies_total = 0usize;
    let mut start_games = [0usize; AzStartSource::COUNT];
    let mut start_phase_ply_sum = [0u64; AzStartSource::COUNT];
    let mut start_age_sum = [0u64; AzStartSource::COUNT];
    let mut start_age_max = [0u32; AzStartSource::COUNT];
    let mut start_temperature_sum = [0.0f32; AzStartSource::COUNT];
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
        let start = choose_selfplay_start(config, &mut rng, game_index);
        let mut position = start.position;
        let mut rule_history = start.rule_history;
        let start_phase_ply = start.phase_ply;
        let start_source = start.source;
        let start_source_index = start_source.index();
        let start_age = config.generation_update.saturating_sub(start.generation);
        start_games[start_source_index] += 1;
        start_phase_ply_sum[start_source_index] += start_phase_ply as u64;
        start_age_sum[start_source_index] += u64::from(start_age);
        start_age_max[start_source_index] = start_age_max[start_source_index].max(start_age);
        start_temperature_sum[start_source_index] += temperature_for_ply(config, start_phase_ply);
        let mut game_samples = Vec::new();
        let mut game_proofs = Vec::new();
        let mut result = None;
        let mut search_failed = false;
        let mut plies = 0usize;

        for local_ply in 0..config.max_plies.saturating_sub(start_phase_ply) {
            let ply = start_phase_ply + local_ply;
            plies = local_ply + 1;
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
                terminal.record_no_legal_moves(&position);
                break;
            }

            search_simulations.searches += 1;

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
            search_simulations.simulations_sum += search.simulations;
            crate::scope_profile!("az.selfplay.post_search");
            let proven_result = proven_root_value(&search.candidates).map(|value| {
                value as f32
                    * if position.side_to_move() == Color::Red {
                        1.0
                    } else {
                        -1.0
                    }
            });
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
            let temperature = temperature_for_ply(config, ply);
            let mv_opt = if temperature <= 1e-6 {
                search.best_move.or_else(|| {
                    choose_selfplay_move(
                        &search.candidates,
                        temperature,
                        config.temperature_visit_offset,
                        &mut rng,
                    )
                })
            } else {
                choose_selfplay_move(
                    &search.candidates,
                    temperature,
                    config.temperature_visit_offset,
                    &mut rng,
                )
            };
            let Some(mv) = mv_opt else {
                terminal.search_no_move += 1;
                search_failed = true;
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
                    search.value_wdl,
                    rng.unit_f32() < config.mirror_probability.clamp(0.0, 1.0),
                    move_meta,
                    search.simulations,
                    1.0,
                );
                game_samples.push(sample);
                game_proofs.push(proven_root_value(&search.candidates));
            }
            if let Some(outcome) = proven_result {
                terminal.search_proven[outcome_index(outcome)] += 1;
                result = Some(outcome);
                break;
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
        if result.is_none() && !search_failed {
            // 最后一手也可能形成将死、困毙或规则禁着，不能先记作步数截断。
            if position.legal_moves_with_rules(&rule_history).is_empty() {
                result = Some(if position.side_to_move() == Color::Red {
                    -1.0
                } else {
                    1.0
                });
                terminal.record_no_legal_moves(&position);
            } else {
                terminal.max_plies += 1;
            }
        }

        let terminal_result = result;
        let result: f32 = terminal_result.unwrap_or(0.0);
        match result.total_cmp(&0.0) {
            std::cmp::Ordering::Greater => red_wins += 1,
            std::cmp::Ordering::Less => black_wins += 1,
            std::cmp::Ordering::Equal => draws += 1,
        }
        plies_total += plies;

        {
            crate::scope_profile!("az.selfplay.finalize_game");
            finalize_value_targets(&mut game_samples, &game_proofs, terminal_result);
        }
        samples.extend(game_samples.clone());
        games.push(game_samples);
    }

    AzSelfplayData {
        samples,
        games,
        position_fens,
        red_wins,
        black_wins,
        draws,
        plies_total,
        start_games,
        start_phase_ply_sum,
        start_age_sum,
        start_age_max,
        start_temperature_sum,
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
    root_search_wdl: [f32; 3],
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
    let repetition_flags = candidates
        .iter()
        .map(|candidate| u8::from(position.move_repeats_history(rule_history, candidate.mv)))
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
        repetition_flags,
        policy,
        value_wdl: scalar_value_to_wdl_target(value),
        root_search_wdl: normalize_wdl_target(root_search_wdl),
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

fn proven_root_value(candidates: &[AzCandidate]) -> Option<i8> {
    if candidates
        .iter()
        .any(|candidate| candidate.solved == Some(1))
    {
        Some(1)
    } else if !candidates.is_empty()
        && candidates
            .iter()
            .all(|candidate| candidate.solved.is_some())
    {
        candidates
            .iter()
            .filter_map(|candidate| candidate.solved)
            .max()
    } else {
        None
    }
}

fn finalize_value_targets(
    samples: &mut [AzTrainingSample],
    proofs: &[Option<i8>],
    terminal_result: Option<f32>,
) {
    assert_eq!(samples.len(), proofs.len());
    if let Some(result) = terminal_result {
        for sample in samples.iter_mut() {
            sample.value = (result * sample.side_sign).clamp(-1.0, 1.0);
            sample.value_wdl = scalar_value_to_wdl_target(sample.value);
        }
    } else {
        // 截断不提供终局标签，但搜索的规则证明仍然有效。
        for sample in samples.iter_mut() {
            sample.value_weight = 0.0;
        }
    }
    for (sample, proof) in samples.iter_mut().zip(proofs) {
        if let Some(value) = proof {
            // 使用显式证明，不能从网络饱和的单点 WDL 推断。
            sample.value = f32::from(*value);
            sample.value_wdl = scalar_value_to_wdl_target(sample.value);
            sample.value_weight = 1.0;
        }
    }
}

fn temperature_for_ply(config: &AzLoopConfig, ply: usize) -> f32 {
    if config.temperature_cutoff_plies > 0 && ply >= config.temperature_cutoff_plies {
        return config.temperature_endgame;
    }
    let decay = if config.temperature_decay_plies == 0 {
        1.0
    } else {
        1.0 - (ply.saturating_sub(config.temperature_decay_delay_plies) / 2) as f32
            / (config.temperature_decay_plies / 2).max(1) as f32
    };
    if config.temperature_start <= 0.0 {
        return 0.0;
    }
    (config.temperature_start * decay.max(0.0)).max(config.temperature_endgame)
}

fn temperature_opening_plies(config: &AzLoopConfig) -> usize {
    config
        .temperature_decay_delay_plies
        .saturating_add(config.temperature_decay_plies)
}

fn choose_selfplay_move(
    candidates: &[AzCandidate],
    temperature: f32,
    visit_offset: f32,
    rng: &mut SplitMix64,
) -> Option<Move> {
    let priority = candidates.iter().map(AzCandidate::proof_priority).max()?;
    let fallback = candidates
        .iter()
        .find(|candidate| candidate.proof_priority() == priority)
        .map(|candidate| candidate.mv);
    if temperature <= 1e-6 {
        return candidates
            .iter()
            .max_by(|left, right| {
                left.proof_priority()
                    .cmp(&right.proof_priority())
                    .then_with(|| {
                        left.policy
                            .total_cmp(&right.policy)
                            .then_with(|| left.visits.cmp(&right.visits))
                    })
            })
            .map(|candidate| candidate.mv);
    }

    let weights = temperature_move_weights(candidates, temperature, visit_offset);
    let total = candidates
        .iter()
        .zip(&weights)
        .map(|(_, weight)| *weight)
        .sum::<f32>();
    if total <= 0.0 {
        return fallback;
    }

    let mut ticket = rng.unit_f32() * total;
    for (candidate, weight) in candidates.iter().zip(weights) {
        if ticket < weight {
            return Some(candidate.mv);
        }
        ticket -= weight;
    }
    fallback
}

fn temperature_move_weights(
    candidates: &[AzCandidate],
    temperature: f32,
    visit_offset: f32,
) -> Vec<f32> {
    let inv_temperature = 1.0 / temperature.max(1e-3);
    let priority = candidates
        .iter()
        .map(AzCandidate::proof_priority)
        .max()
        .unwrap_or(0);
    let max_visits = candidates
        .iter()
        .filter(|candidate| candidate.proof_priority() == priority)
        .map(|c| (c.visits as f32 + visit_offset).max(0.0))
        .fold(0.0f32, f32::max);
    candidates
        .iter()
        .map(|candidate| {
            if candidate.proof_priority() != priority {
                0.0
            } else if max_visits > 0.0 {
                ((candidate.visits as f32 + visit_offset).max(0.0) / max_visits)
                    .powf(inv_temperature)
            } else {
                candidate.prior.max(0.0).powf(inv_temperature)
            }
        })
        .collect()
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
    pub fpu_absolute_at_root: bool,
    pub minimum_kldgain_per_node: f32,
    pub draw_score: f32,
    pub policy_softmax_temp: f32,
}

pub fn play_arena_games_from_positions(
    candidate: &AzNnue,
    baseline: &AzNnue,
    positions: &[Position],
    config: AzArenaConfig,
) -> AzArenaReport {
    let snapshots = positions
        .iter()
        .cloned()
        .map(|position| AzStartSnapshot {
            rule_history: position.initial_rule_history(),
            position,
            phase_ply: 0,
            generation: 0,
        })
        .collect::<Vec<_>>();
    play_arena_games_from_snapshots(candidate, baseline, &snapshots, config)
}

pub fn play_arena_games_from_snapshots(
    candidate: &AzNnue,
    baseline: &AzNnue,
    snapshots: &[AzStartSnapshot],
    config: AzArenaConfig,
) -> AzArenaReport {
    let mut report = AzArenaReport::default();
    let mut red_scores = Vec::with_capacity(config.games_as_red);
    for game_index in 0..config.games_as_red {
        let mut snapshot = arena_start_snapshot(snapshots, config.start_index + game_index);
        let position = &mut snapshot.position;
        position.set_rule60_max_ply(config.rule60_max_ply);
        let outcome = play_arena_game(
            position,
            &snapshot.rule_history,
            candidate,
            baseline,
            config.simulations,
            config.max_plies.saturating_sub(snapshot.phase_ply as usize),
            config.seed ^ (config.start_index + game_index) as u64,
            config.cpuct,
            config.cpuct_at_root,
            config.cpuct_base,
            config.cpuct_factor,
            config.cpuct_base_at_root,
            config.cpuct_factor_at_root,
            config.fpu_value,
            config.fpu_value_at_root,
            config.fpu_absolute_at_root,
            config.minimum_kldgain_per_node,
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
        let mut snapshot = arena_start_snapshot(snapshots, config.start_index + game_index);
        let position = &mut snapshot.position;
        position.set_rule60_max_ply(config.rule60_max_ply);
        let outcome = play_arena_game(
            position,
            &snapshot.rule_history,
            baseline,
            candidate,
            config.simulations,
            config.max_plies.saturating_sub(snapshot.phase_ply as usize),
            config.seed ^ (config.start_index + game_index) as u64,
            config.cpuct,
            config.cpuct_at_root,
            config.cpuct_base,
            config.cpuct_factor,
            config.cpuct_base_at_root,
            config.cpuct_factor_at_root,
            config.fpu_value,
            config.fpu_value_at_root,
            config.fpu_absolute_at_root,
            config.minimum_kldgain_per_node,
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

fn arena_start_snapshot(snapshots: &[AzStartSnapshot], game_index: usize) -> AzStartSnapshot {
    if snapshots.is_empty() {
        let position = Position::startpos();
        AzStartSnapshot {
            rule_history: position.initial_rule_history(),
            position,
            phase_ply: 0,
            generation: 0,
        }
    } else {
        let index = game_index % snapshots.len();
        snapshots[index].clone()
    }
}

fn play_arena_game(
    initial_position: &Position,
    initial_rule_history: &[RuleHistoryEntry],
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
    fpu_absolute_at_root: bool,
    minimum_kldgain_per_node: f32,
    draw_score: f32,
    policy_softmax_temp: f32,
) -> f32 {
    let mut position = initial_position.clone();
    let mut rule_history = initial_rule_history.to_vec();
    if rule_history.is_empty() {
        rule_history = position.initial_rule_history();
    }
    if let Some(outcome) = position.rule_outcome_with_history(&rule_history) {
        return match outcome {
            RuleOutcome::Draw(_) => 0.0,
            RuleOutcome::Win(Color::Red) => 1.0,
            RuleOutcome::Win(Color::Black) => -1.0,
        };
    }
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
                fpu_absolute_at_root,
                minimum_kldgain_per_node,
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
    if position.legal_moves_with_rules(&rule_history).is_empty() {
        if position.side_to_move() == Color::Red {
            -1.0
        } else {
            1.0
        }
    } else {
        0.0
    }
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
            temperature_cutoff_plies: 0,
            temperature_visit_offset: 0.0,
            temperature_endgame: 0.0,
            temperature_decay_delay_plies: 0,
            temperature_decay_plies: 0,
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
            fpu_absolute_at_root: false,
            minimum_kldgain_per_node: 0.0,
            draw_score: 0.0,
            policy_softmax_temp: 1.0,
            opening_positions: Default::default(),
            mirror_probability: 0.0,
            record_fens: false,
            mate_search_plies: 0,
            tactical_search_nodes: 0,
            tactical_search_plies: 8,
            tactical_quiet_plies: 2,
        }
    }

    /// `record_fens` 原来全仓没有读取方，属于"会静默腐烂"的公开诊断开关（默认 false）。
    /// 这里钉住它的契约：每个样本恰好一条 FEN，写的是该样本局面的**有效** 60 回合计数
    /// （`to_fen_with_history`，而不是 `to_fen` 的原始 halfmove clock），且能被回读。
    #[test]
    fn record_fens_writes_one_effective_clock_fen_per_sample() {
        let mut config = selfplay_test_config(1);
        config.record_fens = true;
        config.max_plies = 8;
        let model = AzNnue::random_with_arch(crate::az::AzNnueArch::default(), 11);
        let data = generate_selfplay_data(&model, &config);

        assert!(!data.samples.is_empty());
        assert_eq!(data.position_fens.len(), data.samples.len());
        for fen in &data.position_fens {
            let tokens = fen.split(' ').collect::<Vec<_>>();
            assert_eq!(tokens.len(), 6, "unexpected FEN shape: {fen}");
            assert!(
                tokens[4].parse::<u16>().is_ok(),
                "clock token must be numeric: {fen}"
            );
            Position::from_fen(fen)
                .unwrap_or_else(|err| panic!("recorded FEN does not round-trip `{fen}`: {err}"));
        }
    }

    /// 配置里的连杀预算必须真的作用到自博弈用的那个模型上。
    ///
    /// 用一个 **mate-in-8** 局面开局（不能用 mate-in-1：那只靠旧的 `immediate_mate_child`
    /// 就能证，证明不了新接线）。预算生效时根搜索会证明杀棋，于是该样本的策略目标被压成
    /// 杀着上的一热分布、价值目标变成必胜；关掉之后同一局面必须回到普通 MCTS。
    /// 这是"让模型从自博弈里学到连杀"整条链路的端到端证据
    /// （配置 → 模型补丁 → 搜索证明 → 训练样本）。
    #[test]
    fn selfplay_applies_configured_mate_search() {
        let mate_position = Position::from_fen(
            "2bakab2/9/5r1c1/p1PRC1p2/4P2nP/6P2/4N1r2/7c1/4A4/2BAK1B1R b - - 0 1",
        )
        .unwrap();
        let mate_move = mate_position.parse_uci_move("h2h0").unwrap();
        let mate_index = dense_move_index(canonical_move(mate_position.side_to_move(), mate_move));
        assert!(
            mate_index < crate::az::DENSE_MOVE_SPACE,
            "杀着必须在策略走法空间里"
        );

        let mut config = selfplay_test_config(1);
        config.max_plies = 2;
        config.mate_search_plies = 15;
        config.opening_positions = Arc::from(vec![AzStartSnapshot {
            rule_history: mate_position.initial_rule_history(),
            position: mate_position,
            phase_ply: 0,
            generation: 0,
        }]);
        let model = AzNnue::random_with_arch(crate::az::AzNnueArch::default(), 13);
        let data = generate_selfplay_data(&model, &config);

        let first = data.samples.first().expect("至少要有一个样本");
        let index = first
            .move_indices
            .iter()
            .position(|&candidate| candidate == mate_index)
            .expect("杀着必须在候选里");
        assert!(
            (first.policy[index] - 1.0).abs() < 1e-6,
            "策略目标应压在杀着上：{:?}",
            first.policy
        );
        assert_eq!(first.value, 1.0);
        assert_eq!(first.value_wdl, [1.0, 0.0, 0.0]);

        // 关掉连杀预算之后同一局面必须回到普通 MCTS：64 simulations 撞不出 mate-in-8，
        // 所以既不会出现一热策略目标、也不会被判成必胜。
        config.mate_search_plies = 0;
        let plain = generate_selfplay_data(&model, &config);
        let plain_first = plain.samples.first().expect("至少要有一个样本");
        let plain_index = plain_first
            .move_indices
            .iter()
            .position(|&candidate| candidate == mate_index)
            .expect("杀着必须在候选里");
        assert!(
            plain_first.policy[plain_index] < 1.0 - 1e-6,
            "关闭时不应出现一热策略目标：{:?}",
            plain_first.policy
        );
        assert!(plain_first.value < 1.0, "关闭时不应判成必胜");
    }

    #[test]
    fn px0_kld_selfplay_records_actual_visits() {
        let mut position =
            Position::from_fen("4k1b2/4a4/4ba3/p8/4cN3/3n2N1P/c8/4C4/4A4/2B1KAB2 b").unwrap();
        let checking_move = position.parse_uci_move("a3a0").unwrap();
        position.make_move(checking_move);
        assert_eq!(position.legal_moves(), [Move::from_uci("c0a2").unwrap()]);
        let mut config = selfplay_test_config(1);
        config.max_plies = 1;
        config.simulations = 10_000;
        config.minimum_kldgain_per_node = 0.00005;
        config.opening_positions = vec![AzStartSnapshot {
            rule_history: position.initial_rule_history(),
            position,
            phase_ply: 0,
            generation: 0,
        }]
        .into();
        let data = generate_selfplay_chunk(&AzNnue::random(4, 7), &config);
        assert_eq!(data.samples.len(), 1);
        assert_eq!(data.samples[0].search_simulations, 400);
        assert_eq!(data.search_simulations.searches, 1);
        assert_eq!(data.search_simulations.simulations_sum, 400);
    }

    #[test]
    fn px0_temperature_uses_full_moves_and_cutoff() {
        let mut config = selfplay_test_config(1);
        config.temperature_start = 0.9;
        config.temperature_endgame = 0.6;
        config.temperature_decay_delay_plies = 40;
        config.temperature_decay_plies = 120;
        config.temperature_cutoff_plies = 78;
        assert_eq!(temperature_for_ply(&config, 39), 0.9);
        assert_eq!(temperature_for_ply(&config, 41), 0.9);
        assert!((temperature_for_ply(&config, 42) - 0.885).abs() < 1e-6);
        assert!((temperature_for_ply(&config, 77) - 0.63).abs() < 1e-6);
        assert_eq!(temperature_for_ply(&config, 78), 0.6);
        assert_eq!(temperature_for_ply(&config, 200), 0.6);
        let mut candidates = vec![
            candidate_q(Move::new(0, 1), 0, 0.0),
            candidate_q(Move::new(0, 2), 0, 0.0),
        ];
        candidates[0].prior = 0.2;
        candidates[1].prior = 0.8;
        assert_eq!(
            temperature_move_weights(&candidates, 1.0, -0.8),
            vec![0.2, 0.8]
        );
        candidates[0].visits = 1;
        candidates[1].visits = 2;
        let weights = temperature_move_weights(&candidates, 1.0, -0.8);
        assert!((weights[0] - 1.0 / 6.0).abs() < 1e-6);
        assert_eq!(weights[1], 1.0);
    }

    #[test]
    fn selfplay_keeps_noise_after_temperature_cutoff() {
        let mut config = selfplay_test_config(1);
        config.temperature_cutoff_plies = 40;
        config.temperature_endgame = 0.6;
        config.root_exploration_fraction = 0.1;
        assert_eq!(temperature_for_ply(&config, 40), 0.6);
        assert_eq!(
            selfplay_search_limits(&config, 40, 0).root_exploration_fraction,
            0.1
        );
    }

    #[test]
    fn terminal_monitoring_distinguishes_checkmate_stalemate_and_rule_blocking() {
        let mut stats = AzTerminalStats::default();
        for (fen, checked) in [
            ("4k4/3R1R3/9/9/4P4/9/9/9/9/4K4 b - - 0 1", false),
            ("4k4/3RRR3/9/9/4P4/9/9/9/9/4K4 b - - 0 1", true),
        ] {
            let position = Position::from_fen(fen).unwrap();
            assert!(position.legal_moves().is_empty());
            assert_eq!(position.in_check(Color::Black), checked);
            stats.record_no_legal_moves(&position);
        }
        stats.record_no_legal_moves(&Position::startpos());
        let mut merged = AzTerminalStats::default();
        merged.add_assign(&stats);
        assert_eq!(merged.no_legal_moves, 3);
        assert_eq!(merged.checkmate, 1);
        assert_eq!(merged.stalemate, 1);
        assert_eq!(merged.rule_blocked, 1);
    }

    #[test]
    fn unproven_games_reach_cutoff_without_value_labels() {
        let model = AzNnue::random(16, 20260907);
        let mut config = selfplay_test_config(4);
        config.max_plies = 1;
        config.simulations = 2;
        let data = generate_selfplay_chunk(&model, &config);
        assert_eq!(data.terminal.max_plies, 4);
        assert_eq!(data.terminal.search_proven, [0, 0, 0]);
        assert!(data.samples.iter().all(|sample| sample.value_weight == 0.0));
    }

    #[test]
    fn proven_mate_keeps_the_training_sample() {
        let model = AzNnue::random(16, 20260907);
        let position = Position::from_fen(
            "2bak2r1/4a4/4b4/p2R4p/4C1n2/2P1c3P/P1r3P2/4B4/4A4/2BK1A2R w - - 1 1",
        )
        .unwrap();
        let mate = position.parse_uci_move("d6d9").unwrap();
        let mate_index = dense_move_index(canonical_move(position.side_to_move(), mate));
        let mut config = selfplay_test_config(1);
        config.simulations = 1;
        config.mirror_probability = 0.0;
        config.opening_positions = vec![AzStartSnapshot {
            rule_history: position.initial_rule_history(),
            position,
            phase_ply: 0,
            generation: 0,
        }]
        .into();
        let data = generate_selfplay_chunk(&model, &config);
        assert_eq!(data.terminal.search_proven, [1, 0, 0]);
        assert_eq!(data.samples.len(), 1);
        let sample = &data.samples[0];
        assert_eq!(sample.value_wdl, [1.0, 0.0, 0.0]);
        assert_eq!(sample.value_weight, 1.0);
        let policy = sample
            .move_indices
            .iter()
            .position(|&index| index == mate_index)
            .unwrap();
        assert_eq!(sample.policy[policy], 1.0);
    }

    #[test]
    fn mate_on_last_allowed_ply_keeps_terminal_value_supervision() {
        let mut model = AzNnue::random(16, 20260907);
        let mut position = Position::from_fen(
            "r1baka3/4n4/n3b4/4p3p/1PP3pr1/5RC2/4P1ccP/3CB1N2/5R3/2BAKA3 b - - 3 1",
        )
        .unwrap();
        let mut history = position.initial_rule_history();
        let mv = position.parse_uci_move("a9b9").unwrap();
        history.push(position.rule_history_entry_after_move(mv));
        position.make_move(mv);
        let mate = position.parse_uci_move("f4f9").unwrap();
        let mate_index = dense_move_index(canonical_move(position.side_to_move(), mate));
        model.policy_move_bias.fill(-100.0);
        model.policy_move_bias[mate_index] = 100.0;
        let mut config = selfplay_test_config(1);
        config.max_plies = 1;
        config.simulations = 2;
        config.mirror_probability = 0.0;
        config.opening_positions = vec![AzStartSnapshot {
            position,
            rule_history: history,
            phase_ply: 0,
            generation: 0,
        }]
        .into();
        let data = generate_selfplay_chunk(&model, &config);
        let sample = &data.samples[0];
        assert_eq!(
            sample.move_indices[sample.meta.played_index as usize],
            mate_index
        );
        assert_eq!(data.red_wins, 1, "terminal={:?}", data.terminal);
        assert_eq!(data.terminal.checkmate + data.terminal.search_proven[0], 1);
        assert_eq!(data.terminal.max_plies, 0);
        assert_eq!(data.samples.len(), 1);
        assert_eq!(data.samples[0].value_weight, 1.0);
        assert_eq!(data.samples[0].value_wdl, [1.0, 0.0, 0.0]);
        let start = &config.opening_positions[0];
        assert_eq!(
            play_arena_game(
                &start.position,
                &start.rule_history,
                &model,
                &model,
                config.simulations,
                config.max_plies,
                config.seed,
                config.cpuct,
                config.cpuct_at_root,
                config.cpuct_base,
                config.cpuct_factor,
                config.cpuct_base_at_root,
                config.cpuct_factor_at_root,
                config.fpu_value,
                config.fpu_value_at_root,
                config.fpu_absolute_at_root,
                config.minimum_kldgain_per_node,
                config.draw_score,
                config.policy_softmax_temp,
            ),
            1.0
        );
    }

    #[test]
    fn cutoff_games_keep_policy_but_do_not_train_false_draw_values() {
        let model = AzNnue::random(16, 20260907);
        let mut config = selfplay_test_config(4);
        config.max_plies = 1;
        config.simulations = 2;
        let data = generate_selfplay_chunk(&model, &config);
        assert_eq!(data.terminal.max_plies, 4);
        assert_eq!(data.terminal.search_no_move, 0);
        assert_eq!(data.draws, 4);
        assert!(!data.samples.is_empty());
        assert!(data.samples.iter().all(|sample| sample.value_weight == 0.0));
        assert!(
            data.samples
                .iter()
                .all(|sample| sample.policy_weight == 1.0)
        );
    }

    #[test]
    fn selfplay_data_merge_preserves_start_stats() {
        let mut merged = AzSelfplayData::default();
        let mut chunk = AzSelfplayData::default();
        chunk.start_games[AzStartSource::OpeningBook.index()] = 2;
        chunk.start_phase_ply_sum[AzStartSource::OpeningBook.index()] = 16;
        chunk.start_age_sum[AzStartSource::OpeningBook.index()] = 6;
        chunk.start_age_max[AzStartSource::OpeningBook.index()] = 4;
        chunk.start_temperature_sum[AzStartSource::OpeningBook.index()] = 1.2;

        merged.add_assign(&chunk);

        assert_eq!(merged.start_games, [0, 2, 0]);
        assert_eq!(merged.start_phase_ply_sum, [0, 16, 0]);
        assert_eq!(merged.start_age_sum, [0, 6, 0]);
        assert_eq!(merged.start_age_max, [0, 4, 0]);
        assert_eq!(merged.start_temperature_sum, [0.0, 1.2, 0.0]);
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
    fn start_source_distinguishes_standard_and_opening_positions() {
        let config = selfplay_test_config(1);
        let start = choose_selfplay_start(&config, &mut SplitMix64::new(1), 0);
        assert_eq!(start.source, AzStartSource::Startpos);

        let mut config = selfplay_test_config(1);
        let position =
            Position::from_fen("rnbakabnr/9/1c5c1/p1p1p1p1p/9/4P4/P1P3P1P/1C5C1/9/RNBAKABNR b")
                .unwrap();
        config.opening_positions = vec![AzStartSnapshot {
            rule_history: position.initial_rule_history(),
            position,
            phase_ply: 8,
            generation: 1,
        }]
        .into();
        let start = choose_selfplay_start(&config, &mut SplitMix64::new(1), 0);
        assert_eq!(start.source, AzStartSource::OpeningBook);
        assert_eq!(start.phase_ply, 8);
    }

    fn candidate(mv: Move, policy: f32) -> AzCandidate {
        AzCandidate {
            mv,
            visits: (policy * 100.0) as u32,
            q: 0.0,
            raw_prior: policy,
            prior: policy,
            policy,
            solved: None,
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
            solved: None,
        }
    }

    fn sample(value: f32, side_sign: f32) -> AzTrainingSample {
        AzTrainingSample {
            repetition_flags: Vec::new(),
            features: Vec::new(),
            rule_context: [0.0; crate::az::RULE_CONTEXT_SIZE],
            move_indices: Vec::new(),
            policy: Vec::new(),
            value_wdl: scalar_value_to_wdl_target(value),
            root_search_wdl: scalar_value_to_wdl_target(value),
            value,
            side_sign,
            policy_weight: 1.0,
            value_weight: 1.0,
            search_simulations: 0,
            meta: AzSampleMeta::default(),
        }
    }

    #[test]
    fn proven_value_targets_survive_truncation_and_opponent_blunders() {
        let mut samples = [sample(0.0, 1.0), sample(0.0, -1.0), sample(0.0, 1.0)];
        samples[2].root_search_wdl = [1.0, 0.0, 0.0];
        finalize_value_targets(&mut samples, &[Some(1), Some(-1), None], None);
        assert_eq!(samples[0].value_wdl, [1.0, 0.0, 0.0]);
        assert_eq!(samples[1].value_wdl, [0.0, 0.0, 1.0]);
        assert_eq!(samples[0].value_weight, 1.0);
        assert_eq!(samples[1].value_weight, 1.0);
        assert_eq!(samples[2].value_weight, 0.0);

        finalize_value_targets(&mut samples, &[Some(1), Some(-1), None], Some(-1.0));
        assert_eq!(samples[0].value, 1.0);
        assert_eq!(samples[1].value, -1.0);
        assert_eq!(samples[2].value, -1.0);
    }

    #[test]
    fn root_value_proof_requires_a_win_or_all_replies_solved() {
        let mut candidates = [
            candidate(Move::new(0, 1), 0.5),
            candidate(Move::new(0, 2), 0.5),
        ];
        assert_eq!(proven_root_value(&[]), None);
        candidates[0].solved = Some(0);
        assert_eq!(proven_root_value(&candidates), None);
        candidates[1].solved = Some(-1);
        assert_eq!(proven_root_value(&candidates), Some(0));
        candidates[0].solved = Some(-1);
        assert_eq!(proven_root_value(&candidates), Some(-1));
        candidates[0].solved = Some(1);
        candidates[1].solved = None;
        assert_eq!(proven_root_value(&candidates), Some(1));
    }

    #[test]
    fn terminal_value_targets_use_game_result_for_each_side() {
        let mut samples = [sample(0.0, 1.0), sample(0.0, -1.0)];
        samples[0].meta.root_q = -1.0;
        samples[1].meta.root_q = 1.0;

        finalize_value_targets(&mut samples, &[None, None], Some(1.0));

        assert_eq!(samples[0].value_wdl, [1.0, 0.0, 0.0]);
        assert_eq!(samples[0].value, 1.0);
        assert!(samples.iter().all(|sample| sample.value_weight == 1.0));
        assert_eq!(samples[1].value_wdl, [0.0, 0.0, 1.0]);
        assert_eq!(samples[1].value, -1.0);
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
            [0.6, 0.3, 0.1],
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
        assert_eq!(sample.root_search_wdl, [0.6, 0.3, 0.1]);
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
    fn temperature_weights_remain_finite_for_large_visit_counts() {
        for visits in [65, 85, 65_535, 85_000, u32::MAX] {
            let candidates = vec![
                candidate_q(Move::new(0, 1), visits / 2, 0.0),
                candidate_q(Move::new(0, 2), visits, 0.0),
            ];
            for temperature in [0.001, 0.05, 0.6, 0.9, 1.2] {
                let weights = temperature_move_weights(&candidates, temperature, -0.8);
                assert!(
                    weights
                        .iter()
                        .all(|weight| weight.is_finite() && *weight >= 0.0 && *weight <= 1.0)
                );
                assert_eq!(weights[1], 1.0);
            }
        }
    }

    #[test]
    fn temperature_one_samples_directly_from_visit_counts() {
        let candidates = vec![
            candidate_q(Move::new(0, 1), 1, 0.0),
            candidate_q(Move::new(0, 2), 10, 0.0),
        ];

        let weights = temperature_move_weights(&candidates, 1.0, 0.0);

        assert_eq!(weights, vec![0.1, 1.0]);
    }

    #[test]
    fn opening_temperature_stays_at_start_value_through_40_plies() {
        let mut config = selfplay_test_config(1);
        config.temperature_start = 1.2;
        config.temperature_endgame = 0.05;
        config.temperature_decay_delay_plies = 40;
        config.temperature_decay_plies = 40;

        assert_eq!(temperature_for_ply(&config, 0), 1.2);
        assert_eq!(temperature_for_ply(&config, 40), 1.2);
        assert!(temperature_for_ply(&config, 42) < 1.2);
    }

    #[test]
    fn solver_temperature_excludes_proven_losses_despite_old_visits() {
        let mut lost = candidate(Move::new(0, 1), 0.99);
        lost.visits = u32::MAX;
        lost.solved = Some(-1);
        let safe = candidate(Move::new(2, 3), 0.01);
        let moves = [lost, safe];
        let mut rng = SplitMix64::new(20260927);
        for temperature in [0.0, 0.05, 0.9, 1.2] {
            if temperature > 0.0 {
                assert_eq!(temperature_move_weights(&moves, temperature, -0.8)[0], 0.0);
            }
            for _ in 0..128 {
                assert_eq!(
                    choose_selfplay_move(&moves, temperature, -0.8, &mut rng),
                    Some(moves[1].mv)
                );
            }
        }
    }

    #[test]
    fn solver_temperature_selects_proven_win_over_unresolved_favorite() {
        let favorite = candidate(Move::new(0, 1), 0.99);
        let mut winning = candidate(Move::new(2, 3), 0.01);
        winning.solved = Some(1);
        let moves = [favorite, winning];
        let mut rng = SplitMix64::new(20260927);
        for temperature in [0.0, 0.05, 0.9, 1.2] {
            if temperature > 0.0 {
                assert_eq!(temperature_move_weights(&moves, temperature, -0.8)[0], 0.0);
            }
            for _ in 0..128 {
                assert_eq!(
                    choose_selfplay_move(&moves, temperature, -0.8, &mut rng),
                    Some(moves[1].mv)
                );
            }
        }
    }

    #[test]
    fn solver_temperature_zero_mass_fallback_still_excludes_proven_loss() {
        let mut lost = candidate(Move::new(0, 1), 0.99);
        lost.solved = Some(-1);
        let safe = candidate(Move::new(2, 3), 0.0);
        let moves = [lost, safe];
        let mut rng = SplitMix64::new(20260927);
        assert_eq!(
            choose_selfplay_move(&moves, 0.9, -0.8, &mut rng),
            Some(moves[1].mv)
        );
    }
}
