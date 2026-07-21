use std::sync::Arc;

use rayon::prelude::*;

use crate::nnue::{
    HistoryMove, canonical_move, extract_sparse_features_az_canonical, mirror_file_move,
    mirror_sparse_features_az_canonical_file,
};
use crate::xiangqi::{Color, Move, Position, RuleDrawReason, RuleOutcome};

use super::alphazero::append_history;
use super::{
    AzCandidate, AzLoopConfig, AzNnue, AzSampleMeta, AzSearchLimits, AzSearchResult,
    AzTrainingSample, SplitMix64, alphazero_search_with_history_and_rules, cp_from_q,
    dense_move_index, scalar_value_to_wdl_target,
};

const HIGH_CONFIDENCE_Q_ADVANTAGE: f32 = 0.10;
const HIGH_CONFIDENCE_POLICY_MASS: f32 = 0.50;
const LEAF_VERIFY_MAX_CANDIDATES: usize = 6;
const LEAF_VERIFY_EXPLORER_SLOTS: usize = 1;
const LEAF_VERIFY_POLICY_MIX: f32 = 0.75;
const ENDGAME_SCOUT_SIMULATIONS_PER_LEAF: usize = 500;
const ENDGAME_SCOUT_BRANCHES: usize = 2;
// Branch suffixes carry real terminal labels for both heads. Equal weighting keeps
// this rare coverage visible in replay without creating a policy/value mismatch.
const SCOUT_SUFFIX_TRAIN_WEIGHT: f32 = 4.0;

#[derive(Clone, Debug)]
struct LeafVerification {
    search: AzSearchResult,
    candidate_count: usize,
    capture_count: usize,
    check_count: usize,
    explorer_count: usize,
    verified_indices: Vec<usize>,
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

#[derive(Clone, Copy, Debug, Default)]
pub struct AzSearchSimulationStats {
    pub searches: usize,
    pub low_searches: usize,
    pub simulations_sum: usize,
}

/// Diagnostics for deterministic, no-noise re-searches performed inside normal
/// self-play. They quantify whether extra internal compute actually changes a
/// training target, rather than merely adding cost.
#[derive(Clone, Copy, Debug, Default)]
pub struct AzBranchReanalysisStats {
    pub searches: usize,
    pub simulations_sum: usize,
    pub best_move_changed: usize,
    pub value_delta_abs_sum: f32,
    pub policy_kl_sum: f32,
    /// At a flipped root, the deep-search Q of its selected move minus the
    /// deep-search Q of the shallow selected move.
    pub flipped_q_advantage_sum: f32,
    pub flipped_q_advantage_count: usize,
    /// Flips with a material Q advantage and a concentrated deep visit policy.
    pub high_confidence_flips: usize,
    pub verified_candidate_sum: usize,
    pub verified_capture_candidates: usize,
    pub verified_check_candidates: usize,
    pub verified_explorer_candidates: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AzEndgameRepairStats {
    pub probes: usize,
    pub accepted: usize,
    pub branches_spawned: usize,
    pub rejected_no_flip: usize,
    pub rejected_low_advantage: usize,
    pub verifier: AzBranchReanalysisStats,
}

impl AzEndgameRepairStats {
    pub fn add_assign(&mut self, other: &Self) {
        self.probes += other.probes;
        self.accepted += other.accepted;
        self.branches_spawned += other.branches_spawned;
        self.rejected_no_flip += other.rejected_no_flip;
        self.rejected_low_advantage += other.rejected_low_advantage;
        self.verifier.add_assign(&other.verifier);
    }
}

impl AzBranchReanalysisStats {
    pub fn add_assign(&mut self, other: &Self) {
        self.searches += other.searches;
        self.simulations_sum += other.simulations_sum;
        self.best_move_changed += other.best_move_changed;
        self.value_delta_abs_sum += other.value_delta_abs_sum;
        self.policy_kl_sum += other.policy_kl_sum;
        self.flipped_q_advantage_sum += other.flipped_q_advantage_sum;
        self.flipped_q_advantage_count += other.flipped_q_advantage_count;
        self.high_confidence_flips += other.high_confidence_flips;
        self.verified_candidate_sum += other.verified_candidate_sum;
        self.verified_capture_candidates += other.verified_capture_candidates;
        self.verified_check_candidates += other.verified_check_candidates;
        self.verified_explorer_candidates += other.verified_explorer_candidates;
    }

    fn record(
        &mut self,
        shallow: &super::AzSearchResult,
        deep: &super::AzSearchResult,
        simulations: usize,
        candidate_count: usize,
        capture_count: usize,
        check_count: usize,
        explorer_count: usize,
    ) -> bool {
        self.searches += 1;
        self.simulations_sum += simulations;
        self.verified_candidate_sum += candidate_count;
        self.verified_capture_candidates += capture_count;
        self.verified_check_candidates += check_count;
        self.verified_explorer_candidates += explorer_count;
        self.best_move_changed += usize::from(deep.best_move != shallow.best_move);
        self.value_delta_abs_sum += (deep.value_q - shallow.value_q).abs();
        self.policy_kl_sum += policy_kl(&shallow.candidates, &deep.candidates);
        let mut high_confidence = false;
        if let (Some(shallow_move), Some(deep_move)) = (shallow.best_move, deep.best_move)
            && shallow_move != deep_move
        {
            let shallow_q = deep
                .candidates
                .iter()
                .find(|candidate| candidate.mv == shallow_move)
                .map_or(0.0, |candidate| candidate.q);
            let deep = deep
                .candidates
                .iter()
                .find(|candidate| candidate.mv == deep_move)
                .expect("deep best move must be a deep-search candidate");
            let q_advantage = deep.q - shallow_q;
            self.flipped_q_advantage_sum += q_advantage;
            self.flipped_q_advantage_count += 1;
            high_confidence = is_high_confidence_branch_flip(q_advantage, deep.policy);
            self.high_confidence_flips += usize::from(high_confidence);
        }
        high_confidence
    }
}

impl AzSearchSimulationStats {
    pub fn add_assign(&mut self, other: &Self) {
        self.searches += other.searches;
        self.low_searches += other.low_searches;
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
    pub branch_reanalysis: AzBranchReanalysisStats,
    pub branch_reanalysis_phase: [AzBranchReanalysisStats; 3],
    pub phase_root_counts: [usize; 3],
    pub endgame_audit: AzBranchReanalysisStats,
    pub endgame_repair: AzEndgameRepairStats,
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
        self.branch_reanalysis.add_assign(&other.branch_reanalysis);
        for (left, right) in self
            .branch_reanalysis_phase
            .iter_mut()
            .zip(other.branch_reanalysis_phase)
        {
            left.add_assign(&right);
        }
        for (left, right) in self
            .phase_root_counts
            .iter_mut()
            .zip(other.phase_root_counts)
        {
            *left += right;
        }
        self.endgame_audit.add_assign(&other.endgame_audit);
        self.endgame_repair.add_assign(&other.endgame_repair);
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
        merged
            .branch_reanalysis
            .add_assign(&chunk.branch_reanalysis);
        for (left, right) in merged
            .branch_reanalysis_phase
            .iter_mut()
            .zip(chunk.branch_reanalysis_phase)
        {
            left.add_assign(&right);
        }
        for (left, right) in merged
            .phase_root_counts
            .iter_mut()
            .zip(chunk.phase_root_counts)
        {
            *left += right;
        }
        merged.endgame_audit.add_assign(&chunk.endgame_audit);
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
    let branch_reanalysis = AzBranchReanalysisStats::default();
    let branch_reanalysis_phase = [AzBranchReanalysisStats::default(); 3];
    let mut phase_root_counts = [0usize; 3];
    let mut endgame_audit = AzBranchReanalysisStats::default();
    let mut endgame_repair = AzEndgameRepairStats::default();

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
        let mut result = None;
        let mut plies = 0usize;
        let mut repair_suffixes = Vec::new();
        let mut spawned_repair_suffixes = false;
        let scout_target_ply = random_endgame_scout_ply(config, &mut rng);
        let allow_resign = rng.unit_f32() * 100.0 >= config.resign_playthrough;

        for ply in 0..config.max_plies {
            plies = ply + 1;
            phase_root_counts[phase_for_ply(ply)] += 1;
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

            let search_simulation_count = search_simulations_for_ply(config, &mut rng);
            search_simulations.searches += 1;
            search_simulations.simulations_sum += search_simulation_count;
            search_simulations.low_searches +=
                usize::from(search_simulation_count < config.simulations);
            let search = {
                crate::scope_profile!("az.selfplay.search");
                alphazero_search_with_history_and_rules(
                    &position,
                    &history,
                    Some(rule_history.clone()),
                    Some(legal),
                    model,
                    AzSearchLimits {
                        simulations: search_simulation_count,
                        seed: rng.next_u64() ^ ((game_index as u64) << 32) ^ ply as u64,
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
                        draw_score: config.draw_score,
                        moves_left_max_effect: config.moves_left_max_effect,
                        moves_left_slope: config.moves_left_slope,
                        moves_left_threshold: config.moves_left_threshold,
                        moves_left_constant_factor: config.moves_left_constant_factor,
                        moves_left_scaled_factor: config.moves_left_scaled_factor,
                        moves_left_quadratic_factor: config.moves_left_quadratic_factor,
                        value_scale: 1.0,
                    },
                )
            };
            // The ordinary root remains the main line. Independent search is an
            // exploration trigger only: it creates real terminal-labelled suffixes
            // instead of directly imposing a high-budget policy target.
            if !spawned_repair_suffixes && scout_target_ply == Some(ply) {
                let verification = independent_leaf_verification(
                    &position,
                    &history,
                    &rule_history,
                    model,
                    config,
                    &search,
                    rng.next_u64(),
                    Some(ENDGAME_SCOUT_SIMULATIONS_PER_LEAF),
                    true,
                );
                endgame_repair.probes += 1;
                endgame_repair.verifier.record(
                    &search,
                    &verification.search,
                    verification.search.simulations,
                    verification.candidate_count,
                    verification.capture_count,
                    verification.check_count,
                    verification.explorer_count,
                );
                if endgame_repair_accepts(&search, &verification.search) {
                    endgame_repair.accepted += 1;
                    let suffix_count_before = repair_suffixes.len();
                    for mv in endgame_scout_branch_moves(
                        &search,
                        &verification.search,
                        &verification.verified_indices,
                    ) {
                        let mut branch_position = position.clone();
                        let mut branch_history = history.clone();
                        append_history(&mut branch_history, &branch_position, mv);
                        let mover = branch_position.side_to_move();
                        branch_position.make_move(mv);
                        let mut branch_rules = rule_history.clone();
                        branch_rules.push(
                            branch_position.rule_history_entry_after_moved(mover, mv.to as usize),
                        );
                        repair_suffixes.push((
                            branch_position,
                            branch_history,
                            branch_rules,
                            ply + 1,
                        ));
                    }
                    endgame_repair.branches_spawned +=
                        repair_suffixes.len().saturating_sub(suffix_count_before);
                    spawned_repair_suffixes = !repair_suffixes.is_empty();
                } else {
                    let moved = verification.search.best_move != search.best_move;
                    if !moved {
                        endgame_repair.rejected_no_flip += 1;
                    } else {
                        endgame_repair.rejected_low_advantage += 1;
                    }
                }
            }
            if should_run_endgame_audit(config, ply, &mut rng) {
                // Always use a fresh verification here. Otherwise an audited
                // endgame that happened to be a training branch would bias the
                // audit toward the branch trigger condition.
                let audit = independent_leaf_verification(
                    &position,
                    &history,
                    &rule_history,
                    model,
                    config,
                    &search,
                    rng.next_u64(),
                    None,
                    false,
                );
                endgame_audit.record(
                    &search,
                    &audit.search,
                    config.branch_reanalysis_simulations,
                    audit.candidate_count,
                    audit.capture_count,
                    audit.check_count,
                    audit.explorer_count,
                );
            }
            let effective_search_simulations = search_simulation_count;
            let effective_policy_weight =
                policy_weight_for_search(config, effective_search_simulations);
            let effective_value_weight = 1.0;
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
                let meta = root_search_meta(
                    &search.candidates,
                    search.value_q,
                    config.generation_update,
                    config.seed ^ game_index as u64,
                    ply,
                );
                let sample = make_training_sample(
                    &position,
                    &history,
                    &search.candidates,
                    search.value_q,
                    config.policy_softmax_temp,
                    rng.unit_f32() < config.mirror_probability.clamp(0.0, 1.0),
                    meta,
                    effective_search_simulations,
                    effective_policy_weight,
                    effective_value_weight,
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
                    &history,
                    &search.candidates,
                    search.value_q,
                    config.policy_softmax_temp,
                    rng.unit_f32() < config.mirror_probability.clamp(0.0, 1.0),
                    move_meta,
                    effective_search_simulations,
                    effective_policy_weight,
                    effective_value_weight,
                );
                game_samples.push(sample);
            }
            append_history(&mut history, &position, mv);
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
        for (branch_position, branch_history, branch_rules, branch_ply) in repair_suffixes {
            let suffix = generate_normal_suffix(
                model,
                config,
                branch_position,
                branch_history,
                branch_rules,
                branch_ply,
                rng.next_u64(),
            );
            samples.extend(suffix.clone());
            games.push(suffix);
        }
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
        branch_reanalysis,
        branch_reanalysis_phase,
        phase_root_counts,
        endgame_audit,
        endgame_repair,
    }
}

fn phase_for_ply(ply: usize) -> usize {
    match ply {
        0..=39 => 0,
        40..=119 => 1,
        _ => 2,
    }
}

fn should_run_endgame_audit(config: &AzLoopConfig, ply: usize, rng: &mut SplitMix64) -> bool {
    phase_for_ply(ply) == 2
        && config.branch_reanalysis_simulations > 0
        && config.branch_endgame_audit_probability > 0.0
        && rng.unit_f32() < config.branch_endgame_audit_probability
}

fn random_endgame_scout_ply(config: &AzLoopConfig, rng: &mut SplitMix64) -> Option<usize> {
    if config.branch_endgame_repair_probability <= 0.0
        || rng.unit_f32() >= config.branch_endgame_repair_probability
        || config.max_plies <= 40
    {
        return None;
    }
    Some(40 + (rng.next_u64() as usize % (config.max_plies - 40)))
}

fn endgame_repair_accepts(shallow: &AzSearchResult, verified: &AzSearchResult) -> bool {
    // The scout is exploration only. Real terminal outcomes validate its suffix;
    // its Q estimate is never used as a value target or acceptance threshold.
    shallow.best_move != verified.best_move
}

fn endgame_scout_branch_moves(
    main: &AzSearchResult,
    scout: &AzSearchResult,
    verified_indices: &[usize],
) -> Vec<Move> {
    let mut indices = verified_indices.to_vec();
    indices.sort_by(|&left, &right| {
        scout.candidates[right]
            .q
            .total_cmp(&scout.candidates[left].q)
    });
    indices
        .into_iter()
        .map(|index| scout.candidates[index].mv)
        .filter(|&mv| Some(mv) != main.best_move)
        .take(ENDGAME_SCOUT_BRANCHES)
        .collect()
}

fn generate_normal_suffix(
    model: &AzNnue,
    config: &AzLoopConfig,
    mut position: Position,
    mut history: Vec<HistoryMove>,
    mut rule_history: Vec<crate::xiangqi::RuleHistoryEntry>,
    start_ply: usize,
    seed: u64,
) -> Vec<AzTrainingSample> {
    let mut rng = SplitMix64::new(seed);
    let mut samples = Vec::new();
    let mut result = None;
    for ply in start_ply..config.max_plies {
        let legal = position.legal_moves_with_rules(&rule_history);
        if legal.is_empty() {
            result = Some(if position.side_to_move() == Color::Red {
                -1.0
            } else {
                1.0
            });
            break;
        }
        let simulations = search_simulations_for_ply(config, &mut rng);
        let search = alphazero_search_with_history_and_rules(
            &position,
            &history,
            Some(rule_history.clone()),
            Some(legal),
            model,
            AzSearchLimits {
                simulations,
                seed: rng.next_u64(),
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
                draw_score: config.draw_score,
                moves_left_max_effect: config.moves_left_max_effect,
                moves_left_slope: config.moves_left_slope,
                moves_left_threshold: config.moves_left_threshold,
                moves_left_constant_factor: config.moves_left_constant_factor,
                moves_left_scaled_factor: config.moves_left_scaled_factor,
                moves_left_quadratic_factor: config.moves_left_quadratic_factor,
                value_scale: 1.0,
            },
        );
        let temperature = temperature_for_ply(config, ply);
        let Some(mv) = choose_selfplay_move(
            &search.candidates,
            temperature,
            config.temperature_value_cutoff,
            config.temperature_visit_offset,
            &mut rng,
        ) else {
            result = Some(0.0);
            break;
        };
        let meta = move_search_meta(
            &search.candidates,
            mv,
            search.value_q,
            config.generation_update,
            seed,
            ply,
        );
        samples.push(make_training_sample(
            &position,
            &history,
            &search.candidates,
            search.value_q,
            config.policy_softmax_temp,
            rng.unit_f32() < config.mirror_probability.clamp(0.0, 1.0),
            meta,
            simulations,
            policy_weight_for_search(config, simulations) * SCOUT_SUFFIX_TRAIN_WEIGHT,
            SCOUT_SUFFIX_TRAIN_WEIGHT,
        ));
        append_history(&mut history, &position, mv);
        let mover = position.side_to_move();
        position.make_move(mv);
        rule_history.push(position.rule_history_entry_after_moved(mover, mv.to as usize));
        if let Some(outcome) = position.rule_outcome_with_history(&rule_history) {
            result = Some(match outcome {
                RuleOutcome::Draw(_) => 0.0,
                RuleOutcome::Win(Color::Red) => 1.0,
                RuleOutcome::Win(Color::Black) => -1.0,
            });
            break;
        }
    }
    assign_value_targets(&mut samples, result.unwrap_or(0.0), config);
    assign_moves_left_targets(&mut samples, config.max_plies);
    samples
}

fn independent_leaf_verification(
    position: &Position,
    history: &[HistoryMove],
    rule_history: &[crate::xiangqi::RuleHistoryEntry],
    model: &AzNnue,
    config: &AzLoopConfig,
    shallow: &AzSearchResult,
    seed: u64,
    scout_simulations_per_leaf: Option<usize>,
    verify_all_legal_children: bool,
) -> LeafVerification {
    let (selected, capture_count, check_count, explorer_count) =
        select_leaf_verify_candidates(position, shallow, seed, verify_all_legal_children);
    if selected.is_empty() {
        return LeafVerification {
            search: shallow.clone(),
            candidate_count: 0,
            capture_count: 0,
            check_count: 0,
            explorer_count: 0,
            verified_indices: Vec::new(),
        };
    }
    let per_candidate = scout_simulations_per_leaf
        .unwrap_or_else(|| (config.branch_reanalysis_simulations / selected.len()).max(1));
    let mut verified_q = vec![None; shallow.candidates.len()];
    for &index in &selected {
        let mv = shallow.candidates[index].mv;
        let mut child = position.clone();
        let mut child_history = history.to_vec();
        append_history(&mut child_history, &child, mv);
        let mover = child.side_to_move();
        child.make_move(mv);
        let mut child_rules = rule_history.to_vec();
        child_rules.push(child.rule_history_entry_after_moved(mover, mv.to as usize));
        let mut limits =
            deterministic_branch_limits(config, seed ^ (index as u64).wrapping_mul(0x9E37_79B9));
        limits.simulations = per_candidate;
        let result = alphazero_search_with_history_and_rules(
            &child,
            &child_history,
            Some(child_rules),
            None,
            model,
            limits,
        );
        verified_q[index] = Some(-result.value_q);
    }

    let max_q = selected
        .iter()
        .filter_map(|&index| verified_q[index])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut verified_mass = vec![0.0f32; shallow.candidates.len()];
    let mut sum = 0.0;
    for &index in &selected {
        let weight = ((verified_q[index].unwrap_or(-1.0) - max_q) / 0.15).exp();
        verified_mass[index] = weight;
        sum += weight;
    }
    let mut candidates = shallow.candidates.clone();
    for (index, candidate) in candidates.iter_mut().enumerate() {
        if let Some(q) = verified_q[index] {
            candidate.q = q;
        }
        let verified_policy = if sum > 0.0 {
            verified_mass[index] / sum
        } else {
            0.0
        };
        candidate.policy = (1.0 - LEAF_VERIFY_POLICY_MIX) * candidate.policy
            + LEAF_VERIFY_POLICY_MIX * verified_policy;
    }
    let policy_sum = candidates
        .iter()
        .map(|candidate| candidate.policy)
        .sum::<f32>()
        .max(1e-8);
    for candidate in &mut candidates {
        candidate.policy /= policy_sum;
        candidate.visits = (candidate.policy * config.branch_reanalysis_simulations as f32) as u32;
    }
    let best_index = selected
        .iter()
        .copied()
        .max_by(|&left, &right| {
            verified_q[left]
                .unwrap_or(-1.0)
                .total_cmp(&verified_q[right].unwrap_or(-1.0))
        })
        .unwrap_or(0);
    let value_q = verified_q[best_index].unwrap_or(shallow.value_q);
    LeafVerification {
        search: AzSearchResult {
            best_move: Some(candidates[best_index].mv),
            value_q,
            value_cp: cp_from_q(value_q),
            value_wdl: shallow.value_wdl,
            simulations: per_candidate.saturating_mul(selected.len()),
            search_depth_avg: 0.0,
            search_depth_max: 0,
            search_depth_limit: per_candidate,
            search_depth_cutoffs: 0,
            candidates,
        },
        candidate_count: selected.len(),
        capture_count,
        check_count,
        explorer_count,
        verified_indices: selected,
    }
}

fn select_leaf_verify_candidates(
    position: &Position,
    shallow: &AzSearchResult,
    seed: u64,
    verify_all_legal_children: bool,
) -> (Vec<usize>, usize, usize, usize) {
    let mut captures = Vec::new();
    let mut checks = Vec::new();
    for (index, candidate) in shallow.candidates.iter().enumerate() {
        if position.piece_at(candidate.mv.to as usize).is_some() {
            captures.push(index);
        }
        let mut child = position.clone();
        child.make_move(candidate.mv);
        if child.in_check(child.side_to_move()) {
            checks.push(index);
        }
    }
    if verify_all_legal_children {
        return (
            (0..shallow.candidates.len()).collect(),
            captures.len(),
            checks.len(),
            0,
        );
    }
    let mut selected = Vec::new();
    let mut push = |index: usize| {
        if selected.len() < LEAF_VERIFY_MAX_CANDIDATES - LEAF_VERIFY_EXPLORER_SLOTS
            && !selected.contains(&index)
        {
            selected.push(index);
        }
    };
    let mut by_visits = (0..shallow.candidates.len()).collect::<Vec<_>>();
    by_visits.sort_by_key(|&index| std::cmp::Reverse(shallow.candidates[index].visits));
    for index in by_visits.into_iter().take(2) {
        push(index);
    }
    // Preserve the main search's selected move even in tactically busy nodes.
    if let Some(index) = shallow.best_move.and_then(|mv| {
        shallow
            .candidates
            .iter()
            .position(|candidate| candidate.mv == mv)
    }) {
        push(index);
    }
    // Captures and checks are then given priority over a prior-only candidate.
    for &index in captures.iter().chain(checks.iter()) {
        push(index);
    }
    let mut by_prior = (0..shallow.candidates.len()).collect::<Vec<_>>();
    by_prior.sort_by(|&left, &right| {
        shallow.candidates[right]
            .raw_prior
            .total_cmp(&shallow.candidates[left].raw_prior)
    });
    if let Some(index) = by_prior.first() {
        push(*index);
    }
    let mut by_q = (0..shallow.candidates.len()).collect::<Vec<_>>();
    by_q.sort_by(|&left, &right| {
        shallow.candidates[right]
            .q
            .total_cmp(&shallow.candidates[left].q)
    });
    if let Some(index) = by_q.first() {
        push(*index);
    }
    let selected_captures = selected
        .iter()
        .filter(|&&index| captures.contains(&index))
        .count();
    let selected_checks = selected
        .iter()
        .filter(|&&index| checks.contains(&index))
        .count();
    let explorer_pool = (0..shallow.candidates.len())
        .filter(|index| !selected.contains(index))
        .collect::<Vec<_>>();
    let explorer_count = usize::from(!explorer_pool.is_empty());
    if let Some(&index) =
        explorer_pool.get((seed as usize).wrapping_rem(explorer_pool.len().max(1)))
    {
        selected.push(index);
    }
    (selected, selected_captures, selected_checks, explorer_count)
}

fn deterministic_branch_limits(config: &AzLoopConfig, seed: u64) -> AzSearchLimits {
    AzSearchLimits {
        simulations: config.branch_reanalysis_simulations,
        seed,
        cpuct: config.cpuct,
        cpuct_at_root: config.cpuct_at_root,
        cpuct_base: config.cpuct_base,
        cpuct_factor: config.cpuct_factor,
        cpuct_base_at_root: config.cpuct_base_at_root,
        cpuct_factor_at_root: config.cpuct_factor_at_root,
        max_depth: 0,
        root_dirichlet_alpha: 0.0,
        root_exploration_fraction: 0.0,
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
    }
}

fn policy_kl(source: &[AzCandidate], target: &[AzCandidate]) -> f32 {
    source
        .iter()
        .filter_map(|left| {
            let p = left.policy.max(0.0);
            (p > 1.0e-8).then(|| {
                let q = target
                    .iter()
                    .find(|right| right.mv == left.mv)
                    .map_or(1.0e-8, |right| right.policy.max(1.0e-8));
                p * (p / q).ln()
            })
        })
        .sum()
}

fn is_high_confidence_branch_flip(q_advantage: f32, deep_policy: f32) -> bool {
    q_advantage >= HIGH_CONFIDENCE_Q_ADVANTAGE && deep_policy >= HIGH_CONFIDENCE_POLICY_MASS
}

fn search_simulations_for_ply(config: &AzLoopConfig, rng: &mut SplitMix64) -> usize {
    let high = config.simulations.max(1);
    let low = config.low_simulations.max(1).min(high);
    if low >= high {
        return high;
    }
    if rng.unit_f32() < config.low_simulation_probability.clamp(0.0, 1.0) {
        low
    } else {
        high
    }
}

fn policy_weight_for_search(config: &AzLoopConfig, search_simulations: usize) -> f32 {
    if search_simulations < config.simulations.max(1) {
        config.low_simulation_policy_weight.max(0.0)
    } else {
        1.0
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
    search_simulations: usize,
    policy_weight: f32,
    value_weight: f32,
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
        policy_weight: policy_weight.max(0.0),
        value_weight: value_weight.max(0.0),
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

    fn candidate(mv: Move, policy: f32) -> AzCandidate {
        AzCandidate {
            mv,
            visits: (policy * 100.0) as u32,
            q: 0.0,
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
            &[],
            &candidates,
            0.0,
            1.45,
            true,
            AzSampleMeta::default(),
            1,
            1.0,
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
