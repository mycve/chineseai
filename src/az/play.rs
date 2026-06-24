use std::sync::Arc;

use rayon::prelude::*;

use crate::nnue::{
    HistoryMove, canonical_move, extract_sparse_features_az_canonical, mirror_file_move,
    mirror_sparse_features_az_absolute_file,
};
use crate::xiangqi::{Color, Move, Position, RuleDrawReason, RuleOutcome};

use super::gumbel::append_history;
use super::{
    AzCandidate, AzLoopConfig, AzNnue, AzSampleMeta, AzSearchLimits, AzTrainingSample,
    GumbelSearchConfig, SplitMix64, dense_move_index, gumbel_search_with_history_and_rules,
    scalar_value_to_wdl_target,
};

#[derive(Clone, Copy, Debug)]
struct MoveSearchMeta {
    sample: AzSampleMeta,
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
    pub prior_entropy_sum: f32,
    pub target_entropy_sum: f32,
    pub prior_top1_sum: f32,
    pub prior_top2_sum: f32,
    pub target_top1_sum: f32,
    pub target_top2_sum: f32,
    pub q_gap_sum: f32,
    pub q_top1_abs_sum: f32,
    pub legal_actions_sum: usize,
    pub visited_actions_sum: usize,
    pub shape_count: usize,
    pub sampled_moves: usize,
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
        self.prior_entropy_sum += other.prior_entropy_sum;
        self.target_entropy_sum += other.target_entropy_sum;
        self.prior_top1_sum += other.prior_top1_sum;
        self.prior_top2_sum += other.prior_top2_sum;
        self.target_top1_sum += other.target_top1_sum;
        self.target_top2_sum += other.target_top2_sum;
        self.q_gap_sum += other.q_gap_sum;
        self.q_top1_abs_sum += other.q_top1_abs_sum;
        self.legal_actions_sum += other.legal_actions_sum;
        self.visited_actions_sum += other.visited_actions_sum;
        self.shape_count += other.shape_count;
        self.sampled_moves += other.sampled_moves;
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
        merged.prior_entropy_sum += chunk.prior_entropy_sum;
        merged.target_entropy_sum += chunk.target_entropy_sum;
        merged.prior_top1_sum += chunk.prior_top1_sum;
        merged.prior_top2_sum += chunk.prior_top2_sum;
        merged.target_top1_sum += chunk.target_top1_sum;
        merged.target_top2_sum += chunk.target_top2_sum;
        merged.q_gap_sum += chunk.q_gap_sum;
        merged.q_top1_abs_sum += chunk.q_top1_abs_sum;
        merged.legal_actions_sum += chunk.legal_actions_sum;
        merged.visited_actions_sum += chunk.visited_actions_sum;
        merged.shape_count += chunk.shape_count;
        merged.sampled_moves += chunk.sampled_moves;
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
    let mut prior_entropy_sum = 0.0f32;
    let mut target_entropy_sum = 0.0f32;
    let mut prior_top1_sum = 0.0f32;
    let mut prior_top2_sum = 0.0f32;
    let mut target_top1_sum = 0.0f32;
    let mut target_top2_sum = 0.0f32;
    let mut q_gap_sum = 0.0f32;
    let mut q_top1_abs_sum = 0.0f32;
    let mut legal_actions_sum = 0usize;
    let mut visited_actions_sum = 0usize;
    let mut shape_count = 0usize;
    let mut sampled_moves = 0usize;
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
                max_depth: 0,
                value_scale: 1.0,
                ..AzSearchLimits::default()
            };
            let search = gumbel_search_with_history_and_rules(
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
            );
            let shape = policy_shape_stats(&search.candidates);
            prior_entropy_sum += shape.prior_entropy;
            target_entropy_sum += shape.target_entropy;
            prior_top1_sum += shape.prior_top1;
            prior_top2_sum += shape.prior_top2;
            target_top1_sum += shape.target_top1;
            target_top2_sum += shape.target_top2;
            q_gap_sum += shape.q_gap;
            q_top1_abs_sum += shape.q_top1_abs;
            legal_actions_sum += shape.legal_actions;
            visited_actions_sum += shape.visited_actions;
            shape_count += 1;
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
            let mv_opt =
                choose_selfplay_move(&search.candidates, search.best_move, ply, config, &mut rng);
            let Some(mv) = mv_opt else {
                result = Some(0.0);
                break;
            };
            let bootstrap_value = search
                .candidates
                .iter()
                .find(|candidate| candidate.mv == mv)
                .map(|candidate| candidate.q)
                .unwrap_or(search.value_q);
            let move_meta = move_search_meta(
                &search.candidates,
                mv,
                bootstrap_value,
                config.generation_update,
                config.seed ^ game_index as u64,
                ply,
            );
            sampled_moves += 1;
            let side_sign = if position.side_to_move() == Color::Red {
                1.0
            } else {
                -1.0
            };
            let _ = side_sign;
            game_samples.push(make_training_sample(
                &position,
                &history,
                &search.candidates,
                bootstrap_value,
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

        assign_value_targets(&mut game_samples, result, config);
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
        prior_entropy_sum,
        target_entropy_sum,
        prior_top1_sum,
        prior_top2_sum,
        target_top1_sum,
        target_top2_sum,
        q_gap_sum,
        q_top1_abs_sum,
        legal_actions_sum,
        visited_actions_sum,
        shape_count,
        sampled_moves,
        terminal,
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct PolicyShapeStats {
    prior_entropy: f32,
    target_entropy: f32,
    prior_top1: f32,
    prior_top2: f32,
    target_top1: f32,
    target_top2: f32,
    q_gap: f32,
    q_top1_abs: f32,
    legal_actions: usize,
    visited_actions: usize,
}

fn policy_shape_stats(candidates: &[AzCandidate]) -> PolicyShapeStats {
    let mut prior_top = [0.0f32; 2];
    let mut target_top = [0.0f32; 2];
    let mut q_top = [f32::NEG_INFINITY; 2];
    let mut prior_entropy = 0.0f32;
    let mut target_entropy = 0.0f32;
    let mut visited_actions = 0usize;
    for candidate in candidates {
        let prior = candidate.raw_prior.max(0.0);
        let target = candidate.policy.max(0.0);
        insert_top2(prior, &mut prior_top);
        insert_top2(target, &mut target_top);
        if prior > 0.0 {
            prior_entropy -= prior * prior.ln();
        }
        if target > 0.0 {
            target_entropy -= target * target.ln();
        }
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
        prior_entropy,
        target_entropy,
        prior_top1: prior_top[0],
        prior_top2: prior_top[0] + prior_top[1],
        target_top1: target_top[0],
        target_top2: target_top[0] + target_top[1],
        q_gap,
        q_top1_abs,
        legal_actions: candidates.len(),
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
    mirror_file: bool,
    meta: AzSampleMeta,
) -> AzTrainingSample {
    let side = position.side_to_move();
    let side_sign = if side == Color::Red { 1.0 } else { -1.0 };
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
        .map(|candidate| candidate.policy.max(1e-12))
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
    MoveSearchMeta { sample: meta }
}

fn assign_value_targets(
    samples: &mut [AzTrainingSample],
    game_result_red: f32,
    config: &AzLoopConfig,
) {
    let lambda = config.td_lambda.clamp(0.0, 1.0);
    let result_red = game_result_red.clamp(-1.0, 1.0);
    let mut td_red = result_red;
    for sample in samples.iter_mut().rev() {
        let bootstrap_red = (sample.value * sample.side_sign).clamp(-1.0, 1.0);
        td_red = ((1.0 - lambda) * bootstrap_red + lambda * td_red).clamp(-1.0, 1.0);
        let side_result = (result_red * sample.side_sign).clamp(-1.0, 1.0);
        sample.value_wdl = scalar_value_to_wdl_target(side_result);
        sample.value = (td_red * sample.side_sign).clamp(-1.0, 1.0);
    }
}

fn choose_selfplay_move(
    candidates: &[AzCandidate],
    best_move: Option<Move>,
    ply: usize,
    config: &AzLoopConfig,
    rng: &mut SplitMix64,
) -> Option<Move> {
    if ply >= config.selfplay_sampling_plies {
        return best_move;
    }
    let temperature = config.selfplay_sampling_temperature.max(1e-3);
    let mut total = 0.0f32;
    let weights = candidates
        .iter()
        .map(|candidate| {
            if candidate.visits == 0 {
                0.0
            } else {
                (candidate.visits as f32).powf(1.0 / temperature)
            }
        })
        .inspect(|weight| total += *weight)
        .collect::<Vec<_>>();
    if total <= 0.0 {
        return best_move;
    }
    let mut threshold = rng.unit_f32() * total;
    for (candidate, weight) in candidates.iter().zip(weights) {
        threshold -= weight;
        if threshold <= 0.0 {
            return Some(candidate.mv);
        }
    }
    candidates
        .iter()
        .rev()
        .find(|candidate| candidate.visits > 0)
        .map(|candidate| candidate.mv)
        .or(best_move)
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

#[derive(Clone, Copy, Debug)]
pub struct AzArenaConfig {
    pub simulations: usize,
    pub max_plies: usize,
    pub gumbel_actions: usize,
    pub gumbel_value_scale: f32,
    pub gumbel_maxvisit_init: f32,
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
            GumbelSearchConfig {
                max_num_considered_actions: config.gumbel_actions,
                gumbel_scale: 0.0,
                value_scale: config.gumbel_value_scale,
                maxvisit_init: config.gumbel_maxvisit_init,
            },
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
            GumbelSearchConfig {
                max_num_considered_actions: config.gumbel_actions,
                gumbel_scale: 0.0,
                value_scale: config.gumbel_value_scale,
                maxvisit_init: config.gumbel_maxvisit_init,
            },
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
    gumbel: GumbelSearchConfig,
    seed: u64,
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
        let result = gumbel_search_with_history_and_rules(
            &position,
            &history,
            Some(rule_history.clone()),
            Some(legal),
            model,
            AzSearchLimits {
                simulations,
                seed: seed ^ ((ply as u64) << 32),
                max_depth: 0,
                value_scale: 1.0,
            },
            gumbel,
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
            gumbel_score: 0.0,
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
}
