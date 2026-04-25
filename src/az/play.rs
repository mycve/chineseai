use std::sync::Arc;
use std::thread;

use crate::nnue::{
    HistoryMove, extract_sparse_features_v4, mirror_file_move, mirror_sparse_features_file,
};
use crate::xiangqi::{Color, Move, Position, RuleDrawReason, RuleOutcome};

use super::alphazero::append_history;
use super::{
    AzCandidate, AzLoopConfig, AzNnue, AzSearchLimits, AzTrainingSample, BOARD_HISTORY_FRAMES,
    BOARD_PLANES_SIZE, SplitMix64, VALUE_SCALE_CP, alphazero_search_with_history_and_rules,
    dense_move_index, extract_board_planes,
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

    pub fn elo(&self) -> f32 {
        let total = self.total_games();
        if total == 0 {
            return 0.0;
        }
        let score_rate = ((self.score() + 0.5) / (total as f32 + 1.0)).clamp(1e-6, 1.0 - 1e-6);
        400.0 * (score_rate / (1.0 - score_rate)).log10()
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
    pub temperature_early_entropy_sum: f32,
    pub temperature_early_entropy_count: usize,
    pub temperature_mid_entropy_sum: f32,
    pub temperature_mid_entropy_count: usize,
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
        self.temperature_early_entropy_sum += other.temperature_early_entropy_sum;
        self.temperature_early_entropy_count += other.temperature_early_entropy_count;
        self.temperature_mid_entropy_sum += other.temperature_mid_entropy_sum;
        self.temperature_mid_entropy_count += other.temperature_mid_entropy_count;
        self.terminal.add_assign(&other.terminal);
    }
}

pub fn generate_selfplay_data(model: &AzNnue, config: &AzLoopConfig) -> AzSelfplayData {
    let workers = config.workers.max(1).min(config.games.max(1));
    if workers == 1 || config.games <= 1 {
        return generate_selfplay_chunk(model, config);
    }

    let shared_model = Arc::new(model.clone());
    let mut handles = Vec::with_capacity(workers);
    for worker in 0..workers {
        let games = config.games / workers + usize::from(worker < config.games % workers);
        if games == 0 {
            continue;
        }

        let worker_model = Arc::clone(&shared_model);
        let mut worker_config = config.clone();
        worker_config.games = games;
        worker_config.workers = 1;
        worker_config.seed ^= (worker as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        handles.push(thread::spawn(move || {
            generate_selfplay_chunk(&worker_model, &worker_config)
        }));
    }

    let mut merged = AzSelfplayData::default();
    for handle in handles {
        let chunk = handle.join().expect("selfplay worker panicked");
        merged.samples.extend(chunk.samples);
        merged.games.extend(chunk.games);
        merged.red_wins += chunk.red_wins;
        merged.black_wins += chunk.black_wins;
        merged.draws += chunk.draws;
        merged.plies_total += chunk.plies_total;
        merged.temperature_early_entropy_sum += chunk.temperature_early_entropy_sum;
        merged.temperature_early_entropy_count += chunk.temperature_early_entropy_count;
        merged.temperature_mid_entropy_sum += chunk.temperature_mid_entropy_sum;
        merged.temperature_mid_entropy_count += chunk.temperature_mid_entropy_count;
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
    let mut temperature_early_entropy_sum = 0.0f32;
    let mut temperature_early_entropy_count = 0usize;
    let mut temperature_mid_entropy_sum = 0.0f32;
    let mut temperature_mid_entropy_count = 0usize;
    let mut terminal = AzTerminalStats::default();

    for game_index in 0..config.games {
        let mut position = Position::startpos();
        let mut history = Vec::new();
        let mut rule_history = position.initial_rule_history();
        let mut game_samples = Vec::new();
        let mut result = None;
        let mut plies = 0usize;

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

            let search = alphazero_search_with_history_and_rules(
                &position,
                &history,
                Some(rule_history.clone()),
                Some(legal),
                model,
                AzSearchLimits {
                    simulations: config.simulations,
                    seed: rng.next() ^ ((game_index as u64) << 32) ^ ply as u64,
                    cpuct: config.cpuct,
                    workers: 1,
                    root_dirichlet_alpha: config.root_dirichlet_alpha,
                    root_exploration_fraction: config.root_exploration_fraction,
                },
            );
            let entropy = policy_entropy(&search.candidates);
            let split = config.temperature_decay_plies.max(2).div_ceil(2);
            if ply < split {
                temperature_early_entropy_sum += entropy;
                temperature_early_entropy_count += 1;
            } else if ply < config.temperature_decay_plies.max(split + 1) {
                temperature_mid_entropy_sum += entropy;
                temperature_mid_entropy_count += 1;
            }
            let temperature = temperature_for_ply(config, ply);
            let mv_opt = if temperature <= 1e-6 {
                search
                    .best_move
                    .or_else(|| choose_selfplay_move(&search.candidates, temperature, &mut rng))
            } else {
                choose_selfplay_move(&search.candidates, temperature, &mut rng)
            };
            let Some(mv) = mv_opt else {
                result = Some(0.0);
                break;
            };
            game_samples.push(make_training_sample(
                &position,
                &history,
                &search.candidates,
                search.value_cp as f32 / VALUE_SCALE_CP,
                rng.unit_f32() < config.mirror_probability.clamp(0.0, 1.0),
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

        assign_td_lambda_value_targets(&mut game_samples, result, config.td_lambda);
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
        temperature_early_entropy_sum,
        temperature_early_entropy_count,
        temperature_mid_entropy_sum,
        temperature_mid_entropy_count,
        terminal,
    }
}

fn make_training_sample(
    position: &Position,
    history: &[HistoryMove],
    candidates: &[AzCandidate],
    value: f32,
    mirror_file: bool,
) -> AzTrainingSample {
    let total_policy = candidates
        .iter()
        .map(|candidate| candidate.policy.max(0.0))
        .sum::<f32>()
        .max(1.0);
    let mut features = extract_sparse_features_v4(position, history);
    let mut moves = candidates
        .iter()
        .map(|candidate| candidate.mv)
        .collect::<Vec<_>>();
    if mirror_file {
        mirror_sparse_features_file(&mut features);
        for mv in &mut moves {
            *mv = mirror_file_move(*mv);
        }
    }
    let move_indices = moves.iter().copied().map(dense_move_index).collect();
    let mut board = Vec::new();
    extract_board_planes(position, history, &mut board);
    if mirror_file {
        let mut mirrored = board.clone();
        for frame in 0..BOARD_HISTORY_FRAMES {
            let base = frame * BOARD_PLANES_SIZE;
            for sq in 0..BOARD_PLANES_SIZE {
                mirrored[base + crate::nnue::mirror_file_square(sq)] = board[base + sq];
            }
        }
        board = mirrored;
    }
    AzTrainingSample {
        features,
        board,
        move_indices,
        policy: candidates
            .iter()
            .map(|candidate| candidate.policy.max(0.0) / total_policy)
            .collect(),
        value: value.clamp(-1.0, 1.0),
        side_sign: if position.side_to_move() == Color::Red {
            1.0
        } else {
            -1.0
        },
    }
}

pub(super) fn assign_terminal_value_targets(samples: &mut [AzTrainingSample], game_result: f32) {
    for sample in samples.iter_mut() {
        sample.value = (game_result * sample.side_sign).clamp(-1.0, 1.0);
    }
}

pub(super) fn assign_td_lambda_value_targets(
    samples: &mut [AzTrainingSample],
    game_result: f32,
    td_lambda: f32,
) {
    if samples.is_empty() {
        return;
    }

    let lambda = td_lambda.clamp(0.0, 1.0);
    if lambda >= 1.0 - 1e-6 {
        assign_terminal_value_targets(samples, game_result);
        return;
    }

    let bootstrap_red = samples
        .iter()
        .map(|sample| (sample.value * sample.side_sign).clamp(-1.0, 1.0))
        .collect::<Vec<_>>();
    let mut next_return_red = game_result.clamp(-1.0, 1.0);
    for index in (0..samples.len()).rev() {
        if index + 1 < samples.len() {
            next_return_red = ((1.0 - lambda) * bootstrap_red[index + 1]
                + lambda * next_return_red)
                .clamp(-1.0, 1.0);
        }
        samples[index].value = (next_return_red * samples[index].side_sign).clamp(-1.0, 1.0);
    }
}

fn temperature_for_ply(config: &AzLoopConfig, ply: usize) -> f32 {
    if config.temperature_decay_plies == 0 || ply >= config.temperature_decay_plies {
        return config.temperature_end;
    }
    let progress = ply as f32 / config.temperature_decay_plies as f32;
    config.temperature_start + (config.temperature_end - config.temperature_start) * progress
}

fn choose_selfplay_move(
    candidates: &[AzCandidate],
    temperature: f32,
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

    let inv_temperature = 1.0 / temperature.max(1e-3);
    let weights = candidates
        .iter()
        .map(|candidate| candidate.policy.max(1e-9).powf(inv_temperature))
        .collect::<Vec<_>>();
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

fn policy_entropy(candidates: &[AzCandidate]) -> f32 {
    let total = candidates
        .iter()
        .map(|candidate| candidate.policy.max(0.0))
        .sum::<f32>();
    if total <= 0.0 {
        return 0.0;
    }
    candidates
        .iter()
        .map(|candidate| candidate.policy.max(0.0) / total)
        .filter(|probability| *probability > 1e-9)
        .map(|probability| -probability * probability.ln())
        .sum()
}

pub fn play_arena_games(
    candidate: &AzNnue,
    baseline: &AzNnue,
    simulations: usize,
    max_plies: usize,
    games_as_red: usize,
    games_as_black: usize,
    seed: u64,
    cpuct: f32,
) -> AzArenaReport {
    play_arena_games_from_positions(
        candidate,
        baseline,
        &[],
        simulations,
        max_plies,
        games_as_red,
        games_as_black,
        seed,
        cpuct,
    )
}

pub fn play_arena_games_from_positions(
    candidate: &AzNnue,
    baseline: &AzNnue,
    positions: &[Position],
    simulations: usize,
    max_plies: usize,
    games_as_red: usize,
    games_as_black: usize,
    seed: u64,
    cpuct: f32,
) -> AzArenaReport {
    let mut report = AzArenaReport::default();
    let mut game_seed = seed;
    for game_index in 0..games_as_red {
        let position = arena_start_position(positions, game_index);
        let outcome = play_arena_game(
            &position,
            candidate,
            baseline,
            simulations,
            max_plies,
            game_seed,
            cpuct,
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
    for game_index in 0..games_as_black {
        let position = arena_start_position(positions, game_index);
        let outcome = play_arena_game(
            &position,
            baseline,
            candidate,
            simulations,
            max_plies,
            game_seed,
            cpuct,
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
        positions[game_index % positions.len()].clone()
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
                workers: 1,
                root_dirichlet_alpha: 0.0,
                root_exploration_fraction: 0.0,
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
