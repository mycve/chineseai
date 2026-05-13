use std::sync::Arc;
use std::thread;

use crate::nnue::{
    HistoryMove, canonical_move, extract_sparse_features_v4_canonical, mirror_file_move,
    mirror_sparse_features_file,
};
use crate::xiangqi::{Color, Move, Position, RuleHistoryEntry, summarize_rule_state};

use super::{
    AzCandidate, AzEnv, AzGameEndReason, AzLoopConfig, AzNnue, AzRuleSet, AzSearchLimits,
    AzTrainingSample, SplitMix64, VALUE_SCALE_CP, alphazero_search_env, dense_move_index,
};

#[derive(Clone, Copy, Debug, Default)]
pub struct AzTerminalStats {
    pub no_legal_moves: usize,
    pub no_attacking_material: usize,
    pub halfmove120: usize,
    pub repetition: usize,
    pub repetition_quiet: usize,
    pub repetition_current_check: usize,
    pub repetition_current_chase: usize,
    pub repetition_rule_pressure: usize,
    pub mutual_long_check: usize,
    pub mutual_long_chase: usize,
    pub rule_win_red: usize,
    pub rule_win_black: usize,
    pub max_plies: usize,
}

impl AzTerminalStats {
    pub fn add_assign(&mut self, other: &Self) {
        self.no_legal_moves += other.no_legal_moves;
        self.no_attacking_material += other.no_attacking_material;
        self.halfmove120 += other.halfmove120;
        self.repetition += other.repetition;
        self.repetition_quiet += other.repetition_quiet;
        self.repetition_current_check += other.repetition_current_check;
        self.repetition_current_chase += other.repetition_current_chase;
        self.repetition_rule_pressure += other.repetition_rule_pressure;
        self.mutual_long_check += other.mutual_long_check;
        self.mutual_long_chase += other.mutual_long_chase;
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
    pub draw_max_plies: usize,
    pub draw_no_attacking_material: usize,
    pub draw_halfmove120: usize,
    pub draw_repetition: usize,
    pub draw_mutual_long_check: usize,
    pub draw_mutual_long_chase: usize,
    pub draw_search_empty: usize,
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
        self.draw_max_plies += other.draw_max_plies;
        self.draw_no_attacking_material += other.draw_no_attacking_material;
        self.draw_halfmove120 += other.draw_halfmove120;
        self.draw_repetition += other.draw_repetition;
        self.draw_mutual_long_check += other.draw_mutual_long_check;
        self.draw_mutual_long_chase += other.draw_mutual_long_chase;
        self.draw_search_empty += other.draw_search_empty;
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AzArenaDrawReason {
    MaxPlies,
    NoAttackingMaterial,
    Halfmove120,
    Repetition,
    MutualLongCheck,
    MutualLongChase,
    SearchEmpty,
}

#[derive(Clone, Copy, Debug)]
struct AzArenaGameOutcome {
    result: f32,
    draw_reason: Option<AzArenaDrawReason>,
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
        let mut env = AzEnv::startpos(AzRuleSet::Full);
        let mut game_samples = Vec::new();
        let mut result = None;
        let mut plies = 0usize;

        for ply in 0..config.max_plies {
            plies = ply + 1;
            let legal = env.legal_moves();
            if legal.is_empty() {
                result = Some(if env.position().side_to_move() == Color::Red {
                    -1.0
                } else {
                    1.0
                });
                terminal.no_legal_moves += 1;
                break;
            }

            let search = alphazero_search_env(
                &env,
                Some(legal),
                model,
                AzSearchLimits {
                    simulations: config.simulations,
                    seed: rng.next_u64() ^ ((game_index as u64) << 32) ^ ply as u64,
                    cpuct: config.cpuct,
                    root_dirichlet_alpha: config.root_dirichlet_alpha,
                    root_exploration_fraction: config.root_exploration_fraction,
                    algorithm: config.search_algorithm,
                    gumbel: config.gumbel,
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
                env.position(),
                env.history(),
                env.rule_history(),
                &search.candidates,
                search.value_cp as f32 / VALUE_SCALE_CP,
                rng.unit_f32() < config.mirror_probability.clamp(0.0, 1.0),
            ));
            env.make_move(mv);

            if let Some(outcome) = env.game_result_details() {
                result = Some(outcome.result);
                record_terminal_reason(&mut terminal, outcome.reason, env.rule_history());
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

        assign_terminal_value_targets(&mut game_samples, result);
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

fn record_terminal_reason(
    stats: &mut AzTerminalStats,
    reason: AzGameEndReason,
    rule_history: &[RuleHistoryEntry],
) {
    match reason {
        AzGameEndReason::NoLegalMoves => stats.no_legal_moves += 1,
        AzGameEndReason::NoAttackingMaterial => stats.no_attacking_material += 1,
        AzGameEndReason::Halfmove120 => stats.halfmove120 += 1,
        AzGameEndReason::Repetition => {
            stats.repetition += 1;
            record_repetition_detail(stats, rule_history);
        }
        AzGameEndReason::MutualLongCheck => stats.mutual_long_check += 1,
        AzGameEndReason::MutualLongChase => stats.mutual_long_chase += 1,
        AzGameEndReason::RuleWinRed => stats.rule_win_red += 1,
        AzGameEndReason::RuleWinBlack => stats.rule_win_black += 1,
    }
}

fn record_repetition_detail(stats: &mut AzTerminalStats, rule_history: &[RuleHistoryEntry]) {
    let Some(current) = rule_history.last() else {
        stats.repetition_quiet += 1;
        return;
    };
    if current.gives_check {
        stats.repetition_current_check += 1;
        return;
    }
    if current.chased_mask != 0 {
        stats.repetition_current_chase += 1;
        return;
    }
    let summary = summarize_rule_state(rule_history, current.side_to_move);
    if summary
        .own_check
        .max(summary.own_chase)
        .max(summary.own_alt)
        .max(summary.enemy_check)
        .max(summary.enemy_chase)
        .max(summary.enemy_alt)
        >= 2
    {
        stats.repetition_rule_pressure += 1;
    } else {
        stats.repetition_quiet += 1;
    }
}

fn make_training_sample(
    position: &Position,
    history: &[HistoryMove],
    _rule_history: &[RuleHistoryEntry],
    candidates: &[AzCandidate],
    value: f32,
    mirror_file: bool,
) -> AzTrainingSample {
    let total_policy = candidates
        .iter()
        .map(|candidate| candidate.policy.max(0.0))
        .sum::<f32>()
        .max(1.0);
    let side = position.side_to_move();
    let mut features = extract_sparse_features_v4_canonical(position, history);
    let mut moves = candidates
        .iter()
        .map(|candidate| canonical_move(side, candidate.mv))
        .collect::<Vec<_>>();
    if mirror_file {
        mirror_sparse_features_file(&mut features);
        for mv in &mut moves {
            *mv = mirror_file_move(*mv);
        }
    }
    let move_indices = moves.iter().copied().map(dense_move_index).collect();
    AzTrainingSample {
        features,
        board: Vec::new(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnue::mirror_file_move;

    #[test]
    fn mirror_augmentation_matches_explicitly_mirrored_position() {
        let left = Position::from_fen("4k4/9/9/9/3pp4/5P3/9/9/9/4K4 w").unwrap();
        let right = Position::from_fen("4k4/9/9/9/4pp3/3P5/9/9/9/4K4 w").unwrap();
        let left_move = left.legal_moves()[0];
        let right_move = mirror_file_move(left_move);
        assert!(right.is_legal_move(right_move));

        let left_candidates = [AzCandidate {
            mv: left_move,
            visits: 8,
            q: 0.25,
            prior: 0.4,
            policy: 0.7,
        }];
        let right_candidates = [AzCandidate {
            mv: right_move,
            visits: 8,
            q: 0.25,
            prior: 0.4,
            policy: 0.7,
        }];

        let left_rules = left.initial_rule_history();
        let right_rules = right.initial_rule_history();
        let augmented = make_training_sample(&left, &[], &left_rules, &left_candidates, 0.35, true);
        let explicit =
            make_training_sample(&right, &[], &right_rules, &right_candidates, 0.35, false);

        assert_eq!(augmented.features, explicit.features);
        assert_eq!(augmented.board, explicit.board);
        assert_eq!(augmented.move_indices, explicit.move_indices);
        assert_eq!(augmented.policy, explicit.policy);
        assert_eq!(augmented.value, explicit.value);
        assert_eq!(augmented.side_sign, explicit.side_sign);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AzArenaConfig {
    pub simulations: usize,
    pub max_plies: usize,
    pub games_as_red: usize,
    pub games_as_black: usize,
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
        let position = arena_start_position(positions, config.seed, game_index);
        let outcome = play_arena_game(
            &position,
            candidate,
            baseline,
            config.simulations,
            config.max_plies,
            game_seed,
            config.cpuct,
        );
        match outcome.result.total_cmp(&0.0) {
            std::cmp::Ordering::Greater => {
                report.wins += 1;
                report.wins_as_red += 1;
            }
            std::cmp::Ordering::Less => {
                report.losses += 1;
                report.losses_as_red += 1;
            }
            std::cmp::Ordering::Equal => record_arena_draw(&mut report, outcome.draw_reason),
        }
        game_seed = game_seed.wrapping_add(1);
    }
    for game_index in 0..config.games_as_black {
        let position = arena_start_position(positions, config.seed, game_index);
        let outcome = play_arena_game(
            &position,
            baseline,
            candidate,
            config.simulations,
            config.max_plies,
            game_seed,
            config.cpuct,
        );
        match outcome.result.total_cmp(&0.0) {
            std::cmp::Ordering::Greater => {
                report.losses += 1;
                report.losses_as_black += 1;
            }
            std::cmp::Ordering::Less => {
                report.wins += 1;
                report.wins_as_black += 1;
            }
            std::cmp::Ordering::Equal => record_arena_draw(&mut report, outcome.draw_reason),
        }
        game_seed = game_seed.wrapping_add(1);
    }
    report
}

fn record_arena_draw(report: &mut AzArenaReport, reason: Option<AzArenaDrawReason>) {
    report.draws += 1;
    match reason.unwrap_or(AzArenaDrawReason::MaxPlies) {
        AzArenaDrawReason::MaxPlies => report.draw_max_plies += 1,
        AzArenaDrawReason::NoAttackingMaterial => report.draw_no_attacking_material += 1,
        AzArenaDrawReason::Halfmove120 => report.draw_halfmove120 += 1,
        AzArenaDrawReason::Repetition => report.draw_repetition += 1,
        AzArenaDrawReason::MutualLongCheck => report.draw_mutual_long_check += 1,
        AzArenaDrawReason::MutualLongChase => report.draw_mutual_long_chase += 1,
        AzArenaDrawReason::SearchEmpty => report.draw_search_empty += 1,
    }
}

fn arena_start_position(positions: &[Position], seed: u64, game_index: usize) -> Position {
    if positions.is_empty() {
        Position::startpos()
    } else {
        let mut rng =
            SplitMix64::new(seed ^ (game_index as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
        let index = (rng.next_u64() as usize) % positions.len();
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
) -> AzArenaGameOutcome {
    let mut env = AzEnv::from_position(initial_position.clone(), AzRuleSet::Full);
    for ply in 0..max_plies {
        let legal = env.legal_moves();
        if legal.is_empty() {
            return AzArenaGameOutcome {
                result: if env.position().side_to_move() == Color::Red {
                    -1.0
                } else {
                    1.0
                },
                draw_reason: None,
            };
        }
        let model = if env.position().side_to_move() == Color::Red {
            red_model
        } else {
            black_model
        };
        let result = alphazero_search_env(
            &env,
            Some(legal),
            model,
            AzSearchLimits {
                simulations,
                seed: seed ^ ((ply as u64) << 32),
                cpuct,
                root_dirichlet_alpha: 0.0,
                root_exploration_fraction: 0.0,
                algorithm: Default::default(),
                gumbel: Default::default(),
            },
        );
        let Some(mv) = result.best_move else {
            return AzArenaGameOutcome {
                result: 0.0,
                draw_reason: Some(AzArenaDrawReason::SearchEmpty),
            };
        };
        env.make_move(mv);

        if let Some(outcome) = env.game_result_details() {
            return AzArenaGameOutcome {
                result: outcome.result,
                draw_reason: arena_draw_reason(outcome.reason),
            };
        }
    }
    AzArenaGameOutcome {
        result: 0.0,
        draw_reason: Some(AzArenaDrawReason::MaxPlies),
    }
}

fn arena_draw_reason(reason: AzGameEndReason) -> Option<AzArenaDrawReason> {
    match reason {
        AzGameEndReason::NoAttackingMaterial => Some(AzArenaDrawReason::NoAttackingMaterial),
        AzGameEndReason::Halfmove120 => Some(AzArenaDrawReason::Halfmove120),
        AzGameEndReason::Repetition => Some(AzArenaDrawReason::Repetition),
        AzGameEndReason::MutualLongCheck => Some(AzArenaDrawReason::MutualLongCheck),
        AzGameEndReason::MutualLongChase => Some(AzArenaDrawReason::MutualLongChase),
        _ => None,
    }
}
