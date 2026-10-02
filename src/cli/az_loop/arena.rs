use crate::cli::az_loop_config::AzLoopFileConfig;
use chineseai::az::{
    AzArenaConfig, AzArenaReport, AzNnue, SplitMix64, play_arena_games_from_positions,
};
use chineseai::pikafish::opening_book::Px0OpeningBook;
use chineseai::xiangqi::Position;
use std::{sync::Arc, thread};

pub(crate) fn historical_anchor_index(champion_count: usize, gate_index: usize) -> Option<usize> {
    let current = champion_count.checked_sub(1)?;
    let offsets = [2usize, 4, 8, 16, 32]
        .into_iter()
        .filter(|&offset| offset <= current)
        .collect::<Vec<_>>();
    let offset = offsets.get(gate_index % offsets.len().max(1))?;
    Some(current - offset)
}

pub(crate) fn arena_gate_position_counts(
    total: usize,
    has_previous: bool,
    has_anchor: bool,
) -> (usize, usize, usize) {
    let previous = has_previous.then_some(total / 5).unwrap_or(0);
    let anchor = has_anchor.then_some(total / 5).unwrap_or(0);
    (total - previous - anchor, previous, anchor)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ArenaGateDecision {
    Promote,
    Continue,
    Reject,
}

pub(crate) fn arena_gate_decision(
    current: &AzArenaReport,
    previous: Option<&AzArenaReport>,
    anchor: Option<&AzArenaReport>,
    current_threshold: f32,
    confidence_z: f32,
) -> ArenaGateDecision {
    let z = confidence_z.max(0.0);
    let proven_current_regression = current.score_rate_upper_bound(z) < 0.50;
    let proven_history_regression = previous
        .into_iter()
        .chain(anchor)
        .any(|report| report.score_rate_upper_bound(z) < 0.50);
    let mut combined_history = AzArenaReport::default();
    for report in previous.into_iter().chain(anchor) {
        combined_history.add_assign(report);
    }
    let proven_combined_history_regression =
        combined_history.total_games() > 0 && combined_history.score_rate_upper_bound(z) < 0.50;
    if proven_current_regression || proven_history_regression || proven_combined_history_regression
    {
        return ArenaGateDecision::Reject;
    }

    if current.score_rate_lower_bound(z) > current_threshold {
        ArenaGateDecision::Promote
    } else {
        ArenaGateDecision::Continue
    }
}

pub(crate) fn shuffle_positions(positions: &mut [Position], rng: &mut SplitMix64) {
    for index in (1..positions.len()).rev() {
        positions.swap(index, rng.next_u64() as usize % (index + 1));
    }
}

pub(crate) struct ArenaThreadConfig {
    pub(crate) candidate: Arc<AzNnue>,
    pub(crate) baseline: Arc<AzNnue>,
    pub(crate) eval_starts: ArenaStarts,
    pub(crate) simulations: usize,
    pub(crate) max_plies: usize,
    pub(crate) rule60_max_ply: Option<u16>,
    pub(crate) cpuct: f32,
    pub(crate) cpuct_at_root: f32,
    pub(crate) cpuct_base: f32,
    pub(crate) cpuct_factor: f32,
    pub(crate) cpuct_base_at_root: f32,
    pub(crate) cpuct_factor_at_root: f32,
    pub(crate) fpu_value: f32,
    pub(crate) fpu_value_at_root: f32,
    pub(crate) draw_score: f32,
    pub(crate) policy_softmax_temp: f32,
    pub(crate) thread_count: usize,
    pub(crate) seed: u64,
}

#[derive(Clone)]
pub(crate) enum ArenaStarts {
    Positions(Arc<Vec<Position>>),
}

impl ArenaStarts {
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Positions(positions) => positions.len(),
        }
    }
}

pub(crate) fn run_arena_threads(config: ArenaThreadConfig) -> AzArenaReport {
    let games_per_side = if config.eval_starts.len() == 0 {
        1
    } else {
        config.eval_starts.len().max(1)
    };
    let thread_count = config.thread_count.max(1).min(games_per_side);
    let mut handles = Vec::with_capacity(thread_count);
    let mut start_index = 0usize;
    for index in 0..thread_count {
        let red_games =
            games_per_side / thread_count + usize::from(index < games_per_side % thread_count);
        let black_games = red_games;
        if red_games == 0 && black_games == 0 {
            continue;
        }
        let candidate = Arc::clone(&config.candidate);
        let baseline = Arc::clone(&config.baseline);
        let eval_starts = config.eval_starts.clone();
        let simulations = config.simulations;
        let max_plies = config.max_plies;
        let rule60_max_ply = config.rule60_max_ply;
        let cpuct = config.cpuct;
        let cpuct_at_root = config.cpuct_at_root;
        let cpuct_base = config.cpuct_base;
        let cpuct_factor = config.cpuct_factor;
        let cpuct_base_at_root = config.cpuct_base_at_root;
        let cpuct_factor_at_root = config.cpuct_factor_at_root;
        let fpu_value = config.fpu_value;
        let fpu_value_at_root = config.fpu_value_at_root;
        let draw_score = config.draw_score;
        let policy_softmax_temp = config.policy_softmax_temp;
        // 由全局开局索引派生每对随机流；结果不应随线程切分变化。
        let seed = config.seed;
        let thread_start_index = start_index;
        start_index += red_games;
        handles.push(thread::spawn(move || {
            let arena_config = AzArenaConfig {
                simulations,
                max_plies,
                rule60_max_ply,
                games_as_red: red_games,
                games_as_black: black_games,
                start_index: thread_start_index,
                seed,
                cpuct,
                cpuct_at_root,
                cpuct_base,
                cpuct_factor,
                cpuct_base_at_root,
                cpuct_factor_at_root,
                fpu_value,
                fpu_value_at_root,
                fpu_absolute_at_root: true,
                minimum_kldgain_per_node: 0.0,
                draw_score,
                policy_softmax_temp,
            };
            match eval_starts {
                ArenaStarts::Positions(positions) => play_arena_games_from_positions(
                    candidate.as_ref(),
                    baseline.as_ref(),
                    positions.as_slice(),
                    arena_config,
                ),
            }
        }));
    }

    let mut merged = AzArenaReport::default();
    for handle in handles {
        merged.add_assign(
            &handle
                .join()
                .unwrap_or_else(|_| panic!("arena thread panicked")),
        );
    }
    merged
}

pub(crate) fn build_arena_start_positions(
    config: &AzLoopFileConfig,
    update: usize,
) -> (Vec<Position>, String) {
    // 每次门控使用新的确定性留出折，避免反复在同一小批局面上选择导致评测过拟合。
    // 折内每个局面仍严格交换红黑配对。
    let gate_index = update / config.arena_interval.max(1);
    let seed = config.seed
        ^ 0xD1B5_4A32_D192_ED03
        ^ (gate_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut book = Px0OpeningBook::load(&config.arena_opening_book, seed)
        .unwrap_or_else(|err| panic!("failed to load Px0 arena opening book: {err}"));
    let count = book.len();
    let positions = book
        .next_batch(1000, 0)
        .unwrap_or_else(|err| panic!("invalid arena opening FEN: {err}"))
        .into_iter()
        .map(|snapshot| snapshot.position)
        .collect();
    (
        positions,
        format!("px0(shuffled,count=1000,book_positions={count})"),
    )
}
