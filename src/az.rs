use std::fs;
use std::io;
use std::path::Path;
use std::sync::Arc;
use std::thread;

use crate::nnue::{HistoryMove, V3_INPUT_SIZE, extract_sparse_features_v3};
use crate::xiangqi::{BOARD_SIZE, Color, Move, Position};

const MOVE_SPACE: usize = BOARD_SIZE * BOARD_SIZE;
const VALUE_SCALE_CP: f32 = 800.0;
const COMPLETED_Q_VALUE_SCALE: f32 = 0.1;
const COMPLETED_Q_MAXVISIT_INIT: f32 = 50.0;

#[derive(Clone, Debug)]
pub struct AzNnue {
    pub hidden_size: usize,
    pub input_hidden: Vec<f32>,
    pub hidden_bias: Vec<f32>,
    pub value_hidden: Vec<f32>,
    pub value_bias: f32,
    pub policy_move_hidden: Vec<f32>,
    pub policy_move_bias: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
pub struct AzSearchLimits {
    pub simulations: usize,
    pub top_k: usize,
    pub seed: u64,
    pub gumbel_scale: f32,
    pub workers: usize,
}

impl Default for AzSearchLimits {
    fn default() -> Self {
        Self {
            simulations: 10_000,
            top_k: 32,
            seed: 0,
            gumbel_scale: 1.0,
            workers: 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AzCandidate {
    pub mv: Move,
    pub visits: u32,
    pub q: f32,
    pub prior: f32,
    pub policy: f32,
}

#[derive(Clone, Debug)]
pub struct AzSearchResult {
    pub best_move: Option<Move>,
    pub value_cp: i32,
    pub simulations: usize,
    pub candidates: Vec<AzCandidate>,
}

#[derive(Clone, Debug)]
pub struct AzLoopConfig {
    pub games: usize,
    pub max_plies: usize,
    pub simulations: usize,
    pub top_k: usize,
    pub epochs: usize,
    pub lr: f32,
    pub seed: u64,
    pub workers: usize,
    pub temperature_start: f32,
    pub temperature_end: f32,
    pub temperature_decay_plies: usize,
    pub gumbel_scale: f32,
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
    pub value_mse: f32,
    pub policy_ce: f32,
}

#[derive(Clone, Debug)]
struct AzTrainingSample {
    features: Vec<usize>,
    moves: Vec<Move>,
    policy: Vec<f32>,
    value: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct AzTrainStats {
    loss: f32,
    value_mse: f32,
    policy_ce: f32,
    samples: usize,
}

#[derive(Clone, Debug, Default)]
struct AzSelfplayData {
    samples: Vec<AzTrainingSample>,
    red_wins: usize,
    black_wins: usize,
    draws: usize,
    plies_total: usize,
}

impl AzNnue {
    pub fn random(hidden_size: usize, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let input_hidden = (0..V3_INPUT_SIZE * hidden_size)
            .map(|_| rng.weight(0.015))
            .collect();
        let hidden_bias = vec![0.0; hidden_size];
        let value_hidden = (0..hidden_size).map(|_| rng.weight(0.05)).collect();
        let policy_move_hidden = (0..MOVE_SPACE * hidden_size)
            .map(|_| rng.weight(0.01))
            .collect();
        let policy_move_bias = vec![0.0; MOVE_SPACE];
        Self {
            hidden_size,
            input_hidden,
            hidden_bias,
            value_hidden,
            value_bias: 0.0,
            policy_move_hidden,
            policy_move_bias,
        }
    }

    pub fn save_text(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let text = format!(
            "format: aznnue-v2\ninput_size: {V3_INPUT_SIZE}\nhidden_size: {}\ninput_hidden: {}\nhidden_bias: {}\nvalue_hidden: {}\nvalue_bias: {}\npolicy_move_hidden: {}\npolicy_move_bias: {}\n",
            self.hidden_size,
            format_floats(&self.input_hidden),
            format_floats(&self.hidden_bias),
            format_floats(&self.value_hidden),
            self.value_bias,
            format_floats(&self.policy_move_hidden),
            format_floats(&self.policy_move_bias),
        );
        fs::write(path, text)
    }

    pub fn load_text(path: impl AsRef<Path>) -> io::Result<Self> {
        let text = fs::read_to_string(path)?;
        let mut input_size = None;
        let mut hidden_size = None;
        let mut input_hidden = None;
        let mut hidden_bias = None;
        let mut value_hidden = None;
        let mut value_bias = None;
        let mut policy_move_hidden = None;
        let mut policy_move_bias = None;

        for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            let value = value.trim();
            match key.trim() {
                "input_size" => input_size = value.parse::<usize>().ok(),
                "hidden_size" => hidden_size = value.parse::<usize>().ok(),
                "input_hidden" => input_hidden = Some(parse_floats(value)?),
                "hidden_bias" => hidden_bias = Some(parse_floats(value)?),
                "value_hidden" => value_hidden = Some(parse_floats(value)?),
                "value_bias" => value_bias = value.parse::<f32>().ok(),
                "policy_move_hidden" => policy_move_hidden = Some(parse_floats(value)?),
                "policy_move_bias" => policy_move_bias = Some(parse_floats(value)?),
                _ => {}
            }
        }

        if input_size != Some(V3_INPUT_SIZE) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "aznnue-v2 requires fixed v3 input_size",
            ));
        }
        let hidden_size = hidden_size
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing hidden_size"))?;
        let model = Self {
            hidden_size,
            input_hidden: input_hidden.ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing input_hidden")
            })?,
            hidden_bias: hidden_bias
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing hidden_bias"))?,
            value_hidden: value_hidden.ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing value_hidden")
            })?,
            value_bias: value_bias
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing value_bias"))?,
            policy_move_hidden: policy_move_hidden.ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing policy_move_hidden")
            })?,
            policy_move_bias: policy_move_bias.ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing policy_move_bias")
            })?,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn evaluate(
        &self,
        position: &Position,
        history: &[HistoryMove],
        moves: &[Move],
    ) -> (f32, Vec<f32>) {
        let hidden = self.hidden(position, history);
        let value = self.value_from_hidden(&hidden);
        let logits = moves
            .iter()
            .map(|mv| self.policy_logit_from_hidden(&hidden, *mv))
            .collect();
        (value, logits)
    }

    fn validate(&self) -> io::Result<()> {
        if self.input_hidden.len() != V3_INPUT_SIZE * self.hidden_size
            || self.hidden_bias.len() != self.hidden_size
            || self.value_hidden.len() != self.hidden_size
            || self.policy_move_hidden.len() != MOVE_SPACE * self.hidden_size
            || self.policy_move_bias.len() != MOVE_SPACE
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "aznnue vector length mismatch",
            ));
        }
        Ok(())
    }

    fn hidden(&self, position: &Position, history: &[HistoryMove]) -> Vec<f32> {
        let mut hidden = self.hidden_bias.clone();
        for feature in extract_sparse_features_v3(position, history) {
            let row =
                &self.input_hidden[feature * self.hidden_size..(feature + 1) * self.hidden_size];
            for idx in 0..self.hidden_size {
                hidden[idx] += row[idx];
            }
        }
        for value in &mut hidden {
            *value = value.max(0.0);
        }
        hidden
    }

    fn value_from_hidden(&self, hidden: &[f32]) -> f32 {
        let mut value = self.value_bias;
        for (idx, hidden_value) in hidden.iter().enumerate() {
            value += hidden_value * self.value_hidden[idx];
        }
        value.tanh()
    }

    fn policy_logit_from_hidden(&self, hidden: &[f32], mv: Move) -> f32 {
        let move_index = mv.from as usize * BOARD_SIZE + mv.to as usize;
        let offset = move_index * self.hidden_size;
        let row = &self.policy_move_hidden[offset..offset + self.hidden_size];
        let mut logit = self.policy_move_bias[move_index];
        for (idx, hidden_value) in hidden.iter().enumerate() {
            logit += hidden_value * row[idx];
        }
        logit
    }
}

pub fn selfplay_train_iteration(model: &mut AzNnue, config: &AzLoopConfig) -> AzLoopReport {
    let data = generate_selfplay_data(model, config);
    let mut rng = SplitMix64::new(config.seed ^ 0xA5A5_5A5A_D3C3_B4B4);
    let stats = train_samples(model, &data.samples, config.epochs, config.lr, &mut rng);
    AzLoopReport {
        games: config.games,
        samples: data.samples.len(),
        red_wins: data.red_wins,
        black_wins: data.black_wins,
        draws: data.draws,
        avg_plies: if config.games == 0 {
            0.0
        } else {
            data.plies_total as f32 / config.games as f32
        },
        loss: stats.loss,
        value_mse: stats.value_mse,
        policy_ce: stats.policy_ce,
    }
}

fn generate_selfplay_data(model: &AzNnue, config: &AzLoopConfig) -> AzSelfplayData {
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
        merged.red_wins += chunk.red_wins;
        merged.black_wins += chunk.black_wins;
        merged.draws += chunk.draws;
        merged.plies_total += chunk.plies_total;
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

    for game_index in 0..config.games {
        let mut position = Position::startpos();
        let mut history = Vec::new();
        let mut game_samples = Vec::new();
        let mut result = None;
        let mut plies = 0usize;

        for ply in 0..config.max_plies {
            plies = ply + 1;
            let legal = position.legal_moves();
            if legal.is_empty() {
                result = Some(if position.side_to_move() == Color::Red {
                    -1.0
                } else {
                    1.0
                });
                break;
            }

            let search = gumbel_search_with_history(
                &position,
                &history,
                model,
                AzSearchLimits {
                    simulations: config.simulations,
                    top_k: config.top_k,
                    seed: rng.next() ^ ((game_index as u64) << 32) ^ ply as u64,
                    gumbel_scale: config.gumbel_scale,
                    workers: 1,
                },
            );
            let temperature = temperature_for_ply(config, ply);
            let Some(mv) = choose_selfplay_move(&search.candidates, temperature, &mut rng) else {
                result = Some(0.0);
                break;
            };
            game_samples.push(make_training_sample(
                &position,
                &history,
                &search.candidates,
            ));
            append_history(&mut history, &position, mv);
            position.make_move(mv);

            if !position.has_general(Color::Red) {
                result = Some(-1.0);
                break;
            }
            if !position.has_general(Color::Black) {
                result = Some(1.0);
                break;
            }
        }

        let result: f32 = result.unwrap_or(0.0);
        match result.total_cmp(&0.0) {
            std::cmp::Ordering::Greater => red_wins += 1,
            std::cmp::Ordering::Less => black_wins += 1,
            std::cmp::Ordering::Equal => draws += 1,
        }
        plies_total += plies;

        for sample in &mut game_samples {
            sample.value *= result;
        }
        samples.extend(game_samples);
    }

    AzSelfplayData {
        samples,
        red_wins,
        black_wins,
        draws,
        plies_total,
    }
}

pub fn gumbel_search(
    position: &Position,
    model: &AzNnue,
    limits: AzSearchLimits,
) -> AzSearchResult {
    gumbel_search_with_history(position, &[], model, limits)
}

pub fn gumbel_search_with_history(
    position: &Position,
    history: &[HistoryMove],
    model: &AzNnue,
    limits: AzSearchLimits,
) -> AzSearchResult {
    if limits.workers > 1 && limits.simulations > 1 {
        return gumbel_search_parallel(position, history, model, limits);
    }
    let mut tree = AzTree::new(position.clone(), truncate_history(history), model);
    let root = tree.root;
    tree.expand(root);
    if tree.nodes[root].children.is_empty() {
        return AzSearchResult {
            best_move: None,
            value_cp: (tree.nodes[root].value * VALUE_SCALE_CP) as i32,
            simulations: 0,
            candidates: Vec::new(),
        };
    }

    let mut active = tree.gumbel_top_k(root, limits.top_k.max(1), limits.seed, limits.gumbel_scale);
    let mut used = 0usize;
    while active.len() > 1 && used < limits.simulations {
        let rounds_left = active.len().ilog2().max(1) as usize;
        let per_candidate = ((limits.simulations - used) / (active.len() * rounds_left)).max(1);
        for &child_index in &active {
            for _ in 0..per_candidate {
                tree.simulate_child(root, child_index);
                used += 1;
                if used >= limits.simulations {
                    break;
                }
            }
        }
        let scores = tree.gumbel_scores(root, limits.seed, limits.gumbel_scale);
        active.sort_by(|left, right| scores[*right].total_cmp(&scores[*left]));
        active.truncate(active.len().div_ceil(2));
    }
    while used < limits.simulations {
        let child_index = active
            .first()
            .copied()
            .unwrap_or_else(|| tree.select_child(root));
        tree.simulate_child(root, child_index);
        used += 1;
    }

    let policy = tree.improved_policy(root);
    let mut candidates = tree.nodes[root]
        .children
        .iter()
        .enumerate()
        .map(|(child_index, child)| AzCandidate {
            mv: child.mv,
            visits: child.visits,
            q: child.q(),
            prior: child.prior,
            policy: policy[child_index],
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        right
            .policy
            .total_cmp(&left.policy)
            .then_with(|| right.visits.cmp(&left.visits))
            .then_with(|| right.q.total_cmp(&left.q))
    });
    let best_move = candidates.first().map(|candidate| candidate.mv);
    AzSearchResult {
        best_move,
        value_cp: (tree.nodes[root].value * VALUE_SCALE_CP) as i32,
        simulations: used,
        candidates,
    }
}

fn gumbel_search_parallel(
    position: &Position,
    history: &[HistoryMove],
    model: &AzNnue,
    limits: AzSearchLimits,
) -> AzSearchResult {
    let workers = limits.workers.max(1).min(limits.simulations.max(1));
    if workers <= 1 {
        return gumbel_search_with_history(
            position,
            history,
            model,
            AzSearchLimits {
                workers: 1,
                ..limits
            },
        );
    }

    let shared_model = Arc::new(model.clone());
    let shared_history = Arc::new(truncate_history(history));
    let mut handles = Vec::with_capacity(workers);
    for worker in 0..workers {
        let simulations =
            limits.simulations / workers + usize::from(worker < limits.simulations % workers);
        if simulations == 0 {
            continue;
        }
        let worker_position = position.clone();
        let worker_history = Arc::clone(&shared_history);
        let worker_model = Arc::clone(&shared_model);
        let worker_limits = AzSearchLimits {
            simulations,
            seed: limits.seed ^ (worker as u64).wrapping_mul(0xD1B5_4A32_D192_ED03),
            workers: 1,
            ..limits
        };
        handles.push(thread::spawn(move || {
            gumbel_search_with_history(
                &worker_position,
                worker_history.as_slice(),
                &worker_model,
                worker_limits,
            )
        }));
    }

    let mut partials = Vec::new();
    for handle in handles {
        partials.push(handle.join().expect("search worker panicked"));
    }
    merge_search_results(partials)
}

fn merge_search_results(results: Vec<AzSearchResult>) -> AzSearchResult {
    let mut merged = Vec::<AzCandidate>::new();
    let mut simulations = 0usize;
    let mut value_weight_sum = 0.0;
    let mut value_visit_sum = 0.0;

    for result in results {
        simulations += result.simulations;
        for candidate in result.candidates {
            let visits = candidate.visits.max(1) as f32;
            value_weight_sum += candidate.q * visits;
            value_visit_sum += visits;
            if let Some(existing) = merged
                .iter_mut()
                .find(|existing| existing.mv == candidate.mv)
            {
                let old_visits = existing.visits as f32;
                let new_visits = candidate.visits as f32;
                let total_visits = old_visits + new_visits;
                if total_visits > 0.0 {
                    existing.q =
                        (existing.q * old_visits + candidate.q * new_visits) / total_visits;
                }
                existing.visits += candidate.visits;
                existing.prior = existing.prior.max(candidate.prior);
                existing.policy += candidate.policy;
            } else {
                merged.push(candidate);
            }
        }
    }

    let total_policy = merged
        .iter()
        .map(|candidate| candidate.policy.max(0.0))
        .sum::<f32>()
        .max(1e-9);
    for candidate in &mut merged {
        candidate.policy = candidate.policy.max(0.0) / total_policy;
    }
    merged.sort_by(|left, right| {
        right
            .policy
            .total_cmp(&left.policy)
            .then_with(|| right.visits.cmp(&left.visits))
            .then_with(|| right.q.total_cmp(&left.q))
    });
    let best_move = merged.first().map(|candidate| candidate.mv);
    AzSearchResult {
        best_move,
        value_cp: if value_visit_sum > 0.0 {
            ((value_weight_sum / value_visit_sum) * VALUE_SCALE_CP) as i32
        } else {
            0
        },
        simulations,
        candidates: merged,
    }
}

struct AzTree<'a> {
    nodes: Vec<AzNode>,
    model: &'a AzNnue,
    root: usize,
}

struct AzNode {
    position: Position,
    history: Vec<HistoryMove>,
    children: Vec<AzChild>,
    visits: u32,
    value_sum: f32,
    value: f32,
    expanded: bool,
}

struct AzChild {
    mv: Move,
    prior: f32,
    prior_logit: f32,
    visits: u32,
    value_sum: f32,
    child: Option<usize>,
}

impl AzChild {
    fn q(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }
}

impl<'a> AzTree<'a> {
    fn new(position: Position, history: Vec<HistoryMove>, model: &'a AzNnue) -> Self {
        Self {
            nodes: vec![AzNode {
                position,
                history,
                children: Vec::new(),
                visits: 0,
                value_sum: 0.0,
                value: 0.0,
                expanded: false,
            }],
            model,
            root: 0,
        }
    }

    fn expand(&mut self, node_index: usize) -> f32 {
        if self.nodes[node_index].expanded {
            return self.nodes[node_index].value;
        }
        if let Some(value) = terminal_value(&self.nodes[node_index].position) {
            self.nodes[node_index].children.clear();
            self.nodes[node_index].value = value;
            self.nodes[node_index].expanded = true;
            return value;
        }
        let moves = self.nodes[node_index].position.legal_moves();
        if moves.is_empty() {
            self.nodes[node_index].children.clear();
            self.nodes[node_index].value = -1.0;
            self.nodes[node_index].expanded = true;
            return -1.0;
        }
        let (value, logits) = self.model.evaluate(
            &self.nodes[node_index].position,
            &self.nodes[node_index].history,
            &moves,
        );
        let priors = softmax(&logits);
        self.nodes[node_index].children = moves
            .into_iter()
            .zip(logits)
            .zip(priors)
            .map(|((mv, prior_logit), prior)| AzChild {
                mv,
                prior,
                prior_logit,
                visits: 0,
                value_sum: 0.0,
                child: None,
            })
            .collect();
        self.nodes[node_index].value = value;
        self.nodes[node_index].expanded = true;
        value
    }

    fn simulate(&mut self, node_index: usize) -> f32 {
        if !self.nodes[node_index].expanded {
            let value = self.expand(node_index);
            self.nodes[node_index].visits += 1;
            self.nodes[node_index].value_sum += value;
            return value;
        }
        if self.nodes[node_index].children.is_empty() {
            self.nodes[node_index].visits += 1;
            self.nodes[node_index].value_sum += self.nodes[node_index].value;
            return self.nodes[node_index].value;
        }
        let child_index = self.select_child(node_index);
        self.simulate_child(node_index, child_index)
    }

    fn simulate_child(&mut self, node_index: usize, child_index: usize) -> f32 {
        let child_node =
            if let Some(child_node) = self.nodes[node_index].children[child_index].child {
                child_node
            } else {
                let mv = self.nodes[node_index].children[child_index].mv;
                let mut child_position = self.nodes[node_index].position.clone();
                let mut child_history = self.nodes[node_index].history.clone();
                append_history(&mut child_history, &child_position, mv);
                child_position.make_move(mv);
                let child_node = self.nodes.len();
                self.nodes.push(AzNode {
                    position: child_position,
                    history: child_history,
                    children: Vec::new(),
                    visits: 0,
                    value_sum: 0.0,
                    value: 0.0,
                    expanded: false,
                });
                self.nodes[node_index].children[child_index].child = Some(child_node);
                child_node
            };
        let child_value = self.simulate(child_node);
        let value = -child_value;
        let child = &mut self.nodes[node_index].children[child_index];
        child.visits += 1;
        child.value_sum += value;
        self.nodes[node_index].visits += 1;
        self.nodes[node_index].value_sum += value;
        value
    }

    fn select_child(&self, node_index: usize) -> usize {
        let policy = self.improved_policy(node_index);
        let total_visits = self.nodes[node_index]
            .children
            .iter()
            .map(|child| child.visits)
            .sum::<u32>() as f32;
        let mut best = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (index, child) in self.nodes[node_index].children.iter().enumerate() {
            let visited_fraction = child.visits as f32 / (1.0 + total_visits);
            let score = policy[index] - visited_fraction;
            if score > best_score {
                best_score = score;
                best = index;
            }
        }
        best
    }

    fn gumbel_top_k(
        &self,
        node_index: usize,
        top_k: usize,
        seed: u64,
        gumbel_scale: f32,
    ) -> Vec<usize> {
        let scores = self.gumbel_scores(node_index, seed, gumbel_scale);
        let mut scored = scores
            .into_iter()
            .enumerate()
            .map(|(index, score)| (score, index))
            .collect::<Vec<_>>();
        scored.sort_by(|left, right| right.0.total_cmp(&left.0));
        scored
            .into_iter()
            .take(top_k)
            .map(|(_, index)| index)
            .collect()
    }

    fn gumbel_scores(&self, node_index: usize, seed: u64, gumbel_scale: f32) -> Vec<f32> {
        let completed_q = self.completed_qvalues(node_index);
        let hash = self.nodes[node_index].position.hash() ^ seed;
        self.nodes[node_index]
            .children
            .iter()
            .enumerate()
            .map(|(child_index, child)| {
                child.prior_logit
                    + completed_q[child_index]
                    + gumbel_scale * deterministic_gumbel(hash, child.mv, child_index as u64)
            })
            .collect()
    }

    fn improved_policy(&self, node_index: usize) -> Vec<f32> {
        let completed_q = self.completed_qvalues(node_index);
        let logits = self.nodes[node_index]
            .children
            .iter()
            .enumerate()
            .map(|(index, child)| child.prior_logit + completed_q[index])
            .collect::<Vec<_>>();
        softmax(&logits)
    }

    fn completed_qvalues(&self, node_index: usize) -> Vec<f32> {
        let children = &self.nodes[node_index].children;
        if children.is_empty() {
            return Vec::new();
        }

        let total_visits = children.iter().map(|child| child.visits).sum::<u32>();
        let max_visits = children
            .iter()
            .map(|child| child.visits)
            .max()
            .unwrap_or_default();
        let visited_policy_sum = children
            .iter()
            .filter(|child| child.visits > 0)
            .map(|child| child.prior)
            .sum::<f32>();
        let weighted_q = if visited_policy_sum > 0.0 {
            children
                .iter()
                .filter(|child| child.visits > 0)
                .map(|child| child.prior * child.q())
                .sum::<f32>()
                / visited_policy_sum
        } else {
            self.nodes[node_index].value
        };
        let mixed_value = (self.nodes[node_index].value + total_visits as f32 * weighted_q)
            / (total_visits as f32 + 1.0);

        let mut qvalues = children
            .iter()
            .map(|child| {
                if child.visits > 0 {
                    child.q()
                } else {
                    mixed_value
                }
            })
            .collect::<Vec<_>>();
        normalize_completed_q(&mut qvalues, max_visits);
        qvalues
    }
}

fn normalize_completed_q(qvalues: &mut [f32], total_visits: u32) {
    let min_value = qvalues.iter().copied().fold(f32::INFINITY, f32::min);
    let max_value = qvalues.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = max_value - min_value;
    if range > 1e-6 {
        for value in qvalues.iter_mut() {
            *value = (*value - min_value) / range;
        }
    }

    let maxvisit_scale =
        (COMPLETED_Q_MAXVISIT_INIT + total_visits as f32) * COMPLETED_Q_VALUE_SCALE;
    for value in qvalues.iter_mut() {
        *value *= maxvisit_scale;
    }
}

fn terminal_value(position: &Position) -> Option<f32> {
    if !position.has_general(Color::Red) {
        return Some(if position.side_to_move() == Color::Red {
            -1.0
        } else {
            1.0
        });
    }
    if !position.has_general(Color::Black) {
        return Some(if position.side_to_move() == Color::Black {
            -1.0
        } else {
            1.0
        });
    }
    None
}

fn make_training_sample(
    position: &Position,
    history: &[HistoryMove],
    candidates: &[AzCandidate],
) -> AzTrainingSample {
    let total_policy = candidates
        .iter()
        .map(|candidate| candidate.policy.max(0.0))
        .sum::<f32>()
        .max(1.0);
    AzTrainingSample {
        features: extract_sparse_features_v3(position, history),
        moves: candidates.iter().map(|candidate| candidate.mv).collect(),
        policy: candidates
            .iter()
            .map(|candidate| candidate.policy.max(0.0) / total_policy)
            .collect(),
        value: if position.side_to_move() == Color::Red {
            1.0
        } else {
            -1.0
        },
    }
}

fn append_history(history: &mut Vec<HistoryMove>, position: &Position, mv: Move) {
    if let Some(piece) = position.piece_at(mv.from as usize) {
        history.push(HistoryMove { piece, mv });
        let overflow = history.len().saturating_sub(crate::nnue::HISTORY_PLIES);
        if overflow > 0 {
            history.drain(0..overflow);
        }
    }
}

fn truncate_history(history: &[HistoryMove]) -> Vec<HistoryMove> {
    history
        .iter()
        .rev()
        .take(crate::nnue::HISTORY_PLIES)
        .copied()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
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

fn train_samples(
    model: &mut AzNnue,
    samples: &[AzTrainingSample],
    epochs: usize,
    lr: f32,
    rng: &mut SplitMix64,
) -> AzTrainStats {
    if samples.is_empty() || epochs == 0 || lr <= 0.0 {
        return AzTrainStats::default();
    }

    let mut order = (0..samples.len()).collect::<Vec<_>>();
    let mut stats = AzTrainStats::default();
    for _ in 0..epochs {
        shuffle(&mut order, rng);
        stats = AzTrainStats::default();
        for &index in &order {
            let sample_stats = train_one(model, &samples[index], lr);
            stats.loss += sample_stats.loss;
            stats.value_mse += sample_stats.value_mse;
            stats.policy_ce += sample_stats.policy_ce;
            stats.samples += 1;
        }
    }
    if stats.samples > 0 {
        let denom = stats.samples as f32;
        stats.loss /= denom;
        stats.value_mse /= denom;
        stats.policy_ce /= denom;
    }
    stats
}

fn train_one(model: &mut AzNnue, sample: &AzTrainingSample, lr: f32) -> AzTrainStats {
    let mut hidden = model.hidden_bias.clone();
    for &feature in &sample.features {
        let row =
            &model.input_hidden[feature * model.hidden_size..(feature + 1) * model.hidden_size];
        for idx in 0..model.hidden_size {
            hidden[idx] += row[idx];
        }
    }
    let relu_mask = hidden.iter().map(|value| *value > 0.0).collect::<Vec<_>>();
    for value in &mut hidden {
        *value = value.max(0.0);
    }

    let value = model.value_from_hidden(&hidden);
    let value_error = value - sample.value;
    let value_mse = value_error * value_error;
    let value_grad = (2.0 * value_error * (1.0 - value * value)).clamp(-4.0, 4.0);

    let logits = sample
        .moves
        .iter()
        .map(|mv| model.policy_logit_from_hidden(&hidden, *mv))
        .collect::<Vec<_>>();
    let prediction = softmax(&logits);
    let policy_ce = prediction
        .iter()
        .zip(&sample.policy)
        .map(|(predicted, target)| -target * predicted.max(1e-9).ln())
        .sum::<f32>();

    let mut hidden_grad = vec![0.0; model.hidden_size];
    for idx in 0..model.hidden_size {
        hidden_grad[idx] += value_grad * model.value_hidden[idx];
    }
    for ((mv, predicted), target) in sample.moves.iter().zip(&prediction).zip(&sample.policy) {
        let move_index = mv.from as usize * BOARD_SIZE + mv.to as usize;
        let policy_grad = (predicted - target).clamp(-4.0, 4.0);
        let offset = move_index * model.hidden_size;
        for idx in 0..model.hidden_size {
            hidden_grad[idx] += policy_grad * model.policy_move_hidden[offset + idx];
        }
    }

    for idx in 0..model.hidden_size {
        model.value_hidden[idx] -= lr * value_grad * hidden[idx];
    }
    model.value_bias -= lr * value_grad;

    for ((mv, predicted), target) in sample.moves.iter().zip(&prediction).zip(&sample.policy) {
        let move_index = mv.from as usize * BOARD_SIZE + mv.to as usize;
        let policy_grad = (predicted - target).clamp(-4.0, 4.0);
        let offset = move_index * model.hidden_size;
        for idx in 0..model.hidden_size {
            model.policy_move_hidden[offset + idx] -= lr * policy_grad * hidden[idx];
        }
        model.policy_move_bias[move_index] -= lr * policy_grad;
    }

    let input_lr = lr / (sample.features.len() as f32).sqrt().max(1.0);
    for idx in 0..model.hidden_size {
        if !relu_mask[idx] {
            continue;
        }
        let grad = hidden_grad[idx].clamp(-4.0, 4.0);
        model.hidden_bias[idx] -= input_lr * grad;
        for &feature in &sample.features {
            model.input_hidden[feature * model.hidden_size + idx] -= input_lr * grad;
        }
    }

    AzTrainStats {
        loss: value_mse + policy_ce,
        value_mse,
        policy_ce,
        samples: 1,
    }
}

fn shuffle(values: &mut [usize], rng: &mut SplitMix64) {
    for index in (1..values.len()).rev() {
        let swap_with = (rng.next() as usize) % (index + 1);
        values.swap(index, swap_with);
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    let mut values = Vec::with_capacity(logits.len());
    for logit in logits {
        let value = (*logit - max_logit).exp();
        sum += value;
        values.push(value);
    }
    for value in &mut values {
        *value /= sum.max(1e-12);
    }
    values
}

fn parse_floats(text: &str) -> io::Result<Vec<f32>> {
    text.split_whitespace()
        .map(|value| {
            value.parse::<f32>().map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid float: {value}"),
                )
            })
        })
        .collect()
}

fn format_floats(values: &[f32]) -> String {
    values
        .iter()
        .map(f32::to_string)
        .collect::<Vec<_>>()
        .join(" ")
}

fn deterministic_gumbel(hash: u64, mv: Move, salt: u64) -> f32 {
    let seed = hash
        ^ ((mv.from as u64) << 48)
        ^ ((mv.to as u64) << 32)
        ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let value = splitmix64(seed);
    let unit = (((value >> 11) as f64) + 0.5) * (1.0 / ((1u64 << 53) as f64));
    (-(-unit.ln()).ln()) as f32
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = splitmix64(self.state);
        self.state
    }

    fn unit_f32(&mut self) -> f32 {
        let value = self.next();
        (((value >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))) as f32
    }

    fn weight(&mut self, scale: f32) -> f32 {
        (self.unit_f32() * 2.0 - 1.0) * scale
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut mixed = value;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^ (mixed >> 31)
}
