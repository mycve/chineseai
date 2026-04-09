use std::fs;
use std::io;
use std::path::Path;

use crate::nnue::{V2_INPUT_SIZE, extract_sparse_features_v2};
use crate::xiangqi::{BOARD_SIZE, Move, Position};

const MOVE_SPACE: usize = BOARD_SIZE * BOARD_SIZE;
const VALUE_SCALE_CP: f32 = 800.0;
const C_PUCT: f32 = 1.75;

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
}

impl Default for AzSearchLimits {
    fn default() -> Self {
        Self {
            simulations: 10_000,
            top_k: 32,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AzCandidate {
    pub mv: Move,
    pub visits: u32,
    pub q: f32,
    pub prior: f32,
}

#[derive(Clone, Debug)]
pub struct AzSearchResult {
    pub best_move: Option<Move>,
    pub value_cp: i32,
    pub simulations: usize,
    pub candidates: Vec<AzCandidate>,
}

impl AzNnue {
    pub fn random(hidden_size: usize, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let input_hidden = (0..V2_INPUT_SIZE * hidden_size)
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
            "format: aznnue-v1\ninput_size: {V2_INPUT_SIZE}\nhidden_size: {}\ninput_hidden: {}\nhidden_bias: {}\nvalue_hidden: {}\nvalue_bias: {}\npolicy_move_hidden: {}\npolicy_move_bias: {}\n",
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

        if input_size != Some(V2_INPUT_SIZE) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "aznnue-v1 requires fixed v2 input_size",
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

    pub fn evaluate(&self, position: &Position, moves: &[Move]) -> (f32, Vec<f32>) {
        let hidden = self.hidden(position);
        let value = self.value_from_hidden(&hidden);
        let logits = moves
            .iter()
            .map(|mv| self.policy_logit_from_hidden(&hidden, *mv))
            .collect();
        (value, logits)
    }

    fn validate(&self) -> io::Result<()> {
        if self.input_hidden.len() != V2_INPUT_SIZE * self.hidden_size
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

    fn hidden(&self, position: &Position) -> Vec<f32> {
        let mut hidden = self.hidden_bias.clone();
        for feature in extract_sparse_features_v2(position) {
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

pub fn gumbel_search(
    position: &Position,
    model: &AzNnue,
    limits: AzSearchLimits,
) -> AzSearchResult {
    let mut tree = AzTree::new(position.clone(), model);
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

    let mut active = tree.gumbel_top_k(root, limits.top_k.max(1));
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
        active.sort_by(|left, right| {
            tree.child_improved_score(root, *right)
                .total_cmp(&tree.child_improved_score(root, *left))
        });
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

    let mut candidates = tree.nodes[root]
        .children
        .iter()
        .map(|child| AzCandidate {
            mv: child.mv,
            visits: child.visits,
            q: child.q(),
            prior: child.prior,
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        right
            .visits
            .cmp(&left.visits)
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

struct AzTree<'a> {
    nodes: Vec<AzNode>,
    model: &'a AzNnue,
    root: usize,
}

struct AzNode {
    position: Position,
    children: Vec<AzChild>,
    visits: u32,
    value_sum: f32,
    value: f32,
    expanded: bool,
}

struct AzChild {
    mv: Move,
    prior: f32,
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
    fn new(position: Position, model: &'a AzNnue) -> Self {
        Self {
            nodes: vec![AzNode {
                position,
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
        let moves = self.nodes[node_index].position.legal_moves();
        let (value, logits) = self
            .model
            .evaluate(&self.nodes[node_index].position, &moves);
        let priors = softmax(&logits);
        self.nodes[node_index].children = moves
            .into_iter()
            .zip(priors)
            .map(|(mv, prior)| AzChild {
                mv,
                prior,
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
                child_position.make_move(mv);
                let child_node = self.nodes.len();
                self.nodes.push(AzNode {
                    position: child_position,
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
        let parent_visits = self.nodes[node_index].visits.max(1) as f32;
        let exploration = parent_visits.sqrt();
        let mut best = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (index, child) in self.nodes[node_index].children.iter().enumerate() {
            let u = C_PUCT * child.prior * exploration / (1.0 + child.visits as f32);
            let score = child.q() + u;
            if score > best_score {
                best_score = score;
                best = index;
            }
        }
        best
    }

    fn gumbel_top_k(&self, node_index: usize, top_k: usize) -> Vec<usize> {
        let hash = self.nodes[node_index].position.hash();
        let mut scored = self.nodes[node_index]
            .children
            .iter()
            .enumerate()
            .map(|(index, child)| {
                let score =
                    child.prior.max(1e-6).ln() + deterministic_gumbel(hash, child.mv, index as u64);
                (score, index)
            })
            .collect::<Vec<_>>();
        scored.sort_by(|left, right| right.0.total_cmp(&left.0));
        scored
            .into_iter()
            .take(top_k)
            .map(|(_, index)| index)
            .collect()
    }

    fn child_improved_score(&self, node_index: usize, child_index: usize) -> f32 {
        let child = &self.nodes[node_index].children[child_index];
        child.q() + child.prior * 0.05
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

    fn weight(&mut self, scale: f32) -> f32 {
        let value = self.next();
        let unit = (((value >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))) as f32;
        (unit * 2.0 - 1.0) * scale
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut mixed = value;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^ (mixed >> 31)
}
