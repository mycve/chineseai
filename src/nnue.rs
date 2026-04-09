use std::fs;
use std::io;
use std::path::Path;

use crate::xiangqi::{BOARD_SIZE, Color, PieceKind, Position};

pub const INPUT_SIZE: usize = BOARD_SIZE * 14 + 1;
pub const V2_KING_BUCKETS: usize = 9;
pub const V2_INPUT_SIZE: usize = INPUT_SIZE + 2 * V2_KING_BUCKETS * 14 * BOARD_SIZE;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeatureSet {
    V1,
    V2,
}

impl FeatureSet {
    fn as_str(self) -> &'static str {
        match self {
            Self::V1 => "v1",
            Self::V2 => "v2",
        }
    }

    fn parse(value: &str) -> io::Result<Self> {
        match value {
            "v1" => Ok(Self::V1),
            "v2" => Ok(Self::V2),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported feature_set: {value}"),
            )),
        }
    }
}

#[derive(Clone, Debug)]
pub struct NnueModel {
    pub input_size: usize,
    pub feature_set: FeatureSet,
    pub hidden_size: usize,
    pub input_hidden: Vec<f32>,
    pub hidden_bias: Vec<f32>,
    pub hidden_output: Vec<f32>,
    pub output_bias: f32,
}

impl NnueModel {
    pub fn zeroed(hidden_size: usize) -> Self {
        Self::zeroed_with_feature_set(hidden_size, FeatureSet::V1)
    }

    pub fn zeroed_with_feature_set(hidden_size: usize, feature_set: FeatureSet) -> Self {
        let input_size = input_size_for_feature_set(feature_set);
        Self {
            input_size,
            feature_set,
            hidden_size,
            input_hidden: vec![0.0; input_size * hidden_size],
            hidden_bias: vec![0.0; hidden_size],
            hidden_output: vec![0.0; hidden_size],
            output_bias: 0.0,
        }
    }

    pub fn load_text(path: impl AsRef<Path>) -> io::Result<Self> {
        let text = fs::read_to_string(path)?;
        let mut hidden_size = None;
        let mut input_size = None;
        let mut feature_set = None;
        let mut input_hidden = None;
        let mut hidden_bias = None;
        let mut hidden_output = None;
        let mut output_bias = None;

        for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            let key = key.trim();
            let value = value.trim();
            match key {
                "hidden_size" => hidden_size = value.parse::<usize>().ok(),
                "input_size" => input_size = value.parse::<usize>().ok(),
                "feature_set" => feature_set = Some(FeatureSet::parse(value)?),
                "input_hidden" => input_hidden = Some(parse_floats(value)?),
                "hidden_bias" => hidden_bias = Some(parse_floats(value)?),
                "hidden_output" => hidden_output = Some(parse_floats(value)?),
                "output_bias" => output_bias = value.parse::<f32>().ok(),
                _ => {}
            }
        }

        let hidden_size = hidden_size
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing hidden_size"))?;
        let feature_set = feature_set.unwrap_or(FeatureSet::V1);
        let input_size = input_size.unwrap_or_else(|| input_size_for_feature_set(feature_set));
        let input_hidden = input_hidden
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing input_hidden"))?;
        let hidden_bias = hidden_bias
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing hidden_bias"))?;
        let hidden_output = hidden_output
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing hidden_output"))?;
        let output_bias = output_bias
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing output_bias"))?;

        if input_hidden.len() != input_size * hidden_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "input_hidden length mismatch",
            ));
        }
        if hidden_bias.len() != hidden_size || hidden_output.len() != hidden_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "hidden vector length mismatch",
            ));
        }

        Ok(Self {
            input_size,
            feature_set,
            hidden_size,
            input_hidden,
            hidden_bias,
            hidden_output,
            output_bias,
        })
    }

    pub fn save_text(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let text = format!(
            "feature_set: {}\ninput_size: {}\nhidden_size: {}\ninput_hidden: {}\nhidden_bias: {}\nhidden_output: {}\noutput_bias: {}\n",
            self.feature_set.as_str(),
            self.input_size,
            self.hidden_size,
            format_floats(&self.input_hidden),
            format_floats(&self.hidden_bias),
            format_floats(&self.hidden_output),
            self.output_bias
        );
        fs::write(path, text)
    }

    pub fn evaluate(&self, position: &Position) -> i32 {
        let features = match self.feature_set {
            FeatureSet::V1 => extract_sparse_features(position),
            FeatureSet::V2 => extract_sparse_features_v2(position),
        };
        let mut hidden = self.hidden_bias.clone();

        for feature in features {
            if feature >= self.input_size {
                continue;
            }
            let offset = feature * self.hidden_size;
            for hidden_idx in 0..self.hidden_size {
                hidden[hidden_idx] += self.input_hidden[offset + hidden_idx];
            }
        }

        let mut output = self.output_bias;
        for hidden_idx in 0..self.hidden_size {
            output += hidden[hidden_idx].max(0.0) * self.hidden_output[hidden_idx];
        }

        output.round() as i32
    }
}

pub fn extract_sparse_features(position: &Position) -> Vec<usize> {
    let mut features = Vec::new();
    for sq in 0..BOARD_SIZE {
        let Some(piece) = position.piece_at(sq) else {
            continue;
        };
        let piece_index = color_piece_index(piece.color, piece.kind);
        features.push(piece_index * BOARD_SIZE + sq);
    }

    if position.side_to_move() == Color::Red {
        features.push(INPUT_SIZE - 1);
    }
    features
}

pub fn extract_sparse_features_v2(position: &Position) -> Vec<usize> {
    let mut features = extract_sparse_features(position);
    let red_king_bucket = general_bucket(position, Color::Red).unwrap_or(4);
    let black_king_bucket = general_bucket(position, Color::Black).unwrap_or(4);

    for sq in 0..BOARD_SIZE {
        let Some(piece) = position.piece_at(sq) else {
            continue;
        };
        let piece_index = color_piece_index(piece.color, piece.kind);
        features.push(king_aware_feature_index(
            0,
            red_king_bucket,
            piece_index,
            sq,
        ));
        features.push(king_aware_feature_index(
            1,
            black_king_bucket,
            piece_index,
            sq,
        ));
    }

    features
}

fn input_size_for_feature_set(feature_set: FeatureSet) -> usize {
    match feature_set {
        FeatureSet::V1 => INPUT_SIZE,
        FeatureSet::V2 => V2_INPUT_SIZE,
    }
}

fn king_aware_feature_index(
    perspective: usize,
    king_bucket: usize,
    piece_index: usize,
    sq: usize,
) -> usize {
    INPUT_SIZE
        + (((perspective * V2_KING_BUCKETS + king_bucket) * 14 + piece_index) * BOARD_SIZE + sq)
}

fn general_bucket(position: &Position, color: Color) -> Option<usize> {
    for sq in 0..BOARD_SIZE {
        let Some(piece) = position.piece_at(sq) else {
            continue;
        };
        if piece.color == color && piece.kind == PieceKind::General {
            return Some(palace_bucket(color, sq));
        }
    }
    None
}

fn palace_bucket(color: Color, sq: usize) -> usize {
    let file = (sq % 9).clamp(3, 5) - 3;
    let rank = sq / 9;
    let rank = match color {
        Color::Red => rank.clamp(7, 9) - 7,
        Color::Black => rank.clamp(0, 2),
    };
    rank * 3 + file
}

fn color_piece_index(color: Color, kind: PieceKind) -> usize {
    let base = match color {
        Color::Red => 0,
        Color::Black => 7,
    };
    base + match kind {
        PieceKind::General => 0,
        PieceKind::Advisor => 1,
        PieceKind::Elephant => 2,
        PieceKind::Horse => 3,
        PieceKind::Rook => 4,
        PieceKind::Cannon => 5,
        PieceKind::Soldier => 6,
    }
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
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_features_include_side_to_move() {
        let position = Position::startpos();
        let features = extract_sparse_features(&position);
        assert!(features.contains(&(INPUT_SIZE - 1)));
        assert_eq!(features.len(), 33);
    }

    #[test]
    fn zero_model_evaluates_to_zero() {
        let position = Position::startpos();
        let model = NnueModel::zeroed(8);
        assert_eq!(model.evaluate(&position), 0);
    }

    #[test]
    fn v2_features_add_king_aware_inputs() {
        let position = Position::startpos();
        let features = extract_sparse_features_v2(&position);
        assert_eq!(features.len(), 33 + 32 * 2);
        assert!(features.iter().all(|feature| *feature < V2_INPUT_SIZE));
        assert!(features.iter().any(|feature| *feature >= INPUT_SIZE));
    }

    #[test]
    fn zero_v2_model_evaluates_to_zero() {
        let position = Position::startpos();
        let model = NnueModel::zeroed_with_feature_set(8, FeatureSet::V2);
        assert_eq!(model.input_size, V2_INPUT_SIZE);
        assert_eq!(model.evaluate(&position), 0);
    }
}
