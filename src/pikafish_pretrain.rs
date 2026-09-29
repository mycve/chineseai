//! Pikafish 自博弈 TSV 的教师分数预训练。标签仅使用 UCI `score cp`，
//! 其定义是当前 FEN 行棋方视角，因此不按红黑方翻转符号。

use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use candle_core::{Device, Tensor};
use candle_nn::{Optimizer, SGD};

use crate::ab::pikafish_candle::{PikafishExample, PikafishModel};
use crate::xiangqi::Position;

#[derive(Clone, Copy, Debug)]
pub struct PretrainConfig {
    pub batch_size: usize,
    pub max_samples: usize,
    pub learning_rate: f64,
}

impl Default for PretrainConfig {
    fn default() -> Self {
        Self {
            batch_size: 8,
            max_samples: 1_024,
            learning_rate: 1.0e-4,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PretrainReport {
    pub used_samples: usize,
    pub skipped_unknown_scores: usize,
    pub skipped_unsupported_features: usize,
    pub steps: usize,
    pub mean_loss: f64,
}

fn candle_io(err: impl std::fmt::Display) -> io::Error {
    io::Error::other(err.to_string())
}

/// 将行棋方视角 centipawn 分数约束到 [-1, 1]，供浮点标量价值头学习。
pub fn cp_to_value(score_cp: i32) -> f32 {
    ((score_cp as f64) / 600.0).tanh() as f32
}

fn parse_row(line: &str, fen_col: usize, score_col: usize) -> io::Result<Option<(Position, f32)>> {
    let fields: Vec<_> = line.split('\t').collect();
    let score = fields
        .get(score_col)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing score_cp column"))?;
    if *score == "?" {
        return Ok(None);
    }
    let score_cp: i32 = score.parse().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid score_cp `{score}`"),
        )
    })?;
    let fen = fields
        .get(fen_col)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing fen column"))?;
    let position = Position::from_fen(fen)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, format!("invalid FEN: {err}")))?;
    Ok(Some((position, cp_to_value(score_cp))))
}

fn train_batch(
    model: &PikafishModel,
    optimizer: &mut SGD,
    examples: &[PikafishExample],
    targets: &[f32],
    device: &Device,
) -> candle_core::Result<f64> {
    let prediction = model.forward(examples)?;
    let target = Tensor::from_vec(targets.to_vec(), (targets.len(), 1), device)?;
    let loss = (&prediction - &target)?.sqr()?.mean_all()?;
    let value = loss.to_scalar::<f32>()? as f64;
    if !value.is_finite() {
        candle_core::bail!("Pikafish pretrain loss is not finite");
    }
    optimizer.backward_step(&loss)?;
    Ok(value)
}

/// 读取 `pikafish-selfplay` TSV，训练模型并保存项目自己的 f32 safetensors。
/// 该文件不能作为 Pikafish 的量化 `.nnue` 直接加载。
pub fn train_teacher_tsv(
    model: &PikafishModel,
    input: &Path,
    output: &Path,
    config: PretrainConfig,
) -> io::Result<PretrainReport> {
    if config.batch_size == 0
        || config.max_samples == 0
        || !config.learning_rate.is_finite()
        || config.learning_rate <= 0.0
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "batch_size, max_samples and learning_rate must be positive",
        ));
    }
    let device = model.vars()[0].device().clone();
    let mut optimizer = SGD::new(model.vars(), config.learning_rate).map_err(candle_io)?;
    let mut lines = BufReader::new(File::open(input)?).lines();
    let header = lines
        .next()
        .transpose()?
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "empty Pikafish TSV"))?;
    let columns: Vec<_> = header.split('\t').collect();
    let find_column = |name| {
        columns
            .iter()
            .position(|&value| value == name)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("missing `{name}` column"),
                )
            })
    };
    let fen_col = find_column("fen")?;
    let score_col = find_column("score_cp")?;
    let mut report = PretrainReport::default();
    let mut examples = Vec::with_capacity(config.batch_size);
    let mut targets = Vec::with_capacity(config.batch_size);
    let mut weighted_loss = 0.0;
    for line in lines {
        let line = line?;
        let Some((position, target)) = parse_row(&line, fen_col, score_col)? else {
            report.skipped_unknown_scores += 1;
            continue;
        };
        let Some(example) = PikafishExample::from_position(&position) else {
            report.skipped_unsupported_features += 1;
            continue;
        };
        examples.push(example);
        targets.push(target);
        if examples.len() == config.batch_size
            || report.used_samples + examples.len() == config.max_samples
        {
            let loss = train_batch(model, &mut optimizer, &examples, &targets, &device)
                .map_err(candle_io)?;
            weighted_loss += loss * examples.len() as f64;
            report.used_samples += examples.len();
            report.steps += 1;
            examples.clear();
            targets.clear();
            if report.used_samples == config.max_samples {
                break;
            }
        }
    }
    if !examples.is_empty() && report.used_samples < config.max_samples {
        let loss =
            train_batch(model, &mut optimizer, &examples, &targets, &device).map_err(candle_io)?;
        weighted_loss += loss * examples.len() as f64;
        report.used_samples += examples.len();
        report.steps += 1;
    }
    if report.used_samples == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "no usable score_cp samples in Pikafish TSV",
        ));
    }
    report.mean_loss = weighted_loss / report.used_samples as f64;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    model.save(output).map_err(candle_io)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ab::pikafish_candle::PikafishShape;

    #[test]
    fn tsv_score_uses_side_to_move_without_color_flip() {
        let white = Position::startpos().to_fen();
        let black = white.replacen(" w ", " b ", 1);
        assert_eq!(
            parse_row(&format!("{white}\t100"), 0, 1)
                .unwrap()
                .unwrap()
                .1,
            cp_to_value(100)
        );
        assert_eq!(
            parse_row(&format!("{black}\t100"), 0, 1)
                .unwrap()
                .unwrap()
                .1,
            cp_to_value(100)
        );
        assert!(parse_row(&format!("{white}\t?"), 0, 1).unwrap().is_none());
    }

    #[test]
    fn tiny_model_training_changes_loss() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let model = PikafishModel::with_shape(
            PikafishShape {
                psq_features: 8,
                threat_features: 8,
            },
            &device,
        )?;
        let example = PikafishExample {
            psq: [vec![1, 2], vec![3]],
            threats: [vec![1], vec![2]],
            psqt_bucket: 4,
            layer_stack: 7,
        };
        let mut optimizer = SGD::new(model.vars(), 0.01)?;
        let before = model
            .forward(std::slice::from_ref(&example))?
            .to_vec2::<f32>()?[0][0];
        let loss = train_batch(&model, &mut optimizer, &[example.clone()], &[0.8], &device)?;
        let after = model.forward(&[example])?.to_vec2::<f32>()?[0][0];
        assert!(loss > 0.0);
        assert_ne!(before, after);
        Ok(())
    }
}
