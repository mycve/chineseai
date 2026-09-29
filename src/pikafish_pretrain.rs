//! Pikafish 教师分数预训练与候选自博弈终局结果训练。

use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use candle_core::{Device, Tensor};
use candle_nn::{Optimizer, SGD};

use crate::ab::pikafish_candle::{PikafishExample, PikafishModel};
use crate::xiangqi::{Color, Position};

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
    pub skipped_unknown_results: usize,
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

fn parse_result_row(
    line: &str,
    fen_col: usize,
    result_col: usize,
) -> io::Result<Option<(Position, f32)>> {
    let fields: Vec<_> = line.split('\t').collect();
    let result = fields
        .get(result_col)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing red_result column"))?;
    if *result == "?" {
        return Ok(None);
    }
    let red_value = match *result {
        "1" => 1.0,
        "0" => -1.0,
        "1/2" => 0.0,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid red_result `{result}`"),
            ));
        }
    };
    let fen = fields
        .get(fen_col)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing fen column"))?;
    let position = Position::from_fen(fen)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, format!("invalid FEN: {err}")))?;
    let target = if position.side_to_move() == Color::Red {
        red_value
    } else {
        -red_value
    };
    Ok(Some((position, target)))
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LabelSource {
    TeacherScore,
    CandidateResult,
}

/// 读取带明确来源的 TSV，训练并保存项目自己的 f32 safetensors。
fn train_tsv(
    model: &PikafishModel,
    input: &Path,
    output: &Path,
    config: PretrainConfig,
    label_source: LabelSource,
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
    let source_col = find_column("source")?;
    let label_col = find_column(match label_source {
        LabelSource::TeacherScore => "score_cp",
        LabelSource::CandidateResult => "red_result",
    })?;
    let mut report = PretrainReport::default();
    let mut examples = Vec::with_capacity(config.batch_size);
    let mut targets = Vec::with_capacity(config.batch_size);
    let mut weighted_loss = 0.0;
    for line in lines {
        let line = line?;
        let fields: Vec<_> = line.split('\t').collect();
        let source = fields.get(source_col).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "missing source column value")
        })?;
        let expected = match label_source {
            LabelSource::TeacherScore => "pikafish",
            LabelSource::CandidateResult => "candidate",
        };
        if *source != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("expected source `{expected}`, got `{source}`"),
            ));
        }
        let parsed = match label_source {
            LabelSource::TeacherScore => parse_row(&line, fen_col, label_col)?,
            LabelSource::CandidateResult => parse_result_row(&line, fen_col, label_col)?,
        };
        let Some((position, target)) = parsed else {
            match label_source {
                LabelSource::TeacherScore => report.skipped_unknown_scores += 1,
                LabelSource::CandidateResult => report.skipped_unknown_results += 1,
            }
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
            "no usable training labels in Pikafish TSV",
        ));
    }
    report.mean_loss = weighted_loss / report.used_samples as f64;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    model.save(output).map_err(candle_io)?;
    Ok(report)
}

/// 仅从 Pikafish 引擎的 `score_cp` 训练；要求 TSV 中 `source=pikafish`。
pub fn train_teacher_tsv(
    model: &PikafishModel,
    input: &Path,
    output: &Path,
    config: PretrainConfig,
) -> io::Result<PretrainReport> {
    train_tsv(model, input, output, config, LabelSource::TeacherScore)
}

/// 从候选自博弈的真实终局结果训练；截断对局 `?` 不提供标签。
pub fn train_candidate_results_tsv(
    model: &PikafishModel,
    input: &Path,
    output: &Path,
    config: PretrainConfig,
) -> io::Result<PretrainReport> {
    train_tsv(model, input, output, config, LabelSource::CandidateResult)
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
    fn result_label_flips_with_side_to_move_and_skips_truncation() {
        let red = Position::startpos().to_fen();
        let black = red.replacen(" w ", " b ", 1);
        for (result, red_target, black_target) in
            [("1", 1.0, -1.0), ("0", -1.0, 1.0), ("1/2", 0.0, 0.0)]
        {
            assert_eq!(
                parse_result_row(&format!("{red}\t{result}"), 0, 1)
                    .unwrap()
                    .unwrap()
                    .1,
                red_target
            );
            assert_eq!(
                parse_result_row(&format!("{black}\t{result}"), 0, 1)
                    .unwrap()
                    .unwrap()
                    .1,
                black_target
            );
        }
        assert!(
            parse_result_row(&format!("{red}\t?"), 0, 1)
                .unwrap()
                .is_none()
        );
        assert!(parse_result_row(&format!("{red}\tbad"), 0, 1).is_err());
    }

    #[test]
    fn teacher_rejects_candidate_and_untagged_tsv() -> io::Result<()> {
        let dir = std::env::current_dir()?
            .join("target")
            .join("fast")
            .join(format!(
                "chineseai-pretrain-source-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
        std::fs::create_dir_all(&dir)?;
        let input = dir.join("samples.tsv");
        let output = dir.join("candidate.safetensors");
        let model = PikafishModel::with_shape(
            PikafishShape {
                psq_features: 8,
                threat_features: 8,
            },
            &Device::Cpu,
        )
        .map_err(candle_io)?;
        let fen = Position::startpos().to_fen();
        std::fs::write(
            &input,
            format!("fen\tscore_cp\tred_result\tsource\n{fen}\t100\t1\tcandidate\n"),
        )?;
        let config = PretrainConfig::default();
        assert!(train_teacher_tsv(&model, &input, &output, config).is_err());
        std::fs::write(
            &input,
            format!("fen\tscore_cp\tred_result\n{fen}\t100\t1\n"),
        )?;
        assert!(train_teacher_tsv(&model, &input, &output, config).is_err());
        assert!(!output.exists());
        let _ = std::fs::remove_dir_all(dir);
        Ok(())
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
