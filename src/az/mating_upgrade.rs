//! v54/v55/v56 检查点的显式一次性升级；普通加载器只接受当前格式。
use std::{fs, io, path::Path};

use candle_core::{Device, Var};
use candle_nn::VarMap;

use super::inference::{candle_io_error, insert_candle_var, load_candle_f32_tensor};
use super::{AzNnue, GOAL_CONTEXT_SIZE, MATING_HEAD_SIZE, MATING_TARGET_COUNT};

// v54 的 18 个统计头映射到确认后的 12 类；高钓马、双马饮泉从零初始化。
const OLD_HEADS: [Option<usize>; MATING_HEAD_SIZE] = [
    Some(0),
    Some(2),
    Some(1),
    Some(6),
    Some(7),
    Some(8),
    Some(9),
    Some(10),
    None,
    None,
    Some(11),
    Some(5),
];

const V56_HEADS: [usize; MATING_TARGET_COUNT] = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13];

fn gather_rows(values: &[f32], width: usize, rows: &[usize]) -> io::Result<Vec<f32>> {
    if width == 0
        || values.len() % width != 0
        || rows.iter().any(|&row| row >= values.len() / width)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "旧杀法张量形状无效",
        ));
    }
    Ok(rows
        .iter()
        .flat_map(|&row| values[row * width..(row + 1) * width].iter().copied())
        .collect())
}

fn v56_goal_rows() -> Vec<usize> {
    (0..2)
        .flat_map(|role| V56_HEADS.map(|row| role * 14 + row))
        .collect()
}

fn remap_heads(old: &[f32], width: usize, initial: f32) -> io::Result<Vec<f32>> {
    if old.len() != 18 * width {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "v54 杀法统计头应为 18 行",
        ));
    }
    let mut result = vec![initial; MATING_HEAD_SIZE * width];
    for (row, source) in OLD_HEADS.into_iter().enumerate() {
        if let Some(source) = source {
            result[row * width..(row + 1) * width]
                .copy_from_slice(&old[source * width..(source + 1) * width]);
        }
    }
    Ok(result)
}

impl AzNnue {
    /// 保留全部旧权重，新目标的条件编码初始化为零。输出必须不存在。
    pub fn upgrade_mating_targets(
        input: impl AsRef<Path>,
        output: impl AsRef<Path>,
    ) -> io::Result<()> {
        let tensors = unsafe {
            candle_core::safetensors::MmapedSafetensors::new(input.as_ref())
                .map_err(candle_io_error)?
        };
        let version = load_candle_f32_tensor(&tensors, "az_model_format_version")?;
        let legacy = version == [54.0];
        if !legacy && version != [55.0] && version != [56.0] {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "杀法升级仅接受 v54/v55/v56 模型",
            ));
        }
        let old = tensors
            .load("goal_hidden", &Device::Cpu)
            .map_err(candle_io_error)?;
        let (rows, hidden) = old.dims2().map_err(candle_io_error)?;
        if rows != if legacy { 4 } else { 28 } || hidden == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "旧模型 goal_hidden 形状无效",
            ));
        }
        let old = old
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(candle_io_error)?;
        let mut expanded = vec![0.0; GOAL_CONTEXT_SIZE * hidden];
        let mapping = if legacy {
            vec![
                (0, 0),
                (1, 1),
                (2, MATING_TARGET_COUNT),
                (3, MATING_TARGET_COUNT + 1),
            ]
        } else {
            v56_goal_rows()
                .into_iter()
                .enumerate()
                .map(|(destination, source)| (source, destination))
                .collect()
        };
        for (source, destination) in mapping {
            expanded[destination * hidden..(destination + 1) * hidden]
                .copy_from_slice(&old[source * hidden..(source + 1) * hidden]);
        }
        let vars = VarMap::new();
        for (name, _) in tensors.tensors() {
            match name.as_str() {
                "az_model_format_version" => insert_candle_var(
                    &vars,
                    &name,
                    &[crate::infra::version::MODEL_FORMAT_VERSION],
                    (1,),
                )?,
                "goal_hidden" => {
                    insert_candle_var(&vars, &name, &expanded, (GOAL_CONTEXT_SIZE, hidden))?
                }
                "mating_output" => {
                    let old = load_candle_f32_tensor(&tensors, &name)?;
                    if !legacy && old.len() != 14 * hidden {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "v55/v56 杀法头应为14行",
                        ));
                    }
                    let values = if legacy {
                        remap_heads(&old, hidden, 0.0)?
                    } else {
                        gather_rows(&old, hidden, &V56_HEADS)?
                    };
                    insert_candle_var(&vars, &name, &values, (MATING_HEAD_SIZE, hidden))?;
                }
                "mating_bias" => {
                    let old = load_candle_f32_tensor(&tensors, &name)?;
                    if !legacy && old.len() != 14 {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "v55/v56 杀法头应为14行",
                        ));
                    }
                    let values = if legacy {
                        remap_heads(&old, 1, -4.0)?
                    } else {
                        gather_rows(&old, 1, &V56_HEADS)?
                    };
                    insert_candle_var(&vars, &name, &values, (MATING_HEAD_SIZE,))?;
                }
                "goal_scale" => {
                    let old = load_candle_f32_tensor(&tensors, &name)?;
                    if old.len() != 28 * hidden {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "v56 条件门控应为28行",
                        ));
                    }
                    let values = gather_rows(&old, hidden, &v56_goal_rows())?;
                    insert_candle_var(&vars, &name, &values, (GOAL_CONTEXT_SIZE, hidden))?;
                }
                _ => {
                    let tensor = tensors.load(&name, &Device::Cpu).map_err(candle_io_error)?;
                    let var = Var::from_tensor(&tensor).map_err(candle_io_error)?;
                    vars.data().lock().unwrap().insert(name, var);
                }
            }
        }
        if version != [56.0] {
            insert_candle_var(
                &vars,
                "goal_scale",
                &vec![0.0; GOAL_CONTEXT_SIZE * hidden],
                (GOAL_CONTEXT_SIZE, hidden),
            )?;
        }
        // 先独占创建，拒绝覆盖输入、已有输出或其他检查点。
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output.as_ref())?;
        let saved = vars.save(output.as_ref()).map_err(candle_io_error);
        drop(vars);
        drop(tensors);
        if let Err(err) = saved.and_then(|()| Self::load(output.as_ref()).map(|_| ())) {
            let _ = fs::remove_file(output.as_ref());
            return Err(err);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::az::arch::az_weight_tensors;

    fn pad_rows(values: &[f32], width: usize, rows: &[usize], old_rows: usize) -> Vec<f32> {
        let mut old = vec![99.0; old_rows * width];
        for (current, &previous) in rows.iter().enumerate() {
            old[previous * width..(previous + 1) * width]
                .copy_from_slice(&values[current * width..(current + 1) * width]);
        }
        old
    }

    #[test]
    fn explicit_upgrade_preserves_all_weights_and_rejects_overwrites() {
        let base =
            std::env::temp_dir().join(format!("chineseai-goal-upgrade-{}", std::process::id()));
        let current = base.with_extension("current.safetensors");
        let old = base.with_extension("v54.safetensors");
        let output = base.with_extension("v56.safetensors");
        let mut model = AzNnue::random(8, 1900);
        let rows = [0, 1, MATING_TARGET_COUNT, MATING_TARGET_COUNT + 1];
        for (i, &row) in rows.iter().enumerate() {
            model.goal_hidden[row * 8..(row + 1) * 8].fill((i + 1) as f32 * 0.2);
        }
        for (row, source) in OLD_HEADS.iter().enumerate() {
            if source.is_some() {
                model.mating_output[row * 8..(row + 1) * 8].fill((row + 1) as f32 * 0.02);
                model.mating_bias[row] = (row + 1) as f32 * 0.03;
            }
        }
        model.save(&current).unwrap();
        let tensors =
            unsafe { candle_core::safetensors::MmapedSafetensors::new(&current).unwrap() };
        let vars = VarMap::new();
        for (name, _) in tensors.tensors() {
            match name.as_str() {
                "az_model_format_version" => {
                    insert_candle_var(&vars, &name, &[54.0], (1,)).unwrap()
                }
                "goal_scale" => {}
                "goal_hidden" => {
                    let values = rows
                        .iter()
                        .flat_map(|&row| model.goal_hidden[row * 8..(row + 1) * 8].iter().copied())
                        .collect::<Vec<_>>();
                    insert_candle_var(&vars, &name, &values, (4, 8)).unwrap();
                }
                "mating_output" | "mating_bias" => {
                    let width = if name == "mating_output" { 8 } else { 1 };
                    let values = if width == 8 {
                        &model.mating_output
                    } else {
                        &model.mating_bias
                    };
                    let mut legacy = vec![-0.4; 18 * width];
                    for (row, source) in OLD_HEADS.into_iter().enumerate() {
                        if let Some(source) = source {
                            legacy[source * width..(source + 1) * width]
                                .copy_from_slice(&values[row * width..(row + 1) * width]);
                        }
                    }
                    if width == 8 {
                        insert_candle_var(&vars, &name, &legacy, (18, 8)).unwrap();
                    } else {
                        insert_candle_var(&vars, &name, &legacy, (18,)).unwrap();
                    }
                }
                _ => {
                    vars.data().lock().unwrap().insert(
                        name.clone(),
                        Var::from_tensor(&tensors.load(&name, &Device::Cpu).unwrap()).unwrap(),
                    );
                }
            }
        }
        vars.save(&old).unwrap();
        drop(vars);
        drop(tensors);
        assert!(AzNnue::load(&old).is_err(), "旧模型须显式转换");
        AzNnue::upgrade_mating_targets(&old, &output).unwrap();
        let upgraded = AzNnue::load(&output).unwrap();
        macro_rules! unchanged {
            ($field:ident, [$($dim:expr),+], $decay:expr) => {
                assert_eq!(model.$field, upgraded.$field, "{}", stringify!($field));
            };
        }
        az_weight_tensors!(unchanged, 8);
        assert_eq!(
            AzNnue::upgrade_mating_targets(&old, &output)
                .unwrap_err()
                .kind(),
            io::ErrorKind::AlreadyExists
        );
        assert!(AzNnue::upgrade_mating_targets(&current, base.with_extension("invalid")).is_err());
        for path in [current, old, output] {
            fs::remove_file(path).unwrap();
        }
    }
    #[test]
    fn v55_v56_upgrade_preserves_remaining_weights_and_learned_gates() {
        for version in [55.0, 56.0] {
            let base = std::env::temp_dir().join(format!(
                "chineseai-v{version}-gate-upgrade-{}",
                std::process::id()
            ));
            let current = base.with_extension("current.safetensors");
            let old = base.with_extension(format!("v{version}.safetensors"));
            let output = base.with_extension("v57.safetensors");
            let mut model = AzNnue::random(8, 1903);
            if version == 56.0 {
                model.goal_scale.fill(0.3);
            }
            for (i, value) in model.goal_hidden.iter_mut().enumerate() {
                *value = (i + 1) as f32 * 0.003;
            }
            model.goal_output.fill(0.2);
            model.save(&current).unwrap();
            let tensors =
                unsafe { candle_core::safetensors::MmapedSafetensors::new(&current).unwrap() };
            let vars = VarMap::new();
            for (name, _) in tensors.tensors() {
                match name.as_str() {
                    "goal_scale" if version == 55.0 => {}
                    "goal_hidden" | "goal_scale" => {
                        let values = if name == "goal_hidden" {
                            &model.goal_hidden
                        } else {
                            &model.goal_scale
                        };
                        insert_candle_var(
                            &vars,
                            &name,
                            &pad_rows(values, 8, &v56_goal_rows(), 28),
                            (28, 8),
                        )
                        .unwrap();
                    }
                    "mating_output" => {
                        insert_candle_var(
                            &vars,
                            &name,
                            &pad_rows(&model.mating_output, 8, &V56_HEADS, 14),
                            (14, 8),
                        )
                        .unwrap();
                    }
                    "mating_bias" => {
                        insert_candle_var(
                            &vars,
                            &name,
                            &pad_rows(&model.mating_bias, 1, &V56_HEADS, 14),
                            (14,),
                        )
                        .unwrap();
                    }
                    "az_model_format_version" => {
                        insert_candle_var(&vars, &name, &[version], (1,)).unwrap()
                    }
                    _ => {
                        vars.data().lock().unwrap().insert(
                            name.clone(),
                            Var::from_tensor(&tensors.load(&name, &Device::Cpu).unwrap()).unwrap(),
                        );
                    }
                }
            }
            vars.save(&old).unwrap();
            drop(vars);
            drop(tensors);
            assert!(AzNnue::load(&old).is_err());
            AzNnue::upgrade_mating_targets(&old, &output).unwrap();
            let upgraded = AzNnue::load(&output).unwrap();
            macro_rules! unchanged {
                ($field:ident, [$($dim:expr),+], $decay:expr) => {
                    assert_eq!(model.$field, upgraded.$field, "{}", stringify!($field));
                };
            }
            az_weight_tensors!(unchanged, 8);
            if version == 55.0 {
                assert!(
                    upgraded
                        .goal_scale_activation
                        .iter()
                        .all(|value| *value == 0.0)
                );
            }
            for path in [current, old, output] {
                fs::remove_file(path).unwrap();
            }
        }
    }
}
