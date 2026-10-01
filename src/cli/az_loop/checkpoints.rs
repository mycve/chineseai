use crate::az_loop_config::AzLoopFileConfig;
use chineseai::az::AzNnue;
use std::{
    fs, io,
    path::{Path, PathBuf},
};
use tensorboard_rs::summary_writer::SummaryWriter;

pub(crate) fn save_model(model: &AzNnue, path: &Path) {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).unwrap_or_else(|err| {
            panic!(
                "failed to create model directory `{}`: {err}",
                parent.display()
            );
        });
    }
    model
        .save(path)
        .unwrap_or_else(|err| panic!("failed to save model `{}`: {err}", path.display()));
}

pub(crate) fn tensorboard_encoded_subdir(config: &AzLoopFileConfig) -> String {
    fn f32_slug(x: f32) -> String {
        if x == 0.0 {
            return "0".to_string();
        }
        let s = format!("{:.8}", x)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
        if s.is_empty() || s == "-" {
            return "0".to_string();
        }
        s.replace('.', "p").replace('-', "m")
    }

    let encoded = format!(
        concat!(
            "sim{}_sspu{}_bs{}_lr{}_h{}_mxp{}_sr{}_r60{}_wk{}_",
            "shuf{}_rrw{}_sgd09nesterov_cp{}_cpr{}_fv{}_fvr{}_pst{}_tb{}_teg{}_tdd{}_tde{}_rc{}_",
            "tspu{}_mp{}_cpi{}_ai{}_as{}_acp{}_acpr{}_apst{}_rda{}_ref{}_sd{}"
        ),
        config.simulations,
        config.selfplay_samples_per_update,
        config.batch_size,
        f32_slug(config.lr),
        config.hidden_size,
        config.max_plies,
        u8::from(config.sixty_move_rule),
        config.rule60_max_ply,
        config.workers,
        config.shuffle_size,
        config.replay_recent_games,
        f32_slug(config.cpuct),
        f32_slug(config.cpuct_at_root),
        f32_slug(config.fpu_value),
        f32_slug(config.fpu_value_at_root),
        f32_slug(config.policy_softmax_temp),
        f32_slug(config.temperature_start),
        f32_slug(config.temperature_endgame),
        config.temperature_decay_delay_plies,
        config.temperature_decay_plies,
        config.replay_capacity,
        config.train_samples_per_update,
        f32_slug(config.mirror_probability),
        config.checkpoint_interval,
        config.arena_interval,
        config.arena_simulations,
        f32_slug(config.arena_cpuct),
        f32_slug(config.arena_cpuct_at_root),
        f32_slug(config.arena_policy_softmax_temp),
        f32_slug(config.root_dirichlet_alpha),
        f32_slug(config.root_exploration_fraction),
        config.seed,
    );
    if encoded.len() <= 180 {
        return encoded;
    }

    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in encoded.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    format!(
        "sim{}_bs{}_lr{}_h{}_sd{}_cfg{:016x}",
        config.simulations,
        config.batch_size,
        f32_slug(config.lr),
        config.hidden_size,
        config.seed,
        hash
    )
}

pub(crate) fn tensorboard_effective_logdir(config: &AzLoopFileConfig) -> PathBuf {
    Path::new(&config.tensorboard_logdir).join(tensorboard_encoded_subdir(config))
}

pub(crate) fn checkpoint_path(model_path: &str, checkpoint_dir: &str, update: usize) -> PathBuf {
    let base = Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.safetensors");
    Path::new(checkpoint_dir).join(format!("update-{update:06}-{base}"))
}

pub(crate) fn optimizer_checkpoint_path(model_path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.sgd.safetensors", model_path.display()))
}

pub(crate) fn best_checkpoint_path(model_path: &str, checkpoint_dir: &str, update: usize) -> PathBuf {
    let base = Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.safetensors");
    Path::new(checkpoint_dir).join(format!("best-update-{update:06}-{base}"))
}

pub(crate) fn save_checkpoint_model(
    model: &AzNnue,
    model_path: &str,
    checkpoint_dir: &str,
    update: usize,
) -> PathBuf {
    fs::create_dir_all(checkpoint_dir).unwrap_or_else(|err| {
        panic!("failed to create checkpoint dir `{checkpoint_dir}`: {err}");
    });
    let path = checkpoint_path(model_path, checkpoint_dir, update);
    save_model(model, &path);
    path
}

pub(crate) fn save_best_checkpoint_model(
    model: &AzNnue,
    model_path: &str,
    checkpoint_dir: &str,
    update: usize,
) -> PathBuf {
    fs::create_dir_all(checkpoint_dir).unwrap_or_else(|err| {
        panic!("failed to create checkpoint dir `{checkpoint_dir}`: {err}");
    });
    let path = best_checkpoint_path(model_path, checkpoint_dir, update);
    save_model(model, &path);
    path
}

pub(crate) fn champion_checkpoint_paths(model_path: &str, checkpoint_dir: &str) -> io::Result<Vec<PathBuf>> {
    let directory = Path::new(checkpoint_dir);
    if !directory.exists() {
        return Ok(Vec::new());
    }
    let base = Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.safetensors");
    let prefix = "best-update-";
    let suffix = format!("-{base}");
    let mut paths = fs::read_dir(directory)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(prefix) && name.ends_with(&suffix))
        })
        .collect::<Vec<_>>();
    paths.sort_by_key(|path| checkpoint_number(path).unwrap_or(0));
    Ok(paths)
}

pub(crate) fn prune_old_checkpoints(
    model_path: &str,
    checkpoint_dir: &str,
    max_checkpoints: usize,
) -> io::Result<()> {
    if max_checkpoints == 0 {
        return Ok(());
    }
    let base = Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.safetensors")
        .to_string();
    let prefix = "update-";
    let suffix = format!("-{base}");
    let mut entries = fs::read_dir(checkpoint_dir)?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let path = entry.path();
            let name = path.file_name()?.to_str()?;
            if !name.starts_with(prefix) || !name.ends_with(&suffix) {
                return None;
            }
            let update_text = name
                .strip_prefix(prefix)?
                .strip_suffix(&suffix)?
                .split('-')
                .next()?;
            let update = update_text.parse::<usize>().ok()?;
            Some((update, name.to_string(), path))
        })
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    let to_remove = entries.len().saturating_sub(max_checkpoints);
    for (_, _, path) in entries.into_iter().take(to_remove) {
        let optimizer_path = optimizer_checkpoint_path(&path);
        fs::remove_file(path)?;
        if optimizer_path.exists() {
            fs::remove_file(optimizer_path)?;
        }
    }
    Ok(())
}

pub(crate) fn log_scalar(writer: &mut SummaryWriter, tag: &str, step: usize, value: f32) {
    writer.add_scalar(tag, value, step);
}

pub(crate) fn checkpoint_number(path: &Path) -> Option<u64> {
    let name = path.file_name()?.to_str()?;
    let mut last = None;
    let mut value = 0u64;
    let mut active = false;
    for byte in name.bytes() {
        if byte.is_ascii_digit() {
            value = value
                .saturating_mul(10)
                .saturating_add(u64::from(byte - b'0'));
            active = true;
        } else if active {
            last = Some(value);
            value = 0;
            active = false;
        }
    }
    if active { Some(value) } else { last }
}
