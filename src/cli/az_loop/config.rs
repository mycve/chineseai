use crate::cli::az_loop_config::AzLoopFileConfig;
use chineseai::az::AzLoopConfig;
use chineseai::infra::version::AZ_LOOP_PROGRESS_VERSION;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

pub(crate) fn best_model_path(model_path: &str) -> PathBuf {
    Path::new(model_path)
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .join("best.safetensors")
}

pub(crate) fn az_loop_progress_path(config_path: &str) -> PathBuf {
    PathBuf::from(format!("{config_path}.progress"))
}

pub(crate) fn az_loop_replay_snapshot_path(config_path: &str) -> PathBuf {
    PathBuf::from(format!("{config_path}.replay.lz4"))
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AzLoopProgressState {
    pub(crate) format_version: u32,
    pub(crate) next_update: usize,
    pub(crate) nemesis_update: Option<u64>,
    pub(crate) generated_games: u64,
    pub(crate) generated_samples: u64,
}

impl Default for AzLoopProgressState {
    fn default() -> Self {
        Self {
            format_version: AZ_LOOP_PROGRESS_VERSION,
            next_update: 1,
            nemesis_update: None,
            generated_games: 0,
            generated_samples: 0,
        }
    }
}

impl AzLoopProgressState {
    pub(crate) fn normalize(mut self) -> Self {
        if self.format_version != AZ_LOOP_PROGRESS_VERSION {
            panic!(
                "unsupported AZ loop progress version {}; expected {}",
                self.format_version, AZ_LOOP_PROGRESS_VERSION
            );
        }
        self.next_update = self.next_update.max(1);
        self
    }
}

pub(crate) fn load_az_loop_progress(config_path: &str) -> AzLoopProgressState {
    let path = az_loop_progress_path(config_path);
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return AzLoopProgressState::default();
        }
        Err(err) => panic!("failed to read `{}`: {err}", path.display()),
    };
    toml::from_str::<AzLoopProgressState>(&text)
        .unwrap_or_else(|err| panic!("failed to parse `{}`: {err}", path.display()))
        .normalize()
}

pub(crate) fn save_az_loop_progress(config_path: &str, state: &AzLoopProgressState) {
    let path = az_loop_progress_path(config_path);
    fs::write(
        &path,
        toml::to_string_pretty(&state.clone().normalize()).unwrap(),
    )
    .unwrap_or_else(|err| panic!("failed to write `{}`: {err}", path.display()));
}

pub(crate) fn save_az_loop_progress_pair(
    config_path: &str,
    next_update: usize,
    nemesis_update: Option<u64>,
    generated_games: u64,
    generated_samples: u64,
) {
    save_az_loop_progress(
        config_path,
        &AzLoopProgressState {
            next_update,
            nemesis_update,
            generated_games,
            generated_samples,
            ..Default::default()
        },
    );
}

pub(crate) fn build_az_loop_config(
    config: &AzLoopFileConfig,
    seed: u64,
    workers: usize,
    generation_update: u32,
    opening_positions: &Arc<[chineseai::az::AzStartSnapshot]>,
) -> AzLoopConfig {
    AzLoopConfig {
        games: 1,
        max_plies: config.max_plies,
        rule60_max_ply: config.sixty_move_rule.then_some(config.rule60_max_ply),
        simulations: config.simulations,
        seed,
        workers,
        generation_update,
        temperature_start: config.temperature_start,
        temperature_cutoff_plies: config.temperature_cutoff_plies,
        temperature_visit_offset: config.temperature_visit_offset,
        temperature_endgame: config.temperature_endgame,
        temperature_decay_delay_plies: config.temperature_decay_delay_plies,
        temperature_decay_plies: config.temperature_decay_plies,
        cpuct: config.cpuct,
        cpuct_at_root: config.cpuct_at_root,
        cpuct_base: config.cpuct_base,
        cpuct_factor: config.cpuct_factor,
        cpuct_base_at_root: config.cpuct_base_at_root,
        cpuct_factor_at_root: config.cpuct_factor_at_root,
        root_dirichlet_alpha: config.root_dirichlet_alpha,
        root_exploration_fraction: config.root_exploration_fraction,
        fpu_value: config.fpu_value,
        fpu_value_at_root: config.fpu_value_at_root,
        fpu_absolute_at_root: config.fpu_absolute_at_root,
        minimum_kldgain_per_node: config.minimum_kldgain_per_node,
        draw_score: config.draw_score,
        policy_softmax_temp: config.policy_softmax_temp,
        opening_positions: Arc::clone(opening_positions),
        mirror_probability: config.mirror_probability,
        record_fens: false,
        moves_left_params: config.moves_left_params(),
        mate_search_plies: config.mate_search_plies,
    }
}

#[cfg(test)]
mod progress_tests {
    use super::*;

    #[test]
    fn loading_progress_keeps_it_for_restarts_after_failed_restore() {
        let path = std::env::temp_dir().join(format!(
            "chineseai-progress-retry-{}.toml",
            std::process::id()
        ));
        let config_path = path.to_str().unwrap();
        let progress_path = az_loop_progress_path(config_path);
        assert_eq!(load_az_loop_progress(config_path).next_update, 1);
        save_az_loop_progress_pair(config_path, 44109, Some(43000), 123, 456);
        for _ in 0..2 {
            let loaded = load_az_loop_progress(config_path);
            assert_eq!(loaded.next_update, 44109);
            assert_eq!(loaded.nemesis_update, Some(43000));
            assert_eq!(loaded.generated_games, 123);
            assert_eq!(loaded.generated_samples, 456);
            assert!(progress_path.exists());
        }
        std::fs::remove_file(progress_path).unwrap();
    }
}
