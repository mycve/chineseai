mod args;
mod az_bench;
mod az_calibrate_policy;
mod az_init;
mod az_loop;
mod az_loop_config;
mod az_policy_scale;
mod az_search;
mod dispatch;
mod reporting;
mod training_console;
mod vs_pikafish;

#[cfg(test)]
mod reporting_tests;

pub(crate) use dispatch::run;

// `reporting_tests` is a child module that starts with `use super::*;`, exactly as it
// did when it lived inside the old monolithic `main.rs`. These re-exports restore the
// names that glob used to reach from the crate root.
#[cfg(test)]
pub(crate) use args::{Cli, CliCommand};
#[cfg(test)]
pub(crate) use az_loop::arena::{
    ArenaGateDecision, arena_gate_decision, arena_gate_position_counts, build_arena_start_positions,
    historical_anchor_index,
};
#[cfg(test)]
pub(crate) use az_loop::config::AzLoopProgressState;
#[cfg(test)]
pub(crate) use az_loop::selfplay::{SharedSelfplayModel, publish_selfplay_model};
#[cfg(test)]
pub(crate) use clap::Parser;
#[cfg(test)]
pub(crate) use crate::cli::az_loop_config::AzLoopFileConfig;
#[cfg(test)]
pub(crate) use chineseai::az::{AzArenaReport, AzNnue, AzSearchLimits, policy_target_entropy};
#[cfg(test)]
pub(crate) use chineseai::xiangqi::Position;
#[cfg(test)]
pub(crate) use reporting::{
    LabelEvalStats, PikafishLabelRow, evaluate_pikafish_labels, load_pikafish_label_rows,
};
#[cfg(test)]
pub(crate) use rusqlite::Connection;
#[cfg(test)]
pub(crate) use std::sync::{Arc, RwLock};
