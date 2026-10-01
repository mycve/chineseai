use clap::{CommandFactory, Parser};

use crate::cli::args::{Cli, CliCommand};
use crate::cli::{az_bench, az_init, az_loop, az_policy_scale, az_search, vs_pikafish};

pub(crate) fn run() {
    let cli = Cli::parse();
    match cli.command {
        None => {
            let _ = Cli::command().print_help();
            std::process::exit(0);
        }
        Some(CliCommand::AzInit(cmd)) => az_init::run(cmd),
        Some(CliCommand::AzPolicyScale(cmd)) => az_policy_scale::run(cmd),
        Some(CliCommand::AzSearch(cmd)) => az_search::run(cmd),
        Some(CliCommand::AzBench(cmd)) => az_bench::run(cmd),
        // `az_loop::run` returns false exactly where the old match arm did `return;`,
        // which skipped the profile report below. Keep both paths identical.
        Some(CliCommand::AzLoop(cmd)) => {
            if !az_loop::run(cmd) {
                return;
            }
        }
        Some(CliCommand::VsPikafish(cmd)) => vs_pikafish::run(cmd),
    }
    chineseai::profile::print_report();
}
