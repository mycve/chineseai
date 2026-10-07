use crate::cli::args::AzCalibratePolicyArgs;
use chineseai::az::{AzNnue, calibrate_policy, px0_data};
use std::{fs::OpenOptions, io, path::Path};

pub(crate) fn run(cmd: AzCalibratePolicyArgs) {
    if let Err(error) = calibrate(cmd) {
        eprintln!("策略校准失败：{error}");
        std::process::exit(1);
    }
}

fn calibrate(cmd: AzCalibratePolicyArgs) -> io::Result<()> {
    if cmd.steps == 0 || cmd.games == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "steps 和 games 必须大于零",
        ));
    }
    let output = Path::new(&cmd.output);
    if output.exists() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "输出文件已存在",
        ));
    }
    let mut model = AzNnue::load(&cmd.model)?;
    let samples = px0_data::load_training(Path::new(&cmd.train), cmd.games)?;
    let report = calibrate_policy(&mut model, &samples, cmd.steps, cmd.seed)?;
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    model.save(output)?;
    println!("arch     : hidden={}", model.arch.hidden_size);
    println!("samples  : {}", report.samples);
    println!(
        "train CE : {:.6} -> {:.6}",
        report.before_ce, report.after_ce
    );
    println!("fold err : {:.8}", report.max_fold_error);
    println!("output   : {}", output.display());
    Ok(())
}
