use crate::cli::args::*;
use chineseai::az::AzNnue;

pub(crate) fn upgrade_mating_targets(cmd: AzUpgradeMatingTargetsArgs) {
    AzNnue::upgrade_mating_targets(&cmd.input, &cmd.output)
        .unwrap_or_else(|err| panic!("杀法目标迁移失败: {err}"));
    println!(
        "模型已转换至 v58、11 种具名杀法：{} -> {}。保留对应已训练权重，缺失目标门控补零；请使用新配置路径重建优化器和 v49 经验池。",
        cmd.input, cmd.output
    );
}

pub(crate) fn run(cmd: AzInitArgs) {
    let arch = cmd.arch();
    let output = cmd.output;
    let seed = cmd.seed;
    let model = AzNnue::random_with_arch(arch, seed);
    model.save(&output).unwrap_or_else(|err| {
        panic!("failed to write `{output}`: {err}");
    });
    println!(
        "aznnue   : initialized (safetensors, format v{})",
        chineseai::infra::version::MODEL_FORMAT_VERSION
    );
    println!("arch     : hidden={}", arch.hidden_size);
    println!("seed     : {seed}");
    println!("output   : {output}");
}
