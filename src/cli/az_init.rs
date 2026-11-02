use crate::cli::args::*;
use chineseai::az::AzNnue;

pub(crate) fn upgrade_mating_targets(cmd: AzUpgradeMatingTargetsArgs) {
    AzNnue::upgrade_mating_targets(&cmd.input, &cmd.output)
        .unwrap_or_else(|err| panic!("杀法目标扩容失败: {err}"));
    println!(
        "模型已转换至 v56、14 种具名杀法：{} -> {}。保留已训练权重，新增目标乘性门控零初始化；请使用新配置路径重建优化器，v46 回放可继续使用。",
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
