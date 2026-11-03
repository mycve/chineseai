use crate::cli::args::*;
use chineseai::az::AzNnue;

pub(crate) fn upgrade_mating_targets(cmd: AzUpgradeMatingTargetsArgs) {
    AzNnue::upgrade_mating_targets(&cmd.input, &cmd.output)
        .unwrap_or_else(|err| panic!("杀法目标迁移失败: {err}"));
    println!(
        "模型已迁移至 v60：{} -> {}。保留现有权重并冻结普通参数，只微调目标分支。目标仍是实际命中概率；条件距离只监督真实目标成功轨迹。可复用v50经验池，使用新配置路径和新目标优化器继续训练，不要重新az-init。",
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
