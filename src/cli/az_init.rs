use crate::cli::args::*;
use chineseai::az::AzNnue;

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
