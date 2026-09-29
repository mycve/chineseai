//! 模型、回放和自博弈进化文件的格式版本。

/// 价值单头 safetensors 模型格式。
pub const MODEL_FORMAT_VERSION: f32 = 38.0;
/// 价值样本回放格式。
pub const REPLAY_FILE_VERSION: u32 = 42;
/// AB 进化 TOML 配置格式。
pub const AB_EVOLVE_CONFIG_FORMAT_VERSION: u32 = 32;
/// AB 进化进度格式。
pub const AB_EVOLVE_PROGRESS_VERSION: u32 = 9;
