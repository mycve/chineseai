//! slow-tests 共用同一个 CUDA 设备：每个测试各建一次 context 会让 candle
//! 重复 JIT 编译 kernel（约 10s/测试），共享后只需编译一次。
//!
//! 该模块只在 `cfg(test)` + `slow-tests` 下编译，生产构建与默认测试构建都不受影响。
use std::sync::OnceLock;

/// 进程内共享的 0 号 CUDA 设备。
///
/// `OnceLock::get_or_init` 是线程安全的：多个测试并发首次调用时只有一个真正初始化，
/// 其余等待同一个结果。首次初始化失败会缓存 `None`（同一进程内后续调用直接跳过 CUDA 对照）。
pub(crate) fn shared_cuda_device() -> Option<&'static candle_core::Device> {
    static DEVICE: OnceLock<Option<candle_core::Device>> = OnceLock::new();
    DEVICE
        .get_or_init(|| candle_core::Device::new_cuda(0).ok())
        .as_ref()
}
