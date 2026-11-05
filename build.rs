//! 本仓库不支持 `*-musl` 目标（与 glibc + jemalloc 的部署与性能策略一致）。

fn main() {
    if std::env::var_os("CARGO_FEATURE_ENDGAME").is_some() {
        println!("cargo:rerun-if-changed=native/chessdb");
        cc::Build::new()
            .cpp(true)
            .std("c++17")
            .include("native/chessdb")
            .file("native/chessdb/index.cpp")
            .file("native/chessdb/xiangqi.cpp")
            .warnings(false)
            .flag_if_supported("/utf-8")
            .compile("chessdb_index");
    }
    let target = std::env::var("TARGET").expect("Cargo must set TARGET for build.rs");
    if target.contains("musl") {
        panic!(
            "不支持 musl 目标 `{target}`。\n\
             请改用 `x86_64-unknown-linux-gnu` / `aarch64-unknown-linux-gnu` 等 glibc 三元组，\n\
             或在 Linux 主机上直接 `cargo build --release`。"
        );
    }
}
