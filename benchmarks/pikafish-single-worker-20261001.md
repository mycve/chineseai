# 单 worker profiling 与 Pikafish 源码对照

尚未追平 Pikafish。固定八个教师局面、相同节点预算下，本项目从约 22.4 万提高到 39～41 万 NPS；Pikafish 为约 184～250 万 NPS。完整数据、模型与局面 SHA256、profiling 原始输出见 [JSON](pikafish-single-worker-20261001.json)。

## 基准条件和结果

本机 Ryzen 9 9950X、Windows，Rust `fast` 构建及 `target-cpu=native`。改动前版本为 `33f567d`，前后使用同一浮点权重、同一组 FEN、同一深度上限 64。搜索始终为单 worker，未开启 `profile`，每次搜索重置置换表。8192 节点重复八个局面五次，262144 节点重复三次；表中 NPS 为实际节点总量除以搜索总时间。

| 每步预算 | 本项目前 | 本项目后 | 加速 | Pikafish Threads=1 |
| --- | ---: | ---: | ---: | ---: |
| 8192 | 224333 | 409177 | 1.82 倍 | 1835499 |
| 262144 | 223824 | 389298 | 1.74 倍 | 2504532 |

Pikafish 使用仓库现有 `tools/pikafish.exe`、`tools/pikafish.nnue`，Hash=16，每个局面前执行 `ucinewgame`。其时间包含 Python/UCI 通信，本项目计时为库内搜索。两个引擎权重不同，这不是棋力比赛；浮点初始网络也不能代表服务器当前训练候选。Windows 测量不能代替 Linux 同机复测。

8192 节点的 40 次搜索最佳着全部一致，最大根分数差约 `2.24e-7`。262144 节点的 24 次搜索有 3 次最佳着不同，最大根分数差约 `5.71e-5`：累加器与静态评估缓存改变浮点加法路径，固定预算下可能进一步改变搜索路径。没有据此宣称棋力提升或完全等价。

## 源码对照

参考源码固定在官方仓库 `687170f0d8dbc43554b3f959ce07b5c14bec4f35`，与现有对照二进制不是同一个版本，不混称为同一构建。

| 路径 | Pikafish 实现 | 本次改动及剩余差距 |
| --- | --- | --- |
| `nnue/features/full_threats.cpp` | 对攻击位图与 occupied 求交，遍历实际目标；平时读取 DirtyThreats 改变量 | 将棋子两两判断改为预计算伪攻击位图筛选；双方复用桶/镜像。仍未实现走子时的 DirtyThreats 增量记录 |
| `position.cpp`、`half_ka_v2_hm.cpp` | 缓存将帅位置、材料桶、镜像条件 | 复用 Position 已维护的将帅位置与占用位图；只在必要时计算中线镜像，避免扫描空格。材料/中线编码仍部分重算 |
| `nnue_accumulator.cpp` | 按搜索层保存累加器，沿 dirty 栈惰性更新；寄存器分块处理 i8/i16 数据 | 增加父层累加器恢复，校验父局面 hash 和模型身份；仍使用浮点权重和完整特征差集，未移植整数分块更新 |
| `position.cpp::see_ge` | 以阈值和攻击位图进行交换判断，尽早退出 | 本项目保留原有精确递归交换收益，只生成目标格的合法吃子，不再生成整盘合法着。仍不是原版阈值 SEE |
| `search.cpp` | 复用 TT 静态评估 | 增加线程私有、完整 hash 校验的静态评估缓存；换模型自动失效。规则终局仍在搜索中独立判断，未放宽 TT 完整规则历史要求 |
| `movepick.cpp` | 分阶段生成、排序走法，按需验证合法性 | 本项目仍大量一次性生成合法着与排序，这部分未移植 |
| `nnue/layers/affine_transform_sparse_input.h` | 量化整数 SIMD 和稀疏输入路径 | 本项目仍是 1024 宽浮点变换和浮点头，未完成量化推理或训练校准 |

## profiling 结果

使用 `--features profile` 的线程局部分段计时。存在插桩开销，不能拿插桩 NPS 与上表直接比较；计时包含子调用，不能把各行相加当占比。

最终固定教师局面套件搜索 327680 节点，主要累计时间：

| 范围 | 调用数 | 累计毫秒 |
| --- | ---: | ---: |
| CPU evaluate，含特征与网络 | 259715 | 666.44 |
| CPU finish，含下面的累加器和全连接 | 187224 | 488.46 |
| 浮点累加器更新 | 748896 | 244.12 |
| 合法着及规则检查 | 181905 | 194.07 |
| 全连接 matvec | 374448 | 127.61 |
| 走子后的规则历史构造 | 335380 | 121.33 |
| 威胁特征生成 | 187223 | 62.73 |
| SEE | 63950 | 29.03 |

静态评估缓存命中 72500 次，父层累加器恢复 40590 次。剩余差距不能靠增加节点预算解决：重点仍是 DirtyPiece/DirtyThreats 真正增量更新、量化推理、按需走法生成。量化会改变数值，必须单独验证误差、训练方式和对局棋力，不能把浮点权重强行截成整数。

## Linux 复测

在空闲服务器测量，使用同一份固定权重。`--workers` 仅用于独立的预提取特征评估部分，搜索基准始终串行；这里设置评估次数为 1。

```bash
cargo build --profile fast --example pikafish_perf
./target/fast/examples/pikafish_perf \
  --model runs/pikafish-evolve-128c-temperature/best.safetensors \
  --positions benchmarks/pikafish-search-positions.fen \
  --workers 1 --iterations 1 --nodes 262144 --search-repeats 3
```

内部热点计时：

```bash
cargo build --profile fast --features profile --example pikafish_perf
./target/fast/examples/pikafish_perf \
  --model runs/pikafish-evolve-128c-temperature/best.safetensors \
  --positions benchmarks/pikafish-search-positions.fen \
  --workers 1 --iterations 1 --nodes 8192 --search-repeats 5
```

Pikafish 已有 Linux 可执行文件时，用同一组局面运行 `benchmarks/pikafish_uci_bench.py`，传入可执行文件和它匹配的 NNUE 文件；脚本会设置 Threads=1、Hash=16，预热后逐局重置并统计实际节点与墙钟时间。

```bash
uv run python benchmarks/pikafish_uci_bench.py /实际路径/pikafish \
  --model /实际路径/pikafish.nnue --nodes 262144 --repeats 3
```

需要采样火焰图时，给 `fast` 构建保留符号，再执行 `perf record --call-graph dwarf`；本报告提供的是分段计时，不冒称已完成服务器的 perf 采样。

验证：199 项测试通过、2 项既有忽略项。包括 CUDA 前向/梯度、浮点推理一致性、缓存换模型失效、父层/兄弟分支恢复、100 步合法走子中定点吃子与完整生成器的一致性。
