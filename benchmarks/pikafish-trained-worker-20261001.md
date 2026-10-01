# 真实冠军权重：第二轮 Pikafish 源码对照与单 worker 测量

尚未追平 Pikafish。本轮使用服务器公开看板目录的真实 `best.safetensors`，单 worker 冷缓存由 426900 提升至 596406 NPS，约 1.40 倍；保留静态评估缓存为 742785 NPS。本轮重新测量官方引擎同机冷缓存结果为 2491803 NPS，仍有约 4.2 倍差距。NPS 改善不能证明棋力增长。

原始计时、每局节点/深度/分数和权重 SHA256 见 [JSON](pikafish-trained-worker-20261001.json)。服务器权重未纳入 Git。

## 测量与正确性

Windows / Ryzen 9 9950X / `fast` / `target-cpu=native`，未开启 profiling。八个固定局面，每局 262144 节点，重复三次，深度上限 64。基线为 `c6db0c0` 的可执行文件；`8602e32` 只修复测评，搜索代码相同。两边使用同一份训练权重。

| 条件 | 总节点 | 搜索秒数 | NPS |
| --- | ---: | ---: | ---: |
| 改动前，冷缓存 | 6291456 | 14.73755 | 426900 |
| 改动后，每次搜索清空共享缓存 | 6291456 | 10.54896 | 596406 |
| 改动后，保留共享缓存 | 6291456 | 8.47008 | 742785 |

清空共享缓存不计入搜索时间，对照 UCI 的 `ucinewgame` 在计时外执行。热缓存结果不与官方冷缓存结果混比；本项目仍每次搜索重建搜索 TT，热缓存仅保留静态网络评估。不同引擎使用各自权重，不是同网络算子基准，也不是棋力比赛；本机不能代替服务器同机测量。

对照 Candle 浮点前向的 902 个局面最大误差 `7.152557e-7`。24 次冷缓存和 24 次热缓存搜索的最佳着均与基线相同，最大根分数差 `1.341105e-7`。这些覆盖不能代替长期对局测评。

## 实际阅读的官方源码

源码固定为 [official-pikafish/Pikafish@687170f](https://github.com/official-pikafish/Pikafish/tree/687170f0d8dbc43554b3f959ce07b5c14bec4f35)，现有对照二进制不是该提交的重新构建。

| 源码位置 | 实现及本项目处理 |
| --- | --- |
| `search.cpp:1050–1242` | MovePicker 后的浅层剪枝在 `do_move` 前完成。本项目把晚步安静着剪枝提前，省去被剪枝走法的规则历史、追逐信息和网络状态构造，保留将军例外和合法着序号。 |
| `search.cpp:1667–1715` | 静态评估/stand-pat 截断在 qsearch MovePicker 前完成。本项目先评估再生成吃子；中国象棋无子可动判负，因此截断前仍验证至少存在一着规则合法走法。 |
| `movepick.cpp:292–366` | TT、好吃子、好安静着、坏吃子分阶段处理。本项目已改为候选排序后按需验证合法性；仍一次生成候选并排序，尚未达到原版分阶段 MovePicker。 |
| `nnue/nnue_accumulator.cpp:364–395` | `apply_combined` 将 PSQ 与 Threat 改变量应用到寄存器分块，最后写回。本项目将浮点更新改成 64 通道分块，维持每通道加法顺序；刷新计数按实际特征变更推进，并在恢复父层时恢复计数。尚未融合两个特征表。 |
| `position.cpp:581`、`nnue_accumulator.cpp:402` | 走子时维护 DirtyPiece/DirtyThreats，并沿累加器栈惰性计算。本项目仍生成完整特征、排序和求差集，这是未解决的结构性差距。 |
| `nnue/nnue_common.h:58–60`、`layers/affine_transform_sparse_input.h:203,436` | 变换器权重 i8、累加器 i16，网络输入 u8、输出 i32；按非零输入走整数 SIMD。本项目依然是 f32 权重/累加器/稠密头，不具备对应的量化训练和校准。 |

## 缓存实现

官方从 TT 复用静态评估。我们原有 8192 格缓存是搜索私有的，无法跨着、跨 worker 复用。本轮增加每份只读 CPU 权重共享的 1048576 格缓存，共 16 MiB，完整棋盘 hash、有效位和原始 f32 分数通过 `portable-atomic::AtomicU128` 原子读写。只有运行时确认 128 位原子无锁才启用；否则继续使用原有线程私有缓存，避免在不支持的 CPU 上回退全局互斥锁。

模型换代时新建缓存，不共享旧模型分数。规则终局判断继续先于静态评估；搜索 TT 的完整规则历史校验保持原实现。静态网络缓存只存固定权重下的棋盘评估，不缓存将军/追逐规则裁决或搜索上下界。

曾测试 qsearch 搜索分数缓存、历史前缀编号、显式 AVX512 分块和重排浮点矩阵乘法；实测变慢或无收益，均未保留。按官方方式把两张表合并到同一次分块更新，实测为 591046 NPS，未见收益，已撤回。直接压缩整数累加器的试验未达到完整数值/对局验证要求，也未进入生产路径。不能把 f32 权重直接转 i8 来宣称移植完成。

## 本轮验证与热点

`cargo test --profile fast --all-targets`：206 项通过，2 项既有忽略项，0 失败。覆盖共享缓存完整 key/原子配对及并发碰撞、换模型失效、父层/兄弟分支恢复、按需合法性与完整生成器对照、将军/困毙/重复规则，以及原有 CUDA 前向和梯度测试。

真实权重的 8192 节点套件，主要分段计时：evaluate 582.923 ms，finish 415.226 ms，其中累加器 196.280 ms、全连接 113.473 ms；规则历史构造 111.766 ms，威胁特征 57.602 ms。全部原始范围见 JSON。各范围嵌套，不能相加当总占比；该测量是本机分段计时，不是服务器 `perf` 采样。

## 服务器复测

先在空闲服务器运行冷缓存、前向验证，再独立测热缓存。路径应指向当前真实冠军权重，缺失模型会明确报错，基准不再自动生成随机初始网络。

```bash
cargo build --profile fast --example pikafish_perf
./target/fast/examples/pikafish_perf \
  --model /实际路径/best.safetensors \
  --positions benchmarks/pikafish-search-positions.fen \
  --workers 1 --iterations 1 --nodes 262144 --search-repeats 3 \
  --validate-values --validate-plies 128
# 另跑一次，追加 --warm-cache，仅用于观察跨搜索静态缓存收益。
uv run python benchmarks/pikafish_uci_bench.py /实际路径/pikafish \
  --model /实际路径/pikafish.nnue --nodes 262144 --repeats 3
```

内部计时会改变吞吐，应独立构建并运行，不与上述 NPS 混比：

```bash
cargo build --profile fast --features profile --example pikafish_perf
./target/fast/examples/pikafish_perf \
  --model /实际路径/best.safetensors \
  --positions benchmarks/pikafish-search-positions.fen \
  --workers 1 --iterations 1 --nodes 8192 --search-repeats 5
```

剩余工作是 DirtyPiece/DirtyThreats 增量记录、整数/稀疏推理与训练校准，以及真正分阶段 MovePicker；增加 worker 数不能消除这些单 worker 瓶颈。本轮未改训练目标、未量化现有冠军权重，也未更新服务器正在运行的程序。
