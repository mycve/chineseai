# 旧 NNUE：攻击关系压缩与子力分桶实验

日期：2026-10-19。分支：`codex/continuous-selfplay-publish`。

后续已将 scratch 缓存替换为 MCTS 节点状态，见 [节点增量更新实验](NODE_INCREMENTAL_RESULTS.md)。本文保留第一轮结构与训练对照结果。

## 实现

- 配对表删除从未被激活的车炮行，57,702 → 10,646 行；车炮继续使用原有射线和炮架特征。活跃特征的信息、个数和归一化规则不变。
- 关系索引从 `u32` 缩为 `u16`。威胁 embedding 总行数由 107,750 降到 60,694。
- 双视角增量缓存按来源棋子保存特征，只更新受变动格影响的来源；正确处理分支跳转、吃子、马腿、象眼、车炮射线和炮架。超过 8 个变动格或累计 128 次计算时刷新，限制浮点加减误差。
- value 隐藏层保持共享，WDL 输出层分成 32 个独立的 3×96 权重桶，按双方车数和马炮总数选择。CPU 推理、批数据、Candle 前向和反向、模型存储同步修改。子力计数并入王棋子特征遍历，减少额外扫描。
- 模型格式为 v40。v39 模型需要显式迁移，不静默加载。迁移保留共同权重，删除未激活行，并把旧 value 输出复制到全部 32 个桶；迁移本身不代表重新训练。

思路参考 Pikafish 的 [FullThreats](https://github.com/official-pikafish/Pikafish/blob/master/src/nnue/features/full_threats.cpp) 和 [K6R2NC2](https://github.com/official-pikafish/Pikafish/blob/master/src/nnue/features/k6r2nc2.cpp)。这里保留本项目已有的射线和炮架编码，没有直接替换成 Pikafish 的整个网络。

## 训练对照

没有找到现成的旧架构训练检查点，因此使用本地 `data/data.bin` 做小规模消融。变更前保存的基线可执行文件包含上一轮 64 维点积优化。

- 相同的前 256 局数据划分：25,846 条训练样本，330 条留出样本。
- 共同起点：96 维旧模型，初始化种子 20261019，训练 4 轮。
- 两个分支从同一份权重出发：旧架构继续训练 8 轮；新架构先迁移，再训练 8 轮。
- 两边使用 AdamW、学习率 0.0004、batch 256、shuffle 种子 17，预算相同。没有按验证结果挑选 checkpoint。

| 留出指标 | 旧架构 | 分桶 value |
|---|---:|---:|
| Value 交叉熵 | 0.659151 | 0.639532 |
| Q RMSE | 0.223417 | 0.209038 |
| Policy KL | 1.112408 | 1.111767 |
| Policy top-1 | 20.3030% | 19.3939% |

value 指标改善，policy top-1 下降。两边最终的 value 交叉熵均高于共同起点的 0.618107，存在过拟合或多任务权衡；这轮小样本结果不能外推到充分训练的生产模型。

## NPS

硬件：Ryzen 9 9950X，RTX 5070 Ti；NPS 只用单线程 CPU。使用 `fast` profile 和 `target-cpu=native`，固定同一逻辑核心。开局加 4 个固定书局面，每局面 800 节点、200 次搜索，交替测量三轮，先取各局面中位数，再计算相对速度的几何均值。

测速使用迁移前后同一份共同起点权重，避免训练引起搜索树变化。原始记录：`target/structured-value/nps-final.json`。

| 路径 | 相对旧版 NPS |
|---|---:|
| 旧版 | 100.00% |
| 压缩表 + 32 桶 value，完整遍历 | 99.32% |
| 压缩表 + 32 桶 value，增量缓存 | 86.37% |

**增量缓存没有提速，默认关闭。** 可用 `az-bench --threat-incremental` 单独复测。当前缓存维护开销超过节省的 embedding 求和成本；下一步若继续研究，应改为沿 MCTS 节点保存增量状态，并重新验证内存和缓存命中代价。

96 维检查点：44,399,016 → 32,388,392 字节，减少约 27.1%。速度基本持平，没有实现翻倍。

## 对局结果

固定官方 `book.pgn.gz` 的 128 个开局，每个开局交换红黑，共 256 局。每步 800 节点、最多 400 半回合、60 回合规则 120 半回合；不使用外部引擎估值裁决，达到长度上限的非终局记和。使用项目现有规则处理终局、重复和长将长捉。

| 候选胜 | 和 | 候选负 | 得分率 | 按开局配对估算的 95% 区间 |
|---:|---:|---:|---:|---:|
| 123 | 21 | 112 | 52.15% | 47.05%–57.25% |

区间包含 50%，**未证明棋力提升**。这是相同训练预算的小规模模型对照，尚未进行多随机种子、充分训练或固定时间比赛。不能把点估计当作稳定 Elo 提升。

对照模型由继续训练的旧架构模型显式迁移，所有 value 桶复制相同权重，因此仍表达旧版的共享输出头。两边比赛均关闭实验性增量缓存。完整开局保存在 `target/structured-value/candidate-trained.safetensors.arena.fens`，逐对进度为 `arena.log`、`arena-extra.log`，汇总为同目录的 `arena-summary.json`。

复现这一组对局：

```powershell
target/nnue-opt/fast/examples/structured_value_arena.exe target/structured-value/candidate-trained.safetensors target/structured-value/baseline-trained-v40.safetensors book.pgn.gz 128 800
```

## 正确性与复现

- 库与命令行测试：261 项通过，4 项默认忽略。
- 新增跨走子、吃子、重复访问及随机分支跳转的增量/全量对照，以及 value 分桶选择检查。
- 反向检查：只有选中的 value 桶收到梯度。
- 已训练权重的 CUDA/CPU 前向对照：128 个局面、3,748 个走法；最大 WDL 误差 0.00000048，最大策略 logit 误差 0.00000072。
- 共同起点迁移后的 330 条留出指标与旧版一致到报告的六位小数，确认压缩和复制桶没有改变模型的初始行为。
- 全目标测试仍受旧分支已有的 `examples/tactical_compare.rs` 字段过期错误影响；本次新增的两个 Rust 示例分别构建验证。

工具：

- `examples/structured_value_migrate.py`：v39 → v40，输出文件必须尚不存在；使用本地 uv 环境的 NumPy 和 safetensors。
- `examples/structured_value_experiment.rs`：固定数据和训练预算的训练/留出评估。
- `examples/structured_value_arena.rs`：固定节点、固定开局、成对换先；支持继续测后续开局。
- `examples/structured_value_bench.py`：固定核心，轮换测速顺序，保存原始 NPS。

```powershell
cargo build --profile fast --offline --bin chineseai --bin chineseai-uci --example structured_value_experiment --example structured_value_arena --target-dir target/nnue-opt
uv run --offline python examples/structured_value_migrate.py INPUT_V39.safetensors OUTPUT_V40.safetensors
```

当前小规模候选模型：`target/structured-value/candidate-trained.safetensors`。
已迁移的对照模型：`target/structured-value/baseline-trained-v40.safetensors`。
未替换正式模型。
