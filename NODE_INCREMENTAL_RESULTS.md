# MCTS 节点增量更新实验

2026-10-19，`codex/continuous-selfplay-publish`。

**本轮尚未达到单线程 NPS 翻倍。** 保持相同 v40 检查点和搜索预算，当前默认路径约提升 6%。没有重新训练、修改权重或改变搜索参数。

## 保留的实现

- 删除上一轮在不同局面间维护的 scratch 威胁缓存，改为由 MCTS 节点保存状态索引，实际数据放入可复用 arena。
- 王棋子 16 维池化、子力计数及可选的威胁 64 维累加器，从同视角祖先更新。根节点按需保存两个视角；普通子节点合并父节点走子与本步走子的变化。
- 将帅桶变化时刷新王棋子池化；每 32 半回合按全量顺序重算，限制浮点累积误差。
- 威胁状态保留完整特征索引，只重算受影响的来源。依赖掩码区分攻击目标、马腿、象眼和车炮射线；阻挡格只在占用状态变化时触发更新。没有再次扫描 90 格去猜测两次评估间发生了什么。
- 搜索 workspace 回收节点状态容量；跨步复用网络结果时，当前树的增量状态按需重建，不使用旧树的 arena 偏移。
- 主干与策略累加器的六行更新合并为一次隐藏向量读写，保留每个通道原有的 FMA 顺序。
- AVX-512 矩阵核并行计算四个输出，保留原 AVX2 四路累加、归约及尾部计算顺序。相关 SIMD 回归按浮点位比较。

**默认启用节点王棋子状态，威胁增量仍默认关闭。** 后者正确性通过，但整体收益低于默认组合。

## 单线程基准

CPU 为 Ryzen 9 9950X。使用 `fast` profile、`target-cpu=native`，固定同一逻辑核心。基线是本轮修改前冻结的可执行文件。模型始终为同一份 `target/structured-value/candidate-trained.safetensors`，96 维、v40。

正式复测：开局加四个固定书局面；每次 800 节点、重复 100 次搜索；两个版本交替运行四轮，按各局面中位数计算相对速度的几何均值。原始结果为 `target/node-incremental/nps-final.json`。

最终默认路径相对冻结版本提升 **6.33%**；开局中位数由 **195,986.5 → 208,560.5 NPS**。

另一次四路径消融得到：

| 路径 | 相对冻结基线 |
|---|---:|
| 冻结版本 | 100.00% |
| SIMD 更新与矩阵核，不复用节点价值状态 | 102.01% |
| 节点王棋子状态，默认路径 | 106.22% |
| 再启用节点威胁增量 | 102.48% |

该消融的完整记录为 `target/node-incremental/nps.json`。不同轮次的默认路径增益约为 5%–6%，不应把小数点后的差异当作稳定收益。

也测过两种替代布局：仅保存累加值、更新时提取前后局部关系；以及共享来源特征块。两者均未超过保留方案，已经撤回。原型记录分别为 `nps-delta.json` 和 `nps-source-refs.json`，位于同一结果目录。

威胁累加只占搜索耗时的一小部分。节点缓存即使消除了这部分成本，也不能单独令整体翻倍；策略头和其余搜索开销仍需进一步优化。

## 正确性

- 263 项库和命令行测试通过，4 项默认忽略。
- 分支、吃子、双视角、将帅桶变化及定期刷新，与全量特征累加对照；测试使用非零王棋子权重。
- 128 个固定局面、每局面 800 节点：默认节点路径和完整威胁增量路径，各自与全量路径比较，最佳走法及根候选访问数全部一致；最大搜索 Q 差约为 `6e-8`。
- 默认节点路径在每局面 8,000 节点时，128 个局面的最佳走法及根候选访问数也全部一致。
- 与冻结可执行文件独立对照：128 个局面、5,203 个根候选、每局面 800 节点，最佳走法和访问数没有差异。记录为 `target/node-incremental/frozen-regression.json`。

这些是当前检查点和测试集的回归证据；本轮没有通过重新训练或改变搜索预算换取速度。

## 复现

```powershell
cargo build --profile fast --offline --bin chineseai --bin chineseai-uci --example node_incremental_audit --target-dir target/nnue-opt

# 默认节点路径
target/nnue-opt/fast/chineseai.exe az-bench MODEL.safetensors 800 100

# 全量节点价值计算，对照用
target/nnue-opt/fast/chineseai.exe az-bench --no-node-incremental MODEL.safetensors 800 100

# 节点威胁增量实验
target/nnue-opt/fast/chineseai.exe az-bench --threat-incremental MODEL.safetensors 800 100

# 同模型、同局面、同节点预算核对搜索结果
target/nnue-opt/fast/examples/node_incremental_audit.exe MODEL.safetensors FENS.txt 800 king
target/nnue-opt/fast/examples/node_incremental_audit.exe MODEL.safetensors FENS.txt 800 threat
```

检查点格式和权重保持不变，仍为 v40。本轮不需要再迁移模型。
