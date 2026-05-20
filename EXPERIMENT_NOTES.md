# ChineseAI 模型实验记录

## 当前保留基线

当前继续使用 `codex/pure-nnue-incremental-mcts` 上的轻量 NNUE 版本：

- 保留棋子位置、当前方 canonical 视角、将帅 bucket、history、row/col。
- 保留 tactical/strategic 特征，包括 attacked/protected/hanging 以及 material/halfmove bucket。
- 已回退更复杂的官方式 threat、king attack、分层 head 等特征。

这个版本虽然不是最快，但 policy/value 拟合 MCTS 分布的表现比全删静态评估特征更稳。

## 已删除实验分支

以下实验分支已删除本地和远端：

- `codex/test-minimal-spatial-features`
- `codex/test-policy-value-capacity`
- `codex/test-policy-piece-only`

## 实验结论

### 全删 tactical/strategic 特征

分支：`codex/test-minimal-spatial-features`

改动：

- 移除 tactical 特征：attacked/protected/hanging。
- 移除 strategic 特征：halfmove、total material、material balance、攻击/保护统计 bucket。
- 只保留空间偏置特征。

观察：

- 自对弈和训练吞吐提升明显。
- 但 `policy_ce`、`value_mse`、`loss` 变差。

结论：

- 这些静态特征不是完全无用。
- 当前阶段模型拟合 MCTS 分布能力还不够，删掉它们会降低单样本学习效率。
- 暂不采用。

### 同时增强 policy/value head

分支：`codex/test-policy-value-capacity`

改动：

- policy 加走子棋种条件。
- `POLICY_CONDITION_SIZE` 从 32 增到 64。
- value head 改为 `hidden -> 128 ReLU -> WDL logits`。

观察：

- 速度变慢。
- 拟合指标没有稳定明显优于基线，policy loss 至多接近，部分区间略差。

结论：

- 简单堆容量没有稳定收益。
- 更大 head 可能增加优化难度，不能自动转化为更强的 MCTS 分布拟合。
- 暂不采用。

### 只加走子棋种 policy 条件

分支：`codex/test-policy-piece-only`

改动：

- 只在 policy logits 中加入当前走子棋种条件。
- 不改 policy condition 维度。
- 不改 value head。

观察：

- `policy_ce`、`loss` 与基线几乎重合。
- 速度没有明显收益，训练耗时略有增加。

结论：

- 该信息大概率已可由 `from` 位置和 shared hidden 推断。
- 显式棋种条件属于重复信息，收益很小。
- 暂不采用。

## 暂缓方向

以下方向暂缓：

- 继续堆 policy/value head 容量。
- 加 full threat、复杂 attack map、Stockfish/Pikafish 式 king safety。
- 加每个 dense move 的大规模 from-to embedding。

原因：

- 当前几次结构实验都没有稳定带来更好的 policy/value 拟合。
- 额外复杂度会降低自对弈吞吐，且增加重新训练成本。

## 后续更值得优先验证

优先看训练目标和搜索数据质量，而不是继续堆模型结构：

- policy target 是否过尖或噪声过大。
- Gumbel/top-k 产生的 visit distribution 是否过稀疏。
- 小比例 policy target smoothing 是否能稳定训练。
- opening/midgame/endgame 是否需要不同 smoothing。
- 固定墙钟时间下的 arena/vs-pikafish，而不是只看同 update 的 loss。

注意：policy target smoothing 会改变训练目标分布，不是无代价操作。若要试，建议从很小的 `eps = 0.01` 或 `0.02` 开始，并用 arena 验证是否真的提升棋力。

## Gumbel value scale 经验

当前自对弈样本里的 policy target 使用的是 `candidate.policy`。在 Gumbel AlphaZero 模式下，它不是简单的 `visits / sum(visits)`，而是 Gumbel MuZero 风格的 improved policy：

```text
policy = softmax(policy_logit + completed_qvalue)
```

其中 completed Q 会经过缩放：

```text
scale = (maxvisit_init + max_visit) * gumbel_value_scale
```

因此 `gumbel_value_scale` 必须和模拟次数一起看。之前使用：

```text
sims = 1024 或 2048
gumbel_value_scale = 0.1
maxvisit_init = 50
```

在根节点 `max_visit` 达到几百时，`scale` 会变成几十。例如 `max_visit = 480` 时：

```text
scale = (50 + 480) * 0.1 = 53
```

这会让 `softmax(policy_logit + completed_qvalue)` 接近 one-hot。实际 visits 可能仍然分散，但训练 target 会非常尖，导致 policy head 学习困难、输出过度自信，也会掩盖模型结构实验的收益。

实测同一局面：

- `gumbel_value_scale = 0.1` 时，top policy 可接近 `0.997`，其它动作接近 0。
- `gumbel_value_scale = 0.005` 时，top policy 约 `0.216`，分布更接近 visits。
- `gumbel_value_scale = 0.01` 时，top policy 约 `0.17`，仍保持较软分布，同时保留 Q 改善信号。

后续训练建议：

- 在 `1024/2048` sims 下，优先试 `gumbel_value_scale = 0.01`。
- 如果 policy CE 仍不稳或 target 偏尖，再试 `0.005`。
- 暂时避免用 `0.1` 搭配高 sims。
- UCI 中可通过 `GumbelValueScale` 显式覆盖。当前默认仍保留 `0.1`，避免直接影响对弈/评估强度。

经验结论：

- 之前 policy 拟合差，不一定是模型结构容量不足。
- 更可能是 Gumbel improved policy 的 value scale 与模拟次数不匹配，导致 target 过尖。
- 先校准 `gumbel_value_scale`，再重新判断是否需要模型结构改动。
