# ChineseAI 自对弈训练问题复盘

本文记录这轮从 NNUE 式模型到 Tiny CNN、Line-GNN、Tiny Mobile-CNN 过程中遇到的核心问题、判断方式和处理思路。重点不是“补救某个指标”，而是避免以后再次把结构问题误判成搜索、TD 或参数问题。

## 1. 初始症状

- 自对弈 loss 和 arena 看起来偶尔有提升，但对 Pikafish 测试不稳定。
- 同一模型有时低深度赢得少、高深度反而赢得多，或者继续训练两轮后突然赢不了。
- `policy_ce` 能下降，但 `value_ce/value_mse` 长期不稳，`v_mu` 经常接近 0。
- `td_lambda=1.0` 和 `0.75` 都会卡，现象类似。
- GNN 项目在很低模拟数下能稳定进步，因此搜索模拟数不是第一嫌疑。

核心判断：主要问题不应先归咎于 MCTS 模拟数，而应优先怀疑模型结构、value 读出、目标尺度、训练分布和表示泛化。

## 2. 蒸馏测试得到的信号

用 GNN 产出的蒸馏数据测试后发现：

- 小模型可以学到一些策略，但容易卡在较高 `policy_ce`。
- 增大 `hidden` 明显提升策略拟合，增加 `trunk-depth` 影响不明显。
- 双塔结构让 value 和 policy 的拉扯变小，value 指标明显改善。
- 100w 数据一轮太慢，不适合作为每次结构迭代的主测试。
- 蒸馏只能证明离线容量和读出能力，不等价于在线自对弈一定稳定。

结论：蒸馏可以用来排查“结构是否完全学不会”，但自对弈稳定性还要看 value 在自身分布下能否产生正反馈。

## 3. 曾经踩过的结构坑

- 早期共享 trunk 太浅，policy/value 容易互相拉扯。
- 旧 NNUE/Sparse V4 类特征能帮离线拟合，但会模糊“模型是否真的从棋盘结构学习”的判断。
- value 读出太依赖全局池化或硬 ReLU MLP 时，容易塌到均值 0。
- value 输出用 WDL softmax CE 时，标量 value 梯度路径不够直接，早期容易不稳。
- 只看 arena 对当前 best 的相对胜率，可能掩盖绝对棋力和泛化不稳。

处理方向：移除旧式 NNUE/手工稀疏捷径，把模型改成更标准、更可解释的棋盘网络。

## 4. 数据和目标处理调整

- value 训练从 WDL CE 改为标量 MSE，更直接对应搜索使用的 value。
- 保留三 outcome logits，但推理 value 使用 `tanh((win_logit - loss_logit) * scale)`。
- 加入 `VALUE_LOGIT_SCALE = 0.25`，避免 value logit 差一开始过快进入 tanh 饱和区。
- 对无 policy target 的 batch 跳过 policy CE，避免全 mask policy 分支干扰 value-only 测试。

这一步的目的：让 value 梯度路径更短、更稳定，优先解决“value 学不动/学成均值”的问题。

## 5. 模型结构演进

### 5.1 Tiny CNN

曾尝试往标准 AlphaZero CNN 靠近：

- 棋盘 one-hot planes。
- 3x3 stem。
- 小 residual CNN blocks。
- policy/value heads 分开。

优点是结构更干净；问题是纯 tiny CNN 对象棋的长线关系不够天然，推理变慢后也不一定稳定提升。

### 5.2 Tiny Line-GNN

曾经尝试轻量 Line-GNN：

- local 3x3 卷积负责近邻关系。
- row context 负责横线关系。
- column context 负责纵线关系。
- 每层 row/column 有可学习 gate。
- 保留 CNN 格式，CPU/GPU 实现都比较直接。

这是借鉴 GNN 项目的关键归纳偏置，但实测这个 row/column mean 近似不够像 ZeroForge 的 attention GNN，速度和效果都没有占到便宜。结论：这条分支先判负，不继续加复杂度。

### 5.3 Tiny Mobile-CNN

当前方向改成 Tiny Mobile-CNN：

- sparse 3x3 stem。
- depthwise 3x3 做便宜局部混合。
- pointwise 1x1 做通道混合。
- residual 连接保留局部战术特征。

目的：保留 CNN 的稳定性，同时避开 dense 3x3 Tiny CNN 的 CPU 推理成本。

### 5.4 当前 value head

当前 value 读出是：

- 1x1 linear value tail。
- attention pooling。
- mean pooling。
- max pooling。
- std pooling。
- 小 spatial flatten tail。
- leaky MLP。
- direct residual linear readout。

关键修正：value head 不能只有硬 ReLU MLP。小样本测试中它会很容易只学到全局均值，导致看起来像 TD 或搜索问题。

### 5.5 当前 policy head

当前 policy 是 factorized scorer：

- from-square score。
- to-square score。
- move geometry condition。
- move bias。
- legal move mask。

这比 dense move tower 更轻，也更像 GNN 的“节点 embedding + action scorer”。

## 6. 历史诊断测试

曾经最重要的小测试：

`value_head_can_overfit_tiny_fixed_dataset`

这个测试只给 4 个固定样本，不看搜索，只验证 value 头能不能在极小数据上过拟合。如果它失败，说明不是自对弈策略、MCTS 或 arena 的锅，而是 value 读出/梯度路径本身有问题。该测试已从代码中删除，保留在这里作为历史排障记录。

曾经失败现象：

- `before≈0.78`
- `after≈0.78125`
- 等价于模型输出接近 0，塌到均值。

修复后曾经观察到：

- 该测试通过。
- 说明当前 value 至少具备基本的监督可塑性。

## 7. 删除和简化过的功能

为了让项目更清晰，已移除或不再保留：

- `az-distill`
- `perft`
- `az-search`
- `az-bench`
- `az-train-bench`
- 终端交互测试逻辑
- Web 棋盘 UI
- 历史 NNUE/AzNnue 风格遗留结构

方向是让项目更像一个干净的 AlphaZero 自对弈训练项目，而不是多个实验工具混在一起。

## 8. 当前判断

当前最大嫌疑仍是 value 稳定性，而不是单纯 policy 学不会。

理由：

- policy 离线拟合能力随 hidden 增大明显改善。
- 自对弈中 policy_ce 能下降，但棋力反馈不稳定。
- value 长期接近均值、震荡或饱和，会直接破坏搜索 target 的正反馈。
- GNN 项目稳定，说明带象棋长线归纳偏置的结构更适合这个任务。

当前处理不是“补丁”，而是把模型改成更天然适合象棋的轻量结构：局部 + 行列关系 + 稳定 value 读出。

## 9. 后续观察重点

下一轮自对弈重点看：

- `value_loss/value_mse` 是否不再长期卡住。
- `v_mu` 是否仍长期贴近 0。
- `policy_ce` 是否下降后能带来 arena 持续提升。
- arena promotion 后是否很快回退。
- 平均步数和和棋率是否异常升高。
- 低模拟 Gumbel 是否能产生更清晰的 improved policy target。

如果仍然卡住，优先排查：

- 终局/重复规则 target 是否有系统偏差。
- value target 是否过噪。
- 当前 Mobile-CNN 容量是否不足。
- search backup/value scale 是否仍不匹配。
- 自对弈分布是否过早塌到低质量循环。

## 10. 当前模型一句话总结

当前模型是一个 CPU 友好的 Tiny Mobile-CNN AlphaZero 网络：用 sparse stem、depthwise 3x3 和 pointwise 1x1 低成本表达棋盘局部关系，用稳定 value readout 防止均值坍缩，用 from/to factorized policy scorer 保持推理轻量。

## 11. 本地结构探针结论

新增 `az-arch-probe` 命令，用低成本根节点展开测试候选结构。它不替代自对弈，但可以在结构迭代早期快速过滤明显不合适的宽深组合：

- `us/root`：一次根节点展开的 CPU 平均耗时，包含棋盘 forward、value、合法走子 policy 打分和 softmax。
- `params`：模型参数量，用来估计容量和训练/保存成本。
- `v_abs`：随机初始化下 value 绝对值均值，过大说明初始 value 容易饱和；当前候选都很低。
- `ent`：合法走子 policy 熵占均匀分布熵的比例，接近 1 说明初始策略不过尖。

探针现在还额外输出：

- `win`：局部卷积路径的有效窗口边长。`blocks=3` 时为 `9x9`，已经覆盖象棋棋盘宽度，只差一行纵向距离由 line/global 路径补。
- `line`：行/列长线池化通道数，等于 `2 * channels`。
- `mac%`：相对一个粗略 `64c x 6 dense 3x3 CNN` 的估算 MAC 比例，只作结构成本量级参考。
- `us_sd`：多随机初始化下 CPU 根展开耗时标准差。

在 `eval_fens.txt` 前 16 个局面、`--reps 128 --seed-trials 3`、`--profile fast` 下的本地结果：

| 候选 | channels | blocks | hidden | value channels | value hidden | params | win | line | mac% | us/root | us_sd | v_abs | ent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny-fast | 16 | 2 | 128 | 4 | 128 | 95,319 | 7 | 32 | 6.7 | 423.69 | 37.86 | 0.005 | 0.999 |
| compact-3b | 20 | 3 | 160 | 5 | 160 | 139,578 | 9 | 40 | 8.7 | 566.19 | 11.41 | 0.009 | 0.999 |
| balanced-2b | 24 | 2 | 192 | 6 | 192 | 190,733 | 7 | 48 | 10.4 | 610.05 | 14.29 | 0.008 | 0.999 |
| balanced-3b | 24 | 3 | 192 | 6 | 192 | 191,573 | 9 | 48 | 10.6 | 675.35 | 6.30 | 0.006 | 0.999 |
| wide-2b | 32 | 2 | 256 | 8 | 224 | 290,403 | 7 | 64 | 14.1 | 858.21 | 15.08 | 0.006 | 0.999 |
| current | 32 | 3 | 256 | 8 | 256 | 320,067 | 9 | 64 | 14.7 | 1049.15 | 64.43 | 0.008 | 0.999 |
| wide-heavy | 40 | 2 | 288 | 8 | 256 | 353,587 | 7 | 80 | 17.9 | 1112.78 | 4.66 | 0.012 | 0.999 |

当前推荐下一轮主线结构：

- 主结构：`model_channels=24, model_blocks=3, hidden_size=192, value_head_channels=6, value_hidden_size=192`。
- 对照结构：`model_channels=32, model_blocks=2, hidden_size=256, value_head_channels=8, value_hidden_size=224`。

理由：`24c/3b/h192/vhc6/vh192` 比当前默认结构快约 36%，参数少约 40%，同时具备 `9x9` 局部窗口、48 个行列长线池化通道、稳定 value readout 和很低的随机初始化 value 幅度。它不是最小模型，而是速度、偏置和容量的平衡点。`16c/2b` 虽然最快，但窗口只有 `7x7`、容量和表达余量偏紧，更适合做 smoke test，不适合作为主训练结构。`32c/2b` 是合理容量上限对照，如果它在蒸馏或早期自对弈明显优于 24c/3b，再接受额外 CPU 成本。

关于“达到大 CNN 潜力”的判断：

- 这个结构不是单纯缩小版 CNN，而是 `局部卷积 + 行列全局池化 + 走法几何条件 + from/to/pair action scorer` 的棋盘专用结构。
- `blocks=3` 的局部窗口为 `9x9`，已经能在局部路径覆盖棋盘宽度；纵向完整长线由 row/column pool 和 move geometry 补。
- depthwise separable block 是 dense 3x3 CNN 的低秩分解。增加 `channels` 可以提升 pointwise 混合秩，增加 `blocks` 可以扩大局部组合深度，因此结构族可以平滑扩展到 `32c/3b`、`40c/2b` 这类更大容量，而 CPU 成本仍显著低于 dense CNN。
- 目前探针只能证明底层偏置、初始化稳定性和成本/容量边界，不能单独证明最终棋力。下一步必须用同一套候选跑短蒸馏或短自对弈，把 `value_mse/v_mu/policy_ce/arena` 作为第二层筛选。

## 12. 极小成本长期代理测试

为了避免把结构探针误当成棋力证明，新增 `az-loop --updates N`，可以让临时自对弈训练跑固定 update 数后自动停止。正式训练不传该参数时仍然连续运行。

本机临时测试目录：`target/strength-probe/`。

### 12.1 激进配置红灯

配置：`24c/3b/h192/vhc6/vh192`，`simulations=1`，`selfplay_batch_games=4`，`lr=0.0005`，带 replay 混入。

观察：

- 约 4 个 update 后模型继续训练到第 5 个 update 时出现 `NaN`。
- 权重扫描：初始模型 `nan=0`；训练后模型 `nan=161438/191573`。
- 结论：不能用这个激进小配置证明长期棋力。它暴露出当前训练设置在极低模拟、短局全和、偏高学习率或 replay 混入下有数值稳定性风险。

### 12.2 保守配置正信号

配置：`24c/3b/h192/vhc6/vh192`，`simulations=1`，`selfplay_batch_games=2`，`lr=0.00001`，`replay_samples=0`，`max_plies=30`。

关键输出：

```text
update 0001: loss=3.7159 value_mse=0.0000 v_mu=0.001/0.000 policy_ce=3.7159
update 0009: loss=3.6892 value_mse=0.0001 v_mu=0.006/0.000 policy_ce=3.6891
```

权重扫描：

```text
after update 1: nan=0/191573 inf=0
after update 9: nan=0/191573 inf=0
```

低模拟 arena（训练 9 update 后 vs 同 seed 初始模型，`8` 盘，`1` sim，`60` ply）：

```text
wins=1 losses=1 draws=6
```

解释：

- 这是“长期训练链条没有立刻坏掉”的正证据：多 update 后权重仍有限，`policy_ce` 有轻微下降，`value` 没有明显饱和。
- 这还不是最终棋力证明：`1` sim、短局、随机初始附近的 arena 大量和棋，信号很弱。
- 当前更可靠的结论是：结构本身具备继续训练潜力，但训练超参数必须保守启动；在没有更强 target 或更多模拟前，激进 LR/replay 会把模型打成 NaN。
