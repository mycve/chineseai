# ChineseAI 自对弈训练问题复盘

本文记录这轮从 NNUE 式模型到 Tiny CNN/Line-GNN 过程中遇到的核心问题、判断方式和处理思路。重点不是“补救某个指标”，而是避免以后再次把结构问题误判成搜索、TD 或参数问题。

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

当前方向改成轻量 Line-GNN：

- local 3x3 卷积负责近邻关系。
- row context 负责横线关系。
- column context 负责纵线关系。
- 每层 row/column 有可学习 gate。
- 保留 CNN 格式，CPU/GPU 实现都比较直接。

这是借鉴 GNN 项目的关键归纳偏置，但避免完整图注意力的高推理成本。

### 5.3 当前 value head

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

### 5.4 当前 policy head

当前 policy 是 factorized scorer：

- from-square score。
- to-square score。
- move geometry condition。
- move bias。
- legal move mask。

这比 dense move tower 更轻，也更像 GNN 的“节点 embedding + action scorer”。

## 6. 关键诊断测试

最重要的小测试：

`value_head_can_overfit_tiny_fixed_dataset`

这个测试只给 4 个固定样本，不看搜索，只验证 value 头能不能在极小数据上过拟合。如果它失败，说明不是自对弈策略、MCTS 或 arena 的锅，而是 value 读出/梯度路径本身有问题。

曾经失败现象：

- `before≈0.78`
- `after≈0.78125`
- 等价于模型输出接近 0，塌到均值。

修复后：

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
- 当前 line-GNN 容量是否不足。
- search backup/value scale 是否仍不匹配。
- 自对弈分布是否过早塌到低质量循环。

## 10. 当前模型一句话总结

当前模型是一个 CPU 友好的 Tiny Line-GNN AlphaZero 网络：用 3x3 局部卷积加行列图聚合表达象棋长线关系，用稳定 value readout 防止均值坍缩，用 from/to factorized policy scorer 保持推理轻量。
