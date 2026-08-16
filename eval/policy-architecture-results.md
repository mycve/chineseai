# Policy 结构单轮拟合结果

数据：当前 `best.safetensors` 自博弈采样 100,000 个去重局面，Pikafish depth 12 标注最佳着与 WDL。固定随机种子切分 90,000 训练 / 10,000 验证，每条训练样本严格使用一次，batch size 1024，学习率 0.0007，公共 trunk hidden=128、policy rank=32。

| 结构 | 参数量 | Policy CE | Top-1 | Top-3 |
|---|---:|---:|---:|---:|
| dot | 263,689 | 3.19114 | 14.68% | 30.99% |
| gated dot | 267,817 | 3.18662 | 14.79% | 31.00% |
| hyper2 | 351,081 | 3.11389 | **16.32%** | **32.38%** |
| interaction MLP | 269,962 | **3.08679** | 16.08% | 32.32% |
| residual interaction MLP | 269,962 | 3.17462 | 14.89% | 31.20% |
| product-only MLP | 265,866 | 3.10790 | 15.68% | 31.66% |

结论：`interaction MLP([q, move, q*move])` 以仅比 dot 多 2.38% 的参数取得最低 Policy CE，相对改善 3.27%，是当前优先回植结构。若目标更偏 top-1，双通道 `hyper2` 略优，但参数增加 33.1%。直接给 MLP 增加 dot 残差反而退化。

墙钟数值受每个进程首个 CUDA kernel 初始化影响，不适合作为亚秒级单 epoch 的结构排序；所有结构都执行相同的 88 个优化 step 和 90,000 个样本。

## CE 2.0 追踪实验

在严格 group 隔离验证下继续加入结构化走法特征：canonical from/to、动子、被吃子、是否将军，以及棋子落点后果增量。最优单遍历结果为：

- validation Policy CE：2.38097
- train Policy CE：2.29294
- top-1：31.89%
- top-3：53.99%
- 参数量：470,058

更大的 205 万参数深层网络、棋盘卷积、全局 attention、逐候选 cross-attention 均未超过该结果。重复 2/4 epoch 的验证 CE 分别为 2.329/2.310，显示单纯延长训练不能逼近 2.0。

利用 SQLite 中已有 depth-12 principal variation 扩展出 767,141 个同源局面，并按原始局面 group 隔离验证；单遍历最好 CE 为 2.36293，仅小幅改善。PV 单线分布不能替代独立局面或多走法分布标签。

结论：在当前 100k 独立局面、Pikafish 单一 bestmove one-hot 标签下，尚未达到 CE 2.0。下一步应优先采集 MultiPV/走法分数形成软 policy，或增加独立标注局面；继续扩大 raw-policy 网络的收益已经很低。最终结构仍需回植 Rust 后按固定自博弈墙钟比较 sims/s 和 samples/s。

## 严格控制变量复测

正式复测统一使用相同的 90k/10k group 切分、seed=20260816、hidden=128、batch=16、lr=0.001、value weight=0.25、单 epoch、训练顺序和验证样本。`current_like` 复现 move bias + context dot + consequence delta；实验组只改变 policy 交互及结构化走法输入。

| 结构 | rank | 参数量 | Policy CE | 校准 CE | Top-1 | Top-3 |
|---|---:|---:|---:|---:|---:|---:|
| current-like | 64 | 335,830 | 2.75086 | 2.72692 | 22.12% | 43.05% |
| structured consequence | 64 | 470,058 | 2.39246 | 2.39246 | 30.91% | 53.51% |
| structured consequence（参数匹配） | 32 | 321,194 | 2.42995 | 2.42995 | 30.34% | 52.36% |

固定 rank 下原始 CE 相对改善 13.03%，Top-1 增加 8.79 个百分点。参数匹配组比基线少 4.36% 参数，CE 仍相对改善 11.67%，Top-1 增加 8.22 个百分点。该结果控制了训练变量和参数量，但仍是 PyTorch 等价 policy-head 代理实验；必须回植 Rust 后才能给出真实单节点推理、自博弈 sims/s 和 samples/s 变化。
