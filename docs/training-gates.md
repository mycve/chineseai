# 自博弈模型发布与晋级

开启 `arena_interval > 0` 时，learner 持续训练，产数 actor 与对外 best 分别评测。

- **best**：维持原晋级规则，对当前冠军的得分率置信下界超过晋级线，且历史对手检查未发现显著退步。
- **actor**：每次 arena 检查候选；对 best、上一冠军、历史锚点，以及与 best 不同的当前 actor，逐一检查非劣界限。每个对手至少 `actor_gate_min_games` 局，得分率置信下界必须达到 `0.5 - actor_noninferiority_margin`。best 评测判定 Reject 时，actor 也拒绝发布。
- 证据不足时输出 `Hold` 并保留原 actor；确认越过允许退步界限时输出 `Reject`；全部通过才输出 `Publish`。
- 对 best 和历史对手重复检查，防止只与上一代相比而逐代累积退步。通过 best 晋级也不会跳过当前 actor 检查。
- 重启从已晋级的 best 启动 actor，恢复的原始 learner 必须重新通过评测。学习器与回放池正常恢复。

默认 `actor_noninferiority_margin = 0.02`、`actor_gate_min_games = 400`，置信参数沿用 `arena_promotion_confidence_z = 1.96`。例如得分率 50%，但置信区间下界只有 46%，不会发布；下界达到 48% 才满足默认非劣条件。非劣允许的是有限幅度的退步，统计检验也不保证零风险。

`actor_publish_interval_updates` 是发布之间的最小轮数间隔；开启 arena 后，非评测轮不会发布。`arena_interval = 0` 显式禁用评测，此模式仍按间隔无条件发布，并在启动日志中标明 `ungated`。

`actor-gate` 日志输出当前 actor 轮次、对手数、最差置信下界、门槛和决定。需要额外对当前 actor 比赛时，`actor-match` 输出其得分率及区间。TensorBoard 记录 `actor/published` 和 `actor/worst_score_lower`。

## 验证

```text
cargo test --profile fast --lib --bin chineseai
cargo build --profile fast --bins
uv run --no-sync python tests/actor_gate_smoke.py <fast 主程序路径>
```

冒烟测试使用临时目录，覆盖未晋级但 actor 获准发布、独立当前 actor 对局、样本不足不发布，以及重启不绕过 best。

同次修复还使推理的威胁嵌入、稀疏策略表与投影累加器使用浮点计算，保持与训练前向一致，保留预计算和增量更新；权重文件格式不变。标签评测会排除规则终局，并提供 FPU 参数以便对齐训练内评测。
