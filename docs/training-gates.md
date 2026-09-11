# 自博弈模型发布与晋级

learner 持续训练，自博弈使用已晋级的 best。只有 arena 判定 Promote 时，才保存新 best 并更新自博弈模型；更新在恢复自博弈工作线程前完成。

- 晋级规则不变：对当前冠军的得分率置信下界超过晋级线，历史对手检查未发现显著退步。
- Continue 或 Reject 时保留原 best，自博弈继续产数，learner 继续训练。
- 重启从 best 恢复自博弈模型；没有 best 时以初始 learner 建立 best。
- `arena_interval = 0` 禁用晋级评测，并固定使用当前 best，不会定期发布 learner。
- 已删除独立 actor 发布间隔、非劣界限、最小对局数及其额外评测、日志。

旧配置升级前须移除 `actor_publish_interval_updates`、`actor_noninferiority_margin` 和 `actor_gate_min_games` 三项；配置解析会拒绝这些已删除的字段。

## 验证

```text
cargo test --profile fast --bin chineseai
cargo build --profile fast --bin chineseai
uv run --no-sync python tests/actor_gate_smoke.py target/fast/chineseai.exe
```

冒烟测试使用临时目录，覆盖未晋级不发布、重启使用 best、禁用 arena 不发布，以及晋级后更新自博弈模型。测试中的零晋级线仅用于强制覆盖发布路径，不是训练建议。

训练与推理的浮点一致性修复保持不变。
