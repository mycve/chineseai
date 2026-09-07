# TensorBoard 统计移植核对

基线为 `90a0c8a`，对照恢复前的 `a34ebc0`。本次仅移植监控，不修改自博弈标签、搜索参数和 loss 归一化。

## 历史优化与当前状态

| 历史提交 | 内容 | 当前状态 |
| --- | --- | --- |
| `9277a6d` | 累计对局、样本计数持久化；采集耗时排除发送队列阻塞 | 基线已包含 |
| `a317ee4` | value 统计只计有效样本；空分组不绘制；完局与截断分开；终局分类 | 统计部分已补齐 |
| `e9c22c4` | 重复局面和棋是规则完局，不是截断 | 当前保留这一口径 |
| `24a8254` | 全局、3 个阶段、9 个来源阶段的 value 矩统计合并计算，并脱离反向传播图 | 已移植；与标量参考比较 |
| `a34ebc0` | 诊断评估携带规则历史 | 已在 `a40e6e5` 移植 |

`a40e6e5` 还已加入 4、12、32 步三个短期 value 头的 CE、RMSE、相关性及有效样本数。

## 新增及改名指标

以下对局计数均为当前更新收集的自博弈对局，不是累计计数。

- `selfplay/completed_games`：总对局减去步数预算截断和搜索无返回动作。
- `selfplay/draw_rate_completed`：规则和棋数 / 完局数；没有完局时不写入。
- `truncation/rate`：截断数 / 总对局数；没有对局时不写入。
- `truncation/max_plies`：原 `terminal/max_plies`，达到自博弈预算上限。
- `truncation/search_no_move`：有合法动作但搜索未返回动作的异常中止。
- `terminal/checkmate`、`terminal/stalemate`、`terminal/rule_blocked`：拆分原 `terminal/checkmate_no_legal_moves`。依次表示将死、困毙、棋盘有合法动作但规则过滤后无动作。中国象棋困毙仍判负。
- `terminal/draw_natural_limit`、`terminal/draw_insufficient_material`、`terminal/draw_repetition`：原对应 `rule_draw_*` 标签改名；自然限着与重复和棋均属于完局。双方长将、长捉仍保留独立原因指标。
- `train/value_samples`：参与 value 统计的样本次数（`value_weight > 0`），不是去重局面数，也不是完局样本数。多个训练 epoch 会累计。

保留已有总对局、总样本、训练速度、策略分布、阶段与来源阶段、三个短期 value 头等统计。

## 对读图的限制

旧训练行为仍把预算截断记作和棋标签，value 权重仍为 1。本次仅从完局统计中排除这些对局，不能将 `train/value_samples` 理解成全部都有真实终局监督。控制台原 R/B/D 仍是旧标签口径。

`a317ee4` 中屏蔽未知终局训练标签、按各损失分量的权重和归一化，会改变训练；未在此次监控移植中引入。自博弈存储和规则引擎优化也未混入。

改名会产生新曲线；历史事件文件不会被重写。已运行的训练进程需要在保存进度后重新启动新程序，才会输出新增指标。

## 验证

- `cargo check --profile fast --bin chineseai`、`cargo build --profile fast --bin chineseai` 通过。
- `cargo test --profile fast --lib monitoring -- --test-threads=1`：4 项通过。覆盖终局分类、单局及批量截断、无效样本隔离，以及合并矩统计与标量参考的一致性（CPU，CUDA 可用时也测试）。
- 使用 `runs/monitoring-smoke/config.toml` 完成两轮独立短训练，读取实际事件文件验证。第一轮 4 局全部截断，完局数为 0，未写完局和棋率；第二轮 1 局将死、3 局截断，完局数为 1，截断率为 0.75，完局和棋率为 0。两轮有效 value 样本数均为 256，三个短期 value 头指标均存在。
- 这些结果验证统计实现，不代表棋力或训练速度收益。
