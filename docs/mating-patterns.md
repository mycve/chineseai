# 指定杀法：四类控制与自博弈训练

2026-11-03 用户最终确认：只保留**重炮、铁门栓、马后炮、高钓马**，其他杀法全部退出识别器、训练、统计和 UCI。模型 v59、回放 v50；四个辅助杀法头，八维执行方／目标编码。配置格式仍为 v34，进度格式仍为 v8。

## 固定名单

| UCI 英文键 | 中文名 | 实际将死时的判定 |
| --- | --- | --- |
| `double-cannon` | 重炮 | 将军炮与敌将同线，唯一炮架是己方另一炮，支持横向和纵向。 |
| `iron-bolt` | 铁门栓 | 中炮真实牵制中路防守子，车或兵在底线、肋道将死。 |
| `horse-cannon` | 马后炮 | 马是将军炮的唯一炮架，与敌将相距两格，且实际封锁逃格。 |
| `high-angler-horse` | 高钓马／侧面虎 | 攻黑宫时马位于 c6/g6，攻红宫时位于 c3/g3；由其他子将军，该马实际封逃。 |

识别必须是真实将死：将帅均存在，轮走方被将军且无合法应着。站位、网络概率或搜索证明本身不算实际完成。困毙单独记录，其他将死仍正常获胜，只是四类标签均不命中。

中文、连字符和下划线英文键均可输入；“铁门闩”“侧面虎”保留别名。TOML 使用下划线键。普通争胜用 `none`。已删除的名称在 CLI/TOML 明确报错；UCI 提示无效目标并退回普通争胜、清除目标证明缓存，避免误用上一种目标。

## UCI

`MatingTarget` 下拉框只有普通争胜及四种杀法。默认铁门栓、强度 0.8；显式保存的设置优先。更新引擎后让 GUI 重新读取选项。

```text
uci
setoption name EvalFile value model-v59.safetensors
setoption name MatingTarget value 高钓马
setoption name MatingStrength value 0.4
setoption name MatingSide value red
isready
ucinewgame
position startpos
go nodes 20000
```

等待 `readyok` 后搜索。`MatingSide` 默认 auto，在本盘第一次 go 绑定引擎执子方，直到 ucinewgame 才重新绑定；也可显式 red/black。目标与强度跨回合保留。强度 0 或 none 关闭目标。

## 自博弈与重训

在新目录初始化新模型，使用新配置路径、经验池和优化器。不要把旧权重、旧 best、旧优化器或旧回放自动续入。示例先创建 style4 目录，再运行：

```text
chineseai az-init 96 style4/model.safetensors 20261103
```

原配置字段不变。为观察精简名单的效果，先保留用户现用强度 0.4、目标损失 0.1：

```toml
format_version = 34
selfplay_opening_book = "book.pgn.gz"
mating_goal_mix = true
mating_loss_weight = 0.1
goal_loss_weight = 0.1

[mating_goal]
target = "iron_bolt"
side = "red"
strength = 0.4
```

目标混合仍约为 90% 目标局、10% 普通争胜；在起局仍具备必要资源的四类中均匀采样。标准起局全部具备时，每类约占总对局的 22.5%。不额外随机红黑，执行方固定按配置，目标和执行方盘内不变。对手使用普通争胜收益，不配合目标。关闭混合时使用固定指令。

资源不足立即关闭条件偏好：重炮需两炮；铁门栓需炮、车或兵以及可被牵制的防守子；马后炮需马炮；高钓马需马及其他攻击子。关闭后继续正常争胜，不终止棋局。原目标前缀按失败监督，资源丢失后的样本关闭目标条件；WDL 使用真实结果。暂未搜到、低预测概率或几步没进展不能当成不可达。

策略从第一步读取指令，MCTS 策略训练回条件策略；目标价值学习原指令实际成功／失败，四个辅助头学习杀法标签。WDL 保留，成功轨迹还有按完成距离降权的额外策略监督。正常用尽步数预算记目标失败，未观察的 WDL 仍屏蔽。不开新损失或采样参数。CE 下降不等于实际命中率提高。

## 搜索收束与局限

普通必胜的证明距离和条件策略的预计对局长度分开。高钓马先验使用空盘马步距离，避免直线距离奖励错误跳法。胜势明确时，现有 MLH 门槛和距离偏好参与目标收益比较，胜负不明确时不强行追短。

目标证明只从真实匹配将死传播：执行方需要一条合法证明路线，对手节点必须覆盖全部应对。搜索可在预算内寻找更短证明；最终优先最短的已知目标证明。UCI 和自博弈保存完整证明分支，按目标、执行方、完整历史、当前位置及 searchmoves 校验，剩余杀距上界逐手递减；更短证明可以替换它。UCI 输出 `mating proof plies=... source=goal-proof`。

**尚无证明时仍可能高估目标概率、挪帅和绕路。** 精简名单有利于集中样本和调试，不能直接证明这些问题已解决，也不保证增加模拟数就能收束。不为此降低对手强度，不把目标失败伪造成棋局负局。

用户二鬼拍门反馈末局仍有车马兵；旧定义允许车兵，因此当时没有关闭目标。冻结原权重、强度 0.4、20000 次模拟、固定普通黑方继续完整历史，普通模式 10 半回合获胜，旧目标模式 36 半回合完成车兵杀形，中间仍有绕路。用户决定删除该类。移除后，GUI 残留的中英文旧目标会恢复普通模式；同一第 206 半回合局面已验证与显式 none 一致，选择 d5c5。

## 评测与旧模型对照

只为推理对照提供显式迁移，输出路径必须不存在：

```text
chineseai az-upgrade-mating-targets best.safetensors target/best-mating4-v59.safetensors
cargo run --profile fast --offline --example mating_goal_eval -- target/best-mating4-v59.safetensors --cases eval/mating-goal-user-history.toml --ordinary-opponent --target 高钓马 --simulations 20000 --max-plies 120 --strength 0.4 --output high-angler-eval.toml
cargo run --profile fast --offline --example pawn_threat_probe -- target/best-mating4-v59.safetensors --output pawn-threat.toml
```

v54 至 v58 可显式迁移到 v59，仅保留当前目标对应参数。v56 的四类行来自旧辅助头 0、1、2、10；v57/v58 来自 0、1、2、8。执行方两组编码分别映射，已训练门控保留；v54/v55 缺失门控补零。普通加载器严格检查版本。此次选择重新训练，不迁移旧经验池或优化器。

保留四类的一步杀、关键参与子移除、无杀／无关子反例、镜像、红黑翻转、资源关闭、完整目标证明、历史与 searchmoves 限制、模型迁移和 CPU/CUDA 一致性检查。开局留出集固定起点，不按结果筛选；用例展开只含当前四类。

四类版本 fast 编译通过，完整回归 409 项通过、4 项跳过，另行启用的 CPU/CUDA 条件输出一致性检查通过。真实 v56 直接裁剪与经过 v58 后再裁剪的全部张量逐字节一致。用户三个高钓马完整历史在强度 0.4、20000 次模拟、固定普通黑方下仍分别用 26／12／8 半回合完成，与精简前相同；这验证保留目标行为，没有宣称四类重训已经提高成功率。

过河兵输入已有位置、王与子关系、攻击及保护特征。原权重在四个构造兵车对照中已表现出组合响应并避开送一手杀；不能据此证明长期威胁全部解决，也没有据猜测增加固定兵价值或新网络头。

更早的 14／12／11 类版本、用户高钓马残局与留出集实验归档在 [历史实验记录](mating-patterns-history.md)。旧统计不是当前四类重训效果，旧用例数量也不能直接作为当前分母。
