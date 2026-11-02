# 指定杀法：14 类识别、控制与自博弈训练

名单确认日期：2026-11-02。控制目标、终局标签、辅助头和统计统一使用以下 14 类；不再保留泛化的“车将死、炮将死、马将死、兵卒将死、双将、闷杀终局形、双车配合”分类。不加入三子归边、车马冷着、炮碾丹砂等集结形态或战术过程目标。

## 固定名单与识别范围

所有命中都要求真实将死：双方将帅存在，轮走方被将军且没有合法应着。仅有某种站位、子力数量或网络高概率不算完成目标。杀法是最终致胜机制，一步杀、两步杀是距离，不用固定步数排除某种杀法。

| UCI 目标 | 名称 | 工程判定范围 |
| --- | --- | --- |
| `double-cannon` | 重炮 | 将军炮与敌将同线，唯一炮架为己方另一炮，支持横向和纵向。 |
| `iron-bolt` | 铁门栓 | 中炮实际牵制中路防守子，车或兵在底线、肋道将死。 |
| `horse-cannon` | 马后炮 | 马是将军炮的唯一炮架，与将相距两格并实际封锁逃格。 |
| `heaven-earth-cannon` | 天地炮 | 底线炮直接将军，中炮实际牵制中路守子的严格子集。 |
| `smothered-cannon` | 闷宫 | 底线炮利用守方双士的炮架及自堵完成将死。 |
| `white-faced-general` | 白脸将 | 己将利用将帅不能照面的规则，实际封住敌将逃路。 |
| `elbow-horse` | 卧槽马 | 黑宫受攻时马在 `c8/g8` 直接将军；红宫为 `c1/g1`。 |
| `palace-corner-horse` | 挂角马 | 黑宫受攻时马在 `d7/f7` 直接将军；红宫为 `d2/f2`。 |
| `angler-horse` | 钓鱼马 | 黑宫受攻时马在 `c7/g7`，实际封逃；红宫为 `c2/g2`。 |
| `octagonal-horse` | 八角马 | 马与敌将占据九宫对角，马实际封逃；前角、底角均可。 |
| `high-angler-horse` | 高钓马／侧面虎 | 黑宫受攻时马在 `c6/g6`，其他子将军，该马实际封逃；红宫为 `c3/g3`。 |
| `double-horse` | 双马饮泉 | 一马将军，另一马也将军或实际封逃、保护将军马。 |
| `cross-rooks` | 双车错 | 一车将军，另一车实际封逃或保护将军车；双车分处不同横线和纵线。识别交错两线的末端构型，不反推此前反复交替将军。 |
| `two-ghosts` | 二鬼拍门 | 两个车或兵占据九宫两肋，每个都实际将军或封逃；支持双兵、车兵、双车，也支持第三子最后将军。 |

中文名称、连字符键、下划线键可用于 CLI/UCI；“铁门闩”“侧面虎”作为对应别名。TOML 使用下划线枚举名，例如 `high_angler_horse`。`none` 表示普通争胜。

标签可以重叠，例如双马饮泉与卧槽马、钓鱼马；不用互斥 softmax 强行分桶。“实际封逃”通过执行敌将的伪合法宫内走子后，检查是否仅该支持子阻止此着来认定；重复控制可能不计，属于明确的保守子集。铁门栓中炮另需验证真实牵制作用。未命中不代表没有其他杀法。

困毙单独记录为 `Stalemate`，不算 14 类目标成功。长将、规则判负、缺将、步数截断、仅有搜索证明都不冒充真实将死。

## 开局输入目标

```text
uci
setoption name EvalFile value multi-v55.safetensors
setoption name MatingTarget value horse-cannon
setoption name MatingStrength value 0.8
setoption name MatingSide value red
isready
ucinewgame
position startpos
go nodes 800
```

收到 `readyok` 后搜索。三个选项跨 `position`、回合及 `ucinewgame` 保留；黑方执行设置 `MatingSide=black`。强度为 `0..1`，默认 0，须调高才启用。CLI 示例：

```text
chineseai az-search multi-v55.safetensors 800 1.0 --mating-target 高钓马 --mating-strength 0.8 --mating-side red startpos
chineseai mate-classify "3aka3/9/9/4C4/9/4C4/9/9/9/4K4 b"
```

策略从第一步读取目标残差编码，配合受限布局先验；目标价值头预测固定执行方最终完成指令的概率。MCTS 在执行方节点使用 `普通 Q + 0.25 × strength × 目标价值 + PUCT 探索项`，目标概率回传不取反。普通 WDL、胜负证明保持原含义，对手正常争胜。强度不是损失权重。

关闭目标或必要子力不足时恢复普通策略：重炮至少两炮，马后炮需马炮，双马饮泉需两马，双车错需两车，二鬼拍门需至少两个车或兵；资源条件只用于认定明确不可达，不把搜索暂未找到当作不可能。目标几何先验只改变策略 logits，不能作为成功标签或杀法证明。

带目标也复用网络分支缓存、使用原 KLD 收敛停止。缓存要求目标、执行方、强度、参数及完整历史一致。执行方根节点首次证明普通必胜后，最多再探索 64 次寻找更符合目标的胜法，计入原总预算；已发现真实的指定杀法一手将死则立即停。目标概率是估计，不执行额外杀法 AND/OR 证明。

## 自博弈采样

顶层字段必须放在 `[mating_goal]` 之前：

```toml
format_version = 34
selfplay_opening_book = ""
mating_goal_mix = true
train_value_weight = 1.0
train_policy_weight = 1.0
mating_loss_weight = 0.1
goal_loss_weight = 0.1

[mating_goal]
target = "none"
side = "red"
strength = 0.8
```

- `mating_goal_mix=true`：约 1/3 对局普通争胜，其余 2/3 从确认的 14 种中均匀选目标；不另设目标列表参数。
- `mating_goal_mix=false`：固定使用 `[mating_goal].target`。
- 执行方固定用 `[mating_goal].side`，不随机红黑、不增加随机选方开关。canonical 编码共享双方参数，固定执行方的整盘数据也含执行方及对手两种回合。
- 每盘只指定一方执行目标，目标、执行方和强度盘内不变。对手正常争胜。空开局库从标准初始局面开始。

自博弈把带指令的 MCTS 策略训练回条件策略，尚无终局标签时也可以学习策略。最终真实完成指定杀法则固定执行方目标标签为 1，其他已观察结果为 0；双方此前局面使用相同执行方视角。截断、未知或样本时刻已无必要子力时，目标价值监督权重为 0。不能把旧策略标签事后重标成另一条指令。

14 个辅助头读出共享棋盘表示，预测当前行棋方最终以相应杀法获胜的概率，使用稳定 sigmoid BCE。`mating_loss_weight` 控制该辅助任务；`goal_loss_weight` 控制条件目标价值任务。WDL、策略和距离监督保留。目标价值损失加大不保证实际达成率提高，负例多时可能只让失败预测更准确。

## 模型与回放迁移

当前模型 v55：目标输入从 4 维扩为 28 维，杀法统计头由旧 18 类改为确认后的 14 类。普通加载器只接受当前格式，v54 模型须显式转换，输出文件必须不存在：

```text
chineseai az-upgrade-mating-targets old-v54.safetensors multi-v55.safetensors
```

保留原主干、WDL、策略、距离及原目标价值权重，搬移双方重炮、铁门栓条件编码；按名称重排保留的辅助头，去掉未入选的头。新增目标编码及新增辅助读出权重从零开始，新增辅助偏置采用初始模型的 -4。

回放 v46：八角马对角关系、二鬼拍门范围、双车错定义及标签集合已更新，旧 v45 标签不再混入。训练请复制到新的配置文件路径，设置 `model_path` 为转换后的文件、`checkpoint_dir` 为新目录，重建优化器及经验池；不要让启动器加载旧目录的最佳模型或旧 `.adamw.safetensors`。已有学习成果由转换的模型权重保留，新增指令仍需训练。

## 统计与效果判断

控制台只列 14 类命名杀法，显示批内总局数、实际将死、困毙、未知、各目标尝试／成功／未知及“目标执行红/黑”数量。杀法可重叠，计数不应相加作为总盘数。

TensorBoard：`mating/goal/{名称}/{attempts,successes,unknown,success_rate}`、`mating/goal_side/{red,black}/attempts`，累计对应 `mating_total`。训练和留出集记录 `goal_{bce,samples,target_rate,pred_mean,brier}`。目标率分母包括失败和未知尝试。

全局杀法率包含双方，指令成功率只算指定执行方，两者不能直接比较。应在同模型、同开局上对比无指令和指定指令的执行方胜率、命中率，同时记录平均模拟数、每秒模拟数和对局长度。CE 下降及条件策略变化不等于已学会从开局稳定执行指定杀法。

## 验证与来源

完整回归 380 项通过、3 项原有跳过，全部 14 类条件策略的 CPU/CUDA 一致性检查通过。真实 v54 文件已完成命令行迁移，高钓马命令行搜索、UCI 14 类选项及黑方执行配置均已检查。以上验证管线和局部行为，不表示开局完整对局达成率已经通过实验。

```text
cargo test --profile fast --offline --all-targets --quiet
cargo test --profile fast --offline --features slow-tests --lib goal_conditioning_matches_cpu_on_cuda -- --nocapture
```

检查真实命中、无杀反例、无关子反例、左右镜像、红黑翻转；CPU/CUDA 条件编码及策略一致性；回放指令读写；目标采样；模型迁移保留参数。开局测试核对指令对布局先验、MCTS 访问分布和训练标签的影响，并检查不使用终局价值标签时策略监督仍能训练目标编码。局部用例不替代服务器完整对局的命中率实验。

## 当前设计取舍与待验证项

- 红黑执行方随机不是正确编码的必要条件，已移除该功能。固定选方可能存在先后手分布差异，评价目标效果时使用同执行方、同开局的配对对照。
- 条件策略与目标价值读取目标，原 WDL 不读取目标。这样保留普通争胜估值作为基线，但当目标改变行为时，WDL 可能成为不同策略结果的平均估计。应比较各目标下的 WDL 校准，不能仅凭保留原头就宣称估值完全一致。
- 远期目标的成功监督稀疏，失败预测准确也能降低 BCE。支持 14 类不表示同时均匀训练 14 类就有足够正例；应看正例覆盖、常数频率基线及同开局指令增益，区分手写布局先验与学到的长程控制。
- 目标损失当前按整批样本数平均，有效监督占比变化会改变其有效贡献。报告中的有效样本平均 CE 与优化目标中的实际权重需分开理解。
- 已证普通必胜时，当前最终策略按目标概率最高值筛选；真实目标成功与网络估计的置信度不同，该筛选可能放大未校准估计的噪声。强度连续性、保留探索的软偏好应作为单独对照，不把估计当作杀法证明。

- [世界象联：基础杀法教学资料](https://www.wxf-xiangqi.org/images/hangzhou-chess/2022_47_guo_baocheng_.pdf)。
- [XQinEnglish：马类杀法及马位](https://www.xqinenglish.com/index.php?Itemid=522&catid=166&id=138:bk05horse00&lang=en&option=com_content&view=article)，包括马位名称差异与八角马对角关系。
- [教练编写的杀法总览](https://www.xiangqi.com/articles/checkmate-strategies)。
- [Xiangqi Arena 教学棋例](https://www.xqarena.com/zh-hans/guides/checkmate-patterns)，已有 FEN 与演示着法保留在测试中。新增独立构造用例明确标注为构造，不声称来自教材原题。
