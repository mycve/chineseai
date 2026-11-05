# 普通残局表监督

这是从现有权重继续训练的离线工具。仅 `endgame` feature 编译查询器；普通引擎、自博弈和 MCTS 尚未接入查表，也不会自动替换 `best.safetensors`。

## 表集与规则

数据来自 [ChessDB](https://chessdb.cn/egtb_info_dtm.html) 的公开 DTM、DTC 表，索引来自 [ChessDB 上游](https://github.com/noobpwnftw/chessdb)。2026-11-05 核对的公开 FTP 中，五子以内有 209 类成对表，合计约 285 MiB；六子以内约 48 GiB。公开全部 DTM 与 DTC 共约 28.1 TB。这些是已发布表的范围，并不代表所有象棋残局均有表。

表采用亚洲规则，不能直接当成本项目完整历史棋规的真值。当前流程保守接收：

- 胜负：表值仅用于引导证明。进攻方至少一个分支获胜，防守方所有合法应对均获胜，且使用项目的完整将捉、重复和自然限着判定。只有整条教师轨迹抵达真实终局才训练；剩余步数来自实际完成的轨迹。
- 三子和棋：构建有界安全图，分别证明双方均可只用不将、不捉的走法避免失败。对手全部棋盘合法应对都须覆盖。未知边按证明失败处理。
- 其他表和棋：双方按表保和，必须走到项目完整棋规实际判和才接收对局回报。它是教师策略的 RL 回报，不冒充极小极大和棋证明；只训练 WDL，策略和剩余步数权重为零，防止把保和循环学成收尾策略。报告单列这类对局数量。训练集按固定棋盘键抽样，和棋最多约占三分之一。
- 缺表、两种表的胜负冲突、证明预算不足以及未实际完成的对局均跳过，不把未知或截断标为和棋。

查询器按需解压，解压缓存上限 64 MiB；压缩表只读映射，使用期间不要替换或修改表文件。当前索引最多支持每方三个兵卒，超出范围返回未知。

## 使用

需要 Rust、C++17 编译器，以及标准库 Python。先下载五子以内的全部已发布表，脚本默认只预览，加 `--download` 才下载。支持失败重试和断点续传；既有表不覆盖；默认新增下载上限 1 GiB。

```powershell
python tools/endgame_fetch.py --max-pieces 5 --download
python tools/endgame_seeds.py --output endgame.fens --per-material 8
cargo build --profile fast --features endgame --example endgame_train
```

脚本生成的 FEN 是候选，Rust 查询器会再次验证合法性和可证明性。工具读取当前 v60 格式权重；若是 v59，先用已有的 `az-upgrade-mating-targets 输入权重 输出权重` 命令仅转换格式，不重新初始化参数。

先只评估，不训练：

```powershell
target/fast/examples/endgame_train.exe --model best.safetensors --seeds endgame.fens --output endgame-baseline.safetensors --evaluate-only
```

再从原权重微调，普通 Px0 棋谱作为保留原行为的蒸馏锚点，比例尽可能与残局样本相同。这里的 `--anchors` 是普通 Px0 数据包路径，不是权重文件。

```powershell
target/fast/examples/endgame_train.exe --model best.safetensors --seeds endgame.fens --anchors data/data.bin --output endgame-candidate.safetensors
```

Linux 可通过 `cargo run --profile fast --features endgame --example endgame_train --` 加相同参数执行。完整参数见 `--help`。

每条教师轨迹整体分入训练或留出集；再移除训练集与留出集重合的棋盘及左右镜像、颜色规范化等价局面。同类子力仍可同时出现在两组，因此这是同分布留出测试，不能证明对未见过的子力类型也泛化。

报告写入输出路径对应的 `.report.toml`，记录参数、覆盖、未知、静态指标以及实际兑现测试的 FEN 和长度。兑现测试固定搜索预算 512，每种子力最多一个起点，最多 16 局；每个起点都重新在空历史下证明可赢。对手优先按表值争取最好胜负、延长败局，缺表时回退同一个冻结网络。表只引导对手选择合法动作，最终结果仍按项目完整历史棋规结算。报告记录查表和回退次数。`--evaluate-only --opponent 原权重` 可独立检查候选。它是小规模回归检查，不能替代完整棋力对弈。

提供普通棋谱锚点时，训练后另外选取最多 8 个至少剩 24 子的普通起点，候选与基线交换红黑，共最多 16 局，使用原有 arena 实现，固定 512 次搜索和 256 半回合上限。报告记录得分和置信区间；小样本只能检查明显退化。只读复查候选时同时传入 `--anchors` 和 `--opponent` 才运行这组普通棋力测试。

## 如何验收

先看留出残局实际胜负、是否超长循环、成功收尾所需步数，再看胜负预测和策略误差。误差下降不等于棋力提高。另需用普通对弈验证遗忘程度，再决定是否用候选继续自博弈。

不要把有限采样微调理解为已经把所有表背进模型。当前没有自动启动预热或持续混入自博弈训练；后续自博弈也可能遗忘这些监督。对于要求精确兑现的局面，仍需另行设计符合当前棋规的搜索查表接口。
