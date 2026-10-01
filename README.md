# αβ 自博弈进化

## Pikafish 结构模型自动进化

运行 `cargo build --profile fast --bins`，然后运行 `target/fast/chineseai pikafish-evolve`（也可使用 `target/fast/chineseai ab-evolve --pikafish`）。首次运行自动创建 `pikafish-evolve.toml` 并开始训练，不需要手动串联自博弈、训练和晋级命令。Windows 可执行文件后缀为 `.exe`。

自博弈、GPU 训练与晋级测评重叠执行：`selfplay_workers` 个 CPU worker 使用共享的不可变冠军快照，自博弈结果经 `queue_games` 容量的内存队列进入 FIFO 回放池；队列满时 worker 等待。训练器随机抽样更新候选，独立测评服务使用 `arena_workers` 个 worker 并行完成交换红黑的配对晋级赛。测评固定候选快照，训练继续推进；晋级发布的是实际通过测评的那份快照。一个进程包含这些工作线程，worker 数量不等于进程数量。本机配置为 12 个自博弈 worker、4 个测评 worker。

Pikafish 稀疏层使用手写 CPU/CUDA 融合算子，直接累加特征行并在反向传播时汇总梯度，不生成 `index_select` 的特征矩阵。一个训练批次的两个视角一起处理，全连接层按 16 个网络桶合批执行，重复特征会重复累加，空特征输出零。搜索使用普通只读数组权重，worker 的累加器按特征差集加减、换边时交换双视角，每 32 次评估全量刷新以限制浮点误差；叶节点不创建 Candle 张量或访问共享张量锁。置换表保持原有 32768 个哈希槽和完整规则历史校验，只分配访问过的条目，并在线程内复用条目及历史缓冲。浮点权重格式保持不变；当前温度探索进度使用版本 3，旧随机步配置需使用新输出目录并通过 `seed_model` 继承冠军。

性能检查可运行 `cargo run --profile fast --example pikafish_perf -- --model runs/pikafish-evolve/best.safetensors --workers 128 --iterations 2000`。输出 JSON 包含预提取特征后的评估吞吐和八个固定局面的 128 节点搜索耗时、最佳着及价值。比较改动时必须使用同一权重、硬件和参数。Linux 内核开销用 `pidstat -u -w -t -p PID 1 5` 与 `vmstat 1 6` 同时采样；`vmstat` 首行是开机平均，线程级 `pidstat` 才能观察全部 worker 的上下文切换。

从开局库选起点后，每步都进行 αβ 搜索，再按根着回传分数采样实走着：`p(a) ∝ exp((q(a)-q_max)/T)`。`[temperature]` 配置 `start=0.05`、`end=0.005`、`plies=60`，温度在前 60 个半回合线性下降，随后保持 `end`；温度数值按网络 q 的尺度设置，尚未经过棋力对照调优，不能与访问次数温度的数值直接比较；设 `start=end=0` 即全部贪心，设 `end=0` 则后期贪心。取消了均匀随机开局步，也不对独立叶节点直接抽签。存在已证实必胜着时只采样必胜着；存在非必败着时不采样已证实必败着。晋级和固定基准测评保持最佳着。候选 TSV 的 `playedmove` 和 `score_cp` 记录实走着及其搜索分数，所有已搜索局面都可使用终局 MC 标签训练。AlphaZero 原论文按 MCTS 根访问数生成选着分布，当前实现按 αβ 回传 q 分数生成分布，两者不是同一搜索算法：[官方论文](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf)。

真实终局结果才进入训练，截断不计和棋。候选仅在对冠军得分的置信下界超过 `promotion_rate`、并且对固定初始模型没有显著退步时晋级；未晋级的候选继续训练。每局标记其冠军代数，超过 `max_champion_lag` 的队列结果丢弃，已进入回放的历史标签保留。训练使用无动量 SGD，权重、固定学习率、回放、开局游标和待完成测评一同保存；下次同一命令自动恢复，未完成测评用原权重和原开局重跑。启动目录锁防止两个训练进程同时写同一输出目录。

默认使用本地 `eval/pikafish-selfplay-5000-d20.sqlite` 做一次教师评分预训练，此后使用自博弈终局标签。要完全从零自博弈，删除配置中的 `bootstrap_sqlite`；已有本项目 Pikafish 浮点权重可以通过 `seed_model` 初始化，不能填写官方量化 `.nnue` 或旧 `AbNnue` 权重。初始基准在预训练前固定，故首次晋级可能包含教师预训练带来的提升；这不能单独证明后续纯自博弈持续增长。

输出目录的 `dashboard.html` 每 30 秒刷新，显示候选比赛得分、训练损失和冠军代数，曲线只显示通过晋级的冠军对固定初始模型的实测得分。`metrics.csv` 提供每次比赛的得分及置信区间。区间按配对开局计算，是逐次区间；长期重复测评不代表全程错误率控制。得分不是绝对 Elo，损失下降也不能代替棋力测评。固定基准及其开局在续训时保留，且这些开局不作为自博弈起点。每次晋级赛重新抽取开局，搜索预算保持固定。

`--target-update N` 在完成第 N 次训练更新后保存退出；`target/fast/chineseai pikafish-evolve --stop` 请求后台进程保存后停止，前台可按 Ctrl+C。停止会在当前搜索返回后生效，未完成对局不入训练。续训允许调整 `selfplay_workers`、`arena_workers`、`queue_games`；其余训练参数必须保持一致，更改时使用新的 `output_dir`。输出目录中的 `best.safetensors` 由 UCI 的 `EvalFile` 直接加载，UCI 自动识别新浮点网络的格式。旧 `ab-evolve` 的默认模型及配置仍独立保存。

本机 Ryzen 9 9950X、Windows、`fast` 构建的同权重对照见 [benchmarks/pikafish-fused-20261001.json](benchmarks/pikafish-fused-20261001.json)。八个固定局面的串行 128 节点搜索从约 0.350 秒降至 0.012 秒，约快 29 倍，最佳着全部一致，价值最大误差约 `1.34e-7`。16 worker 的预提取特征评估从约 4 万次/秒升至 278 万次/秒。固定局面会复用缓存，这些数字不代表 Linux 服务器长期对局吞吐或棋力增长；服务器的 `%system` 必须重新采样验证。

128 物理核心服务器的正式长跑配置见 [configs/pikafish-evolve-128c.toml](configs/pikafish-evolve-128c.toml)：96 个自博弈 worker、24 个测评 worker，单步预算分别为 8192 和 32768 节点，512 对晋级开局、200 万局面回放。它使用 `runs/pikafish-evolve-128c-temperature` 新目录，并从当前 `runs/pikafish-evolve/best.safetensors` 继承冠军；没有该文件时应删除 `seed_model` 或填写已有的本项目浮点权重。启动命令为 `RAYON_NUM_THREADS=8 target/fast/chineseai pikafish-evolve configs/pikafish-evolve-128c.toml`；该环境变量约束全局 Rayon 辅助线程，不改变配置中独立创建的自博弈和测评线程数。这份配置提高搜索和训练投入，不承诺顶级棋力：当前约 5000 条教师数据、有限自博弈以及尚未对齐强引擎的搜索，不能支撑这种承诺。应扩大经过核验的教师局面数据，并持续通过与强引擎交换红黑、固定预算的对局检验绝对棋力；内部冠军晋级只表示相对增长。

## 原有 AbNnue 模型

运行 `chineseai ab-evolve`。首次运行生成 `chineseai.ab-evolve.toml`；编辑后再次运行。`selfplay_nodes` 和 `arena_nodes` 分别限制每步自博弈搜索和晋级赛搜索的节点数。`arena_openings` 指定每次晋级赛的配对开局数，每个开局交换红黑各走一局；默认 1000，即冠军对局 2000 局，历史对手存在时从中划分开局。价值单头模型使用格式 32 配置；旧配置、模型、回放和训练状态不兼容，需要重新开始训练。

自博弈使用当前冠军网络和 αβ 搜索产生样本，训练器据此更新候选网络。网络仅预测局面价值，着法由搜索根评分确定；温度采样只影响自博弈选着，不生成策略训练标签。对局截断时不把截断当成和棋价值标签。每隔 `arena_interval` 次更新，候选网络与冠军进行配对对局；只有达到 `arena_promotion_rate` 和置信区间要求才保存为新冠军，并交给自博弈线程。未晋级的候选仍可继续训练。`best.safetensors` 是默认 UCI 对弈模型。新配置默认 `hidden_size = 256`。

训练器按 `train_samples_per_update` 和 `batch_size` 每次更新模型，不再等固定训练周期结束。`checkpoint_interval` 指定模型、优化器、回放池和进度一起保存的更新间隔。`ab-evolve --target-update N` 可在完成第 N 次更新后保存并退出。留出集按优化步数定期评估。

搜索参考 Pikafish／Stockfish 的迭代加深、主变搜索、置换表、aspiration 窗口、着法排序、晚序着法缩减和叶节点静态搜索。置换表只有在完整规则历史一致时才复用分数，避免长将、长捉与循环局面误判。AB 根评分生成自博弈探索权重，权重不进入模型训练。UCI 的 `SearchNodes`、`go nodes`、`go depth`、时限和 `stop` 可控制搜索。当前多 PV 只报告根着法。

网络借鉴稀疏特征累加与王桶变化时刷新的做法，搜索叶节点只计算价值头，并复用增量缓冲。当前模型是项目自己的浮点 WDL 价值单头结构，模型文件使用 safetensors；不兼容 Pikafish 的量化 `.nnue` 文件。搜索剪枝仍需通过象棋规则、节点预算和训练推理一致性测试逐项验证。

最新 Pikafish 网络的精确维度、权重表和训练接线验收见 [PIKAFISH_NNUE_DESIGN.md](PIKAFISH_NNUE_DESIGN.md)。已实现 HalfKAv2_hm、FullThreats 特征编号、完整浮点前向、AB 搜索和自有权重自博弈训练链。新模型通过 `pikafish-evolve` 或 `ab-evolve --pikafish` 接入持久化回放、自动晋级与续训流程。ChineseAI UCI 能加载这条链路的浮点冠军权重，也能直接加载 `tools/pikafish.nnue`；七个测试局面的原始 NNUE 整数值与 Pikafish 一致。搜索分数、最佳着和节点数仍不同，剪枝尚未对齐。

对照测试可让 Pikafish 自己加载其权重：`chineseai vs-pikafish tools/pikafish.exe best.safetensors --pikafish-nnue tools/pikafish.nnue --games 2 --parallel-games 1 --opening-book ""`。程序通过 UCI `EvalFile` 设置该文件，执行一次深度 1 的预检搜索，并核对 Pikafish 报告的实际加载路径；随后才开始对局。`best.safetensors` 始终由 ChineseAI 自己加载与训练，Pikafish 的 `.nnue` 只用于对手测试，可在测试结束后删除。

还可运行 `chineseai pikafish-selfplay --games 1 --depth 4 --output target/fast/pikafish-selfplay.tsv` 生成 Pikafish 同权重双边自博弈的 FEN、搜索着法、行棋方视角 `score_cp` 与结果。开局随机步不写入样本；截断对局结果为 `?`，不得当作和棋标签。运行 `chineseai pikafish-pretrain --input target/fast/pikafish-selfplay.tsv --output target/fast/candidate.safetensors` 用 Pikafish 教师分数训练自己的浮点权重。随后运行 `chineseai pikafish-candidate-selfplay --model target/fast/candidate.safetensors --output target/fast/candidate-selfplay.tsv`，再以 `chineseai pikafish-train-selfplay --input target/fast/candidate-selfplay.tsv --output target/fast/candidate.safetensors --resume` 使用真实终局结果更新同一权重。训练标签按 FEN 行棋方翻转红方结果；截断对局跳过，若全部截断则拒绝训练。候选自博弈的 `score_cp` 是自身搜索价值的逆映射，仅用于诊断，不作为训练标签。两种 TSV 以 `source` 列区分；旧版无该列文件必须重新生成，避免把候选分数误用作教师分数。`chineseai pikafish-candidate-arena --candidate 新权重 --champion 旧冠军 --pairs 100 --promote-output 新冠军路径` 从开局库抽取不同局面并交换红黑；配对得分置信下界超过门槛才发布新冠军，未终局截断会报错。

与 Pikafish 的实现对应关系：搜索采用有界历史分数、失败安静着惩罚及保守的历史相关 LMR；当前参数是本项目的启发式，尚未经过大规模对局调参。UCI 支持 `UCI_ShowWDL` 和 `Move Overhead`，同时保留项目专用的 `EvalFile`、`SearchNodes` 与象棋规则选项。Pikafish 的 `Threads`、`Hash`、`Ponder` 暂无对应的完整执行机制，当前引擎不声明这些选项。网络的王桶及双视角增量累加参考其特征处理思路，但当前浮点特征、价值头与训练格式均为独立实现；模型不能直接互换。

参考源码：[Pikafish 搜索](https://github.com/official-pikafish/Pikafish/blob/master/src/search.cpp)、[Pikafish 网络结构](https://github.com/official-pikafish/Pikafish/blob/master/src/nnue/nnue_architecture.h)、[Pikafish 特征](https://github.com/official-pikafish/Pikafish/blob/master/src/nnue/features/half_ka_v2_hm.h)、[Stockfish 搜索](https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp)。
