# αβ 自博弈进化

运行 `chineseai ab-evolve`。首次运行生成 `chineseai.ab-evolve.toml`；编辑后再次运行。`selfplay_nodes` 和 `arena_nodes` 分别限制每步自博弈搜索和晋级赛搜索的节点数。`arena_openings` 指定每次晋级赛的配对开局数，每个开局交换红黑各走一局；默认 1000，即冠军对局 2000 局，历史对手存在时从中划分开局。价值单头模型使用格式 32 配置；旧配置、模型、回放和训练状态不兼容，需要重新开始训练。

自博弈使用当前冠军网络和 αβ 搜索产生样本，训练器据此更新候选网络。网络仅预测局面价值，着法由搜索根评分确定；温度采样只影响自博弈选着，不生成策略训练标签。对局截断时不把截断当成和棋价值标签。每隔 `arena_interval` 次更新，候选网络与冠军进行配对对局；只有达到 `arena_promotion_rate` 和置信区间要求才保存为新冠军，并交给自博弈线程。未晋级的候选仍可继续训练。`best.safetensors` 是默认 UCI 对弈模型。新配置默认 `hidden_size = 256`。

训练器按 `train_samples_per_update` 和 `batch_size` 每次更新模型，不再等固定训练周期结束。`checkpoint_interval` 指定模型、优化器、回放池和进度一起保存的更新间隔。`ab-evolve --target-update N` 可在完成第 N 次更新后保存并退出。留出集按优化步数定期评估。当前命令行只保留 `ab-evolve`、`vs-pikafish` 和 `pikafish-label-eval`，其他离线实验入口已删除。

搜索参考 Pikafish／Stockfish 的迭代加深、主变搜索、置换表、aspiration 窗口、着法排序、晚序着法缩减和叶节点静态搜索。置换表只有在完整规则历史一致时才复用分数，避免长将、长捉与循环局面误判。AB 根评分生成自博弈探索权重，权重不进入模型训练。UCI 的 `SearchNodes`、`go nodes`、`go depth`、时限和 `stop` 可控制搜索。当前多 PV 只报告根着法。

网络借鉴稀疏特征累加与王桶变化时刷新的做法，搜索叶节点只计算价值头，并复用增量缓冲。当前模型是项目自己的浮点 WDL 价值单头结构，模型文件使用 safetensors；不兼容 Pikafish 的量化 `.nnue` 文件。搜索剪枝仍需通过象棋规则、节点预算和训练推理一致性测试逐项验证。

参考源码：[Pikafish 搜索](https://github.com/official-pikafish/Pikafish/blob/master/src/search.cpp)、[Pikafish 网络结构](https://github.com/official-pikafish/Pikafish/blob/master/src/nnue/nnue_architecture.h)、[Pikafish 特征](https://github.com/official-pikafish/Pikafish/blob/master/src/nnue/features/half_ka_v2_hm.h)、[Stockfish 搜索](https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp)。
