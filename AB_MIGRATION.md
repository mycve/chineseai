# αβ 自博弈进化

运行 `chineseai ab-evolve`。首次运行生成 `chineseai.ab-evolve.toml`；编辑后再次运行。`selfplay_nodes` 和 `arena_nodes` 分别限制每步自博弈搜索和晋级赛搜索的节点数。`arena_openings` 指定每次晋级赛的配对开局数，每个开局交换红黑各走一局；默认 1000，即冠军对局 2000 局，历史对手存在时从中划分开局。旧版配置不再兼容，请使用新生成的格式 31 配置。

自博弈使用当前冠军网络和 αβ 搜索产生样本，训练器据此更新候选网络。自博弈温度采样依据根着法的搜索策略分布；对局截断时不把截断当成和棋价值标签。每隔 `arena_interval` 次更新，候选网络与冠军进行配对对局；只有达到 `arena_promotion_rate` 和置信区间要求才保存为新冠军，并交给自博弈线程。未晋级的候选仍可继续训练。`best.safetensors` 是默认 UCI 对弈模型。新配置默认 `hidden_size = 256`；已有模型恢复训练时须保持其原网络宽度。

搜索参考 Pikafish／Stockfish 的迭代加深、主变搜索、置换表、aspiration 窗口、着法排序、晚序着法缩减和叶节点静态搜索。置换表只有在完整规则历史一致时才复用分数，避免长将、长捉与循环局面误判。策略训练标签由根着法搜索分数归一化得到。UCI 的 `SearchNodes`、`go nodes`、`go depth`、时限和 `stop` 可控制搜索。当前多 PV 只报告根着法。

网络借鉴稀疏特征累加与王桶变化时刷新的做法，搜索叶节点只计算价值头，并复用增量缓冲。当前模型仍使用项目自己的浮点 WDL／策略双头结构；尚未移植 Pikafish 的量化多层网络、攻击特征或 `.nnue` 文件格式。搜索也尚未实现空步剪枝、静态交换评估和 ProbCut。这些功能需在象棋规则和训练推理一致性测试下逐项验证。

参考源码：[Pikafish 搜索](https://github.com/official-pikafish/Pikafish/blob/master/src/search.cpp)、[Pikafish 网络结构](https://github.com/official-pikafish/Pikafish/blob/master/src/nnue/nnue_architecture.h)、[Pikafish 特征](https://github.com/official-pikafish/Pikafish/blob/master/src/nnue/features/half_ka_v2_hm.h)、[Stockfish 搜索](https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp)。
