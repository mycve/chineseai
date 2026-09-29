# αβ 自博弈进化

在 `codex/alphabeta-selfplay-promotion` 分支运行 `chineseai ab-evolve`。首次运行生成 `chineseai.ab-evolve.toml`；编辑后再次运行。`selfplay_nodes` 和 `arena_nodes` 分别限制每步自博弈搜索和晋级赛搜索的节点数。旧版配置不再兼容，请使用新生成的格式 31 配置。

自博弈使用当前冠军网络和 αβ 搜索产生样本，训练器据此更新候选网络。每隔 `arena_interval` 次更新，候选网络与冠军进行配对对局；只有达到 `arena_promotion_rate` 和置信区间要求才保存为新冠军，并交给自博弈线程。未晋级的候选仍可继续训练。`best.safetensors` 是默认 UCI 对弈模型。

搜索采用迭代加深、主变搜索、着法排序和叶节点静态搜索；局面走棋与撤回沿用象棋规则历史。策略训练标签由根着法搜索分数归一化得到。UCI 的 `SearchNodes`、`go nodes`、`go depth`、时限和 `stop` 可控制搜索。当前多 PV 只报告根着法。
