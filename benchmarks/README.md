# Pikafish 同权重 UCI 限时对弈（2026-09-30）

从 `book.pgn.gz` 用种子 `20260930` 抽取 100 个不同 FEN，每个开局交换红黑各下一局，共 200 局。ChineseAI UCI 与 Pikafish UCI 均加载 `tools/pikafish.nnue`，每步发送 `go movetime 500`。同时运行 15 局，即 30 个引擎进程；单引擎默认 1 线程。使用项目象棋规则裁判，每局最多 200 半回合。

| ChineseAI | 胜 | 负 | 和 | 异常 | 得分率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 执红 100 局 | 0 | 98 | 2 | 0 | 1.0% |
| 执黑 100 局 | 0 | 100 | 0 | 0 | 0% |
| 合计 | 0 | 198 | 2 | 0 | 0.5% |

198 局以无合法着结束，2 局按规则判和。平均 35.25 半回合。ChineseAI 共走 3,475 步，平均响应 490.7 ms；Pikafish 共走 3,575 步，平均响应 447.8 ms。抽查三个无合法着终局，两引擎均返回无着。完整逐局结果在 [TSV](pikafish_2026-09-25_100_openings_500ms.tsv)。

运行命令：

```powershell
cargo build --profile fast --bin chineseai --bin chineseai-uci
target\fast\chineseai.exe uci-tournament --opening-positions 100 --parallel-games 15 --movetime-ms 500 --output target\fast\uci-tournament-100.tsv
```

本地 Pikafish 自报 `2026-09-25`，无法从公开仓库确认其对应源码提交；它与 ChineseAI 使用同一个 NNUE 文件。文件 SHA-256：

- `pikafish.exe`: `9824FFF4E3C4A72AFBC6CFDC611C140B32F79E3C5FD5FE4BF6D747833875B86D`
- `pikafish.nnue`: `7D13D73569A9B571BA0EB20CF1596247BC2A42738967E61AFEF6482B231E900E`
- `chineseai-uci.exe`: `AFC5101204681F23773D1FDBB044E8BF68E8A2D986128A71746F350525C3BFD7`

结果衡量当前两套 UCI 搜索实现在同权重、相同每步墙钟时间下的棋力差距。ChineseAI 仍在使用项目自己的搜索和评估缩放；这次对弈不构成剪枝逐项一致性验证。

## 原因复核

对弈器逐局重建同一开局 FEN 和完整走子历史，交换红黑，并用项目合法着生成器裁判。100 对开局全部完整、FEN 两两相同。抽查三个无合法着终局，两引擎均报告无着；没有协议异常或超时判负。

新增的 [500 ms 诊断样本](diagnostic_500ms_10_openings.tsv) 使用前 10 个开局交换红黑、15 局并发，共 20 局：ChineseAI 0 胜 20 负。双方每局累计响应时间接近（约 9.6 秒／9.0 秒），平均完成深度 **5.26／46.17**（ChineseAI／Pikafish）。Pikafish 在部分优势局面达到极深的强制搜索；以更普通的初始局面单进程测试，深度为 **5／21**、节点约 **5.3 万／80 万**，说明限时深度差并非计时不公。

[固定 depth 5 诊断样本](diagnostic_depth5_10_openings.tsv) 使用同一 10 个开局，20 局为 ChineseAI **1 胜 17 负 2 和**、异常 0。ChineseAI 平均完成深度 4.99，Pikafish 为 5.00；每局累计节点约 **49.2 万／1.94 万**。固定深度测试已将 ChineseAI `SearchNodes` 上限调到 100,000,000，避免默认 10,000 节点提前截断。

差距有两层：相同 500 ms 下 ChineseAI 的节点吞吐约为 Pikafish 的十四分之一，而且到达相同 depth 5 所需节点约多 25 倍。即使控制深度，20 局小样本仍明显落后，搜索分支选择及静态评估差异也需继续查。当前 PikafishNet 每次从局面重新提取双视角特征和计算 NNUE，搜索规则与置换表还处理完整规则历史；增量 NNUE、着法排序和选择性剪枝是优先改进点。新结构候选网络的自博弈搜索共用该 AB 路径，在达到可接受的对照质量前，不应把仅由浅层自博弈产生的评分样本当作已验证的强监督数据。

## 源码对照与首轮优化

本地参考源码为 Pikafish `b562d6a`（2026-09-23）；9/25 二进制来源未知，不能认为两者完全相同。公开源码 `search.cpp` 在主搜索展开走子前使用 razoring、反向 futility、空着、ProbCut，静态搜索还有置换表、吃子 SEE 和 futility 裁剪。ChineseAI 此前在 TT 后几乎直接生成全部合法着，静态搜索也展开大多数吃子。官方走子同时维护 NNUE 脏特征和累加栈；本项目目前仍在每次评估时重建双视角威胁特征，仅权重累加实现局面间差量。

首轮加入量化 NNUE 累加缓存、保守静态搜索吃子裁剪，并跳过 PikafishNet 路径的旧模型 hidden/bucket 空操作。初始局面 depth 5：29,616 节点／270 ms → 20,688 节点／143 ms，最佳着未变；500 ms 搜索从约 53,294 节点提高至 71,373 节点，但仍只完成 depth 5。5 个开局固定 depth 5 的最佳着均与旧版一致。新旧 ChineseAI 同权重 5 个配对开局共 10 局，新版 2 胜 2 负 6 和；新版对 Pikafish 同样 10 局仍为 0 胜 10 负。结果分别见 [新旧对局](../target/fast/new-vs-baseline-5openings.tsv) 与 [Pikafish 对局](../target/fast/new-vs-pikafish-5openings.tsv)（本地构建产物，未纳入版本库）。
