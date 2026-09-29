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
