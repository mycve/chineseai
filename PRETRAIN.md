# Px0 数据预训练

```sh
cargo run --locked --profile fast --example px0_pretrain -- data/data.bin
```

默认从整个归档均匀抽取 16,384 盘训练对局和 1,024 盘验证对局。整盘按文件名哈希分组，两组不会共享对局。使用验证组中的 8,192 个局面选优，最多训练 12 轮，连续三轮没有明显改善时提前停止。

结构采用项目默认的 hidden=128，batch=2048。训练调用与自博弈相同的 SGD Nesterov、全局步数学习率和梯度裁剪。标签是 Px0 `best_search_wdl` 和搜索策略分布；短期价值辅助损失关闭。

输出：

- `data/px0-pretrained.safetensors`：验证表现最好的网络。
- `data/px0-pretrained.safetensors.sgd.safetensors`：同一步的 SGD 状态。
- `data/px0-pretrained.latest.safetensors` 及其 SGD 文件：最后一轮网络。

检查网络与优化器状态一致性：

```sh
cargo run --locked --profile fast --example px0_pretrain -- --verify-only
```

## 服务器开始自博弈

使用最新代码，把最好的网络、同名 SGD 文件和仓库的 `book.pgn.gz` 一起复制到新的训练目录。创建新的 `px0-selfplay.toml`：

```toml
format_version = 30
model_path = "px0-pretrained.safetensors"
```

从该目录运行项目可执行文件：

```sh
/path/to/chineseai az-loop px0-selfplay.toml
```

其余参数由代码默认值提供。日志应显示恢复 SGD 动量和全局步数，learner update 从 1 开始。使用新配置和目录，避免混入旧 replay、progress、best 或优化器文件。优化器步数延续预训练；自博弈自行产生新的经验池。

验证指标只衡量对教师标签的拟合和泛化，不代表已获得对应棋力。自博弈后仍需固定开局、交换红黑进行对弈评估。
