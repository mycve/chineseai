# 策略校准

新模型默认隐藏层宽度为 96。显式指定宽度及已有模型的宽度不受默认值影响。

```powershell
cargo run --profile fast --offline --bin chineseai -- az-init
cargo run --profile fast --offline --bin chineseai -- az-calibrate-policy --model model.safetensors --train data/data.bin --output calibrated.safetensors
```

校准冻结网络其余权重，只拟合已有的 448 项棋种与战术签名策略因子，并更新推理缓存。输出沿用现有模型格式，不增加参数或推理运算，价值输出保持不变。

默认使用 CPU AdamW，学习率及权重衰减均为 0.01，batch 至多 256，更新 1000 次，种子 17。`--steps`、`--seed` 可修改更新次数和种子；`--games` 默认 512，限制加载的完整训练对局数。训练 TAR 中名称散列属于固定留出组的对局自动跳过，校准不会读取留出样本。

输出文件必须不存在。命令显示校准前后的训练集 CE 及折叠误差；训练集 CE 不能代替独立质量或对弈评估。校准改变策略分布和搜索树，实际 NPS 仍需测量。

后续训练从输出模型开始；校准不生成优化器快照。
