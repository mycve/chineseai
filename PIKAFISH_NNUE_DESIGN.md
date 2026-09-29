# Pikafish NNUE 权重结构与训练接线

基准源码固定为 `official-pikafish/Pikafish@b562d6aeac5401879e973dc53ddb56053f07bb6a`。不要根据不断变化的 `master` 直接推断文件兼容性。这里记录训练态浮点张量的**目标形状**；导出的量化 `.nnue` 还需要遵守官方的版本、结构哈希、排列和压缩编码。

| 部件 | 官方推理结构 | 训练态目标 |
| --- | --- | --- |
| HalfKAv2_hm | 6 王桶 × 4 攻击桶 × 689 棋子位置 = 16,536 输入 | `psq_ft[16536, 1024]` |
| FullThreats | 45,547 输入 | `threat_ft[45547, 1024]` |
| 特征变换器偏置 | 每视角 1,024 个 `i16` | `ft_bias[1024]` |
| PSQT | 16 桶，棋子与威胁各一套 `i32` | `psq_psqt[16536, 16]`、`threat_psqt[45547, 16]` |
| 后续网络 | 16 个材料桶，每桶 1024→32、64→32、128→1，并含 FC0 两个输出之差的跳连 | 每桶独立的 `fc0`、`fc1`、`fc2` 权重与偏置 |

每个视角分别累加棋子位置与威胁特征。官方将各视角的 1,024 个累加值两两组合为 512 个激活，按行棋方、对方顺序拼成 1,024 输入。材料分桶选择 16 组后续层中的一组，另加双方 PSQT 差。训练应使用可导的浮点模拟，保存训练权重与优化器状态；导出 `.nnue` 时再按官方的 `i8/i16/i32` 量化规则转换，并用 Pikafish 加载、逐局面比较分数验证。

## 接线顺序和验收

1. 精确实现 HalfKAv2_hm 编号、镜像及攻击桶，并与官方选定局面比对索引。
2. 精确实现 FullThreats 编号和攻击关系；与官方局面测试比对。仅有形状常量时不可启用新模型搜索。
3. 实现训练态双视角前向及反向，确认 16 个材料桶、PSQT、跳连均收到梯度；比较逐局面训练态与量化态推理。
4. 将 `ab-evolve` 的搜索评估、回放样本和优化器统一切到新模型，完成至少一次非零损失的自博弈更新及恢复续训。
5. 导出兼容 `.nnue`，让本地 Pikafish 加载后对同一局面给出与训练态在量化误差内一致的分数，再进行晋级赛。

当前 `AbNnue` 仍是 1,260 输入、256 默认隐藏维、WDL 输出的独立模型，不能把它的更新称作 Pikafish 结构训练。`tools/pikafish.nnue` 通过 Pikafish 进程的 `EvalFile` 选项参与对照测试，不进入 ChineseAI 训练器。

本地测试权重是 Zstandard 压缩文件。用户更新后，解压文件头的网络版本为 `0x6A448AFA`，与固定的最新源码版本一致；这只验证版本字段，还不能替代结构哈希、全部权重内容和逐局面输出检查。对照测试仍由 `tools/pikafish.exe` 加载该文件；最新结构训练从自己的浮点权重开始，后续导出时再做完整兼容性验证。

源码：[网络层](https://github.com/official-pikafish/Pikafish/blob/b562d6aeac5401879e973dc53ddb56053f07bb6a/src/nnue/nnue_architecture.h)、[特征变换器](https://github.com/official-pikafish/Pikafish/blob/b562d6aeac5401879e973dc53ddb56053f07bb6a/src/nnue/nnue_feature_transformer.h)、[HalfKAv2_hm](https://github.com/official-pikafish/Pikafish/blob/b562d6aeac5401879e973dc53ddb56053f07bb6a/src/nnue/features/half_ka_v2_hm.cpp)、[FullThreats](https://github.com/official-pikafish/Pikafish/blob/b562d6aeac5401879e973dc53ddb56053f07bb6a/src/nnue/features/full_threats.cpp)。
