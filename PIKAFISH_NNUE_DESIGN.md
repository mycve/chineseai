# Pikafish NNUE 权重结构与训练接线

网络结构基准源码固定为 `official-pikafish/Pikafish@b562d6aeac5401879e973dc53ddb56053f07bb6a`。本地 `tools/pikafish.exe` 现自报 `Pikafish 2026-09-25`，公开仓库没有可核实的对应发布或提交；它作为黑箱实测基准，搜索静态评估实现以已公开的较新源码为依据。旧 9 月 6 日发布版公式仅保留在独立 API 中。这里记录训练态浮点张量的**目标形状**；导出的量化 `.nnue` 还需要遵守官方的版本、结构哈希、排列和压缩编码。

| 部件 | 官方推理结构 | 训练态目标 |
| --- | --- | --- |
| HalfKAv2_hm | 6 王桶 × 4 攻击桶 × 689 棋子位置 = 16,536 输入 | `psq_ft[16536, 1024]` |
| FullThreats | 45,547 输入 | `threat_ft[45547, 1024]` |
| 特征变换器偏置 | 每视角 1,024 个 `i16` | `ft_bias[1024]` |
| PSQT | 16 桶，棋子与威胁各一套 `i32` | `psq_psqt[16536, 16]`、`threat_psqt[45547, 16]` |
| 后续网络 | 16 个材料桶，每桶 1024→32、64→32、128→1，并含 FC0 两个输出之差的跳连 | 每桶独立的 `fc0`、`fc1`、`fc2` 权重与偏置 |

每个视角分别累加棋子位置与威胁特征。官方将各视角的 1,024 个累加值两两组合为 512 个激活，按行棋方、对方顺序拼成 1,024 输入。材料分桶选择 16 组后续层中的一组，另加双方 PSQT 差。训练应使用可导的浮点模拟，保存训练权重与优化器状态；导出 `.nnue` 时再按官方的 `i8/i16/i32` 量化规则转换，并用 Pikafish 加载、逐局面比较分数验证。

## 接线顺序和验收

1. HalfKAv2_hm 编号、镜像及攻击桶已有实现和局面测试；已通过多个局面的最终量化网络值对照，仍需与官方引擎导出的逐局面索引比对。
2. FullThreats 45,547 编号和占位攻击关系已有实现及测试；已通过多个局面的最终量化网络值对照，仍需与官方逐局面索引比对。
3. 训练态双视角前向及反向、16 个材料桶、PSQT 和跳连已有浮点实现与梯度测试；仍需比较逐局面训练态与量化态推理。
4. 新模型已接入共用 AB 搜索、自有权重双边自博弈、非零损失更新和恢复续训。`pikafish-evolve`（别名入口 `ab-evolve --pikafish`）以有界内存队列连接并行 CPU 自博弈和 GPU 训练，独立服务并行测评候选快照；训练使用手写 CPU/CUDA 批量稀疏累加与梯度算子，全连接层按材料桶合批；搜索使用普通只读数组权重及私有增量累加器，每 32 次评估刷新，换边时交换视角，叶节点不创建张量或获取共享张量锁。置换表固定哈希槽并在线程内复用稀疏条目及规则历史缓冲。冠军晋级、固定基准、回放和待完成测评均持久化，CSV/HTML 记录并行峰值、训练期间搜索节点、数据等待和训练耗时。旧格式回放不混用。
5. ChineseAI UCI 已能读取用户提供的 Pikafish `.nnue`，初始局面及六个后续局面的原始 NNUE 整数值与官方引擎一致。自训练浮点权重的兼容 `.nnue` 导出及量化误差验证仍待完成；再进行晋级赛。

当前 `AbNnue` 仍是 1,260 输入、256 默认隐藏维、WDL 输出的独立模型，不能把它的更新称作 Pikafish 结构训练。新自动循环直接训练 `PikafishModel`，UCI 可识别并加载其浮点权重。`tools/pikafish.nnue` 通过 Pikafish 进程的 `EvalFile` 选项参与对照测试，不进入 ChineseAI 训练器。

本地权重是 Zstandard 压缩文件，网络版本为 `0x6A448AFA`。ChineseAI 已完整解析结构哈希、压缩权重与量化层。一局自博弈的 48 个局面中，新二进制对 44 个非被将军局面报告原始 NNUE 整数值，ChineseAI 与其 **44/44 一致**；其余 4 个局面 `eval` 输出 `none (in check)`。搜索分值采用内部单位；单 PV 根部已用窄窗 PVS，静态搜索有保守合法回吃 SEE。新二进制的最终评估换算与当前公开源码不同，ChineseAI UCI 只输出已验证的原始 NNUE 值。初始局面 depth 1：ChineseAI 111 分、45 节点、`h2e2`，新二进制 5 分、53 节点、`b2e2`；depth 2：150 分、392 节点、`h0g2`，对 165 分、108 节点、`b2e2`。这些差异表明搜索静态评估与剪枝尚未对齐，不能把权重前向数值一致当成剪枝一致。自训练权重仍需兼容量化导出与晋级验证。

源码：[网络层](https://github.com/official-pikafish/Pikafish/blob/b562d6aeac5401879e973dc53ddb56053f07bb6a/src/nnue/nnue_architecture.h)、[特征变换器](https://github.com/official-pikafish/Pikafish/blob/b562d6aeac5401879e973dc53ddb56053f07bb6a/src/nnue/nnue_feature_transformer.h)、[HalfKAv2_hm](https://github.com/official-pikafish/Pikafish/blob/b562d6aeac5401879e973dc53ddb56053f07bb6a/src/nnue/features/half_ka_v2_hm.cpp)、[FullThreats](https://github.com/official-pikafish/Pikafish/blob/b562d6aeac5401879e973dc53ddb56053f07bb6a/src/nnue/features/full_threats.cpp)。
