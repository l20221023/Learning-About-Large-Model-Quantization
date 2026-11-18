# M0-5 独立复核

2026-11-18，Asia/Shanghai。**结论 PASS，适用于现存社区 AWQ INT4 checkpoint、第 13 层、两条自编提示和单张 RTX 3090。** 本地读取原始 JSON、日志、GPU 采样和退出码，重新运行聚合器并核查数值及坐标。正式运行 `M0-5-full-20261118T075249-3090` 的 manifest 和退出码文件均为 0。

- 证据包 `experiments/results/M0-5-full-20261118T075249-3090/evidence.tar.gz` 的本地 SHA-256 为 `61d7662d8bd69dd332c16f5451948251f6f7e27ac982d36ac865dee94d35c745`，与执行端报告的哈希一致。正式 runner 的本地 SHA-256 为 `17b93d1646bd59aeda4c61c79855d9b749586d763d124b3efdb79836c2a6fd4a`。工作负载 SHA-256 为 `161af5b41da9294faa657620c05901657b618efdbc31a7f8e3130f284411bdec`，与 manifest 一致。
- 首轮 API probe 的 pre-hook 失败退出 1，第二轮缺少 forward context 退出 1，修复后的 API probe 退出 0。正式运行退出 0。各轮 `run.log`、`gpu.csv` 和 `exit` 均保存在证据包的 `logs/` 对应 run_id 下。
- Manifest 记录 vLLM 0.26.0、PyTorch 2.11.0+cu130、CUDA 13.0、RTX 3090、`AutoAWQMoEMethod` 和 `MarlinExperts`。模型 revision 为 `e9fdf80ec6f1d200a029c41004bee40993763b43`。
- 原始结果共 2 个样本、4 个采样 token、16 条候选专家记录及 24 条专家对记录。逐 token 检查原生 Top-6 ID 各不重复，权重有限且非负，候选 ID 为 2、6、25、63，记录中的入选标志和权重与原生路由一致。4 个 token 均记录 shadow 重构通过，最大相对 L2 为 `0.0002960364`。聚合器和独立核对脚本退出 0，记录见 `verify.log` 与 `analysis-verified/`。
- 逐专家对核对 cosine、范数、absolute L2、relative L2 和范数比的代数关系。聚合 CSV 保留 `expert_a`、`expert_b` 原始顺序，从而保持 relative L2 的分母方向一致。GPU 采样最大显存 10,595 MiB。执行端运行后记录 GPU 0 MiB、系统盘剩余 31,998,705,664 字节、数据盘剩余 37,564,604,416 字节。
- 最终代码通过 `py_compile` 和 35 项 M0 回归，退出 0。失败修复涉及关键字参数 hook、单专家调用的 router logits 与 forward context，以及共享专家合并前的 routed 输出观测点。

原始 JSON 只保存标量指标和 GPU 端的逐元素门槛判定，没有保存 2048 维输出向量，因此本地不能重新计算逐元素 `allclose`。本地复核确认记录覆盖、指标一致性及运行日志。阶段结果尚未扩展到官方 BF16、其他层或大样本质量评测。建议总调度先复核本审计及原始文件，再派发 Phase 1。
