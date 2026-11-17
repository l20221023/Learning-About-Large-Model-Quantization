# P2 证据台账

沿用 `[C]` 配置、`[S]` 源码、`[I]` 索引、`[D]` 推导、`[H]` 实际 Header、`[R]` 项目实测、`[E]` 他人论文、`[U]` 未知。

| 问题 | 当前证据 | 验证安排 |
|---|---|---|
| 27 层、Dense 层 0、MoE 层 1–26、64 Routed、greedy Top-6 和 shared 宽度 2816 | [C/S] M0-1 已验收 | M0-2 核对张量映射，后续验证真实执行 |
| 固定版本各权重的分片、shape、dtype、offset | [I/D/H/R] M0-2 实读四个官方 Header，5,291 个张量全部匹配，均为 BF16；偏移范围合计 31,412,968,448 字节 | [映射表](../experiments/results/M0-2-20261117T0926335375-official/audit/expert_weight_mapping.csv) · [请求与 Header 记录](../experiments/results/M0-2-20261117T0926335375-official/audit/manifest.json) |
| 真实 checkpoint 的单专家数据读取、专家张量是否数值相同 | [U] | 已有本地 shard 时按偏移读取，数值对照留待后续 |
| HF 原生 BF16 eager loader、MLA 模块、MoEGate 及 Top-6 Trace | [S/R] M0-3 官方源码 + 随机 BF16 的三分片合成 checkpoint 通过，15 条记录，插桩前后完整 logits 最大差异 0。26 个路由层的对齐另以合成单元测试验证。 | [M0-3 验收](../reports/M0-3_validation.md) |
| 官方 tokenizer 和 Chat Template | [R] 固定文件生成 25 个 token，过滤后 24 个普通 token | [Manifest](../experiments/results/M0-3-20261117T0951529190-native-final/trace/manifest.json) |
| RTX 3090 上社区 AWQ checkpoint 的预训练模型加载和 Trace | [R] vLLM 0.26.0，Marlin WNA16 MoE，624 条原生路由记录，插桩前后 logits 最大差异 0。显存 250 ms 采样峰值 10,595 MiB。 | [3090 验收报告](../reports/M0-3_3090_AWQ_validation.md) |
| 路由频率、负载、共激活和 Shared 调用 | [R] 社区 AWQ、三类自编小负载、6 prompts、169 个有效 token；26 层全部通过每层有效 token×6 与共激活上三角 token×15 不变量。Shared 模块 hook 156/156 次完整覆盖。探索性数据，不代表真实线上分布。 | [M0-4 自查](../reports/M0-4_validation.md) · [原始 trace 与聚合](../experiments/results/M0-4-20261117T235424-3090/) |
| 官方 BF16 完整预训练模型 Trace、GPTQ、量化质量及长上下文／并发性能 | [U] AWQ 实测覆盖短请求 prefill，暂未验证原版 BF16 或 GPTQ。 | [兼容矩阵](../reports/compatibility_matrix.md) |
| 同输入 Expert 输出相似度、加权贡献与组间差异 | [U] M0-5 profiler 尚无真实模型运行；本地数值 helper 与结果格式单测不构成模型测量 | [M0-5 自查与阻塞记录](../reports/M0-5_validation.md) |
| AutoDL 空间与缓存路径 | [R] `/` 30G free 约30G；`/root/autodl-tmp` 50G free 35G；安全保留线分别8/10 GiB。所有适用缓存变量已指向数据盘。完整官方BF16下载及转换峰值被安全预检拒绝。 | [存储审计报告](../reports/storage_audit_autodl_20261118.md) |
| SERE 公开实验的精确 checkpoint 和环境，Brownout Lite Chat 的直接实验 | [E/U] M0-1 | Phase 2 / Phase 3 核实 |
| 生成质量、校准后性能、显存、上下文长度和部署收益 | [U] | 后续基线及受控实验 |
