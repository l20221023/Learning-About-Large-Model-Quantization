# P2 执行计划

- M0-1 静态架构复核已由用户验收，原始报告见 `inputs/`。
- M0-2 自查 PASS，用户已批准进入 M0-3。固定 revision 的官方索引及全部四个 Header 共核对 5,291 个张量。单专家提取通过合成分片测试。详见 [M0-2 验收报告](../reports/M0-2_validation.md)。
- M0-3 自查 **PASS（社区 AWQ 实测路径）**。3090 上的真实短请求及 26 个 MoE 层的原生 Router Trace 已通过。固定 revision 的官方 BF16 实测仍待可运行环境。[3090 验收报告](../reports/M0-3_3090_AWQ_validation.md) · [本地 CPU 验证](../reports/M0-3_validation.md)
- M0-4 自查 **PASS（探索性社区 AWQ prefill 路径）**。6 条自编提示、3 类负载、169 个有效 token、4,394 条 Trace；选择数、共激活、Shared hook 和插桩前后 logits 不变量通过。[M0-4 自查报告](../reports/M0-4_validation.md)
- AutoDL 存储审计与容量保护已完成；官方 BF16 下载和 BF16→量化共存转换目前受 10 GiB 数据盘保留线限制。[存储审计](../reports/storage_audit_autodl_20261118.md)
- 用户要求后续实验连续逐个验证并在最后统一验收；阶段自查仍需逐项留证，但本轮不在每个阶段暂停等验收。当前继续 M0-5，保留 AWQ/BF16 证据边界和资源授权门禁。
- M0-5 自查及独立复核 **PASS（社区 AWQ INT4 第 13 层小样本）**。正式 3090 运行 4/4 个 token 的 Top-6 shadow 重构门槛通过，16 条候选专家记录和 24 条专家对记录已归档。结果不代表官方 BF16 或整模型质量。[接续报告](../stage_artifacts/M0-5/RESUME_20261118.md) · [独立复核](../stage_artifacts/M0-5/FINAL_AUDIT.md)

## 总调度队列（2026-11-18）

阶段任务由总调度逐个派发、独立目录交付；总调度复核证据后再派发依赖阶段。仅总调度会话维护本文件、`DECISIONS.md` 与 `EVIDENCE_LEDGER.md`。用户要求所有阶段最后统一验收，因此不在阶段之间等待人工验收；单阶段退出门槛仍不可跳过。任何时候最多一个阶段会话占用同一张 3090。

| 顺序 | 会话 | 依赖 | 资源 | 状态 / 当前门槛 |
|---|---|---|---|---|
| 1 | M0-5 AWQ Expert Shadow | M0-2/3/4 自查产物 | 现有 AWQ、单张 3090、现有 vLLM 0.26.0、无下载；保留系统盘 8 GiB / 数据盘 10 GiB | **PASS（限定范围）**。正式运行 `M0-5-full-20261118T075249-3090` 退出 0；4/4 重构门槛通过；原始包与独立复核已归档。 |
| 2 | Phase 1 量化基线 | M0-5 总调度复核 | 现有 3090 串行 | **PARTIAL**。社区 AWQ 短请求先导和固定 WikiText-2 test Parquet PPL=13.40056 实测通过；BF16 对照、GPTQ 与扩展性能门槛未完成，见 [Phase1 验收](../reports/Phase1_validation.md)。 |
| 3 | Phase 2 低成本 MoE 对照 | Phase 1 复核 | CPU 准备；3090 串行 | **WAITING**。Runbook 规定 PARTIAL 阶段不可进入依赖阶段；Phase 1 完成前不派发 GPU 工作。 |
| 4 | Phase 3 United Expert 蒸馏 | Phase 2 复核并选定结构方法 | 低成本 CPU 准备；3090 串行 | 已建立待派发会话；K=2/4 单层开始，禁止扩展到未授权长训练。 |
| 5 | Phase 4 结构 × 量化 | Phase 1–3 复核 | 固定现有设备与权重；空间保留线 | 已建立待派发会话；缺失对照格必须标 BLOCKED，不跨设备推算加速。 |
| 6 | Phase 5 Kernel / Roofline | 依赖阶段复核、可用 CUDA 工具链 | 仅现有 3090 | 已建立待派发会话；RMSNorm、INT4 reference、Triton GEMV 与真实 profiler 数据。 |
| 7 | Phase 6 最终交付 | Phase 1–5 均已交付或明确 BLOCKED | CPU 汇总；远端 Git 状态可核实时推送 | 已建立待派发会话；重算原始数据并形成最终复现报告。 |

### 当前环境与 Git 核对

- 2026-11-18 08:34 起重新以 StrictHostKeyChecking=yes、禁用公钥的密码交互方式登录同一 AutoDL 实例；host key 未变化。登录时 RTX 3090 空闲，系统盘可用 31,998,705,664 B、数据盘 37,557,784,576 B；用户确认剩余 GPU 时长足够，AutoDL banner 未提供精确时长读数。Phase1 两轮 in-process decode 探针与一轮服务先导均由 3090 锁和存储 guard 保护；结束 GPU 0 MiB，磁盘余量仍高于 8/10 GiB 保留线。
- Phase1 首轮服务 `Phase1-AWQ-20261118T082549-3090` 退出 1（15 请求中 8 失败），保留但不计入成功数据。收敛 vLLM 服务限制后，`Phase1-AWQ-20261118T083931-3090` 退出 0、15/15 成功。9 次串行 batch=1 性能先导：TTFT P50/P95 0.1173/0.1198 s，E2E P50/P95 3.5436/3.5740 s，completion tokens/E2E 中位数 18.0608 token/s；250 ms 显存采样峰值 10,595 MiB。6 条自编输出未评分，不是质量评估。详见 [Phase1 报告](../reports/Phase1_validation.md) 与 [独立自查](../stage_artifacts/Phase1/SELF_CHECK.md)。
- Phase1 按 runbook 标 PARTIAL：现有 AWQ 的 WikiText-2 test Parquet PPL 已实测；数据许可/去重、BF16 对照、GPTQ 可行性/实作、扩展性能矩阵及 token 级 ITL 仍缺。数据盘已扩至 100 GiB，BF16 下载与转换仍须逐步峰值空间审计。Phase2 依赖 Phase1 复核；在 Phase1 退出条件满足前不派发依赖的 GPU 阶段。
- GPTQ 保留为独立的 Phase 1 可行性验证项：先做真实 BF16 局部专家量化和资源审计，完整转换仅在磁盘、显存和时长门槛通过后执行。AWQ 继续作为现有部署基线。见 [GPTQ 可行性计划](../stage_artifacts/Phase1/GPTQ_FEASIBILITY_PLAN.md)。用户计划后续增加一张 3090；在实际到位并验证分布式路径前，不计入当前资源。

- 2026-11-18 00:58–01:05 Asia/Shanghai 交互登录确认 AutoDL 主机 `autodl-container-5fe44bab80-7925a7e9`；Ubuntu 22.04.5，单张 RTX 3090 24 GiB，登录时 0 MiB 显存占用、0% 利用率。系统盘可用约 32.00 GB，数据盘约 37.56 GB，均满足保留线。模型快照约 8.5 GiB，隔离 venv 与云端证据仍在数据盘。无明确实例剩余倒计时命令；用户此前估计约 5–6 小时。首轮 probe 真实加载 AWQ 后在 pre-hook kwargs 捕获失败；修复尚未复跑。
- 2026-11-18 07:24–07:25 复查 SSH 36905 返回 TCP Connection refused，原连接被远端关闭；数据盘失败日志尚未拉回。该时点已接近用户预估可用窗口末端。继续需要同一实例恢复/延长或用户授权其他可用时长；未启动新 GPU。
- 已保存的 `[connect.nmb2.seetacloud.com]:36905` 主机密钥条目存在，严格校验登录未出现变更警告。只使用了交互密码认证并禁用公钥；凭据未写入命令、环境或文件。
- 本地首次基线为 `def624fd53e18a4d230bd2ec7f592e23a2569b7b`；本轮 M0-5 修复和 BLOCKED 证据已本地提交到 `2f3144a8829f77084819374dfcbe6eb28a54c414`，工作区干净。远端查询/push 本轮均未返回可确认的 ref 或成功响应，远端 SHA 与是否已发布仍未知；不 force push，也不宣称已发布。首次提交未重做。
## 2026-11-20 Offloading Feasibility Gate 补充

- Task A 的单卡 BF16 功能基线现已完成：官方固定 revision、vLLM 0.26.0 V1、RTX 3090 GPU0、TP=1、UVA 20 GiB，256-token 固定输入成功前向；连续 decode 的服务先导为 9/9 请求成功。历史 V2 NoopOffloader OOM 与 V1 实际 UVA 成功路径已分别留证。CPU eager/UVA 的 255 个目标 token 对齐，但 logprob 平均绝对差 0.051683、最大 0.507828；项目未事先冻结可接受数值容差，故 Task A 的质量/数值验收仍 **PARTIAL**，不能写成逐 token 精确通过。
- Task B **PASS（限定范围）**：固定 workload 下完成真实 BF16 连续 decode trace，6 条提示、每条 32 个输出 token、31 decode steps，26 层 Top-6，共 4,836 decode 路由行；插桩与未插桩输出一致。Task C **PASS（离线模拟范围）**：基于该 trace 独立复算 LRU/LFU/static-hot 共 18 组缓存结果。小样本 trace、冷缓存和理论传输字节不代表线上缓存延迟或 TPOT。
- 自查、原始日志、失败记录、独立审计与复现入口见 [Offloading Gate 报告](../reports/bf16_uva_decode_and_cache_analysis.md) 及 stage_artifacts/OffloadGate/BF16_decode_20261120T172330/。2026-11-20 17:44:53 退出后两卡 0 MiB、项目锁空闲；这是已记录时点快照，不代表之后的实时状态。
- Gate 整体维持 **PARTIAL**，原因仅为 Task A 数值验收边界未闭合。后续先依据现有 CPU/UVA 对照和项目质量要求冻结数值验收规则；在此之前，不把 B/C 的结果外推为动态 expert cache 收益，也不启动缓存管理器实现。
