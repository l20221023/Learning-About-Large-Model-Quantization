# P2 执行计划

- M0-1 静态架构复核已由用户验收，原始报告见 `inputs/`。
- M0-2 自查 PASS，用户已批准进入 M0-3。固定 revision 的官方索引及全部四个 Header 共核对 5,291 个张量。单专家提取通过合成分片测试。详见 [M0-2 验收报告](../reports/M0-2_validation.md)。
- M0-3 自查 **PASS（社区 AWQ 实测路径）**。3090 上的真实短请求及 26 个 MoE 层的原生 Router Trace 已通过。固定 revision 的官方 BF16 实测仍待可运行环境。[3090 验收报告](../reports/M0-3_3090_AWQ_validation.md) · [本地 CPU 验证](../reports/M0-3_validation.md)
- M0-4 自查 **PASS（探索性社区 AWQ prefill 路径）**。6 条自编提示、3 类负载、169 个有效 token、4,394 条 Trace；选择数、共激活、Shared hook 和插桩前后 logits 不变量通过。[M0-4 自查报告](../reports/M0-4_validation.md)
- AutoDL 存储审计与容量保护已完成；官方 BF16 下载和 BF16→量化共存转换目前受 10 GiB 数据盘保留线限制。[存储审计](../reports/storage_audit_autodl_20261118.md)
- 用户要求后续实验连续逐个验证并在最后统一验收；阶段自查仍需逐项留证，但本轮不在每个阶段暂停等验收。当前继续 M0-5，保留 AWQ/BF16 证据边界和资源授权门禁。
- M0-5 自查 **BLOCKED（真实模型探针失败，shadow gate 未完成）**。首轮真实 AWQ 3090 加载后暴露 kwargs pre-hook 错误；hook 与 `MoERunner.forward` 双输入已本地修复，35 项本地测试通过。修复版未复跑；07:24 起云端 TCP Connection refused，远端失败日志未拉回。本地自查及接续项见 [M0-5 阶段审计](../reports/M0-5_validation.md) 和 [`stage_artifacts/M0-5/STATUS.md`](../stage_artifacts/M0-5/STATUS.md)。

## 总调度队列（2026-11-18）

阶段任务由总调度逐个派发、独立目录交付；总调度复核证据后再派发依赖阶段。仅总调度会话维护本文件、`DECISIONS.md` 与 `EVIDENCE_LEDGER.md`。用户要求所有阶段最后统一验收，因此不在阶段之间等待人工验收；单阶段退出门槛仍不可跳过。任何时候最多一个阶段会话占用同一张 3090。

| 顺序 | 会话 | 依赖 | 资源 | 状态 / 当前门槛 |
|---|---|---|---|---|
| 1 | M0-5 AWQ Expert Shadow | M0-2/3/4 自查产物 | 现有 AWQ、单张 3090、现有 vLLM 0.26.0、无下载；保留系统盘 8 GiB / 数据盘 10 GiB | **BLOCKED**。3090 首轮单提示探针真实加载 AWQ，但 pre-hook 报 kwargs capture 错误；本地已修复 hook 和 `MoERunner.forward` 双参并通过 35 tests。修复后未复跑；07:24 后 SSH Connection refused。没有相似度数据。 |
| 2 | Phase 1 量化基线 | M0-5 总调度复核 | CPU 准备；若运行 GPU，仅现有 3090、现存 AWQ | 已建立待派发会话；禁止官方 BF16 下载/大权重量化，除非空间峰值审计和授权通过。 |
| 3 | Phase 2 低成本 MoE 对照 | Phase 1 复核 | CPU 准备；3090 串行 | 已建立待派发会话；先单层与低成本对照。 |
| 4 | Phase 3 United Expert 蒸馏 | Phase 2 复核并选定结构方法 | 低成本 CPU 准备；3090 串行 | 已建立待派发会话；K=2/4 单层开始，禁止扩展到未授权长训练。 |
| 5 | Phase 4 结构 × 量化 | Phase 1–3 复核 | 固定现有设备与权重；空间保留线 | 已建立待派发会话；缺失对照格必须标 BLOCKED，不跨设备推算加速。 |
| 6 | Phase 5 Kernel / Roofline | 依赖阶段复核、可用 CUDA 工具链 | 仅现有 3090 | 已建立待派发会话；RMSNorm、INT4 reference、Triton GEMV 与真实 profiler 数据。 |
| 7 | Phase 6 最终交付 | Phase 1–5 均已交付或明确 BLOCKED | CPU 汇总；远端 Git 状态可核实时推送 | 已建立待派发会话；重算原始数据并形成最终复现报告。 |

### 当前环境与 Git 核对

- 2026-11-18 00:58–01:05 Asia/Shanghai 交互登录确认 AutoDL 主机 `autodl-container-5fe44bab80-7925a7e9`；Ubuntu 22.04.5，单张 RTX 3090 24 GiB，登录时 0 MiB 显存占用、0% 利用率。系统盘可用约 32.00 GB，数据盘约 37.56 GB，均满足保留线。模型快照约 8.5 GiB，隔离 venv 与云端证据仍在数据盘。无明确实例剩余倒计时命令；用户此前估计约 5–6 小时。首轮 probe 真实加载 AWQ 后在 pre-hook kwargs 捕获失败；修复尚未复跑。
- 2026-11-18 07:24–07:25 复查 SSH 36905 返回 TCP Connection refused，原连接被远端关闭；数据盘失败日志尚未拉回。该时点已接近用户预估可用窗口末端。继续需要同一实例恢复/延长或用户授权其他可用时长；未启动新 GPU。
- 已保存的 `[connect.nmb2.seetacloud.com]:36905` 主机密钥条目存在，严格校验登录未出现变更警告。只使用了交互密码认证并禁用公钥；凭据未写入命令、环境或文件。
- 本地 HEAD 为 `def624fd53e18a4d230bd2ec7f592e23a2569b7b`，`main` 工作区原先干净。`git ls-remote origin main` 在本次未返回远端 ref，故远端 SHA 仍未知；不宣称已发布。首个本地提交不重做。
