# LLM Quantization & MoE Optimization — Shared Project Status

> 统一状态入口：已作仓库文件与证据索引审查，不代替原始证据。
> 总状态整合者：**O1 主线任务**；Q1 只读本文件，更新 `task_status/Q1.md`。
> 审查快照：2026-11-21（Asia/Shanghai）；其他任务仍在执行，以下不是实时 GPU/服务状态。
> 并入前 ref 快照：2026-11-21。远端 `main=c21d649e659c080aae3b2daefbab8e673e55825c`，归档分支 `codex/q1-m3-archive-sync-20261121=abed4b01953c941efd82b750906c1e0137124912`；当前 main 同步结果与回读证据见 `reports/q1_m3_mainline_merge_20261121.md`。
> 本轮范围：管理文件合并、Git/聊天路径核验、既有证据阅读；未重跑基准或实验审计器。
> Reviewed by：项目主线整合任务（2026-11-21）；Q1 人工验收与 M3 根因归因边界保留为独立门槛。

## Objective

研究 DeepSeek-V2-Lite-Chat 的量化、BF16 专家卸载、MoE 路由、United Expert 替换及联合优化，保留因果可比性，分别陈述工程效用与研究新颖性。项目独立于 Stage D。

## Workstreams

| ID | Owner / status writer | Status | 已核对证据与边界 | 下一门槛 |
| --- | --- | --- | --- | --- |
| B0_TP2 | 已冻结基线；O1 维护索引 | COMPLETED（已有独立审计记录） | `benchmarks/deepseek_v2_lite/reports/baseline_report.md`；`benchmarks/deepseek_v2_lite/results/B0_TP2/audit.json` 的 passed=true、errors=[]。本轮只读现有审计，未独立重跑 | 保留原始数据；不默认重跑 |
| O1 | O1 主线；`task_status/O1.md` | PARTIAL；预设数值门槛 NUMERIC_FAIL | `stage_artifacts/OffloadGate/O1_acceptance_20261120/independent_audit.json` 审计通过且 9/9 请求成功，但数值门槛未通过；长时稳定性及可归因 UVA PCIe 成本未闭合，详见 `stage_artifacts/OffloadGate/O1_acceptance_20261120/O1_GATE_STATUS.md` | 保持失败与未验收状态；补齐逐 token 数值、长时稳定性及可归因传输成本证据 |
| Q1 | Q1 质量任务；`task_status/Q1.md` | COMPLETED（固定子集工程基线）；人工验收 WAITING | GSM8K 31/256、C-Eval 135/256（含 5 个解析失败）、MMLU 131/256；固定 256 题子集，不是完整官方成绩。逐样本、raw、manifest 与独立 CPU 审计见 `stage_artifacts/P2-M1/` | 保留人工验收待办；后续质量比较须复用冻结样本/协议 |
| M3 | M3 Router Trace；`task_status/M3.md` | COMPLETED（核心固定集实测）；PARTIAL（TP 路由差异根因） | 6 个固定请求 Trace、ON/OFF parity、开销对照和独立审计通过；8,138 layer-token Trace 行、48,828 expert selections。TP 差异的因果归因未证明，详见 `stage_artifacts/P2-M3/` | 保持根因 PARTIAL；不把跨 TP 路由差异解读为实现错误 |
| O2 | OffloadGate；`task_status/O2.md`（当前执行端记录未进入 main） | PARTIAL；Task B FAIL，Task C WAITING | 最新本地 O2 状态记录动态 hidden replay 63 行仅 1 行 bitwise exact、62 行错位/不匹配；GPU 因果验证因 SSH 认证失败未建立目标会话。细节以 O2 原始报告与任务状态为准 | 定位 replay 顺序相关错位；恢复授权连接后先重新做资源门禁，不启动缓存实验 |
| U1 | 待派发；未来 `task_status/U1.md` | NEEDS_REVIEW | M0-5 局部 expert shadow 和 Phase3 计划不等于 UE 训练/整模型替换完成；本轮未做实现盘点 | 先审查实现、教师/学生身份、替换成本与质量损失，获派后执行 |
| J1 | 待派发；未来 `task_status/J1.md` | WAITING | 本轮未确认联合策略实测，不从局部先导推断完成 | 依赖单项通过及可比对照后，由用户/总调度明确派发 |

Q1/M3 轻量产物及任务状态已纳入本次主线候选提交；逐样本 raw、Parquet、M3 原始 Trace/运行数据仍不纳入 Git。当前本机共享 checkout 可见 Q1 data 204 文件约 5.04 MB、正式 quality_raw 9 文件约 6.44 MB、M3 remote-runs 20 文件约 4.51 MB。独立第二份备份和经验证的恢复路径尚未核实；SHA 清单只能验证完整性，不能证明备份存在。历史 AutoDL 磁盘快照不是当前可恢复性证明。

## Immutable Baseline Contract

- 原始参照：BF16、TP=2、2×RTX 3090、vLLM 0.26.0、max_model_len=2304、max_num_seqs=8。
- P01–P08，每格 3 次重复×100 正式请求，共 2400 条；300 条冻结 prompt，chat template 后输入长度 128/512/2048。
- `audit.json` 保存 raw_records=2685；包含预热及保留重试，不是正式请求数。报告说明统计使用每个重复最终序号 0–99。
- ITL=N/A；SSE 文本块不是逐 token 时间戳。请求级 TPOT、prefix caching 和采样限制见原报告。runner 退出码未保留，不补造成功退出码。
- 24 题启发式检查 18/24 不构成正式能力分数，不能代替 Q1。
- 不覆盖 B0 的 prompt、结果、审计或日志；后续服务启停不改变历史基线结果，须另存运行目录。

## P2 plan and acceptance

- `docs/PLAN.md` 保留阶段依赖和派发历史；现有记录仍将 Phase1 标为 PARTIAL、Phase2 为 WAITING。新 B0/O1/Q1 证据不自动升级整个 Phase1 或放行后续 Phase。
- M0-1 已由用户验收；后续自查 PASS 与用户验收区分。
- 原 AGENTS 的逐阶段人工验收规则保留。PLAN 中此前“最后统一验收”的明确授权只适用于其范围和既有派发；本次初始化不新增实验授权，不重启 runbook 历史起点 M0-2。
- PLAN/DECISIONS/EVIDENCE_LEDGER 仍由原总调度维护，本轮不修改。事实冲突由 O1 收集证据交适用所有者审查。

## Coordination and update protocol

1. 每次任务开始先读本文件，再读计划、任务状态与原始证据；核对 cwd、Git root/common-dir、branch、HEAD、remote、worktree 和未提交文件。
2. O1 是唯一日常总状态写入者；Q1 只更新 `task_status/Q1.md`。其他任务写各自文件；初始化引导记录标明来源，不冒充执行者自报。
3. 任务状态写明：目标/边界、状态/时间、模型与评测配置版本、代码 SHA、产物 SHA 或 UNCOMMITTED、原始日志/逐样本/审计路径、失败与未达门槛、资源占用/释放记录、下一步。
4. O1 检查证据、口径、代码/产物提交、独立检查和验收条件后整合，记录 reviewed-by/时间及未核验范围。单方自报不直接升级 COMPLETED。
5. 同目录聊天共享文件、分支和索引，不必相互 pull。仅按本任务路径暂存，Git 写操作串行，不切换其他聊天正在使用的分支。
6. 不同 clone/worktree 先检查脏文件，再 fetch、比较提交关系；干净且可快进时才 pull --ff-only，分叉经审查合并/PR，不强推、不硬重置。步骤见 `docs/project_management/INITIALIZATION.md`。
7. 实验前使用现有锁并核对 GPU/服务；TP=2 必须有两卡窗口。不停止别人的任务，不同时加载冲突模型。本轮未远程登录、推理、启停服务或启动 GPU 作业。
8. 大权重、缓存、大型日志留在指定磁盘；Git 保存轻量 manifest、哈希、摘要和位置。凭据不得进入状态、日志、脚本或提交。
9. 每项任务开始检查本地 `AGENTS.md` 与本文件；结束写任务 ID、验收状态、实测结论、未完成项、证据路径和更新时间。对已完成工作先核对证据，不为同步重复运行。
10. 发布前只暂存状态和必要的小型审计摘要，比较远端分支与本地历史并保护其他任务改动；远端 SHA 验证成功才记作已同步。只允许轻量摘要入 Git，权重、原始大型数据、密钥和敏感配置严禁上传。推送失败如实记录。

### 最近一次 BF16 Offloading / Trace / Cache 状态核对

- 更新：2026-11-20 20:00（Asia/Shanghai）；任务 ID：O1/O2，涉及 `OffloadGate-BF16-decode-20261120T172330`。代码证据提交：`f492352f098197d4a5ecebb92fa18ee8339a45f8`。
- O1：PARTIAL。V1/UVA20 单卡固定短窗功能/因果验证已记录，逐 token 容差与完整质量/长期稳定性仍未验收；详见 `reports/offloading_gpu_causal_validation.md`。
- O2 Task B：限定范围 PASS；6 个请求输出 parity，26 层 Top-6 trace，4836 decode rows；插桩额外开销 20.15%。保留 PyTorch build 导致的一个 softmax BF16 ULP 差异及未持久化逐层 backend inventory 限制。
- O2 Task C：离线缓存模拟与 18 个策略/容量单元独立复算一致，限定范围 PASS。六条提示与冷 cache 假设限制外推；传输字节是理论值，不是 PCIe 实测。
- 未完成：O1 数值门槛/稳定性；真实在线缓存延迟与通用请求覆盖；O2 runtime 局部证据与缓存模拟的人工验收。具体审查记录见 `task_status/O2.md` 与 `reports/o2_acceptance_and_u1_readiness.md`（状态文件由 O2 所有者维护）。
- 更新来源：本轮只读取刚提交的报告、O2 状态及审查摘要，没有重复实验或重跑审计。

## Repository / synchronization snapshot

- 本机 root：`E:/01-Projects/llm_quantization`；common Git dir：`E:/01-Projects/llm_quantization/.git`。
- 分支 `main`；初始化前 HEAD：`718e0eba631936c278f0883dd65e54efe720fa6f`。
- origin：`https://github.com/l20221023/Learning-About-Large-Model-Quantization.git`；本轮初次 `git ls-remote origin refs/heads/main` 核验为相同 SHA。
- worktree list 仅登记上述一个 worktree，没有已登记的另一分支工作树。
- 本机 O1 总调度聊天 `01a1ea71-2fbc-7a72-909f-2aa6e15a13db` 与 Q1 聊天 `01a1f7f6-6c92-7c43-98e7-f092894f3291` 的应用 cwd 和最近命令 cwd 均为此目录。已观察的本机 checkout 共享 main；未检查 AutoDL 其他 clone/分支，不声称云端已同步。
- 初始化提交用 `git log -1 --format=%H -- docs/project_management/INITIALIZATION.md` 查询，不预填自身 SHA。提交后的远端同步情况以交付时 HEAD 与 ls-remote 核对为准。
- 当前远端可通过 `origin` 查询，但 GitHub 网页匿名访问返回 HTTP 200；私有性与用户目标冲突。状态提交 `5475052` 尚未推送。在确认仓库可见性为私有或改用获授权的私有 remote 前，不发布任务/证据状态。
- 初查索引为空，无 unmerged paths；大量他人实验改动/未跟踪产物保留。文本证据差异不能通过覆盖消除。

## Open decisions / evidence discrepancies

- Q1 固定子集工程基线已完成并记录于 `task_status/Q1.md`；人工验收仍待办。准确率仅代表冻结的各 256 条样本。
- O2 保持 PARTIAL：动态 replay 数值门槛 FAIL，Task C WAITING。当前 main 未包含执行端 O2 任务状态文件；不覆盖或代替 O2 所有者的后续更新。
- 物理 GPU/服务窗口以锁和实时检查为准；本轮不宣称当前 GPU 空闲。
- UE 实作、J1 对照、远端实验提交保持待审查/等待，不借初始化升级成果。
- 三份原输入完整保存在 `docs/project_management/`；旧模板的负责人/路径占位信息已用本轮核验替换，原文仍可追溯。
