# LLM Quantization & MoE Optimization — Shared Project Status

> 统一状态入口：已作仓库文件与证据索引审查，不代替原始证据。
> 总状态整合者：**O1 主线任务**；Q1 只读本文件，更新 `task_status/Q1.md`。
> 审查快照：2026-11-20 17:01 起（Asia/Shanghai）；其他任务仍在执行，以下不是实时 GPU/服务状态。
> 本轮范围：管理文件合并、Git/聊天路径核验、既有证据阅读；未重跑基准或实验审计器。
> Reviewed by：管理初始化聊天 `01a1f82e-346b-7d82-9166-1851f5aa272e`（文件审查）；后续由 O1 继续复核维护。

## Objective

研究 DeepSeek-V2-Lite-Chat 的量化、BF16 专家卸载、MoE 路由、United Expert 替换及联合优化，保留因果可比性，分别陈述工程效用与研究新颖性。项目独立于 Stage D。

## Workstreams

| ID | Owner / status writer | Status | 已核对证据与边界 | 下一门槛 |
| --- | --- | --- | --- | --- |
| B0_TP2 | 已冻结基线；O1 维护索引 | COMPLETED（已有独立审计记录） | `benchmarks/deepseek_v2_lite/reports/baseline_report.md`；`benchmarks/deepseek_v2_lite/results/B0_TP2/audit.json` 的 passed=true、errors=[]。本轮只读现有审计，未独立重跑 | 保留原始数据；不默认重跑 |
| O1 | O1 主线；`task_status/O1.md` | PARTIAL | `reports/offloading_gpu_causal_validation.md`：V1/UVA20 单卡固定短窗退出 0，有限功能与因果验证；未冻结逐 token 数值容差，完整质量/长期稳定性未验收 | 明确数值容差及连续 decode、性能/稳定性验证；协调 O2/Q1 资源 |
| Q1 | Q1 质量任务；`task_status/Q1.md` | PARTIAL / NEEDS_REVIEW（记录滞后） | 本地已有 `stage_artifacts/P2-M1/dataset_manifest.json`、`quality_samples.jsonl` 768 行；manifest 列 GSM8K 1、MMLU 1、C-Eval 52 个文件。旧 STATUS 仍称缺数据；当前聊天报告已到 smoke/格式修正，正式结果未在本轮独立核验 | 对齐样本/配置/评分版本，交付 raw、结果审计和最新任务状态；不得沿用旧阻塞结论或声称正式完成 |
| O2 | 现有 OffloadGate 执行任务；结果交 O1 | PARTIAL | `reports/bf16_uva_decode_and_cache_analysis.md`：未插桩 9 请求先导；BF16 trace 有失败/重试，正式覆盖、parity 与缓存结论待核验。已有 AWQ decode trace 只证明 AWQ 路径 | 核查真实 BF16 trace 的一致性、覆盖与开销，再审查离线模拟；理论传输量不等于实测 |
| U1 | 待派发；未来 `task_status/U1.md` | NEEDS_REVIEW | M0-5 局部 expert shadow 和 Phase3 计划不等于 UE 训练/整模型替换完成；本轮未做实现盘点 | 先审查实现、教师/学生身份、替换成本与质量损失，获派后执行 |
| J1 | 待派发；未来 `task_status/J1.md` | WAITING | 本轮未确认联合策略实测，不从局部先导推断完成 | 依赖单项通过及可比对照后，由用户/总调度明确派发 |

表内实验产物在初始化时尚未纳入当前 HEAD；本管理提交只保存入口与证据路径，不声称远端已包含实验文件。路径存在不等于证据已完成独立复核。

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

## Repository / synchronization snapshot

- 本机 root：`E:/01-Projects/llm_quantization`；common Git dir：`E:/01-Projects/llm_quantization/.git`。
- 分支 `main`；初始化前 HEAD：`718e0eba631936c278f0883dd65e54efe720fa6f`。
- origin：`https://github.com/l20221023/Learning-About-Large-Model-Quantization.git`；本轮初次 `git ls-remote origin refs/heads/main` 核验为相同 SHA。
- worktree list 仅登记上述一个 worktree，没有已登记的另一分支工作树。
- 本机 O1 总调度聊天 `01a1ea71-2fbc-7a72-909f-2aa6e15a13db` 与 Q1 聊天 `01a1f7f6-6c92-7c43-98e7-f092894f3291` 的应用 cwd 和最近命令 cwd 均为此目录。已观察的本机 checkout 共享 main；未检查 AutoDL 其他 clone/分支，不声称云端已同步。
- 初始化提交用 `git log -1 --format=%H -- docs/project_management/INITIALIZATION.md` 查询，不预填自身 SHA。提交后的远端同步情况以交付时 HEAD 与 ls-remote 核对为准。
- 初查索引为空，无 unmerged paths；大量他人实验改动/未跟踪产物保留。文本证据差异不能通过覆盖消除。

## Open decisions / evidence discrepancies

- Q1 旧 STATUS/runtime 滞后于 manifest、样本和聊天进展；由 Q1 提交一致版本和审计，O1 整合。当前不指定正式准确率。
- O2 已有基线和 trace 尝试，不沿用原模板 WAITING；正式 BF16 trace/缓存完成状态仍待复核。
- 物理 GPU/服务窗口以锁和实时检查为准；本轮不宣称当前 GPU 空闲。
- UE 实作、J1 对照、远端实验提交保持待审查/等待，不借初始化升级成果。
- 三份原输入完整保存在 `docs/project_management/`；旧模板的负责人/路径占位信息已用本轮核验替换，原文仍可追溯。
