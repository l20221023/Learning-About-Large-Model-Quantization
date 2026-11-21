# Q1 / M3 归档同步与项目状态核对

时间：2026-11-21（Asia/Shanghai）
代码基准：`c21d649e659c080aae3b2daefbab8e673e55825c`。本报告是本地核对记录，不代表远端同步。

## 状态与边界

- Q1：固定子集工程评测完成，人工验收待办。`task_status/Q1.md` 已更新；原 Q1 人工验收状态仍需 O1 审查。GSM8K 31/256、C-Eval 135/256（5 个解析失败按总数计）、MMLU 131/256；API 失败、输出截断、上下文溢出均为 0。
- M3 固定工作集 TP=2 Trace、ON/OFF parity、开销对照和独立审计：COMPLETED。TP 路由差异分析的因果归因：PARTIAL。独立归档审计 `stage_artifacts/P2-M3/archive_audit_run.log` 为 PASS；历史 125 项原 manifest 与 9 个源码快照哈希验证通过。
- O1：PARTIAL，既有独立审计通过但预设数值门槛为 `NUMERIC_FAIL`。本任务不改 `project_status.md`，由 O1 整合。
- O2：按最新 `task_status/O2.md`，工作流 PARTIAL、Task B FAIL、Task C WAITING；顺序相关的前一行输出错配尚未定位到内部算子/生命周期根因。最新 GPU 因果验证因跳板 SSH 严格主机密钥后认证失败而未获准入；未建立目标会话、未启动新 GPU 作业，当前资源状态未知。O2 线程仍 active；本任务使用隔离 worktree/分支，不与其并发写共享索引。

## 哈希与文件核对

- Q1 样本 SHA256：`814fcf6fed16408a284ed478adce700795e774131d926c1ca3a286d6e064c027`。
- Q1 数据集 manifest SHA256：`920806926233534da97ca0a1970ac970a4ff0baa0cb65a8b91011ed50537f9f6`。
- Q1 正式 raw SHA256：`e1dbb7d95a4dee36da28b8942f65e605aff4752b375071b086187fba6a65bebb`。
- M3 `independent_audit.json` 与 `quality_trace.jsonl` 保持归档哈希；本地隔离复核从原始 API 响应及 `.npy` 重算，审计 PASS，8,138 条规范化 Trace 对象顺序和内容与归档一致。未覆盖任何原文件。
- 归档摘要：`stage_artifacts/P2-M3/SHA256SUMS.txt`（125 项）；`closeout_SHA256SUMS.txt`（195 项总目录补充清单）。

## Git 同步

- 共享工作区 `main` HEAD `c21d649e659c080aae3b2daefbab8e673e55825c`；索引为空。仅本任务修改 `task_status/Q1.md` 并新增本报告；`task_status/M3.md` 及 M1/M3 实验目录当前为未跟踪文件。
- 工作区同时有 Phase 1、O1、O2 等并行修改，O2 线程仍 active。因此本任务改用隔离 worktree 和独立 `codex/` 分支提交，不触碰共享分支与索引。
- 首次 `git ls-remote origin refs/heads/main` 曾失败（exit 128，`Recv failure: Connection was reset`）；重试成功后确认远端 `main` 与本地基准同为 `c21d649e659c080aae3b2daefbab8e673e55825c`。隔离分支 `codex/q1-m3-archive-sync-20261121` 已提交并推送，推送后 `ls-remote` 返回 `95dd02e341fe602f57fc5a64a78d8d189efad04a`，与本地提交一致。Q1 状态文件上一次已提交 SHA：`9d885aca2e4fe530c4e0481a5473a7a2b0313615`，不代表本轮修改。

## AutoDL 磁盘只读快照与清理建议

最近成功 SSH 只读检查：2026-11-21 08:54（Asia/Shanghai），`/root/autodl-tmp` 60 GB 总量、46 GB 已用、15 GB 可用。一级目录主要占用：模型目录约 45 GB，experiments 452 MB，cache 42 MB，tmp 与 logs 各约 13 MB，P2-M3 部署目录约 12 MB，benchmarks 34 MB。

只建议后续检查 `tmp`、可再生成的日志副本和 cache 内容；先用清单/引用/哈希确认未被活跃服务或复现实验使用。冻结模型权重、B0/Q1/M3/O1/O2 原始数据、Trace、审计证据一律保留。本次没有删除或改写远端文件。

## 下一轮 GPU 准入

本任务不需要 GPU。下一轮 O1 数值补验或 O2 replay/cache 实验应分别申请任务窗口；运行前重新核验两卡占用、项目 GPU 锁、服务、数据盘余量和 active 线程。O1 预设数值门槛尚未通过；O2 动态 replay 的 63 行中 62 行不匹配，须先解决其数值 gate。不得借本次归档核对启动新的 GPU 工作。
