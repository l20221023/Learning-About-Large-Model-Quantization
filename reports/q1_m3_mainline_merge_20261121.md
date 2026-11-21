# Q1/M3 并入项目主线核验记录

日期：2026-11-21（Asia/Shanghai）

## 结果

- 在 `E:/01-Projects/llm_quantization-mainline-merge` 独立 worktree 完成归档分支 `codex/q1-m3-archive-sync-20261121` 快进并入；基础 main `c21d649e659c080aae3b2daefbab8e673e55825c`，归档分支 `abed4b01953c941efd82b750906c1e0137124912`。无冲突；合并后更新 `project_status.md` 并保留 Q1 人工验收、M3 根因、O1/O2 门槛。
- GitHub API 独立确认远端 main 和归档分支当时分别指向上述 SHA；本机 Git 的 `git fetch origin main codex/q1-m3-archive-sync-20261121` 因 HTTPS 连接失败退出 1。API refs 与本地 refs 一致。直接 Git fetch/ push 通道受限；最终同步状态将在此记录补充。
- 入分支 39 个文件；`git diff --check c21d649..abed4b0` 退出 0。入分支没有 `.npy/.npz/.parquet/.safetensors/.bin/.pt/.pth` 等大型数据文件。源码快照和轻量索引属于证据交付。
- 对入分支 39 个路径与共享 checkout 做逐文件 SHA 比对：36 个现存文件与分支对象一致，3 个不同。差异为同步报告中的 O2 状态文字更新、`quality_report.md` 末尾仅多一空行、Q1 任务状态中“本轮提交”改成已提交到隔离分支的更新说明。独立 worktree 原本干净，没有覆盖共享工作区文件。

## 状态快照

- Q1：COMPLETED（固定 256 条子集工程基线），人工验收 WAITING。GSM8K 31/256，C-Eval 135/256（5 个解析失败计入分母），MMLU 131/256；不是完整官方评测。
- M3：核心固定工作集 Trace、ON/OFF parity、开销和独立审计 COMPLETED；TP 路由差异的因果根因 PARTIAL。
- O1：PARTIAL，预设数值门槛 NUMERIC_FAIL；9/9 性能请求成功不改变数值 gate 失败。
- O2：PARTIAL，Task B FAIL、Task C WAITING；63 条动态 replay 仅 1 条 bitwise exact，62 条错位/不匹配。因果验证受 SSH 认证失败阻塞；不将其描述为 GPU 测试失败。
- B0_TP2 冻结结果未修改；本次无 GPU 实验、无 Q1/M3 重跑。

## 大型原始数据留存与恢复性

- 本机共享 checkout 只读清点到：`stage_artifacts/P2-M1/data` 204 文件、5,044,958 bytes；`quality_raw` 9 文件、6,439,393 bytes；`stage_artifacts/P2-M3/remote-runs` 20 文件、4,505,439 bytes；`remote-evidence` 79 文件、171,342 bytes。原始文件仍在共享 checkout，未复制进此 merge worktree 或 Git。
- Q1/M3 有 SHA256/manifest 清单，可检查仍在本机的副本完整性；这些清单不是备份本身。现有材料提到 AutoDL 远端历史磁盘快照，但本轮未确认原始文件当前仍在远端、另一介质副本或可恢复下载路径。因此独立备份与可恢复性状态为 UNKNOWN / 未验证。没有删除原始数据。

## 审查和命令

- 远端 ref 查询：GitHub API `GET /repos/l20221023/Learning-About-Large-Model-Quantization/branches/main` 与对应归档分支 endpoint。
- 获取尝试：`git fetch origin main codex/q1-m3-archive-sync-20261121`，退出码 1（无法连接 `github.com:443`）。
- 关系：`git merge-base --is-ancestor origin/main origin/codex/q1-m3-archive-sync-20261121` 退出 0；`git diff --name-status origin/main...origin/codex/q1-m3-archive-sync-20261121` 39 条路径，均为 Q1/M3 归档新增/更新，不含 `project_status.md`。
- 合并：`git merge --ff-only origin/codex/q1-m3-archive-sync-20261121`，成功快进；`git diff --check` 退出 0。
- 审计限制：未在此 clean worktree 重跑 Q1/M3 实验或原始数据审计；使用已提交审计 JSON/报告和本机原始路径清点。O2 当前执行端状态文件未纳入 main；总状态只记录其现有失败门槛，不改写 O2 文件。

## 同步结果

- 待本次状态与报告提交及 main push 后补入：状态提交 SHA、最终 main SHA、远端验证命令/结果、精确状态和待处理数据备份事项。

- Push 前校验：待执行本地 Git push；若网络继续失败，将保留失败命令/退出码，不声称已同步。最终 main SHA 与远端 readback 结果在完成后另记。
