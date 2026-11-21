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

- Main push：git push origin HEAD:refs/heads/main 退出 0，Git 输出 c21d649..93680f6 HEAD -> main。push 后 git ls-remote origin refs/heads/main 返回 93680f6ec651a68e9011cef3ecfa11ec3cb4629d；GitHub API branch readback 返回相同 SHA。该 SHA 是本轮第一次完整主线状态/报告同步提交。

- 本次同步报告和状态页再补入上述验证记录；该轻量审计更新在下一个提交后同样推送并单独回读。

## 最终独立回读（2026-11-21）

- 在干净 worktree `E:/01-Projects/llm_quantization-q1-m3-finalize` 读取 `origin/main` 与目标归档分支。`git fetch origin main codex/q1-m3-archive-sync-20261121` 本次退出 0；`origin/main=4d0d6ebca0b34e9b0cf11de53b316a75d7d1ff48`，归档分支仍为 `abed4b01953c941efd82b750906c1e0137124912`。`git merge-base --is-ancestor origin/codex/q1-m3-archive-sync-20261121 origin/main` 退出 0；目标提交已完整包含在 main，无需再次合并。
- `git ls-remote origin refs/heads/main refs/heads/codex/q1-m3-archive-sync-20261121` 回读与 fetch 后 refs 一致。集成采用 fast-forward，没有双亲 merge commit；归档提交 `abed4b01953c941efd82b750906c1e0137124912` 是已并入的分支头，随后状态整合提交 `f5c07a58a5283aff1524425797a7baff94f4d00d`，首次同步提交 `93680f6ec651a68e9011cef3ecfa11ec3cb4629d`，后续验证提交为当前 `4d0d6ebca0b34e9b0cf11de53b316a75d7d1ff48`。
- 相对并入前 `c21d649e659c080aae3b2daefbab8e673e55825c`，当前 main 有 41 个变更路径；归档分支自身含 39 个 Q1/M3 归档路径。`git diff --check c21d649..origin/main` 退出 0；归档路径未包含模型权重、Parquet/NPY/NPZ、PT/PTH 或其他大型原始结果。B0/Q1/M3 冻结输出没有重跑或修改。
- 当前共享 checkout 有并行的未提交改动。只读 SHA 比对发现已并入路径中，`project_status.md`、`task_status/Q1.md`、`reports/q1_m3_archive_sync_20261121.md`、`stage_artifacts/P2-M1/quality_report.md` 与 main 版本不同；`reports/q1_m3_mainline_merge_20261121.md` 在该 checkout 不存在。独立 worktree 的本次修改不触及这些工作副本，原 checkout 保持未改。
- 并入状态保持边界：Q1 固定子集工程基线完成、人工验收 WAITING；M3 核心 Trace/Parity/开销工作集完成、TP 路由根因 PARTIAL；O1 仍 PARTIAL / NUMERIC_FAIL；O2 的 main 状态快照仍是 PARTIAL、Task B FAIL、Task C WAITING。`task_status/O2.md` 不在该归档提交中；本机后续未提交 O2 记录仍为整体 PARTIAL、等待人工验收，本次没有把它并入或改变 O1/O2 状态。
- 原始数据备份补充核查：另一个本地审计分支 `codex/q1-acceptance-backup-20261121` 的报告 `reports/data_backup_audit_20261121/backup_verification_report.md` 记录当次从 AutoDL 工作区恢复 8 个关键对象、共 18,903,791 bytes，8/8 SHA 匹配；目录清单仍有本地/远端路径差异。`/root/autodl-tmp` 是计算实例工作区，未证明独立持久保留，因此长期备份状态仍为 `BACKUP_PENDING`。SHA 清单和这 8 个恢复样本均不等于完整灾备。
- 本次复核命令：`git fetch origin main codex/q1-m3-archive-sync-20261121`；`git merge-base --is-ancestor origin/codex/q1-m3-archive-sync-20261121 origin/main`；`git diff --name-status c21d649..origin/main`；`git diff --check c21d649..origin/main`；`git ls-remote origin refs/heads/main refs/heads/codex/q1-m3-archive-sync-20261121`。远端 main 已在本轮复核前完成推送；复核没有重跑实验或 GPU 操作。
