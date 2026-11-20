# 统一项目管理初始化与同步说明

日期：2026-11-20（Asia/Shanghai）。仅建立管理机制，不启动 GPU 实验，不重跑 B0，不修改权重，不操作模型服务。

## 输入、合并与所有权

- 主要框架：`C:/Users/li/Downloads/AGENTS-1.md`，原样保存在 `AGENTS-1.source.md`。只采纳与管理目标一致的规则，不将输入文档当成新实验派发。
- 原仓库 AGENTS 原样保存在 `AGENTS.original.md`；旧总状态模板原样保存在 `project_status.initial.md`。归档用于追溯，不自动执行其中命令。
- AGENTS-1 的读取顺序、任务隔离、B0 冻结、实验记录、指标边界、总状态写入权及完成定义全部整合；原 AGENTS 的 Stage D 隔离、M0-1 验收、PLAN/runbook、阶段验收及资源授权规则保留。
- O1 主线维护总状态，Q1 只读总状态、写 `task_status/Q1.md`。初始化可建引导记录，后续按所有权更新；原总调度维护 PLAN/DECISIONS/EVIDENCE_LEDGER。
- 逐阶段人工验收与 PLAN 的最后统一验收有适用范围差异：后者仅用于明确授权的既有派发。自查通过不等于用户验收，不借初始化重启旧阶段。
- 原模板 O2 WAITING 与已有先导/trace 尝试不符；Q1 旧“缺数据”落后于现有文件。新总状态记录差异与未核验范围，不新增质量或性能结论。

## 初始 Git 与双任务核验

- root/cwd：`E:/01-Projects/llm_quantization`；Git dir：`E:/01-Projects/llm_quantization/.git`；分支 `main`。
- 初始 HEAD、origin/main tracking ref 和 `git ls-remote origin refs/heads/main` 均为 `718e0eba631936c278f0883dd65e54efe720fa6f`。
- origin：`https://github.com/l20221023/Learning-About-Large-Model-Quantization.git`。
- worktree list 仅登记此工作树；初始索引为空，无 unmerged paths。
- 应用元数据及最近命令 cwd：O1 总调度 `01a1ea71-2fbc-7a72-909f-2aa6e15a13db`、Q1 `01a1f7f6-6c92-7c43-98e7-f092894f3291` 均为上述本机路径，观察到的 checkout 共用 main、工作区和索引。未登录 AutoDL 检查 Git，不声称云端 clone 已同步。
- 原有 7 个 tracked 改动和大量 untracked 实验产物不入本次管理提交。对这 7 个文件及 B0 结果/报告共 44 个文件建立临时 SHA256 快照，供前后检查。其他任务仍活动，外部变更须单独说明。

## 每次任务开始的只读检查

```powershell
Get-Content -LiteralPath ./project_status.md
git rev-parse --show-toplevel
git rev-parse --absolute-git-dir
git branch --show-current
git rev-parse HEAD
git status --short --branch
git worktree list --porcelain
git remote -v
git diff --cached --name-only
git diff --name-only --diff-filter=U
```

接着读取适用 PLAN、任务状态与证据。本轮未创建聊天、派发或发消息给其他任务，不改变其运行目标。运行中的任务不会因文件合并自动重载指令，下次开始任务时按新规则读取。

## 同步方法

同目录：文件修改和本地提交立即共享，不能为了同步两个 Codex 在脏目录强行 pull、checkout 或 reset。Git 写操作串行，提交前后检查索引归属。

不同 clone/worktree：核对 root/common-dir、remote、branch 和 HEAD；发送方仅提交本任务路径，获授权后推送对应分支。接收方保留自己的改动，仅在工作区干净时执行：

```powershell
git fetch origin
git log --oneline --left-right HEAD...origin/main
git merge-base --is-ancestor HEAD origin/main
# 仅上条返回 0 且本地修改已妥善保存时：
git pull --ff-only origin main
```

分叉或不同任务分支通过审查 merge/cherry-pick/PR 合入，不改写历史。云端需另查 root/remote/HEAD，复制脚本不等于 Git 同步。权重和大型输出不入 Git，不用 `git add .` 一并发布。

本地管理提交由两任务立即共享；本轮不执行远端 push。如需其他 clone 获取，由获授权维护者核对远端关系后执行 `git push origin main`，再以 `git ls-remote origin refs/heads/main` 验证，禁止 force push。远端仍为旧 SHA 时标为待推送，不称已发布。

## 验证与提交定位

- 核对 AGENTS-1 的有效规则及原 AGENTS 四项规则均保留，三份输入原文有完整备份。
- 只读 B0 报告和 passed=true 审计，未重跑基准/实验审计器；只读 Q1 的 768 行和 manifest 1/1/52 文件元数据，不当成正式评测完成。
- 提交前检查管理文件差异、引用路径、输入备份哈希、敏感内容、`git diff --check` 和保护文件哈希；只暂存管理文件。
- 初始化实际提交：`git log -1 --format=%H -- docs/project_management/INITIALIZATION.md`。文件不伪造其自身提交 SHA。
- 最终 HEAD、远端 SHA、保护检查和冲突结论在本聊天交付中报告。

### 本轮检查记录

- 44/44 个保护文件 SHA256 与编辑前一致；14/14 个关键引用路径存在。
- 三份输入归档与编辑前副本逐字节一致；AGENTS-1 输入 SHA256：`da9d0b1e210dc4bf9b0b2a5e4d7b4ef97012010822d29cf65efddd330e962b7b`。
- `git diff --check` 退出 0；Git 的 LF/CRLF 提示属于行尾转换提示，不是合并冲突。
- 原有 7 个 tracked 改动保持原样；其他未跟踪实验产物留在工作区，不暂存、不覆盖。
