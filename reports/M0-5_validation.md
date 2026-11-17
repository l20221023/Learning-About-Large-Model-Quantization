# M0-5 同输入 Expert Output Similarity 自查报告

**状态：BLOCKED（真实模型探针失败，最终 shadow gate 未完成）。** 2026-11-18 已在单张 RTX 3090 上实际加载社区 AWQ 模型并启动单提示探针；首次失败暴露 pre-hook 参数读取错误。随后修复 hook 与 `MoERunner.forward` 调用并通过本地 35 项测试，但修复版未能部署复跑：约 07:24 起 AutoDL SSH 端口拒绝连接。没有生成可审查的专家相似度数据；本地测试不等于模型实测通过。

## 目标与边界

依照 `docs/P2_MASTER_RUNBOOK.md` 的 M0-5：在同一 MoE 层、同一 token hidden state 上计算候选专家输出的 cosine、relative L2 和尺度；保留 native Router Top-6 ID/权重；用 native 入选专家的 shadow 输出重构原始 FusedMoE 输出作为正确性门槛；计算固定 Router 权重替换的局部误差。只用已下载的 `TechxGenus/DeepSeek-V2-Lite-Chat-AWQ` revision `e9fdf80ec6f1d200a029c41004bee40993763b43`，证据等级是 `MEASURED_COMMUNITY_AWQ_INT4`。不代表官方 BF16，也不表示可以无损合并专家。

工作负载 `workloads/m0_5_seed_prompts.json` 复用两条项目自编 M0-4 提示，在第 13 层取首尾非特殊 token。候选专家来自 M0-4 第 13 层路由统计：高频 6、63；低频 25、2。每个采样 token 每候选执行一次小批外的单向 shadow forward，合计预期 16 个 candidate-token forward；最多产生 4 份 2048 维 hidden 向量，不保存整段 hidden state、不下载权重。下载或转换的新增磁盘峰值预估为 1 GiB，runner 通过现有 8/10 GiB 保留线守护。

## 已完成实现

- `src/p2/experts/similarity.py`：输出 cosine、relative/absolute L2、范数与尺度比；native 路由输出加权重构；固定 Router 权重替换局部误差。
- `scripts/profile_awq_m0_5.py`：加载现有 vLLM 0.26.0 AWQ 快照；捕获第 13 层 FusedMoE 输入、native 路由权重与 routed 输出；强制候选路由执行 shadow forward；要求 shadow 重构通过后才导出实测记录。
- `scripts/run_m0_5_autodl.sh`：复用现有缓存环境、权重和隔离 venv；1 GiB preflight、磁盘监测及 250 ms GPU 采样。
- `scripts/aggregate_m0_5.py`：验证证据等级、有限数值、样本/token 专家坐标和 pair 完整覆盖，输出专家及 pair 汇总 CSV。
- `tests/test_m0_5.py`：helper 数值、路由加权重构、固定权重误差及聚合器缺失 pair 检查。

## 执行记录与测试

| 命令/检查 | 结果 |
|---|---|
| `C:\Users\li\.conda\envs\pytorch_clean\python.exe -B -m py_compile scripts/profile_awq_m0_5.py scripts/aggregate_m0_5.py tests/test_m0_5.py` | PASS |
| `C:\Users\li\.conda\envs\pytorch_clean\python.exe -B -m unittest discover -s tests -p 'test_m0_*.py'` | PASS，修复后 35 tests；日志见 [`stage_artifacts/M0-5/local-tests.log`](../stage_artifacts/M0-5/local-tests.log) |
| `python -B -m unittest discover ...`（Windows Store shim） | 启动失败；随后改用明确的已安装 conda Python |
| `E:\MiniConda\python.exe ...` | M0-5 导入失败（没有 PyTorch）；未当作测试结果，随后切换到报告指定环境 |
| 2026-11-18 00:58 与 01:05，禁用公钥、StrictHostKeyChecking=yes 的交互 SSH | 登录成功；核对保存主机密钥无变化，确认 RTX 3090、现存 AWQ 快照与 vLLM venv、磁盘余量；数据盘可用约 37.56 GB，系统盘约 32.00 GB。登录 banner 未提供剩余时长。 |
| 部署 M0-5 runner 与单候选探针 workload，远端 `sha256sum` 对部署文件校验 | 成功；首轮代码 SHA `bf368533...`，探针 workload SHA `7529d8e6...`。无权重下载。 |
| `bash /root/autodl-tmp/p2-deepseek-moe/autodl_storage_guard.sh run 1073741824 -- env ... python profile_awq_m0_5.py --workload workloads/api_probe_workload.json ...` | **失败，退出 1**。模型加载和 prefill 调用开始后，`register_forward_pre_hook` 只收到空 positional inputs，错误为 `ValueError: not enough values to unpack (expected 1, got 0)`。完整日志仍在远端 `/root/autodl-tmp/logs/M0-5-api-probe-20261118T010700-3090/run.log`，同目录 `gpu.csv`、`exit`；本次退出码文件为 `1`。原始数据尚未拉回本地，SHA 待实例恢复后计算。 |
| 修复 hook 为 `with_kwargs=True`，并核对/修复 `MoERunner.forward(hidden_states, router_logits, input_ids=None)` 调用 | 本地修复完成；当前 runner SHA `415172c373f735e96b86e5e17acb5b7670d60b32d84ede593bfa31cb84bdbc6f`。修复版未部署、未复跑。 |
| 修复后本地 py_compile 与 M0 回归 | PASS，35 tests；总调度独立复跑退出 0，命令/哈希见 [`local manifest`](../stage_artifacts/M0-5/local-manifest.json)。 |
| 2026-11-18 07:24–07:25 SSH 端口复查 | TCP Connection refused，原会话由远端关闭；已保存主机密钥条目仍在，但连接不可达。发生时间已到达用户预估的 5–6 小时窗口末端。未部署后续修复、未产生最终 shadow 输出。 |

首轮是真实模型集成探针，进入模型前向后因 hook 错误失败；它没有导出 `expert_similarity.json`，不能算 M0-5 shadow forward 成功。GPU 采样日志仍留在远端，未能拉回核查峰值或精确运行资源。没有生成最终专家/pair 相似度 JSON、聚合 CSV 或通过重构门槛的记录。云端原始失败日志路径和 runner 路径已记录，SHA 需访问恢复后补记。原始 AWQ 权重未改动；首轮可用磁盘余量仍高于 8/10 GiB 底线。

## 待恢复连接后的复现命令

在 AutoDL 实例恢复或用户授权继续使用后，重新核对主机密钥、运行时间、3090 独占和 8/10 GiB 磁盘底线；先拉回首轮原始失败日志，再把当前 checkout 的 runner/helper/workload 拷入现有目录（保留云端既有权重和 venv），随后运行：

```bash
bash /root/autodl-tmp/p2-deepseek-moe/run_m0_5_autodl.sh
```

结果目录为新建的 `/root/autodl-tmp/experiments/M0-5-<timestamp>-3090/raw/`；runner 不复用旧输出目录。成功后在本机对 `expert_similarity.json` 和 `expert_pair_similarity.json` 执行：

```powershell
$py = 'C:\Users\li\.conda\envs\pytorch_clean\python.exe'
& $py -B scripts/aggregate_m0_5.py <expert_similarity.json> <expert_pair_similarity.json> --out <new-analysis-dir>
& $py -B -m unittest discover -s tests -p 'test_m0_*.py'
```

## 未验证事项和限制

1. 首轮登录时已现场确认 GPU 与磁盘；当前端口不可达，无法确认实例是否关机或重新读取资源状态。远端失败日志和 GPU CSV 还未下载。
2. 通过本地源码审查了 vLLM 0.26.0 的 `MoERunner.forward` 参数；但修复版尚未部署，forced ID 布局、非目标 filler 的零权行为以及 `atol=rtol=2e-3` 的实际适用性均未经过 3090 核验。
3. M0-4 高频/低频计数基于每类仅两条自编提示，不能视作线上频率。两条 prompt 与首尾 token 只构成小型探索样本。
4. 未测 BF16、专家删除/合并、端到端质量或性能，也未建立生成任务的统计置信区间。

恢复实例访问后先归档首轮失败日志，运行修复版单提示探针和完整 shadow reconstruction gate；只有真实 3090 重构门槛、指标导出与原始数据复核通过，才可把 M0-5 改为 PASS。Phase 1–6 会话已建立为待依赖前序复核后派发；因 M0-5 当前 BLOCKED，不得启动依赖实验。用户需决定是否恢复/延长同一 3090 实例的可用时间；不自动启动新的付费 GPU 资源。
