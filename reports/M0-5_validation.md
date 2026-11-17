# M0-5 同输入 Expert Output Similarity 自查报告

**状态：BLOCKED（3090 实测尚未执行）。** 本阶段实现、离线静态检查和数值 helper 测试已完成；不得将合成单测解释为模型输出相似度结果。阻塞原因是当前 Codex 主机到已授权 AutoDL SSH 地址在 TCP 阶段被拒绝，尚未进入 SSH 认证。

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
| `C:\Users\li\.conda\envs\pytorch_clean\python.exe -B -m unittest discover -s tests -p 'test_m0_*.py'` | PASS，32 tests |
| `python -B -m unittest discover ...`（Windows Store shim） | 启动失败；随后改用明确的已安装 conda Python |
| `E:\MiniConda\python.exe ...` | M0-5 导入失败（没有 PyTorch）；未当作测试结果，随后切换到报告指定环境 |
| `ssh -p 36905 ... root@connect.nmb2.seetacloud.com` | TCP `Permission denied`，不能连接 |
| SSH `-vv` 短超时复查 | 解析为 `1.27.207.179`，TCP 连接仍 `Permission denied`；未进入用户认证 |

没有执行 GPU profiler，也没有生成真实 `expert_similarity.json`。没有远程日志、显存或实测输出可报告。此前 M0-4 结果和云端权重不受此阻塞影响。

## 待恢复连接后的复现命令

在 AutoDL 已连接且存储 preflight 通过后，将此 checkout 的三个新增 runner/helper 文件及 workload 拷入现有工作目录（保留云端既有权重和 venv），然后运行：

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

1. 当前无法读取目标云机上的现存文件和 GPU 状态；M0-4 审计时的空间余量不是本次当前容量观测。runner 自身仍会在远端启动前做最新 preflight。
2. 未验证 vLLM 0.26.0 的实际 MoE router method 绑定方式、候选 forced ID 张量布局、kernel 对非目标 filler experts 的零权处理，也未验证 shadow 重构阈值 `atol=rtol=2e-3` 是否足以且严格。实测如失败必须保留日志并修复后重跑。
3. M0-4 高频/低频计数基于每类仅两条自编提示，不能视作线上频率。两条 prompt 与首尾 token 只构成小型探索样本。
4. 未测 BF16、专家删除/合并、端到端质量或性能，也未建立生成任务的统计置信区间。

恢复连接后先执行受控部署、API 小探针和完整 shadow reconstruction gate；只有实测和归档完成才可把 M0-5 改为 PASS。按用户要求，后续阶段继续逐项执行，最后统一验收。
