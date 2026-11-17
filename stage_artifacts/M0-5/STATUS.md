# M0-5 阶段审计：BLOCKED（云端实例访问中止）

2026-11-18，Asia/Shanghai。本会话完成本地 runner 审查与修复。总调度在 3090 上执行了首轮极小 API 探针，真实模型成功加载，但探针在采集 hook 处失败；修复后的完整 shadow forward **未运行**，没有可复核的专家相似度结果。不能将 35 项本地测试算作 M0-5 实测通过。07:24 Asia/Shanghai 起，远端 SSH 端口拒绝连接，原会话被远端关闭。此时无法部署最终修复、复跑或拉回云端日志。

## 发现与修复

- 原 runner 仅对工作负载中的 4 个候选专家执行 shadow forward，却用它们重构原生 Top-6 路由输出；缺少其余原生专家时，正确性门槛必然失败。现按 `candidate_experts ∪ native_selected_ids` 执行 shadow forward，仅导出 4 个候选专家的比较指标。
- vLLM `LLM.apply_model` 返回每个 worker 的结果列表；现要求且解包单 worker 结果。
- 原 runner 的 forward hook 在执行 shadow forward 时仍然有效，会因二次调用触发异常；现于原生生成后立即移除 hook，恢复 router 方法，再启动逐专家前向。
- 固定权重替换现在以首个真实入选专家为来源，即使 4 个候选专家均未入选也能定义局部替换误差。
- 聚合器新增每个 token 的候选专家完整覆盖和 shadow 重构门槛检查。
- 已核对 vLLM 0.26.0 的 `MoERunner.forward(hidden_states, router_logits, input_ids=None)` 签名。原 shadow 调用漏传必需 `router_logits`；现传入当前 token 的原生 router logits，保留原设备与 dtype，并以关键字参数调用。强制路由 wrapper 仍提供候选专家 ID 和权重；合成签名合约测试已通过。源码：https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/vllm/model_executor/layers/fused_moe/runner/moe_runner.py
- 云端首轮 API probe 报 `ValueError: not enough values to unpack (expected 1, got 0)`：原生 FusedMoE 通过 `hidden_states` 关键字参数调用，原 pre-hook 只解包位置参数。现用 `with_kwargs=True` 注册 hook，分别支持位置参数和 `hidden_states` 关键字参数；合成 PyTorch hook 合约测试已通过。云端修复未复跑。

## 本地验证

完整命令、退出码和文件 SHA-256 见 `local-manifest.json`。`python -B -m py_compile` 退出 0；`python -B -m unittest discover -s tests -p 'test_m0_*.py'` 退出 0，35 项通过。测试输出见 `local-tests.log`。这些检查只覆盖 Python 语法、数值 helper 和合成数据，不验证 vLLM GPU API、AWQ kernel 或重构精度。

## 云端接续

SSH 保存的 `[connect.nmb2.seetacloud.com]:36905` 主机记录存在。总调度通过独立 PTY 交互登录，确认当时的 3090、权重、venv 和磁盘保留线，并将探针与 workload 部署到 `/root/autodl-tmp/p2-deepseek-moe/`。首轮真实模型探针经存储守护启动，**退出码 1**；远端原始文件位于 `/root/autodl-tmp/logs/M0-5-api-probe-20261118T010700-3090/`，含 `run.log`、`gpu.csv`、`exit`。这些文件尚未拉回本地，哈希和 GPU 峰值待实例恢复后核验；当前仅有总调度转述的错误文本，不能当作完整原始日志。07:24 后远端 TCP `Connection refused`，修复版 runner 未完成部署和 GPU 复跑。没有 `expert_similarity.json`、聚合 CSV 或通过重构门槛的采样记录。

## 接续门槛

实例恢复或用户授权新的可用时间后，在安全交互式密码登录的会话中，先拉回首轮探针失败日志与退出码，并重新核查主机密钥、GPU 独占、实例时长和磁盘 8/10 GiB 保留线。把 `scripts/profile_awq_m0_5.py`、`scripts/aggregate_m0_5.py`、`src/p2/experts/similarity.py`、现有 trace/audit 依赖、workload 与 runner 部署到独立目录；禁止下载权重。先通过极小 API 探针验证 `apply_model` 单 worker 返回、MoE router 方法和强制 ID/零权行为，再用新 run_id 执行 `scripts/run_m0_5_autodl.sh`。失败保留原始日志并修复复跑。最终归档原始 JSON、GPU 采样、完整命令/退出码、聚合 CSV、manifest 和哈希，复核每 token 的 6 个原生专家贡献与数值门槛。此门槛通过前不得把 M0-5 改为 PASS。
