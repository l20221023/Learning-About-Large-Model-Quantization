# Phase 1 首轮失败与最小修复探针

2026-11-18，Asia/Shanghai。总调度报告首轮 `Phase1-AWQ-20261118T082549-3090` 退出 1，远端原始目录 `/root/autodl-tmp/experiments/Phase1-AWQ-20261118T082549-3090`，请求 HTTP 500，结束后 GPU 0 MiB。完整 `server.log` 尚未回传本会话，因此这里的栈信息来自总调度转述，待原始日志独立核对。该轮没有有效性能或质量样本，不能计入指标。

> 截至 2026-11-18 08:41（Asia/Shanghai）的复核更新：首轮失败原始包已回传并归档，日志已独立检查；后续最小 decode probe 和修订配置下的 Phase 1 服务先导均通过。下方“尚未实测/未回传”描述仅指初始撰写时状态。最新数据、哈希和阶段结论见 `SELF_CHECK.md` 与 `../../reports/Phase1_validation.md`。

总调度补充的栈位于 vLLM 0.26.0 `vllm/model_executor/layers/attention/mla_attention.py:2223`：`kv_c_normed.to(self.kv_b_proj.weight.dtype)`，异常为 AWQ `ColumnParallelLinear` 无 `.weight`（仅有 `qweight`）。这指向**模型 forward 的 MLA prefill 分支**，并非已有证据可确认的 decode kernel 失败。初轮服务配置 `max_model_len=2048`、`gpu_memory_utilization=.70`，与 M0-3/M0-5 成功的 in-process 路径不同；具体触发分支及是否为模型/后端兼容问题，须由原始堆栈和控制变量复跑确认。禁止直接改 `weight`/`qweight` 或换后端来掩盖错误。

## 最小控制变量探针

新 `probe_awq_inprocess_decode.py` 沿用仓库既有 `trace_awq_3090.py` 的 vLLM 0.26.0 `LLM` 参数：`dtype=half`、`max_model_len=256`、`max_num_batched_tokens=256`、`max_num_seqs=1`、`enforce_eager=True`、`enable_prefix_caching=False`、`trust_remote_code=False`、单卡、`kv_cache_memory_bytes=512 MiB`、相同 seed。模型仍为现存 AWQ revision。使用 Chat Template 固定短提示，`min_tokens=max_tokens=2` 迫使多 token 生成；只有 2-token 运行退出 0 后，才用 `--extend-64` 在**新 run_id** 下增加 64-token 步骤。两次运行都是功能探针，耗时不作为性能指标。

运行入口 `run_decode_probe_autodl.sh` 对当前 GPU 锁使用 `flock -n`，通过存储守护预检并监控 256 MiB 新增数据盘峰值和 8/10 GiB 保留线，保留 GPU CSV、完整日志、manifest、命令、退出码、前后资源及 SHA-256。总调度在安全交互会话中复制两个新文件并核对哈希后执行：

```bash
cd /root/autodl-tmp/p2-phase1-deploy
bash ./run_decode_probe_autodl.sh
# 仅第一轮 PASS 后，在新目录执行：
bash ./run_decode_probe_autodl.sh --extend-64
```

本地验证：`python -B -m py_compile stage_artifacts/Phase1/probe_awq_inprocess_decode.py` 退出 0；`bash -n stage_artifacts/Phase1/run_decode_probe_autodl.sh` 退出 0。输入 SHA-256：探针 `f92b92fc7bd3266bf5ebbbc4e8b92944ff69b78f4cfa3d8c7dd8caa582cadfe1`；runner `ad9a3ba6518dda7cc2609e8c473df8e261a68aeba463dc7d2d1ece7a49d6abd8`。两文件未做真实 GPU 验证。

若相同错误在 in-process 路径复现，应保持 Phase 1 GPU 质量/decode 基线 BLOCKED，提交完整栈、环境与模型配置，由总调度决定是否授权改变后端或方案。若 in-process 通过，下一步只说明服务配置/执行路径存在差异；不能据此宣称服务端根因已证实，须再以最小服务配置逐项核查。
