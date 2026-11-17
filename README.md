# P2 · DeepSeek-V2-Lite-Chat 量化与 MoE 结构优化

本项目独立于 Stage D。执行规范见 [总执行文档](docs/P2_MASTER_RUNBOOK.md)，当前进度见 [执行计划](docs/PLAN.md)。

## M0-2 复现

需要 Python 3.10+。项目根目录下在 PowerShell 执行以下命令。若 `python` 指向 Windows 商店别名，可改用已有的 Python 解释器绝对路径。真实 Header 审计需要访问 Hugging Face 及其分片 CDN。

```powershell
python -B -m unittest discover -s tests -v
$run = "experiments/results/M0-2-$(Get-Date -Format yyyyMMddTHHmmssffff)-repro"
python -B scripts/inspect_safetensors.py --remote-headers --output-dir $run
python -B tests/verify_recorded_audit.py $run
python -B scripts/extract_expert.py --index "$run/model.safetensors.index.json" --layer 10 --expert 7 --dry-run
```

`inspect_safetensors.py` 对每个远端分片只读长度前缀和 Header。指定 `--index <本地索引> --shard-dir <本地分片目录>` 可离线核对本地文件。单专家的本地复现命令如下。

```powershell
python -B scripts/extract_expert.py --index "$run/model.safetensors.index.json" --shard-dir E:/models/deepseek-v2-lite-chat --layer 10 --expert 7 --metadata-only
python -B scripts/extract_expert.py --index "$run/model.safetensors.index.json" --shard-dir E:/models/deepseek-v2-lite-chat --layer 10 --expert 7 --output "$run/layer10-expert7.safetensors"
```

将示例分片目录替换为已有本地文件所在路径。路径解析后分片须位于给定目录内。可将 Hugging Face 缓存中实际含目标 shard 的文件以原分片名提供给脚本。提取文件必须使用新路径，实际读写量取决于选中专家的三个张量。

## 审计数据

| 产物 | 字段与含义 |
|---|---|
| `expert_weight_mapping.csv` | 一行一个权重 key。layer、expert 为零基 ID，仅 Routed 权重填写 expert。shape 是 JSON 数组，dtype 沿用 Safetensors 名称。offset 是相对各 shard 数据区的字节区间 `[start, end)`。|
| `mapping_differences.csv` | 同列，含所有未通过逐项核对的记录。 |
| `manifest.json` 与 `headers/` | 来源、请求时间与范围、响应字节数、对象标识、原始 Header、SHA-256 和审核状态。 |

`VERIFIED` 表示实际 Header 与索引位置和预期一致，`PREDICTED` 表示有索引和预期但缺少 Header，`UNKNOWN` 表示证据不足，`MISMATCH` 表示已有证据发现差异。审计命令退出码 0、1、2 对应通过、失败或部分完成。

## M0-3 Router Trace

RTX 3090 上的社区 AWQ 短请求与 Router Trace 已通过，详见 [3090 AWQ 验收报告](reports/M0-3_3090_AWQ_validation.md) 和 [兼容性矩阵](reports/compatibility_matrix.md)。`scripts/trace_awq_3090.py` 使用已有的社区 AWQ 快照测试 vLLM 原生路由。官方 BF16 的 HF eager 路径与本地合成记录见 [本地 M0-3 报告](reports/M0-3_validation.md)。

M0-4 已在同一 AWQ 环境完成小规模探索性 Router/负载/共激活分析，见 [M0-4 报告](reports/M0-4_validation.md)。AutoDL 数据盘缓存和安全预检/监测入口见 [存储审计](reports/storage_audit_autodl_20261118.md)、`scripts/autodl_storage_env.sh` 和 `scripts/autodl_storage_guard.sh`。官方 BF16 下载与转换须先满足数据盘安全余量。
