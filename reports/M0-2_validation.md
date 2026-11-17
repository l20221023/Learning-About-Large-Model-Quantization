# P2-M0-2 验收报告 · Safetensors Header 与专家权重映射

## 1. 执行摘要

- **状态 PASS，等待用户验收。** 固定 revision 的 4 个官方分片 Header 均已实际读取。5,291 个张量全部通过 key、分片归属、shape、dtype 和数据偏移检查，差异表为 0 行。
- **[H/R] MEASURED** · 四个分片的张量均为 BF16。和索引一致的 payload 字节数为 31,412,968,448，由实际 Header 所记录的数据范围计算。
- **[R] SYNTHETIC** · 16 项测试通过。单专家提取在两个合成分片上覆盖三个真实尺寸矩阵，验证输出数据逐字节相等、未选中张量排除、原始文件内容相等及拒绝覆盖。
- **[U] UNKNOWN** · 真实权重数据区提取、逐专家数值和模型前向待验证。当前阶段只读取官方索引、配置和 Header。

## 2. 输入与环境

- 完整读取两份用户提供的输入，已复制进项目并核对原件和副本 SHA-256 一致。总执行规范 SHA-256 为 `9394c3542e3ccbc03be6e29832f781d98c1e9dc8828cf283b570df34a4708b90`，M0-1 报告 SHA-256 为 `ddc60ae2b58af584d00350700d195ed21f997799f205f340a4f75375e66f4d2b`。[总执行规范](../docs/P2_MASTER_RUNBOOK.md) · [M0-1 输入](../docs/inputs/DeepSeek-V2-Lite-Chat_P2-M0-1_正式复核报告%281%29.md)
- 主模型 `deepseek-ai/DeepSeek-V2-Lite-Chat`，固定 revision `85864749cd611b4353ce1decdb286193298f64c7`。采用 M0-1 第 4 节的 shape 推导值，另核对固定版本配置的拓扑与 BF16 声明。[官方索引](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/85864749cd611b4353ce1decdb286193298f64c7/model.safetensors.index.json) · [官方配置](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/85864749cd611b4353ce1decdb286193298f64c7/config.json) · [固定源码](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/85864749cd611b4353ce1decdb286193298f64c7/modeling_deepseek.py)
- Windows 10 build 19045，Codex 自带 Python 3.12.14，8 个逻辑 CPU，记录时工作盘余量约 56.96 GiB。项目目录初始为空，`git status --short --branch` 显示无 Git 仓库。用源码 SHA-256 记录版本。Stage D 与本项目独立。[环境日志](../experiments/results/M0-2-20261117T0927433550-verification/environment.log) · [源码校验日志](../experiments/results/M0-2-20261117T0927433550-verification/source.log)
- 本次使用 CPU 和 Python 标准库。当前解释器未安装 PyTorch、Transformers、vLLM、Triton、safetensors。PATH 上没有 `nvidia-smi` 或 `nvcc`，Windows GPU CIM 查询拒绝访问，实际 GPU 与驱动未核实。项目目录和探查的 Hugging Face 默认缓存中未发现该模型的完整本地分片。
- 无评测数据集。合成张量由测试代码写入固定 BF16 字节串，无随机种子。仅执行公开索引、配置和有边界的 Header 读取。最终成功轮次应用层读取 1,139,574 字节响应体，其中分片范围内元数据为 658,128 字节。

## 3. 文件变更与实现

| 路径 | 实现 |
|---|---|
| [scripts/inspect_safetensors.py](../scripts/inspect_safetensors.py) | 运行固定模型审计，支持远端 Header、本地 shard 和 dry-run。保存 CSV、差异矩阵、原始输入与请求记录。 |
| [scripts/extract_expert.py](../scripts/extract_expert.py) | 用零基层编号及 Routed Expert ID 定位 gate、up、down。支持索引 dry-run、本地 metadata-only 和按需写出独立专家文件。 |
| [checkpoint.py](../src/p2/weights/checkpoint.py) · [audit.py](../src/p2/weights/audit.py) | 固定版本、预期张量集、索引校验和逐张量对比。输出 `VERIFIED`、`PREDICTED`、`UNKNOWN` 和 `MISMATCH`。 |
| [safetensors_io.py](../src/p2/weights/safetensors_io.py) · [extract.py](../src/p2/weights/extract.py) | 按格式验证 dtype、字节数、完整偏移区间。远程最多读取每个 shard 的 8 字节长度前缀和不超过 4 MiB 的 Header，本地以只读方式打开，输出文件独占创建。 |
| [test_m0_2.py](../tests/test_m0_2.py) · [verify_recorded_audit.py](../tests/verify_recorded_audit.py) | SYNTHETIC 单元与 CLI 测试。离线独立交叉复核真实索引、原始 Header、manifest 和 CSV。 |
| [README](../README.md) · [AGENTS](../AGENTS.md) · [计划](../docs/PLAN.md) · [决策](../docs/DECISIONS.md) · [证据台账](../docs/EVIDENCE_LEDGER.md) | 长期执行入口、复现命令、数据字典、决策、未验证项和阶段门禁。 |

## 4. 实际执行命令与输出

以下时间均为 2026-11-17 上海时间（UTC+08:00）。每次执行目录中的 `command.json` 或 `*.command.json` 保存完整解释器路径、参数、精确起止时间和退出码。初始网络检查命令 `curl.exe --silent --show-error --max-time 15 --head <固定 revision 的索引 resolve URL>` 返回 7，无法在沙箱内建立 HTTPS 连接。

| 时间 | 命令参数与结果 | 退出码 | 记录 |
|---|---|---:|---|
| 09:20:36–09:20:37 | `-B scripts/inspect_safetensors.py --remote-headers --output-dir <新目录>`，沙箱连接 `PermissionError` | 2 | [命令](../experiments/results/M0-2-20261117T0920366452-sandbox/command.json) · [日志](../experiments/results/M0-2-20261117T0920366452-sandbox/audit.log) · [失败明细](../experiments/results/M0-2-20261117T0920366452-sandbox/audit/manifest.json) |
| 09:20:53–09:21:19 | 同命令经自动权限审查放行后首次成功，5,291 行通过 | 0 | [命令](../experiments/results/M0-2-20261117T0920534426-official/command.json) · [日志](../experiments/results/M0-2-20261117T0920534426-official/audit.log) |
| 09:23:45–09:23:48 | `-B -m unittest discover -s tests -v`，首次 15 项中 14 项通过，1 项因 Windows 符号链接权限跳过 | 0 | [命令](../experiments/results/M0-2-20261117T0923456004-tests/command.json) · [日志](../experiments/results/M0-2-20261117T0923456004-tests/unit.log) |
| 09:25:03–09:25:05 | 相同单测命令，补跳转测试及 Windows 路径解析模拟，最终 16 项全通过 | 0 | [命令](../experiments/results/M0-2-20261117T0925037804-tests/command.json) · [日志](../experiments/results/M0-2-20261117T0925037804-tests/unit.log) |
| 09:25:06–09:25:33 | 新版真实 Header 审计，四个 Header 成功，索引请求出现 `SSLEOFError` | 2 | [命令](../experiments/results/M0-2-20261117T0925064472-official/command.json) · [日志](../experiments/results/M0-2-20261117T0925064472-official/audit.log) · [失败明细](../experiments/results/M0-2-20261117T0925064472-official/audit/manifest.json) |
| 09:26:33–09:26:59 | 重试真实审计，最终 5,291 行通过 | 0 | [命令](../experiments/results/M0-2-20261117T0926335375-official/command.json) · [日志](../experiments/results/M0-2-20261117T0926335375-official/audit.log) |
| 09:27:43–09:27:46 | 离线交叉复核、真实索引单专家 dry-run、CLI help/dry-run、内存编译源码、探查环境、比对两次成功结果 | 均为 0 | [复核](../experiments/results/M0-2-20261117T0927433550-verification/offline.log) · [专家定位](../experiments/results/M0-2-20261117T0927433550-verification/expert-10-7.log) · [跨运行比对](../experiments/results/M0-2-20261117T0927433550-verification/repeatability.log) |

## 5. 测试与自查

- **事实与版本** · 官方索引、配置和八次 206 响应的上游 `X-Repo-Commit` 均为固定 revision。每个 shard 的两次请求中，对象大小、ETag 和关联 ETag 一致。四个 Header 在两轮成功执行间逐字节相同，两份 CSV 也相同。
- **程序正确性** · 验证正常跨 shard 提取、越界 ID、索引或 Header 缺 key、错误 shape 和 dtype、截断、重复 key、offset gap/overlap、分片不全、路径越界、原文件与既有导出只读。Windows 无创建符号链接特权，本次用模拟真实路径解析目标覆盖越界分支。Range 测试验证收到 200 或错误 Content-Range 时不读取响应体，重定向保留 Range 并关闭中间响应。
- **实验口径** · 本阶段没有模型推理、质量、吞吐或 GPU 性能数据。审计的网络执行时长仅用于复现日志。
- **数据可靠性** · 离线复核证实每个索引 key 与 CSV 及唯一 Header 一一对应，26×64 个 Routed Expert 均有三条记录。CSV 中 dtype、shape、offset 与原始 Header 一致，逐 shard 数据范围连续且覆盖声明的文件尾。
- **权限与复现** · 每轮结果使用新目录，保留失败记录。远端只打开受控元数据范围，输出仅记录官方 URL、CDN host、对象标识及 Header SHA-256。真实模型 payload 未传输。

## 6. 数据与指标

| 官方 shard | Header 张量条数 | JSON Header 字节 | 数据区字节 |
|---|---:|---:|---:|
| `model-00001-of-000004.safetensors` | 1,329 | 164,584 | 8,594,722,816 |
| `model-00002-of-000004.safetensors` | 1,491 | 185,472 | 8,591,571,968 |
| `model-00003-of-000004.safetensors` | 1,491 | 185,904 | 8,590,532,608 |
| `model-00004-of-000004.safetensors` | 980 | 122,136 | 5,636,141,056 |
| 合计 | 5,291 | 658,096 | 31,412,968,448 |

数值覆盖固定版本的全量索引，无采样。四个 8 字节长度前缀计入 658,128 字节 Range 响应体，索引和配置再贡献 481,446 字节。数据区字节数由 Header offset 和响应中的对象长度校验得出。[主映射 CSV](../experiments/results/M0-2-20261117T0926335375-official/audit/expert_weight_mapping.csv) · [差异 CSV](../experiments/results/M0-2-20261117T0926335375-official/audit/mapping_differences.csv) · [原始 Header 和 SHA-256 清单](../experiments/results/M0-2-20261117T0926335375-official/audit/manifest.json)

其中 Routed 张量 4,992 个，Router 26 个，Shared 张量 78 个，Dense FFN 3 个，其他 192 个。真实映射显示第 10 层第 7 号专家的三个张量均在第 2 个 shard，gate/up 为 `[1408, 2048]`，down 为 `[2048, 1408]`，合计 payload 17,301,504 字节。各张量的精确 offset 见主映射 CSV。

## 7. 问题与未验证项

- **[U] 真实本地专家 payload。** 目前没有可用的本地官方 shard。若已有文件，可按下节命令读取第 2 分片中约 16.5 MiB 的选中专家数据，不需要其余 3 个分片。真实张量数值相等与专家函数相似性仍待测。
- **[U] 模型运行。** 本阶段没有执行真实前向、Router Trace、GPU 兼容性、MLA/KV Cache、量化 kernel、质量或性能验证。GPU 型号和驱动仍待获取。
- **复现要求。** 重新远程审计需允许访问 Hugging Face 及 CDN。无网络环境可用已归档原始 Header、索引和 CSV 做离线复核。若已有本地分片，可选用本地审计路径。

## 8. 计划变更

沿用固定模型和 M0-2 原验收目标。当前环境无独立 Python 安装，使用已可用的 Codex 解释器与标准库进行有上限的读取，无需引入模型运行时。初测后收紧 3xx 响应处理，明确禁止读取其正文。受限沙箱连接失败后只重试有界元数据访问，最终实际复核覆盖全部分片。采用空目录布局及 SHA-256 版本记录，原始输入保存在项目内。

## 9. 验收清单

- [x] 固定 revision 的索引与实际四个 shard 对齐。
- [x] 所有 5,291 个张量的 shape、dtype、offset 和分片归属核验通过，差异表已生成。
- [x] 单专家索引定位、按需读取、dry-run 和 metadata-only 已实现，合成跨分片读写与失败用例通过。
- [x] 原始 Header、网络范围和对象标识、哈希、命令、退出码、日志和未核验事项齐全。
- [x] 完成阶段内自查、修复和复测，停在 M0-2 人工验收门禁。

## 10. 复现与人工验收

从项目根目录在 PowerShell 运行。可将 `$py` 替换成已安装的 Python 3.10+ 路径。

```powershell
$py = "$env:USERPROFILE/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"
$saved = "experiments/results/M0-2-20261117T0926335375-official/audit"
& $py -B -m unittest discover -s tests -v
& $py -B tests/verify_recorded_audit.py $saved
& $py -B scripts/extract_expert.py --index "$saved/model.safetensors.index.json" --layer 10 --expert 7 --dry-run

# 重新获取官方元数据
$run = "experiments/results/M0-2-$(Get-Date -Format yyyyMMddTHHmmssffff)-repro"
& $py -B scripts/inspect_safetensors.py --remote-headers --output-dir $run
& $py -B tests/verify_recorded_audit.py $run

# 可选。已有官方本地 shard 时，替换示例路径
$shards = "E:/models/deepseek-v2-lite-chat"
& $py -B scripts/extract_expert.py --index "$saved/model.safetensors.index.json" --shard-dir $shards --layer 10 --expert 7 --metadata-only
& $py -B scripts/extract_expert.py --index "$saved/model.safetensors.index.json" --shard-dir $shards --layer 10 --expert 7 --output "./layer10-expert7.safetensors"
```

人工验收请确认固定版本的 Header 证据及零差异矩阵，并核对合成提取覆盖与真实 payload 验证边界。通过后才进入 M0-3，检查真实模型运行环境和 Router Trace 路径。当前等待验收。
