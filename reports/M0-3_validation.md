# P2-M0-3 验收报告 · 运行环境与 Router Trace

2026-11-17 更新。M0-3 已补齐 RTX 3090 上的社区 AWQ 真实模型 Trace，当前自查 **PASS（AWQ 路径）**，等待用户验收。下文保留本地 CPU 轮次的结论与复现记录。[3090 实测验收报告](M0-3_3090_AWQ_validation.md)

## 1. 执行摘要

- **阶段状态 BLOCKED。** Trace 实现、CPU 合成测试、固定版官方源码与 HF 加载器验证均已完成。完整预训练模型的 Trace 缺少本地快照及可运行硬件。
- **[R] SYNTHETIC** · 固定版 DeepSeek Remote Code 通过 HF / Accelerate 重载 3 个随机 BF16 Safetensors 分片。在 CPU eager 路径记录 3 个路由层 × 5 个有效 token，共 15 条记录。插桩前后全部模型 logits 逐元素相同，最大绝对差为 0。
- **[R] MEASURED** · 官方 tokenizer 和 Chat Template 用固定提示词生成 25 个 token，其中 24 个通过特殊 token 过滤。该提示词未用于预训练模型前向。
- **[U] UNKNOWN** · 预训练 BF16 / INT4 的实际路由、CUDA、vLLM、KV Cache、量化后端与性能仍待实测。

## 2. 输入与环境

- 用户已批准从 M0-2 推进到 M0-3。固定模型为 `deepseek-ai/DeepSeek-V2-Lite-Chat`，revision `85864749cd611b4353ce1decdb286193298f64c7`。M0-2 的索引、4 个真实 Header 及其 SHA-256 继续作为权重和容量依据。[M0-2 报告](M0-2_validation.md)
- 固定 HF `config.json`、`configuration_deepseek.py`、`modeling_deepseek.py`、tokenizer 与许可均按固定 URL 获取并留存来源和 SHA-256。最终元数据获取一轮共读取 4,714,389 字节。源码以原字节在本地依赖目录执行，保留上游许可。合成权重由固定 seed 生成。[元数据获取清单](../experiments/results/M0-3-20261117T0947098220-source/source.json) · [固定版仓库](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/tree/85864749cd611b4353ce1decdb286193298f64c7)
- Windows 10 build 19045、Intel i5-8265U、RAM 7.85 GiB、探测时可用约 0.57 GiB。注册表显示 Intel UHD 620 和 Oray 虚拟设备，未探测到 CUDA 设备。复用 Python 3.10.20、PyTorch 2.12.1+cpu，使用隔离的 Transformers 4.45.2、Accelerate 0.34.2、Safetensors 0.8.0。无评测数据集，seed 固定为 20261117。[兼容性矩阵](compatibility_matrix.md) · [依赖版本](../configs/m0_3-runtime.txt)
- 当前工作目录无 Git 仓库。归档代码 SHA-256、运行时间、完整命令与日志。本项目继续与 Stage D 独立。没有租用 GPU、下载预训练权重或启动训练。[代码审计日志](../experiments/results/M0-3-20261117T0952334874-audit-final/stdout.log)

## 3. 文件变更与实现

| 文件 | 实现 |
|---|---|
| [trace_router.py](../scripts/trace_router.py) | `probe` 盘点环境。 `real` 对显式指定的本地快照作只读预检查，再用官方 HF 代码运行固定提示词的 prefill。每轮保存 manifest，缺资源时返回 2。 |
| [router.py](../src/p2/trace/router.py) | 在每个 MoE gate 安装 forward hook，记录原生 ID 和权重。从实际输入和 gate weight 重放 FP32 Linear 并计算全概率，核对 Top-6。对同一输入比较完整模型 logits，退出时移除 hook。 |
| [runtime.py](../src/p2/trace/runtime.py) | 硬件探查、模型源码与元数据 SHA-256 检查、预训练分片字节数与加载诊断、BF16 eager loader、官方 tokenizer。 |
| [fetch_m0_3_source.py](../scripts/fetch_m0_3_source.py) · [运行依赖](../configs/m0_3-runtime.txt) | 固定版源码与 tokenizer 的有界下载和 Windows CPU 验证环境。共用 HTTP 客户端的默认限制仍为 2 MiB，仅 tokenizer 显式用 6 MiB 上限。 |
| [test_m0_3.py](../tests/test_m0_3.py) · [原生探针](../tests/probe_native_router.py) · [离线核对](../tests/verify_router_trace.py) | 26 个路由层覆盖、异常测试、官方源码合成 checkpoint 集成测试、独立 JSONL 验证。 |

## 4. 实际执行命令与输出

日期均为 2026-11-17，时间为 UTC+08:00。运行目录的 `command.json` 记录解释器、完整参数、起止时间、环境及退出码，另保存 stdout/stderr。

| 时间 | 行为与结果 | 退出码 | 命令与日志 |
|---|---|---:|---|
| 09:37:55–09:41:47 | 安装隔离的 Transformers 4.44.2 与 CPU 加载依赖，获取第一批官方源码 | 0 | [依赖命令](../experiments/results/M0-3-20261117T0937550490-dependencies/command.json) · [依赖日志](../experiments/results/M0-3-20261117T0937550490-dependencies/pip.log) · [源码获取命令](../experiments/results/M0-3-20261117T0939194133-source/command.json) · [加载依赖日志](../experiments/results/M0-3-20261117T0941383912-loader-dependencies/pip.log) |
| 09:44:34–09:44:42 | 首次运行原生探针。4.44.2 loader 报缺少条件导入的 `flash_attn` | 1 | [命令](../experiments/results/M0-3-20261117T0944343487-native/command.json) · [异常日志](../experiments/results/M0-3-20261117T0944343487-native/stderr.log) |
| 09:46:31–09:48:35 | 安装隔离的 4.45.2、Tokenizers 0.20.3 与其余依赖，复取官方源码并验证 tokenizer 文件 | 0 | [升级命令](../experiments/results/M0-3-20261117T0946317219-loader-retest/hf.command.json) · [安装日志](../experiments/results/M0-3-20261117T0946317219-loader-retest/hf.log) · [其余依赖命令](../experiments/results/M0-3-20261117T0946317219-loader-retest/loader.command.json) · [获取命令](../experiments/results/M0-3-20261117T0947098220-source/command.json) |
| 09:48:53–09:49:07 | 官方源码 BF16 随机模型与官方 tokenizer 复测成功 | 0 | [命令](../experiments/results/M0-3-20261117T0948529721-native/command.json) · [输出](../experiments/results/M0-3-20261117T0948529721-native/stdout.log) |
| 09:48:53–09:48:58 | 执行 `trace_router.py real --device cpu`。缺快照及内存，运行止于预检查 | 2 | [命令](../experiments/results/M0-3-20261117T0948538020-pretrained/command.json) · [阻塞清单](../experiments/results/M0-3-20261117T0948538020-pretrained/trace/manifest.json) |
| 09:48:54–09:48:59 | 执行 `trace_router.py probe`，记录 GPU、驱动、RAM、软件与 BF16 运算 | 0 | [命令](../experiments/results/M0-3-20261117T0948544470-runtime/command.json) · [环境输出](../experiments/results/M0-3-20261117T0948544470-runtime/stdout.log) |
| 09:50:36–09:50:42 | 执行 `trace_router.py real --device cuda --dry-run`。缺快照及 CUDA | 2 | [命令](../experiments/results/M0-3-20261117T0950364021-pretrained-cuda/command.json) · [阻塞清单](../experiments/results/M0-3-20261117T0950364021-pretrained-cuda/trace/manifest.json) |
| 09:51:52–09:52:04 | 最终原生探针成功加载 3 个合成分片，完成 15 条带 SYNTHETIC 字段的记录 | 0 | [命令](../experiments/results/M0-3-20261117T0951529190-native-final/command.json) · [输出](../experiments/results/M0-3-20261117T0951529190-native-final/stdout.log) · [分片加载](../experiments/results/M0-3-20261117T0951529190-native-final/stderr.log) |
| 09:52:33–09:52:39 | 最终 22 项测试、15 个 Python 文件内存编译、22 个依赖版本复核、离线 Trace 验证及 M0-2 证据复核 | 0 | [单测命令](../experiments/results/M0-3-20261117T0952334799-tests-final/command.json) · [单测日志](../experiments/results/M0-3-20261117T0952334799-tests-final/stdout.log) · [审计命令](../experiments/results/M0-3-20261117T0952334874-audit-final/command.json) · [审计日志](../experiments/results/M0-3-20261117T0952334874-audit-final/stdout.log) |

## 5. 测试与自查

- **事实与版本。** 源码、配置和 tokenizer 有固定 revision 与 SHA-256。Loader 强制本地 Safetensors、BF16、eager，报告 missing/unexpected keys 即失败。预训练路径额外检查官方索引和预期分片大小；该检查未读取整个权重做 SHA-256。
- **程序正确性。** 6 项 M0-3 合成单测覆盖 26 个路由层、过滤后的 token 坐标、softmax、原始选中权重、Top-6 ties、重复和越界 ID、NaN、错误归一化、缺少层、训练态和失败后移除 hook。另加 16 项 M0-2 回归，共 22 项通过。
- **数值与可重复。** 官方源码原生探针用相同输入做两次完整前向，切换 hook 时保持 eval/inference、输入、dtype 和 eager 后端一致。最终三分片重复运行复现全部 15 条数值记录。
- **数据口径。** Routed ID 从原生 gate 结果直接读取。64 个 logits 是同一输入/权重的 FP32 重算，全量 softmax 由该重算得到。未归一化的 Top-6 权重由 gate 返回，离线再次核对。只采 prefill、关闭 KV Cache，均未测推理质量或延迟。
- **权限。** 默认权重路径只读且离线。远端获取限于公开源码、配置、许可证和 tokenizer 元数据。各轮使用新目录，保留原始失败和复测。

## 6. 数据与指标

| 指标 | 本次结果 |
|---|---|
| 官方源码集成探针 | 3 个 BF16 合成权重分片，4 个 decoder 层，层 0 Dense，层 1–3 各 64 Routed 和 1 个 Shared MLP |
| 输入与 Trace | 2×5 token 输入，过滤 1 个 padding 和 4 个有效位置上的特殊 token，余 5 个 token × 3 层 = 15 行 |
| 两次模型输出 | 完整 logits 形状 `[2,5,32]`，FP32，逐元素相同，最大绝对差 0 |
| 独立 softmax 复核 | Python FP64 softmax 与记录的 FP32 概率最大绝对差约 `7.48e-9`；六个选中权重之和为 `0.186608–0.249394` |
| 官方 tokenizer 探针 | 固定 Chat Template 渲染后 25 个 token，特殊 token 过滤后 24 个普通 token |
| 纯合成层覆盖 | 26 个 MoE 层，3 个有效 token，78 条内存记录，模型 logits 最大绝对差 0 |

[正式归档 SYNTHETIC Trace JSONL](../experiments/results/M0-3-20261117T0951529190-native-final/trace/router_trace.jsonl) · [可复核运行清单](../experiments/results/M0-3-20261117T0951529190-native-final/trace/manifest.json)

每行使用 `evidence` 标记来源。`layer_id`、`batch_id`、`sequence_index` 均为零基，`sequence_index` 为输入张量的原列坐标。`token_id` 是词表或合成词表 ID。`router_logits` 和 `router_probabilities` 均按 0–63 的 Routed Expert ID 排列。`topk_expert_ids` 与 `topk_weights` 同位置对应，保留原生 gate 的顺序。Shared 不参与 Top-6。无跨请求或 decode 的 token 坐标。

## 7. 问题与未验证项

- **真实 BF16 Trace 阻塞。** 已检查的本地位置没有目标预训练快照。即使快照可用，当前物理 RAM 为 7.85 GiB，低于 29.26 GiB 的 BF16 权重需求，且没有可用 CUDA 设备。提供相同 revision 的已有快照和 RAM 足够的 CPU 主机即可尝试复现，或者在支持原生 BF16、空闲显存高于权重容量且留有运行余量的 CUDA 主机验证。
- **后端待核。** vLLM MLA、FusedMoE、AWQ、GPTQ、Marlin、其他实际 INT4 kernel、CUDA 精度与 KV Cache 均未运行。验证原生 HF eager 合成路径的结果仅适用于本次实测设置。
- **复现所需授权。** 若无法提供已有模型文件，获取完整 BF16 权重涉及约 29.26 GiB 下载。新增 GPU 或该下载需先获得用户授权。本阶段已保留最小真实命令，尚无该类操作。

## 8. 计划变更

保留原 checkpoint、BF16 参考和 Router Trace 核心验收。先复用已有 CPU PyTorch，使用未修改的官方源码与合成 checkpoint 验证 HF 入口。4.44.2 将可选 FlashAttention 误判为必需包，因此采用独立目录的 4.45.2 复测。新增离线文件校验及三分片合成加载，覆盖真实入口可在当前环境触达的部分。GPU 和预训练路径保留 BLOCKED。

## 9. 验收清单

- [x] 探测实际 GPU/CUDA、驱动、RAM、软件版本，并形成 [兼容性矩阵](compatibility_matrix.md)。
- [x] 对 Top-6 ID、原始权重、全量概率、token 对齐、过滤和插桩结果做合成验证。
- [x] 官方固定版 Remote Code 经 CPU eager 运行三分片 BF16 合成 checkpoint，保存标记为 SYNTHETIC 的 JSONL。
- [x] 官方 tokenizer 与模板独立验证，真实 BF16 CPU/CUDA 入口完成预检查。
- [ ] 完整预训练模型执行并产生真实 Trace。缺权重与硬件，保留阻塞清单和复现命令。

## 10. 复现与人工验收

在项目根目录执行，`$py` 可替换为已有 PyTorch 的 Python。Windows CPU 复现环境如下。安装只在缺少项目依赖目录时执行。

```powershell
$py = "$env:USERPROFILE/.conda/envs/pytorch_clean/python.exe"
$env:PYTHONPATH = Join-Path $PWD ".venv/m0_3_verified"
$env:HF_HOME = Join-Path $PWD ".venv/m0_3_hf"
if (-not (Test-Path $env:PYTHONPATH)) {
  & $py -B -m pip --isolated install --index-url https://pypi.org/simple --only-binary=:all: --no-deps --target $env:PYTHONPATH -r configs/m0_3-runtime.txt
}
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
& $py -B scripts/trace_router.py probe
& $py -B -m unittest discover -s tests -v
& $py -B tests/verify_router_trace.py experiments/results/M0-3-20261117T0951529190-native-final/trace
```

重新运行原生合成探针只需官方源码与小元数据。从新目录重新获取时需要联网，`fetch_m0_3_source.py` 本身使用标准库直接访问固定 revision。

```powershell
$stamp = Get-Date -Format yyyyMMddTHHmmssffff
$source = ".venv/m0_3_official_metadata"
if (-not (Test-Path $source)) {
  & $py -B scripts/fetch_m0_3_source.py --output-dir $source
}
$run = "experiments/results/M0-3-$stamp-repro"
& $py -B tests/probe_native_router.py --source-dir $source --scratch-dir ".venv/m0_3_probe_$stamp" --output-dir $run
& $py -B tests/verify_router_trace.py $run
```

在已授权的环境中，使用已有的完整原版快照先预检查，满足条件后执行一次固定提示词的前向对照。`--device cuda` 需要目标机 PyTorch 支持 CUDA 和原生 BF16，CPU 主机可换成 `--device cpu`。

```powershell
$snapshot = "E:/models/DeepSeek-V2-Lite-Chat/85864749cd611b4353ce1decdb286193298f64c7"
$stamp = Get-Date -Format yyyyMMddTHHmmssffff
& $py -B scripts/trace_router.py real --snapshot $snapshot --device cuda --dry-run --output-dir "experiments/results/M0-3-$stamp-preflight"
& $py -B scripts/trace_router.py real --snapshot $snapshot --device cuda --output-dir "experiments/results/M0-3-$stamp-real"
```

人工验收请确认合成与真实证据边界，并决定是否提供运行环境以补齐预训练 Trace。本阶段停在 M0-3，等待验收。M0-4 需要真实 Trace 作为前置证据。
