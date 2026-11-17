# P2-M0-3 运行环境与后端兼容性

2026-11-17，Asia/Shanghai。目标 checkpoint 为 `deepseek-ai/DeepSeek-V2-Lite-Chat@85864749cd611b4353ce1decdb286193298f64c7`。

## 云机实测环境与后端

Ubuntu 22.04.5，单张 RTX 3090 24 GiB、驱动 580.105.08、计算能力 8.6。隔离环境为 Python 3.12.3、PyTorch 2.11.0+cu130、vLLM 0.26.0、Transformers 5.23.0。实测对象是社区 `TechxGenus/DeepSeek-V2-Lite-Chat-AWQ@e9fdf80ec6f1d200a029c41004bee40993763b43`。[环境日志](../experiments/results/M0-3-20261117T221327-3090/probe.log) · [加载与退出日志](../experiments/results/M0-3-20261117T221327-3090/trace04.log)

| 路径 | 实测状态 |
|---|---|
| AWQ INT4 完整模型加载与短请求 | **PASS [R]**。单卡，FP16 激活，max model len 256，512 MiB KV 预算。权重加载后 vLLM 报告使用 8.63 GiB。 |
| CUDA Marlin MoE 与 Router Trace | **PASS [R]**。`quantization=auto_awq`，实际使用 Marlin WNA16，`AutoAWQMoEMethod`、`FusedMoEKernel`、`MarlinExperts`。原生 `GroupedTopKRouter` 采到全部 26 层。 |
| vLLM MLA | **PASS [R]，短 prefill**。模型层为 `DeepseekV2MLAAttention`，引擎选择 `TRITON_MLA`，prefill 日志为 `FLASH_ATTN MLA`；未做长上下文或多步 decode 测试。 |
| 官方 BF16、GPTQ、实际长上下文与并发 | **UNVERIFIED [U]**。此次已测的权重与评测范围限定在社区 AWQ 的单请求 prefill。 |

## 本地 CPU 环境

- Windows 10 build 19045，Intel Core i5-8265U。物理 RAM 为 8,428,179,456 字节（7.85 GiB），探测时可用 608,530,432 字节（约 0.57 GiB）。
- Windows 显示驱动记录为 Intel UHD Graphics 620（27.20.100.8984）和 OrayIddDriver（17.50.19.949）。PATH 无 `nvidia-smi`、`nvcc`，当前 PyTorch 无 CUDA 构建、设备数 0。
- Python 3.10.20，PyTorch 2.12.1+cpu，Transformers 4.45.2，Accelerate 0.34.2，Safetensors 0.8.0，Tokenizers 0.20.3。CPU BF16 矩阵乘法已实测。
- vLLM、Triton、FlashAttention、AutoAWQ、AutoGPTQ 和 GPTQModel 未安装在本次环境。默认模型快照目录及已检查的本地候选位置没有目标 checkpoint。[运行环境日志](../experiments/results/M0-3-20261117T0948544470-runtime/stdout.log)

## 本地兼容性矩阵

| 路径 | 实际状态 | 证据和边界 |
|---|---|---|
| 官方 tokenizer、Chat Template | **PASS [R]** | 官方文件 SHA-256 匹配。固定提示词模板输出 25 个 ID，去除 BOS 后 24 个普通 token。实际类为 `LlamaTokenizerFast`。[记录](../experiments/results/M0-3-20261117T0951529190-native-final/trace/manifest.json) |
| HF Remote Code + CPU eager + Safetensors 加载器 | **PASS [R] SYNTHETIC** | 用未修改的固定版官方 Python 源码运行 `AutoModelForCausalLM`，成功重载随机 BF16 的三个合成分片并执行两次前向。实测 `DeepseekV2ForCausalLM`。[载入日志](../experiments/results/M0-3-20261117T0951529190-native-final/stderr.log) |
| HF 原生 MLA 模块及逐专家 MoE | **PASS [R] SYNTHETIC** | eager `DeepseekV2Attention`、`DeepseekV2MoE`、`MoEGate` 已执行。探针对齐 native gate 输出。运行关闭 KV Cache，实际 KV 布局与显存仍为 [U]。[Trace](../experiments/results/M0-3-20261117T0951529190-native-final/trace/router_trace.jsonl) |
| 官方 27 层 BF16 checkpoint · CPU | **BLOCKED [R/U]** | 缺快照和官方四分片。BF16 权重约 29.26 GiB，超过 7.85 GiB 整机 RAM。本轮无预训练张量加载或前向。[CPU 检查](../experiments/results/M0-3-20261117T0948538020-pretrained/trace/manifest.json) |
| 官方 27 层 BF16 checkpoint · CUDA | **BLOCKED [R/U]** | 缺快照及可用 CUDA 环境。单卡 BF16 权重驻留需要超过 29.26 GiB 的空闲显存并预留运行空间。[CUDA 检查](../experiments/results/M0-3-20261117T0950364021-pretrained-cuda/trace/manifest.json) |
| HF FlashAttention 2 | **UNVERIFIED [U]** | 当前环境无包、无 CUDA；只实测 eager。Transformers 4.44.2 曾将官方源码中条件导入的 `flash_attn` 预检为必需依赖，4.45.2 eager 已复测通过。[失败日志](../experiments/results/M0-3-20261117T0944343487-native/stderr.log) · [4.45.2 导入预检源码](https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/dynamic_module_utils.py) |
| vLLM MLA、FusedMoE | **BLOCKED [S/U]** | v0.7.3 源码可定位 MLA 与 FusedMoE，vLLM 本地未安装，未运行该模型族和后端。实际 KV 布局、后端选择与 trace 接口均待验。[vLLM 固定版源码](https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/models/deepseek_v2.py) |
| AWQ INT4、GPTQ INT4 | **BLOCKED [U]** | 没有已配置的量化 checkpoint、对应 loader 和 CUDA。加载能力、实际 dtype 和专家层覆盖未知。 |
| Marlin、其他 INT4 kernel | **BLOCKED [U]** | 无可执行量化后端，未取得 kernel 选择日志或 profiler 证据。 |
