# P2-M0-3 RTX 3090 云机验收 · AWQ Router Trace

## 1. 执行摘要

**PASS（社区 AWQ 实测路径），等待用户验收。** 一张 RTX 3090 成功加载完整 AWQ checkpoint。原生 Router Trace 覆盖全部 26 个 MoE 层与 24 个有效 prompt token，共 **624 条**。同一请求插桩前后的完整输出 logits 逐元素相同，最大绝对差为 **0**。官方原版 BF16 Trace 保持 [U]。M0-4 未启动。

## 2. 输入、版本与资源

- 用户授权使用已提供的 AutoDL 3090 云实例，并下载社区 `TechxGenus/DeepSeek-V2-Lite-Chat-AWQ@e9fdf80ec6f1d200a029c41004bee40993763b43`。AWQ 为 4 bit、group 64、GEMM、带 zero point。两个 Safetensors 分片合计 9,086,606,872 字节（约 8.46 GiB），最终 SHA-256 与发布者仓库一致。[模型仓库](https://huggingface.co/TechxGenus/DeepSeek-V2-Lite-Chat-AWQ/tree/main) · [分片 1 SHA-256](https://huggingface.co/TechxGenus/DeepSeek-V2-Lite-Chat-AWQ/blob/main/model-00001-of-00002.safetensors) · [分片 2 SHA-256](https://huggingface.co/TechxGenus/DeepSeek-V2-Lite-Chat-AWQ/blob/main/model-00002-of-00002.safetensors)
- Ubuntu 22.04.5，RTX 3090 24 GiB，NVIDIA 驱动 580.105.08，SM 8.6，实例 cgroup 内存上限 90 GiB。云机原有 Python 3.12.3 和 PyTorch 2.12.1+cu130。为匹配 vLLM 0.26.0，在项目数据盘的独立 venv 安装并实测 PyTorch 2.11.0+cu130、Transformers 5.23.0。原有环境保留。 [云机探查](../experiments/results/M0-3-20261117T221327-3090/probe.log) · [安装记录](../experiments/results/M0-3-20261117T221327-3090/install.log)
- 实验只加载已校验的本地 Safetensors，`trust_remote_code=False`，无 API 服务、无外部数据集。固定提示词、seed 20261117、batch 1、最长 256 token、eager、KV Cache 预算 512 MiB。两次均为相同 25-token 输入、采样首个输出 token，Trace 排除 BOS 后保留 24 个输入 token。

## 3. 实现与文件

- [AWQ 探针](../scripts/trace_awq_3090.py) 直接捕获 vLLM `GroupedTopKRouter.select_experts` 的输入 logits 和原生返回的 Top-6 ID、权重。全量概率按捕获的 FP16 dtype 重建。模型路由使用原生返回值，插桩前后比较 102,400 维完整输出 logits。
- [Range 续传](../scripts/fetch_awq_shards.py) 从固定版 ModelScope 同源镜像继续下载。仅将大小和 SHA-256 都通过校验的两个分片交给 loader。
- [离线验证](../tests/verify_router_trace.py) 复核 token 坐标、层覆盖、Top-6、FP16 softmax 舍入和选中权重。共享的 [路由校验](../src/p2/trace/router.py) 默认保持 HF FP32 口径，AWQ 路径显式指定实际 dtype。
- [GPU 样本](../experiments/results/M0-3-20261117T221327-3090/trace04-gpu.csv) · [原生采集](../experiments/results/M0-3-20261117T221327-3090/trace04/raw_route_capture.json) · [最终 Trace](../experiments/results/M0-3-20261117T221327-3090/trace04/router_trace.jsonl) · [最终 manifest](../experiments/results/M0-3-20261117T221327-3090/trace04/manifest.json)

## 4. 执行、失败与复测

日期为 2026-11-17，时间为 UTC+08:00。云机工作目录位于 `/root/autodl-tmp/p2-deepseek-moe`。运行目录 `results/M0-3-20261117T221327-3090` 保存命令脚本、每轮退出码和完整输出。

| 步骤 | 结果 | 记录 |
|---|---|---|
| 22:13:27 硬件与缓存探查 | GPU 空闲、CUDA 可用，已检查的位置无模型缓存 | [探查命令](../experiments/results/M0-3-20261117T221327-3090/probe.sh) · [输出](../experiments/results/M0-3-20261117T221327-3090/probe.log) |
| 22:18:41–22:33:13 权重传输与校验 | 两片均通过，退出 0。官方站点连接超时；直接 PyPI 下载较慢，终止了尚未开始传模型的初始 Hub 工具安装。HF 镜像先下载元数据和分片前段，随后切至 SHA-256 一致的 ModelScope 分段续传，保留已有数据。 | [首次传输脚本](../experiments/results/M0-3-20261117T221327-3090/download_awq_curl.sh) · [续传启动脚本](../experiments/results/M0-3-20261117T221327-3090/fetch_and_probe.sh) · [校验日志](../experiments/results/M0-3-20261117T221327-3090/download-segments.log) |
| vLLM 安装与 metadata 预检 | 安装退出 0，实际 CUDA FP16 GEMM 通过。预检初次引用旧扩展名 `vllm._C` 报错，按实际 wheel 改检 `_C_stable_libtorch` 和 `_moe_C_stable_libtorch` 后通过。 | [安装命令](../experiments/results/M0-3-20261117T221327-3090/install_vllm.sh) · [依赖来源与哈希](../experiments/results/M0-3-20261117T221327-3090/install.json) · [失败预检](../experiments/results/M0-3-20261117T221327-3090/metadata-preflight.log) · [复测](../experiments/results/M0-3-20261117T221327-3090/metadata-preflight-v026.log) |
| 22:33:13–22:33:51 首次 AWQ 加载 | 退出 1。模型与 Marlin 加载成功，FlashInfer 采样预热执行 `ninja` 时 PATH 缺 venv bin。 | [首轮日志](../experiments/results/M0-3-20261117T221327-3090/trace01.log) |
| 22:34:25–22:34:56 短请求成功 | 退出 1。加入 venv PATH，为本次 greedy 测试禁用 FlashInfer 采样。真实短请求通过。原生 FP16 softmax 结果触发原先的 FP32 校验。 | [短请求记录](../experiments/results/M0-3-20261117T221327-3090/trace02/smoke.json) · [校验失败日志](../experiments/results/M0-3-20261117T221327-3090/trace02.log) |
| 22:37:43–22:38:06 完整采集 | 探针已生成全部 624 条。离线验证用 FP64 计算后直接舍入 FP16，两个接近 FP16 中点的概率触发严格判定，外层退出 1。离线检查改为 FP16 可表示值的舍入区间并计入 FP32 中间运算误差，原始数据复核通过。 | [失败日志](../experiments/results/M0-3-20261117T221327-3090/trace03.log) · [离线复测](../experiments/results/M0-3-20261117T221327-3090/trace03-verify-retest.log) |
| 22:39:59–22:40:23 最终重跑 | 退出 0。完整数据与前一次采集逐字节相同，云机和本地独立验证均通过。 | [最终运行脚本](../experiments/results/M0-3-20261117T221327-3090/run_probe_v3.sh) · [运行日志](../experiments/results/M0-3-20261117T221327-3090/trace04.log) · [本地复核](../experiments/results/M0-3-20261117T221327-3090/workspace-verify.log) |

## 5. 测试、数值与自查

- 24 项本地单测通过，覆盖已有 M0-2/M0-3 回归、AWQ 导出 token 坐标，以及 FP16 中点附近的独立校验。[单测日志](../experiments/results/M0-3-20261117T221327-3090/workspace-tests-final.log) · [命令与退出码](../experiments/results/M0-3-20261117T221327-3090/workspace-tests-final.command.json)
- 624 条记录中每条有 64 维有限 logits、64 个 softmax 概率、6 个不同且在 0–63 内的原生 Routed Expert ID，权重与对应概率一致。与 Python FP64 softmax 的最大误差为 0.000220982，符合 FP16 舍入。选中权重之和范围 0.156448–1.000057，末端轻微超过 1 属于 FP16 舍入。Shared 未进入 Top-6。
- 使用相同输入、单卡、相同采样/精度和同一引擎实例比较有无路由 hook。输出 token ID 一致（1640），完整 logits 最大绝对差 0。两个采集轮次的导出 Trace 逐字节一致。
- 运行日志记录 `auto_awq`、`MarlinLinearKernel`、`MarlinExperts` 和 `TRITON_MLA`，prefill backend 为 `FLASH_ATTN MLA`。退出时有 NCCL 清理警告，实测无遗留 GPU 计算进程，GPU 恢复空闲，云实例仍开机。[进程清理后的 GPU 日志](../experiments/results/M0-3-20261117T221327-3090/gpu-after.log)

## 6. 资源实测

| 指标 | 实测 |
|---|---:|
| 最终运行进程窗口（含初始化、两次请求与离线验证） | 24 秒 |
| 最终 vLLM 引擎初始化 | 14.05 秒 |
| 最终 vLLM 权重读取 | 3.87 秒 |
| vLLM 报告的模型加载显存 | 8.63 GiB |
| `nvidia-smi` 250 ms 采样峰值 | 10,595 MiB（约 10.35 GiB），92 个样本 |
| 四次运行窗口合计 | 116 秒 |
| 云机探查至最终完成（含依赖、传输与排障） | 约 27 分钟 |

运行采用单个固定提示词和最多 1 个输出 token。上述耗时保留为功能验证和资源记录，吞吐、TTFT、TPOT 和峰值精确上界仍待专门测量。[可复核统计](../experiments/results/M0-3-20261117T221327-3090/summary.json)

## 7. 未验证项

原版官方 BF16、GPTQ、自主 AWQ/GPTQ 量化、完整 decode、长上下文、多请求并发、任务质量及标准性能基线仍为 [U]。AWQ Trace 是量化后的路由证据。对原版 BF16 作结论前，按原计划补齐独立的原版实测。

## 8. 变更与范围

经用户批准启用指定实例并下载社区 AWQ 文件，使用已获允许的 M0-3 量化参考路径。保留原模型、研究问题和后续 BF16／INT4 对照口径。排障仅变更本次下载路径、独立运行环境、采样预热配置与校验精度口径。没有扩大训练或测试矩阵。

## 9. 验收清单

- [x] 连接用户指定云机、记录实际 GPU、驱动、CUDA、环境和权重来源。
- [x] 社区发布页的两个 AWQ 分片哈希核对通过，3090 完整模型加载与短请求通过。
- [x] 26 层 × 24 token 原生 Top-6 全量核验，模型 logits 插桩一致性通过。
- [x] 保留失败、修复、复测、退出码和用量日志，复核归档及源码 SHA-256。
- [ ] 官方原版 BF16 真实 Trace 仍待可运行环境，结论不由本 AWQ 数据代填。

## 10. 复现命令与下一步

在该云机运行已保留的最终探针。使用已有快照、已有隔离环境和新输出目录。

```bash
root=/root/autodl-tmp/p2-deepseek-moe
run="$root/results/M0-3-20261117T221327-3090"
snapshot="$root/models/TechxGenus-DeepSeek-V2-Lite-Chat-AWQ-e9fdf80"
attempt="repro-$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S)"
VLLM_USE_FLASHINFER_SAMPLER=0 PATH="$root/venv-vllm-0.26.0/bin:$PATH" \
  bash "$run/run_probe_v3.sh" "$root" "$run" "$snapshot" "$attempt" \
  > "$run/$attempt.log" 2>&1
```

在本地工作区可用 Python 标准库复核已归档的 Trace。

```powershell
python -B tests/verify_router_trace.py experiments/results/M0-3-20261117T221327-3090/trace04
```

等待对本阶段 AWQ 真实路径验收。后续阶段须以实际采集的数据来源为准。
