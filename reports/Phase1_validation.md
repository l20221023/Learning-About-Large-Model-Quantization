# Phase 1 量化基线阶段报告

日期：2026-11-18（Asia/Shanghai）

**状态：PARTIAL。** 已完成固定社区 AWQ checkpoint 在现有 RTX 3090 上的功能验证和小规模短请求性能先导。Phase 1 runbook 还要求 GPTQ W4A16 可行性/尝试、held-out 质量评估、质量不确定性与更完整的推理矩阵；这些项目仍未完成。因此不得将阶段标为 PASS，也不得派发依赖 Phase 1 的 Phase 2 GPU 实验。

## 实测范围

- 模型：`TechxGenus/DeepSeek-V2-Lite-Chat-AWQ@e9fdf80ec6f1d200a029c41004bee40993763b43`，已有公开 AWQ checkpoint；不是本项目生成的 GPTQ 量化产物。
- 环境：一张 NVIDIA RTX 3090 24 GiB，vLLM 0.26.0、PyTorch 2.11.0+cu130、CUDA 13.0。服务日志确认 `auto_awq`、`MarlinLinearKernel`、`AutoAWQMoEMethod`、MARLIN WNA16 MoE 与 MLA 路径。
- 正式先导配置：FP16、`max_model_len=256`、`max_num_batched_tokens=256`、`max_num_seqs=1`、512 MiB KV cache、关闭 prefix caching、eager、单卡；本地服务仅监听 `127.0.0.1`。参数由已安装 vLLM 的 `serve --help=all` 核对。
- 工作负载：项目自编通用英文、中文、代码三类各两条提示。6 条探索输出和 9 次性能请求均为 batch=1、顺序运行、最多生成 64 token。

## 运行结果

| Run | 结果 | 说明 |
|---|---|---|
| `Phase1-AWQ-20261118T082549-3090` | **FAIL，退出 1** | 15 请求中 8 个失败；首条触发服务端 MLA prefill 异常。错误请求不纳入任何成功指标。 |
| `Phase1-decode-probe-20261118T083507-3090` | PASS，退出 0 | 独立 in-process 2-token decode 功能探针。 |
| `Phase1-decode-probe-20261118T083555-3090` | PASS，退出 0 | 同一 run 完成 2-token 与 64-token decode。输出将 expert router 误解为网络设备路由器，进一步说明探针不能当质量通过。 |
| `Phase1-AWQ-20261118T083931-3090` | PASS，退出 0 | 正式短请求先导：15/15 成功，其中 6 条探索输出、9 条性能重复。 |

正式先导由原始 `raw.jsonl` 重算：

| 指标 | P50 | P95 | 样本/口径 |
|---|---:|---:|---|
| TTFT | 0.1173167 s | 0.1198383 s | 9 个串行性能请求，首个非空 SSE chunk |
| E2E | 3.5435852 s | 3.5740366 s | 同 9 个请求，从请求开始至流结束 |
| completion tokens / E2E | 中位数 18.0608 token/s | — | 9 个请求，每个 API usage 均报告 64 completion tokens |

ITL/TPOT 留空：SSE 文本块并不总是一块对应一个 token。9 次样本只是描述性先导，不能提供统计置信区间、并发吞吐或稳定服务保证。6 条探索输出没有评分，不构成准确率、PPL 或量化质量结论。

## 正确性、失败修复与资源

- 正式原始 JSONL 独立检查：15 个唯一请求坐标完整，全部 `status=ok`；逐条 `output_text` 与 chunk 文本拼接一致；请求时间和 chunk 时间顺序有效；usage 完整。分析脚本重算值与执行端 `summary.json` 一致。`raw.jsonl` SHA-256：`a5d1399d81873c1652595ea1730677c064cc37f307731c7fdb0f9d7c10f5b7a7`。
- 首轮失败栈在 `mla_attention.py` 前向中访问 `self.kv_b_proj.weight.dtype`，而 AWQ `ColumnParallelLinear` 没有 `.weight` 属性。调整成与已通过 in-process 探针一致的短序列、单序列、固定 KV cache 和无 prefix cache 配置后，后续运行成功。不同执行路径与多项配置同时变化，所以这里只记录关联，不宣称已证明首轮 500 的唯一根因。没有改写权重、伪造 `.weight` 或替换 backend。
- GPU CSV 250 ms 采样：正式先导峰值 10,595 MiB、采样利用率峰值 15%；结束时 `nvidia-smi` 显示 0 MiB。系统盘运行前后可用 31,998,570,496 B；数据盘从 37,557,579,776 B 降至 37,557,469,184 B。存储守护按新增峰值 1 GiB 预检，8 GiB 系统盘和 10 GiB 数据盘保留线均满足。
- 每个 run 均使用 `/root/autodl-tmp/p2-3090.lock` 与 `autodl_storage_guard.sh`；正式运行命令、环境日志、GPU CSV、前后磁盘和 GPU 快照、退出码及 SHA-256 均已归档。

## 复现与归档

在已核验的现有 AutoDL 实例上先确认锁、模型路径和磁盘余量，再执行：

```bash
cd /root/autodl-tmp/p2-phase1-deploy
bash ./run_decode_probe_autodl.sh
bash ./run_decode_probe_autodl.sh --extend-64
bash ./run_phase1_awq_autodl.sh
```

阶段 runner SHA-256：`b94aced8b7d459ef41461a2547a98375455122aa211c7bde58ac1d5f193b843d`。原始归档和压缩包 SHA-256：

- 首轮失败：[`Phase1-AWQ-20261118T082549-3090-failed.tar.gz`](../stage_artifacts/Phase1/evidence/Phase1-AWQ-20261118T082549-3090-failed.tar.gz) — `23b8a76dccff9d96c06b05b5595591b6c8f619916e1dbac8caf4d4d91e106301`
- Decode probes：[`Phase1-decode-probes-20261118.tar.gz`](../stage_artifacts/Phase1/evidence/Phase1-decode-probes-20261118.tar.gz) — `a218a35b9236e8b1839f630852cd6670d39a8fadc0703f13bf0f8022545e8db8`
- 正式先导：[`Phase1-AWQ-20261118T083931-3090.tar.gz`](../stage_artifacts/Phase1/evidence/Phase1-AWQ-20261118T083931-3090.tar.gz) — `414203d62ae69c282f31e3b757735472ddf937ba47ecde6f2338d7f35f4b5aff`

独立自查、逐条统计、环境、哈希及未验证事项见 [`stage_artifacts/Phase1/SELF_CHECK.md`](../stage_artifacts/Phase1/SELF_CHECK.md)。

## 未完成项与阶段出口

1. 缺少正式 held-out 语料、数据许可/去重核验、PPL 或等价评分、失败案例评分和质量置信范围；BF16 原始模型对照仍受存储门槛约束。
2. 未执行自主 GPTQ W4A16；未完成转换显存、峰值磁盘、校准数据与耗时的定量可行性审计。不得把社区 AWQ 说成 GPTQ 结果。
3. 尚无 batch、输入长度、长上下文、并发/到达率扩展矩阵；未验证 token 级 ITL/TPOT。
4. 结论只适用于本社区 AWQ checkpoint、当前 vLLM/Marlin 路径和小型自编请求集。

根据 `docs/P2_MASTER_RUNBOOK.md` 的状态定义，**PARTIAL 不能进入依赖阶段**。在上述核心 Phase 1 缺项完成，或总调度经授权正式修订研究范围并确认新的退出条件前，Phase 2 GPU 任务保持未派发。
