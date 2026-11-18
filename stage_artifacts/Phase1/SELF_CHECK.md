# Phase 1 阶段自查（独立复核）

日期：2026-11-18（Asia/Shanghai）。本复核仅检查本地归档与脚本，不运行 GPU。**建议状态：PARTIAL / 探索性先导 PASS，Phase 1 正式基线未完成。** 已得到社区 AWQ 的 3090 短请求服务数据及 in-process decode 功能探针；未得到正式质量评估或可比的 BF16 对照。

## 证据范围与分类

- **MEASURED_COMMUNITY_AWQ**：固定社区模型 `TechxGenus/DeepSeek-V2-Lite-Chat-AWQ@e9fdf80ec6f1d200a029c41004bee40993763b43`，单张 RTX 3090，vLLM 0.26.0 / PyTorch 2.11.0+cu130 / CUDA 13.0。服务端日志标识 `quantization=auto_awq`、`AutoAWQMoEMethod`、MoE 的 Marlin 路径（`MarlinExperts`）及 AWQ Linear `MarlinLinearKernel`。这是公开 AWQ checkpoint 的路径验证与短请求先导，不是项目自主量化产物。
- **探索性输出**：15 个成功请求，6 条项目自编提示（general/chinese/code 各 2 条）用于观察模型输出；没有评分标准、标注集、基准集或人工盲评，不能称作正式质量分数、PPL 或质量达标结论。
- **MEASURED_COMMUNITY_AWQ decode 功能探针**：in-process `LLM.generate` 对 17-token 固定提示先生成 2 token，再在独立运行中生成 64 token；两次均完成。单请求耗时只用于功能记录，不用作正式性能统计。
- **SYNTHETIC**：本次没有合成数据参与实测聚合；先前 analyzer 的 CPU 合成统计检查只验证代码口径，不属于模型结果。
- **BLOCKED / UNKNOWN**：原始 BF16 权重的质量与性能对照、正式数据集及其许可/评分、PPL、ITL/TPOT、置信区间、长上下文、batch>1/并发与到达率性能矩阵、自主 GPTQ 的可行性实作均未完成。质量与 decode 正式门槛未满足。

## 原始 JSONL 独立重算

正式归档：`experiments/results/Phase1-AWQ-20261118T083931-3090/`。`raw.jsonl` 有效 JSON 行数 15，唯一键 `(phase, group, prompt_index, repeat)` 共 15；6 条 `quality` 与 9 条 `perf`，状态均 `ok`，无 500 错误请求进入数据。Quality 各类各 2 条，Performance 各类取第 1 条并重复 3 次。每行 `output_text` 都等于按序拼接 chunk 的 `text`。请求 usage 均存在，9 个性能请求 completion tokens 都为 64，prompt token 分别按 API 记录。

依据 `raw.jsonl` 重新计算的性能先导（9 个串行请求，非吞吐压测）：

| 指标 | P50 | P95 | 口径 |
|---|---:|---:|---|
| TTFT | 0.1173167 s | 0.1198383 s | 每请求首个非空 SSE chunk 的本地单调时钟偏移 |
| E2E | 3.5435852 s | 3.5740366 s | 请求开始至 SSE 流结束 |
| completion tokens / E2E | 中位数 18.0608 token/s | — | 64 个 API usage token 除以请求 E2E；9 个样本 |

重要口径限制：9 条里有 8 条 `chunk_count=63`、1 条 `chunk_count=64`，但 API `completion_tokens=64`。chunk 是文本片段，存在多 token 合并或片段与 token 数不一的证据。因此 chunk 间隔不能称 ITL/TPOT，chunk_count 也不能替代实际 token 数。P95 采用线性插值（`(n-1)*q`），与脚本 `percentile()` 一致。重复数仅 3 次/提示、总计 9 个性能请求，P50/P95 是描述性先导，不构成置信区间或稳定服务承诺。

独立复核命令（仓库根目录，退出码 0）：

```powershell
$py='C:\Users\li\.conda\envs\pytorch_clean\python.exe'
& $py -B -c "import json,statistics,pathlib,hashlib; p=pathlib.Path('experiments/results/Phase1-AWQ-20261118T083931-3090/raw.jsonl'); rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]; assert len(rows)==15; assert all(r['status']=='ok' and r['output_text']==''.join(c['text'] for c in r['chunks']) for r in rows); perf=[r for r in rows if r['phase']=='perf']; print('rows',len(rows),'ok',sum(r['status']=='ok' for r in rows),'perf',len(perf)); print('ttft_p50',statistics.median(r['ttft_s'] for r in perf),'e2e_p50',statistics.median(r['e2e_s'] for r in perf),'rate_p50',statistics.median(r['usage']['completion_tokens']/r['e2e_s'] for r in perf)); print('raw_sha256',hashlib.sha256(p.read_bytes()).hexdigest())"
```

独立结果：15/15 成功，chunk 拼接一致，TTFT P50 `0.1173166558` 秒、E2E P50 `3.5435852408` 秒、速率中位数 `18.0608044255` token/s。原始 JSONL SHA-256：`a5d1399d81873c1652595ea1730677c064cc37f307731c7fdb0f9d7c10f5b7a7`。归档 summary 的四项数值与重算一致。

## Probe、资源和命令核对

- `Phase1-decode-probe-20261118T083507-3090/`：退出 0；manifest PASS，2/2 token；模型身份、量化配置和运行时一致。
- `Phase1-decode-probe-20261118T083555-3090/`：退出 0；manifest PASS，2-token 与 64-token 步骤均完成。64-token 探针耗时约 3.3399 s；它使用不同调用/阶段且仅一次，不能并入 HTTP 性能统计。
- 正式服务：`command.txt` 记录 08:39:31–08:40:55（Asia/Shanghai），`flock -n /root/autodl-tmp/p2-3090.lock` 与 `autodl_storage_guard.sh run 1073741824`，退出 0。采集器及 analyzer 日志都标成功。服务只绑定 loopback。server log 有启动完成记录，模型与配置如上。
- GPU CSV 334 行逐行重算，250 ms 采样峰值 **10,595 MiB**、利用率峰值 **15%**；结束 `gpu-after.txt` 显示 GPU 0 MiB。此值是采样峰值，受采样间隔限制。
- 系统盘可用空间前后均为 31,998,570,496 bytes；数据盘从 37,557,579,776 降至 37,557,469,184 bytes，减少 110,592 bytes。守护预检记录 data peak 1,073,741,824 bytes、系统 peak 0，并满足 8/10 GiB 保留线。
- 两个 in-process probe 使用存储守护 268,435,456 bytes、GPU 锁；运行前后资源快照及显存采样均已归档。两个 manifest 均记录启动/结束时刻、运行环境、模型版本、量化配置及结果 token IDs。
- SHA-256 清单在各目录 `sha256.txt`。正式输入哈希记录于 `input-sha256.txt`：analyzer `e206e32f…271fa3`、collector `38ff1a81…180c893`、workload `c934f433…37319f0`。远端三项与本地冻结值一致。正式输出 raw JSONL、summary、server log、命令、退出码及资源清单均有 hashes。

复现主命令（总调度远端执行记录）：

```bash
cd /root/autodl-tmp/p2-phase1-deploy
bash ./run_phase1_awq_autodl.sh
```

decode 探针复现：

```bash
bash ./run_decode_probe_autodl.sh
bash ./run_decode_probe_autodl.sh --extend-64
```

采集脚本静态验证记录：Python `py_compile`、Shell `bash -n` 和 analyzer 合成样例检查均退出 0，证据见 `FROZEN_INPUTS.md` 与 `SERVER_FAILURE_AND_DECODE_PROBE.md`。本次复核仅重算本地原始结果，没有重跑 GPU。

## 失败与修复记录

1. `Phase1-AWQ-20261118T082549-3090`：OpenAI-compatible 服务请求返回 500、退出 1。总调度回报 traceback 在 MLA attention prefill 的 `kv_c_normed.to(self.kv_b_proj.weight.dtype)`，AWQ `ColumnParallelLinear` 无 `.weight`（提供 `qweight`）。失败目录原始日志须与转述一并保留；失败请求不纳入本轮 15 个成功请求。
2. 诊断控制：改用与此前成功 M0-5 in-process LLM 相同的关键短序列配置（max_model_len 256、max_num_batched_tokens 256、max_num_seqs 1、eager、prefix cache off、KV 512 MiB），先 2 token、再单独 run_id 64 token。两个探针均 PASS，证明此配置下模型能完成生成；并不能独自证明 HTTP 500 根因已完全定位。
3. 正式服务成功 run 使用更接近小序列的配置并在服务日志见 server startup complete、Marlin kernels。故障和成功之间不只一个参数变化且执行路径不同；不把关联说成因果证明。未改权重、未手工映射 qweight、未替换 backend。

## 阶段结论与未验证项

此阶段目前得到**单张 RTX 3090 上固定社区 AWQ 的短请求性能先导**，可复现的性能样本为 batch 1、三类短自编 prompt、每请求 64 completion tokens、顺序运行 9 次。它不是正式质量评测：输出未打分。没有 BF16 对照、没有 ITL/TPOT、没有置信区间，也没有长上下文、批处理或并发矩阵。模型质量、任务准确率、PPL、用户许可数据集和模型质量门槛均 UNKNOWN。官方 BF16 因整权重下载及存储门槛仍 BLOCKED；GPTQ 自主量化没有执行或定量资源审计。

建议阶段状态：**PARTIAL**。保留社区 AWQ 受控先导为 MEASURED，确认本地原始数据、运行退出码和哈希完整；在正式质量数据与许可、质量门槛、需要的性能扩展及 BF16 可行对照明确以前，不标 Phase 1 完成，也不宣称 AWQ 加速 BF16。
