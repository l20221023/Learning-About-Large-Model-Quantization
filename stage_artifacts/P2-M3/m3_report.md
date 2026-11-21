# P2-M3 Router Trace 实测报告

状态：**COMPLETED**（本次定义的 TP=2 Router Trace、ON/OFF parity、开销和独立审计均通过）。本结果是固定 6 条项目提示的小规模 trace/开销验证，不是完整服务性能基线。

## 执行配置

- 模型：官方 DeepSeek-V2-Lite-Chat BF16 revision `85864749cd611b4353ce1decdb286193298f64c7`，复用服务器已有权重，没有下载或复制模型。
- vLLM：0.26.0；两张 RTX 3090；TP=2；BF16；`max_model_len=2304`；`max_num_seqs=8`；`gpu_memory_utilization=0.85`；eager；Triton Unquantized MoE / TritonExperts。
- Trace ON 使用 vLLM 内置 `--enable-return-routed-experts`；OFF 只移除该 flag。两个服务使用相同参数、环境、输入顺序和请求生成参数。
- vLLM 0.26.0 Model Runner V2 会拒绝 Routed Experts Capture，因此 ON/OFF 配对使用该版本内置 V1 runner。运行时 MoE 后端仍为 TritonExperts。没有编辑或 overlay vLLM 源码。
- 固定工作集：6 条 M0-4 项目自编提示；缓存 tokenizer 的 chat template 生成显式 prompt token IDs；共 175 个 prompt tokens。`temperature=0`、`top_p=1`、`seed=42`、`max_tokens=24`、`logprobs=5`、返回输出 token IDs。另有 1 条 smoke 和 2 次重复 prompt cache probe。
- Parity 门槛在正式请求前冻结：输出 token IDs 完全相同；每个位置的 top-5 logprob token 集合相同，绝对差 ≤`1e-5`。只比较文本不算通过。

## Trace 正确性

- 6 条正式提示得到 313 个实际处理 token rows（175 prefill + 138 decode）；输出 trace 为 8,138 条 token-row/layer JSONL 记录、48,828 个 Top-6 expert 选择。每条记录保留 6 个有序 expert ID。
- 内置响应数组形状为 `[processed_token_rows, 27, 6]`。第 0 层是 dense 层并保持零填充；第 1–26 层是 26 个 MoE 层。对每个返回 token/layer 检查 Top-6 长度、唯一性和 expert ID 范围 `[0,63]`。
- Prefill 行逐项对应冻结 prompt token IDs。Decode 行对应已生成且随后送入 forward 的输入 token，`predicts_output_index` 指向其预测的后一个输出 token。由于 max-token 结束时最后一个输出 token 没有再进入 forward，trace 行数按 `prompt_tokens + completion_tokens - 1` 核验。
- 每个 HTTP 请求只发一个 prompt；API response ID 与请求和 trace 一对一关联。vLLM 内部 scheduler request ID 不在返回协议中，明确记为 `UNKNOWN`。
- 内置 capturer 清空每步 device buffer，并由 scheduler 按实际 token 数和 KV slot mapping 切片；返回数组没有 padding 行。相同 prompt 的重复 ON trace prompt 路由完全相同；ON、OFF 各观测到 48 个 prefix-cache hit tokens。
- TP=2 输出只有一个逻辑路由数组，shape 中没有 TP 维，返回行数也没有乘以 2；因此审计只按每个 token/layer 的一次逻辑选择计数，不把 TP 副本重复计入。两张卡均初始化了 capturer。

## Parity 与采集开销

9 组对应请求（smoke、6 条正式提示、2 次 cache probe）的输出 token IDs 完全一致；top-5 logprob token 集合一致，最大绝对差 `0.0`；回答文本也一致。API 请求成功率：正式请求 ON `6/6`、OFF `6/6`，缓存探针各 `2/2`。

| 指标（正式固定集，N=6） | Trace OFF | Trace ON | ON−OFF |
| --- | ---: | ---: | ---: |
| vLLM TTFT 均值 | 60.208 ms | 60.814 ms | +0.606 ms（+1.01%） |
| vLLM request TPOT 均值 | 49.622 ms | 50.432 ms | +0.810 ms（+1.63%） |
| 客户端平均请求耗时 | 1,207.685 ms | 1,227.345 ms | +19.661 ms（+1.63%） |
| 聚合客户端吞吐 | 19.472 token/s | 19.158 token/s | −0.314 token/s（−1.61%） |
| GPU 峰值显存，GPU0 / GPU1 | 21,027 / 21,027 MiB | 21,027 / 21,027 MiB | 0 / 0 MiB（250 ms 采样） |
| 进程组 RSS 峰值合计 | 6,324.46 MiB | 6,311.63 MiB | −12.83 MiB |

TTFT/TPOT 来自 vLLM request histogram 的六请求增量 sum/count，不是 SSE 文本块推算。CPU 内存是 server process group 的 RSS 累加，共享页可能重复计数；−12.83 MiB 不代表采集降低了实际内存。ON 的延迟/吞吐差异是路由采集、scheduler 映射、返回数组序列化共同产生的采集开销。本轮没有量化或 MoE 算法变化。

## 额外观察与限制

- 与既有 BF16 TP=1 native-router trace 对照时，175 个 prompt token IDs 全部匹配，但 4,550 个 token/layer 路由集合中有 688 个 Top-6 ID 不同。该对照仅作诊断；本轮没有保存 TP=2 完整 router logits，不能把差异归因到特定内核舍入原因，也不声称跨 TP 路由一致。
- vLLM 公共 API 返回 response ID，不暴露内部 scheduler request ID。本轮单请求串行执行，因此 request/token 关联可复现；并发请求的内部 ID 关联未验证。
- 每个模式只做一轮固定 6 条正式请求；开销差异是小样本测量，不应替代 B0_TP2 或用于推断长期吞吐变化。

## 独立审计与证据

独立审计从原始 HTTP 响应重新解码 base64 NumPy 数组，逐项比对 `.npy`，重算路由数量/映射/ID、ON/OFF parity、prefix cache 与性能指标，结果 `PASS`、`errors=[]`。审计日志、route JSONL、原始响应、数组、run/server 日志和 SHA-256 均保存在本目录下。

- 配置：[m3_config.yaml](m3_config.yaml)
- 冻结输入：[workload.json](workload.json)、[workload_source.json](workload_source.json)
- 原始响应及二进制数组：[remote-runs/](remote-runs/)
- 逐 token/layer trace：[quality_trace.jsonl](quality_trace.jsonl)
- Prefix cache 路由：[prefix_cache_trace.jsonl](prefix_cache_trace.jsonl)
- 独立审计：[independent_audit.json](independent_audit.json)、[audit_m3.py](audit_m3.py)
- 源码审查：[built_in_routed_experts_capture.md](built_in_routed_experts_capture.md)、[vllm_source_sha256.txt](vllm_source_sha256.txt)、[source_diff.md](source_diff.md)
- 风险与命令：[profiling_risks.md](profiling_risks.md)、[run_commands.md](run_commands.md)
- 全目录哈希：[SHA256SUMS.txt](SHA256SUMS.txt)
