# P2-O2 BF16 UVA Decode 与 Expert Cache

状态：**PARTIAL，Task B/C 已完成自查；Task A 的输出质量验收未完成**。环境：官方 DeepSeek-V2-Lite-Chat BF16 revision `85864749cd611b4353ce1decdb286193298f64c7`，vLLM 0.26.0，单张 RTX 3090，V1 model runner，UVA 20 GiB，TP=1。此前固定窗口 BF16 CPU/UVA 对照未冻结逐 token 容差；本阶段不擅自设定或放宽容差。

## A. 未插桩 Decode 基线

`P2O2-benchmark-20261120T1629` 复用 Phase1 自编 M0-4 workload：general/chinese/code 各第一条，greedy、输出上限 64、每条 3 次，顺序请求。服务固定单卡 UVA 和 512 MiB KV；退出 0，9/9 请求成功，均以长度上限结束，因此不能据此评判回答完整性。初始化从 16:29:06 至服务健康 16:30:18，约 72 s（包含启动与健康检查）；请求运行至 16:31:46 左右，结束清理至 16:31:51。初始化不是 prefill 或 decode 时间。

| 指标 | 实测先导值 |
|---|---:|
| 首个非空文本块时间 TTFT P50 / P95 | 0.7717 / 0.9324 s |
| E2E P50 / P95 | 9.7387 / 9.8737 s |
| 实际 completion tokens / E2E 中位数 | 6.5717 token/s |
| vLLM prefill histogram 单请求均值 | 0.7701 s |
| vLLM decode histogram 单请求均值 | 8.9599 s |
| vLLM inter-token histogram 总和 / 567 个事件 | 约 0.14222 s/事件 |
| vLLM request TPOT histogram 9 个请求均值 | 约 0.14222 s/token |
| 250ms GPU 显存采样峰值 | 10,399 MiB |

每个请求的 ITL 和 TPOT histogram 计数分别与输出 token 数减一、完成请求数核对一致，prefill/decode 计数亦为每请求 1；百分位只能报告桶区间，逐请求快照与审计结果保存在 [运行目录](../stage_artifacts/OffloadGate/P2O2-benchmark-20261120T1629/)。上述是 9 请求的小样本，vLLM histogram 中 prefill/decode 为服务端计时，TTFT 为客户端首个非空文本块时间；SSE 文本块不一定对应单个 token。CPU→GPU 实际传输字节与耗时 **UNKNOWN**。GPU 时间序列、server RSS 与系统内存快照已保存，但 server RSS 不包括独立 engine 子进程，不能当成全模型 CPU 峰值。

执行命令为 `env P2_RUN_DIR=/root/autodl-tmp/experiments/P2O2-benchmark-20261120T1629 bash /root/autodl-tmp/experiments/P2O2-benchmark-20261120T1629/run_bf16_uva_benchmark.sh`；具体 server 命令和环境见运行目录 `inner.sh`、wrapper 和 `server.log`。采集器复用 [Phase1 stream collector](../stage_artifacts/Phase1/collect_awq_baseline.py)，审计器 [audit_bf16_uva_benchmark.py](../stage_artifacts/OffloadGate/audit_bf16_uva_benchmark.py)；远端清单 43 个文件本地 SHA-256 全部匹配。固定首窗 BF16 CPU/UVA 255 token 对照与未冻结容差见因果报告；本组 9 个回答未执行独立 CPU 数值验收，因此不得声称完整质量通过。

## B. BF16 连续 Decode Expert Trace

状态：**PASS（本阶段范围）**。成功运行 `OffloadGate-BF16-decode-20261120T172330`，固定官方 revision、vLLM 0.26.0、V1 `GPUModelRunner`、TP=1、物理 GPU0、BF16、UVA20 GiB、512 MiB KV 和 M0-4 六条冻结提示。模型使用 Triton unquantized MoE；运行时记录 `UVAOffloader` 和 CPU offload 22,017,126,400 bytes。未下载或转换模型。

4-token smoke 的插桩/未插桩 token ID、文本和结束原因一致；六条正式请求各输出 32 tokens、31 decode steps，三项输出 parity 全部一致。覆盖 26 层、64 experts、native Top-6；prefill 4,550 rows，decode 4,836 rows。trace SHA-256 `dbe9920be8ecbfa8b6828e12d8c8921a53933c001b5ae02b001693b6f7a66993`。GPU0 250 ms 峰值 10,399 MiB、采样利用率峰值 100%；GPU1 未使用。模型退出后双卡均为 0 MiB，双锁释放，2 GiB storage guard 复核通过，退出码 0。

有插桩运行耗时合计 37.0803 s，未插桩 30.8609 s，插桩额外 6.2194 s（20.15%）；这是捕获 logits、CPU 拷贝和路由校验产生的开销，不是未插桩 TPOT。

独立审计确认 trace 行覆盖、输入 token 对齐、Top-6 cutoff、parity 及文件哈希。一个 BF16 softmax 值存在跨 PyTorch build 的单 ULP 差异：服务器 v2.11.0+cu130 对原始 27×64 prefill batch 重算与保存的 native 权重一致；本机 v2.12.0+CPU 重算在 layer 25、sequence index 10、expert 21 得到相邻 BF16 值，绝对差 `0.0001220703125`。float64 参考值为 `0.02899169904408871`，位于舍入 midpoint `0.02899169921875` 下方约 `1.75e-10`。native weight 与 trace 保存的 selected probability 全行完全相同，Top-6 cutoff 通过；本报告不声称跨构建软最大值逐项精确一致。

runner 检查了逐层 `UnquantizedFusedMoEMethod` 和非 monolithic 路径，但最终 manifest 用整数层 ID 覆盖了初始逐层 inventory 对象，因此该对象明细没有持久化；自查对此作了限制披露，没有补造字段。四次早期失败保存在 `163117`、`164537`、`164804`、`164959` 目录，含启动前缺少 `generation_config.json`、wrapper 未导出 `PY`、模型 OOM（当时另一个任务占卡）等；`172020` 的 smoke 暴露 hook 未导入 torch，修复仅为模块级 import 并用新 run 成功。主证据和自查见 [Task B 目录](../stage_artifacts/OffloadGate/BF16_decode_20261120T172330/)；退出后31个文件中30项首轮哈希一致，继续写入的 `bootstrap.log` 在完成后重哈希也一致。

## C. Expert Cache 模拟

状态：**PASS（离线模拟范围）**。使用真实 decode trace，trace hash 与 B 节一致；每专家理论字节为 17,301,504（3×2048×1408 BF16 元素）。模型假设：每层独立 cache、每请求冷启动、Top-6 同步访问、expert 大小相同；static hot 使用 leave-one-request-out，预加载字节计入总量。下表中的 H/M 为命中/缺失访问总数，bytes 为理论传输量，不是测量到的 PCIe 数据量或延迟。

| 每层容量 | LRU H/M；理论 bytes | LFU H/M；理论 bytes | Static hot H/M（预加载 experts）；理论 bytes |
|---:|---:|---:|---:|
| 6 | 7,405 / 21,611；373,902,802,944 | 7,405 / 21,611；373,902,802,944 | 3,530 / 25,486（936）；457,140,338,688 |
| 8 | 8,685 / 20,331；351,756,877,824 | 9,121 / 19,895；344,213,422,080 | 4,645 / 24,371（1,248）；443,247,230,976 |
| 12 | 10,952 / 18,064；312,534,368,256 | 11,441 / 17,575；304,073,932,800 | 6,977 / 22,039（1,872）；413,696,262,144 |
| 16 | 12,974 / 16,042；277,550,727,168 | 13,329 / 15,687；271,408,693,248 | 9,167 / 19,849（2,496）；386,602,106,880 |
| 32 | 18,737 / 10,279；177,842,159,616 | 18,772 / 10,244；177,236,606,976 | 17,276 / 11,740（4,992）；289,488,764,928 |
| 64 | 21,136 / 7,880；136,335,851,520 | 21,136 / 7,880；136,335,851,520 | 28,963 / 53（9,920）；172,547,899,392 |

18 个策略/容量单元的总命中、缺失、逐层分布、预加载 expert 数及 bytes 已由独立 CPU 实现重算一致；细节见 [cache simulation](../stage_artifacts/OffloadGate/BF16_decode_20261120T172330/cache_simulation.json) 与 [独立审计](../stage_artifacts/OffloadGate/BF16_decode_20261120T172330/independent_audit.json)。结果依赖小样本六提示和冷 cache 设定；不外推在线 cache 延迟或跨请求持久化效果。
