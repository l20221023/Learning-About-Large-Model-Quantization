# P2-M3 结果归档与 TP 路由差异 Closeout

日期：2026-11-20（Asia/Shanghai）

## 状态

- 原 M3 固定集 TP=2 Routed Experts Trace、ON/OFF Token ID 与 Logprob parity、开销对照和独立审计：**COMPLETED**，原始数据保持冻结。
- 本轮归档与哈希复核：**COMPLETED**。
- TP=1/TP=2 路由差异现象复算：**COMPLETED（描述性范围）**；最小受控 Prefill 对照与独立审计：**PASS**。
- 差异数值根因归因：**PARTIAL**。TP=2 router logits 不可得，历史执行配置不同，历史 safetensors shard 哈希未记录；因此没有宣称实现错误或已归因。

## 归档冻结

- 原 `stage_artifacts/P2-M3/SHA256SUMS.txt` 保留为 M3 原始证据快照清单，没有重写；验证 125 项全部匹配，原清单自身 SHA256 记录在 `archive_freeze_audit.json`。
- vLLM 0.26.0 M3 实际路径涉及的九个源码文件，已从原安装目录只读复制到 `source_snapshot/vllm/`。逐项 SHA256 与原 `vllm_source_sha256.txt` 完全匹配；无 Overlay 或源码修改。
- 新增受控实验响应、NPY 数组、逐 token/层路由、配置/资源/服务日志、执行脚本与独立审计位于 `tp_control/`。新分析脚本的输入文件 SHA256 位于 `tp_routing_difference_analysis.json`。
- `closeout_SHA256SUMS.txt` 是整个当前 P2-M3 目录和 `task_status/M3.md` 的补充冻结清单，共 195 个条目（194 个 P2-M3 文件加任务状态文件；manifest 不对自身哈希），覆盖新交付并把旧 manifest 作为被哈希文件纳入；生成后逐项验证，0 个缺失或哈希错误。它不替代原 `SHA256SUMS.txt`。

## 核心结论

1. 六条历史输入文本及完整 prompt token IDs 对齐；共 175 个 prefill tokens、4,550 个 token/layer 位置。TP=1 和 TP=2 原始 Top-6 有序数组有 688 行不同，但按 expert ID 集合复算为 388 行不同（8.53%）；300 行只是 Top-6 顺序不同。原 M3 文档中的 688 数字保留，但本 closeout 明确它是有序数组比较结果。
2. Decode 只比较连续相同生成历史：六条请求共 79 个可比 generated-input steps、2,054 token/layer 行，集合不同 178 行（8.67%）。另外 1,534 行位于生成历史分叉之后，1,248 行超出 TP=2 的生成 trace 长度；这些没有当作 route mismatch。
3. TP=1 raw BF16 router logits 显示集合不同样本的 top6-vs-top7 边界 margin 中位数 0.01172；集合相同样本中位数 0.08594。TP=2 不返回完整 router logits，因此只能说观察到边界接近相关性，不能说明 TP2 分数越过边界或确定由何种 reduction/kernel 引起。
4. 同一工作树、相同冻结提示 Token、两张卡空闲且项目 GPU 锁可用时，执行了一次 one-prompt Prefill 控制对照。两次均使用 vLLM 0.26.0 V1、BF16、TritonExperts、max_model_len=256、max_num_seqs=1、Prefix Cache disabled、512 MiB KV、UVA20 和相同生成参数；仅 TP=1/GPU0 与 TP=2/GPU0+GPU1 不同。702 个 token/layer 位置中，expert 集合不同 48 行（6.84%），有序数组不同 93 行（13.25%）。结果说明 cache/maxlen/seqs 这些历史配置差异不是全部现象的充分解释；单提示不代表总体发生率。
5. 控制试验两次都加载同一缓存目录/revision，config、model index、tokenizer、metadata SHA256 一致，输入 token IDs SHA256 一致。试验当时没有逐个 safetensors shard SHA256；历史 O2/M3 运行也没有 shard 哈希。因此严格的历史权重字节恒等仍未证明。

## O2 复用与研究影响

- O2 已有 TP=1 BF16/UVA 实测路由序列和 18 组离线 LRU/LFU/static-hot 仿真；O2 independent audit PASS，O2 acceptance replay 18/18 PASS。本轮未重跑该模拟器。
- 对单卡 Offloading，继续复用 O2 的 TP=1 trace、缓存模拟和审计结果；不能用 TP=2 M3 route 来替代，因为 TP 对照中确有路由集合差异。O2 仍是小型短提示离线模拟，不是 runtime 命中延迟或性能收益验证。
- 对 United Expert，可复用 O2 的有限路由/Top-K 权重数据做专家共现与分组探索；训练/蒸馏仍缺少 teacher activations、expert outputs、完整数值质量证据和 UE checkpoint。本轮没有启动蒸馏。
- 本轮未开始量化、缓存运行时开发或额外模型下载；没有修改 B0/Q1/O1 冻结结果、`project_status.md` 或共享计划/决策/台账。

## 复现入口与验证结果

- 离线 TP 路由重算：`python stage_artifacts/P2-M3/analyze_tp_routing.py`。
- 控制试验独立核验：`python stage_artifacts/P2-M3/tp_control/audit_tp_control.py`，结果 PASS、errors 为空。
- 受控远端运行脚本：`tp_control/run_tp_control_autodl.sh`，TP1/TP2 exit 均为 0；每次结束两张卡均 0 MiB/0% 利用率，进程清单无 vLLM/EngineCore/worker。
- 归档冻结和补充 SHA256 清单：`archive_audit.py`、`archive_freeze_audit.json`、`closeout_SHA256SUMS.txt`。

详细分层数据、位置分组、边界统计和 O2 缓存复用判断见 `tp_routing_difference_analysis.md`。本任务最终状态为 **PARTIAL（因果归因）**，不是 M3 Trace 采集未完成。
