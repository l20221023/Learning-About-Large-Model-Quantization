# P2 证据台账

沿用 `[C]` 配置、`[S]` 源码、`[I]` 索引、`[D]` 推导、`[H]` 实际 Header、`[R]` 项目实测、`[E]` 他人论文、`[U]` 未知。

| 问题 | 当前证据 | 验证安排 |
|---|---|---|
| 27 层、Dense 层 0、MoE 层 1–26、64 Routed、greedy Top-6 和 shared 宽度 2816 | [C/S] M0-1 已验收 | M0-2 核对张量映射，后续验证真实执行 |
| 固定版本各权重的分片、shape、dtype、offset | [I/D/H/R] M0-2 实读四个官方 Header，5,291 个张量全部匹配，均为 BF16；偏移范围合计 31,412,968,448 字节 | [映射表](../experiments/results/M0-2-20261117T0926335375-official/audit/expert_weight_mapping.csv) · [请求与 Header 记录](../experiments/results/M0-2-20261117T0926335375-official/audit/manifest.json) |
| 真实 checkpoint 的单专家数据读取、专家张量是否数值相同 | [U] | 已有本地 shard 时按偏移读取，数值对照留待后续 |
| HF 原生 BF16 eager loader、MLA 模块、MoEGate 及 Top-6 Trace | [S/R] M0-3 官方源码 + 随机 BF16 的三分片合成 checkpoint 通过，15 条记录，插桩前后完整 logits 最大差异 0。26 个路由层的对齐另以合成单元测试验证。 | [M0-3 验收](../reports/M0-3_validation.md) |
| 官方 tokenizer 和 Chat Template | [R] 固定文件生成 25 个 token，过滤后 24 个普通 token | [Manifest](../experiments/results/M0-3-20261117T0951529190-native-final/trace/manifest.json) |
| RTX 3090 上社区 AWQ checkpoint 的预训练模型加载和 Trace | [R] vLLM 0.26.0，Marlin WNA16 MoE，624 条原生路由记录，插桩前后 logits 最大差异 0。显存 250 ms 采样峰值 10,595 MiB。 | [3090 验收报告](../reports/M0-3_3090_AWQ_validation.md) |
| 路由频率、负载、共激活和 Shared 调用 | [R] 社区 AWQ、三类自编小负载、6 prompts、169 个有效 token；26 层全部通过每层有效 token×6 与共激活上三角 token×15 不变量。Shared 模块 hook 156/156 次完整覆盖。探索性数据，不代表真实线上分布。 | [M0-4 自查](../reports/M0-4_validation.md) · [原始 trace 与聚合](../experiments/results/M0-4-20261117T235424-3090/) |
| 社区 AWQ 短请求功能与性能先导 | [R] 固定 AWQ revision、3090、vLLM 0.26.0；短配置下 15/15 请求成功，6 条未评分探索输出、9 次串行 batch=1×64-token 性能请求。TTFT P50/P95 0.1173/0.1198 s，E2E 3.5436/3.5740 s，completion tokens/E2E 中位数 18.0608 token/s；GPU 250 ms 采样峰值 10,595 MiB。仅探索先导，非正式质量或 BF16 对照。 | [Phase1 报告](../reports/Phase1_validation.md) · [自查](../stage_artifacts/Phase1/SELF_CHECK.md) · [原始证据包](../stage_artifacts/Phase1/evidence/) |
| Phase 1 正式量化/质量完整基线 | [U/BLOCKED] 无 held-out 数据及许可核实、PPL/评分与置信范围；未执行自主 GPTQ 量化；无 batch/长度/并发扩展矩阵或 token 级 ITL/TPOT。Phase1 状态 PARTIAL。 | 完成 Phase1 退出项前不派发依赖的 Phase2 GPU 工作；BF16 权重下载仍需空间峰值审计与授权。 |
| 官方 BF16 完整预训练模型 Trace、GPTQ、量化质量及长上下文／并发性能 | [U] AWQ 实测覆盖短请求 prefill，暂未验证原版 BF16 或 GPTQ。 | [兼容矩阵](../reports/compatibility_matrix.md) |
| 同输入 Expert 输出相似度、加权贡献与组间差异 | [R] 社区 AWQ INT4 3090，第 13 层、两条自编提示、4 个 token；4/4 原生 Top-6 shadow 重构通过，16 条候选记录、24 条专家对记录。最大重构 relative L2 为 0.0002960364。仅小样本探索结果。 | [小型原始证据包](../stage_artifacts/M0-5/evidence.tar.gz) · [独立复核](../stage_artifacts/M0-5/FINAL_AUDIT.md) · [聚合 CSV](../stage_artifacts/M0-5/verified/) |
| AutoDL 空间与缓存路径 | [R] `/` 30G free 约30G；`/root/autodl-tmp` 50G free 35G；安全保留线分别8/10 GiB。所有适用缓存变量已指向数据盘。完整官方BF16下载及转换峰值被安全预检拒绝。 | [存储审计报告](../reports/storage_audit_autodl_20261118.md) |
| SERE 公开实验的精确 checkpoint 和环境，Brownout Lite Chat 的直接实验 | [E/U] M0-1 | Phase 2 / Phase 3 核实 |
| 生成质量、校准后性能、显存、上下文长度和部署收益 | [U] | 后续基线及受控实验 |

## 2026-11-18 总调度现场核验

## 2026-11-18 Phase 1 AWQ PPL 补测

- [R] 现有 RTX 3090、vLLM 0.26.0、社区 AWQ；WikiText-2 raw 配置 test Parquet 固定 `Salesforce/wikitext@b08601e04326c79dfdd32d625aee71d232d685c3`，文件 SHA-256 `5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`。原 S3 ZIP 不可获取，两个表示形式的逐字节同一性 UNKNOWN。
- [R] `Phase1-PPL-full-20261118T095538-3090` 退出 0；303,877/303,877 token 唯一覆盖，PPL 13.40055990877775；原始分数重算一致。归档 SHA-256 `e8da033682c76af9d242822e2cc0f3919e1678cd3bbdd2ee871eb7fd5bf3ac8f`，见 [Phase1 PPL 原始包](../stage_artifacts/Phase1/evidence/Phase1-PPL-full-20261118T095538-3090.tar.gz) 及 [验收报告](../reports/Phase1_validation.md)。BF16 相对质量、GPTQ、扩展性能与正式质量置信范围仍 UNKNOWN；阶段 PARTIAL。

| 问题 | 当前证据 | 验证安排 |
|---|---|---|
| AutoDL 连接、主机身份与当前资源 | [R] 2026-11-18 00:58 Asia/Shanghai 成功交互登录。StrictHostKeyChecking=yes 使用保存条目且无变更警告；GPU 为 RTX 3090 24 GiB，空闲；`/` 可用 31,998,771,200 B，`/root/autodl-tmp` 可用 37,565,026,304 B；现存 AWQ 快照约 8.5 GiB。实例精确剩余时间未获得，本轮按用户此前约 5–6 小时估计规划。 | M0-5 runner 启动前及长任务期间重新测余量，并启用既有 guard。 |
| M0-5 shadow forward 正确性 | [U] 现有 runner 的候选列表只有 4 个专家，但 native router 是 Top-6，故旧实现的重构输入不完整。会话正在将 shadow 覆盖扩展为候选集合 ∪ native Top-6，并核查 `apply_model` worker 返回形式；尚无 3090 运行结果。 | 完成 API 探针与真实 AWQ 3090 reconstruction gate；失败须保留证据并修复复跑。 |
| M0-5 第一次 3090 API 探针 | [R/FAIL] 2026-11-18 01:07 左右使用单条 general 提示及单个候选专家真实加载 AWQ；生成阶段 native FusedMoE 调用触发 `ValueError: not enough values to unpack (expected 1, got 0)`，日志 `/root/autodl-tmp/logs/M0-5-api-probe-20261118T010700-3090/run.log`，退出码 1。Traceback 指向 PyTorch pre-hook 只接收 positional inputs，而 vLLM 调用通过 kwargs；没有导出任何相似度行。GPU 已释放，系统/数据盘均仍满足保留线。 | 用 `with_kwargs=True` 捕获并验证 hidden 输入，再重跑 API probe；失败日志保留，不能算作通过。 |
| M0-5 修复版与最终数据 | [R] 已核实 vLLM 0.26.0 `MoERunner.forward(hidden_states, router_logits, input_ids=None)`；补齐 `router_logits` keyword、Top-6 shadow 覆盖、kwarg pre-hook；当前 runner SHA `415172c373f735e96b86e5e17acb5b7670d60b32d84ede593bfa31cb84bdbc6f`。总调度独立复跑 py_compile 和 35 项 M0 回归通过。**修复版未部署/未实测**，无相似度输出。 | 实例访问恢复后先拉回失败日志，再部署固定 SHA 重跑小探针和完整门槛。 |
| AutoDL 当前可达性 | [R/UNKNOWN] 2026-11-18 07:24–07:25 同一保存 host key 下 TCP 36905 Connection refused，旧 SSH 会话已由远端关闭。失败日志留在实例数据盘，未拉回。原因可能为实例时长窗口结束；没有 dashboard 状态证据，不能断言实例具体电源状态。 | 用户决定恢复/延长同一实例后再核对状态与磁盘，不自动新增付费 GPU。 |
| 当前 GitHub `main` | [U] 本地 M0-5 提交为 `2f3144a8829f77084819374dfcbe6eb28a54c414`（父提交为首次基线 `def624fd53e18a4d230bd2ec7f592e23a2569b7b`）。本轮 `git ls-remote` 和普通 `git push origin main` 未得到可确认的 ref/成功响应；远端 SHA 与发布状态未知。 | 网络恢复后核查远端 main；若推进则仅普通 fast-forward push，禁止 force push。 |
| 阶段调度与 GPU 串行 | [R] 已建立 M0-5、Phase 1–6 七个独立 Codex 会话；M0-5 当前 BLOCKED，Phase 1–6 为待派发队列，未并发占用 GPU。 | 总调度复核前序代码、日志、原始数据、复现命令、资源和限制后，逐个派发依赖阶段；保持单 3090 锁。 |

2026-11-18 08:19 接续更新：实例已重新可达，3090 空闲；M0-5 正式运行与独立复核均 PASS，前述 07:24 的连接阻塞为历史状态。Phase 1 已派发。M0-5 证据包 SHA-256 为 `61d7662d8bd69dd332c16f5451948251f6f7e27ac982d36ac865dee94d35c745`；逐元素 allclose 只在 GPU 端判定，原始 JSON 未保存输出向量，因此离线不能重算该逐元素门槛。

## 2026-11-18 Phase 1 3090 实测复核

- SSH StrictHostKeyChecking=yes 成功复连同一实例，主机密钥未变化；登录时单卡 RTX 3090 空闲，`/` 可用 31,998,705,664 B，`/root/autodl-tmp` 可用 37,557,784,576 B。AutoDL banner 未给精确剩余时长；用户确认时长足够。所有 GPU run 均使用单实例 `/root/autodl-tmp/p2-3090.lock` 与存储守护。
- in-process 2-token 和 64-token decode probe 分别 PASS；两者只证明相应配置下可生成，不能视为质量结论或正式性能样本。`Phase1-AWQ-20261118T082549-3090` 失败退出 1，15 个服务请求 8 个失败，原始失败包保留。
- `Phase1-AWQ-20261118T083931-3090` 退出 0，15/15 请求成功。9 个 batch=1 串行性能样本为 64 tokens：TTFT P50/P95 0.1173167/0.1198383 s，E2E 3.5435852/3.5740366 s，completion-token/E2E 中位数 18.0608 token/s。GPU 250 ms 采样峰值 10,595 MiB/15%，结束 0 MiB；系统盘保持约 31,998,570,496 B，数据盘由 37,557,579,776 B 减至 37,557,469,184 B。
- 正式输出是六条自编提示的未评分观察；held-out/PPL、许可及置信范围、官方 BF16 对照、自主 GPTQ、扩展 batch/长度/并发矩阵和 ITL/TPOT 均 UNKNOWN/BLOCKED。Phase1 状态 PARTIAL；依据 runbook 状态定义，依赖阶段 Phase2 暂不派发。原始证据索引与阶段报告见 [Phase1 验收报告](../reports/Phase1_validation.md)、[阶段自查](../stage_artifacts/Phase1/SELF_CHECK.md) 及压缩包 SHA 列表。
## 2026-11-20 Offloading Feasibility Gate 补充

- [R] 官方 BF16 固定 revision、vLLM 0.26.0 V1、单张 RTX 3090、TP=1、UVA 20 GiB 成功完成 256-token 输入前向；固定 CPU eager 对照的 255/255 target token ID 对齐，logprob MAE 0.051683、最大绝对差 0.507828，首窗 PPL CPU 9.174645、UVA 9.184654。数值容差未预先冻结，Task A 功能 PASS、数值/质量验收 PARTIAL。证据：[offloading_gpu_causal_validation.md](../reports/offloading_gpu_causal_validation.md)、[bf16_uva_decode_and_cache_analysis.md](../reports/bf16_uva_decode_and_cache_analysis.md)。
- [R] BF16 连续 decode trace：6 个冻结提示，每条 32 个输出 token/31 decode steps，26 层 Top-6，4,836 decode rows；插桩与基线 token IDs、文本和 finish reason 相符。Trace SHA-256 dbe9920be8ecbfa8b6828e12d8c8921a53933c001b5ae02b001693b6f7a66993。限定为小样本实际 trace，不代表一般流量分布。
- [R/D] LRU、LFU 与 static-hot 在每层容量 6/8/12/16/32/64 的 18 组结果由独立 CPU 实现复算一致。每 expert 字节数为 17,301,504 的等大小假设；miss bytes 是理论值，不是 PCIe 传输量或延迟实测。原始 trace、模拟结果和独立审计见 stage_artifacts/OffloadGate/BF16_decode_20261120T172330/。
- [R] 2026-11-20 17:44:53 远端退出快照显示两卡 0 MiB、项目锁空闲、磁盘 guard 通过；该资源快照是历史观测，不作为当前在线状态。
