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
| 官方 BF16 完整预训练模型 Trace、GPTQ、量化质量及长上下文／并发性能 | [U] AWQ 实测覆盖短请求 prefill，暂未验证原版 BF16 或 GPTQ。 | [兼容矩阵](../reports/compatibility_matrix.md) |
| 同输入 Expert 输出相似度、加权贡献与组间差异 | [U] M0-5 profiler 尚无真实模型运行；本地数值 helper 与结果格式单测不构成模型测量 | [M0-5 自查与阻塞记录](../reports/M0-5_validation.md) |
| AutoDL 空间与缓存路径 | [R] `/` 30G free 约30G；`/root/autodl-tmp` 50G free 35G；安全保留线分别8/10 GiB。所有适用缓存变量已指向数据盘。完整官方BF16下载及转换峰值被安全预检拒绝。 | [存储审计报告](../reports/storage_audit_autodl_20261118.md) |
| SERE 公开实验的精确 checkpoint 和环境，Brownout Lite Chat 的直接实验 | [E/U] M0-1 | Phase 2 / Phase 3 核实 |
| 生成质量、校准后性能、显存、上下文长度和部署收益 | [U] | 后续基线及受控实验 |

## 2026-11-18 总调度现场核验

| 问题 | 当前证据 | 验证安排 |
|---|---|---|
| AutoDL 连接、主机身份与当前资源 | [R] 2026-11-18 00:58 Asia/Shanghai 成功交互登录。StrictHostKeyChecking=yes 使用保存条目且无变更警告；GPU 为 RTX 3090 24 GiB，空闲；`/` 可用 31,998,771,200 B，`/root/autodl-tmp` 可用 37,565,026,304 B；现存 AWQ 快照约 8.5 GiB。实例精确剩余时间未获得，本轮按用户此前约 5–6 小时估计规划。 | M0-5 runner 启动前及长任务期间重新测余量，并启用既有 guard。 |
| M0-5 shadow forward 正确性 | [U] 现有 runner 的候选列表只有 4 个专家，但 native router 是 Top-6，故旧实现的重构输入不完整。会话正在将 shadow 覆盖扩展为候选集合 ∪ native Top-6，并核查 `apply_model` worker 返回形式；尚无 3090 运行结果。 | 完成 API 探针与真实 AWQ 3090 reconstruction gate；失败须保留证据并修复复跑。 |
| M0-5 第一次 3090 API 探针 | [R/FAIL] 2026-11-18 01:07 左右使用单条 general 提示及单个候选专家真实加载 AWQ；生成阶段 native FusedMoE 调用触发 `ValueError: not enough values to unpack (expected 1, got 0)`，日志 `/root/autodl-tmp/logs/M0-5-api-probe-20261118T010700-3090/run.log`，退出码 1。Traceback 指向 PyTorch pre-hook 只接收 positional inputs，而 vLLM 调用通过 kwargs；没有导出任何相似度行。GPU 已释放，系统/数据盘均仍满足保留线。 | 用 `with_kwargs=True` 捕获并验证 hidden 输入，再重跑 API probe；失败日志保留，不能算作通过。 |
| M0-5 修复版与最终数据 | [R] 已核实 vLLM 0.26.0 `MoERunner.forward(hidden_states, router_logits, input_ids=None)`；补齐 `router_logits` keyword、Top-6 shadow 覆盖、kwarg pre-hook；当前 runner SHA `415172c373f735e96b86e5e17acb5b7670d60b32d84ede593bfa31cb84bdbc6f`。总调度独立复跑 py_compile 和 35 项 M0 回归通过。**修复版未部署/未实测**，无相似度输出。 | 实例访问恢复后先拉回失败日志，再部署固定 SHA 重跑小探针和完整门槛。 |
| AutoDL 当前可达性 | [R/UNKNOWN] 2026-11-18 07:24–07:25 同一保存 host key 下 TCP 36905 Connection refused，旧 SSH 会话已由远端关闭。失败日志留在实例数据盘，未拉回。原因可能为实例时长窗口结束；没有 dashboard 状态证据，不能断言实例具体电源状态。 | 用户决定恢复/延长同一实例后再核对状态与磁盘，不自动新增付费 GPU。 |
| 当前 GitHub `main` | [U] 本地 `git ls-remote --heads origin main` 未返回 ref；本地 HEAD 仍为首次提交 `def624fd53e18a4d230bd2ec7f592e23a2569b7b`。 | 网络恢复后查明远端 SHA，禁止强推；每次本地提交单独记录。 |
| 阶段调度与 GPU 串行 | [R] 已建立 M0-5、Phase 1–6 七个独立 Codex 会话；M0-5 当前 BLOCKED，Phase 1–6 为待派发队列，未并发占用 GPU。 | 总调度复核前序代码、日志、原始数据、复现命令、资源和限制后，逐个派发依赖阶段；保持单 3090 锁。 |
