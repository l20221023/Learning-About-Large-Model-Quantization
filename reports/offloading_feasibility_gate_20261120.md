# Offloading Feasibility Gate — 资源预检

## 09:53 BF16 基线补充

- 双张 3090、vLLM 0.26.0、tensor parallel 2、BF16、原 Router 的固定 256-token 窗口前向成功，运行退出码 0，保存 255 个 target logprob。与官方 BF16 CPU eager 全量运行的首窗逐项核对：255/255 的位置与 token ID 相同；logprob 平均绝对差 0.0518613，最大绝对差 0.704801，平均带符号差 -0.00341185；首窗 PPL CPU 9.17465、双卡 9.20600。这是一个窗口的数值对照，未预先冻结数值容差，不能称逐项完全一致或完整精度验收。日志、源代码与退出码见 [TP2 运行](../stage_artifacts/OffloadGate/bf16_tp2_20261120_v2/)；运行结束时两卡释放。
- 单卡 prefetch：GPU 1、group size 2、每组卸载 1 层、prefetch step 1，初始化期间 FusedMoE 权重分配 352 MiB 时 OOM（剩余约 74 MiB），退出码 1；未执行前向。配置与原始日志见 [prefetch 失败记录](../stage_artifacts/OffloadGate/bf16_prefetch_20261120/)。此前 UVA 12/20 GiB 同在权重分配处 OOM；现有证据不足以断言所有卸载配置不可行。
- 双卡首轮启动因 PATH 缺少已有 `ninja` 退出 1；记录见 [失败运行](../stage_artifacts/OffloadGate/bf16_tp2_20261120_failed/)。加入现有 venv 的 bin 路径并以新 run 重试后退出 0。两轮及 prefetch 前检查 2 GiB 峰值和磁盘保留线，执行持 GPU 锁并使用存储 guard。
- 尚需定义与验证适合 BF16 后端数值差异的验收容差、扩展对照样本并复核原始证据；单卡卸载尚未成功。双卡 TP2 为权重并行基线，不是单卡卸载证明。

日期：2026-11-20 09:02（Asia/Shanghai）。**状态：BLOCKED，尚未执行本 gate 的 GPU 前向、decode trace 或缓存模拟。** 本 gate 不改变 P2 总计划或 Router。

09:07 更新：用户扩容后重新核验，数据盘总量 100 GiB、可用 59,374,223,360 bytes（约 55.3 GiB），系统盘可用 31,997,001,728 bytes，1 GiB 写入峰值及 8/10 GiB 保留线预检通过。实例现在枚举两张空闲 3090，本 gate 仍只授权使用其中一张，固定 GPU 0、tensor parallel 1。真实连续 decode 任务已派给独立会话；是否启动及结果以原始日志为准。状态改为 **IN PROGRESS**。

09:40 更新：用户已授权使用两张 GPU。独立会话已完成 GPU 0 的 AWQ 连续 decode trace：6 个自编提示，每条 32 个生成 token、31 个 decode step、26 层、每步 Top-6，共 4,836 行；原始 trace SHA-256 `f6c383f96dd3a44fecb4005605fb05c0d8e1baaff2350012015c49b1f090fb33`。用户侧独立会话报告插桩与基线输出一致，26/26 个远端文件哈希匹配；总调度已核对 trace 文件哈希，完整独立复核仍待完成。[自查与证据](../stage_artifacts/OffloadGate/decode/20261120T092306/SELF_CHECK.md)。

基于上述 AWQ 路由 trace 做初步离线模拟，假设每层独立缓存、每条请求冷启动、Top-6 原子访问、BF16 三个矩阵每专家 3×2048×1408×2 = 17,301,504 bytes，且每次 miss 均传输完整专家。容量每层 6/8/12/16/32/64 个专家时，LRU 命中数分别为 7,836/9,039/11,055/12,771/18,624/21,124，总访问 29,016；LFU 和未来信息策略及理论字节数见 [模拟结果](../stage_artifacts/OffloadGate/cache_simulation_awq_trace_bf16_bytes.json)。该计算使用 AWQ 路由假设 BF16 字节，不代表 BF16 实际路由或测得延迟；未来信息策略的最优性及模拟代码尚需独立验证。BF16 GPU 卸载基线尚未建立，整个 gate 仍未完成。

## 实例核验

保存的 SSH 主机密钥校验通过，主机名 `autodl-container-5fe44bab80-7925a7e9`。单张 RTX 3090：24,576 MiB，检查时 0 MiB、0%；驱动 580.105.08，CUDA 13.0。`free -h` 显示内存总计 755 GiB、可用 733 GiB，无 swap。系统盘总量 30 GiB，约 30 GiB 可用。

**数据盘当前总量仅 50 GiB，已用约 45 GiB，可用 5,687,132,160 bytes（约 5.30 GiB）**，低于用户指定的始终保留 10 GiB。此前阶段记录数据盘总量 150 GiB；容量变化原因尚未确认。挂载为 `/dev/nvme1n1` 上 AutoDL volume，XFS、项目配额。暂停远端写入和可能产生缓存的实验，等待用户核查容量；不删除数据或自行扩容。

官方 BF16 缓存存在，四个 safetensors 分片显示为 8.1、8.1、8.1、5.3 GiB，目录约 30 GiB；社区 AWQ 目录约 8.5 GiB。上述为目录存在性与大小检查，本次没有重新校验全部分片哈希。

## 已安装后端的代码核验

实例 vLLM 0.26.0 的 `vllm/config/offload.py` 定义 `auto`、`uva`、`prefetch` 三种 backend。UVA 的 `cpu_offload_gb` 按 GPU 设置 CPU 卸载容量，可用 `cpu_offload_params` 选择参数名段；prefetch 按层组设置 `offload_group_size`、`offload_num_in_group`、`offload_prefetch_step` 和 `offload_params`，代码描述为异步 H2D 预取。`auto` 优先选择已启用的 prefetch，否则选择 UVA，均未启用则 no-op。这是配置及源码确认，不代表 DeepSeek BF16 FusedMoE 在这两条路径已可运行或支持按专家动态缓存。既有 Phase 1 记录的两次 vLLM BF16 CPU offload 在 FusedMoE 权重分配 OOM；具体命令和原始证据见 Phase 1 报告。

已有官方 BF16 CPU eager 全量前向作为功能/质量基线，未改 Router；其 manifest 自报 PPL 9.96461，但 BF16/AWQ 全量配对审计的 bootstrap 尚未修复，不能视作本 gate 的 3090 卸载推理成功。

## 后续执行及证据门槛

1. 数据盘恢复到至少 10 GiB 保留线以上并满足本次写入峰值预检后，检查磁盘 guard、GPU 锁和 BF16 哈希；在单 3090 上对 UVA 和 prefetch 路径分别做小规模 BF16 原 Router 前向，记录实际配置、日志、GPU/CPU 内存、输出与 CPU baseline 的一致性以及失败原因。
2. M0-4 现有 trace 明确只覆盖 prefill；`scripts/trace_awq_3090.py` 的 router probe 假设每层只调用一次，不能直接当成连续 decode trace。需改用按请求、decode step、layer 和 token 记录的隔离探针，验证与未插桩生成结果一致、完整性及实际 token 对齐。标明 AWQ trace 的范围，不能冒充 BF16 trace。
3. 以真实 decode trace 为输入，定义缓存是否跨层/请求共享、初始冷缓存、同一步 Top-K 同时访问语义、容量单位、权重实际字节及 LFU 平局规则，模拟 LRU、LFU 和明确假设下的理想离线下界。报告每层/整体命中率、缺失专家与理论传输量；理想下界不代表可在线实现。不得将模拟字节或命中率称为实测传输延迟或 TPOT。

## 2026-11-20 17:45 superseding status update

The initial BLOCKED status above is historical and is superseded by completed evidence:

- Single-GPU V1 + UVA 20 GiB BF16 fixed-window forward succeeded; the fixed 255-target CPU/UVA comparison has matching token IDs but no pre-frozen numerical tolerance, so functional feasibility is established while numerical acceptance remains PARTIAL.
- A real BF16 continuous decode trace and independent cache simulation were completed. See [BF16 decode and cache report](bf16_uva_decode_and_cache_analysis.md) and [run self-check](../stage_artifacts/OffloadGate/BF16_decode_20261120T172330/SELF_CHECK.md).
- The decode trace is a six-prompt exploratory sample; simulated miss bytes are theoretical, not PCIe or latency measurements. The feasibility gate is not complete until the Task A numerical acceptance boundary is resolved. Dynamic cache implementation is not started.
