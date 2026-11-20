# Offloading GPU 因果验证

## 2026-11-20 16:14 完成：V1 + UVA 20 GiB 单卡实测

**结论：PASS（有限功能与因果验证）；精度容差和长时间稳定性未验收。** 原先 GPU 占用的服务已释放；GPU 0 和 GPU 1 运行前均为 0 MiB，仅使用 GPU 0。单轮 V1 + UVA 成功加载官方 BF16 并完成固定 256 token 输入前向，退出码 0；结束后两卡均为 0 MiB。没有运行完整矩阵或 Prefetch：UVA 首要假设已得到直接验证，Prefetch 有无同样问题留待独立任务，不将 UVA 成功推及 Prefetch。

运行 `OffloadGate-causal-20261120T1614` 使用 `VLLM_USE_V2_MODEL_RUNNER=0`、`VLLM_ENABLE_V1_MULTIPROCESSING=0`、`CUDA_VISIBLE_DEVICES=0`、`offload_backend=uva`、`cpu_offload_gb=20`、TP=1、BF16、257 最大上下文与 512 MiB KV；逐项配置见 [config.json](../stage_artifacts/OffloadGate/causal_20261120T1614/result/config.json)。服务端实际日志记录 `Offloader set to UVAOffloader` 和 `Total CPU offloaded parameters: 20.51` GiB；事件记录的累计 offloaded bytes 为 22,017,126,400。`before_model` 事件是初始化前全局默认 `NoopOffloader`；26 个 expert 创建过程、模型加载后及前向后事件均为 `UVAOffloader`，不能将初始化前的 Noop 误解为权重构造时未启用卸载。`VLLM_USE_V2_MODEL_RUNNER=0` 为环境配置；实际 runner 类型由控制选项、安装版分支源码和 UVA 日志共同支持。日志中的 “V1 LLM engine” 单独不足以区分 V1/V2 model runner。

探针记录 26 次专家创建：w13 形状 `[64,2816,2048]`、w2 `[64,2048,1408]`、dtype `torch.bfloat16`，创建时 device 为 `cuda:0`。26 层共 52 个 w13/w2 参数的包装后记录中，37 项带 `_vllm_is_uva_offloaded` 标记，15 项未标记；后者可能因预算用尽仍驻留 GPU，不能把 52 项都宣称已卸载。UVA 模式为 true，标记代表已建立 CPU 存储的 UVA 访问路径，即使参数 device 显示 `cuda:0` 也不是单凭此字段判断物理驻留。各层记录保存在 [events.jsonl](../stage_artifacts/OffloadGate/causal_20261120T1614/result/events.jsonl)。

固定首窗 255 个目标 token 的 ID 与官方 BF16 CPU eager 原始结果 255/255 一致，前向生成 token ID 279。对数概率平均绝对差 0.051683、最大绝对差 0.507828；首窗 PPL CPU 9.174645、UVA 9.184654。数值容差未预先冻结，因此只能说输出完整且可比较，不能宣称逐 token 精度通过或位级相同；此窗口不代表全数据质量。

资源：PyTorch `max_memory_allocated` 事件峰值 10,288,803,328 bytes（约 9.58 GiB）；1 秒 `nvidia-smi` 采样峰值 10,361 MiB，采样可能漏掉瞬时峰值。进程 `ru_maxrss` 峰值约 39.57 GiB（不包括其他进程；取决于系统 RSS 计数），不等于系统内存全局峰值。最后一次 `after_forward` reserved 10,437,525,504 bytes、free 14,432,600,064 bytes；事件逐阶段记录 allocated/reserved/free。模型成功加载并前向；没有本轮 OOM。正常退出时日志有未显式调用 `destroy_process_group` 的警告，运行后显存归零，未观察到本轮残留 GPU 进程。

原始 [运行目录](../stage_artifacts/OffloadGate/causal_20261120T1614/) 含执行脚本、`run.log`、`exit`、配置、评分、逐层事件、前后磁盘/内存/GPU 快照、guard 输出和 GPU 采样。远端 SHA-256 列表中的 16 个文件已在本地重算全部匹配。运行前权重存在、此前固定 revision 分片校验通过；本轮未重算全部权重哈希。

复现命令（需目标卡真正空闲且遵守当前 GPU 锁和存储保留线；新 run ID，以下以本轮为例）：

```bash
env P2_RUN_DIR=/root/autodl-tmp/experiments/OffloadGate-causal-20261120T1614 P2_GPU_INDEX=0 \
  bash /root/autodl-tmp/experiments/OffloadGate-causal-20261120T1614/run_causal_v1_uva20.sh
```

脚本通过 flock 与现有 guard 后调用 vLLM 环境的 Python 运行 `causal_v1_uva20_probe.py`；再次执行须先部署脚本到新目录并更换 `P2_RUN_DIR`，避免覆盖证据。输入 ID 的 SHA-256 在脚本中固定核验。

**因果判断：**历史 V2 两次 UVA 20/12 GiB 的失败栈走 `NoopOffloader` 并在 FusedMoE 创建权重时 OOM；本轮固定模型、精度、输入和卸载预算下，V1 明确设置 `UVAOffloader`、记录实际迁移并成功前向。结合安装版源码中 V1 显式设置 offloader、历史 V2 栈进入 Noop，这是支持“历史 V2 路径没有启用预期卸载导致显存不足”的强证据；本轮并非严格消除所有潜在运行时差异，不能称已经证明 V2 的源码缺陷或所有 V2 配置都不可用。无需直接修改生产源码；若未来修复 V2，应在隔离环境核对 offloader 生命周期和实际迁移，并复测正确性、峰值和版本兼容性。

历史预检状态（15:52）：当时 **BLOCKED（目标 GPU 被现有任务占用）**。实例 `autodl-container-5fe44bab80-7925a7e9`，驱动 580.105.08、RTX 3090 两张，各 24,576 MiB。GPU 0 被现有 `VLLM::Worker_TP0` PID 10154 占用 21,070 MiB；GPU 1 被 `VLLM::Worker_TP1` PID 10155 占用 21,068 MiB。两者父进程均为 9937；0% 是采样瞬时利用率，不能视为显存空闲。当时未停止、重启或信号现有进程，也未开始本轮 GPU 前向。后续释放与成功运行见上文。

系统盘可用约 29.8 GiB，100 GiB 数据盘可用约 55.2 GiB；2 GiB 写入峰值、系统盘保留 8 GiB 和数据盘保留 10 GiB 的 guard 预检通过。内存可用约 704 GiB。官方 BF16 固定 revision 的四个 safetensors 分片及 vLLM 0.26.0 环境在；本轮未重算权重哈希。服务器连接使用已有 host key 严格校验。

已审查 CPU 审计与原始历史 traceback：V2 的三次卸载尝试在 `NoopOffloader.wrap_modules` 处展开模型构造并 OOM；V1 的源码显式设置 offloader。这支持本轮 V1 + UVA 20 GiB 作为因果验证，但尚不能宣称 V1 已执行、UVA 已生效或 BF16 单卡前向成功。

安装源码 `config/vllm.py:549–553` 使用 `VLLM_USE_V2_MODEL_RUNNER`；`engine/arg_utils.py` 包含 `offload_backend` 和 `cpu_offload_gb`，`LLM` 的 `**kwargs` 传入 EngineArgs。已审查旧矩阵含八个测试，不会直接运行；仅准备单轮 [causal_v1_uva20_probe.py](../stage_artifacts/OffloadGate/causal_v1_uva20_probe.py)，显式指定 V1、UVA 20 GiB、TP=1、固定 256-token 输入，记录 FusedMoE 参数构造、UVA 参数标记、内存和输出。UVA 视图可能显示 CUDA device，需结合 `_vllm_is_uva_offloaded` 和 offloader 累计字节而非仅以 device 名称判断 CPU 物理驻留。探针尚未运行，worker hook 覆盖及完整性有待验证。

以上为历史等待期间的计划；实际 offloader、参数迁移、资源、前向和固定窗口 CPU 对照见上文。更广泛的数值容差、长时间稳定性与 V1 Prefetch 均未验证。
