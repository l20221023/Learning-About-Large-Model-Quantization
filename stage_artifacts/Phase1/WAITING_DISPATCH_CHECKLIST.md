# Phase 1 待派发执行清单

**状态：WAITING_DISPATCH。** 本文件仅准备清单；未运行 Phase 1 实验、未占用 GPU。收到总调度会话明确派发，并完成下列前置复核后才执行。

## 1. 输入与证据边界

- [ ] 复核 `docs/PLAN.md`、`docs/P2_MASTER_RUNBOOK.md` Phase 1、M0-1 至 M0-5 报告及原始数据。M0-5 当前报告为 **BLOCKED**，须先完成真实 shadow forward、归档并由总调度复核通过。
- [ ] 固定官方模型 `deepseek-ai/DeepSeek-V2-Lite-Chat@85864749cd611b4353ce1decdb286193298f64c7`；固定已有社区 `TechxGenus/DeepSeek-V2-Lite-Chat-AWQ@e9fdf80ec6f1d200a029c41004bee40993763b43`。记录配置、分片哈希、tokenizer、Chat Template、许可证/使用条件和运行后端版本。
- [ ] 将社区 AWQ 标为公开 checkpoint 的实测路径，不标为本项目自主量化产物；官方 BF16 的 Header/架构证据不等于完整权重已可运行。

## 2. 派发前资源门禁

- [ ] 确认总调度派发、GPU 锁归属、单张 RTX 3090 剩余时段与实例可达性。先做 CPU 静态与小样本准备；获得锁后才做 GPU 工作。
- [ ] 重新实测系统盘与数据盘可用空间，按预计写入峰值执行存储守护：系统盘保留至少 8 GiB，数据盘至少 10 GiB；记录显存、主存、缓存及输出上界。
- [ ] 核对质量评估数据集的来源、许可、划分、去重/泄漏风险及本地可用性；不明项保留 UNKNOWN。
- [ ] 不下载官方 BF16 或新的大权重。若需要，先给出源文件、临时副本、输出与缓存的峰值估算及原因，取得用户授权后再执行。

## 3. 受控实验设计（派发后）

- [ ] 锁定可运行的模型身份、精度、后端、kernel、seed、提示与输出长度；保存完整配置和依赖版本。以已有 AWQ 做短请求正确性复测，核验实际 AWQ/Marlin 路径及 fallback。
- [ ] 审查自主 GPTQ W4A16 的算力、内存、磁盘、校准数据、group size、zero/scale、量化覆盖范围与时间可行性。不能满足单卡和存储约束时记录 BLOCKED 与所需替代资源，不启动转换。
- [ ] 质量：固定 tokenizer/Chat Template/EOS、held-out PPL 或等价指标，以及通用、中文、代码/推理任务；保存样本级输出、失败案例、聚合与不确定性。只有可运行的原始模型才做原始 vs AWQ 的同输入质量对照。
- [ ] 性能：先小矩阵再按资源扩展，区分 batch、并发及到达率；采集 TTFT、逐 token ITL/TPOT、E2E、tokens/s、P50/P95、显存 allocated/reserved/采样峰值、模型文件大小和实际输出 token 数。记录采样间隔及可解释范围。
- [ ] 公平性：BF16 若需 CPU offload 或不同 GPU，只用于质量/功能参照；不以其速度计算 3090 AWQ 加速比。缺失原始模型或四格中的任一格如实标 BLOCKED，不推断替代结果。

## 4. 独立交付与退出

- [ ] 全部 Phase 1 产物放在 `stage_artifacts/Phase1/`：输入清单、前置检查、资源预算、配置、逐条命令/起止时间/退出码、原始日志/数据与 SHA-256、聚合脚本、测试及复测记录、失败修复、资源消耗、未知项、阶段报告和下一步。
- [ ] 阶段报告逐项区分 MEASURED、SYNTHETIC、源码/配置证据与 UNKNOWN；给出可复现命令和质量门槛结论。结果可为受控基线，或因缺失模型/资源而为有依据的 BLOCKED。
- [ ] 不改共享 `docs/PLAN.md`、`docs/DECISIONS.md`、`docs/EVIDENCE_LEDGER.md`；阶段结束向总调度交付证据并等待后续指令。
