# P2｜DeepSeek-V2-Lite 量化与 MoE 结构优化：Codex 总执行指令

> **文档版本**：v1.0 · 2026-11-17  
> **用途**：交给 Codex 作为长期执行计划、阶段任务编排、工程质量门禁与自查报告规范。  
> **定位**：工程与实验总计划，允许在证据支持下调整技术实现；**不是已经完成的实验结果，也不是必须一次执行完毕的脚本**。  
> **项目名称**：P2 — LLM Quantization & MoE Structural Optimization  
> **首选模型**：`deepseek-ai/DeepSeek-V2-Lite-Chat`  
> **必读输入**：`DeepSeek-V2-Lite-Chat_P2-M0-1_正式复核报告(1).md`（或用户交付的内容相同的文件）  
> **与 Stage D 的关系**：**完全独立的项目**。Stage D 研究 vLLM Scheduler / KV Cache / Admission；本项目研究模型数值表示、MoE 专家结构及两者的组合。禁止私自混合两项目代码和结果。

---

## 0. 给 Codex 的最高优先级执行规则

你是本项目的执行工程师与实验自动化代理。你的任务不是凭空宣称提出新算法，而是把可验证的设计落地成**代码、测试、日志、数据和报告**，供用户与技术审查者验收。

1. **先阅读再执行**：首先查找并完整阅读本总计划和 P2-M0-1 正式复核报告。不得假装读取不存在的文件。报告中的已确认静态事实是当前审查依据；对实际运行、权重 Header、后端兼容性和性能质量的 UNKNOWN 必须保持 UNKNOWN，直到实测。
2. **阶段内自主**：在当前阶段内可自行阅读公开源码、编写/重构代码、运行允许的测试、修复错误、自查、复测并生成报告；不必逐条请求用户批准。可以决定更合适的程序组织、依赖、测试方法和小范围执行顺序。
3. **阶段间验收**：每个 M0 子里程碑及每个后续 Phase 完成后，必须提交阶段报告和可复现命令，**停在验收门禁**。在用户明确批准进入下一阶段之前，不得因上一阶段自评 PASS 就继续开展高成本训练或改造整个模型。当前默认起点是 **M0-2**；M0-1 已完成静态审查，不重复劳动。
4. **必要时调整计划**：可在技术事实、实际资源、兼容性或测试结果支持下调整子任务和实现技术，更新 `docs/PLAN.md` 与 `docs/DECISIONS.md`，说明**原计划→变更原因→影响→验证**。不得擅自改变模型研究主线、取消质量对照或将论文成果冒充本项目成果。重大变更（替换主模型、改变核心科学问题、放弃关键对照、扩大采购/训练预算）必须请求验收者批准。
5. **能跑就跑；不能跑就交付可运行路径**：先识别当前环境。没有 3090、权重、网络或必要许可时，自动执行静态检查、合成数据测试、CPU 测试和可用的轻量实验；真实 GPU 步骤标记 `BLOCKED`，输出精确命令、前置条件和预计所需资源。绝对不能声称 GPU 测试已通过。
6. **不虚构**：严禁伪造吞吐、TTFT、TPOT、P95/P99、模型质量、显存峰值、论文复现收益、已验证权重 Shape 或不存在的工具输出。所有结果必须可追溯到日志/数据文件/命令及环境。
7. **不越权**：除非用户已明确授权，不得租用/购买 GPU、创建付费云资源、发送邮件、推送到远程仓库、提交 PR、删除已有实验数据、改动无关项目、下载巨量权重或发起耗时高成本全模型训练。发现需额外授权时，写入报告并等待用户决定。已授权的本地文件写入、单测和小型公开元数据获取可以自动执行。
8. **研究真实性与复现优先**：训练、量化、性能评测必须记录随机种子、模型/代码/数据版本、参数配置、机器环境、命令、日志、退出码、运行时长、数据集授权/来源及失败案例。
9. **禁止把演示数据当实测**：合成张量用于单测必须标为 SYNTHETIC。论文数据必须标为 PAPER-REPORTED。只有真实本项目执行可标为 MEASURED。
10. **每阶段执行闭环**：`PLAN → IMPLEMENT → TEST → AUDIT → FIX → RETEST → REPORT → WAIT_FOR_ACCEPTANCE`。不要只写方案不做测试，也不要只给一句“完成”。

### 自动执行权限分级

- **A 级：自动允许**：读公开资料、检查仓库与环境、编写代码、合成/小样本测试、静态分析、生成文档、输出命令；项目工作目录内的非破坏性文件创建与修改。
- **B 级：有资源时可自动执行**：用户已经配置好的本地/远程 GPU 上，按当前阶段指定资源上限执行的短时验证和小规模 Benchmark；记录 GPU 型号、占用、显存/CPU 内存与退出码。若没有明确的运行授权或预算，先列命令，不启动。
- **C 级：先获得许可**：大文件下载（默认单次 >2 GiB 或总量 >5 GiB）、新建/租用云资源、全模型蒸馏/长时间训练、覆盖已存在模型权重、改变重要实验口径、远程代码推送、涉及凭据或服务部署的操作。

用户可在 `configs/resource_policy.yaml` 明确批准 B/C 级具体动作及上限；如该文件缺失，采用更保守的默认规则。禁止将密码、密钥、Token 写入仓库或报告。

---

## 1. 研究目标、边界与评价原则

### 1.1 统一目标

研究问题：**DeepSeek-V2-Lite-Chat 的 Routed Experts 是否存在可利用的冗余？专家重路由、剪枝、分组蒸馏、结构压缩与 GPTQ/AWQ INT4 量化，分别或组合能够在怎样的质量约束下改善部署成本与推理性能？**

必须同时形成两种可验证交付：

1. **工程能力证据**：真实模型架构理解、权重映射、Router/Expert Profiling、修改或替换专家、量化/蒸馏、数值与正确性测试、性能定位、可复现项目交付。
2. **技术结论证据**：在严格说明的模型、硬件、数据、精度和工作负载下，效果改善/无改善/恶化的量化证据及合理原因。不保证优化必然有效；负结果也可以完成实验验收，但必须排除实现错误与不公平比较。

### 1.2 主线范围（必须保持）

- **数值优化**：GPTQ、AWQ（先 GPTQ 可行性，AWQ 视支持情况）、W4A16/INT4、量化格式、部署、数值误差、质量、显存、Decode 性能、必要的 GEMV/Kernel 分析。
- **结构优化**：Expert Routing / Load / Co-activation / Output Similarity；SERE 类重路由、Pruning、Sequential Grouping、United Expert Distillation、Similarity Grouping；BrownoutMoE 的 GRPO 分组属于**扩展项**，只有基础分组和蒸馏可用后才做。
- **交叉验证**：原始结构与压缩结构、FP16/BF16 与 INT4 的受控比较；重点分析“量化 × 结构”的叠加效果与质量—显存—性能权衡。
- **暂不纳入**：HeroServe、跨节点通信/INA/Tofino、分布式网络调度、Stage D 的 Scheduler/KV Admission 研发、全新论文 Idea、所有先进模型的横向部署合集。

### 1.3 核心硬件与公平性约束

- 优先 RTX 3090 24 GiB；实际可用性必须探测，不能假设一直有 GPU。其他 GPU（例如 V100）只按实际硬件/软件兼容性使用。
- 固定模型官方 Chat checkpoint：`deepseek-ai/DeepSeek-V2-Lite-Chat`，**Revision** `85864749cd611b4353ce1decdb286193298f64c7`；不要悄悄换成 Base、完整 V2 或另一种 INT4 checkpoint。
- 该模型总参数约 15.7B，理论 BF16 权重约 **29.26 GiB**，单 3090 不具备完整 BF16 权重 GPU 驻留条件。**不能把 3090 INT4 GPU 推理与 CPU Offload BF16 路径的速度直接比较，宣称是量化加速**。严格的同硬件 BF16 对照需更大显存 GPU，或建立性能口径独立的对照。
- 社区 AWQ/GPTQ checkpoint 可用于**路径验证**，但不得充作自己执行的量化成果；模型 revision、量化配置、出处与运行 kernel 必须单独记录。
- K 个 Expert 合并为 1 个 **不等于**整模型压缩 K 倍；Partial Brownout 可能保留原始+联合两套权重。动态 rerouting 不会自动减少模型文件大小。模型权重缩小、GPU 实际显存下降、计算/访存减少、吞吐提高是**不同指标**，分别测试。
- INT4 Expert Profiling 的观测对象与 BF16 教师模型数值不同；必须标记“量化后 Routing/Similarity”，不能当成原始 BF16 行为。

### 1.4 必须维持的实验对照

把不同方法拆开，避免一次修改多个变量：

1. **Original MoE**：未结构修改的同一模型。用兼容精度/硬件作为基准。
2. **Quant Only**：Original MoE + INT4。
3. **Structure Only**：原精度 + 已验证的结构变化（若硬件允许；不允许则标记 BLOCKED）。
4. **Structure + Quant**：同一种结构方案 + 相同量化策略。

另外用 `Top-K naive reduction`、`Pruning`、`Sequential Grouping`、`Similarity Grouping`、必要时 `SERE` 作为结构基线。并非所有方法都必须完整复现，按阶段价值与资源门禁选择；遗漏必须说明原因。不可把不同 GPU / Kernel / 推理框架 / 请求分布造成的差异归因于单一优化方法。

---

## 2. 正式复核报告的已知事实与待验边界

以随附 **P2-M0-1 正式复核报告**为首要静态技术输入，阅读其第 0–9 节和引用资料。对其内容进行**运行时验证**，不要直接重新撰写静态结论。

截至该报告：

- Decoder-only，共 27 层：Layer 0 Dense FFN，Layer 1–26 MoE；全部层含 MLA。
- 每个 MoE 层：**64 个独立 Routed Experts**，每 token greedy Top-6；一个中间宽度 **2816** 的 Shared MLP（逻辑等效 2 个 1408 宽专家，并非两个独立的 MLP 对象）。Routed Expert 中间宽度 1408，hidden size 2048。
- Lite 的 MLA：`q_lora_rank=null`，`kv_lora_rank=512`，`qk_rope_head_dim=64`；架构允许缓存压缩 latent，但 HF eager 与 vLLM MLA 后端的**实际 KV Cache**不同。
- 权重名称和 Shape 在报告中是 **源码+配置推导**，尚未读取真实 Safetensors Tensor Header。仅 4 个权重 shard 的索引和总大小已静态确认。
- 原始 HF Remote Code 暴露 Router 计算点；vLLM FusedMoE 不直接提供逐 Expert 的 Python Hook。
- **公平的专家输出相似性**：同一 MoE 层、同一 `post_attention_layernorm` 后的 hidden state，分别输入候选 Expert，比较输出。未选中专家须增加 shadow forward。
- SERE 有 DeepSeek-V2-Lite 模型族公开实验；BrownoutServe/BrownoutMoE 的公开主实现目前是 Qwen1.5-MoE，不得宣传已在本项目模型上复现。

报告中的任何 `UNKNOWN` 应建立 `docs/EVIDENCE_LEDGER.md` 记录验证计划及状态。证据分级沿用：`[C]` 配置、`[S]` 源码、`[I]` 索引、`[D]` 推导、`[H]` 实际 Header、`[R]` 本项目实测、`[E]` 他人论文实验、`[U]` 未知。**只允许 `[R]` 支撑本项目实测结论**。

---

## 3. 项目仓库与自动化约定

建议建立如下目录，可根据实际代码结构微调，但职责和输出必须保留：

```text
P2-DeepSeek-MoE-Quant/
├── README.md
├── AGENTS.md                       # 简述本指令优先级及所在路径
├── docs/
│   ├── P2_MASTER_RUNBOOK.md         # 本文档的仓库副本
│   ├── inputs/                      # M0-1 正式复核报告；原始参考文献清单
│   ├── PLAN.md                      # 当前任务 DAG、状态与下一步
│   ├── DECISIONS.md                 # 关键设计/范围变更记录
│   └── EVIDENCE_LEDGER.md            # 结论、来源与证据等级
├── configs/
│   ├── model.yaml                   # 主模型与 checkpoint revision
│   ├── profiling.yaml               # seeds、数据、层、采样上限
│   ├── benchmark.yaml               # 模型/输入/输出/并发/重复次数
│   └── resource_policy.yaml         # 执行权限与资源预算（默认保守）
├── src/p2/
│   ├── weights/                     # header & shard / expert mapping
│   ├── trace/                       # router/hidden-state instrumentation
│   ├── analysis/                    # routing/co-activation/similarity
│   ├── quant/                       # quant metadata, tools, validation
│   ├── moe/                         # pruning/grouping/distillation/reroute
│   └── kernels/                     # RMSNorm/W4A16 参考与实验 kernel
├── scripts/
│   ├── env_probe.sh
│   └── run_phase.sh                  # 可安全重试的阶段运行器
├── tests/                            # CPU synthetic + GPU integration
├── data/                             # 只存小型示例与数据版本清单
├── experiments/
│   ├── configs/                     # 每次实验的冻结配置副本
│   └── results/                     # run_id 子目录、原始 CSV/JSON/log
├── reports/
│   ├── TEMPLATE.md
│   └── <stage_id>_validation.md
└── .gitignore                        # 排除权重、cache、密钥、大规模输出
```

**启动时首先**：清点用户现有仓库，不要覆盖已有代码或重复建立项目；若当前目录是已有仓库，沿用其布局并在报告说明映射关系。将复核报告复制/引用到 `docs/inputs/`（如已提供），验证文件实际存在；不要在不存在时声称已读。

每次运行生成唯一 `run_id`，至少记录：日期时间/时区、模型 ID/revision、代码 commit/dirty 状态、Python/PyTorch/Transformers/vLLM/Triton/CUDA 版本、GPU 型号与显存、驱动、使用的 execution backend、量化 checkpoint/格式、数据集版本/hash、Seed、命令与退出码。

所有脚本支持 `--help`，尽量提供 `--dry-run`，具备明确输入输出、失败时非零退出码、不会默默覆盖既有结果；同一路径数据以 `run_id` 区隔。保存关键 stdout/stderr，并附数据字典。未经同意不自动推送 GitHub；只做本地提交建议或准备变更。

---

## 4. Phase 总览：阶段内自动完成，阶段末自查停机

| 里程碑 | 当前状态 | 目的 | 退出标准 / 核心产物 |
|---|---|---|---|
| **M0-1** 静态技术审查 | **已通过** | 架构、MoE/MLA、执行路径 | 使用用户正式复核报告；不当作动态实测 |
| **M0-2** Header & Weight Mapping | **下一步** | 真实 Header、权重索引、单专家提取 | 映射 CSV、可验证 Header 记录、单测、自查报告 |
| **M0-3** Runtime & Router Trace POC | 未开始 | 环境/INT4/追踪路径验证 | 兼容性矩阵、Top-6 Trace、可复现命令；或可解释 BLOCKED |
| **M0-4** Routing / Load Profiling | 未开始 | 真实专家频率与共激活 | 有数据出处的各层统计、图表与不变量检查 |
| **M0-5** Output Similarity | 未开始 | 同输入专家相似度 | 采样策略、shadow forward、cosine/L2/尺度/误差数据 |
| **Phase 1** 量化基线 | 未开始 | Original vs GPTQ/AWQ INT4 | 模型质量/显存/Decode 基线及格式/Kernel 证据 |
| **Phase 2** 低成本 MoE 对照 | 未开始 | Pruning、Naive Top-K、Grouping / SERE | 单层→整模型质量/执行验证，区分逻辑与物理压缩 |
| **Phase 3** United Expert / Brownout | 未开始 | 蒸馏、替换、可选结构感知分组 | 单层蒸馏误差→整模型质量→有限性能对照 |
| **Phase 4** 联合优化实验 | 未开始 | Structure × Quant | 四组对照、消融与质量—延迟—显存报告 |
| **Phase 5** Kernel & Roofline | 未开始 | RMSNorm、W4A16 GEMV、Profiler | 正确性、形状覆盖、Kernel 耗时与瓶颈证据 |
| **Phase 6** 最终交付 | 未开始 | 稳定复现与结果总结 | README、复现脚本、完整报告、限制与简历事实清单 |

**依赖原则**：M0-2 → M0-3 → M0-4 → M0-5 是 Profiling 主链。Phase 1 的 metadata/CPU 工作可并行准备，但任何涉及 GPU 结果和模型替换的主结论必须通过对应前置验收。必要时改变 Phase 排序，在 `DECISIONS.md` 解释并经阶段报告确认。

---

## 5. 分阶段详细执行规范

### M0-2｜Safetensors Header 与 Expert Mapping（立即执行）

**输入**：M0-1 报告；HF 模型固定 revision；`model.safetensors.index.json`；官方模型源码。

**动作**：

1. 确认 index 文件与 revision；枚举 `weight_map` 和全部 4 个 shards。为 Layer 0、Layer 1–26、每层 64 个 Routed Expert 的 `gate_proj/up_proj/down_proj`、Router、Shared Expert 建索引。注意**不要求把 4 个 shards 全部下载**。
2. 尝试使用受控 HTTP Range / 本地已有 safetensors 仅读真实 Header。Safetensors Header 不是 index.json；先读长度、检查 Content-Range 与内容大小。如果服务端忽略 Range 返回整个多 GB 文件，**立即中止，不要意外下载全量权重**。如果离线/权限不支持，保留 `[U]` 而不是以 `[D]` 冒充 `[H]`。
3. 从 Header 核验 shape、dtype、offset；比较报告的推导 Shape，输出差异/缺失矩阵。记录 Header 来自哪个 shard、何时获取、SHA/revision。
4. 实现 `inspect_safetensors.py` / `extract_expert.py`，对**已存在本地 shard**支持按层和 Expert ID 按需读取三个张量，并提供 `--dry-run`/`--metadata-only`。不修改原权重；不要重复保存所有专家副本。
5. 合成 safetensors shard 测试：正常、缺失 key、错误 shape、错误 dtype、越界 Layer/ID、分片不全、路径安全、只读行为。代码应能清晰区分 `VERIFIED` / `PREDICTED` / `UNKNOWN`。

**产物**：`expert_weight_mapping.csv`（字段至少 `layer_id,expert_id,tensor_key,shard,expected_shape,actual_shape,dtype,evidence,status`）、Header 结果、单测、`reports/M0-2_validation.md`。

**PASS 条件**：所有可访问 Header 已逐项核对、缺失明确；核心索引/提取逻辑合成测试通过。若真实 Header 全不可取，最多标 `PARTIAL/BLOCKED`，不得宣称“Header 已通过”。

### M0-3｜环境、真实后端兼容性与 Router Trace POC

**动作**：

1. 探测 GPU、驱动、CUDA、PyTorch、Transformers Remote Code、vLLM 版本与模型 loader；输出**实际**兼容矩阵，包括 MLA、Fused MoE、AWQ/GPTQ、Marlin/其他实际调用后端。不能用“通用 Ampere 支持”代替 DeepSeek MoE 的真实可加载性。
2. 先使用可控的 CPU/小型 MoE synthetic 模型验证 Router Trace：Top-K ID、router logits、全概率与选中权重、Token→Layer 映射、padding/特殊 token 过滤。避免 Trace 改变 Router 的正常计算结果。
3. GPU 可用且已授权时，选择安全执行路径（HF Remote Code / CPU Offload / 量化模型 + 对应后端）做极小样本的**真实模型**追踪。若 vLLM FusedMoE 直接 Hook 不可行，优先使用 HF 离线参考路径，不必立即修改 FusedMoE kernel。记录数据究竟来源于 BF16、INT4 还是模拟模型。
4. 检查 `topk_idx ∈ [0,63]`、每个有效 Token 每层恰有 6 个 Routed IDs、对应概率有限且符合 softmax、Shared 不计入 Top-6、层数/层 ID 正确。验证插桩前后固定 seed 的 logits 差异。

**产物**：`compatibility_matrix.md`、`router_trace.jsonl/parquet`（小样本）、Trace 程序与单测、`reports/M0-3_validation.md`。

**PASS 条件**：至少一条明确的真实模型可复现 Trace 路径，或标 `BLOCKED` 并提交经过合成测试的代码与真实运行命令。**CPU/合成 PASS ≠ GPU/真实模型 PASS**。

### M0-4｜Expert Routing、负载和共激活 Profiling

**动作**：

1. 固定小规模、来源合法的代表性工作负载（至少区分通用文本和代码/中文等类别，记录文本来源、采样策略、长度分布、Chat Template、EOS、seed、是否 prefill/decode）。不能只用随机 token 代表真实流量。
2. 对每层分别统计 64 个 Routed Experts 的选择计数、路由概率分布、负载偏斜、Top-K 同现矩阵（64×64）、样本差异及层间差异；Shared 单独分析。统计选中次数总和应等于该层有效 Token 数×6。
3. 计算每个工作负载组的指标及可信的变化范围；保存原始采样元数据和可复核聚合脚本。不要因某专家“冷门”就直接宣称可删除。
4. 在采样规模和环境能力允许的前提下，比较 BF16 与 INT4 的 Router 选择是否变化；必须明确 checkpoint/精度身份，不能跨精度误报为同一 Router。

**产物**：`routing_counts.csv`、`coactivation.parquet/csv`、每层图表、数据字典、`reports/M0-4_validation.md`。

**PASS 条件**：真实数据、核心不变量和样本分层检查通过；规模不足时只做探索性描述，不得写泛化结论。

### M0-5｜同输入 Expert Output Similarity

**动作**：

1. 在**同一个 MoE Layer** 记录经过 post-attention RMSNorm 的 `h_l`，选择可复现的 token 子集；区分任务/长度/高频与低频 Expert。
2. 同一个 `h_l` 分别对候选 Expert 执行 `gate/up/down` SwiGLU，执行必要的**反事实 shadow forward**；并记录原始 Router 权重，以便进一步比较加权贡献。
3. 计算输出 cosine、relative L2、norm/尺度差异、路由共激活与选择频率；支持对照某专家被替换时的局部输出误差。不得仅凭 cosine 高就宣称可无损合并。
4. 为减小成本先做 1–2 层、小样本、少量候选 Expert Pair，再扩大；明确是否使用 BF16 教师还是 INT4。计算和数据导出需限额，避免保存全部 hidden states 导致 OOM/爆盘。

**产物**：`expert_similarity` 指标与绘图、shadow-forward 正确性单测、`reports/M0-5_validation.md`。

**PASS 条件**：输入对齐、层/专家身份、数值范围、基准函数一致性与对照解释均通过。

### Phase 1｜量化基线：可部署与可解释的 W4A16

1. 先验证**已有公开 INT4 checkpoint** 能否在目标 3090 + 确定版本的 vLLM/其他 runtime 上稳定加载、正确输出；记录实际 kernel 路径。不能把加载成功冒充“自己完成 GPTQ 量化”。
2. 审查并尝试**自主 GPTQ W4A16**，记录校准数据、group size、zero/scale、哪些权重量化/跳过、硬件与时间要求；如果单 3090 不能在合理条件下执行，必须给出有依据的替代方案（CPU Offload/分层量化/借更大 GPU），不杜撰完成状态。AWQ 为可选对照。
3. 数值与质量评估：固定 tokenizer、Chat Template、EOS/实际输出长度；PPL 或等价 held-out 指标 + 2–3 类实际任务（通用、中文、代码/推理子集）；报告失败样本与置信范围。
4. 推理基准：先 Batch 1/2/4/8/16、prompt 128/512/2048、output 64/256；GPU 限制可缩小为先导矩阵再扩展。分别记录 TTFT、TPOT/ITL、E2E、tokens/s、峰值显存、模型文件大小、P50/P95；区分 batch、并发与请求到达率。
5. **公平性**：3090 上 Original INT4 与 Structure+INT4 可以对比；BF16 原始模型若用 CPU Offload 或另一台 GPU，只能作为**质量/功能**参照，不得用其速度证明 3090 的 INT4 加速比。

**产物**：`quant_config`、量化/部署日志、质量与性能 CSV、版本与 Kernel 证明、`reports/Phase1_validation.md`。

### Phase 2｜结构优化最小对照（先单层，再整模型）

1. 先实现**Naive Top-K Reduction** 和仅用于诊断的 Expert Masking（报告两者与物理剪枝的不同）；验证 Router 权重处理与输出不变量。
2. 实现不改变 MLA/Shared Experts 的小规模 **Pruning**、Sequential Grouping、Similarity Grouping 结构对照。对物理剪枝必须维护/修改 Router 维度、Expert ID 与权重映射，检查实际 checkpoint size 和 GPU 显存。
3. 参考 **SERE**（其作者已有 Lite 模型族实验）做同类对照，但先检查实现/License/作者实际 checkpoint，无法直接集成则独立实现合理的简化基线并正确命名，不冒充论文完全复现。
4. 先测试单层替换误差/选中分布，获得稳定结果后再做整模型质量和性能。不能先追求吞吐而忽略质量下降。

**产物**：结构方法开关、测试、方法矩阵、`reports/Phase2_validation.md`。

### Phase 3｜United Expert 蒸馏：BrownoutServe / BrownoutMoE

1. 阅读原论文/开源实现并记录**已支持 Qwen1.5-MoE 的范围**：原配置的专家数、Top-K、路由/权重格式与 DeepSeek 的差异。禁止直接复制 60 个 Expert 的硬编码逻辑到 64 个 Expert。
2. 基于 DeepSeek 的 routed SwiGLU 实现 K=2、4（K=8 视成本）的单层联合专家；先以同输入教师专家输出构造明确的蒸馏目标，记录是否采用路由权重、分组、样本选择、目标归一化等。验证模型 forward 与梯度、loss 收敛、数值/形状及 OOM 安全。
3. 比较 `Sequential Grouping` 与 `Similarity Grouping`。只有这两者可复现后，再考虑 BrownoutMoE 的 **GRPO grouping** 扩展项；不能把“使用聚类”称为“完成 GRPO 复现”。
4. 先实现单层代理/替换与质量退化测量；再尝试多层和小规模整模型推理。**Partial Brownout** 原专家+联合专家常驻的显存与**Full Replacement** 删除专家的显存分别测；二者的质量/路由语义不同。
5. 对照剪枝和直接重路由，不能假设专家蒸馏会提升吞吐或保留质量；记录负结果原因（expert 函数相似性、尺度、Router 失配、数据覆盖、Kernel 退化）。

**产物**：`grouping_map`、学生权重、蒸馏日志、MSE/质量/显存数据、`reports/Phase3_validation.md`。

### Phase 4｜结构 × 量化联合实验

1. 在已验证的单一结构方法上建立 `Original FP16/BF16`、`Original INT4`、`Structure FP16/BF16`、`Structure INT4` 四格实验；做不到的格子如实标 `BLOCKED`，**不要跨不同 GPU 的吞吐直接算加速比**。
2. 统一相同模型源、数据集、tokenizer、采样设置、运行后端、GPU、Batch/输入输出长度、重复次数。区分量化次序：`压缩后再量化` 与 `先量化再修改结构` 不是天然等价，若只执行一种次序须说明。
3. 核验实际磁盘大小、参数量、GPU allocated/reserved/peak、KV/工作区、精度变化、输出质量、TTFT/TPOT/吞吐；必须有**质量—性能 Pareto 对照**，不以单一百分比宣布“更好”。
4. 对结果异常的组合做最少一组消融（例如仅量化专家 vs 同时量化其他 Linear），严格保持其余配置一致。

**产物**：`ablation_matrix.csv`、实验配置、图表、`reports/Phase4_validation.md`。

### Phase 5｜Kernel、数值基线与 Roofline

1. **延续既有 CUDA 学习任务**：完成并验证 RMSNorm CUDA（从 per-thread sum → warp/block reduction → 结果一致性 → benchmark）。不要把已经完成 CPU/PyTorch baseline 宣称为 CUDA kernel 验收完成。
2. 构造 PyTorch INT4 unpack/dequant 参考实现：packing、group scale/zero、边界 shape 与数值误差。
3. 实现并测试 **Triton W4A16 GEMV (Batch=1)**，覆盖多种 shape、非整除维度、dtype、数值容差；可选小 Batch GEMM（2–16）与 autotune。与 torch 参考、实际 vLLM/Marlin 等生产 kernel 对照；更慢也要解释。
4. Nsight/PyTorch Profiler 实际验证 Kernel 时间、访存、dequant 开销、cache、Kernel Launch 等；Roofline 的带宽/算力假设要与实际 GPU 对齐。量化算法本身、MLA KV 访问和 MoE Expert GEMV 之间的瓶颈要分开。
5. 无 GPU 时编写可执行测试和运行命令，标记 GPU Kernel 实测未完成，不能自评最终 PASS。

**产物**：RMSNorm 和 GEMV 源码、自动测试、数值与性能数据、`reports/Phase5_validation.md`。

### Phase 6｜最终报告与可复现交付

1. 清理目录并提供 `README`：环境创建、依赖锁定、数据获取、模型 revision、命令示例、每阶段运行方式、GPU/显存最低要求。
2. 从原始日志**重新计算**所有表格和图表；核对模型质量、性能、显存和文件大小口径，给出失败/未完成项。
3. 明确分开“工程实现已完成”“基线复现”“有条件观察”“没有证据证明”“论文作者结果”和“我们的测量”。
4. 输出完整实验报告、对照/消融、局限、复现说明、代码目录、最重要的 3–5 条**可由日志支撑**的简历事实（不用提前编造提升比例）。
5. 推荐后续工作必须以结果为依据；不要因为某方法没有提升就为了正结果而更换指标或数据集。

**产物**：`reports/FINAL_REPORT.md`、`reports/REPRODUCIBILITY.md`、`README.md`、冻结的版本与数据 manifest。

---

## 6. 所有阶段统一自查协议（强制）

每个阶段在写报告前依次完成：

**A. 事实/版本审查**：引用的模型、源码、checkpoint、官方论文、脚本与执行后端是否一致？是否混淆 DeepSeek-V2-Lite、DeepSeek-V2 和 R1 蒸馏 Qwen？是否混淆 HF eager 与 vLLM fused/MLA？

**B. 程序正确性**：单测、edge cases、shape/dtype、数值精度、padding/EOS、索引边界、Router ID 映射、shared expert 保留、内存释放、异常退出与重跑幂等性；确保没有把模型 Forward 改坏。

**C. 实验公平性**：相同 checkpoint/输入/seed/采样/硬件/后端、预热与重复次数；区分单请求、批量、请求并发、队列等待；均值/分位数/样本数；性能必须实际测量且能复核。

**D. 数据可靠性**：每张图的原始数据文件是否存在？是否出现 NaN/Inf、Token 计数不守恒、漏采 Layer、重复 Token、失败请求剔除不说明等问题？专家相似度是否真正使用同输入？

**E. 权限与可复现性**：未越过资源预算，未泄露密钥，未毁坏原权重，命令能够复跑，依赖/commit 真实记录。

发现失败时先修复并复测，之后报告最终状态。如因硬件/资源/依赖不能修复，给出**最小阻碍描述、失败命令、日志、可替代方案与预期额外资源**，并将阶段状态设为 `BLOCKED` 或 `PARTIAL`，而不是虚假 PASS。

### 标准阶段验收报告模板

每个 `reports/<stage_id>_validation.md` 至少包含以下标题和内容：

```markdown
# <stage_id> 验收报告

## 1. 执行摘要
- 状态：PASS / PARTIAL / FAIL / BLOCKED
- 本阶段目标与实际完成边界
- 关键发现（MEASURED / SOURCE / INFERRED / UNKNOWN）

## 2. 输入与环境
- 输入报告/模型 revision/代码 commit/依赖/硬件/数据集/预算

## 3. 文件变更与实现
- 新增/修改文件清单与核心函数/设计取舍

## 4. 实际执行命令与输出
- 完整命令、退出码、起止时间、日志路径；哪些没有执行及原因

## 5. 测试与自查
- 单元/集成/数值/数据不变量/性能公平性测试结果

## 6. 数据、图表和指标
- 每项结果的来源路径、单位、样本数、置信度/限制

## 7. 问题与未验证项
- 阻碍、错误日志、UNKNOWN、复现实验限制

## 8. 计划变更
- 变更内容、依据、为何不影响原研究主线

## 9. 验收清单
- [ ] 本阶段每个核心条件及其证据；未通过项逐项说明

## 10. 建议下一阶段与人工验收问题
- 下一阶段建议与依据
- 需要用户/审查者实际验收的 1–3 个关键问题
- 停止执行，等待验收
```

### 验收状态精确定义

- `PASS`：本阶段定义的必须产物齐全，测试与证据充分，且所有实测声明有真实日志。
- `PARTIAL`：完成部分可复用工程工作，但有重要未测/未达成目标；不能进入依赖其结果的下一阶段。
- `BLOCKED`：外部资源/硬件/权限/未支持后端阻止验证；交付可复现代码和精确阻碍。
- `FAIL`：核心实现/测试失败，且已知存在未修复问题；如实记录。

---

## 7. 资源与失败处理策略

1. **能提前准备的都在非 GPU 环境准备**：代码、配置、synthetic 单测、索引处理、数据采样脚本和统计逻辑。租卡后以固定命令批量运行，不现场临时搭建整个工程。
2. **真实 GPU 的最小验证先于大规模实验**：单 batch、短输入、短输出、少数层/专家、少量采样，先验证功能及无 OOM，再阶梯扩大。
3. **出现 OOM**：记录实际显存和失败阶段，检查模型本体 vs KV vs activations vs workspace；可尝试合法的小批/短序列/分层/CPU Offload，但不同执行路径的性能结果不能混用。
4. **版本兼容冲突**：保留失败环境 manifest，换版本应新建环境或记录锁文件差异，不在单次结果中悄悄升级依赖。静态报告对 vLLM `v0.7.3` 的源码描述仅用于参考，项目最终运行版本须单独验证；不得把 Stage D 的 V0/V1 选择强行套用于项目二。
5. **模型不可运行**：先验证公开量化 checkpoint 与实际 backend；必要时在单层权重/合成 MoE 上完成程序验收，整模型标 BLOCKED。替换主模型必须另行获得批准，不能悄悄使用 7B 模型代替 DeepSeek。
6. **方法效果不佳**：复核实现、统计功效、质量集、推理后端与 Kernel 瓶颈；若复测仍无收益，形成有效负结果，不人为删去失败配置。
7. **成本纪律**：任何“GPU 小时数、运行耗时、节省内存、加速倍数”必须来源于实际测量；无法估计时给出测量计划而不是确定性承诺。

---

## 8. 关键文献与对照方法（作为工作索引，不替代文献核查）

- **主模型**：<https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat>；固定 commit `85864749cd611b4353ce1decdb286193298f64c7`。
- **DeepSeek-V2 技术报告**：<https://arxiv.org/abs/2405.04434>。
- **vLLM v0.7.3 参考源码**：<https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/models/deepseek_v2.py>。
- **BrownoutServe**：<https://github.com/beyondHJM/BrownoutServe>，原仓库公开主要适配 Qwen1.5-MoE；注意其 United Expert 权重是否需要另行获取。
- **BrownoutMoE**：<https://github.com/done-77/BrownoutMoE>，包含分组/蒸馏相关代码；当前代码对模型结构存在硬编码，DeepSeek 不能直接套用。
- **SERE**：<https://github.com/JL-Cheng/SERE>；与直接减少 Top-K 的 naive baseline 区分。
- **更多已有方法**：REAP / REAM、MoE-to-Dense。各自的模型、论文版本、可复用代码与实际性能必须单独核验，不作为默认已有的项目成果。

**引用与代码授权**：论文内容用于验证与方法对照；直接复制开源实现前检查 License，明确修改内容、来源 commit 和衍生代码版权信息。网络查询产生的信息应标记调查日期。

---

## 9. 启动动作（Codex 本轮必须完成）

**从这里开始执行，不要从 Phase 3 或联合量化直接开始。**

1. 查找并阅读本文件与用户提供的 `DeepSeek-V2-Lite-Chat_P2-M0-1_正式复核报告(1).md`，核对报告是否完整可读。
2. 清点已有项目代码和资源环境，创建/更新 `docs/PLAN.md`、`docs/EVIDENCE_LEDGER.md`、`docs/DECISIONS.md`，记录 M0-1 已通过和 M0-2 进行中；先给出精简执行计划，然后直接实施，不要等额外的日常确认。
3. **自动完成 M0-2 的全部 A 级工作**：索引读取/Header 的可行性验证、mapping 程序、单专家按需读取接口、合成单元测试、静态检查、错误修复与复测。
4. 对因权重/网络/真实 Header 不可得导致的阻碍，准确写出缺少什么、怎么验证、需要用户执行什么；**不要为了让验收变绿而下载完整 29 GiB BF16 模型**。
5. 运行自查协议，输出 `reports/M0-2_validation.md`，更新本阶段状态与下一阶段建议。
6. **停止，等待用户对 M0-2 验收**。只有验收后才启动 M0-3；以后各阶段复用同一门禁。

**本次交付的最后一条消息格式**：

- `STATUS: PASS / PARTIAL / BLOCKED / FAIL`
- `COMPLETED: <真实完成的工程项>`
- `TESTS: <执行过的测试、成功/失败、日志>`
- `EVIDENCE: <权重 Header 是否真验证，来自何处>`
- `ARTIFACTS: <代码、CSV、报告路径>`
- `BLOCKERS: <没有解决的问题>`
- `NEXT: <待用户验收后进入的阶段>`

**最终原则**：自动化的是执行过程，不是研究结论。每个阶段都交付可被独立审查的代码和证据；先证明代码和测量可信，再判断优化是否有效。
