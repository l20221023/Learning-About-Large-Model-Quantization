# Phase 1 自主 GPTQ W4A16 可行性与最小验证计划

状态：PLAN / 未运行自主 GPTQ。现有社区 AWQ 是 3090 部署与性能基线，不能作为自主量化成果。

## 研究必要性

P2 runbook Phase 1 第 2 项明确要求审查并尝试自主 GPTQ W4A16；项目研究问题也包括 GPTQ/AWQ 与 MoE 结构修改的交互。故至少应完成真实 BF16 来源的局部 GPTQ 量化、数值验证与资源审计。若该路径在当前资源下不可行，须有实测失败与替代方案；不因 AWQ 已能推理就把 GPTQ 标记为完成。完整 GPTQ checkpoint 不是无条件要求。

## 当前可执行的顺序

1. 固定官方 DeepSeek-V2-Lite-Chat BF16 revision、文件清单与每个 safetensors 分片大小；核对许可证、下载方式、临时文件策略和峰值磁盘预算。大权重下载前由总调度说明并取得授权。
2. 固定 GPTQModel 或其他实际支持 `deepseek_v2` 的量化器源码版本；在隔离环境安装，审查 DeepSeek 专家、shared expert、attention Linear 的映射，记录跳过的模块。上游模型族列表只证明声明支持，不能代替本模型实测。
3. 选定独立于 held-out test 的校准数据，记录来源/许可/去重、样本数、token 长度、seed。预先固定 W4A16、group size 128（必要时 64 对照）、zero/scale、act-order、damp、量化覆盖范围及输出格式；原有 AWQ 的 group size 64 不能被误作控制变量一致。
4. 在现有单张 3090 上先对真实 BF16 的一层与一组 routed/shared expert 做 GPTQ 小样本：记录模型加载方式、GPU/CPU 内存、磁盘、耗时、重构误差、导出格式及 vLLM 能否调用目标 kernel。失败原样归档。
5. 仅当局部试验通过、完整源/临时/产物峰值空间满足系统盘 8 GiB 和数据盘 10 GiB 保留线、且估计时间与 GPU 显存可承受，才运行整模型自主 GPTQ。产物与 AWQ 使用相同 held-out 数据及推理工作负载评估质量和成本；不以不同量化算法的绝对 PPL 推断结构优化效果。

## 当前资源门槛

2026-11-18 当前数据盘 100 GiB，可用约 91.05 GB；扣 10 GiB 保留线后约 80.31 GB 可供新增写入。官方 BF16 权重 31,412,968,448 bytes；现存 AWQ 已占盘，但无需复制。若下载临时文件与原件各占一份 BF16，再加约 9.1 GB INT4 输出，峰值新增约 71.9 GB，只剩约 8.4 GB 给校准缓存和量化中间产物，尚不能宣称完整转换安全。单张 3090 的 24 GiB 显存不足整模型 BF16 常驻；增加第二张 3090 也不自动合并显存，必须验证 tensor parallel 或逐层 CPU offload 路径。

## 出口

局部真实权重 GPTQ 成功并能核对数值、格式与资源，或有可复现的失败与明确替代方案；完整模型仅在资源预检通过后启动。报告需区分 `LOCAL_GPTQ_MEASURED`、`FULL_GPTQ_MEASURED` 与 `BLOCKED`，不得用社区 AWQ 或合成权重冒充真实 GPTQ。
