# Phase 1 基线执行规范

## 前置复核

- M0-5 总调度复核为社区 AWQ 路径 PASS；证据见 `stage_artifacts/M0-5/FINAL_AUDIT.md`。该结论限第 13 层、2 条自编提示，不能替代质量或 decode 基线。
- 唯一允许的模型实测对象：已存在的 `TechxGenus/DeepSeek-V2-Lite-Chat-AWQ@e9fdf80ec6f1d200a029c41004bee40993763b43`，单张 RTX 3090，现有 vLLM 0.26.0 环境。官方 `deepseek-ai/DeepSeek-V2-Lite-Chat@85864749cd611b4353ce1decdb286193298f64c7` 未有完整可运行权重；原始模型质量和性能格当前 BLOCKED。
- 远端先确认实例剩余时间、空闲 GPU、模型分片哈希、运行时、数据集许可和磁盘剩余。每条 GPU 命令必须在 `/root/autodl-tmp/p2-3090.lock` 的 `flock -n` 内运行，并由 `autodl_storage_guard.sh run` 包裹。先 `check`，系统盘留 8 GiB、数据盘留 10 GiB。

## 小规模输入与门槛

- **功能/质量先导**：复用 M0-4 的 6 条自编提示（通用英文、中文、代码各 2 条），输入来源和提示原文随原始数据保存。固定 checkpoint 自带 tokenizer、Chat Template、greedy、seed、EOS、`max_tokens=64`；保存逐条 token IDs、可见文本、finish reason、实际输出长度和失败案例。此集合只能提供探索性功能检查，不能计算可靠 PPL、置信区间或模型质量优劣。
- **质量正式基线门槛**：总调度确认许可和本地可用的 held-out 语料后，固定 PPL 或等价指标以及通用、中文、代码/推理任务的样本和评分方式，再执行。当前数据集/许可待核实，因此正式质量比较 BLOCKED。
- **decode 性能先导**：先做 batch=1 顺序请求，从每类自编提示选第 1 条，各重复 3 次，greedy、最多输出 64 token。记录每个流式响应块的时间、TTFT、E2E、finish reason；若 API 返回 usage，再记录 prompt/completion token 数和 completion tokens/E2E。P50/P95 仅限 9 次先导请求。流式块不保证一块一 token，因此此版 runner 将 ITL/TPOT 留空。更长输入、256 输出、批处理和并发矩阵均留待先导成功且资源复核后扩展，不将当前数据称作完整性能矩阵。
- **资源**：记录 250 ms `nvidia-smi` 显存/利用率采样、实际模型文件大小、GPU 运行前后占用、CPU/数据盘写入与时长。既有短请求峰值 10,595 MiB 只供预算参考，不当作 Phase 1 峰值。初始磁盘峰值预算：新增日志/JSON/采样 1 GiB，系统盘新增 0 GiB；超出时重新估算并预检。不得下载/转换权重。

## 公平口径与 GPTQ 审查

- AWQ 是社区 checkpoint；每次运行保存 revision、量化配置、实际 `AutoAWQMoEMethod`、`MarlinExperts` 与 MLA/kernel 日志。若发生 fallback，按实际路径重新标记。
- 原始 BF16 未能在本机单卡完整驻留，且下载/转换违反当前空间预算。其质量格须先满足可用权重与合法数据；其 3090 decode 性能格 BLOCKED。CPU offload 或不同 GPU 的 BF16 速度不可与 3090 AWQ 计算加速比。
- 自主 GPTQ 仅做可行性审查：校准数据与许可、group size、对称性/zero/scale、跳过层、显存/内存、源权重与输出共存磁盘峰值、耗时和后端支持。没有授权的 BF16 源权重及足够空间时不启动转换，不将已有 AWQ 称为 GPTQ 成果。

## 复现与交付门槛

- 远端每次 run 使用新目录；保存命令原文、UTC+08 起止时间、退出码、完整 stdout/stderr、原始请求/流式事件、GPU CSV、环境、SHA-256 和失败修复记录。先 CPU 校验数据解析与统计，再运行 GPU 小探针，最后扩展先导矩阵。
- 所有 Phase 1 结果及阶段报告归档于 `stage_artifacts/Phase1/`。结论分为 MEASURED_COMMUNITY_AWQ、CPU/SYNTHETIC、UNKNOWN/BLOCKED；缺失原始模型或四格不得补值。不修改共享计划、决策、台账，不提交或推送。

## 当前阻塞

此前无法安全接入远端的阻塞已于 2026-11-18 08:34（Asia/Shanghai）解决：总调度以严格 host-key 校验的密码交互登录相同实例，用户确认 GPU 剩余时间充足。2/64-token in-process 探针和 15 请求服务先导均已完成并归档。当前 Phase 1 状态为 PARTIAL：社区 AWQ 性能先导通过，但正式质量/GPTQ/扩展矩阵未完成。现行结果见 `SELF_CHECK.md`、`STATUS.md` 和 `../../reports/Phase1_validation.md`。
