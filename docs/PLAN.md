# P2 执行计划

- M0-1 静态架构复核已由用户验收，原始报告见 `inputs/`。
- M0-2 自查 PASS，用户已批准进入 M0-3。固定 revision 的官方索引及全部四个 Header 共核对 5,291 个张量。单专家提取通过合成分片测试。详见 [M0-2 验收报告](../reports/M0-2_validation.md)。
- M0-3 自查 **PASS（社区 AWQ 实测路径）**。3090 上的真实短请求及 26 个 MoE 层的原生 Router Trace 已通过。固定 revision 的官方 BF16 实测仍待可运行环境。[3090 验收报告](../reports/M0-3_3090_AWQ_validation.md) · [本地 CPU 验证](../reports/M0-3_validation.md)
- M0-4 自查 **PASS（探索性社区 AWQ prefill 路径）**。6 条自编提示、3 类负载、169 个有效 token、4,394 条 Trace；选择数、共激活、Shared hook 和插桩前后 logits 不变量通过。[M0-4 自查报告](../reports/M0-4_validation.md)
- AutoDL 存储审计与容量保护已完成；官方 BF16 下载和 BF16→量化共存转换目前受 10 GiB 数据盘保留线限制。[存储审计](../reports/storage_audit_autodl_20261118.md)
- 用户要求后续实验连续逐个验证并在最后统一验收；阶段自查仍需逐项留证，但本轮不在每个阶段暂停等验收。当前继续 M0-5，保留 AWQ/BF16 证据边界和资源授权门禁。
- M0-5 自查 **BLOCKED（GPU 实测未执行）**。runner/聚合器准备完成，32 项本地测试通过；当前主机连接 AutoDL 的 TCP 在 SSH 认证前被拒绝。待恢复访问后完成真实 shadow forward 与数据归档。[M0-5 自查报告](../reports/M0-5_validation.md)
