# P2 决策记录

## M0-2 实现路径

- 启动目录为空且没有 Git 仓库。按总执行规范的职责布局建立当前阶段所需文件。
- 当前阶段使用 Python 标准库解析 Safetensors 的长度前缀、JSON 和数据偏移。本地提取按偏移复制选中专家的三个张量，输出独立 Safetensors 文件，保留字节和 dtype。
- 官方远端按固定 revision 获取索引和配置。每个 shard 最多请求两个连续 Range，收到响应后先验证 206、Content-Range、长度和对象标识，再按上限读取。读取失败仍保留可用索引以及已获取的其他 Header。
- M0-1 第 4 节为预期 Shape 基准，原版配置声明的 BF16 为预期 dtype。实际 metadata 和推导值分列对照。
- 自查发现 Python 默认跳转处理会读取 3xx 响应体。改为手动跟随有限次 HTTPS 重定向，直接关闭 3xx 响应，保留原 Range。合成测试覆盖跳转和拒绝读取正文，真实复测再次取得四个 Header 和索引。
- 直接沙箱连接失败。经自动权限审查放行的操作只获取有大小上限的公开元数据。复测中索引一次 TLS EOF，原始结果保留，重试完成。
- 暂时没有已有本地官方权重文件。真实索引定位留档，数据区提取以两个合成分片通过功能与只读性验收。无需为本阶段下载完整模型。

## M0-3 实现路径

- 先用已有 PyTorch CPU 环境验证原生 MoE。用未修改的固定版官方 Remote Code、随机 BF16 权重和合成 Safetensors 分片测试真实 HF 加载器路径。官方 tokenizer 和 Chat Template 独立验证。
- 原生 gate forward hook 直接读取 Top-6 ID 与权重。64 维 logits 用实际 gate 输入、实际权重重算同一 FP32 Linear，推导全量概率并核对 native Top-6。按真实 token 位置记录，退出时移除 hook。
- Transformers 4.44.2 把条件导入的 `flash_attn` 当作必需包。保留失败日志，在隔离目标上升级到 4.45.2，CPU eager 复测通过。官方代码未修改。
- 本机缺预训练权重、可用 CUDA 和 BF16 所需 RAM。真实路径提供离线快照和硬件预检查，缺资源时写 `BLOCKED` 清单并退出 2。远端仅获取公开源码与 tokenizer。M0-2 的元数据读取默认 2 MiB 上限保留，只有 M0-3 tokenizer 文件显式使用 6 MiB 上限。
- 用户随后提供并授权使用 RTX 3090 实例，另行批准下载社区 AWQ 权重。官方 BF16 继续用作原版参照。云机选择隔离的 vLLM 0.26.0，先实测 AWQ 路径。vLLM 的 FP16 GroupedTopK 和 HF 的 FP32 softmax 分别按各自 dtype 校验。[云机验收](../reports/M0-3_3090_AWQ_validation.md)

## M0-4 及存储保护

- 用户明确要求后续实验逐个执行，全部完成后统一验收；因此不在 M0-3 自查 PASS 后暂停 M0-4。M0-3 仍未标记为人工验收，后续量化来源各自记录。
- M0-4 使用项目自编的通用英文、中文、代码提示，各两条；小样本仅作探索性描述。每条 prompt 独立 prefill，单独保存原始 native route capture。共享专家以只读 forward hook 分析。
- CUDA FP16 softmax 与 CPU 直接 FP16 复算会在舍入中点相差一个 ULP。保留严格校验，改为 FP64 参考概率并检查 FP16 舍入区间；native Top-6 IDs 和 logits 不变性仍单独核验。失败日志及原始 captures 留档。[M0-4 报告](../reports/M0-4_validation.md)
- AutoDL 按用户要求建立数据盘缓存环境及容量守护，保持 8 GiB 系统盘和 10 GiB 数据盘余量；缓存环境由 `.bashrc` 与项目 runner 加载。没有搬动权重、删除缓存或使用只读 `/autodl-pub`。[存储审计](../reports/storage_audit_autodl_20261118.md)
- 当前 31.41 GB 官方 BF16 checkpoint 下载和需 BF16/INT4 共存的量化峰值无法满足数据盘保留线。没有下载 BF16 或开展完整量化；小型既有 AWQ 实验未触发空间风险。
