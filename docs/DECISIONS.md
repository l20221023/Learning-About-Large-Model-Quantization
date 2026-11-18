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

## 2026-11-18 总调度与 M0-5 恢复

- 用户明确授权在当前既有 RTX 3090 实例上继续实验，并要求阶段间连续推进、最终统一人工验收。总调度以独立 Codex 会话拆分 M0-5 与 Phase 1–6，依赖阶段顺序派发；Phase 1–6 任务创建后保持待派发，不与 M0-5 并发，也不并发占用 3090。总调度独占维护共享计划、决策和证据台账，阶段副本交付到各自独立目录。
- 2026-11-18 00:58 Asia/Shanghai 通过已保存主机密钥的严格校验及禁用公钥的交互密码方式登录成功。现场磁盘余量满足 8/10 GiB 底线，3090 空闲，现存 AWQ 权重和 vLLM 隔离环境可用。用户提供的实例时长是约 5–6 小时；登录 banner 未给出剩余倒计时，不能据此当作精确剩余时间。
- M0-5 静态审查发现原实现只对四个候选专家进行 shadow forward，却需要按六个 native Top-K 专家重构 FusedMoE 输出。该实现无法通过正确性门槛。先扩展执行集合覆盖候选与 native Top-6，并验证单 worker 的 `apply_model` 返回形态；是否正确仍需 3090 实测与 shadow reconstruction gate 证明。
- 首次 AWQ 3090 API 探针（单提示、单候选）实际进入模型生成后失败：PyTorch pre-hook 得到空 positional tuple，说明 vLLM 的 FusedMoE 调用将参数放在 kwargs。退出码 1，完整日志在 `/root/autodl-tmp/logs/M0-5-api-probe-20261118T010700-3090/run.log`，GPU 随后释放。修复方向是使用 `with_kwargs=True` 并按 vLLM 实际参数名取得 hidden；修复后必须重跑探针及完整 shadow reconstruction，不接受只依赖单测。
- 当前远端 `main` 无法由 `git ls-remote` 确认，保留待核验状态；本地首次提交不重做。阶段结束仍按用户明确授权检查 staged diff 和敏感信息后提交/推送；推送失败记录 SHA 并标记未发布。

## 2026-11-18 M0-5 接续实测

- 实例恢复连接后，M0-5 两次失败探针及后续成功探针、正式运行均保留退出码和原始日志。第二次失败暴露独立调用缺 vLLM forward context。最终使用单设备 context，在 shared 专家合并前截取 routed 输出，令原生与 shadow 共享同一观测点。
- 正式 AWQ 3090 运行 `M0-5-full-20261118T075249-3090` 退出 0；4/4 个采样 token 通过 Top-6 routed 输出重构。结果限定社区 AWQ、第 13 层、两条自编提示，不能外推 BF16、专家合并安全性或整模型质量。原始包与独立审计见 `experiments/results/M0-5-full-20261118T075249-3090/`、`stage_artifacts/M0-5/FINAL_AUDIT.md`。
- M0-5 通过总调度复核后已派发 Phase 1；仍只使用现有 3090 和既有 AWQ 权重，GPU 串行锁为 `/root/autodl-tmp/p2-3090.lock`。官方 BF16 权重下载与转换继续受容量和授权门槛约束。
