# Phase 1 当前状态

2026-11-18，Asia/Shanghai。**阶段状态 PARTIAL：社区 AWQ 功能探针和 batch=1 短请求性能先导已实测并复核；Phase 1 完整量化基线未完成。** 总调度重新登录并核查同一 RTX 3090 空闲、系统/数据盘保留线满足，按用户确认剩余时长足够执行。首轮 `Phase1-AWQ-20261118T082549-3090` 退出码为 1、15 条请求中 8 条失败；保留作失败证据且未计入结果。修订服务配置后，正式先导 `Phase1-AWQ-20261118T083931-3090` 退出 0，15/15 请求成功；6 条质量输出仅作未评分探索观察，9 次性能请求的采集与统计已独立复核。阶段独立报告为 `SELF_CHECK.md`。

## 已完成

- 已固定最小 AWQ 功能/质量先导、decode 流式基线、资源峰值、8/10 GiB 存储保留线、公平性和 GPTQ 可行性审查口径，见 `BASELINE_SPEC.md`。
- CPU 校验 `workloads/m0_4_seed_prompts.json`：JSON 可解析，`general`/`chinese`/`code` 各 2 条，合计 6 条非空提示；退出码 0。文件 SHA-256：`c934f433108052632a90b313e1e12a6f04828c67f7c06f67e0a00a28d37319f0`。命令：`C:\Users\li\.conda\envs\pytorch_clean\python.exe -B -c "import json,hashlib,pathlib; p=pathlib.Path('workloads/m0_4_seed_prompts.json'); x=json.loads(p.read_text(encoding='utf-8')); g=x['groups']; assert set(g)=={'general','chinese','code'}; assert all(len(v)==2 and all(isinstance(t,str) and t.strip() for t in v) for v in g.values()); print(hashlib.sha256(p.read_bytes()).hexdigest())"`。
- 已提交独立采集包文件：`collect_awq_baseline.py`、`analyze_awq_baseline.py`、`run_phase1_awq_autodl.sh` 与固定工作负载副本。Python `py_compile` 退出 0；`bash -n` 退出 0；15 条合成记录的聚合检查退出 0，验证 6 条质量输出、9 条性能请求及无 ITL 值。以上均为 CPU/SYNTHETIC，非模型实测。

## 未完成与下一步

- 三个原始归档（首轮失败、decode 探针、正式服务先导）均在 `evidence/`，其压缩包和内部文件哈希见 `SELF_CHECK.md`。
- 本轮支持的结论仅为现存社区 AWQ、单张 RTX 3090、vLLM 0.26.0 的短请求路径与 9 次串行性能先导；不代表项目自主 GPTQ 量化结果，也不是原始 BF16 加速对照。
- Phase 1 完整退出项仍未完成：正式 held-out 质量/PPL 与许可核实、GPTQ 转换可行性定量试验、拓展 batch/输入长度/并发性能矩阵、token 级 ITL/TPOT。根据 runbook，PARTIAL 阶段不能派发依赖它的 Phase 2；在完成缺项或总调度按规范修订范围前继续暂停 Phase 2 GPU 工作。
- 未执行官方 BF16 下载或 GPTQ 转换：现有数据盘余量扣除 10 GiB 保留线后不足以容纳 31.41 GB 官方 BF16 权重，且本阶段尚未锁定合规校准/held-out 数据。未使用额外 GPU、未扩容或删除数据。
