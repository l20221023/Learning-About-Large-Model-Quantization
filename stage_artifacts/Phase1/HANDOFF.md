# 总调度 GPU 执行交接

> **初始交接记录，已被实际运行结果取代。** 远端部署、2/64-token 探针和正式 15 请求先导已完成；见 `SELF_CHECK.md` 与 `../../reports/Phase1_validation.md`。请勿将下文的“尚无模型实测”作为当前状态。

本目录包含自足的四个执行输入：`run_phase1_awq_autodl.sh`、`collect_awq_baseline.py`、`analyze_awq_baseline.py`、`m0_4_seed_prompts.json`。固定工作负载 SHA-256 为 `c934f433108052632a90b313e1e12a6f04828c67f7c06f67e0a00a28d37319f0`。本地仅验证 Python 编译、Shell 语法和合成聚合；尚无模型实测。

请总调度在已严格核验主机密钥的密码交互会话中，将四文件复制到远端同一新目录（例如 `/root/autodl-tmp/p2-phase1-deploy/`），核对 SHA-256 后运行：

```bash
cd /root/autodl-tmp/p2-phase1-deploy
sha256sum *.py *.sh *.json
bash ./run_phase1_awq_autodl.sh
```

runner 自动使用现有 `/root/autodl-tmp/p2-deepseek-moe` 的 AWQ 快照与 vLLM 0.26.0，先检查数据盘新增 1 GiB 峰值和系统/数据盘 8/10 GiB 保留线；GPU 服务、采样及 15 条顺序请求全程由 `flock -n /root/autodl-tmp/p2-3090.lock` 和磁盘守护包裹。server 仅绑定 `127.0.0.1:8765`。每次新建 `Phase1-AWQ-<timestamp>-3090` 输出目录，含 `command.txt`、`exit`、server/runner/采集日志、`raw.jsonl`、`summary.json`、GPU CSV、运行前后 `df`/`nvidia-smi` 与哈希。服务加载若超时或参数不兼容，保留失败目录及退出码，修复后用新 run_id 重跑；不要覆盖。

回传完整 run 目录，或将其归档并附执行端 SHA-256。由本会话对原始请求、usage、成功数、时间顺序、统计、kernel 日志、显存和磁盘证据做独立复核。未回传前不填入任何 Phase 1 实测结论。
