# Phase 1 部署输入冻结记录

2026-11-18，Asia/Shanghai。以下四文件是总调度待部署版本；收到 GPU 原始结果前不再修改。SHA-256 按文件原始字节计算：

| 文件 | SHA-256 |
|---|---|
| `run_phase1_awq_autodl.sh` | `ea7e305fdbd49cc3c27d5fdcb0f661825538446d7c4af37c67301aac85d91062` |
| `collect_awq_baseline.py` | `38ff1a81465b584aca322833d2ed73f0b4d70861f934a031a913aab06180c893` |
| `analyze_awq_baseline.py` | `e206e32fa2fb18831ad0a086036face7e5299c1ec6355cec96e02816da271fa3` |
| `m0_4_seed_prompts.json` | `c934f433108052632a90b313e1e12a6f04828c67f7c06f67e0a00a28d37319f0` |

本地验证（工作目录 `E:\01-Projects\llm_quantization`）：

| 命令 | 退出码 | 结果 |
|---|---:|---|
| `C:\Users\li\.conda\envs\pytorch_clean\python.exe -B -m py_compile stage_artifacts/Phase1/collect_awq_baseline.py stage_artifacts/Phase1/analyze_awq_baseline.py` | 0 | Python 语法通过 |
| `bash -n stage_artifacts/Phase1/run_phase1_awq_autodl.sh` | 0 | Shell 语法通过 |
| 下方合成断言命令 | 0 | 6 条 quality、9 条 perf，token 速率与 ITL 留空口径通过 |

合成断言命令（PowerShell）：

```powershell
$py='C:\Users\li\.conda\envs\pytorch_clean\python.exe'
& $py -B -c "import importlib.util; s=importlib.util.spec_from_file_location('m','stage_artifacts/Phase1/analyze_awq_baseline.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); rows=[dict(phase=p,group=g,prompt_index=i,repeat=r,status='ok',e2e_s=2,ttft_s=.5,chunk_count=1,chunks=[dict(text='x')],output_text='x',usage=dict(completion_tokens=2)) for p in ('quality','perf') for g in ('general','chinese','code') for i in range(2 if p=='quality' else 1) for r in range(1 if p=='quality' else 3)]; z=m.summarize(rows); assert (z['requests'],z['perf_successful'],z['tokens_per_e2e_s_median'],z['itl_tpot'])==(15,9,1,None); print('PASS')"
```

工作负载 JSON 已单独解析检查：三组各 2 条非空提示，合计 6 条，退出码 0。**截至此部署输入冻结时**，上述验证均为 CPU/合成数据而非模型实测，GPU run_id 和原始目录仍待总调度回传独立审计。后续实际部署修订和已完成的 GPU 复核见下节及 `SELF_CHECK.md`。

## 实际部署修订（2026-11-18）

首轮服务在 MLA prefill 因 AWQ `ColumnParallelLinear.weight` 缺失退出 1。修复探针确认将 vLLM 收敛至短上下文、小批量、单序列、512 MiB KV cache 且关闭 prefix cache 后，in-process decode 成功。经核对安装版 vLLM `serve --help=all` 支持对应 CLI 参数；Phase 1 服务 runner 因此采用相同限制。该变更的 runner SHA-256 为 `b94aced8b7d459ef41461a2547a98375455122aa211c7bde58ac1d5f193b843d`，远端与本地一致；其余三个输入 SHA-256 未变，且与服务运行目录中的部署哈希一致。成功实测与完整复核见 `SELF_CHECK.md` 及 `evidence/`。
