# AWQ Wikitext-2 PPL 补齐方案

执行更新（2026-11-18）：原 S3 ZIP 返回重定向及 403，无法取得预期 SHA。实际使用 Salesforce 仓库固定提交 `b08601e04326c79dfdd32d625aee71d232d685c3` 的 test Parquet，文件 SHA-256 `5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`；逐行保留原有换行，对无换行行补 `\n`，得到 1,289,123 bytes 的连续文本。该协议与下文原 ZIP 计划有所不同，不能声称字节等价。3090 全量 PPL 结果和原始证据见 `reports/Phase1_validation.md`。

## 固定数据与授权元数据

使用 `Salesforce/wikitext` 的 `wikitext-2-raw-v1` test split，仅 test 用于 PPL；固定官方 dataset_infos 1.0.0。官方 zip URL 为 `https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip`，大小 4,721,645 bytes，SHA-256 `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`。执行端获授权下载 ZIP 后先校验 ZIP 字节 SHA，再只读取精确成员 `wikitext-2-raw/wiki.test.raw`；应为 1,305,092 bytes、4,358 行，并另记解包 test 文件 SHA。test 文件保持原始 UTF-8 全文与换行/空行，作为单一流一次性 tokenize；不把行拆成独立请求。`train`/`valid` 不进入质量指标，仅可用于 GPTQ calibration 可行性研究。

授权元数据存在差异：当前 Salesforce/Hugging Face 数据卡标注 CC BY-SA 4.0；仓库侧 SPDX 元数据还列 CC-BY-SA-3.0 与 GFDL。此处保留差异并请求按组织流程核验，不作法律结论。拟用归因信息：Salesforce AI Research, *The WikiText Long Term Dependency Language Modeling Dataset*, dataset `wikitext-2-raw-v1`, dataset card and repository, access date 2026-11-18; citation: Stephen Merity, Caiming Xiong, James Bradbury, Richard Socher, “Pointer Sentinel Mixture Models,” ICLR 2017. Attribute source, preserve ShareAlike notices as applicable; do not claim legal clearance.

## 指标及实现

在**本地已有**社区 AWQ snapshot 和经批准下载的固定数据 ZIP 后，使用 vLLM 0.26.0、RTX 3090 运行 `wikitext_awq_ppl.py --data-zip <zip>`。原始 test text 不套 Chat Template、不拼额外 BOS/EOS，按 UTF-8 原样作为一个整体用 tokenizer `encode(add_special_tokens=False)` 一次 tokenize。窗口长 256、stride 128，以 `SamplingParams(prompt_logprobs=1, max_tokens=1)` 要求 prompt token logprob。对每个目标位置核对返回行含目标 token ID 且 logprob 有限；验证全流 N 个 token 中恰好计 N-1 个 causal targets。汇总 `NLL=-sum(log p)`、mean NLL、`PPL=exp(mean NLL)`、目标 token 数、原始逐窗分数，以及 ZIP/test/tokenizer IDs/model/runtime/config 哈希；elapsed 含模型初始化，另记 PPL 计时和失败 manifest。

主要限制：独立 AWQ PPL 只描述此 checkpoint 在固定语料和 tokenizer 下的 next-token fit，不代表相对 BF16/GPTQ 的退化或质量保持；缺少原始教师模型对照仍是 Phase 1 缺项。数据处理版本、换行、tokenizer 与 context/stride 影响可比性，必须随结果归档。

## 空间峰值及守护

预计 zip 下载 4,721,645 bytes（约 4.51 MiB），解包原始文件约 13.5 MiB，转换后的 test JSON 预计 <10 MiB，逐窗分数保守按 1 GiB 预算（逐 token 记录可能约数十至数百 MiB）；另留 256 MiB 临时/manifest。运行前用 1 GiB 新增数据盘峰值预检，整个下载、解包和 GPU PPL 执行均受系统盘 8 GiB、数据盘 10 GiB 保留线；下载数据需总调度单独部署/授权后执行。此任务当前只准备 CPU 代码，不下载、不运行 GPU。

阶段分类仍应是 `MEASURED_COMMUNITY_AWQ`（若实际运行成功），PPL 标作单模型探索性质量指标；合规元数据和许可状态记为 UNKNOWN/待核。
