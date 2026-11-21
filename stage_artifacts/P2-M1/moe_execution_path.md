# P2-M1 MoE execution path (source audit completed)

## Runtime evidence

The historical B0 API and the Q1 service startup log identify the BF16 snapshot `/root/autodl-tmp/p2-deepseek-moe/models/DeepSeek-V2-Lite-Chat-BF16-85864749`, `max_model_len=2304`. The B0 log records vLLM `0.26.0`, `tensor_parallel_size=2`, `data_parallel_size=1`, `enable_return_routed_experts=False`, and `moe_backend='auto'`; the Q1 log confirms the same engine topology and `TritonExperts` runtime backend. The service has since been stopped after quality evaluation. Source inspected in the matching remote environment:

`/root/autodl-tmp/p2-deepseek-moe/venv-vllm-0.26.0/lib/python3.12/site-packages/vllm/`

Model config: `/root/autodl-tmp/p2-deepseek-moe/models/DeepSeek-V2-Lite-Chat-BF16-85864749/config.json`.

## Architecture confirmed from loaded config and model implementation

- 27 decoder layers; `first_k_dense_replace=1`, `moe_layer_freq=1`: layer 0 is dense and layers 1–26 are MoE (`deepseek_v2.py`, `DeepseekV2DecoderLayer` construction and `DeepseekV2Model` layer loop).
- Each MoE layer has 64 routed experts, `top_k=6`, hidden size 2048, expert intermediate size 1408. The config says `n_shared_experts=2`; vLLM constructs one `DeepseekV2MLP` with intermediate size `1408*2=2816` and adds it to routed output (unless fused shared expert support is active; runtime is NVIDIA/CUDA, so the ROCm AITER path is inapplicable).
- Router uses replicated `GateLinear(hidden_size=2048, output=64)`. Its per-token input is `[T,2048]`; logits are `[T,64]`. `fused_topk_router.py` allocates top-k weights as float32 and IDs as int32; CUDA `ops.topk_softmax` emits `[T,6]` weights and IDs. `DeepseekV2MoE.forward` passes logits and hidden states into `FusedMoE`; the runner returns `[T,2048]`.
- Model config states `scoring_func=softmax`, `topk_method=greedy`, `norm_topk_prob=false`, `n_group=1`, `topk_group=1`, `num_experts_per_tok=6`. This selects ordinary `FusedTopKRouter` without grouped routing or correction bias; it calls `ops.topk_softmax`, which applies softmax scoring and selects the top 6. Since `renormalize=false`, selected weights are not renormalized after top-k.
- Expert weights and hidden states are BF16 per engine config. `_get_moe_router_dtype` returns `None` for this config (only explicitly configured `"float32"` selects FP32), so `GateLinear` leaves output dtype at its normal model dtype; router logits are BF16. Fused top-k scores/weights are separately allocated as FP32, expert IDs are int32, and the Triton kernel returns the routed output in the model activation dtype.

## Call chain (Python to fused operator)

`DeepseekV2ForCausalLM.forward` → `DeepseekV2Model.forward` → `DeepseekV2DecoderLayer.forward` → `DeepseekV2MoE.forward` → `GateLinear` → `FusedMoE.forward` → router/top-k and `FusedMoE` runner → selected CUDA MoE backend kernel → weighted routed result, then shared MLP addition.

Primary implementation files:

- `vllm/model_executor/models/deepseek_v2.py`: model, decoder, `DeepseekV2MoE.__init__` and `.forward`.
- `vllm/model_executor/layers/fused_moe/layer.py`: FusedMoE construction, router creation and backend setup.
- `vllm/model_executor/layers/fused_moe/expert_map_manager.py`: logical-to-local expert mapping for EP.
- `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py`: fused softmax top-k wrapper; `ops.topk_softmax` is a CUDA custom op.
- `vllm/model_executor/layers/fused_moe/experts/triton_moe.py` and `vllm/model_executor/layers/fused_moe/fused_moe.py`: Triton expert runner and JIT kernel implementation.
- `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py`: BF16 TP-sharded expert weight allocation and modular method call.
- `vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py` / `fused_moe.py`: weighted top-k expert output reduction through `ops.moe_sum`.

Exact source evidence in the installed vLLM 0.26.0 tree (line numbers from the remote inspection):

| Claim | File and symbol | Evidence |
|---|---|---|
| MoE gate, shared MLP and FusedMoE construction | `model_executor/models/deepseek_v2.py`, `DeepseekV2MoE.__init__` | lines 277–397; gate dimensions and shared width at 308–362 |
| Router and expert call | same file, `DeepseekV2MoE.forward` | lines 397–422; gate at 415, fused call at 416 |
| Layer 0 dense / following layers MoE | same file, `DeepseekV2DecoderLayer.__init__` | class at 1177; layer selection at 1221 onward |
| Model layer loop and CausalLM entry | same file | `DeepseekV2Model` at 1347; forward at 1413; `DeepseekV2ForCausalLM` at 1783 |
| FusedMoE factory and runner selection | `model_executor/layers/fused_moe/layer.py`, `FusedMoE` | line 100 onward |
| Standard top-k tensors and CUDA op | `model_executor/layers/fused_moe/router/fused_topk_router.py` | `fused_topk` at 69; `FusedTopKRouter` at 116; CUDA call is `ops.topk_softmax` at 24 |
| EP disabled/default topology handling | `model_executor/layers/fused_moe/config.py` | `FusedMoEParallelConfig` at 1032; `make` at 1124; `use_ep=False, ep_size=1` path at 1217–1228 |
| BF16 weight tensors with all experts and TP-local intermediate | `model_executor/layers/fused_moe/unquantized_fused_moe_method.py` | `create_weights` at 88; `w13_weight` allocation at 102; `w2_weight` at 121 |
| Triton expert implementation | `model_executor/layers/fused_moe/experts/triton_moe.py` | `TritonExperts` at 54; reduction helper uses `ops.moe_sum` at 539 |
| Triton fused expert kernel and output reduction | `model_executor/layers/fused_moe/fused_moe.py` | `fused_moe_kernel` at 295; dispatch/call sites 853 onward and 1727 onward; reduction at 1790 |

Runtime backend selection evidence is the B0 log: lines 37 and 49 report TRITON Unquantized and `TritonExperts`; line 53 reports the selected E=64,N=704 RTX 3090 default config warning; line 127 records `fused_moe_kernel` Triton JIT compilation during inference. This joins the Python construction path to the actually selected CUDA/Triton path.

## TP=2 expert placement

The server was started with TP=2 and DP=1; its logged non-default arguments contain no `--enable-expert-parallel`, and `/config/parallel.py` defaults `enable_expert_parallel=False`. In `/model_executor/layers/fused_moe/config.py`, `FusedMoEParallelConfig.make` returns `ep_size=1`, `use_ep=False` when EP is disabled. This deployment does **not** divide the 64 global expert IDs into 32-per-GPU EP shards. `unquantized_fused_moe_method.py` allocates `w13_weight` with `num_experts` as dimension 0 and the TP-local intermediate size; both TP ranks therefore represent all 64 routed experts with tensor-parallel-sharded matrices. The server log corroborates the TP-local intermediate `N=704` (`1408/2`) and reports `Using TritonExperts MoE backend`. It also reports `Using TRITON Unquantized MoE backend`; JIT monitor records `fused_moe_kernel` compiled during inference. The exact loaded parameter shapes were not queried in-process, so do not use this audit for exact VRAM accounting.

As a non-production CPU unit check, `determine_expert_map(2, rank, 64)` yielded 32 local IDs for each hypothetical EP rank. This only validates the generic EP mapping utility; it is **not** the runtime topology (runtime `use_ep=False`).

No hidden-state or expert-ID execution trace was collected in this audit. The API flag `enable_return_routed_experts=False` indicates the service does not expose routed IDs through the public response.
