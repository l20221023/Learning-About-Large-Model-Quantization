# Installed vLLM Routed Experts Capture assessment

Inspected the installed remote vLLM 0.26.0 tree at:

`/root/autodl-tmp/p2-deepseek-moe/venv-vllm-0.26.0/lib/python3.12/site-packages/vllm/`

The feature is suitable for the requested expert-ID trace when using the V1 runner; no custom vLLM source overlay is needed.

- `model_executor/layers/fused_moe/router/base_router.py:185` (`set_capture_fn`) stores a callback; lines 295–299 in `_select_experts` call it on logical Top-K IDs before EPLB mapping and return the same IDs/scores to the FusedMoE runner. This model has EPLB off, so logical IDs are the IDs dispatched to the 64 experts.
- `v1/worker/gpu_model_runner.py:7635–7645` (`init_routed_experts_capturer`) allocates the device-side capture buffer. Lines 7677–7690 bind each MoERunner's `layer_id` to its router callback. Lines 3714–3719 copy the buffer and slot mapping to pinned CPU memory nonblocking; line 4733 materializes the routing data.
- `model_executor/layers/fused_moe/routed_experts_capturer.py:93–110` allocate the `int32` buffer and define `capture`; lines 200–212 write IDs and clear unused slots. The manager stores routes by KV block slot and reads them back for reused prefix-cache blocks.
- `v1/core/sched/scheduler.py:1633–1638` stores route batches by slot; lines 1782–1796 retrieve prompt routes and normal decode routes for the actual scheduled model inputs. Speculative-token branches were not used.
- `entrypoints/openai/completion/protocol.py:600` declares `routed_experts`; `entrypoints/openai/completion/serving.py:574–578` serializes it as a base64 NumPy `.npy` array. The response also returned generated `token_ids`.
- `config/vllm.py:2127` (`_get_v2_model_runner_unsupported_features`) and lines 2188–2190 reject `enable_return_routed_experts` under Model Runner V2. The Q1 service log says `Using V2 Model Runner`; the M3 ON/OFF pair therefore explicitly uses the V1 runner. Runtime logs show both TP workers initialized the built-in capturer.

Known schema limits:

- Each response is mapped to its single synchronous API request by the retained HTTP response ID. The internal scheduler request ID is not exposed by the public response and remains `UNKNOWN`.
- Prompt rows map to the frozen prompt token IDs. Subsequent rows map to processed generated input tokens; the final generated token is not routed if it was never fed back into a model forward. Therefore the expected route-row count is `prompt_tokens + completion_tokens - 1` for ordinary greedy generation.
- The capture tensor includes a dense layer 0 slot, which is zero-filled; routed MoE layers are indices 1 through 26. The audit must exclude layer 0 and verify exactly 26 layers × Top-6 per returned row.
- One array shape `[route_rows, 27, 6]` contains one logical routing decision per token/layer; TP rank is not a returned dimension, and route-row count is checked against the exact processed-token count to reject doubled TP accounting.
- This built-in returns expert IDs, not router scores/weights. The current acceptance requires IDs; weights are not claimed.

The source hashes, runtime arguments, captured raw response arrays and independent audit are recorded separately in this directory. No installed source file is modified; `source_diff.patch` is expected to be empty.
