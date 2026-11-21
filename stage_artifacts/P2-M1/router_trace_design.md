# P2-M1 Router trace design (proposal; no trace collected)

## Recommended insertion point

For the current vLLM 0.26.0 deployment, first evaluate `DeepseekV2MoE.forward` immediately after `self.gate(hidden_states)` as the least invasive Python point. At that location, hidden token rows and router logits are available before `FusedMoE` dispatch. However, IDs and `topk_weight` are not yet produced there; add a narrow observer to the actual router callable created by `FusedMoE` (identified from runtime backend selection) or add an opt-in return/observer at the router boundary. Do not hook each expert kernel or copy activations.

Record bounded batches on device and asynchronously drain after the forward/scheduler step. Per row capture: model layer index from module prefix, global routed expert IDs, weights, batch row/token position, and a scheduler-provided request/sequence mapping. Label prefill/decode using scheduler metadata, not token-count heuristics. The model layer prefix is a stable source for `layer_id`; engine-local token row does not by itself identify request or absolute token position.

## Required association

Request ID and phase must originate from V1 scheduler/model-runner metadata and accompany each flattened token through packing/permutation. If the runtime cannot provide a reliable row-to-request map, omit request ID and label phase unknown; never infer them from output ordering. Log a unique trace-session ID, batch/step ID, rank, layer ID, token row, routed IDs, weights, and validity mask.

## Evidence that IDs executed

Before claiming real expert execution, compare captured router IDs against the dispatch metadata consumed by the selected fused kernel for the same layer/batch, and verify counts sum to `valid_tokens * 6` (excluding padding/sentinel rows). Also run a tiny deterministic in-process unit test with known router logits and assert returned IDs/weights against expected top-6. This validates routing semantics, but alone does not prove production dispatch; add a backend dispatch counter or device-side audit in a separate controlled run.

The historical API config logged `enable_return_routed_experts=False`; no trace API result is available. A CUDA unit test of the installed top-k op passed, but it does not observe model routing. The service has since exited (see `runtime_status_20261120.md`), so this remains a design only, not measured router data.
