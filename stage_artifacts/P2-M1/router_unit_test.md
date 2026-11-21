# Installed router op unit test

Tested the installed vLLM 0.26.0 `fused_topk` wrapper without loading a model.

## CPU backend attempt

Two rows of synthetic logits passed to `fused_topk(..., topk=6, scoring_func="softmax")` on CPU failed as expected because custom op `_moe_C::topk_softmax` has no CPU kernel (`NotImplementedError`, registered backend shown as CUDA and Meta). This records a backend limitation; it is not a semantic test result.

## CUDA unit test

Executed on CUDA device 0 using only two `[1,4]` BF16 dummy hidden rows and two `[1,8]` BF16 synthetic logit rows; no model loaded. The first row had ascending logits and the second descending logits. Invocation used `topk=6`, `renormalize=False`, `scoring_func="softmax"`.

Assertions passed:

- expert IDs were `[[7,6,5,4,3,2],[0,1,2,3,4,5]]`;
- weight tensor shape `[2,6]`, dtype FP32;
- ID tensor dtype int32;
- auxiliary token-expert-index shape `[2,6]`;
- selected weights were ordered descending and were not renormalized to sum to 1.

The CUDA op test verifies the router primitive contract only. It does not load the model, validate production logits, prove those IDs were dispatched by the live model, or produce measured routing data. The post-test API/GPU status check found the existing server had shut down; no restart was attempted.
