"""Read-only prefill tracing for DeepSeek-V2-Lite's greedy gate."""

import json
import math
import struct
from contextlib import ExitStack

import torch
from torch.nn import functional as F

EXPERTS = 64
TOP_K = 6


def _fp16_rounding_matches(observed, expected):
    if not math.isfinite(observed) or not math.isfinite(expected) or not 0.0 <= observed <= 1.0:
        return False
    try:
        encoded = struct.pack("<e", observed)
    except (OverflowError, struct.error):
        return False
    rounded, = struct.unpack("<e", encoded)
    if rounded != observed:
        return False
    bits, = struct.unpack("<H", encoded)
    previous, = struct.unpack("<e", struct.pack("<H", max(0, bits - 1)))
    following, = struct.unpack("<e", struct.pack("<H", bits + 1))
    slack = expected * 2e-7
    return (previous + observed) / 2 - slack <= expected <= (observed + following) / 2 + slack


def write_trace(path, rows, *, evidence):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            record = {"evidence": evidence, **row}
            stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")


def check_gate(gate):
    expected = {
        "n_routed_experts": EXPERTS, "top_k": TOP_K, "scoring_func": "softmax",
        "topk_method": "greedy", "norm_topk_prob": False, "routed_scaling_factor": 1.0,
    }
    for field, value in expected.items():
        if getattr(gate, field, None) != value:
            raise ValueError(f"unsupported gate configuration: {field} must equal {value!r}")
    if gate.weight.ndim != 2 or gate.weight.shape[0] != EXPERTS or not gate.weight.is_floating_point():
        raise ValueError("expected a floating-point gate weight with one row per routed expert")


def checked_routes(logits, ids, weights, *, softmax_dtype=torch.float32):
    if logits.shape[-1] != EXPERTS or not torch.isfinite(logits).all():
        raise ValueError("invalid or non-finite router logits")
    if ids.shape != weights.shape or ids.shape != (len(logits), TOP_K) or ids.dtype != torch.long:
        raise ValueError("expected six native routed IDs and weights per token")
    if not ((0 <= ids) & (ids < EXPERTS)).all():
        raise ValueError("routed expert ID outside 0..63")
    sorted_ids = ids.sort(dim=-1).values
    if (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any():
        raise ValueError("duplicate routed expert ID")
    if softmax_dtype is torch.float16:
        # CUDA and CPU FP16 softmax may land on opposite sides of a rounding midpoint.
        reference = logits.to(torch.float64).softmax(dim=-1)
        probabilities = reference.to(torch.float16).float()
        if not torch.equal(weights.float(), weights.to(torch.float16).float()):
            raise ValueError("native selected weights are not FP16 representable")
        for row in range(len(logits)):
            for slot in range(TOP_K):
                expert = int(ids[row, slot].item())
                if not _fp16_rounding_matches(float(weights[row, slot].item()),
                                               float(reference[row, expert].item())):
                    raise ValueError(
                        "native selected weight is outside the FP16 softmax rounding interval "
                        f"(token_row={row}, slot={slot}, expert={expert})"
                    )
        expected_ids = logits.topk(TOP_K, dim=-1).indices.sort(dim=-1).values
        observed_ids = ids.sort(dim=-1).values
        for row in range(len(logits)):
            if not torch.equal(observed_ids[row], expected_ids[row]):
                threshold = probabilities[row].topk(TOP_K, sorted=True).values[-1]
                bits, = struct.unpack("<H", struct.pack("<e", float(threshold.item())))
                previous, = struct.unpack("<e", struct.pack("<H", max(0, bits - 1)))
                selected_scores = probabilities[row].gather(0, ids[row])
                if float(selected_scores.min().item()) < previous:
                    raise ValueError("native selected experts are outside the greedy Top-6")
    else:
        probabilities = logits.softmax(dim=-1, dtype=softmax_dtype).float()
        gathered = probabilities.gather(1, ids)
        if not torch.allclose(weights.float(), gathered, atol=2e-7, rtol=2e-6):
            raise ValueError("native selected weights differ from unnormalized softmax probabilities")
        expected = probabilities.topk(TOP_K, sorted=True).values
        if not torch.allclose(weights.float().sort(dim=-1, descending=True).values, expected,
                              atol=2e-7, rtol=2e-6):
            raise ValueError("native selected experts are outside the greedy Top-6")
    return probabilities


def trace_prefill(model, gates, inputs, *, excluded_token_ids):
    """Compare complete model logits and capture the original gate's output.

    Full gate logits are replayed from the actual gate input and weight.
    Each module fires once, and no hooks are left installed on exit.
    """
    if model.training or not gates:
        raise ValueError("trace requires an eval-mode model and at least one routed gate")
    if len({id(gate) for gate in gates.values()}) != len(gates):
        raise ValueError("expected a distinct gate module for each layer")
    input_ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    if input_ids.ndim != 2 or input_ids.shape != mask.shape or not ((mask == 0) | (mask == 1)).all():
        raise ValueError("expected paired [batch, sequence] IDs and binary attention mask")
    present = mask.bool()
    active = present.clone()
    for token_id in excluded_token_ids:
        active &= input_ids != token_id
    token_indices = active.flatten().nonzero(as_tuple=True)[0].cpu()
    if not token_indices.numel():
        raise ValueError("no valid tokens after padding and special-token filtering")
    width = input_ids.shape[1]
    token_ids = input_ids.flatten().cpu().index_select(0, token_indices).tolist()
    coordinates = [divmod(index, width) for index in token_indices.tolist()]
    by_layer = {}

    def gather(layer):
        def capture(gate, args, output):
            if layer in by_layer:
                raise ValueError(f"unexpected repeated gate call for layer {layer}")
            hidden, = args
            if hidden.ndim != 3 or hidden.shape[:2] != input_ids.shape:
                raise ValueError(f"layer {layer}: gate shape differs from input token layout")
            ids, weights, _ = output
            if ids.shape != (input_ids.numel(), TOP_K) or weights.shape != ids.shape:
                raise ValueError(f"layer {layer}: native gate returned inconsistent token count")
            selection = token_indices.to(hidden.device)
            flat = hidden.detach().view(-1, hidden.shape[-1])
            logits = F.linear(flat.float(), gate.weight.detach().float(), None)
            logits = logits.index_select(0, selection)
            ids = ids.index_select(0, selection)
            weights = weights.index_select(0, selection)
            probabilities = checked_routes(logits, ids, weights)
            by_layer[layer] = [
                {"layer_id": layer, "batch_id": batch, "sequence_index": position,
                 "token_id": token_id, "router_logits": logit,
                 "router_probabilities": probability, "topk_expert_ids": expert_ids,
                 "topk_weights": expert_weights}
                for (batch, position), token_id, logit, probability, expert_ids, expert_weights in zip(
                    coordinates, token_ids, logits.cpu().tolist(), probabilities.cpu().tolist(),
                    ids.cpu().tolist(), weights.cpu().tolist(), strict=True
                )
            ]
        return capture

    for gate in gates.values():
        check_gate(gate)
    with torch.inference_mode():
        baseline = model(**inputs, use_cache=False).logits.detach().clone()
        with ExitStack() as hooks:
            for layer, gate in gates.items():
                handle = gate.register_forward_hook(gather(layer))
                hooks.callback(handle.remove)
            traced = model(**inputs, use_cache=False).logits
        if by_layer.keys() != gates.keys():
            raise ValueError(f"missing gate calls for layers {sorted(gates.keys() - by_layer.keys())}")
        if baseline.shape != traced.shape or not torch.isfinite(baseline).all() or not torch.isfinite(traced).all():
            raise ValueError("model logits differ in shape or contain non-finite values")
        max_difference = (baseline.float() - traced.float()).abs().max().item()
        if not torch.equal(baseline, traced):
            raise ValueError(f"tracing changed model logits (max abs difference {max_difference})")
    rows = [row for layer in sorted(by_layer) for row in by_layer[layer]]
    return rows, {
        "layer_ids": sorted(by_layer), "input_shape": list(input_ids.shape),
        "valid_tokens": len(coordinates), "padding_tokens": int((~present).sum()),
        "special_tokens": int((present & ~active).sum()),
        "records": len(rows), "logits_shape": list(traced.shape),
        "logits_dtype": str(traced.dtype), "logits_exact_equal": True,
        "logits_max_abs_difference": max_difference,
    }
