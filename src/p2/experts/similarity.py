"""Numerical helpers for small same-input routed expert comparisons."""

import torch


def output_similarity(left, right, *, eps=1e-12):
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("expert outputs must be same-shape one-dimensional vectors")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("expert output contains a non-finite value")
    a, b = left.double(), right.double()
    norm_a, norm_b = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    denom = norm_a * norm_b
    cosine = float(torch.dot(a, b) / denom) if float(denom) > eps else 0.0
    relative_l2 = float(torch.linalg.vector_norm(a - b) / max(float(norm_a), eps))
    return {
        "cosine_similarity": cosine,
        "relative_l2": relative_l2,
        "left_norm": float(norm_a),
        "right_norm": float(norm_b),
        "norm_ratio_right_over_left": float(norm_b / max(float(norm_a), eps)),
        "absolute_l2": float(torch.linalg.vector_norm(a - b)),
    }


def reconstruct_routed_output(expert_outputs, expert_ids, router_weights):
    if len(expert_ids) != len(router_weights) or not expert_ids:
        raise ValueError("router IDs and weights must be non-empty and aligned")
    if len(set(expert_ids)) != len(expert_ids):
        raise ValueError("router expert IDs must be unique")
    if any(expert not in expert_outputs for expert in expert_ids):
        raise ValueError("missing expert output for a native routed ID")
    values = [expert_outputs[expert] for expert in expert_ids]
    if any(value.shape != values[0].shape or value.ndim != 1 for value in values):
        raise ValueError("routed expert outputs must be aligned vectors")
    result = torch.zeros_like(values[0], dtype=torch.float32)
    for value, weight in zip(values, router_weights, strict=True):
        result.add_(value.float(), alpha=float(weight))
    return result


def fixed_weight_replacement_relative_l2(base_routed, source_output, replacement_output,
                                         source_router_weight, *, eps=1e-12):
    if base_routed.shape != source_output.shape or base_routed.shape != replacement_output.shape:
        raise ValueError("replacement and routed outputs must share a shape")
    base = base_routed.double()
    delta = float(source_router_weight) * (replacement_output.double() - source_output.double())
    denominator = max(float(torch.linalg.vector_norm(base)), eps)
    return float(torch.linalg.vector_norm(delta) / denominator)
