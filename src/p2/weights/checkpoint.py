"""Pinned DeepSeek-V2-Lite-Chat topology and checkpoint index."""

import json
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"
REVISION = "85864749cd611b4353ce1decdb286193298f64c7"
BASE_URL = f"https://huggingface.co/{MODEL}/resolve/{REVISION}"
INDEX_NAME = "model.safetensors.index.json"
SHARDS = tuple(f"model-{i:05d}-of-000004.safetensors" for i in range(1, 5))
TOTAL_BYTES = 31_412_968_448
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def json_object(data):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    result = json.loads(data.decode("utf-8"), object_pairs_hook=unique_object)
    if not isinstance(result, dict):
        raise ValueError("expected a JSON object")
    return result


@dataclass(frozen=True)
class ExpectedTensor:
    shape: tuple[int, ...]
    layer_id: int | None = None
    expert_id: int | None = None
    role: str = ""
    dtype: str = "BF16"


def mlp_weights(prefix, width, layer, role, expert=None):
    for projection in PROJECTIONS:
        shape = (2048, width) if projection == "down_proj" else (width, 2048)
        yield f"{prefix}.{projection}.weight", ExpectedTensor(shape, layer, expert, role)


def routed_weights(layer, expert):
    if not 1 <= layer <= 26 or not 0 <= expert < 64:
        raise ValueError("routed layer must be 1..26 and expert ID must be 0..63")
    return dict(mlp_weights(f"model.layers.{layer}.mlp.experts.{expert}", 1408,
                            layer, "routed", expert))


def expected_weights():
    result = {
        "model.embed_tokens.weight": ExpectedTensor((102400, 2048), role="embedding"),
        "lm_head.weight": ExpectedTensor((102400, 2048), role="lm_head"),
        "model.norm.weight": ExpectedTensor((2048,), role="norm"),
    }
    attention = {
        "input_layernorm": (2048,),
        "post_attention_layernorm": (2048,),
        "self_attn.q_proj": (3072, 2048),
        "self_attn.kv_a_proj_with_mqa": (576, 2048),
        "self_attn.kv_a_layernorm": (512,),
        "self_attn.kv_b_proj": (4096, 512),
        "self_attn.o_proj": (2048, 2048),
    }
    for layer in range(27):
        prefix = f"model.layers.{layer}"
        for name, shape in attention.items():
            result[f"{prefix}.{name}.weight"] = ExpectedTensor(shape, layer, role="attention_or_norm")
        if layer == 0:
            result.update(mlp_weights(f"{prefix}.mlp", 10944, layer, "dense"))
            continue
        result[f"{prefix}.mlp.gate.weight"] = ExpectedTensor((64, 2048), layer, role="router")
        result.update(mlp_weights(f"{prefix}.mlp.shared_experts", 2816, layer, "shared"))
        for expert in range(64):
            result.update(routed_weights(layer, expert))
    return result


def validate_shard_name(name):
    if (not isinstance(name, str) or not name.endswith(".safetensors")
            or Path(name).name != name or PureWindowsPath(name).name != name
            or any(char in name for char in ':\\/\0')):
        raise ValueError(f"unsafe shard name: {name!r}")


def local_shard(root, name):
    validate_shard_name(name)
    root = root.resolve(strict=True)
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"shard is not a regular file inside checkpoint directory: {name}")
    return path


def parse_index(data):
    index = json_object(data)
    weights = index.get("weight_map")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("index must contain a nonempty weight_map")
    for key, name in weights.items():
        if not key or key == "__metadata__":
            raise ValueError(f"invalid tensor key: {key!r}")
        validate_shard_name(name)
    return index
