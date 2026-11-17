"""Local HF loader and environment inventory for the M0-3 prefill probe."""

import hashlib
import importlib.metadata
import platform
import shutil
import sys

import torch

from p2.weights.checkpoint import INDEX_NAME, SHARDS, TOTAL_BYTES

# Byte identities from the M0-2 and M0-3 pinned Hugging Face retrieval manifests.
SOURCE_SHA256 = {
    "config.json": "f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d",
    "configuration_deepseek.py": "6de5e7445ab490a16482027b131d5bd6a003da9b58a182a11432115fac84ce59",
    "modeling_deepseek.py": "7d8e5221095286eea991137760893fd7ba52727c0b4ebf48ec09e8bc56b45b9c",
}
TOKENIZER_SHA256 = {
    "tokenizer.json": "41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669",
    "tokenizer_config.json": "31181eaf79394ea26728d95ecb54fe7c8413e6f56085dbabc8b0818134380ec8",
}
PINNED_FILES = {
    **SOURCE_SHA256, **TOKENIZER_SHA256,
    INDEX_NAME: "d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf",
}
SHARD_SIZES = dict(zip(SHARDS, (8_594_887_408, 8_591_757_448, 8_590_718_520, 5_636_263_200), strict=True))


def digest(path):
    with path.open("rb") as stream:
        sha256 = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(block)
    return sha256.hexdigest()


def windows_display_drivers():
    if sys.platform != "win32":
        return []
    import winreg

    path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    adapters = []
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
        count, _, _ = winreg.QueryInfoKey(key)
        for index in range(count):
            name = winreg.EnumKey(key, index)
            if not name.isdecimal():
                continue
            try:
                with winreg.OpenKey(key, name) as adapter:
                    description = winreg.QueryValueEx(adapter, "DriverDesc")[0]
                    version = winreg.QueryValueEx(adapter, "DriverVersion")[0]
            except (FileNotFoundError, PermissionError):
                continue
            adapters.append({"description": description, "driver_version": version})
    return adapters


def environment():
    packages = {}
    for name in ("torch", "transformers", "huggingface-hub", "tokenizers", "safetensors",
                 "accelerate", "psutil", "vllm", "triton", "flash-attn",
                 "autoawq", "auto-gptq", "gptqmodel", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        import psutil
        memory = psutil.virtual_memory()
        ram = {"total_bytes": memory.total, "available_bytes": memory.available}
    except ImportError:
        ram = None
    try:
        adapters = windows_display_drivers()
    except OSError:
        adapters = None
    cuda = []
    for index in range(torch.cuda.device_count()):
        info = torch.cuda.get_device_properties(index)
        cuda.append({"index": index, "name": info.name, "total_bytes": info.total_memory,
                     "compute_capability": [info.major, info.minor]})
    a = torch.ones((2, 2), dtype=torch.bfloat16)
    product = a @ a
    return {
        "platform": platform.platform(), "python": sys.version, "executable": sys.executable,
        "packages": packages, "system_ram": ram, "windows_display_drivers": adapters,
        "nvidia_smi": shutil.which("nvidia-smi"), "nvcc": shutil.which("nvcc"),
        "torch_cuda_build": torch.version.cuda, "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_devices": cuda, "cpu_bf16_matmul": {"dtype": str(product.dtype),
                                                        "values": product.tolist()},
    }


def check_snapshot(snapshot):
    problems = []
    evidence = {}
    if (snapshot / "model.safetensors").exists():
        problems.append("unexpected unsharded model.safetensors would take precedence over official shards")
    for name, expected in PINNED_FILES.items():
        path = snapshot / name
        if not path.is_file():
            problems.append(f"missing {name}")
        elif digest(path) != expected:
            problems.append(f"{name}: SHA-256 differs from official revision")
        else:
            evidence[name] = expected
    sizes = {}
    for name, expected in SHARD_SIZES.items():
        path = snapshot / name
        if not path.is_file():
            problems.append(f"missing {name}")
            continue
        sizes[name] = path.stat().st_size
        if sizes[name] != expected:
            problems.append(f"{name}: expected {expected} bytes, got {sizes[name]}")
    return problems, {"metadata_sha256": evidence, "local_shard_bytes": sizes}


def preflight(snapshot, device, runtime):
    problems, evidence = check_snapshot(snapshot)
    for package in ("transformers", "accelerate", "safetensors"):
        if not runtime["packages"][package]:
            problems.append(f"missing {package} in active Python environment")
    if device == "cuda":
        if not runtime["torch_cuda_available"]:
            problems.append("CUDA is unavailable in the active PyTorch environment")
        elif not torch.cuda.is_bf16_supported(including_emulation=False):
            problems.append("default CUDA device lacks native BF16 support")
        elif torch.cuda.mem_get_info()[0] < TOTAL_BYTES:
            problems.append("free VRAM is below the pinned BF16 weight bytes")
    elif runtime["system_ram"] and runtime["system_ram"]["available_bytes"] < TOTAL_BYTES:
        problems.append("available RAM is below the pinned BF16 weight bytes")
    return problems, evidence


def load_local_model(snapshot, device):
    from transformers import AutoModelForCausalLM

    model, diagnostics = AutoModelForCausalLM.from_pretrained(
        str(snapshot.resolve()), trust_remote_code=True, local_files_only=True,
        use_safetensors=True, torch_dtype=torch.bfloat16, device_map={"": device},
        attn_implementation="eager", output_loading_info=True,
    )
    errors = {key: values for key, values in diagnostics.items() if values}
    if errors:
        raise ValueError(f"checkpoint loader reported incomplete or unexpected weights: {errors}")
    return model.eval()


def encode_chat_prompt(snapshot, prompt, *, max_tokens):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot.resolve()), local_files_only=True,
                                              use_fast=True, trust_remote_code=False)
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                         tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    if not 0 < encoded["input_ids"].shape[1] <= max_tokens:
        raise ValueError("tokenized probe exceeds its input budget")
    return encoded, sorted(set(tokenizer.all_special_ids)), {
        "tokenizer_class": type(tokenizer).__name__,
        "formatted_prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def get_native_gates(model, expected_layers):
    layers = model.model.layers
    if len(layers) != expected_layers:
        raise ValueError(f"expected {expected_layers} decoder layers, got {len(layers)}")
    if hasattr(layers[0].mlp, "gate"):
        raise ValueError("layer 0 must be dense")
    gates = {}
    for layer, module in enumerate(layers[1:], start=1):
        moe = module.mlp
        if len(moe.experts) != 64 or not hasattr(moe, "shared_experts"):
            raise ValueError(f"layer {layer}: unexpected Routed/Shared topology")
        gates[layer] = moe.gate
    return gates
