"""Extract the three tensors of a single routed expert from local files."""

from contextlib import ExitStack

from p2.weights.checkpoint import local_shard
from p2.weights.safetensors_io import copy_tensor, encode_header, read_local_header


def expert_plan(expected, weights):
    missing = expected.keys() - weights.keys()
    if missing:
        raise ValueError("expert keys missing from index: " + ", ".join(sorted(missing)))
    return [
        {"tensor_key": key, "shard": weights[key], "expected_shape": spec.shape,
         "expected_dtype": spec.dtype, "evidence": "[D][I][U]", "status": "PREDICTED"}
        for key, spec in expected.items()
    ]


def extract_expert(plan, shard_dir, output=None):
    with ExitStack() as stack:
        sources = {}
        for name in dict.fromkeys(row["shard"] for row in plan):
            path = local_shard(shard_dir, name)
            source = stack.enter_context(path.open("rb"))
            header, _ = read_local_header(source)
            sources[name] = source, header
        selected = []
        rows = []
        for row in plan:
            key = row["tensor_key"]
            source, header = sources[row["shard"]]
            tensor = header.tensors.get(key)
            if tensor is None:
                raise ValueError(f"{key} is missing from {row['shard']}")
            if tensor.shape != row["expected_shape"]:
                raise ValueError(f"{key}: expected shape {row['expected_shape']}, got {tensor.shape}")
            if tensor.dtype != row["expected_dtype"]:
                raise ValueError(f"{key}: expected dtype {row['expected_dtype']}, got {tensor.dtype}")
            rows.append(dict(row, actual_shape=tensor.shape, dtype=tensor.dtype,
                             data_offsets=tensor.offsets, header_sha256=header.sha256,
                             evidence="[D][I][H]", status="VERIFIED"))
            selected.append((key, source, header, tensor))
        if output is not None:
            entries = {"__metadata__": {"format": "pt"}}
            cursor = 0
            for key, _, _, tensor in selected:
                start, end = tensor.offsets
                size = end - start
                entries[key] = dict(tensor.serialize(), data_offsets=(cursor, cursor + size))
                cursor += size
            destination = output.open("xb")
            try:
                with destination:
                    destination.write(encode_header(entries))
                    for _, source, header, tensor in selected:
                        copy_tensor(source, destination, header, tensor)
            except BaseException:
                output.unlink()
                raise
        return rows
