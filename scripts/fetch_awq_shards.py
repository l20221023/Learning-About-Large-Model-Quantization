"""Resume the two pinned, user-authorized AWQ shards via verified byte ranges."""

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPException
from pathlib import Path
from time import sleep
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SOURCE = "https://modelscope.cn/api/v1/models/TechxGenus-MS/DeepSeek-V2-Lite-Chat-AWQ/repo"
REVISION = "d57fd1322a85b9d9fe44d397b2db374970753e9a"
SHARDS = {
    "model-00001-of-00002.safetensors":
        (5000045752, "a3ce643aff668a2a6234d5d684bd5ec109d30ef5444462e6f21c8118c6d3d7af"),
    "model-00002-of-00002.safetensors":
        (4086561120, "6991191b1bb8bda54c8fb14f49225e0b3994f3094b7c2d6486f62cc211c25f39"),
}
CHUNK_SIZE = 256 * 1024**2
BUFFER = 1024**2


def fetch_range(name, size, start, end, path):
    url = SOURCE + "?" + urlencode({"Revision": REVISION, "FilePath": name})
    for attempt in range(4):
        existing = path.stat().st_size if path.exists() else 0
        remaining = end - start - existing + 1
        if remaining == 0:
            return
        if remaining < 0:
            raise ValueError(f"{path}: part is larger than its requested interval")
        first = start + existing
        request = Request(url, headers={"Range": f"bytes={first}-{end}", "Accept-Encoding": "identity"})
        try:
            with urlopen(request, timeout=35) as response:
                if (response.status != 206 or response.headers.get("Content-Range") != f"bytes {first}-{end}/{size}"
                        or response.headers.get("Content-Encoding", "identity") != "identity"):
                    raise ValueError(f"{name}: server did not return the requested interval")
                with path.open("ab") as stream:
                    while remaining:
                        chunk = response.read(min(BUFFER, remaining))
                        if not chunk:
                            raise OSError("range ended early")
                        stream.write(chunk)
                        remaining -= len(chunk)
            print(f"{name} range={start}-{end} complete", flush=True)
            return
        except (URLError, OSError, HTTPException) as error:
            # The CDN's signed redirect is intentionally excluded from the log.
            print(f"{name} range={first}-{end} attempt={attempt+1} error={type(error).__name__}", flush=True)
            if attempt == 3:
                raise RuntimeError(f"range retries exhausted for {name}:{first}-{end}") from None
            sleep(2**attempt)


def merge_files(destination, paths, size, expected):
    temp = destination.with_suffix(".assembling")
    digest = hashlib.sha256()
    with temp.open("xb") as output:
        for path in paths:
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(BUFFER), b""):
                    output.write(block)
                    digest.update(block)
    if temp.stat().st_size != size or digest.hexdigest() != expected:
        raise ValueError(f"{destination.name}: assembly failed SHA-256 or size verification")
    if destination.exists():
        raise FileExistsError(destination)
    temp.rename(destination)
    print(f"{destination.name} bytes={size} sha256={expected} verified", flush=True)


def download(destination, part_dir, workers):
    part_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        plans = []
        for name, (size, expected) in SHARDS.items():
            final = destination / name
            if final.exists():
                with final.open("rb") as stream:
                    valid = final.stat().st_size == size and hashlib.file_digest(stream, "sha256").hexdigest() == expected
                if not valid:
                    raise ValueError(f"{name}: existing completed file differs from pinned shard")
                print(f"{name} already complete, SHA-256 verified", flush=True)
                continue
            prefix = destination / (name + ".part")
            offset = prefix.stat().st_size if prefix.exists() else 0
            plan_file = part_dir / (name + ".json")
            plan = {"prefix_bytes": offset, "range_bytes": CHUNK_SIZE}
            if plan_file.exists():
                if json.loads(plan_file.read_bytes()) != plan:
                    raise ValueError(f"{name}: saved prefix or segment size changed")
            else:
                plan_file.write_text(json.dumps(plan), encoding="utf-8")
            paths = [prefix] if offset else []
            print(f"{name} bytes={size} resuming_from={offset}", flush=True)
            futures = []
            for start in range(offset, size, CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, size) - 1
                part = part_dir / f"{name}.{start}-{end}.part"
                paths.append(part)
                futures.append(executor.submit(fetch_range, name, size, start, end, part))
            plans.append((final, paths, size, expected, futures))
        for final, paths, size, expected, futures in plans:
            for future in futures:
                future.result()
            merge_files(final, paths, size, expected)
            for path in paths:
                path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--part-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=6, choices=range(1, 9))
    args = parser.parse_args()
    free = shutil.disk_usage(args.model_dir).free
    if free < max(size for size, _ in SHARDS.values()) * 2:
        raise RuntimeError("need free space for downloaded parts and a temporary assembled shard")
    download(args.model_dir, args.part_dir, args.workers)


if __name__ == "__main__":
    main()
