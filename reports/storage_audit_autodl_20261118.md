# AutoDL Storage Audit and Protection Report

**Audit time:** 2026-11-17 to 2026-11-18, Asia/Shanghai  
**Scope:** the user supplied RTX 3090 instance and `/root/autodl-tmp`; no files were deleted or moved.

## 1. Measured filesystems

| Mount | Filesystem | Size | Used | Available | Guard reserve |
|---|---|---:|---:|---:|---:|
| `/` | overlay | 30G | 204M | 30G | 8 GiB |
| `/root/autodl-tmp` | XFS, `/dev/nvme1n1` | 50G | 16G | 35G | 10 GiB |
| `/autodl-pub` | shared XFS | 7.0T | 6.5T | 503G | not used |

Byte-level free-space samples after M0-4 were 31,998,775,296 bytes on `/` and 37,565,095,936 bytes on `/root/autodl-tmp`. `/autodl-pub` is mounted `ro` and therefore is not a download or output destination. Another block device appeared in `lsblk` without a usable mount or filesystem path; it was not used.

## 2. Existing data and caches

The initial data volume contents were approximately 15G. The audit attributed about 8.5G to the project model directory, 6.5G to the isolated `venv-vllm-0.26.0`, and 19M to earlier experiment results. The two model shards are 5,000,045,752 and 4,086,561,120 bytes, 9,086,606,872 bytes total. The small remainder includes project metadata and AutoDL files. After the M0-4 output and safety scripts were added, the data volume reported 16G used and 35G available.

Before configuration, `/root/.cache` was 4K and `/tmp` was 11M. Cache variables were unset; the isolated pip cache defaulted to `/root/.cache/pip`. The project venv has `huggingface_hub`, PyTorch and pip installed; the `modelscope` Python package is not installed. The existing AWQ model and venv remain in their original locations under `/root/autodl-tmp/p2-deepseek-moe/`.

No download, inference or package process was running at audit time. No `*.incomplete`, `*.partial`, `*.part` or `*.tmp` files were found. One zero-byte setuptools vendor `.lock` file was present inside the venv; it is not a model download lock and was left untouched. The first unsuccessful M0-4 setup logs and their small partial outputs are preserved in the evidence archive.

## 3. Persistent data-volume configuration

Created the requested data-volume layout at `/root/autodl-tmp/{models,cache,experiments,logs,tmp}`. Existing project models were left in place. The reusable [environment setup](../scripts/autodl_storage_env.sh) exports:

- `HF_HOME=/root/autodl-tmp/cache/huggingface`
- `HF_HUB_CACHE` and `HUGGINGFACE_HUB_CACHE` under that HF directory
- `MODELSCOPE_CACHE=/root/autodl-tmp/cache/modelscope`
- `XDG_CACHE_HOME=/root/autodl-tmp/cache/xdg`
- `TORCH_HOME=/root/autodl-tmp/cache/torch`
- `PIP_CACHE_DIR=/root/autodl-tmp/cache/pip`
- `TMPDIR=/root/autodl-tmp/tmp`
- Triton and Matplotlib caches under `/root/autodl-tmp/cache/`

The setup is sourced by the project runner and is also loaded at the top of `/root/.bashrc`, before its non-interactive-shell early return. Both an isolated Python process and a fresh non-interactive SSH command resolved HF, Torch Hub, pip and tempfile paths to the data volume. The ModelScope destination is configured, but its SDK-specific behavior could not be tested because the SDK is absent.

## 4. Capacity protection

[The storage guard](../scripts/autodl_storage_guard.sh) checks the requested peak data/system bytes before a task starts and preserves at least 10 GiB on the data volume and 8 GiB on `/`. Its `run` mode monitors both filesystems every five seconds. If a reserve is crossed, it sends SIGINT to the task process group, allows up to 60 seconds for orderly exit, then sends SIGTERM and returns status 75.

The positive preflight and monitored-run path passed. A separate 999,999,999,999-byte request was rejected with status 75, and its `touch` command did not run. The monitor was exercised under normal free-space conditions; an actual reserve-crossing shutdown was not forced because that would require filling a user volume.

The [M0-4 runner](../scripts/run_m0_4_autodl.sh) uses a 512 MiB output peak estimate, then runs through the guard. Its completed M0-4 outputs occupied about 11 MiB uncompressed; the compressed evidence package was about 6.2 MiB.

## 5. Model capacity assessment

The existing 9.09 GB AWQ checkpoint is already downloaded, loaded and served a real request on the 3090. It is sufficient for the completed M0-4 prefill profile and the planned small AWQ reference experiments, which need no additional model download.

The fixed official BF16 checkpoint is 31,412,968,448 bytes (29.26 GiB). The data volume currently has about 34.99 GiB free, but preserving the 10 GiB guard reserve leaves only about 24.99 GiB for a new write. A direct full-checkpoint download would leave about 5.73 GiB free, below the reserve, before accounting for any duplicate temporary files. The full BF16 download is therefore blocked under the current policy.

Converting BF16 to a second quantized checkpoint would require the source and output to coexist, plus conversion workspace. That peak also exceeds this volume's safe capacity. The current instance should not start that conversion. If the official BF16 path or quantization stage becomes necessary, provide a larger writable data volume or another authorized writable storage target, then re-run a workflow-specific peak estimate. `/autodl-pub` is read-only and is not a workaround. No cleanup, re-download or expansion was performed.

## 6. Evidence and reproduction

The raw disk audit, environment verification, capacity rejection, GPU sample, final trace, prompt manifest and preserved failed-attempt logs are in [the M0-4 evidence archive](../experiments/results/M0-4-20261117T235424-3090/M0-4-20261117T235424-3090.tar.gz). Archive SHA-256: `d6a604207117c34390fcc00b70c834ad49f748877a20e9a2831d0a948ef55d20`.

To check a future operation before execution, source the environment and pass a conservative peak estimate in bytes:

```bash
source /root/autodl-tmp/p2-deepseek-moe/autodl_storage_env.sh
bash /root/autodl-tmp/p2-deepseek-moe/autodl_storage_guard.sh check PEAK_DATA_BYTES [PEAK_SYSTEM_BYTES]
bash /root/autodl-tmp/p2-deepseek-moe/autodl_storage_guard.sh run PEAK_DATA_BYTES -- COMMAND [ARGS...]
```

No model storage path was changed. The existing 3090 instance remains available with 0 MiB allocated after the experiment; the official BF16 and quantization workflows remain blocked by storage headroom.
