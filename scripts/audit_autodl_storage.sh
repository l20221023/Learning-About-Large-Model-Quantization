#!/usr/bin/env bash
# Read-only audit for the current AutoDL user's mount, cache and partial-file state.
set +e
echo '=== filesystems ==='
df -hT / /root/autodl-tmp
echo '=== block devices and mounts ==='
lsblk -o NAME,FSTYPE,SIZE,FSAVAIL,FSUSE%
echo '=== other mounts (capacity and mount options) ==='
df -hT /autodl-pub 2>/dev/null
findmnt -no TARGET,SOURCE,FSTYPE,OPTIONS /autodl-pub 2>/dev/null
echo '=== data volume usage by top-level directory ==='
du -x -h --max-depth=2 /root/autodl-tmp 2>/dev/null | sort -h | tail -80
echo '=== root cache usage ==='
for path in /root/.cache /root/.cache/huggingface /root/.cache/modelscope /root/.cache/torch /root/.cache/pip /tmp; do
    if test -e "$path"; then du -sh "$path" 2>/dev/null; fi
done
echo '=== relevant cache path variables ==='
for key in HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE MODELSCOPE_CACHE XDG_CACHE_HOME TORCH_HOME TMPDIR PIP_CACHE_DIR; do
    value="${!key-}"
    if test -n "$value"; then printf '%s=%s\n' "$key" "$value"; else printf '%s=<unset>\n' "$key"; fi
done
echo "python_on_path=$(command -v python || true)"
echo '=== isolated environment cache defaults/package availability ==='
P2_PY=/root/autodl-tmp/p2-deepseek-moe/venv-vllm-0.26.0/bin/python
if test -x "$P2_PY"; then
    "$P2_PY" - <<'PY'
import importlib.util
for name in ("huggingface_hub", "modelscope", "torch", "pip"):
    spec = importlib.util.find_spec(name)
    print(f"{name}={'installed' if spec else 'not installed'}" + (f" module={spec.origin}" if spec else ""))
try:
    import modelscope.hub.constants as c
    for name in ("MODELSCOPE_CACHE", "DEFAULT_CACHE_ROOT"):
        if hasattr(c, name): print(f"modelscope.{name}={getattr(c, name)}")
except Exception as e:
    print(f"modelscope cache constant unavailable: {type(e).__name__}")
PY
    "$P2_PY" -m pip cache dir 2>/dev/null
fi
echo '=== likely download or model processes (arguments omitted) ==='
ps -C python,python3,curl,wget,aria2c,hf,modelscope,pip -o pid,comm,etime 2>/dev/null
echo '=== existing shell initialization (first 35 lines) ==='
if test -f /root/.bashrc; then head -35 /root/.bashrc; else echo '/root/.bashrc absent'; fi
echo '=== previous M0-4 launch attempt files/log tail ==='
find /root/autodl-tmp/p2-deepseek-moe/results -maxdepth 2 -path '*M0-4*' -type f -printf '%p %s bytes\n' 2>/dev/null
for log in /root/autodl-tmp/p2-deepseek-moe/results/M0-4-20261117T230000-3090/run.log /root/autodl-tmp/p2-deepseek-moe/results/M0-4-20261117T230000-3090/trace/manifest.json; do
    if test -f "$log"; then echo "--- $log ---"; tail -40 "$log"; fi
done
echo '=== model weight files under data volume (paths and sizes only) ==='
find /root/autodl-tmp -xdev -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.gguf' -o -name '*.pt' -o -name '*.pth' \) -printf '%p %s bytes\n' 2>/dev/null | sort
echo '=== incomplete/temporary/lock files (not deleted) ==='
find /root/autodl-tmp /root/.cache /tmp -xdev -type f \( -name '*.incomplete' -o -name '*.partial' -o -name '*.part' -o -name '*.tmp' -o -name '*.lock' \) -printf '%p %s bytes\n' 2>/dev/null | head -200
