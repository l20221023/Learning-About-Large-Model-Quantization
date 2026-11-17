#!/usr/bin/env bash
set -euo pipefail
WORK=/root/autodl-tmp/p2-deepseek-moe
ENV_FILE="$WORK/autodl_storage_env.sh"
test -r "$ENV_FILE"
line="if [ -r $ENV_FILE ]; then . $ENV_FILE; fi # P2 data-volume cache policy"
if ! grep -Fq "$line" /root/.bashrc; then
    sed -i "1i $line" /root/.bashrc
fi
source "$ENV_FILE"
chmod +x "$WORK/audit_autodl_storage.sh" "$WORK/autodl_storage_env.sh" \
    "$WORK/autodl_storage_guard.sh" "$WORK/run_m0_4_autodl.sh"
PYTHON="$WORK/venv-vllm-0.26.0/bin/python"
"$PYTHON" - <<'PY'
import os, tempfile
from huggingface_hub import constants
import torch
from torch.hub import get_dir
expected = {
    "HF_HOME": "/root/autodl-tmp/cache/huggingface",
    "HF_HUB_CACHE": "/root/autodl-tmp/cache/huggingface/hub",
    "MODELSCOPE_CACHE": "/root/autodl-tmp/cache/modelscope",
    "XDG_CACHE_HOME": "/root/autodl-tmp/cache/xdg",
    "TORCH_HOME": "/root/autodl-tmp/cache/torch",
    "PIP_CACHE_DIR": "/root/autodl-tmp/cache/pip",
    "TMPDIR": "/root/autodl-tmp/tmp",
}
for name, value in expected.items():
    actual = os.environ.get(name)
    print(f"{name}={actual} {'PASS' if actual == value else 'FAIL'}")
    if actual != value: raise SystemExit(1)
print(f"huggingface_hub.HF_HOME={constants.HF_HOME}")
print(f"huggingface_hub.HF_HUB_CACHE={constants.HF_HUB_CACHE}")
print(f"torch.hub.get_dir={get_dir()}")
print(f"tempfile.gettempdir={tempfile.gettempdir()}")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
PY
"$PYTHON" -m pip cache dir
mkdir -p /root/autodl-tmp/logs
{
    echo "shell_startup_line=$line"
    echo 'cache_environment_verification=PASS'
    "$WORK/autodl_storage_guard.sh" check 0
    "$WORK/autodl_storage_guard.sh" run 1048576 -- true
    echo 'guard_preflight_and_monitored_run=PASS'
} | tee /root/autodl-tmp/logs/storage-config-verification.log
