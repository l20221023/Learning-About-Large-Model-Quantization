#!/usr/bin/env bash
# Source this file in every AutoDL shell or job before downloading/running models.
AUTODL_DATA_ROOT=/root/autodl-tmp
P2_WORK_ROOT="$AUTODL_DATA_ROOT/p2-deepseek-moe"
export HF_HOME="$AUTODL_DATA_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export MODELSCOPE_CACHE="$AUTODL_DATA_ROOT/cache/modelscope"
export XDG_CACHE_HOME="$AUTODL_DATA_ROOT/cache/xdg"
export TORCH_HOME="$AUTODL_DATA_ROOT/cache/torch"
export PIP_CACHE_DIR="$AUTODL_DATA_ROOT/cache/pip"
export TMPDIR="$AUTODL_DATA_ROOT/tmp"
export TRITON_CACHE_DIR="$AUTODL_DATA_ROOT/cache/triton"
export MPLCONFIGDIR="$AUTODL_DATA_ROOT/cache/matplotlib"
export P2_MODELS_DIR="$P2_WORK_ROOT/models"
export P2_EXPERIMENTS_DIR="$AUTODL_DATA_ROOT/experiments"
export P2_LOGS_DIR="$AUTODL_DATA_ROOT/logs"
export P2_TMP_DIR="$AUTODL_DATA_ROOT/tmp"
mkdir -p "$AUTODL_DATA_ROOT/models" "$AUTODL_DATA_ROOT/cache" "$AUTODL_DATA_ROOT/experiments" \
    "$AUTODL_DATA_ROOT/logs" "$AUTODL_DATA_ROOT/tmp" "$HF_HUB_CACHE" "$MODELSCOPE_CACHE" \
    "$XDG_CACHE_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"
mkdir -p "$P2_WORK_ROOT/workloads"
