#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/p2-deepseek-moe/venv-vllm-0.26.0/bin/python}"
DATA_DIR="${DATA_DIR:-data}"
RAW_DIR="${RAW_DIR:-quality_raw}"
REPORT_DIR="${REPORT_DIR:-.}"
export PATH="$(dirname "$PYTHON_BIN"):${PATH:-/usr/local/bin:/usr/bin:/bin}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ -e "$RAW_DIR/quality_formal.jsonl" ]]; then
  echo "refusing to overwrite existing formal results; set RAW_DIR to a new directory"
  exit 2
fi

"$PYTHON_BIN" validate_quality_config.py quality_config.yaml
"$PYTHON_BIN" prepare_quality_samples.py \
  --config quality_config.yaml \
  --data-dir "$DATA_DIR" \
  --output quality_samples.jsonl
"$PYTHON_BIN" audit_quality_samples.py \
  --config quality_config.yaml \
  --samples quality_samples.jsonl \
  --data-dir "$DATA_DIR"

# quality_runner gates the formal run on all three 10-row smoke checks.
"$PYTHON_BIN" quality_runner.py \
  --config quality_config.yaml \
  --samples quality_samples.jsonl \
  --raw-dir "$RAW_DIR"

"$PYTHON_BIN" audit_quality.py \
  --config quality_config.yaml \
  --samples quality_samples.jsonl \
  --raw "$RAW_DIR/quality_formal.jsonl" \
  --out-dir "$REPORT_DIR"
