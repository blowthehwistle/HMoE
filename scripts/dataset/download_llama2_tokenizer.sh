#!/usr/bin/env bash
set -euo pipefail

REPO_ID=${REPO_ID:-"meta-llama/Llama-2-7b-hf"}
LOCAL_DIR=${LOCAL_DIR:-"assets/hf"}

args=(
  3rdparty/torchtitan/scripts/download_hf_assets.py
  --repo_id "${REPO_ID}" \
  --local_dir "${LOCAL_DIR}" \
  --assets tokenizer config
)

if [ -n "${HF_TOKEN:-}" ]; then
  args+=(--hf_token "${HF_TOKEN}")
fi

python "${args[@]}"
