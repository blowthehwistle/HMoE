#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-hf}
PRETRAINED=${PRETRAINED:-}
OUTPUT_PATH=${OUTPUT_PATH:-outputs/eval/hmoe_0_4b_table1_lm_eval.json}
BATCH_SIZE=${BATCH_SIZE:-auto}
TASKS=${TASKS:-piqa,hellaswag,boolq,arc_easy,winogrande,social_iqa}
BASELINE=${BASELINE:-hmoe_0_4b_topk}

if [ -z "${PRETRAINED}" ]; then
  echo "Set PRETRAINED to an lm-eval-loadable checkpoint directory." >&2
  echo "For this repo, that should be an exported/converted HMoE checkpoint or a custom lm-eval model adapter target." >&2
  exit 2
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")"

lm_eval \
  --model "${MODEL}" \
  --model_args "pretrained=${PRETRAINED}" \
  --tasks "${TASKS}" \
  --num_fewshot 0 \
  --batch_size "${BATCH_SIZE}" \
  --output_path "${OUTPUT_PATH}"

python scripts/report/compare_hmoe_table1.py "${OUTPUT_PATH}" --baseline "${BASELINE}"
