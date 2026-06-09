#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR=${CHECKPOINT_DIR:-}
OUTPUT_PT=${OUTPUT_PT:-outputs/eval/hmoe_0_4b_checkpoint.pt}

if [ -z "${CHECKPOINT_DIR}" ]; then
  echo "Set CHECKPOINT_DIR to a TorchTitan DCP checkpoint, e.g. outputs/.../checkpoint/step-29100." >&2
  exit 2
fi

mkdir -p "$(dirname "${OUTPUT_PT}")"
python -m torch.distributed.checkpoint.format_utils dcp_to_torch "${CHECKPOINT_DIR}" "${OUTPUT_PT}"
echo "Exported ${CHECKPOINT_DIR} -> ${OUTPUT_PT}"
