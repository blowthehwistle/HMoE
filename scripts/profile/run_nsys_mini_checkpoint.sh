#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-hmoe}
CHECKPOINT_STEP=${CHECKPOINT_STEP:-100}
PROFILE_STEPS=${PROFILE_STEPS:-4}
NGPU=${NGPU:-1}
KEEP_PROFILE_CHECKPOINT=${KEEP_PROFILE_CHECKPOINT:-false}
NVTX_PROFILE=${NVTX_PROFILE:-1}

case "${MODEL}" in
  hmoe)
    MODE=mini_25g_hmoe
    SOURCE_RUN=outputs/mini_25g/hmoe_0_4b_topk
    ;;
  moe)
    MODE=mini_25g_moe
    SOURCE_RUN=outputs/mini_25g/moe_0_4b_topk
    ;;
  *)
    echo "Unsupported MODEL=${MODEL}. Use MODEL=hmoe or MODEL=moe." >&2
    exit 2
    ;;
esac

SOURCE_CHECKPOINT="${SOURCE_RUN}/checkpoint/step-${CHECKPOINT_STEP}"
PROFILE_ROOT="outputs/nsys_compare/${MODEL}_step${CHECKPOINT_STEP}"
if [ "${NVTX_PROFILE}" = "1" ]; then
  PROFILE_KIND=nvtx
else
  PROFILE_KIND=perf
fi
PROFILE_NAME="${MODEL}_${PROFILE_KIND}"
PROFILE_RUN="${PROFILE_ROOT}/${PROFILE_KIND}_run"
FINAL_STEP=$((CHECKPOINT_STEP + PROFILE_STEPS))
if [ ! -f "${SOURCE_CHECKPOINT}/.metadata" ]; then
  echo "Checkpoint metadata not found: ${SOURCE_CHECKPOINT}/.metadata" >&2
  exit 1
fi

mkdir -p "${PROFILE_ROOT}"
rm -rf "${PROFILE_RUN}/checkpoint"

nsys profile \
  --output "${PROFILE_ROOT}/${PROFILE_NAME}" \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --sample=none \
  --stats=true \
  env MODE="${MODE}" NGPU="${NGPU}" TTP_SKIP_FINAL_CHECKPOINT=1 TTP_NVTX_PROFILE="${NVTX_PROFILE}" \
  bash scripts/train/run_hmoe_0_4b_repro.sh \
    --job.dump-folder "${PROFILE_RUN}" \
    --training.steps "${FINAL_STEP}" \
    --checkpoint.initial-load-path "${SOURCE_CHECKPOINT}" \
    --checkpoint.no-initial-load-model-only \
    --checkpoint.no-enable-first-step-checkpoint \
    --checkpoint.interval 1000000 \
    --validation.no-enabled \
    --metrics.no-enable-tensorboard

if [ "${KEEP_PROFILE_CHECKPOINT}" != "true" ]; then
  rm -rf "${PROFILE_RUN}/checkpoint"
fi

echo "Profile report: ${PROFILE_ROOT}/${PROFILE_NAME}.nsys-rep"
