#!/usr/bin/env bash
set -euo pipefail

MODE=${MODE:-mini_25g_hmoe}
NGPU=${NGPU:-}

case "${MODE}" in
  full)
    CONFIG_FILE=${CONFIG_FILE:-"./ttp/experiments/hmoe/train_configs/paper_hmoe_0_4b_llama2_repro.toml"}
    ;;
  redpajama_full)
    CONFIG_FILE=${CONFIG_FILE:-"./ttp/experiments/hmoe/train_configs/paper_hmoe_0_4b_llama2_redpajama_repro.toml"}
    ;;
  moe_redpajama_full)
    CONFIG_FILE=${CONFIG_FILE:-"./ttp/experiments/hmoe/train_configs/paper_moe_0_4b_llama2_redpajama_repro.toml"}
    ;;
  mini_25g_hmoe)
    CONFIG_FILE=${CONFIG_FILE:-"./ttp/experiments/hmoe/train_configs/paper_hmoe_0_4b_llama2_redpajama_mini_25g.toml"}
    ;;
  mini_25g_moe)
    CONFIG_FILE=${CONFIG_FILE:-"./ttp/experiments/hmoe/train_configs/paper_moe_0_4b_llama2_redpajama_mini_25g.toml"}
    ;;
  *)
    echo "Unsupported MODE=${MODE}. Use MODE=full, MODE=redpajama_full, MODE=moe_redpajama_full, MODE=mini_25g_hmoe, or MODE=mini_25g_moe." >&2
    exit 2
    ;;
esac

if [ -z "${NGPU}" ]; then
  CONFIG_FILE="${CONFIG_FILE}" sh scripts/train/run_train.sh "$@"
else
  CONFIG_FILE="${CONFIG_FILE}" NGPU="${NGPU}" sh scripts/train/run_train.sh "$@"
fi
