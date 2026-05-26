#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_train.sh
AVAILABLE_GPUS=$(python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")
NGPU=${NGPU:-${AVAILABLE_GPUS}}
export LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./ttp/experiments/hmoe/train_configs/debug_model.toml"}
MASTER_PORT=${MASTER_PORT:-"5001"}
MASTER_ADDR=${MASTER_ADDR:-${LOCAL_IP:-"localhost"}}

if [ "${AVAILABLE_GPUS}" -eq 0 ]; then
    echo "No CUDA devices are available."
    echo "Check the NVIDIA driver/CUDA visibility before starting distributed training."
    exit 1
elif [ "${AVAILABLE_GPUS}" -lt "${NGPU}" ]; then
    echo "Requested NGPU=${NGPU}, but only ${AVAILABLE_GPUS} CUDA device(s) are available."
    echo "Check the NVIDIA driver/CUDA visibility, or run with a smaller NGPU value."
    exit 1
fi

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" --master_port ${MASTER_PORT} --master_addr ${MASTER_ADDR} \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
-m ttp.train --job.config_file ${CONFIG_FILE} "$@"
