#!/usr/bin/bash

set -e

# TODO: code path
TITAN_CODE_PATH=...

# use envs as local overrides for convenience
# e.g.
NGPU=${NGPU:-"1"}
LOG_RANK=${LOG_RANK:-0}

# TODO
CONFIG_FILE=${CONFIG_FILE:-"##/##.toml"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"##/checkpoint/step-##"}

PROMPT=${PROMPT:-"Washington, DC is the capital of the "}

overrides=()
if [ $# -ne 0 ]; then
	for arg in "$@"; do
		# special case to handle prompt in quotes
		if [[ "$arg" == --prompt=* ]]; then
			PROMPT="${arg#--prompt=}"
            # check if file
            if [[ -f "$PROMPT" ]]; then
                PROMPT=$(<"$PROMPT")
            fi
		else
			# handle other args
			overrides+=("$arg")
		fi
	done
fi

export PYTHONPATH=$TITAN_CODE_PATH:$PYTHONPATH
cd ${TITAN_CODE_PATH}

set -x
torchrun --standalone \
	--nproc_per_node="${NGPU}" \
	--local-ranks-filter="${LOG_RANK}" \
	ttp/inference/test_generate.py \
	--config="${CONFIG_FILE}" \
	--checkpoint="${CHECKPOINT_DIR}" \
	--prompt="${PROMPT}" \
	--max_new_tokens=128 \
	--temperature=1 \
	--top_k=1 \
	"${overrides[@]}"
