# AGENT.md

Short operational notes for agents working in this HMoE repository.

## Overview

- Project: `HMoE: Heterogeneous Mixture of Experts`
- Core package: `ttp`
- Base framework: TorchTitan under `3rdparty/torchtitan`
- Training entrypoint: `python -m ttp.train`
- Default training wrapper: `scripts/train/run_train.sh`

## Key Paths

```text
ttp/                         HMoE implementation
  train.py                   main training entrypoint and TTPTrainer
  config/job_config.py       TorchTitan JobConfig extensions
  components/                dataloader, loss, metrics, tokenizer, checkpoint
  datasets/                  HF and mmap dataset loaders
  experiments/hmoe/          HMoE model, configs, parallelization
  inference/                 checkpoint loading and generation

scripts/
  init/deps.sh               conda/PyTorch/dependency setup
  train/run_train.sh         torchrun training launcher
  train/run_hmoe_0_4b_repro.sh  HMoE/MoE mini_25g and full-run wrappers
  inference/run_generate.sh  generation launcher; currently has placeholders
  dataset/download_fineweb-edu.py
  dataset/download_redpajama.py  bounded RedPajama shard downloader

3rdparty/torchtitan/         vendored/submodule TorchTitan
torchtitan                   symlink to 3rdparty/torchtitan/torchtitan
assets/hf/                   tokenizer assets
fineweb_10BT/                FineWeb sample data
redpajama_1T/                symlink to external RedPajama subset storage
outputs/                     checkpoints, logs, figures, profiles
reference/                   original paper/reference material
```

## Important Files

- `ttp/train.py`: registers models, parses `TTPJobConfig`, chooses `TTPTrainer` or TorchTitan `Trainer`, and adds HMoE auxiliary losses/metrics.
- `ttp/experiments/hmoe/__init__.py`: registers `TrainSpec(name="hmoe")` and defines model flavors such as `debugmodel`, `paper_hmoe_0_4b`, and `paper_hmoe_3b`.
- `ttp/experiments/hmoe/model/`: Hybrid Transformer, MoE, HMoE, and model args.
- `ttp/experiments/hmoe/train_configs/`: runnable TOML configs.

## Setup

```bash
sh scripts/init/deps.sh
```

Expected environment: Python 3.12+, CUDA 12.6+, PyTorch nightly cu126.

Tokenizer assets:

```bash
python 3rdparty/torchtitan/scripts/download_hf_assets.py \
  --repo_id deepseek-ai/deepseek-moe-16b-base \
  --assets tokenizer
```

Demo FineWeb data:

```bash
python scripts/dataset/download_fineweb-edu.py
```

## Training

Default debug run:

```bash
sh scripts/train/run_train.sh
```

Custom config/GPU count:

```bash
CONFIG_FILE="./ttp/experiments/hmoe/train_configs/debug_model.toml" \
NGPU=1 \
sh scripts/train/run_train.sh
```

Common configs:

- `debug_model.toml`: quick debug run using TorchTitan C4 test data.
- `paper_hmoe_0_4b.toml`, `paper_hmoe_3b.toml`: paper-style model configs.
- `paper_hmoe_0_4b_fineweb_split.toml`, `paper_hmoe_3b_fineweb_split.toml`: FineWeb train/validation split configs.
- `paper_hmoe_0_4b_llama2_redpajama_repro.toml`: paper-near HMoE-0.4B run, 29,100 steps for ~`7e19` FLOPs.

Paper-near RedPajama flow:

```bash
# one-time / before each new run; freezes completed shards into manifests
python scripts/dataset/build_redpajama_splits.py --root redpajama_1T
python scripts/dataset/check_redpajama_splits.py --root redpajama_1T

# 1 GPU local run
MODE=redpajama_full NGPU=1 bash scripts/train/run_hmoe_0_4b_repro.sh
```

The RedPajama subset was intentionally capped around 100GiB for a local ~2-week run. Do not restart unbounded full RedPajama download unless explicitly requested.

Checkpoint handoff to another workstation:

```bash
python scripts/dataset/make_redpajama_handoff.py --root redpajama_1T

# on the next workstation, avoid reusing local shards
python scripts/dataset/build_redpajama_splits.py \
  --root redpajama_1T \
  --exclude-relative-manifest outputs/handoff/redpajama_local_100g/used_shards_relative.txt

MODE=redpajama_full NGPU=4 bash scripts/train/run_hmoe_0_4b_repro.sh \
  --checkpoint.exclude_from_loading data_loader
```

## Inference

`scripts/inference/run_generate.sh` still contains placeholders:

```bash
TITAN_CODE_PATH=...
CONFIG_FILE="##/##.toml"
CHECKPOINT_DIR="##/checkpoint/step-##"
```

Use real values before running:

```bash
TITAN_CODE_PATH="/home/hwi/research/Heterogeneous MoE/HMoE" \
CONFIG_FILE="./ttp/experiments/hmoe/train_configs/debug_model.toml" \
CHECKPOINT_DIR="./outputs/<run>/checkpoint/step-<n>" \
PROMPT="Washington, DC is the capital of the " \
NGPU=1 \
sh scripts/inference/run_generate.sh
```

## Notes

- Prefer changing `ttp/` extension code over editing `3rdparty/torchtitan/`.
- Check the original paper/reference material under `reference/` when working on model design, experiment settings, or paper-aligned claims.
- A config must use `model.name="hmoe"` and a `model.flavor` key defined in `ttp/experiments/hmoe/__init__.py`.
- `training.type_name="ttp"` uses `TTPTrainer`; `"titan"` uses vanilla TorchTitan `Trainer`.
- Pipeline parallelism currently rejects `pipeline_parallel_degree > 1` with `global_batch_size > 0`.
- For RedPajama runs, configs read frozen `redpajama_1T/splits/*_manifest.txt`; do not rebuild splits mid-run or immediately before exact checkpoint resume.
- Treat `outputs/`, `fineweb_10BT/`, `redpajama_1T/`, and `assets/hf/` as large/generated data unless asked otherwise.
- Current untracked paths observed: `.codex/`, `reference/`, and this `AGENT.md`. Do not remove user files without explicit request.
