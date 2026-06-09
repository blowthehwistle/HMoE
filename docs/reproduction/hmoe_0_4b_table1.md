# HMoE-0.4B Table 1 Reproduction

This run targets the paper's HMoE-0.4B Top-K row at `7e19` training FLOPs.

## Setup

Download LLaMA2 tokenizer assets:

```bash
HF_TOKEN=... sh scripts/dataset/download_llama2_tokenizer.sh
```

Check the FLOPs target:

```bash
python scripts/report/hmoe_repro_flops.py
```

Check training progress from saved checkpoints:

```bash
python scripts/report/check_hmoe_repro_status.py
```

## Training

Smoke run:

```bash
MODE=smoke NGPU=1 sh scripts/train/run_hmoe_0_4b_repro.sh
```

Full run:

```bash
MODE=full NGPU=8 sh scripts/train/run_hmoe_0_4b_repro.sh
```

The full config uses 29,100 steps, global batch 640, sequence length 4096, and LLaMA2 tokenizer vocabulary padded to 32,768.

## Evaluation

Export a TorchTitan DCP checkpoint to a single PyTorch checkpoint if needed:

```bash
CHECKPOINT_DIR=outputs/hmoe_0_4b_llama2_repro/checkpoint/step-29100 \
  OUTPUT_PT=outputs/eval/hmoe_0_4b_step29100.pt \
  sh scripts/eval/export_dcp_to_torch.sh
```

Run lm-eval after adapting/exporting the checkpoint to an lm-eval-loadable model:

```bash
PRETRAINED=/path/to/eval/checkpoint \
  OUTPUT_PATH=outputs/eval/hmoe_0_4b_table1_lm_eval.json \
  sh scripts/eval/run_lm_eval_hmoe_table1.sh
```

Compare an existing lm-eval output:

```bash
python scripts/report/compare_hmoe_table1.py outputs/eval/hmoe_0_4b_table1_lm_eval.json
```

Paper HMoE-0.4B Top-K reference values: PIQA 56.67, HellaSwag 28.26, BoolQ 59.80, ARC-Easy 31.93, WinoGrande 52.49, SIQA 32.91, AVG 43.68.
