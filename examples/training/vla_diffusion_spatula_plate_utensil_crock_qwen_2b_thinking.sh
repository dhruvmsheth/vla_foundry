#!/bin/bash
set -euo pipefail

# Train a Qwen3VLA SFT checkpoint for BimanualPutSpatulaOnPlateFromUtensilCrock.
# Note: this task is present in packaged eval results, but is not currently
# exposed by the public dataset downloader. Pass local dataset_manifest and
# dataset_statistics overrides if you have the task shards.

uv run torchrun --nproc_per_node=1 --nnodes=1 vla_foundry/main.py \
  --hparams.torchcompile False \
  --config_path vla_foundry/config_presets/training_jobs/vla_diffusion_spatula_plate_utensil_crock_qwen3vl_2b.yaml \
  --num_checkpoints 2 \
  --max_checkpoint_limit 1 \
  --total_train_samples 4096 \
  "$@"
