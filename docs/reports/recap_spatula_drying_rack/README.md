# RECAP Smoke Experiment: Spatula on Plate from Drying Rack

This is the compact experiment record for the first VLA Foundry RECAP smoke run on
`BimanualPutSpatulaOnPlateFromDryingRack`.

The goal was not to claim a final benchmark. The goal was to prove the full loop
works end to end:

1. Train an SFT policy on the LBM task data.
2. Roll out that policy in LBM eval and collect observation/action trajectories.
3. Label trajectories with terminal success/failure returns.
4. Train a Qwen LoRA value model over camera observations.
5. Score rollout steps with the value model.
6. Run a small RECAP policy update from positive-advantage examples.
7. Re-evaluate SFT and RECAP on the exact same scenario IDs.

## Result

Matched evaluation task: `BimanualPutSpatulaOnPlateFromDryingRack`

Matched scenario IDs: `100-109`

| Policy | Success |
|---|---:|
| SFT12k checkpoint 3 | 5/10, 50% |
| RECAP policy, lr `2e-7`, 500 steps | 6/10, 60% |

![SFT vs RECAP](assets/sft_vs_recap_spatula_drying_rack_100_109.png)

Per-scenario outcomes:

| Scenario | SFT12k | RECAP |
|---:|---|---|
| 100 | success | success |
| 101 | fail | fail |
| 102 | success | fail |
| 103 | success | success |
| 104 | success | success |
| 105 | fail | success |
| 106 | fail | fail |
| 107 | fail | success |
| 108 | fail | success |
| 109 | success | fail |

## Value Model

Value-target source:

- SFT rollout checkpoint: `qwen_spatula_plate_drying_rack_ft_12k_lr2e5`, checkpoint 3
- Scenario IDs used for value data: `150-349`
- Episodes: 200
- Step examples: 42,299
- Train examples: 33,170
- Validation examples: 9,129
- Rollout success rate in this collection: 65.5%
- Value bins: 50
- Failure penalty: `c_fail=500`
- Value normalization: RECAP-style normalized return over a 300-step horizon

Qwen LoRA value model validation:

| Metric | Value |
|---|---:|
| validation MAE | 0.292 |
| validation value MSE | 0.178 |
| validation bin accuracy | 0.377 |
| mean predicted value, success episodes | -0.469 |
| mean predicted value, failure episodes | -0.670 |
| success-failure prediction gap | 0.201 |

![Value scoring summary](assets/recap_value_scoring_summary.png)

The value model is not a polished critic yet, but it separates successful and
failed rollouts enough to generate a usable first set of RECAP weights.

## W&B Curves

The static plots below were generated from the W&B histories for the SFT run,
Qwen LoRA value-model run, and RECAP policy-update run.

![W&B training curves](assets/wandb_training_curves.png)

![W&B value validation curves](assets/wandb_value_validation_curves.png)

Run links:

| Run | W&B URL |
|---|---|
| SFT policy, 12k samples | `https://wandb.ai/dsheth_caltech/vla_foundry_recap/runs/xbwvtv3x` |
| Qwen LoRA value model | `https://wandb.ai/dsheth_caltech/vla_foundry_recap/runs/5a69hryp` |
| RECAP policy update | `https://wandb.ai/dsheth_caltech/vla_foundry_recap/runs/kwlsr6ux` |
| Export/artifact sync | `https://wandb.ai/dsheth_caltech/vla_foundry_recap/runs/erf3i1n3` |
| README curve update | `https://wandb.ai/dsheth_caltech/vla_foundry_recap/runs/t3ckse32` |

The W&B run metadata used for these plots is stored in
[`wandb_runs.json`](wandb_runs.json).

Note: the SFT and value-model W&B runs are marked `crashed` because the runtime
was interrupted after the needed checkpoints and metrics had been written. The
exported Hugging Face artifacts are the source of truth for reproduction.

## RECAP Scoring

Scored examples: 42,299

Positive-advantage examples: 12,151

Positive-advantage fraction: 28.7%

Weighting:

- `weight_mode=exp`
- `beta=4.0`
- `max_weight=5.0`
- mean weight over all examples: 1.168
- mean weight over success examples: 1.326
- mean weight over failure examples: 1.000

Policy update:

- Base checkpoint: SFT12k checkpoint 3
- Trainable positive-advantage examples: 12,138
- Optimizer LR: `2e-7`
- Steps: 500
- Batch size: 1

## Artifacts

The Git repo contains only lightweight code and report assets. Large experiment
artifacts are stored on Hugging Face.

Hugging Face model repos:

- `https://huggingface.co/dhruvmsheth/vla-foundry-spatula-drying-rack-sft12k-ckpt3`
- `https://huggingface.co/dhruvmsheth/vla-foundry-spatula-drying-rack-recap-lr2e7-500`
- `https://huggingface.co/dhruvmsheth/vla-foundry-spatula-drying-rack-qwen-value-lora`

Hugging Face dataset repo:

- `https://huggingface.co/datasets/dhruvmsheth/vla-foundry-recap-spatula-drying-rack-artifacts`

Public Hugging Face bucket:

- `https://huggingface.co/buckets/dhruvmsheth/pi06star_recap`

Saved value-function visualizations:

- HF folder: `https://huggingface.co/datasets/dhruvmsheth/vla-foundry-recap-spatula-drying-rack-artifacts/tree/main/value_visualizations`
- HTML index: `https://huggingface.co/datasets/dhruvmsheth/vla-foundry-recap-spatula-drying-rack-artifacts/blob/main/value_visualizations/index.html`
- Included files: 5 rollout plots and 5 MP4s, with 3 success and 2 failure trajectories.

Download all exported artifacts:

```bash
hf download dhruvmsheth/vla-foundry-recap-spatula-drying-rack-artifacts \
  --repo-type dataset \
  --local-dir /workspace/vla_recap/hf_exports/recap_artifacts
```

Download only the model repos:

```bash
hf download dhruvmsheth/vla-foundry-spatula-drying-rack-sft12k-ckpt3 \
  --local-dir /workspace/vla_recap/hf_exports/sft12k_ckpt3

hf download dhruvmsheth/vla-foundry-spatula-drying-rack-recap-lr2e7-500 \
  --local-dir /workspace/vla_recap/hf_exports/recap_lr2e7_500

hf download dhruvmsheth/vla-foundry-spatula-drying-rack-qwen-value-lora \
  --local-dir /workspace/vla_recap/hf_exports/qwen_value_lora
```

W&B project:

- entity: `dsheth_caltech`
- project: `vla_foundry_recap`
- export run: `recap_spatula_drying_rack_export_20260428`

## Reproduction

The commands below assume the RunPod/LBM image setup used during the experiment:
`toyotaresearch/lbm-eval-oss:vla-foundry`, with the repo cloned to
`/workspace/vla_recap/src/vla_foundry`.

### RunPod Template

Use a custom GPU Pod template. Docker privileges are not required.

| Setting | Value |
|---|---|
| Template type | GPU Pod |
| Container image | `toyotaresearch/lbm-eval-oss:vla-foundry` |
| Container start command | `/bin/bash -lc "sleep infinity"` |
| Container disk | `100-120 GB` recommended |
| Persistent storage | Network volume mounted at `/workspace` |
| Network volume size | `500 GB` minimum, `1 TB` comfortable |
| SSH terminal access | Enabled |
| Jupyter notebook | Optional |
| HTTP service | Optional, only needed for browser-viewing HTML artifacts |

Initial pod setup:

```bash
mkdir -p /workspace/vla_recap/src /workspace/vla_recap/{data,outputs,hf,cache,tmp}
cd /workspace/vla_recap/src

git clone -b recap-posttraining-v0 https://github.com/dhruvmsheth/vla_foundry.git
cd vla_foundry

uv sync --group inference --group preprocessing
uv pip install -e .
```

Set the environment:

```bash
export PROJECT_ROOT=/workspace/vla_recap
export VLA_ROOT=$PROJECT_ROOT/src/vla_foundry
export VLA_SITE=$VLA_ROOT/.venv/lib/python3.12/site-packages
export PYTHONPATH=$VLA_SITE:$VLA_ROOT
export HF_HOME=$PROJECT_ROOT/hf
export XDG_CACHE_HOME=$PROJECT_ROOT/cache
export TMPDIR=$PROJECT_ROOT/tmp
export WANDB_ENTITY=dsheth_caltech
export WANDB_PROJECT=vla_foundry_recap
cd $VLA_ROOT
```

Download the LBM task data:

```bash
python vla_foundry/data/scripts/download_dataset.py \
  --task BimanualPutSpatulaOnPlateFromDryingRack \
  --local_path $PROJECT_ROOT/data/lbm_preprocessed
```

Train the SFT policy:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 vla_foundry/main.py \
  --hparams.torchcompile False \
  --config_path vla_foundry/config_presets/training_jobs/vla_diffusion_spatula_plate_drying_rack_qwen3vl_2b.yaml \
  --total_train_samples 12000 \
  --num_checkpoints 3 \
  --max_checkpoint_limit 1 \
  --save_path $PROJECT_ROOT/outputs/qwen_spatula_plate_drying_rack_ft_12k_lr2e5 \
  --wandb True \
  --wandb_entity $WANDB_ENTITY \
  --wandb_project_name $WANDB_PROJECT
```

Evaluate the SFT checkpoint while collecting RECAP trajectories:

```bash
TASK_NAME=BimanualPutSpatulaOnPlateFromDryingRack \
DEMO_INDICES=150:350 \
MODEL_REPO=$PROJECT_ROOT/outputs/qwen_spatula_plate_drying_rack_ft_12k_lr2e5/<run-dir> \
SAVE_DIR=$PROJECT_ROOT/outputs/recap_sft_12k_ckpt3_150_350 \
LOG_DIR=$PROJECT_ROOT/outputs/logs \
RECAP_TRAJECTORY_DIR=$PROJECT_ROOT/outputs/recap_trajectories/sft_12k_ckpt3_150_350 \
RECAP_SAVE_IMAGES=1 \
RECAP_IMAGE_EVERY_N=1 \
POLICY_READY_TIMEOUT=1200 \
TIMEOUT_SECONDS=28800 \
examples/evaluation/runpod_qwen_bellpepper_ft_eval.sh
```

Join eval results with collected trajectories:

```bash
python -m vla_foundry.recap.label_trajectories \
  --trajectory_dir $PROJECT_ROOT/outputs/recap_trajectories/sft_12k_ckpt3_150_350 \
  --results_json $PROJECT_ROOT/outputs/recap_sft_12k_ckpt3_150_350/<timestamp>/results.json \
  --output_dir $PROJECT_ROOT/outputs/recap_labeled/sft_12k_ckpt3_150_350
```

Build value targets:

```bash
python -m vla_foundry.recap.build_value_targets \
  --labeled_dir $PROJECT_ROOT/outputs/recap_labeled/sft_12k_ckpt3_150_350 \
  --output_dir $PROJECT_ROOT/outputs/recap_value_targets/sft_12k_ckpt3_150_350_recap_cfail500_bins50 \
  --c_fail 500 \
  --num_bins 50 \
  --normalization_mode recap \
  --normalization_horizon 299 \
  --preferred_camera scene_right_0
```

Train the Qwen LoRA value model:

```bash
python -m vla_foundry.recap.train_value_qwen \
  --train_jsonl $PROJECT_ROOT/outputs/recap_value_targets/sft_12k_ckpt3_150_350_recap_cfail500_bins50/value_train.jsonl \
  --val_jsonl $PROJECT_ROOT/outputs/recap_value_targets/sft_12k_ckpt3_150_350_recap_cfail500_bins50/value_val.jsonl \
  --output_dir $PROJECT_ROOT/outputs/recap_value_runs/qwen_lora_value_bins50 \
  --model_id Qwen/Qwen3-VL-2B-Thinking \
  --num_bins 50 \
  --batch_size 6 \
  --epochs 3 \
  --lr 2e-5 \
  --value_loss_weight 10 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_last_n_layers 4 \
  --wandb \
  --wandb_entity $WANDB_ENTITY \
  --wandb_project $WANDB_PROJECT \
  --run_name recap_qwen_lora_value_bins50
```

Score all value examples with the value model:

```bash
python -m vla_foundry.recap.score_qwen_value_targets \
  --input_jsonl $PROJECT_ROOT/outputs/recap_value_targets/sft_12k_ckpt3_150_350_recap_cfail500_bins50/value_examples.jsonl \
  --checkpoint $PROJECT_ROOT/outputs/recap_value_runs/qwen_lora_value_bins50/qwen_value_checkpoint.pt \
  --output_dir $PROJECT_ROOT/outputs/recap_value_scores/qwen_lora_bins50_all \
  --camera scene_right_0 \
  --batch_size 64 \
  --weight_mode exp \
  --beta 4 \
  --max_weight 5
```

Run the RECAP policy update:

```bash
python -m vla_foundry.recap.train_recap_policy \
  --scored_jsonl $PROJECT_ROOT/outputs/recap_value_scores/qwen_lora_bins50_all/scored_steps.jsonl \
  --checkpoint_directory $PROJECT_ROOT/outputs/qwen_spatula_plate_drying_rack_ft_12k_lr2e5/<run-dir> \
  --checkpoint_path $PROJECT_ROOT/outputs/qwen_spatula_plate_drying_rack_ft_12k_lr2e5/<run-dir>/checkpoints/checkpoint_3.pt \
  --output_dir $PROJECT_ROOT/outputs/recap_policy_sft12k_ckpt3_qwen_value_epoch2_posadv_lr2e7 \
  --min_positive_advantage 1e-6 \
  --max_train_steps 500 \
  --batch_size 1 \
  --lr 2e-7 \
  --wandb \
  --wandb_entity $WANDB_ENTITY \
  --wandb_project $WANDB_PROJECT \
  --run_name recap_policy_sft12k_ckpt3_lr2e7
```

Evaluate SFT and RECAP on the matched smoke IDs:

```bash
TASK_NAME=BimanualPutSpatulaOnPlateFromDryingRack \
DEMO_INDICES=100:110 \
MODEL_REPO=dhruvmsheth/vla-foundry-spatula-drying-rack-sft12k-ckpt3 \
SAVE_DIR=$PROJECT_ROOT/outputs/eval_sft12k_100_109 \
LOG_DIR=$PROJECT_ROOT/outputs/logs \
TIMEOUT_SECONDS=14400 \
examples/evaluation/runpod_qwen_bellpepper_ft_eval.sh

TASK_NAME=BimanualPutSpatulaOnPlateFromDryingRack \
DEMO_INDICES=100:110 \
MODEL_REPO=dhruvmsheth/vla-foundry-spatula-drying-rack-recap-lr2e7-500 \
SAVE_DIR=$PROJECT_ROOT/outputs/eval_recap_lr2e7_500_100_109 \
LOG_DIR=$PROJECT_ROOT/outputs/logs \
TIMEOUT_SECONDS=14400 \
examples/evaluation/runpod_qwen_bellpepper_ft_eval.sh
```

## Caveats

This is a smoke result. The 10-scenario comparison is deliberately small and
should be treated as a proof of the RECAP engineering path, not as a statistically
stable policy benchmark. The next step is a larger matched eval over 50-100
held-out scenario IDs and a value model trained on more rollout diversity.
