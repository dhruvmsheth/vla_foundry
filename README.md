# RECAP Post-Training in VLA Foundry

This fork turns VLA Foundry into an end-to-end RECAP post-training sandbox for
robotics policies in TRI LBM Eval.

The current experiment is intentionally narrow:

- **Task:** `BimanualPutSpatulaOnPlateFromDryingRack`
- **Base policy:** Foundry Qwen3VLA 2B, SFT on task demonstrations
- **Post-training:** RECAP-style value-weighted policy update from rollout data
- **Simulator:** TRI LBM Eval via the `toyotaresearch/lbm-eval-oss:vla-foundry` image

The upstream VLA Foundry README is preserved at [`README_UPSTREAM.md`](README_UPSTREAM.md).

## Current Result

Matched eval IDs: `100-109`

| Policy | Success |
|---|---:|
| SFT12k checkpoint 3 | 5/10, 50% |
| RECAP policy, lr `2e-7`, 500 steps | 6/10, 60% |

![SFT vs RECAP](docs/reports/recap_spatula_drying_rack/assets/sft_vs_recap_spatula_drying_rack_100_109.png)

This is a small smoke result, not a statistically stable benchmark. It is useful
because it proves the whole loop works: rollout collection, value labeling, value
model training, advantage scoring, RECAP policy update, and matched re-eval.

## What Was Built

RECAP utilities live in [`vla_foundry/recap`](vla_foundry/recap):

| File | Purpose |
|---|---|
| `label_trajectories.py` | Join LBM eval `results.json` with collected policy rollouts. |
| `build_value_targets.py` | Convert terminal success/failure into per-step normalized return targets. |
| `train_value_qwen.py` | Train a Qwen3-VL LoRA value model from camera frames and robot state. |
| `score_qwen_value_targets.py` | Score rollout steps with the value model and compute RECAP weights. |
| `train_recap_policy.py` | Update the policy on positive-advantage weighted rollout samples. |
| `plot_qwen_value_rollouts.py` | Render value prediction plots/videos over saved trajectories. |

Evaluation helpers live in [`examples/evaluation`](examples/evaluation):

| File | Purpose |
|---|---|
| `runpod_qwen_bellpepper_ft_eval.sh` | Run a Qwen policy in LBM Eval, with optional RECAP trajectory collection. |
| `runpod_compare_qwen_bellpepper_10.sh` | Compare two policy repos on the same LBM scenario IDs. |

## Artifact Locations

Large files are stored on Hugging Face, not Git.

| Artifact | Hugging Face repo |
|---|---|
| SFT policy checkpoint | `dhruvmsheth/vla-foundry-spatula-drying-rack-sft12k-ckpt3` |
| RECAP policy checkpoint | `dhruvmsheth/vla-foundry-spatula-drying-rack-recap-lr2e7-500` |
| Qwen LoRA value model | `dhruvmsheth/vla-foundry-spatula-drying-rack-qwen-value-lora` |
| Results, value targets, scores, rollout videos | `dhruvmsheth/vla-foundry-recap-spatula-drying-rack-artifacts` |

W&B:

- entity: `dsheth_caltech`
- project: `vla_foundry_recap`

Detailed report:

- [`docs/reports/recap_spatula_drying_rack/README.md`](docs/reports/recap_spatula_drying_rack/README.md)
- [`docs/reports/recap_spatula_drying_rack/summary.json`](docs/reports/recap_spatula_drying_rack/summary.json)

## Minimal RunPod Setup

Use the LBM Eval image:

```text
toyotaresearch/lbm-eval-oss:vla-foundry
```

Container start command:

```bash
/bin/bash -lc "sleep infinity"
```

Environment:

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

Login:

```bash
hf auth login
wandb login
```

## Reproduce the Smoke Eval

Evaluate the SFT policy:

```bash
TASK_NAME=BimanualPutSpatulaOnPlateFromDryingRack \
DEMO_INDICES=100:110 \
MODEL_REPO=dhruvmsheth/vla-foundry-spatula-drying-rack-sft12k-ckpt3 \
SAVE_DIR=$PROJECT_ROOT/outputs/eval_sft12k_100_109 \
LOG_DIR=$PROJECT_ROOT/outputs/logs \
TIMEOUT_SECONDS=14400 \
examples/evaluation/runpod_qwen_bellpepper_ft_eval.sh
```

Evaluate the RECAP policy:

```bash
TASK_NAME=BimanualPutSpatulaOnPlateFromDryingRack \
DEMO_INDICES=100:110 \
MODEL_REPO=dhruvmsheth/vla-foundry-spatula-drying-rack-recap-lr2e7-500 \
SAVE_DIR=$PROJECT_ROOT/outputs/eval_recap_lr2e7_500_100_109 \
LOG_DIR=$PROJECT_ROOT/outputs/logs \
TIMEOUT_SECONDS=14400 \
examples/evaluation/runpod_qwen_bellpepper_ft_eval.sh
```

Each eval writes a standard LBM Eval `results.json` with:

- `num_evaluated`
- `num_success`
- `success_rate`
- per-scenario success/failure rows

## Value Function Snapshot

The value model was trained from 200 SFT rollout episodes:

| Metric | Value |
|---|---:|
| rollout episodes | 200 |
| value examples | 42,299 |
| rollout success rate | 65.5% |
| validation MAE | 0.292 |
| success/failure prediction gap | 0.201 |
| positive-advantage fraction after scoring | 28.7% |

![Value scoring](docs/reports/recap_spatula_drying_rack/assets/recap_value_scoring_summary.png)

## Next Steps

The next experiment should move beyond smoke testing:

1. Run 50-100 matched held-out eval scenarios for SFT and RECAP.
2. Collect more rollout diversity for the value model.
3. Train a stronger value model and check success/failure calibration.
4. Sweep RECAP update LR, step count, and advantage weighting.
5. Compare against SFT-only and OOTB baselines on the same IDs.
