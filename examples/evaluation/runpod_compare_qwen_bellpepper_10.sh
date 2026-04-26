#!/usr/bin/env bash
set -euo pipefail

# Compare out-of-the-box Qwen3VLA against the BellPepper SFT checkpoint on the
# same LBM Eval scenarios. Run this inside the RunPod lbm_eval image.

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/vla_recap}"
VLA_ROOT="${VLA_ROOT:-${PROJECT_ROOT}/src/vla_foundry}"
EVAL_HELPER="${EVAL_HELPER:-${VLA_ROOT}/examples/evaluation/runpod_qwen_bellpepper_ft_eval.sh}"

TASK_NAME="${TASK_NAME:-BimanualPutRedBellPepperInBin}"
DEMO_INDICES="${DEMO_INDICES:-0:10}"
NUM_FLOW_STEPS="${NUM_FLOW_STEPS:-8}"
OPEN_LOOP_STEPS="${OPEN_LOOP_STEPS:-8}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-14400}"

OOTB_REPO="${OOTB_REPO:-TRI-ML/Foundry-Qwen3VLA-2B}"
SFT_REPO="${SFT_REPO:-dhruvmsheth/vla-foundry-qwen-bellpepper-ft-4k-v2}"

if [[ ! -x "${EVAL_HELPER}" ]]; then
  echo "ERROR: eval helper not found or not executable: ${EVAL_HELPER}" >&2
  exit 1
fi

run_eval() {
  local label="$1"
  local repo="$2"
  local save_dir="${PROJECT_ROOT}/outputs/compare_qwen_bellpepper_10/${label}"

  echo "============================================================"
  echo "Running ${label}: ${repo}"
  echo "Task: ${TASK_NAME}"
  echo "Scenarios: ${DEMO_INDICES}"
  echo "Save dir: ${save_dir}"
  echo "============================================================"

  MODEL_REPO="${repo}" \
  TASK_NAME="${TASK_NAME}" \
  DEMO_INDICES="${DEMO_INDICES}" \
  NUM_FLOW_STEPS="${NUM_FLOW_STEPS}" \
  OPEN_LOOP_STEPS="${OPEN_LOOP_STEPS}" \
  TIMEOUT_SECONDS="${TIMEOUT_SECONDS}" \
  SAVE_DIR="${save_dir}" \
  JOB_NAME="compare_${label}_$(date +%Y%m%d_%H%M%S)" \
  "${EVAL_HELPER}"
}

run_eval "ootb" "${OOTB_REPO}"
run_eval "sft_4k" "${SFT_REPO}"

python3 - <<'PY'
from pathlib import Path
import json

root = Path("/workspace/vla_recap/outputs/compare_qwen_bellpepper_10")


def latest_results(label):
    files = sorted((root / label).glob("*/results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No results.json found for {label}")
    path = files[0]
    return path, json.loads(path.read_text())


rows = []
for label, display in [("ootb", "OOTB"), ("sft_4k", "SFT-4k")]:
    path, result = latest_results(label)
    rows.append(
        {
            "policy": display,
            "episodes": result["num_evaluated"],
            "successes": result["num_success"],
            "success_rate": result["success_rate"],
            "path": str(path),
        }
    )

print("\nComparison")
print("Policy   Episodes  Successes  Success Rate")
print("-------------------------------------------")
for row in rows:
    print(f"{row['policy']:<8} {row['episodes']:>8} {row['successes']:>10} {row['success_rate']:>12.3f}")

print("\nResult files")
for row in rows:
    print(f"{row['policy']}: {row['path']}")
PY
