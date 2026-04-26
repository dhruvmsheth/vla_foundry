#!/usr/bin/env bash
set -euo pipefail

# Run the Qwen BellPepper finetune through the RunPod lbm_eval wrapper.
#
# This script intentionally handles the fragile parts of the RunPod image:
# - keeps HF_HOME on /workspace and copies the login token there if needed
# - restores small dataset metadata files expected by the saved training config
# - avoids run_inference_bundle.sh pipefail exits by using /opt/anzu as workdir
# - uses a "python <script>" command form required by the wrapper parser
# - patches lbm_eval/policy_interfaces PolicyMetadata.save_json at runtime

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/vla_recap}"
VLA_ROOT="${VLA_ROOT:-${PROJECT_ROOT}/src/vla_foundry}"
VLA_SITE="${VLA_SITE:-${VLA_ROOT}/.venv/lib/python3.12/site-packages}"
HF_HOME="${HF_HOME:-${PROJECT_ROOT}/hf}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/cache}"
TMPDIR="${TMPDIR:-${PROJECT_ROOT}/tmp}"

MODEL_REPO="${MODEL_REPO:-dhruvmsheth/vla-foundry-qwen-bellpepper-ft-4k-v2}"
TASK_NAME="${TASK_NAME:-BimanualPutRedBellPepperInBin}"
DEMO_INDICES="${DEMO_INDICES:-0:1}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
NUM_FLOW_STEPS="${NUM_FLOW_STEPS:-8}"
OPEN_LOOP_STEPS="${OPEN_LOOP_STEPS:-8}"
POLICY_READY_TIMEOUT="${POLICY_READY_TIMEOUT:-1200}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/outputs/qwen_foundry_ft_4k_v2_eval}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/outputs/logs}"
LBM_WRAPPER="${LBM_WRAPPER:-/opt/anzu/run_inference_bundle.sh}"

mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}" "${TMPDIR}" "${SAVE_DIR}" "${LOG_DIR}"

if [[ ! -f "${HF_HOME}/token" && -f "${HOME}/.cache/huggingface/token" ]]; then
  cp "${HOME}/.cache/huggingface/token" "${HF_HOME}/token"
fi

if [[ ! -x "${LBM_WRAPPER}" ]]; then
  echo "ERROR: lbm_eval wrapper not found: ${LBM_WRAPPER}" >&2
  exit 1
fi

METADATA_DIR="${PROJECT_ROOT}/data/lbm_preprocessed"
mkdir -p "${METADATA_DIR}"

HF_HOME="${HF_HOME}" PYTHONPATH="${VLA_SITE}:${VLA_ROOT}" /usr/bin/python3.12 - <<PY
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download

repo = "${MODEL_REPO}"
out = Path("${METADATA_DIR}")
out.mkdir(parents=True, exist_ok=True)

for name in ["preprocessing_config.yaml", "stats.json", "processing_metadata.json"]:
    dst = out / name
    if not dst.exists():
        src = hf_hub_download(repo_id=repo, filename=name)
        shutil.copyfile(src, dst)
        print(f"copied {name} -> {dst}")

manifest = out / "manifest.jsonl"
if not manifest.exists():
    manifest.write_text("")
    print(f"created {manifest}")
PY

EVAL_COMPAT_DIR="${PROJECT_ROOT}/eval_compat"
mkdir -p "${EVAL_COMPAT_DIR}"
cat > "${EVAL_COMPAT_DIR}/sitecustomize.py" <<'PY'
import dataclasses
import json
import os


def _save_policy_metadata_json(self, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "policy_metadata.json")
    if dataclasses.is_dataclass(self):
        payload = dataclasses.asdict(self)
    else:
        payload = dict(getattr(self, "__dict__", {}))
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


for module_name in ("policy_interfaces.robot_gym.policy", "robot_gym.policy"):
    try:
        module = __import__(module_name, fromlist=["PolicyMetadata"])
        cls = getattr(module, "PolicyMetadata")
        if not hasattr(cls, "save_json"):
            cls.save_json = _save_policy_metadata_json
    except Exception:
        pass
PY

PYTHON_BIN_DIR="${TMPDIR}/bin"
mkdir -p "${PYTHON_BIN_DIR}"
ln -sf /usr/bin/python3.12 "${PYTHON_BIN_DIR}/python"

POLICY_PY="${VLA_ROOT}/vla_foundry/inference/robotics/inference_policy.py"
if [[ ! -f "${POLICY_PY}" ]]; then
  echo "ERROR: policy script not found: ${POLICY_PY}" >&2
  exit 1
fi

JOB_NAME="${JOB_NAME:-qwen_foundry_ft_4k_v2_eval_$(date +%Y%m%d_%H%M%S)}"

unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
unset HF_DATASETS_OFFLINE

PYTHONPATH="${EVAL_COMPAT_DIR}" \
JOB_NAME="${JOB_NAME}" \
LOG_DIR="${LOG_DIR}" \
LAUNCH_SAVE_DIR="${SAVE_DIR}" \
LAUNCH_SUMMARY_DIR="${SAVE_DIR}" \
TASK_NAME="${TASK_NAME}" \
LAUNCH_DEMONSTRATION_INDICES="${DEMO_INDICES}" \
NUM_PROCESSES="${NUM_PROCESSES}" \
MAX_RETRIES=0 \
POLICY_READY_TIMEOUT="${POLICY_READY_TIMEOUT}" \
INFERENCE_WORKDIR=/opt/anzu \
INFERENCE_CMD_OVERRIDE="PATH=${PYTHON_BIN_DIR}:\$PATH HF_HOME=${HF_HOME} XDG_CACHE_HOME=${XDG_CACHE_HOME} TMPDIR=${TMPDIR} HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 PYTHONPATH=${EVAL_COMPAT_DIR}:${VLA_SITE}:${VLA_ROOT} python ${POLICY_PY} --checkpoint_directory hf://${MODEL_REPO} --device cuda --num_flow_steps ${NUM_FLOW_STEPS} --open_loop_steps ${OPEN_LOOP_STEPS}" \
timeout "${TIMEOUT_SECONDS}" bash "${LBM_WRAPPER}"

latest_result="$(find "${SAVE_DIR}" -name results.json -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
if [[ -n "${latest_result}" ]]; then
  echo "Latest result: ${latest_result}"
  python3 -m json.tool "${latest_result}"
fi
