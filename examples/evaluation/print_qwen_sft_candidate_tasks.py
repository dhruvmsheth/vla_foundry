#!/usr/bin/env python3
"""Print LBM tasks where packaged Qwen SFT improves over packaged Qwen OOTB."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "vla_foundry" / "eval" / "eval_results" / "OSS"
OOTB = "Foundry-Qwen3VLA-2.1B"
SFT = "Foundry-Qwen3VLA-2.1B-FT"


def success_rate(path: Path) -> tuple[float, int, int]:
    result = json.loads(path.read_text())
    evaluations = [e for e in result.get("evaluations", []) if not e.get("is_pending")]
    num_evaluated = len(evaluations)
    num_success = sum(1 for e in evaluations if e.get("is_success") is True)
    return (num_success / num_evaluated if num_evaluated else 0.0, num_success, num_evaluated)


def task_results(model: str) -> dict[str, tuple[float, int, int]]:
    rows = {}
    for path in (ROOT / model).glob("*/rollouts/*/results.json"):
        task = path.parts[-4]
        rows[task] = success_rate(path)
    return rows


def main() -> None:
    ootb = task_results(OOTB)
    sft = task_results(SFT)
    rows = []
    for task in sorted(set(ootb) & set(sft)):
        ootb_sr, ootb_s, ootb_n = ootb[task]
        sft_sr, sft_s, sft_n = sft[task]
        rows.append((sft_sr - ootb_sr, ootb_sr, sft_sr, ootb_s, ootb_n, sft_s, sft_n, task))

    print("Task                                           OOTB        SFT       Delta")
    print("----------------------------------------------------------------------------")
    for delta, ootb_sr, sft_sr, ootb_s, ootb_n, sft_s, sft_n, task in sorted(rows, reverse=True):
        print(
            f"{task:<45} "
            f"{ootb_s:>3}/{ootb_n:<3} {ootb_sr:>6.1%}  "
            f"{sft_s:>3}/{sft_n:<3} {sft_sr:>6.1%}  "
            f"{delta:>+6.1%}"
        )

    print("\nBest first RECAP candidates are usually not the largest delta alone.")
    print("Prefer tasks where the post-SFT policy has both successes and failures, roughly 30-80%.")


if __name__ == "__main__":
    main()
