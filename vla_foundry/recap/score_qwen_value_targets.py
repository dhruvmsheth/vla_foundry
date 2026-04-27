"""Score RECAP rollout steps with a trained Qwen value function.

The output JSONL preserves each input record and adds value-model predictions,
Monte-Carlo advantage estimates, and non-negative RECAP weights for downstream
policy posttraining.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from vla_foundry.recap.label_trajectories import dump_jsonl, load_jsonl
from vla_foundry.recap.plot_qwen_value_rollouts import EpisodeValueDataset, load_qwen_value_model
from vla_foundry.recap.train_value import expected_value_from_logits, make_autocast
from vla_foundry.recap.train_value_qwen import QwenValueCollator, move_batch


def compute_weight(
    positive_advantage: float,
    *,
    weight_mode: str,
    beta: float,
    min_weight: float,
    max_weight: float,
) -> float:
    if weight_mode == "linear":
        weight = 1.0 + beta * positive_advantage
    elif weight_mode == "exp":
        weight = math.exp(beta * positive_advantage)
    elif weight_mode == "binary":
        weight = 1.0 if positive_advantage > 0.0 else 0.0
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")
    return float(min(max_weight, max(min_weight, weight)))


@torch.no_grad()
def score_records(
    records: list[dict[str, Any]],
    *,
    checkpoint: Path,
    camera: str,
    image_size: int,
    batch_size: int,
    device_name: str,
    precision: str,
    weight_mode: str,
    beta: float,
    min_weight: float,
    max_weight: float,
) -> list[dict[str, Any]]:
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, config = load_qwen_value_model(checkpoint, device, precision)
    processor = AutoProcessor.from_pretrained(config["model_id"])
    dataset = EpisodeValueDataset(
        records,
        image_size=image_size,
        state_dim=int(config["state_dim"]),
        camera=camera,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=QwenValueCollator(processor))

    predicted_values: list[float] = []
    for batch in loader:
        batch = move_batch(batch, device)
        batch.pop("target_bin")
        batch.pop("target_value")
        batch.pop("terminal_is_success")
        with make_autocast(device, precision):
            logits = model(batch)
            values = expected_value_from_logits(logits.float())
        predicted_values.extend(float(value) for value in values.detach().cpu().tolist())

    scored = []
    for record, predicted_value in zip(records, predicted_values, strict=True):
        realized_return = float(record["normalized_value"])
        advantage = realized_return - predicted_value
        positive_advantage = max(0.0, advantage)
        scored.append(
            {
                **record,
                "value_prediction_camera": camera,
                "value_model_checkpoint": str(checkpoint),
                "predicted_value": predicted_value,
                "realized_return": realized_return,
                "recap_advantage": advantage,
                "recap_positive_advantage": positive_advantage,
                "recap_weight": compute_weight(
                    positive_advantage,
                    weight_mode=weight_mode,
                    beta=beta,
                    min_weight=min_weight,
                    max_weight=max_weight,
                ),
            }
        )
    return scored


def summarize(scored: list[dict[str, Any]]) -> dict[str, Any]:
    if not scored:
        return {"num_examples": 0}
    weights = np.asarray([float(record["recap_weight"]) for record in scored], dtype=np.float64)
    advantages = np.asarray([float(record["recap_advantage"]) for record in scored], dtype=np.float64)
    predicted = np.asarray([float(record["predicted_value"]) for record in scored], dtype=np.float64)
    realized = np.asarray([float(record["realized_return"]) for record in scored], dtype=np.float64)
    success_mask = np.asarray([bool(record.get("terminal_is_success")) for record in scored], dtype=bool)
    positive_mask = advantages > 0
    return {
        "num_examples": len(scored),
        "num_positive_advantage": int(positive_mask.sum()),
        "positive_advantage_fraction": float(positive_mask.mean()),
        "mean_advantage": float(advantages.mean()),
        "mean_positive_advantage": float(np.maximum(advantages, 0).mean()),
        "mean_weight": float(weights.mean()),
        "max_weight": float(weights.max()),
        "mean_predicted_value": float(predicted.mean()),
        "mean_realized_return": float(realized.mean()),
        "mean_predicted_success": float(predicted[success_mask].mean()) if success_mask.any() else math.nan,
        "mean_predicted_failure": float(predicted[~success_mask].mean()) if (~success_mask).any() else math.nan,
        "mean_weight_success": float(weights[success_mask].mean()) if success_mask.any() else math.nan,
        "mean_weight_failure": float(weights[~success_mask].mean()) if (~success_mask).any() else math.nan,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--camera", default="scene_right_0")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--weight_mode", choices=["linear", "exp", "binary"], default="exp")
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--min_weight", type=float, default=0.0)
    parser.add_argument("--max_weight", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input_jsonl)
    scored = score_records(
        records,
        checkpoint=args.checkpoint,
        camera=args.camera,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device_name=args.device,
        precision=args.precision,
        weight_mode=args.weight_mode,
        beta=args.beta,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(args.output_dir / "scored_steps.jsonl", scored)
    summary = {
        "input_jsonl": str(args.input_jsonl),
        "checkpoint": str(args.checkpoint),
        "output_dir": str(args.output_dir),
        "camera": args.camera,
        "weight_mode": args.weight_mode,
        "beta": args.beta,
        "min_weight": args.min_weight,
        "max_weight": args.max_weight,
        **summarize(scored),
    }
    (args.output_dir / "score_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
