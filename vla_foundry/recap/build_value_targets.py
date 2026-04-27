"""Build RECAP value-function targets from labeled rollout trajectories."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from vla_foundry.recap.label_trajectories import dump_jsonl, load_jsonl


def _value_to_bin(value: float, num_bins: int) -> int:
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")
    clipped = min(0.0, max(-1.0, value))
    return int(round((clipped + 1.0) * (num_bins - 1)))


def _build_absolute_image_paths(step: dict[str, Any], trajectory_dir: Path) -> dict[str, str]:
    image_paths = step.get("image_paths") or {}
    if not isinstance(image_paths, dict):
        return {}
    return {str(camera): str((trajectory_dir / str(path)).resolve()) for camera, path in image_paths.items()}


def _select_image_path(image_paths: dict[str, str], preferred_camera: str | None) -> str | None:
    if preferred_camera and preferred_camera in image_paths:
        return image_paths[preferred_camera]
    for camera in sorted(image_paths):
        return image_paths[camera]
    return None


def build_value_targets(
    labeled_dir: Path,
    output_dir: Path,
    *,
    c_fail: float = 500.0,
    num_bins: int = 201,
    val_fraction: float = 0.2,
    seed: int = 42,
    preferred_camera: str | None = "scene_right_0",
) -> dict[str, Any]:
    if c_fail <= 0:
        raise ValueError("c_fail must be positive")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")

    summary_path = labeled_dir / "label_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    label_summary = json.loads(summary_path.read_text())
    trajectory_dir = Path(label_summary["trajectory_dir"])
    steps = load_jsonl(labeled_dir / "labeled_steps.jsonl")

    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step in steps:
        episode_id = step.get("episode_id")
        if not isinstance(episode_id, str):
            raise ValueError(f"Step is missing episode_id: {step}")
        by_episode[episode_id].append(step)

    examples: list[dict[str, Any]] = []
    for _episode_id, episode_steps in sorted(by_episode.items()):
        episode_steps.sort(key=lambda record: int(record.get("step_index", 0)))
        num_steps = len(episode_steps)
        if num_steps == 0:
            continue

        is_success = bool(episode_steps[-1].get("terminal_is_success"))
        for ordinal, step in enumerate(episode_steps):
            steps_to_terminal = num_steps - 1 - ordinal
            raw_return = -float(steps_to_terminal)
            if not is_success:
                raw_return -= c_fail
            clipped_return = max(-c_fail, min(0.0, raw_return))
            normalized_value = clipped_return / c_fail
            image_paths_absolute = _build_absolute_image_paths(step, trajectory_dir)
            selected_image_path = _select_image_path(image_paths_absolute, preferred_camera)
            examples.append(
                {
                    **step,
                    "episode_num_steps": num_steps,
                    "episode_step_ordinal": ordinal,
                    "steps_to_terminal": steps_to_terminal,
                    "raw_return": raw_return,
                    "clipped_return": clipped_return,
                    "normalized_value": normalized_value,
                    "value_bin": _value_to_bin(normalized_value, num_bins),
                    "num_value_bins": num_bins,
                    "c_fail": c_fail,
                    "trajectory_dir": str(trajectory_dir),
                    "image_paths_absolute": image_paths_absolute,
                    "selected_image_path": selected_image_path,
                    "preferred_camera": preferred_camera,
                }
            )

    episode_ids = sorted(by_episode)
    rng = random.Random(seed)
    rng.shuffle(episode_ids)
    num_val = int(round(len(episode_ids) * val_fraction))
    if val_fraction > 0 and len(episode_ids) > 1:
        num_val = max(1, min(len(episode_ids) - 1, num_val))
    val_episode_ids = set(episode_ids[:num_val])

    train_examples = [example for example in examples if example["episode_id"] not in val_episode_ids]
    val_examples = [example for example in examples if example["episode_id"] in val_episode_ids]

    output_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(output_dir / "value_examples.jsonl", examples)
    dump_jsonl(output_dir / "value_train.jsonl", train_examples)
    dump_jsonl(output_dir / "value_val.jsonl", val_examples)

    num_success_episodes = sum(
        1
        for episode_steps in by_episode.values()
        if episode_steps and bool(episode_steps[-1].get("terminal_is_success"))
    )
    target_summary = {
        "labeled_dir": str(labeled_dir),
        "trajectory_dir": str(trajectory_dir),
        "output_dir": str(output_dir),
        "c_fail": c_fail,
        "num_value_bins": num_bins,
        "preferred_camera": preferred_camera,
        "num_episodes": len(by_episode),
        "num_success_episodes": num_success_episodes,
        "success_rate": num_success_episodes / len(by_episode) if by_episode else 0.0,
        "num_examples": len(examples),
        "num_train_examples": len(train_examples),
        "num_val_examples": len(val_examples),
        "num_val_episodes": len(val_episode_ids),
        "seed": seed,
    }
    (output_dir / "value_target_summary.json").write_text(json.dumps(target_summary, indent=2, sort_keys=True) + "\n")
    return target_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled_dir", type=Path, required=True, help="Directory from label_trajectories.py.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for value target JSONL files.")
    parser.add_argument("--c_fail", type=float, default=500.0, help="Terminal failure penalty.")
    parser.add_argument("--num_bins", type=int, default=201, help="Number of discretized value bins.")
    parser.add_argument("--val_fraction", type=float, default=0.2, help="Episode-level validation split fraction.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preferred_camera", type=str, default="scene_right_0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_value_targets(
        args.labeled_dir,
        args.output_dir,
        c_fail=args.c_fail,
        num_bins=args.num_bins,
        val_fraction=args.val_fraction,
        seed=args.seed,
        preferred_camera=args.preferred_camera,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
