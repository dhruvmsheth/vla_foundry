"""Join collected RECAP trajectories with LBM eval success labels."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LabelSummary:
    trajectory_dir: str
    results_json: str
    output_dir: str
    num_episodes: int
    num_evaluations: int
    num_labeled_episodes: int
    num_labeled_steps: int
    num_success: int
    success_rate: float
    join_strategy: str


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    records = []
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
    return records


def dump_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True))
            f.write("\n")


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text())
    evaluations = data.get("evaluations")
    if not isinstance(evaluations, list):
        raise ValueError(f"{path} does not contain an 'evaluations' list")
    return evaluations


def infer_seed_offset(episodes: list[dict[str, Any]], evaluations: list[dict[str, Any]]) -> int | None:
    offsets = []
    for episode, evaluation in zip(episodes, evaluations, strict=True):
        reset_seed = episode.get("reset_seed")
        scenario_index = evaluation.get("scenario_index")
        if not isinstance(reset_seed, int) or not isinstance(scenario_index, int):
            return None
        offsets.append(reset_seed - scenario_index)
    if not offsets:
        return None
    first = offsets[0]
    return first if all(offset == first for offset in offsets) else None


def trajectory_path(trajectory_dir: Path, episode: dict[str, Any]) -> Path:
    episode_dir = episode.get("episode_dir")
    if not isinstance(episode_dir, str) or not episode_dir:
        raise ValueError(f"Episode is missing a valid episode_dir: {episode}")
    return trajectory_dir / episode_dir / "trajectory.jsonl"


def label_trajectories(
    trajectory_dir: Path,
    results_json: Path,
    output_dir: Path,
    *,
    allow_count_mismatch: bool = False,
) -> LabelSummary:
    episodes_path = trajectory_dir / "episodes.jsonl"
    episodes = load_jsonl(episodes_path)
    evaluations = load_results(results_json)

    if len(episodes) != len(evaluations) and not allow_count_mismatch:
        raise ValueError(
            "Episode/evaluation count mismatch: "
            f"{len(episodes)} episodes in {episodes_path}, {len(evaluations)} evaluations in {results_json}. "
            "Re-run with --allow_count_mismatch only if you intentionally want to truncate to the shorter length."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    count = min(len(episodes), len(evaluations))
    seed_offset = infer_seed_offset(episodes[:count], evaluations[:count])
    join_strategy = "order"
    if seed_offset is not None:
        join_strategy = f"order+constant_seed_offset:{seed_offset}"

    labeled_episodes: list[dict[str, Any]] = []
    labeled_steps: list[dict[str, Any]] = []

    for ordinal, (episode, evaluation) in enumerate(zip(episodes[:count], evaluations[:count], strict=True)):
        path = trajectory_path(trajectory_dir, episode)
        steps = load_jsonl(path)
        is_success = bool(evaluation.get("is_success"))
        terminal_reward = 1.0 if is_success else 0.0
        label = {
            "eval_ordinal": ordinal,
            "scenario_index": evaluation.get("scenario_index"),
            "skill_type": evaluation.get("skill_type"),
            "is_success": is_success,
            "terminal_reward": terminal_reward,
            "total_time": evaluation.get("total_time"),
            "failure_message": evaluation.get("failure_message"),
            "is_pending": evaluation.get("is_pending"),
        }

        labeled_episode = {
            **episode,
            **label,
            "num_steps": len(steps),
            "trajectory_path": str(path.relative_to(trajectory_dir)),
            "join_strategy": join_strategy,
        }
        labeled_episodes.append(labeled_episode)

        for step in steps:
            labeled_steps.append(
                {
                    **step,
                    "eval_ordinal": ordinal,
                    "scenario_index": evaluation.get("scenario_index"),
                    "skill_type": evaluation.get("skill_type"),
                    "terminal_is_success": is_success,
                    "terminal_reward": terminal_reward,
                    "terminal_total_time": evaluation.get("total_time"),
                    "terminal_failure_message": evaluation.get("failure_message"),
                }
            )

    labeled_episodes_path = output_dir / "labeled_episodes.jsonl"
    labeled_steps_path = output_dir / "labeled_steps.jsonl"
    dump_jsonl(labeled_episodes_path, labeled_episodes)
    dump_jsonl(labeled_steps_path, labeled_steps)

    num_success = sum(1 for evaluation in evaluations[:count] if evaluation.get("is_success"))
    success_rate = num_success / count if count else 0.0
    summary = LabelSummary(
        trajectory_dir=str(trajectory_dir),
        results_json=str(results_json),
        output_dir=str(output_dir),
        num_episodes=len(episodes),
        num_evaluations=len(evaluations),
        num_labeled_episodes=count,
        num_labeled_steps=len(labeled_steps),
        num_success=num_success,
        success_rate=success_rate,
        join_strategy=join_strategy,
    )
    (output_dir / "label_summary.json").write_text(json.dumps(summary.__dict__, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory_dir", type=Path, required=True, help="Directory containing episodes.jsonl.")
    parser.add_argument("--results_json", type=Path, required=True, help="LBM eval results.json for the same run.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <trajectory_dir>/labeled.",
    )
    parser.add_argument(
        "--allow_count_mismatch",
        action="store_true",
        help="Truncate to the shorter episode/evaluation count instead of failing on count mismatch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.trajectory_dir / "labeled"
    summary = label_trajectories(
        args.trajectory_dir,
        args.results_json,
        output_dir,
        allow_count_mismatch=args.allow_count_mismatch,
    )
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

