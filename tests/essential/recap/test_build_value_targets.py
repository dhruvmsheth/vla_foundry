import json
from pathlib import Path

from vla_foundry.recap.build_value_targets import build_value_targets


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_build_value_targets_success_and_failure_returns(tmp_path: Path) -> None:
    labeled_dir = tmp_path / "labeled"
    trajectory_dir = tmp_path / "trajectories"
    labeled_dir.mkdir()
    (labeled_dir / "label_summary.json").write_text(json.dumps({"trajectory_dir": str(trajectory_dir)}))
    write_jsonl(
        labeled_dir / "labeled_steps.jsonl",
        [
            {
                "episode_id": "success",
                "step_index": 0,
                "terminal_is_success": True,
                "image_paths": {"scene_right_0": "episodes/success/0.jpg"},
            },
            {
                "episode_id": "success",
                "step_index": 1,
                "terminal_is_success": True,
                "image_paths": {"scene_right_0": "episodes/success/1.jpg"},
            },
            {
                "episode_id": "failure",
                "step_index": 0,
                "terminal_is_success": False,
                "image_paths": {"scene_right_0": "episodes/failure/0.jpg"},
            },
        ],
    )

    summary = build_value_targets(labeled_dir, tmp_path / "targets", c_fail=10.0, num_bins=11, val_fraction=0.0)
    examples = [json.loads(line) for line in (tmp_path / "targets" / "value_examples.jsonl").read_text().splitlines()]

    assert summary["num_examples"] == 3
    by_episode_step = {(example["episode_id"], example["step_index"]): example for example in examples}
    success_first = by_episode_step[("success", 0)]
    success_terminal = by_episode_step[("success", 1)]
    failure = by_episode_step[("failure", 0)]
    assert success_first["raw_return"] == -1.0
    assert success_first["normalized_value"] == -0.1
    assert success_first["value_bin"] == 9
    assert success_terminal["raw_return"] == -0.0
    assert success_terminal["value_bin"] == 10
    assert failure["raw_return"] == -10.0
    assert failure["normalized_value"] == -1.0
    assert failure["value_bin"] == 0
