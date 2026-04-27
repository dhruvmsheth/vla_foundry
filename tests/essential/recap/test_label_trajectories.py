import json
from pathlib import Path

from vla_foundry.recap.label_trajectories import label_trajectories


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_label_trajectories_by_eval_order(tmp_path: Path) -> None:
    trajectory_dir = tmp_path / "trajectories"
    episodes = [
        {
            "episode_id": "client-a_episode_000000",
            "episode_dir": "episodes/client-a_episode_000000",
            "reset_seed": 100000200,
        },
        {
            "episode_id": "client-b_episode_000000",
            "episode_dir": "episodes/client-b_episode_000000",
            "reset_seed": 100000201,
        },
    ]
    write_jsonl(trajectory_dir / "episodes.jsonl", episodes)
    write_jsonl(
        trajectory_dir / "episodes/client-a_episode_000000/trajectory.jsonl",
        [{"episode_id": "client-a_episode_000000", "step_index": 0, "action_fields": {"a": 1}}],
    )
    write_jsonl(
        trajectory_dir / "episodes/client-b_episode_000000/trajectory.jsonl",
        [
            {"episode_id": "client-b_episode_000000", "step_index": 0, "action_fields": {"a": 2}},
            {"episode_id": "client-b_episode_000000", "step_index": 1, "action_fields": {"a": 3}},
        ],
    )
    results_json = tmp_path / "results.json"
    results_json.write_text(
        json.dumps(
            {
                "evaluations": [
                    {"scenario_index": 100, "skill_type": "task", "is_success": True, "total_time": 1.0},
                    {"scenario_index": 101, "skill_type": "task", "is_success": False, "total_time": 2.0},
                ]
            }
        )
    )

    summary = label_trajectories(trajectory_dir, results_json, tmp_path / "labeled")

    assert summary.num_labeled_episodes == 2
    assert summary.num_labeled_steps == 3
    assert summary.num_success == 1
    assert summary.success_rate == 0.5
    assert summary.join_strategy == "order+constant_seed_offset:100000100"

    labeled_episodes = [
        json.loads(line) for line in (tmp_path / "labeled" / "labeled_episodes.jsonl").read_text().splitlines()
    ]
    assert labeled_episodes[0]["scenario_index"] == 100
    assert labeled_episodes[0]["terminal_reward"] == 1.0
    assert labeled_episodes[1]["scenario_index"] == 101
    assert labeled_episodes[1]["terminal_reward"] == 0.0

    labeled_steps = [
        json.loads(line) for line in (tmp_path / "labeled" / "labeled_steps.jsonl").read_text().splitlines()
    ]
    assert [step["terminal_reward"] for step in labeled_steps] == [1.0, 0.0, 0.0]
