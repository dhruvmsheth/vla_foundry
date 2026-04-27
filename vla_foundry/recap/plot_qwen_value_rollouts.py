"""Render RECAP value predictions over rollout camera frames.

This script loads a Qwen-LoRA RECAP value checkpoint, scores selected
rollout trajectories, and writes LeRobot-style static summaries plus MP4s
with camera frames above predicted/labeled normalized returns.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from vla_foundry.recap.label_trajectories import load_jsonl
from vla_foundry.recap.train_value import expected_value_from_logits, make_autocast, vector_from_fields
from vla_foundry.recap.train_value_qwen import QwenValueCollator, QwenValueModel, load_pil, move_batch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def group_records_by_episode(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        episode_id = str(record.get("episode_id") or "")
        if episode_id:
            grouped[episode_id].append(record)
    for episode_records in grouped.values():
        episode_records.sort(key=lambda item: int(item.get("episode_step_ordinal", item.get("step_index", 0))))
    return dict(grouped)


def episode_success(records: list[dict[str, Any]]) -> bool:
    return bool(records[0].get("terminal_is_success")) if records else False


def choose_episode_ids(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    success_episode_ids: list[str],
    failure_episode_ids: list[str],
    num_success: int,
    num_failure: int,
    min_steps: int,
) -> list[str]:
    def candidates(want_success: bool) -> list[str]:
        ids = [
            episode_id
            for episode_id, records in grouped.items()
            if episode_success(records) is want_success and len(records) >= min_steps
        ]
        return sorted(ids, key=lambda episode_id: int(grouped[episode_id][0].get("scenario_index", 0)))

    selected = []
    for episode_id in success_episode_ids:
        if episode_id in grouped and episode_success(grouped[episode_id]):
            selected.append(episode_id)
    for episode_id in candidates(True):
        if len([item for item in selected if episode_success(grouped[item])]) >= num_success:
            break
        if episode_id not in selected:
            selected.append(episode_id)

    for episode_id in failure_episode_ids:
        if episode_id in grouped and not episode_success(grouped[episode_id]):
            selected.append(episode_id)
    for episode_id in candidates(False):
        if len([item for item in selected if not episode_success(grouped[item])]) >= num_failure:
            break
        if episode_id not in selected:
            selected.append(episode_id)

    return selected


def camera_image_path(record: dict[str, Any], camera: str) -> str | None:
    absolute = record.get("image_paths_absolute")
    if isinstance(absolute, dict) and absolute.get(camera):
        return str(absolute[camera])
    relative = record.get("image_paths")
    if isinstance(relative, dict) and relative.get(camera):
        return str(relative[camera])
    selected = record.get("selected_image_path")
    return str(selected) if selected else None


class EpisodeValueDataset:
    def __init__(self, records: list[dict[str, Any]], *, image_size: int, state_dim: int, camera: str) -> None:
        self.records = records
        self.image_size = image_size
        self.state_dim = state_dim
        self.camera = camera

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        observation = record.get("observation_fields") if isinstance(record.get("observation_fields"), dict) else {}
        action = record.get("action_fields") if isinstance(record.get("action_fields"), dict) else {}
        return {
            "image": load_pil(camera_image_path(record, self.camera), self.image_size),
            "state": vector_from_fields({"action": action, "observation": observation}, self.state_dim),
            "text": str(record.get("language_instruction") or ""),
            "target_bin": int(record["value_bin"]),
            "target_value": float(record["normalized_value"]),
            "terminal_is_success": bool(record.get("terminal_is_success")),
            "episode_id": str(record.get("episode_id")),
        }


def dtype_from_precision(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def load_qwen_value_model(
    checkpoint_path: Path,
    device: torch.device,
    precision: str,
) -> tuple[QwenValueModel, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = dict(checkpoint["config"])
    model = QwenValueModel(
        model_id=config["model_id"],
        state_dim=int(config["state_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_bins=int(config["num_bins"]),
        dtype=dtype_from_precision(precision),
        lora_rank=int(config.get("lora_rank", 0)),
        lora_alpha=float(config.get("lora_alpha", 16.0)),
        lora_dropout=float(config.get("lora_dropout", 0.0)),
        lora_last_n_layers=int(config.get("lora_last_n_layers", 0)),
        lora_target_modules=list(config.get("lora_target_modules", ["q_proj", "v_proj"])),
    )
    missing, unexpected = model.load_state_dict(checkpoint["trainable_state"], strict=False)
    trainable_missing = [
        name
        for name in missing
        if "lora_" in name or name.startswith("state_encoder.") or name.startswith("value_head.")
    ]
    if trainable_missing:
        raise RuntimeError(f"Checkpoint is missing trainable value-model keys: {trainable_missing[:20]}")
    if unexpected:
        raise RuntimeError(f"Checkpoint has unexpected keys: {unexpected[:20]}")
    model.to(device)
    model.eval()
    return model, config


@torch.no_grad()
def predict_episode_values(
    records: list[dict[str, Any]],
    *,
    model: QwenValueModel,
    processor: Any,
    device: torch.device,
    precision: str,
    image_size: int,
    state_dim: int,
    camera: str,
    batch_size: int,
) -> np.ndarray:
    dataset = EpisodeValueDataset(records, image_size=image_size, state_dim=state_dim, camera=camera)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=QwenValueCollator(processor))
    predictions: list[float] = []
    for batch in loader:
        batch = move_batch(batch, device)
        batch.pop("target_bin")
        batch.pop("target_value")
        batch.pop("terminal_is_success")
        with make_autocast(device, precision):
            logits = model(batch)
            pred_value = expected_value_from_logits(logits.float())
        predictions.extend(float(value) for value in pred_value.detach().cpu().tolist())
    return np.asarray(predictions, dtype=np.float32)


def read_frame(record: dict[str, Any], camera: str, image_size: int) -> Image.Image:
    return load_pil(camera_image_path(record, camera), image_size)


def figure_to_rgb_array(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()


def plot_value_axis(
    ax: plt.Axes,
    labeled_values: np.ndarray,
    predicted_values: np.ndarray,
    *,
    current_index: int | None,
) -> None:
    x = np.arange(len(labeled_values))
    ax.plot(x, labeled_values, color="red", linewidth=2.0, label="Labeled expected return")
    ax.plot(x, predicted_values, color="green", linewidth=2.0, label="Reconstructed return E[v p(bin)]")
    if current_index is not None:
        ax.axvline(current_index, color="black", alpha=0.35, linewidth=1.2)
    ax.set_xlabel("Trajectory step")
    ax.set_ylabel("Normalized return")
    ax.set_ylim(-1.05, 0.05)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="lower right")


def write_summary_png(
    records: list[dict[str, Any]],
    predicted_values: np.ndarray,
    output_path: Path,
    *,
    camera: str,
    image_size: int,
) -> None:
    labeled_values = np.asarray([float(record["normalized_value"]) for record in records], dtype=np.float32)
    is_success = episode_success(records)
    status = "SUCCESS" if is_success else "FAIL"
    indices = np.linspace(0, len(records) - 1, num=min(8, len(records)), dtype=int)

    fig = plt.figure(figsize=(20, 7.4), constrained_layout=True)
    grid = fig.add_gridspec(2, len(indices), height_ratios=[1.0, 1.9])
    for col, idx in enumerate(indices):
        ax_img = fig.add_subplot(grid[0, col])
        ax_img.imshow(read_frame(records[idx], camera, image_size))
        ax_img.set_title(f"t={int(records[idx].get('step_index', idx))}", fontsize=11)
        ax_img.axis("off")

    ax_plot = fig.add_subplot(grid[1, :])
    plot_value_axis(ax_plot, labeled_values, predicted_values, current_index=None)
    scenario_index = records[0].get("scenario_index", "unknown")
    fig.suptitle(f"Validation Episode {scenario_index} ({status})", fontsize=18, fontweight="bold")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_episode_mp4(
    records: list[dict[str, Any]],
    predicted_values: np.ndarray,
    output_path: Path,
    *,
    camera: str,
    image_size: int,
    frame_stride: int,
    fps: int,
) -> int:
    labeled_values = np.asarray([float(record["normalized_value"]) for record in records], dtype=np.float32)
    frame_indices = list(range(0, len(records), frame_stride))
    if frame_indices[-1] != len(records) - 1:
        frame_indices.append(len(records) - 1)

    is_success = episode_success(records)
    status = "SUCCESS" if is_success else "FAIL"
    scenario_index = records[0].get("scenario_index", "unknown")
    episode_short = str(records[0].get("episode_id", ""))[:8]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for idx in frame_indices:
            fig = plt.figure(figsize=(12, 8), constrained_layout=True)
            grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.35])
            ax_img = fig.add_subplot(grid[0, 0])
            ax_img.imshow(read_frame(records[idx], camera, image_size))
            ax_img.set_title(f"camera={camera}  step={idx}/{len(records) - 1}", fontsize=12)
            ax_img.axis("off")

            ax_plot = fig.add_subplot(grid[1, 0])
            plot_value_axis(ax_plot, labeled_values, predicted_values, current_index=idx)
            fig.suptitle(
                f"Episode {scenario_index} ({status}) - {episode_short}",
                fontsize=16,
                fontweight="bold",
            )
            writer.append_data(figure_to_rgb_array(fig))
            plt.close(fig)
    return len(frame_indices)


def write_index(output_dir: Path, artifacts: list[dict[str, Any]]) -> None:
    rows = []
    for artifact in artifacts:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(artifact['scenario_index']))}</td>"
            f"<td>{html.escape(str(artifact['status']))}</td>"
            f"<td>{html.escape(str(artifact['episode_id']))}</td>"
            f"<td>{artifact['num_steps']}</td>"
            f"<td><a href='{html.escape(artifact['png'])}'>summary png</a></td>"
            f"<td><a href='{html.escape(artifact['mp4'])}'>mp4</a></td>"
            "</tr>"
        )
    index_html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>RECAP Qwen Value Rollout Visualizations</title>
  <style>
    body { font-family: sans-serif; margin: 24px; }
    table { border-collapse: collapse; }
    th, td { border: 1px solid #ddd; padding: 8px 10px; }
    th { background: #f3f3f3; }
  </style>
</head>
<body>
  <h1>RECAP Qwen Value Rollout Visualizations</h1>
  <table>
    <thead>
      <tr><th>Scenario</th><th>Status</th><th>Episode</th><th>Steps</th><th>PNG</th><th>MP4</th></tr>
    </thead>
    <tbody>
"""
    index_html += "\n".join(rows)
    index_html += """
    </tbody>
  </table>
</body>
</html>
"""
    (output_dir / "index.html").write_text(index_html)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--value_jsonl", required=True, help="Path to value_examples.jsonl.")
    parser.add_argument("--checkpoint", required=True, help="Path to qwen_value_checkpoint.pt.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--camera", default="scene_right_0")
    parser.add_argument("--num_success", type=int, default=3)
    parser.add_argument("--num_failure", type=int, default=2)
    parser.add_argument("--success_episode_id", action="append", default=[])
    parser.add_argument("--failure_episode_id", action="append", default=[])
    parser.add_argument("--min_steps", type=int, default=20)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--frame_stride", type=int, default=4)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(Path(args.value_jsonl))
    grouped = group_records_by_episode(records)
    episode_ids = choose_episode_ids(
        grouped,
        success_episode_ids=args.success_episode_id,
        failure_episode_ids=args.failure_episode_id,
        num_success=args.num_success,
        num_failure=args.num_failure,
        min_steps=args.min_steps,
    )
    if not episode_ids:
        raise ValueError("No matching episodes found to visualize")

    device_name = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    model, config = load_qwen_value_model(Path(args.checkpoint), device, args.precision)
    processor = AutoProcessor.from_pretrained(config["model_id"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for episode_id in episode_ids:
        episode_records = grouped[episode_id]
        predicted_values = predict_episode_values(
            episode_records,
            model=model,
            processor=processor,
            device=device,
            precision=args.precision,
            image_size=args.image_size,
            state_dim=int(config["state_dim"]),
            camera=args.camera,
            batch_size=args.batch_size,
        )
        status = "success" if episode_success(episode_records) else "failure"
        scenario_index = episode_records[0].get("scenario_index", "unknown")
        safe_name = f"scenario_{scenario_index}_{status}_{episode_id[:8]}"
        png_path = output_dir / f"{safe_name}.png"
        mp4_path = output_dir / f"{safe_name}.mp4"
        write_summary_png(episode_records, predicted_values, png_path, camera=args.camera, image_size=args.image_size)
        num_video_frames = write_episode_mp4(
            episode_records,
            predicted_values,
            mp4_path,
            camera=args.camera,
            image_size=args.image_size,
            frame_stride=max(1, args.frame_stride),
            fps=args.fps,
        )
        artifact = {
            "episode_id": episode_id,
            "scenario_index": scenario_index,
            "status": status,
            "num_steps": len(episode_records),
            "num_video_frames": num_video_frames,
            "png": png_path.name,
            "mp4": mp4_path.name,
            "mean_labeled_value": float(np.mean([float(record["normalized_value"]) for record in episode_records])),
            "mean_predicted_value": float(np.mean(predicted_values)),
            "final_labeled_value": float(episode_records[-1]["normalized_value"]),
            "final_predicted_value": float(predicted_values[-1]),
        }
        artifacts.append(artifact)
        print(json.dumps(artifact, sort_keys=True))

    write_index(output_dir, artifacts)
    (output_dir / "artifacts.json").write_text(json.dumps(artifacts, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "num_episodes": len(artifacts)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
