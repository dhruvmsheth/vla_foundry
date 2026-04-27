#!/usr/bin/env python3
"""Trajectory capture utilities for closed-loop LBM policy evaluation.

The policy server does not receive final success labels from ``lbm_eval``. This
collector records step-level observations and actions keyed by client UUID; the
resulting files can be joined with the corresponding ``results.json`` after eval.
"""

import json
import logging
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vla_foundry.inference.robotics.utils import any_to_actual_map, relative_to_absolute_map


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            array = np.zeros_like(array, dtype=np.uint8)
        else:
            min_value = float(finite.min())
            max_value = float(finite.max())
            if max_value <= 1.0 and min_value >= 0.0:
                array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
            elif max_value > min_value:
                array = np.clip((array - min_value) / (max_value - min_value) * 255.0, 0, 255).astype(np.uint8)
            else:
                array = np.zeros_like(array, dtype=np.uint8)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    return array


class TrajectoryCollector:
    """Append-only recorder for observations/actions emitted by the policy server."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        checkpoint_directory: str,
        checkpoint_path: str,
        save_images: bool = True,
        image_every_n: int = 1,
        image_format: str = "jpg",
        jpeg_quality: int = 90,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.checkpoint_directory = checkpoint_directory
        self.checkpoint_path = checkpoint_path
        self.save_images = save_images
        self.image_every_n = max(1, image_every_n)
        self.image_format = image_format.lower().lstrip(".")
        self.jpeg_quality = jpeg_quality
        self._episode_counts: defaultdict[str, int] = defaultdict(int)
        self._active_episodes: dict[str, dict[str, Any]] = {}

        if self.image_format not in {"jpg", "jpeg", "png"}:
            raise ValueError(f"Unsupported trajectory image format: {image_format}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(
            self.output_dir / "collector_metadata.json",
            {
                "schema_version": 1,
                "created_at": _utc_now(),
                "checkpoint_directory": checkpoint_directory,
                "checkpoint_path": checkpoint_path,
                "save_images": save_images,
                "image_every_n": self.image_every_n,
                "image_format": self.image_format,
            },
        )
        logging.info("RECAP trajectory collection enabled: %s", self.output_dir)

    def start_episode(self, client_id: uuid.UUID, reset_seed: int | None = None) -> None:
        client_key = str(client_id)
        episode_index = self._episode_counts[client_key]
        self._episode_counts[client_key] += 1

        episode_id = f"{client_key}_episode_{episode_index:06d}"
        episode_dir = self.output_dir / "episodes" / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        episode = {
            "schema_version": 1,
            "episode_id": episode_id,
            "client_id": client_key,
            "episode_index_for_client": episode_index,
            "reset_seed": reset_seed,
            "started_at": _utc_now(),
            "step_count": 0,
            "episode_dir": str(episode_dir.relative_to(self.output_dir)),
        }
        self._active_episodes[client_key] = episode
        self._append_jsonl(self.output_dir / "episodes.jsonl", episode)

    def record_step(
        self,
        *,
        client_id: uuid.UUID,
        observation: Any,
        action: Any,
        adapter: Any,
        step_index: int,
        language_instruction: str | None,
        generated_new_action_chunk: bool,
        remaining_actions: int,
        remaining_slots: int,
        open_loop_step: int,
    ) -> None:
        client_key = str(client_id)
        if client_key not in self._active_episodes:
            self.start_episode(client_id, reset_seed=None)

        episode = self._active_episodes[client_key]
        episode_dir = self.output_dir / episode["episode_dir"]
        rel_image_paths = {}
        if self.save_images and step_index % self.image_every_n == 0:
            rel_image_paths = self._save_images(episode_dir, step_index, observation, adapter)

        step_record = {
            "schema_version": 1,
            "event": "step",
            "recorded_at": _utc_now(),
            "episode_id": episode["episode_id"],
            "client_id": client_key,
            "reset_seed": episode.get("reset_seed"),
            "step_index": step_index,
            "language_instruction": language_instruction,
            "generated_new_action_chunk": generated_new_action_chunk,
            "open_loop_step": open_loop_step,
            "remaining_actions_in_buffer": remaining_actions,
            "remaining_action_slots": remaining_slots,
            "available_cameras": sorted(getattr(observation, "visuo", {}).keys()),
            "image_paths": rel_image_paths,
            "observation_fields": self._extract_observation_fields(observation, adapter),
            "action_fields": self._extract_action_fields(action, adapter),
        }
        self._append_jsonl(episode_dir / "trajectory.jsonl", step_record)
        episode["step_count"] = step_index + 1

    def _extract_observation_fields(self, observation: Any, adapter: Any) -> dict[str, Any]:
        fields = set()
        for field in list(getattr(adapter, "action_fields", [])) + list(getattr(adapter, "proprioception_fields", [])):
            fields.add(any_to_actual_map(relative_to_absolute_map(field)))

        values = {}
        for field in sorted(fields):
            try:
                values[field] = _jsonable(adapter.field_mapping.get_field(observation, field))
            except Exception as exc:  # noqa: BLE001 - best-effort telemetry should not break eval.
                values[field] = {"error": repr(exc)}
        return values

    def _extract_action_fields(self, action: Any, adapter: Any) -> dict[str, Any]:
        try:
            return _jsonable(adapter.action_mapping.from_sim(action))
        except Exception as exc:  # noqa: BLE001 - best-effort telemetry should not break eval.
            return {"error": repr(exc)}

    def _save_images(self, episode_dir: Path, step_index: int, observation: Any, adapter: Any) -> dict[str, str]:
        try:
            images = adapter.field_mapping.get_all_images(observation)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed to read trajectory images: %r", exc)
            return {}

        image_dir = episode_dir / "images" / f"step_{step_index:06d}"
        image_dir.mkdir(parents=True, exist_ok=True)
        rel_paths = {}
        for camera_name, image in images.items():
            suffix = "jpg" if self.image_format == "jpeg" else self.image_format
            path = image_dir / f"{camera_name}.{suffix}"
            pil_image = Image.fromarray(_to_uint8_rgb(image))
            save_kwargs = {}
            if suffix == "jpg":
                pil_image = pil_image.convert("RGB")
                save_kwargs = {"quality": self.jpeg_quality, "optimize": True}
            pil_image.save(path, **save_kwargs)
            rel_paths[camera_name] = str(path.relative_to(self.output_dir))
        return rel_paths

    @staticmethod
    def _write_json(path: Path, record: dict[str, Any]) -> None:
        path.write_text(json.dumps(_jsonable(record), indent=2, sort_keys=True) + "\n")

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        with path.open("a") as handle:
            handle.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
