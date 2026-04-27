"""RECAP-style policy posttraining from scored rollout trajectories.

This is an offline policy-improvement step: it loads closed-loop rollout
steps scored by a value model, keeps non-negative advantages, and applies a
weighted diffusion-policy behavior-cloning update on the collected actions.
Evaluation remains a normal policy rollout; RECAP advantages are not used at
eval time.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from vla_foundry.data.preprocessing.image_utils import ImageResizingMethod, resize_and_crop_image
from vla_foundry.data.processor.robotics_processor import RoboticsProcessor
from vla_foundry.data.robotics.utils import calculate_relative_pose, pose_to_9d, to_pose_matrix
from vla_foundry.file_utils import get_latest_checkpoint, load_model_checkpoint, yaml_load
from vla_foundry.hf_hub import normalize_checkpoint_locator, resolve_hf_path
from vla_foundry.inference.robotics.utils import center_crop, relative_to_absolute_map
from vla_foundry.models.registry import create_model
from vla_foundry.params.train_experiment_params import load_experiment_params_from_yaml
from vla_foundry.precision import get_autocast
from vla_foundry.recap.label_trajectories import load_jsonl

ROOT_FILES = [
    "config.yaml",
    "config_model.yaml",
    "config_normalizer.yaml",
    "config_processor.yaml",
    "preprocessing_config.yaml",
    "processing_metadata.json",
    "stats.json",
]


@dataclass
class RecapPolicyTrainConfig:
    scored_jsonl: str
    checkpoint_directory: str
    checkpoint_path: str | None
    output_dir: str
    min_positive_advantage: float
    min_weight: float
    max_examples: int | None
    max_train_steps: int
    batch_size: int
    lr: float
    weight_decay: float
    grad_clip_norm: float
    precision: str
    device: str
    seed: int
    num_workers: int
    disable_weighted_sampler: bool
    save_optimizer: bool
    log_every: int
    wandb: bool
    wandb_project: str | None
    wandb_entity: str | None
    run_name: str | None


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _as_vector(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1)
    return array.reshape(-1)


def _load_image(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _preprocess_image(
    image: np.ndarray,
    *,
    resize_size: tuple[int, int] | None,
    resize_method: ImageResizingMethod,
    crop_shape: tuple[int, int] | None,
) -> np.ndarray:
    if resize_size is not None:
        image = resize_and_crop_image(image, target_size=resize_size, resize_method=resize_method)
    if crop_shape is not None:
        image = center_crop(image, crop_shape[0], crop_shape[1])
        if isinstance(image, Image.Image):
            image = np.asarray(image)
    return np.asarray(image, dtype=np.uint8)


def _resolve_experiment_dir(checkpoint_directory: str) -> tuple[str, str]:
    locator = normalize_checkpoint_locator(checkpoint_directory).rstrip("/")
    local_dir = resolve_hf_path(locator) if locator.startswith("hf://") else locator
    return locator, local_dir


def _resolve_checkpoint_path(checkpoint_locator: str, checkpoint_path: str | None) -> str:
    if checkpoint_path:
        return normalize_checkpoint_locator(checkpoint_path)
    latest = get_latest_checkpoint(checkpoint_locator)
    if latest is None:
        raise FileNotFoundError(f"No checkpoint_*.pt found under {checkpoint_locator}/checkpoints")
    return latest


def _copy_experiment_files(src_dir: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ROOT_FILES:
        src = Path(src_dir) / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)


def _save_policy_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    *,
    checkpoint_num: int,
    global_step: int,
    save_optimizer: bool,
) -> None:
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_num": checkpoint_num,
            "global_step": global_step,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "shard_shuffle_seed_per_dataset": None,
        },
        checkpoints_dir / f"checkpoint_{checkpoint_num}.pt",
    )
    if save_optimizer:
        torch.save(optimizer.state_dict(), checkpoints_dir / f"optimizer_{checkpoint_num}.pt")


def _pose_group_lookup(pose_groups: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for group in pose_groups or []:
        position_key = group.get("position_key")
        rotation_key = group.get("rotation_key")
        if position_key and rotation_key:
            lookup[position_key] = group
            lookup[rotation_key] = group
    return lookup


def _relative_pose_components(
    *,
    action_sequence: list[dict[str, Any]],
    anchor_record: dict[str, Any],
    pose_group: dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    xyz_key = pose_group["position_key"]
    rot_6d_key = pose_group["rotation_key"]
    observation = anchor_record.get("observation_fields") or {}
    xyz_actual_key = xyz_key.replace("__action__", "__actual__")
    rot_actual_key = rot_6d_key.replace("__action__", "__actual__")
    reference_xyz = _as_vector(observation[xyz_actual_key])
    reference_rot = _as_vector(observation[rot_actual_key])
    action_xyz = np.stack([_as_vector(step[xyz_key]) for step in action_sequence], axis=0)
    action_rot = np.stack([_as_vector(step[rot_6d_key]) for step in action_sequence], axis=0)
    relative_pose = calculate_relative_pose(
        to_pose_matrix(action_xyz, action_rot),
        to_pose_matrix(reference_xyz, reference_rot),
    )
    return pose_to_9d(relative_pose)


class RecapPolicyDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        data_params: Any,
        resize_size: tuple[int, int] | None,
        resize_method: ImageResizingMethod,
        crop_shape: tuple[int, int] | None,
    ) -> None:
        by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_episode[str(record["episode_id"])].append(record)
        for episode_records in by_episode.values():
            episode_records.sort(key=lambda item: int(item.get("step_index", 0)))

        self.data_params = data_params
        self.resize_size = resize_size
        self.resize_method = resize_method
        self.crop_shape = crop_shape
        self.records_by_episode = by_episode
        self.pose_lookup = _pose_group_lookup(getattr(data_params, "pose_groups", []))
        self.examples: list[tuple[str, int]] = []
        for episode_id, episode_records in sorted(by_episode.items()):
            for idx, record in enumerate(episode_records):
                if float(record.get("recap_weight", 0.0)) <= 0.0:
                    continue
                self.examples.append((episode_id, idx))
        if not self.examples:
            raise ValueError("No usable positive-weight RECAP examples found")

    def __len__(self) -> int:
        return len(self.examples)

    @property
    def weights(self) -> list[float]:
        return [
            float(self.records_by_episode[episode_id][idx].get("recap_weight", 0.0))
            for episode_id, idx in self.examples
        ]

    def __getitem__(self, item: int) -> dict[str, Any]:
        episode_id, idx = self.examples[item]
        episode_records = self.records_by_episode[episode_id]
        anchor = episode_records[idx]
        past = int(self.data_params.lowdim_past_timesteps)
        future = int(self.data_params.lowdim_future_timesteps)
        raw_sequence_indices = list(range(idx - past, idx + future + 1))
        valid_sequence = [0 <= source_idx < len(episode_records) for source_idx in raw_sequence_indices]
        sequence_records = [
            episode_records[min(max(source_idx, 0), len(episode_records) - 1)] for source_idx in raw_sequence_indices
        ]
        sequence_actions = [record.get("action_fields") or {} for record in sequence_records]
        if any("error" in action for action in sequence_actions):
            raise ValueError(f"Action extraction error in episode {episode_id} index {idx}")

        images: dict[str, np.ndarray] = {}
        image_indices = list(self.data_params.image_indices)
        for image_offset in image_indices:
            source_idx = min(max(idx + int(image_offset), 0), len(episode_records) - 1)
            source = episode_records[source_idx]
            image_paths = source.get("image_paths_absolute") or {}
            for camera_name in self.data_params.camera_names:
                path = image_paths.get(camera_name)
                if not path:
                    continue
                image = _load_image(path)
                image = _preprocess_image(
                    image,
                    resize_size=self.resize_size,
                    resize_method=self.resize_method,
                    crop_shape=self.crop_shape,
                )
                images[f"{camera_name}_t{image_offset}"] = image

        lowdim: dict[str, np.ndarray] = {}
        relative_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for field in self.data_params.action_fields:
            absolute_field = relative_to_absolute_map(field)
            if field.endswith("_relative"):
                pose_group = self.pose_lookup.get(absolute_field)
                if pose_group is None:
                    raise KeyError(f"No pose group configured for relative action field {field}")
                cache_key = f"{pose_group['position_key']}|{pose_group['rotation_key']}"
                if cache_key not in relative_cache:
                    relative_cache[cache_key] = _relative_pose_components(
                        action_sequence=sequence_actions,
                        anchor_record=anchor,
                        pose_group=pose_group,
                    )
                relative_xyz, relative_rot = relative_cache[cache_key]
                lowdim[field] = relative_xyz if absolute_field == pose_group["position_key"] else relative_rot
            else:
                lowdim[field] = np.stack([_as_vector(action[absolute_field]) for action in sequence_actions], axis=0)

        for field in self.data_params.proprioception_fields:
            absolute_field = relative_to_absolute_map(field)
            lowdim[field] = np.stack(
                [_as_vector((record.get("observation_fields") or {})[absolute_field]) for record in sequence_records],
                axis=0,
            )

        seq_len = len(sequence_records)
        past_mask = torch.zeros(seq_len, dtype=torch.bool)
        future_mask = torch.zeros(seq_len, dtype=torch.bool)
        for seq_idx, (raw_idx, is_valid) in enumerate(zip(raw_sequence_indices, valid_sequence, strict=True)):
            if not is_valid:
                continue
            if raw_idx < idx:
                past_mask[seq_idx] = True
            else:
                future_mask[seq_idx] = True

        return {
            "images": images,
            "lowdim": lowdim,
            "metadata": {
                "anchor_relative_idx": past,
                "original_anchor_relative_idx": past,
                "episode_id": episode_id,
                "step_index": int(anchor.get("step_index", idx)),
            },
            "language_instruction": str(anchor.get("language_instruction") or ""),
            "past_mask": past_mask,
            "future_mask": future_mask,
            "recap_weight": float(anchor["recap_weight"]),
            "recap_advantage": float(anchor.get("recap_advantage", 0.0)),
        }


class RecapPolicyCollator:
    def __init__(self, robotics_processor: RoboticsProcessor, image_names: list[str], data_params: Any) -> None:
        self.robotics_processor = robotics_processor
        self.image_names = image_names
        self.data_params = data_params

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = {
            "images": [example["images"] for example in examples],
            "lowdim": [example["lowdim"] for example in examples],
            "metadata": [example["metadata"] for example in examples],
            "language_instruction": [example["language_instruction"] for example in examples],
            "past_mask": torch.stack([example["past_mask"] for example in examples]),
            "future_mask": torch.stack([example["future_mask"] for example in examples]),
        }
        processed = self.robotics_processor.process_inputs(batch, image_names=self.image_names)
        processed = self.robotics_processor.add_action_and_proprioception_fields(
            processed,
            action_fields=self.data_params.action_fields,
            proprioception_fields=self.data_params.proprioception_fields,
        )
        processed["recap_weight"] = torch.tensor([example["recap_weight"] for example in examples], dtype=torch.float32)
        processed["recap_advantage"] = torch.tensor(
            [example["recap_advantage"] for example in examples], dtype=torch.float32
        )
        return processed


def weighted_diffusion_loss(
    predicted_direction: torch.Tensor,
    target_direction: torch.Tensor,
    future_mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    seq_len = min(future_mask.shape[1], predicted_direction.shape[1])
    predicted_direction = predicted_direction[:, -seq_len:]
    target_direction = target_direction[:, -seq_len:]
    future_mask = future_mask[:, -seq_len:]
    element_loss = F.mse_loss(predicted_direction.float(), target_direction.float(), reduction="none")
    valid = future_mask.unsqueeze(-1).to(element_loss.dtype)
    denominator = (valid.sum(dim=(1, 2)) * element_loss.shape[-1]).clamp_min(1.0)
    per_sample = (element_loss * valid).sum(dim=(1, 2)) / denominator
    weights = weights.to(per_sample.device, dtype=per_sample.dtype)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-6)


def _prepare_records(config: RecapPolicyTrainConfig) -> list[dict[str, Any]]:
    records = load_jsonl(Path(config.scored_jsonl))
    kept = [
        record
        for record in records
        if float(record.get("recap_positive_advantage", 0.0)) >= config.min_positive_advantage
        and float(record.get("recap_weight", 0.0)) >= config.min_weight
    ]
    if config.max_examples is not None:
        rng = random.Random(config.seed)
        rng.shuffle(kept)
        kept = kept[: config.max_examples]
    if not kept:
        raise ValueError("No records left after RECAP advantage/weight filtering")
    return kept


def _preprocess_params(
    local_checkpoint_dir: str,
    cfg: Any,
) -> tuple[tuple[int, int] | None, ImageResizingMethod, tuple[int, int] | None]:
    preprocessing_path = os.path.join(local_checkpoint_dir, "preprocessing_config.yaml")
    resize_size = None
    resize_method = ImageResizingMethod.CENTER_CROP
    if os.path.exists(preprocessing_path):
        preprocessing_config = yaml_load(preprocessing_path)
        raw_resize_size = preprocessing_config.get("resize_images_size")
        if raw_resize_size is not None:
            resize_size = tuple(int(value) for value in raw_resize_size)
        raw_resize_method = preprocessing_config.get("resize_images_method")
        if raw_resize_method:
            resize_method = ImageResizingMethod(str(raw_resize_method).lower())
    crop_shape = None
    crop_cfg = cfg.data.augmentation.image.crop
    if getattr(crop_cfg, "enabled", False):
        crop_shape = tuple(int(value) for value in crop_cfg.shape)
    return resize_size, resize_method, crop_shape


def train(config: RecapPolicyTrainConfig) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    checkpoint_locator, local_checkpoint_dir = _resolve_experiment_dir(config.checkpoint_directory)
    checkpoint_path = _resolve_checkpoint_path(checkpoint_locator, config.checkpoint_path)
    output_dir = Path(config.output_dir)
    _copy_experiment_files(local_checkpoint_dir, output_dir)
    (output_dir / "recap_policy_train_config.json").write_text(
        json.dumps(asdict(config) | {"resolved_checkpoint_path": checkpoint_path}, indent=2, sort_keys=True) + "\n"
    )

    cfg = load_experiment_params_from_yaml(f"{checkpoint_locator}/config.yaml", localize_params=True)
    object.__setattr__(cfg.distributed, "use_distributed", False)
    object.__setattr__(cfg.distributed, "fsdp", False)
    object.__setattr__(cfg.distributed, "world_size", 1)
    object.__setattr__(cfg.distributed, "rank", 0)
    object.__setattr__(cfg.distributed, "local_rank", 0)
    object.__setattr__(cfg.distributed, "device", "cuda:0" if torch.cuda.is_available() else "cpu")
    object.__setattr__(cfg.model, "num_action_head_repeats", None)
    device_name = config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    records = _prepare_records(config)
    resize_size, resize_method, crop_shape = _preprocess_params(local_checkpoint_dir, cfg)
    robotics_processor = RoboticsProcessor.from_pretrained(local_checkpoint_dir)
    dataset = RecapPolicyDataset(
        records,
        data_params=cfg.data,
        resize_size=resize_size,
        resize_method=resize_method,
        crop_shape=crop_shape,
    )
    sampler = None
    if not config.disable_weighted_sampler:
        sampler = WeightedRandomSampler(dataset.weights, num_samples=len(dataset), replacement=True)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=RecapPolicyCollator(robotics_processor, cfg.data.image_names, cfg.data),
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = create_model(cfg.model, load_pretrained=False)
    load_model_checkpoint(model, checkpoint_path)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    autocast = get_autocast(config.precision)

    run = None
    if config.wandb:
        import wandb

        run = wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            name=config.run_name,
            config={
                **asdict(config),
                "resolved_checkpoint_path": checkpoint_path,
                "num_filtered_examples": len(records),
                "num_trainable_examples": len(dataset),
            },
        )

    data_iterator = iter(loader)
    metrics: dict[str, Any] = {
        "num_filtered_examples": len(records),
        "num_trainable_examples": len(dataset),
        "checkpoint_path": checkpoint_path,
    }
    step = 0
    while step < config.max_train_steps:
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(loader)
            batch = next(data_iterator)

        weights = batch.pop("recap_weight").to(device)
        advantages = batch.pop("recap_advantage").to(device)
        for key, value in list(batch.items()):
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device, non_blocking=True)
        batch["noise"] = torch.randn_like(batch["actions"])
        targets = batch["noise"] - batch["actions"]

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            outputs = model(**batch)
            loss = weighted_diffusion_loss(outputs, targets, batch["future_mask"], weights)
        loss.backward()
        if config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()
        step += 1

        if step == 1 or step % config.log_every == 0:
            metrics = {
                "step": step,
                "train/loss": float(loss.detach().cpu().item()),
                "train/mean_weight": float(weights.detach().float().mean().cpu().item()),
                "train/max_weight": float(weights.detach().float().max().cpu().item()),
                "train/mean_advantage": float(advantages.detach().float().mean().cpu().item()),
                "num_filtered_examples": len(records),
                "num_trainable_examples": len(dataset),
            }
            print(json.dumps(metrics, sort_keys=True), flush=True)
            if run is not None:
                run.log(metrics, step=step)

    _save_policy_checkpoint(
        model,
        optimizer,
        output_dir,
        checkpoint_num=1,
        global_step=step,
        save_optimizer=config.save_optimizer,
    )
    metrics = {**metrics, "saved_checkpoint": str(output_dir / "checkpoints" / "checkpoint_1.pt")}
    (output_dir / "recap_policy_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    if run is not None:
        run.log(metrics, step=step)
        run.finish()
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored_jsonl", required=True)
    parser.add_argument("--checkpoint_directory", required=True)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_positive_advantage", type=float, default=1e-6)
    parser.add_argument("--min_weight", type=float, default=1e-6)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_train_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-8)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--precision", default="amp_bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--disable_weighted_sampler", action="store_true")
    parser.add_argument("--save_optimizer", action="store_true")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--run_name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RecapPolicyTrainConfig(**vars(args))
    metrics = train(config)
    print(json.dumps(metrics, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
