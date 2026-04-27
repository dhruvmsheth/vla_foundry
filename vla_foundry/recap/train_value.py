"""Train a lightweight RECAP value function on labeled rollout steps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from vla_foundry.recap.label_trajectories import load_jsonl

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class ValueTrainConfig:
    train_jsonl: str
    val_jsonl: str | None
    output_dir: str
    image_size: int
    max_text_tokens: int
    text_hash_size: int
    text_dim: int
    state_dim: int
    hidden_dim: int
    num_bins: int
    batch_size: int
    epochs: int
    max_train_steps: int | None
    lr: float
    weight_decay: float
    num_workers: int
    seed: int
    device: str
    precision: str
    wandb: bool
    wandb_project: str | None
    wandb_entity: str | None
    run_name: str | None


def flatten_numeric(value: Any) -> list[float]:
    if isinstance(value, bool):
        return [float(value)]
    if isinstance(value, int | float):
        return [float(value)]
    if isinstance(value, list | tuple):
        out: list[float] = []
        for item in value:
            out.extend(flatten_numeric(item))
        return out
    if isinstance(value, dict):
        out = []
        for key in sorted(value):
            out.extend(flatten_numeric(value[key]))
        return out
    return []


def vector_from_fields(fields: dict[str, Any], dim: int) -> torch.Tensor:
    values: list[float] = []
    for key in sorted(fields):
        values.extend(flatten_numeric(fields[key]))
    if len(values) >= dim:
        values = values[:dim]
    else:
        values.extend([0.0] * (dim - len(values)))
    return torch.tensor(values, dtype=torch.float32)


def stable_hash_token(token: str, hash_size: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % hash_size


def tokenize_text(text: str, hash_size: int, max_tokens: int) -> torch.Tensor:
    tokens = [stable_hash_token(match.group(0).lower(), hash_size) for match in TOKEN_RE.finditer(text)]
    tokens = tokens[:max_tokens]
    if len(tokens) < max_tokens:
        tokens.extend([hash_size] * (max_tokens - len(tokens)))
    return torch.tensor(tokens, dtype=torch.long)


def load_image(path: str | None, image_size: int) -> torch.Tensor:
    if not path:
        return torch.zeros(3, image_size, image_size, dtype=torch.float32)
    image_path = Path(path)
    if not image_path.exists():
        return torch.zeros(3, image_size, image_size, dtype=torch.float32)
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
    return (tensor - mean) / std


class RecapValueDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        *,
        image_size: int,
        state_dim: int,
        text_hash_size: int,
        max_text_tokens: int,
    ):
        self.path = Path(jsonl_path)
        self.records = load_jsonl(self.path)
        if not self.records:
            raise ValueError(f"No records found in {self.path}")
        self.image_size = image_size
        self.state_dim = state_dim
        self.text_hash_size = text_hash_size
        self.max_text_tokens = max_text_tokens

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int | float | bool]:
        record = self.records[idx]
        image = load_image(record.get("selected_image_path"), self.image_size)
        observation = record.get("observation_fields") if isinstance(record.get("observation_fields"), dict) else {}
        action = record.get("action_fields") if isinstance(record.get("action_fields"), dict) else {}
        state = vector_from_fields({"action": action, "observation": observation}, self.state_dim)
        text = tokenize_text(str(record.get("language_instruction") or ""), self.text_hash_size, self.max_text_tokens)
        return {
            "image": image,
            "state": state,
            "text_tokens": text,
            "target_bin": torch.tensor(int(record["value_bin"]), dtype=torch.long),
            "target_value": torch.tensor(float(record["normalized_value"]), dtype=torch.float32),
            "terminal_is_success": bool(record.get("terminal_is_success")),
            "episode_id": str(record.get("episode_id")),
        }


class ValueNet(nn.Module):
    def __init__(
        self,
        *,
        num_bins: int,
        state_dim: int,
        text_hash_size: int,
        text_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 256),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.text_embedding = nn.Embedding(text_hash_size + 1, text_dim, padding_idx=text_hash_size)
        self.state_encoder = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, hidden_dim), nn.SiLU())
        self.head = nn.Sequential(
            nn.Linear(256 + text_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_bins),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(image)
        text_mask = text_tokens != self.text_embedding.padding_idx
        embedded = self.text_embedding(text_tokens)
        denominator = text_mask.sum(dim=1, keepdim=True).clamp_min(1)
        text_features = (embedded * text_mask.unsqueeze(-1)).sum(dim=1) / denominator
        state_features = self.state_encoder(state)
        return self.head(torch.cat([image_features, text_features, state_features], dim=-1))


def expected_value_from_logits(logits: torch.Tensor) -> torch.Tensor:
    bins = torch.linspace(-1.0, 0.0, logits.shape[-1], device=logits.device, dtype=logits.dtype)
    return (F.softmax(logits, dim=-1) * bins).sum(dim=-1)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def make_autocast(device: torch.device, precision: str):
    enabled = precision in {"bf16", "fp16"} and device.type == "cuda"
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


@torch.no_grad()
def evaluate(model: ValueNet, loader: DataLoader, device: torch.device, precision: str) -> dict[str, float]:
    model.eval()
    total_loss = total_mae = total_correct = total_count = 0.0
    success_values: list[float] = []
    failure_values: list[float] = []
    for batch in loader:
        batch = move_batch(batch, device)
        with make_autocast(device, precision):
            logits = model(batch["image"], batch["state"], batch["text_tokens"])
            loss = F.cross_entropy(logits, batch["target_bin"])
        pred_value = expected_value_from_logits(logits.float())
        target_value = batch["target_value"]
        count = float(target_value.numel())
        total_loss += float(loss.item()) * count
        total_mae += float((pred_value - target_value).abs().sum().item())
        total_correct += float((logits.argmax(dim=-1) == batch["target_bin"]).sum().item())
        total_count += count
        terminal_success = batch["terminal_is_success"]
        if isinstance(terminal_success, torch.Tensor):
            terminal_success = terminal_success.detach().cpu().tolist()
        for value, is_success in zip(pred_value.detach().cpu().tolist(), terminal_success, strict=True):
            if bool(is_success):
                success_values.append(float(value))
            else:
                failure_values.append(float(value))
    metrics = {
        "loss": total_loss / max(1.0, total_count),
        "mae": total_mae / max(1.0, total_count),
        "bin_accuracy": total_correct / max(1.0, total_count),
        "mean_pred_success": float(np.mean(success_values)) if success_values else math.nan,
        "mean_pred_failure": float(np.mean(failure_values)) if failure_values else math.nan,
        "num_examples": total_count,
    }
    metrics["success_failure_gap"] = metrics["mean_pred_success"] - metrics["mean_pred_failure"]
    return metrics


def save_checkpoint(
    model: ValueNet,
    optimizer: torch.optim.Optimizer,
    config: ValueTrainConfig,
    output_dir: Path,
    step: int,
    metrics: dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(config),
            "step": step,
            "metrics": metrics,
        },
        output_dir / "value_checkpoint.pt",
    )
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def train(config: ValueTrainConfig) -> dict[str, float]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device_name = config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    train_dataset = RecapValueDataset(
        config.train_jsonl,
        image_size=config.image_size,
        state_dim=config.state_dim,
        text_hash_size=config.text_hash_size,
        max_text_tokens=config.max_text_tokens,
    )
    val_dataset = (
        RecapValueDataset(
            config.val_jsonl,
            image_size=config.image_size,
            state_dim=config.state_dim,
            text_hash_size=config.text_hash_size,
            max_text_tokens=config.max_text_tokens,
        )
        if config.val_jsonl
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        if val_dataset is not None
        else None
    )

    model = ValueNet(
        num_bins=config.num_bins,
        state_dim=config.state_dim,
        text_hash_size=config.text_hash_size,
        text_dim=config.text_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    run = None
    if config.wandb:
        import wandb

        run = wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            name=config.run_name,
            config=asdict(config),
        )

    step = 0
    latest_metrics: dict[str, float] = {}
    output_dir = Path(config.output_dir)
    for epoch in range(config.epochs):
        model.train()
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with make_autocast(device, config.precision):
                logits = model(batch["image"], batch["state"], batch["text_tokens"])
                loss = F.cross_entropy(logits, batch["target_bin"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1

            if step % 20 == 0 or step == 1:
                pred_value = expected_value_from_logits(logits.detach().float())
                train_metrics = {
                    "train/loss": float(loss.item()),
                    "train/mae": float((pred_value - batch["target_value"]).abs().mean().item()),
                    "train/bin_accuracy": float((logits.argmax(dim=-1) == batch["target_bin"]).float().mean().item()),
                    "epoch": epoch,
                    "step": step,
                }
                print(json.dumps(train_metrics, sort_keys=True))
                if run is not None:
                    run.log(train_metrics, step=step)

            if config.max_train_steps is not None and step >= config.max_train_steps:
                break
        if val_loader is not None:
            evaluated_metrics = evaluate(model, val_loader, device, config.precision)
            val_metrics = {f"val/{key}": value for key, value in evaluated_metrics.items()}
            latest_metrics = {"step": step, "epoch": epoch, **val_metrics}
            print(json.dumps(latest_metrics, indent=2, sort_keys=True))
            if run is not None:
                run.log(latest_metrics, step=step)
            save_checkpoint(model, optimizer, config, output_dir, step, latest_metrics)
        else:
            latest_metrics = {"step": step, "epoch": epoch}
            save_checkpoint(model, optimizer, config, output_dir, step, latest_metrics)

        if config.max_train_steps is not None and step >= config.max_train_steps:
            break

    if run is not None:
        run.finish()
    return latest_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_text_tokens", type=int, default=32)
    parser.add_argument("--text_hash_size", type=int, default=4096)
    parser.add_argument("--text_dim", type=int, default=128)
    parser.add_argument("--state_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_bins", type=int, default=201)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--run_name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ValueTrainConfig(**vars(args))
    metrics = train(config)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
