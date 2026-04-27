"""Train a Qwen3-VL RECAP value function with optional lightweight LoRA adapters."""

from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModelForImageTextToText, AutoProcessor

from vla_foundry.recap.label_trajectories import load_jsonl
from vla_foundry.recap.train_value import expected_value_from_logits, make_autocast, vector_from_fields

LAYER_RE = re.compile(r"language_model\.layers\.(\d+)\.")


@dataclass
class QwenValueTrainConfig:
    train_jsonl: str
    val_jsonl: str | None
    output_dir: str
    model_id: str
    image_size: int
    state_dim: int
    hidden_dim: int
    num_bins: int
    batch_size: int
    epochs: int
    max_train_steps: int | None
    lr: float
    weight_decay: float
    value_loss_weight: float
    balance_success_failure: bool
    num_workers: int
    seed: int
    device: str
    precision: str
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    lora_last_n_layers: int
    lora_target_modules: list[str]
    wandb: bool
    wandb_project: str | None
    wandb_entity: str | None
    run_name: str | None


def load_pil(path: str | None, image_size: int) -> Image.Image:
    if path and Path(path).exists():
        return Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    return Image.new("RGB", (image_size, image_size), color=(0, 0, 0))


class QwenValueDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, *, image_size: int, state_dim: int):
        self.path = Path(jsonl_path)
        self.records = load_jsonl(self.path)
        if not self.records:
            raise ValueError(f"No records found in {self.path}")
        self.image_size = image_size
        self.state_dim = state_dim

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        observation = record.get("observation_fields") if isinstance(record.get("observation_fields"), dict) else {}
        action = record.get("action_fields") if isinstance(record.get("action_fields"), dict) else {}
        return {
            "image": load_pil(record.get("selected_image_path"), self.image_size),
            "state": vector_from_fields({"action": action, "observation": observation}, self.state_dim),
            "text": str(record.get("language_instruction") or ""),
            "target_bin": int(record["value_bin"]),
            "target_value": float(record["normalized_value"]),
            "terminal_is_success": bool(record.get("terminal_is_success")),
            "episode_id": str(record.get("episode_id")),
        }


class QwenValueCollator:
    def __init__(self, processor: Any):
        self.processor = processor

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        images = [example["image"] for example in examples]
        texts = []
        for example in examples:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example["image"]},
                        {"type": "text", "text": f"Estimate task progress value: {example['text']}"},
                    ],
                }
            ]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        model_inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        model_inputs["state"] = torch.stack([example["state"] for example in examples])
        model_inputs["target_bin"] = torch.tensor([example["target_bin"] for example in examples], dtype=torch.long)
        model_inputs["target_value"] = torch.tensor(
            [example["target_value"] for example in examples],
            dtype=torch.float32,
        )
        model_inputs["terminal_is_success"] = torch.tensor(
            [example["terminal_is_success"] for example in examples],
            dtype=torch.bool,
        )
        return model_inputs


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b) * self.scaling
        return base_out + lora_out.to(base_out.dtype)


class QwenValueModel(nn.Module):
    def __init__(
        self,
        *,
        model_id: str,
        state_dim: int,
        hidden_dim: int,
        num_bins: int,
        dtype: torch.dtype,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
        lora_last_n_layers: int,
        lora_target_modules: list[str],
    ) -> None:
        super().__init__()
        self.backbone = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        self.lora_module_names: list[str] = []
        if lora_rank > 0:
            self.lora_module_names = add_lora_adapters(
                self.backbone,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                last_n_layers=lora_last_n_layers,
                target_modules=lora_target_modules,
            )

        hidden_size = int(getattr(self.backbone.config, "text_config", self.backbone.config).hidden_size)
        self.state_encoder = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, hidden_dim), nn.SiLU())
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_bins),
        )

    def forward(self, model_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        state = model_inputs.pop("state")
        outputs = self.backbone(**model_inputs, output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states[-1]
        mask = model_inputs["attention_mask"].to(hidden.dtype)
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        state_features = self.state_encoder(state.to(pooled.dtype))
        logits = self.value_head(torch.cat([pooled, state_features], dim=-1).float())
        return logits


def _module_parent(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _layer_index(module_name: str) -> int | None:
    match = LAYER_RE.search(module_name)
    return int(match.group(1)) if match else None


def add_lora_adapters(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    last_n_layers: int,
    target_modules: list[str],
) -> list[str]:
    layer_indices = [idx for name, _module in model.named_modules() if (idx := _layer_index(name)) is not None]
    min_layer = 0
    if layer_indices and last_n_layers > 0:
        min_layer = max(layer_indices) - last_n_layers + 1

    replacements = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(target) for target in target_modules):
            continue
        layer_idx = _layer_index(name)
        if layer_idx is not None and layer_idx < min_layer:
            continue
        parent, child_name = _module_parent(model, name)
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        replacements.append(name)
    if not replacements:
        raise ValueError("No LoRA target modules matched the Qwen backbone")
    return replacements


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def value_supervision_loss(
    logits: torch.Tensor,
    target_bin: torch.Tensor,
    target_value: torch.Tensor,
    value_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ce_loss = F.cross_entropy(logits, target_bin)
    pred_value = expected_value_from_logits(logits.float())
    value_mse = F.mse_loss(pred_value, target_value.float())
    total_loss = ce_loss + value_loss_weight * value_mse
    return total_loss, ce_loss, value_mse, pred_value


def success_failure_sampler(dataset: QwenValueDataset) -> WeightedRandomSampler:
    labels = [bool(record.get("terminal_is_success")) for record in dataset.records]
    counts = {label: max(1, labels.count(label)) for label in {False, True}}
    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


@torch.no_grad()
def evaluate(
    model: QwenValueModel,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    value_loss_weight: float,
) -> dict[str, float]:
    model.eval()
    total_loss = total_ce_loss = total_value_mse = total_mae = total_correct = total_count = 0.0
    success_values: list[float] = []
    failure_values: list[float] = []
    for batch in loader:
        batch = move_batch(batch, device)
        target_bin = batch.pop("target_bin")
        target_value = batch.pop("target_value")
        terminal_success = batch.pop("terminal_is_success")
        with make_autocast(device, precision):
            logits = model(batch)
            loss, ce_loss, value_mse, pred_value = value_supervision_loss(
                logits,
                target_bin,
                target_value,
                value_loss_weight,
            )
        count = float(target_value.numel())
        total_loss += float(loss.item()) * count
        total_ce_loss += float(ce_loss.item()) * count
        total_value_mse += float(value_mse.item()) * count
        total_mae += float((pred_value - target_value).abs().sum().item())
        total_correct += float((logits.argmax(dim=-1) == target_bin).sum().item())
        total_count += count
        for value, is_success in zip(pred_value.detach().cpu().tolist(), terminal_success.cpu().tolist(), strict=True):
            if bool(is_success):
                success_values.append(float(value))
            else:
                failure_values.append(float(value))

    metrics = {
        "loss": total_loss / max(1.0, total_count),
        "ce_loss": total_ce_loss / max(1.0, total_count),
        "value_mse": total_value_mse / max(1.0, total_count),
        "mae": total_mae / max(1.0, total_count),
        "bin_accuracy": total_correct / max(1.0, total_count),
        "mean_pred_success": float(np.mean(success_values)) if success_values else math.nan,
        "mean_pred_failure": float(np.mean(failure_values)) if failure_values else math.nan,
        "num_examples": total_count,
    }
    metrics["success_failure_gap"] = metrics["mean_pred_success"] - metrics["mean_pred_failure"]
    return metrics


def save_checkpoint(
    model: QwenValueModel,
    optimizer: torch.optim.Optimizer,
    config: QwenValueTrainConfig,
    output_dir: Path,
    step: int,
    metrics: dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trainable_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if "lora_" in name or name.startswith("state_encoder.") or name.startswith("value_head.")
    }
    torch.save(
        {
            "trainable_state": trainable_state,
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(config),
            "lora_module_names": model.lora_module_names,
            "step": step,
            "metrics": metrics,
        },
        output_dir / "qwen_value_checkpoint.pt",
    )
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def train(config: QwenValueTrainConfig) -> dict[str, float]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device_name = config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if config.precision == "bf16":
        dtype = torch.bfloat16
    elif config.precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    processor = AutoProcessor.from_pretrained(config.model_id)
    train_dataset = QwenValueDataset(config.train_jsonl, image_size=config.image_size, state_dim=config.state_dim)
    val_dataset = (
        QwenValueDataset(config.val_jsonl, image_size=config.image_size, state_dim=config.state_dim)
        if config.val_jsonl
        else None
    )
    collator = QwenValueCollator(processor)
    sampler = success_failure_sampler(train_dataset) if config.balance_success_failure else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collator,
        )
        if val_dataset is not None
        else None
    )

    model = QwenValueModel(
        model_id=config.model_id,
        state_dim=config.state_dim,
        hidden_dim=config.hidden_dim,
        num_bins=config.num_bins,
        dtype=dtype,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_last_n_layers=config.lora_last_n_layers,
        lora_target_modules=config.lora_target_modules,
    ).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=config.lr, weight_decay=config.weight_decay)

    run = None
    if config.wandb:
        import wandb

        run = wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            name=config.run_name,
            config=asdict(config) | {"lora_module_names": model.lora_module_names},
        )

    output_dir = Path(config.output_dir)
    step = 0
    latest_metrics: dict[str, float] = {}
    for epoch in range(config.epochs):
        model.train()
        for batch in train_loader:
            batch = move_batch(batch, device)
            target_bin = batch.pop("target_bin")
            target_value = batch.pop("target_value")
            batch.pop("terminal_is_success")
            optimizer.zero_grad(set_to_none=True)
            with make_autocast(device, config.precision):
                logits = model(batch)
                loss, ce_loss, value_mse, pred_value = value_supervision_loss(
                    logits,
                    target_bin,
                    target_value,
                    config.value_loss_weight,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            optimizer.step()
            step += 1

            if step % 10 == 0 or step == 1:
                train_metrics = {
                    "train/loss": float(loss.item()),
                    "train/ce_loss": float(ce_loss.item()),
                    "train/value_mse": float(value_mse.item()),
                    "train/mae": float((pred_value - target_value).abs().mean().item()),
                    "train/bin_accuracy": float((logits.argmax(dim=-1) == target_bin).float().mean().item()),
                    "epoch": epoch,
                    "step": step,
                }
                print(json.dumps(train_metrics, sort_keys=True))
                if run is not None:
                    run.log(train_metrics, step=step)

            if config.max_train_steps is not None and step >= config.max_train_steps:
                break

        if val_loader is not None:
            evaluated_metrics = evaluate(model, val_loader, device, config.precision, config.value_loss_weight)
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
    parser.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Thinking")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--state_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_bins", type=int, default=201)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--value_loss_weight",
        type=float,
        default=1.0,
        help="Weight on MSE between expected distributional value and scalar normalized return.",
    )
    parser.add_argument(
        "--balance_success_failure",
        action="store_true",
        help="Sample successful and failed terminal trajectories with equal probability.",
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_last_n_layers", type=int, default=4)
    parser.add_argument("--lora_target_modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--run_name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = QwenValueTrainConfig(**vars(args))
    metrics = train(config)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
