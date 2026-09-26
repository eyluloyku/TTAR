"""Stage A training loop.

Usage:
    python -m src.finetune.train --config configs/finetune/da_v2_vits_lora.yaml
    python -m src.finetune.train --config configs/finetune/da_v2_vits_lora.yaml \\
        --set train.lr_head=1e-4 model.encoder=vitb train.epochs=10

Resuming:
    python -m src.finetune.train --config ... --resume outputs/.../checkpoints/last.pt
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.common.data import AugConfig, DepthDataset, make_splits
from src.common.metrics import si_rmse
from src.common.utils import Logger, load_ckpt, save_ckpt, set_seed
from src.finetune.losses import silog_loss
from src.finetune.model import LoRAConfig, build_model, count_trainable


# ---------------------------------------------------------------------------
# Config handling
# ---------------------------------------------------------------------------

def _coerce(s: str):
    """Parse a CLI override value into bool/int/float/str/None."""
    low = s.lower()
    if low in ("true", "false"): return low == "true"
    if low in ("none", "null"):  return None
    try: return int(s)
    except ValueError: pass
    try: return float(s)
    except ValueError: pass
    return s


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for kv in overrides:
        if "=" not in kv:
            raise ValueError(f"override must be key=value, got: {kv!r}")
        key, value = kv.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(value)
    return cfg


def load_config(path: str, overrides: list[str]) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return _apply_overrides(cfg, overrides)


# ---------------------------------------------------------------------------
# Optimiser + scheduler
# ---------------------------------------------------------------------------

def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """Two param groups: LoRA adapters vs DPT head (+ any other trainable params).

    If no LoRA is attached, all trainable params land in the head group and
    the lr_head value is used.
    """
    lora_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (lora_params if "lora_" in n else head_params).append(p)

    groups = []
    if lora_params:
        groups.append({"params": lora_params, "lr": cfg["train"]["lr_lora"]})
    if head_params:
        groups.append({"params": head_params, "lr": cfg["train"]["lr_head"]})
    if not groups:
        raise RuntimeError("No trainable parameters found.")

    return torch.optim.AdamW(
        groups,
        weight_decay=cfg["train"]["weight_decay"],
        betas=(0.9, 0.999),
    )


def cosine_lr(step: int, total_steps: int, warmup_steps: int) -> float:
    """Multiplier in [0, 1]. Linear warmup then cosine decay to 0."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Training + validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, cfg, logger, global_step, total_steps, warmup_steps):
    model.train()
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    lambda_ = cfg["train"]["silog_lambda"]
    grad_clip = cfg["train"].get("grad_clip", 0.0)
    log_every = cfg["logging"].get("log_every", 50)

    running_loss = 0.0
    n_batches = 0
    t0 = time.time()

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        pred = model(images)
        loss = silog_loss(pred, depth, mask, lambda_=lambda_)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]],
                grad_clip,
            )

        # LR schedule applied per-step via a multiplier on the base LR of each group.
        mult = cosine_lr(global_step, total_steps, warmup_steps)
        for g, base in zip(optimizer.param_groups, base_lrs):
            g["lr"] = base * mult

        optimizer.step()

        running_loss += loss.item()
        n_batches += 1
        global_step += 1

        if global_step % log_every == 0:
            logger.log(
                global_step,
                train_loss=running_loss / n_batches,
                lr_mult=mult,
                sec_per_step=(time.time() - t0) / n_batches,
            )

    return running_loss / max(1, n_batches), global_step


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        pred = model(images)
        v = si_rmse(pred, depth, mask)
        if torch.isfinite(v):
            total += v.item() * images.size(0)
            n += images.size(0)
    return total / max(1, n)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], help="dot-path overrides, e.g. train.lr_head=1e-4")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)

    set_seed(cfg["data"]["seed"])
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Data
    train_files, val_files = make_splits(
        cfg["data"]["root"],
        val_frac=cfg["data"]["val_frac"],
        seed=cfg["data"]["seed"],
    )
    max_train = cfg["data"].get("max_train_samples")
    max_val = cfg["data"].get("max_val_samples")
    if max_train: train_files = train_files[:int(max_train)]
    if max_val:   val_files = val_files[:int(max_val)]
    train_ds = DepthDataset(
        train_files, has_depth=True,
        augment=AugConfig(
            crop_size=cfg["data"].get("crop_size"),
            hflip_prob=cfg["data"].get("hflip", 0.5),
            color_jitter=cfg["data"].get("color_jitter", 0.2),
        ),
    )
    val_ds = DepthDataset(val_files, has_depth=True, augment=None)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"].get("val_batch_size", cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )

    # Model
    lora_dict = cfg["model"].get("lora")
    lora_cfg = LoRAConfig(**lora_dict) if lora_dict else None
    model = build_model(
        encoder=cfg["model"]["encoder"],
        checkpoint=cfg["model"].get("checkpoint"),
        lora_cfg=lora_cfg,
    ).to(device)

    tr, tot = count_trainable(model)
    print(f"[model] trainable={tr/1e6:.2f}M / total={tot/1e6:.2f}M ({100*tr/tot:.1f}%)")

    # Optimiser + logger
    optimizer = build_optimizer(model, cfg)
    out_dir = Path(cfg["logging"]["out_dir"])
    ckpt_dir = out_dir / "checkpoints"
    logger = Logger(out_dir, use_tb=cfg["logging"].get("use_tb", True))

    # Scheduler schedule
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg["train"]["epochs"]
    warmup_steps = steps_per_epoch * cfg["train"].get("warmup_epochs", 0)

    # Resume
    start_epoch = 0
    global_step = 0
    best_sirmse = float("inf")
    if args.resume:
        meta = load_ckpt(args.resume, model, optimizer, map_location=device, strict=False)
        start_epoch = meta.get("epoch", 0)
        global_step = meta.get("global_step", 0)
        best_sirmse = meta.get("best_sirmse", float("inf"))
        print(f"[resume] from epoch {start_epoch}, step {global_step}, best_sirmse {best_sirmse:.4f}")

    # Train
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        t0 = time.time()
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, device, cfg, logger,
            global_step, total_steps, warmup_steps,
        )
        val_sirmse = validate(model, val_loader, device)
        elapsed = time.time() - t0

        logger.log(
            global_step,
            epoch=epoch,
            epoch_train_loss=train_loss,
            val_sirmse=val_sirmse,
            epoch_sec=elapsed,
        )
        print(f"[epoch {epoch:03d}] train_loss={train_loss:.4f} val_sirmse={val_sirmse:.4f} ({elapsed:.0f}s)")

        meta = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "val_sirmse": val_sirmse,
            "best_sirmse": min(best_sirmse, val_sirmse),
            "config": cfg,
        }
        save_ckpt(ckpt_dir / "last.pt", model, optimizer, meta=meta)
        if val_sirmse < best_sirmse:
            best_sirmse = val_sirmse
            meta["best_sirmse"] = best_sirmse
            save_ckpt(ckpt_dir / "best.pt", model, optimizer, meta=meta)
            print(f"[epoch {epoch:03d}] new best siRMSE {best_sirmse:.4f}, saved checkpoint")

    logger.close()
    print(f"[done] best val_sirmse: {best_sirmse:.4f}")


if __name__ == "__main__":
    main()
