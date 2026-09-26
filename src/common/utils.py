"""Seeding, logging, checkpoint I/O."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Logger:
    """CSV + optional TensorBoard. One row per log() call.

    Writes are appended incrementally, so a killed job still leaves a usable CSV.
    TB is optional — if the package isn't installed or ``use_tb=False``, the
    logger silently skips scalar-writing without failing.
    """

    def __init__(self, log_dir: str | Path, use_tb: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "metrics.csv"
        self._csv_file = None
        self._csv_writer = None
        self._csv_fields: list[str] | None = None

        self.tb = None
        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(log_dir=str(self.log_dir / "tb"))
            except ImportError:
                self.tb = None

    def log(self, step: int, **scalars: float) -> None:
        row = {"step": step, **{k: float(v) for k, v in scalars.items()}}

        if self._csv_writer is None:
            self._csv_fields = list(row.keys())
            self._csv_file = open(self.csv_path, "a", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_fields)
            if self.csv_path.stat().st_size == 0:
                self._csv_writer.writeheader()
        else:
            # Header is fixed after the first row: missing keys become "" and
            # any keys not in the original header are dropped.
            row = {k: row.get(k, "") for k in self._csv_fields}

        self._csv_writer.writerow(row)
        self._csv_file.flush()

        if self.tb is not None:
            for k, v in scalars.items():
                self.tb.add_scalar(k, v, step)

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
        if self.tb is not None:
            self.tb.close()


def save_ckpt(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    meta: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "meta": meta or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


def load_ckpt(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return payload.get("meta", {})
