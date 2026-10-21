"""Experiment logging utilities supporting CSV metrics persistence and optional TensorBoard integration."""

from __future__ import annotations

import csv
from pathlib import Path
import time
from typing import Any

import pandas as pd


class ExperimentLogger:
    """Logs experiment metrics to CSV and optional TensorBoard event logs."""

    def __init__(
        self,
        log_dir: Path | str,
        experiment_name: str | None = None,
        enable_tensorboard: bool = True,
    ) -> None:
        """Initialize ExperimentLogger.

        Args:
            log_dir: Base directory for storing logs.
            experiment_name: Optional subdirectory name for this experiment run.
            enable_tensorboard: Whether to initialize TensorBoard SummaryWriter if available.
        """
        base_path = Path(log_dir)
        self.exp_dir = base_path / experiment_name if experiment_name else base_path
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.exp_dir / "scalars.csv"
        self._init_csv()

        self.tb_writer: Any = None
        if enable_tensorboard:
            self._init_tensorboard()

    def _init_csv(self) -> None:
        """Ensure CSV file exists with header."""
        if not self.csv_path.exists():
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "tag", "value", "timestamp"])

    def _init_tensorboard(self) -> None:
        """Attempt to initialize TensorBoard SummaryWriter without crashing on ImportError."""
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-untyped]

            self.tb_writer = SummaryWriter(log_dir=str(self.exp_dir))
        except (ImportError, ModuleNotFoundError):
            try:
                from tensorboardX import SummaryWriter  # type: ignore[import-untyped]

                self.tb_writer = SummaryWriter(logdir=str(self.exp_dir))
            except (ImportError, ModuleNotFoundError):
                self.tb_writer = None

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar metric at a specific step.

        Args:
            tag: Metric identifier (e.g., 'loss', 'win_rate', 'elo').
            value: Float metric value.
            step: Iteration step or game index.
        """
        timestamp = time.time()
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([step, tag, float(value), timestamp])

        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, float(value), step)

    def read_scalars(self, tag: str | None = None) -> pd.DataFrame:
        """Read logged scalars from CSV as a pandas DataFrame.

        Args:
            tag: Optional tag to filter records.

        Returns:
            DataFrame containing columns: step, tag, value, timestamp.
        """
        if not self.csv_path.exists():
            return pd.DataFrame(columns=["step", "tag", "value", "timestamp"])

        df = pd.read_csv(self.csv_path)
        if tag is not None:
            return df[df["tag"] == tag].reset_index(drop=True)
        return df

    def flush(self) -> None:
        """Flush TensorBoard buffer if available."""
        if self.tb_writer is not None:
            self.tb_writer.flush()

    def close(self) -> None:
        """Close TensorBoard writer if available."""
        if self.tb_writer is not None:
            self.tb_writer.close()
