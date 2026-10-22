"""Spec tests for ExperimentLogger CSV persistence and schema validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from micromagic.evaluation import ExperimentLogger


def test_logger_creates_directory_and_csv(tmp_path: Path) -> None:
    """Verify ExperimentLogger creates the experiment directory and headers."""
    log_dir = tmp_path / "logs"
    logger = ExperimentLogger(log_dir=log_dir, experiment_name="test_run")

    assert logger.exp_dir.exists()
    assert logger.csv_path.exists()

    df = pd.read_csv(logger.csv_path)
    assert list(df.columns) == ["step", "tag", "value", "timestamp"]
    assert len(df) == 0


def test_logger_records_scalars_and_filters(tmp_path: Path) -> None:
    """Verify scalar values are appended to CSV and can be filtered by tag."""
    logger = ExperimentLogger(log_dir=tmp_path, experiment_name="scalars_test")

    logger.log_scalar(tag="loss", value=0.55, step=1)
    logger.log_scalar(tag="loss", value=0.42, step=2)
    logger.log_scalar(tag="win_rate", value=0.75, step=2)

    df_all = logger.read_scalars()
    assert len(df_all) == 3
    assert df_all["step"].tolist() == [1, 2, 2]
    assert df_all["tag"].tolist() == ["loss", "loss", "win_rate"]
    assert pytest.approx(df_all["value"].tolist()) == [0.55, 0.42, 0.75]

    df_loss = logger.read_scalars(tag="loss")
    assert len(df_loss) == 2
    assert df_loss["tag"].tolist() == ["loss", "loss"]
    assert pytest.approx(df_loss["value"].tolist()) == [0.55, 0.42]

    df_win_rate = logger.read_scalars(tag="win_rate")
    assert len(df_win_rate) == 1
    assert pytest.approx(df_win_rate["value"].iloc[0]) == 0.75


def test_logger_flush_and_close_safe(tmp_path: Path) -> None:
    """Verify flush and close do not raise exceptions even if TensorBoard is absent."""
    logger = ExperimentLogger(log_dir=tmp_path, enable_tensorboard=True)
    logger.log_scalar(tag="metric", value=1.0, step=0)
    logger.flush()
    logger.close()
