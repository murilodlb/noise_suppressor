import json
import logging

from src.logger import (
    APP_LOGGER_NAME,
    configure_logging,
    get_logger,
    plot_evaluation_bars,
    plot_training_curves,
    save_training_history,
)


def test_configure_logging_is_idempotent_and_emits_structured_records(caplog, tmp_path):
    log_file = tmp_path / "run.log"
    logger_first = configure_logging(level="INFO", log_file=log_file)
    handler_count_first = len(logger_first.handlers)

    logger_second = configure_logging(level="DEBUG", log_file=log_file)
    assert logger_first is logger_second
    assert logger_first.name == APP_LOGGER_NAME
    assert len(logger_second.handlers) == handler_count_first

    child_logger = get_logger("src.train")
    assert child_logger.name == f"{APP_LOGGER_NAME}.train"

    with caplog.at_level(logging.INFO, logger=APP_LOGGER_NAME):
        child_logger.info("Structured training test message")
        child_logger.warning("Structured warning message")

    assert any("Structured training test message" in rec.message for rec in caplog.records)
    assert any(rec.levelname == "WARNING" for rec in caplog.records)
    for handler in list(logger_second.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger_second.removeHandler(handler)
    assert log_file.exists()


def test_save_and_plot_training_curves(tmp_path):
    history = {
        "metadata": {
            "model_name": "res_tinyunet",
            "base_channels": 16,
            "loss_name": "asym_mag_mse",
            "scheduler_type": "cosine",
        },
        "epochs": [1, 2, 3, 4],
        "train_loss": [0.28, 0.21, 0.16, 0.13],
        "val_loss": [0.25, 0.19, 0.15, 0.14],
        "val_mae": [0.22, 0.18, 0.16, 0.15],
        "learning_rate": [1e-3, 9e-4, 6e-4, 2e-4],
        "best_epoch": 4,
        "best_val_loss": 0.14,
    }

    history_path = save_training_history(history, tmp_path / "training_report")
    assert history_path.exists()
    loaded = json.loads(history_path.read_text(encoding="utf-8"))
    assert loaded["epochs"] == [1, 2, 3, 4]
    assert loaded["best_val_loss"] == 0.14

    curves_path = plot_training_curves(
        history, tmp_path / "training_report" / "training_curves.png"
    )
    assert curves_path.exists()
    assert curves_path.stat().st_size > 1024

    # Single-epoch edge case
    single_epoch_history = {
        "metadata": {"model_name": "tinyunet", "loss_name": "bce"},
        "epochs": [1],
        "train_loss": [0.4],
        "val_loss": [0.35],
        "val_mae": [0.29],
        "learning_rate": [1e-3],
    }
    single_curves = plot_training_curves(single_epoch_history, tmp_path / "single.png")
    assert single_curves.exists()
    assert single_curves.stat().st_size > 1024


def test_plot_evaluation_bars_handles_normal_and_non_finite_metrics(tmp_path):
    summary = {
        "model_name": "res_tinyunet",
        "base_channels": 16,
        "loss_name": "asym_mag_mse",
        "total_samples": 100,
        "systems": {
            "noisy": {
                "si_sdr": {"mean": 4.82, "std": 5.40},
                "pesq": {"mean": 1.38, "std": 0.35},
                "stoi": {"mean": 0.780, "std": 0.110},
            },
            "model": {
                "si_sdr": {"mean": 14.10, "std": 5.10},
                "pesq": {"mean": 2.18, "std": 0.45},
                "stoi": {"mean": 0.885, "std": 0.080},
            },
            "oracle": {
                "si_sdr": {"mean": 16.79, "std": 4.90},
                "pesq": {"mean": 3.85, "std": 0.40},
                "stoi": {"mean": 0.945, "std": 0.045},
            },
        },
        "mask_metrics": {
            "f1": {"mean": 0.720, "std": 0.120},
        },
    }

    bars_path = plot_evaluation_bars(summary, tmp_path / "evaluation_bars.png")
    assert bars_path.exists()
    assert bars_path.stat().st_size > 1024

    # Non-finite edge case (e.g., NaN / Inf metrics)
    nan_summary = {
        "model_name": "tinyunet",
        "base_channels": 8,
        "loss_name": "bce",
        "total_samples": 2,
        "systems": {
            "noisy": {
                "si_sdr": {"mean": float("nan"), "std": float("nan")},
                "pesq": {"mean": float("nan"), "std": 0.0},
                "stoi": {"mean": float("inf"), "std": 0.0},
            },
            "model": {
                "si_sdr": {"mean": -2.5, "std": 1.2},
                "pesq": {"mean": float("nan"), "std": 0.0},
                "stoi": {"mean": 0.65, "std": 0.05},
            },
            "oracle": {
                "si_sdr": {"mean": 12.0, "std": 2.0},
                "pesq": {"mean": float("nan"), "std": 0.0},
                "stoi": {"mean": 0.91, "std": 0.03},
            },
        },
        "mask_metrics": {
            "f1": {"mean": float("nan"), "std": float("nan")},
        },
    }
    nan_bars_path = plot_evaluation_bars(nan_summary, tmp_path / "nan_bars.png")
    assert nan_bars_path.exists()
    assert nan_bars_path.stat().st_size > 1024
