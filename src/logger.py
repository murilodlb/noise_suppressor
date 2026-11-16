import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

if (
    "matplotlib.pyplot" not in sys.modules
    and os.environ.get("MPLBACKEND") is None
    and not os.environ.get("DISPLAY")
):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

APP_LOGGER_NAME = "noise_suppressor"
DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_PUBLICATION_RC_PARAMS: dict[Any, Any] = {
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _resolve_log_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    normalized = level.strip().upper()
    resolved = logging.getLevelName(normalized)
    if not isinstance(resolved, int):
        raise ValueError(f"Invalid log level '{level}'. Expected DEBUG, INFO, WARNING, or ERROR.")
    return resolved


def configure_logging(
    level: str | int = "INFO",
    log_file: str | Path | None = None,
) -> logging.Logger:
    """
    Configures the root 'noise_suppressor' logger idempotently for console and optional file output.
    """
    numeric_level = _resolve_log_level(level)
    root_logger = logging.getLogger(APP_LOGGER_NAME)
    root_logger.setLevel(numeric_level)

    formatter = logging.Formatter(fmt=DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    has_stream_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root_logger.handlers
    )
    if not has_stream_handler:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    for handler in root_logger.handlers:
        handler.setLevel(numeric_level)

    if log_file is not None:
        resolved_log_path = Path(log_file).resolve()
        resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
        has_file_handler = any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == resolved_log_path
            for h in root_logger.handlers
        )
        if not has_file_handler:
            file_handler = logging.FileHandler(resolved_log_path, encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

    # Keep third-party image/plot internals quiet even when app logging is verbose.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    return root_logger


def get_logger(module_name: str) -> logging.Logger:
    """Returns a namespaced child logger under 'noise_suppressor'."""
    if module_name == APP_LOGGER_NAME or module_name.startswith(f"{APP_LOGGER_NAME}."):
        return logging.getLogger(module_name)
    short_name = module_name.split(".")[-1]
    if short_name in ("__main__", "__init__"):
        return logging.getLogger(APP_LOGGER_NAME)
    return logging.getLogger(f"{APP_LOGGER_NAME}.{short_name}")


def save_training_history(history: dict[str, Any], output_dir: str | Path) -> Path:
    """Serializes per-epoch training history and experiment metadata to training_history.json."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as json_file:
        json.dump(history, json_file, indent=2)
    return history_path


def plot_training_curves(
    history: dict[str, Any],
    output_path: str | Path,
    dpi: int = 300,
) -> Path:
    """
    Generates a publication-ready multi-panel training figure containing:
      1. Training vs. Validation Loss over epochs with best validation epoch highlighted.
      2. Dual-axis Validation MAE and Learning Rate schedule over epochs.
    """
    epochs = list(history.get("epochs", []))
    train_loss = [float(v) for v in history.get("train_loss", [])]
    val_loss = [float(v) for v in history.get("val_loss", [])]
    val_mae = [float(v) for v in history.get("val_mae", [])]
    learning_rate = [float(v) for v in history.get("learning_rate", [])]

    if not epochs:
        max_len = max(len(train_loss), len(val_loss), len(val_mae), len(learning_rate), 1)
        epochs = list(range(1, max_len + 1))

    metadata = history.get("metadata", {})
    model_name = str(metadata.get("model_name", "model"))
    base_channels = metadata.get("base_channels", "")
    loss_name = str(metadata.get("loss_name", "loss"))
    scheduler_type = str(metadata.get("scheduler_type", ""))
    subtitle_parts = [model_name]
    if base_channels != "":
        subtitle_parts.append(f"ch={base_channels}")
    subtitle_parts.append(f"loss={loss_name}")
    if scheduler_type:
        subtitle_parts.append(f"sched={scheduler_type}")
    subtitle = " | ".join(subtitle_parts)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(_PUBLICATION_RC_PARAMS):
        fig, (ax_loss, ax_mae) = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)

        # Panel 1: Loss vs. Epoch
        if train_loss:
            ax_loss.plot(
                epochs[: len(train_loss)],
                train_loss,
                color="#1f77b4",
                linewidth=2.0,
                marker="o" if len(train_loss) <= 25 else None,
                markersize=4.0,
                label="Train Loss",
            )
        if val_loss:
            ax_loss.plot(
                epochs[: len(val_loss)],
                val_loss,
                color="#d62728",
                linewidth=2.0,
                marker="s" if len(val_loss) <= 25 else None,
                markersize=4.0,
                label="Validation Loss",
            )

            finite_pairs = [
                (epochs[idx], val)
                for idx, val in enumerate(val_loss)
                if idx < len(epochs) and math.isfinite(val)
            ]
            if finite_pairs:
                best_epoch_val, best_loss_val = min(finite_pairs, key=lambda item: item[1])
                ax_loss.axvline(
                    best_epoch_val,
                    color="#2ca02c",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.75,
                )
                ax_loss.scatter(
                    [best_epoch_val],
                    [best_loss_val],
                    color="#2ca02c",
                    marker="*",
                    s=120,
                    zorder=5,
                    label=f"Best Val ({best_loss_val:.4f} @ ep {best_epoch_val})",
                )

        ax_loss.set_title("Loss vs. Epoch")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True, linestyle="--", alpha=0.35)
        if len(epochs) == 1:
            ax_loss.set_xlim(epochs[0] - 0.5, epochs[0] + 0.5)
        ax_loss.legend(loc="best", frameon=True)

        # Panel 2: Validation MAE & Learning Rate Schedule (Dual Axis)
        lines = []
        labels = []
        if val_mae:
            (mae_line,) = ax_mae.plot(
                epochs[: len(val_mae)],
                val_mae,
                color="#9467bd",
                linewidth=2.0,
                marker="^" if len(val_mae) <= 25 else None,
                markersize=4.0,
                label="Validation MAE",
            )
            lines.append(mae_line)
            labels.append("Validation MAE")

        ax_mae.set_title("Validation MAE & Learning Rate Schedule")
        ax_mae.set_xlabel("Epoch")
        ax_mae.set_ylabel("Validation MAE (IRM L1 Error)", color="#9467bd")
        ax_mae.tick_params(axis="y", labelcolor="#9467bd")
        ax_mae.grid(True, linestyle="--", alpha=0.35)
        if len(epochs) == 1:
            ax_mae.set_xlim(epochs[0] - 0.5, epochs[0] + 0.5)

        if learning_rate:
            ax_lr = ax_mae.twinx()
            ax_lr.spines["top"].set_visible(False)
            (lr_line,) = ax_lr.plot(
                epochs[: len(learning_rate)],
                learning_rate,
                color="#ff7f0e",
                linestyle="--",
                linewidth=1.8,
                label="Learning Rate",
            )
            ax_lr.set_ylabel("Learning Rate", color="#ff7f0e")
            ax_lr.tick_params(axis="y", labelcolor="#ff7f0e")
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-3, 3))
            ax_lr.yaxis.set_major_formatter(formatter)
            lines.append(lr_line)
            labels.append("Learning Rate")

        if lines:
            ax_mae.legend(lines, labels, loc="best", frameon=True)

        fig.suptitle(f"Training Dynamics ({subtitle})", fontsize=12.5, fontweight="bold")
        fig.savefig(out_file, dpi=dpi)
        plt.close(fig)

    return out_file


def _extract_finite_metric(bucket: dict[str, Any], metric_key: str) -> tuple[float, float]:
    """Safely extracts (mean, std) from a summary systems bucket, substituting NaN for non-finite values."""
    entry = bucket.get(metric_key, {})
    mean_val = float(entry.get("mean", float("nan")))
    std_val = float(entry.get("std", 0.0))
    clean_mean = mean_val if np.isfinite(mean_val) else float("nan")
    clean_std = std_val if np.isfinite(std_val) and std_val >= 0.0 else 0.0
    return clean_mean, clean_std


def plot_evaluation_bars(
    summary: dict[str, Any],
    output_path: str | Path,
    dpi: int = 300,
) -> Path:
    """
    Generates a side-by-side comparison bar chart comparing Noisy Baseline, Trained Model,
    and Oracle IRM Upper Bound across SI-SDR, PESQ, STOI, and Mask F1.
    """
    model_name = str(summary.get("model_name", "Model"))
    base_channels = summary.get("base_channels", "")
    loss_name = str(summary.get("loss_name", ""))
    total_samples = summary.get("total_samples", 0)

    systems = summary.get("systems", {})
    noisy_sys = systems.get("noisy", {})
    model_sys = systems.get("model", {})
    oracle_sys = systems.get("oracle", {})
    mask_metrics = summary.get("mask_metrics", {})

    model_label = f"{model_name}" + (f" (ch={base_channels})" if base_channels != "" else "")

    metric_specs = [
        ("si_sdr", "SI-SDR (dB)", None, "{:.2f}"),
        ("pesq", "PESQ (Wideband)", (0.0, 4.75), "{:.2f}"),
        ("stoi", "STOI", (0.0, 1.12), "{:.3f}"),
        ("mask_f1", "Mask F1-Score", (0.0, 1.12), "{:.3f}"),
    ]

    colors = ["#7f7f7f", "#1f77b4", "#2ca02c"]
    system_labels = ["Noisy (Baseline)", model_label, "Oracle IRM"]

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(_PUBLICATION_RC_PARAMS):
        fig, axes = plt.subplots(1, len(metric_specs), figsize=(15.0, 4.9), constrained_layout=True)

        for ax, (metric_key, title, y_limits, fmt) in zip(axes, metric_specs, strict=True):
            if metric_key == "mask_f1":
                f1_entry = mask_metrics.get("f1", {})
                model_f1 = float(f1_entry.get("mean", float("nan")))
                model_f1_std = float(f1_entry.get("std", 0.0))
                means = [
                    float("nan"),
                    model_f1 if np.isfinite(model_f1) else float("nan"),
                    1.0,
                ]
                stds = [
                    0.0,
                    model_f1_std if np.isfinite(model_f1_std) and model_f1_std >= 0.0 else 0.0,
                    0.0,
                ]
            else:
                n_mean, n_std = _extract_finite_metric(noisy_sys, metric_key)
                m_mean, m_std = _extract_finite_metric(model_sys, metric_key)
                o_mean, o_std = _extract_finite_metric(oracle_sys, metric_key)
                means = [n_mean, m_mean, o_mean]
                stds = [n_std, m_std, o_std]

            x_pos = np.arange(len(system_labels))
            plot_heights = [val if np.isfinite(val) else 0.0 for val in means]
            plot_errs = [
                err if np.isfinite(val) and np.isfinite(err) else 0.0
                for val, err in zip(means, stds, strict=True)
            ]

            bars = ax.bar(
                x_pos,
                plot_heights,
                yerr=plot_errs,
                capsize=4,
                color=colors,
                edgecolor="#222222",
                linewidth=0.7,
                alpha=0.92,
                error_kw={"elinewidth": 1.1, "ecolor": "#333333"},
            )

            finite_tops = [
                h + e if h >= 0 else h - e
                for val, h, e in zip(means, plot_heights, plot_errs, strict=True)
                if np.isfinite(val)
            ]
            span = (max(finite_tops) - min(0.0, min(finite_tops))) if finite_tops else 1.0
            span = max(span, 0.25)

            for bar, raw_val, err_val in zip(bars, means, plot_errs, strict=True):
                if np.isfinite(raw_val):
                    label_text = fmt.format(raw_val)
                    y_anchor = (
                        raw_val + err_val + 0.03 * span
                        if raw_val >= 0
                        else raw_val - err_val - 0.05 * span
                    )
                    va = "bottom" if raw_val >= 0 else "top"
                else:
                    label_text = "N/A"
                    y_anchor = 0.04 * span
                    va = "bottom"
                    bar.set_alpha(0.2)

                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    y_anchor,
                    label_text,
                    ha="center",
                    va=va,
                    fontsize=8.5,
                    fontweight="semibold",
                )

            ax.set_title(title)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(["Noisy", "Model", "Oracle"])
            ax.axhline(0.0, color="#444444", linewidth=0.8)
            ax.grid(axis="y", linestyle="--", alpha=0.35)

            if y_limits is not None:
                ax.set_ylim(*y_limits)
            elif finite_tops:
                low = min(0.0, min(finite_tops) - 0.18 * span)
                high = max(finite_tops) + 0.18 * span
                ax.set_ylim(low, high)

        header = f"Evaluation Comparison — {model_label}"
        if loss_name:
            header += f" | loss={loss_name}"
        if total_samples:
            header += f" | N={total_samples}"
        fig.suptitle(header, fontsize=12.5, fontweight="bold")
        fig.savefig(out_file, dpi=dpi)
        plt.close(fig)

    return out_file
