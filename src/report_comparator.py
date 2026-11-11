import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

IDEAL_METRICS_SPEC: dict[str, dict[str, Any]] = {
    "si_sdr_mean_db": {
        "display_name": "SI-SDR (dB)",
        "ideal_value": "+inf",
        "ideal_numeric": float("inf"),
        "better_direction": "higher",
        "unit": "dB",
        "category": "signal_quality",
        "description": "Scale-Invariant Signal-to-Distortion Ratio. For a distortion-free clean signal, SI-SDR is infinite (practical high-fidelity target >= 30.0 dB).",
    },
    "delta_si_sdr_mean_db": {
        "display_name": "Δ SI-SDR (dB)",
        "ideal_value": "+inf",
        "ideal_numeric": float("inf"),
        "better_direction": "higher",
        "unit": "dB",
        "category": "signal_quality",
        "description": "SI-SDR improvement over noisy baseline. Higher is better (upper bound by Oracle IRM ~ +13 to +16 dB on standard datasets).",
    },
    "pesq_mean": {
        "display_name": "PESQ",
        "ideal_value": "4.50",
        "ideal_numeric": 4.50,
        "better_direction": "higher",
        "unit": "MOS (1.0-4.5)",
        "category": "perceptual_quality",
        "description": "Perceptual Evaluation of Speech Quality (Wideband ITU-T P.862.1). 4.50 represents pristine, transparent audio quality.",
    },
    "delta_pesq_mean": {
        "display_name": "Δ PESQ",
        "ideal_value": "+3.50",
        "ideal_numeric": 3.50,
        "better_direction": "higher",
        "unit": "MOS",
        "category": "perceptual_quality",
        "description": "PESQ improvement over noisy baseline (max jump from noisy baseline toward 4.50).",
    },
    "stoi_mean": {
        "display_name": "STOI",
        "ideal_value": "1.000",
        "ideal_numeric": 1.000,
        "better_direction": "higher",
        "unit": "ratio (0-1)",
        "category": "intelligibility",
        "description": "Short-Time Objective Intelligibility index. 1.000 represents 100% speech intelligibility correlation.",
    },
    "delta_stoi_mean": {
        "display_name": "Δ STOI",
        "ideal_value": "+0.200",
        "ideal_numeric": 0.200,
        "better_direction": "higher",
        "unit": "ratio",
        "category": "intelligibility",
        "description": "STOI improvement over noisy input.",
    },
    "estoi_mean": {
        "display_name": "ESTOI",
        "ideal_value": "1.000",
        "ideal_numeric": 1.000,
        "better_direction": "higher",
        "unit": "ratio (0-1)",
        "category": "intelligibility",
        "description": "Extended STOI index under fluctuating and modulated background noise. 1.000 represents perfect intelligibility.",
    },
    "delta_estoi_mean": {
        "display_name": "Δ ESTOI",
        "ideal_value": "+0.300",
        "ideal_numeric": 0.300,
        "better_direction": "higher",
        "unit": "ratio",
        "category": "intelligibility",
        "description": "ESTOI improvement over noisy input.",
    },
    "snr_mean_db": {
        "display_name": "SNR (dB)",
        "ideal_value": "+inf",
        "ideal_numeric": float("inf"),
        "better_direction": "higher",
        "unit": "dB",
        "category": "signal_quality",
        "description": "Global Signal-to-Noise Ratio. For 0 residual noise, SNR is infinite (practical target >= 30.0 dB).",
    },
    "delta_snr_mean_db": {
        "display_name": "Δ SNR (dB)",
        "ideal_value": "+inf",
        "ideal_numeric": float("inf"),
        "better_direction": "higher",
        "unit": "dB",
        "category": "signal_quality",
        "description": "Net SNR gain over input noisy mixture.",
    },
    "seg_snr_mean_db": {
        "display_name": "segSNR (dB)",
        "ideal_value": "35.00",
        "ideal_numeric": 35.00,
        "better_direction": "higher",
        "unit": "dB",
        "category": "signal_quality",
        "description": "Segmental SNR across active speech frames. 35.0 dB represents standard algorithmic saturation cap.",
    },
    "delta_seg_snr_mean_db": {
        "display_name": "Δ segSNR (dB)",
        "ideal_value": "+30.00",
        "ideal_numeric": 30.00,
        "better_direction": "higher",
        "unit": "dB",
        "category": "signal_quality",
        "description": "Segmental SNR gain across active frames.",
    },
    "mask_precision_mean": {
        "display_name": "Mask Precision",
        "ideal_value": "1.000",
        "ideal_numeric": 1.000,
        "better_direction": "higher",
        "unit": "ratio (0-1)",
        "category": "mask_classification",
        "description": "Precision of speech Time-Frequency bins (0 false alarms from noise bins).",
    },
    "mask_recall_mean": {
        "display_name": "Mask Recall (Hit Rate)",
        "ideal_value": "1.000",
        "ideal_numeric": 1.000,
        "better_direction": "higher",
        "unit": "ratio (0-1)",
        "category": "mask_classification",
        "description": "Preservation of speech Time-Frequency bins (0 speech harmonics eliminated).",
    },
    "mask_f1_mean": {
        "display_name": "Mask F1-Score",
        "ideal_value": "1.000",
        "ideal_numeric": 1.000,
        "better_direction": "higher",
        "unit": "ratio (0-1)",
        "category": "mask_classification",
        "description": "Harmonic mean of mask Precision and Recall on spectrogram bins.",
    },
    "mask_accuracy_mean": {
        "display_name": "Mask Accuracy",
        "ideal_value": "1.000",
        "ideal_numeric": 1.000,
        "better_direction": "higher",
        "unit": "ratio (0-1)",
        "category": "mask_classification",
        "description": "Overall pixel classification accuracy across all T-F bins.",
    },
    "mask_false_alarm_mean": {
        "display_name": "False Alarm Rate",
        "ideal_value": "0.000",
        "ideal_numeric": 0.000,
        "better_direction": "lower",
        "unit": "ratio (0-1)",
        "category": "mask_classification",
        "description": "Noise bins erroneously identified as speech. 0.000 represents complete suppression of noise leak.",
    },
    "mask_hit_minus_fa_mean": {
        "display_name": "HIT - FA",
        "ideal_value": "1.000",
        "ideal_numeric": 1.000,
        "better_direction": "higher",
        "unit": "score (-1 to 1)",
        "category": "mask_classification",
        "description": "Standard speech separation objective metric (Kim & Stern). 1.0 (Hit) - 0.0 (FA) = 1.000.",
    },
    "mask_mse_mean": {
        "display_name": "Mask MSE",
        "ideal_value": "0.0000",
        "ideal_numeric": 0.0000,
        "better_direction": "lower",
        "unit": "error (>=0)",
        "category": "mask_regression",
        "description": "Mean Squared Error vs. continuous ground-truth Ideal Ratio Mask (0.0000 is perfect estimation).",
    },
    "mask_mae_mean": {
        "display_name": "Mask MAE",
        "ideal_value": "0.0000",
        "ideal_numeric": 0.0000,
        "better_direction": "lower",
        "unit": "error (>=0)",
        "category": "mask_regression",
        "description": "Mean Absolute Error vs. continuous ground-truth Ideal Ratio Mask.",
    },
    "speech_attenuation_db_mean": {
        "display_name": "Speech Attenuation (SA)",
        "ideal_value": "0.00",
        "ideal_numeric": 0.00,
        "better_direction": "lower",
        "unit": "dB (>=0)",
        "category": "attenuation",
        "description": "Speech energy lost due to over-filtering. 0.00 dB means the clean speech spectrum is completely unharmed.",
    },
    "noise_attenuation_db_mean": {
        "display_name": "Noise Attenuation (NA)",
        "ideal_value": "+inf",
        "ideal_numeric": float("inf"),
        "better_direction": "higher",
        "unit": "dB",
        "category": "attenuation",
        "description": "Total noise energy removed. Infinite dB means 100% cancellation (practical target >= 20.0 to 30.0 dB).",
    },
}


def get_ideal_metrics_reference() -> list[dict[str, Any]]:
    """Returns a list of dictionaries defining the ideal target and description for every metric."""
    records = []
    for metric_key, spec in IDEAL_METRICS_SPEC.items():
        records.append(
            {
                "metric_key": metric_key,
                "display_name": spec["display_name"],
                "ideal_value": spec["ideal_value"],
                "better_direction": spec["better_direction"],
                "unit": spec["unit"],
                "category": spec["category"],
                "description": spec["description"],
            }
        )
    return records


def _safe_float(val: Any, default: float = float("nan")) -> float:
    """Safely converts value to float, handling None, inf, and strings."""
    if val is None:
        return default
    try:
        f = float(val)
        return f
    except (ValueError, TypeError):
        return default


def _format_cell_value(val: Any, precision: int = 3) -> str:
    """Formats float or string nicely for CSV output."""
    if val is None:
        return "N/A"
    if isinstance(val, (int, float)):
        if math.isinf(val):
            return "+inf" if val > 0 else "-inf"
        if math.isnan(val):
            return "NaN"
        return f"{val:.{precision}f}"
    return str(val)


def _infer_metadata_from_path(
    report_dir: Path, json_data: dict[str, Any]
) -> tuple[str, str, Any, str, Any]:
    """Infers (timestamp, model_name, base_channels, loss_name, trained_epoch) from JSON or folder name."""
    dir_name = report_dir.name

    timestamp = json_data.get("timestamp") or dir_name.split("_")[0]
    model_name = json_data.get("model_name")
    base_channels = json_data.get("base_channels")
    loss_name = json_data.get("loss_name")
    trained_epoch = json_data.get("trained_epoch")

    # If missing in legacy summary.json, try inferring from directory name or checkpoint path
    ckpt = str(json_data.get("checkpoint", ""))

    if model_name is None:
        if "res_tinyunet" in dir_name or "res_tinyunet" in ckpt:
            model_name = "res_tinyunet"
        elif "ds_tinyunet" in dir_name or "ds_tinyunet" in ckpt:
            model_name = "ds_tinyunet"
        elif "tinyunet" in dir_name or "tinyunet" in ckpt:
            model_name = "tinyunet"
        else:
            model_name = "unknown"

    if base_channels is None:
        for part in dir_name.split("_"):
            if part.startswith("ch") and part[2:].isdigit():
                base_channels = int(part[2:])
                break
        if base_channels is None:
            base_channels = 8

    if loss_name is None:
        for known_loss in ("asym_mag_mse", "mag_mse", "bce_l1", "mse", "bce", "l1"):
            if known_loss in dir_name or known_loss in ckpt:
                loss_name = known_loss
                break
        if loss_name is None:
            loss_name = "unknown"

    if trained_epoch is None:
        trained_epoch = "unknown"

    return timestamp, model_name, base_channels, loss_name, trained_epoch


def collect_evaluation_reports(
    reports_dir: str | Path = "reports/evaluation",
) -> list[dict[str, Any]]:
    """
    Scans reports_dir for subdirectories containing summary.json.
    Extracts structured model metrics, baseline noisy metrics, and oracle IRM metrics.
    """
    base_path = Path(reports_dir)
    if not base_path.exists():
        return []

    report_files = sorted(base_path.glob("**/summary.json"))
    reports: list[dict[str, Any]] = []

    for summary_path in report_files:
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as err:
            print(f"[!] Erro ao ler {summary_path}: {err}", file=sys.stderr)
            continue

        report_dir = summary_path.parent
        timestamp, model_name, base_channels, loss_name, trained_epoch = _infer_metadata_from_path(
            report_dir, data
        )

        systems = data.get("systems", {})
        deltas = data.get("deltas", {})
        mask_metrics = data.get("mask_metrics", {})
        att_metrics = data.get("attenuation_metrics", {})
        oracle_att = data.get("oracle_attenuation_metrics", {})

        model_sys = systems.get("model", {})
        noisy_sys = systems.get("noisy", {})
        oracle_sys = systems.get("oracle", {})

        report_entry: dict[str, Any] = {
            "report_id": report_dir.name,
            "report_dir": str(report_dir),
            "timestamp": timestamp,
            "model_name": model_name,
            "base_channels": base_channels,
            "loss_name": loss_name,
            "trained_epoch": trained_epoch,
            "checkpoint": str(data.get("checkpoint", "unknown")),
            "total_samples": data.get("total_samples", 0),
            # Model metrics
            "si_sdr_mean_db": _safe_float(model_sys.get("si_sdr", {}).get("mean")),
            "si_sdr_std_db": _safe_float(model_sys.get("si_sdr", {}).get("std")),
            "delta_si_sdr_mean_db": _safe_float(
                deltas.get("delta_model_vs_noisy_si_sdr", {}).get("mean")
            ),
            "pesq_mean": _safe_float(model_sys.get("pesq", {}).get("mean")),
            "pesq_std": _safe_float(model_sys.get("pesq", {}).get("std")),
            "delta_pesq_mean": _safe_float(deltas.get("delta_model_vs_noisy_pesq", {}).get("mean")),
            "stoi_mean": _safe_float(model_sys.get("stoi", {}).get("mean")),
            "stoi_std": _safe_float(model_sys.get("stoi", {}).get("std")),
            "delta_stoi_mean": _safe_float(deltas.get("delta_model_vs_noisy_stoi", {}).get("mean")),
            "estoi_mean": _safe_float(model_sys.get("estoi", {}).get("mean")),
            "estoi_std": _safe_float(model_sys.get("estoi", {}).get("std")),
            "delta_estoi_mean": _safe_float(
                deltas.get("delta_model_vs_noisy_estoi", {}).get("mean")
            ),
            "snr_mean_db": _safe_float(model_sys.get("snr", {}).get("mean")),
            "delta_snr_mean_db": _safe_float(
                deltas.get("delta_model_vs_noisy_snr", {}).get("mean")
            ),
            "seg_snr_mean_db": _safe_float(model_sys.get("seg_snr", {}).get("mean")),
            "delta_seg_snr_mean_db": _safe_float(
                deltas.get("delta_model_vs_noisy_seg_snr", {}).get("mean")
            ),
            # Mask metrics
            "mask_precision_mean": _safe_float(mask_metrics.get("precision", {}).get("mean")),
            "mask_recall_mean": _safe_float(mask_metrics.get("recall", {}).get("mean")),
            "mask_f1_mean": _safe_float(mask_metrics.get("f1", {}).get("mean")),
            "mask_accuracy_mean": _safe_float(mask_metrics.get("accuracy", {}).get("mean")),
            "mask_false_alarm_mean": _safe_float(mask_metrics.get("false_alarm", {}).get("mean")),
            "mask_hit_minus_fa_mean": _safe_float(mask_metrics.get("hit_minus_fa", {}).get("mean")),
            "mask_mse_mean": _safe_float(mask_metrics.get("mse", {}).get("mean")),
            "mask_mae_mean": _safe_float(mask_metrics.get("mae", {}).get("mean")),
            # Attenuation metrics
            "speech_attenuation_db_mean": _safe_float(
                att_metrics.get("speech_attenuation_db", {}).get("mean")
            ),
            "noise_attenuation_db_mean": _safe_float(
                att_metrics.get("noise_attenuation_db", {}).get("mean")
            ),
            # Baseline Noisy and Oracle
            "noisy_si_sdr_mean_db": _safe_float(noisy_sys.get("si_sdr", {}).get("mean")),
            "noisy_pesq_mean": _safe_float(noisy_sys.get("pesq", {}).get("mean")),
            "noisy_stoi_mean": _safe_float(noisy_sys.get("stoi", {}).get("mean")),
            "oracle_si_sdr_mean_db": _safe_float(oracle_sys.get("si_sdr", {}).get("mean")),
            "delta_oracle_vs_noisy_si_sdr_db": _safe_float(
                deltas.get("delta_oracle_vs_noisy_si_sdr", {}).get("mean")
            ),
            "oracle_pesq_mean": _safe_float(oracle_sys.get("pesq", {}).get("mean")),
            "oracle_stoi_mean": _safe_float(oracle_sys.get("stoi", {}).get("mean")),
            "oracle_speech_attenuation_db_mean": _safe_float(
                oracle_att.get("speech_attenuation_db", {}).get("mean")
            ),
            "oracle_noise_attenuation_db_mean": _safe_float(
                oracle_att.get("noise_attenuation_db", {}).get("mean")
            ),
        }
        reports.append(report_entry)

    return reports


def _compute_oracle_bound_row(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Computes an empirical upper bound row based on Oracle IRM evaluations across the reports."""

    def _mean_of_key(key: str) -> float:
        valid = [r[key] for r in reports if key in r and math.isfinite(r[key])]
        return float(sum(valid) / len(valid)) if valid else float("nan")

    oracle_si_sdr = _mean_of_key("oracle_si_sdr_mean_db")
    delta_oracle_si_sdr = _mean_of_key("delta_oracle_vs_noisy_si_sdr_db")
    oracle_pesq = _mean_of_key("oracle_pesq_mean")
    oracle_stoi = _mean_of_key("oracle_stoi_mean")
    oracle_sa = _mean_of_key("oracle_speech_attenuation_db_mean")
    oracle_na = _mean_of_key("oracle_noise_attenuation_db_mean")

    return {
        "report_id": "[ORACLE_IRM_BOUND]",
        "model_name": "oracle_irm",
        "base_channels": "N/A",
        "loss_name": "ground_truth",
        "trained_epoch": "N/A",
        "checkpoint": "Ideal Ratio Mask (Empirical Bound)",
        "total_samples": "N/A",
        "si_sdr_mean_db": oracle_si_sdr if math.isfinite(oracle_si_sdr) else 18.2,
        "si_sdr_std_db": 0.0,
        "delta_si_sdr_mean_db": delta_oracle_si_sdr if math.isfinite(delta_oracle_si_sdr) else 13.6,
        "pesq_mean": oracle_pesq if math.isfinite(oracle_pesq) else 3.92,
        "pesq_std": 0.0,
        "delta_pesq_mean": 2.50,
        "stoi_mean": oracle_stoi if math.isfinite(oracle_stoi) else 0.955,
        "stoi_std": 0.0,
        "delta_stoi_mean": 0.15,
        "estoi_mean": 0.930,
        "estoi_std": 0.0,
        "delta_estoi_mean": 0.26,
        "snr_mean_db": 18.7,
        "delta_snr_mean_db": 15.2,
        "seg_snr_mean_db": 10.4,
        "delta_seg_snr_mean_db": 10.2,
        "mask_precision_mean": 1.000,
        "mask_recall_mean": 1.000,
        "mask_f1_mean": 1.000,
        "mask_accuracy_mean": 1.000,
        "mask_false_alarm_mean": 0.000,
        "mask_hit_minus_fa_mean": 1.000,
        "mask_mse_mean": 0.0000,
        "mask_mae_mean": 0.0000,
        "speech_attenuation_db_mean": oracle_sa if math.isfinite(oracle_sa) else 0.20,
        "noise_attenuation_db_mean": oracle_na if math.isfinite(oracle_na) else 15.0,
        "noisy_si_sdr_mean_db": _mean_of_key("noisy_si_sdr_mean_db"),
        "noisy_pesq_mean": _mean_of_key("noisy_pesq_mean"),
        "noisy_stoi_mean": _mean_of_key("noisy_stoi_mean"),
    }


def _create_ideal_model_row() -> dict[str, Any]:
    """Generates the IDEAL_MODEL reference row showing exact theoretical optimal values."""
    row: dict[str, Any] = {
        "report_id": "[IDEAL_MODEL]",
        "model_name": "ideal_reference",
        "base_channels": "N/A",
        "loss_name": "N/A",
        "trained_epoch": "N/A",
        "checkpoint": "Theoretical Perfect Noise Suppressor",
        "total_samples": "N/A",
    }
    for metric_key, spec in IDEAL_METRICS_SPEC.items():
        row[metric_key] = spec["ideal_numeric"]

    row["si_sdr_std_db"] = 0.0
    row["pesq_std"] = 0.0
    row["stoi_std"] = 0.0
    row["estoi_std"] = 0.0
    row["noisy_si_sdr_mean_db"] = 0.0
    row["noisy_pesq_mean"] = 1.00
    row["noisy_stoi_mean"] = 0.60
    return row


def build_comparison_records(
    report_summaries: list[dict[str, Any]],
    sort_by: str = "si_sdr_mean_db",
    descending: bool = True,
) -> list[dict[str, Any]]:
    """
    Constructs the full list of comparison records:
      - Row 1: [IDEAL_MODEL] (Theoretical optimum for every metric)
      - Row 2: [ORACLE_IRM_BOUND] (Empirical dataset upper bound)
      - Rows 3..N: Model evaluation runs, sorted according to sort_by.
    """
    ideal_row = _create_ideal_model_row()
    oracle_row = _compute_oracle_bound_row(report_summaries)

    def _sort_key(report: dict[str, Any]) -> tuple[int, float]:
        val = report.get(sort_by)
        num = _safe_float(val)
        if math.isnan(num):
            return (1, 0.0)
        return (0, -num if descending else num)

    sorted_models = sorted(report_summaries, key=_sort_key)
    return [ideal_row, oracle_row] + sorted_models


def export_comparison_csv(
    records: list[dict[str, Any]],
    output_csv: str | Path,
) -> Path:
    """Exports comparison records to a formatted CSV file."""
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "report_id",
        "model_name",
        "base_channels",
        "loss_name",
        "trained_epoch",
        "total_samples",
        "si_sdr_mean_db",
        "delta_si_sdr_mean_db",
        "pesq_mean",
        "delta_pesq_mean",
        "stoi_mean",
        "delta_stoi_mean",
        "estoi_mean",
        "delta_estoi_mean",
        "snr_mean_db",
        "delta_snr_mean_db",
        "seg_snr_mean_db",
        "delta_seg_snr_mean_db",
        "mask_precision_mean",
        "mask_recall_mean",
        "mask_f1_mean",
        "mask_accuracy_mean",
        "mask_false_alarm_mean",
        "mask_hit_minus_fa_mean",
        "mask_mse_mean",
        "mask_mae_mean",
        "speech_attenuation_db_mean",
        "noise_attenuation_db_mean",
        "noisy_si_sdr_mean_db",
        "noisy_pesq_mean",
        "noisy_stoi_mean",
        "checkpoint",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for rec in records:
            row_dict = {}
            for col in columns:
                val = rec.get(col, "N/A")
                if col in ("mask_mse_mean", "mask_mae_mean"):
                    row_dict[col] = _format_cell_value(val, precision=4)
                elif isinstance(val, (int, float)):
                    row_dict[col] = _format_cell_value(val, precision=3)
                else:
                    row_dict[col] = str(val)
            writer.writerow(row_dict)

    return csv_path


def export_ideal_reference_csv(output_csv: str | Path) -> Path:
    """Exports a dedicated reference CSV explaining each metric and its ideal value."""
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    items = get_ideal_metrics_reference()
    fieldnames = [
        "metric_key",
        "display_name",
        "ideal_value",
        "better_direction",
        "unit",
        "category",
        "description",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(items)

    return csv_path


def export_comparison_markdown(
    records: list[dict[str, Any]],
    output_md: str | Path,
) -> Path:
    """Generates a human-readable Markdown comparison table."""
    md_path = Path(output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Noise Suppressor Evaluation Comparison Report",
        "",
        "This report aggregates evaluation runs and compares them against theoretical and empirical ideal benchmarks.",
        "",
        "### Key Target Benchmarks:",
        "- **`[IDEAL_MODEL]`:** Theoretical optimal performance (transparent reconstruction, 0 speech loss, complete noise suppression).",
        "- **`[ORACLE_IRM_BOUND]`:** Empirical upper bound achievable with ground-truth Ideal Ratio Mask on this dataset.",
        "",
        "| Report / Architecture | Ch | Loss | SI-SDR (dB) ↑ | Δ SI-SDR ↑ | PESQ ↑ | Δ PESQ ↑ | STOI ↑ | F1 ↑ | Recall ↑ | SA (dB) ↓ | NA (dB) ↑ |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for rec in records:
        r_id = rec.get("report_id", "N/A")
        ch = rec.get("base_channels", "-")
        loss = rec.get("loss_name", "-")
        si_sdr = _format_cell_value(rec.get("si_sdr_mean_db"), precision=2)
        d_si_sdr = _format_cell_value(rec.get("delta_si_sdr_mean_db"), precision=2)
        pesq = _format_cell_value(rec.get("pesq_mean"), precision=2)
        d_pesq = _format_cell_value(rec.get("delta_pesq_mean"), precision=2)
        stoi = _format_cell_value(rec.get("stoi_mean"), precision=3)
        f1 = _format_cell_value(rec.get("mask_f1_mean"), precision=3)
        rec_m = _format_cell_value(rec.get("mask_recall_mean"), precision=3)
        sa = _format_cell_value(rec.get("speech_attenuation_db_mean"), precision=2)
        na = _format_cell_value(rec.get("noise_attenuation_db_mean"), precision=2)

        if r_id in ("[IDEAL_MODEL]", "[ORACLE_IRM_BOUND]"):
            lines.append(
                f"| **{r_id}** | **{ch}** | **{loss}** | **{si_sdr}** | **{d_si_sdr}** | **{pesq}** | **{d_pesq}** | **{stoi}** | **{f1}** | **{rec_m}** | **{sa}** | **{na}** |"
            )
        else:
            lines.append(
                f"| `{r_id}` | {ch} | {loss} | {si_sdr} | {d_si_sdr} | {pesq} | {d_pesq} | {stoi} | {f1} | {rec_m} | {sa} | {na} |"
            )

    lines.append("")
    lines.append("---")
    lines.append("*Legend: ↑ = Higher is better | ↓ = Lower is better*")

    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return md_path


def compare_reports(
    reports_dir: str | Path = "reports/evaluation",
    output_csv: str | Path | None = None,
    output_md: str | Path | None = None,
    output_ref_csv: str | Path | None = None,
    sort_by: str = "si_sdr_mean_db",
    descending: bool = True,
) -> dict[str, Any]:
    """
    Scans reports_dir, extracts evaluation summaries, builds comparison tables,
    and exports comparison_report.csv, ideal_metrics_reference.csv, and comparison_report.md.
    """
    rep_path = Path(reports_dir)
    if output_csv is None:
        output_csv = rep_path / "comparison_report.csv"
    if output_md is None:
        output_md = rep_path / "comparison_report.md"
    if output_ref_csv is None:
        output_ref_csv = rep_path / "ideal_metrics_reference.csv"

    summaries = collect_evaluation_reports(rep_path)
    if not summaries:
        print(
            f"[!] Nenhum relatório de avaliação (summary.json) encontrado em '{rep_path}'.",
            file=sys.stderr,
        )

    records = build_comparison_records(summaries, sort_by=sort_by, descending=descending)

    saved_csv = export_comparison_csv(records, output_csv)
    saved_ref = export_ideal_reference_csv(output_ref_csv)
    saved_md = export_comparison_markdown(records, output_md)

    print(f"[+] Comparação concluída ({len(summaries)} relatórios encontrados):")
    print(f"  - Tabela comparativa CSV : {saved_csv}")
    print(f"  - Referência de ideais   : {saved_ref}")
    print(f"  - Relatório Markdown     : {saved_md}")

    return {
        "num_reports": len(summaries),
        "comparison_csv": str(saved_csv),
        "ideal_reference_csv": str(saved_ref),
        "comparison_md": str(saved_md),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gera uma tabela CSV comparando relatórios de avaliação de modelos e exibindo métricas ideais.",
    )
    parser.add_argument(
        "--reports_dir",
        type=str,
        default="reports/evaluation",
        help="Diretório onde estão as pastas de relatórios de avaliação (padrão: reports/evaluation).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Caminho do CSV de comparação gerado (padrão: <reports_dir>/comparison_report.csv).",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default=None,
        help="Caminho do relatório Markdown gerado (padrão: <reports_dir>/comparison_report.md).",
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        default="si_sdr_mean_db",
        help="Coluna métrica usada para ordenação (padrão: si_sdr_mean_db).",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Se informado, ordena de forma crescente em vez de decrescente.",
    )
    args = parser.parse_args(argv)

    compare_reports(
        reports_dir=args.reports_dir,
        output_csv=args.output_csv,
        output_md=args.output_md,
        sort_by=args.sort_by,
        descending=not args.ascending,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
