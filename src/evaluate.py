from collections import defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path
import random
import time
from typing import Dict, Any, List, Optional

import numpy as np
import torch

from .dsp_utils import compute_stft, compute_istft, compute_irm, mix_at_snr
from .inference import load_model, enhance_signal, load_audio_file
from .metrics import (
    compute_snr,
    compute_signal_metrics,
    compute_mask_threshold,
    compute_mask_classification_metrics,
    compute_mask_regression_metrics,
    compute_attenuation_metrics,
)
from .train import _split_indices_by_clean_source


FALLBACK_SNR_RANGE = (-5.0, 15.0)


def _sample_fallback_snr(seed: int, row_index: int) -> float:
    """SNR determinística por amostra quando metadata.csv não traz snr_db."""
    return random.Random(seed + row_index).uniform(*FALLBACK_SNR_RANGE)


def _get_snr_bucket(snr_val: float) -> str:
    if snr_val < 0.0:
        return "< 0 dB"
    elif snr_val < 5.0:
        return "0 - 5 dB"
    elif snr_val < 10.0:
        return "5 - 10 dB"
    else:
        return ">= 10 dB"


def _fix_audio_length(audio: np.ndarray, target_samples: int) -> np.ndarray:
    if len(audio) > target_samples:
        return audio[:target_samples]
    elif len(audio) < target_samples:
        return np.pad(audio, (0, target_samples - len(audio)), mode="constant")
    return audio


def evaluate(
    checkpoint_path: Optional[str | Path] = "checkpoints/tinyunet.pth",
    data_root: str | Path = "data",
    output_dir: str | Path = "reports/evaluation",
    validation_ratio: float = 0.2,
    seed: int = 42,
    max_samples: Optional[int] = None,
    lc_db: float = 0.0,
    device: str = "cpu",
    target_seconds: float = 2.0,
) -> Dict[str, Any]:
    """
    Evaluates the noise suppression model on the held-out validation split.
    Computes signal quality metrics (SI-SDR, SNR, segSNR, PESQ, STOI, ESTOI),
    mask metrics (Precision, Recall, F1, Accuracy, HIT-FA, MSE, MAE),
    and speech/noise attenuation across 3 systems (noisy, model, oracle_irm).
    """
    root_path = Path(data_root)
    torch_device = torch.device(device)
    model, dsp_config = load_model(checkpoint_path, device=torch_device)
    model.eval()

    sr = dsp_config.get("sr", 16000)
    n_fft = dsp_config.get("n_fft", 512)
    hop_length = dsp_config.get("hop_length", 256)
    target_samples = int(sr * target_seconds)
    threshold = compute_mask_threshold(lc_db=lc_db, beta=0.5)

    # 1. Determine validation items
    metadata_path = root_path / "precomputed" / "metadata.csv"
    if not metadata_path.exists():
        metadata_path = root_path / "generated" / "metadata.csv"

    eval_pairs: List[Dict[str, Any]] = []

    if metadata_path.exists():
        _, val_indices = _split_indices_by_clean_source(
            metadata_path=metadata_path,
            validation_ratio=validation_ratio,
            seed=seed,
        )
        val_indices_set = set(val_indices)
        with metadata_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if idx in val_indices_set:
                    raw_snr = (row.get("snr_db") or "").strip()
                    snr_db = float(raw_snr) if raw_snr else _sample_fallback_snr(seed, idx)
                    eval_pairs.append({
                        "clean_file": row["clean_file"],
                        "noise_file": row["noise_file"],
                        "snr_db": snr_db,
                    })
    else:
        # Fallback if metadata does not exist
        clean_files = sorted(list((root_path / "clean").glob("*.*")))
        noise_files = sorted(list((root_path / "noise").glob("*.*")))
        if not clean_files or not noise_files:
            raise ValueError(f"Arquivos de áudio não encontrados em {root_path}")
        rng = random.Random(seed)
        shuffled_clean = clean_files[:]
        rng.shuffle(shuffled_clean)
        val_count = max(1, int(len(shuffled_clean) * validation_ratio))
        val_clean = shuffled_clean[:val_count]
        for file_index, clean_p in enumerate(val_clean):
            eval_pairs.append({
                "clean_file": str(clean_p),
                "noise_file": str(rng.choice(noise_files)),
                "snr_db": _sample_fallback_snr(seed, file_index),
            })

    if max_samples is not None and max_samples < len(eval_pairs):
        rng = random.Random(seed)
        eval_pairs = rng.sample(eval_pairs, max_samples)

    print(f"[i] Avaliando {len(eval_pairs)} amostras do split de validação (lc_db={lc_db:.1f} dB, threshold={threshold:.4f})...")

    records: List[Dict[str, Any]] = []

    for i, pair in enumerate(eval_pairs, 1):
        clean_audio = _fix_audio_length(load_audio_file(pair["clean_file"], sr=sr), target_samples)
        noise_audio = _fix_audio_length(load_audio_file(pair["noise_file"], sr=sr), target_samples)
        noisy_audio, matched_noise = mix_at_snr(clean_audio, noise_audio, pair["snr_db"])

        clean_mag, _ = compute_stft(clean_audio, n_fft=n_fft, hop_length=hop_length)
        noise_mag, _ = compute_stft(matched_noise, n_fft=n_fft, hop_length=hop_length)
        noisy_mag, noisy_phase = compute_stft(noisy_audio, n_fft=n_fft, hop_length=hop_length)

        # Oracle IRM
        oracle_irm = compute_irm(clean_mag, noise_mag, beta=0.5)
        oracle_enhanced = compute_istft(noisy_mag * oracle_irm, noisy_phase, hop_length=hop_length, length=len(clean_audio))

        # Model enhancement
        t0 = time.perf_counter()
        enhancement = enhance_signal(model, noisy_audio, n_fft=n_fft, hop_length=hop_length, device=torch_device)
        infer_latency_ms = (time.perf_counter() - t0) * 1000.0

        model_enhanced = enhancement.enhanced_signal
        model_mask = enhancement.predicted_mask

        # Metrics for the 3 systems
        noisy_sig_metrics = compute_signal_metrics(clean_audio, noisy_audio, sr=sr)
        model_sig_metrics = compute_signal_metrics(clean_audio, model_enhanced, sr=sr)
        oracle_sig_metrics = compute_signal_metrics(clean_audio, oracle_enhanced, sr=sr)

        # Mask metrics
        model_mask_cls = compute_mask_classification_metrics(model_mask, oracle_irm, threshold=threshold)
        model_mask_reg = compute_mask_regression_metrics(model_mask, oracle_irm)
        model_attenuation = compute_attenuation_metrics(clean_mag, noise_mag, model_mask)

        oracle_mask_cls = compute_mask_classification_metrics(oracle_irm, oracle_irm, threshold=threshold)
        oracle_mask_reg = compute_mask_regression_metrics(oracle_irm, oracle_irm)
        oracle_attenuation = compute_attenuation_metrics(clean_mag, noise_mag, oracle_irm)

        input_snr = compute_snr(clean_audio, noisy_audio)
        snr_bucket = _get_snr_bucket(input_snr)

        record = {
            "sample_index": i,
            "clean_file": pair["clean_file"],
            "noise_file": pair["noise_file"],
            "target_snr_db": pair["snr_db"],
            "input_snr_db": input_snr,
            "snr_bucket": snr_bucket,
            "infer_latency_ms": infer_latency_ms,
            # Noisy
            "noisy_si_sdr": noisy_sig_metrics["si_sdr"],
            "noisy_snr": noisy_sig_metrics["snr"],
            "noisy_seg_snr": noisy_sig_metrics["segmental_snr"],
            "noisy_pesq": noisy_sig_metrics["pesq"],
            "noisy_stoi": noisy_sig_metrics["stoi"],
            "noisy_estoi": noisy_sig_metrics["estoi"],
            # Model
            "model_si_sdr": model_sig_metrics["si_sdr"],
            "model_snr": model_sig_metrics["snr"],
            "model_seg_snr": model_sig_metrics["segmental_snr"],
            "model_pesq": model_sig_metrics["pesq"],
            "model_stoi": model_sig_metrics["stoi"],
            "model_estoi": model_sig_metrics["estoi"],
            "model_mask_precision": model_mask_cls["precision"],
            "model_mask_recall": model_mask_cls["recall"],
            "model_mask_f1": model_mask_cls["f1"],
            "model_mask_accuracy": model_mask_cls["accuracy"],
            "model_mask_hit_rate": model_mask_cls["hit_rate"],
            "model_mask_false_alarm": model_mask_cls["false_alarm_rate"],
            "model_mask_hit_minus_fa": model_mask_cls["hit_minus_false_alarm"],
            "model_mask_mse": model_mask_reg["mse"],
            "model_mask_mae": model_mask_reg["mae"],
            "model_speech_attenuation_db": model_attenuation["speech_attenuation_db"],
            "model_noise_attenuation_db": model_attenuation["noise_attenuation_db"],
            # Oracle IRM
            "oracle_si_sdr": oracle_sig_metrics["si_sdr"],
            "oracle_snr": oracle_sig_metrics["snr"],
            "oracle_seg_snr": oracle_sig_metrics["segmental_snr"],
            "oracle_pesq": oracle_sig_metrics["pesq"],
            "oracle_stoi": oracle_sig_metrics["stoi"],
            "oracle_estoi": oracle_sig_metrics["estoi"],
            "oracle_mask_precision": oracle_mask_cls["precision"],
            "oracle_mask_recall": oracle_mask_cls["recall"],
            "oracle_mask_f1": oracle_mask_cls["f1"],
            "oracle_mask_hit_minus_fa": oracle_mask_cls["hit_minus_false_alarm"],
            "oracle_speech_attenuation_db": oracle_attenuation["speech_attenuation_db"],
            "oracle_noise_attenuation_db": oracle_attenuation["noise_attenuation_db"],
        }
        records.append(record)

        if i % 10 == 0 or i == len(eval_pairs):
            print(f" - [{i}/{len(eval_pairs)}] processados...")

    # Aggregating metrics
    def summarize(values: List[float]) -> Dict[str, float]:
        arr = np.array([v for v in values if not np.isnan(v)])
        if len(arr) == 0:
            return {"mean": float("nan"), "std": float("nan"), "median": float("nan")}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
        }

    metric_names = [
        "si_sdr", "snr", "seg_snr", "pesq", "stoi", "estoi",
    ]

    systems_summary: Dict[str, Dict[str, Any]] = {"noisy": {}, "model": {}, "oracle": {}}
    for m in metric_names:
        systems_summary["noisy"][m] = summarize([r[f"noisy_{m}"] for r in records])
        systems_summary["model"][m] = summarize([r[f"model_{m}"] for r in records])
        systems_summary["oracle"][m] = summarize([r[f"oracle_{m}"] for r in records])

    # Model specific mask metrics
    mask_metric_names = [
        "precision", "recall", "f1", "accuracy", "hit_rate", "false_alarm",
        "hit_minus_fa", "mse", "mae",
    ]
    model_mask_summary = {}
    for mm in mask_metric_names:
        model_mask_summary[mm] = summarize([r[f"model_mask_{mm}"] for r in records])

    oracle_attenuation_summary = {
        "speech_attenuation_db": summarize([r["oracle_speech_attenuation_db"] for r in records]),
        "noise_attenuation_db": summarize([r["oracle_noise_attenuation_db"] for r in records]),
    }

    model_attenuation_summary = {
        "speech_attenuation_db": summarize([r["model_speech_attenuation_db"] for r in records]),
        "noise_attenuation_db": summarize([r["model_noise_attenuation_db"] for r in records]),
    }

    # Deltas
    deltas = {}
    for m in metric_names:
        diff_model = [r[f"model_{m}"] - r[f"noisy_{m}"] for r in records]
        diff_oracle = [r[f"oracle_{m}"] - r[f"noisy_{m}"] for r in records]
        deltas[f"delta_model_vs_noisy_{m}"] = summarize(diff_model)
        deltas[f"delta_oracle_vs_noisy_{m}"] = summarize(diff_oracle)

    # Per-bucket summary
    bucket_records = defaultdict(list)
    for r in records:
        bucket_records[r["snr_bucket"]].append(r)

    buckets_summary = {}
    for b_name, b_recs in sorted(bucket_records.items()):
        buckets_summary[b_name] = {
            "sample_count": len(b_recs),
            "si_sdr": {
                "noisy": summarize([x["noisy_si_sdr"] for x in b_recs])["mean"],
                "model": summarize([x["model_si_sdr"] for x in b_recs])["mean"],
                "oracle": summarize([x["oracle_si_sdr"] for x in b_recs])["mean"],
            },
            "pesq": {
                "noisy": summarize([x["noisy_pesq"] for x in b_recs])["mean"],
                "model": summarize([x["model_pesq"] for x in b_recs])["mean"],
                "oracle": summarize([x["oracle_pesq"] for x in b_recs])["mean"],
            },
            "stoi": {
                "noisy": summarize([x["noisy_stoi"] for x in b_recs])["mean"],
                "model": summarize([x["model_stoi"] for x in b_recs])["mean"],
                "oracle": summarize([x["oracle_stoi"] for x in b_recs])["mean"],
            },
            "mask_f1": summarize([x["model_mask_f1"] for x in b_recs])["mean"],
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir_path = Path(output_dir) / timestamp
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # 1. Save per_sample.csv
    csv_path = out_dir_path / "per_sample.csv"
    if records:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    full_results = {
        "timestamp": timestamp,
        "checkpoint": str(checkpoint_path),
        "total_samples": len(records),
        "threshold_config": {
            "lc_db": lc_db,
            "threshold": threshold,
        },
        "systems": systems_summary,
        "deltas": deltas,
        "mask_metrics": model_mask_summary,
        "attenuation_metrics": model_attenuation_summary,
        "oracle_attenuation_metrics": oracle_attenuation_summary,
        "snr_buckets": buckets_summary,
    }

    # 2. Save summary.json
    json_path = out_dir_path / "summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    # 3. Generate summary.md
    md_path = out_dir_path / "summary.md"
    md_content = f"""# Signal Quality & Mask Evaluation Report

- **Date:** {timestamp}
- **Checkpoint:** `{checkpoint_path}`
- **Validation Samples:** {len(records)}
- **T-F Mask Decision Threshold (LC = {lc_db} dB):** {threshold:.4f}

## 1. System Performance Comparison

| Metric | Noisy (Baseline) | TinyU-Net (Model) | Oracle IRM (Upper Bound) | Δ Model vs Noisy |
|---|---|---|---|---|
| **SI-SDR (dB)** | {systems_summary['noisy']['si_sdr']['mean']:.2f} ± {systems_summary['noisy']['si_sdr']['std']:.2f} | {systems_summary['model']['si_sdr']['mean']:.2f} ± {systems_summary['model']['si_sdr']['std']:.2f} | {systems_summary['oracle']['si_sdr']['mean']:.2f} ± {systems_summary['oracle']['si_sdr']['std']:.2f} | **{deltas['delta_model_vs_noisy_si_sdr']['mean']:+.2f} dB** |
| **SNR (dB)** | {systems_summary['noisy']['snr']['mean']:.2f} ± {systems_summary['noisy']['snr']['std']:.2f} | {systems_summary['model']['snr']['mean']:.2f} ± {systems_summary['model']['snr']['std']:.2f} | {systems_summary['oracle']['snr']['mean']:.2f} ± {systems_summary['oracle']['snr']['std']:.2f} | **{deltas['delta_model_vs_noisy_snr']['mean']:+.2f} dB** |
| **segSNR (dB)** | {systems_summary['noisy']['seg_snr']['mean']:.2f} ± {systems_summary['noisy']['seg_snr']['std']:.2f} | {systems_summary['model']['seg_snr']['mean']:.2f} ± {systems_summary['model']['seg_snr']['std']:.2f} | {systems_summary['oracle']['seg_snr']['mean']:.2f} ± {systems_summary['oracle']['seg_snr']['std']:.2f} | **{deltas['delta_model_vs_noisy_seg_snr']['mean']:+.2f} dB** |
| **PESQ (Wideband)** | {systems_summary['noisy']['pesq']['mean']:.2f} ± {systems_summary['noisy']['pesq']['std']:.2f} | {systems_summary['model']['pesq']['mean']:.2f} ± {systems_summary['model']['pesq']['std']:.2f} | {systems_summary['oracle']['pesq']['mean']:.2f} ± {systems_summary['oracle']['pesq']['std']:.2f} | **{deltas['delta_model_vs_noisy_pesq']['mean']:+.2f}** |
| **STOI** | {systems_summary['noisy']['stoi']['mean']:.3f} ± {systems_summary['noisy']['stoi']['std']:.3f} | {systems_summary['model']['stoi']['mean']:.3f} ± {systems_summary['model']['stoi']['std']:.3f} | {systems_summary['oracle']['stoi']['mean']:.3f} ± {systems_summary['oracle']['stoi']['std']:.3f} | **{deltas['delta_model_vs_noisy_stoi']['mean']:+.3f}** |
| **ESTOI** | {systems_summary['noisy']['estoi']['mean']:.3f} ± {systems_summary['noisy']['estoi']['std']:.3f} | {systems_summary['model']['estoi']['mean']:.3f} ± {systems_summary['model']['estoi']['std']:.3f} | {systems_summary['oracle']['estoi']['mean']:.3f} ± {systems_summary['oracle']['estoi']['std']:.3f} | **{deltas['delta_model_vs_noisy_estoi']['mean']:+.3f}** |

## 2. Time-Frequency Mask Evaluation (Classification & Regression)

| Metric | TinyU-Net Value | Description |
|---|---|---|
| **Precision** | {model_mask_summary['precision']['mean']:.3f} | Speech T-F bins predicted correctly / all predicted speech bins |
| **Recall (Hit Rate)** | {model_mask_summary['recall']['mean']:.3f} | Speech T-F bins preserved / total ground-truth speech bins |
| **F1-Score** | {model_mask_summary['f1']['mean']:.3f} | Harmonic mean of Precision and Recall on spectrogram |
| **Accuracy** | {model_mask_summary['accuracy']['mean']:.3f} | Overall pixel accuracy across all T-F bins |
| **False Alarm Rate** | {model_mask_summary['false_alarm']['mean']:.3f} | Noise T-F bins erroneously classified as speech |
| **HIT - FA** | {model_mask_summary['hit_minus_fa']['mean']:.3f} | Standard speech separation metric (Kim et al., 2009) |
| **Mask MSE** | {model_mask_summary['mse']['mean']:.4f} | Mean Squared Error vs. continuous Ideal Ratio Mask |
| **Mask MAE** | {model_mask_summary['mae']['mean']:.4f} | Mean Absolute Error vs. continuous Ideal Ratio Mask |
| **Speech Attenuation (SA)** | {model_attenuation_summary['speech_attenuation_db']['mean']:.2f} dB (oracle: {oracle_attenuation_summary['speech_attenuation_db']['mean']:.2f} dB) | Speech energy loss (lower is better) |
| **Noise Attenuation (NA)** | {model_attenuation_summary['noise_attenuation_db']['mean']:.2f} dB (oracle: {oracle_attenuation_summary['noise_attenuation_db']['mean']:.2f} dB) | Noise energy removed (higher is better) |

## 3. Breakdown by Input SNR Bucket

| SNR Bucket | Count | Noisy SI-SDR | Model SI-SDR | Model PESQ | Model STOI | Mask F1 |
|---|---|---|---|---|---|---|
"""
    for b_name, b_val in sorted(buckets_summary.items()):
        md_content += f"| {b_name} | {b_val['sample_count']} | {b_val['si_sdr']['noisy']:.2f} dB | {b_val['si_sdr']['model']:.2f} dB | {b_val['pesq']['model']:.2f} | {b_val['stoi']['model']:.3f} | {b_val['mask_f1']:.3f} |\n"

    with md_path.open("w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n[+] Relatórios de avaliação gerados:")
    print(f"  - CSV detalhado: {csv_path}")
    print(f"  - Resumo JSON:   {json_path}")
    print(f"  - Resumo MD:     {md_path}")

    # Console summary print
    print("\n" + "=" * 65)
    print(f"{'METRIC':<18} | {'NOISY':<10} | {'MODEL':<10} | {'ORACLE':<10} | {'Δ MODEL'}")
    print("-" * 65)
    for m in ["si_sdr", "pesq", "stoi"]:
        n_m = systems_summary["noisy"][m]["mean"]
        mod_m = systems_summary["model"][m]["mean"]
        o_m = systems_summary["oracle"][m]["mean"]
        d_m = deltas[f"delta_model_vs_noisy_{m}"]["mean"]
        print(f"{m.upper():<18} | {n_m:<10.2f} | {mod_m:<10.2f} | {o_m:<10.2f} | {d_m:+.2f}")
    print(f"{'MASK F1':<18} | {'-':<10} | {model_mask_summary['f1']['mean']:<10.3f} | {'1.000':<10} | -")
    print(f"{'HIT - FA':<18} | {'-':<10} | {model_mask_summary['hit_minus_fa']['mean']:<10.3f} | {'1.000':<10} | -")
    print("=" * 65)

    return full_results
