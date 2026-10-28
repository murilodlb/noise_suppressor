# Walkthrough: uv Pipeline, Signal Quality Evaluation & Overhead Benchmark

We implemented a complete `uv`-managed pipeline for the noise suppression project, supporting preprocessing, training, inference, PoC demonstration, signal quality evaluation with standard audio and time-frequency mask metrics (analogous to Precision/Recall/F1 in classification), and computational overhead benchmarking for TinyML.

---

## 1. Environment & Package Modernization (`uv`)

- **Configuration:** Created [`pyproject.toml`](file:///home/murilo/Desktop/projetos/noise_suppressor/pyproject.toml) defining dependencies, Python 3.12 target, Hatchling build system, and the CPU PyTorch index (`pytorch-cpu`).
- **Reproducibility:** Pinned Python in [`.python-version`](file:///home/murilo/Desktop/projetos/noise_suppressor/.python-version) and locked the entire dependency tree in [`uv.lock`](file:///home/murilo/Desktop/projetos/noise_suppressor/uv.lock). Removed obsolete `requirements.txt`.
- **Command-Line Entrypoint:** Registered the unified `noise_suppressor` script with subcommands:
  ```bash
  uv run noise_suppressor {preprocess,train,infer,demo,evaluate,benchmark}
  ```

---

## 2. Signal Quality Evaluation ("Precision / Recall / F1 for Signals")

Created [`src/metrics.py`](file:///home/murilo/Desktop/projetos/noise_suppressor/src/metrics.py) and [`src/evaluate.py`](file:///home/murilo/Desktop/projetos/noise_suppressor/src/evaluate.py) to provide multi-dimensional quality assessment:

### Analogies between Classic ML and Signal Processing
| Classification / ML Concept | Signal Processing Metric | Description |
|---|---|---|
| **Precision** | **T-F Bin Precision** | Ratio of true speech bins among all bins where the predicted mask is open (avoids speech artifacts & noise leak). |
| **Recall (Hit Rate)** | **T-F Bin Recall** | Ratio of preserved speech bins among all ground-truth speech bins (avoids clipping speaker voice). |
| **F1-Score** | **T-F Mask F1** | Harmonic mean of speech preservation and noise rejection on the spectrogram. |
| **HIT − FA** | **Hit Rate − False Alarm** | Canonical speech separation metric (Kim et al., 2009). |
| **Regression Error** | **Mask MSE & MAE** | Difference against continuous Ideal Ratio Mask (IRM). |
| **Speech Attenuation (SA)** | $10 \log_{10} \frac{\|S\|^2}{\|M \odot S\|^2}$ | Speech loss in dB (lower is better). |
| **Noise Attenuation (NA)** | $10 \log_{10} \frac{\|N\|^2}{\|M \odot N\|^2}$ | Noise removed in dB (higher is better). |
| **Waveform Accuracy** | **SI-SDR, SNR, segSNR (dB)** | Scale-invariant signal-to-distortion ratio against clean reference. |
| **Perceptual Quality** | **PESQ (Wideband)** | ITU-T P.862.2 perceptual speech quality score (1.0 to 4.5). |
| **Intelligibility** | **STOI / ESTOI** | Short-Time Objective Intelligibility score (0.0 to 1.0). |

### Multi-System Benchmarking
Each validation sample is tested against 3 systems:
1. `noisy`: Unprocessed input (lower bound).
2. `model`: TinyU-Net inference output.
3. `oracle_irm`: Ground-truth Ideal Ratio Mask with noisy phase (theoretical upper bound).

#### Evaluation Results on Held-out Validation Data (30 samples)
| Metric | Noisy (Baseline) | TinyU-Net (Model) | Oracle IRM (Upper Bound) | Δ Model vs Noisy |
|---|---|---|---|---|
| **SI-SDR** | 10.54 dB | **18.98 dB** | 23.81 dB | **+8.44 dB** |
| **SNR** | 10.52 dB | **19.27 dB** | 23.93 dB | **+8.76 dB** |
| **segSNR** | 4.61 dB | **9.61 dB** | 14.20 dB | **+5.00 dB** |
| **PESQ** | 1.96 | **2.69** | 4.15 | **+0.79** |
| **STOI** | 0.854 | **0.894** | 0.974 | **+0.041** |
| **T-F Mask Precision** | - | **0.870** | 1.000 | - |
| **T-F Mask Recall** | - | **0.646** | 1.000 | - |
| **T-F Mask F1** | - | **0.734** | 1.000 | - |
| **HIT − FA** | - | **0.546** | 1.000 | - |
| **Speech Attenuation** | - | **0.54 dB** | 0.49 dB | - |
| **Noise Attenuation** | - | **11.15 dB** | 14.82 dB | - |

---

## 3. Computational Overhead Benchmark (TinyML)

Created [`src/benchmark.py`](file:///home/murilo/Desktop/projetos/noise_suppressor/src/benchmark.py) to measure model footprint and inference overhead:

- **Parameter Count:** 29,641 parameters
- **Storage / Memory Footprint:**
  - Float32 weights: **117 KB**
  - Quantized Int8 estimate: **29.3 KB**
  - Peak activation memory: **21.1 MB** (for a 2.0s clip)
- **Computational Complexity:**
  - MACs per 2.0s clip: ~196 MMACs
  - Normalized complexity: **98.0 MMACs / second of audio**
- **Latency Breakdown on CPU (2.0s audio clip):**
  - STFT: **0.84 ms** (5.8%)
  - Log1p & Z-Score Preprocessing: **0.37 ms** (2.6%)
  - TinyU-Net Forward Pass: **11.59 ms** (79.8%)
  - iSTFT Synthesis: **1.73 ms** (11.9%)
  - **Total Pipeline Latency:** **14.53 ms**
- **Real-Time Factor (RTF):** `0.0073` (**137× faster than real-time** on standard CPU).
- **Algorithmic Latency:** STFT window size 32 ms (hop size 16 ms), non-causal model requiring full clip buffering.

---

## 4. Verification & Testing

### Automated Test Suite
All 15 unit tests passed:
```bash
uv run pytest -v
# 15 passed in 11.27s (test_metrics, test_benchmark, test_inference)
```

### CLI Subcommands Verified
- `uv run noise_suppressor --help`
- `uv run noise_suppressor demo` (reconstructed audio + spectrograms)
- `uv run noise_suppressor infer --input data/audio/noisy.wav` (saved `reports/inference/noisy_enhanced.wav`)
- `uv run noise_suppressor benchmark --repeats 20` (saved `reports/benchmark/benchmark.json`)
- `uv run noise_suppressor evaluate --max_samples 30` (saved `reports/evaluation/<timestamp>/summary.md`)
- `uv run noise_suppressor train --epochs 1 --checkpoint_path reports/smoke/tinyunet_smoke.pth`

