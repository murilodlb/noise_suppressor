import math

import numpy as np
import pytest

from src.metrics import (
    compute_attenuation_metrics,
    compute_mask_classification_metrics,
    compute_mask_regression_metrics,
    compute_mask_threshold,
    compute_pesq,
    compute_segmental_snr,
    compute_si_sdr,
    compute_snr,
    compute_stoi,
)


def test_compute_snr_identical():
    x = np.sin(np.linspace(0, 100, 16000))
    snr = compute_snr(x, x)
    assert snr > 80.0


def test_compute_si_sdr_scale_invariance():
    # SI-SDR should be invariant to positive scaling
    t = np.linspace(0, 1.0, 16000)
    ref = np.sin(2 * np.pi * 440 * t)
    scaled_est = 0.3 * ref
    sdr_exact = compute_si_sdr(ref, ref)
    sdr_scaled = compute_si_sdr(ref, scaled_est)
    assert sdr_exact >= 90.0
    assert sdr_scaled >= 90.0


def test_compute_si_sdr_with_known_noise():
    np.random.seed(42)
    t = np.linspace(0, 1.0, 16000)
    signal = np.sin(2 * np.pi * 440 * t)
    noise = np.random.normal(0, 1.0, len(t))
    # Scale noise to 10 dB SNR
    sig_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    target_snr_db = 10.0
    k = math.sqrt(sig_power / (noise_power * (10 ** (target_snr_db / 10.0))))
    noisy = signal + k * noise

    si_sdr = compute_si_sdr(signal, noisy)
    assert abs(si_sdr - target_snr_db) < 0.6


def test_compute_mask_threshold():
    # At LC = 0 dB and beta = 0.5, threshold = (1 / 2)^0.5 = 1 / sqrt(2) ≈ 0.7071
    th = compute_mask_threshold(0.0, beta=0.5)
    assert abs(th - 0.70710678) < 1e-4


def test_compute_mask_classification_metrics_perfect():
    mask = np.array([[0.1, 0.9], [0.8, 0.2]])
    res = compute_mask_classification_metrics(mask, mask, threshold=0.5)
    assert math.isclose(res["precision"], 1.0, rel_tol=1e-5)
    assert math.isclose(res["recall"], 1.0, rel_tol=1e-5)
    assert math.isclose(res["f1"], 1.0, rel_tol=1e-5)
    assert math.isclose(res["accuracy"], 1.0, rel_tol=1e-5)
    assert math.isclose(res["hit_minus_false_alarm"], 1.0, rel_tol=1e-5)


def test_compute_mask_regression_metrics():
    p = np.array([[1.0, 0.0], [0.5, 0.5]])
    t = np.array([[0.0, 0.0], [0.5, 1.0]])
    res = compute_mask_regression_metrics(p, t)
    # diff: [1.0, 0.0, 0.0, -0.5] -> sq: [1.0, 0.0, 0.0, 0.25] -> mean: 1.25 / 4 = 0.3125
    assert math.isclose(res["mse"], 0.3125, rel_tol=1e-5)
    assert math.isclose(res["mae"], (1.0 + 0.5) / 4.0, rel_tol=1e-5)


def test_compute_attenuation_metrics():
    clean_mag = np.ones((10, 10))
    noise_mag = np.ones((10, 10))

    # Mask of ones preserves clean and noise (0 dB attenuation)
    all_ones = np.ones((10, 10))
    res1 = compute_attenuation_metrics(clean_mag, noise_mag, all_ones)
    assert abs(res1["speech_attenuation_db"]) < 1e-3
    assert abs(res1["noise_attenuation_db"]) < 1e-3

    # Mask of 0.5 -> power = 0.25 -> 10 * log10(1 / 0.25) = 10 * log10(4) ≈ 6.02 dB
    half = np.full((10, 10), 0.5)
    res2 = compute_attenuation_metrics(clean_mag, noise_mag, half)
    assert abs(res2["speech_attenuation_db"] - 6.0206) < 1e-3
    assert abs(res2["noise_attenuation_db"] - 6.0206) < 1e-3


def test_pesq_and_stoi_on_speech():
    sr = 16000
    t = np.linspace(0, 1.0, sr)
    speech_like = (0.5 * np.sin(2 * np.pi * 200 * t) + 0.3 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
    stoi_val = compute_stoi(speech_like, speech_like, sr=sr)
    assert stoi_val > 0.95


def test_compute_stoi_returns_nan_on_short_active_speech():
    sr = 16000
    # 2.0s de áudio total com apenas 100ms (1600 amostras) de atividade (< 384ms exigidos pelo pystoi)
    short_burst = np.zeros(32000, dtype=np.float32)
    t = np.linspace(0, 0.1, 1600)
    short_burst[:1600] = (0.5 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)
    stoi_val = compute_stoi(short_burst, short_burst, sr=sr)
    assert math.isnan(stoi_val)

