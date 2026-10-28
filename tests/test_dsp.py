import csv
from unittest import mock

import numpy as np
import pytest
import torch

from src.dataset_builder import NoiseSuppressionDataset, PrecomputedNoiseSuppressionDataset
from src.dsp_utils import mix_at_snr, plot_spectrograms
from src.metrics import compute_snr

SR = 16000


def _make_clean(num_samples: int = 32000) -> np.ndarray:
    t = np.arange(num_samples) / SR
    return (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


@pytest.mark.parametrize("target_snr_db", [-5.0, 0.0, 10.0])
def test_mix_at_snr_hits_target(target_snr_db):
    rng = np.random.default_rng(0)
    clean = _make_clean()
    noise = (0.01 * rng.standard_normal(32000)).astype(np.float32)  # ruído propositalmente fraco
    noisy, scaled_noise = mix_at_snr(clean, noise, target_snr_db)
    assert abs(compute_snr(clean, noisy) - target_snr_db) < 1e-3


def test_mix_at_snr_linear_relation():
    rng = np.random.default_rng(1)
    clean = _make_clean()
    noise = rng.standard_normal(32000).astype(np.float32)
    noisy, scaled_noise = mix_at_snr(clean, noise, -5.0)
    assert noisy.dtype == np.float32 and scaled_noise.dtype == np.float32
    assert np.allclose(noisy - clean, scaled_noise, atol=1e-6)


@pytest.mark.parametrize("noise_length", [48000, 10000])
def test_mix_at_snr_length_matching(noise_length):
    rng = np.random.default_rng(2)
    clean = _make_clean()
    noise = rng.standard_normal(noise_length).astype(np.float32)
    noisy, scaled_noise = mix_at_snr(clean, noise, 3.0)
    assert len(noisy) == len(clean) == len(scaled_noise)
    assert abs(compute_snr(clean, noisy) - 3.0) < 1e-3


def test_mix_at_snr_zero_noise():
    clean = _make_clean()
    noisy, scaled_noise = mix_at_snr(clean, np.zeros_like(clean), 5.0)
    assert np.all(np.isfinite(noisy))
    assert np.allclose(noisy, clean)
    assert np.all(scaled_noise == 0.0)


def test_mix_at_snr_silent_clean():
    rng = np.random.default_rng(3)
    noise = rng.standard_normal(32000).astype(np.float32)
    noisy, scaled_noise = mix_at_snr(np.zeros(32000, dtype=np.float32), noise, 5.0)
    assert np.all(np.isfinite(noisy)) and np.all(np.isfinite(scaled_noise))


def test_add_noise_samples_from_snr_range(tmp_path):
    dataset = NoiseSuppressionDataset(str(tmp_path), str(tmp_path), snr_range=(3.0, 3.0))
    rng = np.random.default_rng(4)
    clean = _make_clean()
    noisy, scaled_noise, snr_db = dataset._add_noise(clean, rng.standard_normal(32000).astype(np.float32))
    assert snr_db == 3.0
    assert abs(compute_snr(clean, noisy) - 3.0) < 1e-3


def test_plot_spectrograms_time_axis_matches_duration(tmp_path):
    # 2.0 s a 16 kHz com hop 256 -> 126 frames
    mag = np.abs(np.random.default_rng(5).standard_normal((257, 126))).astype(np.float32) + 1e-3
    captured = {}

    def fake_savefig(*args, **kwargs):
        import matplotlib.pyplot as plt
        captured["xlim"] = plt.gcf().axes[0].get_xlim()

    with mock.patch("src.dsp_utils.plt.savefig", side_effect=fake_savefig):
        plot_spectrograms(mag, mag, mag, sr=SR, output_path=str(tmp_path / "s.png"), hop_length=256)

    x_max = captured["xlim"][1]
    assert abs(x_max - 2.0) < 0.05, f"eixo do tempo termina em {x_max:.2f}s (esperado ~2.0s)"


def test_precomputed_dataset_follows_metadata_order(tmp_path):
    for name, value in [("b", 1.0), ("a", 2.0), ("stale", 9.0)]:
        torch.save({"noisy_mag": torch.full((1, 2, 2), value), "irm": torch.zeros(1, 2, 2)}, tmp_path / f"{name}.pt")

    with (tmp_path / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_file", "clean_file", "noise_file", "sample_index"])
        writer.writeheader()
        writer.writerow({"sample_file": str(tmp_path / "b.pt"), "clean_file": "c1", "noise_file": "n", "sample_index": 0})
        writer.writerow({"sample_file": str(tmp_path / "a.pt"), "clean_file": "c2", "noise_file": "n", "sample_index": 0})

    dataset = PrecomputedNoiseSuppressionDataset(str(tmp_path))
    assert len(dataset) == 2  # arquivo órfão ignorado
    assert float(dataset[0][0][0, 0, 0]) == 1.0  # linha 0 do CSV -> b.pt
    assert float(dataset[1][0][0, 0, 0]) == 2.0
