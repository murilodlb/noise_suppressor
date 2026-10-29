import numpy as np
import torch

from src.inference import enhance_signal, load_model, predict_mask
from src.model import TinyUNet


def test_enhance_signal_output_length_preserved():
    model = TinyUNet(in_channels=1, base_channels=8)
    model.eval()

    # Test with standard length (2.0s at 16kHz = 32000 samples)
    sig_standard = np.random.randn(32000).astype(np.float32)
    res_std = enhance_signal(model, sig_standard, n_fft=512, hop_length=256)
    assert len(res_std.enhanced_signal) == 32000

    # Test with arbitrary non-power-of-two length (e.g. 3.7s = 59200 samples)
    sig_arb = np.random.randn(59200).astype(np.float32)
    res_arb = enhance_signal(model, sig_arb, n_fft=512, hop_length=256)
    assert len(res_arb.enhanced_signal) == 59200


def test_mask_values_in_unit_interval():
    model = TinyUNet(in_channels=1, base_channels=8)
    noisy_mag = np.abs(np.random.randn(257, 126)).astype(np.float32)
    mask = predict_mask(model, noisy_mag)
    assert mask.shape == noisy_mag.shape
    assert np.all(mask >= 0.0)
    assert np.all(mask <= 1.0)


def test_load_model_fallback():
    model, cfg = load_model("non_existent_checkpoint.pth")
    assert isinstance(model, TinyUNet)
    assert cfg["sr"] == 16000
