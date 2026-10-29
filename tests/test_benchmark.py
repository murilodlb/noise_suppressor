import torch
import torch.nn as nn

from src.benchmark import (
    compute_model_size_bytes,
    count_macs,
    count_parameters,
    estimate_activation_memory,
)
from src.model import TinyUNet


def test_count_parameters():
    model = TinyUNet(in_channels=1, base_channels=8)
    info = count_parameters(model)
    actual_sum = sum(p.numel() for p in model.parameters())
    assert info["total_parameters"] == actual_sum
    assert info["total_parameters"] > 0


def test_compute_model_size():
    model = TinyUNet(in_channels=1, base_channels=8)
    size_info = compute_model_size_bytes(model)
    assert size_info["float32_bytes"] > 0
    assert size_info["int8_estimated_bytes"] == size_info["float32_bytes"] // 4


def test_count_macs_simple_conv():
    conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, bias=False)
    # Input shape: (1, 1, 10, 10)
    # Output shape: (1, 8, 10, 10)
    # MACs = 1 * 8 * 10 * 10 * 1 * 3 * 3 = 7200
    mac_res = count_macs(conv, (1, 1, 10, 10))
    assert mac_res["total_macs"] == 7200


def test_estimate_activation_memory():
    model = TinyUNet(in_channels=1, base_channels=8)
    mem_info = estimate_activation_memory(model, (1, 1, 257, 126))
    assert mem_info["activation_bytes"] > 0
    assert mem_info["activation_kb"] > 0
