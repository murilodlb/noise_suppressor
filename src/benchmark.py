import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .dsp_utils import compute_istft, compute_stft, preprocess_noisy_magnitude
from .inference import load_model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Counts total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffer_params = sum(b.numel() for b in model.buffers())
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "buffer_parameters": buffer_params,
    }


def compute_model_size_bytes(model: nn.Module) -> Dict[str, Any]:
    """Computes actual float32 footprint and estimated int8 quantized footprint."""
    param_numel = sum(p.numel() for p in model.parameters())
    buffer_numel = sum(b.numel() for b in model.buffers())
    total_numel = param_numel + buffer_numel

    fp32_bytes = total_numel * 4
    int8_est_bytes = total_numel * 1

    return {
        "float32_bytes": fp32_bytes,
        "float32_kb": fp32_bytes / 1024.0,
        "float32_mb": fp32_bytes / (1024.0 * 1024.0),
        "int8_estimated_bytes": int8_est_bytes,
        "int8_estimated_kb": int8_est_bytes / 1024.0,
    }


def count_macs(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
    """
    Computes Multiply-Accumulate operations (MACs) for the network using forward hooks.
    Counts Conv2d, ConvTranspose2d, BatchNorm2d, and Linear.
    """
    macs_by_layer = {}
    handles = []

    def conv2d_hook(name: str):
        def hook(layer: nn.Conv2d, inputs, outputs):
            # Output shape: (B, C_out, H_out, W_out)
            output = outputs
            b, c_out, h_out, w_out = output.shape
            k_h, k_w = layer.kernel_size
            c_in = layer.in_channels // layer.groups
            layer_macs = b * c_out * h_out * w_out * c_in * k_h * k_w
            macs_by_layer[name] = layer_macs
        return hook

    def conv_transpose2d_hook(name: str):
        def hook(layer: nn.ConvTranspose2d, inputs, outputs):
            # Input shape: (B, C_in, H_in, W_in)
            inp = inputs[0]
            b, c_in, h_in, w_in = inp.shape
            k_h, k_w = layer.kernel_size
            c_out = layer.out_channels // layer.groups
            layer_macs = b * c_in * c_out * k_h * k_w * h_in * w_in
            macs_by_layer[name] = layer_macs
        return hook

    def bn2d_hook(name: str):
        def hook(layer: nn.BatchNorm2d, inputs, outputs):
            # Normalization + affine transformation
            output = outputs
            macs_by_layer[name] = int(output.numel())
        return hook

    def linear_hook(name: str):
        def hook(layer: nn.Linear, inputs, outputs):
            output = outputs
            num_instances = int(output.numel() // layer.out_features)
            macs_by_layer[name] = num_instances * layer.in_features * layer.out_features
        return hook

    def rnn_hook(name: str, num_gates: int):
        def hook(layer: nn.RNNBase, inputs, outputs):
            inp = inputs[0]
            if layer.batch_first:
                batch_size, seq_len, _ = inp.shape
            else:
                seq_len, batch_size, _ = inp.shape
            num_directions = 2 if layer.bidirectional else 1
            hidden_size = layer.hidden_size
            total_layer_macs = 0
            for layer_idx in range(layer.num_layers):
                in_sz = layer.input_size if layer_idx == 0 else hidden_size * num_directions
                step_macs = num_gates * (in_sz * hidden_size + hidden_size * hidden_size)
                total_layer_macs += batch_size * seq_len * num_directions * step_macs
            macs_by_layer[name] = int(total_layer_macs)
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv2d_hook(name)))
        elif isinstance(module, nn.ConvTranspose2d):
            handles.append(module.register_forward_hook(conv_transpose2d_hook(name)))
        elif isinstance(module, nn.BatchNorm2d):
            handles.append(module.register_forward_hook(bn2d_hook(name)))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook(name)))
        elif isinstance(module, nn.GRU):
            handles.append(module.register_forward_hook(rnn_hook(name, num_gates=3)))
        elif isinstance(module, nn.LSTM):
            handles.append(module.register_forward_hook(rnn_hook(name, num_gates=4)))

    device = next(model.parameters()).device
    dummy_input = torch.zeros(input_shape, device=device)

    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)

    for h in handles:
        h.remove()

    total_macs = sum(macs_by_layer.values())
    return {
        "total_macs": total_macs,
        "total_m_macs": total_macs / 1e6,
        "macs_by_layer": macs_by_layer,
    }


def estimate_activation_memory(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
    """Measures the cumulative peak activation tensor memory during a forward pass."""
    total_elements = 0
    handles = []

    def hook(module, inputs, outputs):
        nonlocal total_elements
        if isinstance(outputs, torch.Tensor):
            total_elements += outputs.numel()
        elif isinstance(outputs, (tuple, list)):
            for o in outputs:
                if isinstance(o, torch.Tensor):
                    total_elements += o.numel()

    for module in model.modules():
        if len(list(module.children())) == 0:  # Leaf module
            handles.append(module.register_forward_hook(hook))

    device = next(model.parameters()).device
    dummy_input = torch.zeros(input_shape, device=device)

    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)

    for h in handles:
        h.remove()

    activation_bytes = total_elements * 4  # float32
    return {
        "activation_elements": total_elements,
        "activation_bytes": activation_bytes,
        "activation_kb": activation_bytes / 1024.0,
        "activation_mb": activation_bytes / (1024.0 * 1024.0),
    }


def time_pipeline_stages(
    model: nn.Module,
    signal: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 256,
    device: Optional[torch.device] = None,
    warmup: int = 5,
    repeats: int = 50,
) -> Dict[str, Any]:
    """Benchmarks latency for each stage: STFT, Preprocessing, Model forward, Masking + iSTFT, and Total."""
    if device is None:
        device = next(model.parameters()).device

    # Warmup
    for _ in range(warmup):
        mag, phase = compute_stft(signal, n_fft=n_fft, hop_length=hop_length)
        inp = preprocess_noisy_magnitude(mag)
        t = torch.from_numpy(inp).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            mask = model(t).squeeze().cpu().numpy()
        _ = compute_istft(mag * mask, phase, hop_length=hop_length, length=len(signal))

    stft_times = []
    pre_times = []
    forward_times = []
    istft_times = []
    total_times = []

    for _ in range(repeats):
        t0 = time.perf_counter()

        # Stage 1: STFT
        t_start = time.perf_counter()
        mag, phase = compute_stft(signal, n_fft=n_fft, hop_length=hop_length)
        t_stft = time.perf_counter() - t_start

        # Stage 2: Preprocess
        t_start = time.perf_counter()
        inp = preprocess_noisy_magnitude(mag)
        t = torch.from_numpy(inp).float().unsqueeze(0).unsqueeze(0).to(device)
        t_pre = time.perf_counter() - t_start

        # Stage 3: Model forward
        t_start = time.perf_counter()
        with torch.no_grad():
            mask = model(t)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_fwd = time.perf_counter() - t_start

        # Stage 4: Masking & iSTFT
        t_start = time.perf_counter()
        mask_np = mask.squeeze().cpu().numpy()
        _ = compute_istft(mag * mask_np, phase, hop_length=hop_length, length=len(signal))
        t_istft = time.perf_counter() - t_start

        t_total = time.perf_counter() - t0

        stft_times.append(t_stft * 1000.0)
        pre_times.append(t_pre * 1000.0)
        forward_times.append(t_fwd * 1000.0)
        istft_times.append(t_istft * 1000.0)
        total_times.append(t_total * 1000.0)

    def stats(arr):
        return {
            "mean_ms": float(np.mean(arr)),
            "median_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "std_ms": float(np.std(arr)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
        }

    return {
        "stft": stats(stft_times),
        "preprocessing": stats(pre_times),
        "model_forward": stats(forward_times),
        "istft_masking": stats(istft_times),
        "total": stats(total_times),
    }


def run_benchmark(
    checkpoint_path: Optional[str | Path] = None,
    seconds: float = 2.0,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 256,
    warmup: int = 5,
    repeats: int = 50,
    num_threads: Optional[int] = None,
    device: str = "cpu",
    output_path: Optional[str | Path] = None,
    model_name: Optional[str] = None,
    base_channels: int = 8,
) -> Dict[str, Any]:
    """Runs a full computational overhead benchmark and exports results."""
    if num_threads is not None and num_threads > 0:
        torch.set_num_threads(num_threads)

    torch_device = torch.device(device)
    model, dsp_config = load_model(
        checkpoint_path,
        device=torch_device,
        model_name=model_name,
        base_channels=base_channels,
    )
    model.eval()

    resolved_model_name = dsp_config.get("model_name", model_name or "tinyunet")
    resolved_base_channels = dsp_config.get("base_channels", base_channels)
    sr = dsp_config.get("sr", sr)
    n_fft = dsp_config.get("n_fft", n_fft)
    hop_length = dsp_config.get("hop_length", hop_length)

    # 1. Parameter counts
    param_info = count_parameters(model)
    size_info = compute_model_size_bytes(model)

    # 2. MACs and Activation Memory
    num_samples = int(sr * seconds)
    dummy_signal = np.random.randn(num_samples).astype(np.float32)
    dummy_mag, _ = compute_stft(dummy_signal, n_fft=n_fft, hop_length=hop_length)
    input_shape = (1, 1, dummy_mag.shape[0], dummy_mag.shape[1])

    mac_info = count_macs(model, input_shape)
    activation_info = estimate_activation_memory(model, input_shape)

    macs_per_clip = mac_info["total_macs"]
    macs_per_second = macs_per_clip / seconds

    # 3. Latency timing
    timing_info = time_pipeline_stages(
        model=model,
        signal=dummy_signal,
        n_fft=n_fft,
        hop_length=hop_length,
        device=torch_device,
        warmup=warmup,
        repeats=repeats,
    )

    total_mean_ms = timing_info["total"]["mean_ms"]
    audio_duration_ms = seconds * 1000.0
    rtf = total_mean_ms / audio_duration_ms
    throughput_x_rt = audio_duration_ms / total_mean_ms if total_mean_ms > 0 else 0.0

    # 4. Latency breakdown percentages
    breakdown = {
        "stft_pct": (timing_info["stft"]["mean_ms"] / total_mean_ms) * 100.0,
        "preprocessing_pct": (timing_info["preprocessing"]["mean_ms"] / total_mean_ms) * 100.0,
        "model_forward_pct": (timing_info["model_forward"]["mean_ms"] / total_mean_ms) * 100.0,
        "istft_pct": (timing_info["istft_masking"]["mean_ms"] / total_mean_ms) * 100.0,
    }

    # 5. Algorithmic latency description
    stft_window_ms = (n_fft / sr) * 1000.0
    hop_size_ms = (hop_length / sr) * 1000.0
    algorithmic_latency = {
        "stft_window_ms": stft_window_ms,
        "hop_size_ms": hop_size_ms,
        "model_causality": f"non-causal ({type(model).__name__} uses pooling and 2D convolutions over the full time frame)",
        "minimum_buffering_required_seconds": seconds,
    }

    system_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": str(torch_device),
        "num_threads": torch.get_num_threads(),
    }

    results = {
        "model_name": resolved_model_name,
        "base_channels": resolved_base_channels,
        "system_info": system_info,
        "audio_config": {
            "sr": sr,
            "seconds": seconds,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "input_spectrogram_shape": list(input_shape),
        },
        "model_parameters": param_info,
        "model_size": size_info,
        "computational_complexity": {
            "macs_per_clip": macs_per_clip,
            "macs_per_second": macs_per_second,
            "m_macs_per_second": macs_per_second / 1e6,
        },
        "activation_memory": activation_info,
        "latency_stages": timing_info,
        "latency_breakdown_percentage": breakdown,
        "real_time_factor": {
            "rtf": rtf,
            "throughput_x_real_time": throughput_x_rt,
            "is_real_time_capable": rtf < 1.0,
        },
        "algorithmic_latency": algorithmic_latency,
    }

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[+] Relatório de benchmark salvo em: {out_p}")

    return results
