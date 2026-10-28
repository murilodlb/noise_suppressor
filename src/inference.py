from dataclasses import dataclass
import io
from pathlib import Path
import subprocess
from typing import Tuple, Dict, Any, Optional

import librosa
import numpy as np
import soundfile as sf
import soxr
import torch

from .dsp_utils import compute_stft, compute_istft, preprocess_noisy_magnitude
from .model import TinyUNet


@dataclass
class EnhancementResult:
    enhanced_signal: np.ndarray
    predicted_mask: np.ndarray
    noisy_mag: np.ndarray
    noisy_phase: np.ndarray
    filtered_mag: np.ndarray


def load_audio_file(file_path: str | Path, sr: int = 16000) -> np.ndarray:
    """Loads an audio file as mono float32 at the target sample rate."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in (".wav", ".mp3", ".flac", ".ogg"):
        audio, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        if file_sr != sr:
            mono = soxr.resample(mono, file_sr, sr)
        return mono.astype(np.float32)

    ffmpeg_command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "wav",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "pipe:1",
    ]
    result = subprocess.run(ffmpeg_command, check=True, capture_output=True)
    audio_buffer = io.BytesIO(result.stdout)
    audio, _ = sf.read(audio_buffer, dtype="float32", always_2d=False)
    return np.asarray(audio, dtype=np.float32)


def load_model(
    checkpoint_path: Optional[str | Path] = None,
    device: Optional[torch.device] = None,
    base_channels: int = 8,
) -> Tuple[TinyUNet, Dict[str, Any]]:
    """
    Loads TinyUNet from a checkpoint, returning the model and its DSP metadata.
    If checkpoint_path is None or doesn't exist, returns an uninitialized model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dsp_config: Dict[str, Any] = {
        "sr": 16000,
        "n_fft": 512,
        "hop_length": 256,
        "base_channels": base_channels,
    }

    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if path.exists():
            checkpoint = torch.load(path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                base_ch = checkpoint.get("base_channels", base_channels)
                dsp_config["base_channels"] = base_ch
                dsp_config["sr"] = checkpoint.get("sr", 16000)
                dsp_config["n_fft"] = checkpoint.get("n_fft", 512)
                dsp_config["hop_length"] = checkpoint.get("hop_length", 256)
                model = TinyUNet(in_channels=1, base_channels=base_ch).to(device)
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model = TinyUNet(in_channels=1, base_channels=base_channels).to(device)
                model.load_state_dict(checkpoint)
            model.eval()
            return model, dsp_config

    model = TinyUNet(in_channels=1, base_channels=base_channels).to(device)
    model.eval()
    return model, dsp_config


def predict_mask(
    model: TinyUNet,
    noisy_mag: np.ndarray,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Predicts a 2D ratio mask in [0, 1] from a noisy magnitude spectrogram."""
    if device is None:
        device = next(model.parameters()).device

    noisy_mag_input = preprocess_noisy_magnitude(noisy_mag)
    input_tensor = torch.from_numpy(noisy_mag_input).float().unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        predicted_mask = model(input_tensor)

    mask_np = predicted_mask.squeeze().cpu().numpy()
    return mask_np.astype(np.float32)


def enhance_signal(
    model: TinyUNet,
    noisy_signal: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 256,
    device: Optional[torch.device] = None,
) -> EnhancementResult:
    """Applies STFT, model mask prediction, spectral masking and iSTFT synthesis."""
    noisy_mag, noisy_phase = compute_stft(noisy_signal, n_fft=n_fft, hop_length=hop_length)
    mask = predict_mask(model, noisy_mag, device=device)
    filtered_mag = noisy_mag * mask
    enhanced_signal = compute_istft(
        filtered_mag,
        noisy_phase,
        hop_length=hop_length,
        length=len(noisy_signal),
    )

    return EnhancementResult(
        enhanced_signal=enhanced_signal,
        predicted_mask=mask,
        noisy_mag=noisy_mag,
        noisy_phase=noisy_phase,
        filtered_mag=filtered_mag,
    )
