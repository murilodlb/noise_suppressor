import random
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from .dsp_utils import (
    compute_irm,
    compute_istft,
    compute_stft,
    crop_or_pad_audio,
    mix_at_snr,
    plot_spectrograms,
    plot_time_signals,
)
from .inference import enhance_signal, load_audio_file, load_model
from .logger import configure_logging, get_logger
from .metrics import compute_signal_metrics

logger = get_logger(__name__)


def _pick_holdout_clean_file(
    clean_files: list[Path], validation_ratio: float = 0.2, seed: int = 42
) -> Path:
    if len(clean_files) < 2:
        raise ValueError("É preciso ter pelo menos 2 arquivos limpos para selecionar um holdout.")

    shuffled_files = clean_files[:]
    random.Random(seed).shuffle(shuffled_files)
    holdout_count = max(1, int(len(shuffled_files) * validation_ratio))
    holdout_files = shuffled_files[:holdout_count]
    return holdout_files[0]


def run_demo(
    clean_path: str | Path | None = None,
    noise_path: str | Path | None = None,
    checkpoint_path: str | Path = "checkpoints/tinyunet.pth",
    output_dir: str | Path = "data",
    seed: int = 42,
    snr_db: float = 5.0,
    model_name: str | None = None,
) -> dict[str, Any]:
    logger.info("--- STARTING PROOF OF CONCEPT DEMO ---")

    sr = 16000
    n_fft = 512
    hop_length = 256
    target_seconds = 2.0
    target_samples = int(sr * target_seconds)

    out_root = Path(output_dir)
    data_root = Path("data")
    clean_dir = data_root / "clean"
    noise_dir = data_root / "noise"

    clean_files = sorted(
        list(clean_dir.glob("*.wav"))
        + list(clean_dir.glob("*.mp3"))
        + list(clean_dir.glob("*.webm"))
    )
    noise_files = sorted(
        list(noise_dir.glob("*.wav"))
        + list(noise_dir.glob("*.mp3"))
        + list(noise_dir.glob("*.webm"))
    )

    # 1. Carregar áudios
    def fix_length(sig: np.ndarray) -> np.ndarray:
        return crop_or_pad_audio(sig, target_samples)

    if clean_path is not None and Path(clean_path).exists():
        selected_clean = Path(clean_path)
    elif (data_root / "test" / "test.mp3").exists():
        selected_clean = data_root / "test" / "test.mp3"
    elif clean_files:
        selected_clean = _pick_holdout_clean_file(clean_files, validation_ratio=0.2, seed=seed)
    else:
        selected_clean = None

    if noise_path is not None and Path(noise_path).exists():
        selected_noise = Path(noise_path)
    elif noise_files:
        rng = random.Random(seed)
        selected_noise = rng.choice(noise_files)
    else:
        selected_noise = None

    if selected_clean is not None and selected_noise is not None:
        logger.info("[1] Loading real speech and noise samples...")
        logger.info(" - Test clean speech: %s", selected_clean.name)
        logger.info(" - Test noise sample: %s", selected_noise.name)

        clean_signal = fix_length(load_audio_file(selected_clean, sr=sr))
        noise_signal = fix_length(load_audio_file(selected_noise, sr=sr))
        # noise_signal passa a ser o ruído escalado (usado na IRM de fallback)
        noisy_signal, noise_signal = mix_at_snr(clean_signal, noise_signal, snr_db)
        logger.info(" - Target mix SNR: %.1f dB", snr_db)
        logger.info(" - Signal shape: %s", noisy_signal.shape)
    else:
        logger.warning("[1] Speech/noise files not found; using synthetic fallback signal...")
        t = np.linspace(0, target_seconds, target_samples, endpoint=False)
        clean_signal = (
            0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)
        ).astype(np.float32)
        noise_signal = np.random.normal(0, 0.2, len(t)).astype(np.float32)
        noisy_signal, noise_signal = mix_at_snr(clean_signal, noise_signal, snr_db)
        logger.info(" - Target mix SNR: %.1f dB", snr_db)
        logger.info(" - Signal shape: %s", noisy_signal.shape)

    # 2. STFT
    logger.info("[2] Applying STFT (Time Domain -> Frequency Domain)...")
    clean_mag, _ = compute_stft(clean_signal, n_fft=n_fft, hop_length=hop_length)
    noise_mag, _ = compute_stft(noise_signal, n_fft=n_fft, hop_length=hop_length)
    noisy_mag, noisy_phase = compute_stft(noisy_signal, n_fft=n_fft, hop_length=hop_length)
    logger.info(" - Magnitude matrix (Freqs x Frames): %s", noisy_mag.shape)

    # 3. Carregar modelo e inferência
    ckpt_path = Path(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ckpt_path.exists():
        logger.info("[3] Loading trained checkpoint: %s", ckpt_path)
        model, dsp_cfg = load_model(ckpt_path, device=device, model_name=model_name)
        enhancement = enhance_signal(
            model, noisy_signal, n_fft=n_fft, hop_length=hop_length, device=device
        )
        enhanced_signal = enhancement.enhanced_signal
        mask_np = enhancement.predicted_mask
        filtered_mag = enhancement.filtered_mag
        mask_source = f"modelo treinado ({dsp_cfg.get('model_name', 'tinyunet')})"
    else:
        logger.warning("[3] No checkpoint found; applying Oracle IRM fallback...")
        mask_np = compute_irm(clean_mag, noise_mag, beta=0.5)
        filtered_mag = noisy_mag * mask_np
        enhanced_signal = compute_istft(
            filtered_mag, noisy_phase, hop_length=hop_length, length=len(noisy_signal)
        )
        mask_source = "máscara ideal (IRM)"

    logger.info(" - Predicted mask shape (Freq, Time): %s", mask_np.shape)
    logger.info(
        " - Mask values range [0, 1]: %.4f to %.4f", float(mask_np.min()), float(mask_np.max())
    )
    logger.info(" - Reconstructed signal shape: %s", enhanced_signal.shape)
    logger.info(" - Mask source: %s", mask_source)

    # 4. Avaliação das métricas do demo
    noisy_metrics = compute_signal_metrics(clean_signal, noisy_signal, sr=sr)
    enhanced_metrics = compute_signal_metrics(clean_signal, enhanced_signal, sr=sr)

    logger.info("--- DEMO METRICS ---")
    logger.info(
        " - SI-SDR: Noisy %.2f dB -> Enhanced %.2f dB (Δ %+.2f dB)",
        noisy_metrics["si_sdr"],
        enhanced_metrics["si_sdr"],
        enhanced_metrics["si_sdr"] - noisy_metrics["si_sdr"],
    )
    logger.info(
        " - PESQ:   Noisy %.2f    -> Enhanced %.2f    (Δ %+.2f)",
        noisy_metrics["pesq"],
        enhanced_metrics["pesq"],
        enhanced_metrics["pesq"] - noisy_metrics["pesq"],
    )
    logger.info(
        " - STOI:   Noisy %.3f   -> Enhanced %.3f   (Δ %+.3f)",
        noisy_metrics["stoi"],
        enhanced_metrics["stoi"],
        enhanced_metrics["stoi"] - noisy_metrics["stoi"],
    )

    # 5. Visualização e Salvamento de Gráficos e Áudios
    logger.info("[5] Generating plots and exporting WAV audio...")
    enhanced_mag, _ = compute_stft(enhanced_signal, n_fft=n_fft, hop_length=hop_length)

    imgs_dir = out_root / "imgs"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    plot_time_signals(
        clean_signal,
        noisy_signal,
        enhanced_signal,
        sr=sr,
        output_path=str(imgs_dir / "time_domain.png"),
    )
    plot_spectrograms(
        clean_mag,
        noisy_mag,
        enhanced_mag,
        sr=sr,
        output_path=str(imgs_dir / "spectrograms.png"),
        hop_length=hop_length,
    )

    audio_dir = out_root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_dir / "clean.wav"), clean_signal, sr)
    sf.write(str(audio_dir / "noisy.wav"), noisy_signal, sr)
    sf.write(str(audio_dir / "enhanced.wav"), enhanced_signal, sr)
    logger.info("Saved demo WAV files to: %s", audio_dir)
    logger.info("--- DEMO COMPLETED SUCCESSFULLY ---")

    return {
        "clean_signal": clean_signal,
        "noisy_signal": noisy_signal,
        "enhanced_signal": enhanced_signal,
        "mask_source": mask_source,
        "noisy_metrics": noisy_metrics,
        "enhanced_metrics": enhanced_metrics,
    }


def main():
    configure_logging()
    run_demo()


if __name__ == "__main__":
    main()
