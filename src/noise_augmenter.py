import io
import math
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

try:
    from .dsp_utils import is_silent_audio, mix_at_snr, sample_noise_window
except ImportError:
    from dsp_utils import is_silent_audio, mix_at_snr, sample_noise_window


def _load_base_noise_file(file_path: Path, sr: int = 16000) -> np.ndarray:
    """Carrega um arquivo de ruído mono em float32 na taxa de amostragem alvo."""
    file_extension = file_path.suffix.lower()

    if file_extension in (".wav", ".mp3", ".flac", ".ogg"):
        audio, file_sr = sf.read(str(file_path), dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        if file_sr != sr:
            mono = soxr.resample(mono, file_sr, sr)
        return np.asarray(mono, dtype=np.float32)

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        import imageio_ffmpeg

        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

    ffmpeg_command = [
        ffmpeg_bin,
        "-v",
        "error",
        "-i",
        str(file_path),
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


def _collect_base_noise_files(noise_dir: Path, prefix_exclude: str = "aug_noise_") -> list[Path]:
    """Coleta arquivos de ruído originais, ignorando arquivos já sintetizados (aug_noise_*)."""
    candidates: list[Path] = []
    for ext in ("*.wav", "*.webm", "*.mp3", "*.flac", "*.ogg"):
        for path in noise_dir.glob(ext):
            if not path.name.startswith(prefix_exclude):
                candidates.append(path)
    return sorted(candidates)


def _crossfade_concat(segments: list[np.ndarray], crossfade_samples: int) -> np.ndarray:
    """Concatena segmentos 1D usando crossfade suave (Hann) para transições não-estacionárias."""
    if not segments:
        return np.zeros(0, dtype=np.float32)
    if len(segments) == 1 or crossfade_samples <= 0:
        return np.concatenate(segments).astype(np.float32)

    fade_in = np.sin(np.linspace(0.0, math.pi / 2.0, crossfade_samples, dtype=np.float32)) ** 2
    fade_out = np.cos(np.linspace(0.0, math.pi / 2.0, crossfade_samples, dtype=np.float32)) ** 2

    output = segments[0].astype(np.float32, copy=True)
    for seg in segments[1:]:
        seg_f32 = seg.astype(np.float32, copy=True)
        overlap = min(crossfade_samples, len(output), len(seg_f32))
        if overlap <= 0:
            output = np.concatenate([output, seg_f32])
            continue

        if overlap < crossfade_samples:
            local_in = np.sin(np.linspace(0.0, math.pi / 2.0, overlap, dtype=np.float32)) ** 2
            local_out = np.cos(np.linspace(0.0, math.pi / 2.0, overlap, dtype=np.float32)) ** 2
        else:
            local_in = fade_in
            local_out = fade_out

        blended = output[-overlap:] * local_out + seg_f32[:overlap] * local_in
        output = np.concatenate([output[:-overlap], blended, seg_f32[overlap:]])

    return output.astype(np.float32)


def _build_concatenated_track(
    base_audios: list[np.ndarray],
    target_samples: int,
    sr: int,
    rng: random.Random,
) -> np.ndarray:
    """
    Constrói uma trilha de ruído concatenando fatias aleatórias de múltiplos áudios base,
    aplicando reversão temporal ocasional e variação de taxa (pitch/rate shift).
    """
    crossfade_samples = int(0.05 * sr)
    segments: list[np.ndarray] = []
    accumulated_samples = 0

    while accumulated_samples < target_samples + crossfade_samples:
        source_audio = rng.choice(base_audios)
        seg_duration_sec = rng.uniform(1.5, 4.0)
        raw_seg_samples = max(int(0.5 * sr), int(seg_duration_sec * sr))
        segment = sample_noise_window(source_audio, raw_seg_samples, rng=rng).copy()

        # Reversão temporal aleatória (30% de chance) para quebrar padrões direcionais
        if rng.random() < 0.30:
            segment = segment[::-1].copy()

        # Deslocamento espectral via reamostragem (0.85x a 1.15x)
        rate_factor = rng.uniform(0.85, 1.15)
        warped_sr = max(8000, int(round(sr * rate_factor)))
        if warped_sr != sr and len(segment) > 32:
            segment = np.asarray(soxr.resample(segment, sr, warped_sr), dtype=np.float32)

        # Normaliza RMS local do segmento para evitar degraus irreais antes do crossfade
        rms = float(np.sqrt(np.mean(segment**2) + 1e-8))
        target_seg_rms = rng.uniform(0.08, 0.25)
        segment = segment * (target_seg_rms / rms)

        segments.append(segment)
        accumulated_samples += max(1, len(segment) - crossfade_samples)

    concatenated = _crossfade_concat(segments, crossfade_samples=crossfade_samples)
    if len(concatenated) >= target_samples:
        return concatenated[:target_samples]
    return np.pad(concatenated, (0, target_samples - len(concatenated)), mode="wrap")


def _merge_noise_tracks(
    track_a: np.ndarray,
    track_b: np.ndarray,
    relative_snr_db: float,
) -> np.ndarray:
    """Superpõe duas trilhas de ruído independentes em uma SNR relativa."""
    mixed, _ = mix_at_snr(track_a, track_b, relative_snr_db)
    return mixed.astype(np.float32)


def _apply_spectral_and_envelope_modulation(
    audio: np.ndarray,
    sr: int,
    rng: random.Random,
) -> np.ndarray:
    """
    Aplica inclinação espectral (1st-order tilt filter) e modulação de envoltória LFO
    para simular fontes de ruído não-estacionárias em movimento.
    """
    alpha = rng.uniform(-0.55, 0.55)
    tilted = np.empty_like(audio, dtype=np.float32)
    tilted[0] = audio[0]
    tilted[1:] = audio[1:] - alpha * audio[:-1]

    num_samples = len(tilted)
    time_axis = np.arange(num_samples, dtype=np.float32) / float(sr)
    mod_freq_hz = rng.uniform(0.3, 2.0)
    mod_depth = rng.uniform(0.20, 0.65)
    mod_phase = rng.uniform(0.0, 2.0 * math.pi)
    envelope = 1.0 - mod_depth * 0.5 * (
        1.0 + np.sin(2.0 * math.pi * mod_freq_hz * time_axis + mod_phase)
    )

    modulated = (tilted * envelope.astype(np.float32)).astype(np.float32)

    peak = float(np.max(np.abs(modulated)))
    if peak > 1e-8:
        target_peak = rng.uniform(0.65, 0.92)
        modulated = modulated * (target_peak / peak)

    return np.clip(modulated, -1.0, 1.0).astype(np.float32)


def augment_noise_bank(
    noise_dir: str | Path = "data/noise",
    output_dir: str | Path | None = None,
    num_files: int = 90,
    duration_seconds: float = 10.0,
    sr: int = 16000,
    seed: int = 42,
    clean_existing: bool = True,
) -> list[Path]:
    """
    Expande o banco de ruídos combinando concatenação de fatias, superposição multi-fonte,
    deslocamento espectral e modulação de envoltória a partir dos arquivos originais.
    """
    noise_path = Path(noise_dir)
    out_path = Path(output_dir) if output_dir is not None else noise_path
    out_path.mkdir(parents=True, exist_ok=True)

    base_files = _collect_base_noise_files(noise_path, prefix_exclude="aug_noise_")
    if not base_files:
        raise ValueError(f"Nenhum arquivo de ruído base encontrado em {noise_path}.")

    base_audios: list[np.ndarray] = []
    for file_path in base_files:
        audio = _load_base_noise_file(file_path, sr=sr)
        if not is_silent_audio(audio):
            base_audios.append(audio)

    if not base_audios:
        raise ValueError(f"Todos os arquivos de ruído base em {noise_path} são silenciosos.")

    if clean_existing:
        for old_aug in out_path.glob("aug_noise_*.wav"):
            old_aug.unlink()

    rng = random.Random(seed)
    target_samples = max(1, int(round(duration_seconds * sr)))
    created_files: list[Path] = []

    print(
        f"[i] Gerando {num_files} arquivos de ruído não-estacionário ({duration_seconds:.1f}s cada) "
        f"a partir de {len(base_audios)} fontes base em {out_path}..."
    )

    for idx in range(1, num_files + 1):
        track_a = _build_concatenated_track(
            base_audios, target_samples=target_samples, sr=sr, rng=rng
        )

        # Em 75% dos casos, superpõe uma segunda trilha independente em SNR relativa aleatória
        if len(base_audios) >= 2 and rng.random() < 0.75:
            track_b = _build_concatenated_track(
                base_audios, target_samples=target_samples, sr=sr, rng=rng
            )
            rel_snr = rng.uniform(-5.0, 5.0)
            merged = _merge_noise_tracks(track_a, track_b, relative_snr_db=rel_snr)
        else:
            merged = track_a

        synthesized = _apply_spectral_and_envelope_modulation(merged, sr=sr, rng=rng)
        out_file = out_path / f"aug_noise_{idx:04d}.wav"
        sf.write(str(out_file), synthesized, sr, subtype="FLOAT")
        created_files.append(out_file)

    print(f"[+] {len(created_files)} arquivos de ruído expandidos salvos em: {out_path}")
    return created_files
