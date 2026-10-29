import csv
import glob
import io
import os
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import soxr
import torch
from torch.utils.data import Dataset

try:
    from .dsp_utils import (
        compute_irm,
        compute_stft,
        crop_or_pad_audio,
        mix_at_snr,
        preprocess_noisy_magnitude,
    )
except ImportError:
    from dsp_utils import (
        compute_irm,
        compute_stft,
        crop_or_pad_audio,
        mix_at_snr,
        preprocess_noisy_magnitude,
    )


def _print_export_progress(current: int, total: int, prefix: str = "Preprocess") -> None:
    bar_length = 30
    filled_length = int(bar_length * current // total) if total else bar_length
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    pct = (100.0 * current / total) if total else 100.0
    print(f"\r{prefix} |{bar}| {current}/{total} ({pct:5.1f}%)", end="", flush=True)


class NoiseSuppressionDataset(Dataset):
    """
    Dataset customizado que carrega áudios, soma limpo + ruído,
    e retorna o espectrograma sujo e a máscara IRM correspondente.
    """
    def __init__(self, clean_dir: str, noise_dir: str, sr: int = 16000, n_fft: int = 512, hop_length: int = 256, snr_range: Tuple[float, float] = (-5.0, 15.0), target_seconds: float = 2.0):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.snr_range = snr_range
        self.target_seconds = target_seconds
        self.target_num_samples = int(sr * target_seconds)

        self.clean_files: List[str] = self._collect_audio_files(clean_dir)
        self.noise_files: List[str] = self._collect_audio_files(noise_dir)
        self._noise_cache: Dict[str, np.ndarray] = {}

    def _collect_audio_files(self, directory: str) -> List[str]:
        files: List[str] = []
        for extension in ("*.wav", "*.webm", "*.mp3"):
            files.extend(glob.glob(os.path.join(directory, extension)))
        return sorted(files)

    def _load_audio(self, file_path: str) -> np.ndarray:
        file_extension = Path(file_path).suffix.lower()

        if file_extension in (".wav", ".mp3", ".flac", ".ogg"):
            audio, file_sr = sf.read(file_path, dtype="float32", always_2d=True)
            mono = audio.mean(axis=1)
            if file_sr != self.sr:
                mono = soxr.resample(mono, file_sr, self.sr)
            return mono.astype(np.float32)

        ffmpeg_command = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            file_path,
            "-f",
            "wav",
            "-ac",
            "1",
            "-ar",
            str(self.sr),
            "pipe:1",
        ]

        result = subprocess.run(ffmpeg_command, check=True, capture_output=True)
        audio_buffer = io.BytesIO(result.stdout)
        audio, _ = sf.read(audio_buffer, dtype="float32", always_2d=False)
        return np.asarray(audio, dtype=np.float32)

    def _get_cached_noise_audio(self, noise_file: str) -> np.ndarray:
        if noise_file not in self._noise_cache:
            self._noise_cache[noise_file] = self._fix_length(self._load_audio(noise_file))
        return self._noise_cache[noise_file]

    def _fix_length(self, audio: np.ndarray) -> np.ndarray:
        return crop_or_pad_audio(audio, self.target_num_samples)

    def __len__(self) -> int:
        return len(self.clean_files)

    def _add_noise(self, clean: np.ndarray, noise: np.ndarray, snr_db: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Mistura limpo + ruído na SNR alvo (dB) via mix_at_snr.
        Se snr_db for None, sorteia uniformemente de self.snr_range.
        Retorna (noisy, scaled_noise, snr_db).
        """
        if snr_db is None:
            snr_db = random.uniform(*self.snr_range)
        noisy, scaled_noise = mix_at_snr(clean, noise, snr_db)
        return noisy, scaled_noise, float(snr_db)

    def export_noisy_dataset(self, output_dir: str, samples_per_clean: int = 1, save_clean_copies: bool = True) -> Path:
        """Gera um dataset ruidoso em disco a partir das pastas de limpo e ruído."""
        if not self.clean_files:
            raise ValueError("Nenhum arquivo limpo encontrado para exportação.")

        if not self.noise_files:
            raise ValueError("Nenhum arquivo de ruído encontrado para exportação.")

        output_path = Path(output_dir)
        clean_output_dir = output_path / "clean"
        noisy_output_dir = output_path / "noisy"
        clean_output_dir.mkdir(parents=True, exist_ok=True)
        noisy_output_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = output_path / "metadata.csv"
        fieldnames = ["clean_file", "noise_file", "noisy_file", "sample_index", "snr_db"]

        total_samples = len(self.clean_files) * samples_per_clean
        sample_counter = 0

        with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for clean_file in self.clean_files:
                clean_audio = self._fix_length(self._load_audio(clean_file))
                clean_stem = Path(clean_file).stem

                for sample_index in range(samples_per_clean):
                    noise_file = random.choice(self.noise_files)
                    noise_audio = self._get_cached_noise_audio(noise_file)
                    noisy_audio, _, snr_db = self._add_noise(clean_audio, noise_audio)

                    output_name = f"{clean_stem}__noise-{Path(noise_file).stem}__{sample_index:03d}.wav"
                    noisy_path = noisy_output_dir / output_name
                    sf.write(noisy_path, noisy_audio, self.sr, subtype="FLOAT")

                    if save_clean_copies:
                        clean_path = clean_output_dir / output_name
                        sf.write(clean_path, clean_audio, self.sr, subtype="FLOAT")

                    writer.writerow(
                        {
                            "clean_file": clean_file,
                            "noise_file": noise_file,
                            "noisy_file": str(noisy_path),
                            "sample_index": sample_index,
                            "snr_db": round(snr_db, 4),
                        }
                    )
                    sample_counter += 1
                    if sample_counter % 25 == 0 or sample_counter == total_samples:
                        _print_export_progress(sample_counter, total_samples, prefix="Export Noisy")

        if total_samples > 0:
            print()

        return metadata_path

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Carrega os áudios
        clean_path = self.clean_files[idx]
        clean_audio = self._fix_length(self._load_audio(clean_path))
        
        noise_path = random.choice(self.noise_files)
        noise_audio = self._get_cached_noise_audio(noise_path)
        
        # 2. Soma limpa + ruído no domínio do tempo
        # SNR sorteada a cada chamada (augmentation on-the-fly)
        # O ruído escalado retornado é o ruído real adicionado (necessário para a IRM)
        noisy_audio, noise_audio_matched, _ = self._add_noise(clean_audio, noise_audio)
        
        # 3. Transforma para o domínio da frequência
        clean_mag, _ = compute_stft(clean_audio, self.n_fft, self.hop_length)
        noise_mag, _ = compute_stft(noise_audio_matched, self.n_fft, self.hop_length)
        noisy_mag, _ = compute_stft(noisy_audio, self.n_fft, self.hop_length)
        noisy_mag_input = preprocess_noisy_magnitude(noisy_mag)
        
        # 4. Calcula a Máscara de Proporção Ideal (Gabarito)
        irm = compute_irm(clean_mag, noise_mag)
        
        # 5. Conversão para Tensores PyTorch
        # Adiciona a dimensão de canal (Batch, Channel, Height, Width)
        noisy_mag_tensor = torch.from_numpy(noisy_mag_input).float().unsqueeze(0)
        irm_tensor = torch.from_numpy(irm).float().unsqueeze(0)
        
        return noisy_mag_tensor, irm_tensor


class PrecomputedNoiseSuppressionDataset(Dataset):
    """Dataset de leitura rápida que carrega pares (noisy_mag, irm) salvos em .pt."""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        metadata_path = self.dataset_dir / "metadata.csv"

        if metadata_path.exists():
            # A ordem dos arquivos segue as linhas do metadata.csv, garantindo que o índice i
            # corresponda à linha i usada por _split_indices_by_clean_source (sem leakage).
            with metadata_path.open("r", newline="", encoding="utf-8") as csv_file:
                self.sample_files = [Path(row["sample_file"]) for row in csv.DictReader(csv_file)]
            missing_files = [p for p in self.sample_files if not p.exists()]
            if missing_files:
                raise FileNotFoundError(
                    f"{len(missing_files)} arquivos .pt listados em {metadata_path} não existem "
                    f"(ex.: {missing_files[0]}). Rode o preprocess novamente."
                )
        else:
            # Comportamento legado sem metadata
            self.sample_files = sorted(self.dataset_dir.glob("*.pt"))

    def __len__(self) -> int:
        return len(self.sample_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = torch.load(self.sample_files[idx], map_location="cpu")
        return sample["noisy_mag"], sample["irm"]

    @staticmethod
    def export_precomputed_dataset(
        clean_dir: str,
        noise_dir: str,
        output_dir: str,
        sr: int = 16000,
        n_fft: int = 512,
        hop_length: int = 256,
        target_seconds: float = 2.0,
        samples_per_clean: int = 1,
        snr_range: Tuple[float, float] = (-5.0, 15.0),
    ) -> Path:
        """Gera um dataset pré-computado em .pt com noisy_mag e irm, mixando na SNR sorteada de snr_range."""
        source_dataset = NoiseSuppressionDataset(
            clean_dir=clean_dir,
            noise_dir=noise_dir,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            target_seconds=target_seconds,
            snr_range=snr_range,
        )

        if not source_dataset.clean_files:
            raise ValueError(f"Nenhum arquivo limpo encontrado em {clean_dir}.")

        if not source_dataset.noise_files:
            raise ValueError(f"Nenhum arquivo de ruído encontrado em {noise_dir}.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        total_samples = len(source_dataset.clean_files) * samples_per_clean
        print(
            f"[i] Pré-computando {total_samples} amostras "
            f"({len(source_dataset.clean_files)} limpos x {samples_per_clean}, "
            f"SNR em [{snr_range[0]:.1f}, {snr_range[1]:.1f}] dB)..."
        )

        # Pré-carrega os arquivos de ruído em memória uma única vez
        for nf in source_dataset.noise_files:
            source_dataset._get_cached_noise_audio(nf)

        metadata_path = output_path / "metadata.csv"
        temp_metadata_path = output_path / "metadata.csv.tmp"
        fieldnames = ["sample_file", "clean_file", "noise_file", "sample_index", "snr_db"]
        written_files: set[Path] = set()

        with temp_metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            sample_counter = 0
            for clean_file in source_dataset.clean_files:
                clean_audio = source_dataset._fix_length(source_dataset._load_audio(clean_file))
                clean_stem = Path(clean_file).stem
                clean_mag, _ = compute_stft(clean_audio, n_fft, hop_length)

                for sample_index in range(samples_per_clean):
                    noise_file = random.choice(source_dataset.noise_files)
                    noise_audio = source_dataset._get_cached_noise_audio(noise_file)
                    noisy_audio, noise_audio_matched, snr_db = source_dataset._add_noise(clean_audio, noise_audio)

                    noise_mag, _ = compute_stft(noise_audio_matched, n_fft, hop_length)
                    noisy_mag, _ = compute_stft(noisy_audio, n_fft, hop_length)
                    irm = compute_irm(clean_mag, noise_mag)
                    noisy_mag_input = preprocess_noisy_magnitude(noisy_mag)

                    sample_file = output_path / f"{clean_stem}__noise-{Path(noise_file).stem}__{sample_index:03d}.pt"
                    torch.save(
                        {
                            "noisy_mag": torch.from_numpy(noisy_mag_input).float().unsqueeze(0),
                            "irm": torch.from_numpy(irm).float().unsqueeze(0),
                        },
                        sample_file,
                    )
                    written_files.add(sample_file.resolve())

                    writer.writerow(
                        {
                            "sample_file": str(sample_file),
                            "clean_file": clean_file,
                            "noise_file": noise_file,
                            "sample_index": sample_index,
                            "snr_db": round(snr_db, 4),
                        }
                    )
                    sample_counter += 1
                    if sample_counter % 25 == 0 or sample_counter == total_samples:
                        _print_export_progress(sample_counter, total_samples, prefix="Preprocess")

        if total_samples > 0:
            print()

        temp_metadata_path.replace(metadata_path)

        # Remove apenas tensores órfãos que não fazem parte da nova exportação
        stale_files = [p for p in output_path.glob("*.pt") if p.resolve() not in written_files]
        for stale_file in stale_files:
            stale_file.unlink()
        if stale_files:
            print(f"[i] {len(stale_files)} arquivos .pt antigos removidos de {output_path}")

        return metadata_path
