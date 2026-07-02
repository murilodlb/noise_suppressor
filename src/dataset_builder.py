import os
import glob
import random
import csv
import io
import subprocess
from pathlib import Path
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from typing import Tuple, List
import soundfile as sf

try:
    from .dsp_utils import compute_stft, compute_irm, preprocess_noisy_magnitude
except ImportError:
    from dsp_utils import compute_stft, compute_irm, preprocess_noisy_magnitude

class NoiseSuppressionDataset(Dataset):
    """
    Dataset customizado que carrega áudios, soma limpo + ruído,
    e retorna o espectrograma sujo e a máscara IRM correspondente.
    """
    def __init__(self, clean_dir: str, noise_dir: str, sr: int = 16000, n_fft: int = 512, hop_length: int = 256, snr_range: Tuple[float, float] = (0.0, 10.0), target_seconds: float = 2.0):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.snr_range = snr_range
        self.target_num_samples = int(sr * target_seconds)

        self.clean_files: List[str] = self._collect_audio_files(clean_dir)
        self.noise_files: List[str] = self._collect_audio_files(noise_dir)

    def _collect_audio_files(self, directory: str) -> List[str]:
        files: List[str] = []
        for extension in ("*.wav", "*.webm", "*.mp3"):
            files.extend(glob.glob(os.path.join(directory, extension)))
        return sorted(files)

    def _load_audio(self, file_path: str) -> np.ndarray:
        file_extension = Path(file_path).suffix.lower()

        if file_extension == ".wav":
            audio, _ = librosa.load(file_path, sr=self.sr, mono=True)
            return audio

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
        audio, _ = librosa.load(audio_buffer, sr=self.sr, mono=True)
        return audio

    def _fix_length(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) > self.target_num_samples:
            return audio[: self.target_num_samples]

        if len(audio) < self.target_num_samples:
            return np.pad(audio, (0, self.target_num_samples - len(audio)), mode="constant")

        return audio

    def __len__(self) -> int:
        return len(self.clean_files)

    def _add_noise(self, clean: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """Soma diretamente o sinal limpo com o ruído, ajustando apenas o tamanho."""
        if len(noise) > len(clean):
            noise = noise[:len(clean)]
        else:
            # Se for menor, preenche com zeros (padding)
            noise = np.pad(noise, (0, len(clean) - len(noise)), 'constant')

        return clean + noise

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
        fieldnames = ["clean_file", "noise_file", "noisy_file", "sample_index"]

        with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for clean_file in self.clean_files:
                clean_audio = self._fix_length(self._load_audio(clean_file))
                clean_stem = Path(clean_file).stem

                for sample_index in range(samples_per_clean):
                    noise_file = random.choice(self.noise_files)
                    noise_audio = self._fix_length(self._load_audio(noise_file))
                    noisy_audio = self._add_noise(clean_audio, noise_audio)

                    output_name = f"{clean_stem}__noise-{Path(noise_file).stem}__{sample_index:03d}.wav"
                    noisy_path = noisy_output_dir / output_name
                    sf.write(noisy_path, noisy_audio, self.sr)

                    if save_clean_copies:
                        clean_path = clean_output_dir / output_name
                        sf.write(clean_path, clean_audio, self.sr)

                    writer.writerow(
                        {
                            "clean_file": clean_file,
                            "noise_file": noise_file,
                            "noisy_file": str(noisy_path),
                            "sample_index": sample_index,
                        }
                    )

        return metadata_path

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Carrega os áudios
        clean_path = self.clean_files[idx]
        clean_audio = self._fix_length(self._load_audio(clean_path))
        
        noise_path = random.choice(self.noise_files)
        noise_audio = self._fix_length(self._load_audio(noise_path))
        
        # 2. Soma limpa + ruído no domínio do tempo
        noisy_audio = self._add_noise(clean_audio, noise_audio)
        
        # Calcula o ruído real adicionado (necessário para a IRM)
        noise_audio_matched = noisy_audio - clean_audio
        
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
    ) -> Path:
        """Gera um dataset pré-computado em .pt com noisy_mag e irm."""
        source_dataset = NoiseSuppressionDataset(
            clean_dir=clean_dir,
            noise_dir=noise_dir,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            target_seconds=target_seconds,
        )

        if not source_dataset.clean_files:
            raise ValueError(f"Nenhum arquivo limpo encontrado em {clean_dir}.")

        if not source_dataset.noise_files:
            raise ValueError(f"Nenhum arquivo de ruído encontrado em {noise_dir}.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        metadata_path = output_path / "metadata.csv"
        fieldnames = ["sample_file", "clean_file", "noise_file", "sample_index"]

        with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            sample_counter = 0
            for clean_file in source_dataset.clean_files:
                clean_audio = source_dataset._fix_length(source_dataset._load_audio(clean_file))
                clean_stem = Path(clean_file).stem
                clean_mag, _ = compute_stft(clean_audio, n_fft, hop_length)

                for sample_index in range(samples_per_clean):
                    noise_file = random.choice(source_dataset.noise_files)
                    noise_audio = source_dataset._fix_length(source_dataset._load_audio(noise_file))
                    noisy_audio = source_dataset._add_noise(clean_audio, noise_audio)
                    noise_audio_matched = noisy_audio - clean_audio

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

                    writer.writerow(
                        {
                            "sample_file": str(sample_file),
                            "clean_file": clean_file,
                            "noise_file": noise_file,
                            "sample_index": sample_index,
                        }
                    )
                    sample_counter += 1

        return metadata_path
