import os
import glob
import random
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from typing import Tuple, List
from dsp_utils import compute_stft, compute_irm

class NoiseSuppressionDataset(Dataset):
    """
    Dataset customizado que carrega áudios, mistura-os com SNR aleatório,
    e retorna o espectrograma sujo e a máscara IRM correspondente.
    """
    def __init__(self, clean_dir: str, noise_dir: str, sr: int = 16000, n_fft: int = 512, hop_length: int = 256, snr_range: Tuple[float, float] = (0.0, 10.0)):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.snr_range = snr_range
        
        self.clean_files: List[str] = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))
        self.noise_files: List[str] = sorted(glob.glob(os.path.join(noise_dir, "*.wav")))

    def __len__(self) -> int:
        return len(self.clean_files)

    def _mix_at_snr(self, clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
        """Mistura o sinal limpo e o ruído baseado em uma SNR alvo."""
        # Se o ruído for maior que o limpo, trunca o ruído
        if len(noise) > len(clean):
            noise = noise[:len(clean)]
        else:
            # Se for menor, preenche com zeros (padding)
            noise = np.pad(noise, (0, len(clean) - len(noise)), 'constant')
            
        # Calcula a potência (energia) RMS de ambos os sinais
        clean_power = np.mean(clean ** 2)
        noise_power = np.mean(noise ** 2)
        
        # Calcula o fator de escala para o ruído atender à SNR desejada
        snr_linear = 10 ** (snr_db / 10)
        scale = np.sqrt(clean_power / (noise_power * snr_linear + 1e-8))
        
        # Mistura no domínio do tempo
        noisy_signal = clean + noise * scale
        
        # Normalização para evitar clipping (limita entre -1 e 1)
        max_val = np.max(np.abs(noisy_signal))
        if max_val > 1.0:
            noisy_signal = noisy_signal / max_val
            
        return noisy_signal

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Carrega os áudios
        clean_path = self.clean_files[idx]
        clean_audio, _ = librosa.load(clean_path, sr=self.sr, mono=True)
        
        noise_path = random.choice(self.noise_files)
        noise_audio, _ = librosa.load(noise_path, sr=self.sr, mono=True)
        
        # 2. Sorteia SNR e mistura
        snr_db = random.uniform(self.snr_range[0], self.snr_range[1])
        noisy_audio = self._mix_at_snr(clean_audio, noise_audio, snr_db)
        
        # Calcula o ruído real adicionado (necessário para a IRM)
        noise_audio_matched = noisy_audio - clean_audio
        
        # 3. Transforma para o domínio da frequência
        clean_mag, _ = compute_stft(clean_audio, self.n_fft, self.hop_length)
        noise_mag, _ = compute_stft(noise_audio_matched, self.n_fft, self.hop_length)
        noisy_mag, _ = compute_stft(noisy_audio, self.n_fft, self.hop_length)
        
        # 4. Calcula a Máscara de Proporção Ideal (Gabarito)
        irm = compute_irm(clean_mag, noise_mag)
        
        # 5. Conversão para Tensores PyTorch
        # Adiciona a dimensão de canal (Batch, Channel, Height, Width)
        noisy_mag_tensor = torch.from_numpy(noisy_mag).float().unsqueeze(0)
        irm_tensor = torch.from_numpy(irm).float().unsqueeze(0)
        
        return noisy_mag_tensor, irm_tensor
