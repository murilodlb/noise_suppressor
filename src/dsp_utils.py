import numpy as np
import librosa
from typing import Tuple
import matplotlib.pyplot as plt

def compute_stft(audio: np.ndarray, n_fft: int = 512, hop_length: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula a Transformada de Fourier de Curto Termo (STFT).
    Retorna a magnitude e a fase separadamente.
    """
    # A STFT gera uma matriz complexa (frequências x frames de tempo)
    stft_complex = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    
    # Conversão de coordenadas polares: magnitude (raio) e fase (ângulo)
    magnitude, phase = librosa.magphase(stft_complex)
    return magnitude, phase

def compute_istft(magnitude: np.ndarray, phase: np.ndarray, hop_length: int = 256) -> np.ndarray:
    """
    Calcula a Transformada Inversa (iSTFT).
    Recebe a magnitude filtrada e a fase original para reconstruir o sinal.
    """
    # Reconstrução do número complexo: magnitude * exp(j * fase)
    stft_complex = magnitude * phase
    
    # Reconstrói o sinal no domínio do tempo usando o algoritmo de Overlap-Add
    audio = librosa.istft(stft_complex, hop_length=hop_length)
    return audio

def compute_irm(clean_mag: np.ndarray, noise_mag: np.ndarray, beta: float = 0.5) -> np.ndarray:
    """
    Calcula a Máscara de Proporção Ideal (IRM) baseada na energia ao quadrado.
    Fórmula: IRM = (|S|^2 / (|S|^2 + |N|^2))^beta
    """
    # Eleva as matrizes ao quadrado para obter a densidade espectral de potência
    clean_power = np.square(clean_mag)
    noise_power = np.square(noise_mag)
    
    # Evita divisão por zero adicionando um épsilon minúsculo
    eps = 1e-8
    irm = (clean_power / (clean_power + noise_power + eps)) ** beta
    
    # Os valores da máscara são garantidamente contínuos entre 0 e 1
    return irm

def plot_time_signals(clean: np.ndarray, noisy: np.ndarray, enhanced: np.ndarray, sr: int, output_path: str = "time_domain.png") -> None:
    """
    Plota e salva os sinais no domínio do tempo (Voz limpa, Ruidosa e Filtrada).
    """
    # Cria o eixo temporal em segundos baseado na taxa de amostragem
    t = np.linspace(0, len(clean) / sr, len(clean))
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    axes[0].plot(t, clean, color='green', linewidth=0.7)
    axes[0].set_title('Sinal Limpo (Ground Truth)')
    axes[0].set_ylabel('Amplitude')
    
    axes[1].plot(t, noisy, color='red', linewidth=0.7)
    axes[1].set_title('Sinal com Ruído (Entrada da Rede)')
    axes[1].set_ylabel('Amplitude')
    
    axes[2].plot(t, enhanced, color='blue', linewidth=0.7)
    axes[2].set_title('Sinal Filtrado (Saída da iSTFT)')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_xlabel('Tempo (s)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Gráfico do domínio do tempo salvo em: {output_path}")

def plot_spectrograms(clean_mag: np.ndarray, noisy_mag: np.ndarray, enhanced_mag: np.ndarray, sr: int, output_path: str = "spectrograms.png") -> None:
    """
    Plota e salva os espectrogramas em escala de Decibéis (dB).
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, sharey=True)
    
    # Converte magnitude para dB (escala logarítmica) para visualização auditiva correta
    clean_db = librosa.amplitude_to_db(clean_mag, ref=np.max)
    noisy_db = librosa.amplitude_to_db(noisy_mag, ref=np.max)
    enhanced_db = librosa.amplitude_to_db(enhanced_mag, ref=np.max)
    
    # Plot 1: Limpo
    img1 = librosa.display.specshow(clean_db, sr=sr, ax=axes[0], y_axis='log', x_axis='time', cmap='magma')
    axes[0].set_title('Espectrograma Limpo (dB)')
    
    # Plot 2: Ruidoso
    img2 = librosa.display.specshow(noisy_db, sr=sr, ax=axes[1], y_axis='log', x_axis='time', cmap='magma')
    axes[1].set_title('Espectrograma com Ruído (dB)')
    
    # Plot 3: Filtrado
    img3 = librosa.display.specshow(enhanced_db, sr=sr, ax=axes[2], y_axis='log', x_axis='time', cmap='magma')
    axes[2].set_title('Espectrograma Filtrado (dB)')
    axes[2].set_xlabel('Tempo (s)')
    
    # Adiciona barra de cores compartilhada
    fig.colorbar(img3, ax=axes, format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Espectrogramas salvos em: {output_path}")
