import torch
import numpy as np
from model import TinyUNet
from dsp_utils import compute_stft, compute_istft, plot_time_signals, plot_spectrograms

def main():
    print("--- INICIANDO PROVA DE CONCEITO (PoC) ---")
    
    # Hiperparâmetros de PDS
    SR = 16000
    N_FFT = 512
    HOP_LENGTH = 256
    
    # 1. Simulação de dado no Domínio do Tempo (Tensão/Ampitude vs Tempo)
    print("[1] Gerando sinal no domínio do tempo (dummy audio)...")
    duration = 2.0
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    # Simula voz (combinação de senoides) e ruído (ruído branco gaussiano)
    clean_signal = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)
    noise_signal = np.random.normal(0, 0.2, len(t))
    noisy_signal = clean_signal + noise_signal
    print(f" - Forma do sinal (amostras): {noisy_signal.shape}")
    
    # 2. Análise: Domínio do Tempo -> Domínio da Frequência (Espectrograma)
    print("[2] Aplicando STFT (Domínio do Tempo -> Domínio da Frequência)...")
    clean_mag, _ = compute_stft(clean_signal, n_fft=N_FFT, hop_length=HOP_LENGTH)
    noisy_mag, noisy_phase = compute_stft(noisy_signal, n_fft=N_FFT, hop_length=HOP_LENGTH)
    print(f" - Matriz de Magnitude (Frequências x Frames): {noisy_mag.shape}")
    
    # Prepara tensor para a U-Net (Batch=1, Channel=1, Freq, Time)
    input_tensor = torch.from_numpy(noisy_mag).float().unsqueeze(0).unsqueeze(0)
    
    # 3. Inferência: Passagem pela U-Net (CNN)
    print("[3] Realizando forward pass na TinyU-Net...")
    model = TinyUNet(in_channels=1, base_channels=8)
    model.eval() # Modo inferência
    
    with torch.no_grad():
        predicted_mask = model(input_tensor)
    print(f" - Máscara predita (Batch, Canal, Freq, Time): {predicted_mask.shape}")
    print(f" - Valores da máscara contidos entre [0, 1]: {torch.min(predicted_mask).item():.4f} a {torch.max(predicted_mask).item():.4f}")
    
    # 4. Filtragem Espacial: Multiplica a máscara pelo espectrograma sujo original
    print("[4] Aplicando máscara de ganho (Multiplicação elemento a elemento)...")
    # Remove dimensões de batch e canal para operação matricial no numpy
    mask_np = predicted_mask.squeeze().numpy()
    filtered_mag = noisy_mag * mask_np
    print(f" - Espectrograma filtrado gerado: {filtered_mag.shape}")
    
    # 5. Síntese: Domínio da Frequência -> Domínio do Tempo (iSTFT)
    print("[5] Aplicando iSTFT (Frequência -> Tempo)...")
    enhanced_signal = compute_istft(filtered_mag, noisy_phase, hop_length=HOP_LENGTH)
    print(f" - Sinal reconstruído (amostras): {enhanced_signal.shape}")
    
    print("--- FLUXO COMPLETO VALIDADO COM SUCESSO ---")

    # 6. Visualização e Salvamento de Gráficos
    print("[6] Gerando gráficos para análise visual...")
    
    # Calcula a magnitude filtrada para o espectrograma final
    enhanced_mag, _ = compute_stft(enhanced_signal, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    # Gera os gráficos (serão salvos na raiz do projeto)
    plot_time_signals(clean_signal, noisy_signal, enhanced_signal, sr=SR, output_path="data/imgs/time_domain.png")
    plot_spectrograms(clean_mag, noisy_mag, enhanced_mag, sr=SR, output_path="data/imgs/spectrograms.png")

if __name__ == "__main__":
    main()
