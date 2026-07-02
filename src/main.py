from pathlib import Path
import random

import numpy as np
import soundfile as sf
import torch

from .dataset_builder import NoiseSuppressionDataset
from .model import TinyUNet
from .dsp_utils import compute_irm, compute_stft, compute_istft, plot_time_signals, plot_spectrograms, preprocess_noisy_magnitude


def _pick_holdout_clean_file(clean_files: list[Path], validation_ratio: float = 0.2, seed: int = 42) -> Path:
    if len(clean_files) < 2:
        raise ValueError("É preciso ter pelo menos 2 arquivos limpos para selecionar um holdout.")

    shuffled_files = clean_files[:]
    random.Random(seed).shuffle(shuffled_files)
    holdout_count = max(1, int(len(shuffled_files) * validation_ratio))
    holdout_files = shuffled_files[:holdout_count]
    return holdout_files[0]

def main():
    print("--- INICIANDO PROVA DE CONCEITO (PoC) ---")
    
    # Hiperparâmetros de PDS
    SR = 16000
    N_FFT = 512
    HOP_LENGTH = 256
    
    data_root = Path("data")
    clean_dir = data_root / "clean"
    noise_dir = data_root / "noise"

    dataset = NoiseSuppressionDataset(
        clean_dir=str(clean_dir),
        noise_dir=str(noise_dir),
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        target_seconds=2.0,
    )

    # 1. Sinal de fala real do Common Voice, escolhendo um holdout determinístico
    if dataset.clean_files and dataset.noise_files:
        print("[1] Carregando amostra real de voz do Common Voice...")
        # clean_path = _pick_holdout_clean_file([Path(file_path) for file_path in dataset.clean_files], validation_ratio=0.2, seed=42)
        clean_path = data_root / "test" / "test.mp3"
        noise_path = Path(random.choice(dataset.noise_files))

        print(f" - Voz de teste: {clean_path.name}")
        print(f" - Ruído de teste: {noise_path.name}")

        clean_signal = dataset._fix_length(dataset._load_audio(str(clean_path)))
        noise_signal = dataset._fix_length(dataset._load_audio(str(noise_path)))
        noisy_signal = dataset._add_noise(clean_signal, noise_signal)
        print(f" - Forma do sinal (amostras): {noisy_signal.shape}")
    else:
        print("[1] Dados de voz/ruído não encontrados; usando sinal sintético como fallback...")
        duration = 2.0
        t = np.linspace(0, duration, int(SR * duration), endpoint=False)
        clean_signal = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)
        noise_signal = np.random.normal(0, 0.2, len(t))
        noisy_signal = clean_signal + noise_signal
        print(f" - Forma do sinal (amostras): {noisy_signal.shape}")
    
    # 2. Análise: Domínio do Tempo -> Domínio da Frequência (Espectrograma)
    print("[2] Aplicando STFT (Domínio do Tempo -> Domínio da Frequência)...")
    clean_mag, _ = compute_stft(clean_signal, n_fft=N_FFT, hop_length=HOP_LENGTH)
    noisy_mag, noisy_phase = compute_stft(noisy_signal, n_fft=N_FFT, hop_length=HOP_LENGTH)
    noise_mag, _ = compute_stft(noise_signal, n_fft=N_FFT, hop_length=HOP_LENGTH)
    noisy_mag_input = preprocess_noisy_magnitude(noisy_mag)
    print(f" - Matriz de Magnitude (Frequências x Frames): {noisy_mag.shape}")
    
    # Prepara tensor para a U-Net (Batch=1, Channel=1, Freq, Time)
    input_tensor = torch.from_numpy(noisy_mag_input).float().unsqueeze(0).unsqueeze(0)
    
    # 3. Inferência: Passagem pela U-Net (CNN)
    print("[3] Realizando forward pass na TinyU-Net...")
    model = TinyUNet(in_channels=1, base_channels=8)
    model.eval() # Modo inferência
    
    with torch.no_grad():
        predicted_mask = model(input_tensor)

    checkpoint_path = Path("checkpoints/tinyunet.pth")
    if checkpoint_path.exists():
        print(f"[3b] Usando checkpoint treinado: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        with torch.no_grad():
            predicted_mask = model(input_tensor)
        mask_np = predicted_mask.squeeze().cpu().numpy()
        mask_source = "modelo treinado"
    else:
        print("[3b] Nenhum checkpoint encontrado; usando máscara IRM ideal para a PoC.")
        mask_np = compute_irm(clean_mag, noise_mag)
        mask_source = "máscara ideal (IRM)"

    print(f" - Máscara predita (Batch, Canal, Freq, Time): {predicted_mask.shape}")
    print(f" - Valores da máscara contidos entre [0, 1]: {float(mask_np.min()):.4f} a {float(mask_np.max()):.4f}")
    
    # 4. Filtragem Espacial: Multiplica a máscara pelo espectrograma sujo original
    print("[4] Aplicando máscara de ganho (Multiplicação elemento a elemento)...")
    filtered_mag = noisy_mag * mask_np
    print(f" - Espectrograma filtrado gerado: {filtered_mag.shape}")
    
    # 5. Síntese: Domínio da Frequência -> Domínio do Tempo (iSTFT)
    print("[5] Aplicando iSTFT (Frequência -> Tempo)...")
    enhanced_signal = compute_istft(filtered_mag, noisy_phase, hop_length=HOP_LENGTH)
    print(f" - Sinal reconstruído (amostras): {enhanced_signal.shape}")
    print(f" - Máscara utilizada: {mask_source}")
    
    print("--- FLUXO COMPLETO VALIDADO COM SUCESSO ---")

    # 6. Visualização e Salvamento de Gráficos
    print("[6] Gerando gráficos para análise visual...")
    
    # Calcula a magnitude filtrada para o espectrograma final
    enhanced_mag, _ = compute_stft(enhanced_signal, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    # Gera os gráficos (serão salvos na raiz do projeto)
    plot_time_signals(clean_signal, noisy_signal, enhanced_signal, sr=SR, output_path="data/imgs/time_domain.png")
    plot_spectrograms(clean_mag, noisy_mag, enhanced_mag, sr=SR, output_path="data/imgs/spectrograms.png")

    audio_dir = Path("data/audio")
    audio_dir.mkdir(parents=True, exist_ok=True)
    sf.write(audio_dir / "clean.wav", clean_signal, SR)
    sf.write(audio_dir / "noisy.wav", noisy_signal, SR)
    sf.write(audio_dir / "enhanced.wav", enhanced_signal, SR)
    print(f"[+] Áudios salvos em: {audio_dir}")

    return {
        "clean_signal": clean_signal,
        "noisy_signal": noisy_signal,
        "enhanced_signal": enhanced_signal,
        "mask_source": mask_source,
    }

if __name__ == "__main__":
    main()
