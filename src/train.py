from pathlib import Path
import csv
import random
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset

try:
    from .dataset_builder import NoiseSuppressionDataset, PrecomputedNoiseSuppressionDataset
    from .model import TinyUNet
except ImportError:
    from dataset_builder import NoiseSuppressionDataset, PrecomputedNoiseSuppressionDataset
    from model import TinyUNet


def _print_progress_bar(current: int, total: int, prefix: str, suffix: str = "") -> None:
    bar_length = 30
    filled_length = int(bar_length * current // total) if total else bar_length
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    print(f"\r{prefix} |{bar}| {current}/{total} {suffix}", end="", flush=True)


def _split_indices_by_clean_source(metadata_path: Path, validation_ratio: float, seed: int = 42) -> tuple[list[int], list[int]]:
    clean_to_indices: dict[str, list[int]] = defaultdict(list)

    with metadata_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for sample_index, row in enumerate(reader):
            clean_file = row["clean_file"]
            clean_to_indices[clean_file].append(sample_index)

    unique_clean_files = list(clean_to_indices.keys())
    if len(unique_clean_files) < 2:
        raise ValueError("É preciso ter pelo menos 2 arquivos limpos distintos para fazer split por origem.")

    random.Random(seed).shuffle(unique_clean_files)

    validation_clean_count = max(1, int(len(unique_clean_files) * validation_ratio))
    if validation_clean_count >= len(unique_clean_files):
        validation_clean_count = 1

    validation_clean_files = set(unique_clean_files[:validation_clean_count])

    train_indices: list[int] = []
    validation_indices: list[int] = []

    for clean_file, sample_indices in clean_to_indices.items():
        if clean_file in validation_clean_files:
            validation_indices.extend(sample_indices)
        else:
            train_indices.extend(sample_indices)

    if not train_indices or not validation_indices:
        raise ValueError("O split por origem ficou vazio em treino ou validação. Ajuste validation_ratio.")

    return train_indices, validation_indices


def train(epochs: int = 20) -> Path:
    sr = 16000
    n_fft = 512
    hop_length = 256
    batch_size = 4
    validation_ratio = 0.2
    learning_rate = 1e-3

    data_root = Path("data")
    clean_dir = data_root / "clean"
    noise_dir = data_root / "noise"
    checkpoint_dir = Path("checkpoints")
    clean_dir.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "tinyunet.pth"

    precomputed_dir = data_root / "precomputed"
    use_precomputed_dataset = precomputed_dir.exists() and any(precomputed_dir.glob("*.pt"))

    if use_precomputed_dataset:
        print(f"[i] Usando dataset pré-computado em: {precomputed_dir}")
        dataset = PrecomputedNoiseSuppressionDataset(dataset_dir=str(precomputed_dir))
    else:
        print("[i] Nenhum dataset pré-computado encontrado. Usando pré-processamento online.")
        dataset = NoiseSuppressionDataset(
            clean_dir=str(clean_dir),
            noise_dir=str(noise_dir),
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            snr_range=(0.0, 10.0),
        )

        if not dataset.clean_files:
            raise ValueError(
                f"Nenhum arquivo de áudio encontrado em {clean_dir}. Adicione arquivos .wav, .mp3 ou .webm antes de treinar."
            )

        if not dataset.noise_files:
            raise ValueError(
                f"Nenhum arquivo de áudio encontrado em {noise_dir}. Adicione arquivos .wav, .mp3 ou .webm antes de treinar."
            )

    if len(dataset) < 2:
        raise ValueError("É preciso ter pelo menos 2 amostras para treinar/validar.")

    if use_precomputed_dataset:
        metadata_path = precomputed_dir / "metadata.csv"
        if not metadata_path.exists():
            raise ValueError(f"{metadata_path} não encontrado. Gere o dataset pré-computado novamente.")

        train_indices, validation_indices = _split_indices_by_clean_source(
            metadata_path=metadata_path,
            validation_ratio=validation_ratio,
            seed=42,
        )
        train_dataset = Subset(dataset, train_indices)
        validation_dataset = Subset(dataset, validation_indices)
    else:
        validation_size = max(1, int(len(dataset) * validation_ratio))
        train_size = len(dataset) - validation_size
        if train_size <= 0:
            train_size = len(dataset) - 1
            validation_size = 1

        train_dataset, validation_dataset = random_split(
            dataset,
            [train_size, validation_size],
            generator=torch.Generator().manual_seed(42),
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyUNet(in_channels=1, base_channels=8).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    best_validation_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = len(train_loader)

        for batch_index, (noisy_mag, irm) in enumerate(train_loader, start=1):
            noisy_mag = noisy_mag.to(device)
            irm = irm.to(device)

            optimizer.zero_grad(set_to_none=True)
            predicted_mask = model(noisy_mag)
            loss = criterion(predicted_mask, irm)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * noisy_mag.size(0)
            _print_progress_bar(
                batch_index,
                train_batches,
                prefix=f"Epoch {epoch:02d}/{epochs} [train]",
                suffix=f"loss={loss.item():.4f}",
            )

        print()

        model.eval()
        validation_loss_sum = 0.0
        validation_batches = len(validation_loader)
        with torch.no_grad():
            for batch_index, (noisy_mag, irm) in enumerate(validation_loader, start=1):
                noisy_mag = noisy_mag.to(device)
                irm = irm.to(device)
                predicted_mask = model(noisy_mag)
                loss = criterion(predicted_mask, irm)
                validation_loss_sum += loss.item() * noisy_mag.size(0)
                _print_progress_bar(
                    batch_index,
                    validation_batches,
                    prefix=f"Epoch {epoch:02d}/{epochs} [val ]",
                    suffix=f"loss={loss.item():.4f}",
                )

        print()

        mean_train_loss = train_loss_sum / len(train_loader.dataset)
        mean_validation_loss = validation_loss_sum / len(validation_loader.dataset)
        scheduler.step(mean_validation_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{epochs} | train_loss={mean_train_loss:.4f} | "
            f"val_loss={mean_validation_loss:.4f} | lr={current_lr:.6f}"
        )

        if mean_validation_loss < best_validation_loss:
            best_validation_loss = mean_validation_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "validation_loss": mean_validation_loss,
                    "sr": sr,
                    "n_fft": n_fft,
                    "hop_length": hop_length,
                },
                checkpoint_path,
            )
            print(f"[+] Checkpoint salvo em: {checkpoint_path}")

    return checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina a TinyUNet para supressao de ruido.")
    parser.add_argument("--epochs", type=int, default=20, help="Numero de epocas de treino.")
    args = parser.parse_args()
    train(epochs=args.epochs)