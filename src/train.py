import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset, random_split

try:
    from .dataset_builder import (
        NoiseSuppressionDataset,
        PrecomputedNoiseSuppressionDataset,
    )
    from .losses import build_loss, list_available_losses
    from .model import build_model, list_available_models
except ImportError:
    from dataset_builder import (
        NoiseSuppressionDataset,
        PrecomputedNoiseSuppressionDataset,
    )
    from losses import build_loss, list_available_losses
    from model import build_model, list_available_models


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


def _resolve_device(device: str) -> torch.device:
    requested = device.strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Dispositivo '{device}' solicitado, mas torch.cuda.is_available() é False "
            f"(torch={torch.__version__}). No cluster/GPU, instale a wheel CUDA do PyTorch "
            f"(ex.: uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu124)."
        )
    return torch.device(device)


def _apply_spec_augmentation(noisy_mag: torch.Tensor, irm: torch.Tensor) -> torch.Tensor:
    """
    Aplica perturbações leves on-the-fly no espectrograma de entrada durante o treino:
      1. Variação aleatória de contraste/ganho e offset Z-score.
      2. Inclinação espectral suave (spectral tilt) ao longo do eixo de frequência.
      3. Micro-mascaramento T-F (SpecAugment) restrito às regiões dominadas por ruído (irm < 0.2),
         preservando a consistência exata com o gabarito IRM.
    """
    batch_size, _, freq_bins, time_frames = noisy_mag.shape
    device = noisy_mag.device
    dtype = noisy_mag.dtype

    scale = torch.empty((batch_size, 1, 1, 1), device=device, dtype=dtype).uniform_(0.90, 1.10)
    shift = torch.empty((batch_size, 1, 1, 1), device=device, dtype=dtype).uniform_(-0.15, 0.15)
    slope = torch.empty((batch_size, 1, 1, 1), device=device, dtype=dtype).uniform_(-0.15, 0.15)
    freq_axis = torch.linspace(-1.0, 1.0, steps=freq_bins, device=device, dtype=dtype).view(
        1, 1, freq_bins, 1
    )

    augmented = noisy_mag * scale + shift + slope * freq_axis

    if freq_bins > 16 and time_frames > 12:
        f_width = int(torch.randint(4, 13, (1,)).item())
        f_start = int(torch.randint(0, freq_bins - f_width, (1,)).item())
        t_width = int(torch.randint(3, 9, (1,)).item())
        t_start = int(torch.randint(0, time_frames - t_width, (1,)).item())

        patch_mask = torch.zeros_like(augmented)
        patch_mask[:, :, f_start : f_start + f_width, :] = 1.0
        patch_mask[:, :, :, t_start : t_start + t_width] = 1.0

        noise_only_mask = (irm < 0.2).to(dtype) * patch_mask
        floor_val = augmented.amin(dim=(-2, -1), keepdim=True)
        augmented = augmented * (1.0 - 0.5 * noise_only_mask) + (0.5 * floor_val) * noise_only_mask

    return augmented


def train(
    epochs: int = 20,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    validation_ratio: float = 0.2,
    seed: int = 42,
    data_root: Path = Path("data"),
    checkpoint_path: Optional[Path] | None = None,
    base_channels: int = 8,
    num_workers: int = 0,
    snr_range: tuple[float, float] = (-5.0, 15.0),
    device: str = "auto",
    model_name: str = "tinyunet",
    loss_name: str = "bce",
    scheduler_patience: int = 5,
    scheduler_type: str = "plateau",
    augment_spec: bool = False,
    weight_decay: float = 1e-4,
) -> Path:
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    normalized_model_name = model_name.lower().strip()
    normalized_loss_name = loss_name.lower().strip()
    normalized_scheduler = scheduler_type.lower().strip()
    if normalized_scheduler not in ("plateau", "cosine"):
        raise ValueError(f"Scheduler '{scheduler_type}' inválido. Use 'plateau' ou 'cosine'.")
    if checkpoint_path is None:
        checkpoint_path = Path(
            f"checkpoints/{normalized_model_name}_ch{base_channels}_{normalized_loss_name}.pth"
        )

    sr = 16000
    n_fft = 512
    hop_length = 256

    clean_dir = data_root / "clean"
    noise_dir = data_root / "noise"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)

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
            snr_range=snr_range,
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
            seed=seed,
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
            generator=torch.Generator().manual_seed(seed),
        )

    resolved_device = _resolve_device(device)
    use_cuda = resolved_device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        gpu_index = resolved_device.index if resolved_device.index is not None else torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_index)
        vram_gb = torch.cuda.get_device_properties(gpu_index).total_memory / (1024 ** 3)
        print(f"[i] Treinando em GPU CUDA: {gpu_name} ({vram_gb:.1f} GB VRAM) | device={resolved_device}")
    else:
        print(f"[i] Treinando em CPU | device={resolved_device}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
    )

    model = build_model(
        model_name=normalized_model_name,
        in_channels=1,
        base_channels=base_channels,
    ).to(resolved_device)
    criterion = build_loss(normalized_loss_name).to(resolved_device)
    print(
        f"[i] Modelo: {normalized_model_name} (ch={base_channels}) | "
        f"Função de perda: {normalized_loss_name} | "
        f"scheduler={normalized_scheduler} (patience={scheduler_patience}) | "
        f"augment_spec={augment_spec} | weight_decay={weight_decay}"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    if normalized_scheduler == "cosine":
        warmup_epochs = min(5, max(1, epochs // 5)) if epochs > 5 else 0
        if warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.2,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, epochs - warmup_epochs),
                eta_min=1e-6,
            )
            scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, epochs),
                eta_min=1e-6,
            )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=scheduler_patience,
            min_lr=1e-6,
        )

    best_validation_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = len(train_loader)

        for batch_index, (noisy_mag, irm) in enumerate(train_loader, start=1):
            noisy_mag = noisy_mag.to(resolved_device, non_blocking=use_cuda)
            irm = irm.to(resolved_device, non_blocking=use_cuda)
            model_input = _apply_spec_augmentation(noisy_mag, irm) if augment_spec else noisy_mag

            optimizer.zero_grad(set_to_none=True)
            predicted_mask = model(model_input)
            loss = criterion(predicted_mask, irm, noisy_mag)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
        validation_mae_sum = 0.0
        validation_batches = len(validation_loader)
        with torch.no_grad():
            for batch_index, (noisy_mag, irm) in enumerate(validation_loader, start=1):
                noisy_mag = noisy_mag.to(resolved_device, non_blocking=use_cuda)
                irm = irm.to(resolved_device, non_blocking=use_cuda)
                predicted_mask = model(noisy_mag)
                loss = criterion(predicted_mask, irm, noisy_mag)
                batch_mae = torch.mean(torch.abs(predicted_mask - irm)).item()
                validation_loss_sum += loss.item() * noisy_mag.size(0)
                validation_mae_sum += batch_mae * noisy_mag.size(0)
                _print_progress_bar(
                    batch_index,
                    validation_batches,
                    prefix=f"Epoch {epoch:02d}/{epochs} [val  ]",
                    suffix=f"loss={loss.item():.4f} mae={batch_mae:.4f}",
                )

        print()

        mean_train_loss = train_loss_sum / len(train_loader.dataset)
        mean_validation_loss = validation_loss_sum / len(validation_loader.dataset)
        mean_validation_mae = validation_mae_sum / len(validation_loader.dataset)
        if normalized_scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(mean_validation_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{epochs} | train_loss={mean_train_loss:.4f} | "
            f"val_loss={mean_validation_loss:.4f} | val_mae={mean_validation_mae:.4f} | "
            f"lr={current_lr:.6f}"
        )

        if mean_validation_loss < best_validation_loss:
            best_validation_loss = mean_validation_loss
            torch.save(
                {
                    "model_name": normalized_model_name,
                    "loss_name": normalized_loss_name,
                    "scheduler_type": normalized_scheduler,
                    "scheduler_patience": scheduler_patience,
                    "augment_spec": augment_spec,
                    "weight_decay": weight_decay,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "validation_loss": mean_validation_loss,
                    "validation_mae": mean_validation_mae,
                    "sr": sr,
                    "n_fft": n_fft,
                    "hop_length": hop_length,
                    "base_channels": base_channels,
                    "validation_ratio": validation_ratio,
                    "seed": seed,
                },
                checkpoint_path,
            )
            print(f"[+] Checkpoint salvo em: {checkpoint_path}")

    return checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina um modelo registrado para supressão de ruído.")
    parser.add_argument("--model", type=str, default="tinyunet", choices=list_available_models(), help="Arquitetura a treinar.")
    parser.add_argument("--loss", type=str, default="bce", choices=list_available_losses(), help="Função de perda.")
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["plateau", "cosine"], help="Tipo de agendador de LR.")
    parser.add_argument("--scheduler_patience", type=int, default=5, help="Paciência do ReduceLROnPlateau.")
    parser.add_argument("--augment_spec", action="store_true", help="Ativa perturbação on-the-fly do espectrograma no treino.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay do AdamW.")
    parser.add_argument("--epochs", type=int, default=20, help="Numero de epocas de treino.")
    parser.add_argument("--device", type=str, default="auto", help="Dispositivo (auto, cpu, cuda, cuda:0).")
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        device=args.device,
        model_name=args.model,
        loss_name=args.loss,
        scheduler_patience=args.scheduler_patience,
        scheduler_type=args.scheduler,
        augment_spec=args.augment_spec,
        weight_decay=args.weight_decay,
    )
