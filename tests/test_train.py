import csv
import json
import math
from pathlib import Path

import torch

from src.train import train


def _create_synthetic_precomputed_dataset(data_root: Path, num_clean_sources: int = 3) -> None:
    precomputed_dir = data_root / "precomputed"
    precomputed_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = precomputed_dir / "metadata.csv"

    rng = torch.Generator().manual_seed(0)
    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["sample_file", "clean_file", "noise_file", "sample_index", "snr_db"],
        )
        writer.writeheader()
        for src_idx in range(num_clean_sources):
            for sample_idx in range(2):
                sample_path = precomputed_dir / f"clean_{src_idx}_sample_{sample_idx}.pt"
                noisy_mag = torch.randn((1, 65, 32), generator=rng)
                irm = torch.rand((1, 65, 32), generator=rng)
                torch.save({"noisy_mag": noisy_mag, "irm": irm}, sample_path)
                writer.writerow(
                    {
                        "sample_file": str(sample_path),
                        "clean_file": f"clean_{src_idx}.wav",
                        "noise_file": "noise_0.wav",
                        "sample_index": sample_idx,
                        "snr_db": 5.0,
                    }
                )


def test_train_smoke_exports_history_and_curves(tmp_path):
    data_root = tmp_path / "data"
    _create_synthetic_precomputed_dataset(data_root, num_clean_sources=3)

    ckpt_path = tmp_path / "checkpoints" / "smoke_res_tinyunet.pth"
    report_dir = tmp_path / "reports" / "training" / "smoke_res_tinyunet"

    saved_ckpt = train(
        epochs=2,
        batch_size=2,
        learning_rate=1e-3,
        validation_ratio=0.34,
        seed=42,
        data_root=data_root,
        checkpoint_path=ckpt_path,
        base_channels=4,
        device="cpu",
        model_name="res_tinyunet",
        loss_name="asym_mag_mse",
        scheduler_type="cosine",
        report_dir=report_dir,
    )

    assert saved_ckpt == ckpt_path
    assert ckpt_path.exists()

    history_path = report_dir / "training_history.json"
    curves_path = report_dir / "training_curves.png"
    assert history_path.exists()
    assert curves_path.exists()
    assert curves_path.stat().st_size > 1024

    history = json.loads(history_path.read_text(encoding="utf-8"))
    assert history["epochs"] == [1, 2]
    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert len(history["val_mae"]) == 2
    assert len(history["learning_rate"]) == 2
    assert all(math.isfinite(x) for x in history["train_loss"])
    assert all(math.isfinite(x) for x in history["val_loss"])
    assert all(math.isfinite(x) for x in history["val_mae"])
    assert all(math.isfinite(x) and x > 0.0 for x in history["learning_rate"])
    assert history["metadata"]["model_name"] == "res_tinyunet"
    assert history["metadata"]["loss_name"] == "asym_mag_mse"
    assert history["best_epoch"] in (1, 2)
