import random
from pathlib import Path
from typing import Tuple

try:
    from .dataset_builder import PrecomputedNoiseSuppressionDataset
except ImportError:
    from dataset_builder import PrecomputedNoiseSuppressionDataset


def run_preprocess(
    clean_dir: str | Path = "data/clean",
    noise_dir: str | Path = "data/noise",
    output_dir: str | Path = "data/precomputed",
    samples_per_clean: int = 1,
    seed: int = 42,
    snr_range: Tuple[float, float] = (-5.0, 15.0),
) -> Path:
    random.seed(seed)
    metadata_path = PrecomputedNoiseSuppressionDataset.export_precomputed_dataset(
        clean_dir=str(clean_dir),
        noise_dir=str(noise_dir),
        output_dir=str(output_dir),
        samples_per_clean=samples_per_clean,
        snr_range=snr_range,
    )
    print(f"[+] Dataset pré-computado salvo em: {metadata_path.parent}")
    return metadata_path


def main() -> Path:
    data_root = Path("data")
    return run_preprocess(
        clean_dir=data_root / "clean",
        noise_dir=data_root / "noise",
        output_dir=data_root / "precomputed",
        samples_per_clean=1,
    )


if __name__ == "__main__":
    metadata_path = main()
    print(f"[+] Dataset pré-computado salvo em: {metadata_path.parent}")
