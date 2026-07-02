from pathlib import Path

try:
    from .dataset_builder import PrecomputedNoiseSuppressionDataset
except ImportError:
    from dataset_builder import PrecomputedNoiseSuppressionDataset


def main() -> Path:
    data_root = Path("data")
    clean_dir = data_root / "clean"
    noise_dir = data_root / "noise"
    output_dir = data_root / "precomputed"

    return PrecomputedNoiseSuppressionDataset.export_precomputed_dataset(
        clean_dir=str(clean_dir),
        noise_dir=str(noise_dir),
        output_dir=str(output_dir),
        samples_per_clean=1,
    )


if __name__ == "__main__":
    metadata_path = main()
    print(f"[+] Dataset pré-computado salvo em: {metadata_path.parent}")