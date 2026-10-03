import argparse
from pathlib import Path
import sys
from typing import List, Optional
import soundfile as sf

from .dsp_utils import compute_stft
from .inference import load_model, enhance_signal, load_audio_file
from .metrics import compute_signal_metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="noise_suppressor",
        description="Non-stationary noise suppression CLI for training, inference, evaluation, and benchmarking.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to execute")

    # 1. Preprocess
    p_prep = subparsers.add_parser("preprocess", help="Precompute STFT magnitudes and IRM masks to disk (.pt)")
    p_prep.add_argument("--clean_dir", type=str, default="data/clean", help="Directory with clean audio samples")
    p_prep.add_argument("--noise_dir", type=str, default="data/noise", help="Directory with noise audio samples")
    p_prep.add_argument("--output_dir", type=str, default="data/precomputed", help="Output directory for .pt files")
    p_prep.add_argument("--samples_per_clean", type=int, default=1, help="Number of noise mixes per clean file")
    p_prep.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # 2. Train
    p_train = subparsers.add_parser("train", help="Train the TinyUNet model")
    p_train.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    p_train.add_argument("--batch_size", type=int, default=4, help="Batch size")
    p_train.add_argument("--learning_rate", type=float, default=1e-3, help="Initial Adam learning rate")
    p_train.add_argument("--validation_ratio", type=float, default=0.2, help="Ratio of clean sources for validation")
    p_train.add_argument("--seed", type=int, default=42, help="Random seed")
    p_train.add_argument("--data_root", type=str, default="data", help="Root directory for dataset")
    p_train.add_argument("--checkpoint_path", type=str, default="checkpoints/tinyunet.pth", help="Checkpoint save path")
    p_train.add_argument("--base_channels", type=int, default=8, help="Base channel count for TinyUNet")
    p_train.add_argument("--num_workers", type=int, default=0, help="PyTorch DataLoader worker count")

    # 3. Infer
    p_infer = subparsers.add_parser("infer", help="Denoise one or more audio files")
    p_infer.add_argument("--input", nargs="+", required=True, help="Path(s) to noisy audio files")
    p_infer.add_argument("--checkpoint_path", type=str, default="checkpoints/tinyunet.pth", help="Model checkpoint path")
    p_infer.add_argument("--output_dir", type=str, default="reports/inference", help="Directory to save enhanced files")
    p_infer.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")

    # 4. Demo (PoC)
    p_demo = subparsers.add_parser("demo", help="Run end-to-end PoC demonstration with plots and audio exports")
    p_demo.add_argument("--clean_path", type=str, default=None, help="Path to clean test audio")
    p_demo.add_argument("--noise_path", type=str, default=None, help="Path to noise audio")
    p_demo.add_argument("--checkpoint_path", type=str, default="checkpoints/tinyunet.pth", help="Model checkpoint path")
    p_demo.add_argument("--output_dir", type=str, default="data", help="Output directory for demo audio and plots")
    p_demo.add_argument("--seed", type=int, default=42, help="Random seed")

    # 5. Evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate model on validation split against Noisy and Oracle IRM")
    p_eval.add_argument("--checkpoint_path", type=str, default="checkpoints/tinyunet.pth", help="Model checkpoint path")
    p_eval.add_argument("--data_root", type=str, default="data", help="Data directory root")
    p_eval.add_argument("--output_dir", type=str, default="reports/evaluation", help="Directory for evaluation reports")
    p_eval.add_argument("--validation_ratio", type=float, default=0.2, help="Ratio used to reconstruct validation split")
    p_eval.add_argument("--seed", type=int, default=42, help="Random seed for validation split")
    p_eval.add_argument("--max_samples", type=int, default=None, help="Maximum number of validation samples to evaluate")
    p_eval.add_argument("--lc_db", type=float, default=0.0, help="Local Criterion in dB for mask binarization threshold")
    p_eval.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")

    # 6. Benchmark
    p_bench = subparsers.add_parser("benchmark", help="Benchmark MACs, parameters, memory, latency, and RTF")
    p_bench.add_argument("--checkpoint_path", type=str, default="checkpoints/tinyunet.pth", help="Model checkpoint path")
    p_bench.add_argument("--seconds", type=float, default=2.0, help="Duration of test audio in seconds")
    p_bench.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    p_bench.add_argument("--repeats", type=int, default=50, help="Number of timed benchmark repetitions")
    p_bench.add_argument("--num_threads", type=int, default=None, help="Torch CPU intra-op thread count")
    p_bench.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    p_bench.add_argument("--output_path", type=str, default="reports/benchmark/benchmark.json", help="Output JSON path")

    return parser


def run_infer(input_files: List[str], checkpoint_path: str, output_dir: str, device: str = "cpu") -> None:
    import torch
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch_device = torch.device(device)
    model, dsp_config = load_model(checkpoint_path, device=torch_device)
    sr = dsp_config.get("sr", 16000)
    n_fft = dsp_config.get("n_fft", 512)
    hop_length = dsp_config.get("hop_length", 256)

    print(f"[i] Executando inferência em {len(input_files)} arquivo(s)...")
    for file_str in input_files:
        p = Path(file_str)
        if not p.exists():
            print(f"[!] Arquivo não encontrado: {p}")
            continue

        noisy_sig = load_audio_file(p, sr=sr)
        result = enhance_signal(
            model=model,
            noisy_signal=noisy_sig,
            n_fft=n_fft,
            hop_length=hop_length,
            device=torch_device,
        )

        out_name = f"{p.stem}_enhanced.wav"
        out_file = out_dir / out_name
        sf.write(str(out_file), result.enhanced_signal, sr)
        print(f"[+] Áudio aprimorado salvo em: {out_file}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "preprocess":
        from .preprocess_dataset import run_preprocess
        run_preprocess(
            clean_dir=args.clean_dir,
            noise_dir=args.noise_dir,
            output_dir=args.output_dir,
            samples_per_clean=args.samples_per_clean,
            seed=args.seed,
        )
    elif args.command == "train":
        from .train import train
        train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
            data_root=Path(args.data_root),
            checkpoint_path=Path(args.checkpoint_path),
            base_channels=args.base_channels,
            num_workers=args.num_workers,
        )
    elif args.command == "infer":
        run_infer(
            input_files=args.input,
            checkpoint_path=args.checkpoint_path,
            output_dir=args.output_dir,
            device=args.device,
        )
    elif args.command == "demo":
        from .main import run_demo
        run_demo(
            clean_path=args.clean_path,
            noise_path=args.noise_path,
            checkpoint_path=args.checkpoint_path,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    elif args.command == "evaluate":
        from .evaluate import evaluate
        evaluate(
            checkpoint_path=args.checkpoint_path,
            data_root=args.data_root,
            output_dir=args.output_dir,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
            max_samples=args.max_samples,
            lc_db=args.lc_db,
            device=args.device,
        )
    elif args.command == "benchmark":
        from .benchmark import run_benchmark
        run_benchmark(
            checkpoint_path=args.checkpoint_path,
            seconds=args.seconds,
            warmup=args.warmup,
            repeats=args.repeats,
            num_threads=args.num_threads,
            device=args.device,
            output_path=args.output_path,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
