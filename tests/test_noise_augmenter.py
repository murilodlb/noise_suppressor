import numpy as np
import soundfile as sf

from src.cli import main as cli_main
from src.noise_augmenter import _crossfade_concat, augment_noise_bank


def test_crossfade_concat_smooth_length():
    seg_a = np.ones(1000, dtype=np.float32) * 0.5
    seg_b = np.ones(1000, dtype=np.float32) * -0.5
    out = _crossfade_concat([seg_a, seg_b], crossfade_samples=200)
    assert len(out) == 1800
    assert np.all(np.isfinite(out))


def test_augment_noise_bank_generates_files_and_ignores_previous_aug(tmp_path):
    sr = 16000
    rng = np.random.default_rng(7)
    for i in range(3):
        base_noise = (0.3 * rng.standard_normal(sr * 4)).astype(np.float32)
        sf.write(str(tmp_path / f"sample-{i}.wav"), base_noise, sr)

    created = augment_noise_bank(
        noise_dir=tmp_path,
        num_files=4,
        duration_seconds=3.0,
        sr=sr,
        seed=42,
    )
    assert len(created) == 4
    for path in created:
        assert path.exists()
        audio, file_sr = sf.read(str(path), dtype="float32")
        assert file_sr == sr
        assert len(audio) == sr * 3
        assert float(np.max(np.abs(audio))) > 0.1

    # Re-run via CLI to verify idempotency (old aug_noise_* are replaced, not compounded)
    exit_code = cli_main(
        [
            "augment_noise",
            "--noise_dir",
            str(tmp_path),
            "--num_files",
            "2",
            "--duration",
            "2.0",
            "--seed",
            "99",
        ]
    )
    assert exit_code == 0
    aug_files = sorted(tmp_path.glob("aug_noise_*.wav"))
    assert len(aug_files) == 2
