import json
import numpy as np
import torch
import torch.nn as nn

from src.benchmark import count_macs, count_parameters
from src.cli import main as cli_main
from src.inference import enhance_signal, load_model
from src.model import DSTinyUNet, TinyUNet, build_model, list_available_models

INPUT_SHAPE = (1, 1, 257, 126)


def test_build_model_returns_ds_tinyunet():
    assert "ds_tinyunet" in list_available_models()
    model = build_model("ds_tinyunet", in_channels=1, base_channels=8)
    assert isinstance(model, DSTinyUNet)


def test_ds_tinyunet_uses_depthwise_convolutions():
    model = DSTinyUNet(in_channels=1, base_channels=8)
    spatial_convs = [m for m in model.modules() if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3)]
    stem_conv = spatial_convs[0]
    assert stem_conv.in_channels == 1 and stem_conv.out_channels == 8 and stem_conv.groups == 1
    for conv in spatial_convs[1:]:
        assert conv.groups == conv.in_channels == conv.out_channels


def test_ds_tinyunet_output_shape_and_range():
    model = DSTinyUNet(in_channels=1, base_channels=8).eval()
    with torch.no_grad():
        output = model(torch.randn(INPUT_SHAPE))
    assert output.shape == INPUT_SHAPE
    assert float(output.min()) >= 0.0 and float(output.max()) <= 1.0


def test_ds_tinyunet_enhance_signal_preserves_length():
    model = DSTinyUNet(in_channels=1, base_channels=8).eval()
    for num_samples in (32000, 59200):
        noisy_signal = np.random.randn(num_samples).astype(np.float32)
        result = enhance_signal(model, noisy_signal, n_fft=512, hop_length=256)
        assert len(result.enhanced_signal) == num_samples


def test_ds_tinyunet_efficiency_vs_tinyunet():
    baseline = TinyUNet(in_channels=1, base_channels=8)
    ds_model = DSTinyUNet(in_channels=1, base_channels=8)

    baseline_params = count_parameters(baseline)["total_parameters"]
    ds_params = count_parameters(ds_model)["total_parameters"]
    baseline_macs = count_macs(baseline, INPUT_SHAPE)["total_macs"]
    ds_macs = count_macs(ds_model, INPUT_SHAPE)["total_macs"]

    assert ds_params < 0.45 * baseline_params
    assert ds_macs < 0.45 * baseline_macs


def test_ds_tinyunet_checkpoint_roundtrip(tmp_path):
    source_model = DSTinyUNet(in_channels=1, base_channels=8)
    ckpt_path = tmp_path / "ds_tinyunet.pth"
    torch.save(
        {"model_name": "ds_tinyunet", "model_state_dict": source_model.state_dict(), "base_channels": 8},
        ckpt_path,
    )
    loaded_model, dsp_config = load_model(ckpt_path, device=torch.device("cpu"))
    assert isinstance(loaded_model, DSTinyUNet)
    assert dsp_config["model_name"] == "ds_tinyunet"


def test_cli_benchmark_model_flag_ignores_default_tinyunet_checkpoint(tmp_path, monkeypatch):
    # Simula um diretório de trabalho que possui checkpoints/tinyunet.pth (base_channels=8)
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tiny_model = TinyUNet(in_channels=1, base_channels=8)
    torch.save(
        {"model_name": "tinyunet", "base_channels": 8, "model_state_dict": tiny_model.state_dict()},
        ckpt_dir / "tinyunet.pth",
    )
    monkeypatch.chdir(tmp_path)

    out_json = tmp_path / "bench_ds.json"
    exit_code = cli_main(
        [
            "benchmark",
            "--model",
            "ds_tinyunet",
            "--base_channels",
            "16",
            "--warmup",
            "1",
            "--repeats",
            "2",
            "--output_path",
            str(out_json),
        ]
    )
    assert exit_code == 0
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["model_name"] == "ds_tinyunet"
    assert report["base_channels"] == 16

