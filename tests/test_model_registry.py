import pytest
import torch
import torch.nn as nn

from src.benchmark import count_macs
from src.inference import load_model
from src.model import TinyUNet, build_model, list_available_models


def test_list_available_models_contains_tinyunet():
    models = list_available_models()
    assert "tinyunet" in models
    assert models == sorted(models)


@pytest.mark.parametrize("model_name", ["tinyunet", "TinyUNet", "  tinyunet  "])
def test_build_model_instantiates_tinyunet(model_name):
    model = build_model(model_name=model_name, in_channels=1, base_channels=4)
    assert isinstance(model, TinyUNet)
    assert model.base_channels == 4


def test_build_model_invalid_name_raises_value_error():
    with pytest.raises(ValueError, match="tinyunet"):
        build_model("non_existent_architecture")


def test_load_model_self_describing_checkpoint(tmp_path):
    source_model = build_model("tinyunet", in_channels=1, base_channels=4)
    ckpt_path = tmp_path / "custom_ckpt.pth"
    torch.save(
        {
            "model_name": "tinyunet",
            "model_state_dict": source_model.state_dict(),
            "base_channels": 4,
            "sr": 16000,
            "n_fft": 512,
            "hop_length": 256,
        },
        ckpt_path,
    )

    loaded_model, dsp_config = load_model(ckpt_path, device=torch.device("cpu"))
    assert isinstance(loaded_model, TinyUNet)
    assert loaded_model.base_channels == 4
    assert dsp_config["model_name"] == "tinyunet"
    assert dsp_config["base_channels"] == 4


def test_load_model_without_checkpoint_uses_requested_model():
    model, dsp_config = load_model(
        checkpoint_path=None,
        device=torch.device("cpu"),
        model_name="tinyunet",
        base_channels=4,
    )
    assert isinstance(model, TinyUNet)
    assert model.base_channels == 4
    assert dsp_config["model_name"] == "tinyunet"
    assert dsp_config["base_channels"] == 4


def test_count_macs_supports_linear_and_gru():
    class ToySequenceNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(16, 8)
            self.gru = nn.GRU(input_size=8, hidden_size=4, num_layers=1, batch_first=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch=1, seq=5, features=16)
            out = self.linear(x)
            out, _ = self.gru(out)
            return out

    toy_net = ToySequenceNet()
    # Linear: 5 steps * 16 * 8 = 640 MACs
    # GRU: 5 steps * 1 layer * 1 dir * 3 gates * 4 * (8 + 4) = 5 * 144 = 720 MACs
    mac_stats = count_macs(toy_net, input_shape=(1, 5, 16))
    assert mac_stats["total_macs"] == 640 + 720
    assert mac_stats["macs_by_layer"] == {"linear": 640, "gru": 720}
