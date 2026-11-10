import pytest
import torch

from src.inference import load_model
from src.losses import (
    AsymmetricPerceptualMaskLoss,
    BCEL1HybridLoss,
    BCEMaskLoss,
    L1MaskLoss,
    MagnitudeWeightedMSELoss,
    MSEMaskLoss,
    build_loss,
    list_available_losses,
)
from src.model import build_model


def test_list_available_losses():
    assert list_available_losses() == ["bce", "mse", "l1", "bce_l1", "mag_mse", "asym_mag_mse"]


@pytest.mark.parametrize("loss_name", list_available_losses())
def test_all_losses_return_finite_differentiable_scalar(loss_name):
    torch.manual_seed(0)
    criterion = build_loss(loss_name)
    logits = torch.randn(2, 1, 257, 126, requires_grad=True)
    predicted_mask = torch.sigmoid(logits)
    target_irm = torch.rand(2, 1, 257, 126)
    noisy_mag = torch.randn(2, 1, 257, 126)

    loss = criterion(predicted_mask, target_irm, noisy_mag)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.requires_grad

    loss.backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


def test_build_loss_invalid_raises():
    with pytest.raises(ValueError, match="mag_mse"):
        build_loss("invalid_loss")


def test_build_loss_is_case_insensitive():
    loss_fn = build_loss("  MSE  ")
    assert isinstance(loss_fn, MSEMaskLoss)


def test_mag_mse_prioritizes_high_energy_bins():
    criterion = MagnitudeWeightedMSELoss()
    # Metade superior com alta energia (+3.0) e metade inferior com baixa energia (-3.0)
    noisy_mag = torch.full((1, 1, 8, 8), -3.0)
    noisy_mag[:, :, :4, :] = 3.0

    target_irm = torch.zeros(1, 1, 8, 8)

    # Erro de 0.5 apenas nos bins de alta energia
    pred_high_err = torch.zeros(1, 1, 8, 8)
    pred_high_err[:, :, :4, :] = 0.5

    # Erro idêntico de 0.5 apenas nos bins de baixa energia
    pred_low_err = torch.zeros(1, 1, 8, 8)
    pred_low_err[:, :, 4:, :] = 0.5

    loss_high = criterion(pred_high_err, target_irm, noisy_mag)
    loss_low = criterion(pred_low_err, target_irm, noisy_mag)
    assert float(loss_high) > float(loss_low)


def test_mag_mse_falls_back_to_mse_without_input():
    mag_mse = MagnitudeWeightedMSELoss()
    plain_mse = MSEMaskLoss()
    pred = torch.rand(2, 1, 16, 16)
    target = torch.rand(2, 1, 16, 16)
    assert torch.allclose(mag_mse(pred, target, None), plain_mse(pred, target))


def test_bce_l1_is_weighted_sum():
    hybrid = BCEL1HybridLoss(alpha=0.5)
    bce = BCEMaskLoss()
    l1 = L1MaskLoss()
    pred = torch.clamp(torch.rand(2, 1, 16, 16), 1e-4, 1.0 - 1e-4)
    target = torch.rand(2, 1, 16, 16)
    expected = 0.5 * bce(pred, target) + 0.5 * l1(pred, target)
    assert torch.allclose(hybrid(pred, target), expected, atol=1e-6)


def test_load_model_extracts_loss_name(tmp_path):
    model = build_model("tinyunet", in_channels=1, base_channels=4)
    ckpt_path = tmp_path / "model_mag_mse.pth"
    torch.save(
        {
            "model_name": "tinyunet",
            "loss_name": "mag_mse",
            "base_channels": 4,
            "model_state_dict": model.state_dict(),
        },
        ckpt_path,
    )
    _, dsp_config = load_model(ckpt_path, device=torch.device("cpu"))
    assert dsp_config["loss_name"] == "mag_mse"


def test_asym_mag_mse_penalizes_speech_underestimation_more_than_overestimation():
    criterion = AsymmetricPerceptualMaskLoss(recall_weight=0.6, l1_weight=0.2)
    target_irm = torch.full((1, 1, 8, 8), 0.8)
    noisy_mag = torch.ones((1, 1, 8, 8))

    # Sub-estimação da máscara em 0.2 (pred=0.6 < target=0.8)
    pred_under = torch.full((1, 1, 8, 8), 0.6)
    # Sobre-estimação simétrica em 0.2 (pred=1.0 > target=0.8)
    pred_over = torch.full((1, 1, 8, 8), 1.0)

    loss_under = criterion(pred_under, target_irm, noisy_mag)
    loss_over = criterion(pred_over, target_irm, noisy_mag)
    assert float(loss_under) > float(loss_over)

