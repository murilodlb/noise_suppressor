from typing import Any, Dict, List, Optional, Type

import torch
import torch.nn as nn


class BCEMaskLoss(nn.Module):
    """Perda de Entropia Cruzada Binária padrão para estimação de máscara."""

    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(
        self,
        predicted_mask: torch.Tensor,
        target_irm: torch.Tensor,
        noisy_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.bce(predicted_mask, target_irm)


class MSEMaskLoss(nn.Module):
    """Perda de Erro Quadrático Médio (MSE) para regressão contínua da IRM."""

    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        predicted_mask: torch.Tensor,
        target_irm: torch.Tensor,
        noisy_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.mse(predicted_mask, target_irm)


class L1MaskLoss(nn.Module):
    """Perda de Erro Absoluto Médio (L1), promovendo esparsidade nos bins de ruído."""

    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(
        self,
        predicted_mask: torch.Tensor,
        target_irm: torch.Tensor,
        noisy_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.l1(predicted_mask, target_irm)


class BCEL1HybridLoss(nn.Module):
    """Combinação híbrida de BCE e L1 para convergência rápida e esparsidade de fundo."""

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.bce = nn.BCELoss()
        self.l1 = nn.L1Loss()

    def forward(
        self,
        predicted_mask: torch.Tensor,
        target_irm: torch.Tensor,
        noisy_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.alpha * self.bce(predicted_mask, target_irm) + (1.0 - self.alpha) * self.l1(
            predicted_mask, target_irm
        )


class MagnitudeWeightedMSELoss(nn.Module):
    """
    Erro Quadrático Médio ponderado pela energia espectral de entrada.
    Prioriza harmônicos de fala de alta energia sobre bins de silêncio/ruído de baixa energia.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        predicted_mask: torch.Tensor,
        target_irm: torch.Tensor,
        noisy_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sq_error = (predicted_mask - target_irm) ** 2
        if noisy_input is None:
            return torch.mean(sq_error)

        weights = 1.0 + torch.sigmoid(noisy_input.detach())
        weights = weights / (weights.mean(dim=(-2, -1), keepdim=True) + self.eps)
        return torch.mean(weights * sq_error)


LOSS_REGISTRY: Dict[str, Type[nn.Module]] = {
    "bce": BCEMaskLoss,
    "mse": MSEMaskLoss,
    "l1": L1MaskLoss,
    "bce_l1": BCEL1HybridLoss,
    "mag_mse": MagnitudeWeightedMSELoss,
}


def list_available_losses() -> List[str]:
    """Retorna a lista de funções de perda registradas em LOSS_REGISTRY."""
    return list(LOSS_REGISTRY.keys())


def build_loss(loss_name: str = "bce", **kwargs: Any) -> nn.Module:
    """Instancia uma função de perda pelo nome registrado em LOSS_REGISTRY."""
    normalized_name = loss_name.lower().strip()
    if normalized_name not in LOSS_REGISTRY:
        available = ", ".join(list_available_losses())
        raise ValueError(
            f"Função de perda '{loss_name}' desconhecida. Opções disponíveis: [{available}]"
        )
    loss_cls = LOSS_REGISTRY[normalized_name]
    return loss_cls(**kwargs)
