from typing import Any

import torch


class BCEMaskLoss(torch.nn.Module):
    """Perda de Entropia Cruzada Binária padrão para estimação de máscara."""

    def __init__(self) -> None:
        super().__init__()
        self.bce = torch.nn.BCELoss()

    def forward(
        self,
        predicted_mask: torch.Tensor,
        target_irm: torch.Tensor,
        noisy_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.bce(predicted_mask, target_irm)


class MSEMaskLoss(torch.nn.Module):
    """Perda de Erro Quadrático Médio (MSE) para regressão contínua da IRM."""

    def __init__(self) -> None:
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(
        self,
        predicted_mask: torch.Tensor,
        target_irm: torch.Tensor,
        noisy_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.mse(predicted_mask, target_irm)


class L1MaskLoss(torch.nn.Module):
    """Perda de Erro Absoluto Médio (L1), promovendo esparsidade nos bins de ruído."""

    def __init__(self) -> None:
        super().__init__()
        self.l1 = torch.nn.L1Loss()

    def forward(
        self,
        predicted_mask: torch.Tensor,
        target_irm: torch.Tensor,
        noisy_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.l1(predicted_mask, target_irm)


class BCEL1HybridLoss(torch.nn.Module):
    """Combinação híbrida de BCE e L1 para convergência rápida e esparsidade de fundo."""

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.bce = torch.nn.BCELoss()
        self.l1 = torch.nn.L1Loss()

    def forward(
        self,
        predicted_mask: torch.Tensor,
        target_irm: torch.Tensor,
        noisy_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.alpha * self.bce(predicted_mask, target_irm) + (1.0 - self.alpha) * self.l1(
            predicted_mask, target_irm
        )


class MagnitudeWeightedMSELoss(torch.nn.Module):
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
        noisy_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sq_error = (predicted_mask - target_irm) ** 2
        if noisy_input is None:
            return torch.mean(sq_error)

        weights = 1.0 + torch.sigmoid(noisy_input.detach())
        weights = weights / (weights.mean(dim=(-2, -1), keepdim=True) + self.eps)
        return torch.mean(weights * sq_error)


class AsymmetricPerceptualMaskLoss(torch.nn.Module):
    """
    Perda híbrida perceptual assimétrica (Energy-Weighted MSE + Asymmetric Recall + L1).
    Combina:
      1. Ponderação por energia espectral de entrada (foco nos harmônicos de fala).
      2. Penalização assimétrica quando predicted_mask < target_irm em bins de fala
         (aumenta o Recall evitando atenuação excessiva da voz).
      3. Termo L1 ponderado para manter gradientes fortes quando |erro| < 0.15 (evita platô).
    """

    def __init__(
        self,
        recall_weight: float = 0.6,
        l1_weight: float = 0.2,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.recall_weight = float(recall_weight)
        self.l1_weight = float(l1_weight)
        self.eps = eps

    def forward(
        self,
        predicted_mask: torch.Tensor,
        target_irm: torch.Tensor,
        noisy_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        diff = predicted_mask - target_irm
        sq_error = diff**2
        abs_error = torch.abs(diff)

        under_estimation = (diff < 0.0).to(predicted_mask.dtype)
        asym_weights = 1.0 + self.recall_weight * target_irm.detach() * under_estimation

        if noisy_input is not None:
            energy_weights = 1.0 + torch.sigmoid(noisy_input.detach())
            energy_weights = energy_weights / (
                energy_weights.mean(dim=(-2, -1), keepdim=True) + self.eps
            )
        else:
            energy_weights = torch.ones_like(predicted_mask)

        combined_weights = energy_weights * asym_weights
        quad_term = torch.mean(combined_weights * sq_error)
        l1_term = torch.mean(combined_weights * abs_error)
        return (1.0 - self.l1_weight) * quad_term + self.l1_weight * l1_term


LOSS_REGISTRY: dict[str, type[torch.nn.Module]] = {
    "bce": BCEMaskLoss,
    "mse": MSEMaskLoss,
    "l1": L1MaskLoss,
    "bce_l1": BCEL1HybridLoss,
    "mag_mse": MagnitudeWeightedMSELoss,
    "asym_mag_mse": AsymmetricPerceptualMaskLoss,
}


def list_available_losses() -> list[str]:
    """Retorna a lista de funções de perda registradas em LOSS_REGISTRY."""
    return list(LOSS_REGISTRY.keys())


def build_loss(loss_name: str = "bce", **kwargs: Any) -> torch.nn.Module:
    """Instancia uma função de perda pelo nome registrado em LOSS_REGISTRY."""
    normalized_name = loss_name.lower().strip()
    if normalized_name not in LOSS_REGISTRY:
        available = ", ".join(list_available_losses())
        raise ValueError(
            f"Função de perda '{loss_name}' desconhecida. Opções disponíveis: [{available}]"
        )
    loss_cls = LOSS_REGISTRY[normalized_name]
    return loss_cls(**kwargs)
