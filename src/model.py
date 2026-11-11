from typing import Any

import torch


def center_crop(encoder_layer: torch.Tensor, decoder_layer: torch.Tensor) -> torch.Tensor:
    """
    Corta o tensor do encoder para que suas dimensões espaciais
    correspondam exatamente às do tensor do decoder.
    """
    _, _, h_enc, w_enc = encoder_layer.size()
    _, _, h_dec, w_dec = decoder_layer.size()

    diff_h = (h_enc - h_dec) // 2
    diff_w = (w_enc - w_dec) // 2

    return encoder_layer[:, :, diff_h : (diff_h + h_dec), diff_w : (diff_w + w_dec)]


class DoubleConv(torch.nn.Module):
    """Bloco padrão da U-Net: Duas convoluções 3x3 seguidas de BatchNorm e ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class TinyUNet(torch.nn.Module):
    """
    Arquitetura U-Net minimalista focada em viabilidade para TinyML.
    Possui poucos canais para reduzir a quantidade de parâmetros e MACs.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels

        # Encoder (Downsampling)
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.pool = torch.nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 2, base_channels * 4)

        # Decoder (Upsampling)
        self.upconv2 = torch.nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.upconv1 = torch.nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        # Camada de saída: Convolução 1x1 mapeando para 1 canal (a máscara)
        self.out_conv = torch.nn.Conv2d(base_channels, in_channels, kernel_size=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        c1 = self.enc1(x)
        p1 = self.pool(c1)

        c2 = self.enc2(p1)
        p2 = self.pool(c2)

        # Bottleneck
        c3 = self.bottleneck(p2)

        # Decoder path com skip connections
        u2 = self.upconv2(c3)
        c2_cropped = center_crop(c2, u2)
        cat2 = torch.cat((u2, c2_cropped), dim=1)
        c4 = self.dec2(cat2)

        u1 = self.upconv1(c4)
        c1_cropped = center_crop(c1, u1)
        cat1 = torch.cat((u1, c1_cropped), dim=1)
        c5 = self.dec1(cat1)

        # Alinha o mapa de atributos antes da projeção final 1x1 e Sigmoid
        # para evitar suavização bilinear (blur) das bordas harmônicas na máscara.
        if c5.shape[2:] != x.shape[2:]:
            c5 = torch.nn.functional.interpolate(
                c5, size=x.shape[2:], mode="bilinear", align_corners=False
            )

        out = self.sigmoid(self.out_conv(c5))
        return out


class DepthwiseSeparableConv(torch.nn.Module):
    """Depthwise 3x3 (groups=in_channels) + pointwise 1x1, cada um seguido de BatchNorm e ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.depthwise = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=in_channels,
                bias=bias,
            ),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(inplace=True),
        )
        self.pointwise = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=bias),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class DSDoubleConv(torch.nn.Module):
    """Duas convoluções depthwise separable. Usa stem 3x3 padrão quando in_channels == 1."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        if in_channels == 1:
            first_layer: torch.nn.Module = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True),
            )
        else:
            first_layer = DepthwiseSeparableConv(in_channels, out_channels)
        self.double_conv = torch.nn.Sequential(
            first_layer, DepthwiseSeparableConv(out_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DSTinyUNet(torch.nn.Module):
    """Variante da TinyUNet com convoluções depthwise separable para TinyML."""

    def __init__(self, in_channels: int = 1, base_channels: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels

        self.enc1 = DSDoubleConv(in_channels, base_channels)
        self.enc2 = DSDoubleConv(base_channels, base_channels * 2)
        self.pool = torch.nn.MaxPool2d(2)
        self.bottleneck = DSDoubleConv(base_channels * 2, base_channels * 4)
        self.upconv2 = torch.nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.dec2 = DSDoubleConv(base_channels * 4, base_channels * 2)
        self.upconv1 = torch.nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1 = DSDoubleConv(base_channels * 2, base_channels)
        self.out_conv = torch.nn.Conv2d(base_channels, in_channels, kernel_size=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.enc1(x)
        c2 = self.enc2(self.pool(c1))
        c3 = self.bottleneck(self.pool(c2))

        u2 = self.upconv2(c3)
        c4 = self.dec2(torch.cat((u2, center_crop(c2, u2)), dim=1))

        u1 = self.upconv1(c4)
        c5 = self.dec1(torch.cat((u1, center_crop(c1, u1)), dim=1))

        if c5.shape[2:] != x.shape[2:]:
            c5 = torch.nn.functional.interpolate(
                c5, size=x.shape[2:], mode="bilinear", align_corners=False
            )
        out = self.sigmoid(self.out_conv(c5))
        return out


class SEBlock(torch.nn.Module):
    """Bloco Squeeze-and-Excitation leve para atenção por canal."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid_channels = max(2, channels // reduction)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(channels, mid_channels, kernel_size=1, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_channels, channels, kernel_size=1, bias=True),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.pool(x))
        return x * scale


class ResDoubleConv(torch.nn.Module):
    """Bloco convolucional duplo com conexão residual, atenção SE e suporte a dilatação."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            torch.nn.BatchNorm2d(out_channels),
        )
        self.se = SEBlock(out_channels)
        if in_channels != out_channels:
            self.shortcut: torch.nn.Module = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                torch.nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = torch.nn.Identity()
        self.act = torch.nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        return self.act(out + residual)


class ResTinyUNet(torch.nn.Module):
    """
    U-Net residual com codificação posicional de frequência (CoordConv),
    atenção por canal (SE) e bottleneck dilatado para maior campo receptivo T-F.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels

        # Entrada recebe canal extra de coordenada de frequência normalizada [0, 1]
        self.enc1 = ResDoubleConv(in_channels + 1, base_channels)
        self.enc2 = ResDoubleConv(base_channels, base_channels * 2)
        self.pool = torch.nn.MaxPool2d(2)

        # Bottleneck com convolução dilatada (dilation=2) para dobrar o campo receptivo
        self.bottleneck = ResDoubleConv(base_channels * 2, base_channels * 4, dilation=2)

        self.upconv2 = torch.nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.dec2 = ResDoubleConv(base_channels * 4, base_channels * 2)

        self.upconv1 = torch.nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1 = ResDoubleConv(base_channels * 2, base_channels)

        self.out_conv = torch.nn.Conv2d(base_channels, in_channels, kernel_size=1)
        self.sigmoid = torch.nn.Sigmoid()

    @staticmethod
    def _append_freq_coord(x: torch.Tensor) -> torch.Tensor:
        batch_size, _, freq_bins, time_frames = x.shape
        freq_coord = torch.linspace(0.0, 1.0, steps=freq_bins, device=x.device, dtype=x.dtype)
        freq_map = freq_coord.view(1, 1, freq_bins, 1).expand(batch_size, 1, freq_bins, time_frames)
        return torch.cat((x, freq_map), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_coord = self._append_freq_coord(x)
        c1 = self.enc1(x_coord)
        c2 = self.enc2(self.pool(c1))
        c3 = self.bottleneck(self.pool(c2))

        u2 = self.upconv2(c3)
        c4 = self.dec2(torch.cat((u2, center_crop(c2, u2)), dim=1))

        u1 = self.upconv1(c4)
        c5 = self.dec1(torch.cat((u1, center_crop(c1, u1)), dim=1))

        if c5.shape[2:] != x.shape[2:]:
            c5 = torch.nn.functional.interpolate(
                c5, size=x.shape[2:], mode="bilinear", align_corners=False
            )
        out = self.sigmoid(self.out_conv(c5))
        return out


MODEL_REGISTRY: dict[str, type[torch.nn.Module]] = {
    "tinyunet": TinyUNet,
    "ds_tinyunet": DSTinyUNet,
    "res_tinyunet": ResTinyUNet,
}


def list_available_models() -> list[str]:
    """Retorna a lista ordenada de arquiteturas registradas no MODEL_REGISTRY."""
    return sorted(MODEL_REGISTRY.keys())


def build_model(
    model_name: str = "tinyunet",
    in_channels: int = 1,
    base_channels: int = 8,
    **kwargs: Any,
) -> torch.nn.Module:
    """Instancia uma arquitetura pelo nome registrado em MODEL_REGISTRY."""
    normalized_name = model_name.lower().strip()
    if normalized_name not in MODEL_REGISTRY:
        available = ", ".join(list_available_models())
        raise ValueError(
            f"Arquitetura '{model_name}' desconhecida. Modelos disponíveis: [{available}]"
        )
    model_cls = MODEL_REGISTRY[normalized_name]
    return model_cls(in_channels=in_channels, base_channels=base_channels, **kwargs)
