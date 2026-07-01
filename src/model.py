import torch
import torch.nn as nn
import torch.nn.functional as F

def center_crop(encoder_layer: torch.Tensor, decoder_layer: torch.Tensor) -> torch.Tensor:
    """
    Corta o tensor do encoder para que suas dimensões espaciais 
    correspondam exatamente às do tensor do decoder.
    """
    _, _, h_enc, w_enc = encoder_layer.size()
    _, _, h_dec, w_dec = decoder_layer.size()
    
    diff_h = (h_enc - h_dec) // 2
    diff_w = (w_enc - w_dec) // 2
    
    return encoder_layer[:, :, diff_h:(diff_h + h_dec), diff_w:(diff_w + w_dec)]

class DoubleConv(nn.Module):
    """Bloco padrão da U-Net: Duas convoluções 3x3 seguidas de BatchNorm e ReLU."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class TinyUNet(nn.Module):
    """
    Arquitetura U-Net minimalista focada em viabilidade para TinyML.
    Possui poucos canais para reduzir a quantidade de parâmetros e MACs.
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 8):
        super().__init__()
        
        # Encoder (Downsampling)
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 2, base_channels * 4)
        
        # Decoder (Upsampling)
        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        
        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)
        
        # Camada de saída: Convolução 1x1 mapeando para 1 canal (a máscara)
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

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
        
        # Saída entre 0 e 1
        out = self.sigmoid(self.out_conv(c5))
        
        # --- CORREÇÃO AQUI ---
        # Força a saída a ter exatamente as mesmas dimensões espaciais (H, W) da entrada x.
        # Isso resolve o mismatch de pixels perdidos nas bordas durante as convoluções.
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
            
        return out
