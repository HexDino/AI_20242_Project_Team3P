import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    """
    Attention Gate module to focus on relevant regions in the feature maps.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x, mask=None):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Incorporate mask information if provided
        if mask is not None:
            # Resize mask to match psi dimensions
            mask_resized = F.interpolate(mask, size=psi.size()[2:], mode='bilinear', align_corners=False)
            # Use mask to enhance attention in the selected region
            psi = psi * mask_resized + 0.5 * psi * (1 - mask_resized)
            
        return x * psi


class DoubleConv(nn.Module):
    """
    Double Convolution block with BatchNorm and ReLU.
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, use_residual=False):
        super().__init__()
        self.use_residual = use_residual
        
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Add residual connection if input and output channels match
        if use_residual and in_channels == out_channels:
            self.residual = nn.Identity()
        elif use_residual:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        
    def forward(self, x):
        if self.use_residual:
            residual = self.residual(x)
            return self.double_conv(x) + residual
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv.
    """
    def __init__(self, in_channels, out_channels, use_residual=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_residual=use_residual)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling then double conv with attention gate.
    """
    def __init__(self, in_channels, out_channels, bilinear=True, use_residual=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, use_residual=use_residual)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, use_residual=use_residual)
            
        self.attn = AttentionGate(F_g=in_channels//2, F_l=in_channels//2, F_int=in_channels//4)

    def forward(self, x1, x2, mask=None):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Apply attention mechanism
        x2_attn = self.attn(g=x1, x=x2, mask=mask)
        
        # Concatenate
        x = torch.cat([x2_attn, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Output convolution layer.
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetAttention(nn.Module):
    """
    U-Net with Attention Gate for image sharpening.
    """
    def __init__(self, n_channels=3, n_classes=3, bilinear=True, with_mask_channel=True, use_residual=True):
        super(UNetAttention, self).__init__()
        self.n_channels = n_channels + (1 if with_mask_channel else 0)  # Add mask as channel if needed
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.with_mask_channel = with_mask_channel
        self.use_residual = use_residual

        self.inc = DoubleConv(self.n_channels, 64, use_residual=use_residual)
        self.down1 = Down(64, 128, use_residual=use_residual)
        self.down2 = Down(128, 256, use_residual=use_residual)
        self.down3 = Down(256, 512, use_residual=use_residual)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, use_residual=use_residual)
        
        self.up1 = Up(1024, 512 // factor, bilinear, use_residual=use_residual)
        self.up2 = Up(512, 256 // factor, bilinear, use_residual=use_residual)
        self.up3 = Up(256, 128 // factor, bilinear, use_residual=use_residual)
        self.up4 = Up(128, 64, bilinear, use_residual=use_residual)
        
        self.outc = OutConv(64, n_classes)
        
        # Sharpen kernel for residual learning
        self.sharpen_kernel = nn.Parameter(torch.tensor([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ]).float().reshape(1, 1, 3, 3), requires_grad=True)
        
        # Learnable weight for residual sharpening
        self.sharpen_weight = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        
        # Initialize weights for better training convergence
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize weights using kaiming initialization for better convergence.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask=None):
        # Keep original input for residual connection
        original_x = x
        
        # If mask is provided and we're using it as a channel
        if mask is not None and self.with_mask_channel:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # Add channel dimension if missing
            # Concatenate mask as additional channel
            x = torch.cat([x, mask], dim=1)
        
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with attention
        x = self.up1(x5, x4, mask)
        x = self.up2(x, x3, mask)
        x = self.up3(x, x2, mask)
        x = self.up4(x, x1, mask)
        
        # Output layer
        enhanced = self.outc(x)
        
        # Apply sharpening using the learned kernel only to residual
        residual = enhanced - original_x
        
        # Blend the original image with the enhanced version based on the mask
        if mask is not None:
            # Ensure mask has correct dimensions
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            # Resize mask if needed
            if mask.size()[2:] != original_x.size()[2:]:
                mask = F.interpolate(mask, size=original_x.size()[2:], mode='bilinear', align_corners=False)
            
            # Apply mask to blend original and enhanced images
            result = original_x + residual * mask
        else:
            result = enhanced
            
        return result
        
    def export_to_onnx(self, save_path, input_shape=(1, 3, 256, 256), mask_shape=(1, 1, 256, 256)):
        """
        Export the model to ONNX format for faster inference in production.
        
        Args:
            save_path: Path to save the ONNX model
            input_shape: Shape of the input tensor (batch_size, channels, height, width)
            mask_shape: Shape of the mask tensor (batch_size, channels, height, width)
        """
        dummy_input = torch.randn(input_shape)
        dummy_mask = torch.randn(mask_shape)
        
        # Set model to evaluation mode
        self.eval()
        
        # Export the model
        torch.onnx.export(
            self,
            (dummy_input, dummy_mask),
            save_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input', 'mask'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'mask': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'}
            }
        )
        
        print(f"Model exported to {save_path}") 