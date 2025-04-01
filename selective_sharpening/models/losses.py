import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features.
    """
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.weights = weights
        
        # Load pretrained VGG19
        vgg = models.vgg19(pretrained=True).features
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(*[vgg[i] for i in range(4)]))  # relu1_2
        self.layers.append(nn.Sequential(*[vgg[i] for i in range(4, 9)]))  # relu2_2
        self.layers.append(nn.Sequential(*[vgg[i] for i in range(9, 18)]))  # relu3_4
        self.layers.append(nn.Sequential(*[vgg[i] for i in range(18, 27)]))  # relu4_4
        self.layers.append(nn.Sequential(*[vgg[i] for i in range(27, 36)]))  # relu5_4
        
        # Set VGG to eval mode
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x, y, mask=None):
        # Normalize the images to match VGG input normalization
        x = self._normalize(x)
        y = self._normalize(y)
        
        # Calculate perceptual loss at each layer
        loss = 0
        for i, layer in enumerate(self.layers):
            x = layer(x)
            y = layer(y)
            
            if mask is not None:
                # Resize mask to match feature map size
                mask_resized = F.interpolate(mask, size=x.shape[2:], mode='bilinear', align_corners=False)
                # Apply mask weighting to loss
                layer_loss = F.mse_loss(x * mask_resized, y * mask_resized, reduction='sum') / (mask_resized.sum() + 1e-8)
                layer_loss += 0.1 * F.mse_loss(x * (1-mask_resized), y * (1-mask_resized), reduction='sum') / ((1-mask_resized).sum() + 1e-8)
            else:
                layer_loss = F.mse_loss(x, y)
                
            loss += self.weights[i] * layer_loss
            
        return loss
    
    def _normalize(self, x):
        # VGG normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        return (x - mean) / std


class MaskedL1Loss(nn.Module):
    """
    L1 loss weighted by mask. Higher weight on masked region, lower weight on non-masked region.
    """
    def __init__(self, masked_weight=10.0, unmasked_weight=0.1):
        super(MaskedL1Loss, self).__init__()
        self.masked_weight = masked_weight
        self.unmasked_weight = unmasked_weight
        
    def forward(self, pred, target, mask):
        # Ensure mask is the right shape
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        
        # Resize mask if needed
        if mask.size()[2:] != pred.size()[2:]:
            mask = F.interpolate(mask, size=pred.size()[2:], mode='bilinear', align_corners=False)
            
        # Calculate loss in masked region (higher weight)
        masked_loss = F.l1_loss(pred * mask, target * mask, reduction='sum') / (mask.sum() + 1e-8)
        
        # Calculate loss in non-masked region (lower weight)
        unmasked_loss = F.l1_loss(pred * (1-mask), target * (1-mask), reduction='sum') / ((1-mask).sum() + 1e-8)
        
        # Weighted sum
        total_loss = self.masked_weight * masked_loss + self.unmasked_weight * unmasked_loss
        
        return total_loss


class TotalVariationLoss(nn.Module):
    """
    Total Variation Loss for smoothness near edges of the mask.
    """
    def __init__(self):
        super(TotalVariationLoss, self).__init__()
        
    def forward(self, img, mask):
        # Ensure mask is the right shape
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
            
        # Create an expanded mask that includes a boundary region around the mask
        kernel = torch.ones(1, 1, 3, 3).to(mask.device)
        expanded_mask = F.conv2d(mask, kernel, padding=1)
        expanded_mask = (expanded_mask > 0).float()
        boundary_mask = expanded_mask - mask
        
        # Calculate total variation in the boundary region
        x_diff = img[:, :, :, 1:] - img[:, :, :, :-1]
        y_diff = img[:, :, 1:, :] - img[:, :, :-1, :]
        
        # Adjust boundary mask to match x_diff and y_diff dimensions
        x_boundary_mask = boundary_mask[:, :, :, 1:]
        y_boundary_mask = boundary_mask[:, :, 1:, :]
        
        x_diff_masked = x_diff * x_boundary_mask
        y_diff_masked = y_diff * y_boundary_mask
        
        # Sum up variation along the mask boundary
        loss = torch.sum(torch.abs(x_diff_masked)) + torch.sum(torch.abs(y_diff_masked))
        loss = loss / (x_boundary_mask.sum() + y_boundary_mask.sum() + 1e-8)
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function using L1, Perceptual, and Total Variation losses.
    """
    def __init__(self, l1_weight=1.0, perceptual_weight=0.1, tv_weight=0.01):
        super(CombinedLoss, self).__init__()
        self.l1_loss = MaskedL1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.tv_loss = TotalVariationLoss()
        
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.tv_weight = tv_weight
        
    def forward(self, pred, target, mask):
        l1 = self.l1_loss(pred, target, mask)
        perceptual = self.perceptual_loss(pred, target, mask)
        tv = self.tv_loss(pred, mask)
        
        total_loss = self.l1_weight * l1 + self.perceptual_weight * perceptual + self.tv_weight * tv
        
        return total_loss, {"l1": l1.item(), "perceptual": perceptual.item(), "tv": tv.item()} 