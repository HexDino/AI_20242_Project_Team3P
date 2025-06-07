import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
import cv2
from skimage import exposure

class SharpeningDataset(Dataset):
    """
    Dataset for selective image sharpening.
    Generates pairs of blurred and original images with masks.
    """
    def __init__(self, image_dir, mask_dir=None, transform=None, create_masks=True, mask_size_range=(0.1, 0.4)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.create_masks = create_masks
        self.mask_size_range = mask_size_range
        
        # Get all image files
        self.image_files = [f for f in os.listdir(image_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        # Get mask files if provided
        if mask_dir:
            self.mask_files = [f for f in os.listdir(mask_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        else:
            self.mask_files = None
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load original image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        
        # Apply transform if provided
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        
        # Load or generate mask
        if self.mask_dir and self.mask_files:
            # Use provided masks
            mask_path = os.path.join(self.mask_dir, self.mask_files[idx % len(self.mask_files)])
            mask = Image.open(mask_path).convert('L')
            
            # Resize mask to match image size
            mask = mask.resize((img.shape[2], img.shape[1]), Image.LANCZOS)
            mask = transforms.ToTensor()(mask)
        elif self.create_masks:
            # Generate random mask
            mask = self._generate_random_mask(img.shape[1], img.shape[2])
        else:
            # No mask, use full image
            mask = torch.ones((1, img.shape[1], img.shape[2]), dtype=torch.float32)
        
        # Create a blurred version of the image (input)
        blurred_img = self._create_blurred_image(img, mask)
        
        return {
            'input': blurred_img,
            'target': img,
            'mask': mask,
            'filename': self.image_files[idx]
        }
    
    def _generate_random_mask(self, height, width):
        """Generate a random rectangular or elliptical mask"""
        mask = np.zeros((height, width), dtype=np.float32)
        
        # Randomly choose between rectangle and ellipse
        shape_type = np.random.choice(['rectangle', 'ellipse'])
        
        # Determine mask size as a percentage of image size
        mask_size = np.random.uniform(self.mask_size_range[0], self.mask_size_range[1])
        
        # Calculate mask dimensions
        mask_height = int(height * np.sqrt(mask_size))
        mask_width = int(width * np.sqrt(mask_size))
        
        # Random position
        y = np.random.randint(0, height - mask_height)
        x = np.random.randint(0, width - mask_width)
        
        if shape_type == 'rectangle':
            # Create rectangular mask
            mask[y:y+mask_height, x:x+mask_width] = 1.0
        else:
            # Create elliptical mask
            center_y = y + mask_height // 2
            center_x = x + mask_width // 2
            radius_y = mask_height // 2
            radius_x = mask_width // 2
            
            y_indices, x_indices = np.ogrid[:height, :width]
            distance = ((y_indices - center_y) ** 2) / (radius_y ** 2) + ((x_indices - center_x) ** 2) / (radius_x ** 2)
            mask[distance <= 1.0] = 1.0
            
        # Apply Gaussian blur to the edges for soft mask
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        return torch.from_numpy(mask).unsqueeze(0)
        
    def _create_blurred_image(self, img, mask):
        """Create a blurred version of the image, applying blur only to the masked region"""
        # Convert to numpy for processing
        img_np = img.permute(1, 2, 0).numpy()
        mask_np = mask.squeeze().numpy()
        
        # Create blurred version (using Gaussian blur)
        blurred = cv2.GaussianBlur(img_np, (7, 7), 0)
        
        # Apply blur to masked region, keep original elsewhere
        inverse_mask = 1.0 - mask_np
        result = img_np * inverse_mask[:, :, np.newaxis] + blurred * mask_np[:, :, np.newaxis]
        
        # Convert back to tensor
        return torch.from_numpy(result).permute(2, 0, 1)


def create_data_loaders(image_dir, mask_dir=None, batch_size=8, num_workers=4, train_ratio=0.8):
    """
    Create train and validation data loaders.
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Create dataset
    dataset = SharpeningDataset(image_dir, mask_dir, transform)
    
    # Split into train and validation sets
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def poisson_blend(original, enhanced, mask):
    """
    Apply Poisson blending for seamless integration of enhanced region with original image.
    """
    # Convert tensors to numpy arrays
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().permute(1, 2, 0).numpy()
    if isinstance(enhanced, torch.Tensor):
        enhanced = enhanced.detach().cpu().permute(1, 2, 0).numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().squeeze().numpy()
    
    # Scale to 0-255 and convert to uint8
    original = (original * 255).astype(np.uint8)
    enhanced = (enhanced * 255).astype(np.uint8)
    
    # Create a binary mask (Poisson blending requires binary mask)
    binary_mask = (mask > 0.5).astype(np.uint8) * 255
    
    # Apply Poisson blending
    # center = (original.shape[1] // 2, original.shape[0] // 2)
    try:
        result = cv2.seamlessClone(enhanced, original, binary_mask, (original.shape[1]//2, original.shape[0]//2), cv2.NORMAL_CLONE)
    except:
        # Fallback to alpha blending if Poisson blending fails
        alpha = mask[:, :, np.newaxis]
        result = enhanced * alpha + original * (1 - alpha)
        result = result.astype(np.uint8)
    
    # Convert back to float in range [0, 1]
    result = result.astype(np.float32) / 255.0
    
    return result 