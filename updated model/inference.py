import os
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt

from models.unet_attention import UNetAttention
from utils.data_utils import poisson_blend

def parse_args():
    parser = argparse.ArgumentParser(description='Selective Image Sharpening Inference')
    parser.add_argument('--input_image', type=str, required=True, help='Path to input image')
    parser.add_argument('--mask_image', type=str, required=True, help='Path to mask image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save output images')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--use_mask_channel', action='store_true', default=True, help='Use mask as input channel')
    parser.add_argument('--use_poisson_blend', action='store_true', default=True, help='Use Poisson blending for seamless results')
    
    return parser.parse_args()

def load_image(image_path, size=None):
    """Load an image and convert to tensor."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    img = Image.open(image_path).convert('RGB')
    
    if size:
        img = img.resize(size, Image.LANCZOS)
        
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    return transform(img).unsqueeze(0)

def load_mask(mask_path, size=None):
    """Load a mask image and convert to tensor."""
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask image not found at {mask_path}")
        
    mask = Image.open(mask_path).convert('L')
    
    if size:
        mask = mask.resize(size, Image.LANCZOS)
        
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    return transform(mask).unsqueeze(0)

def sharpen_image(model, input_image, mask, args):
    """Apply selective sharpening to the input image."""
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Move tensors to device
    input_tensor = input_image.to(device)
    mask_tensor = mask.to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor, mask_tensor)
    
    # Apply Poisson blending if requested
    if args.use_poisson_blend:
        # Convert tensors to numpy
        input_np = input_tensor[0].cpu().detach().permute(1, 2, 0).numpy()
        output_np = output[0].cpu().detach().permute(1, 2, 0).numpy()
        mask_np = mask_tensor[0].cpu().detach().squeeze().numpy()
        
        # Apply Poisson blending
        result_np = poisson_blend(input_np, output_np, mask_np)
        
        # Convert back to tensor
        result = torch.from_numpy(result_np).permute(2, 0, 1).unsqueeze(0)
    else:
        # Simple alpha blending
        # Ensure mask has correct dimensions for blending
        mask_tensor_resized = mask_tensor
        if mask_tensor.size()[2:] != input_tensor.size()[2:]:
            mask_tensor_resized = torch.nn.functional.interpolate(
                mask_tensor, size=input_tensor.size()[2:], mode='bilinear', align_corners=False
            )
        result = input_tensor * (1 - mask_tensor_resized) + output * mask_tensor_resized
    
    return output, result

def visualize_results(input_image, output_image, blended_image, mask, save_path):
    """Visualize and save the results."""
    # Convert tensors to numpy arrays
    input_np = input_image[0].cpu().permute(1, 2, 0).numpy()
    output_np = output_image[0].cpu().permute(1, 2, 0).numpy()
    blended_np = blended_image[0].cpu().permute(1, 2, 0).numpy()
    mask_np = mask[0].cpu().squeeze().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Plot images
    axes[0].imshow(input_np)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title('Mask')
    axes[1].axis('off')
    
    axes[2].imshow(output_np)
    axes[2].set_title('Sharpened (Direct Output)')
    axes[2].axis('off')
    
    axes[3].imshow(blended_np)
    axes[3].set_title('Final Result (Blended)')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Results saved to {save_path}")
    plt.close()
    
    # Save individual images
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_image(input_image, save_path.replace('.png', '_input.png'))
    save_image(mask, save_path.replace('.png', '_mask.png'))
    save_image(output_image, save_path.replace('.png', '_sharpened.png'))
    save_image(blended_image, save_path.replace('.png', '_result.png'))

def main(args=None):
    # Parse arguments if not provided
    if args is None:
        args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device setup
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA is not available, using CPU instead")
        args.device = 'cpu'
    
    # Load model
    print(f"Loading model from {args.model_path}")
    try:
        checkpoint = torch.load(args.model_path, map_location='cpu')
        model = UNetAttention(n_channels=3, n_classes=3, with_mask_channel=args.use_mask_channel)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load input image and mask
    try:
        print(f"Loading input image: {args.input_image}")
        input_image = load_image(args.input_image)
        
        print(f"Loading mask image: {args.mask_image}")
        original_size = (input_image.shape[3], input_image.shape[2])  # width, height
        mask = load_mask(args.mask_image, size=original_size)
    except Exception as e:
        print(f"Error loading images: {e}")
        return
    
    # Apply selective sharpening
    print("Applying selective sharpening...")
    sharpened, result = sharpen_image(model, input_image, mask, args)
    
    # Save results
    output_path = os.path.join(args.output_dir, 'sharpening_result.png')
    visualize_results(input_image, sharpened, result, mask, output_path)
    
    # Save standalone result
    result_image = Image.fromarray((result[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    result_path = os.path.join(args.output_dir, 'final_result.png')
    result_image.save(result_path)
    print(f"Final result saved to {result_path}")
    
    print("Selective sharpening completed successfully!")
    return result_path

if __name__ == '__main__':
    main() 