import os
import argparse
import glob
import time
from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

from models.unet_attention import UNetAttention
from utils.data_utils import poisson_blend
from inference import load_image, load_mask, sharpen_image, visualize_results

def parse_args():
    parser = argparse.ArgumentParser(description='Batch Selective Image Sharpening')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory containing mask images')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save output images')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--use_mask_channel', action='store_true', default=True, help='Use mask as input channel')
    parser.add_argument('--use_poisson_blend', action='store_true', default=True, help='Use Poisson blending for seamless results')
    parser.add_argument('--match_filenames', action='store_true', help='Match input and mask files by filename rather than index')
    parser.add_argument('--save_visualizations', action='store_true', help='Save visualization images with grid of input, mask, output')
    
    return parser.parse_args()

def find_matching_files(input_dir, mask_dir, match_by_filename=False):
    """Find input and mask file pairs."""
    # Find all image files in input directory
    input_files = sorted(glob.glob(os.path.join(input_dir, '*.jpg')) + 
                          glob.glob(os.path.join(input_dir, '*.jpeg')) + 
                          glob.glob(os.path.join(input_dir, '*.png')) + 
                          glob.glob(os.path.join(input_dir, '*.bmp')))
    
    # Find all mask files in mask directory
    mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.jpg')) + 
                         glob.glob(os.path.join(mask_dir, '*.jpeg')) + 
                         glob.glob(os.path.join(mask_dir, '*.png')) + 
                         glob.glob(os.path.join(mask_dir, '*.bmp')))
    
    if not input_files:
        raise ValueError(f"No image files found in {input_dir}")
    
    if not mask_files:
        raise ValueError(f"No mask files found in {mask_dir}")
    
    if match_by_filename:
        # Match files by filename (without extension)
        pairs = []
        input_basenames = {Path(f).stem: f for f in input_files}
        mask_basenames = {Path(f).stem: f for f in mask_files}
        
        # Find common basenames
        common_names = set(input_basenames.keys()) & set(mask_basenames.keys())
        
        if not common_names:
            print("Warning: No matching filenames found between input and mask directories.")
            print("Falling back to index-based matching.")
            return [(input_files[i], mask_files[i % len(mask_files)]) for i in range(len(input_files))]
        
        # Create pairs with matching filenames
        pairs = [(input_basenames[name], mask_basenames[name]) for name in sorted(common_names)]
        return pairs
    else:
        # Match files by index (cycling through masks if needed)
        return [(input_files[i], mask_files[i % len(mask_files)]) for i in range(len(input_files))]

def process_image_pair(model, input_path, mask_path, output_dir, args):
    """Process a single input image and mask pair."""
    # Load input image
    input_image = load_image(input_path)
    
    # Get image dimensions for mask resizing
    input_size = (input_image.shape[3], input_image.shape[2])  # width, height
    
    # Load and resize mask
    mask = load_mask(mask_path, size=input_size)
    
    # Apply sharpening
    sharpened, result = sharpen_image(model, input_image, mask, args)
    
    # Create output filename based on input filename
    input_filename = os.path.basename(input_path)
    output_base = os.path.splitext(input_filename)[0]
    
    # Save result image
    result_path = os.path.join(output_dir, f"{output_base}_sharpened.png")
    img_result = Image.fromarray((result[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    img_result.save(result_path)
    
    # Save visualization if requested
    if args.save_visualizations:
        viz_path = os.path.join(output_dir, f"{output_base}_visualization.png")
        visualize_results(input_image, sharpened, result, mask, viz_path)
    
    return result_path

def main(args=None):
    """Main function for batch processing."""
    # Parse arguments if not provided
    if args is None:
        args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device setup
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA is not available, using CPU instead.")
        device = 'cpu'
    args.device = device
    
    # Load model
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model = UNetAttention(n_channels=3, n_classes=3, with_mask_channel=args.use_mask_channel)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # Find input and mask pairs
    print(f"Finding image pairs between {args.input_dir} and {args.mask_dir}")
    file_pairs = find_matching_files(args.input_dir, args.mask_dir, args.match_filenames)
    print(f"Found {len(file_pairs)} image pairs to process")
    
    # Process all image pairs
    print("Starting batch processing...")
    start_time = time.time()
    
    results = []
    for input_path, mask_path in tqdm(file_pairs, desc="Processing images"):
        try:
            output_path = process_image_pair(model, input_path, mask_path, args.output_dir, args)
            results.append((input_path, output_path))
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\nBatch processing complete!")
    print(f"Processed {len(results)} images in {elapsed_time:.2f} seconds")
    print(f"({elapsed_time/len(results):.2f} seconds per image)")
    print(f"Results saved to {args.output_dir}")

if __name__ == '__main__':
    main() 