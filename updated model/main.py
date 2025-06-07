#!/usr/bin/env python
"""
Selective Image Sharpening - Main entry point
This script provides a unified interface to all functionality:
- Training
- Inference
- Interactive demo
"""

import os
import sys
import argparse
from importlib import import_module

def parse_args():
    parser = argparse.ArgumentParser(
        description='Selective Image Sharpening',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Train a new model
  python main.py train --data_dir data/images --output_dir output

  # Run inference on an image
  python main.py inference --input_image test.jpg --mask_image mask.png --model_path checkpoints/best_model.pth

  # Launch interactive demo
  python main.py demo --model_path checkpoints/best_model.pth
  
  # Launch advanced UI
  python main.py advanced --model_path checkpoints/best_model.pth
  
  # Launch simplified UI (Vietnamese)
  python main.py simple --model_path checkpoints/best_model.pth
        '''
    )
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Training mode
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training images')
    train_parser.add_argument('--mask_dir', type=str, default=None, help='Directory containing masks (optional)')
    train_parser.add_argument('--output_dir', type=str, default='output', help='Directory to save outputs')
    train_parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    train_parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    train_parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    train_parser.add_argument('--use_mask_channel', action='store_true', default=True, help='Concatenate mask as input channel')
    train_parser.add_argument('--save_every', type=int, default=5, help='Save model every N epochs')
    train_parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    train_parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    train_parser.add_argument('--amp', action='store_true', help='Use mixed precision training')
    
    # Inference mode
    inference_parser = subparsers.add_parser('inference', help='Run inference on images')
    inference_parser.add_argument('--input_image', type=str, required=True, help='Path to input image')
    inference_parser.add_argument('--mask_image', type=str, required=True, help='Path to mask image')
    inference_parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    inference_parser.add_argument('--output_dir', type=str, default='results', help='Directory to save output images')
    inference_parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    inference_parser.add_argument('--use_mask_channel', action='store_true', default=True, help='Use mask as input channel')
    inference_parser.add_argument('--use_poisson_blend', action='store_true', default=True, help='Use Poisson blending for seamless results')
    
    # Demo mode
    demo_parser = subparsers.add_parser('demo', help='Launch interactive demo')
    demo_parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth', help='Path to trained model checkpoint')
    demo_parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    
    # Advanced UI mode
    advanced_parser = subparsers.add_parser('advanced', help='Launch advanced user interface')
    advanced_parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth', help='Path to trained model checkpoint')
    advanced_parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    
    # Simple UI mode (Vietnamese)
    simple_parser = subparsers.add_parser('simple', help='Launch simplified user interface with Vietnamese language')
    simple_parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth', help='Path to trained model checkpoint')
    simple_parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    
    # Batch processing mode
    batch_parser = subparsers.add_parser('batch', help='Process multiple images')
    batch_parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images')
    batch_parser.add_argument('--mask_dir', type=str, required=True, help='Directory containing mask images')
    batch_parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    batch_parser.add_argument('--output_dir', type=str, default='results', help='Directory to save output images')
    batch_parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    batch_parser.add_argument('--use_mask_channel', action='store_true', default=True, help='Use mask as input channel')
    batch_parser.add_argument('--use_poisson_blend', action='store_true', default=True, help='Use Poisson blending for seamless results')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.mode is None:
        print("Error: No mode specified. Use one of: train, inference, demo, advanced, simple, batch")
        sys.exit(1)
    
    # Import the appropriate module based on mode
    if args.mode == 'train':
        from train import main as train_main
        train_main(args)
    
    elif args.mode == 'inference':
        from inference import main as inference_main
        inference_main(args)
    
    elif args.mode == 'demo':
        from demo import main as demo_main
        demo_main(args)
        
    elif args.mode == 'advanced':
        try:
            from advanced_ui import main as advanced_main
            advanced_main()
        except ImportError as e:
            print(f"Error: {e}")
            print("The advanced UI requires additional dependencies. Please run:")
            print("pip install torch torchvision matplotlib opencv-python")
            sys.exit(1)
    
    elif args.mode == 'simple':
        try:
            from simple_ui import main as simple_main
            simple_main()
        except ImportError as e:
            print(f"Error: {e}")
            print("The simplified UI requires additional dependencies. Please run:")
            print("pip install torch torchvision matplotlib opencv-python")
            sys.exit(1)
        
    elif args.mode == 'batch':
        from batch_process import main as batch_main
        batch_main(args)
    
    else:
        print(f"Error: Unknown mode '{args.mode}'")
        sys.exit(1)

if __name__ == "__main__":
    main() 