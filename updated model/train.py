import os
import argparse
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.cuda.amp import autocast, GradScaler

from models.unet_attention import UNetAttention
from models.losses import CombinedLoss
from utils.data_utils import create_data_loaders, poisson_blend

def parse_args():
    parser = argparse.ArgumentParser(description='Train Selective Image Sharpening Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training images')
    parser.add_argument('--mask_dir', type=str, default=None, help='Directory containing masks (optional)')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save outputs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--use_mask_channel', action='store_true', default=True, help='Concatenate mask as input channel')
    parser.add_argument('--save_every', type=int, default=5, help='Save model every N epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--amp', action='store_true', help='Use mixed precision training')
    
    return parser.parse_args()

def train(model, train_loader, val_loader, args):
    # Create directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'val'), exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    model = model.to(device)
    
    # Define loss function
    criterion = CombinedLoss(l1_weight=1.0, perceptual_weight=0.1, tv_weight=0.01)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Setup gradient scaler for mixed precision training
    scaler = GradScaler() if args.amp else None
    
    # Initialize training variables
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_val_loss = checkpoint['loss']
            
            if 'train_losses' in checkpoint:
                train_losses = checkpoint['train_losses']
                val_losses = checkpoint['val_losses']
                
            print(f"Resumed training from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {args.resume}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]') as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Get data
                input_img = batch['input'].to(device)
                target_img = batch['target'].to(device)
                mask = batch['mask'].to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass with mixed precision if enabled
                if args.amp:
                    with autocast():
                        output = model(input_img, mask)
                        loss, loss_components = criterion(output, target_img, mask)
                        
                    # Backward pass with scaling
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard forward and backward pass
                    output = model(input_img, mask)
                    loss, loss_components = criterion(output, target_img, mask)
                    loss.backward()
                    optimizer.step()
                
                # Update progress
                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item(), l1=loss_components['l1'], perceptual=loss_components['perceptual'], tv=loss_components['tv'])
                
                # Save sample images (first batch only)
                if batch_idx == 0 and epoch % args.save_every == 0:
                    save_samples(input_img, output, target_img, mask, 
                                os.path.join(args.output_dir, 'train', f'epoch_{epoch+1}.png'))
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f'Train Epoch: {epoch+1} \t Loss: {avg_train_loss:.6f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]') as pbar:
                for batch_idx, batch in enumerate(pbar):
                    # Get data
                    input_img = batch['input'].to(device)
                    target_img = batch['target'].to(device)
                    mask = batch['mask'].to(device)
                    
                    # Forward pass
                    output = model(input_img, mask)
                    
                    # Calculate loss
                    loss, _ = criterion(output, target_img, mask)
                    
                    # Update progress
                    val_loss += loss.item()
                    pbar.set_postfix(loss=loss.item())
                    
                    # Save sample validation images (first batch only)
                    if batch_idx == 0 and epoch % args.save_every == 0:
                        save_samples(input_img, output, target_img, mask, 
                                    os.path.join(args.output_dir, 'val', f'epoch_{epoch+1}.png'))
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f'Val Epoch: {epoch+1} \t Loss: {avg_val_loss:.6f}')
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save model checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'amp': args.amp,
            }, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f"Saved best model with validation loss: {best_val_loss:.6f}")
            
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'amp': args.amp,
            }, os.path.join(args.checkpoint_dir, f'model_epoch_{epoch+1}.pth'))
            
        # Plot and save loss curves
        plot_losses(train_losses, val_losses, os.path.join(args.output_dir, 'loss_curves.png'))

def save_samples(input_imgs, outputs, targets, masks, save_path):
    """Save sample images from a batch."""
    # Select a subset of images to display (maximum 4)
    num_samples = min(4, input_imgs.size(0))
    
    # Create a grid of images: input, output, target
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    for i in range(num_samples):
        # Get images
        input_img = input_imgs[i].cpu().detach().permute(1, 2, 0).numpy()
        output_img = outputs[i].cpu().detach().permute(1, 2, 0).numpy()
        target_img = targets[i].cpu().detach().permute(1, 2, 0).numpy()
        mask_img = masks[i].cpu().detach().squeeze().numpy()
        
        # Apply Poisson blending for seamless result
        blended_img = poisson_blend(input_img, output_img, mask_img)
        
        # Plot images
        if num_samples == 1:
            axes[0].imshow(input_img)
            axes[0].set_title('Input (Blurred)')
            axes[0].axis('off')
            
            axes[1].imshow(output_img)
            axes[1].set_title('Output (Sharpened)')
            axes[1].axis('off')
            
            axes[2].imshow(blended_img)
            axes[2].set_title('Blended Result')
            axes[2].axis('off')
            
            axes[3].imshow(target_img)
            axes[3].set_title('Target (Ground Truth)')
            axes[3].axis('off')
        else:
            axes[i, 0].imshow(input_img)
            axes[i, 0].set_title('Input (Blurred)')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(output_img)
            axes[i, 1].set_title('Output (Sharpened)')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(blended_img)
            axes[i, 2].set_title('Blended Result')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(target_img)
            axes[i, 3].set_title('Target (Ground Truth)')
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_losses(train_losses, val_losses, save_path):
    """Plot and save loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main(args=None):
    # Parse arguments if not provided
    if args is None:
        args = parse_args()
    
    print("Starting training with the following parameters:")
    print(f"- Data directory: {args.data_dir}")
    print(f"- Mask directory: {args.mask_dir}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Epochs: {args.epochs}")
    print(f"- Learning rate: {args.lr}")
    print(f"- Device: {args.device}")
    print(f"- Mixed precision: {'Enabled' if args.amp else 'Disabled'}")
    print(f"- Resume from: {args.resume if args.resume else 'None'}")
    
    # Check if CUDA is available when requested
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA is not available, using CPU instead")
        args.device = 'cpu'
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        args.data_dir, args.mask_dir, args.batch_size, args.num_workers
    )
    
    # Create model
    model = UNetAttention(n_channels=3, n_classes=3, bilinear=True, with_mask_channel=args.use_mask_channel)
    
    # Print model summary
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    
    # Train model
    train(model, train_loader, val_loader, args)
    
    print("Training completed!")

if __name__ == '__main__':
    main() 