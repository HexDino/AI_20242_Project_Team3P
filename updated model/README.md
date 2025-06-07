# Selective Image Sharpening AI Agent

This project implements an AI agent for selective image sharpening, which enhances the sharpness of a selected region in an image while keeping the rest unchanged. The implementation uses deep learning with a U-Net architecture and attention mechanism.

## Key Features

- Selectively sharpen regions of an image defined by a mask
- U-Net architecture with attention gates for focused enhancement
- Multiple loss functions (L1, Perceptual, Total Variation) for natural results
- Poisson blending for seamless integration of sharpened regions
- Interactive demo application with intuitive UI for easy testing
- Model export to ONNX for fast production deployment
- Batch processing capabilities for multiple images
- Unified command-line interface for all operations

## Installation

### Requirements

- Python 3.6+
- PyTorch 1.7+
- OpenCV
- NumPy
- Matplotlib
- Pillow (PIL)
- tqdm

Install the required packages:

```bash
pip install -r requirements.txt
```

### Optional Dependencies
For model optimization and deployment:
```bash
pip install onnx onnxruntime onnxoptimizer
```

## Project Structure

```
selective_sharpening/
├── data/                   # Directory for training data
├── models/                 # Model architecture definitions
│   ├── unet_attention.py   # U-Net with attention gates
│   └── losses.py           # Loss functions
├── utils/                  # Utility functions
│   └── data_utils.py       # Dataset and data handling utilities
├── checkpoints/            # Saved model checkpoints
├── output/                 # Training outputs and visualizations
├── main.py                 # Unified CLI entry point
├── train.py                # Training script
├── inference.py            # Inference script for single images
├── batch_process.py        # Process multiple images in batch
├── demo.py                 # Interactive demo application
├── export_model.py         # Export model to ONNX format
├── requirements.txt        # Package dependencies
└── README.md               # Project documentation
```

## Unified Command-Line Interface

The project provides a unified command-line interface through `main.py`:

```bash
# Train a new model
python main.py train --data_dir data/images --output_dir output

# Run inference on a single image
python main.py inference --input_image test.jpg --mask_image mask.png --model_path checkpoints/best_model.pth

# Launch interactive demo
python main.py demo --model_path checkpoints/best_model.pth

# Process multiple images in batch
python main.py batch --input_dir input_images/ --mask_dir masks/ --model_path checkpoints/best_model.pth
```

## Training

To train the model, place your training images in the `data/` directory and run:

```bash
python train.py --data_dir data/images --mask_dir data/masks --output_dir output --checkpoint_dir checkpoints --epochs 100
```

### Advanced Training Options

- **Resume Training**: `--resume checkpoints/model_epoch_50.pth`
- **Mixed Precision**: `--amp` (speeds up training on compatible GPUs)
- **Use Residual Connections**: The model uses residual connections by default for better performance

If no mask directory is provided, random masks will be generated during training.

## Inference

To apply selective sharpening to a single image:

```bash
python inference.py --input_image path/to/image.jpg --mask_image path/to/mask.png --model_path checkpoints/best_model.pth --output_dir results
```

## Batch Processing

Process multiple images with corresponding masks:

```bash
python batch_process.py --input_dir input_images/ --mask_dir masks/ --model_path checkpoints/best_model.pth --output_dir results
```

Options:
- `--match_filenames`: Match input and mask files by filename rather than index
- `--save_visualizations`: Save visualization grids with input, mask, and output 

## Interactive Demo

The demo application provides an intuitive interface to test the model on your images:

```bash
python demo.py --model_path checkpoints/best_model.pth
```

In the demo:
1. Click "Load Image" to select an image
2. Draw on the image to create a mask for the regions you want to sharpen
3. Use "Brush" mode to add to the mask, "Eraser" mode to remove from it
4. Adjust brush size using the slider
5. Click "Process Image" to apply sharpening
6. Save the result and optionally the mask using "Save Result"

## Model Export for Deployment

Export the trained model to ONNX format for deployment:

```bash
python export_model.py --model_path checkpoints/best_model.pth --output_path model.onnx --optimize
```

Options:
- `--input_shape`: Specify the input shape (default: 1,3,512,512)
- `--test_image` and `--test_mask`: Test the exported model with sample inputs
- `--optimize`: Apply ONNX optimizations for faster inference

## Performance Optimizations

This implementation includes several performance optimizations:

1. **Residual Connections**: Improved model convergence and gradient flow
2. **Kaiming Initialization**: Better weight initialization for faster training
3. **Mixed Precision Training**: Reduced memory usage and faster training on compatible GPUs
4. **ONNX Export**: Convert models to ONNX format for faster inference
5. **Batch Processing**: Efficiently process multiple images without UI overhead

## Model Architecture

The model is based on U-Net with attention gates and residual connections:

1. **Encoder**: Extracts multi-scale features using convolutional layers
2. **Attention Gates**: Focus on the masked region for selective enhancement
3. **Decoder**: Reconstructs the image with enhanced features
4. **Residual Learning**: The model learns to enhance the residual (difference) between the input and target image
5. **Skip Connections**: Preserve spatial information across encoder-decoder pairs

## Training Methodology

1. **Data Preparation**: 
   - During training, input images are artificially blurred in masked regions
   - The original image serves as the ground truth

2. **Loss Functions**:
   - Pixel-wise L1 Loss: Ensures accurate reconstruction
   - Perceptual Loss: Preserves textures and structures
   - Total Variation Loss: Maintains smoothness across region boundaries

3. **Seamless Blending**:
   - Poisson blending is applied to ensure natural transitions between enhanced and non-enhanced regions

## Results

The model produces natural-looking sharpening effects in the selected regions while maintaining the original appearance in other areas. The attention mechanism helps focus the enhancement on relevant details within the masked region.

## Acknowledgments

This project is inspired by image-to-image translation techniques and attention mechanisms in deep learning.

## License

This project is available under the MIT License. 