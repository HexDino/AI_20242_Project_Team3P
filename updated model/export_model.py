#!/usr/bin/env python
"""
Export a trained PyTorch model to ONNX format for deployment

This script handles:
1. Loading a trained PyTorch model checkpoint
2. Converting and optimizing it to ONNX format
3. Testing the exported ONNX model with sample inputs
"""

import os
import argparse
import torch
import onnx
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from models.unet_attention import UNetAttention

def parse_args():
    parser = argparse.ArgumentParser(description='Export Selective Sharpening Model to ONNX')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output_path', type=str, default='model.onnx', help='Path to save ONNX model')
    parser.add_argument('--input_shape', type=str, default='1,3,512,512', help='Input shape (batch_size,channels,height,width)')
    parser.add_argument('--test_image', type=str, default=None, help='Optional image to test the exported model')
    parser.add_argument('--test_mask', type=str, default=None, help='Optional mask to test the exported model')
    parser.add_argument('--use_mask_channel', action='store_true', default=True, help='Whether model uses mask as input channel')
    parser.add_argument('--optimize', action='store_true', help='Apply ONNX optimizations')
    
    return parser.parse_args()

def load_model(model_path, device='cpu', use_mask_channel=True):
    """Load the trained model from checkpoint"""
    print(f"Loading model from {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model = UNetAttention(n_channels=3, n_classes=3, with_mask_channel=use_mask_channel)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

def optimize_onnx_model(input_path, output_path=None):
    """Optimize the ONNX model for inference"""
    if output_path is None:
        output_path = input_path
        
    print(f"Optimizing ONNX model...")
    try:
        import onnxoptimizer
        
        model = onnx.load(input_path)
        passes = [
            "eliminate_unused_initializer",
            "eliminate_deadend",
            "eliminate_identity",
            "fuse_add_bias_into_conv",
            "fuse_bn_into_conv",
            "fuse_consecutive_squeezes",
            "fuse_consecutive_transposes",
            "fuse_matmul_add_bias_into_gemm",
            "fuse_pad_into_conv",
            "fuse_transpose_into_gemm"
        ]
        
        optimized_model = onnxoptimizer.optimize(model, passes)
        onnx.save(optimized_model, output_path)
        print(f"Optimized ONNX model saved to {output_path}")
    except ImportError:
        print("onnxoptimizer not found. Install with: pip install onnxoptimizer")
        return False
    except Exception as e:
        print(f"Failed to optimize ONNX model: {e}")
        return False
    
    return True

def test_onnx_model(model_path, image_path, mask_path):
    """Test the exported ONNX model with a sample image"""
    print(f"Testing ONNX model with sample image...")
    try:
        import onnxruntime as ort
        
        # Load sample image and mask
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).numpy()
        
        # Load and preprocess mask
        mask = Image.open(mask_path).convert('L')
        mask_tensor = transform(mask).unsqueeze(0).numpy()
        
        # Create ONNX Runtime session
        session = ort.InferenceSession(model_path)
        
        # Run inference
        input_name = session.get_inputs()[0].name
        mask_name = session.get_inputs()[1].name
        output_name = session.get_outputs()[0].name
        
        result = session.run([output_name], {input_name: input_tensor, mask_name: mask_tensor})[0]
        
        # Save result
        result_array = np.clip(result[0].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        result_img = Image.fromarray(result_array)
        
        output_dir = os.path.dirname(model_path)
        result_path = os.path.join(output_dir, 'onnx_test_result.png')
        result_img.save(result_path)
        
        print(f"ONNX model test successful. Result saved to {result_path}")
        return True
    except ImportError:
        print("onnxruntime not found. Install with: pip install onnxruntime")
        return False
    except Exception as e:
        print(f"Failed to test ONNX model: {e}")
        return False

def main():
    args = parse_args()
    
    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(',')))
    if len(input_shape) != 4:
        raise ValueError("Input shape must be in format: batch_size,channels,height,width")
    
    # Prepare mask shape based on input shape
    mask_shape = (input_shape[0], 1, input_shape[2], input_shape[3])
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load PyTorch model
    model = load_model(args.model_path, use_mask_channel=args.use_mask_channel)
    
    # Export to ONNX
    print(f"Exporting model to ONNX format with input shape {input_shape}...")
    try:
        model.export_to_onnx(
            args.output_path,
            input_shape=input_shape,
            mask_shape=mask_shape
        )
        print(f"Model exported to {args.output_path}")
    except Exception as e:
        print(f"Failed to export model: {e}")
        # Try standard export
        print("Trying standard ONNX export...")
        dummy_input = torch.randn(input_shape)
        dummy_mask = torch.randn(mask_shape)
        torch.onnx.export(
            model,
            (dummy_input, dummy_mask),
            args.output_path,
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
        print(f"Standard export successful: {args.output_path}")
    
    # Optimize ONNX model if requested
    if args.optimize:
        optimize_onnx_model(args.output_path)
    
    # Test the exported model if test image is provided
    if args.test_image and args.test_mask:
        test_onnx_model(args.output_path, args.test_image, args.test_mask)
    
    print("Export completed successfully!")
    
    # Print deployment instructions
    print("\nDeployment Instructions:")
    print("1. Install onnxruntime: pip install onnxruntime")
    print("2. Use the following Python code for inference:")
    print("""
    import onnxruntime as ort
    import numpy as np
    from PIL import Image
    import torchvision.transforms as transforms

    # Load ONNX model
    session = ort.InferenceSession("model.onnx")
    
    # Load and preprocess input image
    img = Image.open("input.jpg").convert('RGB')
    transform = transforms.ToTensor()
    input_tensor = transform(img).unsqueeze(0).numpy()
    
    # Load and preprocess mask
    mask = Image.open("mask.png").convert('L')
    mask_tensor = transform(mask).unsqueeze(0).numpy()
    
    # Run inference
    input_name = session.get_inputs()[0].name
    mask_name = session.get_inputs()[1].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: input_tensor, mask_name: mask_tensor})[0]
    
    # Post-process and save result
    result_array = np.clip(result[0].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    result_img = Image.fromarray(result_array)
    result_img.save("result.png")
    """)

if __name__ == "__main__":
    main() 