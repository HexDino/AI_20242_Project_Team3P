#!/usr/bin/env python
"""
Launcher for the Advanced Selective Image Sharpening UI
This script provides a convenient way to launch the advanced UI with proper error handling
"""

import os
import sys
import subprocess
import importlib.util
import argparse

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        ('torch', 'PyTorch', 'pip install torch torchvision'),
        ('PIL', 'Pillow', 'pip install Pillow'),
        ('numpy', 'NumPy', 'pip install numpy'),
        ('matplotlib', 'Matplotlib', 'pip install matplotlib'),
        ('cv2', 'OpenCV', 'pip install opencv-python')
    ]
    
    missing_packages = []
    
    for package, name, install_cmd in required_packages:
        try:
            importlib.util.find_spec(package)
        except ImportError:
            missing_packages.append((name, install_cmd))
    
    return missing_packages

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Launch Advanced Selective Image Sharpening UI')
    parser.add_argument('--model_path', type=str, default=None, 
                       help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--skip_checks', action='store_true', 
                       help='Skip dependency checks')
    return parser.parse_args()

def main():
    """Main launcher function"""
    args = parse_args()
    
    # Print banner
    print("===============================================")
    print("  Advanced Selective Image Sharpening UI")
    print("===============================================")
    
    # Check if we're in the right directory
    if not os.path.exists('advanced_ui.py'):
        print("Error: Cannot find advanced_ui.py in the current directory.")
        print("Please run this script from the selective_sharpening directory.")
        return 1
    
    # Check dependencies if not skipped
    if not args.skip_checks:
        print("Checking dependencies...")
        missing_packages = check_dependencies()
        
        if missing_packages:
            print("\nMissing required packages:")
            for name, install_cmd in missing_packages:
                print(f"  - {name} (install with: {install_cmd})")
            
            print("\nPlease install the missing packages and try again.")
            print("You can run with --skip_checks to bypass this check.")
            return 1
        
        print("All dependencies found!")
    
    # Prepare command arguments
    cmd_args = [sys.executable, 'advanced_ui.py']
    if args.model_path:
        cmd_args.extend(['--model_path', args.model_path])
    if args.device:
        cmd_args.extend(['--device', args.device])
    
    # Launch the UI
    try:
        print("\nLaunching Advanced UI...")
        subprocess.run(cmd_args)
        return 0
    except Exception as e:
        print(f"Error launching Advanced UI: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 