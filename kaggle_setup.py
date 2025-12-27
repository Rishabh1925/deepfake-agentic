#!/usr/bin/env python3
"""
Quick setup script for Kaggle environment
Run this first in your Kaggle notebook
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages in Kaggle"""
    print("Installing required packages...")
    
    # Packages that might not be pre-installed in Kaggle
    packages = [
        'facenet-pytorch',
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"{package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")

def check_gpu():
    """Check if GPU is available"""
    import torch
    
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("No GPU available, using CPU")
        return False

def check_dataset():
    """Check if dataset is properly mounted"""
    data_paths = [
        "/kaggle/input/video-data-sample/data/real",
        "/kaggle/input/video-data-sample/data/fake"
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            files = os.listdir(path)
            video_files = [f for f in files if f.endswith(('.mp4', '.avi', '.mov'))]
            print(f"Found {len(video_files)} videos in {path}")
        else:
            print(f"Path not found: {path}")
            return False
    
    return True

def main():
    """Main setup function"""
    print("E-Raksha Kaggle Setup")
    print("="*40)
    
    # Install requirements
    install_requirements()
    print()
    
    # Check GPU
    gpu_available = check_gpu()
    print()
    
    # Check dataset
    dataset_ok = check_dataset()
    print()
    
    if dataset_ok:
        print("Setup complete! Ready to run training.")
        print("\nNext steps:")
        print("1. Run the complete training script")
        print("2. Download the trained model weights")
        print("3. Use in your local E-Raksha project")
    else:
        print("Setup incomplete. Please check dataset mounting.")
    
    print("="*40)

if __name__ == "__main__":
    main()