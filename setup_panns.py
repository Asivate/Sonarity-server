#!/usr/bin/env python3
"""
Setup script to install the required dependencies for PANNs inference.
Run this on your VM to make sure all required packages are installed.
"""

import os
import sys
import subprocess
import importlib.util

def check_package(package_name):
    """Check if a package is installed"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_package(package_name):
    """Install a package using pip"""
    print(f"Installing {package_name}...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", package_name],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(f"Successfully installed {package_name}")
        return True
    else:
        print(f"Failed to install {package_name}")
        print(f"Error: {result.stderr}")
        return False

def setup_panns():
    """Set up all required dependencies for PANNs inference"""
    # Required packages
    required_packages = [
        "torch",
        "librosa",
        "panns_inference",
        "numpy",
        "matplotlib",
        "sounddevice",
    ]
    
    print("Checking and installing required packages...")
    
    # Check and install each package
    for package in required_packages:
        if check_package(package):
            print(f"{package} is already installed")
        else:
            if not install_package(package):
                return False
    
    # Try to import panns_inference specifically
    try:
        import panns_inference
        from panns_inference import AudioTagging
        print("Successfully imported panns_inference")
    except ImportError as e:
        print(f"Failed to import panns_inference: {e}")
        return False
    
    print("\nAll required packages installed successfully!")
    return True

if __name__ == "__main__":
    success = setup_panns()
    
    if success:
        print("\nPANNs setup completed successfully!")
        print("\nTo test the PANNs inference, run:")
        print("  python server/panns_direct_test.py")
        print("\nThen to start the server with PANNs:")
        print("  python interactive_start.py")
    else:
        print("\nPANNs setup failed. Please check the error messages above.")
    
    sys.exit(0 if success else 1) 