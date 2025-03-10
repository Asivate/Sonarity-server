#!/usr/bin/env python3
"""
Setup Script for PANNs CNN13 Model

This script downloads and sets up the larger CNN13 model with better accuracy (mAP=0.423)
for SoundWatch. It also configures CPU optimizations for Debian 12.

Usage:
    python setup_cnn13.py [--force]
"""

import os
import sys
import shutil
import requests
import argparse
import logging
import subprocess
import multiprocessing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
MODEL_URL = "https://zenodo.org/record/3987831/files/Cnn13_mAP%3D0.423.pth?download=1"
MODEL_PATH = os.path.join(MODEL_DIR, 'Cnn13_mAP=0.423.pth')
CSV_PATH = os.path.join(MODEL_DIR, 'validate_meta.csv')
SCALAR_PATH = os.path.join(MODEL_DIR, 'scalar.h5')

def check_system():
    """Check system specifications and provide recommendations"""
    logger.info("Checking system specifications...")
    
    try:
        # Check CPU info
        cpu_count = multiprocessing.cpu_count()
        logger.info(f"CPU cores detected: {cpu_count}")
        
        # Check RAM
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
            logger.info(f"RAM detected: {ram_gb:.2f} GB")
            
            if ram_gb < 2:
                logger.warning("Low memory: CNN13 model requires at least 2GB of RAM")
            elif ram_gb < 4:
                logger.warning("Limited memory: Performance may be affected")
            else:
                logger.info("Memory is sufficient for CNN13 model")
                
            # Check available disk space
            disk_gb = psutil.disk_usage('/').free / (1024 * 1024 * 1024)
            logger.info(f"Available disk space: {disk_gb:.2f} GB")
            
            if disk_gb < 2:
                logger.warning("Low disk space: CNN13 model requires about 1.3GB")
                
        except ImportError:
            logger.warning("psutil not installed. Cannot check RAM. Install with 'pip install psutil'")
        
        # Check if we're on Debian
        try:
            # Try to detect Debian version
            if os.path.exists('/etc/os-release'):
                with open('/etc/os-release', 'r') as f:
                    os_info = f.read()
                    if 'debian' in os_info.lower():
                        logger.info("Debian OS detected")
                        if 'VERSION="12' in os_info:
                            logger.info("Debian 12 detected - good match for our optimizations")
        except Exception as e:
            logger.warning(f"Could not detect OS version: {e}")
            
        # Check for PyTorch
        try:
            import torch
            logger.info(f"PyTorch version: {torch.__version__}")
            
            if torch.cuda.is_available():
                logger.info("CUDA is available - will use GPU acceleration")
                logger.info(f"CUDA version: {torch.version.cuda}")
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("CUDA not available - using CPU only")
        except ImportError:
            logger.warning("PyTorch not installed. Install with 'pip install torch'")
            
        return True
    except Exception as e:
        logger.error(f"Error checking system: {e}")
        return False

def download_file(url, destination):
    """Download a file with progress reporting"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1 MB
        downloaded = 0
        
        logger.info(f"Downloading {url} to {destination}")
        logger.info(f"File size: {total_size / (1024 * 1024):.2f} MB")
        
        with open(destination, 'wb') as file:
            for data in response.iter_content(block_size):
                file.write(data)
                downloaded += len(data)
                progress = downloaded / total_size * 100
                # Show progress every 5%
                if downloaded % (total_size // 20) < block_size:
                    logger.info(f"Downloaded: {progress:.1f}% ({downloaded / (1024 * 1024):.2f} MB)")
        
        logger.info(f"Download complete: {destination}")
        return True
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return False

def setup_environment():
    """Set up the Python environment for CNN13 model"""
    logger.info("Setting up environment for CNN13 model...")
    
    try:
        # Install required packages
        logger.info("Checking for required packages...")
        packages = ["torch", "torchaudio", "psutil", "h5py", "librosa", "numpy", "pandas"]
        
        # Check which packages are installed
        installed_packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
        installed_packages = installed_packages.decode("utf-8").lower()
        
        missing_packages = []
        for package in packages:
            if package.lower() not in installed_packages:
                missing_packages.append(package)
        
        if missing_packages:
            logger.info(f"Installing missing packages: {', '.join(missing_packages)}")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
                logger.info("All required packages installed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error installing packages: {e}")
                logger.info("Please install the following packages manually:")
                logger.info("  pip install " + " ".join(missing_packages))
        else:
            logger.info("All required packages already installed")
            
        # Set up CPU optimization environment variables
        logger.info("Setting up CPU optimization environment variables...")
        try:
            # Get the number of CPU cores
            cpu_count = multiprocessing.cpu_count()
            logger.info(f"Setting PyTorch to use {cpu_count} CPU cores")
            
            # Create/update .env file with CPU optimization variables
            env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
            env_vars = {
                'OMP_NUM_THREADS': str(cpu_count),
                'MKL_NUM_THREADS': str(cpu_count),
                'NUMEXPR_NUM_THREADS': str(cpu_count),
                'VECLIB_MAXIMUM_THREADS': str(cpu_count),
                'OPENBLAS_NUM_THREADS': str(cpu_count),
            }
            
            # Write to .env file
            with open(env_file, 'w') as f:
                for key, value in env_vars.items():
                    f.write(f"{key}={value}\n")
            
            logger.info(f"Environment variables written to {env_file}")
        except Exception as e:
            logger.error(f"Error setting CPU optimization: {e}")
        
        return True
    except Exception as e:
        logger.error(f"Error setting up environment: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Set up PANNs CNN13 model and optimize for CPU")
    parser.add_argument('--force', action='store_true', help="Force download even if files already exist")
    args = parser.parse_args()
    
    # Check system specifications
    check_system()
    
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Set up environment
    if not setup_environment():
        logger.warning("Environment setup encountered issues, but continuing...")
    
    success = True
    
    # Download or check model file
    if not os.path.exists(MODEL_PATH) or args.force:
        logger.info("Downloading PANNs CNN13 model (this is a large 1.3GB file, please be patient)...")
        if not download_file(MODEL_URL, MODEL_PATH):
            success = False
    else:
        logger.info(f"CNN13 model file already exists: {MODEL_PATH}")
        logger.info("Use --force to re-download if needed")
    
    # Check scalar and CSV files
    for file_path, file_name in [(CSV_PATH, "class labels"), (SCALAR_PATH, "scalar")]:
        if not os.path.exists(file_path):
            logger.warning(f"{file_name.capitalize()} file not found at {file_path}")
            logger.info("Running download_panns_model.py to get required files...")
            try:
                subprocess.check_call([sys.executable, "download_panns_model.py"])
                if os.path.exists(file_path):
                    logger.info(f"{file_name.capitalize()} file downloaded successfully")
                else:
                    logger.error(f"Failed to download {file_name} file")
                    success = False
            except subprocess.CalledProcessError as e:
                logger.error(f"Error running download_panns_model.py: {e}")
                success = False
        else:
            logger.info(f"{file_name.capitalize()} file already exists: {file_path}")
    
    # Enable PANNs model in SoundWatch
    try:
        logger.info("Enabling PANNs model in SoundWatch...")
        subprocess.check_call([sys.executable, "toggle_panns.py", "on"])
        logger.info("PANNs model enabled successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error enabling PANNs model: {e}")
        success = False
    
    # Print summary
    if success:
        logger.info("")
        logger.info("="*60)
        logger.info("CNN13 MODEL SETUP COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info("Model file: " + MODEL_PATH)
        logger.info("Model size: 1.3GB")
        logger.info("mAP score: 0.423 (higher than CNN9's 0.37)")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Restart your SoundWatch server with:")
        logger.info("   python interactive_start.py")
        logger.info("2. Test the model with some audio samples")
        logger.info("3. Monitor CPU and memory usage during inference")
        logger.info("="*60)
        return 0
    else:
        logger.error("")
        logger.error("="*60)
        logger.error("CNN13 MODEL SETUP ENCOUNTERED ERRORS")
        logger.error("="*60)
        logger.error("Please check the error messages above and try again.")
        logger.error("You may need to manually download the model from:")
        logger.error("https://zenodo.org/record/3987831/files/Cnn13_mAP%3D0.423.pth")
        logger.error("And place it in the models directory.")
        logger.error("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 