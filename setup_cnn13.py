#!/usr/bin/env python3
"""
Setup Script for PANNs CNN13 Model

This script downloads and sets up the larger CNN13 model with better accuracy (mAP=0.42)
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
MODEL_URL = "https://zenodo.org/records/3576599/files/Cnn13_GMP_64x64_520000_iterations_mAP=0.42.pth?download=1"
MODEL_PATH = os.path.join(MODEL_DIR, 'Cnn13_GMP_64x64_520000_iterations_mAP=0.42.pth')
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

        # Check for and set up model code for CNN13 if needed
        demo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'General-Purpose-Sound-Recognition-Demo')
        models_code_path = os.path.join(demo_path, 'General-Purpose-Sound-Recognition-Demo-2019', 'models_code')
        
        if os.path.exists(models_code_path):
            logger.info(f"Found models_code directory at {models_code_path}")
            
            # Check for modules.py and ensure it's available for our server
            models_py = os.path.join(models_code_path, 'models.py')
            if os.path.exists(models_py):
                logger.info(f"Found models.py at {models_py}")
                
                # Create a models_code directory in our server if it doesn't exist
                server_models_code = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_code')
                os.makedirs(server_models_code, exist_ok=True)
                
                # Copy the models.py file to our server
                server_models_py = os.path.join(server_models_code, 'models.py')
                if not os.path.exists(server_models_py) or args.force:
                    logger.info(f"Copying models.py to {server_models_py}")
                    shutil.copy2(models_py, server_models_py)
                else:
                    logger.info(f"models.py already exists at {server_models_py}")
                
                # Copy the __init__.py file to our server
                init_py = os.path.join(models_code_path, '__init__.py')
                if os.path.exists(init_py):
                    server_init_py = os.path.join(server_models_code, '__init__.py')
                    if not os.path.exists(server_init_py) or args.force:
                        logger.info(f"Copying __init__.py to {server_init_py}")
                        shutil.copy2(init_py, server_init_py)
                    else:
                        logger.info(f"__init__.py already exists at {server_init_py}")
                else:
                    # Create an __init__.py file if it doesn't exist
                    server_init_py = os.path.join(server_models_code, '__init__.py')
                    if not os.path.exists(server_init_py):
                        logger.info(f"Creating __init__.py at {server_init_py}")
                        with open(server_init_py, 'w') as f:
                            f.write("from .models import Cnn9_GMP_64x64\n")
                            f.write("from .models import Cnn13_GMP_64x64\n")
            else:
                logger.warning(f"models.py not found at {models_py}")
        else:
            logger.warning(f"models_code directory not found at {models_code_path}")
            logger.info("Will try to use existing model code in the server")
        
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
    
    # Create models_code directory if it doesn't exist
    models_code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_code')
    os.makedirs(models_code_dir, exist_ok=True)
    
    # Create models_code/__init__.py file if it doesn't exist
    init_py = os.path.join(models_code_dir, '__init__.py')
    if not os.path.exists(init_py) or args.force:
        logger.info(f"Creating {init_py}")
        with open(init_py, 'w') as f:
            f.write("from .models import Cnn9_GMP_64x64\n")
            f.write("from .models import Cnn13_GMP_64x64\n")
    
    # Check if we have the CNN13 model code
    models_py = os.path.join(models_code_dir, 'models.py')
    if not os.path.exists(models_py) or args.force:
        demo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'General-Purpose-Sound-Recognition-Demo')
        models_code_path = os.path.join(demo_path, 'General-Purpose-Sound-Recognition-Demo-2019', 'models_code')
        demo_models_py = os.path.join(models_code_path, 'models.py')
        
        if os.path.exists(demo_models_py):
            logger.info(f"Copying {demo_models_py} to {models_py}")
            shutil.copy2(demo_models_py, models_py)
        else:
            # Create a minimal implementation that includes Cnn13_GMP_64x64
            logger.info(f"Creating basic models.py with CNN13 implementation")
            with open(models_py, 'w') as f:
                f.write("""
import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_layer(layer, nonlinearity='leaky_relu'):
    \"\"\"Initialize a Linear or Convolutional layer. \"\"\"
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
    
    
def init_bn(bn):
    \"\"\"Initialize a Batchnorm layer. \"\"\"
    
    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')
        
        return x


class Cnn13_GMP_64x64(nn.Module):
    def __init__(self, classes_num, strong_target_training=False):
        
        super(Cnn13_GMP_64x64, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_audioset)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')

        x = torch.mean(x, dim=3)
        (x, _) = torch.max(x, dim=2)
        output = torch.sigmoid(self.fc_audioset(x))
        
        return output

class Cnn9_GMP_64x64(nn.Module):
    def __init__(self, classes_num, strong_target_training=False):
        
        super(Cnn9_GMP_64x64, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc_audioset = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_audioset)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')

        x = torch.mean(x, dim=3)
        (x, _) = torch.max(x, dim=2)
        
        output = torch.sigmoid(self.fc_audioset(x))
        
        return output
""")
    
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
        logger.info("mAP score: 0.42 (higher than CNN9's 0.37)")
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
        logger.error("https://zenodo.org/records/3576599/files/Cnn13_GMP_64x64_520000_iterations_mAP=0.42.pth")
        logger.error("And place it in the models directory.")
        logger.error("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 