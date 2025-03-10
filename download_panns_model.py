#!/usr/bin/env python3
"""
Download Script for PANNs Model

This script downloads the PANNs CNN13 model and required files for SoundWatch.
"""

import os
import sys
import shutil
import requests
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
# Updated to use the larger CNN13 model (1.3GB) with better accuracy (mAP=0.42)
MODEL_URL = "https://zenodo.org/record/3987831/files/Cnn13_mAP%3D0.423.pth?download=1"
MODEL_PATH = os.path.join(MODEL_DIR, 'Cnn13_mAP=0.423.pth')

# Reference paths for copying files from csv files directory
CSV_FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csv files')
REF_CSV = os.path.join(CSV_FILES_DIR, 'validate_meta.csv')

# Alternative reference paths (fallback)
REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                      'General-Purpose-Sound-Recognition-Demo',
                      'General-Purpose-Sound-Recognition-Demo-2019',
                      'models')
ALT_REF_CSV = os.path.join(REF_DIR, 'validate_meta.csv')
REF_SCALAR = os.path.join(REF_DIR, 'scalar.h5')

# Target paths
CSV_PATH = os.path.join(MODEL_DIR, 'validate_meta.csv')
SCALAR_PATH = os.path.join(MODEL_DIR, 'scalar.h5')

def download_file(url, destination):
    """Download a file with progress reporting"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
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

def copy_file(source, destination):
    """Copy a file with error handling"""
    try:
        logger.info(f"Copying {source} to {destination}")
        shutil.copy2(source, destination)
        logger.info(f"Copy complete: {destination}")
        return True
    except Exception as e:
        logger.error(f"Error copying file: {str(e)}")
        return False

def analyze_model_file(model_path):
    """Analyze the structure of the model file to help with debugging"""
    try:
        import torch
        logger.info(f"Analyzing model file: {model_path}")
        
        # Load the model file
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Basic info
        logger.info(f"Checkpoint type: {type(checkpoint)}")
        
        # Check if it's a dictionary
        if isinstance(checkpoint, dict):
            logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            # Check for 'model' key
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
                logger.info(f"Model state_dict type: {type(state_dict)}")
                logger.info(f"Number of layers: {len(state_dict)}")
                
                # Print some layer names to help with debugging
                if len(state_dict) > 0:
                    logger.info("First 10 layer names:")
                    for i, (key, _) in enumerate(list(state_dict.items())[:10]):
                        logger.info(f"  {i+1}. {key}")
                
                # Check for fc layer naming patterns
                fc_layers = [key for key in state_dict.keys() if 'fc' in key]
                logger.info(f"Found {len(fc_layers)} FC layer keys: {fc_layers}")
                
                # Analyze shapes
                logger.info("Layer shapes:")
                for key, tensor in state_dict.items():
                    if 'fc' in key:  # Focus on fully connected layers
                        logger.info(f"  {key}: {tensor.shape}")
                
            else:
                logger.info("No 'model' key found in checkpoint")
                
                # Maybe it's directly a state dict?
                if any(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                    logger.info("Checkpoint appears to be a direct state_dict")
                    state_dict = checkpoint
                    logger.info(f"Number of layers: {len(state_dict)}")
                    
                    # Print some layer names
                    if len(state_dict) > 0:
                        logger.info("First 10 layer names:")
                        for i, (key, _) in enumerate(list(state_dict.items())[:10]):
                            logger.info(f"  {i+1}. {key}")
                    
                    # Check for fc layer naming patterns
                    fc_layers = [key for key in state_dict.keys() if 'fc' in key]
                    logger.info(f"Found {len(fc_layers)} FC layer keys: {fc_layers}")
        else:
            logger.info("Checkpoint is not a dictionary")
            
        logger.info("Analysis complete")
        return True
        
    except Exception as e:
        logger.error(f"Error analyzing model file: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Download PANNs model files")
    parser.add_argument('--force', action='store_true', help="Force download even if files already exist")
    parser.add_argument('--diagnose', action='store_true', help="Run diagnostics on the model file")
    args = parser.parse_args()
    
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # If diagnose mode is enabled, just analyze the model file
    if args.diagnose and os.path.exists(MODEL_PATH):
        logger.info("Running model diagnostics...")
        analyze_model_file(MODEL_PATH)
        return 0
    
    success = True
    
    # Download or copy model file
    if not os.path.exists(MODEL_PATH) or args.force:
        logger.info("Downloading PANNs CNN13 model (this is a large 1.3GB file, please be patient)...")
        if not download_file(MODEL_URL, MODEL_PATH):
            success = False
    else:
        logger.info(f"Model file already exists: {MODEL_PATH}")
    
    # Copy label CSV file
    if not os.path.exists(CSV_PATH) or args.force:
        logger.info("Copying class labels file...")
        # First try to copy from csv files directory
        if os.path.exists(REF_CSV):
            logger.info(f"Using labels file from csv files directory: {REF_CSV}")
            if not copy_file(REF_CSV, CSV_PATH):
                success = False
        # Fall back to alternative location if available
        elif os.path.exists(ALT_REF_CSV):
            logger.info(f"Using labels file from alternative location: {ALT_REF_CSV}")
            if not copy_file(ALT_REF_CSV, CSV_PATH):
                success = False
        else:
            logger.error(f"Labels file not found in any location. Please place validate_meta.csv in the 'csv files' directory.")
            success = False
    else:
        logger.info(f"Labels file already exists: {CSV_PATH}")
    
    # Handle scalar file - create it if it doesn't exist
    if not os.path.exists(SCALAR_PATH) or args.force:
        logger.info("Handling scalar file...")
        # Try to copy from reference location if available
        if os.path.exists(REF_SCALAR):
            logger.info(f"Copying scalar file from: {REF_SCALAR}")
            if not copy_file(REF_SCALAR, SCALAR_PATH):
                success = False
        else:
            # Generate a default scalar file
            logger.info("Reference scalar file not found. Creating a default scalar file...")
            try:
                import h5py
                import numpy as np
                
                # Default scalar values from the PANNs implementation
                mean = np.array([-6.6268077], dtype=np.float32)
                std = np.array([5.358466], dtype=np.float32)
                
                with h5py.File(SCALAR_PATH, 'w') as f:
                    f.create_dataset('mean', data=mean)
                    f.create_dataset('std', data=std)
                
                logger.info(f"Created default scalar file: {SCALAR_PATH}")
            except Exception as e:
                logger.error(f"Error creating default scalar file: {str(e)}")
                success = False
    else:
        logger.info(f"Scalar file already exists: {SCALAR_PATH}")
    
    # Print summary
    if success:
        logger.info("All files downloaded/copied successfully!")
        logger.info(f"Model file: {MODEL_PATH}")
        logger.info(f"Labels file: {CSV_PATH}")
        logger.info(f"Scalar file: {SCALAR_PATH}")
        
        # Run diagnostics if requested
        if args.diagnose or os.path.exists(MODEL_PATH):
            logger.info("Running model diagnostics...")
            analyze_model_file(MODEL_PATH)
            
        return 0
    else:
        logger.error("Some files could not be downloaded or copied.")
        logger.error("Please check the errors above and try again.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 