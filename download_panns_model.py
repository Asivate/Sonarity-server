#!/usr/bin/env python3
"""
Download Script for PANNs Model

This script downloads the PANNs CNN9 model and required files for SoundWatch.
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
MODEL_URL = "https://zenodo.org/record/3576599/files/Cnn9_GMP_64x64_300000_iterations_mAP%3D0.37.pth?download=1"
MODEL_PATH = os.path.join(MODEL_DIR, 'Cnn9_GMP_64x64_300000_iterations_mAP=0.37.pth')

# Reference paths for copying from demo directory
REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                      'General-Purpose-Sound-Recognition-Demo',
                      'General-Purpose-Sound-Recognition-Demo-2019',
                      'models')
REF_CSV = os.path.join(REF_DIR, 'validate_meta.csv')
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

def main():
    parser = argparse.ArgumentParser(description="Download PANNs model files")
    parser.add_argument('--force', action='store_true', help="Force download even if files already exist")
    args = parser.parse_args()
    
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    success = True
    
    # Download or copy model file
    if not os.path.exists(MODEL_PATH) or args.force:
        logger.info("Downloading PANNs CNN9 model...")
        if not download_file(MODEL_URL, MODEL_PATH):
            success = False
    else:
        logger.info(f"Model file already exists: {MODEL_PATH}")
    
    # Copy label CSV file
    if not os.path.exists(CSV_PATH) or args.force:
        logger.info("Copying class labels file...")
        if os.path.exists(REF_CSV):
            if not copy_file(REF_CSV, CSV_PATH):
                success = False
        else:
            logger.error(f"Reference labels file not found: {REF_CSV}")
            success = False
    else:
        logger.info(f"Labels file already exists: {CSV_PATH}")
    
    # Copy scalar file
    if not os.path.exists(SCALAR_PATH) or args.force:
        logger.info("Copying scalar file...")
        if os.path.exists(REF_SCALAR):
            if not copy_file(REF_SCALAR, SCALAR_PATH):
                success = False
        else:
            logger.error(f"Reference scalar file not found: {REF_SCALAR}")
            success = False
    else:
        logger.info(f"Scalar file already exists: {SCALAR_PATH}")
    
    # Print summary
    if success:
        logger.info("All files downloaded/copied successfully!")
        logger.info(f"Model file: {MODEL_PATH}")
        logger.info(f"Labels file: {CSV_PATH}")
        logger.info(f"Scalar file: {SCALAR_PATH}")
        return 0
    else:
        logger.error("Some files could not be downloaded or copied.")
        logger.error("Please check the errors above and try again.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 