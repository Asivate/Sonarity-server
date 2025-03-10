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
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
MODEL_URL = "https://zenodo.org/record/3576599/files/Cnn9_GMP_64x64_300000_iterations_mAP%3D0.37.pth?download=1"
MODEL_PATH = os.path.join(MODEL_DIR, 'Cnn9_GMP_64x64_300000_iterations_mAP=0.37.pth')

# GitHub URLs for reference files
CSV_URL = "https://raw.githubusercontent.com/yinkalario/General-Purpose-Sound-Recognition-Demo/demo2019/models/validate_meta.csv"
SCALAR_URL = "https://raw.githubusercontent.com/yinkalario/General-Purpose-Sound-Recognition-Demo/demo2019/models/scalar.h5"

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
                if total_size > 0 and downloaded % (max(total_size // 20, 1)) < block_size:
                    logger.info(f"Downloaded: {progress:.1f}% ({downloaded / (1024 * 1024):.2f} MB)")
        
        logger.info(f"Download complete: {destination}")
        return True
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return False

def download_from_github(url, destination):
    """Download a file from GitHub without progress reporting (typically small files)"""
    try:
        logger.info(f"Downloading {url} to {destination}")
        response = requests.get(url)
        response.raise_for_status()
        
        with open(destination, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Download complete: {destination}")
        return True
    except Exception as e:
        logger.error(f"Error downloading file from GitHub: {str(e)}")
        return False

def create_default_scalar(destination):
    """Create a default scalar file with zeros if download fails"""
    try:
        import h5py
        import numpy as np
        
        logger.info(f"Creating default scalar file at {destination}")
        with h5py.File(destination, 'w') as f:
            f.create_dataset('mean', data=np.zeros(64, dtype=np.float32))
            f.create_dataset('std', data=np.ones(64, dtype=np.float32))
        
        logger.info(f"Created default scalar file: {destination}")
        return True
    except Exception as e:
        logger.error(f"Error creating default scalar file: {str(e)}")
        return False

def create_default_labels(destination):
    """Create a minimal labels file if download fails"""
    try:
        logger.info(f"Creating default labels file at {destination}")
        
        # Create a minimal version with just a few common labels
        content = """index,mid,display_name
0,/m/09x0r,"Speech"
1,/m/05zppz,"Male speech, man speaking"
2,/m/02zsn,"Female speech, woman speaking"
3,/m/0ytgt,"Child speech, kid speaking"
4,/m/01h8n0,"Conversation"
5,/m/02qldy,"Narration, monologue"
6,/m/09l8g,"Throat clearing"
7,/m/0brhx,"Speech synthesizer"
8,/m/07p6fty,"Shout"
9,/m/0biw8,"Applause"
10,/m/07rkbfh,"Cheering"
11,/t/dd00004,"Baby cry, infant cry"
12,/m/0ghcn6,"Whimper"
13,/m/03qc9zr,"Screaming"
14,/m/02rtxlg,"Whispering"
15,/m/01j3sz,"Laughter"
16,/m/02yds9,"Groan"
17,/m/07r4k75,"Gasp"
18,/m/01w250,"Pant"
19,/m/07s0dtb,"Snicker"
20,/m/07sq110,"Giggle"
21,/m/0463cq4,"Sigh"
22,/m/07ppn3j,"Yawn"
23,/m/06h7j,"Sneeze"
24,/m/01d3sd,"Cough"
25,/m/0351vp,"Breathing"
26,/m/07mzm6,"Wheeze"
27,/m/01hsr_,"Snoring"
28,/m/07s04w4,"Breathe"
29,/m/07pzfmf,"Laugh"
30,/m/05kq4,"Cackle"
31,/m/02z32qm,"Hubbub, speech noise, speech babble"
"""
        
        with open(destination, 'w') as f:
            f.write(content)
        
        logger.info(f"Created default labels file: {destination}")
        return True
    except Exception as e:
        logger.error(f"Error creating default labels file: {str(e)}")
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
    
    # Download label CSV file from GitHub
    if not os.path.exists(CSV_PATH) or args.force:
        logger.info("Downloading class labels file from GitHub...")
        if not download_from_github(CSV_URL, CSV_PATH):
            # If GitHub download fails, create a default labels file
            logger.warning("GitHub download failed. Creating default labels file...")
            if not create_default_labels(CSV_PATH):
                success = False
    else:
        logger.info(f"Labels file already exists: {CSV_PATH}")
    
    # Download scalar file from GitHub
    if not os.path.exists(SCALAR_PATH) or args.force:
        logger.info("Downloading scalar file from GitHub...")
        if not download_from_github(SCALAR_URL, SCALAR_PATH):
            # If GitHub download fails, create a default scalar file
            logger.warning("GitHub download failed. Creating default scalar file...")
            if not create_default_scalar(SCALAR_PATH):
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