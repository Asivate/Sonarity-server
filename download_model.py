#!/usr/bin/env python
"""
SoundWatch Model Downloader

This script downloads the pre-trained PANNs model for SoundWatch.
"""

import os
import sys
import argparse
import logging
import urllib.request
import hashlib
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("model_downloader")

# Model information - Updated to correct model
MODEL_URL = "https://zenodo.org/records/3576599/files/Cnn13_GMP_64x64_520000_iterations_mAP=0.42.pth?download=1"
MODEL_FILENAME = "Cnn13_GMP_64x64_520000_iterations_mAP=0.42.pth"
MODEL_SIZE = 85000000  # Approximate size in bytes
# This MD5 hash is for demonstration - you should verify the correct hash
MODEL_MD5 = None  # We'll skip MD5 verification

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def get_md5(filename):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_model(output_dir, force=False):
    """Download the pre-trained model"""
    # Create models directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Full path to the model file
    model_path = os.path.join(output_dir, MODEL_FILENAME)
    
    # Check if model already exists
    if os.path.exists(model_path) and not force:
        logger.info(f"Model already exists at {model_path}")
        
        # Skip verification since we don't have the correct MD5
        if MODEL_MD5:
            logger.info("Verifying model file integrity...")
            file_md5 = get_md5(model_path)
            if file_md5 == MODEL_MD5:
                logger.info("Model file verified successfully!")
                return True
            else:
                logger.warning("Model file verification failed. File may be corrupted.")
                logger.info("Re-downloading model...")
        else:
            logger.info("Skipping MD5 verification - hash not provided")
            logger.info("If you experience issues, try using --force to re-download")
            return True
    
    # Download the model
    logger.info(f"Downloading model from {MODEL_URL}")
    logger.info(f"This may take a while. Model size is approximately {MODEL_SIZE / 1024 / 1024:.1f} MB")
    
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
            urllib.request.urlretrieve(
                MODEL_URL, 
                model_path,
                reporthook=t.update_to
            )
        
        # Verify the downloaded model if we have a hash
        if MODEL_MD5:
            logger.info("Verifying downloaded model...")
            file_md5 = get_md5(model_path)
            if file_md5 == MODEL_MD5:
                logger.info("Model downloaded and verified successfully!")
                return True
            else:
                logger.error("Model verification failed. File may be corrupted.")
                logger.error("Please try downloading again with --force flag.")
                return False
        else:
            logger.info("Model downloaded successfully! (MD5 verification skipped)")
            return True
            
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Download pre-trained model for SoundWatch")
    parser.add_argument("--output-dir", type=str, default="models", 
                        help="Directory to save the model (default: models)")
    parser.add_argument("--force", action="store_true", 
                        help="Force download even if model already exists")
    
    args = parser.parse_args()
    
    # Get absolute path for output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_dir)
    
    # Download the model
    success = download_model(output_dir, args.force)
    
    if success:
        logger.info(f"Model saved to {os.path.join(output_dir, MODEL_FILENAME)}")
        return 0
    else:
        logger.error("Failed to download model")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 