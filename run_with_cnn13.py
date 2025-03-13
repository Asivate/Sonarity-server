#!/usr/bin/env python3
"""
Run SoundWatch Server with CNN13 Model

This script sets up the environment to use the CNN13 model for sound recognition
and then launches the SoundWatch server.

Usage:
    python run_with_cnn13.py
"""

import os
import sys
import subprocess
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Set environment variables for using PANNs model
    os.environ["USE_PANNS_MODEL"] = "1"
    
    # Make sure we have the CNN13 model
    if not os.path.exists("models/Cnn13_GMP_64x64_520000_iterations_mAP=0.42.pth"):
        logger.warning("CNN13 model not found. Running setup_cnn13.py...")
        try:
            subprocess.run([sys.executable, "setup_cnn13.py"], check=True)
        except subprocess.CalledProcessError:
            logger.error("Failed to run setup_cnn13.py. Please run it manually.")
            return False
    
    # Create models_code directory if needed
    if not os.path.exists("models_code") or not os.path.exists("models_code/models.py"):
        logger.warning("models_code directory or models.py not found. Running setup_cnn13.py...")
        try:
            subprocess.run([sys.executable, "setup_cnn13.py"], check=True)
        except subprocess.CalledProcessError:
            logger.error("Failed to run setup_cnn13.py. Please run it manually.")
            return False
    
    # Run the server
    logger.info("Starting SoundWatch server with CNN13 model...")
    logger.info("=" * 60)
    logger.info("SOUNDWATCH SERVER WITH CNN13 MODEL")
    logger.info("=" * 60)
    logger.info("Model: Cnn13_GMP_64x64_520000_iterations_mAP=0.42.pth")
    logger.info("Model accuracy (mAP): 0.42 (higher than CNN9's 0.37)")
    logger.info("=" * 60)
    
    try:
        # Start server
        logger.info("Starting server in 3 seconds...")
        time.sleep(3)
        subprocess.run([sys.executable, "server.py"], check=True)
        return True
    except subprocess.CalledProcessError:
        logger.error("Failed to run server.py. Please check the error messages.")
        return False
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")
        return True

if __name__ == "__main__":
    main() 