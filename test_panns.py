#!/usr/bin/env python3
"""
Test script for PANNs model integration in SoundWatch.

This script loads the PANNs model and runs a simple inference test on a sample audio file.
It's useful for verifying that the model and all dependencies are correctly set up.

Usage:
    python test_panns.py [--audio_file AUDIO_FILE]

Example:
    python test_panns.py --audio_file sample.wav
"""

import os
import sys
import argparse
import librosa
import numpy as np
import torch
import pandas as pd
import h5py
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('test_panns')

try:
    from panns_model import PANNsModelInference, predict_with_panns, get_available_labels
    logger.info("Successfully imported PANNs model modules")
except ImportError as e:
    logger.error(f"Failed to import PANNs model modules: {str(e)}")
    logger.error("Make sure you're running this script from the server directory")
    sys.exit(1)

def load_audio(audio_path, sr=22050, duration=5):
    """
    Load audio file for testing
    """
    try:
        logger.info(f"Loading audio file: {audio_path}")
        if os.path.exists(audio_path):
            audio, _ = librosa.load(audio_path, sr=sr, duration=duration)
            logger.info(f"Loaded audio with shape: {audio.shape}, duration: {len(audio)/sr:.2f}s")
            return audio
        else:
            logger.error(f"Audio file not found: {audio_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading audio: {str(e)}")
        return None

def generate_test_audio(duration=5, sr=22050):
    """
    Generate a test audio signal (white noise) if no audio file is provided
    """
    logger.info(f"Generating test audio: {duration}s of white noise")
    # Generate white noise
    audio = np.random.uniform(-0.1, 0.1, size=int(duration * sr))
    # Add some sine waves at different frequencies to make it more interesting
    t = np.arange(0, duration, 1/sr)
    for freq in [440, 880, 1760]:  # A4, A5, A6
        audio += 0.1 * np.sin(2 * np.pi * freq * t)
    # Normalize
    audio = audio / np.max(np.abs(audio))
    return audio

def test_panns_model(audio_data=None, audio_file=None):
    """
    Test the PANNs model by running inference on audio data
    """
    logger.info("Testing PANNs model...")
    
    # Load audio data if not provided
    if audio_data is None:
        if audio_file:
            audio_data = load_audio(audio_file)
        
        if audio_data is None:
            logger.info("No valid audio provided, generating test audio")
            audio_data = generate_test_audio()
    
    # Get available labels
    try:
        labels = get_available_labels()
        if labels:
            logger.info(f"Found {len(labels)} labels. Sample labels: {labels[:5]}")
        else:
            logger.warning("No labels found")
    except Exception as e:
        logger.error(f"Error getting labels: {str(e)}")
    
    # Initialize and test PANNs model
    try:
        # Use the predict function
        logger.info("Running prediction with PANNs model...")
        start_time = time.time()
        
        predictions = predict_with_panns(
            audio_data=audio_data,
            top_k=5,
            threshold=0.1
        )
        
        end_time = time.time()
        
        if predictions:
            logger.info(f"Prediction successful in {end_time - start_time:.2f} seconds")
            logger.info(f"Top 5 predictions:")
            for i, (label, score) in enumerate(predictions):
                logger.info(f"  {i+1}. {label}: {score:.4f}")
            return True
        else:
            logger.error("No predictions returned")
            return False
            
    except Exception as e:
        logger.error(f"Error testing PANNs model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def check_model_files():
    """
    Check if necessary model files exist
    """
    logger.info("Checking for required model files...")
    
    # Define paths relative to the workspace root
    root_dir = Path(__file__).resolve().parent.parent
    model_dir = root_dir / "models"
    asset_dir = root_dir / "assets"
    
    # Files to check
    files_to_check = [
        (model_dir / "Cnn9_GMP_64x64_300000_iterations_mAP=0.37.pth", "CNN9 Model"),
        (model_dir / "class_labels_indices.csv", "Class Labels CSV"),
        (asset_dir / "scalar.h5", "Scalar file"),
        (asset_dir / "audioset_labels.csv", "AudioSet Labels"),
        (asset_dir / "domestic_labels.csv", "Domestic Labels")
    ]
    
    all_files_exist = True
    for file_path, file_name in files_to_check:
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"✓ {file_name} exists ({size_mb:.2f} MB)")
        else:
            logger.error(f"✗ {file_name} missing: {file_path}")
            all_files_exist = False
    
    return all_files_exist

def main():
    parser = argparse.ArgumentParser(description="Test PANNs model integration")
    parser.add_argument("--audio_file", type=str, help="Path to audio file for testing")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("PANNs Model Integration Test".center(60))
    logger.info("=" * 60)
    
    # Check dependencies
    try:
        logger.info("Checking dependencies...")
        import librosa
        import torch
        import pandas as pd
        import h5py
        logger.info("All required packages are installed")
    except ImportError as e:
        logger.error(f"Missing dependencies: {str(e)}")
        logger.error("Please run setup_panns.sh or setup_panns.ps1 to install dependencies")
        return False
    
    # Check if required model files exist
    if not check_model_files():
        logger.warning("Some model files are missing. The test may fail.")
    
    # Test the model
    success = test_panns_model(audio_file=args.audio_file)
    
    if success:
        logger.info("=" * 60)
        logger.info("PANNs Model Test: SUCCESS".center(60))
        logger.info("=" * 60)
        return 0
    else:
        logger.error("=" * 60)
        logger.error("PANNs Model Test: FAILED".center(60))
        logger.error("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 