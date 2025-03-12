#!/usr/bin/env python3
"""
PANNs Model Downloader

This script downloads all necessary files for using the PANNs (Pretrained Audio Neural Networks)
model for sound recognition. It will download:
1. The CNN9 model file
2. The scalar file for normalization
3. The label CSV files

Usage:
    python download_panns_files.py [--force]
    
Optional arguments:
    --force: Force download even if files already exist
"""

import os
import sys
import argparse
import wget
import requests
from pathlib import Path
import time
import h5py
import numpy as np

# Define paths
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SERVER_DIR, 'models')
ASSET_DIR = os.path.join(SERVER_DIR, 'assets')

# Model URLs
MODEL_URLS = {
    'cnn9': 'https://zenodo.org/record/3987831/files/Cnn9_GMP_64x64_300000_iterations_mAP%3D0.37.pth?download=1',
    'cnn13': 'https://zenodo.org/record/3987831/files/Cnn13_mAP%3D0.423.pth?download=1',
    'cnn14': 'https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1',
}

# Label and scalar files
LABEL_URLS = {
    'audioset_labels': 'https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/metadata/class_labels_indices.csv',
    'validate_meta': 'https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/metadata/class_labels_indices.csv',
}

def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(ASSET_DIR, exist_ok=True)
    print(f"Created directories: {MODEL_DIR}, {ASSET_DIR}")

def download_file(url, output_path, force=False):
    """Download a file from URL to output_path."""
    if os.path.exists(output_path) and not force:
        print(f"File already exists at {output_path}. Use --force to download anyway.")
        return False
    
    try:
        print(f"Downloading {url} to {output_path}...")
        start_time = time.time()
        wget.download(url, out=output_path)
        elapsed_time = time.time() - start_time
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\nDownloaded {file_size_mb:.2f} MB in {elapsed_time:.2f} seconds")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def create_scalar_file():
    """Create a scalar file with the mean and standard deviation values."""
    scalar_path = os.path.join(ASSET_DIR, 'scalar.h5')
    
    if os.path.exists(scalar_path):
        print(f"Scalar file already exists at {scalar_path}")
        return
    
    # These values are from the original PANNs implementation
    mean = np.float32([
        -14.050895, -13.107869, -13.1390915, -13.255364, -13.917199,
        -14.087848, -14.855916, -15.266642,  -15.884036, -16.491768,
        -17.067415, -17.717588, -18.075916,  -18.84405,  -19.233824,
        -19.954256, -20.180824, -20.695705,  -21.031914, -21.33451,
        -21.758745, -21.917028, -22.283598,  -22.737364, -22.920172,
        -23.23437,  -23.66509,  -23.965239,  -24.580393, -24.67597,
        -25.194445, -25.55243,  -25.825129,  -26.309643, -26.703104,
        -27.28697,  -27.839067, -28.228388,  -28.746237, -29.236507,
        -29.937782, -30.755503, -31.674414,  -32.853516, -33.959763,
        -34.88149,  -35.81145,  -36.72929,   -37.746593, -39.000496,
        -40.069244, -40.947514, -41.79767,   -42.81981,  -43.8541,
        -44.895683, -46.086784, -47.255924,  -48.520145, -50.726765,
        -52.932228, -54.713795, -56.69902,   -59.078354])
    
    std = np.float32([
        22.680508, 22.13264,  21.857653, 21.656355, 21.565693, 21.525793,
        21.450764, 21.377304, 21.338581, 21.3247,   21.289171, 21.221565,
        21.175856, 21.049534, 20.954664, 20.891844, 20.849905, 20.809206,
        20.71186,  20.726717, 20.72358,  20.655743, 20.650305, 20.579372,
        20.583157, 20.604849, 20.5452,   20.561695, 20.448244, 20.46753,
        20.433657, 20.412025, 20.47265,  20.456116, 20.487215, 20.387547,
        20.331848, 20.310328, 20.292257, 20.292326, 20.241796, 20.19396,
        20.23783,  20.564362, 21.075726, 21.332186, 21.508852, 21.644777,
        21.727905, 22.251642, 22.65972,  22.800117, 22.783764, 22.78581,
        22.86413,  22.948992, 23.12939,  23.180748, 23.03542,  23.131435,
        23.454556, 23.39839,  23.254364, 23.198978])
    
    try:
        print(f"Creating scalar file at {scalar_path}...")
        with h5py.File(scalar_path, 'w') as hf:
            hf.create_dataset('mean', data=mean)
            hf.create_dataset('std', data=std)
        print(f"Created scalar file successfully")
    except Exception as e:
        print(f"Error creating scalar file: {e}")

def create_domestic_labels_file():
    """Create a domestic labels file with common household sounds."""
    domestic_labels_path = os.path.join(ASSET_DIR, 'domestic_labels.csv')
    
    if os.path.exists(domestic_labels_path):
        print(f"Domestic labels file already exists at {domestic_labels_path}")
        return
    
    # Common household sounds from AudioSet
    domestic_labels = [
        # Common domestic sounds
        ("Knock", "Knocking"),
        ("Door", "Door"), 
        ("Bell", "Bell"),
        ("Telephone", "Telephone"),
        ("Alarm", "Alarm"),
        ("Water", "Water"),
        ("Microwave", "Microwave oven"),
        ("Fridge", "Refrigerator"),
        ("Blender", "Blender"),
        ("Vacuum", "Vacuum cleaner"),
        ("Clock", "Clock"),
        ("Fan", "Fan"),
        ("Sink", "Sink (filling or washing)"),
        ("Toilet", "Toilet flush"),
        ("Door", "Door"),
        ("Doorbell", "Doorbell"),
        ("Keys", "Keys jangling"),
        ("Drawer", "Drawer open or close"),
        ("Cutlery", "Cutlery, silverware"),
        ("Dishes", "Dishes, pots, and pans"),
        ("Chop", "Chopping (food)"),
        ("Frying", "Frying (food)"),
        ("Microwave", "Microwave oven"),
        ("Water", "Water tap, faucet"),
        ("Stove", "Stove burner"),
        ("Typing", "Computer keyboard"),
        ("Mouse", "Computer mouse, computing mouse"),
        ("Printer", "Printer"),
        ("Camera", "Camera"),
        ("Telephone", "Telephone"),
        ("Cell", "Cell phone, mobile phone"),
        ("Alarm", "Alarm clock"),
        ("Clock", "Clock tick"),
        ("Tick", "Tick"),
        ("Tick-tock", "Tick-tock"),
        ("Coin", "Coin (dropping)"),
        ("Clink", "Clink"),
        ("Cash", "Cash register"),
        ("Coins", "Coins, money"),
        ("Walk", "Walk, footsteps"),
        ("Run", "Run, running"),
        ("Slam", "Slam"),
        ("Knock", "Knock"),
        ("Tap", "Tap"),
        ("Thump", "Thump, thud"),
        ("Squeak", "Squeak"),
    ]
    
    try:
        print(f"Creating domestic labels file at {domestic_labels_path}...")
        with open(domestic_labels_path, 'w') as f:
            f.write("index,mid,display_name\n")
            for i, (mid, display_name) in enumerate(domestic_labels):
                f.write(f"{i},{mid},{display_name}\n")
        print(f"Created domestic labels file with {len(domestic_labels)} labels")
    except Exception as e:
        print(f"Error creating domestic labels file: {e}")

def download_all_files(force=False):
    """Download all necessary files."""
    create_directories()
    
    # Download model files
    cnn9_path = os.path.join(MODEL_DIR, 'Cnn9_GMP_64x64_300000_iterations_mAP=0.37.pth')
    download_file(MODEL_URLS['cnn9'], cnn9_path, force)
    
    # Download labels
    audioset_labels_path = os.path.join(ASSET_DIR, 'audioset_labels.csv')
    validate_meta_path = os.path.join(ASSET_DIR, 'validate_meta.csv')
    class_labels_path = os.path.join(MODEL_DIR, 'class_labels_indices.csv')
    
    download_file(LABEL_URLS['audioset_labels'], audioset_labels_path, force)
    download_file(LABEL_URLS['validate_meta'], validate_meta_path, force)
    download_file(LABEL_URLS['audioset_labels'], class_labels_path, force)
    
    # Create scalar and domestic labels files
    create_scalar_file()
    create_domestic_labels_file()
    
    # Copy to both directories for redundancy
    if os.path.exists(audioset_labels_path):
        import shutil
        shutil.copy2(audioset_labels_path, os.path.join(MODEL_DIR, 'audioset_labels.csv'))
    
    if os.path.exists(os.path.join(ASSET_DIR, 'scalar.h5')):
        import shutil
        shutil.copy2(os.path.join(ASSET_DIR, 'scalar.h5'), os.path.join(MODEL_DIR, 'scalar.h5'))
    
    print("\nDownload complete. Files are ready for use with PANNs model.")

def main():
    parser = argparse.ArgumentParser(description='Download PANNs model files')
    parser.add_argument('--force', action='store_true', help='Force download even if files exist')
    args = parser.parse_args()
    
    download_all_files(args.force)

if __name__ == '__main__':
    main() 