#!/bin/bash

# setup_panns.sh
# This script sets up the PANNs model for SoundWatch
# It downloads the necessary models and dependencies

set -e

# Display header
echo "======================================================================"
echo "                   PANNs Model Setup for SoundWatch                  "
echo "======================================================================"
echo ""

# Check if running in virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "WARNING: It's recommended to run this script inside a virtual environment."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Please activate your virtual environment and try again."
        exit 1
    fi
fi

# Check Python version
PYTHON_VERSION=$(python --version | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [[ $PYTHON_MAJOR -lt 3 || ($PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 6) ]]; then
    echo "ERROR: Python 3.6 or higher is required."
    echo "Current version: $PYTHON_VERSION"
    exit 1
fi

echo "Using Python $PYTHON_VERSION"

# Create directories
echo "Creating necessary directories..."
mkdir -p models assets

# Install requirements
echo "Installing required packages..."
pip install -q wget h5py torch numpy pandas librosa

# Run the download script
echo "Downloading PANNs model files..."
python download_panns_files.py

# Verify files
echo "Verifying downloaded files..."

# Function to check if a file exists and display its status
check_file() {
    local file=$1
    local name=$2
    if [[ -f "$file" ]]; then
        local size=$(du -h "$file" | cut -f1)
        echo "[✓] $name ($size)"
    else
        echo "[✗] $name (missing)"
        MISSING_FILES=1
    fi
}

MISSING_FILES=0

# Check essential files
check_file "models/Cnn9_GMP_64x64_300000_iterations_mAP=0.37.pth" "CNN9 Model"
check_file "models/class_labels_indices.csv" "Labels file"
check_file "assets/scalar.h5" "Scalar file"
check_file "assets/audioset_labels.csv" "AudioSet labels"
check_file "assets/domestic_labels.csv" "Domestic labels"

# Check if we can import the required modules
echo "Testing imports..."
python -c "import torch; import librosa; import numpy as np; import h5py; import pandas as pd; print('All imports successful')" || { echo "Error importing required modules"; exit 1; }

# Test the PANNs model file
echo "Testing PANNs model..."

if [[ "$MISSING_FILES" -eq 1 ]]; then
    echo "WARNING: Some files are missing. Please run this script again or download them manually."
else
    echo "All required files are present."
fi

echo ""
echo "======================================================================"
echo "                          Setup Complete                             "
echo "======================================================================"
echo ""
echo "You can now run SoundWatch with PANNs by executing:"
echo "python server.py --use_panns"
echo "" 