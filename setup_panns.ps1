# setup_panns.ps1
# This script sets up the PANNs model for SoundWatch in Windows environments

# Display header
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "                   PANNs Model Setup for SoundWatch                  " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if running in a virtual environment
$inVirtualEnv = $false
if ($env:VIRTUAL_ENV) {
    Write-Host "Running in virtual environment: $env:VIRTUAL_ENV" -ForegroundColor Green
    $inVirtualEnv = $true
} else {
    Write-Host "WARNING: Not running in a virtual environment." -ForegroundColor Yellow
    $response = Read-Host "Do you want to continue? (y/n)"
    if ($response -ne "y") {
        Write-Host "Exiting. Please activate a virtual environment and try again."
        exit 1
    }
}

# Check Python version
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python (\d+)\.(\d+)\.(\d+)") {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 6)) {
            Write-Host "ERROR: Python 3.6 or higher is required." -ForegroundColor Red
            Write-Host "Current version: $pythonVersion" -ForegroundColor Red
            exit 1
        }
        
        Write-Host "Using $pythonVersion" -ForegroundColor Green
    } else {
        Write-Host "ERROR: Could not determine Python version." -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "ERROR: Python not found. Please install Python 3.6 or higher." -ForegroundColor Red
    exit 1
}

# Create directories
Write-Host "Creating necessary directories..." -ForegroundColor Cyan
if (-not (Test-Path "..\models")) {
    New-Item -Path "..\models" -ItemType Directory | Out-Null
    Write-Host "Created models directory" -ForegroundColor Green
} else {
    Write-Host "Models directory already exists" -ForegroundColor Green
}

if (-not (Test-Path "..\assets")) {
    New-Item -Path "..\assets" -ItemType Directory | Out-Null
    Write-Host "Created assets directory" -ForegroundColor Green
} else {
    Write-Host "Assets directory already exists" -ForegroundColor Green
}

# Install requirements
Write-Host "Installing required packages..." -ForegroundColor Cyan
python -m pip install wget h5py torch numpy pandas librosa

# Check if the download script exists
if (-not (Test-Path "download_panns_files.py")) {
    Write-Host "ERROR: download_panns_files.py not found." -ForegroundColor Red
    Write-Host "Please make sure you're running this script from the server directory." -ForegroundColor Red
    exit 1
}

# Run the download script
Write-Host "Downloading PANNs model files..." -ForegroundColor Cyan
python download_panns_files.py

# Verify files
Write-Host "Verifying downloaded files..." -ForegroundColor Cyan

$missingFiles = $false

# Function to check if a file exists and display its status
function Check-File {
    param (
        [string]$file,
        [string]$name
    )
    
    if (Test-Path $file) {
        # Get file size
        $size = "{0:N2} MB" -f ((Get-Item $file).Length / 1MB)
        Write-Host "[✓] $name ($size)" -ForegroundColor Green
        return $true
    } else {
        Write-Host "[✗] $name (missing)" -ForegroundColor Red
        return $false
    }
}

# Check essential files
$missingFiles = $missingFiles -or -not (Check-File "..\models\Cnn9_GMP_64x64_300000_iterations_mAP=0.37.pth" "CNN9 Model")
$missingFiles = $missingFiles -or -not (Check-File "..\models\class_labels_indices.csv" "Labels file")
$missingFiles = $missingFiles -or -not (Check-File "..\assets\scalar.h5" "Scalar file")
$missingFiles = $missingFiles -or -not (Check-File "..\assets\audioset_labels.csv" "AudioSet labels")
$missingFiles = $missingFiles -or -not (Check-File "..\assets\domestic_labels.csv" "Domestic labels")

# Check if we can import the required modules
Write-Host "Testing imports..." -ForegroundColor Cyan
try {
    python -c "import torch; import librosa; import numpy as np; import h5py; import pandas as pd; print('All imports successful')"
    Write-Host "All imports successful" -ForegroundColor Green
} catch {
    Write-Host "Error importing required modules" -ForegroundColor Red
    $missingFiles = $true
}

if ($missingFiles) {
    Write-Host "WARNING: Some files or dependencies are missing. Please run this script again or download them manually." -ForegroundColor Yellow
} else {
    Write-Host "All required files are present." -ForegroundColor Green
}

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "                          Setup Complete                             " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "You can now run SoundWatch with PANNs by executing:" -ForegroundColor Cyan
Write-Host "python server.py --use_panns" -ForegroundColor Yellow
Write-Host ""
