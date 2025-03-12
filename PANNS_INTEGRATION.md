# PANNs Integration for SoundWatch

This document provides instructions for integrating the PANNs (Pretrained Audio Neural Networks) model with SoundWatch.

## Overview

PANNs are a family of audio neural networks trained on AudioSet for audio pattern recognition. They can recognize a wide variety of sounds, making them well-suited for the SoundWatch application.

## Requirements

- Python 3.6 or higher
- PyTorch 1.6 or higher
- Libraries: librosa, numpy, pandas, h5py, wget
- Approximately 500MB of disk space for model files

## Setup Instructions

### Automatic Setup

We provide setup scripts for both Windows and Linux/macOS environments:

#### Windows (PowerShell)

```powershell
# Navigate to the server directory
cd server

# Run the setup script
.\setup_panns.ps1
```

#### Linux/macOS (Bash)

```bash
# Navigate to the server directory
cd server

# Make the script executable
chmod +x setup_panns.sh

# Run the setup script
./setup_panns.sh
```

The setup scripts will:
1. Check if you're running in a virtual environment
2. Verify Python version
3. Create necessary directories
4. Install required packages
5. Download model files
6. Verify the installation

### Manual Setup

If you prefer to set up manually:

1. Create directories for models and assets:
   ```bash
   mkdir -p models assets
   ```

2. Install required packages:
   ```bash
   pip install wget h5py torch numpy pandas librosa
   ```

3. Download the model file manually:
   ```bash
   python server/download_panns_files.py
   ```

## Testing the Installation

To test if the PANNs model is properly installed and functioning:

```bash
python server/test_panns.py
```

You can also test with your own audio file:

```bash
python server/test_panns.py --audio_file path/to/your/audio.wav
```

## Running SoundWatch with PANNs

After setup is complete, you can run SoundWatch with PANNs enabled:

```bash
python server.py --use_panns
```

## Configuration

PANNs can be configured in several ways:

### Model Selection

PANNs includes two model architectures:
- CNN9 (default): Smaller, faster, but slightly less accurate
- CNN13: Larger, slower, but more accurate

You can select which model to use in `panns_model.py`.

### Labels

The system will look for labels in the following files (in order):
1. `assets/class_labels_indices.csv` (AudioSet labels)
2. `assets/audioset_labels.csv` (Alternative format)
3. `assets/domestic_labels.csv` (Common household sounds subset)

### Inference Settings

In `server.py`, you can adjust settings for PANNs inference:
- `top_k`: Number of top predictions to return (default: 5)
- `threshold`: Minimum confidence threshold (default: 0.1)

## Troubleshooting

### Missing Model Files

If you see errors about missing model files, run the setup script again:
```bash
./setup_panns.sh  # or setup_panns.ps1 on Windows
```

### Import Errors

If you encounter import errors, make sure you've installed all required packages:
```bash
pip install wget h5py torch numpy pandas librosa
```

### CUDA Issues

If you encounter CUDA-related errors but have a CUDA-capable GPU:
1. Install the CUDA version compatible with your PyTorch installation
2. Set environment variable: `export CUDA_VISIBLE_DEVICES=0`

If you want to force CPU-only mode, set:
```bash
export CUDA_VISIBLE_DEVICES=""
```

### Memory Errors

The CNN13 model requires more memory. If you encounter memory errors:
1. Switch to the CNN9 model which is more memory-efficient
2. Reduce batch size if processing multiple audio files

## Additional Resources

- [PANNs Paper](https://arxiv.org/abs/1912.10211)
- [Original PANNs Repository](https://github.com/qiuqiangkong/audioset_tagging_cnn/)
- [AudioSet Dataset](https://research.google.com/audioset/) 