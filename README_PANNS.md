# PANNs Model for SoundWatch

## Quick Start

To set up the PANNs model for SoundWatch:

**Windows:**
```powershell
# Run from the server directory
.\setup_panns.ps1
```

**Linux/macOS:**
```bash
# Run from the server directory
chmod +x setup_panns.sh
./setup_panns.sh
```

**Testing:**
```bash
python test_panns.py
```

**Run SoundWatch with PANNs:**
```bash
python server.py --use_panns
```

## What is PANNs?

PANNs (Pretrained Audio Neural Networks) are a family of audio neural networks trained on AudioSet for audio pattern recognition. They can recognize a wide variety of sounds from the AudioSet ontology, which includes 527 sound classes.

## Key Files

- `panns_model.py`: Main implementation of PANNs model for SoundWatch
- `setup_panns.ps1`/`setup_panns.sh`: Setup scripts for Windows and Linux/macOS
- `download_panns_files.py`: Script to download model files
- `test_panns.py`: Test script to verify the model is working
- `PANNS_INTEGRATION.md`: Detailed documentation

## Model Performance

The CNN9 model (default) offers a good balance between speed and accuracy:
- Model size: ~75MB
- Inference time: ~100-200ms on CPU
- mAP (mean Average Precision): 0.37 on AudioSet evaluation set

## Main Features

- Audio classification for 527 sound classes
- Integration with SoundWatch's existing sound recognition pipeline
- Fallback mechanisms for different label formats
- Compatible with both CPU and GPU inference

## For Detailed Information

See `PANNS_INTEGRATION.md` for complete documentation including:
- Detailed setup instructions
- Configuration options
- Troubleshooting
- Additional resources 