# PANNs Integration for SoundWatch

This document explains how to set up and test the PANNs (Pretrained Audio Neural Networks) integration for SoundWatch on your virtual machine.

## What is PANNs?

PANNs are large-scale pretrained audio neural networks for audio pattern recognition. They were trained on the AudioSet dataset containing 5000+ hours of audio with 527 sound classes. PANNs can detect a wide variety of sounds with high accuracy.

## Setup Instructions

Follow these steps to set up PANNs on your virtual machine:

1. **Install Required Packages**

   Run the setup script to install all required dependencies:

   ```bash
   python server/setup_panns.py
   ```

   This will install PyTorch, librosa, panns_inference, and other required packages.

2. **Test the Direct PANNs Implementation**

   Run the direct test script to verify that PANNs works correctly:

   ```bash
   python server/panns_direct_test.py
   ```

   This script uses the same approach as the reference example and should show output similar to:

   ```
   ------ Audio tagging ------
   Top 10 predictions:
   Sine wave: 0.987654
   Music: 0.123456
   ...
   ```

3. **Test SoundWatch with PANNs**

   Run the comparison script to verify that our SoundWatch integration with PANNs works correctly:

   ```bash
   python server/compare_panns_implementations.py
   ```

   This will compare our implementation with the direct PANNs implementation.

4. **Run the Server**

   Start the SoundWatch server with PANNs:

   ```bash
   python interactive_start.py
   ```

   Then select PANNs as the sound recognition model.

## Troubleshooting

If you encounter any issues:

1. **Package Missing**

   If you get ImportError, run the setup script again:

   ```bash
   python server/setup_panns.py
   ```

2. **Model Downloading Issues**

   The PANNs model checkpoint is downloaded automatically when first used. If this fails, you may need to manually download it:

   ```bash
   mkdir -p ~/panns_data
   cd ~/panns_data
   wget https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1 -O Cnn14_mAP=0.431.pth
   ```

3. **Label Issues**

   We now use the built-in PANNs labels directly, so no more label file issues should occur.

4. **Low Confidence Predictions**

   If predictions have low confidence:
   - Make sure the audio volume is sufficient (check the dB level in logs)
   - Try different sounds - some sounds are easier for PANNs to recognize than others
   - PANNs is biased towards AudioSet categories, which may not align perfectly with everyday home sounds

## Reference Example

Our implementation follows the official PANNs inference example:

```python
import librosa
import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels

audio_path = 'examples/R9_ZSCveAHg_7s.wav'
(audio, _) = librosa.core.load(audio_path, sr=32000, mono=True)
audio = audio[None, :]  # (batch_size, segment_samples)

print('------ Audio tagging ------')
at = AudioTagging(checkpoint_path=None, device='cuda')
(clipwise_output, embedding) = at.inference(audio)
```

## Documentation

For more information, see:
- PANNs original repository: https://github.com/qiuqiangkong/audioset_tagging_cnn
- PANNs inference repository: https://github.com/qiuqiangkong/panns_inference 