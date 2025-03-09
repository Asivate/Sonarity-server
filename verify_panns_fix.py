#!/usr/bin/env python3
"""
Simple script to verify that the PANNs model works with our fixes.

Run this script to test the PANNs model without starting the full server.
It will use sample sounds to check if predictions work correctly.
"""

import os
import sys
import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt
import json

# Import our PANNs model
import panns_model

def create_test_sounds():
    """Create several test sounds with different characteristics"""
    # Dictionary to store test sounds: name -> (audio, sample_rate)
    test_sounds = {}
    
    # 1. Sine wave (pure tone)
    sr = 16000
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz (A4 note)
    test_sounds["sine_wave"] = (sine_wave, sr)
    
    # 2. White noise
    white_noise = np.random.uniform(-0.5, 0.5, size=int(sr * duration))
    test_sounds["white_noise"] = (white_noise, sr)
    
    # 3. Click train (for finger snap-like sounds)
    click_train = np.zeros(int(sr * duration))
    click_positions = [int(sr * 0.2), int(sr * 0.4), int(sr * 0.6), int(sr * 0.8)]
    click_width = int(sr * 0.01)  # 10ms click
    for pos in click_positions:
        click_train[pos:pos+click_width] = 0.8 * np.random.uniform(0.5, 1.0, size=click_width)
    test_sounds["click_train"] = (click_train, sr)
    
    # 4. Simulated speech envelope (amplitude modulation)
    speech_envelope = np.random.uniform(-0.3, 0.3, size=int(sr * duration))
    # Apply low-pass filter to simulate speech envelope
    b = np.ones(200) / 200  # Simple moving average filter
    speech_envelope = np.convolve(speech_envelope, b, mode='same')
    # Modulate with a carrier
    carrier = np.sin(2 * np.pi * 200 * t)
    simulated_speech = 0.5 * speech_envelope * carrier
    test_sounds["simulated_speech"] = (simulated_speech, sr)
    
    # 5. Mix of multiple frequencies (for alarm/bell-like sounds)
    bell_sound = 0.3 * np.sin(2 * np.pi * 800 * t) + 0.2 * np.sin(2 * np.pi * 1200 * t) + 0.1 * np.sin(2 * np.pi * 1600 * t)
    # Apply exponential decay
    decay = np.exp(-5.0 * t)
    bell_sound = bell_sound * decay
    test_sounds["bell_sound"] = (bell_sound, sr)
    
    return test_sounds

def print_divider(title=None):
    """Print a divider with optional title"""
    width = 80
    if title:
        print("\n" + "=" * 10 + f" {title} " + "=" * (width - len(title) - 12) + "\n")
    else:
        print("\n" + "=" * width + "\n")

def test_panns_model():
    """Test the PANNs model with various sounds"""
    # First, initialize the model
    print_divider("INITIALIZING PANNS MODEL")
    if not panns_model.initialize():
        print("Failed to initialize PANNs model!")
        return False
    
    # Create test sounds
    print_divider("CREATING TEST SOUNDS")
    test_sounds = create_test_sounds()
    print(f"Created {len(test_sounds)} test sounds:")
    for name, (audio, sr) in test_sounds.items():
        print(f"  - {name}: {audio.shape} samples, {sr}Hz, min={audio.min():.2f}, max={audio.max():.2f}")
    
    # Try to predict each sound
    for name, (audio, sr) in test_sounds.items():
        print_divider(f"TESTING SOUND: {name.upper()}")
        
        # Predict sound
        results = panns_model.predict_sound(audio, sr, threshold=0.01)
        
        # Print raw top predictions
        print("\nTop predictions (raw):")
        for i, pred in enumerate(results["top_predictions"][:10]):
            print(f"  {i+1}. {pred['label']}: {pred['confidence']:.4f}")
        
        # Print mapped predictions
        print("\nMapped predictions:")
        if results["mapped_predictions"]:
            for i, pred in enumerate(results["mapped_predictions"][:10]):
                print(f"  {i+1}. {pred['label']} (from '{pred['original_label']}'): {pred['confidence']:.4f}")
        else:
            print("  No mapped predictions")
    
    return True

if __name__ == "__main__":
    # Set environment variables
    os.environ['USE_PANNS_MODEL'] = '1'
    
    # Run the test
    success = test_panns_model()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 