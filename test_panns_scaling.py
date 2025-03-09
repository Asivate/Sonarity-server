#!/usr/bin/env python3
"""
Test script to analyze PANNs model output ranges and determine if scaling is needed.
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
    
    # 3. Real-world sound simulation (filtered noise)
    # Create pink noise by filtering white noise
    noise = np.random.normal(0, 1, int(sr * duration))
    # Simple approximation of pink noise by low-pass filtering white noise
    b = np.ones(500) / 500
    pink_noise = np.convolve(noise, b, mode='same') * 0.5
    test_sounds["pink_noise"] = (pink_noise, sr)
    
    return test_sounds

def analyze_model_outputs(model=None):
    """Analyze the PANNs model output range and characteristics"""
    # Initialize the model if not provided
    if model is None:
        print("Initializing PANNs model...")
        if not panns_model.initialize():
            print("Failed to initialize PANNs model!")
            return False
    
    # Create test sounds
    print("\n=== Creating test sounds ===")
    test_sounds = create_test_sounds()
    
    # Store results for analysis
    all_outputs = []
    
    # Process each test sound
    for name, (audio, sr) in test_sounds.items():
        print(f"\n=== Processing {name} ===")
        
        # Run prediction
        results = panns_model.predict_sound(audio, sr, threshold=0.001)  # Use very low threshold
        
        # Extract raw outputs
        if "raw_predictions" in results:
            raw_outputs = results["raw_predictions"]
            if isinstance(raw_outputs, list):
                raw_outputs = np.array(raw_outputs)
            
            # Store for later analysis
            all_outputs.append(raw_outputs)
            
            # Analyze output range
            min_val = np.min(raw_outputs)
            max_val = np.max(raw_outputs)
            mean_val = np.mean(raw_outputs)
            median_val = np.median(raw_outputs)
            percentile_95 = np.percentile(raw_outputs, 95)
            
            print(f"Output range: min={min_val:.6f}, max={max_val:.6f}")
            print(f"Mean: {mean_val:.6f}, Median: {median_val:.6f}")
            print(f"95th percentile: {percentile_95:.6f}")
            
            # Count values in different ranges
            range_counts = {
                "0.0-0.01": np.sum((raw_outputs >= 0.0) & (raw_outputs < 0.01)),
                "0.01-0.05": np.sum((raw_outputs >= 0.01) & (raw_outputs < 0.05)),
                "0.05-0.1": np.sum((raw_outputs >= 0.05) & (raw_outputs < 0.1)),
                "0.1-0.2": np.sum((raw_outputs >= 0.1) & (raw_outputs < 0.2)),
                "0.2-0.5": np.sum((raw_outputs >= 0.2) & (raw_outputs < 0.5)),
                "0.5-1.0": np.sum((raw_outputs >= 0.5) & (raw_outputs <= 1.0))
            }
            
            print("Value distribution:")
            for range_name, count in range_counts.items():
                percentage = (count / len(raw_outputs)) * 100
                print(f"  {range_name}: {count} values ({percentage:.2f}%)")
    
    # If we have outputs from multiple sounds, analyze combined stats
    if len(all_outputs) > 1:
        print("\n=== Combined Analysis ===")
        combined = np.concatenate(all_outputs)
        
        # Get stats
        min_val = np.min(combined)
        max_val = np.max(combined)
        mean_val = np.mean(combined)
        median_val = np.median(combined)
        percentile_95 = np.percentile(combined, 95)
        
        print(f"Combined output range: min={min_val:.6f}, max={max_val:.6f}")
        print(f"Mean: {mean_val:.6f}, Median: {median_val:.6f}")
        print(f"95th percentile: {percentile_95:.6f}")
        
        # Calculate potential scaling factors
        if max_val < 0.5:
            suggested_scale = 1.0 / max_val if max_val > 0 else 1.0
            print(f"Suggested scaling factor: {suggested_scale:.2f}")
            print(f"This would map the maximum value {max_val:.6f} to 1.0")
        
        # Plot histogram of values
        try:
            plt.figure(figsize=(10, 6))
            plt.hist(combined, bins=50, alpha=0.7)
            plt.title('Distribution of PANNs Model Output Values')
            plt.xlabel('Output Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig('panns_output_histogram.png')
            print("Saved histogram to panns_output_histogram.png")
        except Exception as e:
            print(f"Could not create histogram: {e}")
    
    return True

if __name__ == "__main__":
    # Set environment variables
    os.environ['USE_PANNS_MODEL'] = '1'
    
    # Run the analysis
    analyze_model_outputs() 