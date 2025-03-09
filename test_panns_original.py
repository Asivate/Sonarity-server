#!/usr/bin/env python3
"""
Test script to verify PANNs model output using the original panns_inference package.
This will help us compare the direct package output with our implementation.
"""

import os
import sys
import numpy as np
import librosa
import time

def test_original_panns():
    """Test the original PANNs inference package directly"""
    print("Testing original PANNs inference package...")
    
    try:
        # Import the package
        from panns_inference import AudioTagging, SoundEventDetection, labels
        print("Successfully imported panns_inference")
        
        # Test with white noise first
        print("\n=== Testing with white noise ===")
        sr = 32000
        duration = 1.0
        audio = np.random.uniform(-0.5, 0.5, size=int(sr * duration))
        audio = audio[np.newaxis, :]  # Add batch dimension
        
        # Create AudioTagging model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        at = AudioTagging(checkpoint_path=None, device=device)
        print("Model loaded")
        
        # Run inference
        print("Running inference on white noise...")
        start_time = time.time()
        (clipwise_output, embedding) = at.inference(audio)
        end_time = time.time()
        print(f"Inference time: {end_time - start_time:.4f} seconds")
        
        # Convert to numpy if needed
        if isinstance(clipwise_output, torch.Tensor):
            clipwise_output = clipwise_output.squeeze().detach().cpu().numpy()
        
        # Get top predictions
        print("\nTop 10 predictions (white noise):")
        sorted_indices = np.argsort(clipwise_output)[::-1]
        for i in range(10):
            idx = sorted_indices[i]
            print(f"{at.labels[idx]}: {clipwise_output[idx]:.6f}")
        
        # Now test with a sine wave (pure tone)
        print("\n=== Testing with pure sine wave ===")
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz (A4 note)
        sine_wave = sine_wave[np.newaxis, :]  # Add batch dimension
        
        # Run inference
        print("Running inference on sine wave...")
        (clipwise_output, embedding) = at.inference(sine_wave)
        
        # Convert to numpy if needed
        if isinstance(clipwise_output, torch.Tensor):
            clipwise_output = clipwise_output.squeeze().detach().cpu().numpy()
        
        # Get top predictions
        print("\nTop 10 predictions (sine wave):")
        sorted_indices = np.argsort(clipwise_output)[::-1]
        for i in range(10):
            idx = sorted_indices[i]
            print(f"{at.labels[idx]}: {clipwise_output[idx]:.6f}")
        
        # Create a more complex sound (mix of frequencies)
        print("\n=== Testing with complex tone ===")
        complex_tone = 0.3 * np.sin(2 * np.pi * 800 * t) + 0.2 * np.sin(2 * np.pi * 1200 * t) + 0.1 * np.sin(2 * np.pi * 1600 * t)
        complex_tone = complex_tone[np.newaxis, :]  # Add batch dimension
        
        # Run inference
        print("Running inference on complex tone...")
        (clipwise_output, embedding) = at.inference(complex_tone)
        
        # Convert to numpy if needed
        if isinstance(clipwise_output, torch.Tensor):
            clipwise_output = clipwise_output.squeeze().detach().cpu().numpy()
        
        # Get top predictions
        print("\nTop 10 predictions (complex tone):")
        sorted_indices = np.argsort(clipwise_output)[::-1]
        for i in range(10):
            idx = sorted_indices[i]
            print(f"{at.labels[idx]}: {clipwise_output[idx]:.6f}")
        
        print("\nOriginal PANNs inference test completed successfully!")
        
    except Exception as e:
        print(f"Error testing original PANNs inference: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    # Import torch here to avoid potential import issues
    import torch
    
    # Run the test
    success = test_original_panns()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 