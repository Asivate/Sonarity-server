#!/usr/bin/env python3
"""
Direct test of panns_inference using the same approach as the reference example.
Used to verify everything works properly on the VM.
"""

import os
import sys
import numpy as np
import librosa
import time

def create_test_audio():
    """Create some test audio samples"""
    sample_rate = 32000
    duration = 1.0
    
    # Create a sine wave (pure tone)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz = A4 note
    
    return audio, sample_rate

def direct_panns_test():
    """
    Run a direct test of panns_inference using the exact approach from the reference example
    """
    try:
        print("Testing panns_inference directly")
        
        # Import packages exactly as in the reference example
        import librosa
        import panns_inference
        from panns_inference import AudioTagging, SoundEventDetection, labels
        
        print("Successfully imported panns_inference packages")
        
        # Create test audio
        audio, sample_rate = create_test_audio()
        print(f"Created test audio: {audio.shape}, {sample_rate}Hz")
        
        # Prepare audio exactly as in the reference example
        audio = audio[None, :]  # Add batch dimension (batch_size, segment_samples)
        
        # Create AudioTagging model exactly as in the reference example
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        print('------ Audio tagging ------')
        at = AudioTagging(checkpoint_path=None, device=device)
        
        # Run inference
        start_time = time.time()
        (clipwise_output, embedding) = at.inference(audio)
        end_time = time.time()
        
        print(f"Inference completed in {end_time - start_time:.4f} seconds")
        
        # Print output shape and embedding shape
        print(f"clipwise_output shape: {clipwise_output.shape}")
        print(f"embedding shape: {embedding.shape}")
        
        # Convert to numpy if needed
        if isinstance(clipwise_output, torch.Tensor):
            clipwise_output = clipwise_output.squeeze().detach().cpu().numpy()
        
        # Print top 10 predictions
        print("\nTop 10 predictions:")
        sorted_indices = np.argsort(clipwise_output)[::-1]
        for i in range(10):
            idx = sorted_indices[i]
            print(f"{at.labels[idx]}: {clipwise_output[idx]:.6f}")
        
        # Print output range
        print(f"\nOutput range: min={np.min(clipwise_output):.6f}, max={np.max(clipwise_output):.6f}")
        
        # Try sound event detection too if available
        try:
            print('\n------ Sound event detection ------')
            sed = SoundEventDetection(checkpoint_path=None, device=device)
            framewise_output = sed.inference(audio)
            print(f"framewise_output shape: {framewise_output.shape}")
        except Exception as e:
            print(f"Sound event detection failed: {e}")
        
        print("\nDirect panns_inference test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error in direct panns_inference test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = direct_panns_test()
    sys.exit(0 if success else 1) 