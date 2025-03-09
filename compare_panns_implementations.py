#!/usr/bin/env python3
"""
Script to compare our PANNs implementation with the original panns_inference package.
This helps diagnose differences between the two implementations.
"""

import os
import sys
import numpy as np
import time
import torch

# Import our implementation
import panns_model

def create_test_audio():
    """Create several test sounds to compare model outputs"""
    test_sounds = {}
    
    # Set parameters
    sr = 32000  # Sample rate
    duration = 1.0  # Duration in seconds
    
    # Create a sine wave (pure tone)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz (A4 note)
    test_sounds["sine_wave"] = sine_wave
    
    # Create white noise
    white_noise = np.random.uniform(-0.5, 0.5, size=int(sr * duration))
    test_sounds["white_noise"] = white_noise
    
    # Create a mix of frequencies (for alarm/bell-like sounds)
    bell_sound = 0.3 * np.sin(2 * np.pi * 800 * t) + 0.2 * np.sin(2 * np.pi * 1200 * t) + 0.1 * np.sin(2 * np.pi * 1600 * t)
    bell_sound = bell_sound * np.exp(-5.0 * t)  # Apply exponential decay
    test_sounds["bell_sound"] = bell_sound
    
    # Create a simulated speech envelope (amplitude modulation)
    speech_envelope = np.random.uniform(-0.3, 0.3, size=int(sr * duration))
    b = np.ones(200) / 200  # Simple moving average filter
    speech_envelope = np.convolve(speech_envelope, b, mode='same')
    carrier = np.sin(2 * np.pi * 200 * t)
    simulated_speech = 0.5 * speech_envelope * carrier
    test_sounds["simulated_speech"] = simulated_speech
    
    # Create a click train (for finger snap-like sounds)
    click_train = np.zeros(int(sr * duration))
    click_positions = [int(sr * 0.2), int(sr * 0.4), int(sr * 0.6), int(sr * 0.8)]
    click_width = int(sr * 0.01)  # 10ms click
    for pos in click_positions:
        if pos + click_width < len(click_train):
            click_train[pos:pos+click_width] = 0.8 * np.random.uniform(0.5, 1.0, size=click_width)
    test_sounds["click_train"] = click_train
    
    return test_sounds, sr

def test_original_panns(audio, sample_rate):
    """Test the original PANNs inference package"""
    print("\n=== Testing with original panns_inference ===")
    
    try:
        # Import the package
        from panns_inference import AudioTagging
        
        # Prepare audio - add batch dimension
        audio_batch = audio[np.newaxis, :]
        
        # Create AudioTagging model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        at = AudioTagging(checkpoint_path=None, device=device)
        
        # Run inference
        print("Running inference...")
        start_time = time.time()
        (clipwise_output, embedding) = at.inference(audio_batch)
        end_time = time.time()
        print(f"Inference time: {end_time - start_time:.4f} seconds")
        
        # Convert to numpy if needed
        if isinstance(clipwise_output, torch.Tensor):
            clipwise_output = clipwise_output.squeeze().detach().cpu().numpy()
        
        # Get top predictions
        print("\nTop 10 predictions:")
        sorted_indices = np.argsort(clipwise_output)[::-1]
        for i in range(10):
            idx = sorted_indices[i]
            print(f"{at.labels[idx]}: {clipwise_output[idx]:.6f}")
        
        return clipwise_output, at.labels
        
    except Exception as e:
        print(f"Error testing original PANNs inference: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_our_implementation(audio, sample_rate):
    """Test our PANNs implementation"""
    print("\n=== Testing with our implementation ===")
    
    try:
        # Initialize our model
        if not panns_model.initialize():
            print("Failed to initialize PANNs model")
            return None
        
        # Run inference
        print("Running inference...")
        start_time = time.time()
        results = panns_model.predict_sound(audio, sample_rate, threshold=0.01)
        end_time = time.time()
        print(f"Inference time: {end_time - start_time:.4f} seconds")
        
        # Print top predictions
        if "top_predictions" in results and results["top_predictions"]:
            print("\nTop predictions:")
            for pred in results["top_predictions"][:10]:
                print(f"{pred['label']}: {pred['confidence']:.6f}")
        else:
            print("No predictions above threshold")
        
        # Return raw predictions
        return results.get("raw_predictions", None)
        
    except Exception as e:
        print(f"Error testing our PANNs implementation: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_results(original_output, our_output, original_labels):
    """Compare results between original and our implementation"""
    print("\n=== Comparison ===")
    
    if original_output is None or our_output is None:
        print("Cannot compare - one or both outputs are None")
        return
    
    # Convert to numpy arrays
    if not isinstance(original_output, np.ndarray):
        original_output = np.array(original_output)
    if not isinstance(our_output, np.ndarray):
        our_output = np.array(our_output)
    
    # Calculate statistics
    mean_diff = np.mean(np.abs(original_output - our_output))
    max_diff = np.max(np.abs(original_output - our_output))
    correlation = np.corrcoef(original_output, our_output)[0, 1]
    
    print(f"Mean absolute difference: {mean_diff:.6f}")
    print(f"Maximum absolute difference: {max_diff:.6f}")
    print(f"Correlation: {correlation:.6f}")
    
    # Get the top 5 indices for each
    original_top5 = np.argsort(original_output)[::-1][:5]
    our_top5 = np.argsort(our_output)[::-1][:5]
    
    # Check how many of the top 5 predictions match
    matches = set(original_top5) & set(our_top5)
    print(f"Top 5 matches: {len(matches)}/{5}")
    
    # Print the top 5 differences
    print("\nBiggest differences (original vs ours):")
    diff = original_output - our_output
    diff_idx = np.argsort(np.abs(diff))[::-1]
    for i in range(5):
        idx = diff_idx[i]
        label = original_labels[idx] if idx < len(original_labels) else f"index_{idx}"
        print(f"{label}: {original_output[idx]:.6f} vs {our_output[idx]:.6f} (diff: {diff[idx]:.6f})")

def main():
    """Main function to run tests"""
    print("Creating test audio...")
    test_sounds, sample_rate = create_test_audio()
    
    # Run tests on each sound
    for name, audio in test_sounds.items():
        print(f"\n\n{'='*40}\nTesting with {name}\n{'='*40}")
        
        # Test with original implementation
        original_output, original_labels = test_original_panns(audio, sample_rate)
        
        # Test with our implementation
        our_output = test_our_implementation(audio, sample_rate)
        
        # Compare results
        if original_output is not None and our_output is not None:
            compare_results(original_output, our_output, original_labels)

if __name__ == "__main__":
    main() 