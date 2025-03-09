#!/usr/bin/env python3
"""
Comprehensive test script for PANNs model integration.
This script tests the PANNs model with detailed diagnostics.
"""

import os
import sys
import numpy as np
import traceback
import torch
import json

# Import our PANNs module
import panns_model

def test_model():
    """Test the model initialization and inference"""
    print("\n=== Testing PANNs Model Initialization ===")
    if not panns_model.initialize():
        print("ERROR: Failed to initialize PANNs model")
        return False
    
    print("\nModel initialized successfully!")
    
    # Check if we have the labels file
    home_dir = os.path.expanduser("~")
    labels_path = os.path.join(home_dir, "panns_data", "audioset_labels.json")
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        print(f"Labels file found with {len(labels)} classes")
    else:
        print("WARNING: Labels file not found")
    
    # Check if PANNS_MODEL is properly initialized
    if panns_model.PANNS_MODEL is not None:
        print(f"PANNS_MODEL object type: {type(panns_model.PANNS_MODEL)}")
        
        # Check if it has labels attribute or method
        if hasattr(panns_model.PANNS_MODEL, 'labels'):
            if callable(getattr(panns_model.PANNS_MODEL, 'labels')):
                print("PANNS_MODEL.labels is a callable method")
            else:
                print("PANNS_MODEL.labels is an attribute")
                print(f"Number of labels: {len(panns_model.PANNS_MODEL.labels)}")
        else:
            print("WARNING: PANNS_MODEL does not have 'labels' attribute or method")
    else:
        print("ERROR: PANNS_MODEL is None")
        return False
    
    return True

def generate_test_audio():
    """Generate a test audio signal"""
    print("\n=== Generating Test Audio ===")
    sample_rate = 16000
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Generate a 440 Hz sine wave
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    print(f"Test audio shape: {test_audio.shape}, sample rate: {sample_rate}")
    print(f"Test audio data type: {test_audio.dtype}")
    print(f"Test audio min: {np.min(test_audio)}, max: {np.max(test_audio)}")
    
    return test_audio, sample_rate

def test_raw_inference():
    """Test raw inference directly with the PANNs model"""
    print("\n=== Testing Raw PANNs Inference ===")
    try:
        # Generate a test audio clip
        audio, sr = generate_test_audio()
        
        # Resample if needed
        if sr != 32000:
            import librosa
            print(f"Resampling from {sr}Hz to 32000Hz")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=32000)
            print(f"Resampled audio shape: {audio.shape}")
        
        # Add batch dimension
        audio_batch = audio[np.newaxis, :]
        print(f"Audio batch shape: {audio_batch.shape}")
        
        # Run raw inference
        print("Running raw inference...")
        with torch.no_grad():
            clipwise_output, embedding = panns_model.PANNS_MODEL.inference(audio_batch)
        
        # Check output types and shapes
        if isinstance(clipwise_output, torch.Tensor):
            print(f"Output is a torch.Tensor: shape={clipwise_output.shape}, device={clipwise_output.device}")
            # Convert to numpy for further processing
            clipwise_output_np = clipwise_output.detach().cpu().numpy()
        else:
            clipwise_output_np = clipwise_output
            print(f"Output is type {type(clipwise_output)}: shape={clipwise_output_np.shape}")
            
        # Get top predictions
        if len(clipwise_output_np.shape) > 1:
            # If we have a batch dimension, take first element
            clipwise_output_np = clipwise_output_np.squeeze(0)
            
        print(f"Final output shape: {clipwise_output_np.shape}")
        top_indices = np.argsort(clipwise_output_np)[::-1][:5]
        
        print("\nTop 5 predictions from raw inference:")
        for i in top_indices:
            print(f"  Class {i}: {clipwise_output_np[i]:.6f}")
            
        return True
    except Exception as e:
        print(f"Error in raw inference: {e}")
        traceback.print_exc()
        return False

def test_predict_sound():
    """Test the predict_sound function"""
    print("\n=== Testing predict_sound Function ===")
    try:
        # Generate a test audio clip
        audio, sr = generate_test_audio()
        
        # Call predict_sound
        print("Calling predict_sound...")
        results = panns_model.predict_sound(audio, sr, threshold=0.01)
        
        # Check results
        if results:
            print("\npredict_sound returned successfully!")
            print("\nTop predictions:")
            for pred in results["top_predictions"][:5]:
                print(f"  {pred['label']}: {pred['confidence']:.4f}")
            
            print("\nMapped predictions:")
            for pred in results["mapped_predictions"]:
                print(f"  {pred['label']} (from {pred['original_label']}): {pred['confidence']:.4f}")
            
            return True
        else:
            print("ERROR: predict_sound returned empty results")
            return False
    except Exception as e:
        print(f"Error in predict_sound: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=== Comprehensive PANNs Model Test ===")
    
    # Test model initialization
    if not test_model():
        print("Model initialization test failed")
        return False
    
    # Test raw inference
    if not test_raw_inference():
        print("Raw inference test failed")
        return False
    
    # Test predict_sound function
    if not test_predict_sound():
        print("predict_sound test failed")
        return False
    
    print("\n=== All tests completed successfully! ===")
    return True

if __name__ == "__main__":
    # Set environment variables
    os.environ['USE_PANNS_MODEL'] = '1'
    
    # Run the test
    success = main()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 