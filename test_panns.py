#!/usr/bin/env python3
"""
Test script for PANNs model integration.
This script tests whether the PANNs model can correctly load and run inference.
"""

import os
import sys
import numpy as np
import traceback

# Import our PANNs module
import panns_model

def main():
    """Main test function"""
    print("Testing PANNs model integration")
    
    # Step 1: Initialize the model
    print("\n=== Step 1: Initialize the model ===")
    if not panns_model.initialize():
        print("Failed to initialize PANNs model")
        return False
    
    # Step 2: Generate test audio data
    print("\n=== Step 2: Generate test audio data ===")
    # Generate a 1-second sine wave at 440 Hz
    sample_rate = 16000
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    print(f"Generated test audio with shape {test_audio.shape}, sample rate {sample_rate}")
    
    # Step 3: Run prediction
    print("\n=== Step 3: Run prediction ===")
    try:
        results = panns_model.predict_sound(test_audio, sample_rate, threshold=0.01)
        
        # Step 4: Display results
        print("\n=== Step 4: Results ===")
        print("Top predictions:")
        for pred in results["top_predictions"][:10]:
            print(f"  {pred['label']}: {pred['confidence']:.4f}")
        
        print("\nMapped predictions:")
        for pred in results["mapped_predictions"]:
            print(f"  {pred['label']} (from {pred['original_label']}): {pred['confidence']:.4f}")
            
        print("\nTest completed successfully!")
        return True
    except Exception as e:
        print(f"Error during prediction: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Set environment variables
    os.environ['USE_PANNS_MODEL'] = '1'
    
    # Run the test
    success = main()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 