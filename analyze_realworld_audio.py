#!/usr/bin/env python3
"""
Script to analyze real-world audio and debug PANNs model predictions.

This script creates a separate audio capturing thread like the server does,
and performs detailed analysis of the audio and predictions.
"""

import os
import sys
import numpy as np
import time
import threading
import queue
import traceback
import librosa
import sounddevice as sd

# Import our model
import panns_model

# Global variables
audio_queue = queue.Queue()
stop_event = threading.Event()

def audio_callback(indata, frames, time, status):
    """Callback for audio capture"""
    if status:
        print(f"Audio callback status: {status}")
    # Put the audio data in the queue
    audio_queue.put(indata.copy())

def audio_capture_thread(sample_rate=16000, chunk_size=8000):
    """Thread to capture audio continuously"""
    try:
        # Start the audio stream
        print(f"Starting audio capture: {sample_rate} Hz, chunk size {chunk_size}")
        with sd.InputStream(samplerate=sample_rate, channels=1, callback=audio_callback,
                           blocksize=chunk_size, dtype='float32'):
            # Keep the stream open until stopped
            while not stop_event.is_set():
                time.sleep(0.1)
    except Exception as e:
        print(f"Error in audio capture thread: {e}")
        traceback.print_exc()

def analyze_predictions(audio_buffer, sample_rate):
    """Perform detailed analysis of predictions on the audio buffer"""
    print("\n" + "=" * 50)
    print(f"Analyzing audio buffer: {len(audio_buffer)} samples, SR: {sample_rate}")
    
    # Audio stats
    rms = np.sqrt(np.mean(np.square(audio_buffer)))
    db_level = 20 * np.log10(rms) if rms > 0 else -100
    print(f"Audio stats: RMS = {rms:.6f}, dB = {db_level:.2f}")
    
    # Run prediction with very low threshold to see all outputs
    debug_threshold = 0.001
    results = panns_model.predict_sound(audio_buffer, sample_rate, threshold=debug_threshold)
    
    # If no predictions were returned
    if not results or "top_predictions" not in results:
        print("No predictions returned from model")
        return
    
    # Detailed view of predictions
    print("\nDetailed predictions (threshold = {debug_threshold}):")
    if results["top_predictions"]:
        for i, pred in enumerate(results["top_predictions"][:20]):  # Show more predictions
            print(f"  {i+1}. {pred['label']}: {pred['confidence']:.6f}")
    else:
        print("  No predictions above threshold")
    
    # Check low-confidence predictions that could be important
    if "raw_predictions" in results:
        raw = results["raw_predictions"]
        if isinstance(raw, list):
            raw = np.array(raw)
        
        # Check for predictions in specific categories
        categories_to_check = [
            "dog", "cat", "speech", "door", "knock", "doorbell", "water", 
            "alarm", "baby", "cry", "phone", "finger snap"
        ]
        
        # Get class labels if possible
        try:
            class_labels = panns_model.PANNS_MODEL.labels()
        except:
            try:
                class_labels = panns_model.PANNS_MODEL.labels
            except:
                class_labels = [f"class_{i}" for i in range(len(raw))]
        
        print("\nRelevant predictions that didn't pass threshold:")
        found = False
        for i, conf in enumerate(raw):
            label = class_labels[i] if i < len(class_labels) else f"class_{i}"
            # Check if this label contains any category we're interested in
            for category in categories_to_check:
                if category.lower() in label.lower() and conf > 0.01:
                    print(f"  {label}: {conf:.6f} - Below threshold but relevant")
                    found = True
        
        if not found:
            print("  No relevant predictions found below threshold")
    
    # Mapping analysis
    print("\nMapping analysis:")
    print(f"Original threshold in map_panns_labels_to_homesounds: 0.05")
    # Check if any predictions would be mapped with different thresholds
    thresholds_to_try = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1]
    
    for threshold in thresholds_to_try:
        # Try mapping with this threshold
        mapped = panns_model.map_panns_labels_to_homesounds(results["top_predictions"], threshold)
        print(f"\nWith threshold {threshold}:")
        if mapped:
            for pred in mapped:
                print(f"  {pred['label']} (from {pred['original_label']}): {pred['confidence']:.6f}")
        else:
            print("  No mapped predictions")

def main():
    """Main function"""
    # Initialize the PANNs model
    print("Initializing PANNs model...")
    if not panns_model.initialize():
        print("Failed to initialize PANNs model!")
        return
    
    # Start audio capture in a separate thread
    sample_rate = 16000
    chunk_size = 8000  # 0.5 seconds at 16kHz
    audio_thread = threading.Thread(target=audio_capture_thread, 
                                   args=(sample_rate, chunk_size))
    audio_thread.daemon = True
    audio_thread.start()
    
    # Buffer to accumulate audio
    buffer_duration = 1.0  # seconds
    buffer_size = int(sample_rate * buffer_duration)
    audio_buffer = np.zeros(buffer_size, dtype=np.float32)
    buffer_position = 0
    
    print(f"Listening for audio... (Press Ctrl+C to stop)")
    print(f"Buffer size: {buffer_size} samples ({buffer_duration} seconds)")
    
    try:
        while True:
            # Get audio data from queue
            try:
                audio_data = audio_queue.get(timeout=1.0)
                # Flatten if needed
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.flatten()
                
                # Add to buffer
                space_left = buffer_size - buffer_position
                to_copy = min(len(audio_data), space_left)
                audio_buffer[buffer_position:buffer_position+to_copy] = audio_data[:to_copy]
                buffer_position += to_copy
                
                # If buffer is full, process it
                if buffer_position >= buffer_size:
                    # Calculate RMS to check if audio is not just silence
                    rms = np.sqrt(np.mean(np.square(audio_buffer)))
                    db_level = 20 * np.log10(rms) if rms > 0 else -100
                    
                    if db_level > -60:  # Only analyze non-silent audio
                        print(f"Processing buffer (dB level: {db_level:.2f})")
                        analyze_predictions(audio_buffer, sample_rate)
                    else:
                        print(f"Buffer too quiet (dB level: {db_level:.2f}), skipping")
                    
                    # Reset buffer
                    buffer_position = 0
                    
                    # Sleep a bit to avoid too frequent analysis
                    time.sleep(0.5)
                
            except queue.Empty:
                pass
    
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Stop the audio thread
        stop_event.set()
        audio_thread.join(timeout=1.0)
        print("Done.")

if __name__ == "__main__":
    main() 