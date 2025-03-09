#!/usr/bin/env python3
"""
PANNs (Pretrained Audio Neural Networks) Model Implementation for SoundWatch

This module provides functionality to integrate PANNs models into the SoundWatch
application for sound recognition.

PANNs is a collection of pretrained audio neural networks trained on the AudioSet dataset.
Author: Qiuqiang Kong
Source: https://github.com/qiuqiangkong/audioset_tagging_cnn
"""

import os
import time
import traceback
import numpy as np
import torch
import librosa
from typing import Dict, List, Tuple, Optional, Union
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Global variables
USE_PANNS_MODEL = False  # Controls whether PANNs is used as the primary model
PANNS_MODEL = None  # The PANNs model instance
PANNS_MODEL_TYPE = None  # The type of PANNs model being used

# Choose from different PANNs variants
# CNN10 - Lighter, faster model with good performance
# CNN14 - Larger model with better accuracy
# CNN14_16k - 16kHz variant compatible with SoundWatch's 16kHz audio
DEFAULT_MODEL_TYPE = "CNN14_16k"  # Default model type


def check_panns_availability():
    """
    Check if the required packages for PANNs are available
    
    Returns:
        bool: True if all required packages are available, False otherwise
    """
    try:
        import panns_inference
        return True
    except ImportError:
        logger.warning("panns_inference package not found. PANNs model will not be available.")
        logger.warning("To install: pip install panns-inference librosa")
        return False


def map_panns_labels_to_homesounds(predictions: List[Dict], threshold: float = 0.05) -> List[Dict]:
    """
    Map PANNs labels to homesounds categories
    
    Args:
        predictions (List[Dict]): List of predictions from the PANNs model
        threshold (float): Confidence threshold for mapping
        
    Returns:
        List[Dict]: Mapped predictions with homesounds categories
    """
    # PANNs predicts AudioSet classes which need to be mapped to homesounds categories
    mapping = {
        # Speech and human sounds
        "Speech": "speech",
        "Male speech, man speaking": "speech",
        "Female speech, woman speaking": "speech",
        "Child speech, kid speaking": "speech",
        "Conversation": "speech",
        "Narration, monologue": "speech",
        "Babbling": "speech",
        "Crowd": "speech",
        "Shout": "speech",
        "Hubbub, speech noise, speech babble": "speech",
        
        # Doorbell related sounds
        "Doorbell": "doorbell",
        "Ding-dong": "doorbell",
        "Buzzer": "doorbell",
        "Reversing beeps": "doorbell",
        "Bell": "doorbell",
        
        # Knocker sounds
        "Knock": "knock",
        "Tapping": "knock",
        "Tap": "knock",
        
        # Baby crying/baby sounds
        "Crying, sobbing": "baby",
        "Baby cry, infant cry": "baby",
        "Whimper": "baby",
        
        # Alarms and smoke alarms
        "Alarm": "fire-alarm",
        "Siren": "fire-alarm",
        "Fire alarm": "fire-alarm",
        "Smoke detector, smoke alarm": "fire-alarm",
        "Carbon monoxide detector, CO detector": "fire-alarm",
        "Emergency vehicle": "fire-alarm",
        
        # Sink running water
        "Water tap, faucet": "sink-water-running",
        "Sink (filling or washing)": "sink-water-running",
        "Water": "sink-water-running",
        "Liquid": "sink-water-running",
        
        # Dog barking
        "Bark": "dog-bark",
        "Howl": "dog-bark",
        "Dog": "dog-bark",
        "Canidae, dogs, wolves": "dog-bark",
        "Yip": "dog-bark",
        
        # Phone ring
        "Telephone": "phone-ring",
        "Telephone bell ringing": "phone-ring",
        "Ringtone": "phone-ring",
        "Cellphone": "phone-ring",
        "Mobile phone": "phone-ring",
        
        # Finger snapping
        "Finger snapping": "finger-snap",
        "Snap": "finger-snap",
        "Click": "finger-snap",
        
        # Music
        "Music": "music",
        "Musical instrument": "music",
        "Singing": "music",
        "Guitar": "music",
        "Piano": "music",
        "Percussion": "music",
        "Bass guitar": "music",
        "Drum": "music",
        "Synthesizer": "music",
        
        # Beeping sounds
        "Beep, bleep": "beep",
        "Microwave oven": "beep",
        "Electronic device": "beep",
        "Computer": "beep",
        
        # Alarms and sirens
        "Siren": "siren",
        "Police car (siren)": "siren",
        "Ambulance (siren)": "siren",
        "Fire truck, fire engine (siren)": "siren",
        "Civil defense siren": "siren",
        "Buzzer": "siren",
        
        # Sine Wave (for Fire/Smoke Alarm)
        "Sine wave": "fire-alarm",
    }
    
    # Special handling for finger snapping with lower threshold
    finger_snap_prediction = None
    finger_snap_confidence = 0
    
    mapped_predictions = []
    
    for pred in predictions:
        # Extract label and confidence
        panns_label = pred["label"]
        confidence = pred["confidence"]
        
        # Special case for finger snapping
        if "Finger snap" in panns_label or panns_label == "Snap" or panns_label == "Click":
            if confidence > finger_snap_confidence and confidence > 0.03:  # Lower threshold for finger snap
                finger_snap_prediction = {
                    "original_label": panns_label,
                    "label": "finger-snap",
                    "confidence": confidence
                }
                finger_snap_confidence = confidence
        
        # Check if this label is mapped
        for key in mapping:
            if key in panns_label:
                # Only add if confidence is above threshold
                if confidence >= threshold:
                    mapped_predictions.append({
                        "original_label": panns_label,
                        "label": mapping[key],
                        "confidence": confidence
                    })
                break
                
    # Add finger snap if found
    if finger_snap_prediction is not None:
        mapped_predictions.append(finger_snap_prediction)
    
    # Sort mapped predictions by confidence
    mapped_predictions = sorted(mapped_predictions, key=lambda x: x["confidence"], reverse=True)
    
    return mapped_predictions


def load_panns_model(model_type=DEFAULT_MODEL_TYPE):
    """
    Load the PANNs model for audio recognition
    
    Args:
        model_type (str): The type of PANNs model to load
        
    Returns:
        tuple: (model, success_flag)
    """
    global PANNS_MODEL, PANNS_MODEL_TYPE
    
    try:
        from panns_inference import AudioTagging
        
        print(f"Loading PANNs model: {model_type}")
        
        # Check for available device (CPU or CUDA)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Initialize model
        at = AudioTagging(checkpoint_path=None, device=device)
        PANNS_MODEL = at
        PANNS_MODEL_TYPE = model_type
        
        # Test model with a silent audio sample
        silent_audio = np.zeros((1, 16000), dtype=np.float32)  # 1 second of silence
        with torch.no_grad():
            _ = at.inference(silent_audio)
        
        print(f"PANNs model loaded successfully: {model_type}")
        return at, True
        
    except Exception as e:
        print(f"Error loading PANNs model: {e}")
        traceback.print_exc()
        return None, False


def predict_sound(audio_data, sample_rate, threshold=0.05, top_k=5):
    """
    Process audio input and return predictions using PANNs
    
    Args:
        audio_data (np.ndarray): Audio data as a numpy array
        sample_rate (int): Sample rate of the audio data
        threshold (float): Confidence threshold for predictions
        top_k (int): Number of top predictions to return
        
    Returns:
        dict: Dictionary with raw and mapped predictions
    """
    global PANNS_MODEL
    
    try:
        start_time = time.time()
        
        # Ensure the audio data is the right shape and type
        if len(audio_data.shape) > 1:
            # If multi-channel, convert to mono by averaging
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalize audio data
        if np.abs(audio_data).max() > 1.0:
            audio_data = audio_data / np.abs(audio_data).max()
        
        # Force float32 precision for audio data
        audio_data = audio_data.astype(np.float32)
        
        # Calculate RMS and dB for silence detection
        rms = np.sqrt(np.mean(audio_data**2))
        db_level = 20 * np.log10(rms) if rms > 0 else -100
        
        print(f"Audio data shape: {audio_data.shape}, sample rate: {sample_rate}")
        
        # Check for silence first (very low volume)
        if db_level < -65:
            print(f"Silence detected (db level: {db_level:.2f})")
            return {
                "top_predictions": [{"label": "Silence", "confidence": 0.95}],
                "mapped_predictions": [{"original_label": "Silence", "label": "silence", "confidence": 0.95}],
                "raw_predictions": None  # No raw predictions for silence
            }
            
        # PANNs expects a batch dimension and 32kHz sample rate
        # Add batch dimension
        audio_data = audio_data[np.newaxis, :]
        
        # Resample if needed - PANNs models expects 32kHz (or 16kHz for the 16k variant)
        target_sr = 16000 if PANNS_MODEL_TYPE == "CNN14_16k" else 32000
        if sample_rate != target_sr:
            print(f"Resampling from {sample_rate}Hz to {target_sr}Hz")
            audio_data = librosa.resample(
                audio_data[0], orig_sr=sample_rate, target_sr=target_sr
            )[np.newaxis, :]
        
        # Run inference
        with torch.no_grad():
            clipwise_output, _ = PANNS_MODEL.inference(audio_data)
            
        # Process predictions to get the top results
        clipwise_output = clipwise_output.cpu().numpy()[0]
        sorted_indexes = np.argsort(clipwise_output)[::-1]
        
        # Create top predictions list
        top_predictions = []
        for idx in sorted_indexes[:top_k]:
            label = PANNS_MODEL.labels[idx]
            confidence = float(clipwise_output[idx])
            
            # Only include predictions above threshold
            if confidence >= threshold:
                top_predictions.append({
                    "label": label,
                    "confidence": confidence
                })
        
        # Print top predictions
        print("===== PANNS MODEL RAW PREDICTIONS =====")
        for i in range(min(5, len(top_predictions))):
            if i < len(top_predictions):
                print(f"  {top_predictions[i]['label']}: {top_predictions[i]['confidence']:.6f}")
        
        # Map PANNs labels to homesounds categories
        mapped_predictions = map_panns_labels_to_homesounds(top_predictions, threshold)
        
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"PANNs prediction completed in {elapsed:.2f} seconds")
        
        return {
            "top_predictions": top_predictions,
            "mapped_predictions": mapped_predictions,
            "raw_predictions": clipwise_output
        }
    
    except Exception as e:
        print(f"Critical error in PANNs prediction: {str(e)}")
        traceback.print_exc()
        return {
            "top_predictions": [{"label": "Error", "confidence": 0.0}],
            "mapped_predictions": [],
            "raw_predictions": None
        }


def initialize():
    """
    Initialize the PANNs model and set global variables
    
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    global USE_PANNS_MODEL
    
    # Check if PANNs is available
    if not check_panns_availability():
        USE_PANNS_MODEL = False
        return False
    
    # Load PANNs model
    model, success = load_panns_model()
    
    if success:
        USE_PANNS_MODEL = True
        return True
    else:
        USE_PANNS_MODEL = False
        return False


if __name__ == "__main__":
    # Simple test for the module
    initialize()
    if USE_PANNS_MODEL:
        # Generate a test tone
        sr = 16000
        duration = 1.0  # seconds
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Generate a 440 Hz sine wave
        test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # Run prediction
        results = predict_sound(test_audio, sr)
        print("Test prediction results:")
        for pred in results["top_predictions"]:
            print(f"  {pred['label']}: {pred['confidence']:.4f}")
        
        print("\nMapped predictions:")
        for pred in results["mapped_predictions"]:
            print(f"  {pred['label']} (from {pred['original_label']}): {pred['confidence']:.4f}") 