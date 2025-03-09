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
import sys
import time
import traceback
import numpy as np
import torch
import librosa
from typing import Dict, List, Tuple, Optional, Union
import logging
import wget

# Setup logging
logger = logging.getLogger(__name__)

try:
    # Import PANNs libraries - use the classes available in 0.1.1
    from panns_inference.inference import AudioTagging
    from panns_inference.models import Cnn14  # Cnn14 is available in 0.1.1
    import librosa
    
    # Flag indicating PANNs is available
    PANNS_AVAILABLE = True
except ImportError as e:
    print(f"PANNs import error: {e}")
    PANNS_AVAILABLE = False

# Global variables
USE_PANNS_MODEL = False  # Controls whether PANNs is used as the primary model
PANNS_MODEL = None  # The PANNs model instance
PANNS_MODEL_TYPE = None  # The type of PANNs model being used
MODEL_INITIALIZED = False  # Flag to check if the model is initialized

# Choose from different PANNs variants
# CNN10 - Lighter, faster model with good performance
# CNN14 - Larger model with better accuracy
# CNN14_16k - 16kHz variant compatible with SoundWatch's 16kHz audio
DEFAULT_MODEL_TYPE = "CNN14"  # The default model in 0.1.1


def check_panns_availability():
    """
    Check if the PANNs package is available
    
    Returns:
        bool: True if PANNs is available, False otherwise
    """
    global PANNS_AVAILABLE
    
    try:
        # Try to import PANNs-related packages
        from panns_inference.inference import AudioTagging
        from panns_inference.models import Cnn14  # Only check for Cnn14 which is in 0.1.1
        import librosa
        
        # If we got here, PANNs is available
        PANNS_AVAILABLE = True
        print("PANNs package is available")
        return True
    except ImportError as e:
        # Failed to import PANNs
        PANNS_AVAILABLE = False
        print(f"PANNs package is not available: {e}")
        return False


def map_panns_labels_to_homesounds(predictions: List[Dict], threshold: float = 0.05) -> List[Dict]:
    """
    Map PANNs model predictions to homesounds categories
    
    Args:
        predictions: List of prediction dictionaries from PANNs model
        threshold: Minimum confidence threshold
        
    Returns:
        List of mapped predictions with homesounds labels
    """
    # Dictionary mapping AudioSet classes to homesounds categories
    # This mapping connects the 527 AudioSet classes to our smaller set of homesound categories
    mapping = {
        # Door related sounds
        "door": "door",
        "knock": "door",
        "tap": "door",
        "knock": "door",
        "doorbell": "door",
        "door": "door",
        
        # Water related sounds
        "water": "water",
        "splash": "water",
        "water tap": "water",
        "sink (filling or washing)": "water",
        "bathtub (filling or washing)": "water",
        "flowing water": "water",
        
        # Alarm related sounds
        "alarm": "alarm",
        "alarm clock": "alarm",
        "siren": "alarm",
        "buzzer": "alarm",
        "smoke detector": "alarm",
        "fire alarm": "alarm",
        "carbon monoxide detector": "alarm",
        "civil defense siren": "alarm",
        "air raid siren": "alarm",
        "ambulance (siren)": "alarm",
        "fire engine, fire truck (siren)": "alarm",
        "police car (siren)": "alarm",
        
        # Appliance and home device sounds
        "microwave oven": "microwave",
        "microwave": "microwave",
        "dishwasher": "appliance",
        "washing machine": "appliance",
        "kettle": "appliance", 
        "blender": "appliance",
        "coffee maker": "appliance",
        "toaster": "appliance",
        "refrigerator": "appliance",
        "air conditioning": "appliance",
        "vacuum cleaner": "appliance",
        
        # Phone related sounds
        "telephone": "phone",
        "telephone bell ringing": "phone",
        "ringtone": "phone",
        "cell phone": "phone",
        "telephone dialing": "phone",
        
        # Baby sounds
        "baby cry": "baby",
        "baby laughter": "baby",
        "infant cry": "baby",
        "children shouting": "baby",
        
        # Speech and person
        "speech": "speech",
        "male speech": "speech",
        "female speech": "speech",
        "child speech": "speech",
        "conversation": "speech",
        "narration": "speech",
        
        # Cat sounds
        "cat": "cat",
        "meow": "cat",
        "purr": "cat",
        "hiss": "cat",
        
        # Dog sounds
        "dog": "dog",
        "bark": "dog",
        "howl": "dog",
        "growling": "dog",
        
        # Special handling for finger snaps
        "finger snapping": "finger_snap",
    }
    
    # Store the mapped predictions
    mapped_predictions = []
    
    # Special handling for finger snapping - use a lower threshold
    finger_snap_detected = False
    finger_snap_confidence = 0.0
    
    # First pass to check for finger snapping with lower threshold
    for pred in predictions:
        label = pred["label"].lower()
        confidence = pred["confidence"]
        
        if "finger snapping" in label or "finger snap" in label:
            if confidence >= 0.03:  # Lower threshold for finger snaps
                finger_snap_detected = True
                finger_snap_confidence = confidence
                break
    
    # Map each prediction to a homesounds category if possible
    for pred in predictions:
        # Get original label and confidence
        original_label = pred["label"]
        label = original_label.lower()
        confidence = pred["confidence"]
        
        # Skip if below threshold (except for finger snapping)
        if confidence < threshold and "finger snap" not in label:
            continue
            
        # Try to map to a homesounds category
        mapped_label = None
        
        # Direct mapping
        for audio_label, home_label in mapping.items():
            if audio_label.lower() in label:
                mapped_label = home_label
                break
                
        # If no mapping was found, try to use a more general category
        if mapped_label is None:
            # Generic mappings based on partial matches
            if any(word in label for word in ["dog", "bark", "howl"]):
                mapped_label = "dog"
            elif any(word in label for word in ["cat", "meow", "purr"]):
                mapped_label = "cat"
            elif any(word in label for word in ["door", "knock", "bell"]):
                mapped_label = "door"
            elif any(word in label for word in ["water", "drip", "sink", "shower"]):
                mapped_label = "water"
            elif any(word in label for word in ["alarm", "siren", "alert"]):
                mapped_label = "alarm"
            elif any(word in label for word in ["speech", "voice", "talking", "conversation"]):
                mapped_label = "speech"
            elif any(word in label for word in ["phone", "telephone", "ringtone", "cell"]):
                mapped_label = "phone"
            elif any(word in label for word in ["baby", "cry", "infant"]):
                mapped_label = "baby"
            elif "finger snap" in label:
                mapped_label = "finger_snap"
        
        # Add to mapped predictions if a mapping was found
        if mapped_label:
            mapped_predictions.append({
                "original_label": original_label,
                "label": mapped_label,
                "confidence": confidence
            })
    
    # Special case: Add finger snap if detected with lower threshold but not already included
    if finger_snap_detected and not any(p["label"] == "finger_snap" for p in mapped_predictions):
        mapped_predictions.append({
            "original_label": "Finger snapping",
            "label": "finger_snap",
            "confidence": finger_snap_confidence
        })
    
    # Sort by confidence
    mapped_predictions.sort(key=lambda x: x["confidence"], reverse=True)
    
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
    Process audio input and return predictions using PANNs model
    
    Args:
        audio_data: Raw audio data as numpy array
        sample_rate: Sample rate of the audio data
        threshold: Minimum confidence threshold for predictions
        top_k: Number of top predictions to return
        
    Returns:
        Dictionary containing predictions
    """
    global PANNS_MODEL
    
    try:
        # Print shape info for debugging
        print(f"Audio data shape: {audio_data.shape}, sample rate: {sample_rate}")
        
        # Check if model is loaded
        if PANNS_MODEL is None:
            print("PANNs model not loaded, initializing...")
            if not initialize():
                return {"top_predictions": [{"label": "Error", "confidence": 0.0}], "mapped_predictions": []}
        
        # Convert to float32 if needed
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            
        # If audio is multichannel, convert to mono
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalize audio if not already normalized
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / 32768.0  # Normalize 16-bit PCM
            
        # Calculate RMS and dB level for silence detection
        rms = np.sqrt(np.mean(np.square(audio_data)))
        db_level = 20 * np.log10(rms) if rms > 0 else -100
        
        # Skip processing if audio is too quiet
        if db_level < -75:
            print(f"Audio too quiet: {db_level} dB")
            return {"top_predictions": [{"label": "Silence", "confidence": 0.95}], "mapped_predictions": []}
            
        # Make sure we're using exactly 32kHz sample rate for CNN14 (different from CNN14_16k)
        # Standard CNN14 model expects 32kHz audio
        if sample_rate != 32000:
            print(f"Resampling from {sample_rate}Hz to 32000Hz")
            # Use librosa for better quality resampling
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=32000)
            
        # Ensure audio length is sufficient - CNN14 needs at least 1 second at 32kHz
        min_audio_length = 32000  # 1 second at 32kHz
        if len(audio_data) < min_audio_length:
            # Pad the audio to minimum length
            padding = min_audio_length - len(audio_data)
            audio_data = np.pad(audio_data, (0, padding), 'constant')
            print(f"Padded audio from {len(audio_data) - padding} to {len(audio_data)} samples")
        
        # Make prediction with PANNs model
        try:
            # Run inference on the audio data
            print("Running PANNs inference...")
            
            # Save original data type for reference
            original_type = type(audio_data)
            print(f"Audio data type before inference: {original_type}")
            
            # Make a copy to avoid modifying the original data
            inference_audio = audio_data.copy()
            
            # In version 0.1.1, AudioTagging expects numpy array, not torch tensor
            if isinstance(inference_audio, torch.Tensor):
                inference_audio = inference_audio.cpu().numpy()
            
            # In panns-inference 0.1.1, AudioTagging.inference() accepts audio 
            # as a plain positional argument (no named parameters)
            clipwise_output = PANNS_MODEL.inference(inference_audio)
            
            # Convert from torch tensor to numpy if needed
            if isinstance(clipwise_output, torch.Tensor):
                clipwise_output = clipwise_output.cpu().numpy()
            
            # Convert predictions to list of dictionaries
            predictions = []
            
            # Get the class labels
            class_labels = PANNS_MODEL.labels()
            
            # Sort predictions by confidence
            sorted_indices = np.argsort(clipwise_output)[::-1]
            
            # Get top-k predictions above threshold
            for i in sorted_indices[:top_k]:
                confidence = float(clipwise_output[i])
                if confidence >= threshold:
                    predictions.append({
                        "label": class_labels[i],
                        "confidence": confidence
                    })
            
            # Print top predictions for debugging
            print("===== PANNS MODEL PREDICTIONS =====")
            if predictions:
                for pred in predictions[:5]:  # Print top 5 for debugging
                    print(f"  {pred['label']}: {pred['confidence']:.6f}")
            else:
                print("  No predictions above threshold")
            
            # Map PANNs labels to homesounds categories
            mapped_predictions = map_panns_labels_to_homesounds(predictions, threshold)
            
            # Return both raw and mapped predictions
            return {
                "top_predictions": predictions,
                "mapped_predictions": mapped_predictions,
                "raw_predictions": clipwise_output.tolist() if hasattr(clipwise_output, 'tolist') else clipwise_output
            }
            
        except Exception as e:
            print(f"Critical error in PANNs prediction: {str(e)}")
            traceback.print_exc()
            return {"top_predictions": [{"label": "Error", "confidence": 0.0}], "mapped_predictions": []}
        
    except Exception as e:
        print(f"Error in PANNs predict_sound: {str(e)}")
        traceback.print_exc()
        return {"top_predictions": [{"label": "Error", "confidence": 0.0}], "mapped_predictions": []}


def initialize():
    """
    Initialize the PANNs model for sound recognition
    
    Returns:
        bool: True if model was successfully loaded, False otherwise
    """
    global PANNS_MODEL, PANNS_MODEL_TYPE, MODEL_INITIALIZED
    
    if MODEL_INITIALIZED:
        print("PANNs model already initialized")
        return True
    
    try:
        print("Checking PANNs availability...")
        if not check_panns_availability():
            print("PANNs package not available - won't use PANNs model")
            return False
            
        print("Loading PANNs model: CNN14")
        
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Set model type - Always use CNN14 for v0.1.1
        PANNS_MODEL_TYPE = "CNN14"
        
        # Set up data directory - this is where PANNs will download the model checkpoint
        home_dir = os.path.expanduser("~")
        panns_data_dir = os.path.join(home_dir, "panns_data")
        os.makedirs(panns_data_dir, exist_ok=True)
        
        # In version 0.1.1, AudioTagging class loads CNN14 by default
        # No parameters are needed for initialization
        PANNS_MODEL = AudioTagging()
        
        # In version 0.1.1, AudioTagging uses cuda() method to move to GPU
        if device == 'cuda' and torch.cuda.is_available():
            print("Using CUDA for PANNs model")
            PANNS_MODEL.cuda()
        else:
            print("Using CPU for PANNs model")
        
        # Verify model is loaded by checking if labels() function works
        try:
            test_labels = PANNS_MODEL.labels()
            print(f"Model loaded with {len(test_labels)} label classes")
        except Exception as e:
            print(f"Warning: Couldn't verify model labels: {e}")
        
        print(f"PANNs model loaded successfully: {PANNS_MODEL_TYPE}")
        MODEL_INITIALIZED = True
        return True
        
    except Exception as e:
        print(f"Error initializing PANNs model: {e}")
        traceback.print_exc()
        PANNS_MODEL = None
        MODEL_INITIALIZED = False
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