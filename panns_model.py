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
MODEL_LOADED = False  # Flag to check if the model is loaded

# Choose from different PANNs variants
# CNN10 - Lighter, faster model with good performance
# CNN14 - Larger model with better accuracy
# CNN14_16k - 16kHz variant compatible with SoundWatch's 16kHz audio
DEFAULT_MODEL_TYPE = "CNN14"  # The default model in 0.1.1


def check_panns_availability():
    """
    Check if PANNs and its dependencies are available
    
    Returns:
        bool: True if PANNs is available, False otherwise
    """
    try:
        # Check for required packages
        import torch
        import librosa
        import panns_inference
        
        # Try to import AudioTagging specifically
        from panns_inference import AudioTagging
        
        # All imports succeeded, PANNs is available
        print("PANNs and all dependencies are available")
        return True
        
    except ImportError as e:
        print(f"PANNs dependencies missing: {e}")
        print("Please install required packages:")
        print("  pip install torch panns_inference librosa")
        return False
    except Exception as e:
        print(f"Error checking PANNs availability: {e}")
        return False


def map_panns_labels_to_homesounds(predictions: List[Dict], threshold: float = 0.01) -> List[Dict]:
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
        "doorbell": "door",
        "slam": "door",
        "ding-dong": "door",
        "sliding door": "door",
        "cupboard open or close": "door",
        "drawer open or close": "door",
        "door knock": "door",
        
        # Water related sounds
        "water": "water",
        "splash": "water",
        "water tap": "water",
        "sink (filling or washing)": "water",
        "bathtub (filling or washing)": "water",
        "flowing water": "water",
        "drip": "water",
        "pour": "water",
        "trickle, dribble": "water",
        "gush": "water",
        "fill (with liquid)": "water",
        "spray": "water",
        "pump (liquid)": "water",
        "boiling": "water",
        "toilet flush": "water",
        "gurgling": "water",
        "water running": "water",
        
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
        "bell": "alarm",
        "church bell": "alarm",
        "jingle bell": "alarm",
        "bicycle bell": "alarm",
        "telephone bell ringing": "alarm",
        "emergency vehicle": "alarm",
        "beep, bleep": "alarm",
        "chime": "alarm",
        
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
        "mechanical fan": "appliance",
        "hair dryer": "appliance",
        "domestic sounds, home sounds": "appliance",
        "frying (food)": "appliance",
        "chopping (food)": "appliance",
        "kitchen utensil": "appliance",
        
        # Phone related sounds
        "telephone": "phone",
        "telephone bell ringing": "phone",
        "ringtone": "phone",
        "cell phone": "phone",
        "telephone dialing": "phone",
        "dial tone": "phone",
        "busy signal": "phone",
        "cell phone vibrating": "phone",
        "text message notification": "phone",
        "phone notification": "phone",
        
        # Baby sounds
        "baby cry": "baby",
        "baby laughter": "baby",
        "infant cry": "baby",
        "children shouting": "baby",
        "baby cry, infant cry": "baby",
        "whimper": "baby",
        "crying, sobbing": "baby",
        "baby babbling": "baby",
        "child speech": "baby",
        
        # Speech and person
        "speech": "speech",
        "male speech": "speech",
        "female speech": "speech",
        "child speech": "speech",
        "conversation": "speech",
        "narration": "speech",
        "speech synthesizer": "speech",
        "male speech, man speaking": "speech",
        "female speech, woman speaking": "speech",
        "child speech, kid speaking": "speech",
        "narration, monologue": "speech",
        "human voice": "speech",
        "shout": "speech",
        "scream": "speech",
        "whispering": "speech",
        "laughter": "speech",
        
        # Cat sounds
        "cat": "cat",
        "meow": "cat",
        "purr": "cat",
        "hiss": "cat",
        "domestic animals, pets": "cat",
        "caterwaul": "cat",
        "cat communication": "cat",
        
        # Dog sounds
        "dog": "dog",
        "bark": "dog",
        "howl": "dog",
        "growling": "dog",
        "bow-wow": "dog",
        "yip": "dog",
        "whimper (dog)": "dog",
        "domestic animals, pets": "dog",
        "dog whimpering": "dog",
        
        # Special handling for finger snaps
        "finger snapping": "finger_snap",
        "finger snap": "finger_snap",
        "clapping": "finger_snap",
        "hands": "finger_snap",
        
        # Music categorization (often confused with other sounds)
        "music": "music",
        "musical instrument": "music",
        "singing": "music",
        "drum": "music",
        "guitar": "music",
        "piano": "music",
        "violin, fiddle": "music",
        "percussion": "music",
        "keyboard (musical)": "music",
        
        # Vehicle sounds (often confused with appliances)
        "vehicle": "vehicle",
        "engine": "vehicle",
        "car": "vehicle",
        "traffic noise, roadway noise": "vehicle",
        "road": "vehicle",
        "car passing by": "vehicle",
        "bus": "vehicle",
        "truck": "vehicle",
        "motorcycle": "vehicle",
    }
    
    # For fuzzy matching - used for words that appear in labels
    # This improves matching for partial or variant names
    fuzzy_mapping = {
        # Door related terms
        "door": "door",
        "knock": "door",
        "slam": "door",
        "ding-dong": "door",
        "cupboard": "door",
        "drawer": "door",
        
        # Water related terms
        "water": "water",
        "splash": "water",
        "sink": "water",
        "bath": "water",
        "flow": "water",
        "drip": "water",
        "toilet": "water",
        "liquid": "water",
        "gush": "water",
        "pour": "water",
        
        # Alarm related terms
        "alarm": "alarm",
        "siren": "alarm",
        "buzzer": "alarm",
        "detector": "alarm",
        "bell": "alarm",
        "emergency": "alarm",
        "alert": "alarm",
        
        # Phone related terms
        "phone": "phone",
        "telephone": "phone",
        "ringtone": "phone",
        "dial": "phone",
        "vibrat": "phone",  # Capture vibrating, vibration
        
        # Baby sound terms
        "baby": "baby",
        "infant": "baby",
        "child": "baby",
        "cry": "baby",
        
        # Speech related terms
        "speech": "speech",
        "voice": "speech",
        "talk": "speech",
        "speak": "speech",
        "conversation": "speech",
        "human": "speech",
        "vocal": "speech",
        
        # Animal sounds
        "cat": "cat",
        "meow": "cat",
        "purr": "cat",
        "feline": "cat",
        "dog": "dog",
        "bark": "dog",
        "howl": "dog",
        "canine": "dog",
        
        # Finger snap
        "finger": "finger_snap",
        "snap": "finger_snap",
        "clap": "finger_snap",
        
        # Music terms
        "music": "music",
        "musical": "music",
        "instrument": "music",
        "song": "music",
        "singing": "music",
        
        # Vehicle terms
        "vehicle": "vehicle",
        "car": "vehicle",
        "traffic": "vehicle",
        "engine": "vehicle",
        "motor": "vehicle",
        
        # Appliance terms
        "appliance": "appliance",
        "machine": "appliance",
        "vacuum": "appliance",
        "microwave": "microwave",
        "kitchen": "appliance",
        "blender": "appliance",
        "fan": "appliance",
        "domestic": "appliance",
    }
    
    # Print a status message for the mapping process
    print(f"Mapping {len(predictions)} predictions to homesounds categories (threshold: {threshold})")
    
    # Store the mapped predictions
    mapped_predictions = []
    
    # Special handling for music - often falsely detected in ambient noise
    # Only map music if it has very high confidence (0.3+)
    music_confidence_threshold = 0.3
    
    # Map each prediction to a homesounds category if possible
    for pred in predictions:
        # Get original label and confidence
        original_label = pred["label"]
        label_lower = original_label.lower()
        confidence = pred["confidence"]
        
        # Skip if below threshold (except for special cases)
        if confidence < threshold:
            continue
            
        # Special handling for music (often false positive)
        if "music" in label_lower and confidence < music_confidence_threshold:
            print(f"  Skipping '{original_label}' with confidence {confidence:.4f} (below music threshold {music_confidence_threshold})")
            continue
            
        # Try to map to a homesounds category
        mapped_label = None
        
        # Direct mapping
        for audio_label, home_label in mapping.items():
            if audio_label.lower() == label_lower:  # Exact match
                mapped_label = home_label
                print(f"  Exact match: '{original_label}' -> '{home_label}' (confidence: {confidence:.4f})")
                break
        
        # If no exact match, try substring matching
        if mapped_label is None:
            for audio_label, home_label in mapping.items():
                if audio_label.lower() in label_lower or label_lower in audio_label.lower():
                    mapped_label = home_label
                    print(f"  Substring match: '{original_label}' contains or is contained in '{audio_label}' -> '{home_label}' (confidence: {confidence:.4f})")
                    break
                
        # If no mapping was found, try fuzzy matching
        if mapped_label is None:
            for keyword, home_label in fuzzy_mapping.items():
                if keyword in label_lower:
                    mapped_label = home_label
                    print(f"  Fuzzy matching: '{original_label}' contains '{keyword}' -> '{home_label}' (confidence: {confidence:.4f})")
                    break
        
        # Add to mapped predictions if a mapping was found
        if mapped_label:
            mapped_predictions.append({
                "original_label": original_label,
                "label": mapped_label,
                "confidence": confidence
            })
    
    # Special handling for finger snapping - look for it with lower threshold
    has_finger_snap = False
    for pred in predictions:
        label_lower = pred["label"].lower()
        confidence = pred["confidence"]
        
        # Add finger snap with a lower threshold (0.008)
        if ("finger" in label_lower or "snap" in label_lower or "clap" in label_lower) and confidence >= 0.008:
            has_finger_snap = True
            # Check if it's already in mapped_predictions
            if not any(p["label"] == "finger_snap" for p in mapped_predictions):
                mapped_predictions.append({
                    "original_label": pred["label"],
                    "label": "finger_snap",
                    "confidence": confidence
                })
                print(f"  Special case: Added 'finger_snap' from '{pred['label']}' with confidence {confidence:.4f}")
                break
    
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
        
        # Set up data directory
        home_dir = os.path.expanduser("~")
        panns_data_dir = os.path.join(home_dir, "panns_data")
        os.makedirs(panns_data_dir, exist_ok=True)
        
        # Get the CNN14 model checkpoint path
        checkpoint_path = os.path.join(panns_data_dir, "Cnn14_mAP=0.431.pth")
        print(f"Checkpoint path: {checkpoint_path}")
        
        # Check if the model file exists, download if it doesn't
        if not os.path.exists(checkpoint_path):
            print("Downloading PANNs model checkpoint...")
            model_url = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
            wget.download(model_url, checkpoint_path)
            print(f"\nDownloaded checkpoint to {checkpoint_path}")
        
        # Initialize model with the proper arguments for v0.1.1
        at = AudioTagging(checkpoint_path=checkpoint_path, device=device)
        PANNS_MODEL = at
        PANNS_MODEL_TYPE = model_type
        
        # Test model with a silent audio sample
        silent_audio = np.zeros((1, 32000), dtype=np.float32)  # 1 second of silence with batch dimension
        with torch.no_grad():
            _, _ = at.inference(silent_audio)
        
        print(f"PANNs model loaded successfully: {model_type}")
        return at, True
        
    except Exception as e:
        print(f"Error loading PANNs model: {e}")
        traceback.print_exc()
        return None, False


def predict_sound(audio_data, sample_rate, threshold=0.05, top_k=10):
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
            
        # Make sure we're using exactly 32kHz sample rate for CNN14
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
            
            # Add batch dimension as needed by the model (N, T) - exactly like the example
            if len(audio_data.shape) == 1:
                audio_data = audio_data[np.newaxis, :]  # Add batch dimension: (samples,) -> (1, samples)
            
            # Call inference - returns (clipwise_output, embedding)
            clipwise_output, _ = PANNS_MODEL.inference(audio_data)
            
            # Handle output shape correctly
            if isinstance(clipwise_output, torch.Tensor):
                # If it's a torch tensor, convert to numpy
                clipwise_output = clipwise_output.squeeze(0).detach().cpu().numpy()
            elif isinstance(clipwise_output, np.ndarray) and clipwise_output.shape[0] == 1:
                # If it's a numpy array with a batch dimension
                clipwise_output = clipwise_output.squeeze(0)
            
            # Print shape for debugging
            print(f"Clipwise output shape: {clipwise_output.shape}")
                
            # Print min/max values to verify normalization
            print(f"Output range: min={np.min(clipwise_output):.6f}, max={np.max(clipwise_output):.6f}")
            
            # Convert predictions to list of dictionaries
            predictions = []
            
            # Get the class labels directly from the model
            class_labels = PANNS_MODEL.labels
            
            # Sort predictions by confidence
            sorted_indices = np.argsort(clipwise_output)[::-1]
            
            # Get top-k predictions above threshold
            # Look at top 50 regardless of threshold for diagnostic purposes
            top_50_indices = sorted_indices[:50]
            print(f"Top 5 predictions (before threshold filtering):")
            for i in top_50_indices[:5]:
                confidence = float(clipwise_output[i])
                print(f"  {class_labels[i]}: {confidence:.6f}")
            
            # Now filter by threshold for actual predictions
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
            
            # If there are mapped predictions, print them
            print("===== MAPPED PREDICTIONS =====")
            if mapped_predictions:
                for pred in mapped_predictions[:5]:
                    print(f"  {pred['label']} (from {pred['original_label']}): {pred['confidence']:.6f}")
            else:
                print("  No mapped predictions")
            
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
        print(f"Error in predict_sound: {str(e)}")
        traceback.print_exc()
        return {"top_predictions": [{"label": "Error", "confidence": 0.0}], "mapped_predictions": []}


def initialize():
    """
    Initialize PANNs model and resources
    
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    global PANNS_MODEL, MODEL_LOADED
    
    try:
        # Check if model is already loaded
        if MODEL_LOADED:
            print("PANNs model already loaded")
            return True
        
        # Make sure we've got the required packages
        if not check_panns_availability():
            print("PANNs dependencies not available")
            return False
        
        print("Initializing PANNs model...")
        # Import here to avoid loading PyTorch until needed
        from panns_inference import AudioTagging
        
        # Determine device (CPU or CUDA)
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Load the AudioTagging model exactly like the example
        # This automatically handles downloading the checkpoint if needed
        print("Loading PANNs model... this may take a moment.")
        PANNS_MODEL = AudioTagging(checkpoint_path=None, device=device)
        
        # Verify the model loaded successfully
        if PANNS_MODEL is not None:
            MODEL_LOADED = True
            print("PANNs model loaded successfully")
            
            # Print model information
            model_type = PANNS_MODEL.__class__.__name__
            print(f"Model type: {model_type}")
            
            # Print number of available labels
            try:
                num_classes = len(PANNS_MODEL.labels)
                print(f"Number of classes: {num_classes}")
                print(f"Sample classes: {PANNS_MODEL.labels[:5]} ...")
            except Exception as e:
                print(f"Could not get labels: {e}")
                
            return True
        else:
            print("Failed to load PANNs model")
            return False
            
    except Exception as e:
        print(f"Error initializing PANNs model: {e}")
        import traceback
        traceback.print_exc()
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