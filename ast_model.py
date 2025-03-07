import os
import torch
import numpy as np
import soundfile as sf
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import warnings
import time

# Suppress transformers warnings
warnings.filterwarnings("ignore", category=UserWarning)

def load_ast_model(model_name="MIT/ast-finetuned-audioset-10-10-0.4593"):
    """
    Load the Audio Spectrogram Transformer model and feature extractor from Hugging Face
    
    Args:
        model_name (str): The name of the model on Hugging Face
        
    Returns:
        tuple: (model, feature_extractor)
    """
    print(f"Loading AST model: {model_name}")
    try:
        # Load feature extractor and model
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModelForAudioClassification.from_pretrained(model_name)
        
        # Put model in evaluation mode and move to CPU (or GPU if available)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model = model.to(device)
        
        return model, feature_extractor
    except Exception as e:
        print(f"Error loading AST model: {e}")
        raise e

# Map AST model labels to homesounds labels
# This mapping is based on semantic similarity between the labels from both systems
def map_ast_labels_to_homesounds():
    """
    Define mappings from AudioSet/AST labels to homesounds labels
    
    Returns:
        dict: Mapping from AST labels to homesounds labels
    """
    # Comprehensive mapping from AST/AudioSet labels to homesounds labels
    # These mappings are based on semantic similarity
    ast_to_homesounds = {
        # Test sounds (for development and testing)
        "Sine wave": "hazard-alarm",  # Map sine wave to hazard-alarm for testing
        "Beep, bleep": "hazard-alarm",
        "Chirp tone": "hazard-alarm",
        "Dial tone": "phone-ring",
        "Busy signal": "phone-ring",
        "White noise": "water-running",
        "Static": "water-running",
        "Noise": "water-running",
        
        # Dog sounds
        "Dog": "dog-bark",
        "Bark": "dog-bark",
        "Howl": "dog-bark",
        
        # Tool sounds
        "Drill": "drill",
        "Drilling": "drill",
        "Power tool": "drill",
        "Electric drill": "drill",
        "Hammer": "hammer",
        "Hammering": "hammer",
        "Saw": "saw",
        "Chainsaw": "saw",
        "Vacuum cleaner": "vacuum",
        "Vacuum": "vacuum",
        
        # Alarms
        "Smoke alarm": "hazard-alarm",
        "Fire alarm": "hazard-alarm",
        "Alarm": "hazard-alarm",
        "Carbon monoxide detector": "hazard-alarm",
        "Siren": "hazard-alarm",
        "Emergency vehicle": "hazard-alarm",
        "Smoke detector": "hazard-alarm",
        "Alarm clock": "alarm-clock",
        "Beep": "hazard-alarm",
        
        # Phones
        "Telephone": "phone-ring",
        "Telephone bell ringing": "phone-ring",
        "Ringtone": "phone-ring",
        "Ringer": "phone-ring",
        "Cell phone": "phone-ring",
        "Telephone ringing": "phone-ring",
        
        # Human sounds
        "Speech": "speech",
        "Male speech": "male-conversation",
        "Female speech": "female-conversation",
        "Conversation": "conversation",
        "Narration": "conversation",
        "Talk": "conversation",
        "Child speech": "conversation",
        "Crying, sobbing": "baby-cry",
        "Baby cry": "baby-cry",
        "Baby crying": "baby-cry",
        "Infant cry": "baby-cry",
        "Cough": "cough",
        "Coughing": "cough",
        "Snore": "snore",
        "Snoring": "snore",
        "Typing": "typing",
        "Keyboard": "typing",
        "Computer keyboard": "typing",
        "Typing on keyboard": "typing",
        "Keypress": "typing",
        "Laugh": "laugh",
        "Laughter": "laugh",
        "Chuckle": "laugh",
        "Giggle": "laugh",
        
        # Kitchen appliances and activities
        "Blender": "blender",
        "Food processor": "blender",
        "Mixer": "blender",
        "Microwave oven": "microwave",
        "Microwave": "microwave",
        "Chopping (food)": "chopping",
        "Knife": "chopping",
        "Cutting": "chopping",
        "Slicing": "chopping",
        "Dishwasher": "dishwasher",
        "Frying (food)": "cooking",
        "Sizzle": "cooking",
        "Frying pan": "cooking",
        "Cooking": "cooking",
        "Cutlery": "cooking",
        "Silverware": "cooking",
        "Food": "cooking",
        
        # Door sounds
        "Door": "door",
        "Slam": "door",
        "Knock": "knock",
        "Knocking": "knock",
        "Doorbell": "doorbell",
        "Ding-dong": "doorbell",
        "Bell": "doorbell",
        
        # Bathroom sounds
        "Water": "water-running",
        "Water tap": "water-running",
        "Running water": "water-running",
        "Sink (filling or washing)": "water-running",
        "Bathtub (filling or washing)": "water-running",
        "Faucet": "water-running",
        "Toilet flush": "flush",
        "Toilet": "flush",
        "Flush toilet": "flush",
        "Electric toothbrush": "toothbrush",
        "Toothbrush": "toothbrush",
        "Electric razor": "shaver",
        "Shaver": "shaver",
        "Electric shaver": "shaver",
        "Hair dryer": "hair-dryer",
        "Blow-dryer": "hair-dryer",
        
        # Vehicle sounds
        "Car": "engine",
        "Vehicle": "engine",
        "Engine": "engine",
        "Motor vehicle (road)": "engine",
        "Engine starting": "engine",
        "Car passing by": "engine",
        "Traffic noise": "engine",
        "Road noise": "engine",
        "Car horn": "car-horn",
        "Honk": "car-horn",
        "Beeping": "car-horn",
        
        # Animal sounds
        "Cat": "cat-meow",
        "Meow": "cat-meow",
        "Cat meowing": "cat-meow",
        
        # NEW: Hand sounds - special focus on finger snapping
        "Finger snapping": "finger-snap",
        "Snap": "finger-snap",
        "Clap": "hand-clap",
        "Clapping": "hand-clap",
        "Hands": "hand-sounds",
        "Slap": "hand-sounds",
        "Tap": "hand-sounds",
        "Percussion": "hand-sounds",
        "Click": "hand-sounds",
        "Applause": "applause",
        
        # NEW: Ambient and silence related
        "Silence": "silence",
        "Quiet": "silence",
        "Room tone": "silence",
        "Background noise": "background",
        "Ambient noise": "background",
        "Inside, small room": "background",
        "Inside, large room": "background",
        "Rustling": "background",
        
        # NEW: Music and instruments
        "Music": "music",
        "Musical instrument": "music",
        "Singing": "music",
        "Guitar": "music",
        "Piano": "music",
        "Drum": "music",
        "Percussion": "music",
        "Wind instrument": "music",
        "String instrument": "music",
        "Plucked string instrument": "music",
        "Brass instrument": "music",
        "Synthesizer": "music",
        "Sound effect": "sound-effect",
        
        # NEW: Electronic sounds
        "Electronic sound": "electronic-sound",
        "Computer": "electronic-sound",
        "Electronic device": "electronic-sound",
        "Notification": "notification",
        "Alert": "notification",
        "Chime": "notification",
    }
    
    return ast_to_homesounds

def predict_sound(audio_data, sample_rate, model, feature_extractor, threshold=0.05, top_k=5):
    """
    Process audio input and return predictions
    
    Args:
        audio_data (np.ndarray): Audio data as a numpy array
        sample_rate (int): Sample rate of the audio data
        model: The AST model
        feature_extractor: The feature extractor for the AST model
        threshold (float): Confidence threshold for predictions
        top_k (int): Number of top predictions to return
    
    Returns:
        dict: Dictionary with raw and mapped predictions
    """
    try:
        start_time = time.time()
        
        # Ensure the audio data is the right shape and type
        if len(audio_data.shape) > 1:
            # If multi-channel, convert to mono by averaging
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalize audio data
        if np.abs(audio_data).max() > 1.0:
            audio_data = audio_data / np.abs(audio_data).max()
        
        # Calculate RMS and dB for silence detection
        rms = np.sqrt(np.mean(audio_data**2))
        db_level = 20 * np.log10(rms) if rms > 0 else -100
        
        print(f"Audio data shape: {audio_data.shape}, sample rate: {sample_rate}")
        
        # Check for silence first (very low volume)
        if db_level < -65:
            print(f"Silence detected (db level: {db_level:.2f})")
            return {
                "top_predictions": [{"label": "Silence", "confidence": 0.95}],
                "mapped_predictions": [{"original_label": "Silence", "label": "silence", "confidence": 0.95}]
            }
        
        # Extract features
        inputs = feature_extractor(
            audio_data, 
            sampling_rate=sample_rate, 
            return_tensors="pt",
            padding=True
        )
        
        # Move inputs to the same device as the model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        print(f"Feature extraction completed. Running inference...")
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get the model's id2label mapping
            id2label = model.config.id2label
            
            # Convert to numpy for easier processing
            probs_np = probs[0].cpu().numpy()
            
            # Get indices of top k predictions
            top_indices = np.argsort(probs_np)[::-1][:top_k]
            
            # Create list of top predictions
            top_predictions = []
            for idx in top_indices:
                label = id2label[idx]
                confidence = float(probs_np[idx])
                top_predictions.append({
                    "label": label,
                    "confidence": confidence
                })
            
            # Map AST labels to homesounds labels
            ast_to_homesounds = map_ast_labels_to_homesounds()
            mapped_predictions = []
            
            # Special handling for finger snapping with lower threshold
            finger_snap_prediction = None
            finger_snap_confidence = 0
            
            for pred in top_predictions:
                ast_label = pred["label"]
                confidence = pred["confidence"]
                
                # Special case for finger snapping
                if ast_label == "Finger snapping" or ast_label == "Snap":
                    if confidence > finger_snap_confidence:
                        finger_snap_prediction = {
                            "original_label": ast_label,
                            "label": "finger-snap",
                            "confidence": confidence
                        }
                        finger_snap_confidence = confidence
                
                # Try to map the AST label to a homesounds label
                if ast_label in ast_to_homesounds:
                    homesounds_label = ast_to_homesounds[ast_label]
                    mapped_predictions.append({
                        "original_label": ast_label,
                        "label": homesounds_label,
                        "confidence": confidence
                    })
            
            # Add finger snapping if detected, even with lower confidence
            if finger_snap_prediction and finger_snap_confidence > 0.03:  # Lower threshold just for finger snapping
                # If already in mapped_predictions, remove it (to avoid duplicates)
                mapped_predictions = [p for p in mapped_predictions if p["label"] != "finger-snap"]
                mapped_predictions.append(finger_snap_prediction)
            
            # Filter by threshold (except for the special case handled above)
            mapped_predictions = [p for p in mapped_predictions if p["confidence"] >= threshold]
            
            # Sort by confidence
            mapped_predictions.sort(key=lambda x: x["confidence"], reverse=True)
            
            elapsed_time = time.time() - start_time
            print(f"Prediction completed in {elapsed_time:.2f} seconds")
            
            return {
                "top_predictions": top_predictions,
                "mapped_predictions": mapped_predictions
            }
    
    except Exception as e:
        print(f"Error in predict_sound: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "top_predictions": [],
            "mapped_predictions": []
        }

# Testing
if __name__ == "__main__":
    # Generate a test sound (1 second of 440Hz sine wave)
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Load the model
    model, feature_extractor = load_ast_model()
    
    # Run prediction
    predictions = predict_sound(audio_data, sample_rate, model, feature_extractor)
    
    # Print results
    print("\nTop predictions from AST model:")
    for pred in predictions["top_predictions"]:
        print(f"  {pred['label']}: {pred['confidence']:.6f}")
    
    print("\nMapped predictions for SoundWatch:")
    for pred in predictions["mapped_predictions"]:
        print(f"  {pred['label']} (from {pred['original_label']}): {pred['confidence']:.6f}") 