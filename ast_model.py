import os
import torch
import numpy as np
import soundfile as sf
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import warnings
import time
import traceback

# Suppress transformers warnings
warnings.filterwarnings("ignore", category=UserWarning)

def check_float16_support():
    """Check if the system supports float16 operations in a backward-compatible way."""
    try:
        # For server stability, we'll be conservative and recommend not using float16
        # unless we're 100% certain it's compatible with the system
        if hasattr(torch.backends, 'cpu') and hasattr(torch.backends.cpu, 'supports_float16'):
            return False  # Returning False for now as we've seen compatibility issues
        return False
    except Exception as e:
        print(f"Error checking float16 support: {e}")
        return False

def load_ast_model(model_name="MIT/ast-finetuned-audioset-10-10-0.4593", **kwargs):
    """
    Load the Audio Spectrogram Transformer model and feature extractor from Hugging Face
    
    Args:
        model_name (str): The name of the model on Hugging Face
        **kwargs: Additional arguments to pass to the from_pretrained method
            - attn_implementation: "eager" or "sdpa" for Scaled Dot Product Attention
            - torch_dtype: torch.float32, torch.float16 or torch.bfloat16
        
    Returns:
        tuple: (model, feature_extractor)
    """
    print(f"Loading AST model: {model_name}")
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        attn_implementation = kwargs.get("attn_implementation", None)
        print("Using standard precision (float32) for maximum compatibility")
        if "torch_dtype" in kwargs:
            del kwargs["torch_dtype"]
        if attn_implementation:
            print(f"Using attention implementation: {attn_implementation}")
        model = AutoModelForAudioClassification.from_pretrained(model_name, torch_dtype=torch.float32, **kwargs)
        for param in model.parameters():
            if param.dtype != torch.float32:
                param.data = param.data.to(torch.float32)
        model = model.to(torch.float32)
        print("Verified all model parameters are using float32 precision")
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model = model.to(device)
        return model, feature_extractor
    except Exception as e:
        print(f"Error loading AST model: {e}")
        raise e

def map_ast_labels_to_homesounds(predictions, threshold=0.05):
    """
    Map AST model predictions to homesounds labels
    
    Args:
        predictions: List of prediction dictionaries from AST model
        threshold: Confidence threshold for predictions (default: 0.05)
    
    Returns:
        list: Mapped predictions with homesounds labels
    """
    ast_to_homesounds = {
        "Sine wave": "hazard-alarm", "Beep, bleep": "hazard-alarm", "Chirp tone": "hazard-alarm",
        "Dial tone": "phone-ring", "Busy signal": "phone-ring", "White noise": "water-running",
        "Static": "water-running", "Noise": "water-running", "Dog": "dog-bark", "Bark": "dog-bark",
        "Howl": "dog-bark", "Drill": "drill", "Drilling": "drill", "Power tool": "drill",
        "Electric drill": "drill", "Hammer": "hammer", "Hammering": "hammer", "Saw": "saw",
        "Chainsaw": "saw", "Vacuum cleaner": "vacuum", "Vacuum": "vacuum", "Smoke alarm": "hazard-alarm",
        "Fire alarm": "hazard-alarm", "Alarm": "hazard-alarm", "Carbon monoxide detector": "hazard-alarm",
        "Siren": "hazard-alarm", "Emergency vehicle": "hazard-alarm", "Smoke detector": "hazard-alarm",
        "Alarm clock": "alarm-clock", "Beep": "hazard-alarm", "Telephone": "phone-ring",
        "Telephone bell ringing": "phone-ring", "Ringtone": "phone-ring", "Ringer": "phone-ring",
        "Cell phone": "phone-ring", "Telephone ringing": "phone-ring", "Speech": "speech",
        "Male speech": "male-conversation", "Female speech": "female-conversation", "Conversation": "conversation",
        "Narration": "conversation", "Talk": "conversation", "Child speech": "conversation",
        "Crying, sobbing": "baby-cry", "Baby cry": "baby-cry", "Baby crying": "baby-cry",
        "Infant cry": "baby-cry", "Cough": "cough", "Coughing": "cough", "Snore": "snore",
        "Snoring": "snore", "Typing": "typing", "Keyboard": "typing", "Computer keyboard": "typing",
        "Typing on keyboard": "typing", "Keypress": "typing", "Laugh": "laugh", "Laughter": "laugh",
        "Chuckle": "laugh", "Giggle": "laugh", "Blender": "blender", "Food processor": "blender",
        "Mixer": "blender", "Microwave oven": "microwave", "Microwave": "microwave",
        "Chopping (food)": "chopping", "Knife": "chopping", "Cutting": "chopping", "Slicing": "chopping",
        "Dishwasher": "dishwasher", "Frying (food)": "cooking", "Sizzle": "cooking", "Frying pan": "cooking",
        "Cooking": "cooking", "Cutlery": "cooking", "Silverware": "cooking", "Food": "cooking",
        "Door": "door", "Slam": "door", "Knock": "knock", "Knocking": "knock", "Doorbell": "doorbell",
        "Ding-dong": "doorbell", "Bell": "doorbell", "Water": "water-running", "Water tap": "water-running",
        "Running water": "water-running", "Sink (filling or washing)": "water-running",
        "Bathtub (filling or washing)": "water-running", "Faucet": "water-running", "Toilet flush": "flush",
        "Toilet": "flush", "Flush toilet": "flush", "Electric toothbrush": "toothbrush", "Toothbrush": "toothbrush",
        "Electric razor": "shaver", "Shaver": "shaver", "Electric shaver": "shaver", "Hair dryer": "hair-dryer",
        "Blow-dryer": "hair-dryer", "Car": "engine", "Vehicle": "engine", "Engine": "engine",
        "Motor vehicle (road)": "engine", "Engine starting": "engine", "Car passing by": "engine",
        "Traffic noise": "engine", "Road noise": "engine", "Car horn": "car-horn", "Honk": "car-horn",
        "Beeping": "car-horn", "Cat": "cat-meow", "Meow": "cat-meow", "Cat meowing": "cat-meow",
        "Finger snapping": "finger-snap", "Snap": "finger-snap", "Clap": "hand-clap", "Clapping": "hand-clap",
        "Hands": "hand-sounds", "Slap": "hand-sounds", "Tap": "hand-sounds", "Percussion": "hand-sounds",
        "Click": "hand-sounds", "Applause": "applause", "Silence": "silence", "Quiet": "silence",
        "Room tone": "silence", "Background noise": "background", "Ambient noise": "background",
        "Inside, small room": "background", "Inside, large room": "background", "Rustling": "background",
        "Music": "music", "Musical instrument": "music", "Singing": "music", "Guitar": "music",
        "Piano": "music", "Drum": "music", "Wind instrument": "music", "String instrument": "music",
        "Plucked string instrument": "music", "Brass instrument": "music", "Synthesizer": "music",
        "Sound effect": "sound-effect", "Electronic sound": "electronic-sound", "Computer": "electronic-sound",
        "Electronic device": "electronic-sound", "Notification": "notification", "Alert": "notification",
        "Chime": "notification",
    }
    mapped_predictions = []
    finger_snap_prediction = None
    finger_snap_confidence = 0
    for pred in predictions:
        ast_label = pred["label"]
        confidence = pred["confidence"]
        if ast_label in ("Finger snapping", "Snap"):
            if confidence > finger_snap_confidence:
                finger_snap_prediction = {"original_label": ast_label, "label": "finger-snap", "confidence": confidence}
                finger_snap_confidence = confidence
        if ast_label in ast_to_homesounds:
            homesound_label = ast_to_homesounds[ast_label]
            mapped_predictions.append({"original_label": ast_label, "label": homesound_label, "confidence": confidence})
    if finger_snap_prediction and finger_snap_confidence > 0.03:
        mapped_predictions = [p for p in mapped_predictions if p["label"] != "finger-snap"]
        mapped_predictions.append(finger_snap_prediction)
    mapped_predictions = [p for p in mapped_predictions if p["confidence"] >= threshold or p["label"] == "finger-snap"]
    return sorted(mapped_predictions, key=lambda x: x["confidence"], reverse=True)

def preprocess_audio_for_ast(audio_data, sample_rate, feature_extractor):
    """Preprocess audio data specifically for AST model"""
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    if np.abs(audio_data).max() > 1.0:
        audio_data = audio_data / np.abs(audio_data).max()
    audio_data = audio_data.astype(np.float32)
    inputs = feature_extractor(audio_data, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    return {k: v.to(torch.float32) for k, v in inputs.items()}

def predict_sound(audio_data, sample_rate, model, feature_extractor, threshold=0.05, top_k=5):
    """Process audio input and return predictions"""
    try:
        start_time = time.time()
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        if np.abs(audio_data).max() > 1.0:
            audio_data = audio_data / np.abs(audio_data).max()
        audio_data = audio_data.astype(np.float32)
        rms = np.sqrt(np.mean(audio_data**2))
        db_level = 20 * np.log10(rms) if rms > 0 else -100
        print(f"Audio data shape: {audio_data.shape}, sample rate: {sample_rate}")
        if db_level < -65:
            print(f"Silence detected (db level: {db_level:.2f})")
            return {
                "top_predictions": [{"label": "Silence", "confidence": 0.95}],
                "mapped_predictions": [{"original_label": "Silence", "label": "silence", "confidence": 0.95}],
                "raw_predictions": None
            }
        param_dtypes = {param.dtype: 0 for name, param in model.named_parameters()}
        for param in model.parameters():
            param_dtypes[param.dtype] += 1
        print(f"Model is using dtype: {next(model.parameters()).dtype}")
        inputs = preprocess_audio_for_ast(audio_data, sample_rate, feature_extractor)
        device = next(model.parameters()).device
        for key in inputs:
            if inputs[key].dtype != torch.float32:
                inputs[key] = inputs[key].to(dtype=torch.float32)
            if inputs[key].device != device:
                inputs[key] = inputs[key].to(device=device)
        print("Feature extraction completed. Running inference...")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.float()
            probs = torch.nn.functional.softmax(logits, dim=-1)
            id2label = model.config.id2label
            probs_np = probs[0].cpu().numpy()
            raw_predictions = probs_np
            predictions = process_predictions(probs_np, id2label, threshold, top_k)
            mapped_predictions = map_ast_labels_to_homesounds(predictions["top_predictions"], threshold)
            predictions["raw_predictions"] = raw_predictions
            predictions["mapped_predictions"] = mapped_predictions
            end_time = time.time()
            print(f"AST prediction completed in {end_time - start_time:.2f} seconds")
            return predictions
    except Exception as e:
        print(f"Critical error in AST prediction: {str(e)}")
        traceback.print_exc()
        return {
            "top_predictions": [{"label": "Error", "confidence": 0.0}],
            "mapped_predictions": [],
            "raw_predictions": None
        }

def process_predictions(probs_np, id2label, threshold=0.05, top_k=5):
    """Process raw model predictions to get top-k predictions with labels"""
    top_indices = np.argsort(probs_np)[::-1][:top_k]
    top_probs = probs_np[top_indices]
    top_labels = [id2label[idx] for idx in top_indices]
    top_predictions = [
        {"label": top_labels[i], "confidence": float(top_probs[i])}
        for i in range(len(top_indices))
        if top_probs[i] >= threshold
    ]
    print("===== AST MODEL RAW PREDICTIONS =====")
    for i, pred in enumerate(top_predictions[:5]):
        print(f"  {pred['label']}: {pred['confidence']:.6f}")
    mapped_predictions = map_ast_labels_to_homesounds(top_predictions, threshold)
    return {
        "top_predictions": top_predictions,
        "mapped_predictions": mapped_predictions,
        "has_predictions": len(top_predictions) > 0
    }

class_labels = []

def initialize_class_labels(model):
    """Initialize the class labels from the model"""
    global class_labels
    if model and hasattr(model, 'config') and hasattr(model.config, 'id2label'):
        class_labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    return class_labels

if __name__ == "__main__":
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
    model, feature_extractor = load_ast_model()
    predictions = predict_sound(audio_data, sample_rate, model, feature_extractor)
    print("\nTop predictions from AST model:")
    for pred in predictions["top_predictions"]:
        print(f"  {pred['label']}: {pred['confidence']:.6f}")
    print("\nMapped predictions for SoundWatch:")
    for pred in predictions["mapped_predictions"]:
        print(f"  {pred['label']} (from {pred['original_label']}): {pred['confidence']:.6f}")