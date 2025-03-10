from threading import Lock
from flask import Flask, render_template, session, request, \
    copy_current_request_context, Response, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room, \
    close_room, rooms, disconnect
import tensorflow as tf
from tensorflow import keras
import numpy as np
from vggish_input import waveform_to_examples
import homesounds
from pathlib import Path
import time
import argparse
import wget
import traceback
from helpers import dbFS
import os
import socket
import torch
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import base64
from scipy import signal
import threading
import queue
from transformers import pipeline
import logging

# Import our AST model implementation
import ast_model

# Import our PANNs model implementation
import panns_model

# Import our sentiment analysis modules
from sentiment_analyzer import analyze_sentiment
from speech_to_text import transcribe_audio, SpeechToText
from google_speech import transcribe_with_google, GoogleSpeechToText

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Memory optimization settings
MEMORY_OPTIMIZATION_LEVEL = os.environ.get('MEMORY_OPTIMIZATION', '1')  # 0=None, 1=Moderate, 2=Aggressive
if MEMORY_OPTIMIZATION_LEVEL == '1':
    logger.info("Using moderate memory optimization")
    torch.set_num_threads(4)  # Limit threads for CPU usage
    EMPTY_CACHE_FREQ = 10  # Empty cache every 10 predictions
elif MEMORY_OPTIMIZATION_LEVEL == '2':
    logger.info("Using aggressive memory optimization")
    torch.set_num_threads(2)  # More strict thread limiting
    EMPTY_CACHE_FREQ = 5  # Empty cache every 5 predictions
else:
    logger.info("No memory optimization")
    EMPTY_CACHE_FREQ = 0  # Never automatically empty cache

# Global counter for memory cleanup
prediction_counter = 0

# Function to clean up memory periodically
def cleanup_memory():
    """Clean up unused memory to prevent memory leaks"""
    global prediction_counter
    
    prediction_counter += 1
    
    if EMPTY_CACHE_FREQ == 0:
        return
    
    if prediction_counter % EMPTY_CACHE_FREQ == 0:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            import gc
            gc.collect()
        logger.info(f"Memory cleanup performed (cycle {prediction_counter})")

# Speech recognition settings
USE_GOOGLE_SPEECH = False  # Set to True to use Google Cloud Speech-to-Text instead of Whisper

# Add the current directory to the path so we can import our modules
os.path.dirname(os.path.abspath(__file__))

# Helper function to get the computer's IP addresses
def get_ip_addresses():
    ip_list = []
    try:
        hostname = socket.gethostname()
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('10.255.255.255', 1))
            primary_ip = s.getsockname()[0]
            ip_list.append(primary_ip)
        except Exception:
            pass
        finally:
            s.close()
            
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if ip not in ip_list and not ip.startswith('127.'):
                ip_list.append(ip)
    except Exception as e:
        print(f"Error getting IP addresses: {e}")
    
    return ip_list

# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.
async_mode = None

# Configure TensorFlow to use compatibility mode with TF 1.x code
tf.compat.v1.disable_eager_execution()

# Create a TensorFlow lock for thread safety, and another lock for the AST model
tf_lock = Lock()
tf_graph = tf.Graph()
tf_session = tf.compat.v1.Session(graph=tf_graph)
ast_lock = Lock()
speech_lock = Lock()  # Lock for speech processing

# Prediction aggregation system - store recent predictions to improve accuracy
MAX_PREDICTIONS_HISTORY = 2  # Reduced from 3 to 2 for general sounds
SPEECH_PREDICTIONS_HISTORY = 4  # Keep more prediction history for speech
recent_predictions = []  # Store recent prediction probabilities for each sound category
speech_predictions = []  # Separate storage for speech predictions
prediction_lock = Lock()  # Lock for thread-safe access to prediction history

app = Flask(__name__)
app.config['SECRET_KEY'] = 'soundwatch_secret!'
socketio = SocketIO(app, async_mode=async_mode)
thread = None
thread_lock = Lock()

# contexts
context = homesounds.everything
active_context = homesounds.everything

# thresholds
PREDICTION_THRES = 0.15  # Increased from 0.05 to reduce false positives 
FINGER_SNAP_THRES = 0.03  # Special lower threshold for finger snapping
DBLEVEL_THRES = -60  # Adjusted from -65 to -60 to filter out more background noise
SILENCE_THRES = -75  # Threshold for silence detection
SPEECH_SENTIMENT_THRES = 0.35  # Increased from 0.12 to 0.35 to reduce false positives
CHOPPING_THRES = 0.70  # Higher threshold for chopping sounds to prevent false positives
SPEECH_PREDICTION_THRES = 0.70  # Higher threshold for speech to reduce false positives
SPEECH_DETECTION_THRES = 0.30  # Lower threshold just for detecting potential speech (0.3 = 30%)

CHANNELS = 1
RATE = 16000
CHUNK = RATE  # 1 second chunks
SPEECH_CHUNK_MULTIPLIER = 4.0  # Increased from 2.5 to 4.0 for better speech recognition
MICROPHONES_DESCRIPTION = []
FPS = 60.0

# Minimum word length for meaningful transcription
MIN_TRANSCRIPTION_LENGTH = 3  # Minimum characters in transcription to analyze
MIN_MEANINGFUL_WORDS = 2  # Minimum number of meaningful words (not just "you" or "thank you")
COMMON_FALSE_POSITIVES = ["you", "the", "thank you", "thanks", "a", "to", "and", "is", "it", "that"]

# Load sentiment analysis model
try:
    sentiment_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    logger.info("Sentiment analysis model loaded successfully")
except Exception as e:
    logger.error(f"Error loading sentiment model: {str(e)}")
    sentiment_pipeline = None

# Dictionary to map emotion to emojis
EMOTION_TO_EMOJI = {
    "joy": "😄",
    "neutral": "😀",
    "surprise": "😮",
    "sadness": "😢",
    "fear": "😨",
    "anger": "😠",
    "disgust": "🤢"
}

# Grouped emotions for simplified categories
EMOTION_GROUPS = {
    "Happy": ["joy", "love", "admiration", "approval", "caring", "excitement", "amusement", "gratitude", "optimism", "pride", "relief"],
    "Neutral": ["neutral", "realization", "curiosity"],
    "Surprised": ["surprise", "confusion", "nervousness"],
    "Unpleasant": ["sadness", "fear", "anger", "disgust", "disappointment", "embarrassment", "grief", "remorse", "annoyance", "disapproval"]
}

# Dictionary to store our models
models = {
    "tensorflow": None,
    "ast": None,
    "feature_extractor": None,
    "panns": None
}

# Initialize speech recognition systems
speech_processor = SpeechToText()
google_speech_processor = None  # Will be lazy-loaded when needed

# Load models
def load_models():
    """Load all required models for sound recognition and speech processing."""
    global models, USE_AST_MODEL, USE_PANNS_MODEL
    
    models = {
        "tensorflow": None,
        "ast": None,
        "feature_extractor": None,
        "sentiment_analyzer": None,
        "speech_processor": None,
        "panns": None
    }
    
    USE_AST_MODEL = os.environ.get('USE_AST_MODEL', '1') == '1'  # Default to enabled
    USE_PANNS_MODEL = os.environ.get('USE_PANNS_MODEL', '0') == '1'  # Default to disabled
    print(f"AST model {'enabled' if USE_AST_MODEL else 'disabled'} based on environment settings")
    print(f"PANNs model {'enabled' if USE_PANNS_MODEL else 'disabled'} based on environment settings")
    
    MODEL_URL = "https://www.dropbox.com/s/cq1d7uqg0l28211/example_model.hdf5?dl=1"
    MODEL_PATH = "models/example_model.hdf5"
    
    try:
        print("Loading AST model...")
        with ast_lock:
            ast_kwargs = {
                "torch_dtype": torch.float32
            }
            if torch.__version__ >= '2.1.1':
                ast_kwargs["attn_implementation"] = "sdpa"
                print("Using Scaled Dot Product Attention (SDPA) for faster inference")
            print("Using standard precision (float32) for maximum compatibility")
            model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
            models["ast"], models["feature_extractor"] = ast_model.load_ast_model(
                model_name=model_name,
                **ast_kwargs
            )
            ast_model.initialize_class_labels(models["ast"])
            print("AST model loaded successfully")
    except Exception as e:
        print(f"Error loading AST model: {e}")
        traceback.print_exc()
        USE_AST_MODEL = False

    model_filename = os.path.abspath(MODEL_PATH)
    if not USE_AST_MODEL or True:
        print("Loading TensorFlow model as backup...")
        os.makedirs(os.path.dirname(model_filename), exist_ok=True)
        
        homesounds_model = Path(model_filename)
        if not homesounds_model.is_file():
            print("Downloading example_model.hdf5 [867MB]: ")
            wget.download(MODEL_URL, MODEL_PATH)
        
        print("Using TensorFlow model: %s" % (model_filename))
        
        try:
            with tf_graph.as_default():
                with tf_session.as_default():
                    models["tensorflow"] = keras.models.load_model(model_filename)
                    print("Initializing TensorFlow model with a dummy prediction...")
                    dummy_input = np.zeros((1, 96, 64, 1))
                    _ = models["tensorflow"].predict(dummy_input)
                    print("Model prediction function initialized successfully")
            print("TensorFlow model loaded successfully")
            if not USE_AST_MODEL:
                print("TensorFlow model will be used as primary model")
                models["tensorflow"].summary()
        except Exception as e:
            print(f"Error loading TensorFlow model with standard method: {e}")
            try:
                with tf_graph.as_default():
                    with tf_session.as_default():
                        models["tensorflow"] = tf.keras.models.load_model(model_filename, compile=False)
                        print("Initializing TensorFlow model with a dummy prediction...")
                        dummy_input = np.zeros((1, 96, 64, 1))
                        _ = models["tensorflow"].predict(dummy_input)
                        print("Model prediction function initialized successfully")
                print("TensorFlow model loaded with compile=False option")
            except Exception as e2:
                print(f"Error with fallback method: {e2}")
                if not USE_AST_MODEL:
                    raise Exception("Could not load TensorFlow model with any method, and AST model is not enabled")

    print(f"Using {'AST' if USE_AST_MODEL else 'TensorFlow'} model as primary model")

    try:
        if not USE_GOOGLE_SPEECH:
            print("Loading Whisper model for speech recognition...")
            speech_processor = SpeechToText()
            print("Whisper model loaded successfully")
            models["speech_processor"] = speech_processor
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        traceback.print_exc()

    if USE_PANNS_MODEL:
        try:
            print("Loading PANNs model...")
            panns_loaded = panns_model.load_panns_model()
            if panns_loaded:
                models["panns"] = True
                print("PANNs model loaded successfully")
            else:
                print("Failed to load PANNs model")
                USE_PANNS_MODEL = False
        except Exception as e:
            print(f"Error loading PANNs model: {e}")
            traceback.print_exc()
            USE_PANNS_MODEL = False

    if not USE_AST_MODEL and not USE_PANNS_MODEL:
        print("Both AST and PANNs models are disabled, using TensorFlow model as primary model")

# Add a comprehensive debug function
def debug_predictions(predictions, label_list):
    print("===== DEBUGGING ALL PREDICTIONS (BEFORE THRESHOLD) =====")
    try:
        # Convert predictions to numpy array if needed
        if not isinstance(predictions, np.ndarray):
            try:
                predictions = np.array(predictions)
            except:
                print(f"Could not convert predictions to numpy array. Type: {type(predictions)}")
        
        # Handle different label_list types
        if label_list is None:
            # Handle case where label_list is None
            for idx, pred in enumerate(predictions):
                print(f"Prediction {idx}: {pred:.6f}")
        elif isinstance(label_list, dict):
            # Handle dictionary labels (e.g., {label: index})
            # For dictionaries, try to find the matching label for each index
            for idx, pred in enumerate(predictions):
                label_found = False
                for label, label_idx in label_list.items():
                    if label_idx == idx:
                        print(f"{label}: {pred:.6f}")
                        label_found = True
                        break
                if not label_found:
                    print(f"Unknown({idx}): {pred:.6f}")
        elif isinstance(label_list, (list, tuple, np.ndarray)):
            # Handle list, tuple, or array of labels
            if len(label_list) == 0:
                # Empty list case
                for idx, pred in enumerate(predictions):
                    print(f"Prediction {idx}: {pred:.6f}")
            else:
                # Normal case with labels
                for idx, pred in enumerate(predictions):
                    if idx < len(label_list):
                        try:
                            print(f"{label_list[idx]}: {pred:.6f}")
                        except Exception as e:
                            print(f"Prediction {idx}: {pred:.6f} (Error accessing label: {e})")
                    else:
                        print(f"Unknown({idx}): {pred:.6f}")
        else:
            # Fallback for other label_list types
            print(f"Unexpected label_list type: {type(label_list)}")
            for idx, pred in enumerate(predictions):
                print(f"Prediction {idx}: {pred:.6f}")
        
        print("=======================================================")
    except Exception as e:
        print(f"Error in debug_predictions: {e}")
        traceback.print_exc()

MODEL_URL = "https://www.dropbox.com/s/cq1d7uqg0l28211/example_model.hdf5?dl=1"
MODEL_PATH = "models/example_model.hdf5"
print("=====")
print("Setting up sound recognition models...")

USE_AST_MODEL = os.environ.get('USE_AST_MODEL', '1') == '1'
USE_PANNS_MODEL = os.environ.get('USE_PANNS_MODEL', '0') == '1'
print(f"AST model {'enabled' if USE_AST_MODEL else 'disabled'} based on environment settings")
print(f"PANNs model {'enabled' if USE_PANNS_MODEL else 'disabled'} based on environment settings")

try:
    print("Loading AST model...")
    ast_model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    ast_kwargs = {}
    if torch.__version__ >= '2.1.1':
        ast_kwargs["attn_implementation"] = "sdpa"
        print("Using Scaled Dot Product Attention (SDPA) for faster inference")
    ast_kwargs = {"torch_dtype": torch.float32}
    with ast_lock:
        models["ast"], models["feature_extractor"] = ast_model.load_ast_model(
            model_name=ast_model_name,
            **ast_kwargs
        )
        ast_model.initialize_class_labels(models["ast"])
    print("AST model loaded successfully")
except Exception as e:
    print(f"Error loading AST model: {e}")
    traceback.print_exc()
    USE_AST_MODEL = False

model_filename = os.path.abspath(MODEL_PATH)
if not USE_AST_MODEL or True:
    print("Loading TensorFlow model as backup...")
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    
    homesounds_model = Path(model_filename)
    if not homesounds_model.is_file():
        print("Downloading example_model.hdf5 [867MB]: ")
        wget.download(MODEL_URL, MODEL_PATH)
    
    print("Using TensorFlow model: %s" % (model_filename))
    
    try:
        with tf_graph.as_default():
            with tf_session.as_default():
                models["tensorflow"] = keras.models.load_model(model_filename)
                print("Initializing TensorFlow model with a dummy prediction...")
                dummy_input = np.zeros((1, 96, 64, 1))
                _ = models["tensorflow"].predict(dummy_input)
                print("Model prediction function initialized successfully")
        print("TensorFlow model loaded successfully")
        if not USE_AST_MODEL:
            print("TensorFlow model will be used as primary model")
            models["tensorflow"].summary()
    except Exception as e:
        print(f"Error loading TensorFlow model with standard method: {e}")
        try:
            with tf_graph.as_default():
                with tf_session.as_default():
                    models["tensorflow"] = tf.keras.models.load_model(model_filename, compile=False)
                    print("Initializing TensorFlow model with a dummy prediction...")
                    dummy_input = np.zeros((1, 96, 64, 1))
                    _ = models["tensorflow"].predict(dummy_input)
                    print("Model prediction function initialized successfully")
            print("TensorFlow model loaded with compile=False option")
        except Exception as e2:
            print(f"Error with fallback method: {e2}")
            if not USE_AST_MODEL:
                raise Exception("Could not load TensorFlow model with any method, and AST model is not enabled")

print(f"Using {'AST' if USE_AST_MODEL else 'TensorFlow'} model as primary model")

def audio_samples(in_data, frame_count, time_info, status_flags):
    np_wav = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
    rms = np.sqrt(np.mean(np_wav**2))
    db = dbFS(rms)

    x = waveform_to_examples(np_wav, RATE)
    predictions = []
    
    if x.shape[0] != 0:
        x = x.reshape(len(x), 96, 64, 1)
        print('Reshape x successful', x.shape)
        pred = models["tensorflow"].predict(x)
        predictions.append(pred)
    
    print('Prediction succeeded')
    for prediction in predictions:
        context_prediction = np.take(
            prediction[0], [homesounds.labels[x] for x in active_context])
        m = np.argmax(context_prediction)
        if (context_prediction[m] > PREDICTION_THRES and db > DBLEVEL_THRES):
            print("Prediction: %s (%0.2f)" % (
                homesounds.to_human_labels[active_context[m]], context_prediction[m]))

    print("Raw audio min/max:", np.min(np_wav), np.max(np_wav))
    print("Processed audio shape:", x.shape)

    return (in_data, 0)

@socketio.on('audio_feature_data')
def handle_source(json_data):
    """Handle audio features sent from client."""
    try:
        data = json_data.get('data', [])
        db = json_data.get('db')
        record_time = json_data.get('record_time', None)
        
        if -db < SILENCE_THRES:
            socketio.emit('audio_label', {
                'label': 'Silence',
                'accuracy': '0.95',
                'db': str(db)
            })
            print(f"EMITTING: Silence (db: {db})")
            return
        
        if -db > DBLEVEL_THRES:
            if USE_AST_MODEL:
                with ast_lock:
                    np_data = np.array(data, dtype=np.float32)
                    print(f"Processing with AST model, data shape: {np_data.shape}")
                    ast_predictions = ast_model.predict_sound(
                        np_data, 
                        RATE, 
                        models["ast"], 
                        models["feature_extractor"], 
                        threshold=PREDICTION_THRES
                    )
                    print("===== AST MODEL RAW PREDICTIONS =====")
                    for pred in ast_predictions["top_predictions"][:5]:
                        print(f"  {pred['label']}: {pred['confidence']:.6f}")
                    
                    raw_predictions = ast_predictions["raw_predictions"]
                    is_speech_prediction = False
                    for pred in ast_predictions["top_predictions"]:
                        if pred["label"].lower() == "speech" and pred["confidence"] > SPEECH_DETECTION_THRES:
                            is_speech_prediction = True
                            logger.info(f"AST detected potential speech with confidence: {pred['confidence']:.4f}")
                            break
                    
                    if raw_predictions is not None and len(raw_predictions) > 0:
                        aggregated_predictions = aggregate_predictions(
                            raw_predictions, 
                            ast_model.class_labels,
                            is_speech=is_speech_prediction
                        )
                        ast_predictions = ast_model.process_predictions(
                            aggregated_predictions, 
                            ast_model.class_labels,
                            threshold=PREDICTION_THRES
                        )
                        print("===== AGGREGATED AST MODEL PREDICTIONS =====")
                        for pred in ast_predictions["top_predictions"][:5]:
                            print(f"  {pred['label']}: {pred['confidence']:.6f}")
                    
                    finger_snap_detected = False
                    for pred in ast_predictions["top_predictions"]:
                        if (pred["label"] == "Finger snapping" or pred["label"] == "Snap") and pred["confidence"] > FINGER_SNAP_THRES:
                            finger_snap_detected = True
                            print(f"Finger snap detected with confidence: {pred['confidence']:.6f}")
                            socketio.emit('audio_label', {
                                'label': 'Finger Snap',
                                'accuracy': str(pred["confidence"]),
                                'db': str(db)
                            })
                            print(f"EMITTING: Finger Snap ({pred['confidence']:.6f})")
                            cleanup_memory()
                            return
                    
                    mapped_preds = ast_predictions["mapped_predictions"]
                    
                    if mapped_preds:
                        top_pred = mapped_preds[0]
                        top_label = top_pred["label"]
                        top_confidence = top_pred["confidence"]
                        
                        for pred in mapped_preds:
                            if pred["label"] == "finger-snap" and pred["confidence"] > FINGER_SNAP_THRES:
                                top_label = pred["label"]
                                top_confidence = pred["confidence"]
                                break
                        
                        human_label = homesounds.to_human_labels.get(top_label, top_label)
                        if top_label == "finger-snap" and human_label == "finger-snap":
                            human_label = "Finger Snap"
                        
                        print(f"Top AST prediction: {human_label} ({top_confidence:.6f})")
                        
                        if human_label == "Speech" and top_confidence > SPEECH_SENTIMENT_THRES:
                            print("Speech detected. Processing sentiment...")
                            sentiment_result = process_speech_with_sentiment(np_data)
                            
                            if sentiment_result:
                                if isinstance(sentiment_result, dict) and 'sentiment' in sentiment_result and isinstance(sentiment_result['sentiment'], dict) and 'category' in sentiment_result['sentiment']:
                                    label = f"Speech {sentiment_result['sentiment']['category']}"
                                    socketio.emit('audio_label', {
                                        'label': label,
                                        'accuracy': str(sentiment_result['sentiment']['confidence']),
                                        'db': str(db),
                                        'emoji': sentiment_result['sentiment']['emoji'],
                                        'transcription': sentiment_result['text'],
                                        'emotion': sentiment_result['sentiment']['original_emotion'],
                                        'sentiment_score': str(sentiment_result['sentiment']['confidence'])
                                    })
                                    print(f"EMITTING SPEECH WITH SENTIMENT: {label} with emoji {sentiment_result['sentiment']['emoji']}")
                                else:
                                    label = "Speech"
                                    if isinstance(sentiment_result, dict):
                                        transcription = sentiment_result.get('transcription', sentiment_result.get('text', ''))
                                        sentiment_value = sentiment_result.get('sentiment', 'neutral')
                                        confidence = sentiment_result.get('confidence', 0.5)
                                        
                                        if isinstance(sentiment_value, dict):
                                            category = sentiment_value.get('category', 'Neutral')
                                            emoji = sentiment_value.get('emoji', '😐')
                                            emotion = sentiment_value.get('original_emotion', 'neutral')
                                            confidence = sentiment_value.get('confidence', confidence)
                                        else:
                                            category = 'Neutral' if isinstance(sentiment_value, str) else 'Neutral'
                                            emoji = '😐'
                                            emotion = 'neutral'
                                        
                                        label = f"Speech {category}"
                                        socketio.emit('audio_label', {
                                            'label': label,
                                            'accuracy': str(confidence),
                                            'db': str(db),
                                            'emoji': emoji,
                                            'transcription': transcription,
                                            'emotion': emotion,
                                            'sentiment_score': str(confidence)
                                        })
                                        print(f"EMITTING SPEECH WITH BASIC SENTIMENT: {label}")
                                    else:
                                        socketio.emit('audio_label', {
                                            'label': 'Speech',
                                            'accuracy': '0.6',
                                            'db': str(db)
                                        })
                                        print("EMITTING BASIC SPEECH DETECTION (no sentiment)")
                                    cleanup_memory()
                                    return
                        
                        if top_confidence > PREDICTION_THRES or (top_label == "finger-snap" and top_confidence > FINGER_SNAP_THRES):
                            socketio.emit('audio_label', {
                                'label': human_label,
                                'accuracy': str(top_confidence),
                                'db': str(db)
                            })
                            print(f"EMITTING AST PREDICTION: {human_label} ({top_confidence:.6f})")
                        else:
                            print(f"Top AST prediction {human_label} ({top_confidence:.6f}) below threshold")
                            if not finger_snap_detected:
                                socketio.emit('audio_label', {
                                    'label': 'Unrecognized Sound',
                                    'accuracy': '0.2',
                                    'db': str(db)
                                })
                    else:
                        print("No mapped predictions found from AST model")
                        if not finger_snap_detected:
                            socketio.emit('audio_label', {
                                'label': 'Unrecognized Sound',
                                'accuracy': '0.2',
                                'db': str(db)
                            })
            else:
                with tf_lock:
                    np_data = np.array(data, dtype=np.float32)
                    np_data = np.reshape(np_data, (1, 96, 64, 1))
                    print(f"Successfully convert to NP rep {np_data[0][0][0]}")
                    print(f"Db... {-db}")
                    print("Making prediction with TensorFlow model...")
                    with tf_graph.as_default():
                        with tf_session.as_default():
                            predictions = models["tensorflow"].predict(np_data)
                    
                    if np.ndim(predictions) > 0 and len(predictions) > 0:
                        debug_predictions(predictions[0], homesounds.labels)
                        is_speech_prediction = False
                        speech_idx = -1
                        for idx, label in enumerate(homesounds.labels):
                            if label.lower() == "speech":
                                speech_idx = idx
                                break
                        if speech_idx >= 0 and speech_idx < len(predictions[0]):
                            if predictions[0][speech_idx] > SPEECH_DETECTION_THRES:
                                is_speech_prediction = True
                                logger.info(f"Detected potential speech with confidence: {predictions[0][speech_idx]:.4f}")
                        
                        aggregated_predictions = aggregate_predictions(
                            predictions[0], 
                            homesounds.labels,
                            is_speech=is_speech_prediction
                        )
                        logger.info("Aggregated predictions:")
                        debug_predictions(aggregated_predictions, homesounds.labels)
                        
                        pred_max = -1
                        pred_max_val = 0
                        pred_label = None
                        for l in active_context:
                            i = homesounds.labels.get(l, -1)
                            if i >= 0 and i < len(aggregated_predictions) and aggregated_predictions[i] > pred_max_val:
                                pred_max = i
                                pred_max_val = aggregated_predictions[i]
                        
                        if pred_max != -1 and pred_max_val > PREDICTION_THRES:
                            for label, index in homesounds.labels.items():
                                if index == pred_max:
                                    human_label = homesounds.to_human_labels.get(label, label)
                                    print(f"Top prediction: {human_label} ({pred_max_val:.4f})")
                                    if human_label == "Chopping" and pred_max_val < CHOPPING_THRES:
                                        print(f"Ignoring Chopping sound with confidence {pred_max_val:.4f} < {CHOPPING_THRES} threshold")
                                        socketio.emit('audio_label', {
                                            'label': 'Unrecognized Sound',
                                            'accuracy': '0.2',
                                            'db': str(db)
                                        })
                                        cleanup_memory()
                                        return
                                    if human_label == "Speech" and pred_max_val > SPEECH_SENTIMENT_THRES:
                                        print("Speech detected with TensorFlow model. Processing sentiment...")
                                        sentiment_result = process_speech_with_sentiment(np_data)
                                        if sentiment_result:
                                            if isinstance(sentiment_result, dict) and 'sentiment' in sentiment_result and isinstance(sentiment_result['sentiment'], dict) and 'category' in sentiment_result['sentiment']:
                                                label = f"Speech {sentiment_result['sentiment']['category']}"
                                                socketio.emit('audio_label', {
                                                    'label': label,
                                                    'accuracy': str(sentiment_result['sentiment']['confidence']),
                                                    'db': str(db),
                                                    'emoji': sentiment_result['sentiment']['emoji'],
                                                    'transcription': sentiment_result['text'],
                                                    'emotion': sentiment_result['sentiment']['original_emotion'],
                                                    'sentiment_score': str(sentiment_result['sentiment']['confidence'])
                                                })
                                                print(f"EMITTING SPEECH WITH SENTIMENT: {label} with emoji {sentiment_result['sentiment']['emoji']}")
                                            else:
                                                label = "Speech"
                                                if isinstance(sentiment_result, dict):
                                                    transcription = sentiment_result.get('transcription', sentiment_result.get('text', ''))
                                                    sentiment_value = sentiment_result.get('sentiment', 'neutral')
                                                    confidence = sentiment_result.get('confidence', 0.5)
                                                    if isinstance(sentiment_value, dict):
                                                        category = sentiment_value.get('category', 'Neutral')
                                                        emoji = sentiment_value.get('emoji', '😐')
                                                        emotion = sentiment_value.get('original_emotion', 'neutral')
                                                        confidence = sentiment_value.get('confidence', confidence)
                                                    else:
                                                        category = 'Neutral' if isinstance(sentiment_value, str) else 'Neutral'
                                                        emoji = '😐'
                                                        emotion = 'neutral'
                                                    label = f"Speech {category}"
                                                    socketio.emit('audio_label', {
                                                        'label': label,
                                                        'accuracy': str(confidence),
                                                        'db': str(db),
                                                        'emoji': emoji,
                                                        'transcription': transcription,
                                                        'emotion': emotion,
                                                        'sentiment_score': str(confidence)
                                                    })
                                                    print(f"EMITTING SPEECH WITH BASIC SENTIMENT: {label}")
                                                else:
                                                    socketio.emit('audio_label', {
                                                        'label': 'Speech',
                                                        'accuracy': '0.6',
                                                        'db': str(db)
                                                    })
                                                    print("EMITTING BASIC SPEECH DETECTION (no sentiment)")
                                                cleanup_memory()
                                            return
                                    socketio.emit('audio_label', {
                                        'label': human_label,
                                        'accuracy': str(pred_max_val),
                                        'db': str(db)
                                    })
                                    cleanup_memory()
                                    break
                        else:
                            print(f"No prediction above threshold: {pred_max_val:.4f}")
                            socketio.emit('audio_label', {
                                'label': 'Unrecognized Sound',
                                'accuracy': '0.2',
                                'db': str(db)
                            })
                    else:
                        print("Invalid prediction format")
                        socketio.emit('audio_label', {
                            'label': 'Error Processing',
                            'accuracy': '0.0',
                            'db': str(db)
                        })
        else:
            print(f"Sound level ({-db} dB) below threshold ({DBLEVEL_THRES} dB)")
            socketio.emit('audio_label', {
                'label': 'Too Quiet',
                'accuracy': '0.9',
                'db': str(db)
            })
    except Exception as e:
        print(f"Error in handle_source: {str(e)}")
        traceback.print_exc()
        socketio.emit('audio_label', {
            'label': 'Error Processing Audio',
            'accuracy': '0.0',
            'db': str(db) if 'db' in locals() else '-100'
        })

@socketio.on('audio_data')
def handle_audio(data):
    try:
        record_time = None
        if isinstance(data, dict):
            if 'data' in data:
                audio_values = data['data']
                values = np.array(audio_values, dtype=np.int32)
                if 'record_time' in data:
                    record_time = data['record_time']
                    print(f"Received record_time: {record_time}")
            else:
                raise ValueError("Missing 'data' field in the received JSON object")
        else:
            values = np.array([int(x) for x in data.split(',') if x.strip()])
        np_wav = values.astype(np.float32) / 32768.0
    except Exception as e:
        print(f"Error processing audio data: {str(e)}")
        try:
            if isinstance(data, dict):
                print(f"Data keys: {data.keys()}")
                if 'data' in data:
                    print(f"Data array length: {len(data['data'])}")
                    if len(data['data']) > 0:
                        print(f"First few values: {data['data'][:10]}")
            else:
                print(f"Data type: {type(data)}")
                if isinstance(data, str):
                    print(f"Data sample: {data[:100]}...")
        except Exception as debug_err:
            print(f"Error in debug print: {str(debug_err)}")
        socketio.emit('audio_label', {
            'label': 'Error Processing Audio',
            'accuracy': '0.0',
            'db': '-100'
        })
        return
    
    print('Successfully convert to NP rep', np_wav)
    rms = np.sqrt(np.mean(np_wav**2))
    db = dbFS(rms)
    print('Db...', db)
    
    if -db < SILENCE_THRES:
        socketio.emit('audio_label', {
            'label': 'Silence',
            'accuracy': '0.95',
            'db': str(db)
        })
        print(f"EMITTING: Silence (db: {db})")
        return
    
    if -db > DBLEVEL_THRES:
        # Determine which model to use based on settings
        if USE_AST_MODEL:
            try:
                with ast_lock:
                    print("Processing with AST model (from audio_data)...")
                    ast_predictions = ast_model.predict_sound(
                        np_wav, 
                        RATE, 
                        models["ast"], 
                        models["feature_extractor"], 
                        threshold=PREDICTION_THRES
                    )
                    print("===== AST MODEL RAW PREDICTIONS =====")
                    for pred in ast_predictions["top_predictions"][:5]:
                        print(f"  {pred['label']}: {pred['confidence']:.6f}")
                    
                    raw_predictions = ast_predictions["raw_predictions"]
                    is_speech_prediction = False
                    for pred in ast_predictions["top_predictions"]:
                        if pred["label"].lower() == "speech" and pred["confidence"] > SPEECH_DETECTION_THRES:
                            is_speech_prediction = True
                            logger.info(f"AST detected potential speech with confidence: {pred['confidence']:.4f}")
                            break
                    
                    if raw_predictions is not None and len(raw_predictions) > 0:
                        aggregated_predictions = aggregate_predictions(
                            raw_predictions, 
                            ast_model.class_labels,
                            is_speech=is_speech_prediction
                        )
                        debug_predictions(aggregated_predictions, ast_model.class_labels)
                        
                        # Find top prediction and emit result
                        top_idx = np.argmax(aggregated_predictions)
                        if top_idx < len(ast_model.class_labels):
                            top_label = ast_model.class_labels[top_idx]
                            top_score = float(aggregated_predictions[top_idx])
                            
                            if top_score > PREDICTION_THRES:
                                print(f"Top prediction: {top_label} ({top_score:.4f})")
                                
                                # Handle speech detection
                                if top_label.lower() == "speech" and top_score > SPEECH_DETECTION_THRES:
                                    print(f"Speech detected with AST model. Processing sentiment...")
                                    if os.environ.get('USE_SPEECH', '0') == '1' and os.environ.get('USE_SENTIMENT', '0') == '1':
                                        # Process speech
                                        process_speech(np_wav, record_time, top_score)
                                    else:
                                        # Normal sound emission
                                        socketio.emit('audio_label', {
                                            'label': top_label,
                                            'accuracy': str(top_score),
                                            'db': str(db),
                                            'timestamp': record_time
                                        })
                                else:
                                    # Normal sound emission
                                    socketio.emit('audio_label', {
                                        'label': top_label,
                                        'accuracy': str(top_score),
                                        'db': str(db),
                                        'timestamp': record_time
                                    })
                                print(f"EMITTING: {top_label} ({top_score:.2f})")
                            else:
                                print(f"Top prediction {top_label} ({top_score:.4f}) below threshold, not emitting")
                        else:
                            print(f"Invalid index {top_idx} for label list with {len(homesounds.labels)} elements")
                            socketio.emit('audio_label', {
                                'label': 'Unrecognized Sound',
                                'accuracy': '0.2',
                                'db': str(db),
                                'timestamp': record_time
                            })
                    else:
                        print("No valid predictions from AST model")
            except Exception as e:
                print(f"Error with AST prediction: {e}")
                traceback.print_exc()
                print("Falling back to next available model due to AST error")
                
                # Choose next available model based on settings
                if USE_PANNS_MODEL:
                    process_with_panns_model(np_wav, record_time, db)
                else:
                    predict_with_tensorflow(np_wav, record_time, db)
                    
        elif USE_PANNS_MODEL:
            process_with_panns_model(np_wav, record_time, db)
        else:
            predict_with_tensorflow(np_wav, record_time, db)
    else:
        print(f"Sound level ({-db} dB) below threshold ({DBLEVEL_THRES} dB)")
        socketio.emit('audio_label', {
            'label': 'Too Quiet',
            'accuracy': '0.9',
            'db': str(db)
        })

# Helper function to process with PANNs model with fallback to TensorFlow
def process_with_panns_model(np_wav, record_time=None, db=None):
    """Process audio using PANNs model with fallback to TensorFlow"""
    try:
        # Use PANNs model for prediction
        print("Processing with PANNs model...")
        print(f"Audio shape: {np_wav.shape}, min: {np_wav.min():.6f}, max: {np_wav.max():.6f}, mean: {np_wav.mean():.6f}")
        
        # Ensure audio is normalized properly
        if np.abs(np_wav).max() > 1.0:
            print(f"Warning: Audio data exceeds normalized range [-1.0, 1.0]. Max value: {np.abs(np_wav).max():.6f}")
            # Normalize if needed
            np_wav = np_wav / np.abs(np_wav).max()
            print(f"Audio normalized. New range: [{np_wav.min():.6f}, {np_wav.max():.6f}]")
            
        # Ensure audio is long enough for processing
        if len(np_wav) < 8000:  # At least 0.5 seconds at 16kHz
            print(f"Audio too short ({len(np_wav)} samples). Padding to 16000 samples.")
            padded = np.zeros(16000)
            padded[:len(np_wav)] = np_wav
            np_wav = padded
        
        # Start processing with original audio
        panns_results = panns_model.predict_with_panns(
            np_wav, 
            top_k=10, 
            threshold=PREDICTION_THRES,
            map_to_homesounds_format=True
        )
        
        # Check if we got any results
        if panns_results and "output" in panns_results and len(panns_results["output"]) > 0:
            # Display predictions
            print("===== PANNs MODEL PREDICTIONS =====")
            for pred in panns_results["output"][:5]:
                print(f"  {pred['label']}: {pred['score']:.6f}")
            
            # Get top prediction
            top_prediction = panns_results["output"][0]
            top_label = top_prediction["label"]
            top_score = float(top_prediction["score"])
            
            print(f"Top prediction: {top_label} ({top_score:.4f})")
            
            # Handle speech detection
            if top_label.lower() == "speech" and top_score > SPEECH_DETECTION_THRES:
                print(f"Speech detected with PANNs model. Processing sentiment...")
                if os.environ.get('USE_SPEECH', '0') == '1' and os.environ.get('USE_SENTIMENT', '0') == '1':
                    # Process speech
                    process_speech(np_wav, record_time, top_score)
                else:
                    # Normal sound emission
                    socketio.emit('audio_label', {
                        'label': top_label,
                        'accuracy': str(top_score),
                        'db': str(db)
                    })
                    print(f"EMITTING: {top_label} ({top_score:.2f})")
            else:
                # Emit the top prediction
                if top_score > PREDICTION_THRES:
                    socketio.emit('audio_label', {
                        'label': top_label,
                        'accuracy': str(top_score),
                        'db': str(db)
                    })
                    print(f"EMITTING: {top_label} ({top_score:.2f})")
                else:
                    print(f"Top prediction {top_label} ({top_score:.4f}) below threshold, not emitting")
                    socketio.emit('audio_label', {
                        'label': 'Unrecognized Sound',
                        'accuracy': '0.2',
                        'db': str(db)
                    })
        else:
            print("No valid predictions from PANNs model")
            # Fall back to TensorFlow model
            predict_with_tensorflow(np_wav, record_time, db)
            
    except Exception as e:
        print(f"Error with PANNs prediction: {e}")
        traceback.print_exc()
        print("Falling back to TensorFlow model due to PANNs error")
        predict_with_tensorflow(np_wav, record_time, db)

recent_audio_buffer = []
MAX_BUFFER_SIZE = 5

def process_speech_with_sentiment(audio_data):
    """Process speech audio, transcribe it and analyze sentiment."""
    SPEECH_MAX_BUFFER_SIZE = 5
    MIN_WORD_COUNT = 3
    MIN_CONFIDENCE = 0.7
    
    if not hasattr(process_speech_with_sentiment, "recent_audio_buffer"):
        process_speech_with_sentiment.recent_audio_buffer = []
    
    process_speech_with_sentiment.recent_audio_buffer.append(audio_data)
    
    if len(process_speech_with_sentiment.recent_audio_buffer) > SPEECH_MAX_BUFFER_SIZE:
        process_speech_with_sentiment.recent_audio_buffer = process_speech_with_sentiment.recent_audio_buffer[-SPEECH_MAX_BUFFER_SIZE:]
    
    if len(process_speech_with_sentiment.recent_audio_buffer) > 1:
        num_chunks = min(SPEECH_MAX_BUFFER_SIZE, len(process_speech_with_sentiment.recent_audio_buffer))
        logger.info(f"Using concatenated audio from {num_chunks} chunks for speech transcription")
        concatenated_audio = np.concatenate(process_speech_with_sentiment.recent_audio_buffer[-num_chunks:])
    else:
        concatenated_audio = audio_data
    
    min_samples = RATE * 4.0
    if len(concatenated_audio) < min_samples:
        pad_size = int(min_samples) - len(concatenated_audio)
        concatenated_audio = np.pad(concatenated_audio, (0, pad_size), mode='reflect')
        logger.info(f"Padded speech audio to size: {len(concatenated_audio)} samples ({len(concatenated_audio)/RATE:.1f} seconds)")
    
    rms = np.sqrt(np.mean(np.square(concatenated_audio)))
    if rms < 0.001:
        logger.info(f"Audio too quiet (RMS: {rms:.4f}), skipping transcription")
        return {
            "text": "",
            "sentiment": {
                "category": "Neutral",
                "original_emotion": "neutral",
                "confidence": 0.5,
                "emoji": "😐"
            }
        }
    
    if rms < 0.05:
        logger.info(f"Boosting audio signal (original RMS: {rms:.4f})")
        target_rms = 0.1
        gain_factor = target_rms / (rms + 1e-10)
        enhanced_audio = concatenated_audio * gain_factor
        if np.max(np.abs(enhanced_audio)) > 0.99:
            logger.info("Using peak normalization to avoid clipping")
            peak_value = np.max(np.abs(concatenated_audio))
            if peak_value > 0:
                gain_factor = 0.95 / peak_value
                enhanced_audio = concatenated_audio * gain_factor
            else:
                enhanced_audio = concatenated_audio
        new_rms = np.sqrt(np.mean(np.square(enhanced_audio)))
        logger.info(f"Audio boosted from RMS {rms:.4f} to {new_rms:.4f}")
        concatenated_audio = enhanced_audio
    
    logger.info("Transcribing speech to text...")
    if USE_GOOGLE_SPEECH:
        transcription = transcribe_with_google(concatenated_audio, RATE)
        logger.info(f"Used Google Cloud Speech-to-Text for transcription")
    else:
        transcription = speech_processor.transcribe(concatenated_audio, RATE)
        logger.info(f"Used Whisper for transcription")
    
    if not transcription:
        logger.info("No valid transcription found")
        return None
    
    common_words = ["the", "a", "an", "and", "but", "or", "if", "then", "so", "to", "of", "for", "in", "on", "at"]
    meaningful_words = [word for word in transcription.lower().split() if word not in common_words]
    
    if len(meaningful_words) < MIN_WORD_COUNT:
        logger.info(f"Transcription has too few meaningful words: '{transcription}'")
        return None
    
    logger.info(f"Transcription: {transcription}")
    sentiment = analyze_sentiment(transcription)
    
    if sentiment:
        result = {
            "text": transcription,
            "sentiment": sentiment
        }
        logger.info(f"Sentiment analysis result: Speech {sentiment['category']} with emoji {sentiment['emoji']}")
        return result
    
    return None

def background_thread():
    """Example of how to send server generated events to clients."""
    count = 0
    while True:
        socketio.sleep(10)
        count += 1
        socketio.emit('my_response',
                      {'data': 'Server generated event', 'count': count},
                      namespace='/test')

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """API health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": time.time()})

@app.route('/status')
def status():
    """Return the status of the server, including model loading status."""
    ip_addresses = get_ip_addresses()
    return jsonify({
        'status': 'running',
        'tensorflow_model_loaded': models["tensorflow"] is not None,
        'ast_model_loaded': models["ast"] is not None,
        'using_ast_model': USE_AST_MODEL,
        'speech_recognition': 'Google Cloud' if USE_GOOGLE_SPEECH else 'Whisper',
        'sentiment_analysis_enabled': True,
        'ip_addresses': ip_addresses,
        'uptime': time.time() - start_time,
        'version': '1.2.0',
        'active_clients': len(active_clients)
    })

@app.route('/api/toggle-speech-recognition', methods=['POST'])
def toggle_speech_recognition():
    """Toggle between Whisper and Google Cloud Speech-to-Text"""
    global USE_GOOGLE_SPEECH
    data = request.get_json()
    
    if data and 'use_google_speech' in data:
        USE_GOOGLE_SPEECH = data['use_google_speech']
        logger.info(f"Speech recognition system changed to: {'Google Cloud' if USE_GOOGLE_SPEECH else 'Whisper'}")
        return jsonify({
            "success": True,
            "message": f"Speech recognition system set to {'Google Cloud' if USE_GOOGLE_SPEECH else 'Whisper'}",
            "use_google_speech": USE_GOOGLE_SPEECH
        })
    else:
        USE_GOOGLE_SPEECH = not USE_GOOGLE_SPEECH
        logger.info(f"Speech recognition system toggled to: {'Google Cloud' if USE_GOOGLE_SPEECH else 'Whisper'}")
        return jsonify({
            "success": True,
            "message": f"Speech recognition system toggled to {'Google Cloud' if USE_GOOGLE_SPEECH else 'Whisper'}",
            "use_google_speech": USE_GOOGLE_SPEECH
        })

@socketio.on('send_message')
def handle_source(json_data):
    print('Receive message...' + str(json_data['message']))
    text = json_data['message'].encode('ascii', 'ignore')
    socketio.emit('echo', {'echo': 'Server Says: ' + str(text)})
    print('Sending message back..')

@socketio.on('disconnect_request', namespace='/test')
def disconnect_request():
    @copy_current_request_context
    def can_disconnect():
        disconnect()

    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': 'Disconnected!', 'count': session['receive_count']},
         callback=can_disconnect)

@socketio.on('connect', namespace='/test')
def test_connect():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)
    emit('my_response', {'data': 'Connected', 'count': 0})

@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected', request.sid)

@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")
    emit('server_status', {'status': 'connected', 'message': 'Connected to SoundWatch server'})

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")

def aggregate_predictions(new_prediction, label_list, is_speech=False, num_samples=None):
    """Aggregate predictions from multiple overlapping segments to improve accuracy.
    
    Args:
        new_prediction: The new prediction array to add to history
        label_list: List of class labels
        is_speech: Whether to use speech-specific aggregation
        num_samples: Optional number of samples to consider (if None, use all available)
    """
    global recent_predictions, speech_predictions
    
    with prediction_lock:
        if is_speech:
            speech_predictions.append(new_prediction)
            if len(speech_predictions) > SPEECH_PREDICTIONS_HISTORY:
                speech_predictions = speech_predictions[-SPEECH_PREDICTIONS_HISTORY:]
            predictions_list = speech_predictions
            history_len = SPEECH_PREDICTIONS_HISTORY
            logger.info(f"Using speech-specific aggregation with {len(predictions_list)} samples")
        else:
            recent_predictions.append(new_prediction)
            if len(recent_predictions) > MAX_PREDICTIONS_HISTORY:
                recent_predictions = recent_predictions[-MAX_PREDICTIONS_HISTORY:]
            predictions_list = recent_predictions
            history_len = MAX_PREDICTIONS_HISTORY
        
        # If num_samples is specified, limit the number of predictions to use
        if num_samples is not None and num_samples > 0 and num_samples < len(predictions_list):
            predictions_list = predictions_list[-num_samples:]
            logger.info(f"Using most recent {num_samples} samples for aggregation")
        
        if len(predictions_list) > 1:
            expected_shape = predictions_list[0].shape
            valid_predictions = []
            for pred in predictions_list:
                if pred.shape == expected_shape:
                    valid_predictions.append(pred)
                else:
                    logger.warning(f"Skipping prediction with incompatible shape: {pred.shape} (expected {expected_shape})")
            if valid_predictions:
                aggregated = np.zeros_like(valid_predictions[0])
                for pred in valid_predictions:
                    aggregated += pred
                aggregated /= len(valid_predictions)
                logger.info(f"Aggregating {len(valid_predictions)} predictions {'(speech)' if is_speech else ''}")
            else:
                logger.warning("No predictions with matching shapes, using most recent prediction")
                aggregated = predictions_list[-1]
        else:
            aggregated = new_prediction
        
        orig_top_idx = np.argmax(new_prediction)
        agg_top_idx = np.argmax(aggregated)
        
        if orig_top_idx != agg_top_idx:
            # Handle different label_list types
            if isinstance(label_list, dict):
                # Dictionary type - find label by index value
                orig_label = "unknown"
                agg_label = "unknown"
                for label, idx in label_list.items():
                    if idx == orig_top_idx:
                        orig_label = label
                    if idx == agg_top_idx:
                        agg_label = label
            elif isinstance(label_list, (list, tuple, np.ndarray)) and len(label_list) > 0:
                # List type - use direct indexing if in range
                orig_label = label_list[orig_top_idx] if orig_top_idx < len(label_list) else f"unknown({orig_top_idx})"
                agg_label = label_list[agg_top_idx] if agg_top_idx < len(label_list) else f"unknown({agg_top_idx})"
            else:
                # Other types or empty label_list - just use index values
                orig_label = f"index_{orig_top_idx}"
                agg_label = f"index_{agg_top_idx}"
                
            logger.info(f"Aggregation changed top prediction: {orig_label} ({new_prediction[orig_top_idx]:.4f}) -> {agg_label} ({aggregated[agg_top_idx]:.4f})")
        else:
            # Same handling for unchanged prediction
            if isinstance(label_list, dict):
                # Dictionary type
                label = "unknown"
                for lbl, idx in label_list.items():
                    if idx == orig_top_idx:
                        label = lbl
                        break
            elif isinstance(label_list, (list, tuple, np.ndarray)) and len(label_list) > 0:
                # List type
                label = label_list[orig_top_idx] if orig_top_idx < len(label_list) else f"unknown({orig_top_idx})"
            else:
                # Other types
                label = f"index_{orig_top_idx}"
                
            logger.info(f"Aggregation kept same top prediction: {label}, confidence: {new_prediction[orig_top_idx]:.4f} -> {aggregated[agg_top_idx]:.4f}")
        
        return aggregated

@socketio.on('predict')
def predict(message):
    audio_data = np.array(message['audio_data'])
    timestamp = message['timestamp']

    if USE_AST_MODEL:
        print("Using AST model for prediction")
        pass
    elif USE_PANNS_MODEL:
        print("Using PANNs model for prediction")
        try:
            panns_results = panns_model.predict_with_panns(audio_data, top_k=10, threshold=0.1, map_to_homesounds_format=True)
            panns_results["timestamp"] = timestamp
            emit('prediction', panns_results)
            cleanup_memory()
        except Exception as e:
            print(f"Error with PANNs prediction: {e}")
            traceback.print_exc()
            print("Falling back to TensorFlow model due to PANNs error")
            pass
    else:
        print("Using TensorFlow model for prediction")
        pass

@socketio.on('predict_raw')
def predict_raw(message):
    if not USE_PANNS_MODEL:
        print("PANNs model is not enabled, ignoring predict_raw request")
        return

    audio_data = np.array(message['audio_data'])
    sample_rate = int(message.get('sample_rate', 32000))
    timestamp = message['timestamp']
    
    try:
        results = panns_model.predict_with_panns(
            audio_data, 
            top_k=10, 
            threshold=0.1, 
            map_to_homesounds_format=False
        )
        output = {
            "timestamp": timestamp,
            "output": [{"label": label, "score": float(score)} for label, score in results]
        }
        emit('prediction_raw', output)
        cleanup_memory()
    except Exception as e:
        print(f"Error with PANNs raw prediction: {e}")
        traceback.print_exc()
        emit('prediction_raw', {
            "timestamp": timestamp,
            "output": [],
            "error": str(e)
        })

# Function to predict using TensorFlow model
def predict_with_tensorflow(audio_data, record_time=None, db=None):
    """Predict sounds using the TensorFlow model"""
    try:
        # Store original audio data shape
        orig_shape = audio_data.shape
        orig_size = audio_data.size
        print(f"Original np_wav shape: {orig_shape}, size: {orig_size}")
        
        # Pad audio to 1-second length if too short
        if audio_data.size < 16000:  # Assuming 16kHz sampling rate
            padded_data = np.zeros(16000)
            padded_data[:audio_data.size] = audio_data
            audio_data = padded_data
            print(f"Padded audio data to size: {audio_data.size} samples (1 second)")
        
        # Convert to vggish input
        with tf_lock:
            try:
                # Extract features (will return multiple frames for longer audio)
                examples = waveform_to_examples(audio_data, RATE)
                
                # Take first frame for simplicity
                if examples.shape[0] > 1:
                    print(f"Using first frame from multiple frames: {examples.shape[1:3]}")
                example = examples[0, :]
                example = example.reshape(1, *example.shape, 1)  # Add batch and channel dimensions
                print(f"Processed audio data shape: {example.shape}")
                
                # Make prediction with TensorFlow model
                print("Making prediction with TensorFlow model...")
                with tf_graph.as_default():
                    with tf_session.as_default():
                        prediction = models["tensorflow"].predict(example)[0]
                        
                        try:
                            # Debug predictions
                            debug_predictions(prediction, homesounds.labels)
                            
                            # Find top prediction
                            top_idx = np.argmax(prediction)
                            top_score = float(prediction[top_idx])
                            
                            # Check if it might be speech
                            speech_idx = -1
                            is_speech_prediction = False
                            
                            # Find speech index if exists
                            if isinstance(homesounds.labels, dict):
                                for label, idx in homesounds.labels.items():
                                    if label.lower() == "speech":
                                        speech_idx = idx
                                        break
                            elif isinstance(homesounds.labels, (list, tuple, np.ndarray)) and len(homesounds.labels) > 0:
                                if "speech" in homesounds.labels:
                                    speech_idx = homesounds.labels.index("speech")
                                else:
                                    # Try case-insensitive search
                                    for idx, label in enumerate(homesounds.labels):
                                        if isinstance(label, str) and label.lower() == "speech":
                                            speech_idx = idx
                                            break
                            
                            if speech_idx >= 0 and speech_idx < len(prediction) and prediction[speech_idx] > SPEECH_DETECTION_THRES:
                                is_speech_prediction = True
                                logger.info(f"Detected potential speech with confidence: {prediction[speech_idx]:.4f}")
                            
                            # Aggregate predictions (potentially considering speech)
                            aggregated = aggregate_predictions(
                                prediction, 
                                homesounds.labels,
                                is_speech=is_speech_prediction, 
                                num_samples=1
                            )
                            
                            # Log aggregation results
                            agg_top_idx = np.argmax(aggregated)
                            agg_top_score = float(aggregated[agg_top_idx])
                            
                            # Get label names for logging
                            if agg_top_idx != top_idx:
                                if isinstance(homesounds.labels, dict):
                                    # Find labels for indices
                                    orig_label = "unknown"
                                    agg_label = "unknown"
                                    for label, idx in homesounds.labels.items():
                                        if idx == top_idx:
                                            orig_label = label
                                        if idx == agg_top_idx:
                                            agg_label = label
                                    logger.info(f"Aggregation changed top prediction: {orig_label} ({top_score:.4f}) -> {agg_label} ({agg_top_score:.4f})")
                                elif isinstance(homesounds.labels, (list, tuple, np.ndarray)) and len(homesounds.labels) > 0:
                                    orig_label = homesounds.labels[top_idx] if top_idx < len(homesounds.labels) else "unknown"
                                    agg_label = homesounds.labels[agg_top_idx] if agg_top_idx < len(homesounds.labels) else "unknown"
                                    logger.info(f"Aggregation changed top prediction: {orig_label} ({top_score:.4f}) -> {agg_label} ({agg_top_score:.4f})")
                                else:
                                    logger.info(f"Aggregation changed top prediction index: {top_idx} ({top_score:.4f}) -> {agg_top_idx} ({agg_top_score:.4f})")
                            else:
                                if isinstance(homesounds.labels, dict):
                                    # Find label for index
                                    label = "unknown"
                                    for lbl, idx in homesounds.labels.items():
                                        if idx == top_idx:
                                            label = lbl
                                            break
                                    logger.info(f"Aggregation kept same top prediction: {label}, confidence: {top_score:.4f} -> {agg_top_score:.4f}")
                                elif isinstance(homesounds.labels, (list, tuple, np.ndarray)) and len(homesounds.labels) > 0:
                                    label = homesounds.labels[top_idx] if top_idx < len(homesounds.labels) else "unknown"
                                    logger.info(f"Aggregation kept same top prediction: {label}, confidence: {top_score:.4f} -> {agg_top_score:.4f}")
                                else:
                                    logger.info(f"Aggregation kept same top prediction index: {top_idx}, confidence: {top_score:.4f} -> {agg_top_score:.4f}")
                            
                            # Log aggregated predictions for debugging
                            logger.info("Aggregated predictions:")
                            debug_predictions(aggregated, homesounds.labels)
                            
                            # Get top prediction after aggregation
                            top_idx = np.argmax(aggregated)
                            top_score = float(aggregated[top_idx])
                            
                            # Get the label for the top prediction
                            top_label = "Unknown"
                            if isinstance(homesounds.labels, dict):
                                # Find label for index
                                for lbl, idx in homesounds.labels.items():
                                    if idx == top_idx:
                                        top_label = lbl
                                        break
                            elif isinstance(homesounds.labels, (list, tuple, np.ndarray)) and len(homesounds.labels) > 0 and top_idx < len(homesounds.labels):
                                top_label = homesounds.labels[top_idx]
                            
                            # Human-readable conversion if available
                            human_label = top_label
                            if hasattr(homesounds, 'to_human_labels') and isinstance(homesounds.to_human_labels, dict) and top_label in homesounds.to_human_labels:
                                human_label = homesounds.to_human_labels[top_label]
                            
                            print(f"Top prediction: {human_label} ({top_score:.4f})")
                            
                            # Handle speech detection
                            if top_label.lower() == "speech" and top_score > SPEECH_DETECTION_THRES:
                                print(f"Speech detected with TensorFlow model. Processing sentiment...")
                                if os.environ.get('USE_SPEECH', '0') == '1' and os.environ.get('USE_SENTIMENT', '0') == '1':
                                    # Process speech
                                    process_speech(audio_data, record_time, top_score)
                                else:
                                    # Normal sound emission
                                    socketio.emit('audio_label', {
                                        'label': human_label,
                                        'accuracy': str(top_score),
                                        'db': str(db)
                                    })
                                    print(f"EMITTING: {human_label} ({top_score:.2f})")
                            else:
                                # Emit the top prediction if above threshold
                                if top_score > PREDICTION_THRES:
                                    socketio.emit('audio_label', {
                                        'label': human_label,
                                        'accuracy': str(top_score),
                                        'db': str(db)
                                    })
                                    print(f"EMITTING: {human_label} ({top_score:.2f})")
                                else:
                                    print(f"Top prediction {human_label} ({top_score:.4f}) below threshold, not emitting")
                                    socketio.emit('audio_label', {
                                        'label': 'Unrecognized Sound',
                                        'accuracy': '0.2',
                                        'db': str(db)
                                    })
                        except Exception as inner_e:
                            print(f"Error processing TensorFlow prediction results: {inner_e}")
                            traceback.print_exc()
                            # Emit an error
                            socketio.emit('audio_label', {
                                'label': 'Error Processing',
                                'accuracy': '0.0',
                                'db': str(db)
                            })
            except Exception as e:
                print(f"Error during TensorFlow prediction: {e}")
                traceback.print_exc()
                # Emit an error
                socketio.emit('audio_label', {
                    'label': 'Error Processing',
                    'accuracy': '0.0',
                    'db': str(db)
                })
    except Exception as e:
        print(f"Error in predict_with_tensorflow: {e}")
        traceback.print_exc()
        # Emit an error
        socketio.emit('audio_label', {
            'label': 'Error Processing',
            'accuracy': '0.0',
            'db': str(db)
        })
    
    # Always clean up memory
    cleanup_memory()

# Function to process speech with optional sentiment analysis
def process_speech(audio_data, record_time=None, confidence=0.0):
    """Process speech audio with transcription and optional sentiment analysis."""
    try:
        # Ensure we have enough audio data for proper speech recognition
        # Typically we want at least 1 second, preferably more
        if audio_data.size < 16000:  # Assuming 16kHz sampling rate
            logger.info(f"Padded speech audio to size: 64000 samples (4.0 seconds)")
            padded_data = np.zeros(64000)  # 4 seconds
            padded_data[:audio_data.size] = audio_data
            audio_data = padded_data
        
        # Boost audio signal for better transcription
        rms = np.sqrt(np.mean(audio_data**2))
        logger.info(f"Boosting audio signal (original RMS: {rms:.4f})")
        
        # Target RMS around 0.1 is usually good for speech recognition
        if rms > 0:
            target_rms = 0.1
            gain = target_rms / rms
            
            # Limit gain to avoid clipping
            max_abs = np.max(np.abs(audio_data))
            if max_abs * gain > 0.9:
                logger.info(f"Using peak normalization to avoid clipping")
                gain = 0.9 / max_abs
                
            # Apply gain
            boosted_audio = audio_data * gain
            new_rms = np.sqrt(np.mean(np.square(boosted_audio)))
            logger.info(f"Audio boosted from RMS {rms:.4f} to {new_rms:.4f}")
            audio_data = boosted_audio
        
        # Transcribe speech
        logger.info(f"Transcribing speech to text...")
        print("Transcribing with enhanced audio processing...")
        
        # Choose which speech recognition system to use
        transcription_result = None
        
        if os.environ.get('USE_GOOGLE_SPEECH', '0') == '1':
            # Use Google Cloud Speech
            if models.get("google_speech_processor", None) is None:
                # Lazy-load Google Cloud Speech
                from google_speech import GoogleSpeechToText
                models["google_speech_processor"] = GoogleSpeechToText()
            
            transcription_result = models["google_speech_processor"].transcribe(audio_data, sample_rate=RATE)
            logger.info(f"Used Google Cloud Speech for transcription")
        else:
            # Use Whisper
            transcription_result = models["speech_processor"].transcribe(audio_data, sample_rate=RATE)
            logger.info(f"Used Whisper for transcription")
        
        # Check if we have a valid transcription
        if not transcription_result or not transcription_result.get('text'):
            logger.info(f"No valid transcription found")
            # Just emit the speech label without sentiment
            socketio.emit('audio_label', {
                'label': 'Speech',
                'accuracy': str(confidence),
                'db': '-30',
                'timestamp': record_time
            })
            return
        
        # We have a valid transcription
        text = transcription_result['text']
        print(f"Transcribed: '{text}'")
        
        # Apply sentiment analysis if enabled
        if os.environ.get('USE_SENTIMENT', '0') == '1':
            sentiment_result = analyze_sentiment(text)
            if sentiment_result:
                # Format the result for emission
                category = sentiment_result.get('category', 'Neutral')
                emoji = sentiment_result.get('emoji', '😐')
                emotion = sentiment_result.get('original_emotion', 'neutral')
                sentiment_score = sentiment_result.get('confidence', 0.5)
                
                # Create the label with emoji
                label = f"Speech {category}"
                
                # Emit result with all sentiment information
                socketio.emit('audio_label', {
                    'label': label,
                    'accuracy': str(confidence),
                    'db': '-30',
                    'timestamp': record_time,
                    'emoji': emoji,
                    'transcription': text,
                    'emotion': emotion,
                    'sentiment_score': str(sentiment_score)
                })
                print(f"Emitting speech with sentiment: {label} ({emoji})")
                return
        
        # If we get here, just emit the speech with transcription
        socketio.emit('audio_label', {
            'label': 'Speech',
            'accuracy': str(confidence),
            'db': '-30',
            'timestamp': record_time,
            'transcription': text
        })
        print(f"Emitting speech with transcription (no sentiment)")
        
    except Exception as e:
        print(f"Error processing speech: {str(e)}")
        traceback.print_exc()
        # Fall back to basic speech detection
        socketio.emit('audio_label', {
            'label': 'Speech',
            'accuracy': str(confidence),
            'db': '-30',
            'timestamp': record_time
        })
        print("Emitting basic speech (error in processing)")
    finally:
        # Clean up memory
        cleanup_memory()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sonarity Audio Analysis Server')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on (default: 8080)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--use-google-speech', action='store_true', help='Use Google Cloud Speech-to-Text instead of Whisper')
    args = parser.parse_args()
    
    if args.use_google_speech:
        USE_GOOGLE_SPEECH = True
        logger.info("Using Google Cloud Speech-to-Text for speech recognition")
    else:
        USE_GOOGLE_SPEECH = False
        logger.info("Using Whisper for speech recognition")
    
    print("=====")
    print("Setting up sound recognition models...")
    load_models()
    
    ip_addresses = get_ip_addresses()
    
    print("\n" + "="*60)
    print("SONARITY SERVER STARTED")
    print("="*60)
    
    if ip_addresses:
        print("Server is available at:")
        for i, ip in enumerate(ip_addresses):
            print(f"{i+1}. http://{ip}:{args.port}")
            print(f"   WebSocket: ws://{ip}:{args.port}")
        print("\nExternal access: http://34.16.101.179:%d" % args.port)
        print("External WebSocket: ws://34.16.101.179:%d" % args.port)
        print("\nPreferred connection address: http://%s:%d" % (ip_addresses[0], args.port))
        print("Preferred WebSocket address: ws://%s:%d" % (ip_addresses[0], args.port))
    else:
        print("Could not determine IP address. Make sure you're connected to a network.")
        print(f"Try connecting to your server's IP address on port {args.port}")
        print("\nExternal access: http://34.16.101.179:%d" % args.port)
        print("External WebSocket: ws://34.16.101.179:%d" % args.port)
    
    print("="*60 + "\n")
    
    port = int(os.environ.get('PORT', args.port))
    socketio.run(app, host='0.0.0.0', port=port, debug=args.debug)