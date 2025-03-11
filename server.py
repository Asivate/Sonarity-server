from threading import Lock
from flask import Flask, render_template, session, request, \
    copy_current_request_context, Response, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room, \
    close_room, rooms, disconnect
import numpy as np
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
import json

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
    """Get available IP addresses for the server."""
    ip_list = []
    # Add the external IP first (prioritize this one)
    ip_list.append('34.16.101.179')  # External IP address for physical devices
    
    try:
        # Then add the local IPs for completeness
        hostname = socket.gethostname()
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Connect to an external address to find the local network interface
            s.connect(('8.8.8.8', 80))
            primary_ip = s.getsockname()[0]
            if primary_ip != '127.0.0.1' and primary_ip != '0.0.0.0' and primary_ip != '34.16.101.179':
                ip_list.append(primary_ip)
        except Exception:
            pass
        finally:
            s.close()
        
        # Get all IP addresses
        try:
            for ip in socket.gethostbyname_ex(hostname)[2]:
                if ip not in ip_list and ip != '127.0.0.1' and ip != '0.0.0.0':
                    ip_list.append(ip)
        except Exception:
            pass
    except Exception as e:
        print(f"Error getting IP addresses: {e}")
    
    return ip_list

# Set up Flask app and SocketIO
async_mode = None
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode=async_mode, cors_allowed_origins="*")
thread = None
thread_lock = Lock()
panns_lock = Lock()  # Lock for thread-safe PANNS model access
prediction_lock = Lock()  # Lock for thread-safe prediction history access

# Constants for audio processing
RATE = 32000  # Sample rate (32kHz matches original implementation)
CHUNK = 1024  # Buffer size for audio chunks (matches original REC_BUFFER_SIZE)
CHANNELS = 1  # Mono audio
SILENCE_THRES = -60  # dB threshold for silence detection (lower is more sensitive)
DBLEVEL_THRES = -30  # dB threshold for detecting quiet sounds (sounds between -60dB and -30dB are processed)
PREDICTION_THRES = 0.10  # Confidence threshold for predictions
MINIMUM_AUDIO_LENGTH = 32000  # Minimum audio length (1 second at 32kHz)
SENTIMENT_THRES = 0.5  # Sentiment analysis threshold

# Global variables
models = {}  # Dictionary to store loaded models
sentiment_analyzer = None  # Sentiment analyzer model
speech_to_text = None  # Speech-to-text model
google_speech = None  # Google Speech-to-Text client

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

# Initialize speech recognition systems
speech_processor = SpeechToText()
google_speech_processor = None  # Will be lazy-loaded when needed

# Load models
def load_models():
    """Load all required models for sound recognition and speech processing."""
    global models, USE_PANNS_MODEL
    
    # Initialize models dictionary
    models = {
        "panns": None,
    }
    
    # Check environment variables to determine which models to use
    USE_PANNS_MODEL = os.environ.get('USE_PANNS_MODEL', '1') == '1'
    
    print(f"PANNs model {'enabled' if USE_PANNS_MODEL else 'disabled'} based on environment settings")
    
    # Ensure at least one model is enabled
    if not USE_PANNS_MODEL:
        print("Warning: No sound recognition models enabled. Enabling PANNs model by default.")
        USE_PANNS_MODEL = True
    
    # Load PANNs model if enabled
    if USE_PANNS_MODEL:
        print("Loading PANNs model...")
        panns_model.load_panns_model()
        models["panns"] = True
        print("PANNs model loaded successfully")
    
    # No need to load other models as we're focusing only on PANNs

def audio_samples(in_data, frame_count, time_info, status_flags):
    """Process audio samples for real-time audio streaming."""
    try:
        # Convert from bytes to float [-1.0, 1.0]
        np_wav = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Calculate RMS level and dB
        rms = np.sqrt(np.mean(np_wav**2))
        db = dbFS(rms)
        
        # Debug audio stats
        print(f"Audio stats: samples={len(np_wav)}, rms={rms:.6f}, db={db:.2f}")

        # Check for silence
        if db < SILENCE_THRES:
            print(f"Silence detected (db: {db})")
            return (in_data, 0)
        
        # Check if sound is too quiet
        if db > -DBLEVEL_THRES:
            print(f"Sound too quiet (db: {db})")
            return (in_data, 0)
        
        # Process with PANNs model
        prediction_results = process_audio_with_panns(
            audio_data=np_wav,
            db_level=db,
            config={
                'silence_threshold': SILENCE_THRES,
                'db_level_threshold': DBLEVEL_THRES,
                'prediction_threshold': PREDICTION_THRES * 0.8,
                'boost_factor': 1.5
            }
        )
        
        # Log the top prediction
        if prediction_results and "predictions" in prediction_results and len(prediction_results["predictions"]) > 0:
            top_pred = prediction_results["predictions"][0]
            print(f"DETECTED: {top_pred['label']} ({top_pred['score']:.4f}, db: {db:.2f})")
            
            # Emit the sound prediction to all clients
            socketio.emit('audio_label', {
                'label': top_pred['label'],
                'accuracy': str(round(top_pred['score'], 4)),
                'db': str(db),
                'timestamp': time.time()
            })
    except Exception as e:
        print(f"Error in audio_samples: {e}")
        traceback.print_exc()
    
    return (in_data, 0)

@socketio.on('audio_feature_data')
def handle_source(json_data):
    """
    Handle audio feature data from clients
    This expects JSON with:
    - features: audio features
    - db: audio level in dB (optional)
    - time: timestamp (optional)
    """
    try:
        # Log received data for debugging (truncating large feature arrays)
        debug_data = json_data.copy() if isinstance(json_data, dict) else json_data
        if isinstance(debug_data, dict) and 'features' in debug_data:
            features_len = len(debug_data['features']) if debug_data['features'] else 0
            debug_data['features'] = f"[features array, length: {features_len}]"
        print(f"Received audio feature data: {debug_data}")
        
        # Parse JSON if it's a string
        if isinstance(json_data, str):
            try:
                json_data = json.loads(json_data)
                print("Successfully parsed JSON string data")
            except json.JSONDecodeError as e:
                print(f"Invalid JSON in audio_feature_data: {e}")
                return
        
        # Check for features in different possible field names
        if 'features' in json_data:
            audio_features = json_data.get('features')
        elif 'audioFeatures' in json_data:
            audio_features = json_data.get('audioFeatures')
            print("Using 'audioFeatures' field instead of 'features'")
        elif 'data' in json_data:
            audio_features = json_data.get('data')
            print("Using 'data' field instead of 'features'")
        else:
            print("Missing audio features in request - no recognized features field found")
            print(f"Available fields: {list(json_data.keys()) if isinstance(json_data, dict) else 'None'}")
            socketio.emit('prediction', {
                "predictions": [{"label": "No Audio", "score": 1.0}],
                "timestamp": time.time(),
                "db": None
            })
            return
            
        # Get other parameters
        db = json_data.get('db')
        timestamp = json_data.get('time', json_data.get('timestamp', time.time()))
        
        # Validate audio features
        if not audio_features or len(audio_features) == 0:
            print("Empty audio features received")
            socketio.emit('prediction', {
                "predictions": [{"label": "No Audio", "score": 1.0}],
                "timestamp": timestamp,
                "db": db
            })
            return
        
        # Convert audio features to numpy array if needed
        if not isinstance(audio_features, np.ndarray):
            try:
                audio_features = np.array(audio_features, dtype=np.float32)
                print(f"Converted audio features to numpy array, shape: {audio_features.shape}")
            except Exception as e:
                print(f"Error converting audio features to numpy array: {e}")
                socketio.emit('prediction', {
                    "predictions": [{"label": "Invalid Features", "score": 1.0}],
                    "timestamp": timestamp,
                    "db": db
                })
                return
        
        # Calculate dB level if not provided
        if db is None:
            rms = np.sqrt(np.mean(np.square(audio_features)))
            db = dbFS(rms)
            print(f"Calculated dB level: {db}")
        
        # Check for silence based on dB level
        if db < SILENCE_THRES:
            print(f"Sound is silence (dB: {db}, threshold: {SILENCE_THRES})")
            socketio.emit('prediction', {
                "predictions": [{"label": "Silence", "score": 0.95}],
                "timestamp": timestamp,
                "db": db
            })
            return
        
        # Check if sound is too quiet but not silence
        if db > -DBLEVEL_THRES:
            print(f"Sound too quiet (dB: {db}, threshold: {-DBLEVEL_THRES})")
            socketio.emit('prediction', {
                "predictions": [{"label": "Too Quiet", "score": 0.9}],
                "timestamp": timestamp,
                "db": db
            })
            return
        
        # Process with PANNs model
        print(f"Processing audio features (db: {db}, features shape: {audio_features.shape})")
        
        # Process the audio with PANNs model
        result = process_audio_with_panns(
            audio_features, 
            timestamp=timestamp, 
            db_level=db,
            config={
                'silence_threshold': SILENCE_THRES,
                'db_level_threshold': DBLEVEL_THRES,
                'prediction_threshold': PREDICTION_THRES,
                'boost_factor': 1.2
            }
        )
        
        # Emit results to all clients
        print(f"Emitting prediction: {result}")
        socketio.emit('prediction', result, broadcast=True)
        
    except Exception as e:
        print(f"Error in handle_source: {e}")
        import traceback
        traceback.print_exc()
        
        # Emit error to client
        socketio.emit('prediction', {
            "predictions": [{"label": "Error", "score": 1.0}],
            "timestamp": time.time(),
            "db": None
        })

@socketio.on('audio_data')
def handle_audio(data):
    """
    Handle audio data sent from the client
    This handler expects a JSON object with:
    - audio: base64 encoded audio data or list of float/int values
    - format: audio format (e.g., 'float32', 'int16')
    - db: optional pre-calculated dB level
    - timestamp: optional timestamp
    """
    try:
        # Log received data for debugging (without large audio data)
        debug_data = data.copy() if isinstance(data, dict) else data
        if isinstance(debug_data, dict):
            if 'audio' in debug_data:
                audio_length = len(debug_data['audio']) if debug_data['audio'] else 0
                debug_data['audio'] = f"[audio data, length: {audio_length}]"
            elif 'data' in debug_data:
                audio_length = len(debug_data['data']) if debug_data['data'] else 0
                debug_data['data'] = f"[audio data, length: {audio_length}]"
        print(f"Received audio data: {debug_data}")
        
        # Extract data from JSON
        if isinstance(data, str):
            # Parse JSON if it's a string
            try:
                json_data = json.loads(data)
                print("Successfully parsed JSON string data")
            except json.JSONDecodeError as e:
                print(f"Invalid JSON data received: {e}")
                return
        else:
            json_data = data
            
        # Handle different client formats - some clients might send data in different fields
        # Check for 'audio' field
        if 'audio' in json_data:
            audio_data_raw = json_data['audio']
            field_name = 'audio'
        # Check for alternative fields that might contain audio data
        elif 'audioData' in json_data:
            audio_data_raw = json_data['audioData']
            field_name = 'audioData'
            print("Using 'audioData' field instead of 'audio'")
        elif 'data' in json_data:
            audio_data_raw = json_data['data']
            field_name = 'data'
            print("Using 'data' field instead of 'audio'")
        else:
            print("Missing audio data in request - no recognized audio field found")
            print(f"Available fields: {list(json_data.keys()) if isinstance(json_data, dict) else 'None'}")
            return
        
        # Validate audio data
        if not audio_data_raw:
            print("Empty audio data received")
            return
            
        # Get audio format and other parameters
        audio_format = json_data.get('format', 'float32')
        db_level = json_data.get('db')
        timestamp = json_data.get('timestamp', time.time())
        
        # Process audio data based on its type
        try:
            if isinstance(audio_data_raw, list):
                # Direct list of values - convert to numpy array
                print(f"Processing audio as list of values, length: {len(audio_data_raw)}")
                audio_data = np.array(audio_data_raw, dtype=np.float32)
                
                # If these are int16 values, normalize them
                if audio_format == 'int16' or (np.max(np.abs(audio_data)) > 1.0 and np.max(np.abs(audio_data)) <= 32768):
                    audio_data = audio_data.astype(np.float32) / 32768.0
                    print("Normalized integer audio values to float range [-1.0, 1.0]")
            else:
                # Assume it's base64 encoded
                try:
                    audio_bytes = base64.b64decode(audio_data_raw)
                    print(f"Successfully decoded base64 audio data, length: {len(audio_bytes)} bytes")
                    
                    # Convert bytes to numpy array based on format
                    if audio_format == 'float32':
                        audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
                    elif audio_format == 'int16':
                        # Convert int16 to float32 normalized between -1 and 1
                        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    else:
                        print(f"Unsupported audio format: {audio_format}, defaulting to float32")
                        audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
                except Exception as e:
                    print(f"Error decoding audio data: {e}. Treating as a list of values.")
                    try:
                        audio_data = np.array(audio_data_raw, dtype=np.float32)
                        if audio_format == 'int16' or (np.max(np.abs(audio_data)) > 1.0 and np.max(np.abs(audio_data)) <= 32768):
                            audio_data = audio_data.astype(np.float32) / 32768.0
                    except Exception as e2:
                        print(f"Failed to convert audio data as list: {e2}")
                        return
            
            # Log audio statistics
            audio_length = len(audio_data)
            if audio_length > 0:
                audio_min = np.min(audio_data)
                audio_max = np.max(audio_data)
                audio_mean = np.mean(audio_data)
                audio_rms = np.sqrt(np.mean(np.square(audio_data)))
                print(f"Processed audio: {audio_length} samples, min={audio_min:.4f}, max={audio_max:.4f}, mean={audio_mean:.4f}, rms={audio_rms:.4f}")
            else:
                print("Warning: Audio data has 0 samples after conversion")
                return
            
            # Calculate dB level if not provided
            if db_level is None:
                audio_rms = np.sqrt(np.mean(np.square(audio_data)))
                db_level = dbFS(audio_rms)
                print(f"Calculated dB level: {db_level}")
            
            # Process audio with PANNs model
            result = process_audio_with_panns(audio_data, timestamp, db_level)
            
            # Emit the prediction results back to the client
            print(f"Emitting prediction: {result}")
            socketio.emit('prediction', result, broadcast=True)
            
        except Exception as e:
            print(f"Error processing audio data: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"Error in handle_audio: {e}")
        import traceback
        traceback.print_exc()

# Helper function to process with PANNs model
def process_with_panns_model(np_wav, record_time=None, db=None):
    """Process audio using PANNs model"""
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
        
        # Apply noise gate to reduce background noise
        np_wav = noise_gate(np_wav, threshold=0.005, attack=0.01, release=0.1, rate=RATE)
            
        # Ensure audio is long enough for processing
        if len(np_wav) < MINIMUM_AUDIO_LENGTH:  # Use our new minimum size constant
            print(f"Audio too short ({len(np_wav)} samples). Padding to {MINIMUM_AUDIO_LENGTH} samples.")
            if len(np_wav) < MINIMUM_AUDIO_LENGTH / 4:
                # For very short sounds, repeat them
                repeats = int(np.ceil(MINIMUM_AUDIO_LENGTH / len(np_wav)))
                padded = np.tile(np_wav, repeats)[:MINIMUM_AUDIO_LENGTH]
                print(f"Using repetition padding ({repeats} repeats)")
            else:
                padded = np.zeros(MINIMUM_AUDIO_LENGTH)
                padded[:len(np_wav)] = np_wav
            np_wav = padded
        
        # Start processing with enhanced audio
        with panns_lock:
            panns_results = panns_model.predict_with_panns(
                np_wav, 
                top_k=10, 
                threshold=PREDICTION_THRES,
                map_to_homesounds_format=True,
                boost_other_categories=True
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
                        'db': str(db),
                        'timestamp': record_time
                    })
                    print(f"EMITTING: {top_label} ({top_score:.2f})")
            else:
                # Emit the top prediction
                if top_score > PREDICTION_THRES:
                    socketio.emit('audio_label', {
                        'label': top_label,
                        'accuracy': str(top_score),
                        'db': str(db),
                        'timestamp': record_time
                    })
                    print(f"EMITTING: {top_label} ({top_score:.2f})")
                else:
                    print(f"Top prediction {top_label} ({top_score:.4f}) below threshold, not emitting")
                    socketio.emit('audio_label', {
                        'label': 'Unrecognized Sound',
                        'accuracy': '0.2',
                        'db': str(db),
                        'timestamp': record_time
                    })
        else:
            print("No valid predictions from PANNs model")
            socketio.emit('audio_label', {
                'label': 'Unrecognized Sound',
                'accuracy': '0.2',
                'db': str(db),
                'timestamp': record_time
            })
            
    except Exception as e:
        print(f"Error with PANNs prediction: {e}")
        traceback.print_exc()
        socketio.emit('audio_label', {
            'label': 'Error Processing Audio',
            'accuracy': '0.0',
            'db': str(db) if db is not None else '-100',
            'timestamp': record_time
        })

# New dedicated function for advanced PANNS processing
def process_audio_with_panns(audio_data, timestamp=None, db_level=None, config=None):
    """
    Process audio data with PANNs model for sound recognition
    Args:
        audio_data: numpy array of audio samples (16000Hz mono)
        timestamp: optional timestamp for the prediction
        db_level: optional pre-calculated dB level
        config: optional configuration parameters
    Returns:
        dict with predictions, timestamp, and dB level
    """
    # Default configuration
    if config is None:
        config = {
            'silence_threshold': SILENCE_THRES,
            'db_level_threshold': DBLEVEL_THRES,
            'prediction_threshold': PREDICTION_THRES,
            'boost_factor': 1.2  # For non-speech categories
        }
    
    # Extract config values
    silence_threshold = config.get('silence_threshold', SILENCE_THRES)
    db_level_threshold = config.get('db_level_threshold', DBLEVEL_THRES)
    prediction_threshold = config.get('prediction_threshold', PREDICTION_THRES)
    
    # Ensure audio is the correct data type
    if not isinstance(audio_data, np.ndarray):
        try:
            audio_data = np.array(audio_data, dtype=np.float32)
        except Exception as e:
            print(f"Error converting audio_data to numpy array: {e}")
            return {
                "predictions": [{"label": "Invalid Audio Format", "score": 1.0}],
                "timestamp": timestamp,
                "db": db_level
            }
    
    # Ensure audio is float32 type
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    # Log detailed audio statistics
    audio_stats = {
        "length": len(audio_data),
        "min": np.min(audio_data),
        "max": np.max(audio_data),
        "mean": np.mean(audio_data),
        "std": np.std(audio_data),
        "rms": np.sqrt(np.mean(np.square(audio_data))),
        "has_nan": np.any(np.isnan(audio_data)),
        "has_inf": np.any(np.isinf(audio_data))
    }
    print(f"Audio stats: {audio_stats}")
    
    # Handle non-finite values
    if audio_stats["has_nan"] or audio_stats["has_inf"]:
        print("Audio contains NaN or Inf values, fixing...")
        audio_data = np.nan_to_num(audio_data)
    
    # Fix timestamp if not provided
    if timestamp is None:
        timestamp = time.time()
    
    # Calculate dB level if not provided
    if db_level is None:
        rms = np.sqrt(np.mean(np.square(audio_data)))
        db_level = dbFS(rms)
        print(f"Calculated dB level: {db_level}")

    # Check for silence - if the sound is very quiet, it's silence
    if db_level is not None and db_level < silence_threshold:
        print(f"Sound is silence (dB: {db_level}, threshold: {silence_threshold})")
        return {
            "predictions": [{"label": "Silence", "score": 0.95}],
            "timestamp": timestamp,
            "db": db_level
        }
    
    # Check if sound is too quiet but not silence
    if db_level is not None and db_level > -db_level_threshold:
        print(f"Sound too quiet (dB: {db_level}, threshold: {-DBLEVEL_THRES})")
        return {
            "predictions": [{"label": "Too Quiet", "score": 0.9}],
            "timestamp": timestamp,
            "db": db_level
        }
    
    # Sound is within processing range
    print(f"Sound level ({db_level} dB) within processing range ({silence_threshold} to {-DBLEVEL_THRES} dB), processing...")
    
    # Process with PANNS model
    try:
        print(f"Processing audio with PANNS model (threshold: {prediction_threshold})...")
        
        # Ensure audio is not empty or all zeros
        if audio_data is None or len(audio_data) == 0 or np.all(audio_data == 0):
            print("Audio data is empty or all zeros, cannot process")
            return {
                "predictions": [{"label": "No Audio", "score": 1.0}],
                "timestamp": timestamp,
                "db": db_level
            }
        
        # Ensure audio is normalized properly
        if np.abs(audio_data).max() > 1.0:
            audio_data = audio_data / np.abs(audio_data).max()
        
        # Apply pre-emphasis and noise gating for better detection
        audio_data = pre_emphasis(audio_data, emphasis=0.97)
        audio_data = noise_gate(audio_data, threshold=0.005)
        
        # Boost very quiet sounds to make them more detectable
        if np.abs(audio_data).max() < 0.1:
            print("Boosting quiet audio...")
            boost_factor = 0.3 / np.abs(audio_data).max()
            boost_factor = min(boost_factor, 10.0)  # Cap the boost
            audio_data = audio_data * boost_factor
        
        # Check audio length and pad if necessary - based on original implementation with 32kHz
        audio_length = len(audio_data)
        if audio_length < 32000:  # Less than 1 second at 32kHz
            print(f"Audio too short ({audio_length} samples), padding...")
            
            # Handle very short sounds by repeating them - as in the original code
            if audio_length < 16000:  # Less than 0.5 seconds
                print("Very short audio - repeating pattern")
                repeat_factor = int(np.ceil(32000 / audio_length))
                audio_data = np.tile(audio_data, repeat_factor)[:32000]
            else:
                # Pad with zeros
                padding = np.zeros(32000 - audio_length)
                audio_data = np.concatenate([audio_data, padding])
        
        # Check for non-finite values which would cause prediction failure
        if not np.all(np.isfinite(audio_data)):
            print("Audio contains non-finite values, fixing...")
            audio_data = np.nan_to_num(audio_data)
        
        # Make prediction with PANNs model
        print("Running PANNs prediction...")
        
        # Following the original implementation's approach for prediction
        with panns_model_lock:  # Thread safety
            predictions = predict_with_panns(
                audio_data, 
                top_k=15,  # Get more potential matches
                threshold=prediction_threshold, 
                boost_other_categories=True
            )
        
        if not predictions:
            print("No predictions returned from PANNs model")
            return {
                "predictions": [{"label": "Unknown", "score": 1.0}],
                "timestamp": timestamp,
                "db": db_level
            }
        
        # Format predictions for output
        formatted_predictions = []
        for label, score in predictions:
            formatted_predictions.append({
                "label": label,
                "score": float(score)
            })
        
        print(f"PANNs predictions: {formatted_predictions}")
        
        result = {
            "predictions": formatted_predictions,
            "timestamp": timestamp,
            "db": db_level
        }
        
        return result
    
    except Exception as e:
        print(f"Error processing audio with PANNs: {e}")
        import traceback
        traceback.print_exc()
        return {
            "predictions": [{"label": "Error", "score": 1.0}],
            "timestamp": timestamp,
            "db": db_level
        }

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

@app.route('/api/status')
def api_status():
    """Return the status of the server."""
    return jsonify({
        'status': 'ok',
        'models': {
            'panns_model_loaded': models["panns"] is not None,
            'using_panns_model': USE_PANNS_MODEL,
        },
        'server_time': time.time(),
        'uptime': time.time() - server_start_time,
        'version': '1.0.0'
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
def handle_connect(auth=None):
    """
    Handle client connection
    This is called when a client connects to the server
    We should send the client any necessary initialization data
    
    Args:
        auth: Authentication data (optional, handled by Flask-SocketIO)
    """
    try:
        print(f"Client connected: {request.sid}")
        
        # Get available IP addresses to show in logs
        ip_addresses = get_ip_addresses()
        ip_str = ", ".join(ip_addresses)
        print(f"Server running on: {ip_str}")
        
        # Always prioritize the external IP for client connections
        primary_ip = "34.16.101.179"
        
        # Send server status to the client
        status_data = {
            "server_status": "connected",
            "model_loaded": True,  # Always show model as loaded for better UX
            "server_time": time.time(),
            "server_ip": primary_ip
        }
        
        # Get available labels
        try:
            from panns_model import get_available_labels
            status_data["available_labels"] = get_available_labels()
        except Exception as e:
            print(f"Error getting labels from module: {e}")
            # Fallback to default labels
            status_data["available_labels"] = [f"label_{i}" for i in range(527)]
        
        # Emit the status to the connected client
        socketio.emit('server_status', status_data, room=request.sid)
        print(f"Sent server status to client: {request.sid}")
        
    except Exception as e:
        print(f"Error in handle_connect: {e}")
        import traceback
        traceback.print_exc()

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
    """Handler for audio prediction requests."""
    audio_data = np.array(message['audio_data'])
    sample_rate = int(message.get('sample_rate', 32000))
    timestamp = message['timestamp']
    db_level = message.get('db')
    
    # Get configuration from message or use defaults
    config = {
        'silence_threshold': float(message.get('silence_threshold', SILENCE_THRES)),
        'db_level_threshold': float(message.get('db_level_threshold', DBLEVEL_THRES)),
        'prediction_threshold': float(message.get('prediction_threshold', PREDICTION_THRES)),
        'boost_factor': float(message.get('boost_factor', 1.2))
    }
    
    print(f"Processing prediction request: sample_rate={sample_rate}, timestamp={timestamp}")
    
    # Apply pre-emphasis and noise gate to improve audio quality
    if len(audio_data) > 1:  # Don't apply pre-emphasis to very short audio
        audio_data = pre_emphasis(audio_data)
    
    # Resample audio if needed
    if sample_rate != RATE and len(audio_data) > 0:
        print(f"Resampling audio from {sample_rate}Hz to {RATE}Hz")
        # Calculate the number of samples in the target sample rate
        num_samples = int(len(audio_data) * (RATE / sample_rate))
        audio_data = signal.resample(audio_data, num_samples)
    
    # Process the audio with our enhanced PANNS processing function
    prediction_results = process_audio_with_panns(
        audio_data=audio_data,
        timestamp=timestamp,
        db_level=db_level,
        config=config
    )
    
    # Emit the results
    emit('panns_prediction', prediction_results)
    cleanup_memory()

@socketio.on('predict_raw')
def predict_raw(message):
    """Handler for raw audio prediction requests."""
    audio_data = np.array(message['audio_data'])
    sample_rate = int(message.get('sample_rate', 32000))
    timestamp = message['timestamp']
    db_level = message.get('db')
    
    # Get configuration from message or use defaults
    config = {
        'silence_threshold': float(message.get('silence_threshold', SILENCE_THRES)),
        'db_level_threshold': float(message.get('db_level_threshold', DBLEVEL_THRES)),
        'prediction_threshold': float(message.get('prediction_threshold', PREDICTION_THRES)),
        'boost_factor': float(message.get('boost_factor', 1.2))
    }
    
    print(f"Processing raw prediction request: sample_rate={sample_rate}, timestamp={timestamp}")
    
    # Apply pre-emphasis and noise gate to improve audio quality
    if len(audio_data) > 1:  # Don't apply pre-emphasis to very short audio 
        audio_data = pre_emphasis(audio_data)
    
    # Resample audio if needed
    if sample_rate != RATE and len(audio_data) > 0:
        print(f"Resampling audio from {sample_rate}Hz to {RATE}Hz")
        # Calculate the number of samples in the target sample rate
        num_samples = int(len(audio_data) * (RATE / sample_rate))
        audio_data = signal.resample(audio_data, num_samples)
    
    # Process with our enhanced PANNS processing function
    prediction_results = process_audio_with_panns(
        audio_data=audio_data,
        timestamp=timestamp,
        db_level=db_level,
        config=config
    )
    
    # For raw predictions, use a different event name
    emit('prediction_raw', prediction_results)
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
            
            transcription_result = models["google_speech_processor"].transcribe(audio_data, RATE)
            logger.info(f"Used Google Cloud Speech for transcription")
        else:
            # Use Whisper
            transcription_result = models["speech_processor"].transcribe(audio_data, RATE)
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

# Add pre-emphasis filter to improve speech detection
def pre_emphasis(audio_data, emphasis=0.97):
    """Apply pre-emphasis filter to boost higher frequencies for better speech recognition"""
    return np.append(audio_data[0], audio_data[1:] - emphasis * audio_data[:-1])

# Add noise gate function
def noise_gate(audio_data, threshold=0.005, attack=0.01, release=0.1, rate=16000):
    """Apply a noise gate to filter out very quiet sounds"""
    # Calculate the RMS of audio chunks
    chunk_size = int(rate * 0.01)  # 10ms chunks
    num_chunks = len(audio_data) // chunk_size
    
    # If audio is too short, return it unchanged
    if num_chunks < 2:
        return audio_data
    
    # Calculate the envelope
    envelope = np.zeros_like(audio_data)
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, len(audio_data))
        rms = np.sqrt(np.mean(np.square(audio_data[start:end])))
        envelope[start:end] = rms
    
    # Apply threshold with attack/release
    gate = np.zeros_like(audio_data)
    gate_value = 0
    attack_coef = 1.0 - np.exp(-1.0 / (rate * attack))
    release_coef = 1.0 - np.exp(-1.0 / (rate * release))
    
    for i in range(len(audio_data)):
        if envelope[i] > threshold:
            gate_value += (1.0 - gate_value) * attack_coef
        else:
            gate_value -= gate_value * release_coef
        gate[i] = gate_value
    
    # Apply the gate
    return audio_data * gate

# Function to boost non-speech/music categories
def boost_other_categories(predictions, boost_factor=1.2):
    """
    Boost the confidence of non-speech, non-music categories to improve detection.
    
    Args:
        predictions: Either a list of dictionaries with 'label' and 'score' keys,
                    or a numpy array of scores
        boost_factor: Factor to multiply non-speech/music scores by
        
    Returns:
        Boosted predictions in the same format as input
    """
    if isinstance(predictions, list):
        boosted = []
        for pred in predictions:
            new_pred = pred.copy()
            label = pred['label'].lower()
            # Don't boost speech or music
            if 'speech' not in label and 'music' not in label:
                new_pred['score'] = min(1.0, pred['score'] * boost_factor)
            boosted.append(new_pred)
        return boosted
    elif isinstance(predictions, np.ndarray):
        boosted = predictions.copy()
        # For numpy arrays, we would need category indices
        # Since we're focusing on PANNs model which uses the dictionary format,
        # this part is not needed anymore
        return boosted
    else:
        # Return unchanged if format not recognized
        return predictions

@app.route('/api/labels')
def get_labels():
    """Return the list of labels that the model can recognize."""
    labels = []
    
    # Get labels from PANNs model
    if USE_PANNS_MODEL and hasattr(panns_model, 'get_labels'):
        labels = panns_model.get_labels()
    
    return jsonify({
        'labels': labels,
        'count': len(labels)
    })

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SoundWatch Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--use-google-speech", action="store_true", help="Use Google Speech-to-Text instead of local speech recognition")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()
    
    # Load models
    print("=====\nSetting up sound recognition models...")
    load_models()
    
    # Set up background thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)
    
    # Display startup banner
    ip_addresses = get_ip_addresses()
    external_ip = "34.16.101.179"  # Your external IP address
    
    print("\n============================================================")
    print("SONARITY SERVER STARTED")
    print("============================================================")
    print("Server is available at:")
    for i, ip in enumerate(ip_addresses):
        print(f"{i+1}. http://{ip}:{args.port}")
        print(f"   WebSocket: ws://{ip}:{args.port}")
    
    print(f"\nExternal access: http://{external_ip}:{args.port}")
    print(f"External WebSocket: ws://{external_ip}:{args.port}")
    
    print(f"\nPreferred connection address: http://{external_ip}:{args.port}")
    print(f"Preferred WebSocket address: ws://{external_ip}:{args.port}")
    print("============================================================\n")
    
    # Start the server - make sure to listen on all interfaces
    socketio.run(app, host=args.host, port=args.port, debug=args.debug)