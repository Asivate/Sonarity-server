# Import Statements
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

# Custom Imports
import panns_model
from sentiment_analyzer import analyze_sentiment
from speech_to_text import transcribe_audio, SpeechToText
from google_speech import transcribe_with_google, GoogleSpeechToText

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Terminal Color Definitions
class TermColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Logging Functions
def log_audio_receive(data_type, length, timestamp, db):
    """Format log for received audio data"""
    print(f"\n{TermColors.HEADER}{'='*80}{TermColors.ENDC}")
    print(f"{TermColors.BOLD}RECEIVED AUDIO DATA:{TermColors.ENDC} [{data_type}]")
    print(f"  {TermColors.BLUE}Length:{TermColors.ENDC} {length} samples")
    print(f"  {TermColors.BLUE}Time:{TermColors.ENDC} {timestamp}")
    print(f"  {TermColors.BLUE}dB Level:{TermColors.ENDC} {db:.2f}")

def log_audio_stats(stats):
    """Format log for audio statistics"""
    print(f"\n{TermColors.BOLD}AUDIO STATS:{TermColors.ENDC}")
    print(f"  {TermColors.CYAN}Min:{TermColors.ENDC} {stats['min']:.4f}")
    print(f"  {TermColors.CYAN}Max:{TermColors.ENDC} {stats['max']:.4f}")
    print(f"  {TermColors.CYAN}Mean:{TermColors.ENDC} {stats['mean']:.6f}")
    print(f"  {TermColors.CYAN}RMS:{TermColors.ENDC} {stats['rms']:.4f}")
    if stats.get('has_nan', False) or stats.get('has_inf', False):
        print(f"  {TermColors.RED}WARNING:{TermColors.ENDC} Contains NaN: {stats.get('has_nan', False)}, Inf: {stats.get('has_inf', False)}")

def log_prediction(prediction, db_level):
    """Format log for prediction results"""
    print(f"\n{TermColors.GREEN}PREDICTION RESULT:{TermColors.ENDC}")
    
    # Check if prediction is already a dictionary with 'predictions' key
    if isinstance(prediction, dict) and 'predictions' in prediction:
        predictions_list = prediction['predictions']
    # Check if prediction is a list of dictionaries with 'label' and 'score'
    elif isinstance(prediction, list) and all(isinstance(p, dict) and 'label' in p and 'score' in p for p in prediction):
        predictions_list = prediction
    # Check if prediction is a list of tuples (label, score)
    elif isinstance(prediction, list) and all(isinstance(p, tuple) and len(p) == 2 for p in prediction):
        # Convert tuples to dictionaries
        predictions_list = [{'label': p[0], 'score': p[1]} for p in prediction]
    else:
        # Fallback for unknown format
        print(f"  {TermColors.RED}WARNING:{TermColors.ENDC} Unknown prediction format: {prediction}")
        return
    
    # Now log each prediction
    for pred in predictions_list:
        if isinstance(pred, dict):
            label = pred.get('label', 'Unknown')
            score = float(pred.get('score', 0))
            print(f"  {TermColors.BOLD}{label}:{TermColors.ENDC} {score:.2f}")
        else:
            print(f"  {TermColors.BOLD}Unknown format:{TermColors.ENDC} {pred}")
    
    print(f"  {TermColors.BLUE}dB Level:{TermColors.ENDC} {db_level:.2f}")
    print(f"{TermColors.HEADER}{'-'*80}{TermColors.ENDC}\n")

def log_status(message, status="info"):
    """Log processing status messages with color coding"""
    if status == "info":
        print(f"{TermColors.BLUE}INFO:{TermColors.ENDC} {message}")
    elif status == "warning":
        print(f"{TermColors.YELLOW}WARNING:{TermColors.ENDC} {message}")
    elif status == "error":
        print(f"{TermColors.RED}ERROR:{TermColors.ENDC} {message}")
    elif status == "success":
        print(f"{TermColors.GREEN}SUCCESS:{TermColors.ENDC} {message}")

# Memory Optimization Settings
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

# Global Variables
prediction_counter = 0
USE_GOOGLE_SPEECH = False  # Set to True to use Google Cloud Speech-to-Text instead of Whisper

# Memory Cleanup Function
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

# Utility Functions
def get_ip_addresses():
    """Get available IP addresses for the server."""
    ip_list = []
    ip_list.append('34.16.101.179')  # External IP address for physical devices
    
    try:
        hostname = socket.gethostname()
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8', 80))
            primary_ip = s.getsockname()[0]
            if primary_ip != '127.0.0.1' and primary_ip != '0.0.0.0' and primary_ip != '34.16.101.179':
                ip_list.append(primary_ip)
        except Exception:
            pass
        finally:
            s.close()
            
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if ip not in ip_list and ip != '127.0.0.1' and ip != '0.0.0.0':
                ip_list.append(ip)
    except Exception as e:
        print(f"Error getting IP addresses: {e}")
    
    return ip_list

# Background Thread Function
def background_thread():
    """
    Background thread for handling periodic tasks and keeping the server alive.
    This is started by socketio and runs continuously in the background.
    """
    print("Background thread started")
    count = 0
    while True:
        socketio.sleep(10)  # Sleep for 10 seconds
        count += 1
        if count % 6 == 0:  # Every minute
            # Ping clients to keep connections alive
            socketio.emit('server_status', {'status': 'alive', 'timestamp': time.time()})
            print(f"Server heartbeat ping sent (iteration {count})")

# Flask and SocketIO Setup
async_mode = None
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode=async_mode, cors_allowed_origins="*")
thread = None
thread_lock = Lock()
panns_lock = Lock()
panns_model_lock = Lock()
prediction_lock = Lock()

# Audio Processing Constants
RATE = 32000
CHUNK = 1024
CHANNELS = 1
SILENCE_THRES = -60
DBLEVEL_THRES = 30
PREDICTION_THRES = 0.10
MINIMUM_AUDIO_LENGTH = 32000
SENTIMENT_THRES = 0.5

# Model Initialization
models = {}
sentiment_analyzer = None
speech_to_text = None
google_speech = None

try:
    sentiment_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    logger.info("Sentiment analysis model loaded successfully")
except Exception as e:
    logger.error(f"Error loading sentiment model: {str(e)}")
    sentiment_pipeline = None

EMOTION_TO_EMOJI = {
    "joy": "😄",
    "neutral": "😀",
    "surprise": "😮",
    "sadness": "😢",
    "fear": "😨",
    "anger": "😠",
    "disgust": "🤢"
}

EMOTION_GROUPS = {
    "Happy": ["joy", "love", "admiration", "approval", "caring", "excitement", "amusement", "gratitude", "optimism", "pride", "relief"],
    "Neutral": ["neutral", "realization", "curiosity"],
    "Surprised": ["surprise", "confusion", "nervousness"],
    "Unpleasant": ["sadness", "fear", "anger", "disgust", "disappointment", "embarrassment", "grief", "remorse", "annoyance", "disapproval"]
}

speech_processor = SpeechToText()
google_speech_processor = None

def load_models():
    """Load all required models for sound recognition and speech processing."""
    global models, USE_PANNS_MODEL
    
    models = {"panns": None}
    
    USE_PANNS_MODEL = os.environ.get('USE_PANNS_MODEL', '1') == '1'
    
    print(f"PANNs model {'enabled' if USE_PANNS_MODEL else 'disabled'} based on environment settings")
    
    if not USE_PANNS_MODEL:
        print("Warning: No sound recognition models enabled. Enabling PANNs model by default.")
        USE_PANNS_MODEL = True

    if USE_PANNS_MODEL:
        print("Loading PANNs model...")
        panns_model.load_panns_model()
        models["panns"] = True
        print("PANNs model loaded successfully")

# Audio Processing Functions
def audio_samples(in_data, frame_count, time_info, status_flags):
    """Process audio samples for real-time audio streaming."""
    try:
        np_wav = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        rms = np.sqrt(np.mean(np_wav**2))
        db = dbFS(rms)
        
        audio_stats = {
            'length': len(np_wav),
            'min': float(np.min(np_wav)),
            'max': float(np.max(np_wav)),
            'mean': float(np.mean(np_wav)),
            'rms': float(rms),
        }
        log_audio_stats(audio_stats)
        log_status(f"Live audio capture: {len(np_wav)} samples, {db:.2f} dB", "info")

        if db < SILENCE_THRES:
            log_status(f"Silence detected (dB: {db:.2f}, threshold: {SILENCE_THRES})", "info")
            return (in_data, 0)
        
        if db < DBLEVEL_THRES:
            log_status(f"Sound too quiet (dB: {db:.2f}, threshold: {DBLEVEL_THRES})", "info")
            return (in_data, 0)
        
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
        
        if prediction_results and "predictions" in prediction_results and len(prediction_results["predictions"]) > 0:
            top_pred = prediction_results["predictions"][0]
            print(f"DETECTED: {top_pred['label']} ({top_pred['score']:.4f}, db: {db:.2f})")
            
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

# SocketIO Event Handlers
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
        debug_data = json_data.copy() if isinstance(json_data, dict) else json_data
        if isinstance(debug_data, dict) and 'features' in debug_data:
            features_len = len(debug_data['features']) if debug_data['features'] else 0
            debug_data['features'] = f"[features array, length: {features_len}]"
        print(f"Received audio feature data: {debug_data}")
        
        if isinstance(json_data, str):
            try:
                json_data = json.loads(json_data)
                print("Successfully parsed JSON string data")
            except json.JSONDecodeError as e:
                print(f"Invalid JSON in audio_feature_data: {e}")
                return
        
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
        
        db = json_data.get('db')
        timestamp = json_data.get('time', json_data.get('timestamp', time.time()))
        
        if not audio_features or len(audio_features) == 0:
            print("Empty audio features received")
            socketio.emit('prediction', {
                "predictions": [{"label": "No Audio", "score": 1.0}],
                "timestamp": timestamp,
                "db": db
            })
            return
                    
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
        
        if db is None:
            rms = np.sqrt(np.mean(np.square(audio_features)))
            db = dbFS(rms)
            print(f"Calculated dB level: {db}")
        
        if db < SILENCE_THRES:
            print(f"Sound is silence (dB: {db})")
            socketio.emit('prediction', {
                "predictions": [{"label": "Silence", "score": 0.95}],
                "timestamp": timestamp,
                "db": db
            })
            return
                        
        if db < DBLEVEL_THRES:
            print(f"Sound too quiet (dB: {db}, threshold: {DBLEVEL_THRES})")
            socketio.emit('prediction', {
                "predictions": [{"label": "Too Quiet", "score": 0.9}],
                "timestamp": timestamp,
                "db": db
            })
            return
        
        print(f"Processing audio features (db: {db}, features shape: {audio_features.shape})")
        
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
        
        print(f"Emitting prediction: {result}")
        socketio.emit('prediction', result, broadcast=True)
        
    except Exception as e:
        print(f"Error in handle_source: {e}")
        import traceback
        traceback.print_exc()
        
        socketio.emit('prediction', {
            "predictions": [{"label": "Error", "score": 1.0}],
            "timestamp": time.time(),
            "db": None
        })

@socketio.on('audio_data')
def handle_audio(data):
    """
    Handle audio data sent from the client
    This handler now expects audio data with 32000 samples (1 second at 32kHz)
    directly from the client, which eliminates the need for server-side buffering.
    """
    # Initialize variables outside try block to ensure they're available in exception handler
    audio_field = None
    audio_data = None
    timestamp = time.time() * 1000  # Default to current time in ms
    db_level = None
    audio_format = 'float32'  # Default format
    
    try:
        # Start timing for performance measurement
        start_time = time.time()
        
        log_status("RECEIVED AUDIO DATA", "info")
        
        # Identify which field contains the audio data
        if isinstance(data, dict):
            # Check various possible field names
            if 'audio' in data and data['audio']:
                audio_field = 'audio'
                audio_data = data['audio']
                log_status(f"Using 'audio' field", "info")
            elif 'audioData' in data and data['audioData']:
                audio_field = 'audioData'
                audio_data = data['audioData']
                log_status(f"Using 'audioData' field", "info")
            elif 'data' in data and data['data']:
                audio_field = 'data'
                audio_data = data['data']
                log_status(f"Using 'data' field instead of 'audio'", "info")
            
            # Get other metadata if available
            if 'format' in data:
                audio_format = data['format']
            if 'db' in data:
                db_level = data['db']
            if 'timestamp' in data:
                timestamp = data['timestamp']
            elif 'time' in data:
                timestamp = data['time']
        
        # Handle case where no valid audio data was found
        if audio_data is None:
            log_status("No valid audio data found in message", "error")
            return
    
        # Log the received data info
        log_status(f"Length: {len(audio_data)} samples", "info")
        log_status(f"Time: {timestamp}", "info")
        if db_level:
            log_status(f"dB Level: {db_level:.2f}", "info")
        
        # Process the audio data based on its type
        if isinstance(audio_data, list):
            log_status(f"Processing audio as list of values, length: {len(audio_data)}", "info")
            
            # Convert the list to a numpy array
            audio_data = np.array(audio_data, dtype=np.float32)
            
            # If these are int16 values, normalize them
            if audio_format == 'int16' or (np.max(np.abs(audio_data)) > 1.0 and np.max(np.abs(audio_data)) <= 32768):
                audio_data = audio_data.astype(np.float32) / 32768.0
                log_status("Normalized integer audio values to float range [-1.0, 1.0]", "info")
                
        elif isinstance(audio_data, str):
            # Assume it's base64 encoded
            try:
                log_status("Processing audio as base64 encoded data", "info")
                audio_bytes = base64.b64decode(audio_data)
                
                if audio_format == 'int16':
                    # Convert bytes to int16 array
                    audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    # Assume float32
                    audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
            except Exception as e:
                log_status(f"Error decoding base64 audio: {str(e)}", "error")
                return
        
        # Calculate dB level if not provided
        if db_level is None:
            rms = np.sqrt(np.mean(np.square(audio_data)))
            db_level = dbFS(rms)
        
        # Log audio statistics for debugging
        print("\nAUDIO STATS:")
        # Create a stats dictionary for log_audio_stats
        audio_stats = {
            'length': len(audio_data),
            'min': float(np.min(audio_data)),
            'max': float(np.max(audio_data)),
            'mean': float(np.mean(audio_data)),
            'rms': float(np.sqrt(np.mean(audio_data**2)))
        }
        log_audio_stats(audio_stats)
        
        # Process the audio with our model directly without buffering 
        # (since the client now sends the correct size)
        process_audio_with_panns(audio_data, db_level, timestamp)
        
    except Exception as e:
        log_status(f"Error in handle_audio: {str(e)}", "error")
        import traceback
        traceback.print_exc()
        
        # Return an error prediction with safely initialized variables
        emit_prediction([("Error", 1.0)], db_level, timestamp)

# Remaining functions and code
def process_with_panns_model(np_wav, record_time=None, db=None):
    """Process audio using PANNs model"""
    try:
        print("Processing with PANNs model...")
        print(f"Audio shape: {np_wav.shape}, min: {np_wav.min():.6f}, max: {np_wav.max():.6f}, mean: {np_wav.mean():.6f}")
        
        if np.abs(np_wav).max() > 1.0:
            print(f"Warning: Audio data exceeds normalized range [-1.0, 1.0]. Max value: {np.abs(np_wav).max():.6f}")
            np_wav = np_wav / np.abs(np_wav).max()
            print(f"Audio normalized. New range: [{np_wav.min():.6f}, {np_wav.max():.6f}]")
        
        np_wav = noise_gate(np_wav, threshold=0.005, attack=0.01, release=0.1, rate=RATE)
            
        if len(np_wav) < MINIMUM_AUDIO_LENGTH:
            print(f"Audio too short ({len(np_wav)} samples). Padding to {MINIMUM_AUDIO_LENGTH} samples.")
            if len(np_wav) < MINIMUM_AUDIO_LENGTH / 4:
                repeats = int(np.ceil(MINIMUM_AUDIO_LENGTH / len(np_wav)))
                padded = np.tile(np_wav, repeats)[:MINIMUM_AUDIO_LENGTH]
                print(f"Using repetition padding ({repeats} repeats)")
            else:
                padded = np.zeros(MINIMUM_AUDIO_LENGTH)
                padded[:len(np_wav)] = np_wav
            np_wav = padded
        
        with panns_lock:
            panns_results = panns_model.predict_with_panns(
                np_wav, 
                top_k=10, 
                threshold=PREDICTION_THRES,
                map_to_homesounds_format=False,
                boost_other_categories=True
            )
        
        if panns_results and "output" in panns_results and len(panns_results["output"]) > 0:
            print("===== PANNs MODEL PREDICTIONS =====")
            for pred in panns_results["output"][:5]:
                print(f"  {pred['label']}: {pred['score']:.6f}")
            
            top_prediction = panns_results["output"][0]
            top_label = top_prediction["label"]
            top_score = float(top_prediction["score"])
            
            print(f"Top prediction: {top_label} ({top_score:.4f})")
            
            if top_label.lower() == "speech" and top_score > SPEECH_DETECTION_THRES:
                print(f"Speech detected with PANNs model. Processing sentiment...")
                if os.environ.get('USE_SPEECH', '0') == '1' and os.environ.get('USE_SENTIMENT', '0') == '1':
                    process_speech(np_wav, record_time, top_score)
                else:
                    socketio.emit('audio_label', {
                        'label': top_label,
                        'accuracy': str(top_score),
                        'db': str(db),
                        'timestamp': record_time
                    })
                    print(f"EMITTING: {top_label} ({top_score:.2f})")
            else:
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

# Audio buffer for accumulating enough samples for PANNS model
panns_audio_buffer = []

def process_audio_with_panns(audio_data, db_level=None, timestamp=None, config=None):
    """
    Process audio with PANNs model for sound recognition.
    This function expects audio data with 32000 samples (1 second at 32kHz).
    
    Args:
        audio_data: numpy array of audio samples
        db_level: pre-calculated dB level or None to calculate
        timestamp: timestamp of the audio data
        config: dictionary with configuration parameters
        
    Returns:
        None - results are emitted via socketio
    """
    log_status("Processing audio with PANNs model", "info")
    
    if config is None:
        config = {
            'silence_threshold': SILENCE_THRES,
            'db_level_threshold': DBLEVEL_THRES,
            'prediction_threshold': PREDICTION_THRES,
            'boost_factor': 1.2
        }
    
    silence_threshold = config.get('silence_threshold', SILENCE_THRES)
    db_level_threshold = config.get('db_level_threshold', DBLEVEL_THRES)
    prediction_threshold = config.get('prediction_threshold', PREDICTION_THRES)
    
    try:
        # Convert audio_data to numpy array if it's not already
        if not isinstance(audio_data, np.ndarray):
            try:
                audio_data = np.array(audio_data, dtype=np.float32)
                log_status("Converted audio to numpy array", "info")
            except Exception as e:
                log_status(f"Error converting audio_data to numpy array: {e}", "error")
                emit_prediction([("Invalid Audio Format", 1.0)], db_level, timestamp)
                return
        
        # Ensure we have float32 data
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Log detailed audio statistics
        audio_stats = {
            'length': len(audio_data),
            'min': float(np.min(audio_data)),
            'max': float(np.max(audio_data)),
            'mean': float(np.mean(audio_data)),
            'rms': float(np.sqrt(np.mean(audio_data**2)))
        }
        log_audio_stats(audio_stats)
        
        # Fix NaN or Inf values if present
        if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
            log_status("Audio contains NaN or Inf values, fixing...", "warning")
            audio_data = np.nan_to_num(audio_data)
        
        # Calculate dB level if not provided
        if db_level is None:
            rms = np.sqrt(np.mean(np.square(audio_data)))
            db_level = dbFS(rms)
            log_status(f"Calculated dB level: {db_level:.2f}", "info")
        
        # Check for silence
        if db_level < silence_threshold:
            log_status(f"Sound is silence (dB: {db_level:.2f}, threshold: {silence_threshold})", "info")
            emit_prediction([("Silence", 0.95)], db_level, timestamp)
            return
            
        # Check if sound is too quiet
        if db_level < db_level_threshold:
            log_status(f"Sound too quiet (dB: {db_level:.2f}, threshold: {db_level_threshold})", "info")
            emit_prediction([("Too Quiet", 0.90)], db_level, timestamp)
            return
        
        # Sound level is good, proceed with PANNs model
        log_status(f"Sound level ({db_level:.2f} dB) within processing range", "success")
        
        # Handle audio buffer size - PANNs expects 32000 samples (1 second @ 32kHz)
        audio_length = len(audio_data)
        if audio_length < MINIMUM_AUDIO_LENGTH:
            # Pad with zeros if too short
            log_status(f"Padding short audio: {audio_length} → {MINIMUM_AUDIO_LENGTH} samples", "info")
            padding = np.zeros(MINIMUM_AUDIO_LENGTH - audio_length, dtype=np.float32)
            audio_data = np.concatenate([audio_data, padding])
        elif audio_length > MINIMUM_AUDIO_LENGTH:
            # Trim if too long - use most recent samples
            log_status(f"Trimming long audio: {audio_length} → {MINIMUM_AUDIO_LENGTH} samples", "info")
            audio_data = audio_data[-MINIMUM_AUDIO_LENGTH:]
        
        # Normalize audio to [-1.0, 1.0] range if needed
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
            log_status("Normalized audio to [-1.0, 1.0] range", "info")
        
        # Add preprocessing step to detect knocking sounds in the time domain
        # This can be more reliable than spectral analysis for percussive sounds
        has_knocking_pattern = False
        try:
            from scipy.signal import find_peaks
            
            # Get the envelope of the audio signal
            envelope = np.abs(audio_data)
            
            # Smooth the envelope to reduce noise
            window_size = 320  # About 10ms at 32kHz
            smoothed = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
            
            # Find peaks in the envelope
            peaks, peak_props = find_peaks(
                smoothed, 
                height=0.05*np.max(smoothed),  # Low threshold to detect softer knocks
                distance=500,  # Minimum distance between peaks
                prominence=0.1*np.max(smoothed)  # Ensure peaks stand out
            )
            
            # Calculate peak decay rate (important for percussive sounds)
            peak_decay = 0.0
            if len(peaks) > 0:
                peak_heights = smoothed[peaks]
                if len(peaks) > 1:
                    peak_spacing = np.diff(peaks) / 32000  # Convert to seconds
                    avg_spacing = np.mean(peak_spacing)
                    
                    # Calculate consistency of spacing
                    std_spacing = np.std(peak_spacing)
                    spacing_consistency = std_spacing / avg_spacing if avg_spacing > 0 else 999
                    
                    # Calculate decay rate (how quickly sound fades after peaks)
                    if len(peaks) >= 2:
                        decay_windows = []
                        for i in range(len(peaks)-1):
                            start_idx = peaks[i]
                            end_idx = min(peaks[i+1], start_idx + 3200)  # Look at 100ms after peak
                            if start_idx < end_idx:
                                segment = smoothed[start_idx:end_idx]
                                if len(segment) > 0 and segment[0] > 0:
                                    decay_rate = 1.0 - (segment[-1] / segment[0])
                                    decay_windows.append(decay_rate)
                        
                        peak_decay = np.mean(decay_windows) if decay_windows else 0.0
                    
                    print(f"Percussion details: peaks={len(peaks)}, spacing={avg_spacing:.2f}s, consistency={spacing_consistency:.2f}, decay={peak_decay:.2f}")
                    
                    # Check if this looks like knocking
                    if ((0.03 < avg_spacing < 0.5) and spacing_consistency < 0.6 and peak_decay > 0.4) or \
                       ((0.03 < avg_spacing < 0.2) and len(peaks) >= 3) or \
                       (len(peaks) >= 2 and peak_decay > 0.7):
                        has_knocking_pattern = True
                        print("PERCUSSION DETECTED: Strong evidence of knock/tap sound in time domain analysis")
                elif len(peaks) == 1:
                    # Single peak analysis
                    start_idx = peaks[0]
                    end_idx = min(start_idx + 3200, len(smoothed) - 1)
                    if start_idx < end_idx:
                        segment = smoothed[start_idx:end_idx]
                        if len(segment) > 0 and segment[0] > 0:
                            peak_decay = 1.0 - (segment[-1] / segment[0])
                    
                    print(f"Single peak percussion analysis: decay={peak_decay:.2f}, height={peak_heights[0]}")
                    if peak_decay > 0.8 or peak_heights[0] > 0.3:
                        has_knocking_pattern = True
                        print("PERCUSSION DETECTED: Strong evidence of knock/tap sound in time domain analysis")
                
            if has_knocking_pattern:
                # Search for available percussion-related labels
                available_labels = []
                try:
                    available_labels = panns_model.get_labels()
                    if available_labels:
                        knock_labels = [(i, label) for i, label in enumerate(available_labels) 
                                     if any(keyword in label.lower() for keyword in 
                                           ['knock', 'tap', 'thump', 'bang', 'drum', 'percussion'])]
                        print(f"Total available labels: {len(available_labels)}")
                        print(f"Found {len(knock_labels)} knock-related labels: {knock_labels}")
                except Exception as e:
                    print(f"Error getting available labels: {e}")
        except Exception as e:
            print(f"Error in time domain knocking detection: {e}")
        
        # Run prediction with PANNs model
        log_status(f"Running PANNs prediction with threshold: {prediction_threshold:.2f}", "info")
        
        try:
            with panns_model_lock:
                # Get detailed stats about the audio for debugging
                audio_std = np.std(audio_data)
                audio_abs_max = np.max(np.abs(audio_data))
                print(f"Audio stats - Mean: {np.mean(audio_data):.6f}, Std: {audio_std:.6f}, Min: {np.min(audio_data):.6f}, Max: {np.max(audio_data):.6f}, Abs Max: {audio_abs_max:.6f}")
                
                # Lower the threshold for percussion sounds to improve detection
                effective_threshold = prediction_threshold * 0.5 if has_knocking_pattern else prediction_threshold
                
                # Get raw predictions from the PANNs model
                predictions = panns_model.predict_with_panns(
                    audio_data, 
                    top_k=20,  # Get more predictions to have a wider selection
                    threshold=effective_threshold,
                    map_to_homesounds_format=False  # Use raw AudioSet labels
                )
                
                # Log the raw model output
                if predictions and isinstance(predictions, list):
                    # Already have a list of tuples (label, score)
                    predictions_list = predictions
                    print(f"PANNs prediction results: {predictions_list[:5]}")
                    
                    # Check if there are any percussion-related labels
                    percussion_labels = [pred for pred in predictions_list 
                                       if any(keyword in pred[0].lower() for keyword in 
                                           ['knock', 'tap', 'thump', 'bang', 'drum', 'percussion'])]
                    
                    if has_knocking_pattern and not percussion_labels:
                        print("Percussive sound detected, using lower threshold for percussion sounds")
                        # Try again with much lower threshold to catch percussion sounds
                        retry_predictions = panns_model.predict_with_panns(
                            audio_data, 
                            top_k=40,  # Try more predictions
                            threshold=0.01,  # Much lower threshold
                            map_to_homesounds_format=False
                        )
                        
                        if retry_predictions and isinstance(retry_predictions, list):
                            retry_list = retry_predictions
                            print(f"Preliminary predictions with low threshold: {retry_list[:5]}")
                            
                            # Look for percussion labels in retried predictions
                            percussion_labels = [pred for pred in retry_list 
                                             if any(keyword in pred[0].lower() for keyword in 
                                                ['knock', 'tap', 'thump', 'bang', 'drum', 'percussion'])]
                    
                    # MODIFIED: Use the highest confidence predictions, regardless of whether they're percussion
                    # Still note when percussion is detected, but don't force it to be the only prediction
                    final_predictions = []
                    
                    # Filter out low confidence predictions
                    final_predictions = [pred for pred in predictions_list if pred[1] > prediction_threshold]
                    
                    # If we have strong time-domain evidence of knocking and found percussion labels
                    # add the highest confidence percussion label if it's not already in the list
                    if has_knocking_pattern and percussion_labels:
                        percussion_labels.sort(key=lambda x: x[1], reverse=True)
                        best_percussion = percussion_labels[0]
                        print(f"Using percussion label from model: {best_percussion}")
                        
                        # Only add the percussion label if it's not already in the list
                        if best_percussion not in final_predictions:
                            final_predictions.append(best_percussion)
                            
                    # If everything is low confidence but we have some predictions
                    if not final_predictions and predictions_list:
                        # Take the highest confidence prediction regardless of threshold
                        final_predictions.append(predictions_list[0])
                    
                    print(f"Final predictions: {final_predictions}")
                    emit_prediction(final_predictions, db_level, timestamp)
                else:
                    log_status("No predictions returned from model", "warning")
                    emit_prediction([("No Sound Detected", 0.8)], db_level, timestamp)
                
        except Exception as e:
            log_status(f"Error in PANNs prediction: {str(e)}", "error")
            traceback.print_exc()
            emit_prediction([("Prediction Error", 1.0)], db_level, timestamp)
            
    except Exception as e:
        log_status(f"Error processing audio: {str(e)}", "error")
        traceback.print_exc()
        emit_prediction([("Processing Error", 1.0)], db_level, timestamp)

def emit_prediction(predictions, db_level, timestamp=None):
    """
    Emit prediction results via socketio.
    
    Args:
        predictions: List of predictions, either as tuples of (label, score) 
                    or dictionaries with 'label' and 'score' keys
        db_level: Audio decibel level or None
        timestamp: Timestamp of the audio data or None
        
    Returns:
        None
    """
    # Set default values if None is provided
    if db_level is None:
        db_level = -100  # Default low dB level
    
    if timestamp is None:
        timestamp = time.time() * 1000  # Current time in milliseconds
    
    # Print detailed debugging information
    print(f"EMITTING PREDICTION: {predictions}")
    print(f"  dB Level: {db_level}")
    print(f"  Timestamp: {timestamp}")
    
    # Format the predictions for emission
    formatted_predictions = []
    
    # Check if we have percussion/knock sounds
    has_percussion = False
    for pred in predictions:
        pred_label = ""
        if isinstance(pred, tuple) and len(pred) == 2:
            pred_label = pred[0].lower()
        elif isinstance(pred, dict) and 'label' in pred:
            pred_label = pred['label'].lower()
            
        # Check for percussion keywords
        if any(keyword in pred_label for keyword in 
              ['knock', 'tap', 'thump', 'bang', 'drum', 'percussion']):
            has_percussion = True
            print(f"PERCUSSION SOUND DETECTED in predictions: {pred}")
    
    # Handle predictions based on the format
    for pred in predictions:
        if isinstance(pred, tuple) and len(pred) == 2:
            # Handle tuple format (label, score)
            label, score = pred
            formatted_predictions.append({
                'label': label,
                'score': str(round(float(score), 4))
            })
        elif isinstance(pred, dict) and 'label' in pred and 'score' in pred:
            # Handle dictionary format
            formatted_predictions.append({
                'label': pred['label'],
                'score': str(round(float(pred['score']), 4))
            })
        else:
            print(f"Warning: Unrecognized prediction format: {pred}")
    
    # Ensure we have at least one prediction
    if not formatted_predictions:
        print("No valid predictions found, using default")
        formatted_predictions.append({
            'label': 'Unknown',
            'score': '0.5'
        })
    
    # Create the message to emit
    message = {
        'predictions': formatted_predictions,
        'db': str(round(float(db_level), 2)) if db_level is not None else '-100',
        'timestamp': timestamp
    }
    
    # For backward compatibility with clients expecting a single prediction
    # Use the highest confidence prediction as the main label
    sorted_preds = sorted(formatted_predictions, key=lambda x: float(x['score']), reverse=True)
    if sorted_preds:
        message['label'] = sorted_preds[0]['label']
        message['accuracy'] = sorted_preds[0]['score']
    
    # Log prediction and emit to clients
    log_prediction(formatted_predictions, db_level)
    socketio.emit('prediction', message)

recent_audio_buffer = []
MAX_BUFFER_SIZE = 5

def process_speech_with_sentiment(audio_data, record_time=None, confidence=0.8):
    """Process speech audio, transcribe it and analyze sentiment."""
    SPEECH_MAX_BUFFER_SIZE = 5
    MIN_WORD_COUNT = 3
    MIN_CONFIDENCE = 0.7
    
    # Set default values if not provided
    if record_time is None:
        record_time = time.time() * 1000
    
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
    print("Transcribing with enhanced audio processing...")
        
    try:
        transcription_result = None
        
        if os.environ.get('USE_GOOGLE_SPEECH', '0') == '1':
            if models.get("google_speech_processor", None) is None:
                from google_speech import GoogleSpeechToText
                models["google_speech_processor"] = GoogleSpeechToText()
            
            transcription_result = models["google_speech_processor"].transcribe(concatenated_audio, RATE)
            logger.info(f"Used Google Cloud Speech for transcription")
        else:
            if models.get("speech_processor", None) is None:
                from speech_to_text import SpeechToText
                models["speech_processor"] = SpeechToText()
                
            transcription_result = models["speech_processor"].transcribe(concatenated_audio, RATE)
            logger.info(f"Used Whisper for transcription")
        
        if not transcription_result or not transcription_result.get('text'):
            logger.info(f"No valid transcription found")
            socketio.emit('audio_label', {
                'label': 'Speech',
                'accuracy': str(confidence),
                'db': '-30',
                'timestamp': record_time
            })
            return
        
        text = transcription_result['text']
        print(f"Transcribed: '{text}'")
        
        if os.environ.get('USE_SENTIMENT', '0') == '1':
            sentiment_result = analyze_sentiment(text)
            if sentiment_result:
                category = sentiment_result.get('category', 'Neutral')
                emoji = sentiment_result.get('emoji', '😐')
                emotion = sentiment_result.get('original_emotion', 'neutral')
                sentiment_score = sentiment_result.get('confidence', 0.5)
                
                label = f"Speech {category}"
                
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
        socketio.emit('audio_label', {
            'label': 'Speech',
            'accuracy': str(confidence),
            'db': '-30',
            'timestamp': record_time
        })
        print("Emitting basic speech (error in processing)")
    finally:
        cleanup_memory()

# Audio Enhancement Functions
def pre_emphasis(audio_data, emphasis=0.97):
    """Apply pre-emphasis filter to boost higher frequencies for better speech recognition"""
    return np.append(audio_data[0], audio_data[1:] - emphasis * audio_data[:-1])

def noise_gate(audio_data, threshold=0.005, attack=0.01, release=0.1, rate=16000):
    """Apply a noise gate to filter out very quiet sounds"""
    chunk_size = int(rate * 0.01)
    num_chunks = len(audio_data) // chunk_size
    
    if num_chunks < 2:
        return audio_data
    
    envelope = np.zeros_like(audio_data)
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, len(audio_data))
        rms = np.sqrt(np.mean(np.square(audio_data[start:end])))
        envelope[start:end] = rms
    
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
    
    return audio_data * gate

def boost_other_categories(predictions, boost_factor=1.2):
    """
    Boost the confidence of non-speech, non-music categories to improve detection.
    """
    if isinstance(predictions, list):
        boosted = []
        for pred in predictions:
            new_pred = pred.copy()
            label = pred['label'].lower()
            if 'speech' not in label and 'music' not in label:
                new_pred['score'] = min(1.0, pred['score'] * boost_factor)
            boosted.append(new_pred)
        return boosted
    elif isinstance(predictions, np.ndarray):
        boosted = predictions.copy()
        return boosted
    else:
        return predictions

@app.route('/api/labels')
def get_labels():
    """Return the list of labels that the model can recognize."""
    labels = []
    
    if USE_PANNS_MODEL and hasattr(panns_model, 'get_labels'):
        labels = panns_model.get_labels()
    
    return jsonify({
        'labels': labels,
        'count': len(labels)
    })

def calculate_db(audio_data):
    """
    Calculate the decibel level of audio data.
    
    Args:
        audio_data: numpy array of audio samples
        
    Returns:
        float: Decibel level (dBFS)
    """
    if audio_data is None or len(audio_data) == 0:
        return -100  # Default low value for empty audio
        
    try:
        rms = np.sqrt(np.mean(audio_data**2))
        return dbFS(rms)
    except Exception as e:
        log_status(f"Error calculating dB level: {str(e)}", "error")
        return -100  # Default low value on error

# Main Execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SoundWatch Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--use-google-speech", action="store_true", help="Use Google Speech-to-Text instead of local speech recognition")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()
    
    print("=====\nSetting up sound recognition models...")
    load_models()
    
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)
    
    ip_addresses = get_ip_addresses()
    external_ip = "34.16.101.179"
    
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
    
    socketio.run(app, host=args.host, port=args.port, debug=args.debug)