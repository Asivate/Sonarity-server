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
        
        # Get all IP addresses
        try:
            for ip in socket.gethostbyname_ex(hostname)[2]:
                if ip not in ip_list:
                    ip_list.append(ip)
        except:
            pass
    except:
        pass
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
RATE = 16000  # Audio sample rate
CHUNK = 1024  # Audio chunk size
CHANNELS = 1  # Mono audio
SILENCE_THRES = 30  # Silence threshold in dB
DBLEVEL_THRES = 30  # Decibel level threshold (changed from 60 to 30)
PREDICTION_THRES = 0.15  # Prediction confidence threshold
SPEECH_DETECTION_THRES = 0.3  # Threshold for speech detection
MINIMUM_AUDIO_LENGTH = 16000  # Minimum audio length (1 second at 16kHz)
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
    np_wav = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
    rms = np.sqrt(np.mean(np_wav**2))
    db = dbFS(rms)

    # Check for silence
    if -db < SILENCE_THRES:
        print(f"Silence detected (db: {db})")
        return (in_data, 0)
    
    # Check if sound is too quiet
    if -db <= DBLEVEL_THRES:
        print(f"Sound too quiet (db: {db})")
        return (in_data, 0)
    
    # Process with PANNs model
    try:
        # Use our process_audio_with_panns function
        prediction_results = process_audio_with_panns(
            audio_data=np_wav,
            db_level=db,
            config=None  # Use default config
        )
        
        # Log the top prediction
        if prediction_results and "predictions" in prediction_results and len(prediction_results["predictions"]) > 0:
            top_pred = prediction_results["predictions"][0]
            print(f"Top prediction: {top_pred['label']} ({top_pred['score']:.4f})")
    except Exception as e:
        print(f"Error processing audio with PANNs model: {e}")
        traceback.print_exc()
    
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
            # Process with PANNs model
            # Convert data to numpy array
            np_data = np.array(data, dtype=np.float32)
            print(f"Processing with PANNs model, data shape: {np_data.shape}")
            
            # Use our process_audio_with_panns function
            prediction_results = process_audio_with_panns(
                audio_data=np_data,
                timestamp=record_time,
                db_level=db,
                config=None  # Use default config
            )
            
            # Emit the results
            socketio.emit('panns_prediction', prediction_results)
            cleanup_memory()
        else:
            print(f"Sound level ({-db} dB) below threshold ({DBLEVEL_THRES} dB)")
            socketio.emit('audio_label', {
                'label': 'Too Quiet',
                'accuracy': '0.9',
                'db': str(db)
            })
    except Exception as e:
        print(f"Error in handle_source: {e}")
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
        # Process with PANNs model using our consistent function
        prediction_results = process_audio_with_panns(
            audio_data=np_wav,
            timestamp=record_time,
            db_level=db,
            config=None  # Use default config
        )
        
        # Emit the results
        socketio.emit('panns_prediction', prediction_results)
        cleanup_memory()
    else:
        print(f"Sound level ({-db} dB) below threshold ({DBLEVEL_THRES} dB)")
        socketio.emit('audio_label', {
            'label': 'Too Quiet',
            'accuracy': '0.9',
            'db': str(db)
        })

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
    Process audio with PANNS model using configurable thresholds.
    
    Args:
        audio_data: Audio waveform data as numpy array
        timestamp: Timestamp for the recording
        db_level: Loudness in decibels (optional)
        config: Dictionary with configuration parameters:
            - silence_threshold: Threshold for detecting silence (default: use SILENCE_THRES)
            - db_level_threshold: Threshold for audio loudness (default: use DBLEVEL_THRES)
            - prediction_threshold: Confidence threshold for predictions (default: use PREDICTION_THRES)
            - boost_factor: Boost factor for non-speech/music categories (default: 1.2)
            
    Returns:
        Dictionary with prediction results
    """
    # Set default config if not provided
    if config is None:
        config = {}
    
    # Extract configuration with defaults
    silence_threshold = config.get('silence_threshold', SILENCE_THRES)
    db_level_threshold = config.get('db_level_threshold', DBLEVEL_THRES)
    prediction_threshold = config.get('prediction_threshold', PREDICTION_THRES)
    boost_factor = config.get('boost_factor', 1.2)
    
    # Check if timestamp is provided
    if timestamp is None:
        timestamp = time.time()
    
    # Check audio and calculate dB if not provided
    if db_level is None and audio_data is not None:
        rms = np.sqrt(np.mean(audio_data**2))
        db_level = dbFS(rms)
        print(f"Calculated dB level: {db_level}")

    # Check for silence
    if db_level is not None and -db_level < silence_threshold:
        print(f"Sound is silence (dB: {db_level}, threshold: {silence_threshold})")
        return {
            "predictions": [{"label": "Silence", "score": 0.95}],
            "timestamp": timestamp,
            "db": db_level
        }
    
    # Check if sound is loud enough to process
    if db_level is not None and -db_level < db_level_threshold:
        print(f"Sound level ({-db_level} dB) below threshold ({db_level_threshold} dB), processing...")
    else:
        print(f"Sound too quiet (dB: {db_level}, threshold: {db_level_threshold})")
        return {
            "predictions": [{"label": "Too Quiet", "score": 0.9}],
            "timestamp": timestamp,
            "db": db_level
        }
    
    # Process with PANNS model
    try:
        print(f"Processing audio with PANNS model (threshold: {prediction_threshold})...")
        
        # Ensure audio is normalized properly
        if np.abs(audio_data).max() > 1.0:
            audio_data = audio_data / np.abs(audio_data).max()
            print("Audio normalized to range [-1.0, 1.0]")
        
        # Apply noise gate to reduce background noise
        audio_data = noise_gate(audio_data, threshold=0.005, attack=0.01, release=0.1, rate=RATE)
        
        # Ensure audio is long enough for processing
        if len(audio_data) < MINIMUM_AUDIO_LENGTH:
            print(f"Audio too short ({len(audio_data)} samples). Padding to {MINIMUM_AUDIO_LENGTH} samples.")
            if len(audio_data) < MINIMUM_AUDIO_LENGTH / 4:
                # For very short sounds, repeat them
                repeats = int(np.ceil(MINIMUM_AUDIO_LENGTH / len(audio_data)))
                padded = np.tile(audio_data, repeats)[:MINIMUM_AUDIO_LENGTH]
                print(f"Using repetition padding ({repeats} repeats)")
            else:
                padded = np.zeros(MINIMUM_AUDIO_LENGTH)
                padded[:len(audio_data)] = audio_data
            audio_data = padded
        
        # Get PANNS predictions
        with panns_lock:
            results = panns_model.predict_with_panns(
                audio_data, 
                top_k=10, 
                threshold=prediction_threshold,
                map_to_homesounds_format=True,
                boost_other_categories=(boost_factor > 1.0)
            )
        
        # Format results for client
        if results and "output" in results and len(results["output"]) > 0:
            # Display predictions
            print("===== PANNS MODEL PREDICTIONS =====")
            for pred in results["output"][:5]:
                print(f"  {pred['label']}: {pred['score']:.6f}")
            
            # Return formatted results
            return {
                "predictions": results["output"],
                "timestamp": timestamp,
                "db": db_level
            }
        else:
            print("No valid predictions from PANNS model")
            return {
                "predictions": [{"label": "Unrecognized Sound", "score": 0.2}],
                "timestamp": timestamp,
                "db": db_level
            }
            
    except Exception as e:
        print(f"Error processing audio with PANNS model: {e}")
        traceback.print_exc()
        return {
            "predictions": [{"label": "Error", "score": 0.1}],
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
    
    # Set primary_model to PANNs
    primary_model = "panns"
    
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