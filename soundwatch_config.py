"""
SoundWatch Configuration Module

This module contains all the configuration settings for the SoundWatch server,
including model paths, audio processing parameters, and server settings.

It provides consistent configuration across different modules and makes it
easy to change settings from a single place.
"""

import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("soundwatch_config")

# Base paths
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SERVER_DIR, "models")
ASSETS_PATH = os.path.join(SERVER_DIR, "assets")

# Ensure directories exist
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(ASSETS_PATH, exist_ok=True)

# Model filenames
PYTORCH_MODEL = "Cnn13_GMP_64x64_520000_iterations_mAP=0.42.pth"
ONNX_MODEL = "Cnn13_GMP_64x64_520000_iterations_mAP=0.42.onnx"
QUANTIZED_MODEL = "Cnn13_GMP_64x64_520000_iterations_mAP=0.42_quantized.onnx"
TENSORRT_ENGINE = "Cnn13_GMP_64x64_520000_iterations_mAP=0.42.engine"

# Full paths to model files
PYTORCH_MODEL_PATH = os.path.join(MODEL_PATH, PYTORCH_MODEL)
ONNX_MODEL_PATH = os.path.join(MODEL_PATH, ONNX_MODEL)
QUANTIZED_MODEL_PATH = os.path.join(MODEL_PATH, QUANTIZED_MODEL)
TENSORRT_ENGINE_PATH = os.path.join(MODEL_PATH, TENSORRT_ENGINE)

# Audio processing parameters
SAMPLE_RATE = 32000
CHUNK_SIZE = 1024
CHANNELS = 1
MINIMUM_AUDIO_LENGTH = 32000  # 1 second at 32kHz

# Mel spectrogram parameters
N_FFT = 1024
HOP_LENGTH = 320
MEL_BINS = 64
FMIN = 50
FMAX = 14000

# Detection thresholds
SILENCE_THRES = -60  # dB threshold for silence
DBLEVEL_THRES = 30   # dB threshold for quiet sounds
PREDICTION_THRES = 0.10  # Confidence threshold for predictions
SPEECH_DETECTION_THRES = 0.5  # Threshold for speech detection

# Memory optimization levels
MEMORY_OPTIMIZATION_LEVELS = {
    0: {"threads": 0, "cache_freq": 0, "description": "No optimization"},
    1: {"threads": 4, "cache_freq": 10, "description": "Moderate optimization"},
    2: {"threads": 2, "cache_freq": 5, "description": "Aggressive optimization"}
}

# Server settings
DEFAULT_PORT = 8080
DEFAULT_HOST = "0.0.0.0"
DEBUG_MODE = False

# Environment variable names
ENV_USE_PANNS_MODEL = "USE_PANNS_MODEL"
ENV_PANNS_MODEL_PATH = "PANNS_MODEL_PATH"
ENV_PANNS_MODEL_TYPE = "PANNS_MODEL_TYPE"
ENV_MEMORY_OPTIMIZATION = "MEMORY_OPTIMIZATION"
ENV_USE_SPEECH = "USE_SPEECH"
ENV_USE_SENTIMENT = "USE_SENTIMENT"
ENV_USE_GOOGLE_SPEECH = "USE_GOOGLE_SPEECH"

def get_memory_optimization_config():
    """Get memory optimization config based on environment variable"""
    level = int(os.environ.get(ENV_MEMORY_OPTIMIZATION, 1))
    config = MEMORY_OPTIMIZATION_LEVELS.get(level, MEMORY_OPTIMIZATION_LEVELS[1])
    logger.info(f"Using {config['description']} (level {level})")
    return config

def get_model_type():
    """Get the current model type based on environment variable"""
    return os.environ.get(ENV_PANNS_MODEL_TYPE, "pytorch").lower()

def get_model_path():
    """Get the current model path based on environment variable"""
    model_path = os.environ.get(ENV_PANNS_MODEL_PATH)
    if not model_path:
        model_path = PYTORCH_MODEL_PATH
    return model_path

def get_label_categories():
    """Get label categories for grouping similar sounds"""
    return {
        "Percussion": ['knock', 'tap', 'thump', 'bang', 'drum', 'percussion'],
        "Speech": ['speech', 'voice', 'talking', 'spoken'],
        "Music": ['music', 'song', 'singing', 'musical'],
        "Animal": ['animal', 'dog', 'cat', 'bird', 'bark'],
        "Alert": ['alarm', 'siren', 'alert', 'beep', 'horn'],
        "Mechanical": ['engine', 'motor', 'machine', 'mechanical'],
        "Household": ['kitchen', 'vacuum', 'microwave', 'door', 'bell']
    }

def print_config_summary():
    """Print a summary of the current configuration"""
    logger.info("=== SoundWatch Configuration ===")
    logger.info(f"Model path: {get_model_path()}")
    logger.info(f"Model type: {get_model_type()}")
    mem_config = get_memory_optimization_config()
    logger.info(f"Memory optimization: {mem_config['description']}")
    logger.info(f"Sample rate: {SAMPLE_RATE} Hz")
    logger.info(f"Silence threshold: {SILENCE_THRES} dB")
    logger.info(f"Prediction threshold: {PREDICTION_THRES}")
    logger.info("===============================")

if __name__ == "__main__":
    print_config_summary() 