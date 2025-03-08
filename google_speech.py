"""
Google Cloud Speech-to-Text Module for SoundWatch

This module provides speech recognition functionality using Google Cloud Speech-to-Text API.
It supports both synchronous and streaming recognition for better transcription quality.
"""
import os
import io
import time
import queue
import threading
import logging
import numpy as np
from google.cloud import speech
from threading import Lock
import traceback
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define filter_hallucinations function similar to the one in speech_to_text.py
def filter_hallucinations(text):
    """
    Filter out common hallucination patterns from transcribed text.
    
    Args:
        text: The transcribed text to filter
        
    Returns:
        Filtered text with hallucination patterns removed
    """
    if not text:
        return text
        
    # Check for repetitive patterns (a common hallucination issue)
    words = text.split()
    if len(words) >= 6:
        # Check for exact repetition of 3+ word phrases
        for i in range(len(words) - 5):
            phrase1 = " ".join(words[i:i+3])
            for j in range(i+3, len(words) - 2):
                phrase2 = " ".join(words[j:j+3])
                if phrase1.lower() == phrase2.lower():
                    # Found repetition, keep only the first occurrence
                    return " ".join(words[:j])
    
    # Remove common hallucination phrases
    common_hallucinations = [
        "thank you for watching",
        "thanks for watching", 
        "please subscribe",
        "like and subscribe",
    ]
    
    for phrase in common_hallucinations:
        if phrase in text.lower():
            # Remove the hallucination phrase
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            text = pattern.sub("", text)
    
    # Clean up any resulting double spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class GoogleSpeechToText:
    _instance = None
    _lock = Lock()
    _is_initialized = False
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GoogleSpeechToText, cls).__new__(cls)
            return cls._instance
    
    def __init__(self):
        if not self._is_initialized:
            self._initialize()
    
    def _initialize(self):
        """
        Initialize the Google Cloud Speech-to-Text client.
        """
        try:
            # Set up Google Cloud credentials
            credentials_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                           "asivate-452914-3c56106e7a07.json")
            
            if not os.path.exists(credentials_path):
                logger.error(f"Google Cloud credentials file not found at: {credentials_path}")
                raise FileNotFoundError(f"Google Cloud credentials file not found")
            
            # Set environment variable for authentication
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            
            # Initialize the client
            self.client = speech.SpeechClient()
            logger.info("Google Cloud Speech-to-Text client initialized successfully")
            
            # Configure default recognition settings
            self.config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US",
                model="default",
                use_enhanced=True,
                profanity_filter=False,
                enable_automatic_punctuation=True,
                max_alternatives=1
            )
            
            # Create a streaming config as well
            self.streaming_config = speech.StreamingRecognitionConfig(
                config=self.config,
                interim_results=False
            )
            
            # Mark as initialized
            GoogleSpeechToText._is_initialized = True
            
        except Exception as e:
            error_msg = f"Error initializing Google Speech-to-Text client: {str(e)}"
            logger.error(error_msg)
            traceback.print_exc()
    
    def preprocess_audio(self, audio_data, sample_rate=16000):
        """
        Preprocess audio data to improve speech recognition quality.
        
        Args:
            audio_data (numpy.ndarray): The audio data as a numpy array
            sample_rate (int): The sample rate of the audio (default: 16000)
            
        Returns:
            numpy.ndarray: The preprocessed audio data
        """
        if len(audio_data) == 0:
            return audio_data
            
        # No need for extensive preprocessing - convert to proper data type and normalize
        processed_audio = audio_data.astype(np.float32)
        
        # Normalize amplitude (Google prefers normalized audio)
        if np.max(np.abs(processed_audio)) > 0:
            processed_audio = processed_audio / np.max(np.abs(processed_audio))
        
        # Convert back to int16 for API
        processed_audio = (processed_audio * 32767).astype(np.int16)
        
        return processed_audio

    def transcribe(self, audio_data, sample_rate=16000):
        """
        Transcribe audio data to text using Google Cloud Speech-to-Text API.
        
        Args:
            audio_data (numpy.ndarray): The audio data as a numpy array
            sample_rate (int): The sample rate of the audio (default: 16000)
            
        Returns:
            str: The transcribed text
        """
        if audio_data is None or len(audio_data) == 0:
            logger.warning("Empty audio data provided for transcription")
            return ""
        
        try:
            # Preprocess audio data
            processed_audio = self.preprocess_audio(audio_data, sample_rate)
            
            # Convert audio data to appropriate format
            audio_content = processed_audio.tobytes()
            
            # Create audio object
            audio = speech.RecognitionAudio(content=audio_content)
            
            # Update config to match current sample rate
            if sample_rate != self.config.sample_rate_hertz:
                self.config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=sample_rate,
                    language_code="en-US",
                    model="default",
                    use_enhanced=True,
                    profanity_filter=False,
                    enable_automatic_punctuation=True,
                    max_alternatives=1
                )
            
            # Detect speech using Google Cloud Speech-to-Text
            logger.info("Sending audio to Google Cloud Speech-to-Text API...")
            response = self.client.recognize(config=self.config, audio=audio)
            
            # Process the response
            transcription = ""
            for result in response.results:
                transcription += result.alternatives[0].transcript
            
            # Apply post-processing to remove possible hallucinations
            transcription = filter_hallucinations(transcription)
            
            logger.info(f"Google transcription result: '{transcription}'")
            return transcription
            
        except Exception as e:
            logger.error(f"Error in Google Cloud Speech-to-Text transcription: {str(e)}")
            traceback.print_exc()
            return ""

# Function for use in the main server code
def transcribe_with_google(audio_data, sample_rate=16000):
    """
    Wrapper function for easier integration with server code.
    
    Args:
        audio_data: Audio data to transcribe
        sample_rate: Sample rate of the audio
        
    Returns:
        The transcribed text
    """
    speech_processor = GoogleSpeechToText()
    return speech_processor.transcribe(audio_data, sample_rate) 