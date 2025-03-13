"""
PANNs Model Module for SoundWatch

This module integrates the CNN13 model from the PANNs (Pretrained Audio Neural Networks)
for audio recognition. It is based on the General-Purpose-Sound-Recognition-Demo project
by Yin Cao, Qiuqiang Kong, et al.

Reference: https://github.com/yinkalario/General-Purpose-Sound-Recognition-Demo
Paper: https://arxiv.org/abs/1912.10211
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import librosa
import h5py
import logging
import math
from scipy.interpolate import interp1d
import time
import gc
import psutil
import threading
import json
import traceback
import wget
import csv

# Try to import from models_code if it exists
try:
    from models_code import Cnn13_GMP_64x64
    print("Successfully imported Cnn13_GMP_64x64 from models_code")
except ImportError:
    # Fall back to the built-in model implementation
    print("Could not import Cnn13_GMP_64x64 from models_code, using built-in implementation")

# CPU optimization - set number of threads to use all cores
# Get the number of CPU cores and set torch to use all of them
try:
    num_cores = psutil.cpu_count(logical=True)
    if num_cores:
        torch.set_num_threads(num_cores)
        torch.set_num_interop_threads(num_cores)
        os.environ['OMP_NUM_THREADS'] = str(num_cores)
        os.environ['MKL_NUM_THREADS'] = str(num_cores)
        print(f"PyTorch configured to use {num_cores} CPU cores")
except Exception as e:
    print(f"Warning: Could not optimize CPU threads: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for file paths and resources
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models')  # Updated to match server structure
ASSETS_PATH = os.path.join(os.path.dirname(__file__), 'assets')
MODEL_FN = os.path.join(MODEL_PATH, 'Cnn13_GMP_64x64_520000_iterations_mAP=0.42.pth')  # Updated to use the stronger Cnn13 model
SCALAR_FN = os.path.join(ASSETS_PATH, 'scalar.h5')  # Scalar values for normalization
CSV_FNAME = os.path.join(ASSETS_PATH, 'audioset_labels.csv')  # AudioSet labels CSV
ALT_CSV_FNAME = os.path.join(ASSETS_PATH, 'validate_meta.csv')  # Alternative AudioSet labels
DOMESTIC_CSV_FNAME = os.path.join(ASSETS_PATH, 'domestic_labels.csv')  # Curated domestic sounds
CLASS_LABELS_CSV = os.path.join(ASSETS_PATH, 'class_labels_indices.csv')  # Updated path to class labels

# Global variables
panns_model = None  # The PyTorch model object
panns_inference = None  # PANNs inference helper
is_panns_loaded = False  # Flag to track if model is loaded
available_labels = None  # Cached available labels

# Default audio parameters
SAMPLE_RATE = 32000
N_FFT = 1024
HOP_LENGTH = 500
MEL_BINS = 64
FMIN = 50
FMAX = 14000

# Model initialization helpers
def init_layer(layer, nonlinearity='leaky_relu'):
    """Initialize a Linear or Convolutional layer."""
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)

# CNN model architecture
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.init_weights()
        
    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = torch.relu_(self.bn1(self.conv1(x)))
        x = torch.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = torch.nn.functional.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = torch.nn.functional.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')
        
        return x

# CNN13 model architecture for better accuracy (mAP=0.423)
class Cnn13(nn.Module):
    def __init__(self, classes_num, strong_target_training=False):
        super(Cnn13, self).__init__()

        # CNN architecture with 13 convolutional layers
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        
        # Fully connected layers
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        self.init_weights()
        
    def init_weights(self):
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
        
    def get_bottleneck(self, input):
        """Process input through convolutional layers to get bottleneck features.
        
        Args:
            input: Tensor with shape [batch_size, times_steps, freq_bins]
        
        Returns:
            Tensor with shape [batch_size, 2048]
        """
        # Debug input shape
        logger.info(f"get_bottleneck input shape: {input.shape}")
        
        # Add channel dimension to input: [batch_size, 1, times_steps, freq_bins]
        x = input[:, None, :, :]
        logger.info(f"After adding channel dimension: {x.shape}")
        
        # First conv block with reduced pooling
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        logger.info(f"After conv_block1: {x.shape}")
        x = F.dropout(x, p=0.2)
        
        # Second conv block with reduced pooling
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        logger.info(f"After conv_block2: {x.shape}")
        x = F.dropout(x, p=0.2)
        
        # Third conv block with reduced pooling
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        logger.info(f"After conv_block3: {x.shape}")
        x = F.dropout(x, p=0.2)
        
        # Fourth conv block with reduced pooling
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        logger.info(f"After conv_block4: {x.shape}")
        x = F.dropout(x, p=0.2)
        
        # Fifth conv block with reduced pooling (CNN13 specific)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        logger.info(f"After conv_block5: {x.shape}")
        x = F.dropout(x, p=0.2)
        
        # Sixth conv block with reduced pooling (CNN13 specific)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        logger.info(f"After conv_block6: {x.shape}")
        x = F.dropout(x, p=0.2)
        
        # Global pooling
        x = torch.mean(x, dim=3)
        logger.info(f"After mean pooling on freq dimension: {x.shape}")
        
        (x, _) = torch.max(x, dim=2)
        logger.info(f"After max pooling on time dimension: {x.shape}")
        
        # Final FC layer
        x = F.relu_(self.fc1(x))
        logger.info(f"Final bottleneck features: {x.shape}")
        
        return x
        
    def forward(self, input):
        x = self.get_bottleneck(input)
        x = F.dropout(x, p=0.5)
        x = self.fc_audioset(x)
        x = torch.sigmoid(x)
        return x

# Keep CNN9 for backward compatibility
class Cnn9_GMP_64x64(nn.Module):
    def __init__(self, classes_num, strong_target_training=False):
        super(Cnn9_GMP_64x64, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        
        self.fc = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        
        self.init_weights()
        
    def init_weights(self):
        init_layer(self.fc)
        init_layer(self.fc_audioset)
        
    def get_bottleneck(self, input):
        """Process input through convolutional layers to get bottleneck features.
        
        Args:
            input: Tensor with shape [batch_size, times_steps, freq_bins]
        
        Returns:
            Tensor with shape [batch_size, 512]
        """
        # Debug input shape
        logger.info(f"get_bottleneck input shape: {input.shape}")
        
        # Add channel dimension to input: [batch_size, 1, times_steps, freq_bins]
        # This matches the original implementation
        x = input[:, None, :, :]
        logger.info(f"After adding channel dimension: {x.shape}")
        
        # Process through the CNN blocks with original pooling settings
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        logger.info(f"After conv_block1: {x.shape}")
        
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        logger.info(f"After conv_block2: {x.shape}")
        
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        logger.info(f"After conv_block3: {x.shape}")
        
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        logger.info(f"After conv_block4: {x.shape}")
        
        # Global pooling
        x = torch.mean(x, dim=3)
        logger.info(f"After mean pooling on freq dimension: {x.shape}")
        
        (x, _) = torch.max(x, dim=2)
        logger.info(f"After max pooling on time dimension: {x.shape}")
        
        # Final FC layer for feature refinement
        x = F.relu_(self.fc(x))
        logger.info(f"Final bottleneck features: {x.shape}")
        
        return x
        
    def forward(self, input):
        """
        Input: (batch_size, times_steps, freq_bins)
        Output: (batch_size, classes_num)
        """
        x = self.get_bottleneck(input)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc_audioset(x)
        x = torch.sigmoid(x)
        return x

class PANNsModelInference:
    """
    This class manages all aspects of the PANNs model for sound recognition:
    - Model loading
    - Audio preprocessing
    - Inference
    - Label management
    """
    
    # Constants from the original implementation
    LOGMEL_MEANS = np.float32([
        -14.050895, -13.107869, -13.1390915, -13.255364, -13.917199,
        -14.087848, -14.855916, -15.266642,  -15.884036, -16.491768,
        -17.067415, -17.717588, -18.075916,  -18.84405,  -19.233824,
        -19.954256, -20.180824, -20.695705,  -21.031914, -21.33451,
        -21.758745, -21.917028, -22.283598,  -22.737364, -22.920172,
        -23.23437,  -23.66509,  -23.965239,  -24.580393, -24.67597,
        -25.194445, -25.55243,  -25.825129,  -26.309643, -26.703104,
        -27.28697,  -27.839067, -28.228388,  -28.746237, -29.236507,
        -29.937782, -30.755503, -31.674414,  -32.853516, -33.959763,
        -34.88149,  -35.81145,  -36.72929,   -37.746593, -39.000496,
        -40.069244, -40.947514, -41.79767,   -42.81981,  -43.8541,
        -44.895683, -46.086784, -47.255924,  -48.520145, -50.726765,
        -52.932228, -54.713795, -56.69902,   -59.078354])
    
    LOGMEL_STDDEVS = np.float32([
        22.680508, 22.13264,  21.857653, 21.656355, 21.565693, 21.525793,
        21.450764, 21.377304, 21.338581, 21.3247,   21.289171, 21.221565,
        21.175856, 21.049534, 20.954664, 20.891844, 20.849905, 20.809206,
        20.71186,  20.726717, 20.72358,  20.655743, 20.650305, 20.579372,
        20.583157, 20.604849, 20.5452,   20.561695, 20.448244, 20.46753,
        20.433657, 20.412025, 20.47265,  20.456116, 20.487215, 20.387547,
        20.331848, 20.310328, 20.292257, 20.292326, 20.241796, 20.19396,
        20.23783,  20.564362, 21.075726, 21.332186, 21.508852, 21.644777,
        21.727905, 22.251642, 22.65972,  22.800117, 22.783764, 22.78581,
        22.86413,  22.948992, 23.12939,  23.180748, 23.03542,  23.131435,
        23.454556, 23.39839,  23.254364, 23.198978])
    
    def __init__(self):
        """Initialize the PANNs model inference engine."""
        self.model = None
        self.labels = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scalar = None
        self._initialized = False
        self.sample_rate = SAMPLE_RATE
        # Add a threading lock for thread safety during prediction
        self.lock = threading.Lock()
        
    def initialize(self):
        """
        Initialize the model - load weights, setup device, etc.
        """
        if self._initialized:
            print("Model already initialized")
            return True
            
        try:
            print("Initializing PANNs model for inference")
            
            # Set device - use CUDA if available, otherwise CPU
            if torch.cuda.is_available():
                print("CUDA available - using GPU for inference")
                self.device = torch.device('cuda')
            else:
                print("CUDA not available - using CPU for inference")
                self.device = torch.device('cpu')
            
            # Set default parameters
            self.sample_rate = SAMPLE_RATE
            self.nfft = N_FFT
            self.hopsize = HOP_LENGTH
            self.melbins = MEL_BINS
            self.fmin = FMIN
            self.fmax = FMAX
                
            # Load model
            try:
                # Load the model - Using the stronger Cnn13_GMP_64x64 model (mAP=0.42) instead of Cnn9 (mAP=0.37)
                # This provides better audio recognition accuracy
                model_path = MODEL_FN
                print(f"Loading PANNs model from {model_path}")
                
                # Try to use the imported Cnn13_GMP_64x64 from models_code if available
                # If not available, fall back to the built-in Cnn13 class
                try:
                    # Check if we successfully imported Cnn13_GMP_64x64 from models_code
                    if 'Cnn13_GMP_64x64' in globals():
                        print("Using Cnn13_GMP_64x64 from models_code for inference engine")
                        self.model = Cnn13_GMP_64x64(classes_num=527)
                    else:
                        print("Using built-in Cnn13 class for inference engine")
                        self.model = Cnn13(classes_num=527)
                except NameError:
                    print("Cnn13_GMP_64x64 not found, using built-in Cnn13 class for inference engine")
                    self.model = Cnn13(classes_num=527)
                
                # Load weights
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                
                self.model.to(self.device)
                self.model.eval()
                print(f"Successfully loaded model from {model_path}")
                
                self._initialized = True
                print("PANNs model initialized successfully")
                return True
            
            except Exception as e:
                print(f"Error loading model weights: {e}")
                print("Detailed error information:")
                traceback.print_exc()
                return False
                
        except Exception as e:
            print(f"Error initializing PANNs model: {e}")
            traceback.print_exc()
            return False
    
    def logmel_extract(self, audio):
        """
        Extract log mel spectrogram features from audio data.
        The PANNs model expects a spectrogram with shape (time_steps, 64).
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Log mel spectrogram with shape (time_steps, 64) for CNN model
        """
        try:
            # Audio should already be normalized to [-1, 1] and sampled at 32kHz
            # Parameters based on original PANNs implementation
            n_fft = 1024  # FFT window size
            hop_length = 320  # Hop size - adjusted for CNN model
            n_mels = 64  # Number of mel bins
            fmin = 50  # Minimum frequency (Hz)
            fmax = 14000  # Maximum frequency (Hz)
            
            # Compute STFT
            stft = librosa.stft(
                y=audio, 
                n_fft=n_fft,
                hop_length=hop_length,
                window='hann', 
                center=True,
                pad_mode='reflect'
            )
            
            # Convert to power spectrogram
            power_spec = np.abs(stft) ** 2
            
            # Create mel filterbank if not already created
            if not hasattr(self, 'melW'):
                self.melW = librosa.filters.mel(
                    sr=self.sample_rate,
                    n_fft=n_fft,
                    n_mels=n_mels,
                    fmin=fmin,
                    fmax=fmax
                )
            
            # Apply mel filterbank
            mel_spec = np.dot(self.melW, power_spec)
            
            # Convert to log scale
            log_mel_spec = librosa.power_to_db(mel_spec, ref=1.0, amin=1e-10, top_db=None)
            
            # Transpose to get (time_steps, n_mels) shape
            log_mel_spec = log_mel_spec.T
            
            # Normalize the spectrogram
            if self.scalar is not None:
                # Use loaded scalar values
                log_mel_spec = (log_mel_spec - self.scalar['mean']) / self.scalar['std']
            else:
                # Use hardcoded values
                log_mel_spec = (log_mel_spec - self.LOGMEL_MEANS) / self.LOGMEL_STDDEVS
            
            # Log the shape for debugging
            logger.info(f"Extracted log mel spectrogram with shape {log_mel_spec.shape}")
            
            return log_mel_spec
            
        except Exception as e:
            print(f"Error in logmel_extract: {e}")
            traceback.print_exc()
            # Return an empty spectrogram in case of error
            return np.zeros((101, 64), dtype=np.float32)  # Default shape for CNN9
    
    def predict(self, audio_data, top_k=5, threshold=0.2, boost_other_categories=False):
        """
        Run inference on audio data and return predictions.
        
        Args:
            audio_data: Audio data as numpy array
            top_k: Number of top predictions to return
            threshold: Confidence threshold for predictions
            boost_other_categories: Whether to boost categories other than speech/music
            
        Returns:
            List of (label, score) tuples for the top K predictions
        """
        try:
            # Process audio
            x = self._preprocess_audio(audio_data)
            
            # Thread safety - don't allow multiple predictions at once
            with self.lock:
                # Run inference
                with torch.no_grad():
                    self.model.eval()
                    prediction = self.model(x)
                
                # Get probabilities
                probs = prediction.squeeze().cpu().numpy()
            
            # Check for percussive sounds - they have characteristic time-domain patterns
            is_percussive = self._is_percussive_sound(audio_data)
            
            # MODIFIED: Still detect percussion sounds, but don't prioritize them over higher confidence predictions
            if is_percussive:
                print("Percussive sound detected, but using standard prediction logic")
            
            # Get top-k indices and their probabilities
            indices = np.argsort(probs)[-top_k:][::-1]
            selected_probs = probs[indices]
            
            # Filter by threshold
            mask = selected_probs >= threshold
            indices = indices[mask]
            selected_probs = selected_probs[mask]
            
            # If no predictions above threshold, include at least the top prediction
            if len(indices) == 0 and len(probs) > 0:
                indices = [np.argmax(probs)]
                selected_probs = [probs[indices[0]]]
            
            # Convert to labels
            result = []
            for i, p in zip(indices, selected_probs):
                if i < len(self.labels):
                    label = self.labels[i]
                    result.append((label, float(p)))
                else:
                    print(f"Warning: Index {i} out of bounds for labels list of length {len(self.labels)}")
            
            # Sort by probability in descending order
            result.sort(key=lambda x: x[1], reverse=True)
            print(f"PANNs prediction results: {result}")
            
            return result
                
        except Exception as e:
            print(f"Error in PANNs prediction: {e}")
            traceback.print_exc()
            return []
            
    def _is_percussive_sound(self, audio_data):
        """
        Analyzes the audio to determine if it's likely a percussive sound like knocking.
        Percussive sounds have characteristic shapes in the time domain:
        - Sharp onsets (sudden increases in amplitude)
        - Quick decay
        - Often multiple peaks in quick succession (for knocking)
        
        Args:
            audio_data: numpy array of audio samples
            
        Returns:
            bool: True if the audio is likely percussive
        """
        try:
            # Get the envelope of the audio signal
            envelope = np.abs(audio_data)
            
            # Smooth the envelope to reduce noise
            window_size = 320  # About 10ms at 32kHz (smaller window for better detection)
            smoothed = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
            
            # Calculate the energy of the audio
            rms = np.sqrt(np.mean(np.square(audio_data)))
            if rms < 0.003:  # Increased threshold for very quiet audio
                print("Audio too quiet for percussion detection")
                return False
            
            # Find peaks in the envelope
            from scipy.signal import find_peaks
            # More strict peak detection with higher thresholds
            peaks, peak_props = find_peaks(
                smoothed, 
                height=0.15*np.max(smoothed),  # Increased threshold to detect only stronger knocks
                distance=500,  # Minimum distance between peaks (~15ms at 32kHz)
                prominence=0.2*np.max(smoothed)  # Increased to ensure peaks stand out more
            )
            
            # Print detailed information about detected peaks
            peak_heights = peak_props['peak_heights']
            print(f"Detected {len(peaks)} peaks with heights: {peak_heights}")
            
            # Characteristics of percussion:
            # 1. Number of prominent peaks (2-10 for knocking)
            # 2. Decay rate after peaks (fast decay for percussion)
            # 3. Spacing between peaks (consistent for intentional knocking)
            
            # Calculate peak spacing and decay rates
            if len(peaks) >= 2 and len(peaks) <= 15:  # Multiple distinct peaks
                # Check peak spacing - knocking typically has peaks spaced 0.1-0.5s apart
                peak_spacing = np.diff(peaks) / 32000  # Convert to seconds
                avg_spacing = np.mean(peak_spacing)
                std_spacing = np.std(peak_spacing)
                
                # Calculate consistency of spacing (lower std/mean ratio = more consistent)
                spacing_consistency = std_spacing / avg_spacing if avg_spacing > 0 else 999
                
                # Check decay rates - percussion has fast decay
                decay_rates = []
                for peak in peaks:
                    if peak + 3000 < len(smoothed):  # Look 100ms ahead
                        decay_window = smoothed[peak:peak+3000]
                        if len(decay_window) > 0 and decay_window[0] > 0:
                            # Calculate decay rate over 100ms window
                            decay_rate = (decay_window[0] - np.min(decay_window)) / decay_window[0]
                            decay_rates.append(decay_rate)
                
                avg_decay = np.mean(decay_rates) if decay_rates else 0
                
                # Print detailed percussion characteristics for debugging
                print(f"Percussion details: peaks={len(peaks)}, spacing={avg_spacing:.2f}s, consistency={spacing_consistency:.2f}, decay={avg_decay:.2f}")
                
                # Make knocking criteria more strict
                is_percussion = (
                    (0.05 < avg_spacing < 0.3) and  # More narrow range for knocking timing
                    (spacing_consistency < 0.4) and  # Require more consistent spacing
                    (avg_decay > 0.6) and  # Require faster decay
                    (len(peaks) >= 2 and len(peaks) <= 10)  # More limited range of peaks
                )
                
                # Stronger evidence of knocking if we have 3-6 evenly spaced peaks
                if (3 <= len(peaks) <= 6) and spacing_consistency < 0.25 and avg_decay > 0.7:
                    print("STRONG EVIDENCE of knocking pattern detected!")
                    return True
                
                return is_percussion
                
            # Single sharp impact could also be percussion
            elif len(peaks) == 1:
                # Check for fast decay after the peak
                peak = peaks[0]
                if peak + 3000 < len(smoothed):
                    decay_window = smoothed[peak:peak+3000]
                    if len(decay_window) > 0 and decay_window[0] > 0:
                        decay_rate = (decay_window[0] - np.min(decay_window)) / decay_window[0]
                        
                        print(f"Single peak percussion analysis: decay={decay_rate:.2f}, height={peak_heights[0]}")
                        
                        # Make threshold more strict
                        return decay_rate > 0.85 and peak_heights[0] > 0.4*np.max(smoothed)
            
            return False
                
        except Exception as e:
            print(f"Error in percussion detection: {e}")
            traceback.print_exc()
            return False
    
    def _get_percussion_indices(self):
        """
        Returns indices of percussion-related categories in the label list.
        Percussion sounds include knocking, tapping, and similar impact sounds.
        """
        percussion_keywords = [
            'knock', 'tap', 'drum', 'percussion', 'thump', 'impact', 'bang',
            'hit', 'pound', 'slam', 'clap', 'strike', 'beat'
        ]
        
        indices = []
        
        for i, label in enumerate(self.labels):
            label_lower = label.lower()
            if any(keyword in label_lower for keyword in percussion_keywords):
                indices.append(i)
        
        return indices
    
    def _get_speech_music_indices(self):
        """Returns indices of speech and music categories in the label list."""
        speech_music_keywords = ['speech', 'speak', 'voice', 'talk', 'music', 'singing']
        
        indices = []
        
        for i, label in enumerate(self.labels):
            label_lower = label.lower()
            if any(keyword in label_lower for keyword in speech_music_keywords):
                indices.append(i)
        
        return indices
    
    def map_to_homesounds(self, results, threshold=0.2):
        """
        Map PANNs AudioSet labels to homesounds categories.
        This provides compatibility with the existing SoundWatch categories.
        
        Args:
            results: List of (label, confidence) tuples from predict()
            threshold: Minimum confidence threshold
            
        Returns:
            Dictionary of mapped homesounds predictions
        """
        # Dict to store homesounds prediction scores
        homesounds_scores = {}
        
        # Mapping from AudioSet labels to homesounds categories
        # This is a simplified mapping and can be expanded as needed
        audioset_to_homesounds = {
            # Speech and human sounds
            "Speech": "speech",
            "Male speech, man speaking": "speech",
            "Female speech, woman speaking": "speech",
            "Child speech, kid speaking": "speech",
            "Conversation": "speech",
            "Narration, monologue": "speech",
            "Babbling": "speech",
            "Whispering": "speech",
            "Laughter": "speech",
            "Crying, sobbing": "speech",
            
            # Animal sounds
            "Animal": "animal",
            "Domestic animals, pets": "animal",
            "Dog": "animal-dog",
            "Bark": "animal-dog",
            "Howl": "animal-dog",
            "Growling": "animal-dog",
            "Whimper (dog)": "animal-dog",
            "Cat": "animal-cat",
            "Purr": "animal-cat",
            "Meow": "animal-cat",
            "Bird": "animal-bird",
            "Bird vocalization, bird call, bird song": "animal-bird",
            
            # Alerts and alarms
            "Alarm": "hazard-alarm",
            "Fire alarm": "hazard-alarm",
            "Smoke detector, smoke alarm": "hazard-alarm",
            "Siren": "hazard-siren",
            "Civil defense siren": "hazard-siren",
            "Police car (siren)": "hazard-siren",
            "Ambulance (siren)": "hazard-siren",
            "Fire engine, fire truck (siren)": "hazard-siren",
            "Air horn, truck horn": "hazard-siren",
            "Car alarm": "hazard-alarm",
            "Buzzer": "hazard-alarm",
            "Smoke detector, smoke alarm": "hazard-alarm",
            "Bell": "household-bell",
            "Doorbell": "household-doorbell",
            "Ding-dong": "household-doorbell",
            "Beep, bleep": "hazard-alarm",
            
            # Knocking sounds
            "Knock": "household-knock",
            "Door knock": "household-knock",
            "Knocking": "household-knock",
            "Tap": "household-knock",
            "Tapping": "household-knock",
            "Knock (door)": "household-knock",
            "Percussion": "household-knock",
            "Drum": "household-knock",
            
            # Household sounds
            "Water tap, faucet": "household-water",
            "Sink (filling or washing)": "household-water",
            "Bathtub (filling or washing)": "household-water",
            "Microwave oven": "household-microwave",
            "Blender": "household-blender",
            "Electric mixer": "household-blender",
            "Refrigerator": "household-refrigerator",
            "Kettle whistle": "household-whistling",
            
            # Vehicle sounds
            "Vehicle": "vehicle",
            "Motor vehicle (road)": "vehicle",
            "Car": "vehicle-car",
            "Car passing by": "vehicle-car",
            "Bus": "vehicle",
            "Truck": "vehicle",
            "Train": "vehicle-train",
            "Rail transport": "vehicle-train",
            "Aircraft": "vehicle-aircraft",
            "Bicycle": "vehicle",
            "Motorcycle": "vehicle",
            
            # Other common mappings
            "Telephone": "household-phone",
            "Telephone bell ringing": "household-phone",
            "Cell phone": "household-phone",
            "Ringtone": "household-phone",
            "Mechanical fan": "household-fan",
            "Air conditioning": "household-fan",
            "Vacuum cleaner": "household-vacuum",
            "Clock": "household-clock",
            "Tick": "household-clock",
            "Tick-tock": "household-clock",
            "Alarm clock": "household-clock",
            "Dishes, pots, and pans": "household-dishes",
            "Cutlery, silverware": "household-dishes",
            "Frying (food)": "household-cooking",
            "Toilet flush": "household-toilet",
            "Drawer open or close": "household-door",
            "Cupboard open or close": "household-door",
        }
        
        # Process results
        for label, confidence in results:
            if confidence < threshold:
                continue
                
            # Check if we have a mapping
            if label in audioset_to_homesounds:
                homesound_label = audioset_to_homesounds[label]
                
                # Add or update score (take max if multiple matches)
                if homesound_label in homesounds_scores:
                    homesounds_scores[homesound_label] = max(homesounds_scores[homesound_label], confidence)
                else:
                    homesounds_scores[homesound_label] = confidence
        
        # Format output to match expected homesounds prediction format
        if homesounds_scores:
            # Sort by confidence (descending)
            sorted_results = sorted(homesounds_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Convert to the format expected by the server - return a list of predictions directly
            return [{"label": label, "score": float(score)} for label, score in sorted_results]
        else:
            # Return a generic "Unknown Sound" prediction if nothing matches
            return [{"label": "Unknown Sound", "score": 0.7}]

    def get_available_labels(self):
        """
        Get the list of available labels for this model
        Returns:
            List of label strings
        """
        try:
            if self.labels is not None:
                return self.labels.tolist() if isinstance(self.labels, np.ndarray) else self.labels
            return []
        except Exception as e:
            print(f"Error getting available labels: {e}")
            return []

    def _preprocess_audio(self, audio_data):
        """
        Preprocess audio data for model inference.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            PyTorch tensor ready for model input
        """
        # Log audio statistics for debugging
        audio_mean = np.mean(audio_data)
        audio_std = np.std(audio_data)
        audio_min = np.min(audio_data)
        audio_max = np.max(audio_data)
        audio_abs_max = np.max(np.abs(audio_data))
        
        print(f"Audio stats - Mean: {audio_mean:.6f}, Std: {audio_std:.6f}, Min: {audio_min:.6f}, Max: {audio_max:.6f}, Abs Max: {audio_abs_max:.6f}")
        
        # Check for valid audio data
        if audio_data is None or len(audio_data) == 0:
            print("Empty audio data received")
            raise ValueError("Empty audio data received")
    
        # Check for non-finite values
        if not np.all(np.isfinite(audio_data)):
            print("Audio contains non-finite values, fixing...")
            audio_data = np.nan_to_num(audio_data)
        
        # Normalize audio if needed
        if np.max(np.abs(audio_data)) > 1.0:
            print("Normalizing audio...")
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Make sure audio is long enough (at least 1 second at 32kHz = 32000 samples)
        original_length = len(audio_data)
        if original_length < 32000:
            print(f"Audio too short ({original_length} samples), padding to 32000 samples")
            # If it's very short, repeat the audio to reach minimum length
            if original_length < 16000:
                # Repeat the audio multiple times to reach 32000 samples
                repeat_count = int(np.ceil(32000 / original_length))
                audio_data = np.tile(audio_data, repeat_count)[:32000]
            else:
                # Pad with zeros to reach 32000 samples
                padding = np.zeros(32000 - original_length)
                audio_data = np.concatenate([audio_data, padding])
        
        # Extract log mel spectrogram features
        x = self.logmel_extract(audio_data)
        
        # Verify that we have the correct shape for the CNN13 model
        if x.shape != (128, 64):
            print(f"WARNING: Fixing spectrogram shape from {x.shape} to (128, 64)")
            # Create a properly sized spectrogram
            fixed_spec = np.zeros((128, 64), dtype=np.float32)
            
            # Copy what we can from the original spectrogram
            h = min(x.shape[0], 128)
            w = min(x.shape[1], 64)
            fixed_spec[:h, :w] = x[:h, :w]
            
            x = fixed_spec
        
        # Normalize with mean and std values (essential for correct predictions)
        try:
            # Check if we need to reshape the mean/std vectors
            if hasattr(self, 'mean') and hasattr(self, 'std') and len(self.mean) != x.shape[0]:
                print(f"Reshaping normalization vectors from {len(self.mean)} to {x.shape[0]}")
                # Create new mean/std vectors of the right size
                new_mean = np.zeros(x.shape[0], dtype=np.float32)
                new_std = np.ones(x.shape[0], dtype=np.float32)
                
                # Copy the values we have
                common_len = min(len(self.mean), x.shape[0])
                new_mean[:common_len] = self.mean[:common_len]
                new_std[:common_len] = self.std[:common_len]
                
                # Apply the normalization
                x = (x - new_mean.reshape(-1, 1)) / new_std.reshape(-1, 1)
            elif hasattr(self, 'mean') and hasattr(self, 'std'):
                # Apply the normalization as usual
                x = (x - self.mean.reshape(-1, 1)) / self.std.reshape(-1, 1)
        except Exception as e:
            print(f"Error applying normalization: {e}, using default normalization")
            # Use a simple global normalization as a fallback
            if np.std(x) > 0:
                x = (x - np.mean(x)) / np.std(x)
        
        # Convert to PyTorch tensor and reshape for model input
        x = torch.tensor(x, dtype=torch.float32)
        
        # Add batch dimension: (time_steps, freq_bins) -> (1, time_steps, freq_bins)
        x = x.unsqueeze(0)
        print(f"Input tensor shape: {x.shape}")
        
        return x

# Create singleton instance
panns_inference = PANNsModelInference()

def get_available_labels():
    """
    Get the list of available sound labels from the loaded model
    Returns a list of label strings
    """
    global available_labels
    
    # Return cached labels if available
    if available_labels is not None and len(available_labels) > 0:
        print(f"Using cached labels, {len(available_labels)} labels available")
        return available_labels
        
    try:
        # Try to get from the PANNs inference object first
        if hasattr(panns_inference, 'get_available_labels'):
            labels = panns_inference.get_available_labels()
            print(f"Got {len(labels)} labels from panns_inference object")
            if labels and len(labels) > 0:
                print(f"Sample labels: {labels[:10]}...")
                available_labels = labels
                return labels
            else:
                print("No labels returned from panns_inference object")
        
        labels = []
        
        # First check if class_labels_indices.csv exists in the assets directory (most reliable source)
        if os.path.exists(CLASS_LABELS_CSV):
            try:
                print(f"Loading labels from {CLASS_LABELS_CSV}")
                df = pd.read_csv(CLASS_LABELS_CSV)
                
                if 'display_name' in df.columns:
                    labels = df['display_name'].tolist()
                    print(f"Loaded {len(labels)} labels from {CLASS_LABELS_CSV} using display_name column")
                    print(f"Sample labels: {labels[:10]}...")
                    available_labels = labels
                    return labels
                elif 'name' in df.columns:  # Different format
                    labels = df['name'].tolist()
                    print(f"Loaded {len(labels)} labels from {CLASS_LABELS_CSV} using name column")
                    print(f"Sample labels: {labels[:10]}...")
                    available_labels = labels
                    return labels
                else:
                    print(f"Could not find display_name or name column in {CLASS_LABELS_CSV}")
                    print(f"Available columns: {df.columns.tolist()}")
            except Exception as e:
                print(f"Error loading labels from {CLASS_LABELS_CSV}: {e}")
                print(f"Attempting alternative loading methods...")
        
        # Try loading manually with CSV reader
        if os.path.exists(CLASS_LABELS_CSV):
            try:
                print(f"Loading labels from {CLASS_LABELS_CSV} using CSV reader")
                labels = []
                with open(CLASS_LABELS_CSV, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f, delimiter=',')
                    header = next(reader, None)
                    print(f"CSV header: {header}")
                    
                    # Find the appropriate column index for display_name or name
                    name_col_idx = None
                    if header:
                        for i, col in enumerate(header):
                            if col.lower() in ['display_name', 'name', 'label']:
                                name_col_idx = i
                                print(f"Using column {i} ({col}) for labels")
                                break
                    
                    # If we found a column, use it, otherwise default to the last column
                    if name_col_idx is None:
                        name_col_idx = -1 if len(header) > 2 else 1
                        print(f"No explicit name column found, using column {name_col_idx}")
                    
                    # Read the labels
                    for row in reader:
                        if len(row) > abs(name_col_idx):
                            # Handle quoted values
                            label = row[name_col_idx].strip('"\'')
                            labels.append(label)
                
                if labels:
                    print(f"Loaded {len(labels)} labels from {CLASS_LABELS_CSV} using CSV reader")
                    print(f"Sample labels: {labels[:10]}...")
                    available_labels = labels
                    return labels
                else:
                    print(f"No labels found in {CLASS_LABELS_CSV} using CSV reader")
            except Exception as e:
                print(f"Error loading labels from {CLASS_LABELS_CSV} using CSV reader: {e}")
                traceback.print_exc()
        
        # Try primary audioset labels
        if os.path.exists(CSV_FNAME):
            try:
                print(f"Loading labels from {CSV_FNAME}")
                df = pd.read_csv(CSV_FNAME)
                if 'display_name' in df.columns:
                    labels = df['display_name'].tolist()
                    print(f"Loaded {len(labels)} labels from {CSV_FNAME}")
                    print(f"Sample labels: {labels[:10]}...")
                    available_labels = labels
                    return labels
            except Exception as e:
                print(f"Error loading labels from {CSV_FNAME}: {e}")
        
        # Try alternate validate_meta.csv
        if os.path.exists(ALT_CSV_FNAME):
            try:
                print(f"Loading labels from {ALT_CSV_FNAME}")
                df = pd.read_csv(ALT_CSV_FNAME)
                if 'display_name' in df.columns:
                    labels = df['display_name'].tolist()
                    print(f"Loaded {len(labels)} labels from {ALT_CSV_FNAME}")
                    print(f"Sample labels: {labels[:10]}...")
                    available_labels = labels
                    return labels
            except Exception as e:
                print(f"Error loading labels from {ALT_CSV_FNAME}: {e}")
        
        # Try domestic labels as last resort
        if os.path.exists(DOMESTIC_CSV_FNAME):
            try:
                print(f"Loading labels from {DOMESTIC_CSV_FNAME}")
                df = pd.read_csv(DOMESTIC_CSV_FNAME)
                if 'display_name' in df.columns:
                    labels = df['display_name'].tolist()
                    print(f"Loaded {len(labels)} labels from {DOMESTIC_CSV_FNAME}")
                    print(f"Sample labels: {labels[:10]}...")
                    available_labels = labels
                    return labels
            except Exception as e:
                print(f"Error loading labels from {DOMESTIC_CSV_FNAME}: {e}")
        
        # If all else fails, create default labels
        print("Could not load labels from any source, using default AudioSet labels")
        # Create default labels for the 527 AudioSet classes
        labels = [f"AudioSet_Class_{i}" for i in range(527)]
        available_labels = labels
        
        return labels
    except Exception as e:
        print(f"Error getting available labels: {e}")
        traceback.print_exc()
        return []

# Add a new function that matches what server.py is expecting
def get_labels():
    """
    Alias for get_available_labels() to maintain compatibility with server.py
    Returns a list of label strings
    """
    return get_available_labels()

# Load the model
panns_model = None

def load_panns_model():
    """
    Load the PANNs model
    
    Returns:
        bool: Whether the model was loaded successfully
    """
    global panns_model, panns_inference, is_panns_loaded
    
    try:
        # Try to load the inference helper if it's not already initialized
        if panns_inference is None:
            print("Initializing PANNs inference helper")
            panns_inference = PANNsModelInference()
            panns_inference.initialize()
        
        # Check for model file - targeting specifically the Cnn13 model
        checkpoint_path = MODEL_FN
        
        # If not found, look for alternatives
        if not os.path.exists(checkpoint_path):
            print(f"Cnn13 model checkpoint not found at {checkpoint_path}")
            
            # Look for PTH models in the models directory
            model_dir = MODEL_PATH
            print(f"Searching for alternative models in {model_dir}...")
            
            if os.path.exists(model_dir):
                # Look specifically for Cnn13 models first
                cnn13_models = [
                    file for file in os.listdir(model_dir) 
                    if (file.endswith('.pth') and 'Cnn13' in file)
                ]
                
                if cnn13_models:
                    alt_model_path = os.path.join(model_dir, cnn13_models[0])
                    print(f"Found Cnn13 model: {alt_model_path}")
                    checkpoint_path = alt_model_path
                else:
                    # Fall back to any CNN model if Cnn13 not found
                    potential_models = [
                        file for file in os.listdir(model_dir) 
                        if (file.endswith('.pth') and ('Cnn' in file or 'cnn' in file))
                    ]
                    
                    if potential_models:
                        alt_model_path = os.path.join(model_dir, potential_models[0])
                        print(f"No Cnn13 model found. Using alternative model: {alt_model_path}")
                        checkpoint_path = alt_model_path
                    else:
                        print(f"No CNN models found in {model_dir}")
                        return False
            else:
                print(f"Model directory {model_dir} does not exist")
                return False
        
        print(f"Loading PANNs model from {checkpoint_path}")
        
        try:
            # Initialize the model - explicitly using Cnn13_GMP_64x64 for better accuracy
            print("Using Cnn13_GMP_64x64 model architecture (mAP=0.42) for better accuracy")
            
            # Try to use the imported Cnn13_GMP_64x64 from models_code if available
            # If not available, fall back to the built-in Cnn13 class
            try:
                # Check if we successfully imported Cnn13_GMP_64x64 from models_code
                if 'Cnn13_GMP_64x64' in globals():
                    print("Using Cnn13_GMP_64x64 from models_code")
                    panns_model = Cnn13_GMP_64x64(classes_num=527)
                else:
                    print("Using built-in Cnn13 class")
                    panns_model = Cnn13(classes_num=527)
            except NameError:
                print("Cnn13_GMP_64x64 not found, using built-in Cnn13 class")
                panns_model = Cnn13(classes_num=527)
            
            # Load weights - use appropriate method based on file extension
            if checkpoint_path.endswith('.pth'):
                checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
                
                # Check if it's a dict containing the model state
                if isinstance(checkpoint, dict):
                    if 'model' in checkpoint:
                        print("Loading from 'model' key in checkpoint")
                        panns_model.load_state_dict(checkpoint['model'])
                    elif 'state_dict' in checkpoint:
                        print("Loading from 'state_dict' key in checkpoint")
                        panns_model.load_state_dict(checkpoint['state_dict'])
                    else:
                        print("Using checkpoint directly as state dictionary")
                        panns_model.load_state_dict(checkpoint)
                else:
                    print("Loading direct model weights")
                    panns_model.load_state_dict(checkpoint)
            else:
                # For h5 files
                print("Loading h5 model file - this may require conversion")
                
                # Simple h5 loading for PyTorch (this might need to be adapted)
                import h5py
                with h5py.File(checkpoint_path, 'r') as f:
                    for name, param in panns_model.named_parameters():
                        if name in f:
                            param.data = torch.from_numpy(f[name][()])
            
            panns_model.eval()
            print("PANNs Cnn13_GMP_64x64 model loaded successfully")
            is_panns_loaded = True
            return True
        except Exception as e:
            print(f"Error loading PANNs model: {str(e)}")
            traceback.print_exc()
            return False
    except Exception as e:
        print(f"Error in load_panns_model: {str(e)}")
        traceback.print_exc()
        return False

def predict_with_panns(audio_data, top_k=5, threshold=0.1, map_to_homesounds_format=False, boost_other_categories=False):
    """
    Run inference on audio data using the PANNs model
    
    Args:
        audio_data: Audio data as numpy array
        top_k: Number of top predictions to return
        threshold: Confidence threshold for predictions
        map_to_homesounds_format: Whether to map predictions to homesounds format
        boost_other_categories: Whether to boost non-percussion categories
    
    Returns:
        List of (label, score) tuples for the top K predictions
    """
    try:
        # Load the model if it hasn't been loaded yet
        global panns_model, panns_inference
        if panns_model is None or panns_inference is None:
            success = load_panns_model()
            if not success:
                print("Failed to load PANNs model")
                return [("Error loading model", 1.0)]
        
        if panns_model is None:
            print("PANNs model is not loaded")
            return [("Model not loaded", 1.0)]

        # Process audio
        audio_data = np.array(audio_data).astype(np.float32)
        audio_stats = {
            "mean": np.mean(audio_data),
            "std": np.std(audio_data),
            "min": np.min(audio_data),
            "max": np.max(audio_data),
            "abs_max": np.max(np.abs(audio_data))
        }
        
        print(f"Audio stats - Mean: {audio_stats['mean']:.6f}, Std: {audio_stats['std']:.6f}, "
              f"Min: {audio_stats['min']:.6f}, Max: {audio_stats['max']:.6f}, Abs Max: {audio_stats['abs_max']:.6f}")
        
        # Process the audio using the PANNsModelInference class
        x = panns_inference._preprocess_audio(audio_data)
        
        # Get model predictions
        with torch.no_grad():
            panns_model.eval()
            # Forward pass
            output = panns_model(x)
            # Get output as numpy array
            output = output.cpu().numpy()[0]
        
        # Sort predictions and create output
        sorted_indexes = np.argsort(output)[::-1]
        
        # Get labels if available
        labels_list = get_available_labels()
        if not labels_list:
            print("No labels available, using generic labels")
            labels_list = [f"Class_{i}" for i in range(len(output))]
        
        # Check if number of model outputs matches number of labels
        if len(output) != len(labels_list):
            print(f"WARNING: Model output size ({len(output)}) does not match labels size ({len(labels_list)})")
            # If model output is larger, use default labels for the extra outputs
            if len(output) > len(labels_list):
                for i in range(len(labels_list), len(output)):
                    labels_list.append(f"Unknown_{i}")
        
        # Always print the top 5 predictions for debugging
        print("Top 5 raw predictions:")
        for i in range(min(len(sorted_indexes), 5)):
            idx = sorted_indexes[i]
            score = float(output[idx])
            if idx < len(labels_list):
                label_name = labels_list[idx]
                print(f"  {i+1}. {label_name}: {score:.4f}")
            else:
                print(f"  {i+1}. Unknown_{idx}: {score:.4f}")
                
        # Format results based on whether we want homesounds format or not
        if map_to_homesounds_format:
            # Homesounds format is a dict with an "output" key containing predictions
            output_dict = {"output": []}
            predictions_collected = 0
            
            # Add predictions above threshold
            for i in range(min(len(sorted_indexes), 100)):  # Try up to 100 candidates
                idx = sorted_indexes[i]
                score = float(output[idx])
                
                if idx < len(labels_list):
                    label_name = labels_list[idx]
                    
                    # Add prediction if score is above threshold or we haven't collected enough
                    if score >= threshold or predictions_collected < top_k:
                        output_dict["output"].append({
                            "label": label_name,
                            "score": score
                        })
                        predictions_collected += 1
                        
                        # Stop if we have enough predictions above threshold
                        if predictions_collected >= top_k and score < threshold:
                            break
                else:
                    # Handle the case where idx is out of range
                    if predictions_collected < top_k:
                        label_name = f"Unknown_{idx}"
                        output_dict["output"].append({
                            "label": label_name,
                            "score": score
                        })
                        predictions_collected += 1
            
            # Convert to list of tuples for consistency with other models
            predictions = [(item["label"], item["score"]) for item in output_dict["output"]]
        else:
            # Standard format is just a list of (label, score) tuples
            predictions = []
            predictions_collected = 0
            
            for i in range(min(len(sorted_indexes), 100)):
                idx = sorted_indexes[i]
                score = float(output[idx])
                
                if idx < len(labels_list):
                    label_name = labels_list[idx]
                    
                    # Add prediction if score is above threshold or we haven't collected enough
                    if score >= threshold or predictions_collected < top_k:
                        predictions.append((label_name, score))
                        predictions_collected += 1
                        
                        # Stop if we have enough predictions above threshold
                        if predictions_collected >= top_k and score < threshold:
                            break
                else:
                    # Handle the case where idx is out of range
                    if predictions_collected < top_k:
                        label_name = f"Unknown_{idx}"
                        predictions.append((label_name, score))
                        predictions_collected += 1
        
        # If no predictions meet the threshold, include at least the top prediction
        if not predictions and len(sorted_indexes) > 0:
            idx = sorted_indexes[0]
            score = float(output[idx])
            label_name = labels_list[idx] if idx < len(labels_list) else f"Unknown_{idx}"
            predictions = [(label_name, score)]
        
        print(f"Final prediction results: {predictions}")
        
        return predictions
        
    except Exception as e:
        print(f"Error in predict_with_panns: {e}")
        traceback.print_exc()
        return [("Error", 0.0)]