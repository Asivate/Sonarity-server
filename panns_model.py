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

# Define model paths and constants
# Use a relative path with fallback to absolute path for better compatibility
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
if not os.path.exists(MODEL_DIR):
    # Fallback to the absolute path if needed
    MODEL_DIR = '/home/hirwa0250/Sonarity-server/models'

# Prioritize the larger CNN13 model first (the 1GB model)
CNN13_MODEL_PATH = os.path.join(MODEL_DIR, 'Cnn13_GMP_64x64_520000_iterations_mAP=0.42.pth')
CNN9_MODEL_PATH = os.path.join(MODEL_DIR, 'Cnn9_GMP_64x64_300000_iterations_mAP=0.37.pth')

# Check which model exists and set it as the default
if os.path.exists(CNN13_MODEL_PATH):
    print(f"Using the larger CNN13 model (1GB): {CNN13_MODEL_PATH}")
    MODEL_PATH = CNN13_MODEL_PATH
    USE_CNN13 = True
elif os.path.exists(CNN9_MODEL_PATH):
    print(f"Using the smaller CNN9 model: {CNN9_MODEL_PATH}")
    MODEL_PATH = CNN9_MODEL_PATH
    USE_CNN13 = False
else:
    print("No pre-trained PANNs models found in models directory")
    # Just set a default, we'll handle the missing file during initialization
    MODEL_PATH = CNN13_MODEL_PATH
    USE_CNN13 = True

# Define asset directory for CSV files and scalar file
ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
if not os.path.exists(ASSET_DIR):
    # Fallback to model directory if assets dir doesn't exist
    ASSET_DIR = MODEL_DIR

# Try to find scalar file in both directories
SCALAR_FN = os.path.join(ASSET_DIR, 'scalar.h5')
if not os.path.exists(SCALAR_FN):
    SCALAR_FN = os.path.join(MODEL_DIR, 'scalar.h5')

# CSV files for label mapping
CSV_FNAME = os.path.join(ASSET_DIR, 'audioset_labels.csv')
ALT_CSV_FNAME = os.path.join(ASSET_DIR, 'validate_meta.csv')
DOMESTIC_CSV_FNAME = os.path.join(ASSET_DIR, 'domestic_labels.csv')

# Fallback path for CSV files
if not os.path.exists(CSV_FNAME) and os.path.exists(os.path.join(MODEL_DIR, 'audioset_labels.csv')):
    CSV_FNAME = os.path.join(MODEL_DIR, 'audioset_labels.csv')

if not os.path.exists(ALT_CSV_FNAME) and os.path.exists(os.path.join(MODEL_DIR, 'validate_meta.csv')):
    ALT_CSV_FNAME = os.path.join(MODEL_DIR, 'validate_meta.csv')

if not os.path.exists(DOMESTIC_CSV_FNAME) and os.path.exists(os.path.join(MODEL_DIR, 'domestic_labels.csv')):
    DOMESTIC_CSV_FNAME = os.path.join(MODEL_DIR, 'domestic_labels.csv')

# Additional path for CSV files
CSV_FILES_FNAME = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'csv_files', 'validate_meta.csv')

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
            input: Tensor with shape [batch_size, channels, mel_bins, time]
        
        Returns:
            Tensor with shape [batch_size, 2048]
        """
        # Debug input shape
        logger.info(f"get_bottleneck input shape: {input.shape}")
        
        # Ensure the input has the right shape for the CNN
        # CNN13 expects [batch, channels, mel_bins, time] where time is at least 64
        if input.shape[3] < 64:
            # Pad the time dimension to at least 64 frames
            padding_needed = 64 - input.shape[3]
            logger.info(f"Padding time dimension by {padding_needed} frames to reach 64")
            padding = torch.zeros(input.shape[0], input.shape[1], input.shape[2], padding_needed, device=input.device)
            input = torch.cat([input, padding], dim=3)
        
        # Use smaller pooling sizes to preserve more spatial features
        # for small spectrograms
        
        # First conv block with reduced pooling
        x = self.conv_block1(input, pool_size=(2, 2), pool_type='avg')
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
        
        # Reshape features if needed
        if x.shape[2] * x.shape[3] < 1:
            logger.warning(f"Feature dimensions too small: {x.shape}")
            # Create a minimum viable feature map
            device = x.device
            x = torch.zeros(x.shape[0], x.shape[1], 1, 1, device=device)
            
        # Global pooling
        x = torch.mean(x, dim=3)
        logger.info(f"After mean pooling: {x.shape}")
        x = torch.mean(x, dim=2)
        logger.info(f"After global pooling: {x.shape}")
        
        # Final FC layer
        x = F.dropout(x, p=0.5)
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
            input: Tensor with shape [batch_size, channels, mel_bins, time]
        
        Returns:
            Tensor with shape [batch_size, 512]
        """
        # Debug input shape
        logger.info(f"get_bottleneck input shape: {input.shape}")
        
        # Ensure the input has the right shape for the CNN
        # CNN9 expects [batch, channels, mel_bins, time] where time is at least 64
        if input.shape[3] < 64:
            # Pad the time dimension to at least 64 frames
            padding_needed = 64 - input.shape[3]
            logger.info(f"Padding time dimension by {padding_needed} frames to reach 64")
            padding = torch.zeros(input.shape[0], input.shape[1], input.shape[2], padding_needed, device=input.device)
            input = torch.cat([input, padding], dim=3)
        
        # Use smaller pooling sizes to preserve more spatial features
        # Original model uses (2,2) pooling which reduces dimensions too quickly
        # for our small spectrograms
        
        # First conv block with reduced pooling
        x = self.conv_block1(input, pool_size=(1, 2), pool_type='avg')  # Using (1,2) instead of (2,2)
        logger.info(f"After conv_block1: {x.shape}")
        x = F.dropout(x, p=0.2)
        
        # Second conv block with reduced pooling
        x = self.conv_block2(x, pool_size=(1, 2), pool_type='avg')  # Using (1,2) instead of (2,2)
        logger.info(f"After conv_block2: {x.shape}")
        x = F.dropout(x, p=0.2)
        
        # Third conv block with smaller pooling
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg')  # Using (1,2) instead of (2,2)
        logger.info(f"After conv_block3: {x.shape}")
        x = F.dropout(x, p=0.2)
        
        # Fourth conv block with smaller pooling
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')  # Using (1,2) instead of (2,2)
        logger.info(f"After conv_block4: {x.shape}")
        x = F.dropout(x, p=0.2)
        
        # Reshape features to 512 if needed
        if x.shape[2] * x.shape[3] != 512:
            logger.info(f"Reshaping features: current shape = {x.shape}")
            
            # Flatten and resize to 512 features
            x_flat = x.view(x.shape[0], x.shape[1], -1)  # Flatten spatial dimensions
            logger.info(f"Flattened shape: {x_flat.shape}")
            
            # Ensure we have at least 512 features
            if x_flat.shape[2] < 512:
                # Upsample features to 512
                x_flat = F.interpolate(x_flat, size=512, mode='linear', align_corners=False)
                logger.info(f"Upsampled to: {x_flat.shape}")
            elif x_flat.shape[2] > 512:
                # Downsample features to 512
                x_flat = F.adaptive_avg_pool1d(x_flat, 512)
                logger.info(f"Downsampled to: {x_flat.shape}")
                
            # Convert back to expected format for further processing
            x = x_flat
        
        # Global pooling
        x = torch.mean(x, dim=2)
        logger.info(f"After global pooling: {x.shape}")
        
        # Final FC layer
        x = F.dropout(x, p=0.5)
        x = F.relu_(self.fc(x))
        logger.info(f"Final bottleneck features: {x.shape}")
        
        return x
        
    def forward(self, input):
        x = self.get_bottleneck(input)
        x = F.dropout(x, p=0.5)
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
        
    def initialize(self):
        """Initialize the PANNs model, loading weights and preparing for inference."""
        try:
            if self._initialized:
                print("PANNs model already initialized")
                return True
            
            print("Initializing PANNs model...")
            
            # Set up the mel filterbank for 32kHz audio (matching original implementation)
            self.melW = librosa.filters.mel(
                sr=32000,  # Sample rate - matching original implementation
                n_fft=1024,  # FFT window size
                n_mels=64,  # Number of mel bins
                fmin=50,    # Min frequency
                fmax=14000  # Max frequency
            )
            
            # Load the model if available
            model_path = MODEL_PATH
            if not os.path.exists(model_path):
                print(f"Model checkpoint not found at {model_path}")
                
                # Try to find any PANNs model in the models directory
                model_dir = os.path.dirname(model_path)
                print(f"Searching for alternative models in {model_dir}...")
                
                if os.path.exists(model_dir):
                    potential_models = [
                        file for file in os.listdir(model_dir) 
                        if file.endswith('.pth') and ('Cnn' in file or 'cnn' in file)
                    ]
                    
                    if potential_models:
                        # Use the first available model
                        alt_model_path = os.path.join(model_dir, potential_models[0])
                        print(f"Found alternative model: {alt_model_path}")
                        model_path = alt_model_path
                    else:
                        print(f"No alternative models found in {model_dir}")
                        return False
                else:
                    print(f"Model directory {model_dir} does not exist")
                    return False
            
            # Load labels
            if os.path.exists(CSV_FNAME):
                try:
                    df = pd.read_csv(CSV_FNAME)
                    self.labels = df['display_name'].values
                    print(f"Loaded {len(self.labels)} labels")
                except Exception as e:
                    print(f"Error loading labels: {e}")
                    self.labels = [f"label_{i}" for i in range(527)]  # Default labels
            else:
                print(f"Labels file not found at {CSV_FNAME}")
                self.labels = [f"label_{i}" for i in range(527)]  # Default labels
            
            # Load model
            try:
                # Determine model architecture based on global variable or filename
                if USE_CNN13 or 'Cnn13' in model_path or 'cnn13' in model_path.lower():
                    print("Using CNN13 model architecture")
                    # For CNN13, use the custom class defined in this file
                    self.model = Cnn13(classes_num=527)
                else:
                    # Default to CNN9
                    print("Using CNN9 model architecture")
                    self.model = Cnn9_GMP_64x64(classes_num=527)
                
                # Load weights with specialized handling for different model formats
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    print(f"Loaded checkpoint with keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dictionary'}")
                    
                    # Check for different checkpoint formats
                    if isinstance(checkpoint, dict):
                        if 'model' in checkpoint:
                            print("Loading from 'model' key in checkpoint")
                            state_dict = checkpoint['model']
                        elif 'state_dict' in checkpoint:
                            print("Loading from 'state_dict' key in checkpoint")
                            state_dict = checkpoint['state_dict']
                        else:
                            print("Using checkpoint directly as state dictionary")
                            state_dict = checkpoint
                    else:
                        print("Checkpoint is not a dictionary, cannot load model")
                        return False
                    
                    # Filter the state dict to match the model architecture
                    model_state_dict = self.model.state_dict()
                    
                    # Only load parameters that match the current model architecture
                    # This allows loading weights even if some layers don't match exactly
                    filtered_state_dict = {}
                    for name, param in state_dict.items():
                        if name in model_state_dict and param.size() == model_state_dict[name].size():
                            filtered_state_dict[name] = param
                        else:
                            print(f"Skipping parameter {name} due to size mismatch or not in model")
                    
                    # Load the filtered state dict
                    self.model.load_state_dict(filtered_state_dict, strict=False)
                    
                    # Initialize any missing parameters with default values
                    missing_keys = set(model_state_dict.keys()) - set(filtered_state_dict.keys())
                    if missing_keys:
                        print(f"The following parameters were initialized with default values: {missing_keys}")
                    
                    self.model.to(self.device)
                    self.model.eval()
                    print(f"Successfully loaded model from {model_path}")
                    
                    # Acquire mean and std for normalization
                    scalar_path = os.path.join(os.path.dirname(model_path), 'scalar.h5')
                    if os.path.exists(scalar_path):
                        with h5py.File(scalar_path, 'r') as hf:
                            self.mean = hf['mean'][:]
                            self.std = hf['std'][:]
                            print(f"Loaded normalization values from {scalar_path}")
                    else:
                        # Use default values from the original implementation
                        self.mean = self.LOGMEL_MEANS
                        self.std = self.LOGMEL_STDDEVS
                        print("Using default normalization values")
                    
                    # Create thread lock for prediction
                    self.lock = threading.Lock()
                    
                    self._initialized = True
                    print("PANNs model initialized successfully")
                    return True
                
                except Exception as e:
                    print(f"Error loading model weights: {e}")
                    print("Detailed error information:")
                    traceback.print_exc()
                    return False
            
            except Exception as e:
                print(f"Error loading model: {e}")
                traceback.print_exc()
                return False
            
        except Exception as e:
            print(f"Error initializing PANNs model: {e}")
            traceback.print_exc()
            return False
    
    def logmel_extract(self, audio):
        """
        Extract log mel spectrogram features from audio data.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Log mel spectrogram as tensor of shape (batch_size, time_steps, freq_bins)
        """
        try:
            # Audio should already be normalized to [-1, 1] and sampled at 32kHz (matching original)
            # Parameters based on original implementation
            n_fft = 1024  # FFT window size
            hop_length = 500  # Hop size - matches original implementation
            n_mels = 64  # Number of mel bins
            fmin = 50  # Minimum frequency
            fmax = 14000  # Maximum frequency
            
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
            
            # Apply mel filterbank
            mel_spec = np.dot(self.melW, power_spec)
            
            # Convert to log scale
            log_mel_spec = librosa.power_to_db(mel_spec, ref=1.0, amin=1e-10, top_db=None)
            
            # Ensure proper shape (time, freq) - transpose if needed
            if log_mel_spec.shape[0] != n_mels:
                log_mel_spec = log_mel_spec.T
            
            # Normalize with standardized values (optional - will also be done in the predict method)
            # log_mel_spec = (log_mel_spec - self.LOGMEL_MEANS.reshape(-1, 1)) / self.LOGMEL_STDDEVS.reshape(-1, 1)
        
            # We need a larger spectrogram for the CNN to extract enough features
            # PANNs model was trained on 64x500 spectrograms, but we have much smaller ones
            # Ensure spectrogram is at least 64 frames wide (time dimension)
            min_time_frames = 64
            if log_mel_spec.shape[1] < min_time_frames:
                # Calculate padding required
                pad_size = min_time_frames - log_mel_spec.shape[1]
                print(f"Padding spectrogram time dimension from {log_mel_spec.shape[1]} to {min_time_frames}")
                # Add padding to the time dimension
                log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_size)), mode='constant')
            
            # Return the log mel spectrogram
            print(f"Extracted log mel spectrogram with shape {log_mel_spec.shape}")
            return log_mel_spec
            
        except Exception as e:
            print(f"Error in logmel_extract: {e}")
            traceback.print_exc()
            # Return an empty spectrogram in case of error
            return np.zeros((64, 64), dtype=np.float32)
    
    def predict(self, audio_data, top_k=5, threshold=0.2, boost_other_categories=True):
        """
        Predict sound classes from audio data
        Args:
            audio_data: numpy array of audio samples (32000 samples/sec)
            top_k: number of top predictions to return
            threshold: minimum confidence threshold for predictions
            boost_other_categories: whether to boost non-speech categories
        Returns:
            list of (label, score) tuples
        """
        try:
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
                return []
            
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
            
            # Shape check and reshape if needed
            if x.shape[0] != 128 or x.shape[1] != 64:
                print(f"Unusual spectrogram shape: {x.shape}, expecting (128, 64)")
                if len(x) > 128:
                    x = x[:128, :]
                # Reshape if needed - we need (batch_size, time_steps, freq_bins)
            
            # Normalize with mean and std values (essential for correct predictions)
            x = (x - self.mean) / self.std
            
            # Convert to PyTorch tensor and reshape for model input
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

            # Thread safety - don't allow multiple predictions at once
            with self.lock:
                # Run inference
                with torch.no_grad():
                    self.model.eval()
                    prediction = self.model(x)
                
                # Get probabilities
                probs = prediction.squeeze().cpu().numpy()
            
            # Get top-k indices and their probabilities
            indices = np.argsort(probs)[-top_k:][::-1]
            selected_probs = probs[indices]
            
            # Filter by threshold
            mask = selected_probs >= threshold
            indices = indices[mask]
            selected_probs = selected_probs[mask]
            
            # If no predictions above threshold, return top prediction anyway
            if len(indices) == 0 and len(probs) > 0:
                indices = [np.argmax(probs)]
                selected_probs = [probs[indices[0]]]
            
            # Boost non-speech categories if requested
            if boost_other_categories:
                # This will be handled later in process_audio_with_panns
                pass
            
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
            "Knock": "household-knock",
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
            "Microwave oven": "household-microwave",
            "Blender": "household-blender",
            "Toilet flush": "household-toilet",
            "Door": "household-door",
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

# Create singleton instance
panns_inference = PANNsModelInference()

def get_available_labels():
    """
    Get the list of available sound labels from the loaded model
    Returns a list of label strings
    """
    try:
        # Try to get from the PANNs inference object first
        if hasattr(panns_inference, 'get_available_labels'):
            return panns_inference.get_available_labels()
        
        # Fallback to loading labels directly from CSV files
        labels = []
        
        # Try primary audioset labels
        if os.path.exists(CSV_FNAME):
            try:
                df = pd.read_csv(CSV_FNAME)
                if 'display_name' in df.columns:
                    labels = df['display_name'].tolist()
                    print(f"Loaded {len(labels)} labels from {CSV_FNAME}")
                    return labels
            except Exception as e:
                print(f"Error loading labels from {CSV_FNAME}: {e}")
        
        # Try alternate validate_meta.csv
        if os.path.exists(ALT_CSV_FNAME):
            try:
                df = pd.read_csv(ALT_CSV_FNAME)
                if 'display_name' in df.columns:
                    labels = df['display_name'].tolist()
                    print(f"Loaded {len(labels)} labels from {ALT_CSV_FNAME}")
                    return labels
            except Exception as e:
                print(f"Error loading labels from {ALT_CSV_FNAME}: {e}")
        
        # Try domestic labels as last resort
        if os.path.exists(DOMESTIC_CSV_FNAME):
            try:
                df = pd.read_csv(DOMESTIC_CSV_FNAME)
                if 'display_name' in df.columns:
                    labels = df['display_name'].tolist()
                    print(f"Loaded {len(labels)} labels from {DOMESTIC_CSV_FNAME}")
                    return labels
            except Exception as e:
                print(f"Error loading labels from {DOMESTIC_CSV_FNAME}: {e}")
        
        # If all else fails, return default labels
        if not labels:
            print("Could not load labels from any source, using default labels")
            labels = [f"label_{i}" for i in range(527)]
        
        return labels
    except Exception as e:
        print(f"Error in get_available_labels: {e}")
        traceback.print_exc()
        return [f"label_{i}" for i in range(527)]  # Return default labels on error

def load_panns_model():
    """Initialize the PANNs model"""
    panns_inference.initialize()

def predict_with_panns(audio_data, top_k=5, threshold=0.2, map_to_homesounds_format=False, boost_other_categories=True):
    """
    Predict audio categories using PANNs model
    
    Args:
        audio_data: Audio waveform as numpy array
        top_k: Number of top predictions to return
        threshold: Confidence threshold for predictions
        map_to_homesounds_format: Whether to map results to homesounds categories
        boost_other_categories: Whether to boost non-speech/music categories
        
    Returns:
        If map_to_homesounds_format is True: Dictionary with homesounds format
        Otherwise: List of (label, confidence) tuples
    """
    print(f"DEBUG: predict_with_panns called with top_k={top_k}, threshold={threshold}")
    print(f"DEBUG: audio_data shape: {audio_data.shape if hasattr(audio_data, 'shape') else 'Not a numpy array'}")
    print(f"DEBUG: audio_data type: {type(audio_data)}")
    try:
        # Check if model is initialized
        if not hasattr(panns_inference, '_initialized') or not panns_inference._initialized:
            print("WARNING: PANNs model not initialized, initializing now...")
            panns_inference.initialize()
            if not hasattr(panns_inference, '_initialized') or not panns_inference._initialized:
                print("ERROR: Failed to initialize PANNs model")
                return [{"label": "Model Error", "score": 1.0}]
        
        # Call the predict method with detailed error handling
        try:
            results = panns_inference.predict(audio_data, top_k=top_k, threshold=threshold, boost_other_categories=boost_other_categories)
            print(f"DEBUG: PANNs raw prediction results: {results}")
        except Exception as e:
            print(f"ERROR in panns_inference.predict: {e}")
            import traceback
            traceback.print_exc()
            return [{"label": "Prediction Error", "score": 1.0}]
        
        # Map results to homesounds format if requested
        if map_to_homesounds_format:
            try:
                mapped_results = panns_inference.map_to_homesounds(results, threshold=threshold)
                print(f"DEBUG: Mapped to homesounds format: {mapped_results}")
                return mapped_results
            except Exception as e:
                print(f"ERROR in mapping to homesounds format: {e}")
                import traceback
                traceback.print_exc()
                # Fall back to returning raw results
                return [{"label": result[0], "score": float(result[1])} for result in results] if results else [{"label": "Mapping Error", "score": 1.0}]
        
        return results
    except Exception as e:
        print(f"ERROR in predict_with_panns: {e}")
        import traceback
        traceback.print_exc()
        return [{"label": "Function Error", "score": 1.0}]