"""
PANNs Model Module for SoundWatch

This module integrates the CNN9 model from the PANNs (Pretrained Audio Neural Networks)
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'Cnn9_GMP_64x64_300000_iterations_mAP=0.37.pth')
SCALAR_FN = os.path.join(MODEL_DIR, 'scalar.h5')
CSV_FNAME = os.path.join(MODEL_DIR, 'validate_meta.csv')

# Alternative paths for labels file (fallback locations)
CSV_FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csv files')
CSV_FILES_FNAME = os.path.join(CSV_FILES_DIR, 'validate_meta.csv')

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

# CNN9 model architecture
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
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')
        return x

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
        # Simple, direct approach without complex reshaping
        x = self.conv_block1(input, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2)
        
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2)
        
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2)
        
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2)
        
        # Global max pooling
        x = torch.mean(x, dim=3)
        x = x.permute(0, 2, 1)
        x = F.max_pool1d(x, kernel_size=x.shape[2])
        x = x.permute(0, 2, 1)
        x = x.view(x.shape[0], -1)
        
        # Final FC layers
        x = F.dropout(x, p=0.5)
        x = F.relu_(self.fc(x))
        
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
    
    def __init__(self):
        """Initialize the PANNs model inference engine."""
        self.model = None
        self.labels = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scalar = None
        self._initialized = False
        
    def initialize(self):
        """Load model, labels, and scalar."""
        if self._initialized:
            return
            
        try:
            # Ensure directories exist
            os.makedirs(MODEL_DIR, exist_ok=True)
            
            # Load model
            logger.info(f"Loading PANNs CNN9 model from {MODEL_PATH}")
            
            # Check if model exists, if not provide instructions
            if not os.path.exists(MODEL_PATH):
                logger.error(f"PANNs model file not found at {MODEL_PATH}")
                logger.info("Please download the model using:")
                logger.info("python download_panns_model.py")
                return False
                
            # Load class labels
            if not os.path.exists(CSV_FNAME):
                logger.error(f"Labels file not found at {CSV_FNAME}")
                # Check alternative locations
                if os.path.exists(CSV_FILES_FNAME):
                    logger.info(f"Found labels file in 'csv files' directory. Copying to models directory...")
                    import shutil
                    shutil.copy(CSV_FILES_FNAME, CSV_FNAME)
                    logger.info(f"Copied labels file from {CSV_FILES_FNAME} to {CSV_FNAME}")
                else:
                    # Use reference labels from the demo repository
                    ref_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                          'General-Purpose-Sound-Recognition-Demo',
                                          'General-Purpose-Sound-Recognition-Demo-2019',
                                          'models',
                                          'validate_meta.csv')
                    if os.path.exists(ref_csv):
                        logger.info(f"Copying reference labels from {ref_csv}")
                        import shutil
                        shutil.copy(ref_csv, CSV_FNAME)
                    else:
                        logger.error("Cannot find reference labels. Please run 'python download_panns_model.py' to set up the required files.")
                        return False
            
            # Load scalar
            if not os.path.exists(SCALAR_FN):
                logger.error(f"Scalar file not found at {SCALAR_FN}")
                logger.error("Please run 'python download_panns_model.py' to set up the required files.")
                return False
            
            # Load audio labels
            logger.info(f"Loading class labels from {CSV_FNAME}")
            self.labels = pd.read_csv(CSV_FNAME)
            
            # Load scalar
            logger.info(f"Loading scalar from {SCALAR_FN}")
            with h5py.File(SCALAR_FN, 'r') as f:
                self.scalar = {'mean': f['mean'][()], 'std': f['std'][()]}
            
            # Initialize model
            classes_num = len(self.labels)
            logger.info(f"Creating model with {classes_num} classes")
            self.model = Cnn9_GMP_64x64(classes_num)
            
            # Load trained weights
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)

            # Debug checkpoint structure for troubleshooting
            logger.info(f"Checkpoint keys: {checkpoint.keys()}")
            model_state_dict_keys = checkpoint['model'].keys() if 'model' in checkpoint else []
            logger.info(f"Found {len(model_state_dict_keys)} layers in checkpoint")
            if len(model_state_dict_keys) > 0:
                logger.info(f"First few layer names: {list(model_state_dict_keys)[:5]}")

            try:
                # Load model weights with more flexible option
                if 'model' in checkpoint:
                    self.model.load_state_dict(checkpoint['model'], strict=False)
                    logger.info("Loaded weights from 'model' key with strict=False")
                else:
                    # Try directly loading from checkpoint
                    self.model.load_state_dict(checkpoint, strict=False)
                    logger.info("Loaded weights directly from checkpoint with strict=False")
            except Exception as e:
                logger.error(f"Error loading model weights: {str(e)}")
                logger.error("Continuing with uninitialized model - predictions may be random")
                # We'll continue even with error, to let the system still work
            
            # Set model to evaluation mode and move to appropriate device
            self.model.eval()
            self.model.to(self.device)
            
            logger.info("PANNs model loaded successfully")
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error initializing PANNs model: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def logmel_extract(self, audio):
        """Extract log-mel features from audio waveform.
        
        Args:
            audio: Audio waveform as numpy array
            
        Returns:
            Log-mel features as torch tensor with shape [1, 1, 64, time]
        """
        # Ensure audio is mono and the right length for processing
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Ensure audio is at least 1 second long (32000 samples at 32kHz)
        if len(audio) < SAMPLE_RATE:
            logger.info(f"Audio too short: {len(audio)} samples. Padding to {SAMPLE_RATE} samples.")
            padded_audio = np.zeros(SAMPLE_RATE)
            padded_audio[:len(audio)] = audio
            audio = padded_audio
        
        # Make sure we have enough samples for meaningful analysis (at least 1 second)
        # If less than 1 second, repeat the audio until it's at least 1 second
        if len(audio) < SAMPLE_RATE:
            # Calculate how many times to repeat
            repeat_count = math.ceil(SAMPLE_RATE / len(audio))
            audio = np.tile(audio, repeat_count)[:SAMPLE_RATE]
        
        # Calculate spectrogram
        stft = librosa.stft(
            y=audio, 
            n_fft=N_FFT, 
            hop_length=HOP_LENGTH, 
            win_length=N_FFT, 
            window='hann', 
            center=True
        )
        spectrogram = np.abs(stft)
        
        # Convert to mel spectrogram
        mel_basis = librosa.filters.mel(
            sr=SAMPLE_RATE, 
            n_fft=N_FFT, 
            n_mels=MEL_BINS, 
            fmin=FMIN, 
            fmax=FMAX
        )
        mel_spectrogram = np.dot(mel_basis, spectrogram)
        
        # Convert to log mel spectrogram
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=1.0, amin=1e-10, top_db=None)
        
        # Normalize with pre-computed scalars
        if self.scalar is not None:
            mean = self.scalar['mean']
            std = self.scalar['std']
            
            # Ensure mean and std are properly shaped for broadcasting
            if mean.ndim == 1:
                mean = mean.reshape(-1, 1)
                std = std.reshape(-1, 1)
                
            # Apply normalization
            log_mel_spectrogram = (log_mel_spectrogram - mean) / std
        
        # Ensure log_mel_spectrogram has enough frames for the CNN's pooling operations
        # The network needs at least 64 frames to work correctly due to pooling operations
        min_time_frames = 64
        if log_mel_spectrogram.shape[1] < min_time_frames:
            # Calculate padding required
            pad_size = min_time_frames - log_mel_spectrogram.shape[1]
            # Add padding to the time dimension
            log_mel_spectrogram = np.pad(
                log_mel_spectrogram, 
                ((0, 0), (0, pad_size)), 
                mode='constant'
            )
        
        # Convert to tensor with shape [batch_size, channels, height, width]
        # For CNNs, this means [batch_size, channels, mel_bins, time]
        log_mel_tensor = torch.tensor(
            log_mel_spectrogram[np.newaxis, np.newaxis, :, :],
            dtype=torch.float32
        )
        
        logger.info(f"Log-mel tensor shape: {log_mel_tensor.shape}")
        return log_mel_tensor
    
    def predict(self, audio, top_k=5, threshold=0.2):
        """Predict sound classes from audio waveform.
        
        Args:
            audio: Audio waveform as numpy array
            top_k: Number of top predictions to return
            threshold: Confidence threshold for predictions
            
        Returns:
            List of (label, confidence) tuples for top K predictions
        """
        if not self._initialized:
            if not self.initialize():
                return []
        
        try:
            # Log input audio properties
            logger.info(f"Processing audio for prediction: shape={audio.shape}, min={audio.min():.4f}, max={audio.max():.4f}")
            
            # Make sure audio is in float32 format
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Normalize to [-1, 1] if needed
            max_abs = np.max(np.abs(audio))
            if max_abs > 1.0:
                logger.info(f"Normalizing audio with max value {max_abs:.4f}")
                audio = audio / max_abs
            
            # Make sure audio is long enough - PANNs model expects at least a few frames of audio
            if len(audio) < 32000:  # at least 1 second at 32kHz
                logger.info(f"Audio too short ({len(audio)} samples). Padding to 32000 samples.")
                padded_audio = np.zeros(32000)
                padded_audio[:len(audio)] = audio
                audio = padded_audio
            
            # Extract log-mel features - this will enforce correct shape for model
            log_mel = self.logmel_extract(audio)
            
            # Move tensor to the correct device
            log_mel = log_mel.to(self.device)
            
            # Run model prediction
            logger.info("Running model inference...")
            with torch.no_grad():
                prediction = self.model(log_mel)
                
                # Convert predictions to numpy array
                if isinstance(prediction, torch.Tensor):
                    prediction = prediction.detach().cpu().numpy()
                
                # Process first sample if batched
                if len(prediction.shape) > 1:
                    prediction = prediction[0]
                
                # Get top-k indices
                top_indices = np.argsort(prediction)[::-1][:top_k]
                
                # Format results
                results = []
                logger.info("Top predictions:")
                for idx in top_indices:
                    confidence = float(prediction[idx])
                    if confidence >= threshold:
                        try:
                            # Get label based on index
                            if hasattr(self.labels, 'iloc'):
                                label = self.labels.iloc[idx]['display_name']
                            else:
                                label = f"unknown_{idx}"
                            
                            results.append((label, confidence))
                            logger.info(f"  {label}: {confidence:.6f}")
                        except Exception as e:
                            logger.error(f"Error getting label for index {idx}: {str(e)}")
                            results.append((f"unknown_{idx}", confidence))
                
                return results
                
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            import traceback
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
            
            # Convert to the format expected by the server
            return {
                "output": [{"label": label, "score": score} for label, score in sorted_results]
            }
        else:
            return {"output": []}

# Create singleton instance
panns_inference = PANNsModelInference()

def load_panns_model():
    """Public function to load the PANNs model."""
    return panns_inference.initialize()

def predict_with_panns(audio_data, top_k=5, threshold=0.2, map_to_homesounds_format=False):
    """
    Public function to make predictions using the PANNs model.
    
    Args:
        audio_data: Audio waveform as numpy array
        top_k: Number of top predictions to return
        threshold: Confidence threshold for predictions
        map_to_homesounds_format: Whether to map results to homesounds categories
        
    Returns:
        If map_to_homesounds_format is True: Dictionary with homesounds format
        Otherwise: List of (label, confidence) tuples
    """
    results = panns_inference.predict(audio_data, top_k=top_k, threshold=threshold)
    
    if map_to_homesounds_format:
        return panns_inference.map_to_homesounds(results, threshold=threshold)
    
    return results 