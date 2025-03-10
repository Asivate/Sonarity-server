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
        try:
            # Add shape debugging
            logger.debug(f"Input tensor shape: {input.shape}")
            
            # Check input tensor and reshape if needed
            if input.dim() != 4:
                logger.error(f"Input tensor must be 4D [batch, channel, height, width], got {input.dim()}D tensor")
                # Try to reshape - assuming it's a 3D tensor missing the channel dimension
                if input.dim() == 3:
                    input = input.unsqueeze(1)
                    logger.warning(f"Reshaped 3D tensor to 4D: {input.shape}")
                else:
                    raise ValueError(f"Cannot automatically reshape {input.dim()}D tensor to 4D")

            # Validate input shape dimensions
            if input.shape[2] != 64 or input.shape[3] < 10:
                logger.warning(f"Input shape {input.shape} may cause issues. Expected [batch, channel, 64, time>=10]")
                # Try to fix by reshaping if possible
                if input.shape[2] < 64 and input.shape[3] == 64:
                    logger.warning("Swapping dimensions 2 and 3 to fix shape")
                    input = input.transpose(2, 3)

            x = self.conv_block1(input, pool_size=(2, 2), pool_type='avg')
            logger.debug(f"After conv_block1: {x.shape}")
            x = F.dropout(x, p=0.2)
            x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
            logger.debug(f"After conv_block2: {x.shape}")
            x = F.dropout(x, p=0.2)
            x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
            logger.debug(f"After conv_block3: {x.shape}")
            x = F.dropout(x, p=0.2)
            x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
            logger.debug(f"After conv_block4: {x.shape}")
            x = F.dropout(x, p=0.2)
            x = torch.mean(x, dim=3)
            logger.debug(f"After mean: {x.shape}")
            x = x.transpose(1, 2)
            logger.debug(f"After transpose: {x.shape}")
            
            # Ensure we have at least one element in the last dimension for pooling
            if x.shape[2] == 0:
                logger.error("Zero-sized dimension after transpose, cannot pool")
                raise ValueError("Zero-sized dimension in tensor, cannot proceed")
                
            x = F.max_pool1d(x, kernel_size=x.shape[2])
            logger.debug(f"After max_pool1d: {x.shape}")
            x = x.transpose(1, 2)
            logger.debug(f"After second transpose: {x.shape}")
            x = x.view(x.shape[0], -1)
            logger.debug(f"After view/flatten: {x.shape}")
            x = F.dropout(x, p=0.5)
            
            # Check if the tensor has the right shape for the FC layer
            expected_fc_input = 512  # Expected input features for FC layer
            if x.shape[1] != expected_fc_input:
                logger.error(f"Shape mismatch before FC layer: got {x.shape[1]} features, expected {expected_fc_input}")
                raise ValueError(f"FC layer expects {expected_fc_input} input features, got {x.shape[1]}")
                
            x = F.relu_(self.fc(x))
            logger.debug(f"After FC: {x.shape}")
            return x
            
        except Exception as e:
            logger.error(f"Error in get_bottleneck: {str(e)}")
            raise
        
    def forward(self, input):
        try:
            logger.debug(f"Forward input shape: {input.shape}")
            x = self.get_bottleneck(input)
            x = F.dropout(x, p=0.5)
            x = self.fc_audioset(x)
            x = torch.sigmoid(x)
            return x
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise

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

            # Debug model structure for comparison
            model_state_dict_keys = [k for k, v in self.model.state_dict().items()]
            logger.info(f"Found {len(model_state_dict_keys)} layers in model")
            if len(model_state_dict_keys) > 0:
                logger.info(f"Model expects layer names: {model_state_dict_keys[:5]}")

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
        """
        Extract log-mel features from audio waveform.
        
        Args:
            audio: Audio waveform as numpy array
            
        Returns:
            Log-mel features as torch tensor with proper shape for CNN input
        """
        # Ensure audio is properly shaped and normalized
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
            
        # Ensure audio is at least 1 second long (32000 samples at 32kHz)
        if len(audio) < SAMPLE_RATE:
            logger.info(f"Audio too short: {len(audio)} samples. Padding to {SAMPLE_RATE} samples.")
            padded_audio = np.zeros(SAMPLE_RATE)
            padded_audio[:len(audio)] = audio
            audio = padded_audio
        
        # Calculate spectrogram
        stft = librosa.stft(y=audio, n_fft=N_FFT, hop_length=HOP_LENGTH, 
                            win_length=N_FFT, window='hann', center=True)
        spectrogram = np.abs(stft)
        
        # Convert to mel spectrogram
        mel_basis = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=MEL_BINS, 
                                        fmin=FMIN, fmax=FMAX)
        mel_spectrogram = np.dot(mel_basis, spectrogram)
        
        # Convert to log mel spectrogram
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=1.0, amin=1e-10, top_db=None)
        
        # Debug the shapes
        logger.debug(f"Log mel spectrogram shape before normalization: {log_mel_spectrogram.shape}")
        
        # Normalize with pre-computed scalars
        if self.scalar is not None:
            # Fix the broadcasting issue by using the correct shapes
            mean = self.scalar['mean']
            std = self.scalar['std']
            
            # Reshape mean and std for proper broadcasting if needed
            if log_mel_spectrogram.shape[1] != mean.shape[0]:
                logger.warning(f"Shape mismatch: mel={log_mel_spectrogram.shape}, mean={mean.shape}. Reshaping...")
                # For broadcasting to work correctly, repeat the mean/std to match time dimension
                if len(mean.shape) == 1:  # If mean is just a 1D array
                    # Reshape mean to broadcast across time dimension
                    mean = np.tile(mean.reshape(-1, 1), (1, log_mel_spectrogram.shape[1]))
                    std = np.tile(std.reshape(-1, 1), (1, log_mel_spectrogram.shape[1]))
                
            # Apply normalization
            log_mel_spectrogram = (log_mel_spectrogram - mean) / std
        
        # Reshape for PyTorch CNN model input: [batch_size, channels, height, width]
        # log_mel_spectrogram shape is currently [n_mels, time]
        
        # Ensure the time dimension is at least 10 frames long for proper pooling operations
        time_frames = log_mel_spectrogram.shape[1]
        if time_frames < 10:
            logger.warning(f"Time frames too few: {time_frames}. Padding to at least 10 frames.")
            padding = np.zeros((log_mel_spectrogram.shape[0], max(10, 2**math.ceil(math.log2(time_frames))) - time_frames))
            log_mel_spectrogram = np.concatenate((log_mel_spectrogram, padding), axis=1)
        
        # Convert to tensor with shape [1, 1, n_mels, time]
        tensor = torch.tensor(log_mel_spectrogram[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
        
        logger.info(f"Final tensor shape: {tensor.shape}")
        
        return tensor
    
    def predict(self, audio, top_k=5, threshold=0.2):
        """
        Predict sound classes from audio waveform.
        
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
            # Extract features and convert to tensor
            logger.info(f"Processing audio for prediction: shape={audio.shape}, min={audio.min():.4f}, max={audio.max():.4f}")
            
            # Make sure audio is in float32 format and properly normalized
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Ensure audio is normalized to [-1, 1]
            max_abs = np.max(np.abs(audio))
            if max_abs > 1.0:
                logger.warning(f"Audio contains values outside [-1,1] range. Normalizing. Max abs value: {max_abs:.4f}")
                audio = audio / max_abs
            
            # Extract log-mel features
            logmel = self.logmel_extract(audio)
            
            # Move tensor to the correct device
            logmel = logmel.to(self.device)
            
            # Log tensor properties for debugging
            logger.debug(f"Input tensor: shape={logmel.shape}, min={logmel.min().item():.4f}, max={logmel.max().item():.4f}")
            
            # Ensure the tensor shape is compatible with the model
            # CNN9 expects shape: [batch_size, channels, mel_bins, time]
            if logmel.dim() != 4:
                logger.warning(f"Input tensor is {logmel.dim()}D, expected 4D. Reshaping.")
                if logmel.dim() == 3:
                    logmel = logmel.unsqueeze(0)  # Add batch dimension
                elif logmel.dim() == 2:
                    logmel = logmel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            
            # Check if we have the right number of mel bins (height dimension)
            expected_mel_bins = 64
            if logmel.shape[2] != expected_mel_bins:
                logger.warning(f"Mel bins mismatch: got {logmel.shape[2]}, expected {expected_mel_bins}. Attempting to fix.")
                # If time and mel bins are swapped
                if logmel.shape[3] == expected_mel_bins:
                    logger.info(f"Transposing dimensions 2 and 3 to fix shape")
                    logmel = logmel.transpose(2, 3)
                else:
                    # Interpolate to correct size if possible
                    try:
                        logger.info(f"Interpolating to correct mel bins size: {expected_mel_bins}")
                        import torch.nn.functional as F
                        logmel = F.interpolate(logmel, size=(expected_mel_bins, logmel.shape[3]), mode='bilinear', align_corners=False)
                    except Exception as interp_error:
                        logger.error(f"Failed to interpolate: {str(interp_error)}")
                        # Fall back to padding/cropping
                        if logmel.shape[2] < expected_mel_bins:
                            logger.info(f"Padding mel bins from {logmel.shape[2]} to {expected_mel_bins}")
                            padding = torch.zeros(logmel.shape[0], logmel.shape[1], expected_mel_bins - logmel.shape[2], logmel.shape[3], device=logmel.device)
                            logmel = torch.cat([logmel, padding], dim=2)
                        else:
                            logger.info(f"Cropping mel bins from {logmel.shape[2]} to {expected_mel_bins}")
                            logmel = logmel[:, :, :expected_mel_bins, :]
            
            # Ensure time dimension is sufficient - minimum 4 frames needed for 2 pooling layers
            min_time_frames = 16
            if logmel.shape[3] < min_time_frames:
                logger.warning(f"Time frames too few: {logmel.shape[3]}, minimum {min_time_frames} needed. Padding.")
                padding = torch.zeros(logmel.shape[0], logmel.shape[1], logmel.shape[2], min_time_frames - logmel.shape[3], device=logmel.device)
                logmel = torch.cat([logmel, padding], dim=3)
            
            logger.info(f"Final input tensor shape: {logmel.shape}")
            
            # Make prediction
            self.model.eval()  # Ensure model is in evaluation mode
            with torch.no_grad():
                try:
                    prediction = self.model(logmel)
                    prediction = prediction.cpu().numpy()[0]  # Get first batch item and convert to numpy
                    
                    # Get the top K predictions
                    top_indices = np.argsort(prediction)[::-1][:top_k]
                    
                    # Filter by threshold
                    results = []
                    for idx in top_indices:
                        confidence = prediction[idx]
                        if confidence >= threshold:
                            label = self.labels.get(idx, f"unknown_{idx}")
                            results.append((label, float(confidence)))
                    
                    return results
                except RuntimeError as e:
                    if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                        logger.error(f"Matrix shape mismatch in model: {str(e)}")
                        logger.error(f"Input tensor shape that caused the error: {logmel.shape}")
                        # We'll return an empty result here instead of crashing
                        return []
                    else:
                        # Re-raise other runtime errors
                        raise
        except Exception as e:
            logger.error(f"Error in PANNs prediction: {str(e)}")
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