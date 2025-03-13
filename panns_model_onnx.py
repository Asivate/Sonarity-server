"""
PANNs Model Module with ONNX and TensorRT support for SoundWatch

This module extends the functionality of panns_model.py with support for
ONNX Runtime and TensorRT for optimized inference.
"""

import os
import sys
import numpy as np
import logging
import time
import traceback

# Import base PANNs model functionality
from panns_model import (
    get_available_labels, 
    get_labels,
    PANNsModelInference,
    SAMPLE_RATE, 
    N_FFT, 
    HOP_LENGTH, 
    MEL_BINS, 
    FMIN, 
    FMAX,
    MODEL_PATH,
    MODEL_FN
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import ONNX runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    logger.info(f"ONNX Runtime version: {ort.__version__}")
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available. Install with: pip install onnxruntime")

# Try to import TensorRT
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
    logger.info(f"TensorRT version: {trt.__version__}")
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("TensorRT not available. This is optional for NVIDIA GPUs.")

class OnnxModelInference:
    """
    ONNX Runtime inference for the PANNs model.
    This class implements the same interface as PANNsModelInference
    but uses ONNX Runtime for optimized inference.
    """
    
    def __init__(self, model_path=None):
        """Initialize the ONNX inference engine."""
        self.model_path = model_path or os.environ.get("PANNS_MODEL_PATH")
        if not self.model_path:
            # Default to standard ONNX model path
            self.model_path = os.path.join(MODEL_PATH, "Cnn13_GMP_64x64_520000_iterations_mAP=0.42.onnx")
        
        self.session = None
        self.labels = None
        self._initialized = False
        self.sample_rate = SAMPLE_RATE
        self.nfft = N_FFT
        self.hopsize = HOP_LENGTH
        self.melbins = MEL_BINS
        self.fmin = FMIN
        self.fmax = FMAX
        
        # Constants from the original implementation
        self.LOGMEL_MEANS = PANNsModelInference.LOGMEL_MEANS
        self.LOGMEL_STDDEVS = PANNsModelInference.LOGMEL_STDDEVS
    
    def initialize(self):
        """Initialize the ONNX model."""
        if self._initialized:
            logger.info("ONNX model already initialized")
            return True
            
        try:
            if not ONNX_AVAILABLE:
                logger.error("ONNX Runtime not available. Cannot initialize ONNX model.")
                return False
                
            logger.info(f"Initializing ONNX model from {self.model_path}")
            
            # Set up ONNX Runtime session
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Create ONNX Runtime session - use CUDA provider if available
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
            logger.info(f"Using ONNX Runtime providers: {providers}")
            
            # Load the model
            self.session = ort.InferenceSession(self.model_path, session_options, providers=providers)
            
            # Get model input and output details
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            # Get input shape
            input_shape = self.session.get_inputs()[0].shape
            logger.info(f"ONNX model input shape: {input_shape}")
            
            self._initialized = True
            logger.info("ONNX model initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing ONNX model: {e}")
            traceback.print_exc()
            return False
    
    def predict(self, audio_data, top_k=5, threshold=0.2, boost_other_categories=False):
        """
        Run inference using ONNX Runtime and return predictions.
        
        Args:
            audio_data: Audio data as numpy array
            top_k: Number of top predictions to return
            threshold: Confidence threshold for predictions
            boost_other_categories: Whether to boost categories other than speech/music
            
        Returns:
            List of (label, score) tuples for the top K predictions
        """
        try:
            # Process audio to get the input for the model
            x = self._preprocess_audio(audio_data)
            
            # Run inference
            ort_inputs = {self.input_name: x.astype(np.float32)}
            ort_outputs = self.session.run(None, ort_inputs)
            
            # Get probabilities
            probs = ort_outputs[0].squeeze()
            
            # Check for percussive sounds - but don't prioritize them over higher confidence predictions
            is_percussive = self._is_percussive_sound(audio_data)
            if is_percussive:
                logger.info("Percussive sound detected, but using standard prediction logic")
            
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
                    logger.warning(f"Index {i} out of bounds for labels list of length {len(self.labels)}")
            
            # Sort by probability in descending order
            result.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"ONNX prediction results: {result}")
            
            return result
                
        except Exception as e:
            logger.error(f"Error in ONNX prediction: {e}")
            traceback.print_exc()
            return []
    
    def _preprocess_audio(self, audio_data):
        """
        Preprocess audio data for ONNX model inference.
        This function replicates the preprocessing done in PANNsModelInference.
        """
        # Copy preprocessing steps from PANNsModelInference
        # Log audio statistics for debugging
        audio_mean = np.mean(audio_data)
        audio_std = np.std(audio_data)
        audio_min = np.min(audio_data)
        audio_max = np.max(audio_data)
        audio_abs_max = np.max(np.abs(audio_data))
        
        logger.info(f"Audio stats - Mean: {audio_mean:.6f}, Std: {audio_std:.6f}, Min: {audio_min:.6f}, Max: {audio_max:.6f}, Abs Max: {audio_abs_max:.6f}")
        
        # Check for valid audio data
        if audio_data is None or len(audio_data) == 0:
            logger.error("Empty audio data received")
            raise ValueError("Empty audio data received")
    
        # Check for non-finite values
        if not np.all(np.isfinite(audio_data)):
            logger.warning("Audio contains non-finite values, fixing...")
            audio_data = np.nan_to_num(audio_data)
        
        # Normalize audio if needed
        if np.max(np.abs(audio_data)) > 1.0:
            logger.info("Normalizing audio...")
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Make sure audio is long enough (at least 1 second at 32kHz = 32000 samples)
        original_length = len(audio_data)
        if original_length < 32000:
            logger.warning(f"Audio too short ({original_length} samples), padding to 32000 samples")
            # If it's very short, repeat the audio to reach minimum length
            if original_length < 16000:
                # Repeat the audio multiple times to reach 32000 samples
                repeat_count = int(np.ceil(32000 / original_length))
                audio_data = np.tile(audio_data, repeat_count)[:32000]
            else:
                # Pad with zeros to reach 32000 samples
                padding = np.zeros(32000 - original_length)
                audio_data = np.concatenate([audio_data, padding])
        
        # Extract log mel spectrogram features - replicate the logic from PANNsModelInference
        from panns_model import PANNsModelInference
        if not hasattr(self, 'panns_preprocess'):
            self.panns_preprocess = PANNsModelInference()
        
        log_mel_spec = self.panns_preprocess.logmel_extract(audio_data)
        
        # Verify that we have the correct shape for the CNN13 model
        if log_mel_spec.shape != (128, 64):
            logger.warning(f"Fixing spectrogram shape from {log_mel_spec.shape} to (128, 64)")
            # Create a properly sized spectrogram
            fixed_spec = np.zeros((128, 64), dtype=np.float32)
            
            # Copy what we can from the original spectrogram
            h = min(log_mel_spec.shape[0], 128)
            w = min(log_mel_spec.shape[1], 64)
            fixed_spec[:h, :w] = log_mel_spec[:h, :w]
            
            log_mel_spec = fixed_spec
        
        # Add batch dimension: (time_steps, freq_bins) -> (1, time_steps, freq_bins)
        # ONNX models often expect a specific input shape
        x = np.expand_dims(log_mel_spec, axis=0)
        logger.info(f"Input array shape: {x.shape}")
        
        return x
    
    def _is_percussive_sound(self, audio_data):
        """
        Analyzes the audio to determine if it's likely a percussive sound.
        This is a direct copy from PANNsModelInference for compatibility.
        """
        from panns_model import PANNsModelInference
        if not hasattr(self, 'panns_preprocess'):
            self.panns_preprocess = PANNsModelInference()
        
        return self.panns_preprocess._is_percussive_sound(audio_data)
    
    def set_labels(self, labels):
        """Set the available labels for the model."""
        self.labels = labels
        
    def get_available_labels(self):
        """Get the list of available labels for this model."""
        return self.labels

class TensorRTModelInference:
    """
    TensorRT inference for the PANNs model.
    This class implements the same interface as PANNsModelInference
    but uses TensorRT for optimized inference on NVIDIA GPUs.
    """
    
    def __init__(self, engine_path=None):
        """Initialize the TensorRT inference engine."""
        self.engine_path = engine_path or os.environ.get("PANNS_MODEL_PATH")
        if not self.engine_path:
            # Default to standard TensorRT engine path
            self.engine_path = os.path.join(MODEL_PATH, "cnn13.engine")
        
        self.engine = None
        self.context = None
        self.bindings = []
        self.inputs = []
        self.outputs = []
        self.labels = None
        self._initialized = False
        self.sample_rate = SAMPLE_RATE
        
        # Constants from the original implementation
        self.LOGMEL_MEANS = PANNsModelInference.LOGMEL_MEANS
        self.LOGMEL_STDDEVS = PANNsModelInference.LOGMEL_STDDEVS
    
    def initialize(self):
        """Initialize the TensorRT engine."""
        if self._initialized:
            logger.info("TensorRT engine already initialized")
            return True
            
        try:
            if not TENSORRT_AVAILABLE:
                logger.error("TensorRT not available. Cannot initialize TensorRT engine.")
                return False
                
            logger.info(f"Initializing TensorRT engine from {self.engine_path}")
            
            # Create TensorRT runtime
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            
            # Load the engine
            with open(self.engine_path, 'rb') as f:
                engine_data = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_data)
            
            # Create execution context
            self.context = self.engine.create_execution_context()
            
            # Allocate memory for inputs and outputs
            self.bindings = []
            self.inputs = []
            self.outputs = []
            
            for binding_idx in range(self.engine.num_bindings):
                binding_name = self.engine.get_binding_name(binding_idx)
                binding_shape = self.engine.get_binding_shape(binding_idx)
                binding_dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))
                
                # Allocate memory for this binding
                binding_size = trt.volume(binding_shape) * binding_dtype.itemsize
                device_mem = cuda.mem_alloc(binding_size)
                
                # Append to bindings
                self.bindings.append(device_mem)
                
                # Keep track of inputs and outputs
                if self.engine.binding_is_input(binding_idx):
                    self.inputs.append({
                        'name': binding_name,
                        'shape': binding_shape,
                        'dtype': binding_dtype,
                        'binding_idx': binding_idx
                    })
                else:
                    self.outputs.append({
                        'name': binding_name,
                        'shape': binding_shape,
                        'dtype': binding_dtype,
                        'binding_idx': binding_idx
                    })
            
            logger.info(f"TensorRT input shapes: {[inp['shape'] for inp in self.inputs]}")
            logger.info(f"TensorRT output shapes: {[out['shape'] for out in self.outputs]}")
            
            self._initialized = True
            logger.info("TensorRT engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing TensorRT engine: {e}")
            traceback.print_exc()
            return False
    
    def predict(self, audio_data, top_k=5, threshold=0.2, boost_other_categories=False):
        """
        Run inference using TensorRT and return predictions.
        
        Args:
            audio_data: Audio data as numpy array
            top_k: Number of top predictions to return
            threshold: Confidence threshold for predictions
            boost_other_categories: Whether to boost categories other than speech/music
            
        Returns:
            List of (label, score) tuples for the top K predictions
        """
        try:
            # Process audio to get the input for the model
            x = self._preprocess_audio(audio_data)
            
            # Create output buffer
            outputs = []
            for output in self.outputs:
                output_shape = output['shape']
                output_dtype = output['dtype']
                outputs.append(np.zeros(output_shape, dtype=output_dtype))
            
            # Copy input data to device
            input_idx = self.inputs[0]['binding_idx']
            cuda.memcpy_htod(self.bindings[input_idx], x.astype(np.float32).ravel())
            
            # Run inference
            self.context.execute_v2(self.bindings)
            
            # Copy outputs from device to host
            for i, output in enumerate(self.outputs):
                output_idx = output['binding_idx']
                cuda.memcpy_dtoh(outputs[i], self.bindings[output_idx])
            
            # Get probabilities
            probs = outputs[0].squeeze()
            
            # Check for percussive sounds - but don't prioritize them over higher confidence predictions
            is_percussive = self._is_percussive_sound(audio_data)
            if is_percussive:
                logger.info("Percussive sound detected, but using standard prediction logic")
            
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
                    logger.warning(f"Index {i} out of bounds for labels list of length {len(self.labels)}")
            
            # Sort by probability in descending order
            result.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"TensorRT prediction results: {result}")
            
            return result
                
        except Exception as e:
            logger.error(f"Error in TensorRT prediction: {e}")
            traceback.print_exc()
            return []
    
    def _preprocess_audio(self, audio_data):
        """
        Preprocess audio data for TensorRT inference.
        This function replicates the preprocessing done in PANNsModelInference.
        """
        # For consistency, we'll reuse the preprocessing from ONNX
        onnx_preprocess = OnnxModelInference()
        return onnx_preprocess._preprocess_audio(audio_data)
    
    def _is_percussive_sound(self, audio_data):
        """
        Analyzes the audio to determine if it's likely a percussive sound.
        This is a direct copy from PANNsModelInference for compatibility.
        """
        from panns_model import PANNsModelInference
        if not hasattr(self, 'panns_preprocess'):
            self.panns_preprocess = PANNsModelInference()
        
        return self.panns_preprocess._is_percussive_sound(audio_data)
    
    def set_labels(self, labels):
        """Set the available labels for the model."""
        self.labels = labels
        
    def get_available_labels(self):
        """Get the list of available labels for this model."""
        return self.labels

def create_inference_engine(model_type=None, model_path=None):
    """
    Create an optimized inference engine based on the model type.
    
    Args:
        model_type: Type of model ('pytorch', 'onnx', 'tensorrt')
        model_path: Path to the model file
        
    Returns:
        Inference engine object with the same interface as PANNsModelInference
    """
    # Get model type from environment if not provided
    if model_type is None:
        model_type = os.environ.get("PANNS_MODEL_TYPE", "pytorch").lower()
    
    # Get model path from environment if not provided
    if model_path is None:
        model_path = os.environ.get("PANNS_MODEL_PATH")
    
    # Create appropriate inference engine
    if model_type == "onnx" and ONNX_AVAILABLE:
        logger.info(f"Creating ONNX inference engine with model: {model_path}")
        inference_engine = OnnxModelInference(model_path)
    elif model_type == "tensorrt" and TENSORRT_AVAILABLE:
        logger.info(f"Creating TensorRT inference engine with model: {model_path}")
        inference_engine = TensorRTModelInference(model_path)
    else:
        # Fall back to PyTorch inference
        logger.info(f"Creating PyTorch inference engine with model: {model_path}")
        from panns_model import panns_inference, load_panns_model
        
        # Set custom model path if provided
        if model_path and model_path != MODEL_FN:
            # Update model path in panns_model
            global MODEL_FN
            MODEL_FN = model_path
        
        # Load the model
        if not load_panns_model():
            logger.error("Failed to load PyTorch model")
            return None
            
        inference_engine = panns_inference
    
    # Initialize the inference engine
    if not inference_engine.initialize():
        logger.error(f"Failed to initialize {model_type} inference engine")
        return None
    
    # Set labels
    labels = get_labels()
    if hasattr(inference_engine, 'set_labels'):
        inference_engine.set_labels(labels)
    
    return inference_engine

def predict_with_optimized_model(audio_data, top_k=5, threshold=0.1, map_to_homesounds_format=False, boost_other_categories=False):
    """
    Run inference on audio data using the optimized model.
    This function provides the same interface as predict_with_panns from panns_model.py
    but uses the optimized inference engine.
    
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
        # Create inference engine
        inference_engine = create_inference_engine()
        if inference_engine is None:
            logger.error("Failed to create inference engine")
            return [("Error", 0.0)]
        
        # Convert audio data to numpy array
        audio_data = np.array(audio_data).astype(np.float32)
        audio_stats = {
            "mean": np.mean(audio_data),
            "std": np.std(audio_data),
            "min": np.min(audio_data),
            "max": np.max(audio_data),
            "abs_max": np.max(np.abs(audio_data))
        }
        
        logger.info(f"Audio stats - Mean: {audio_stats['mean']:.6f}, Std: {audio_stats['std']:.6f}, "
                   f"Min: {audio_stats['min']:.6f}, Max: {audio_stats['max']:.6f}, Abs Max: {audio_stats['abs_max']:.6f}")
        
        # Run prediction with optimized model
        predictions = inference_engine.predict(audio_data, top_k=top_k, threshold=threshold, boost_other_categories=boost_other_categories)
        
        # Map to homesounds format if requested
        if map_to_homesounds_format and hasattr(inference_engine, 'map_to_homesounds'):
            mapped_predictions = inference_engine.map_to_homesounds(predictions, threshold=threshold)
            return mapped_predictions
        
        # If no predictions meet the threshold, include at least the top prediction
        if not predictions and len(audio_data) > 0:
            predictions = [("No prediction", 0.0)]
        
        logger.info(f"Final prediction results with optimized model: {predictions}")
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error in predict_with_optimized_model: {e}")
        traceback.print_exc()
        return [("Error", 0.0)] 