#!/usr/bin/env python3
"""
Sound Recognition Model Optimizer

This script helps optimize the CNN13 model for faster inference by:
1. Exporting PyTorch models to ONNX format
2. Quantizing ONNX models for reduced memory and improved performance 
3. Benchmarking performance improvements

Usage:
    # Export model to ONNX
    python model_optimizer.py --export-onnx
    
    # Quantize an ONNX model
    python model_optimizer.py --quantize
    
    # Run benchmarks on different model formats
    python model_optimizer.py --benchmark
"""

import os
import sys
import argparse
import time
import logging
import numpy as np
import torch
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model_optimizer")

# Import constants from panns_model
try:
    from panns_model import (
        MODEL_PATH, 
        MODEL_FN,
        PANNsModelInference, 
        SAMPLE_RATE
    )
except ImportError:
    logger.error("Failed to import from panns_model. Make sure you're running this script from the server directory.")
    MODEL_PATH = "models"
    MODEL_FN = "Cnn13_GMP_64x64_520000_iterations_mAP=0.42.pth"
    SAMPLE_RATE = 32000

# Check for ONNX
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available. Install with: pip install onnx onnxruntime")

# Check for TensorRT 
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("TensorRT not available. This is optional for NVIDIA GPUs.")

def load_model(model_path=None):
    """
    Load a PyTorch CNN13 model from the specified path.
    
    Args:
        model_path: Path to the model checkpoint
        
    Returns:
        Loaded PyTorch model
    """
    if model_path is None:
        model_path = os.path.join(MODEL_PATH, MODEL_FN)
    
    logger.info(f"Loading PyTorch model from {model_path}")
    
    try:
        # Import PANNs inference code here to avoid circular imports
        import torch
        from panns_inference.models import Cnn14, Cnn10, Cnn14_DecisionLevelMax
        
        # Detect device
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        logger.info(f"Using device: {device}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Determine model type from checkpoint
        if 'model' in checkpoint:
            model = checkpoint['model']
            logger.info("Loaded model directly from checkpoint")
        else:
            # Try to infer model type from filename
            if 'cnn13' in model_path.lower():
                # CNN13 is just CNN14 with one less layer
                model = Cnn14(sample_rate=SAMPLE_RATE, 
                           window_size=1024, 
                           hop_size=320, 
                           mel_bins=64, 
                           fmin=50, 
                           fmax=14000, 
                           classes_num=527)
                model.load_state_dict(checkpoint)
                logger.info("Created CNN13/14 model and loaded state dict")
            elif 'cnn10' in model_path.lower():
                model = Cnn10(sample_rate=SAMPLE_RATE, 
                           window_size=1024, 
                           hop_size=320, 
                           mel_bins=64, 
                           fmin=50, 
                           fmax=14000, 
                           classes_num=527)
                model.load_state_dict(checkpoint)
                logger.info("Created CNN10 model and loaded state dict")
            else:
                # Default to CNN14
                model = Cnn14(sample_rate=SAMPLE_RATE, 
                           window_size=1024, 
                           hop_size=320, 
                           mel_bins=64, 
                           fmin=50, 
                           fmax=14000, 
                           classes_num=527)
                model.load_state_dict(checkpoint)
                logger.info("Created default CNN14 model and loaded state dict")
        
        # Move model to device and set to evaluation mode
        model = model.to(device)
        model.eval()
        logger.info("Model loaded successfully")
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        traceback.print_exc()
        return None

def export_to_onnx(model=None, model_path=None, output_path=None):
    """
    Export a PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to export (if None, it will be loaded from model_path)
        model_path: Path to the PyTorch model checkpoint
        output_path: Path to save the ONNX model
        
    Returns:
        Path to the exported ONNX model
    """
    if not ONNX_AVAILABLE:
        logger.error("ONNX is not available. Install with: pip install onnx onnxruntime")
        return None
        
    try:
        # Load model if not provided
        if model is None:
            model = load_model(model_path)
            if model is None:
                return None
        
        # Set default output path if not provided
        if output_path is None:
            if model_path is None:
                model_path = os.path.join(MODEL_PATH, MODEL_FN)
            output_path = os.path.join(MODEL_PATH, Path(model_path).stem + ".onnx")
        
        logger.info(f"Exporting PyTorch model to ONNX: {output_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create dummy input tensor - spectrogram shape (batch_size, channels, time, freq)
        # The CNN13 model expects input shape (batch_size, time_steps, freq_bins)
        dummy_input = torch.randn(1, 1, 500, 64, device=model.device)
        
        # Export the model
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'time_steps'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify the ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        logger.info(f"ONNX model exported successfully to {output_path}")
        logger.info(f"ONNX model graph input: {onnx_model.graph.input}")
        logger.info(f"ONNX model graph output: {onnx_model.graph.output}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error exporting to ONNX: {e}")
        traceback.print_exc()
        return None

def quantize_onnx_model(model_path=None, output_path=None):
    """
    Quantize an ONNX model to INT8 for reduced memory usage and improved performance.
    
    Args:
        model_path: Path to the ONNX model to quantize
        output_path: Path to save the quantized model
        
    Returns:
        Path to the quantized ONNX model
    """
    if not ONNX_AVAILABLE:
        logger.error("ONNX is not available. Install with: pip install onnx onnxruntime")
        return None
    
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        # Set default paths if not provided
        if model_path is None:
            model_path = os.path.join(MODEL_PATH, Path(MODEL_FN).stem + ".onnx")
        
        if output_path is None:
            output_path = os.path.join(MODEL_PATH, Path(model_path).stem + "_quantized.onnx")
        
        logger.info(f"Quantizing ONNX model from {model_path} to {output_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Quantize the model
        quantize_dynamic(
            model_path,
            output_path,
            weight_type=QuantType.QInt8
        )
        
        logger.info(f"ONNX model quantized successfully to {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error quantizing ONNX model: {e}")
        traceback.print_exc()
        return None

def optimize_onnx_model(model_path=None, output_path=None):
    """
    Optimize an ONNX model using ONNX Runtime's graph optimizations.
    
    Args:
        model_path: Path to the ONNX model to optimize
        output_path: Path to save the optimized model
        
    Returns:
        Path to the optimized ONNX model
    """
    if not ONNX_AVAILABLE:
        logger.error("ONNX is not available. Install with: pip install onnx onnxruntime")
        return None
    
    try:
        # Set default paths if not provided
        if model_path is None:
            model_path = os.path.join(MODEL_PATH, Path(MODEL_FN).stem + ".onnx")
        
        if output_path is None:
            output_path = os.path.join(MODEL_PATH, Path(model_path).stem + "_optimized.onnx")
        
        logger.info(f"Optimizing ONNX model from {model_path} to {output_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Load model
        model = onnx.load(model_path)
        
        # Optimize the model
        from onnxruntime.transformers import optimizer
        opt_model = optimizer.optimize_model(
            model_path,
            model_type='conv',
            num_heads=0,
            hidden_size=0,
            optimization_level=99
        )
        opt_model.save_model_to_file(output_path)
        
        logger.info(f"ONNX model optimized successfully to {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error optimizing ONNX model: {e}")
        traceback.print_exc()
        return None

def create_tensorrt_engine(model_path=None, output_path=None):
    """
    Create a TensorRT engine from an ONNX model.
    
    Args:
        model_path: Path to the ONNX model
        output_path: Path to save the TensorRT engine
        
    Returns:
        Path to the TensorRT engine
    """
    if not TENSORRT_AVAILABLE:
        logger.error("TensorRT is not available. This is optional for NVIDIA GPUs.")
        return None
    
    try:
        # Set default paths if not provided
        if model_path is None:
            model_path = os.path.join(MODEL_PATH, Path(MODEL_FN).stem + ".onnx")
        
        if output_path is None:
            output_path = os.path.join(MODEL_PATH, Path(model_path).stem + ".engine")
        
        logger.info(f"Creating TensorRT engine from {model_path} to {output_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create a TensorRT builder
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        
        # Parse the ONNX model
        parser = trt.OnnxParser(network, TRT_LOGGER)
        with open(model_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                logger.error("Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                return None
        
        # Create a config for the builder
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 28  # 256 MiB
        
        # Set precision (FP16 if available)
        if builder.platform_has_fast_fp16:
            logger.info("Using FP16 precision")
            config.set_flag(trt.BuilderFlag.FP16)
        
        # Build the engine
        serialized_engine = builder.build_serialized_network(network, config)
        
        # Save the engine
        with open(output_path, 'wb') as f:
            f.write(serialized_engine)
        
        logger.info(f"TensorRT engine created successfully at {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating TensorRT engine: {e}")
        traceback.print_exc()
        return None

def benchmark_models(pytorch_model_path=None, onnx_model_path=None, 
                    quantized_model_path=None, tensorrt_engine_path=None,
                    num_iterations=100):
    """
    Benchmark inference speed for different model formats.
    
    Args:
        pytorch_model_path: Path to PyTorch model
        onnx_model_path: Path to ONNX model
        quantized_model_path: Path to quantized ONNX model
        tensorrt_engine_path: Path to TensorRT engine
        num_iterations: Number of inference runs for benchmarking
        
    Returns:
        Dictionary of benchmark results
    """
    results = {}
    
    # Generate random audio data for benchmarking
    audio_data = np.random.randn(SAMPLE_RATE).astype(np.float32)  # 1 second of audio
    
    # Create preprocessing instance
    preprocess = PANNsModelInference()
    
    # Benchmark PyTorch model
    if pytorch_model_path is not None:
        try:
            logger.info(f"Benchmarking PyTorch model: {pytorch_model_path}")
            
            # Load the model
            model = load_model(pytorch_model_path)
            if model is None:
                logger.error("Failed to load PyTorch model for benchmarking")
            else:
                # Preprocess audio
                device = model.device
                x = preprocess.logmel_extract(audio_data)
                x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)
                
                # Warm-up runs
                for _ in range(10):
                    with torch.no_grad():
                        model(x)
                
                # Benchmarking runs
                start_time = time.time()
                for _ in range(num_iterations):
                    with torch.no_grad():
                        model(x)
                end_time = time.time()
                
                # Calculate average inference time
                avg_time = (end_time - start_time) / num_iterations
                results['pytorch'] = {
                    'avg_inference_time': avg_time,
                    'inferences_per_second': 1.0 / avg_time
                }
                
                logger.info(f"PyTorch model: {avg_time*1000:.2f} ms per inference, {1.0/avg_time:.2f} inferences/second")
                
        except Exception as e:
            logger.error(f"Error benchmarking PyTorch model: {e}")
    
    # Benchmark ONNX model
    if ONNX_AVAILABLE and onnx_model_path is not None:
        try:
            logger.info(f"Benchmarking ONNX model: {onnx_model_path}")
            
            # Create ONNX session
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
            session = ort.InferenceSession(onnx_model_path, session_options, providers=providers)
            
            # Get input name
            input_name = session.get_inputs()[0].name
            
            # Preprocess audio
            x = preprocess.logmel_extract(audio_data)
            x = np.expand_dims(x, axis=0).astype(np.float32)
            
            # Warm-up runs
            for _ in range(10):
                session.run(None, {input_name: x})
            
            # Benchmarking runs
            start_time = time.time()
            for _ in range(num_iterations):
                session.run(None, {input_name: x})
            end_time = time.time()
            
            # Calculate average inference time
            avg_time = (end_time - start_time) / num_iterations
            results['onnx'] = {
                'avg_inference_time': avg_time,
                'inferences_per_second': 1.0 / avg_time
            }
            
            logger.info(f"ONNX model: {avg_time*1000:.2f} ms per inference, {1.0/avg_time:.2f} inferences/second")
            
        except Exception as e:
            logger.error(f"Error benchmarking ONNX model: {e}")
    
    # Benchmark quantized ONNX model
    if ONNX_AVAILABLE and quantized_model_path is not None:
        try:
            logger.info(f"Benchmarking quantized ONNX model: {quantized_model_path}")
            
            # Create ONNX session
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
            session = ort.InferenceSession(quantized_model_path, session_options, providers=providers)
            
            # Get input name
            input_name = session.get_inputs()[0].name
            
            # Preprocess audio
            x = preprocess.logmel_extract(audio_data)
            x = np.expand_dims(x, axis=0).astype(np.float32)
            
            # Warm-up runs
            for _ in range(10):
                session.run(None, {input_name: x})
            
            # Benchmarking runs
            start_time = time.time()
            for _ in range(num_iterations):
                session.run(None, {input_name: x})
            end_time = time.time()
            
            # Calculate average inference time
            avg_time = (end_time - start_time) / num_iterations
            results['quantized_onnx'] = {
                'avg_inference_time': avg_time,
                'inferences_per_second': 1.0 / avg_time
            }
            
            logger.info(f"Quantized ONNX model: {avg_time*1000:.2f} ms per inference, {1.0/avg_time:.2f} inferences/second")
            
        except Exception as e:
            logger.error(f"Error benchmarking quantized ONNX model: {e}")
    
    # Benchmark TensorRT engine
    if TENSORRT_AVAILABLE and tensorrt_engine_path is not None:
        try:
            logger.info(f"Benchmarking TensorRT engine: {tensorrt_engine_path}")
            
            # Load TensorRT engine
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            with open(tensorrt_engine_path, 'rb') as f:
                engine_data = f.read()
                engine = runtime.deserialize_cuda_engine(engine_data)
            
            # Create execution context
            context = engine.create_execution_context()
            
            # Allocate memory for inputs and outputs
            input_idx = 0
            output_idx = 1
            
            # Prepare input data
            x = preprocess.logmel_extract(audio_data)
            x = np.expand_dims(x, axis=0).astype(np.float32)
            
            # Allocate device memory
            d_input = cuda.mem_alloc(x.nbytes)
            d_output = cuda.mem_alloc(1 * 527 * np.dtype(np.float32).itemsize)  # Assuming 527 output classes
            
            # Create a CUDA stream
            stream = cuda.Stream()
            
            # Warm-up runs
            for _ in range(10):
                # Copy input data to device
                cuda.memcpy_htod_async(d_input, x, stream)
                
                # Run inference
                context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
                
                # Synchronize stream
                stream.synchronize()
            
            # Benchmarking runs
            start_time = time.time()
            for _ in range(num_iterations):
                # Copy input data to device
                cuda.memcpy_htod_async(d_input, x, stream)
                
                # Run inference
                context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
                
                # Synchronize stream
                stream.synchronize()
            end_time = time.time()
            
            # Free device memory
            d_input.free()
            d_output.free()
            
            # Calculate average inference time
            avg_time = (end_time - start_time) / num_iterations
            results['tensorrt'] = {
                'avg_inference_time': avg_time,
                'inferences_per_second': 1.0 / avg_time
            }
            
            logger.info(f"TensorRT engine: {avg_time*1000:.2f} ms per inference, {1.0/avg_time:.2f} inferences/second")
            
        except Exception as e:
            logger.error(f"Error benchmarking TensorRT engine: {e}")
    
    # Print comparison if multiple models were benchmarked
    if len(results) > 1:
        logger.info("=== Benchmark Results ===")
        baseline = None
        for model_type, result in results.items():
            if baseline is None:
                baseline = result['avg_inference_time']
                logger.info(f"{model_type}: {result['avg_inference_time']*1000:.2f} ms (baseline)")
            else:
                speedup = baseline / result['avg_inference_time']
                logger.info(f"{model_type}: {result['avg_inference_time']*1000:.2f} ms ({speedup:.2f}x speedup)")
    
    return results

def main():
    """Main function to run the model optimizer."""
    parser = argparse.ArgumentParser(description="Sound Recognition Model Optimizer")
    
    # Add arguments
    parser.add_argument("--export-onnx", action="store_true", help="Export PyTorch model to ONNX format")
    parser.add_argument("--quantize", action="store_true", help="Quantize ONNX model to INT8")
    parser.add_argument("--optimize", action="store_true", help="Optimize ONNX model")
    parser.add_argument("--create-tensorrt", action="store_true", help="Create TensorRT engine from ONNX model")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark inference speed for different model formats")
    parser.add_argument("--pytorch-model", type=str, help="Path to PyTorch model")
    parser.add_argument("--onnx-model", type=str, help="Path to ONNX model")
    parser.add_argument("--quantized-model", type=str, help="Path to quantized ONNX model")
    parser.add_argument("--tensorrt-engine", type=str, help="Path to TensorRT engine")
    parser.add_argument("--output-dir", type=str, help="Output directory for optimized models")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations for benchmarking")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = MODEL_PATH
    
    # Set default model paths if not specified
    pytorch_model_path = args.pytorch_model or os.path.join(MODEL_PATH, MODEL_FN)
    
    if not args.onnx_model:
        onnx_model_path = os.path.join(output_dir, Path(pytorch_model_path).stem + ".onnx")
    else:
        onnx_model_path = args.onnx_model
    
    if not args.quantized_model:
        quantized_model_path = os.path.join(output_dir, Path(onnx_model_path).stem + "_quantized.onnx")
    else:
        quantized_model_path = args.quantized_model
    
    if not args.tensorrt_engine:
        tensorrt_engine_path = os.path.join(output_dir, Path(onnx_model_path).stem + ".engine")
    else:
        tensorrt_engine_path = args.tensorrt_engine
    
    # Export PyTorch model to ONNX
    if args.export_onnx:
        logger.info("=== Exporting PyTorch model to ONNX ===")
        export_to_onnx(model_path=pytorch_model_path, output_path=onnx_model_path)
    
    # Quantize ONNX model
    if args.quantize:
        logger.info("=== Quantizing ONNX model ===")
        quantize_onnx_model(model_path=onnx_model_path, output_path=quantized_model_path)
    
    # Optimize ONNX model
    if args.optimize:
        logger.info("=== Optimizing ONNX model ===")
        optimize_onnx_model(model_path=onnx_model_path, output_path=os.path.join(output_dir, Path(onnx_model_path).stem + "_optimized.onnx"))
    
    # Create TensorRT engine
    if args.create_tensorrt:
        logger.info("=== Creating TensorRT engine ===")
        create_tensorrt_engine(model_path=onnx_model_path, output_path=tensorrt_engine_path)
    
    # Benchmark models
    if args.benchmark:
        logger.info("=== Benchmarking models ===")
        benchmark_models(
            pytorch_model_path=pytorch_model_path,
            onnx_model_path=onnx_model_path if os.path.exists(onnx_model_path) else None,
            quantized_model_path=quantized_model_path if os.path.exists(quantized_model_path) else None,
            tensorrt_engine_path=tensorrt_engine_path if os.path.exists(tensorrt_engine_path) else None,
            num_iterations=args.iterations
        )

if __name__ == "__main__":
    main() 