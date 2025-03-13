#!/usr/bin/env python3
"""
SoundWatch Optimized Server Launcher

This script sets up the environment for the optimized CNN13 model
and launches the SoundWatch server with performance optimizations.

Usage:
    python run_optimized.py [--onnx] [--quantized] [--tensorrt] [--benchmark] [--cleanup]

Options:
    --onnx         Use ONNX Runtime for inference instead of PyTorch
    --quantized    Use quantized ONNX model (requires --onnx)
    --tensorrt     Use TensorRT engine for inference (requires GPU)
    --benchmark    Run a benchmark before starting the server
    --cleanup      Remove temporary files after server exits
"""

import os
import sys
import subprocess
import logging
import time
import argparse
import shutil
import platform

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("soundwatch_optimized")

# Set up paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models")
PYTORCH_MODEL = "Cnn13_GMP_64x64_520000_iterations_mAP=0.42.pth"
ONNX_MODEL = "Cnn13_GMP_64x64_520000_iterations_mAP=0.42.onnx"
QUANTIZED_MODEL = "Cnn13_GMP_64x64_520000_iterations_mAP=0.42_quantized.onnx"
TENSORRT_ENGINE = "Cnn13_GMP_64x64_520000_iterations_mAP=0.42.engine"

def setup_onnx_runtime():
    """
    Check if ONNX Runtime is installed, and if not, install it.
    
    Returns:
        bool: True if ONNX Runtime is available, False otherwise
    """
    try:
        import onnxruntime
        logger.info(f"ONNX Runtime version {onnxruntime.__version__} is already installed")
        return True
    except ImportError:
        logger.warning("ONNX Runtime not installed. Attempting to install...")
        
        try:
            # Check if we need ONNX Runtime GPU or CPU version
            import torch
            has_cuda = torch.cuda.is_available()
            if has_cuda:
                logger.info("CUDA is available, installing ONNX Runtime GPU")
                package = "onnxruntime-gpu"
            else:
                logger.info("CUDA is not available, installing ONNX Runtime CPU")
                package = "onnxruntime"
                
            # Install ONNX Runtime
            logger.info(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--no-cache-dir", "--quiet"])
            
            # Verify installation
            import onnxruntime
            logger.info(f"Successfully installed ONNX Runtime version {onnxruntime.__version__}")
            return True
                
        except Exception as e:
            logger.error(f"Failed to install ONNX Runtime: {e}")
            return False

def check_model_files():
    """
    Check if the required model files exist, and if not, try to run setup_cnn13.py
    
    Returns:
        bool: True if all needed files are available, False otherwise
    """
    # Check if PyTorch model exists
    pytorch_model_path = os.path.join(MODEL_PATH, PYTORCH_MODEL)
    if not os.path.exists(pytorch_model_path):
        logger.warning(f"PyTorch model not found at {pytorch_model_path}")
        
        # Check if setup_cnn13.py exists
        setup_script = os.path.join(SCRIPT_DIR, "setup_cnn13.py")
        if os.path.exists(setup_script):
            logger.info("Running setup_cnn13.py to download the model...")
            
            try:
                subprocess.check_call([sys.executable, setup_script])
                
                # Check if model was downloaded
                if os.path.exists(pytorch_model_path):
                    logger.info("Model downloaded successfully")
                else:
                    logger.error("Model download failed")
                    return False
                    
            except Exception as e:
                logger.error(f"Error running setup_cnn13.py: {e}")
                return False
        else:
            logger.error(f"setup_cnn13.py not found at {setup_script}")
            return False
    
    return True

def prepare_optimized_model(use_onnx=False, use_quantized=False, use_tensorrt=False, run_benchmark=False):
    """
    Prepare the optimized model based on the selected options.
    
    Args:
        use_onnx: Whether to use ONNX Runtime
        use_quantized: Whether to use quantized ONNX model
        use_tensorrt: Whether to use TensorRT engine
        run_benchmark: Whether to run a benchmark
        
    Returns:
        tuple: (model_type, model_path) for the selected model
    """
    # Validate options
    if use_quantized and not use_onnx:
        logger.warning("Quantized model requires ONNX Runtime. Enabling ONNX Runtime.")
        use_onnx = True
    
    # Set default model type and path
    model_type = "pytorch"
    model_path = os.path.join(MODEL_PATH, PYTORCH_MODEL)
    
    # Check if optimizer script exists
    optimizer_script = os.path.join(SCRIPT_DIR, "model_optimizer.py")
    if not os.path.exists(optimizer_script):
        logger.error(f"model_optimizer.py not found at {optimizer_script}")
        if use_onnx or use_quantized or use_tensorrt:
            logger.error("Cannot use optimization options without model_optimizer.py")
            return model_type, model_path
    
    # First, ensure dependencies are installed for optimizations
    if use_onnx or use_quantized or use_tensorrt:
        try:
            # Install ONNX and onnxruntime if we're using ONNX models
            subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx", "onnxruntime", "--no-cache-dir", "--quiet"])
            logger.info("ONNX dependencies installed successfully")
        except Exception as e:
            logger.error(f"Failed to install optimization dependencies: {e}")
            logger.warning("Falling back to PyTorch model")
            return model_type, model_path
    
    # Handle TensorRT
    if use_tensorrt:
        tensorrt_path = os.path.join(MODEL_PATH, TENSORRT_ENGINE)
        
        if not os.path.exists(tensorrt_path):
            logger.info("TensorRT engine not found. Creating it...")
            
            # First check if ONNX model exists
            onnx_path = os.path.join(MODEL_PATH, ONNX_MODEL)
            if not os.path.exists(onnx_path):
                logger.info("ONNX model not found. Creating it...")
                try:
                    # Run with explicit Python path to ensure proper environment
                    result = subprocess.run(
                        [sys.executable, optimizer_script, "--export-onnx"],
                        capture_output=True,
                        text=True
                    )
                    
                    # Log the output for debugging
                    if result.stdout:
                        logger.info(f"ONNX export output: {result.stdout}")
                    if result.stderr:
                        logger.warning(f"ONNX export errors: {result.stderr}")
                    
                    # Verify ONNX model was created
                    if os.path.exists(onnx_path):
                        logger.info(f"ONNX model created successfully at {onnx_path}")
                    else:
                        logger.error("ONNX model creation failed - file not found after export")
                        return model_type, model_path
                        
                except Exception as e:
                    logger.error(f"Failed to create ONNX model: {e}")
                    return model_type, model_path
            
            # Now create TensorRT engine
            try:
                # Run with explicit Python path to ensure proper environment
                result = subprocess.run(
                    [sys.executable, optimizer_script, "--create-tensorrt"],
                    capture_output=True,
                    text=True
                )
                
                # Log the output for debugging
                if result.stdout:
                    logger.info(f"TensorRT engine creation output: {result.stdout}")
                if result.stderr:
                    logger.warning(f"TensorRT engine creation errors: {result.stderr}")
                    
                # Verify TensorRT engine was created
                if os.path.exists(tensorrt_path):
                    logger.info(f"TensorRT engine created successfully at {tensorrt_path}")
                else:
                    logger.error("TensorRT engine creation failed - file not found after creation")
                    return model_type, model_path
            except Exception as e:
                logger.error(f"Failed to create TensorRT engine: {e}")
                return model_type, model_path
        
        if os.path.exists(tensorrt_path):
            model_type = "tensorrt"
            model_path = tensorrt_path
            logger.info(f"Using TensorRT engine: {model_path}")
        else:
            logger.error("Failed to find or create TensorRT engine")
    
    # Handle ONNX quantized model
    elif use_quantized:
        quantized_path = os.path.join(MODEL_PATH, QUANTIZED_MODEL)
        
        if not os.path.exists(quantized_path):
            logger.info("Quantized ONNX model not found. Creating it...")
            
            # First check if ONNX model exists
            onnx_path = os.path.join(MODEL_PATH, ONNX_MODEL)
            if not os.path.exists(onnx_path):
                logger.info("ONNX model not found. Creating it...")
                try:
                    # Run with explicit Python path to ensure proper environment
                    result = subprocess.run(
                        [sys.executable, optimizer_script, "--export-onnx"],
                        capture_output=True,
                        text=True
                    )
                    
                    # Log the output for debugging
                    if result.stdout:
                        logger.info(f"ONNX export output: {result.stdout}")
                    if result.stderr:
                        logger.warning(f"ONNX export errors: {result.stderr}")
                    
                    # Verify ONNX model was created
                    if os.path.exists(onnx_path):
                        logger.info(f"ONNX model created successfully at {onnx_path}")
                    else:
                        logger.error("ONNX model creation failed - file not found after export")
                        return model_type, model_path
                        
                except Exception as e:
                    logger.error(f"Failed to create ONNX model: {e}")
                    return model_type, model_path
            
            # Now create quantized model
            try:
                # Run with explicit Python path to ensure proper environment
                result = subprocess.run(
                    [sys.executable, optimizer_script, "--quantize"],
                    capture_output=True,
                    text=True
                )
                
                # Log the output for debugging
                if result.stdout:
                    logger.info(f"Quantized model creation output: {result.stdout}")
                if result.stderr:
                    logger.warning(f"Quantized model creation errors: {result.stderr}")
                
                # Verify quantized model was created
                if os.path.exists(quantized_path):
                    logger.info(f"Quantized ONNX model created successfully at {quantized_path}")
                else:
                    logger.error("Quantized ONNX model creation failed - file not found after creation")
                    return model_type, model_path
            except Exception as e:
                logger.error(f"Failed to create quantized model: {e}")
                return model_type, model_path
        
        if os.path.exists(quantized_path):
            model_type = "onnx"
            model_path = quantized_path
            logger.info(f"Using quantized ONNX model: {model_path}")
        else:
            logger.error("Failed to find or create quantized ONNX model")
    
    # Handle ONNX model
    elif use_onnx:
        onnx_path = os.path.join(MODEL_PATH, ONNX_MODEL)
        
        if not os.path.exists(onnx_path):
            logger.info("ONNX model not found. Creating it...")
            try:
                # Run with explicit Python path to ensure proper environment
                result = subprocess.run(
                    [sys.executable, optimizer_script, "--export-onnx"],
                    capture_output=True,
                    text=True
                )
                
                # Log the output for debugging
                if result.stdout:
                    logger.info(f"ONNX export output: {result.stdout}")
                if result.stderr:
                    logger.warning(f"ONNX export errors: {result.stderr}")
                
                # Verify ONNX model was created
                if os.path.exists(onnx_path):
                    logger.info(f"ONNX model created successfully at {onnx_path}")
                else:
                    logger.error("ONNX model creation failed - file not found after export")
                    return model_type, model_path
                    
            except Exception as e:
                logger.error(f"Failed to create ONNX model: {e}")
                return model_type, model_path
        
        if os.path.exists(onnx_path):
            model_type = "onnx"
            model_path = onnx_path
            logger.info(f"Using ONNX model: {model_path}")
        else:
            logger.error("Failed to find or create ONNX model")
    
    # Run benchmark if requested
    if run_benchmark and os.path.exists(optimizer_script):
        logger.info("Running benchmark...")
        try:
            benchmark_cmd = [sys.executable, optimizer_script, "--benchmark"]
            
            # Add model paths based on what's available
            if model_type == "pytorch":
                benchmark_cmd.extend(["--pytorch-model", model_path])
            
            onnx_path = os.path.join(MODEL_PATH, ONNX_MODEL)
            if os.path.exists(onnx_path):
                benchmark_cmd.extend(["--onnx-model", onnx_path])
            
            quantized_path = os.path.join(MODEL_PATH, QUANTIZED_MODEL)
            if os.path.exists(quantized_path):
                benchmark_cmd.extend(["--quantized-model", quantized_path])
            
            tensorrt_path = os.path.join(MODEL_PATH, TENSORRT_ENGINE)
            if os.path.exists(tensorrt_path):
                benchmark_cmd.extend(["--tensorrt-engine", tensorrt_path])
            
            subprocess.check_call(benchmark_cmd)
            
        except Exception as e:
            logger.error(f"Failed to run benchmark: {e}")
    
    return model_type, model_path

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="SoundWatch Optimized Server Launcher")
    
    parser.add_argument("--onnx", action="store_true", help="Use ONNX Runtime for inference")
    parser.add_argument("--quantized", action="store_true", help="Use quantized ONNX model")
    parser.add_argument("--tensorrt", action="store_true", help="Use TensorRT engine for inference")
    parser.add_argument("--benchmark", action="store_true", help="Run a benchmark before starting the server")
    parser.add_argument("--cleanup", action="store_true", help="Remove temporary files after server exits")
    
    args = parser.parse_args()
    
    # Welcome message
    logger.info("==================================================")
    logger.info("SoundWatch Optimized Server Launcher")
    logger.info("==================================================")
    
    # Log system information
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Operating system: {platform.system()} {platform.version()}")
    
    # Check if the directory structure is correct
    if not os.path.exists(SCRIPT_DIR):
        logger.error(f"Script directory not found: {SCRIPT_DIR}")
        return 1
    
    # Create models directory if it doesn't exist
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Check if server.py exists
    server_script = os.path.join(SCRIPT_DIR, "server.py")
    if not os.path.exists(server_script):
        logger.error(f"server.py not found at {server_script}")
        return 1
    
    # Check if we need to set up ONNX Runtime
    if args.onnx or args.quantized:
        if not setup_onnx_runtime():
            logger.error("Failed to set up ONNX Runtime. Falling back to PyTorch.")
            args.onnx = False
            args.quantized = False
    
    # Check if model files exist
    if not check_model_files():
        logger.error("Required model files not found")
        return 1
    
    # Prepare optimized model
    model_type, model_path = prepare_optimized_model(
        use_onnx=args.onnx,
        use_quantized=args.quantized,
        use_tensorrt=args.tensorrt,
        run_benchmark=args.benchmark
    )
    
    # Set environment variables
    os.environ["USE_PANNS_MODEL"] = "1"
    os.environ["PANNS_MODEL_PATH"] = model_path
    os.environ["PANNS_MODEL_TYPE"] = model_type
    
    logger.info(f"Using model type: {model_type}")
    logger.info(f"Model path: {model_path}")
    
    # Countdown to server start
    logger.info("Starting SoundWatch server in:")
    for i in range(3, 0, -1):
        logger.info(f"{i}...")
        time.sleep(1)
    
    logger.info("Server starting now!")
    
    # Start the server
    try:
        # Change to the server directory
        os.chdir(SCRIPT_DIR)
        
        # Run the server
        subprocess.check_call([sys.executable, "server.py"])
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error running server: {e}")
        return 1
    finally:
        # Clean up if requested
        if args.cleanup:
            logger.info("Cleaning up temporary files...")
            
            # Remove ONNX models if they were created just for this run
            if args.onnx and not os.path.exists(os.path.join(MODEL_PATH, ONNX_MODEL)) and model_type == "onnx":
                try:
                    os.remove(model_path)
                    logger.info(f"Removed temporary model: {model_path}")
                except:
                    pass
    
    logger.info("Server shut down")
    return 0

if __name__ == "__main__":
    sys.exit(main())
