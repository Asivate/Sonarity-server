#!/usr/bin/env python
"""
SoundWatch Server Launcher

This script provides a unified way to start the SoundWatch server with various options,
including optimized models, memory management, and server configuration.
"""

import os
import sys
import time
import argparse
import platform
import subprocess
import logging
import shutil
from pathlib import Path
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("soundwatch_launcher")

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SoundWatch Server Launcher")
    
    # Model optimization options
    model_group = parser.add_argument_group('Model Optimization')
    model_group.add_argument('--onnx', action='store_true', help='Use ONNX Runtime for model inference')
    model_group.add_argument('--quantized', action='store_true', help='Use quantized ONNX model (implies --onnx)')
    model_group.add_argument('--tensorrt', action='store_true', help='Use TensorRT for model inference (NVIDIA GPUs only)')
    model_group.add_argument('--benchmark', action='store_true', help='Run benchmarks on all available model types')
    model_group.add_argument('--model-path', type=str, help='Custom path to model file')
    
    # Server configuration
    server_group = parser.add_argument_group('Server Configuration')
    server_group.add_argument('--port', type=int, default=8080, help='Port to run the server on (default: 8080)')
    server_group.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to (default: 0.0.0.0)')
    server_group.add_argument('--debug', action='store_true', help='Run server in debug mode')
    
    # Memory management
    memory_group = parser.add_argument_group('Memory Management')
    memory_group.add_argument('--memory-optimization', type=int, choices=[0, 1, 2], default=1,
                             help='Memory optimization level (0=none, 1=moderate, 2=aggressive)')
    memory_group.add_argument('--cleanup', action='store_true', help='Clean up temporary files after server exit')
    
    # Speech recognition
    speech_group = parser.add_argument_group('Speech Recognition')
    speech_group.add_argument('--speech', action='store_true', help='Enable speech recognition')
    speech_group.add_argument('--google-speech', action='store_true', help='Use Google Speech API (requires credentials)')
    speech_group.add_argument('--sentiment', action='store_true', help='Enable sentiment analysis for speech')
    
    return parser.parse_args()

def check_dependencies():
    """Check if required dependencies are installed"""
    logger.info("Checking dependencies...")
    
    # Check if server.py exists
    server_path = os.path.join(SCRIPT_DIR, "server.py")
    if not os.path.exists(server_path):
        logger.error(f"Server script not found at {server_path}")
        return False
    
    # Check if soundwatch_config.py exists
    config_path = os.path.join(SCRIPT_DIR, "soundwatch_config.py")
    if not os.path.exists(config_path):
        logger.warning(f"Configuration module not found at {config_path}")
        logger.warning("Will use default configuration values")
    
    # Check if model_optimizer.py exists (needed for optimized models)
    optimizer_path = os.path.join(SCRIPT_DIR, "model_optimizer.py")
    if not os.path.exists(optimizer_path):
        logger.warning(f"Model optimizer not found at {optimizer_path}")
        logger.warning("Optimized models will not be available")
    
    return True

def prepare_optimized_model(args):
    """Prepare optimized model based on command line arguments"""
    # Set model type based on arguments
    model_type = "pytorch"  # Default
    
    if args.tensorrt:
        model_type = "tensorrt"
    elif args.quantized:
        model_type = "quantized"
    elif args.onnx:
        model_type = "onnx"
    
    # If using default PyTorch model, no preparation needed
    if model_type == "pytorch":
        logger.info("Using standard PyTorch model")
        return True
    
    # Check if model_optimizer.py exists
    optimizer_path = os.path.join(SCRIPT_DIR, "model_optimizer.py")
    if not os.path.exists(optimizer_path):
        logger.error(f"Model optimizer not found at {optimizer_path}")
        logger.error("Cannot use optimized models")
        return False
    
    # Try to import required modules
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
    except ImportError:
        logger.error("PyTorch not installed. Required for model optimization.")
        logger.info("Attempting to install PyTorch...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
            logger.info("PyTorch installed successfully")
        except subprocess.CalledProcessError:
            logger.error("Failed to install PyTorch")
            return False
    
    # Install ONNX if needed
    if model_type in ["onnx", "quantized", "tensorrt"]:
        try:
            import onnx
            import onnxruntime
            logger.info(f"ONNX version: {onnx.__version__}")
            logger.info(f"ONNX Runtime version: {onnxruntime.__version__}")
        except ImportError:
            logger.info("ONNX or ONNX Runtime not installed. Installing...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx", "onnxruntime"])
                logger.info("ONNX and ONNX Runtime installed successfully")
            except subprocess.CalledProcessError:
                logger.error("Failed to install ONNX dependencies")
                return False
    
    # Install TensorRT if needed
    if model_type == "tensorrt":
        try:
            import tensorrt
            logger.info(f"TensorRT version: {tensorrt.__version__}")
        except ImportError:
            logger.warning("TensorRT not installed. This requires manual installation.")
            logger.warning("See https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html")
            logger.warning("Falling back to ONNX Runtime")
            model_type = "onnx"
    
    # Set environment variables for model type
    os.environ["PANNS_MODEL_TYPE"] = model_type
    if args.model_path:
        os.environ["PANNS_MODEL_PATH"] = args.model_path
    
    # Run benchmark if requested
    if args.benchmark:
        logger.info("Running model benchmarks...")
        benchmark_cmd = [sys.executable, optimizer_path, "--benchmark"]
        if args.model_path:
            benchmark_cmd.extend(["--model-path", args.model_path])
        
        try:
            subprocess.check_call(benchmark_cmd)
        except subprocess.CalledProcessError:
            logger.error("Benchmark failed")
            return False
    
    # Create ONNX model if needed
    if model_type in ["onnx", "quantized", "tensorrt"] and not args.model_path:
        logger.info(f"Creating {model_type} model...")
        optimizer_cmd = [sys.executable, optimizer_path]
        
        if model_type == "onnx":
            optimizer_cmd.append("--export-onnx")
        elif model_type == "quantized":
            optimizer_cmd.extend(["--export-onnx", "--quantize"])
        elif model_type == "tensorrt":
            optimizer_cmd.extend(["--export-onnx", "--tensorrt"])
        
        try:
            subprocess.check_call(optimizer_cmd)
            logger.info(f"{model_type.upper()} model created successfully")
        except subprocess.CalledProcessError:
            logger.error(f"Failed to create {model_type} model")
            logger.warning("Falling back to PyTorch model")
            os.environ["PANNS_MODEL_TYPE"] = "pytorch"
            return False
    
    return True

def create_health_check_files():
    """Create health check files to monitor server status"""
    health_dir = os.path.join(SCRIPT_DIR, "health")
    os.makedirs(health_dir, exist_ok=True)
    
    # Create starting file
    with open(os.path.join(health_dir, "starting.txt"), "w") as f:
        f.write(f"Server starting at {datetime.datetime.now().isoformat()}\n")
        f.write(f"Model type: {os.environ.get('PANNS_MODEL_TYPE', 'pytorch')}\n")
        f.write(f"Memory optimization: {os.environ.get('MEMORY_OPTIMIZATION', '1')}\n")
    
    # Remove running file if it exists
    running_file = os.path.join(health_dir, "running.txt")
    if os.path.exists(running_file):
        os.remove(running_file)

def run_server(args):
    """Run the SoundWatch server with the specified options"""
    # Create health check files
    create_health_check_files()
    
    # Set environment variables
    os.environ["MEMORY_OPTIMIZATION"] = str(args.memory_optimization)
    os.environ["USE_PANNS_MODEL"] = "1"  # Always enable PANNs model
    
    # Set speech recognition options
    if args.speech:
        os.environ["USE_SPEECH"] = "1"
    if args.sentiment:
        os.environ["USE_SENTIMENT"] = "1"
    if args.google_speech:
        os.environ["USE_GOOGLE_SPEECH"] = "1"
    
    # Prepare command
    server_path = os.path.join(SCRIPT_DIR, "server.py")
    cmd = [sys.executable, server_path]
    
    # Add port and host if specified
    cmd.extend(["--port", str(args.port)])
    cmd.extend(["--host", args.host])
    
    # Add debug flag if specified
    if args.debug:
        cmd.append("--debug")
    
    # Print startup message
    logger.info("=" * 50)
    logger.info(f"Starting SoundWatch server on {args.host}:{args.port}")
    logger.info(f"Model type: {os.environ.get('PANNS_MODEL_TYPE', 'pytorch')}")
    logger.info(f"Memory optimization level: {args.memory_optimization}")
    logger.info("=" * 50)
    
    # Countdown
    for i in range(3, 0, -1):
        logger.info(f"Starting in {i}...")
        time.sleep(1)
    
    # Run the server
    try:
        # Create running file when server starts
        health_dir = os.path.join(SCRIPT_DIR, "health")
        with open(os.path.join(health_dir, "running.txt"), "w") as f:
            f.write(f"Server running since {datetime.datetime.now().isoformat()}\n")
            f.write(f"Model type: {os.environ.get('PANNS_MODEL_TYPE', 'pytorch')}\n")
            f.write(f"Memory optimization: {os.environ.get('MEMORY_OPTIMIZATION', '1')}\n")
        
        # Run the server
        subprocess.check_call(cmd)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except subprocess.CalledProcessError as e:
        logger.error(f"Server exited with error code {e.returncode}")
    finally:
        # Remove running file when server stops
        running_file = os.path.join(health_dir, "running.txt")
        if os.path.exists(running_file):
            os.remove(running_file)
        
        # Clean up temporary files if requested
        if args.cleanup:
            cleanup_temp_files()

def cleanup_temp_files():
    """Clean up temporary files created during server execution"""
    logger.info("Cleaning up temporary files...")
    
    # Clean up health check files
    health_dir = os.path.join(SCRIPT_DIR, "health")
    if os.path.exists(health_dir):
        for file in os.listdir(health_dir):
            os.remove(os.path.join(health_dir, file))
        os.rmdir(health_dir)
    
    # Clean up ONNX models if they were created temporarily
    if os.environ.get("PANNS_MODEL_TYPE") in ["onnx", "quantized", "tensorrt"]:
        models_dir = os.path.join(SCRIPT_DIR, "models")
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith(".onnx") or file.endswith(".engine"):
                    logger.info(f"Removing temporary model file: {file}")
                    os.remove(os.path.join(models_dir, file))

def main():
    """Main entry point"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Log system information
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"Operating system: {platform.system()} {platform.release()}")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed")
        return 1
    
    # Prepare optimized model if requested
    if args.onnx or args.quantized or args.tensorrt or args.benchmark:
        if not prepare_optimized_model(args):
            logger.warning("Failed to prepare optimized model, using PyTorch model instead")
    
    # Run the server
    run_server(args)
    
    # Clean up if requested
    if args.cleanup:
        cleanup_temp_files()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 