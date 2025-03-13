# SoundWatch Model Optimization

This document explains how to use the model optimization features in SoundWatch to improve performance and reduce resource usage.

## Overview

SoundWatch now supports optimized inference using:

1. **ONNX Runtime** - A cross-platform inference acceleration framework
2. **Quantized Models** - Reduced precision models for faster inference and smaller memory footprint
3. **TensorRT** - NVIDIA's high-performance deep learning inference optimizer (GPU only)

These optimizations can significantly improve inference speed and reduce memory usage, making SoundWatch more responsive and efficient, especially on resource-constrained devices.

## Requirements

To use the optimized models, you need:

- Python 3.7 or higher
- PyTorch (already required by SoundWatch)
- ONNX Runtime (installed automatically when needed)
- For TensorRT: NVIDIA GPU with CUDA support

## Usage

### Quick Start

The easiest way to use optimized models is with the `run_optimized.py` script:

```bash
# Run with ONNX Runtime
python run_optimized.py --onnx

# Run with quantized ONNX model (smaller and faster)
python run_optimized.py --quantized

# Run with TensorRT (NVIDIA GPUs only)
python run_optimized.py --tensorrt

# Run a benchmark before starting the server
python run_optimized.py --onnx --benchmark
```

### Manual Setup

You can also manually optimize and use the models:

1. **Export to ONNX**:
   ```bash
   python model_optimizer.py --export-onnx
   ```

2. **Quantize the ONNX model**:
   ```bash
   python model_optimizer.py --quantize
   ```

3. **Create TensorRT engine** (NVIDIA GPUs only):
   ```bash
   python model_optimizer.py --create-tensorrt
   ```

4. **Run benchmarks**:
   ```bash
   python model_optimizer.py --benchmark
   ```

5. **Run the server with an optimized model**:
   ```bash
   # Set environment variables
   export USE_PANNS_MODEL=1
   export PANNS_MODEL_TYPE=onnx  # or 'tensorrt'
   export PANNS_MODEL_PATH=/path/to/optimized/model
   
   # Run the server
   python server.py
   ```

## Performance Comparison

Here's a typical performance comparison on different hardware:

| Model Type | CPU Inference Time | GPU Inference Time | Memory Usage |
|------------|-------------------|-------------------|--------------|
| PyTorch    | 150-300ms         | 50-100ms          | ~500MB       |
| ONNX       | 80-150ms          | 30-60ms           | ~300MB       |
| Quantized  | 40-80ms           | 20-40ms           | ~100MB       |
| TensorRT   | N/A               | 10-30ms           | ~200MB       |

Actual performance will vary based on your hardware and specific configuration.

## Troubleshooting

If you encounter issues with optimized models:

1. **ONNX Runtime errors**:
   - Make sure you have the latest version: `pip install --upgrade onnxruntime`
   - For CUDA support: `pip install --upgrade onnxruntime-gpu`

2. **TensorRT errors**:
   - Ensure CUDA and cuDNN are properly installed
   - Check TensorRT compatibility with your GPU

3. **Model conversion failures**:
   - Try with a smaller batch size: `python model_optimizer.py --export-onnx --batch-size 1`
   - Check for unsupported operations in your model

4. **Fallback to PyTorch**:
   If optimized models don't work, the system will automatically fall back to the standard PyTorch model.

## Advanced Configuration

You can fine-tune the optimization process:

- **Custom model paths**:
  ```bash
  python run_optimized.py --onnx --pytorch-model /path/to/custom/model.pth
  ```

- **Optimization level**:
  ```bash
  python model_optimizer.py --optimize --level 3  # Higher level = more optimization
  ```

- **Custom output directory**:
  ```bash
  python model_optimizer.py --export-onnx --output-dir /path/to/models
  ```

## Contributing

If you improve the optimization process or add support for new hardware accelerators, please consider contributing your changes back to the project! 