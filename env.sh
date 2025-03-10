#!/bin/bash
# Optimized environment variables for CNN13 model on AMD EPYC 7B13 (8 cores)

# Set this to enable PANNs model
export USE_PANNS_MODEL=1

# Set this to disable AST model (using only one at a time for better performance)
export USE_AST_MODEL=0

# CPU cores optimization
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8

# PyTorch performance tuning
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_COLLECT_ALLOCATIONS=1

# Set to 1 for speech detection and sentiment analysis
export USE_SPEECH=1
export USE_SENTIMENT=1

# Set to 0 for no memory optimization (using all available CPU resources)
export MEMORY_OPTIMIZATION=0

# Print configuration
echo "Environment variables set for CNN13 model optimization:"
echo "USE_PANNS_MODEL: $USE_PANNS_MODEL"
echo "USE_AST_MODEL: $USE_AST_MODEL"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "MKL_NUM_THREADS: $MKL_NUM_THREADS"
echo "NUMEXPR_NUM_THREADS: $NUMEXPR_NUM_THREADS"
echo "OPENBLAS_NUM_THREADS: $OPENBLAS_NUM_THREADS"
echo "MEMORY_OPTIMIZATION: $MEMORY_OPTIMIZATION"
echo "USE_SPEECH: $USE_SPEECH"
echo "USE_SENTIMENT: $USE_SENTIMENT"
echo ""
echo "To use these settings, run:"
echo "source env.sh"
echo "python interactive_start.py" 