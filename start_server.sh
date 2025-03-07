#!/bin/bash

# Exit on error
set -e

# Configuration
PORT=5000
DEBUG=false
VENV_DIR="venv"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --port=*)
      PORT="${1#*=}"
      shift
      ;;
    --debug)
      DEBUG=true
      shift
      ;;
    --help)
      echo "Usage: ./start_server.sh [--port=PORT] [--debug]"
      echo ""
      echo "Options:"
      echo "  --port=PORT    Specify the port to run the server on (default: 5000)"
      echo "  --debug        Run in debug mode"
      echo "  --help         Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create models directory if it doesn't exist
if [ ! -d "models" ]; then
  echo "Creating models directory..."
  mkdir -p models
fi

# Run the server
echo "Starting Sonarity server on port $PORT"
if [ "$DEBUG" = true ]; then
  python server.py --port "$PORT" --debug
else
  python server.py --port "$PORT"
fi 