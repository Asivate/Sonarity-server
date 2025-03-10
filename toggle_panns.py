#!/usr/bin/env python3
"""
Toggle PANNs Model

This script enables or disables the PANNs (Pretrained Audio Neural Networks) model
for sound recognition in SoundWatch. It modifies the environment variable that 
controls which model is used at runtime.

Usage:
  python toggle_panns.py on  # Enable PANNs model
  python toggle_panns.py off  # Disable PANNs model
  python toggle_panns.py      # Show current status
"""

import os
import sys
import argparse
import subprocess
import platform

def get_current_state():
    """Check if PANNs model is currently enabled in environment variables"""
    # First, check environment variables
    env_value = os.environ.get("USE_PANNS_MODEL")
    if env_value is not None:
        return env_value == "1"
    
    # Default: disabled
    return False

def set_state(enabled):
    """Set the PANNs model state in environment variables"""
    # Set environment variable
    os.environ["USE_PANNS_MODEL"] = "1" if enabled else "0"
    
    # On Windows, we also need to set it at system level for persistence
    if platform.system() == "Windows":
        cmd = f'setx USE_PANNS_MODEL {"1" if enabled else "0"}'
        subprocess.run(cmd, shell=True, check=True)
    else:
        # On Unix-like systems, we need to modify shell profile files
        # This is complex and might require user privileges, so we'll
        # just provide instructions
        home = os.path.expanduser("~")
        print(f"\nTo make this setting permanent, add this line to your shell profile:")
        print(f'export USE_PANNS_MODEL={"1" if enabled else "0"}')
        print("\nFor example, add it to one of these files:")
        print(f"  - {os.path.join(home, '.bashrc')}")
        print(f"  - {os.path.join(home, '.bash_profile')}")
        print(f"  - {os.path.join(home, '.zshrc')}")
        print("\nOr set it in your startup script before launching the server.\n")

def print_status(enabled):
    """Print the current status of the PANNs model"""
    if enabled:
        print("""
✅ PANNs model is ENABLED
"""[1:-1])
    else:
        print("""
❌ PANNs model is DISABLED
"""[1:-1])

def main():
    parser = argparse.ArgumentParser(description="Toggle PANNs model on/off")
    parser.add_argument("state", nargs="?", choices=["on", "off"], 
                        help="Set PANNs model state (on/off). If omitted, shows current state.")
    args = parser.parse_args()
    
    # Get current state
    current_state = get_current_state()
    
    if args.state is None:
        # Just show current status
        print_status(current_state)
    elif args.state == "on":
        # Enable PANNs model
        set_state(True)
        print_status(True)
        
        # Warn if AST model is also enabled
        if os.environ.get("USE_AST_MODEL") == "1":
            print("""
⚠️  WARNING: Both AST and PANNs models are enabled.
   This may cause conflicts. Consider disabling AST model:
   python toggle_ast.py off
"""[1:-1])

        # Check if model files exist
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        model_path = os.path.join(model_dir, "Cnn9_GMP_64x64_300000_iterations_mAP=0.37.pth")
        if not os.path.exists(model_path):
            print("""
⚠️  WARNING: PANNs model file not found.
   Please download the required files:
   python download_panns_model.py
"""[1:-1])
    else:  # args.state == "off"
        # Disable PANNs model
        set_state(False)
        print_status(False)
        
        # Warn if AST model is also disabled
        if os.environ.get("USE_AST_MODEL") != "1":
            print("""
⚠️  WARNING: Both AST and PANNs models are disabled.
   The server will fall back to the TensorFlow model.
   This may cause lower accuracy in sound recognition.
"""[1:-1])

if __name__ == "__main__":
    main() 