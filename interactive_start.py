#!/usr/bin/env python3
"""
Interactive Startup Script for SoundWatch Server
This script prompts the user to select recognition models and settings before starting the server.
"""

import os
import sys
import subprocess
import platform
import argparse

def clear_screen():
    """Clear the terminal screen for better readability"""
    os.system('cls' if platform.system() == 'Windows' else 'clear')

def print_header():
    """Print a nice header for the interactive startup"""
    clear_screen()
    print("\n" + "=" * 80)
    print("  🔊  SOUNDWATCH SERVER - INTERACTIVE STARTUP  🔊  ".center(80))
    print("=" * 80 + "\n")

def get_choice(prompt, options):
    """Get a user choice from a list of options"""
    while True:
        print(prompt)
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")
        
        try:
            choice = int(input("\nEnter your choice (number): "))
            if 1 <= choice <= len(options):
                return choice
            else:
                print("\n❌ Invalid choice. Please try again.\n")
        except ValueError:
            print("\n❌ Please enter a number.\n")

def confirm_settings(settings):
    """Show a summary of settings and ask for confirmation"""
    print("\n" + "=" * 80)
    print("  SUMMARY OF SETTINGS  ".center(80))
    print("=" * 80)
    
    for key, value in settings.items():
        print(f"  • {key}: {value}")
    
    print("\nAre these settings correct?")
    return input("Press Enter to continue or 'n' to restart: ").lower() != 'n'

def main():
    print_header()
    
    # Initial settings - always use port 8080 and no debug mode
    settings = {
        "Sound Recognition Model": "TensorFlow (default)",
        "Speech Recognition System": "Whisper (default)",
        "Server Port": 8080,
        "Debug Mode": "Disabled"
    }
    
    # Get sound recognition model choice
    model_choice = get_choice(
        "Which sound recognition model would you like to use?",
        ["TensorFlow (default, best for general sound recognition)",
         "AST (Audio Spectrogram Transformer, better for some specific sounds)",
         "PANNs (Pretrained Audio Neural Networks, excellent sound discrimination)"]
    )
    
    if model_choice == 1:
        settings["Sound Recognition Model"] = "TensorFlow"
    elif model_choice == 2:
        settings["Sound Recognition Model"] = "AST"
    else:
        settings["Sound Recognition Model"] = "PANNs"
    
    # Get speech recognition system choice
    speech_choice = get_choice(
        "Which speech recognition system would you like to use?",
        ["Whisper (default, works offline, no internet needed)",
         "Google Cloud Speech-to-Text (better accuracy, requires internet)"]
    )
    settings["Speech Recognition System"] = "Whisper" if speech_choice == 1 else "Google Cloud"
    
    # Show summary and confirm
    if not confirm_settings(settings):
        return main()  # Restart if user wants to change settings
    
    # Build the command to start the server
    cmd = ["python3" if platform.system() != "Windows" else "python", "server.py"]
    
    # For the AST and PANNs models, we use environment variables since the server.py 
    # doesn't have command line options for these
    env = os.environ.copy()
    
    # First, reset all model selection variables
    env["USE_AST_MODEL"] = "0"
    env["USE_PANNS_MODEL"] = "0"
    
    # Then set the chosen model's variable
    if settings["Sound Recognition Model"] == "AST":
        env["USE_AST_MODEL"] = "1"
        print("Using AST model (via environment variable)")
    elif settings["Sound Recognition Model"] == "PANNs":
        env["USE_PANNS_MODEL"] = "1"
        print("Using PANNs model (via environment variable)")
    else:
        print("Using TensorFlow model (via environment variable)")
    
    # Add speech recognition flag if Google is selected
    if settings["Speech Recognition System"] == "Google Cloud":
        cmd.append("--use-google-speech")
    
    # Always use port 8080
    cmd.append("--port=8080")
    
    # Show the command
    print("\n" + "=" * 80)
    print("  STARTING SERVER  ".center(80))
    print("=" * 80)
    print(f"\nExecuting: {' '.join(cmd)}\n")
    
    # Start the server with the updated environment
    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except Exception as e:
        print(f"\nError starting server: {str(e)}")

if __name__ == "__main__":
    main() 