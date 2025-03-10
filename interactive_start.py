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
    parser = argparse.ArgumentParser(description="Interactive startup for SoundWatch Server")
    parser.add_argument('--host', default="0.0.0.0", help="Host to bind to")
    parser.add_argument('--port', type=int, default=8080, help="Port to bind to")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    args = parser.parse_args()
    
    while True:
        print_header()
        
        settings = {}
        
        # Step 1: Select recognition model
        model_choice = get_choice(
            "📦 Select sound recognition model:",
            [
                "AST (Audio Spectrogram Transformer) - Recommended for modern devices",
                "PANNs (Pretrained Audio Neural Networks) - Alternative model with more classes",
                "TensorFlow (Legacy) - Better for older devices"
            ]
        )
        
        if model_choice == 1:
            settings["Recognition Model"] = "AST (Audio Spectrogram Transformer)"
            os.environ["USE_AST_MODEL"] = "1"
            os.environ["USE_PANNS_MODEL"] = "0"
        elif model_choice == 2:
            settings["Recognition Model"] = "PANNs (Pretrained Audio Neural Networks)"
            os.environ["USE_AST_MODEL"] = "0"
            os.environ["USE_PANNS_MODEL"] = "1"
        else:
            settings["Recognition Model"] = "TensorFlow (Legacy)"
            os.environ["USE_AST_MODEL"] = "0"
            os.environ["USE_PANNS_MODEL"] = "0"
        
        # Step 2: Select speech recognition
        speech_choice = get_choice(
            "🗣️ Enable speech recognition?",
            ["Yes - Whisper (local, private)", "Yes - Google Cloud (requires API key)", "No"]
        )
        
        # Initialize command line arguments list with just port
        cmd_args = [f"--port={args.port}"]
        
        if args.debug:
            cmd_args.append("--debug")
        
        if speech_choice == 1:
            settings["Speech Recognition"] = "Whisper (local)"
            os.environ["USE_SPEECH"] = "1"
            os.environ["USE_GOOGLE_SPEECH"] = "0"
        elif speech_choice == 2:
            settings["Speech Recognition"] = "Google Cloud"
            os.environ["USE_SPEECH"] = "1"
            os.environ["USE_GOOGLE_SPEECH"] = "1"
            # Add the Google Speech flag to command line args
            cmd_args.append("--use-google-speech")
        else:
            settings["Speech Recognition"] = "Disabled"
            os.environ["USE_SPEECH"] = "0"
        
        # Step 3: Select sentiment analysis (if speech is enabled)
        if speech_choice != 3:  # If speech is enabled
            sentiment_choice = get_choice(
                "😊 Enable sentiment analysis on speech?",
                ["Yes", "No"]
            )
            
            if sentiment_choice == 1:
                settings["Sentiment Analysis"] = "Enabled"
                os.environ["USE_SENTIMENT"] = "1"
            else:
                settings["Sentiment Analysis"] = "Disabled"
                os.environ["USE_SENTIMENT"] = "0"
        else:
            settings["Sentiment Analysis"] = "Disabled (Speech recognition is off)"
            os.environ["USE_SENTIMENT"] = "0"
        
        # Step 4: Select memory optimization
        memory_choice = get_choice(
            "🧠 Memory optimization level:",
            [
                "None - For powerful computers with plenty of RAM",
                "Moderate - Recommended for most computers",
                "Aggressive - For computers with limited RAM"
            ]
        )
        
        if memory_choice == 1:
            settings["Memory Optimization"] = "None"
            os.environ["MEMORY_OPTIMIZATION"] = "0"
        elif memory_choice == 2:
            settings["Memory Optimization"] = "Moderate"
            os.environ["MEMORY_OPTIMIZATION"] = "1"
        else:
            settings["Memory Optimization"] = "Aggressive"
            os.environ["MEMORY_OPTIMIZATION"] = "2"
        
        # Confirm settings
        if confirm_settings(settings):
            break
    
    # Add host if specified and different from default
    if args.host != "0.0.0.0":
        cmd_args.append(f"--host={args.host}")
    
    cmd = [sys.executable, "server.py"] + cmd_args
    
    print("\n" + "=" * 80)
    print("  STARTING SERVER  ".center(80))
    print("=" * 80)
    print(f"\nRunning command: {' '.join(cmd)}")
    print("\nEnvironment variables set:")
    for key in ["USE_AST_MODEL", "USE_PANNS_MODEL", "USE_SPEECH", "USE_GOOGLE_SPEECH", 
                "USE_SENTIMENT", "MEMORY_OPTIMIZATION"]:
        print(f"  {key}={os.environ.get(key, 'not set')}")
    print("\nPress Ctrl+C to stop the server\n")
    
    # Run the server with the environment variables set
    try:
        # Create a new environment with the updated variables
        env = os.environ.copy()
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\nServer stopped by user.")

if __name__ == "__main__":
    main() 