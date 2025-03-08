#!/bin/bash
# Interactive startup script for SoundWatch Server
# This script sets up the environment and runs the interactive Python script

# Change to the directory containing this script
cd "$(dirname "$0")"

# Set up environment variables for Google Cloud credentials if available
if [ -f "/home/hirwa0250/asivate-452914-5c12101797af.json" ]; then
    export GOOGLE_APPLICATION_CREDENTIALS="/home/hirwa0250/asivate-452914-5c12101797af.json"
    echo -e "\e[32mSet Google Cloud credentials from: $GOOGLE_APPLICATION_CREDENTIALS\e[0m"
fi

# Make sure the interactive script is executable
chmod +x interactive_start.py

# Run the interactive startup script
python3 interactive_start.py

# Exit with the same status code as the Python script
exit $? 