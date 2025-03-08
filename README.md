# Sonarity Server

A Flask-based server application for audio analysis and recognition, designed to run on cloud environments.

## Features

- Real-time audio processing and recognition
- Speech-to-text transcription with sentiment analysis
- WebSocket communication for real-time client updates
- REST API endpoints for application status

## New Features

### Interactive Startup

The server now includes an interactive startup option that lets you choose:
- Which sound recognition model to use (TensorFlow or AST)
- Which speech recognition system to use (Whisper or Google Cloud)
- Port number
- Debug mode

To use the interactive startup:

```bash
# On Linux/macOS
chmod +x start_interactive.sh
./start_interactive.sh

# Or directly with Python
python3 interactive_start.py
```

### Speech Recognition Options

The server now supports two speech recognition systems:

1. **Whisper AI** (Default): OpenAI's Whisper model for local speech recognition
2. **Google Cloud Speech-to-Text**: Google's cloud-based speech recognition service

#### Using Google Cloud Speech-to-Text

To use Google Cloud Speech-to-Text:

1. Make sure the Google Cloud credentials file (`asivate-452914-3c56106e7a07.json`) is in the server directory.

2. Start the server with the `--use-google-speech` flag:
   ```
   python server.py --use-google-speech
   ```

3. You can also toggle between speech recognition systems at runtime by making a POST request to the API:
   ```
   curl -X POST http://<server-ip>:8080/api/toggle-speech-recognition -H "Content-Type: application/json" -d '{"use_google_speech": true}'
   ```

4. Check current speech recognition system status:
   ```
   curl http://<server-ip>:8080/status
   ```

#### Advantages of Each System

- **Whisper AI**: Works offline, doesn't require internet connection, no usage limits
- **Google Cloud Speech-to-Text**: Higher accuracy, better handling of accents, real-time streaming capability, better noise handling

## Using Google Cloud Credentials

There are two ways to provide Google Cloud credentials for Speech-to-Text functionality:

### Method 1: Credentials File (Traditional)

Place your Google Cloud service account key file in one of these locations:
- In the server directory: `asivate-452914-5c12101797af.json`
- In your home directory (Linux): `/home/hirwa0250/asivate-452914-5c12101797af.json`

The server will automatically find and use this file.

### Method 2: Environment Variables (Recommended)

For better security, use the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to specify the path to your credentials file:

#### On Linux (Debian/Ubuntu):

```bash
# Set the environment variable
export GOOGLE_APPLICATION_CREDENTIALS="/home/hirwa0250/asivate-452914-5c12101797af.json"

# Start the server with Google Speech
python3 server.py --use-google-speech
```

#### On Linux using the provided script:

```bash
# Make the script executable (first time only)
chmod +x start_with_credentials.sh

# Run with default credentials location and Google Speech enabled
./start_with_credentials.sh -g

# Or specify a custom credentials path
./start_with_credentials.sh -c /path/to/your/credentials.json -g
```

#### On Windows (PowerShell):

```powershell
# Set the environment variable
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your\credentials.json"

# Start the server with Google Speech
python server.py --use-google-speech
```

#### On Windows using the provided script:

```powershell
# Run the script with your credentials file path
.\start_with_credentials.ps1 -CredentialsPath "C:\path\to\your\credentials.json" -UseGoogleSpeech
```

This approach keeps your credentials more secure and easier to update without modifying code.

## Setup

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/Asivate/Sonarity-server.git
   cd Sonarity-server
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Linux/Mac
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the server:
   ```
   python server.py
   ```

## Deployment on Google Cloud VM

1. Create a VM instance on Google Cloud Platform
2. Install required dependencies
3. Clone the repository
4. Set up the virtual environment and install dependencies
5. Configure the server to run on startup

## Usage

- The server will be available at: `http://<server-ip>:5000`
- WebSocket endpoint: `ws://<server-ip>:5000`

## Note

The models folder is excluded from version control and will be downloaded at runtime as needed.

### Improved Speech Recognition

The speech recognition has been improved:
- Captures more context by using more audio chunks (5 instead of 3)
- Uses longer minimum audio duration (1.5 seconds instead of 0.5)
- Google Speech-to-Text now uses the "video" model which handles shorter utterances better
- Added speech adaptation for common phrases
