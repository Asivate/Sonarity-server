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
- Which sound recognition model to use (TensorFlow, AST, or PANNs)
- Which speech recognition system to use (Whisper or Google Cloud)

The server will always use port 8080 by default.

To use the interactive startup:

```bash
# On Linux/macOS
chmod +x start_interactive.sh
./start_interactive.sh

# Or directly with Python
python3 interactive_start.py
```

### Sound Recognition Models

The server supports three sound recognition models:

1. **TensorFlow Model** (Default): The traditional model used for general sound recognition
2. **AST (Audio Spectrogram Transformer)**: A newer model that may perform better for some specific sounds
3. **PANNs (Pretrained Audio Neural Networks)**: High-performance sound recognition using models pre-trained on AudioSet

#### Using the AST Model

To use the AST model:

1. Set the `USE_AST_MODEL` environment variable to `1`:
   ```bash
   # On Linux
   export USE_AST_MODEL=1
   python3 server.py
   
   # On Windows PowerShell
   $env:USE_AST_MODEL=1
   python server.py
   ```

2. Or use the interactive startup script which will set this for you:
   ```bash
   ./start_interactive.sh
   ```

3. Or use the toggle script to switch between models:
   ```bash
   # On Linux/macOS
   python3 toggle_ast.py on  # Enable AST model
   python3 toggle_ast.py off  # Disable AST model
   
   # On Windows
   python toggle_ast.py on  # Enable AST model
   python toggle_ast.py off  # Disable AST model
   
   # Without arguments it will toggle between enabled/disabled
   python toggle_ast.py  # Toggle the current setting
   ```
   The toggle script will update the environment variable and offer to restart the server with the new settings.

#### Using the PANNs Model

PANNs (Pretrained Audio Neural Networks) provides excellent sound recognition capabilities with models trained on AudioSet.

To use the PANNs model:

1. Set the `USE_PANNS_MODEL` environment variable to `1`:
   ```bash
   # On Linux
   export USE_PANNS_MODEL=1
   python3 server.py
   
   # On Windows PowerShell
   $env:USE_PANNS_MODEL=1
   python server.py
   ```

2. Or use the interactive startup script which will set this for you:
   ```bash
   ./start_interactive.sh
   ```

3. First-time usage will automatically download the model weights (approximately 85MB)

**Note**: When enabling PANNs, make sure both AST and TensorFlow models are disabled. The interactive startup script handles this automatically.

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

3. Or use the toggle script to switch between models:
   ```bash
   # On Linux/macOS
   python3 toggle_panns.py on    # Enable PANNs model
   python3 toggle_panns.py off   # Disable PANNs model

   # On Windows
   python toggle_panns.py on     # Enable PANNs model
   python toggle_panns.py off    # Disable PANNs model

   # Without arguments it will toggle between enabled/disabled
   python toggle_panns.py        # Toggle the current setting
   ```
   The toggle script will update the environment variable and offer to restart the server with the new settings.
