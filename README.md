# Sonarity Server

A Flask-based server application for audio analysis and recognition, designed to run on cloud environments.

## Features

- Real-time audio processing and recognition
- Speech-to-text transcription with sentiment analysis
- WebSocket communication for real-time client updates
- REST API endpoints for application status

## New Features

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
