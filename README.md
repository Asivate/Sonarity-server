# Sonarity Server

A Flask-based server application for audio analysis and recognition, designed to run on cloud environments.

## Features

- Real-time audio processing and recognition
- Speech-to-text transcription with sentiment analysis
- WebSocket communication for real-time client updates
- REST API endpoints for application status

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

- The server will be available at: `http://<server-ip>:8080`
- WebSocket endpoint: `ws://<server-ip>:8080`

## Note

The models folder is excluded from version control and will be downloaded at runtime as needed.
