# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BANDIT - Professional AprilTag detection and pose estimation system developed by Dark Matter Labs. Features 6DOF pose estimation, camera calibration for dimensional measurements, multi-camera support, and enterprise-grade web interface with real-time visualization.

## Commands

```bash
# Install dependencies (use virtual environment)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the application with options
python app.py                    # Default settings
python app.py --source 1         # Use camera index 1
python app.py --port 8080        # Custom port
python app.py --debug            # Debug mode

# Access web interface
# Open browser to http://localhost:5000

# API testing
curl http://localhost:5000/api/cameras      # List cameras
curl http://localhost:5000/api/calibration  # Get calibration
```

## Architecture

### Core Detection Engine (`apriltag_detector.py`)
- **AprilTagDetector class**: Main detection and pose estimation
- **Tag families**: Supports `tag36h11` and `tag25h9`
- **6DOF pose**: Uses PnP algorithm with camera intrinsics
- **PPI calculation**: Pixel-to-inches ratio using 6" reference tags
- **Coordinate conversion**: Rotation matrices to Euler angles

### Web Server (`app.py`)
- **Flask application**: RESTful API and web interface
- **VideoCamera class**: Multi-camera management and switching
- **SocketIO**: Real-time bidirectional communication
- **API endpoints**: Camera management, calibration, measurement
- **File management**: Calibration data persistence

### Professional Interface (`templates/index.html`)
- **Dark Matter Labs branding**: Professional enterprise design
- **Real-time video**: Live streaming with pose overlays
- **Camera options**: Collapsible panel for camera selection
- **Interactive controls**: Detection, calibration, camera switching
- **Professional notifications**: Success/error feedback

## Key Features

1. **Multi-Camera Support**: Automatic detection and real-time switching
2. **Camera Calibration**: Professional PPI calculation and storage
3. **Dimensional Measurement**: API for external dimensioning systems
4. **Professional Interface**: Enterprise-grade UI with Dark Matter Labs branding
5. **Real-time Communication**: WebSocket for low-latency updates
6. **API Integration**: RESTful endpoints for external systems

## Configuration

- **Tag families**: `tag36h11`, `tag25h9`
- **Tag size**: 6" × 6" (0.1524m) for calibration
- **Camera resolution**: 640×480 (configurable)
- **Server defaults**: Host 0.0.0.0, Port 5000
- **Calibration storage**: `calibration/` directory

## File Structure

```
apriltag_detect/
├── app.py                      # Main Flask application
├── apriltag_detector.py        # Core detection engine  
├── templates/index.html        # Professional web interface
├── calibration/               # Calibration data storage
│   ├── current_calibration.json
│   └── camera_calibration_*.json
├── requirements.txt           # Dependencies
├── README.md                 # Main documentation
├── CALIBRATION_API.md        # API reference
├── USAGE.md                  # Usage guide
├── TECHNICAL_DOCUMENTATION.md # Technical specs
└── tracker.md               # Architecture & troubleshooting
```

## Development Notes

- Uses eventlet for WebSocket support
- Professional color scheme: whites, grays, blues
- Inter font family for modern professional look
- Responsive design for multiple screen sizes
- Error handling with user-friendly notifications