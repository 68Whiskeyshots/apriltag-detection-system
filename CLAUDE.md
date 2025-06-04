# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AprilTag detection and pose estimation system with real-time web visualization. The system detects AprilTag markers in a camera feed, calculates their 6DOF pose (position and orientation), and displays the results in a web browser with overlay graphics.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

# Access web interface
# Open browser to http://localhost:5000
```

## Architecture

- **apriltag_detector.py**: Core detection engine using OpenCV and AprilTag library
  - AprilTagDetector class handles tag detection and pose estimation
  - Uses PnP algorithm with camera intrinsics for 6DOF pose calculation
  - Converts rotation matrices to Euler angles for user-friendly display

- **app.py**: Flask web server with real-time communication
  - VideoCamera class manages camera capture and tag detection
  - Flask routes serve web interface and video streaming
  - SocketIO enables real-time tag data transmission to browser

- **templates/index.html**: Interactive web frontend
  - Live video feed with pose visualization overlays
  - Real-time tag information display (ID, distance, orientation)
  - WebSocket communication for low-latency updates

## Key Components

1. **Camera Calibration**: Uses camera intrinsic matrix for accurate pose estimation
2. **Pose Visualization**: Draws 3D coordinate axes on detected tags (RGB = XYZ)
3. **Real-time Streaming**: Video feed at ~30 FPS with tag overlays
4. **Web Interface**: Modern responsive design with detailed tag metrics

## Configuration

- Default tag family: `tag36h11`
- Default tag size: 5cm (configurable in AprilTagDetector)
- Camera resolution: 640x480 (configurable in VideoCamera)
- Default camera matrix optimized for standard webcams