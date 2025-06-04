# AprilTag Detection & Pose Estimation

A real-time AprilTag detection system with 6DOF pose estimation and web-based visualization.

## Features

- Real-time AprilTag detection using the `tag36h11` family
- 6DOF pose estimation with translation and rotation
- Live video streaming to web browser
- Interactive web interface with detailed tag information
- Coordinate axis visualization overlaid on video
- Distance measurement and Euler angle calculation

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have a camera connected (webcam or USB camera)

## Usage

1. Start the server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. The system will automatically start detecting AprilTags in the camera feed

## Camera Calibration

The system uses default camera parameters suitable for most webcams. For better accuracy, you should calibrate your camera and update the camera matrix in `apriltag_detector.py`.

Default camera matrix (for 640x480 resolution):
```python
camera_matrix = np.array([
    [800, 0, 320],
    [0, 800, 240],
    [0, 0, 1]
], dtype=np.float32)
```

## AprilTag Setup

- Print AprilTag markers from: https://april.eecs.umich.edu/software/apriltag
- Use the `tag36h11` family for best compatibility
- Measure your printed tag size and update `tag_size` parameter in the detector (default: 5cm)

## System Architecture

- `apriltag_detector.py`: Core AprilTag detection and pose estimation
- `app.py`: Flask web server with SocketIO for real-time communication
- `templates/index.html`: Web frontend with live video and tag information
- Uses OpenCV for computer vision and PnP algorithm for pose estimation

## Coordinate System

- X-axis: Red arrow (right)
- Y-axis: Green arrow (up)  
- Z-axis: Blue arrow (forward, out of tag)
- Units: meters for translation, degrees for rotation