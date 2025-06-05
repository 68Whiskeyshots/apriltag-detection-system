# BANDIT - AprilTag Detection & Pose Estimation System

Professional AprilTag detection system with 6DOF pose estimation, camera calibration, and enterprise-grade web interface. Developed by Dark Matter Labs.

## Features

### Core Detection
- Real-time AprilTag detection using `tag36h11` and `tag25h9` families
- 6DOF pose estimation with translation and rotation
- Sub-pixel accuracy corner detection
- Distance measurement and Euler angle calculation

### Camera Management
- Multi-camera support with automatic detection
- Real-time camera switching via web interface
- Camera resolution and FPS information
- Professional camera calibration system

### Dimensional Measurement
- Pixel-to-inches calibration using 6" AprilTags
- Real-time pixels-per-inch (PPI) calculation
- API endpoints for external dimensioning systems
- Calibration data persistence and management

### Professional Interface
- Enterprise-grade web interface with Dark Matter Labs branding
- Real-time video streaming with pose overlays
- Collapsible camera options panel
- Live statistics and tag information display
- Professional notifications and status indicators

## Installation

1. Clone the repository and navigate to the directory
2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure you have camera(s) connected (webcam or USB camera)

## Quick Start

1. Start the BANDIT server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. The system will automatically start detecting AprilTags

## Camera Setup & Calibration

### Camera Selection
- Click "Camera Options" in the web interface to view available cameras
- Switch between cameras in real-time
- View camera specifications (resolution, FPS)

### AprilTag Calibration
1. Print 6" × 6" AprilTags from: https://april.eecs.umich.edu/software/apriltag
2. Use `tag36h11` or `tag25h9` families
3. Place tags in camera view and click "Calibrate Camera"
4. System automatically calculates pixels-per-inch ratio
5. Calibration data saved to `calibration/` directory

## API Integration

### Get Current Calibration
```bash
curl http://localhost:5000/api/calibration
```

### Measure Distance Between Points
```bash
curl -X POST http://localhost:5000/api/measure \
  -H "Content-Type: application/json" \
  -d '{"point1": {"x": 100, "y": 150}, "point2": {"x": 200, "y": 250}}'
```

### List Available Cameras
```bash
curl http://localhost:5000/api/cameras
```

## Command Line Options

```bash
python app.py --help
```

Options:
- `--source`: Camera index or video file path (default: 0)
- `--port`: Server port (default: 5000)
- `--host`: Server host (default: 0.0.0.0)
- `--debug`: Enable debug mode

## System Architecture

### Core Components
- **apriltag_detector.py**: Detection engine with pose estimation and PPI calculation
- **app.py**: Flask server with WebSocket communication and API endpoints
- **templates/index.html**: Professional web interface with real-time updates

### Data Flow
1. Camera captures frames → AprilTag detection → Pose estimation → PPI calculation
2. WebSocket streams real-time data to browser
3. Calibration system saves measurement data for external use

### File Structure
```
apriltag_detect/
├── app.py                      # Main Flask application
├── apriltag_detector.py        # Core detection engine
├── templates/index.html        # Web interface
├── calibration/               # Calibration data storage
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── CALIBRATION_API.md        # API documentation
├── USAGE.md                  # Detailed usage guide
├── TECHNICAL_DOCUMENTATION.md # Technical specifications
└── tracker.md               # System architecture & troubleshooting
```

## Coordinate System

- **X-axis**: Red arrow (right)
- **Y-axis**: Green arrow (up)  
- **Z-axis**: Blue arrow (forward, out of tag)
- **Units**: Meters for translation, degrees for rotation, PPI for calibration

## Support & Documentation

- **API Reference**: See `CALIBRATION_API.md`
- **Usage Guide**: See `USAGE.md`
- **Technical Details**: See `TECHNICAL_DOCUMENTATION.md`
- **Troubleshooting**: See `tracker.md`