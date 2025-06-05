# BANDIT - Multi-Modal Detection & Measurement System
# Main documentation covering installation, features, API usage, and system overview.
# Contains: Feature list, setup instructions, API examples, and file structure.

Professional detection system combining AprilTag pose estimation with AI object detection. Features 6DOF pose estimation, camera calibration, RF-DETR integration, and enterprise-grade web interface. Developed by Dark Matter Labs.

## Features

### AprilTag Detection
- Real-time AprilTag detection using `tag36h11` and `tag25h9` families
- 6DOF pose estimation with translation and rotation
- Sub-pixel accuracy corner detection
- Distance measurement and Euler angle calculation

### AI Object Detection
- RF-DETR (Real-time Detection Transformer) integration
- Custom model weight support (.pth files)
- Configurable object classes and detection parameters
- Real-time inference with GPU acceleration
- Two-tier control system (enable detection + start inference)

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
- Collapsible camera options and object detection panels
- Live statistics and detection information display
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

4. Set up RF-DETR model (optional):
```bash
# Download example model for testing
python models/download_example_model.py

# OR place your custom trained model
cp /path/to/your/model.pth models/rf_detr_custom.pth
```

5. Ensure you have camera(s) connected (webcam or USB camera)

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

4. For object detection (optional):
   - Go to "Object Detection" panel in the web interface
   - Enable object detection
   - Click "Start Inference" to begin AI detection

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

### Object Detection Status
```bash
curl http://localhost:5000/api/object_detection/status
```

### Configure Object Detection
```bash
curl -X POST http://localhost:5000/api/object_detection/config \
  -H "Content-Type: application/json" \
  -d '{"confidence_threshold": 0.7, "nms_threshold": 0.3}'
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
- **apriltag_detector.py**: AprilTag detection engine with pose estimation and PPI calculation
- **object_detector.py**: RF-DETR object detection module with custom weight support
- **app.py**: Flask server with WebSocket communication and API endpoints
- **templates/index.html**: Professional web interface with real-time updates

### Data Flow
1. Camera captures frames → AprilTag detection → Object detection (if enabled) → Pose estimation → PPI calculation
2. WebSocket streams real-time data to browser
3. Calibration system saves measurement data for external use
4. ML inference runs independently with start/stop control

### File Structure
```
apriltag_detect/
├── app.py                      # Main Flask application
├── apriltag_detector.py        # AprilTag detection engine
├── object_detector.py          # RF-DETR object detection
├── templates/index.html        # Web interface
├── models/                     # ML model weights and configs
│   ├── rf_detr_custom.pth     # Your custom model weights
│   ├── rf_detr_config.json    # Model configuration
│   ├── README.md              # Model setup guide
│   ├── model_validation.py    # Validation script
│   └── download_example_model.py # Example downloader
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