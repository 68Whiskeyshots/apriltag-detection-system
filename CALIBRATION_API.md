# BANDIT Camera Calibration & Management API

*Professional AprilTag Detection System by Dark Matter Labs*

## Overview
The BANDIT system provides camera calibration functionality that calculates pixels-per-inch ratios using 6" AprilTags as reference objects. This calibration can then be used by external dimensioning systems.

## Calibration Process

### 1. Perform Calibration
**Endpoint:** WebSocket event `calibrate_camera`
**Method:** Emit via Socket.IO

**Requirements:**
- At least one 6" × 6" AprilTag must be visible in the camera view
- Tags should be reasonably flat and unoccluded

**Process:**
1. Click the "Calibrate Camera" button in the web interface
2. System detects all visible AprilTags
3. Calculates pixels-per-inch ratio from each tag
4. Averages the measurements for final calibration
5. Saves calibration data to `calibration/current_calibration.json`

### 2. Get Available Cameras
**Endpoint:** `GET /api/cameras`
**Response:**
```json
{
  "success": true,
  "cameras": [
    {
      "index": 0,
      "name": "Camera 0",
      "resolution": "640x480",
      "fps": 30.0,
      "is_current": false
    },
    {
      "index": 1,
      "name": "Camera 1", 
      "resolution": "1280x720",
      "fps": 30.0,
      "is_current": true
    }
  ],
  "current_camera": 1
}
```

### 3. Switch Camera
**Endpoint:** WebSocket event `switch_camera`
**Request:**
```javascript
socket.emit('switch_camera', {camera_index: 1});
```
**Response:**
```javascript
socket.on('camera_switch_result', function(data) {
  // data.success: boolean
  // data.message: string
  // data.camera_index: number (if successful)
});
```

### 4. Get Current Calibration
**Endpoint:** `GET /api/calibration`
**Response:**
```json
{
  "success": true,
  "calibration": {
    "timestamp": "2024-06-05T00:30:15.123456",
    "calibrated_pixels_per_inch": 85.2,
    "tags_used": 2,
    "individual_measurements": [
      {
        "tag_id": 1,
        "ppi_width": 84.5,
        "ppi_height": 85.1,
        "ppi_average": 84.8,
        "tag_size_pixels": {
          "width": 507.2,
          "height": 510.6
        },
        "distance": 0.45
      }
    ],
    "camera_resolution": {
      "width": 640,
      "height": 480
    },
    "accuracy_notes": "Calibrated using 2 AprilTag(s) at various distances"
  }
}
```

### 5. Measure Distance Between Points
**Endpoint:** `POST /api/measure`
**Request Body:**
```json
{
  "point1": {"x": 100, "y": 150},
  "point2": {"x": 200, "y": 250}
}
```

**Response:**
```json
{
  "success": true,
  "measurement": {
    "pixel_distance": 141.42,
    "inch_distance": 1.66,
    "cm_distance": 4.21,
    "mm_distance": 42.1
  }
}
```

## Integration with External Systems

### Example: Python Integration
```python
import requests
import json
import socketio

# Create SocketIO client for real-time features
sio = socketio.Client()

@sio.on('camera_switch_result')
def on_camera_switch(data):
    print(f"Camera switch: {data['message']}")

# Connect to BANDIT server
sio.connect('http://localhost:5000')

# Get available cameras
cameras_response = requests.get('http://localhost:5000/api/cameras')
cameras = cameras_response.json()

if cameras['success']:
    print("Available cameras:")
    for camera in cameras['cameras']:
        current = " (Current)" if camera['is_current'] else ""
        print(f"  {camera['name']}: {camera['resolution']} @ {camera['fps']} FPS{current}")

# Switch to a different camera
if len(cameras['cameras']) > 1:
    sio.emit('switch_camera', {'camera_index': 1})

# Check if calibration exists
response = requests.get('http://localhost:5000/api/calibration')
calibration = response.json()

if calibration['success']:
    ppi = calibration['calibration']['calibrated_pixels_per_inch']
    print(f"Camera calibrated at {ppi:.2f} pixels per inch")
    
    # Measure distance between two points
    measurement = requests.post('http://localhost:5000/api/measure', 
        json={
            "point1": {"x": 100, "y": 100},
            "point2": {"x": 200, "y": 200}
        })
    
    result = measurement.json()
    if result['success']:
        print(f"Distance: {result['measurement']['inch_distance']:.2f} inches")
        print(f"Distance: {result['measurement']['cm_distance']:.2f} cm")
        print(f"Distance: {result['measurement']['mm_distance']:.2f} mm")

sio.disconnect()
```

### Example: JavaScript Integration
```javascript
// Connect to BANDIT WebSocket
const socket = io('http://localhost:5000');

// Get available cameras
fetch('/api/cameras')
  .then(response => response.json())
  .then(data => {
    if (data.success) {
      console.log('Available cameras:', data.cameras);
      // Switch to camera 1 if available
      if (data.cameras.length > 1) {
        socket.emit('switch_camera', {camera_index: 1});
      }
    }
  });

// Listen for camera switch results
socket.on('camera_switch_result', function(data) {
  console.log('Camera switch result:', data.message);
});

// Get current calibration
fetch('/api/calibration')
  .then(response => response.json())
  .then(data => {
    if (data.success) {
      const ppi = data.calibration.calibrated_pixels_per_inch;
      console.log(`Calibrated at ${ppi.toFixed(2)} PPI`);
      console.log(`Using ${data.calibration.tags_used} tags`);
    }
  });

// Perform calibration via WebSocket
socket.emit('calibrate_camera');
socket.on('calibration_result', function(data) {
  if (data.success) {
    console.log('Calibration successful:', data.calibration_data);
  } else {
    console.error('Calibration failed:', data.message);
  }
});

// Measure distance between points
fetch('/api/measure', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    point1: {x: 100, y: 150},
    point2: {x: 300, y: 450}
  })
})
.then(response => response.json())
.then(data => {
  if (data.success) {
    console.log(`Distance: ${data.measurement.inch_distance.toFixed(2)} inches`);
    console.log(`Distance: ${data.measurement.cm_distance.toFixed(2)} cm`);
    console.log(`Distance: ${data.measurement.mm_distance.toFixed(2)} mm`);
  }
});
```

## Calibration Files

### File Locations
- **Current calibration:** `calibration/current_calibration.json`
- **Historical calibrations:** `calibration/camera_calibration_YYYYMMDD_HHMMSS.json`

### File Format
```json
{
  "timestamp": "2024-06-05T00:30:15.123456",
  "calibrated_pixels_per_inch": 85.2,
  "tags_used": 2,
  "individual_measurements": [
    {
      "tag_id": 1,
      "ppi_width": 84.5,
      "ppi_height": 85.1,
      "ppi_average": 84.8,
      "tag_size_pixels": {
        "width": 507.2,
        "height": 510.6
      },
      "distance": 0.45
    }
  ],
  "camera_resolution": {
    "width": 640,
    "height": 480
  },
  "accuracy_notes": "Calibrated using 2 AprilTag(s) at various distances"
}
```

## Best Practices

1. **Multiple Tags:** Use multiple AprilTags for better accuracy
2. **Various Distances:** Include tags at different distances from camera
3. **Flat Mounting:** Ensure tags are mounted flat and perpendicular to camera
4. **Good Lighting:** Adequate lighting improves detection accuracy
5. **Recalibration:** Recalibrate if camera position or zoom changes

## Error Handling

- **No tags detected:** Ensure AprilTags are visible and properly lit
- **Calibration file errors:** Check write permissions in calibration directory
- **Measurement errors:** Verify calibration exists before measuring