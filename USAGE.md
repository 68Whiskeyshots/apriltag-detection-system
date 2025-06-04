# AprilTag Detection Usage Guide

## Quick Start

1. **Basic camera detection:**
```bash
python app.py
```

2. **Specify camera index:**
```bash
python app.py --source 1
```

3. **Use a video file:**
```bash
python app.py --source /path/to/video.mp4
```

4. **Run on different port:**
```bash
python app.py --port 8080
```

## Supported Tag Families

The system supports both:
- **tag36h11** (default, more robust)
- **tag25h9** (smaller, faster processing)

Both families are detected simultaneously.

## Camera Issues

If you see "No Camera Detected":

1. **Check camera availability:**
```bash
python test_camera.py
```

2. **Try different camera index:**
```bash
python app.py --source 1
python app.py --source 2
```

3. **Check camera permissions:**
```bash
ls -la /dev/video*
```

## AprilTag Generation

1. **Download tags:** https://april.eecs.umich.edu/software/apriltag
2. **Print recommended families:**
   - tag36h11 (recommended)
   - tag25h9 (also supported)
3. **Measure your printed tag size and update tag_size in apriltag_detector.py**

## Web Interface

1. **Open browser:** http://localhost:5000
2. **Features:**
   - Live video feed with pose overlays
   - Real-time tag detection statistics
   - 6DOF pose information (position + orientation)
   - Tag family identification

## Coordinate System

- **X-axis:** Red arrow (right)
- **Y-axis:** Green arrow (up)
- **Z-axis:** Blue arrow (forward/out of tag)
- **Units:** meters for position, degrees for rotation

## Performance Tips

- Use 640x480 resolution for best performance
- Ensure good lighting conditions
- Keep tags flat and unobstructed
- Optimal distance: 0.5-3 meters from camera