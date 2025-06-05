# BANDIT System Architecture & Troubleshooting Tracker

## System Overview

BANDIT (By Dark Matter Labs) is a professional AprilTag detection and pose estimation system designed for enterprise-grade dimensional measurement applications. The system combines computer vision, real-time web interfaces, and calibration APIs to provide accurate spatial measurements.

## Architecture Design Decisions

### 1. Modular Architecture
**Decision**: Separate detection engine from web interface
**Reasoning**: 
- Allows independent testing and development of core detection logic
- Enables potential CLI usage without web overhead
- Facilitates integration with other systems
- Improves maintainability and debugging

**Implementation**:
```
apriltag_detector.py (Core Engine) ← app.py (Web Server) ← index.html (Interface)
```

### 2. Flask + SocketIO for Real-time Communication
**Decision**: Use Flask with eventlet and SocketIO instead of pure WebRTC or WebSocket
**Reasoning**:
- Flask provides robust HTTP API for external system integration
- SocketIO handles WebSocket fallbacks and connection management
- eventlet enables concurrent handling without threading complexity
- Proven technology stack for real-time applications

**Trade-offs**:
- **Pro**: Easy development, excellent browser compatibility
- **Con**: Slightly higher latency than native WebSocket implementations

### 3. Multi-Camera Management Architecture
**Decision**: Global camera instance with dynamic switching
**Reasoning**:
- Prevents resource conflicts when switching cameras
- Centralizes camera state management
- Enables clean resource cleanup
- Supports hot-swapping without server restart

**Implementation**:
```python
# Global camera instance management
camera = None
def get_camera():
    global camera
    if camera is None:
        camera = VideoCamera(0)
    return camera
```

### 4. Calibration Data Persistence
**Decision**: JSON files instead of database
**Reasoning**:
- Simple deployment without database setup
- Human-readable calibration data
- Easy backup and transfer
- Version control friendly
- Adequate for single-user/single-site deployments

**File Structure**:
```
calibration/
├── current_calibration.json     # Active calibration
└── camera_calibration_*.json    # Historical calibrations
```

### 5. Professional Interface Design
**Decision**: Clean, enterprise-grade UI over flashy designs
**Reasoning**:
- Professional appearance for enterprise environments
- Reduces visual fatigue during long measurement sessions
- Consistent with enterprise software standards
- Improves user trust and adoption

**Design Principles**:
- Inter font for professional typography
- Subtle shadows and borders
- Consistent spacing and color palette
- Responsive design for various screen sizes

## Component Breakdown

### Core Detection Engine (`apriltag_detector.py`)

#### Purpose
Handles all computer vision operations including tag detection, pose estimation, and dimensional calculations.

#### Key Methods
- `detect_tags()`: Main detection pipeline
- `calculate_pixels_per_inch()`: PPI calculation for calibration
- `rotation_matrix_to_euler_angles()`: Pose conversion
- `draw_pose()`: Visualization overlay

#### Design Rationale
- **Single Responsibility**: Only handles computer vision tasks
- **Stateless Operation**: Each detection is independent
- **Configurable Parameters**: Camera matrix, tag size, families
- **Error Isolation**: CV errors don't affect web server

### Web Server (`app.py`)

#### Purpose
Provides web interface, API endpoints, and manages system state.

#### Key Components
- **VideoCamera Class**: Camera management and frame capture
- **Flask Routes**: HTTP API endpoints
- **SocketIO Handlers**: Real-time communication
- **Global State**: Camera instance and configuration

#### Design Patterns
- **Singleton Pattern**: Global camera instance
- **Observer Pattern**: WebSocket events for state changes
- **RESTful API**: Standard HTTP endpoints for integration

### Professional Interface (`templates/index.html`)

#### Purpose
Enterprise-grade user interface for system control and monitoring.

#### Architecture Decisions
- **Single Page Application**: No page reloads for smooth UX
- **Component-based Design**: Modular UI sections
- **Real-time Updates**: WebSocket integration
- **Responsive Layout**: CSS Grid for professional appearance

## Data Flow Architecture

### 1. Video Stream Pipeline
```
Camera → VideoCapture → AprilTag Detection → Pose Estimation → WebSocket → Browser
```

### 2. Calibration Pipeline
```
AprilTags → PPI Calculation → JSON Storage → API Access → External Systems
```

### 3. Camera Management Pipeline
```
System Scan → Camera List → User Selection → WebSocket Switch → Resource Cleanup
```

## Common Issues & Solutions

### Camera Detection Issues

#### Problem: No cameras detected
**Symptoms**: Camera Options panel shows "No cameras found"
**Root Cause**: Camera permissions, driver issues, or cameras in use by other applications
**Solution**:
```bash
# Check camera permissions (Linux)
ls -la /dev/video*

# Kill processes using camera
sudo fuser -k /dev/video0

# Test camera manually
ffmpeg -f v4l2 -list_devices true -i dummy  # Linux
```

#### Problem: Camera switching fails
**Symptoms**: "Failed to switch camera" notification
**Root Cause**: Incomplete resource cleanup or camera busy
**Solution**:
```python
# Implemented in VideoCamera.__del__()
if self.camera:
    self.camera.release()
```

### Video Feed Issues

#### Problem: Black video feed or "Failed to load resource"
**Symptoms**: Video container is black, browser console shows ERR_INCOMPLETE_CHUNKED_ENCODING
**Root Cause**: 
1. Camera not properly initialized
2. Codec issues with video streaming
3. Browser cache issues

**Solutions**:
1. **Server Side**:
```python
# Ensure proper frame encoding
ret, buffer = cv2.imencode('.jpg', frame)
if not ret:
    return self.get_dummy_frame()
```

2. **Client Side**:
- Hard refresh browser (Ctrl+F5)
- Clear browser cache
- Try incognito/private window

#### Problem: Low FPS or lag
**Symptoms**: Video appears choppy or delayed
**Root Cause**: Frame processing bottleneck or network issues
**Solutions**:
```python
# Buffer size optimization
self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Frame rate control
eventlet.sleep(0.03)  # ~30 FPS
```

### AprilTag Detection Issues

#### Problem: Tags not detected
**Symptoms**: No overlay graphics appear on visible tags
**Root Causes & Solutions**:

1. **Poor lighting**:
   - Ensure adequate, even lighting
   - Avoid shadows and glare
   - Use diffused lighting sources

2. **Wrong tag family**:
```python
# Verify tag family in detector
families = ["tag36h11", "tag25h9"]  # Supported families
```

3. **Incorrect tag size**:
```python
# Update tag size for calibration (6 inches = 0.1524m)
tag_size = 0.1524
```

4. **Tag quality issues**:
   - Print tags at high resolution (300+ DPI)
   - Use matte paper to reduce glare
   - Ensure tags are flat and uncrumpled

#### Problem: Inaccurate pose estimation
**Symptoms**: Coordinate axes appear misaligned or jittery
**Root Causes & Solutions**:

1. **Camera calibration**:
```python
# Update camera matrix for your specific camera
camera_matrix = np.array([
    [fx, 0, cx],    # fx, cx from camera calibration
    [0, fy, cy],    # fy, cy from camera calibration
    [0, 0, 1]
], dtype=np.float32)
```

2. **Tag distance**:
   - Keep tags within 0.5-2 meters for best accuracy
   - Avoid extreme angles (>45 degrees)

### Calibration System Issues

#### Problem: Calibration fails with "No AprilTags detected"
**Symptoms**: Calibration button gives error message
**Solution**:
1. Ensure 6" AprilTags are visible and well-lit
2. Check tag family compatibility
3. Verify camera is working properly

#### Problem: Inconsistent PPI measurements
**Symptoms**: Large variations in pixels-per-inch between calibrations
**Root Causes**:
1. **Camera zoom/focus changes**: Re-calibrate after any camera adjustments
2. **Varying distances**: Use multiple tags at similar distances
3. **Tag printing variations**: Verify printed tag dimensions

**Best Practices**:
- Use multiple tags simultaneously for averaging
- Maintain consistent camera settings
- Recalibrate when camera position changes

### API Integration Issues

#### Problem: External systems can't access calibration
**Symptoms**: API calls return connection errors
**Solutions**:
```bash
# Check server accessibility
curl -I http://localhost:5000/api/calibration

# Verify firewall settings
sudo ufw allow 5000  # Linux

# Check if server is binding to all interfaces
python app.py --host 0.0.0.0
```

#### Problem: Measurement API returns incorrect values
**Symptoms**: `/api/measure` returns unexpected distances
**Root Causes**:
1. **No calibration data**: Ensure camera is calibrated first
2. **Coordinate system confusion**: Verify pixel coordinates are correct
3. **Units mismatch**: API returns inches, cm, and mm

## Performance Optimization

### Memory Management
```python
# Proper resource cleanup in VideoCamera
def __del__(self):
    if self.camera:
        self.camera.release()
```

### CPU Optimization
```python
# Frame skip for better performance
self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Reduced detection frequency for UI updates
setInterval(() => {
    if (isDetecting) {
        socket.emit('get_frame');
    }
}, 100); // 10 FPS for tag data updates
```

### Network Optimization
```python
# Efficient video streaming
ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
```

## System Requirements

### Hardware
- **CPU**: Multi-core processor (2+ GHz recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **Camera**: USB 2.0+ webcam or industrial camera
- **Network**: 100Mbps for remote access

### Software Dependencies
- **Python**: 3.7+ (3.9+ recommended)
- **OpenCV**: 4.5+ with apriltag support
- **Flask**: 2.0+ for web framework
- **SocketIO**: 5.0+ for real-time communication

### Browser Compatibility
- **Chrome**: 80+ (recommended)
- **Firefox**: 75+
- **Safari**: 13+
- **Edge**: 80+

## Monitoring & Maintenance

### Log Analysis
```bash
# Monitor server logs
python app.py --debug

# Check for common error patterns
grep -i "error\|exception\|failed" server.log
```

### Performance Monitoring
- Monitor FPS counter in web interface
- Check CPU usage during operation
- Verify memory usage stays stable

### Preventive Maintenance
1. **Weekly**: Check calibration accuracy
2. **Monthly**: Clean camera lens and AprilTags
3. **Quarterly**: Update dependencies and review logs
4. **Annually**: Recalibrate system completely

## Security Considerations

### Network Security
- Run on private networks when possible
- Use HTTPS in production environments
- Implement authentication for sensitive deployments

### Data Protection
- Calibration files contain measurement data
- Video streams are not recorded by default
- API endpoints have no built-in authentication

## Future Enhancement Opportunities

### Planned Improvements
1. **Database Integration**: PostgreSQL for enterprise deployments
2. **User Authentication**: Multi-user support with role-based access
3. **Advanced Calibration**: Support for lens distortion correction
4. **Performance Metrics**: Built-in system monitoring dashboard
5. **Multi-Camera Sync**: Simultaneous multi-camera detection

### Integration Possibilities
1. **CAD Software**: Direct measurement export to AutoCAD, SolidWorks
2. **Quality Control**: Integration with manufacturing QC systems
3. **Robotics**: Real-time pose data for robotic guidance
4. **AR/VR**: Spatial tracking for augmented reality applications

## Conclusion

BANDIT represents a robust, professional-grade AprilTag detection system designed for enterprise dimensional measurement applications. The modular architecture, comprehensive error handling, and professional interface make it suitable for production environments while maintaining ease of use and integration capabilities.

The system's design prioritizes reliability, accuracy, and professional presentation while providing comprehensive APIs for external system integration. Regular maintenance and proper calibration ensure consistent performance in demanding measurement applications.