# CLAUDE.md
# Development guidance for Claude Code when working with the BANDIT codebase.
# Contains: Architecture overview, commands, key features, file structure, and development notes.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BANDIT - Multi-modal detection and measurement system developed by Dark Matter Labs. Combines AprilTag detection with AI object detection using RF-DETR. Features 6DOF pose estimation, camera calibration for dimensional measurements, multi-camera support, custom ML model integration, and enterprise-grade web interface with real-time visualization.

## Commands

```bash
# Install dependencies (use virtual environment)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set up ML models (optional)
python models/download_example_model.py     # Download test model
python models/model_validation.py          # Validate model setup

# Run the application with options
python app.py                    # Default settings
python app.py --source 1         # Use camera index 1
python app.py --port 8080        # Custom port
python app.py --debug            # Debug mode

# Access web interface
# Open browser to http://localhost:5000

# API testing
curl http://localhost:5000/api/cameras                      # List cameras
curl http://localhost:5000/api/calibration                  # Get calibration
curl http://localhost:5000/api/object_detection/status      # Object detection status
```

## Architecture

### AprilTag Detection Engine (`apriltag_detector.py`)
- **AprilTagDetector class**: Main detection and pose estimation
- **Tag families**: Supports `tag36h11` and `tag25h9`
- **6DOF pose**: Uses PnP algorithm with camera intrinsics
- **PPI calculation**: Pixel-to-inches ratio using 6" reference tags
- **Coordinate conversion**: Rotation matrices to Euler angles

### Object Detection Engine (`object_detector.py`)
- **RFDETRObjectDetector class**: AI object detection using RT-DETR
- **Custom weights**: Supports .pth model files
- **Configurable classes**: JSON-based class definitions
- **Two-tier control**: Enable detection + start/stop inference
- **GPU acceleration**: CUDA support with CPU fallback

### Web Server (`app.py`)
- **Flask application**: RESTful API and web interface
- **UnifiedVideoProcessor class**: Integrated camera, AprilTag, and ML processing
  - Single-pass frame processing for both detection types
  - Built-in performance tracking and FPS calculation
  - Optimized camera initialization with buffer control
- **MJPEG streaming**: Direct video delivery at `/video_feed`
- **SocketIO**: Real-time detection data push
- **API endpoints**: Camera, calibration, object detection management
- **File management**: Calibration and model configuration persistence

### Professional Interface (`templates/index.html`)
- **Dark Matter Labs branding**: Professional enterprise design
- **Real-time video**: Live streaming with pose and object overlays
- **Control panels**: Collapsible camera options and object detection
- **Interactive controls**: Detection, calibration, camera switching, ML inference
- **Professional notifications**: Success/error feedback with status indicators

## Key Features

1. **Multi-Camera Support**: Automatic detection and real-time switching
2. **Camera Calibration**: Professional PPI calculation and storage
3. **Dimensional Measurement**: API for external dimensioning systems
4. **AI Object Detection**: RF-DETR integration with custom model support
5. **Professional Interface**: Enterprise-grade UI with Dark Matter Labs branding
6. **Real-time Communication**: WebSocket for low-latency updates
7. **API Integration**: RESTful endpoints for external systems
8. **ML Model Management**: Custom weights, validation, and configuration

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
├── apriltag_detector.py        # AprilTag detection engine  
├── object_detector.py          # RF-DETR object detection
├── templates/index.html        # Professional web interface
├── models/                     # ML model weights and configs
│   ├── rf_detr_custom.pth     # Custom model weights
│   ├── rf_detr_config.json    # Model configuration
│   ├── README.md              # Model setup guide
│   ├── model_validation.py    # Validation script
│   └── download_example_model.py # Example downloader
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

## Git Commit Guidelines

**CRITICAL**: Never include Claude/AI attribution, flair, or signatures in commit messages
- NO "🤖 Generated with [Claude Code]" or similar attribution
- NO "Co-Authored-By: Claude" signatures  
- NO AI-related comments or acknowledgments
- Use conventional commit format: `type: description`
- Keep messages concise and professional
- Focus on what changed and why, not who wrote it
- Commit messages should appear as if written by a human developer

## Documentation Maintenance Before Commits

**IMPORTANT**: Always update relevant documentation files before committing code changes.

### Documentation File Responsibilities

#### README.md - Main Project Documentation
**Update when**: New features, installation changes, API changes, major system updates
**Contains**: 
- Feature overview and capabilities
- Installation and setup instructions
- Quick start guide and basic usage
- API examples and command-line options
- High-level system architecture overview

#### TECHNICAL_DOCUMENTATION.md - Deep Technical Reference
**Update when**: Algorithm changes, architecture modifications, performance optimizations, new technical implementations
**Contains**:
- Detailed architecture and design decisions
- Mathematical foundations and algorithms
- Complete API specifications with examples
- Performance optimization techniques
- Advanced usage patterns and extensions
- Troubleshooting for technical issues

#### USAGE.md - User Guide and Practical Instructions
**Update when**: New usage patterns, configuration options, workflow changes
**Contains**:
- Step-by-step usage instructions
- Configuration and setup details
- Common usage patterns and workflows
- Tips for optimal performance
- Basic troubleshooting for end users

#### tracker.md - Architecture Decisions and System Tracking
**Update when**: Design decisions, system architecture changes, component modifications, known issues, **ALL ENCOUNTERED ERRORS**
**Contains**:
- Architecture design rationale and decisions
- Component breakdown and responsibilities
- Common issues with detailed solutions
- Performance monitoring and maintenance
- System requirements and dependencies
- Future enhancement plans
- **Complete error documentation with causes and fixes**

### Pre-Commit Documentation Checklist

Before any git commit involving code changes:

1. **Feature Changes**: Update README.md with new capabilities
2. **Technical Changes**: Update TECHNICAL_DOCUMENTATION.md with implementation details
3. **Usage Changes**: Update USAGE.md with new instructions or workflows
4. **Architecture Changes**: Update tracker.md with design decisions and rationale
5. **API Changes**: Update both README.md (examples) and TECHNICAL_DOCUMENTATION.md (specifications)

### Documentation Quality Standards

- Keep technical accuracy high - verify all code examples work
- Maintain professional tone consistent with enterprise software
- Use clear, concise language without jargon when possible
- Include practical examples and real-world usage scenarios
- Ensure documentation reflects current system capabilities accurately

### Error Documentation Protocol

**CRITICAL**: Document ALL encountered errors in tracker.md using this format:

```markdown
#### Problem: [Brief description]
**Symptoms**: [What the user sees/experiences]
**Root Cause**: [Technical explanation of why it happens]
**Solutions**:
1. [Primary fix with code if applicable]
2. [Alternative approaches]
3. [Prevention measures]
```

**Include for each error**:
- Exact error messages
- System conditions when occurred
- Environmental factors (OS, browser, hardware)
- Code snippets that caused/fixed the issue
- Related configuration settings
- Prevention strategies