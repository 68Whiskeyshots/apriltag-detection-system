import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, Response, jsonify, request, send_from_directory, abort
from flask_socketio import SocketIO, emit
import cv2
import json
import numpy as np
from apriltag_detector import AprilTagDetector
from object_detector import RFDETRObjectDetector, setup_model_directory
import base64
import os
from datetime import datetime
from werkzeug.utils import secure_filename
from PIL import Image
import mimetypes
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'apriltag_detection_secret'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file upload
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Image management configuration
IMAGES_DIR = 'images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'svg'}
THUMBNAIL_SIZES = [150, 300, 600]

class VideoCamera:
    def __init__(self, source=0):
        self.camera = None
        self.detector = AprilTagDetector()
        self.object_detector = RFDETRObjectDetector()
        self.source = source
        self.is_video_file = isinstance(source, str)
        self.object_detection_enabled = False  # Single control flag
        self.initialize_camera()
        
    def initialize_camera(self):
        """Initialize camera with fallback options"""
        if self.is_video_file:
            # Try to open video file
            try:
                self.camera = cv2.VideoCapture(self.source)
                if self.camera.isOpened():
                    ret, frame = self.camera.read()
                    if ret and frame is not None:
                        print(f"Successfully opened video file: {self.source}")
                        return
                    else:
                        self.camera.release()
                        self.camera = None
                        print(f"Could not read from video file: {self.source}")
                else:
                    print(f"Could not open video file: {self.source}")
            except Exception as e:
                print(f"Error opening video file {self.source}: {e}")
                if self.camera:
                    self.camera.release()
                    self.camera = None
        else:
            # Try camera indices
            camera_indices = [self.source, 0, 1, 2] if self.source != 0 else [0, 1, 2]
            
            for idx in camera_indices:
                try:
                    print(f"Trying camera index {idx}...")
                    self.camera = cv2.VideoCapture(idx)
                    
                    # Set buffer size to reduce latency (already implemented)
                    self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    if self.camera.isOpened():
                        # Test if we can read a frame
                        ret, frame = self.camera.read()
                        if ret and frame is not None:
                            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            self.source = idx
                            print(f"Successfully initialized camera at index {idx}")
                            
                            # Read and discard a few frames to stabilize (already implemented)
                            for _ in range(5):
                                self.camera.read()
                            return
                        else:
                            print(f"Could not read frame from camera {idx}")
                            self.camera.release()
                            self.camera = None
                    else:
                        print(f"Could not open camera {idx}")
                        if self.camera:
                            self.camera.release()
                            self.camera = None
                except Exception as e:
                    print(f"Error with camera {idx}: {e}")
                    if self.camera:
                        self.camera.release()
                        self.camera = None
        
        print("Warning: No camera found. Using dummy video feed.")
        
    def __del__(self):
        if self.camera:
            self.camera.release()
        # Clean up object detector threads
        if hasattr(self, 'object_detector'):
            self.object_detector.cleanup()
    
    def get_dummy_frame(self):
        """Generate a dummy frame when no camera is available"""
        # Create a black frame with text
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "No Camera Detected", (150, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        cv2.putText(frame, "Connect a camera and restart", (120, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Or use a video file", (180, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Encode frame as JPEG with explicit quality
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
        ret, buffer = cv2.imencode('.jpg', frame, encode_params)
        if ret:
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            return frame_base64, []
        return None, []
    
    def get_frame(self):
        if not self.camera or not self.camera.isOpened():
            dummy_frame, dummy_tags = self.get_dummy_frame()
            return dummy_frame, dummy_tags, []
            
        success, frame = self.camera.read()
        if not success:
            # If it's a video file, try to loop back to beginning
            if self.is_video_file:
                self.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = self.camera.read()
                if not success:
                    dummy_frame, dummy_tags = self.get_dummy_frame()
                    return dummy_frame, dummy_tags, []
            else:
                dummy_frame, dummy_tags = self.get_dummy_frame()
                return dummy_frame, dummy_tags, []
            
        # Detect AprilTags
        detected_tags = self.detector.detect_tags(frame)
        
        # Draw pose estimation on frame
        for tag in detected_tags:
            frame = self.detector.draw_pose(frame, tag)
        
        # Object detection (if enabled)
        detected_objects = []
        if self.object_detection_enabled and self.object_detector.is_available():
            detected_objects = self.object_detector.detect_objects(frame)
            # Draw object detections
            frame = self.object_detector.draw_detections(frame, detected_objects)
        
        # Encode frame as JPEG with explicit quality
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
        ret, buffer = cv2.imencode('.jpg', frame, encode_params)
        if not ret:
            return self.get_dummy_frame()
            
        # Convert to base64 for web transmission
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return frame_base64, detected_tags, detected_objects

class UnifiedVideoProcessor:
    """Unified video processor that handles both AprilTag and ML object detection"""
    def __init__(self, source=0):
        self.camera = None
        self.source = source
        self.is_video_file = isinstance(source, str)
        
        # Detectors
        self.apriltag_detector = AprilTagDetector()
        self.object_detector = RFDETRObjectDetector()
        
        # Control flags
        self.apriltag_enabled = True
        self.object_detection_enabled = False
        
        # Performance tracking
        self.frame_count = 0
        self.total_processing_time = 0
        self.last_fps_update = time.time()
        self.current_fps = 0
        
        # Initialize camera
        self.initialize_camera()
    
    def initialize_camera(self):
        """Initialize camera with optimizations from BANDIT"""
        if self.is_video_file:
            try:
                self.camera = cv2.VideoCapture(self.source)
                if self.camera.isOpened():
                    ret, frame = self.camera.read()
                    if ret and frame is not None:
                        print(f"Successfully opened video file: {self.source}")
                        return
                    else:
                        self.camera.release()
                        self.camera = None
            except Exception as e:
                print(f"Error opening video file {self.source}: {e}")
                if self.camera:
                    self.camera.release()
                    self.camera = None
        else:
            # Try camera indices
            camera_indices = [self.source, 0, 1, 2] if self.source != 0 else [0, 1, 2]
            
            for idx in camera_indices:
                try:
                    print(f"Trying camera index {idx}...")
                    self.camera = cv2.VideoCapture(idx)
                    
                    # Set buffer size to reduce latency
                    self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    if self.camera.isOpened():
                        ret, frame = self.camera.read()
                        if ret and frame is not None:
                            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            self.source = idx
                            print(f"Successfully initialized camera at index {idx}")
                            
                            # Frame stabilization
                            for _ in range(5):
                                self.camera.read()
                            return
                        else:
                            self.camera.release()
                            self.camera = None
                except Exception as e:
                    print(f"Error with camera {idx}: {e}")
                    if self.camera:
                        self.camera.release()
                        self.camera = None
        
        print("Warning: No camera found. Using dummy video feed.")
    
    def process_frame(self):
        """Process a single frame with both detectors"""
        start_time = time.time()
        
        if not self.camera or not self.camera.isOpened():
            return self.get_dummy_frame()
        
        success, frame = self.camera.read()
        if not success:
            if self.is_video_file:
                self.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = self.camera.read()
                if not success:
                    return self.get_dummy_frame()
            else:
                return self.get_dummy_frame()
        
        # Always run AprilTag detection (lightweight)
        detected_tags = []
        if self.apriltag_enabled:
            detected_tags = self.apriltag_detector.detect_tags(frame)
            for tag in detected_tags:
                frame = self.apriltag_detector.draw_pose(frame, tag)
        
        # Run object detection if enabled
        detected_objects = []
        if self.object_detection_enabled and self.object_detector.is_available():
            detected_objects = self.object_detector.detect_objects(frame)
            frame = self.object_detector.draw_detections(frame, detected_objects)
        
        # Performance tracking
        processing_time = (time.time() - start_time) * 1000  # ms
        self.total_processing_time += processing_time
        self.frame_count += 1
        
        # Update FPS every second
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.last_fps_update)
            self.frame_count = 0
            self.last_fps_update = current_time
        
        return frame, detected_tags, detected_objects
    
    def get_dummy_frame(self):
        """Generate a dummy frame when no camera is available"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "No Camera Detected", (150, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        cv2.putText(frame, "Connect a camera and restart", (120, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame, [], []
    
    def get_performance_stats(self):
        """Get performance statistics"""
        avg_processing_time = 0
        if self.frame_count > 0:
            avg_processing_time = self.total_processing_time / self.frame_count
        
        return {
            'fps': round(self.current_fps, 2),
            'avg_processing_time_ms': round(avg_processing_time, 2),
            'object_detection_stats': self.object_detector.get_performance_stats() if hasattr(self.object_detector, 'get_performance_stats') else {}
        }
    
    def cleanup(self):
        """Clean up resources"""
        if self.camera:
            self.camera.release()
        if hasattr(self.object_detector, 'cleanup'):
            self.object_detector.cleanup()
    
    def __del__(self):
        self.cleanup()

# Global instances
camera = None  # Keep for backward compatibility
unified_processor = None  # New unified processor

def get_camera():
    """Get or initialize the global camera instance (backward compatibility)"""
    global camera
    if camera is None:
        camera = VideoCamera(0)  # Default to camera 0
    return camera

def get_unified_processor():
    """Get or initialize the global unified processor"""
    global unified_processor
    if unified_processor is None:
        unified_processor = UnifiedVideoProcessor(0)
    return unified_processor

def serialize_tags(tags):
    """Convert tag data to JSON-serializable format"""
    serializable_tags = []
    for tag in tags:
        serializable_tag = {
            'id': int(tag['id']),
            'family': tag['family'],
            'center': tag['center'].tolist(),
            'corners': tag['corners'].tolist(),
            'distance': float(tag['distance']),
            'pose': {
                'translation': tag['pose']['translation'].tolist(),
                'euler_angles': tag['pose']['euler_angles'].tolist()
            },
            'pixels_per_inch': {
                'width': float(tag['pixels_per_inch']['width']),
                'height': float(tag['pixels_per_inch']['height']),
                'average': float(tag['pixels_per_inch']['average']),
                'tag_width_pixels': float(tag['pixels_per_inch']['tag_width_pixels']),
                'tag_height_pixels': float(tag['pixels_per_inch']['tag_height_pixels'])
            }
        }
        serializable_tags.append(serializable_tag)
    return serializable_tags

def serialize_objects(objects):
    """Convert object detection data to JSON-serializable format"""
    serializable_objects = []
    for obj in objects:
        serializable_obj = {
            'class': obj.get('class_name', obj.get('class', 'unknown')),
            'class_id': obj.get('class_id', -1),
            'confidence': float(obj['confidence']),
            'bbox': obj['bbox'],  # Already a list of floats
            'center': obj.get('center', []),
            'width': obj.get('width', 0),
            'height': obj.get('height', 0)
        }
        serializable_objects.append(serializable_obj)
    return serializable_objects

# Image Management Functions
def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_image_config():
    """Load image configuration from config.json"""
    config_path = os.path.join(IMAGES_DIR, 'config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Return default config if file doesn't exist
        return {
            "company": {"name": "Your Company", "main_logo": None},
            "products": {"categories": []},
            "display_settings": {"max_upload_size_mb": 10}
        }

def save_image_config(config):
    """Save image configuration to config.json"""
    config_path = os.path.join(IMAGES_DIR, 'config.json')
    os.makedirs(IMAGES_DIR, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def get_image_info(category, filename):
    """Get information about an image file"""
    filepath = os.path.join(IMAGES_DIR, category, filename)
    if not os.path.exists(filepath):
        return None
    
    stat = os.stat(filepath)
    info = {
        'filename': filename,
        'category': category,
        'size_bytes': stat.st_size,
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'mime_type': mimetypes.guess_type(filepath)[0]
    }
    
    # Try to get image dimensions
    try:
        if info['mime_type'] and info['mime_type'].startswith('image/'):
            with Image.open(filepath) as img:
                info['width'] = img.width
                info['height'] = img.height
                info['format'] = img.format
    except Exception:
        pass
    
    return info

def list_images(category=None):
    """List all images in a category or all categories"""
    images = []
    
    if category:
        # List images in specific category
        category_path = os.path.join(IMAGES_DIR, category)
        if os.path.exists(category_path):
            for filename in os.listdir(category_path):
                if filename.startswith('.'):
                    continue
                filepath = os.path.join(category_path, filename)
                if os.path.isfile(filepath) and allowed_file(filename):
                    info = get_image_info(category, filename)
                    if info:
                        images.append(info)
    else:
        # List all images
        if os.path.exists(IMAGES_DIR):
            for root, dirs, files in os.walk(IMAGES_DIR):
                # Skip hidden directories and files
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for filename in files:
                    if filename.startswith('.') or not allowed_file(filename):
                        continue
                    
                    # Get relative category path
                    rel_path = os.path.relpath(root, IMAGES_DIR)
                    if rel_path == '.':
                        continue
                    
                    info = get_image_info(rel_path, filename)
                    if info:
                        images.append(info)
    
    return sorted(images, key=lambda x: x['filename'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/tags')
def get_tags():
    """API endpoint to get current tag detection data"""
    cam = get_camera()
    frame_data, tags, objects = cam.get_frame()
    if tags is None:
        return jsonify({'error': 'No camera data'}), 500
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_tags = []
    for tag in tags:
        serializable_tag = {
            'id': int(tag['id']),
            'family': tag['family'],
            'center': tag['center'].tolist(),
            'corners': tag['corners'].tolist(),
            'distance': float(tag['distance']),
            'pose': {
                'translation': tag['pose']['translation'].tolist(),
                'euler_angles': tag['pose']['euler_angles'].tolist()
            },
            'pixels_per_inch': {
                'width': float(tag['pixels_per_inch']['width']),
                'height': float(tag['pixels_per_inch']['height']),
                'average': float(tag['pixels_per_inch']['average']),
                'tag_width_pixels': float(tag['pixels_per_inch']['tag_width_pixels']),
                'tag_height_pixels': float(tag['pixels_per_inch']['tag_height_pixels'])
            }
        }
        serializable_tags.append(serializable_tag)
    
    # Convert objects for JSON serialization
    serializable_objects = []
    for obj in objects:
        serializable_obj = {
            'class_id': obj['class_id'],
            'class_name': obj['class_name'],
            'confidence': obj['confidence'],
            'bbox': obj['bbox'],
            'center': obj['center'],
            'width': obj['width'],
            'height': obj['height']
        }
        serializable_objects.append(serializable_obj)
    
    return jsonify({
        'tags': serializable_tags,
        'objects': serializable_objects,
        'object_detection_enabled': cam.object_detection_enabled
    })

def generate_frames():
    """Generator function for MJPEG video streaming with unified processor"""
    processor = get_unified_processor()
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
    
    while True:
        try:
            # Process frame with unified processor
            frame, tags, objects = processor.process_frame()
            
            # Encode frame directly to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, encode_params)
            if ret:
                frame_bytes = buffer.tobytes()
                
                # Send detection data via SocketIO if needed
                if tags or objects:
                    socketio.emit('detection_data', {
                        'tags': serialize_tags(tags),
                        'objects': serialize_objects(objects),
                        'performance': processor.get_performance_stats()
                    }, room=None, broadcast=True)
                
                # Yield MJPEG frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Control frame rate (~30 FPS)
            time.sleep(0.033)
            
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            # Generate error frame
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Stream Error", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_frame, encode_params)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'msg': 'Connected to AprilTag detector'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('get_frame')
def handle_get_frame():
    """Send frame and tag data via WebSocket"""
    cam = get_camera()
    frame_data, tags, objects = cam.get_frame()
    if frame_data is not None and tags is not None:
        # Convert numpy arrays to lists for JSON serialization
        serializable_tags = []
        for tag in tags:
            serializable_tag = {
                'id': int(tag['id']),
                'family': tag['family'],
                'center': tag['center'].tolist(),
                'corners': tag['corners'].tolist(),
                'distance': float(tag['distance']),
                'pose': {
                    'translation': tag['pose']['translation'].tolist(),
                    'euler_angles': tag['pose']['euler_angles'].tolist()
                },
                'pixels_per_inch': {
                    'width': float(tag['pixels_per_inch']['width']),
                    'height': float(tag['pixels_per_inch']['height']),
                    'average': float(tag['pixels_per_inch']['average']),
                    'tag_width_pixels': float(tag['pixels_per_inch']['tag_width_pixels']),
                    'tag_height_pixels': float(tag['pixels_per_inch']['tag_height_pixels'])
                }
            }
            serializable_tags.append(serializable_tag)
        
        # Convert objects for JSON serialization
        serializable_objects = []
        for obj in objects:
            serializable_obj = {
                'class_id': obj['class_id'],
                'class_name': obj['class_name'],
                'confidence': obj['confidence'],
                'bbox': obj['bbox'],
                'center': obj['center'],
                'width': obj['width'],
                'height': obj['height']
            }
            serializable_objects.append(serializable_obj)
        
        emit('frame_data', {
            'image': frame_data,
            'tags': serializable_tags,
            'objects': serializable_objects,
            'object_detection_enabled': cam.object_detection_enabled
        })

@socketio.on('calibrate_camera')
def handle_calibrate_camera():
    """Calibrate camera using current AprilTag detections and save calibration data"""
    cam = get_camera()
    frame_data, tags, objects = cam.get_frame()
    
    if not tags or len(tags) == 0:
        emit('calibration_result', {
            'success': False,
            'message': 'No AprilTags detected. Please ensure at least one 6" AprilTag is visible in the camera view.'
        })
        return
    
    # Calculate average pixels per inch from all detected tags
    total_ppi = 0
    tag_count = len(tags)
    calibration_data = {
        'timestamp': datetime.now().isoformat(),
        'tags_used': tag_count,
        'individual_measurements': [],
        'camera_resolution': {
            'width': 640,
            'height': 480
        }
    }
    
    for tag in tags:
        ppi = tag['pixels_per_inch']
        total_ppi += ppi['average']
        calibration_data['individual_measurements'].append({
            'tag_id': tag['id'],
            'ppi_width': ppi['width'],
            'ppi_height': ppi['height'],
            'ppi_average': ppi['average'],
            'tag_size_pixels': {
                'width': ppi['tag_width_pixels'],
                'height': ppi['tag_height_pixels']
            },
            'distance': tag['distance']
        })
    
    # Calculate final calibrated pixels per inch
    calibrated_ppi = total_ppi / tag_count
    calibration_data['calibrated_pixels_per_inch'] = calibrated_ppi
    calibration_data['accuracy_notes'] = f'Calibrated using {tag_count} AprilTag(s) at various distances'
    
    # Save calibration to file
    try:
        os.makedirs('calibration', exist_ok=True)
        calibration_file = f'calibration/camera_calibration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(calibration_file, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        # Also save as the current/active calibration
        with open('calibration/current_calibration.json', 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        emit('calibration_result', {
            'success': True,
            'message': f'Camera calibrated successfully! Pixels per inch: {calibrated_ppi:.2f}',
            'calibration_data': {
                'pixels_per_inch': calibrated_ppi,
                'tags_used': tag_count,
                'timestamp': calibration_data['timestamp'],
                'filename': calibration_file
            }
        })
        
    except Exception as e:
        emit('calibration_result', {
            'success': False,
            'message': f'Failed to save calibration: {str(e)}'
        })

@app.route('/api/cameras')
def get_available_cameras():
    """Get list of available camera devices"""
    available_cameras = []
    
    # Test camera indices 0-5
    for index in range(6):
        try:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                # Try to read a frame to verify it's working
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Get camera properties
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    available_cameras.append({
                        'index': index,
                        'name': f'Camera {index}',
                        'resolution': f'{width}x{height}',
                        'fps': fps if fps > 0 else 'Unknown',
                        'is_current': index == (camera.source if camera else 0)
                    })
                cap.release()
        except:
            continue
    
    return jsonify({
        'success': True,
        'cameras': available_cameras,
        'current_camera': camera.source if camera else None
    })

@socketio.on('switch_camera')
def handle_switch_camera(data):
    """Switch to a different camera"""
    global unified_processor
    new_camera_index = data.get('camera_index', 0)
    
    try:
        # Release current processor
        if unified_processor:
            unified_processor.cleanup()
            unified_processor = None
        
        # Initialize new processor with the selected camera
        unified_processor = UnifiedVideoProcessor(new_camera_index)
        
        emit('camera_switch_result', {
            'success': True,
            'message': f'Switched to Camera {new_camera_index}',
            'camera_index': new_camera_index
        })
        
    except Exception as e:
        emit('camera_switch_result', {
            'success': False,
            'message': f'Failed to switch camera: {str(e)}'
        })

@socketio.on('toggle_object_detection')
def handle_toggle_object_detection(data):
    """Toggle object detection on/off"""
    processor = get_unified_processor()
    enabled = data.get('enabled', False)
    
    if enabled:
        processor.object_detector.enable()
        processor.object_detection_enabled = True
        print("Object detection enabled with async inference")
    else:
        processor.object_detector.disable()
        processor.object_detection_enabled = False
        print("Object detection disabled, async thread stopped")
    
    emit('object_detection_toggle_result', {
        'success': True,
        'enabled': processor.object_detection_enabled,
        'available': processor.object_detector.model is not None,
        'async_inference': True,
        'message': f"Object detection {'enabled' if enabled else 'disabled'} (unified processor)",
        'performance': processor.get_performance_stats()
    })

@app.route('/api/object_detection/status')
def get_object_detection_status():
    """Get object detection status and capabilities"""
    processor = get_unified_processor()
    return jsonify({
        'available': processor.object_detector.model is not None,
        'enabled': processor.object_detection_enabled,
        'class_names': processor.object_detector.get_class_names(),
        'confidence_threshold': processor.object_detector.confidence_threshold,
        'nms_threshold': processor.object_detector.nms_threshold,
        'async_inference': True,
        'debugging': {
            'enabled': processor.object_detector.debugging_enabled,
            'save_misclassifications': processor.object_detector.save_misclassifications,
            'misclassifications_dir': processor.object_detector.misclassifications_dir
        },
        'performance_stats': processor.get_performance_stats()
    })

@app.route('/api/object_detection/config', methods=['POST'])
def update_object_detection_config():
    """Update object detection configuration"""
    cam = get_camera()
    data = request.get_json()
    
    try:
        confidence = data.get('confidence_threshold')
        nms = data.get('nms_threshold')
        
        cam.object_detector.update_thresholds(confidence, nms)
        
        return jsonify({
            'success': True,
            'message': 'Configuration updated successfully (async inference)',
            'confidence_threshold': cam.object_detector.confidence_threshold,
            'nms_threshold': cam.object_detector.nms_threshold,
            'async_inference': True,
            'performance_stats': cam.object_detector.get_performance_stats()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to update configuration: {str(e)}'
        })

@app.route('/api/calibration')
def get_current_calibration():
    """Get the current camera calibration data"""
    try:
        if os.path.exists('calibration/current_calibration.json'):
            with open('calibration/current_calibration.json', 'r') as f:
                calibration_data = json.load(f)
            return jsonify({
                'success': True,
                'calibration': calibration_data
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No calibration data found. Please calibrate the camera first.'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error reading calibration: {str(e)}'
        })

@app.route('/api/measure', methods=['POST'])
def measure_distance():
    """Use calibration to measure distances between points in pixels"""
    try:
        data = request.get_json()
        if not os.path.exists('calibration/current_calibration.json'):
            return jsonify({
                'success': False,
                'message': 'No calibration data available. Please calibrate camera first.'
            })
        
        with open('calibration/current_calibration.json', 'r') as f:
            calibration = json.load(f)
        
        pixels_per_inch = calibration['calibrated_pixels_per_inch']
        
        # Calculate distance between two points in pixels
        x1, y1 = data['point1']['x'], data['point1']['y']
        x2, y2 = data['point2']['x'], data['point2']['y']
        
        pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        inch_distance = pixel_distance / pixels_per_inch
        
        return jsonify({
            'success': True,
            'measurement': {
                'pixel_distance': pixel_distance,
                'inch_distance': inch_distance,
                'cm_distance': inch_distance * 2.54,
                'mm_distance': inch_distance * 25.4
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Measurement error: {str(e)}'
        })

@app.route('/api/measure_rectangle', methods=['POST'])
def measure_rectangle():
    """Use calibration to measure rectangle dimensions in pixels"""
    try:
        data = request.get_json()
        if not os.path.exists('calibration/current_calibration.json'):
            return jsonify({
                'success': False,
                'message': 'No calibration data available. Please calibrate camera first.'
            })
        
        with open('calibration/current_calibration.json', 'r') as f:
            calibration = json.load(f)
        
        pixels_per_inch = calibration['calibrated_pixels_per_inch']
        
        # Get rectangle corners
        x1, y1 = data['point1']['x'], data['point1']['y']
        x2, y2 = data['point2']['x'], data['point2']['y']
        
        # Calculate width and height in pixels
        width_pixels = abs(x2 - x1)
        height_pixels = abs(y2 - y1)
        area_pixels = width_pixels * height_pixels
        
        # Convert to real world measurements
        width_inches = width_pixels / pixels_per_inch
        height_inches = height_pixels / pixels_per_inch
        area_square_inches = area_pixels / (pixels_per_inch ** 2)
        
        # Convert to metric
        width_mm = width_inches * 25.4
        height_mm = height_inches * 25.4
        width_cm = width_inches * 2.54
        height_cm = height_inches * 2.54
        area_cm2 = area_square_inches * 6.4516  # 1 sq inch = 6.4516 sq cm
        
        # Convert to feet for large measurements
        area_sqft = area_square_inches / 144  # 144 sq inches = 1 sq ft
        
        return jsonify({
            'success': True,
            'measurement': {
                'width_pixels': width_pixels,
                'height_pixels': height_pixels,
                'area_pixels': area_pixels,
                'width_in': width_inches,
                'height_in': height_inches,
                'area_sqin': area_square_inches,
                'area_sqft': area_sqft,
                'width_mm': width_mm,
                'height_mm': height_mm,
                'width_cm': width_cm,
                'height_cm': height_cm,
                'area_cm2': area_cm2
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Rectangle measurement error: {str(e)}'
        })

# Image Management Routes
@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve images from the images directory"""
    try:
        # Security check - prevent directory traversal
        if '..' in filename or filename.startswith('/'):
            abort(404)
        
        return send_from_directory(IMAGES_DIR, filename)
    except FileNotFoundError:
        abort(404)

@app.route('/api/images/list')
@app.route('/api/images/list/<category>')
def api_list_images(category=None):
    """API endpoint to list images"""
    try:
        images = list_images(category)
        return jsonify({
            'success': True,
            'category': category,
            'images': images,
            'count': len(images)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error listing images: {str(e)}'
        }), 500

@app.route('/api/images/info/<category>/<filename>')
def api_image_info(category, filename):
    """Get detailed information about a specific image"""
    try:
        info = get_image_info(category, filename)
        if info:
            return jsonify({
                'success': True,
                'image': info
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Image not found'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting image info: {str(e)}'
        }), 500

@app.route('/api/images/config')
def api_get_image_config():
    """Get image configuration"""
    try:
        config = load_image_config()
        return jsonify({
            'success': True,
            'config': config
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error loading config: {str(e)}'
        }), 500

@app.route('/api/images/config', methods=['POST'])
def api_update_image_config():
    """Update image configuration"""
    try:
        data = request.get_json()
        config = load_image_config()
        
        # Update config with provided data
        if 'company' in data:
            config['company'].update(data['company'])
        if 'products' in data:
            config['products'].update(data['products'])
        if 'display_settings' in data:
            config['display_settings'].update(data['display_settings'])
        
        save_image_config(config)
        
        return jsonify({
            'success': True,
            'message': 'Configuration updated successfully',
            'config': config
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error updating config: {str(e)}'
        }), 500

@app.route('/api/images/upload/<category>', methods=['POST'])
def api_upload_image(category):
    """Upload image to specified category"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No file provided'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'message': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Secure the filename
        filename = secure_filename(file.filename)
        
        # Create category directory if it doesn't exist
        category_path = os.path.join(IMAGES_DIR, category)
        os.makedirs(category_path, exist_ok=True)
        
        # Save the file
        filepath = os.path.join(category_path, filename)
        file.save(filepath)
        
        # Get info about the uploaded file
        info = get_image_info(category, filename)
        
        return jsonify({
            'success': True,
            'message': f'Image uploaded successfully to {category}',
            'image': info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error uploading image: {str(e)}'
        }), 500

@app.route('/api/images/delete/<category>/<filename>', methods=['DELETE'])
def api_delete_image(category, filename):
    """Delete an image"""
    try:
        filepath = os.path.join(IMAGES_DIR, category, filename)
        
        if not os.path.exists(filepath):
            return jsonify({
                'success': False,
                'message': 'Image not found'
            }), 404
        
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'message': f'Image {filename} deleted successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error deleting image: {str(e)}'
        }), 500

@app.route('/api/images/categories')
def api_list_categories():
    """List all available image categories"""
    try:
        categories = []
        if os.path.exists(IMAGES_DIR):
            for item in os.listdir(IMAGES_DIR):
                item_path = os.path.join(IMAGES_DIR, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    # Count images in category
                    image_count = len(list_images(item))
                    categories.append({
                        'name': item,
                        'path': item,
                        'image_count': image_count
                    })
        
        return jsonify({
            'success': True,
            'categories': sorted(categories, key=lambda x: x['name'])
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error listing categories: {str(e)}'
        }), 500

@app.route('/api/object_detection/debugging', methods=['POST'])
def toggle_debugging():
    """Toggle RF-DETR debugging mode"""
    try:
        cam = get_camera()
        data = request.get_json()
        
        enabled = data.get('enabled', False)
        save_misclassifications = data.get('save_misclassifications', True)
        
        cam.object_detector.set_debugging(enabled, save_misclassifications)
        
        return jsonify({
            'success': True,
            'message': f'Debugging {"enabled" if enabled else "disabled"}',
            'debugging': {
                'enabled': cam.object_detector.debugging_enabled,
                'save_misclassifications': cam.object_detector.save_misclassifications,
                'misclassifications_dir': cam.object_detector.misclassifications_dir
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error toggling debugging: {str(e)}'
        }), 500

if __name__ == '__main__':
    import argparse
    
    # Setup model directory on startup
    setup_model_directory()
    
    parser = argparse.ArgumentParser(description='AprilTag Detection Server')
    parser.add_argument('--source', type=str, default=0,
                       help='Video source: camera index (0,1,2...) or video file path')
    parser.add_argument('--port', type=int, default=5000,
                       help='Server port (default: 5000)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Server host (default: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Convert source to int if it's a number
    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    
    print("Starting AprilTag Detection Server...")
    print(f"Video source: {source}")
    print(f"Server: http://{args.host}:{args.port}")
    print("Open your browser to view the detection")
    
    # Initialize camera with specified source
    camera = VideoCamera(source)
    
    socketio.run(app, host=args.host, port=args.port, debug=args.debug)