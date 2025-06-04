import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import cv2
import json
import numpy as np
from apriltag_detector import AprilTagDetector
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'apriltag_detection_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

class VideoCamera:
    def __init__(self, source=0):
        self.camera = None
        self.detector = AprilTagDetector()
        self.source = source
        self.is_video_file = isinstance(source, str)
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
                    
                    # Set buffer size to reduce latency
                    self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    if self.camera.isOpened():
                        # Test if we can read a frame
                        ret, frame = self.camera.read()
                        if ret and frame is not None:
                            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            self.source = idx
                            print(f"Successfully initialized camera at index {idx}")
                            
                            # Read and discard a few frames to stabilize
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
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            return frame_base64, []
        return None, []
    
    def get_frame(self):
        if not self.camera or not self.camera.isOpened():
            return self.get_dummy_frame()
            
        success, frame = self.camera.read()
        if not success:
            # If it's a video file, try to loop back to beginning
            if self.is_video_file:
                self.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = self.camera.read()
                if not success:
                    return self.get_dummy_frame()
            else:
                return self.get_dummy_frame()
            
        # Detect AprilTags
        detected_tags = self.detector.detect_tags(frame)
        
        # Draw pose estimation on frame
        for tag in detected_tags:
            frame = self.detector.draw_pose(frame, tag)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return self.get_dummy_frame()
            
        # Convert to base64 for web transmission
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return frame_base64, detected_tags

# Global camera instance - initialized when first accessed
camera = None

def get_camera():
    """Get or initialize the global camera instance"""
    global camera
    if camera is None:
        camera = VideoCamera(0)  # Default to camera 0
    return camera

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/tags')
def get_tags():
    """API endpoint to get current tag detection data"""
    cam = get_camera()
    frame_data, tags = cam.get_frame()
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
            }
        }
        serializable_tags.append(serializable_tag)
    
    return jsonify({'tags': serializable_tags})

def generate_frames():
    """Generator function for video streaming"""
    cam = get_camera()
    while True:
        frame_data, tags = cam.get_frame()
        if frame_data is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + base64.b64decode(frame_data) + b'\r\n')
        eventlet.sleep(0.03)  # ~30 FPS

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
    frame_data, tags = cam.get_frame()
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
                }
            }
            serializable_tags.append(serializable_tag)
        
        emit('frame_data', {
            'image': frame_data,
            'tags': serializable_tags
        })

if __name__ == '__main__':
    import argparse
    
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