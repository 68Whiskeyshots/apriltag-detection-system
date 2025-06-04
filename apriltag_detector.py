import cv2
import apriltag
import numpy as np
import math

class AprilTagDetector:
    def __init__(self, camera_matrix=None, dist_coeffs=None, tag_size=0.05, families=["tag36h11", "tag25h9"]):
        """
        Initialize AprilTag detector with camera calibration parameters
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            tag_size: Physical size of AprilTag in meters (default 5cm)
            families: List of tag families to detect (default: ["tag36h11", "tag25h9"])
        """
        # Convert list to space-separated string for apriltag library
        families_str = " ".join(families) if isinstance(families, list) else families
        self.detector = apriltag.Detector(apriltag.DetectorOptions(families=families_str))
        self.tag_size = tag_size
        self.families = families
        
        # Default camera matrix for webcam (adjust based on your camera)
        if camera_matrix is None:
            self.camera_matrix = np.array([
                [800, 0, 320],
                [0, 800, 240],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            self.camera_matrix = camera_matrix
            
        # Default distortion coefficients (assuming minimal distortion)
        if dist_coeffs is None:
            self.dist_coeffs = np.zeros((4, 1))
        else:
            self.dist_coeffs = dist_coeffs
    
    def detect_tags(self, image):
        """
        Detect AprilTags in the given image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detected tags with pose information
        """
        # Convert to grayscale for AprilTag detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect AprilTags
        tags = self.detector.detect(gray)
        
        detected_tags = []
        
        for tag in tags:
            # Get corner coordinates
            corners = tag.corners.reshape(4, 2)
            
            # Define 3D object points (tag corners in tag coordinate system)
            object_points = np.array([
                [-self.tag_size/2, -self.tag_size/2, 0],
                [ self.tag_size/2, -self.tag_size/2, 0],
                [ self.tag_size/2,  self.tag_size/2, 0],
                [-self.tag_size/2,  self.tag_size/2, 0]
            ], dtype=np.float32)
            
            # Solve PnP to get pose
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                corners,
                self.camera_matrix,
                self.dist_coeffs
            )
            
            if success:
                # Convert rotation vector to rotation matrix
                rmat, _ = cv2.Rodrigues(rvec)
                
                # Calculate Euler angles from rotation matrix
                euler_angles = self.rotation_matrix_to_euler_angles(rmat)
                
                detected_tag = {
                    'id': tag.tag_id,
                    'family': tag.tag_family.decode('utf-8') if hasattr(tag.tag_family, 'decode') else str(tag.tag_family),
                    'corners': corners,
                    'center': tag.center,
                    'pose': {
                        'translation': tvec.flatten(),
                        'rotation_vector': rvec.flatten(),
                        'rotation_matrix': rmat,
                        'euler_angles': euler_angles
                    },
                    'distance': np.linalg.norm(tvec)
                }
                
                detected_tags.append(detected_tag)
        
        return detected_tags
    
    def rotation_matrix_to_euler_angles(self, R):
        """
        Convert rotation matrix to Euler angles (roll, pitch, yaw)
        """
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(R[2,1], R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        
        return np.array([x, y, z]) * 180 / math.pi  # Convert to degrees
    
    def draw_pose(self, image, tag_data):
        """
        Draw pose visualization on image
        
        Args:
            image: Input image
            tag_data: Detected tag data with pose information
        """
        corners = tag_data['corners']
        center = tag_data['center']
        pose = tag_data['pose']
        
        # Draw tag outline
        cv2.polylines(image, [corners.astype(int)], True, (0, 255, 0), 2)
        
        # Draw tag ID and family
        cv2.putText(image, f"ID: {tag_data['id']} ({tag_data['family']})", 
                   (int(center[0]) - 50, int(center[1]) - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw coordinate axes
        axis_length = self.tag_size
        axis_points = np.array([
            [0, 0, 0],  # Origin
            [axis_length, 0, 0],  # X-axis (red)
            [0, axis_length, 0],  # Y-axis (green)
            [0, 0, -axis_length]  # Z-axis (blue)
        ], dtype=np.float32)
        
        # Project axis points to image
        image_points, _ = cv2.projectPoints(
            axis_points,
            pose['rotation_vector'],
            pose['translation'],
            self.camera_matrix,
            self.dist_coeffs
        )
        
        image_points = image_points.reshape(-1, 2).astype(int)
        
        # Draw axes
        origin = tuple(image_points[0])
        x_end = tuple(image_points[1])
        y_end = tuple(image_points[2])
        z_end = tuple(image_points[3])
        
        cv2.arrowedLine(image, origin, x_end, (0, 0, 255), 3)  # X-axis (red)
        cv2.arrowedLine(image, origin, y_end, (0, 255, 0), 3)  # Y-axis (green)
        cv2.arrowedLine(image, origin, z_end, (255, 0, 0), 3)  # Z-axis (blue)
        
        # Draw pose information
        distance = tag_data['distance']
        euler = pose['euler_angles']
        
        info_text = [
            f"Distance: {distance:.3f}m",
            f"Roll: {euler[0]:.1f}°",
            f"Pitch: {euler[1]:.1f}°",
            f"Yaw: {euler[2]:.1f}°"
        ]
        
        y_offset = int(center[1]) + 30
        for i, text in enumerate(info_text):
            cv2.putText(image, text,
                       (int(center[0]) - 60, y_offset + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return image