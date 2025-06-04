import cv2
import numpy as np
from apriltag_detector import AprilTagDetector
import math

def create_demo_video():
    """Create a demo video with moving AprilTags for testing"""
    
    # Video properties
    width, height = 640, 480
    fps = 30
    duration_seconds = 10
    total_frames = fps * duration_seconds
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('demo_apriltag_video.mp4', fourcc, fps, (width, height))
    
    print(f"Creating demo video with {total_frames} frames...")
    
    # Load a sample AprilTag image (you can download from april.eecs.umich.edu)
    # For now, we'll create a simple synthetic tag
    tag_size = 100
    
    for frame_num in range(total_frames):
        # Create black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Calculate moving positions for multiple tags
        t = frame_num / total_frames * 2 * math.pi
        
        # Tag 1 - moving in circle
        center_x1 = int(width/2 + 150 * math.cos(t))
        center_y1 = int(height/2 + 100 * math.sin(t))
        
        # Tag 2 - moving linearly
        center_x2 = int(50 + (width - 100) * (frame_num / total_frames))
        center_y2 = int(height - 150)
        
        # Draw white squares as placeholder tags
        # Tag 1
        cv2.rectangle(frame, 
                     (center_x1 - tag_size//2, center_y1 - tag_size//2),
                     (center_x1 + tag_size//2, center_y1 + tag_size//2),
                     (255, 255, 255), -1)
        
        # Add black border and pattern to make it look like an AprilTag
        border = 10
        cv2.rectangle(frame,
                     (center_x1 - tag_size//2 + border, center_y1 - tag_size//2 + border),
                     (center_x1 + tag_size//2 - border, center_y1 + tag_size//2 - border),
                     (0, 0, 0), -1)
        
        # Add simple pattern
        pattern_size = tag_size - 2 * border
        quarter = pattern_size // 4
        
        # Create a simple 4x4 pattern for tag ID 0
        pattern = [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ]
        
        for i in range(4):
            for j in range(4):
                if pattern[i][j]:
                    cv2.rectangle(frame,
                                 (center_x1 - tag_size//2 + border + j * quarter,
                                  center_y1 - tag_size//2 + border + i * quarter),
                                 (center_x1 - tag_size//2 + border + (j+1) * quarter,
                                  center_y1 - tag_size//2 + border + (i+1) * quarter),
                                 (255, 255, 255), -1)
        
        # Tag 2 - similar but smaller
        tag_size2 = 80
        cv2.rectangle(frame,
                     (center_x2 - tag_size2//2, center_y2 - tag_size2//2),
                     (center_x2 + tag_size2//2, center_y2 + tag_size2//2),
                     (255, 255, 255), -1)
        
        # Add frame number for reference
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        
        if frame_num % 30 == 0:
            print(f"Progress: {frame_num}/{total_frames} frames")
    
    out.release()
    print("Demo video created: demo_apriltag_video.mp4")
    print("You can now run: python app.py --source demo_apriltag_video.mp4")

if __name__ == "__main__":
    create_demo_video()