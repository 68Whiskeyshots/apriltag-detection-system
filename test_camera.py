import cv2

def test_camera_access():
    """Test camera access and capabilities"""
    print("Testing camera access...")
    
    for idx in range(5):
        print(f"\nTrying camera index {idx}:")
        cap = cv2.VideoCapture(idx)
        
        if cap.isOpened():
            print(f"  ✓ Camera {idx} opened successfully")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret and frame is not None:
                height, width = frame.shape[:2]
                print(f"  ✓ Successfully read frame: {width}x{height}")
                
                # Try to set resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"  ✓ Resolution set to: {actual_width}x{actual_height}")
                
                cap.release()
                print(f"  ✓ Camera {idx} is working properly!")
                return idx
            else:
                print(f"  ✗ Could not read frame from camera {idx}")
                cap.release()
        else:
            print(f"  ✗ Could not open camera {idx}")
            cap.release()
    
    print("\n❌ No working cameras found")
    return None

if __name__ == "__main__":
    working_camera = test_camera_access()
    if working_camera is not None:
        print(f"\n✅ Use camera index {working_camera} in your application")
    else:
        print("\n💡 Suggestions:")
        print("  1. Connect a USB webcam")
        print("  2. Use a video file instead")
        print("  3. Check camera permissions")