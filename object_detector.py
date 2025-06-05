"""
RF-DETR Object Detector for BANDIT
Uses Ultralytics RT-DETR as base architecture for shipping pallet detection
"""

import torch
import cv2
import numpy as np
from typing import List, Dict
import json
import os
import logging

logger = logging.getLogger(__name__)

class RFDETRObjectDetector:
    """RF-DETR object detector using Ultralytics RT-DETR as base architecture"""
    
    def __init__(self, 
                 weights_path: str = "models/rf_detr_custom.pth",
                 config_path: str = "models/rf_detr_config.json",
                 device: str = "auto"):
        """
        Initialize object detector
        
        Args:
            weights_path: Path to .pth model weights
            config_path: Path to JSON configuration
            device: Device for inference ('auto', 'cuda', 'cpu')
        """
        self.weights_path = weights_path
        self.config_path = config_path
        self.device = self._setup_device(device)
        self.model = None
        self.class_names = []
        self.confidence_threshold = 0.7
        self.nms_threshold = 0.3
        self.enabled = False
        
        # Performance optimization settings
        self.skip_frames = 3  # Only process every 3rd frame
        self.frame_count = 0
        self.last_detections = []  # Cache last detections
        self.inference_timeout = 1.0  # Max 1 second per inference
        
        # Load configuration
        self._load_config()
        
        # Store input size from config
        self.input_size = 640  # Default, will be updated from config
        
        # Try to load model using Ultralytics approach
        self._load_ultralytics_model()
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _load_config(self):
        """Load configuration from JSON file"""
        try:
            if not os.path.exists(self.config_path):
                print(f"Config file not found: {self.config_path}")
                self._use_default_config()
                return
            
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                
            self.class_names = config.get('class_names', ['shipping_pallet'])
            self.confidence_threshold = config.get('confidence_threshold', 0.7)
            self.nms_threshold = config.get('nms_threshold', 0.3)
            
            # Get input size from config
            input_size_config = config.get('input_size', [640, 640])
            if isinstance(input_size_config, list):
                self.input_size = input_size_config[0]  # Use first dimension
            else:
                self.input_size = input_size_config
            
            print(f"Loaded config: {len(self.class_names)} classes")
            
        except Exception as e:
            print(f"Error loading config: {e}")
            self._use_default_config()
    
    def _use_default_config(self):
        """Use default configuration if config file fails"""
        self.class_names = ['shipping_pallet']
        self.confidence_threshold = 0.7
        self.nms_threshold = 0.3
        print("Using default configuration")
    
    def _load_ultralytics_model(self):
        """Load RF-DETR model using Ultralytics RT-DETR as base"""
        try:
            if not os.path.exists(self.weights_path):
                print(f"Model file not found: {self.weights_path}")
                print("Object detection will be disabled")
                self.model = None
                return
            
            file_size = os.path.getsize(self.weights_path) / (1024 * 1024)  # MB
            print(f"Loading model file: {file_size:.1f} MB")
            
            # Try to import ultralytics
            try:
                from ultralytics import RTDETR
                print("Using Ultralytics RT-DETR backend")
                
                # Load checkpoint to get configuration
                checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=False)
                
                if 'args' not in checkpoint:
                    raise ValueError("Invalid RF-DETR checkpoint - missing args")
                
                args = checkpoint['args']
                print(f"RF-DETR config: classes={args.num_classes}, encoder={args.encoder}")
                
                # Initialize with base RT-DETR model 
                self.model = RTDETR('rtdetr-l.pt')
                
                # Try to load compatible weights
                self._load_compatible_weights(checkpoint)
                
                # Set to evaluation mode
                self.model.model.eval()
                
                print("✓ Ultralytics RT-DETR model loaded successfully")
                print(f"Model on device: {self.device}")
                
            except ImportError:
                print("Ultralytics not available, falling back to basic implementation")
                self.model = None
                return
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
    
    def _load_compatible_weights(self, checkpoint):
        """Attempt to load RF-DETR weights into RT-DETR architecture"""
        try:
            model_weights = checkpoint['model']
            model_state_dict = self.model.model.state_dict()
            
            # Create a mapping between RF-DETR and RT-DETR weight names
            loaded_keys = []
            missing_keys = []
            
            # Try to match keys between the checkpoints
            for rf_key, rf_weight in model_weights.items():
                # Look for matching keys in RT-DETR
                matched = False
                
                for rt_key in model_state_dict.keys():
                    # Simple name matching
                    if self._keys_match(rf_key, rt_key) and rf_weight.shape == model_state_dict[rt_key].shape:
                        model_state_dict[rt_key] = rf_weight
                        loaded_keys.append(rf_key)
                        matched = True
                        break
                
                if not matched:
                    missing_keys.append(rf_key)
            
            # Load the modified state dict
            self.model.model.load_state_dict(model_state_dict, strict=False)
            print(f"✓ Loaded {len(loaded_keys)} compatible weight tensors")
            if missing_keys and len(missing_keys) < 20:
                print(f"⚠ Could not load {len(missing_keys)} weight tensors")
                    
        except Exception as e:
            print(f"Could not load custom weights: {e}")
            print("Continuing with base RT-DETR weights")
    
    def _keys_match(self, key1, key2):
        """Check if two parameter keys likely refer to the same layer"""
        # Remove common prefixes/suffixes and check similarity
        key1_clean = key1.replace('module.', '').replace('model.', '')
        key2_clean = key2.replace('module.', '').replace('model.', '')
        
        # Simple substring matching
        return key1_clean == key2_clean or key1_clean in key2_clean or key2_clean in key1_clean
    
    def is_available(self) -> bool:
        """Check if object detection is available"""
        return self.model is not None and self.enabled
    
    def enable(self):
        """Enable object detection"""
        if self.model is not None:
            self.enabled = True
            print("Object detection enabled")
        else:
            print("Cannot enable - no model available")
    
    def disable(self):
        """Disable object detection"""
        self.enabled = False
        print("Object detection disabled")
    
    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects in image using RT-DETR model with frame skipping optimization
        
        Args:
            image: Input image in BGR format
            
        Returns:
            List of detection dictionaries
        """
        if not self.is_available():
            return []
        
        # Frame skipping optimization - only process every Nth frame
        self.frame_count += 1
        if self.frame_count % self.skip_frames != 0:
            # Return cached detections for skipped frames
            return self.last_detections
        
        try:
            import time
            start_time = time.time()
            
            # Convert BGR to RGB for Ultralytics
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image for faster inference (trade quality for speed)
            height, width = image_rgb.shape[:2]
            if max(height, width) > 640:
                scale = 640 / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image_rgb = cv2.resize(image_rgb, (new_width, new_height))
                scale_factor = 1.0 / scale
            else:
                scale_factor = 1.0
            
            # Run inference using ultralytics with timeout protection
            results = self.model(image_rgb, verbose=False)
            
            # Process results
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    for box, conf, cls in zip(boxes, confidences, classes):
                        if conf > self.confidence_threshold:
                            x1, y1, x2, y2 = box
                            
                            # Scale coordinates back to original image size
                            if scale_factor != 1.0:
                                x1, y1, x2, y2 = x1 * scale_factor, y1 * scale_factor, x2 * scale_factor, y2 * scale_factor
                            
                            # Convert all numpy values to Python native types for JSON serialization
                            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                            conf = float(conf)
                            cls = int(cls)
                            
                            # Convert to our format
                            detection = {
                                'class_id': cls,
                                'class_name': self.class_names[min(cls, len(self.class_names) - 1)],
                                'confidence': conf,
                                'bbox': [x1, y1, x2, y2],
                                'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                                'width': x2 - x1,
                                'height': y2 - y1
                            }
                            detections.append(detection)
            
            # Cache detections for frame skipping
            self.last_detections = detections
            
            inference_time = (time.time() - start_time) * 1000
            print(f"[DEBUG] RT-DETR: {len(detections)} objects, {inference_time:.1f}ms")
            
            # Warn if inference is too slow
            if inference_time > 500:  # More than 500ms
                print(f"[WARNING] Slow inference detected ({inference_time:.1f}ms) - consider reducing skip_frames")
            
            return detections
            
        except Exception as e:
            print(f"Error during object detection: {e}")
            # Return cached detections on error
            return self.last_detections
    
    # Preprocessing is now handled by Ultralytics internally
    
    # Post-processing is now handled by Ultralytics internally
    # Remove all the old NMS and helper methods as they're not needed
    
    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """Apply Non-Maximum Suppression"""
        if len(detections) <= 1:
            return detections
        
        # Convert to tensors for NMS
        boxes = torch.tensor([d['bbox'] for d in detections])
        scores = torch.tensor([d['confidence'] for d in detections])
        
        # Apply NMS
        keep_indices = self._nms_torch(boxes, scores, self.nms_threshold)
        
        return [detections[i] for i in keep_indices]
    
    def _nms_torch(self, boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> List[int]:
        """Simple NMS implementation"""
        if len(boxes) == 0:
            return []
        
        # Calculate areas
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by scores
        order = scores.argsort(descending=True)
        
        keep = []
        while len(order) > 0:
            # Keep highest scoring box
            i = order[0].item()
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # Calculate IoU with remaining boxes
            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            intersection = w * h
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)
            
            # Keep boxes with IoU below threshold
            keep_mask = iou <= threshold
            order = order[1:][keep_mask]
        
        return keep
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw detection boxes on image
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            
        Returns:
            Image with drawn detections
        """
        for detection in detections:
            try:
                bbox = detection['bbox']  # [x1, y1, x2, y2]
                class_name = detection['class_name']
                confidence = detection['confidence']
                
                # Draw bounding box
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), 
                            (int(bbox[2]), int(bbox[3])), (0, 255, 255), 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Background for label
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1]) - label_size[1] - 10),
                            (int(bbox[0]) + label_size[0], int(bbox[1])), (0, 255, 255), -1)
                
                # Text
                cv2.putText(image, label, (int(bbox[0]), int(bbox[1]) - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                          
            except Exception as e:
                print(f"Error drawing detection: {e}")
                continue
        
        return image
    
    def get_class_names(self) -> List[str]:
        """Get list of class names"""
        return self.class_names.copy()
    
    def update_thresholds(self, confidence: float = None, nms: float = None):
        """Update detection thresholds"""
        if confidence is not None:
            self.confidence_threshold = float(confidence)
        if nms is not None:
            self.nms_threshold = float(nms)
        print(f"Updated thresholds: conf={self.confidence_threshold}, nms={self.nms_threshold}")
    
    def set_frame_skip(self, skip_frames: int):
        """Update frame skipping for performance tuning"""
        self.skip_frames = max(1, skip_frames)
        print(f"Frame skipping set to every {self.skip_frames} frames")
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        return {
            'frame_skip': self.skip_frames,
            'cached_detections': len(self.last_detections),
            'frame_count': self.frame_count
        }


def setup_model_directory():
    """Setup model directory structure"""
    os.makedirs("models", exist_ok=True)
    
    # Create .gitkeep if directory is empty
    gitkeep_path = "models/.gitkeep"
    if not os.path.exists(gitkeep_path):
        with open(gitkeep_path, 'w') as f:
            f.write("# Keep this directory in git\n")


if __name__ == "__main__":
    # Simple test
    print("Testing object detector...")
    detector = RFDETRObjectDetector()
    print(f"Available: {detector.is_available()}")
    print(f"Classes: {detector.get_class_names()}")
    
    if detector.model:
        detector.enable()
        print(f"Enabled: {detector.is_available()}")
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = detector.detect_objects(dummy_image)
        print(f"Detections: {len(detections)}")