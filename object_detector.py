"""
Simplified Object Detector for BANDIT
Only supports basic functionality with clear error handling
"""

import torch
import cv2
import numpy as np
from typing import List, Dict
import json
import os


class RFDETRObjectDetector:
    """Simplified object detector with clear boundaries and error handling"""
    
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
        
        # Load configuration
        self._load_config()
        
        # Try to load model
        self._load_model()
    
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
    
    def _load_model(self):
        """Load actual DETR model from checkpoint"""
        try:
            if not os.path.exists(self.weights_path):
                print(f"Model file not found: {self.weights_path}")
                print("Object detection will be disabled")
                self.model = None
                return
            
            file_size = os.path.getsize(self.weights_path) / (1024 * 1024)  # MB
            print(f"Loading model file: {file_size:.1f} MB")
            
            # Load the actual DETR model
            from detr_model import build_model_from_checkpoint
            self.model, self.model_args = build_model_from_checkpoint(
                self.weights_path, 
                device=self.device
            )
            
            print("DETR model loaded successfully")
            print(f"Model on device: {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
    
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
        Detect objects in image using DETR model
        
        Args:
            image: Input image in BGR format
            
        Returns:
            List of detection dictionaries
        """
        if not self.is_available():
            return []
        
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(processed_image)
            
            # Post-process outputs
            detections = self._post_process_outputs(outputs, image.shape[:2])
            
            return detections
            
        except Exception as e:
            print(f"Error during object detection: {e}")
            return []
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for DETR model"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Use the model's expected input size (from args)
        input_size = getattr(self.model_args, 'resolution', 616)
        
        # Resize image
        resized = cv2.resize(rgb_image, (input_size, input_size))
        
        # Convert to tensor and normalize
        tensor_image = torch.from_numpy(resized).float() / 255.0
        tensor_image = tensor_image.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        
        # Normalize using ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor_image = (tensor_image - mean) / std
        
        return tensor_image.to(self.device)
    
    def _post_process_outputs(self, outputs: dict, original_shape: tuple) -> List[Dict]:
        """Post-process DETR outputs to detection format"""
        logits = outputs['logits'][0]  # [num_queries, num_classes]
        boxes = outputs['pred_boxes'][0]  # [num_queries, 4] 
        
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Get confidence scores (max probability, excluding background class 0)
        if probs.shape[-1] > 1:
            # For multi-class, take max of non-background classes
            object_probs = probs[:, 1:]  # Exclude background
            confidences, pred_classes = torch.max(object_probs, dim=-1)
            pred_classes = pred_classes + 1  # Adjust for background class
        else:
            # Single class case
            confidences = probs[:, 0]
            pred_classes = torch.zeros_like(confidences).long()
        
        # Filter by confidence threshold
        keep = confidences > self.confidence_threshold
        
        if not keep.any():
            return []
        
        # Filter detections
        kept_boxes = boxes[keep]
        kept_confidences = confidences[keep]
        kept_classes = pred_classes[keep]
        
        # Convert normalized coordinates to pixel coordinates
        h, w = original_shape
        kept_boxes = kept_boxes * torch.tensor([w, h, w, h], device=kept_boxes.device)
        
        # Convert to detection format
        detections = []
        for i in range(len(kept_boxes)):
            # Convert center + size format to x1,y1,x2,y2
            cx, cy, box_w, box_h = kept_boxes[i]
            x1 = cx - box_w / 2
            y1 = cy - box_h / 2
            x2 = cx + box_w / 2
            y2 = cy + box_h / 2
            
            # Clamp to image bounds
            x1 = max(0, min(w, x1.item()))
            y1 = max(0, min(h, y1.item()))
            x2 = max(0, min(w, x2.item()))
            y2 = max(0, min(h, y2.item()))
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            class_id = kept_classes[i].item()
            confidence = kept_confidences[i].item()
            
            detection = {
                'class_id': class_id,
                'class_name': self.class_names[min(class_id, len(self.class_names) - 1)],
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2],
                'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                'width': x2 - x1,
                'height': y2 - y1
            }
            detections.append(detection)
        
        # Apply NMS
        if len(detections) > 1:
            detections = self._apply_nms(detections)
        
        return detections
    
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