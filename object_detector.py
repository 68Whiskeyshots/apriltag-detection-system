import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import os

class RFDETRObjectDetector:
    def __init__(self, 
                 weights_path: str = "models/rf_detr_custom.pth",
                 config_path: str = "models/rf_detr_config.json",
                 device: str = "auto",
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4):
        """
        Initialize RF-DETR object detector with custom weights
        
        Args:
            weights_path: Path to custom .pth weights file
            config_path: Path to model configuration JSON
            device: Device to run inference on ('cpu', 'cuda', 'auto')
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
        """
        self.weights_path = weights_path
        self.config_path = config_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.model = None
        self.class_names = []
        self.device = self._setup_device(device)
        self.transform = self._setup_transforms()
        
        # Load model configuration
        self._load_config()
        
        # Initialize model
        self._load_model()
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _setup_transforms(self) -> transforms.Compose:
        """Setup image preprocessing transforms"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((640, 640)),  # Standard DETR input size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_config(self):
        """Load model configuration and class names"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.class_names = config.get('class_names', [])
                    self.num_classes = len(self.class_names)
            else:
                print(f"Warning: Config file {self.config_path} not found. Using default COCO classes.")
                self._load_default_classes()
        except Exception as e:
            print(f"Error loading config: {e}. Using default COCO classes.")
            self._load_default_classes()
    
    def _load_default_classes(self):
        """Load default COCO class names"""
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        self.num_classes = len(self.class_names)
    
    def _load_model(self):
        """Load RF-DETR model with custom weights"""
        try:
            if not os.path.exists(self.weights_path):
                raise FileNotFoundError(f"Weights file not found: {self.weights_path}")
            
            # Load the model state dict with weights_only=False for compatibility
            # Note: Only use this with trusted model files
            checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=False)
            
            # Create RF-DETR model architecture
            self.model = self._create_rf_detr_model()
            
            # Note: For custom DETR using build_detr_from_checkpoint,
            # weights are already loaded inside that function
            # So we only need to load weights for other model types
            if hasattr(self.model, 'is_custom_detr') and self.model.is_custom_detr:
                # Custom model - weights need to be loaded
                if 'model' in checkpoint:
                    try:
                        self.model.load_state_dict(checkpoint['model'], strict=False)
                    except Exception as e:
                        print(f"Warning: Could not load full model state dict: {e}")
            elif not hasattr(self.model, '__class__') or 'DETRModel' not in str(self.model.__class__):
                # Only load if it's not our custom DETR (which loads in build function)
                if 'model' in checkpoint:
                    self.model.load_state_dict(checkpoint['model'], strict=False)
                else:
                    self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"RF-DETR model loaded successfully from {self.weights_path}")
            print(f"Running on device: {self.device}")
            print(f"Number of classes: {self.num_classes}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Object detection will be disabled.")
            self.model = None
    
    def _create_rf_detr_model(self):
        """Create RF-DETR model architecture"""
        # Check if this is a custom DETR model with transformer architecture
        try:
            checkpoint = torch.load(self.weights_path, map_location='cpu', weights_only=False)
            
            # Check if it's a custom DETR format with 'model' and 'args'
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                # This is a custom DETR model, create matching architecture
                model_state = checkpoint['model']
                
                # Check for transformer.decoder pattern (custom DETR)
                if any('transformer.decoder' in k for k in model_state.keys()):
                    print("Detected custom DETR architecture")
                    # Use the proper DETR implementation
                    from detr_model import build_detr_from_checkpoint
                    return build_detr_from_checkpoint(self.weights_path, self.device)
            
            # Fall back to RT-DETR from transformers
            from transformers import RTDetrForObjectDetection, RTDetrConfig
            
            config = RTDetrConfig(
                num_labels=self.num_classes,
                num_queries=300,
            )
            
            model = RTDetrForObjectDetection(config)
            return model
            
        except ImportError:
            print("Warning: transformers library not available. Using DETR model.")
            from detr_model import build_detr_from_checkpoint
            return build_detr_from_checkpoint(self.weights_path, self.device)
        except Exception as e:
            print(f"Error creating model architecture: {e}")
            return self._create_simple_detr_model()
    
    def _create_custom_detr_model(self, args):
        """Create custom DETR model based on checkpoint args"""
        import torch.nn as nn
        
        class CustomDETR(nn.Module):
            def __init__(self, num_classes, hidden_dim=256, num_queries=300):
                super().__init__()
                self.num_classes = num_classes
                self.num_queries = num_queries
                
                # Create a minimal structure that matches the checkpoint
                self.query_embed = nn.Embedding(num_queries, hidden_dim)
                self.class_embed = nn.Linear(hidden_dim, num_classes)
                self.bbox_embed = nn.Linear(hidden_dim, 4)
                
                # Flag to indicate custom model
                self.is_custom_detr = True
                
            def forward(self, x):
                # Forward pass will be handled by the loaded model weights
                raise NotImplementedError("Forward pass should use loaded model weights")
        
        # Extract parameters from args if available
        if args:
            num_classes = getattr(args, 'num_classes', self.num_classes)
            hidden_dim = getattr(args, 'hidden_dim', 256)
            num_queries = getattr(args, 'num_queries', 300)
        else:
            num_classes = self.num_classes
            hidden_dim = 256
            num_queries = 300
            
        return CustomDETR(num_classes, hidden_dim, num_queries)
    
    def _create_simple_detr_model(self):
        """Create a simplified DETR-like model structure"""
        import torch.nn as nn
        
        class SimpleDETR(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                # Create model structure
                self.backbone = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False)
                self.backbone.class_embed = nn.Linear(256, num_classes)
                
            def forward(self, x):
                return self.backbone(x)
        
        return SimpleDETR(self.num_classes)
    
    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects in the given image
        
        Args:
            image: Input image in BGR format (OpenCV)
            
        Returns:
            List of detected objects with bounding boxes, classes, and confidence scores
        """
        if self.model is None:
            return []
        
        try:
            # Preprocess image
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
            
            # Post-process outputs
            detections = self._post_process_outputs(outputs, image.shape[:2])
            
            return detections
            
        except Exception as e:
            print(f"Error during object detection: {e}")
            return []
    
    def _post_process_outputs(self, outputs, original_shape: Tuple[int, int]) -> List[Dict]:
        """
        Post-process model outputs to extract detections
        
        Args:
            outputs: Model outputs
            original_shape: Original image shape (height, width)
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        try:
            # Handle different output formats
            if hasattr(outputs, 'logits') and hasattr(outputs, 'pred_boxes'):
                # Transformers DETR format
                logits = outputs.logits[0]  # Remove batch dimension
                boxes = outputs.pred_boxes[0]
            else:
                # Custom format - adapt based on your model
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('pred_logits'))
                    boxes = outputs.get('pred_boxes')
                else:
                    # Assume tuple/list output
                    logits, boxes = outputs[:2]
            
            if logits is None or boxes is None:
                return detections
            
            # Convert logits to probabilities
            probs = torch.softmax(logits, -1)
            
            # Get scores and labels
            scores, labels = probs.max(-1)
            
            # Filter by confidence threshold
            keep = scores > self.confidence_threshold
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]
            
            # Convert boxes to image coordinates
            h, w = original_shape
            boxes = boxes * torch.tensor([w, h, w, h], device=boxes.device)
            
            # Convert to detection format
            for i in range(len(scores)):
                if labels[i] < len(self.class_names):  # Valid class
                    x1, y1, x2, y2 = boxes[i].cpu().numpy()
                    
                    # Convert center-width-height to x1,y1,x2,y2 if needed
                    if x2 < x1:  # Likely center-width format
                        cx, cy, w_box, h_box = x1, y1, x2, y2
                        x1 = cx - w_box / 2
                        y1 = cy - h_box / 2
                        x2 = cx + w_box / 2
                        y2 = cy + h_box / 2
                    
                    detection = {
                        'class_id': int(labels[i].cpu()),
                        'class_name': self.class_names[labels[i]],
                        'confidence': float(scores[i].cpu()),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'center': [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                        'width': float(x2 - x1),
                        'height': float(y2 - y1)
                    }
                    detections.append(detection)
            
            # Apply NMS if multiple detections
            if len(detections) > 1:
                detections = self._apply_nms(detections)
            
        except Exception as e:
            print(f"Error in post-processing: {e}")
        
        return detections
    
    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """Apply Non-Maximum Suppression to remove duplicate detections"""
        if len(detections) <= 1:
            return detections
        
        try:
            import torchvision.ops as ops
            
            # Convert to tensors
            boxes = torch.tensor([det['bbox'] for det in detections])
            scores = torch.tensor([det['confidence'] for det in detections])
            
            # Apply NMS
            keep_indices = ops.nms(boxes, scores, self.nms_threshold)
            
            # Return filtered detections
            return [detections[i] for i in keep_indices.tolist()]
            
        except ImportError:
            # Fallback: simple overlap-based filtering
            return self._simple_nms(detections)
    
    def _simple_nms(self, detections: List[Dict]) -> List[Dict]:
        """Simple NMS implementation without torchvision"""
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        filtered = []
        for detection in detections:
            # Check overlap with already selected detections
            keep = True
            for selected in filtered:
                if self._calculate_iou(detection['bbox'], selected['bbox']) > self.nms_threshold:
                    keep = False
                    break
            
            if keep:
                filtered.append(detection)
        
        return filtered
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw detection results on image
        
        Args:
            image: Input image
            detections: List of detection results
            
        Returns:
            Image with drawn detections
        """
        result_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = [int(coord) for coord in detection['bbox']]
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Draw bounding box
            color = self._get_class_color(detection['class_id'])
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for label
            cv2.rectangle(result_image, 
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1),
                         color, -1)
            
            # Label text
            cv2.putText(result_image, label,
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 2)
        
        return result_image
    
    def _get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get a consistent color for each class"""
        # Generate color based on class ID
        np.random.seed(class_id)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        return color
    
    def is_available(self) -> bool:
        """Check if object detection is available"""
        return self.model is not None
    
    def get_class_names(self) -> List[str]:
        """Get list of class names"""
        return self.class_names.copy()
    
    def update_thresholds(self, confidence: float = None, nms: float = None):
        """Update detection thresholds"""
        if confidence is not None:
            self.confidence_threshold = confidence
        if nms is not None:
            self.nms_threshold = nms

# Helper function to create model directory and config
def setup_model_directory():
    """Create model directory and example configuration"""
    os.makedirs("models", exist_ok=True)
    
    # Create example configuration
    config = {
        "model_name": "RF-DETR Custom",
        "description": "Custom trained RF-DETR model",
        "class_names": [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench"
        ],
        "input_size": [640, 640],
        "confidence_threshold": 0.5,
        "nms_threshold": 0.4
    }
    
    config_path = "models/rf_detr_config.json"
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created example config at {config_path}")
    
    # Create README for model setup
    readme_path = "models/README.md"
    if not os.path.exists(readme_path):
        with open(readme_path, 'w') as f:
            f.write("""# RF-DETR Model Setup

## Model Files Required

1. **rf_detr_custom.pth** - Your custom trained RF-DETR weights
2. **rf_detr_config.json** - Model configuration file

## Configuration Format

```json
{
  "model_name": "RF-DETR Custom",
  "description": "Custom trained RF-DETR model",
  "class_names": ["class1", "class2", "..."],
  "input_size": [640, 640],
  "confidence_threshold": 0.5,
  "nms_threshold": 0.4
}
```

## Usage

Place your custom .pth weights file in this directory and update the configuration
with your specific class names and parameters.

The system will automatically detect and load your model when available.
""")
        print(f"Created model setup guide at {readme_path}")

if __name__ == "__main__":
    # Setup model directory
    setup_model_directory()
    
    # Test initialization
    detector = RFDETRObjectDetector()
    print(f"Object detector initialized. Available: {detector.is_available()}")
    print(f"Classes: {detector.get_class_names()[:10]}...")  # Show first 10 classes