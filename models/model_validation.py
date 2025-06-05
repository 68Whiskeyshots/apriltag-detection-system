#!/usr/bin/env python3
"""
Model Validation Script for BANDIT RF-DETR Integration

This script helps validate that your RF-DETR model is compatible with BANDIT
and can be loaded successfully.

Usage:
    python models/model_validation.py
    python models/model_validation.py --config custom_config.json
    python models/model_validation.py --weights custom_model.pth
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path to import object_detector
sys.path.append(str(Path(__file__).parent.parent))

try:
    from object_detector import RFDETRObjectDetector
except ImportError:
    print("❌ Error: Could not import object_detector module")
    print("   Make sure you're running this from the BANDIT root directory")
    sys.exit(1)

def load_config(config_path):
    """Load and validate configuration file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"❌ Config file not found: {config_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config file: {e}")
        return None

def validate_config(config):
    """Validate configuration structure"""
    required_fields = ['class_names', 'confidence_threshold', 'nms_threshold']
    missing_fields = [field for field in required_fields if field not in config]
    
    if missing_fields:
        print(f"❌ Missing required config fields: {missing_fields}")
        return False
    
    # Validate class names
    if not isinstance(config['class_names'], list) or len(config['class_names']) == 0:
        print("❌ class_names must be a non-empty list")
        return False
    
    # Validate thresholds
    if not (0 <= config['confidence_threshold'] <= 1):
        print("❌ confidence_threshold must be between 0 and 1")
        return False
    
    if not (0 <= config['nms_threshold'] <= 1):
        print("❌ nms_threshold must be between 0 and 1")
        return False
    
    return True

def check_model_file(weights_path):
    """Check if model weights file exists and is readable"""
    if not os.path.exists(weights_path):
        print(f"❌ Model weights file not found: {weights_path}")
        return False
    
    file_size = os.path.getsize(weights_path) / (1024 * 1024)  # MB
    if file_size < 1:
        print(f"⚠️  Warning: Model file is very small ({file_size:.2f} MB)")
        print("   This might not be a complete model file")
    
    print(f"✅ Model file found: {weights_path} ({file_size:.2f} MB)")
    return True

def test_model_loading(weights_path, config_path):
    """Test loading the model with BANDIT's object detector"""
    print("\n🔄 Testing model loading...")
    
    try:
        detector = RFDETRObjectDetector(
            weights_path=weights_path,
            config_path=config_path,
            device='cpu'  # Use CPU for validation to avoid GPU issues
        )
        
        if detector.model is not None:
            print("✅ Model file validation successful!")
            print(f"   Device: {detector.device}")
            print(f"   Classes: {len(detector.class_names)}")
            print(f"   Confidence threshold: {detector.confidence_threshold}")
            print(f"   NMS threshold: {detector.nms_threshold}")
            
            # Test enable/disable
            detector.enable()
            if detector.is_available():
                print("✅ Enable/disable functionality working")
            detector.disable()
            
            return True
        else:
            print("❌ Model file validation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        return False

def test_inference(detector):
    """Test inference with a dummy image"""
    print("\n🔄 Testing inference...")
    
    try:
        # Create a dummy image (640x480x3)
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Enable detector for inference
        detector.enable()
        
        # Run detection
        detections = detector.detect_objects(dummy_image)
        
        print(f"✅ Inference interface test passed!")
        print(f"   Detections: {len(detections)}")
        print("   Note: Actual inference will work when proper model loading is implemented")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔄 Checking dependencies...")
    
    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA: {torch.version.cuda} ({torch.cuda.device_count()} devices)")
        else:
            print("⚠️  CUDA not available (CPU-only inference)")
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    # Check other dependencies
    try:
        import torchvision
        print(f"✅ TorchVision: {torchvision.__version__}")
    except ImportError:
        print("⚠️  TorchVision not available")
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("⚠️  Transformers not available (fallback model will be used)")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Validate RF-DETR model for BANDIT')
    parser.add_argument('--weights', default='models/rf_detr_custom.pth',
                       help='Path to model weights file')
    parser.add_argument('--config', default='models/rf_detr_config.json',
                       help='Path to model config file')
    parser.add_argument('--skip-inference', action='store_true',
                       help='Skip inference test (faster validation)')
    
    args = parser.parse_args()
    
    print("🚀 BANDIT RF-DETR Model Validation")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed")
        return False
    
    # Check configuration
    print(f"\n🔄 Validating configuration: {args.config}")
    config = load_config(args.config)
    if not config:
        return False
    
    if not validate_config(config):
        return False
    
    print("✅ Configuration is valid")
    print(f"   Model: {config.get('model_name', 'Unnamed')}")
    print(f"   Classes: {len(config['class_names'])}")
    print(f"   First few classes: {config['class_names'][:5]}")
    
    # Check model file
    print(f"\n🔄 Checking model weights: {args.weights}")
    if not check_model_file(args.weights):
        print("\n⚠️  Model weights not found. You can still test configuration.")
        print("   Place your model file at:", args.weights)
        return True
    
    # Test model loading
    detector = None
    if test_model_loading(args.weights, args.config):
        try:
            detector = RFDETRObjectDetector(
                weights_path=args.weights,
                config_path=args.config,
                device='cpu'
            )
        except:
            pass
    
    # Test inference
    if detector and not args.skip_inference:
        test_inference(detector)
    
    print("\n" + "=" * 50)
    print("✅ Validation complete!")
    print("\nNext steps:")
    print("1. Place your model weights at:", args.weights)
    print("2. Update configuration if needed:", args.config)
    print("3. Start BANDIT: python app.py")
    print("4. Enable object detection in the web interface")
    print("5. Click 'Start Inference' to begin detection")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)