#!/usr/bin/env python3
"""
Download Example RF-DETR Model for BANDIT

This script downloads a pre-trained RT-DETR model that can be used
as an example with the BANDIT system. Useful for testing the integration
before you have your own custom weights.

Usage:
    python models/download_example_model.py
    python models/download_example_model.py --model rtdetr_r50vd
"""

import os
import sys
import argparse
import json
from pathlib import Path

def download_huggingface_model(model_name="PekingU/rtdetr_r50vd_coco_o365"):
    """Download RT-DETR model from Hugging Face"""
    try:
        from transformers import RTDetrForObjectDetection
        import torch
        
        print(f"🔄 Downloading {model_name}...")
        print("   This may take several minutes...")
        
        # Download the model
        model = RTDetrForObjectDetection.from_pretrained(model_name)
        
        # Save the state dict
        weights_path = "models/rf_detr_custom.pth"
        torch.save(model.state_dict(), weights_path)
        
        print(f"✅ Model downloaded and saved to: {weights_path}")
        
        # Get model info
        config = model.config
        num_classes = config.num_labels
        
        # Create compatible config
        model_config = {
            "model_name": f"Example RT-DETR ({model_name})",
            "description": "Pre-trained RT-DETR model from Hugging Face",
            "version": "1.0.0",
            "input_size": [640, 640],
            "confidence_threshold": 0.5,
            "nms_threshold": 0.4,
            "max_detections": 100,
            "class_names": get_coco_classes()[:num_classes],
            "model_info": {
                "architecture": "RT-DETR",
                "backbone": "ResNet-50",
                "num_queries": getattr(config, 'num_queries', 300),
                "num_classes": num_classes,
                "training_dataset": "COCO + Objects365",
                "source": "Hugging Face",
                "model_id": model_name
            }
        }
        
        # Save config
        config_path = "models/rf_detr_config.json"
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        print(f"✅ Configuration saved to: {config_path}")
        return True
        
    except ImportError:
        print("❌ Error: transformers library not installed")
        print("   Install with: pip install transformers")
        return False
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

def download_torchvision_model():
    """Download DETR model from torchvision as fallback"""
    try:
        import torch
        import torchvision.models as models
        
        print("🔄 Downloading torchvision DETR model...")
        
        # Download DETR model
        model = models.detection.detr_resnet50(pretrained=True)
        
        # Save the state dict
        weights_path = "models/rf_detr_custom.pth"
        torch.save(model.state_dict(), weights_path)
        
        print(f"✅ Model downloaded and saved to: {weights_path}")
        
        # Create compatible config
        model_config = {
            "model_name": "Example DETR (TorchVision)",
            "description": "Pre-trained DETR model from TorchVision",
            "version": "1.0.0",
            "input_size": [800, 800],
            "confidence_threshold": 0.7,
            "nms_threshold": 0.5,
            "max_detections": 100,
            "class_names": get_coco_classes(),
            "model_info": {
                "architecture": "DETR",
                "backbone": "ResNet-50",
                "num_queries": 100,
                "num_classes": 91,
                "training_dataset": "COCO 2017",
                "source": "TorchVision"
            }
        }
        
        # Save config
        config_path = "models/rf_detr_config.json"
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        print(f"✅ Configuration saved to: {config_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading torchvision model: {e}")
        return False

def get_coco_classes():
    """Get COCO dataset class names"""
    return [
        "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
        "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

def main():
    parser = argparse.ArgumentParser(description='Download example RF-DETR model for BANDIT')
    parser.add_argument('--model', choices=['rtdetr', 'detr', 'auto'], default='auto',
                       help='Model type to download')
    parser.add_argument('--force', action='store_true',
                       help='Overwrite existing model files')
    
    args = parser.parse_args()
    
    print("🚀 BANDIT Example Model Downloader")
    print("=" * 50)
    
    # Check if model already exists
    weights_path = "models/rf_detr_custom.pth"
    config_path = "models/rf_detr_config.json"
    
    if os.path.exists(weights_path) and not args.force:
        print(f"⚠️  Model weights already exist: {weights_path}")
        print("   Use --force to overwrite")
        return True
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    success = False
    
    if args.model == 'rtdetr' or args.model == 'auto':
        print("\n🔄 Attempting to download RT-DETR from Hugging Face...")
        success = download_huggingface_model()
    
    if not success and (args.model == 'detr' or args.model == 'auto'):
        print("\n🔄 Falling back to TorchVision DETR...")
        success = download_torchvision_model()
    
    if success:
        print("\n" + "=" * 50)
        print("✅ Example model downloaded successfully!")
        print("\nNext steps:")
        print("1. Start BANDIT: python app.py")
        print("2. Open web interface: http://localhost:5000")
        print("3. Go to Object Detection panel")
        print("4. Enable object detection")
        print("5. Click 'Start Inference' to test")
        print("\nNote: This is an example model for testing.")
        print("Replace with your custom trained weights for production use.")
    else:
        print("\n❌ Failed to download example model")
        print("Please check your internet connection and dependencies.")
    
    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)