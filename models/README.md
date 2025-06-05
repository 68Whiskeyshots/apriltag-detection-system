# RF-DETR Model Setup for BANDIT

## Overview

This directory contains the RF-DETR object detection model configuration and weights for the BANDIT system. The system supports custom trained RT-DETR models with configurable class names and detection parameters.

## Required Files

### 1. Model Weights
- **File**: `rf_detr_custom.pth` ⚠️ **NOT INCLUDED - You must provide this file**
- **Description**: Your custom trained RF-DETR model weights
- **Format**: PyTorch state dict (.pth)
- **Size**: Typically 100-500MB depending on model architecture
- **Note**: This file is not included in the repository due to size constraints. You must place your trained model file here before running the system.

### 2. Configuration File
- **File**: `rf_detr_config.json` ✅ (Provided)
- **Description**: Model configuration including class names and parameters
- **Format**: JSON

## Quick Setup

1. **Place your model weights**:
   ```bash
   # Copy your trained model to this directory
   cp /path/to/your/model.pth models/rf_detr_custom.pth
   ```

2. **Update configuration** (if needed):
   ```bash
   # Edit the config file to match your model's classes
   nano models/rf_detr_config.json
   ```

3. **Restart BANDIT**:
   ```bash
   python app.py
   ```

## Model Requirements

### Supported Architectures
- **RT-DETR** (Real-Time Detection Transformer)
- **DETR variants** with similar output format
- **Custom DETR implementations**

### Input Specifications
- **Input Size**: 640×640 (configurable)
- **Format**: RGB images
- **Normalization**: ImageNet standard (configurable)
- **Batch Size**: 1 (real-time inference)

### Output Format
Your model should output:
```python
{
    'logits': torch.Tensor,      # [batch, num_queries, num_classes]
    'pred_boxes': torch.Tensor   # [batch, num_queries, 4]
}
```

## Configuration File Format

```json
{
  "model_name": "Your Model Name",
  "class_names": [
    "class1",
    "class2",
    "..."
  ],
  "confidence_threshold": 0.5,
  "nms_threshold": 0.4,
  "input_size": [640, 640],
  "model_info": {
    "num_classes": 80,
    "num_queries": 300
  }
}
```

## Customizing Classes

To use your own object classes:

1. **Edit `rf_detr_config.json`**:
   ```json
   {
     "class_names": [
       "your_class_1",
       "your_class_2",
       "your_class_3"
     ],
     "model_info": {
       "num_classes": 3
     }
   }
   ```

2. **Ensure model compatibility**: Your .pth file should have the same number of output classes.

## Model Training

If you need to train a custom RF-DETR model:

### Recommended Resources
- **Hugging Face Transformers**: RT-DETR implementation
- **Official RT-DETR**: Original Baidu research code
- **Custom datasets**: COCO format annotation support

### Training Example
```python
from transformers import RTDetrForObjectDetection, RTDetrConfig

# Create configuration
config = RTDetrConfig(
    num_labels=your_num_classes,
    num_queries=300
)

# Create and train model
model = RTDetrForObjectDetection(config)
# ... training code ...

# Save for BANDIT
torch.save(model.state_dict(), 'models/rf_detr_custom.pth')
```

## Performance Optimization

### Hardware Recommendations
- **GPU**: NVIDIA RTX 3060 or better
- **VRAM**: 6GB+ for 640×640 input
- **CPU**: 8+ cores for CPU-only inference
- **RAM**: 8GB+ system memory

### Inference Speed
- **GPU (RTX 3080)**: ~50-100 FPS
- **GPU (RTX 3060)**: ~30-50 FPS  
- **CPU (Intel i7)**: ~5-15 FPS
- **CPU (ARM)**: ~2-8 FPS

### Optimization Tips
1. **Use FP16**: Reduce memory and increase speed
2. **Optimize input size**: Smaller inputs = faster inference
3. **Batch processing**: Process multiple frames together
4. **Model pruning**: Remove unnecessary parameters

## Troubleshooting

### Common Issues

#### 1. Model Not Loading
```
Error: Weights file not found
```
**Solution**: Ensure `rf_detr_custom.pth` exists in `models/` directory

#### 2. Class Mismatch
```
Error: Size mismatch for classifier
```
**Solution**: Verify `num_classes` in config matches your model

#### 3. Memory Issues
```
Error: CUDA out of memory
```
**Solutions**:
- Reduce input size in config
- Use CPU inference
- Close other GPU applications

#### 4. Slow Inference
**Solutions**:
- Enable GPU acceleration
- Reduce confidence threshold
- Optimize NMS threshold
- Use smaller input resolution

### Debug Mode
Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run BANDIT with debug info
python app.py --debug
```

## Integration with BANDIT

### API Endpoints
- **Status**: `GET /api/object_detection/status`
- **Config**: `POST /api/object_detection/config`
- **Toggle**: WebSocket `toggle_object_detection`
- **Inference**: WebSocket `start_ml_inference`

### Real-time Control
1. **Enable Detection**: Toggle in web interface
2. **Start Inference**: Click "Start Inference" button
3. **Adjust Parameters**: Use confidence/NMS sliders
4. **Monitor Performance**: Check FPS and detection counts

### Example Usage
```python
import requests

# Check if model is available
status = requests.get('http://localhost:5000/api/object_detection/status')
print(f"Model available: {status.json()['available']}")

# Update detection thresholds
config = requests.post('http://localhost:5000/api/object_detection/config', 
    json={
        'confidence_threshold': 0.7,
        'nms_threshold': 0.3
    })
```

## Support

For issues with:
- **Model training**: Consult Hugging Face Transformers documentation
- **BANDIT integration**: Check `tracker.md` for troubleshooting
- **Performance optimization**: See TECHNICAL_DOCUMENTATION.md

## License

Model weights are subject to their original training dataset licenses (e.g., COCO dataset license for COCO-trained models).

## Version History

- **v1.0.0**: Initial RF-DETR integration with BANDIT
- **v1.1.0**: Added custom class support
- **v1.2.0**: Performance optimizations and better error handling

---

*Developed by Dark Matter Labs for the BANDIT detection system*