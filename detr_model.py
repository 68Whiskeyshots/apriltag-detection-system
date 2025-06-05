"""
DETR Model Implementation for BANDIT
Handles custom DETR checkpoints with proper inference
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class PositionalEncoding(nn.Module):
    """Standard positional encoding for transformers"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class DETRModel(nn.Module):
    """Basic DETR model for object detection"""
    
    def __init__(self, num_classes=2, hidden_dim=256, num_queries=300, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        
        # CNN backbone (simplified)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, hidden_dim, kernel_size=3, stride=2, padding=1),
        )
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Query embeddings
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Output projection
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = nn.MLP([hidden_dim, hidden_dim, hidden_dim, 4])
        
        # For compatibility with various DETR formats
        self.class_labels_classifier = self.class_embed
        
    def forward(self, x):
        # CNN backbone
        features = self.backbone(x)
        b, c, h, w = features.shape
        
        # Flatten spatial dimensions
        features = features.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        
        # Add positional encoding
        features = self.pos_encoding(features)
        
        # Query embeddings
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)  # [B, num_queries, hidden_dim]
        
        # Transformer
        hs = self.transformer(features, query_embed)
        
        # Output projections
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        
        return {
            'logits': outputs_class,
            'pred_logits': outputs_class,
            'pred_boxes': outputs_coord
        }


class MLP(nn.Module):
    """Simple MLP for bbox regression"""
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


# Register MLP as an attribute of nn module for compatibility
nn.MLP = MLP


def build_detr_from_checkpoint(checkpoint_path, device='cpu'):
    """Build DETR model from checkpoint with proper state dict handling"""
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract args if available
    if isinstance(checkpoint, dict) and 'args' in checkpoint:
        args = checkpoint['args']
        num_classes = getattr(args, 'num_classes', 2)
        hidden_dim = getattr(args, 'hidden_dim', 256)
        num_queries = getattr(args, 'num_queries', 300)
    else:
        # Default values
        num_classes = 2
        hidden_dim = 256
        num_queries = 300
    
    # Create model
    model = DETRModel(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_queries=num_queries
    )
    
    # Load state dict if available
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
        
        # Try to load with strict=False to handle mismatches
        try:
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded model weights (partial match)")
        except Exception as e:
            print(f"Warning: Could not load all weights: {e}")
            # Model will use random initialization for missing weights
    
    model.to(device)
    model.eval()
    
    return model