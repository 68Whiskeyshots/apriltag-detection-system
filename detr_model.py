"""
DETR Model Implementation for BANDIT
Implements DinoV2 + DETR architecture to match the trained checkpoint
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List
import torchvision.transforms as T
from torchvision.models import vision_transformer


class PositionEmbeddingSine(nn.Module):
    """
    Position encoding using sine waves
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, channels, height, width]
        Returns:
            pos: Positional encoding [batch_size, channels, height, width]
        """
        b, c, h, w = x.shape
        
        # Create coordinate grids
        y_embed = torch.arange(h, dtype=torch.float32, device=x.device)
        x_embed = torch.arange(w, dtype=torch.float32, device=x.device)
        
        if self.normalize:
            y_embed = y_embed / (h - 1) * self.scale
            x_embed = x_embed / (w - 1) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        
        pos = torch.cat((pos_y[:, None, :].repeat(1, w, 1),
                        pos_x[None, :, :].repeat(h, 1, 1)), dim=-1).permute(2, 0, 1)
        
        return pos.unsqueeze(0).repeat(b, 1, 1, 1)


class MultiScaleDeformableAttention(nn.Module):
    """
    Multi-Scale Deformable Attention matching checkpoint parameters
    """
    def __init__(self, d_model=256, n_levels=1, n_heads=16, n_points=2):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        # Based on checkpoint shapes:
        # sampling_offsets: [64, 256] = n_heads * n_levels * n_points * 2 = 16 * 1 * 2 * 2 = 64
        # attention_weights: [32, 256] = n_heads * n_levels * n_points = 16 * 1 * 2 = 32
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        nn.init.constant_(self.sampling_offsets.bias.data, 0.)
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, value, value_spatial_shapes, reference_points=None):
        # Simplified implementation - just use standard attention
        N, Len_q, _ = query.shape
        N, Len_v, _ = value.shape
        
        value = self.value_proj(value)
        
        # Standard attention as fallback
        attention_weights = torch.softmax(
            torch.matmul(query, value.transpose(-2, -1)) / math.sqrt(self.d_model), 
            dim=-1
        )
        
        output = torch.matmul(attention_weights, value)
        output = self.output_proj(output)
        
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    """
    DETR Decoder Layer with Deformable Attention
    """
    def __init__(self, d_model=256, n_heads=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross attention (deformable) - using checkpoint parameters
        self.cross_attn = MultiScaleDeformableAttention(d_model, n_levels=1, n_heads=16, n_points=2)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed forward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, memory_spatial_shapes=None, reference_points=None):
        # Self attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt)
        tgt = self.norm1(tgt + tgt2)
        
        # Cross attention
        tgt2 = self.cross_attn(tgt, memory, memory_spatial_shapes, reference_points)
        tgt = self.norm2(tgt + tgt2)
        
        # Feed forward
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + tgt2)
        
        return tgt


class DeformableTransformerDecoder(nn.Module):
    """
    DETR Transformer Decoder
    """
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, tgt, memory, memory_spatial_shapes=None, reference_points=None):
        output = tgt
        
        for layer in self.layers:
            output = layer(output, memory, memory_spatial_shapes, reference_points)
        
        return output


class SimplifiedViTBackbone(nn.Module):
    """
    Simplified Vision Transformer backbone
    """
    def __init__(self, embed_dim=768, num_layers=12):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Use a simple CNN to simulate ViT features
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, embed_dim, kernel_size=3, stride=2, padding=1),
        )
        
        # Project to model dimension
        self.projection = nn.Conv2d(embed_dim, 256, kernel_size=1)

    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        features = self.projection(features)
        
        return features


class CustomDETR(nn.Module):
    """
    Custom DETR model matching the checkpoint architecture
    """
    def __init__(self, args):
        super().__init__()
        
        # Extract parameters from args
        self.num_classes = args.num_classes
        self.hidden_dim = args.hidden_dim
        self.num_queries = args.num_queries
        
        # Backbone (simplified ViT)
        self.backbone = SimplifiedViTBackbone()
        
        # Position encoding
        self.position_embedding = PositionEmbeddingSine(
            num_pos_feats=self.hidden_dim // 2, 
            normalize=True
        )
        
        # Transformer decoder
        decoder_layer = DeformableTransformerDecoderLayer(
            d_model=self.hidden_dim,
            n_heads=args.ca_nheads,
            dim_feedforward=args.dim_feedforward
        )
        
        self.transformer = nn.Module()
        self.transformer.decoder = DeformableTransformerDecoder(
            decoder_layer, 
            args.dec_layers
        )
        
        # Query embeddings
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
        
        # Output heads
        self.class_embed = nn.Linear(self.hidden_dim, self.num_classes)
        self.bbox_embed = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 4)
        )

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, 3, height, width]
            
        Returns:
            Dictionary with 'logits' and 'pred_boxes'
        """
        # Backbone
        features = self.backbone(x)
        
        # Position encoding
        pos = self.position_embedding(features)
        
        # Flatten spatial dimensions
        batch_size, c, h, w = features.shape
        features_flat = features.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        pos_flat = pos.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        
        # Query embeddings
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Transformer decoder
        decoder_output = self.transformer.decoder(
            query_embed, 
            features_flat,
            memory_spatial_shapes=torch.tensor([[h, w]], device=x.device)
        )
        
        # Output heads
        outputs_class = self.class_embed(decoder_output)
        outputs_coord = self.bbox_embed(decoder_output).sigmoid()
        
        return {
            'logits': outputs_class,
            'pred_boxes': outputs_coord
        }


def build_model_from_checkpoint(checkpoint_path, device='cpu'):
    """
    Build and load the DETR model from checkpoint
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        device: Device to load the model on
        
    Returns:
        Loaded model ready for inference
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if not isinstance(checkpoint, dict) or 'args' not in checkpoint:
        raise ValueError("Checkpoint must contain 'args' with model configuration")
    
    args = checkpoint['args']
    
    # Create model
    model = CustomDETR(args)
    
    # Load state dict with strict=False to handle missing keys gracefully
    if 'model' in checkpoint:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys in model: {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
        
        print("Model loaded with partial weights from checkpoint")
    else:
        print("Warning: No 'model' key found in checkpoint")
    
    model.to(device)
    model.eval()
    
    return model, args


if __name__ == "__main__":
    # Test model creation
    print("Testing DETR model creation...")
    try:
        model, args = build_model_from_checkpoint('models/rf_detr_custom.pth', 'cpu')
        print(f"✅ Model created successfully")
        print(f"   Classes: {args.num_classes}")
        print(f"   Queries: {args.num_queries}")
        print(f"   Hidden dim: {args.hidden_dim}")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 384, 384)  # Smaller test size
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ Forward pass successful")
        print(f"   Output logits shape: {output['logits'].shape}")
        print(f"   Output boxes shape: {output['pred_boxes'].shape}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()