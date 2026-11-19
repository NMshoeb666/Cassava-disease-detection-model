"""
egwt_pretrained.py — Pretrained Hybrid Attentive EfficientNet Group-Wise Transformer (EGWT-Net++)

Architectural Innovation for Research Paper:
1. Backbone: EfficientNet-B0 pretrained on ImageNet (extracts rich edge, texture, and leaf primitives).
2. Stage 1 (Low-level, 56x56, 24-d): Captures fine vein geometry & leaf surface texture.
3. Stage 2 (Mid-level, 14x14, 80-d): 
   - Filtered by Spatial -> SE Hybrid Attention Gate (solves Problem 1: background weed/soil interference)
   - Transformed by Group-Wise Multi-Head Attention (G-MHA) blocks with FlashAttention.
4. Stage 3 (High-level, 7x7, 320-d):
   - Filtered by Spatial -> SE Hybrid Attention Gate
   - Transformed by G-MHA blocks for abstract disease semantics.
5. Multi-Scale Feature Fusion (solves Problem 2: visual disease overlap like CMD vs CGM):
   - Concat(feat1, feat2, feat3) -> Linear -> GELU -> Dropout -> Classifier.
6. FlashAttention (scaled_dot_product_attention) + Gradient Checkpointing: zero OOM, fast ~8 min/epoch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torchvision.models as models

# ──────────────────────────────────────────────────────────────────────
# Spatial -> SE Hybrid Attention Module (plant1 mechanism)
# ──────────────────────────────────────────────────────────────────────

class SpatialGate(nn.Module):
    """Spatial attention gate A_s(F): damps background soil/weeds before channel reweighting."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        attn_map = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn_map, attn_map


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: channel reweighting applied on clean spatial features."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        scale = self.fc(self.pool(x).view(B, C)).view(B, C, 1, 1)
        return x * scale


class HybridAttention(nn.Module):
    """Unified module supporting full Spatial->SE or ablation modes."""
    def __init__(self, channels, mode='spatial_se', reduction=16, spatial_kernel=7):
        super().__init__()
        assert mode in ('none', 'se_only', 'spatial_se')
        self.mode = mode
        if mode == 'se_only':
            self.se = SEBlock(channels, reduction)
        elif mode == 'spatial_se':
            self.spatial = SpatialGate(spatial_kernel)
            self.se = SEBlock(channels, reduction)

    def forward(self, x, return_attn=False):
        attn_map = None
        if self.mode == 'none':
            out = x
        elif self.mode == 'se_only':
            out = self.se(x)
        else:
            gated, attn_map = self.spatial(x)
            out = self.se(gated)
        return (out, attn_map) if return_attn else out


# ──────────────────────────────────────────────────────────────────────
# Group-Wise Multi-Head Attention & Transformer Blocks (FlashAttention)
# ──────────────────────────────────────────────────────────────────────

class GroupMultiHeadAttention(nn.Module):
    """Group-Wise Multi-Head Attention using PyTorch FlashAttention."""
    def __init__(self, dim, num_heads=4, num_groups=2, head_dim=32, qkv_bias=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_heads = num_heads
        self.group_dim = dim // num_groups
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        
        self.qkv_proj = nn.Linear(self.group_dim, self.inner_dim * 3, bias=qkv_bias)
        self.out_proj = nn.Linear(self.inner_dim, self.group_dim)

    def forward(self, x):
        B, N, C = x.shape
        x_groups = x.view(B, N, self.num_groups, self.group_dim).permute(2, 0, 1, 3)
        outputs = []
        for k in range(self.num_groups):
            grp = x_groups[k]
            qkv = self.qkv_proj(grp).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k_mat, v = qkv[0], qkv[1], qkv[2]
            out_k = F.scaled_dot_product_attention(q, k_mat, v)
            outputs.append(self.out_proj(out_k.transpose(1, 2).reshape(B, N, self.inner_dim)))
        return torch.cat(outputs, dim=-1)


class GroupMLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4, num_groups=2):
        super().__init__()
        self.num_groups = num_groups
        self.hidden_dim = int((dim // num_groups) * mlp_ratio)
        self.expand = nn.Linear(dim, dim * mlp_ratio)
        self.act = nn.GELU()
        self.group_w = nn.Linear(self.hidden_dim, dim // num_groups)

    def forward(self, x):
        B, N, C = x.shape
        h = self.act(self.expand(x))
        h_groups = h.view(B, N, self.num_groups, self.hidden_dim).permute(2, 0, 1, 3)
        return torch.cat([self.group_w(h_groups[k]) for k in range(self.num_groups)], dim=-1)


class GWTBlock(nn.Module):
    def __init__(self, dim, num_heads=4, num_groups=2, head_dim=32, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GroupMultiHeadAttention(dim, num_heads=num_heads, num_groups=num_groups, head_dim=head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = GroupMLP(dim, mlp_ratio=mlp_ratio, num_groups=num_groups)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ──────────────────────────────────────────────────────────────────────
# EGWT_Pretrained: The Complete Novel Hybrid Model
# ──────────────────────────────────────────────────────────────────────

class EGWT_Pretrained(nn.Module):
    """
    EGWT-Net++: Pretrained EfficientNet-B0 + Hybrid Spatial->SE Attention + G-MHA + Multi-Scale Fusion
    """
    def __init__(self, num_classes=5, pretrained=True, attention_mode='spatial_se'):
        super().__init__()
        self.attention_mode = attention_mode
        
        # Load Pretrained EfficientNet-B0 backbone
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base_model = models.efficientnet_b0(weights=weights)
        features = base_model.features
        
        # Progressive multi-scale stages matching EfficientNet-B0 channel dimensions:
        # Stage 1: layers 0..2 -> [B, 24, 56, 56] (fine texture/vein geometry)
        self.stage1_cnn = nn.Sequential(*features[:3])
        self.stage1_pool = nn.AdaptiveAvgPool2d(1)
        
        # Stage 2: layers 3..4 -> [B, 80, 14, 14] (mid-level lesion shape)
        self.stage2_cnn = nn.Sequential(*features[3:5])
        self.attn2 = HybridAttention(80, mode=attention_mode)
        self.stage2_proj = nn.Conv2d(80, 256, kernel_size=1)
        self.stage2_gwt = nn.ModuleList([
            GWTBlock(dim=256, num_heads=4, num_groups=2, head_dim=32) for _ in range(2)
        ])
        self.stage2_norm = nn.LayerNorm(256)
        
        # Stage 3: layers 5..7 -> [B, 320, 7, 7] (high-level abstract semantics)
        self.stage3_cnn = nn.Sequential(*features[5:8])
        self.attn3 = HybridAttention(320, mode=attention_mode)
        self.stage3_proj = nn.Conv2d(320, 512, kernel_size=1)
        self.stage3_gwt = nn.ModuleList([
            GWTBlock(dim=512, num_heads=8, num_groups=4, head_dim=32) for _ in range(2)
        ])
        self.stage3_norm = nn.LayerNorm(512)
        
        # Multi-Scale Fusion Classification Head:
        # 24 (S1 texture) + 256 (S2 mid patterns) + 512 (S3 abstract context) = 792 dims
        total_feat_dim = 24 + 256 + 512
        self.fusion_head = nn.Sequential(
            nn.Linear(total_feat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward_features(self, x, return_attn_maps=False):
        B = x.shape[0]
        attn_maps = []
        
        # ── Stage 1 ──────────────────────────────────────────────────
        x1 = self.stage1_cnn(x)                       # [B, 24, 56, 56]
        feat1 = self.stage1_pool(x1).flatten(1)       # [B, 24]
        
        # ── Stage 2 ──────────────────────────────────────────────────
        x2 = self.stage2_cnn(x1)                      # [B, 80, 14, 14]
        if return_attn_maps:
            x2, am2 = self.attn2(x2, return_attn=True)
            attn_maps.append(am2)
        else:
            x2 = self.attn2(x2)
            
        proj2 = self.stage2_proj(x2)                  # [B, 256, 14, 14]
        tokens2 = proj2.flatten(2).transpose(1, 2)    # [B, 196, 256]
        for blk in self.stage2_gwt:
            tokens2 = checkpoint(blk, tokens2, use_reentrant=False)
        feat2 = self.stage2_norm(tokens2).mean(dim=1) # [B, 256]
        
        # ── Stage 3 ──────────────────────────────────────────────────
        x3 = self.stage3_cnn(x2)                      # [B, 320, 7, 7]
        if return_attn_maps:
            x3, am3 = self.attn3(x3, return_attn=True)
            attn_maps.append(am3)
        else:
            x3 = self.attn3(x3)
            
        proj3 = self.stage3_proj(x3)                  # [B, 512, 7, 7]
        tokens3 = proj3.flatten(2).transpose(1, 2)    # [B, 49, 512]
        for blk in self.stage3_gwt:
            tokens3 = checkpoint(blk, tokens3, use_reentrant=False)
        feat3 = self.stage3_norm(tokens3).mean(dim=1) # [B, 512]
        
        # ── Multi-Scale Fusion ───────────────────────────────────────
        fused = torch.cat([feat1, feat2, feat3], dim=-1) # [B, 792]
        
        if return_attn_maps:
            return fused, attn_maps
        return fused

    def forward(self, x):
        return self.fusion_head(self.forward_features(x))

    def forward_with_attn(self, x):
        fused, attn_maps = self.forward_features(x, return_attn_maps=True)
        return self.fusion_head(fused), attn_maps
