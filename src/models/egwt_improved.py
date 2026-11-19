"""
EGWT_Improved — EfficientNet Convolutional Group-Wise Transformer (improved)

Attention mechanism follows plant1 (hybrid attention on mid- and high-level features):
    F̂ = SE(F ⊙ A_s(F))
Order is deliberate: spatial attention damps background FIRST,
then SE reweights channels on the already-cleaned feature map.

Supports three ablation modes (set via `attention_mode` arg):
    'none'       → (A) backbone only — no attention
    'se_only'    → (B) SE only       — channel reweighting, no spatial gate
    'spatial_se' → (C) Spatial → SE  — full plant1 mechanism (default)

Multi-scale fusion (addresses similar disease symptoms):
    Stage1(768) + Stage2(768) + Stage3(1024) → concat [2560] → classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ──────────────────────────────────────────────────────────────────────
#  Shared backbone blocks (identical to baseline EGWT)
# ──────────────────────────────────────────────────────────────────────

class DWConv2d(nn.Module):
    """Depthwise Separable Convolution (DWTE layer)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.bn  = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.bn_out(self.pointwise(self.bn(self.depthwise(x)))))
        return x


class ConvProjection(nn.Module):
    """Convolutional Projection for Q, K, V generation (Eq. 3 in paper)."""
    def __init__(self, dim, proj_dim, kernel_size=3, stride=1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, proj_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.GELU()
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        proj_2d = self.proj(x.transpose(1, 2).view(B, C, H, W))
        B, C_new, H_new, W_new = proj_2d.shape
        return proj_2d.flatten(2).transpose(1, 2), H_new, W_new


class EfficientNetProjection(nn.Module):
    """Stage 3 EfficientNet-B0 Inverted Residual Block Projection."""
    def __init__(self, in_dim, out_dim, expansion=2):
        super().__init__()
        hidden_dim = in_dim * expansion
        self.conv_dw = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False), nn.BatchNorm2d(hidden_dim), nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.SiLU()
        )
        self.se_fc1 = nn.Conv2d(hidden_dim, max(1, hidden_dim // 4), 1)
        self.se_fc2 = nn.Conv2d(max(1, hidden_dim // 4), hidden_dim, 1)
        self.project_out = nn.Sequential(
            nn.Conv2d(hidden_dim, out_dim, 1, bias=False), nn.BatchNorm2d(out_dim)
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        h = self.conv_dw(x.transpose(1, 2).view(B, C, H, W))
        se = torch.sigmoid(self.se_fc2(F.silu(self.se_fc1(F.adaptive_avg_pool2d(h, 1)))))
        out = self.project_out(h * se)
        B, C_new, H_new, W_new = out.shape
        return out.flatten(2).transpose(1, 2), H_new, W_new


class GroupMultiHeadAttention(nn.Module):
    """Group-Wise Multi-Head Attention (G-MHA)."""
    def __init__(self, dim, num_heads=6, num_groups=2, head_dim=64, qkv_bias=True):
        super().__init__()
        self.num_groups = num_groups
        self.group_dim  = dim // num_groups
        self.head_dim   = head_dim
        self.inner_dim  = num_heads * head_dim
        self.scale      = head_dim ** -0.5
        self.qkv_proj   = nn.Linear(self.group_dim, self.inner_dim * 3, bias=qkv_bias)
        self.out_proj   = nn.Linear(self.inner_dim, self.group_dim)
        self.num_heads  = num_heads

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
    """Group-Wise MLP (G-MLP)."""
    def __init__(self, dim, mlp_ratio=4, num_groups=2):
        super().__init__()
        self.num_groups = num_groups
        self.hidden_dim = int((dim // num_groups) * mlp_ratio)
        self.expand  = nn.Linear(dim, dim * mlp_ratio)
        self.act     = nn.GELU()
        self.group_w = nn.Linear(self.hidden_dim, dim // num_groups)

    def forward(self, x):
        B, N, C = x.shape
        h = self.act(self.expand(x))
        h_groups = h.view(B, N, self.num_groups, self.hidden_dim).permute(2, 0, 1, 3)
        return torch.cat([self.group_w(h_groups[k]) for k in range(self.num_groups)], dim=-1)


class GWTBlock(nn.Module):
    """Single Convolutional Group-Wise Transformer Block."""
    def __init__(self, dim, num_heads=6, num_groups=2, head_dim=64, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = GroupMultiHeadAttention(dim, num_heads=num_heads,
                                             num_groups=num_groups, head_dim=head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = GroupMLP(dim, mlp_ratio=mlp_ratio, num_groups=num_groups)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ──────────────────────────────────────────────────────────────────────
#  NEW: Hybrid Spatial → SE Attention  (plant1 mechanism)
#  F̂ = SE(F ⊙ A_s(F))
# ──────────────────────────────────────────────────────────────────────

class SpatialGate(nn.Module):
    """
    Spatial attention gate A_s(F).
    Average-pool and max-pool across channels → 7×7 conv → sigmoid.
    Produces a [B, 1, H, W] map that suppresses background pixels.
    Returned separately so it can be overlaid / used in masking experiments.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)        # [B, 1, H, W]
        max_out, _ = x.max(dim=1, keepdim=True)      # [B, 1, H, W]
        attn_map = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn_map, attn_map                # gated features + raw map


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention.
    Global avg pool → FC → ReLU → FC → Sigmoid → channel scale.
    Applied AFTER spatial gating so it reweights channels on clean features.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        scale = self.fc(self.pool(x).view(B, C)).view(B, C, 1, 1)
        return x * scale


class HybridAttention(nn.Module):
    """
    Unified module supporting all three ablation conditions:

        mode='none'       → (A) No attention — pure backbone
        mode='se_only'    → (B) SE only — channel attention, no spatial gate
        mode='spatial_se' → (C) Spatial → SE — plant1 full mechanism (default)

    In mode C the order F̂ = SE(F ⊙ A_s(F)) is preserved:
    spatial gate runs first so background is damped before SE reweights channels.

    `return_attn=True` returns the spatial attention map [B, 1, H, W]
    (None for modes A and B) — used for the 20%-masking experiment and
    Grad-CAM overlay comparison.
    """
    def __init__(self, channels, mode='spatial_se', reduction=16, spatial_kernel=7):
        super().__init__()
        assert mode in ('none', 'se_only', 'spatial_se'), \
            f"attention_mode must be 'none', 'se_only', or 'spatial_se', got '{mode}'"
        self.mode = mode
        if mode == 'se_only':
            self.se = SEBlock(channels, reduction)
        elif mode == 'spatial_se':
            self.spatial = SpatialGate(spatial_kernel)
            self.se      = SEBlock(channels, reduction)

    def forward(self, x, return_attn=False):
        attn_map = None
        if self.mode == 'none':
            out = x
        elif self.mode == 'se_only':
            out = self.se(x)
        else:  # spatial_se
            gated, attn_map = self.spatial(x)   # background damped
            out = self.se(gated)                 # channels reweighted on clean features
        return (out, attn_map) if return_attn else out


# ──────────────────────────────────────────────────────────────────────
#  EGWT_Improved
# ──────────────────────────────────────────────────────────────────────

class EGWT_Improved(nn.Module):
    """
    Improved EGWT addressing three research problems:

    Problem 1 — Background interference (plant1 mechanism):
        HybridAttention (Spatial → SE) inserted after Stage 2 and Stage 3 DWConv
        (mid- and high-level features, as in plant1).
        Ablation via `attention_mode`:
            'none'       → baseline backbone only
            'se_only'    → adds SE channel attention
            'spatial_se' → adds spatial gate then SE (full, default)

    Problem 2 — Similar disease symptoms (multi-scale fusion):
        Global-pool features from Stage 1, 2, 3 and concatenate before classifier.
        Stage 1: 56×56 fine-grain texture  → [B, 768]
        Stage 2: 28×28 mid-level patterns  → [B, 768]
        Stage 3: 14×14 abstract semantics  → [B, 1024]
        Fused:                              → [B, 2560] → Linear(512) → classes

    Problem 3 — Generalization:
        Backbone unchanged → trained weights transfer to cross-dataset eval
        on India Mendeley dataset (evaluate_cross.py).

    Args:
        num_classes    (int)  : number of output classes (default 5)
        attention_mode (str)  : 'none' / 'se_only' / 'spatial_se' (default)
    """
    def __init__(self, num_classes=5, in_channels=3, attention_mode='spatial_se'):
        super().__init__()
        self.attention_mode = attention_mode

        # ── Stage 1 (low-level, 56×56) ───────────────────────────────
        self.stage1_dwte      = DWConv2d(in_channels, 64, kernel_size=7, stride=4, padding=3)
        # No attention on Stage 1 — too early, features not yet semantic
        self.stage1_proj      = ConvProjection(64, 768, kernel_size=3, stride=1)
        self.stage1_blocks    = nn.ModuleList([
            GWTBlock(dim=768, num_heads=6, num_groups=2, head_dim=64, mlp_ratio=4)
            for _ in range(2)
        ])
        self.stage1_norm      = nn.LayerNorm(768)    # for multi-scale pool
        self.stage1_to_stage2 = nn.Conv2d(768, 64, kernel_size=1)

        # ── Stage 2 (mid-level, 28×28) ───────────────────────────────
        self.stage2_dwte      = DWConv2d(64, 192, kernel_size=3, stride=2, padding=1)
        self.attn2            = HybridAttention(192, mode=attention_mode)   # Problem 1
        self.stage2_proj      = ConvProjection(192, 768, kernel_size=3, stride=1)
        self.stage2_blocks    = nn.ModuleList([
            GWTBlock(dim=768, num_heads=6, num_groups=2, head_dim=64, mlp_ratio=4)
            for _ in range(10)
        ])
        self.stage2_norm      = nn.LayerNorm(768)    # for multi-scale pool
        self.stage2_to_stage3 = nn.Conv2d(768, 192, kernel_size=1)

        # ── Stage 3 (high-level, 14×14) ──────────────────────────────
        self.stage3_dwte   = DWConv2d(192, 1024, kernel_size=3, stride=2, padding=1)
        self.attn3         = HybridAttention(1024, mode=attention_mode)    # Problem 1
        self.stage3_proj   = EfficientNetProjection(in_dim=1024, out_dim=1024)
        self.stage3_blocks = nn.ModuleList([
            GWTBlock(dim=1024, num_heads=12, num_groups=4, head_dim=32, mlp_ratio=4)
            for _ in range(3)
        ])
        self.norm = nn.LayerNorm(1024)

        # ── Multi-Scale Fusion Head ───────────────────────────────────
        # 768 (S1) + 768 (S2) + 1024 (S3) = 2560
        self.fusion_head = nn.Sequential(
            nn.Linear(768 + 768 + 1024, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward_features(self, x, return_attn_maps=False):
        """
        Args:
            x               : [B, 3, 224, 224]
            return_attn_maps: if True, also return list of spatial attention
                              maps [attn_map_s2, attn_map_s3] for masking
                              experiment / Grad-CAM overlay.
        Returns:
            fused features [B, 2560] and optionally list of attention maps.
        """
        B = x.shape[0]
        attn_maps = []

        # ── Stage 1 ──────────────────────────────────────────────────
        x = self.stage1_dwte(x)                      # [B, 64, 56, 56]
        H1, W1 = x.shape[2], x.shape[3]
        tokens, H1p, W1p = self.stage1_proj(x.flatten(2).transpose(1, 2), H1, W1)
        for blk in self.stage1_blocks:
            tokens = checkpoint(blk, tokens, use_reentrant=False)
        feat1 = self.stage1_norm(tokens).mean(dim=1)          # [B, 768]
        s1_out = self.stage1_to_stage2(
            tokens.transpose(1, 2).view(B, -1, H1p, W1p))

        # ── Stage 2 ──────────────────────────────────────────────────
        x = self.stage2_dwte(s1_out)                 # [B, 192, 28, 28]
        if return_attn_maps:
            x, am2 = self.attn2(x, return_attn=True)
            attn_maps.append(am2)                    # [B, 1, 28, 28]
        else:
            x = self.attn2(x)
        H2, W2 = x.shape[2], x.shape[3]
        tokens, H2p, W2p = self.stage2_proj(x.flatten(2).transpose(1, 2), H2, W2)
        for blk in self.stage2_blocks:
            tokens = checkpoint(blk, tokens, use_reentrant=False)
        feat2 = self.stage2_norm(tokens).mean(dim=1)          # [B, 768]
        s2_out = self.stage2_to_stage3(
            tokens.transpose(1, 2).view(B, -1, H2p, W2p))

        # ── Stage 3 ──────────────────────────────────────────────────
        x = self.stage3_dwte(s2_out)                 # [B, 1024, 14, 14]
        if return_attn_maps:
            x, am3 = self.attn3(x, return_attn=True)
            attn_maps.append(am3)                    # [B, 1, 14, 14]
        else:
            x = self.attn3(x)
        H3, W3 = x.shape[2], x.shape[3]
        tokens, H3p, W3p = self.stage3_proj(x.flatten(2).transpose(1, 2), H3, W3)
        for blk in self.stage3_blocks:
            tokens = checkpoint(blk, tokens, use_reentrant=False)
        feat3 = self.norm(tokens).mean(dim=1)                  # [B, 1024]

        # ── Multi-Scale Fusion ────────────────────────────────────────
        fused = torch.cat([feat1, feat2, feat3], dim=-1)       # [B, 2560]

        if return_attn_maps:
            return fused, attn_maps
        return fused

    def forward(self, x):
        return self.fusion_head(self.forward_features(x))

    def forward_with_attn(self, x):
        """
        Used for masking experiment and attention overlay.
        Returns (logits, [attn_map_s2, attn_map_s3]).
        attn_map shapes: [B, 1, 28, 28] and [B, 1, 14, 14].
        """
        fused, attn_maps = self.forward_features(x, return_attn_maps=True)
        logits = self.fusion_head(fused)
        return logits, attn_maps
