import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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
        self.bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.pointwise(x)
        x = self.bn_out(x)
        x = self.act(x)
        return x

class ConvProjection(nn.Module):
    """
    Convolutional Projection for Q, K, V generation as in Eq. (3):
    x_i^{q/k/v} = Flatten(DWConv2D(Reshape2D(x_i), S))
    """
    def __init__(self, dim, proj_dim, kernel_size=3, stride=1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, proj_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.GELU()
        )
        
    def forward(self, x, H, W):
        B, N, C = x.shape
        x_2d = x.transpose(1, 2).view(B, C, H, W)
        proj_2d = self.proj(x_2d)
        B, C_new, H_new, W_new = proj_2d.shape
        proj_1d = proj_2d.flatten(2).transpose(1, 2)
        return proj_1d, H_new, W_new

class EfficientNetProjection(nn.Module):
    """
    Stage 3 EfficientNet-B0 Inverted Residual Block Projection
    """
    def __init__(self, in_dim, out_dim, expansion=2):
        super().__init__()
        hidden_dim = in_dim * expansion
        self.conv_dw = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )
        self.se_fc1 = nn.Conv2d(hidden_dim, max(1, hidden_dim // 4), kernel_size=1)
        self.se_fc2 = nn.Conv2d(max(1, hidden_dim // 4), hidden_dim, kernel_size=1)
        self.project_out = nn.Sequential(
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim)
        )
        
    def forward(self, x, H, W):
        B, N, C = x.shape
        x_2d = x.transpose(1, 2).view(B, C, H, W)
        h = self.conv_dw(x_2d)
        se = torch.sigmoid(self.se_fc2(F.silu(self.se_fc1(F.adaptive_avg_pool2d(h, 1)))))
        h = h * se
        out = self.project_out(h)
        B, C_new, H_new, W_new = out.shape
        return out.flatten(2).transpose(1, 2), H_new, W_new

class GroupMultiHeadAttention(nn.Module):
    """
    Group-Wise Multi-Head Attention (G-MHA).
    Divides feature channels into k groups and performs attention with shared parameters.
    """
    def __init__(self, dim, num_heads=6, num_groups=2, head_dim=64, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_groups = num_groups
        self.num_heads = num_heads
        self.group_dim = dim // num_groups
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale = self.head_dim ** -0.5
        
        # Shared projection weights across groups
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
            out_k = out_k.transpose(1, 2).reshape(B, N, self.inner_dim)
            out_k = self.out_proj(out_k)
            outputs.append(out_k)
            
        out = torch.cat(outputs, dim=-1)
        return out

class GroupMLP(nn.Module):
    """
    Group-Wise MLP (G-MLP).
    Expanding projection followed by shared linear transformation across groups.
    """
    def __init__(self, dim, mlp_ratio=4, num_groups=2):
        super().__init__()
        self.dim = dim
        self.num_groups = num_groups
        self.group_dim = dim // num_groups
        self.hidden_dim = int(self.group_dim * mlp_ratio)
        
        self.expand = nn.Linear(dim, dim * mlp_ratio)
        self.act = nn.GELU()
        self.group_w = nn.Linear(self.hidden_dim, self.group_dim)
        
    def forward(self, x):
        B, N, C = x.shape
        h = self.act(self.expand(x))
        h_groups = h.view(B, N, self.num_groups, self.hidden_dim).permute(2, 0, 1, 3)
        outputs = [self.group_w(h_groups[k]) for k in range(self.num_groups)]
        out = torch.cat(outputs, dim=-1)
        return out

class GWTBlock(nn.Module):
    """Single Convolutional Group-Wise Transformer Block"""
    def __init__(self, dim, num_heads=6, num_groups=2, head_dim=64, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GroupMultiHeadAttention(dim, num_heads=num_heads, num_groups=num_groups, head_dim=head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = GroupMLP(dim, mlp_ratio=mlp_ratio, num_groups=num_groups)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class EGWT(nn.Module):
    """
    EfficientNet Convolutional Group-Wise Transformer (EGWT).
    Architecture specification from Feng et al. (IEEE Access 2024, Table 2).
    """
    def __init__(self, num_classes=5, in_channels=3):
        super().__init__()
        
        # --- STAGE 1 ---
        self.stage1_dwte = DWConv2d(in_channels, 64, kernel_size=7, stride=4, padding=3)
        self.stage1_proj = ConvProjection(64, 768, kernel_size=3, stride=1)
        self.stage1_blocks = nn.ModuleList([
            GWTBlock(dim=768, num_heads=6, num_groups=2, head_dim=64, mlp_ratio=4) for _ in range(2)
        ])
        self.stage1_to_stage2 = nn.Conv2d(768, 64, kernel_size=1)
        
        # --- STAGE 2 ---
        self.stage2_dwte = DWConv2d(64, 192, kernel_size=3, stride=2, padding=1)
        self.stage2_proj = ConvProjection(192, 768, kernel_size=3, stride=1)
        self.stage2_blocks = nn.ModuleList([
            GWTBlock(dim=768, num_heads=6, num_groups=2, head_dim=64, mlp_ratio=4) for _ in range(10)
        ])
        self.stage2_to_stage3 = nn.Conv2d(768, 192, kernel_size=1)
        
        # --- STAGE 3 ---
        self.stage3_dwte = DWConv2d(192, 1024, kernel_size=3, stride=2, padding=1)
        self.stage3_proj = EfficientNetProjection(in_dim=1024, out_dim=1024)
        self.stage3_blocks = nn.ModuleList([
            GWTBlock(dim=1024, num_heads=12, num_groups=4, head_dim=32, mlp_ratio=4) for _ in range(3)
        ])
        
        # Classifier Head
        self.norm = nn.LayerNorm(1024)
        self.head = nn.Linear(1024, num_classes)
        
    def forward_features(self, x):
        B = x.shape[0]
        
        # Stage 1
        x = self.stage1_dwte(x)
        H1, W1 = x.shape[2], x.shape[3]
        tokens, H1_proj, W1_proj = self.stage1_proj(x.flatten(2).transpose(1, 2), H1, W1)
        for blk in self.stage1_blocks:
            tokens = checkpoint(blk, tokens, use_reentrant=False)
        s1_out = self.stage1_to_stage2(tokens.transpose(1, 2).view(B, -1, H1_proj, W1_proj))
        
        # Stage 2 (10 blocks — biggest memory consumer, always checkpoint)
        x = self.stage2_dwte(s1_out)
        H2, W2 = x.shape[2], x.shape[3]
        tokens, H2_proj, W2_proj = self.stage2_proj(x.flatten(2).transpose(1, 2), H2, W2)
        for blk in self.stage2_blocks:
            tokens = checkpoint(blk, tokens, use_reentrant=False)
        s2_out = self.stage2_to_stage3(tokens.transpose(1, 2).view(B, -1, H2_proj, W2_proj))
        
        # Stage 3
        x = self.stage3_dwte(s2_out)
        H3, W3 = x.shape[2], x.shape[3]
        tokens, H3_proj, W3_proj = self.stage3_proj(x.flatten(2).transpose(1, 2), H3, W3)
        for blk in self.stage3_blocks:
            tokens = checkpoint(blk, tokens, use_reentrant=False)
            
        tokens = self.norm(tokens)
        feat = tokens.mean(dim=1)
        return feat
        
    def forward(self, x):
        feat = self.forward_features(x)
        logits = self.head(feat)
        return logits
