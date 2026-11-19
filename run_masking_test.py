"""
Attention Masking Experiment — quantifies whether the model actually uses
the spatial attention regions it learned.

Protocol (from plant1 reference paper):
    1. For each test image, extract spatial attention maps from Stage 2 and 3.
    2. Upsample to input resolution (224×224).
    3. Average the two maps → combined importance map.
    4. Mask the TOP 20% most-attended pixels (set to channel mean).
    5. Also mask a RANDOM 20% as control.
    6. Measure accuracy drop in each case.

Expected result if attention is meaningful:
    Masking top-20% attended pixels → larger accuracy drop than random masking.

Run:
    python run_masking_test.py --checkpoint outputs/egwt_improved_fold0.pth --fold 0
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import MAIN_FOLDS_CSV, MAIN_IMAGES_DIR, DEVICE, NUM_CLASSES
from src.models.egwt_improved import EGWT_Improved
from src.models.egwt_pretrained import EGWT_Pretrained

# ── Minimal dataset (no dependency on dataset.py) ────────────────────
from torchvision import transforms
from PIL import Image

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMG_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

class SimpleCassavaDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_dir, transform):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.image_dir, row['image_id'])).convert('RGB')
        return self.transform(img), int(row['label'])


def apply_mask(images, mask_positions):
    """
    Zero-out (set to channel mean) the specified pixel positions.
    images         : [B, 3, H, W] normalised tensor
    mask_positions : [B, H, W] bool tensor — True = mask this pixel
    """
    masked = images.clone()
    # Channel means in normalised space are 0 (after normalisation)
    # so setting masked pixels to 0 ≈ setting to dataset mean
    mask = mask_positions.unsqueeze(1).expand_as(masked)
    masked[mask] = 0.0
    return masked


def get_top_k_mask(attn_map, k_frac=0.20):
    """
    attn_map : [B, 1, H, W]
    Returns bool mask [B, H, W] — True where attention is in top k_frac.
    """
    B, _, H, W = attn_map.shape
    flat = attn_map.view(B, -1)                          # [B, H*W]
    threshold = torch.quantile(flat, 1.0 - k_frac, dim=1, keepdim=True)
    top_mask = (flat >= threshold).view(B, H, W)
    return top_mask


def get_random_mask(attn_map, k_frac=0.20, seed=None):
    """Random mask with the same number of pixels as the top-k mask."""
    B, _, H, W = attn_map.shape
    n_pixels = H * W
    n_mask   = int(n_pixels * k_frac)
    if seed is not None:
        torch.manual_seed(seed)
    rand_mask = torch.zeros(B, H, W, dtype=torch.bool)
    for b in range(B):
        perm = torch.randperm(n_pixels)[:n_mask]
        idx = torch.stack([perm // W, perm % W], dim=1)
        rand_mask[b, idx[:, 0], idx[:, 1]] = True
    return rand_mask


def run_masking_test(checkpoint_path, fold, batch_size=8, k_frac=0.20,
                     attention_mode='spatial_se', max_batches=None, model_type='pretrained'):
    print(f"\n{'='*60}")
    print(f"Attention Masking Experiment")
    print(f"Model Type : {model_type}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Fold       : {fold}  (using fold as validation set)")
    print(f"Mask frac  : {k_frac*100:.0f}%")
    print(f"Attn mode  : {attention_mode}")
    print(f"{'='*60}\n")

    # ── Load model ───────────────────────────────────────────────────
    if model_type == 'pretrained':
        model = EGWT_Pretrained(num_classes=NUM_CLASSES, pretrained=False, attention_mode=attention_mode).to(DEVICE)
    else:
        model = EGWT_Improved(num_classes=NUM_CLASSES, attention_mode=attention_mode).to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")

    # ── Load validation fold ──────────────────────────────────────────
    df = pd.read_csv(MAIN_FOLDS_CSV)
    val_df = df[df['fold'] == fold].reset_index(drop=True)
    dataset = SimpleCassavaDataset(val_df, MAIN_IMAGES_DIR, val_transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Validation samples: {len(dataset)}\n")

    correct_normal  = 0
    correct_top_k   = 0
    correct_random  = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            B, C, H, W = images.shape

            # ── Normal inference ─────────────────────────────────────
            logits, attn_maps = model.forward_with_attn(images)
            preds_normal = logits.argmax(dim=-1)
            correct_normal += (preds_normal == labels).sum().item()

            if attention_mode == 'spatial_se' and len(attn_maps) == 2:
                # Upsample both maps to 224×224 and average
                am2 = F.interpolate(attn_maps[0], size=(H, W), mode='bilinear', align_corners=False)
                am3 = F.interpolate(attn_maps[1], size=(H, W), mode='bilinear', align_corners=False)
                combined_attn = (am2 + am3) / 2.0      # [B, 1, 224, 224]

                # ── Top-k mask ────────────────────────────────────────
                top_mask = get_top_k_mask(combined_attn, k_frac).to(DEVICE)
                masked_images = apply_mask(images, top_mask)
                logits_top = model(masked_images)
                correct_top_k += (logits_top.argmax(dim=-1) == labels).sum().item()

                # ── Random mask (control) ─────────────────────────────
                rand_mask = get_random_mask(combined_attn, k_frac).to(DEVICE)
                masked_rand = apply_mask(images, rand_mask)
                logits_rand = model(masked_rand)
                correct_random += (logits_rand.argmax(dim=-1) == labels).sum().item()
            else:
                # No spatial maps available (mode=none or se_only)
                correct_top_k  += (preds_normal == labels).sum().item()
                correct_random += (preds_normal == labels).sum().item()

            total += labels.size(0)

            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx+1}: processed {total} samples so far...")

    # ── Report ───────────────────────────────────────────────────────
    acc_normal = correct_normal / total * 100
    acc_topk   = correct_top_k  / total * 100
    acc_rand   = correct_random / total * 100
    drop_topk  = acc_normal - acc_topk
    drop_rand  = acc_normal - acc_rand

    print(f"\n{'='*60}")
    print(f"Results — {total} validation images, top-{k_frac*100:.0f}% masking")
    print(f"{'='*60}")
    print(f"Normal accuracy         : {acc_normal:.2f}%")
    print(f"Top-{k_frac*100:.0f}% attended masked  : {acc_topk:.2f}%  (drop: {drop_topk:+.2f}%)")
    print(f"Random {k_frac*100:.0f}% masked         : {acc_rand:.2f}%  (drop: {drop_rand:+.2f}%)")
    print(f"\nAttention vs Random drop : {drop_topk - drop_rand:+.2f}%")
    if drop_topk > drop_rand:
        print("✓ Masking attended regions hurts MORE → model uses spatial attention correctly.")
    else:
        print("✗ Attended masking does not hurt more → attention may not be spatially meaningful.")
    print(f"{'='*60}\n")

    return {
        'acc_normal': acc_normal,
        'acc_topk_masked': acc_topk,
        'acc_random_masked': acc_rand,
        'drop_topk': drop_topk,
        'drop_random': drop_rand,
        'attention_advantage': drop_topk - drop_rand
    }


if __name__ == '__main__':
    import torch.nn.functional as F

    parser = argparse.ArgumentParser(description='Attention masking experiment')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained EGWT_Improved .pth checkpoint')
    parser.add_argument('--fold', type=int, default=0,
                        help='Which fold to use as validation set (0-4)')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--k_frac', type=float, default=0.20,
                        help='Fraction of pixels to mask (default 0.20 = top 20%%)')
    parser.add_argument('--attention_mode', type=str, default='spatial_se',
                        choices=['none', 'se_only', 'spatial_se'])
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Limit batches for quick test (omit for full eval)')
    parser.add_argument('--model', type=str, default='pretrained', choices=['pretrained', 'improved'])
    args = parser.parse_args()

    results = run_masking_test(
        checkpoint_path=args.checkpoint,
        fold=args.fold,
        batch_size=args.batch_size,
        k_frac=args.k_frac,
        attention_mode=args.attention_mode,
        max_batches=args.max_batches,
        model_type=args.model
    )
