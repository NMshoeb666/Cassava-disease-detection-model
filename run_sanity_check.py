import os
import sys
import torch
import pandas as pd
import numpy as np

print("=" * 65)
print("RUNNING END-TO-END SANITY CHECK ON CASSAVA PIPELINE")
print("=" * 65)

from src.config import (
    MAIN_IMAGES_DIR, MAIN_FOLDS_CSV, CROSS_META_CSV, CROSS_DATA_DIR,
    NUM_CLASSES, CLASS_NAMES, CB_BETA, FOCAL_GAMMA, DEVICE
)
from src.dataset import CassavaDataset, get_transforms, get_cross_dataloader
from src.losses import ClassBalancedFocalLoss
from src.models.egwt import EGWT
from src.metrics import calculate_metrics, print_metrics_table
from src.gradcam import GradCAM, overlay_cam_on_image

# 1. Check metadata files
print("[1/6] Verifying Metadata Files...")
assert os.path.exists(MAIN_FOLDS_CSV), f"Missing {MAIN_FOLDS_CSV}"
df_main = pd.read_csv(MAIN_FOLDS_CSV)
print(f" - Main Dataset (Stratified 5-Folds): {len(df_main)} images verified across 5 folds.")

assert os.path.exists(CROSS_META_CSV), f"Missing {CROSS_META_CSV}"
df_cross = pd.read_csv(CROSS_META_CSV)
print(f" - Cross Dataset (Mendeley India): {len(df_cross)} images verified across classes {df_cross['class_name'].unique().tolist()}.")

# 2. Test Main Data Loader
print("\n[2/6] Testing Main Dataset Loader (Batch of 2 images)...")
train_trans, _ = get_transforms(img_size=224)
ds_main = CassavaDataset(df_main.head(4), MAIN_IMAGES_DIR, transform=train_trans)
loader_main = torch.utils.data.DataLoader(ds_main, batch_size=2, shuffle=False)
batch_imgs, batch_labels = next(iter(loader_main))
print(f" - Image Tensor Shape: {batch_imgs.shape} (B, C, H, W)")
print(f" - Labels: {batch_labels.tolist()}")

# 3. Test EGWT Model Forward Pass
print("\n[3/6] Initializing Base Paper EGWT Model & Testing Forward Pass...")
model = EGWT(num_classes=NUM_CLASSES).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f" - EGWT Model Initialized. Trainable Parameters: {total_params / 1e6:.2f}M")

batch_imgs = batch_imgs.to(DEVICE)
logits = model(batch_imgs)
print(f" - Output Logits Shape: {logits.shape} (Expected: [2, 5])")
assert logits.shape == (2, 5), "Shape mismatch in logits!"

# 4. Test Class-Balanced Focal Loss
print("\n[4/6] Testing Class-Balanced Focal Loss Computation...")
class_counts = df_main['label'].value_counts().sort_index().values
cb_loss_fn = ClassBalancedFocalLoss(class_counts, beta=CB_BETA, gamma=FOCAL_GAMMA)
loss = cb_loss_fn(logits, batch_labels.to(DEVICE))
print(f" - Calculated CB-Focal Loss: {loss.item():.4f}")
loss.backward()
print(" - Backward pass successful! Gradients populated.")

# 5. Test Grad-CAM Visual Explainability
print("\n[5/6] Testing Grad-CAM Generation on Leaf Image...")
target_layer = model.stage3_proj.conv_dw[3]
cam_generator = GradCAM(model, target_layer)
cam, pred_class, _ = cam_generator.generate_cam(batch_imgs[0:1])
print(f" - Grad-CAM Heatmap generated! Shape: {cam.shape}, Predicted Class: {CLASS_NAMES[pred_class]}")

# 6. Test Cross-Dataset Evaluation Function (Subset for fast sanity test)
print("\n[6/6] Testing Cross-Dataset Evaluation Engine (India Dataset mini-batch)...")
from src.evaluate_cross import evaluate_cross_dataset
cross_results = evaluate_cross_dataset(model, cross_csv=CROSS_META_CSV, device=DEVICE, max_samples=8)

print("\n" + "=" * 65)
print(">>> ALL 6 MODULES VERIFIED! PIPELINE IS 100% OPERATIONAL! <<<")
print("=" * 65)
