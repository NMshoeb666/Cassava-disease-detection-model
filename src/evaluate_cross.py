"""
evaluate_cross.py — Cross-dataset generalization evaluation on Mendeley India dataset.
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from .config import (
    CROSS_DATA_DIR, CROSS_META_CSV, CROSS_CLASSES_SUBSET,
    CLASS_NAMES, DEVICE, IMG_SIZE
)
from .dataset import get_cross_dataloader
from .metrics import calculate_metrics, print_metrics_table

def evaluate_cross_dataset(model, cross_csv=CROSS_META_CSV, device=DEVICE, max_samples=None):
    """
    Evaluates a trained model on the independent Mendeley India dataset (228 images).
    Evaluates on the 3 shared classes: CBB (0), CMD (3), Healthy (4).
    """
    if not os.path.exists(cross_csv):
        print(f"Error: Cross dataset metadata not found at {cross_csv}")
        return None

    cross_df = pd.read_csv(cross_csv)
    if max_samples is not None:
        cross_df = cross_df.head(max_samples)

    loader = get_cross_dataloader(cross_df, batch_size=8)
    model.eval()

    all_preds, all_targets, all_probs = [], [], []

    print("\n" + "=" * 65)
    print("STARTING CROSS-DATASET GENERALIZATION EVALUATION (INDIA)")
    print(f"Total Samples: {len(cross_df)}")
    print(f"Shared Classes: CBB (0), CMD (3), Healthy (4)")
    print("=" * 65)

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Cross Eval"):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    metrics = calculate_metrics(all_targets, all_preds, y_probs=all_probs, class_indices=CROSS_CLASSES_SUBSET)
    print_metrics_table(metrics, class_names=CLASS_NAMES)

    return metrics
