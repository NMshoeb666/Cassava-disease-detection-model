import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from .config import (
    MAIN_FOLDS_CSV, NUM_CLASSES, CLASS_NAMES, BATCH_SIZE, NUM_EPOCHS,
    LEARNING_RATE, WEIGHT_DECAY, CB_BETA, FOCAL_GAMMA, DEVICE, OUTPUT_DIR
)
from .dataset import get_dataloaders
from .losses import ClassBalancedFocalLoss, WeightedCrossEntropyLoss
from .models.egwt import EGWT
from .models.egwt_improved import EGWT_Improved
from .models.egwt_pretrained import EGWT_Pretrained
from .augmentations import MixUpCutMixAugmentor, soft_cross_entropy
from .metrics import calculate_metrics, print_metrics_table

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None, augmentor=None):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        soft_labels = None
        if augmentor is not None:
            images, soft_labels = augmentor(images, labels)
            
        if scaler is not None:
            with autocast():
                logits = model(images)
                if soft_labels is not None:
                    loss = soft_cross_entropy(logits, soft_labels, class_weights=augmentor.class_weights)
                else:
                    loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            if soft_labels is not None:
                loss = soft_cross_entropy(logits, soft_labels, class_weights=augmentor.class_weights)
            else:
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = calculate_metrics(all_targets, all_preds)
    return epoch_loss, metrics

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []
    
    pbar = tqdm(dataloader, desc="Validation", leave=False)
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            
            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    val_loss = running_loss / len(dataloader.dataset)
    metrics = calculate_metrics(all_targets, all_preds, y_probs=np.array(all_probs))
    return val_loss, metrics, np.array(all_preds), np.array(all_probs)

def run_cross_validation(
    folds_csv=MAIN_FOLDS_CSV,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    use_cb_loss=True,
    use_balanced_sampler=True,
    specific_fold=None,
    model_type="baseline",
    attention_mode="spatial_se",
    use_mixup=False
):
    df_folds = pd.read_csv(folds_csv)
    fold_list = [specific_fold] if specific_fold is not None else list(range(5))
    
    oof_predictions = np.zeros((len(df_folds), NUM_CLASSES))
    fold_metrics = []
    
    loss_name = "ClassBalancedFocalLoss" if use_cb_loss else "Standard CrossEntropy (Baseline)"
    sampler_name = "WeightedRandomSampler" if use_balanced_sampler else "RandomSampler (Baseline)"
    aug_name = "CutMix/MixUp (Minority-Weighted)" if use_mixup else "None"
    
    print("=" * 65)
    print("STARTING STRATIFIED 5-FOLD CROSS-VALIDATION")
    print(f"Model: {model_type.upper()}" + (f" (Attention: {attention_mode})" if model_type == "improved" else ""))
    print(f"Loss: {loss_name}")
    print(f"Sampler: {sampler_name}")
    print(f"Batch Augmentation: {aug_name}")
    print(f"Device: {DEVICE}")
    print(f"Target Folds: {fold_list}")
    print(f"Epochs per Fold: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}")
    print("=" * 65)
    
    for fold in fold_list:
        print(f"\n>>> STARTING FOLD {fold} / 4 <<<")
        train_loader, val_loader = get_dataloaders(
            df_folds, val_fold=fold, batch_size=batch_size,
            use_balanced_sampler=use_balanced_sampler
        )
        
        train_df = df_folds[df_folds['fold'] != fold]
        samples_per_class = train_df['label'].value_counts().sort_index().values
        
        if model_type == "pretrained" or model_type == "egwt_plus":
            model = EGWT_Pretrained(num_classes=NUM_CLASSES, pretrained=True, attention_mode=attention_mode).to(DEVICE)
        elif model_type == "improved":
            model = EGWT_Improved(num_classes=NUM_CLASSES, attention_mode=attention_mode).to(DEVICE)
        else:
            model = EGWT(num_classes=NUM_CLASSES).to(DEVICE)
            
        is_cuda = (DEVICE.type == 'cuda') if isinstance(DEVICE, torch.device) else (str(DEVICE) == 'cuda')
        scaler = GradScaler() if is_cuda else None
        
        augmentor = None
        if use_mixup:
            counts = np.maximum(samples_per_class, 1).astype(np.float32)
            inv_weights = torch.tensor(1.0 / counts, dtype=torch.float32)
            inv_weights = inv_weights / inv_weights.sum() * len(counts)
            augmentor = MixUpCutMixAugmentor(
                num_classes=NUM_CLASSES,
                class_weights=inv_weights,
                apply_prob=0.5,
                mixup_prob=0.5
            )
        
        if use_cb_loss:
            criterion = ClassBalancedFocalLoss(samples_per_class, beta=CB_BETA, gamma=FOCAL_GAMMA)
        else:
            criterion = torch.nn.CrossEntropyLoss()
            
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        best_f1 = 0.0
        base_tag = "cb_focal" if use_cb_loss else "baseline"
        if model_type == "improved":
            ckpt_tag = f"egwt_improved_{attention_mode}_{base_tag}"
        else:
            ckpt_tag = f"egwt_{base_tag}"
        if use_mixup:
            ckpt_tag += "_mixup"
            
        best_checkpoint = os.path.join(OUTPUT_DIR, f"{ckpt_tag}_fold{fold}.pth")
        fallback_checkpoint = os.path.join(OUTPUT_DIR, f"egwt_best_fold{fold}.pth")
        
        for epoch in range(1, epochs + 1):
            train_loss, train_m = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler, augmentor=augmentor)
            val_loss, val_m, val_preds, val_probs = validate(model, val_loader, criterion, DEVICE)
            scheduler.step()
            
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_m['top1_accuracy']*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_m['top1_accuracy']*100:.2f}% Macro-F1: {val_m['macro_f1']*100:.2f}%")
                  
            if val_m['macro_f1'] > best_f1:
                best_f1 = val_m['macro_f1']
                torch.save(model.state_dict(), best_checkpoint)
                torch.save(model.state_dict(), fallback_checkpoint)
                print(f"  --> Saved new best checkpoint to {best_checkpoint} (F1: {best_f1*100:.2f}%)")
                
        # Load best weights
        if os.path.exists(best_checkpoint):
            model.load_state_dict(torch.load(best_checkpoint, map_location=DEVICE))
        _, best_val_m, val_preds, val_probs = validate(model, val_loader, criterion, DEVICE)
        val_indices = df_folds[df_folds['fold'] == fold].index.values
        oof_predictions[val_indices] = val_probs
        fold_metrics.append(best_val_m)
        print(f"\n>>> FOLD {fold} FINISHED! Best Validation Macro F1: {best_f1*100:.2f}% <<<\n")
        print_metrics_table(best_val_m, class_names=CLASS_NAMES)
        
    if specific_fold is None:
        print("\n" + "=" * 65)
        print("OVERALL 5-FOLD OUT-OF-FOLD (OOF) PERFORMANCE")
        print("=" * 65)
        oof_targets = df_folds['label'].values
        oof_preds = np.argmax(oof_predictions, axis=1)
        oof_m = calculate_metrics(oof_targets, oof_preds, y_probs=oof_predictions)
        print_metrics_table(oof_m, class_names=CLASS_NAMES)
        
        f1_scores = [m['macro_f1'] for m in fold_metrics]
        acc_scores = [m['top1_accuracy'] for m in fold_metrics]
        print(f"5-Fold Mean Top-1 Accuracy: {np.mean(acc_scores)*100:.2f}% +/- {np.std(acc_scores)*100:.2f}%")
        print(f"5-Fold Mean Macro F1-Score: {np.mean(f1_scores)*100:.2f}% +/- {np.std(f1_scores)*100:.2f}%")
