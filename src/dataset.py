import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .samplers import get_balanced_sampler
from .config import MAIN_IMAGES_DIR, CROSS_DATA_DIR, IMG_SIZE, BATCH_SIZE, NUM_WORKERS

class CassavaDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_cross=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_cross = is_cross
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.is_cross:
            img_path = os.path.join(self.img_dir, row['image_path'])
        else:
            img_path = os.path.join(self.img_dir, row['image_id'])
            
        image = Image.open(img_path).convert('RGB')
        label = int(row['label'])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms(img_size=IMG_SIZE):
    """
    Train  : augmentation applied (augmentation on training folds ONLY).
    Val    : deterministic resize → center-crop — no augmentation, no data leakage.
    Both   : 384×384, ImageNet mean/std normalisation.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
    val_resize    = int(img_size / 0.875)   # 384 → 438: resize slightly larger before crop

    train_transform = transforms.Compose([
        # RandomResizedCrop handles scale diversity — no pre-resize needed
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.25), ratio=(0.3, 3.3), value='random'),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(val_resize),          # 438
        transforms.CenterCrop(img_size),        # 384 — deterministic, no augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_transform, val_transform


def get_dataloaders(folds_df, val_fold=0, batch_size=BATCH_SIZE, use_balanced_sampler=True, num_workers=NUM_WORKERS):
    train_df = folds_df[folds_df['fold'] != val_fold].copy().reset_index(drop=True)
    val_df = folds_df[folds_df['fold'] == val_fold].copy().reset_index(drop=True)
    
    train_transform, val_transform = get_transforms()
    
    train_ds = CassavaDataset(train_df, MAIN_IMAGES_DIR, transform=train_transform)
    val_ds = CassavaDataset(val_df, MAIN_IMAGES_DIR, transform=val_transform)
    
    sampler = None
    shuffle = True
    if use_balanced_sampler:
        sampler = get_balanced_sampler(train_df['label'].values)
        shuffle = False
        
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader

def get_cross_dataloader(cross_df, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    _, val_transform = get_transforms()
    cross_ds = CassavaDataset(cross_df, CROSS_DATA_DIR, transform=val_transform, is_cross=True)
    loader = DataLoader(
        cross_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    return loader
