import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler

def get_balanced_sampler(labels):
    """
    Creates a WeightedRandomSampler so each class has equal sampling probability per batch.
    """
    labels = np.array(labels)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler
