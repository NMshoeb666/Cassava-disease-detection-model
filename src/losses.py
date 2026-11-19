import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss (Lin et al., ICCV 2017).
    Down-weights easy well-classified examples (e.g. CMD), focusing gradients on hard examples.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        log_prob = F.log_softmax(logits, dim=-1)
        prob = torch.exp(log_prob)
        
        target_log_prob = log_prob.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        target_prob = prob.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        
        focal_weight = torch.pow(1.0 - target_prob, self.gamma)
        loss = -focal_weight * target_log_prob
        
        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_weight = self.alpha.gather(dim=0, index=targets)
            loss = alpha_weight * loss
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss (Cui et al., CVPR 2019).
    Replaces naive SMOTE over-sampling with effective-number weighting:
        E_n = (1 - beta^n) / (1 - beta)
        weight_c = (1 - beta) / (1 - beta^{n_c})
    """
    def __init__(self, samples_per_class, beta=0.999, gamma=2.0):
        super().__init__()
        samples_per_class = np.array(samples_per_class, dtype=np.float32)
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(samples_per_class)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.focal_loss = FocalLoss(alpha=self.weights, gamma=gamma, reduction='mean')

    def forward(self, logits, targets):
        if self.weights.device != logits.device:
            self.focal_loss.alpha = self.focal_loss.alpha.to(logits.device)
        return self.focal_loss(logits, targets)

class WeightedCrossEntropyLoss(nn.Module):
    """Inverse-frequency weighted cross-entropy baseline."""
    def __init__(self, samples_per_class):
        super().__init__()
        counts = np.array(samples_per_class, dtype=np.float32)
        weights = 1.0 / counts
        weights = weights / np.sum(weights) * len(counts)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, logits, targets):
        if self.weights.device != logits.device:
            self.weights = self.weights.to(logits.device)
        return F.cross_entropy(logits, targets, weight=self.weights)
