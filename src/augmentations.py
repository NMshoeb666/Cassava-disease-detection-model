"""
augmentations.py — Batch-level augmentations for cassava training.

Implements MixUp and CutMix with optional minority-class-weighted partner
sampling, so rare classes (CBB, Healthy) appear more often as mix partners.

Both produce soft (mixed) one-hot labels, which are consumed by
soft_cross_entropy() instead of the standard criterion.

Typical usage in train_one_epoch:
    augmentor = MixUpCutMixAugmentor(num_classes=5, class_weights=inv_freq_weights)
    ...
    images, soft_labels = augmentor(images, labels)   # may return None for soft_labels
    if soft_labels is not None:
        loss = soft_cross_entropy(logits, soft_labels)
    else:
        loss = criterion(logits, labels)
"""

import torch
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Soft-label cross-entropy (used when MixUp/CutMix is active)
# ─────────────────────────────────────────────────────────────────────

def soft_cross_entropy(logits, soft_labels, class_weights=None):
    """
    Cross-entropy loss that accepts soft (mixed) labels instead of integer targets.

    If class_weights are provided (torch tensor [num_classes]), minority-class
    contributions are upweighted within the mixed label before computing loss.
    This gives extra gradient weight to minority class features in every mixed batch.

    Args:
        logits        : [B, C]  raw model outputs
        soft_labels   : [B, C]  mixed one-hot labels, values in [0, 1], sum to 1
        class_weights : [C]     per-class weights (e.g. inverse frequency), optional

    Returns:
        scalar loss
    """
    log_probs = F.log_softmax(logits, dim=-1)                   # [B, C]

    if class_weights is not None:
        if class_weights.device != logits.device:
            class_weights = class_weights.to(logits.device)
        # Re-weight each class contribution, then re-normalise per sample
        weighted = soft_labels * class_weights.unsqueeze(0)     # [B, C]
        weighted = weighted / weighted.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    else:
        weighted = soft_labels

    loss = -(weighted * log_probs).sum(dim=-1).mean()
    return loss


# ─────────────────────────────────────────────────────────────────────
# MixUp  (Zhang et al., ICLR 2018)
# ─────────────────────────────────────────────────────────────────────

class MixUp:
    """
    Linearly blends two images and their labels.

        mixed_x  = λ · x_i  + (1-λ) · x_j
        mixed_y  = λ · y_i  + (1-λ) · y_j       (soft one-hot)

    λ ~ Beta(alpha, alpha).  alpha=0.4 keeps the dominant image contributing
    at least ~60% so features remain recognisable.

    Args:
        alpha (float): Beta distribution parameter. Typical range 0.2–1.0.
    """
    def __init__(self, alpha=0.4):
        self.alpha = alpha

    def __call__(self, images, labels, num_classes, partner_idx=None):
        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1.0 - lam)                      # dominant image ≥ 50%

        B = images.size(0)
        idx = partner_idx if partner_idx is not None else torch.randperm(B, device=images.device)

        mixed = lam * images + (1.0 - lam) * images[idx]

        y_a = F.one_hot(labels,      num_classes).float()
        y_b = F.one_hot(labels[idx], num_classes).float()
        soft_labels = lam * y_a + (1.0 - lam) * y_b

        return mixed, soft_labels


# ─────────────────────────────────────────────────────────────────────
# CutMix  (Yun et al., ICCV 2019)
# ─────────────────────────────────────────────────────────────────────

class CutMix:
    """
    Cuts a random rectangular patch from image j and pastes it onto image i.
    Labels are mixed proportionally to the pixel area of the pasted patch.

        λ ~ Beta(alpha, alpha) → controls patch size
        actual λ  = 1 − patch_area / total_area  (re-computed after clipping)

    Args:
        alpha (float): Beta parameter. 1.0 = uniform patch sizes.
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    @staticmethod
    def _rand_bbox(H, W, lam):
        """Returns (x1, y1, x2, y2) for the cut region."""
        cut_h = int(H * np.sqrt(1.0 - lam))
        cut_w = int(W * np.sqrt(1.0 - lam))
        cy = np.random.randint(H)
        cx = np.random.randint(W)
        y1 = max(0, cy - cut_h // 2)
        y2 = min(H, cy + cut_h // 2)
        x1 = max(0, cx - cut_w // 2)
        x2 = min(W, cx + cut_w // 2)
        return x1, y1, x2, y2

    def __call__(self, images, labels, num_classes, partner_idx=None):
        lam_init = np.random.beta(self.alpha, self.alpha)
        B, C, H, W = images.shape
        idx = partner_idx if partner_idx is not None else torch.randperm(B, device=images.device)

        x1, y1, x2, y2 = self._rand_bbox(H, W, lam_init)

        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]

        # Re-compute λ based on actual pasted area
        patch_area = (y2 - y1) * (x2 - x1)
        lam = 1.0 - patch_area / (H * W)

        y_a = F.one_hot(labels,      num_classes).float()
        y_b = F.one_hot(labels[idx], num_classes).float()
        soft_labels = lam * y_a + (1.0 - lam) * y_b

        return mixed, soft_labels


# ─────────────────────────────────────────────────────────────────────
# Combined augmentor with minority-class-weighted partner sampling
# ─────────────────────────────────────────────────────────────────────

class MixUpCutMixAugmentor:
    """
    Randomly applies MixUp or CutMix with probability `apply_prob`.

    Key design — minority-class-weighted partner sampling:
        Instead of torch.randperm (uniform random mix partner),
        each sample's partner is drawn with probability proportional to
        the INVERSE frequency of its class.  This means CBB (~5.7%) and
        Healthy (~11%) images are chosen as partners ~10× more often than
        CMD (~58.7%), giving minority-class features extra representation
        in every mixed batch without changing the sampler.

    Args:
        num_classes      (int)   : number of classes
        class_weights    (Tensor): [C] inverse-frequency weights per class
                                   (from samplers or losses module)
        apply_prob       (float) : probability of applying any augmentation
        mixup_prob       (float) : given augmentation is applied, prob of MixUp
                                   vs CutMix (remainder = CutMix prob)
        mixup_alpha      (float) : MixUp Beta parameter
        cutmix_alpha     (float) : CutMix Beta parameter
    """
    def __init__(self, num_classes, class_weights=None,
                 apply_prob=0.5, mixup_prob=0.5,
                 mixup_alpha=0.4, cutmix_alpha=1.0):
        self.num_classes   = num_classes
        self.class_weights = class_weights       # [C] tensor, or None
        self.apply_prob    = apply_prob
        self.mixup_prob    = mixup_prob
        self.mixup         = MixUp(alpha=mixup_alpha)
        self.cutmix        = CutMix(alpha=cutmix_alpha)

    def _weighted_partner_idx(self, labels):
        """
        Sample partner indices with higher probability for minority classes.
        labels: [B] integer class labels for the current batch.
        Returns: [B] index tensor (with replacement).
        """
        if self.class_weights is None:
            return torch.randperm(labels.size(0), device=labels.device)

        cw = self.class_weights.to(labels.device)
        # Weight of each sample = inverse freq of its class
        sample_w = cw[labels].float()                   # [B]
        partner  = torch.multinomial(sample_w, num_samples=labels.size(0),
                                     replacement=True)  # [B]
        return partner

    def __call__(self, images, labels):
        """
        Args:
            images : [B, 3, H, W]  normalised batch
            labels : [B]            integer class labels

        Returns:
            (mixed_images, soft_labels) if augmentation is applied
            (images, None)              otherwise  — caller uses normal criterion
        """
        if np.random.random() > self.apply_prob:
            return images, None                         # no aug this batch

        partner_idx = self._weighted_partner_idx(labels)

        if np.random.random() < self.mixup_prob:
            return self.mixup(images, labels, self.num_classes, partner_idx)
        else:
            return self.cutmix(images, labels, self.num_classes, partner_idx)
