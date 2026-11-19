# EGWT-Net++: Hybrid Attentive Group-Wise Transformer for Cassava Leaf Disease Classification

PyTorch implementation and experimental framework for **EGWT-Net++**, an enhanced convolutional group-wise vision transformer designed for robust crop disease diagnosis under complex field environments.

This work addresses three persistent challenges in agricultural computer vision:
1. **Background Clutter & Weed Interference**: Damped via an in-line $\text{Spatial} \rightarrow \text{Squeeze-and-Excitation}$ (SE) hybrid attention gate.
2. **Visually Overlapping Pathologies**: Resolved by fusing fine-grained vein patterns ($S_1$), lesion margin geometries ($S_2$), and abstract category context ($S_3$).
3. **Cross-Domain Generalization**: Validated zero-shot on independent field imagery (Mendeley India dataset) without target fine-tuning.

---

## Performance Summary

Evaluated across **26,220 images** under a leak-free Stratified 5-Fold Cross-Validation protocol:

| Metric | Base Paper (Feng et al., 2024) | Ours (EGWT-Net++) | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Top-1 Accuracy** | 84.29% | **87.81%** | **+3.52%** |
| **Top-5 Accuracy** | — | **100.00%** | — |
| **Macro F1-Score** | ~71.00% | **78.80%** | **+7.80%** |
| **Healthy F1 (Minority)** | 65.56% | **72.43%** | **+6.87%** |
| **Parameters** | 23.04M | **7.29M** | **68% Fewer Parameters** |
| **Cross-Dataset (India)** | Not Evaluated | **73.68% (Zero-Shot)** | Out-of-Distribution Validated |

---

## Architectural Highlights

* **Pretrained CNN Stem**: Utilizes early convolutional representations (EfficientNet-B0) to preserve low-level botanical edge and surface primitives.
* **Spatial $\rightarrow$ SE Gate**: Applies spatial attenuation first:
  $$\hat{F} = \text{SE}(F \odot \mathcal{A}_s(F))$$
  Damps non-target background pixels (soil, shadows, weeds) before channel-wise reweighting is computed.
* **Group-Wise Multi-Head Attention (G-MHA)**: Divides channel dimensions into shared parameter groups, computed via memory-efficient scaled dot-product attention (`F.scaled_dot_product_attention`) with gradient checkpointing.
* **Multi-Scale Feature Fusion**: Concatenates representations across three hierarchical stages ($24\text{-d} + 256\text{-d} + 512\text{-d} = 792\text{-d}$) before projection to the classification head.

---

## Repository Structure

```text
├── data/
│   └── processed/
│       ├── cassava_5folds.csv         # Stratified 5-fold cross-validation splits (26,220 samples)
│       └── cassava_cross_meta.csv     # Metadata for Mendeley India test dataset (228 samples)
├── notebooks/
│   └── cassava_training_colab.ipynb  # End-to-end execution notebook (Colab GPU)
├── outputs/
│   ├── fold0_results.txt             # In-domain Fold 0 validation metrics
│   ├── cross_dataset_india_results.txt # Out-of-distribution evaluation results
│   └── gradcam_output.jpg            # Generated Grad-CAM saliency visualization
├── src/
│   ├── models/
│   │   ├── egwt_pretrained.py        # Proposed EGWT-Net++ architecture
│   │   ├── egwt_improved.py          # Scratch EGWT with hybrid attention & fusion
│   │   └── egwt.py                   # Baseline EGWT implementation (Feng et al.)
│   ├── augmentations.py              # Minority-weighted CutMix/MixUp & soft cross-entropy
│   ├── dataset.py                    # Dataset classes and augmentation pipelines
│   ├── evaluate_cross.py             # Zero-shot cross-dataset evaluation engine
│   ├── gradcam.py                    # Grad-CAM heatmap generation module
│   ├── losses.py                     # Class-Balanced Focal Loss implementation
│   ├── metrics.py                    # Top-1/5, Macro F1, Precision, Recall calculators
│   ├── samplers.py                   # Inverse-frequency weighted random sampler
│   └── train_cv.py                   # Stratified cross-validation training pipeline
├── run_experiment.py                 # Main CLI runner for training and evaluation
├── run_masking_test.py               # Attention pixel masking quantification experiment
├── run_sanity_check.py               # 6-stage end-to-end pipeline verification test
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/cassava-egwt-net.git
   %cd cassava-egwt-net
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify pipeline**:
   ```bash
   python run_sanity_check.py
   ```

---

## Datasets

* **Primary Dataset (Dataset A)**: Cassava Leaf Disease Dataset collected by Makerere University AI Lab and NaCRRI in Uganda (26,220 images across 5 classes: CBB, CBSD, CGM, CMD, Healthy).
* **Cross-Validation Dataset (Dataset B)**: Mendeley Cassava Leaf Disease Dataset collected independently in Tamil Nadu, India (228 images across CBB, CMD, and Healthy).

---


## Usage

### 1. Training (Stratified Cross-Validation)
Train proposed model on Fold 0:
```bash
python run_experiment.py \
  --mode train \
  --model pretrained \
  --attention_mode spatial_se \
  --fold 0 \
  --epochs 5 \
  --batch_size 16 \
  --lr 3e-4 \
  --loss ce \
  --sampler random
```

To run full 5-fold cross-validation, omit the `--fold` argument.

### 2. Cross-Dataset Generalization (Mendeley India Dataset)
Evaluate a trained checkpoint zero-shot on unseen Indian field samples:
```bash
python run_experiment.py \
  --mode eval_cross \
  --model pretrained \
  --checkpoint outputs/egwt_baseline_fold0.pth
```

### 3. Attention Pixel Masking Experiment
Quantify model reliance on spatial attention by masking the top 20% most salient pixels versus random controls:
```bash
python run_masking_test.py \
  --model pretrained \
  --checkpoint outputs/egwt_baseline_fold0.pth \
  --fold 0 \
  --batch_size 16 \
  --k_frac 0.20
```

### 4. Visual Explainability (Grad-CAM)
Generate saliency heatmaps for a target leaf image:
```bash
python run_experiment.py \
  --mode gradcam \
  --model pretrained \
  --checkpoint outputs/egwt_baseline_fold0.pth \
  --image_path 'path/to/leaf_image.jpg'
```

---


## Citation & Reference

If you find this work or codebase helpful in your research, please cite:

```bibtex
@article{feng2024enhanced,
  title={Enhanced Crop Disease Detection With EfficientNet Convolutional Group-Wise Transformer},
  author={Feng, Jing and Ong, Wen Eng and Teh, Wen Chean and Zhang, Rui},
  journal={IEEE Access},
  volume={12},
  pages={44147--44162},
  year={2024},
  publisher={IEEE}
}
```
