# MH-F Stroke Classification (EfficientNetB0 + CapsuleNet Hybrid)

**Repository / Notebook:** `mhf-stroke.ipynb`

**Purpose**
This README documents the stroke classification project implemented in `mhf-stroke.ipynb`. The core model is a hybrid architecture that combines a pretrained **EfficientNetB0** backbone (feature extractor) with a **Capsule Network (CapsuleNet)** classifier head. The hybrid design leverages EfficientNet's strong image-level feature extraction and CapsuleNet's equivariant representation and routing-by-agreement for robust class-specific instantiation vectors — useful in medical imaging where spatial relationships matter.

---

## Table of contents

1. Overview
2. Dataset and labels
3. Preprocessing & augmentation
4. Hybrid model: EfficientNetB0 + CapsuleNet
5. Training setup & losses
6. Evaluation & metrics
7. Inference
8. File / Notebook structure
9. Dependencies & environment
10. How to run
11. Hyperparameters (typical)
12. Tips, caveats & reproducibility
13. License & citation

---

## 1. Overview

This project performs automated stroke classification from brain scans (2D slices or 2.5D/stacked slices). The pipeline:

* Load and label dataset (three classes: Normal, Ischemic, Hemorrhagic)
* Optional grouping to binary risk labels (Low Risk vs High Risk)
* Preprocess images (resizing, intensity normalization, windowing for CT)
* Train a hybrid model: EfficientNetB0 backbone → CapsuleNet head
* Evaluate with confusion matrices, class-wise metrics, ROC/AUC
* Export model and inference routines

The hybrid architecture improves robustness to viewpoint and small spatial variations, while still benefiting from strong pretrained features.

## 2. Dataset and labels

Expected directory layout (example):

```
data/
  ├─ train/
  │   ├─ normal/
  │   ├─ ischemic/
  │   └─ hemorrhagic/
  ├─ val/
  └─ test/
```

Label mapping used in the notebook:

* `0` — Normal
* `1` — Ischemic Stroke
* `2` — Hemorrhagic Stroke

Binary risk grouping (for triage):

* Low Risk  = Normal
* High Risk = Ischemic or Hemorrhagic

## 3. Preprocessing & augmentation

Key preprocessing steps implemented in the notebook:

* Resize / center-crop to target resolution (recommended 224×224 for EfficientNetB0)
* Intensity normalization: per-image z-score or use ImageNet mean/std if fine-tuning pretrained weights
* CT-specific windowing (brain window) and clipping where appropriate

Augmentations (use medically plausible transforms only):

* Small rotations (±10–15°), flips (if anatomically valid), random crop/scale
* Brightness/contrast jitter (careful on CT intensities)
* Avoid extreme elastic transforms that distort anatomy

## 4. Hybrid model: EfficientNetB0 + CapsuleNet

The notebook implements a two-stage hybrid model:

**Backbone — EfficientNetB0**

* Pretrained on ImageNet (via `timm` or `torchvision` implementations). Remove the final classification head and use the convolutional feature map as embedding.
* Output: a feature map tensor (e.g., `B x C x H x W`) which is spatially rich and compact.

**CapsuleNet Head**

* A small set of convolutional primary capsules converts the backbone feature map into capsule pose vectors.
* Followed by `DigitCaps` (or ClassCaps) fully connected capsule layer using routing-by-agreement (typically 3 routing iterations).
* Class probabilities are the lengths (L2 norms) of the output capsule vectors.

**Typical architecture flow (conceptual):**

```
Input image -> EfficientNetB0 backbone (remove classifier) -> conv projection -> PrimaryCapsules -> DigitCaps (routing) -> class vector lengths -> Predicted class
```

**Design choices implemented in notebook**

* Feature projection: a 1×1 conv or small conv block to map EfficientNet features into the channel-dimension expected by PrimaryCaps.
* PrimaryCaps: `NumCapsules = Np`, `CapsuleDim = Dp`, implemented as convolutional capsules (each 'capsule' is a Dp‑dim vector at each spatial location).
* DigitCaps: `NumClasses = 3`, `CapsuleDim = Dc` (typical Dc=16), dynamic routing (3 iterations).
* Reconstruction decoder (optional): use the DigitCaps outputs to reconstruct input for a regularization loss (helps capsules learn better instantiation parameters).

**Why this hybrid?**

* EfficientNet provides powerful hierarchical features while being parameter-efficient.
* CapsuleNet retains part-whole relationships and equivariance — useful to capture localized lesion patterns and their relative configurations.

## 5. Training setup & losses

**Loss functions**

* **Margin loss (Capsule)** — the canonical capsule loss for multi-class training (margin loss with `m+ = 0.9`, `m- = 0.1` and down-weight factor `λ` for absent classes). This supervises the capsule lengths.
* Optionally combine with **reconstruction loss** (MSE) scaled by a small factor (e.g., `0.0005`) if reconstruction decoder is used.
* If you prefer classic logits for multi-class cross-entropy, you can compute logits from capsule lengths and use `CrossEntropyLoss` — the notebook shows both patterns so you can experiment.

**Optimizer & schedule**

* Optimizer: `AdamW` or `Adam` is recommended for faster convergence on small batches.
* Initial LR: `1e-4` (reduce on plateau or use cosine scheduler)
* Weight decay: `1e-4` or `0`

**Other training details**

* Routing iterations: 3 (default)
* Batch size: tune to GPU memory — capsule routing and EfficientNet cost more memory; start with 8–32 for 2D.
* Checkpoint on best validation metric (AUC / F1 macro)

## 6. Evaluation & metrics

* Confusion matrix (per-class)
* Accuracy, precision, recall, F1 (per-class and macro/micro)
* ROC curves and AUC for binary risk grouping
* Per-patient aggregation: if you predict on slices, aggregate slice predictions to patient-level by majority vote or max probability for clinical relevance

The notebook saves per-sample predictions to CSV for external analysis.

## 7. Inference

Provided utilities (notebook cells):

* `predict_image(img_path, model)` — returns capsule lengths (class probabilities), top-1 label and optionally reconstruction output
* `predict_folder(folder_path)` — batch inference and write predictions to CSV
* `to_risk_label(pred_label)` — map `0/1/2` → `Low/High` risk

Deployment tips:

* Export only the backbone + capsule head weights; wrap preprocessing and postprocessing into a single inference function.
* For CPU deployment consider `torch.jit.trace` but validate behavior of dynamic routing (some routing implementations may not be jittable). If JIT fails, export PyTorch weights and run inference with a small wrapper.

## 8. File / Notebook structure

Main notebook: `mhf-stroke.ipynb` — contains the full pipeline and the reference implementation of the hybrid model.
Suggested modular files to extract if you refactor:

* `models/efficientnet_capsule.py` — contains `EfficientNetCapsule` class that wires EfficientNetB0 backbone to capsule head
* `models/capsule_layers.py` — PrimaryCaps, DigitCaps, routing implementation
* `datasets.py` — dataset loaders, transforms, and patient-level grouping logic
* `train.py` — training loop, metrics, and checkpointing
* `infer.py` — inference utilities and batch predictors
* `utils.py` — plotting, metrics, helpers

## 9. Dependencies & environment

Recommended environment (example):

* Python 3.9+
* PyTorch 1.12+ / 2.x
* torchvision
* timm (for easy EfficientNetB0 access) — `pip install timm`
* numpy, pandas, scikit-learn
* matplotlib, seaborn
* albumentations or torchvision.transforms
* tqdm
* nibabel or pydicom (if using NIfTI / DICOM)

Install example:

```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision timm numpy pandas scikit-learn matplotlib albumentations tqdm nibabel pydicom
```

## 10. How to run

Open and run `mhf-stroke.ipynb` in Jupyter or Colab.

Notebook quick-start:

1. Set dataset paths at the top of the notebook.
2. Choose `MODE = '2D'` and ensure `MODEL_HEAD = 'capsule'` and `BACKBONE = 'efficientnet_b0'` in the config cell.
3. Configure `TRAIN_PARAMS` (epochs, lr, batch_size).
4. Run preprocessing cells, then training cells.
5. Run evaluation and inference cells.

If you extract scripts, example commands:

```bash
python train.py --config configs/efficientnet_capsule.yaml
python infer.py --weights weights/best_model.pth --input data/test
```

## 11. Hyperparameters (typical)

* LR (AdamW): `1e-4`
* Batch size (2D): `8–32` (adjust to GPU)
* Epochs: `30–100` with early stopping
* Routing iterations: `3`
* PrimaryCaps dim (Dp): `8`
* DigitCaps dim (Dc): `16`
* Reconstruction scale: `0.0005` (if using reconstruction decoder)

## 12. Tips, caveats & reproducibility

* Capsule networks are sensitive to initialization & batch size — keep seeds consistent and run multiple seeds.
* Avoid unphysiological augmentations.
* When training on slices, consider patient-level splits to avoid data leakage.
* If routing is slow, minimize routing iterations or optimize capsule implementation (vectorized matrix ops).
* If you need to produce calibration-friendly probabilities, calibrate capsule lengths via temperature scaling.

## 13. License & citation

This notebook is intended for research use. Obtain necessary approvals for working with medical images.

If you use this work in publications, cite the notebook and reference the hybrid architecture (EfficientNetB0 backbone + CapsuleNet head).
