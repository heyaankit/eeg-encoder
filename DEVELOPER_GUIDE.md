# Developer Guide

A comprehensive guide to understanding, using, and extending this EEG motor imagery classification system.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Quick Start](#quick-start)
3. [Dataset Preparation](#dataset-preparation)
4. [Training Pipeline](#training-pipeline)
5. [Understanding the Code](#understanding-the-code)
6. [Key Decisions & Rationale](#key-decisions--rationale)
7. [Extending the Project](#extending-the-project)
8. [Troubleshooting](#troubleshooting)

---

## Project Overview

### What This Project Does

This project implements a deep learning system for classifying EEG signals into four motor imagery classes:
- Left hand
- Right hand
- Feet
- Tongue

### Why It Matters

Motor imagery classification is fundamental to brain-computer interfaces (BCIs) for prosthetics. The system learns to decode mental commands from EEG signals, enabling paralyzed patients to control prosthetic limbs.

### Architecture Summary

```
EEGEncoder (Paper: Scientific Reports 2025)
├── ConvBlock          # EEGNet-style front-end
├── DSTS Branches (5) # Temporal Convolutional Network + Transformer
├── Domain Classifier  # Multi-subject learning
└── Task Classifier   # 4-class MI classification
```

**Total Parameters**: ~172K (lightweight for real-time inference)

---

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/heyaankit/eeg-encoder.git
cd eeg-encoder
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
# Download BCI Competition IV-2a from:
# http://bnci-horizon-2020.eu/database/data-sets

# Place files in:
src/data/BCICIV_2a_gdf/
# Expected: A01T.gdf, A01E.gdf, A02T.gdf, ... A09E.gdf
```

### 3. Train a Model

**Option A: Per-Subject Training** (for production with individual users)
```bash
python train_complete.py --subject A01 --epochs 300
```

**Option B: Multi-Subject Training** (recommended for generalization)
```bash
python train_dat.py --epochs 80
```

### 4. Evaluate

The training script automatically evaluates on each subject after training completes.

---

## Dataset Preparation

### BCI Competition IV-2a Dataset

| Property | Value |
|----------|-------|
| Subjects | 9 (A01-A09) |
| Trials | 288 per subject (72 per class) |
| Channels | 22 EEG + 3 EOG |
| Sampling | 250 Hz |
| Duration | 4.5 seconds per trial |
| Classes | 4 (left hand, right hand, feet, tongue) |

### Important Discovery

**The dataset is already pre-filtered** by the competition organizers:
- Bandpass: 0.5-100 Hz
- Notch: 50 Hz

**We do NOT apply additional filtering** because:
1. Deep learning models learn frequency selection internally
2. Additional bandpass filtering was found to reduce performance (92.44% → 86.84%)
3. The model can extract relevant features from the broader frequency range

### Data Format

Each subject has two files:
- `A01T.gdf` - Training session (288 trials)
- `A01E.gdf` - Evaluation session (different event codes, not used)

We use only the training session data.

---

## Training Pipeline

### Two Approaches

#### 1. Per-Subject Training (`train_complete.py`)

Trains a separate model for each subject.

**Pros**:
- Optimized for individual user
- Better accuracy per subject

**Cons**:
- Requires more data per user
- Not suitable for quick calibration

```bash
python train_complete.py --subject A01 --epochs 300
```

#### 2. Domain Adversarial Training (`train_dat.py`)

Trains one model on all subjects together.

**Pros**:
- Learns subject-invariant features
- Better generalization
- Works with limited per-subject data

**Cons**:
- May underfit individual differences

```bash
python train_dat.py --epochs 80
```

**This is our recommended approach** for prosthetics because:
- New patients need minimal calibration
- Model generalizes across individuals
- Achieves 92.44% per-subject accuracy

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning Rate | 0.001 | Standard for Adam optimizer |
| Batch Size | 32 | Balance between GPU utilization and gradient quality |
| Epochs | 80-300 | Early stopping prevents overfitting |
| Label Smoothing | 0.1 | Prevents overconfident predictions |
| Weight Decay | 0.0001 | Light L2 regularization |

### Augmentation Strategy

We apply augmentation to increase effective training data:

| Technique | Parameter | Why |
|----------|-----------|-----|
| Time-shift | ±40 samples | EEG signals vary in timing |
| Channel dropout | 30% | Electrodes may fail |
| Gaussian noise | σ=0.2 | Real EEG has noise |
| Scaling | 0.8-1.2x | Signal amplitude varies |
| MixUp | α=0.4 | Regularization via interpolation |

**Augmentation ratio**: 2x (each original sample becomes 2 augmented samples)

---

## Understanding the Code

### File Structure

```
src/
├── data/
│   └── bcic_iv_2a.py           # Raw GDF file loading
├── preprocessing/
│   └── motor_imagery_pipeline.py # Epoch extraction & preprocessing
├── models/
│   ├── eegencoder.py           # EEGEncoder architecture
│   └── domain_adversarial.py   # DAT module
├── augmentation/
│   └── augmentations.py        # Augmentation transforms
└── training/
    └── trainer.py             # Training loop
```

### Key Classes

#### `MotorImageryPreprocessor`

Handles:
1. Loading GDF files
2. Event detection (codes 769-772)
3. Epoch extraction (0 to 4.5s)
4. Channel selection (EEG only)
5. Standardization

```python
preprocessor = MotorImageryPreprocessor(
    data_dir='src/data/BCICIV_2a_gdf',
    filter_alpha_beta=False  # Dataset already pre-filtered
)
X, y, metadata = preprocessor.load_and_preprocess('A01')
```

#### `EEGAugmentor`

Applies random augmentations during training:

```python
augmentor = EEAAugmentor()
X_aug, y_aug = augmentor.augment(X, y)
```

#### `create_eegencoder`

Factory function for model creation:

```python
model = create_eegencoder(
    n_channels=25,
    n_times=1126,
    n_classes=4,
    n_branches=5,       # Parallel DSTS branches
    hidden_channels=16  # Model capacity
)
```

### Data Flow

```
GDF Files → Raw EEG → Epochs → Standardization → Augmentation → Model → Predictions
```

---

## Key Decisions & Rationale

### 1. Why EEGEncoder?

| Alternative | Why We Chose EEGEncoder |
|------------|------------------------|
| EEGNet | Good baseline but less capacity |
| CSP + SVM | Classic but requires feature engineering |
| DeepCNN | Less efficient than transformer-based |
| **EEGEncoder** | **SOTA on BCI IV-2a (86.46%), end-to-end, ~166K params** |

Reference: "EEGEncoder: Advancing BCI with Transformer-Based Motor Imagery Classification" (Scientific Reports 2025)

### 2. Why No Additional Filtering?

We experimented with adding 8-32Hz bandpass filtering:

| Configuration | Per-Subject Accuracy |
|--------------|---------------------|
| With 8-32Hz bandpass | 86.84% |
| Without (just standardization) | 92.44% |

**Reason**: The CNN learns frequency selection automatically. Additional filtering removes useful information.

### 3. Why Domain Adversarial Training?

For prosthetics, we need:
- Quick calibration (minimal data from new patient)
- Generalization across patients
- Robustness to signal quality variations

DAT addresses these by learning **subject-invariant features** - the model can't tell which subject the data came from, so it must learn the underlying motor imagery patterns.

### 4. Why Multi-Task Architecture?

Our final model has two classification heads:
1. **Task head**: Predicts motor imagery class
2. **Domain head**: Predicts subject ID

Even without adversarial training, this provides implicit regularization - the feature extractor must produce representations useful for both tasks.

---

## Extending the Project

### Adding a New Model

1. Create new model file in `src/models/`
2. Implement standard interface:

```python
class YourModel(nn.Module):
    def __init__(self, n_channels, n_times, n_classes):
        # Define architecture
        
    def forward(self, x):
        # Forward pass
        return logits
```

3. Update training scripts to accept your model

### Using Different Dataset

1. Implement dataset loader in `src/data/`
2. Ensure format matches: `(n_trials, n_channels, n_times)`
3. Labels should be 0-indexed (0, 1, 2, 3 for 4 classes)

### Hyperparameter Tuning

Key parameters to tune:

| Parameter | Range | Impact |
|-----------|-------|--------|
| `n_branches` | 3-7 | Model capacity |
| `hidden_channels` | 8-32 | Feature dimension |
| `learning_rate` | 1e-4 to 1e-2 | Training stability |
| `mixup_alpha` | 0.2-0.6 | Regularization |

---

## Troubleshooting

### Common Issues

#### "No valid motor imagery events found"

The evaluation files (A01E.gdf) use different event codes. This is normal - we only use training files.

#### CUDA out of memory

Reduce batch size:
```bash
python train_dat.py --batch-size 16
```

#### Low accuracy

1. Check data loading is correct:
```python
# Should be ~288 trials, 25 channels, 1126 timepoints
print(X.shape)  # (288, 25, 1126)
```

2. Verify augmentation is working:
```python
# Augmentation should increase data
print(X_augmented.shape)  # Should be 2x original
```

3. Check for data leakage:
```python
# Validation data should NOT be augmented
# Only training data gets augmentation
```

### Getting Help

1. Check `results/RESULTS_DAT.md` for detailed experiment history
2. Review training logs in `logs/` directory
3. Check model checkpoints in `checkpoints/`

---

## Performance Summary

| Metric | Value |
|--------|-------|
| Validation Accuracy | 79.63% |
| Per-Subject Average | 92.44% |
| Best Subject | 97.57% (A08) |
| Worst Subject | 85.07% (A04) |
| Model Parameters | 172,609 |

---

## Citation

If you use this code, please cite:

```bibtex
@article{eegencoder2025,
  title={EEGEncoder: Advancing BCI with Transformer-Based Motor Imagery Classification},
  author={Liao, W. and Liu, H. and Wang, W.},
  journal={Scientific Reports},
  year={2025}
}
```

---

## License

Apache License 2.0 - See LICENSE.md
