---
name: motor-imagery-bci
description: Use this skill when working with EEG motor imagery datasets, preprocessing signals, extracting features, or building ML pipelines for BCI research.
license: MIT
compatibility: opencode
metadata:
  audience: researchers, developers
  workflow: data-processing, machine-learning
  version: 1.0.0
---

# Motor Imagery BCI Skill

This skill provides guidance for building Brain-Computer Interface (BCI) pipelines for motor imagery classification using EEG signals.

## When This Skill MUST Be Used

**ALWAYS invoke this skill for ANY of these tasks:**

- Loading or preprocessing motor imagery EEG datasets (BCI Competition IV-2a, PhysioNet)
- Implementing EEG signal processing pipelines
- Building CNN/Transformer models for EEG classification
- Creating feature extraction (CSP, band power)
- Training or evaluating motor imagery models
- Handling EEG artifacts or noise removal
- Dataset splitting or evaluation metrics

## What This Skill Provides

- Guidance on proper EEG preprocessing pipeline
- Dataset loading patterns (BCI IV-2a format)
- Model architecture recommendations
- Common pitfalls and how to avoid them
- Reproducibility best practices

## Key Concepts

### Dataset Format

BCI Competition IV-2a:
- Files: `A01T.gdf` (training), `A01E.gdf` (evaluation)
- 22 EEG channels + 3 EOG channels
- 250 Hz sampling rate
- 4 classes: left hand (769), right hand (770), feet (771), tongue (772)
- 288 trials per subject (72 per class)

### Preprocessing Pipeline

Standard motor imagery preprocessing:
1. Bandpass filter: 0.5-100 Hz
2. Notch filter: 50 Hz (line noise)
3. Re-reference: average reference
4. Baseline correction
5. Normalization (StandardScaler)

### Model Recommendations

For motor imagery classification:
- **EEGNet**: Good baseline (8-16 channels, 4 classes)
- **EEGConformer**: Transformer-based
- **CSP + SVM**: Classic approach
- **EEGEncoder**: SOTA (from paper)

### Training Strategy

- Per-subject models (recommended for prosthetics)
- 144 trials training, 144 trials test (official split)
- Early stopping (patience=30)
- Label smoothing (0.1)
- Learning rate: 0.001

## Common Mistakes to Avoid

### 1. Data Leakage
- NEVER normalize on combined train+test
- Fit scaler only on training data

### 2. Wrong Event Codes
- BCI IV-2a uses GDF codes: 769, 770, 771, 772
- Evaluation files may have different codes

### 3. Channel Mismatch
- Handle duplicate channels (EEG-0, EEG-1, etc.)
- Verify channel names match montage

### 4. Class Imbalance
- Check distribution: should be 72 trials per class
- Use stratified splits

### 5. Overfitting
- Use dropout (0.25-0.5)
- Implement early stopping
- Don't exceed 300 epochs

## Project Structure

This project uses:

```
src/
├── data/
│   └── bcic_iv_2a.py      # Dataset loader
├── preprocessing/
│   └── zuna_pipeline.py    # ZUNA preprocessing
├── models/
│   └── eegencoder.py       # Model architecture
└── training/
    └── trainer.py          # Training loop
```

## Reproducibility Rules

1. **Always set random seeds** before training
2. **Log all hyperparameters** in config
3. **Save checkpoints** at each epoch
4. **Log metrics** (accuracy, loss per epoch)

## Decision Framework

When working on BCI tasks:

1. **Loading data** -> Use `src/data/bcic_iv_2a.py`
2. **Preprocessing** -> Use `src/preprocessing/` modules
3. **Training** -> Use `src/training/trainer.py`
4. **Evaluation** -> Use accuracy, kappa metrics
5. **Reproducibility** -> Set seeds, log everything

## Out of Scope

This skill does not cover:
- Real-time inference
- Model deployment to hardware
- Medical device certification
- Data collection/recording
