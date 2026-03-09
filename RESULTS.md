# EEGEncoder - BCI IV-2a Training Results

## Executive Summary

This document captures the training experiments and findings for EEGEncoder on the BCI Competition IV-2a dataset. After multiple rounds of experimentation, we achieved **71.54% average accuracy** (best: 72.71% with MixUp augmentation).

## Final Results

| Subject | Val Accuracy |
|---------|-------------|
| A01     | 68.42%      |
| A02     | 54.39%      |
| A03     | 85.96%      |
| A04     | 40.35%      |
| A05     | 85.96%      |
| A06     | 57.89%      |
| A07     | 82.46%      |
| A08     | 92.98%      |
| A09     | 75.44%      |
| **Average** | **71.54%**  |

## Target vs Actual

| Metric | Target | Actual | Gap |
|--------|--------|--------|-----|
| Subject A01 | >75% | 68.42% | -6.58% |
| Average | >80% | 71.54% | -8.46% |
| Paper Benchmark | 86.46% | 71.54% | -14.92% |

## Experiment History

### Round 1: Baseline (Before Fixes)
- **Accuracy**: ~42%
- **Issue**: Preprocessing bug - bandpass filter was zeroing out data due to filtfilt issue

### Round 2: After Preprocessing Fix
- **Accuracy**: 71.35%
- **Fix**: Used proper standard scaling instead of problematic filter
- **Config**: 5 branches, 16 hidden channels, 2x augmentation

### Round 3: MixUp Addition
- **Accuracy**: 72.71% ✓ (BEST)
- **Change**: Added MixUp augmentation (α=0.4, p=0.5)
- **Impact**: +1.36% improvement

### Round 4: Reduced Augmentation
- **Accuracy**: 64.52%
- **Change**: Lower augmentation probabilities (p=0.2-0.3)
- **Impact**: -8.19% - underfitting

### Round 5: Increased Augmentation
- **Accuracy**: 66.08%
- **Change**: 3x augmentation ratio
- **Impact**: -6.63% - too aggressive

### Round 6: Larger Model
- **Accuracy**: 71.54%
- **Change**: 24 hidden channels
- **Impact**: No significant change - diminishing returns

### Round 7: Wider Model (7 branches)
- **Accuracy**: 66.08%
- **Change**: 7 branches, 32 hidden channels
- **Impact**: -6.63% - overfitting

### Round 8: Cosine Annealing
- **Accuracy**: 71.54%
- **Change**: CosineAnnealingLR instead of ReduceLROnPlateau
- **Impact**: Similar performance

## Key Findings

### What Worked
1. **MixUp Augmentation**: +1.4% improvement, best single addition
2. **Proper preprocessing**: Fixed critical bug that caused 42% accuracy
3. **Moderate augmentation**: 2x with balanced probabilities works best

### What Didn't Help
1. **Larger models**: 24 or 32 hidden channels - no improvement, potential overfitting
2. **More branches**: 7 branches - overfitting
3. **Reduced augmentation**: Causes underfitting
4. **Aggressive augmentation**: 3x ratio - worse than 2x

### Subject Variability
Large variance between subjects suggests:
- Some subjects have inherently better signal quality
- A04, A06, A02 may need specialized preprocessing or subject-specific tuning

## Training Configuration (Final)

```python
{
    "model": "EEGEncoder",
    "n_branches": 5,
    "hidden_channels": 16,
    "parameters": 166388,
    "epochs": 300,
    "early_stopping_patience": 30,
    "batch_size": 32,
    "learning_rate": 0.001,
    "scheduler": "ReduceLROnPlateau",
    "label_smoothing": 0.1,
    "weight_decay": 0.0001,
    "augmentation": {
        "ratio": 2,
        "time_shift": {"prob": 0.5, "max_shift": 40},
        "channel_dropout": {"prob": 0.5, "ratio": 0.3},
        "noise": {"prob": 0.5, "std": 0.2},
        "scaling": {"prob": 0.5, "range": [0.8, 1.2]},
        "mixup": {"alpha": 0.4, "prob": 0.5}
    }
}
```

## Code Architecture

### Key Files
- `src/models/eegencoder.py`: EEGEncoder implementation (based on official GitHub)
- `src/preprocessing/motor_imagery_pipeline.py`: Data preprocessing
- `src/augmentation/augmentations.py`: MixUp and other augmentations
- `src/training/trainer.py`: Training loop with MixUp support
- `train_complete.py`: Main training script

### Dependencies
- PyTorch 2.0+
- MNE-Python (EEG processing)
- NumPy, SciPy, Scikit-learn

## Recommendations for Future Work

### High Priority
1. **Full ZUNA Integration**: Use ZUNA foundation model for EEG denoising before classification
2. **Domain Adversarial Training (DAT)**: Learn domain-invariant features to help difficult subjects

### Medium Priority
3. **Per-subject optimization**: Tune hyperparameters for difficult subjects (A04, A06)
4. **Ensemble methods**: Combine predictions from multiple models

### Lower Priority
5. **More training data**: Current uses only 288 trials/subject (training sessions only)
6. **Transfer learning**: Pre-train on other BCI datasets

## Checkpoints

All model checkpoints saved to `checkpoints/`:
- `best_A0X.pt` - Best validation model per subject
- `final_A0X.pt` - Final epoch model
- `history_A0X.json` - Training history

## Appendix: Training Logs

Training typically converges within 40-80 epochs with early stopping. Key observations:
- Training accuracy reaches 95%+ before early stopping
- Validation accuracy plateaus around epoch 40-60
- Learning rate reduces from 0.001 to ~0.00003 over training
- MixUp helps prevent validation accuracy from plateauing too early
