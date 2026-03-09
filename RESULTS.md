# EEGEncoder-2a Training - BCI IV Results

## Training Results Summary (With MixUp Augmentation)

| Subject | Val Accuracy |
|---------|-------------|
| A01     | 64.91%      |
| A02     | 61.40%      |
| A03     | 80.70%      |
| A04     | 47.37%      |
| A05     | 89.47%      |
| A06     | 54.39%      |
| A07     | 92.98%      |
| A08     | 94.74%      |
| A09     | 68.42%      |
| **Average** | **72.71%**  |

## Target vs Actual

| Metric | Target | Actual |
|--------|--------|--------|
| Subject A01 | >75% | 64.91% |
| Average | >80% | 72.71% |
| Paper Benchmark | 86.46% | 72.71% |

## Improvements Made

1. **Fixed Preprocessing**: Added proper standard scaling
2. **Proper EEGEncoder**: Implemented official architecture from GitHub (166K params)
   - ConvBlock (EEGNet-like front-end)
   - 5 parallel DSTS branches (TCN + Transformer)
   - Dropout 0.3 + label smoothing
3. **Data Augmentation**: time-shift, channel dropout, noise, scaling
4. **MixUp Augmentation**: Added MixUp for better generalization

## What Still Needs Work

1. **ZUNA Integration**: Not fully integrated - using basic preprocessing
2. **Domain Adversarial Training (DAT)**: Would help cross-subject generalization
3. **Hyperparameter Tuning**: Learning rate, model capacity adjustments

## Per-Subject Analysis

- **Best**: A08 (95%), A07 (93%), A05 (89%), A03 (81%)
- **Challenging**: A04 (47%), A06 (54%), A02 (61%)

## Checkpoints

All model checkpoints saved to `checkpoints/`:
- `best_A0X.pt` - Best validation model per subject

## Training Configuration

- Model: EEGEncoder (5 DSTS branches, 16 hidden channels)
- Parameters: 166,388
- Epochs: 200 (with early stopping)
- Batch Size: 32
- Learning Rate: 0.001 with ReduceLROnPlateau
- Label Smoothing: 0.1
- Augmentation: 2x (time-shift, channel dropout, noise, scaling) + MixUp
- MixUp: alpha=0.4, prob=0.5
- Device: CUDA
