# EEGEncoder - BCI IV-2a Training Results

## Training Results Summary (With MixUp Augmentation)

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

| Metric | Target | Actual |
|--------|--------|--------|
| Subject A01 | >75% | 68.42% |
| Average | >80% | 71.54% |
| Paper Benchmark | 86.46% | 71.54% |

## Improvements Made

1. **Fixed Preprocessing**: Added proper standard scaling
2. **Proper EEGEncoder**: Implemented official architecture from GitHub (166K params)
   - ConvBlock (EEGNet-like front-end)
   - 5 parallel DSTS branches (TCN + Transformer)
   - Dropout 0.3 + label smoothing
3. **Data Augmentation**: time-shift, channel dropout, noise, scaling
4. **MixUp Augmentation**: Added MixUp for better generalization
5. **Cosine Annealing**: Added optional learning rate scheduling

## What Still Needs Work

1. **ZUNA Integration**: Not fully integrated - using basic preprocessing
2. **Domain Adversarial Training (DAT)**: Would help cross-subject generalization
3. **Per-subject optimization**: Some subjects remain difficult (A04, A06)

## Per-Subject Analysis

- **Best**: A08 (93%), A03 (86%), A05 (86%), A09 (75%)
- **Challenging**: A04 (40%), A02 (54%), A06 (58%)

## Checkpoints

All model checkpoints saved to `checkpoints/`:
- `best_A0X.pt` - Best validation model per subject

## Training Configuration

- Model: EEGEncoder (5 DSTS branches, 16 hidden channels)
- Parameters: 166,388
- Epochs: 300 (with early stopping)
- Batch Size: 32
- Learning Rate: 0.001 with ReduceLROnPlateau
- Label Smoothing: 0.1
- Augmentation: 2x (time-shift, channel dropout, noise, scaling) + MixUp
- MixUp: alpha=0.4, prob=0.5
- Device: CUDA
