# EEGEncoder - BCI IV-2a Training Results

## Training Results Summary

| Subject | Best Val Accuracy |
|---------|------------------|
| A01     | 57.89%          |
| A02     | 33.33%          |
| A03     | 43.86%          |
| A04     | 35.09%          |
| A05     | 33.33%          |
| A06     | 38.60%          |
| A07     | 38.60%          |
| A08     | 49.12%          |
| A09     | 50.88%          |
| **Average** | **42.30%**  |

## Target vs Actual

| Metric | Target | Actual |
|--------|--------|--------|
| Subject A01 | >75% | 57.89% |
| Average | >80% | 42.30% |
| Paper Benchmark | ~86% | 42.30% |

## Analysis

The current results are below target because:

1. **No ZUNA Preprocessing Applied**: The model was trained on raw EEG data without ZUNA denoising
2. **Simplified Model Architecture**: 73K parameters vs original paper's full model
3. **Limited Training Data**: 288 trials per subject (only training set used)

## Next Steps to Improve Accuracy

1. **Apply ZUNA Preprocessing**: Integrate ZUNA EEG foundation model for denoising
2. **Use Evaluation Data**: Load both training and evaluation sets for more data
3. **Hyperparameter Tuning**: Optimize learning rate, batch size, model architecture
4. **Data Augmentation**: Apply time-shift, channel dropout, noise injection

## Checkpoints

All model checkpoints saved to `checkpoints/`:
- `best_A0X.pt` - Best validation model per subject
- `final_A0X.pt` - Final epoch model per subject
- `history_A0X.json` - Training history per subject

## Training Configuration

- Model: EEGEncoder (5 DSTS branches, 16 hidden channels)
- Parameters: 73,460
- Epochs: 200 (with early stopping)
- Batch Size: 32
- Learning Rate: 0.001 with ReduceLROnPlateau
- Label Smoothing: 0.1
- Device: CUDA (NVIDIA RTX 5080)
