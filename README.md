# EEGEncoder - Motor Imagery Classification

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/heyaankit/eeg-encoder/blob/main/LICENSE)
[![Paper](https://img.shields.io/badge/paper-Scientific%20Reports-2025-blue.svg)](https://www.nature.com/articles/s41598-025-06364-4)
[![BCI Competition](https://img.shields.io/badge/BCI-IV--2a-dataset-4--classes-green.svg)](http://bnci-horizon-2020.eu/)

Deep learning system for EEG-based Motor Imagery (MI) classification using the EEGEncoder architecture. Built for the BCI Competition IV-2a dataset to enable brain-computer interface applications for prosthetics.

---

## Performance Highlights

| Metric | Score |
|--------|-------|
| Validation Accuracy | **78.86%** |
| Per-Subject Average | **90.82%** |
| Best Subject (A08) | 97.92% |
| Parameters | ~172K |

---

## Installation

```bash
# Clone repository
git clone https://github.com/heyaankit/eeg-encoder.git
cd eeg-encoder

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Dataset

This project uses the **BCI Competition IV-2a dataset**. Download from:
- Official: [BNCI Horizon 2020](http://bnci-horizon-2020.eu/database/data-sets)
- Place in: `src/data/BCICIV_2a_gdf/`

Expected format: GDF files (A01.gdf to A09.gdf)

---

## Usage

### Training Options

**Option 1: Per-Subject Training**
```bash
# Train single subject
python train_complete.py --subject A01

# Train all subjects
python train_complete.py --all
```

**Option 2: Domain Adversarial Training (Multi-Subject)**
```bash
# Train on all subjects together
python train_dat.py --epochs 80
```

### Configuration Options

| Flag | Default | Description |
|------|---------|-------------|
| `--subject` | A01 | Subject ID (A01-A09) |
| `--epochs` | 300 | Number of training epochs |
| `--batch-size` | 32 | Batch size |
| `--hidden-channels` | 16 | Model hidden channels |
| `--augmentation-ratio` | 2 | Data augmentation multiplier |

---

## Project Structure

```
EEGEncoder/
├── src/
│   ├── data/
│   │   └── bcic_iv_2a.py              # BCI IV-2a dataset loader
│   ├── models/
│   │   ├── eegencoder.py              # EEGEncoder architecture
│   │   └── domain_adversarial.py      # Domain Adversarial Training
│   ├── preprocessing/
│   │   └── motor_imagery_pipeline.py  # EEG preprocessing
│   ├── augmentation/
│   │   └── augmentations.py           # Data augmentation
│   └── training/
│       └── trainer.py                 # Training utilities
├── checkpoints/                        # Model checkpoints
│   └── dat/                          # DAT model
├── results/                           # Experiment results
├── train_complete.py                   # Per-subject training
├── train_dat.py                        # Multi-subject training
└── requirements.txt                    # Dependencies
```

---

## Model Architecture

### Why EEGEncoder?

EEGEncoder is a state-of-the-art deep learning framework introduced in *"Advancing BCI with Transformer-Based Motor Imagery Classification"* (Scientific Reports 2025). We chose EEGEncoder because:

1. **Novel Architecture**: Combines Temporal Convolutional Networks (TCN) with Transformer encoders in a Dual-Stream Temporal-Spatial (DSTS) block - capturing both temporal dynamics and spatial patterns in EEG signals.

2. **Parallel Multi-Branch Design**: Uses 5 parallel DSTS branches to enhance feature extraction diversity, improving classification robustness.

3. **Proven Performance**: Achieved 86.46% accuracy on BCI Competition IV-2a (subject-dependent) and 74.48% (subject-independent) in the original paper.

4. **End-to-End Learning**: Eliminates manual feature engineering - learns representations directly from raw EEG.

5. **Efficient**: ~166K parameters makes it suitable for real-time BCI applications.

### Architecture Details

| Component | Description |
|-----------|-------------|
| ConvBlock | EEGNet-style front-end for initial feature extraction |
| DSTS Branches | 5 parallel blocks, each with TCN + Transformer |
| Classifier | Fully connected head for 4-class MI classification |
| Parameters | ~172K (with domain discriminator) |

### Input Format

- Channels: 22-25 EEG channels
- Time points: 1126 samples @ 250Hz (~4.5 seconds)
- Classes: Left hand, Right hand, Feet, Tongue

---

## Preprocessing Pipeline

Our preprocessing follows established BCI research practices:

1. **High-pass filter** (1Hz): Removes DC drift and slow artifacts
2. **Notch filter** (50Hz): Removes power line interference
3. **Bandpass filter** (8-32Hz): Extracts motor imagery alpha/beta bands (most discriminative for MI)
4. **Epoch extraction**: -0.5s to +4s relative to cue onset
5. **Baseline correction**: Using pre-cue period (-0.5s to 0s)
6. **Channel selection**: 22 EEG channels (excluding EOG)
7. **Standardization**: Zero mean, unit variance per subject

---

## Data Augmentation

We employ multiple augmentation strategies to improve generalization:

| Technique | Parameters | Purpose |
|-----------|------------|---------|
| Time-shift | ±40 samples | Temporal variability |
| Channel dropout | 30% channels | Robustness to electrode issues |
| Gaussian noise | σ=0.2 | Noise resilience |
| Scaling | 0.8-1.2x | Amplitude invariance |
| MixUp | α=0.4 | Regularization via interpolation |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=10) |
| Label smoothing | 0.1 |
| Weight decay | 0.0001 |
| Early stopping | patience=30 |
| Batch size | 32 |

---

## Results

### Experiment Summary

| Configuration | Accuracy | Notes |
|--------------|----------|-------|
| Baseline (fixed preprocessing) | 71.35% | After fixing bandpass filter bug |
| + MixUp augmentation | 72.71% | Best per-subject |
| + Domain Adversarial Training | **78.86%** | Multi-subject, **target achieved** |

### Per-Subject Performance (DAT)

| Subject | Accuracy |
|---------|----------|
| A01 | 91.32% |
| A02 | 83.33% |
| A03 | 91.32% |
| A04 | 86.11% |
| A05 | 90.28% |
| A06 | 87.50% |
| A07 | 94.79% |
| A08 | 97.92% |
| A09 | 94.79% |
| **Average** | **90.82%** |

See [results/RESULTS_DAT.md](results/RESULTS_DAT.md) for detailed experiment documentation.

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- MNE-Python
- NumPy, SciPy, Scikit-learn
- NVIDIA GPU with CUDA (recommended)

---

## License

For research purposes only.
