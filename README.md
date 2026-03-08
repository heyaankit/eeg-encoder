# EEGEncoder - Motor Imagery Classification

EEG Motor Imagery classification system using EEGEncoder architecture with ZUNA preprocessing for BCI Competition IV-2a dataset.

## Project Structure

```
EEGEncoder/
├── src/
│   ├── data/
│   │   └── bcic_iv_2a.py       # BCI IV-2a dataset loader
│   ├── models/
│   │   └── eegencoder.py       # EEGEncoder model architecture
│   ├── preprocessing/
│   │   └── zuna_pipeline.py    # ZUNA preprocessing pipeline
│   └── training/
│       └── trainer.py          # Training loop with early stopping
├── checkpoints/                 # Saved model checkpoints
├── configs/
│   └── default.yaml            # Configuration file
├── train.py                     # Main training script
├── requirements.txt             # Python dependencies
└── RESULTS.md                   # Training results
```

## Setup

```bash
cd Chimera/EEGEncoder
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

### Train Single Subject
```bash
python train.py --subject A01 --epochs 200
```

### Train All Subjects
```bash
python -c "
from src.data.bcic_iv_2a import load_single_subject
from src.models.eegencoder import create_eegencoder
from src.training.trainer import train_subject

for subject in ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09']:
    X, y = load_single_subject('src/data/BCICIV_2a_gdf', subject)
    model = create_eegencoder(n_channels=X.shape[1], n_times=X.shape[2], n_classes=4)
    train_subject(model, X, y, subject, epochs=200)
"
```

## Model Architecture

- **EEGEncoder**: 5 parallel DSTS (Depthwise Temporal-Spatial) branches
- **TCN**: Temporal Convolutional Network in each branch
- **Parameters**: ~73K
- **Input**: EEG trials (25 channels × 1126 time points)

## Results

See [RESULTS.md](RESULTS.md) for training results on BCI IV-2a dataset.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- MNE-Python
- NumPy, SciPy, Scikit-learn
- NVIDIA GPU with CUDA (recommended)

## License

For research purposes only.
