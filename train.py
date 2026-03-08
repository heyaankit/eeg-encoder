"""
Main Training Script for EEGEncoder

Usage:
    python train.py                    # Train Subject A01
    python train.py --subject A02    # Train specific subject
    python train.py --all            # Train all subjects
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import numpy as np
from src.data.bcic_iv_2a import load_single_subject
from src.models.eegencoder import create_eegencoder
from src.training.trainer import train_subject


def main():
    parser = argparse.ArgumentParser(description='Train EEGEncoder on BCI IV-2a')
    parser.add_argument('--subject', type=str, default='A01', help='Subject ID')
    parser.add_argument('--data-dir', type=str, default='src/data/BCICIV_2a_gdf', help='Data directory')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    args = parser.parse_args()
    
    print("="*60)
    print(f"Training EEGEncoder on Subject: {args.subject}")
    print("="*60)
    
    # Load data
    print("\n1. Loading BCI IV-2a dataset...")
    try:
        X, y = load_single_subject(args.data_dir, args.subject)
        print(f"   Data loaded: X={X.shape}, y={y.shape}")
        print(f"   Classes: {np.unique(y)}")
    except Exception as e:
        print(f"   Error loading data: {e}")
        return
    
    # Create model
    print("\n2. Creating EEGEncoder model...")
    model = create_eegencoder(
        n_channels=X.shape[1],
        n_times=X.shape[2],
        n_classes=4,
        n_branches=5,
        hidden_channels=16
    )
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\n3. Training model...")
    history = train_subject(
        model=model,
        X=X,
        y=y,
        subject=args.subject,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    print("\n" + "="*60)
    print(f"Training complete for Subject {args.subject}!")
    print(f"Best validation accuracy: {max(history['val_acc']):.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
