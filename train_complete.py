"""
Complete Training Script for EEGEncoder with Proper Preprocessing and Augmentation

This script implements the full pipeline for BCI IV-2a motor imagery classification:
1. Enhanced preprocessing (bandpass 8-32Hz)
2. Data augmentation
3. EEGEncoder model training
4. Early stopping and checkpointing
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import argparse
import numpy as np
import torch

from src.preprocessing.motor_imagery_pipeline import MotorImageryPreprocessor
from src.augmentation.augmentations import EEGAugmentor, create_augmented_dataloader
from src.models.eegencoder import create_eegencoder
from src.training.trainer import train_subject


def main():
    parser = argparse.ArgumentParser(
        description="Train EEGEncoder on BCI IV-2a with preprocessing"
    )
    parser.add_argument("--subject", type=str, default="A01", help="Subject ID")
    parser.add_argument(
        "--data-dir", type=str, default="src/data/BCICIV_2a_gdf", help="Data directory"
    )
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--augmentation-ratio",
        type=int,
        default=2,
        help="Augmented samples per original",
    )
    parser.add_argument(
        "--filter-bands",
        action="store_true",
        default=True,
        help="Apply 8-32Hz bandpass",
    )
    parser.add_argument(
        "--n-branches", type=int, default=5, help="Number of DSTS branches"
    )
    parser.add_argument(
        "--hidden-channels", type=int, default=16, help="Hidden channels"
    )
    parser.add_argument(
        "--use-zuna",
        action="store_true",
        default=False,
        help="Use ZUNA denoising (not used - unsuitable for epoched MI data)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"Training EEGEncoder on Subject: {args.subject}")
    print(f"Preprocessing: Bandpass = {args.filter_bands}")
    print(f"Augmentation ratio: {args.augmentation_ratio}")
    print("=" * 60)

    # Step 1: Load and preprocess data
    print("\n1. Loading and preprocessing BCI IV-2a dataset...")
    preprocessor = MotorImageryPreprocessor(
        data_dir=args.data_dir,
        filter_alpha_beta=args.filter_bands,
    )

    X, y, metadata = preprocessor.load_and_preprocess(args.subject)
    print(f"   Data loaded: X={X.shape}, y={y.shape}")
    print(f"   Classes: {np.unique(y)}")
    print(f"   Metadata: {metadata}")

    # Step 2: Create model
    print("\n2. Creating EEGEncoder model...")
    model = create_eegencoder(
        n_channels=X.shape[1],
        n_times=X.shape[2],
        n_classes=4,
        n_branches=args.n_branches,
        hidden_channels=args.hidden_channels,
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {params:,}")

    # Step 3: Train with augmentation
    print("\n3. Training model with augmentation...")
    train_loader, val_loader = create_augmented_dataloader(
        X,
        y,
        batch_size=args.batch_size,
        augmentation_ratio=args.augmentation_ratio,
        val_split=0.2,
    )

    print(f"   Training batches: {len(train_loader)} (with augmentation)")
    print(f"   Validation batches: {len(val_loader)}")

    # Train using the trainer
    from src.training.trainer import Trainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=0.001,
        label_smoothing=0.1,
        weight_decay=0.0001,
        use_mixup=True,
        mixup_alpha=0.4,
        mixup_prob=0.5,
        use_cosine_annealing=True,
    )

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        early_stopping_patience=50,
        checkpoint_dir="checkpoints",
        subject=args.subject,
    )

    print("\n" + "=" * 60)
    print(f"Training complete for Subject {args.subject}!")
    print(f"Best validation accuracy: {max(history['val_acc']):.4f}")
    print("=" * 60)


def train_all_subjects():
    """Train all 9 subjects with increased capacity"""
    subjects = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09"]
    results = {}

    for subject in subjects:
        print(f"\n{'=' * 60}")
        print(f"Training Subject: {subject}")
        print("=" * 60)

        # Load data with bandpass filtering (ZUNA not used - unsuitable for epoched MI data)
        preprocessor = MotorImageryPreprocessor(
            data_dir="src/data/BCICIV_2a_gdf",
            filter_alpha_beta=True,
        )
        X, y, metadata = preprocessor.load_and_preprocess(subject)

        # Create model with original configuration
        model = create_eegencoder(
            n_channels=X.shape[1],
            n_times=X.shape[2],
            n_classes=4,
            n_branches=5,
            hidden_channels=16,
        )

        # Train with augmentation
        train_loader, val_loader = create_augmented_dataloader(
            X, y, batch_size=32, augmentation_ratio=2, val_split=0.2
        )

        # Train with ReduceLROnPlateau
        from src.training.trainer import Trainer

        trainer = Trainer(
            model=model,
            learning_rate=0.001,
            use_mixup=True,
            mixup_alpha=0.4,
            mixup_prob=0.5,
            use_cosine_annealing=False,
        )
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=300,
            checkpoint_dir="checkpoints",
            subject=subject,
        )
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=300,  # More epochs
            checkpoint_dir="checkpoints",
            subject=subject,
        )
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=250,
            checkpoint_dir="checkpoints",
            subject=subject,
        )

        best_acc = max(history["val_acc"])
        results[subject] = best_acc
        print(f"{subject} Best: {best_acc:.4f}")

    # Print summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS (Increased Model Capacity)")
    print("=" * 60)
    for subject, acc in results.items():
        print(f"{subject}: {acc:.4f}")
    print(f"Average: {sum(results.values()) / len(results):.4f}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        train_all_subjects()
    else:
        main()
