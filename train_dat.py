"""
Domain Adversarial Training (DAT) for EEGEncoder

This script trains EEGEncoder with Domain Adversarial Training to learn
subject-invariant features, improving generalization across subjects.

How DAT works:
1. Train on ALL subjects together (not per-subject)
2. Each subject is treated as a "domain"
3. Feature extractor learns features that CANNOT predict the subject
4. This forces learning of SUBJECT-INVARIANT features
5. Result: Better generalization to new subjects

Reference: CDAN (Conditional Domain Adversarial Network)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.preprocessing.motor_imagery_pipeline import MotorImageryPreprocessor
from src.augmentation.augmentations import EEGAugmentor
from src.models.eegencoder import create_eegencoder
from src.models.domain_adversarial import DomainAdversarialEEGEncoder, DATTrainer


def load_all_subjects(data_dir, subjects):
    """Load data from all subjects for DAT training."""
    print("Loading all subjects for DAT training...")

    preprocessor = MotorImageryPreprocessor(
        data_dir=data_dir, filter_alpha_beta=False, use_zuna=False
    )

    all_X = []
    all_y = []
    all_domain = []

    for idx, subject in enumerate(subjects):
        print(f"  Loading {subject}...")
        X, y, metadata = preprocessor.load_and_preprocess(subject)

        all_X.append(X)
        all_y.append(y)
        all_domain.append(np.full(len(y), idx))  # Subject ID as domain

        print(f"    {subject}: {X.shape} trials")

    # Concatenate all subjects
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    domain = np.concatenate(all_domain, axis=0)

    print(f"\nTotal: {X.shape[0]} trials from {len(subjects)} subjects")

    return X, y, domain


def create_dataloader_with_domain(
    X, y, domain, batch_size=32, augmentation_ratio=2, val_split=0.2
):
    """Create dataloader with domain labels."""
    # Apply augmentation
    augmentor = EEGAugmentor()

    augmented_X = [X]
    augmented_y = [y]
    augmented_domain = [domain]

    for _ in range(augmentation_ratio):
        for i in range(len(X)):
            aug_x, _ = augmentor.augment(X[i : i + 1])
            augmented_X.append(aug_x)
            augmented_y.append(y[i : i + 1])
            augmented_domain.append(domain[i : i + 1])

    X_aug = np.concatenate(augmented_X, axis=0)
    y_aug = np.concatenate(augmented_y, axis=0)
    domain_aug = np.concatenate(augmented_domain, axis=0)

    # Shuffle
    indices = np.random.permutation(len(X_aug))
    X_aug = X_aug[indices]
    y_aug = y_aug[indices]
    domain_aug = domain_aug[indices]

    # Split
    n_train = int(len(X_aug) * (1 - val_split))

    # EEGEncoder expects (batch, 1, channels, times)
    X_train = torch.FloatTensor(X_aug[:n_train]).unsqueeze(1)
    y_train = torch.LongTensor(y_aug[:n_train])
    domain_train = torch.LongTensor(domain_aug[:n_train])

    X_val = torch.FloatTensor(X_aug[n_train:]).unsqueeze(1)
    y_val = torch.LongTensor(y_aug[n_train:])
    domain_val = torch.LongTensor(domain_aug[n_train:])

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, domain_train


def train_with_dat(data_dir, subjects, epochs=150, batch_size=32):
    """Train with Domain Adversarial Training."""

    # Load all subjects
    X, y, domain = load_all_subjects(data_dir, subjects)

    # Create dataloaders
    train_loader, val_loader, domain_train = create_dataloader_with_domain(
        X, y, domain, batch_size=batch_size, augmentation_ratio=2
    )

    # Create DAT model
    print("\nCreating Domain Adversarial EEGEncoder...")
    eeg_encoder = create_eegencoder(
        n_channels=X.shape[1],
        n_times=X.shape[2],
        n_classes=4,
        n_branches=5,
        hidden_channels=16,
    )

    model = DomainAdversarialEEGEncoder(
        eeg_encoder,
        n_classes=4,
        n_domains=len(subjects),
        hidden_dim=128,
        domain_lambda=0.5,
    )

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train with DAT
    print("\nTraining with Domain Adversarial Training...")
    trainer = DATTrainer(
        model=model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        learning_rate=0.001,
        label_smoothing=0.1,
        weight_decay=0.0001,
        domain_lambda=0.5,
        use_mixup=True,
        mixup_alpha=0.4,
        mixup_prob=0.5,
    )

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        domain_labels=domain_train,
        epochs=epochs,
        checkpoint_dir="checkpoints/dat",
    )

    print(f"\nBest validation accuracy: {max(history['val_acc']):.4f}")

    return history, model


def evaluate_per_subject(model, data_dir, subjects):
    """Evaluate model on each subject individually."""
    model.eval()

    # Get device from model
    device = next(model.parameters()).device

    results = {}

    preprocessor = MotorImageryPreprocessor(
        data_dir=data_dir, filter_alpha_beta=False, use_zuna=False
    )

    for subject in subjects:
        X, y, _ = preprocessor.load_and_preprocess(subject)

        X_tensor = torch.FloatTensor(X).unsqueeze(1).to(device)
        y_tensor = torch.LongTensor(y).to(device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32)

        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in loader:
                task_logits, _ = model(batch_x)
                _, predicted = task_logits.max(1)
                correct += predicted.eq(batch_y).sum().item()
                total += batch_y.size(0)

        acc = correct / total
        results[subject] = acc
        print(f"  {subject}: {acc:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="DAT Training for EEGEncoder")
    parser.add_argument("--data-dir", type=str, default="src/data/BCICIV_2a_gdf")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--domain-lambda", type=float, default=0.5)
    args = parser.parse_args()

    subjects = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09"]

    print("=" * 60)
    print("Domain Adversarial Training for EEGEncoder")
    print("=" * 60)

    # Train with DAT
    history, model = train_with_dat(
        data_dir=args.data_dir,
        subjects=subjects,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Evaluate per subject
    print("\n" + "=" * 60)
    print("Per-Subject Evaluation")
    print("=" * 60)

    results = evaluate_per_subject(model, args.data_dir, subjects)

    avg_acc = np.mean(list(results.values()))
    print(f"\nAverage accuracy: {avg_acc:.4f}")

    return results


if __name__ == "__main__":
    main()
